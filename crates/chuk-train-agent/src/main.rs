//! chuk-train-agent — a generic compute worker (chuk-compute-spec §7).
//!
//! Runs identically on Colab, a Vast container, or any Linux/macOS box:
//!
//!     chuk-train-agent --url wss://cp.example.com/ws/agent --token <JOIN_TOKEN>
//!
//! Dials OUT to the control plane, advertises its capabilities, heartbeats,
//! executes assigned jobs (stage inputs, run one command, stream logs/metrics,
//! collect outputs), and reconnects with exponential backoff. The worker knows
//! nothing about what a job computes — every workload specific arrives inside
//! the [`chuk_compute_wire::Job`].
//!
//! One job in flight; streamed data is dropped while the control plane is
//! unreachable (a disk spool + replay is a later milestone). If the session
//! drops mid-job the child is killed and the control plane requeues.

mod capabilities;
mod executor;
mod httpclient;
mod inputs;
mod metrics;
mod outputs;
mod procio;
mod sandbox;
mod seq;

use std::time::Duration;

use anyhow::{Context, Result};
use chuk_compute_wire::{CpToWorker, KillReason, Resume, WorkerId, WorkerToCp, PROTOCOL_VERSION};
use chuk_train_proto::{
    DEFAULT_DRAIN_WINDOW_MIN, HEARTBEAT_INTERVAL, RECONNECT_BACKOFF_MAX, RECONNECT_BACKOFF_MIN,
};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

use crate::executor::RunningJob;
use crate::seq::Seq;

/// Separator between labels passed on the `--labels` flag.
const LABEL_SEPARATOR: char = ',';
/// Seconds in a minute — lease budgets arrive in minutes.
const SECONDS_PER_MINUTE: f64 = 60.0;

#[derive(Parser, Debug)]
#[command(name = "chuk-train-agent", about)]
struct Args {
    /// Control plane websocket URL, e.g. wss://cp.example.com/ws/agent
    #[arg(long)]
    url: String,
    /// Join token
    #[arg(long)]
    token: String,
    /// Stable worker id to present on reconnect; the control plane mints one at
    /// first handshake if omitted.
    #[arg(long)]
    worker_id: Option<String>,
    /// Comma-separated labels, e.g. site=colab,t4
    #[arg(long, default_value = "")]
    labels: String,
    /// Lease budget in minutes. The worker self-drains at T-drain even with no
    /// connectivity (belt); the control plane destroys at T-0 (braces).
    #[arg(long)]
    lease_min: Option<f64>,
    /// Drain window in minutes (how far before T-0 to self-drain); the control
    /// plane passes its own value so belt and braces agree.
    #[arg(long)]
    drain_window_min: Option<f64>,
}

/// This build's compile target as `<arch>-<os>` (e.g. `aarch64-macos`).
fn target_triple() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let labels: Vec<String> = args
        .labels
        .split(LABEL_SEPARATOR)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    // A leased/spot host may be reclaimed under the worker at any time.
    let preemptible = args.lease_min.is_some();

    // The self-drain deadline (belt): computed once from process start so it
    // survives reconnects. The control plane's Drain at T-drain and destroy at
    // T-0 are the authoritative braces.
    let drain_window_min = args.drain_window_min.unwrap_or(DEFAULT_DRAIN_WINDOW_MIN);
    let self_drain_at = args.lease_min.map(|minutes| {
        let secs = (minutes * SECONDS_PER_MINUTE - drain_window_min * SECONDS_PER_MINUTE).max(0.0);
        tokio::time::Instant::now() + Duration::from_secs_f64(secs)
    });

    let mut worker_id = args.worker_id.map(WorkerId::from);
    let mut backoff = RECONNECT_BACKOFF_MIN;
    loop {
        match session(
            &args.url,
            &args.token,
            worker_id.clone(),
            &labels,
            preemptible,
            self_drain_at,
        )
        .await
        {
            Ok(assigned_id) => {
                // Keep the id the control plane confirmed so reconnects present
                // the same worker.
                worker_id = Some(assigned_id);
                backoff = RECONNECT_BACKOFF_MIN;
            }
            Err(error) => {
                warn!(%error, "session error");
            }
        }
        info!(delay_s = backoff.as_secs(), "reconnecting after backoff");
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
    }
}

/// One websocket session: handshake, then message loop until the socket dies.
/// Returns the worker id the control plane confirmed.
async fn session(
    url: &str,
    token: &str,
    worker_id: Option<WorkerId>,
    labels: &[String],
    preemptible: bool,
    self_drain_at: Option<tokio::time::Instant>,
) -> Result<WorkerId> {
    // REST origin for input fetch and output upload, derived from the same host
    // we dial the websocket on.
    let origin = httpclient::origin_from_ws_url(url)?;
    let (socket, _response) = connect_async(url).await.context("connecting")?;
    let (mut sink, mut stream) = socket.split();

    // On reconnect we present the retained id (with nothing to replay: M1 has no
    // spool, and the prior child was killed on disconnect); first join sends no
    // resume at all.
    let resume = worker_id.map(|id| Resume {
        worker_id: id,
        running_jobs: Vec::new(),
        high_water: 0,
    });
    let hello = WorkerToCp::Hello {
        protocol_version: PROTOCOL_VERSION,
        worker_semver: env!("CARGO_PKG_VERSION").into(),
        target_triple: target_triple(),
        token: token.to_owned(),
        capabilities: capabilities::detect(labels, preemptible).await,
        resume,
    };
    sink.send(to_frame(&hello)?).await.context("sending hello")?;

    let worker_id = match next_message(&mut stream).await {
        Some(CpToWorker::HelloAck {
            worker_id,
            class,
            telemetry,
            wall_deadline,
        }) => {
            // Telemetry sampling and the wall clock are accepted here; wiring
            // the system-telemetry sampler is a later milestone.
            info!(worker = %worker_id, ?class, ?telemetry, ?wall_deadline, "hello acknowledged");
            worker_id
        }
        Some(CpToWorker::HelloReject {
            reason,
            min_protocol,
            ..
        }) => anyhow::bail!("hello rejected: {reason} (min_protocol={min_protocol})"),
        other => anyhow::bail!("unexpected handshake response: {other:?}"),
    };

    // One sequence counter per session; every streamed WorkerToCp draws from it.
    let seq = Seq::new();
    // Job tasks talk back through this channel; the select loop owns the sink.
    let (tx, mut outbound) = mpsc::unbounded_channel::<WorkerToCp>();
    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Current job slot; aborting the handle kills the child (kill_on_drop).
    let mut current: Option<RunningJob> = None;
    // Once draining, take no new work and report Drained.
    let mut draining = false;

    let session_result = loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                if sink.send(to_frame(&WorkerToCp::Heartbeat)?).await.is_err() {
                    break Ok(());
                }
            }
            // Self-drain belt: fire once at T-drain even if the control plane is
            // dark. `pending()` when there is no lease so the branch never fires.
            _ = async {
                match self_drain_at {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            }, if !draining => {
                warn!("lease T-drain reached; self-draining");
                draining = true;
                let _ = tx.send(WorkerToCp::Drained);
            }
            msg = outbound.recv() => {
                let msg = msg.expect("tx lives in this scope");
                if sink.send(to_frame(&msg)?).await.is_err() {
                    break Ok(());
                }
            }
            inbound = next_message(&mut stream) => match inbound {
                Some(CpToWorker::AssignJob { job }) => {
                    if draining {
                        warn!(job = %job.id, "assign received while draining; ignoring");
                        continue;
                    }
                    if current.as_ref().is_some_and(|j| !j.is_finished()) {
                        // The control plane never double-assigns; if it does,
                        // refuse loudly rather than corrupt the slot.
                        warn!(job = %job.id, "assign received while busy; ignoring");
                        continue;
                    }
                    info!(job = %job.id, "assigned");
                    current = Some(executor::spawn(job, tx.clone(), seq.clone(), origin.clone()));
                }
                Some(CpToWorker::Cancel { job_id }) => {
                    if let Some(job) = current.take() {
                        info!(job = %job_id, "cancelling");
                        let killed = job.job_id().clone();
                        job.abort();
                        let msg = WorkerToCp::JobKilled {
                            seq: seq.next(),
                            job_id: killed,
                            reason: KillReason::Cancel,
                        };
                        if sink.send(to_frame(&msg)?).await.is_err() {
                            break Ok(());
                        }
                    }
                }
                Some(CpToWorker::Drain { deadline }) => {
                    info!(deadline, "drain requested by control plane");
                    draining = true;
                    let _ = tx.send(WorkerToCp::Drained);
                }
                Some(other) => {
                    // HelloAck/HelloReject mid-session, or a variant a newer
                    // control plane introduced: tolerate and ignore.
                    warn!(?other, "unexpected control message mid-session; ignoring");
                }
                None => break Ok(()),
            },
        }
    };

    // Session over: kill any in-flight job so our state matches the control
    // plane's (it requeues the job on disconnect).
    if let Some(job) = current.take() {
        if !job.is_finished() {
            warn!("session dropped mid-job; killing child (control plane requeues)");
        }
        job.abort();
    }
    session_result.map(|_| worker_id)
}

async fn next_message(
    stream: &mut (impl StreamExt<Item = tokio_tungstenite::tungstenite::Result<Message>> + Unpin),
) -> Option<CpToWorker> {
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Text(text)) => match serde_json::from_str::<CpToWorker>(text.as_str()) {
                Ok(msg) => return Some(msg),
                Err(error) => warn!(%error, "unparseable control message; skipping"),
            },
            Ok(Message::Close(_)) | Err(_) => return None,
            Ok(_) => {}
        }
    }
    None
}

fn to_frame(msg: &WorkerToCp) -> Result<Message> {
    Ok(Message::Text(serde_json::to_string(msg)?.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_triple_is_arch_dash_os() {
        let triple = target_triple();
        assert_eq!(
            triple,
            format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
        );
        assert!(triple.contains('-'));
    }

    #[test]
    fn a_worker_message_serialises_to_a_text_frame() {
        let frame = to_frame(&WorkerToCp::Heartbeat).unwrap();
        let Message::Text(text) = frame else {
            panic!("expected a text frame");
        };
        assert_eq!(text.as_str(), r#"{"type":"heartbeat"}"#);
    }
}
