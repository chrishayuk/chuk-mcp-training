//! chuk-compute-worker — a generic compute worker (chuk-compute-spec §7).
//!
//! Runs identically on Colab, a Vast container, or any Linux/macOS box:
//!
//!     chuk-compute-worker --url wss://cp.example.com/ws/agent --token <JOIN_TOKEN>
//!
//! Dials OUT to the control plane, advertises its capabilities, heartbeats,
//! executes assigned jobs (stage inputs, run one command, stream logs/metrics,
//! collect outputs), and reconnects with jittered exponential backoff. The worker
//! knows nothing about what a job computes — every workload specific arrives
//! inside the [`chuk_compute_wire::Job`].
//!
//! One job in flight. Durable state (the sequence counter, the executor's event
//! sink, the running job, and the replay outbox) lives for the worker's whole
//! lifetime in [`main`], so a dropped websocket does not have to end a job: a
//! **persistent** worker keeps the child running while disconnected and, on
//! reconnect, drains and replays the events the control plane has not yet seen
//! (chuk-compute M3.2). A **leased** worker still abandons its job on disconnect
//! and lets the control plane requeue it.

mod backoff;
mod capabilities;
mod constants;
mod executor;
mod httpclient;
mod inputs;
mod metrics;
mod outbox;
mod outputs;
mod procio;
mod sandbox;
mod selfupdate;
mod seq;
mod telemetry;

use std::time::Duration;

use anyhow::{Context, Result};
use chuk_compute_wire::{
    CpToWorker, KillReason, Resume, WorkerClass, WorkerId, WorkerToCp, PROTOCOL_VERSION,
};
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use crate::backoff::with_jitter;
use crate::constants::{
    DEFAULT_DRAIN_WINDOW_MIN, EXIT_CODE_REJECTED, HEARTBEAT_INTERVAL, RECONNECT_BACKOFF_MAX,
    RECONNECT_BACKOFF_MIN, SYS_SAMPLE_INTERVAL,
};
use crate::executor::RunningJob;
use crate::outbox::{event_seq, trim_to_high_water, SEQ_ORIGIN};
use crate::seq::Seq;

/// Separator between labels passed on the `--labels` flag.
const LABEL_SEPARATOR: char = ',';
/// Seconds in a minute — lease budgets arrive in minutes.
const SECONDS_PER_MINUTE: f64 = 60.0;

#[derive(Parser, Debug)]
#[command(name = "chuk-compute-worker", about)]
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

/// The distribution target triple for this host — the one the control plane
/// serves a worker under (`/agent/<triple>`), so it must match the CP's
/// `SUPPORTED_TARGETS` and `install.sh`'s uname mapping. Falls back to a bare
/// `<arch>-<os>` (which the CP won't have a binary for) on an unknown platform.
fn target_triple() -> String {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => "x86_64-unknown-linux-musl",
        ("linux", "aarch64") => "aarch64-unknown-linux-musl",
        ("macos", "aarch64") => "aarch64-apple-darwin",
        ("macos", "x86_64") => "x86_64-apple-darwin",
        (os, arch) => return format!("{arch}-{os}"),
    }
    .to_owned()
}

/// Re-exec `exe` with our original arguments, replacing this process (used after
/// a self-update, spec §3). Only returns — as an error — if the exec syscall
/// fails; on success the running image is replaced and this never returns.
#[cfg(unix)]
fn reexec(exe: &std::path::Path) -> anyhow::Error {
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .exec();
    anyhow::anyhow!("re-exec of {} failed: {err}", exe.display())
}

#[cfg(not(unix))]
fn reexec(exe: &std::path::Path) -> anyhow::Error {
    anyhow::anyhow!("self-update re-exec is unix-only (target {})", exe.display())
}

/// Cheap entropy for reconnect jitter: the sub-second slice of the wall clock.
/// Enough to de-correlate a fleet without pulling in an RNG dependency.
fn jitter_entropy() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| u64::from(since.subsec_nanos()))
        .unwrap_or(0)
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

    // The self-drain deadline (belt): computed once from process start so it
    // survives reconnects. The control plane's Drain at T-drain and destroy at
    // T-0 are the authoritative braces.
    let drain_window_min = args.drain_window_min.unwrap_or(DEFAULT_DRAIN_WINDOW_MIN);
    let self_drain_at = args.lease_min.map(|minutes| {
        let secs = (minutes * SECONDS_PER_MINUTE - drain_window_min * SECONDS_PER_MINUTE).max(0.0);
        tokio::time::Instant::now() + Duration::from_secs_f64(secs)
    });

    // Durable worker-lifetime state, held across every session so a persistent
    // worker's job survives a dropped socket (chuk-compute M3.2):
    //   - `seq`: one monotonic counter, created once, cloned into every job, so
    //     replayed events keep unique, monotonic sequence numbers across
    //     reconnects (the control plane dedups by seq);
    //   - `job_tx`/`job_rx`: the executor's event sink, which keeps filling while
    //     disconnected — the next session drains and replays it;
    //   - `current`: the running job (a leased worker aborts it on disconnect, a
    //     persistent one leaves it running);
    //   - `outbox`: every streamed event produced for the current job, cleared
    //     when a new job is assigned, so replay memory is bounded;
    //   - `class`/`worker_id`: learned from `HelloAck` and retained across
    //     reconnects.
    let seq = Seq::new();
    let (job_tx, mut job_rx) = mpsc::unbounded_channel::<WorkerToCp>();
    let mut current: Option<RunningJob> = None;
    let mut outbox: Vec<(u64, WorkerToCp)> = Vec::new();
    let mut class = WorkerClass::Leased;
    let mut worker_id = args.worker_id.as_deref().map(WorkerId::from);

    let mut backoff = RECONNECT_BACKOFF_MIN;
    loop {
        match run_session(
            &args,
            &mut worker_id,
            &mut class,
            &mut current,
            &mut outbox,
            &seq,
            &job_tx,
            &mut job_rx,
            self_drain_at,
        )
        .await
        {
            Ok(()) => backoff = RECONNECT_BACKOFF_MIN,
            Err(error) => warn!(%error, "session error"),
        }

        // Class-gated disconnect handling. A leased worker abandons its job (as
        // before): killing the child matches the control plane, which requeues.
        // A persistent worker leaves `current` running — its events keep flowing
        // into `job_rx`, and the next session drains and replays them.
        if class == WorkerClass::Leased {
            if let Some(job) = current.take() {
                if !job.is_finished() {
                    warn!("leased worker lost its session mid-job; killing child (control plane requeues)");
                }
                job.abort();
            }
        }

        let delay = with_jitter(backoff, jitter_entropy());
        info!(delay_ms = delay.as_millis() as u64, "reconnecting after backoff");
        tokio::time::sleep(delay).await;
        backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
    }
}

/// One websocket session over the durable worker state: connect, handshake
/// (carrying a resume once we have an identity), drain-trim-replay the outbox,
/// then pump messages until the socket dies. Because the job, outbox, class, and
/// worker id outlive the session in [`main`], returning here is just a
/// disconnect, not a job loss — the caller reconnects and the next session
/// replays. Returns `Ok(())` on a clean disconnect, `Err` on a handshake failure
/// worth logging.
#[allow(clippy::too_many_arguments)]
async fn run_session(
    args: &Args,
    worker_id: &mut Option<WorkerId>,
    class: &mut WorkerClass,
    current: &mut Option<RunningJob>,
    outbox: &mut Vec<(u64, WorkerToCp)>,
    seq: &Seq,
    job_tx: &mpsc::UnboundedSender<WorkerToCp>,
    job_rx: &mut mpsc::UnboundedReceiver<WorkerToCp>,
    self_drain_at: Option<tokio::time::Instant>,
) -> Result<()> {
    // REST origin for input fetch and output upload, derived from the same host
    // we dial the websocket on.
    let origin = httpclient::origin_from_ws_url(&args.url)?;
    let (socket, _response) = connect_async(args.url.as_str()).await.context("connecting")?;
    let (mut sink, mut stream) = socket.split();

    // Present the retained identity on reconnect: the job still running and the
    // highest seq we have produced, so the control plane resynchronises instead
    // of requeuing or double-assigning. A first join has no identity yet and
    // sends no resume at all.
    let resume = worker_id.as_ref().map(|id| Resume {
        worker_id: id.clone(),
        running_jobs: current
            .as_ref()
            .filter(|running| !running.is_finished())
            .map(|running| running.job_id().clone())
            .into_iter()
            .collect(),
        high_water: outbox.last().map(|(s, _)| *s).unwrap_or(SEQ_ORIGIN),
    });

    let labels: Vec<String> = args
        .labels
        .split(LABEL_SEPARATOR)
        .filter(|label| !label.is_empty())
        .map(str::to_owned)
        .collect();
    // A leased/spot host may be reclaimed under the worker at any time.
    let preemptible = args.lease_min.is_some();
    let hello = WorkerToCp::Hello {
        protocol_version: PROTOCOL_VERSION,
        worker_semver: env!("CARGO_PKG_VERSION").into(),
        target_triple: target_triple(),
        token: args.token.clone(),
        capabilities: capabilities::detect(&labels, preemptible).await,
        resume,
    };
    sink.send(to_frame(&hello)?).await.context("sending hello")?;

    let resumed_high_water = match next_message(&mut stream).await {
        Some(CpToWorker::HelloAck {
            worker_id: assigned,
            class: assigned_class,
            telemetry,
            wall_deadline,
            resumed_high_water,
            ..
        }) => {
            // Telemetry sampling and the wall clock are accepted here; wiring the
            // system-telemetry sampler is a later milestone. The class decides
            // whether a future disconnect abandons the job; the high water tells
            // us where to resume the replay.
            info!(
                worker = %assigned, ?assigned_class, ?telemetry, ?wall_deadline,
                resumed_high_water, "hello acknowledged"
            );
            *worker_id = Some(assigned);
            *class = assigned_class;
            resumed_high_water
        }
        Some(CpToWorker::HelloReject {
            reason,
            min_protocol,
            url,
            sha256,
        }) => {
            // The control plane sends a binary url + checksum only to a worker it
            // wants to self-update (a persistent one, spec §3). With both present
            // we download → verify → atomically replace → re-exec. Otherwise
            // (a leased worker, or no binary to offer) we exit rather than
            // reconnect-loop against a version we can't satisfy.
            match (url, sha256) {
                (Some(url), Some(sha256)) => {
                    warn!(%reason, min_protocol, %url, "control plane requires a newer worker; self-updating");
                    let exe = selfupdate::download_and_replace(&url, &sha256)
                        .await
                        .context("self-update")?;
                    // Re-exec the replaced binary with our original args; on
                    // success this never returns (the process is replaced).
                    return Err(reexec(&exe));
                }
                _ => {
                    error!(%reason, min_protocol, "hello rejected with no update available; exiting");
                    std::process::exit(EXIT_CODE_REJECTED);
                }
            }
        }
        other => anyhow::bail!("unexpected handshake response: {other:?}"),
    };

    // Drain, trim, and replay before the loop so the control plane sees every
    // event exactly once (chuk-compute M3.2):
    //   1. sweep up events the executor produced while we were disconnected,
    //   2. drop those the control plane has already applied,
    //   3. resend the rest; a failed send means the fresh socket is already gone,
    //      so bail and let the next session retry.
    while let Ok(event) = job_rx.try_recv() {
        outbox.push((event_seq(&event), event));
    }
    trim_to_high_water(outbox, resumed_high_water);
    for (_, event) in outbox.iter() {
        if sink.send(to_frame(event)?).await.is_err() {
            return Ok(());
        }
    }

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Host telemetry (chuk-compute M4): sample GPU/CPU/memory and stream it as a
    // `sys/*` metric. Like the heartbeat it is latest-value data, so it is sent
    // straight to the socket, never outboxed or replayed.
    let mut sampler = telemetry::Sampler::new();
    let mut sample = tokio::time::interval(SYS_SAMPLE_INTERVAL);
    sample.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Once draining, take no new work and report Drained.
    let mut draining = false;

    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                // Liveness carries no seq and is never outboxed or replayed.
                if sink.send(to_frame(&WorkerToCp::Heartbeat)?).await.is_err() {
                    return Ok(());
                }
            }
            _ = sample.tick() => {
                let values = sampler.sample().await;
                if !values.is_empty() {
                    // Host telemetry is out-of-band from the job event stream: no
                    // job_id/step, seq 0, sent straight to the socket (never
                    // outboxed). It must NOT share the job seq counter — a sample
                    // racing a job event to the socket could otherwise bump the
                    // control plane's high-water and make it drop that event.
                    let msg = WorkerToCp::Metric {
                        seq: 0,
                        job_id: None,
                        step: None,
                        values,
                    };
                    if sink.send(to_frame(&msg)?).await.is_err() {
                        return Ok(());
                    }
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
                // Drained is a lifecycle signal (no seq); send it straight to the
                // socket, not through the outbox.
                if sink.send(to_frame(&WorkerToCp::Drained)?).await.is_err() {
                    return Ok(());
                }
            }
            // An event the executor produced: record it for replay, then stream
            // it. On a send error it stays in the outbox for the next session.
            event = job_rx.recv() => {
                let event = event.expect("job_tx is held by main for the worker's lifetime");
                outbox.push((event_seq(&event), event.clone()));
                if sink.send(to_frame(&event)?).await.is_err() {
                    return Ok(());
                }
            }
            inbound = next_message(&mut stream) => match inbound {
                Some(CpToWorker::AssignJob { job }) => {
                    if draining {
                        warn!(job = %job.id, "assign received while draining; ignoring");
                        continue;
                    }
                    if current.as_ref().is_some_and(|running| !running.is_finished()) {
                        // The control plane never double-assigns; if it does,
                        // refuse loudly rather than corrupt the slot.
                        warn!(job = %job.id, "assign received while busy; ignoring");
                        continue;
                    }
                    info!(job = %job.id, "assigned");
                    // A new job starts a fresh outbox; the previous job's events
                    // are done and must never replay against this one.
                    outbox.clear();
                    *current = Some(executor::spawn(job, job_tx.clone(), seq.clone(), origin.clone()));
                }
                Some(CpToWorker::Cancel { job_id }) => {
                    if let Some(job) = current.take() {
                        info!(job = %job_id, "cancelling");
                        let killed = job.job_id().clone();
                        job.abort();
                        // Aborting drops the executor future silently, so we emit
                        // the terminal event ourselves — routed through the
                        // executor channel so it is outboxed and replayable like
                        // any streamed event, and stamped from the shared counter.
                        let _ = job_tx.send(WorkerToCp::JobKilled {
                            seq: seq.next(),
                            job_id: killed,
                            reason: KillReason::Cancel,
                        });
                    }
                }
                Some(CpToWorker::Drain { deadline }) => {
                    info!(deadline, "drain requested by control plane");
                    draining = true;
                    if sink.send(to_frame(&WorkerToCp::Drained)?).await.is_err() {
                        return Ok(());
                    }
                }
                Some(other) => {
                    // HelloAck/HelloReject mid-session, or a variant a newer
                    // control plane introduced: tolerate and ignore.
                    warn!(?other, "unexpected control message mid-session; ignoring");
                }
                None => return Ok(()),
            },
        }
    }
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
    fn target_triple_is_a_distribution_triple() {
        // On a supported host it is a full distribution triple the control plane
        // serves (so the self-update URL /agent/<triple> is valid); on anything
        // else it falls back to `<arch>-<os>`.
        let triple = target_triple();
        let supported = [
            "x86_64-unknown-linux-musl",
            "aarch64-unknown-linux-musl",
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
        ];
        let fallback = format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS);
        assert!(
            supported.contains(&triple.as_str()) || triple == fallback,
            "unexpected triple: {triple}"
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
