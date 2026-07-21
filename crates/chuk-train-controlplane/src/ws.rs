//! The worker websocket endpoint (spec §7): one outbound connection per worker,
//! `Hello` first, then a bidirectional message loop over the compute-generic
//! protocol (`chuk-compute-wire`).

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use chuk_compute_wire as wire;
use chuk_train_proto::{Hardware, WorkerId, REGISTER_TIMEOUT};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::apikey;
use crate::AppState;

const WORKER_ID_PREFIX: &str = "w-";
const REJECT_BAD_TOKEN: &str = "bad join token";
const REJECT_NOT_HELLO: &str = "first message must be hello";
const BYTES_PER_MB: u64 = 1_048_576;

pub async fn agent_ws(State(state): State<Arc<AppState>>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| session(state, socket))
}

async fn session(state: Arc<AppState>, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();

    // Phase 1: handshake, bounded by REGISTER_TIMEOUT.
    let handshake = tokio::time::timeout(REGISTER_TIMEOUT, next_worker_message(&mut stream)).await;
    let (worker_id, class, labels, hardware) = match handshake {
        Ok(Some(wire::WorkerToCp::Hello {
            token,
            protocol_version,
            target_triple,
            capabilities,
            resume,
            ..
        })) => {
            let Some((worker_id, class)) = resolve_join(&state, &token, resume).await else {
                warn!("worker presented a bad join token");
                let _ = send(&mut sink, &reject(REJECT_BAD_TOKEN)).await;
                return;
            };
            // Version gate (M3.3): a worker below the accepted minimum is
            // rejected — a persistent one with a self-update payload, a leased
            // one bare (it re-downloads on its next provision).
            if protocol_version < state.config.min_protocol {
                warn!(protocol_version, min = state.config.min_protocol, class = ?class, "worker protocol too old");
                let _ = send(&mut sink, &version_reject(&state, class, &target_triple).await).await;
                return;
            }
            let labels = labels_of(&capabilities);
            let hardware = hardware_of(&capabilities);
            (worker_id, class, labels, hardware)
        }
        Ok(Some(_)) => {
            let _ = send(&mut sink, &reject(REJECT_NOT_HELLO)).await;
            return;
        }
        Ok(None) | Err(_) => {
            debug!("socket closed or timed out before handshake");
            return;
        }
    };

    // Phase 2: acknowledge, attach to the hub; the hub writes to `tx`, we pump
    // `rx` to the sink.
    let (tx, mut rx) = mpsc::unbounded_channel::<wire::CpToWorker>();
    let ack = wire::CpToWorker::HelloAck {
        worker_id: wire::WorkerId::from(worker_id.0.clone()),
        class,
        telemetry: wire::TelemetryConfig::default(),
        wall_deadline: None,
        resumed_high_water: state.hub.resume_high_water(&worker_id),
    };
    if send(&mut sink, &ack).await.is_err() {
        return;
    }
    if let Err(error) = state.hub.attach(&worker_id, tx, &labels, &hardware).await {
        warn!(worker = %worker_id, %error, "attach failed");
        return;
    }

    loop {
        tokio::select! {
            outbound = rx.recv() => match outbound {
                Some(msg) => {
                    if send(&mut sink, &msg).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
            inbound = next_worker_message(&mut stream) => match inbound {
                Some(msg) => {
                    if let Err(error) = state.hub.on_message(&worker_id, msg).await {
                        warn!(worker = %worker_id, %error, "error handling worker message");
                    }
                }
                None => break,
            },
        }
    }

    info!(worker = %worker_id, "session ended");
    if let Err(error) = state.hub.detach(&worker_id, class).await {
        warn!(worker = %worker_id, %error, "detach failed");
    }
}

fn reject(reason: &str) -> wire::CpToWorker {
    wire::CpToWorker::HelloReject {
        reason: reason.to_owned(),
        min_protocol: wire::PROTOCOL_VERSION,
        url: None,
        sha256: None,
    }
}

/// The reject for a too-old worker (M3.3). A **persistent** worker gets a
/// self-update payload — the download URL for its target + that binary's
/// checksum — so it can update in place; a leased worker (or one whose target we
/// don't serve) gets a bare reject and exits.
async fn version_reject(
    state: &AppState,
    class: wire::WorkerClass,
    target_triple: &str,
) -> wire::CpToWorker {
    let min = state.config.min_protocol;
    let reason = format!("worker protocol below the minimum ({min})");
    if class == wire::WorkerClass::Persistent {
        if let Some(sha256) =
            crate::api::binary_sha(state.config.agent_dir.as_deref(), target_triple).await
        {
            let base = state.config.public_url.trim_end_matches('/');
            return wire::CpToWorker::HelloReject {
                reason,
                min_protocol: min,
                url: Some(format!("{base}/agent/{target_triple}")),
                sha256: Some(sha256),
            };
        }
    }
    wire::CpToWorker::HelloReject {
        reason,
        min_protocol: min,
        url: None,
        sha256: None,
    }
}

/// Resolve a `Hello` token to `(worker id, class)`, or `None` for a bad token.
/// A single-use `cj_` provision join token (spec §12) enrols exactly the
/// **leased** worker it was minted for; a **persistent** worker token (spec §5,
/// M3) resolves to its bound, stable id; the legacy shared join token stays
/// accepted for local dev / manual joins — production provisioning no longer
/// hands it out.
async fn resolve_join(
    state: &AppState,
    token: &str,
    resume: Option<wire::Resume>,
) -> Option<(WorkerId, wire::WorkerClass)> {
    if token.starts_with(chuk_train_proto::JOIN_TOKEN_PREFIX) {
        // Never fall through to the other token kinds: a cj_ token is either
        // valid for its bound identity or rejected.
        let resolved = state
            .hub
            .store
            .resolve_join_token(&apikey::hash_token(token), now())
            .await
            .ok()
            .flatten();
        return admit_join_token(resolved, resume.as_ref());
    }
    if token == state.config.join_token {
        // The wire and control-plane WorkerId are distinct types; convert across.
        let worker_id = resume
            .map(|r| WorkerId(r.worker_id.0))
            .unwrap_or_else(|| WorkerId(format!("{WORKER_ID_PREFIX}{}", short_id())));
        return Some((worker_id, wire::WorkerClass::Leased));
    }
    if let Ok(Some(info)) = state
        .hub
        .store
        .resolve_worker_token(&apikey::hash_token(token))
        .await
    {
        let _ = state.hub.store.touch_worker_token(&info.id, now()).await;
        return Some((info.worker_id, wire::WorkerClass::Persistent));
    }
    None
}

/// The single-use admission rule, pure for testability: first use enrols the
/// token's bound identity (whatever the Hello claims); a consumed token
/// readmits ONLY a reconnect of that same bound id — it can never enrol a
/// second worker or claim a different identity.
fn admit_join_token(
    resolved: Option<(WorkerId, bool)>,
    resume: Option<&wire::Resume>,
) -> Option<(WorkerId, wire::WorkerClass)> {
    match resolved {
        Some((worker_id, true)) => Some((worker_id, wire::WorkerClass::Leased)),
        Some((worker_id, false))
            if resume.is_some_and(|r| r.worker_id.0 == worker_id.0) =>
        {
            Some((worker_id, wire::WorkerClass::Leased))
        }
        _ => None,
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Flatten the worker's label map into the control plane's `k=v` / bare-`k` list.
fn labels_of(caps: &wire::Capabilities) -> Vec<String> {
    caps.labels
        .iter()
        .map(|(k, v)| if v.is_empty() { k.clone() } else { format!("{k}={v}") })
        .collect()
}

/// Project the worker's generic capabilities onto the control plane's `Hardware`
/// record (what the fleet view + scheduler read today).
fn hardware_of(caps: &wire::Capabilities) -> Hardware {
    let (gpu, vram_mb, driver) = match &caps.accelerator {
        wire::Accelerator::Cuda { devices } => match devices.first() {
            Some(d) => (
                Some(d.name.clone()),
                Some(d.vram_bytes / BYTES_PER_MB),
                d.driver_version.clone(),
            ),
            None => (None, None, None),
        },
        wire::Accelerator::Mps { chip, unified_memory_bytes } => {
            (Some(chip.clone()), Some(unified_memory_bytes / BYTES_PER_MB), None)
        }
        // Cpu, plus any future accelerator kind (#[non_exhaustive]).
        _ => (None, None, None),
    };
    Hardware {
        host: format!("{}-{}", caps.os, caps.arch),
        os: caps.os.clone(),
        gpu,
        vram_mb,
        driver,
    }
}

/// Read the next parseable worker message; `None` means the socket is gone.
/// Unparseable frames are logged and skipped rather than killing the session.
async fn next_worker_message(
    stream: &mut (impl StreamExt<Item = Result<Message, axum::Error>> + Unpin),
) -> Option<wire::WorkerToCp> {
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Text(text)) => match serde_json::from_str::<wire::WorkerToCp>(&text) {
                Ok(msg) => return Some(msg),
                Err(error) => warn!(%error, "unparseable worker message; skipping"),
            },
            Ok(Message::Close(_)) => return None,
            Ok(_) => {} // ping/pong/binary: nothing to do
            Err(_) => return None,
        }
    }
    None
}

async fn send(
    sink: &mut (impl SinkExt<Message, Error = axum::Error> + Unpin),
    msg: &wire::CpToWorker,
) -> Result<(), axum::Error> {
    let payload = serde_json::to_string(msg).expect("CpToWorker always serialises");
    sink.send(Message::Text(payload.into())).await
}

fn short_id() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn join_token_admission_is_single_identity() {
        let bound = WorkerId("vast-abc123".into());
        let resume_bound = wire::Resume {
            worker_id: wire::WorkerId::from(bound.0.clone()),
            running_jobs: Vec::new(),
            high_water: 0,
        };
        let resume_other = wire::Resume {
            worker_id: wire::WorkerId::from("vast-other"),
            running_jobs: Vec::new(),
            high_water: 0,
        };
        // First use enrols the bound identity.
        assert_eq!(
            admit_join_token(Some((bound.clone(), true)), None),
            Some((bound.clone(), wire::WorkerClass::Leased))
        );
        // A consumed token readmits only its own bound id (reconnect)...
        assert_eq!(
            admit_join_token(Some((bound.clone(), false)), Some(&resume_bound)),
            Some((bound.clone(), wire::WorkerClass::Leased))
        );
        // ...never a different claimed identity, and never a fresh enrol.
        assert_eq!(admit_join_token(Some((bound.clone(), false)), Some(&resume_other)), None);
        assert_eq!(admit_join_token(Some((bound, false)), None), None);
        // Unknown/expired token: rejected outright.
        assert_eq!(admit_join_token(None, Some(&resume_bound)), None);
    }

    fn caps(accelerator: wire::Accelerator, labels: BTreeMap<String, String>) -> wire::Capabilities {
        wire::Capabilities {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpu_cores: 8,
            ram_bytes: 0,
            free_disk_bytes: 0,
            preemptible: true,
            accelerator,
            labels,
        }
    }

    #[test]
    fn labels_flatten_to_bare_and_kv() {
        let labels = BTreeMap::from([
            ("colab".into(), String::new()),
            ("site".into(), "home".into()),
        ]);
        let mut out = labels_of(&caps(wire::Accelerator::Cpu, labels));
        out.sort();
        assert_eq!(out, vec!["colab".to_owned(), "site=home".to_owned()]);
    }

    #[test]
    fn cuda_capabilities_map_to_gpu_hardware() {
        let acc = wire::Accelerator::Cuda {
            devices: vec![wire::GpuInfo {
                name: "Tesla T4".into(),
                vram_bytes: 16 * BYTES_PER_MB * 1024,
                driver_version: Some("535".into()),
                cuda_version: None,
            }],
        };
        let hw = hardware_of(&caps(acc, BTreeMap::new()));
        assert_eq!(hw.gpu.as_deref(), Some("Tesla T4"));
        assert_eq!(hw.vram_mb, Some(16 * 1024));
        assert_eq!(hw.driver.as_deref(), Some("535"));
        assert_eq!(hw.os, "linux");
        assert_eq!(hw.host, "linux-x86_64");
    }

    #[test]
    fn cpu_and_mps_accelerators_map() {
        let cpu = hardware_of(&caps(wire::Accelerator::Cpu, BTreeMap::new()));
        assert!(cpu.gpu.is_none() && cpu.vram_mb.is_none());
        let mps = hardware_of(&caps(
            wire::Accelerator::Mps { chip: "Apple M2".into(), unified_memory_bytes: 8 * BYTES_PER_MB },
            BTreeMap::new(),
        ));
        assert_eq!(mps.gpu.as_deref(), Some("Apple M2"));
        assert_eq!(mps.vram_mb, Some(8));
    }
}
