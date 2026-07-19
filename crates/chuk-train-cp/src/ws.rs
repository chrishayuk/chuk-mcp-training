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

use crate::AppState;

const WORKER_ID_PREFIX: &str = "w-";
const REJECT_BAD_TOKEN: &str = "bad join token";
const REJECT_NOT_HELLO: &str = "first message must be hello";
/// M1 serves a single worker class; the persistent class arrives at M3.
const M1_WORKER_CLASS: wire::WorkerClass = wire::WorkerClass::Leased;
const BYTES_PER_MB: u64 = 1_048_576;

pub async fn agent_ws(State(state): State<Arc<AppState>>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| session(state, socket))
}

async fn session(state: Arc<AppState>, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();

    // Phase 1: handshake, bounded by REGISTER_TIMEOUT.
    let handshake = tokio::time::timeout(REGISTER_TIMEOUT, next_worker_message(&mut stream)).await;
    let (worker_id, labels, hardware) = match handshake {
        Ok(Some(wire::WorkerToCp::Hello {
            token,
            capabilities,
            resume,
            ..
        })) => {
            if token != state.config.join_token {
                warn!("worker presented a bad join token");
                let _ = send(&mut sink, &reject(REJECT_BAD_TOKEN)).await;
                return;
            }
            // The wire and control-plane WorkerId are distinct types (different
            // crates); convert across the boundary.
            let worker_id = resume
                .map(|r| WorkerId(r.worker_id.0))
                .unwrap_or_else(|| WorkerId(format!("{WORKER_ID_PREFIX}{}", short_id())));
            let labels = labels_of(&capabilities);
            let hardware = hardware_of(&capabilities);
            (worker_id, labels, hardware)
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
        class: M1_WORKER_CLASS,
        telemetry: wire::TelemetryConfig::default(),
        wall_deadline: None,
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
    if let Err(error) = state.hub.detach(&worker_id).await {
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
