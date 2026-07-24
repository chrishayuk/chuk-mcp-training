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
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::routing::get;
    use axum::Router;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::{SqliteStore, Store};

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

    #[test]
    fn a_cuda_worker_with_no_devices_reports_no_gpu() {
        let hw = hardware_of(&caps(
            wire::Accelerator::Cuda { devices: Vec::new() },
            BTreeMap::new(),
        ));
        assert!(hw.gpu.is_none() && hw.vram_mb.is_none() && hw.driver.is_none());
    }

    // -- the frame reader / writer, over a plain stream ----------------------

    #[tokio::test]
    async fn unparseable_and_uninteresting_frames_are_skipped_not_fatal() {
        let frames = vec![
            Ok(Message::Ping(Vec::new().into())),
            Ok(Message::Binary(vec![0xff].into())),
            Ok(Message::Text("{not json".into())),
            Ok(Message::Text(
                serde_json::to_string(&wire::WorkerToCp::Heartbeat)
                    .expect("serialise")
                    .into(),
            )),
        ];
        let mut stream = futures_util::stream::iter(frames);
        assert_eq!(
            next_worker_message(&mut stream).await,
            Some(wire::WorkerToCp::Heartbeat),
            "the reader skips past everything it can't act on"
        );
    }

    #[tokio::test]
    async fn a_close_frame_and_an_exhausted_stream_both_end_the_session() {
        let mut closed = futures_util::stream::iter(vec![Ok(Message::Close(None))]);
        assert_eq!(next_worker_message(&mut closed).await, None);

        let mut empty = futures_util::stream::iter(Vec::<Result<Message, axum::Error>>::new());
        assert_eq!(next_worker_message(&mut empty).await, None);

        let mut errored =
            futures_util::stream::iter(vec![Err(axum::Error::new(std::fmt::Error))]);
        assert_eq!(next_worker_message(&mut errored).await, None);
    }

    // -- the session, end to end over a real socket --------------------------

    const MIN_PROTOCOL: u32 = 3;
    const TARGET: &str = "aarch64-apple-darwin";

    /// A control plane serving only the agent websocket, on an ephemeral port.
    struct TestCp {
        url: String,
        state: Arc<AppState>,
    }

    async fn serve(configure: impl FnOnce(&mut Config)) -> TestCp {
        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let mut config = Config {
            api_token: "test-api-token".into(),
            join_token: "test-join-token".into(),
            store_spec: ":memory:".into(),
            artifacts_spec: "file:./unused".into(),
            public_url: "https://cp.example.com".into(),
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 0,
            providers: "mock".into(),
            agent_ws_url: "ws://127.0.0.1:0/ws".into(),
            agent_bin: None,
            agent_dir: None,
            min_protocol: MIN_PROTOCOL,
            vast_api_key: None,
            drain_window_min: 5.0,
            confirm_cost_threshold: 0.0,
            reconcile_interval: Duration::from_secs(30),
            idle_reap: Duration::from_secs(60),
            google_client_id: None,
            google_client_secret: None,
            allowed_emails: vec![],
            sysadmin_email: None,
        };
        configure(&mut config);
        let leases = LeaseManager::new(hub.clone(), Arc::new(build_providers("mock", None, None)), config.clone());
        let state = Arc::new(AppState {
            config,
            hub,
            artifacts,
            leases,
            drive: None,
            archiver: None,
            key_encryption_key: None,
        });
        let app = Router::new()
            .route("/ws/agent", get(agent_ws))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        TestCp { url: format!("ws://{addr}/ws/agent"), state }
    }

    /// A connected worker socket, with helpers for the two directions.
    struct Socket {
        inner: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    }

    impl Socket {
        async fn connect(cp: &TestCp) -> Self {
            let (inner, _) = tokio_tungstenite::connect_async(&cp.url)
                .await
                .expect("websocket handshake");
            Self { inner }
        }

        async fn send(&mut self, msg: &wire::WorkerToCp) {
            self.inner
                .send(ClientMessage::Text(
                    serde_json::to_string(msg).expect("serialise").into(),
                ))
                .await
                .expect("send");
        }

        /// The next control-plane message, or `None` if the socket closed.
        async fn recv(&mut self) -> Option<wire::CpToWorker> {
            loop {
                let frame = tokio::time::timeout(Duration::from_secs(5), self.inner.next())
                    .await
                    .expect("the control plane must answer promptly")?;
                match frame.expect("frame") {
                    ClientMessage::Text(text) => {
                        return Some(serde_json::from_str(&text).expect("parse CpToWorker"))
                    }
                    ClientMessage::Close(_) => return None,
                    _ => continue,
                }
            }
        }

        async fn close(mut self) {
            let _ = self.inner.close(None).await;
        }
    }

    fn hello(token: &str, protocol_version: u32, resume: Option<wire::Resume>) -> wire::WorkerToCp {
        wire::WorkerToCp::Hello {
            protocol_version,
            worker_semver: "0.1.0".into(),
            target_triple: TARGET.into(),
            token: token.into(),
            capabilities: caps(
                wire::Accelerator::Mps {
                    chip: "Apple M2".into(),
                    unified_memory_bytes: 16 * BYTES_PER_MB,
                },
                BTreeMap::from([("site".into(), "home".into())]),
            ),
            resume,
        }
    }

    /// Wait for a worker to appear in the fleet (attach happens just after the
    /// ack lands, so a test that reads the store races the session task).
    async fn await_worker(state: &Arc<AppState>, worker_id: &WorkerId) -> chuk_train_proto::WorkerInfo {
        for _ in 0..100 {
            if let Ok(Some(worker)) = state.hub.store.worker(worker_id).await {
                return worker;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("worker {worker_id} never joined the fleet");
    }

    #[tokio::test]
    async fn the_shared_join_token_admits_a_leased_worker_and_records_its_hardware() {
        let cp = serve(|_| {}).await;
        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello("test-join-token", MIN_PROTOCOL, None)).await;

        let wire::CpToWorker::HelloAck { worker_id, class, resumed_high_water, .. } =
            socket.recv().await.expect("an ack")
        else {
            panic!("expected a HelloAck");
        };
        assert!(worker_id.0.starts_with(WORKER_ID_PREFIX), "generated id: {worker_id:?}");
        assert_eq!(class, wire::WorkerClass::Leased);
        assert_eq!(resumed_high_water, 0, "a fresh worker has replayed nothing");

        let worker = await_worker(&cp.state, &WorkerId(worker_id.0.clone())).await;
        assert_eq!(worker.labels, vec!["site=home".to_owned()]);
        assert_eq!(worker.hardware.gpu.as_deref(), Some("Apple M2"));
        assert_eq!(worker.hardware.host, "linux-x86_64");
    }

    #[tokio::test]
    async fn a_bad_token_is_rejected_before_anything_is_attached() {
        let cp = serve(|_| {}).await;
        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello("not-the-token", MIN_PROTOCOL, None)).await;

        let wire::CpToWorker::HelloReject { reason, url, .. } =
            socket.recv().await.expect("a reject")
        else {
            panic!("expected a HelloReject");
        };
        assert_eq!(reason, REJECT_BAD_TOKEN);
        assert!(url.is_none(), "a stranger is never handed an update payload");
        assert!(cp.state.hub.store.fleet().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_first_message_that_is_not_hello_is_rejected() {
        let cp = serve(|_| {}).await;
        let mut socket = Socket::connect(&cp).await;
        socket.send(&wire::WorkerToCp::Heartbeat).await;

        let wire::CpToWorker::HelloReject { reason, .. } = socket.recv().await.expect("a reject")
        else {
            panic!("expected a HelloReject");
        };
        assert_eq!(reason, REJECT_NOT_HELLO);
    }

    #[tokio::test]
    async fn a_socket_that_closes_before_saying_hello_just_ends() {
        let cp = serve(|_| {}).await;
        let socket = Socket::connect(&cp).await;
        socket.close().await;
        // Nothing to assert but the absence of a panic and of a fleet entry.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(cp.state.hub.store.fleet().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_leased_worker_below_the_minimum_protocol_gets_a_bare_reject() {
        let cp = serve(|_| {}).await;
        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello("test-join-token", MIN_PROTOCOL - 1, None)).await;

        let wire::CpToWorker::HelloReject { reason, min_protocol, url, sha256 } =
            socket.recv().await.expect("a reject")
        else {
            panic!("expected a HelloReject");
        };
        assert_eq!(min_protocol, MIN_PROTOCOL);
        assert!(reason.contains(&MIN_PROTOCOL.to_string()), "unexpected reason: {reason}");
        assert!(url.is_none() && sha256.is_none(), "a leased worker re-downloads on provision");
    }

    /// A directory holding a fake worker binary for [`TARGET`], so the
    /// self-update payload has something to checksum.
    struct AgentDir {
        path: std::path::PathBuf,
    }

    impl AgentDir {
        fn with_binary() -> Self {
            let path = std::env::temp_dir().join(format!(
                "chuk-ws-test-agent-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&path).expect("create agent dir");
            std::fs::write(path.join(TARGET), b"a worker binary").expect("write fake binary");
            Self { path }
        }
    }

    impl Drop for AgentDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    async fn persistent_token(state: &Arc<AppState>, worker_id: &WorkerId) -> String {
        let token = format!("cw_{}", uuid::Uuid::new_v4().simple());
        state
            .hub
            .store
            .create_worker_token(
                &uuid::Uuid::new_v4().simple().to_string(),
                worker_id,
                "test worker",
                "cw_test",
                &apikey::hash_token(&token),
            )
            .await
            .expect("create worker token");
        token
    }

    #[tokio::test]
    async fn a_persistent_worker_below_the_minimum_is_told_where_to_self_update() {
        let agent_dir = AgentDir::with_binary();
        let dir = agent_dir.path.to_string_lossy().into_owned();
        let cp = serve(|config| config.agent_dir = Some(dir)).await;
        let bound = WorkerId("mac-studio".into());
        let token = persistent_token(&cp.state, &bound).await;

        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello(&token, MIN_PROTOCOL - 1, None)).await;

        let wire::CpToWorker::HelloReject { url, sha256, .. } = socket.recv().await.expect("reject")
        else {
            panic!("expected a HelloReject");
        };
        assert_eq!(url.as_deref(), Some(&*format!("https://cp.example.com/agent/{TARGET}")));
        assert_eq!(
            sha256.as_deref(),
            crate::api::binary_sha(Some(&agent_dir.path.to_string_lossy()), TARGET)
                .await
                .as_deref(),
            "the checksum must match the binary we would serve"
        );
    }

    #[tokio::test]
    async fn a_persistent_worker_we_cannot_serve_a_binary_for_gets_a_bare_reject() {
        // A configured directory with no binary for this target: nothing to
        // self-update to, so it is told to stop rather than to update.
        let cp = serve(|config| config.agent_dir = Some("/nonexistent/agent/dir".into())).await;
        let token = persistent_token(&cp.state, &WorkerId("mac-studio".into())).await;

        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello(&token, MIN_PROTOCOL - 1, None)).await;

        let wire::CpToWorker::HelloReject { url, sha256, .. } = socket.recv().await.expect("reject")
        else {
            panic!("expected a HelloReject");
        };
        assert!(url.is_none() && sha256.is_none());
    }

    #[tokio::test]
    async fn a_persistent_worker_token_resolves_to_its_own_stable_id() {
        let cp = serve(|_| {}).await;
        let bound = WorkerId("mac-studio".into());
        let token = persistent_token(&cp.state, &bound).await;

        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello(&token, MIN_PROTOCOL, None)).await;

        let wire::CpToWorker::HelloAck { worker_id, class, .. } = socket.recv().await.expect("ack")
        else {
            panic!("expected a HelloAck");
        };
        assert_eq!(worker_id.0, bound.0, "a persistent worker keeps its id across reconnects");
        assert_eq!(class, wire::WorkerClass::Persistent);
        await_worker(&cp.state, &bound).await;
    }

    #[tokio::test]
    async fn a_provision_join_token_enrols_exactly_the_worker_it_was_minted_for() {
        let cp = serve(|_| {}).await;
        let bound = WorkerId("mock-abc123".into());
        let token = apikey::mint_join_token(cp.state.hub.store.as_ref(), &bound)
            .await
            .expect("mint join token");

        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello(&token, MIN_PROTOCOL, None)).await;
        let wire::CpToWorker::HelloAck { worker_id, class, .. } = socket.recv().await.expect("ack")
        else {
            panic!("expected a HelloAck");
        };
        assert_eq!(worker_id.0, bound.0);
        assert_eq!(class, wire::WorkerClass::Leased);

        // Consumed: a second worker cannot enrol on the same credential.
        let mut impostor = Socket::connect(&cp).await;
        impostor.send(&hello(&token, MIN_PROTOCOL, None)).await;
        let wire::CpToWorker::HelloReject { reason, .. } =
            impostor.recv().await.expect("a reject")
        else {
            panic!("expected a HelloReject");
        };
        assert_eq!(reason, REJECT_BAD_TOKEN);
    }

    #[tokio::test]
    async fn a_reconnect_with_the_shared_token_readopts_the_id_it_claims() {
        let cp = serve(|_| {}).await;
        let resumed = WorkerId("w-resumed".into());
        let mut socket = Socket::connect(&cp).await;
        socket
            .send(&hello(
                "test-join-token",
                MIN_PROTOCOL,
                Some(wire::Resume {
                    worker_id: wire::WorkerId::from(resumed.0.clone()),
                    running_jobs: Vec::new(),
                    high_water: 7,
                }),
            ))
            .await;

        let wire::CpToWorker::HelloAck { worker_id, .. } = socket.recv().await.expect("ack") else {
            panic!("expected a HelloAck");
        };
        assert_eq!(worker_id.0, resumed.0);
    }

    #[tokio::test]
    async fn messages_flow_both_ways_until_the_worker_disconnects() {
        let cp = serve(|_| {}).await;
        let mut socket = Socket::connect(&cp).await;
        socket.send(&hello("test-join-token", MIN_PROTOCOL, None)).await;
        let wire::CpToWorker::HelloAck { worker_id, .. } = socket.recv().await.expect("ack") else {
            panic!("expected a HelloAck");
        };
        let worker_id = WorkerId(worker_id.0);
        await_worker(&cp.state, &worker_id).await;

        // Worker → control plane: an unparseable frame doesn't kill the
        // session, and the heartbeat behind it is still handled.
        socket
            .inner
            .send(ClientMessage::Text("{not json".into()))
            .await
            .expect("send garbage");
        socket.send(&wire::WorkerToCp::Heartbeat).await;

        // Control plane → worker: whatever the hub queues arrives on the wire.
        let deadline = 1_234.0;
        for _ in 0..100 {
            if cp
                .state
                .hub
                .send_to(&worker_id, wire::CpToWorker::Drain { deadline })
                .await
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            socket.recv().await.expect("a drain"),
            wire::CpToWorker::Drain { deadline }
        );

        // And a close detaches it from the fleet.
        socket.close().await;
        for _ in 0..100 {
            match cp.state.hub.store.worker(&worker_id).await.expect("worker") {
                Some(worker) if worker.state == chuk_train_proto::WorkerState::Disconnected => return,
                _ => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }
        panic!("the worker never left the fleet");
    }
}
