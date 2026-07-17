//! Every number and path the control plane and agent must agree on.

use std::time::Duration;

/// Default TCP port for the control plane (HTTP + agent websocket).
pub const DEFAULT_PORT: u16 = 8700;

/// Path agents dial for the outbound websocket (spec §7).
pub const AGENT_WS_PATH: &str = "/ws/agent";
/// Prefix for the bearer-authenticated REST API.
pub const API_PREFIX: &str = "/api";
/// Unauthenticated liveness probe.
pub const HEALTH_PATH: &str = "/healthz";

/// How often a connected agent sends a heartbeat.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// Heartbeat silence after which the control plane treats a worker as
/// unreachable (spec §7: "Heartbeat loss > 90s ⇒ unreachable").
pub const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(90);
/// How long the control plane waits for the `register` message on a fresh
/// websocket before dropping it.
pub const REGISTER_TIMEOUT: Duration = Duration::from_secs(15);

/// Agent reconnect backoff bounds (exponential, doubling).
pub const RECONNECT_BACKOFF_MIN: Duration = Duration::from_secs(2);
pub const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Default wall-clock limit for a shell run.
pub const DEFAULT_SHELL_TIMEOUT: Duration = Duration::from_secs(600);
/// Default wall-clock limit for a single train slice (spec: slices run until
/// drain; M0/M1 has no lease wall yet, so this is the only bound).
pub const DEFAULT_TRAIN_TIMEOUT: Duration = Duration::from_secs(12 * 3600);

/// How often the agent scans the checkpoint directory for freshly-`.ready`
/// checkpoints to upload.
pub const CHECKPOINT_SCAN_INTERVAL: Duration = Duration::from_secs(3);
/// How long a per-run upload grant (scoped capability token) stays valid.
pub const UPLOAD_GRANT_TTL: Duration = Duration::from_secs(24 * 3600);
/// Default TTL for a signed artifact URL handed to a reader (lazarus pulls).
pub const DEFAULT_ARTIFACT_URL_TTL: Duration = Duration::from_secs(3600);

/// Default number of log lines returned by the tail endpoint.
pub const DEFAULT_LOG_TAIL_LINES: u32 = 100;
/// Default page size for run listings.
pub const DEFAULT_RUN_LIST_LIMIT: u32 = 50;
/// Default max metric points returned per key before downsampling kicks in.
pub const DEFAULT_METRIC_DOWNSAMPLE: u32 = 500;

// -- script contract filenames (spec §5.1, §11.2) --------------------------

/// Per-checkpoint subdirectory name pattern is `step_<n>`; this is the prefix.
pub const CHECKPOINT_DIR_PREFIX: &str = "step_";
/// Marker file the trainer touches once a checkpoint dir is fully written;
/// the agent only uploads checkpoints that carry it (avoids partial reads).
pub const CHECKPOINT_READY_MARKER: &str = ".ready";
/// Lineage sidecar written next to the model weights in every checkpoint dir.
pub const CHECKPOINT_META_FILE: &str = "meta.json";
/// Model weights filename the harness looks for (and lazarus loads).
pub const CHECKPOINT_MODEL_FILE: &str = "model.safetensors";
/// Optimizer state filename (optional; excluded from lazarus pulls, spec §10).
pub const CHECKPOINT_OPTIM_FILE: &str = "optim.pt";

/// Code-unit filenames (spec §11.1).
pub const CODE_UNIT_TARBALL: &str = "unit.tar.zst";
pub const CODE_UNIT_MANIFEST: &str = "unit.toml";
pub const CODE_UNIT_LOCKFILE: &str = "uv.lock";

/// Synthetic exit code recorded when the agent kills a run for exceeding its
/// timeout (mirrors SIGKILL's shell convention).
pub const EXIT_CODE_TIMEOUT: i64 = -9;
/// Synthetic exit code recorded when the agent failed to spawn or supervise
/// the process at all.
pub const EXIT_CODE_AGENT_ERROR: i64 = -1;

/// Environment variable names shared across components.
pub mod env {
    /// Bearer token protecting `/api/*` (and the MCP surface that fronts it).
    pub const API_TOKEN: &str = "CHUK_TRAIN_API_TOKEN";
    /// Token agents present in their `register` message.
    pub const JOIN_TOKEN: &str = "CHUK_TRAIN_JOIN_TOKEN";
    /// Store backend URL: `sqlite:path.db` (bare path = SQLite), `redis:` reserved.
    pub const STORE_URL: &str = "CHUK_TRAIN_STORE";
    /// Legacy/simple form: bare SQLite path (used when STORE_URL is unset).
    pub const DB_PATH: &str = "CHUK_TRAIN_DB";
    /// Control-plane bind host.
    pub const HOST: &str = "CHUK_TRAIN_HOST";
    /// Control-plane bind port (PaaS-style `PORT` is honoured as a fallback).
    pub const PORT: &str = "CHUK_TRAIN_PORT";
    /// Base URL of the control plane, used by the MCP client.
    pub const CP_URL: &str = "CHUK_TRAIN_URL";
    /// Root directory for the artifact blob store (filesystem backend).
    pub const ARTIFACTS_DIR: &str = "CHUK_TRAIN_ARTIFACTS";
    /// Directory the agent caches extracted code units in, keyed by sha.
    pub const AGENT_CACHE_DIR: &str = "CHUK_TRAIN_CACHE";
}

/// Environment variables the harness exports to a train entrypoint — the
/// script contract (spec §5.1). A trainer reads these; ~5 lines to adopt.
pub mod script_env {
    /// Absolute path to the resolved config file (empty if the job set none).
    pub const CONFIG: &str = "CHUK_CONFIG";
    /// JSON object of config overrides (e.g. `{"seed": 81}`).
    pub const OVERRIDES: &str = "CHUK_OVERRIDES";
    /// Absolute path the trainer appends metrics to, one JSON object per line.
    pub const METRICS: &str = "CHUK_METRICS";
    /// Absolute directory the trainer writes `step_<n>/` checkpoints into.
    pub const CKPT_DIR: &str = "CHUK_CKPT_DIR";
    /// Absolute path to a checkpoint dir to resume from (empty on a fresh run).
    pub const RESUME_CKPT: &str = "CHUK_RESUME_CKPT";
    /// The run id, for the trainer's own logging/provenance.
    pub const RUN_ID: &str = "CHUK_RUN_ID";
    /// The seed for this run (from the job's seed/overrides), if any.
    pub const SEED: &str = "CHUK_SEED";
}
