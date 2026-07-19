//! Every number and path the control plane and agent must agree on.

use std::time::Duration;

/// Default TCP port for the control plane (HTTP + agent websocket).
pub const DEFAULT_PORT: u16 = 8700;

/// Path workers dial for the outbound websocket (spec §7).
pub const AGENT_WS_PATH: &str = "/ws/agent";
/// Unauthenticated liveness probe.
pub const HEALTH_PATH: &str = "/healthz";

// -- worker distribution (chuk-compute M2) ----------------------------------
// The control plane serves a per-target worker binary + its checksum, a version
// endpoint for the self-updater, and the one-shot installer. All public (the
// worker is public code, spec §12).

/// Route for the per-target worker binary + its checksum: `/agent/<target>` and
/// `/agent/<target>.sha256`, where `<target>` is one of [`SUPPORTED_TARGETS`].
pub const AGENT_DOWNLOAD_ROUTE: &str = "/agent/{name}";
/// Suffix that turns a binary path into its checksum path.
pub const AGENT_SHA256_SUFFIX: &str = ".sha256";
/// Current worker version, for the persistent worker's self-updater (M3).
pub const AGENT_VERSION_PATH: &str = "/agent/version";
/// The rustup-style one-shot installer: detects the target, downloads + verifies
/// the matching binary, and execs the worker joined to the fleet.
pub const INSTALL_SCRIPT_PATH: &str = "/install.sh";

/// Target triples the control plane distributes a worker for. Also the download
/// allowlist (a request for anything else 404s — no path traversal). The two
/// linux-musl targets are cross-built into the deployed image; the darwin
/// targets come from a macOS build (CI / local).
pub const SUPPORTED_TARGETS: [&str; 4] = [
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
];

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

// -- leases (spec §3) -------------------------------------------------------

/// Minutes reserved at the end of a lease for checkpoint + upload (T-drain).
pub const DEFAULT_DRAIN_WINDOW_MIN: f64 = 5.0;
/// Below this remaining time the scheduler stops assigning non-atomic jobs.
pub const ASSIGN_CUTOFF_MIN: f64 = 10.0;
/// A leased worker idle (empty queue, nothing assignable) this long is
/// drained + destroyed early — the lease is a ceiling, not a commitment.
pub const DEFAULT_IDLE_REAP: Duration = Duration::from_secs(10 * 60);
/// How often the lease manager re-checks every lease's clock.
pub const LEASE_TICK_INTERVAL: Duration = Duration::from_secs(1);
/// How often the reconcile loop diffs provider instances against the registry.
pub const DEFAULT_RECONCILE_INTERVAL: Duration = Duration::from_secs(10 * 60);
/// After calling destroy, how long to keep polling provider status for `Gone`
/// before raising an orphan alert.
pub const DESTROY_VERIFY_TIMEOUT: Duration = Duration::from_secs(120);
/// How often to poll provider status while verifying a destroy.
pub const DESTROY_VERIFY_POLL: Duration = Duration::from_secs(2);

/// How often the archive/retention sweep runs — it promptly tiers newly
/// completed runs to Drive and backstops any a prior pass missed (spec §11.5).
pub const DEFAULT_ARCHIVE_INTERVAL: Duration = Duration::from_secs(60);
/// How often the experiments-server reporting outbox retries undelivered
/// events (created/state/checkpoint/result reports that failed on first try).
pub const DEFAULT_EXPERIMENTS_OUTBOX_INTERVAL: Duration = Duration::from_secs(30);
/// Days after which R2 lifecycle expires hot (ckpt-hot) checkpoints, and the
/// promoted final (ckpt-final) copies. The hot window is the error-recovery
/// grace; the final window is a warm cache since Drive holds the canonical copy.
pub const CKPT_HOT_TTL_DAYS: i32 = 1;
pub const CKPT_FINAL_TTL_DAYS: i32 = 30;

/// The single default team (RBAC). Multi-team is a later addition; `users` and
/// `api_keys` already carry `team_id` so it won't be a refactor.
pub const DEFAULT_TEAM_ID: &str = "default";
pub const DEFAULT_TEAM_NAME: &str = "Default";
/// Prefix on generated MCP API keys, so they're recognisable.
pub const API_KEY_PREFIX: &str = "ck_";
/// Prefix on generated persistent worker tokens (chuk-compute M3.1), so they're
/// recognisable and distinct from `ck_` user/MCP keys.
pub const WORKER_TOKEN_PREFIX: &str = "cw_";

/// chuk-experiments-server reporting mirror (spec §11.6) — the default
/// programme/experiment harness runs report into. Optional and gated: the whole
/// mirror is a no-op unless `EXPERIMENTS_URL` + `EXPERIMENTS_API_KEY` are set.
/// Both slugs are env-overridable; the experiments-server auto-creates the
/// programme the first time an experiment references it.
pub const DEFAULT_EXPERIMENTS_PROGRAMME: &str = "gpu-training-harness";
pub const DEFAULT_EXPERIMENTS_PROGRAMME_TITLE: &str = "GPU training harness";
pub const DEFAULT_EXPERIMENTS_EXPERIMENT: &str = "harness-runs";
pub const DEFAULT_EXPERIMENTS_EXPERIMENT_TITLE: &str = "Harness runs";

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

// -- artifact storage prefixes (spec §11.5) --------------------------------
// R2 lifecycle rules filter by leading key prefix only, so ephemeral vs
// promoted checkpoints get *top-level* prefixes (not `runs/<id>/…`, where the
// id precedes `ckpt` and no prefix could isolate them).

/// Hot (ephemeral) step checkpoints the agent uploads live under this prefix.
/// An R2 lifecycle rule expires it on a short timer (the grace window).
pub const CKPT_HOT_PREFIX: &str = "ckpt-hot";
/// The promoted final checkpoint (copied here on run completion) lives under
/// this prefix; a longer R2 lifecycle rule expires it once Drive holds the
/// canonical copy.
pub const CKPT_FINAL_PREFIX: &str = "ckpt-final";

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
/// Synthetic exit code recorded for an operator-cancelled run (mirrors SIGTERM,
/// the first signal the worker sends on `Cancel`).
pub const EXIT_CODE_CANCELLED: i64 = -15;

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
    /// Reconcile-loop interval override (seconds); short values speed tests.
    pub const RECONCILE_INTERVAL_S: &str = "CHUK_TRAIN_RECONCILE_S";
    /// Idle-reaper threshold override (seconds).
    pub const IDLE_REAP_S: &str = "CHUK_TRAIN_IDLE_REAP_S";
    /// Drain-window override (minutes); short values speed local lease tests.
    pub const DRAIN_WINDOW_MIN: &str = "CHUK_TRAIN_DRAIN_WINDOW_MIN";
    /// Provider selection for the mock/real driver registry (e.g. "mock,vast").
    pub const PROVIDERS: &str = "CHUK_TRAIN_PROVIDERS";
    /// Vast API key (VastProvider only; VPS-side, never on workers).
    pub const VAST_API_KEY: &str = "CHUK_TRAIN_VAST_API_KEY";
    /// S3/R2 endpoint, e.g. `https://<account>.r2.cloudflarestorage.com`.
    pub const S3_ENDPOINT: &str = "CHUK_TRAIN_S3_ENDPOINT";
    /// S3/R2 signing region (R2 uses `auto`).
    pub const S3_REGION: &str = "CHUK_TRAIN_S3_REGION";
    /// S3/R2 access key id (R2 API token access key).
    pub const S3_ACCESS_KEY_ID: &str = "CHUK_TRAIN_S3_ACCESS_KEY_ID";
    /// S3/R2 secret access key (R2 API token secret).
    pub const S3_SECRET_ACCESS_KEY: &str = "CHUK_TRAIN_S3_SECRET_ACCESS_KEY";
    /// Control-plane websocket URL provisioned workers dial back on.
    pub const AGENT_WS_URL: &str = "CHUK_TRAIN_AGENT_WS_URL";
    /// Google OAuth web-client id (dashboard sign-in). Auth is off if unset.
    pub const GOOGLE_CLIENT_ID: &str = "CHUK_TRAIN_GOOGLE_CLIENT_ID";
    /// Google OAuth web-client secret.
    pub const GOOGLE_CLIENT_SECRET: &str = "CHUK_TRAIN_GOOGLE_CLIENT_SECRET";
    /// Long-lived offline refresh token (drive.file scope) for the Drive cold
    /// archive tier. Refreshed against the client id/secret above. When unset,
    /// the archive tier is off and everything stays on R2.
    pub const GOOGLE_REFRESH_TOKEN: &str = "CHUK_TRAIN_GOOGLE_REFRESH_TOKEN";
    /// Comma-separated allowlist of emails permitted to view the dashboard.
    /// (Legacy: with the RBAC users table this seeds nothing new; kept for the
    /// dashboard auth gate + as the sysadmin fallback when SYSADMIN_EMAIL unset.)
    pub const ALLOWED_EMAILS: &str = "CHUK_TRAIN_ALLOWED_EMAILS";
    /// Email seeded as the bootstrap sysadmin on startup (falls back to the
    /// first ALLOWED_EMAILS entry). Only seeded if the user doesn't yet exist.
    pub const SYSADMIN_EMAIL: &str = "CHUK_TRAIN_SYSADMIN_EMAIL";
    /// Path to the worker binary the mock provider launches as fake instances.
    pub const AGENT_BIN: &str = "CHUK_TRAIN_AGENT_BIN";
    /// Directory of per-target worker binaries the control plane serves for
    /// download (files named by target triple, e.g. `x86_64-unknown-linux-musl`).
    pub const AGENT_DIR: &str = "CHUK_TRAIN_AGENT_DIR";
    /// Minimum worker protocol version the control plane accepts; a worker below
    /// it is rejected (and a persistent one told to self-update). Defaults to the
    /// build's `PROTOCOL_VERSION`; raise it to force a fleet-wide worker update.
    pub const MIN_PROTOCOL: &str = "CHUK_TRAIN_MIN_PROTOCOL";
    /// chuk-experiments-server base URL (e.g. https://chuk-experiments-server.fly.dev).
    /// The reporting mirror (spec §11.6) is OFF unless this and the key are set.
    pub const EXPERIMENTS_URL: &str = "CHUK_EXPERIMENTS_URL";
    /// A WRITE-scoped experiments-server API key (raw bearer token, not `ck_`).
    pub const EXPERIMENTS_API_KEY: &str = "CHUK_EXPERIMENTS_API_KEY";
    /// Programme slug harness runs report under (default: gpu-training-harness).
    pub const EXPERIMENTS_PROGRAMME: &str = "CHUK_EXPERIMENTS_PROGRAMME";
    /// Experiment slug harness runs attach to (default: harness-runs).
    pub const EXPERIMENTS_EXPERIMENT: &str = "CHUK_EXPERIMENTS_EXPERIMENT";
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
