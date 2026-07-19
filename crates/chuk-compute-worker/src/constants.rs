//! Worker-local behavioural constants: the worker's own operating cadences and
//! exit sentinels. These govern worker behaviour rather than the message
//! contract, so they live here, not in the wire protocol crate.

use std::time::Duration;

/// How often a connected worker sends a heartbeat.
pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Worker reconnect backoff bounds (exponential, doubling).
pub const RECONNECT_BACKOFF_MIN: Duration = Duration::from_secs(2);
pub const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Minutes reserved at the end of a lease for the final checkpoint + upload
/// (T-drain).
pub const DEFAULT_DRAIN_WINDOW_MIN: f64 = 5.0;

/// How often the executor rescans the sandbox for freshly-appeared outputs to
/// upload while a job runs.
pub const OUTPUT_SCAN_INTERVAL: Duration = Duration::from_secs(3);

/// How often the worker samples host telemetry (GPU/CPU/memory) and streams it
/// as a `sys/*` metric. Fast enough to feel live on a dashboard, slow enough to
/// stay negligible against the heartbeat and job event traffic.
pub const SYS_SAMPLE_INTERVAL: Duration = Duration::from_secs(3);

/// Synthetic exit code recorded when the worker failed to spawn or supervise
/// the process at all.
pub const EXIT_CODE_AGENT_ERROR: i64 = -1;

/// Process exit code when the control plane rejects the handshake and there is
/// no self-update to apply (e.g. a leased worker on a version mismatch) — exit
/// rather than reconnect-loop against a version we cannot satisfy.
pub const EXIT_CODE_REJECTED: i32 = 3;
