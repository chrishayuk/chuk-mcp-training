//! Protocol constants. Everything a message shape depends on numerically lives
//! here, named — no magic numbers scattered across the type definitions.

/// Wire protocol version exchanged in the handshake. Bumped **only** for
/// breaking changes; additive changes rely on serde defaults instead (see the
/// crate docs). The handshake rejects a worker whose version is below the
/// control plane's minimum.
pub const PROTOCOL_VERSION: u32 = 1;

/// Default seconds between SIGTERM and SIGKILL when a job is stopped — the
/// window a workload has to flush state before it is force-killed. Overridable
/// per job ([`crate::Job::term_grace_secs`]).
pub const DEFAULT_TERM_GRACE_SECS: u64 = 30;

/// Default seconds between system-telemetry samples. Overridable by the control
/// plane per worker ([`crate::TelemetryConfig::interval_secs`]).
pub const DEFAULT_TELEMETRY_INTERVAL_SECS: u64 = 5;

/// Namespace prefix for worker-sampled system metrics (GPU/CPU/memory/…), so
/// they never collide with a workload's own metric keys and the control plane
/// can group them. E.g. `sys/gpu_util`, `sys/ram_bytes`.
pub const SYS_METRIC_PREFIX: &str = "sys/";

/// Placeholder the worker substitutes with a job's sandbox root wherever it
/// appears — in env values, command arguments, input destinations, output
/// globs, and the metrics file. It lets the control plane express
/// sandbox-relative paths without knowing the worker's filesystem, so a job
/// spec is portable across every worker that runs it.
pub const SANDBOX_PLACEHOLDER: &str = "${SANDBOX}";
