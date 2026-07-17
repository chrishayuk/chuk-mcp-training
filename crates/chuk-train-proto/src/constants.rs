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

/// Default number of log lines returned by the tail endpoint.
pub const DEFAULT_LOG_TAIL_LINES: u32 = 100;
/// Default page size for run listings.
pub const DEFAULT_RUN_LIST_LIMIT: u32 = 50;

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
}
