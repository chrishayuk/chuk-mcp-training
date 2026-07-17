//! Storage adapter seam: one async trait, pluggable backends.
//!
//! M0 ships SQLite (single Fly machine with a volume). The `redis:` scheme is
//! reserved for an M2+ backend that would let the control plane's persistent
//! state live off-box. Note that a Redis store alone does not make the
//! control plane stateless: live agent websockets are in-process state, so
//! running >1 machine also needs sticky routing or pubsub fan-out.

mod sqlite;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    EventKind, Hardware, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary, WorkerId,
    WorkerInfo,
};

pub use sqlite::SqliteStore;

const SCHEME_SQLITE: &str = "sqlite:";
const SCHEME_REDIS: &str = "redis:";

/// Persistent control-plane state. Every backend must provide exactly this.
#[async_trait]
pub trait Store: Send + Sync {
    // workers
    async fn worker_joined(
        &self,
        id: &WorkerId,
        labels: &[String],
        hardware: &Hardware,
    ) -> Result<()>;
    async fn worker_seen(&self, id: &WorkerId) -> Result<()>;
    async fn worker_left(&self, id: &WorkerId) -> Result<()>;
    async fn set_worker_run(&self, id: &WorkerId, run: Option<&RunId>) -> Result<()>;
    async fn worker(&self, id: &WorkerId) -> Result<Option<WorkerInfo>>;
    async fn fleet(&self) -> Result<Vec<WorkerInfo>>;

    // runs
    async fn create_run(&self, name: &str, spec: &RunSpec) -> Result<RunId>;
    async fn transition(
        &self,
        run_id: &RunId,
        state: RunState,
        worker_id: Option<&WorkerId>,
        exit_code: Option<i64>,
        detail: serde_json::Value,
    ) -> Result<()>;
    async fn next_queued(&self) -> Result<Option<RunRecord>>;
    async fn run(&self, run_id: &RunId) -> Result<Option<RunRecord>>;
    async fn runs(&self, limit: u32) -> Result<Vec<RunSummary>>;

    // logs & events
    async fn append_log(&self, run_id: &RunId, line: &str) -> Result<()>;
    async fn tail_logs(&self, run_id: &RunId, lines: u32) -> Result<Vec<String>>;
    async fn add_event(
        &self,
        run_id: &RunId,
        event: EventKind,
        detail: serde_json::Value,
    ) -> Result<()>;
    async fn events(&self, run_id: &RunId) -> Result<Vec<RunEvent>>;
}

/// Open a store from a URL-ish spec: `sqlite:path.db` (a bare path also means
/// SQLite), `redis:...` reserved.
pub async fn open_store(spec: &str) -> Result<Box<dyn Store>> {
    if let Some(path) = spec.strip_prefix(SCHEME_SQLITE) {
        return Ok(Box::new(SqliteStore::open(path).await?));
    }
    if spec.starts_with(SCHEME_REDIS) {
        anyhow::bail!("redis store backend is reserved for M2+; use sqlite for now");
    }
    Ok(Box::new(SqliteStore::open(spec).await?))
}
