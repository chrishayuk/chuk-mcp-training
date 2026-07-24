//! Storage adapter seam: one async trait, pluggable backends.
//!
//! M0 ships SQLite (single Fly machine with a volume). The `postgres:` /
//! `postgresql:` scheme selects the Postgres (Neon) backend, which lets the
//! control plane's persistent state live off-box for a multi-machine deploy.
//! The `redis:` scheme is reserved for an M2+ backend. Note that moving the
//! store off-box does not by itself make the control plane stateless: live
//! agent websockets are in-process state, so running >1 machine also needs
//! sticky routing or pubsub fan-out.

mod ids;
mod postgres;
mod sqlite;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    ApiKeyInfo, Budget, CheckpointInfo, CheckpointLocation, CheckpointMeta, CodeRef, CodeUnitInfo,
    CodeUnitManifest, EventKind, GateAction, GateInfo, Hardware, Lease, LeaseExtension, LeaseState,
    LedgerEntry,
    MetricSeries, OutboxRow, Role, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary,
    UnixSeconds, User, WorkerId, WorkerInfo, WorkerTelemetry, WorkerTokenInfo,
};

pub use postgres::PgStore;
pub use sqlite::SqliteStore;

const SCHEME_SQLITE: &str = "sqlite:";
const SCHEME_POSTGRES: &str = "postgres:";
const SCHEME_POSTGRESQL: &str = "postgresql:";
const SCHEME_REDIS: &str = "redis:";

// Persistent control-plane state, split into cohesive per-domain traits.
// `Store` is the object-safe union every backend provides; callers hold an
// `Arc<dyn Store>` and every sub-trait method is callable on it (supertrait
// methods live in the trait object's vtable), so the split is caller-invisible.

/// Fleet membership, presence, run-binding, persistence, and host telemetry.
#[async_trait]
pub trait WorkerStore: Send + Sync {
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

    /// Whether this worker id is bound to a live persistent-worker token
    /// (chuk-compute M3.1) — i.e. its class is `Persistent`. Used by the
    /// heartbeat reaper to leave a persistent worker's run assigned.
    async fn worker_is_persistent(&self, id: &WorkerId) -> Result<bool>;

    /// Upsert a worker's latest host-telemetry sample (chuk-compute M4 `sys/*`),
    /// stamped now — one row per worker, overwriting the previous sample.
    async fn record_worker_samples(
        &self,
        worker_id: &WorkerId,
        values: &std::collections::BTreeMap<String, f64>,
    ) -> Result<()>;

    /// A worker's latest telemetry sample, or `None` if it never reported one.
    async fn worker_telemetry(&self, worker_id: &WorkerId) -> Result<Option<WorkerTelemetry>>;
}

/// Filter + page for [`RunStore::runs`]: optional state / external-parent
/// filters, and the page offset. `Default` means "everything, first page".
#[derive(Debug, Clone, Default)]
pub struct RunQuery {
    pub state: Option<RunState>,
    /// Match runs attached to this experiments-server logical run (`RUN-…`).
    pub experiment_ref: Option<String>,
    /// Match children of this sweep (`SWEEP-…`).
    pub sweep_id: Option<String>,
    pub offset: u32,
}

/// Run rows + the monotonic exec sequence, plus the experiments-server
/// mirror ids and the durable reporting outbox (retried mirror events).
#[async_trait]
pub trait RunStore: Send + Sync {
    /// Create a queued run. `experiment_ref` is the optional external parent —
    /// the experiments-server logical run (`RUN-…`) this execution realises.
    /// `created_by` is the submitting user's email (`AuthContext.owner_email`),
    /// or `None` for pre-attribution callers. `sweep_id` marks a sweep child.
    async fn create_run(
        &self,
        name: &str,
        spec: &RunSpec,
        experiment_ref: Option<&str>,
        created_by: Option<&str>,
        sweep_id: Option<&str>,
    ) -> Result<RunId>;

    /// Next value of the monotonic execution sequence (the 5-digit id tail).
    async fn next_run_seq(&self) -> Result<i64>;

    /// Persist the chuk-experiments-server run id this run is mirrored to
    /// (spec §11.6), so later lifecycle/artifact reports can address it.
    async fn set_experiments_run_id(&self, run_id: &RunId, ext_run_id: &str) -> Result<()>;

    /// The mirrored experiments-server run id, or `None` if not mirrored.
    async fn experiments_run_id(&self, run_id: &RunId) -> Result<Option<String>>;

    /// Persist a pending mirror event, due immediately. Returns its row id.
    async fn enqueue_outbox_event(
        &self,
        run_id: &RunId,
        kind: &str,
        payload: &str,
        at: UnixSeconds,
    ) -> Result<i64>;

    /// Undelivered events whose `next_attempt_at` has passed, oldest first —
    /// processing order matters: a run's later events (state/checkpoint/result)
    /// must not be retried ahead of its own not-yet-delivered `created` event.
    async fn due_outbox_events(&self, at: UnixSeconds, limit: i64) -> Result<Vec<OutboxRow>>;

    /// Mark an event delivered; it's excluded from future sweeps.
    async fn mark_outbox_event_done(&self, id: i64) -> Result<()>;

    /// Record a failed attempt and reschedule it (caller computes backoff).
    async fn mark_outbox_event_failed(
        &self,
        id: i64,
        error: &str,
        next_attempt_at: UnixSeconds,
    ) -> Result<()>;

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

    /// Newest-first page of runs matching `query` (id-tiebroken so paging is
    /// stable across equal timestamps).
    async fn runs(&self, query: &RunQuery, limit: u32) -> Result<Vec<RunSummary>>;
}

/// A run's streamed logs and its append-only lifecycle event log.
#[async_trait]
pub trait RunLogStore: Send + Sync {
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

/// Deployable code units (spec §11.1).
#[async_trait]
pub trait CodeUnitStore: Send + Sync {
    async fn register_code_unit(
        &self,
        code: &CodeRef,
        manifest: &CodeUnitManifest,
        uri: &str,
    ) -> Result<()>;

    async fn code_unit(&self, name: &str, sha: &str) -> Result<Option<CodeUnitInfo>>;
}

/// Training metric ingest + series read (spec §5.1 → §6).
#[async_trait]
pub trait MetricStore: Send + Sync {
    async fn append_metrics(
        &self,
        run_id: &RunId,
        step: u64,
        values: &std::collections::BTreeMap<String, f64>,
    ) -> Result<()>;

    async fn metric_series(
        &self,
        run_id: &RunId,
        keys: Option<&[String]>,
        since_step: u64,
        downsample: u32,
    ) -> Result<MetricSeries>;
}

/// Lineage-complete checkpoints: record, list, pin, and archive tiering (§11.2).
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn record_checkpoint(
        &self,
        run_id: &RunId,
        step: u64,
        uri: &str,
        model_hash: &str,
        meta: &CheckpointMeta,
    ) -> Result<()>;

    async fn checkpoints(&self, run_id: &RunId) -> Result<Vec<CheckpointInfo>>;

    async fn latest_checkpoint(&self, run_id: &RunId) -> Result<Option<CheckpointInfo>>;

    /// Pin a checkpoint by step; returns false if no such checkpoint exists.
    async fn pin_checkpoint(&self, run_id: &RunId, step: u64, name: &str) -> Result<bool>;

    /// Update a checkpoint's recorded location (e.g. hot → final on promotion).
    async fn set_checkpoint_location(
        &self,
        run_id: &RunId,
        step: u64,
        location: CheckpointLocation,
    ) -> Result<()>;

    /// Mark a checkpoint archived to Drive: location = drive, the per-file Drive
    /// ids (filename → id), and the archive timestamp.
    async fn mark_checkpoint_archived(
        &self,
        run_id: &RunId,
        step: u64,
        drive_file_ids: &std::collections::BTreeMap<String, String>,
        archived_at: UnixSeconds,
    ) -> Result<()>;

    /// The Drive file ids (filename → id) recorded when a checkpoint was
    /// archived, or `None` if it has no Drive copy. Used by the retrieval
    /// resolver to serve an archived checkpoint from Drive.
    async fn checkpoint_drive_ids(
        &self,
        run_id: &RunId,
        step: u64,
    ) -> Result<Option<std::collections::BTreeMap<String, String>>>;
}

/// One sweep's stored record (spec §5.2): the template + axes it fans out,
/// with `template`/`axes` kept as their JSON text.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepRow {
    pub id: String,
    pub name: String,
    pub template: String,
    pub axes: String,
    pub concurrency: u32,
    pub created_at: UnixSeconds,
    pub created_by: Option<String>,
}

/// Sweeps: the fan-out record and the child-concurrency count the scheduler
/// caps against.
#[async_trait]
pub trait SweepStore: Send + Sync {
    /// Create a sweep row; the store mints its `SWEEP-…` id.
    async fn create_sweep(
        &self,
        name: &str,
        template: &str,
        axes: &str,
        concurrency: u32,
        created_by: Option<&str>,
    ) -> Result<String>;

    async fn sweep(&self, sweep_id: &str) -> Result<Option<SweepRow>>;

    /// How many of this sweep's children are currently assigned or running.
    async fn sweep_active_children(&self, sweep_id: &str) -> Result<u32>;
}

/// A timestamped metric observation for gate evaluation (ascending by step).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricObservation {
    pub ts: UnixSeconds,
    pub value: f64,
}

/// Gates + watchdogs (spec §6/§8): registration, latest verdicts, and the
/// per-key metric history gate evaluation reads.
#[async_trait]
pub trait GateStore: Send + Sync {
    /// Upsert a gate by (scope, scope_id, name); a re-register replaces the
    /// expression/action and clears the previous verdict.
    async fn register_gate(
        &self,
        scope: &str,
        scope_id: &str,
        name: &str,
        expr: &str,
        action: GateAction,
    ) -> Result<()>;

    async fn gates(&self, scope: &str, scope_id: &str) -> Result<Vec<GateInfo>>;

    /// Persist a gate's latest verdict.
    #[allow(clippy::too_many_arguments)]
    async fn record_gate_result(
        &self,
        scope: &str,
        scope_id: &str,
        name: &str,
        tripped: bool,
        last_value: Option<f64>,
        detail: &str,
        at: UnixSeconds,
    ) -> Result<()>;

    /// One key's full (ts, value) history for a run, ascending by step.
    async fn metric_history(&self, run_id: &RunId, key: &str)
        -> Result<Vec<MetricObservation>>;
}

/// Provider leases: the wall the lease manager + reconcile loop enforce (§3).
#[async_trait]
pub trait LeaseStore: Send + Sync {
    async fn create_lease(&self, lease: &Lease) -> Result<()>;

    async fn lease(&self, worker_id: &WorkerId) -> Result<Option<Lease>>;

    /// Leases not yet destroyed (active or draining) — what the lease manager
    /// and reconcile loop iterate.
    async fn live_leases(&self) -> Result<Vec<Lease>>;

    async fn set_lease_state(&self, worker_id: &WorkerId, state: LeaseState) -> Result<()>;

    /// Append an extension and return the updated lease (None if no lease).
    async fn extend_lease(
        &self,
        worker_id: &WorkerId,
        ext: LeaseExtension,
    ) -> Result<Option<Lease>>;
}

/// The spend ledger — the source of truth for realised cost (§8) — plus the
/// budget caps provision/extend refuse against.
#[async_trait]
pub trait LedgerStore: Send + Sync {
    async fn ledger_append(&self, entry: &LedgerEntry) -> Result<()>;

    async fn ledger_entries(&self) -> Result<Vec<LedgerEntry>>;

    /// Upsert a budget by scope (`global` / `provider:<name>`).
    async fn set_budget(&self, budget: &Budget) -> Result<()>;

    /// Remove a budget; returns false if no such scope was set.
    async fn delete_budget(&self, scope: &str) -> Result<bool>;

    async fn budgets(&self) -> Result<Vec<Budget>>;
}

/// RBAC: teams, users (+ their linked experiments key), and hashed API keys.
#[async_trait]
pub trait AuthStore: Send + Sync {
    /// Create the team if absent (idempotent); never downgrades an existing name.
    async fn ensure_team(&self, id: &str, name: &str) -> Result<()>;

    /// Create or update a user's team + role.
    async fn upsert_user(&self, email: &str, team_id: &str, role: Role) -> Result<()>;

    async fn get_user(&self, email: &str) -> Result<Option<User>>;

    async fn list_users(&self, team_id: &str) -> Result<Vec<User>>;

    async fn remove_user(&self, email: &str) -> Result<()>;

    /// Link (or clear, with `None`) this user's own chuk-experiments-server API
    /// key so their mirrored runs report under their own identity instead of the
    /// shared default. Stored as an opaque encrypted blob — this layer never
    /// sees or needs the plaintext key.
    async fn set_user_experiments_key(&self, email: &str, encrypted: Option<&str>) -> Result<()>;

    /// The linked key (still encrypted), or `None` if this user hasn't linked
    /// one.
    async fn user_experiments_key(&self, email: &str) -> Result<Option<String>>;

    #[allow(clippy::too_many_arguments)]
    async fn create_api_key(
        &self,
        id: &str,
        team_id: &str,
        created_by: &str,
        name: &str,
        prefix: &str,
        key_hash: &str,
        role: Role,
    ) -> Result<()>;

    async fn list_api_keys(&self, team_id: &str) -> Result<Vec<ApiKeyInfo>>;

    /// Revoke a key by id; returns false if there was no such live key.
    async fn revoke_api_key(&self, id: &str) -> Result<bool>;

    /// Resolve a bearer key by its sha256 hash to its (non-revoked) info.
    async fn resolve_api_key(&self, key_hash: &str) -> Result<Option<ApiKeyInfo>>;

    async fn touch_api_key(&self, id: &str, at: UnixSeconds) -> Result<()>;
}

/// Persistent-worker tokens (chuk-compute M3.1): hashed, bound to a stable id.
#[async_trait]
pub trait WorkerTokenStore: Send + Sync {
    async fn create_worker_token(
        &self,
        id: &str,
        worker_id: &WorkerId,
        name: &str,
        prefix: &str,
        token_hash: &str,
    ) -> Result<()>;

    /// Resolve a worker token by its sha256 hash to its (non-revoked) info.
    async fn resolve_worker_token(&self, token_hash: &str) -> Result<Option<WorkerTokenInfo>>;

    /// All worker tokens, newest first.
    async fn list_worker_tokens(&self) -> Result<Vec<WorkerTokenInfo>>;

    /// Revoke a token by id; returns false if there was no such live token.
    async fn revoke_worker_token(&self, id: &str) -> Result<bool>;

    async fn touch_worker_token(&self, id: &str, at: UnixSeconds) -> Result<()>;

    /// Mint a single-use provision join token (spec §12), bound to the
    /// provisioned worker id, hashed at rest, expiring for first use.
    async fn create_join_token(
        &self,
        id: &str,
        worker_id: &WorkerId,
        token_hash: &str,
        expires_at: UnixSeconds,
    ) -> Result<()>;

    /// Resolve a presented join token. An unused, unexpired token is
    /// atomically consumed → `(bound_worker_id, first_use = true)`. An
    /// already-consumed token → `(bound_worker_id, false)`; the ws layer
    /// admits that only as a reconnect of the same bound id. Unknown, or
    /// unused-but-expired → `None`.
    async fn resolve_join_token(
        &self,
        token_hash: &str,
        at: UnixSeconds,
    ) -> Result<Option<(WorkerId, bool)>>;
}

/// The full store surface — the union of every domain trait.
pub trait Store:
    WorkerStore
    + RunStore
    + RunLogStore
    + CodeUnitStore
    + MetricStore
    + CheckpointStore
    + GateStore
    + SweepStore
    + LeaseStore
    + LedgerStore
    + AuthStore
    + WorkerTokenStore
    + Send
    + Sync
{}
impl<T> Store for T where
    T: WorkerStore
    + RunStore
    + RunLogStore
    + CodeUnitStore
    + MetricStore
    + CheckpointStore
    + GateStore
    + SweepStore
    + LeaseStore
    + LedgerStore
    + AuthStore
    + WorkerTokenStore + Send + Sync
{}

/// Every store domain trait in one glob — for an adapter method calling another
/// domain's method on `self` (a concrete type needs the trait in scope):
/// `use crate::store::prelude::*;`.
pub(crate) mod prelude {
    pub(crate) use super::{
        AuthStore, CheckpointStore, CodeUnitStore, GateStore, LeaseStore, LedgerStore, MetricStore,
        RunLogStore, RunStore, SweepStore, WorkerStore, WorkerTokenStore,
    };
}

/// Open a store from a URL-ish spec: `postgres:`/`postgresql:` → Postgres
/// (Neon), `sqlite:path.db` (a bare path also means SQLite) → SQLite,
/// `redis:...` reserved.
pub async fn open_store(spec: &str) -> Result<Box<dyn Store>> {
    if spec.starts_with(SCHEME_POSTGRES) || spec.starts_with(SCHEME_POSTGRESQL) {
        // Pass the URL through unchanged: the driver needs the scheme, and Neon
        // needs the credentials and `?sslmode=require` in the query string.
        return Ok(Box::new(PgStore::open(spec).await?));
    }
    if let Some(path) = spec.strip_prefix(SCHEME_SQLITE) {
        return Ok(Box::new(SqliteStore::open(path).await?));
    }
    if spec.starts_with(SCHEME_REDIS) {
        anyhow::bail!("redis store backend is reserved for M2+; use sqlite or postgres for now");
    }
    Ok(Box::new(SqliteStore::open(spec).await?))
}
