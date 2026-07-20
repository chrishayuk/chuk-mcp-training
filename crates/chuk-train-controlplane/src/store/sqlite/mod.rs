//! SQLite adapter (sqlx, async native).

use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    ApiKeyInfo, CheckpointInfo, CheckpointLocation, CheckpointMeta, CodeRef, CodeUnitInfo,
    CodeUnitManifest, DEFAULT_TEAM_ID, EventKind, Hardware, Lease, LeaseExtension, LeaseState,
    LedgerEntry, MetricPoint, MetricSeries, OutboxRow, Role, RunEvent, RunId, RunRecord, RunSpec,
    RunState, RunSummary, UnixSeconds, User, WorkerId, WorkerInfo, WorkerState, WorkerTelemetry,
    WorkerTokenInfo, WORKER_SAMPLE_RETENTION,
};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use super::ids::{
    downsample_in_place, enum_from_string, enum_to_string, merge_field, new_run_id, now,
    worker_telemetry_from_samples,
};

/// Counter row name backing the monotonic execution sequence (the 5-digit tail
/// of our `EXEC-…` ids).
const RUN_SEQ_COUNTER: &str = "run_seq";

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS workers (
  id           TEXT PRIMARY KEY,
  labels       TEXT NOT NULL DEFAULT '[]',
  hardware     TEXT NOT NULL DEFAULT '{}',
  connected    INTEGER NOT NULL DEFAULT 1,
  current_run  TEXT,
  joined_at    REAL NOT NULL,
  last_seen    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS worker_samples (
  worker_id    TEXT NOT NULL,
  ts           REAL NOT NULL,
  payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_worker_samples ON worker_samples (worker_id, ts);
CREATE TABLE IF NOT EXISTS runs (
  id             TEXT PRIMARY KEY,
  name           TEXT NOT NULL,
  kind           TEXT NOT NULL,
  spec           TEXT NOT NULL,
  state          TEXT NOT NULL,
  worker_id      TEXT,
  exit_code      INTEGER,
  experiment_ref TEXT,
  created_by     TEXT,
  created_at     REAL NOT NULL,
  updated_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_logs (
  run_id       TEXT NOT NULL,
  n            INTEGER NOT NULL,
  ts           REAL NOT NULL,
  line         TEXT NOT NULL,
  PRIMARY KEY (run_id, n)
);
CREATE TABLE IF NOT EXISTS run_events (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id       TEXT NOT NULL,
  ts           REAL NOT NULL,
  event        TEXT NOT NULL,
  detail       TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS code_units (
  name         TEXT NOT NULL,
  sha          TEXT NOT NULL,
  manifest     TEXT NOT NULL,
  uri          TEXT NOT NULL,
  created_at   REAL NOT NULL,
  PRIMARY KEY (name, sha)
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id       TEXT NOT NULL,
  step         INTEGER NOT NULL,
  key          TEXT NOT NULL,
  value        REAL NOT NULL,
  ts           REAL NOT NULL,
  PRIMARY KEY (run_id, key, step)
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id         TEXT NOT NULL,
  step           INTEGER NOT NULL,
  uri            TEXT NOT NULL,
  model_hash     TEXT NOT NULL,
  meta           TEXT NOT NULL,
  pinned         INTEGER NOT NULL DEFAULT 0,
  pin_name       TEXT,
  created_at     REAL NOT NULL,
  location       TEXT NOT NULL DEFAULT 'r2_hot',
  drive_file_ids TEXT,
  archived_at    REAL,
  PRIMARY KEY (run_id, step)
);
CREATE TABLE IF NOT EXISTS leases (
  worker_id        TEXT PRIMARY KEY,
  provider         TEXT NOT NULL,
  instance_id      TEXT NOT NULL,
  price_hr         REAL NOT NULL,
  granted_min      REAL NOT NULL,
  drain_window_min REAL NOT NULL,
  started_at       REAL NOT NULL,
  state            TEXT NOT NULL,
  extensions       TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS ledger (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           REAL NOT NULL,
  worker_id    TEXT NOT NULL,
  provider     TEXT NOT NULL,
  event        TEXT NOT NULL,
  minutes      REAL NOT NULL,
  cost         REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS teams (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
  email                          TEXT PRIMARY KEY,
  team_id                        TEXT NOT NULL,
  role                           TEXT NOT NULL,
  created_at                     REAL NOT NULL,
  experiments_api_key_encrypted  TEXT
);
CREATE TABLE IF NOT EXISTS api_keys (
  id           TEXT PRIMARY KEY,
  team_id      TEXT NOT NULL,
  created_by   TEXT NOT NULL,
  name         TEXT NOT NULL,
  prefix       TEXT NOT NULL,
  key_hash     TEXT NOT NULL,
  role         TEXT NOT NULL,
  created_at   REAL NOT NULL,
  last_used_at REAL,
  revoked_at   REAL
);
CREATE TABLE IF NOT EXISTS worker_tokens (
  id           TEXT PRIMARY KEY,
  worker_id    TEXT NOT NULL,
  name         TEXT NOT NULL,
  prefix       TEXT NOT NULL,
  token_hash   TEXT NOT NULL,
  created_at   REAL NOT NULL,
  last_used_at REAL,
  revoked_at   REAL
);
CREATE TABLE IF NOT EXISTS counters (
  name         TEXT PRIMARY KEY,
  value        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments_outbox (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT NOT NULL,
  kind             TEXT NOT NULL,
  payload          TEXT NOT NULL,
  attempts         INTEGER NOT NULL DEFAULT 0,
  last_error       TEXT,
  created_at       REAL NOT NULL,
  next_attempt_at  REAL NOT NULL,
  completed_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_worker_tokens_hash ON worker_tokens (token_hash);
CREATE INDEX IF NOT EXISTS idx_runs_state   ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run   ON run_events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics (run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_ckpt_run     ON checkpoints (run_id, step);
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases (state);
CREATE INDEX IF NOT EXISTS idx_outbox_due   ON experiments_outbox (next_attempt_at) WHERE completed_at IS NULL;
"#;

pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    pub async fn open(path: &str) -> Result<Self> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);
        // One writer connection sidesteps SQLite writer contention; WAL keeps
        // readers unblocked. Plenty for M0's event rates.
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        // Additive migrations for DBs created before a column existed. SQLite
        // has no ADD COLUMN IF NOT EXISTS, so run each and ignore the
        // duplicate-column error a fresh (already-columned) db raises.
        for stmt in [
            "ALTER TABLE checkpoints ADD COLUMN location TEXT NOT NULL DEFAULT 'r2_hot'",
            "ALTER TABLE checkpoints ADD COLUMN drive_file_ids TEXT",
            "ALTER TABLE checkpoints ADD COLUMN archived_at REAL",
            "ALTER TABLE runs ADD COLUMN experiments_run_id TEXT",
            "ALTER TABLE runs ADD COLUMN experiment_ref TEXT",
            "ALTER TABLE runs ADD COLUMN created_by TEXT",
            "ALTER TABLE users ADD COLUMN experiments_api_key_encrypted TEXT",
        ] {
            let _ = sqlx::query(stmt).execute(&pool).await;
        }
        Ok(Self { pool })
    }
}

mod workers;
mod runs;
mod logs;
mod codeunits;
mod metrics;
mod checkpoints;
mod leases;
mod ledger;
mod auth;
mod tokens;


fn api_key_from_row(row: sqlx::sqlite::SqliteRow) -> Result<ApiKeyInfo> {
    Ok(ApiKeyInfo {
        id: row.get("id"),
        team_id: row.get("team_id"),
        created_by: row.get("created_by"),
        name: row.get("name"),
        prefix: row.get("prefix"),
        role: enum_from_string(row.get::<String, _>("role"))?,
        created_at: row.get("created_at"),
        last_used_at: row.get("last_used_at"),
        revoked_at: row.get("revoked_at"),
    })
}

fn worker_token_from_row(row: sqlx::sqlite::SqliteRow) -> Result<WorkerTokenInfo> {
    Ok(WorkerTokenInfo {
        id: row.get("id"),
        worker_id: WorkerId(row.get("worker_id")),
        name: row.get("name"),
        prefix: row.get("prefix"),
        created_at: row.get("created_at"),
        last_used_at: row.get("last_used_at"),
        revoked_at: row.get("revoked_at"),
    })
}

fn user_from_row(row: sqlx::sqlite::SqliteRow) -> Result<User> {
    Ok(User {
        email: row.get("email"),
        team_id: row.get("team_id"),
        role: enum_from_string(row.get::<String, _>("role"))?,
        created_at: row.get("created_at"),
    })
}

fn lease_from_row(row: sqlx::sqlite::SqliteRow) -> Result<Lease> {
    Ok(Lease {
        worker_id: WorkerId(row.get("worker_id")),
        provider: row.get("provider"),
        instance_id: row.get("instance_id"),
        price_hr: row.get("price_hr"),
        granted_min: row.get("granted_min"),
        drain_window_min: row.get("drain_window_min"),
        started_at: row.get("started_at"),
        state: enum_from_string(row.get::<String, _>("state"))?,
        extensions: serde_json::from_str(&row.get::<String, _>("extensions"))?,
    })
}

fn checkpoint_from_row(row: sqlx::sqlite::SqliteRow) -> Result<CheckpointInfo> {
    Ok(CheckpointInfo {
        run_id: RunId(row.get("run_id")),
        step: row.get::<i64, _>("step") as u64,
        uri: row.get("uri"),
        model_hash: row.get("model_hash"),
        pinned: row.get::<i64, _>("pinned") == 1,
        pin_name: row.get("pin_name"),
        meta: serde_json::from_str(&row.get::<String, _>("meta"))?,
        created_at: row.get("created_at"),
        location: enum_from_string(row.get::<String, _>("location"))?,
        archived_at: row.get("archived_at"),
    })
}

fn worker_from_row(row: sqlx::sqlite::SqliteRow) -> Result<WorkerInfo> {
    let last_seen: f64 = row.get("last_seen");
    Ok(WorkerInfo {
        id: WorkerId(row.get("id")),
        labels: serde_json::from_str(&row.get::<String, _>("labels"))?,
        hardware: serde_json::from_str(&row.get::<String, _>("hardware"))?,
        state: if row.get::<i64, _>("connected") == 1 {
            WorkerState::Connected
        } else {
            WorkerState::Disconnected
        },
        current_run: row.get::<Option<String>, _>("current_run").map(RunId),
        joined_at: row.get("joined_at"),
        last_seen,
        heartbeat_age_s: ((now() - last_seen) * 10.0).round() / 10.0,
        // Populated by worker()/fleet() from the leases + telemetry tables.
        lease: None,
        telemetry: None,
    })
}

fn run_from_row(row: sqlx::sqlite::SqliteRow) -> Result<RunRecord> {
    let spec: RunSpec = serde_json::from_str(&row.get::<String, _>("spec"))?;
    Ok(RunRecord {
        summary: RunSummary {
            id: RunId(row.get("id")),
            name: row.get("name"),
            kind: row.get("kind"),
            state: enum_from_string(row.get::<String, _>("state"))?,
            worker_id: row.get::<Option<String>, _>("worker_id").map(WorkerId),
            exit_code: row.get("exit_code"),
            experiment_ref: row.get("experiment_ref"),
            created_by: row.get("created_by"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        },
        spec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::prelude::*;
    use chuk_train_proto::ShellSpec;

    async fn mem_store() -> SqliteStore {
        // A private in-memory db; the single pooled connection keeps it alive
        // for the whole test.
        SqliteStore::open(":memory:").await.expect("open :memory:")
    }

    fn shell_spec() -> RunSpec {
        RunSpec::Shell(ShellSpec { command: "echo hi".into(), timeout_s: 60 })
    }

    #[tokio::test]
    async fn our_ids_use_the_exec_prefix() {
        let store = mem_store().await;
        let id = store.create_run("r", &shell_spec(), None, None).await.expect("create");
        assert!(id.0.starts_with("EXEC-"), "{}", id.0);
    }

    #[tokio::test]
    async fn experiment_ref_round_trips_the_external_parent() {
        let store = mem_store().await;
        let attached = store
            .create_run("attached", &shell_spec(), Some("RUN-20260718-160217-00042"), None)
            .await
            .expect("create attached");
        let scratch = store
            .create_run("scratch", &shell_spec(), None, None)
            .await
            .expect("create scratch");

        let attached = store.run(&attached).await.expect("run").expect("some");
        assert_eq!(
            attached.summary.experiment_ref.as_deref(),
            Some("RUN-20260718-160217-00042"),
        );
        let scratch = store.run(&scratch).await.expect("run").expect("some");
        assert_eq!(scratch.summary.experiment_ref, None);
    }

    #[tokio::test]
    async fn worker_is_persistent_tracks_a_live_bound_token() {
        let store = mem_store().await;
        let persist = WorkerId("mac-01".into());
        // No token yet → not persistent.
        assert!(!store.worker_is_persistent(&persist).await.expect("query"));
        store
            .create_worker_token("tok-1", &persist, "mac", "cw_abcd1234", "hash")
            .await
            .expect("create token");
        assert!(store.worker_is_persistent(&persist).await.expect("query"));
        // An unrelated worker id stays non-persistent.
        assert!(!store
            .worker_is_persistent(&WorkerId("ephemeral-9".into()))
            .await
            .expect("query"));
        // Revoking the token drops the worker back to ephemeral.
        assert!(store.revoke_worker_token("tok-1").await.expect("revoke"));
        assert!(!store.worker_is_persistent(&persist).await.expect("query"));
    }

    #[tokio::test]
    async fn worker_telemetry_keeps_history_and_latest() {
        let store = mem_store().await;
        let w = WorkerId("gpu-1".into());
        store.worker_joined(&w, &[], &Hardware::default()).await.expect("join");
        // Nothing reported yet.
        assert!(store.worker_telemetry(&w).await.expect("query").is_none());

        store
            .record_worker_samples(&w, &BTreeMap::from([("sys/gpu_util".to_owned(), 0.5)]))
            .await
            .expect("record");
        store
            .record_worker_samples(
                &w,
                &BTreeMap::from([
                    ("sys/gpu_util".to_owned(), 0.9),
                    ("sys/cpu_util".to_owned(), 0.2),
                ]),
            )
            .await
            .expect("record");

        let t = store.worker_telemetry(&w).await.expect("query").expect("some");
        // Gauges read the newest sample.
        assert_eq!(t.values["sys/gpu_util"], 0.9);
        assert_eq!(t.values["sys/cpu_util"], 0.2);
        // Sparkline history retains both samples.
        assert_eq!(t.series["sys/gpu_util"].len(), 2);

        // The fleet view carries each worker's latest values inline (no N+1 fetch).
        let fleet = store.fleet().await.expect("fleet");
        let me = fleet.iter().find(|x| x.id == w).expect("worker in fleet");
        assert_eq!(me.telemetry.as_ref().expect("telemetry")["sys/gpu_util"], 0.9);
    }

    #[tokio::test]
    async fn outbox_event_becomes_due_then_done() {
        let store = mem_store().await;
        let run_id = store.create_run("r", &shell_spec(), None, None).await.expect("create");
        let at = now();
        let id = store
            .enqueue_outbox_event(&run_id, "state", "{}", at)
            .await
            .expect("enqueue");

        let due = store.due_outbox_events(at, 10).await.expect("due");
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, id);
        assert_eq!(due[0].run_id, run_id);
        assert_eq!(due[0].kind, "state");
        assert_eq!(due[0].attempts, 0);

        store.mark_outbox_event_done(id).await.expect("mark done");
        let due = store.due_outbox_events(at, 10).await.expect("due after done");
        assert!(due.is_empty(), "a completed event must not be retried");
    }

    #[tokio::test]
    async fn failed_outbox_event_waits_for_its_scheduled_retry_time() {
        let store = mem_store().await;
        let run_id = store.create_run("r", &shell_spec(), None, None).await.expect("create");
        let at = now();
        let id = store
            .enqueue_outbox_event(&run_id, "state", "{}", at)
            .await
            .expect("enqueue");

        store
            .mark_outbox_event_failed(id, "boom", at + 3_600.0)
            .await
            .expect("mark failed");

        let due = store.due_outbox_events(at, 10).await.expect("due before backoff elapses");
        assert!(due.is_empty(), "must not retry before next_attempt_at");

        let due = store
            .due_outbox_events(at + 3_600.0, 10)
            .await
            .expect("due once backoff elapses");
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].attempts, 1, "a failed attempt must be recorded");
    }

    #[tokio::test]
    async fn create_run_persists_the_submitting_user() {
        let store = mem_store().await;
        let run_id = store
            .create_run("r", &shell_spec(), None, Some("chris@example.com"))
            .await
            .expect("create");
        let run = store.run(&run_id).await.expect("run").expect("some");
        assert_eq!(run.summary.created_by.as_deref(), Some("chris@example.com"));
    }

    #[tokio::test]
    async fn create_run_without_a_submitter_leaves_created_by_unset() {
        let store = mem_store().await;
        let run_id = store.create_run("r", &shell_spec(), None, None).await.expect("create");
        let run = store.run(&run_id).await.expect("run").expect("some");
        assert_eq!(run.summary.created_by, None);
    }

    #[tokio::test]
    async fn user_experiments_key_round_trips_and_clears() {
        let store = mem_store().await;
        store
            .upsert_user("chris@example.com", "default", Role::Write)
            .await
            .expect("upsert");

        assert_eq!(
            store.user_experiments_key("chris@example.com").await.expect("get"),
            None,
            "unset by default"
        );

        store
            .set_user_experiments_key("chris@example.com", Some("encrypted-blob"))
            .await
            .expect("set");
        assert_eq!(
            store.user_experiments_key("chris@example.com").await.expect("get"),
            Some("encrypted-blob".to_owned()),
        );

        store
            .set_user_experiments_key("chris@example.com", None)
            .await
            .expect("clear");
        assert_eq!(store.user_experiments_key("chris@example.com").await.expect("get"), None);
    }

    #[tokio::test]
    async fn set_user_experiments_key_works_for_a_user_with_no_prior_row() {
        // A first-time session sign-in resolves to Role::Read without ever
        // being `upsert_user`'d (see `resolve_auth`) — linking a key must
        // still work rather than silently no-op against a missing row.
        let store = mem_store().await;
        store
            .set_user_experiments_key("new@example.com", Some("encrypted-blob"))
            .await
            .expect("set");
        assert_eq!(
            store.user_experiments_key("new@example.com").await.expect("get"),
            Some("encrypted-blob".to_owned()),
        );
        let user = store.get_user("new@example.com").await.expect("get_user").expect("some");
        assert_eq!(user.role, Role::Read);
        assert_eq!(user.team_id, "default");
    }

    #[tokio::test]
    async fn set_user_experiments_key_does_not_clobber_an_existing_users_role() {
        let store = mem_store().await;
        store
            .upsert_user("admin@example.com", "default", Role::Admin)
            .await
            .expect("upsert");
        store
            .set_user_experiments_key("admin@example.com", Some("encrypted-blob"))
            .await
            .expect("set");
        let user = store.get_user("admin@example.com").await.expect("get_user").expect("some");
        assert_eq!(user.role, Role::Admin, "linking a key must not downgrade an existing role");
    }
}
