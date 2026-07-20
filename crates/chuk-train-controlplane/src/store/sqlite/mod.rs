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
    use std::collections::BTreeMap;

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

    // ---- coverage: the remaining domains, exercised end-to-end on :memory: ----

    fn a_lease(w: &str) -> Lease {
        Lease {
            worker_id: WorkerId(w.into()), provider: "vast".into(), instance_id: "i-1".into(),
            price_hr: 0.5, granted_min: 60.0, drain_window_min: 5.0, started_at: 1000.0,
            state: LeaseState::Active, extensions: vec![],
        }
    }

    #[tokio::test]
    async fn code_unit_registers_and_reads_back() {
        let store = mem_store().await;
        let code = CodeRef { name: "gpt-nano".into(), sha: "abc123".into() };
        let manifest = CodeUnitManifest {
            name: "gpt-nano".into(), version: "0.1".into(),
            entrypoints: BTreeMap::from([("train".to_owned(), "python train.py".to_owned())]),
            python: Some("3.11".into()), requires: Default::default(),
        };
        assert!(store.code_unit("gpt-nano", "abc123").await.expect("q").is_none());
        store.register_code_unit(&code, &manifest, "s3://unit.tar.zst").await.expect("register");
        let got = store.code_unit("gpt-nano", "abc123").await.expect("q").expect("some");
        assert_eq!(got.uri, "s3://unit.tar.zst");
        assert_eq!(got.manifest.entrypoints["train"], "python train.py");
        assert!(store.code_unit("gpt-nano", "wrong").await.expect("q").is_none());
    }

    #[tokio::test]
    async fn lease_lifecycle_create_extend_drain_destroy() {
        let store = mem_store().await;
        let w = WorkerId("w1".into());
        assert!(store.lease(&w).await.expect("q").is_none());
        store.create_lease(&a_lease("w1")).await.expect("create");
        assert_eq!(store.lease(&w).await.expect("q").expect("some").provider, "vast");
        assert_eq!(store.live_leases().await.expect("live").len(), 1);
        let ext = store
            .extend_lease(&w, LeaseExtension { minutes: 30.0, at: 2000.0, reason: "more".into() })
            .await.expect("extend").expect("some");
        assert_eq!(ext.extensions.len(), 1);
        store.set_lease_state(&w, LeaseState::Draining).await.expect("drain");
        assert!(matches!(store.lease(&w).await.expect("q").expect("some").state, LeaseState::Draining));
        store.set_lease_state(&w, LeaseState::Destroyed).await.expect("destroy");
        assert!(store.live_leases().await.expect("live").is_empty());
        assert!(store
            .extend_lease(&WorkerId("nope".into()), LeaseExtension { minutes: 1.0, at: 0.0, reason: String::new() })
            .await.expect("extend").is_none());
    }

    #[tokio::test]
    async fn ledger_appends_and_lists() {
        let store = mem_store().await;
        assert!(store.ledger_entries().await.expect("q").is_empty());
        for (ts, cost) in [(100.0, 0.5), (200.0, 0.25)] {
            store.ledger_append(&LedgerEntry {
                ts, worker_id: WorkerId("w".into()), provider: "vast".into(),
                event: "lease_end".into(), minutes: 60.0, cost,
            }).await.expect("append");
        }
        let e = store.ledger_entries().await.expect("q");
        assert_eq!(e.len(), 2);
        assert!((e.iter().map(|x| x.cost).sum::<f64>() - 0.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn checkpoint_record_pin_locate_archive() {
        let store = mem_store().await;
        let run = store.create_run("r", &shell_spec(), None, None).await.expect("run");
        assert!(store.latest_checkpoint(&run).await.expect("q").is_none());
        for step in [100u64, 200] {
            let meta = CheckpointMeta { step, seed: Some(42), arch: Some("cn7".into()), ..Default::default() };
            store.record_checkpoint(&run, step, &format!("ckpt-hot/r/step_{step}/model.safetensors"), &format!("hash{step}"), &meta).await.expect("record");
        }
        assert_eq!(store.checkpoints(&run).await.expect("q").len(), 2);
        assert_eq!(store.latest_checkpoint(&run).await.expect("q").expect("some").step, 200);
        assert!(store.pin_checkpoint(&run, 100, "best").await.expect("pin"));
        assert!(!store.pin_checkpoint(&run, 999, "x").await.expect("pin"));
        store.set_checkpoint_location(&run, 200, CheckpointLocation::R2Final).await.expect("loc");
        let ids = BTreeMap::from([("model.safetensors".to_owned(), "drive-1".to_owned())]);
        store.mark_checkpoint_archived(&run, 100, &ids, 5000.0).await.expect("archive");
        assert_eq!(store.checkpoint_drive_ids(&run, 100).await.expect("q").expect("some")["model.safetensors"], "drive-1");
        assert!(store.checkpoint_drive_ids(&run, 200).await.expect("q").is_none());
    }

    #[tokio::test]
    async fn logs_and_events_round_trip() {
        let store = mem_store().await;
        let run = store.create_run("r", &shell_spec(), None, None).await.expect("run");
        for i in 0..5 { store.append_log(&run, &format!("line {i}")).await.expect("log"); }
        assert_eq!(store.tail_logs(&run, 3).await.expect("tail"), vec!["line 2", "line 3", "line 4"]);
        store.add_event(&run, EventKind::Running, serde_json::json!({ "worker": "w1" })).await.expect("event");
        let ev = store.events(&run).await.expect("events");
        assert!(ev.iter().any(|e| matches!(e.event, EventKind::Running)));
        assert!(ev.len() >= 3); // create_run seeds Created + Queued
    }

    #[tokio::test]
    async fn metrics_ingest_series_filter_and_window() {
        let store = mem_store().await;
        let run = store.create_run("r", &shell_spec(), None, None).await.expect("run");
        for step in 0u64..3 {
            store.append_metrics(&run, step, &BTreeMap::from([
                ("loss".to_owned(), 1.0 - step as f64 * 0.1), ("lr".to_owned(), 0.001),
            ])).await.expect("metrics");
        }
        let s = store.metric_series(&run, Some(&["loss".to_owned()]), 0, 500).await.expect("series");
        assert_eq!(s.series["loss"].len(), 3);
        assert!(!s.series.contains_key("lr"));
        let since = store.metric_series(&run, None, 1, 500).await.expect("series");
        assert!(since.series["loss"].iter().all(|p| p.step >= 1));
    }

    #[tokio::test]
    async fn users_teams_and_api_keys() {
        let store = mem_store().await;
        store.ensure_team("t1", "Team One").await.expect("team");
        store.ensure_team("t1", "Renamed").await.expect("team");
        store.upsert_user("a@x.com", "t1", Role::Write).await.expect("user");
        store.upsert_user("b@x.com", "t1", Role::Admin).await.expect("user");
        assert_eq!(store.get_user("a@x.com").await.expect("q").expect("some").role, Role::Write);
        assert_eq!(store.list_users("t1").await.expect("q").len(), 2);
        store.remove_user("b@x.com").await.expect("remove");
        assert_eq!(store.list_users("t1").await.expect("q").len(), 1);
        store.create_api_key("k1", "t1", "a@x.com", "ci", "ck_abcd", "hash1", Role::Write).await.expect("key");
        assert_eq!(store.list_api_keys("t1").await.expect("q").len(), 1);
        assert_eq!(store.resolve_api_key("hash1").await.expect("q").expect("some").id, "k1");
        store.touch_api_key("k1", 9999.0).await.expect("touch");
        assert!(store.revoke_api_key("k1").await.expect("revoke"));
        assert!(!store.revoke_api_key("k1").await.expect("revoke"));
        assert!(store.resolve_api_key("hash1").await.expect("q").is_none());
    }

    #[tokio::test]
    async fn worker_tokens_resolve_list_touch() {
        let store = mem_store().await;
        store.create_worker_token("tok-1", &WorkerId("mac".into()), "mac", "cw_abcd", "hash").await.expect("create");
        assert_eq!(store.resolve_worker_token("hash").await.expect("q").expect("some").worker_id, WorkerId("mac".into()));
        assert_eq!(store.list_worker_tokens().await.expect("q").len(), 1);
        store.touch_worker_token("tok-1", 8888.0).await.expect("touch");
        assert!(store.revoke_worker_token("tok-1").await.expect("revoke"));
        assert!(store.resolve_worker_token("hash").await.expect("q").is_none());
    }

    #[tokio::test]
    async fn run_transitions_and_next_queued() {
        let store = mem_store().await;
        let run = store.create_run("r", &shell_spec(), None, None).await.expect("run");
        assert_eq!(store.next_queued().await.expect("q").expect("some").summary.id, run);
        let w = WorkerId("w1".into());
        store.transition(&run, RunState::Assigned, Some(&w), None, serde_json::json!({})).await.expect("t");
        store.transition(&run, RunState::Running, Some(&w), None, serde_json::json!({})).await.expect("t");
        store.transition(&run, RunState::Completed, None, Some(0), serde_json::json!({})).await.expect("t");
        let rec = store.run(&run).await.expect("q").expect("some");
        assert_eq!(rec.summary.state, RunState::Completed);
        assert_eq!(rec.summary.exit_code, Some(0));
        assert!(store.next_queued().await.expect("q").is_none());
        let s1 = store.next_run_seq().await.expect("seq");
        assert_eq!(store.next_run_seq().await.expect("seq"), s1 + 1);
    }

    #[tokio::test]
    async fn experiments_run_id_round_trips_and_runs_lists() {
        let store = mem_store().await;
        let run = store.create_run("r", &shell_spec(), None, None).await.expect("run");
        assert!(store.experiments_run_id(&run).await.expect("q").is_none());
        store.set_experiments_run_id(&run, "RUN-20260101-000000-00001").await.expect("set");
        assert_eq!(
            store.experiments_run_id(&run).await.expect("q").as_deref(),
            Some("RUN-20260101-000000-00001"),
        );
        store.create_run("r2", &shell_spec(), None, None).await.expect("run2");
        assert_eq!(store.runs(&Default::default(), 10).await.expect("runs").len(), 2);
    }

    #[tokio::test]
    async fn runs_filters_by_state_and_ref_and_pages() {
        let store = mem_store().await;
        let attached = store
            .create_run("a", &shell_spec(), Some("RUN-20260101-000000-00001"), None)
            .await
            .expect("run");
        let scratch = store.create_run("b", &shell_spec(), None, None).await.expect("run");
        store
            .transition(&scratch, RunState::Completed, None, Some(0), serde_json::json!({}))
            .await
            .expect("t");

        let completed = crate::store::RunQuery {
            state: Some(RunState::Completed),
            ..Default::default()
        };
        let got = store.runs(&completed, 10).await.expect("runs");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, scratch);

        let by_ref = crate::store::RunQuery {
            experiment_ref: Some("RUN-20260101-000000-00001".into()),
            ..Default::default()
        };
        let got = store.runs(&by_ref, 10).await.expect("runs");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, attached);

        // Offset paging: newest first, id-tiebroken, so page 2 of size 1 is the
        // older run; an offset past the end is empty, not an error.
        let page2 = crate::store::RunQuery { offset: 1, ..Default::default() };
        let got = store.runs(&page2, 1).await.expect("runs");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, attached);
        let past_end = crate::store::RunQuery { offset: 5, ..Default::default() };
        assert!(store.runs(&past_end, 10).await.expect("runs").is_empty());
    }
}
