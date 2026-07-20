//! PostgreSQL adapter (sqlx, async native).
//!
//! A dialect port of [`super::sqlite`]: same tables, columns, and semantics,
//! rewritten for Postgres. It backs a multi-machine control plane where the
//! state must live off-box (Fly machine → Neon over TLS). Notable differences
//! from the SQLite schema, all forced by the dialect:
//!   * placeholders are `$1, $2, …` (sqlx does not translate `?1`);
//!   * `REAL`/`INTEGER`/`TEXT` → `double precision`/`bigint`/`text`;
//!   * the 0/1 integer flags (`connected`, `pinned`) become native `boolean`;
//!   * `INTEGER PRIMARY KEY AUTOINCREMENT` (rowid surrogate) → `bigserial`.
//!
//! Timestamps stay `double precision` unix seconds (`f64`) — the rest of the
//! code treats them as such; they are deliberately not `timestamptz`.

use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    ApiKeyInfo, CheckpointInfo, CheckpointLocation, CheckpointMeta, CodeRef, CodeUnitInfo, DEFAULT_TEAM_ID,
    CodeUnitManifest, EventKind, Hardware, Lease, LeaseExtension, LeaseState, LedgerEntry,
    MetricPoint, MetricSeries, OutboxRow, Role, RunEvent, RunId, RunRecord, RunSpec, RunState,
    RunSummary, UnixSeconds, User, WorkerId, WorkerInfo, WorkerState, WorkerTelemetry,
    WorkerTokenInfo, WORKER_SAMPLE_RETENTION,
};
use sqlx::postgres::{PgConnectOptions, PgPool, PgPoolOptions, PgRow};
use sqlx::Row;

use super::ids::{
    downsample_in_place, enum_from_string, enum_to_string, merge_field, new_run_id, now,
    worker_telemetry_from_samples,
};

/// Pool ceiling. Fly runs one small machine against Neon's pooled endpoint, so
/// a handful of connections is plenty and keeps us well under Neon's limits.
const MAX_CONNECTIONS: u32 = 5;

/// Explicit column list for checkpoint reads. Never `SELECT *`: on Neon's
/// pooled endpoint an additive migration would otherwise invalidate a shared
/// backend's cached plan ("cached plan must not change result type"). Naming
/// the columns keeps a read's result type stable across `ADD COLUMN`.
const CKPT_COLUMNS: &str =
    "run_id, step, uri, model_hash, meta, pinned, pin_name, created_at, location, archived_at";

/// Schema DDL, mirroring the SQLite adapter table-for-table with Postgres
/// types. Run on every `open()`; `IF NOT EXISTS` makes it idempotent.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS workers (
  id           text PRIMARY KEY,
  labels       text NOT NULL DEFAULT '[]',
  hardware     text NOT NULL DEFAULT '{}',
  connected    boolean NOT NULL DEFAULT true,
  current_run  text,
  joined_at    double precision NOT NULL,
  last_seen    double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS worker_samples (
  worker_id    text NOT NULL,
  ts           double precision NOT NULL,
  payload      text NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id             text PRIMARY KEY,
  name           text NOT NULL,
  kind           text NOT NULL,
  spec           text NOT NULL,
  state          text NOT NULL,
  worker_id      text,
  exit_code      bigint,
  experiment_ref text,
  created_by     text,
  created_at     double precision NOT NULL,
  updated_at     double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS run_logs (
  run_id       text NOT NULL,
  n            bigint NOT NULL,
  ts           double precision NOT NULL,
  line         text NOT NULL,
  PRIMARY KEY (run_id, n)
);
CREATE TABLE IF NOT EXISTS run_events (
  seq          bigserial PRIMARY KEY,
  run_id       text NOT NULL,
  ts           double precision NOT NULL,
  event        text NOT NULL,
  detail       text NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS code_units (
  name         text NOT NULL,
  sha          text NOT NULL,
  manifest     text NOT NULL,
  uri          text NOT NULL,
  created_at   double precision NOT NULL,
  PRIMARY KEY (name, sha)
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id       text NOT NULL,
  step         bigint NOT NULL,
  key          text NOT NULL,
  value        double precision NOT NULL,
  ts           double precision NOT NULL,
  PRIMARY KEY (run_id, key, step)
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id         text NOT NULL,
  step           bigint NOT NULL,
  uri            text NOT NULL,
  model_hash     text NOT NULL,
  meta           text NOT NULL,
  pinned         boolean NOT NULL DEFAULT false,
  pin_name       text,
  created_at     double precision NOT NULL,
  location       text NOT NULL DEFAULT 'r2_hot',
  drive_file_ids text,
  archived_at    double precision,
  PRIMARY KEY (run_id, step)
);
CREATE TABLE IF NOT EXISTS leases (
  worker_id        text PRIMARY KEY,
  provider         text NOT NULL,
  instance_id      text NOT NULL,
  price_hr         double precision NOT NULL,
  granted_min      double precision NOT NULL,
  drain_window_min double precision NOT NULL,
  started_at       double precision NOT NULL,
  state            text NOT NULL,
  extensions       text NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS ledger (
  id           bigserial PRIMARY KEY,
  ts           double precision NOT NULL,
  worker_id    text NOT NULL,
  provider     text NOT NULL,
  event        text NOT NULL,
  minutes      double precision NOT NULL,
  cost         double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS teams (
  id           text PRIMARY KEY,
  name         text NOT NULL,
  created_at   double precision NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
  email                          text PRIMARY KEY,
  team_id                        text NOT NULL,
  role                           text NOT NULL,
  created_at                     double precision NOT NULL,
  experiments_api_key_encrypted  text
);
CREATE TABLE IF NOT EXISTS api_keys (
  id           text PRIMARY KEY,
  team_id      text NOT NULL,
  created_by   text NOT NULL,
  name         text NOT NULL,
  prefix       text NOT NULL,
  key_hash     text NOT NULL,
  role         text NOT NULL,
  created_at   double precision NOT NULL,
  last_used_at double precision,
  revoked_at   double precision
);
CREATE TABLE IF NOT EXISTS worker_tokens (
  id           text PRIMARY KEY,
  worker_id    text NOT NULL,
  name         text NOT NULL,
  prefix       text NOT NULL,
  token_hash   text NOT NULL,
  created_at   double precision NOT NULL,
  last_used_at double precision,
  revoked_at   double precision
);
CREATE TABLE IF NOT EXISTS experiments_outbox (
  id               bigserial PRIMARY KEY,
  run_id           text NOT NULL,
  kind             text NOT NULL,
  payload          text NOT NULL,
  attempts         bigint NOT NULL DEFAULT 0,
  last_error       text,
  created_at       double precision NOT NULL,
  next_attempt_at  double precision NOT NULL,
  completed_at     double precision
);
CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_worker_tokens_hash ON worker_tokens (token_hash);
CREATE INDEX IF NOT EXISTS idx_runs_state   ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run   ON run_events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics (run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_ckpt_run     ON checkpoints (run_id, step);
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases (state);
CREATE INDEX IF NOT EXISTS idx_outbox_due   ON experiments_outbox (next_attempt_at) WHERE completed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_worker_samples ON worker_samples (worker_id, ts);
-- Additive migrations for a checkpoints table created before these columns
-- (Postgres supports ADD COLUMN IF NOT EXISTS, so this is idempotent).
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS location text NOT NULL DEFAULT 'r2_hot';
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS drive_file_ids text;
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS archived_at double precision;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS experiments_run_id text;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS experiment_ref text;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS created_by text;
ALTER TABLE users ADD COLUMN IF NOT EXISTS experiments_api_key_encrypted text;
-- Monotonic execution sequence (the 5-digit tail of our EXEC-… ids). Ours
-- alone — deliberately independent of chuk-experiments-server's run_ref_seq,
-- since our execution ids no longer share their run-id shape.
CREATE SEQUENCE IF NOT EXISTS exec_ref_seq;
"#;

pub struct PgStore {
    pool: PgPool,
}

impl PgStore {
    /// Open a pool against `url` (the full connection string, e.g. Neon's
    /// pooled endpoint including `?sslmode=require`) and apply the schema.
    pub async fn open(url: &str) -> Result<Self> {
        // Neon's pooled endpoint (pgbouncer-style) reuses server sessions across
        // clients, so server-side cached plans go stale across schema migrations
        // ("cached plan must not change result type"). Disabling sqlx's prepared-
        // statement cache sidesteps that entirely; the re-prepare cost is
        // negligible at our query rates.
        let options = url
            .parse::<PgConnectOptions>()?
            .statement_cache_capacity(0);
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .connect_with(options)
            .await?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
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


fn api_key_from_row(row: &PgRow) -> Result<ApiKeyInfo> {
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

fn worker_token_from_row(row: &PgRow) -> Result<WorkerTokenInfo> {
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

fn user_from_row(row: &PgRow) -> Result<User> {
    Ok(User {
        email: row.get("email"),
        team_id: row.get("team_id"),
        role: enum_from_string(row.get::<String, _>("role"))?,
        created_at: row.get("created_at"),
    })
}

// ---- row parsers (PgRow ≠ SqliteRow, so these mirror the sqlite ones) ------

fn lease_from_row(row: &PgRow) -> Result<Lease> {
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

fn checkpoint_from_row(row: &PgRow) -> Result<CheckpointInfo> {
    Ok(CheckpointInfo {
        run_id: RunId(row.get("run_id")),
        step: row.get::<i64, _>("step") as u64,
        uri: row.get("uri"),
        model_hash: row.get("model_hash"),
        // Native boolean in Postgres (SQLite stored this as INTEGER 0/1).
        pinned: row.get::<bool, _>("pinned"),
        pin_name: row.get("pin_name"),
        meta: serde_json::from_str(&row.get::<String, _>("meta"))?,
        created_at: row.get("created_at"),
        location: enum_from_string(row.get::<String, _>("location"))?,
        archived_at: row.get("archived_at"),
    })
}

fn worker_from_row(row: &PgRow) -> Result<WorkerInfo> {
    let last_seen: f64 = row.get("last_seen");
    Ok(WorkerInfo {
        id: WorkerId(row.get("id")),
        labels: serde_json::from_str(&row.get::<String, _>("labels"))?,
        hardware: serde_json::from_str(&row.get::<String, _>("hardware"))?,
        // Native boolean in Postgres (SQLite stored this as INTEGER 0/1).
        state: if row.get::<bool, _>("connected") {
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

fn run_from_row(row: &PgRow) -> Result<RunRecord> {
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

/// Live round-trip against real Neon. Ignored by default (needs a postgres
/// `CHUK_TRAIN_STORE` in the env); run with `.env` sourced:
///   cargo test -p chuk-train-controlplane store::postgres::pg_live::round_trip -- --ignored --nocapture
/// Exercises the dialect's risky bits — boolean columns, `bigserial`, the
/// metric transaction, upserts — and deletes its own rows so the shared DB
/// stays clean.
#[cfg(test)]
mod pg_live {
    use super::*;
    use crate::store::prelude::*;
    use chuk_train_proto::{Hardware, LedgerEntry, ShellSpec};

    fn pg_url() -> Option<String> {
        match std::env::var("CHUK_TRAIN_STORE") {
            Ok(u) if u.starts_with("postgres") => Some(u),
            _ => None,
        }
    }

    #[ignore]
    #[tokio::test]
    async fn round_trip() {
        let Some(url) = pg_url() else {
            eprintln!("skip: CHUK_TRAIN_STORE is not a postgres url");
            return;
        };
        let store = PgStore::open(&url).await.expect("open + schema");

        // Purge leftovers from any prior interrupted run (a panic before the
        // fix skipped cleanup), so this shared db / dashboard stays clean.
        let purge = PgPoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("purge pool");
        for stmt in [
            "DELETE FROM run_logs   WHERE run_id IN (SELECT id FROM runs WHERE name='pg-live')",
            "DELETE FROM run_events WHERE run_id IN (SELECT id FROM runs WHERE name='pg-live')",
            "DELETE FROM metrics    WHERE run_id IN (SELECT id FROM runs WHERE name='pg-live')",
            "DELETE FROM checkpoints WHERE run_id IN (SELECT id FROM runs WHERE name='pg-live')",
            "DELETE FROM runs   WHERE name='pg-live'",
            "DELETE FROM workers WHERE id LIKE 'pgtest-%'",
            "DELETE FROM leases  WHERE worker_id LIKE 'pgtest-%'",
            "DELETE FROM ledger  WHERE worker_id LIKE 'pgtest-%'",
        ] {
            let _ = sqlx::query(stmt).execute(&purge).await;
        }

        // workers: boolean `connected`, upsert, Hardware json round-trip
        let wid = WorkerId(format!("pgtest-{}", new_run_id(now(), store.next_run_seq().await.unwrap())));
        let hw = Hardware {
            host: "t".into(),
            os: "linux".into(),
            gpu: Some("T4".into()),
            vram_mb: Some(16_384),
            driver: None,
        };
        store.worker_joined(&wid, &["gpu".into()], &hw).await.expect("worker_joined");
        let w = store.worker(&wid).await.expect("worker").expect("some");
        assert_eq!(w.hardware.gpu.as_deref(), Some("T4"));
        assert!(matches!(w.state, WorkerState::Connected));
        store.worker_left(&wid).await.expect("worker_left");
        let w2 = store.worker(&wid).await.expect("worker").expect("some");
        assert!(matches!(w2.state, WorkerState::Disconnected));

        // runs + events + transition + sortable id
        let spec = RunSpec::Shell(ShellSpec { command: "echo hi".into(), timeout_s: 60 });
        let run_id = store.create_run("pg-live", &spec, None, None).await.expect("create_run");
        assert!(run_id.0.starts_with("EXEC-"), "{}", run_id.0);
        let rec = store.run(&run_id).await.expect("run").expect("some");
        assert_eq!(rec.summary.name, "pg-live");
        assert!(matches!(rec.summary.state, RunState::Queued));
        store
            .transition(&run_id, RunState::Running, Some(&wid), None, serde_json::json!({}))
            .await
            .expect("transition");
        let rec2 = store.run(&run_id).await.expect("run").expect("some");
        assert!(matches!(rec2.summary.state, RunState::Running));
        assert!(store.runs(5).await.expect("runs").iter().any(|r| r.id == run_id));
        assert!(!store.events(&run_id).await.expect("events").is_empty());

        // logs: monotonic n via COALESCE(MAX(n)+1), tail order
        for i in 0..3 {
            store.append_log(&run_id, &format!("line {i}")).await.expect("log");
        }
        let tail = store.tail_logs(&run_id, 2).await.expect("tail");
        assert_eq!(tail, vec!["line 1".to_string(), "line 2".to_string()]);

        // metrics: transaction + upsert
        let mut m = BTreeMap::new();
        m.insert("loss".to_string(), 1.5);
        store.append_metrics(&run_id, 10, &m).await.expect("metrics");
        let series = store.metric_series(&run_id, None, 0, 0).await.expect("series");
        assert_eq!(series.series.get("loss").expect("loss series")[0].value, 1.5);

        // checkpoints + pin: the other boolean column
        let meta = CheckpointMeta { step: 10, ..Default::default() };
        store.record_checkpoint(&run_id, 10, "r2://x", "hash", &meta).await.expect("record_ckpt");
        assert!(store.pin_checkpoint(&run_id, 10, "best").await.expect("pin"));
        let cks = store.checkpoints(&run_id).await.expect("cks");
        assert_eq!(cks.len(), 1);
        assert!(cks[0].pinned);
        assert_eq!(cks[0].pin_name.as_deref(), Some("best"));

        // ledger: bigserial id + ORDER BY id
        store
            .ledger_append(&LedgerEntry {
                ts: now(),
                worker_id: wid.clone(),
                provider: "mock".into(),
                event: "grant".into(),
                minutes: 15.0,
                cost: 0.03,
            })
            .await
            .expect("ledger");
        assert!(store
            .ledger_entries()
            .await
            .expect("ledger_entries")
            .iter()
            .any(|e| e.worker_id == wid));

        // cleanup — remove this test's rows so the shared Neon DB stays tidy
        let pool = PgPoolOptions::new().max_connections(1).connect(&url).await.expect("cleanup pool");
        for stmt in [
            "DELETE FROM run_logs WHERE run_id = $1",
            "DELETE FROM run_events WHERE run_id = $1",
            "DELETE FROM metrics WHERE run_id = $1",
            "DELETE FROM checkpoints WHERE run_id = $1",
            "DELETE FROM runs WHERE id = $1",
        ] {
            sqlx::query(stmt).bind(&run_id.0).execute(&pool).await.expect("cleanup run rows");
        }
        for stmt in [
            "DELETE FROM workers WHERE id = $1",
            "DELETE FROM leases WHERE worker_id = $1",
            "DELETE FROM ledger WHERE worker_id = $1",
        ] {
            sqlx::query(stmt).bind(&wid.0).execute(&pool).await.expect("cleanup worker rows");
        }
        eprintln!("pg live round-trip ok: {}", run_id.0);
    }
}
