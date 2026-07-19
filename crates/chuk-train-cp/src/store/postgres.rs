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
    ApiKeyInfo, CheckpointInfo, CheckpointLocation, CheckpointMeta, CodeRef, CodeUnitInfo,
    CodeUnitManifest, EventKind, Hardware, Lease, LeaseExtension, LeaseState, LedgerEntry,
    MetricPoint, MetricSeries, Role, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary,
    UnixSeconds, User, WorkerId, WorkerInfo, WorkerState, WorkerTokenInfo,
};
use sqlx::postgres::{PgConnectOptions, PgPool, PgPoolOptions, PgRow};
use sqlx::Row;

use super::ids::{
    downsample_in_place, enum_from_string, enum_to_string, merge_field, new_run_id, now,
};
use super::Store;

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
CREATE TABLE IF NOT EXISTS runs (
  id           text PRIMARY KEY,
  name         text NOT NULL,
  kind         text NOT NULL,
  spec         text NOT NULL,
  state        text NOT NULL,
  worker_id    text,
  exit_code    bigint,
  created_at   double precision NOT NULL,
  updated_at   double precision NOT NULL
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
  email        text PRIMARY KEY,
  team_id      text NOT NULL,
  role         text NOT NULL,
  created_at   double precision NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_worker_tokens_hash ON worker_tokens (token_hash);
CREATE INDEX IF NOT EXISTS idx_runs_state   ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run   ON run_events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics (run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_ckpt_run     ON checkpoints (run_id, step);
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases (state);
-- Additive migrations for a checkpoints table created before these columns
-- (Postgres supports ADD COLUMN IF NOT EXISTS, so this is idempotent).
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS location text NOT NULL DEFAULT 'r2_hot';
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS drive_file_ids text;
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS archived_at double precision;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS experiments_run_id text;
-- Monotonic run sequence (the 5-digit run-id tail). Matches
-- chuk-experiments-server's run_ref_seq so the two systems mint the same shape.
CREATE SEQUENCE IF NOT EXISTS run_ref_seq;
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

#[async_trait]
impl Store for PgStore {
    // ---- workers ----------------------------------------------------------

    async fn worker_joined(
        &self,
        id: &WorkerId,
        labels: &[String],
        hardware: &Hardware,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO workers (id, labels, hardware, connected, joined_at, last_seen)
             VALUES ($1, $2, $3, true, $4, $4)
             ON CONFLICT(id) DO UPDATE SET
               labels = excluded.labels, hardware = excluded.hardware,
               connected = true, last_seen = excluded.last_seen",
        )
        .bind(&id.0)
        .bind(serde_json::to_string(labels)?)
        .bind(serde_json::to_string(hardware)?)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn worker_seen(&self, id: &WorkerId) -> Result<()> {
        sqlx::query("UPDATE workers SET last_seen = $1 WHERE id = $2")
            .bind(now())
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn worker_left(&self, id: &WorkerId) -> Result<()> {
        sqlx::query("UPDATE workers SET connected = false WHERE id = $1")
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_worker_run(&self, id: &WorkerId, run: Option<&RunId>) -> Result<()> {
        sqlx::query("UPDATE workers SET current_run = $1 WHERE id = $2")
            .bind(run.map(|r| r.0.as_str()))
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn worker(&self, id: &WorkerId) -> Result<Option<WorkerInfo>> {
        let row = sqlx::query("SELECT * FROM workers WHERE id = $1")
            .bind(&id.0)
            .fetch_optional(&self.pool)
            .await?;
        let Some(mut worker) = row.map(|r| worker_from_row(&r)).transpose()? else {
            return Ok(None);
        };
        worker.lease = self.lease(id).await?;
        Ok(Some(worker))
    }

    async fn fleet(&self) -> Result<Vec<WorkerInfo>> {
        let rows = sqlx::query("SELECT * FROM workers ORDER BY joined_at")
            .fetch_all(&self.pool)
            .await?;
        let mut workers: Vec<WorkerInfo> = rows
            .iter()
            .map(worker_from_row)
            .collect::<Result<_>>()?;
        for worker in &mut workers {
            worker.lease = self.lease(&worker.id).await?;
        }
        Ok(workers)
    }

    // ---- runs -------------------------------------------------------------

    async fn next_run_seq(&self) -> Result<i64> {
        // nextval() is atomic across sessions (even through Neon's pooler), so
        // the run sequence stays gap-tolerant but never collides.
        let row = sqlx::query("SELECT nextval('run_ref_seq') AS value")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get::<i64, _>("value"))
    }

    async fn set_experiments_run_id(&self, run_id: &RunId, ext_run_id: &str) -> Result<()> {
        sqlx::query("UPDATE runs SET experiments_run_id = $2 WHERE id = $1")
            .bind(&run_id.0)
            .bind(ext_run_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn experiments_run_id(&self, run_id: &RunId) -> Result<Option<String>> {
        let row = sqlx::query("SELECT experiments_run_id FROM runs WHERE id = $1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.and_then(|r| r.get::<Option<String>, _>("experiments_run_id")))
    }

    async fn create_run(&self, name: &str, spec: &RunSpec) -> Result<RunId> {
        let run_id = RunId(new_run_id(now(), self.next_run_seq().await?));
        sqlx::query(
            "INSERT INTO runs (id, name, kind, spec, state, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $6)",
        )
        .bind(&run_id.0)
        .bind(name)
        .bind(spec.kind_str())
        .bind(serde_json::to_string(spec)?)
        .bind(RunState::Queued.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        self.add_event(
            &run_id,
            EventKind::Created,
            serde_json::json!({ "name": name }),
        )
        .await?;
        self.add_event(&run_id, EventKind::Queued, serde_json::Value::Null)
            .await?;
        Ok(run_id)
    }

    async fn transition(
        &self,
        run_id: &RunId,
        state: RunState,
        worker_id: Option<&WorkerId>,
        exit_code: Option<i64>,
        detail: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE runs SET state = $1, updated_at = $2,
               worker_id = COALESCE($3, worker_id),
               exit_code = COALESCE($4, exit_code)
             WHERE id = $5",
        )
        .bind(state.as_str())
        .bind(now())
        .bind(worker_id.map(|w| w.0.as_str()))
        .bind(exit_code)
        .bind(&run_id.0)
        .execute(&self.pool)
        .await?;

        let mut event_detail = detail;
        if let Some(w) = worker_id {
            merge_field(&mut event_detail, "worker", serde_json::json!(w.0));
        }
        if let Some(code) = exit_code {
            merge_field(&mut event_detail, "exit_code", serde_json::json!(code));
        }
        self.add_event(run_id, EventKind::from(state), event_detail)
            .await
    }

    async fn next_queued(&self) -> Result<Option<RunRecord>> {
        let row = sqlx::query("SELECT * FROM runs WHERE state = $1 ORDER BY created_at LIMIT 1")
            .bind(RunState::Queued.as_str())
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| run_from_row(&r)).transpose()
    }

    async fn run(&self, run_id: &RunId) -> Result<Option<RunRecord>> {
        let row = sqlx::query("SELECT * FROM runs WHERE id = $1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| run_from_row(&r)).transpose()
    }

    async fn runs(&self, limit: u32) -> Result<Vec<RunSummary>> {
        // Postgres has no unsigned integers, so LIMIT is bound as i64.
        let rows = sqlx::query("SELECT * FROM runs ORDER BY created_at DESC LIMIT $1")
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| Ok(run_from_row(r)?.summary))
            .collect()
    }

    // ---- logs & events ----------------------------------------------------

    async fn append_log(&self, run_id: &RunId, line: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO run_logs (run_id, n, ts, line)
             SELECT $1, COALESCE(MAX(n), 0) + 1, $2, $3 FROM run_logs WHERE run_id = $1",
        )
        .bind(&run_id.0)
        .bind(now())
        .bind(line)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn tail_logs(&self, run_id: &RunId, lines: u32) -> Result<Vec<String>> {
        let rows =
            sqlx::query("SELECT line FROM run_logs WHERE run_id = $1 ORDER BY n DESC LIMIT $2")
                .bind(&run_id.0)
                .bind(lines as i64)
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .into_iter()
            .rev()
            .map(|r| r.get::<String, _>("line"))
            .collect())
    }

    async fn add_event(
        &self,
        run_id: &RunId,
        event: EventKind,
        detail: serde_json::Value,
    ) -> Result<()> {
        let detail = if detail.is_null() {
            serde_json::json!({})
        } else {
            detail
        };
        sqlx::query("INSERT INTO run_events (run_id, ts, event, detail) VALUES ($1, $2, $3, $4)")
            .bind(&run_id.0)
            .bind(now())
            .bind(enum_to_string(&event))
            .bind(serde_json::to_string(&detail)?)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn events(&self, run_id: &RunId) -> Result<Vec<RunEvent>> {
        let rows =
            sqlx::query("SELECT ts, event, detail FROM run_events WHERE run_id = $1 ORDER BY seq")
                .bind(&run_id.0)
                .fetch_all(&self.pool)
                .await?;
        rows.into_iter()
            .map(|r| {
                Ok(RunEvent {
                    ts: r.get("ts"),
                    event: enum_from_string(r.get::<String, _>("event"))?,
                    detail: serde_json::from_str(&r.get::<String, _>("detail"))?,
                })
            })
            .collect()
    }

    // ---- code units -------------------------------------------------------

    async fn register_code_unit(
        &self,
        code: &CodeRef,
        manifest: &CodeUnitManifest,
        uri: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO code_units (name, sha, manifest, uri, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT(name, sha) DO UPDATE SET manifest = excluded.manifest, uri = excluded.uri",
        )
        .bind(&code.name)
        .bind(&code.sha)
        .bind(serde_json::to_string(manifest)?)
        .bind(uri)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn code_unit(&self, name: &str, sha: &str) -> Result<Option<CodeUnitInfo>> {
        let row = sqlx::query("SELECT * FROM code_units WHERE name = $1 AND sha = $2")
            .bind(name)
            .bind(sha)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| {
            Ok(CodeUnitInfo {
                code: CodeRef {
                    name: r.get("name"),
                    sha: r.get("sha"),
                },
                manifest: serde_json::from_str(&r.get::<String, _>("manifest"))?,
                uri: r.get("uri"),
                created_at: r.get("created_at"),
            })
        })
        .transpose()
    }

    // ---- metrics ----------------------------------------------------------

    async fn append_metrics(
        &self,
        run_id: &RunId,
        step: u64,
        values: &BTreeMap<String, f64>,
    ) -> Result<()> {
        let ts = now();
        let step = step as i64;
        let mut tx = self.pool.begin().await?;
        for (key, value) in values {
            sqlx::query(
                "INSERT INTO metrics (run_id, step, key, value, ts) VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT(run_id, key, step) DO UPDATE SET value = excluded.value",
            )
            .bind(&run_id.0)
            .bind(step)
            .bind(key)
            .bind(value)
            .bind(ts)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn metric_series(
        &self,
        run_id: &RunId,
        keys: Option<&[String]>,
        since_step: u64,
        downsample: u32,
    ) -> Result<MetricSeries> {
        // Filtering by key list is done in Rust: the set is tiny and this
        // avoids building a dynamic `IN ($1, $2, ...)` clause.
        let rows = sqlx::query(
            "SELECT step, key, value FROM metrics
             WHERE run_id = $1 AND step >= $2 ORDER BY key, step",
        )
        .bind(&run_id.0)
        .bind(since_step as i64)
        .fetch_all(&self.pool)
        .await?;

        let wanted: Option<std::collections::HashSet<&str>> =
            keys.map(|ks| ks.iter().map(String::as_str).collect());
        let mut series: BTreeMap<String, Vec<MetricPoint>> = BTreeMap::new();
        for row in rows {
            let key: String = row.get("key");
            if wanted.as_ref().is_some_and(|w| !w.contains(key.as_str())) {
                continue;
            }
            series.entry(key).or_default().push(MetricPoint {
                step: row.get::<i64, _>("step") as u64,
                value: row.get("value"),
            });
        }
        if downsample > 0 {
            for points in series.values_mut() {
                downsample_in_place(points, downsample as usize);
            }
        }
        Ok(MetricSeries {
            run_id: run_id.clone(),
            series,
        })
    }

    // ---- checkpoints ------------------------------------------------------

    async fn record_checkpoint(
        &self,
        run_id: &RunId,
        step: u64,
        uri: &str,
        model_hash: &str,
        meta: &CheckpointMeta,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO checkpoints (run_id, step, uri, model_hash, meta, created_at)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT(run_id, step) DO UPDATE SET
               uri = excluded.uri, model_hash = excluded.model_hash, meta = excluded.meta",
        )
        .bind(&run_id.0)
        .bind(step as i64)
        .bind(uri)
        .bind(model_hash)
        .bind(serde_json::to_string(meta)?)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn checkpoints(&self, run_id: &RunId) -> Result<Vec<CheckpointInfo>> {
        let rows = sqlx::query(&format!(
            "SELECT {CKPT_COLUMNS} FROM checkpoints WHERE run_id = $1 ORDER BY step"
        ))
        .bind(&run_id.0)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(checkpoint_from_row).collect()
    }

    async fn latest_checkpoint(&self, run_id: &RunId) -> Result<Option<CheckpointInfo>> {
        let row = sqlx::query(&format!(
            "SELECT {CKPT_COLUMNS} FROM checkpoints WHERE run_id = $1 ORDER BY step DESC LIMIT 1"
        ))
        .bind(&run_id.0)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| checkpoint_from_row(&r)).transpose()
    }

    async fn pin_checkpoint(&self, run_id: &RunId, step: u64, name: &str) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE checkpoints SET pinned = true, pin_name = $3 WHERE run_id = $1 AND step = $2",
        )
        .bind(&run_id.0)
        .bind(step as i64)
        .bind(name)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn set_checkpoint_location(
        &self,
        run_id: &RunId,
        step: u64,
        location: CheckpointLocation,
    ) -> Result<()> {
        sqlx::query("UPDATE checkpoints SET location = $3 WHERE run_id = $1 AND step = $2")
            .bind(&run_id.0)
            .bind(step as i64)
            .bind(location.as_str())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn mark_checkpoint_archived(
        &self,
        run_id: &RunId,
        step: u64,
        drive_file_ids: &BTreeMap<String, String>,
        archived_at: UnixSeconds,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE checkpoints SET location = $3, drive_file_ids = $4, archived_at = $5
             WHERE run_id = $1 AND step = $2",
        )
        .bind(&run_id.0)
        .bind(step as i64)
        .bind(CheckpointLocation::Drive.as_str())
        .bind(serde_json::to_string(drive_file_ids)?)
        .bind(archived_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn checkpoint_drive_ids(
        &self,
        run_id: &RunId,
        step: u64,
    ) -> Result<Option<BTreeMap<String, String>>> {
        let row =
            sqlx::query("SELECT drive_file_ids FROM checkpoints WHERE run_id = $1 AND step = $2")
                .bind(&run_id.0)
                .bind(step as i64)
                .fetch_optional(&self.pool)
                .await?;
        match row.and_then(|r| r.get::<Option<String>, _>("drive_file_ids")) {
            Some(json) => Ok(Some(serde_json::from_str(&json)?)),
            None => Ok(None),
        }
    }

    // ---- leases -----------------------------------------------------------

    async fn create_lease(&self, lease: &Lease) -> Result<()> {
        sqlx::query(
            "INSERT INTO leases
               (worker_id, provider, instance_id, price_hr, granted_min, drain_window_min,
                started_at, state, extensions)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             ON CONFLICT(worker_id) DO UPDATE SET
               provider=excluded.provider, instance_id=excluded.instance_id,
               price_hr=excluded.price_hr, granted_min=excluded.granted_min,
               drain_window_min=excluded.drain_window_min, started_at=excluded.started_at,
               state=excluded.state, extensions=excluded.extensions",
        )
        .bind(&lease.worker_id.0)
        .bind(&lease.provider)
        .bind(&lease.instance_id)
        .bind(lease.price_hr)
        .bind(lease.granted_min)
        .bind(lease.drain_window_min)
        .bind(lease.started_at)
        .bind(lease.state.as_str())
        .bind(serde_json::to_string(&lease.extensions)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn lease(&self, worker_id: &WorkerId) -> Result<Option<Lease>> {
        let row = sqlx::query("SELECT * FROM leases WHERE worker_id = $1")
            .bind(&worker_id.0)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| lease_from_row(&r)).transpose()
    }

    async fn live_leases(&self) -> Result<Vec<Lease>> {
        let rows = sqlx::query("SELECT * FROM leases WHERE state != $1")
            .bind(LeaseState::Destroyed.as_str())
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(lease_from_row).collect()
    }

    async fn set_lease_state(&self, worker_id: &WorkerId, state: LeaseState) -> Result<()> {
        sqlx::query("UPDATE leases SET state = $2 WHERE worker_id = $1")
            .bind(&worker_id.0)
            .bind(state.as_str())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn extend_lease(
        &self,
        worker_id: &WorkerId,
        ext: LeaseExtension,
    ) -> Result<Option<Lease>> {
        let Some(mut lease) = self.lease(worker_id).await? else {
            return Ok(None);
        };
        lease.extensions.push(ext);
        sqlx::query("UPDATE leases SET extensions = $2 WHERE worker_id = $1")
            .bind(&worker_id.0)
            .bind(serde_json::to_string(&lease.extensions)?)
            .execute(&self.pool)
            .await?;
        Ok(Some(lease))
    }

    // ---- ledger -----------------------------------------------------------

    async fn ledger_append(&self, entry: &LedgerEntry) -> Result<()> {
        sqlx::query(
            "INSERT INTO ledger (ts, worker_id, provider, event, minutes, cost)
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(entry.ts)
        .bind(&entry.worker_id.0)
        .bind(&entry.provider)
        .bind(&entry.event)
        .bind(entry.minutes)
        .bind(entry.cost)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn ledger_entries(&self) -> Result<Vec<LedgerEntry>> {
        let rows = sqlx::query("SELECT * FROM ledger ORDER BY id")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .into_iter()
            .map(|r| LedgerEntry {
                ts: r.get("ts"),
                worker_id: WorkerId(r.get("worker_id")),
                provider: r.get("provider"),
                event: r.get("event"),
                minutes: r.get("minutes"),
                cost: r.get("cost"),
            })
            .collect())
    }

    // ---- users & teams ----------------------------------------------------

    async fn ensure_team(&self, id: &str, name: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO teams (id, name, created_at) VALUES ($1, $2, $3)
             ON CONFLICT(id) DO NOTHING",
        )
        .bind(id)
        .bind(name)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn upsert_user(&self, email: &str, team_id: &str, role: Role) -> Result<()> {
        sqlx::query(
            "INSERT INTO users (email, team_id, role, created_at) VALUES ($1, $2, $3, $4)
             ON CONFLICT(email) DO UPDATE SET team_id = excluded.team_id, role = excluded.role",
        )
        .bind(email)
        .bind(team_id)
        .bind(role.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_user(&self, email: &str) -> Result<Option<User>> {
        let row =
            sqlx::query("SELECT email, team_id, role, created_at FROM users WHERE email = $1")
                .bind(email)
                .fetch_optional(&self.pool)
                .await?;
        row.map(|r| user_from_row(&r)).transpose()
    }

    async fn list_users(&self, team_id: &str) -> Result<Vec<User>> {
        let rows = sqlx::query(
            "SELECT email, team_id, role, created_at FROM users WHERE team_id = $1 ORDER BY email",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(user_from_row).collect()
    }

    async fn remove_user(&self, email: &str) -> Result<()> {
        sqlx::query("DELETE FROM users WHERE email = $1")
            .bind(email)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn create_api_key(
        &self,
        id: &str,
        team_id: &str,
        created_by: &str,
        name: &str,
        prefix: &str,
        key_hash: &str,
        role: Role,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO api_keys (id, team_id, created_by, name, prefix, key_hash, role, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(id)
        .bind(team_id)
        .bind(created_by)
        .bind(name)
        .bind(prefix)
        .bind(key_hash)
        .bind(role.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn list_api_keys(&self, team_id: &str) -> Result<Vec<ApiKeyInfo>> {
        let rows = sqlx::query(
            "SELECT id, team_id, created_by, name, prefix, role, created_at, last_used_at, revoked_at
             FROM api_keys WHERE team_id = $1 ORDER BY created_at DESC",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(api_key_from_row).collect()
    }

    async fn revoke_api_key(&self, id: &str) -> Result<bool> {
        let result =
            sqlx::query("UPDATE api_keys SET revoked_at = $2 WHERE id = $1 AND revoked_at IS NULL")
                .bind(id)
                .bind(now())
                .execute(&self.pool)
                .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn resolve_api_key(&self, key_hash: &str) -> Result<Option<ApiKeyInfo>> {
        let row = sqlx::query(
            "SELECT id, team_id, created_by, name, prefix, role, created_at, last_used_at, revoked_at
             FROM api_keys WHERE key_hash = $1 AND revoked_at IS NULL",
        )
        .bind(key_hash)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| api_key_from_row(&r)).transpose()
    }

    async fn touch_api_key(&self, id: &str, at: UnixSeconds) -> Result<()> {
        sqlx::query("UPDATE api_keys SET last_used_at = $2 WHERE id = $1")
            .bind(id)
            .bind(at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ---- worker tokens ----------------------------------------------------

    async fn create_worker_token(
        &self,
        id: &str,
        worker_id: &WorkerId,
        name: &str,
        prefix: &str,
        token_hash: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO worker_tokens (id, worker_id, name, prefix, token_hash, created_at)
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(id)
        .bind(&worker_id.0)
        .bind(name)
        .bind(prefix)
        .bind(token_hash)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn resolve_worker_token(&self, token_hash: &str) -> Result<Option<WorkerTokenInfo>> {
        let row = sqlx::query(
            "SELECT id, worker_id, name, prefix, created_at, last_used_at, revoked_at
             FROM worker_tokens WHERE token_hash = $1 AND revoked_at IS NULL",
        )
        .bind(token_hash)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|r| worker_token_from_row(&r)).transpose()
    }

    async fn list_worker_tokens(&self) -> Result<Vec<WorkerTokenInfo>> {
        let rows = sqlx::query(
            "SELECT id, worker_id, name, prefix, created_at, last_used_at, revoked_at
             FROM worker_tokens ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.iter().map(worker_token_from_row).collect()
    }

    async fn revoke_worker_token(&self, id: &str) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE worker_tokens SET revoked_at = $2 WHERE id = $1 AND revoked_at IS NULL",
        )
        .bind(id)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn touch_worker_token(&self, id: &str, at: UnixSeconds) -> Result<()> {
        sqlx::query("UPDATE worker_tokens SET last_used_at = $2 WHERE id = $1")
            .bind(id)
            .bind(at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

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
        // Populated by worker()/fleet() from the leases table after conversion.
        lease: None,
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
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        },
        spec,
    })
}

/// Live round-trip against real Neon. Ignored by default (needs a postgres
/// `CHUK_TRAIN_STORE` in the env); run with `.env` sourced:
///   cargo test -p chuk-train-cp store::postgres::pg_live::round_trip -- --ignored --nocapture
/// Exercises the dialect's risky bits — boolean columns, `bigserial`, the
/// metric transaction, upserts — and deletes its own rows so the shared DB
/// stays clean.
#[cfg(test)]
mod pg_live {
    use super::*;
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
        let run_id = store.create_run("pg-live", &spec).await.expect("create_run");
        assert!(run_id.0.starts_with("RUN-"), "{}", run_id.0);
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
