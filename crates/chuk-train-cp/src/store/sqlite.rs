//! SQLite adapter (sqlx, async native).

use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    ApiKeyInfo, CheckpointInfo, CheckpointLocation, CheckpointMeta, CodeRef, CodeUnitInfo,
    CodeUnitManifest, EventKind, Hardware, Lease, LeaseExtension, LeaseState, LedgerEntry,
    MetricPoint, MetricSeries, Role, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary,
    UnixSeconds, User, WorkerId, WorkerInfo, WorkerState,
};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use super::ids::{
    downsample_in_place, enum_from_string, enum_to_string, merge_field, new_run_id, now,
};
use super::Store;

/// Counter row name backing the monotonic run sequence (the run-id tail).
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
CREATE TABLE IF NOT EXISTS runs (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  kind         TEXT NOT NULL,
  spec         TEXT NOT NULL,
  state        TEXT NOT NULL,
  worker_id    TEXT,
  exit_code    INTEGER,
  created_at   REAL NOT NULL,
  updated_at   REAL NOT NULL
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
  email        TEXT PRIMARY KEY,
  team_id      TEXT NOT NULL,
  role         TEXT NOT NULL,
  created_at   REAL NOT NULL
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
CREATE TABLE IF NOT EXISTS counters (
  name         TEXT PRIMARY KEY,
  value        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apikeys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_runs_state   ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run   ON run_events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics (run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_ckpt_run     ON checkpoints (run_id, step);
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases (state);
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
        ] {
            let _ = sqlx::query(stmt).execute(&pool).await;
        }
        Ok(Self { pool })
    }
}

#[async_trait]
impl Store for SqliteStore {
    // ---- workers ----------------------------------------------------------

    async fn worker_joined(
        &self,
        id: &WorkerId,
        labels: &[String],
        hardware: &Hardware,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO workers (id, labels, hardware, connected, joined_at, last_seen)
             VALUES (?1, ?2, ?3, 1, ?4, ?4)
             ON CONFLICT(id) DO UPDATE SET
               labels = excluded.labels, hardware = excluded.hardware,
               connected = 1, last_seen = excluded.last_seen",
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
        sqlx::query("UPDATE workers SET last_seen = ?1 WHERE id = ?2")
            .bind(now())
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn worker_left(&self, id: &WorkerId) -> Result<()> {
        sqlx::query("UPDATE workers SET connected = 0 WHERE id = ?1")
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_worker_run(&self, id: &WorkerId, run: Option<&RunId>) -> Result<()> {
        sqlx::query("UPDATE workers SET current_run = ?1 WHERE id = ?2")
            .bind(run.map(|r| r.0.as_str()))
            .bind(&id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn worker(&self, id: &WorkerId) -> Result<Option<WorkerInfo>> {
        let row = sqlx::query("SELECT * FROM workers WHERE id = ?1")
            .bind(&id.0)
            .fetch_optional(&self.pool)
            .await?;
        let Some(mut worker) = row.map(worker_from_row).transpose()? else {
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
            .into_iter()
            .map(worker_from_row)
            .collect::<Result<_>>()?;
        for worker in &mut workers {
            worker.lease = self.lease(&worker.id).await?;
        }
        Ok(workers)
    }

    // ---- runs -------------------------------------------------------------

    async fn next_run_seq(&self) -> Result<i64> {
        // Atomic bump-and-return: seed at 1 on first call, +1 thereafter.
        // SQLite 3.35+ supports RETURNING, and the single writer connection
        // makes the upsert race-free.
        let row = sqlx::query(
            "INSERT INTO counters (name, value) VALUES (?1, 1)
             ON CONFLICT(name) DO UPDATE SET value = value + 1
             RETURNING value",
        )
        .bind(RUN_SEQ_COUNTER)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("value"))
    }

    async fn set_experiments_run_id(&self, run_id: &RunId, ext_run_id: &str) -> Result<()> {
        sqlx::query("UPDATE runs SET experiments_run_id = ?2 WHERE id = ?1")
            .bind(&run_id.0)
            .bind(ext_run_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn experiments_run_id(&self, run_id: &RunId) -> Result<Option<String>> {
        let row = sqlx::query("SELECT experiments_run_id FROM runs WHERE id = ?1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.and_then(|r| r.get::<Option<String>, _>("experiments_run_id")))
    }

    async fn create_run(&self, name: &str, spec: &RunSpec) -> Result<RunId> {
        let run_id = RunId(new_run_id(now(), self.next_run_seq().await?));
        sqlx::query(
            "INSERT INTO runs (id, name, kind, spec, state, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
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
            "UPDATE runs SET state = ?1, updated_at = ?2,
               worker_id = COALESCE(?3, worker_id),
               exit_code = COALESCE(?4, exit_code)
             WHERE id = ?5",
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
        let row = sqlx::query("SELECT * FROM runs WHERE state = ?1 ORDER BY created_at LIMIT 1")
            .bind(RunState::Queued.as_str())
            .fetch_optional(&self.pool)
            .await?;
        row.map(run_from_row).transpose()
    }

    async fn run(&self, run_id: &RunId) -> Result<Option<RunRecord>> {
        let row = sqlx::query("SELECT * FROM runs WHERE id = ?1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        row.map(run_from_row).transpose()
    }

    async fn runs(&self, limit: u32) -> Result<Vec<RunSummary>> {
        let rows = sqlx::query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?1")
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter()
            .map(|r| Ok(run_from_row(r)?.summary))
            .collect()
    }

    // ---- logs & events ----------------------------------------------------

    async fn append_log(&self, run_id: &RunId, line: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO run_logs (run_id, n, ts, line)
             SELECT ?1, COALESCE(MAX(n), 0) + 1, ?2, ?3 FROM run_logs WHERE run_id = ?1",
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
            sqlx::query("SELECT line FROM run_logs WHERE run_id = ?1 ORDER BY n DESC LIMIT ?2")
                .bind(&run_id.0)
                .bind(lines)
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
        sqlx::query("INSERT INTO run_events (run_id, ts, event, detail) VALUES (?1, ?2, ?3, ?4)")
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
            sqlx::query("SELECT ts, event, detail FROM run_events WHERE run_id = ?1 ORDER BY seq")
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
             VALUES (?1, ?2, ?3, ?4, ?5)
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
        let row = sqlx::query("SELECT * FROM code_units WHERE name = ?1 AND sha = ?2")
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
                "INSERT INTO metrics (run_id, step, key, value, ts) VALUES (?1, ?2, ?3, ?4, ?5)
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
        // avoids building a dynamic `IN (?, ?, ...)` clause.
        let rows = sqlx::query(
            "SELECT step, key, value FROM metrics
             WHERE run_id = ?1 AND step >= ?2 ORDER BY key, step",
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
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
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
        let rows = sqlx::query("SELECT * FROM checkpoints WHERE run_id = ?1 ORDER BY step")
            .bind(&run_id.0)
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter().map(checkpoint_from_row).collect()
    }

    async fn latest_checkpoint(&self, run_id: &RunId) -> Result<Option<CheckpointInfo>> {
        let row =
            sqlx::query("SELECT * FROM checkpoints WHERE run_id = ?1 ORDER BY step DESC LIMIT 1")
                .bind(&run_id.0)
                .fetch_optional(&self.pool)
                .await?;
        row.map(checkpoint_from_row).transpose()
    }

    async fn pin_checkpoint(&self, run_id: &RunId, step: u64, name: &str) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE checkpoints SET pinned = 1, pin_name = ?3 WHERE run_id = ?1 AND step = ?2",
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
        sqlx::query("UPDATE checkpoints SET location = ?3 WHERE run_id = ?1 AND step = ?2")
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
            "UPDATE checkpoints SET location = ?3, drive_file_ids = ?4, archived_at = ?5
             WHERE run_id = ?1 AND step = ?2",
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
        let row = sqlx::query("SELECT drive_file_ids FROM checkpoints WHERE run_id = ?1 AND step = ?2")
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
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
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
        let row = sqlx::query("SELECT * FROM leases WHERE worker_id = ?1")
            .bind(&worker_id.0)
            .fetch_optional(&self.pool)
            .await?;
        row.map(lease_from_row).transpose()
    }

    async fn live_leases(&self) -> Result<Vec<Lease>> {
        let rows = sqlx::query("SELECT * FROM leases WHERE state != ?1")
            .bind(LeaseState::Destroyed.as_str())
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter().map(lease_from_row).collect()
    }

    async fn set_lease_state(&self, worker_id: &WorkerId, state: LeaseState) -> Result<()> {
        sqlx::query("UPDATE leases SET state = ?2 WHERE worker_id = ?1")
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
        sqlx::query("UPDATE leases SET extensions = ?2 WHERE worker_id = ?1")
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
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
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
            "INSERT INTO teams (id, name, created_at) VALUES (?1, ?2, ?3)
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
            "INSERT INTO users (email, team_id, role, created_at) VALUES (?1, ?2, ?3, ?4)
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
            sqlx::query("SELECT email, team_id, role, created_at FROM users WHERE email = ?1")
                .bind(email)
                .fetch_optional(&self.pool)
                .await?;
        row.map(user_from_row).transpose()
    }

    async fn list_users(&self, team_id: &str) -> Result<Vec<User>> {
        let rows = sqlx::query(
            "SELECT email, team_id, role, created_at FROM users WHERE team_id = ?1 ORDER BY email",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(user_from_row).collect()
    }

    async fn remove_user(&self, email: &str) -> Result<()> {
        sqlx::query("DELETE FROM users WHERE email = ?1")
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
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
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
             FROM api_keys WHERE team_id = ?1 ORDER BY created_at DESC",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(api_key_from_row).collect()
    }

    async fn revoke_api_key(&self, id: &str) -> Result<bool> {
        let result =
            sqlx::query("UPDATE api_keys SET revoked_at = ?2 WHERE id = ?1 AND revoked_at IS NULL")
                .bind(id)
                .bind(now())
                .execute(&self.pool)
                .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn resolve_api_key(&self, key_hash: &str) -> Result<Option<ApiKeyInfo>> {
        let row = sqlx::query(
            "SELECT id, team_id, created_by, name, prefix, role, created_at, last_used_at, revoked_at
             FROM api_keys WHERE key_hash = ?1 AND revoked_at IS NULL",
        )
        .bind(key_hash)
        .fetch_optional(&self.pool)
        .await?;
        row.map(api_key_from_row).transpose()
    }

    async fn touch_api_key(&self, id: &str, at: UnixSeconds) -> Result<()> {
        sqlx::query("UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1")
            .bind(id)
            .bind(at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

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
        // Populated by worker()/fleet() from the leases table after conversion.
        lease: None,
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
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        },
        spec,
    })
}
