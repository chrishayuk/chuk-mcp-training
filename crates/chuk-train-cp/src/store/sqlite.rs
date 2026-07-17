//! SQLite adapter (sqlx, async native).

use std::collections::BTreeMap;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    CheckpointInfo, CheckpointMeta, CodeRef, CodeUnitInfo, CodeUnitManifest, EventKind, Hardware,
    MetricPoint, MetricSeries, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary,
    UnixSeconds, WorkerId, WorkerInfo, WorkerState,
};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use sqlx::Row;

use super::Store;

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
  run_id       TEXT NOT NULL,
  step         INTEGER NOT NULL,
  uri          TEXT NOT NULL,
  model_hash   TEXT NOT NULL,
  meta         TEXT NOT NULL,
  pinned       INTEGER NOT NULL DEFAULT 0,
  pin_name     TEXT,
  created_at   REAL NOT NULL,
  PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS idx_runs_state  ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run  ON run_events (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics (run_id, key, step);
CREATE INDEX IF NOT EXISTS idx_ckpt_run    ON checkpoints (run_id, step);
"#;

/// Length of generated run ids (hex chars of a UUID4).
const RUN_ID_LEN: usize = 12;

fn now() -> UnixSeconds {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

fn enum_to_string<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("unit enum serialises to a string")
}

fn enum_from_string<T: serde::de::DeserializeOwned>(raw: String) -> Result<T> {
    Ok(serde_json::from_value(serde_json::Value::String(raw))?)
}

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
        row.map(worker_from_row).transpose()
    }

    async fn fleet(&self) -> Result<Vec<WorkerInfo>> {
        let rows = sqlx::query("SELECT * FROM workers ORDER BY joined_at")
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter().map(worker_from_row).collect()
    }

    // ---- runs -------------------------------------------------------------

    async fn create_run(&self, name: &str, spec: &RunSpec) -> Result<RunId> {
        let run_id = RunId(uuid::Uuid::new_v4().simple().to_string()[..RUN_ID_LEN].to_owned());
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
}

/// Stride-downsample to at most `max` points, always keeping the last point so
/// the latest step is never dropped.
fn downsample_in_place(points: &mut Vec<MetricPoint>, max: usize) {
    if max == 0 || points.len() <= max {
        return;
    }
    let stride = points.len().div_ceil(max);
    let last = points.last().copied();
    let mut kept: Vec<MetricPoint> = points.iter().step_by(stride).copied().collect();
    if let Some(last) = last {
        if kept.last() != Some(&last) {
            kept.push(last);
        }
    }
    *points = kept;
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
    })
}

fn merge_field(value: &mut serde_json::Value, key: &str, field: serde_json::Value) {
    if !value.is_object() {
        *value = serde_json::json!({});
    }
    value
        .as_object_mut()
        .expect("just ensured object")
        .insert(key.to_owned(), field);
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
