//! SQLite adapter (sqlx, async native).

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{
    EventKind, Hardware, RunEvent, RunId, RunRecord, RunSpec, RunState, RunSummary, UnixSeconds,
    WorkerId, WorkerInfo, WorkerState,
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
CREATE INDEX IF NOT EXISTS idx_runs_state  ON runs (state, created_at);
CREATE INDEX IF NOT EXISTS idx_events_run  ON run_events (run_id, seq);
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
