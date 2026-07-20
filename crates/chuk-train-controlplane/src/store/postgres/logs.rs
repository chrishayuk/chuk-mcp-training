//! PgStore — [`RunLogStore`] impl (logs & events).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl RunLogStore for PgStore {
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
}
