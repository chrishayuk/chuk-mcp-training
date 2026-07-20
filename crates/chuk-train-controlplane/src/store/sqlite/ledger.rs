//! SqliteStore — [`LedgerStore`] impl (ledger).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl LedgerStore for SqliteStore {
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
}
