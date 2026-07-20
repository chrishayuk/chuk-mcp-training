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

    async fn set_budget(&self, budget: &Budget) -> Result<()> {
        sqlx::query(
            "INSERT INTO budgets (scope, cap, period, updated_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(scope) DO UPDATE SET cap = ?2, period = ?3, updated_at = ?4",
        )
        .bind(&budget.scope)
        .bind(budget.cap)
        .bind(&budget.period)
        .bind(budget.updated_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn delete_budget(&self, scope: &str) -> Result<bool> {
        let done = sqlx::query("DELETE FROM budgets WHERE scope = ?1")
            .bind(scope)
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    async fn budgets(&self) -> Result<Vec<Budget>> {
        let rows = sqlx::query("SELECT * FROM budgets ORDER BY scope")
            .fetch_all(&self.pool)
            .await?;
        Ok(rows
            .into_iter()
            .map(|r| Budget {
                scope: r.get("scope"),
                cap: r.get("cap"),
                period: r.get("period"),
                updated_at: r.get("updated_at"),
            })
            .collect())
    }
}
