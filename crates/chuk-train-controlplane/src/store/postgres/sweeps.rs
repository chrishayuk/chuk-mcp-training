//! PgStore — [`SweepStore`] impl (sweeps).

use super::*;
use crate::store::prelude::*;
use crate::store::SweepRow;
use super::super::ids::new_sweep_id;

#[async_trait]
impl SweepStore for PgStore {
    async fn create_sweep(
        &self,
        name: &str,
        template: &str,
        axes: &str,
        concurrency: u32,
        created_by: Option<&str>,
    ) -> Result<String> {
        let sweep_id = new_sweep_id(now(), self.next_run_seq().await?);
        sqlx::query(
            "INSERT INTO sweeps (id, name, template, axes, concurrency, created_by, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(&sweep_id)
        .bind(name)
        .bind(template)
        .bind(axes)
        .bind(concurrency as i64)
        .bind(created_by)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(sweep_id)
    }

    async fn sweep(&self, sweep_id: &str) -> Result<Option<SweepRow>> {
        let row = sqlx::query(
            "SELECT id, name, template, axes, concurrency, created_by, created_at
             FROM sweeps WHERE id = $1",
        )
        .bind(sweep_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| SweepRow {
            id: r.get("id"),
            name: r.get("name"),
            template: r.get("template"),
            axes: r.get("axes"),
            concurrency: r.get::<i64, _>("concurrency") as u32,
            created_at: r.get("created_at"),
            created_by: r.get("created_by"),
        }))
    }

    async fn sweep_active_children(&self, sweep_id: &str) -> Result<u32> {
        let row = sqlx::query(
            "SELECT COUNT(*) AS n FROM runs WHERE sweep_id = $1 AND state IN ($2, $3)",
        )
        .bind(sweep_id)
        .bind(RunState::Assigned.as_str())
        .bind(RunState::Running.as_str())
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("n") as u32)
    }
}
