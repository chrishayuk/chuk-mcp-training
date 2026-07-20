//! PgStore — [`WorkerStore`] impl (workers).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl WorkerStore for PgStore {
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

    async fn worker_is_persistent(&self, id: &WorkerId) -> Result<bool> {
        let row = sqlx::query(
            "SELECT 1 FROM worker_tokens WHERE worker_id = $1 AND revoked_at IS NULL LIMIT 1",
        )
        .bind(&id.0)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.is_some())
    }

    async fn record_worker_samples(
        &self,
        worker_id: &WorkerId,
        values: &BTreeMap<String, f64>,
    ) -> Result<()> {
        let at = now();
        sqlx::query("INSERT INTO worker_samples (worker_id, ts, payload) VALUES ($1, $2, $3)")
            .bind(&worker_id.0)
            .bind(at)
            .bind(serde_json::to_string(values)?)
            .execute(&self.pool)
            .await?;
        // Bound the history to the sparkline window, pruned on write.
        sqlx::query("DELETE FROM worker_samples WHERE worker_id = $1 AND ts < $2")
            .bind(&worker_id.0)
            .bind(at - WORKER_SAMPLE_RETENTION.as_secs_f64())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn worker_telemetry(&self, worker_id: &WorkerId) -> Result<Option<WorkerTelemetry>> {
        let rows =
            sqlx::query("SELECT ts, payload FROM worker_samples WHERE worker_id = $1 ORDER BY ts")
                .bind(&worker_id.0)
                .fetch_all(&self.pool)
                .await?;
        let samples = rows
            .into_iter()
            .map(|r| Ok((r.get::<f64, _>("ts"), r.get::<String, _>("payload"))))
            .collect::<Result<Vec<_>>>()?;
        worker_telemetry_from_samples(worker_id, samples)
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
        // Latest sys/* sample per worker (one query), for the fleet's live column.
        let latest = sqlx::query(
            "SELECT worker_id, payload FROM worker_samples s
             WHERE ts = (SELECT MAX(ts) FROM worker_samples WHERE worker_id = s.worker_id)",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut telemetry: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for row in &latest {
            telemetry.insert(
                row.get("worker_id"),
                serde_json::from_str(&row.get::<String, _>("payload"))?,
            );
        }
        for worker in &mut workers {
            worker.lease = self.lease(&worker.id).await?;
            worker.telemetry = telemetry.remove(&worker.id.0);
        }
        Ok(workers)
    }
}
