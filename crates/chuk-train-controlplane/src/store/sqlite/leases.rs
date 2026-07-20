//! SqliteStore — [`LeaseStore`] impl (leases).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl LeaseStore for SqliteStore {
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
}
