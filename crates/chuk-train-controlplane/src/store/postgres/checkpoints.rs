//! PgStore — [`CheckpointStore`] impl (checkpoints).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl CheckpointStore for PgStore {
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
}
