//! SqliteStore — [`WorkerTokenStore`] impl (worker tokens).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl WorkerTokenStore for SqliteStore {
    async fn create_worker_token(
        &self,
        id: &str,
        worker_id: &WorkerId,
        name: &str,
        prefix: &str,
        token_hash: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO worker_tokens (id, worker_id, name, prefix, token_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .bind(id)
        .bind(&worker_id.0)
        .bind(name)
        .bind(prefix)
        .bind(token_hash)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn resolve_worker_token(&self, token_hash: &str) -> Result<Option<WorkerTokenInfo>> {
        let row = sqlx::query(
            "SELECT id, worker_id, name, prefix, created_at, last_used_at, revoked_at
             FROM worker_tokens WHERE token_hash = ?1 AND revoked_at IS NULL",
        )
        .bind(token_hash)
        .fetch_optional(&self.pool)
        .await?;
        row.map(worker_token_from_row).transpose()
    }

    async fn list_worker_tokens(&self) -> Result<Vec<WorkerTokenInfo>> {
        let rows = sqlx::query(
            "SELECT id, worker_id, name, prefix, created_at, last_used_at, revoked_at
             FROM worker_tokens ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(worker_token_from_row).collect()
    }

    async fn revoke_worker_token(&self, id: &str) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE worker_tokens SET revoked_at = ?2 WHERE id = ?1 AND revoked_at IS NULL",
        )
        .bind(id)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn touch_worker_token(&self, id: &str, at: UnixSeconds) -> Result<()> {
        sqlx::query("UPDATE worker_tokens SET last_used_at = ?2 WHERE id = ?1")
            .bind(id)
            .bind(at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn create_join_token(
        &self,
        id: &str,
        worker_id: &WorkerId,
        token_hash: &str,
        expires_at: UnixSeconds,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO join_tokens (id, worker_id, token_hash, created_at, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .bind(id)
        .bind(&worker_id.0)
        .bind(token_hash)
        .bind(now())
        .bind(expires_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn resolve_join_token(
        &self,
        token_hash: &str,
        at: UnixSeconds,
    ) -> Result<Option<(WorkerId, bool)>> {
        // First use: atomically consume an unused, unexpired token.
        let consumed = sqlx::query(
            "UPDATE join_tokens SET used_at = ?2
             WHERE token_hash = ?1 AND used_at IS NULL AND expires_at > ?2
             RETURNING worker_id",
        )
        .bind(token_hash)
        .bind(at)
        .fetch_optional(&self.pool)
        .await?;
        if let Some(row) = consumed {
            return Ok(Some((WorkerId(row.get("worker_id")), true)));
        }
        // Already consumed: a reconnect candidate for its bound id only. An
        // unused-but-expired token has used_at NULL and lands on None here.
        let row = sqlx::query(
            "SELECT worker_id FROM join_tokens WHERE token_hash = ?1 AND used_at IS NOT NULL",
        )
        .bind(token_hash)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| (WorkerId(r.get("worker_id")), false)))
    }
}
