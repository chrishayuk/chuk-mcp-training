//! SqliteStore — [`AuthStore`] impl (users & teams).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl AuthStore for SqliteStore {
    async fn ensure_team(&self, id: &str, name: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO teams (id, name, created_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO NOTHING",
        )
        .bind(id)
        .bind(name)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn upsert_user(&self, email: &str, team_id: &str, role: Role) -> Result<()> {
        sqlx::query(
            "INSERT INTO users (email, team_id, role, created_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(email) DO UPDATE SET team_id = excluded.team_id, role = excluded.role",
        )
        .bind(email)
        .bind(team_id)
        .bind(role.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn get_user(&self, email: &str) -> Result<Option<User>> {
        let row =
            sqlx::query("SELECT email, team_id, role, created_at FROM users WHERE email = ?1")
                .bind(email)
                .fetch_optional(&self.pool)
                .await?;
        row.map(user_from_row).transpose()
    }

    async fn list_users(&self, team_id: &str) -> Result<Vec<User>> {
        let rows = sqlx::query(
            "SELECT email, team_id, role, created_at FROM users WHERE team_id = ?1 ORDER BY email",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(user_from_row).collect()
    }

    async fn remove_user(&self, email: &str) -> Result<()> {
        sqlx::query("DELETE FROM users WHERE email = ?1")
            .bind(email)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn set_user_experiments_key(&self, email: &str, encrypted: Option<&str>) -> Result<()> {
        // Upsert: a caller linking their own key may not have a `users` row yet
        // (e.g. a first-time session sign-in defaults to Role::Read without
        // one — see `resolve_auth`) — an UPDATE-only query would silently
        // no-op for them while still reporting success.
        sqlx::query(
            "INSERT INTO users (email, team_id, role, created_at, experiments_api_key_encrypted)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(email) DO UPDATE SET experiments_api_key_encrypted = excluded.experiments_api_key_encrypted",
        )
        .bind(email)
        .bind(DEFAULT_TEAM_ID)
        .bind(Role::Read.as_str())
        .bind(now())
        .bind(encrypted)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn user_experiments_key(&self, email: &str) -> Result<Option<String>> {
        let row = sqlx::query("SELECT experiments_api_key_encrypted FROM users WHERE email = ?1")
            .bind(email)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.and_then(|r| r.get::<Option<String>, _>("experiments_api_key_encrypted")))
    }

    async fn create_api_key(
        &self,
        id: &str,
        team_id: &str,
        created_by: &str,
        name: &str,
        prefix: &str,
        key_hash: &str,
        role: Role,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO api_keys (id, team_id, created_by, name, prefix, key_hash, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(id)
        .bind(team_id)
        .bind(created_by)
        .bind(name)
        .bind(prefix)
        .bind(key_hash)
        .bind(role.as_str())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn list_api_keys(&self, team_id: &str) -> Result<Vec<ApiKeyInfo>> {
        let rows = sqlx::query(
            "SELECT id, team_id, created_by, name, prefix, role, created_at, last_used_at, revoked_at
             FROM api_keys WHERE team_id = ?1 ORDER BY created_at DESC",
        )
        .bind(team_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(api_key_from_row).collect()
    }

    async fn revoke_api_key(&self, id: &str) -> Result<bool> {
        let result =
            sqlx::query("UPDATE api_keys SET revoked_at = ?2 WHERE id = ?1 AND revoked_at IS NULL")
                .bind(id)
                .bind(now())
                .execute(&self.pool)
                .await?;
        Ok(result.rows_affected() > 0)
    }

    async fn resolve_api_key(&self, key_hash: &str) -> Result<Option<ApiKeyInfo>> {
        let row = sqlx::query(
            "SELECT id, team_id, created_by, name, prefix, role, created_at, last_used_at, revoked_at
             FROM api_keys WHERE key_hash = ?1 AND revoked_at IS NULL",
        )
        .bind(key_hash)
        .fetch_optional(&self.pool)
        .await?;
        row.map(api_key_from_row).transpose()
    }

    async fn touch_api_key(&self, id: &str, at: UnixSeconds) -> Result<()> {
        sqlx::query("UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1")
            .bind(id)
            .bind(at)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
