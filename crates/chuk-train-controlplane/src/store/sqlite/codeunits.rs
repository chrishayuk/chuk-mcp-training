//! SqliteStore — [`CodeUnitStore`] impl (code units).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl CodeUnitStore for SqliteStore {
    async fn register_code_unit(
        &self,
        code: &CodeRef,
        manifest: &CodeUnitManifest,
        uri: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO code_units (name, sha, manifest, uri, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(name, sha) DO UPDATE SET manifest = excluded.manifest, uri = excluded.uri",
        )
        .bind(&code.name)
        .bind(&code.sha)
        .bind(serde_json::to_string(manifest)?)
        .bind(uri)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn code_unit(&self, name: &str, sha: &str) -> Result<Option<CodeUnitInfo>> {
        let row = sqlx::query("SELECT * FROM code_units WHERE name = ?1 AND sha = ?2")
            .bind(name)
            .bind(sha)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| {
            Ok(CodeUnitInfo {
                code: CodeRef {
                    name: r.get("name"),
                    sha: r.get("sha"),
                },
                manifest: serde_json::from_str(&r.get::<String, _>("manifest"))?,
                uri: r.get("uri"),
                created_at: r.get("created_at"),
            })
        })
        .transpose()
    }
}
