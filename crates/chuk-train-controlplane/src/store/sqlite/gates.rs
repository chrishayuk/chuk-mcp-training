//! SqliteStore — [`GateStore`] impl (gates + watchdogs).

use super::*;
use crate::store::prelude::*;
use crate::store::MetricObservation;
use chuk_train_proto::{GateAction, GateInfo};

fn gate_from_row(r: sqlx::sqlite::SqliteRow) -> Result<GateInfo> {
    Ok(GateInfo {
        scope: r.get("scope"),
        scope_id: r.get("scope_id"),
        name: r.get("name"),
        expr: r.get("expr"),
        action: enum_from_string(r.get::<String, _>("action"))?,
        created_at: r.get("created_at"),
        tripped: r.get::<Option<bool>, _>("tripped"),
        last_value: r.get("last_value"),
        evaluated_at: r.get("evaluated_at"),
        detail: r.get("detail"),
    })
}

#[async_trait]
impl GateStore for SqliteStore {
    async fn register_gate(
        &self,
        scope: &str,
        scope_id: &str,
        name: &str,
        expr: &str,
        action: GateAction,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO gates (scope, scope_id, name, expr, action, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(scope, scope_id, name) DO UPDATE SET
               expr = ?4, action = ?5, created_at = ?6,
               tripped = NULL, last_value = NULL, evaluated_at = NULL, detail = NULL",
        )
        .bind(scope)
        .bind(scope_id)
        .bind(name)
        .bind(expr)
        .bind(enum_to_string(&action))
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn gates(&self, scope: &str, scope_id: &str) -> Result<Vec<GateInfo>> {
        let rows = sqlx::query(
            "SELECT * FROM gates WHERE scope = ?1 AND scope_id = ?2 ORDER BY name",
        )
        .bind(scope)
        .bind(scope_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(gate_from_row).collect()
    }

    async fn record_gate_result(
        &self,
        scope: &str,
        scope_id: &str,
        name: &str,
        tripped: bool,
        last_value: Option<f64>,
        detail: &str,
        at: UnixSeconds,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE gates SET tripped = ?4, last_value = ?5, evaluated_at = ?6, detail = ?7
             WHERE scope = ?1 AND scope_id = ?2 AND name = ?3",
        )
        .bind(scope)
        .bind(scope_id)
        .bind(name)
        .bind(tripped)
        .bind(last_value)
        .bind(at)
        .bind(detail)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn metric_history(
        &self,
        run_id: &RunId,
        key: &str,
    ) -> Result<Vec<MetricObservation>> {
        let rows = sqlx::query(
            "SELECT ts, value FROM metrics WHERE run_id = ?1 AND key = ?2 ORDER BY step",
        )
        .bind(&run_id.0)
        .bind(key)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| MetricObservation {
                ts: r.get("ts"),
                // SQLite has no NaN: binding one stores NULL. Reading NULL back
                // as NaN keeps `isnan(last(...))` gates able to see it.
                value: r.get::<Option<f64>, _>("value").unwrap_or(f64::NAN),
            })
            .collect())
    }
}
