//! SqliteStore — [`MetricStore`] impl (metrics).

use super::*;
use crate::store::prelude::*;

#[async_trait]
impl MetricStore for SqliteStore {
    async fn append_metrics(
        &self,
        run_id: &RunId,
        step: u64,
        values: &BTreeMap<String, f64>,
    ) -> Result<()> {
        let ts = now();
        let step = step as i64;
        let mut tx = self.pool.begin().await?;
        for (key, value) in values {
            sqlx::query(
                "INSERT INTO metrics (run_id, step, key, value, ts) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(run_id, key, step) DO UPDATE SET value = excluded.value",
            )
            .bind(&run_id.0)
            .bind(step)
            .bind(key)
            .bind(value)
            .bind(ts)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn metric_series(
        &self,
        run_id: &RunId,
        keys: Option<&[String]>,
        since_step: u64,
        downsample: u32,
    ) -> Result<MetricSeries> {
        // Filtering by key list is done in Rust: the set is tiny and this
        // avoids building a dynamic `IN (?, ?, ...)` clause.
        let rows = sqlx::query(
            "SELECT step, key, value FROM metrics
             WHERE run_id = ?1 AND step >= ?2 ORDER BY key, step",
        )
        .bind(&run_id.0)
        .bind(since_step as i64)
        .fetch_all(&self.pool)
        .await?;

        let wanted: Option<std::collections::HashSet<&str>> =
            keys.map(|ks| ks.iter().map(String::as_str).collect());
        let mut series: BTreeMap<String, Vec<MetricPoint>> = BTreeMap::new();
        for row in rows {
            let key: String = row.get("key");
            if wanted.as_ref().is_some_and(|w| !w.contains(key.as_str())) {
                continue;
            }
            series.entry(key).or_default().push(MetricPoint {
                step: row.get::<i64, _>("step") as u64,
                value: row.get("value"),
            });
        }
        if downsample > 0 {
            for points in series.values_mut() {
                downsample_in_place(points, downsample as usize);
            }
        }
        Ok(MetricSeries {
            run_id: run_id.clone(),
            series,
        })
    }
}
