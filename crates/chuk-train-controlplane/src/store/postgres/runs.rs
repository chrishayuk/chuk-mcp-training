//! PgStore — [`RunStore`] impl (runs).

use super::*;
use crate::store::prelude::*;
use crate::store::RunQuery;

#[async_trait]
impl RunStore for PgStore {
    async fn next_run_seq(&self) -> Result<i64> {
        // nextval() is atomic across sessions (even through Neon's pooler), so
        // the execution sequence stays gap-tolerant but never collides.
        let row = sqlx::query("SELECT nextval('exec_ref_seq') AS value")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get::<i64, _>("value"))
    }

    async fn set_experiments_run_id(&self, run_id: &RunId, ext_run_id: &str) -> Result<()> {
        sqlx::query("UPDATE runs SET experiments_run_id = $2 WHERE id = $1")
            .bind(&run_id.0)
            .bind(ext_run_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn experiments_run_id(&self, run_id: &RunId) -> Result<Option<String>> {
        let row = sqlx::query("SELECT experiments_run_id FROM runs WHERE id = $1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.and_then(|r| r.get::<Option<String>, _>("experiments_run_id")))
    }

    async fn enqueue_outbox_event(
        &self,
        run_id: &RunId,
        kind: &str,
        payload: &str,
        at: UnixSeconds,
    ) -> Result<i64> {
        let row = sqlx::query(
            "INSERT INTO experiments_outbox (run_id, kind, payload, created_at, next_attempt_at)
             VALUES ($1, $2, $3, $4, $4)
             RETURNING id",
        )
        .bind(&run_id.0)
        .bind(kind)
        .bind(payload)
        .bind(at)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("id"))
    }

    async fn due_outbox_events(&self, at: UnixSeconds, limit: i64) -> Result<Vec<OutboxRow>> {
        let rows = sqlx::query(
            "SELECT id, run_id, kind, payload, attempts FROM experiments_outbox
             WHERE completed_at IS NULL AND next_attempt_at <= $1
             ORDER BY created_at ASC
             LIMIT $2",
        )
        .bind(at)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| OutboxRow {
                id: r.get::<i64, _>("id"),
                run_id: RunId(r.get::<String, _>("run_id")),
                kind: r.get::<String, _>("kind"),
                payload: r.get::<String, _>("payload"),
                attempts: r.get::<i64, _>("attempts"),
            })
            .collect())
    }

    async fn mark_outbox_event_done(&self, id: i64) -> Result<()> {
        sqlx::query("UPDATE experiments_outbox SET completed_at = $2 WHERE id = $1")
            .bind(id)
            .bind(now())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn mark_outbox_event_failed(
        &self,
        id: i64,
        error: &str,
        next_attempt_at: UnixSeconds,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE experiments_outbox
             SET attempts = attempts + 1, last_error = $2, next_attempt_at = $3
             WHERE id = $1",
        )
        .bind(id)
        .bind(error)
        .bind(next_attempt_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn create_run(
        &self,
        name: &str,
        spec: &RunSpec,
        experiment_ref: Option<&str>,
        created_by: Option<&str>,
        sweep_id: Option<&str>,
    ) -> Result<RunId> {
        let run_id = RunId(new_run_id(now(), self.next_run_seq().await?));
        sqlx::query(
            "INSERT INTO runs (id, name, kind, spec, state, experiment_ref, created_by, sweep_id, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)",
        )
        .bind(&run_id.0)
        .bind(name)
        .bind(spec.kind_str())
        .bind(serde_json::to_string(spec)?)
        .bind(RunState::Queued.as_str())
        .bind(experiment_ref)
        .bind(created_by)
        .bind(sweep_id)
        .bind(now())
        .execute(&self.pool)
        .await?;
        self.add_event(
            &run_id,
            EventKind::Created,
            serde_json::json!({ "name": name }),
        )
        .await?;
        self.add_event(&run_id, EventKind::Queued, serde_json::Value::Null)
            .await?;
        Ok(run_id)
    }

    async fn transition(
        &self,
        run_id: &RunId,
        state: RunState,
        worker_id: Option<&WorkerId>,
        exit_code: Option<i64>,
        detail: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE runs SET state = $1, updated_at = $2,
               worker_id = COALESCE($3, worker_id),
               exit_code = COALESCE($4, exit_code)
             WHERE id = $5",
        )
        .bind(state.as_str())
        .bind(now())
        .bind(worker_id.map(|w| w.0.as_str()))
        .bind(exit_code)
        .bind(&run_id.0)
        .execute(&self.pool)
        .await?;

        let mut event_detail = detail;
        if let Some(w) = worker_id {
            merge_field(&mut event_detail, "worker", serde_json::json!(w.0));
        }
        if let Some(code) = exit_code {
            merge_field(&mut event_detail, "exit_code", serde_json::json!(code));
        }
        self.add_event(run_id, EventKind::from(state), event_detail)
            .await
    }

    async fn next_queued(&self) -> Result<Option<RunRecord>> {
        let row = sqlx::query("SELECT * FROM runs WHERE state = $1 ORDER BY created_at LIMIT 1")
            .bind(RunState::Queued.as_str())
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| run_from_row(&r)).transpose()
    }

    async fn run(&self, run_id: &RunId) -> Result<Option<RunRecord>> {
        let row = sqlx::query("SELECT * FROM runs WHERE id = $1")
            .bind(&run_id.0)
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| run_from_row(&r)).transpose()
    }

    async fn runs(&self, query: &RunQuery, limit: u32) -> Result<Vec<RunSummary>> {
        // Postgres has no unsigned integers, so LIMIT/OFFSET are bound as i64;
        // the `$n::text` casts keep NULL binds typed for the null-means-any filters.
        let rows = sqlx::query(
            "SELECT * FROM runs
             WHERE ($1::text IS NULL OR state = $1)
               AND ($2::text IS NULL OR experiment_ref = $2)
               AND ($3::text IS NULL OR sweep_id = $3)
             ORDER BY created_at DESC, id DESC
             LIMIT $4 OFFSET $5",
        )
        .bind(query.state.map(RunState::as_str))
        .bind(query.experiment_ref.as_deref())
        .bind(query.sweep_id.as_deref())
        .bind(limit as i64)
        .bind(query.offset as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| Ok(run_from_row(r)?.summary))
            .collect()
    }
}
