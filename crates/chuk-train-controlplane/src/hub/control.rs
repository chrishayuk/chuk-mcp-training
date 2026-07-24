//! Hub — run lifecycle control: gate evaluation (watchdogs, spec §6/§8),
//! cancel, and resume.

use super::*;

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

impl Hub {
    /// Evaluate every gate on a run against its metric history, persist the
    /// verdicts, and act on newly-tripped ones (spec §6/§8): a
    /// `gate_evaluated` run event, plus — for `stop_run` watchdogs — the stop
    /// path (the worker's SIGTERM grace is the trainer's checkpoint window).
    /// Runs on metric ingest and on `check_gates` reads, so verdicts stay
    /// current even when a run stops emitting metrics. Returns the refreshed
    /// gates. Events fire only on a verdict *flip* to tripped, not per batch.
    pub async fn evaluate_gates(&self, run_id: &RunId) -> Result<Vec<GateInfo>> {
        let gates = self.store.gates(GATE_SCOPE_RUN, &run_id.0).await?;
        let mut refreshed = Vec::with_capacity(gates.len());
        for info in gates {
            // Expressions are validated at registration; tolerate a bad stored
            // row (e.g. a grammar change) rather than poisoning ingest.
            let expr = match gate::parse(&info.expr) {
                Ok(expr) => expr,
                Err(reason) => {
                    warn!(run = %run_id.0, gate = %info.name, %reason, "unparseable stored gate");
                    refreshed.push(info);
                    continue;
                }
            };
            let history = self.store.metric_history(run_id, expr.key()).await?;
            let at = unix_now();
            let verdict = gate::evaluate(&expr, &history, at);
            let newly_tripped = verdict.tripped && info.tripped != Some(true);
            self.store
                .record_gate_result(
                    GATE_SCOPE_RUN,
                    &run_id.0,
                    &info.name,
                    verdict.tripped,
                    verdict.last_value,
                    &verdict.detail,
                    at,
                )
                .await?;
            if newly_tripped {
                self.store
                    .add_event(
                        run_id,
                        EventKind::GateEvaluated,
                        serde_json::json!({
                            "gate": info.name,
                            "expr": info.expr,
                            "tripped": true,
                            "action": info.action.as_str(),
                            "detail": verdict.detail,
                        }),
                    )
                    .await?;
            }
            // The event fires only on a verdict flip, but a stop must CONVERGE,
            // not fire once: a run resurrected after a lost kill report (a
            // flapped link replaying JobStarted — observed on Colab during EI0)
            // would otherwise outlive its verdict forever, because the latch
            // never flips again. Re-stop on every evaluation while tripped; the
            // state check keeps it quiet once the run is actually terminal.
            if verdict.tripped && info.action == GateAction::StopRun {
                let live = self
                    .store
                    .run(run_id)
                    .await?
                    .is_some_and(|run| !run.summary.state.is_terminal());
                if live {
                    warn!(run = %run_id.0, gate = %info.name, detail = %verdict.detail,
                          "watchdog tripped; stopping run");
                    if let Err(error) = self.stop_run(run_id).await {
                        warn!(run = %run_id.0, %error, "watchdog stop failed");
                    }
                }
            }
            refreshed.push(GateInfo {
                tripped: Some(verdict.tripped),
                last_value: verdict.last_value,
                evaluated_at: Some(at),
                detail: Some(verdict.detail),
                ..info
            });
        }
        Ok(refreshed)
    }

    /// Cancel a run. A running/assigned run with a still-connected worker is
    /// signalled (`Cancel` → the worker stops the process and reports
    /// `JobKilled{Cancel}`, which lands the run in `Cancelled`); a queued run, or
    /// one whose worker link is already gone, is finalised here and now. A
    /// terminal run is rejected.
    pub async fn stop_run(&self, run_id: &RunId) -> Result<()> {
        let run = self
            .store
            .run(run_id)
            .await?
            .with_context(|| format!("no such run: {}", run_id.0))?;
        if run.summary.state.is_terminal() {
            anyhow::bail!(
                "run {} is already {} — nothing to cancel",
                run_id.0,
                run.summary.state.as_str()
            );
        }
        // Prefer signalling the live worker so it can checkpoint/clean up; the
        // `JobKilled{Cancel}` it reports is what finalises the run.
        if let Some(worker_id) = run.summary.worker_id.clone() {
            let job_id = wire::JobId::from(run_id.0.clone());
            if self.send_to(&worker_id, wire::CpToWorker::Cancel { job_id }).await {
                info!(run = %run_id.0, worker = %worker_id, "cancel signalled to worker");
                return Ok(());
            }
        }
        // Queued (no worker), or the worker vanished: finalise directly.
        self.cancel_now(run_id, run.summary.worker_id.as_ref()).await
    }

    /// Finalise a run to `Cancelled` without a worker round-trip.
    async fn cancel_now(&self, run_id: &RunId, worker_id: Option<&WorkerId>) -> Result<()> {
        self.grants.revoke_run(run_id);
        self.store
            .transition(
                run_id,
                RunState::Cancelled,
                None,
                Some(EXIT_CODE_CANCELLED),
                serde_json::json!({ "reason": "operator_cancel" }),
            )
            .await?;
        self.mirror_state(run_id, RunState::Cancelled);
        self.resume_state
            .lock()
            .expect("resume_state lock")
            .remove(run_id);
        if let Some(worker_id) = worker_id {
            self.store.set_worker_run(worker_id, None).await?;
        }
        self.pump().await
    }

    /// Re-queue a terminal run so it runs again. On reassignment a train run
    /// resumes from its latest uploaded checkpoint (a shell run restarts); a
    /// non-terminal run is rejected.
    pub async fn resume_run(&self, run_id: &RunId) -> Result<()> {
        let run = self
            .store
            .run(run_id)
            .await?
            .with_context(|| format!("no such run: {}", run_id.0))?;
        if !run.summary.state.is_terminal() {
            anyhow::bail!(
                "run {} is {} — only a terminal run can be resumed",
                run_id.0,
                run.summary.state.as_str()
            );
        }
        self.store
            .transition(
                run_id,
                RunState::Queued,
                None,
                None,
                serde_json::json!({ "reason": "operator_resume" }),
            )
            .await?;
        info!(run = %run_id.0, "run re-queued for resume");
        self.pump().await
    }
}
