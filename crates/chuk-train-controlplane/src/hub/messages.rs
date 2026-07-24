//! Hub — messages from workers: the `on_message` dispatcher, terminal-state
//! finalisation, and checkpoint-artifact ingestion (the lineage merge a
//! worker used to do, now the control plane's job).

use super::*;

/// The artifact class the control plane records checkpoint outputs under (the
/// value it stamps into each train job's checkpoint [`wire::OutputRule`]).
const ARTIFACT_CHECKPOINT: &str = "checkpoint";

impl Hub {
    pub async fn on_message(&self, worker_id: &WorkerId, msg: wire::WorkerToCp) -> Result<()> {
        self.store.worker_seen(worker_id).await?;
        // Deduplicate replayed streamed events (M3.2): a reconnecting worker
        // re-sends its outbox; anything at or below the high-water is already
        // applied. Non-streamed messages (Hello/Heartbeat/Drained) have no seq.
        if let Some(seq) = event_seq(&msg) {
            let mut hw = self.high_water.lock().expect("high_water lock");
            if hw.get(worker_id).is_some_and(|&cur| seq <= cur) {
                debug!(worker = %worker_id, seq, "duplicate streamed event (replay); dropping");
                return Ok(());
            }
            hw.insert(worker_id.clone(), seq);
        }
        match msg {
            wire::WorkerToCp::Hello { .. } => {
                debug!(worker = %worker_id, "duplicate hello ignored");
            }
            wire::WorkerToCp::Heartbeat => {}
            wire::WorkerToCp::Log { job_id, line, .. } => {
                self.store.append_log(&run_id(&job_id), &line).await?;
            }
            wire::WorkerToCp::Metric {
                job_id, step, values, ..
            } => match job_id {
                // A job's own metric: the (job, step)-indexed training series.
                Some(job_id) => {
                    if let Some(step) = step {
                        let rid = run_id(&job_id);
                        self.store.append_metrics(&rid, step, &values).await?;
                        // Gate evaluation rides ingest; a policy failure must
                        // never fail the metric write itself.
                        if let Err(error) = self.evaluate_gates(&rid).await {
                            warn!(run = %rid.0, %error, "gate evaluation failed");
                        }
                    }
                }
                // Host telemetry (chuk-compute M4): `sys/*` samples carry no job
                // or step — the latest sample per worker, for the live dashboard.
                None => {
                    if !values.is_empty() {
                        self.store.record_worker_samples(worker_id, &values).await?;
                    }
                }
            },
            wire::WorkerToCp::Artifact {
                job_id, class, uri, ..
            } => {
                if class.as_str() == ARTIFACT_CHECKPOINT {
                    self.ingest_checkpoint(&run_id(&job_id), &uri).await?;
                } else {
                    debug!(class = %class, uri = %uri, "artifact of unhandled class; ignoring");
                }
            }
            wire::WorkerToCp::JobStarted { job_id, .. } => {
                let run_id = run_id(&job_id);
                // A flapped link can replay JobStarted for a job the CP has
                // already finalised (observed on Colab during EI0 — see the
                // watchdog convergence note in control.rs::evaluate_gates).
                // seq dedup only catches replays at-or-below the high-water;
                // a genuinely new JobStarted for an already-terminal run must
                // not resurrect it. A separate read-then-transition here would
                // leave a window where a concurrent JobExited/JobKilled lands
                // between the two, so the liveness check and the write must be
                // one atomic store operation (transition_if_live), not two.
                if self
                    .store
                    .transition_if_live(&run_id, RunState::Running, Some(worker_id))
                    .await?
                {
                    self.mirror_state(&run_id, RunState::Running);
                } else {
                    warn!(run = %run_id.0, %worker_id, "JobStarted for terminal run; ignoring (stale/replayed)");
                }
            }
            wire::WorkerToCp::JobExited { job_id, code, .. } => {
                let state = if code == 0 {
                    RunState::Completed
                } else {
                    RunState::Failed
                };
                self.finish_run(worker_id, &run_id(&job_id), state, code).await?;
            }
            wire::WorkerToCp::JobKilled { job_id, reason, .. } => {
                info!(run = %run_id(&job_id), ?reason, "job killed");
                let (state, code) = kill_reason_state(&reason);
                self.finish_run(worker_id, &run_id(&job_id), state, code).await?;
            }
            wire::WorkerToCp::ServiceReady { job_id, ports, .. } => {
                // Services land at M5; for now just note readiness.
                debug!(run = %run_id(&job_id), ?ports, "service ready (unhandled until M5)");
            }
            wire::WorkerToCp::Drained => {
                // The worker has flushed + uploaded and stopped work. The lease
                // manager still owns the provider-verified destroy at T-0 — the
                // wall never depends on this message arriving.
                info!(worker = %worker_id, "worker drained");
            }
            // The protocol is #[non_exhaustive]; a variant this build doesn't
            // know is tolerated (forward compatibility, spec §3).
            other => debug!(worker = %worker_id, ?other, "unhandled worker message"),
        }
        Ok(())
    }

    /// Common terminal-state handling: revoke the grant, transition, mirror,
    /// forget resume bookkeeping, free the worker, and re-pump.
    async fn finish_run(
        &self,
        worker_id: &WorkerId,
        run_id: &RunId,
        state: RunState,
        code: i64,
    ) -> Result<()> {
        self.grants.revoke_run(run_id);
        self.store
            .transition(run_id, state, Some(worker_id), Some(code), Value::Null)
            .await?;
        self.mirror_state(run_id, state);
        self.resume_state
            .lock()
            .expect("resume_state lock")
            .remove(run_id);
        self.store.set_worker_run(worker_id, None).await?;
        self.pump().await
    }

    /// Interpret a checkpoint-class artifact: read back its model + partial
    /// sidecar from the store, complete the lineage (the merge a worker used to
    /// do), rewrite the sidecar, and record it.
    async fn ingest_checkpoint(&self, run_id: &RunId, uri: &str) -> Result<()> {
        let Some(step) = step_from_uri(uri) else {
            warn!(uri, "checkpoint artifact uri has no step_<n> segment; ignoring");
            return Ok(());
        };
        let model = self
            .artifacts
            .get(&keys::checkpoint_file(&run_id.0, step, CHECKPOINT_MODEL_FILE))
            .await
            .with_context(|| format!("reading checkpoint step_{step} model"))?;
        let model_hash = hex::encode(Sha256::digest(&model));

        let meta = self.complete_meta(run_id, step).await?;
        // Rewrite the sidecar so retrieval + lazarus see complete lineage.
        self.artifacts
            .put(
                &keys::checkpoint_file(&run_id.0, step, CHECKPOINT_META_FILE),
                serde_json::to_vec_pretty(&meta)?,
            )
            .await?;

        let uri = self.artifacts.uri(&keys::checkpoint_dir(&run_id.0, step));
        self.store
            .record_checkpoint(run_id, step, &uri, &model_hash, &meta)
            .await?;
        self.store
            .add_event(
                run_id,
                EventKind::Checkpoint,
                serde_json::json!({ "step": step, "uri": uri, "model_hash": model_hash }),
            )
            .await?;
        self.mirror_checkpoint(run_id, step, &uri, &meta);
        info!(run = %run_id, step, "checkpoint recorded");
        Ok(())
    }

    /// Merge the trainer's partial sidecar with control-plane-known lineage:
    /// run id, code, seed, arch, parent, and the slice history.
    pub(super) async fn complete_meta(&self, run_id: &RunId, step: u64) -> Result<CheckpointMeta> {
        let mut meta: CheckpointMeta = match self
            .artifacts
            .get(&keys::checkpoint_file(&run_id.0, step, CHECKPOINT_META_FILE))
            .await
        {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => CheckpointMeta::default(),
        };
        meta.step = step;
        meta.run_id = Some(run_id.clone());

        if let Some(RunSpec::Train(train)) = self.store.run(run_id).await?.map(|r| r.spec) {
            meta.code.get_or_insert_with(|| train.code.clone());
            if meta.seed.is_none() {
                meta.seed = train
                    .seed
                    .or_else(|| train.overrides.get("seed").and_then(Value::as_i64));
            }
            if meta.arch.is_none() {
                meta.arch = train.arch.clone();
            }
        }

        // Parent = the run's latest checkpoint recorded before this one.
        if meta.parent_checkpoint.is_none() {
            if let Some(prev) = self.store.latest_checkpoint(run_id).await? {
                if prev.step < step {
                    meta.parent_checkpoint = Some(keys::checkpoint_dir(&run_id.0, prev.step));
                }
            }
        }

        let ResumeState { from_step, base_slices } = self
            .resume_state
            .lock()
            .expect("resume_state lock")
            .get(run_id)
            .cloned()
            .unwrap_or_default();
        let mut slices = base_slices;
        slices.push([from_step, step]);
        meta.slices = slices;

        // Sourced from the resolved `data:` block, not the trainer's sidecar
        // (spec §7.3) — overrides rather than fills, since the harness's
        // resolve response is the authority, not a trainer-supplied guess.
        if let Some(DataState { content_sha, plan_sha, dataset_label }) =
            self.data_state.lock().expect("data_state lock").get(run_id).cloned()
        {
            meta.dataset_sha = Some(content_sha);
            meta.plan_sha = plan_sha;
            meta.datasets = vec![dataset_label];
        }
        Ok(meta)
    }
}

/// A [`wire::JobId`] is a run's id verbatim on the fabric.
pub(super) fn run_id(job_id: &wire::JobId) -> RunId {
    RunId(job_id.0.clone())
}

/// The `seq` of a streamed worker→CP event, or `None` for the non-streamed
/// messages (Hello / Heartbeat / Drained) that are never replayed.
fn event_seq(msg: &wire::WorkerToCp) -> Option<u64> {
    match msg {
        wire::WorkerToCp::JobStarted { seq, .. }
        | wire::WorkerToCp::JobExited { seq, .. }
        | wire::WorkerToCp::JobKilled { seq, .. }
        | wire::WorkerToCp::ServiceReady { seq, .. }
        | wire::WorkerToCp::Log { seq, .. }
        | wire::WorkerToCp::Artifact { seq, .. } => Some(*seq),
        // A job's own metric is part of the replayable stream; a host `sys/*`
        // sample (no job_id) is out-of-band telemetry — never outboxed or
        // deduped, so it must not participate in the seq high-water.
        wire::WorkerToCp::Metric {
            seq,
            job_id: Some(_),
            ..
        } => Some(*seq),
        _ => None,
    }
}

/// Parse the trailing `step_<n>` segment of a checkpoint uri, e.g.
/// `ckpt-hot/RUN/step_500` → `500`.
pub(super) fn step_from_uri(uri: &str) -> Option<u64> {
    uri.rsplit('/')
        .next()?
        .strip_prefix(CHECKPOINT_DIR_PREFIX)?
        .parse()
        .ok()
}

/// Map a worker's kill reason to the run's terminal state + recorded exit code.
/// An explicit `Cancel` lands in `Cancelled`; every other reason is a failure
/// (drain/wall preemption of a resumable run is handled by the requeue paths,
/// not here).
pub(super) fn kill_reason_state(reason: &wire::KillReason) -> (RunState, i64) {
    match reason {
        wire::KillReason::Cancel => (RunState::Cancelled, EXIT_CODE_CANCELLED),
        _ => (RunState::Failed, EXIT_CODE_AGENT_ERROR),
    }
}
