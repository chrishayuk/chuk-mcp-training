//! Live worker connections + the scheduler.
//!
//! Scheduling is still M0-shaped (spec §14): FIFO queue, one job in flight per
//! worker, no packing, no leases in the scheduler. `pump()` runs whenever
//! capacity or work may have appeared: worker attach, run submission, run exit,
//! worker detach.
//!
//! Since chuk-compute M1 the worker speaks the compute-generic protocol
//! (`chuk-compute-wire`): the control plane translates a run's `RunSpec` into a
//! generic [`wire::Job`] (via [`crate::jobspec`]) on assignment, and interprets
//! the worker's generic [`wire::WorkerToCp::Artifact`] events back into
//! checkpoints — the lineage merge (code/seed/arch/parent/slices) that a worker
//! used to do now lives here, its correct home.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{Context, Result};
use chuk_compute_wire as wire;
use chuk_train_proto::{
    keys, CheckpointMeta, EventKind, Hardware, LeaseState, RunId, RunSpec, RunState, TrainSpec,
    WorkerId, WorkerInfo, WorkerState, CHECKPOINT_DIR_PREFIX, CHECKPOINT_META_FILE,
    CHECKPOINT_MODEL_FILE, EXIT_CODE_AGENT_ERROR, EXIT_CODE_CANCELLED, HEARTBEAT_PREEMPT_TIMEOUT,
    HEARTBEAT_TIMEOUT,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::artifacts::ArtifactStore;
use crate::experiments::Experiments;
use crate::grant::GrantTable;
use crate::jobspec::{self, ResumeStaging, TrainStaging};
use crate::store::Store;

/// The artifact class the control plane records checkpoint outputs under (the
/// value it stamps into each train job's checkpoint [`wire::OutputRule`]).
const ARTIFACT_CHECKPOINT: &str = "checkpoint";

/// Sender half of a worker's outbound message queue; the websocket task owns the
/// receiving half and the socket itself.
pub type AgentTx = mpsc::UnboundedSender<wire::CpToWorker>;

/// Per-run slice bookkeeping for checkpoint lineage: where the current worker
/// session resumed from, and the slice history carried by that resume point.
/// Mirrors what a worker session used to track locally (spec §11.2 `slices`).
#[derive(Clone, Default)]
struct ResumeState {
    from_step: u64,
    base_slices: Vec<[u64; 2]>,
}

pub struct Hub {
    pub store: Arc<dyn Store>,
    pub artifacts: Arc<dyn ArtifactStore>,
    /// chuk-experiments-server reporting mirror; `None` when unconfigured.
    experiments: Option<Arc<Experiments>>,
    grants: GrantTable,
    links: Mutex<HashMap<WorkerId, AgentTx>>,
    resume_state: StdMutex<HashMap<RunId, ResumeState>>,
    /// Highest streamed-event `seq` processed per worker (chuk-compute M3.2). A
    /// reconnecting worker replays its outbox; anything at or below its
    /// high-water here has already been applied and is dropped (spec §8).
    high_water: StdMutex<HashMap<WorkerId, u64>>,
}

impl Hub {
    pub fn new(
        store: Arc<dyn Store>,
        artifacts: Arc<dyn ArtifactStore>,
        experiments: Option<Arc<Experiments>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            artifacts,
            experiments,
            grants: GrantTable::default(),
            links: Mutex::new(HashMap::new()),
            resume_state: StdMutex::new(HashMap::new()),
            high_water: StdMutex::new(HashMap::new()),
        })
    }

    /// The highest streamed-event seq processed for a worker — echoed in HelloAck
    /// so a reconnecting worker replays only what follows.
    pub fn resume_high_water(&self, worker_id: &WorkerId) -> u64 {
        self.high_water
            .lock()
            .expect("high_water lock")
            .get(worker_id)
            .copied()
            .unwrap_or(0)
    }

    pub fn grants(&self) -> &GrantTable {
        &self.grants
    }

    // ---- experiments-server mirror (best-effort, fire-and-forget) ----------

    fn mirror_created(&self, run_id: &RunId, spec: &RunSpec, experiment_ref: Option<&str>) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id, spec) = (exp.clone(), run_id.clone(), spec.clone());
            let experiment_ref = experiment_ref.map(str::to_owned);
            tokio::spawn(async move { exp.report_created(run_id, spec, experiment_ref).await });
        }
    }

    fn mirror_state(&self, run_id: &RunId, state: RunState) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id) = (exp.clone(), run_id.clone());
            tokio::spawn(async move { exp.report_state(run_id, state).await });
        }
    }

    fn mirror_checkpoint(&self, run_id: &RunId, step: u64, uri: &str, meta: &CheckpointMeta) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id, uri, meta) =
                (exp.clone(), run_id.clone(), uri.to_owned(), meta.clone());
            tokio::spawn(async move { exp.report_checkpoint(run_id, step, uri, meta).await });
        }
    }

    /// Send a control message to one worker if it is connected; returns false if
    /// there is no live link (e.g. the worker is already gone).
    pub async fn send_to(&self, worker_id: &WorkerId, msg: wire::CpToWorker) -> bool {
        match self.links.lock().await.get(worker_id) {
            Some(tx) => tx.send(msg).is_ok(),
            None => false,
        }
    }

    // ---- connection lifecycle ---------------------------------------------

    pub async fn attach(
        &self,
        worker_id: &WorkerId,
        tx: AgentTx,
        labels: &[String],
        hardware: &Hardware,
    ) -> Result<()> {
        self.store.worker_joined(worker_id, labels, hardware).await?;
        self.links.lock().await.insert(worker_id.clone(), tx);
        info!(worker = %worker_id, gpu = hardware.gpu.as_deref().unwrap_or("cpu"), "worker attached");
        self.pump().await
    }

    /// A job on a vanished worker is requeued. For a train run that means the
    /// next assignment resumes from its latest uploaded checkpoint (spec §14 M1:
    /// close the tab → re-queues and resumes); for a shell run it restarts.
    ///
    /// A **persistent** worker (M3.2) is exempt: its dropped connection is a
    /// blip, not a loss — the job keeps running on the worker and it reconnects
    /// and replays, so the run stays assigned rather than being requeued.
    pub async fn detach(&self, worker_id: &WorkerId, class: wire::WorkerClass) -> Result<()> {
        self.links.lock().await.remove(worker_id);
        self.store.worker_left(worker_id).await?;
        if class == wire::WorkerClass::Persistent {
            info!(worker = %worker_id, "persistent worker disconnected; keeping its run assigned");
            return self.pump().await;
        }
        if let Some(worker) = self.store.worker(worker_id).await? {
            if let Some(run_id) = worker.current_run {
                warn!(worker = %worker_id, run = %run_id, "worker vanished mid-run; requeueing");
                self.grants.revoke_run(&run_id);
                self.store
                    .transition(
                        &run_id,
                        RunState::Queued,
                        None,
                        None,
                        serde_json::json!({ "reason": "worker_disconnected" }),
                    )
                    .await?;
                self.store.set_worker_run(worker_id, None).await?;
            }
        }
        self.pump().await
    }

    /// Sweep the fleet for heartbeat-lost workers (spec §7). A worker still
    /// marked connected but silent past the preempt timeout is presumed gone and
    /// detached — which re-queues a leased worker's resumable run (a persistent
    /// worker keeps its run for when it reconnects, M3.2). This is the backstop
    /// for a half-open link (a frozen Colab tab) that never delivered a socket
    /// close, so `detach` was never called from the websocket task.
    pub async fn reap_stale_workers(&self) -> Result<()> {
        let preempt_after = HEARTBEAT_PREEMPT_TIMEOUT.as_secs_f64();
        for worker in self.store.fleet().await? {
            if !should_reap(&worker, preempt_after) {
                continue;
            }
            let class = if self.store.worker_is_persistent(&worker.id).await? {
                wire::WorkerClass::Persistent
            } else {
                wire::WorkerClass::Leased
            };
            warn!(worker = %worker.id, age_s = worker.heartbeat_age_s, ?class, "heartbeat lost; reaping worker");
            self.detach(&worker.id, class).await?;
        }
        Ok(())
    }

    /// Run [`Self::reap_stale_workers`] forever on a fixed interval. Spawned once
    /// at startup; a failed sweep is logged and retried on the next tick.
    pub async fn run_reaper_loop(self: Arc<Self>, interval: std::time::Duration) {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            if let Err(error) = self.reap_stale_workers().await {
                warn!(%error, "heartbeat reaper sweep failed");
            }
        }
    }

    // ---- scheduling --------------------------------------------------------

    /// Assign queued runs to idle connected workers, FIFO. The links lock is held
    /// throughout, which also serialises concurrent pumps.
    pub async fn pump(&self) -> Result<()> {
        let mut links = self.links.lock().await;
        let mut dead: Vec<WorkerId> = Vec::new();
        for (worker_id, tx) in links.iter() {
            let Some(worker) = self.store.worker(worker_id).await? else {
                continue;
            };
            if !eligible_for_assignment(&worker, HEARTBEAT_TIMEOUT.as_secs_f64()) {
                continue;
            }
            let Some(run) = self.store.next_queued().await? else {
                break;
            };
            let run_id = run.summary.id.clone();
            let job = self.assemble_job(&run_id, run.spec).await?;
            self.store
                .transition(
                    &run_id,
                    RunState::Assigned,
                    Some(worker_id),
                    None,
                    serde_json::Value::Null,
                )
                .await?;
            self.store.set_worker_run(worker_id, Some(&run_id)).await?;
            if tx.send(wire::CpToWorker::AssignJob { job }).is_err() {
                warn!(worker = %worker_id, run = %run_id, "assign failed (link closed); requeueing");
                self.grants.revoke_run(&run_id);
                self.store
                    .transition(
                        &run_id,
                        RunState::Queued,
                        None,
                        None,
                        serde_json::json!({ "reason": "send_failed" }),
                    )
                    .await?;
                self.store.set_worker_run(worker_id, None).await?;
                dead.push(worker_id.clone());
            }
        }
        for worker_id in dead {
            links.remove(&worker_id);
        }
        Ok(())
    }

    /// Translate a run into a compute-generic [`wire::Job`]. For a train run this
    /// resolves the entrypoint from the code-unit manifest, mints a scoped grant,
    /// records the resume slice history, and adds a `Resumed` event.
    async fn assemble_job(&self, run_id: &RunId, spec: RunSpec) -> Result<wire::Job> {
        match spec {
            RunSpec::Shell(shell) => Ok(jobspec::shell_job(run_id, &shell)),
            RunSpec::Train(train) => self.assemble_train_job(run_id, &train).await,
        }
    }

    async fn assemble_train_job(&self, run_id: &RunId, train: &TrainSpec) -> Result<wire::Job> {
        let unit = self
            .store
            .code_unit(&train.code.name, &train.code.sha)
            .await?
            .with_context(|| format!("code unit {} not registered", train.code))?;
        let entrypoint_cmd = unit
            .manifest
            .entrypoint(&train.entrypoint)
            .with_context(|| format!("entrypoint {:?} not in unit manifest", train.entrypoint))?
            .to_owned();
        let grant = self.grants.mint(run_id.clone(), train.code.clone());
        let code_unit_uri = keys::code_unit_tarball(&train.code.name, &train.code.sha);

        // Resume from the latest uploaded checkpoint, if any, and remember the
        // slice history so this session's checkpoints extend it correctly.
        let latest = self.store.latest_checkpoint(run_id).await?;
        let (resume, resume_state) = match &latest {
            Some(ckpt) => (
                Some((
                    keys::checkpoint_file(&run_id.0, ckpt.step, CHECKPOINT_MODEL_FILE),
                    keys::checkpoint_file(&run_id.0, ckpt.step, CHECKPOINT_META_FILE),
                )),
                ResumeState {
                    from_step: ckpt.step,
                    base_slices: ckpt.meta.slices.clone(),
                },
            ),
            None => (None, ResumeState::default()),
        };
        self.resume_state
            .lock()
            .expect("resume_state lock")
            .insert(run_id.clone(), resume_state);
        if let Some(ckpt) = &latest {
            self.store
                .add_event(
                    run_id,
                    EventKind::Resumed,
                    serde_json::json!({ "from_step": ckpt.step }),
                )
                .await?;
        }

        let resume_staging = resume.as_ref().map(|(model, meta)| ResumeStaging {
            model_uri: model,
            meta_uri: meta,
        });
        Ok(jobspec::train_job(
            run_id,
            train,
            &TrainStaging {
                entrypoint_cmd: &entrypoint_cmd,
                code_unit_uri: &code_unit_uri,
                grant: &grant,
                resume: resume_staging,
            },
        ))
    }

    // ---- messages from workers --------------------------------------------

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
                        self.store
                            .append_metrics(&run_id(&job_id), step, &values)
                            .await?;
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
                self.store
                    .transition(&run_id, RunState::Running, Some(worker_id), None, Value::Null)
                    .await?;
                self.mirror_state(&run_id, RunState::Running);
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
    async fn complete_meta(&self, run_id: &RunId, step: u64) -> Result<CheckpointMeta> {
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
        Ok(meta)
    }

    // ---- submissions -------------------------------------------------------

    pub async fn submit(
        &self,
        name: &str,
        spec: &RunSpec,
        experiment_ref: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<RunId> {
        let run_id = self
            .store
            .create_run(name, spec, experiment_ref, created_by)
            .await?;
        self.mirror_created(&run_id, spec, experiment_ref);
        self.pump().await?;
        Ok(run_id)
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

/// Map a worker's kill reason to the run's terminal state + recorded exit code.
/// An explicit `Cancel` lands in `Cancelled`; every other reason is a failure
/// (drain/wall preemption of a resumable run is handled by the requeue paths,
/// not here).
fn kill_reason_state(reason: &wire::KillReason) -> (RunState, i64) {
    match reason {
        wire::KillReason::Cancel => (RunState::Cancelled, EXIT_CODE_CANCELLED),
        _ => (RunState::Failed, EXIT_CODE_AGENT_ERROR),
    }
}

/// Whether an idle connected worker may be handed a queued run. It must be free
/// (no current run), not draining toward its lease wall (spec §4), and not
/// unreachable — a worker silent past the heartbeat timeout gets no new work
/// (spec §7) so we never assign to a frozen tab.
fn eligible_for_assignment(worker: &WorkerInfo, unreachable_after_s: f64) -> bool {
    worker.current_run.is_none()
        && !worker
            .lease
            .as_ref()
            .is_some_and(|l| l.state == LeaseState::Draining)
        && worker.heartbeat_age_s <= unreachable_after_s
}

/// Whether the heartbeat reaper should give up on a worker: still marked
/// connected, but silent past the preempt timeout (spec §7). An already
/// disconnected worker has been handled by `detach`, so it is skipped.
fn should_reap(worker: &WorkerInfo, preempt_after_s: f64) -> bool {
    worker.state == WorkerState::Connected && worker.heartbeat_age_s >= preempt_after_s
}

/// A [`wire::JobId`] is a run's id verbatim on the fabric.
fn run_id(job_id: &wire::JobId) -> RunId {
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
fn step_from_uri(uri: &str) -> Option<u64> {
    uri.rsplit('/')
        .next()?
        .strip_prefix(CHECKPOINT_DIR_PREFIX)?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::store::SqliteStore;
    use chuk_train_proto::ShellSpec;

    #[test]
    fn step_parses_from_the_trailing_segment() {
        assert_eq!(step_from_uri("ckpt-hot/RUN-1/step_500"), Some(500));
        assert_eq!(step_from_uri("step_0"), Some(0));
        assert_eq!(step_from_uri("ckpt-hot/RUN-1/final"), None);
        assert_eq!(step_from_uri("ckpt-hot/RUN-1/step_x"), None);
    }

    #[test]
    fn job_id_maps_to_run_id_verbatim() {
        assert_eq!(run_id(&wire::JobId::from("RUN-9")), RunId::from("RUN-9"));
    }

    #[test]
    fn cancel_lands_cancelled_every_other_reason_fails() {
        assert_eq!(
            kill_reason_state(&wire::KillReason::Cancel),
            (RunState::Cancelled, EXIT_CODE_CANCELLED)
        );
        for reason in [
            wire::KillReason::Wall,
            wire::KillReason::MaxRuntime,
            wire::KillReason::Drain,
            wire::KillReason::OomGuard,
        ] {
            assert_eq!(
                kill_reason_state(&reason),
                (RunState::Failed, EXIT_CODE_AGENT_ERROR)
            );
        }
    }

    async fn test_hub() -> Arc<Hub> {
        let store = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        // Shell runs never touch the artifact store; temp_dir is never written.
        let artifacts = Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        Hub::new(store, artifacts, None)
    }

    fn shell_run() -> RunSpec {
        RunSpec::Shell(ShellSpec { command: "sleep 1".into(), timeout_s: 60 })
    }

    #[tokio::test]
    async fn stop_queued_run_cancels_immediately() {
        let hub = test_hub().await;
        let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
        // No worker connected — the run is queued; stop finalises it here.
        hub.stop_run(&run).await.unwrap();
        let rec = hub.store.run(&run).await.unwrap().unwrap();
        assert_eq!(rec.summary.state, RunState::Cancelled);
        assert_eq!(rec.summary.exit_code, Some(EXIT_CODE_CANCELLED));
    }

    #[tokio::test]
    async fn stop_rejects_an_already_terminal_run() {
        let hub = test_hub().await;
        let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
        hub.stop_run(&run).await.unwrap();
        assert!(hub.stop_run(&run).await.is_err());
    }

    #[tokio::test]
    async fn resume_requeues_a_terminal_run_but_not_a_live_one() {
        let hub = test_hub().await;
        let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
        hub.stop_run(&run).await.unwrap(); // → Cancelled
        hub.resume_run(&run).await.unwrap();
        let rec = hub.store.run(&run).await.unwrap().unwrap();
        assert_eq!(rec.summary.state, RunState::Queued);
        // A queued (non-terminal) run cannot be resumed.
        assert!(hub.resume_run(&run).await.is_err());
    }

    #[tokio::test]
    async fn stop_running_run_signals_worker_then_jobkilled_finalises() {
        let hub = test_hub().await;
        let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
        // Attach a worker; the pump assigns the queued run to it.
        let (tx, mut rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-test".into());
        hub.attach(&worker, tx, &[], &Hardware::default())
            .await
            .unwrap();
        assert!(matches!(
            rx.recv().await.expect("assign"),
            wire::CpToWorker::AssignJob { .. }
        ));
        // Stop signals the worker (run assigned, link live) — not finalised yet.
        hub.stop_run(&run).await.unwrap();
        match rx.recv().await.expect("cancel") {
            wire::CpToWorker::Cancel { job_id } => assert_eq!(job_id.0, run.0),
            other => panic!("expected Cancel, got {other:?}"),
        }
        let mid = hub.store.run(&run).await.unwrap().unwrap();
        assert!(!mid.summary.state.is_terminal(), "state={:?}", mid.summary.state);
        // The worker confirms the kill; the run lands in Cancelled.
        hub.on_message(
            &worker,
            wire::WorkerToCp::JobKilled {
                seq: 1,
                job_id: wire::JobId::from(run.0.clone()),
                reason: wire::KillReason::Cancel,
            },
        )
        .await
        .unwrap();
        let done = hub.store.run(&run).await.unwrap().unwrap();
        assert_eq!(done.summary.state, RunState::Cancelled);
    }

    // ---- heartbeat reaper (spec §7) ---------------------------------------

    fn worker_info(state: WorkerState, current_run: Option<&str>, age_s: f64) -> WorkerInfo {
        WorkerInfo {
            id: WorkerId("w".into()),
            labels: vec![],
            hardware: Hardware::default(),
            state,
            current_run: current_run.map(|r| RunId(r.into())),
            joined_at: 0.0,
            last_seen: 0.0,
            heartbeat_age_s: age_s,
            lease: None,
        }
    }

    #[test]
    fn assignment_eligibility_honours_run_drain_and_staleness() {
        let fresh = |age| worker_info(WorkerState::Connected, None, age);
        // Free, fresh, unleased → eligible.
        assert!(eligible_for_assignment(&fresh(1.0), 90.0));
        // Already running a job → not eligible.
        assert!(!eligible_for_assignment(
            &worker_info(WorkerState::Connected, Some("EXEC-1"), 1.0),
            90.0
        ));
        // Silent past the unreachable timeout → no new work (spec §7).
        assert!(!eligible_for_assignment(&fresh(120.0), 90.0));
        // Exactly at the boundary is still eligible (strictly greater excludes).
        assert!(eligible_for_assignment(&fresh(90.0), 90.0));
    }

    #[test]
    fn reap_targets_only_connected_workers_past_the_preempt_wall() {
        // Connected + silent past preempt → reap.
        assert!(should_reap(&worker_info(WorkerState::Connected, Some("EXEC-1"), 601.0), 600.0));
        // Connected but recently seen → keep.
        assert!(!should_reap(&worker_info(WorkerState::Connected, Some("EXEC-1"), 120.0), 600.0));
        // Already disconnected (detach handled it) → skip.
        assert!(!should_reap(&worker_info(WorkerState::Disconnected, None, 9999.0), 600.0));
    }

    #[tokio::test]
    async fn reaper_leaves_a_freshly_seen_worker_alone() {
        let hub = test_hub().await;
        let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-fresh".into());
        hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
        assert_eq!(
            hub.store.run(&run).await.unwrap().unwrap().summary.state,
            RunState::Assigned
        );
        // The worker was just seen (age ~0), so the sweep must not touch it.
        hub.reap_stale_workers().await.unwrap();
        assert_eq!(
            hub.store.run(&run).await.unwrap().unwrap().summary.state,
            RunState::Assigned
        );
        assert!(matches!(
            hub.store.worker(&worker).await.unwrap().unwrap().state,
            WorkerState::Connected
        ));
    }

    #[tokio::test]
    async fn detach_requeues_leased_but_keeps_persistent() {
        // Leased: a lost worker's run is requeued (bounded loss).
        let hub = test_hub().await;
        let run = hub.submit("leased", &shell_run(), None, None).await.unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-leased".into());
        hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
        hub.detach(&worker, wire::WorkerClass::Leased).await.unwrap();
        assert_eq!(
            hub.store.run(&run).await.unwrap().unwrap().summary.state,
            RunState::Queued
        );

        // Persistent: the run stays assigned for the worker to reconnect (M3.2).
        let hub = test_hub().await;
        let run = hub.submit("persist", &shell_run(), None, None).await.unwrap();
        let (tx, _rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-persist".into());
        hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
        hub.detach(&worker, wire::WorkerClass::Persistent).await.unwrap();
        assert_eq!(
            hub.store.run(&run).await.unwrap().unwrap().summary.state,
            RunState::Assigned
        );
    }
}
