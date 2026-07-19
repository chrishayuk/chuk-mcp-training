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
    WorkerId, CHECKPOINT_DIR_PREFIX, CHECKPOINT_META_FILE, CHECKPOINT_MODEL_FILE,
    EXIT_CODE_AGENT_ERROR,
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
        })
    }

    pub fn grants(&self) -> &GrantTable {
        &self.grants
    }

    // ---- experiments-server mirror (best-effort, fire-and-forget) ----------

    fn mirror_created(&self, run_id: &RunId, spec: &RunSpec) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id, spec) = (exp.clone(), run_id.clone(), spec.clone());
            tokio::spawn(async move { exp.report_created(run_id, spec).await });
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
    pub async fn detach(&self, worker_id: &WorkerId) -> Result<()> {
        self.links.lock().await.remove(worker_id);
        self.store.worker_left(worker_id).await?;
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
            if worker.current_run.is_some() {
                continue;
            }
            // A draining worker takes no new work (spec §4: past T-drain).
            if worker
                .lease
                .as_ref()
                .is_some_and(|l| l.state == LeaseState::Draining)
            {
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
            } => {
                // Job metrics are (job, step)-indexed; host `sys/*` samples carry
                // neither and are not yet ingested (M4).
                if let (Some(job_id), Some(step)) = (job_id, step) {
                    self.store
                        .append_metrics(&run_id(&job_id), step, &values)
                        .await?;
                }
            }
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
                self.finish_run(
                    worker_id,
                    &run_id(&job_id),
                    RunState::Failed,
                    EXIT_CODE_AGENT_ERROR,
                )
                .await?;
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

    pub async fn submit(&self, name: &str, spec: &RunSpec) -> Result<RunId> {
        let run_id = self.store.create_run(name, spec).await?;
        self.mirror_created(&run_id, spec);
        self.pump().await?;
        Ok(run_id)
    }
}

/// A [`wire::JobId`] is a run's id verbatim on the fabric.
fn run_id(job_id: &wire::JobId) -> RunId {
    RunId(job_id.0.clone())
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
}
