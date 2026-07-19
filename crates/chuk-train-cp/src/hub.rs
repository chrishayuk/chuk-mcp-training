//! Live worker connections + the scheduler.
//!
//! Scheduling is still M0-shaped (spec §14): FIFO queue, one run in flight per
//! worker, no packing, no leases. `pump()` runs whenever capacity or work may
//! have appeared: worker attach, run submission, run exit, worker detach.
//!
//! M1 adds train mechanics: on assigning a train run the hub mints a
//! run-scoped upload grant and resolves the resume point from the run's latest
//! uploaded checkpoint, and it ingests streamed metrics and checkpoint reports.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chuk_train_proto::{
    AgentToCp, CheckpointMeta, CpToAgent, EventKind, Hardware, JobAssignment, LeaseState,
    ResumeInfo, RunId, RunSpec, RunState, UploadGrant, WorkerId,
};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::artifacts::{keys, ArtifactStore};
use crate::experiments::Experiments;
use crate::grant::GrantTable;
use crate::store::Store;

/// Sender half of a worker's outbound message queue; the websocket task owns
/// the receiving half and the socket itself.
pub type AgentTx = mpsc::UnboundedSender<CpToAgent>;

pub struct Hub {
    pub store: Arc<dyn Store>,
    pub artifacts: Arc<dyn ArtifactStore>,
    /// chuk-experiments-server reporting mirror; `None` when unconfigured
    /// (spec §11.6). Every use is best-effort and off the run's critical path.
    experiments: Option<Arc<Experiments>>,
    grants: GrantTable,
    links: Mutex<HashMap<WorkerId, AgentTx>>,
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
        })
    }

    // ---- experiments-server mirror (best-effort, fire-and-forget) ----------
    // Each spawns a detached task so a slow/down experiments-server can never
    // block or fail a run; a no-op when the mirror is unconfigured.

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

    pub fn grants(&self) -> &GrantTable {
        &self.grants
    }

    /// Send a control message to one worker if it is connected; returns false
    /// if there is no live link (e.g. the agent is already gone).
    pub async fn send_to(&self, worker_id: &WorkerId, msg: CpToAgent) -> bool {
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
        self.store
            .worker_joined(worker_id, labels, hardware)
            .await?;
        self.links.lock().await.insert(worker_id.clone(), tx);
        info!(worker = %worker_id, gpu = hardware.gpu.as_deref().unwrap_or("cpu"), "worker attached");
        self.pump().await
    }

    /// A run on a vanished worker is requeued. For a train run that means the
    /// next assignment resumes from its latest uploaded checkpoint (spec §14
    /// M1: close the tab → re-queues and resumes); for a shell run it restarts.
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

    /// Assign queued runs to idle connected workers, FIFO. The links lock is
    /// held throughout, which also serialises concurrent pumps.
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
            let assignment = self.assemble_assignment(&run_id, run.spec).await?;
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
            if let Some(resume) = &assignment.resume {
                self.store
                    .add_event(
                        &run_id,
                        EventKind::Resumed,
                        serde_json::json!({ "from_step": resume.from_step }),
                    )
                    .await?;
            }
            if tx.send(CpToAgent::Assign { job: assignment }).is_err() {
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

    /// Build a `JobAssignment`, adding a scoped grant and resume point for
    /// train runs.
    async fn assemble_assignment(&self, run_id: &RunId, spec: RunSpec) -> Result<JobAssignment> {
        let (grant, resume) = match &spec {
            RunSpec::Train(train) => {
                let token = self.grants.mint(run_id.clone(), train.code.clone());
                let resume = self
                    .store
                    .latest_checkpoint(run_id)
                    .await?
                    .map(|ckpt| ResumeInfo {
                        from_step: ckpt.step,
                        checkpoint_path: keys::checkpoint_dir(&run_id.0, ckpt.step),
                    });
                (Some(UploadGrant { token }), resume)
            }
            RunSpec::Shell(_) => (None, None),
        };
        Ok(JobAssignment {
            run_id: run_id.clone(),
            spec,
            resume,
            grant,
        })
    }

    // ---- messages from agents ---------------------------------------------

    pub async fn on_message(&self, worker_id: &WorkerId, msg: AgentToCp) -> Result<()> {
        self.store.worker_seen(worker_id).await?;
        match msg {
            AgentToCp::Register { .. } => {
                debug!(worker = %worker_id, "duplicate register ignored");
            }
            AgentToCp::Heartbeat => {}
            AgentToCp::Log { run_id, line } => self.store.append_log(&run_id, &line).await?,
            AgentToCp::Metric {
                run_id,
                step,
                values,
            } => {
                self.store.append_metrics(&run_id, step, &values).await?;
            }
            AgentToCp::Checkpoint {
                run_id,
                step,
                model_hash,
                meta,
            } => {
                self.ingest_checkpoint(&run_id, step, &model_hash, &meta)
                    .await?;
            }
            AgentToCp::JobStarted { run_id } => {
                self.store
                    .transition(
                        &run_id,
                        RunState::Running,
                        Some(worker_id),
                        None,
                        serde_json::Value::Null,
                    )
                    .await?;
                self.mirror_state(&run_id, RunState::Running);
            }
            AgentToCp::JobExited { run_id, code } => {
                let state = if code == 0 {
                    RunState::Completed
                } else {
                    RunState::Failed
                };
                self.grants.revoke_run(&run_id);
                self.store
                    .transition(
                        &run_id,
                        state,
                        Some(worker_id),
                        Some(code),
                        serde_json::Value::Null,
                    )
                    .await?;
                self.mirror_state(&run_id, state);
                self.store.set_worker_run(worker_id, None).await?;
                self.pump().await?;
            }
            AgentToCp::Drained => {
                // The worker has flushed + uploaded and stopped work. The lease
                // manager still owns the provider-verified destroy at T-0 — the
                // wall never depends on this message arriving.
                info!(worker = %worker_id, "worker drained");
            }
        }
        Ok(())
    }

    async fn ingest_checkpoint(
        &self,
        run_id: &RunId,
        step: u64,
        model_hash: &str,
        meta: &CheckpointMeta,
    ) -> Result<()> {
        let uri = self.artifacts.uri(&keys::checkpoint_dir(&run_id.0, step));
        self.store
            .record_checkpoint(run_id, step, &uri, model_hash, meta)
            .await?;
        self.store
            .add_event(
                run_id,
                EventKind::Checkpoint,
                serde_json::json!({ "step": step, "uri": uri, "model_hash": model_hash }),
            )
            .await?;
        self.mirror_checkpoint(run_id, step, &uri, meta);
        info!(run = %run_id, step, "checkpoint recorded");
        Ok(())
    }

    // ---- submissions -------------------------------------------------------

    pub async fn submit(&self, name: &str, spec: &RunSpec) -> Result<RunId> {
        let run_id = self.store.create_run(name, spec).await?;
        self.mirror_created(&run_id, spec);
        self.pump().await?;
        Ok(run_id)
    }
}
