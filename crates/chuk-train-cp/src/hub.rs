//! Live worker connections + the M0 scheduler.
//!
//! M0 scheduling (spec §14): FIFO queue, one run in flight per worker, no
//! packing, no leases. `pump()` runs whenever capacity or work may have
//! appeared: worker attach, run submission, run exit, worker detach.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use chuk_train_proto::{AgentToCp, CpToAgent, Hardware, JobAssignment, RunId, RunState, WorkerId};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::store::Store;

/// Sender half of a worker's outbound message queue; the websocket task owns
/// the receiving half and the socket itself.
pub type AgentTx = mpsc::UnboundedSender<CpToAgent>;

pub struct Hub {
    pub store: Arc<dyn Store>,
    links: Mutex<HashMap<WorkerId, AgentTx>>,
}

impl Hub {
    pub fn new(store: Arc<dyn Store>) -> Arc<Self> {
        Arc::new(Self {
            store,
            links: Mutex::new(HashMap::new()),
        })
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

    /// M0: there is no resume machinery yet, and the agent kills its child
    /// process when the session drops — so a run on a vanished worker is
    /// requeued and will restart from scratch.
    pub async fn detach(&self, worker_id: &WorkerId) -> Result<()> {
        self.links.lock().await.remove(worker_id);
        self.store.worker_left(worker_id).await?;
        if let Some(worker) = self.store.worker(worker_id).await? {
            if let Some(run_id) = worker.current_run {
                warn!(worker = %worker_id, run = %run_id, "worker vanished mid-run; requeueing");
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
            let Some(run) = self.store.next_queued().await? else {
                break;
            };
            let run_id = run.summary.id.clone();
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
            let assign = CpToAgent::Assign {
                job: JobAssignment {
                    run_id: run_id.clone(),
                    spec: run.spec,
                },
            };
            if tx.send(assign).is_err() {
                warn!(worker = %worker_id, run = %run_id, "assign failed (link closed); requeueing");
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

    // ---- messages from agents ---------------------------------------------

    pub async fn on_message(&self, worker_id: &WorkerId, msg: AgentToCp) -> Result<()> {
        self.store.worker_seen(worker_id).await?;
        match msg {
            AgentToCp::Register { .. } => {
                debug!(worker = %worker_id, "duplicate register ignored");
            }
            AgentToCp::Heartbeat => {}
            AgentToCp::Log { run_id, line } => self.store.append_log(&run_id, &line).await?,
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
            }
            AgentToCp::JobExited { run_id, code } => {
                let state = if code == 0 {
                    RunState::Completed
                } else {
                    RunState::Failed
                };
                self.store
                    .transition(
                        &run_id,
                        state,
                        Some(worker_id),
                        Some(code),
                        serde_json::Value::Null,
                    )
                    .await?;
                self.store.set_worker_run(worker_id, None).await?;
                self.pump().await?;
            }
        }
        Ok(())
    }

    // ---- submissions -------------------------------------------------------

    pub async fn submit(&self, name: &str, spec: &chuk_train_proto::RunSpec) -> Result<RunId> {
        let run_id = self.store.create_run(name, spec).await?;
        self.pump().await?;
        Ok(run_id)
    }
}
