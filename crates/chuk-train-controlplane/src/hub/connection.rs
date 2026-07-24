//! Hub — worker connection lifecycle: attach on websocket handshake, detach
//! on disconnect (requeueing a leased worker's live run), and the heartbeat
//! reaper that backstops a half-open link (spec §7).

use super::*;

impl Hub {
    /// Send a control message to one worker if it is connected; returns false if
    /// there is no live link (e.g. the worker is already gone).
    pub async fn send_to(&self, worker_id: &WorkerId, msg: wire::CpToWorker) -> bool {
        match self.links.lock().await.get(worker_id) {
            Some(tx) => tx.send(msg).is_ok(),
            None => false,
        }
    }

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
}

/// Whether the heartbeat reaper should give up on a worker: still marked
/// connected, but silent past the preempt timeout (spec §7). An already
/// disconnected worker has been handled by `detach`, so it is skipped.
pub(super) fn should_reap(worker: &WorkerInfo, preempt_after_s: f64) -> bool {
    worker.state == WorkerState::Connected && worker.heartbeat_age_s >= preempt_after_s
}
