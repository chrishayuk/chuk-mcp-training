//! The lease manager (spec §3): enforces the wall, drains, and — the property
//! E2 exists to prove — destroys the instance at T-0 whether or not the agent
//! responds, verified against the provider API. Plus an idle reaper and a
//! reconcile loop that kills orphaned instances the registry lost track of.
//!
//! Teardown never depends on the agent (design principle 5): every destroy
//! goes to the provider directly and is verified by polling `status`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use chuk_train_proto::{
    CpToAgent, Instance, InstanceStatus, Lease, LeaseExtension, LeaseState, LedgerEntry,
    ProvisionRequest, ProvisionResult, TeardownResult, WorkerId, DESTROY_VERIFY_POLL,
    DESTROY_VERIFY_TIMEOUT, LEASE_TICK_INTERVAL,
};
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::Config;
use crate::hub::Hub;
use crate::provider::{Provider, Providers, ProvisionContext};

/// Ledger event names (spec §8).
const EV_LEASE_START: &str = "lease_start";
const EV_EXTEND: &str = "extend";
const EV_LEASE_END: &str = "lease_end";

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub struct LeaseManager {
    hub: Arc<Hub>,
    providers: Arc<Providers>,
    config: Config,
    /// When each worker became idle, for the idle reaper.
    idle_since: Mutex<HashMap<WorkerId, Instant>>,
}

impl LeaseManager {
    pub fn new(hub: Arc<Hub>, providers: Arc<Providers>, config: Config) -> Arc<Self> {
        Arc::new(Self {
            hub,
            providers,
            config,
            idle_since: Mutex::new(HashMap::new()),
        })
    }

    // ---- provisioning ------------------------------------------------------

    pub async fn offers(
        &self,
        provider: &str,
        gpu: Option<&str>,
        max_price_hr: Option<f64>,
    ) -> Result<Vec<chuk_train_proto::Offer>> {
        self.provider(provider)?.offers(gpu, max_price_hr).await
    }

    pub async fn provision(&self, req: &ProvisionRequest) -> Result<ProvisionResult> {
        let provider = self.provider(&req.provider)?;
        let worker_id = WorkerId(format!(
            "{}-{}",
            req.provider,
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        let ctx = ProvisionContext {
            ws_url: self.config.agent_ws_url.clone(),
            join_token: self.config.join_token.clone(),
            worker_id: worker_id.0.clone(),
            drain_window_min: self.config.drain_window_min,
        };
        let instance = provider
            .provision(req, &ctx)
            .await
            .context("provider provision")?;

        let lease = Lease {
            worker_id: worker_id.clone(),
            provider: req.provider.clone(),
            instance_id: instance.id.clone(),
            price_hr: instance.price_hr,
            granted_min: req.lease_min,
            drain_window_min: self.config.drain_window_min,
            started_at: now(),
            state: LeaseState::Active,
            extensions: Vec::new(),
        };
        self.hub.store.create_lease(&lease).await?;
        self.ledger(
            &lease,
            EV_LEASE_START,
            lease.total_granted_min(),
            lease.projected_cost(),
        )
        .await?;
        info!(worker = %worker_id, provider = %req.provider, lease_min = req.lease_min, "provisioned");
        Ok(ProvisionResult {
            worker_id,
            lease,
            bootstrap: String::new(),
        })
    }

    pub async fn extend(
        &self,
        worker_id: &WorkerId,
        minutes: f64,
        reason: &str,
    ) -> Result<Option<Lease>> {
        let ext = LeaseExtension {
            minutes,
            at: now(),
            reason: reason.to_owned(),
        };
        let updated = self.hub.store.extend_lease(worker_id, ext).await?;
        if let Some(lease) = &updated {
            // Re-activate a draining lease whose wall has moved out.
            if lease.state == LeaseState::Draining {
                self.hub
                    .store
                    .set_lease_state(worker_id, LeaseState::Active)
                    .await?;
            }
            self.ledger(lease, EV_EXTEND, minutes, lease.price_hr * minutes / 60.0)
                .await?;
            info!(worker = %worker_id, minutes, "lease extended");
        }
        Ok(updated)
    }

    /// Tear a lease down now: destroy the instance, verify it is gone, record
    /// the spend. `drain_first` sends a best-effort drain and short grace; the
    /// destroy happens regardless.
    pub async fn teardown(
        &self,
        worker_id: &WorkerId,
        drain_first: bool,
    ) -> Result<TeardownResult> {
        let Some(lease) = self.hub.store.lease(worker_id).await? else {
            anyhow::bail!("no lease for worker {worker_id}");
        };
        if drain_first {
            self.drain(&lease).await?;
        }
        self.destroy_and_verify(&lease).await
    }

    // ---- background loops --------------------------------------------------

    /// The lease clock: drives every live lease through T-drain and T-0, and
    /// runs the idle reaper. Never returns.
    pub async fn run_clock(self: Arc<Self>) {
        let mut tick = tokio::time::interval(LEASE_TICK_INTERVAL);
        loop {
            tick.tick().await;
            if let Err(error) = self.clock_pass().await {
                warn!(%error, "lease clock pass failed");
            }
        }
    }

    async fn clock_pass(&self) -> Result<()> {
        let leases = self.hub.store.live_leases().await?;
        let now = now();
        for lease in leases {
            let elapsed = now - lease.started_at;
            if elapsed >= lease.wall_secs() {
                // T-0: the wall. Destroy regardless of agent state.
                info!(worker = %lease.worker_id, "lease reached T-0; destroying");
                self.destroy_and_verify(&lease).await?;
                continue;
            }
            if lease.state == LeaseState::Active && elapsed >= lease.drain_secs() {
                // T-drain: ask the agent to wind down; the destroy still lands
                // at T-0 whether or not it complies.
                self.drain(&lease).await?;
            }
        }
        self.reap_idle().await
    }

    /// The reconcile loop (spec §3): list real provider instances and kill any
    /// the registry does not own — the backstop for a hung agent, a dead
    /// tunnel, or a wedged instance that still bills. Never returns.
    pub async fn run_reconcile(self: Arc<Self>) {
        let mut tick = tokio::time::interval(self.config.reconcile_interval);
        loop {
            tick.tick().await;
            if let Err(error) = self.reconcile_pass().await {
                warn!(%error, "reconcile pass failed");
            }
        }
    }

    async fn reconcile_pass(&self) -> Result<()> {
        let live = self.hub.store.live_leases().await?;
        for provider in self.providers.all() {
            let owned: std::collections::HashSet<&str> = live
                .iter()
                .filter(|l| l.provider == provider.name())
                .map(|l| l.instance_id.as_str())
                .collect();
            let instances = match provider.list_instances().await {
                Ok(instances) => instances,
                Err(error) => {
                    warn!(provider = provider.name(), %error, "list_instances failed");
                    continue;
                }
            };
            for instance in instances {
                if !owned.contains(instance.id.as_str()) {
                    self.kill_orphan(provider.as_ref(), &instance).await;
                }
            }
        }
        Ok(())
    }

    // ---- helpers -----------------------------------------------------------

    async fn drain(&self, lease: &Lease) -> Result<()> {
        if lease.state != LeaseState::Draining {
            self.hub
                .store
                .set_lease_state(&lease.worker_id, LeaseState::Draining)
                .await?;
        }
        let deadline = lease.started_at + lease.wall_secs();
        let sent = self
            .hub
            .send_to(&lease.worker_id, CpToAgent::Drain { deadline })
            .await;
        info!(worker = %lease.worker_id, delivered = sent, "drain sent");
        Ok(())
    }

    async fn destroy_and_verify(&self, lease: &Lease) -> Result<TeardownResult> {
        let provider = self.provider(&lease.provider)?;
        provider
            .destroy(&lease.instance_id)
            .await
            .context("provider destroy")?;

        // Verify: poll until the provider says Gone, or alert on timeout.
        let status = self
            .verify_gone(provider.as_ref(), &lease.instance_id)
            .await;
        if status != InstanceStatus::Gone {
            warn!(
                worker = %lease.worker_id, instance = %lease.instance_id, ?status,
                "ORPHAN ALERT: instance not verified gone after destroy; reconcile will retry"
            );
        }

        self.hub
            .store
            .set_lease_state(&lease.worker_id, LeaseState::Destroyed)
            .await?;
        self.hub.store.worker_left(&lease.worker_id).await?;
        self.idle_since.lock().await.remove(&lease.worker_id);

        let elapsed_min = ((now() - lease.started_at) / 60.0).max(0.0);
        let cost = lease.price_hr * elapsed_min / 60.0;
        self.ledger(lease, EV_LEASE_END, elapsed_min, cost).await?;
        info!(worker = %lease.worker_id, elapsed_min, cost, "lease torn down");
        Ok(TeardownResult {
            worker_id: lease.worker_id.clone(),
            destroyed: status == InstanceStatus::Gone,
            status,
        })
    }

    async fn verify_gone(&self, provider: &dyn Provider, instance_id: &str) -> InstanceStatus {
        let deadline = Instant::now() + DESTROY_VERIFY_TIMEOUT;
        loop {
            match provider.status(instance_id).await {
                Ok(InstanceStatus::Gone) => return InstanceStatus::Gone,
                Ok(other) => {
                    if Instant::now() >= deadline {
                        return other;
                    }
                }
                Err(error) => {
                    warn!(%error, "status poll failed");
                    if Instant::now() >= deadline {
                        return InstanceStatus::Unknown;
                    }
                }
            }
            tokio::time::sleep(DESTROY_VERIFY_POLL).await;
        }
    }

    async fn kill_orphan(&self, provider: &dyn Provider, instance: &Instance) {
        warn!(
            provider = provider.name(), instance = %instance.id,
            "ORPHAN: billed instance not owned by any live lease; auto-killing"
        );
        if let Err(error) = provider.destroy(&instance.id).await {
            warn!(instance = %instance.id, %error, "orphan destroy failed");
        }
    }

    /// Drain + destroy any worker idle past the reaper threshold (spec §3): a
    /// lease is a ceiling, not a commitment to burn.
    async fn reap_idle(&self) -> Result<()> {
        let leases = self.hub.store.live_leases().await?;
        let queue_empty = self.hub.store.next_queued().await?.is_none();
        let mut idle_since = self.idle_since.lock().await;
        let mut reap: Vec<Lease> = Vec::new();
        for lease in leases {
            let worker = self.hub.store.worker(&lease.worker_id).await?;
            let idle = queue_empty
                && lease.state == LeaseState::Active
                && worker.as_ref().is_some_and(|w| w.current_run.is_none());
            if !idle {
                idle_since.remove(&lease.worker_id);
                continue;
            }
            let since = *idle_since
                .entry(lease.worker_id.clone())
                .or_insert_with(Instant::now);
            if since.elapsed() >= self.config.idle_reap {
                reap.push(lease);
            }
        }
        drop(idle_since);
        for lease in reap {
            info!(worker = %lease.worker_id, "idle reaper: draining + destroying early");
            self.drain(&lease).await?;
            self.destroy_and_verify(&lease).await?;
        }
        Ok(())
    }

    fn provider(&self, name: &str) -> Result<Arc<dyn Provider>> {
        self.providers.get(name).with_context(|| {
            format!(
                "unknown provider {name:?}; configured: {:?}",
                self.providers.names()
            )
        })
    }

    async fn ledger(&self, lease: &Lease, event: &str, minutes: f64, cost: f64) -> Result<()> {
        self.hub
            .store
            .ledger_append(&LedgerEntry {
                ts: now(),
                worker_id: lease.worker_id.clone(),
                provider: lease.provider.clone(),
                event: event.to_owned(),
                minutes,
                cost,
            })
            .await
    }
}
