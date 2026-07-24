//! The lease manager (spec §3): enforces the wall, drains, and — the property
//! E2 exists to prove — destroys the instance at T-0 whether or not the agent
//! responds, verified against the provider API. Plus an idle reaper and a
//! reconcile loop that kills orphaned instances the registry lost track of.
//!
//! Teardown never depends on the agent (design principle 5): every destroy
//! goes to the provider directly and is verified by polling `status`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chuk_compute_wire as wire;
use chuk_train_proto::{
    Instance, InstanceStatus, Lease, LeaseExtension, LeaseState, LedgerEntry,
    ProvisionRequest, ProvisionResult, TeardownResult, WorkerId, DESTROY_VERIFY_POLL,
    DESTROY_VERIFY_TIMEOUT, LEASE_TICK_INTERVAL,
};
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

use crate::budget;
use crate::config::Config;
use crate::hub::Hub;
use crate::provider::{Provider, Providers, ProvisionContext};

/// Ledger event names (spec §8).
const EV_LEASE_START: &str = chuk_train_proto::LEDGER_EVENT_LEASE_START;
const EV_EXTEND: &str = chuk_train_proto::LEDGER_EVENT_EXTEND;
const EV_LEASE_END: &str = chuk_train_proto::LEDGER_EVENT_LEASE_END;

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// How hard `destroy_and_verify` presses the provider for confirmation that an
/// instance is really gone: the spec's poll interval and timeout (§3) in
/// production, shrunk in tests so the not-verified path is exercised without
/// waiting out two real minutes.
#[derive(Debug, Clone, Copy)]
struct VerifyPolicy {
    timeout: Duration,
    poll: Duration,
}

impl VerifyPolicy {
    fn spec() -> Self {
        Self {
            timeout: DESTROY_VERIFY_TIMEOUT,
            poll: DESTROY_VERIFY_POLL,
        }
    }
}

pub struct LeaseManager {
    hub: Arc<Hub>,
    providers: Arc<Providers>,
    config: Config,
    verify: VerifyPolicy,
    /// When each worker became idle, for the idle reaper.
    idle_since: Mutex<HashMap<WorkerId, Instant>>,
}

impl LeaseManager {
    pub fn new(hub: Arc<Hub>, providers: Arc<Providers>, config: Config) -> Arc<Self> {
        Self::verifying(hub, providers, config, VerifyPolicy::spec())
    }

    fn verifying(
        hub: Arc<Hub>,
        providers: Arc<Providers>,
        config: Config,
        verify: VerifyPolicy,
    ) -> Arc<Self> {
        Arc::new(Self {
            hub,
            providers,
            config,
            verify,
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

    /// Refuse if spending `candidate_cost` more on `provider` would breach a
    /// budget (spec §8: live leases count as committed, ledger is spent).
    async fn budget_check(&self, provider: &str, candidate_cost: f64) -> Result<()> {
        let (budgets, ledger, live) = tokio::try_join!(
            self.hub.store.budgets(),
            self.hub.store.ledger_entries(),
            self.hub.store.live_leases(),
        )?;
        if let Some(breach) =
            budget::evaluate(&budgets, &ledger, &live, provider, candidate_cost, now())
        {
            anyhow::bail!("{breach}");
        }
        Ok(())
    }

    pub async fn provision(&self, req: &ProvisionRequest) -> Result<ProvisionResult> {
        let provider = self.provider(&req.provider)?;
        // Cheap pre-flight before renting anything: refuse when headroom is
        // already gone even with a zero-cost candidate.
        self.budget_check(&req.provider, 0.0).await?;
        let worker_id = WorkerId(format!(
            "{}-{}",
            req.provider,
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        // Single-use enrolment credential (spec §12): bound to this worker id,
        // consumed on first join — never the shared config token.
        let join_token = crate::apikey::mint_join_token(self.hub.store.as_ref(), &worker_id).await?;
        let ctx = ProvisionContext {
            ws_url: self.config.agent_ws_url.clone(),
            join_token,
            worker_id: worker_id.0.clone(),
            drain_window_min: self.config.drain_window_min,
        };
        let instance = provider
            .provision(req, &ctx)
            .await
            .context("provider provision")?;

        // Exact check now the real price is known. On breach (or a store
        // failure that leaves the budget unverifiable) destroy immediately so
        // a refused provision never leaves a billing instance behind.
        let candidate_cost = instance.price_hr * req.lease_min / 60.0;
        if let Err(refused) = self.budget_check(&req.provider, candidate_cost).await {
            warn!(
                provider = %req.provider, instance = %instance.id, %refused,
                "provision refused post-price; destroying instance"
            );
            if let Err(error) = provider.destroy(&instance.id).await {
                warn!(instance = %instance.id, %error, "destroy of refused instance failed; reconcile will retry");
            }
            return Err(refused);
        }

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
        // The extension's cost is known up front — check before moving the wall.
        if let Some(lease) = self.hub.store.lease(worker_id).await? {
            self.budget_check(&lease.provider, lease.price_hr * minutes / 60.0)
                .await?;
        }
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
            .send_to(&lease.worker_id, wire::CpToWorker::Drain { deadline })
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
        let deadline = Instant::now() + self.verify.timeout;
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
            tokio::time::sleep(self.verify.poll).await;
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

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use chuk_train_proto::{
        Budget, Offer, ProvisionRequest, BUDGET_PERIOD_ALL, BUDGET_SCOPE_GLOBAL,
    };

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::store::{SqliteStore, Store};

    const PROVIDER: &str = "fake";
    const DRAIN_WINDOW_MIN: f64 = 1.0;
    /// Short enough that a test can wait one out, long enough that a pass which
    /// runs immediately after another still sees the worker as freshly idle.
    const IDLE_REAP: Duration = Duration::from_millis(80);

    /// A provider whose every answer is scriptable: what it hands back from
    /// `provision`, whether `destroy`/`list_instances` fail, and what `status`
    /// reports (so destroy-verification can be made to succeed, or never to).
    /// The mock provider can't do any of that — it launches real processes.
    struct FakeProvider {
        price_hr: f64,
        status: InstanceStatus,
        destroy_fails: bool,
        list_fails: bool,
        /// Instances `list_instances` reports as still billing.
        listed: StdMutex<Vec<Instance>>,
        destroyed: StdMutex<Vec<String>>,
        status_polls: AtomicUsize,
    }

    impl FakeProvider {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                price_hr: 1.0,
                status: InstanceStatus::Gone,
                destroy_fails: false,
                list_fails: false,
                listed: StdMutex::new(Vec::new()),
                destroyed: StdMutex::new(Vec::new()),
                status_polls: AtomicUsize::new(0),
            })
        }

        fn with(f: impl FnOnce(&mut Self)) -> Arc<Self> {
            let mut provider = Arc::into_inner(Self::new()).expect("sole owner");
            f(&mut provider);
            Arc::new(provider)
        }

        fn instance(id: &str) -> Instance {
            Instance {
                id: id.to_owned(),
                provider: PROVIDER.to_owned(),
                gpu: "fake-t4".to_owned(),
                price_hr: 1.0,
            }
        }

        fn destroyed(&self) -> Vec<String> {
            self.destroyed.lock().expect("destroyed lock").clone()
        }
    }

    #[async_trait]
    impl Provider for FakeProvider {
        fn name(&self) -> &str {
            PROVIDER
        }

        async fn offers(&self, _gpu: Option<&str>, _max: Option<f64>) -> Result<Vec<Offer>> {
            Ok(vec![Offer {
                id: format!("{PROVIDER}:t4"),
                provider: PROVIDER.to_owned(),
                gpu: "fake-t4".to_owned(),
                price_hr: self.price_hr,
                vram_gb: Some(16),
                region: None,
            }])
        }

        async fn provision(&self, _req: &ProvisionRequest, ctx: &ProvisionContext) -> Result<Instance> {
            let instance = Instance {
                id: format!("i-{}", ctx.worker_id),
                provider: PROVIDER.to_owned(),
                gpu: "fake-t4".to_owned(),
                price_hr: self.price_hr,
            };
            self.listed.lock().expect("listed lock").push(instance.clone());
            Ok(instance)
        }

        async fn destroy(&self, instance_id: &str) -> Result<()> {
            anyhow::ensure!(!self.destroy_fails, "provider refused destroy");
            self.destroyed.lock().expect("destroyed lock").push(instance_id.to_owned());
            self.listed.lock().expect("listed lock").retain(|i| i.id != instance_id);
            Ok(())
        }

        async fn status(&self, _instance_id: &str) -> Result<InstanceStatus> {
            self.status_polls.fetch_add(1, Ordering::Relaxed);
            anyhow::ensure!(self.status != InstanceStatus::Unknown, "status unavailable");
            Ok(self.status)
        }

        async fn list_instances(&self) -> Result<Vec<Instance>> {
            anyhow::ensure!(!self.list_fails, "provider list failed");
            Ok(self.listed.lock().expect("listed lock").clone())
        }
    }

    fn test_config() -> Config {
        Config {
            api_token: "test-api-token".into(),
            join_token: "test-join-token".into(),
            store_spec: ":memory:".into(),
            artifacts_spec: "file:./unused".into(),
            public_url: "http://127.0.0.1:9".into(),
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 9,
            providers: PROVIDER.into(),
            agent_ws_url: "ws://127.0.0.1:9/ws".into(),
            agent_bin: None,
            agent_dir: None,
            min_protocol: 0,
            vast_api_key: None,
            drain_window_min: DRAIN_WINDOW_MIN,
            confirm_cost_threshold: 0.0,
            reconcile_interval: Duration::from_millis(10),
            idle_reap: IDLE_REAP,
            google_client_id: None,
            google_client_secret: None,
            allowed_emails: vec![],
            sysadmin_email: None,
        }
    }

    async fn manager(provider: Arc<FakeProvider>) -> (Arc<LeaseManager>, Arc<dyn Store>) {
        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store.clone(), artifacts, None, None);
        let providers = Arc::new(Providers::of([provider as Arc<dyn Provider>]));
        // Milliseconds instead of the spec's two minutes: the polling loop is
        // the same, it just doesn't hold the test open while it runs.
        let verify = VerifyPolicy {
            timeout: Duration::from_millis(60),
            poll: Duration::from_millis(5),
        };
        (
            LeaseManager::verifying(hub, providers, test_config(), verify),
            store,
        )
    }

    /// A lease that started `age_secs` ago, so a clock pass sees it at whatever
    /// point of its life the test wants.
    fn lease_started(worker: &str, granted_min: f64, age_secs: f64) -> Lease {
        Lease {
            worker_id: WorkerId(worker.into()),
            provider: PROVIDER.into(),
            instance_id: format!("i-{worker}"),
            price_hr: 1.0,
            granted_min,
            drain_window_min: DRAIN_WINDOW_MIN,
            started_at: now() - age_secs,
            state: LeaseState::Active,
            extensions: Vec::new(),
        }
    }

    fn provision_request() -> ProvisionRequest {
        ProvisionRequest {
            provider: PROVIDER.into(),
            lease_min: 10.0,
            offer_id: None,
            gpu: None,
            max_price_hr: None,
        }
    }

    async fn set_cap(store: &Arc<dyn Store>, cap: f64) {
        store
            .set_budget(&Budget {
                scope: BUDGET_SCOPE_GLOBAL.into(),
                cap,
                period: BUDGET_PERIOD_ALL.into(),
                updated_at: now(),
            })
            .await
            .expect("set budget");
    }

    // ---- provisioning ------------------------------------------------------

    #[tokio::test]
    async fn offers_come_from_the_named_provider_and_an_unknown_one_names_what_is_configured() {
        let (leases, _store) = manager(FakeProvider::new()).await;
        assert_eq!(leases.offers(PROVIDER, None, None).await.unwrap().len(), 1);

        let error = leases.offers("nope", None, None).await.unwrap_err();
        assert!(error.to_string().contains("unknown provider"), "unexpected error: {error}");
        assert!(error.to_string().contains(PROVIDER), "must name what is configured: {error}");
    }

    #[tokio::test]
    async fn provision_records_a_lease_and_opens_the_ledger() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;

        let result = leases.provision(&provision_request()).await.expect("provision");
        assert!(result.worker_id.0.starts_with(PROVIDER));
        let lease = store.lease(&result.worker_id).await.unwrap().expect("lease stored");
        assert_eq!(lease.state, LeaseState::Active);
        assert_eq!(lease.granted_min, 10.0);
        assert_eq!(lease.drain_window_min, DRAIN_WINDOW_MIN);

        let ledger = store.ledger_entries().await.unwrap();
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger[0].event, EV_LEASE_START);
        // 10 minutes at $1/hr.
        assert!((ledger[0].cost - 1.0 / 6.0).abs() < 1e-9, "unexpected cost: {}", ledger[0].cost);
    }

    #[tokio::test]
    async fn provision_refuses_before_renting_anything_when_headroom_is_already_gone() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        // A live lease already commits $1; the cap is $0.50. Even a free
        // candidate breaches, so the pre-flight refuses before renting.
        store.create_lease(&lease_started("w-committed", 60.0, 0.0)).await.unwrap();
        set_cap(&store, 0.5).await;

        let error = leases.provision(&provision_request()).await.unwrap_err();
        assert!(error.to_string().contains("budget"), "unexpected error: {error}");
        assert!(provider.destroyed().is_empty(), "nothing was rented, nothing to destroy");
        assert_eq!(store.live_leases().await.unwrap().len(), 1, "only the pre-existing lease");
    }

    #[tokio::test]
    async fn provision_destroys_the_instance_when_the_real_price_breaches_the_budget() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        // Room for a zero-cost candidate, but not for 10 minutes at $1/hr.
        set_cap(&store, 0.01).await;

        let error = leases.provision(&provision_request()).await.unwrap_err();
        assert!(error.to_string().contains("budget"), "unexpected error: {error}");
        assert_eq!(provider.destroyed().len(), 1, "a refused provision must not leave it billing");
        assert!(store.live_leases().await.unwrap().is_empty(), "no lease is recorded");
    }

    #[tokio::test]
    async fn a_refused_provision_whose_destroy_also_fails_still_surfaces_the_refusal() {
        // The instance is left for reconcile to catch; the caller still sees
        // the budget refusal rather than the destroy error.
        let provider = FakeProvider::with(|p| p.destroy_fails = true);
        let (leases, store) = manager(provider.clone()).await;
        set_cap(&store, 0.01).await;

        let error = leases.provision(&provision_request()).await.unwrap_err();
        assert!(error.to_string().contains("budget"), "unexpected error: {error}");
        assert!(provider.destroyed().is_empty());
    }

    // ---- extend ------------------------------------------------------------

    #[tokio::test]
    async fn extend_moves_the_wall_out_and_reactivates_a_draining_lease() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider).await;
        let lease = lease_started("w-drain", 10.0, 0.0);
        store.create_lease(&lease).await.unwrap();
        store.set_lease_state(&lease.worker_id, LeaseState::Draining).await.unwrap();

        let updated = leases
            .extend(&lease.worker_id, 5.0, "more time")
            .await
            .expect("extend")
            .expect("lease exists");
        assert_eq!(updated.total_granted_min(), 15.0);
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Active,
            "a lease whose wall moved back out is no longer draining"
        );
        let ledger = store.ledger_entries().await.unwrap();
        assert_eq!(ledger[0].event, EV_EXTEND);
        assert_eq!(ledger[0].minutes, 5.0);
    }

    #[tokio::test]
    async fn extend_is_a_no_op_for_a_worker_with_no_lease() {
        let (leases, store) = manager(FakeProvider::new()).await;
        let updated = leases.extend(&WorkerId("ghost".into()), 5.0, "why").await.unwrap();
        assert!(updated.is_none());
        assert!(store.ledger_entries().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn extend_refuses_when_the_extra_minutes_would_breach_a_budget() {
        let (leases, store) = manager(FakeProvider::new()).await;
        let lease = lease_started("w-cap", 10.0, 0.0);
        store.create_lease(&lease).await.unwrap();
        set_cap(&store, 0.1).await;

        let error = leases.extend(&lease.worker_id, 60.0, "much more").await.unwrap_err();
        assert!(error.to_string().contains("budget"), "unexpected error: {error}");
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().total_granted_min(),
            10.0,
            "the wall must not have moved"
        );
    }

    // ---- teardown ----------------------------------------------------------

    #[tokio::test]
    async fn teardown_drains_then_destroys_and_closes_the_ledger() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-down", 10.0, 120.0);
        store.create_lease(&lease).await.unwrap();

        let result = leases.teardown(&lease.worker_id, true).await.expect("teardown");
        assert!(result.destroyed);
        assert_eq!(result.status, InstanceStatus::Gone);
        assert_eq!(provider.destroyed(), vec![lease.instance_id.clone()]);
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Destroyed
        );
        let ledger = store.ledger_entries().await.unwrap();
        assert_eq!(ledger[0].event, EV_LEASE_END);
        // 2 minutes elapsed at $1/hr.
        assert!((ledger[0].minutes - 2.0).abs() < 0.1, "unexpected minutes: {}", ledger[0].minutes);
    }

    #[tokio::test]
    async fn teardown_of_an_unknown_worker_is_an_error() {
        let (leases, _store) = manager(FakeProvider::new()).await;
        let error = leases.teardown(&WorkerId("ghost".into()), false).await.unwrap_err();
        assert!(error.to_string().contains("no lease for worker"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn a_destroy_the_provider_will_not_confirm_reports_undestroyed() {
        // The provider still says Running after the destroy call: the lease is
        // closed out either way, but the result says so and reconcile retries.
        let provider = FakeProvider::with(|p| p.status = InstanceStatus::Running);
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-stuck", 10.0, 60.0);
        store.create_lease(&lease).await.unwrap();

        let result = leases.teardown(&lease.worker_id, false).await.expect("teardown");
        assert!(!result.destroyed, "the provider never confirmed it was gone");
        assert_eq!(result.status, InstanceStatus::Running);
        assert!(provider.status_polls.load(Ordering::Relaxed) > 1, "it polled until the deadline");
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Destroyed
        );
    }

    #[tokio::test]
    async fn a_provider_that_cannot_be_polled_reports_unknown() {
        let provider = FakeProvider::with(|p| p.status = InstanceStatus::Unknown);
        let (leases, store) = manager(provider).await;
        let lease = lease_started("w-blind", 10.0, 60.0);
        store.create_lease(&lease).await.unwrap();

        let result = leases.teardown(&lease.worker_id, false).await.expect("teardown");
        assert_eq!(result.status, InstanceStatus::Unknown);
        assert!(!result.destroyed);
    }

    #[tokio::test]
    async fn a_provider_that_refuses_the_destroy_fails_the_teardown() {
        let provider = FakeProvider::with(|p| p.destroy_fails = true);
        let (leases, store) = manager(provider).await;
        let lease = lease_started("w-refused", 10.0, 60.0);
        store.create_lease(&lease).await.unwrap();

        let error = leases.teardown(&lease.worker_id, false).await.unwrap_err();
        assert!(error.to_string().contains("provider destroy"), "unexpected error: {error}");
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Active,
            "an unconfirmed destroy must not be recorded as torn down"
        );
    }

    // ---- the lease clock ---------------------------------------------------

    #[tokio::test]
    async fn the_clock_drains_a_lease_that_has_reached_t_drain() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        // 2-minute lease with a 1-minute drain window: 90s in is past T-drain,
        // short of T-0.
        let lease = lease_started("w-tdrain", 2.0, 90.0);
        store.create_lease(&lease).await.unwrap();

        leases.clock_pass().await.expect("clock pass");
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Draining
        );
        assert!(provider.destroyed().is_empty(), "T-0 has not arrived yet");
    }

    #[tokio::test]
    async fn the_clock_destroys_at_t_zero_whatever_the_agent_is_doing() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        // Past the wall, and no agent is attached to accept a drain.
        let lease = lease_started("w-tzero", 1.0, 120.0);
        store.create_lease(&lease).await.unwrap();

        leases.clock_pass().await.expect("clock pass");
        assert_eq!(provider.destroyed(), vec![lease.instance_id.clone()]);
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Destroyed
        );
        assert!(store.live_leases().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_clock_keeps_ticking_until_the_wall_lands() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-loop", 1.0, 120.0);
        store.create_lease(&lease).await.unwrap();

        let clock = tokio::spawn(leases.clone().run_clock());
        let destroyed = wait_for(|| !provider.destroyed().is_empty()).await;
        clock.abort();
        assert!(destroyed, "the clock loop must reach T-0 on its own");
    }

    // ---- reconcile ---------------------------------------------------------

    #[tokio::test]
    async fn reconcile_kills_an_instance_no_live_lease_owns() {
        let provider = FakeProvider::new();
        provider
            .listed
            .lock()
            .expect("listed lock")
            .push(FakeProvider::instance("i-orphan"));
        let (leases, _store) = manager(provider.clone()).await;

        leases.reconcile_pass().await.expect("reconcile");
        assert_eq!(provider.destroyed(), vec!["i-orphan".to_owned()]);
    }

    #[tokio::test]
    async fn reconcile_leaves_an_instance_a_live_lease_owns_alone() {
        let provider = FakeProvider::new();
        provider
            .listed
            .lock()
            .expect("listed lock")
            .push(FakeProvider::instance("i-w-owned"));
        let (leases, store) = manager(provider.clone()).await;
        store.create_lease(&lease_started("w-owned", 10.0, 0.0)).await.unwrap();

        leases.reconcile_pass().await.expect("reconcile");
        assert!(provider.destroyed().is_empty(), "an owned instance must survive");
    }

    #[tokio::test]
    async fn reconcile_survives_a_provider_that_cannot_be_listed() {
        let provider = FakeProvider::with(|p| p.list_fails = true);
        let (leases, _store) = manager(provider.clone()).await;

        leases.reconcile_pass().await.expect("a listing failure is not fatal");
        assert!(provider.destroyed().is_empty());
    }

    #[tokio::test]
    async fn an_orphan_that_will_not_die_is_left_for_the_next_pass() {
        let provider = FakeProvider::with(|p| {
            p.destroy_fails = true;
            p.listed = StdMutex::new(vec![FakeProvider::instance("i-immortal")]);
        });
        let (leases, _store) = manager(provider.clone()).await;

        leases.reconcile_pass().await.expect("a failed orphan kill is not fatal");
        assert!(provider.destroyed().is_empty());
    }

    #[tokio::test]
    async fn run_reconcile_keeps_sweeping_for_orphans() {
        let provider = FakeProvider::new();
        provider
            .listed
            .lock()
            .expect("listed lock")
            .push(FakeProvider::instance("i-orphan"));
        let (leases, _store) = manager(provider.clone()).await;

        let reconcile = tokio::spawn(leases.clone().run_reconcile());
        let killed = wait_for(|| !provider.destroyed().is_empty()).await;
        reconcile.abort();
        assert!(killed, "the reconcile loop must find the orphan on its own");
    }

    // ---- the idle reaper ---------------------------------------------------

    /// Register a worker so the reaper can see whether it is running anything.
    async fn attach_idle_worker(store: &Arc<dyn Store>, worker: &WorkerId) {
        store
            .worker_joined(worker, &[], &chuk_train_proto::Hardware::default())
            .await
            .expect("worker joined");
    }

    #[tokio::test]
    async fn the_reaper_destroys_a_worker_that_has_sat_idle_past_the_threshold() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-idle", 60.0, 0.0);
        store.create_lease(&lease).await.unwrap();
        attach_idle_worker(&store, &lease.worker_id).await;

        // First pass only starts the idle clock — a lease is a ceiling, not a
        // commitment, but it isn't reaped the instant it goes quiet either.
        leases.clock_pass().await.expect("first pass");
        assert!(provider.destroyed().is_empty(), "not idle long enough yet");

        tokio::time::sleep(IDLE_REAP + Duration::from_millis(20)).await;
        leases.clock_pass().await.expect("second pass");
        assert_eq!(provider.destroyed(), vec![lease.instance_id.clone()]);
        assert_eq!(
            store.lease(&lease.worker_id).await.unwrap().unwrap().state,
            LeaseState::Destroyed
        );
    }

    #[tokio::test]
    async fn the_reaper_leaves_a_worker_alone_while_the_queue_has_work() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-busy", 60.0, 0.0);
        store.create_lease(&lease).await.unwrap();
        attach_idle_worker(&store, &lease.worker_id).await;
        let spec: chuk_train_proto::RunSpec = serde_json::from_value(serde_json::json!({
            "kind": "shell",
            "command": "true",
        }))
        .expect("valid shell spec");
        store.create_run("queued-work", &spec, None, None, None).await.expect("queue a run");

        leases.clock_pass().await.expect("first pass");
        tokio::time::sleep(IDLE_REAP + Duration::from_millis(20)).await;
        leases.clock_pass().await.expect("second pass");
        assert!(
            provider.destroyed().is_empty(),
            "a worker with queued work waiting is not idle"
        );
    }

    #[tokio::test]
    async fn the_idle_clock_restarts_when_a_worker_picks_up_a_run() {
        let provider = FakeProvider::new();
        let (leases, store) = manager(provider.clone()).await;
        let lease = lease_started("w-onoff", 60.0, 0.0);
        store.create_lease(&lease).await.unwrap();
        attach_idle_worker(&store, &lease.worker_id).await;

        leases.clock_pass().await.expect("idle clock starts");
        // It takes a run, which clears the idle clock...
        let spec: chuk_train_proto::RunSpec = serde_json::from_value(serde_json::json!({
            "kind": "shell",
            "command": "true",
        }))
        .expect("valid shell spec");
        let run = store.create_run("busy", &spec, None, None, None).await.expect("run");
        store.set_worker_run(&lease.worker_id, Some(&run)).await.expect("bind run");
        store
            .transition(&run, chuk_train_proto::RunState::Running, Some(&lease.worker_id), None, serde_json::Value::Null)
            .await
            .expect("running");
        tokio::time::sleep(IDLE_REAP + Duration::from_millis(20)).await;
        leases.clock_pass().await.expect("busy pass");
        assert!(provider.destroyed().is_empty(), "it was working, not idle");

        // ...and going quiet again starts the clock from scratch.
        store.set_worker_run(&lease.worker_id, None).await.expect("unbind run");
        leases.clock_pass().await.expect("idle again");
        assert!(provider.destroyed().is_empty(), "the idle clock restarted");
        tokio::time::sleep(IDLE_REAP + Duration::from_millis(20)).await;
        leases.clock_pass().await.expect("reap");
        assert_eq!(provider.destroyed(), vec![lease.instance_id.clone()]);
    }

    /// Poll `done` for up to a second of real time — for the loops that have no
    /// completion signal of their own (`run_clock`, `run_reconcile`).
    async fn wait_for(done: impl Fn() -> bool) -> bool {
        for _ in 0..100 {
            if done() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }
}
