//! Mock provider: launches the agent binary as local processes to stand in
//! for rented instances. This makes provider-verified destroy genuinely real
//! (the OS process is provably gone) and lets the E2 ladder run locally,
//! including the deliberately-hung-agent case (`kill -STOP`), with no money.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chuk_train_proto::{Instance, InstanceStatus, Offer, ProvisionRequest};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use super::{Provider, ProvisionContext};

pub const NAME: &str = "mock";
const AGENT_BIN_NAME: &str = "chuk-compute-worker";
/// Fake catalogue the mock advertises.
const MOCK_OFFERS: [(&str, f64, u64); 2] = [("mock-t4", 0.10, 16), ("mock-a6000", 0.40, 48)];

struct MockInstance {
    child: Child,
    gpu: String,
    price_hr: f64,
}

pub struct MockProvider {
    agent_bin: Option<PathBuf>,
    instances: Mutex<HashMap<String, MockInstance>>,
}

impl MockProvider {
    pub fn new(agent_bin: Option<String>) -> Self {
        Self {
            agent_bin: agent_bin.map(PathBuf::from),
            instances: Mutex::new(HashMap::new()),
        }
    }

    /// Locate the agent binary: explicit env, else next to this executable.
    fn agent_binary(&self) -> Result<PathBuf> {
        if let Some(path) = &self.agent_bin {
            return Ok(path.clone());
        }
        let sibling = std::env::current_exe()?
            .parent()
            .map(|dir| dir.join(AGENT_BIN_NAME))
            .filter(|p| p.exists());
        sibling.context("agent binary not found; set CHUK_TRAIN_AGENT_BIN")
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        NAME
    }

    async fn offers(&self, gpu: Option<&str>, max_price_hr: Option<f64>) -> Result<Vec<Offer>> {
        Ok(MOCK_OFFERS
            .iter()
            .filter(|(name, price, _)| {
                gpu.is_none_or(|g| name.contains(g)) && max_price_hr.is_none_or(|m| *price <= m)
            })
            .map(|(name, price, vram)| Offer {
                id: format!("{NAME}:{name}"),
                provider: NAME.to_owned(),
                gpu: (*name).to_owned(),
                price_hr: *price,
                vram_gb: Some(*vram),
                region: Some("local".to_owned()),
            })
            .collect())
    }

    async fn provision(&self, req: &ProvisionRequest, ctx: &ProvisionContext) -> Result<Instance> {
        let agent = self.agent_binary()?;
        let instance_id = format!("{NAME}-{}", Uuid::new_v4().simple());
        let gpu = req
            .gpu
            .clone()
            .unwrap_or_else(|| MOCK_OFFERS[0].0.to_owned());
        let price_hr = req.max_price_hr.unwrap_or(MOCK_OFFERS[0].1);

        // The launched agent dials back exactly like a real rented worker,
        // registering under the control-plane-chosen id, and self-drains on
        // its own lease clock (belt).
        let child = Command::new(&agent)
            .arg("--url")
            .arg(&ctx.ws_url)
            .arg("--token")
            .arg(&ctx.join_token)
            .arg("--worker-id")
            .arg(&ctx.worker_id)
            .arg("--labels")
            .arg(format!("{NAME},gpu"))
            .arg("--lease-min")
            .arg(req.lease_min.to_string())
            .arg("--drain-window-min")
            .arg(ctx.drain_window_min.to_string())
            .spawn()
            .with_context(|| format!("launching mock agent {}", agent.display()))?;

        info!(instance = %instance_id, gpu, price_hr, "mock instance provisioned");
        self.instances.lock().await.insert(
            instance_id.clone(),
            MockInstance {
                child,
                gpu: gpu.clone(),
                price_hr,
            },
        );
        Ok(Instance {
            id: instance_id,
            provider: NAME.to_owned(),
            gpu,
            price_hr,
        })
    }

    async fn destroy(&self, instance_id: &str) -> Result<()> {
        let mut instances = self.instances.lock().await;
        if let Some(mut instance) = instances.remove(instance_id) {
            // SIGKILL + reap: terminates even a SIGSTOP'd (hung) agent, and
            // guarantees the process is truly gone before we return.
            let _ = instance.child.kill().await;
            info!(instance = %instance_id, "mock instance destroyed");
        }
        Ok(())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceStatus> {
        let mut instances = self.instances.lock().await;
        match instances.get_mut(instance_id) {
            None => Ok(InstanceStatus::Gone),
            Some(instance) => match instance.child.try_wait() {
                Ok(Some(_)) => {
                    instances.remove(instance_id);
                    Ok(InstanceStatus::Gone)
                }
                Ok(None) => Ok(InstanceStatus::Running),
                Err(_) => Ok(InstanceStatus::Unknown),
            },
        }
    }

    async fn list_instances(&self) -> Result<Vec<Instance>> {
        let mut instances = self.instances.lock().await;
        let mut live = Vec::new();
        let ids: Vec<String> = instances.keys().cloned().collect();
        for id in ids {
            let reaped = instances
                .get_mut(&id)
                .and_then(|i| i.child.try_wait().ok().flatten())
                .is_some();
            if reaped {
                instances.remove(&id);
                continue;
            }
            if let Some(instance) = instances.get(&id) {
                live.push(Instance {
                    id,
                    provider: NAME.to_owned(),
                    gpu: instance.gpu.clone(),
                    price_hr: instance.price_hr,
                });
            }
        }
        Ok(live)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// A throwaway executable script standing in for the real agent binary,
    /// so `provision` can launch a genuine local child process without
    /// needing the actual `chuk-compute-worker` build or any GPU
    /// provisioning. It ignores whatever CLI args the mock provider passes
    /// it (mirroring the real agent's flags), same as `MockProvider` expects
    /// of the real one.
    struct FakeAgent {
        path: PathBuf,
    }

    impl FakeAgent {
        /// Stays alive until killed — stands in for a running rented worker.
        fn sleeping() -> Self {
            Self::new("sleep 30\n")
        }

        /// Exits immediately — stands in for a worker that already died.
        fn exiting() -> Self {
            Self::new("exit 0\n")
        }

        fn new(body: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "chuk-mock-agent-test-{}-{}",
                std::process::id(),
                Uuid::new_v4().simple()
            ));
            std::fs::write(&path, format!("#!/bin/sh\n{body}")).expect("write fake agent script");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                    .expect("chmod fake agent script");
            }
            Self { path }
        }
    }

    impl Drop for FakeAgent {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn ctx(token: &str, worker_id: &str) -> ProvisionContext {
        ProvisionContext {
            ws_url: "ws://127.0.0.1:9/agent".to_owned(),
            join_token: token.to_owned(),
            worker_id: worker_id.to_owned(),
            drain_window_min: 1.0,
        }
    }

    fn req(gpu: Option<&str>, max_price_hr: Option<f64>) -> ProvisionRequest {
        ProvisionRequest {
            provider: NAME.to_owned(),
            lease_min: 5.0,
            offer_id: None,
            gpu: gpu.map(str::to_owned),
            max_price_hr,
        }
    }

    fn offer_ids(offers: &[Offer]) -> Vec<&str> {
        offers.iter().map(|o| o.id.as_str()).collect()
    }

    #[test]
    fn name_reports_mock() {
        assert_eq!(MockProvider::new(None).name(), NAME);
    }

    #[tokio::test]
    async fn offers_lists_both_fakes_with_no_filter() {
        let offers = MockProvider::new(None).offers(None, None).await.unwrap();
        assert_eq!(offer_ids(&offers), vec!["mock:mock-t4", "mock:mock-a6000"]);
        assert_eq!(offers[0].provider, NAME);
        assert_eq!(offers[0].price_hr, 0.10);
        assert_eq!(offers[0].vram_gb, Some(16));
        assert_eq!(offers[0].region.as_deref(), Some("local"));
    }

    #[tokio::test]
    async fn offers_filters_by_gpu_substring() {
        let offers = MockProvider::new(None).offers(Some("a6000"), None).await.unwrap();
        assert_eq!(offer_ids(&offers), vec!["mock:mock-a6000"]);
    }

    #[tokio::test]
    async fn offers_filters_by_max_price() {
        let offers = MockProvider::new(None).offers(None, Some(0.15)).await.unwrap();
        assert_eq!(offer_ids(&offers), vec!["mock:mock-t4"]);
    }

    #[tokio::test]
    async fn offers_combined_filters_can_exclude_everything() {
        let offers = MockProvider::new(None)
            .offers(Some("a6000"), Some(0.15))
            .await
            .unwrap();
        assert!(offers.is_empty());
    }

    #[test]
    fn agent_binary_returns_the_explicit_override_verbatim() {
        let provider = MockProvider::new(Some("/opt/fake/agent".to_owned()));
        assert_eq!(provider.agent_binary().unwrap(), PathBuf::from("/opt/fake/agent"));
    }

    #[test]
    fn agent_binary_errors_when_no_sibling_binary_exists() {
        // `cargo test` binaries live in target/{profile}/deps/, never next to
        // the chuk-compute-worker bin output, so the sibling lookup misses.
        let err = MockProvider::new(None).agent_binary().unwrap_err();
        assert!(err.to_string().contains("CHUK_TRAIN_AGENT_BIN"));
    }

    #[tokio::test]
    async fn status_and_list_instances_report_an_unknown_id_as_absent() {
        let provider = MockProvider::new(None);
        assert_eq!(provider.status("no-such-instance").await.unwrap(), InstanceStatus::Gone);
        assert!(provider.list_instances().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn destroying_an_unknown_id_is_a_harmless_no_op() {
        assert!(MockProvider::new(None).destroy("no-such-instance").await.is_ok());
    }

    #[tokio::test]
    async fn provision_surfaces_a_spawn_error_when_the_agent_binary_is_missing() {
        let provider = MockProvider::new(Some("/nonexistent/chuk-mock-agent-test".to_owned()));
        let err = provider
            .provision(&req(None, None), &ctx("tok", "worker"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("launching mock agent"));
    }

    #[tokio::test]
    async fn provision_defaults_gpu_and_price_from_the_first_mock_offer_when_unset() {
        let agent = FakeAgent::sleeping();
        let provider = MockProvider::new(Some(agent.path.to_string_lossy().into_owned()));
        let instance = provider
            .provision(&req(None, None), &ctx("tok", "worker"))
            .await
            .expect("provision");
        assert_eq!(instance.gpu, MOCK_OFFERS[0].0);
        assert_eq!(instance.price_hr, MOCK_OFFERS[0].1);
        provider.destroy(&instance.id).await.unwrap();
    }

    #[tokio::test]
    async fn provision_lease_and_destroy_lifecycle_tracks_a_real_child_process() {
        let agent = FakeAgent::sleeping();
        let provider = MockProvider::new(Some(agent.path.to_string_lossy().into_owned()));
        let instance = provider
            .provision(&req(Some("mock-a6000"), Some(0.40)), &ctx("tok-1", "worker-1"))
            .await
            .expect("provision");
        assert_eq!(instance.provider, NAME);
        assert_eq!(instance.gpu, "mock-a6000");
        assert_eq!(instance.price_hr, 0.40);

        // The child is alive and tracked: status is Running and it shows up
        // in list_instances with the same gpu/price the lease was granted.
        assert_eq!(provider.status(&instance.id).await.unwrap(), InstanceStatus::Running);
        let listed = provider.list_instances().await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, instance.id);
        assert_eq!(listed[0].gpu, "mock-a6000");
        assert_eq!(listed[0].price_hr, 0.40);

        // destroy SIGKILLs and reaps it; the registry forgets it, so status
        // and list_instances both report it gone.
        provider.destroy(&instance.id).await.unwrap();
        assert_eq!(provider.status(&instance.id).await.unwrap(), InstanceStatus::Gone);
        assert!(provider.list_instances().await.unwrap().is_empty());

        // Destroying an already-gone instance is idempotent, not an error.
        provider.destroy(&instance.id).await.unwrap();
    }

    #[tokio::test]
    async fn status_reaps_a_process_that_exited_on_its_own() {
        let agent = FakeAgent::exiting();
        let provider = MockProvider::new(Some(agent.path.to_string_lossy().into_owned()));
        let instance = provider
            .provision(&req(None, None), &ctx("tok-2", "worker-2"))
            .await
            .expect("provision");

        // Poll status (not list_instances) so this exercises status()'s own
        // reap branch specifically.
        let mut status = provider.status(&instance.id).await.unwrap();
        for _ in 0..50 {
            if status == InstanceStatus::Gone {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            status = provider.status(&instance.id).await.unwrap();
        }
        assert_eq!(status, InstanceStatus::Gone);
    }

    #[tokio::test]
    async fn list_instances_reaps_a_process_that_exited_on_its_own() {
        let agent = FakeAgent::exiting();
        let provider = MockProvider::new(Some(agent.path.to_string_lossy().into_owned()));
        provider
            .provision(&req(None, None), &ctx("tok-3", "worker-3"))
            .await
            .expect("provision");

        // Poll list_instances (not status) so this exercises list_instances'
        // own reap branch specifically.
        let mut listed = provider.list_instances().await.unwrap();
        for _ in 0..50 {
            if listed.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
            listed = provider.list_instances().await.unwrap();
        }
        assert!(listed.is_empty(), "expected the exited agent to be reaped");
    }
}
