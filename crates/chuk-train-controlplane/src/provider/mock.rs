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
