//! Vast.ai provider driver (spec §14 M2, §15 E2).
//!
//! Written to the [`Provider`] trait against Vast's documented REST API. It is
//! **not yet verified against the live API** — E2 (rent 15 min, hang the agent,
//! prove destroy at T-0) is what exercises this with real credentials and real
//! money. Until then the tested provider is [`super::MockProvider`]. The shape
//! here — offers → provision (onstart boots the agent) → destroy → status →
//! list — is what E2 will validate and adjust.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chuk_train_proto::{Instance, InstanceStatus, Offer, ProvisionRequest};
use serde::Deserialize;

use super::{Provider, ProvisionContext};

pub const NAME: &str = "vast";
const API_BASE: &str = "https://console.vast.ai/api/v0";
/// Boot script the rented instance runs: fetch the static agent and join.
/// The real binary URL is filled from an env/release in E2 wiring.
const ONSTART_TEMPLATE: &str = "\
curl -fsSL \"$CHUK_AGENT_URL\" -o /usr/local/bin/chuk-train-agent && \
chmod +x /usr/local/bin/chuk-train-agent && \
chuk-train-agent --url '{ws_url}' --token '{join_token}' \
--worker-id '{worker_id}' --lease-min {lease_min} --drain-window-min {drain_window_min}";

pub struct VastProvider {
    api_key: Option<String>,
    http: reqwest::Client,
}

impl VastProvider {
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            http: reqwest::Client::new(),
        }
    }

    fn key(&self) -> Result<&str> {
        self.api_key
            .as_deref()
            .context("CHUK_TRAIN_VAST_API_KEY not set")
    }

    fn url(&self, path: &str) -> Result<String> {
        Ok(format!("{API_BASE}{path}?api_key={}", self.key()?))
    }
}

#[async_trait]
impl Provider for VastProvider {
    fn name(&self) -> &str {
        NAME
    }

    async fn offers(&self, gpu: Option<&str>, max_price_hr: Option<f64>) -> Result<Vec<Offer>> {
        #[derive(Deserialize)]
        struct Bundles {
            offers: Vec<Bundle>,
        }
        #[derive(Deserialize)]
        struct Bundle {
            id: u64,
            gpu_name: String,
            dph_total: f64,
            gpu_ram: Option<f64>,
            geolocation: Option<String>,
        }
        let body: Bundles = self
            .http
            .get(self.url("/bundles/")?)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .context("parsing vast bundles")?;
        Ok(body
            .offers
            .into_iter()
            .filter(|b| {
                gpu.is_none_or(|g| b.gpu_name.to_lowercase().contains(&g.to_lowercase()))
                    && max_price_hr.is_none_or(|m| b.dph_total <= m)
            })
            .map(|b| Offer {
                id: format!("{NAME}:{}", b.id),
                provider: NAME.to_owned(),
                gpu: b.gpu_name,
                price_hr: b.dph_total,
                vram_gb: b.gpu_ram.map(|g| (g / 1024.0) as u64),
                region: b.geolocation,
            })
            .collect())
    }

    async fn provision(&self, req: &ProvisionRequest, ctx: &ProvisionContext) -> Result<Instance> {
        let offer_id = req
            .offer_id
            .as_deref()
            .and_then(|id| id.strip_prefix("vast:").unwrap_or(id).parse::<u64>().ok())
            .context("vast provision needs an offer_id from provider_offers")?;
        let onstart = ONSTART_TEMPLATE
            .replace("{ws_url}", &ctx.ws_url)
            .replace("{join_token}", &ctx.join_token)
            .replace("{worker_id}", &ctx.worker_id)
            .replace("{lease_min}", &req.lease_min.to_string())
            .replace("{drain_window_min}", &ctx.drain_window_min.to_string());

        #[derive(Deserialize)]
        struct Created {
            new_contract: u64,
        }
        let created: Created = self
            .http
            .put(self.url(&format!("/asks/{offer_id}/"))?)
            .json(&serde_json::json!({ "onstart": onstart, "disk": 30 }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
            .context("creating vast instance")?;
        Ok(Instance {
            id: created.new_contract.to_string(),
            provider: NAME.to_owned(),
            gpu: req.gpu.clone().unwrap_or_default(),
            price_hr: req.max_price_hr.unwrap_or_default(),
        })
    }

    async fn destroy(&self, instance_id: &str) -> Result<()> {
        let response = self
            .http
            .delete(self.url(&format!("/instances/{instance_id}/"))?)
            .send()
            .await?;
        // A already-gone instance (404) is a successful destroy — idempotent.
        if response.status().is_success() || response.status().as_u16() == 404 {
            return Ok(());
        }
        bail!("vast destroy failed: {}", response.status());
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceStatus> {
        #[derive(Deserialize)]
        struct One {
            instances: Option<serde_json::Value>,
        }
        let response = self
            .http
            .get(self.url(&format!("/instances/{instance_id}/"))?)
            .send()
            .await?;
        if response.status().as_u16() == 404 {
            return Ok(InstanceStatus::Gone);
        }
        let one: One = response.error_for_status()?.json().await?;
        // A present, non-null instance record means still billing.
        match one.instances {
            Some(v) if !v.is_null() => Ok(InstanceStatus::Running),
            _ => Ok(InstanceStatus::Gone),
        }
    }

    async fn list_instances(&self) -> Result<Vec<Instance>> {
        #[derive(Deserialize)]
        struct Listing {
            instances: Vec<Running>,
        }
        #[derive(Deserialize)]
        struct Running {
            id: u64,
            gpu_name: Option<String>,
            dph_total: Option<f64>,
        }
        let body: Listing = self
            .http
            .get(self.url("/instances/")?)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(body
            .instances
            .into_iter()
            .map(|r| Instance {
                id: r.id.to_string(),
                provider: NAME.to_owned(),
                gpu: r.gpu_name.unwrap_or_default(),
                price_hr: r.dph_total.unwrap_or_default(),
            })
            .collect())
    }
}
