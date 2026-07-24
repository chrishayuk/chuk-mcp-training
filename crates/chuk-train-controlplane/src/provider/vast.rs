//! Vast.ai provider driver (spec §14 M2, §15 E2).
//!
//! Written to the [`Provider`] trait against Vast's documented REST API. It is
//! **not yet verified against the live API** — E2 (rent 15 min, hang the agent,
//! prove destroy at T-0) is what exercises this with real credentials and real
//! money. Until then the tested provider is [`super::MockProvider`]. The shape
//! here — offers → provision (onstart boots the agent) → destroy → status →
//! list — is what E2 will validate and adjust.
//!
//! The request/response shape parsing, offer filtering + price mapping, and
//! error mapping below are split out into plain functions so they can be unit
//! tested without a real HTTP call to vast.ai; only the thin `Provider` trait
//! methods do actual I/O.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chuk_train_proto::{Instance, InstanceStatus, Offer, ProvisionRequest};
use serde::Deserialize;

use super::{Provider, ProvisionContext};

pub const NAME: &str = "vast";
const API_BASE: &str = "https://console.vast.ai/api/v0";
/// Boot script the rented instance runs: the control plane's one-shot installer
/// detects the box's target, downloads + verifies the matching worker, and joins
/// (chuk-compute M2). `{cp_url}` is the control plane's HTTP base.
const ONSTART_TEMPLATE: &str = "\
curl -fsSL '{cp_url}/install.sh' | sh -s -- \
--cp '{cp_url}' --token '{join_token}' \
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

/// Parse the numeric Vast ask id out of an offer id like `vast:12345` (a bare
/// `12345` is also accepted). `provider_offers` is what mints these ids.
fn parse_offer_id(offer_id: Option<&str>) -> Result<u64> {
    offer_id
        .and_then(|id| id.strip_prefix("vast:").unwrap_or(id).parse::<u64>().ok())
        .context("vast provision needs an offer_id from provider_offers")
}

/// Derive the control plane's HTTP base from its agent websocket URL (what
/// `install.sh` needs to reach it): swap the ws(s) scheme for http(s) and drop
/// the agent websocket path suffix.
fn derive_cp_url(ws_url: &str) -> String {
    ws_url
        .replace("wss://", "https://")
        .replace("ws://", "http://")
        .trim_end_matches(chuk_train_proto::AGENT_WS_PATH)
        .to_owned()
}

/// Fill the onstart template's placeholders for a specific lease.
fn render_onstart(
    cp_url: &str,
    join_token: &str,
    worker_id: &str,
    lease_min: f64,
    drain_window_min: f64,
) -> String {
    ONSTART_TEMPLATE
        .replace("{cp_url}", cp_url)
        .replace("{join_token}", join_token)
        .replace("{worker_id}", worker_id)
        .replace("{lease_min}", &lease_min.to_string())
        .replace("{drain_window_min}", &drain_window_min.to_string())
}

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

/// Parse a `/bundles/` response body and apply the same gpu/price filters
/// `offers` is asked for.
fn parse_offers(body: &[u8], gpu: Option<&str>, max_price_hr: Option<f64>) -> Result<Vec<Offer>> {
    let body: Bundles = serde_json::from_slice(body).context("parsing vast bundles")?;
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

#[derive(Deserialize)]
struct Created {
    new_contract: u64,
}

/// Parse a successful `PUT /asks/{id}/` response into the provisioned
/// instance. `gpu`/`price_hr` come from the request, not the response — Vast
/// doesn't echo them back at creation time.
fn parse_created_instance(body: &[u8], gpu: String, price_hr: f64) -> Result<Instance> {
    let created: Created = serde_json::from_slice(body).context("creating vast instance")?;
    Ok(Instance {
        id: created.new_contract.to_string(),
        provider: NAME.to_owned(),
        gpu,
        price_hr,
    })
}

/// Map a `DELETE /instances/{id}/` response status to a destroy outcome.
/// Idempotent: an already-gone instance (404) counts as destroyed.
fn destroy_outcome(status: reqwest::StatusCode) -> Result<()> {
    if status.is_success() || status.as_u16() == 404 {
        return Ok(());
    }
    bail!("vast destroy failed: {status}");
}

#[derive(Deserialize)]
struct InstanceRecord {
    instances: Option<serde_json::Value>,
}

/// Parse a `GET /instances/{id}/` body (for the non-404 case — 404 itself is
/// handled before parsing, since a missing instance has no body to parse). A
/// present, non-null instance record means it is still billing.
fn parse_status_response(body: &[u8]) -> Result<InstanceStatus> {
    let record: InstanceRecord =
        serde_json::from_slice(body).context("parsing vast instance status")?;
    match record.instances {
        Some(v) if !v.is_null() => Ok(InstanceStatus::Running),
        _ => Ok(InstanceStatus::Gone),
    }
}

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

/// Parse a `GET /instances/` fleet listing.
fn parse_instance_list(body: &[u8]) -> Result<Vec<Instance>> {
    let listing: Listing = serde_json::from_slice(body).context("parsing vast instance list")?;
    Ok(listing
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

#[async_trait]
impl Provider for VastProvider {
    fn name(&self) -> &str {
        NAME
    }

    async fn offers(&self, gpu: Option<&str>, max_price_hr: Option<f64>) -> Result<Vec<Offer>> {
        let body = self
            .http
            .get(self.url("/bundles/")?)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await
            .context("reading vast bundles response")?;
        parse_offers(&body, gpu, max_price_hr)
    }

    async fn provision(&self, req: &ProvisionRequest, ctx: &ProvisionContext) -> Result<Instance> {
        let offer_id = parse_offer_id(req.offer_id.as_deref())?;
        // install.sh needs the HTTP base; derive it from the websocket URL.
        let cp_url = derive_cp_url(&ctx.ws_url);
        let onstart = render_onstart(
            &cp_url,
            &ctx.join_token,
            &ctx.worker_id,
            req.lease_min,
            ctx.drain_window_min,
        );

        let body = self
            .http
            .put(self.url(&format!("/asks/{offer_id}/"))?)
            .json(&serde_json::json!({ "onstart": onstart, "disk": 30 }))
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await
            .context("reading vast instance creation response")?;
        parse_created_instance(
            &body,
            req.gpu.clone().unwrap_or_default(),
            req.max_price_hr.unwrap_or_default(),
        )
    }

    async fn destroy(&self, instance_id: &str) -> Result<()> {
        let response = self
            .http
            .delete(self.url(&format!("/instances/{instance_id}/"))?)
            .send()
            .await?;
        destroy_outcome(response.status())
    }

    async fn status(&self, instance_id: &str) -> Result<InstanceStatus> {
        let response = self
            .http
            .get(self.url(&format!("/instances/{instance_id}/"))?)
            .send()
            .await?;
        if response.status().as_u16() == 404 {
            return Ok(InstanceStatus::Gone);
        }
        let body = response.error_for_status()?.bytes().await?;
        parse_status_response(&body)
    }

    async fn list_instances(&self) -> Result<Vec<Instance>> {
        let body = self
            .http
            .get(self.url("/instances/")?)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        parse_instance_list(&body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_reports_vast() {
        assert_eq!(VastProvider::new(None).name(), NAME);
    }

    #[test]
    fn key_returns_the_configured_api_key() {
        assert_eq!(
            VastProvider::new(Some("secret-1".to_owned())).key().unwrap(),
            "secret-1"
        );
    }

    #[test]
    fn key_errors_with_an_actionable_message_when_unset() {
        let err = VastProvider::new(None).key().unwrap_err();
        assert!(err.to_string().contains("CHUK_TRAIN_VAST_API_KEY"));
    }

    #[test]
    fn url_embeds_the_path_and_api_key() {
        let provider = VastProvider::new(Some("secret-1".to_owned()));
        assert_eq!(
            provider.url("/bundles/").unwrap(),
            "https://console.vast.ai/api/v0/bundles/?api_key=secret-1"
        );
    }

    #[test]
    fn url_propagates_the_missing_key_error() {
        assert!(VastProvider::new(None).url("/bundles/").is_err());
    }

    #[test]
    fn parse_offer_id_accepts_the_provider_prefixed_form() {
        assert_eq!(parse_offer_id(Some("vast:4821")).unwrap(), 4821);
    }

    #[test]
    fn parse_offer_id_accepts_a_bare_numeric_id() {
        assert_eq!(parse_offer_id(Some("4821")).unwrap(), 4821);
    }

    #[test]
    fn parse_offer_id_rejects_missing_or_malformed_ids() {
        assert!(parse_offer_id(None).is_err());
        assert!(parse_offer_id(Some("vast:not-a-number")).is_err());
        assert!(parse_offer_id(Some("")).is_err());
    }

    #[test]
    fn derive_cp_url_swaps_the_websocket_scheme_and_drops_the_agent_path() {
        assert_eq!(
            derive_cp_url(&format!("wss://cp.example.com{}", chuk_train_proto::AGENT_WS_PATH)),
            "https://cp.example.com"
        );
        assert_eq!(
            derive_cp_url(&format!("ws://127.0.0.1:8080{}", chuk_train_proto::AGENT_WS_PATH)),
            "http://127.0.0.1:8080"
        );
    }

    #[test]
    fn render_onstart_fills_every_placeholder_into_the_install_command() {
        let script = render_onstart("https://cp.example.com", "tok-xyz", "worker-9", 90.0, 5.0);
        assert!(script.contains("--cp 'https://cp.example.com'"));
        assert!(script.contains("--token 'tok-xyz'"));
        assert!(script.contains("--worker-id 'worker-9'"));
        assert!(script.contains("--lease-min 90"));
        assert!(script.contains("--drain-window-min 5"));
    }

    #[test]
    fn parse_offers_maps_bundle_fields_into_offers() {
        let body = br#"{"offers":[
            {"id":101,"gpu_name":"RTX 4090","dph_total":0.35,"gpu_ram":24576.0,"geolocation":"US-CA"},
            {"id":102,"gpu_name":"A100 SXM4","dph_total":1.10,"gpu_ram":81920.0,"geolocation":null}
        ]}"#;
        let offers = parse_offers(body, None, None).unwrap();
        assert_eq!(offers.len(), 2);
        assert_eq!(offers[0].id, "vast:101");
        assert_eq!(offers[0].provider, NAME);
        assert_eq!(offers[0].gpu, "RTX 4090");
        assert_eq!(offers[0].price_hr, 0.35);
        assert_eq!(offers[0].vram_gb, Some(24));
        assert_eq!(offers[0].region.as_deref(), Some("US-CA"));
        assert_eq!(offers[1].vram_gb, Some(80));
        assert_eq!(offers[1].region, None);
    }

    #[test]
    fn parse_offers_filters_by_gpu_substring_case_insensitively() {
        let body = br#"{"offers":[
            {"id":1,"gpu_name":"RTX 4090","dph_total":0.35,"gpu_ram":null,"geolocation":null},
            {"id":2,"gpu_name":"A100 SXM4","dph_total":1.10,"gpu_ram":null,"geolocation":null}
        ]}"#;
        let offers = parse_offers(body, Some("a100"), None).unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].gpu, "A100 SXM4");
    }

    #[test]
    fn parse_offers_filters_by_max_price() {
        let body = br#"{"offers":[
            {"id":1,"gpu_name":"RTX 4090","dph_total":0.35,"gpu_ram":null,"geolocation":null},
            {"id":2,"gpu_name":"A100 SXM4","dph_total":1.10,"gpu_ram":null,"geolocation":null}
        ]}"#;
        let offers = parse_offers(body, None, Some(0.5)).unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].gpu, "RTX 4090");
    }

    #[test]
    fn parse_offers_rejects_malformed_json() {
        assert!(parse_offers(b"not json", None, None).is_err());
    }

    #[test]
    fn parse_created_instance_uses_the_contract_id_and_requested_gpu_price() {
        let body = br#"{"new_contract": 555}"#;
        let instance = parse_created_instance(body, "RTX 4090".to_owned(), 0.35).unwrap();
        assert_eq!(instance.id, "555");
        assert_eq!(instance.provider, NAME);
        assert_eq!(instance.gpu, "RTX 4090");
        assert_eq!(instance.price_hr, 0.35);
    }

    #[test]
    fn parse_created_instance_rejects_malformed_json() {
        assert!(parse_created_instance(b"{}", "x".into(), 0.0).is_err());
    }

    #[test]
    fn destroy_outcome_treats_success_and_404_as_already_gone() {
        assert!(destroy_outcome(reqwest::StatusCode::OK).is_ok());
        assert!(destroy_outcome(reqwest::StatusCode::NO_CONTENT).is_ok());
        assert!(destroy_outcome(reqwest::StatusCode::NOT_FOUND).is_ok());
    }

    #[test]
    fn destroy_outcome_bails_on_other_statuses() {
        let err = destroy_outcome(reqwest::StatusCode::INTERNAL_SERVER_ERROR).unwrap_err();
        assert!(err.to_string().contains("vast destroy failed"));
    }

    #[test]
    fn parse_status_response_running_when_instance_record_present() {
        let body = br#"{"instances": {"id": 1, "actual_status": "running"}}"#;
        assert_eq!(parse_status_response(body).unwrap(), InstanceStatus::Running);
    }

    #[test]
    fn parse_status_response_gone_when_instance_record_is_null_or_absent() {
        assert_eq!(
            parse_status_response(br#"{"instances": null}"#).unwrap(),
            InstanceStatus::Gone
        );
        assert_eq!(parse_status_response(br#"{}"#).unwrap(), InstanceStatus::Gone);
    }

    #[test]
    fn parse_status_response_rejects_malformed_json() {
        assert!(parse_status_response(b"???").is_err());
    }

    #[test]
    fn parse_instance_list_maps_running_instances() {
        let body = br#"{"instances": [
            {"id": 7, "gpu_name": "RTX 3090", "dph_total": 0.28},
            {"id": 8, "gpu_name": null, "dph_total": null}
        ]}"#;
        let instances = parse_instance_list(body).unwrap();
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].id, "7");
        assert_eq!(instances[0].provider, NAME);
        assert_eq!(instances[0].gpu, "RTX 3090");
        assert_eq!(instances[0].price_hr, 0.28);
        assert_eq!(instances[1].gpu, "");
        assert_eq!(instances[1].price_hr, 0.0);
    }

    #[test]
    fn parse_instance_list_handles_an_empty_fleet() {
        assert_eq!(parse_instance_list(br#"{"instances": []}"#).unwrap(), Vec::new());
    }

    #[test]
    fn parse_instance_list_rejects_malformed_json() {
        assert!(parse_instance_list(b"nope").is_err());
    }
}
