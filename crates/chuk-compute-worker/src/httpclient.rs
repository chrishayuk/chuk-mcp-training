//! Grant-authorised blob transfer against the control plane's REST API.
//! The agent holds only a run-scoped grant token — never the API token.
//!
//! Every transfer is presign-first: the agent asks the control plane where to
//! send/get a blob (`/api/blob_url`). With an S3/R2 backend that is a presigned
//! URL the agent hits directly, so ~500 MB checkpoints never transit the
//! control plane; with the filesystem backend it points back at the control
//! plane and the agent attaches its grant token.

use anyhow::{Context, Result};
use chuk_compute_wire::{BlobMethod, BlobUrlRequest, BlobUrlResponse, API_PREFIX};

const BEARER_PREFIX: &str = "Bearer ";

#[derive(Clone)]
pub struct HttpClient {
    /// Control-plane origin, e.g. `http://127.0.0.1:8700`.
    origin: String,
    token: String,
    http: reqwest::Client,
}

impl HttpClient {
    pub fn new(origin: String, token: String) -> Self {
        Self {
            origin,
            token,
            http: reqwest::Client::new(),
        }
    }

    fn bearer(&self) -> String {
        format!("{BEARER_PREFIX}{}", self.token)
    }

    /// Ask the control plane where to transfer `key` in the given direction.
    async fn blob_url(&self, method: BlobMethod, key: &str) -> Result<BlobUrlResponse> {
        let plan = self
            .http
            .post(format!("{}{API_PREFIX}/blob_url", self.origin))
            .header(reqwest::header::AUTHORIZATION, self.bearer())
            .json(&BlobUrlRequest {
                key: key.to_owned(),
                method,
            })
            .send()
            .await
            .with_context(|| format!("requesting blob url for {key}"))?
            .error_for_status()
            .with_context(|| format!("requesting blob url for {key}"))?
            .json::<BlobUrlResponse>()
            .await
            .with_context(|| format!("parsing blob url for {key}"))?;
        Ok(plan)
    }

    /// Download a blob the grant may read (code unit, resume checkpoint).
    pub async fn fetch(&self, key: &str) -> Result<Vec<u8>> {
        let plan = self.blob_url(BlobMethod::Get, key).await?;
        let mut req = self.http.get(&plan.url);
        if plan.requires_grant_header {
            req = req.header(reqwest::header::AUTHORIZATION, self.bearer());
        }
        let response = req
            .send()
            .await
            .with_context(|| format!("fetching {key}"))?
            .error_for_status()
            .with_context(|| format!("fetching {key}"))?;
        Ok(response.bytes().await?.to_vec())
    }

    /// Upload a blob into the grant's run tree (a checkpoint file).
    pub async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        let plan = self.blob_url(BlobMethod::Put, key).await?;
        let mut req = self.http.put(&plan.url).body(bytes);
        if plan.requires_grant_header {
            req = req.header(reqwest::header::AUTHORIZATION, self.bearer());
        }
        req.send()
            .await
            .with_context(|| format!("uploading {key}"))?
            .error_for_status()
            .with_context(|| format!("uploading {key}"))?;
        Ok(())
    }
}

/// Derive the HTTP origin from the agent's websocket URL:
/// `ws://h:p/ws/agent` → `http://h:p`, `wss://h/ws/agent` → `https://h`.
pub fn origin_from_ws_url(ws_url: &str) -> Result<String> {
    let (scheme, rest) = ws_url
        .split_once("://")
        .context("agent url must include a scheme, e.g. ws://host/ws/agent")?;
    let http_scheme = match scheme {
        "ws" => "http",
        "wss" => "https",
        other => anyhow::bail!("unexpected agent url scheme: {other}"),
    };
    let authority = rest.split('/').next().unwrap_or(rest);
    Ok(format!("{http_scheme}://{authority}"))
}
