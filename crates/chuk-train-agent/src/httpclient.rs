//! Grant-authorised blob transfer against the control plane's REST API.
//! The agent holds only a run-scoped grant token — never the API token.

use anyhow::{Context, Result};
use chuk_train_proto::API_PREFIX;

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

    fn url(&self, action: &str, key: &str) -> String {
        format!("{}{API_PREFIX}/{action}/{key}", self.origin)
    }

    /// `GET /api/fetch/<key>` — download a blob the grant may read.
    pub async fn fetch(&self, key: &str) -> Result<Vec<u8>> {
        let response = self
            .http
            .get(self.url("fetch", key))
            .header(
                reqwest::header::AUTHORIZATION,
                format!("{BEARER_PREFIX}{}", self.token),
            )
            .send()
            .await
            .with_context(|| format!("fetching {key}"))?;
        let response = response
            .error_for_status()
            .with_context(|| format!("fetching {key}"))?;
        Ok(response.bytes().await?.to_vec())
    }

    /// `PUT /api/upload/<key>` — upload a blob into the grant's run tree.
    pub async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        self.http
            .put(self.url("upload", key))
            .header(
                reqwest::header::AUTHORIZATION,
                format!("{BEARER_PREFIX}{}", self.token),
            )
            .body(bytes)
            .send()
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
