//! Control-plane configuration, sourced entirely from environment variables.

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use anyhow::{Context, Result};
use chuk_train_proto::{
    env, AGENT_WS_PATH, DEFAULT_DRAIN_WINDOW_MIN, DEFAULT_IDLE_REAP, DEFAULT_PORT,
    DEFAULT_RECONCILE_INTERVAL,
};

const DEFAULT_STORE_SPEC: &str = "sqlite:chuk_train.db";
const DEFAULT_ARTIFACTS_SPEC: &str = "file:./chuk_train_artifacts";
const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
/// PaaS convention honoured when CHUK_TRAIN_PORT is unset.
const FALLBACK_PORT_VAR: &str = "PORT";
/// Public base URL for building fetchable artifact URLs (spec §6 artifact_url).
const PUBLIC_URL_VAR: &str = "CHUK_TRAIN_PUBLIC_URL";
const DEFAULT_PROVIDERS: &str = "mock";

#[derive(Debug, Clone)]
pub struct Config {
    pub api_token: String,
    pub join_token: String,
    /// Store backend spec: `sqlite:path.db`, bare path (SQLite), `redis:` reserved.
    pub store_spec: String,
    /// Artifact blob backend spec: `file:/path` (bare path ok), `s3:`/`r2:` reserved.
    pub artifacts_spec: String,
    /// Externally reachable base URL, used to build artifact fetch URLs.
    pub public_url: String,
    pub host: IpAddr,
    pub port: u16,
    // -- M2 lease/provider config -----------------------------------------
    /// Comma-separated provider selection (e.g. `mock`, `mock,vast`).
    pub providers: String,
    /// Websocket URL provisioned workers dial back on.
    pub agent_ws_url: String,
    /// Agent binary the mock provider launches (None → auto-detect).
    pub agent_bin: Option<String>,
    /// Vast API key (VastProvider only).
    pub vast_api_key: Option<String>,
    /// Minutes reserved at the end of a lease for drain.
    pub drain_window_min: f64,
    /// How often the reconcile loop runs.
    pub reconcile_interval: Duration,
    /// Idle-reaper threshold.
    pub idle_reap: Duration,
    // -- dashboard auth (Google OAuth) ------------------------------------
    /// Google OAuth client id/secret; when both are set, the dashboard is
    /// gated behind Google sign-in. When unset, the dashboard falls back to
    /// the API-token box (local dev).
    pub google_client_id: Option<String>,
    pub google_client_secret: Option<String>,
    /// Emails permitted to view the dashboard.
    pub allowed_emails: Vec<String>,
}

impl Config {
    /// Dashboard Google sign-in is enforced only when a client is configured.
    pub fn auth_enabled(&self) -> bool {
        self.google_client_id.is_some() && self.google_client_secret.is_some()
    }
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_token = required_token(env::API_TOKEN)?;
        let join_token = required_token(env::JOIN_TOKEN)?;
        let store_spec = std::env::var(env::STORE_URL)
            .or_else(|_| std::env::var(env::DB_PATH))
            .unwrap_or_else(|_| DEFAULT_STORE_SPEC.to_owned());
        let artifacts_spec =
            std::env::var(env::ARTIFACTS_DIR).unwrap_or_else(|_| DEFAULT_ARTIFACTS_SPEC.to_owned());
        let host = match std::env::var(env::HOST) {
            Ok(raw) => raw
                .parse()
                .with_context(|| format!("parsing {}", env::HOST))?,
            Err(_) => DEFAULT_HOST,
        };
        let port = match std::env::var(env::PORT).or_else(|_| std::env::var(FALLBACK_PORT_VAR)) {
            Ok(raw) => raw
                .parse()
                .with_context(|| format!("parsing {}", env::PORT))?,
            Err(_) => DEFAULT_PORT,
        };
        // 0.0.0.0 is a bind address, not a dialable one — advertise loopback
        // for local dev when no explicit public URL is set.
        let public_url =
            std::env::var(PUBLIC_URL_VAR).unwrap_or_else(|_| format!("http://127.0.0.1:{port}"));

        let providers =
            std::env::var(env::PROVIDERS).unwrap_or_else(|_| DEFAULT_PROVIDERS.to_owned());
        let agent_ws_url = std::env::var(env::AGENT_WS_URL)
            .unwrap_or_else(|_| format!("ws://127.0.0.1:{port}{AGENT_WS_PATH}"));
        let agent_bin = std::env::var(env::AGENT_BIN).ok();
        let vast_api_key = std::env::var(env::VAST_API_KEY).ok();
        let reconcile_interval =
            duration_from_env(env::RECONCILE_INTERVAL_S, DEFAULT_RECONCILE_INTERVAL)?;
        let idle_reap = duration_from_env(env::IDLE_REAP_S, DEFAULT_IDLE_REAP)?;
        let drain_window_min = match std::env::var(env::DRAIN_WINDOW_MIN) {
            Ok(raw) => raw
                .parse()
                .with_context(|| format!("parsing {}", env::DRAIN_WINDOW_MIN))?,
            Err(_) => DEFAULT_DRAIN_WINDOW_MIN,
        };

        let allowed_emails = std::env::var(env::ALLOWED_EMAILS)
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();

        Ok(Self {
            api_token,
            join_token,
            store_spec,
            artifacts_spec,
            public_url,
            host,
            port,
            providers,
            agent_ws_url,
            agent_bin,
            vast_api_key,
            drain_window_min,
            reconcile_interval,
            idle_reap,
            google_client_id: std::env::var(env::GOOGLE_CLIENT_ID)
                .ok()
                .filter(|s| !s.is_empty()),
            google_client_secret: std::env::var(env::GOOGLE_CLIENT_SECRET)
                .ok()
                .filter(|s| !s.is_empty()),
            allowed_emails,
        })
    }
}

fn duration_from_env(var: &str, default: Duration) -> Result<Duration> {
    match std::env::var(var) {
        Ok(raw) => {
            let secs: f64 = raw.parse().with_context(|| format!("parsing {var}"))?;
            Ok(Duration::from_secs_f64(secs))
        }
        Err(_) => Ok(default),
    }
}

/// Tokens are required: a control plane that silently generates its own
/// credentials invites a deployment where nobody knows them. Fail loudly.
fn required_token(var: &str) -> Result<String> {
    let value = std::env::var(var).with_context(|| format!("{var} must be set"))?;
    anyhow::ensure!(!value.trim().is_empty(), "{var} must not be empty");
    Ok(value)
}
