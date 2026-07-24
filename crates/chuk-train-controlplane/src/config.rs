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
    /// Worker binary the mock provider launches (None → auto-detect).
    pub agent_bin: Option<String>,
    /// Directory of per-target worker binaries served for download (M2). None
    /// → the default image path; a request for an absent target 404s.
    pub agent_dir: Option<String>,
    /// Minimum worker protocol version accepted at handshake (M3.3). A worker
    /// below it is rejected; a persistent one is told to self-update.
    pub min_protocol: u32,
    /// Vast API key (VastProvider only).
    pub vast_api_key: Option<String>,
    /// Minutes reserved at the end of a lease for drain.
    pub drain_window_min: f64,
    /// Dollars above which a submission's worst-case cost estimate requires
    /// confirm_cost=true (spec §8 pre-flight).
    pub confirm_cost_threshold: f64,
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
    /// Bootstrap sysadmin email, seeded on startup (falls back to the first
    /// allowed email). Only seeded when that user doesn't already exist.
    pub sysadmin_email: Option<String>,
}

impl Config {
    /// Dashboard Google sign-in is enforced only when a client is configured
    /// *and* an allowlist says who may in. Requiring the allowlist keeps the
    /// Drive archive tier — which reuses the same client id/secret to refresh
    /// its token — from accidentally gating the dashboard when those creds are
    /// present for storage alone, and stops a client without an allowlist from
    /// locking everyone out.
    pub fn auth_enabled(&self) -> bool {
        self.google_client_id.is_some()
            && self.google_client_secret.is_some()
            && !self.allowed_emails.is_empty()
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
        let agent_dir = std::env::var(env::AGENT_DIR).ok();
        let min_protocol = std::env::var(env::MIN_PROTOCOL)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(chuk_compute_wire::PROTOCOL_VERSION);
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
        let confirm_cost_threshold = match std::env::var(env::CONFIRM_COST_THRESHOLD) {
            Ok(raw) => raw
                .parse()
                .with_context(|| format!("parsing {}", env::CONFIRM_COST_THRESHOLD))?,
            Err(_) => chuk_train_proto::DEFAULT_CONFIRM_COST_THRESHOLD,
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
            agent_dir,
            min_protocol,
            vast_api_key,
            drain_window_min,
            confirm_cost_threshold,
            reconcile_interval,
            idle_reap,
            google_client_id: std::env::var(env::GOOGLE_CLIENT_ID)
                .ok()
                .filter(|s| !s.is_empty()),
            google_client_secret: std::env::var(env::GOOGLE_CLIENT_SECRET)
                .ok()
                .filter(|s| !s.is_empty()),
            allowed_emails,
            sysadmin_email: std::env::var(env::SYSADMIN_EMAIL)
                .ok()
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty()),
        })
    }

    /// The email to seed as the bootstrap sysadmin on startup, if any.
    pub fn bootstrap_sysadmin(&self) -> Option<String> {
        self.sysadmin_email
            .clone()
            .or_else(|| self.allowed_emails.first().cloned())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes every test in this module: `Config::from_env` reads
    /// process-wide env vars, and `cargo test` runs tests on multiple
    /// threads within the same process, so unsynchronized tests would
    /// stomp on each other's env state.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Every env var `Config::from_env` reads, including the `PORT`
    /// PaaS-convention fallback. Cleared at the top of every test so a
    /// leftover value from a previous test (or the real process env —
    /// never the repo's `.env` file, which is not loaded here) can't
    /// leak into an assertion.
    const ALL_VARS: &[&str] = &[
        env::API_TOKEN,
        env::JOIN_TOKEN,
        env::STORE_URL,
        env::DB_PATH,
        env::ARTIFACTS_DIR,
        PUBLIC_URL_VAR,
        env::HOST,
        env::PORT,
        FALLBACK_PORT_VAR,
        env::PROVIDERS,
        env::AGENT_WS_URL,
        env::AGENT_BIN,
        env::AGENT_DIR,
        env::MIN_PROTOCOL,
        env::VAST_API_KEY,
        env::RECONCILE_INTERVAL_S,
        env::IDLE_REAP_S,
        env::DRAIN_WINDOW_MIN,
        env::CONFIRM_COST_THRESHOLD,
        env::ALLOWED_EMAILS,
        env::GOOGLE_CLIENT_ID,
        env::GOOGLE_CLIENT_SECRET,
        env::SYSADMIN_EMAIL,
    ];

    fn clear_env() {
        for var in ALL_VARS {
            std::env::remove_var(var);
        }
    }

    fn set_required_tokens() {
        std::env::set_var(env::API_TOKEN, "api-secret");
        std::env::set_var(env::JOIN_TOKEN, "join-secret");
    }

    /// A fully-populated `Config` for exercising the pure methods
    /// (`auth_enabled`, `bootstrap_sysadmin`) without touching env vars.
    fn base_config() -> Config {
        Config {
            api_token: "t".into(),
            join_token: "j".into(),
            store_spec: "sqlite:test.db".into(),
            artifacts_spec: "file:/tmp/artifacts".into(),
            public_url: "http://127.0.0.1:8700".into(),
            host: DEFAULT_HOST,
            port: DEFAULT_PORT,
            providers: DEFAULT_PROVIDERS.into(),
            agent_ws_url: "ws://127.0.0.1:8700/ws/agent".into(),
            agent_bin: None,
            agent_dir: None,
            min_protocol: 1,
            vast_api_key: None,
            drain_window_min: DEFAULT_DRAIN_WINDOW_MIN,
            confirm_cost_threshold: chuk_train_proto::DEFAULT_CONFIRM_COST_THRESHOLD,
            reconcile_interval: DEFAULT_RECONCILE_INTERVAL,
            idle_reap: DEFAULT_IDLE_REAP,
            google_client_id: None,
            google_client_secret: None,
            allowed_emails: Vec::new(),
            sysadmin_email: None,
        }
    }

    #[test]
    fn missing_api_token_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::API_TOKEN));

        clear_env();
    }

    #[test]
    fn empty_or_whitespace_api_token_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();

        std::env::set_var(env::API_TOKEN, "");
        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains("must not be empty"));

        std::env::set_var(env::API_TOKEN, "   ");
        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains("must not be empty"));

        clear_env();
    }

    #[test]
    fn missing_join_token_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var(env::API_TOKEN, "api-secret");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::JOIN_TOKEN));

        clear_env();
    }

    #[test]
    fn defaults_when_optional_vars_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();

        let cfg = Config::from_env().unwrap();

        assert_eq!(cfg.api_token, "api-secret");
        assert_eq!(cfg.join_token, "join-secret");
        assert_eq!(cfg.store_spec, "sqlite:chuk_train.db");
        assert_eq!(cfg.artifacts_spec, "file:./chuk_train_artifacts");
        assert_eq!(cfg.host, IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(cfg.port, 8700);
        assert_eq!(cfg.public_url, "http://127.0.0.1:8700");
        assert_eq!(cfg.providers, "mock");
        assert_eq!(cfg.agent_ws_url, "ws://127.0.0.1:8700/ws/agent");
        assert_eq!(cfg.agent_bin, None);
        assert_eq!(cfg.agent_dir, None);
        assert_eq!(cfg.min_protocol, chuk_compute_wire::PROTOCOL_VERSION);
        assert_eq!(cfg.vast_api_key, None);
        assert_eq!(cfg.drain_window_min, 5.0);
        assert_eq!(cfg.confirm_cost_threshold, 5.0);
        assert_eq!(cfg.reconcile_interval, Duration::from_secs(600));
        assert_eq!(cfg.idle_reap, Duration::from_secs(600));
        assert_eq!(cfg.google_client_id, None);
        assert_eq!(cfg.google_client_secret, None);
        assert!(cfg.allowed_emails.is_empty());
        assert_eq!(cfg.sysadmin_email, None);
        assert!(!cfg.auth_enabled());
        assert_eq!(cfg.bootstrap_sysadmin(), None);

        clear_env();
    }

    #[test]
    fn full_custom_config_overrides_all_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::STORE_URL, "sqlite:explicit.db");
        std::env::set_var(env::DB_PATH, "should-be-ignored.db");
        std::env::set_var(env::ARTIFACTS_DIR, "file:/srv/artifacts");
        std::env::set_var(PUBLIC_URL_VAR, "https://train.example.com");
        std::env::set_var(env::HOST, "127.0.0.1");
        std::env::set_var(env::PORT, "9100");
        std::env::set_var(FALLBACK_PORT_VAR, "9999"); // CHUK_TRAIN_PORT wins
        std::env::set_var(env::PROVIDERS, "mock,vast");
        std::env::set_var(env::AGENT_WS_URL, "wss://train.example.com/ws/agent");
        std::env::set_var(env::AGENT_BIN, "/opt/chuk-worker");
        std::env::set_var(env::AGENT_DIR, "/opt/chuk-worker-bins");
        std::env::set_var(env::MIN_PROTOCOL, "3");
        std::env::set_var(env::VAST_API_KEY, "vast-key");
        std::env::set_var(env::RECONCILE_INTERVAL_S, "30");
        std::env::set_var(env::IDLE_REAP_S, "15.5");
        std::env::set_var(env::DRAIN_WINDOW_MIN, "12.5");
        std::env::set_var(env::CONFIRM_COST_THRESHOLD, "42");
        std::env::set_var(env::ALLOWED_EMAILS, " Foo@Example.com ,, bar@EXAMPLE.com,   ");
        std::env::set_var(env::GOOGLE_CLIENT_ID, "client-id");
        std::env::set_var(env::GOOGLE_CLIENT_SECRET, "client-secret");
        std::env::set_var(env::SYSADMIN_EMAIL, " Admin@Example.COM ");

        let cfg = Config::from_env().unwrap();

        assert_eq!(cfg.store_spec, "sqlite:explicit.db");
        assert_eq!(cfg.artifacts_spec, "file:/srv/artifacts");
        assert_eq!(cfg.public_url, "https://train.example.com");
        assert_eq!(cfg.host, IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(cfg.port, 9100);
        assert_eq!(cfg.providers, "mock,vast");
        assert_eq!(cfg.agent_ws_url, "wss://train.example.com/ws/agent");
        assert_eq!(cfg.agent_bin.as_deref(), Some("/opt/chuk-worker"));
        assert_eq!(cfg.agent_dir.as_deref(), Some("/opt/chuk-worker-bins"));
        assert_eq!(cfg.min_protocol, 3);
        assert_eq!(cfg.vast_api_key.as_deref(), Some("vast-key"));
        assert_eq!(cfg.reconcile_interval, Duration::from_secs_f64(30.0));
        assert_eq!(cfg.idle_reap, Duration::from_secs_f64(15.5));
        assert_eq!(cfg.drain_window_min, 12.5);
        assert_eq!(cfg.confirm_cost_threshold, 42.0);
        assert_eq!(
            cfg.allowed_emails,
            vec!["foo@example.com".to_string(), "bar@example.com".to_string()]
        );
        assert_eq!(cfg.google_client_id.as_deref(), Some("client-id"));
        assert_eq!(cfg.google_client_secret.as_deref(), Some("client-secret"));
        assert_eq!(cfg.sysadmin_email.as_deref(), Some("admin@example.com"));
        assert!(cfg.auth_enabled());
        assert_eq!(cfg.bootstrap_sysadmin().as_deref(), Some("admin@example.com"));

        clear_env();
    }

    #[test]
    fn port_falls_back_to_generic_port_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(FALLBACK_PORT_VAR, "5555");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.port, 5555);
        assert_eq!(cfg.public_url, "http://127.0.0.1:5555");
        assert_eq!(cfg.agent_ws_url, "ws://127.0.0.1:5555/ws/agent");

        clear_env();
    }

    #[test]
    fn store_spec_falls_back_to_db_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::DB_PATH, "legacy.db");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.store_spec, "legacy.db");

        clear_env();
    }

    #[test]
    fn malformed_host_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::HOST, "not-an-ip");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::HOST));

        clear_env();
    }

    #[test]
    fn malformed_port_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::PORT, "not-a-number");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::PORT));

        clear_env();
    }

    #[test]
    fn malformed_drain_window_min_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::DRAIN_WINDOW_MIN, "not-a-number");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::DRAIN_WINDOW_MIN));

        clear_env();
    }

    #[test]
    fn malformed_confirm_cost_threshold_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::CONFIRM_COST_THRESHOLD, "not-a-number");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::CONFIRM_COST_THRESHOLD));

        clear_env();
    }

    #[test]
    fn malformed_reconcile_interval_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::RECONCILE_INTERVAL_S, "not-a-number");

        let err = Config::from_env().unwrap_err();
        assert!(err.to_string().contains(env::RECONCILE_INTERVAL_S));

        clear_env();
    }

    #[test]
    fn min_protocol_falls_back_to_default_on_parse_failure() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::MIN_PROTOCOL, "not-a-number");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.min_protocol, chuk_compute_wire::PROTOCOL_VERSION);

        clear_env();
    }

    #[test]
    fn google_client_credentials_empty_string_filtered_to_none() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::GOOGLE_CLIENT_ID, "");
        std::env::set_var(env::GOOGLE_CLIENT_SECRET, "");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.google_client_id, None);
        assert_eq!(cfg.google_client_secret, None);
        assert!(!cfg.auth_enabled());

        clear_env();
    }

    #[test]
    fn sysadmin_email_blank_after_trim_filtered_to_none() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        set_required_tokens();
        std::env::set_var(env::SYSADMIN_EMAIL, "   ");

        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.sysadmin_email, None);

        clear_env();
    }

    #[test]
    fn auth_enabled_requires_client_id_and_secret_and_nonempty_allowed_emails() {
        assert!(!base_config().auth_enabled());

        let id_only = Config {
            google_client_id: Some("id".into()),
            ..base_config()
        };
        assert!(!id_only.auth_enabled());

        let id_and_secret_no_emails = Config {
            google_client_id: Some("id".into()),
            google_client_secret: Some("secret".into()),
            ..base_config()
        };
        assert!(!id_and_secret_no_emails.auth_enabled());

        let fully_configured = Config {
            google_client_id: Some("id".into()),
            google_client_secret: Some("secret".into()),
            allowed_emails: vec!["a@example.com".into()],
            ..base_config()
        };
        assert!(fully_configured.auth_enabled());
    }

    #[test]
    fn bootstrap_sysadmin_prefers_explicit_then_first_allowed_email_then_none() {
        let explicit = Config {
            sysadmin_email: Some("admin@example.com".into()),
            allowed_emails: vec!["other@example.com".into()],
            ..base_config()
        };
        assert_eq!(
            explicit.bootstrap_sysadmin().as_deref(),
            Some("admin@example.com")
        );

        let fallback_to_first_allowed = Config {
            sysadmin_email: None,
            allowed_emails: vec!["first@example.com".into(), "second@example.com".into()],
            ..base_config()
        };
        assert_eq!(
            fallback_to_first_allowed.bootstrap_sysadmin().as_deref(),
            Some("first@example.com")
        );

        let none_available = Config {
            sysadmin_email: None,
            allowed_emails: Vec::new(),
            ..base_config()
        };
        assert_eq!(none_available.bootstrap_sysadmin(), None);
    }
}
