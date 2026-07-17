//! Control-plane configuration, sourced entirely from environment variables.

use std::net::{IpAddr, Ipv4Addr};

use anyhow::{Context, Result};
use chuk_train_proto::{env, DEFAULT_PORT};

const DEFAULT_STORE_SPEC: &str = "sqlite:chuk_train.db";
const DEFAULT_HOST: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
/// PaaS convention honoured when CHUK_TRAIN_PORT is unset.
const FALLBACK_PORT_VAR: &str = "PORT";

#[derive(Debug, Clone)]
pub struct Config {
    pub api_token: String,
    pub join_token: String,
    /// Store backend spec: `sqlite:path.db`, bare path (SQLite), `redis:` reserved.
    pub store_spec: String,
    pub host: IpAddr,
    pub port: u16,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_token = required_token(env::API_TOKEN)?;
        let join_token = required_token(env::JOIN_TOKEN)?;
        let store_spec = std::env::var(env::STORE_URL)
            .or_else(|_| std::env::var(env::DB_PATH))
            .unwrap_or_else(|_| DEFAULT_STORE_SPEC.to_owned());
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
        Ok(Self {
            api_token,
            join_token,
            store_spec,
            host,
            port,
        })
    }
}

/// Tokens are required: a control plane that silently generates its own
/// credentials invites a deployment where nobody knows them. Fail loudly.
fn required_token(var: &str) -> Result<String> {
    let value = std::env::var(var).with_context(|| format!("{var} must be set"))?;
    anyhow::ensure!(!value.trim().is_empty(), "{var} must not be empty");
    Ok(value)
}
