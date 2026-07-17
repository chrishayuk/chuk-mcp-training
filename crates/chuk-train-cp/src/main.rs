//! chuk-train-cp — the chuk-mcp-training control plane daemon (M0).
//!
//! Surfaces:
//!   * `/ws/agent`  — outbound-dial websocket for chuk-train-agent workers
//!   * `/api/*`     — bearer-authenticated REST for the MCP server + dashboard
//!   * `/`          — dashboard stub
//!   * `/healthz`   — unauthenticated liveness

mod api;
mod artifacts;
mod codeunit;
mod config;
mod dash;
mod grant;
mod hub;
mod store;
mod upload;
mod ws;

use std::sync::Arc;

use anyhow::Result;
use axum::routing::{get, post, put};
use axum::Router;
use chuk_train_proto::{AGENT_WS_PATH, API_PREFIX, HEALTH_PATH};
use tracing::info;

use crate::artifacts::{open_artifact_store, ArtifactStore};
use crate::config::Config;
use crate::hub::Hub;
use crate::store::open_store;

pub struct AppState {
    pub config: Config,
    pub hub: Arc<Hub>,
    pub artifacts: Arc<dyn ArtifactStore>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env()?;
    let store: Arc<dyn store::Store> = Arc::from(open_store(&config.store_spec).await?);
    let artifacts: Arc<dyn ArtifactStore> = Arc::from(open_artifact_store(&config.artifacts_spec)?);
    let hub = Hub::new(store, artifacts.clone());
    let bind = (config.host, config.port);
    let state = Arc::new(AppState {
        config,
        hub,
        artifacts,
    });

    // Bearer-authenticated surface: the MCP server and dashboard.
    let api_bearer = Router::new()
        .route("/fleet", get(api::fleet))
        .route("/runs/shell", post(api::submit_shell))
        .route("/runs", get(api::list_runs).post(api::submit_run))
        .route("/runs/{run_id}", get(api::run_status))
        .route("/runs/{run_id}/logs", get(api::tail_logs))
        .route("/runs/{run_id}/events", get(api::run_events))
        .route("/runs/{run_id}/metrics", get(api::run_metrics))
        .route("/runs/{run_id}/checkpoints", get(api::list_checkpoints))
        .route("/runs/{run_id}/checkpoints/pin", post(api::pin_checkpoint))
        .route("/code_units", post(api::build_code_unit))
        .route("/artifact_url", get(api::artifact_url))
        .route("/blob/{*key}", get(api::blob))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            api::require_bearer,
        ));

    // Grant-authenticated surface: workers uploading/fetching their own blobs.
    let api_grant = Router::new()
        .route("/upload/{*key}", put(upload::upload))
        .route("/fetch/{*key}", get(upload::fetch));

    let app = Router::new()
        .route("/", get(dash::page))
        .route(HEALTH_PATH, get(api::healthz))
        .route(AGENT_WS_PATH, get(ws::agent_ws))
        .nest(API_PREFIX, api_bearer.merge(api_grant))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(addr = %listener.local_addr()?, "chuk-train-cp listening");
    axum::serve(listener, app).await?;
    Ok(())
}
