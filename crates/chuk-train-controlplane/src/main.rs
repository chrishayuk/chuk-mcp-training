//! chuk-train-controlplane — the chuk-mcp-training control plane daemon (M0).
//!
//! Surfaces:
//!   * `/ws/agent`  — outbound-dial websocket for chuk-compute-worker workers
//!   * `/api/*`     — bearer-authenticated REST for the MCP server + dashboard
//!   * `/`          — dashboard stub
//!   * `/healthz`   — unauthenticated liveness

mod api;
mod apikey;
mod archive;
mod artifacts;
mod auth;
mod codeunit;
mod config;
mod crypto;
mod dash;
mod drive;
mod experiments;
mod grant;
mod hub;
mod jobspec;
mod lease;
mod provider;
mod store;
mod upload;
mod ws;

use std::sync::Arc;

use anyhow::Result;
use axum::routing::{delete, get, post, put};
use axum::Router;
use chuk_compute_wire::API_PREFIX;
use chuk_train_proto::{
    AGENT_DOWNLOAD_ROUTE, AGENT_VERSION_PATH, AGENT_WS_PATH, HEALTH_PATH, INSTALL_SCRIPT_PATH,
};
use tracing::{info, warn};

use crate::archive::Archiver;
use crate::artifacts::{open_artifact_store, ArtifactStore};
use crate::config::Config;
use crate::drive::DriveClient;
use crate::experiments::Experiments;
use crate::hub::Hub;
use crate::lease::LeaseManager;
use crate::provider::build_providers;
use crate::store::open_store;

pub struct AppState {
    pub config: Config,
    pub hub: Arc<Hub>,
    pub artifacts: Arc<dyn ArtifactStore>,
    pub leases: Arc<LeaseManager>,
    /// Drive cold-archive client; `None` when the archive tier is off.
    pub drive: Option<Arc<DriveClient>>,
    /// Archive/retention worker; `None` when the archive tier is off.
    pub archiver: Option<Arc<Archiver>>,
    /// Decrypts/encrypts a user's own linked chuk-experiments-server key
    /// (`api::access`'s `/me/experiments-key` routes). `None` — that feature
    /// is off — unless `CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY` is set and valid.
    pub key_encryption_key: Option<[u8; 32]>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env for local runs; a no-op in deployment (Fly injects secrets as
    // real env vars, and there is no .env in the image).
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env()?;
    let store: Arc<dyn store::Store> = Arc::from(open_store(&config.store_spec).await?);
    // Seed the default team + bootstrap sysadmin (idempotent — only seeds a user
    // that doesn't already exist, so later role changes persist across restarts).
    store
        .ensure_team(
            chuk_train_proto::DEFAULT_TEAM_ID,
            chuk_train_proto::DEFAULT_TEAM_NAME,
        )
        .await?;
    if let Some(admin) = config.bootstrap_sysadmin() {
        if store.get_user(&admin).await?.is_none() {
            store
                .upsert_user(
                    &admin,
                    chuk_train_proto::DEFAULT_TEAM_ID,
                    chuk_train_proto::Role::Sysadmin,
                )
                .await?;
            info!(email = %admin, "seeded bootstrap sysadmin");
        }
    }
    let artifacts: Arc<dyn ArtifactStore> = Arc::from(open_artifact_store(&config.artifacts_spec)?);
    // R2 lifecycle: expire the hot / promoted-final checkpoint tiers on a timer
    // (spec §11.5) so the control plane never deletes them itself. Idempotent
    // (the whole config is re-set each boot); a no-op for the fs backend.
    {
        use chuk_train_proto::{
            CKPT_FINAL_PREFIX, CKPT_FINAL_TTL_DAYS, CKPT_HOT_PREFIX, CKPT_HOT_TTL_DAYS,
        };
        let rules = vec![
            (format!("{CKPT_HOT_PREFIX}/"), CKPT_HOT_TTL_DAYS),
            (format!("{CKPT_FINAL_PREFIX}/"), CKPT_FINAL_TTL_DAYS),
        ];
        match artifacts.apply_lifecycle(&rules).await {
            Ok(()) => info!(
                hot_days = CKPT_HOT_TTL_DAYS,
                final_days = CKPT_FINAL_TTL_DAYS,
                "artifact lifecycle rules set"
            ),
            Err(e) => warn!(error = %e, "applying artifact lifecycle rules (non-fatal)"),
        }
    }
    let drive = DriveClient::from_env()?.map(Arc::new);
    info!(archive_tier = drive.is_some(), "drive cold-archive tier");
    // chuk-experiments-server reporting mirror (spec §11.6): optional and gated —
    // off unless CHUK_EXPERIMENTS_URL + CHUK_EXPERIMENTS_API_KEY are set. When on,
    // a startup ensure creates-or-confirms the default experiment and validates
    // the credentials early; the mirror otherwise reports lazily off run events.
    let experiments = Experiments::from_env(store.clone(), &config.public_url);
    info!(
        experiments = experiments.is_some(),
        "experiments-server reporting mirror"
    );
    if let Some(exp) = experiments.clone() {
        let ensure_exp = exp.clone();
        tokio::spawn(async move {
            match ensure_exp.ensure().await {
                Ok(()) => info!("experiments-server: default experiment ready"),
                Err(e) => warn!(error = %e, "experiments-server ensure failed (mirror retries lazily)"),
            }
        });
        // Retries any mirror event (created/state/checkpoint/result) that
        // failed on first attempt, so a transient failure never silently
        // drops an observation.
        tokio::spawn(exp.run_outbox_loop(chuk_train_proto::DEFAULT_EXPERIMENTS_OUTBOX_INTERVAL));
    }
    let hub = Hub::new(store, artifacts.clone(), experiments);
    // Heartbeat reaper: sweeps the fleet for workers whose link is half-open
    // (a frozen tab that never delivered a socket close) and re-queues their
    // stranded resumable runs (spec §7). Always on — a core scheduling guarantee.
    tokio::spawn(
        hub.clone()
            .run_reaper_loop(chuk_train_proto::HEARTBEAT_REAP_INTERVAL),
    );
    // Archive/retention: when Drive is configured, a background loop tiers each
    // completed run (final checkpoint + logs/metrics) to Drive and records the
    // location; it is also the backstop for any run a prior pass missed.
    let archiver = drive.clone().map(|d| {
        let a = Archiver::new(hub.store.clone(), artifacts.clone(), d);
        tokio::spawn(a.clone().run_loop(chuk_train_proto::DEFAULT_ARCHIVE_INTERVAL));
        a
    });
    let providers = Arc::new(build_providers(
        &config.providers,
        config.agent_bin.clone(),
        config.vast_api_key.clone(),
    ));
    info!(providers = ?providers.names(), "provider registry");
    let leases = LeaseManager::new(hub.clone(), providers, config.clone());
    // The lease clock and reconcile loop are the M2 cleanup guarantees; they
    // run for the life of the process.
    tokio::spawn(leases.clone().run_clock());
    tokio::spawn(leases.clone().run_reconcile());

    let bind = (config.host, config.port);
    let state = Arc::new(AppState {
        config,
        hub,
        artifacts,
        leases,
        drive,
        archiver,
        key_encryption_key: crypto::key_from_env(),
    });

    // Bearer-authenticated surface: the MCP server and dashboard.
    let api_bearer = Router::new()
        .route("/fleet", get(api::fleet))
        .route("/runs/shell", post(api::submit_shell))
        .route("/runs", get(api::list_runs).post(api::submit_run))
        .route("/runs/from-experiment/{run_id}", post(api::submit_run_from_experiment))
        .route("/runs/{run_id}", get(api::run_status))
        .route("/runs/{run_id}/stop", post(api::stop_run))
        .route("/runs/{run_id}/resume", post(api::resume_run))
        .route("/runs/{run_id}/logs", get(api::tail_logs))
        .route("/runs/{run_id}/events", get(api::run_events))
        .route("/runs/{run_id}/metrics", get(api::run_metrics))
        .route("/runs/{run_id}/checkpoints", get(api::list_checkpoints))
        .route("/runs/{run_id}/checkpoints/pin", post(api::pin_checkpoint))
        .route("/code_units", post(api::build_code_unit))
        .route("/artifact_url", get(api::artifact_url))
        .route("/blob/{*key}", get(api::blob))
        .route("/provider_offers", get(api::provider_offers))
        .route("/provision", post(api::provision))
        .route("/colab_cell", get(api::colab_cell))
        .route("/workers/{worker_id}/telemetry", get(api::worker_telemetry))
        .route("/workers/{worker_id}/lease", get(api::lease_status))
        .route("/workers/{worker_id}/extend", post(api::extend_lease))
        .route("/workers/{worker_id}/teardown", post(api::teardown))
        .route("/spend", get(api::spend_status))
        .route("/runs/{run_id}/archive", post(api::archive_run))
        .route("/archive", post(api::archive_all).get(api::archive_status))
        .route(
            "/checkpoint/{run_id}/{step}/{file}",
            get(api::serve_checkpoint),
        )
        .route("/me", get(api::whoami))
        .route(
            "/me/experiments-key",
            put(api::set_experiments_key).delete(api::clear_experiments_key),
        )
        .route("/users", get(api::list_users).post(api::upsert_user))
        .route("/users/{email}", delete(api::remove_user))
        .route("/apikeys", get(api::list_api_keys).post(api::create_api_key))
        .route("/apikeys/{id}", delete(api::revoke_api_key))
        .route(
            "/worker_tokens",
            get(api::list_worker_tokens).post(api::create_worker_token),
        )
        .route("/worker_tokens/{id}", delete(api::revoke_worker_token))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            api::require_bearer,
        ));

    // Grant-authenticated surface: workers uploading/fetching their own blobs.
    // Checkpoints are ~440 MB at 115M scale (spec §11.5), so the default 2 MB
    // body limit is disabled here. (Streaming to disk / direct-to-R2 is the
    // proper scale fix; buffered upload with adequate machine RAM suffices for
    // the proving runs.)
    let api_grant = Router::new()
        .route("/upload/{*key}", put(upload::upload))
        .route("/fetch/{*key}", get(upload::fetch))
        .route("/blob_url", post(upload::blob_url))
        .layer(axum::extract::DefaultBodyLimit::disable());

    let app = Router::new()
        .route("/", get(dash::dashboard))
        .route("/auth/login", get(auth::login))
        .route("/auth/callback", get(auth::callback))
        .route("/auth/logout", get(auth::logout))
        .route(HEALTH_PATH, get(api::healthz))
        .route(INSTALL_SCRIPT_PATH, get(api::serve_install))
        .route(AGENT_VERSION_PATH, get(api::agent_version))
        .route(AGENT_DOWNLOAD_ROUTE, get(api::serve_agent))
        .route(AGENT_WS_PATH, get(ws::agent_ws))
        .nest(API_PREFIX, api_bearer.merge(api_grant))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(addr = %listener.local_addr()?, "chuk-train-controlplane listening");
    axum::serve(listener, app).await?;
    Ok(())
}
