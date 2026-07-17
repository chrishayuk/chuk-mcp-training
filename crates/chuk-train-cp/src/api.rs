//! Bearer-authenticated REST API — the surface the MCP server (and the
//! dashboard's JavaScript) consume.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiError, BuildCodeUnitRequest, CheckpointInfo, LogsResponse, MetricSeries,
    PinCheckpointRequest, RunEvent, RunId, RunRecord, RunSpec, RunSummary, ShellSpec, SignedUrl,
    SubmitRunRequest, SubmitRunResponse, SubmitShellRequest, WorkerInfo, DEFAULT_ARTIFACT_URL_TTL,
    DEFAULT_LOG_TAIL_LINES, DEFAULT_METRIC_DOWNSAMPLE, DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT,
};
use serde::Deserialize;

use crate::{codeunit, AppState};

const BEARER_PREFIX: &str = "Bearer ";
const ERR_UNAUTHORIZED: &str = "bad or missing bearer token";
const ERR_RUN_NOT_FOUND: &str = "no such run";
const ERR_CKPT_NOT_FOUND: &str = "no such checkpoint";
const ERR_INTERNAL: &str = "internal error";
/// Query-param key for the metric-key filter list (comma-separated).
const METRIC_KEYS_SEP: char = ',';

/// 500-with-envelope for store failures; the MCP layer relays the envelope.
fn internal(error: anyhow::Error) -> Response {
    tracing::error!(%error, "api internal error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError {
            error: ERR_INTERNAL.into(),
        }),
    )
        .into_response()
}

fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ApiError {
            error: ERR_RUN_NOT_FOUND.into(),
        }),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Auth middleware for /api/*
// ---------------------------------------------------------------------------

pub async fn require_bearer(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix(BEARER_PREFIX))
        .is_some_and(|token| token == state.config.api_token);
    if !authorized {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ApiError {
                error: ERR_UNAUTHORIZED.into(),
            }),
        )
            .into_response();
    }
    next.run(request).await
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn fleet(State(state): State<Arc<AppState>>) -> Response {
    match state.hub.store.fleet().await {
        Ok(workers) => Json::<Vec<WorkerInfo>>(workers).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn submit_shell(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SubmitShellRequest>,
) -> Response {
    let spec = RunSpec::Shell(ShellSpec {
        command: request.command,
        timeout_s: request.timeout_s.unwrap_or(DEFAULT_SHELL_TIMEOUT.as_secs()),
    });
    match state.hub.submit(&request.name, &spec).await {
        Ok(run_id) => Json(SubmitRunResponse { run_id }).into_response(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct ListParams {
    limit: Option<u32>,
}

pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> Response {
    match state
        .hub
        .store
        .runs(params.limit.unwrap_or(DEFAULT_RUN_LIST_LIMIT))
        .await
    {
        Ok(runs) => Json::<Vec<RunSummary>>(runs).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn run_status(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.run(&RunId(run_id)).await {
        Ok(Some(run)) => Json::<RunRecord>(run).into_response(),
        Ok(None) => not_found(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct TailParams {
    lines: Option<u32>,
}

pub async fn tail_logs(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Query(params): Query<TailParams>,
) -> Response {
    let run_id = RunId(run_id);
    match state
        .hub
        .store
        .tail_logs(&run_id, params.lines.unwrap_or(DEFAULT_LOG_TAIL_LINES))
        .await
    {
        Ok(lines) => Json(LogsResponse { run_id, lines }).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn run_events(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.events(&RunId(run_id)).await {
        Ok(events) => Json::<Vec<RunEvent>>(events).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

// ---------------------------------------------------------------------------
// M2 handlers: leases, provisioning, spend
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub struct OffersParams {
    provider: String,
    gpu: Option<String>,
    max_price_hr: Option<f64>,
}

pub async fn provider_offers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OffersParams>,
) -> Response {
    match state
        .leases
        .offers(&params.provider, params.gpu.as_deref(), params.max_price_hr)
        .await
    {
        Ok(offers) => Json::<Vec<chuk_train_proto::Offer>>(offers).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn provision(
    State(state): State<Arc<AppState>>,
    Json(request): Json<chuk_train_proto::ProvisionRequest>,
) -> Response {
    match state.leases.provision(&request).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn lease_status(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Response {
    match state
        .hub
        .store
        .lease(&chuk_train_proto::WorkerId(worker_id))
        .await
    {
        Ok(Some(lease)) => Json(lease).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no lease".into(),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}

pub async fn extend_lease(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::ExtendLeaseRequest>,
) -> Response {
    let worker_id = chuk_train_proto::WorkerId(worker_id);
    match state
        .leases
        .extend(&worker_id, request.minutes, &request.reason)
        .await
    {
        Ok(Some(lease)) => Json(lease).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no lease".into(),
            }),
        )
            .into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn teardown(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::TeardownRequest>,
) -> Response {
    let worker_id = chuk_train_proto::WorkerId(worker_id);
    // force skips the drain grace and destroys immediately.
    match state.leases.teardown(&worker_id, !request.force).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn spend_status(State(state): State<Arc<AppState>>) -> Response {
    use std::collections::{BTreeMap, BTreeSet};
    let (live, ledger) = match tokio::try_join!(
        state.hub.store.live_leases(),
        state.hub.store.ledger_entries(),
    ) {
        Ok(pair) => pair,
        Err(error) => return internal(error),
    };
    // committed = projected cost of leases still running; spent = realised
    // lease_end costs from the ledger (spec §8: ledger is the source of truth).
    let mut committed: BTreeMap<String, f64> = BTreeMap::new();
    for lease in &live {
        *committed.entry(lease.provider.clone()).or_default() += lease.projected_cost();
    }
    let mut spent: BTreeMap<String, f64> = BTreeMap::new();
    for entry in &ledger {
        if entry.event == "lease_end" {
            *spent.entry(entry.provider.clone()).or_default() += entry.cost;
        }
    }
    let mut providers: BTreeSet<String> = BTreeSet::new();
    providers.extend(committed.keys().cloned());
    providers.extend(spent.keys().cloned());
    let lines: Vec<chuk_train_proto::SpendLine> = providers
        .into_iter()
        .map(|provider| chuk_train_proto::SpendLine {
            committed: *committed.get(&provider).unwrap_or(&0.0),
            spent: *spent.get(&provider).unwrap_or(&0.0),
            provider,
        })
        .collect();
    let report = chuk_train_proto::SpendReport {
        total_committed: lines.iter().map(|l| l.committed).sum(),
        total_spent: lines.iter().map(|l| l.spent).sum(),
        lines,
    };
    Json(report).into_response()
}

// ---------------------------------------------------------------------------
// M1 handlers: code units, train submission, metrics, checkpoints, artifacts
// ---------------------------------------------------------------------------

pub async fn build_code_unit(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BuildCodeUnitRequest>,
) -> Response {
    let built = codeunit::build(
        state.artifacts.as_ref(),
        &request.repo,
        request.commit.as_deref(),
        request.name.as_deref(),
    )
    .await;
    match built {
        Ok(info) => {
            if let Err(error) = state
                .hub
                .store
                .register_code_unit(&info.code, &info.manifest, &info.uri)
                .await
            {
                return internal(error);
            }
            Json(info).into_response()
        }
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn submit_run(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SubmitRunRequest>,
) -> Response {
    match state.hub.submit(&request.name, &request.spec).await {
        Ok(run_id) => Json(SubmitRunResponse { run_id }).into_response(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct MetricParams {
    keys: Option<String>,
    since_step: Option<u64>,
    downsample: Option<u32>,
}

pub async fn run_metrics(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Query(params): Query<MetricParams>,
) -> Response {
    let run_id = RunId(run_id);
    let keys: Option<Vec<String>> = params.keys.as_deref().map(|raw| {
        raw.split(METRIC_KEYS_SEP)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    });
    let series = state
        .hub
        .store
        .metric_series(
            &run_id,
            keys.as_deref(),
            params.since_step.unwrap_or(0),
            params.downsample.unwrap_or(DEFAULT_METRIC_DOWNSAMPLE),
        )
        .await;
    match series {
        Ok(series) => Json::<MetricSeries>(series).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn list_checkpoints(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.checkpoints(&RunId(run_id)).await {
        Ok(ckpts) => Json::<Vec<CheckpointInfo>>(ckpts).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn pin_checkpoint(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Json(request): Json<PinCheckpointRequest>,
) -> Response {
    match state
        .hub
        .store
        .pin_checkpoint(&RunId(run_id), request.step, &request.name)
        .await
    {
        Ok(true) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: ERR_CKPT_NOT_FOUND.into(),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct ArtifactUrlParams {
    key: String,
    ttl_s: Option<u64>,
}

/// Time-limited fetch URL for an artifact key (spec §6 artifact_url). The
/// filesystem backend has no native signed URL, so this points at the control
/// plane's own authenticated `/api/blob` endpoint.
pub async fn artifact_url(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ArtifactUrlParams>,
) -> Response {
    let ttl =
        std::time::Duration::from_secs(params.ttl_s.unwrap_or(DEFAULT_ARTIFACT_URL_TTL.as_secs()));
    let native = match state.artifacts.presign_get(&params.key, ttl) {
        Ok(url) => url,
        Err(error) => return bad_request(&error.to_string()),
    };
    let signed = native.unwrap_or_else(|| SignedUrl {
        url: format!(
            "{}/api/blob/{}",
            state.config.public_url.trim_end_matches('/'),
            params.key
        ),
        expires_at: now() + ttl.as_secs_f64(),
    });
    Json(signed).into_response()
}

/// Serve artifact bytes (bearer-authed). Used by artifact_url consumers such as
/// lazarus pulling checkpoints to the Mac.
pub async fn blob(State(state): State<Arc<AppState>>, Path(key): Path<String>) -> Response {
    match state.artifacts.get(&key).await {
        Ok(bytes) => bytes.into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no such artifact".into(),
            }),
        )
            .into_response(),
    }
}

fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
        .into_response()
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}
