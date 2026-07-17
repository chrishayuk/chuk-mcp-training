//! Bearer-authenticated REST API — the surface the MCP server (and the
//! dashboard's JavaScript) consume.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{header, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiError, LogsResponse, RunEvent, RunId, RunRecord, RunSpec, RunSummary, SubmitRunResponse,
    SubmitShellRequest, WorkerInfo, DEFAULT_LOG_TAIL_LINES, DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT,
};
use serde::Deserialize;

use crate::AppState;

const BEARER_PREFIX: &str = "Bearer ";
const ERR_UNAUTHORIZED: &str = "bad or missing bearer token";
const ERR_RUN_NOT_FOUND: &str = "no such run";
const ERR_INTERNAL: &str = "internal error";

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
    let spec = RunSpec::Shell {
        command: request.command,
        timeout_s: request.timeout_s.unwrap_or(DEFAULT_SHELL_TIMEOUT.as_secs()),
    };
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
