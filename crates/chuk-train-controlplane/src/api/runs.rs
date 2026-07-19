//! Run lifecycle, logs, events, metrics, and code units.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    BuildCodeUnitRequest, LogsResponse, MetricSeries, Role, RunEvent, RunId, RunRecord, RunSpec,
    RunSummary, ShellSpec, SubmitRunRequest, SubmitRunResponse, SubmitShellRequest, WorkerInfo,
    DEFAULT_LOG_TAIL_LINES, DEFAULT_METRIC_DOWNSAMPLE, DEFAULT_RUN_LIST_LIMIT, DEFAULT_SHELL_TIMEOUT,
};
use serde::Deserialize;

use super::{bad_request, internal, not_found, require_role};
use crate::apikey::AuthContext;
use crate::{codeunit, AppState};

/// Query-param key for the metric-key filter list (comma-separated).
const METRIC_KEYS_SEP: char = ',';

pub async fn fleet(State(state): State<Arc<AppState>>) -> Response {
    match state.hub.store.fleet().await {
        Ok(workers) => Json::<Vec<WorkerInfo>>(workers).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn submit_shell(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SubmitShellRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let spec = RunSpec::Shell(ShellSpec {
        command: request.command,
        timeout_s: request.timeout_s.unwrap_or(DEFAULT_SHELL_TIMEOUT.as_secs()),
    });
    // A shell probe is always an unattached scratch run — never mirrored to an
    // experiments-server logical run.
    match state.hub.submit(&request.name, &spec, None).await {
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

pub async fn build_code_unit(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<BuildCodeUnitRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
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
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SubmitRunRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    match state
        .hub
        .submit(&request.name, &request.spec, request.experiment_ref.as_deref())
        .await
    {
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
