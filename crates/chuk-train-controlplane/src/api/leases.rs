//! Provisioning, leases, spend, and the Colab bootstrap cell (M2).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{ApiError, Role};

use super::{bad_request, internal, require_role};
use crate::apikey::AuthContext;
use crate::AppState;

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
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<chuk_train_proto::ProvisionRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
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
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::ExtendLeaseRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
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
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::TeardownRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let worker_id = chuk_train_proto::WorkerId(worker_id);
    // force skips the drain grace and destroys immediately.
    match state.leases.teardown(&worker_id, !request.force).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

#[derive(serde::Deserialize)]
pub struct ColabCellParams {
    lease_min: Option<f64>,
    labels: Option<String>,
}

const DEFAULT_COLAB_LABELS: &str = "colab,t4";

/// Generate a ready-to-paste Colab bootstrap cell (spec §6). The control plane
/// fills in its own public URL + join token, so there is nothing to hand-edit.
pub async fn colab_cell(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ColabCellParams>,
) -> Response {
    let labels = params
        .labels
        .unwrap_or_else(|| DEFAULT_COLAB_LABELS.to_owned());
    // Optional lease flags: passing lease_min makes the worker self-drain at
    // T-drain (belt) matching the control plane's window.
    let lease_flags = match params.lease_min {
        Some(m) => format!(
            " --lease-min {m} --drain-window-min {}",
            state.config.drain_window_min
        ),
        None => String::new(),
    };
    // Bootstrap through install.sh (uname → target triple → download + verify →
    // exec), so the cell never hardcodes a per-target agent path.
    let cell = format!(
        r#"# chuk-train · Colab worker — paste into ONE cell (Runtime → T4 GPU), then run.
CP_URL = "{url}"
JOIN_TOKEN = "{token}"
LABELS = "{labels}"

import subprocess
cmd = ("curl -fsSL " + CP_URL + "/install.sh | sh -s -- "
       "--cp " + CP_URL + " --token " + JOIN_TOKEN + " --labels " + LABELS + "{lease_flags}")
print("[chuk-train] bootstrapping worker via install.sh …")
subprocess.run(cmd, shell=True, check=False)
"#,
        url = state.config.public_url,
        token = state.config.join_token,
    );
    Json(chuk_train_proto::ColabCell { cell }).into_response()
}

#[derive(serde::Deserialize)]
pub struct SpendParams {
    period: Option<String>,
}

/// Spend per provider over a period (spec §8): committed = projected cost of
/// live leases, spent = realised lease_end cost from the ledger, with
/// cap/headroom attached where a matching-period budget exists.
pub async fn spend_status(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SpendParams>,
) -> Response {
    let period = params
        .period
        .unwrap_or_else(|| chuk_train_proto::DEFAULT_BUDGET_PERIOD.to_owned());
    if let Err(reason) = crate::budget::validate_period(&period) {
        return bad_request(&reason);
    }
    let (budgets, ledger, live) = match tokio::try_join!(
        state.hub.store.budgets(),
        state.hub.store.ledger_entries(),
        state.hub.store.live_leases(),
    ) {
        Ok(all) => all,
        Err(error) => return internal(error),
    };
    Json(crate::budget::report(&budgets, &ledger, &live, &period, unix_now())).into_response()
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub async fn list_budgets(State(state): State<Arc<AppState>>) -> Response {
    match state.hub.store.budgets().await {
        Ok(budgets) => Json::<Vec<chuk_train_proto::Budget>>(budgets).into_response(),
        Err(error) => internal(error),
    }
}

/// Upsert a budget cap (spec §6 `set_budget`). Admin-scoped: a cap is a
/// governance decision, not a per-run knob.
pub async fn set_budget(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<chuk_train_proto::SetBudgetRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let period = request
        .period
        .unwrap_or_else(|| chuk_train_proto::DEFAULT_BUDGET_PERIOD.to_owned());
    if let Err(reason) = crate::budget::validate_scope(&request.scope) {
        return bad_request(&reason);
    }
    if let Err(reason) = crate::budget::validate_period(&period) {
        return bad_request(&reason);
    }
    if !request.cap.is_finite() || request.cap < 0.0 {
        return bad_request("cap must be a non-negative number");
    }
    let budget = chuk_train_proto::Budget {
        scope: request.scope,
        cap: request.cap,
        period,
        updated_at: unix_now(),
    };
    match state.hub.store.set_budget(&budget).await {
        Ok(()) => Json(budget).into_response(),
        Err(error) => internal(error),
    }
}

#[derive(serde::Deserialize)]
pub struct DeleteBudgetParams {
    scope: String,
}

/// Remove a budget cap by scope (query param — scopes contain `:`). Admin-scoped.
pub async fn delete_budget(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Query(params): Query<DeleteBudgetParams>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    match state.hub.store.delete_budget(&params.scope).await {
        Ok(true) => Json(serde_json::json!({ "deleted": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("no budget set for scope {:?}", params.scope),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}
