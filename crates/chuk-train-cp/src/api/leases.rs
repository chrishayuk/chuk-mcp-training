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
    // Optional lease line: passing lease_min makes the worker self-drain at
    // T-drain (belt) matching the control plane's window.
    let lease_args = match params.lease_min {
        Some(m) => format!(
            "\nargs += [\"--lease-min\", \"{m}\", \"--drain-window-min\", \"{}\"]",
            state.config.drain_window_min
        ),
        None => String::new(),
    };
    let cell = format!(
        r#"# chuk-train · Colab worker — paste into ONE cell (Runtime → T4 GPU), then run.
CP_URL = "{url}"
JOIN_TOKEN = "{token}"
LABELS = "{labels}"

import os, stat, subprocess, urllib.request
base = CP_URL.rstrip("/"); agent = "/tmp/chuk-train-agent"
urllib.request.urlretrieve(base + "/agent/linux-x86_64", agent)
os.chmod(agent, os.stat(agent).st_mode | stat.S_IEXEC)
ws = base.replace("https://", "wss://").replace("http://", "ws://") + "/ws/agent"
args = [agent, "--url", ws, "--token", JOIN_TOKEN, "--labels", LABELS]{lease_args}
print("[chuk-train] joining", ws)
subprocess.run(args, check=False)
"#,
        url = state.config.public_url,
        token = state.config.join_token,
    );
    Json(chuk_train_proto::ColabCell { cell }).into_response()
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
