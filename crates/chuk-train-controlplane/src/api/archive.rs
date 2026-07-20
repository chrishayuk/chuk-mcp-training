//! Archive endpoints (spec §11.5).
//!
//! Note: this is `crate::api::archive`, distinct from `crate::archive` (the
//! archiver itself), which is referred to by its full path below.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{Role, RunId};
use serde::Deserialize;

use super::{bad_request, internal, require_role};
use crate::apikey::AuthContext;
use crate::AppState;

#[derive(Deserialize, Default)]
pub struct ArchiveParams {
    #[serde(default)]
    force: bool,
}

fn archive_outcome_json(outcome: crate::archive::Outcome) -> serde_json::Value {
    use crate::archive::Outcome;
    match outcome {
        Outcome::Archived { step, files } => {
            serde_json::json!({ "status": "archived", "step": step, "files": files })
        }
        Outcome::AlreadyArchived => serde_json::json!({ "status": "already_archived" }),
        Outcome::NoCheckpoint => serde_json::json!({ "status": "no_checkpoint" }),
    }
}

/// Archive one run's final checkpoint + logs/metrics to Drive now (spec §11.5).
pub async fn archive_run(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Query(params): Query<ArchiveParams>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let Some(archiver) = state.archiver.clone() else {
        return bad_request("archive tier not configured (no Drive credentials)");
    };
    match archiver.archive_run(&RunId(run_id), params.force).await {
        Ok(outcome) => Json(archive_outcome_json(outcome)).into_response(),
        Err(error) => internal(error),
    }
}

/// Sweep: archive every completed run not yet on Drive (backstop, on demand).
pub async fn archive_all(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let Some(archiver) = state.archiver.clone() else {
        return bad_request("archive tier not configured (no Drive credentials)");
    };
    match archiver.sweep_once().await {
        Ok(archived) => Json(serde_json::json!({ "archived": archived })).into_response(),
        Err(error) => internal(error),
    }
}

/// Per-run archive state: each recent run's final checkpoint location + when.
pub async fn archive_status(State(state): State<Arc<AppState>>) -> Response {
    let runs = match state
        .hub
        .store
        .runs(&Default::default(), chuk_train_proto::DEFAULT_RUN_LIST_LIMIT)
        .await
    {
        Ok(runs) => runs,
        Err(error) => return internal(error),
    };
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        let ckpt = state.hub.store.latest_checkpoint(&run.id).await.ok().flatten();
        out.push(serde_json::json!({
            "run_id": run.id.0,
            "name": run.name,
            "state": run.state.as_str(),
            "final_step": ckpt.as_ref().map(|c| c.step),
            "location": ckpt.as_ref().map(|c| c.location),
            "archived_at": ckpt.as_ref().and_then(|c| c.archived_at),
        }));
    }
    Json(out).into_response()
}
