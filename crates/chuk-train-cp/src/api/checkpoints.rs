//! Checkpoints and artifact retrieval.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiError, CheckpointInfo, PinCheckpointRequest, Role, RunId, SignedUrl, DEFAULT_ARTIFACT_URL_TTL,
};
use serde::Deserialize;

use super::{bad_request, internal, not_found, now, require_role, ERR_CKPT_NOT_FOUND};
use crate::apikey::AuthContext;
use crate::AppState;

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
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Json(request): Json<PinCheckpointRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
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

/// Stable, location-resolving fetch for one checkpoint file (spec §11.5). The
/// durable handle handed to lazarus / experiment servers: it redirects to a
/// presigned R2 URL while the object is on R2 (hot or promoted-final), or
/// streams it from Drive once archived — the bytes move underneath the URL.
pub async fn serve_checkpoint(
    State(state): State<Arc<AppState>>,
    Path((run_id, step, file)): Path<(String, u64, String)>,
) -> Response {
    use chuk_train_proto::{keys, CheckpointLocation};
    if file.contains('/') || !keys::is_safe_key(&file) {
        return bad_request("invalid checkpoint file");
    }
    let rid = RunId(run_id.clone());
    let ckpts = match state.hub.store.checkpoints(&rid).await {
        Ok(ckpts) => ckpts,
        Err(error) => return internal(error),
    };
    let Some(ckpt) = ckpts.into_iter().find(|c| c.step == step) else {
        return not_found();
    };
    match ckpt.location {
        CheckpointLocation::Drive => {
            let Some(drive) = state.drive.clone() else {
                return internal(anyhow::anyhow!("checkpoint on drive but drive not configured"));
            };
            let ids = match state.hub.store.checkpoint_drive_ids(&rid, step).await {
                Ok(ids) => ids.unwrap_or_default(),
                Err(error) => return internal(error),
            };
            let Some(file_id) = ids.get(&file) else {
                return not_found();
            };
            match drive.download(file_id).await {
                Ok(bytes) => bytes.into_response(),
                Err(error) => internal(error),
            }
        }
        location => {
            let key = if location == CheckpointLocation::R2Final {
                keys::checkpoint_final_file(&run_id, step, &file)
            } else {
                keys::checkpoint_file(&run_id, step, &file)
            };
            match state.artifacts.presign_get(&key, DEFAULT_ARTIFACT_URL_TTL).await {
                Ok(Some(signed)) => axum::response::Redirect::temporary(&signed.url).into_response(),
                Ok(None) => match state.artifacts.get(&key).await {
                    Ok(bytes) => bytes.into_response(),
                    Err(_) => not_found(),
                },
                Err(error) => internal(error),
            }
        }
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
    let native = match state.artifacts.presign_get(&params.key, ttl).await {
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
