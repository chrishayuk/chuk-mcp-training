//! Worker-facing blob transfer, authorised by run-scoped grants (spec §12),
//! not the API token. Agents upload checkpoints here and fetch their code unit
//! and resume checkpoints — each request checked against the grant's run.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiError, BlobMethod, BlobUrlRequest, BlobUrlResponse, API_PREFIX, DEFAULT_ARTIFACT_URL_TTL,
};

use crate::grant::Grant;
use crate::AppState;

const BEARER_PREFIX: &str = "Bearer ";

/// `PUT /api/upload/<key>` — write a blob into the grant's run tree.
pub async fn upload(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(key): Path<String>,
    body: Bytes,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    if !grant.may_write(&key) {
        return forbidden();
    }
    match state.artifacts.put(&key, body.to_vec()).await {
        Ok(_) => (StatusCode::CREATED, Json(serde_json::json!({ "ok": true }))).into_response(),
        Err(error) => {
            tracing::error!(%error, key, "upload failed");
            internal()
        }
    }
}

/// `GET /api/fetch/<key>` — read a blob the grant is allowed to see.
pub async fn fetch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(key): Path<String>,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    if !grant.may_read(&key) {
        return forbidden();
    }
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

/// `POST /api/blob_url` — where should the worker transfer this blob? With an
/// S3/R2 backend this is a presigned URL the worker hits directly; with the
/// filesystem backend it points back at this control plane's upload/fetch.
pub async fn blob_url(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<BlobUrlRequest>,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    let allowed = match req.method {
        BlobMethod::Put => grant.may_write(&req.key),
        BlobMethod::Get => grant.may_read(&req.key),
    };
    if !allowed {
        return forbidden();
    }

    let presigned = match req.method {
        BlobMethod::Put => {
            state
                .artifacts
                .presign_put(&req.key, DEFAULT_ARTIFACT_URL_TTL)
                .await
        }
        BlobMethod::Get => {
            state
                .artifacts
                .presign_get(&req.key, DEFAULT_ARTIFACT_URL_TTL)
                .await
        }
    };
    let response = match presigned {
        Ok(Some(signed)) => BlobUrlResponse {
            url: signed.url,
            requires_grant_header: false,
        },
        Ok(None) => {
            // Filesystem backend: fall back to transferring through the control
            // plane, which needs the grant token on the request.
            let action = match req.method {
                BlobMethod::Put => "upload",
                BlobMethod::Get => "fetch",
            };
            let base = state.config.public_url.trim_end_matches('/');
            BlobUrlResponse {
                url: format!("{base}{API_PREFIX}/{action}/{}", req.key),
                requires_grant_header: true,
            }
        }
        Err(error) => {
            tracing::error!(%error, key = req.key, "presign failed");
            return internal();
        }
    };
    Json(response).into_response()
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Option<Grant> {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix(BEARER_PREFIX))?;
    state.hub.grants().resolve(token)
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ApiError {
            error: "bad or missing grant token".into(),
        }),
    )
        .into_response()
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ApiError {
            error: "grant does not cover this key".into(),
        }),
    )
        .into_response()
}

fn internal() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError {
            error: "internal error".into(),
        }),
    )
        .into_response()
}
