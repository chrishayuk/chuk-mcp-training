//! Bearer-authenticated REST API — the surface the MCP server (and the
//! dashboard's JavaScript) consume.

mod access;
mod archive;
mod checkpoints;
mod leases;
mod runs;
mod system;

pub use access::*;
pub use archive::*;
pub use checkpoints::*;
pub use leases::*;
pub use runs::*;
pub use system::*;

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{ApiError, Role, DEFAULT_TEAM_ID};

use crate::apikey::{self, AuthContext};
use crate::AppState;

pub(crate) const BEARER_PREFIX: &str = "Bearer ";
pub(crate) const ERR_UNAUTHORIZED: &str = "bad or missing bearer token";
pub(crate) const ERR_RUN_NOT_FOUND: &str = "no such run";
pub(crate) const ERR_CKPT_NOT_FOUND: &str = "no such checkpoint";
pub(crate) const ERR_INTERNAL: &str = "internal error";

/// 500-with-envelope for store failures; the MCP layer relays the envelope.
pub(crate) fn internal(error: anyhow::Error) -> Response {
    tracing::error!(%error, "api internal error");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError {
            error: ERR_INTERNAL.into(),
        }),
    )
        .into_response()
}

pub(crate) fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ApiError {
            error: ERR_RUN_NOT_FOUND.into(),
        }),
    )
        .into_response()
}

pub(crate) fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
        .into_response()
}

pub(crate) fn forbidden(message: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
        .into_response()
}

/// A feature that's optional and gated off (e.g. no encryption key configured)
/// rather than genuinely broken — distinct from `internal`.
pub(crate) fn service_unavailable(message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
        .into_response()
}

pub(crate) fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// Auth middleware for /api/*
// ---------------------------------------------------------------------------

pub async fn require_bearer(
    State(state): State<Arc<AppState>>,
    mut request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let Some(ctx) = resolve_auth(&state, request.headers()).await else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ApiError {
                error: ERR_UNAUTHORIZED.into(),
            }),
        )
            .into_response();
    };
    // Handlers extract the context (via require_role) to enforce min-role.
    request.extensions_mut().insert(ctx);
    next.run(request).await
}

/// Resolve a request to a role-bearing auth context: a bearer token (the legacy
/// master token → sysadmin, or a scoped API key → its role), or a Google session
/// cookie → the signed-in user's role. `None` means unauthenticated.
async fn resolve_auth(state: &AppState, headers: &HeaderMap) -> Option<AuthContext> {
    if let Some(token) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix(BEARER_PREFIX))
    {
        if token == state.config.api_token {
            return Some(AuthContext {
                role: Role::Sysadmin,
                team_id: DEFAULT_TEAM_ID.to_owned(),
                subject: apikey::MASTER_TOKEN_SENTINEL.to_owned(),
                owner_email: apikey::MASTER_TOKEN_SENTINEL.to_owned(),
            });
        }
        if let Ok(Some(info)) = state
            .hub
            .store
            .resolve_api_key(&apikey::hash_token(token))
            .await
        {
            let _ = state.hub.store.touch_api_key(&info.id, now()).await;
            return Some(AuthContext {
                role: info.role,
                team_id: info.team_id,
                owner_email: info.created_by.clone(),
                subject: info.prefix,
            });
        }
    }
    if state.config.auth_enabled() {
        if let Some(email) = crate::auth::session_email(state, headers) {
            let user = state.hub.store.get_user(&email).await.ok().flatten();
            let (role, team_id) = user
                .map(|u| (u.role, u.team_id))
                .unwrap_or((Role::Read, DEFAULT_TEAM_ID.to_owned()));
            return Some(AuthContext {
                role,
                team_id,
                owner_email: email.clone(),
                subject: email,
            });
        }
    }
    None
}

/// Enforce a minimum role inside a handler. Returns `Ok` when allowed, or a
/// ready 403 response to return early. The [`AuthContext`] is provided by
/// [`require_bearer`] via `Extension<AuthContext>`.
pub fn require_role(ctx: &AuthContext, min: Role) -> Result<(), Response> {
    if ctx.may(min) {
        return Ok(());
    }
    Err((
        StatusCode::FORBIDDEN,
        Json(ApiError {
            error: format!("requires {} role", min.as_str()),
        }),
    )
        .into_response())
}
