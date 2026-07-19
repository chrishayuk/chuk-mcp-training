//! RBAC: users + API keys (self-service, with admin-scoped operations).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiKeyInfo, CreateApiKeyRequest, CreateWorkerTokenRequest, CreatedApiKey, CreatedWorkerToken,
    Role, WorkerId, WorkerTokenInfo,
};
use serde::Deserialize;

use super::{bad_request, forbidden, internal, not_found, now, require_role, service_unavailable};
use crate::apikey::{self, AuthContext};
use crate::AppState;

#[derive(Deserialize)]
pub struct UpsertUserRequest {
    email: String,
    role: Role,
}

pub async fn list_users(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    match state.hub.store.list_users(&ctx.team_id).await {
        Ok(users) => Json(users).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn upsert_user(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<UpsertUserRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    if request.role > ctx.role {
        return bad_request("cannot grant a role above your own");
    }
    let email = request.email.trim().to_lowercase();
    if email.is_empty() {
        return bad_request("email required");
    }
    match state
        .hub
        .store
        .upsert_user(&email, &ctx.team_id, request.role)
        .await
    {
        Ok(()) => Json(serde_json::json!({ "ok": true })).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn remove_user(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(email): Path<String>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let email = email.trim().to_lowercase();
    if email == ctx.subject {
        return bad_request("cannot remove yourself");
    }
    match state.hub.store.remove_user(&email).await {
        Ok(()) => Json(serde_json::json!({ "ok": true })).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn list_api_keys(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    // Self-service: any authenticated caller lists keys. Admins (and up) see the
    // whole team's keys; everyone else sees only the keys they created.
    match state.hub.store.list_api_keys(&ctx.team_id).await {
        Ok(keys) => {
            let visible: Vec<ApiKeyInfo> = if ctx.may(Role::Admin) {
                keys
            } else {
                keys.into_iter()
                    .filter(|k| k.created_by == ctx.subject)
                    .collect()
            };
            Json(visible).into_response()
        }
        Err(error) => internal(error),
    }
}

pub async fn create_api_key(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<CreateApiKeyRequest>,
) -> Response {
    // Self-service: any authenticated caller mints their own keys, scoped at or
    // below their own role (matching chuk-experiments-server). The ceiling check
    // is the only guard — there is no admin gate on minting.
    if request.role > ctx.role {
        return forbidden("cannot grant a key a role above your own");
    }
    let name = request.name.trim().to_owned();
    if name.is_empty() {
        return bad_request("key name required");
    }
    let (plaintext, prefix, hash) = apikey::generate();
    let id = uuid::Uuid::new_v4().simple().to_string();
    if let Err(error) = state
        .hub
        .store
        .create_api_key(&id, &ctx.team_id, &ctx.subject, &name, &prefix, &hash, request.role)
        .await
    {
        return internal(error);
    }
    let info = ApiKeyInfo {
        id,
        team_id: ctx.team_id.clone(),
        created_by: ctx.subject.clone(),
        name,
        prefix,
        role: request.role,
        created_at: now(),
        last_used_at: None,
        revoked_at: None,
    };
    Json(CreatedApiKey {
        key: plaintext,
        info,
    })
    .into_response()
}

pub async fn revoke_api_key(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    // Self-service: resolve the key within the caller's own team first, so a
    // non-admin can only ever touch their own keys and no one can revoke another
    // team's key by guessing its id. Admins (and up) may revoke any team key.
    let keys = match state.hub.store.list_api_keys(&ctx.team_id).await {
        Ok(keys) => keys,
        Err(error) => return internal(error),
    };
    let Some(key) = keys.into_iter().find(|k| k.id == id) else {
        return not_found();
    };
    if !ctx.may(Role::Admin) && key.created_by != ctx.subject {
        return forbidden("you can only revoke keys you created");
    }
    match state.hub.store.revoke_api_key(&id).await {
        Ok(true) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(false) => not_found(),
        Err(error) => internal(error),
    }
}

// ---------------------------------------------------------------------------
// Persistent worker tokens (chuk-compute M3.1) — admin-scoped infrastructure.
// Unlike api_keys (self-service user/MCP keys), these enrol a persistent worker
// and bind it to a stable worker id, so minting/listing/revoking is Admin-only.
// ---------------------------------------------------------------------------

pub async fn list_worker_tokens(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    match state.hub.store.list_worker_tokens().await {
        Ok(tokens) => Json(tokens).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn create_worker_token(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<CreateWorkerTokenRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let name = request.name.trim().to_owned();
    if name.is_empty() {
        return bad_request("worker token name required");
    }
    let (plaintext, prefix, hash) = apikey::generate_worker_token();
    // Stable worker id minted at creation time; the persistent worker resolves
    // it at websocket join (handled separately in ws.rs).
    let worker_id = WorkerId(format!(
        "w-{}",
        uuid::Uuid::new_v4().simple().to_string()[..8].to_owned()
    ));
    let id = uuid::Uuid::new_v4().simple().to_string();
    if let Err(error) = state
        .hub
        .store
        .create_worker_token(&id, &worker_id, &name, &prefix, &hash)
        .await
    {
        return internal(error);
    }
    let info = WorkerTokenInfo {
        id,
        worker_id,
        name,
        prefix,
        created_at: now(),
        last_used_at: None,
        revoked_at: None,
    };
    Json(CreatedWorkerToken {
        token: plaintext,
        info,
    })
    .into_response()
}

pub async fn revoke_worker_token(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(id): Path<String>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    match state.hub.store.revoke_worker_token(&id).await {
        Ok(true) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(false) => not_found(),
        Err(error) => internal(error),
    }
}

/// The caller's own role/identity — lets the dashboard show admin-only controls.
pub async fn whoami(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    let experiments_key_set = state
        .hub
        .store
        .user_experiments_key(&ctx.owner_email)
        .await
        .ok()
        .flatten()
        .is_some();
    Json(serde_json::json!({
        "role": ctx.role.as_str(),
        "team_id": ctx.team_id,
        "subject": ctx.subject,
        "experiments_key_set": experiments_key_set,
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct SetExperimentsKeyRequest {
    api_key: String,
}

/// Link the caller's own chuk-experiments-server API key (one they minted
/// themselves on chuk-experiments-server's own Team screen) so their mirrored
/// runs report under their own identity instead of the shared default. Stored
/// encrypted; never echoed back after this call.
pub async fn set_experiments_key(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SetExperimentsKeyRequest>,
) -> Response {
    let Some(encryption_key) = &state.key_encryption_key else {
        return service_unavailable("CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY not configured");
    };
    if ctx.owner_email == apikey::MASTER_TOKEN_SENTINEL {
        return bad_request(
            "the shared master token has no per-user identity to link a key against — \
             sign in or use a scoped API key",
        );
    }
    let raw = request.api_key.trim();
    if raw.is_empty() {
        return bad_request("api_key required");
    }
    let encrypted = crate::crypto::encrypt(encryption_key, raw);
    match state
        .hub
        .store
        .set_user_experiments_key(&ctx.owner_email, Some(&encrypted))
        .await
    {
        Ok(()) => Json(serde_json::json!({ "experiments_key_set": true })).into_response(),
        Err(error) => internal(error),
    }
}

/// Clear the caller's own linked chuk-experiments-server key; their mirrored
/// runs fall back to the shared default.
pub async fn clear_experiments_key(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    if state.key_encryption_key.is_none() {
        return service_unavailable("CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY not configured");
    }
    if ctx.owner_email == apikey::MASTER_TOKEN_SENTINEL {
        return bad_request(
            "the shared master token has no per-user identity to link a key against — \
             sign in or use a scoped API key",
        );
    }
    match state
        .hub
        .store
        .set_user_experiments_key(&ctx.owner_email, None)
        .await
    {
        Ok(()) => Json(serde_json::json!({ "experiments_key_set": false })).into_response(),
        Err(error) => internal(error),
    }
}
