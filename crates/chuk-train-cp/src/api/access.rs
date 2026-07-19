//! RBAC: users + API keys (self-service, with admin-scoped operations).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{ApiKeyInfo, CreateApiKeyRequest, CreatedApiKey, Role};
use serde::Deserialize;

use super::{bad_request, forbidden, internal, not_found, now, require_role};
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

/// The caller's own role/identity — lets the dashboard show admin-only controls.
pub async fn whoami(axum::Extension(ctx): axum::Extension<AuthContext>) -> Response {
    Json(serde_json::json!({
        "role": ctx.role.as_str(),
        "team_id": ctx.team_id,
        "subject": ctx.subject,
    }))
    .into_response()
}
