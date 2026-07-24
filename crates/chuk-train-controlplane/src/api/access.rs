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

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use chuk_train_proto::{ApiError, User, WorkerId};

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::SqliteStore;
    use crate::AppState;

    fn base_config() -> Config {
        Config {
            api_token: "test-api-token".into(),
            join_token: "test-join-token".into(),
            store_spec: ":memory:".into(),
            artifacts_spec: "file:./unused".into(),
            public_url: "http://127.0.0.1:9".into(),
            host: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            port: 9,
            providers: "mock".into(),
            agent_ws_url: "ws://127.0.0.1:9/ws".into(),
            agent_bin: None,
            agent_dir: None,
            min_protocol: 0,
            vast_api_key: None,
            drain_window_min: 5.0,
            confirm_cost_threshold: 0.0,
            reconcile_interval: Duration::from_secs(30),
            idle_reap: Duration::from_secs(60),
            google_client_id: None,
            google_client_secret: None,
            allowed_emails: vec![],
            sysadmin_email: None,
        }
    }

    /// A real (if minimal) `AppState`: every handler here takes
    /// `State<Arc<AppState>>` directly, so there is no lighter seam than
    /// building one — mirrors `dash.rs`'s and `api/system.rs`'s `test_state`
    /// helper. `key_encryption_key` is the one field these handlers vary on.
    async fn test_state(key_encryption_key: Option<[u8; 32]>) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let config = base_config();
        let leases = LeaseManager::new(hub.clone(), providers, config.clone());
        Arc::new(AppState {
            config,
            hub,
            artifacts,
            leases,
            drive: None,
            archiver: None,
            key_encryption_key,
        })
    }

    /// Builds an `AuthContext` the way `resolve_auth` would for a signed-in
    /// user or scoped API key: `subject` and `owner_email` both the given
    /// email (the `subject != owner_email` case only happens for a scoped
    /// key, which none of these handlers special-case beyond `owner_email`).
    fn ctx(role: Role, team_id: &str, email: &str) -> AuthContext {
        AuthContext {
            role,
            team_id: team_id.to_owned(),
            subject: email.to_owned(),
            owner_email: email.to_owned(),
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    async fn body_error(resp: Response) -> String {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let error: ApiError = serde_json::from_slice(&bytes).expect("json body");
        error.error
    }

    // -- list_users / upsert_user / remove_user -----------------------------

    #[tokio::test]
    async fn list_users_requires_admin_role() {
        let state = test_state(None).await;
        let resp = list_users(State(state), axum::Extension(ctx(Role::Write, "default", "w@example.com"))).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn list_users_returns_only_the_callers_team() {
        let state = test_state(None).await;
        state
            .hub
            .store
            .upsert_user("a@example.com", "default", Role::Write)
            .await
            .expect("seed a");
        state
            .hub
            .store
            .upsert_user("b@example.com", "other-team", Role::Read)
            .await
            .expect("seed b");

        let resp = list_users(State(state), axum::Extension(ctx(Role::Admin, "default", "admin@example.com"))).await;

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let users: Vec<User> = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].email, "a@example.com");
    }

    #[tokio::test]
    async fn upsert_user_requires_admin_role() {
        let state = test_state(None).await;
        let resp = upsert_user(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "w@example.com")),
            Json(UpsertUserRequest {
                email: "new@example.com".into(),
                role: Role::Read,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn upsert_user_rejects_granting_role_above_the_callers_own() {
        let state = test_state(None).await;
        let resp = upsert_user(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Json(UpsertUserRequest {
                email: "new@example.com".into(),
                role: Role::Sysadmin,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "cannot grant a role above your own");
    }

    #[tokio::test]
    async fn upsert_user_rejects_empty_email() {
        let state = test_state(None).await;
        let resp = upsert_user(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Json(UpsertUserRequest {
                email: "   ".into(),
                role: Role::Read,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "email required");
    }

    #[tokio::test]
    async fn upsert_user_lowercases_the_email_and_is_idempotent_on_role_change() {
        let state = test_state(None).await;
        let admin = ctx(Role::Admin, "default", "admin@example.com");
        let resp = upsert_user(
            State(state.clone()),
            axum::Extension(admin.clone()),
            Json(UpsertUserRequest {
                email: "Foo@Example.COM".into(),
                role: Role::Read,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Re-upserting the same (now-lowercased) email with a new role updates
        // it in place rather than creating a second row.
        let resp = upsert_user(
            State(state.clone()),
            axum::Extension(admin),
            Json(UpsertUserRequest {
                email: "foo@example.com".into(),
                role: Role::Write,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);

        let users = state.hub.store.list_users("default").await.expect("list");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].email, "foo@example.com");
        assert_eq!(users[0].role, Role::Write);
    }

    #[tokio::test]
    async fn remove_user_requires_admin_role() {
        let state = test_state(None).await;
        let resp = remove_user(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "w@example.com")),
            Path("other@example.com".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn remove_user_rejects_removing_yourself() {
        let state = test_state(None).await;
        let resp = remove_user(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Path("admin@example.com".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "cannot remove yourself");
    }

    #[tokio::test]
    async fn remove_user_removes_another_user() {
        let state = test_state(None).await;
        state
            .hub
            .store
            .upsert_user("gone@example.com", "default", Role::Read)
            .await
            .expect("seed");

        let resp = remove_user(
            State(state.clone()),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Path("gone@example.com".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["ok"], true);
        assert!(state
            .hub
            .store
            .list_users("default")
            .await
            .expect("list")
            .is_empty());
    }

    // -- list_api_keys / create_api_key / revoke_api_key ---------------------

    #[tokio::test]
    async fn list_api_keys_admin_sees_the_whole_team() {
        let state = test_state(None).await;
        create_api_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(CreateApiKeyRequest {
                name: "alice-key".into(),
                role: Role::Write,
            }),
        )
        .await;
        create_api_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Admin, "default", "bob@example.com")),
            Json(CreateApiKeyRequest {
                name: "bob-key".into(),
                role: Role::Admin,
            }),
        )
        .await;

        let resp = list_api_keys(State(state), axum::Extension(ctx(Role::Admin, "default", "bob@example.com"))).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let keys: Vec<ApiKeyInfo> = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(keys.len(), 2);
    }

    #[tokio::test]
    async fn list_api_keys_non_admin_sees_only_their_own() {
        let state = test_state(None).await;
        create_api_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(CreateApiKeyRequest {
                name: "alice-key".into(),
                role: Role::Write,
            }),
        )
        .await;
        create_api_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Admin, "default", "bob@example.com")),
            Json(CreateApiKeyRequest {
                name: "bob-key".into(),
                role: Role::Admin,
            }),
        )
        .await;

        let resp = list_api_keys(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
        )
        .await;
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let keys: Vec<ApiKeyInfo> = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].created_by, "alice@example.com");
    }

    #[tokio::test]
    async fn create_api_key_rejects_a_role_above_the_callers_own() {
        let state = test_state(None).await;
        let resp = create_api_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(CreateApiKeyRequest {
                name: "escalate".into(),
                role: Role::Admin,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            body_error(resp).await,
            "cannot grant a key a role above your own"
        );
    }

    #[tokio::test]
    async fn create_api_key_rejects_an_empty_name() {
        let state = test_state(None).await;
        let resp = create_api_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(CreateApiKeyRequest {
                name: "   ".into(),
                role: Role::Read,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "key name required");
    }

    #[tokio::test]
    async fn create_api_key_mints_a_key_scoped_to_the_requested_role() {
        let state = test_state(None).await;
        let resp = create_api_key(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "alice@example.com")),
            Json(CreateApiKeyRequest {
                name: "my-key".into(),
                role: Role::Write,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let created: CreatedApiKey = serde_json::from_slice(&bytes).expect("json body");
        assert!(created.key.starts_with(chuk_train_proto::API_KEY_PREFIX));
        assert_eq!(created.info.name, "my-key");
        assert_eq!(created.info.role, Role::Write);
        assert_eq!(created.info.created_by, "alice@example.com");
        assert!(created.info.revoked_at.is_none());
    }

    #[tokio::test]
    async fn revoke_api_key_404s_for_an_unknown_id() {
        let state = test_state(None).await;
        let resp = revoke_api_key(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Path("no-such-key".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    async fn seed_api_key(state: &Arc<AppState>, owner: &str, role: Role) -> String {
        let resp = create_api_key(
            State(state.clone()),
            axum::Extension(ctx(role, "default", owner)),
            Json(CreateApiKeyRequest {
                name: "seeded".into(),
                role,
            }),
        )
        .await;
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let created: CreatedApiKey = serde_json::from_slice(&bytes).expect("json body");
        created.info.id
    }

    #[tokio::test]
    async fn revoke_api_key_forbids_a_non_owner_non_admin() {
        let state = test_state(None).await;
        let id = seed_api_key(&state, "alice@example.com", Role::Write).await;

        let resp = revoke_api_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "mallory@example.com")),
            Path(id),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(body_error(resp).await, "you can only revoke keys you created");
    }

    #[tokio::test]
    async fn revoke_api_key_allows_the_owner_to_revoke_their_own_key() {
        let state = test_state(None).await;
        let id = seed_api_key(&state, "alice@example.com", Role::Write).await;

        let resp = revoke_api_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Path(id),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["ok"], true);
    }

    #[tokio::test]
    async fn revoke_api_key_allows_an_admin_to_revoke_anyones_key() {
        let state = test_state(None).await;
        let id = seed_api_key(&state, "alice@example.com", Role::Write).await;

        let resp = revoke_api_key(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Path(id),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["ok"], true);
    }

    #[tokio::test]
    async fn revoke_api_key_404s_on_a_second_revoke_of_an_already_revoked_key() {
        let state = test_state(None).await;
        let id = seed_api_key(&state, "alice@example.com", Role::Write).await;
        let owner = ctx(Role::Write, "default", "alice@example.com");

        let first = revoke_api_key(State(state.clone()), axum::Extension(owner.clone()), Path(id.clone())).await;
        assert_eq!(first.status(), StatusCode::OK);

        let second = revoke_api_key(State(state), axum::Extension(owner), Path(id)).await;
        assert_eq!(second.status(), StatusCode::NOT_FOUND);
    }

    // -- worker tokens (admin-only) -------------------------------------------

    #[tokio::test]
    async fn list_worker_tokens_requires_admin_role() {
        let state = test_state(None).await;
        let resp = list_worker_tokens(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "w@example.com")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn create_worker_token_requires_admin_role() {
        let state = test_state(None).await;
        let resp = create_worker_token(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "w@example.com")),
            Json(CreateWorkerTokenRequest { name: "gpu-box".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn create_worker_token_rejects_an_empty_name() {
        let state = test_state(None).await;
        let resp = create_worker_token(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Json(CreateWorkerTokenRequest { name: "  ".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "worker token name required");
    }

    #[tokio::test]
    async fn create_worker_token_mints_a_token_with_a_stable_worker_id_and_lists_it() {
        let state = test_state(None).await;
        let admin = ctx(Role::Admin, "default", "admin@example.com");
        let resp = create_worker_token(
            State(state.clone()),
            axum::Extension(admin.clone()),
            Json(CreateWorkerTokenRequest { name: "gpu-box".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let created: CreatedWorkerToken = serde_json::from_slice(&bytes).expect("json body");
        assert!(created.token.starts_with(chuk_train_proto::WORKER_TOKEN_PREFIX));
        assert!(created.info.worker_id.0.starts_with("w-"));
        assert_eq!(created.info.name, "gpu-box");

        let listed = list_worker_tokens(State(state), axum::Extension(admin)).await;
        assert_eq!(listed.status(), StatusCode::OK);
        let bytes = to_bytes(listed.into_body(), usize::MAX).await.expect("body");
        let tokens: Vec<WorkerTokenInfo> = serde_json::from_slice(&bytes).expect("json body");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].id, created.info.id);
    }

    #[tokio::test]
    async fn revoke_worker_token_requires_admin_role() {
        let state = test_state(None).await;
        let resp = revoke_worker_token(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "w@example.com")),
            Path("no-such-token".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn revoke_worker_token_404s_for_an_unknown_id() {
        let state = test_state(None).await;
        let resp = revoke_worker_token(
            State(state),
            axum::Extension(ctx(Role::Admin, "default", "admin@example.com")),
            Path("no-such-token".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn revoke_worker_token_revokes_an_existing_token() {
        let state = test_state(None).await;
        let admin = ctx(Role::Admin, "default", "admin@example.com");
        let created_resp = create_worker_token(
            State(state.clone()),
            axum::Extension(admin.clone()),
            Json(CreateWorkerTokenRequest { name: "gpu-box".into() }),
        )
        .await;
        let bytes = to_bytes(created_resp.into_body(), usize::MAX).await.expect("body");
        let created: CreatedWorkerToken = serde_json::from_slice(&bytes).expect("json body");

        let resp = revoke_worker_token(State(state.clone()), axum::Extension(admin.clone()), Path(created.info.id.clone())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["ok"], true);

        // Already revoked: a second call now 404s.
        let resp = revoke_worker_token(State(state), axum::Extension(admin), Path(created.info.id)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // -- whoami ----------------------------------------------------------------

    #[tokio::test]
    async fn whoami_reports_role_team_subject_and_no_linked_experiments_key() {
        let state = test_state(None).await;
        let resp = whoami(
            State(state),
            axum::Extension(ctx(Role::Write, "my-team", "alice@example.com")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["role"], "write");
        assert_eq!(body["team_id"], "my-team");
        assert_eq!(body["subject"], "alice@example.com");
        assert_eq!(body["experiments_key_set"], false);
    }

    #[tokio::test]
    async fn whoami_reports_experiments_key_set_once_linked() {
        let state = test_state(None).await;
        state
            .hub
            .store
            .set_user_experiments_key("alice@example.com", Some("ciphertext"))
            .await
            .expect("seed linked key");

        let resp = whoami(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
        )
        .await;
        assert_eq!(body_json(resp).await["experiments_key_set"], true);
    }

    // -- set_experiments_key / clear_experiments_key ----------------------------

    fn test_encryption_key() -> [u8; 32] {
        [7u8; 32]
    }

    #[tokio::test]
    async fn set_experiments_key_503s_when_encryption_is_not_configured() {
        let state = test_state(None).await;
        let resp = set_experiments_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(SetExperimentsKeyRequest { api_key: "secret".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn set_experiments_key_rejects_the_master_token_sentinel() {
        let state = test_state(Some(test_encryption_key())).await;
        let resp = set_experiments_key(
            State(state),
            axum::Extension(ctx(
                Role::Sysadmin,
                "default",
                apikey::MASTER_TOKEN_SENTINEL,
            )),
            Json(SetExperimentsKeyRequest { api_key: "secret".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(body_error(resp).await.contains("shared master token"));
    }

    #[tokio::test]
    async fn set_experiments_key_rejects_an_empty_key() {
        let state = test_state(Some(test_encryption_key())).await;
        let resp = set_experiments_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(SetExperimentsKeyRequest { api_key: "   ".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_error(resp).await, "api_key required");
    }

    #[tokio::test]
    async fn set_experiments_key_links_and_encrypts_the_key_so_it_round_trips() {
        let key = test_encryption_key();
        let state = test_state(Some(key)).await;
        let resp = set_experiments_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
            Json(SetExperimentsKeyRequest {
                api_key: "  hf_secret_token  ".into(),
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["experiments_key_set"], true);

        // The stored value is encrypted (never the plaintext) but decrypts
        // back to the trimmed input under the configured key.
        let stored = state
            .hub
            .store
            .user_experiments_key("alice@example.com")
            .await
            .expect("lookup")
            .expect("a key was linked");
        assert_ne!(stored, "hf_secret_token");
        assert_eq!(crate::crypto::decrypt(&key, &stored).expect("decrypt"), "hf_secret_token");
    }

    #[tokio::test]
    async fn clear_experiments_key_503s_when_encryption_is_not_configured() {
        let state = test_state(None).await;
        let resp = clear_experiments_key(
            State(state),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn clear_experiments_key_rejects_the_master_token_sentinel() {
        let state = test_state(Some(test_encryption_key())).await;
        let resp = clear_experiments_key(
            State(state),
            axum::Extension(ctx(
                Role::Sysadmin,
                "default",
                apikey::MASTER_TOKEN_SENTINEL,
            )),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(body_error(resp).await.contains("shared master token"));
    }

    #[tokio::test]
    async fn clear_experiments_key_clears_a_previously_linked_key() {
        let key = test_encryption_key();
        let state = test_state(Some(key)).await;
        state
            .hub
            .store
            .set_user_experiments_key("alice@example.com", Some(&crate::crypto::encrypt(&key, "old-key")))
            .await
            .expect("seed linked key");

        let resp = clear_experiments_key(
            State(state.clone()),
            axum::Extension(ctx(Role::Write, "default", "alice@example.com")),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["experiments_key_set"], false);
        assert!(state
            .hub
            .store
            .user_experiments_key("alice@example.com")
            .await
            .expect("lookup")
            .is_none());
    }

    // WorkerId is only exercised indirectly above (via CreatedWorkerToken); this
    // guards the import doesn't go dead if that changes.
    #[test]
    fn worker_id_prefix_matches_the_minted_convention() {
        assert!(WorkerId("w-abc12345".into()).0.starts_with("w-"));
    }
}
