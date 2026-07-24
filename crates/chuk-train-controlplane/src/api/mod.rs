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
// One call per request, not a hot loop — boxing `Response` would just push
// an allocation onto every one of this fn's ~20 call sites for no benefit.
#[allow(clippy::result_large_err)]
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

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use axum::extract::Extension;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

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

    /// A dashboard-auth-enabled config: a Google client is configured and
    /// `allowed_emails` is non-empty, so `Config::auth_enabled()` is true and
    /// `resolve_auth` will consider the Google-session cookie branch.
    fn auth_enabled_config(allowed_email: &str) -> Config {
        Config {
            google_client_id: Some("client-id".into()),
            google_client_secret: Some("client-secret".into()),
            allowed_emails: vec![allowed_email.to_owned()],
            ..base_config()
        }
    }

    /// A real (if minimal) `AppState` — `require_bearer`/`resolve_auth` take
    /// `&AppState`/`State<Arc<AppState>>` directly, so there's no lighter seam.
    /// Mirrors `system.rs`'s/`checkpoints.rs`'s `test_state` helper.
    async fn test_state(config: Config) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let leases = LeaseManager::new(hub.clone(), providers, config.clone());
        Arc::new(AppState {
            config,
            hub,
            artifacts,
            leases,
            drive: None,
            archiver: None,
            key_encryption_key: None,
        })
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn bearer_headers(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            format!("Bearer {token}").parse().expect("header value"),
        );
        headers
    }

    /// Builds a `chuk_session=...` cookie header identical in format to
    /// `auth::sign`/`session_email`'s scheme (HMAC-SHA256 over `email|expiry`,
    /// keyed by the API token) — the only way to exercise `resolve_auth`'s
    /// Google-session branch without a live Google OAuth round trip.
    fn session_cookie_header(api_token: &str, email: &str, expires_at: i64) -> HeaderMap {
        let payload = format!("{email}|{expires_at}");
        let b64 = URL_SAFE_NO_PAD.encode(&payload);
        let mut mac =
            Hmac::<Sha256>::new_from_slice(api_token.as_bytes()).expect("hmac accepts any key length");
        mac.update(payload.as_bytes());
        let sig = hex::encode(mac.finalize().into_bytes());
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            format!("chuk_session={b64}.{sig}").parse().expect("header value"),
        );
        headers
    }

    fn far_future() -> i64 {
        // `session_email` compares against `now()` in whole seconds; a day out
        // is comfortably not-expired without hardcoding a specific instant.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs() as i64
            + 86_400
    }

    // -- response builders ---------------------------------------------------

    #[tokio::test]
    async fn internal_hides_the_real_error_behind_a_generic_message() {
        let resp = internal(anyhow::anyhow!("db connection reset by peer"));
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(resp).await;
        // The real error is only for `tracing::error!` -- never handed to the
        // client, which only ever sees the generic ERR_INTERNAL message.
        assert_eq!(body["error"], ERR_INTERNAL);
    }

    #[tokio::test]
    async fn not_found_reports_the_shared_not_found_message() {
        let resp = not_found();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"], ERR_RUN_NOT_FOUND);
    }

    #[tokio::test]
    async fn bad_request_echoes_the_given_message() {
        let resp = bad_request("missing field: run_id");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "missing field: run_id");
    }

    #[tokio::test]
    async fn forbidden_echoes_the_given_message() {
        let resp = forbidden("not your team");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "not your team");
    }

    #[tokio::test]
    async fn service_unavailable_echoes_the_given_message() {
        let resp = service_unavailable("archive tier not configured");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "archive tier not configured");
    }

    #[test]
    fn now_returns_the_current_unix_time_in_seconds() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs_f64();
        let got = now();
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs_f64();
        assert!(
            got >= before && got <= after,
            "now() = {got} not within [{before}, {after}]"
        );
    }

    // -- require_role ---------------------------------------------------------

    fn ctx(role: Role) -> AuthContext {
        AuthContext {
            role,
            team_id: "default".into(),
            subject: "tester".into(),
            owner_email: "tester@example.com".into(),
        }
    }

    #[test]
    fn require_role_allows_a_context_that_meets_or_exceeds_the_bar() {
        assert!(require_role(&ctx(Role::Admin), Role::Write).is_ok());
        assert!(require_role(&ctx(Role::Write), Role::Write).is_ok());
        assert!(require_role(&ctx(Role::Sysadmin), Role::Sysadmin).is_ok());
    }

    #[tokio::test]
    async fn require_role_rejects_below_the_bar_with_a_403_naming_the_required_role() {
        let resp = require_role(&ctx(Role::Read), Role::Admin).expect_err("below bar");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "requires admin role");
    }

    // -- resolve_auth -----------------------------------------------------------

    #[tokio::test]
    async fn resolve_auth_is_none_with_no_authorization_header_and_auth_disabled() {
        let state = test_state(base_config()).await;
        assert!(resolve_auth(&state, &HeaderMap::new()).await.is_none());
    }

    #[tokio::test]
    async fn resolve_auth_is_none_for_a_header_missing_the_bearer_prefix() {
        let state = test_state(base_config()).await;
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Basic dGVzdA==".parse().unwrap());
        assert!(resolve_auth(&state, &headers).await.is_none());
    }

    #[tokio::test]
    async fn resolve_auth_recognizes_the_configured_master_token_as_sysadmin() {
        let state = test_state(base_config()).await;
        let resolved = resolve_auth(&state, &bearer_headers("test-api-token"))
            .await
            .expect("master token resolves");
        assert_eq!(resolved.role, Role::Sysadmin);
        assert_eq!(resolved.team_id, DEFAULT_TEAM_ID);
        assert_eq!(resolved.subject, apikey::MASTER_TOKEN_SENTINEL);
        assert_eq!(resolved.owner_email, apikey::MASTER_TOKEN_SENTINEL);
    }

    #[tokio::test]
    async fn resolve_auth_resolves_a_scoped_api_key_and_touches_its_last_used_at() {
        let state = test_state(base_config()).await;
        state
            .hub
            .store
            .ensure_team("team-x", "Team X")
            .await
            .expect("team");
        let token = "ck_testtoken0000000000000000";
        let hash = apikey::hash_token(token);
        state
            .hub
            .store
            .create_api_key(
                "key-1",
                "team-x",
                "owner@example.com",
                "ci key",
                "ck_test0",
                &hash,
                Role::Write,
            )
            .await
            .expect("create key");

        let resolved = resolve_auth(&state, &bearer_headers(token))
            .await
            .expect("api key resolves");
        assert_eq!(resolved.role, Role::Write);
        assert_eq!(resolved.team_id, "team-x");
        assert_eq!(resolved.owner_email, "owner@example.com");
        assert_eq!(resolved.subject, "ck_test0");

        // `resolve_auth` also touches `last_used_at` on every successful use.
        let keys = state.hub.store.list_api_keys("team-x").await.expect("list");
        let key = keys.iter().find(|k| k.id == "key-1").expect("key present");
        assert!(key.last_used_at.is_some());
    }

    #[tokio::test]
    async fn resolve_auth_is_none_for_an_unresolvable_bearer_token_when_auth_is_disabled() {
        let state = test_state(base_config()).await;
        assert!(resolve_auth(&state, &bearer_headers("not-a-real-token"))
            .await
            .is_none());
    }

    #[tokio::test]
    async fn resolve_auth_ignores_a_well_formed_session_cookie_when_auth_is_disabled() {
        let state = test_state(base_config()).await;
        let headers = session_cookie_header("test-api-token", "someone@example.com", far_future());
        assert!(resolve_auth(&state, &headers).await.is_none());
    }

    #[tokio::test]
    async fn resolve_auth_accepts_a_google_session_and_uses_the_stored_users_role_and_team() {
        let state = test_state(auth_enabled_config("allowed@example.com")).await;
        state
            .hub
            .store
            .ensure_team("team-y", "Team Y")
            .await
            .expect("team");
        state
            .hub
            .store
            .upsert_user("allowed@example.com", "team-y", Role::Admin)
            .await
            .expect("user");

        let headers = session_cookie_header("test-api-token", "allowed@example.com", far_future());
        let resolved = resolve_auth(&state, &headers)
            .await
            .expect("session resolves");
        assert_eq!(resolved.role, Role::Admin);
        assert_eq!(resolved.team_id, "team-y");
        assert_eq!(resolved.subject, "allowed@example.com");
        assert_eq!(resolved.owner_email, "allowed@example.com");
    }

    #[tokio::test]
    async fn resolve_auth_defaults_a_first_time_session_user_to_read_role_on_the_default_team() {
        let state = test_state(auth_enabled_config("allowed@example.com")).await;
        // No `upsert_user` call -- this email has never signed in before.
        let headers = session_cookie_header("test-api-token", "allowed@example.com", far_future());
        let resolved = resolve_auth(&state, &headers)
            .await
            .expect("session resolves even with no user row yet");
        assert_eq!(resolved.role, Role::Read);
        assert_eq!(resolved.team_id, DEFAULT_TEAM_ID);
    }

    #[tokio::test]
    async fn resolve_auth_rejects_an_expired_session_cookie() {
        let state = test_state(auth_enabled_config("allowed@example.com")).await;
        let expired = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs() as i64
            - 3600;
        let headers = session_cookie_header("test-api-token", "allowed@example.com", expired);
        assert!(resolve_auth(&state, &headers).await.is_none());
    }

    #[tokio::test]
    async fn resolve_auth_rejects_a_session_for_an_email_off_the_allowlist() {
        let state = test_state(auth_enabled_config("allowed@example.com")).await;
        let headers = session_cookie_header("test-api-token", "not-allowed@example.com", far_future());
        assert!(resolve_auth(&state, &headers).await.is_none());
    }

    // -- require_bearer (the actual axum middleware) ---------------------------

    async fn probe(Extension(ctx): Extension<AuthContext>) -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "role": ctx.role.as_str(),
            "team_id": ctx.team_id,
            "subject": ctx.subject,
            "owner_email": ctx.owner_email,
        }))
    }

    /// Wires `require_bearer` in front of a probe route exactly the way
    /// `main.rs` wires it in front of `/api/*`, then serves it on a loopback
    /// port. Only a real request through the middleware pipeline exercises
    /// `require_bearer`'s own 401 short-circuit and its
    /// `request.extensions_mut().insert(ctx)` handoff to the next handler --
    /// `resolve_auth` alone (tested above) can't reach those lines.
    async fn spawn_probe_server(state: Arc<AppState>) -> String {
        let app = axum::Router::new()
            .route("/probe", axum::routing::get(probe))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                require_bearer,
            ))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("test server");
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn require_bearer_rejects_missing_and_malformed_auth_with_401() {
        let state = test_state(base_config()).await;
        let base = spawn_probe_server(state).await;
        let client = reqwest::Client::new();

        let resp = client
            .get(format!("{base}/probe"))
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["error"], ERR_UNAUTHORIZED);

        let resp = client
            .get(format!("{base}/probe"))
            .header(reqwest::header::AUTHORIZATION, "Basic dGVzdA==")
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), reqwest::StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn require_bearer_admits_the_master_token_and_hands_the_handler_a_sysadmin_context() {
        let state = test_state(base_config()).await;
        let base = spawn_probe_server(state).await;
        let client = reqwest::Client::new();

        let resp = client
            .get(format!("{base}/probe"))
            .header(reqwest::header::AUTHORIZATION, "Bearer test-api-token")
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value = resp.json().await.expect("json body");
        assert_eq!(body["role"], "sysadmin");
        assert_eq!(body["subject"], apikey::MASTER_TOKEN_SENTINEL);
    }
}
