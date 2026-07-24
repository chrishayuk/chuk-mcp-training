//! The operator dashboard (spec §9): served by the control plane itself,
//! self-contained (no external assets), dark mission-control theme. Two views
//! behind a hash router — an **overview** (health · fleet · runs · money) and a
//! full **per-run** view (live loss curve, streamed logs, config, checkpoints
//! with metadata + download links, events, out-links). Reads the same `/api/*`
//! the MCP tools do; the Google session cookie (or the API-token box in local
//! dev) authenticates every call; live updates via polling.
//!
//! The page is assembled from three sibling assets so each language lives in its
//! own file with proper tooling — the HTML shell (`dash/index.html`), the
//! stylesheet (`dash/dash.css`), and the app script (`dash/app.js`). `render()`
//! inlines them (the served page carries no external assets) and fills the two
//! runtime slots.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Redirect, Response};

use crate::AppState;

/// Overview refresh cadence; the per-run view polls faster (see `RUN_MS` in
/// `dash/app.js`).
const REFRESH_MS: u32 = 4000;

/// The three inlined assets. `{CSS}` / `{JS}` in the shell are the injection
/// slots; `{REFRESH_MS}` (in the script) and `{HEADER_RIGHT}` (in the header) are
/// filled per request by [`render`].
const HTML: &str = include_str!("dash/index.html");
const CSS: &str = include_str!("dash/dash.css");
const APP_JS: &str = include_str!("dash/app.js");

/// Serve the dashboard. When Google sign-in is configured, a valid session
/// cookie is required (else redirect to login); otherwise the API-token box is
/// shown (local dev).
pub async fn dashboard(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if state.config.auth_enabled() {
        match crate::auth::session_email(&state, &headers) {
            Some(email) => Html(render(Some(&email), true)).into_response(),
            None => Redirect::to("/auth/login").into_response(),
        }
    } else {
        Html(render(None, false)).into_response()
    }
}

fn render(user: Option<&str>, auth_enabled: bool) -> String {
    let header_right = if auth_enabled {
        format!(
            r#"<span class="who">{} · <a href="/auth/logout">sign out</a></span>"#,
            user.unwrap_or("")
        )
    } else {
        r#"<input id="tok" type="password" placeholder="API token" autocomplete="off">"#.to_owned()
    };
    // Inline the assets first, then fill the runtime slots (the script's
    // `{REFRESH_MS}` and the header's `{HEADER_RIGHT}`).
    HTML.replace("{CSS}", CSS)
        .replace("{JS}", APP_JS)
        .replace("{HEADER_RIGHT}", &header_right)
        .replace("{REFRESH_MS}", &REFRESH_MS.to_string())
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Arc;
    use std::time::Duration;

    use axum::http::{header, HeaderValue, StatusCode};

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::SqliteStore;
    use crate::AppState;

    fn base_config(auth_enabled: bool) -> Config {
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
            google_client_id: auth_enabled.then(|| "client-id".to_owned()),
            google_client_secret: auth_enabled.then(|| "client-secret".to_owned()),
            allowed_emails: if auth_enabled { vec!["ops@example.com".into()] } else { vec![] },
            sysadmin_email: None,
        }
    }

    /// A real (if minimal) `AppState`: `dashboard` takes `State<Arc<AppState>>`
    /// directly, so there is no lighter seam than building one — a mock/no-op
    /// `Providers` (the `mock` provider does no I/O at construction) keeps it
    /// cheap.
    async fn test_state(auth_enabled: bool) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let config = base_config(auth_enabled);
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

    /// Mints a session cookie `auth::session_email` (private to that module)
    /// will accept: same HMAC-SHA256-over-`email|expiry` scheme, keyed by the
    /// api token, that `auth::sign` uses — close enough to drive `dashboard`'s
    /// authenticated branch without reaching into `auth`'s private API.
    fn signed_session_cookie(api_token: &str, email: &str) -> HeaderValue {
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        use base64::Engine;
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let exp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_secs() as i64
            + 3600;
        let payload = format!("{email}|{exp}");
        let mut mac = Hmac::<Sha256>::new_from_slice(api_token.as_bytes()).expect("hmac key");
        mac.update(payload.as_bytes());
        let sig = hex::encode(mac.finalize().into_bytes());
        let b64 = URL_SAFE_NO_PAD.encode(&payload);
        HeaderValue::from_str(&format!("chuk_session={b64}.{sig}")).expect("valid header value")
    }

    #[tokio::test]
    async fn serves_the_token_box_when_google_signin_is_not_configured() {
        let state = test_state(false).await;
        let response = dashboard(State(state), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap().to_str().unwrap(),
            "text/html; charset=utf-8"
        );
    }

    #[tokio::test]
    async fn redirects_to_login_when_signin_required_but_no_session_cookie() {
        let state = test_state(true).await;
        let response = dashboard(State(state), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            response.headers().get(header::LOCATION).unwrap().to_str().unwrap(),
            "/auth/login"
        );
    }

    #[tokio::test]
    async fn serves_the_page_for_a_valid_allowlisted_session() {
        let state = test_state(true).await;
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            signed_session_cookie(&state.config.api_token, "ops@example.com"),
        );
        let response = dashboard(State(state), headers).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap().to_str().unwrap(),
            "text/html; charset=utf-8"
        );
    }

    #[test]
    fn render_shows_the_token_box_when_auth_is_disabled() {
        let html = render(None, false);
        assert!(html.contains(r#"id="tok""#));
        assert!(html.contains("4000"), "REFRESH_MS must be filled in");
        assert!(!html.contains("{REFRESH_MS}") && !html.contains("{HEADER_RIGHT}"));
        assert!(!html.contains("sign out"));
    }

    #[test]
    fn render_shows_the_signed_in_user_and_sign_out_link_when_auth_is_enabled() {
        let html = render(Some("chris@example.com"), true);
        assert!(html.contains("chris@example.com"));
        assert!(html.contains("sign out"));
        assert!(!html.contains(r#"id="tok""#));
    }
}
