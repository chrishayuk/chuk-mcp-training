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
