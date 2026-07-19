//! Dashboard sign-in via Google OAuth (spec §12 — the dashboard behind auth).
//!
//! Flow: `/auth/login` redirects to Google; `/auth/callback` exchanges the code,
//! reads the verified email from Google's userinfo, checks it against the
//! allowlist, and sets a signed session cookie. The dashboard page requires
//! that cookie; the bearer-API middleware accepts it as an alternative to the
//! API token, so a logged-in browser needs no token while MCP + agents keep
//! their tokens. Enabled only when a Google client is configured — otherwise
//! the dashboard falls back to the API-token box (local dev).

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

use crate::AppState;

const SESSION_COOKIE: &str = "chuk_session";
const STATE_COOKIE: &str = "chuk_oauth_state";
const SESSION_TTL: Duration = Duration::from_secs(7 * 24 * 3600);
const STATE_TTL_SECS: i64 = 600;
const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_USERINFO_URL: &str = "https://openidconnect.googleapis.com/v1/userinfo";

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs() as i64
}

// -- signed cookies (HMAC-SHA256, keyed by the API token) -------------------

fn hmac_hex(key: &[u8], data: &[u8]) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    hex::encode(mac.finalize().into_bytes())
}

fn hmac_ok(key: &[u8], data: &[u8], sig_hex: &str) -> bool {
    let Ok(sig) = hex::decode(sig_hex) else {
        return false;
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.verify_slice(&sig).is_ok()
}

/// `base64url(payload).hmac_hex` where payload is `email|expiry`.
fn sign(key: &[u8], payload: &str) -> String {
    format!(
        "{}.{}",
        URL_SAFE_NO_PAD.encode(payload),
        hmac_hex(key, payload.as_bytes())
    )
}

fn unsign(key: &[u8], token: &str) -> Option<String> {
    let (b64, sig) = token.split_once('.')?;
    let payload = String::from_utf8(URL_SAFE_NO_PAD.decode(b64).ok()?).ok()?;
    if !hmac_ok(key, payload.as_bytes(), sig) {
        return None;
    }
    Some(payload)
}

fn cookie_from(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(header::COOKIE)?
        .to_str()
        .ok()?
        .split(';')
        .find_map(|kv| {
            let (k, v) = kv.trim().split_once('=')?;
            (k == name).then(|| v.to_owned())
        })
}

fn set_cookie(name: &str, value: &str, max_age: i64) -> String {
    format!("{name}={value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age={max_age}")
}

/// The verified, allowlisted email from a valid session cookie, if any.
pub fn session_email(state: &AppState, headers: &HeaderMap) -> Option<String> {
    let raw = cookie_from(headers, SESSION_COOKIE)?;
    let payload = unsign(state.config.api_token.as_bytes(), &raw)?;
    let (email, exp) = payload.split_once('|')?;
    if exp.parse::<i64>().ok()? < now() {
        return None;
    }
    is_allowed(state, email).then(|| email.to_owned())
}

fn is_allowed(state: &AppState, email: &str) -> bool {
    let email = email.to_lowercase();
    state.config.allowed_emails.iter().any(|a| a == &email)
}

// -- handlers ---------------------------------------------------------------

pub async fn login(State(state): State<Arc<AppState>>) -> Response {
    let Some(client_id) = &state.config.google_client_id else {
        return (StatusCode::NOT_FOUND, "sign-in not configured").into_response();
    };
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let state_token = sign(
        state.config.api_token.as_bytes(),
        &format!("{nonce}|{}", now() + STATE_TTL_SECS),
    );
    let redirect_uri = format!(
        "{}/auth/callback",
        state.config.public_url.trim_end_matches('/')
    );
    let auth_url = format!(
        "{GOOGLE_AUTH_URL}?client_id={}&redirect_uri={}&response_type=code&scope=openid%20email\
         &state={nonce}&access_type=online&prompt=select_account",
        urlencode(client_id),
        urlencode(&redirect_uri),
    );
    (
        [(
            header::SET_COOKIE,
            set_cookie(STATE_COOKIE, &state_token, STATE_TTL_SECS),
        )],
        Redirect::to(&auth_url),
    )
        .into_response()
}

#[derive(Deserialize)]
pub struct Callback {
    code: Option<String>,
    state: Option<String>,
}

pub async fn callback(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<Callback>,
) -> Response {
    let (Some(client_id), Some(client_secret)) = (
        &state.config.google_client_id,
        &state.config.google_client_secret,
    ) else {
        return (StatusCode::NOT_FOUND, "sign-in not configured").into_response();
    };
    let Some(code) = params.code else {
        return (StatusCode::BAD_REQUEST, "missing code").into_response();
    };

    // CSRF: the state nonce must match our signed, unexpired state cookie.
    let ok_state = params
        .state
        .as_deref()
        .zip(
            cookie_from(&headers, STATE_COOKIE)
                .and_then(|c| unsign(state.config.api_token.as_bytes(), &c)),
        )
        .is_some_and(|(nonce, payload)| {
            payload
                .split_once('|')
                .is_some_and(|(n, exp)| n == nonce && exp.parse::<i64>().unwrap_or(0) >= now())
        });
    if !ok_state {
        return (StatusCode::BAD_REQUEST, "bad oauth state").into_response();
    }

    let redirect_uri = format!(
        "{}/auth/callback",
        state.config.public_url.trim_end_matches('/')
    );
    let email = match exchange_and_identify(client_id, client_secret, &code, &redirect_uri).await {
        Ok(email) => email,
        Err(error) => {
            tracing::warn!(%error, "oauth exchange failed");
            return (StatusCode::BAD_GATEWAY, "sign-in failed").into_response();
        }
    };
    if !is_allowed(&state, &email) {
        tracing::warn!(email, "sign-in denied (not on allowlist)");
        return (StatusCode::FORBIDDEN, format!("{email} is not permitted")).into_response();
    }

    let session = sign(
        state.config.api_token.as_bytes(),
        &format!("{email}|{}", now() + SESSION_TTL.as_secs() as i64),
    );
    (
        [(
            header::SET_COOKIE,
            set_cookie(SESSION_COOKIE, &session, SESSION_TTL.as_secs() as i64),
        )],
        Redirect::to("/"),
    )
        .into_response()
}

pub async fn logout() -> Response {
    (
        [(header::SET_COOKIE, set_cookie(SESSION_COOKIE, "", 0))],
        Redirect::to("/"),
    )
        .into_response()
}

/// Exchange the auth code and return the verified email from Google userinfo.
async fn exchange_and_identify(
    client_id: &str,
    client_secret: &str,
    code: &str,
    redirect_uri: &str,
) -> anyhow::Result<String> {
    #[derive(Deserialize)]
    struct Token {
        access_token: String,
    }
    #[derive(Deserialize)]
    struct UserInfo {
        email: String,
        #[serde(default)]
        email_verified: bool,
    }
    let http = reqwest::Client::new();
    let token: Token = http
        .post(GOOGLE_TOKEN_URL)
        .form(&[
            ("code", code),
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("redirect_uri", redirect_uri),
            ("grant_type", "authorization_code"),
        ])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let info: UserInfo = http
        .get(GOOGLE_USERINFO_URL)
        .bearer_auth(&token.access_token)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    anyhow::ensure!(info.email_verified, "email not verified by Google");
    Ok(info.email)
}

fn urlencode(s: &str) -> String {
    // Minimal percent-encoding for the URL query values we build.
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
