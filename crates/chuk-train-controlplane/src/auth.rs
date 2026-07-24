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
    complete(&state, &headers, params, &Google::live()).await
}

/// [`callback`]'s body with the Google endpoints supplied rather than fixed, so
/// the whole flow — CSRF check, code exchange, allowlist, session cookie — is
/// exercisable against a loopback server. Production always passes
/// [`Google::live`].
async fn complete(
    state: &AppState,
    headers: &HeaderMap,
    params: Callback,
    google: &Google,
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
            cookie_from(headers, STATE_COOKIE)
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
    let email = match google
        .exchange_and_identify(client_id, client_secret, &code, &redirect_uri)
        .await
    {
        Ok(email) => email,
        Err(error) => {
            tracing::warn!(%error, "oauth exchange failed");
            return (StatusCode::BAD_GATEWAY, "sign-in failed").into_response();
        }
    };
    if !is_allowed(state, &email) {
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

/// The two Google endpoints the code exchange calls. Held as values rather than
/// used as constants directly so tests can point them at a loopback server.
struct Google {
    token_url: String,
    userinfo_url: String,
}

impl Google {
    fn live() -> Self {
        Self {
            token_url: GOOGLE_TOKEN_URL.to_owned(),
            userinfo_url: GOOGLE_USERINFO_URL.to_owned(),
        }
    }

    /// Exchange the auth code and return the verified email from Google userinfo.
    async fn exchange_and_identify(
        &self,
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
            .post(&self.token_url)
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
            .get(&self.userinfo_url)
            .bearer_auth(&token.access_token)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        anyhow::ensure!(info.email_verified, "email not verified by Google");
        Ok(info.email)
    }
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

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use axum::http::HeaderValue;

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::fakehttp::{FakeHttp, Reply, REFUSED_ORIGIN};
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::SqliteStore;

    const API_TOKEN: &str = "test-api-token";
    const ALLOWED: &str = "ops@example.com";

    /// A real (if minimal) `AppState` — these handlers take
    /// `State<Arc<AppState>>` directly, so there is no lighter seam (same
    /// pattern as `dash.rs`'s `test_state`).
    async fn test_state(signin_configured: bool) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let config = Config {
            api_token: API_TOKEN.into(),
            join_token: "test-join-token".into(),
            store_spec: ":memory:".into(),
            artifacts_spec: "file:./unused".into(),
            // A trailing slash the redirect_uri builder must trim.
            public_url: "https://cp.example.com/".into(),
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
            reconcile_interval: std::time::Duration::from_secs(30),
            idle_reap: std::time::Duration::from_secs(60),
            google_client_id: signin_configured.then(|| "client id/with spaces".to_owned()),
            google_client_secret: signin_configured.then(|| "client-secret".to_owned()),
            allowed_emails: vec![ALLOWED.into()],
            sysadmin_email: None,
        };
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

    fn cookies(raw: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_str(raw).expect("cookie value"));
        headers
    }

    fn set_cookie_of(response: &Response) -> &str {
        response
            .headers()
            .get(header::SET_COOKIE)
            .expect("Set-Cookie")
            .to_str()
            .expect("ascii cookie")
    }

    fn location_of(response: &Response) -> &str {
        response
            .headers()
            .get(header::LOCATION)
            .expect("Location")
            .to_str()
            .expect("ascii location")
    }

    // -- signing ------------------------------------------------------------

    #[test]
    fn sign_round_trips_and_only_the_signing_key_opens_it() {
        let token = sign(b"key", "ops@example.com|123");
        assert_eq!(unsign(b"key", &token).as_deref(), Some("ops@example.com|123"));
        assert_eq!(unsign(b"other-key", &token), None, "a different key must not verify");
    }

    #[test]
    fn unsign_rejects_every_shape_of_bad_token() {
        let token = sign(b"key", "ops@example.com|123");
        let (b64, sig) = token.split_once('.').expect("signed shape");
        assert_eq!(unsign(b"key", b64), None, "no signature at all");
        assert_eq!(unsign(b"key", &format!("{b64}.{}", "0".repeat(sig.len()))), None, "wrong signature");
        assert_eq!(unsign(b"key", &format!("{b64}.zz")), None, "signature isn't hex");
        assert_eq!(unsign(b"key", "!!!.abcd"), None, "payload isn't base64url");
        // Valid base64url, but not valid UTF-8 once decoded.
        let raw = URL_SAFE_NO_PAD.encode([0xff, 0xfe]);
        assert_eq!(unsign(b"key", &sign_raw(b"key", &raw)), None, "payload isn't utf-8");
    }

    /// Sign an already-encoded payload, to build the not-UTF-8 case above.
    fn sign_raw(key: &[u8], b64: &str) -> String {
        let payload = URL_SAFE_NO_PAD.decode(b64).expect("b64");
        format!("{b64}.{}", hmac_hex(key, &payload))
    }

    #[test]
    fn set_cookie_is_httponly_secure_and_lax() {
        let cookie = set_cookie(SESSION_COOKIE, "value", 60);
        assert_eq!(
            cookie,
            "chuk_session=value; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=60"
        );
    }

    #[test]
    fn cookie_from_finds_its_own_cookie_among_others() {
        let headers = cookies("other=1; chuk_session=abc; last=2");
        assert_eq!(cookie_from(&headers, SESSION_COOKIE).as_deref(), Some("abc"));
        assert_eq!(cookie_from(&headers, STATE_COOKIE), None, "absent cookie");
        assert_eq!(cookie_from(&HeaderMap::new(), SESSION_COOKIE), None, "no Cookie header");
        // A malformed pair (no `=`) must not panic or match.
        assert_eq!(cookie_from(&cookies("garbage"), SESSION_COOKIE), None);
    }

    // -- session_email ------------------------------------------------------

    fn session_cookie(email: &str, expires_in: i64) -> HeaderMap {
        let token = sign(API_TOKEN.as_bytes(), &format!("{email}|{}", now() + expires_in));
        cookies(&format!("{SESSION_COOKIE}={token}"))
    }

    #[tokio::test]
    async fn session_email_accepts_a_live_signed_allowlisted_cookie() {
        let state = test_state(true).await;
        assert_eq!(
            session_email(&state, &session_cookie(ALLOWED, 3600)),
            Some(ALLOWED.to_owned())
        );
    }

    #[tokio::test]
    async fn session_email_rejects_expired_unsigned_unlisted_and_absent_cookies() {
        let state = test_state(true).await;
        assert_eq!(session_email(&state, &session_cookie(ALLOWED, -1)), None, "expired");
        assert_eq!(
            session_email(&state, &session_cookie("stranger@example.com", 3600)),
            None,
            "not on the allowlist"
        );
        assert_eq!(session_email(&state, &HeaderMap::new()), None, "no cookie");
        // Right shape, wrong key: a forged cookie must not authenticate.
        let forged = sign(b"not-the-api-token", &format!("{ALLOWED}|{}", now() + 3600));
        assert_eq!(
            session_email(&state, &cookies(&format!("{SESSION_COOKIE}={forged}"))),
            None,
            "forged signature"
        );
        // Signed by us, but the payload isn't `email|expiry`.
        let shapeless = sign(API_TOKEN.as_bytes(), "no-separator");
        assert_eq!(
            session_email(&state, &cookies(&format!("{SESSION_COOKIE}={shapeless}"))),
            None,
            "payload has no expiry"
        );
        // Signed by us, but the expiry isn't a number.
        let unparseable = sign(API_TOKEN.as_bytes(), &format!("{ALLOWED}|soon"));
        assert_eq!(
            session_email(&state, &cookies(&format!("{SESSION_COOKIE}={unparseable}"))),
            None,
            "expiry is not an integer"
        );
    }

    #[tokio::test]
    async fn the_allowlist_ignores_case() {
        let state = test_state(true).await;
        assert!(is_allowed(&state, "OPS@Example.com"));
        assert!(!is_allowed(&state, "someone@example.com"));
    }

    #[test]
    fn urlencode_keeps_the_unreserved_set_and_escapes_the_rest() {
        assert_eq!(urlencode("aZ0-_.~"), "aZ0-_.~");
        assert_eq!(urlencode("a b/c?d"), "a%20b%2Fc%3Fd");
    }

    // -- /auth/login --------------------------------------------------------

    #[tokio::test]
    async fn login_is_not_found_when_no_google_client_is_configured() {
        let state = test_state(false).await;
        let response = login(State(state)).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn login_redirects_to_google_and_sets_a_signed_state_cookie() {
        let state = test_state(true).await;
        let response = login(State(state.clone())).await;

        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        let location = location_of(&response);
        assert!(location.starts_with(GOOGLE_AUTH_URL), "unexpected location: {location}");
        // Both values are percent-encoded, and the redirect_uri's trailing
        // slash is trimmed before `/auth/callback` is appended.
        assert!(location.contains("client_id=client%20id%2Fwith%20spaces"), "{location}");
        assert!(
            location.contains("redirect_uri=https%3A%2F%2Fcp.example.com%2Fauth%2Fcallback"),
            "{location}"
        );

        let cookie = set_cookie_of(&response);
        let token = cookie
            .strip_prefix(&format!("{STATE_COOKIE}="))
            .and_then(|rest| rest.split(';').next())
            .expect("state cookie");
        let payload = unsign(API_TOKEN.as_bytes(), token).expect("state cookie is signed by us");
        let (nonce, expiry) = payload.split_once('|').expect("nonce|expiry");
        assert!(location.contains(&format!("state={nonce}")), "{location}");
        assert!(expiry.parse::<i64>().expect("numeric expiry") > now());
    }

    // -- /auth/callback -----------------------------------------------------

    /// The state cookie + matching `state` query param for a callback that
    /// should pass the CSRF check.
    fn valid_state(expires_in: i64) -> (HeaderMap, String) {
        let nonce = "nonce-123";
        let token = sign(API_TOKEN.as_bytes(), &format!("{nonce}|{}", now() + expires_in));
        (cookies(&format!("{STATE_COOKIE}={token}")), nonce.to_owned())
    }

    fn params(code: Option<&str>, state: Option<String>) -> Callback {
        Callback { code: code.map(str::to_owned), state }
    }

    /// A Google that hands back `email`, verified unless stated otherwise.
    fn fake_google(email: &'static str, verified: bool) -> (FakeHttp, Google) {
        let server = FakeHttp::start(move |req, _| {
            if req.path() == "/token" {
                Reply::ok(r#"{"access_token":"ya29.token"}"#)
            } else {
                Reply::ok(format!(
                    r#"{{"email":"{email}","email_verified":{verified}}}"#
                ))
            }
        });
        let google = Google {
            token_url: format!("{}/token", server.origin),
            userinfo_url: format!("{}/userinfo", server.origin),
        };
        (server, google)
    }

    #[tokio::test]
    async fn callback_is_not_found_when_no_google_client_is_configured() {
        let state = test_state(false).await;
        let (headers, nonce) = valid_state(600);
        let response = complete(&state, &headers, params(Some("code"), Some(nonce)), &Google::live()).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn callback_rejects_a_missing_code_before_any_exchange() {
        let state = test_state(true).await;
        let (headers, nonce) = valid_state(600);
        let response = complete(&state, &headers, params(None, Some(nonce)), &Google::live()).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn callback_rejects_every_failure_of_the_csrf_state_check() {
        let state = test_state(true).await;
        let (headers, nonce) = valid_state(600);
        let google = Google::live();

        // No state param at all.
        let response = complete(&state, &headers, params(Some("c"), None), &google).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // A nonce that doesn't match the cookie's.
        let response = complete(&state, &headers, params(Some("c"), Some("other".into())), &google).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // No state cookie.
        let response = complete(&state, &HeaderMap::new(), params(Some("c"), Some(nonce.clone())), &google).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // An expired state cookie.
        let (stale, stale_nonce) = valid_state(-1);
        let response = complete(&state, &stale, params(Some("c"), Some(stale_nonce)), &google).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn callback_is_a_bad_gateway_when_the_exchange_fails() {
        let state = test_state(true).await;
        let (headers, nonce) = valid_state(600);
        let unreachable = Google {
            token_url: format!("{REFUSED_ORIGIN}/token"),
            userinfo_url: format!("{REFUSED_ORIGIN}/userinfo"),
        };
        let response = complete(&state, &headers, params(Some("code"), Some(nonce)), &unreachable).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn callback_forbids_an_email_that_is_not_on_the_allowlist() {
        let state = test_state(true).await;
        let (headers, nonce) = valid_state(600);
        let (_server, google) = fake_google("stranger@example.com", true);

        let response = complete(&state, &headers, params(Some("code"), Some(nonce)), &google).await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(response.headers().get(header::SET_COOKIE).is_none(), "no session for a stranger");
    }

    #[tokio::test]
    async fn callback_sets_a_session_cookie_the_dashboard_accepts_and_sends_the_user_home() {
        let state = test_state(true).await;
        let (headers, nonce) = valid_state(600);
        let (server, google) = fake_google(ALLOWED, true);

        let response = complete(&state, &headers, params(Some("the-code"), Some(nonce)), &google).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(location_of(&response), "/");

        // The exchange posted the code + our credentials, then presented the
        // access token as a bearer to userinfo.
        let exchange = server.requests();
        assert_eq!(exchange.len(), 2);
        let form = String::from_utf8(exchange[0].body.clone()).expect("form body");
        assert!(form.contains("code=the-code"), "unexpected form: {form}");
        assert!(form.contains("grant_type=authorization_code"), "unexpected form: {form}");
        assert!(
            form.contains("redirect_uri=https%3A%2F%2Fcp.example.com%2Fauth%2Fcallback"),
            "unexpected form: {form}"
        );
        assert_eq!(exchange[1].header("authorization"), "Bearer ya29.token");

        // The cookie it sets is exactly what `session_email` accepts.
        let cookie = set_cookie_of(&response);
        let token = cookie
            .strip_prefix(&format!("{SESSION_COOKIE}="))
            .and_then(|rest| rest.split(';').next())
            .expect("session cookie");
        assert_eq!(
            session_email(&state, &cookies(&format!("{SESSION_COOKIE}={token}"))),
            Some(ALLOWED.to_owned())
        );
    }

    #[tokio::test]
    async fn an_unverified_google_email_is_never_admitted() {
        let (_server, google) = fake_google(ALLOWED, false);
        let error = google
            .exchange_and_identify("id", "secret", "code", "https://cp.example.com/auth/callback")
            .await
            .expect_err("an unverified email must not sign in");
        assert!(error.to_string().contains("not verified"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn an_error_from_google_fails_the_exchange() {
        let server = FakeHttp::start(|_, _| Reply::new(401, "invalid_grant"));
        let google = Google {
            token_url: format!("{}/token", server.origin),
            userinfo_url: format!("{}/userinfo", server.origin),
        };
        let error = google
            .exchange_and_identify("id", "secret", "code", "https://cp.example.com/auth/callback")
            .await
            .expect_err("a 401 from the token endpoint must fail");
        assert!(error.to_string().contains("401"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn logout_clears_the_session_cookie_immediately() {
        let response = logout().await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert_eq!(location_of(&response), "/");
        assert!(set_cookie_of(&response).starts_with(&format!("{SESSION_COOKIE}=;")));
        assert!(set_cookie_of(&response).contains("Max-Age=0"));
    }

    #[tokio::test]
    async fn the_live_endpoints_are_googles() {
        let google = Google::live();
        assert_eq!(google.token_url, GOOGLE_TOKEN_URL);
        assert_eq!(google.userinfo_url, GOOGLE_USERINFO_URL);
    }
}
