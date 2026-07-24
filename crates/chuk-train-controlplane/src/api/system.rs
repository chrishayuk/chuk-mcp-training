//! Health check + worker distribution (chuk-compute M2): per-target binary +
//! checksum, the version endpoint, and the one-shot installer. All public — the
//! worker is public code (spec §12) — so a rented box needs only the control
//! plane URL + a join token to bootstrap.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{ApiError, AGENT_SHA256_SUFFIX, SUPPORTED_TARGETS};
use sha2::{Digest, Sha256};

use crate::AppState;

/// Default directory of per-target worker binaries inside the deployed image.
const DEFAULT_AGENT_DIR: &str = "/app/agents";
/// Downloaded worker filename suggested to the client.
const WORKER_FILENAME: &str = "chuk-compute-worker";
/// The installer script, embedded so the control plane can serve it verbatim.
const INSTALL_SCRIPT: &str = include_str!("../../../../scripts/install.sh");

pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

/// The current worker version, for the persistent worker's self-updater (M3).
/// The worker is built from this same workspace, so its version tracks the CP's.
pub async fn agent_version() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// The one-shot installer (`/install.sh`).
pub async fn serve_install() -> Response {
    (
        [(header::CONTENT_TYPE, "text/x-shellscript; charset=utf-8")],
        INSTALL_SCRIPT,
    )
        .into_response()
}

/// Serve `/agent/<target>` (the binary) or `/agent/<target>.sha256` (its
/// checksum). `<target>` must be one of [`SUPPORTED_TARGETS`] — anything else
/// 404s, which also closes off path traversal (the served path is built from the
/// matched allowlist entry, never the raw request).
pub async fn serve_agent(State(state): State<Arc<AppState>>, Path(name): Path<String>) -> Response {
    let Some((target, wants_checksum)) = resolve_download(&name) else {
        return not_available("unknown worker target");
    };
    let dir = state
        .config
        .agent_dir
        .clone()
        .unwrap_or_else(|| DEFAULT_AGENT_DIR.to_owned());
    let bytes = match tokio::fs::read(format!("{dir}/{target}")).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, target, "worker binary not available for target");
            return not_available("worker binary not available for this target");
        }
    };
    if wants_checksum {
        return (
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            hex::encode(Sha256::digest(&bytes)),
        )
            .into_response();
    }
    (
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (
                header::CONTENT_DISPOSITION,
                &format!("attachment; filename=\"{WORKER_FILENAME}\""),
            ),
        ],
        bytes,
    )
        .into_response()
}

/// The sha256 hex of the served worker binary for `target`, or `None` if we
/// don't distribute that target or have no binary for it. Used to build the
/// self-update payload in the handshake (M3.3).
pub async fn binary_sha(agent_dir: Option<&str>, target: &str) -> Option<String> {
    if !SUPPORTED_TARGETS.contains(&target) {
        return None;
    }
    let dir = agent_dir.unwrap_or(DEFAULT_AGENT_DIR);
    let bytes = tokio::fs::read(format!("{dir}/{target}")).await.ok()?;
    Some(hex::encode(Sha256::digest(&bytes)))
}

/// Parse a download name into `(target, wants_checksum)`, or `None` if the
/// target is not one we distribute. Returns the `'static` allowlist entry so the
/// caller builds the filesystem path from trusted input.
fn resolve_download(name: &str) -> Option<(&'static str, bool)> {
    let (requested, wants_checksum) = match name.strip_suffix(AGENT_SHA256_SUFFIX) {
        Some(base) => (base, true),
        None => (name, false),
    };
    SUPPORTED_TARGETS
        .iter()
        .copied()
        .find(|t| *t == requested)
        .map(|t| (t, wants_checksum))
}

fn not_available(message: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ApiError {
            error: message.to_owned(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::SqliteStore;
    use crate::AppState;

    fn base_config(agent_dir: Option<String>) -> Config {
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
            agent_dir,
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

    /// A real (if minimal) `AppState`: `serve_agent` takes `State<Arc<AppState>>`
    /// directly, so there is no lighter seam than building one — mirrors
    /// `dash.rs`'s `test_state` helper.
    async fn test_state(agent_dir: Option<String>) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let config = base_config(agent_dir);
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

    /// A fresh, empty directory under the system temp dir, unique per call so
    /// parallel tests never collide.
    fn unique_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("chuk-system-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).expect("create temp agent dir");
        dir
    }

    #[test]
    fn resolves_known_targets_and_checksum_suffix() {
        assert_eq!(
            resolve_download("aarch64-apple-darwin"),
            Some(("aarch64-apple-darwin", false))
        );
        assert_eq!(
            resolve_download("x86_64-unknown-linux-musl.sha256"),
            Some(("x86_64-unknown-linux-musl", true))
        );
    }

    #[test]
    fn rejects_unknown_targets_and_traversal() {
        assert_eq!(resolve_download("version"), None);
        assert_eq!(resolve_download("linux-x86_64"), None); // the retired M1 name
        assert_eq!(resolve_download("../../etc/passwd"), None);
        assert_eq!(resolve_download("x86_64-unknown-linux-gnu"), None); // gnu not distributed
        assert_eq!(resolve_download(".sha256"), None); // empty target
    }

    #[tokio::test]
    async fn version_reports_the_build_version() {
        let Json(body) = agent_version().await;
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn install_script_is_served() {
        let resp = serve_install().await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(INSTALL_SCRIPT.starts_with("#!/bin/sh"));
    }

    #[tokio::test]
    async fn healthz_is_ok() {
        let Json(body) = healthz().await;
        assert_eq!(body["ok"], true);
    }

    #[tokio::test]
    async fn serve_agent_404s_for_a_target_not_in_the_allowlist() {
        let state = test_state(None).await;
        let resp = serve_agent(State(state), Path("bogus-target".to_owned())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let error: ApiError = serde_json::from_slice(&body).expect("json body");
        assert_eq!(error.error, "unknown worker target");
    }

    #[tokio::test]
    async fn serve_agent_falls_back_to_the_default_agent_dir_when_unconfigured() {
        // No agent_dir configured: exercises the `unwrap_or_else(DEFAULT_AGENT_DIR)`
        // fallback. Nothing lives there in the test sandbox, so this still 404s,
        // but via the default-dir path rather than a configured one.
        let state = test_state(None).await;
        let resp = serve_agent(State(state), Path(SUPPORTED_TARGETS[0].to_owned())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn serve_agent_404s_when_the_binary_is_missing_on_disk() {
        let dir = unique_dir();
        let state = test_state(Some(dir.to_string_lossy().into_owned())).await;
        let target = SUPPORTED_TARGETS[0];
        let resp = serve_agent(State(state), Path(target.to_owned())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        let error: ApiError = serde_json::from_slice(&body).expect("json body");
        assert_eq!(error.error, "worker binary not available for this target");
    }

    #[tokio::test]
    async fn serve_agent_serves_the_checksum_for_a_present_binary() {
        let dir = unique_dir();
        let target = SUPPORTED_TARGETS[0];
        let contents = b"fake worker binary bytes";
        std::fs::write(dir.join(target), contents).expect("write fake binary");
        let state = test_state(Some(dir.to_string_lossy().into_owned())).await;

        let resp = serve_agent(State(state), Path(format!("{target}{AGENT_SHA256_SUFFIX}"))).await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain; charset=utf-8"
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        assert_eq!(
            std::str::from_utf8(&body).unwrap(),
            hex::encode(Sha256::digest(contents))
        );
    }

    #[tokio::test]
    async fn serve_agent_serves_the_binary_with_download_headers() {
        let dir = unique_dir();
        let target = SUPPORTED_TARGETS[0];
        let contents = b"fake worker binary bytes";
        std::fs::write(dir.join(target), contents).expect("write fake binary");
        let state = test_state(Some(dir.to_string_lossy().into_owned())).await;

        let resp = serve_agent(State(state), Path(target.to_owned())).await;

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/octet-stream"
        );
        assert_eq!(
            resp.headers().get(header::CONTENT_DISPOSITION).unwrap(),
            &format!("attachment; filename=\"{WORKER_FILENAME}\"")
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        assert_eq!(&body[..], contents);
    }

    #[tokio::test]
    async fn binary_sha_is_none_for_a_target_not_in_the_allowlist() {
        let dir = unique_dir();
        assert_eq!(binary_sha(Some(dir.to_str().unwrap()), "bogus-target").await, None);
    }

    #[tokio::test]
    async fn binary_sha_is_none_when_the_binary_is_missing_on_disk() {
        let dir = unique_dir();
        assert_eq!(binary_sha(Some(dir.to_str().unwrap()), SUPPORTED_TARGETS[0]).await, None);
    }

    #[tokio::test]
    async fn binary_sha_hashes_the_binary_when_present() {
        let dir = unique_dir();
        let target = SUPPORTED_TARGETS[1];
        let contents = b"another fake worker binary";
        std::fs::write(dir.join(target), contents).expect("write fake binary");

        let sha = binary_sha(Some(dir.to_str().unwrap()), target).await;

        assert_eq!(sha, Some(hex::encode(Sha256::digest(contents))));
    }

    #[tokio::test]
    async fn binary_sha_uses_the_default_dir_when_none_is_configured() {
        // No agent dir configured and nothing at the default image path in the
        // test sandbox, so this exercises the `unwrap_or(DEFAULT_AGENT_DIR)`
        // fallback and still resolves to `None`.
        assert_eq!(binary_sha(None, SUPPORTED_TARGETS[0]).await, None);
    }
}
