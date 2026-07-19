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
    use super::*;

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
}
