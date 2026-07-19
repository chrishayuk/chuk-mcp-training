//! Health check and the worker agent binary.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::ApiError;

use crate::AppState;

pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

/// Default path the agent binary lives at inside the deployed image.
const DEFAULT_AGENT_BIN: &str = "/app/chuk-train-agent";

/// Serve the worker agent binary (unauthenticated: it is public code, spec
/// §12). The Colab/Vast bootstrap downloads it from here, so a worker needs
/// only the control-plane URL + join token — nothing else to host.
pub async fn serve_agent(State(state): State<Arc<AppState>>) -> Response {
    let path = state
        .config
        .agent_bin
        .clone()
        .unwrap_or_else(|| DEFAULT_AGENT_BIN.to_owned());
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"chuk-train-agent\"",
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(error) => {
            tracing::error!(%error, path, "agent binary not available to serve");
            (
                StatusCode::NOT_FOUND,
                Json(ApiError {
                    error: "agent binary not available".into(),
                }),
            )
                .into_response()
        }
    }
}
