//! Checkpoints and artifact retrieval.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    ApiError, CheckpointInfo, PinCheckpointRequest, Role, RunId, SignedUrl, DEFAULT_ARTIFACT_URL_TTL,
};
use serde::Deserialize;

use super::{bad_request, internal, not_found, now, require_role, ERR_CKPT_NOT_FOUND};
use crate::apikey::AuthContext;
use crate::AppState;

pub async fn list_checkpoints(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.checkpoints(&RunId(run_id)).await {
        Ok(ckpts) => Json::<Vec<CheckpointInfo>>(ckpts).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn pin_checkpoint(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Json(request): Json<PinCheckpointRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    match state
        .hub
        .store
        .pin_checkpoint(&RunId(run_id), request.step, &request.name)
        .await
    {
        Ok(true) => Json(serde_json::json!({ "ok": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: ERR_CKPT_NOT_FOUND.into(),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}

/// Stable, location-resolving fetch for one checkpoint file (spec §11.5). The
/// durable handle handed to lazarus / experiment servers: it redirects to a
/// presigned R2 URL while the object is on R2 (hot or promoted-final), or
/// streams it from Drive once archived — the bytes move underneath the URL.
pub async fn serve_checkpoint(
    State(state): State<Arc<AppState>>,
    Path((run_id, step, file)): Path<(String, u64, String)>,
) -> Response {
    use chuk_train_proto::{keys, CheckpointLocation};
    if file.contains('/') || !keys::is_safe_key(&file) {
        return bad_request("invalid checkpoint file");
    }
    let rid = RunId(run_id.clone());
    let ckpts = match state.hub.store.checkpoints(&rid).await {
        Ok(ckpts) => ckpts,
        Err(error) => return internal(error),
    };
    let Some(ckpt) = ckpts.into_iter().find(|c| c.step == step) else {
        return not_found();
    };
    match ckpt.location {
        CheckpointLocation::Drive => {
            let Some(drive) = state.drive.clone() else {
                return internal(anyhow::anyhow!("checkpoint on drive but drive not configured"));
            };
            let ids = match state.hub.store.checkpoint_drive_ids(&rid, step).await {
                Ok(ids) => ids.unwrap_or_default(),
                Err(error) => return internal(error),
            };
            let Some(file_id) = ids.get(&file) else {
                return not_found();
            };
            match drive.download(file_id).await {
                Ok(bytes) => bytes.into_response(),
                Err(error) => internal(error),
            }
        }
        location => {
            let key = if location == CheckpointLocation::R2Final {
                keys::checkpoint_final_file(&run_id, step, &file)
            } else {
                keys::checkpoint_file(&run_id, step, &file)
            };
            match state.artifacts.presign_get(&key, DEFAULT_ARTIFACT_URL_TTL).await {
                Ok(Some(signed)) => axum::response::Redirect::temporary(&signed.url).into_response(),
                Ok(None) => match state.artifacts.get(&key).await {
                    Ok(bytes) => bytes.into_response(),
                    Err(_) => not_found(),
                },
                Err(error) => internal(error),
            }
        }
    }
}

#[derive(Deserialize)]
pub struct ArtifactUrlParams {
    key: String,
    ttl_s: Option<u64>,
}

/// Time-limited fetch URL for an artifact key (spec §6 artifact_url). The
/// filesystem backend has no native signed URL, so this points at the control
/// plane's own authenticated `/api/blob` endpoint.
pub async fn artifact_url(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ArtifactUrlParams>,
) -> Response {
    let ttl =
        std::time::Duration::from_secs(params.ttl_s.unwrap_or(DEFAULT_ARTIFACT_URL_TTL.as_secs()));
    let native = match state.artifacts.presign_get(&params.key, ttl).await {
        Ok(url) => url,
        Err(error) => return bad_request(&error.to_string()),
    };
    let signed = native.unwrap_or_else(|| SignedUrl {
        url: format!(
            "{}/api/blob/{}",
            state.config.public_url.trim_end_matches('/'),
            params.key
        ),
        expires_at: now() + ttl.as_secs_f64(),
    });
    Json(signed).into_response()
}

/// Serve artifact bytes (bearer-authed). Used by artifact_url consumers such as
/// lazarus pulling checkpoints to the Mac.
pub async fn blob(State(state): State<Arc<AppState>>, Path(key): Path<String>) -> Response {
    match state.artifacts.get(&key).await {
        Ok(bytes) => bytes.into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no such artifact".into(),
            }),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use chuk_train_proto::{keys, CheckpointMeta, RunSpec, ShellSpec};

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::config::Config;
    use crate::lease::LeaseManager;
    use crate::provider::build_providers;
    use crate::store::SqliteStore;
    use crate::AppState;

    fn test_ctx(role: Role) -> AuthContext {
        AuthContext {
            role,
            team_id: "default".into(),
            subject: "tester".into(),
            owner_email: "tester@example.com".into(),
        }
    }

    fn shell_run() -> RunSpec {
        RunSpec::Shell(ShellSpec { command: "true".into(), timeout_s: 60 })
    }

    /// A real (if minimal) `AppState`, matching `dash.rs`'s/`archive.rs`'s
    /// pattern — these handlers take `State<Arc<AppState>>` directly, so
    /// there's no lighter seam. Each call gets its own artifact-store root (a
    /// fresh temp dir) since — unlike `dash.rs`'s state, which never touches
    /// the artifact store — `serve_checkpoint`/`blob` here read and write real
    /// bytes and must not collide across tests.
    async fn test_state() -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let root = std::env::temp_dir().join(format!("chuk-checkpoints-test-{}", uuid::Uuid::new_v4()));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> = Arc::new(FsArtifactStore::new(root));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", None, None));
        let config = Config {
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

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    async fn body_bytes(resp: Response) -> Vec<u8> {
        to_bytes(resp.into_body(), usize::MAX).await.expect("read body").to_vec()
    }

    // ---- list_checkpoints --------------------------------------------------

    #[tokio::test]
    async fn list_checkpoints_returns_recorded_checkpoints_in_step_order() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 2, "ckpt-hot/r/step_2", "hash2", &CheckpointMeta::default())
            .await
            .unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 1, "ckpt-hot/r/step_1", "hash1", &CheckpointMeta::default())
            .await
            .unwrap();

        let resp = list_checkpoints(State(state), Path(run.0.clone())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 2);
        // Stored ordered by step ascending, regardless of recording order.
        assert_eq!(rows[0]["step"], 1);
        assert_eq!(rows[1]["step"], 2);
    }

    #[tokio::test]
    async fn list_checkpoints_is_empty_for_an_unknown_run() {
        let state = test_state().await;
        let resp = list_checkpoints(State(state), Path("RUN-does-not-exist".into())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await, serde_json::json!([]));
    }

    // ---- pin_checkpoint -----------------------------------------------------

    #[tokio::test]
    async fn pin_checkpoint_refuses_below_write_role() {
        let state = test_state().await;
        let resp = pin_checkpoint(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Path("RUN-1".into()),
            Json(PinCheckpointRequest { step: 1, name: "best".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn pin_checkpoint_pins_an_existing_checkpoint() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 3, "ckpt-hot/r/step_3", "hash3", &CheckpointMeta::default())
            .await
            .unwrap();

        let resp = pin_checkpoint(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
            Json(PinCheckpointRequest { step: 3, name: "best".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await, serde_json::json!({ "ok": true }));

        let ckpts = state.hub.store.checkpoints(&run).await.unwrap();
        assert!(ckpts[0].pinned);
        assert_eq!(ckpts[0].pin_name.as_deref(), Some("best"));
    }

    #[tokio::test]
    async fn pin_checkpoint_404s_for_an_unknown_checkpoint() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        let resp = pin_checkpoint(
            State(state),
            axum::Extension(test_ctx(Role::Admin)),
            Path(run.0.clone()),
            Json(PinCheckpointRequest { step: 99, name: "nope".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"], ERR_CKPT_NOT_FOUND);
    }

    // ---- serve_checkpoint ----------------------------------------------------

    #[tokio::test]
    async fn serve_checkpoint_rejects_an_unsafe_file_name() {
        let state = test_state().await;
        let resp = serve_checkpoint(State(state), Path(("RUN-1".into(), 1, "..".into()))).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn serve_checkpoint_rejects_a_nested_file_name() {
        let state = test_state().await;
        let resp = serve_checkpoint(State(state), Path(("RUN-1".into(), 1, "a/b".into()))).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn serve_checkpoint_404s_when_the_step_is_not_recorded() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        let resp = serve_checkpoint(State(state), Path((run.0.clone(), 1, "model.bin".into()))).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn serve_checkpoint_404s_when_the_recorded_bytes_are_missing() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        // Metadata recorded, but the file was never actually uploaded to the
        // artifact store — the hot-path fallback (no presign) must 404, not panic.
        state
            .hub
            .store
            .record_checkpoint(&run, 1, "ckpt-hot/r/step_1", "hash1", &CheckpointMeta::default())
            .await
            .unwrap();
        let resp = serve_checkpoint(State(state), Path((run.0.clone(), 1, "model.bin".into()))).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn serve_checkpoint_serves_hot_bytes_from_the_artifact_store() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 1, "ckpt-hot/r/step_1", "hash1", &CheckpointMeta::default())
            .await
            .unwrap();
        state
            .artifacts
            .put(&keys::checkpoint_file(&run.0, 1, "model.bin"), b"weights-v1".to_vec())
            .await
            .unwrap();

        let resp = serve_checkpoint(State(state), Path((run.0.clone(), 1, "model.bin".into()))).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_bytes(resp).await, b"weights-v1");
    }

    #[tokio::test]
    async fn serve_checkpoint_serves_promoted_final_bytes_from_the_final_prefix() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 1, "ckpt-hot/r/step_1", "hash1", &CheckpointMeta::default())
            .await
            .unwrap();
        state
            .hub
            .store
            .set_checkpoint_location(&run, 1, chuk_train_proto::CheckpointLocation::R2Final)
            .await
            .unwrap();
        // Only the *final* key holds bytes — proves the handler reads from
        // ckpt-final/, not ckpt-hot/, once promoted.
        state
            .artifacts
            .put(&keys::checkpoint_final_file(&run.0, 1, "model.bin"), b"weights-final".to_vec())
            .await
            .unwrap();

        let resp = serve_checkpoint(State(state), Path((run.0.clone(), 1, "model.bin".into()))).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_bytes(resp).await, b"weights-final");
    }

    #[tokio::test]
    async fn serve_checkpoint_errors_when_marked_on_drive_but_drive_is_not_configured() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&run, 1, "ckpt-hot/r/step_1", "hash1", &CheckpointMeta::default())
            .await
            .unwrap();
        state
            .hub
            .store
            .set_checkpoint_location(&run, 1, chuk_train_proto::CheckpointLocation::Drive)
            .await
            .unwrap();

        let resp = serve_checkpoint(State(state), Path((run.0.clone(), 1, "model.bin".into()))).await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ---- artifact_url ---------------------------------------------------------

    #[tokio::test]
    async fn artifact_url_falls_back_to_the_blob_endpoint_for_a_backend_with_no_native_signing() {
        let state = test_state().await;
        let before = super::now();
        let resp = artifact_url(
            State(state),
            Query(ArtifactUrlParams { key: "some/key".into(), ttl_s: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["url"], "http://127.0.0.1:9/api/blob/some/key");
        let expires_at = body["expires_at"].as_f64().expect("expires_at");
        // Default TTL is 3600s; allow generous slack for test wall-clock.
        assert!(expires_at >= before + 3500.0 && expires_at <= before + 3700.0);
    }

    #[tokio::test]
    async fn artifact_url_honors_a_custom_ttl() {
        let state = test_state().await;
        let before = super::now();
        let resp = artifact_url(
            State(state),
            Query(ArtifactUrlParams { key: "k".into(), ttl_s: Some(60) }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let expires_at = body["expires_at"].as_f64().expect("expires_at");
        assert!(expires_at >= before + 30.0 && expires_at <= before + 120.0);
    }

    // ---- blob -------------------------------------------------------------

    #[tokio::test]
    async fn blob_serves_bytes_that_were_put() {
        let state = test_state().await;
        state.artifacts.put("some/key", b"blob-bytes".to_vec()).await.unwrap();
        let resp = blob(State(state), Path("some/key".into())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_bytes(resp).await, b"blob-bytes");
    }

    #[tokio::test]
    async fn blob_404s_for_a_missing_key() {
        let state = test_state().await;
        let resp = blob(State(state), Path("no/such/key".into())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "no such artifact");
    }
}
