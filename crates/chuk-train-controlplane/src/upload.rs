//! Worker-facing blob transfer, authorised by run-scoped grants (spec §12),
//! not the API token. Agents upload checkpoints here and fetch their code unit
//! and resume checkpoints — each request checked against the grant's run.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_compute_wire::{BlobMethod, BlobUrlRequest, BlobUrlResponse, API_PREFIX};
use chuk_train_proto::{ApiError, DEFAULT_ARTIFACT_URL_TTL};

use crate::grant::Grant;
use crate::AppState;

const BEARER_PREFIX: &str = "Bearer ";

/// `PUT /api/upload/<key>` — write a blob into the grant's run tree.
pub async fn upload(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(key): Path<String>,
    body: Bytes,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    if !grant.may_write(&key) {
        return forbidden();
    }
    match state.artifacts.put(&key, body.to_vec()).await {
        Ok(_) => (StatusCode::CREATED, Json(serde_json::json!({ "ok": true }))).into_response(),
        Err(error) => {
            tracing::error!(%error, key, "upload failed");
            internal()
        }
    }
}

/// `GET /api/fetch/<key>` — read a blob the grant is allowed to see.
pub async fn fetch(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(key): Path<String>,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    if !grant.may_read(&key) {
        return forbidden();
    }
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

/// `POST /api/blob_url` — where should the worker transfer this blob? With an
/// S3/R2 backend this is a presigned URL the worker hits directly; with the
/// filesystem backend it points back at this control plane's upload/fetch.
pub async fn blob_url(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<BlobUrlRequest>,
) -> Response {
    let Some(grant) = authorize(&state, &headers) else {
        return unauthorized();
    };
    let allowed = match req.method {
        BlobMethod::Put => grant.may_write(&req.key),
        BlobMethod::Get => grant.may_read(&req.key),
    };
    if !allowed {
        return forbidden();
    }

    let presigned = match req.method {
        BlobMethod::Put => {
            state
                .artifacts
                .presign_put(&req.key, DEFAULT_ARTIFACT_URL_TTL)
                .await
        }
        BlobMethod::Get => {
            state
                .artifacts
                .presign_get(&req.key, DEFAULT_ARTIFACT_URL_TTL)
                .await
        }
    };
    let response = match presigned {
        Ok(Some(signed)) => BlobUrlResponse {
            url: signed.url,
            requires_grant_header: false,
        },
        Ok(None) => {
            // Filesystem backend: fall back to transferring through the control
            // plane, which needs the grant token on the request.
            let action = match req.method {
                BlobMethod::Put => "upload",
                BlobMethod::Get => "fetch",
            };
            let base = state.config.public_url.trim_end_matches('/');
            BlobUrlResponse {
                url: format!("{base}{API_PREFIX}/{action}/{}", req.key),
                requires_grant_header: true,
            }
        }
        Err(error) => {
            tracing::error!(%error, key = req.key, "presign failed");
            return internal();
        }
    };
    Json(response).into_response()
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Option<Grant> {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix(BEARER_PREFIX))?;
    state.hub.grants().resolve(token)
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ApiError {
            error: "bad or missing grant token".into(),
        }),
    )
        .into_response()
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(ApiError {
            error: "grant does not cover this key".into(),
        }),
    )
        .into_response()
}

fn internal() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError {
            error: "internal error".into(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use anyhow::Result;
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use axum::http::{HeaderValue, StatusCode};
    use chuk_train_proto::{keys, CodeRef, RunId, SignedUrl};
    use serde::de::DeserializeOwned;

    use super::*;
    use crate::artifacts::{ArtifactStore, FsArtifactStore};
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
            public_url: "http://cp.example:9".into(),
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

    fn unique_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "chuk-upload-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).expect("create temp artifacts dir");
        dir
    }

    /// A real (if minimal) `AppState`, `upload`/`fetch`/`blob_url` all take
    /// `State<Arc<AppState>>` directly, so there is no lighter seam than
    /// building one. The artifact store is real too — a filesystem store
    /// rooted at a fresh temp directory — so upload/fetch round-trip actual
    /// bytes on disk (a mock/no-op `Providers` keeps the rest cheap).
    async fn test_state(artifacts: Arc<dyn ArtifactStore>) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
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
            key_encryption_key: None,
        })
    }

    async fn fs_state() -> Arc<AppState> {
        test_state(Arc::new(FsArtifactStore::new(unique_dir()))).await
    }

    fn bearer(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("{BEARER_PREFIX}{token}")).expect("header value"),
        );
        headers
    }

    async fn body_json<T: DeserializeOwned>(resp: Response) -> T {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn mint(state: &AppState, run_id: &str) -> String {
        state.hub.grants().mint(
            RunId(run_id.to_owned()),
            CodeRef {
                name: "unit".into(),
                sha: "abc123".into(),
            },
        )
    }

    // ---- upload -------------------------------------------------------

    #[tokio::test]
    async fn upload_then_fetch_round_trips_bytes_for_a_valid_grant() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let key = "runs/RUN-1/logs/slice_0.log".to_owned();

        let put_resp = upload(
            State(state.clone()),
            bearer(&token),
            Path(key.clone()),
            Bytes::from_static(b"hello worker"),
        )
        .await;
        assert_eq!(put_resp.status(), StatusCode::CREATED);

        let get_resp = fetch(State(state), bearer(&token), Path(key)).await;
        assert_eq!(get_resp.status(), StatusCode::OK);
        let bytes = to_bytes(get_resp.into_body(), usize::MAX)
            .await
            .expect("body");
        assert_eq!(&bytes[..], b"hello worker");
    }

    #[tokio::test]
    async fn upload_rejects_a_missing_grant_token() {
        let state = fs_state().await;
        let resp = upload(
            State(state),
            HeaderMap::new(),
            Path("runs/RUN-1/x".into()),
            Bytes::from_static(b"x"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let error: ApiError = body_json(resp).await;
        assert_eq!(error.error, "bad or missing grant token");
    }

    #[tokio::test]
    async fn upload_rejects_an_unknown_grant_token() {
        let state = fs_state().await;
        let resp = upload(
            State(state),
            bearer("grant-does-not-exist"),
            Path("runs/RUN-1/x".into()),
            Bytes::from_static(b"x"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn upload_rejects_a_key_outside_the_grants_scope() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");

        // RUN-1's grant does not cover RUN-2's tree.
        let resp = upload(
            State(state),
            bearer(&token),
            Path("runs/RUN-2/logs/slice_0.log".into()),
            Bytes::from_static(b"x"),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let error: ApiError = body_json(resp).await;
        assert_eq!(error.error, "grant does not cover this key");
    }

    #[tokio::test]
    async fn upload_returns_internal_error_when_the_store_write_fails() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");

        let file_key = "runs/RUN-1/blob".to_owned();
        let put1 = upload(
            State(state.clone()),
            bearer(&token),
            Path(file_key.clone()),
            Bytes::from_static(b"x"),
        )
        .await;
        assert_eq!(put1.status(), StatusCode::CREATED);

        // `blob` now exists as a plain file; writing "under" it as if it were
        // a directory makes the filesystem backend's `create_dir_all` fail,
        // driving `upload`'s `Err` branch for real (not a mocked failure).
        let nested_key = format!("{file_key}/nested");
        let put2 = upload(
            State(state),
            bearer(&token),
            Path(nested_key),
            Bytes::from_static(b"y"),
        )
        .await;
        assert_eq!(put2.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ---- fetch ----------------------------------------------------------

    #[tokio::test]
    async fn fetch_rejects_a_missing_grant_token() {
        let state = fs_state().await;
        let resp = fetch(State(state), HeaderMap::new(), Path("runs/RUN-1/x".into())).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn fetch_rejects_a_key_outside_the_grants_scope() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let resp = fetch(State(state), bearer(&token), Path("runs/RUN-2/x".into())).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn fetch_404s_for_a_key_in_scope_that_was_never_uploaded() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let resp = fetch(
            State(state),
            bearer(&token),
            Path("runs/RUN-1/never-written".into()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let error: ApiError = body_json(resp).await;
        assert_eq!(error.error, "no such artifact");
    }

    #[tokio::test]
    async fn fetch_may_read_the_runs_assigned_code_unit_outside_its_run_tree() {
        let state = fs_state().await;
        let code = CodeRef {
            name: "unit".into(),
            sha: "deadbeef".into(),
        };
        let token = state.hub.grants().mint(RunId("RUN-1".into()), code.clone());
        let key = keys::code_unit_tarball(&code.name, &code.sha);
        state
            .artifacts
            .put(&key, b"tarball bytes".to_vec())
            .await
            .expect("seed code unit");

        let resp = fetch(State(state), bearer(&token), Path(key)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("body");
        assert_eq!(&bytes[..], b"tarball bytes");
    }

    // ---- blob_url ---------------------------------------------------------

    #[tokio::test]
    async fn blob_url_rejects_a_missing_grant_token() {
        let state = fs_state().await;
        let resp = blob_url(
            State(state),
            HeaderMap::new(),
            Json(BlobUrlRequest {
                key: "runs/RUN-1/x".into(),
                method: BlobMethod::Put,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn blob_url_rejects_a_key_outside_the_grants_scope() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let resp = blob_url(
            State(state),
            bearer(&token),
            Json(BlobUrlRequest {
                key: "runs/RUN-2/x".into(),
                method: BlobMethod::Get,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn blob_url_falls_back_to_the_control_plane_upload_route_when_the_backend_cannot_presign()
    {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let key = "runs/RUN-1/ckpt/model.safetensors".to_owned();

        let resp = blob_url(
            State(state),
            bearer(&token),
            Json(BlobUrlRequest {
                key: key.clone(),
                method: BlobMethod::Put,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: BlobUrlResponse = body_json(resp).await;
        assert_eq!(
            body.url,
            format!("http://cp.example:9{API_PREFIX}/upload/{key}")
        );
        assert!(body.requires_grant_header);
    }

    #[tokio::test]
    async fn blob_url_falls_back_to_the_control_plane_fetch_route_for_a_get() {
        let state = fs_state().await;
        let token = mint(&state, "RUN-1");
        let key = "runs/RUN-1/ckpt/model.safetensors".to_owned();

        let resp = blob_url(
            State(state),
            bearer(&token),
            Json(BlobUrlRequest {
                key: key.clone(),
                method: BlobMethod::Get,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: BlobUrlResponse = body_json(resp).await;
        assert_eq!(
            body.url,
            format!("http://cp.example:9{API_PREFIX}/fetch/{key}")
        );
        assert!(body.requires_grant_header);
    }

    /// `ArtifactStore` double whose `presign_*` calls return a fixed result
    /// instead of the filesystem backend's `Ok(None)` default — exercises the
    /// direct-presigned-URL and presign-failure branches of `blob_url` that an
    /// S3/R2 backend takes but `FsArtifactStore` never does.
    struct PresignStub {
        fails: bool,
    }

    #[async_trait]
    impl ArtifactStore for PresignStub {
        async fn put(&self, _key: &str, _bytes: Vec<u8>) -> Result<String> {
            unreachable!("blob_url never writes through the store")
        }
        async fn get(&self, _key: &str) -> Result<Vec<u8>> {
            unreachable!("blob_url never reads through the store")
        }
        async fn exists(&self, _key: &str) -> Result<bool> {
            unreachable!()
        }
        async fn delete(&self, _key: &str) -> Result<()> {
            unreachable!()
        }
        async fn copy(&self, _src: &str, _dst: &str) -> Result<()> {
            unreachable!()
        }
        fn uri(&self, key: &str) -> String {
            format!("mock://{key}")
        }
        async fn presign_get(&self, key: &str, _ttl: Duration) -> Result<Option<SignedUrl>> {
            self.presign(key)
        }
        async fn presign_put(&self, key: &str, _ttl: Duration) -> Result<Option<SignedUrl>> {
            self.presign(key)
        }
    }

    impl PresignStub {
        fn presign(&self, key: &str) -> Result<Option<SignedUrl>> {
            if self.fails {
                anyhow::bail!("presign backend unreachable");
            }
            Ok(Some(SignedUrl {
                url: format!("https://signed.example/{key}"),
                expires_at: 0.0,
            }))
        }
    }

    #[tokio::test]
    async fn blob_url_returns_the_presigned_url_directly_when_the_backend_provides_one() {
        let state = test_state(Arc::new(PresignStub { fails: false })).await;
        let token = mint(&state, "RUN-1");
        let key = "runs/RUN-1/ckpt/model.safetensors".to_owned();

        let resp = blob_url(
            State(state),
            bearer(&token),
            Json(BlobUrlRequest {
                key: key.clone(),
                method: BlobMethod::Put,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body: BlobUrlResponse = body_json(resp).await;
        assert_eq!(body.url, format!("https://signed.example/{key}"));
        assert!(!body.requires_grant_header);
    }

    #[tokio::test]
    async fn blob_url_returns_internal_error_when_presigning_fails() {
        let state = test_state(Arc::new(PresignStub { fails: true })).await;
        let token = mint(&state, "RUN-1");

        let resp = blob_url(
            State(state),
            bearer(&token),
            Json(BlobUrlRequest {
                key: "runs/RUN-1/ckpt/model.safetensors".into(),
                method: BlobMethod::Get,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
