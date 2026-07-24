//! Archive endpoints (spec §11.5).
//!
//! Note: this is `crate::api::archive`, distinct from `crate::archive` (the
//! archiver itself), which is referred to by its full path below.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{Role, RunId};
use serde::Deserialize;

use super::{bad_request, internal, require_role};
use crate::apikey::AuthContext;
use crate::AppState;

#[derive(Deserialize, Default)]
pub struct ArchiveParams {
    #[serde(default)]
    force: bool,
}

fn archive_outcome_json(outcome: crate::archive::Outcome) -> serde_json::Value {
    use crate::archive::Outcome;
    match outcome {
        Outcome::Archived { step, files } => {
            serde_json::json!({ "status": "archived", "step": step, "files": files })
        }
        Outcome::AlreadyArchived => serde_json::json!({ "status": "already_archived" }),
        Outcome::NoCheckpoint => serde_json::json!({ "status": "no_checkpoint" }),
    }
}

/// Archive one run's final checkpoint + logs/metrics to Drive now (spec §11.5).
pub async fn archive_run(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Query(params): Query<ArchiveParams>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let Some(archiver) = state.archiver.clone() else {
        return bad_request("archive tier not configured (no Drive credentials)");
    };
    match archiver.archive_run(&RunId(run_id), params.force).await {
        Ok(outcome) => Json(archive_outcome_json(outcome)).into_response(),
        Err(error) => internal(error),
    }
}

/// Sweep: archive every completed run not yet on Drive (backstop, on demand).
pub async fn archive_all(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let Some(archiver) = state.archiver.clone() else {
        return bad_request("archive tier not configured (no Drive credentials)");
    };
    match archiver.sweep_once().await {
        Ok(archived) => Json(serde_json::json!({ "archived": archived })).into_response(),
        Err(error) => internal(error),
    }
}

/// Per-run archive state: each recent run's final checkpoint location + when.
pub async fn archive_status(State(state): State<Arc<AppState>>) -> Response {
    let runs = match state
        .hub
        .store
        .runs(&Default::default(), chuk_train_proto::DEFAULT_RUN_LIST_LIMIT)
        .await
    {
        Ok(runs) => runs,
        Err(error) => return internal(error),
    };
    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        let ckpt = state.hub.store.latest_checkpoint(&run.id).await.ok().flatten();
        out.push(serde_json::json!({
            "run_id": run.id.0,
            "name": run.name,
            "state": run.state.as_str(),
            "final_step": ckpt.as_ref().map(|c| c.step),
            "location": ckpt.as_ref().map(|c| c.location),
            "archived_at": ckpt.as_ref().and_then(|c| c.archived_at),
        }));
    }
    Json(out).into_response()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use chuk_train_proto::{CheckpointMeta, RunSpec, ShellSpec};

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

    /// A real (if minimal) `AppState`, matching `dash.rs`'s pattern — these
    /// handlers take `State<Arc<AppState>>` directly, so there's no lighter
    /// seam. `archiver` stays `None`: `Archiver` only wraps a real
    /// `DriveClient`, which needs live Google credentials to construct, so the
    /// "archive tier not configured" refusal path is what's reachable here;
    /// the credentialed success path is exercised by `archive.rs`'s own
    /// `#[ignore]`d live test instead.
    async fn test_state() -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let root = std::env::temp_dir().join(format!("chuk-archive-test-{}", uuid::Uuid::new_v4()));
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

    #[test]
    fn outcome_json_covers_every_variant() {
        assert_eq!(
            archive_outcome_json(crate::archive::Outcome::Archived { step: 5, files: 3 }),
            serde_json::json!({ "status": "archived", "step": 5, "files": 3 })
        );
        assert_eq!(
            archive_outcome_json(crate::archive::Outcome::AlreadyArchived),
            serde_json::json!({ "status": "already_archived" })
        );
        assert_eq!(
            archive_outcome_json(crate::archive::Outcome::NoCheckpoint),
            serde_json::json!({ "status": "no_checkpoint" })
        );
    }

    #[tokio::test]
    async fn archive_run_refuses_below_admin_role() {
        let state = test_state().await;
        let resp = archive_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("RUN-1".into()),
            Query(ArchiveParams::default()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "requires admin role");
    }

    #[tokio::test]
    async fn archive_run_without_a_configured_archiver_is_a_bad_request() {
        let state = test_state().await;
        let resp = archive_run(
            State(state),
            axum::Extension(test_ctx(Role::Admin)),
            Path("RUN-1".into()),
            Query(ArchiveParams::default()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "archive tier not configured (no Drive credentials)");
    }

    #[tokio::test]
    async fn archive_all_refuses_below_admin_role() {
        let state = test_state().await;
        let resp = archive_all(State(state), axum::Extension(test_ctx(Role::Write))).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn archive_all_without_a_configured_archiver_is_a_bad_request() {
        let state = test_state().await;
        let resp = archive_all(State(state), axum::Extension(test_ctx(Role::Sysadmin))).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "archive tier not configured (no Drive credentials)");
    }

    #[tokio::test]
    async fn archive_status_is_empty_with_no_runs() {
        let state = test_state().await;
        let resp = archive_status(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await, serde_json::json!([]));
    }

    #[tokio::test]
    async fn archive_status_reports_each_runs_latest_checkpoint() {
        let state = test_state().await;
        let with_ckpt = state.hub.submit("has-ckpt", &shell_run(), None, None).await.unwrap();
        state
            .hub
            .store
            .record_checkpoint(&with_ckpt, 7, "ckpt-hot/has-ckpt/step_7", "hash7", &CheckpointMeta::default())
            .await
            .unwrap();
        let bare = state.hub.submit("no-ckpt", &shell_run(), None, None).await.unwrap();

        let resp = archive_status(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 2);

        let by_id = |id: &str| rows.iter().find(|r| r["run_id"] == id).expect("row present");
        let ckpt_row = by_id(&with_ckpt.0);
        assert_eq!(ckpt_row["name"], "has-ckpt");
        assert_eq!(ckpt_row["final_step"], 7);
        assert_eq!(ckpt_row["location"], "r2_hot");
        assert!(ckpt_row["archived_at"].is_null());

        let bare_row = by_id(&bare.0);
        assert_eq!(bare_row["name"], "no-ckpt");
        assert!(bare_row["final_step"].is_null());
        assert!(bare_row["location"].is_null());
    }
}
