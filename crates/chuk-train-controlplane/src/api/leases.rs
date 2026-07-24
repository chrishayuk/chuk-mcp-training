//! Provisioning, leases, spend, and the Colab bootstrap cell (M2).

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{ApiError, Role};

use super::{bad_request, internal, require_role};
use crate::apikey::AuthContext;
use crate::AppState;

#[derive(serde::Deserialize)]
pub struct OffersParams {
    provider: String,
    gpu: Option<String>,
    max_price_hr: Option<f64>,
}

pub async fn provider_offers(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OffersParams>,
) -> Response {
    match state
        .leases
        .offers(&params.provider, params.gpu.as_deref(), params.max_price_hr)
        .await
    {
        Ok(offers) => Json::<Vec<chuk_train_proto::Offer>>(offers).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn provision(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<chuk_train_proto::ProvisionRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    match state.leases.provision(&request).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn lease_status(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Response {
    match state
        .hub
        .store
        .lease(&chuk_train_proto::WorkerId(worker_id))
        .await
    {
        Ok(Some(lease)) => Json(lease).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no lease".into(),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}

pub async fn extend_lease(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::ExtendLeaseRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let worker_id = chuk_train_proto::WorkerId(worker_id);
    match state
        .leases
        .extend(&worker_id, request.minutes, &request.reason)
        .await
    {
        Ok(Some(lease)) => Json(lease).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: "no lease".into(),
            }),
        )
            .into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn teardown(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(worker_id): Path<String>,
    Json(request): Json<chuk_train_proto::TeardownRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let worker_id = chuk_train_proto::WorkerId(worker_id);
    // force skips the drain grace and destroys immediately.
    match state.leases.teardown(&worker_id, !request.force).await {
        Ok(result) => Json(result).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

#[derive(serde::Deserialize)]
pub struct ColabCellParams {
    lease_min: Option<f64>,
    labels: Option<String>,
}

const DEFAULT_COLAB_LABELS: &str = "colab,t4";

/// Worker-id prefix for Colab joins enrolled via a generated cell.
const COLAB_WORKER_ID_PREFIX: &str = "colab-";

/// Generate a ready-to-paste Colab bootstrap cell (spec §6). The control plane
/// fills in its own public URL + a **single-use join token** (spec §12) bound
/// to a freshly-minted worker id — never the shared config token — so a
/// leaked cell can only ever enrol/readmit that one identity.
pub async fn colab_cell(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ColabCellParams>,
) -> Response {
    let labels = params
        .labels
        .unwrap_or_else(|| DEFAULT_COLAB_LABELS.to_owned());
    let worker_id = chuk_train_proto::WorkerId(format!(
        "{COLAB_WORKER_ID_PREFIX}{}",
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    ));
    let token = match crate::apikey::mint_join_token(state.hub.store.as_ref(), &worker_id).await {
        Ok(token) => token,
        Err(error) => return internal(error),
    };
    // Optional lease flags: passing lease_min makes the worker self-drain at
    // T-drain (belt) matching the control plane's window.
    let lease_flags = match params.lease_min {
        Some(m) => format!(
            " --lease-min {m} --drain-window-min {}",
            state.config.drain_window_min
        ),
        None => String::new(),
    };
    // Bootstrap through install.sh (uname → target triple → download + verify →
    // exec), so the cell never hardcodes a per-target agent path. --worker-id
    // makes reconnects resume the token's bound identity.
    let cell = format!(
        r#"# chuk-train · Colab worker — paste into ONE cell (Runtime → T4 GPU), then run.
CP_URL = "{url}"
JOIN_TOKEN = "{token}"
WORKER_ID = "{worker}"
LABELS = "{labels}"

import subprocess
cmd = ("curl -fsSL " + CP_URL + "/install.sh | sh -s -- "
       "--cp " + CP_URL + " --token " + JOIN_TOKEN + " --worker-id " + WORKER_ID
       + " --labels " + LABELS + "{lease_flags}")
print("[chuk-train] bootstrapping worker via install.sh …")
subprocess.run(cmd, shell=True, check=False)
"#,
        url = state.config.public_url,
        worker = worker_id.0,
    );
    Json(chuk_train_proto::ColabCell { cell }).into_response()
}

#[derive(serde::Deserialize)]
pub struct SpendParams {
    period: Option<String>,
}

/// Spend per provider over a period (spec §8): committed = projected cost of
/// live leases, spent = realised lease_end cost from the ledger, with
/// cap/headroom attached where a matching-period budget exists.
pub async fn spend_status(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SpendParams>,
) -> Response {
    let period = params
        .period
        .unwrap_or_else(|| chuk_train_proto::DEFAULT_BUDGET_PERIOD.to_owned());
    if let Err(reason) = crate::budget::validate_period(&period) {
        return bad_request(&reason);
    }
    let (budgets, ledger, live) = match tokio::try_join!(
        state.hub.store.budgets(),
        state.hub.store.ledger_entries(),
        state.hub.store.live_leases(),
    ) {
        Ok(all) => all,
        Err(error) => return internal(error),
    };
    Json(crate::budget::report(&budgets, &ledger, &live, &period, unix_now())).into_response()
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub async fn list_budgets(State(state): State<Arc<AppState>>) -> Response {
    match state.hub.store.budgets().await {
        Ok(budgets) => Json::<Vec<chuk_train_proto::Budget>>(budgets).into_response(),
        Err(error) => internal(error),
    }
}

/// Upsert a budget cap (spec §6 `set_budget`). Admin-scoped: a cap is a
/// governance decision, not a per-run knob.
pub async fn set_budget(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<chuk_train_proto::SetBudgetRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    let period = request
        .period
        .unwrap_or_else(|| chuk_train_proto::DEFAULT_BUDGET_PERIOD.to_owned());
    if let Err(reason) = crate::budget::validate_scope(&request.scope) {
        return bad_request(&reason);
    }
    if let Err(reason) = crate::budget::validate_period(&period) {
        return bad_request(&reason);
    }
    if !request.cap.is_finite() || request.cap < 0.0 {
        return bad_request("cap must be a non-negative number");
    }
    let budget = chuk_train_proto::Budget {
        scope: request.scope,
        cap: request.cap,
        period,
        updated_at: unix_now(),
    };
    match state.hub.store.set_budget(&budget).await {
        Ok(()) => Json(budget).into_response(),
        Err(error) => internal(error),
    }
}

#[derive(serde::Deserialize)]
pub struct DeleteBudgetParams {
    scope: String,
}

/// Remove a budget cap by scope (query param — scopes contain `:`). Admin-scoped.
pub async fn delete_budget(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Query(params): Query<DeleteBudgetParams>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Admin) {
        return resp;
    }
    match state.hub.store.delete_budget(&params.scope).await {
        Ok(true) => Json(serde_json::json!({ "deleted": true })).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("no budget set for scope {:?}", params.scope),
            }),
        )
            .into_response(),
        Err(error) => internal(error),
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use chuk_train_proto::{
        Budget, Lease, LeaseState, WorkerId, LEDGER_EVENT_EXTEND, LEDGER_EVENT_LEASE_END,
    };

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

    /// A real (if minimal) `AppState` — matches `checkpoints.rs`'s/`system.rs`'s
    /// `test_state` pattern: these handlers take `State<Arc<AppState>>`
    /// directly, so there's no lighter seam. `agent_bin` lets a provision test
    /// point the mock provider at a real (fake) agent script instead of the
    /// sibling-binary lookup, which always misses under `cargo test`.
    async fn build_state(agent_bin: Option<String>) -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store, artifacts.clone(), None, None);
        let providers = Arc::new(build_providers("mock", agent_bin, None));
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

    async fn test_state() -> Arc<AppState> {
        build_state(None).await
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.expect("read body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn ts_now() -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_secs_f64()
    }

    /// Insert a lease record directly into the store — the API layer under
    /// test doesn't provision real instances, so most handler tests don't need
    /// a live process behind the lease (`provision`'s own tests do, below).
    fn test_lease(worker: &str, provider: &str, price_hr: f64, granted_min: f64) -> Lease {
        Lease {
            worker_id: WorkerId(worker.into()),
            provider: provider.into(),
            instance_id: format!("{provider}-instance-not-tracked"),
            price_hr,
            granted_min,
            drain_window_min: 1.0,
            started_at: ts_now(),
            state: LeaseState::Active,
            extensions: Vec::new(),
        }
    }

    /// A throwaway shell script standing in for the real agent binary, so
    /// `provision` can launch a genuine (short-lived) local child process
    /// without needing the actual `chuk-compute-worker` build — mirrors
    /// `provider::mock`'s own `FakeAgent` test helper.
    struct FakeAgent {
        path: std::path::PathBuf,
    }

    impl FakeAgent {
        fn sleeping() -> Self {
            let path = std::env::temp_dir().join(format!(
                "chuk-leases-api-test-agent-{}-{}",
                std::process::id(),
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::write(&path, "#!/bin/sh\nsleep 30\n").expect("write fake agent script");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                    .expect("chmod fake agent script");
            }
            Self { path }
        }
    }

    impl Drop for FakeAgent {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    // ---- provider_offers ----------------------------------------------------

    #[tokio::test]
    async fn provider_offers_lists_the_mock_catalog_with_no_filters() {
        let state = test_state().await;
        let resp = provider_offers(
            State(state),
            Query(OffersParams { provider: "mock".into(), gpu: None, max_price_hr: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let offers = body.as_array().expect("array");
        assert_eq!(offers.len(), 2);
        assert_eq!(offers[0]["provider"], "mock");
    }

    #[tokio::test]
    async fn provider_offers_applies_gpu_and_price_filters() {
        let state = test_state().await;
        let resp = provider_offers(
            State(state),
            Query(OffersParams {
                provider: "mock".into(),
                gpu: Some("a6000".into()),
                max_price_hr: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let offers = body.as_array().expect("array");
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0]["gpu"], "mock-a6000");
    }

    #[tokio::test]
    async fn provider_offers_bad_request_for_an_unknown_provider() {
        let state = test_state().await;
        let resp = provider_offers(
            State(state),
            Query(OffersParams { provider: "bogus".into(), gpu: None, max_price_hr: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("unknown provider"));
    }

    // ---- provision ------------------------------------------------------------

    fn provision_request() -> chuk_train_proto::ProvisionRequest {
        chuk_train_proto::ProvisionRequest {
            provider: "mock".into(),
            lease_min: 5.0,
            offer_id: None,
            gpu: None,
            max_price_hr: None,
        }
    }

    #[tokio::test]
    async fn provision_refuses_below_write_role() {
        let state = test_state().await;
        let resp = provision(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Json(provision_request()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn provision_surfaces_a_provider_failure_as_bad_request() {
        // No agent_bin override: the mock provider's sibling-binary lookup
        // always misses under `cargo test` (the test binary lives in
        // target/*/deps/, never next to chuk-compute-worker), so this
        // exercises provision's Err branch without any real process.
        let state = test_state().await;
        let resp = provision(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(provision_request()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("provider provision"));
    }

    #[tokio::test]
    async fn provision_then_extend_and_teardown_round_trip_through_the_real_lease_manager() {
        let agent = FakeAgent::sleeping();
        let state = build_state(Some(agent.path.to_string_lossy().into_owned())).await;

        let resp = provision(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Write)),
            Json(provision_request()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["lease"]["provider"], "mock");
        assert_eq!(body["lease"]["granted_min"], 5.0);
        assert_eq!(body["lease"]["state"], "active");
        // `bootstrap` is only populated by the (Colab) provider path; empty
        // means omitted (`skip_serializing_if`), not present-and-empty.
        assert!(body.get("bootstrap").is_none());
        let worker_id = body["worker_id"].as_str().expect("worker_id").to_owned();

        // extend: the lease's base grant is untouched; the extension is recorded
        // separately, and appended to the ledger.
        let resp = extend_lease(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Write)),
            Path(worker_id.clone()),
            Json(chuk_train_proto::ExtendLeaseRequest { minutes: 10.0, reason: "more time".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["granted_min"], 5.0);
        assert_eq!(body["extensions"][0]["minutes"], 10.0);
        assert_eq!(body["extensions"][0]["reason"], "more time");
        let ledger = state.hub.store.ledger_entries().await.unwrap();
        assert!(ledger.iter().any(|e| e.event == LEDGER_EVENT_EXTEND && e.minutes == 10.0));

        // teardown (force=false): drains first, then destroys the real child
        // process — provider-verified gone, not just marked so.
        let resp = teardown(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(worker_id),
            Json(chuk_train_proto::TeardownRequest { force: false }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["destroyed"], true);
        assert_eq!(body["status"], "gone");
    }

    // ---- lease_status ---------------------------------------------------------

    #[tokio::test]
    async fn lease_status_404s_for_an_unknown_worker() {
        let state = test_state().await;
        let resp = lease_status(State(state), Path("no-such-worker".into())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "no lease");
    }

    #[tokio::test]
    async fn lease_status_returns_a_recorded_lease() {
        let state = test_state().await;
        let lease = test_lease("w-1", "mock", 0.5, 30.0);
        state.hub.store.create_lease(&lease).await.unwrap();

        let resp = lease_status(State(state), Path("w-1".into())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["provider"], "mock");
        assert_eq!(body["price_hr"], 0.5);
        assert_eq!(body["granted_min"], 30.0);
        assert_eq!(body["state"], "active");
    }

    // ---- extend_lease -----------------------------------------------------

    #[tokio::test]
    async fn extend_lease_refuses_below_write_role() {
        let state = test_state().await;
        let resp = extend_lease(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Path("w-1".into()),
            Json(chuk_train_proto::ExtendLeaseRequest { minutes: 5.0, reason: String::new() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn extend_lease_404s_for_an_unknown_worker() {
        let state = test_state().await;
        let resp = extend_lease(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("no-such-worker".into()),
            Json(chuk_train_proto::ExtendLeaseRequest { minutes: 5.0, reason: String::new() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "no lease");
    }

    #[tokio::test]
    async fn extend_lease_bad_request_when_it_would_breach_a_budget() {
        let state = test_state().await;
        let lease = test_lease("w-budget", "mock", 10.0, 60.0);
        state.hub.store.create_lease(&lease).await.unwrap();
        state
            .hub
            .store
            .set_budget(&Budget { scope: "provider:mock".into(), cap: 0.0, period: "all".into(), updated_at: 0.0 })
            .await
            .unwrap();

        let resp = extend_lease(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("w-budget".into()),
            Json(chuk_train_proto::ExtendLeaseRequest { minutes: 60.0, reason: "push it".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("budget breach"));
    }

    // ---- teardown -----------------------------------------------------------

    #[tokio::test]
    async fn teardown_refuses_below_write_role() {
        let state = test_state().await;
        let resp = teardown(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Path("w-1".into()),
            Json(chuk_train_proto::TeardownRequest { force: true }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn teardown_bad_request_for_an_unknown_worker() {
        let state = test_state().await;
        let resp = teardown(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("no-such-worker".into()),
            Json(chuk_train_proto::TeardownRequest { force: true }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("no lease for worker"));
    }

    #[tokio::test]
    async fn teardown_with_force_skips_the_drain_and_still_destroys() {
        let state = test_state().await;
        let lease = test_lease("w-force", "mock", 0.2, 5.0);
        state.hub.store.create_lease(&lease).await.unwrap();

        // force=true means drain_first=false; the mock provider has never
        // heard of this instance id, so destroy is a harmless no-op and
        // status reports it already gone.
        let resp = teardown(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("w-force".into()),
            Json(chuk_train_proto::TeardownRequest { force: true }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["destroyed"], true);
        assert_eq!(body["status"], "gone");
    }

    // ---- colab_cell -----------------------------------------------------------

    #[tokio::test]
    async fn colab_cell_defaults_labels_and_omits_lease_flags_when_unset() {
        let state = test_state().await;
        let resp = colab_cell(
            State(state),
            Query(ColabCellParams { lease_min: None, labels: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let cell = body["cell"].as_str().expect("cell");
        assert!(cell.contains(r#"LABELS = "colab,t4""#));
        assert!(cell.contains(r#"CP_URL = "http://127.0.0.1:9""#));
        assert!(cell.contains(r#"WORKER_ID = "colab-"#));
        assert!(cell.contains(r#"JOIN_TOKEN = "cjt_"#) || cell.contains("JOIN_TOKEN = \""));
        assert!(!cell.contains("--lease-min"));
    }

    #[tokio::test]
    async fn colab_cell_includes_lease_flags_and_custom_labels_when_set() {
        let state = test_state().await;
        let resp = colab_cell(
            State(state),
            Query(ColabCellParams { lease_min: Some(30.0), labels: Some("custom,tag".into()) }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let cell = body["cell"].as_str().expect("cell");
        assert!(cell.contains(r#"LABELS = "custom,tag""#));
        // config's drain_window_min is 5.0 in the test fixture.
        assert!(cell.contains(" --lease-min 30 --drain-window-min 5"));
    }

    // ---- spend_status ---------------------------------------------------------

    #[tokio::test]
    async fn spend_status_bad_request_for_an_invalid_period() {
        let state = test_state().await;
        let resp = spend_status(State(state), Query(SpendParams { period: Some("week".into()) })).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("unsupported budget period"));
    }

    #[tokio::test]
    async fn spend_status_reports_zero_totals_with_nothing_leased_or_spent() {
        let state = test_state().await;
        let resp = spend_status(State(state), Query(SpendParams { period: None })).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(
            body,
            serde_json::json!({
                "period": "month",
                "lines": [],
                "total_committed": 0.0,
                "total_spent": 0.0,
            })
        );
    }

    #[tokio::test]
    async fn spend_status_reports_committed_spent_and_headroom_against_a_budget() {
        let state = test_state().await;
        state
            .hub
            .store
            .set_budget(&Budget { scope: "provider:mock".into(), cap: 100.0, period: "month".into(), updated_at: 0.0 })
            .await
            .unwrap();
        state.hub.store.create_lease(&test_lease("w-spend", "mock", 1.0, 60.0)).await.unwrap();
        state
            .hub
            .store
            .ledger_append(&chuk_train_proto::LedgerEntry {
                ts: ts_now(),
                worker_id: WorkerId("w-spend-past".into()),
                provider: "mock".into(),
                event: LEDGER_EVENT_LEASE_END.into(),
                minutes: 60.0,
                cost: 5.0,
            })
            .await
            .unwrap();

        let resp = spend_status(State(state), Query(SpendParams { period: Some("month".into()) })).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let line = body["lines"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["provider"] == "mock")
            .expect("mock line");
        assert_eq!(line["committed"], 1.0);
        assert_eq!(line["spent"], 5.0);
        assert_eq!(line["cap"], 100.0);
        assert_eq!(line["headroom"], 94.0);
    }

    // ---- list_budgets -----------------------------------------------------

    #[tokio::test]
    async fn list_budgets_is_empty_with_none_configured() {
        let state = test_state().await;
        let resp = list_budgets(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await, serde_json::json!([]));
    }

    #[tokio::test]
    async fn list_budgets_returns_every_configured_budget() {
        let state = test_state().await;
        state
            .hub
            .store
            .set_budget(&Budget { scope: "global".into(), cap: 10.0, period: "month".into(), updated_at: 0.0 })
            .await
            .unwrap();
        state
            .hub
            .store
            .set_budget(&Budget { scope: "provider:mock".into(), cap: 20.0, period: "all".into(), updated_at: 0.0 })
            .await
            .unwrap();

        let resp = list_budgets(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let scopes: Vec<&str> = body.as_array().unwrap().iter().map(|b| b["scope"].as_str().unwrap()).collect();
        assert!(scopes.contains(&"global"));
        assert!(scopes.contains(&"provider:mock"));
    }

    // ---- set_budget -----------------------------------------------------------

    #[tokio::test]
    async fn set_budget_requires_admin_role() {
        let state = test_state().await;
        let resp = set_budget(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(chuk_train_proto::SetBudgetRequest { scope: "global".into(), cap: 10.0, period: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn set_budget_bad_request_for_an_unsupported_scope() {
        let state = test_state().await;
        let resp = set_budget(
            State(state),
            axum::Extension(test_ctx(Role::Admin)),
            Json(chuk_train_proto::SetBudgetRequest { scope: "label:cn7".into(), cap: 10.0, period: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("unsupported budget scope"));
    }

    #[tokio::test]
    async fn set_budget_bad_request_for_an_unsupported_period() {
        let state = test_state().await;
        let resp = set_budget(
            State(state),
            axum::Extension(test_ctx(Role::Admin)),
            Json(chuk_train_proto::SetBudgetRequest {
                scope: "global".into(),
                cap: 10.0,
                period: Some("week".into()),
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("unsupported budget period"));
    }

    #[tokio::test]
    async fn set_budget_bad_request_for_a_negative_or_non_finite_cap() {
        let state = test_state().await;
        for cap in [-1.0, f64::NAN, f64::INFINITY] {
            let resp = set_budget(
                State(state.clone()),
                axum::Extension(test_ctx(Role::Admin)),
                Json(chuk_train_proto::SetBudgetRequest { scope: "global".into(), cap, period: None }),
            )
            .await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "cap {cap} should be refused");
            let body = body_json(resp).await;
            assert!(body["error"].as_str().unwrap().contains("cap must be a non-negative number"));
        }
    }

    #[tokio::test]
    async fn set_budget_upserts_and_defaults_the_period_to_month() {
        let state = test_state().await;
        let resp = set_budget(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Admin)),
            Json(chuk_train_proto::SetBudgetRequest { scope: "provider:mock".into(), cap: 50.0, period: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["scope"], "provider:mock");
        assert_eq!(body["cap"], 50.0);
        assert_eq!(body["period"], "month");

        let budgets = state.hub.store.budgets().await.unwrap();
        assert_eq!(budgets.len(), 1);
        assert_eq!(budgets[0].scope, "provider:mock");
    }

    // ---- delete_budget --------------------------------------------------------

    #[tokio::test]
    async fn delete_budget_requires_admin_role() {
        let state = test_state().await;
        let resp = delete_budget(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Query(DeleteBudgetParams { scope: "global".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn delete_budget_404s_when_no_budget_is_set_for_the_scope() {
        let state = test_state().await;
        let resp = delete_budget(
            State(state),
            axum::Extension(test_ctx(Role::Admin)),
            Query(DeleteBudgetParams { scope: "provider:none".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("no budget set for scope"));
    }

    #[tokio::test]
    async fn delete_budget_removes_an_existing_budget() {
        let state = test_state().await;
        state
            .hub
            .store
            .set_budget(&Budget { scope: "global".into(), cap: 10.0, period: "month".into(), updated_at: 0.0 })
            .await
            .unwrap();

        let resp = delete_budget(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Admin)),
            Query(DeleteBudgetParams { scope: "global".into() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await, serde_json::json!({ "deleted": true }));
        assert!(state.hub.store.budgets().await.unwrap().is_empty());
    }
}
