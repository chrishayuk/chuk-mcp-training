//! Run lifecycle, logs, events, metrics, and code units.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use chuk_train_proto::{
    BuildCodeUnitRequest, LogsResponse, MetricSeries, RegisterGateRequest, Role, RunEvent, RunId,
    RunRecord, RunSpec, RunState, RunSummary, ShellSpec, SubmitRunRequest, SubmitRunResponse,
    SubmitShellRequest, SubmitSweepRequest, SubmitSweepResponse, WorkerId, WorkerInfo,
    WorkerTelemetry, DEFAULT_LOG_TAIL_LINES, DEFAULT_METRIC_DOWNSAMPLE, DEFAULT_RUN_LIST_LIMIT,
    DEFAULT_SHELL_TIMEOUT, GATE_SCOPE_RUN,
};
use serde::Deserialize;

use super::{bad_request, internal, not_found, require_role};
use crate::apikey::AuthContext;
use crate::{codeunit, AppState};

/// Query-param key for the metric-key filter list (comma-separated).
const METRIC_KEYS_SEP: char = ',';

pub async fn fleet(State(state): State<Arc<AppState>>) -> Response {
    match state.hub.store.fleet().await {
        Ok(workers) => Json::<Vec<WorkerInfo>>(workers).into_response(),
        Err(error) => internal(error),
    }
}

/// `GET /workers/{id}/telemetry` — the worker's latest host sample (chuk-compute
/// M4 `sys/*`): GPU/CPU/memory for the live dashboard. 404 if it never reported.
pub async fn worker_telemetry(
    State(state): State<Arc<AppState>>,
    Path(worker_id): Path<String>,
) -> Response {
    match state.hub.store.worker_telemetry(&WorkerId(worker_id)).await {
        Ok(Some(telemetry)) => Json::<WorkerTelemetry>(telemetry).into_response(),
        Ok(None) => not_found(),
        Err(error) => internal(error),
    }
}

pub async fn submit_shell(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SubmitShellRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let spec = RunSpec::Shell(ShellSpec {
        command: request.command,
        timeout_s: request.timeout_s.unwrap_or(DEFAULT_SHELL_TIMEOUT.as_secs()),
    });
    // A shell probe is always an unattached scratch run — never mirrored to an
    // experiments-server logical run.
    match state
        .hub
        .submit(&request.name, &spec, None, Some(&ctx.owner_email))
        .await
    {
        Ok(run_id) => Json(SubmitRunResponse { run_id }).into_response(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct ListParams {
    limit: Option<u32>,
    offset: Option<u32>,
    state: Option<RunState>,
    experiment_ref: Option<String>,
    sweep_id: Option<String>,
}

pub async fn list_runs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> Response {
    let query = crate::store::RunQuery {
        state: params.state,
        experiment_ref: params.experiment_ref,
        sweep_id: params.sweep_id,
        offset: params.offset.unwrap_or(0),
    };
    match state
        .hub
        .store
        .runs(&query, params.limit.unwrap_or(DEFAULT_RUN_LIST_LIMIT))
        .await
    {
        Ok(runs) => Json::<Vec<RunSummary>>(runs).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn run_status(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.run(&RunId(run_id)).await {
        Ok(Some(run)) => Json::<RunRecord>(run).into_response(),
        Ok(None) => not_found(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct TailParams {
    lines: Option<u32>,
}

pub async fn tail_logs(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Query(params): Query<TailParams>,
) -> Response {
    let run_id = RunId(run_id);
    match state
        .hub
        .store
        .tail_logs(&run_id, params.lines.unwrap_or(DEFAULT_LOG_TAIL_LINES))
        .await
    {
        Ok(lines) => Json(LogsResponse { run_id, lines }).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn run_events(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.store.events(&RunId(run_id)).await {
        Ok(events) => Json::<Vec<RunEvent>>(events).into_response(),
        Err(error) => internal(error),
    }
}

pub async fn build_code_unit(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<BuildCodeUnitRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let built = codeunit::build(
        state.artifacts.as_ref(),
        &request.repo,
        request.commit.as_deref(),
        request.name.as_deref(),
        request.path.as_deref(),
    )
    .await;
    match built {
        Ok(info) => {
            if let Err(error) = state
                .hub
                .store
                .register_code_unit(&info.code, &info.manifest, &info.uri)
                .await
            {
                return internal(error);
            }
            Json(info).into_response()
        }
        Err(error) => bad_request(&error.to_string()),
    }
}

pub async fn submit_run(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SubmitRunRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    // Spec §8 pre-flight: an expensive submission must be a confirmed one.
    if !request.confirm_cost {
        let live = match state.hub.store.live_leases().await {
            Ok(live) => live,
            Err(error) => return internal(error),
        };
        let estimate = crate::budget::estimate_run_cost(&live, request.spec.timeout_s());
        let threshold = state.config.confirm_cost_threshold;
        if estimate > threshold {
            return bad_request(&format!(
                "estimated worst-case cost ${estimate:.2} ({:.1}h timeout at the most \
                 expensive live lease) exceeds the ${threshold:.2} confirm threshold — \
                 resubmit with confirm_cost=true, or lower timeout_s",
                request.spec.timeout_s() as f64 / 3600.0,
            ));
        }
    }
    match state
        .hub
        .submit(
            &request.name,
            &request.spec,
            request.experiment_ref.as_deref(),
            Some(&ctx.owner_email),
        )
        .await
    {
        Ok(run_id) => Json(SubmitRunResponse { run_id }).into_response(),
        Err(error) => internal(error),
    }
}

/// `POST /runs/{id}/stop` — cancel a run. Signals the worker (Cancel →
/// Cancelled); a queued run is cancelled immediately. Returns the run's current
/// record (a running run may still show `running` until the worker confirms).
pub async fn stop_run(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let run_id = RunId(run_id);
    match state.hub.stop_run(&run_id).await {
        Ok(()) => current_run(&state, &run_id).await,
        // Bad state (already terminal) or unknown id — a client error.
        Err(error) => bad_request(&error.to_string()),
    }
}

#[derive(Deserialize)]
pub struct SubmitFromExperimentParams {
    name: Option<String>,
}

/// `POST /runs/from-experiment/{run_id}` — submit a train run built entirely
/// from an existing chuk-experiments-server run's own `config`/`workspec`
/// (spec §11.6 push mode): `run_id` is that run's `RUN-…` id, fetched over
/// REST and mapped straight to a `TrainSpec`, then submitted attached to it —
/// no client-side re-specification of the training job. Optional `?name=`
/// overrides the harness run's display name (defaults to `run_id`).
pub async fn submit_run_from_experiment(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Query(params): Query<SubmitFromExperimentParams>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    match state
        .hub
        .submit_from_experiment(&run_id, params.name.as_deref(), Some(&ctx.owner_email))
        .await
    {
        Ok(run_id) => Json(SubmitRunResponse { run_id }).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

/// `POST /api/sweeps` — fan a sweep out into child runs (spec §5.2/§6
/// `submit_sweep`). The §8 pre-flight multiplies the per-child worst case by
/// the fan-out, so sweep multiplication is confirmed knowingly or not at all.
pub async fn submit_sweep(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Json(request): Json<SubmitSweepRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    // Expand first: a bad axis is a 400 before anything is recorded, and the
    // pre-flight needs the child count.
    let children = match crate::sweep::expand(&request.spec.template, &request.spec.axes) {
        Ok(children) => children,
        Err(reason) => return bad_request(&reason),
    };
    if !request.confirm_cost {
        let live = match state.hub.store.live_leases().await {
            Ok(live) => live,
            Err(error) => return internal(error),
        };
        let per_child =
            crate::budget::estimate_run_cost(&live, request.spec.template.timeout_s);
        let total = per_child * children.len() as f64;
        let threshold = state.config.confirm_cost_threshold;
        if total > threshold {
            return bad_request(&format!(
                "estimated worst-case cost ${total:.2} ({} children × ${per_child:.2}) \
                 exceeds the ${threshold:.2} confirm threshold — resubmit with \
                 confirm_cost=true, or narrow the axes / lower timeout_s",
                children.len(),
            ));
        }
    }
    match state
        .hub
        .submit_sweep(&request.name, &request.spec, Some(&ctx.owner_email))
        .await
    {
        Ok((sweep_id, run_ids)) => Json(SubmitSweepResponse { sweep_id, run_ids }).into_response(),
        Err(error) => bad_request(&error.to_string()),
    }
}

#[derive(Deserialize)]
pub struct SweepStatusParams {
    key: Option<String>,
}

/// `GET /api/sweeps/{sweep_id}` — children + cross-child mean/std/range of one
/// metric key at matched steps (spec §5.2 `sweep_status`; `?key=` defaults to
/// `loss`).
pub async fn sweep_status(
    State(state): State<Arc<AppState>>,
    Path(sweep_id): Path<String>,
    Query(params): Query<SweepStatusParams>,
) -> Response {
    let key = params
        .key
        .unwrap_or_else(|| chuk_train_proto::DEFAULT_SWEEP_METRIC_KEY.to_owned());
    match state.hub.sweep_status(&sweep_id, &key).await {
        Ok(Some(status)) => Json(status).into_response(),
        Ok(None) => not_found(),
        Err(error) => internal(error),
    }
}

/// `POST /runs/{run_id}/gates` — register (upsert by name) a gate on a run
/// (spec §6). The expression is validated against the closed grammar here, so
/// a typo is a 400 at registration, never a silently-dead watchdog.
pub async fn register_gate(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
    Json(request): Json<RegisterGateRequest>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    if request.name.trim().is_empty() {
        return bad_request("gate name required");
    }
    if let Err(reason) = crate::gate::parse(&request.expr) {
        return bad_request(&reason);
    }
    match state.hub.store.run(&RunId(run_id.clone())).await {
        Ok(Some(_)) => {}
        Ok(None) => return not_found(),
        Err(error) => return internal(error),
    }
    match state
        .hub
        .store
        .register_gate(
            GATE_SCOPE_RUN,
            &run_id,
            request.name.trim(),
            request.expr.trim(),
            request.action,
        )
        .await
    {
        Ok(()) => match state.hub.store.gates(GATE_SCOPE_RUN, &run_id).await {
            Ok(gates) => Json(gates).into_response(),
            Err(error) => internal(error),
        },
        Err(error) => internal(error),
    }
}

/// `GET /runs/{run_id}/gates` — evaluate every gate fresh and return the
/// verdicts (spec §6 `check_gates`). Evaluating on read keeps `no_improve`
/// honest for a run that has stopped emitting metrics.
pub async fn check_gates(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
) -> Response {
    match state.hub.evaluate_gates(&RunId(run_id)).await {
        Ok(gates) => Json(gates).into_response(),
        Err(error) => internal(error),
    }
}

/// `POST /runs/{id}/resume` — re-queue a terminal run; a train run resumes from
/// its latest checkpoint on reassignment.
pub async fn resume_run(
    State(state): State<Arc<AppState>>,
    axum::Extension(ctx): axum::Extension<AuthContext>,
    Path(run_id): Path<String>,
) -> Response {
    if let Err(resp) = require_role(&ctx, Role::Write) {
        return resp;
    }
    let run_id = RunId(run_id);
    match state.hub.resume_run(&run_id).await {
        Ok(()) => current_run(&state, &run_id).await,
        Err(error) => bad_request(&error.to_string()),
    }
}

/// Fetch and return a run's current record (shared by stop/resume responses).
async fn current_run(state: &Arc<AppState>, run_id: &RunId) -> Response {
    match state.hub.store.run(run_id).await {
        Ok(Some(run)) => Json::<RunRecord>(run).into_response(),
        Ok(None) => not_found(),
        Err(error) => internal(error),
    }
}

#[derive(Deserialize)]
pub struct MetricParams {
    keys: Option<String>,
    since_step: Option<u64>,
    downsample: Option<u32>,
}

pub async fn run_metrics(
    State(state): State<Arc<AppState>>,
    Path(run_id): Path<String>,
    Query(params): Query<MetricParams>,
) -> Response {
    let run_id = RunId(run_id);
    let keys: Option<Vec<String>> = params.keys.as_deref().map(|raw| {
        raw.split(METRIC_KEYS_SEP)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    });
    let series = state
        .hub
        .store
        .metric_series(
            &run_id,
            keys.as_deref(),
            params.since_step.unwrap_or(0),
            params.downsample.unwrap_or(DEFAULT_METRIC_DOWNSAMPLE),
        )
        .await;
    match series {
        Ok(series) => Json::<MetricSeries>(series).into_response(),
        Err(error) => internal(error),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;

    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use chuk_compute_wire as wire;
    use chuk_train_proto::{CodeRef, GateAction, Hardware, Lease, LeaseState, SweepSpec, TrainSpec};
    use tokio::sync::mpsc;

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

    fn shell_spec(timeout_s: u64) -> RunSpec {
        RunSpec::Shell(ShellSpec { command: "true".into(), timeout_s })
    }

    fn train_template() -> TrainSpec {
        TrainSpec {
            code: CodeRef { name: "unit".into(), sha: "abc".into() },
            entrypoint: "train".into(),
            config: None,
            overrides: serde_json::json!({}),
            artifacts_in: Vec::new(),
            data: None,
            checkpoint: Default::default(),
            seed: None,
            arch: None,
            timeout_s: 3600,
            links: Vec::new(),
        }
    }

    /// A live lease priced at `price_hr`, standing in for a connected worker's
    /// billing rate for the spec §8 pre-flight estimate. `estimate_run_cost`
    /// reads straight off `live_leases()`, so no worker attach is needed to
    /// drive the confirm_cost refusal path.
    fn priced_lease(price_hr: f64) -> Lease {
        Lease {
            worker_id: WorkerId("w-priced".into()),
            provider: "vast".into(),
            instance_id: "i-1".into(),
            price_hr,
            granted_min: 60.0,
            drain_window_min: 5.0,
            started_at: 1_000.0,
            state: LeaseState::Active,
            extensions: vec![],
        }
    }

    fn list_params() -> ListParams {
        ListParams { limit: None, offset: None, state: None, experiment_ref: None, sweep_id: None }
    }

    /// A throwaway local-directory "repo" with a minimal code unit at its
    /// root, mirroring `codeunit.rs`'s own `scratch_repo` test helper (kept
    /// local here since that one is private to its module).
    fn scratch_repo() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("chuk-runs-codeunit-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create scratch repo dir");
        std::fs::write(
            dir.join("unit.toml"),
            "name = \"scratch\"\nversion = \"0.1.0\"\n[entrypoints]\ntrain = \"true\"\n",
        )
        .expect("write manifest");
        std::fs::write(dir.join("train.py"), "print('hi')\n").expect("write entrypoint");
        dir
    }

    /// A real (if minimal) `AppState`, matching `checkpoints.rs`'s/`system.rs`'s
    /// pattern — these handlers take `State<Arc<AppState>>` directly, so
    /// there's no lighter seam. Each call gets its own artifact-store root (a
    /// fresh temp dir), matching `checkpoints.rs`'s rationale (build_code_unit
    /// writes real tarball bytes and must not collide across tests).
    async fn test_state() -> Arc<AppState> {
        let store: Arc<dyn crate::store::Store> =
            Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let root = std::env::temp_dir().join(format!("chuk-runs-test-{}", uuid::Uuid::new_v4()));
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

    // ---- fleet / worker_telemetry ------------------------------------------

    #[tokio::test]
    async fn fleet_lists_an_attached_worker() {
        let state = test_state().await;
        let (tx, _rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-fleet".into());
        state.hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();

        let resp = fleet(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], "w-fleet");
    }

    #[tokio::test]
    async fn worker_telemetry_404s_for_a_worker_that_never_reported() {
        let state = test_state().await;
        let resp = worker_telemetry(State(state), Path("w-unknown".into())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn worker_telemetry_returns_the_latest_host_sample() {
        let state = test_state().await;
        let (tx, _rx) = mpsc::unbounded_channel();
        let worker = WorkerId("w-telemetry".into());
        state.hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
        let mut sys = BTreeMap::new();
        sys.insert("sys/cpu".to_owned(), 0.5);
        state
            .hub
            .on_message(
                &worker,
                wire::WorkerToCp::Metric { seq: 1, job_id: None, step: None, values: sys },
            )
            .await
            .unwrap();

        let resp = worker_telemetry(State(state), Path("w-telemetry".into())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["values"]["sys/cpu"], 0.5);
    }

    // ---- submit_shell ---------------------------------------------------------

    #[tokio::test]
    async fn submit_shell_refuses_below_write_role() {
        let state = test_state().await;
        let resp = submit_shell(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Json(SubmitShellRequest { name: "probe".into(), command: "true".into(), timeout_s: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn submit_shell_creates_an_unattached_scratch_run() {
        let state = test_state().await;
        let resp = submit_shell(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitShellRequest { name: "probe".into(), command: "true".into(), timeout_s: Some(30) }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let run_id = RunId(body["run_id"].as_str().expect("run_id").to_owned());
        let rec = state.hub.store.run(&run_id).await.unwrap().expect("run recorded");
        assert!(matches!(rec.spec, RunSpec::Shell(_)));
        assert!(rec.summary.experiment_ref.is_none(), "a shell probe is never attached");
        assert_eq!(rec.summary.created_by.as_deref(), Some("tester@example.com"));
    }

    // ---- list_runs --------------------------------------------------------

    #[tokio::test]
    async fn list_runs_filters_by_state() {
        let state = test_state().await;
        let queued = state.hub.submit("q", &shell_spec(60), None, None).await.unwrap();
        let done = state.hub.submit("d", &shell_spec(60), None, None).await.unwrap();
        state
            .hub
            .store
            .transition(&done, RunState::Completed, None, Some(0), serde_json::json!({}))
            .await
            .unwrap();

        let resp = list_runs(
            State(state),
            Query(ListParams { state: Some(RunState::Completed), ..list_params() }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], done.0.as_str());
        assert_ne!(rows[0]["id"], queued.0.as_str());
    }

    #[tokio::test]
    async fn list_runs_filters_by_experiment_ref() {
        let state = test_state().await;
        let attached = state.hub.submit("a", &shell_spec(60), Some("RUN-exp-1"), None).await.unwrap();
        state.hub.submit("s", &shell_spec(60), None, None).await.unwrap();

        let resp = list_runs(
            State(state),
            Query(ListParams { experiment_ref: Some("RUN-exp-1".into()), ..list_params() }),
        )
        .await;
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], attached.0.as_str());
    }

    #[tokio::test]
    async fn list_runs_filters_by_sweep_id() {
        let state = test_state().await;
        let spec = SweepSpec {
            template: train_template(),
            axes: BTreeMap::from([("seed".to_owned(), vec![1.into(), 2.into()])]),
            concurrency: 0,
        };
        let (sweep_id, children) = state.hub.submit_sweep("var", &spec, None).await.unwrap();
        state.hub.submit("s", &shell_spec(60), None, None).await.unwrap();

        let resp = list_runs(
            State(state),
            Query(ListParams { sweep_id: Some(sweep_id), ..list_params() }),
        )
        .await;
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), children.len());
    }

    #[tokio::test]
    async fn list_runs_honors_limit_and_offset() {
        let state = test_state().await;
        state.hub.submit("1", &shell_spec(60), None, None).await.unwrap();
        let second = state.hub.submit("2", &shell_spec(60), None, None).await.unwrap();
        state.hub.submit("3", &shell_spec(60), None, None).await.unwrap();

        // Newest-first ordering: [3, 2, 1]; limit=1 offset=1 skips the newest
        // and returns exactly the middle one.
        let resp = list_runs(
            State(state.clone()),
            Query(ListParams { limit: Some(1), offset: Some(1), ..list_params() }),
        )
        .await;
        let body = body_json(resp).await;
        let rows = body.as_array().expect("array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], second.0.as_str());

        // An offset past the end returns nothing, not an error.
        let resp = list_runs(
            State(state),
            Query(ListParams { limit: Some(50), offset: Some(50), ..list_params() }),
        )
        .await;
        let body = body_json(resp).await;
        assert_eq!(body.as_array().unwrap().len(), 0);
    }

    // ---- run_status ---------------------------------------------------------

    #[tokio::test]
    async fn run_status_returns_the_record() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        let resp = run_status(State(state), Path(run.0.clone())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["id"], run.0.as_str());
        assert_eq!(body["state"], "queued");
    }

    #[tokio::test]
    async fn run_status_404s_for_an_unknown_run() {
        let state = test_state().await;
        let resp = run_status(State(state), Path("EXEC-does-not-exist".into())).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ---- tail_logs ------------------------------------------------------------

    #[tokio::test]
    async fn tail_logs_returns_recorded_lines_in_order() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        state.hub.store.append_log(&run, "line one").await.unwrap();
        state.hub.store.append_log(&run, "line two").await.unwrap();

        let resp =
            tail_logs(State(state), Path(run.0.clone()), Query(TailParams { lines: None })).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["run_id"], run.0.as_str());
        assert_eq!(body["lines"], serde_json::json!(["line one", "line two"]));
    }

    #[tokio::test]
    async fn tail_logs_honors_a_custom_line_limit() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        for line in ["a", "b", "c"] {
            state.hub.store.append_log(&run, line).await.unwrap();
        }
        let resp =
            tail_logs(State(state), Path(run.0.clone()), Query(TailParams { lines: Some(1) })).await;
        let body = body_json(resp).await;
        assert_eq!(body["lines"], serde_json::json!(["c"]));
    }

    // ---- run_events -----------------------------------------------------------

    #[tokio::test]
    async fn run_events_returns_the_submission_events() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        let resp = run_events(State(state), Path(run.0.clone())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let events = body.as_array().expect("array");
        // create_run always records Created then Queued.
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["event"], "created");
        assert_eq!(events[1]["event"], "queued");
    }

    // ---- run_metrics ----------------------------------------------------------

    #[tokio::test]
    async fn run_metrics_filters_by_the_requested_keys() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        state
            .hub
            .store
            .append_metrics(&run, 1, &BTreeMap::from([("loss".to_owned(), 1.0), ("acc".to_owned(), 0.5)]))
            .await
            .unwrap();
        state
            .hub
            .store
            .append_metrics(&run, 2, &BTreeMap::from([("loss".to_owned(), 0.5)]))
            .await
            .unwrap();

        let resp = run_metrics(
            State(state),
            Path(run.0.clone()),
            Query(MetricParams { keys: Some("loss".into()), since_step: None, downsample: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let series = body["series"].as_object().expect("series map");
        assert_eq!(series.len(), 1, "acc must be filtered out by ?keys=loss");
        assert_eq!(series["loss"].as_array().unwrap().len(), 2);
    }

    // ---- submit_run (spec §8 pre-flight) ---------------------------------

    #[tokio::test]
    async fn submit_run_refuses_below_write_role() {
        let state = test_state().await;
        let resp = submit_run(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Json(SubmitRunRequest {
                name: "r".into(),
                spec: shell_spec(60),
                experiment_ref: None,
                confirm_cost: false,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn submit_run_succeeds_unconfirmed_when_under_the_threshold() {
        let state = test_state().await;
        // No live leases: the worst-case estimate is $0.00, which never
        // exceeds the (also $0.00) default test threshold, so an unconfirmed
        // submission still goes through.
        let resp = submit_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitRunRequest {
                name: "r".into(),
                spec: shell_spec(60),
                experiment_ref: None,
                confirm_cost: false,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn submit_run_refuses_an_unconfirmed_submission_over_threshold_with_the_exact_estimate() {
        let state = test_state().await;
        state.hub.store.create_lease(&priced_lease(2.5)).await.unwrap();

        let resp = submit_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitRunRequest {
                name: "r".into(),
                spec: shell_spec(3600),
                experiment_ref: None,
                confirm_cost: false,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(
            body["error"],
            "estimated worst-case cost $2.50 (1.0h timeout at the most expensive live lease) exceeds the $0.00 confirm threshold — resubmit with confirm_cost=true, or lower timeout_s"
        );
    }

    #[tokio::test]
    async fn submit_run_confirm_cost_bypasses_the_pre_flight_check() {
        let state = test_state().await;
        state.hub.store.create_lease(&priced_lease(2.5)).await.unwrap();

        let resp = submit_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitRunRequest {
                name: "r".into(),
                spec: shell_spec(3600),
                experiment_ref: None,
                confirm_cost: true,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ---- submit_run_from_experiment ----------------------------------------

    #[tokio::test]
    async fn submit_run_from_experiment_refuses_below_write_role() {
        let state = test_state().await;
        let resp = submit_run_from_experiment(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Path("RUN-ext-1".into()),
            Query(SubmitFromExperimentParams { name: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn submit_run_from_experiment_refuses_when_no_mirror_is_configured() {
        // test_state() builds the Hub with experiments: None — the only guard
        // testable without a real chuk-experiments-server (mirrors
        // hub/tests.rs's own `submit_from_experiment_errors_when_mirror_not_configured`).
        let state = test_state().await;
        let resp = submit_run_from_experiment(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("RUN-ext-1".into()),
            Query(SubmitFromExperimentParams { name: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("not configured"));
    }

    // ---- stop_run / resume_run ---------------------------------------------

    #[tokio::test]
    async fn stop_run_refuses_below_write_role() {
        let state = test_state().await;
        let resp =
            stop_run(State(state), axum::Extension(test_ctx(Role::Read)), Path("EXEC-1".into())).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn stop_run_cancels_a_queued_run_and_returns_its_record() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        let resp = stop_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["state"], "cancelled");
    }

    #[tokio::test]
    async fn stop_run_rejects_an_already_terminal_run() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        state.hub.stop_run(&run).await.unwrap();
        let resp = stop_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("already"));
    }

    #[tokio::test]
    async fn resume_run_refuses_below_write_role() {
        let state = test_state().await;
        let resp = resume_run(State(state), axum::Extension(test_ctx(Role::Read)), Path("EXEC-1".into()))
            .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn resume_run_requeues_a_terminal_run_and_returns_its_record() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        state.hub.stop_run(&run).await.unwrap();
        let resp = resume_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["state"], "queued");
    }

    #[tokio::test]
    async fn resume_run_rejects_a_still_live_run() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        let resp = resume_run(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("terminal"));
    }

    // ---- submit_sweep (spec §8 pre-flight, multiplied) ---------------------

    #[tokio::test]
    async fn submit_sweep_refuses_below_write_role() {
        let state = test_state().await;
        let spec = SweepSpec {
            template: train_template(),
            axes: BTreeMap::from([("seed".to_owned(), vec![1.into()])]),
            concurrency: 0,
        };
        let resp = submit_sweep(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Json(SubmitSweepRequest { name: "sw".into(), spec, confirm_cost: false }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn submit_sweep_rejects_an_empty_axes_map_before_recording_anything() {
        let state = test_state().await;
        let spec = SweepSpec { template: train_template(), axes: BTreeMap::new(), concurrency: 0 };
        let resp = submit_sweep(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitSweepRequest { name: "sw".into(), spec, confirm_cost: false }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert!(body["error"].as_str().unwrap().contains("axes must not be empty"));
    }

    #[tokio::test]
    async fn submit_sweep_refuses_an_unconfirmed_submission_over_the_multiplied_threshold() {
        let state = test_state().await;
        state.hub.store.create_lease(&priced_lease(2.0)).await.unwrap();
        let spec = SweepSpec {
            template: train_template(),
            axes: BTreeMap::from([("seed".to_owned(), vec![1.into(), 2.into()])]),
            concurrency: 0,
        };
        let resp = submit_sweep(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitSweepRequest { name: "sw".into(), spec, confirm_cost: false }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(
            body["error"],
            "estimated worst-case cost $4.00 (2 children × $2.00) exceeds the $0.00 confirm threshold — resubmit with confirm_cost=true, or narrow the axes / lower timeout_s"
        );
    }

    #[tokio::test]
    async fn submit_sweep_confirm_cost_bypasses_the_pre_flight_and_fans_out() {
        let state = test_state().await;
        state.hub.store.create_lease(&priced_lease(2.0)).await.unwrap();
        let spec = SweepSpec {
            template: train_template(),
            axes: BTreeMap::from([("seed".to_owned(), vec![1.into(), 2.into()])]),
            concurrency: 0,
        };
        let resp = submit_sweep(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(SubmitSweepRequest { name: "sw".into(), spec, confirm_cost: true }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["run_ids"].as_array().unwrap().len(), 2);
    }

    // ---- sweep_status -----------------------------------------------------

    #[tokio::test]
    async fn sweep_status_404s_for_an_unknown_sweep() {
        let state = test_state().await;
        let resp = sweep_status(
            State(state),
            Path("SWEEP-nope".into()),
            Query(SweepStatusParams { key: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn sweep_status_returns_children_for_a_known_sweep() {
        let state = test_state().await;
        let spec = SweepSpec {
            template: train_template(),
            axes: BTreeMap::from([("seed".to_owned(), vec![1.into(), 2.into()])]),
            concurrency: 0,
        };
        let (sweep_id, _run_ids) = state.hub.submit_sweep("sw", &spec, None).await.unwrap();
        let resp = sweep_status(State(state), Path(sweep_id), Query(SweepStatusParams { key: None }))
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["children"].as_array().unwrap().len(), 2);
    }

    // ---- register_gate / check_gates ---------------------------------------

    #[tokio::test]
    async fn register_gate_refuses_below_write_role() {
        let state = test_state().await;
        let resp = register_gate(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Path("EXEC-1".into()),
            Json(RegisterGateRequest {
                name: "g".into(),
                expr: "last(loss) > 1".into(),
                action: GateAction::Record,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn register_gate_rejects_an_empty_name() {
        let state = test_state().await;
        let resp = register_gate(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("EXEC-1".into()),
            Json(RegisterGateRequest {
                name: "   ".into(),
                expr: "last(loss) > 1".into(),
                action: GateAction::Record,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_json(resp).await;
        assert_eq!(body["error"], "gate name required");
    }

    #[tokio::test]
    async fn register_gate_rejects_an_invalid_expression() {
        let state = test_state().await;
        let resp = register_gate(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("EXEC-1".into()),
            Json(RegisterGateRequest {
                name: "g".into(),
                expr: "not a real gate".into(),
                action: GateAction::Record,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn register_gate_404s_for_an_unknown_run() {
        let state = test_state().await;
        let resp = register_gate(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path("EXEC-does-not-exist".into()),
            Json(RegisterGateRequest {
                name: "g".into(),
                expr: "last(loss) > 1".into(),
                action: GateAction::Record,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn register_gate_registers_and_returns_it() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        let resp = register_gate(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Path(run.0.clone()),
            Json(RegisterGateRequest {
                name: " g ".into(),
                expr: " last(loss) > 1 ".into(),
                action: GateAction::StopRun,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let gates = body.as_array().expect("array");
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0]["name"], "g");
        assert_eq!(gates[0]["expr"], "last(loss) > 1");
    }

    #[tokio::test]
    async fn check_gates_evaluates_and_reports_a_tripped_verdict() {
        let state = test_state().await;
        let run = state.hub.submit("r", &shell_spec(60), None, None).await.unwrap();
        state
            .hub
            .store
            .register_gate(GATE_SCOPE_RUN, &run.0, "g", "last(loss) > 1", GateAction::Record)
            .await
            .unwrap();
        state
            .hub
            .store
            .append_metrics(&run, 1, &BTreeMap::from([("loss".to_owned(), 5.0)]))
            .await
            .unwrap();

        let resp = check_gates(State(state), Path(run.0.clone())).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        let gates = body.as_array().expect("array");
        assert_eq!(gates[0]["tripped"], true);
    }

    // ---- build_code_unit ----------------------------------------------------

    #[tokio::test]
    async fn build_code_unit_refuses_below_write_role() {
        let state = test_state().await;
        let resp = build_code_unit(
            State(state),
            axum::Extension(test_ctx(Role::Read)),
            Json(BuildCodeUnitRequest { repo: "/tmp".into(), commit: None, name: None, path: None }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn build_code_unit_builds_and_registers_a_local_directory_repo() {
        let state = test_state().await;
        let repo = scratch_repo();
        let resp = build_code_unit(
            State(state.clone()),
            axum::Extension(test_ctx(Role::Write)),
            Json(BuildCodeUnitRequest {
                repo: repo.to_string_lossy().into_owned(),
                commit: None,
                name: None,
                path: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["manifest"]["name"], "scratch");
        let sha = body["code"]["sha"].as_str().expect("sha").to_owned();
        let code = state.hub.store.code_unit("scratch", &sha).await.unwrap();
        assert!(code.is_some(), "build_code_unit must register the unit for later assignment");
        let _ = std::fs::remove_dir_all(&repo);
    }

    #[tokio::test]
    async fn build_code_unit_400s_when_the_manifest_is_missing() {
        let state = test_state().await;
        let dir = std::env::temp_dir().join(format!("chuk-runs-codeunit-empty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let resp = build_code_unit(
            State(state),
            axum::Extension(test_ctx(Role::Write)),
            Json(BuildCodeUnitRequest {
                repo: dir.to_string_lossy().into_owned(),
                commit: None,
                name: None,
                path: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
