//! Hub — the shared test module for every `impl Hub` slice (connection,
//! schedule, messages, submit, control, mirror). Kept as one file, mirroring
//! `store/sqlite/mod.rs`'s own test module, so cross-slice scenarios (e.g. a
//! cancel racing a replayed JobStarted) don't need cross-file test plumbing.

use super::connection::should_reap;
use super::messages::{kill_reason_state, run_id, step_from_uri};
use super::schedule::{dataset_label, eligible_for_assignment};
use super::*;
use crate::artifacts::FsArtifactStore;
use crate::datasets::Datasets;
use crate::experiments::Experiments;
use crate::store::SqliteStore;
use chuk_train_proto::ShellSpec;

#[test]
fn step_parses_from_the_trailing_segment() {
    assert_eq!(step_from_uri("ckpt-hot/RUN-1/step_500"), Some(500));
    assert_eq!(step_from_uri("step_0"), Some(0));
    assert_eq!(step_from_uri("ckpt-hot/RUN-1/final"), None);
    assert_eq!(step_from_uri("ckpt-hot/RUN-1/step_x"), None);
}

#[test]
fn job_id_maps_to_run_id_verbatim() {
    assert_eq!(run_id(&wire::JobId::from("RUN-9")), RunId::from("RUN-9"));
}

#[test]
fn dataset_label_keeps_the_name_and_swaps_in_the_resolved_sha() {
    assert_eq!(dataset_label("tiny/ds@askedsha", "resolvedsha"), "tiny/ds@sha256:resolvedsha");
    assert_eq!(dataset_label("bareshaonly", "resolvedsha"), "sha256:resolvedsha");
}

#[test]
fn cancel_lands_cancelled_every_other_reason_fails() {
    assert_eq!(
        kill_reason_state(&wire::KillReason::Cancel),
        (RunState::Cancelled, EXIT_CODE_CANCELLED)
    );
    for reason in [
        wire::KillReason::Wall,
        wire::KillReason::MaxRuntime,
        wire::KillReason::Drain,
        wire::KillReason::OomGuard,
    ] {
        assert_eq!(
            kill_reason_state(&reason),
            (RunState::Failed, EXIT_CODE_AGENT_ERROR)
        );
    }
}

pub(super) async fn test_hub() -> Arc<Hub> {
    let store = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    // Shell runs never touch the artifact store; temp_dir is never written.
    let artifacts = Arc::new(FsArtifactStore::new(std::env::temp_dir()));
    Hub::new(store, artifacts, None, None)
}

pub(super) fn shell_run() -> RunSpec {
    RunSpec::Shell(ShellSpec { command: "sleep 1".into(), timeout_s: 60 })
}

pub(super) fn train_template() -> TrainSpec {
    TrainSpec {
        code: chuk_train_proto::CodeRef { name: "unit".into(), sha: "abc".into() },
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

#[tokio::test]
async fn sweep_fans_out_caps_concurrency_and_aggregates() {
    let hub = test_hub().await;
    let spec = chuk_train_proto::SweepSpec {
        template: train_template(),
        axes: std::collections::BTreeMap::from([(
            "seed".to_owned(),
            vec![80.into(), 81.into(), 82.into()],
        )]),
        concurrency: 2,
    };
    let (sweep_id, run_ids) = hub.submit_sweep("var", &spec, None).await.unwrap();
    assert_eq!(run_ids.len(), 3);
    let status = hub.sweep_status(&sweep_id, "loss").await.unwrap().unwrap();
    assert_eq!(status.children.len(), 3);
    assert_eq!(status.children[0].assignment["seed"], serde_json::json!(80));
    assert_eq!(status.children[2].assignment["seed"], serde_json::json!(82));

    // Oldest child is assignable first; once two are active the sweep is
    // at its concurrency and the third is held back...
    let first = hub.next_assignable_queued().await.unwrap().unwrap().summary.id;
    assert_eq!(first, run_ids[0]);
    for id in &run_ids[..2] {
        hub.store
            .transition(id, RunState::Running, None, None, serde_json::json!({}))
            .await
            .unwrap();
    }
    assert!(hub.next_assignable_queued().await.unwrap().is_none());
    // ...while a non-sweep run stays assignable.
    let scratch = hub.submit("probe", &shell_run(), None, None).await.unwrap();
    assert_eq!(
        hub.next_assignable_queued().await.unwrap().unwrap().summary.id,
        scratch
    );
    // A child finishing frees a slot for the held-back (older) child.
    hub.store
        .transition(&run_ids[0], RunState::Completed, None, Some(0), serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(
        hub.next_assignable_queued().await.unwrap().unwrap().summary.id,
        run_ids[2]
    );

    // Cross-child aggregation at the matched step.
    for (id, loss) in [(&run_ids[0], 2.0), (&run_ids[1], 4.0)] {
        let values = std::collections::BTreeMap::from([("loss".to_owned(), loss)]);
        hub.store.append_metrics(id, 0, &values).await.unwrap();
    }
    let status = hub.sweep_status(&sweep_id, "loss").await.unwrap().unwrap();
    assert_eq!(status.aggregate.len(), 1);
    assert_eq!(status.aggregate[0].n, 2);
    assert_eq!(status.aggregate[0].mean, 3.0);
    assert_eq!((status.aggregate[0].min, status.aggregate[0].max), (2.0, 4.0));
}

#[tokio::test]
async fn sweep_status_is_none_for_an_unknown_sweep() {
    let hub = test_hub().await;
    assert!(hub.sweep_status("SWEEP-does-not-exist", "loss").await.unwrap().is_none());
}

#[tokio::test]
async fn tripped_watchdog_gate_stops_the_run_and_records_the_event() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    hub.store
        .register_gate(GATE_SCOPE_RUN, &run.0, "grad-blowup", "last(grad_norm) > 1e3", GateAction::StopRun)
        .await
        .unwrap();

    // Healthy metrics: evaluated, not tripped, run untouched.
    let mut healthy = std::collections::BTreeMap::new();
    healthy.insert("grad_norm".to_owned(), 10.0);
    hub.store.append_metrics(&run, 1, &healthy).await.unwrap();
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(false));
    let rec = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(rec.summary.state, RunState::Queued);

    // A gradient blowup trips the watchdog: run stops (queued → cancelled
    // directly), and the flip is a gate_evaluated event — exactly once.
    let mut blowup = std::collections::BTreeMap::new();
    blowup.insert("grad_norm".to_owned(), 5e3);
    hub.store.append_metrics(&run, 2, &blowup).await.unwrap();
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(true));
    let rec = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(rec.summary.state, RunState::Cancelled);

    // Re-evaluating while still tripped re-events nothing (the flip fired
    // once) and leaves the terminal run untouched.
    hub.evaluate_gates(&run).await.unwrap();
    let events = hub.store.events(&run).await.unwrap();
    let flips = events.iter().filter(|e| e.event == EventKind::GateEvaluated).count();
    assert_eq!(flips, 1);
}

#[tokio::test]
async fn tripped_stop_gate_converges_on_a_resurrected_run() {
    // EI0's Colab lesson: a lost JobKilled + replayed JobStarted can bring
    // a watchdog-stopped run back to life. The stop must re-fire on the
    // next evaluation while the verdict holds — without a second flip event.
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    hub.store
        .register_gate(GATE_SCOPE_RUN, &run.0, "grad-blowup", "last(grad_norm) > 1e3", GateAction::StopRun)
        .await
        .unwrap();
    let blowup = std::collections::BTreeMap::from([("grad_norm".to_owned(), 5e3)]);
    hub.store.append_metrics(&run, 1, &blowup).await.unwrap();
    hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Cancelled);

    // Resurrect (stands in for the replayed-JobStarted zombie).
    hub.resume_run(&run).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);

    // Still-tripped verdict re-stops the resurrected run...
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(true));
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Cancelled);
    // ...but the flip event stays singular.
    let events = hub.store.events(&run).await.unwrap();
    let flips = events.iter().filter(|e| e.event == EventKind::GateEvaluated).count();
    assert_eq!(flips, 1);
}

#[tokio::test]
async fn introspect_namespace_metrics_are_gateable_end_to_end() {
    // chuk-introspect I0 (spec §5.1/§6): `introspect/*` keys ride the
    // ordinary append_metrics + gate path with zero new machinery. This is
    // the CP half of the EI0 proof — the dead-ReLU watchdog.
    use chuk_train_proto::introspect::{layer_key, FAMILY_DEAD_FRAC};

    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let key = layer_key(FAMILY_DEAD_FRAC, 1);
    hub.store
        .register_gate(
            GATE_SCOPE_RUN,
            &run.0,
            "dead-relu",
            &format!("last({key}) > 0.5"),
            GateAction::StopRun,
        )
        .await
        .unwrap();

    // Healthy pulse: gate evaluates clean.
    let healthy = std::collections::BTreeMap::from([(key.clone(), 0.05)]);
    hub.store.append_metrics(&run, 1, &healthy).await.unwrap();
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(false));

    // Poisoned pulse (dead_frac = 1.0): the watchdog stops the run.
    let poisoned = std::collections::BTreeMap::from([(key.clone(), 1.0)]);
    hub.store.append_metrics(&run, 2, &poisoned).await.unwrap();
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(true));
    let rec = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(rec.summary.state, RunState::Cancelled);

    // And the series read returns the introspect key like any other.
    let series = hub
        .store
        .metric_series(&run, None, 0, 0)
        .await
        .unwrap();
    assert!(series.series.contains_key(&key));
}

#[tokio::test]
async fn record_gate_observes_without_stopping() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    hub.store
        .register_gate(GATE_SCOPE_RUN, &run.0, "nan-check", "isnan(last(loss))", GateAction::Record)
        .await
        .unwrap();
    let mut nan = std::collections::BTreeMap::new();
    nan.insert("loss".to_owned(), f64::NAN);
    hub.store.append_metrics(&run, 1, &nan).await.unwrap();
    let gates = hub.evaluate_gates(&run).await.unwrap();
    assert_eq!(gates[0].tripped, Some(true));
    // Recorded, not enforced: the run stays queued.
    let rec = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(rec.summary.state, RunState::Queued);
}

#[tokio::test]
async fn submit_from_experiment_errors_when_mirror_not_configured() {
    // test_hub() builds with experiments: None — the only guard testable
    // without a real chuk-experiments-server to fetch a run from.
    let hub = test_hub().await;
    let err = hub
        .submit_from_experiment("RUN-20260718-160217-00397", None, None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not configured"), "unexpected error: {err}");
}

#[tokio::test]
async fn stop_queued_run_cancels_immediately() {
    let hub = test_hub().await;
    let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
    // No worker connected — the run is queued; stop finalises it here.
    hub.stop_run(&run).await.unwrap();
    let rec = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(rec.summary.state, RunState::Cancelled);
    assert_eq!(rec.summary.exit_code, Some(EXIT_CODE_CANCELLED));
}

#[tokio::test]
async fn stop_rejects_an_already_terminal_run() {
    let hub = test_hub().await;
    let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
    hub.stop_run(&run).await.unwrap();
    let err = hub.stop_run(&run).await.unwrap_err();
    assert!(err.to_string().contains("already"), "unexpected error: {err}");
}

#[tokio::test]
async fn resume_requeues_a_terminal_run_but_not_a_live_one() {
    let hub = test_hub().await;
    let run = hub.submit("q", &shell_run(), None, None).await.unwrap();
    let err = hub.resume_run(&run).await.unwrap_err();
    assert!(err.to_string().contains("terminal"), "unexpected error: {err}");

    hub.stop_run(&run).await.unwrap();
    hub.resume_run(&run).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);
}

#[tokio::test]
async fn stop_running_run_signals_worker_then_jobkilled_finalises() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    // Attach a worker; the pump assigns the queued run to it.
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-test".into());
    hub.attach(&worker, tx, &[], &Hardware::default())
        .await
        .unwrap();
    assert!(matches!(
        rx.recv().await.expect("assign"),
        wire::CpToWorker::AssignJob { .. }
    ));
    // Stop signals the worker (run assigned, link live) — not finalised yet.
    hub.stop_run(&run).await.unwrap();
    match rx.recv().await.expect("cancel") {
        wire::CpToWorker::Cancel { job_id } => assert_eq!(job_id.0, run.0),
        other => panic!("expected Cancel, got {other:?}"),
    }
    let mid = hub.store.run(&run).await.unwrap().unwrap();
    assert!(!mid.summary.state.is_terminal(), "state={:?}", mid.summary.state);
    // The worker confirms the kill; the run lands in Cancelled.
    hub.on_message(
        &worker,
        wire::WorkerToCp::JobKilled {
            seq: 1,
            job_id: wire::JobId::from(run.0.clone()),
            reason: wire::KillReason::Cancel,
        },
    )
    .await
    .unwrap();
    let done = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(done.summary.state, RunState::Cancelled);
}

// ---- heartbeat reaper (spec §7) ---------------------------------------

fn worker_info(state: WorkerState, current_run: Option<&str>, age_s: f64) -> WorkerInfo {
    WorkerInfo {
        id: WorkerId("w".into()),
        labels: vec![],
        hardware: Hardware::default(),
        state,
        current_run: current_run.map(|r| RunId(r.into())),
        joined_at: 0.0,
        last_seen: 0.0,
        heartbeat_age_s: age_s,
        lease: None,
        telemetry: None,
    }
}

#[test]
fn assignment_eligibility_honours_run_drain_and_staleness() {
    let fresh = |age| worker_info(WorkerState::Connected, None, age);
    // Free, fresh, unleased → eligible.
    assert!(eligible_for_assignment(&fresh(1.0), 90.0));
    // Already running a job → not eligible.
    assert!(!eligible_for_assignment(
        &worker_info(WorkerState::Connected, Some("EXEC-1"), 1.0),
        90.0
    ));
    // Silent past the unreachable timeout → no new work (spec §7).
    assert!(!eligible_for_assignment(&fresh(120.0), 90.0));
    // Exactly at the boundary is still eligible (strictly greater excludes).
    assert!(eligible_for_assignment(&fresh(90.0), 90.0));
}

#[test]
fn reap_targets_only_connected_workers_past_the_preempt_wall() {
    // Connected + silent past preempt → reap.
    assert!(should_reap(&worker_info(WorkerState::Connected, Some("EXEC-1"), 601.0), 600.0));
    // Connected but recently seen → keep.
    assert!(!should_reap(&worker_info(WorkerState::Connected, Some("EXEC-1"), 120.0), 600.0));
    // Already disconnected (detach handled it) → skip.
    assert!(!should_reap(&worker_info(WorkerState::Disconnected, None, 9999.0), 600.0));
}

#[tokio::test]
async fn reaper_leaves_a_freshly_seen_worker_alone() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-fresh".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert_eq!(
        hub.store.run(&run).await.unwrap().unwrap().summary.state,
        RunState::Assigned
    );
    // The worker was just seen (age ~0), so the sweep must not touch it.
    hub.reap_stale_workers().await.unwrap();
    assert_eq!(
        hub.store.run(&run).await.unwrap().unwrap().summary.state,
        RunState::Assigned
    );
    assert!(matches!(
        hub.store.worker(&worker).await.unwrap().unwrap().state,
        WorkerState::Connected
    ));
}

#[tokio::test]
async fn detach_requeues_leased_but_keeps_persistent() {
    // Leased: a lost worker's run is requeued (bounded loss).
    let hub = test_hub().await;
    let run = hub.submit("leased", &shell_run(), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-leased".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    hub.detach(&worker, wire::WorkerClass::Leased).await.unwrap();
    assert_eq!(
        hub.store.run(&run).await.unwrap().unwrap().summary.state,
        RunState::Queued
    );

    // Persistent: the run stays assigned for the worker to reconnect (M3.2).
    let hub = test_hub().await;
    let run = hub.submit("persist", &shell_run(), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-persist".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    hub.detach(&worker, wire::WorkerClass::Persistent).await.unwrap();
    assert_eq!(
        hub.store.run(&run).await.unwrap().unwrap().summary.state,
        RunState::Assigned
    );
}

#[tokio::test]
async fn detach_on_an_idle_worker_is_a_plain_no_op() {
    // No run ever assigned: current_run is None, so the mid-run requeue
    // branch is skipped entirely rather than touching any run.
    let hub = test_hub().await;
    let worker = WorkerId("w-idle".into());
    let (tx, _rx) = mpsc::unbounded_channel();
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    hub.detach(&worker, wire::WorkerClass::Leased).await.unwrap();
    assert!(hub.store.worker(&worker).await.unwrap().unwrap().current_run.is_none());
}

#[tokio::test]
async fn send_to_an_unlinked_worker_returns_false() {
    let hub = test_hub().await;
    let unlinked = WorkerId("w-never-attached".into());
    let msg = wire::CpToWorker::Cancel { job_id: wire::JobId::from("unused") };
    assert!(!hub.send_to(&unlinked, msg).await);
}

// ---- chuk-datasets dispatch-time resolution (spec §6/§7.3) -------------

/// A one-route mock chuk-datasets server returning a canned `resolve`
/// response, so dispatch-time resolution is tested against a real HTTP
/// round trip rather than asserted on the client crate alone.
async fn mock_resolve_server() -> String {
    let body = serde_json::json!({
        "content_sha": "dsha123",
        "schema": "chuk-manifest-core-1",
        "verification": "verified-2run",
        "manifest": {
            "schema": "chuk-manifest-core-1",
            "shards": [{"sha256": "shard-a", "size": "10", "offset": "0"}]
        },
        "locations": [
            {"role": "canonical", "backend": "r2", "uri": "https://store/ds", "status": "live"}
        ],
        "lengths_sha": null,
        "plan": {"plan_sha": "psha123", "params": {}}
    });
    let app = axum::Router::new()
        .route("/v1/resolve", axum::routing::get(move || {
            let body = body.clone();
            async move { axum::Json(body) }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("mock resolve server") });
    format!("http://{addr}")
}

#[tokio::test]
async fn resolve_data_stages_shards_and_complete_meta_reads_the_resolved_identity() {
    let store = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    let artifacts = Arc::new(FsArtifactStore::new(std::env::temp_dir()));
    let base_url = mock_resolve_server().await;
    let hub = Hub::new(store, artifacts, None, Some(Arc::new(Datasets::at(base_url))));

    let mut train = train_template();
    // The client belt-and-braces-checks the asked sha against what the
    // server resolves, so the asked ref must name the mock's content_sha.
    train.data = Some(chuk_train_proto::DataRef {
        dataset: "tiny/ds@dsha123".into(),
        plan: Some("plan-ref".into()),
    });
    let run_id = hub.submit("t", &RunSpec::Train(Box::new(train.clone())), None, None).await.unwrap();

    let staging = hub.resolve_data(&run_id, &train.data).await.unwrap().expect("data staging");
    assert_eq!(staging.content_sha, "dsha123");
    assert_eq!(staging.plan_sha.as_deref(), Some("psha123"));
    assert_eq!(staging.shards.len(), 1);
    assert_eq!(staging.shards[0].sha256, "shard-a");
    assert_eq!(staging.shards[0].uri, "https://store/ds/shard-a");

    // complete_meta reads back the *resolved* identity, not the asked ref.
    let meta = hub.complete_meta(&run_id, 10).await.unwrap();
    assert_eq!(meta.dataset_sha.as_deref(), Some("dsha123"));
    assert_eq!(meta.plan_sha.as_deref(), Some("psha123"));
    assert_eq!(meta.datasets, vec!["tiny/ds@sha256:dsha123".to_string()]);
}

#[tokio::test]
async fn resolve_data_errors_when_a_run_declares_data_but_no_client_is_configured() {
    let hub = test_hub().await;
    let mut train = train_template();
    train.data = Some(chuk_train_proto::DataRef { dataset: "tiny/ds@sha".into(), plan: None });
    let run_id = hub.submit("t", &RunSpec::Train(Box::new(train.clone())), None, None).await.unwrap();
    let err = hub.resolve_data(&run_id, &train.data).await.unwrap_err();
    assert!(format!("{err:#}").contains("CHUK_DATASETS_URL"));
}

#[tokio::test]
async fn resolve_data_is_a_no_op_without_a_data_block() {
    let hub = test_hub().await;
    let run_id = hub.submit("t", &RunSpec::Train(Box::new(train_template())), None, None).await.unwrap();
    assert!(hub.resolve_data(&run_id, &None).await.unwrap().is_none());
    // complete_meta leaves the dataset fields unset for a run that never resolved one.
    let meta = hub.complete_meta(&run_id, 1).await.unwrap();
    assert!(meta.dataset_sha.is_none() && meta.plan_sha.is_none() && meta.datasets.is_empty());
}

#[tokio::test]
async fn attach_pump_assembles_and_assigns_a_train_run() {
    let hub = test_hub().await;
    let code = chuk_train_proto::CodeRef { name: "unit".into(), sha: "abc".into() };
    let manifest = chuk_train_proto::CodeUnitManifest {
        name: "unit".into(),
        version: "0.1".into(),
        entrypoints: std::collections::BTreeMap::from([("train".to_owned(), "python train.py".to_owned())]),
        python: None,
        requires: Default::default(),
    };
    hub.store.register_code_unit(&code, &manifest, "s3://unit.tar.zst").await.unwrap();

    let mut train = train_template();
    train.code = code;
    let run = hub.submit("t", &RunSpec::Train(Box::new(train)), None, None).await.unwrap();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-train".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();

    let wire::CpToWorker::AssignJob { job } = rx.recv().await.expect("assign") else {
        panic!("expected AssignJob");
    };
    assert_eq!(job.template.as_str(), "train");
    assert!(job.grant.is_some());
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Assigned);
}

#[tokio::test]
async fn attach_pump_resumes_from_the_latest_checkpoint() {
    let hub = test_hub().await;
    let code = chuk_train_proto::CodeRef { name: "unit".into(), sha: "abc".into() };
    let manifest = chuk_train_proto::CodeUnitManifest {
        name: "unit".into(),
        version: "0.1".into(),
        entrypoints: std::collections::BTreeMap::from([("train".to_owned(), "python train.py".to_owned())]),
        python: None,
        requires: Default::default(),
    };
    hub.store.register_code_unit(&code, &manifest, "s3://unit.tar.zst").await.unwrap();
    let mut train = train_template();
    train.code = code;
    let run = hub.submit("t", &RunSpec::Train(Box::new(train)), None, None).await.unwrap();

    let (tx1, mut rx1) = mpsc::unbounded_channel();
    let worker1 = WorkerId("w-resume-1".into());
    hub.attach(&worker1, tx1, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx1.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    // Record a checkpoint (as if the worker uploaded one), then the
    // worker vanishes — requeueing the run for a fresh assignment.
    hub.artifacts
        .put(&keys::checkpoint_file(&run.0, 5, CHECKPOINT_MODEL_FILE), b"weights".to_vec())
        .await
        .unwrap();
    let meta = hub.complete_meta(&run, 5).await.unwrap();
    hub.store.record_checkpoint(&run, 5, "ckpt-hot/t/step_5", "hash", &meta).await.unwrap();
    hub.detach(&worker1, wire::WorkerClass::Leased).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);

    // The next assignment resumes: the resume dir is set and the model +
    // meta ride alongside the code unit as staged inputs.
    let (tx2, mut rx2) = mpsc::unbounded_channel();
    let worker2 = WorkerId("w-resume-2".into());
    hub.attach(&worker2, tx2, &[], &Hardware::default()).await.unwrap();
    let wire::CpToWorker::AssignJob { job } = rx2.recv().await.expect("assign") else {
        panic!("expected AssignJob");
    };
    assert_eq!(job.env[chuk_train_proto::script_env::RESUME_CKPT], "${SANDBOX}/resume");
    assert_eq!(job.inputs.len(), 3, "unit + resume model + resume meta");
}

#[tokio::test]
async fn pump_requeues_when_the_worker_link_has_gone_away() {
    let hub = test_hub().await;
    let (tx, rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-gone".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    // The worker's receiver is dropped without a detach — the link is
    // still registered, but sending to it will now fail.
    drop(rx);

    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);
    assert!(hub.store.worker(&worker).await.unwrap().unwrap().current_run.is_none());
}

#[tokio::test]
async fn pump_skips_a_worker_already_running_a_job() {
    let hub = test_hub().await;
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-busy".into());
    let first = hub.submit("first", &shell_run(), None, None).await.unwrap();
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    // A second run stays queued: the only worker is already busy.
    let second = hub.submit("second", &shell_run(), None, None).await.unwrap();
    assert_eq!(hub.store.run(&first).await.unwrap().unwrap().summary.state, RunState::Assigned);
    assert_eq!(hub.store.run(&second).await.unwrap().unwrap().summary.state, RunState::Queued);
}

#[tokio::test]
async fn attach_pump_fails_closed_when_the_code_unit_is_not_registered() {
    let hub = test_hub().await;
    let run = hub.submit("t", &RunSpec::Train(Box::new(train_template())), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-missing-unit".into());
    // pump() propagates the assembly error rather than silently dropping
    // the run; attach() surfaces it to the caller. The run is never
    // transitioned, so it stays assignable on the next successful pump.
    let err = hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap_err();
    assert!(err.to_string().contains("not registered"), "unexpected error: {err}");
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);
}

#[tokio::test]
async fn on_message_drives_a_full_job_lifecycle_to_completion() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-life".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    let job_id = wire::JobId::from(run.0.clone());

    hub.on_message(&worker, wire::WorkerToCp::JobStarted { seq: 1, job_id: job_id.clone() })
        .await
        .unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Running);

    hub.on_message(
        &worker,
        wire::WorkerToCp::Log { seq: 2, job_id: job_id.clone(), line: "hello".into() },
    )
    .await
    .unwrap();

    let mut values = std::collections::BTreeMap::new();
    values.insert("loss".to_owned(), 1.0);
    hub.on_message(
        &worker,
        wire::WorkerToCp::Metric { seq: 3, job_id: Some(job_id.clone()), step: Some(1), values },
    )
    .await
    .unwrap();

    // Host telemetry (no job_id) records worker samples instead of a
    // metric series entry.
    let mut sys = std::collections::BTreeMap::new();
    sys.insert("sys/cpu".to_owned(), 0.5);
    hub.on_message(&worker, wire::WorkerToCp::Metric { seq: 4, job_id: None, step: None, values: sys })
        .await
        .unwrap();

    // Duplicate replay of an already-applied seq is dropped, not reapplied.
    let mut replay = std::collections::BTreeMap::new();
    replay.insert("loss".to_owned(), 999.0);
    hub.on_message(
        &worker,
        wire::WorkerToCp::Metric { seq: 3, job_id: Some(job_id.clone()), step: Some(1), values: replay },
    )
    .await
    .unwrap();
    let series = hub.store.metric_series(&run, None, 0, 0).await.unwrap();
    assert_eq!(series.series["loss"].len(), 1, "the replayed duplicate must not double-apply");

    // A checkpoint artifact: the model bytes must already be in the
    // artifact store before the control plane can hash + record them.
    hub.artifacts
        .put(&keys::checkpoint_file(&run.0, 1, CHECKPOINT_MODEL_FILE), b"weights".to_vec())
        .await
        .unwrap();
    hub.on_message(
        &worker,
        wire::WorkerToCp::Artifact {
            seq: 5,
            job_id: job_id.clone(),
            class: wire::ArtifactClass::from("checkpoint"),
            uri: "ckpt-hot/r/step_1".into(),
            sha256: None,
            bytes: None,
            meta: Value::Null,
        },
    )
    .await
    .unwrap();
    let ckpt = hub.store.latest_checkpoint(&run).await.unwrap().expect("checkpoint recorded");
    assert_eq!(ckpt.step, 1);

    // An artifact of an unhandled class is ignored, not an error.
    hub.on_message(
        &worker,
        wire::WorkerToCp::Artifact {
            seq: 6,
            job_id: job_id.clone(),
            class: wire::ArtifactClass::from("logs"),
            uri: "logs/r".into(),
            sha256: None,
            bytes: None,
            meta: Value::Null,
        },
    )
    .await
    .unwrap();

    // Hello/Heartbeat/ServiceReady/Drained, and any future variant, are
    // tolerated no-ops (spec §3 forward compatibility) — never an error.
    hub.on_message(&worker, wire::WorkerToCp::Heartbeat).await.unwrap();
    hub.on_message(
        &worker,
        wire::WorkerToCp::ServiceReady { seq: 7, job_id: job_id.clone(), ports: vec![8080] },
    )
    .await
    .unwrap();
    hub.on_message(&worker, wire::WorkerToCp::Drained).await.unwrap();

    hub.on_message(&worker, wire::WorkerToCp::JobExited { seq: 8, job_id: job_id.clone(), code: 0 })
        .await
        .unwrap();
    let done = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(done.summary.state, RunState::Completed);
}

#[tokio::test]
async fn on_message_job_started_after_cancel_does_not_resurrect_the_run() {
    // Live-observed on Colab: operator cancels, then a flapped reconnect
    // replays JobStarted with a genuinely new (higher) seq — dedup alone
    // doesn't catch it. The run must stay Cancelled with its exit code
    // intact, not flip back to Running.
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-flap".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    let job_id = wire::JobId::from(run.0.clone());
    hub.on_message(&worker, wire::WorkerToCp::JobStarted { seq: 1, job_id: job_id.clone() })
        .await
        .unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Running);

    // Stop signals the worker; the run finalises once it confirms the kill.
    hub.stop_run(&run).await.unwrap();
    assert!(matches!(rx.recv().await.expect("cancel"), wire::CpToWorker::Cancel { .. }));
    hub.on_message(
        &worker,
        wire::WorkerToCp::JobKilled { seq: 2, job_id: job_id.clone(), reason: wire::KillReason::Cancel },
    )
    .await
    .unwrap();
    let cancelled = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(cancelled.summary.state, RunState::Cancelled);
    assert_eq!(cancelled.summary.exit_code, Some(EXIT_CODE_CANCELLED));

    // The replay: a higher seq, so dedup lets it through to the handler.
    hub.on_message(&worker, wire::WorkerToCp::JobStarted { seq: 3, job_id: job_id.clone() })
        .await
        .unwrap();
    let after = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(after.summary.state, RunState::Cancelled, "replayed JobStarted must not resurrect a terminal run");
    assert_eq!(after.summary.exit_code, Some(EXIT_CODE_CANCELLED), "exit code must not be left stale by an ignored transition");
}

#[tokio::test]
async fn reap_stuck_assignments_requeues_an_unconfirmed_assign() {
    // Live-observed: a worker's outbound link can look open (`send_to`
    // reports success) while the underlying socket has gone stale, so
    // AssignJob never actually reaches the worker and no JobStarted ever
    // follows. The run must not be stuck in Assigned forever.
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-zombie".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Assigned);
    // The zombie link: still registered in `links` (nothing ever detached
    // it), but nothing is receiving on it any more — same shape as
    // `pump_requeues_when_the_worker_link_has_gone_away`. Without this, the
    // reap's own re-pump would just hand the freed run right back to the
    // still-"eligible" worker and mask the requeue.
    drop(rx);

    // max_age 0 makes any Assigned run immediately eligible, without
    // waiting out the real ASSIGNMENT_STUCK_TIMEOUT.
    hub.reap_stuck_assignments_older_than(0.0).await.unwrap();

    let after = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(after.summary.state, RunState::Queued);
}

#[tokio::test]
async fn reap_stuck_assignments_leaves_a_fresh_assignment_alone() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-ok".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    // A generous max_age means nothing looks stuck yet.
    hub.reap_stuck_assignments_older_than(3600.0).await.unwrap();

    let after = hub.store.run(&run).await.unwrap().unwrap();
    assert_eq!(after.summary.state, RunState::Assigned, "a fresh assignment must not be requeued");
}

#[tokio::test]
async fn on_message_job_exited_nonzero_fails_the_run() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-fail".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    hub.on_message(
        &worker,
        wire::WorkerToCp::JobExited { seq: 1, job_id: wire::JobId::from(run.0.clone()), code: 1 },
    )
    .await
    .unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Failed);
}

// ---- resume_high_water / grants() (mod.rs's own public getters) ----------

#[tokio::test]
async fn resume_high_water_defaults_to_zero_then_tracks_the_highest_streamed_seq() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-hw".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));

    // Never streamed anything yet: the HelloAck this worker would get back
    // echoes 0. An entirely unknown worker also defaults to 0, not a panic.
    assert_eq!(hub.resume_high_water(&worker), 0);
    assert_eq!(hub.resume_high_water(&WorkerId("never-seen".into())), 0);

    let job_id = wire::JobId::from(run.0.clone());
    hub.on_message(&worker, wire::WorkerToCp::JobStarted { seq: 3, job_id: job_id.clone() })
        .await
        .unwrap();
    assert_eq!(hub.resume_high_water(&worker), 3);

    // A higher seq advances the watermark...
    hub.on_message(
        &worker,
        wire::WorkerToCp::Log { seq: 7, job_id: job_id.clone(), line: "x".into() },
    )
    .await
    .unwrap();
    assert_eq!(hub.resume_high_water(&worker), 7);

    // ...but a stale replay at-or-below it is dropped and must not move it
    // backwards (messages.rs's own dedup already covers "dropped"; this
    // covers what the public getter reports afterwards).
    hub.on_message(&worker, wire::WorkerToCp::Log { seq: 4, job_id, line: "stale".into() })
        .await
        .unwrap();
    assert_eq!(hub.resume_high_water(&worker), 7);
}

#[tokio::test]
async fn grants_getter_resolves_the_very_token_the_scheduler_minted() {
    let hub = test_hub().await;
    let code = chuk_train_proto::CodeRef { name: "unit".into(), sha: "abc".into() };
    let manifest = chuk_train_proto::CodeUnitManifest {
        name: "unit".into(),
        version: "0.1".into(),
        entrypoints: std::collections::BTreeMap::from([("train".to_owned(), "python train.py".to_owned())]),
        python: None,
        requires: Default::default(),
    };
    hub.store.register_code_unit(&code, &manifest, "s3://unit.tar.zst").await.unwrap();
    let mut train = train_template();
    train.code = code;
    let run = hub.submit("t", &RunSpec::Train(Box::new(train)), None, None).await.unwrap();

    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-grant".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    let wire::CpToWorker::AssignJob { job } = rx.recv().await.expect("assign") else {
        panic!("expected AssignJob");
    };
    let token = job.grant.expect("a train run mints an upload grant");

    // The public getter must resolve to the very table the scheduler mints
    // into (upload.rs's only caller does exactly this), not a private copy.
    let grant = hub.grants().resolve(&token).expect("token resolves");
    assert_eq!(grant.run_id, run);
    assert!(hub.grants().resolve("not-a-real-token").is_none());
}

// ---- chuk-experiments-server reporting mirror (hub/mirror.rs) ------------

/// One captured call: (method, path, JSON body).
type MockCall = (&'static str, String, Value);

/// Captures every call a [`Hub`] configured with a real [`Experiments`]
/// client makes against it — method, path, and JSON body — so a test can
/// assert the mirror actually reported what it claims to, not just that
/// nothing panicked.
#[derive(Clone, Default)]
struct MockExperiments {
    calls: Arc<StdMutex<Vec<MockCall>>>,
}

impl MockExperiments {
    fn record(&self, method: &'static str, path: String, body: Value) {
        self.calls.lock().expect("mock calls lock").push((method, path, body));
    }

    fn calls(&self, method: &str, path: &str) -> Vec<Value> {
        self.calls
            .lock()
            .expect("mock calls lock")
            .iter()
            .filter(|(m, p, _)| *m == method && p == path)
            .map(|(_, _, b)| b.clone())
            .collect()
    }
}

/// A minimal chuk-experiments-server mock covering exactly the routes
/// [`Experiments`] calls (ensure / create-run / patch-run / post-artifact) —
/// enough to drive `Hub::mirror_created`/`mirror_state`/`mirror_checkpoint`
/// (spawned fire-and-forget by `submit`/`on_message`, only reachable when
/// `experiments` is configured — `test_hub()` always builds with `None`)
/// against a real HTTP round trip. Same shape as `mock_resolve_server` above
/// for chuk-datasets.
async fn mock_experiments_server() -> (String, MockExperiments) {
    let mock = MockExperiments::default();

    async fn ensure(
        axum::extract::State(mock): axum::extract::State<MockExperiments>,
        axum::Json(body): axum::Json<Value>,
    ) -> axum::Json<Value> {
        mock.record("POST", "/v1/experiments".into(), body);
        axum::Json(serde_json::json!({}))
    }
    async fn create_run(
        axum::extract::State(mock): axum::extract::State<MockExperiments>,
        axum::extract::Path(experiment): axum::extract::Path<String>,
        axum::Json(body): axum::Json<Value>,
    ) -> axum::Json<Value> {
        mock.record("POST", format!("/v1/experiments/{experiment}/runs"), body);
        axum::Json(serde_json::json!({ "id": "EXP-RUN-mock-1" }))
    }
    async fn patch_run(
        axum::extract::State(mock): axum::extract::State<MockExperiments>,
        axum::extract::Path(id): axum::extract::Path<String>,
        axum::Json(body): axum::Json<Value>,
    ) -> axum::Json<Value> {
        mock.record("PATCH", format!("/v1/runs/{id}"), body);
        axum::Json(serde_json::json!({}))
    }
    async fn post_artifact(
        axum::extract::State(mock): axum::extract::State<MockExperiments>,
        axum::extract::Path(id): axum::extract::Path<String>,
        axum::Json(body): axum::Json<Value>,
    ) -> axum::Json<Value> {
        mock.record("POST", format!("/v1/runs/{id}/artifacts"), body);
        axum::Json(serde_json::json!({}))
    }

    let app = axum::Router::new()
        .route("/v1/experiments", axum::routing::post(ensure))
        .route("/v1/experiments/{experiment}/runs", axum::routing::post(create_run))
        .route("/v1/runs/{id}", axum::routing::patch(patch_run))
        .route("/v1/runs/{id}/artifacts", axum::routing::post(post_artifact))
        .with_state(mock.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("mock experiments server") });
    (format!("http://{addr}"), mock)
}

#[tokio::test]
async fn mirror_reports_created_state_and_checkpoint_to_a_configured_experiments_server() {
    let (base_url, mock) = mock_experiments_server().await;
    // Experiments::from_env only reads env at construction time, so the
    // mutation window is just this one call (mirrors the pattern the
    // #[ignore]d live tests in experiments.rs already use).
    std::env::set_var(chuk_train_proto::env::EXPERIMENTS_URL, &base_url);
    std::env::set_var(chuk_train_proto::env::EXPERIMENTS_API_KEY, "test-key");
    let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    let exp = Experiments::from_env(store.clone(), &base_url).expect("experiments client (env just set)");
    std::env::remove_var(chuk_train_proto::env::EXPERIMENTS_URL);
    std::env::remove_var(chuk_train_proto::env::EXPERIMENTS_API_KEY);

    let artifacts = Arc::new(FsArtifactStore::new(std::env::temp_dir()));
    let hub = Hub::new(store, artifacts, Some(exp), None);

    let code = chuk_train_proto::CodeRef { name: "unit".into(), sha: "abc".into() };
    let manifest = chuk_train_proto::CodeUnitManifest {
        name: "unit".into(),
        version: "0.1".into(),
        entrypoints: std::collections::BTreeMap::from([("train".to_owned(), "python train.py".to_owned())]),
        python: None,
        requires: Default::default(),
    };
    hub.store.register_code_unit(&code, &manifest, "s3://unit.tar.zst").await.unwrap();
    let mut train = train_template();
    train.code = code;
    let run = hub.submit("t", &RunSpec::Train(Box::new(train)), None, None).await.unwrap();

    // mirror_created (submit.rs) spawns try_create; poll for it to land
    // instead of a blind sleep (same pattern as experiments.rs's live test).
    let mut mirrored = None;
    for _ in 0..100 {
        if let Ok(Some(ext)) = hub.store.experiments_run_id(&run).await {
            mirrored = Some(ext);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(mirrored.as_deref(), Some("EXP-RUN-mock-1"), "mirror_created must land the run");
    assert_eq!(mock.calls("POST", "/v1/experiments").len(), 1, "ensure() called once");
    assert_eq!(
        mock.calls("POST", &format!("/v1/experiments/{}/runs", chuk_train_proto::DEFAULT_EXPERIMENTS_EXPERIMENT))
            .len(),
        1
    );

    // mirror_state (messages.rs, JobStarted -> Running).
    let (tx, mut rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-mirror".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert!(matches!(rx.recv().await.unwrap(), wire::CpToWorker::AssignJob { .. }));
    let job_id = wire::JobId::from(run.0.clone());
    hub.on_message(&worker, wire::WorkerToCp::JobStarted { seq: 1, job_id: job_id.clone() })
        .await
        .unwrap();

    let mut running_patch = None;
    for _ in 0..100 {
        if let Some(hit) = mock
            .calls("PATCH", "/v1/runs/EXP-RUN-mock-1")
            .into_iter()
            .find(|body| body["status"] == "running")
        {
            running_patch = Some(hit);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(running_patch.is_some(), "mirror_state must PATCH status=running");

    // mirror_checkpoint (messages.rs, checkpoint artifact ingest).
    hub.artifacts
        .put(&keys::checkpoint_file(&run.0, 1, CHECKPOINT_MODEL_FILE), b"weights".to_vec())
        .await
        .unwrap();
    hub.on_message(
        &worker,
        wire::WorkerToCp::Artifact {
            seq: 2,
            job_id,
            class: wire::ArtifactClass::from("checkpoint"),
            uri: "ckpt-hot/t/step_1".into(),
            sha256: None,
            bytes: None,
            meta: Value::Null,
        },
    )
    .await
    .unwrap();

    let mut artifact_call = None;
    for _ in 0..100 {
        if let Some(hit) = mock.calls("POST", "/v1/runs/EXP-RUN-mock-1/artifacts").into_iter().next() {
            artifact_call = Some(hit);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let artifact_call = artifact_call.expect("mirror_checkpoint must post an artifact");
    assert_eq!(artifact_call["kind"], "checkpoint");
    assert_eq!(artifact_call["name"], "unit", "groups under the code unit name (no arch set)");
}

// ---- connection.rs: reconnect, unknown-worker detach, heartbeat sweep --

#[tokio::test]
async fn reap_stale_workers_older_than_reaps_a_silent_leased_worker_and_requeues_its_run() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-stale".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Assigned);

    // A zero preempt age makes any connected worker immediately eligible,
    // without waiting out the real HEARTBEAT_PREEMPT_TIMEOUT (spec §7) —
    // same test-seam shape as `reap_stuck_assignments_older_than`.
    hub.reap_stale_workers_older_than(0.0).await.unwrap();

    // No worker token bound: the reap detaches it as Leased, requeueing
    // its run exactly like an explicit `detach` would.
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Queued);
    assert!(matches!(
        hub.store.worker(&worker).await.unwrap().unwrap().state,
        WorkerState::Disconnected
    ));
}

#[tokio::test]
async fn reap_stale_workers_older_than_keeps_a_persistent_workers_run_assigned() {
    let hub = test_hub().await;
    let run = hub.submit("r", &shell_run(), None, None).await.unwrap();
    let (tx, _rx) = mpsc::unbounded_channel();
    let worker = WorkerId("w-persist-stale".into());
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();
    // A live worker token (chuk-compute M3.1) makes this worker's class
    // Persistent, so the reap's own class lookup (not just `detach`'s
    // caller-supplied class) must resolve it correctly.
    hub.store
        .create_worker_token("tok-stale", &worker, "mac", "cw_abcd1234", "hash-stale")
        .await
        .unwrap();

    hub.reap_stale_workers_older_than(0.0).await.unwrap();

    // The link is dropped (it reconnects and replays, M3.2), but the run
    // stays assigned rather than being requeued.
    assert_eq!(hub.store.run(&run).await.unwrap().unwrap().summary.state, RunState::Assigned);
}

#[tokio::test]
async fn run_reaper_loop_ticks_forever_driving_both_sweeps() {
    let hub = test_hub().await;
    // The loop itself always uses the real (long) timeouts, so this proves
    // it actually ticks and calls both sweeps without erroring or exiting —
    // not a particular reap outcome (covered by the `_older_than` seams).
    let handle = tokio::spawn(hub.clone().run_reaper_loop(std::time::Duration::from_millis(1)));
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    assert!(!handle.is_finished(), "the reaper loop must run forever, not exit after one tick");
    handle.abort();
}

#[tokio::test]
async fn detach_of_a_never_attached_worker_is_a_safe_no_op() {
    // Unknown to the store entirely (never `attach`ed) — distinct from the
    // "idle" case, which is attached but never assigned a run.
    let hub = test_hub().await;
    let worker = WorkerId("w-ghost".into());
    hub.detach(&worker, wire::WorkerClass::Leased).await.unwrap();
    assert!(hub.store.worker(&worker).await.unwrap().is_none());
}

#[tokio::test]
async fn send_to_a_connected_worker_delivers_the_message() {
    let hub = test_hub().await;
    let worker = WorkerId("w-linked".into());
    let (tx, mut rx) = mpsc::unbounded_channel();
    hub.attach(&worker, tx, &[], &Hardware::default()).await.unwrap();

    let msg = wire::CpToWorker::Cancel { job_id: wire::JobId::from("some-run") };
    assert!(hub.send_to(&worker, msg).await);
    assert!(matches!(rx.recv().await.expect("delivered"), wire::CpToWorker::Cancel { .. }));
}

#[tokio::test]
async fn reattaching_the_same_worker_id_replaces_its_link() {
    // A reconnect: a fresh socket re-attaches under the same worker id with
    // a new outbound channel; the stale one must stop receiving.
    let hub = test_hub().await;
    let worker = WorkerId("w-reconnect".into());
    let (tx1, mut rx1) = mpsc::unbounded_channel();
    hub.attach(&worker, tx1, &[], &Hardware::default()).await.unwrap();

    let (tx2, mut rx2) = mpsc::unbounded_channel();
    hub.attach(&worker, tx2, &[], &Hardware::default()).await.unwrap();

    let msg = wire::CpToWorker::Cancel { job_id: wire::JobId::from("some-run") };
    assert!(hub.send_to(&worker, msg).await);
    assert!(matches!(
        rx2.recv().await.expect("delivered on the new link"),
        wire::CpToWorker::Cancel { .. }
    ));
    assert!(rx1.try_recv().is_err(), "the replaced link must not receive anything");
}
