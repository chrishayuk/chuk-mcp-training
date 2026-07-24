//! Live worker connections + the scheduler.
//!
//! Scheduling is still M0-shaped (spec §14): FIFO queue, one job in flight per
//! worker, no packing, no leases in the scheduler. `pump()` runs whenever
//! capacity or work may have appeared: worker attach, run submission, run exit,
//! worker detach.
//!
//! Since chuk-compute M1 the worker speaks the compute-generic protocol
//! (`chuk-compute-wire`): the control plane translates a run's `RunSpec` into a
//! generic [`wire::Job`] (via [`crate::jobspec`]) on assignment, and interprets
//! the worker's generic [`wire::WorkerToCp::Artifact`] events back into
//! checkpoints — the lineage merge (code/seed/arch/parent/slices) that a worker
//! used to do now lives here, its correct home.
//!
//! Split by concern (mirrors `store/sqlite/`'s per-domain files): this file
//! holds the `Hub` struct, its constructor, and the shared test module;
//! [`mirror`], [`connection`], [`schedule`], [`messages`], [`submit`], and
//! [`control`] each add one `impl Hub` block for their slice of behaviour.

mod connection;
mod control;
mod messages;
mod mirror;
mod schedule;
mod submit;

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use anyhow::{Context, Result};
use chuk_compute_wire as wire;
use chuk_train_proto::{
    keys, CheckpointMeta, EventKind, GateAction, GateInfo, Hardware, LeaseState, RunId, RunRecord, RunSpec,
    RunState, TrainSpec, WorkerId, WorkerInfo, WorkerState, CHECKPOINT_DIR_PREFIX,
    CHECKPOINT_META_FILE, CHECKPOINT_MODEL_FILE, EXIT_CODE_AGENT_ERROR, EXIT_CODE_CANCELLED,
    GATE_SCOPE_RUN, HEARTBEAT_PREEMPT_TIMEOUT, HEARTBEAT_TIMEOUT,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::artifacts::ArtifactStore;
use crate::datasets::Datasets;
use crate::experiments::Experiments;
use crate::gate;
use crate::grant::GrantTable;
use crate::jobspec::{self, DataStaging, ResumeStaging, ShardInput, TrainStaging};
use crate::store::Store;
use crate::sweep;

/// Sender half of a worker's outbound message queue; the websocket task owns the
/// receiving half and the socket itself.
pub type AgentTx = mpsc::UnboundedSender<wire::CpToWorker>;

/// Per-run slice bookkeeping for checkpoint lineage: where the current worker
/// session resumed from, and the slice history carried by that resume point.
/// Mirrors what a worker session used to track locally (spec §11.2 `slices`).
#[derive(Clone, Default)]
struct ResumeState {
    from_step: u64,
    base_slices: Vec<[u64; 2]>,
}

/// The dataset identity a run's most recently assembled job resolved
/// (spec §7.3) — read back by [`schedule`]'s `complete_meta` (via
/// [`messages`]) so `CheckpointMeta` carries the harness's resolved values,
/// not a trainer's guess.
#[derive(Clone)]
struct DataState {
    content_sha: String,
    plan_sha: Option<String>,
    /// `name@sha256:<content_sha>` (or `sha256:<content_sha>` when the run
    /// referenced a bare sha), the `CheckpointMeta.datasets` convention
    /// [`chuk_train_proto::CodeRef`]'s `Display` already uses for code.
    dataset_label: String,
}

pub struct Hub {
    pub store: Arc<dyn Store>,
    pub artifacts: Arc<dyn ArtifactStore>,
    /// chuk-experiments-server reporting mirror; `None` when unconfigured.
    experiments: Option<Arc<Experiments>>,
    /// chuk-datasets dispatch-time resolution (spec §6/§7.3); `None` when
    /// unconfigured — a run declaring `data:` then fails to dispatch.
    datasets: Option<Arc<Datasets>>,
    grants: GrantTable,
    links: Mutex<HashMap<WorkerId, AgentTx>>,
    resume_state: StdMutex<HashMap<RunId, ResumeState>>,
    data_state: StdMutex<HashMap<RunId, DataState>>,
    /// Highest streamed-event `seq` processed per worker (chuk-compute M3.2). A
    /// reconnecting worker replays its outbox; anything at or below its
    /// high-water here has already been applied and is dropped (spec §8).
    high_water: StdMutex<HashMap<WorkerId, u64>>,
}

impl Hub {
    pub fn new(
        store: Arc<dyn Store>,
        artifacts: Arc<dyn ArtifactStore>,
        experiments: Option<Arc<Experiments>>,
        datasets: Option<Arc<Datasets>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            artifacts,
            experiments,
            datasets,
            grants: GrantTable::default(),
            links: Mutex::new(HashMap::new()),
            resume_state: StdMutex::new(HashMap::new()),
            data_state: StdMutex::new(HashMap::new()),
            high_water: StdMutex::new(HashMap::new()),
        })
    }

    /// The highest streamed-event seq processed for a worker — echoed in HelloAck
    /// so a reconnecting worker replays only what follows.
    pub fn resume_high_water(&self, worker_id: &WorkerId) -> u64 {
        self.high_water
            .lock()
            .expect("high_water lock")
            .get(worker_id)
            .copied()
            .unwrap_or(0)
    }

    pub fn grants(&self) -> &GrantTable {
        &self.grants
    }
}

#[cfg(test)]
mod tests {
    use super::connection::should_reap;
    use super::messages::{kill_reason_state, run_id, step_from_uri};
    use super::schedule::{dataset_label, eligible_for_assignment};
    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::datasets::Datasets;
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
}
