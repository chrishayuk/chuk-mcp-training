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
    RunState, TrainSpec, WorkerId, WorkerInfo, WorkerState, ASSIGNMENT_STUCK_TIMEOUT,
    CHECKPOINT_DIR_PREFIX, CHECKPOINT_META_FILE, CHECKPOINT_MODEL_FILE, EXIT_CODE_AGENT_ERROR,
    EXIT_CODE_CANCELLED, GATE_SCOPE_RUN, HEARTBEAT_PREEMPT_TIMEOUT, HEARTBEAT_TIMEOUT,
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
mod tests;
