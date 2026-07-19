//! `chuk-compute-wire` — the compute-generic protocol between a worker and the
//! control plane of the chuk compute fabric (`docs/specs/chuk-compute-spec.md`).
//!
//! This crate is **substrate**, and three rules keep it that way:
//!
//! 1. **Serde-only.** No tokio, no transport, no control-plane internals — just
//!    the types that cross the wire. Both the CP and the worker depend on it; the
//!    worker depends on nothing else from the workspace.
//! 2. **No domain vocabulary.** No higher-layer workload noun appears here —
//!    `tests/no_domain_vocabulary.rs` fails the build if the canonical forbidden
//!    token leaks in. The workload model is generic: a [`Job`] stages inputs,
//!    runs a command, streams metrics, and collects outputs, and the workload
//!    *kind* rides as an opaque [`Template`] tag the worker never branches on.
//!    Domain concepts (checkpoints, runs, evals) live one layer up.
//! 3. **Forward-compatible by construction.** Additive fields carry
//!    `#[serde(default)]`, growable enums are `#[non_exhaustive]`, and no type
//!    uses `deny_unknown_fields`. Old workers tolerate new control planes and
//!    vice versa; the handshake gates only on [`PROTOCOL_VERSION`], bumped solely
//!    for breaking changes.
//!
//! The daemon that speaks this protocol is a **worker**, never an "agent" — that
//! word is reserved for the agentic workloads that run *on* the fabric.

mod blob;
mod capability;
mod constants;
mod ids;
mod job;
mod message;
mod telemetry;

pub use blob::{BlobMethod, BlobUrlRequest, BlobUrlResponse};
pub use capability::{Accelerator, Capabilities, GpuInfo, WorkerClass};
pub use constants::{
    API_PREFIX, DEFAULT_TELEMETRY_INTERVAL_SECS, DEFAULT_TERM_GRACE_SECS, PROTOCOL_VERSION,
    SANDBOX_PLACEHOLDER, SYS_METRIC_PREFIX,
};
pub use ids::{ArtifactClass, CampaignId, JobId, Template, WorkerId};
pub use job::{
    InputArtifact, Job, OutputRule, Placement, Readiness, RestartPolicy, ServiceRef, ServiceSpec,
    UploadPolicy,
};
pub use message::{CpToWorker, KillReason, Resume, WorkerToCp};
pub use telemetry::TelemetryConfig;

/// Fractional unix seconds — the fabric's single timestamp representation, so
/// wall deadlines and event times are byte-for-byte identical on both ends.
pub type UnixSeconds = f64;
