//! Messages that cross process boundaries: the agent websocket (spec §7) and
//! the REST API payloads the MCP surface consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::domain::{
    ArtifactKind, CheckpointLocation, CodeRef, EventKind, Hardware, Role, RunId, RunSpec, RunState,
    UnixSeconds, WorkerId, WorkerState,
};
use crate::lease::Lease;
use crate::manifest::{CheckpointMeta, CodeUnitManifest};

// ---------------------------------------------------------------------------
// REST API payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitShellRequest {
    pub name: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitRunResponse {
    pub run_id: RunId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: WorkerId,
    pub labels: Vec<String>,
    pub hardware: Hardware,
    pub state: WorkerState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_run: Option<RunId>,
    pub joined_at: UnixSeconds,
    pub last_seen: UnixSeconds,
    pub heartbeat_age_s: f64,
    /// The worker's lease, if it was provisioned under one (spec §3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<Lease>,
}

/// A worker's latest host-telemetry sample (chuk-compute M4): the `sys/*` metric
/// map (GPU/CPU/memory utilisation, VRAM, temperature, power) as of `sampled_at`.
/// One sample per worker — the live values a dashboard renders as gauges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerTelemetry {
    pub worker_id: WorkerId,
    pub sampled_at: UnixSeconds,
    pub values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: RunId,
    pub name: String,
    pub kind: String,
    pub state: RunState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<WorkerId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    /// The experiments-server *logical run* this execution belongs to, if any
    /// (its `RUN-…` id). Our own id (`EXEC-…`) names the execution attempt; this
    /// is the external parent reference to the research run it realises. `None`
    /// for an unattached scratch run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_ref: Option<String>,
    /// Email of the user who submitted this run (resolved from their session or
    /// the owning email of the API key they used — never a bare key prefix).
    /// `None` for runs submitted before this was tracked, or via the legacy
    /// master token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    pub created_at: UnixSeconds,
    pub updated_at: UnixSeconds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    #[serde(flatten)]
    pub summary: RunSummary,
    pub spec: RunSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEvent {
    pub ts: UnixSeconds,
    pub event: EventKind,
    pub detail: serde_json::Value,
}

/// A pending or previously-failed experiments-server mirror event, as read back
/// from the outbox for retry. `payload` is opaque serialized JSON (an
/// `experiments::OutboxEvent`) — the store never needs to know its shape, only
/// `kind` (a human-readable label for logging) and `run_id` (which run it's for).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutboxRow {
    pub id: i64,
    pub run_id: RunId,
    pub kind: String,
    pub payload: String,
    pub attempts: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogsResponse {
    pub run_id: RunId,
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiError {
    pub error: String,
}

// ---------------------------------------------------------------------------
// M1 REST payloads: code units, train submission, metrics, checkpoints
// ---------------------------------------------------------------------------

/// `build_code_unit(repo, commit)` — spec §6. `repo` may be a git URL or a
/// local path; `commit` is optional (defaults to the working tree / HEAD).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildCodeUnitRequest {
    pub repo: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeUnitInfo {
    pub code: CodeRef,
    pub manifest: CodeUnitManifest,
    pub uri: String,
    pub created_at: UnixSeconds,
}

/// `submit_run(spec)` — spec §6. The JobSpec: a name plus what to run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitRunRequest {
    pub name: String,
    pub spec: RunSpec,
    /// Optional external parent: the experiments-server *logical run* (`RUN-…`)
    /// this execution realises. When set, the reporting mirror reports *into*
    /// that run instead of minting a new one. Omit for an unattached scratch run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub run_id: RunId,
    pub step: u64,
    pub uri: String,
    pub model_hash: String,
    pub pinned: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_name: Option<String>,
    pub meta: CheckpointMeta,
    pub created_at: UnixSeconds,
    /// Where the canonical bytes live now: R2 hot/final or Drive (spec §11.5).
    #[serde(default)]
    pub location: CheckpointLocation,
    /// When this checkpoint was archived to Drive (None until archived).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archived_at: Option<UnixSeconds>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinCheckpointRequest {
    pub step: u64,
    pub name: String,
}

/// An API key's metadata for display — **never** the hash or plaintext (the
/// plaintext is shown once at creation via [`CreatedApiKey`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub team_id: String,
    pub created_by: String,
    pub name: String,
    /// Short display prefix (e.g. `ck_a1b2c3d4`); enough to recognise, not use.
    pub prefix: String,
    pub role: Role,
    pub created_at: UnixSeconds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<UnixSeconds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<UnixSeconds>,
}

/// Request to mint a new API key (admin-scoped; role must be ≤ the creator's).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    pub role: Role,
}

/// The one-time response with the plaintext key — shown once, never stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreatedApiKey {
    pub key: String,
    pub info: ApiKeyInfo,
}

/// A persistent worker token's metadata for display (chuk-compute M3.1) —
/// **never** the hash or plaintext (the plaintext is shown once at creation via
/// [`CreatedWorkerToken`]). Bound to a stable `worker_id` minted at creation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerTokenInfo {
    pub id: String,
    pub worker_id: WorkerId,
    pub name: String,
    /// Short display prefix (e.g. `cw_a1b2c3d4`); enough to recognise, not use.
    pub prefix: String,
    pub created_at: UnixSeconds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_used_at: Option<UnixSeconds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<UnixSeconds>,
}

/// Request to mint a new persistent worker token (admin-scoped infrastructure).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateWorkerTokenRequest {
    pub name: String,
}

/// The one-time response with the plaintext worker token — shown once, never
/// stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreatedWorkerToken {
    pub token: String,
    pub info: WorkerTokenInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub step: u64,
    pub value: f64,
}

/// `run_metrics(run_id, keys, since_step, downsample)` result (spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSeries {
    pub run_id: RunId,
    /// metric key → points, ascending by step.
    pub series: BTreeMap<String, Vec<MetricPoint>>,
}

/// `artifact_url(name, ttl)` result — a time-limited fetch URL (spec §6, §10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedUrl {
    pub url: String,
    pub expires_at: UnixSeconds,
}

/// One row in a general artifact listing (`list_artifacts`, spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactInfo {
    pub name: String,
    pub kind: ArtifactKind,
    pub sha: String,
    pub uri: String,
    pub created_at: UnixSeconds,
}

// ---------------------------------------------------------------------------
// M2 REST payloads: leases, provisioning, spend
// ---------------------------------------------------------------------------

/// `extend_lease(worker_id, minutes, reason)` request (spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtendLeaseRequest {
    pub minutes: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

/// `teardown(worker_id, force)` request (spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeardownRequest {
    #[serde(default)]
    pub force: bool,
}

/// A spend summary line, per provider (spec §8, minimal for M2's ledger).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpendLine {
    pub provider: String,
    pub committed: f64,
    pub spent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpendReport {
    pub lines: Vec<SpendLine>,
    pub total_committed: f64,
    pub total_spent: f64,
}

/// A ready-to-paste Colab bootstrap cell (spec §6: Colab `provision` returns
/// cell text). The control plane fills in its own URL + join token, so the
/// caller pastes it verbatim into a T4 notebook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColabCell {
    pub cell: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_spec_is_internally_tagged_and_defaults_timeout() {
        let spec: RunSpec =
            serde_json::from_str(r#"{"kind":"shell","command":"nvidia-smi"}"#).unwrap();
        let RunSpec::Shell(shell) = &spec else {
            panic!("expected shell")
        };
        assert_eq!(shell.command, "nvidia-smi");
        assert_eq!(
            shell.timeout_s,
            crate::constants::DEFAULT_SHELL_TIMEOUT.as_secs()
        );
    }

    #[test]
    fn train_spec_round_trips_and_flattens_kind() {
        let json = r#"{
            "kind": "train",
            "code": {"name": "cn7-trainer", "sha": "ab12"},
            "entrypoint": "train",
            "config": "configs/r1_1.yaml",
            "overrides": {"seed": 81},
            "seed": 81
        }"#;
        let spec: RunSpec = serde_json::from_str(json).unwrap();
        let RunSpec::Train(train) = &spec else {
            panic!("expected train")
        };
        assert_eq!(train.code.name, "cn7-trainer");
        assert_eq!(train.entrypoint, "train");
        assert_eq!(train.checkpoint.every_steps, 500); // default applied
                                                       // kind is injected back on serialize (internal tagging).
        assert_eq!(serde_json::to_value(&spec).unwrap()["kind"], "train");
    }
}
