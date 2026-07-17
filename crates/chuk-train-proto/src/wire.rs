//! Messages that cross process boundaries: the agent websocket (spec §7) and
//! the REST API payloads the MCP surface consumes.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::domain::{
    ArtifactKind, CodeRef, EventKind, Hardware, RunId, RunSpec, RunState, UnixSeconds, WorkerId,
    WorkerState,
};
use crate::manifest::{CheckpointMeta, CodeUnitManifest};

// ---------------------------------------------------------------------------
// Agent websocket (spec §7): worker → control plane
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentToCp {
    /// First message on every connection; the join token is exchanged for a
    /// worker identity.
    Register {
        token: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worker_id: Option<WorkerId>,
        #[serde(default)]
        labels: Vec<String>,
        hardware: Hardware,
    },
    Heartbeat,
    Log {
        run_id: RunId,
        line: String,
    },
    /// One parsed metrics record (spec §5.1: JSONL to `$CHUK_METRICS`).
    Metric {
        run_id: RunId,
        step: u64,
        values: BTreeMap<String, f64>,
    },
    /// A checkpoint the agent has finished uploading to the artifact store.
    /// The bytes went up out-of-band (REST, scoped grant); this records the
    /// lineage sidecar and the weights hash for provenance.
    Checkpoint {
        run_id: RunId,
        step: u64,
        model_hash: String,
        meta: CheckpointMeta,
    },
    JobStarted {
        run_id: RunId,
    },
    JobExited {
        run_id: RunId,
        code: i64,
    },
}

// ---------------------------------------------------------------------------
// Agent websocket (spec §7): control plane → worker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CpToAgent {
    Registered { worker_id: WorkerId },
    Rejected { reason: String },
    Assign { job: JobAssignment },
    Cancel { run_id: RunId },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobAssignment {
    pub run_id: RunId,
    pub spec: RunSpec,
    /// Present when this is a resumed train slice: where to pick up from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<ResumeInfo>,
    /// Present for train runs: the scoped capability to fetch inputs and
    /// upload checkpoints for this run (spec §7 `credentials(scoped)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant: Option<UploadGrant>,
}

/// Where a resumed slice continues from (spec §5.3 `resumed(from_step)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResumeInfo {
    pub from_step: u64,
    /// Store-relative path to the checkpoint directory to resume from,
    /// e.g. `runs/<run_id>/ckpt/step_<n>`. The agent fetches it via the grant.
    pub checkpoint_path: String,
}

/// A short-lived, run-scoped capability (spec §12: workers never hold provider
/// or admin credentials — only a token bound to one run's read/write needs).
/// The agent derives the REST base from its own control-plane URL, so the
/// grant carries only the bearer token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UploadGrant {
    pub token: String,
}

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinCheckpointRequest {
    pub step: u64,
    pub name: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_messages_round_trip_with_snake_case_tags() {
        let msg = AgentToCp::JobExited {
            run_id: RunId::from("abc123"),
            code: 0,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"job_exited""#), "{json}");
        assert_eq!(serde_json::from_str::<AgentToCp>(&json).unwrap(), msg);
    }

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

    #[test]
    fn assign_message_shape() {
        let msg = CpToAgent::Assign {
            job: JobAssignment {
                run_id: RunId::from("r1"),
                spec: RunSpec::Shell(crate::domain::ShellSpec {
                    command: "echo hi".into(),
                    timeout_s: 5,
                }),
                resume: None,
                grant: None,
            },
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "assign");
        assert_eq!(json["job"]["spec"]["kind"], "shell");
    }
}
