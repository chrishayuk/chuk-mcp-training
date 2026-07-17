//! Messages that cross process boundaries: the agent websocket (spec §7) and
//! the REST API payloads the MCP surface consumes.

use serde::{Deserialize, Serialize};

use crate::domain::{
    EventKind, Hardware, RunId, RunSpec, RunState, UnixSeconds, WorkerId, WorkerState,
};

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
    fn run_spec_is_internally_tagged_and_defaults_timeout() {
        let spec: RunSpec =
            serde_json::from_str(r#"{"kind":"shell","command":"nvidia-smi"}"#).unwrap();
        let RunSpec::Shell { command, timeout_s } = spec;
        assert_eq!(command, "nvidia-smi");
        assert_eq!(timeout_s, crate::constants::DEFAULT_SHELL_TIMEOUT.as_secs());
    }

    #[test]
    fn assign_message_shape() {
        let msg = CpToAgent::Assign {
            job: JobAssignment {
                run_id: RunId::from("r1"),
                spec: RunSpec::Shell {
                    command: "echo hi".into(),
                    timeout_s: 5,
                },
            },
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["type"], "assign");
        assert_eq!(json["job"]["spec"]["kind"], "shell");
    }
}
