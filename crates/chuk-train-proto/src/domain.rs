//! Domain types shared by the control plane, the agent, and the REST API.

use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_SHELL_TIMEOUT;

macro_rules! string_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }
    };
}

string_id!(
    /// Identifier of a run (a queued/executing job).
    RunId
);
string_id!(
    /// Identifier of a worker (one agent process on one machine).
    WorkerId
);

/// Lifecycle state of a run (spec §5.3 vocabulary, M0 subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Queued,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl RunState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Assigned => "assigned",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Connection state of a worker as the control plane sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Connected,
    Disconnected,
}

/// Append-only lifecycle event names (spec §5.3: the provenance record).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Created,
    Queued,
    Assigned,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl From<RunState> for EventKind {
    fn from(state: RunState) -> Self {
        match state {
            RunState::Queued => Self::Queued,
            RunState::Assigned => Self::Assigned,
            RunState::Running => Self::Running,
            RunState::Completed => Self::Completed,
            RunState::Failed => Self::Failed,
            RunState::Cancelled => Self::Cancelled,
        }
    }
}

/// What a run actually executes. Internally tagged so the stored/wire JSON is
/// self-describing: `{"kind": "shell", "command": ..., "timeout_s": ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunSpec {
    Shell {
        command: String,
        #[serde(default = "default_shell_timeout_s")]
        timeout_s: u64,
    },
}

impl RunSpec {
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Shell { .. } => "shell",
        }
    }
}

fn default_shell_timeout_s() -> u64 {
    DEFAULT_SHELL_TIMEOUT.as_secs()
}

/// Hardware the agent detects at registration time.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hardware {
    pub host: String,
    pub os: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

/// Unix timestamp in seconds. One alias so the choice is written down once.
pub type UnixSeconds = f64;
