//! Domain types shared by the control plane, the agent, and the REST API.

use serde::{Deserialize, Serialize};

use crate::constants::{DEFAULT_SHELL_TIMEOUT, DEFAULT_TRAIN_TIMEOUT};

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

/// Where a checkpoint's canonical bytes currently live (spec §11.5). A hot copy
/// is on R2 under `ckpt-hot/` (short lifecycle); a promoted final is on R2 under
/// `ckpt-final/` (longer lifecycle); `drive` means it has been archived to
/// Google Drive (the durable copy) and the R2 copy may have expired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointLocation {
    #[default]
    R2Hot,
    R2Final,
    Drive,
}

impl CheckpointLocation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::R2Hot => "r2_hot",
            Self::R2Final => "r2_final",
            Self::Drive => "drive",
        }
    }
}

/// Access role, declared least→most privileged so `role >= Role::Admin` is a
/// clean min-role check. read = view; write = submit/manage runs; admin = team
/// admin (archive/retention + manage the team's users + API keys); sysadmin =
/// everything, across teams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    #[default]
    Read,
    Write,
    Admin,
    Sysadmin,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
            Self::Sysadmin => "sysadmin",
        }
    }
}

/// A member of a team with a role (RBAC).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct User {
    pub email: String,
    pub team_id: String,
    pub role: Role,
    pub created_at: UnixSeconds,
}

/// A team. A single "default" team for now; the seam for multi-tenant later —
/// `users` and `api_keys` already carry `team_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Team {
    pub id: String,
    pub name: String,
    pub created_at: UnixSeconds,
}

/// Connection state of a worker as the control plane sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Connected,
    Disconnected,
}

/// Append-only lifecycle event names (spec §5.3: the provenance record).
/// The first block mirrors `RunState`; the rest are non-state milestones.
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
    // M1 milestones:
    Checkpoint,
    Sliced,
    Resumed,
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
/// self-describing: `{"kind": "shell", ...}` / `{"kind": "train", ...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunSpec {
    Shell(ShellSpec),
    // Boxed: TrainSpec is much larger than ShellSpec, so an unboxed variant
    // would bloat every RunSpec (clippy large_enum_variant).
    Train(Box<TrainSpec>),
}

impl RunSpec {
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Shell(_) => "shell",
            Self::Train(_) => "train",
        }
    }

    /// The wall-clock limit this run's slice should honour.
    pub fn timeout_s(&self) -> u64 {
        match self {
            Self::Shell(s) => s.timeout_s,
            Self::Train(t) => t.timeout_s,
        }
    }
}

/// A one-off shell command (M0 job kind).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellSpec {
    pub command: String,
    #[serde(default = "default_shell_timeout_s")]
    pub timeout_s: u64,
}

fn default_shell_timeout_s() -> u64 {
    DEFAULT_SHELL_TIMEOUT.as_secs()
}

fn default_train_timeout_s() -> u64 {
    DEFAULT_TRAIN_TIMEOUT.as_secs()
}

/// A train job (spec §5.1). Resolved: `code` is a concrete code-unit ref, not
/// the `repo`+`commit` sugar (that is resolved to a unit at submit time).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainSpec {
    /// The deployable code unit to run (spec §11.1).
    pub code: CodeRef,
    /// Named entrypoint from the unit's manifest (e.g. `train`).
    pub entrypoint: String,
    /// Config file path *within the code unit*, materialised to `$CHUK_CONFIG`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
    /// Config overrides, passed as JSON in `$CHUK_OVERRIDES`.
    #[serde(default)]
    pub overrides: serde_json::Value,
    /// Input artifacts the run reads (checkpoints, datasets, tokenizer).
    #[serde(default)]
    pub artifacts_in: Vec<ArtifactRef>,
    /// Checkpoint schedule + retention (spec §5.1 `checkpoint`).
    #[serde(default)]
    pub checkpoint: CheckpointPolicy,
    /// Seed for this run; exported as `$CHUK_SEED` and stamped into lineage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Architecture tag recorded in checkpoint lineage (spec §11.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// Wall-clock limit for one slice.
    #[serde(default = "default_train_timeout_s")]
    pub timeout_s: u64,
    /// Out-links to this experiment elsewhere (Weights & Biases, the
    /// experiments-server dashboard, …), surfaced on the dashboard's run view.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<RunLink>,
}

/// A labelled out-link shown on a run's dashboard page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunLink {
    /// Short kind used for the icon/accent: `wandb`, `exp`, `r2`, … (free-form).
    #[serde(default)]
    pub kind: String,
    pub label: String,
    pub url: String,
}

/// Reference to a code unit: a name plus its content hash (spec §11.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeRef {
    pub name: String,
    pub sha: String,
}

impl std::fmt::Display for CodeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@sha256:{}", self.name, self.sha)
    }
}

/// Typed, content-addressed artifact kinds (spec §11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Code,
    Env,
    Dataset,
    Checkpoint,
    Metrics,
    Logs,
    Deck,
}

/// Reference to an input artifact a run declares (spec §5.1 `artifacts_in`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub name: String,
    pub kind: ArtifactKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
}

/// Checkpoint schedule + retention policy (spec §5.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPolicy {
    /// Advisory: how often the trainer is expected to checkpoint. The harness
    /// uploads whatever the trainer marks `.ready`; it does not drive cadence.
    pub every_steps: u64,
    pub keep_last: u32,
    pub keep_every: u64,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            every_steps: 500,
            keep_last: 3,
            keep_every: 5000,
        }
    }
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
