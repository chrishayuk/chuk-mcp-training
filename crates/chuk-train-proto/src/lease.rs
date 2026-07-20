//! Leases + provisioning (spec §3, §6). A lease is the contract between a
//! worker and its budget: a runtime wall that is enforced control-plane-side,
//! independent of the agent.

use serde::{Deserialize, Serialize};

use crate::domain::{UnixSeconds, WorkerId};

/// Where a worker's lease stands in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseState {
    /// Within budget, accepting jobs.
    Active,
    /// Past T-drain: finishing/checkpointing current work, no new jobs.
    Draining,
    /// Past T-0 and provider-verified gone (or torn down early).
    Destroyed,
}

impl LeaseState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Draining => "draining",
            Self::Destroyed => "destroyed",
        }
    }
}

/// One extension of a lease's wall (spec §3: a budget decision, recorded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeaseExtension {
    pub minutes: f64,
    pub at: UnixSeconds,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

/// The lease record (spec §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lease {
    pub worker_id: WorkerId,
    pub provider: String,
    /// Opaque provider instance id — what `destroy` and `status` act on.
    pub instance_id: String,
    /// Dollars per hour; colab leases price in compute units (recorded, not $).
    pub price_hr: f64,
    /// The granted runtime budget, in minutes.
    pub granted_min: f64,
    /// Minutes reserved at the end for checkpoint + upload before T-0.
    pub drain_window_min: f64,
    pub started_at: UnixSeconds,
    pub state: LeaseState,
    #[serde(default)]
    pub extensions: Vec<LeaseExtension>,
}

impl Lease {
    /// Total granted minutes including extensions.
    pub fn total_granted_min(&self) -> f64 {
        self.granted_min + self.extensions.iter().map(|e| e.minutes).sum::<f64>()
    }

    /// Seconds from `started_at` until T-0 (the hard wall).
    pub fn wall_secs(&self) -> f64 {
        self.total_granted_min() * 60.0
    }

    /// Seconds from `started_at` until T-drain (wall minus drain window).
    pub fn drain_secs(&self) -> f64 {
        (self.wall_secs() - self.drain_window_min * 60.0).max(0.0)
    }

    /// Minutes remaining until T-0, given the current time.
    pub fn remaining_min(&self, now: UnixSeconds) -> f64 {
        (self.wall_secs() - (now - self.started_at)) / 60.0
    }

    /// Projected cost of the whole lease so far (spec §8 ledger input).
    pub fn projected_cost(&self) -> f64 {
        self.price_hr * self.total_granted_min() / 60.0
    }
}

/// A rentable offer from a provider (spec §6 `provider_offers`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub id: String,
    pub provider: String,
    pub gpu: String,
    pub price_hr: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_gb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
}

/// A provisioned provider instance (before/independent of a lease record).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub id: String,
    pub provider: String,
    pub gpu: String,
    pub price_hr: f64,
}

/// Whether a provider instance is still billable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InstanceStatus {
    /// Alive and billing.
    Running,
    /// Provably gone (the goal state after destroy).
    Gone,
    /// The provider did not answer definitively; treat as still billing.
    Unknown,
}

/// `provision(...)` request payload (spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvisionRequest {
    pub provider: String,
    /// Requested runtime budget, in minutes (fractional allowed for testing).
    pub lease_min: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offer_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_price_hr: Option<f64>,
}

/// `provision(...)` result: the worker ref + its lease (spec §6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvisionResult {
    pub worker_id: WorkerId,
    pub lease: Lease,
    /// For Colab, the bootstrap cell text to paste; empty for API providers.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bootstrap: String,
}

/// A cost record in the ledger (spec §8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub ts: UnixSeconds,
    pub worker_id: WorkerId,
    pub provider: String,
    pub event: String,
    pub minutes: f64,
    pub cost: f64,
}

/// A spend cap (spec §8). `scope` is `global` or `provider:<name>`; `period`
/// is `month` (the current UTC calendar month) or `all` (all-time). Colab
/// budgets cap compute units rather than dollars (same numeric treatment).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub scope: String,
    pub cap: f64,
    pub period: String,
    pub updated_at: UnixSeconds,
}

/// `set_budget(...)` request (spec §6). Omitted period defaults to `month`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetBudgetRequest {
    pub scope: String,
    pub cap: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<String>,
}

/// Report from a teardown / lease-end (spec §6 Ack shape, richer here).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeardownResult {
    pub worker_id: WorkerId,
    /// Whether the provider confirmed the instance is gone.
    pub destroyed: bool,
    pub status: InstanceStatus,
}
