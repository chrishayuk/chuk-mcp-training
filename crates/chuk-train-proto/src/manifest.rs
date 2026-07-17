//! On-artifact JSON schemas: the code-unit manifest (`unit.toml`) and the
//! checkpoint lineage sidecar (`meta.json`, spec §11.2).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::domain::{CodeRef, RunId};

/// Parsed `unit.toml` (spec §11.1). Names the deployable unit and its runnable
/// entrypoints; `requires` carries default job requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeUnitManifest {
    pub name: String,
    #[serde(default)]
    pub version: String,
    /// entrypoint name → shell command, e.g. `{"train": "python train.py"}`.
    pub entrypoints: BTreeMap<String, String>,
    #[serde(default)]
    pub python: Option<String>,
    #[serde(default)]
    pub requires: UnitRequires,
}

impl CodeUnitManifest {
    pub fn entrypoint(&self, name: &str) -> Option<&str> {
        self.entrypoints.get(name).map(String::as_str)
    }
}

/// Default hardware requirements a unit declares (spec §11.1 `requires`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UnitRequires {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<u64>,
}

/// Lineage-complete checkpoint metadata (`meta.json`, spec §11.2).
///
/// The trainer writes a *partial* sidecar (typically `arch` + `tokenizer_hash`,
/// the facts only it knows); the harness fills the rest — code unit,
/// config hash, parent, datasets, run id, and slice bounds — before upload, so
/// every number on a panel is mechanically traceable (spec §10).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<CodeRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_hash: Option<String>,
    /// The tokenizer fingerprint lazarus verifies at load time (spec §10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<RunId>,
    /// `[from_step, to_step]` pairs across resumes (spec §11.2 `slices`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slices: Vec<[u64; 2]>,
}
