//! What a worker advertises about itself at handshake, and how the control
//! plane classifies it. The scheduler may match against these; the wire does
//! not interpret the free-form label map.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// How the control plane owns a worker's lifecycle. An enum, not a flag, so that
/// destroying a [`WorkerClass::Persistent`] worker is unrepresentable in
/// control-plane code rather than merely checked at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkerClass {
    /// Control-plane-provisioned, joins with a single-use token, carries a hard
    /// wall from its lease, and is destroyed by the control plane at wall end.
    Leased,
    /// Self-enrolled (e.g. a machine you own), long-lived revocable token, no
    /// wall, never destroyed, reconnects forever.
    Persistent,
}

/// The compute a worker offers. `Cuda` carries per-device detail; `Mps` is Apple
/// Silicon unified memory; `Cpu` is no accelerator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Accelerator {
    Cuda { devices: Vec<GpuInfo> },
    Mps { chip: String, unified_memory_bytes: u64 },
    Cpu,
}

/// One CUDA device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<String>,
}

/// The worker's self-description, sent once in the handshake.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub os: String,
    pub arch: String,
    pub cpu_cores: u32,
    pub ram_bytes: u64,
    pub free_disk_bytes: u64,
    /// Whether the host may be reclaimed under the worker at any time (a rented
    /// spot box); persistent hardware sets this false.
    pub preemptible: bool,
    pub accelerator: Accelerator,
    /// Free-form scheduler hints (e.g. `site=colab`, `site=home`). The wire does
    /// not interpret these; the control plane may match jobs against them.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_class_is_snake_case() {
        assert_eq!(serde_json::to_string(&WorkerClass::Leased).unwrap(), r#""leased""#);
        assert_eq!(
            serde_json::from_str::<WorkerClass>(r#""persistent""#).unwrap(),
            WorkerClass::Persistent
        );
    }

    #[test]
    fn accelerator_variants_round_trip_tagged() {
        let cuda = Accelerator::Cuda {
            devices: vec![GpuInfo {
                name: "Tesla T4".into(),
                vram_bytes: 16_106_127_360,
                driver_version: Some("535.104".into()),
                cuda_version: None,
            }],
        };
        let value = serde_json::to_value(&cuda).unwrap();
        assert_eq!(value["kind"], "cuda");
        // `cuda_version` was None → skipped on the wire.
        assert!(value["devices"][0].get("cuda_version").is_none());
        assert_eq!(serde_json::from_value::<Accelerator>(value).unwrap(), cuda);

        for acc in [
            Accelerator::Mps { chip: "Apple M2".into(), unified_memory_bytes: 1 << 35 },
            Accelerator::Cpu,
        ] {
            let round = serde_json::from_str(&serde_json::to_string(&acc).unwrap()).unwrap();
            assert_eq!(acc, round);
        }
    }

    #[test]
    fn capabilities_round_trip_and_labels_default() {
        // A payload omitting `labels` deserialises to an empty map (serde default).
        let json = r#"{"os":"linux","arch":"x86_64","cpu_cores":8,"ram_bytes":34359738368,
            "free_disk_bytes":1000,"preemptible":true,"accelerator":{"kind":"cpu"}}"#;
        let caps: Capabilities = serde_json::from_str(json).unwrap();
        assert!(caps.labels.is_empty());
        assert!(caps.preemptible);
        assert_eq!(caps.accelerator, Accelerator::Cpu);

        let mut with_labels = caps.clone();
        with_labels.labels.insert("site".into(), "home".into());
        let round: Capabilities =
            serde_json::from_str(&serde_json::to_string(&with_labels).unwrap()).unwrap();
        assert_eq!(round, with_labels);
    }
}
