//! Build the worker's [`Capabilities`] self-description sent in the handshake.
//! Best-effort by design: fields we cannot cheaply determine (RAM, free disk)
//! are reported as `0` rather than pulling in heavy platform dependencies.

use std::collections::BTreeMap;

use chuk_compute_wire::{Accelerator, Capabilities, GpuInfo};
use tokio::process::Command;

/// The accelerator probe and how its output is shaped.
const NVIDIA_SMI: &str = "nvidia-smi";
const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
];
const NVIDIA_SMI_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Bytes per MiB — `nvidia-smi` reports VRAM in MiB; the wire wants bytes.
const BYTES_PER_MIB: u64 = 1_048_576;
/// Separator between a label's key and value (`k=v`); a bare label has none.
const LABEL_KV_SEPARATOR: char = '=';
/// Field separator in an `nvidia-smi` CSV row.
const GPU_FIELD_SEPARATOR: char = ',';
/// Fallback core count when the platform will not report parallelism.
const MIN_CPU_CORES: u32 = 1;

/// Detect this worker's capabilities. `labels` are the operator-supplied
/// scheduler hints (each `k=v` or a bare key); `preemptible` reflects whether
/// the host may be reclaimed under the worker (true for a leased/spot box).
pub async fn detect(labels: &[String], preemptible: bool) -> Capabilities {
    let accelerator = match query_nvidia().await {
        Some(gpu) => Accelerator::Cuda { devices: vec![gpu] },
        None => Accelerator::Cpu,
    };
    Capabilities {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        cpu_cores: cpu_cores(),
        // Best-effort: reported as 0 in M1 rather than adding a sysinfo dep.
        ram_bytes: 0,
        free_disk_bytes: 0,
        preemptible,
        accelerator,
        labels: parse_labels(labels),
    }
}

/// Available parallelism, clamped to at least one core.
fn cpu_cores() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(MIN_CPU_CORES)
}

/// Split each `k=v` label into a map entry; a bare label becomes a key with an
/// empty value.
fn parse_labels(labels: &[String]) -> BTreeMap<String, String> {
    labels
        .iter()
        .map(|label| match label.split_once(LABEL_KV_SEPARATOR) {
            Some((key, value)) => (key.to_owned(), value.to_owned()),
            None => (label.clone(), String::new()),
        })
        .collect()
}

/// Probe the first CUDA device via `nvidia-smi`; `None` when there is no GPU or
/// the tool is absent/slow.
async fn query_nvidia() -> Option<GpuInfo> {
    let output = tokio::time::timeout(
        NVIDIA_SMI_TIMEOUT,
        Command::new(NVIDIA_SMI).args(NVIDIA_SMI_ARGS).output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_gpu_line(stdout.lines().next()?)
}

/// Parse one `nvidia-smi` CSV row into a [`GpuInfo`]:
/// `Tesla T4, 15360 MiB, 550.54.15`.
fn parse_gpu_line(line: &str) -> Option<GpuInfo> {
    let mut parts = line.split(GPU_FIELD_SEPARATOR).map(str::trim);
    let name = parts.next().filter(|s| !s.is_empty())?.to_owned();
    let vram_bytes = parts.next().and_then(parse_mib).unwrap_or(0) * BYTES_PER_MIB;
    let driver_version = parts.next().filter(|s| !s.is_empty()).map(str::to_owned);
    Some(GpuInfo {
        name,
        vram_bytes,
        driver_version,
        cuda_version: None,
    })
}

/// `"15360 MiB"` → `15360`.
fn parse_mib(raw: &str) -> Option<u64> {
    raw.split_whitespace().next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_labels_handles_kv_and_bare() {
        let labels = vec!["site=colab".to_owned(), "t4".to_owned(), "k=v=w".to_owned()];
        let map = parse_labels(&labels);
        assert_eq!(map.get("site"), Some(&"colab".to_owned()));
        assert_eq!(map.get("t4"), Some(&String::new()));
        // split_once keeps everything after the first `=` as the value.
        assert_eq!(map.get("k"), Some(&"v=w".to_owned()));
    }

    #[test]
    fn parse_labels_empty_is_empty_map() {
        assert!(parse_labels(&[]).is_empty());
    }

    #[test]
    fn parse_mib_reads_the_leading_integer() {
        assert_eq!(parse_mib("15360 MiB"), Some(15360));
        assert_eq!(parse_mib("512"), Some(512));
        assert_eq!(parse_mib("not a number"), None);
        assert_eq!(parse_mib(""), None);
    }

    #[test]
    fn parse_gpu_line_maps_mib_to_bytes() {
        let gpu = parse_gpu_line("Tesla T4, 15360 MiB, 550.54.15").unwrap();
        assert_eq!(gpu.name, "Tesla T4");
        assert_eq!(gpu.vram_bytes, 15360 * BYTES_PER_MIB);
        assert_eq!(gpu.driver_version.as_deref(), Some("550.54.15"));
        assert!(gpu.cuda_version.is_none());
    }

    #[test]
    fn parse_gpu_line_tolerates_missing_trailing_fields() {
        let gpu = parse_gpu_line("Some GPU").unwrap();
        assert_eq!(gpu.name, "Some GPU");
        assert_eq!(gpu.vram_bytes, 0);
        assert!(gpu.driver_version.is_none());
        // An empty leading field is not a GPU.
        assert!(parse_gpu_line("").is_none());
        assert!(parse_gpu_line("  , 1 MiB, drv").is_none());
    }

    #[test]
    fn cpu_cores_is_at_least_one() {
        assert!(cpu_cores() >= MIN_CPU_CORES);
    }

    #[tokio::test]
    async fn detect_fills_os_arch_labels_and_flags() {
        let caps = detect(&["site=ci".to_owned()], true).await;
        assert_eq!(caps.os, std::env::consts::OS);
        assert_eq!(caps.arch, std::env::consts::ARCH);
        assert!(caps.cpu_cores >= MIN_CPU_CORES);
        assert!(caps.preemptible);
        assert_eq!(caps.labels.get("site"), Some(&"ci".to_owned()));
        // CI has no GPU, so the accelerator falls back to Cpu.
        if let Accelerator::Cuda { devices } = &caps.accelerator {
            assert!(!devices.is_empty());
        }
    }
}
