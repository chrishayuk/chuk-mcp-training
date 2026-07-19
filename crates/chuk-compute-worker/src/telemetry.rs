//! Host telemetry sampler (chuk-compute M4): GPU + CPU + memory, streamed as a
//! `sys/*` metric namespace over the existing `Metric` channel — the same pipe
//! job metrics use, so no new wire message is needed.
//!
//! GPU is sampled by shelling out to `nvidia-smi` (as the capability probe
//! already does), *not* NVML-via-libloading: the distributed worker is a static
//! musl binary, which cannot `dlopen` `libnvidia-ml.so`, so a subprocess is the
//! portable choice and degrades cleanly to "no GPU metrics" when the tool is
//! absent. CPU + memory come from `sysinfo`. Apple-Silicon GPU (via `macmon`)
//! is a later addition; on a Mac today the CPU/memory metrics still flow.
//!
//! Everything is best-effort: a failed probe drops that sample's affected
//! metrics rather than erroring — telemetry must never disturb the session.

use std::collections::BTreeMap;

use sysinfo::System;
use tokio::process::Command;

/// Namespace every host metric carries, distinguishing it from a job's own
/// metrics. The control plane keys ingestion off the absent `job_id`; the prefix
/// keeps the two legible on a dashboard, and the test below holds the keys to it.
#[cfg(test)]
const SYS_PREFIX: &str = "sys/";

// Metric keys (each already prefixed). Utilisations are fractions in `[0, 1]`;
// byte counts are raw bytes; temperature is Celsius; power is watts.
const CPU_UTIL: &str = "sys/cpu_util";
const MEM_USED_BYTES: &str = "sys/mem_used_bytes";
const MEM_TOTAL_BYTES: &str = "sys/mem_total_bytes";
const MEM_UTIL: &str = "sys/mem_util";
const GPU_UTIL: &str = "sys/gpu_util";
const GPU_MEM_USED_BYTES: &str = "sys/gpu_mem_used_bytes";
const GPU_MEM_TOTAL_BYTES: &str = "sys/gpu_mem_total_bytes";
const GPU_MEM_UTIL: &str = "sys/gpu_mem_util";
const GPU_TEMP_C: &str = "sys/gpu_temp_c";
const GPU_POWER_W: &str = "sys/gpu_power_w";

/// `nvidia-smi` and the fields we sample. `nounits` gives bare numbers (util in
/// %, memory in MiB, temperature in C, power in W) so parsing stays trivial.
const NVIDIA_SMI: &str = "nvidia-smi";
const NVIDIA_SMI_SAMPLE_ARGS: [&str; 2] = [
    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
    "--format=csv,noheader,nounits",
];
const NVIDIA_SMI_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Bytes per MiB — `nvidia-smi` reports VRAM in MiB; the wire wants bytes.
const BYTES_PER_MIB: f64 = 1_048_576.0;
/// Percent → fraction, so utilisations land in `[0, 1]`.
const PERCENT: f64 = 100.0;
/// Field separator in an `nvidia-smi` CSV row.
const GPU_FIELD_SEPARATOR: char = ',';

/// Samples host telemetry. Holds the `sysinfo` handle (which needs to persist so
/// CPU deltas are meaningful) and remembers once GPU sampling has proven absent,
/// so a CPU-only host stops spawning a doomed `nvidia-smi` every tick.
pub struct Sampler {
    sys: System,
    gpu_absent: bool,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            sys: System::new(),
            gpu_absent: false,
        }
    }

    /// Take one sample: CPU + memory always, GPU when present. Never errors — a
    /// probe that fails simply omits its metrics.
    pub async fn sample(&mut self) -> BTreeMap<String, f64> {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let mut values = cpu_mem_metrics(
            self.sys.global_cpu_usage() as f64,
            self.sys.used_memory(),
            self.sys.total_memory(),
        );
        if !self.gpu_absent {
            match query_gpu().await {
                Some(line) => values.extend(parse_gpu_sample(&line)),
                // No GPU (or the tool is gone): stop trying for this process.
                None => self.gpu_absent = true,
            }
        }
        values
    }
}

/// Build the CPU + memory metrics from a percentage and raw byte counts.
fn cpu_mem_metrics(cpu_percent: f64, mem_used: u64, mem_total: u64) -> BTreeMap<String, f64> {
    let mut values = BTreeMap::new();
    values.insert(CPU_UTIL.to_owned(), (cpu_percent / PERCENT).clamp(0.0, 1.0));
    values.insert(MEM_USED_BYTES.to_owned(), mem_used as f64);
    values.insert(MEM_TOTAL_BYTES.to_owned(), mem_total as f64);
    if mem_total > 0 {
        values.insert(MEM_UTIL.to_owned(), mem_used as f64 / mem_total as f64);
    }
    values
}

/// Run the `nvidia-smi` sample query; `None` when there is no GPU or the tool is
/// absent/slow.
async fn query_gpu() -> Option<String> {
    let output = tokio::time::timeout(
        NVIDIA_SMI_TIMEOUT,
        Command::new(NVIDIA_SMI)
            .args(NVIDIA_SMI_SAMPLE_ARGS)
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // First device only for now; multi-GPU keys (sys/gpu0_*) are a later add.
    stdout.lines().next().map(str::to_owned)
}

/// Parse one `nvidia-smi` sample row —
/// `utilization, mem.used, mem.total, temp, power` (nounits) e.g.
/// `45, 8192, 15360, 62, 70.50` — into `sys/gpu_*` metrics. Every field is
/// optional: a missing or `[N/A]` value just omits its metric.
fn parse_gpu_sample(line: &str) -> BTreeMap<String, f64> {
    let mut fields = line.split(GPU_FIELD_SEPARATOR).map(str::trim);
    let mut values = BTreeMap::new();
    let mut next = || fields.next().and_then(|f| f.parse::<f64>().ok());

    if let Some(util) = next() {
        values.insert(GPU_UTIL.to_owned(), (util / PERCENT).clamp(0.0, 1.0));
    }
    let used_mib = next();
    let total_mib = next();
    if let Some(used) = used_mib {
        values.insert(GPU_MEM_USED_BYTES.to_owned(), used * BYTES_PER_MIB);
    }
    if let Some(total) = total_mib {
        values.insert(GPU_MEM_TOTAL_BYTES.to_owned(), total * BYTES_PER_MIB);
    }
    if let (Some(used), Some(total)) = (used_mib, total_mib) {
        if total > 0.0 {
            values.insert(GPU_MEM_UTIL.to_owned(), used / total);
        }
    }
    if let Some(temp) = next() {
        values.insert(GPU_TEMP_C.to_owned(), temp);
    }
    if let Some(power) = next() {
        values.insert(GPU_POWER_W.to_owned(), power);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_metric_key_is_sys_namespaced() {
        for key in [
            CPU_UTIL,
            MEM_USED_BYTES,
            MEM_TOTAL_BYTES,
            MEM_UTIL,
            GPU_UTIL,
            GPU_MEM_USED_BYTES,
            GPU_MEM_TOTAL_BYTES,
            GPU_MEM_UTIL,
            GPU_TEMP_C,
            GPU_POWER_W,
        ] {
            assert!(key.starts_with(SYS_PREFIX), "{key} is not sys/-namespaced");
        }
    }

    #[test]
    fn cpu_mem_metrics_normalise_and_derive_util() {
        let v = cpu_mem_metrics(42.5, 4_000, 8_000);
        assert_eq!(v[CPU_UTIL], 0.425);
        assert_eq!(v[MEM_USED_BYTES], 4_000.0);
        assert_eq!(v[MEM_TOTAL_BYTES], 8_000.0);
        assert_eq!(v[MEM_UTIL], 0.5);
    }

    #[test]
    fn cpu_util_clamps_and_zero_total_omits_util() {
        // Some platforms momentarily report >100% aggregate CPU.
        assert_eq!(cpu_mem_metrics(150.0, 0, 0)[CPU_UTIL], 1.0);
        assert!(!cpu_mem_metrics(10.0, 0, 0).contains_key(MEM_UTIL));
    }

    #[test]
    fn parse_gpu_sample_maps_all_fields() {
        let v = parse_gpu_sample("45, 8192, 15360, 62, 70.50");
        assert_eq!(v[GPU_UTIL], 0.45);
        assert_eq!(v[GPU_MEM_USED_BYTES], 8192.0 * BYTES_PER_MIB);
        assert_eq!(v[GPU_MEM_TOTAL_BYTES], 15360.0 * BYTES_PER_MIB);
        assert!((v[GPU_MEM_UTIL] - 8192.0 / 15360.0).abs() < 1e-9);
        assert_eq!(v[GPU_TEMP_C], 62.0);
        assert_eq!(v[GPU_POWER_W], 70.5);
    }

    #[tokio::test]
    async fn sampler_reports_cpu_and_memory_on_any_host() {
        // Exercises the real sysinfo path (the pure fns above don't): CPU and
        // memory are present on every platform. GPU is host-dependent, so this
        // asserts only the always-available metrics.
        let mut sampler = Sampler::new();
        let values = sampler.sample().await;
        assert!(values.contains_key(CPU_UTIL));
        assert!(values.contains_key(MEM_TOTAL_BYTES));
        assert!(values[MEM_TOTAL_BYTES] > 0.0);
    }

    #[test]
    fn parse_gpu_sample_tolerates_na_and_short_rows() {
        // A `[N/A]` power field (common on Colab T4) just drops that one metric.
        let v = parse_gpu_sample("30, 1024, 2048, 55, [N/A]");
        assert_eq!(v[GPU_UTIL], 0.30);
        assert_eq!(v[GPU_TEMP_C], 55.0);
        assert!(!v.contains_key(GPU_POWER_W));
        // A truncated row yields only what parsed; total missing ⇒ no mem_util.
        let short = parse_gpu_sample("10, 512");
        assert_eq!(short[GPU_MEM_USED_BYTES], 512.0 * BYTES_PER_MIB);
        assert!(!short.contains_key(GPU_MEM_UTIL));
        // Nothing parseable ⇒ empty.
        assert!(parse_gpu_sample("").is_empty());
    }
}
