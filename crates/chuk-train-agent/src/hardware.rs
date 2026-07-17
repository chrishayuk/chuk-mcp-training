//! Best-effort hardware detection at registration time.

use chuk_train_proto::Hardware;
use tokio::process::Command;

const NVIDIA_SMI: &str = "nvidia-smi";
const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
];
const NVIDIA_SMI_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

pub async fn detect() -> Hardware {
    let mut hw = Hardware {
        host: hostname(),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        ..Hardware::default()
    };
    if let Some((gpu, vram_mb, driver)) = query_nvidia().await {
        hw.gpu = Some(gpu);
        hw.vram_mb = vram_mb;
        hw.driver = Some(driver);
    }
    hw
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| std::process::id().to_string())
}

/// Parse the first GPU line of `nvidia-smi --query-gpu=...`:
/// `Tesla T4, 15360 MiB, 550.54.15`
async fn query_nvidia() -> Option<(String, Option<u64>, String)> {
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
    let first = stdout.lines().next()?;
    let mut parts = first.split(',').map(str::trim);
    let name = parts.next()?.to_owned();
    let vram_mb = parts.next().and_then(parse_mib);
    let driver = parts.next()?.to_owned();
    Some((name, vram_mb, driver))
}

/// `"15360 MiB"` → `15360`
fn parse_mib(raw: &str) -> Option<u64> {
    raw.split_whitespace().next()?.parse().ok()
}
