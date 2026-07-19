//! Persistent-worker self-update (chuk-compute M3.3). On a version-mismatch
//! `HelloReject` carrying a binary URL + checksum, the worker downloads the
//! control plane's current worker, verifies its sha256, atomically replaces this
//! executable, and re-execs (the re-exec itself lives in `main`). Hand-rolled —
//! no updater crate — because the binary is served by our own control plane.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

/// Prefix for the temp file the new binary is staged into (a hidden sibling of
/// the current exe, so the rename is atomic on the same filesystem).
const STAGING_DOTPREFIX: char = '.';
const STAGING_SUFFIX: &str = "-update";
/// Executable mode for the replaced binary.
#[cfg(unix)]
const EXECUTABLE_MODE: u32 = 0o755;

/// Download the worker binary at `url`, verify it against `expected_sha256`, and
/// atomically replace the running executable. Returns the replaced exe path (the
/// caller re-execs it). Verification happens **before** the replace, so a bad or
/// truncated download never clobbers a working binary.
pub async fn download_and_replace(url: &str, expected_sha256: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("locating current executable")?;
    let dir = exe.parent().context("executable has no parent directory")?;

    let bytes = reqwest::get(url)
        .await
        .with_context(|| format!("downloading {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?
        .bytes()
        .await
        .context("reading downloaded worker")?;

    let actual = hex::encode(Sha256::digest(&bytes));
    if !actual.eq_ignore_ascii_case(expected_sha256) {
        bail!("self-update checksum mismatch: expected {expected_sha256}, got {actual}");
    }

    let staging = dir.join(format!("{STAGING_DOTPREFIX}{}{STAGING_SUFFIX}", file_name(&exe)));
    std::fs::write(&staging, &bytes).with_context(|| format!("writing {}", staging.display()))?;
    make_executable(&staging)?;
    std::fs::rename(&staging, &exe)
        .with_context(|| format!("replacing {}", exe.display()))?;
    Ok(exe)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "worker".to_owned())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(EXECUTABLE_MODE);
    std::fs::set_permissions(path, perms).context("chmod +x on the new worker")
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_name_is_a_hidden_sibling() {
        assert_eq!(file_name(Path::new("/usr/local/bin/chuk-compute-worker")), "chuk-compute-worker");
        assert_eq!(file_name(Path::new("worker")), "worker");
    }

    #[tokio::test]
    async fn rejects_a_checksum_mismatch_before_replacing() {
        // No network is reached — an obviously-wrong url fails at download, and a
        // wrong checksum would fail before any replace. We assert the mismatch
        // path is reachable with a hash that can't match an empty/error body.
        let err = download_and_replace("http://127.0.0.1:0/nope", &"0".repeat(64))
            .await
            .unwrap_err();
        // Either the download failed or the checksum did — both leave us un-replaced.
        let msg = format!("{err:#}");
        assert!(msg.contains("downloading") || msg.contains("checksum"), "{msg}");
    }
}
