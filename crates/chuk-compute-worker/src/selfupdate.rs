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
    let bytes = download_and_verify(url, expected_sha256).await?;
    replace_exe(&exe, &bytes)
}

/// The network half of [`download_and_replace`]: fetch `url` and verify it
/// hashes to `expected_sha256`. Split out so it is unit-testable against a
/// local server, without ever touching the filesystem or a real executable.
async fn download_and_verify(url: &str, expected_sha256: &str) -> Result<Vec<u8>> {
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
    Ok(bytes.to_vec())
}

/// The offline half of [`download_and_replace`]: atomically replace `exe` with
/// `bytes` by staging into a hidden sibling file first (so the rename lands on
/// the same filesystem) and swapping it in. Split out so it is unit-testable
/// against a throwaway file rather than the real running executable.
fn replace_exe(exe: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let dir = exe.parent().context("executable has no parent directory")?;
    let staging = dir.join(format!("{STAGING_DOTPREFIX}{}{STAGING_SUFFIX}", file_name(exe)));
    std::fs::write(&staging, bytes).with_context(|| format!("writing {}", staging.display()))?;
    make_executable(&staging)?;
    std::fs::rename(&staging, exe).with_context(|| format!("replacing {}", exe.display()))?;
    Ok(exe.to_owned())
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // A refused loopback address, so error paths fail fast and
    // deterministically without a network (mirrors inputs.rs's convention).
    const REFUSED_ORIGIN: &str = "http://127.0.0.1:1/worker-bin";

    /// A minimal raw-socket HTTP/1.1 server returning `status`/`body` for one
    /// request, bound to an ephemeral port — self-contained so the download
    /// tests need no mock-server dependency (mirrors inputs.rs's `serve_once`).
    async fn serve_once(status: &'static str, body: &'static [u8]) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = socket.read(&mut buf).await;
            let header = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            let _ = socket.write_all(header.as_bytes()).await;
            let _ = socket.write_all(body).await;
            let _ = socket.shutdown().await;
        });
        format!("http://{addr}/worker-bin")
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("chuk-selfupdate-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn staging_name_is_a_hidden_sibling() {
        assert_eq!(file_name(Path::new("/usr/local/bin/chuk-compute-worker")), "chuk-compute-worker");
        assert_eq!(file_name(Path::new("worker")), "worker");
    }

    #[tokio::test]
    async fn download_and_verify_returns_the_bytes_when_the_hash_matches() {
        let body: &'static [u8] = b"pretend-worker-binary-bytes";
        let sha = hex::encode(Sha256::digest(body));
        let url = serve_once("200 OK", body).await;
        let bytes = download_and_verify(&url, &sha).await.unwrap();
        assert_eq!(bytes, body);
    }

    #[tokio::test]
    async fn download_and_verify_rejects_a_tampered_download_naming_both_digests() {
        let body: &'static [u8] = b"tampered bytes on the wire";
        let wrong = "0".repeat(64);
        let url = serve_once("200 OK", body).await;
        let error = download_and_verify(&url, &wrong).await.unwrap_err();
        let msg = format!("{error:#}");
        assert!(msg.contains("self-update checksum mismatch"), "{msg}");
        assert!(msg.contains(&wrong), "{msg}");
        assert!(msg.contains(&hex::encode(Sha256::digest(body))), "{msg}");
    }

    #[tokio::test]
    async fn download_and_verify_errors_with_context_on_a_non_2xx_status() {
        let url = serve_once("404 Not Found", b"nope").await;
        let error = download_and_verify(&url, &"0".repeat(64)).await.unwrap_err();
        assert!(format!("{error:#}").contains("downloading"));
    }

    #[tokio::test]
    async fn download_and_verify_errors_with_context_when_the_connection_is_refused() {
        let error = download_and_verify(REFUSED_ORIGIN, &"0".repeat(64)).await.unwrap_err();
        assert!(format!("{error:#}").contains("downloading"));
    }

    #[test]
    fn replace_exe_atomically_swaps_in_new_bytes_and_marks_it_executable() {
        let dir = scratch("replace");
        let exe = dir.join("fake-worker");
        std::fs::write(&exe, b"old binary bytes").unwrap();

        let replaced = replace_exe(&exe, b"new binary bytes").unwrap();
        assert_eq!(replaced, exe);
        assert_eq!(std::fs::read(&exe).unwrap(), b"new binary bytes");
        // The staging file was renamed over the target, not left behind.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&exe).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, EXECUTABLE_MODE);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replace_exe_rejects_an_exe_path_with_no_parent_directory() {
        let error = replace_exe(Path::new("/"), b"data").unwrap_err();
        assert!(format!("{error:#}").contains("no parent directory"));
    }

    #[test]
    fn replace_exe_errors_with_context_when_the_staging_write_fails() {
        let exe = Path::new("/no/such/directory/at/all/fake-worker");
        let error = replace_exe(exe, b"data").unwrap_err();
        assert!(format!("{error:#}").contains("writing"));
    }

    #[tokio::test]
    async fn download_and_replace_bails_before_touching_any_file_when_verification_fails() {
        // The public entry point, exercised end-to-end (current_exe lookup +
        // delegation) while staying safe: a hash that cannot match means
        // download_and_verify bails before replace_exe ever runs, so the real
        // running test binary is never touched.
        let body: &'static [u8] = b"a would-be worker binary";
        let url = serve_once("200 OK", body).await;
        let error = download_and_replace(&url, &"0".repeat(64)).await.unwrap_err();
        assert!(format!("{error:#}").contains("self-update checksum mismatch"));
    }

    #[tokio::test]
    async fn download_and_replace_propagates_a_download_failure() {
        let error = download_and_replace(REFUSED_ORIGIN, &"0".repeat(64)).await.unwrap_err();
        assert!(format!("{error:#}").contains("downloading"));
    }
}
