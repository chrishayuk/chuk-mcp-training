//! Stage a job's input artifacts into the sandbox before the command runs.
//! Each input is fetched (from a direct URL or a grant-authed store key),
//! optionally hash-verified, then either written as a file or unpacked as an
//! archive into its destination.

use std::path::Path;

use anyhow::{bail, Context, Result};
use chuk_compute_wire::InputArtifact;
use sha2::{Digest, Sha256};

use crate::httpclient::HttpClient;
use crate::sandbox::subst;

const HTTP_SCHEME: &str = "http://";
const HTTPS_SCHEME: &str = "https://";
/// Leading bytes of a zstd frame (magic `0xFD2FB528`, little-endian on disk).
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

/// Fetch, verify, and place one input into the sandbox at `sandbox_path`.
pub async fn stage(input: &InputArtifact, sandbox_path: &str, client: &HttpClient) -> Result<()> {
    let bytes = fetch(&input.uri, client)
        .await
        .with_context(|| format!("fetching input {}", input.uri))?;
    place(input, &bytes, sandbox_path).await
}

/// The offline half of [`stage`]: verify the hash then write or unpack the
/// already-fetched `bytes`. Split out so the placement logic is unit-testable
/// without a network fetch.
async fn place(input: &InputArtifact, bytes: &[u8], sandbox_path: &str) -> Result<()> {
    if let Some(expected) = &input.sha256 {
        verify_sha256(bytes, expected)?;
    }
    let dest = subst(&input.dest, sandbox_path);
    let dest = Path::new(&dest);
    if input.unpack {
        unpack_archive(bytes, dest)
            .await
            .with_context(|| format!("unpacking input into {}", dest.display()))
    } else {
        write_file(bytes, dest)
            .await
            .with_context(|| format!("writing input to {}", dest.display()))
    }
}

/// Whether `uri` is a direct URL (fetched with a plain GET) rather than a store
/// key (fetched grant-authed through the control plane).
fn is_url(uri: &str) -> bool {
    uri.starts_with(HTTP_SCHEME) || uri.starts_with(HTTPS_SCHEME)
}

/// Fetch input bytes: a direct GET for a URL, or a grant-authed store fetch for
/// a store key.
async fn fetch(uri: &str, client: &HttpClient) -> Result<Vec<u8>> {
    if is_url(uri) {
        let response = reqwest::get(uri).await?.error_for_status()?;
        Ok(response.bytes().await?.to_vec())
    } else {
        client.fetch(uri).await
    }
}

/// Verify `bytes` hash to `expected` (hex, case-insensitive).
fn verify_sha256(bytes: &[u8], expected: &str) -> Result<()> {
    let actual = hex::encode(Sha256::digest(bytes));
    if !actual.eq_ignore_ascii_case(expected) {
        bail!("sha256 mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

/// Write `bytes` to `dest`, creating parent directories as needed.
async fn write_file(bytes: &[u8], dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(dest, bytes).await?;
    Ok(())
}

/// Unpack a tar archive (optionally zstd-compressed) into the directory `dest`.
/// The archive kind is detected from the leading bytes, so both `.tar` and
/// `.tar.zst` inputs work.
async fn unpack_archive(bytes: &[u8], dest: &Path) -> Result<()> {
    let bytes = bytes.to_vec();
    let dest = dest.to_path_buf();
    tokio::task::spawn_blocking(move || unpack_blocking(&bytes, &dest)).await?
}

fn unpack_blocking(bytes: &[u8], dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    if bytes.starts_with(&ZSTD_MAGIC) {
        let decoder = zstd::Decoder::new(std::io::Cursor::new(bytes))?;
        tar::Archive::new(decoder).unpack(dest)?;
    } else {
        tar::Archive::new(std::io::Cursor::new(bytes)).unpack(dest)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tar_bytes() -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        let payload = b"hello file";
        let mut header = tar::Header::new_gnu();
        header.set_path("dir/inner.txt").unwrap();
        header.set_size(payload.len() as u64);
        header.set_cksum();
        builder.append(&header, &payload[..]).unwrap();
        builder.into_inner().unwrap()
    }

    fn zstd_tar_bytes() -> Vec<u8> {
        let mut encoder = zstd::Encoder::new(Vec::new(), 0).unwrap();
        encoder.write_all(&tar_bytes()).unwrap();
        encoder.finish().unwrap()
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("chuk-inputs-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn is_url_distinguishes_urls_from_store_keys() {
        assert!(is_url("http://h/x"));
        assert!(is_url("https://h/x"));
        assert!(!is_url("runs/j1/code.tar.zst"));
        assert!(!is_url("ftp://h/x"));
    }

    #[test]
    fn verify_sha256_accepts_match_and_rejects_mismatch() {
        let bytes = b"payload";
        let good = hex::encode(Sha256::digest(bytes));
        verify_sha256(bytes, &good).unwrap();
        verify_sha256(bytes, &good.to_uppercase()).unwrap(); // case-insensitive
        assert!(verify_sha256(bytes, "deadbeef").is_err());
    }

    #[tokio::test]
    async fn place_writes_a_plain_file_creating_parents() {
        let dir = scratch("file");
        let input = InputArtifact {
            uri: "https://ignored".into(),
            dest: format!("{}/nested/out.bin", dir.display()),
            sha256: Some(hex::encode(Sha256::digest(b"data"))),
            unpack: false,
        };
        place(&input, b"data", "unused").await.unwrap();
        let written = std::fs::read(dir.join("nested/out.bin")).unwrap();
        assert_eq!(written, b"data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn place_rejects_a_bad_hash_before_writing() {
        let dir = scratch("badhash");
        let input = InputArtifact {
            uri: "https://ignored".into(),
            dest: format!("{}/out.bin", dir.display()),
            sha256: Some("deadbeef".into()),
            unpack: false,
        };
        assert!(place(&input, b"data", "unused").await.is_err());
        assert!(!dir.join("out.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn place_unpacks_a_zstd_archive_into_a_directory() {
        let dir = scratch("zst");
        let input = InputArtifact {
            uri: "key".into(),
            dest: dir.display().to_string(),
            sha256: None,
            unpack: true,
        };
        place(&input, &zstd_tar_bytes(), "unused").await.unwrap();
        assert_eq!(std::fs::read(dir.join("dir/inner.txt")).unwrap(), b"hello file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn place_unpacks_a_plain_tar_archive() {
        let dir = scratch("tar");
        let input = InputArtifact {
            uri: "key".into(),
            dest: dir.display().to_string(),
            sha256: None,
            unpack: true,
        };
        place(&input, &tar_bytes(), "unused").await.unwrap();
        assert_eq!(std::fs::read(dir.join("dir/inner.txt")).unwrap(), b"hello file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A refused loopback address, so the fetch paths fail fast and
    // deterministically without a network. (The transfer happy path itself is
    // integration-tested against a live control plane elsewhere.)
    const REFUSED_ORIGIN: &str = "http://127.0.0.1:1";

    #[tokio::test]
    async fn stage_propagates_a_url_fetch_error_with_context() {
        let client = HttpClient::new(REFUSED_ORIGIN.into(), String::new());
        let input = InputArtifact {
            uri: format!("{REFUSED_ORIGIN}/missing"),
            dest: "/unused".into(),
            sha256: None,
            unpack: false,
        };
        let error = stage(&input, "unused", &client).await.unwrap_err();
        assert!(format!("{error:#}").contains("fetching input"));
    }

    #[tokio::test]
    async fn fetch_routes_a_non_url_key_through_the_store() {
        let client = HttpClient::new(REFUSED_ORIGIN.into(), String::new());
        // A store key is fetched grant-authed through the (here unreachable)
        // control plane rather than by a direct GET.
        assert!(fetch("runs/j1/code.tar.zst", &client).await.is_err());
    }

    #[tokio::test]
    async fn place_substitutes_the_sandbox_placeholder_in_dest() {
        let dir = scratch("subst");
        let input = InputArtifact {
            uri: "key".into(),
            dest: format!("{}/f.txt", chuk_compute_wire::SANDBOX_PLACEHOLDER),
            sha256: None,
            unpack: false,
        };
        place(&input, b"x", &dir.display().to_string()).await.unwrap();
        assert_eq!(std::fs::read(dir.join("f.txt")).unwrap(), b"x");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
