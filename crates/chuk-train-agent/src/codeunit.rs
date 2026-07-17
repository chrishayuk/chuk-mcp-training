//! Fetch a code unit and cache it by content hash (spec §11.1). A warm worker
//! that already has `<sha>` extracted skips the download entirely — this is
//! where the packing scheduler's env-prep overhead goes to die.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chuk_train_proto::{keys, CodeRef, CodeUnitManifest, CODE_UNIT_MANIFEST};
use tracing::info;

use crate::httpclient::HttpClient;

/// Ensure the unit is present locally; return its extracted directory.
pub async fn ensure_local(
    client: &HttpClient,
    cache_dir: &Path,
    code: &CodeRef,
) -> Result<PathBuf> {
    let unit_dir = cache_dir.join(&code.sha);
    if unit_dir.join(CODE_UNIT_MANIFEST).is_file() {
        info!(unit = %code, "code unit cache hit");
        return Ok(unit_dir);
    }
    info!(unit = %code, "fetching code unit");
    let tarball = client
        .fetch(&keys::code_unit_tarball(&code.name, &code.sha))
        .await?;
    extract(&tarball, cache_dir, &code.sha).await?;
    Ok(unit_dir)
}

/// Decompress + untar into a temp sibling, then atomically rename into place so
/// a concurrent reader never sees a half-extracted unit.
async fn extract(tarball: &[u8], cache_dir: &Path, sha: &str) -> Result<()> {
    let cache_dir = cache_dir.to_path_buf();
    let sha = sha.to_owned();
    let bytes = tarball.to_vec();
    tokio::task::spawn_blocking(move || -> Result<()> {
        std::fs::create_dir_all(&cache_dir)?;
        let staging = cache_dir.join(format!(".staging-{sha}"));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;
        let decoder = zstd::Decoder::new(std::io::Cursor::new(bytes))?;
        tar::Archive::new(decoder)
            .unpack(&staging)
            .context("unpacking code unit")?;
        let final_dir = cache_dir.join(&sha);
        let _ = std::fs::remove_dir_all(&final_dir);
        std::fs::rename(&staging, &final_dir)?;
        Ok(())
    })
    .await??;
    Ok(())
}

pub async fn read_manifest(unit_dir: &Path) -> Result<CodeUnitManifest> {
    let text = tokio::fs::read_to_string(unit_dir.join(CODE_UNIT_MANIFEST))
        .await
        .context("reading unit.toml from extracted code unit")?;
    toml::from_str(&text).context("parsing unit.toml")
}
