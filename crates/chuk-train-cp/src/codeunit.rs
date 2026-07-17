//! Building a code unit (spec §11.1): tar the repo at a commit, pin the
//! manifest + lockfile, hash the tarball, store it content-addressed.
//!
//! M1 supports a **local directory** as `repo` (what the E1 stub uses) and a
//! git URL (shelled `git clone`). The unit id is the sha256 of the tarball;
//! agents cache extracted units by that id, so a warm worker skips env prep.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chuk_train_proto::{
    CodeRef, CodeUnitInfo, CodeUnitManifest, CODE_UNIT_LOCKFILE, CODE_UNIT_MANIFEST,
};

use crate::artifacts::{keys, sha256_hex, ArtifactStore};

/// Directory entries never worth shipping in a code unit.
const TAR_EXCLUDES: [&str; 5] = [".git", "target", "__pycache__", ".venv", ".mypy_cache"];
/// Fixed mtime for tar headers so the same tree hashes the same across builds.
const REPRODUCIBLE_MTIME: u64 = 0;

pub async fn build(
    store: &dyn ArtifactStore,
    repo: &str,
    commit: Option<&str>,
    name_override: Option<&str>,
) -> Result<CodeUnitInfo> {
    let source = materialize_source(repo, commit).await?;
    let dir = source.path();

    let manifest_path = dir.join(CODE_UNIT_MANIFEST);
    let manifest_text = tokio::fs::read_to_string(&manifest_path)
        .await
        .with_context(|| format!("code unit needs a {CODE_UNIT_MANIFEST} at its root"))?;
    let mut manifest: CodeUnitManifest =
        toml::from_str(&manifest_text).context("parsing unit.toml")?;
    if let Some(name) = name_override {
        manifest.name = name.to_owned();
    }
    if manifest.name.is_empty() {
        bail!("code unit manifest is missing a name");
    }

    let lockfile = tokio::fs::read(dir.join(CODE_UNIT_LOCKFILE)).await.ok();
    let tarball = tar_zstd(dir).context("packing code unit tarball")?;
    let sha = sha256_hex(&tarball);
    let code = CodeRef {
        name: manifest.name.clone(),
        sha: sha.clone(),
    };

    store
        .put(&keys::code_unit_tarball(&code.name, &sha), tarball)
        .await?;
    let uri = store
        .put(
            &keys::code_unit_manifest(&code.name, &sha),
            manifest_text.into_bytes(),
        )
        .await?;
    if let Some(lock) = lockfile {
        store
            .put(&keys::code_unit_lockfile(&code.name, &sha), lock)
            .await?;
    }

    Ok(CodeUnitInfo {
        code,
        manifest,
        uri,
        created_at: now(),
    })
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Either a borrowed local directory or an owned temp clone; both expose a path.
enum Source {
    Local(PathBuf),
    Cloned(TempDir),
}

impl Source {
    fn path(&self) -> &Path {
        match self {
            Source::Local(p) => p,
            Source::Cloned(t) => &t.0,
        }
    }
}

async fn materialize_source(repo: &str, commit: Option<&str>) -> Result<Source> {
    let local = Path::new(repo);
    if local.is_dir() {
        if commit.is_some() {
            tracing::warn!("commit ignored: {repo} is a local directory, not a git checkout");
        }
        return Ok(Source::Local(local.to_path_buf()));
    }
    // Otherwise treat it as a git URL and clone into a temp dir.
    let temp = TempDir::new()?;
    run_git(&["clone", "--depth", "1", repo, &temp.0.to_string_lossy()]).await?;
    if let Some(commit) = commit {
        run_git(&[
            "-C",
            &temp.0.to_string_lossy(),
            "fetch",
            "--depth",
            "1",
            "origin",
            commit,
        ])
        .await?;
        run_git(&["-C", &temp.0.to_string_lossy(), "checkout", commit]).await?;
    }
    Ok(Source::Cloned(temp))
}

async fn run_git(args: &[&str]) -> Result<()> {
    let status = tokio::process::Command::new("git")
        .args(args)
        .status()
        .await
        .context("running git (is it installed?)")?;
    anyhow::ensure!(status.success(), "git {:?} failed", args);
    Ok(())
}

fn tar_zstd(dir: &Path) -> Result<Vec<u8>> {
    let mut encoder = zstd::Encoder::new(Vec::new(), 0)?;
    {
        let mut builder = tar::Builder::new(&mut encoder);
        append_dir(&mut builder, dir, dir)?;
        builder.finish()?;
    }
    Ok(encoder.finish()?)
}

fn append_dir(
    builder: &mut tar::Builder<impl std::io::Write>,
    root: &Path,
    dir: &Path,
) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<std::io::Result<_>>()?;
    // Sort for a stable tar order → a stable hash for the same tree.
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let name = entry.file_name();
        if TAR_EXCLUDES.iter().any(|ex| name == *ex) {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(root)?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            append_dir(builder, root, &path)?;
        } else if file_type.is_file() {
            let data = std::fs::read(&path)?;
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(file_mode(&path));
            header.set_mtime(REPRODUCIBLE_MTIME);
            header.set_cksum();
            builder.append_data(&mut header, rel, data.as_slice())?;
        }
    }
    Ok(())
}

fn file_mode(path: &Path) -> u32 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .map(|m| m.permissions().mode())
            .unwrap_or(0o644)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        0o644
    }
}

/// A temp directory removed on drop. Small local helper to avoid a dep.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self> {
        let base = std::env::temp_dir().join(format!(
            "chuk-train-build-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&base)?;
        Ok(Self(base))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
