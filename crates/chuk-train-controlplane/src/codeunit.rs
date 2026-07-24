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
    path: Option<&str>,
) -> Result<CodeUnitInfo> {
    let source = materialize_source(repo, commit).await?;
    let root = source.path();
    // A monorepo subdirectory (e.g. `examples/v11-pretrain`): the unit lives
    // under `root`, not at its top — both the manifest lookup and the
    // tarball below must be scoped to it, or the unit ships the whole repo.
    let dir = match path {
        Some(sub) => {
            // `starts_with` on the joined path won't catch `..` segments (it
            // compares components lexically, before any `..` is resolved),
            // so reject them outright rather than trusting containment.
            anyhow::ensure!(
                chuk_train_proto::keys::is_safe_key(sub),
                "path must stay within the repo: {sub}"
            );
            root.join(sub)
        }
        None => root.to_path_buf(),
    };
    let dir = dir.as_path();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::FsArtifactStore;

    /// A throwaway local-directory "repo" with a unit optionally nested under
    /// a subdirectory, mirroring a monorepo layout like `examples/<name>`.
    /// `marker` keeps each test's tarball content (and so its content-address)
    /// distinct — these tests share one `FsArtifactStore` root, and identical
    /// content run concurrently races on that backend's fixed temp-file name.
    fn scratch_repo(unit_subdir: Option<&str>, marker: &str) -> TempDir {
        let repo = TempDir::new().unwrap();
        let unit_root = match unit_subdir {
            Some(sub) => {
                let dir = repo.0.join(sub);
                std::fs::create_dir_all(&dir).unwrap();
                dir
            }
            None => repo.0.clone(),
        };
        std::fs::write(
            unit_root.join(CODE_UNIT_MANIFEST),
            "name = \"scratch\"\nversion = \"0.1.0\"\n[entrypoints]\ntrain = \"true\"\n",
        )
        .unwrap();
        std::fs::write(unit_root.join("train.py"), format!("print('{marker}')\n")).unwrap();
        repo
    }

    #[tokio::test]
    async fn build_finds_the_manifest_under_a_monorepo_subdirectory() {
        let repo = scratch_repo(Some("examples/v11-pretrain"), "finds-manifest");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(
            &store,
            &repo.0.to_string_lossy(),
            None,
            None,
            Some("examples/v11-pretrain"),
        )
        .await
        .unwrap();
        assert_eq!(info.manifest.name, "scratch");
    }

    #[tokio::test]
    async fn build_without_a_path_still_reads_the_root_manifest() {
        let repo = scratch_repo(None, "root-manifest");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(&store, &repo.0.to_string_lossy(), None, None, None)
            .await
            .unwrap();
        assert_eq!(info.manifest.name, "scratch");
    }

    #[tokio::test]
    async fn build_rejects_a_path_that_escapes_the_repo() {
        let repo = scratch_repo(None, "escape-guard");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let err = build(&store, &repo.0.to_string_lossy(), None, None, Some("../../etc"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("must stay within"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn build_tars_only_the_subdirectory_not_the_whole_repo() {
        let repo = scratch_repo(Some("examples/v11-pretrain"), "tars-subdir-only");
        // A file outside the unit subdir — must never end up in the tarball.
        std::fs::write(repo.0.join("unrelated.txt"), "secret").unwrap();
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(
            &store,
            &repo.0.to_string_lossy(),
            None,
            None,
            Some("examples/v11-pretrain"),
        )
        .await
        .unwrap();
        let tarball = store
            .get(&keys::code_unit_tarball(&info.code.name, &info.code.sha))
            .await
            .unwrap();
        let decoded = zstd::decode_all(tarball.as_slice()).unwrap();
        let mut archive = tar::Archive::new(decoded.as_slice());
        let names: Vec<String> = archive
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"train.py".to_owned()), "names={names:?}");
        assert!(!names.iter().any(|n| n.contains("unrelated.txt")), "names={names:?}");
    }
}
