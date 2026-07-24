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
    async fn build_applies_a_name_override() {
        let repo = scratch_repo(None, "name-override");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(
            &store,
            &repo.0.to_string_lossy(),
            None,
            Some("overridden-name"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(info.manifest.name, "overridden-name");
        assert_eq!(info.code.name, "overridden-name");
        // The tarball must be stored under the override, not the manifest's
        // original name, or a later fetch by the returned CodeRef would miss.
        store
            .get(&keys::code_unit_tarball("overridden-name", &info.code.sha))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn build_rejects_an_empty_manifest_name() {
        let repo = TempDir::new().unwrap();
        std::fs::write(
            repo.0.join(CODE_UNIT_MANIFEST),
            "name = \"\"\nversion = \"0.1.0\"\n[entrypoints]\ntrain = \"true\"\n",
        )
        .unwrap();
        let store = FsArtifactStore::new(std::env::temp_dir());
        let err = build(&store, &repo.0.to_string_lossy(), None, None, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing a name"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn build_stores_the_lockfile_when_present() {
        let repo = scratch_repo(None, "with-lockfile");
        std::fs::write(repo.0.join(CODE_UNIT_LOCKFILE), "# pinned deps\n").unwrap();
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(&store, &repo.0.to_string_lossy(), None, None, None)
            .await
            .unwrap();
        let stored = store
            .get(&keys::code_unit_lockfile(&info.code.name, &info.code.sha))
            .await
            .unwrap();
        assert_eq!(stored, b"# pinned deps\n");
    }

    #[tokio::test]
    async fn build_ignores_a_commit_pin_for_a_local_directory() {
        // A local directory has no notion of a commit; `build` should warn
        // and proceed against the directory as-is rather than failing.
        let repo = scratch_repo(None, "commit-ignored-for-local-dir");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(
            &store,
            &repo.0.to_string_lossy(),
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(info.manifest.name, "scratch");
    }

    /// Initializes a throwaway local git repo (no network involved) with one
    /// commit, so the git-clone path in `materialize_source` can be exercised
    /// against a `file://` URL instead of a real remote.
    fn init_git_repo(marker: &str) -> (TempDir, String) {
        let repo = TempDir::new().unwrap();
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(&repo.0)
                .status()
                .expect("git must be installed to run this test");
            assert!(status.success(), "git {args:?} failed");
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "codeunit-test@example.com"]);
        git(&["config", "user.name", "codeunit-test"]);
        std::fs::write(
            repo.0.join(CODE_UNIT_MANIFEST),
            "name = \"scratch\"\nversion = \"0.1.0\"\n[entrypoints]\ntrain = \"true\"\n",
        )
        .unwrap();
        std::fs::write(repo.0.join("train.py"), format!("print('{marker}')\n")).unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "init"]);
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&repo.0)
            .output()
            .unwrap();
        assert!(out.status.success(), "git rev-parse HEAD failed");
        let sha = String::from_utf8(out.stdout).unwrap().trim().to_owned();
        (repo, sha)
    }

    #[tokio::test]
    async fn build_clones_a_local_git_repo_over_a_file_url() {
        let (repo, _sha) = init_git_repo("clone-over-file-url");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let url = format!("file://{}", repo.0.display());
        let info = build(&store, &url, None, None, None).await.unwrap();
        assert_eq!(info.manifest.name, "scratch");
    }

    #[tokio::test]
    async fn build_clones_a_local_git_repo_and_checks_out_a_pinned_commit() {
        let (repo, sha) = init_git_repo("clone-pinned-commit");
        let store = FsArtifactStore::new(std::env::temp_dir());
        let url = format!("file://{}", repo.0.display());
        let info = build(&store, &url, Some(&sha), None, None).await.unwrap();
        assert_eq!(info.manifest.name, "scratch");
    }

    #[tokio::test]
    async fn build_surfaces_a_git_clone_failure() {
        let store = FsArtifactStore::new(std::env::temp_dir());
        // Not a directory (so it takes the clone path) and not a real git
        // remote, so `git clone` itself fails and the error should propagate.
        let err = build(
            &store,
            "file:///definitely/not/a/real/git/repo/anywhere",
            None,
            None,
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("git"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn build_excludes_ignored_dirs_and_recurses_into_real_subdirectories() {
        let repo = scratch_repo(None, "excludes-and-recurses");
        // `target` is a TAR_EXCLUDES entry: it and everything under it must
        // never reach the tarball, even though it sits right at the root.
        std::fs::create_dir_all(repo.0.join("target")).unwrap();
        std::fs::write(repo.0.join("target").join("built.bin"), b"binary").unwrap();
        // A real (non-excluded) nested subdirectory: append_dir must recurse
        // into it and preserve the relative path in the tar entry name.
        std::fs::create_dir_all(repo.0.join("src").join("nested")).unwrap();
        std::fs::write(repo.0.join("src").join("nested").join("helper.py"), "pass\n").unwrap();
        let store = FsArtifactStore::new(std::env::temp_dir());
        let info = build(&store, &repo.0.to_string_lossy(), None, None, None)
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
        assert!(
            names.contains(&"src/nested/helper.py".to_owned()),
            "names={names:?}"
        );
        assert!(!names.iter().any(|n| n.contains("target")), "names={names:?}");
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
