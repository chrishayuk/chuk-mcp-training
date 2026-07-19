//! The per-job sandbox: a fresh working directory the command runs in, plus the
//! placeholder substitution that turns sandbox-relative paths in a job spec into
//! absolute paths on this worker's filesystem.

use std::path::{Path, PathBuf};

use chuk_compute_wire::{JobId, SANDBOX_PLACEHOLDER};

/// Directory-name prefix for a job's sandbox under the system temp dir.
const SANDBOX_DIR_PREFIX: &str = "chuk-job-";

/// Replace every occurrence of [`SANDBOX_PLACEHOLDER`] in `template` with the
/// sandbox's absolute path. Applied to command arguments, environment values,
/// input destinations, output globs, and the metrics-file path — so a job spec
/// expresses paths portably without knowing this worker's filesystem.
pub fn subst(template: &str, sandbox_path: &str) -> String {
    template.replace(SANDBOX_PLACEHOLDER, sandbox_path)
}

/// A freshly-created working directory for one job. Dropping the value leaves
/// the directory on disk (the executor owns cleanup policy); creating a new
/// sandbox for the same job id first removes any stale directory so a prior
/// run's leftovers never leak into a new one.
pub struct Sandbox {
    root: PathBuf,
    path: String,
}

impl Sandbox {
    /// Create (or recreate) the sandbox directory for `job_id`.
    pub fn create(job_id: &JobId) -> std::io::Result<Self> {
        let root = std::env::temp_dir().join(format!("{SANDBOX_DIR_PREFIX}{job_id}"));
        // A stale directory from an earlier run on this machine must not bleed
        // into this one; start from empty.
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        let path = root.to_string_lossy().into_owned();
        Ok(Self { root, path })
    }

    /// The sandbox root as a path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The sandbox root as the string used for [`subst`].
    pub fn path(&self) -> &str {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subst_replaces_every_occurrence() {
        let out = subst(
            &format!("{SANDBOX_PLACEHOLDER}/a:{SANDBOX_PLACEHOLDER}/b"),
            "/tmp/box",
        );
        assert_eq!(out, "/tmp/box/a:/tmp/box/b");
    }

    #[test]
    fn subst_is_a_noop_without_the_placeholder() {
        assert_eq!(subst("/plain/path", "/tmp/box"), "/plain/path");
        assert_eq!(subst("", "/tmp/box"), "");
    }

    #[test]
    fn create_makes_a_fresh_empty_directory() {
        let id = JobId::from(format!("test-{}", std::process::id()));
        // Seed a stale file to prove recreation clears it.
        let sandbox = Sandbox::create(&id).unwrap();
        std::fs::write(sandbox.root().join("stale"), b"x").unwrap();

        let sandbox = Sandbox::create(&id).unwrap();
        assert!(sandbox.root().is_dir());
        assert!(!sandbox.root().join("stale").exists());
        assert_eq!(sandbox.path(), sandbox.root().to_string_lossy());

        let _ = std::fs::remove_dir_all(sandbox.root());
    }
}
