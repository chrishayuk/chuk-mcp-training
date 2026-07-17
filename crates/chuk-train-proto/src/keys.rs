//! Store-relative key layout (spec §11.5). Shared by the control plane (which
//! stores + serves) and the agent (which uploads to the same paths), so the
//! layout lives in exactly one place.

use crate::constants::{CODE_UNIT_LOCKFILE, CODE_UNIT_MANIFEST, CODE_UNIT_TARBALL};
use crate::CHECKPOINT_DIR_PREFIX;

/// `artifacts/code/<name>/<sha>` — a code unit's directory.
pub fn code_unit_dir(name: &str, sha: &str) -> String {
    format!("artifacts/code/{name}/{sha}")
}

pub fn code_unit_tarball(name: &str, sha: &str) -> String {
    format!("{}/{CODE_UNIT_TARBALL}", code_unit_dir(name, sha))
}

pub fn code_unit_manifest(name: &str, sha: &str) -> String {
    format!("{}/{CODE_UNIT_MANIFEST}", code_unit_dir(name, sha))
}

pub fn code_unit_lockfile(name: &str, sha: &str) -> String {
    format!("{}/{CODE_UNIT_LOCKFILE}", code_unit_dir(name, sha))
}

/// `runs/<run_id>/ckpt/step_<n>` — a checkpoint directory.
pub fn checkpoint_dir(run_id: &str, step: u64) -> String {
    format!("runs/{run_id}/ckpt/{CHECKPOINT_DIR_PREFIX}{step}")
}

/// A file within a checkpoint directory, e.g. `.../step_5/model.safetensors`.
pub fn checkpoint_file(run_id: &str, step: u64, filename: &str) -> String {
    format!("{}/{filename}", checkpoint_dir(run_id, step))
}

/// Reject keys that could escape the store root via traversal or absolute
/// paths. Layout functions build trusted keys, but uploads carry a
/// client-supplied filename tail, so this guards that seam.
pub fn is_safe_key(key: &str) -> bool {
    !key.is_empty()
        && !key.starts_with('/')
        && !key
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_spec() {
        assert_eq!(
            code_unit_tarball("cn7", "ab12"),
            "artifacts/code/cn7/ab12/unit.tar.zst"
        );
        assert_eq!(checkpoint_dir("r1", 15000), "runs/r1/ckpt/step_15000");
        assert_eq!(
            checkpoint_file("r1", 500, "model.safetensors"),
            "runs/r1/ckpt/step_500/model.safetensors"
        );
    }

    #[test]
    fn rejects_traversal() {
        assert!(!is_safe_key("../etc/passwd"));
        assert!(!is_safe_key("/abs"));
        assert!(!is_safe_key("runs/../x"));
        assert!(is_safe_key("runs/r1/ckpt/step_5/model.safetensors"));
    }
}
