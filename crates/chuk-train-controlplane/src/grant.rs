//! Run-scoped upload/read capabilities (spec §12).
//!
//! A worker never holds the API token or any provider credential. When a train
//! run is assigned, the control plane mints a random token bound to that run,
//! authorising exactly: writing anywhere under `runs/<run_id>/`, and reading
//! that run's own checkpoints plus its assigned code unit. Grants live in
//! memory and are dropped when the run leaves the worker — a control-plane
//! restart drops the websocket, which requeues the run and mints a fresh grant.

use std::collections::HashMap;
use std::time::Instant;

use chuk_train_proto::{CodeRef, RunId, CKPT_FINAL_PREFIX, CKPT_HOT_PREFIX, UPLOAD_GRANT_TTL};

use crate::artifacts::keys;

/// The capability behind one grant token.
#[derive(Debug, Clone)]
pub struct Grant {
    pub run_id: RunId,
    pub code: CodeRef,
    expires: Instant,
}

impl Grant {
    fn is_valid(&self) -> bool {
        Instant::now() < self.expires
    }

    /// Store-key prefixes that belong to this grant's run: its run tree plus its
    /// hot and promoted-final checkpoint trees. Checkpoints live under the
    /// top-level `ckpt-hot/`/`ckpt-final/` prefixes (spec §11.5, for R2 lifecycle
    /// targeting), *not* under `runs/<id>/`, so the grant must name them too.
    fn run_prefixes(&self) -> [String; 3] {
        [
            format!("runs/{}/", self.run_id),
            format!("{CKPT_HOT_PREFIX}/{}/", self.run_id),
            format!("{CKPT_FINAL_PREFIX}/{}/", self.run_id),
        ]
    }

    /// May this grant write the given store key? Only under its run's trees.
    pub fn may_write(&self, key: &str) -> bool {
        keys::is_safe_key(key) && self.run_prefixes().iter().any(|p| key.starts_with(p))
    }

    /// May this grant read the given store key? Its run's trees, or its own
    /// assigned code unit (for fetching the tarball to run).
    pub fn may_read(&self, key: &str) -> bool {
        if !keys::is_safe_key(key) {
            return false;
        }
        self.run_prefixes().iter().any(|p| key.starts_with(p))
            || key.starts_with(&format!(
                "{}/",
                keys::code_unit_dir(&self.code.name, &self.code.sha)
            ))
    }
}

/// A thread-safe registry of live grants, keyed by opaque token.
#[derive(Default)]
pub struct GrantTable {
    grants: std::sync::Mutex<HashMap<String, Grant>>,
}

impl GrantTable {
    pub fn mint(&self, run_id: RunId, code: CodeRef) -> String {
        let token = format!("grant-{}", uuid::Uuid::new_v4().simple());
        let grant = Grant {
            run_id,
            code,
            expires: Instant::now() + UPLOAD_GRANT_TTL,
        };
        self.grants
            .lock()
            .expect("grant lock")
            .insert(token.clone(), grant);
        token
    }

    /// Resolve a token to a still-valid grant, if any.
    pub fn resolve(&self, token: &str) -> Option<Grant> {
        let mut grants = self.grants.lock().expect("grant lock");
        let grant = grants.get(token)?;
        if grant.is_valid() {
            Some(grant.clone())
        } else {
            grants.remove(token);
            None
        }
    }

    /// Drop every grant for a run (called when it leaves the worker).
    pub fn revoke_run(&self, run_id: &RunId) {
        self.grants
            .lock()
            .expect("grant lock")
            .retain(|_, g| &g.run_id != run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grant(run: &str) -> Grant {
        Grant {
            run_id: RunId(run.to_owned()),
            code: CodeRef {
                name: "unit".into(),
                sha: "abc123".into(),
            },
            expires: Instant::now() + UPLOAD_GRANT_TTL,
        }
    }

    #[test]
    fn grant_scopes_to_run_hot_and_final_checkpoint_trees() {
        let g = grant("RUN-1");
        // the new top-level checkpoint prefixes (spec §11.5) are authorised
        assert!(g.may_write("ckpt-hot/RUN-1/step_30/model.safetensors"));
        assert!(g.may_read("ckpt-hot/RUN-1/step_30/model.safetensors"));
        assert!(g.may_read("ckpt-final/RUN-1/step_30/meta.json"));
        assert!(g.may_write("runs/RUN-1/logs/slice_0.log"));
        // another run's tree, traversal, and foreign keys are refused
        assert!(!g.may_write("ckpt-hot/RUN-2/step_30/model.safetensors"));
        assert!(!g.may_write("ckpt-hot/RUN-1/../escape"));
        // its own code unit is readable but not writable
        assert!(g.may_read("artifacts/code/unit/abc123/unit.tar.zst"));
        assert!(!g.may_write("artifacts/code/unit/abc123/unit.tar.zst"));
    }

    #[test]
    fn may_read_also_refuses_unsafe_keys() {
        // `may_write` short-circuits on an unsafe key via the same
        // `keys::is_safe_key` guard; `may_read` must too, even for a
        // traversal that would otherwise land inside the run's own tree.
        let g = grant("RUN-1");
        assert!(!g.may_read("runs/RUN-1/../../etc/passwd"));
    }

    #[test]
    fn is_valid_reflects_the_expiry_deadline() {
        let live = grant("RUN-1");
        assert!(live.is_valid());

        let mut expired = grant("RUN-1");
        expired.expires = Instant::now() - std::time::Duration::from_secs(1);
        assert!(!expired.is_valid());
    }

    #[test]
    fn mint_then_resolve_round_trips_the_run_and_code() {
        let table = GrantTable::default();
        let code = CodeRef { name: "unit".into(), sha: "deadbeef".into() };
        let token = table.mint(RunId("RUN-1".into()), code.clone());

        let resolved = table.resolve(&token).expect("freshly minted grant resolves");
        assert_eq!(resolved.run_id, RunId("RUN-1".into()));
        assert_eq!(resolved.code, code);
    }

    #[test]
    fn resolve_returns_none_for_an_unknown_token() {
        let table = GrantTable::default();
        assert!(table.resolve("grant-does-not-exist").is_none());
    }

    #[test]
    fn resolve_evicts_and_refuses_an_expired_grant() {
        let table = GrantTable::default();
        // Insert directly (rather than via `mint`, which always sets a live
        // TTL) so expiry can be tested without sleeping.
        let mut expired = grant("RUN-1");
        expired.expires = Instant::now() - std::time::Duration::from_secs(1);
        table.grants.lock().unwrap().insert("grant-expired".into(), expired);

        assert!(table.resolve("grant-expired").is_none());
        // The lookup also swept the dead entry out of the table.
        assert!(!table.grants.lock().unwrap().contains_key("grant-expired"));
    }

    #[test]
    fn revoke_run_drops_only_that_runs_grants() {
        let table = GrantTable::default();
        let code = CodeRef { name: "unit".into(), sha: "abc123".into() };
        let a = table.mint(RunId("RUN-A".into()), code.clone());
        let b = table.mint(RunId("RUN-B".into()), code);

        table.revoke_run(&RunId("RUN-A".into()));

        assert!(table.resolve(&a).is_none(), "RUN-A's grant must be gone");
        assert!(table.resolve(&b).is_some(), "RUN-B's grant must be untouched");
    }
}
