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

use chuk_train_proto::{CodeRef, RunId, UPLOAD_GRANT_TTL};

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

    /// May this grant write the given store key? Only under its run's tree.
    pub fn may_write(&self, key: &str) -> bool {
        keys::is_safe_key(key) && key.starts_with(&format!("runs/{}/", self.run_id))
    }

    /// May this grant read the given store key? Its run's tree, or its own
    /// assigned code unit (for fetching the tarball to run).
    pub fn may_read(&self, key: &str) -> bool {
        if !keys::is_safe_key(key) {
            return false;
        }
        key.starts_with(&format!("runs/{}/", self.run_id))
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
