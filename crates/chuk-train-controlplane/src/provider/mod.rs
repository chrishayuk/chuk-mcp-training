//! Provider drivers behind an adapter trait (spec §6). Only provisioning
//! differs per provider; everything after `agent --join` is identical.
//!
//! Teardown never depends on the agent (spec design principle 5): `destroy`
//! and `status` act on the provider's API directly, and the reconcile loop
//! lists real instances to catch anything the registry lost track of.
//!
//! M2 ships the **mock** provider (launches the agent binary as local
//! processes — the E2 analog that makes provider-verified destroy genuinely
//! real) and a **Vast** skeleton written to the same trait.

mod mock;
mod vast;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::{Instance, InstanceStatus, Offer, ProvisionRequest};

pub use mock::MockProvider;
pub use vast::VastProvider;

/// What the control plane needs from any GPU provider.
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    /// Rentable offers, optionally filtered.
    async fn offers(&self, gpu: Option<&str>, max_price_hr: Option<f64>) -> Result<Vec<Offer>>;

    /// Launch an instance that boots the agent dialing back to `ctx.ws_url`
    /// with `ctx.join_token`, under a wall of `req.lease_min`.
    async fn provision(&self, req: &ProvisionRequest, ctx: &ProvisionContext) -> Result<Instance>;

    /// Destroy an instance. Idempotent: destroying an already-gone instance
    /// succeeds. Must not depend on the agent being responsive.
    async fn destroy(&self, instance_id: &str) -> Result<()>;

    /// Current billing status of an instance (polled to verify destroy).
    async fn status(&self, instance_id: &str) -> Result<InstanceStatus>;

    /// Every instance this provider believes it is running — the reconcile
    /// loop diffs this against the registry to find orphans.
    async fn list_instances(&self) -> Result<Vec<Instance>>;
}

/// Everything a freshly-booted worker needs to dial home, handed to
/// `provision`. Secrets stay control-plane-side; the worker gets only these.
/// The control plane picks `worker_id` so the lease correlates with the worker
/// that registers.
#[derive(Debug, Clone)]
pub struct ProvisionContext {
    pub ws_url: String,
    pub join_token: String,
    pub worker_id: String,
    /// The drain window the agent should self-drain by (matches the control
    /// plane's, so the belt and the braces agree on T-drain).
    pub drain_window_min: f64,
}

/// A registry of the providers this control plane can drive.
pub struct Providers {
    map: HashMap<String, Arc<dyn Provider>>,
}

impl Providers {
    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.map.get(name).cloned()
    }

    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn Provider>> {
        self.map.values()
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.map.keys().cloned().collect();
        names.sort();
        names
    }
}

/// Build the provider registry from a comma-separated selection (env
/// `CHUK_TRAIN_PROVIDERS`, default `mock`).
pub fn build_providers(
    selection: &str,
    agent_bin: Option<String>,
    vast_api_key: Option<String>,
) -> Providers {
    let mut map: HashMap<String, Arc<dyn Provider>> = HashMap::new();
    for name in selection
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        match name {
            mock::NAME => {
                map.insert(
                    name.to_owned(),
                    Arc::new(MockProvider::new(agent_bin.clone())),
                );
            }
            vast::NAME => {
                map.insert(
                    name.to_owned(),
                    Arc::new(VastProvider::new(vast_api_key.clone())),
                );
            }
            other => tracing::warn!("unknown provider {other:?}; skipping"),
        }
    }
    Providers { map }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_known_providers_by_name() {
        let providers = build_providers("mock,vast", None, None);
        assert_eq!(providers.names(), vec!["mock".to_owned(), "vast".to_owned()]);
        assert_eq!(providers.get(mock::NAME).unwrap().name(), mock::NAME);
        assert_eq!(providers.get(vast::NAME).unwrap().name(), vast::NAME);
        assert_eq!(providers.all().count(), 2);
    }

    #[test]
    fn names_are_sorted_regardless_of_selection_order() {
        // "vast" sorts after "mock" lexicographically; selecting them in the
        // opposite order must not leak through to `names()`.
        let providers = build_providers("vast,mock", None, None);
        assert_eq!(providers.names(), vec!["mock".to_owned(), "vast".to_owned()]);
    }

    #[test]
    fn trims_whitespace_and_collapses_duplicate_entries() {
        let providers = build_providers(" mock , mock ,mock", None, None);
        assert_eq!(providers.names(), vec!["mock".to_owned()]);
    }

    #[test]
    fn blank_entries_between_commas_are_skipped() {
        let providers = build_providers("mock,,", None, None);
        assert_eq!(providers.names(), vec!["mock".to_owned()]);
    }

    #[test]
    fn unknown_provider_name_is_skipped_not_errored() {
        let providers = build_providers("mock,bogus", None, None);
        assert_eq!(providers.names(), vec!["mock".to_owned()]);
        assert!(providers.get("bogus").is_none());
    }

    #[test]
    fn empty_selection_yields_an_empty_registry() {
        let providers = build_providers("", None, None);
        assert!(providers.names().is_empty());
        assert_eq!(providers.all().count(), 0);
        assert!(providers.get(mock::NAME).is_none());
    }
}
