//! chuk-datasets-server dataset resolution (spec §6/§7.3): the dispatch-time
//! `resolve` call that turns a `TrainSpec.data` block into a concrete
//! `content_sha` (+ optional `plan_sha`) and shard fetch URLs, pre-warming
//! the worker client's own `resolve → fetch → verify → cache → refuse`
//! contract so the common case never blocks on the chuk-datasets server.
//!
//! Optional and gated: `None` unless `CHUK_DATASETS_URL` + `CHUK_DATASETS_API_KEY`
//! are set — but unlike the best-effort chuk-experiments-server mirror, this is
//! not a side-channel: a run that declares a `data:` block with no configured
//! client fails to dispatch (the data is required to run at all).

use chuk_datasets_client::{ResolveClient, ResolvedContent};
use chuk_train_proto::env;

pub struct Datasets {
    client: ResolveClient,
}

impl Datasets {
    /// Build from the environment. Returns `None` — dataset resolution is off
    /// — unless both `CHUK_DATASETS_URL` and `CHUK_DATASETS_API_KEY` are set.
    pub fn from_env() -> Option<Self> {
        let base = std::env::var(env::DATASETS_URL)
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_owned())
            .filter(|s| !s.is_empty())?;
        let key = std::env::var(env::DATASETS_API_KEY)
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())?;
        Some(Self { client: ResolveClient::new(base, Some(key)) })
    }

    /// A client pointed at an explicit base URL, bypassing env gating —
    /// for tests that spin up a mock resolve server.
    #[cfg(test)]
    pub(crate) fn at(base_url: impl Into<String>) -> Self {
        Self { client: ResolveClient::new(base_url, Some("test-key".to_owned())) }
    }

    /// `dataset` is `<name>@<content_sha>` or a bare content sha; `plan` a
    /// concrete plan sha or a planset member ref. Resolves to the concrete
    /// identity the worker will fetch and verify against.
    pub async fn resolve(&self, dataset: &str, plan: Option<&str>) -> anyhow::Result<ResolvedContent> {
        self.client.resolve(dataset, plan).await.map_err(anyhow::Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A dedicated env-var mutex would be overkill for one test; the two vars
    // this touches (CHUK_DATASETS_URL/_API_KEY) aren't read anywhere else in
    // this crate's test suite.

    #[test]
    fn from_env_is_none_unless_both_vars_are_set() {
        std::env::remove_var(env::DATASETS_URL);
        std::env::remove_var(env::DATASETS_API_KEY);
        assert!(Datasets::from_env().is_none());

        std::env::set_var(env::DATASETS_URL, "https://chuk-datasets.fly.dev");
        assert!(Datasets::from_env().is_none(), "url alone is not enough");
        std::env::remove_var(env::DATASETS_URL);

        std::env::set_var(env::DATASETS_API_KEY, "ck_test");
        assert!(Datasets::from_env().is_none(), "key alone is not enough");

        std::env::set_var(env::DATASETS_URL, "https://chuk-datasets.fly.dev/");
        assert!(Datasets::from_env().is_some());

        std::env::remove_var(env::DATASETS_URL);
        std::env::remove_var(env::DATASETS_API_KEY);
    }
}
