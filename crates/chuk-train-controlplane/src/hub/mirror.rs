//! Hub — chuk-experiments-server reporting mirror hooks (best-effort,
//! fire-and-forget). Called from [`super::submit`], [`super::messages`], and
//! [`super::control`] wherever a run's state/lineage changes; a no-op when
//! the mirror is unconfigured.

use super::*;

impl Hub {
    pub(super) fn mirror_created(&self, run_id: &RunId, spec: &RunSpec, experiment_ref: Option<&str>) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id, spec) = (exp.clone(), run_id.clone(), spec.clone());
            let experiment_ref = experiment_ref.map(str::to_owned);
            tokio::spawn(async move { exp.report_created(run_id, spec, experiment_ref).await });
        }
    }

    pub(super) fn mirror_state(&self, run_id: &RunId, state: RunState) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id) = (exp.clone(), run_id.clone());
            tokio::spawn(async move { exp.report_state(run_id, state).await });
        }
    }

    pub(super) fn mirror_checkpoint(&self, run_id: &RunId, step: u64, uri: &str, meta: &CheckpointMeta) {
        if let Some(exp) = &self.experiments {
            let (exp, run_id, uri, meta) =
                (exp.clone(), run_id.clone(), uri.to_owned(), meta.clone());
            tokio::spawn(async move { exp.report_checkpoint(run_id, step, uri, meta).await });
        }
    }
}
