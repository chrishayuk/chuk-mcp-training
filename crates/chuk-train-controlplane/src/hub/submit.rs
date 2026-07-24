//! Hub — run submission: single runs, sweeps (spec §5.2 fan-out), and the
//! chuk-experiments-server "push" path that builds a `TrainSpec` from an
//! existing experiments-server run.

use super::*;

impl Hub {
    pub async fn submit(
        &self,
        name: &str,
        spec: &RunSpec,
        experiment_ref: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<RunId> {
        let run_id = self
            .store
            .create_run(name, spec, experiment_ref, created_by, None)
            .await?;
        self.mirror_created(&run_id, spec, experiment_ref);
        self.pump().await?;
        Ok(run_id)
    }

    /// Fan a sweep out into child runs (spec §5.2): expand template × axes,
    /// record the sweep, queue every child (named `{name}-{i:03}` in axis
    /// order), and pump — the scheduler holds children to the sweep's
    /// concurrency. Children are unattached scratch runs mirror-wise.
    pub async fn submit_sweep(
        &self,
        name: &str,
        spec: &chuk_train_proto::SweepSpec,
        created_by: Option<&str>,
    ) -> Result<(String, Vec<RunId>)> {
        let children =
            sweep::expand(&spec.template, &spec.axes).map_err(|reason| anyhow::anyhow!(reason))?;
        let sweep_id = self
            .store
            .create_sweep(
                name,
                &serde_json::to_string(&spec.template)?,
                &serde_json::to_string(&spec.axes)?,
                spec.concurrency,
                created_by,
            )
            .await?;
        let mut run_ids = Vec::with_capacity(children.len());
        for (i, (_assignment, child_spec)) in children.into_iter().enumerate() {
            let run_id = self
                .store
                .create_run(
                    &format!("{name}-{i:03}"),
                    &RunSpec::Train(Box::new(child_spec)),
                    None,
                    created_by,
                    Some(&sweep_id),
                )
                .await?;
            run_ids.push(run_id);
        }
        info!(sweep = %sweep_id, children = run_ids.len(), "sweep fanned out");
        self.pump().await?;
        Ok((sweep_id, run_ids))
    }

    /// A sweep's children + the cross-child aggregate of one metric key at
    /// matched steps (spec §5.2 `sweep_status`). `None` if no such sweep.
    pub async fn sweep_status(
        &self,
        sweep_id: &str,
        key: &str,
    ) -> Result<Option<chuk_train_proto::SweepStatus>> {
        let Some(row) = self.store.sweep(sweep_id).await? else {
            return Ok(None);
        };
        let axes: std::collections::BTreeMap<String, Vec<serde_json::Value>> =
            serde_json::from_str(&row.axes)?;
        let query = crate::store::RunQuery {
            sweep_id: Some(sweep_id.to_owned()),
            ..Default::default()
        };
        let summaries = self
            .store
            .runs(&query, chuk_train_proto::MAX_SWEEP_CHILDREN as u32)
            .await?;
        let mut children = Vec::with_capacity(summaries.len());
        let mut series: Vec<Vec<chuk_train_proto::MetricPoint>> = Vec::new();
        // Oldest first, matching the submit-time `{name}-{i:03}` order.
        for summary in summaries.into_iter().rev() {
            let assignment = match self.store.run(&summary.id).await?.map(|r| r.spec) {
                Some(RunSpec::Train(train)) => sweep::assignment_of(&train, &axes),
                _ => Default::default(),
            };
            let child_series = self
                .store
                .metric_series(&summary.id, Some(std::slice::from_ref(&key.to_owned())), 0, 0)
                .await?;
            if let Some(points) = child_series.series.get(key) {
                if !points.is_empty() {
                    series.push(points.clone());
                }
            }
            children.push(chuk_train_proto::SweepChild {
                run_id: summary.id,
                state: summary.state,
                assignment,
            });
        }
        Ok(Some(chuk_train_proto::SweepStatus {
            sweep_id: row.id,
            name: row.name,
            concurrency: row.concurrency,
            children,
            key: key.to_owned(),
            aggregate: sweep::aggregate(&series),
        }))
    }

    /// Submit a train run built entirely from an existing chuk-experiments-server
    /// run's own `config`/`workspec` — the "push" half of the experiments-server
    /// integration (spec §11.6): create an experiment + run there, then point it
    /// here instead of re-specifying the same `TrainSpec` by hand. Delegates to
    /// [`Self::submit`] with `experiment_ref` already set, so it attaches to
    /// `experiment_run_id` rather than minting a duplicate (same guarantee as a
    /// manual `submit(..., experiment_ref: Some(_))` call).
    pub async fn submit_from_experiment(
        &self,
        experiment_run_id: &str,
        name: Option<&str>,
        created_by: Option<&str>,
    ) -> Result<RunId> {
        let exp = self
            .experiments
            .as_ref()
            .context("chuk-experiments-server mirror not configured")?;
        let snapshot = exp.fetch_run(experiment_run_id, created_by).await?;
        anyhow::ensure!(
            snapshot.harness_session_id.is_none(),
            "run {experiment_run_id} is already attached to a harness execution ({})",
            snapshot.harness_session_id.as_deref().unwrap_or_default()
        );
        anyhow::ensure!(
            snapshot.status == "queued",
            "run {experiment_run_id} is not queued (status={})",
            snapshot.status
        );
        let spec = crate::experiments::train_spec_from_experiments_run(&snapshot)?;
        let name = name.unwrap_or(experiment_run_id);
        self.submit(name, &RunSpec::Train(Box::new(spec)), Some(experiment_run_id), created_by)
            .await
    }
}
