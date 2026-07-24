//! Hub — the FIFO scheduler (spec §14: one job in flight per worker, no
//! packing) and train-run assembly: code-unit resolution, resume staging,
//! and chuk-datasets dispatch-time `data:` resolution (spec §6/§7.3).

use super::*;

/// How many queued runs one pump pass scans for an assignable candidate.
const QUEUE_SCAN_LIMIT: u32 = 500;

impl Hub {
    /// Assign queued runs to idle connected workers, FIFO. The links lock is held
    /// throughout, which also serialises concurrent pumps.
    pub async fn pump(&self) -> Result<()> {
        let mut links = self.links.lock().await;
        let mut dead: Vec<WorkerId> = Vec::new();
        for (worker_id, tx) in links.iter() {
            let Some(worker) = self.store.worker(worker_id).await? else {
                continue;
            };
            if !eligible_for_assignment(&worker, HEARTBEAT_TIMEOUT.as_secs_f64()) {
                continue;
            }
            let Some(run) = self.next_assignable_queued().await? else {
                break;
            };
            let run_id = run.summary.id.clone();
            let job = self.assemble_job(&run_id, run.spec).await?;
            self.store
                .transition(
                    &run_id,
                    RunState::Assigned,
                    Some(worker_id),
                    None,
                    serde_json::Value::Null,
                )
                .await?;
            self.store.set_worker_run(worker_id, Some(&run_id)).await?;
            if tx.send(wire::CpToWorker::AssignJob { job: Box::new(job) }).is_err() {
                warn!(worker = %worker_id, run = %run_id, "assign failed (link closed); requeueing");
                self.grants.revoke_run(&run_id);
                self.store
                    .transition(
                        &run_id,
                        RunState::Queued,
                        None,
                        None,
                        serde_json::json!({ "reason": "send_failed" }),
                    )
                    .await?;
                self.store.set_worker_run(worker_id, None).await?;
                dead.push(worker_id.clone());
            }
        }
        for worker_id in dead {
            links.remove(&worker_id);
        }
        Ok(())
    }

    /// The oldest queued run that may actually be assigned: a sweep child is
    /// held back while its sweep already has `concurrency` children
    /// assigned/running (0 = unlimited). Non-sweep runs are never held.
    pub(super) async fn next_assignable_queued(&self) -> Result<Option<RunRecord>> {
        let queued = crate::store::RunQuery {
            state: Some(RunState::Queued),
            ..Default::default()
        };
        // The store pages newest-first; scan a generous window and walk it
        // oldest-first. A backlog deeper than the window waits its turn.
        let page = self.store.runs(&queued, QUEUE_SCAN_LIMIT).await?;
        for summary in page.into_iter().rev() {
            if let Some(sweep_id) = &summary.sweep_id {
                if let Some(sweep_row) = self.store.sweep(sweep_id).await? {
                    if sweep_row.concurrency > 0
                        && self.store.sweep_active_children(sweep_id).await?
                            >= sweep_row.concurrency
                    {
                        continue;
                    }
                }
            }
            return self.store.run(&summary.id).await;
        }
        Ok(None)
    }

    /// Translate a run into a compute-generic [`wire::Job`]. For a train run this
    /// resolves the entrypoint from the code-unit manifest, mints a scoped grant,
    /// records the resume slice history, and adds a `Resumed` event.
    async fn assemble_job(&self, run_id: &RunId, spec: RunSpec) -> Result<wire::Job> {
        match spec {
            RunSpec::Shell(shell) => Ok(jobspec::shell_job(run_id, &shell)),
            RunSpec::Train(train) => self.assemble_train_job(run_id, &train).await,
        }
    }

    async fn assemble_train_job(&self, run_id: &RunId, train: &TrainSpec) -> Result<wire::Job> {
        let unit = self
            .store
            .code_unit(&train.code.name, &train.code.sha)
            .await?
            .with_context(|| format!("code unit {} not registered", train.code))?;
        let entrypoint_cmd = unit
            .manifest
            .entrypoint(&train.entrypoint)
            .with_context(|| format!("entrypoint {:?} not in unit manifest", train.entrypoint))?
            .to_owned();
        let grant = self.grants.mint(run_id.clone(), train.code.clone());
        let code_unit_uri = keys::code_unit_tarball(&train.code.name, &train.code.sha);

        // Resume from the latest uploaded checkpoint, if any, and remember the
        // slice history so this session's checkpoints extend it correctly.
        let latest = self.store.latest_checkpoint(run_id).await?;
        let (resume, resume_state) = match &latest {
            Some(ckpt) => (
                Some((
                    keys::checkpoint_file(&run_id.0, ckpt.step, CHECKPOINT_MODEL_FILE),
                    keys::checkpoint_file(&run_id.0, ckpt.step, CHECKPOINT_META_FILE),
                )),
                ResumeState {
                    from_step: ckpt.step,
                    base_slices: ckpt.meta.slices.clone(),
                },
            ),
            None => (None, ResumeState::default()),
        };
        self.resume_state
            .lock()
            .expect("resume_state lock")
            .insert(run_id.clone(), resume_state);
        if let Some(ckpt) = &latest {
            self.store
                .add_event(
                    run_id,
                    EventKind::Resumed,
                    serde_json::json!({ "from_step": ckpt.step }),
                )
                .await?;
        }

        let resume_staging = resume.as_ref().map(|(model, meta)| ResumeStaging {
            model_uri: model,
            meta_uri: meta,
        });
        let data_staging = self.resolve_data(run_id, &train.data).await?;
        Ok(jobspec::train_job(
            run_id,
            train,
            &TrainStaging {
                entrypoint_cmd: &entrypoint_cmd,
                code_unit_uri: &code_unit_uri,
                grant: &grant,
                resume: resume_staging,
                data: data_staging,
            },
        ))
    }

    /// Requeue any run stuck in `Assigned` past [`chuk_train_proto::ASSIGNMENT_STUCK_TIMEOUT`]
    /// without a `JobStarted` confirmation. `Hub::send_to` only confirms that
    /// the worker's local outbound channel accepted the write, not that the
    /// websocket actually delivered it — a link that still looks open in
    /// `links` can have gone stale underneath, silently swallowing an
    /// `AssignJob` with nothing else left to ever move the run off `Assigned`
    /// (live-observed: a throwaway probe run sat `Assigned` well past its own
    /// job timeout while the worker's heartbeat stayed fresh). Same backstop
    /// shape as `reap_stale_workers`, for this different failure mode.
    pub async fn reap_stuck_assignments(&self) -> Result<()> {
        self.reap_stuck_assignments_older_than(ASSIGNMENT_STUCK_TIMEOUT.as_secs_f64())
            .await
    }

    /// Test seam for [`Self::reap_stuck_assignments`]: takes the max age
    /// directly so a test can force an immediate sweep instead of waiting out
    /// the real timeout.
    pub(super) async fn reap_stuck_assignments_older_than(&self, max_age_s: f64) -> Result<()> {
        let assigned = crate::store::RunQuery {
            state: Some(RunState::Assigned),
            ..Default::default()
        };
        let cutoff = now() - max_age_s;
        for summary in self.store.runs(&assigned, QUEUE_SCAN_LIMIT).await? {
            if summary.updated_at > cutoff {
                continue;
            }
            warn!(
                run = %summary.id.0, worker = ?summary.worker_id,
                "assignment never confirmed; requeueing"
            );
            self.grants.revoke_run(&summary.id);
            if let Some(worker_id) = &summary.worker_id {
                self.store.set_worker_run(worker_id, None).await?;
            }
            self.store
                .transition(
                    &summary.id,
                    RunState::Queued,
                    None,
                    None,
                    serde_json::json!({ "reason": "assignment_timed_out" }),
                )
                .await?;
        }
        self.pump().await
    }

    /// Dispatch-time `data:` resolution (spec §6/§7.3): pre-warm the resolve
    /// call the worker's own client would otherwise make on first fetch, so
    /// the common case never blocks on the chuk-datasets server. Records the
    /// resolved identity for `messages::complete_meta` to read back later.
    pub(super) async fn resolve_data(
        &self,
        run_id: &RunId,
        data: &Option<chuk_train_proto::DataRef>,
    ) -> Result<Option<DataStaging>> {
        let Some(data_ref) = data else { return Ok(None) };
        let datasets = self.datasets.as_ref().with_context(|| {
            format!(
                "run declares a data: block ({}) but CHUK_DATASETS_URL/CHUK_DATASETS_API_KEY are not configured",
                data_ref.dataset
            )
        })?;
        let resolved = datasets
            .resolve(&data_ref.dataset, data_ref.plan.as_deref())
            .await
            .with_context(|| format!("resolving dataset {}", data_ref.dataset))?;
        let plan_sha = resolved.plan.as_ref().map(|p| p.plan_sha.clone());
        let shards = resolved
            .manifest
            .shards
            .iter()
            .map(|shard| {
                let uri = chuk_datasets_client::shard_url(&resolved, &shard.sha256)
                    .with_context(|| format!("no fetchable location for shard {}", shard.sha256))?;
                Ok(ShardInput { uri, sha256: shard.sha256.clone() })
            })
            .collect::<Result<Vec<_>>>()?;
        self.data_state.lock().expect("data_state lock").insert(
            run_id.clone(),
            DataState {
                content_sha: resolved.content_sha.clone(),
                plan_sha: plan_sha.clone(),
                dataset_label: dataset_label(&data_ref.dataset, &resolved.content_sha),
            },
        );
        Ok(Some(DataStaging { content_sha: resolved.content_sha, plan_sha, shards }))
    }
}

/// Whether an idle connected worker may be handed a queued run. It must be free
/// (no current run), not draining toward its lease wall (spec §4), and not
/// unreachable — a worker silent past the heartbeat timeout gets no new work
/// (spec §7) so we never assign to a frozen tab.
pub(super) fn eligible_for_assignment(worker: &WorkerInfo, unreachable_after_s: f64) -> bool {
    worker.current_run.is_none()
        && !worker
            .lease
            .as_ref()
            .is_some_and(|l| l.state == LeaseState::Draining)
        && worker.heartbeat_age_s <= unreachable_after_s
}

/// The `name@sha256:<content_sha>` lineage string `CheckpointMeta.datasets`
/// expects (the convention [`chuk_train_proto::CodeRef`]'s `Display` already
/// uses for code). `asked` is the run's `data.dataset` ref, `<name>@<sha>` or
/// a bare sha; only the name (if any) survives resolution unchanged.
pub(super) fn dataset_label(asked: &str, content_sha: &str) -> String {
    match asked.rsplit_once('@') {
        Some((name, _)) => format!("{name}@sha256:{content_sha}"),
        None => format!("sha256:{content_sha}"),
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}
