//! chuk-experiments-server reporting mirror (spec §11.6).
//!
//! Optional and gated: constructed only when `CHUK_EXPERIMENTS_URL` +
//! `CHUK_EXPERIMENTS_API_KEY` are set, and **every method is best-effort** — a
//! slow or down experiments-server logs a warning and never blocks or fails a
//! run. The harness's own store stays the source of truth (design principle 10).
//!
//! The experiments-server is the research system-of-record; its `RUN-…` ids name
//! *logical runs*, while ours (`EXEC-…`) name *execution attempts* — two parallel,
//! independent namespaces. A submitted run reports in one of two modes:
//!
//! - **Attached** — the caller supplied an `experiment_ref` (an existing logical
//!   `RUN-…`). We report *into* that run: link our execution to it
//!   (`harness_session_id` = our `EXEC-…` id) and address later lifecycle/artifact
//!   reports at it. We never mint a second run for the same intent.
//! - **Unattached** — no ref. We create a fresh run on their side (they mint the
//!   `RUN-…`), send our id as its `slug` + `harness_session_id`, and persist the
//!   minted id on our run row.
//!
//! Either way the minted/attached id is stored as `experiments_run_id`, and the
//! harness's own store stays the source of truth.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chuk_compute_wire::API_PREFIX;
use chuk_train_proto::{
    env, CheckpointMeta, RunId, RunSpec, RunState, TrainSpec, CHECKPOINT_MODEL_FILE,
    DEFAULT_EXPERIMENTS_EXPERIMENT, DEFAULT_EXPERIMENTS_EXPERIMENT_TITLE,
    DEFAULT_EXPERIMENTS_PROGRAMME, DEFAULT_EXPERIMENTS_PROGRAMME_TITLE,
};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::store::Store;

/// Out-link kind (in `TrainSpec.links`) that carries the Weights & Biases URL.
const LINK_KIND_WANDB: &str = "wandb";
/// Artifact URI schemes the experiments-server accepts (else it 400s).
const ACCEPTED_URI_SCHEMES: [&str; 4] = ["s3://", "gdrive://", "https://", "http://"];
/// Max outbox rows retried per sweep tick.
const OUTBOX_SWEEP_BATCH: i64 = 20;
/// Retry backoff for a failed outbox event: `BASE * 2^attempts`, capped at MAX.
const OUTBOX_BASE_BACKOFF_SECS: u64 = 30;
const OUTBOX_MAX_BACKOFF_SECS: u64 = 3_600;
/// The shared default bearer a test client reports with (see [`Experiments::at`]).
#[cfg(test)]
const SHARED_KEY: &str = "shared-write-key";

/// Serializes every test in this crate that mutates the process-global
/// `CHUK_EXPERIMENTS_*` env vars — `from_env`'s own test here, and
/// `hub::tests`'s mirror test, which sets them just long enough to construct a
/// client. Without it `cargo test`'s default parallelism interleaves the two
/// and one reads an env the other is halfway through changing. Same convention
/// (and the same poison-recovering std `Mutex`) as `artifacts`'s S3 env lock.
#[cfg(test)]
pub(crate) fn lock_experiments_env() -> std::sync::MutexGuard<'static, ()> {
    static EXPERIMENTS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    EXPERIMENTS_ENV_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Reporting client for one experiments-server + default programme/experiment.
pub struct Experiments {
    http: Client,
    /// Base URL, trimmed of any trailing slash (e.g. `https://…fly.dev`).
    base: String,
    /// Raw WRITE bearer token (experiments-server keys are not `ck_`-prefixed).
    key: String,
    programme: String,
    programme_title: String,
    experiment: String,
    experiment_title: String,
    /// Our own public base, used to build stable per-checkpoint resolver URLs.
    public_url: String,
    store: Arc<dyn Store>,
    /// The default experiment is ensured (created-or-exists) exactly once.
    ensured: AtomicBool,
    /// Decrypts a user's linked personal chuk-experiments-server key
    /// (`bearer_for`). `None` — the per-user-key feature is off — unless
    /// `CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY` is set and valid.
    key_encryption_key: Option<[u8; 32]>,
}

impl Experiments {
    /// Build from the environment. Returns `None` — the mirror is off — unless
    /// both the URL and an API key are set. Programme/experiment slugs default to
    /// [`DEFAULT_EXPERIMENTS_PROGRAMME`]/[`DEFAULT_EXPERIMENTS_EXPERIMENT`].
    pub fn from_env(store: Arc<dyn Store>, public_url: &str) -> Option<Arc<Self>> {
        let base = std::env::var(env::EXPERIMENTS_URL)
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_owned())
            .filter(|s| !s.is_empty())?;
        let key = std::env::var(env::EXPERIMENTS_API_KEY)
            .ok()
            .filter(|s| !s.is_empty())?;
        let non_empty = |var: &str, default: &str| {
            std::env::var(var)
                .ok()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| default.to_owned())
        };
        Some(Arc::new(Self {
            http: Client::new(),
            base,
            key,
            programme: non_empty(env::EXPERIMENTS_PROGRAMME, DEFAULT_EXPERIMENTS_PROGRAMME),
            programme_title: DEFAULT_EXPERIMENTS_PROGRAMME_TITLE.to_owned(),
            experiment: non_empty(env::EXPERIMENTS_EXPERIMENT, DEFAULT_EXPERIMENTS_EXPERIMENT),
            experiment_title: DEFAULT_EXPERIMENTS_EXPERIMENT_TITLE.to_owned(),
            public_url: public_url.trim().trim_end_matches('/').to_owned(),
            store,
            ensured: AtomicBool::new(false),
            key_encryption_key: crate::crypto::key_from_env(),
        }))
    }

    /// A client pointed at a fake experiments-server on `base`, bypassing the
    /// env gating — for tests that spin one up (mirrors `datasets.rs`'s
    /// `Datasets::at`). `key_encryption_key` turns the per-user-key feature on.
    #[cfg(test)]
    pub(crate) fn at(
        base: &str,
        store: Arc<dyn Store>,
        public_url: &str,
        key_encryption_key: Option<[u8; 32]>,
    ) -> Arc<Self> {
        Arc::new(Self {
            http: Client::new(),
            base: base.trim_end_matches('/').to_owned(),
            key: SHARED_KEY.to_owned(),
            programme: DEFAULT_EXPERIMENTS_PROGRAMME.to_owned(),
            programme_title: DEFAULT_EXPERIMENTS_PROGRAMME_TITLE.to_owned(),
            experiment: DEFAULT_EXPERIMENTS_EXPERIMENT.to_owned(),
            experiment_title: DEFAULT_EXPERIMENTS_EXPERIMENT_TITLE.to_owned(),
            public_url: public_url.trim_end_matches('/').to_owned(),
            store,
            ensured: AtomicBool::new(false),
            key_encryption_key,
        })
    }

    /// Resolve which bearer token to use for chuk-experiments-server calls
    /// made on `email`'s behalf: their own linked key if they have one (and
    /// it decrypts cleanly), else the shared default. Never fails — any
    /// missing link in the chain (per-user-key feature off, no email, no
    /// linked key, bad ciphertext) falls back silently, same principle as the
    /// rest of this module: the mirror never blocks or fails a run over this.
    async fn bearer_for_email(&self, email: Option<&str>) -> String {
        let Some(key_encryption_key) = &self.key_encryption_key else {
            return self.key.clone();
        };
        let Some(email) = email else {
            return self.key.clone();
        };
        let Ok(Some(encrypted)) = self.store.user_experiments_key(email).await else {
            return self.key.clone();
        };
        crate::crypto::decrypt(key_encryption_key, &encrypted).unwrap_or_else(|_| self.key.clone())
    }

    /// Resolve the bearer token for a run already known to our store, by
    /// looking up its `created_by` email first. See [`Self::bearer_for_email`],
    /// which also covers the personal-key-feature-off short circuit; this
    /// wrapper additionally skips the store lookup in that same case.
    async fn bearer_for(&self, run_id: &RunId) -> String {
        if self.key_encryption_key.is_none() {
            return self.key.clone();
        }
        let Ok(Some(run)) = self.store.run(run_id).await else {
            return self.key.clone();
        };
        self.bearer_for_email(run.summary.created_by.as_deref()).await
    }

    // ---- best-effort entrypoints (the hub spawns these) -------------------
    //
    // Each of these persists an outbox row before making the first delivery
    // attempt, so a transient failure (network blip, experiments-server 5xx)
    // is retried by `run_outbox_loop` instead of silently dropping the
    // observation — the guarantee this module never blocks or fails a real
    // run stays exactly as before; only "log a warning and forget" changes.

    /// Mirror a freshly-submitted run. Only **train** runs are reported; shell
    /// probes are skipped (and their later transitions no-op, having no id).
    /// With an `experiment_ref` we attach to that existing logical run; without
    /// one we create a fresh run on the experiments-server.
    pub async fn report_created(&self, run_id: RunId, spec: RunSpec, experiment_ref: Option<String>) {
        if !matches!(spec, RunSpec::Train(_)) {
            return;
        }
        self.enqueue_and_attempt(&run_id, OutboxEvent::Created { spec, experiment_ref })
            .await;
    }

    /// Mirror a run state transition (running / terminal). No-op for a run that
    /// was never mirrored (shell run, or the create is still in flight — the
    /// state event just waits in the outbox until the created event lands).
    /// On `Completed`, also enqueues one durable `Result` event per final metric.
    pub async fn report_state(&self, run_id: RunId, state: RunState) {
        self.enqueue_and_attempt(&run_id, OutboxEvent::State { state }).await;
        if matches!(state, RunState::Completed) {
            match self.store.metric_series(&run_id, None, 0, 0).await {
                Ok(series) => {
                    for (name, points) in series.series {
                        if let Some(last) = points.last() {
                            self.enqueue_and_attempt(&run_id, OutboxEvent::Result { name, value: last.value })
                                .await;
                        }
                    }
                }
                Err(e) => warn!(run = %run_id.0, error = %e, "experiments: reading final metrics failed"),
            }
        }
    }

    /// Register an uploaded checkpoint as a `checkpoint` artifact.
    pub async fn report_checkpoint(&self, run_id: RunId, step: u64, uri: String, meta: CheckpointMeta) {
        self.enqueue_and_attempt(
            &run_id,
            OutboxEvent::Checkpoint { step, uri, meta: Box::new(meta) },
        )
        .await;
    }

    /// Persist an outbox row for `event`, then make one immediate delivery
    /// attempt (keeps today's low-latency common case; a failure just leaves
    /// the row for `run_outbox_loop` to retry).
    async fn enqueue_and_attempt(&self, run_id: &RunId, event: OutboxEvent) {
        let payload = match serde_json::to_string(&event) {
            Ok(p) => p,
            Err(e) => {
                warn!(run = %run_id.0, error = %e, "experiments-outbox: serializing event failed");
                return;
            }
        };
        let id = match self
            .store
            .enqueue_outbox_event(run_id, event.kind_label(), &payload, now_secs())
            .await
        {
            Ok(id) => id,
            Err(e) => {
                warn!(run = %run_id.0, error = %e, "experiments-outbox: enqueue failed (event not durable)");
                return;
            }
        };
        self.attempt(id, run_id, event, 0).await;
    }

    /// Try delivering one outbox event. On success, marks it done; on failure,
    /// records the error and reschedules it with backoff. `attempts_so_far` is
    /// how many prior attempts already failed (0 for a fresh event), used to
    /// compute how long to wait before the *next* attempt.
    async fn attempt(&self, id: i64, run_id: &RunId, event: OutboxEvent, attempts_so_far: i64) -> bool {
        let kind = event.kind_label();
        let result = match event {
            OutboxEvent::Created { spec, experiment_ref } => match spec {
                RunSpec::Train(train) => match experiment_ref {
                    Some(ext) => self.try_attach(run_id, &train, &ext).await,
                    None => self.try_create(run_id, &train).await,
                },
                // Only Train specs are ever enqueued (see report_created); a
                // stray non-Train row has nothing to deliver.
                _ => Ok(()),
            },
            OutboxEvent::State { state } => self.try_state(run_id, state).await,
            OutboxEvent::Checkpoint { step, uri, meta } => {
                self.try_checkpoint(run_id, step, &uri, &meta).await
            }
            OutboxEvent::Result { name, value } => self.try_result(run_id, &name, value).await,
        };
        match result {
            Ok(()) => {
                if let Err(e) = self.store.mark_outbox_event_done(id).await {
                    warn!(id, error = %e, "experiments-outbox: marking event done failed");
                }
                true
            }
            Err(e) => {
                let next_attempt_at = now_secs() + backoff_for(attempts_so_far).as_secs_f64();
                warn!(
                    run = %run_id.0, kind, attempts = attempts_so_far, error = %e,
                    "experiments-outbox: attempt failed, will retry"
                );
                if let Err(store_err) = self
                    .store
                    .mark_outbox_event_failed(id, &e.to_string(), next_attempt_at)
                    .await
                {
                    warn!(id, error = %store_err, "experiments-outbox: recording failure failed");
                }
                false
            }
        }
    }

    /// One pass over due outbox events, oldest first. Sequential (not
    /// concurrent) so a run's later events never get retried ahead of its own
    /// not-yet-delivered `created` event.
    async fn sweep_outbox_once(&self) -> Result<usize> {
        let due = self.store.due_outbox_events(now_secs(), OUTBOX_SWEEP_BATCH).await?;
        let mut delivered = 0;
        for row in due {
            let event: OutboxEvent = match serde_json::from_str(&row.payload) {
                Ok(e) => e,
                Err(e) => {
                    // Can never be replayed successfully — drop it rather than
                    // retry a parse error forever every sweep.
                    warn!(id = row.id, error = %e, "experiments-outbox: corrupt payload, dropping");
                    if let Err(e) = self.store.mark_outbox_event_done(row.id).await {
                        warn!(id = row.id, error = %e, "experiments-outbox: dropping corrupt row failed");
                    }
                    continue;
                }
            };
            if self.attempt(row.id, &row.run_id, event, row.attempts).await {
                delivered += 1;
            }
        }
        Ok(delivered)
    }

    /// Runs for the life of the process, retrying undelivered outbox events.
    pub async fn run_outbox_loop(self: Arc<Self>, interval: Duration) {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            match self.sweep_outbox_once().await {
                Ok(n) if n > 0 => info!(delivered = n, "experiments-outbox sweep"),
                Ok(_) => {}
                Err(e) => warn!(error = %e, "experiments-outbox sweep failed"),
            }
        }
    }

    /// Ensure the default programme/experiment exists. Public so startup can
    /// validate the config early; also called lazily before the first create.
    pub async fn ensure(&self) -> Result<()> {
        if self.ensured.load(Ordering::Relaxed) {
            return Ok(());
        }
        let resp = self
            .http
            .post(format!("{}/v1/experiments", self.base))
            .bearer_auth(&self.key)
            .json(&json!({
                "programme": self.programme,
                "programme_name": self.programme_title,
                "slug": self.experiment,
                "title": self.experiment_title,
                "status": "running",
            }))
            .send()
            .await
            .context("POST /v1/experiments")?;
        let status = resp.status();
        // 409 = the experiment already exists, which is exactly what we want.
        if status.is_success() || status == StatusCode::CONFLICT {
            self.ensured.store(true, Ordering::Relaxed);
            Ok(())
        } else {
            anyhow::bail!("ensure experiment: {status} {}", body_of(resp).await);
        }
    }

    /// Fetch a chuk-experiments-server run's own record — used by
    /// `Hub::submit_from_experiment` to build a `TrainSpec` straight from its
    /// `config`/`workspec` rather than requiring the caller to re-specify the
    /// training job by hand. `created_by` resolves the bearer token the same
    /// way the outbound reports do (the submitting user's own linked key,
    /// falling back to the shared default) — see [`Self::bearer_for_email`].
    pub async fn fetch_run(&self, ext_id: &str, created_by: Option<&str>) -> Result<ExperimentsRunSnapshot> {
        let token = self.bearer_for_email(created_by).await;
        let resp = self
            .http
            .get(format!("{}/v1/runs/{}", self.base, ext_id))
            .bearer_auth(&token)
            .send()
            .await
            .context("GET run")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("fetch run: {status} {}", body_of(resp).await);
        }
        resp.json().await.context("parse run")
    }

    // ---- the actual reports ------------------------------------------------

    async fn try_create(&self, run_id: &RunId, train: &TrainSpec) -> Result<()> {
        self.ensure().await?;
        let token = self.bearer_for(run_id).await;
        let config = json!({
            "entrypoint": train.entrypoint,
            "config_path": train.config,
            "overrides": train.overrides,
            "seed": train.seed,
            "arch": train.arch,
            "code": { "name": train.code.name, "sha": train.code.sha },
        });
        let resp = self
            .http
            .post(format!("{}/v1/experiments/{}/runs", self.base, self.experiment))
            .bearer_auth(&token)
            .json(&json!({
                "slug": run_id.0,
                "config": config,
                "budget_seconds": train.timeout_s,
                "workspec": { "entrypoint": train.entrypoint, "code": { "name": train.code.name, "sha": train.code.sha } },
                "status": "queued",
            }))
            .send()
            .await
            .context("POST run")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("create run: {status} {}", body_of(resp).await);
        }
        let created: CreatedRun = resp.json().await.context("parse created run")?;
        // Best-effort; the primary link is `slug` = our id even if this PATCH fails.
        if let Err(e) = self
            .patch_run(&created.id, linkback_patch(run_id, train), &token)
            .await
        {
            warn!(run = %run_id.0, ext = %created.id, error = %e, "experiments: linkback patch failed");
        }
        self.store.set_experiments_run_id(run_id, &created.id).await?;
        info!(run = %run_id.0, ext = %created.id, "experiments: run mirrored");
        Ok(())
    }

    /// Attach this execution to a logical run the caller already created on the
    /// experiments-server (`ext`). Unlike [`Self::try_create`], we did not mint
    /// the run, so the linkback PATCH doubles as an existence check: if it fails
    /// we bail *without* recording the id, rather than committing to a bad ref
    /// that would make every later report warn.
    async fn try_attach(&self, run_id: &RunId, train: &TrainSpec, ext: &str) -> Result<()> {
        let token = self.bearer_for(run_id).await;
        self.patch_run(ext, linkback_patch(run_id, train), &token).await?;
        self.store.set_experiments_run_id(run_id, ext).await?;
        info!(run = %run_id.0, ext = %ext, "experiments: execution attached to existing run");
        Ok(())
    }

    /// `run_id` not yet mirrored (its `Created` event is still pending in the
    /// outbox) is a **retryable failure** here, not a silent no-op — this event
    /// will succeed on a later sweep once `Created` lands, so it must stay
    /// pending rather than being marked delivered without ever being sent.
    async fn try_state(&self, run_id: &RunId, state: RunState) -> Result<()> {
        let Some(ext) = self.store.experiments_run_id(run_id).await? else {
            anyhow::bail!("run not yet mirrored (created event still pending)");
        };
        let Some(status) = map_status(state) else {
            return Ok(());
        };
        let mut body = json!({ "status": status });
        let stamp = rfc3339(now_secs());
        if matches!(state, RunState::Running) {
            body["started_at"] = json!(stamp);
        } else if state.is_terminal() {
            body["ended_at"] = json!(stamp);
        }
        let token = self.bearer_for(run_id).await;
        self.patch_run(&ext, body, &token).await
    }

    async fn try_checkpoint(
        &self,
        run_id: &RunId,
        step: u64,
        recorded_uri: &str,
        meta: &CheckpointMeta,
    ) -> Result<()> {
        let Some(ext) = self.store.experiments_run_id(run_id).await? else {
            anyhow::bail!("run not yet mirrored (created event still pending)");
        };
        let Some(uri) = self.artifact_uri(run_id, step, recorded_uri) else {
            anyhow::bail!("no experiments-server-accepted uri for checkpoint (uri={recorded_uri})");
        };
        // Group a model's checkpoints as versions: arch, else code unit, else run.
        let name = meta
            .arch
            .clone()
            .or_else(|| meta.code.as_ref().map(|c| c.name.clone()))
            .unwrap_or_else(|| run_id.0.clone());
        let token = self.bearer_for(run_id).await;
        let resp = self
            .http
            .post(format!("{}/v1/runs/{}/artifacts", self.base, ext))
            .bearer_auth(&token)
            .json(&json!({
                "kind": "checkpoint",
                "uri": uri,
                "role": "produced",
                "name": name,
                "meta": meta,
            }))
            .send()
            .await
            .context("POST artifact")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("register artifact: {status} {}", body_of(resp).await);
        }
        Ok(())
    }

    async fn try_result(&self, run_id: &RunId, name: &str, value: f64) -> Result<()> {
        let Some(ext) = self.store.experiments_run_id(run_id).await? else {
            anyhow::bail!("run not yet mirrored (created event still pending)");
        };
        let token = self.bearer_for(run_id).await;
        let resp = self
            .http
            .post(format!("{}/v1/runs/{}/results", self.base, ext))
            .bearer_auth(&token)
            .json(&json!({ "name": name, "value": value }))
            .send()
            .await
            .context("POST result")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("submit result: {status} {}", body_of(resp).await);
        }
        Ok(())
    }

    async fn patch_run(&self, ext_id: &str, body: Value, token: &str) -> Result<()> {
        let resp = self
            .http
            .patch(format!("{}/v1/runs/{}", self.base, ext_id))
            .bearer_auth(token)
            .json(&body)
            .send()
            .await
            .context("PATCH run")?;
        let status = resp.status();
        if !status.is_success() {
            anyhow::bail!("patch run: {status} {}", body_of(resp).await);
        }
        Ok(())
    }

    /// Choose the artifact URI. Prefer our stable per-checkpoint resolver URL
    /// (browsable, resolves R2-or-Drive, survives lifecycle expiry) whenever we
    /// have an http(s) public base — which is the normal case, and in prod is the
    /// real Fly host. Fall back to the recorded uri only if it already carries an
    /// accepted scheme (e.g. a raw `s3://`); else `None` (e.g. a bare `file://`
    /// path with no public base), and the checkpoint is skipped with a warning.
    fn artifact_uri(&self, run_id: &RunId, step: u64, recorded: &str) -> Option<String> {
        if self.public_url.starts_with("https://") || self.public_url.starts_with("http://") {
            return Some(format!(
                "{}{API_PREFIX}/checkpoint/{}/{}/{}",
                self.public_url, run_id.0, step, CHECKPOINT_MODEL_FILE
            ));
        }
        ACCEPTED_URI_SCHEMES
            .iter()
            .any(|p| recorded.starts_with(p))
            .then(|| recorded.to_owned())
    }
}

#[derive(serde::Deserialize)]
struct CreatedRun {
    id: String,
}

/// The subset of a chuk-experiments-server run record needed to submit it as
/// a harness execution (`Hub::submit_from_experiment`). Deliberately narrow —
/// not the server's full `Run` shape — and tolerant of unknown fields.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ExperimentsRunSnapshot {
    pub status: String,
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub workspec: Value,
    #[serde(default)]
    pub budget_seconds: Option<u64>,
    #[serde(default)]
    pub harness_session_id: Option<String>,
}

/// A durable, replayable mirror event. Stored as opaque serialized JSON in the
/// outbox (the store layer never inspects it, same as `runs.spec`); deserialized
/// back by `sweep_outbox_once` to retry a failed delivery.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OutboxEvent {
    Created {
        spec: RunSpec,
        experiment_ref: Option<String>,
    },
    State {
        state: RunState,
    },
    Checkpoint {
        step: u64,
        uri: String,
        meta: Box<CheckpointMeta>,
    },
    Result {
        name: String,
        value: f64,
    },
}

impl OutboxEvent {
    /// Human-readable label stored alongside the payload (logging/debugging
    /// only — dispatch always matches on the deserialized enum itself).
    fn kind_label(&self) -> &'static str {
        match self {
            OutboxEvent::Created { .. } => "created",
            OutboxEvent::State { .. } => "state",
            OutboxEvent::Checkpoint { .. } => "checkpoint",
            OutboxEvent::Result { .. } => "result",
        }
    }
}

/// The linkback body applied to the experiments-server run (whether freshly
/// created or attached): our execution id as the correlation field, plus any
/// Weights & Biases out-link declared on the run.
fn linkback_patch(run_id: &RunId, train: &TrainSpec) -> Value {
    let mut patch = json!({ "harness_session_id": run_id.0 });
    if let Some(url) = train
        .links
        .iter()
        .find(|l| l.kind.eq_ignore_ascii_case(LINK_KIND_WANDB))
        .map(|l| l.url.clone())
    {
        patch["wandb_url"] = json!(url);
    }
    patch
}

/// Build a `TrainSpec` from a chuk-experiments-server run, for
/// `Hub::submit_from_experiment`. Prefers `config` — the richer shape
/// `try_create` itself writes (`entrypoint`, `config_path`, `overrides`,
/// `seed`, `arch`, `code.{name,sha}`) — falling back to `workspec` (which only
/// ever carries `entrypoint`/`code`, per `RunCreate`'s "everything a harness
/// worker needs, no other context" contract) when `config` doesn't have them.
/// Errors clearly rather than submitting a half-built spec.
pub(crate) fn train_spec_from_experiments_run(run: &ExperimentsRunSnapshot) -> Result<TrainSpec> {
    let entrypoint = run
        .config
        .get("entrypoint")
        .and_then(Value::as_str)
        .or_else(|| run.workspec.get("entrypoint").and_then(Value::as_str))
        .context("run has no entrypoint in config or workspec")?;

    let code = run
        .config
        .get("code")
        .filter(|c| !c.is_null())
        .or_else(|| run.workspec.get("code"))
        .context("run has no code reference in config or workspec")?;
    let code_name = code
        .get("name")
        .and_then(Value::as_str)
        .context("run's code reference has no name")?;
    let code_sha = code
        .get("sha")
        .and_then(Value::as_str)
        .context("run's code reference has no sha")?;

    let mut spec = json!({
        "code": { "name": code_name, "sha": code_sha },
        "entrypoint": entrypoint,
    });
    if let Some(v) = run.config.get("config_path").filter(|v| !v.is_null()) {
        spec["config"] = v.clone();
    }
    if let Some(v) = run.config.get("overrides").filter(|v| !v.is_null()) {
        spec["overrides"] = v.clone();
    }
    if let Some(v) = run.config.get("seed").filter(|v| !v.is_null()) {
        spec["seed"] = v.clone();
    }
    if let Some(v) = run.config.get("arch").filter(|v| !v.is_null()) {
        spec["arch"] = v.clone();
    }
    if let Some(secs) = run.budget_seconds {
        spec["timeout_s"] = json!(secs);
    }

    serde_json::from_value(spec).context("building TrainSpec from experiments-server run")
}

/// Read a response body for an error message, tolerating a read failure.
async fn body_of(resp: reqwest::Response) -> String {
    resp.text().await.unwrap_or_default()
}

/// Map our run state to the experiments-server status vocabulary. `None` for the
/// states we don't mirror (queued/assigned — their queue owns those).
fn map_status(state: RunState) -> Option<&'static str> {
    match state {
        RunState::Running => Some("running"),
        RunState::Completed => Some("completed"),
        RunState::Failed => Some("failed"),
        RunState::Cancelled => Some("cancelled"),
        RunState::Queued | RunState::Assigned => None,
    }
}

/// Capped exponential backoff for the `attempts_so_far`-th retry: `30s * 2^n`,
/// clamped to a 1h ceiling. No give-up cutoff — a permanently-misconfigured
/// mirror just retries quietly at the ceiling forever, consistent with never
/// silently losing an observation.
fn backoff_for(attempts_so_far: i64) -> Duration {
    let shift = attempts_so_far.clamp(0, 10) as u32; // 2^10 * 30s already exceeds the cap
    let secs = OUTBOX_BASE_BACKOFF_SECS.saturating_mul(1u64 << shift);
    Duration::from_secs(secs.min(OUTBOX_MAX_BACKOFF_SECS))
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Format UTC unix seconds as an RFC3339 timestamp (seconds precision) — what the
/// experiments-server's `datetime` fields parse. Self-contained so the reporter
/// stays decoupled from the store's date helpers.
fn rfc3339(secs: f64) -> String {
    let s = secs as i64;
    let (days, rem) = (s.div_euclid(86_400), s.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Howard Hinnant's civil-from-days: days since the unix epoch → (year, month,
/// day), proleptic Gregorian. Avoids a date-crate dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_utc_seconds() {
        assert_eq!(rfc3339(1_609_459_200.0), "2021-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_609_545_600.0), "2021-01-02T00:00:00Z");
        // A well-known unix timestamp with a non-midnight time-of-day.
        assert_eq!(rfc3339(1_700_000_000.0), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn maps_only_reportable_states() {
        assert_eq!(map_status(RunState::Running), Some("running"));
        assert_eq!(map_status(RunState::Completed), Some("completed"));
        assert_eq!(map_status(RunState::Failed), Some("failed"));
        assert_eq!(map_status(RunState::Queued), None);
        assert_eq!(map_status(RunState::Assigned), None);
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_for(0), Duration::from_secs(30));
        assert_eq!(backoff_for(1), Duration::from_secs(60));
        assert_eq!(backoff_for(2), Duration::from_secs(120));
        // Caps at 1h well before the shift could overflow.
        assert_eq!(backoff_for(20), Duration::from_secs(3_600));
        assert_eq!(backoff_for(-1), Duration::from_secs(30)); // clamps negative to 0
    }

    fn train_with_links(links: Value) -> TrainSpec {
        serde_json::from_value(json!({
            "code": { "name": "gpt-nano", "sha": "abc123" },
            "entrypoint": "train",
            "links": links,
        }))
        .expect("valid TrainSpec")
    }

    #[test]
    fn linkback_carries_execution_id_and_wandb() {
        let run = RunId("EXEC-20260718-160217-00397".into());
        let train = train_with_links(json!([
            { "kind": "wandb", "label": "W&B", "url": "https://wandb.ai/x/y" },
        ]));
        let patch = linkback_patch(&run, &train);
        assert_eq!(patch["harness_session_id"], json!(run.0));
        assert_eq!(patch["wandb_url"], json!("https://wandb.ai/x/y"));
    }

    #[test]
    fn linkback_omits_wandb_when_absent() {
        let run = RunId("EXEC-20260718-160217-00397".into());
        // A non-wandb out-link must not be mistaken for the W&B url.
        let train = train_with_links(json!([
            { "kind": "exp", "label": "Experiment", "url": "https://exp/RUN-1" },
        ]));
        let patch = linkback_patch(&run, &train);
        assert_eq!(patch["harness_session_id"], json!(run.0));
        assert!(patch.get("wandb_url").is_none());
    }

    fn snapshot(config: Value, workspec: Value, budget_seconds: Option<u64>) -> ExperimentsRunSnapshot {
        ExperimentsRunSnapshot {
            status: "queued".into(),
            config,
            workspec,
            budget_seconds,
            harness_session_id: None,
        }
    }

    #[test]
    fn train_spec_builds_from_the_richer_config_shape() {
        let run = snapshot(
            json!({
                "entrypoint": "train",
                "config_path": "configs/base.yaml",
                "overrides": { "lr": 0.001 },
                "seed": 42,
                "arch": "gpt-nano",
                "code": { "name": "tok-v12", "sha": "abc123" },
            }),
            json!({}),
            Some(3600),
        );
        let spec = train_spec_from_experiments_run(&run).expect("valid spec");
        assert_eq!(spec.entrypoint, "train");
        assert_eq!(spec.config.as_deref(), Some("configs/base.yaml"));
        assert_eq!(spec.overrides, json!({ "lr": 0.001 }));
        assert_eq!(spec.seed, Some(42));
        assert_eq!(spec.arch.as_deref(), Some("gpt-nano"));
        assert_eq!(spec.code.name, "tok-v12");
        assert_eq!(spec.code.sha, "abc123");
        assert_eq!(spec.timeout_s, 3600);
    }

    #[test]
    fn train_spec_falls_back_to_workspec_entrypoint_and_code() {
        // config empty ({}), like a run enqueued by hand with only workspec set.
        let run = snapshot(
            json!({}),
            json!({ "entrypoint": "train", "code": { "name": "tok-v12", "sha": "abc123" } }),
            None,
        );
        let spec = train_spec_from_experiments_run(&run).expect("valid spec");
        assert_eq!(spec.entrypoint, "train");
        assert_eq!(spec.code.name, "tok-v12");
        assert_eq!(spec.code.sha, "abc123");
        assert!(spec.config.is_none());
        assert!(spec.seed.is_none());
    }

    #[test]
    fn train_spec_errors_when_entrypoint_missing_everywhere() {
        let run = snapshot(json!({}), json!({}), None);
        let err = train_spec_from_experiments_run(&run).unwrap_err();
        assert!(err.to_string().contains("entrypoint"), "unexpected error: {err}");
    }

    #[test]
    fn train_spec_errors_when_code_reference_missing() {
        let run = snapshot(json!({ "entrypoint": "train" }), json!({}), None);
        let err = train_spec_from_experiments_run(&run).unwrap_err();
        assert!(err.to_string().contains("code reference"), "unexpected error: {err}");
    }

    // -- the mirror itself, against a loopback experiments-server ------------
    //
    // Everything below drives the real reporting path (ensure, create/attach,
    // state, checkpoint, result, and the durable outbox's retry) against
    // `fakehttp`, asserting the REST calls we actually make and — the property
    // that matters most here — that a failed report is never silently lost and
    // never blocks the run. The live checks against a real chuk-experiments-
    // server stay in `experiments/tests.rs`.

    use std::collections::BTreeMap;

    use chuk_train_proto::{OutboxRow, Role};

    use crate::fakehttp::{FakeHttp, Received, Reply, REFUSED_ORIGIN};
    use crate::store::SqliteStore;

    const PUBLIC_URL: &str = "https://cp.example.com";
    const EXT_RUN: &str = "RUN-20260724-0001";

    async fn store() -> Arc<dyn Store> {
        Arc::new(SqliteStore::open(":memory:").await.expect("store"))
    }

    fn train_spec() -> TrainSpec {
        serde_json::from_value(json!({
            "code": { "name": "gpt-nano", "sha": "abc123" },
            "entrypoint": "train",
            "config": "configs/base.yaml",
            "seed": 42,
        }))
        .expect("valid TrainSpec")
    }

    /// A real queued run row, as `Hub::submit` would have created before the
    /// mirror ever sees it (`set_experiments_run_id` needs a row to update).
    async fn queued_run(store: &Arc<dyn Store>, created_by: Option<&str>) -> (RunId, RunSpec) {
        let spec = RunSpec::Train(Box::new(train_spec()));
        let run_id = store
            .create_run("mirror-test", &spec, None, created_by, None)
            .await
            .expect("create_run");
        (run_id, spec)
    }

    /// An experiments-server that accepts everything: the experiment ensure,
    /// run creates (minting [`EXT_RUN`]), patches, artifacts and results.
    fn accepting_server() -> FakeHttp {
        FakeHttp::start(|_, _| Reply::ok(format!(r#"{{"id":"{EXT_RUN}"}}"#)))
    }

    fn paths(server: &FakeHttp) -> Vec<String> {
        server
            .requests()
            .iter()
            .map(|r| format!("{} {}", r.method, r.path()))
            .collect()
    }

    fn body_of_request(server: &FakeHttp, method: &str, path: &str) -> Value {
        server
            .requests()
            .iter()
            .find(|r| r.method == method && r.path() == path)
            .map(Received::json)
            .unwrap_or_else(|| panic!("no {method} {path} in {:?}", paths(server)))
    }

    async fn pending(store: &Arc<dyn Store>) -> Vec<OutboxRow> {
        // Far enough in the future to see rows whatever backoff they carry.
        store
            .due_outbox_events(now_secs() + OUTBOX_MAX_BACKOFF_SECS as f64 + 1.0, 50)
            .await
            .expect("due outbox events")
    }

    // -- ensure --------------------------------------------------------------

    #[tokio::test]
    async fn ensure_creates_the_default_experiment_once_and_treats_409_as_done() {
        let server = accepting_server();
        let store = store().await;
        let exp = Experiments::at(&server.origin, store, PUBLIC_URL, None);

        exp.ensure().await.expect("ensure");
        exp.ensure().await.expect("already ensured");
        assert_eq!(server.hits(), 1, "the default experiment is ensured exactly once");
        let body = body_of_request(&server, "POST", "/v1/experiments");
        assert_eq!(body["programme"], DEFAULT_EXPERIMENTS_PROGRAMME);
        assert_eq!(body["slug"], DEFAULT_EXPERIMENTS_EXPERIMENT);
        assert_eq!(server.requests()[0].header("authorization"), format!("Bearer {SHARED_KEY}"));

        // A 409 means someone else already created it — equally ensured.
        let conflicting = FakeHttp::start(|_, _| Reply::new(409, "already exists"));
        let exp = Experiments::at(&conflicting.origin, store2().await, PUBLIC_URL, None);
        exp.ensure().await.expect("409 is success");
    }

    async fn store2() -> Arc<dyn Store> {
        store().await
    }

    #[tokio::test]
    async fn ensure_surfaces_a_refusal_with_the_servers_own_body() {
        let server = FakeHttp::start(|_, _| Reply::new(500, "boom"));
        let exp = Experiments::at(&server.origin, store().await, PUBLIC_URL, None);
        let error = exp.ensure().await.unwrap_err();
        assert!(error.to_string().contains("ensure experiment"), "unexpected error: {error}");
        assert!(error.to_string().contains("boom"), "the body must survive: {error}");
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_transport_error_on_ensure() {
        let exp = Experiments::at(REFUSED_ORIGIN, store().await, PUBLIC_URL, None);
        let error = exp.ensure().await.unwrap_err();
        assert!(error.to_string().contains("/v1/experiments"), "unexpected error: {error}");
    }

    // -- report_created ------------------------------------------------------

    #[tokio::test]
    async fn a_shell_run_is_never_mirrored() {
        let server = accepting_server();
        let store = store().await;
        let spec: RunSpec = serde_json::from_value(json!({ "kind": "shell", "command": "true" }))
            .expect("valid shell spec");
        let run_id = store.create_run("probe", &spec, None, None, None).await.expect("run");
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, None).await;
        assert_eq!(server.hits(), 0, "shell probes are not research runs");
        assert!(pending(&store).await.is_empty(), "and nothing is queued for retry");
    }

    #[tokio::test]
    async fn an_unattached_run_creates_a_run_on_their_side_and_records_the_minted_id() {
        let server = accepting_server();
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, None).await;

        assert_eq!(
            store.experiments_run_id(&run_id).await.unwrap().as_deref(),
            Some(EXT_RUN)
        );
        let created = body_of_request(
            &server,
            "POST",
            &format!("/v1/experiments/{DEFAULT_EXPERIMENTS_EXPERIMENT}/runs"),
        );
        assert_eq!(created["slug"], run_id.0, "our id is the primary link");
        assert_eq!(created["status"], "queued");
        assert_eq!(created["config"]["entrypoint"], "train");
        assert_eq!(created["config"]["seed"], 42);
        assert_eq!(created["workspec"]["code"]["sha"], "abc123");
        // ...and the linkback patch carries our execution id back to them.
        let patch = body_of_request(&server, "PATCH", &format!("/v1/runs/{EXT_RUN}"));
        assert_eq!(patch["harness_session_id"], run_id.0);
        assert!(pending(&store).await.is_empty(), "delivered, so nothing is left pending");
    }

    #[tokio::test]
    async fn a_failed_linkback_patch_still_records_the_run_as_mirrored() {
        // The run really was created; the patch is best-effort decoration.
        let server = FakeHttp::start(|req, _| match req.method.as_str() {
            "PATCH" => Reply::new(500, "patch failed"),
            _ => Reply::ok(format!(r#"{{"id":"{EXT_RUN}"}}"#)),
        });
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, None).await;
        assert_eq!(
            store.experiments_run_id(&run_id).await.unwrap().as_deref(),
            Some(EXT_RUN),
            "the create landed, so the mapping stands"
        );
    }

    #[tokio::test]
    async fn an_attached_run_patches_the_existing_logical_run_instead_of_minting_one() {
        let server = accepting_server();
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, Some(EXT_RUN.to_owned())).await;

        assert_eq!(
            store.experiments_run_id(&run_id).await.unwrap().as_deref(),
            Some(EXT_RUN)
        );
        assert_eq!(
            paths(&server),
            vec![format!("PATCH /v1/runs/{EXT_RUN}")],
            "no second run is ever minted for the same intent"
        );
    }

    #[tokio::test]
    async fn an_attach_to_a_ref_that_does_not_exist_is_not_recorded_and_stays_pending() {
        let server = FakeHttp::start(|_, _| Reply::new(404, "no such run"));
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, Some("RUN-does-not-exist".into())).await;

        assert!(
            store.experiments_run_id(&run_id).await.unwrap().is_none(),
            "never commit to a bad ref"
        );
        let rows = pending(&store).await;
        assert_eq!(rows.len(), 1, "the observation is durable, not dropped");
        assert_eq!(rows[0].attempts, 1);
    }

    #[tokio::test]
    async fn a_created_report_that_cannot_be_delivered_waits_in_the_outbox() {
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(REFUSED_ORIGIN, store.clone(), PUBLIC_URL, None);

        exp.report_created(run_id.clone(), spec, None).await;

        assert!(store.experiments_run_id(&run_id).await.unwrap().is_none());
        let rows = pending(&store).await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "created");
    }

    // -- report_state / report_checkpoint / results --------------------------

    /// Report a create against `server` and return the mirrored run id.
    async fn mirrored(server: &FakeHttp, store: &Arc<dyn Store>) -> (Arc<Experiments>, RunId) {
        let (run_id, spec) = queued_run(store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);
        exp.report_created(run_id.clone(), spec, None).await;
        (exp, run_id)
    }

    #[tokio::test]
    async fn a_running_transition_patches_status_and_a_start_time() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;

        exp.report_state(run_id, RunState::Running).await;
        let patches: Vec<Value> = server
            .requests()
            .iter()
            .filter(|r| r.method == "PATCH")
            .map(Received::json)
            .collect();
        let state = patches.last().expect("a state patch");
        assert_eq!(state["status"], "running");
        assert!(state["started_at"].as_str().expect("started_at").ends_with('Z'));
        assert!(state.get("ended_at").is_none());
    }

    #[tokio::test]
    async fn a_terminal_transition_patches_an_end_time() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;

        exp.report_state(run_id, RunState::Failed).await;
        let state = server
            .requests()
            .iter()
            .filter(|r| r.method == "PATCH")
            .map(Received::json)
            .next_back()
            .expect("a state patch");
        assert_eq!(state["status"], "failed");
        assert!(state["ended_at"].as_str().expect("ended_at").ends_with('Z'));
    }

    #[tokio::test]
    async fn a_state_their_queue_owns_is_delivered_without_a_patch() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;
        let before = server.hits();

        exp.report_state(run_id, RunState::Assigned).await;
        assert_eq!(server.hits(), before, "assigned/queued are theirs to track");
        assert!(pending(&store).await.is_empty(), "and the event is still marked delivered");
    }

    #[tokio::test]
    async fn completion_also_reports_each_final_metric_as_a_result() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;
        for (step, loss) in [(1u64, 2.5), (2, 1.25)] {
            store
                .append_metrics(&run_id, step, &BTreeMap::from([("loss".to_owned(), loss)]))
                .await
                .expect("append metrics");
        }

        exp.report_state(run_id.clone(), RunState::Completed).await;

        let result = body_of_request(&server, "POST", &format!("/v1/runs/{EXT_RUN}/results"));
        assert_eq!(result["name"], "loss");
        assert_eq!(result["value"], 1.25, "the final value, not the first");
    }

    #[tokio::test]
    async fn a_state_report_for_a_run_that_is_not_mirrored_yet_stays_pending() {
        // Its `created` event is still in flight: the state event must wait for
        // it rather than be marked delivered without ever being sent.
        let server = accepting_server();
        let store = store().await;
        let (run_id, _spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        exp.report_state(run_id, RunState::Running).await;
        let rows = pending(&store).await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "state");
        assert_eq!(rows[0].attempts, 1, "attempted once, and kept for the next sweep");
    }

    fn checkpoint_meta() -> CheckpointMeta {
        serde_json::from_value(json!({
            "run_id": "EXEC-1",
            "step": 500,
            "arch": "gpt-nano",
        }))
        .expect("valid CheckpointMeta")
    }

    #[tokio::test]
    async fn a_checkpoint_is_registered_as_an_artifact_at_our_own_resolver_url() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;

        exp.report_checkpoint(run_id.clone(), 500, "s3://bucket/k".into(), checkpoint_meta())
            .await;

        let artifact = body_of_request(&server, "POST", &format!("/v1/runs/{EXT_RUN}/artifacts"));
        assert_eq!(artifact["kind"], "checkpoint");
        assert_eq!(artifact["role"], "produced");
        assert_eq!(artifact["name"], "gpt-nano", "versions group by arch");
        assert_eq!(
            artifact["uri"],
            format!("{PUBLIC_URL}{API_PREFIX}/checkpoint/{}/500/{CHECKPOINT_MODEL_FILE}", run_id.0),
            "our stable resolver url survives R2 lifecycle expiry"
        );
    }

    #[tokio::test]
    async fn a_refused_artifact_registration_stays_pending() {
        let server = FakeHttp::start(|req, _| match req.path() {
            p if p.ends_with("/artifacts") => Reply::new(400, "bad uri scheme"),
            _ => Reply::ok(format!(r#"{{"id":"{EXT_RUN}"}}"#)),
        });
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;

        exp.report_checkpoint(run_id, 500, "s3://bucket/k".into(), checkpoint_meta()).await;
        let rows = pending(&store).await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "checkpoint");
        assert_eq!(rows[0].attempts, 1);
    }

    #[tokio::test]
    async fn a_checkpoint_with_no_acceptable_uri_is_reported_as_undeliverable() {
        // No http(s) public base and a bare file:// recorded uri: there is
        // nothing the experiments-server would accept.
        let server = accepting_server();
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), "file:///var/artifacts", None);
        exp.report_created(run_id.clone(), spec, None).await;

        exp.report_checkpoint(run_id, 500, "file:///var/artifacts/k".into(), checkpoint_meta())
            .await;
        let rows = pending(&store).await;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "checkpoint", "undeliverable, but never silently dropped");
    }

    #[tokio::test]
    async fn a_recorded_s3_uri_is_used_when_we_have_no_public_http_base() {
        let server = accepting_server();
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(&server.origin, store.clone(), "file:///var/artifacts", None);
        exp.report_created(run_id.clone(), spec, None).await;

        exp.report_checkpoint(run_id, 500, "s3://bucket/k".into(), checkpoint_meta()).await;
        let artifact = body_of_request(&server, "POST", &format!("/v1/runs/{EXT_RUN}/artifacts"));
        assert_eq!(artifact["uri"], "s3://bucket/k");
    }

    #[tokio::test]
    async fn a_checkpoint_without_an_arch_falls_back_to_the_code_unit_then_the_run() {
        let server = accepting_server();
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;

        let by_code: CheckpointMeta = serde_json::from_value(json!({
            "run_id": "EXEC-1",
            "step": 1,
            "code": { "name": "tok-v12", "sha": "abc123" },
        }))
        .expect("valid CheckpointMeta");
        exp.report_checkpoint(run_id.clone(), 1, "s3://b/k".into(), by_code).await;
        assert_eq!(
            body_of_request(&server, "POST", &format!("/v1/runs/{EXT_RUN}/artifacts"))["name"],
            "tok-v12"
        );

        let bare: CheckpointMeta =
            serde_json::from_value(json!({ "run_id": "EXEC-1", "step": 2 })).expect("valid meta");
        exp.report_checkpoint(run_id.clone(), 2, "s3://b/k".into(), bare).await;
        let names: Vec<String> = server
            .requests()
            .iter()
            .filter(|r| r.path().ends_with("/artifacts"))
            .map(|r| r.json()["name"].as_str().unwrap_or_default().to_owned())
            .collect();
        assert_eq!(names.last().map(String::as_str), Some(run_id.0.as_str()));
    }

    #[tokio::test]
    async fn a_refused_result_stays_pending() {
        let server = FakeHttp::start(|req, _| match req.path() {
            p if p.ends_with("/results") => Reply::new(422, "no such metric"),
            _ => Reply::ok(format!(r#"{{"id":"{EXT_RUN}"}}"#)),
        });
        let store = store().await;
        let (exp, run_id) = mirrored(&server, &store).await;
        store
            .append_metrics(&run_id, 1, &BTreeMap::from([("loss".to_owned(), 0.5)]))
            .await
            .expect("append metrics");

        exp.report_state(run_id, RunState::Completed).await;
        let rows = pending(&store).await;
        assert!(rows.iter().any(|r| r.kind == "result"), "the result is retried, not lost");
    }

    // -- the outbox ----------------------------------------------------------

    #[tokio::test]
    async fn a_sweep_delivers_what_the_first_attempt_could_not() {
        // Down when the run was created, up by the time the sweep runs — the
        // durable row, not the client, is what makes the observation survive.
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        Experiments::at(REFUSED_ORIGIN, store.clone(), PUBLIC_URL, None)
            .report_created(run_id.clone(), spec, None)
            .await;
        assert_eq!(pending(&store).await.len(), 1);

        let server = accepting_server();
        let up = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);
        // The row's own backoff has not elapsed, so sweep from a later "now".
        let due = store
            .due_outbox_events(now_secs() + 120.0, OUTBOX_SWEEP_BATCH)
            .await
            .expect("due");
        let event: OutboxEvent = serde_json::from_str(&due[0].payload).expect("payload");
        assert!(up.attempt(due[0].id, &run_id, event, due[0].attempts).await);
        assert_eq!(
            store.experiments_run_id(&run_id).await.unwrap().as_deref(),
            Some(EXT_RUN)
        );
        assert!(pending(&store).await.is_empty(), "delivered rows are marked done");
    }

    #[tokio::test]
    async fn a_sweep_pass_drops_a_row_it_could_never_replay() {
        let server = accepting_server();
        let store = store().await;
        let (run_id, _spec) = queued_run(&store, None).await;
        store
            .enqueue_outbox_event(&run_id, "created", "{not json", now_secs() - 1.0)
            .await
            .expect("enqueue corrupt row");
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);

        assert_eq!(exp.sweep_outbox_once().await.expect("sweep"), 0);
        assert!(
            pending(&store).await.is_empty(),
            "a payload that can never parse is dropped, not retried forever"
        );
    }

    #[tokio::test]
    async fn a_sweep_pass_delivers_every_due_row_and_reports_how_many() {
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        Experiments::at(REFUSED_ORIGIN, store.clone(), PUBLIC_URL, None)
            .report_created(run_id.clone(), spec, None)
            .await;

        let server = accepting_server();
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);
        // Nothing is due yet (the failed row is backed off), then it is.
        assert_eq!(exp.sweep_outbox_once().await.expect("early sweep"), 0);
        store
            .mark_outbox_event_failed(pending(&store).await[0].id, "retry now", now_secs() - 1.0)
            .await
            .expect("make it due");
        assert_eq!(exp.sweep_outbox_once().await.expect("sweep"), 1);
        assert!(pending(&store).await.is_empty());
    }

    #[tokio::test]
    async fn the_outbox_loop_keeps_retrying_until_the_event_lands() {
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        Experiments::at(REFUSED_ORIGIN, store.clone(), PUBLIC_URL, None)
            .report_created(run_id.clone(), spec, None)
            .await;
        // Make the row due immediately so the first tick picks it up.
        store
            .mark_outbox_event_failed(pending(&store).await[0].id, "retry now", now_secs() - 1.0)
            .await
            .expect("make it due");

        let server = accepting_server();
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, None);
        let loop_task = tokio::spawn(exp.run_outbox_loop(Duration::from_millis(10)));
        let mut mirrored = false;
        for _ in 0..100 {
            if store.experiments_run_id(&run_id).await.unwrap().is_some() {
                mirrored = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        loop_task.abort();
        assert!(mirrored, "the loop must deliver the pending create on its own");
    }

    #[tokio::test]
    async fn a_failed_attempt_records_the_error_and_backs_off() {
        let store = store().await;
        let (run_id, spec) = queued_run(&store, None).await;
        let exp = Experiments::at(REFUSED_ORIGIN, store.clone(), PUBLIC_URL, None);
        exp.report_created(run_id.clone(), spec, None).await;

        assert_eq!(pending(&store).await[0].attempts, 1);
        assert!(
            store.due_outbox_events(now_secs(), 50).await.unwrap().is_empty(),
            "a failed event backs off rather than retrying in a hot loop"
        );
        assert!(
            !store
                .due_outbox_events(now_secs() + OUTBOX_BASE_BACKOFF_SECS as f64 + 1.0, 50)
                .await
                .unwrap()
                .is_empty(),
            "and comes due again one backoff later"
        );
    }

    // -- fetch_run -----------------------------------------------------------

    #[tokio::test]
    async fn fetch_run_returns_the_snapshot_submit_from_experiment_builds_on() {
        let server = FakeHttp::start(|_, _| {
            Reply::ok(
                json!({
                    "status": "queued",
                    "config": { "entrypoint": "train", "code": { "name": "n", "sha": "s" } },
                    "budget_seconds": 60,
                    "unknown_field": "tolerated",
                })
                .to_string(),
            )
        });
        let exp = Experiments::at(&server.origin, store().await, PUBLIC_URL, None);

        let snapshot = exp.fetch_run(EXT_RUN, None).await.expect("fetch");
        assert_eq!(snapshot.status, "queued");
        assert_eq!(snapshot.budget_seconds, Some(60));
        assert_eq!(server.requests()[0].path(), format!("/v1/runs/{EXT_RUN}"));
    }

    #[tokio::test]
    async fn fetch_run_surfaces_a_miss_with_the_servers_body() {
        let server = FakeHttp::start(|_, _| Reply::new(404, "no such run"));
        let exp = Experiments::at(&server.origin, store().await, PUBLIC_URL, None);
        let error = exp.fetch_run("RUN-nope", None).await.unwrap_err();
        assert!(error.to_string().contains("fetch run"), "unexpected error: {error}");
        assert!(error.to_string().contains("no such run"), "unexpected error: {error}");
    }

    // -- per-user keys -------------------------------------------------------

    const ENCRYPTION_KEY: [u8; 32] = [7u8; 32];
    const EMAIL: &str = "researcher@example.com";

    async fn user_with_linked_key(store: &Arc<dyn Store>, personal: &str) {
        store
            .upsert_user(EMAIL, chuk_train_proto::DEFAULT_TEAM_ID, Role::Write)
            .await
            .expect("upsert user");
        let encrypted = crate::crypto::encrypt(&ENCRYPTION_KEY, personal);
        store
            .set_user_experiments_key(EMAIL, Some(&encrypted))
            .await
            .expect("link key");
    }

    #[tokio::test]
    async fn a_users_own_linked_key_is_preferred_over_the_shared_default() {
        let server = accepting_server();
        let store = store().await;
        user_with_linked_key(&store, "personal-key").await;
        let (run_id, spec) = queued_run(&store, Some(EMAIL)).await;
        let exp = Experiments::at(&server.origin, store.clone(), PUBLIC_URL, Some(ENCRYPTION_KEY));

        exp.report_created(run_id, spec, None).await;

        // The ensure always uses the shared default; the run's own reports use
        // the submitting user's key.
        let created = server
            .requests()
            .into_iter()
            .find(|r| r.path().ends_with("/runs"))
            .expect("the create");
        assert_eq!(created.header("authorization"), "Bearer personal-key");
        assert_eq!(server.requests()[0].header("authorization"), format!("Bearer {SHARED_KEY}"));
    }

    #[tokio::test]
    async fn key_resolution_falls_back_to_the_shared_default_at_every_missing_link() {
        let store = store().await;
        let exp = Experiments::at("http://unused", store.clone(), PUBLIC_URL, Some(ENCRYPTION_KEY));

        assert_eq!(exp.bearer_for_email(None).await, SHARED_KEY, "no email");
        assert_eq!(
            exp.bearer_for_email(Some("stranger@example.com")).await,
            SHARED_KEY,
            "no linked key"
        );
        assert_eq!(
            exp.bearer_for(&RunId("EXEC-nope".into())).await,
            SHARED_KEY,
            "no such run"
        );

        // A linked key that doesn't decrypt with our key (rotated, corrupt)
        // must not fail the report — it falls back too.
        store
            .upsert_user(EMAIL, chuk_train_proto::DEFAULT_TEAM_ID, Role::Write)
            .await
            .expect("upsert user");
        store
            .set_user_experiments_key(EMAIL, Some("not-even-base64"))
            .await
            .expect("link key");
        assert_eq!(exp.bearer_for_email(Some(EMAIL)).await, SHARED_KEY, "undecryptable");
    }

    #[tokio::test]
    async fn the_shared_default_is_used_wholesale_when_per_user_keys_are_off() {
        let store = store().await;
        user_with_linked_key(&store, "personal-key").await;
        let (run_id, _spec) = queued_run(&store, Some(EMAIL)).await;
        let exp = Experiments::at("http://unused", store.clone(), PUBLIC_URL, None);

        assert_eq!(exp.bearer_for(&run_id).await, SHARED_KEY);
        assert_eq!(exp.bearer_for_email(Some(EMAIL)).await, SHARED_KEY);
    }

    // -- from_env ------------------------------------------------------------

    /// Touches the process-global experiments env vars, so it is one test
    /// rather than several that could interleave, and it takes the shared lock
    /// against `hub::tests`'s mirror test. The store is built first so the
    /// guard is never held across an await.
    #[tokio::test]
    async fn from_env_is_off_unless_both_the_url_and_a_key_are_set() {
        let store = store().await;
        let vars = [
            env::EXPERIMENTS_URL,
            env::EXPERIMENTS_API_KEY,
            env::EXPERIMENTS_PROGRAMME,
            env::EXPERIMENTS_EXPERIMENT,
        ];
        let _guard = lock_experiments_env();
        let restore: Vec<(&str, Option<String>)> =
            vars.iter().map(|v| (*v, std::env::var(v).ok())).collect();
        for var in vars {
            std::env::remove_var(var);
        }

        assert!(Experiments::from_env(store.clone(), PUBLIC_URL).is_none(), "nothing set");
        std::env::set_var(env::EXPERIMENTS_URL, "https://exp.example.com/");
        assert!(
            Experiments::from_env(store.clone(), PUBLIC_URL).is_none(),
            "a url alone is not enough"
        );
        std::env::set_var(env::EXPERIMENTS_API_KEY, "write-key");
        let exp = Experiments::from_env(store.clone(), &format!("{PUBLIC_URL}/")).expect("configured");
        assert_eq!(exp.base, "https://exp.example.com", "trailing slash trimmed");
        assert_eq!(exp.public_url, PUBLIC_URL);
        assert_eq!(exp.programme, DEFAULT_EXPERIMENTS_PROGRAMME);
        assert_eq!(exp.experiment, DEFAULT_EXPERIMENTS_EXPERIMENT);

        // Explicit programme/experiment slugs override the defaults; blank ones
        // fall back rather than naming an empty slug.
        std::env::set_var(env::EXPERIMENTS_PROGRAMME, "  ");
        std::env::set_var(env::EXPERIMENTS_EXPERIMENT, " my-experiment ");
        let exp = Experiments::from_env(store, PUBLIC_URL).expect("configured");
        assert_eq!(exp.programme, DEFAULT_EXPERIMENTS_PROGRAMME);
        assert_eq!(exp.experiment, "my-experiment");

        for (var, value) in restore {
            match value {
                Some(value) => std::env::set_var(var, value),
                None => std::env::remove_var(var),
            }
        }
    }

    #[test]
    fn every_outbox_event_carries_its_own_label() {
        let meta = Box::new(checkpoint_meta());
        assert_eq!(
            OutboxEvent::Created { spec: RunSpec::Train(Box::new(train_spec())), experiment_ref: None }
                .kind_label(),
            "created"
        );
        assert_eq!(OutboxEvent::State { state: RunState::Running }.kind_label(), "state");
        assert_eq!(
            OutboxEvent::Checkpoint { step: 1, uri: "s3://b/k".into(), meta }.kind_label(),
            "checkpoint"
        );
        assert_eq!(OutboxEvent::Result { name: "loss".into(), value: 1.0 }.kind_label(), "result");
    }
}

/// Live proof against a real chuk-experiments-server — see
/// `experiments/tests.rs`. Kept in a `tests.rs` sibling because those checks
/// are `#[ignore]`d and can never run in CI (they need a real server and a real
/// write key): the coverage gate excludes `tests.rs` files, so
/// permanently-unrunnable lines don't count against this module's coverage.
#[cfg(test)]
#[path = "experiments/tests.rs"]
mod live;
