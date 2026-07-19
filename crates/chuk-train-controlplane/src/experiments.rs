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
}

/// Live proof that the outbox actually recovers from a real failure, not just
/// a simulated one. Ignored by default; run in isolation (it mutates the
/// process-wide `CHUK_EXPERIMENTS_URL` env var) against a real local
/// chuk-experiments-server:
///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
///   cargo test -p chuk-train-controlplane experiments::live::outbox_recovers_after_experiments_server_was_unreachable -- --ignored --nocapture
#[cfg(test)]
mod live {
    use super::*;
    use crate::store::SqliteStore;

    #[ignore]
    #[tokio::test]
    async fn outbox_recovers_after_experiments_server_was_unreachable() {
        let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
        let key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
        if base.is_empty() || key.is_empty() {
            eprintln!(
                "skip: need {} + {} pointed at a real local chuk-experiments-server",
                env::EXPERIMENTS_URL,
                env::EXPERIMENTS_API_KEY
            );
            return;
        }

        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let train: TrainSpec = serde_json::from_value(json!({
            "code": { "name": "outbox-smoke-test", "sha": "0000000000000000000000000000000000000000" },
            "entrypoint": "true",
        }))
        .expect("valid TrainSpec");
        let spec = RunSpec::Train(Box::new(train));
        // A real `runs` row, exactly as Hub::submit creates before mirroring —
        // set_experiments_run_id later needs a matching row to update.
        let run_id = store
            .create_run("outbox-smoke-test", &spec, None, None)
            .await
            .expect("create_run");

        // Phase 1: nothing listens here -- the create must fail and land
        // durably in the outbox rather than being silently dropped.
        std::env::set_var(env::EXPERIMENTS_URL, "http://127.0.0.1:1");
        let down = Experiments::from_env(store.clone(), "http://localhost:9").expect("down client");
        down.report_created(run_id.clone(), spec, None).await;

        assert!(
            store.experiments_run_id(&run_id).await.unwrap().is_none(),
            "must not be marked mirrored -- the create never actually landed"
        );
        let pending = store
            .due_outbox_events(now_secs() + 3_700.0, 10) // past the max backoff, just to introspect the row
            .await
            .expect("due");
        assert_eq!(pending.len(), 1, "the failed create must be sitting in the outbox, not lost");
        assert_eq!(pending[0].attempts, 1);

        // Phase 2: replay it against a client pointed at the real, healthy
        // server -- proves the *outbox row*, not the client, is what makes
        // this durable (a fresh client + the same store recovers it).
        std::env::set_var(env::EXPERIMENTS_URL, &base);
        let up = Experiments::from_env(store.clone(), "http://localhost:9").expect("up client");
        let event: OutboxEvent = serde_json::from_str(&pending[0].payload).expect("payload");
        let delivered = up.attempt(pending[0].id, &run_id, event, pending[0].attempts).await;
        assert!(delivered, "retry against the real, now-healthy server must succeed");
        assert!(
            store.experiments_run_id(&run_id).await.unwrap().is_some(),
            "now genuinely mirrored on the real server"
        );

        std::env::remove_var(env::EXPERIMENTS_URL); // don't leak into other tests
    }

    /// Live proof that `bearer_for` really *prefers* a linked personal key —
    /// not just that each half works in isolation. The shared default stays
    /// valid (so `ensure()`, which always uses it, still succeeds); the
    /// user's *linked* key is deliberately garbage. If resolution ever fell
    /// back to the working shared default instead of genuinely preferring the
    /// (here broken) personal key, this mirror call would succeed — it must
    /// not.
    ///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
    ///   cargo test -p chuk-train-controlplane experiments::live::bearer_for_prefers_the_submitting_users_own_linked_key -- --ignored --nocapture
    #[ignore]
    #[tokio::test]
    async fn bearer_for_prefers_the_submitting_users_own_linked_key() {
        use base64::engine::general_purpose::STANDARD;
        use base64::Engine;
        use chuk_train_proto::Role;

        let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
        let real_key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
        if base.is_empty() || real_key.is_empty() {
            eprintln!(
                "skip: need {} + {} pointed at a real local chuk-experiments-server",
                env::EXPERIMENTS_URL,
                env::EXPERIMENTS_API_KEY
            );
            return;
        }
        let _ = real_key; // the shared default stays valid throughout this test

        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let email = "personal-key-test@example.com";
        store
            .upsert_user(email, "default", Role::Write)
            .await
            .expect("upsert user");

        let encryption_key = [3u8; 32];
        let encrypted = crate::crypto::encrypt(&encryption_key, "deliberately-invalid-personal-key");
        store
            .set_user_experiments_key(email, Some(&encrypted))
            .await
            .expect("link key");
        std::env::set_var(env::EXPERIMENTS_KEY_ENCRYPTION_KEY, STANDARD.encode(encryption_key));

        let exp = Experiments::from_env(store.clone(), "http://localhost:9").expect("client");
        let train: TrainSpec = serde_json::from_value(json!({
            "code": { "name": "bearer-for-test", "sha": "1111111111111111111111111111111111111111" },
            "entrypoint": "true",
        }))
        .expect("valid TrainSpec");
        let spec = RunSpec::Train(Box::new(train));
        let run_id = store
            .create_run("bearer-for-test", &spec, None, Some(email))
            .await
            .expect("create_run");

        exp.report_created(run_id.clone(), spec, None).await;

        assert!(
            store.experiments_run_id(&run_id).await.unwrap().is_none(),
            "must have failed using the user's own (deliberately invalid) linked key, \
             not silently succeeded by falling back to the still-valid shared default"
        );
        let pending = store
            .due_outbox_events(now_secs() + 3_700.0, 10)
            .await
            .expect("due");
        assert_eq!(pending.len(), 1, "the failed create must be sitting in the outbox");

        std::env::remove_var(env::EXPERIMENTS_KEY_ENCRYPTION_KEY);
    }

    /// Live proof of the actual feature: seed a queued run directly on a real
    /// chuk-experiments-server (exactly as `enqueue_run` would), submit it via
    /// `Hub::submit_from_experiment`, and confirm the harness execution
    /// *attaches* to that seeded run (via the store's local
    /// `experiments_run_id` mapping, the same thing `try_attach` sets) rather
    /// than a second, unrelated run being minted.
    ///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
    ///   cargo test -p chuk-train-controlplane experiments::live::submit_from_experiment_attaches_to_an_existing_queued_run -- --ignored --nocapture
    #[ignore]
    #[tokio::test]
    async fn submit_from_experiment_attaches_to_an_existing_queued_run() {
        let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
        let key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
        if base.is_empty() || key.is_empty() {
            eprintln!(
                "skip: need {} + {} pointed at a real local chuk-experiments-server",
                env::EXPERIMENTS_URL,
                env::EXPERIMENTS_API_KEY
            );
            return;
        }

        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let exp = Experiments::from_env(store.clone(), "http://localhost:9").expect("client");
        exp.ensure().await.expect("ensure default experiment");

        // Seed a queued run directly via REST, exactly as an external caller
        // (e.g. chuk-experiments-server's own `enqueue_run`) would.
        let http = reqwest::Client::new();
        let resp = http
            .post(format!("{base}/v1/experiments/{DEFAULT_EXPERIMENTS_EXPERIMENT}/runs"))
            .bearer_auth(&key)
            .json(&json!({
                "slug": format!("submit-from-experiment-smoke-{}", now_secs() as i64),
                "config": {
                    "entrypoint": "true",
                    "code": { "name": "submit-from-experiment-smoke", "sha": "0000000000000000000000000000000000000000" },
                },
                "status": "queued",
            }))
            .send()
            .await
            .expect("seed run");
        assert!(resp.status().is_success(), "seed run: {}", resp.status());
        let created: Value = resp.json().await.expect("parse seeded run");
        let ext_id = created["id"].as_str().expect("id").to_owned();

        let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
            Arc::new(crate::artifacts::FsArtifactStore::new(std::env::temp_dir()));
        let hub = crate::hub::Hub::new(store.clone(), artifacts, Some(exp));

        let run_id = hub
            .submit_from_experiment(&ext_id, None, None)
            .await
            .expect("submit_from_experiment");

        // mirror_created spawns the attach call rather than awaiting it
        // inline, so poll the store's local mapping instead of a blind sleep.
        let mut attached = None;
        for _ in 0..50 {
            if let Ok(Some(got)) = store.experiments_run_id(&run_id).await {
                attached = Some(got);
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(
            attached.as_deref(),
            Some(ext_id.as_str()),
            "must attach to the seeded run, not mint or point at a different one"
        );
    }
}
