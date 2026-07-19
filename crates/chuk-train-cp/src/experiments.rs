//! chuk-experiments-server reporting mirror (spec §11.6).
//!
//! Optional and gated: constructed only when `CHUK_EXPERIMENTS_URL` +
//! `CHUK_EXPERIMENTS_API_KEY` are set, and **every method is best-effort** — a
//! slow or down experiments-server logs a warning and never blocks or fails a
//! run. The harness's own store stays the source of truth (design principle 10).
//!
//! The experiments-server is the research system-of-record; it *always mints its
//! own* `RUN-…` id, so we send our run id as the run `slug` + `harness_session_id`
//! and persist the id it returns on our run row (`experiments_run_id`) to address
//! later lifecycle/artifact reports. Their id-space and ours are parallel and
//! independent; `harness_session_id` links them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chuk_train_proto::{
    env, CheckpointMeta, RunId, RunSpec, RunState, TrainSpec, API_PREFIX, CHECKPOINT_MODEL_FILE,
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
        }))
    }

    // ---- best-effort entrypoints (the hub spawns these) -------------------

    /// Mirror a freshly-submitted run. Only **train** runs are reported; shell
    /// probes are skipped (and their later transitions no-op, having no id).
    pub async fn report_created(&self, run_id: RunId, spec: RunSpec) {
        let RunSpec::Train(train) = spec else { return };
        if let Err(e) = self.try_create(&run_id, &train).await {
            warn!(run = %run_id.0, error = %e, "experiments: create run failed (mirror only)");
        }
    }

    /// Mirror a run state transition (running / terminal). No-op for a run that
    /// was never mirrored (shell run, or the create is still in flight).
    pub async fn report_state(&self, run_id: RunId, state: RunState) {
        if let Err(e) = self.try_state(&run_id, state).await {
            warn!(run = %run_id.0, ?state, error = %e, "experiments: status report failed");
        }
    }

    /// Register an uploaded checkpoint as a `checkpoint` artifact.
    pub async fn report_checkpoint(&self, run_id: RunId, step: u64, uri: String, meta: CheckpointMeta) {
        if let Err(e) = self.try_checkpoint(&run_id, step, &uri, &meta).await {
            warn!(run = %run_id.0, step, error = %e, "experiments: checkpoint artifact failed");
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

    // ---- the actual reports ------------------------------------------------

    async fn try_create(&self, run_id: &RunId, train: &TrainSpec) -> Result<()> {
        self.ensure().await?;
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
            .bearer_auth(&self.key)
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
        // Link back: our id as the correlation field, plus any W&B out-link.
        let mut patch = json!({ "harness_session_id": run_id.0 });
        if let Some(url) = train
            .links
            .iter()
            .find(|l| l.kind.eq_ignore_ascii_case(LINK_KIND_WANDB))
            .map(|l| l.url.clone())
        {
            patch["wandb_url"] = json!(url);
        }
        // Best-effort; the primary link is `slug` = our id even if this PATCH fails.
        if let Err(e) = self.patch_run(&created.id, patch).await {
            warn!(run = %run_id.0, ext = %created.id, error = %e, "experiments: linkback patch failed");
        }
        self.store.set_experiments_run_id(run_id, &created.id).await?;
        info!(run = %run_id.0, ext = %created.id, "experiments: run mirrored");
        Ok(())
    }

    async fn try_state(&self, run_id: &RunId, state: RunState) -> Result<()> {
        let Some(ext) = self.store.experiments_run_id(run_id).await? else {
            return Ok(());
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
        self.patch_run(&ext, body).await?;
        // Final metrics → results, on success only. Extra, so failures are swallowed.
        if matches!(state, RunState::Completed) {
            if let Err(e) = self.report_final_metrics(run_id, &ext).await {
                warn!(run = %run_id.0, error = %e, "experiments: final metrics report failed");
            }
        }
        Ok(())
    }

    async fn try_checkpoint(
        &self,
        run_id: &RunId,
        step: u64,
        recorded_uri: &str,
        meta: &CheckpointMeta,
    ) -> Result<()> {
        let Some(ext) = self.store.experiments_run_id(run_id).await? else {
            return Ok(());
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
        let resp = self
            .http
            .post(format!("{}/v1/runs/{}/artifacts", self.base, ext))
            .bearer_auth(&self.key)
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

    async fn report_final_metrics(&self, run_id: &RunId, ext: &str) -> Result<()> {
        let series = self.store.metric_series(run_id, None, 0, 0).await?;
        for (name, points) in series.series {
            if let Some(last) = points.last() {
                // Best-effort per metric; a failed result never blocks the rest.
                let _ = self
                    .http
                    .post(format!("{}/v1/runs/{}/results", self.base, ext))
                    .bearer_auth(&self.key)
                    .json(&json!({ "name": name, "value": last.value }))
                    .send()
                    .await;
            }
        }
        Ok(())
    }

    async fn patch_run(&self, ext_id: &str, body: Value) -> Result<()> {
        let resp = self
            .http
            .patch(format!("{}/v1/runs/{}", self.base, ext_id))
            .bearer_auth(&self.key)
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
}
