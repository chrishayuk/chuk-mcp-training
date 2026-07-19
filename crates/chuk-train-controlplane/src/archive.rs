//! Archive tier (spec §11.5). When a run completes, its **final** checkpoint +
//! logs + metrics are tiered to Google Drive as the durable copy, the final is
//! promoted to `ckpt-final/` on R2 (a 30-day warm cache that R2 lifecycle later
//! expires), and the checkpoint's location is recorded as `drive`.
//!
//! A single periodic loop is both roles: it archives newly completed runs
//! promptly *and* backstops anything a prior pass or a failed attempt missed —
//! every step is idempotent (a run already on Drive is skipped). `archive_run`
//! is also called directly by the MCP `archive_run`/`archive_runs` tools.
//! Deletion of the expired R2 copies is R2's lifecycle job, not ours.
//! See the retention policy: Drive = canonical, R2 = hot cache with TTLs.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chuk_train_proto::{
    keys, CheckpointLocation, RunId, RunState, CHECKPOINT_DIR_PREFIX, CHECKPOINT_META_FILE,
    CHECKPOINT_MODEL_FILE,
};
use tracing::{info, warn};

use crate::artifacts::ArtifactStore;
use crate::drive::{DriveClient, ARCHIVE_ROOT_FOLDER};
use crate::store::Store;

/// Checkpoint files we archive. `optim.pt` is resume-only + large; a *final*
/// checkpoint doesn't need it and lazarus excludes it anyway (spec §10).
const ARCHIVE_FILES: [&str; 2] = [CHECKPOINT_MODEL_FILE, CHECKPOINT_META_FILE];
const LOGS_NAME: &str = "logs.txt";
const METRICS_NAME: &str = "metrics.json";
const OCTET: &str = "application/octet-stream";
const TEXT: &str = "text/plain; charset=utf-8";
const JSON: &str = "application/json";
/// Effectively "all lines" for our run sizes.
const LOGS_TAIL: u32 = 100_000;
/// How many recent runs a sweep pass considers.
const SWEEP_LIMIT: u32 = 200;

/// What one `archive_run` did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Archived { step: u64, files: usize },
    AlreadyArchived,
    NoCheckpoint,
}

pub struct Archiver {
    store: Arc<dyn Store>,
    artifacts: Arc<dyn ArtifactStore>,
    drive: Arc<DriveClient>,
}

impl Archiver {
    pub fn new(
        store: Arc<dyn Store>,
        artifacts: Arc<dyn ArtifactStore>,
        drive: Arc<DriveClient>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            artifacts,
            drive,
        })
    }

    /// Tier a run's final checkpoint + logs/metrics to Drive. Idempotent — a run
    /// already archived is skipped unless `force`.
    pub async fn archive_run(&self, run_id: &RunId, force: bool) -> Result<Outcome> {
        let Some(final_ckpt) = self.store.latest_checkpoint(run_id).await? else {
            return Ok(Outcome::NoCheckpoint);
        };
        let already =
            final_ckpt.location == CheckpointLocation::Drive || final_ckpt.archived_at.is_some();
        if already && !force {
            return Ok(Outcome::AlreadyArchived);
        }
        let step = final_ckpt.step;
        // Drive layout mirrors the run: chuk-train/runs/<run_id>/… — sortable and
        // browsable (execution ids are EXEC-YYYYMMDD-…).
        let run_folder = format!("{ARCHIVE_ROOT_FOLDER}/runs/{}", run_id.0);
        let ckpt_folder = format!("{run_folder}/ckpt/{CHECKPOINT_DIR_PREFIX}{step}");
        let mut drive_ids: BTreeMap<String, String> = BTreeMap::new();

        // 1. checkpoint files: read the hot copy, promote to ckpt-final (the warm
        //    R2 cache), and upload to Drive (the canonical copy).
        for file in ARCHIVE_FILES {
            let hot_key = keys::checkpoint_file(&run_id.0, step, file);
            let Ok(bytes) = self.artifacts.get(&hot_key).await else {
                continue; // optional/expired file — skip it
            };
            let final_key = keys::checkpoint_final_file(&run_id.0, step, file);
            if let Err(e) = self.artifacts.copy(&hot_key, &final_key).await {
                warn!(run = %run_id.0, %file, error = %e, "promote hot→final failed (continuing)");
            }
            let id = self
                .drive
                .upload_to_path(&ckpt_folder, file, Some(OCTET), &bytes)
                .await
                .with_context(|| format!("drive upload {file}"))?;
            drive_ids.insert(file.to_owned(), id);
        }
        self.store
            .set_checkpoint_location(run_id, step, CheckpointLocation::R2Final)
            .await?;

        // 2. logs → logs.txt
        let logs = self.store.tail_logs(run_id, LOGS_TAIL).await.unwrap_or_default();
        if !logs.is_empty() {
            let body = logs.join("\n").into_bytes();
            let id = self
                .drive
                .upload_to_path(&run_folder, LOGS_NAME, Some(TEXT), &body)
                .await?;
            drive_ids.insert(LOGS_NAME.to_owned(), id);
        }

        // 3. metrics → metrics.json (the whole series)
        if let Ok(series) = self.store.metric_series(run_id, None, 0, 0).await {
            if !series.series.is_empty() {
                let body = serde_json::to_vec(&series)?;
                let id = self
                    .drive
                    .upload_to_path(&run_folder, METRICS_NAME, Some(JSON), &body)
                    .await?;
                drive_ids.insert(METRICS_NAME.to_owned(), id);
            }
        }

        // 4. record the archived location on the final checkpoint.
        let files = drive_ids.len();
        self.store
            .mark_checkpoint_archived(run_id, step, &drive_ids, now())
            .await?;
        info!(run = %run_id.0, step, files, "archived run to drive");
        Ok(Outcome::Archived { step, files })
    }

    /// One sweep pass: archive every completed run whose final isn't yet on
    /// Drive. Returns how many it archived this pass. Idempotent.
    pub async fn sweep_once(&self) -> Result<usize> {
        let runs = self.store.runs(SWEEP_LIMIT).await?;
        let mut archived = 0;
        for run in runs
            .into_iter()
            .filter(|r| matches!(r.state, RunState::Completed))
        {
            match self.archive_run(&run.id, false).await {
                Ok(Outcome::Archived { .. }) => archived += 1,
                Ok(_) => {}
                Err(e) => {
                    warn!(run = %run.id.0, error = %e, "archive failed (retries next sweep)")
                }
            }
        }
        Ok(archived)
    }

    /// The archive/backstop loop — runs for the life of the process.
    pub async fn run_loop(self: Arc<Self>, interval: Duration) {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            match self.sweep_once().await {
                Ok(n) if n > 0 => info!(archived = n, "archive sweep"),
                Ok(_) => {}
                Err(e) => warn!(error = %e, "archive sweep failed"),
            }
        }
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Live end-to-end archive of one real completed run. Ignored by default; run
/// against the deployed backends with `.env` sourced + these overrides:
///   CHUK_TRAIN_ARTIFACTS=r2://<bucket> ARCHIVE_TEST_RUN=<run_id> \
///   cargo test -p chuk-train-controlplane archive::live::archive_completed_run -- --ignored --nocapture
#[cfg(test)]
mod live {
    use super::*;
    use crate::artifacts::open_artifact_store;
    use crate::store::open_store;

    #[ignore]
    #[tokio::test]
    async fn archive_completed_run() {
        let store_url = std::env::var("CHUK_TRAIN_STORE").unwrap_or_default();
        let art_spec = std::env::var("CHUK_TRAIN_ARTIFACTS").unwrap_or_default();
        let run = std::env::var("ARCHIVE_TEST_RUN").unwrap_or_default();
        let Some(drive) = DriveClient::from_env().expect("drive").map(Arc::new) else {
            eprintln!("skip: no drive creds");
            return;
        };
        if !store_url.starts_with("postgres") || !art_spec.starts_with("r2") || run.is_empty() {
            eprintln!("skip: need postgres CHUK_TRAIN_STORE, r2 CHUK_TRAIN_ARTIFACTS, ARCHIVE_TEST_RUN");
            return;
        }
        let store: Arc<dyn Store> = Arc::from(open_store(&store_url).await.expect("store"));
        let artifacts: Arc<dyn ArtifactStore> =
            Arc::from(open_artifact_store(&art_spec).expect("artifacts"));
        let archiver = Archiver::new(store.clone(), artifacts, drive);

        let rid = RunId(run);
        let outcome = archiver.archive_run(&rid, true).await.expect("archive");
        eprintln!("outcome: {outcome:?}");
        assert!(matches!(outcome, Outcome::Archived { .. }));
        let ck = store.latest_checkpoint(&rid).await.unwrap().expect("final ckpt");
        assert_eq!(ck.location, CheckpointLocation::Drive);
        assert!(ck.archived_at.is_some());
        eprintln!("final step {} archived to drive at {:?}", ck.step, ck.archived_at);
    }
}
