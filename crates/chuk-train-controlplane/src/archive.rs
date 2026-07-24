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
use async_trait::async_trait;
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

/// The one Drive operation `archive_run` needs — factored out as a trait so
/// this module's actual responsibility (eligibility, idempotency, promotion,
/// and the per-file archive record) is unit-testable without live Google
/// OAuth credentials. [`DriveClient`]'s only public constructor,
/// [`DriveClient::from_env`], needs a real refresh token to build one at all
/// (and its fields are private outside `drive.rs`, so it can't be
/// hand-constructed either) — production wires the real client below; tests
/// wire an in-memory fake instead (see `tests::FakeDrive`).
#[async_trait]
pub trait DriveUploader: Send + Sync {
    async fn upload_to_path(
        &self,
        folder_path: &str,
        name: &str,
        mime: Option<&str>,
        bytes: &[u8],
    ) -> Result<String>;
}

#[async_trait]
impl DriveUploader for DriveClient {
    async fn upload_to_path(
        &self,
        folder_path: &str,
        name: &str,
        mime: Option<&str>,
        bytes: &[u8],
    ) -> Result<String> {
        // UFCS to the inherent method by name — `DriveUploader::upload_to_path`
        // would otherwise shadow it here and recurse.
        DriveClient::upload_to_path(self, folder_path, name, mime, bytes).await
    }
}

pub struct Archiver {
    store: Arc<dyn Store>,
    artifacts: Arc<dyn ArtifactStore>,
    drive: Arc<dyn DriveUploader>,
}

impl Archiver {
    /// `drive` takes anything that can do the one Drive upload this module
    /// needs — production passes a live `Arc<DriveClient>` (it coerces to
    /// `Arc<dyn DriveUploader>` automatically), tests pass a fake.
    pub fn new(
        store: Arc<dyn Store>,
        artifacts: Arc<dyn ArtifactStore>,
        drive: Arc<dyn DriveUploader>,
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
        let completed = crate::store::RunQuery {
            state: Some(RunState::Completed),
            ..Default::default()
        };
        let runs = self.store.runs(&completed, SWEEP_LIMIT).await?;
        let mut archived = 0;
        for run in runs {
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

/// Exercises the archive/backstop decision logic (eligibility, idempotency,
/// promotion, the per-file archive record, and the sweep/loop drivers) via
/// [`DriveUploader`] — an in-memory fake instead of a live [`DriveClient`].
///
/// What's deliberately *not* covered here, and why: the `warn!("promote
/// hot→final failed")` branch (only reachable if `ArtifactStore::copy`
/// fails after a successful `get`, which the filesystem backend used below
/// never does), and `run_loop`'s `Err(e)` arm (only reachable if
/// `Store::runs` itself fails, which a healthy in-memory SQLite never does).
/// Both would need a fault-injecting `Store`/`ArtifactStore` to reach, which
/// is out of scope for this file.
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use chuk_train_proto::{CheckpointMeta, RunSpec, ShellSpec};

    use super::*;
    use crate::artifacts::FsArtifactStore;
    use crate::store::SqliteStore;

    /// One recorded call to [`FakeDrive::upload_to_path`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Upload {
        folder: String,
        name: String,
    }

    /// In-memory stand-in for [`DriveClient`] (see this module's doc comment
    /// and [`DriveUploader`]'s). Records every call it accepts; optionally
    /// refuses all of them, to exercise `archive_run`'s error path and
    /// `sweep_once`'s continue-past-a-failure behaviour.
    #[derive(Default)]
    struct FakeDrive {
        uploads: Mutex<Vec<Upload>>,
        next_id: AtomicUsize,
        should_fail: bool,
    }

    impl FakeDrive {
        fn failing() -> Self {
            Self { should_fail: true, ..Default::default() }
        }

        fn uploads(&self) -> Vec<Upload> {
            self.uploads.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl DriveUploader for FakeDrive {
        async fn upload_to_path(
            &self,
            folder_path: &str,
            name: &str,
            _mime: Option<&str>,
            _bytes: &[u8],
        ) -> Result<String> {
            if self.should_fail {
                anyhow::bail!("fake drive refuses uploads (folder={folder_path}, name={name})");
            }
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            self.uploads.lock().unwrap().push(Upload {
                folder: folder_path.to_owned(),
                name: name.to_owned(),
            });
            Ok(format!("fake-drive-id-{id}"))
        }
    }

    fn shell_spec() -> RunSpec {
        RunSpec::Shell(ShellSpec { command: "true".into(), timeout_s: 60 })
    }

    async fn new_run(store: &Arc<dyn Store>, name: &str) -> RunId {
        store.create_run(name, &shell_spec(), None, None, None).await.expect("create_run")
    }

    /// Move a run to `Completed` — the state `sweep_once` filters on.
    async fn complete(store: &Arc<dyn Store>, run: &RunId) {
        store
            .transition(run, RunState::Completed, None, Some(0), serde_json::json!({}))
            .await
            .expect("transition to completed");
    }

    /// A fresh in-memory `Store` + a fresh scratch-dir `ArtifactStore`, wired
    /// into an `Archiver` backed by `drive`. Returns the store/artifacts
    /// handles alongside so tests can seed data and assert on the result —
    /// `Archiver`'s own fields are private.
    async fn test_env(drive: Arc<FakeDrive>) -> (Arc<Archiver>, Arc<dyn Store>, Arc<dyn ArtifactStore>) {
        let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
        let root = std::env::temp_dir().join(format!("chuk-archive-test-{}", uuid::Uuid::new_v4()));
        let artifacts: Arc<dyn ArtifactStore> = Arc::new(FsArtifactStore::new(root));
        let archiver = Archiver::new(store.clone(), artifacts.clone(), drive);
        (archiver, store, artifacts)
    }

    #[tokio::test]
    async fn archive_run_with_no_checkpoint_is_a_noop() {
        let drive = Arc::new(FakeDrive::default());
        let (archiver, store, _artifacts) = test_env(drive.clone()).await;
        let run = new_run(&store, "bare").await;

        let outcome = archiver.archive_run(&run, false).await.expect("archive_run");
        assert_eq!(outcome, Outcome::NoCheckpoint);
        assert!(drive.uploads().is_empty(), "no checkpoint means nothing to upload");
    }

    #[tokio::test]
    async fn archive_run_uploads_checkpoint_files_logs_and_metrics_then_marks_drive() {
        let drive = Arc::new(FakeDrive::default());
        let (archiver, store, artifacts) = test_env(drive.clone()).await;
        let run = new_run(&store, "full-archive").await;
        let step = 3u64;

        store
            .record_checkpoint(&run, step, "ckpt-hot/x/step_3", "hash", &CheckpointMeta::default())
            .await
            .expect("record_checkpoint");
        for file in ARCHIVE_FILES {
            let key = keys::checkpoint_file(&run.0, step, file);
            artifacts.put(&key, b"fake-checkpoint-bytes".to_vec()).await.expect("seed hot file");
        }
        store.append_log(&run, "line one").await.expect("append_log");
        store.append_log(&run, "line two").await.expect("append_log");
        let mut metrics = BTreeMap::new();
        metrics.insert("loss".to_owned(), 0.5);
        store.append_metrics(&run, step, &metrics).await.expect("append_metrics");

        let outcome = archiver.archive_run(&run, false).await.expect("archive_run");
        assert_eq!(outcome, Outcome::Archived { step, files: 4 });

        let ckpt = store.latest_checkpoint(&run).await.unwrap().expect("checkpoint");
        assert_eq!(ckpt.location, CheckpointLocation::Drive);
        assert!(ckpt.archived_at.is_some());

        let drive_ids = store.checkpoint_drive_ids(&run, step).await.unwrap().expect("drive ids");
        assert_eq!(drive_ids.len(), 4);
        assert!(drive_ids.contains_key(LOGS_NAME));
        assert!(drive_ids.contains_key(METRICS_NAME));
        for file in ARCHIVE_FILES {
            assert!(drive_ids.contains_key(file));
            let final_key = keys::checkpoint_final_file(&run.0, step, file);
            assert!(
                artifacts.exists(&final_key).await.unwrap(),
                "hot {file} must be promoted to the ckpt-final tier"
            );
        }

        let uploads = drive.uploads();
        assert_eq!(uploads.len(), 4);
        assert!(uploads.iter().any(|u| u.name == LOGS_NAME));
        assert!(uploads.iter().any(|u| u.name == METRICS_NAME));
    }

    #[tokio::test]
    async fn archive_run_is_idempotent_then_force_bypasses_and_reuploads() {
        let drive = Arc::new(FakeDrive::default());
        let (archiver, store, artifacts) = test_env(drive.clone()).await;
        let run = new_run(&store, "force-reupload").await;
        let step = 1u64;

        store
            .record_checkpoint(&run, step, "uri", "hash", &CheckpointMeta::default())
            .await
            .expect("record_checkpoint");
        for file in ARCHIVE_FILES {
            let key = keys::checkpoint_file(&run.0, step, file);
            artifacts.put(&key, b"bytes".to_vec()).await.expect("seed hot file");
        }

        let first = archiver.archive_run(&run, false).await.expect("first archive");
        assert_eq!(first, Outcome::Archived { step, files: 2 });

        // Already on Drive: a second non-forced call is a pure no-op.
        let second = archiver.archive_run(&run, false).await.expect("second archive");
        assert_eq!(second, Outcome::AlreadyArchived);
        assert_eq!(drive.uploads().len(), 2, "the no-op call must not touch drive again");

        // force=true bypasses the already-archived short-circuit and genuinely
        // re-runs the whole upload, proving idempotency is a default, not a hard rule.
        let third = archiver.archive_run(&run, true).await.expect("forced re-archive");
        assert_eq!(third, Outcome::Archived { step, files: 2 });
        assert_eq!(drive.uploads().len(), 4, "force must re-upload rather than skip");
    }

    #[tokio::test]
    async fn sweep_once_archives_only_completed_runs_not_yet_on_drive() {
        let drive = Arc::new(FakeDrive::default());
        let (archiver, store, _artifacts) = test_env(drive.clone()).await;

        // Eligible: completed, has a checkpoint, not yet archived.
        let eligible = new_run(&store, "eligible").await;
        complete(&store, &eligible).await;
        store
            .record_checkpoint(&eligible, 1, "uri", "hash", &CheckpointMeta::default())
            .await
            .unwrap();

        // Completed but no checkpoint at all: skipped, not an error.
        let bare = new_run(&store, "bare-completed").await;
        complete(&store, &bare).await;

        // Completed and already archived: skipped.
        let done = new_run(&store, "already-done").await;
        complete(&store, &done).await;
        store.record_checkpoint(&done, 2, "uri", "hash", &CheckpointMeta::default()).await.unwrap();
        store.mark_checkpoint_archived(&done, 2, &BTreeMap::new(), 1.0).await.unwrap();

        // Has a checkpoint but is still queued: the state filter must exclude
        // it from the sweep entirely, not just decline to archive it.
        let not_done = new_run(&store, "still-queued").await;
        store.record_checkpoint(&not_done, 5, "uri", "hash", &CheckpointMeta::default()).await.unwrap();

        let archived = archiver.sweep_once().await.expect("sweep_once");
        assert_eq!(archived, 1);

        assert_eq!(
            store.latest_checkpoint(&eligible).await.unwrap().unwrap().location,
            CheckpointLocation::Drive
        );
        assert_eq!(
            store.latest_checkpoint(&not_done).await.unwrap().unwrap().location,
            CheckpointLocation::R2Hot,
            "a non-completed run must never be swept"
        );
        // No hot bytes were seeded for `eligible`, so its archive still went
        // through drive for logs/metrics only if any were recorded — here
        // none were, so nothing needed a real upload at all.
        assert!(drive.uploads().is_empty());
    }

    #[tokio::test]
    async fn sweep_once_continues_past_a_failed_archive_and_returns_the_successful_count() {
        let drive = Arc::new(FakeDrive::failing());
        let (archiver, store, artifacts) = test_env(drive).await;

        // Needs a real upload, which the failing fake refuses — sweep_once
        // must log the failure and continue rather than abort the whole pass.
        let will_fail = new_run(&store, "will-fail").await;
        complete(&store, &will_fail).await;
        store
            .record_checkpoint(&will_fail, 1, "uri", "hash", &CheckpointMeta::default())
            .await
            .unwrap();
        let key = keys::checkpoint_file(&will_fail.0, 1, ARCHIVE_FILES[0]);
        artifacts.put(&key, b"bytes".to_vec()).await.unwrap();

        // Needs nothing from drive at all (no hot bytes, no logs, no metrics),
        // so it succeeds even though the fake refuses every upload.
        let trivially_ok = new_run(&store, "trivially-ok").await;
        complete(&store, &trivially_ok).await;
        store
            .record_checkpoint(&trivially_ok, 1, "uri", "hash", &CheckpointMeta::default())
            .await
            .unwrap();

        let archived = archiver.sweep_once().await.expect("sweep_once itself must not error");
        assert_eq!(archived, 1, "only the run needing no real upload gets counted");

        assert_eq!(
            store.latest_checkpoint(&will_fail).await.unwrap().unwrap().location,
            CheckpointLocation::R2Hot,
            "a run whose upload failed must not be marked archived"
        );
        assert_eq!(
            store.latest_checkpoint(&trivially_ok).await.unwrap().unwrap().location,
            CheckpointLocation::Drive
        );
    }

    #[tokio::test]
    async fn run_loop_sweeps_pending_work_on_its_first_tick() {
        let drive = Arc::new(FakeDrive::default());
        let (archiver, store, _artifacts) = test_env(drive).await;
        let run = new_run(&store, "loop-eligible").await;
        complete(&store, &run).await;
        store.record_checkpoint(&run, 1, "uri", "hash", &CheckpointMeta::default()).await.unwrap();

        let handle = tokio::spawn(archiver.run_loop(Duration::from_millis(5)));
        tokio::time::sleep(Duration::from_millis(80)).await;
        handle.abort();

        assert_eq!(
            store.latest_checkpoint(&run).await.unwrap().unwrap().location,
            CheckpointLocation::Drive,
            "run_loop must have swept at least once"
        );
    }

    #[tokio::test]
    async fn run_loop_tick_with_nothing_pending_is_a_quiet_noop() {
        let (archiver, store, _artifacts) = test_env(Arc::new(FakeDrive::default())).await;

        let handle = tokio::spawn(archiver.run_loop(Duration::from_millis(5)));
        tokio::time::sleep(Duration::from_millis(50)).await;
        handle.abort();

        assert!(store.runs(&Default::default(), 10).await.unwrap().is_empty());
    }
}
