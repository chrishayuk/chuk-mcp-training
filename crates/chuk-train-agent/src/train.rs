//! Train run execution (spec §5.1 script contract, §11.2 lineage).
//!
//! The harness fetches the code unit, lays out a work dir, exports the
//! `$CHUK_*` env vars, runs the entrypoint, and while it runs: streams stdout
//! as logs, tails `$CHUK_METRICS` as metric records, and uploads any checkpoint
//! the trainer marks `.ready` — augmenting each `meta.json` to lineage-complete
//! before upload. On resume it fetches the last checkpoint and points
//! `$CHUK_RESUME_CKPT` at it, extending the slice list.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chuk_train_proto::{
    env as env_vars, keys, script_env, AgentToCp, CheckpointMeta, CodeRef, JobAssignment,
    ResumeInfo, RunId, RunSpec, TrainSpec, CHECKPOINT_META_FILE, CHECKPOINT_MODEL_FILE,
    CHECKPOINT_OPTIM_FILE, CHECKPOINT_READY_MARKER, CHECKPOINT_SCAN_INTERVAL,
    EXIT_CODE_AGENT_ERROR, EXIT_CODE_TIMEOUT,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

use crate::codeunit;
use crate::httpclient::HttpClient;
use crate::procio::{agent_line, pump_lines};

const SHELL: &str = "/bin/sh";
const SHELL_FLAG: &str = "-c";
const METRIC_POLL_INTERVAL: Duration = Duration::from_millis(500);
const METRIC_STEP_KEY: &str = "step";

/// Execute a train run to completion; returns the process exit code (or a
/// synthetic code on setup failure / timeout).
pub async fn run(job: JobAssignment, tx: &UnboundedSender<AgentToCp>, origin: &str) -> i64 {
    let RunSpec::Train(train) = &job.spec else {
        agent_line(
            tx,
            &job.run_id,
            "internal error: train::run on a non-train spec",
        );
        return EXIT_CODE_AGENT_ERROR;
    };
    let Some(grant) = &job.grant else {
        agent_line(
            tx,
            &job.run_id,
            "train run assigned without an upload grant",
        );
        return EXIT_CODE_AGENT_ERROR;
    };
    let client = HttpClient::new(origin.to_owned(), grant.token.clone());

    match execute(&job.run_id, train, job.resume.as_ref(), &client, tx).await {
        Ok(code) => code,
        Err(error) => {
            agent_line(tx, &job.run_id, &format!("train setup failed: {error:#}"));
            EXIT_CODE_AGENT_ERROR
        }
    }
}

async fn execute(
    run_id: &RunId,
    train: &TrainSpec,
    resume: Option<&ResumeInfo>,
    client: &HttpClient,
    tx: &UnboundedSender<AgentToCp>,
) -> Result<i64> {
    // 1. Code unit (cached by sha) + entrypoint command.
    let cache_dir = cache_dir();
    let unit_dir = codeunit::ensure_local(client, &cache_dir, &train.code).await?;
    let manifest = codeunit::read_manifest(&unit_dir).await?;
    let command = manifest
        .entrypoint(&train.entrypoint)
        .with_context(|| format!("entrypoint {:?} not in unit manifest", train.entrypoint))?
        .to_owned();

    // 2. Work dir layout.
    let work = WorkDir::create(run_id)?;
    let seed = train
        .seed
        .or_else(|| train.overrides.get("seed").and_then(Value::as_i64));
    let config_hash = compute_config_hash(&unit_dir, train).await?;

    // 3. Resume: fetch the last checkpoint locally, seed the slice history.
    let mut base_slices: Vec<[u64; 2]> = Vec::new();
    let mut parent: Option<String> = None;
    let from_step = resume.map(|r| r.from_step).unwrap_or(0);
    if let Some(resume) = resume {
        let (slices, meta_parent) = fetch_resume(client, resume, &work.resume_dir).await?;
        base_slices = slices;
        parent = meta_parent;
        agent_line(
            tx,
            run_id,
            &format!("resuming from step {}", resume.from_step),
        );
    }

    // 4. Spawn the entrypoint with the script-contract environment.
    let mut child = Command::new(SHELL)
        .arg(SHELL_FLAG)
        .arg(&command)
        .current_dir(&unit_dir)
        .envs(script_environment(
            run_id, train, seed, resume, &unit_dir, &work,
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("spawning train entrypoint")?;

    let stdout_pump = child
        .stdout
        .take()
        .map(|o| pump_lines(o, run_id.clone(), tx.clone()));
    let stderr_pump = child
        .stderr
        .take()
        .map(|e| pump_lines(e, run_id.clone(), tx.clone()));

    // 5. Supervise: race process exit against metric/checkpoint polling and the
    //    slice wall. Everything runs inside this future, so an abort (the agent
    //    losing its session) drops it all and kill_on_drop takes the child.
    let mut uploader = Checkpoints {
        run_id: run_id.clone(),
        code: train.code.clone(),
        ckpt_dir: work.ckpt_dir.clone(),
        seed,
        arch: train.arch.clone(),
        config_hash,
        from_step,
        base_slices,
        uploaded: BTreeSet::new(),
        parent,
    };
    let mut metrics = MetricTail::new(work.metrics_path.clone());

    let deadline = Instant::now() + Duration::from_secs(train.timeout_s);
    let mut scan_tick = tokio::time::interval(CHECKPOINT_SCAN_INTERVAL);
    let mut metric_tick = tokio::time::interval(METRIC_POLL_INTERVAL);
    let code = loop {
        tokio::select! {
            status = child.wait() => break exit_code(status),
            _ = scan_tick.tick() => uploader.scan_and_upload(client, tx).await,
            _ = metric_tick.tick() => metrics.drain(run_id, tx).await,
            _ = tokio::time::sleep_until(deadline.into()) => {
                agent_line(tx, run_id, &format!("killed: exceeded timeout_s={}", train.timeout_s));
                let _ = child.kill().await;
                break EXIT_CODE_TIMEOUT;
            }
        }
    };

    // 6. Final drain: catch the last metrics and any checkpoint written just
    //    before exit (the resume test depends on that last one being uploaded).
    metrics.drain(run_id, tx).await;
    uploader.scan_and_upload(client, tx).await;
    if let Some(p) = stdout_pump {
        let _ = p.await;
    }
    if let Some(p) = stderr_pump {
        let _ = p.await;
    }
    Ok(code)
}

fn exit_code(status: std::io::Result<std::process::ExitStatus>) -> i64 {
    match status {
        Ok(exit) => exit.code().map(i64::from).unwrap_or(EXIT_CODE_AGENT_ERROR),
        Err(_) => EXIT_CODE_AGENT_ERROR,
    }
}

// ---------------------------------------------------------------------------
// Work directory + environment
// ---------------------------------------------------------------------------

struct WorkDir {
    ckpt_dir: PathBuf,
    metrics_path: PathBuf,
    resume_dir: PathBuf,
}

impl WorkDir {
    fn create(run_id: &RunId) -> Result<Self> {
        let root = std::env::temp_dir().join(format!("chuk-train-run-{run_id}"));
        // Start each slice from a clean local dir: the authoritative
        // checkpoints live in the store (fetched into resume/ if resuming), so
        // any leftover step_<n>/ from a prior slice on this machine must not be
        // re-uploaded with the wrong slice bounds.
        let _ = std::fs::remove_dir_all(&root);
        let ckpt_dir = root.join("ckpt");
        let resume_dir = root.join("resume");
        std::fs::create_dir_all(&ckpt_dir)?;
        std::fs::create_dir_all(&resume_dir)?;
        let metrics_path = root.join("metrics.jsonl");
        std::fs::write(&metrics_path, b"")?;
        Ok(Self {
            ckpt_dir,
            metrics_path,
            resume_dir,
        })
    }
}

fn script_environment(
    run_id: &RunId,
    train: &TrainSpec,
    seed: Option<i64>,
    resume: Option<&ResumeInfo>,
    unit_dir: &Path,
    work: &WorkDir,
) -> Vec<(String, String)> {
    // Config path is relative to the code unit; export it absolute so the
    // trainer resolves it regardless of its own working directory.
    let config = train
        .config
        .as_ref()
        .map(|rel| unit_dir.join(rel).display().to_string())
        .unwrap_or_default();
    let resume_ckpt = resume
        .map(|_| work.resume_dir.display().to_string())
        .unwrap_or_default();
    vec![
        (script_env::RUN_ID.to_owned(), run_id.0.clone()),
        (script_env::CONFIG.to_owned(), config),
        (script_env::OVERRIDES.to_owned(), overrides_json(train)),
        (
            script_env::METRICS.to_owned(),
            work.metrics_path.display().to_string(),
        ),
        (
            script_env::CKPT_DIR.to_owned(),
            work.ckpt_dir.display().to_string(),
        ),
        (script_env::RESUME_CKPT.to_owned(), resume_ckpt),
        (
            script_env::SEED.to_owned(),
            seed.map(|s| s.to_string()).unwrap_or_default(),
        ),
    ]
}

fn overrides_json(train: &TrainSpec) -> String {
    if train.overrides.is_null() {
        "{}".to_owned()
    } else {
        train.overrides.to_string()
    }
}

async fn compute_config_hash(unit_dir: &Path, train: &TrainSpec) -> Result<Option<String>> {
    let mut hasher = Sha256::new();
    let mut any = false;
    if let Some(cfg) = &train.config {
        if let Ok(bytes) = tokio::fs::read(unit_dir.join(cfg)).await {
            hasher.update(&bytes);
            any = true;
        }
    }
    if !train.overrides.is_null() {
        hasher.update(train.overrides.to_string().as_bytes());
        any = true;
    }
    Ok(any.then(|| hex::encode(hasher.finalize())))
}

fn cache_dir() -> PathBuf {
    std::env::var(env_vars::AGENT_CACHE_DIR)
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("chuk-train-cache"))
}

// ---------------------------------------------------------------------------
// Resume
// ---------------------------------------------------------------------------

/// Fetch the resume checkpoint's model + meta into `resume_dir`; return its
/// slice history and a parent pointer for the next checkpoint's lineage.
async fn fetch_resume(
    client: &HttpClient,
    resume: &ResumeInfo,
    resume_dir: &Path,
) -> Result<(Vec<[u64; 2]>, Option<String>)> {
    let model_key = format!("{}/{CHECKPOINT_MODEL_FILE}", resume.checkpoint_path);
    let meta_key = format!("{}/{CHECKPOINT_META_FILE}", resume.checkpoint_path);
    let model = client.fetch(&model_key).await?;
    tokio::fs::write(resume_dir.join(CHECKPOINT_MODEL_FILE), model).await?;

    let mut slices = Vec::new();
    if let Ok(meta_bytes) = client.fetch(&meta_key).await {
        tokio::fs::write(resume_dir.join(CHECKPOINT_META_FILE), &meta_bytes).await?;
        if let Ok(meta) = serde_json::from_slice::<CheckpointMeta>(&meta_bytes) {
            slices = meta.slices;
        }
    }
    Ok((slices, Some(resume.checkpoint_path.clone())))
}

// ---------------------------------------------------------------------------
// Checkpoint uploads
// ---------------------------------------------------------------------------

struct Checkpoints {
    run_id: RunId,
    code: CodeRef,
    ckpt_dir: PathBuf,
    seed: Option<i64>,
    arch: Option<String>,
    config_hash: Option<String>,
    from_step: u64,
    base_slices: Vec<[u64; 2]>,
    uploaded: BTreeSet<u64>,
    parent: Option<String>,
}

impl Checkpoints {
    /// Upload every `.ready` checkpoint not yet sent, in ascending step order.
    async fn scan_and_upload(&mut self, client: &HttpClient, tx: &UnboundedSender<AgentToCp>) {
        for step in self.ready_steps() {
            if let Err(error) = self.upload_one(client, tx, step).await {
                agent_line(
                    tx,
                    &self.run_id,
                    &format!("checkpoint step_{step} upload failed: {error:#}"),
                );
            }
            // Mark done regardless: a malformed checkpoint should not be retried
            // every tick forever.
            self.uploaded.insert(step);
        }
    }

    fn ready_steps(&self) -> Vec<u64> {
        let mut steps = Vec::new();
        let Ok(entries) = std::fs::read_dir(&self.ckpt_dir) else {
            return steps;
        };
        for entry in entries.flatten() {
            let Some(step) = parse_step_dir(&entry.file_name().to_string_lossy()) else {
                continue;
            };
            if self.uploaded.contains(&step) {
                continue;
            }
            if entry.path().join(CHECKPOINT_READY_MARKER).exists() {
                steps.push(step);
            }
        }
        steps.sort_unstable();
        steps
    }

    async fn upload_one(
        &mut self,
        client: &HttpClient,
        tx: &UnboundedSender<AgentToCp>,
        step: u64,
    ) -> Result<()> {
        let dir = self
            .ckpt_dir
            .join(format!("{}{step}", chuk_train_proto::CHECKPOINT_DIR_PREFIX));
        let model = tokio::fs::read(dir.join(CHECKPOINT_MODEL_FILE))
            .await
            .with_context(|| format!("checkpoint step_{step} has no {CHECKPOINT_MODEL_FILE}"))?;
        let model_hash = hex::encode(Sha256::digest(&model));

        let meta = self.build_meta(&dir, step).await?;
        let meta_bytes = serde_json::to_vec_pretty(&meta)?;

        client
            .upload(
                &keys::checkpoint_file(&self.run_id.0, step, CHECKPOINT_MODEL_FILE),
                model,
            )
            .await?;
        client
            .upload(
                &keys::checkpoint_file(&self.run_id.0, step, CHECKPOINT_META_FILE),
                meta_bytes,
            )
            .await?;
        // Optimizer state is optional and excluded from lazarus pulls, but we
        // still store it so a slice can resume with its optimizer intact.
        if let Ok(optim) = tokio::fs::read(dir.join(CHECKPOINT_OPTIM_FILE)).await {
            client
                .upload(
                    &keys::checkpoint_file(&self.run_id.0, step, CHECKPOINT_OPTIM_FILE),
                    optim,
                )
                .await?;
        }

        tx.send(AgentToCp::Checkpoint {
            run_id: self.run_id.clone(),
            step,
            model_hash,
            meta,
        })
        .ok();
        self.parent = Some(keys::checkpoint_dir(&self.run_id.0, step));
        Ok(())
    }

    /// Merge the trainer's partial sidecar with harness-known lineage.
    async fn build_meta(&self, dir: &Path, step: u64) -> Result<CheckpointMeta> {
        let mut meta: CheckpointMeta = match tokio::fs::read(dir.join(CHECKPOINT_META_FILE)).await {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => CheckpointMeta::default(),
        };
        meta.step = step;
        meta.run_id = Some(self.run_id.clone());
        meta.code.get_or_insert_with(|| self.code.clone());
        if meta.seed.is_none() {
            meta.seed = self.seed;
        }
        if meta.arch.is_none() {
            meta.arch = self.arch.clone();
        }
        if meta.config_hash.is_none() {
            meta.config_hash = self.config_hash.clone();
        }
        if meta.parent_checkpoint.is_none() {
            meta.parent_checkpoint = self.parent.clone();
        }
        let mut slices = self.base_slices.clone();
        slices.push([self.from_step, step]);
        meta.slices = slices;
        Ok(meta)
    }
}

fn parse_step_dir(name: &str) -> Option<u64> {
    name.strip_prefix(chuk_train_proto::CHECKPOINT_DIR_PREFIX)?
        .parse()
        .ok()
}

// ---------------------------------------------------------------------------
// Metrics tail
// ---------------------------------------------------------------------------

struct MetricTail {
    path: PathBuf,
    processed: usize,
}

impl MetricTail {
    fn new(path: PathBuf) -> Self {
        Self { path, processed: 0 }
    }

    /// Parse newly-appended complete JSONL lines and stream them as metrics.
    async fn drain(&mut self, run_id: &RunId, tx: &UnboundedSender<AgentToCp>) {
        let Ok(content) = tokio::fs::read_to_string(&self.path).await else {
            return;
        };
        let lines = complete_lines(&content);
        for line in lines.iter().skip(self.processed) {
            if let Some((step, values)) = parse_metric_line(line) {
                if !values.is_empty() {
                    tx.send(AgentToCp::Metric {
                        run_id: run_id.clone(),
                        step,
                        values,
                    })
                    .ok();
                }
            }
        }
        self.processed = lines.len();
    }
}

/// The complete (newline-terminated) lines in `content`. Dropping the final
/// split element handles both cases: on a `\n`-terminated file it is the empty
/// string after the last newline; mid-write it is a partial record to hold for
/// the next pass. Everything before it is a complete record.
fn complete_lines(content: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = content.split('\n').collect();
    lines.pop();
    lines
}

/// One JSONL metric record → (step, numeric fields). Records without a numeric
/// `step` are skipped (metrics are indexed by step, spec §6).
fn parse_metric_line(line: &str) -> Option<(u64, std::collections::BTreeMap<String, f64>)> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let record: Value = serde_json::from_str(line).ok()?;
    let obj = record.as_object()?;
    let step = obj.get(METRIC_STEP_KEY).and_then(Value::as_f64)? as u64;
    let values = obj
        .iter()
        .filter(|(k, _)| k.as_str() != METRIC_STEP_KEY)
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect();
    Some((step, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_lines_holds_partial_and_drops_trailing_newline() {
        // Newline-terminated: both real lines are complete.
        assert_eq!(complete_lines("a\nb\n"), vec!["a", "b"]);
        // Mid-write: the last fragment is held back.
        assert_eq!(complete_lines("a\nb\nc"), vec!["a", "b"]);
        // Advancing `processed` across passes never skips a line.
        assert_eq!(complete_lines("").len(), 0);
        assert_eq!(complete_lines("x").len(), 0);
    }

    #[test]
    fn parse_metric_line_extracts_step_and_numbers() {
        let (step, values) = parse_metric_line(r#"{"step": 7, "loss": 1.5, "note": "x"}"#).unwrap();
        assert_eq!(step, 7);
        assert_eq!(values.get("loss"), Some(&1.5));
        assert!(!values.contains_key("note")); // non-numeric dropped
        assert!(parse_metric_line(r#"{"loss": 1.0}"#).is_none()); // no step
    }
}
