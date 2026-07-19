//! The generic executor. Given a [`Job`] it stages inputs into a fresh sandbox,
//! runs one command under supervision, streams logs and metrics, collects
//! outputs, and reports the terminal state — knowing nothing about what the job
//! computes. All workload specifics arrive encoded in the [`Job`].

use std::collections::BTreeMap;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result};
use chuk_compute_wire::{Job, JobId, KillReason, UploadPolicy, WorkerToCp};
// Reused, domain-free: the cadence at which we rescan for new outputs.
use chuk_train_proto::{CHECKPOINT_SCAN_INTERVAL as OUTPUT_SCAN_INTERVAL, EXIT_CODE_AGENT_ERROR};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tokio::time::Instant;

use crate::httpclient::HttpClient;
use crate::inputs;
use crate::metrics::MetricTail;
use crate::outputs::OutputCollector;
use crate::procio::{pump_lines, worker_line};
use crate::sandbox::{subst, Sandbox};
use crate::seq::Seq;

/// How often the metrics file is tailed while a job runs.
const METRIC_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// A supervised job in flight. Aborting the handle drops the executor future;
/// `kill_on_drop` then takes out the child process.
pub struct RunningJob {
    handle: JoinHandle<()>,
    job_id: JobId,
}

impl RunningJob {
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    pub fn job_id(&self) -> &JobId {
        &self.job_id
    }

    /// Aborting drops the child future; `kill_on_drop` takes the process out.
    pub fn abort(self) {
        self.handle.abort();
    }
}

/// Spawn the supervisor task for `job`. `origin` is the control-plane HTTP
/// origin used to fetch store-keyed inputs and upload outputs.
pub fn spawn(job: Job, tx: UnboundedSender<WorkerToCp>, seq: Seq, origin: String) -> RunningJob {
    let job_id = job.id.clone();
    RunningJob {
        handle: tokio::spawn(execute(job, tx, seq, origin)),
        job_id,
    }
}

/// How a job finished: a process exit code, or a kill for a stated reason.
enum Outcome {
    Exited(i64),
    Killed(KillReason),
}

async fn execute(job: Job, tx: UnboundedSender<WorkerToCp>, seq: Seq, origin: String) {
    let job_id = job.id.clone();
    let _ = tx.send(WorkerToCp::JobStarted {
        seq: seq.next(),
        job_id: job_id.clone(),
    });

    let outcome = match run(job, &tx, &seq, &origin).await {
        Ok(outcome) => outcome,
        Err(error) => {
            worker_line(&seq, &job_id, &tx, &format!("job setup failed: {error:#}"));
            Outcome::Exited(EXIT_CODE_AGENT_ERROR)
        }
    };

    let message = match outcome {
        Outcome::Exited(code) => WorkerToCp::JobExited {
            seq: seq.next(),
            job_id,
            code,
        },
        Outcome::Killed(reason) => WorkerToCp::JobKilled {
            seq: seq.next(),
            job_id,
            reason,
        },
    };
    let _ = tx.send(message);
}

async fn run(
    job: Job,
    tx: &UnboundedSender<WorkerToCp>,
    seq: &Seq,
    origin: &str,
) -> Result<Outcome> {
    let job_id = job.id.clone();
    let sandbox = Sandbox::create(&job.id).context("creating sandbox")?;
    let sandbox_path = sandbox.path();
    let client = HttpClient::new(origin.to_owned(), job.grant.clone().unwrap_or_default());

    // 1. Stage inputs into the sandbox.
    for input in &job.inputs {
        inputs::stage(input, sandbox_path, &client).await?;
    }

    // 2. Metrics tail + output collectors, resolved against the sandbox.
    let mut metrics = job
        .metrics_file
        .as_ref()
        .map(|file| MetricTail::new(subst(file, sandbox_path).into()));
    let (mut on_appearance, mut on_exit) = collectors_by_policy(&job, sandbox_path);

    // 3. Spawn the command directly (argv, no shell) inside the sandbox.
    let (program, args, env) = command_parts(&job.command, &job.env, sandbox_path)
        .context("job command is empty")?;
    let mut child = Command::new(&program)
        .args(&args)
        .envs(env)
        .current_dir(sandbox.root())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawning {program}"))?;

    let stdout_pump = child
        .stdout
        .take()
        .map(|out| pump_lines(out, job_id.clone(), seq.clone(), tx.clone()));
    let stderr_pump = child
        .stderr
        .take()
        .map(|err| pump_lines(err, job_id.clone(), seq.clone(), tx.clone()));

    // 4. Supervise: race process exit against metric/output polling and the
    //    runtime budget. Everything runs inside this future, so aborting the
    //    task (a lost session, or a cancel) drops it all and kill_on_drop takes
    //    the child.
    let deadline = job
        .max_runtime_secs
        .map(|secs| Instant::now() + Duration::from_secs(secs));
    let mut metric_tick = tokio::time::interval(METRIC_POLL_INTERVAL);
    let mut output_tick = tokio::time::interval(OUTPUT_SCAN_INTERVAL);

    let outcome = loop {
        tokio::select! {
            status = child.wait() => break Outcome::Exited(exit_code(status)),
            _ = metric_tick.tick() => tail_metrics(&mut metrics, &job_id, seq, tx).await,
            _ = output_tick.tick() => collect_all(&mut on_appearance, &client, &job_id, seq, tx).await,
            _ = sleep_until_deadline(deadline) => {
                worker_line(seq, &job_id, tx, "killed: exceeded max_runtime_secs");
                let _ = child.kill().await;
                break Outcome::Killed(KillReason::MaxRuntime);
            }
        }
    };

    // 5. Final drain (critical for parity): the last metrics and any output
    //    written just before exit must still be captured, then the OnExit rules
    //    process, then the log pumps finish flushing.
    tail_metrics(&mut metrics, &job_id, seq, tx).await;
    collect_all(&mut on_appearance, &client, &job_id, seq, tx).await;
    collect_all(&mut on_exit, &client, &job_id, seq, tx).await;
    if let Some(pump) = stdout_pump {
        let _ = pump.await;
    }
    if let Some(pump) = stderr_pump {
        let _ = pump.await;
    }
    Ok(outcome)
}

/// Split a job's output rules into the buckets the supervisor drives: those
/// collected as new matches appear, and those collected once on exit. `Stream`
/// rules produce no artifact in M1 (live tailing rides the metric/log channel).
fn collectors_by_policy(job: &Job, sandbox_path: &str) -> (Vec<OutputCollector>, Vec<OutputCollector>) {
    let mut on_appearance = Vec::new();
    let mut on_exit = Vec::new();
    for rule in &job.outputs {
        match rule.upload {
            UploadPolicy::OnAppearance => on_appearance.push(OutputCollector::new(rule, sandbox_path)),
            UploadPolicy::OnExit => on_exit.push(OutputCollector::new(rule, sandbox_path)),
            // Stream (and any future policy) produces no collected artifact in
            // M1; live tailing rides the metric/log channel instead.
            _ => {}
        }
    }
    (on_appearance, on_exit)
}

/// A command resolved for spawning: the program, its arguments, and the
/// environment pairs, each with the sandbox path substituted in.
type CommandParts = (String, Vec<String>, Vec<(String, String)>);

/// Resolve the command into `(program, args, env)` with the sandbox path
/// substituted throughout. `None` when the command is empty.
fn command_parts(
    command: &[String],
    env: &BTreeMap<String, String>,
    sandbox_path: &str,
) -> Option<CommandParts> {
    let (program, args) = command.split_first()?;
    Some((
        subst(program, sandbox_path),
        args.iter().map(|arg| subst(arg, sandbox_path)).collect(),
        env.iter()
            .map(|(key, value)| (key.clone(), subst(value, sandbox_path)))
            .collect(),
    ))
}

async fn tail_metrics(
    metrics: &mut Option<MetricTail>,
    job_id: &JobId,
    seq: &Seq,
    tx: &UnboundedSender<WorkerToCp>,
) {
    if let Some(tail) = metrics.as_mut() {
        tail.drain(job_id, seq, tx).await;
    }
}

async fn collect_all(
    collectors: &mut [OutputCollector],
    client: &HttpClient,
    job_id: &JobId,
    seq: &Seq,
    tx: &UnboundedSender<WorkerToCp>,
) {
    for collector in collectors.iter_mut() {
        collector.collect(client, job_id, seq, tx).await;
    }
}

/// Sleep until `deadline` if there is one, else never resolve — the runtime
/// budget branch is inert for a job without one.
async fn sleep_until_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(at) => tokio::time::sleep_until(at).await,
        None => std::future::pending().await,
    }
}

fn exit_code(status: std::io::Result<std::process::ExitStatus>) -> i64 {
    match status {
        Ok(exit) => exit.code().map(i64::from).unwrap_or(EXIT_CODE_AGENT_ERROR),
        Err(_) => EXIT_CODE_AGENT_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_compute_wire::{ArtifactClass, OutputRule, Placement, Template};
    use tokio::sync::mpsc;

    fn bare_job(id: &str, command: Vec<&str>) -> Job {
        Job {
            id: JobId::from(id),
            template: Template::from("test"),
            command: command.into_iter().map(str::to_owned).collect(),
            env: BTreeMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metrics_file: None,
            max_runtime_secs: None,
            term_grace_secs: 5,
            service: None,
            needs: Vec::new(),
            campaign: None,
            placement: Placement::default(),
            grant: None,
        }
    }

    #[test]
    fn command_parts_substitutes_and_rejects_empty() {
        let mut env = BTreeMap::new();
        env.insert("OUT".to_owned(), "${SANDBOX}/o".to_owned());
        let command = vec!["${SANDBOX}/bin".to_owned(), "--dir=${SANDBOX}".to_owned()];
        let (program, args, env) = command_parts(&command, &env, "/box").unwrap();
        assert_eq!(program, "/box/bin");
        assert_eq!(args, vec!["--dir=/box".to_owned()]);
        assert_eq!(env, vec![("OUT".to_owned(), "/box/o".to_owned())]);

        assert!(command_parts(&[], &BTreeMap::new(), "/box").is_none());
    }

    #[test]
    fn collectors_by_policy_buckets_the_rules() {
        let rule = |policy| OutputRule {
            class: ArtifactClass::from("c"),
            glob: "${SANDBOX}/out/*".into(),
            upload: policy,
            key_prefix: "k".into(),
            ready_marker: None,
        };
        let mut job = bare_job("j", vec!["true"]);
        job.outputs = vec![
            rule(UploadPolicy::OnAppearance),
            rule(UploadPolicy::OnExit),
            rule(UploadPolicy::OnAppearance),
            rule(UploadPolicy::Stream),
        ];
        let (on_appearance, on_exit) = collectors_by_policy(&job, "/box");
        assert_eq!(on_appearance.len(), 2);
        assert_eq!(on_exit.len(), 1); // Stream produces no collector
    }

    #[cfg(unix)]
    #[test]
    fn exit_code_reads_the_process_code() {
        use std::os::unix::process::ExitStatusExt;
        let ok = std::process::ExitStatus::from_raw(0);
        assert_eq!(exit_code(Ok(ok)), 0);
        // A signal-terminated status has no code → the worker-error sentinel.
        let signalled = std::process::ExitStatus::from_raw(9);
        assert_eq!(exit_code(Ok(signalled)), EXIT_CODE_AGENT_ERROR);
        let err = Err(std::io::Error::other("boom"));
        assert_eq!(exit_code(err), EXIT_CODE_AGENT_ERROR);
    }

    fn collect_messages(rx: &mut mpsc::UnboundedReceiver<WorkerToCp>) -> Vec<WorkerToCp> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            out.push(msg);
        }
        out
    }

    #[tokio::test]
    async fn execute_reports_started_then_exited_with_the_code() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let job = bare_job("j-ok", vec!["/bin/sh", "-c", "printf 'hi\\n'; exit 3"]);
        execute(job, tx, Seq::new(), "http://unused".into()).await;

        let messages = collect_messages(&mut rx);
        assert!(matches!(messages.first(), Some(WorkerToCp::JobStarted { .. })));
        assert!(messages
            .iter()
            .any(|m| matches!(m, WorkerToCp::Log { line, .. } if line == "hi")));
        assert!(messages
            .iter()
            .any(|m| matches!(m, WorkerToCp::JobExited { code: 3, .. })));
    }

    #[tokio::test]
    async fn execute_reports_agent_error_when_the_command_is_empty() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        execute(bare_job("j-empty", vec![]), tx, Seq::new(), "http://unused".into()).await;
        let messages = collect_messages(&mut rx);
        assert!(matches!(messages.first(), Some(WorkerToCp::JobStarted { .. })));
        assert!(messages
            .iter()
            .any(|m| matches!(m, WorkerToCp::JobExited { code, .. } if *code == EXIT_CODE_AGENT_ERROR)));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn execute_kills_a_job_that_exceeds_its_runtime_budget() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut job = bare_job("j-slow", vec!["/bin/sh", "-c", "sleep 30"]);
        job.max_runtime_secs = Some(0); // deadline is immediate
        execute(job, tx, Seq::new(), "http://unused".into()).await;

        let messages = collect_messages(&mut rx);
        assert!(messages
            .iter()
            .any(|m| matches!(m, WorkerToCp::JobKilled { reason: KillReason::MaxRuntime, .. })));
        // A killed job reports no ordinary exit.
        assert!(!messages
            .iter()
            .any(|m| matches!(m, WorkerToCp::JobExited { .. })));
    }
}
