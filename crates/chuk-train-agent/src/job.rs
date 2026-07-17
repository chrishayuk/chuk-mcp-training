//! Run dispatch: turn an assignment into a supervised child job. Shell runs
//! (M0) execute here; train runs (M1) delegate to [`crate::train`].

use std::process::Stdio;
use std::time::Duration;

use chuk_train_proto::{
    AgentToCp, JobAssignment, RunId, RunSpec, EXIT_CODE_AGENT_ERROR, EXIT_CODE_TIMEOUT,
};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tracing::warn;

use crate::procio::{agent_line, pump_lines};

const SHELL: &str = "/bin/sh";
const SHELL_FLAG: &str = "-c";

pub struct RunningJob {
    handle: JoinHandle<()>,
}

impl RunningJob {
    pub fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Aborting drops the child future; `kill_on_drop` takes the process out.
    pub fn abort(self) {
        self.handle.abort();
    }
}

/// Spawn the supervisor task for an assignment. `origin` is the control-plane
/// HTTP origin, needed by train runs to fetch code + upload checkpoints.
pub fn spawn(job: JobAssignment, tx: UnboundedSender<AgentToCp>, origin: String) -> RunningJob {
    RunningJob {
        handle: tokio::spawn(execute(job, tx, origin)),
    }
}

async fn execute(job: JobAssignment, tx: UnboundedSender<AgentToCp>, origin: String) {
    let run_id = job.run_id.clone();
    tx.send(AgentToCp::JobStarted {
        run_id: run_id.clone(),
    })
    .ok();
    let code = match job.spec {
        RunSpec::Shell(shell) => {
            run_shell(
                &run_id,
                &shell.command,
                Duration::from_secs(shell.timeout_s),
                &tx,
            )
            .await
        }
        RunSpec::Train(_) => crate::train::run(job, &tx, &origin).await,
    };
    tx.send(AgentToCp::JobExited { run_id, code }).ok();
}

async fn run_shell(
    run_id: &RunId,
    command: &str,
    timeout: Duration,
    tx: &UnboundedSender<AgentToCp>,
) -> i64 {
    let spawned = Command::new(SHELL)
        .arg(SHELL_FLAG)
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            agent_line(tx, run_id, &format!("spawn failed: {error}"));
            return EXIT_CODE_AGENT_ERROR;
        }
    };

    let stdout_pump = child
        .stdout
        .take()
        .map(|out| pump_lines(out, run_id.clone(), tx.clone()));
    let stderr_pump = child
        .stderr
        .take()
        .map(|err| pump_lines(err, run_id.clone(), tx.clone()));

    let status = tokio::time::timeout(timeout, child.wait()).await;
    let code = match status {
        Ok(Ok(exit)) => exit.code().map(i64::from).unwrap_or(EXIT_CODE_AGENT_ERROR),
        Ok(Err(error)) => {
            agent_line(tx, run_id, &format!("wait failed: {error}"));
            EXIT_CODE_AGENT_ERROR
        }
        Err(_elapsed) => {
            agent_line(
                tx,
                run_id,
                &format!("killed: exceeded timeout_s={}", timeout.as_secs()),
            );
            if let Err(error) = child.kill().await {
                warn!(%error, "kill after timeout failed");
            }
            EXIT_CODE_TIMEOUT
        }
    };

    // The process is dead on every path above, so the pipes are closed and the
    // pumps drain whatever is buffered, then finish.
    if let Some(pump) = stdout_pump {
        let _ = pump.await;
    }
    if let Some(pump) = stderr_pump {
        let _ = pump.await;
    }
    code
}
