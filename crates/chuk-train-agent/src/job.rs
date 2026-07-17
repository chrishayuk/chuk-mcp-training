//! Shell run execution: spawn, stream stdout/stderr lines, enforce timeout,
//! report exit.

use std::process::Stdio;
use std::time::Duration;

use chuk_train_proto::{
    AgentToCp, JobAssignment, RunId, RunSpec, EXIT_CODE_AGENT_ERROR, EXIT_CODE_TIMEOUT,
};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tracing::warn;

const SHELL: &str = "/bin/sh";
const SHELL_FLAG: &str = "-c";
/// Prefix for lines the agent itself injects into a run's log stream.
const AGENT_LOG_PREFIX: &str = "[agent]";

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

pub fn spawn(job: JobAssignment, tx: UnboundedSender<AgentToCp>) -> RunningJob {
    RunningJob {
        handle: tokio::spawn(execute(job, tx)),
    }
}

async fn execute(job: JobAssignment, tx: UnboundedSender<AgentToCp>) {
    let JobAssignment { run_id, spec } = job;
    let RunSpec::Shell { command, timeout_s } = spec;
    let _ = tx.send(AgentToCp::JobStarted {
        run_id: run_id.clone(),
    });
    let code = run_shell(&run_id, &command, Duration::from_secs(timeout_s), &tx).await;
    let _ = tx.send(AgentToCp::JobExited { run_id, code });
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
            send_agent_line(tx, run_id, &format!("spawn failed: {error}"));
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
            send_agent_line(tx, run_id, &format!("wait failed: {error}"));
            EXIT_CODE_AGENT_ERROR
        }
        Err(_elapsed) => {
            send_agent_line(
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

    // The process is dead on every path above, so the pipes are closed and
    // the pumps drain whatever is buffered, then finish.
    if let Some(pump) = stdout_pump {
        let _ = pump.await;
    }
    if let Some(pump) = stderr_pump {
        let _ = pump.await;
    }
    code
}

fn pump_lines(
    reader: impl AsyncRead + Unpin + Send + 'static,
    run_id: RunId,
    tx: UnboundedSender<AgentToCp>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = tx.send(AgentToCp::Log {
                run_id: run_id.clone(),
                line,
            });
        }
    })
}

fn send_agent_line(tx: &UnboundedSender<AgentToCp>, run_id: &RunId, message: &str) {
    let _ = tx.send(AgentToCp::Log {
        run_id: run_id.clone(),
        line: format!("{AGENT_LOG_PREFIX} {message}"),
    });
}
