//! Shared child-process I/O: stream stdout/stderr lines back as run logs, and
//! inject the agent's own annotations into the same stream.

use chuk_train_proto::{AgentToCp, RunId};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

/// Prefix for lines the agent itself injects into a run's log stream.
const AGENT_LOG_PREFIX: &str = "[agent]";

/// Spawn a task that forwards every line from `reader` as a `Log` message.
pub fn pump_lines(
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

/// Emit an agent-authored line into a run's log stream.
pub fn agent_line(tx: &UnboundedSender<AgentToCp>, run_id: &RunId, message: &str) {
    let _ = tx.send(AgentToCp::Log {
        run_id: run_id.clone(),
        line: format!("{AGENT_LOG_PREFIX} {message}"),
    });
}
