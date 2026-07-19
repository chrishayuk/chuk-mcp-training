//! Child-process I/O: stream a command's stdout/stderr lines back as job logs,
//! and inject the worker's own annotations into the same stream. Every line is
//! stamped with the next sequence value so the control plane can order them.

use chuk_compute_wire::{JobId, WorkerToCp};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

use crate::seq::Seq;

/// Prefix for lines the worker itself injects into a job's log stream.
const WORKER_LOG_PREFIX: &str = "[worker]";

/// Spawn a task that forwards every line from `reader` as a [`WorkerToCp::Log`].
/// The task ends when the reader hits EOF (the child's pipe closes on exit).
pub fn pump_lines(
    reader: impl AsyncRead + Unpin + Send + 'static,
    job_id: JobId,
    seq: Seq,
    tx: UnboundedSender<WorkerToCp>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let _ = tx.send(WorkerToCp::Log {
                seq: seq.next(),
                job_id: job_id.clone(),
                line,
            });
        }
    })
}

/// Emit a worker-authored line into a job's log stream.
pub fn worker_line(seq: &Seq, job_id: &JobId, tx: &UnboundedSender<WorkerToCp>, message: &str) {
    let _ = tx.send(WorkerToCp::Log {
        seq: seq.next(),
        job_id: job_id.clone(),
        line: format!("{WORKER_LOG_PREFIX} {message}"),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn drain(rx: &mut mpsc::UnboundedReceiver<WorkerToCp>) -> Vec<(u64, String)> {
        let mut out = Vec::new();
        while let Ok(WorkerToCp::Log { seq, line, .. }) = rx.try_recv() {
            out.push((seq, line));
        }
        out
    }

    #[tokio::test]
    async fn pump_forwards_each_line_with_ascending_seq() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let reader: &'static [u8] = b"first\nsecond\nthird\n";
        pump_lines(reader, JobId::from("j1"), Seq::new(), tx)
            .await
            .unwrap();

        let lines = drain(&mut rx);
        assert_eq!(
            lines,
            vec![
                (0, "first".to_owned()),
                (1, "second".to_owned()),
                (2, "third".to_owned()),
            ]
        );
    }

    #[tokio::test]
    async fn pump_forwards_a_final_unterminated_line() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let reader: &'static [u8] = b"no newline";
        pump_lines(reader, JobId::from("j1"), Seq::new(), tx)
            .await
            .unwrap();
        assert_eq!(drain(&mut rx), vec![(0, "no newline".to_owned())]);
    }

    #[test]
    fn worker_line_prefixes_and_stamps() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let seq = Seq::new();
        worker_line(&seq, &JobId::from("j1"), &tx, "hello");
        let WorkerToCp::Log { seq, job_id, line } = rx.try_recv().unwrap() else {
            panic!("expected a log");
        };
        assert_eq!(seq, 0);
        assert_eq!(job_id, JobId::from("j1"));
        assert_eq!(line, format!("{WORKER_LOG_PREFIX} hello"));
    }
}
