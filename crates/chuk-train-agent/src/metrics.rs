//! Tail a job's metrics file (JSON-lines) and stream each record as a
//! [`WorkerToCp::Metric`]. A record is indexed by a numeric `step`; the
//! remaining numeric fields become the metric values. Records without a numeric
//! `step`, and records that carry no numeric values, are skipped.

use std::collections::BTreeMap;
use std::path::PathBuf;

use chuk_compute_wire::{JobId, WorkerToCp};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::seq::Seq;

/// The record field that carries the metric step index.
const METRIC_STEP_KEY: &str = "step";

/// Incremental tail over a metrics file. Each [`MetricTail::drain`] parses only
/// the lines appended since the previous pass, so no record is streamed twice.
pub struct MetricTail {
    path: PathBuf,
    processed: usize,
}

impl MetricTail {
    pub fn new(path: PathBuf) -> Self {
        Self { path, processed: 0 }
    }

    /// Parse newly-appended complete JSON-lines and stream them as metrics.
    pub async fn drain(&mut self, job_id: &JobId, seq: &Seq, tx: &UnboundedSender<WorkerToCp>) {
        let Ok(content) = tokio::fs::read_to_string(&self.path).await else {
            return;
        };
        let lines = complete_lines(&content);
        for line in lines.iter().skip(self.processed) {
            if let Some((step, values)) = parse_metric_line(line) {
                if !values.is_empty() {
                    let _ = tx.send(WorkerToCp::Metric {
                        seq: seq.next(),
                        job_id: Some(job_id.clone()),
                        step: Some(step),
                        values,
                    });
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

/// One JSON-lines metric record → `(step, numeric fields)`. Records without a
/// numeric `step` are skipped (metrics are indexed by step).
fn parse_metric_line(line: &str) -> Option<(u64, BTreeMap<String, f64>)> {
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
        assert!(parse_metric_line("   ").is_none()); // blank
        assert!(parse_metric_line("not json").is_none()); // unparseable
        assert!(parse_metric_line("[1,2,3]").is_none()); // not an object
    }

    #[tokio::test]
    async fn drain_streams_only_new_records_across_passes() {
        use tokio::sync::mpsc;

        let dir = std::env::temp_dir().join(format!("chuk-metrics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metrics.jsonl");
        std::fs::write(&path, b"{\"step\":1,\"loss\":2.0}\n").unwrap();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let seq = Seq::new();
        let mut tail = MetricTail::new(path.clone());
        tail.drain(&JobId::from("j1"), &seq, &tx).await;

        // A record with a numeric value is streamed with job + step present.
        let WorkerToCp::Metric { seq: s, job_id, step, values } = rx.try_recv().unwrap() else {
            panic!("expected a metric");
        };
        assert_eq!(s, 0);
        assert_eq!(job_id, Some(JobId::from("j1")));
        assert_eq!(step, Some(1));
        assert_eq!(values.get("loss"), Some(&2.0));
        assert!(rx.try_recv().is_err()); // only one so far

        // Append a value-less record (skipped) and a real one; only the real one
        // streams, and the first record is not re-sent.
        std::fs::write(
            &path,
            b"{\"step\":1,\"loss\":2.0}\n{\"step\":2}\n{\"step\":3,\"loss\":1.0}\n",
        )
        .unwrap();
        tail.drain(&JobId::from("j1"), &seq, &tx).await;
        let WorkerToCp::Metric { step, .. } = rx.try_recv().unwrap() else {
            panic!("expected a metric");
        };
        assert_eq!(step, Some(3));
        assert!(rx.try_recv().is_err());

        // A vanished file is a no-op rather than an error.
        std::fs::remove_file(&path).unwrap();
        tail.drain(&JobId::from("j1"), &seq, &tx).await;
        assert!(rx.try_recv().is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
