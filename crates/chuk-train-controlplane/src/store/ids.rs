//! Backend-agnostic store helpers shared by every `Store` adapter.
//!
//! The module's namesake is run-id / date formatting, but it also holds the
//! other pure, dialect-independent helpers both the SQLite and Postgres
//! adapters need: metric downsampling and the small serde/JSON converters.
//! Centralising them means the id/date logic (and the rest) lives in exactly
//! one place regardless of which backend is compiled in — no per-adapter drift.

use std::collections::BTreeMap;

use anyhow::Result;
use chuk_train_proto::{MetricPoint, TelemetryPoint, UnixSeconds, WorkerId, WorkerTelemetry};

/// Our run ids are `EXEC-YYYYMMDD-HHMMSS-{5-digit sequence}` (UTC) —
/// self-describing, chronologically sortable, and **deliberately distinct from
/// chuk-experiments-server's `RUN-…` ids**. Ours names an *execution attempt*
/// (this seed ran here, disconnected, resumed, finished); theirs names a
/// *research run* (the logical experiment). The two live in independent
/// namespaces, linked by an explicit parent reference — never conflated by a
/// shared prefix. The sequence comes from a store-backed monotonic counter, so
/// it's stable and collision-free.
const EXEC_ID_PREFIX: &str = "EXEC-";
/// Sweep ids share the sortable shape under their own prefix (spec §5.2); a
/// sweep names a *fan-out*, not an execution, so it gets its own namespace.
const SWEEP_ID_PREFIX: &str = "SWEEP-";
/// Zero-pad width of the sequence tail (grows past 5 digits after 99999).
const RUN_ID_SEQ_WIDTH: usize = 5;

/// Wall-clock now as fractional unix seconds. Shared by every adapter so the
/// timestamp semantics (`f64` unix seconds) are byte-for-byte identical across
/// backends.
pub(super) fn now() -> UnixSeconds {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Build a run id from a wall-clock timestamp + a store-backed sequence number.
pub(super) fn new_run_id(at: UnixSeconds, seq: i64) -> String {
    stamped_id(EXEC_ID_PREFIX, at, seq)
}

/// Build a sweep id (same sequence counter as runs — ids stay unique and
/// chronologically sortable across both namespaces).
pub(super) fn new_sweep_id(at: UnixSeconds, seq: i64) -> String {
    stamped_id(SWEEP_ID_PREFIX, at, seq)
}

fn stamped_id(prefix: &str, at: UnixSeconds, seq: i64) -> String {
    let (y, m, d, hh, mm, ss) = utc_parts(at as i64);
    format!(
        "{prefix}{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}-{seq:0width$}",
        width = RUN_ID_SEQ_WIDTH
    )
}

/// Break a unix timestamp into UTC (year, month, day, hour, minute, second).
fn utc_parts(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    (
        y,
        m,
        d,
        (rem / 3600) as u32,
        (rem % 3600 / 60) as u32,
        (rem % 60) as u32,
    )
}

/// Howard Hinnant's civil-from-days: days since the unix epoch → (year, month,
/// day) in the proleptic Gregorian calendar. Keeps the one timestamp we format
/// from pulling in a date crate.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Serialise a unit (fieldless) enum to the `&str` its serde repr uses. The
/// stored `state`/`event` columns are these strings, so both adapters go
/// through here to keep the on-disk vocabulary identical.
pub(super) fn enum_to_string<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .expect("unit enum serialises to a string")
}

/// Inverse of [`enum_to_string`]: parse a stored string back into its enum.
pub(super) fn enum_from_string<T: serde::de::DeserializeOwned>(raw: String) -> Result<T> {
    Ok(serde_json::from_value(serde_json::Value::String(raw))?)
}

/// Stride-downsample to at most `max` points, always keeping the last point so
/// the latest step is never dropped. Pure over `MetricPoint`, so it is shared
/// verbatim by every adapter's `metric_series`.
pub(super) fn downsample_in_place(points: &mut Vec<MetricPoint>, max: usize) {
    if max == 0 || points.len() <= max {
        return;
    }
    let stride = points.len().div_ceil(max);
    let last = points.last().copied();
    let mut kept: Vec<MetricPoint> = points.iter().step_by(stride).copied().collect();
    if let Some(last) = last {
        if kept.last() != Some(&last) {
            kept.push(last);
        }
    }
    *points = kept;
}

/// Insert `field` under `key` into a JSON object value, coercing non-objects to
/// an empty object first. Used to fold `worker`/`exit_code` into event details.
pub(super) fn merge_field(value: &mut serde_json::Value, key: &str, field: serde_json::Value) {
    if !value.is_object() {
        *value = serde_json::json!({});
    }
    value
        .as_object_mut()
        .expect("just ensured object")
        .insert(key.to_owned(), field);
}

/// Fold a worker's stored samples — `(ts, payload_json)` rows **ascending by
/// ts** — into a [`WorkerTelemetry`]: `values`/`sampled_at` from the newest
/// sample (live gauges), and per-key `series` across the whole window
/// (sparklines). `None` when the worker has no samples. Shared by both adapters.
pub(super) fn worker_telemetry_from_samples(
    worker_id: &WorkerId,
    samples: Vec<(UnixSeconds, String)>,
) -> Result<Option<WorkerTelemetry>> {
    if samples.is_empty() {
        return Ok(None);
    }
    let mut series: BTreeMap<String, Vec<TelemetryPoint>> = BTreeMap::new();
    let mut latest: Option<(UnixSeconds, BTreeMap<String, f64>)> = None;
    for (ts, payload) in samples {
        let values: BTreeMap<String, f64> = serde_json::from_str(&payload)?;
        for (key, &value) in &values {
            series
                .entry(key.clone())
                .or_default()
                .push(TelemetryPoint { ts, value });
        }
        latest = Some((ts, values));
    }
    let (sampled_at, values) = latest.expect("non-empty checked above");
    Ok(Some(WorkerTelemetry {
        worker_id: worker_id.clone(),
        sampled_at,
        values,
        series,
    }))
}

#[cfg(test)]
mod id_tests {
    use super::*;

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1)); // unix epoch
        assert_eq!(civil_from_days(10_957), (2000, 1, 1)); // 30y + 7 leaps
        assert_eq!(civil_from_days(-1), (1969, 12, 31)); // day before epoch
        assert_eq!(civil_from_days(59), (1970, 3, 1)); // 1970 Feb has 28 days
    }

    #[test]
    fn exec_id_is_distinct_from_experiments_run_ids() {
        // EXEC-YYYYMMDD-HHMMSS-{5-digit sequence}, e.g. EXEC-20260718-160217-00397.
        let id = new_run_id(1_609_459_200.0, 397); // 2021-01-01T00:00:00Z, seq 397
        assert_eq!(id, "EXEC-20210101-000000-00397");
        // Deliberately NOT the experiments-server's `RUN-…` shape: an execution
        // attempt must never be mistaken for a logical research run.
        assert!(!id.starts_with("RUN-"));
        // Sequences past 99999 keep growing (lpad, no truncation).
        assert_eq!(new_run_id(1_609_459_200.0, 100_000), "EXEC-20210101-000000-100000");
        // Lexical order over the timestamp portion tracks wall-clock time.
        let later = new_run_id(1_609_545_600.0, 1); // +1 day
        let stamp = |s: &str| s[..EXEC_ID_PREFIX.len() + 8].to_owned();
        assert!(stamp(&later) > stamp(&id));
    }

    #[test]
    fn telemetry_folds_latest_values_and_full_series() {
        let w = WorkerId("gpu-1".into());
        assert!(worker_telemetry_from_samples(&w, vec![]).unwrap().is_none());

        let samples = vec![
            (100.0, r#"{"sys/gpu_util":0.4,"sys/cpu_util":0.1}"#.to_owned()),
            (103.0, r#"{"sys/gpu_util":0.9,"sys/cpu_util":0.2}"#.to_owned()),
        ];
        let t = worker_telemetry_from_samples(&w, samples).unwrap().unwrap();
        // Latest gauge values + timestamp come from the newest sample.
        assert_eq!(t.sampled_at, 103.0);
        assert_eq!(t.values["sys/gpu_util"], 0.9);
        // Series carries the whole window, ascending, for sparklines.
        let gpu = &t.series["sys/gpu_util"];
        assert_eq!(gpu.len(), 2);
        assert_eq!((gpu[0].ts, gpu[0].value), (100.0, 0.4));
        assert_eq!((gpu[1].ts, gpu[1].value), (103.0, 0.9));
    }
}
