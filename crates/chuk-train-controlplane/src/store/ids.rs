//! Backend-agnostic store helpers shared by every `Store` adapter.
//!
//! The module's namesake is run-id / date formatting, but it also holds the
//! other pure, dialect-independent helpers both the SQLite and Postgres
//! adapters need: metric downsampling and the small serde/JSON converters.
//! Centralising them means the id/date logic (and the rest) lives in exactly
//! one place regardless of which backend is compiled in — no per-adapter drift.

use anyhow::Result;
use chuk_train_proto::{MetricPoint, UnixSeconds};

/// Run ids are `RUN-YYYYMMDD-HHMMSS-{5-digit sequence}` (UTC) — self-describing,
/// chronologically sortable, and **matching chuk-experiments-server's format**
/// (which reports/records these same runs). The sequence comes from a
/// store-backed monotonic counter, so it's stable and collision-free.
const RUN_ID_PREFIX: &str = "RUN-";
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
    let (y, m, d, hh, mm, ss) = utc_parts(at as i64);
    format!(
        "{RUN_ID_PREFIX}{y:04}{m:02}{d:02}-{hh:02}{mm:02}{ss:02}-{seq:0width$}",
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
    fn run_id_matches_experiments_server_format() {
        // RUN-YYYYMMDD-HHMMSS-{5-digit sequence}, e.g. RUN-20260718-160217-00397.
        let id = new_run_id(1_609_459_200.0, 397); // 2021-01-01T00:00:00Z, seq 397
        assert_eq!(id, "RUN-20210101-000000-00397");
        // Sequences past 99999 keep growing (lpad, no truncation).
        assert_eq!(new_run_id(1_609_459_200.0, 100_000), "RUN-20210101-000000-100000");
        // Lexical order over the timestamp portion tracks wall-clock time.
        let later = new_run_id(1_609_545_600.0, 1); // +1 day
        let stamp = |s: &str| s[..RUN_ID_PREFIX.len() + 8].to_owned();
        assert!(stamp(&later) > stamp(&id));
    }
}
