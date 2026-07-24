//! Gate expressions (spec §6/§8): a deliberately closed grammar — three forms,
//! parsed at registration (so a bad expression is a 400, not a silent no-op)
//! and re-parsed at evaluation. No general expression language: the operator
//! set is a whitelist, never interpolated from input.
//!
//!   isnan(last(<key>))            e.g. isnan(last(loss))
//!   no_improve(<key>, <N>min)     e.g. no_improve(loss, 120min)
//!   last(<key>) <op> <value>      e.g. last(grad_norm) > 1e3   (op: > >= < <=)
//!
//! Evaluation is a pure function over the metric history, so watchdog policy
//! is unit-testable without a store or a worker.

use chuk_train_proto::UnixSeconds;

pub use crate::store::MetricObservation;

const FN_ISNAN_PREFIX: &str = "isnan(last(";
const FN_ISNAN_SUFFIX: &str = "))";
const FN_NO_IMPROVE_PREFIX: &str = "no_improve(";
const FN_NO_IMPROVE_SUFFIX: &str = ")";
const FN_LAST_PREFIX: &str = "last(";
const FN_LAST_SUFFIX: &str = ")";
const MINUTES_SUFFIX: &str = "min";
const SECS_PER_MIN: f64 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
}

impl CmpOp {
    fn apply(self, left: f64, right: f64) -> bool {
        match self {
            Self::Gt => left > right,
            Self::Ge => left >= right,
            Self::Lt => left < right,
            Self::Le => left <= right,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Lt => "<",
            Self::Le => "<=",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GateExpr {
    /// `isnan(last(key))` — trips when the latest value is not finite.
    IsNan { key: String },
    /// `no_improve(key, Nmin)` — trips when the best (minimum) value in the
    /// trailing window is no better than the best before it.
    NoImprove { key: String, window_min: f64 },
    /// `last(key) op value` — trips when the latest value satisfies the
    /// comparison.
    Threshold { key: String, op: CmpOp, value: f64 },
}

impl GateExpr {
    /// The metric key this gate reads.
    pub fn key(&self) -> &str {
        match self {
            Self::IsNan { key } | Self::NoImprove { key, .. } | Self::Threshold { key, .. } => key,
        }
    }
}

/// The outcome of one evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub tripped: bool,
    /// The latest observed value, if the key has any observations.
    pub last_value: Option<f64>,
    /// Human-readable one-liner of why (shown in check_gates + run events).
    pub detail: String,
}

fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '/' | '.'))
}

pub fn parse(expr: &str) -> Result<GateExpr, String> {
    let expr = expr.trim();

    if let Some(rest) = expr.strip_prefix(FN_ISNAN_PREFIX) {
        let key = rest
            .strip_suffix(FN_ISNAN_SUFFIX)
            .ok_or_else(|| format!("malformed isnan gate {expr:?}: expected isnan(last(<key>))"))?
            .trim();
        if !valid_key(key) {
            return Err(format!("invalid metric key {key:?}"));
        }
        return Ok(GateExpr::IsNan { key: key.to_owned() });
    }

    if let Some(rest) = expr.strip_prefix(FN_NO_IMPROVE_PREFIX) {
        let inner = rest.strip_suffix(FN_NO_IMPROVE_SUFFIX).ok_or_else(|| {
            format!("malformed no_improve gate {expr:?}: expected no_improve(<key>, <N>min)")
        })?;
        let (key, window) = inner
            .split_once(',')
            .ok_or_else(|| "no_improve needs a window: no_improve(<key>, <N>min)".to_owned())?;
        let key = key.trim();
        if !valid_key(key) {
            return Err(format!("invalid metric key {key:?}"));
        }
        let window_min: f64 = window
            .trim()
            .strip_suffix(MINUTES_SUFFIX)
            .ok_or_else(|| format!("window {window:?} must end in {MINUTES_SUFFIX:?}"))?
            .trim()
            .parse()
            .map_err(|_| format!("window {window:?} is not a number of minutes"))?;
        if !(window_min.is_finite() && window_min > 0.0) {
            return Err("window must be a positive number of minutes".to_owned());
        }
        return Ok(GateExpr::NoImprove { key: key.to_owned(), window_min });
    }

    if let Some(rest) = expr.strip_prefix(FN_LAST_PREFIX) {
        let (key, comparison) = rest.split_once(FN_LAST_SUFFIX).ok_or_else(|| {
            format!("malformed threshold gate {expr:?}: expected last(<key>) <op> <value>")
        })?;
        let key = key.trim();
        if !valid_key(key) {
            return Err(format!("invalid metric key {key:?}"));
        }
        let comparison = comparison.trim();
        // Longest-match first so ">=" never parses as ">" + "=…".
        let (op, value) = if let Some(v) = comparison.strip_prefix(">=") {
            (CmpOp::Ge, v)
        } else if let Some(v) = comparison.strip_prefix("<=") {
            (CmpOp::Le, v)
        } else if let Some(v) = comparison.strip_prefix('>') {
            (CmpOp::Gt, v)
        } else if let Some(v) = comparison.strip_prefix('<') {
            (CmpOp::Lt, v)
        } else {
            return Err(format!("unsupported operator in {expr:?}: use > >= < <="));
        };
        let value: f64 = value
            .trim()
            .parse()
            .map_err(|_| format!("threshold {value:?} is not a number"))?;
        return Ok(GateExpr::Threshold { key: key.to_owned(), op, value });
    }

    Err(format!(
        "unrecognised gate expression {expr:?}: use isnan(last(<key>)), \
         no_improve(<key>, <N>min), or last(<key>) <op> <value>"
    ))
}

pub fn evaluate(expr: &GateExpr, history: &[MetricObservation], now: UnixSeconds) -> Verdict {
    let last = history.last().copied();
    match expr {
        GateExpr::IsNan { key } => match last {
            Some(obs) if !obs.value.is_finite() => Verdict {
                tripped: true,
                last_value: Some(obs.value),
                detail: format!("last({key}) is not finite"),
            },
            Some(obs) => Verdict {
                tripped: false,
                last_value: Some(obs.value),
                detail: format!("last({key}) = {} is finite", obs.value),
            },
            None => no_data(key),
        },
        GateExpr::Threshold { key, op, value } => match last {
            Some(obs) => {
                let tripped = op.apply(obs.value, *value);
                Verdict {
                    tripped,
                    last_value: Some(obs.value),
                    detail: format!(
                        "last({key}) = {} {} {value} is {tripped}",
                        obs.value,
                        op.as_str()
                    ),
                }
            }
            None => no_data(key),
        },
        GateExpr::NoImprove { key, window_min } => {
            let cutoff = now - window_min * SECS_PER_MIN;
            let best_before = history
                .iter()
                .filter(|o| o.ts < cutoff)
                .map(|o| o.value)
                .fold(f64::INFINITY, f64::min);
            let best_within = history
                .iter()
                .filter(|o| o.ts >= cutoff)
                .map(|o| o.value)
                .fold(f64::INFINITY, f64::min);
            if best_before.is_infinite() {
                // Not enough history to have a "before the window" baseline.
                return Verdict {
                    tripped: false,
                    last_value: last.map(|o| o.value),
                    detail: format!("{key} has < {window_min}min of history"),
                };
            }
            let tripped = best_within >= best_before;
            Verdict {
                tripped,
                last_value: last.map(|o| o.value),
                detail: format!(
                    "best {key} in trailing {window_min}min = {best_within} vs {best_before} before"
                ),
            }
        }
    }
}

fn no_data(key: &str) -> Verdict {
    Verdict {
        tripped: false,
        last_value: None,
        detail: format!("no observations of {key} yet"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(points: &[(f64, f64)]) -> Vec<MetricObservation> {
        points.iter().map(|&(ts, value)| MetricObservation { ts, value }).collect()
    }

    #[test]
    fn parses_the_three_forms_and_rejects_everything_else() {
        assert_eq!(parse("isnan(last(loss))").unwrap(), GateExpr::IsNan { key: "loss".into() });
        assert_eq!(
            parse(" no_improve( loss , 120min ) ").unwrap(),
            GateExpr::NoImprove { key: "loss".into(), window_min: 120.0 }
        );
        assert_eq!(
            parse("last(grad_norm) > 1e3").unwrap(),
            GateExpr::Threshold { key: "grad_norm".into(), op: CmpOp::Gt, value: 1000.0 }
        );
        assert_eq!(
            parse("last(val/acc) <= 0.5").unwrap(),
            GateExpr::Threshold { key: "val/acc".into(), op: CmpOp::Le, value: 0.5 }
        );
        // introspect/* keys (chuk-introspect spec §6) parse with zero grammar changes.
        assert_eq!(
            parse("last(introspect/dead_frac/L12) > 0.5").unwrap(),
            GateExpr::Threshold {
                key: "introspect/dead_frac/L12".into(),
                op: CmpOp::Gt,
                value: 0.5
            }
        );
        for bad in [
            "loss > 3",                    // bare key: not a form
            "last(loss) == 3",             // unsupported operator
            "no_improve(loss)",            // missing window
            "no_improve(loss, 120s)",      // wrong unit
            "no_improve(loss, -5min)",     // negative window
            "isnan(loss)",                 // isnan takes last(...)
            "last(loss; DROP TABLE) > 1",  // invalid key characters
            "mean(loss) > 3",              // unknown function
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn isnan_trips_on_non_finite_only() {
        let expr = parse("isnan(last(loss))").unwrap();
        assert!(!evaluate(&expr, &obs(&[(0.0, 2.5)]), 10.0).tripped);
        assert!(evaluate(&expr, &obs(&[(0.0, 2.5), (5.0, f64::NAN)]), 10.0).tripped);
        assert!(evaluate(&expr, &obs(&[(5.0, f64::INFINITY)]), 10.0).tripped);
        assert!(!evaluate(&expr, &[], 10.0).tripped); // no data ≠ NaN
    }

    #[test]
    fn threshold_compares_the_latest_value() {
        let expr = parse("last(grad_norm) > 1e3").unwrap();
        let history = obs(&[(0.0, 2000.0), (5.0, 900.0)]);
        // Only the latest matters — the earlier spike doesn't trip it.
        assert!(!evaluate(&expr, &history, 10.0).tripped);
        assert!(evaluate(&expr, &obs(&[(5.0, 1500.0)]), 10.0).tripped);
    }

    #[test]
    fn no_improve_needs_a_baseline_then_trips_on_stall() {
        let expr = parse("no_improve(loss, 2min)").unwrap();
        // All history inside the window: no baseline yet, never trips.
        assert!(!evaluate(&expr, &obs(&[(100.0, 3.0), (110.0, 3.0)]), 120.0).tripped);
        // Baseline 2.0 before the window; the window's best is 2.5: stalled.
        let stalled = obs(&[(0.0, 2.0), (130.0, 2.5), (170.0, 2.6)]);
        assert!(evaluate(&expr, &stalled, 200.0).tripped);
        // The window's best 1.5 beats the 2.0 baseline: improving.
        let improving = obs(&[(0.0, 2.0), (130.0, 1.5), (170.0, 1.8)]);
        assert!(!evaluate(&expr, &improving, 200.0).tripped);
    }

    #[test]
    fn threshold_supports_ge_lt_le_not_just_gt() {
        // threshold_compares_the_latest_value only exercises `>`, and the
        // three-forms test only parses `<=` without evaluating it — so
        // CmpOp::apply/as_str's Ge/Lt/Le arms are otherwise never hit.
        let ge = parse("last(loss) >= 3").unwrap();
        assert_eq!(ge, GateExpr::Threshold { key: "loss".into(), op: CmpOp::Ge, value: 3.0 });
        let verdict = evaluate(&ge, &obs(&[(0.0, 3.0)]), 10.0);
        assert!(verdict.tripped);
        assert!(verdict.detail.contains(">="));

        let lt = parse("last(loss) < 3").unwrap();
        assert_eq!(lt, GateExpr::Threshold { key: "loss".into(), op: CmpOp::Lt, value: 3.0 });
        let verdict = evaluate(&lt, &obs(&[(0.0, 2.0)]), 10.0);
        assert!(verdict.tripped);
        assert!(!verdict.detail.contains("<="), "expected bare `<`, got {:?}", verdict.detail);

        let le = parse("last(loss) <= 3").unwrap();
        let verdict = evaluate(&le, &obs(&[(0.0, 3.0)]), 10.0);
        assert!(verdict.tripped);
        assert!(verdict.detail.contains("<="));

        // The flip side of each: comparison false, verdict not tripped.
        assert!(!evaluate(&ge, &obs(&[(0.0, 2.0)]), 10.0).tripped);
        assert!(!evaluate(&lt, &obs(&[(0.0, 3.0)]), 10.0).tripped);
    }

    #[test]
    fn threshold_evaluates_to_no_data_verdict_when_history_is_empty() {
        // isnan_trips_on_non_finite_only covers the IsNan arm's empty-history
        // path; the Threshold arm has its own `None => no_data(key)` branch.
        let expr = parse("last(loss) > 3").unwrap();
        let verdict = evaluate(&expr, &[], 10.0);
        assert!(!verdict.tripped);
        assert_eq!(verdict.last_value, None);
        assert!(verdict.detail.contains("no observations of loss"));
    }

    #[test]
    fn isnan_rejects_an_invalid_metric_key() {
        assert!(parse("isnan(last(bad key!))").is_err());
    }

    #[test]
    fn no_improve_rejects_a_missing_close_paren_and_an_invalid_key() {
        // Starts with the no_improve prefix but never closes: the
        // strip_suffix(")") lookup fails before the window is even parsed.
        assert!(parse("no_improve(loss, 120min").is_err());
        // Closes fine, but the key itself has disallowed characters.
        assert!(parse("no_improve(bad key!, 120min)").is_err());
    }

    #[test]
    fn threshold_rejects_an_expression_missing_last_close_paren() {
        // Starts with the last( prefix but there's no `)` anywhere in the
        // rest of the string, so split_once(")") fails.
        assert!(parse("last(loss > 3").is_err());
    }
}
