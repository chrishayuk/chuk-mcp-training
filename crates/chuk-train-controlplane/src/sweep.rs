//! Sweep fan-out (spec §5.2): pure expansion of a template over axes, the
//! inverse (which axis point a child got), and the cross-child aggregation
//! `sweep_status` reports — all store-free and unit-testable.

use std::collections::BTreeMap;

use chuk_train_proto::{MetricPoint, SweepAggregatePoint, TrainSpec, MAX_SWEEP_CHILDREN};
use serde_json::Value;

/// Axis paths are a closed set: the run's seed, or one override key.
const AXIS_SEED: &str = "seed";
const AXIS_OVERRIDES_PREFIX: &str = "overrides.";

/// Axis path → the value one child got.
pub type Assignment = BTreeMap<String, Value>;

/// Expand a template over axes into (assignment, concrete spec) children —
/// the cartesian product, deterministic (axes iterate in key order).
pub fn expand(
    template: &TrainSpec,
    axes: &BTreeMap<String, Vec<Value>>,
) -> Result<Vec<(Assignment, TrainSpec)>, String> {
    if axes.is_empty() {
        return Err("axes must not be empty — a sweep with no axes is submit_run".to_owned());
    }
    for (path, values) in axes {
        validate_path(path)?;
        if values.is_empty() {
            return Err(format!("axis {path:?} has no values"));
        }
    }
    let total: usize = axes.values().map(Vec::len).product();
    if total > MAX_SWEEP_CHILDREN {
        return Err(format!(
            "sweep would fan out {total} children (max {MAX_SWEEP_CHILDREN}) — split it up"
        ));
    }
    let mut combos: Vec<Assignment> = vec![BTreeMap::new()];
    for (path, values) in axes {
        combos = combos
            .into_iter()
            .flat_map(|combo| {
                values.iter().map(move |value| {
                    let mut next = combo.clone();
                    next.insert(path.clone(), value.clone());
                    next
                })
            })
            .collect();
    }
    combos
        .into_iter()
        .map(|assignment| {
            let mut spec = template.clone();
            for (path, value) in &assignment {
                apply(&mut spec, path, value)?;
            }
            Ok((assignment, spec))
        })
        .collect()
}

/// Which axis point a child spec carries — the inverse of [`expand`]'s apply,
/// so assignments need no extra storage.
pub fn assignment_of(spec: &TrainSpec, axes: &BTreeMap<String, Vec<Value>>) -> Assignment {
    axes.keys()
        .filter_map(|path| {
            let value = if path == AXIS_SEED {
                spec.seed.map(Value::from)
            } else {
                path.strip_prefix(AXIS_OVERRIDES_PREFIX)
                    .and_then(|key| spec.overrides.get(key).cloned())
            };
            value.map(|v| (path.clone(), v))
        })
        .collect()
}

fn validate_path(path: &str) -> Result<(), String> {
    if path == AXIS_SEED {
        return Ok(());
    }
    match path.strip_prefix(AXIS_OVERRIDES_PREFIX) {
        Some(key) if !key.is_empty() => Ok(()),
        _ => Err(format!(
            "unsupported axis path {path:?}: use \"{AXIS_SEED}\" or \"{AXIS_OVERRIDES_PREFIX}<key>\""
        )),
    }
}

fn apply(spec: &mut TrainSpec, path: &str, value: &Value) -> Result<(), String> {
    if path == AXIS_SEED {
        let seed = value
            .as_i64()
            .ok_or_else(|| format!("seed axis value {value} is not an integer"))?;
        spec.seed = Some(seed);
        return Ok(());
    }
    let key = path.strip_prefix(AXIS_OVERRIDES_PREFIX).expect("path validated");
    if spec.overrides.is_null() {
        spec.overrides = Value::Object(Default::default());
    }
    spec.overrides
        .as_object_mut()
        .ok_or_else(|| "template overrides is not a JSON object".to_owned())?
        .insert(key.to_owned(), value.clone());
    Ok(())
}

/// Cross-child aggregation at matched steps (spec §5.2): every step any child
/// reported, with `n` saying how many matched there.
pub fn aggregate(series: &[Vec<MetricPoint>]) -> Vec<SweepAggregatePoint> {
    let mut by_step: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for child in series {
        for point in child {
            by_step.entry(point.step).or_default().push(point.value);
        }
    }
    by_step
        .into_iter()
        .map(|(step, values)| {
            let n = values.len();
            let mean = values.iter().sum::<f64>() / n as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            SweepAggregatePoint {
                step,
                n: n as u32,
                mean,
                std: variance.sqrt(),
                min: values.iter().copied().fold(f64::INFINITY, f64::min),
                max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_train_proto::CodeRef;

    fn template() -> TrainSpec {
        TrainSpec {
            code: CodeRef { name: "unit".into(), sha: "abc".into() },
            entrypoint: "train".into(),
            config: None,
            overrides: serde_json::json!({ "lr": 0.1 }),
            artifacts_in: Vec::new(),
            data: None,
            checkpoint: Default::default(),
            seed: None,
            arch: None,
            timeout_s: 3600,
            links: Vec::new(),
        }
    }

    #[test]
    fn expands_the_cartesian_product_and_applies_axes() {
        let axes = BTreeMap::from([
            ("seed".to_owned(), vec![80.into(), 81.into()]),
            ("overrides.lr".to_owned(), vec![0.1.into(), 0.2.into()]),
        ]);
        let children = expand(&template(), &axes).expect("expand");
        assert_eq!(children.len(), 4);
        let (assignment, spec) = &children[0];
        assert_eq!(assignment["seed"], Value::from(80));
        assert_eq!(spec.seed, Some(80));
        assert_eq!(spec.overrides["lr"], assignment["overrides.lr"]);
        // The inverse recovers the assignment from the spec alone.
        assert_eq!(&assignment_of(spec, &axes), assignment);
    }

    #[test]
    fn rejects_bad_axes() {
        let t = template();
        for (axes, why) in [
            (BTreeMap::new(), "empty axes"),
            (BTreeMap::from([("seed".to_owned(), vec![])]), "empty values"),
            (BTreeMap::from([("config".to_owned(), vec![1.into()])]), "unsupported path"),
            (BTreeMap::from([("overrides.".to_owned(), vec![1.into()])]), "empty override key"),
            (BTreeMap::from([("seed".to_owned(), vec!["nope".into()])]), "non-integer seed"),
        ] {
            assert!(expand(&t, &axes).is_err(), "{why} should be rejected");
        }
        let huge: Vec<Value> = (0..1000).map(Value::from).collect();
        let axes = BTreeMap::from([("seed".to_owned(), huge)]);
        assert!(expand(&t, &axes).is_err(), "fan-out over the ceiling should be rejected");
    }

    #[test]
    fn aggregates_mean_std_range_at_matched_steps() {
        let series = vec![
            vec![MetricPoint { step: 0, value: 2.0 }, MetricPoint { step: 10, value: 1.0 }],
            vec![MetricPoint { step: 0, value: 4.0 }],
        ];
        let agg = aggregate(&series);
        assert_eq!(agg.len(), 2);
        assert_eq!(agg[0].step, 0);
        assert_eq!(agg[0].n, 2);
        assert_eq!(agg[0].mean, 3.0);
        assert_eq!(agg[0].std, 1.0);
        assert_eq!((agg[0].min, agg[0].max), (2.0, 4.0));
        // The unmatched step still reports, with n = 1 and zero spread.
        assert_eq!((agg[1].step, agg[1].n, agg[1].std), (10, 1, 0.0));
    }
}
