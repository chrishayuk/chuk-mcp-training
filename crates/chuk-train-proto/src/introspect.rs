//! Introspection metric-key grammar (chuk-introspect spec §5.1).
//!
//! The `introspect/` namespace is domain vocabulary, so it lives here — never
//! in the wire/worker crates (which are lexically domain-free; the worker
//! ships these keys as opaque strings). Key segment order is **normative**
//! and parsed positionally on both sides of the seam:
//!
//! ```text
//! introspect/<family>[/<qualifier>]/L{i}[@<corpus>]
//! ```
//!
//! - `family` comes from the closed set of constants below;
//! - `qualifier` exists only where the family declares one (snapshot-era
//!   families like `probe_acc`/`lens_rank`/`jspace_align_z` — I1/I2);
//! - `L{i}` is the layer segment, absent only on model-global keys;
//! - `@<corpus>` is the corpus suffix (spec §3 rule 4), always last, absent
//!   on the lab-standard corpus.
//!
//! Mirrored (hand-maintained, "must match" comment) in
//! `introspect/src/chuk_introspect/constants.py`.

/// Namespace prefix for all introspection metrics (spec §5.1).
pub const INTROSPECT_METRIC_PREFIX: &str = "introspect/";

/// Layer-segment prefix: layer 4 renders as `L4`.
pub const LAYER_SEGMENT_PREFIX: &str = "L";

/// Corpus-suffix separator (spec §3 rule 4); lab-standard keys carry none.
pub const CORPUS_SUFFIX_SEPARATOR: &str = "@";

// -- pulse families (Tier 0, per-layer unless noted) ------------------------

/// Per-layer activation norm on the live training batch.
pub const FAMILY_ACT_NORM: &str = "act_norm";
/// Per-layer post-accumulation gradient norm.
pub const FAMILY_GRAD_NORM: &str = "grad_norm";
/// Per-layer dead-neuron fraction.
pub const FAMILY_DEAD_FRAC: &str = "dead_frac";
/// Model-global logit entropy (no layer segment).
pub const FAMILY_LOGIT_ENTROPY: &str = "logit_entropy";

// -- library health families (model-global, no layer segment) ---------------

/// Introspection overhead vs the rolling clean-step baseline (spec §13).
pub const FAMILY_OVERHEAD_PCT: &str = "overhead_pct";
/// A probe failed and was skipped; the training step survived (spec §4.1).
pub const FAMILY_PROBE_ERROR: &str = "probe_error";

/// Build a metric key per the normative grammar. `family` is one of the
/// `FAMILY_*` constants; `qualifier`/`layer`/`corpus` per the family's shape.
pub fn metric_key(
    family: &str,
    qualifier: Option<&str>,
    layer: Option<u32>,
    corpus: Option<&str>,
) -> String {
    let mut key = format!("{INTROSPECT_METRIC_PREFIX}{family}");
    if let Some(q) = qualifier {
        key.push('/');
        key.push_str(q);
    }
    if let Some(i) = layer {
        key.push('/');
        key.push_str(LAYER_SEGMENT_PREFIX);
        key.push_str(&i.to_string());
    }
    if let Some(c) = corpus {
        key.push_str(CORPUS_SUFFIX_SEPARATOR);
        key.push_str(c);
    }
    key
}

/// Per-layer key on the lab-standard corpus, e.g. `introspect/act_norm/L4`.
pub fn layer_key(family: &str, layer: u32) -> String {
    metric_key(family, None, Some(layer), None)
}

/// Model-global key on the lab-standard corpus, e.g. `introspect/logit_entropy`.
pub fn global_key(family: &str) -> String {
    metric_key(family, None, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_layer_key_matches_spec_grammar() {
        assert_eq!(layer_key(FAMILY_ACT_NORM, 4), "introspect/act_norm/L4");
        assert_eq!(layer_key(FAMILY_DEAD_FRAC, 12), "introspect/dead_frac/L12");
    }

    #[test]
    fn global_key_has_no_layer_segment() {
        assert_eq!(global_key(FAMILY_LOGIT_ENTROPY), "introspect/logit_entropy");
        assert_eq!(global_key(FAMILY_OVERHEAD_PCT), "introspect/overhead_pct");
    }

    #[test]
    fn qualifier_sits_between_family_and_layer() {
        // Snapshot-era shape (I1/I2), pinned by the grammar today.
        assert_eq!(
            metric_key("probe_acc", Some("content"), Some(20), None),
            "introspect/probe_acc/content/L20"
        );
    }

    #[test]
    fn corpus_suffix_is_always_last() {
        assert_eq!(
            metric_key(FAMILY_ACT_NORM, None, Some(4), Some("progA-v1")),
            "introspect/act_norm/L4@progA-v1"
        );
    }
}
