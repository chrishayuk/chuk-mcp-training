"""Key-grammar parity with the Rust proto, and ProbePlan v3 validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chuk_introspect import ProbePlan
from chuk_introspect import constants as c


class TestKeyGrammar:
    """Expected strings mirror crates/chuk-train-proto/src/introspect.rs tests."""

    def test_per_layer(self):
        assert c.layer_key(c.FAMILY_ACT_NORM, 4) == "introspect/act_norm/L4"
        assert c.layer_key(c.FAMILY_DEAD_FRAC, 12) == "introspect/dead_frac/L12"

    def test_global(self):
        assert c.global_key(c.FAMILY_LOGIT_ENTROPY) == "introspect/logit_entropy"
        assert c.global_key(c.FAMILY_OVERHEAD_PCT) == "introspect/overhead_pct"

    def test_qualifier_between_family_and_layer(self):
        assert (
            c.metric_key("probe_acc", qualifier="content", layer=20)
            == "introspect/probe_acc/content/L20"
        )

    def test_corpus_suffix_last(self):
        assert (
            c.metric_key(c.FAMILY_ACT_NORM, layer=4, corpus="progA-v1")
            == "introspect/act_norm/L4@progA-v1"
        )


class TestProbePlan:
    def test_pulse_only_plan_is_valid(self):
        plan = ProbePlan.model_validate(
            {"version": 3, "model": {"d_model": 8, "n_layers": 2}}
        )
        assert plan.pulse is None and plan.snapshot is None

    def test_version_2_rejected(self):
        with pytest.raises(ValidationError):
            ProbePlan.model_validate(
                {"version": 2, "model": {"d_model": 8, "n_layers": 2}}
            )

    def test_snapshot_requires_corpus(self):
        with pytest.raises(ValidationError, match="requires a corpus"):
            ProbePlan.model_validate(
                {
                    "version": 3,
                    "model": {"d_model": 8, "n_layers": 2},
                    "snapshot": {
                        "corpus_subsample": 4,
                        "layers": [0],
                        "capture": ["hidden"],
                    },
                }
            )

    def test_layer_resolution(self):
        plan = ProbePlan.model_validate(
            {
                "version": 3,
                "model": {"d_model": 8, "n_layers": 3},
                "pulse": {"layers": "all"},
            }
        )
        assert plan.pulse_layer_indices() == [0, 1, 2]
        plan = ProbePlan.model_validate(
            {
                "version": 3,
                "model": {"d_model": 8, "n_layers": 3},
                "pulse": {"layers": [0, 2]},
            }
        )
        assert plan.pulse_layer_indices() == [0, 2]
