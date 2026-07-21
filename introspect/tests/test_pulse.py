"""Pulse-tier tests (CPU torch). These pin the five I0 library guarantees that
apply before snapshots exist: inert-without-plan, correct key grammar,
post-accumulation grad norms, dead-fraction signal, crash isolation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from chuk_introspect import Introspector, NullIntrospector
from chuk_introspect import constants as c
from chuk_introspect.pulse import PulseStep


def make_plan(tmp_path: Path, **overrides) -> Path:
    plan = {
        "version": 3,
        "model": {"d_model": 8, "n_layers": 4},
        "pulse": {
            "every_steps": 1,
            "metrics": ["act_norm", "grad_norm", "dead_frac", "logit_entropy"],
            "layers": "all",
        },
    }
    plan.update(overrides)
    path = tmp_path / "probe_plan.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture
def live_env(tmp_path, monkeypatch) -> Path:
    """Env with a pulse-only plan; returns the metrics JSONL path."""
    metrics = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(c.ENV_PROBE_PLAN, str(make_plan(tmp_path)))
    monkeypatch.setenv(c.ENV_METRICS, str(metrics))
    return metrics


def tiny_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))


def read_records(metrics: Path) -> list[dict]:
    if not metrics.exists():
        return []
    return [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]


def run_steps(intr, model, metrics_or_none, n_steps: int = 2) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for step in range(1, n_steps + 1):
        with intr.pulse(model, step):
            out = model(torch.randn(4, 8))
            loss = out.pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()


class TestInert:
    def test_no_env_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.delenv(c.ENV_PROBE_PLAN, raising=False)
        monkeypatch.setenv(c.ENV_METRICS, str(tmp_path / "metrics.jsonl"))
        intr = Introspector.from_env()
        assert isinstance(intr, NullIntrospector)
        model = tiny_model()
        run_steps(intr, model, None)
        assert not (tmp_path / "metrics.jsonl").exists()

    def test_malformed_plan_is_inert_not_fatal(self, tmp_path, monkeypatch):
        bad = tmp_path / "plan.json"
        bad.write_text("{not json")
        monkeypatch.setenv(c.ENV_PROBE_PLAN, str(bad))
        monkeypatch.setenv(c.ENV_METRICS, str(tmp_path / "metrics.jsonl"))
        assert isinstance(Introspector.from_env(), NullIntrospector)


class TestPulseEmission:
    def test_keys_and_shape(self, live_env):
        intr = Introspector.from_env()
        assert intr.enabled
        model = tiny_model()
        run_steps(intr, model, live_env, n_steps=3)

        records = read_records(live_env)
        assert [r[c.METRIC_STEP_KEY] for r in records] == [1, 2, 3]
        rec = records[0]
        # Sequential fallback: 3 child modules -> layers L0..L2.
        for i in range(3):
            assert c.layer_key(c.FAMILY_ACT_NORM, i) in rec
            assert c.layer_key(c.FAMILY_DEAD_FRAC, i) in rec
        # Grad norms only for parametered layers (ReLU at L1 has none).
        assert c.layer_key(c.FAMILY_GRAD_NORM, 0) in rec
        assert c.layer_key(c.FAMILY_GRAD_NORM, 1) not in rec
        assert c.layer_key(c.FAMILY_GRAD_NORM, 2) in rec
        # Head = trailing Linear -> model-global logit entropy, no layer segment.
        assert c.global_key(c.FAMILY_LOGIT_ENTROPY) in rec
        assert c.global_key(c.FAMILY_OVERHEAD_PCT) in rec
        assert all(isinstance(v, (int, float)) for v in rec.values())

    def test_every_steps_gates_emission(self, tmp_path, monkeypatch):
        metrics = tmp_path / "metrics.jsonl"
        plan = make_plan(
            tmp_path,
            pulse={"every_steps": 2, "metrics": ["act_norm"], "layers": "all"},
        )
        monkeypatch.setenv(c.ENV_PROBE_PLAN, str(plan))
        monkeypatch.setenv(c.ENV_METRICS, str(metrics))
        intr = Introspector.from_env()
        run_steps(intr, tiny_model(), metrics, n_steps=4)
        assert [r[c.METRIC_STEP_KEY] for r in read_records(metrics)] == [2, 4]


class TestGradNorm:
    def test_post_accumulation_value(self, live_env):
        """Two micro-batch backwards before the step: the emitted grad_norm must
        equal the norm of the fully ACCUMULATED grad, not the first firing's."""
        intr = Introspector.from_env()
        model = nn.Sequential(nn.Linear(8, 8))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(1)
        batches = [torch.randn(4, 8), torch.randn(4, 8)]

        with intr.pulse(model, 1):
            for b in batches:
                model(b).pow(2).mean().backward()
            expected = math.sqrt(
                sum(p.grad.float().pow(2).sum().item() for p in model.parameters())
            )
            opt.step()
            opt.zero_grad()

        rec = read_records(live_env)[0]
        assert rec[c.layer_key(c.FAMILY_GRAD_NORM, 0)] == pytest.approx(expected, rel=1e-5)


class TestDeadFrac:
    def test_fully_dead_relu_reads_one(self, live_env):
        intr = Introspector.from_env()
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.fill_(-1.0)  # pre-activations all negative -> ReLU dead
        with intr.pulse(model, 1):
            model(torch.randn(4, 8)).sum().backward()

        rec = read_records(live_env)[0]
        assert rec[c.layer_key(c.FAMILY_DEAD_FRAC, 1)] == pytest.approx(1.0)

    def test_healthy_relu_reads_below_one(self, live_env):
        intr = Introspector.from_env()
        model = tiny_model()
        with intr.pulse(model, 1):
            model(torch.randn(64, 8)).sum().backward()
        rec = read_records(live_env)[0]
        assert rec[c.layer_key(c.FAMILY_DEAD_FRAC, 1)] < 1.0


class TestCrashIsolation:
    def test_hook_failure_survives_and_flags(self, live_env, monkeypatch):
        def boom(self, layer_idx, output):
            raise RuntimeError("injected probe failure")

        monkeypatch.setattr(PulseStep, "_capture_act", boom)
        intr = Introspector.from_env()
        model = tiny_model()
        # The training step must complete despite every act hook failing.
        run_steps(intr, model, live_env, n_steps=1)

        rec = read_records(live_env)[0]
        assert rec[c.global_key(c.FAMILY_PROBE_ERROR)] == 1.0
        assert c.layer_key(c.FAMILY_ACT_NORM, 0) not in rec

    def test_trainer_exception_propagates_unswallowed(self, live_env):
        intr = Introspector.from_env()
        model = tiny_model()
        with pytest.raises(ValueError, match="trainer bug"):
            with intr.pulse(model, 1):
                raise ValueError("trainer bug")
        assert read_records(live_env) == []  # no emission for a failed step
