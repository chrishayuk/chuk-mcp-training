"""The trainer-facing facade (spec §4.1).

``Introspector.from_env()`` is the only entry point a trainer needs. Without
``$CHUK_PROBE_PLAN`` it returns an inert object — zero overhead, torch never
imported by this module path. With a plan, ``pulse(model, step)`` returns a
context manager wrapping the training step on due steps and a null context
otherwise. A malformed plan is crash-isolated: warn on stderr, run inert —
introspection must never take a training run down (spec §2.2).
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import ContextManager, Iterable

from .constants import ENV_METRICS, ENV_PROBE_PLAN
from .plan import ProbePlan


class NullIntrospector:
    """Inert stand-in when no plan is present."""

    enabled = False

    def pulse(self, model: object, step: int, layers: object = None) -> ContextManager[None]:
        return contextlib.nullcontext()


class Introspector:
    enabled = True

    def __init__(self, plan: ProbePlan, metrics_path: str) -> None:
        # torch (via the pulse module) is imported only on this live path.
        from .emit import MetricsEmitter
        from .pulse import PulseState

        self.plan = plan
        self._state = PulseState(plan, MetricsEmitter(metrics_path))

    @staticmethod
    def from_env() -> "Introspector | NullIntrospector":
        plan_path = os.environ.get(ENV_PROBE_PLAN, "")
        metrics_path = os.environ.get(ENV_METRICS, "")
        if not plan_path or not metrics_path:
            return NullIntrospector()
        try:
            plan = ProbePlan.load(plan_path)
            return Introspector(plan, metrics_path)
        except Exception as err:  # crash isolation: inert beats dead trainer
            print(
                f"[chuk-introspect] disabled — could not load probe plan "
                f"{plan_path!r}: {err}",
                file=sys.stderr,
                flush=True,
            )
            return NullIntrospector()

    def pulse(
        self,
        model: object,
        step: int,
        layers: Iterable[object] | None = None,
    ) -> ContextManager[object]:
        """Wrap one training step; inert unless the step is due per the plan."""
        pulse = self.plan.pulse
        if pulse is None or step % pulse.every_steps != 0:
            return contextlib.nullcontext()
        from .pulse import PulseStep

        try:
            self._state.bind(model, layers)  # type: ignore[arg-type]
            return PulseStep(self._state, model, step)  # type: ignore[arg-type]
        except Exception:
            return contextlib.nullcontext()
