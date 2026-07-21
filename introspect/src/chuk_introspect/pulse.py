"""Tier-0 pulse capture (spec §4.2): read-only hooks on the live training step.

Forward hooks cover the activation-side stats; ``grad_norm`` uses
``register_post_accumulate_grad_hook`` (torch ≥ 2.1) — it fires per parameter
inside ``train_step``'s own backward, which is what lets the context manager
avoid needing a backward/step seam. Each firing overwrites the stored norm, so
the value emitted at exit is the post-accumulation gradient norm regardless of
micro-batch count (reading ``param.grad`` at exit is not viable — ``zero_grad``
may already have run inside ``train_step``).

Everything is computed on-device as 0-dim tensors; the only host transfer is
``.item()`` at emit time. Any capture failure sets ``introspect/probe_error``
and never propagates into the training step.
"""

from __future__ import annotations

import time
from typing import Iterable

import torch
from torch import nn

from . import constants as c
from .emit import MetricsEmitter
from .plan import ProbePlan, PulseMetric

# Duck-typed layer-container discovery, most specific first (the chuk-mlx
# ModelAccessor pattern; HF naming schemes).
_LAYER_CONTAINER_PATHS = (
    "model.layers",
    "transformer.h",
    "model.decoder.layers",
    "layers",
    "blocks",
)
_HEAD_ATTRS = ("lm_head", "head")

_EMA_ALPHA = 0.2  # rolling clean-step baseline smoothing (spec §13)
_EPS = 1e-9


def _resolve_path(root: nn.Module, dotted: str) -> object | None:
    obj: object = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def discover_layer_modules(model: nn.Module) -> list[nn.Module]:
    """Transformer blocks via common attribute paths; nn.Sequential children
    as a fallback so toy models (and the stub trainer) work unmodified."""
    for path in _LAYER_CONTAINER_PATHS:
        container = _resolve_path(model, path)
        if isinstance(container, (nn.ModuleList, nn.Sequential)):
            return list(container)
    if isinstance(model, nn.Sequential):
        return list(model)
    return []


def discover_head_module(model: nn.Module) -> nn.Module | None:
    for attr in _HEAD_ATTRS:
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            return head
    children = list(model.children())
    if children and isinstance(children[-1], nn.Linear):
        return children[-1]
    return None


def _first_tensor(output: object) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


class PulseState:
    """Per-run state that outlives individual pulse steps."""

    def __init__(self, plan: ProbePlan, emitter: MetricsEmitter) -> None:
        self.plan = plan
        self.emitter = emitter
        self.step_time_ema: float | None = None
        self.layer_modules: list[tuple[int, nn.Module]] | None = None
        self.head_module: nn.Module | None = None

    def bind(self, model: nn.Module, layers: Iterable[nn.Module] | None) -> None:
        if self.layer_modules is not None:
            return
        modules = list(layers) if layers is not None else discover_layer_modules(model)
        wanted = set(self.plan.pulse_layer_indices())
        self.layer_modules = [
            (i, m) for i, m in enumerate(modules) if not wanted or i in wanted
        ]
        self.head_module = discover_head_module(model)


class PulseStep:
    """Context manager for one due training step."""

    def __init__(self, state: PulseState, model: nn.Module, step: int) -> None:
        self._state = state
        self._model = model
        self._step = step
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._act: dict[int, torch.Tensor] = {}
        self._dead: dict[int, torch.Tensor] = {}
        self._grad_sq: dict[int, dict[int, torch.Tensor]] = {}
        self._entropy: torch.Tensor | None = None
        self._failed = False
        self._self_seconds = 0.0
        self._entered_at = 0.0

    # -- hook bodies (all _safe-wrapped) ------------------------------------

    def _capture_act(self, layer_idx: int, output: object) -> None:
        x = _first_tensor(output)
        if x is None:
            return
        x = x.detach()
        with torch.no_grad():
            self._act[layer_idx] = x.float().pow(2).mean().sqrt()
            flat = x.reshape(-1, x.shape[-1])
            self._dead[layer_idx] = flat.le(0).all(dim=0).float().mean()

    def _capture_entropy(self, output: object) -> None:
        logits = _first_tensor(output)
        if logits is None:
            return
        with torch.no_grad():
            logp = torch.log_softmax(logits.detach().float(), dim=-1)
            self._entropy = -(logp.exp() * logp).sum(dim=-1).mean()

    def _capture_grad(self, layer_idx: int, param_slot: int, param: torch.Tensor) -> None:
        if param.grad is None:
            return
        with torch.no_grad():
            # Overwrite per firing: the last firing holds the accumulated grad.
            self._grad_sq.setdefault(layer_idx, {})[param_slot] = (
                param.grad.detach().float().pow(2).sum()
            )

    def _safe(self, fn, *args) -> None:
        if self._failed:
            return
        t0 = time.perf_counter()
        try:
            fn(*args)
        except Exception:
            self._failed = True
        finally:
            self._self_seconds += time.perf_counter() - t0

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "PulseStep":
        t0 = time.perf_counter()
        try:
            self._register()
        except Exception:
            self._failed = True
            self._remove_handles()
        self._self_seconds += time.perf_counter() - t0
        self._entered_at = time.perf_counter()
        return self

    def _register(self) -> None:
        state = self._state
        metrics = set(state.plan.pulse.metrics if state.plan.pulse else [])
        assert state.layer_modules is not None
        for idx, module in state.layer_modules:
            if PulseMetric.ACT_NORM in metrics or PulseMetric.DEAD_FRAC in metrics:
                self._handles.append(
                    module.register_forward_hook(
                        lambda _m, _inp, out, i=idx: self._safe(self._capture_act, i, out)
                    )
                )
            if PulseMetric.GRAD_NORM in metrics:
                for slot, param in enumerate(module.parameters(recurse=True)):
                    if param.requires_grad:
                        self._handles.append(
                            param.register_post_accumulate_grad_hook(
                                lambda p, i=idx, s=slot: self._safe(self._capture_grad, i, s, p)
                            )
                        )
        if PulseMetric.LOGIT_ENTROPY in metrics and state.head_module is not None:
            self._handles.append(
                state.head_module.register_forward_hook(
                    lambda _m, _inp, out: self._safe(self._capture_entropy, out)
                )
            )

    def __exit__(self, exc_type, exc, tb) -> None:
        step_seconds = time.perf_counter() - self._entered_at
        t0 = time.perf_counter()
        self._remove_handles()
        try:
            # Never swallow the trainer's own exception; skip emission instead.
            if exc_type is None:
                self._emit(step_seconds, t0)
        except Exception:
            pass

    def _remove_handles(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()

    def _emit(self, step_seconds: float, exit_t0: float) -> None:
        state = self._state
        values: dict[str, float] = {}
        metrics = set(state.plan.pulse.metrics if state.plan.pulse else [])
        if not self._failed:
            if PulseMetric.ACT_NORM in metrics:
                for i, v in self._act.items():
                    values[c.layer_key(c.FAMILY_ACT_NORM, i)] = v.item()
            if PulseMetric.DEAD_FRAC in metrics:
                for i, v in self._dead.items():
                    values[c.layer_key(c.FAMILY_DEAD_FRAC, i)] = v.item()
            if PulseMetric.GRAD_NORM in metrics:
                for i, sqs in self._grad_sq.items():
                    total = torch.stack(list(sqs.values())).sum()
                    values[c.layer_key(c.FAMILY_GRAD_NORM, i)] = total.sqrt().item()
            if PulseMetric.LOGIT_ENTROPY in metrics and self._entropy is not None:
                values[c.global_key(c.FAMILY_LOGIT_ENTROPY)] = self._entropy.item()
        else:
            values[c.global_key(c.FAMILY_PROBE_ERROR)] = 1.0

        # Budget honesty (spec §13, I0 subset): our own wall-clock against a
        # rolling clean-step baseline. Hook bodies are timed individually via
        # _safe; forward-pass slowdown from hook presence is not separable here.
        self._self_seconds += time.perf_counter() - exit_t0
        ema = state.step_time_ema
        state.step_time_ema = (
            step_seconds if ema is None else (1 - _EMA_ALPHA) * ema + _EMA_ALPHA * step_seconds
        )
        baseline = max(state.step_time_ema - self._self_seconds, _EPS)
        values[c.global_key(c.FAMILY_OVERHEAD_PCT)] = 100.0 * self._self_seconds / baseline

        state.emitter.emit(self._step, values)
