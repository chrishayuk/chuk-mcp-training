#!/usr/bin/env python3
"""A stub trainer that honours the chuk-train script contract (spec §5.1).

It runs no real model — it exists to exercise the harness end-to-end (E1 at
"rig" scale) and to bring the operator dashboard to life with realistic
telemetry: it reads $CHUK_CONFIG + $CHUK_OVERRIDES, appends metrics JSONL to
$CHUK_METRICS (loss, lr, grad_norm, tokens_per_s, tflops), writes checkpoints
(model + partial meta.json + a `.ready` marker) under $CHUK_CKPT_DIR, and
resumes from $CHUK_RESUME_CKPT when set.

Adopting the contract in a real trainer is these ~5 touch-points, nothing more.

I0 addition (chuk-introspect spec §15 EI0): when torch + chuk_introspect are
importable and $CHUK_PROBE_PLAN is set (run.sh sets it — CP delivery is I1),
each step also trains a tiny real torch model inside `Introspector.pulse`, so
real `introspect/*` pulse metrics stream beside the synthetic telemetry. With
`poison_dead_relu` in the config the model's ReLU is born dead — the watchdog
gate `last(introspect/dead_frac/L1) > 0.5` fires. Without torch the stub runs
exactly as before.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Vendored deps (chuk_introspect is copied in at unit-build time — see README).
sys.path.insert(0, str(Path(__file__).parent / "vendor"))

# A fixed fingerprint the harness carries into checkpoint lineage; lazarus would
# verify it against the local tokenizer at load time (spec §10).
TOKENIZER_HASH = "tok-stub-0001"
READY_MARKER = ".ready"

# Telemetry shape (stand-in numbers, but plausible for a small model on a T4).
LOSS_HI, LOSS_LO = 2.34, 1.36
PEAK_LR = 3.0e-4
BASE_TOKENS_PER_S = 1850.0
BASE_TFLOPS = 4.1


def load_config() -> dict:
    config: dict = {}
    config_path = os.environ.get("CHUK_CONFIG", "")
    if config_path and Path(config_path).is_file():
        config = json.loads(Path(config_path).read_text())
    overrides = os.environ.get("CHUK_OVERRIDES", "")
    if overrides:
        config.update(json.loads(overrides))
    return config


def resume_step() -> int:
    """Steps already completed, read from the resume checkpoint's meta.json."""
    resume_dir = os.environ.get("CHUK_RESUME_CKPT", "")
    if not resume_dir:
        return 0
    meta_path = Path(resume_dir) / "meta.json"
    if meta_path.is_file():
        return int(json.loads(meta_path.read_text()).get("step", 0))
    return 0


def telemetry(step: int, total: int, rng: random.Random) -> dict:
    """Plausible per-step metrics: loss decays with noise, lr warms then cosine-
    decays, grad_norm settles, throughput jitters around a baseline."""
    frac = step / max(1, total)
    loss = LOSS_LO + (LOSS_HI - LOSS_LO) * math.exp(-3.0 * frac)
    loss += rng.gauss(0, 0.03 * (1 - 0.6 * frac))
    warm = min(1.0, frac / 0.08)
    lr = PEAK_LR * warm * (0.5 * (1 + math.cos(math.pi * max(0.0, frac - 0.08) / 0.92)))
    grad = 0.25 + 0.7 * math.exp(-2.2 * frac) + rng.gauss(0, 0.02)
    return {
        "step": step,
        "loss": round(max(loss, 0.01), 4),
        "lr": round(max(lr, 1e-6), 7),
        "grad_norm": round(max(grad, 0.01), 4),
        "tokens_per_s": round(BASE_TOKENS_PER_S + rng.gauss(0, 60), 1),
        "tflops": round(BASE_TFLOPS + rng.gauss(0, 0.15), 3),
    }


def build_pulse(config: dict):
    """Optional real-model pulse rig: (introspector, model, step_fn) or None.

    Degrades gracefully — no torch, no chuk_introspect, or no probe plan all
    mean the stub behaves exactly as it did before I0.
    """
    try:
        import torch
        from torch import nn

        from chuk_introspect import Introspector
    except ImportError as err:
        print(f"[stub-trainer] introspection off ({err})", flush=True)
        return None

    intr = Introspector.from_env()
    if not intr.enabled:
        print("[stub-trainer] introspection off (no $CHUK_PROBE_PLAN)", flush=True)
        return None

    torch.manual_seed(int(os.environ.get("CHUK_SEED", "0") or 0))
    # Shape must match probe_plan.json's model block: 3 modules -> L0..L2.
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
    if config.get("poison_dead_relu"):
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.fill_(-1.0)  # every pre-activation negative -> dead ReLU
        print("[stub-trainer] POISONED: ReLU born dead (EI0 gate proof)", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    def step_fn() -> None:
        out = model(torch.randn(8, 16))
        out.pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

    print("[stub-trainer] introspection ON (pulse tier)", flush=True)
    return intr, model, step_fn


def write_checkpoint(ckpt_dir: Path, step: int, arch: str) -> None:
    step_dir = ckpt_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    # Stand-in weights: distinct bytes per step so hashes differ.
    (step_dir / "model.safetensors").write_bytes(f"stub-weights@step={step}".encode())
    # Partial sidecar: the facts only the trainer knows. The harness fills in
    # code / config_hash / parent / run_id / slices before upload.
    (step_dir / "meta.json").write_text(json.dumps({
        "step": step,
        "arch": arch,
        "tokenizer_hash": TOKENIZER_HASH,
    }))
    # Signal completeness last, so the agent never uploads a partial checkpoint.
    (step_dir / READY_MARKER).touch()


def main() -> None:
    config = load_config()
    total_steps = int(config.get("total_steps", 20))
    checkpoint_every = int(config.get("checkpoint_every", 5))
    step_delay_s = float(config.get("step_delay_s", 0.4))
    arch = str(config.get("arch", "stub-net-v0"))
    seed = os.environ.get("CHUK_SEED", "0")
    rng = random.Random(int(seed) if seed.lstrip("-").isdigit() else 0)

    metrics = Path(os.environ["CHUK_METRICS"])
    ckpt_dir = Path(os.environ["CHUK_CKPT_DIR"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start = resume_step()
    print(f"[stub-trainer] device=cuda:0 · {arch} · seq256 · bs64 · seed={seed}", flush=True)
    print(f"[stub-trainer] resume_step={start} · total_steps={total_steps} · ckpt_every={checkpoint_every}", flush=True)

    pulse_rig = build_pulse(config)

    with metrics.open("a") as m:
        for step in range(start + 1, total_steps + 1):
            if pulse_rig is not None:
                intr, model, step_fn = pulse_rig
                with intr.pulse(model, step):
                    step_fn()
            t = telemetry(step, total_steps, rng)
            m.write(json.dumps(t) + "\n")
            m.flush()
            pct = int(step / total_steps * 100)
            print(
                f"[stub-trainer] step {step}/{total_steps} | loss {t['loss']:.3f} | "
                f"lr {t['lr']:.2e} | grad {t['grad_norm']:.3f} | "
                f"{t['tokens_per_s']:.0f} tok/s | {t['tflops']:.1f} TFLOP/s | {pct}%",
                flush=True,
            )
            if step % checkpoint_every == 0:
                write_checkpoint(ckpt_dir, step, arch)
                print(f"[stub-trainer] checkpoint step_{step} written → uploading", flush=True)
            time.sleep(step_delay_s)

    print("[stub-trainer] done", flush=True)


if __name__ == "__main__":
    main()
