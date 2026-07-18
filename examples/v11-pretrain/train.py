#!/usr/bin/env python3
"""Contract-aware v11 TinyStories pretrain for the chuk-train harness (E1).

A thin wrapper: it reuses the v11 model (tiny_model_v11) and mirrors the loss
loop of tiny-model/model/v11-train/train_tinystories.py, but speaks the harness
script contract (spec §5.1) instead of that script's hardcoded, Mac-only setup:

  - device is CUDA-first (the original only checked MPS/CPU);
  - token budget / batch / lr come from $CHUK_CONFIG + $CHUK_OVERRIDES, so a
    short proving slice is a config change, not an edit;
  - metrics are appended as JSONL to $CHUK_METRICS;
  - checkpoints land in $CHUK_CKPT_DIR/step_<n>/ (model.safetensors with the
    tied-weight/buffer stripping the repo's convert_to_safetensors.py uses,
    optim.pt for resume, a partial meta.json with arch + tokenizer_hash, then a
    .ready marker);
  - $CHUK_RESUME_CKPT (set by the harness on a resumed slice) is loaded to
    continue from the last checkpoint's step.

The original train_tinystories.py is left untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from tiny_model_v11 import TinyModel, load_config

HERE = Path(__file__).resolve().parent
V11_DIR = HERE / "v11"                 # vendored: config.json
TOKENIZER_JSON = HERE / "tokenizer.json"

READY_MARKER = ".ready"
MODEL_FILE = "model.safetensors"
META_FILE = "meta.json"
# Keys safetensors cannot serialise: the tied lm_head aliases embed, and
# rope_freqs is a recomputed buffer (matches convert_to_safetensors.py).
STRIP_KEYS = {"lm_head.weight"}
STRIP_SUFFIX = "rope_freqs"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_run_config() -> dict:
    cfg: dict = {}
    config_path = env("CHUK_CONFIG")
    if config_path and Path(config_path).is_file():
        cfg = json.loads(Path(config_path).read_text())
    overrides = env("CHUK_OVERRIDES")
    if overrides:
        cfg.update(json.loads(overrides))
    seed = env("CHUK_SEED")
    if seed:
        cfg["seed"] = int(seed)
    return cfg


def pick_device(override: str = "") -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stripped_state(model: torch.nn.Module) -> dict:
    return {
        k: v.contiguous().cpu()
        for k, v in model.state_dict().items()
        if k not in STRIP_KEYS and not k.endswith(STRIP_SUFFIX)
    }


def stream_batches(dataset: str, tok: Tokenizer, seq_len: int, batch_size: int,
                   seed: int, device: torch.device):
    """Yield [batch_size, seq_len] token batches, streaming TinyStories."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
    buf: list[int] = []
    seqs: list[list[int]] = []
    for sample in ds:
        buf.extend(tok.encode(sample["text"]).ids)
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(seqs) == batch_size:
                yield torch.tensor(seqs, dtype=torch.long, device=device)
                seqs = []


def write_checkpoint(ckpt_dir: Path, step: int, model, arch: dict,
                     tokenizer_hash: str) -> None:
    # Weights only: the harness resume path reloads weights (fresh optimizer),
    # so we skip optim.pt — it would double a ~440 MB checkpoint for nothing.
    step_dir = ckpt_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    save_file(stripped_state(model), str(step_dir / MODEL_FILE))
    (step_dir / META_FILE).write_text(json.dumps({
        "step": step,
        "arch": "tinymodel-v11",
        "tokenizer_hash": tokenizer_hash,
        "config": arch,
    }))
    (step_dir / READY_MARKER).touch()  # signal completeness last


def resume_step(model) -> int:
    """If the harness set $CHUK_RESUME_CKPT, load weights and return the step."""
    resume = env("CHUK_RESUME_CKPT")
    if not resume:
        return 0
    resume_dir = Path(resume)
    model_path = resume_dir / MODEL_FILE
    if model_path.is_file():
        # strict=False: the tied lm_head + rope_freqs buffers are not in the file.
        model.load_state_dict(load_file(str(model_path)), strict=False)
    meta_path = resume_dir / META_FILE
    step = int(json.loads(meta_path.read_text()).get("step", 0)) if meta_path.is_file() else 0
    print(f"[resume] loaded checkpoint at step {step}", flush=True)
    return step


def main() -> None:
    cfg = load_run_config()
    total_tokens = int(cfg.get("total_tokens", 300_000))
    batch_size = int(cfg.get("batch_size", 8))
    lr = float(cfg.get("lr", 3e-4))
    seed = int(cfg.get("seed", 42))
    ckpt_every = int(cfg.get("checkpoint_every_steps", 50))
    log_every = int(cfg.get("log_every_steps", 10))
    dataset = str(cfg.get("dataset", "roneneldan/TinyStories"))

    device = pick_device(str(cfg.get("device", "")))
    torch.manual_seed(seed)

    model_cfg = load_config(V11_DIR)
    seq_len = int(cfg.get("seq_len", model_cfg.max_seq))
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    tokenizer_hash = sha256_file(TOKENIZER_JSON)
    vocab_size = tok.get_vocab_size()

    model = TinyModel(
        vocab_size=vocab_size, dim=model_cfg.dim, n_layers=model_cfg.n_layers,
        ffn_dim=model_cfg.ffn_dim, n_heads=model_cfg.n_heads,
        n_kv_heads=model_cfg.n_kv_heads, max_seq=model_cfg.max_seq,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    arch = model_cfg.to_dict()

    start_step = resume_step(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    metrics_path = Path(env("CHUK_METRICS", "metrics.jsonl"))
    ckpt_dir = Path(env("CHUK_CKPT_DIR", "ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_step = batch_size * seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    print(f"[v11-pretrain] device={device} params={n_params:,} vocab={vocab_size} "
          f"seq={seq_len} batch={batch_size} steps {start_step}->{total_steps}", flush=True)

    model.train()
    t0 = time.time()
    step = start_step
    with metrics_path.open("a") as metrics:
        for batch in stream_batches(dataset, tok, seq_len, batch_size, seed + step, device):
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab_size),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=0,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % log_every == 0 or step == total_steps:
                tok_s = (step - start_step) * tokens_per_step / max(time.time() - t0, 1e-6)
                metrics.write(json.dumps({
                    "step": step, "loss": round(loss.item(), 5),
                    "grad_norm": round(float(grad_norm), 4), "lr": lr,
                    "tok_s": round(tok_s, 1),
                }) + "\n")
                metrics.flush()
                print(f"  step {step}/{total_steps} loss={loss.item():.4f} {tok_s:.0f} tok/s", flush=True)

            if step % ckpt_every == 0 or step >= total_steps:
                write_checkpoint(ckpt_dir, step, model, arch, tokenizer_hash)
                print(f"  checkpoint at step {step}", flush=True)

            if step >= total_steps:
                break

    print(f"[v11-pretrain] done at step {step} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
