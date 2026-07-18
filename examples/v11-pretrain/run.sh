#!/usr/bin/env bash
# Entrypoint for the v11 pretrain code unit. Colab already ships a CUDA torch,
# so we add only the light deps and put the vendored v11 model package on the
# path (it needs nothing but torch) — no ~2 GB torch reinstall.
set -euo pipefail

echo "[run] installing deps (reusing pre-installed torch)…"
pip install -q --disable-pip-version-check tokenizers datasets safetensors

export PYTHONPATH="$PWD/v11-core/src:${PYTHONPATH:-}"
echo "[run] starting v11 pretrain…"
exec python3 train.py
