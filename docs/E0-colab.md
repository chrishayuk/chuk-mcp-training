# E0 · first real run on a Colab T4

Goal (spec §15 E0): the worker joins from a real Colab T4, `fleet` shows it with
the right GPU, a shell job runs `nvidia-smi` + a matmul throughput probe, and
logs stream live through `tail_logs`. Costs only Colab units.

The control plane is deployed to Fly and **serves the worker binary itself**, so
the Colab cell needs only the Fly URL + join token — nothing else to host.

## 1 · Deploy the control plane to Fly

From the repo root (you're already logged in as `fly auth whoami`):

```bash
# First time: create the app + volume (edit the app name in deploy/fly.toml first).
fly launch --no-deploy --copy-config -c deploy/fly.toml
fly volumes create chuk_train_data --size 1 -c deploy/fly.toml

# Secrets (the two tokens are the whole security model for M0/E0).
fly secrets set -c deploy/fly.toml \
  CHUK_TRAIN_API_TOKEN=$(openssl rand -hex 24) \
  CHUK_TRAIN_JOIN_TOKEN=$(openssl rand -hex 24)

fly deploy -c deploy/fly.toml --dockerfile deploy/Dockerfile
```

The Dockerfile builds the control plane (native) and the worker (static musl,
portable to Colab) on Fly's x86_64 builder. After deploy, point public URLs at
the real host:

```bash
APP=$(fly status -c deploy/fly.toml --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["Name"])')
fly secrets set -c deploy/fly.toml \
  CHUK_TRAIN_PUBLIC_URL=https://$APP.fly.dev \
  CHUK_TRAIN_AGENT_WS_URL=wss://$APP.fly.dev/ws/agent
```

Sanity check:

```bash
curl -s  https://$APP.fly.dev/healthz            # {"ok":true}
curl -s  https://$APP.fly.dev/agent/version      # {"version":"…"}
curl -sI https://$APP.fly.dev/agent/x86_64-unknown-linux-musl | grep -i content-length  # ~5.7 MB
```

Read the two secret values back for the next steps:

```bash
# You set these above; keep them handy. The join token goes in the Colab cell,
# the API token drives the MCP tools from the Mac.
```

## 2 · Join a Colab T4

The easy path: call the **`colab_cell`** MCP tool — the control plane returns a
ready-to-paste cell with its own URL and a **single-use `cj_` join token**
(spec §12) already filled in, bound to a fresh worker id. Paste it into one
cell of a T4 notebook (Runtime → Change runtime type → **T4 GPU**) and run it.

Manual fallback (`bootstrap/colab_cell.py`):

1. New Colab notebook → Runtime → Change runtime type → **T4 GPU**.
2. Paste `bootstrap/colab_cell.py` into one cell.
3. Fill in `CP_URL = "https://<app>.fly.dev"` and `JOIN_TOKEN = "<join token>"`
   (the shared `CHUK_TRAIN_JOIN_TOKEN` — dev fallback; `colab_cell`'s minted
   single-use token is preferred).
4. Run the cell. It downloads the worker and dials home; **leave it running**.

Within a second the worker appears in `fleet`.

## 3 · Drive E0 from the Mac (MCP)

```bash
cd mcp && uv sync
export CHUK_TRAIN_URL=https://<app>.fly.dev CHUK_TRAIN_API_TOKEN=<api token>
```

Then, via the MCP tools (mcp-cli / Claude / a quick `uv run python`):

- `fleet()` — the Colab worker, `state=connected`, `hardware.gpu = "Tesla T4"`.
- Submit the E0 probe as a shell job:

  ```
  submit_shell(name="e0-probe", command="""
    nvidia-smi -L
    python3 - <<'PY'
    import time, torch
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize()
    t0 = time.time(); n = 0
    while time.time() - t0 < 30:
        x = x @ x; n += 1
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"{n} matmuls in {dt:.1f}s  ~{2*4096**3*n/dt/1e12:.1f} TFLOP/s")
    PY
  """, timeout_s=120)
  ```

- `tail_logs(run_id)` — watch `nvidia-smi` output and the TFLOP/s line stream in.
- `run_status(run_id)` → `completed`, exit 0. **E0 green.**

## 4 · Tear down

Close the Colab tab (the worker exits; the worker goes `disconnected`). Nothing
to clean up — no lease, no rented instance. That's the whole point of proving on
Colab first.

## Notes

- The worker is a 5.5 MB static binary — no Python deps, no glibc concerns.
- If the TLS handshake ever fails on an unusual image, the fix is swapping the
  worker's rustls roots from `native-roots` to bundled `webpki-roots` (one line
  in `crates/chuk-compute-worker/Cargo.toml`); Colab's normal image has the certs.
- E1 (real training) reuses this exact setup: build a code unit from your v11
  trainer repo with `build_code_unit`, then `submit_run` against the Colab
  worker — checkpoints upload to `/data/artifacts` on Fly with full lineage.
