# chuk-mcp-training

MCP-controlled remote training harness for Colab and rented single GPUs.
Spec: `docs/specs/chuk-mcp-training-spec.md` (v0.4). This repo is at **M1**:
train runs end-to-end — code units, metrics tailing, lineage-complete
checkpoints, and resume-after-kill — on top of the M0 join loop / shell runs /
log streaming / fleet. Verified locally against the **E0** and **E1** ladders.

**Stack:** Rust control plane + Rust worker agent; the MCP tool surface is
Python on `chuk-mcp-server`, a thin client over the control plane's REST API.
House rules: async native, no magic strings, no magic numbers, pydantic native
on the Python side — shared names/numbers live in `chuk-train-proto`
(Rust) and are mirrored in `chuk_train_mcp/constants.py` (Python).

## Layout

- `crates/chuk-train-proto` — shared wire protocol, domain types, constants,
  and the store key layout. The single source of truth for everything that
  crosses a process boundary.
- `crates/chuk-train-cp` — control plane daemon (axum + tokio + sqlx):
  `/ws/agent` (worker websocket), `/api/*` (bearer-auth REST + grant-auth
  upload/fetch), `/` (dashboard stub), `/healthz`. Two adapter seams — the
  metadata store (`sqlite:path.db`, `redis:` reserved) and the artifact blob
  store (`file:/path`, `s3:`/`r2:` reserved). Builds code units, mints
  run-scoped upload grants, ingests metrics + checkpoints, and resumes.
- `crates/chuk-train-agent` — worker agent binary: dials out, registers
  hardware, heartbeats, runs shell + train jobs, streams logs/metrics, fetches
  code units (cached by sha), uploads lineage-complete checkpoints, resumes,
  reconnects with backoff. Builds to a static musl binary workers download.
- `mcp/` — `chuk-train-mcp` Python package: `fleet`, `submit_shell`,
  `list_runs`, `run_status`, `tail_logs`, `run_events`, plus M1 tools
  `build_code_unit`, `submit_run`, `run_metrics`, `list_checkpoints`,
  `pin_checkpoint`, `artifact_url`.
- `examples/stub-trainer/` — a contract-honouring stub trainer code unit; the
  E1 fixture (reads `$CHUK_CONFIG`, writes metrics + checkpoints, resumes).
- `bootstrap/colab_cell.py` — the one Colab cell that joins a T4 as a worker (E0).
- `deploy/` — Dockerfile + fly.toml (`auto_stop_machines = "off"`,
  volume-backed SQLite).

## Run locally

```bash
export CHUK_TRAIN_API_TOKEN=$(openssl rand -hex 24)
export CHUK_TRAIN_JOIN_TOKEN=$(openssl rand -hex 24)

cargo run -p chuk-train-cp                                   # control plane :8700
cargo run -p chuk-train-agent -- \
  --url ws://127.0.0.1:8700/ws/agent --token $CHUK_TRAIN_JOIN_TOKEN

cd mcp && uv sync && CHUK_TRAIN_URL=http://127.0.0.1:8700 \
  uv run chuk-train-mcp                                      # MCP (stdio)
```

Dashboard: <http://127.0.0.1:8700/> (paste the API token into the token box).

## Deploy (Fly)

```bash
fly launch --no-deploy --copy-config -c deploy/fly.toml
fly volumes create chuk_train_data --size 1
fly secrets set CHUK_TRAIN_API_TOKEN=$(openssl rand -hex 24) \
                CHUK_TRAIN_JOIN_TOKEN=$(openssl rand -hex 24)
fly deploy -c deploy/fly.toml --dockerfile deploy/Dockerfile
```

Build the agent for workers (linux x86_64, static):

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release -p chuk-train-agent --target x86_64-unknown-linux-musl
```

Host the binary anywhere reachable, fill in `bootstrap/colab_cell.py`, paste it
into a Colab notebook → the T4 appears in `fleet` and on the dashboard. That's E0.

## Train a run (E1)

```bash
# Build a code unit from a repo/commit (or a local path, as here) …
build_code_unit(repo="examples/stub-trainer", name="stub-trainer")   # → code sha
# … then queue a train run against it.
submit_run(name="e1", code_name="stub-trainer", code_sha="<sha>",
           entrypoint="train", config="configs/stub.json", seed=81)
```

Checkpoints upload to the artifact store with lineage-complete `meta.json`
(code, config hash, tokenizer hash, parent, run id, seed, slices). Kill the
worker mid-run and the control plane requeues it; a fresh worker resumes from
the last uploaded checkpoint and the slice list records both halves. Follow it
with `run_metrics`, `list_checkpoints`, and `run_events`; pull a checkpoint to
the Mac with `artifact_url`.

## The script contract (spec §5.1)

A train entrypoint reads a handful of env vars — about five lines to adopt:

| var | meaning |
|-----|---------|
| `$CHUK_CONFIG` | absolute path to the resolved config file |
| `$CHUK_OVERRIDES` | JSON object of config overrides |
| `$CHUK_METRICS` | append one JSON object per line (step + numeric fields) |
| `$CHUK_CKPT_DIR` | write `step_<n>/` dirs; `touch step_<n>/.ready` when complete |
| `$CHUK_RESUME_CKPT` | a checkpoint dir to resume from (empty on a fresh run) |
| `$CHUK_RUN_ID`, `$CHUK_SEED` | provenance |

See `examples/stub-trainer/train.py` for a minimal working example.

## Current limits (deliberate — see spec §14)

No lease walls, packing, budgets, or provider provisioning yet; one run in
flight per worker; logs/metrics are dropped while the control plane is dark.
A dropped train run resumes from its last uploaded checkpoint; a dropped shell
run restarts. M2 adds leases + provable cleanup; M3 packing; M4 budgets.
