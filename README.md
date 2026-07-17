# chuk-mcp-training

MCP-controlled remote training harness for Colab and rented single GPUs.
Spec: `docs/specs/chuk-mcp-training-spec.md` (v0.4). This repo is at **M2**:
leased workers with a hard wall — drain at T-drain, provider-verified destroy
at T-0 (even with the agent hung), a reconcile loop that auto-kills orphaned
instances, an idle reaper, and a cost ledger — on top of M1 train runs (code
units, metrics, lineage checkpoints, resume) and the M0 join loop. Verified
locally against the **E0**, **E1**, and **E2** ladders (E2 via a mock provider
that launches real agent processes; the Vast driver awaits live-credential
verification).

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
  upload/fetch), `/` (dashboard stub), `/healthz`. Three adapter seams — the
  metadata store (`sqlite:path.db`, `redis:` reserved), the artifact blob
  store (`file:/path`, `s3:`/`r2:` reserved), and the provider registry
  (`mock` now, `vast` skeleton). Builds code units, mints run-scoped upload
  grants, ingests metrics + checkpoints, resumes, and runs the lease clock +
  reconcile loop that enforce the wall and kill orphans.
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

## First real run: Colab (E0)

Colab is the proving backend (spec §14) — units already paid for, no rental.
Full runbook: **[docs/E0-colab.md](docs/E0-colab.md)**. In short:

```bash
fly launch --no-deploy --copy-config -c deploy/fly.toml
fly volumes create chuk_train_data --size 1 -c deploy/fly.toml
fly secrets set -c deploy/fly.toml CHUK_TRAIN_API_TOKEN=$(openssl rand -hex 24) \
                                   CHUK_TRAIN_JOIN_TOKEN=$(openssl rand -hex 24)
fly deploy -c deploy/fly.toml --dockerfile deploy/Dockerfile
```

The Fly image builds both binaries and the **control plane serves the agent**
at `/agent/linux-x86_64` — so the Colab cell needs only the Fly URL + join
token. Fill those into `bootstrap/colab_cell.py`, paste it into a T4 notebook,
and the worker appears in `fleet`. Submit the E0 probe (`nvidia-smi` + a matmul
throughput run) with `submit_shell` and watch it stream via `tail_logs`.

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

## Leases + cleanup (E2)

```bash
# Provision a leased worker (mock launches a real agent process locally) …
provider_offers(provider="mock")
provision(provider="mock", lease_min=15, gpu="mock-t4", max_price_hr=0.10)
# … it runs jobs until the wall. extend_lease is the only path past it.
```

A lease is a hard wall (spec §3). At T-drain the control plane sends the agent
`drain` (and the agent self-drains on its own clock if the CP is dark); at T-0
it destroys the provider instance and **verifies it is gone by polling the
provider API — whether or not the agent ever responded**. A reconcile loop
lists real instances every interval and auto-kills any the registry does not
own (a hung agent, a dead tunnel, a wedged box). An idle reaper drains and
destroys a worker sitting idle past its threshold. Every lease and teardown
writes a cost record; `spend_status` reads the ledger.

Local E2 uses the `mock` provider, which launches the agent binary as real
processes, so provider-verified destroy is genuinely real (the OS process is
provably gone) — including the `kill -STOP` hung-agent case. The `vast` driver
is written to the same trait; a real 15-minute Vast lease is the live E2 test.

Set `CHUK_TRAIN_PROVIDERS`, `CHUK_TRAIN_AGENT_BIN` (mock), `CHUK_TRAIN_VAST_API_KEY`
(vast). `CHUK_TRAIN_RECONCILE_S`, `CHUK_TRAIN_IDLE_REAP_S`, and
`CHUK_TRAIN_DRAIN_WINDOW_MIN` override timings for fast local runs.

## Current limits (deliberate — see spec §14)

No packing or budgets/caps yet; one run in flight per worker; logs/metrics are
dropped while the control plane is dark. A dropped train run resumes from its
last uploaded checkpoint; a dropped shell run restarts. M3 adds the packing
scheduler; M4 adds budget caps + the one-page dashboard + watchdog gates.
