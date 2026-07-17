# chuk-mcp-training

MCP-controlled remote training harness for Colab and rented single GPUs.
Spec: `docs/specs/chuk-mcp-training-spec.md` (v0.4). This repo is at **M0**:
join loop, shell runs, log streaming, fleet — proven by experiment **E0**.

**Stack:** Rust control plane + Rust worker agent; the MCP tool surface is
Python on `chuk-mcp-server`, a thin client over the control plane's REST API.
House rules: async native, no magic strings, no magic numbers, pydantic native
on the Python side — shared names/numbers live in `chuk-train-proto`
(Rust) and are mirrored in `chuk_train_mcp/constants.py` (Python).

## Layout

- `crates/chuk-train-proto` — shared wire protocol, domain types, constants.
  The single source of truth for everything that crosses a process boundary.
- `crates/chuk-train-cp` — control plane daemon (axum + tokio + sqlx):
  `/ws/agent` (worker websocket), `/api/*` (bearer-auth REST), `/` (dashboard
  stub), `/healthz`. Storage is behind an adapter trait — SQLite now
  (`sqlite:path.db`), `redis:` reserved for M2+ off-box state.
- `crates/chuk-train-agent` — worker agent binary: dials out, registers
  hardware, heartbeats, executes assigned shell runs, streams logs, reconnects
  with backoff. Builds to a static musl binary workers download and exec.
- `mcp/` — `chuk-train-mcp` Python package: the MCP tools (`fleet`,
  `submit_shell`, `list_runs`, `run_status`, `tail_logs`, `run_events`).
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

## M0 limits (deliberate — see spec §14)

Shell runs only; one run in flight per worker; no lease walls, artifact store,
packing, budgets, or log buffering while the control plane is dark. If the
websocket drops mid-run the agent kills the child and the control plane
requeues the run. M1 adds code units, checkpoints + lineage, and resume; M2
adds leases + provable cleanup.
