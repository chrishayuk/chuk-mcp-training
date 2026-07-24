# chuk-mcp-training

MCP-controlled remote training harness for Colab and rented single GPUs.
Spec: `docs/specs/chuk-mcp-training-spec.md` (v0.8) · status + plan: `ROADMAP.md`.

Milestones **M0–M2** are built and **M4 (budgets + watchdogs)** and the first
slice of **M5 (sweeps)** are code-complete; the control plane is deployed on Fly
(`chuk-mcp-training.fly.dev`), **stateless on Neon** (serverless Postgres), with
checkpoints in Cloudflare R2 and cold storage on Google Drive, and the MCP tool
surface hosted at **`chuk-train-mcp.fly.dev/mcp`**. Proven on real hardware:
**E0** (agent joins a Colab T4, `nvidia-smi` + matmul probe, live logs)
and **E1** (a 115M-param model trains on the T4, metrics stream, lineage-complete
checkpoints upload directly to R2, and the resume test passes — bounce the Colab
cell mid-run and it resumes from the R2 checkpoint). **M2** (leases, drain,
provider-verified destroy, reconcile / orphan-kill, ledger) is verified locally
via a mock provider; live Vast E2 and the E4 budget/watchdog proving run pending.
Newest proof (2026-07-22): the **full registry loop ran on a live Colab T4** — an
experiments-server run dispatched with `submit_run_from_experiment`, joined via a
dashboard-minted single-use token, trained with **chuk-introspect pulse metrics**
(10 `introspect/*` keys at ~3% overhead — spec `docs/specs/chuk-introspect-spec.md`),
uploaded lineage checkpoints to R2, and mirrored results + artifacts back into the
registry.

Beyond the milestones, the harness now has a **full Google-authed operator
dashboard** — a tabbed per-run view (a light **Overview** that drills into
dedicated **Training** / **Introspection** / **Logs** / **Events** / **System**
screens — the System tab graphing live GPU/VRAM/temp/power/CPU/memory from the
worker running the run, the Introspection tab charting live `introspect/*` pulse
metrics) plus fleet/runs filters, pagination, per-worker live GPU-util, Money/
Budgets and Gates cards, per-sweep views, and a **Join** screen that mints a
paste-ready Colab cell (single-use token) and watches for the worker to dial home; **real-time
host telemetry** (connected workers stream `sys/*` every few seconds — chuk-compute
M4); a complete **Drive cold-archive tier** (completed runs auto-tier
their final checkpoint + logs to Drive, R2 lifecycle expires the hot copies,
retrieval resolves R2 *or* Drive, with `archive_run`/`archive_runs`/`archive_status`
MCP tools); and **RBAC** — users + roles (sysadmin › admin › write › read) in a
team, with **self-service scoped MCP API keys** (any signed-in user mints their
own ≤ their role; admins manage the team); and an **optional
chuk-experiments-server reporting mirror** (gated off by default — when
configured, run lifecycle + checkpoints-as-artifacts + final-metrics-as-results
mirror into the experiments registry through a durable, retrying outbox — a
transient failure is retried, never silently dropped — with each run attributed
to whichever user's own linked chuk-experiments-server key submitted it, falling
back to the shared server-wide key otherwise).

**Cost governance + watchdogs (spec §8, M4):** **budget caps** (`global` or
`provider:<name>`, per calendar-month or all-time) are enforced on
`provision`/`extend_lease` — projected spend (realised ledger + committed live
leases + the candidate) over any cap refuses, and a post-price breach destroys
the fresh instance rather than keep it billing; a **`confirm_cost` pre-flight**
refuses expensive submissions with the estimate shown (sweeps show the
*multiplied* total); and **watchdog gates** — `isnan(last(loss))`,
`no_improve(loss, 120min)`, `last(grad_norm) > 1e3` — evaluate on metric ingest
and can checkpoint-then-stop a run. **Sweeps (spec §5.2, M5):** `submit_sweep`
fans one template over `seed`/`overrides.*` axes into child runs the scheduler
holds to a per-sweep concurrency; `sweep_status` reports cross-child
mean/std/range at matched steps. **Join security (spec §12):** provisioning and
the Colab cell mint **single-use `cj_` join tokens** bound to one worker id —
consumed on first join, only ever readmitting their own identity afterwards.
Next M-work: the E2/E4 proving runs, then M3 packing. See `ROADMAP.md`.

**Runs standalone.** Every external tier is gated and optional — no R2 (falls
back to `file:`), no Drive (archive tier off), no Google auth (API-token box),
and no chuk-experiments-server (reporting mirror off). The harness's own Neon/
SQLite store + queue are always the source of truth; nothing outside is a hard
dependency.

**chuk-datasets integration (2026-07-24, cross-repo with the sibling
[chuk-datasets-server](../chuk-datasets-server)).** A `TrainSpec` can carry a
`data:` block naming a content-addressed, fingerprinted dataset + batch plan;
`assemble_train_job` resolves it dispatch-time against chuk-datasets-server
and stages each manifest shard as a hash-verified input, the same contract
code units already use. Unlike the other integrations above this one is
**not** best-effort once opted into: a run that declares `data:` with no
`CHUK_DATASETS_URL`/`CHUK_DATASETS_API_KEY` configured fails to dispatch
rather than silently training on nothing. `CheckpointMeta.dataset_sha`/
`plan_sha` record the resolved (not trainer-claimed) identity. The worker
links `chuk-datasets-client` directly and caches fetched shards locally by
hash. Gate G1 is proven live locally (real control planes + a real worker,
zero mocks, checkpoints stamped with the resolved identity); the deployed
chuk-datasets.fly.dev + chuk-mcp-training.fly.dev run is the remaining
proof — see `ROADMAP.md`.

**chuk-compute substrate (M1–M4 done).** Underneath the training-first control
plane the rig is being factored into a **compute fabric**: a permanently
compute-generic worker + wire protocol (spec `docs/specs/chuk-compute-spec.md`).
- **M1** — the worker (`chuk-compute-worker`) is a domain-free executor that runs
  generic *jobs* (stage inputs → command → outputs) over `chuk-compute-wire`; the
  control plane translates a run into a job and interprets results back into
  checkpoints. Same behaviour as before (proven — including the E1 resume/slice path).
- **M2** — the CP distributes the worker per target: `/agent/{triple}` + `.sha256`
  + `/agent/version` + a one-shot `/install.sh` (uname → download → verify → join),
  so Colab, Vast, and a Mac all bootstrap the same way (CI builds the target matrix).
- **M3 (persistent worker class)** — long-lived, revocable, per-worker tokens
  (`cw_`) bound to a stable id, and **survive-disconnect**: a persistent worker
  (e.g. a Mac you own) keeps its job running across a dropped connection — even the
  control plane restarting — and replays buffered events on reconnect; no lease ⇒
  never torn down. A version-mismatched persistent worker **self-updates** in
  place (download → verify → atomic replace → re-exec; leased workers just exit).
  All three parts proven end-to-end.
- **M4 (host telemetry)** — a sampler streams `sys/*` (GPU via `nvidia-smi`, CPU/
  memory via `sysinfo`) over the existing Metric channel, out-of-band; the CP keeps
  a pruned per-worker window and the dashboard renders live gauges + per-metric
  graphs. macmon (Apple-Silicon GPU) + OOM/thermal gates are the follow-ups.

The substrate will grow to run evals, benchmarks, cells, agents, and RL loops
(spec §10–§11) while the control plane stays training-first.

**Stack:** Rust control plane + Rust worker (`chuk-compute-worker`); the MCP tool
surface is Python on `chuk-mcp-server`, a thin client over the control plane's
REST API, served two ways from one tool registration: **stdio** (local,
single-user, token from the env) and **hosted HTTP** (`chuk-train-mcp.fly.dev/mcp`,
its own scale-to-zero Fly app) — a **zero-credential proxy** that forwards each
caller's own bearer per request, so the control plane enforces RBAC per caller
and the proxy holds no key. House rules: async native, no magic strings, no
magic numbers, pydantic native on the Python side, clean/decoupled modules,
≥90% test coverage per file.
The worker↔control-plane protocol lives in `chuk-compute-wire` (generic, serde-
only); training domain types + constants live in `chuk-train-proto` (Rust,
control-plane side) and are mirrored in `chuk_train_mcp/constants.py` (Python).

## Layout

- `crates/chuk-compute-wire` — the compute-generic worker↔control-plane protocol
  (chuk-compute-spec): the `Hello` handshake, the generic `Job` model (inputs →
  command → outputs, batch-vs-service), capabilities, worker classes, telemetry
  config, and the blob-transfer contract. Serde-only, no domain vocabulary (a
  lexical guard enforces it). The worker depends on nothing else in the workspace.
- `crates/chuk-train-proto` — control-plane domain types, constants, and the
  store key layout (run/train/checkpoint specs, RBAC types, REST payloads). The
  source of truth for the training domain + the CP↔MCP REST surface.
- `crates/chuk-train-controlplane` — control plane daemon (axum + tokio + sqlx):
  `/ws/agent` (worker websocket), `/api/*` (role-authed REST + grant-auth
  upload/fetch), `/` (the operator dashboard), `/healthz`. Adapter seams — the
  metadata store (`postgres:`/`postgresql:` → Neon, or `sqlite:path.db` local),
  the artifact blob store (`r2:`/`s3:`, or `file:/path`), and the provider
  registry (`mock`, `vast` skeleton). Modules include `store` (SQLite + Postgres
  adapters, each split into per-domain files behind 10 cohesive sub-traits —
  `WorkerStore`/`RunStore`/…), `archive` (Drive tiering + backstop sweep), `drive`
  (Drive v3 client), `apikey` (RBAC keys + bearer→role resolution), `datasets`
  (the chuk-datasets-server resolve client a `TrainSpec.data:` block dispatches
  through — gated on `CHUK_DATASETS_URL`/`CHUK_DATASETS_API_KEY`), and `dash` (the
  dashboard — a thin Rust handler inlining `dash/{index.html,dash.css,app.js}`). Builds code units, mints run-scoped upload grants, ingests metrics
  + checkpoints, resumes, auto-archives completed runs, and runs the lease clock
  + reconcile loop that enforce the wall and kill orphans.
- `crates/chuk-compute-worker` — the join-anywhere worker binary (depends only on
  `chuk-compute-wire`): dials out, `Hello` handshake, heartbeats, and runs generic
  jobs — stages inputs (fetch/unpack into a sandbox), runs the command under
  supervision, streams logs/metrics, collects outputs (uploaded as artifacts),
  reconnects with backoff. Domain-free: the training-ness (code units, resume,
  checkpoint lineage) is expressed by the control plane in the job it sends.
  Builds to a static musl binary workers download.
- `introspect/` — the `chuk_introspect` Python capture library (pulse tier:
  ProbePlan → per-layer torch hooks → `introspect/*` metrics on the existing
  metric channel; vendored into code units at build time — see
  `examples/stub-trainer/README.md` and `docs/specs/chuk-introspect-spec.md`).
- `mcp/` — `chuk-train-mcp` Python package: 34 tools across identity (`whoami`),
  fleet (`fleet`, `worker_telemetry`), runs (`submit_shell`, `build_code_unit`,
  `submit_run`, `submit_run_from_experiment`, `list_runs`, `run_status`,
  `stop_run`, `resume_run`, `tail_logs`, `run_events`, `run_metrics`), sweeps
  (`submit_sweep`, `sweep_status`), gates (`register_gate`, `check_gates`),
  checkpoints (`list_checkpoints`, `pin_checkpoint`, `artifact_url`), leases
  (`provider_offers`, `provision`, `lease_status`, `extend_lease`, `teardown`,
  `colab_cell`), cost (`spend_status`, `set_budget`, `list_budgets`,
  `delete_budget`), and archive (`archive_run`, `archive_runs`,
  `archive_status`). Every list result carries `count` + a self-describing
  message when empty. Runs as stdio (`chuk-train-mcp`) or HTTP
  (`chuk-train-mcp --http`, the hosted per-caller bearer proxy).
- `examples/stub-trainer/` — a contract-honouring stub trainer code unit (the E1
  fixture + demo trainer): reads `$CHUK_CONFIG`, emits rich metrics
  (loss/lr/grad_norm/tokens_per_s/tflops) + logs, writes checkpoints, resumes.
- `scripts/demo.sh` — one-command local demo: a CP + mock workers running the
  stub-trainer so the dashboard fills with live data (isolated from prod).
- `scripts/authorize-drive.py` — one-time `drive.file` offline auth → refresh token.
- `bootstrap/colab_cell.py` — the one Colab cell that joins a T4 as a worker (E0).
- `deploy/` — Dockerfile + fly.toml for the control plane (`auto_stop_machines =
  "off"`); stateless on Neon (the store URL is a Fly secret; the `/data` volume is
  legacy). `deploy/mcp/` — the hosted MCP endpoint's image + fly.toml
  (`chuk-train-mcp.fly.dev`, scale-to-zero, zero secrets — pure bearer
  passthrough). Deploys are **CI/CD** — a push to `main` that passes clippy +
  tests (incl. the Postgres adapter against a CI `postgres` service) + the worker
  target-matrix auto-deploys both apps to Fly (`.github/workflows/ci.yml`).

## Run locally

```bash
export CHUK_TRAIN_API_TOKEN=$(openssl rand -hex 24)
export CHUK_TRAIN_JOIN_TOKEN=$(openssl rand -hex 24)

cargo run -p chuk-train-controlplane                                   # control plane :8700
cargo run -p chuk-compute-worker -- \
  --url ws://127.0.0.1:8700/ws/agent --token $CHUK_TRAIN_JOIN_TOKEN

cd mcp && uv sync && CHUK_TRAIN_URL=http://127.0.0.1:8700 \
  uv run chuk-train-mcp                                      # MCP (stdio)
# or serve MCP over HTTP at /mcp (per-caller bearer passthrough):
#   uv run chuk-train-mcp --http --port 8710
```

Or the one-command demo (mock workers running the stub-trainer, so the dashboard
fills with live runs):

```bash
./scripts/demo.sh          # then open http://127.0.0.1:8700 ; Ctrl-C to stop
```

Dashboard: <http://127.0.0.1:8700/> — Google sign-in when configured, else the
API-token box (local dev). Set `CHUK_TRAIN_STORE=postgresql://…` (Neon's pooled
endpoint) to run against Postgres instead of local SQLite.

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

The Fly image builds both binaries and the **control plane serves the worker per
target** at `/agent/{triple}` (+ `.sha256`), with a one-shot **`/install.sh`** that
detects the box's target, downloads + checksum-verifies the matching worker, and
joins — so the Colab cell needs only the Fly URL + join token. Fill those into
`bootstrap/colab_cell.py` (a one-line `curl … | sh` over `/install.sh`), paste it
into a T4 notebook, and the worker appears in `fleet`. Submit the E0 probe
(`nvidia-smi` + a matmul throughput run) with `submit_shell` and watch it stream
via `tail_logs`.

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
writes a cost record; `spend_status(period)` reads the ledger and shows
cap/headroom where budgets are set. `set_budget("provider:vast", 50)` caps a
provider (or `"global"` everything): provision and extend refuse on projected
breach — and a provision whose *actual* price breaches is destroyed rather than
kept billing. Each provision mints a **single-use `cj_` join token** bound to
its worker id, so the bootstrap credential can't enrol anything else.

Local E2 uses the `mock` provider, which launches the agent binary as real
processes, so provider-verified destroy is genuinely real (the OS process is
provably gone) — including the `kill -STOP` hung-agent case. The `vast` driver
is written to the same trait; a real 15-minute Vast lease is the live E2 test.

Set `CHUK_TRAIN_PROVIDERS`, `CHUK_TRAIN_AGENT_BIN` (mock), `CHUK_TRAIN_VAST_API_KEY`
(vast). `CHUK_TRAIN_RECONCILE_S`, `CHUK_TRAIN_IDLE_REAP_S`, and
`CHUK_TRAIN_DRAIN_WINDOW_MIN` override timings for fast local runs.

## Dashboard, access & archive

The operator dashboard (`/`, spec §9) is served by the control plane and gated
behind **Google sign-in** (session cookie; the API-token box is the local-dev
fallback). It has an overview (fleet/runs/spend/health with filters + pagination)
and a per-run view: live loss curve + metric toggles, streamed logs, config,
checkpoints with full metadata + download, events, and out-links (W&B /
experiments-server). Executions get sortable `EXEC-YYYYMMDD-HHMMSS-NNNNN` ids (a
store-backed 5-digit sequence) — deliberately distinct from chuk-experiments-server's
`RUN-…` *logical run* ids: ours names an execution attempt, theirs the research run.
A run may carry an `experiment_ref` pointing at the logical `RUN-…` it realises.

**Access (RBAC):** users have a role — sysadmin › admin › write › read — in a
team (single default team; multi-team scaffolded). `read`/`write`/`admin` mirror
chuk-experiments-server; `sysadmin` is the extra platform-owner tier (the legacy
master token resolves to it). The **Access** screen is **self-service**: any
signed-in user mints, lists, and revokes **their own** MCP API keys, always
scoped **at or below their own role** (shown once, hashed at rest). Admins
additionally manage team members and see every key in the team. Roles are
enforced per endpoint: read = view, write = submit/manage runs, admin = archive +
manage access. Give MCP clients a scoped key (`CHUK_TRAIN_API_TOKEN=ck_…`) instead
of the master token.

**Archive tier (spec §11.5):** Drive is the durable, browsable home; R2 is a hot
cache. When a run completes, its final checkpoint + logs + metrics tier to Google
Drive automatically (a background loop is both the prompt archiver and the
idempotent backstop); the final is promoted to `ckpt-final/` on R2, and R2
lifecycle expires the hot copies (`ckpt-hot/` 1d, `ckpt-final/` 30d). A stable
per-checkpoint URL resolves R2-or-Drive. Trigger/inspect via `archive_run`,
`archive_runs`, `archive_status`. (The two R2 lifecycle rules are live on the
bucket; the CP's boot-time apply merges with existing rules rather than
replacing them, and will self-manage once its token gains lifecycle permission.)

## Current limits (deliberate — see spec §14)

No packing scheduler or `submit_batch` yet (M3 — waiting on rented-GPU
pressure); one run in flight per worker; logs/metrics are dropped while the
control plane is dark. A dropped train run resumes from its last uploaded
checkpoint; a dropped shell run restarts. Label-scope budgets are unenforced
(leases don't carry labels); and the E2 (live Vast) + E4 (budget/watchdog)
proving runs are still to be run on real hardware.
