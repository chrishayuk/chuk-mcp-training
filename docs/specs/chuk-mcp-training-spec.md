# chuk-mcp-training — Specification v0.7

**MCP-controlled remote training harness for Colab and rented single GPUs**
Control plane + worker agent in **Rust** (axum · tokio · sqlx) · MCP tool surface in
**Python** on `chuk-mcp-server` (thin client over the control plane's REST API) ·
Workers: Google Colab, Vast.ai, Lambda Cloud

v0.7 changes: the control plane is **stateless on Neon** (serverless Postgres) via the
`Store` seam (SQLite kept for local dev + tests). The **archive tier (§11.5) is complete** —
completed runs auto-tier their final checkpoint + logs + metrics to Drive, R2 lifecycle
expires the hot copies, and a stable per-checkpoint URL resolves R2-or-Drive
(`archive_run`/`archive_runs`/`archive_status`). **RBAC (§12)**: users + roles
(sysadmin › admin › write › read) in a team, with scoped MCP **API keys** managed from the
dashboard and enforced per endpoint. The dashboard (§9) gains a per-run view, fleet/runs
filters, pagination, and an admin **Access** screen. Checkpoints move to top-level
`ckpt-hot/` / `ckpt-final/` prefixes so R2 lifecycle can target them.
v0.6 changes: the one-page **dashboard (§9) is built** — served by the Rust control plane
and gated behind app-level **Google sign-in** (email allowlist, HMAC-signed session
cookie) while MCP/agents stay on bearer tokens (§12). Storage gains a **cold-archive tier
(§11.5)**: R2 stays hot, and completed runs' checkpoints/logs tier to **Google Drive**
(`drive.file`, resumable upload) as durable, browsable cold storage.
v0.5 changes (implementation reality): the control plane and agent are **Rust**, not a
Python/FastAPI process — only the MCP tool surface is Python. The dashboard (§9) is
served by the Rust control plane (static HTML + fetch/SSE), not FastAPI+htmx. The
artifact store (§11.5) runs on **Cloudflare R2** with **presigned direct transfer**
(workers PUT/GET checkpoints straight to R2; bytes never touch the control plane). Adds
a `colab_cell` tool (the control plane generates the paste-ready bootstrap cell) and
serves the agent binary itself at `/agent/linux-x86_64`. See **§0 Implementation status**.
v0.4: **proving-experiment ladder E0–E5** (§15); control plane hosted on **Fly.io** (§13).
v0.3: unified artifact model (§11) — deployable code units, lineage-complete checkpoints.
v0.2: control plane off the Mac; leases with hard walls + guaranteed cleanup; packing
scheduler; cost governance and dashboard; lazarus reduced to the artifact contract.

---

## 0. Implementation status (v0.7, 2026-07-19)

| Milestone | State | Gate | Proven |
|-----------|-------|------|--------|
| **M0** join loop, fleet, shell, logs | ✅ done | E0 | **on real Colab T4** — agent joins, `nvidia-smi` + matmul probe, live logs |
| **M1** train: code units, metrics, lineage checkpoints, resume | ✅ done | E1 | ✅ **on real Colab T4** — v11 (115M) trains on CUDA, metrics stream, ~460 MB checkpoints to R2 with full lineage, **resume test passed** (bounced the cell mid-run → resumed from the R2 checkpoint → completed; `slices [[0,80],[80,390]]`) |
| **M2** leases + provable cleanup | ✅ done | E2 | **locally via a mock provider** (launches real agent processes: drain, T-0 verified destroy with the agent hung, reconcile/orphan-kill, ledger); **live Vast E2 not yet run** (costs $) |
| **M3** packing scheduler | ⬜ not started | E3 | — |
| **M4** budgets + dashboard | 🟡 partial | E4 | ledger + `spend_status` (in M2) and the **one-page dashboard done** (Fleet · Runs · Money · Health, served by the CP), gated behind **Google sign-in** (email allowlist, HMAC session cookie); **budget caps + watchdog gates not done** |
| **M5** sweeps + panel gates + lazarus `load_checkpoint` + dynamics curve | ⬜ not started | E5 | — |

Deployed: control plane on Fly (`chuk-mcp-training.fly.dev`), **stateless on Neon**
(serverless Postgres), artifacts on R2, dashboard gated behind Google sign-in. The
**Drive cold-archive tier** (§11.5) is **complete and live-proven** — a real completed
Colab run's final checkpoint + logs + metrics tiered to Drive, promoted to `ckpt-final` on
R2, and streamed back through the retrieval resolver. **RBAC** (§12) is live: users +
roles (sysadmin › admin › write › read) in a team, with **self-service** scoped MCP API
keys (any signed-in user mints their own ≤ their role; admins manage the team) from the
dashboard's Access screen. The **chuk-experiments-server reporting mirror** (§11.6) is built
and verified end-to-end — optional and gated (off unless configured), it mirrors run
lifecycle + checkpoints (as artifacts) + final metrics (as results). Providers: `mock`
(tested), `vast` (skeleton, untested against the live API). Not yet built: the packing
scheduler (M3), budget caps + watchdog gates (M4), sweeps + lazarus integration + Lambda
driver (M5). (The R2 lifecycle rules that expire the hot copies need an Admin R/W R2 token,
or a manual dashboard config.)

---

## 1. Overview

`chuk-mcp-training` is a control plane, exposed as an MCP server, that provisions
**leased** GPU workers, packs queued experiments into those leases, and tears workers
down — guaranteed — when the lease expires. A single pip-installable worker agent
(`chuk-train-agent`) runs identically inside a Colab notebook, a Vast.ai container, or a
Lambda instance, dials **out** to the control plane, and executes assigned jobs
back-to-back until its lease ends.

Session kinds: `train` (async job → checkpoints → metrics), `eval` (batch evaluation /
panel deck against a checkpoint), `shell` (escape hatch, off by default).

**chuk-mcp-lazarus stays a separate server.** The only coupling is the artifact
contract (§10): the harness writes checkpoints with verifiable metadata; lazarus (on the
Mac) reads them via one `load_checkpoint(run_id, step)` tool. No shared code, no proxy.

### Design principles

1. **Workers dial out, always.** All worker↔control-plane traffic is one outbound
   websocket. No worker-side listener, ever.
2. **One agent, every backend.** Only provisioning differs per provider; everything after
   `agent --join <token>` is identical.
3. **A lease is a wall.** Every worker is provisioned with a runtime budget. When it
   expires: drain, checkpoint, upload, destroy. The only way past the wall is an explicit
   `extend_lease` MCP call, which is a budget decision, not a default.
4. **Pack the lease.** A rented GPU-hour is an asset to fill. The scheduler runs as many
   queued experiments as fit, back-to-back, and pours leftover time into resumable
   training slices. Utilization (busy GPU-min / leased GPU-min) is a first-class metric.
5. **Teardown never depends on the agent.** Cleanup is control-plane-driven against the
   provider API, idempotent, verified, and backed by a reconcile loop. A hung agent, a
   dead tunnel, or a wedged instance still gets destroyed and stops billing.
6. **Checkpoint-first.** Both backends preempt; every train job checkpoints on a step
   schedule *and* on drain. A preempted or drained job is a re-queued job.
7. **Multi-seed is first-class.** Sweeps fan out; cross-seed variance is queryable.
8. **Gates as code.** Panel criteria are registered and evaluated by the control plane
   from streamed metrics; watchdog gates auto-stop runaway runs.
9. **chuk conventions.** `chuk-mcp-server`, `@tool`, Pydantic schemas, error envelopes.
10. **Runs standalone; every external tier is optional.** The harness's own store + queue
    (Neon or SQLite) are always the source of truth. R2 (falls back to `file:`), Google
    Drive (archive tier), Google auth (falls back to the token box), and
    chuk-experiments-server (the reporting mirror, §11.6) are each **gated and optional** —
    unset ⇒ that tier is a no-op, and nothing outside the harness is ever a hard dependency.

---

## 2. Architecture

```
   Mac (client only)                        small VPS (always-on)
   ┌──────────────────┐   MCP (HTTP)   ┌────────────────────────────────┐
   │ Claude / mcp-cli ├───────────────►│ chuk-mcp-training              │
   │ chuk-mcp-lazarus │                │  lease manager · packing sched │
   │  (reads ckpts)   │                │  job queue · run registry      │
   └────────┬─────────┘                │  metrics store · gate engine   │
            │ signed URLs              │  budget/cost engine            │
            ▼                          │  web dashboard (same process)  │
   Artifact store (R2/B2) ◄────────────┤  provider drivers · reconciler │
            ▲                          └───────┬────────────────────────┘
            │ scoped uploads                   │ outbound WSS (agents join)
   ┌────────┴──────────┬───────────────────────┴──────┐
   ▼                   ▼                              ▼
 Colab notebook    Vast.ai instance             Lambda instance
 chuk-train-agent  chuk-train-agent             chuk-train-agent
```

The VPS is the only always-on component (SQLite + websocket server + static dashboard —
a £4/mo box is enough). The Mac is purely a client: it drives the MCP tools, views the
dashboard, and pulls checkpoints for lazarus. Provider API keys and store-registration
credentials live only on the VPS; workers receive scoped, expiring upload tokens per job.

---

## 3. Leases

A **lease** is the contract between a worker and its budget.

```jsonc
{
  "worker_id": "vast-8821",
  "provider": "vast",
  "price_hr": 0.38,                 // dollars; colab leases price in compute units
  "granted_min": 60,                // the runtime budget
  "started_at": "...",
  "drain_window_min": 5,            // reserved at the end for checkpoint + upload
  "extensions": []                  // each: {minutes, granted_by, at, projected_cost}
}
```

Lifecycle:

- **T-10 min** — scheduler stops assigning jobs whose estimate doesn't fit; only
  short atomic jobs or resumable slices may still start.
- **T-drain** (default T-5) — `drain` sent to the agent: running train jobs get
  `checkpoint_now` + stop; eval jobs get a grace period then kill; agent flushes logs
  and metrics, uploads, reports `drained`.
- **T-0** — control plane calls provider `destroy`, **whether or not the agent
  responded**. Destroy is verified via the provider API (instance status polled until
  gone); failure to verify raises an orphan alert and retries.
- **Reconcile loop** (every 10 min, always) — list instances at each provider, diff
  against the registry. Any billed instance the registry doesn't own, or owns but
  believes destroyed, is an **orphan**: alert + auto-kill (configurable, default on).

`extend_lease(worker_id, minutes)` is the *only* path past the wall. It re-checks the
budget (projected additional spend vs remaining cap), records who/when/why in the lease,
and reschedules the wall. Colab leases can be extended within platform limits only; the
lease there is our own wall layered under Colab's session limits and priced in compute
units.

Idle reaper: a leased worker with an empty queue and no assignable jobs for
`idle_reap_min` (default 10) is drained and destroyed early — the lease is a *ceiling*,
not a commitment to burn.

---

## 4. Packing scheduler

The queue holds many small jobs (spokes, seeds, evals, corpus builds — the hub-and-spoke
programme is exactly this shape). The scheduler's objective: **maximize useful GPU-min
per leased GPU-min.**

**Job classes**

- `atomic` — must complete within a single lease (evals, probes, corpus generation,
  short finetunes). Carries `est_minutes` (stated on first submit; thereafter learned:
  measured wall-time of prior runs of the same config/entrypoint, p90).
- `resumable` — train jobs that slice across leases via checkpoints. These are the
  **filler of last resort**: they can absorb any remaining lease time, run until drain,
  checkpoint, and requeue with `from_step` advanced.

**Assignment rule** (on worker-free or job-complete):

1. Filter queue to jobs whose `requirements` fit the worker (VRAM, disk, labels).
2. Prefer the highest-priority `atomic` job with `est_minutes × safety_factor`
   (default 1.25) ≤ remaining lease minutes (before drain window).
3. If none fit, assign the highest-priority `resumable` job — any remaining time is
   useful time for it.
4. If the queue is truly empty → idle reaper countdown starts.

An atomic job that overruns its estimate into the drain window is killed at drain and
marked `overran` (its learned estimate updates; it re-queues only if flagged
`retry_on_overrun`). This is deliberate: the wall is the wall.

**Batch submission**: `submit_batch([specs...])` enqueues a set with shared labels and a
target lease plan — the control plane responds with a packing preview: "these 9 jobs ≈
2× 60-min A6000 leases at ~$0.80 total; 1 resumable filler". You approve the provision
(or it auto-provisions under `auto_provision` policy with a spend cap).

**Utilization** is computed per lease and per provider-month:
`busy_min / granted_min`, with drain and env-prep time broken out (env prep is overhead
worth watching — image caching / pre-baked Vast templates are the fix if it grows).

---

## 5. Job model

### 5.1 JobSpec (kind = train)

```jsonc
{
  "name": "cn7-r1.1-masked-s81",
  "kind": "train",
  "class": "resumable",                  // "atomic" | "resumable"
  "est_minutes": 90,                     // required for atomic; advisory for resumable
  "priority": 5,                         // 1 (low) .. 9 (high)
  "code": "cn7-trainer@sha256:ab12…",   // deployable code unit (§11); sugar form:
                                         //   "repo": "…", "commit": "d045561" →
                                         //   resolved + built into a unit at submit time
  "entrypoint": "train",                 // named entrypoint from the unit's manifest
  "config": "configs/r1_1_masked.yaml",
  "overrides": { "seed": 81 },
  "artifacts_in": [
    { "name": "v11-base", "kind": "checkpoint" },
    { "name": "r1.1-corpus", "kind": "dataset" }
  ],
  "requirements": { "min_vram_gb": 16, "cuda": true, "disk_gb": 30 },
  "checkpoint": { "every_steps": 500, "keep_last": 3, "keep_every": 5000 },
  "max_hours": 12,                       // cumulative across slices
  "max_cost": 4.00,                      // cumulative; checkpoint-then-kill at cap
  "labels": ["cn7", "spoke:numeracy"]
}
```

Script contract unchanged: `$CHUK_CONFIG`, metrics JSONL to `$CHUK_METRICS`, checkpoints
to `$CHUK_CKPT_DIR`, exit 0. Additionally the harness exports `$CHUK_RESUME_CKPT` when a
slice resumes. ~5 lines to adopt.

### 5.2 SweepSpec

```jsonc
{ "template": { ...JobSpec... },
  "axes": { "overrides.seed": [80, 81, 82] },
  "concurrency": 2 }
```

Sweep children inherit class/estimates; `sweep_status` aggregates cross-seed
mean/std/range at matched steps; sweep-scope gates supported (multi-seed W_f rule).

### 5.3 Lifecycle events

`created, queued, assigned(worker, lease), started, sliced(from_step, to_step),
heartbeat_lost, preempted, drained, resumed(from_step), checkpoint(step, uri, hash),
gate_evaluated(gate, verdict), overran, cost_capped, completed, failed(reason),
cancelled` — append-only per run; the provenance record.

---

## 6. MCP tool surface

All chuk-mcp-server `@tool`, error envelopes throughout.

### Leases, fleet, provisioning

```python
@tool
def provider_offers(provider: str, gpu: str | None = None,
                    max_price_hr: float | None = None) -> list[Offer]
@tool
def provision(provider: str, lease_min: int, offer_id: str | None = None,
              gpu: str | None = None, max_price_hr: float | None = None) -> ProvisionResult
              # returns worker ref + lease; colab → bootstrap cell text (lease still enforced)
@tool
def fleet() -> list[WorkerInfo]        # gpu, price_hr, lease remaining, current job, util
@tool
def lease_status(worker_id: str) -> Lease
@tool
def extend_lease(worker_id: str, minutes: int, reason: str = "") -> Lease
@tool
def teardown(worker_id: str, force: bool = False) -> Ack   # drain-first unless force
```

### Runs, sweeps, batches

```python
@tool
def submit_run(spec: JobSpec, confirm_cost: bool = False) -> RunRef
@tool
def submit_sweep(spec: SweepSpec, confirm_cost: bool = False) -> SweepRef
@tool
def submit_batch(specs: list[JobSpec], auto_provision: bool = False,
                 spend_cap: float | None = None) -> BatchPlan   # packing preview
@tool
def run_status(run_id: str) -> RunStatus
@tool
def sweep_status(sweep_id: str) -> SweepStatus
@tool
def run_metrics(run_id: str, keys: list[str] | None = None,
                since_step: int = 0, downsample: int = 500) -> MetricSeries
@tool
def tail_logs(run_id: str, lines: int = 100) -> LogChunk
@tool
def stop_run(run_id: str, checkpoint_first: bool = True) -> Ack
@tool
def resume_run(run_id: str, from_step: int | None = None) -> RunRef
@tool
def run_events(run_id: str) -> list[Event]
```

### Budgets & spend

```python
@tool
def spend_status(period: str = "month") -> SpendReport
     # per provider + per label: spent, committed (live leases), projected, cap, headroom
@tool
def set_budget(scope: str, cap: float, period: str = "month") -> Ack
     # scope: "provider:vast" | "label:cn7" | "global"; colab caps in compute units
@tool
def utilization(period: str = "month") -> UtilReport   # busy/granted, overhead breakdown
```

### Checkpoints, artifacts, gates

```python
@tool
def list_checkpoints(run_id: str) -> list[CheckpointInfo]   # location: r2 | drive
@tool
def pin_checkpoint(run_id: str, step: int, name: str) -> Ack
@tool
def archive_run(run_id: str, force: bool = False) -> ArchiveReport
     # apply retention now for one run: drop mid-run checkpoints past the 24h grace
     # (keep final + pins), archive final + logs to Drive if the run is >30d old
     # (force ignores both the grace window and the age threshold)
@tool
def archive_runs(force: bool = False) -> ArchiveReport   # same policy across all runs
@tool
def archive_status(run_id: str | None = None) -> ArchiveStatus
     # per run: hot (R2) vs cold (Drive), grace/age remaining, bytes freed
@tool
def build_code_unit(repo: str, commit: str, name: str | None = None) -> CodeUnitRef
     # tarball + uv.lock + manifest, hashed, uploaded; agents cache by hash
@tool
def register_artifact(name: str, kind: str, uri: str, hash: str) -> Ack
@tool
def list_artifacts(kind: str | None = None, name: str | None = None) -> list[ArtifactInfo]
@tool
def artifact_lineage(ref: str, direction: str = "up") -> LineageGraph
     # "up": everything this was built from; "down": everything built from this
@tool
def artifact_url(name: str, ttl_min: int = 60) -> SignedUrl    # lazarus pulls via this
@tool
def register_gate(scope_id: str, name: str, expr: str, scope: str = "run",
                  action: str = "record") -> Ack   # action: "record" | "stop_run"
@tool
def check_gates(scope_id: str) -> list[GateVerdict]
```

Watchdogs are gates with `action="stop_run"`: `isnan(last(loss))`,
`no_improve(loss, 120min)`, `last(grad_norm) > 1e3`. Auto-stop always checkpoints first.

---

## 7. Agent protocol

> **Being reworked → `chuk-compute-spec.md`.** The worker daemon + wire protocol are being
> extracted into a compute-generic substrate (crates `chuk-compute-wire` + `chuk-compute-worker`):
> the daemon is a **worker** (not "agent"), the workload model generalizes to batch-vs-service,
> and per-run `sys/*` telemetry + a persistent (BYO/Mac) worker class are folded in. This section
> describes today's implementation; the target design and its M1–M7 sequencing live in
> `chuk-compute-spec.md`, with §12 there fixing the experiment-vs-service boundary against the
> agent/MCP deployment platform.

Single outbound WSS; JSON messages; per-worker monotonic sequence numbers; reconnect
replays from last acked seq. **The agent must tolerate a dark control plane**: if the
websocket drops, it keeps executing the current job, keeps checkpointing on schedule,
buffers metrics/logs to disk, and reconciles on reconnect. Local lease enforcement: the
agent knows its own wall time and self-drains at T-drain even with no connectivity —
belt; the provider-API destroy at T-0 is braces.

Worker → CP: `register, heartbeat, log, metric, checkpoint_uploaded, job_started,
job_exited(code), drained`.
CP → worker: `assign(job), cancel(job), checkpoint_now, drain(deadline), resume(from),
credentials(scoped, expiring)`.

Heartbeat loss > 90s ⇒ `unreachable`; running resumable job on an unreachable worker
past 10 min ⇒ `preempted` + re-queue (the checkpoint schedule bounds the loss).

---

## 8. Cost governance

- **Budget objects**: global, per-provider, per-label caps per period. `provision`,
  `extend_lease`, and `auto_provision` all check *projected* spend (live leases count as
  committed) against headroom and refuse on breach.
- **Pre-flight estimates**: submissions above a configurable threshold require
  `confirm_cost=True`; sweeps and batches show the multiplied total.
- **Per-job `max_cost`**: cumulative across slices; breach ⇒ checkpoint-then-kill,
  event `cost_capped`.
- **Colab compute units**: tracked as their own currency; per-GPU-class burn rates
  configurable (they drift); dashboard gauge shows estimated units remaining.
- **The three leaks, closed**: forgotten instances (lease wall + idle reaper + reconcile
  orphan-kill), runaway runs (watchdog gates + max_hours/max_cost), sweep multiplication
  (pre-flight multiplied estimate + confirm flag).
- **Ledger**: every lease and extension writes a cost record; `spend_status` is computed
  from the ledger, not from provider billing APIs (those reconcile monthly as a check).

---

## 9. Dashboard

Served by the Rust control plane itself (static HTML + `fetch`; no Grafana stack to
babysit; metrics JSONL remains exportable if that ever changes) and **gated behind
app-level Google sign-in** — an email allowlist carried in an HMAC-signed session cookie,
so only the owner sees it while MCP/agents stay on bearer tokens (§12); the token box is
the local-dev fallback. **Built.** One page, four bands:

1. **Fleet** — per worker: GPU, $/hr ticking cost, lease remaining (countdown), current
   job, utilization bar, **kill button** (one click, unconditional, drain-first with a
   force option).
2. **Runs** — live loss sparklines, step rate, last checkpoint age; sweeps show seed
   bands (mean ± range) on shared axes.
3. **Money** — budget burn-down per provider and per label; committed vs spent vs
   projected month-end; Colab units gauge.
4. **Health** — gate verdict board (panel + watchdogs), orphan alerts, queue depth,
   utilization trend.

The MCP tools (`fleet`, `spend_status`, `run_metrics`, `check_gates`) are the same
queries — conversational surface and dashboard read one store.

---

## 10. Lazarus integration (artifact contract only)

Separate servers; the entire interface is the checkpoint layout:

```
runs/<run_id>/ckpt/step_<n>/
  model.safetensors
  optim.pt                 # optional; excluded from lazarus pulls
  meta.json                # commit, config_hash, seed, step, tokenizer_hash, arch
```

chuk-mcp-lazarus gains one tool, `load_checkpoint(run_id, step | pin_name)`, which
resolves via `artifact_url`, downloads `model.safetensors` + `meta.json`, **verifies
tokenizer_hash against the local tokenizer artifact and refuses mismatches** (the CN-7
day-1 class of bug becomes a load-time error), and loads into the existing MLX path.

What this buys immediately: training-dynamics archaeology — any of the lazarus tool
surface (fingerprint rank, probes, decode_residual) run against step 0/5k/10k/…
checkpoints on the Mac, async, on owned hardware. Live on-GPU probing and
lazarus-on-rented-GPU are explicitly out of scope for this repo; if wanted later they are
new decisions, not latent features here.

---

## 11. Artifact model & storage

Everything the rig moves is a **typed, content-addressed artifact with recorded
lineage**. Kinds: `code`, `env`, `dataset`, `checkpoint`, `metrics`, `logs`, `deck`
(panel/eval outputs). Identity is the content hash; names and pins are pointers.

### 11.1 Code units — the deployable unit

"Clone repo + pip install" fails three ways at once: it's repeated per lease (env-prep
overhead), it drifts (unpinned transitive deps make yesterday's run unreproducible), and
it can't be containerized uniformly (Vast is docker-first, Colab can't run docker). The
deployable unit is therefore **Python-level, with the container as an optional cache**:

```
code unit  =  tarball of repo@commit
           +  uv.lock (fully pinned, hashes included)
           +  unit.toml manifest:
                name, version
                entrypoints = { train = "python train.py",
                                eval_deck = "python eval.py", … }
                python = "3.11"
                requires = { cuda = ">=12", min_vram_gb = 16 }   # defaults for jobs
unit id    =  sha256 over the tarball
```

- Built by `build_code_unit(repo, commit)` on the control plane (or CI on push) →
  `artifacts/code/<name>/<sha>/`.
- **Agents cache by hash** — a warm worker starts the next packed job in seconds; this
  is where the packing scheduler's env-prep overhead goes to die.
- On Vast, an `env` artifact may map the unit's dep-set to a pre-baked docker image
  (image digest recorded); on Colab the same unit installs from the lockfile into the
  session. **Same hash, two substrates, one behaviour.**
- JobSpec accepts `repo`+`commit` sugar (resolved to a unit at submit) so quick
  iteration never requires a manual build step.

### 11.2 Checkpoints — lineage-complete

`meta.json` grows from "enough to load" to "enough to reproduce":

```jsonc
{
  "step": 15000, "seed": 81, "arch": "tinymodel-v11",
  "code": "cn7-trainer@sha256:ab12…",
  "config_hash": "…", "tokenizer_hash": "…",
  "parent_checkpoint": "v11-base@sha256:…",        // resume/base lineage
  "datasets": ["r1.1-corpus@sha256:…"],
  "run_id": "…", "slices": [[0, 9500], [9500, 15000]]
}
```

Every number on a panel is mechanically traceable to exact code, corpus, tokenizer,
base, and seed. The tokenizer-hash check stays a load-time refusal in lazarus.

### 11.3 Logs & metrics as artifacts

Live logs stream over the agent websocket; at drain/completion each slice's log is
compressed and uploaded (`logs` kind), indexed by run + slice. `tail_logs` serves live
from the stream, historical from the artifact. Retention per kind: logs 90 days,
metrics indefinite (small), checkpoints per keep-policy, **code units indefinite** —
they are the reproducibility substrate. `pin` exempts anything from retention.

### 11.4 Lineage queries

`artifact_lineage(ref)` walks the graph both ways. "Up" answers *what produced this
checkpoint*. "Down" answers *what did this corpus contaminate* — the query you reach
for the day a corpus bug is found, returning every run, checkpoint, and panel deck
downstream of the bad hash.

### 11.5 Storage layout (S3-compatible; R2 preferred for zero egress)

```
s3://chuk-train/                                # R2 = hot cache; Drive = cold canonical
  artifacts/
    code/<name>/<sha>/{unit.tar.zst, unit.toml, uv.lock}
    env/<name>/<sha>.json                       # docker digest / colab install recipe
    datasets/<name>/<sha>/...
    decks/<name>/<sha>/...
  ckpt-hot/<run_id>/step_<n>/{model.safetensors, meta.json, optim.pt}   # agent uploads; ~1d lifecycle
  ckpt-final/<run_id>/step_<n>/{model.safetensors, meta.json}           # promoted on completion; ~30d
  runs/<run_id>/{spec.json, events.jsonl, ...}                          # run tree (logs/metrics live in the store)
  ledger/<yyyy-mm>.jsonl                         # cost records

drive:chuk-train/runs/<run_id>/                 # canonical cold copy (Drive v3, drive.file)
  ckpt/step_<n>/{model.safetensors, meta.json}  logs.txt  metrics.json
```

Top-level `ckpt-hot/` / `ckpt-final/` prefixes (not `runs/<id>/ckpt/`) let R2 lifecycle
rules target them by prefix. The `checkpoints` metadata row records each checkpoint's
`location` (`r2_hot` | `r2_final` | `drive`) + Drive file ids, so a stable per-checkpoint
URL (`/api/checkpoint/<run>/<step>/<file>`) resolves R2 (302 to a presigned URL) or Drive
(streamed) transparently.

Workers upload with credentials scoped to `runs/<run_id>/` and read scoped to declared
`artifacts_in` plus their assigned code unit. Reads (Mac pulling checkpoints) dominate
long-term — R2's zero egress is the argument. ~500MB/checkpoint with optimizer state at
115M scale.

**Cold-archive tier.** R2 is hot storage; completed runs' checkpoints and logs tier to
**Google Drive** (5 TB, `drive.file` scope, resumable chunked upload) as durable,
browsable cold storage — freeing R2 while honoring `keep_last`/`keep_every`/pins. An
archived object records a `drive://<fileId>` location, and `artifact_url`/`blob` resolve
R2 *or* Drive transparently. Auth is a long-lived offline refresh token, minted once via
`scripts/authorize-drive.py` and refreshed against the dashboard's Google client; the
`DriveClient` (token refresh, folder-ensure, resumable upload/download/delete) is built
and live-proven, with the tiering/retrieval job pending.

### 11.6 chuk-experiments-server reporting (optional mirror)

[chuk-experiments-server](https://github.com/chrishayuk/chuk-experiments-server) is the
research **system of record** (programme → experiment → run → result/artifact). The harness
reports into it as an **optional, gated mirror** — never a dependency (design principle 10).
When `CHUK_EXPERIMENTS_URL` + a WRITE API key are configured, the control plane:

- **on run submit** — POSTs to `/v1/experiments/{slug}/runs` (a configurable default
  experiment; programme auto-creates). The server **mints its own** `RUN-…` id, so we send
  **our** run id as the `slug` and, via a follow-up `PATCH /v1/runs/{id}`, as
  `harness_session_id`; we store their returned id on our run row for later calls.
- **on lifecycle transitions** — `PATCH /v1/runs/{id}` with the mapped `status`
  (queued/running/completed/failed/killed/cancelled), `started_at`/`ended_at`, `cost_usd`,
  `wandb_url`.
- **on checkpoint upload** — `POST /v1/runs/{id}/artifacts` (`kind=checkpoint`,
  `role=produced`, the stable `https` per-checkpoint URL, `bytes`, `sha256`, and our lineage
  superset as `meta`); optional `result` rows for final metrics and `/v1/pins` for
  `latest`/`best`.

Unset ⇒ a complete no-op; the harness's own store + queue remain authoritative. Their run
ids and ours share the same `RUN-YYYYMMDD-HHMMSS-NNNNN` shape but are **parallel, independent
id-spaces** (each from its own store sequence), linked by `harness_session_id`. Their
`/v1/queue` claim/lease contract (workers *pull* work the experiments-server enqueued) is a
possible **later opt-in** execution mode — an add-on, never a replacement for our own queue.

Every report is **fire-and-forget**, spawned off the run's critical path (a slow or down
experiments-server logs a warning and never blocks or fails a run). Built and verified
end-to-end (`crates/chuk-train-cp/src/experiments.rs`). *Known limitation:* a transient
failure drops that one update (no retry) — a durable outbox with retry is future work, so
the mirror is best-effort, not a guaranteed-consistent replica.

---

## 12. Security

- Join tokens: single-use, short-lived, minted per provision; exchanged for a worker
  credential bound to worker id + lease.
- Provider keys, HF tokens, store-admin credentials: VPS only, never on workers.
- `shell` sessions: off by default, explicit server-config allow.
- Rented GPUs are untrusted hardware: only public code/data ships to them.
- Agent/worker surfaces behind bearer tokens (join token → worker credential; run-scoped
  upload grants for checkpoint writes).
- Dashboard behind **app-level Google OAuth**: an HMAC-signed session cookie; the token
  box is the local-dev fallback. Optionally Cloudflare Access as an outer layer.
- **RBAC.** Users have a role — **sysadmin › admin › write › read** — in a team; the
  Google session's email maps to a user + role. The `/api/*` MCP surface is authenticated
  by an **API key** (or the legacy master token → sysadmin) that resolves to a role. Keys
  are `ck_…`, stored only as a **sha256 hash + short prefix**, shown once at creation, and
  revocable. Roles are enforced per endpoint: read = view; write = submit/manage runs,
  provision, teardown; admin = archive/retention + manage the team's users + all keys.
  Key management is **self-service**: any signed-in user mints/lists/revokes their **own**
  keys scoped ≤ their own role from the dashboard's **Access** screen (admins additionally
  see and revoke every key in the team). `read`/`write`/`admin` mirror
  chuk-experiments-server; `sysadmin` is the extra platform-owner tier. A `teams`
  table scaffolds multi-tenant (single default team today).
- Drive archive uses the narrow `drive.file` scope (only files this app creates); the
  refresh token lives on the control plane only.

---

## 13. Rig topology

- **Control plane on Fly.io**: one `shared-cpu-1x` machine (a few $/mo), Dockerfile +
  `fly.toml`, **auto-stop OFF** (agents need the WSS endpoint up 24/7). SQLite on a Fly
  volume; nightly snapshot to R2 (`ledger/` and `events.jsonl` are mirrored there
  anyway, so the machine is rebuildable from the repo + one restore). Secrets (provider
  keys, R2 creds) via `fly secrets`. DNS: `train.chukai.io` → Fly app (WSS for agents;
  HTTPS for MCP endpoint + dashboard). This is what makes the dashboard reachable from
  anywhere — phone included — and MCP control possible remotely.
- **Auth**: Fly gives TLS, not auth — the app enforces a bearer token on the MCP
  endpoint and session auth on the dashboard; optionally Cloudflare Access in front of
  the domain for a second layer. Agent WSS auth is the join-token exchange, unchanged.
- **Alternative**: any small always-on VPS works identically; nothing in the design is
  Fly-specific beyond the deploy files.
- **Mac (M3 Max)**: client only — mcp-cli/Claude driving tools, dashboard in a browser,
  chuk-mcp-lazarus pulling checkpoints. Nothing breaks when it sleeps.
- **Artifact store**: R2/B2, off-site; the home uplink is never in the checkpoint path.
- **Workers**: ephemeral by construction — leased, packed, drained, destroyed.
- **Backups**: nightly SQLite snapshot + `ledger/` already in object storage; the VPS is
  rebuildable from a repo + one restore.

---

## 14. Build order

**Colab is the proving backend** — the account already exists, the marginal cost is
units already paid for, and E0/E1 exercise the entire agent/checkpoint/resume machinery
without renting anything. No dollar leaves the building until M2/E2.

1. **M0** ✅ — Fly control plane skeleton: registry, queue, WSS endpoint; agent join +
   heartbeat; `fleet`, `tail_logs`. Prove the pull loop on a Colab T4. *(E0 green on a
   real Colab T4.)*
2. **M1** ✅ — `train` end-to-end: code-unit build + agent hash-cache, JobSpec, metrics
   tailing, checkpoint upload with lineage `meta.json`, resume-after-kill on Colab
   (close the tab; job re-queues and resumes from the uploaded checkpoint). *(E1 green on
   a real Colab T4: v11 trains, checkpoints to R2 with lineage, and the tab-bounce resume
   test passed — `slices [[0,80],[80,390]]`.)*
3. **M2** ✅ — **leases + cleanup**: lease walls, drain protocol, provider-verified destroy,
   reconcile loop + orphan kill, idle reaper. Vast driver (`provider_offers`,
   `provision`, `teardown`). *Test: rent 15 min on Vast, confirm the instance is
   provably gone at T-0 with the agent deliberately hung.* *(Verified locally via a mock
   provider that launches real agent processes; the live Vast E2 has not been run.)*
4. **M3** ⬜ — **packing**: job classes, learned estimates, assignment rule, resumable
   slicing, `submit_batch` preview, utilization metric. *Test: 6 short evals + 1
   resumable train packed into one 60-min lease, ≥85% utilization.*
5. **M4** 🟡 — budgets + dashboard: ledger, caps, `spend_status`, watchdog gates, the
   one-page dashboard. *(Ledger + `spend_status` done in M2; caps, watchdog gates, and
   the real dashboard remain — only a M0 stub dashboard exists.)*
6. **M5** ⬜ — sweeps + panel gates; lazarus `load_checkpoint` + tokenizer-hash
   verification; first training-dynamics probe curve. Lambda driver.

M2 before M3 deliberately: **cleanup is trusted before packing makes leases busy.**

---

## 15. Proving experiments (E0–E5)

Principle: **prove the rig at v11 scale** — 115M params, TinyStories, jobs measured in
minutes, checkpoints ~500MB. Every experiment uses assets that already exist (v11 base,
R1.1-style corpus, frozen greedy-continuation deck, W_f fitting code). Nothing larger
runs until E5 passes. Each E gates its milestone; the milestone isn't done until its E
is green.

- **E0 · join loop** *(gates M0)* — bootstrap the agent on a Colab T4; `fleet` shows
  it with correct GPU/VRAM; a `shell` job (allow-flagged for this test only) runs
  `nvidia-smi` plus a 30-second matmul throughput probe; logs stream live through
  `tail_logs`. Minutes; costs only Colab units.

- **E1 · first real job** *(gates M1)* — single-seed W_f finetune on v11 (the CN-1
  addressing fit) as an `atomic` job on Colab: code unit built from the existing repo,
  `v11-base` and the corpus registered as artifacts, checkpoint lands in R2 with full
  lineage `meta.json` (tokenizer hash included). Then the resume test: **close the
  Colab tab mid-run** — job re-queues, resumes from the uploaded checkpoint, completes.
  Acceptance: final novel|seen rank matches a Mac-run reference within seed noise
  (~90s–30min of GPU; the {93, 208} variance lesson says compare distributions, not
  points).

- **E2 · the wall** *(gates M2)* — cheapest Vast GPU on a **15-minute lease**; one
  frozen greedy-continuation deck as an atomic eval. Then re-run with the agent
  deliberately hung (`kill -STOP` on the process): confirm provider-verified destroy at
  T-0, reconcile loop reports zero orphans, and the ledger shows total spend ≈ $0.10.
  This is the experiment that earns trust in cleanup before packing makes leases busy.

- **E3 · packing showcase** *(gates M3)* — one 60-minute Vast lease packed with: a
  3-seed W_f sweep (atomic; estimates learned from E1) + two panel-deck evals + a
  resumable R1.1-style midtrain slice as filler. Acceptance: ≥85% utilization,
  cross-seed std reported by `sweep_status` (the multi-seed W_f gate as a query), the
  filler slice checkpointed and re-queued at drain. This experiment **is** the current
  manual workflow, mechanized — if E3 is green the rig already pays rent.

- **E4 · money & watchdogs** *(gates M4)* — submit a deliberately diverging run (10×
  LR): the watchdog gate checkpoints-then-kills within its window and the event log
  shows why. Set a $1 cap on a test label and submit a batch projected at $2:
  `submit_batch` refuses with the projection. Sanity-check the Colab units gauge
  against observed unit drain over one session.

- **E5 · first new science** *(gates M5)* — an R1.1 remix midtrain (3 seeds) run
  entirely on the rig with save-every checkpoints; panel gates registered and evaluated
  by the control plane; lazarus `load_checkpoint` pulls the step ladder to the Mac and
  produces the first **training-dynamics curve** — fingerprint rank vs training step
  (when does addressing form?). Acceptance: a result quotable in the CN log whose
  provenance comes entirely from `artifact_lineage`, with no hand-kept notes required.

E0–E2 should cost under a pound total; E3–E5 a few pounds. The ladder is deliberately
boring until E5 — the rig earns the right to run new science by first re-running old
science identically.

---

## 16. Open questions

- Name: `chuk-mcp-training` stands now that lazarus is fully separate.
- Estimate learning: p90 of prior wall-times per (entrypoint, config-hash, gpu-class) —
  is gpu-class normalization needed day one, or only once the fleet is heterogeneous?
- Auto-provision policy: allow the scheduler to rent under a cap when queue depth × est
  time exceeds a threshold, or keep provisioning strictly human/MCP-initiated? (v0.2
  default: strictly initiated; `auto_provision` exists but ships off.)
- Vast template images: pre-baked docker image with common deps to cut env-prep overhead
  — measure first via the utilization overhead breakout.
- Colab unit burn rates: config table vs scraped — start with a config table and
  reconcile against observed unit drain.
