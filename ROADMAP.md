# Roadmap

Status of `chuk-mcp-training` against the spec (`docs/specs/chuk-mcp-training-spec.md`),
plus the cross-cutting work that isn't a single milestone. Milestones are gated by the
proving experiments E0–E5 (spec §15): a milestone isn't done until its E is green.

## Milestones

| # | Scope | Code | Proven | Gate |
|---|-------|------|--------|------|
| **M0** | join loop, fleet, shell runs, log streaming | ✅ | ✅ **real Colab T4** (E0) | E0 |
| **M1** | train: code units, metrics, lineage checkpoints, resume | ✅ | ✅ **real Colab T4** (E1) — trains, R2 checkpoints, resume passed | E1 |
| **M2** | leases, drain, provider-verified destroy, reconcile/orphan-kill, ledger | ✅ | 🟡 **mock provider locally** — live Vast pending | E2 |
| **M3** | packing scheduler (job classes, learned estimates, `submit_batch`, utilization) | ⬜ | — | E3 |
| **M4** | budget caps, watchdog gates, one-page dashboard | 🟡 **dashboard done**; caps + watchdogs pending | dashboard live | E4 |
| **M5** | sweeps, panel gates, lazarus `load_checkpoint`, dynamics curve, Lambda driver | ⬜ | — | E5 |

## What's built beyond the milestone list

- **Rust control plane + agent**, Python MCP tool surface (thin REST client).
- **Neon (serverless Postgres) store** — the control plane is stateless on Fly; the
  `Store` seam still ships SQLite for local dev + tests. Store URL selects the backend.
- **R2 artifact store** with presigned direct upload/download (spec §11.5/§12) — live;
  checkpoints live under `ckpt-hot/<id>/` (agent uploads) and `ckpt-final/<id>/`
  (promoted on completion), so R2 lifecycle can target them.
- **Full operator dashboard** (spec §9) — served by the CP, Google-authed. A clean
  per-run view (live loss + metric toggles, streamed logs, checkpoints with metadata +
  download, events, config, out-links) and an overview with fleet/runs filters +
  pagination. Sortable `RUN-YYYYMMDD-HHMMSS-NNNNN` ids (store-backed 5-digit
  sequence, matching chuk-experiments-server).
- **Drive cold-archive tier** (complete) — completed runs auto-tier their final
  checkpoint + logs + metrics to Google Drive (canonical), the final is promoted to
  `ckpt-final` on R2, and R2 lifecycle expires the hot copies. Stable per-checkpoint URLs
  resolve R2 or Drive; `archive_run`/`archive_runs`/`archive_status` MCP tools; idempotent
  backstop sweep. **Proven on a real completed Colab run** (final streamed back from Drive).
- **RBAC / auth** — users + roles (sysadmin › admin › write › read) in a default team
  (multi-team scaffolded), **self-service scoped API keys** (hashed at rest, shown once):
  any signed-in user mints/lists/revokes their **own** keys ≤ their role from the
  dashboard **Access** screen, admins manage the whole team; per-endpoint role
  enforcement. `read`/`write`/`admin` mirror chuk-experiments-server; `sysadmin` is the
  extra platform tier the legacy master token resolves to.
- **chuk-experiments-server reporting mirror** (§11.6) — optional + gated (off unless
  `CHUK_EXPERIMENTS_URL` + a WRITE key are set): the CP creates the run there (our id as
  `slug`/`harness_session_id`), PATCHes lifecycle, registers checkpoints as artifacts, and
  submits final metrics as results — all **fire-and-forget** off the run's critical path.
  Unset ⇒ no-op; the harness always runs standalone. Verified end-to-end.
- **Dogfooding demo** — `scripts/demo.sh` spins up a local CP + mock workers running the
  (enriched) stub-trainer, so the dashboard fills with live data; isolated from prod.
- **Fly deploy**: `chuk-mcp-training.fly.dev`; the CP serves the agent binary and
  generates the Colab bootstrap cell (`colab_cell`).

## Immediate next steps

1. **Run stop/cancel + resume** — quick win: the agent already fully handles `CpToAgent::Cancel`,
   but nothing on the CP can send it (no route, no hub method, no MCP tool), so the only way to
   stop a run today is to tear down its worker. Add a `stop_run` (send Cancel → `Cancelled`) +
   `resume_run` (re-queue a terminal run). Most plumbing exists.
2. **Heartbeat-timeout requeue** — `last_seen` is recorded on every message but never scanned,
   so a half-open tunnel (frozen Colab tab) strands a run until the lease wall instead of the
   spec's 90s→unreachable / 10min→preempted. A periodic staleness scan that requeues resumable
   runs bounds the loss the checkpoint schedule is meant to bound.
3. **M4 budgets + watchdogs** — the dashboard's done; the remaining M4 is per-provider/
   label caps checked on provision/extend, and watchdog gates (isnan/no-improve/grad-blowup)
   that checkpoint-then-stop (reusing the stop path from step 1).
4. **Single-use, per-provision join tokens** — the spec wants tokens minted per provision and
   bound to a worker id + lease; today a single static config token lets any holder join as any
   worker forever. Mint a one-time token in `provision`, bind + expire on first use.
5. **Live Vast E2** — rent 15 min, hang the agent, prove provider-verified destroy.
6. **M3 packing** when there's rented-GPU pressure.
7. **R2 lifecycle permission** (see backlog) so the R2 hot copies actually auto-expire.

*(E0 and E1 are done — both proven on a real Colab T4, including E1's resume test.)*

## Then, by milestone

- **M3 · packing** — atomic vs resumable job classes, learned p90 estimates per
  (entrypoint, config, gpu-class), the `est × safety_factor` fit rule, resumable slices
  as filler, `submit_batch` packing preview, utilization metric. Gate E3: ≥85% util.
- **M4 · budgets + watchdogs** (dashboard shipped) — per-provider/label caps checked on
  provision/extend, watchdog gates (`isnan(loss)`, `no_improve`, `grad_norm` blowups)
  that checkpoint-then-stop. Gate E4.
- **M5 · science** — sweeps (`submit_sweep`, cross-seed variance), panel gates evaluated
  from streamed metrics, lazarus `load_checkpoint` + tokenizer-hash verification (the
  archive tier's stable checkpoint URLs are the handle it consumes), the first
  training-dynamics curve, Lambda driver. Gate E5.

## Feature candidates (functionality review, 2026-07-19)

A gap analysis of the built system, ranked within each group by value-to-effort. The
highest-value quick wins are already promoted into *Immediate next steps* above.

**Portable "join anywhere" agent harness + run telemetry (direction, 2026-07-19)**

The agent should be a standard harness that runs identically on Colab, Vast, a Mac, or any
box, and should capture rich system telemetry per run — not just the trainer's app metrics.

- **Multi-target agent builds + target-aware download** (M) — the download path is a single
  hardcoded `/agent/linux-x86_64` (`crates/chuk-train-proto/src/constants.rs:15`). Build and
  serve `/agent/{os}-{arch}` for `linux-x86_64`, `linux-aarch64`, `darwin-arm64`,
  `darwin-x86_64` so a Mac (or ARM box) can join. Prereq for "run on my Mac."
- **One detect-and-install bootstrap** (S–M) — a single `curl … | sh` installer that detects
  OS+arch, pulls the matching binary, and runs `agent --join <token>`. Unify today's
  per-environment paths: the Colab cell (`bootstrap/colab_cell.py`) and the Vast-specific
  onstart string templating (`crates/chuk-train-cp/src/provider/vast.rs:21`) both become thin
  wrappers over it; Mac + any Linux box use it directly. (Composes with "generalized provider
  bootstrap" below.)
- **BYO / persistent worker class** (M) — the lease model assumes provider-provisioned,
  wall-enforced, **destroyable** instances (`crates/chuk-train-cp/src/lease.rs`). A Mac or a
  box you own is different: a long-lived join, **no lease wall, no teardown**, marked
  non-preemptible. Add a persistent worker class the CP never tries to destroy (and a
  long-lived join-token class for it — pairs with the single-use-token item, which stays for
  leased workers). This is the real "run on my Mac, whenever" capability.
- **Richer capability registration** (S–M) — `Hardware` is thin today (host/os/gpu/vram_mb/
  driver, `crates/chuk-train-proto/src/domain.rs`). Add cpu_cores, ram_gb, arch, accelerator
  kind (`cuda`/`mps`/`cpu`), and preemptible/persistent flags, so the scheduler can match jobs
  (feeds *Requirements-aware assignment* and packing).
- **Per-run system telemetry** (M, high value) — the agent detects the GPU once at register
  via `nvidia-smi` (`crates/chuk-train-agent/src/hardware.rs`) but never samples during a run.
  Add a periodic sampler streaming a `sys/*` metric namespace over the existing `Metric` wire
  channel: GPU util % / mem used+total / temp / power / SM clock (nvidia-smi→NVML), CPU util /
  RAM / disk+net I/O (sysinfo/procfs), and per-process (trainer PID) CPU+RAM. Flows into the
  existing metrics store → dashboard utilization curves → experiments-server. Feeds the
  packing utilization metric, watchdog gates (GPU-mem OOM, thermal throttle), and MFU/efficiency
  (tokens_per_s vs GPU util). Turns a training dashboard into an ops dashboard. *(Apple-Silicon
  GPU telemetry is best-effort — `powermetrics` needs sudo; treat MPS util as a known gap.)*

**Spec gaps beyond the milestone headers**
- **Requirements-aware assignment** (S–M) — `pump()` assigns the oldest queued run to any
  idle worker with no fit check; add `requirements {min_vram_gb, labels}` + priority so a
  16 GB job never lands on a T4. A de-risking down-payment on M3 packing.
- **`eval` job kind + `deck` artifact** (M) — only `Shell`/`Train` exist; panel gates and
  E3's eval decks are unbuildable without it. Prereq for the M5 science story.
- **Learned `est_minutes`** (M) — store p90 of prior wall-times per `(entrypoint,
  config_hash)` so packing's fit rule is real (§16 open question; skip gpu-class first).

**Operational hardening**
- **Durable outbox for the experiments mirror** (S–M) — reports are fire-and-forget, so a
  transient 5xx silently drops a checkpoint-artifact registration and diverges the registry;
  persist pending ops with retry (the Drive archiver's pattern) without making it a dependency.
- **Multi-machine story** (L) — live agent sockets, the grant table, and idle timers are all
  in-process; >1 Fly machine needs sticky agent routing or Redis pubsub fan-out (the store
  module already flags this).
- **`/metrics` endpoint + structured request logging** (S) — only `/healthz` is exposed; the
  CP already holds queue depth, live leases, spend rate, orphan counts for a Prometheus surface.
- **Stream large objects** (M) — uploads buffer the whole body and the Drive proxy read pulls
  ~440 MB into RAM; the fs fallback + Drive proxy path should stream.
- **Provider timeouts/backoff + parallel teardown** (M) — the Vast client has no timeout/retry
  and the lease clock destroys+verifies serially inline, so one slow `destroy` stalls T-0 for
  every other expiring lease.
- **Agent durable buffering + seq/ack replay** (M–L) — logs/metrics are dropped while the CP is
  dark; a disk-backed outbound buffer + monotonic seq/ack makes "tolerate a dark CP" real.

**Product / UX**
- **Notifications** (S–M) — run complete/fail, gate trip, orphan kill, budget breach to a
  webhook/Slack/email sink; turns the rig from pull to push. Orphan kills only warn to the log today.
- **Live dashboard push (SSE)** (S–M) — the per-run view polls five endpoints every 2s; relay
  the already-streamed logs/metrics over one SSE stream. Add a full historical-log view.
- **Submit-run / provision from the dashboard** (M) — it's read-only except teardown + access;
  an offers browser + submit form makes it usable without an MCP client.
- **Artifact & lineage browsing** (M–L) — `list_artifacts`/`artifact_lineage`/`register_artifact`
  ("what did this corpus contaminate"); the edges are latent in checkpoint `meta.json` but unindexed.
- **Multi-team scoping** (M) — `team_id` rides on users/keys but fleet/runs/spend ignore it; make
  runs/leases carry + filter by team to turn the scaffolding into real tenancy.
- **Cost preflight + `confirm_cost` on submit; Colab unit accounting** (S–M) — the spec's cost
  guardrails (§8): a shown estimate gate on submit, and Colab compute-units as their own currency.

**Integration depth**
- **Native W&B logging** (M) — W&B is only a forwarded out-link today; the CP already ingests
  every metric batch, so creating a W&B run + streaming metrics is a natural add.
- **experiments-server pull/queue executor mode** (L) — the deepest integration: a gated adapter
  that turns harness workers into executors for externally-queued experiments (§11.6), an add-on
  that never replaces our own queue.
- **lazarus `load_checkpoint` + tokenizer-hash verify** (M, mostly lazarus-side) — the harness
  side is ready (stable resolver URLs + `tokenizer_hash` in meta); produces E5's first dynamics curve.
- **Lambda driver + generalized bootstrap** (M) — the `Provider` trait is clean; a Lambda driver
  plus a shared onstart/bootstrap generator makes adding providers uniform.

## Hardening backlog

Things we've hit or know are soft, roughly by priority:

- **R2 lifecycle permission** — the archive tier applies expiry rules on boot, but the
  current R2 token lacks bucket-lifecycle permission (AccessDenied). Recreate the token
  with **Admin Read & Write**, or set the rules in the Cloudflare dashboard (`ckpt-hot/`
  expire 1 day, `ckpt-final/` 30 days). Until then checkpoints archive to Drive fine but
  the R2 copies don't auto-expire.
- **Code-unit build workflow** — `build_code_unit(local path)` only works against a local
  control plane (the deployed one can't see local files and has no git). Today's Colab
  path builds against a local CP pointed at the same R2 + Neon, then submits the sha. Add
  git-in-image + build-from-git-URL, or a signed tarball-upload endpoint.
- **Large-object robustness** — the `file:` backend and the CP's own `/api/blob` still
  buffer whole objects in memory; fine now that R2 is the default (presigned, bytes bypass
  the CP), but the fs fallback, code-unit puts, and Drive proxy reads should stream.
- **Live provider validation** — run the real Vast E2 (rent 15 min, hang the agent, prove
  destroy) and confirm the Vast driver against the live API.
- **Auth hardening** — RBAC (users/roles/teams + self-service scoped API keys) done.
  Remaining: periodic join/API token rotation; optional Cloudflare Access as an outer
  layer; scope upload grants tighter.
- **Observability** — structured request logging, a `/metrics` endpoint, orphan/gate
  alerting beyond log lines.
- **Tests** — integration tests for the agent↔CP protocol and the lease state machine;
  today's coverage is unit tests + live round-trip tests (Neon, Drive, R2 lifecycle) +
  manual/`demo.sh` end-to-end runs.
