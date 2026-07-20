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
  pagination. Sortable `EXEC-YYYYMMDD-HHMMSS-NNNNN` ids (store-backed 5-digit
  sequence — deliberately distinct from chuk-experiments-server's own `RUN-…`
  shape; see *Integration depth* below).
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
  submits final metrics as results — every event persists to a durable outbox before the
  first delivery attempt and retries with capped backoff on failure, never silently dropped
  (see *Integration depth* below). Unset ⇒ no-op; the harness always runs standalone.
  Verified end-to-end.
- **`submit_run_from_experiment`** (2026-07-20, closes the push half of the "pull/queue
  executor mode" item below) — `POST /runs/from-experiment/{run_id}` / the matching MCP
  tool takes an existing chuk-experiments-server `RUN-…` id, fetches that run's own
  `config`/`workspec` (`Experiments::fetch_run` + `train_spec_from_experiments_run`,
  `experiments.rs`), builds the `TrainSpec` from it, and submits attached
  (`experiment_ref` already set) — so "create an experiment, get it running here" is one
  call instead of re-specifying the same training job by hand at `submit_run` time. Guards
  against resubmitting an already-attached or non-`queued` run. The **pull** direction
  (this harness polling `/v1/queue` and self-selecting work, rather than being told which
  run to run) is still open — see below.
- **Dogfooding demo** — `scripts/demo.sh` spins up a local CP + mock workers running the
  (enriched) stub-trainer, so the dashboard fills with live data; isolated from prod.
- **Fly deploy via CI/CD**: `chuk-mcp-training.fly.dev`. A push to `main` runs clippy + tests
  (incl. the Postgres adapter against a throwaway `postgres:16` CI service) and the worker
  target-matrix, then **auto-deploys to Fly on green** — no hand deploys. The CP serves the agent
  binary and generates the Colab bootstrap cell (`colab_cell`, now `install.sh`-based).
- **Real-time host telemetry** (chuk-compute M4): connected workers stream `sys/*` (GPU/CPU/memory)
  every few seconds; the dashboard shows live gauges + per-metric graphs (see the tabbed run view).
- **Modular internals**: the operator dashboard is now three inlined assets (`dash/{index.html,
  dash.css,app.js}`) behind a thin Rust handler, and the store's monolithic adapters are split into
  10 per-domain sub-traits (`WorkerStore`/`RunStore`/… → `store/{sqlite,postgres}/*.rs`); every
  store file is ≥90% line-covered (sqlite in-process, postgres via the CI Postgres service).

## Immediate next steps

1. ~~**Run stop/cancel + resume**~~ ✅ **done (2026-07-19).** `stop_run` signals the run's worker
   (`Cancel` → `JobKilled{Cancel}` → `Cancelled`; a queued or link-gone run finalises directly),
   `resume_run` re-queues a terminal run (train resumes from its latest checkpoint). REST
   `/runs/{id}/stop`+`/resume`, MCP tools, and dashboard buttons.
2. ~~**Heartbeat-timeout requeue**~~ ✅ **done (2026-07-19).** `last_seen` is now swept: an
   always-on reaper loop preempts + re-queues a resumable run on a worker silent past 10 min
   (spec §7, reusing the `detach` requeue; persistent workers keep their run per M3.2), and the
   scheduler no longer assigns new work to a worker unreachable past 90s. Both thresholds are
   pure decisions (`eligible_for_assignment`/`should_reap`) off the fleet's `heartbeat_age_s`.
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

**chuk-compute: worker & wire substrate (direction, 2026-07-19) — see
`docs/specs/chuk-compute-spec.md`**

The join-anywhere worker + per-run telemetry directions are folded into a larger reframe: the
rig is a **compute fabric**, not a training system. Two new crates form a permanently
compute-generic substrate under the training-first control plane — `chuk-compute-wire`
(serde-only protocol) + `chuk-compute-worker` (join-anywhere daemon). Naming discipline: the
daemon is a **worker**, never an "agent" (reserved for LLM/agentic workloads that run *on* the
fabric); the word "train" must never appear in the wire or worker crate (a lexical CI grep can
enforce it). The workload model is batch-vs-service — one `service: Option<ServiceSpec>` +
`needs`/campaigns admits evals, benches, cells, agents, and RL loops with **zero new wire
messages**; training stays the product, every other workload earns its place by serving the
training loop. §12 fixes the experiment-vs-service rule and the CP-fork tripwires so the
agent/MCP-deployment platform never colonizes the rig. Sequencing (spec §11):

- **M1 — extract the substrate** ✅ **done** — `chuk-compute-wire` (serde-only generic protocol,
  lexical guard, ~99% cov) + `chuk-compute-worker` (domain-free executor, depends only on the wire
  crate). CP translates `RunSpec`→`Job` and interprets `Artifact` events back into checkpoints
  (lineage merge moved CP-side). **Parity proven** on the local demo — a train run completes with
  lineage-complete checkpoints, and the E1 resume path yields slices `[[0,10],[10,40]]`. CI runs the
  guards. Single-target build retained (M2 changes that).
- **M2 — target matrix + bootstrap** ✅ **done** — the CP serves `/agent/{triple}` + `.sha256`
  + `/agent/version` + `/install.sh` (allowlisted targets; retired the hardcoded
  `/agent/linux-x86_64`). One rustup-style `install.sh` (uname → triple → download + verify +
  exec) is the single bootstrap the Colab cell + Vast onstart wrap. CI matrix cross-builds all
  three targets (zigbuild + macOS). **Proven: the Mac joins via `curl <CP>/install.sh | sh`.**
  Follow-up: bake the aarch64-musl + darwin CI artifacts into the deployed image (serves x86_64
  today; the Mac builds locally / from CI).
- **M3 — persistent worker class** ✅ **done.** M3.1: long-lived revocable worker tokens
  (`cw_`, hashed at rest) bound to a stable id; a persistent token → `Persistent` class + that id
  in HelloAck, so a Mac keeps one identity across reconnects/restarts; no lease ⇒ never torn down.
  M3.2 **survive-disconnect**: the worker's job supervisor + replay outbox outlive a session, so a
  persistent worker keeps its job running across a dropped socket (or the CP restarting) and
  replays buffered events on reconnect, trimmed by a `HelloAck` high-water the CP dedups by; the CP
  doesn't requeue a persistent worker's live job. M3.3 **self-update**: a version-mismatched
  persistent worker downloads → verifies → atomically replaces itself → re-execs (leased workers
  just exit). **All three proven** (bounce-the-CP survive-disconnect; forced-version self-update).
  `WorkerClass` is an enum so destroying a persistent worker is unrepresentable.
- **M4 — `sys/*` telemetry sampler** ✅ **done (2026-07-20).** One sampler task streams `sys/*`
  over the existing Metric channel (out-of-band: no `job_id`, not outboxed) — GPU via `nvidia-smi`
  (the distributed worker is static musl and can't `dlopen` NVML, so a subprocess is the portable
  choice; degrades to absent GPU metrics, not zeros) + CPU/memory via `sysinfo`; macmon MPS once the
  Mac is on. The CP ingests host samples into a pruned `worker_samples` window (latest-per-worker on
  the fleet, per-key series for sparklines); the dashboard's **System** tab renders a graph per
  metric. macmon (Apple-Silicon GPU) + the OOM/thermal gates + packing-util/MFU are the follow-ups.
- **M5 — service jobs** — `ServiceSpec` + registry + `needs` wiring + `Secret` env refs;
  LARQL-on-Mac as the first service, cell-runtime second.
- **M6 — campaigns + budgets** — `submit_campaign(template, matrix)` fan-out with CP-side spend
  budgets enforced at submit; the bench template's pinning gate (digest/seed in-spec).
- **M7 — first RL composition** — controller job + rollout campaign + cell-signed scoring
  against an existing train template. No new wire; RL is a composition.

Local enforcement stays worker-side (setsid process groups, SIGTERM→grace→SIGKILL, `kill_on_drop`,
wall enforced even with the control link down). Each milestone independently shippable, each proven
by a real workload (v11-scale run, tokenizer_bench campaign, broker eval). This **supersedes** the
individual "portable agent / telemetry" bullets; the related quick-win *Single-use join tokens* and
candidate *Requirements-aware assignment* / *generalized bootstrap* items feed into M2/M3.

**Spec gaps beyond the milestone headers**
- **Requirements-aware assignment** (S–M) — `pump()` assigns the oldest queued run to any
  idle worker with no fit check; add `requirements {min_vram_gb, labels}` + priority so a
  16 GB job never lands on a T4. A de-risking down-payment on M3 packing.
- **`eval` job kind + `deck` artifact** (M) — only `Shell`/`Train` exist; panel gates and
  E3's eval decks are unbuildable without it. Prereq for the M5 science story.
- **Learned `est_minutes`** (M) — store p90 of prior wall-times per `(entrypoint,
  config_hash)` so packing's fit rule is real (§16 open question; skip gpu-class first).

**Operational hardening**
- **Outbox delivery is at-least-once, not idempotent** (S–M, 2026-07-19 review finding) —
  the durable outbox retries a failed attempt, but if the experiments-server actually
  committed the write and only the HTTP response was lost, the retry can duplicate a
  result row, or (for `create`) hit a slug conflict that `try_create` currently treats as
  just another failure rather than looking up the existing run and completing the
  attachment. Needs a stable per-event id carried in the outbox row and either a
  fetch-existing-on-conflict path for `create`, or an idempotency key the
  experiments-server dedups results/checkpoints against.
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
- **Overview → drill-in screen hierarchy + a real-time telemetry tab** ✅ **done (2026-07-20).**
  The per-run view is now tabbed *progressive disclosure*: a light **Overview** (status strip +
  compact system gauges + last few events + recent-log tail, each drilling into its screen) plus
  dedicated **Training** (metrics chart + checkpoints), full **Logs**, full **Events**, and a
  **System** tab with a time-series graph per `sys/*` metric (GPU util / VRAM / temp / power / CPU /
  memory) over the retained window. The fleet table also shows live GPU-util per worker. Still open
  here: log search/filter/follow and a metrics explorer (pick/compare) — and **live push (SSE)**
  below so the detailed screens stream rather than 2s-poll.
- **Notifications** (S–M) — run complete/fail, gate trip, orphan kill, budget breach to a
  webhook/Slack/email sink; turns the rig from pull to push. Orphan kills only warn to the log today.
- **Live dashboard push (SSE)** (S–M) — the per-run view polls five endpoints every 2s; relay
  the already-streamed logs/metrics/`sys/*` over one SSE stream. Add a full historical-log view.
- **Submit-run / provision from the dashboard** (M) — it's read-only except teardown + access;
  an offers browser + submit form makes it usable without an MCP client.
- **Artifact & lineage browsing** (M–L) — `list_artifacts`/`artifact_lineage`/`register_artifact`
  ("what did this corpus contaminate"); the edges are latent in checkpoint `meta.json` but unindexed.
- **Multi-team scoping** (M) — `team_id` rides on users/keys but fleet/runs/spend ignore it; make
  runs/leases carry + filter by team to turn the scaffolding into real tenancy.
- **Cost preflight + `confirm_cost` on submit; Colab unit accounting** (S–M) — the spec's cost
  guardrails (§8): a shown estimate gate on submit, and Colab compute-units as their own currency.

**Integration depth**

*chuk-experiments-server pairing (from a 2026-07-19 architecture review). The framing: the
experiments-server owns the **research record** (what/why/concluded/evidence), the harness owns
**execution** (what ran, where, did it survive, checkpoints/cost). Keep the boundary
unmistakable — the harness reports **observations, never conclusions**.*
- **Distinct logical-run vs execution IDs** ✅ *done (2026-07-19) — supersedes ID #44.* The
  harness now mints `EXEC-YYYYMMDD-HHMMSS-NNNNN` execution ids (same store-backed 5-digit
  sequence, deliberately **not** the experiments-server's `RUN-…` shape), and a run carries an
  optional `experiment_ref` — the external parent reference to the logical `RUN-…` it realises.
  The reporting mirror uses it: with a ref it reports *into* that run (one-to-many intent→attempts)
  instead of minting a second run. `submit_run` (REST + MCP) takes `experiment_ref`; a shell probe
  is always unattached. *Still open (rec #3):* making the reference **required** on formal training
  jobs (vs the current opt-in), with an explicit scratch-run mode.
- **Durable reporting outbox** ✅ *done (2026-07-19).* Every mirror event (create/state/checkpoint/
  final-metric) is now persisted to a new `experiments_outbox` table (both SQLite and Postgres)
  before the first delivery attempt, instead of the old `tokio::spawn`-and-forget-on-failure shape.
  A failed attempt is recorded (`attempts`, `last_error`) and rescheduled with capped exponential
  backoff (30s → 1h ceiling, no give-up cutoff); `run_outbox_loop` sweeps due events every 30s.
  Real correctness fix along the way: a `state`/`checkpoint`/`result` event for a run whose `created`
  event hadn't landed yet used to silently no-op (`Ok(())`) — now it's a retryable failure, so it
  waits for `created` to land instead of being marked delivered without ever being sent. Verified
  live against a real local chuk-experiments-server (`experiments::live::
  outbox_recovers_after_experiments_server_was_unreachable`, `--ignored`): points a client at an
  unreachable address, confirms the event lands in the outbox instead of being lost, then replays it
  against a client pointed at the real server and confirms it's genuinely delivered.
- **Per-run attribution + per-user chuk-experiments-server key** ✅ *done (2026-07-19).* Both
  systems already have their own team/user model and self-service API-key issuance, so instead of
  one shared instance-wide `CHUK_EXPERIMENTS_API_KEY` mirroring every run under the same identity,
  each harness user can now link *their own* chuk-experiments-server key and have their mirrored
  runs attributed to it. `AuthContext` gained `owner_email` — always a real email (or the
  master-token sentinel) regardless of auth method, resolved from the already-fetched
  `ApiKeyInfo.created_by` for API-key calls rather than the bare key prefix `subject` carries.
  `runs.created_by` records who submitted each run; `users.experiments_api_key_encrypted` (new
  AES-256-GCM `crypto.rs` module, key from `CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY`) stores each user's
  linked key. `Experiments::bearer_for(run_id)` resolves the run's submitter's own key at mirror
  time, falling back to the shared default when any link in that chain is missing — the mirror
  never blocks or fails a run over this. New self-service `PUT`/`DELETE /api/me/experiments-key` +
  a dashboard card mirror the existing API-keys card's shape. Verified live
  (`experiments::live::bearer_for_prefers_the_submitting_users_own_linked_key`, `--ignored`) that
  resolution genuinely prefers the personal key over a working shared default. A real bug turned up
  in manual REST smoke-testing: `set_user_experiments_key` was `UPDATE`-only, so it silently no-oped
  (while still reporting success) for any caller with no pre-existing `users` row — true of the
  master-token sentinel, and of any first-time session sign-in before an admin grants them a role.
  Fixed as an upsert (defaulting a new row to `Role::Read`/the default team, never clobbering an
  existing role), and the master-token sentinel is now explicitly rejected by the link/unlink routes
  rather than silently accepted.
- **Required experiment reference on formal jobs** (S) — a `submit_run` for a formal experiment
  must carry an experiments-server experiment/run reference; keep an explicit *unattached scratch
  run* mode for quick probes.
- **One authoritative scheduler** (principle) — the **harness owns the real GPU execution queue**
  (it knows fit, resumability, lease headroom, budget); the experiments-server holds *intent*
  (planned → approved → dispatched → running → reported) and does not independently lease/schedule
  the same GPU work. Its generic `/v1/queue` stays available for simpler external harnesses (the
  opt-in pull mode below), but the direct pairing has one scheduler.
- **Native W&B logging** (M) — W&B is only a forwarded out-link today; the CP already ingests
  every metric batch, so creating a W&B run + streaming metrics is a natural add.
- **experiments-server pull mode** (M, narrowed 2026-07-20 — push is done, see
  `submit_run_from_experiment` above) — the harness itself polling `/v1/queue` and
  self-selecting eligible work, rather than being explicitly told (via
  `submit_run_from_experiment`) which experiments-server run to execute. An add-on that
  never replaces our own queue.
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
