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
- **Dogfooding demo** — `scripts/demo.sh` spins up a local CP + mock workers running the
  (enriched) stub-trainer, so the dashboard fills with live data; isolated from prod.
- **Fly deploy**: `chuk-mcp-training.fly.dev`; the CP serves the agent binary and
  generates the Colab bootstrap cell (`colab_cell`).

## Immediate next steps

1. **M4 budgets + watchdogs** — the dashboard's done; the remaining M4 is per-provider/
   label caps checked on provision/extend, and watchdog gates that checkpoint-then-stop.
2. **Live Vast E2** — rent 15 min, hang the agent, prove provider-verified destroy.
3. **M3 packing** when there's rented-GPU pressure.
4. **R2 lifecycle permission** (see backlog) so the R2 hot copies actually auto-expire.

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
