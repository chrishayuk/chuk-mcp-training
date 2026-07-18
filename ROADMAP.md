# Roadmap

Status of `chuk-mcp-training` against the spec (`docs/specs/chuk-mcp-training-spec.md`),
plus the cross-cutting work that isn't a single milestone. Milestones are gated by the
proving experiments E0–E5 (spec §15): a milestone isn't done until its E is green.

## Milestones

| # | Scope | Code | Proven | Gate |
|---|-------|------|--------|------|
| **M0** | join loop, fleet, shell runs, log streaming | ✅ | ✅ **real Colab T4** (E0) | E0 |
| **M1** | train: code units, metrics, lineage checkpoints, resume | ✅ | ✅ **real Colab T4** (E1) — v11 trains, R2 checkpoints, resume passed | E1 |
| **M2** | leases, drain, provider-verified destroy, reconcile/orphan-kill, ledger | ✅ | 🟡 **mock provider locally** — live Vast pending | E2 |
| **M3** | packing scheduler (job classes, learned estimates, `submit_batch`, utilization) | ⬜ | — | E3 |
| **M4** | budget caps, watchdog gates, one-page dashboard | 🟡 ledger+`spend_status` only | — | E4 |
| **M5** | sweeps, panel gates, lazarus `load_checkpoint`, dynamics curve, Lambda driver | ⬜ | — | E5 |

## What's built beyond the milestone list

- **Rust control plane + agent**, Python MCP tool surface (thin REST client).
- **R2 artifact store** with presigned direct upload/download (spec §11.5/§12) — live.
- **Fly deploy**: `chuk-mcp-training.fly.dev`; the control plane serves the agent binary
  and generates the Colab bootstrap cell (`colab_cell`).
- Three storage/provider adapter seams: metadata store (SQLite), artifact store
  (R2/`file:`), provider registry (`mock`/`vast`).

## Immediate next steps

1. **M4 dashboard** — the one-page operator view (spec §9); high leverage now that real
   runs produce fleet/metrics/spend to look at.
2. **Hardening** (cross-cutting — see below); MCP client retry/backoff first.
3. **Live Vast E2** and **M3 packing** when there's rented-GPU pressure.

*(E0 and E1 are done — both proven on a real Colab T4, including E1's resume test.)*

## Then, by milestone

- **M3 · packing** — atomic vs resumable job classes, learned p90 estimates per
  (entrypoint, config, gpu-class), the `est × safety_factor` fit rule, resumable slices
  as filler, `submit_batch` packing preview, utilization metric. Gate E3: ≥85% util.
- **M4 · budgets + dashboard** — per-provider/label caps checked on provision/extend,
  watchdog gates (`isnan(loss)`, `no_improve`, `grad_norm` blowups) that checkpoint-then-
  stop, and the four-band dashboard (Fleet · Runs · Money · Health). Gate E4.
- **M5 · science** — sweeps (`submit_sweep`, cross-seed variance), panel gates evaluated
  from streamed metrics, lazarus `load_checkpoint` + tokenizer-hash verification, the
  first training-dynamics curve, Lambda driver. Gate E5.

## Hardening backlog

Things we've hit or know are soft, roughly by priority:

- **MCP client resilience** — `run_status`/`artifact_url` intermittently error against the
  shared-cpu-1x Fly machine during checkpoint ingest; add retry/backoff in the Python
  client and consider a larger machine or async checkpoint recording.
- **Code-unit build workflow** — `build_code_unit(local path)` only works against a local
  control plane (the deployed one can't see local files and has no git). Add git-in-image
  + build-from-git-URL, or a signed tarball-upload endpoint.
- **Large-object robustness** — the `file:` backend and the CP's own `/api/blob` still
  buffer whole objects in memory; fine now that R2 is the default (presigned, bytes bypass
  the CP), but the fs fallback and code-unit puts should stream.
- **Live provider validation** — run the real Vast E2 (rent 15 min, hang the agent, prove
  destroy) and confirm the Vast driver against the live API.
- **Retention + pinning** — enforce checkpoint keep-policies (`keep_last`/`keep_every`)
  and `pin` exemptions; nothing prunes R2 yet.
- **Auth hardening** — the dashboard is gated by app-level **Google OAuth** (email
  allowlist; session cookie), keeping MCP/agents on tokens. Remaining: Cloudflare Access
  as an outer layer if wanted (the Cloudflare plugin at
  `developers.cloudflare.com/agent-setup` can configure it); rotate join/API tokens;
  scope grants tighter.
- **Observability** — structured request logging, a `/metrics` endpoint, orphan/gate
  alerting beyond log lines.
- **Tests** — integration tests for the agent↔CP protocol and the lease state machine;
  today's coverage is unit tests + manual end-to-end runs.
