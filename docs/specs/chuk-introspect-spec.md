# chuk-introspect — Specification v0.4

**Runtime introspection for training runs: capture on the GPU, reduce in the control
plane, read from lazarus, watch on the dashboard**
On-GPU capture library in **Python** (torch hooks, imported by the trainer) · transport
over the existing domain-free **chuk-compute** channels · reduction, storage, gates and
dashboard in **Rust** (control plane) · deep-dive analysis in **chuk-mcp-lazarus** (Mac)
· j-space (Jacobian-lens) tracking as a first-class probe family

v0.4 changes (review pass on v0.3): **row targets get the same self-vs-pinned contract
as whitening** — `row_targets: "model_top1"` regenerates completions with the *current*
model, so target tokens, positions, and filler masks drift across snapshots and a
cross-step row curve confounds "`J_ℓ` changed" with "the targets changed under it"; new
`row_target_reference: "self" | "pinned:<ckpt>"` (pinned ⇒ completions generated once
by the pinned checkpoint, teacher-forced thereafter), realized completions recorded in
the manifest in both modes, cross-step row claims require pinned (§3, §7, §8.7).
Pinning also repairs tolerant-mode pass-2 reproducibility — a top-1 flip within kernel
tolerance is a discrete target change, not a `rel: 1e-3` perturbation (§7). §4.1's
pass-2 mechanics rewritten to match `row_positions` (v0.3 changed the recipe but left
the mechanics at one-backward-per-prompt): a completion-acquisition stage precedes the
row backwards, which run per completion position; E22's per-layer cost figure predates
the ~`completion_len`× target count and is re-measured at EI1 (§7). ProbePlan bumped to
**v3** — `row_positions` is validation-load-bearing; v2 plans are rejected at submit
(§3). §5.1's normative key segment order actually lands (v0.3's changelog claimed it;
the section text never changed). Minor: torch ≥ 2.1 floor stated (§4); `grad_norm`
defined as the post-accumulation value under gradient accumulation (§4.2); the
filler-survival discount is a named validator constant (§3.2); the plan example's
overhead comment points at the rolling baseline (§13); EI4 requires pinned row targets
for its cross-step claim (§15).

v0.3 changes (review pass on v0.2): strict-mode cuBLAS determinism moved to **process
launch** — `CUBLAS_WORKSPACE_CONFIG` must be set before cuBLAS initializes, so per-pass
cuBLAS strictness is impossible; `jobspec.rs` now sets it in the job environment when
the plan declares strict, and the small whole-run cost is stated honestly (§3.1). The
j-space row recipe is made explicit about positions (`row_positions`: rows at every
completion position, filler-excluded) and **plans are validated at submit** for nominal
rows ≥ `min_rows` — v0.2's example plan produced ~40 rows against a 512-row floor and
would have flagged every snapshot (§3.2, §7). Pulse grad capture names its mechanism
(`register_post_accumulate_grad_hook` — the context manager has no backward/step seam
inside `train_step` and doesn't need one, §4.1/§4.2). Skip-first coupling documented:
skipping pass 2 degrades all whitened metrics at that step to raw, by design (§8.5).
Corpus subsampling is stratified per split (§3 rule 1). The overhead baseline
re-measures on a rolling window (§13). Metric key segment order is normative (§5.1).

v0.2 changes: snapshot split into an explicit two-pass protocol (T3 gradient rows
cannot run under `no_grad` — v0.1 §4.1 contradicted itself); whitening nonstationarity
promoted to an §8 invariant (spectrum emitted beside every z; pinned-reference
whitening for cross-step claims); whitening covariance sourced from `jspace_rows` with
a minimum row count (last-position `hidden` at k=40 is rank-39 at d=2560 — unusable as
a background covariance); full `J_ℓ` fits dispatchable as **eval jobs** off the
checkpoint event (`full_fit_mode`) so milestone fits stop stalling the training GPU;
degradation policy flipped to **skip-first** (tier gaps are honest, thinned-corpus
points are not — corpus thinning removed entirely); determinism claim scoped to a
declared mode (strict deterministic kernels during capture passes, or documented
tolerance); sklearn dropped from the trainer library (closed-form ridge/LDA in torch);
corpus governance **resolved**: one lab-standard frozen corpus, programme corpora only
ever additive.

v0.1: initial spec. Synthesizes three proven codebases: the chuk-mlx introspection +
batching frameworks (`chuk-mlx/src/chuk_lazarus/introspection/`, `data/batching/`),
the chuk-mcp-lazarus tool surface (`chuk-ai/mcp-servers/chuk-mcp-lazarus`), and the
E22/E23/E24 j-space campaign (`chris-experiments/fleet/E22_jspace_transfer/`, cloned
`jacobian-lens/`). This spec is the **deliberate scope expansion** that
chuk-mcp-training-spec §10 reserves: live introspection of a training loop on a remote
GPU, not just Mac-side reads of finished checkpoints.

---

## 0. Implementation status (v0.4, 2026-07-21)

| Milestone | State | Gate | Proven |
|-----------|-------|------|--------|
| **I0** pulse metrics: `introspect/*` namespace end-to-end (trainer → metrics → dashboard → gates) | ⬜ not started | EI0 | — |
| **I1** probe plan + snapshot artifacts: `ProbePlan` contract, `$CHUK_PROBE_PLAN`/`$CHUK_PROBE_DIR`, `introspection` artifact class, two-pass step-indexed capture on the probe corpus | ⬜ not started | EI1 | — |
| **I2** control-plane reduction (Rust): PR / overlap / drift / whitened-alignment computed server-side from snapshot artifacts; Introspection dashboard tab | ⬜ not started | EI2 | — |
| **I3** lazarus attach: `attach_run` remote mode — lazarus analysis math over harness-captured artifacts, step axis in results | ⬜ not started | EI3 | — |
| **I4** j-space over training: cheap rows every snapshot, paper-faithful `J_ℓ` at milestone checkpoints (inline + eval-job modes), nulls enforced, curves dashboarded | ⬜ not started | EI4 | — |
| **I5** introspection sweeps: cross-run comparison (probe curves across a sweep's children) | ⬜ not started | EI5 | — |

Nothing is built. The load-bearing decision already made upstream: the worker and wire
crates are **permanently domain-free** (chuk-compute-spec §1; enforced by the lexical CI
guard `chuk-compute-wire/tests/no_domain_vocabulary.rs`). Everything below rides
existing generic channels — a new metric namespace, a new artifact class string, two new
script-contract env vars. **Zero wire changes in the entire spec.** (The v0.2 eval-job
fit mode reuses the existing batch-eval job type — still zero wire changes.)

---

## 1. Overview

`chuk-introspect` is the framework that lets a training run be *watched from the
inside*: per-layer training dynamics streamed as metrics, mechanistic snapshots
(activations, logit lens, probes, geometry, j-space) captured on a fixed probe corpus at
checkpoint cadence, reduced to curves by the control plane, rendered on the dashboard,
and drillable from chuk-mcp-lazarus on the Mac.

Today the introspection story is inference-only and Mac-only:

- **chuk-mlx** (`chuk_lazarus.introspection`) has the strongest capture toolkit — hooks,
  logit lens, ablation, patching, steering, probes, circuit sweeps — but it is MLX-bound
  and attaches to models via four MLX-specific hacks (manual forward unroll in
  `hooks.py::ModelHooks.forward`, `__call__` monkey-patching in `moe/hooks.py`,
  layer-list swapping in `interventions.py`, weight mutation in `ablation/adapter.py`)
  because MLX has no hook registry.
- **chuk-mcp-lazarus** exposes ~80 of those primitives as MCP tools, but assumes the
  model is a live MLX module *in the same process* (`model_state.py::ModelState`), and
  every result is a point-in-time snapshot with **no step axis**. Its own ROADMAP
  ("Training Integration", "The Fine-Tuning Delta Problem") already aspires to training
  analysis but only as post-hoc two-model comparison.
- **j-space** (E22/E23/E24) built the read machinery for Anthropic's Jacobian-lens
  workspace on open models — and banked the harder lesson: three successive "discoveries"
  were each killed same-day by a pre-registered adversarial null. Every j-space artifact
  in the lab is a single-checkpoint frozen read; nothing tracks it over training.

This spec closes all three gaps at once, on the harness's own terms:

1. **Capture moves onto the training GPU** — a `chuk-introspect` Python library the
   trainer imports, using real `register_forward_hook`s on the torch model (the MLX
   hacks become unnecessary), driven by a fingerprinted `ProbePlan` artifact.
2. **The step axis becomes first-class** — every introspection result is keyed by
   `(run_id, step)`, streamed and stored exactly like training metrics and checkpoints.
3. **j-space becomes a probe family** — refit per checkpoint against a frozen probe
   corpus, with the E22-banked nulls (noise floor, whitened null, capability ceiling)
   attached to every curve as a schema requirement, not a convention.
4. **lazarus becomes the analysis layer, not the model owner** — a new attach mode reads
   harness-captured artifacts; its backend-agnostic math (probing, directions, geometry,
   comparison kernels, `attribution_sweep` aggregation) transfers unchanged.

### Design principles

1. **The worker never learns what a probe is.** Introspection is expressed entirely
   through the generic channels the worker already has: the `Metric` stream (new
   `introspect/` namespace, exactly like `sys/`), the `Artifact` stream (new open-string
   class `"introspection"`, exactly like `"checkpoint"`), and output rules. The domain
   lives in the trainer (Python) and the control plane (Rust). This is the chuk-compute
   fork tripwire and it holds here.
2. **Capture where the tensors are; reduce where the dashboard is.** Hooks and backward
   passes can only run inside a process that owns the model (the trainer for pulse and
   snapshots; a batch eval job for offloaded milestone fits). Everything downstream of
   the captured artifact — PR, principal-angle overlap, whitening, nulls, downsampling,
   gates, rendering — is small-matrix math on k×d / n_tok×d fp16/fp32 blocks and runs
   in the **Rust control plane** (`ndarray`/`faer`), the same way `complete_meta`
   already does checkpoint lineage server-side. No Python analysis hop between worker
   and dashboard.
3. **Plan as contract.** The `ProbePlan` is a versioned, fingerprinted artifact (the
   `BatchPlan` philosophy from `chuk-mlx/docs/batching.md` applied to introspection):
   same plan + same checkpoint ⇒ reproducible capture under the plan's declared
   determinism mode (§3.1). Probe corpora are content-hashed and frozen per run — a
   probe curve is meaningless if the corpus drifted under it.
4. **Config-in / result-out, typed everywhere.** Every capture family has an explicit
   config and an explicit result schema (Pydantic in the trainer library, serde in
   `chuk-train-proto`, mirrored Pydantic in the MCP client) — chuk-mlx's
   `*Config`/`*Result` discipline, house rules (no magic strings/numbers).
5. **Every signal ships with its null — and the null's own stick is visible.** E22's
   verdict is the constitution here: a probe metric without its pre-registered
   adversarial control (within-split noise floor, whitened/spectrum-matched null,
   capability ceiling) is not rendered as a finding. The schema carries the null next
   to the value; the dashboard plots them together. v0.2 addition: whitened metrics
   also carry the whitening spectrum, because over training the background covariance
   is nonstationary and a z-score in a drifting metric is its own artifact class (§8).
6. **Cheap always-on, expensive on milestones, heavy off-GPU.** Tiered cadence: pulse
   scalars ride the training step (~free), snapshot probes run at checkpoint cadence on
   the probe corpus (seconds–minutes, two passes), heavy fits (full `J_ℓ`) run at
   milestone checkpoints only — inline on single-session backends, or as a **separate
   eval job** on fleet backends so the training GPU never stalls (§7.1) — each tier
   with an explicit overhead budget the plan declares and the harness enforces.
7. **Read-only during training.** chuk-introspect observes; it never steers, ablates, or
   patches a live training run. Interventional work stays lazarus-side on loaded
   checkpoints. (A future service-job mode may relax this — §17.)
8. **Runs standalone; every tier optional.** No plan ⇒ the trainer runs exactly as
   today. A run with pulse metrics only, or snapshots only, is valid. Lazarus, the
   experiments mirror, and the dashboard are consumers, never dependencies.

---

## 2. Architecture

```
        Colab / Vast GPU                        Fly.io                        Mac
┌──────────────────────────────┐   ┌────────────────────────────┐   ┌──────────────────┐
│ trainer process (Python)     │   │ chuk-train-controlplane    │   │ chuk-mcp-lazarus │
│  ├─ training loop (torch)    │   │  (Rust)                    │   │  (MLX + analysis)│
│  └─ chuk-introspect lib      │   │  ├─ ingest: introspect/*   │   │  ├─ attach_run ──┼──▶ REST
│     ├─ pulse hooks ──────────┼─┐ │  │   metrics + artifacts   │   │  │  (read arts)  │
│     ├─ snapshot pass 1       │ │ │  ├─ reduce (ndarray/faer): │   │  └─ existing 80  │
│     │  (no_grad: T1 capture) │ │ │  │   PR·overlap·whiten·null│   │     tools on     │
│     ├─ snapshot pass 2       │ │ │  ├─ gates (introspect/*)   │   │     checkpoints  │
│     │  (grad: T3 rows)       │ │ │  ├─ store keys + retention │   └──────────────────┘
│     └─ jlens fits (inline    │ │ │  ├─ eval-job dispatch ─────┼─┐ ┌──────────────────┐
│        mode only)            │ │ │  │  (milestone J_ℓ, §7.1)  │ │ │ chuk-train MCP   │
│          │                   │ │ │  └─ dashboard: Introspect  │ │ │  (Python, thin)  │
│          ▼                   │ │ │      tab (layer×step)      │ │ │  probe tools §10 │
│  $CHUK_METRICS (jsonl) ──────┼─┤ └────────────▲───────────────┘ │ └──────────────────┘
│  $CHUK_PROBE_DIR/step_N/ ────┼─┤              │ REST            │
└──────────────────────────────┘ │              │                 ▼
┌──────────────────────────────┐ │      R2: introspect-hot/…   batch eval worker
│ chuk-compute-worker (Rust)   │ │      Drive: archive tier    (loads milestone ckpt,
│  DOMAIN-FREE — tails metrics,├─┘                              fits J_ℓ, ships same
│  ships artifacts. Unchanged. │                                artifact key)
└──────────────────────────────┘
```

### 2.1 The Rust / Python boundary

This is a first-class design axis. The rule: **Python where the autograd graph is; Rust
everywhere else.**

| Concern | Owner | Why |
|---|---|---|
| Hooks, forward/backward capture, jlens fits | **Python** — `chuk-introspect` lib inside the trainer process (or a batch eval job for offloaded fits) | needs a live torch model + autograd; nothing else can do it |
| Transport (tail metrics, ship artifacts, retry, seq/replay) | **Rust** — `chuk-compute-worker`, unchanged | already built, domain-free, proven |
| Proto types, constants, storage keys | **Rust** — `chuk-train-proto` (`introspect.rs`) | house pattern; single source of truth, mirrored to Pydantic |
| Ingest, validation, retention | **Rust** — control plane (`hub.rs` artifact arm + `IntrospectStore`) | same seam as checkpoint ingest |
| Reduction: PR, principal-angle overlap, whitening, nulls, drift, downsampling | **Rust** — control plane, `ndarray`/`faer` (f32/f64 accumulation — never fp16 linear algebra) | inputs are k×d / n_tok×d fp16 blocks (KBs–MBs); SVD/eig at d≤4096 is milliseconds; keeps the dashboard live with no Python hop |
| Milestone `J_ℓ` fit dispatch | **Rust** — control plane, off the checkpoint-promotion event | reuses the existing batch-eval job type; no new machinery |
| Gates on introspection metrics | **Rust** — existing gate engine (`gate.rs`), pointed at `introspect/*` keys | already evaluates on every metric ingest |
| Dashboard rendering | **Rust** — CP-served dash, new tab | System tab is the exact precedent |
| Deep-dive, interventional, exploratory analysis | **Python** — lazarus on the Mac | the 80-tool surface, torch-native probe math, human-in-the-loop |
| Heavy offline recompute (re-fit lens variants, big PCA) | **Python** — lazarus or a batch `eval` run on a GPU worker | beyond CP budget; artifacts make it reproducible |

Explicitly rejected: a Rust on-GPU capture agent (`tch`/`candle` cannot hook a live
Python-owned torch training loop) and a Python reduction service between CP and
dashboard (adds a process, breaks "runs standalone").

### 2.2 What each layer must never do

- The **wire/worker** must never gain a message, field, or word for probes. (CI-enforced.)
- The **control plane** must never load a model or touch a GPU tensor larger than a
  snapshot block; anything needing a forward pass is trainer-side or a batch job.
- **chuk-introspect** must never write outside `$CHUK_PROBE_DIR`/`$CHUK_METRICS`, block
  the training step beyond its declared budget, or mutate the model (hooks are
  read-only; row/jlens backward passes run on a cloned-or-frozen eval context,
  `torch.no_grad` everywhere except the grad-enabled capture passes, which touch only
  activations — never parameters).

---

## 3. The ProbePlan — plan as contract

A run opts into introspection by attaching a **ProbePlan** to `submit_run` (or a sweep).
The control plane stores it, stamps its fingerprint into run lineage (like
`complete_meta` does for checkpoints), and hands it to the trainer via the script
contract. The plan is the introspection analogue of chuk-mlx's `BatchPlan`: a
fingerprinted, versioned artifact that is the *contract*, not a config suggestion.

```jsonc
// ProbePlan v3 (stored CP-side; delivered as JSON at $CHUK_PROBE_PLAN)
// v2 plans are REJECTED at submit: v3 added validation-load-bearing fields
//   (row_positions, row_target_reference) and nothing deployed ever spoke v2
{
  "version": 3,
  "fingerprint": "pp_9f2c…",            // content hash of everything below
  "model": { "d_model": 2560, "n_layers": 34, "dtype": "auto" },  // fp16 on T4 (no bf16), bf16 on Ampere+

  "determinism": {                       // §3.1 — capture reproducibility, declared not implied
    "mode": "strict",                    // strict: deterministic algorithm selection during capture
    "tolerance": null                    //   passes + CUBLAS_WORKSPACE_CONFIG set at PROCESS LAUNCH
  },                                     //   by jobspec (whole-run, §3.1); tolerant: {"rel": 1e-3}

  "corpus": {                            // FROZEN probe corpus — the fixed measuring stick
    "id": "probe-corpus/lab-std-v1",     // THE lab-standard corpus (§3 rule 4) — content-addressed
    "sha256": "…",
    "n_prompts": 100, "seq_len": 128,
    "splits": { "a": 50, "b": 50 }       // disjoint halves — feeds the stability/noise-floor nulls
  },

  "pulse": {                             // Tier 0 — rides the training step, ~free
    "every_steps": 1,
    "metrics": ["act_norm", "grad_norm", "dead_frac", "logit_entropy"],
    "layers": "all"
  },

  "snapshot": {                          // Tier 1/2/3-rows — TWO PASSES at ckpt cadence (§4.1)
    "on_checkpoint": true,               // piggyback the existing checkpoint schedule
    "corpus_subsample": 40,              // fixed for the whole run — never thinned (§8.5);
                                         //   stratified per split (§3 rule 1): 40 of 50/50 ⇒ 20/20
    "layers": [4, 12, 20, 26, 32],       // or "evenly_spaced:5" (chuk-mlx LayerStrategy)
    "position": "last",                  // T1 readout positions; NOT the whitening source (§3.2)
    "capture": ["hidden", "logit_lens", "linear_probe"],
    "probes": [                          // linear probes, spec'd not code'd; closed-form fit (§4.2)
      { "name": "content", "labels_from": "corpus_meta.category" }
    ]
  },

  "whitening": {                         // §3.2 — the covariance stick, made explicit
    "source": "jspace_rows",             // n_tok×d — never last-position hidden (k=40 ⇒ rank-39 at d=2560)
    "min_rows": 512,                     // below this the CP refuses whitened z (emits raw + flag)
    "reference": "self",                 // "self" | "pinned:<ckpt_id>" — pin for cross-step z claims
    "emit_spectrum": true                // whiten_pr + whiten_trace beside every z (§8.3)
  },

  "jspace": {                            // Tier 3 — see §7
    "rows_on_snapshot": true,            // gradient rows in snapshot pass 2 (grad-enabled)
    "row_targets": "model_top1",         // E22 recipe: model's own clean completions
    "row_target_reference": "self",      // "self" | "pinned:<ckpt_id>" — §7: self regenerates
                                         //   completions per snapshot (targets drift with the
                                         //   model); pin + teacher-force for cross-step row claims
    "row_positions": {                   // what makes min_rows satisfiable (§3.2): rows come from
      "completion_len": 20,              //   EVERY completion position, filler-excluded — here
      "exclude_filler": true             //   40 × 20 = 800 nominal, ≥512 expected post-filter;
    },                                   //   validated against whitening.min_rows at submit
    "full_fit_mode": "auto",             // "inline" | "eval_job" | "auto" (§7.1)
    "milestone_every_ckpts": 10,
    "fit": { "dim_batch": 8, "skip_first": 16, "source_layers": [4, 12, 20, 26, 32] }
  },

  "budget": {                            // enforced (§13); degradation is skip-first (§8.5)
    "pulse_overhead_pct_max": 2,         // vs. the rolling clean-step baseline (§13)
    "snapshot_forward_seconds_max": 60,  // pass 1 (no_grad, T1)
    "snapshot_rows_seconds_max": 120,    // pass 2 (grad, T3 rows) — the expensive half, budgeted apart
    "milestone_seconds_max": 3600        // inline fits only; eval-job fits bill their own lease (§13)
  }
}
```

Rules:

1. **The corpus is frozen per run.** Same corpus, same subsample seed, same subsample
   *size*, every snapshot — otherwise curves compare apples to oranges. Subsampling is
   **stratified per split**: a 40-subsample of 50/50 splits is exactly 20/20, so the
   overlap noise floor always compares balanced halves — an unstratified (deterministic
   but skewed) draw like 27/13 would quietly bias the null. Realized per-split sizes
   are recorded in the manifest. Corpus artifacts are content-addressed and cached on
   the worker exactly like code units (§11.1 of the training spec).
2. **Fingerprint into lineage.** `meta.json` of every checkpoint produced by a probed
   run carries `probe_plan: pp_…`, so any later offline recompute (lazarus, batch eval,
   an eval-job fit) can verify it is measuring with the same stick.
3. **No plan ⇒ no-op.** `$CHUK_PROBE_PLAN` unset ⇒ chuk-introspect is inert; the
   trainer contract is unchanged from today's five touch-points.
4. **One lab-standard corpus.** (Resolved from v0.1 §17.5.) `probe-corpus/lab-std-v1`
   is the default and the comparability anchor across every run, forever — E24's
   finding that readout quality saturates by ~100 prompts makes the cost of a universal
   stick negligible. Programme-specific corpora may be **added** as second frozen
   sticks in the same plan (each with its own key suffix in every emitted metric); they
   never replace the standard one. A curve on lab-std-v1 from any run is comparable to
   any other, by construction.

### 3.1 Determinism, scoped

"Bit-identical" is a promise CUDA only keeps on request. The plan declares one of two
modes, and the claim in principle 3 is scoped to it:

- **`strict`** — two mechanisms with two different scopes, and the spec is honest
  about both. *Algorithm selection* (`torch.use_deterministic_algorithms(True)`) is
  runtime-toggleable: enabled for capture passes (snapshot pass 1, pass 2, inline
  fits) and restored to the trainer's own settings afterwards — this, the larger
  cost, is capture-scoped. *Deterministic cuBLAS* is not toggleable:
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set **before cuBLAS initializes**, i.e.
  at process launch — so when a plan declares strict, **`jobspec.rs` sets the env var
  in the job environment** (one line in `script_environment`; this is a jobspec
  change, not a library one, and lands before EI1). Strict mode therefore carries a
  small whole-run cuBLAS workspace cost on every training step; it is measured as
  part of the run-start baseline and emitted, not hand-waved. Same plan + same
  checkpoint + same backend ⇒ bit-identical artifacts. If an op in the capture path
  has no deterministic kernel on the backend, the library fails the snapshot loudly
  (`introspect/probe_error`) rather than silently downgrading — a strict plan that
  can't be strict is a bug, not a fallback.
- **`tolerant`** — capture runs with the trainer's kernel settings; artifacts are
  reproducible within the declared relative tolerance, and the manifest records the
  mode so no downstream consumer mistakes it for strict.

Corpus order and subsampling are derived from the plan fingerprint + step in both
modes — never wall-clock or RNG state shared with training.

### 3.2 The whitening stick, made explicit

v0.1 left two metrology holes that E22 existed to catch. Fixed as plan-level contract:

- **Source.** The background covariance for whitening comes from `jspace_rows`
  (n_tok×d — hundreds to thousands of rows per layer), never from last-position T1
  `hidden` blocks: 40 samples give a rank-39 covariance in a d=2560 space, and a
  "spectrum-matched null" estimated there is fiction. `min_rows` is enforced CP-side —
  below it, the reduction emits the raw (unwhitened) value with an explicit
  `insufficient_rows` flag instead of a fake z. The flag is a runtime guard, **not the
  default experience**: at submit the CP computes the plan's nominal row yield
  (`corpus_subsample × completion_len × EXPECTED_FILLER_SURVIVAL` — the discount is a
  named conservative constant in the validator, 0.6 at v1, per house rules not
  folklore) and rejects plans that cannot clear `min_rows` — a plan whose whitening floor
  is arithmetically unreachable is a config error, caught before a single GPU-second
  is spent. (v0.2's own example failed this check: `model_top1` at final position
  yields ~1 row/prompt — ~40 rows against a 512 floor. Hence `row_positions`.)
- **Reference.** Whitening against each snapshot's *own* covariance means the stick
  itself drifts over training: z at step 1k and z at step 50k are scores in different
  metrics, and a rising curve could mean rising alignment *or* a shrinking background
  spectrum. `reference: "self"` remains the default for within-snapshot comparisons,
  but any cross-step claim must either (a) read the emitted spectrum metrics
  (`whiten_pr`, `whiten_trace`) alongside the z-curve — the dashboard draws them in
  the same band — or (b) use `reference: "pinned:<ckpt>"`, which whitens every
  snapshot against one fixed checkpoint's covariance (E22's actual setup, extended
  with a step axis). Both are first-class; the schema records which was used.

---

## 4. On-GPU capture: the `chuk-introspect` Python library

A small, dependency-light library the trainer imports (ships inside the code unit, or as
a declared dependency in `unit.toml`). torch is the only heavyweight dependency — floor
**torch ≥ 2.1**, where `register_post_accumulate_grad_hook` (the pulse tier's
load-bearing mechanism, §4.2) first shipped — and the probe fits that pulled in sklearn
in chuk-mlx are closed-form here (§4.2). It is the
torch/CUDA generalization of `chuk_lazarus.introspection`, keeping the shape and
dropping the MLX hacks:

| chuk-mlx (MLX) mechanism | chuk-introspect (torch) replacement |
|---|---|
| `ModelHooks.forward` manual forward unroll (`hooks.py:398`) | `nn.Module.register_forward_hook` on real modules; `output_hidden_states=True` zero-hook path for residuals |
| `moe/hooks.py` monkey-patches `mlp.__call__` | forward hook on the router submodule |
| `interventions.py` swaps `model.layers` entries (reference-aliasing hazard) | hook handles (`handle.remove()` in `finally`) — no structural mutation |
| immutable-tensor `mx.concatenate` position writes | not needed (capture is read-only; interventions are out of scope during training) |
| `mx.stop_gradient` / periodic `mx.eval` | `.detach()` under `torch.no_grad()` for pass 1; scoped `requires_grad` on source-layer activations for pass 2 |
| sklearn probe fits in `probes/` | closed-form ridge / LDA on the captured k×d block, in torch (§4.2) |
| per-prompt Python loop in `circuit/collector.py` | true GPU batching of the probe corpus — the major throughput win |

Kept from chuk-mlx wholesale (they are already framework-agnostic patterns):
`CaptureConfig`-style layer/position selection (`LayerStrategy`, `PositionSelection`),
the `Config → Service → Result` layering, `CollectedActivations`-style
safetensors + JSON-sidecar artifacts, `ModelAccessor` duck-typing for architecture
variance (HF naming schemes), and the circuit pipeline's staged, resumable
dataset → collect → reduce discipline.

### 4.1 API shape (trainer-side) — the two-pass snapshot

The v0.1 snapshot claimed to produce T3 gradient rows "under no_grad", which is
impossible. The snapshot is now explicitly **two passes**, separately budgeted:

```python
from chuk_introspect import Introspector

intr = Introspector.from_env()            # reads $CHUK_PROBE_PLAN, $CHUK_PROBE_DIR, $CHUK_METRICS; inert if unset

for step, batch in enumerate(loader):
    with intr.pulse(model, step):         # T0: on entry (iff due), registers forward hooks (act
        loss = train_step(model, batch)   #   stats) + per-param post-accumulate-grad hooks (grad
                                          #   norms) — the hooks fire inside train_step's own
                                          #   backward; on exit, scalars are appended to
                                          #   $CHUK_METRICS as introspect/* and handles removed

    if ckpt_due(step):
        write_checkpoint(...)             # unchanged
        intr.snapshot(model, step)        # pass 1 (no_grad): T1 capture — hidden, lens, probes
                                          # pass 2 (grad):   T3 rows — forward+backward on the
                                          #   probe corpus; only source-layer activations retain
                                          #   grad; parameters never touched; own budget cap
                                          # writes $CHUK_PROBE_DIR/step_N/… then .ready
        if intr.milestone_due(step) and intr.full_fit_inline():
            intr.fit_jlens(model, step)   # T3 full fit — INLINE MODE ONLY (§7.1); in eval_job
                                          #   mode the CP dispatches this off the checkpoint
                                          #   event and the trainer never sees it
```

Pass 2 mechanics: first a **completion-acquisition stage** — under
`row_target_reference: "self"` the model greedily generates `completion_len` tokens
per prompt (batched; ~`completion_len` cheap forwards); under `"pinned:<ckpt>"` the
pinned checkpoint's recorded completions are teacher-forced instead (§7). Then a fresh
forward over prompt+completion with `requires_grad` enabled only on the hooked
source-layer outputs, and one backward per target logit per **completion position**
(batched across prompts — nominal `corpus_subsample × completion_len` targets, ~800 in
the §3 example, so ~`completion_len` backward batches; each backward yields the row at
every hooked source layer simultaneously), rows detached and
unembed-row-orthogonalized on the GPU before the fp16 host copy. The realized
completions (token ids, filler mask) go in the manifest in both modes. Parameters keep
`requires_grad` as the trainer left them but receive no `.grad` accumulation (grads
taken w.r.t. activations via `torch.autograd.grad(inputs=…)`, and the optimizer state
is untouched — asserted, not assumed: the library snapshots
`param.grad is None`-consistency before/after and fails loudly on violation).

Five properties the library guarantees:

1. **Inert without a plan** — `from_env()` returns a no-op object; zero overhead.
2. **Budget-honest, skip-first** — measures its own wall-clock, emits
   `introspect/overhead_pct` and per-pass timings, and degrades by **skipping whole
   tiers** in reverse-cost order (pass 2 rows → pass 1 lens/probes → pulse never),
   logging every skip as a metric. It never thins the corpus: a gap in a curve is
   honestly missing, while a lower-N point under the same key looks comparable and
   isn't (§8.5).
3. **Crash-isolated** — a probe failure logs `introspect/probe_error` and skips; it
   never takes the training step down with it.
4. **Deterministic per the declared mode** — strict mode is bit-identical on a given
   backend; tolerant mode is reproducible within the declared tolerance; corpus order
   and subsample come from the plan fingerprint + step in both (§3.1).
5. **Ready-marker protocol** — snapshot dirs are written then sealed with `.ready`
   only after *both* passes complete (or a pass is skipped-and-logged), identical to
   checkpoints, so the worker's `on_appearance` output rule ships complete directories
   only.

### 4.2 Capture families

- **T0 pulse (per training step, hooks on the live batch):** per-layer activation norm,
  per-layer/component grad norm, dead-neuron fraction, logit entropy, router
  load/entropy on MoE. Grad norms are captured via
  `torch.Tensor.register_post_accumulate_grad_hook` — fires per-parameter after its
  grad is accumulated in backward, before the trainer calls `optimizer.step()` — which
  is what makes the §4.1 shape work at all: `backward()` and `step()` both live inside
  `train_step`, so the context manager has **no seam** between them and doesn't need
  one; the hooks do the interposition, the CM only registers on entry and
  emits/removes on exit. Under gradient accumulation the hook fires once per
  micro-batch backward; each firing overwrites the stored norm, so the value emitted
  at CM exit is the **post-accumulation** grad norm — per-micro-batch values are not
  emitted, and reading `param.grad` at exit is not an option (`zero_grad` may already
  have run inside `train_step`). (Plain forward hooks cover the activation-side
  stats; they never see gradients.) Scalars are computed on-GPU (`.norm()` etc.) and transferred
  at logging cadence — no per-step k×d host copies. Emitted as `introspect/…` records
  in `$CHUK_METRICS` — no artifacts, no extra forward.
- **T1 lens (snapshot pass 1, no_grad probe-corpus forward):** last-position residuals
  per selected layer (`hidden.safetensors`, k×d fp16), calibrated logit-lens top-k +
  tracked-token ranks (JSON), linear-probe accuracy per layer per registered probe —
  the fit is **closed-form ridge (or LDA) in torch** on the captured k×d block
  (`(XᵀX + λI)⁻¹Xᵀy` at k=40, d=2560 is sub-millisecond; no sklearn in the trainer
  environment), attention entropy summaries.
- **T2 geometry (derived, mostly CP-side — §6):** PCA basis + participation ratio,
  split-A/split-B stability overlap, representation drift vs previous snapshot and vs a
  pinned reference checkpoint (the lazarus `compare_*` kernels with a step axis).
- **T3 j-space (§7):** gradient rows in snapshot pass 2; full `J_ℓ` at milestones
  (inline or eval-job).

---

## 5. Data model and flow

### 5.1 Metrics — the `introspect/` namespace

Third metric shape alongside training metrics and `sys/*`, riding the **training-metric
path** (job-scoped, step-indexed): records land in `$CHUK_METRICS`, the worker tails and
ships them as `Metric{job_id, step, values}`, the CP appends via
`MetricStore::append_metrics`, `run_metrics`/dashboard/gates read them back downsampled.
Namespace constant `INTROSPECT_METRIC_PREFIX = "introspect/"` in `chuk-train-proto`
(the worker never sees it as anything but a string key).

Key grammar (all named constants, no free-form strings). Segment order is
**normative** — both sides of the seam (Rust emitters and reducers, Python client,
dashboard) parse positionally, never by substring search:

```
introspect/<family>[/<qualifier>]/L{i}[@<corpus>]
```

`family` comes from the closed set below; `qualifier` exists only where the family
declares one (`probe_acc` → probe name, `lens_rank` → tracked token,
`jspace_align_z` → reference id); `L{i}` is the layer segment, absent only on the
model-global keys (`logit_entropy`, `overhead_pct`, `pass1_seconds`, `pass2_seconds`,
`tier_skipped`, `probe_error`); `@<corpus>` is the corpus suffix (§3 rule 4), always
last, absent on lab-std-v1.

```
introspect/act_norm/L{i}         introspect/probe_acc/{name}/L{i}
introspect/grad_norm/L{i}        introspect/lens_rank/{token}/L{i}
introspect/dead_frac/L{i}        introspect/jspace_pr/L{i}
introspect/logit_entropy         introspect/jspace_overlap/L{i}
introspect/overhead_pct          introspect/jspace_overlap_floor/L{i}   // the null, always beside it
introspect/pass1_seconds         introspect/jspace_align_z/{ref}/L{i}   // whitened z, never raw
introspect/pass2_seconds         introspect/whiten_pr/L{i}              // the stick's own drift (§3.2)
introspect/tier_skipped          introspect/whiten_trace/L{i}           //   — emitted beside every z
introspect/probe_error
```

Metrics measured on a non-default corpus carry the `@<corpus>` suffix in the key
(§3 rule 4); lab-std-v1 keys are unsuffixed.

### 5.2 Artifacts — the `introspection` class

New open-string `ArtifactClass` value `"introspection"` (no enum change — the class is
already an open string, chuk-compute §6). `jobspec.rs::train_job` adds one output rule
when a plan is present: `$CHUK_PROBE_DIR` → `probe/step_*` dirs, `on_appearance`, gated
by `.ready`, keyed to `introspect-hot/<run>/step_<n>/`.

```
introspect-hot/<run>/step_<n>/
  manifest.json          // plan fingerprint, tiers, layers, per-pass timings, determinism mode,
                         //   whitening source+reference, skips, provenance (trainer | eval_job)
  hidden.safetensors     // T1: per-layer last-position residuals [k, d] fp16
  lens.json              // T1: logit-lens topk + tracked ranks per layer
  probes.json            // T1: per-probe per-layer accuracy (+ CV)
  jspace_rows.safetensors// T3-rows (pass 2): per-token gradient rows [n_tok, d] per layer,
                         //   unembed-orthogonalized — ALSO the whitening covariance source (§3.2)
  jlens.pt               // T3-full (milestones only): {ℓ: J_ℓ} fp16 (~13 MB/layer @ d=2560)
```

Retention follows checkpoints: `introspect-hot/` R2 lifecycle (default 7d — longer than
`ckpt-hot/`'s 1d because curves get recomputed), milestone `jlens.pt` promoted alongside
`ckpt-final/` and eligible for the Drive archive tier. Sizes are small: a 5-layer T1
snapshot at k=40, d=2560 is ~1 MB; only milestone lens fits reach tens of MB.

The CP interprets the class in `hub.rs::on_message` exactly where `"checkpoint"` is
special-cased today: validate the manifest against the stored plan, record the snapshot
(`IntrospectStore`), fire a `ProbeSnapshot` event (new `EventKind`), and enqueue
reduction (§6). An eval-job fit ships to the **same key** with
`provenance: "eval_job"` in the manifest — downstream consumers never care which
process produced it.

### 5.3 Script contract additions

Two env vars (in `chuk-train-proto::script_env` + `jobspec.rs::script_environment`):

| Var | Meaning |
|---|---|
| `$CHUK_PROBE_PLAN` | path to the ProbePlan JSON staged with the job inputs (absent ⇒ introspection off) |
| `$CHUK_PROBE_DIR`  | where the trainer writes `step_<n>/` snapshot dirs (+ `.ready`) |

The probe corpus arrives as a staged input (content-addressed, worker-cached), same
mechanism as code units and resume checkpoints. Eval-job fits receive the identical
contract (plan slice + corpus + checkpoint as staged inputs) — the fit script is the
same library entry point, `chuk_introspect.fit_jlens_offline`.

---

## 6. Control-plane reduction (Rust)

The CP turns snapshot artifacts into dashboard-ready curves the moment they land —
`Hub::ingest_probe_snapshot`, mirroring `ingest_checkpoint`. All inputs are k×d or
n_tok×d fp16 blocks (KBs–MBs), promoted to f32/f64 before any linear algebra; with
`faer`/`ndarray` the whole reduction is milliseconds, so it runs inline on ingest like
gate evaluation does.

Reductions (each writes `introspect/*` metric points at the snapshot's step, so the
entire downstream — series API, dashboard, gates, experiments mirror — is the existing
metric machinery):

1. **PCA + participation ratio** per layer from `hidden.safetensors` and from
   `jspace_rows` → `jspace_pr/L{i}`, effective-k.
2. **Stability overlap with noise floor** — principal-angle overlap between split-A and
   split-B row subspaces, *and* the within-split floor (split A halved), emitted as a
   pair (`jspace_overlap/L{i}` + `jspace_overlap_floor/L{i}`). E22's correction, made
   structural: the dashboard plots lift-over-floor, never raw overlap.
3. **Whitened alignment** vs registered reference bases (stored per-run: probe
   directions, a pinned earlier checkpoint's basis, user-supplied taxonomy axes) —
   whitened per the plan's `whitening` block (§3.2): covariance from `jspace_rows`
   with `min_rows` enforced (insufficient rows ⇒ raw value + `insufficient_rows`
   flag, never a fake z); reference = self or pinned; z-scored against the
   spectrum-matched null (`random_null_overlap_dim`), never the isotropic null (E22
   Part A2's lesson, enforced in code); `whiten_pr/L{i}` + `whiten_trace/L{i}`
   emitted at the same step so the stick's drift is always on the same axes as the z.
4. **Drift** — CKA / subspace overlap of `hidden` vs previous snapshot and vs the pinned
   reference → `introspect/drift/L{i}`, `introspect/drift_ref/L{i}`.
5. **Downsampling** — the existing `downsample_in_place` path applies unchanged.

Rust crate note: the linear algebra lives in a small `introspect-reduce` module inside
`chuk-train-controlplane` (or a workspace crate if it grows), reading safetensors via
the `safetensors` crate. Reference implementations to port: `pr_of`, `pcs_of`,
`overlap`, `whitening_basis`, `whiten_directions`, `random_null_overlap_dim` from
`E22_jspace_transfer/e22_part0_jlens.py` / `e22_partA2_whitened.py` — each is <40 lines
of NumPy and ports 1:1. Parity gate: the Rust reductions must reproduce E22's published
numbers on its committed artifacts before EI2 closes.

**Gates:** because reductions land as ordinary metrics, the existing three-form gate
grammar applies immediately — `register_gate(metric="introspect/probe_acc/content/L20",
rule="no_improve(50, 200)", action="record")` or a watchdog
`last(introspect/dead_frac/L12) > 0.5 → stop_run`. One grammar addition is deferred to
§17 (delta-vs-floor forms).

---

## 7. The j-space probe family

Definition (from the E22/E24 campaign, which ports Anthropic's *Verbalizable
Representations Form a Global Workspace in Language Models*): the **J-lens** transports
a layer-ℓ residual into the final-layer basis via `J_ℓ = E[∂h_final/∂h_ℓ]` (expectation
over a generic corpus, positions, and future targets) and decodes with the model's own
unembedding; **j-space** is the span of sparse non-negative combinations of the J-lens
vectors (rows of `W_U·J_ℓ`). It is a *gradient/output-sensitivity* object, deliberately
contrasted with activation-variance (PCA) objects. `J_ℓ` is a property of the model —
so tracking it over training means **refitting per checkpoint against the frozen probe
corpus**; it cannot be derived from training gradients and does not piggyback on the
training backward.

Two recipes, mapped to cadence tiers (costs measured in E22/E24):

| | **Cheap rows** (E22 recipe, snapshot pass 2) | **Full `J_ℓ`** (paper-faithful, `jacobian-lens/jlens/fitting.py`) |
|---|---|---|
| What | per-target-token gradient rows `∂logit_v/∂h_ℓ` at **every completion position** (`row_positions`, filler-excluded), unembed-row orthogonalized | dense `[d,d]` Jacobian, all valid positions, one-hot cotangents |
| Cost | E22's ~7–40 s/layer (MLX) was measured for **final-position-only** rows — the `row_positions` recipe has ~`completion_len`× the targets (though one backward serves all source layers at once); re-measured on CUDA at EI1, and the pass-2 budget cap governs regardless | `ceil(d/dim_batch)` backwards/prompt — 320 at d=2560; ~100 prompts; GPU-minutes→hours |
| Where | trainer, grad-enabled snapshot pass 2 | inline (trainer) **or eval job** — §7.1 |
| Artifact | `[n_tok, d]` fp16 — KBs; doubles as the whitening covariance source (§3.2) | ~13 MB fp16/layer; resumable running-sum checkpoint; shardable via `merge()` |
| Cadence | **every snapshot** | **milestone checkpoints only** |
| Yields | PR/effective-k curve, stability-overlap-vs-floor curve, whitened alignment z | true lens: per-layer transport quality vs plain logit lens, workspace-band search |

**Row targets are a stick too (v0.4).** `row_targets: "model_top1"` means the
completions are the *current* model's — regenerated at every snapshot, so the target
tokens, their positions, and the filler mask all change as the model changes. A
cross-step overlap or alignment curve then moves for two confounded reasons: `J_ℓ`
changed, *and* the measuring targets changed under it — the §3.2 nonstationarity one
layer down. Same contract, then: `row_target_reference: "self"` (default; right for
within-snapshot geometry, and the honest "property of the current model" reading) or
`"pinned:<ckpt>"` — completions generated once by the pinned checkpoint and
teacher-forced through every later snapshot, so rows always measure sensitivity toward
the same token sequences. **Cross-step row claims use pinned targets or they aren't
claims** (§8.7). The realized completions are recorded in the manifest in both modes.
Pinning also repairs tolerant-mode reproducibility for pass 2: with self targets, a
top-1 flip inside kernel tolerance changes a target token outright — a different
measurement, not a perturbation within `rel: 1e-3`; with pinned targets the tolerance
claim is again about values, not token identities.

### 7.1 `full_fit_mode` — where the milestone fit runs

The full `J_ℓ` is a property of the **checkpoint**, not of the live training process —
the only thing the trainer contributes is an already-loaded model. So the fit does not
have to stall a training GPU for up to an hour:

- **`inline`** — the trainer runs `fit_jlens` at the milestone, budget-capped and
  resumable (running-sum checkpoint + `merge()`; a capped fit resumes at the next
  milestone). Right on single-session backends where there is no fleet to dispatch to.
- **`eval_job`** — the milestone checkpoint's promotion event triggers the CP to
  submit a **batch eval job** (existing job type, zero new machinery): checkpoint +
  probe corpus + plan slice as staged inputs, `chuk_introspect.fit_jlens_offline` as
  the entry point, output rule shipping `jlens.pt` to the same
  `introspect-hot/<run>/step_<n>/` key with `provenance: "eval_job"`. The training
  GPU never pauses; the fit bills its own lease (§13); a fit worker dying mid-fit
  resumes from the running-sum artifact like any resumable job.
- **`auto`** (default) — inline on single-session backends (Colab), eval_job on fleet
  backends (Vast/Lambda). The CP resolves the mode at submit time and records it in
  the plan's stored copy.

Time-series the family emits (each with its null — §8): `jspace_pr/L{i}`,
`jspace_overlap/L{i}` + `_floor`, `jspace_align_z/{ref}/L{i}` (whitened per §3.2, with
`whiten_pr`/`whiten_trace` beside it), and from milestone fits `jspace_lens_gain/L{i}`
— J-lens pass@k / median-rank **minus the plain logit-lens baseline**
(`use_jacobian=False`), reported alongside the **capability ceiling** (rank of the
target in the model's own output) so a capability floor is never mistaken for lens
failure (E24's gate).

The open research questions this makes tractable for the first time — does a stable
j-space *emerge* during training, at which depth, and when — are exactly the questions
E22 could not ask on frozen checkpoints. The framework's job is to produce honest
curves; the science stays in chris-experiments.

---

## 8. The honesty layer: nulls are schema, not etiquette

E22's arc (three same-day-killed "discoveries": small-N noise → noise-floor artifact →
anisotropic-null artifact) is baked in as invariants:

1. **Paired emission.** Reduction never emits a stability/alignment value without its
   control in the adjacent key (`…_floor`, `…_z` computed against the spectrum-matched
   null). Schema-enforced in `chuk-train-proto`.
2. **Whitened by default, from sufficient rows.** Alignment scores are whitened
   against a covariance estimated from `jspace_rows` with `min_rows` enforced; the
   isotropic null is not implemented; a rank-deficient covariance never silently
   masquerades as a spectrum-matched null (§3.2).
3. **The stick's own drift is visible.** Every whitened z is emitted alongside the
   whitening spectrum summaries (`whiten_pr`, `whiten_trace`) at the same step, and
   the dashboard draws them in the same band — because over training the background
   covariance is nonstationary and a z-curve without its stick is exactly the kind of
   artifact this layer exists to kill. Cross-step z *claims* use pinned-reference
   whitening (§3.2) or they aren't claims.
4. **Capability ceiling on every readout metric.** Lens/probe quality is reported
   relative to what the model can do at all at that step.
5. **Degradation is skip-first, and skips are data.** Over budget ⇒ skip the whole
   tier (pass 2 first, then pass 1; pulse never), logged in the manifest and as
   `introspect/tier_skipped`. The corpus subsample is **never thinned**: a gap in a
   curve is honestly missing, while a lower-N point under the same key looks
   comparable and is not — the exact small-N failure E22 opened with. One documented
   coupling: pass 2's rows are also the whitening covariance source (§3.2), so
   skipping pass 2 degrades **every** whitened metric at that step — including ones
   computed over pass-1 `hidden` blocks — to raw + `insufficient_rows`. That is the
   budget setting working as designed, not a whitening bug; the dashboard labels
   these points with the tier skip, not just the flag, so nobody debugs the reduction
   when the cause is a seconds cap.
6. **Unembed-row orthogonalization** is applied to all j-space rows before any
   geometry (the skip-path component inflates every stability estimate identically).
7. **Row targets are pinned for cross-step claims.** Self-targeted rows
   (`row_target_reference: "self"`) measure the current model against its own
   completions — valid within a snapshot, confounded across steps (§7). Any
   cross-step j-space claim uses `pinned` row targets, exactly as cross-step z claims
   use pinned-reference whitening (invariant 3); the manifest records the mode, and
   the dashboard labels self-targeted cross-step curves as such.

---

## 9. chuk-mcp-lazarus integration

Lazarus stays a separate server (training-spec §10's stance) and becomes the **analysis
layer over harness-captured artifacts** — it never owns the training model.

New attach mode beside `ModelState`: `attach_run(run_id)` creates a run handle backed by
the CP REST API (list snapshots, fetch artifacts via `artifact_url`, read
`introspect/*` series). Tools gain a step axis where they gain anything:

- `list_probe_snapshots(run_id)` → steps, tiers, manifests.
- `probe_timeline(run_id, metric, layers)` → step-indexed series (thin proxy over
  `run_metrics`).
- Existing backend-agnostic math pointed at fetched `hidden.safetensors` blocks:
  the probe stack, `extract_direction` (diff-means/LDA/PCA/probe-weights),
  geometry (`compute_subspace`, `feature_dimensionality`, angles),
  `attribution_sweep`-style aggregation, and the `compare_*` kernels for
  step-N-vs-step-M and run-vs-run deltas ("The Fine-Tuning Delta Problem", now with the
  delta measured *during* the run, not after).
- Deep interventional work (ablate, patch, steer, causal trace) still requires a loaded
  model: `load_checkpoint(run_id, step)` (the M5 tool) pulls the checkpoint from
  R2/Drive and runs the existing MLX toolchain on the Mac. Snapshot artifacts tell you
  *which* checkpoint and layer deserve that expensive attention.

The torch capture path deliberately does **not** move into lazarus — it lives in
`chuk-introspect` inside the trainer. Lazarus's MLX engine stays Mac-side; the shared
surface is artifacts and math, not model objects. (`chuk_lazarus`'s
`models_v2/core/backend/torch_backend.py` suggests a future shared-capture library —
noted in §17, not required by this spec.)

---

## 10. MCP tool surface (chuk-train additions)

Follows the established five-step seam (proto → store trait × sqlite+postgres → hub →
REST → Python client/tool). Kept deliberately small; series queries reuse `run_metrics`.

| Tool | Purpose |
|---|---|
| `submit_run` / `submit_sweep` (extended) | optional `probe_plan` (inline or by fingerprint); validated against declared budget caps and nominal row yield ≥ `min_rows` (§3.2); `full_fit_mode: auto` resolved and recorded at submit |
| `probe_plan_status(run_id)` | plan fingerprint, tier config, determinism mode, per-tier counts, overhead so far, skips, resolved fit mode |
| `list_probe_snapshots(run_id)` | steps + tiers + manifest summaries (incl. provenance: trainer / eval_job) |
| `probe_artifact_url(run_id, step, file)` | presigned/stable URL, R2-or-Drive resolver (mirrors `artifact_url`) |
| `probe_series(run_id, keys, layers?)` | convenience wrapper over `run_metrics` for `introspect/*` keys (envelope + `empty_hint` as usual) |
| `register_reference_basis(run_id, name, source)` | pin a reference (earlier checkpoint basis, direction set, pinned whitening covariance) for CP-side alignment/drift reduction |

Gates need no new tools — `register_gate`/`check_gates` already accept arbitrary metric
keys.

---

## 11. Dashboard

A new **Introspection** tab in the per-run view, beside Training/Logs/Events/System —
the System tab (one time-series graph per `sys/*` key) is the direct precedent, feeding
from the same store and the same 2 s polling (SSE when it lands).

- **Pulse band:** small multiples of `act_norm`/`grad_norm`/`dead_frac` per layer —
  rendered as **layer × step heatmaps** (the natural shape for "which depth, when"),
  with per-layer line drill-down.
- **Probe band:** probe accuracy per layer over steps (emergence-layer trajectory), lens
  tracked-token rank curves.
- **j-space band:** PR-by-layer over steps; overlap **with the floor curve drawn in the
  same axes** (principle: the null is always visible); whitened alignment z **with the
  whitening spectrum (`whiten_pr`/`whiten_trace`) drawn in the same band** (§8.3);
  milestone lens-gain vs plain logit lens with the capability ceiling shaded.
- **Snapshot table:** step, tiers run/skipped, per-pass durations, provenance
  (trainer / eval_job), artifact links, "open in lazarus" hint (the `attach_run`
  invocation).
- Fleet view unchanged; a run with a plan shows a probe badge; a pending eval-job fit
  shows as an ordinary queued job linked from the run.

---

## 12. Introspection sweeps

Sweeps compose for free: a `ProbePlan` on `submit_sweep` fans out to every child, and
because all curves are ordinary metrics, `sweep_status`'s cross-child aggregation
applies. The one genuinely new read is **cross-run comparison keyed by step**: "probe
emergence layer vs learning rate across the sweep" is a join the CP can serve from
existing tables (EI5's gate). Heavier cross-run geometry (shared-basis alignment across
children) is a lazarus/batch-eval job over the artifacts, not a CP feature. The
lab-standard corpus (§3 rule 4) is what makes cross-sweep and cross-programme
comparison meaningful at all.

This is also where chuk-mlx's batching DNA closes the loop: corpus artifacts and probe
plans are fingerprinted, sharded, and resumable in exactly the way `BatchPlan` made
training data — one IR, multiple consumers (trainer capture, eval-job fits, CP
reduction, lazarus analysis, sweep aggregation).

---

## 13. Cost governance

- **Declared budgets are enforced, not advisory** (§3 `budget` block): the library
  measures clean-step time at run start and **re-baselines on a rolling window**
  thereafter — a frozen run-start baseline drifts out from under curriculum or
  seq-len schedules and silently loosens (or tightens) enforcement — then **skips
  tiers** (never thins the corpus) to stay inside the caps; snapshot passes are budgeted separately
  (`snapshot_forward_seconds_max` / `snapshot_rows_seconds_max` — pass 2 is the
  expensive half of a "cheap" snapshot and hides inside a single cap); inline
  milestone fits get a wall-clock cap with checkpoint-resumable fits (`jlens`
  running-sum + `merge()`) so a capped fit resumes at the next milestone instead of
  discarding work.
- **Introspection time is lease time — attributed where it's spent.** Pulse/snapshot
  seconds are attributed in the utilization ledger (busy-GPU-min already first-class)
  under a probe bucket on the *training* lease; an eval-job fit bills its **own**
  lease, linked to the run in the ledger — so "how much did watching cost" is
  queryable per run, split by where the GPU-minutes actually went, and a fleet-mode
  run's training utilization is never diluted by fit time.
- **Milestone fits respect drain.** On lease drain an inline fit checkpoints and stops
  like the trainer does; an eval-job fit drains like any job; a partial fit is a valid
  resumable artifact either way.

---

## 14. Security

Unchanged threat model (training-spec §12): rented GPUs are untrusted, so **probe
corpora and probes must be public/synthetic** — the lab-standard corpus is by design a
generic web-text-like fixed set, and plans should be reviewed like code units.
Introspection artifacts (from trainers and eval-job fitters alike) flow through the
same grant-authed upload path and RBAC-gated REST as checkpoints; the dashboard tab is
behind the same sign-in. No new listener, no new credential, no worker-side change.

---

## 15. Proving ladder

Each milestone closes only on a real proven run, in the E0-ladder tradition (first
proofs on the paid Colab T4, per the colab-first rule; fp16 there — no bf16 on T4).

| Gate | Proof |
|---|---|
| **EI0** | stub trainer + pulse tier on Colab T4: `introspect/act_norm/L*` streams live to the dashboard; a watchdog gate on `introspect/dead_frac` fires in a poisoned run |
| **EI1** | v11 (115M) run with a ProbePlan: **both snapshot passes** land in `introspect-hot/`, `.ready` protocol honored (sealed only after both passes), manifest validated (determinism mode + per-pass timings recorded), `list_probe_snapshots` + `probe_artifact_url` work; each pass within its declared budget; one deliberately over-budget snapshot demonstrates skip-first degradation as a logged gap, not a thinned point |
| **EI2** | Rust reductions reproduce E22's committed numbers (PR, overlap+floor, whitened z) from E22's own artifacts (parity gate), then run live on EI1 snapshots; `min_rows` guard demonstrated (a starved snapshot emits raw + flag, not a fake z); Introspection tab renders layer×step heatmaps with floors and whitening-spectrum bands drawn |
| **EI3** | lazarus `attach_run` on the EI1 run: fetch a snapshot block, train a probe on it Mac-side, plot a timeline; `load_checkpoint` deep-dive on the layer the curves flag |
| **EI4** | full j-space over a real training run: rows every snapshot + ≥2 milestone `J_ℓ` fits — **at least one dispatched as an eval job** on a fleet backend (resumable across a lease bounce), at least one inline on Colab; lens-gain-vs-ceiling curve on the dashboard; cross-step alignment shown both self-whitened-with-spectrum and pinned-reference-whitened, with **pinned row targets** for the cross-step claim (§7, §8.7) — the first "does a stable j-space emerge during training" artifact anywhere in the lab |
| **EI5** | a 3-child LR sweep with a shared plan: cross-child probe-emergence comparison queryable via MCP and rendered on the sweep page |

---

## 16. Non-goals (v1)

- **Interventions during training** (steering/ablation/patching of a live run) — lazarus
  on checkpoints only.
- **A live on-GPU query endpoint** (service-job probing) — deferred until chuk-compute
  M5 services exist; the artifact path covers the need at checkpoint cadence.
- **MoE-specific probes, KV-cache introspection, attention-map archives at scale** —
  the capture families are extensible; these are follow-on plan tiers, not v1.
- **A Rust capture agent** — capture is Python-side by necessity (§2.1).
- **Replacing chris-experiments** — the framework produces honest curves; experimental
  design, hypotheses, and verdicts stay in the experiments repo / experiments server.

## 17. Open questions

1. **Gate grammar growth** — is `last(k) - last(k_floor) < v` (delta-vs-paired-null)
   worth a fourth form, or do we emit the lift as its own derived key and keep the
   grammar closed?
2. **Reduction placement at scale** — if snapshot cadence × layer count makes inline
   reduction contend with ingest, does reduction move to a CP background queue (same
   process) or a batch job? (Numbers say inline is fine at v1 scale.)
3. **Shared capture library** — should `chuk-introspect` and `chuk_lazarus`'s torch
   backend converge into one package with MLX/torch backends, or stay separate with a
   shared artifact schema only?
4. **Full-matrix `J_ℓ` retention** — keep every milestone lens (13 MB × layers ×
   milestones) or keep last-N + archive the rest to Drive? Interacts with the archive
   policy's 24 h / 30 d tiers.
5. **Deterministic-kernel coverage** — strict mode fails loudly on ops without
   deterministic kernels; if a target architecture (e.g. a MoE router op) turns out to
   lack one on the T4, do we accept per-op tolerant carve-outs declared in the plan,
   or hold the strict line and exclude the op from capture?
6. **Service-job live probing** (chuk-compute M5) — when services land, does a
   `probe` service beside the trainer (shared filesystem, reads latest checkpoint)
   become the interactive path, and what does its MCP surface look like?

*(v0.1's Q5 — corpus governance — resolved in v0.2 as §3 rule 4: one lab-standard
corpus, programme corpora additive only.)*