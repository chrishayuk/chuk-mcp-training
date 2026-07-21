# chuk-introspect — Specification v0.1

**Runtime introspection for training runs: capture on the GPU, reduce in the control
plane, read from lazarus, watch on the dashboard**
On-GPU capture library in **Python** (torch hooks, imported by the trainer) · transport
over the existing domain-free **chuk-compute** channels · reduction, storage, gates and
dashboard in **Rust** (control plane) · deep-dive analysis in **chuk-mcp-lazarus** (Mac)
· j-space (Jacobian-lens) tracking as a first-class probe family

v0.1 changes: initial spec. Synthesizes three proven codebases: the chuk-mlx
introspection + batching frameworks (`chuk-mlx/src/chuk_lazarus/introspection/`,
`data/batching/`), the chuk-mcp-lazarus tool surface
(`chuk-ai/mcp-servers/chuk-mcp-lazarus`), and the E22/E23/E24 j-space campaign
(`chris-experiments/fleet/E22_jspace_transfer/`, cloned `jacobian-lens/`). This spec is
the **deliberate scope expansion** that chuk-mcp-training-spec §10 reserves: live
introspection of a training loop on a remote GPU, not just Mac-side reads of finished
checkpoints.

---

## 0. Implementation status (v0.1, 2026-07-21)

| Milestone | State | Gate | Proven |
|-----------|-------|------|--------|
| **I0** pulse metrics: `introspect/*` namespace end-to-end (trainer → metrics → dashboard → gates) | ⬜ not started | EI0 | — |
| **I1** probe plan + snapshot artifacts: `ProbePlan` contract, `$CHUK_PROBE_PLAN`/`$CHUK_PROBE_DIR`, `introspection` artifact class, step-indexed capture on the probe corpus | ⬜ not started | EI1 | — |
| **I2** control-plane reduction (Rust): PR / overlap / drift / whitened-alignment computed server-side from snapshot artifacts; Introspection dashboard tab | ⬜ not started | EI2 | — |
| **I3** lazarus attach: `attach_run` remote mode — lazarus analysis math over harness-captured artifacts, step axis in results | ⬜ not started | EI3 | — |
| **I4** j-space over training: cheap rows every snapshot, paper-faithful `J_ℓ` at milestone checkpoints, nulls enforced, curves dashboarded | ⬜ not started | EI4 | — |
| **I5** introspection sweeps: cross-run comparison (probe curves across a sweep's children) | ⬜ not started | EI5 | — |

Nothing is built. The load-bearing decision already made upstream: the worker and wire
crates are **permanently domain-free** (chuk-compute-spec §1; enforced by the lexical CI
guard `chuk-compute-wire/tests/no_domain_vocabulary.rs`). Everything below rides
existing generic channels — a new metric namespace, a new artifact class string, two new
script-contract env vars. **Zero wire changes in the entire spec.**

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
   passes can only run inside the training process (Python/torch, on the GPU). Everything
   downstream of the captured artifact — PR, principal-angle overlap, whitening, nulls,
   downsampling, gates, rendering — is small-matrix math on k×d fp16/fp32 blocks and
   runs in the **Rust control plane** (`ndarray`/`faer`), the same way `complete_meta`
   already does checkpoint lineage server-side. No Python analysis hop between worker
   and dashboard.
3. **Plan as contract.** The `ProbePlan` is a versioned, fingerprinted artifact (the
   `BatchPlan` philosophy from `chuk-mlx/docs/batching.md` applied to introspection):
   same plan + same checkpoint ⇒ bit-identical capture. Probe corpora are content-hashed
   and frozen per run — a probe curve is meaningless if the corpus drifted under it.
4. **Config-in / result-out, typed everywhere.** Every capture family has an explicit
   config and an explicit result schema (Pydantic in the trainer library, serde in
   `chuk-train-proto`, mirrored Pydantic in the MCP client) — chuk-mlx's
   `*Config`/`*Result` discipline, house rules (no magic strings/numbers).
5. **Every signal ships with its null.** E22's verdict is the constitution here: a probe
   metric without its pre-registered adversarial control (within-split noise floor,
   whitened/spectrum-matched null, capability ceiling) is not rendered as a finding.
   The schema carries the null next to the value; the dashboard plots them together.
6. **Cheap always-on, expensive on milestones.** Tiered cadence: pulse scalars ride the
   training step (~free), snapshot probes run at checkpoint cadence on a subsampled
   corpus (seconds–minutes), heavy fits (full `J_ℓ`) run at milestone checkpoints only
   (minutes–hours) — each tier with an explicit overhead budget the plan declares and
   the harness enforces.
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
│     ├─ snapshot probes       │ │ │  ├─ reduce (ndarray/faer): │   │  └─ existing 80  │
│     │  (probe corpus fwd)    │ │ │  │   PR·overlap·whiten·null│   │     tools on     │
│     └─ jlens fits (backward) │ │ │  ├─ gates (introspect/*)   │   │     checkpoints  │
│          │                   │ │ │  ├─ store keys + retention │   └──────────────────┘
│          ▼                   │ │ │  └─ dashboard: Introspect  │   ┌──────────────────┐
│  $CHUK_METRICS (jsonl) ──────┼─┤ │      tab (layer×step)      │   │ chuk-train MCP   │
│  $CHUK_PROBE_DIR/step_N/ ────┼─┤ └────────────▲───────────────┘   │  (Python, thin)  │
└──────────────────────────────┘ │              │ REST              │  probe tools §10 │
┌──────────────────────────────┐ │              │                   └──────────────────┘
│ chuk-compute-worker (Rust)   │ │      R2: introspect-hot/…
│  DOMAIN-FREE — tails metrics,├─┘      Drive: archive tier
│  ships artifacts. Unchanged. │
└──────────────────────────────┘
```

### 2.1 The Rust / Python boundary

This is a first-class design axis. The rule: **Python where the autograd graph is; Rust
everywhere else.**

| Concern | Owner | Why |
|---|---|---|
| Hooks, forward/backward capture, jlens fits | **Python** — `chuk-introspect` lib inside the trainer process | needs the live torch model + autograd; nothing else can do it |
| Transport (tail metrics, ship artifacts, retry, seq/replay) | **Rust** — `chuk-compute-worker`, unchanged | already built, domain-free, proven |
| Proto types, constants, storage keys | **Rust** — `chuk-train-proto` (`introspect.rs`) | house pattern; single source of truth, mirrored to Pydantic |
| Ingest, validation, retention | **Rust** — control plane (`hub.rs` artifact arm + `IntrospectStore`) | same seam as checkpoint ingest |
| Reduction: PR, principal-angle overlap, whitening, nulls, drift, downsampling | **Rust** — control plane, `ndarray`/`faer` | inputs are k×d fp16 blocks (KBs–MBs); SVD/eig at d≤4096 is milliseconds; keeps the dashboard live with no Python hop |
| Gates on introspection metrics | **Rust** — existing gate engine (`gate.rs`), pointed at `introspect/*` keys | already evaluates on every metric ingest |
| Dashboard rendering | **Rust** — CP-served dash, new tab | System tab is the exact precedent |
| Deep-dive, interventional, exploratory analysis | **Python** — lazarus on the Mac | the 80-tool surface, sklearn probes, human-in-the-loop |
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
  read-only; jlens backward passes run on a cloned-or-frozen eval context, `torch.no_grad`
  everywhere except the lens cotangent passes).

---

## 3. The ProbePlan — plan as contract

A run opts into introspection by attaching a **ProbePlan** to `submit_run` (or a sweep).
The control plane stores it, stamps its fingerprint into run lineage (like
`complete_meta` does for checkpoints), and hands it to the trainer via the script
contract. The plan is the introspection analogue of chuk-mlx's `BatchPlan`: a
fingerprinted, versioned artifact that is the *contract*, not a config suggestion.

```jsonc
// ProbePlan v1 (stored CP-side; delivered as JSON at $CHUK_PROBE_PLAN)
{
  "version": 1,
  "fingerprint": "pp_9f2c…",            // content hash of everything below
  "model": { "d_model": 2560, "n_layers": 34, "dtype": "auto" },  // fp16 on T4 (no bf16), bf16 on Ampere+

  "corpus": {                            // FROZEN probe corpus — the fixed measuring stick
    "id": "probe-corpus/webtext-v1",     // content-addressed code-unit-style artifact (§11.1 pattern)
    "sha256": "…",
    "n_prompts": 100, "seq_len": 128,
    "splits": { "a": 50, "b": 50 }       // disjoint halves — feeds the stability/noise-floor nulls
  },

  "pulse": {                             // Tier 0 — rides the training step, ~free
    "every_steps": 1,
    "metrics": ["act_norm", "grad_norm", "dead_frac", "logit_entropy"],
    "layers": "all"
  },

  "snapshot": {                          // Tier 1/2 — probe-corpus forward at ckpt cadence
    "on_checkpoint": true,               // piggyback the existing checkpoint schedule
    "corpus_subsample": 40,              // E22: PR/stability shape visible at N=20–40
    "layers": [4, 12, 20, 26, 32],       // or "evenly_spaced:5" (chuk-mlx LayerStrategy)
    "position": "last",                  // CaptureConfig default; memory-bounded
    "capture": ["hidden", "logit_lens", "linear_probe"],
    "probes": [                          // sklearn-style linear probes, spec'd not code'd
      { "name": "content", "labels_from": "corpus_meta.category" }
    ]
  },

  "jspace": {                            // Tier 3 — see §7
    "rows_on_snapshot": true,            // cheap per-token gradient rows every snapshot
    "row_targets": "model_top1",         // E22 recipe: model's own clean completions, filler-excluded
    "full_fit_on": "milestone",          // paper-faithful J_ℓ only at milestone ckpts
    "milestone_every_ckpts": 10,
    "fit": { "dim_batch": 8, "skip_first": 16, "source_layers": [4, 12, 20, 26, 32] }
  },

  "budget": {                            // enforced (§13)
    "pulse_overhead_pct_max": 2,         // vs. clean step time, measured at run start
    "snapshot_seconds_max": 120,
    "milestone_seconds_max": 3600
  }
}
```

Rules:

- **The corpus is frozen per run.** Same corpus, same subsample seed, every snapshot —
  otherwise curves compare apples to oranges. Corpus artifacts are content-addressed and
  cached on the worker exactly like code units (§11.1 of the training spec).
- **Fingerprint into lineage.** `meta.json` of every checkpoint produced by a probed run
  carries `probe_plan: pp_…`, so any later offline recompute (lazarus, batch eval) can
  verify it is measuring with the same stick.
- **No plan ⇒ no-op.** `$CHUK_PROBE_PLAN` unset ⇒ chuk-introspect is inert; the trainer
  contract is unchanged from today's five touch-points.

---

## 4. On-GPU capture: the `chuk-introspect` Python library

A small, dependency-light library the trainer imports (ships inside the code unit, or as
a declared dependency in `unit.toml`). It is the torch/CUDA generalization of
`chuk_lazarus.introspection`, keeping the shape and dropping the MLX hacks:

| chuk-mlx (MLX) mechanism | chuk-introspect (torch) replacement |
|---|---|
| `ModelHooks.forward` manual forward unroll (`hooks.py:398`) | `nn.Module.register_forward_hook` on real modules; `output_hidden_states=True` zero-hook path for residuals |
| `moe/hooks.py` monkey-patches `mlp.__call__` | forward hook on the router submodule |
| `interventions.py` swaps `model.layers` entries (reference-aliasing hazard) | hook handles (`handle.remove()` in `finally`) — no structural mutation |
| immutable-tensor `mx.concatenate` position writes | not needed (capture is read-only; interventions are out of scope during training) |
| `mx.stop_gradient` / periodic `mx.eval` | `.detach()` under `torch.no_grad()`; real batched forwards instead of per-prompt loops |
| per-prompt Python loop in `circuit/collector.py` | true GPU batching of the probe corpus — the major throughput win |

Kept from chuk-mlx wholesale (they are already framework-agnostic patterns):
`CaptureConfig`-style layer/position selection (`LayerStrategy`, `PositionSelection`),
the `Config → Service → Result` layering, `CollectedActivations`-style
safetensors + JSON-sidecar artifacts, `ModelAccessor` duck-typing for architecture
variance (HF naming schemes), and the circuit pipeline's staged, resumable
dataset → collect → reduce discipline.

### 4.1 API shape (trainer-side)

```python
from chuk_introspect import Introspector

intr = Introspector.from_env()            # reads $CHUK_PROBE_PLAN, $CHUK_PROBE_DIR, $CHUK_METRICS; inert if unset

for step, batch in enumerate(loader):
    with intr.pulse(model, step):         # T0: attaches hooks for this step iff step % every == 0
        loss = train_step(model, batch)   # pulse scalars appended to $CHUK_METRICS as introspect/*

    if ckpt_due(step):
        write_checkpoint(...)             # unchanged
        intr.snapshot(model, step)        # T1/T2 (+T3 rows): probe-corpus forward under no_grad,
                                          #   writes $CHUK_PROBE_DIR/step_N/… then .ready
        if intr.milestone_due(step):
            intr.fit_jlens(model, step)   # T3 full fit: resumable, sharded, budget-capped (§7)
```

Five properties the library guarantees:

1. **Inert without a plan** — `from_env()` returns a no-op object; zero overhead.
2. **Budget-honest** — measures its own wall-clock, emits `introspect/overhead_pct`,
   and *degrades* (halve corpus subsample → skip tier) rather than blow the declared
   budget; every degradation is logged as a metric so silent truncation can't read as
   full coverage.
3. **Crash-isolated** — a probe failure logs `introspect/probe_error` and skips; it
   never takes the training step down with it.
4. **Deterministic** — corpus order and subsample come from the plan fingerprint + step,
   never wall-clock or RNG state shared with training.
5. **Ready-marker protocol** — snapshot dirs are written then sealed with `.ready`,
   identical to checkpoints, so the worker's `on_appearance` output rule ships complete
   directories only.

### 4.2 Capture families

- **T0 pulse (per training step, hooks on the live batch):** per-layer activation norm,
  per-layer/component grad norm (read post-backward, pre-step), dead-neuron fraction,
  logit entropy, router load/entropy on MoE. Emitted as `introspect/…` records in
  `$CHUK_METRICS` — no artifacts, no extra forward.
- **T1 lens (per snapshot, probe-corpus forward):** last-position residuals per selected
  layer (`hidden.safetensors`, k×d fp16), calibrated logit-lens top-k + tracked-token
  ranks (JSON), linear-probe accuracy per layer per registered probe (the sklearn fit
  runs trainer-side on the captured k×d block — cheap), attention entropy summaries.
- **T2 geometry (derived, mostly CP-side — §6):** PCA basis + participation ratio,
  split-A/split-B stability overlap, representation drift vs previous snapshot and vs a
  pinned reference checkpoint (the lazarus `compare_*` kernels with a step axis).
- **T3 j-space (§7):** cheap gradient rows every snapshot; full `J_ℓ` at milestones.

---

## 5. Data model and flow

### 5.1 Metrics — the `introspect/` namespace

Third metric shape alongside training metrics and `sys/*`, riding the **training-metric
path** (job-scoped, step-indexed): records land in `$CHUK_METRICS`, the worker tails and
ships them as `Metric{job_id, step, values}`, the CP appends via
`MetricStore::append_metrics`, `run_metrics`/dashboard/gates read them back downsampled.
Namespace constant `INTROSPECT_METRIC_PREFIX = "introspect/"` in `chuk-train-proto`
(the worker never sees it as anything but a string key).

Key grammar (all named constants, no free-form strings):

```
introspect/act_norm/L{i}         introspect/probe_acc/{name}/L{i}
introspect/grad_norm/L{i}        introspect/lens_rank/{token}/L{i}
introspect/dead_frac/L{i}        introspect/jspace_pr/L{i}
introspect/logit_entropy         introspect/jspace_overlap/L{i}
introspect/overhead_pct          introspect/jspace_overlap_floor/L{i}   // the null, always beside it
introspect/probe_error           introspect/jspace_align_z/{ref}/L{i}   // whitened z, not raw
```

### 5.2 Artifacts — the `introspection` class

New open-string `ArtifactClass` value `"introspection"` (no enum change — the class is
already an open string, chuk-compute §6). `jobspec.rs::train_job` adds one output rule
when a plan is present: `$CHUK_PROBE_DIR` → `probe/step_*` dirs, `on_appearance`, gated
by `.ready`, keyed to `introspect-hot/<run>/step_<n>/`.

```
introspect-hot/<run>/step_<n>/
  manifest.json          // plan fingerprint, tier, layers, corpus subsample, timings, degradations
  hidden.safetensors     // T1: per-layer last-position residuals [k, d] fp16
  lens.json              // T1: logit-lens topk + tracked ranks per layer
  probes.json            // T1: per-probe per-layer accuracy (+ CV)
  jspace_rows.safetensors// T3-cheap: per-token gradient rows [n_tok, d] per layer, unembed-orthogonalized
  jlens.pt               // T3-full (milestones only): {ℓ: J_ℓ} fp16 (~13 MB/layer @ d=2560)
```

Retention follows checkpoints: `introspect-hot/` R2 lifecycle (default 7d — longer than
`ckpt-hot/`'s 1d because curves get recomputed), milestone `jlens.pt` promoted alongside
`ckpt-final/` and eligible for the Drive archive tier. Sizes are small: a 5-layer T1
snapshot at k=40, d=2560 is ~1 MB; only milestone lens fits reach tens of MB.

The CP interprets the class in `hub.rs::on_message` exactly where `"checkpoint"` is
special-cased today: validate the manifest against the stored plan, record the snapshot
(`IntrospectStore`), fire a `ProbeSnapshot` event (new `EventKind`), and enqueue
reduction (§6).

### 5.3 Script contract additions

Two env vars (in `chuk-train-proto::script_env` + `jobspec.rs::script_environment`):

| Var | Meaning |
|---|---|
| `$CHUK_PROBE_PLAN` | path to the ProbePlan JSON staged with the job inputs (absent ⇒ introspection off) |
| `$CHUK_PROBE_DIR`  | where the trainer writes `step_<n>/` snapshot dirs (+ `.ready`) |

The probe corpus arrives as a staged input (content-addressed, worker-cached), same
mechanism as code units and resume checkpoints.

---

## 6. Control-plane reduction (Rust)

The CP turns snapshot artifacts into dashboard-ready curves the moment they land —
`Hub::ingest_probe_snapshot`, mirroring `ingest_checkpoint`. All inputs are k×d or
n_tok×d fp16 blocks (KBs–MBs); with `faer`/`ndarray` the whole reduction is
milliseconds, so it runs inline on ingest like gate evaluation does.

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
   whitened by the snapshot's own background covariance, z-scored against the
   spectrum-matched null (`random_null_overlap_dim`), never the isotropic null
   (E22 Part A2's lesson, enforced in code).
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

| | **Cheap rows** (E22 recipe) | **Full `J_ℓ`** (paper-faithful, `jacobian-lens/jlens/fitting.py`) |
|---|---|---|
| What | per-target-token gradient rows `∂logit_v/∂h_ℓ`, final position, unembed-row orthogonalized | dense `[d,d]` Jacobian, all valid positions, one-hot cotangents |
| Cost | ~7–40 s/layer for tens of prompts (MLX-measured; CUDA faster) | `ceil(d/dim_batch)` backwards/prompt — 320 at d=2560; ~100 prompts; GPU-minutes→hours |
| Artifact | `[n_tok, d]` fp16 — KBs | ~13 MB fp16/layer; resumable running-sum checkpoint; shardable via `merge()` |
| Cadence | **every snapshot** | **milestone checkpoints only** |
| Yields | PR/effective-k curve, stability-overlap-vs-floor curve, whitened alignment z | true lens: per-layer transport quality vs plain logit lens, workspace-band search |

Time-series the family emits (each with its null — §8): `jspace_pr/L{i}`,
`jspace_overlap/L{i}` + `_floor`, `jspace_align_z/{ref}/L{i}` (whitened), and from
milestone fits `jspace_lens_gain/L{i}` — J-lens pass@k / median-rank **minus the plain
logit-lens baseline** (`use_jacobian=False`), reported alongside the **capability
ceiling** (rank of the target in the model's own output) so a capability floor is never
mistaken for lens failure (E24's gate).

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
2. **Whitened by default.** Alignment scores are whitened against the snapshot's own
   background covariance; the isotropic null is not implemented.
3. **Capability ceiling on every readout metric.** Lens/probe quality is reported
   relative to what the model can do at all at that step.
4. **Degradation is data.** Corpus subsampling, skipped tiers, budget clamps land in the
   manifest and as metrics — a curve that thinned its corpus mid-run says so on the
   dashboard.
5. **Unembed-row orthogonalization** is applied to all j-space rows before any geometry
   (the skip-path component inflates every stability estimate identically).

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
  the sklearn probe stack, `extract_direction` (diff-means/LDA/PCA/probe-weights),
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
| `submit_run` / `submit_sweep` (extended) | optional `probe_plan` (inline or by fingerprint); validated against declared budget caps |
| `probe_plan_status(run_id)` | plan fingerprint, tier config, per-tier counts, overhead so far, degradations |
| `list_probe_snapshots(run_id)` | steps + tiers + manifest summaries |
| `probe_artifact_url(run_id, step, file)` | presigned/stable URL, R2-or-Drive resolver (mirrors `artifact_url`) |
| `probe_series(run_id, keys, layers?)` | convenience wrapper over `run_metrics` for `introspect/*` keys (envelope + `empty_hint` as usual) |
| `register_reference_basis(run_id, name, source)` | pin a reference (earlier checkpoint basis, direction set) for CP-side alignment/drift reduction |

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
  same axes** (principle: the null is always visible); whitened alignment z; milestone
  lens-gain vs plain logit lens with the capability ceiling shaded.
- **Snapshot table:** step, tier, duration, degradations, artifact links, "open in
  lazarus" hint (the `attach_run` invocation).
- Fleet view unchanged; a run with a plan shows a probe badge.

---

## 12. Introspection sweeps

Sweeps compose for free: a `ProbePlan` on `submit_sweep` fans out to every child, and
because all curves are ordinary metrics, `sweep_status`'s cross-child aggregation
applies. The one genuinely new read is **cross-run comparison keyed by step**: "probe
emergence layer vs learning rate across the sweep" is a join the CP can serve from
existing tables (EI5's gate). Heavier cross-run geometry (shared-basis alignment across
children) is a lazarus/batch-eval job over the artifacts, not a CP feature.

This is also where chuk-mlx's batching DNA closes the loop: corpus artifacts and probe
plans are fingerprinted, sharded, and resumable in exactly the way `BatchPlan` made
training data — one IR, multiple consumers (trainer capture, CP reduction, lazarus
analysis, sweep aggregation).

---

## 13. Cost governance

- **Declared budgets are enforced, not advisory** (§3 `budget` block): the library
  measures clean-step time at run start, then degrades tiers to stay inside
  `pulse_overhead_pct_max`; snapshots and milestone fits get wall-clock caps with
  checkpoint-resumable fits (`jlens` running-sum + `merge()`) so a capped fit resumes at
  the next milestone instead of discarding work.
- **Introspection time is lease time.** Snapshot/fit seconds are attributed in the
  utilization ledger (busy-GPU-min already first-class) under a probe bucket, visible in
  `spend_status` — so "how much did watching cost" is queryable per run.
- **Milestone fits respect drain.** On lease drain the fit checkpoints and stops like
  the trainer does; a partial fit is a valid resumable artifact.

---

## 14. Security

Unchanged threat model (training-spec §12): rented GPUs are untrusted, so **probe
corpora and probes must be public/synthetic** — the corpus is by design a generic
web-text-like fixed set, and plans should be reviewed like code units. Introspection
artifacts flow through the same grant-authed upload path and RBAC-gated REST as
checkpoints; the dashboard tab is behind the same sign-in. No new listener, no new
credential, no worker-side change.

---

## 15. Proving ladder

Each milestone closes only on a real proven run, in the E0-ladder tradition (first
proofs on the paid Colab T4, per the colab-first rule; fp16 there — no bf16 on T4).

| Gate | Proof |
|---|---|
| **EI0** | stub trainer + pulse tier on Colab T4: `introspect/act_norm/L*` streams live to the dashboard; a watchdog gate on `introspect/dead_frac` fires in a poisoned run |
| **EI1** | v11 (115M) run with a ProbePlan: T1 snapshots land in `introspect-hot/`, `.ready` protocol honored, manifest validated, `list_probe_snapshots` + `probe_artifact_url` work; overhead within declared budget |
| **EI2** | Rust reductions reproduce E22's committed numbers (PR, overlap+floor, whitened z) from E22's own artifacts (parity gate), then run live on EI1 snapshots; Introspection tab renders layer×step heatmaps with floors drawn |
| **EI3** | lazarus `attach_run` on the EI1 run: fetch a snapshot block, train a probe on it Mac-side, plot a timeline; `load_checkpoint` deep-dive on the layer the curves flag |
| **EI4** | full j-space over a real training run: cheap rows every snapshot + ≥2 milestone `J_ℓ` fits (resumable across a lease bounce), lens-gain-vs-ceiling curve on the dashboard — the first "does a stable j-space emerge during training" artifact anywhere in the lab |
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
5. **Probe corpus governance** — one lab-standard frozen corpus (comparable across all
   runs forever) vs per-programme corpora (better matched to the model's data)? E24
   suggests quality saturates fast (~100 prompts), which argues for one standard.
6. **Service-job live probing** (chuk-compute M5) — when services land, does a
   `probe` service beside the trainer (shared filesystem, reads latest checkpoint)
   become the interactive path, and what does its MCP surface look like?
