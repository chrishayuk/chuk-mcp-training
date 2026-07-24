# chuk-compute: Worker & Wire Specification (v1)

Status: design. Supersedes the "agent" sections of the chuk-mcp-training ROADMAP; the two
roadmap directions (join-anywhere worker, per-run sys telemetry) are folded in here alongside
the generalized workload model. Substrate spec for the training rig — see the main spec
`chuk-mcp-training-spec.md` for the control-plane/product layer that sits on top.

## 1. Scope and principles

The rig is a compute fabric, not a training system. The training-ness lives in control-plane
templates, dashboards, watchdogs, and the experiments-server integration; the worker daemon and
the wire protocol are permanently compute-generic. The enforcement test is lexical: the word
"train" must never appear in the wire crate or the worker crate. A second naming rule follows
from the workload model: the daemon that joins the fabric is a **worker**, never an "agent" —
that word is reserved for LLM/agentic workloads that run *on* the fabric.

The workload model must express six things without protocol changes beyond what this spec
defines: pretraining/midtraining runs, evals, benchmark campaigns, cell80 runtimes, agent
workloads, and RL loops. Anything that would require the scheduler to understand workflow
structure (DAGs, loop semantics, tool routing, conversation state) is out of scope by design;
those live in CP templates or in controller jobs driving the CP over MCP.

**Training primacy.** The rig's product is the training research loop; that focus is a
requirement, not a default to drift from. Every other workload type is admitted because it
serves that loop — evals validate checkpoints, benches validate designs (tokenizers, cell
libraries), cells verify and author curricula, agents and RL close the improvement loop, and
all of them pack otherwise-wasted lease time around training runs. The generic layer is the
bottom two layers only (wire, worker); the top of the stack — templates, dashboards, watchdogs,
MFU, experiments-server integration — is training-first and stays that way. Two standing rules
enforce this: no workload template is built until a real run needs it, and every milestone demo
must be an improvement to the training loop, not to the substrate for its own sake.

Cost control remains a first-order requirement. Every leased worker enforces its wall locally,
every job carries a runtime budget, and campaigns carry a spend budget enforced at submission
time by the CP.

## 2. Crate layout

Two new crates inside the existing workspace, with a hard dependency rule.

**`chuk-compute-wire`** (lib) is the protocol crate: message enums, job and artifact types,
capability structs, worker classes, the protocol version constant. It is serde-only — no tokio,
no transport, no CP internals. Both the CP and the worker depend on it; the worker depends on
nothing else from the workspace. Serde discipline for compatibility: `#[serde(default)]` on
every additive field, no `deny_unknown_fields` anywhere, `#[non_exhaustive]` on enums the CP may
grow. Old workers must tolerate new CPs and vice versa; the handshake gates only on
`PROTOCOL_VERSION`, which is bumped exclusively for breaking changes.

**`chuk-compute-worker`** (bin) is the join-anywhere daemon: transport, job supervision,
telemetry sampling, artifact staging, self-update. Cargo features `nvidia` and `apple` gate the
telemetry backends; both are compiled into release binaries and detected at runtime.

The crates stay in the chuk-mcp-training repo for now (path deps, atomic protocol changes while
the protocol settles); the only discipline required is that the worker's Cargo.toml names
`chuk-compute-wire` alone, which keeps a later repo extraction trivial. The repo keeps its name
— that is the product (the training rig); these crates are its substrate.

Both crates carry the workspace's engineering bar: **≥90% line coverage per file**, gated in
CI with no allowlist, plus the lexical guards below. The worker's own third-party edges (its
HTTP client, input staging, self-update) are tested against loopback servers rather than
mocked, so the assertions are on real requests.

## 3. Protocol, handshake, self-update

Before anything else the worker sends `Hello { protocol_version, worker_semver, target_triple,
token, capabilities, resume }`. The CP replies `HelloAck { worker_id, class, telemetry }` or
`HelloReject { reason, min_protocol, url, sha256 }`.

Version-mismatch handling differs by worker class. Leased workers simply exit on reject — the
bootstrap re-downloads the current binary on every provision, so they are always current by
construction. Persistent workers self-update from the reject payload: download to a temp path,
verify the sha256, atomically rename over the running binary, re-exec. This is deliberately
hand-rolled (~80 lines) rather than a crate dependency, since binaries are served from our own CP.

`resume` is set on reconnect and carries the worker id, the still-running job ids, and a
metric-sequence high-water mark, letting the CP resynchronize instead of double-assigning and
letting the metric replay deduplicate. See §8 for the spool that makes replay possible.

## 4. Build and distribution

Release targets are `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`, and
`aarch64-apple-darwin`. Musl static linking is what makes the leased case safe: the binary
depends on nothing but the kernel, so it runs on whatever image Vast or Lambda hands us. TLS is
rustls throughout — no OpenSSL means no cross-compilation sysroot pain and no runtime library
expectations.

CI builds both Linux targets from a single runner with cargo-zigbuild (with rustls there are no
awkward native deps, which is the case zigbuild handles cleanly) and the darwin target on a
macOS runner, which also keeps a clean path to signing/notarization if that is ever wanted. The
CP serves `/agent/{os}-{arch}` with a `.sha256` alongside each binary and a `/agent/version`
endpoint for the self-updater. The existing hardcoded `/agent/linux-x86_64` path (constants.rs)
is retired.

The bootstrap is one `install.sh` in the rustup style: map `uname -s`/`uname -m` to a triple,
curl the binary, verify the checksum, exec `worker --join $URL --token $TOKEN`. The Colab cell
and the Vast onstart template (provider/vast.rs) become one-line wrappers over it; the Mac runs
it directly with a persistent token.

## 5. Worker classes and tokens

`WorkerClass` is an enum, not a flag, so that destroying a persistent worker is unrepresentable
in CP code rather than merely checked at runtime.

A **leased** worker is CP-provisioned (Vast, Lambda, Colab), joins with a single-use
per-provision token, carries a hard wall deadline from its lease, and is destroyed by the CP
through the provider API when the lease ends. The wall is enforced worker-side as well as CP-side
(§8), so "when the session is over it's over" holds even with the control link down.

A **persistent** worker is self-enrolled (the Mac), holds a long-lived token that is revocable
server-side and scoped to its worker identity, has no wall, is never destroyed, and reconnects
forever with jittered exponential backoff. In-flight jobs survive a dropped connection: the
supervisor keeps running, metrics spool to disk, and state replays on reconnect.

Capabilities advertised at Hello cover os/arch, cpu cores, RAM, free disk, preemptibility, an
accelerator description (`Cuda` with per-GPU VRAM and driver/CUDA versions, `Mps` with chip and
unified-memory size, or `Cpu`), and a free-form label map ("site=colab", "site=home") the
scheduler may match against but the wire does not interpret. There are no `Provision`/`Destroy`
wire messages: provisioning is CP↔provider traffic, and the worker never participates in its own
destruction — the leased/persistent asymmetry lives entirely server-side.

## 6. Jobs, services, artifacts, secrets, campaigns

A job is: stage artifacts in, run one command under supervision, stream metrics, collect
artifacts out. The worker knows nothing else. `template` is an opaque tag ("train", "eval",
"bench", "agent") used by CP dashboards and packing heuristics; the worker must not branch on it.

**Batch vs service.** `service: Option<ServiceSpec>` is the single extension that admits cells,
agents, and inference. A batch job has `max_runtime_secs` and its effective deadline is
min(worker wall, now + max_runtime) — which is also what makes packing expressible: the CP fits
jobs whose budgets sum under a lease's remainder. A service job (`ServiceSpec { name, ports,
readiness, restart }`) runs until Cancel, Drain, or wall; the worker waits for readiness (HTTP
probe, TCP open, or log-line match), reports `ServiceReady` with its ports, and restarts per
policy. The CP maintains the service registry and composes endpoint URLs from worker addresses. A
consuming job declares `needs: [ServiceRef { name, env }]`; the CP resolves each name at assign
time, injects the URL into the job's environment, and holds the job in Staging until its
dependencies are Ready. Placement is a hint, not a guarantee: `placement.prefer_worker` and
`placement.require_labels` let rollout jobs land next to their policy service.

**Artifacts.** `ArtifactClass` is an open string, not an enum — "checkpoint", "log", "report",
"dataset", "rollouts", "cell-library" are conventions of the CP and experiments-server, so a new
artifact kind never touches the protocol. Inputs are presigned URLs minted at assign time (the
worker never holds storage credentials) with a sha256 and a destination path; resuming a run is
nothing but the CP staging a prior checkpoint artifact into the sandbox plus the trainer's own
resume flag — the worker does not know what a resume is. Output rules pair a class with a glob
and an upload policy: `on_exit` for reports and final state, `on_appearance` for periodic
checkpoints (a preempted box loses minutes, not the run), `stream` for live log tailing over the
metric channel.

**Secrets.** Environment values are `Plain(String) | Secret(String)` where the secret variant is
a CP-held reference resolved at assign time. Persisted job specs — including everything mirrored
into the experiments DB — carry only the reference, never the material. This exists for agent
workloads (API keys) but applies uniformly.

**Campaigns.** `campaign: Option<CampaignId>` groups fan-out — N benchmark configs, N rollout
shards — as a flat set with no edges. Grid expansion is a CP MCP tool (`submit_campaign(template,
matrix)`); aggregation is the experiments-server's job. Campaigns carry a CP-side spend budget
(GPU-hours or currency) enforced by refusing submissions past it; this is the wallet-level guard
for loop workloads, complementing the per-box wall. Deliberately absent: any dependency edges. "B
after A" is the controller pattern (§10.6), not a scheduler feature.

## 7. Telemetry

The worker runs one sampler task (default 5 s, CP-configurable via `TelemetryConfig` in HelloAck)
emitting a `sys/*` namespace over the existing Metric channel, so the whole pipeline — metrics
store, dashboard curves, experiments-server — is reused with no new plumbing. Counters (disk and
network I/O) are emitted as worker-side deltas. Samples carry worker id and, when attributable,
job id, so the CP can join telemetry to runs; idle-worker samples carry no job id.

NVIDIA sampling uses `nvml-wrapper`, which loads NVML at runtime via libloading — one binary
degrades gracefully on GPU-less boxes and on the Mac — and yields structured utilization, memory,
temperature, power, clocks, and per-process usage without parsing nvidia-smi text. NVML's
backward-compatibility guarantee covers driver drift on rented images. Host CPU/RAM/disk/net and
per-process (trainer PID) views come from `sysinfo`.

Apple Silicon is better than the old "powermetrics wants sudo" caveat: the IOReport family of
private APIs provides GPU power, utilization, and temperature without sudo, as demonstrated by
macmon (itself Rust — usable as FFI reference) and mactop. Phase one shells out to `macmon pipe`
and parses its JSON stream; phase two moves the IOReport calls in-process. Because these are
private APIs that drift across macOS releases and new silicon, MPS telemetry is tier-2
best-effort and gaps are surfaced as absent metrics, never zeros.

This feed is what turns the training dashboard into an ops dashboard downstream:
packing-utilization, OOM and thermal-throttle watchdog gates, and MFU (tokens/s against GPU
utilization) all consume `sys/*` — but those are CP features, out of scope here.

## 8. Supervision and local enforcement

The hard-wall guarantee lives in the worker, not the CP. Each job runs in its own sandbox
directory and its own process group (setsid). On wall expiry, `max_runtime` expiry, or Cancel,
the supervisor sends SIGTERM to the group, waits `term_grace_secs` (default 30 — the window for a
checkpoint flush), then SIGKILLs the group, so forked dataloader workers cannot orphan.
`kill_on_drop` backstops supervisor panics. The corresponding `JobEvent::Killed { reason }` is
reported after the fact and is eventually-delivered — the design assumes it may only be sendable
on reconnect, because a dead control link at wall time is exactly the failure mode that matters on
preemptible boxes.

Metrics and job events spool to disk in sequence-numbered batches and replay on reconnect against
the CP's high-water mark. This serves both the persistent worker's long disconnects and flaky
leased-box networking.

An OOM guard (from `sys/*` GPU-memory samples crossing a threshold) may preemptively kill with
`KillReason::OomGuard`; thresholds arrive in `TelemetryConfig` later — reserved in the enum now.

## 9. Non-goals

The rig is not an agent framework: no conversation state, no tool routing, no A2A semantics in the
wire — mcp-cli and chuk-tool-processor are the agent runtime, and the rig runs them as processes.
It is not a workflow engine: no DAGs, no loop constructs, no inter-job edges. It is not
multi-tenant and has no generic quota system. It does not embed cell80: cells reach the fabric as
artifacts and services (§10.4), and the worker cannot tell. Network-egress policy for jobs is
acknowledged as a future label-level concern and deliberately unbuilt.

## 10. Workload cookbook

**10.1 Training.** Batch job, template "train". Dataset and optional resume-checkpoint as inputs;
`on_appearance` checkpoint uploads; `stream` logs; job metrics under `train/*`. The CP's training
template layers dashboards, watchdogs, and experiments-server records on top; none of that is
visible to the worker.

**10.2 Eval.** Batch job, template "eval", typically packed into lease tails after a training job.
Checkpoint artifact in, `report` artifact out, `eval/*` metrics.

**10.3 Benchmarking.** A campaign of batch jobs over a config grid. The template refuses unpinned
submissions: library digest, dataset digest, image label, and seed must all appear in the spec,
so two reports are comparable iff their specs hash equal modulo the varied axis. Short
known-budget bench jobs are the scheduler's packing filler of choice.

**10.4 Cells.** Two modes. In-process: a job declares the cell library snapshot
(`ArtifactClass("cell-library")`, content-addressed and signed — already speaking the artifact
system's language) as an input and loads the runtime itself; this is the path for
corpus-build-time execution, the dense in-loop eval harness, and STaR-style verification, where no
network hop belongs. Service: the cell80 MCP server as `ServiceSpec { name: "cell-runtime" }` for
brokers, agents, and shared use — a pure, filesystem-inert runtime being the safest possible thing
to expose on a rented box. In both modes the experiment record pins the exact library digest that
executed, extending prereg discipline to the tool substrate.

**10.5 Agents.** Batch (an eval campaign that exits) or service (a long-running loop), template
"agent". `needs` wires them to inference ("policy"/"larql") and "cell-runtime" services; secrets
carry their API keys; they are bursty, mostly-CPU, and cheap — good packing filler. The flagship
composition: LARQL as a restart-on-crash service on the persistent Mac worker, agent jobs anywhere
on the fabric consuming it.

**10.6 RL.** Zero new wire messages; RL is a composition. An inference service serves the current
policy; rollout batch jobs (`needs: policy, cell-runtime`) generate episodes and upload `rollouts`
artifacts, fanned out as a campaign; scoring happens in-process via cells, making the reward
channel cell-signed, with permutation-null yield checks as a standing gate in the scorer; a filter
job packs accepted rollouts into a corpus artifact; the existing train template consumes it and
emits a checkpoint; the CP bounces the policy service with the new checkpoint staged (restart is
fine at current model sizes — hot reload, if ever needed, is the service's own business). The loop
is driven by a controller that is itself a job on the fabric, talking to the CP over the existing
MCP surface (submit, await artifact, read metrics) and stopping on a registered criterion; the
campaign budget bounds its spend, and the controller's transcript joins the experiment record.

## 11. Sequencing

**M1 — DONE** — extracted `chuk-compute-wire` (serde-only generic protocol; lexical guard; ~99%
coverage) and `chuk-compute-worker` (domain-free executor depending only on the wire crate), with
the Hello handshake, batch jobs, and the current single-target build. The control plane translates
`RunSpec`→`Job` and interprets `Artifact` events back into checkpoints (lineage merge moved
control-plane-side). **Behaviour parity proven** on the local demo: a train run completes with
lineage-complete checkpoints, and the E1 resume path yields slices `[[0,10],[10,40]]`. CI runs
both lexical guards. **M2 — DONE** — the control plane serves per-target binaries at
`/agent/<triple>` (+ `.sha256`, `/agent/version`, `/install.sh`), validated against a
`SUPPORTED_TARGETS` allowlist; `scripts/install.sh` (rustup-style: uname → triple → download +
verify + exec) is the single bootstrap the Colab cell and Vast onstart wrap; a CI matrix
cross-builds all three targets (zigbuild for linux-musl, macOS runner for darwin). **Proven: the
Mac joins via `curl <CP>/install.sh | sh`.** (Follow-up: bake the aarch64-musl + darwin CI
artifacts into the deployed image — it currently serves x86_64.) **M3.1 + M3.2 — DONE** (persistent
worker class): long-lived revocable `cw_` tokens bound to a stable id (M3.1); and
**survive-disconnect** (M3.2) — the worker's job supervisor + replay outbox outlive a session, so
a persistent worker keeps its job running across a dropped socket / CP restart and replays
buffered events on reconnect, trimmed by a `HelloAck` high-water the CP dedups by; no lease ⇒
never torn down. **Proven** (kill the CP mid-run → job runs through the outage → reconnect →
replay → completes). **M3.3 — DONE:** a version-mismatched persistent worker self-updates from a
`HelloReject` (the CP sends the target's `/agent/<triple>` URL + sha256; the worker downloads →
verifies → atomically replaces itself → re-execs; a leased worker gets a bare reject and exits).
Proven. **M3 complete.** **M4 — DONE:** the `sys/*` telemetry sampler streams GPU (via `nvidia-smi`
subprocess — the distributed worker is static musl and can't `dlopen` NVML) + CPU/memory (`sysinfo`)
over the existing Metric channel, out-of-band (no `job_id`, not outboxed); the CP ingests it into a
pruned per-worker window and the dashboard renders live gauges + per-metric graphs. macmon-based MPS
(Apple-Silicon GPU) once the Mac is on. M5 — service jobs, registry,
`needs` wiring, secrets; LARQL-on-Mac as first service, cell-runtime second. M6 — campaigns with
budgets and the bench template's pinning gate. M7 — first RL composition: controller job + rollout
campaign + cell-signed scoring against an existing training template.

Each milestone is independently shippable and each is proven with a small real workload from
current work (a v11-scale run, a tokenizer_bench campaign, a broker eval against cell-runtime) —
infrastructure gated on use, per house rules.

## 12. Relationship to the agent/MCP deployment platform

Separate *products*, shared *substrate* — and the crate split above lets you have that without
deciding today, which is its quiet strength.

"Agent platform" splits into two very different things. **Agents-as-experiments** — RL
controllers, broker evals against cell-runtime, agent benchmarks, anything finite that emits
artifacts and lands in the experiments-server — belong on the rig: they're part of the research
loop and interlock with everything else there (they need LARQL, cells, checkpoints, packing).
**Agents-as-production** — always-on MCP servers, public endpoints, the chukai.io fleet — is a
different animal with different physics: uptime SLOs, ingress, TLS, domains, secrets at rest, no
walls, no leases. And crucially, **that platform already exists** — the MCP servers are deployed
and live today on conventional hosting. Don't build an agent platform; don't accidentally rebuild
the one you have inside the training rig.

**Decision rule:** *if it's an experiment, it runs on the rig; if it's a service someone depends
on, it runs on the existing deployment platform.* The rig's service jobs (LARQL, cell-runtime)
pass the test because they're experiment infrastructure — consumed by runs, restartable, private
to the fabric — not products.

A hard platform split *now* would be the wrong separation: the two platforms share ~everything
below the control plane (join, supervise, telemetry, artifact staging, process groups,
spool/replay). Splitting today means either duplicating that substrate or extracting it anyway —
and extracting it anyway is exactly what M1 does. That is the real payoff of the crate split:
**platform separation becomes a control-plane decision, deferrable and cheap.** If an agent
platform ever earns its existence, it is a second CP binary depending on the same
`chuk-compute-wire` and driving the same `chuk-compute-worker`, with its own registry, ingress,
and secrets story — not a rewrite, and not a colonization of the training CP.

**Fork tripwires** — register the conditions, don't pre-build the platform (gate-not-gradient).
Fork the CP when any one fires:

- you need public ingress / TLS / domains for something running on the fabric (the moment the rig
  grows a reverse proxy, it's becoming a PaaS);
- uptime expectations appear that conflict with lease churn and walls;
- agent-specific CP features start outnumbering training features in the backlog;
- anyone other than you depends on a service the rig hosts.

Until one fires: one fabric, one CP, training-first — and the substrate crates keep the exit door
open at near-zero cost.
