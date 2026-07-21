# stub-trainer

A dependency-free trainer that honours the chuk-train script contract — the
harness's demo/dogfood unit (`scripts/demo.sh`) and the EI0 introspection
proof vehicle.

## Introspection (I0 pulse tier)

`run.sh` points `$CHUK_PROBE_PLAN` at the shipped `probe_plan.json` (the unit
enables introspection itself at I0; CP-delivered plans arrive with I1). When
torch (≥ 2.1) and `chuk_introspect` are importable, each step trains a tiny
real torch model inside `Introspector.pulse(...)`, streaming
`introspect/act_norm/L*`, `grad_norm`, `dead_frac`, `logit_entropy`,
`overhead_pct` beside the synthetic telemetry. Missing either dependency, the
stub runs exactly as before (a log line says why introspection is off).

Vendor the library into the unit before `build_code_unit` (it is pure Python;
torch comes from the environment — Colab's preinstalled CUDA torch, or a local
venv):

```sh
mkdir -p vendor && cp -R ../../introspect/src/chuk_introspect vendor/
```

`train.py` adds `vendor/` to `sys.path` automatically. `chuk_introspect` also
needs `pydantic>=2.7` (preinstalled on Colab).

### EI0 gate proof

Submit with `configs/poisoned.json` — the tiny model's ReLU is born dead, so
`introspect/dead_frac/L1` reads 1.0 from step 1 and a watchdog gate
`last(introspect/dead_frac/L1) > 0.5 → stop_run` fires.
