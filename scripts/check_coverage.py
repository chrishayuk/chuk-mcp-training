#!/usr/bin/env python3
"""Per-file coverage gate: every source file must have >= 90% line coverage.

Ported from chuk-datasets-server's scripts/check_coverage.py (2026-07-24).
Reads `cargo llvm-cov report --json` on stdin. Exclusions (each needs a
reason here, not in CI config):
  - src/main.rs        — binary glue: env + serve loop, no logic.
  - /examples/          — probes run by scripts, not part of the library.

The BOOTSTRAP_EXCLUDE list below is different in kind: it's not structural
(unlike the two above), it's a temporary allowlist for files that predate
this gate and don't have tests yet. Each entry is a real gap, not a design
choice — see ROADMAP.md's hardening backlog. Shrink this list as files gain
tests; don't add to it to dodge the gate on new code.

Usage: cargo llvm-cov report --json | python3 scripts/check_coverage.py
"""

import json
import sys

THRESHOLD = 90.0
EXCLUDE = ("src/main.rs", "/examples/")

# 2026-07-24 bootstrap snapshot, when the gate was introduced on an existing
# codebase that had never been coverage-gated. Being paid down file by file.
BOOTSTRAP_EXCLUDE = {
    "chuk-train-controlplane/src/api/access.rs",
    "chuk-train-controlplane/src/api/archive.rs",
    "chuk-train-controlplane/src/api/checkpoints.rs",
    "chuk-train-controlplane/src/api/leases.rs",
    "chuk-train-controlplane/src/api/mod.rs",
    "chuk-train-controlplane/src/api/runs.rs",
    "chuk-train-controlplane/src/api/system.rs",
    "chuk-train-controlplane/src/archive.rs",
    "chuk-train-controlplane/src/artifacts/fs.rs",
    "chuk-train-controlplane/src/artifacts/mod.rs",
    "chuk-train-controlplane/src/artifacts/s3.rs",
    "chuk-train-controlplane/src/auth.rs",
    "chuk-train-controlplane/src/apikey.rs",
    "chuk-train-controlplane/src/codeunit.rs",
    "chuk-train-controlplane/src/config.rs",
    "chuk-train-controlplane/src/crypto.rs",
    "chuk-train-controlplane/src/dash.rs",
    "chuk-train-controlplane/src/drive.rs",
    "chuk-train-controlplane/src/experiments.rs",
    "chuk-train-controlplane/src/gate.rs",
    "chuk-train-controlplane/src/grant.rs",
    "chuk-train-controlplane/src/hub/connection.rs",
    "chuk-train-controlplane/src/hub/mirror.rs",
    "chuk-train-controlplane/src/hub/mod.rs",
    "chuk-train-controlplane/src/lease.rs",
    "chuk-train-controlplane/src/provider/mock.rs",
    "chuk-train-controlplane/src/provider/mod.rs",
    "chuk-train-controlplane/src/provider/vast.rs",
    "chuk-train-controlplane/src/store/mod.rs",
    "chuk-train-controlplane/src/store/postgres/runs.rs",
    "chuk-train-controlplane/src/store/postgres/sweeps.rs",
    "chuk-train-controlplane/src/upload.rs",
    "chuk-train-controlplane/src/ws.rs",
    "chuk-train-proto/src/lease.rs",
    "chuk-compute-worker/src/httpclient.rs",
    "chuk-compute-worker/src/selfupdate.rs",
}

data = json.load(sys.stdin)
rows = []
for export in data["data"]:
    for f in export["files"]:
        name = f["filename"]
        if any(pat in name for pat in EXCLUDE):
            continue
        lines = f["summary"]["lines"]
        if lines["count"] == 0:
            continue
        short = name.split("/crates/")[-1]
        rows.append((name, short, lines["percent"], lines["count"]))

rows.sort(key=lambda r: r[2])
gated = [r for r in rows if r[1] not in BOOTSTRAP_EXCLUDE]
failures = [r for r in gated if r[2] < THRESHOLD]
bootstrapped = [r for r in rows if r[1] in BOOTSTRAP_EXCLUDE]

width = max(len(r[1]) for r in rows) if rows else 10
for _, short, pct, count in rows:
    tag = "boot" if short in BOOTSTRAP_EXCLUDE else ("FAIL" if pct < THRESHOLD else "  ok")
    print(f"{tag}  {short:<{width}}  {pct:6.2f}%  ({count} lines)")

seen_names = {r[1] for r in rows}
missing_bootstrap = BOOTSTRAP_EXCLUDE - seen_names
if missing_bootstrap:
    print(
        f"\nnote: {len(missing_bootstrap)} BOOTSTRAP_EXCLUDE entr{'y' if len(missing_bootstrap)==1 else 'ies'} "
        f"no longer appear in the report (deleted, or now filtered elsewhere): "
        + ", ".join(sorted(missing_bootstrap)),
        file=sys.stderr,
    )

if failures:
    print(f"\n{len(failures)} gated file(s) below {THRESHOLD}% line coverage", file=sys.stderr)
    for _, short, pct, count in failures:
        print(f"  {short}  {pct:.2f}%  ({count} lines)", file=sys.stderr)
    sys.exit(1)
print(
    f"\nall {len(gated)} gated files >= {THRESHOLD}% line coverage "
    f"({len(bootstrapped)} bootstrap-excluded, shrink this over time)"
)
