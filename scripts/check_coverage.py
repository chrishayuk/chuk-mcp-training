#!/usr/bin/env python3
"""Per-file coverage gate: every source file must have >= 90% line coverage.

Ported from chuk-datasets-server's scripts/check_coverage.py (2026-07-24).
Reads `cargo llvm-cov report --json` on stdin. Exclusions (each needs a
reason here, not in CI config):
  - src/main.rs        — binary glue: env + serve loop, no logic.
  - /examples/          — probes run by scripts, not part of the library.

Note that cargo-llvm-cov itself also drops `tests.rs` files and `tests/`
directories from the report. That is where the `#[ignore]`d live checks live
(`drive/tests.rs`, `experiments/tests.rs`, `artifacts/s3/tests.rs`): they need
real R2 / Drive / experiments-server credentials, so CI can never run them and
counting them here would only measure how much unrunnable code a module
carries. Everything else — including a module's ordinary unit tests — is
gated.

There is no bootstrap allowlist: as of 2026-07-25 every gated file clears the
threshold on its own (the last six — auth, drive, experiments, lease, ws and
artifacts/s3 — were paid down against loopback fakes for Google, Drive, the
experiments-server, S3/R2 and the worker websocket). Don't reintroduce one to
dodge the gate on new code.

Usage: cargo llvm-cov report --json | python3 scripts/check_coverage.py
"""

import json
import sys

THRESHOLD = 90.0
EXCLUDE = ("src/main.rs", "/examples/")

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
failures = [r for r in rows if r[2] < THRESHOLD]

width = max(len(r[1]) for r in rows) if rows else 10
for _, short, pct, count in rows:
    tag = "FAIL" if pct < THRESHOLD else "  ok"
    print(f"{tag}  {short:<{width}}  {pct:6.2f}%  ({count} lines)")

if failures:
    print(f"\n{len(failures)} file(s) below {THRESHOLD}% line coverage", file=sys.stderr)
    for _, short, pct, count in failures:
        print(f"  {short}  {pct:.2f}%  ({count} lines)", file=sys.stderr)
    sys.exit(1)
print(f"\nall {len(rows)} files >= {THRESHOLD}% line coverage")
