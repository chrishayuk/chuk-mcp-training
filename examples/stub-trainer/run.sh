#!/bin/sh
# Stub-trainer entrypoint. I0 stance (chuk-introspect spec): the unit enables
# introspection itself by pointing $CHUK_PROBE_PLAN at the plan it ships —
# CP-delivered plans arrive with I1. Everything else is `python3 train.py`.
set -e
cd "$(dirname "$0")"
export CHUK_PROBE_PLAN="${CHUK_PROBE_PLAN:-$PWD/probe_plan.json}"
exec python3 train.py
