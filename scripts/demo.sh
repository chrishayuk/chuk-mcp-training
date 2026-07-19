#!/usr/bin/env bash
# Bring the operator dashboard to life locally — and dogfood the agent path.
#
# Starts a self-contained control plane (local SQLite + file artifacts, mock
# provider launching REAL agent processes), provisions a couple of workers,
# builds the stub-trainer code unit, and submits a mix of runs (training,
# shell, one that fails). The stub-trainer streams realistic loss/logs/
# checkpoints, so the dashboard fills with live data.
#
#   ./scripts/demo.sh          # then open the printed URL, paste the token
#   Ctrl-C                     # stops the control plane + mock workers
#
# It uses an ISOLATED env (no Google auth, a throwaway SQLite db) so nothing
# touches the shared Neon/prod deployment.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
SCRATCH="${TMPDIR:-/tmp}/chuk-train-demo"
mkdir -p "$SCRATCH"

# Load tokens + Google sign-in creds from the repo .env, then override
# EVERYTHING that must be local so the demo never touches prod Neon / R2 / Fly.
set -a; [ -f "$REPO/.env" ] && . "$REPO/.env"; set +a
export CHUK_TRAIN_STORE="sqlite:$SCRATCH/demo.db"
export CHUK_TRAIN_ARTIFACTS="file:$SCRATCH/artifacts"
export CHUK_TRAIN_PROVIDERS="mock"
export CHUK_TRAIN_AGENT_BIN="$REPO/target/debug/chuk-train-agent"
export CHUK_TRAIN_AGENT_WS_URL="ws://127.0.0.1:8700/ws/agent"
export CHUK_TRAIN_PUBLIC_URL="http://127.0.0.1:8700"
export CHUK_TRAIN_HOST="127.0.0.1"
export CHUK_TRAIN_PORT="8700"
export CHUK_TRAIN_RECONCILE_S="3"
export CHUK_TRAIN_GOOGLE_REFRESH_TOKEN=""   # Drive archive off in the demo
# Google sign-in stays ON (client id/secret + allowlist from .env), so the
# dashboard uses Google auth — same as prod, not an API-token box. The OAuth
# client must allow redirect  http://127.0.0.1:8700/auth/callback.

BASE="http://127.0.0.1:8700"
TOK="$CHUK_TRAIN_API_TOKEN"   # used only for the seed's own /api calls below

echo "› building control plane + agent (debug)…"
cargo build -q -p chuk-train-cp -p chuk-train-agent
rm -f "$SCRATCH"/demo.db*

echo "› starting control plane…"
# Run the built binary from an EMPTY dir so it does not pick up the repo's .env
# (which points at prod Neon / the Fly wss endpoint / Google auth). The demo is
# fully isolated — only the exported vars above configure it.
( cd "$SCRATCH" && exec "$REPO/target/debug/chuk-train-cp" ) >"$SCRATCH/cp.log" 2>&1 &
CP_PID=$!
cleanup(){ echo; echo "› stopping demo…"; kill "$CP_PID" 2>/dev/null || true; pkill -f 'target/debug/chuk-train-agent' 2>/dev/null || true; }
trap cleanup INT TERM EXIT

for _ in $(seq 1 60); do curl -fsS "$BASE/healthz" >/dev/null 2>&1 && break; sleep 0.5; done
curl -fsS "$BASE/healthz" >/dev/null 2>&1 || { echo "control plane did not come up — see $SCRATCH/cp.log"; exit 1; }

post(){ curl -fsS -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" -X POST "$BASE$1" -d "$2"; }

echo "› provisioning mock workers…"
for _ in 1 2; do
  post /api/provision '{"provider":"mock","lease_min":60,"gpu":"mock-t4","max_price_hr":0.5}' >/dev/null || true
  sleep 0.4
done

echo "› building stub-trainer code unit…"
SHA=$(post /api/code_units "{\"repo\":\"$REPO/examples/stub-trainer\",\"name\":\"stub-trainer\"}" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["code"]["sha"])')
echo "  sha=$SHA"

submit_train(){ # name seed total_steps step_delay ckpt_every
  local body
  body=$(RUN_NAME="$1" RUN_SEED="$2" RUN_STEPS="$3" RUN_DELAY="$4" RUN_CKPT="$5" SHA="$SHA" python3 -c '
import os, json
name = os.environ["RUN_NAME"]
print(json.dumps({
  "name": name,
  "spec": {"kind":"train","code":{"name":"stub-trainer","sha":os.environ["SHA"]},
           "entrypoint":"train","config":"configs/demo.json",
           "overrides":{"total_steps":int(os.environ["RUN_STEPS"]),
                        "step_delay_s":float(os.environ["RUN_DELAY"]),
                        "checkpoint_every":int(os.environ["RUN_CKPT"])},
           "seed":int(os.environ["RUN_SEED"]),
           "links":[{"kind":"wandb","label":"Weights & Biases","url":"https://wandb.ai/chukai/tiny-model/runs/"+name},
                    {"kind":"exp","label":"experiments-server","url":"https://experiments.chukai.io/runs/"+name}]}}))')
  post /api/runs "$body" >/dev/null
}

echo "› submitting runs…"
# A long lead run so there's always live streaming to watch (~16 min), plus
# shorter ones that complete + churn the fleet.
submit_train v11-pretrain      81 2000 0.5 100
submit_train v11-warmup        81   60 0.4  15
submit_train v11-sweep-seed7    7  140 0.5  30
post /api/runs/shell '{"name":"probe-t4","command":"echo Tesla T4 15360MiB; echo matmul 8192^3 = 4.1 TFLOP/s; echo done"}' >/dev/null
post /api/runs/shell '{"name":"broken-probe","command":"echo starting; sleep 1; echo boom 1>&2; exit 1"}' >/dev/null

cat <<EOF

===================================================================
 dashboard   $BASE   → sign in with Google ($CHUK_TRAIN_ALLOWED_EMAILS)
 cp log      $SCRATCH/cp.log
 Ctrl-C to stop the control plane + mock workers.
 (needs http://127.0.0.1:8700/auth/callback on the OAuth client)
===================================================================
EOF
wait "$CP_PID"
