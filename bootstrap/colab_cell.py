# --- chuk-train E0 bootstrap: paste this into one Colab cell -----------------
# Downloads the static agent binary and joins this notebook's GPU to the fleet.
# Until a GitHub release exists, host the binary anywhere reachable (R2 works)
# and point AGENT_BINARY_URL at it. Build it with:
#   cargo build --release -p chuk-train-agent --target x86_64-unknown-linux-musl

CP_WSS_URL = "wss://YOUR-APP.fly.dev/ws/agent"
JOIN_TOKEN = "PASTE_JOIN_TOKEN"
AGENT_BINARY_URL = "https://YOUR-HOST/chuk-train-agent-x86_64-unknown-linux-musl"
LABELS = "colab"

import os
import stat
import subprocess
import urllib.request

AGENT_PATH = "/tmp/chuk-train-agent"
urllib.request.urlretrieve(AGENT_BINARY_URL, AGENT_PATH)
os.chmod(AGENT_PATH, os.stat(AGENT_PATH).st_mode | stat.S_IEXEC)
subprocess.run(
    [AGENT_PATH, "--url", CP_WSS_URL, "--token", JOIN_TOKEN, "--labels", LABELS],
    check=False,
)
