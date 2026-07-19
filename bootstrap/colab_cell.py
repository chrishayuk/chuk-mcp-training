# ── chuk-train · Colab worker bootstrap (E0) ────────────────────────────────
# Paste this into ONE cell of a Colab notebook set to a GPU runtime (T4).
# It downloads the agent from your control plane and joins the fleet. The cell
# blocks while the agent runs — keep it running for the session's lifetime.
#
# Fill in two values from your Fly deploy:
CP_URL = "https://YOUR-APP.fly.dev"     # your control plane
JOIN_TOKEN = "PASTE_JOIN_TOKEN"         # the CHUK_TRAIN_JOIN_TOKEN fly secret
LABELS = "colab,t4"                     # shows up in `fleet`

import os
import stat
import subprocess
import urllib.request

base = CP_URL.rstrip("/")
agent_path = "/tmp/chuk-compute-worker"

# 1. Download the agent binary the control plane serves (public, no auth).
urllib.request.urlretrieve(base + "/agent/linux-x86_64", agent_path)
os.chmod(agent_path, os.stat(agent_path).st_mode | stat.S_IEXEC)

# 2. Derive the websocket URL and dial home (blocks; the agent runs here).
ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + "/ws/agent"
print(f"[chuk-train] joining {ws_url} as '{LABELS}'…")
subprocess.run(
    [agent_path, "--url", ws_url, "--token", JOIN_TOKEN, "--labels", LABELS],
    check=False,
)
