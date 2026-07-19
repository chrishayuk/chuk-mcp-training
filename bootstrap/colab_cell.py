# ── chuk-compute · Colab worker bootstrap (E0) ──────────────────────────────
# Paste this into ONE cell of a Colab notebook set to a GPU runtime (T4).
# It runs the control plane's one-shot installer, which detects this box's
# target, downloads + checksum-verifies the matching worker, and joins the
# fleet. The cell blocks while the worker runs — keep it running for the session.
#
# Fill in two values from your Fly deploy:
CP_URL = "https://YOUR-APP.fly.dev"     # your control plane (HTTP base)
JOIN_TOKEN = "PASTE_JOIN_TOKEN"         # the CHUK_TRAIN_JOIN_TOKEN fly secret
LABELS = "colab,t4"                     # shows up in `fleet`

import subprocess

base = CP_URL.rstrip("/")
print(f"[chuk-compute] installing + joining {base} as '{LABELS}'…")
# curl <CP>/install.sh | sh -s -- --cp <CP> --token <TOKEN> --labels <LABELS>
subprocess.run(
    f"curl -fsSL {base}/install.sh | sh -s -- "
    f"--cp {base} --token {JOIN_TOKEN} --labels {LABELS}",
    shell=True,
    check=False,
)
