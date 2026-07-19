#!/usr/bin/env python3
"""One-time Google Drive authorization for the chuk-train control plane.

Runs the OAuth *offline* flow so the control plane gets a long-lived refresh
token it can use to archive checkpoints/logs to your Drive. Reuses the OAuth
client's already-registered redirect (http://localhost:8000/oauth/callback), so
nothing new needs registering in Google Cloud Console.

Usage (from the repo root, with the client in .env or the environment):

    CHUK_TRAIN_GOOGLE_CLIENT_ID=... CHUK_TRAIN_GOOGLE_CLIENT_SECRET=... \
        python3 scripts/authorize-drive.py

It prints a consent URL: open it, approve, and the script prints the refresh
token. Set it as a Fly secret:

    fly secrets set -a chuk-mcp-training CHUK_TRAIN_GOOGLE_REFRESH_TOKEN=<token>
"""

from __future__ import annotations

import http.server
import os
import urllib.parse
import urllib.request
import webbrowser

# drive.file = only files this app creates; enough for our archive, nothing else.
SCOPE = "https://www.googleapis.com/auth/drive.file"
REDIRECT = "http://localhost:8000/oauth/callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

CLIENT_ID = os.environ.get("CHUK_TRAIN_GOOGLE_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = (
    os.environ.get("CHUK_TRAIN_GOOGLE_CLIENT_SECRET") or os.environ.get("GOOGLE_CLIENT_SECRET", "")
)

_result: dict[str, str] = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _result["code"] = params.get("code", [""])[0]
        _result["error"] = params.get("error", [""])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "authorization received — you can close this tab" if _result["code"] else "authorization failed"
        self.wfile.write(f"<html><body style='font-family:sans-serif'>{msg}</body></html>".encode())

    def log_message(self, *args):  # silence the default logging
        pass


def exchange(code: str) -> dict:
    data = urllib.parse.urlencode({
        "code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT, "grant_type": "authorization_code",
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data)) as r:
        import json
        return json.load(r)


def main() -> None:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit("set CHUK_TRAIN_GOOGLE_CLIENT_ID and CHUK_TRAIN_GOOGLE_CLIENT_SECRET")

    consent = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": CLIENT_ID, "redirect_uri": REDIRECT, "response_type": "code",
        "scope": SCOPE, "access_type": "offline", "prompt": "consent",
    })
    print("\nOpen this URL and approve access:\n\n  " + consent + "\n", flush=True)
    try:
        webbrowser.open(consent)
    except Exception:
        pass

    server = http.server.HTTPServer(("127.0.0.1", 8000), Handler)
    print("waiting for the redirect on http://localhost:8000/oauth/callback …", flush=True)
    while "code" not in _result:
        server.handle_request()

    if _result.get("error") or not _result.get("code"):
        raise SystemExit(f"authorization failed: {_result.get('error') or 'no code'}")

    tokens = exchange(_result["code"])
    refresh = tokens.get("refresh_token")
    if not refresh:
        raise SystemExit("no refresh_token returned (revoke prior grant and retry with prompt=consent)")
    print("\n=== refresh token (set as CHUK_TRAIN_GOOGLE_REFRESH_TOKEN) ===\n")
    print(refresh + "\n", flush=True)


if __name__ == "__main__":
    main()
