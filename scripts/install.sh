#!/bin/sh
# chuk-compute worker installer (chuk-compute-spec §4).
#
# Detects this machine's target triple, downloads the matching worker binary +
# checksum from the control plane, verifies it, and execs the worker joined to
# the fleet. Served by the control plane at /install.sh; the Colab cell and the
# Vast onstart are one-line wrappers over it, and a Mac runs it directly.
#
#   curl -fsSL <CP>/install.sh | sh -s -- --cp <CP_URL> --token <JOIN_TOKEN> \
#       [--labels a,b] [--lease-min N] [--worker-id ID] [--drain-window-min N]
#
# --cp is the control plane's HTTP(S) base; the websocket URL is derived from it.
set -eu

CP=""
TOKEN=""
FORWARD=""   # extra args passed through to the worker

while [ $# -gt 0 ]; do
    case "$1" in
        --cp) CP="$2"; shift 2 ;;
        --token) TOKEN="$2"; shift 2 ;;
        --labels) FORWARD="$FORWARD --labels $2"; shift 2 ;;
        --lease-min) FORWARD="$FORWARD --lease-min $2"; shift 2 ;;
        --worker-id) FORWARD="$FORWARD --worker-id $2"; shift 2 ;;
        --drain-window-min) FORWARD="$FORWARD --drain-window-min $2"; shift 2 ;;
        *) echo "chuk-compute install: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$CP" ] || { echo "chuk-compute install: --cp <CP_URL> is required" >&2; exit 2; }
[ -n "$TOKEN" ] || { echo "chuk-compute install: --token <JOIN_TOKEN> is required" >&2; exit 2; }
CP="${CP%/}"

# 1. Detect the target triple from uname (must match SUPPORTED_TARGETS).
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
    Linux)
        case "$arch" in
            x86_64|amd64) target="x86_64-unknown-linux-musl" ;;
            aarch64|arm64) target="aarch64-unknown-linux-musl" ;;
            *) echo "chuk-compute install: unsupported Linux arch: $arch" >&2; exit 3 ;;
        esac ;;
    Darwin)
        case "$arch" in
            arm64|aarch64) target="aarch64-apple-darwin" ;;
            x86_64) target="x86_64-apple-darwin" ;;
            *) echo "chuk-compute install: unsupported macOS arch: $arch" >&2; exit 3 ;;
        esac ;;
    *) echo "chuk-compute install: unsupported OS: $os" >&2; exit 3 ;;
esac

# 2. Download the binary + its checksum, and verify.
bin="$(mktemp)"
trap 'rm -f "$bin"' EXIT
echo "chuk-compute install: fetching $target worker from $CP …" >&2
curl -fsSL "$CP/agent/$target" -o "$bin"
want="$(curl -fsSL "$CP/agent/$target.sha256" | awk '{print $1}')"
if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$bin" | awk '{print $1}')"
else
    got="$(shasum -a 256 "$bin" | awk '{print $1}')"
fi
if [ "$want" != "$got" ]; then
    echo "chuk-compute install: checksum mismatch (want $want, got $got)" >&2
    exit 4
fi
chmod +x "$bin"

# 3. Derive the websocket URL and exec the worker (blocks for the session).
ws="$(printf '%s' "$CP" | sed -e 's#^https://#wss://#' -e 's#^http://#ws://#')/ws/agent"
echo "chuk-compute install: joining $ws …" >&2
# shellcheck disable=SC2086
exec "$bin" --url "$ws" --token "$TOKEN" $FORWARD
