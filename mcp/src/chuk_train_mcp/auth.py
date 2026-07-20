"""Caller identity for the HTTP transport (mirrors chuk-experiments-server).

Served over HTTP, this MCP server is a zero-credential proxy: every tool call
forwards the *calling agent's* own bearer token to the control plane, which
enforces RBAC per caller — the server itself holds no key. Served over stdio
there is no HTTP request (and no ASGI scope to read), so the client falls back
to the process's own CHUK_TRAIN_API_TOKEN.

chuk_mcp_server doesn't pass the request into @mcp.tool functions; its
ContextMiddleware stores the raw ASGI scope in a ContextVar, read back here.
"""

from __future__ import annotations

AUTHORIZATION_HEADER = "authorization"
BEARER_PREFIX = "Bearer "
# Header bytes are latin-1 per the ASGI spec.
HEADER_ENCODING = "latin-1"


def _ambient_scope() -> dict | None:
    from chuk_mcp_server.context import get_http_request

    return get_http_request()


def http_request_active() -> bool:
    """Whether this call is being served over HTTP (vs stdio)."""
    return _ambient_scope() is not None


def bearer_from_mcp_context() -> str | None:
    """The calling agent's bearer token from the ambient request, if any."""
    scope = _ambient_scope()
    if not scope:
        return None
    for key, value in scope.get("headers", []):
        if key.decode(HEADER_ENCODING).lower() == AUTHORIZATION_HEADER:
            header = value.decode(HEADER_ENCODING)
            if header.startswith(BEARER_PREFIX):
                header = header[len(BEARER_PREFIX) :]
            return header.strip() or None
    return None
