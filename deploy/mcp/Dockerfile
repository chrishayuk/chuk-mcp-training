# chuk-train-mcp — the hosted MCP surface (HTTP, per-caller bearer passthrough).
# Build context is the repo root:
#   fly deploy -c deploy/mcp/fly.toml --dockerfile deploy/mcp/Dockerfile

FROM python:3.12-slim AS build
RUN pip install --no-cache-dir uv
COPY mcp/ /build/
WORKDIR /build
RUN uv pip install --system --no-cache .

FROM python:3.12-slim
RUN useradd --create-home mcp
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/chuk-train-mcp /usr/local/bin/chuk-train-mcp
USER mcp
EXPOSE 8710
# Zero credentials on purpose: every MCP caller's own bearer is forwarded to
# the control plane, which enforces RBAC per caller. Only the CP URL is set
# (fly.toml [env]).
CMD ["chuk-train-mcp", "--http", "--host", "0.0.0.0", "--port", "8710"]
