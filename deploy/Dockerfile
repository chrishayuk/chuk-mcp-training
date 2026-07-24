# Builds on Fly's x86_64 Linux builder. The control plane runs inside this
# image (native gnu is fine); the agent is built static musl so it is portable
# to any Colab / Vast box the control plane serves it to.
#
# Build context is the PARENT of this repo (see deploy/fly.toml), not this repo
# alone: chuk-datasets-client is a path dependency into the sibling
# chuk-datasets-server repo (`../chuk-datasets-server` relative to this repo's
# Cargo.toml, same as local dev under chris-source/), so that sibling has to be
# present in the build context and mirrored at the same relative layout inside
# the image.
FROM rust:1-bookworm AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
    && rustup target add x86_64-unknown-linux-musl \
    && rm -rf /var/lib/apt/lists/*
COPY chuk-datasets-server/Cargo.toml chuk-datasets-server/Cargo.lock chuk-datasets-server/
COPY chuk-datasets-server/crates chuk-datasets-server/crates
COPY chuk-mcp-training/Cargo.toml chuk-mcp-training/Cargo.lock chuk-mcp-training/
COPY chuk-mcp-training/crates chuk-mcp-training/crates
# The control plane embeds scripts/install.sh via include_str!, so it must be in
# the build context (not just crates/).
COPY chuk-mcp-training/scripts chuk-mcp-training/scripts
WORKDIR /src/chuk-mcp-training
# Control plane: native, runs inside the runtime image below.
RUN cargo build --release -p chuk-train-controlplane
# Agent: static musl, downloaded + run on bare Colab/Vast Linux.
RUN cargo build --release -p chuk-compute-worker --target x86_64-unknown-linux-musl

FROM debian:bookworm-slim
# git is required at runtime: build_code_unit's git-URL path (codeunit.rs)
# shells out to `git clone` for any repo that isn't already a local directory
# on this container -- i.e. every real caller. Without it here, every non-local
# build_code_unit call fails at process-spawn time with a misleading
# "running git (is it installed?)" error, indistinguishable from a bad repo/commit.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/chuk-mcp-training/target/release/chuk-train-controlplane /usr/local/bin/chuk-train-controlplane
# Per-target worker binaries the control plane serves at /agent/<triple>
# (chuk-compute M2). Only the x86_64-musl target is cross-built in this
# x86_64-Linux image; aarch64-musl + the darwin targets come from the CI
# release matrix (see .github/workflows/ci.yml) and are a follow-up to bake in.
COPY --from=build /src/chuk-mcp-training/target/x86_64-unknown-linux-musl/release/chuk-compute-worker \
     /app/agents/x86_64-unknown-linux-musl
# Store comes from CHUK_TRAIN_STORE at runtime (Neon via `fly secrets`); no
# image-level default so nothing silently shadows the secret.
ENV CHUK_TRAIN_AGENT_DIR=/app/agents
# The mock provider (local only) launches this path; unused in the deploy.
ENV CHUK_TRAIN_AGENT_BIN=/app/agents/x86_64-unknown-linux-musl
EXPOSE 8700
CMD ["chuk-train-controlplane"]
