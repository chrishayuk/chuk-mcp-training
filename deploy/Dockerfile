# Builds on Fly's x86_64 Linux builder. The control plane runs inside this
# image (native gnu is fine); the agent is built static musl so it is portable
# to any Colab / Vast box the control plane serves it to.
FROM rust:1-bookworm AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends musl-tools \
    && rustup target add x86_64-unknown-linux-musl \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# Control plane: native, runs inside the runtime image below.
RUN cargo build --release -p chuk-train-cp
# Agent: static musl, downloaded + run on bare Colab/Vast Linux.
RUN cargo build --release -p chuk-train-agent --target x86_64-unknown-linux-musl

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/chuk-train-cp /usr/local/bin/chuk-train-cp
COPY --from=build /src/target/x86_64-unknown-linux-musl/release/chuk-train-agent /app/chuk-train-agent
ENV CHUK_TRAIN_STORE=sqlite:/data/chuk_train.db
# The control plane serves this at /agent/linux-x86_64 for the bootstrap cell.
ENV CHUK_TRAIN_AGENT_BIN=/app/chuk-train-agent
EXPOSE 8700
CMD ["chuk-train-cp"]
