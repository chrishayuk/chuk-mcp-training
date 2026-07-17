FROM rust:1.95-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p chuk-train-cp

FROM debian:bookworm-slim
COPY --from=build /src/target/release/chuk-train-cp /usr/local/bin/chuk-train-cp
ENV CHUK_TRAIN_STORE=sqlite:/data/chuk_train.db
EXPOSE 8700
CMD ["chuk-train-cp"]
