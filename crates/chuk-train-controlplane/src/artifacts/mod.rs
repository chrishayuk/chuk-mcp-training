//! Artifact blob store: content-addressed byte storage behind an adapter
//! trait, mirroring the metadata `Store` seam.
//!
//! Backends: filesystem (`file:`, local dev — a control-plane volume) and
//! S3-compatible (`s3:` / `r2:`, spec §11.5 — R2 preferred for zero egress).
//! With the S3 backend, workers upload/download **directly** via presigned
//! URLs (spec §12: scoped, expiring), so ~500 MB checkpoints never transit the
//! control plane.

mod fs;
mod s3;

/// Store-relative key layout lives in the shared protocol crate so the agent
/// uploads to the exact paths the control plane serves.
pub use chuk_train_proto::keys;

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::SignedUrl;
use sha2::{Digest, Sha256};

pub use fs::FsArtifactStore;
pub use s3::S3ArtifactStore;

const SCHEME_FILE: &str = "file:";
const SCHEME_S3: &str = "s3:";
const SCHEME_R2: &str = "r2:";

/// Byte storage for artifacts. Keys are store-relative POSIX paths under the
/// layout in [`keys`]; the store never interprets them beyond joining.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Write `bytes` at `key` (overwrites); returns the durable storage uri.
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String>;
    /// Read the bytes at `key`.
    async fn get(&self, key: &str) -> Result<Vec<u8>>;
    async fn exists(&self, key: &str) -> Result<bool>;
    /// Delete the object at `key`. Idempotent — a missing key is not an error
    /// (so the retention sweep can re-run safely after R2 lifecycle beat it).
    async fn delete(&self, key: &str) -> Result<()>;
    /// Copy `src` → `dst`. S3/R2 does this server-side (CopyObject — no bytes
    /// transit the control plane, so promoting a ~460 MB final is cheap); the
    /// filesystem backend falls back to read-then-write.
    async fn copy(&self, src: &str, dst: &str) -> Result<()>;
    /// Apply object-expiration lifecycle rules — `(key_prefix, days_to_expire)`.
    /// Lets R2 expire the hot/final checkpoint tiers on a timer (spec §11.5), so
    /// the control plane never has to delete them. No-op for backends without
    /// lifecycle (the filesystem backend has nothing to expire on a schedule).
    async fn apply_lifecycle(&self, _rules: &[(String, i32)]) -> Result<()> {
        Ok(())
    }
    /// The durable storage uri for `key` (no existence check).
    fn uri(&self, key: &str) -> String;
    /// A time-limited direct-download URL, or `None` when the backend has none
    /// (the filesystem backend has none — the control plane serves those bytes
    /// itself). Agents and lazarus fetch large artifacts through this.
    async fn presign_get(&self, _key: &str, _ttl: Duration) -> Result<Option<SignedUrl>> {
        Ok(None)
    }
    /// A time-limited direct-upload URL, or `None` when the backend has none
    /// (filesystem — the agent falls back to uploading through the control
    /// plane). Workers PUT checkpoint bytes straight to this.
    async fn presign_put(&self, _key: &str, _ttl: Duration) -> Result<Option<SignedUrl>> {
        Ok(None)
    }
}

/// Open an artifact store from a URL-ish spec: `file:/path` (or a bare path)
/// for the filesystem backend; `s3://bucket` / `r2://bucket` for S3-compatible
/// (endpoint, region, and credentials come from the environment).
pub fn open_artifact_store(spec: &str) -> Result<Box<dyn ArtifactStore>> {
    if let Some(bucket) = spec
        .strip_prefix(SCHEME_S3)
        .or_else(|| spec.strip_prefix(SCHEME_R2))
    {
        let bucket = bucket.trim_start_matches('/');
        return Ok(Box::new(S3ArtifactStore::from_env(bucket)?));
    }
    let root = spec.strip_prefix(SCHEME_FILE).unwrap_or(spec);
    Ok(Box::new(FsArtifactStore::new(root)))
}

/// Lowercase hex sha256 of `bytes` — the harness's content-address function.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use chuk_train_proto::env;

    use super::*;

    // These four vars aren't read anywhere else in this crate's test suite
    // (they're only sourced from `.env` by the `main.rs` binary entrypoint,
    // never by `cargo test`), so a dedicated env-var mutex would be overkill
    // for one test — same reasoning as `datasets::tests::from_env_is_none_…`.
    fn clear_s3_env() {
        std::env::remove_var(env::S3_ENDPOINT);
        std::env::remove_var(env::S3_REGION);
        std::env::remove_var(env::S3_ACCESS_KEY_ID);
        std::env::remove_var(env::S3_SECRET_ACCESS_KEY);
    }

    #[test]
    fn file_scheme_and_bare_path_both_select_the_filesystem_backend() {
        let scheme = open_artifact_store("file:/tmp/chuk-train-artifacts").unwrap();
        assert_eq!(
            scheme.uri("ckpt-hot/run-1/step-10.pt"),
            "file:///tmp/chuk-train-artifacts/ckpt-hot/run-1/step-10.pt"
        );

        // No recognised scheme at all still resolves to the filesystem
        // backend, treating the whole spec as a root path.
        let bare = open_artifact_store("/tmp/chuk-train-artifacts").unwrap();
        assert_eq!(
            bare.uri("ckpt-hot/run-1/step-10.pt"),
            "file:///tmp/chuk-train-artifacts/ckpt-hot/run-1/step-10.pt"
        );
    }

    // Both cases below touch the same four process-global env vars, so they
    // live in one #[test] (run sequenced, not interleaved with a sibling
    // test) rather than two — see `clear_s3_env`'s doc comment.
    #[test]
    fn s3_backend_requires_credentials_then_selects_and_strips_the_bucket() {
        clear_s3_env();

        // No credentials at all: refused, naming the missing var.
        let result = open_artifact_store("s3://my-bucket");
        let Err(err) = result else {
            panic!("expected missing S3 credentials to error");
        };
        assert!(
            err.to_string().contains(env::S3_ENDPOINT),
            "expected the missing-endpoint error to name {}, got: {err}",
            env::S3_ENDPOINT
        );

        std::env::set_var(env::S3_ENDPOINT, "https://example.r2.cloudflarestorage.com");
        std::env::set_var(env::S3_ACCESS_KEY_ID, "test-access-key");
        std::env::set_var(env::S3_SECRET_ACCESS_KEY, "test-secret-key");

        // `s3://bucket` and `r2://bucket` both strip the leading `//` down to
        // a bare bucket name, and both dispatch to the S3-compatible backend
        // (R2 is just S3 with a different endpoint) — its `uri()` is always
        // `s3://`-prefixed regardless of which scheme opened it.
        let s3 = open_artifact_store("s3://my-bucket").unwrap();
        assert_eq!(s3.uri("k"), "s3://my-bucket/k");

        let r2 = open_artifact_store("r2://my-bucket").unwrap();
        assert_eq!(r2.uri("k"), "s3://my-bucket/k");

        clear_s3_env();
    }

    #[tokio::test]
    async fn filesystem_backend_has_no_lifecycle_or_presigning() {
        // The filesystem backend doesn't override the trait's defaults (only
        // the S3/R2 backend does — it serves bytes itself, so there's no
        // expiry timer or direct-transfer URL to hand out).
        let store = open_artifact_store("file:/tmp/chuk-train-artifacts").unwrap();
        store
            .apply_lifecycle(&[("ckpt-hot/".to_owned(), 1)])
            .await
            .expect("apply_lifecycle is a no-op, not an error");
        assert_eq!(
            store.presign_get("k", Duration::from_secs(60)).await.unwrap(),
            None
        );
        assert_eq!(
            store.presign_put("k", Duration::from_secs(60)).await.unwrap(),
            None
        );
    }

    #[test]
    fn sha256_hex_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
