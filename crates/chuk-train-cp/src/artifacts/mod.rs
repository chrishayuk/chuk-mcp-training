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
