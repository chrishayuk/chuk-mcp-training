//! Artifact blob store: content-addressed byte storage behind an adapter
//! trait, mirroring the metadata `Store` seam.
//!
//! M1 ships the filesystem backend (one Fly volume). `s3:` / `r2:` is reserved
//! for the off-box backend the spec argues for (§11.5, R2 zero-egress) — it
//! slots in behind this trait without touching callers.

mod fs;

/// Store-relative key layout lives in the shared protocol crate so the agent
/// uploads to the exact paths the control plane serves.
pub use chuk_train_proto::keys;

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use chuk_train_proto::SignedUrl;
use sha2::{Digest, Sha256};

pub use fs::FsArtifactStore;

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
    /// The durable storage uri for `key` (no existence check).
    fn uri(&self, key: &str) -> String;
    /// A backend-native time-limited fetch URL, or `None` when the backend has
    /// none (the filesystem backend has none — the control plane serves those
    /// bytes from its own authenticated `/api/blob` endpoint instead).
    fn presign_get(&self, _key: &str, _ttl: Duration) -> Result<Option<SignedUrl>> {
        Ok(None)
    }
}

/// Open an artifact store from a URL-ish spec: `file:/path` (or a bare path)
/// for the filesystem backend; `s3:`/`r2:` reserved.
pub fn open_artifact_store(spec: &str) -> Result<Box<dyn ArtifactStore>> {
    if spec.starts_with(SCHEME_S3) || spec.starts_with(SCHEME_R2) {
        anyhow::bail!(
            "s3/r2 artifact backend is reserved for a later milestone; use file: for now"
        );
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
