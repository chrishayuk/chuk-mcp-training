//! Filesystem artifact backend: bytes under a root directory, written
//! atomically (temp file + rename) so a reader never sees a partial blob.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use async_trait::async_trait;

use super::keys::is_safe_key;
use super::ArtifactStore;

pub struct FsArtifactStore {
    root: PathBuf,
}

impl FsArtifactStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    fn resolve(&self, key: &str) -> Result<PathBuf> {
        anyhow::ensure!(is_safe_key(key), "unsafe artifact key: {key:?}");
        Ok(self.root.join(key))
    }
}

#[async_trait]
impl ArtifactStore for FsArtifactStore {
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String> {
        let path = self.resolve(key)?;
        let uri = self.uri(key);
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            // Same-directory temp keeps the final rename atomic (same filesystem).
            let tmp = path.with_extension("tmp-write");
            std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
            std::fs::rename(&tmp, &path)
                .with_context(|| format!("renaming into {}", path.display()))?;
            Ok(())
        })
        .await??;
        Ok(uri)
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.resolve(key)?;
        let bytes = tokio::task::spawn_blocking(move || std::fs::read(&path)).await??;
        Ok(bytes)
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let path = self.resolve(key)?;
        Ok(tokio::task::spawn_blocking(move || path.exists()).await?)
    }

    fn uri(&self, key: &str) -> String {
        format!("file://{}/{key}", self.root.display())
    }
}
