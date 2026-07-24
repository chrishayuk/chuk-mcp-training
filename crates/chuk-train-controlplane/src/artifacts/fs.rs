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

    async fn delete(&self, key: &str) -> Result<()> {
        let path = self.resolve(key)?;
        tokio::task::spawn_blocking(move || match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Idempotent: already gone is success.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(anyhow::Error::from(e))
                .with_context(|| format!("removing {}", path.display())),
        })
        .await??;
        Ok(())
    }

    async fn copy(&self, src: &str, dst: &str) -> Result<()> {
        let src_path = self.resolve(src)?;
        let dst_path = self.resolve(dst)?;
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(parent) = dst_path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::copy(&src_path, &dst_path).with_context(|| {
                format!("copy {} -> {}", src_path.display(), dst_path.display())
            })?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    fn uri(&self, key: &str) -> String {
        format!("file://{}/{key}", self.root.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp directory removed on drop. Small local helper to avoid a dep —
    /// mirrors `codeunit::TempDir` (same reasoning, kept local rather than
    /// shared since it's only needed here and there).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!(
                "chuk-train-fs-test-{}",
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[tokio::test]
    async fn put_then_get_round_trips_bytes_under_a_nested_key_creating_parents() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        let uri = store
            .put("runs/r1/step_5/model.bin", b"hello".to_vec())
            .await
            .unwrap();
        assert_eq!(
            uri,
            format!("file://{}/runs/r1/step_5/model.bin", root.0.display())
        );
        assert_eq!(store.get("runs/r1/step_5/model.bin").await.unwrap(), b"hello");
    }

    #[tokio::test]
    async fn put_overwrites_existing_bytes_atomically() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        store.put("k", b"first".to_vec()).await.unwrap();
        store.put("k", b"second".to_vec()).await.unwrap();
        assert_eq!(store.get("k").await.unwrap(), b"second");
    }

    #[tokio::test]
    async fn get_of_a_missing_key_errors() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        let err = store.get("missing").await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("no such file"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn exists_reflects_presence() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        assert!(!store.exists("missing").await.unwrap());
        store.put("present", b"x".to_vec()).await.unwrap();
        assert!(store.exists("present").await.unwrap());
    }

    #[tokio::test]
    async fn delete_removes_the_file_and_is_idempotent_once_gone() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        store.put("k", b"x".to_vec()).await.unwrap();
        store.delete("k").await.unwrap();
        assert!(!store.exists("k").await.unwrap());
        // Already gone: still Ok, not an error (retention sweeps re-run safely).
        store.delete("k").await.unwrap();
    }

    #[tokio::test]
    async fn copy_duplicates_bytes_leaves_src_intact_and_creates_dst_parents() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        store.put("src", b"payload".to_vec()).await.unwrap();
        store.copy("src", "nested/dst").await.unwrap();
        assert_eq!(store.get("nested/dst").await.unwrap(), b"payload");
        assert_eq!(store.get("src").await.unwrap(), b"payload");
    }

    #[tokio::test]
    async fn copy_of_a_missing_src_errors() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        let err = store.copy("missing", "dst").await.unwrap_err();
        assert!(err.to_string().contains("copy"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn unsafe_keys_are_rejected_by_every_operation_before_touching_disk() {
        let root = TempDir::new();
        let store = FsArtifactStore::new(&root.0);
        let unsafe_key = "../escape";

        let put_err = store.put(unsafe_key, b"x".to_vec()).await.unwrap_err();
        let get_err = store.get(unsafe_key).await.unwrap_err();
        let exists_err = store.exists(unsafe_key).await.unwrap_err();
        let delete_err = store.delete(unsafe_key).await.unwrap_err();
        let copy_src_err = store.copy(unsafe_key, "dst").await.unwrap_err();
        let copy_dst_err = store.copy("src", unsafe_key).await.unwrap_err();

        for err in [put_err, get_err, exists_err, delete_err, copy_src_err, copy_dst_err] {
            assert!(
                err.to_string().contains("unsafe artifact key"),
                "unexpected error: {err}"
            );
        }
    }

    #[test]
    fn uri_formats_as_a_file_scheme_url_joining_root_and_key() {
        let store = FsArtifactStore::new("/data/artifacts");
        assert_eq!(
            store.uri("ckpt-hot/r1/step_5/model.bin"),
            "file:///data/artifacts/ckpt-hot/r1/step_5/model.bin"
        );
    }
}
