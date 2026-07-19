//! Google Drive cold-archive client (archive tier).
//!
//! Drive is the *cold* half of the storage split: R2 stays the hot artifact
//! store (presigned, zero-egress), and completed runs' checkpoints/logs are
//! tiered here to free R2 while keeping the user's 5 TB of Drive as durable,
//! browsable cold storage. This module is the low-level Drive v3 wrapper the
//! archive/retention job drives; it does not implement [`ArtifactStore`]
//! because Drive is file-id + folder addressed, not POSIX-key addressed.
//!
//! What it does:
//!   * refreshes a long-lived offline token into short-lived access tokens
//!     (cached until just before expiry),
//!   * ensures a nested folder path exists (`chuk-train/runs/<id>/…`), caching
//!     each resolved folder id so a run's many checkpoint files cost one walk,
//!   * uploads bytes via Drive's **resumable** protocol in 256 KiB-aligned
//!     chunks, so a dropped connection retries a chunk rather than a ~460 MB
//!     object,
//!   * downloads by file id and deletes by file id.
//!
//! Auth uses the `drive.file` scope: this client only ever sees the files it
//! created, never the rest of the user's Drive.
//!
//! The upload/download/delete/ensure methods are wired by the archive +
//! retrieval job (a separate task); the module is allowed to carry them ahead
//! of that caller.
#![allow(dead_code)]

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use chuk_train_proto::env as env_vars;
use reqwest::header::{CONTENT_RANGE, LOCATION, RANGE};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use tokio::sync::Mutex;

const OAUTH_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const DRIVE_FILES_URL: &str = "https://www.googleapis.com/drive/v3/files";
const DRIVE_UPLOAD_URL: &str = "https://www.googleapis.com/upload/drive/v3/files";
const FOLDER_MIME: &str = "application/vnd.google-apps.folder";
const OCTET_STREAM: &str = "application/octet-stream";
const GRANT_REFRESH: &str = "refresh_token";
/// Drive's "My Drive" root; a valid parent for the top archive folder.
const ROOT_PARENT: &str = "root";
/// Top-level archive folder created under the user's Drive root.
pub const ARCHIVE_ROOT_FOLDER: &str = "chuk-train";

/// Resumable chunk size. Drive requires every non-final chunk to be a multiple
/// of 256 KiB; 8 MiB keeps request count low without holding much extra.
const UPLOAD_CHUNK: usize = 8 * 1024 * 1024;
/// Refresh the access token this many seconds before it actually expires, so a
/// long upload never straddles the boundary mid-request.
const TOKEN_SLACK_S: f64 = 60.0;
/// Per-chunk retry budget: on a transient failure we re-query the committed
/// offset and resume, up to this many times before giving up on the object.
const MAX_CHUNK_RETRIES: u32 = 4;

/// A cached access token with its absolute expiry (unix seconds).
struct CachedToken {
    access_token: String,
    expires_at: f64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    /// Lifetime in seconds; Google returns an integer, parsed loosely as f64.
    expires_in: f64,
}

#[derive(Deserialize)]
struct FileRef {
    id: String,
}

#[derive(Deserialize)]
struct FileList {
    #[serde(default)]
    files: Vec<FileRef>,
}

/// A Drive client bound to one user's offline grant.
pub struct DriveClient {
    http: Client,
    client_id: String,
    client_secret: String,
    refresh_token: String,
    token: Mutex<Option<CachedToken>>,
    /// Full-path → folder-id cache (e.g. `chuk-train/runs/r1` → `<id>`).
    folders: Mutex<HashMap<String, String>>,
}

impl DriveClient {
    /// Build from the environment. Returns `Ok(None)` when the archive tier is
    /// simply off (no refresh token) — the client id/secret alone belong to the
    /// dashboard sign-in, so their presence must not imply Drive is configured.
    pub fn from_env() -> Result<Option<Self>> {
        let refresh_token = match std::env::var(env_vars::GOOGLE_REFRESH_TOKEN) {
            Ok(v) if !v.trim().is_empty() => v,
            _ => return Ok(None),
        };
        let client_id = std::env::var(env_vars::GOOGLE_CLIENT_ID).with_context(|| {
            format!(
                "{} is set but {} is not — the archive tier needs both",
                env_vars::GOOGLE_REFRESH_TOKEN,
                env_vars::GOOGLE_CLIENT_ID
            )
        })?;
        let client_secret = std::env::var(env_vars::GOOGLE_CLIENT_SECRET).with_context(|| {
            format!(
                "{} is set but {} is not — the archive tier needs both",
                env_vars::GOOGLE_REFRESH_TOKEN,
                env_vars::GOOGLE_CLIENT_SECRET
            )
        })?;
        Ok(Some(Self {
            http: Client::new(),
            client_id,
            client_secret,
            refresh_token,
            token: Mutex::new(None),
            folders: Mutex::new(HashMap::new()),
        }))
    }

    /// A valid access token, refreshing (and caching) when the cached one is
    /// missing or within [`TOKEN_SLACK_S`] of expiry.
    async fn access_token(&self) -> Result<String> {
        let mut slot = self.token.lock().await;
        if let Some(tok) = slot.as_ref() {
            if tok.expires_at - TOKEN_SLACK_S > now() {
                return Ok(tok.access_token.clone());
            }
        }
        let resp = self
            .http
            .post(OAUTH_TOKEN_URL)
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", self.refresh_token.as_str()),
                ("grant_type", GRANT_REFRESH),
            ])
            .send()
            .await
            .context("drive token refresh request")?;
        let resp = error_for_status_with_body(resp, "drive token refresh").await?;
        let body: TokenResponse = resp.json().await.context("parsing drive token response")?;
        let access_token = body.access_token;
        *slot = Some(CachedToken {
            access_token: access_token.clone(),
            expires_at: now() + body.expires_in,
        });
        Ok(access_token)
    }

    /// Verify the grant is usable (one token refresh); for a health probe.
    pub async fn probe(&self) -> Result<()> {
        self.access_token().await.map(|_| ())
    }

    /// Resolve a `/`-separated folder path under the user's Drive root to its
    /// leaf folder id, creating any missing segments. Cached per path prefix.
    pub async fn ensure_folder_path(&self, path: &str) -> Result<String> {
        let mut parent = ROOT_PARENT.to_owned();
        let mut prefix = String::new();
        for segment in folder_segments(path) {
            if prefix.is_empty() {
                prefix.push_str(segment);
            } else {
                prefix.push('/');
                prefix.push_str(segment);
            }
            if let Some(id) = self.folders.lock().await.get(&prefix).cloned() {
                parent = id;
                continue;
            }
            let id = match self.find_child_folder(&parent, segment).await? {
                Some(id) => id,
                None => self.create_folder(&parent, segment).await?,
            };
            self.folders.lock().await.insert(prefix.clone(), id.clone());
            parent = id;
        }
        Ok(parent)
    }

    /// Find a child folder by name under `parent`, if it already exists.
    async fn find_child_folder(&self, parent: &str, name: &str) -> Result<Option<String>> {
        let token = self.access_token().await?;
        let query = format!(
            "name = '{}' and '{}' in parents and mimeType = '{}' and trashed = false",
            escape_query(name),
            escape_query(parent),
            FOLDER_MIME,
        );
        let resp = self
            .http
            .get(DRIVE_FILES_URL)
            .bearer_auth(&token)
            .query(&[
                ("q", query.as_str()),
                ("fields", "files(id)"),
                ("spaces", "drive"),
                ("pageSize", "1"),
            ])
            .send()
            .await
            .context("drive folder lookup request")?;
        let resp = error_for_status_with_body(resp, "drive folder lookup").await?;
        let list: FileList = resp.json().await.context("parsing drive folder list")?;
        Ok(list.files.into_iter().next().map(|f| f.id))
    }

    /// Create a folder named `name` under `parent`; returns its id.
    async fn create_folder(&self, parent: &str, name: &str) -> Result<String> {
        let token = self.access_token().await?;
        let metadata = serde_json::json!({
            "name": name,
            "mimeType": FOLDER_MIME,
            "parents": [parent],
        });
        let resp = self
            .http
            .post(DRIVE_FILES_URL)
            .bearer_auth(&token)
            .query(&[("fields", "id")])
            .json(&metadata)
            .send()
            .await
            .context("drive folder create request")?;
        let resp = error_for_status_with_body(resp, "drive folder create").await?;
        let created: FileRef = resp.json().await.context("parsing created folder")?;
        Ok(created.id)
    }

    /// Ensure `folder_path` exists, then upload `bytes` as `name` inside it.
    /// Returns the new file's Drive id (the archive location the caller records).
    pub async fn upload_to_path(
        &self,
        folder_path: &str,
        name: &str,
        mime: Option<&str>,
        bytes: &[u8],
    ) -> Result<String> {
        let parent = self.ensure_folder_path(folder_path).await?;
        self.upload(&parent, name, mime.unwrap_or(OCTET_STREAM), bytes)
            .await
    }

    /// Resumable upload of `bytes` as a new file `name` under `parent_id`.
    async fn upload(&self, parent_id: &str, name: &str, mime: &str, bytes: &[u8]) -> Result<String> {
        let session = self.start_resumable(parent_id, name, mime, bytes.len()).await?;
        self.put_chunks(&session, bytes).await
    }

    /// Open a resumable-upload session; returns the session URI to PUT bytes to.
    async fn start_resumable(
        &self,
        parent_id: &str,
        name: &str,
        mime: &str,
        total: usize,
    ) -> Result<String> {
        let token = self.access_token().await?;
        let metadata = serde_json::json!({ "name": name, "parents": [parent_id] });
        let resp = self
            .http
            .post(DRIVE_UPLOAD_URL)
            .bearer_auth(&token)
            .query(&[("uploadType", "resumable"), ("fields", "id")])
            .header("X-Upload-Content-Type", mime)
            .header("X-Upload-Content-Length", total.to_string())
            .json(&metadata)
            .send()
            .await
            .context("drive resumable initiate request")?;
        let resp = error_for_status_with_body(resp, "drive resumable initiate").await?;
        let session = resp
            .headers()
            .get(LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .context("drive resumable initiate: no Location header")?;
        Ok(session)
    }

    /// PUT `bytes` to a resumable session in [`UPLOAD_CHUNK`]-sized pieces,
    /// resuming from the server's committed offset on transient failure.
    async fn put_chunks(&self, session: &str, bytes: &[u8]) -> Result<String> {
        let total = bytes.len() as u64;
        // Empty object: a single zero-length finalising PUT.
        if total == 0 {
            let resp = self
                .http
                .put(session)
                .header(CONTENT_RANGE, format!("bytes */{total}"))
                .send()
                .await
                .context("drive resumable empty finalise")?;
            return finalise_id(resp).await;
        }

        let mut offset: u64 = 0;
        let mut retries = 0u32;
        while offset < total {
            let end = (offset + UPLOAD_CHUNK as u64).min(total);
            let chunk = bytes[offset as usize..end as usize].to_vec();
            let resp = self
                .http
                .put(session)
                .header(CONTENT_RANGE, content_range(offset, end, total))
                .body(chunk)
                .send()
                .await;

            match resp {
                // Non-final chunk accepted: Drive replies 308 Resume Incomplete.
                Ok(r) if r.status() == StatusCode::PERMANENT_REDIRECT => {
                    offset = end;
                    retries = 0;
                }
                // Final chunk: 200/201 with the file metadata.
                Ok(r) if r.status().is_success() => return finalise_id(r).await,
                // Anything else: re-query committed offset and resume, bounded.
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    if retries >= MAX_CHUNK_RETRIES {
                        bail!("drive chunk upload failed ({status}): {body}");
                    }
                    retries += 1;
                    offset = self.committed_offset(session, total).await?;
                }
                Err(e) => {
                    if retries >= MAX_CHUNK_RETRIES {
                        return Err(e).context("drive chunk upload transport");
                    }
                    retries += 1;
                    offset = self.committed_offset(session, total).await?;
                }
            }
        }
        bail!("drive upload ended without a finalising response");
    }

    /// Query how many bytes the resumable session has committed, so a retry
    /// resumes at the right offset. A fresh session reports nothing → 0.
    async fn committed_offset(&self, session: &str, total: u64) -> Result<u64> {
        let resp = self
            .http
            .put(session)
            .header(CONTENT_RANGE, format!("bytes */{total}"))
            .send()
            .await
            .context("drive resumable status query")?;
        // 200/201 here means it actually finished; treat as fully committed.
        if resp.status().is_success() {
            return Ok(total);
        }
        // 308 carries `Range: bytes=0-<last>`; absence means zero committed.
        let committed = resp
            .headers()
            .get(RANGE)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_committed_end)
            .map(|last| last + 1)
            .unwrap_or(0);
        Ok(committed)
    }

    /// Download a file's bytes by id.
    pub async fn download(&self, file_id: &str) -> Result<Vec<u8>> {
        let token = self.access_token().await?;
        let url = format!("{DRIVE_FILES_URL}/{file_id}");
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&token)
            .query(&[("alt", "media")])
            .send()
            .await
            .context("drive download request")?;
        let resp = error_for_status_with_body(resp, "drive download").await?;
        Ok(resp.bytes().await.context("reading drive download body")?.to_vec())
    }

    /// Permanently delete a file by id (used when a checkpoint is pruned).
    pub async fn delete(&self, file_id: &str) -> Result<()> {
        let token = self.access_token().await?;
        let url = format!("{DRIVE_FILES_URL}/{file_id}");
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&token)
            .send()
            .await
            .context("drive delete request")?;
        error_for_status_with_body(resp, "drive delete").await?;
        Ok(())
    }
}

/// Split a `/`-separated Drive path into non-empty segments.
fn folder_segments(path: &str) -> impl Iterator<Item = &str> {
    path.split('/').filter(|s| !s.is_empty())
}

/// `Content-Range` value for a chunk covering `[start, end)` of `total` bytes.
fn content_range(start: u64, end: u64, total: u64) -> String {
    format!("bytes {start}-{}/{total}", end - 1)
}

/// Escape a value for embedding in a Drive `q` query string literal.
fn escape_query(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

/// Parse the last committed byte out of a resumable `Range: bytes=0-<last>`.
fn parse_committed_end(range: &str) -> Option<u64> {
    range.rsplit('-').next().and_then(|n| n.parse().ok())
}

/// Extract the created file's id from a finalising upload response.
async fn finalise_id(resp: reqwest::Response) -> Result<String> {
    let resp = error_for_status_with_body(resp, "drive upload finalise").await?;
    let created: FileRef = resp.json().await.context("parsing uploaded file id")?;
    Ok(created.id)
}

/// Like `Response::error_for_status`, but attaches the response body so Drive's
/// JSON error (the useful part) survives into the anyhow chain.
async fn error_for_status_with_body(resp: reqwest::Response, what: &str) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    bail!("{what} failed ({status}): {body}")
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_skip_empties_and_slashes() {
        let got: Vec<_> = folder_segments("/chuk-train//runs/r1/").collect();
        assert_eq!(got, vec!["chuk-train", "runs", "r1"]);
    }

    #[test]
    fn content_range_is_inclusive_end() {
        assert_eq!(content_range(0, 100, 460), "bytes 0-99/460");
        assert_eq!(content_range(100, 460, 460), "bytes 100-459/460");
    }

    #[test]
    fn parse_committed_handles_range_and_absence() {
        assert_eq!(parse_committed_end("bytes=0-262143"), Some(262_143));
        assert_eq!(parse_committed_end("0-5"), Some(5));
        assert_eq!(parse_committed_end("bytes=0-"), None);
    }

    #[test]
    fn escape_query_neutralises_quotes() {
        assert_eq!(escape_query("o'brien"), "o\\'brien");
        assert_eq!(escape_query(r"a\b"), r"a\\b");
    }

    /// Live round-trip against real Drive. Ignored by default (needs a grant in
    /// the env); run with `.env` sourced:
    ///   cargo test -p chuk-train-controlplane drive::tests::live_round_trip -- --ignored --nocapture
    /// Uses a >1-chunk payload so the resumable 308 → finalise path is exercised.
    #[ignore]
    #[tokio::test]
    async fn live_round_trip() {
        let client = match DriveClient::from_env().expect("build client") {
            Some(c) => c,
            None => {
                eprintln!("skip: no {} in env", env_vars::GOOGLE_REFRESH_TOKEN);
                return;
            }
        };
        client.probe().await.expect("token refresh");

        // 10 MiB pattern → two chunks (8 MiB + 2 MiB): a 308 then a finalise.
        let payload: Vec<u8> = (0..10 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
        let folder = format!("{ARCHIVE_ROOT_FOLDER}/_smoke");
        let suffix = now() as u64;
        let name = format!("round-trip-{suffix}.bin");

        let file_id = client
            .upload_to_path(&folder, &name, None, &payload)
            .await
            .expect("upload");
        eprintln!("uploaded {name} -> {file_id}");

        let got = client.download(&file_id).await.expect("download");
        assert_eq!(got.len(), payload.len(), "size round-trips");
        assert_eq!(got, payload, "bytes round-trip");

        client.delete(&file_id).await.expect("delete");
        eprintln!("deleted {file_id} — round-trip ok");
    }
}
