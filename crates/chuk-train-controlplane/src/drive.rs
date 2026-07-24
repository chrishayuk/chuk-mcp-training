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
    /// The three Google endpoints this client calls, held as values rather than
    /// used as constants directly so tests can point it at a loopback Drive.
    token_url: String,
    files_url: String,
    upload_url: String,
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
            token_url: OAUTH_TOKEN_URL.to_owned(),
            files_url: DRIVE_FILES_URL.to_owned(),
            upload_url: DRIVE_UPLOAD_URL.to_owned(),
            token: Mutex::new(None),
            folders: Mutex::new(HashMap::new()),
        }))
    }

    /// A client pointed at a fake Drive on `base`, bypassing the env gating —
    /// for tests that spin one up (mirrors `datasets.rs`'s `Datasets::at`).
    #[cfg(test)]
    pub(crate) fn at(base: &str) -> Self {
        Self {
            http: Client::new(),
            client_id: "test-client-id".to_owned(),
            client_secret: "test-client-secret".to_owned(),
            refresh_token: "test-refresh-token".to_owned(),
            token_url: format!("{base}/token"),
            files_url: format!("{base}/drive/v3/files"),
            upload_url: format!("{base}/upload/drive/v3/files"),
            token: Mutex::new(None),
            folders: Mutex::new(HashMap::new()),
        }
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
            .post(&self.token_url)
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
            .get(&self.files_url)
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
            .post(&self.files_url)
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
            .post(&self.upload_url)
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
        let url = format!("{}/{file_id}", self.files_url);
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
        let url = format!("{}/{file_id}", self.files_url);
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
    use crate::fakehttp::{FakeHttp, Received, Reply, REFUSED_ORIGIN};

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

    // -- the client itself, against a loopback fake Drive -------------------
    //
    // Everything below drives the real reqwest client through the real Drive v3
    // request shapes (token refresh + caching, the folder walk, the resumable
    // upload's 308/finalise/resume protocol, download, delete) against
    // `fakehttp`. The live round-trip against Google's own servers stays in
    // `drive/tests.rs`.

    const TOKEN_JSON: &str = r#"{"access_token":"ya29.access","expires_in":3600}"#;

    /// A Drive that answers the token refresh and hands out sequential file ids
    /// for lookups/creates. `handler` sees only the non-token requests.
    fn fake_drive<F>(handler: F) -> (FakeHttp, DriveClient)
    where
        F: Fn(&Received, usize) -> Reply + Send + Sync + 'static,
    {
        let server = FakeHttp::start(move |req, nth| {
            if req.path().ends_with("/token") {
                Reply::ok(TOKEN_JSON)
            } else {
                handler(req, nth)
            }
        });
        let client = DriveClient::at(&server.origin);
        (server, client)
    }

    /// Drive's "no such folder" answer to a lookup: an empty file list.
    const NO_MATCH: &str = r#"{"files":[]}"#;

    #[tokio::test]
    async fn the_access_token_is_refreshed_once_and_then_served_from_the_cache() {
        let (server, client) = fake_drive(|_, _| Reply::ok(NO_MATCH));
        client.probe().await.expect("probe refreshes the grant");
        let first = client.access_token().await.expect("token");
        let second = client.access_token().await.expect("cached token");

        assert_eq!(first, "ya29.access");
        assert_eq!(second, first);
        assert_eq!(server.hits(), 1, "one refresh serves every later call");
        let refresh = &server.requests()[0];
        assert_eq!(refresh.method, "POST");
        let form = String::from_utf8(refresh.body.clone()).expect("form body");
        assert!(form.contains("grant_type=refresh_token"), "unexpected form: {form}");
        assert!(form.contains("refresh_token=test-refresh-token"), "unexpected form: {form}");
    }

    #[tokio::test]
    async fn a_token_inside_the_slack_window_is_refreshed_again() {
        let (server, client) = fake_drive(|_, _| Reply::ok(NO_MATCH));
        // Still valid, but inside TOKEN_SLACK_S of expiry: a long upload must
        // not straddle the boundary, so it refreshes now.
        *client.token.lock().await = Some(CachedToken {
            access_token: "about-to-expire".to_owned(),
            expires_at: now() + TOKEN_SLACK_S / 2.0,
        });
        assert_eq!(client.access_token().await.expect("token"), "ya29.access");
        assert_eq!(server.hits(), 1);
    }

    #[tokio::test]
    async fn a_refused_grant_surfaces_googles_own_error_body() {
        let server = FakeHttp::start(|_, _| Reply::new(400, r#"{"error":"invalid_grant"}"#));
        let client = DriveClient::at(&server.origin);
        let error = client.probe().await.unwrap_err();
        assert!(error.to_string().contains("drive token refresh"), "unexpected error: {error}");
        assert!(error.to_string().contains("invalid_grant"), "the body must survive: {error}");
    }

    #[tokio::test]
    async fn an_unreachable_drive_is_a_transport_error_not_a_panic() {
        let client = DriveClient::at(REFUSED_ORIGIN);
        let error = client.probe().await.unwrap_err();
        assert!(
            error.to_string().contains("drive token refresh request"),
            "unexpected error: {error}"
        );
    }

    // -- the folder walk ----------------------------------------------------

    #[tokio::test]
    async fn ensure_folder_path_creates_each_missing_segment_under_the_last() {
        let (server, client) = fake_drive(|req, _| match req.method.as_str() {
            // Nothing exists yet...
            "GET" => Reply::ok(NO_MATCH),
            // ...so each segment is created, named after itself for the assert.
            _ => {
                let name = req.json()["name"].as_str().expect("name").to_owned();
                Reply::ok(format!(r#"{{"id":"id-{name}"}}"#))
            }
        });

        let leaf = client
            .ensure_folder_path("chuk-train/runs/r1")
            .await
            .expect("ensure folder path");
        assert_eq!(leaf, "id-r1");

        let creates: Vec<serde_json::Value> = server
            .requests()
            .iter()
            .filter(|r| r.method == "POST" && !r.path().ends_with("/token"))
            .map(Received::json)
            .collect();
        assert_eq!(creates.len(), 3, "one create per segment");
        assert_eq!(creates[0]["parents"], serde_json::json!(["root"]));
        assert_eq!(creates[1]["parents"], serde_json::json!(["id-chuk-train"]));
        assert_eq!(creates[2]["parents"], serde_json::json!(["id-runs"]));
        assert_eq!(creates[2]["mimeType"], FOLDER_MIME);
    }

    #[tokio::test]
    async fn an_existing_folder_is_found_rather_than_duplicated() {
        let (server, client) = fake_drive(|req, _| {
            assert_eq!(req.method, "GET", "an existing folder must not be re-created");
            Reply::ok(r#"{"files":[{"id":"already-there"}]}"#)
        });

        assert_eq!(client.ensure_folder_path("chuk-train").await.unwrap(), "already-there");
        // The lookup escapes the name into Drive's `q` grammar.
        let lookup = &server.requests()[1];
        assert!(lookup.target.contains("trashed"), "unexpected query: {}", lookup.target);
    }

    #[tokio::test]
    async fn a_resolved_folder_is_cached_so_a_runs_many_files_cost_one_walk() {
        let (server, client) = fake_drive(|_, _| Reply::ok(r#"{"files":[{"id":"cached"}]}"#));
        client.ensure_folder_path("chuk-train/runs").await.unwrap();
        let after_first = server.hits();
        client.ensure_folder_path("chuk-train/runs").await.unwrap();
        assert_eq!(server.hits(), after_first, "the second walk hit the cache only");
    }

    #[tokio::test]
    async fn a_refused_folder_lookup_is_reported_with_its_body() {
        let (_server, client) = fake_drive(|_, _| Reply::new(403, "insufficientPermissions"));
        let error = client.ensure_folder_path("chuk-train").await.unwrap_err();
        assert!(error.to_string().contains("drive folder lookup"), "unexpected error: {error}");
        assert!(error.to_string().contains("insufficientPermissions"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn a_refused_folder_create_is_reported_with_its_body() {
        let (_server, client) = fake_drive(|req, _| match req.method.as_str() {
            "GET" => Reply::ok(NO_MATCH),
            _ => Reply::new(403, "storageQuotaExceeded"),
        });
        let error = client.ensure_folder_path("chuk-train").await.unwrap_err();
        assert!(error.to_string().contains("drive folder create"), "unexpected error: {error}");
        assert!(error.to_string().contains("storageQuotaExceeded"), "unexpected error: {error}");
    }

    // -- the resumable upload ------------------------------------------------

    // The resumable protocol's session URI is a URL Drive hands back, so these
    // handlers are built with `start_with_origin`: they need the address of the
    // very server they are about to run on.

    #[tokio::test]
    async fn upload_walks_the_folder_path_then_finalises_in_one_chunk() {
        let payload = b"a small checkpoint".to_vec();
        let total = payload.len();
        let server = FakeHttp::start_with_origin(move |origin, req, _| {
            match (req.method.as_str(), req.path()) {
                (_, p) if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
                ("GET", "/drive/v3/files") => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
                ("POST", "/upload/drive/v3/files") => {
                    Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
                }
                _ => {
                    assert_eq!(
                        req.header(CONTENT_RANGE.as_str()),
                        format!("bytes 0-{}/{total}", total - 1)
                    );
                    Reply::ok(r#"{"id":"uploaded-file"}"#)
                }
            }
        });
        let client = DriveClient::at(&server.origin);

        let id = client
            .upload_to_path("chuk-train/runs/r1", "model.bin", None, &payload)
            .await
            .expect("upload");
        assert_eq!(id, "uploaded-file");

        let initiate = server
            .requests()
            .into_iter()
            .find(|r| r.path() == "/upload/drive/v3/files")
            .expect("initiate");
        assert_eq!(initiate.header("x-upload-content-type"), OCTET_STREAM);
        assert_eq!(initiate.header("x-upload-content-length"), total.to_string());
        assert_eq!(initiate.json()["parents"], serde_json::json!(["folder-id"]));
        assert_eq!(initiate.json()["name"], "model.bin");
    }

    #[tokio::test]
    async fn a_multi_chunk_upload_resumes_after_each_308_and_reassembles_exactly() {
        // Just over one chunk, so the 308 → next-chunk → finalise path runs.
        let payload: Vec<u8> = (0..UPLOAD_CHUNK + 1024).map(|i| (i % 251) as u8).collect();
        let total = payload.len();
        let server = FakeHttp::start_with_origin(move |origin, req, _| {
            match (req.method.as_str(), req.path()) {
                (_, p) if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
                ("GET", "/drive/v3/files") => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
                ("POST", "/upload/drive/v3/files") => {
                    Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
                }
                _ => {
                    let range = req.header(CONTENT_RANGE.as_str());
                    if range.starts_with(&format!("bytes 0-{}", UPLOAD_CHUNK - 1)) {
                        Reply::new(308, Vec::new())
                    } else {
                        assert_eq!(range, format!("bytes {UPLOAD_CHUNK}-{}/{total}", total - 1));
                        Reply::ok(r#"{"id":"big-file"}"#)
                    }
                }
            }
        });
        let client = DriveClient::at(&server.origin);

        let id = client
            .upload_to_path("chuk-train", "big.bin", Some("application/x-tar"), &payload)
            .await
            .expect("upload");
        assert_eq!(id, "big-file");

        let sent: Vec<u8> = server
            .requests()
            .into_iter()
            .filter(|r| r.path() == "/session")
            .flat_map(|r| r.body)
            .collect();
        assert_eq!(sent.len(), total, "every byte was sent exactly once");
        assert_eq!(sent, payload, "and reassembles to the original object");
    }

    #[tokio::test]
    async fn an_empty_object_is_a_single_zero_length_finalising_put() {
        let server = FakeHttp::start_with_origin(|origin, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            "/upload/drive/v3/files" => {
                Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
            }
            _ => {
                assert_eq!(req.header(CONTENT_RANGE.as_str()), "bytes */0");
                assert!(req.body.is_empty());
                Reply::ok(r#"{"id":"empty-file"}"#)
            }
        });
        let client = DriveClient::at(&server.origin);
        let id = client.upload_to_path("chuk-train", "empty.bin", None, &[]).await.unwrap();
        assert_eq!(id, "empty-file");
    }

    #[tokio::test]
    async fn an_initiate_without_a_session_uri_is_an_error() {
        let server = FakeHttp::start_with_origin(|_, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            // 200, but no Location header.
            _ => Reply::ok(Vec::new()),
        });
        let client = DriveClient::at(&server.origin);
        let error = client.upload_to_path("chuk-train", "x.bin", None, b"x").await.unwrap_err();
        assert!(error.to_string().contains("no Location header"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn a_refused_initiate_is_reported_with_its_body() {
        let server = FakeHttp::start_with_origin(|_, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            _ => Reply::new(403, "quotaExceeded"),
        });
        let client = DriveClient::at(&server.origin);
        let error = client.upload_to_path("chuk-train", "x.bin", None, b"x").await.unwrap_err();
        assert!(error.to_string().contains("drive resumable initiate"), "unexpected error: {error}");
        assert!(error.to_string().contains("quotaExceeded"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn a_failed_chunk_is_re_sent_from_the_offset_drive_says_it_committed() {
        // The first chunk PUT fails; the status query reports 512 bytes
        // committed, so the retry resumes from there rather than restarting.
        let payload: Vec<u8> = (0..2048).map(|i| (i % 251) as u8).collect();
        let server = FakeHttp::start_with_origin(|origin, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            "/upload/drive/v3/files" => {
                Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
            }
            _ => {
                let range = req.header(CONTENT_RANGE.as_str()).to_owned();
                if range == "bytes */2048" {
                    // The status query: 512 bytes are committed.
                    Reply::new(308, Vec::new()).header(RANGE.as_str(), "bytes=0-511")
                } else if range.starts_with("bytes 0-") {
                    Reply::new(503, "backendError")
                } else {
                    assert_eq!(range, "bytes 512-2047/2048");
                    Reply::ok(r#"{"id":"resumed-file"}"#)
                }
            }
        });
        let client = DriveClient::at(&server.origin);

        let id = client.upload_to_path("chuk-train", "x.bin", None, &payload).await.unwrap();
        assert_eq!(id, "resumed-file");
    }

    #[tokio::test]
    async fn a_status_query_that_reports_no_committed_range_resumes_from_zero() {
        let payload: Vec<u8> = vec![7; 1024];
        let server = FakeHttp::start_with_origin(|origin, req, nth| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            "/upload/drive/v3/files" => {
                Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
            }
            _ => {
                let range = req.header(CONTENT_RANGE.as_str()).to_owned();
                if range == "bytes */1024" {
                    Reply::new(308, Vec::new()) // no Range header at all
                } else if nth < 4 {
                    Reply::new(503, "backendError")
                } else {
                    assert_eq!(range, "bytes 0-1023/1024", "restarted from the beginning");
                    Reply::ok(r#"{"id":"restarted-file"}"#)
                }
            }
        });
        let client = DriveClient::at(&server.origin);
        assert_eq!(
            client.upload_to_path("chuk-train", "x.bin", None, &payload).await.unwrap(),
            "restarted-file"
        );
    }

    #[tokio::test]
    async fn a_status_query_that_says_it_already_finished_stops_re_sending() {
        let payload: Vec<u8> = vec![7; 1024];
        let server = FakeHttp::start_with_origin(|origin, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            "/upload/drive/v3/files" => {
                Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
            }
            _ if req.header(CONTENT_RANGE.as_str()) == "bytes */1024" => {
                // 200 to the status query: the object actually landed.
                Reply::ok(r#"{"id":"already-there"}"#)
            }
            // Every real chunk fails, so only the status query can end this.
            _ => Reply::new(503, "backendError"),
        });
        let client = DriveClient::at(&server.origin);

        // Offset reaches total, the loop exits without a finalising response.
        let error = client.upload_to_path("chuk-train", "x.bin", None, &payload).await.unwrap_err();
        assert!(
            error.to_string().contains("without a finalising response"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn a_chunk_that_keeps_failing_gives_up_after_the_retry_budget() {
        let payload: Vec<u8> = vec![7; 1024];
        let server = FakeHttp::start_with_origin(|origin, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            "/upload/drive/v3/files" => {
                Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{origin}/session"))
            }
            _ if req.header(CONTENT_RANGE.as_str()) == "bytes */1024" => {
                Reply::new(308, Vec::new()).header(RANGE.as_str(), "bytes=0-0")
            }
            _ => Reply::new(500, "internalError"),
        });
        let client = DriveClient::at(&server.origin);

        let error = client.upload_to_path("chuk-train", "x.bin", None, &payload).await.unwrap_err();
        assert!(error.to_string().contains("drive chunk upload failed"), "unexpected error: {error}");
        assert!(error.to_string().contains("internalError"), "unexpected error: {error}");
        // MAX_CHUNK_RETRIES retries, then the giving-up attempt.
        let chunks = server
            .requests()
            .iter()
            .filter(|r| r.path() == "/session" && !r.body.is_empty())
            .count();
        assert_eq!(chunks as u32, MAX_CHUNK_RETRIES + 1);
    }

    #[tokio::test]
    async fn a_dropped_connection_mid_upload_is_a_transport_error_after_the_budget() {
        // The session URI points somewhere that refuses connections, so every
        // chunk PUT *and* every status query fails at the transport layer.
        let server = FakeHttp::start_with_origin(|_, req, _| match req.path() {
            p if p.ends_with("/token") => Reply::ok(TOKEN_JSON),
            "/drive/v3/files" => Reply::ok(r#"{"files":[{"id":"folder-id"}]}"#),
            _ => Reply::ok(Vec::new()).header(LOCATION.as_str(), format!("{REFUSED_ORIGIN}/session")),
        });
        let client = DriveClient::at(&server.origin);

        let error = client.upload_to_path("chuk-train", "x.bin", None, b"bytes").await.unwrap_err();
        assert!(
            error.to_string().contains("drive resumable status query"),
            "unexpected error: {error}"
        );
    }

    // -- download / delete ---------------------------------------------------

    #[tokio::test]
    async fn download_asks_for_the_media_and_returns_the_bytes() {
        let (server, client) = fake_drive(|_, _| Reply::ok(b"checkpoint bytes".to_vec()));
        let got = client.download("file-123").await.expect("download");

        assert_eq!(got, b"checkpoint bytes");
        let get = &server.requests()[1];
        assert_eq!(get.path(), "/drive/v3/files/file-123");
        assert!(get.target.contains("alt=media"), "unexpected target: {}", get.target);
        assert_eq!(get.header("authorization"), "Bearer ya29.access");
    }

    #[tokio::test]
    async fn a_missing_file_fails_the_download_with_drives_message() {
        let (_server, client) = fake_drive(|_, _| Reply::new(404, "File not found: file-123"));
        let error = client.download("file-123").await.unwrap_err();
        assert!(error.to_string().contains("drive download"), "unexpected error: {error}");
        assert!(error.to_string().contains("File not found"), "unexpected error: {error}");
    }

    #[tokio::test]
    async fn delete_removes_the_file_by_id() {
        let (server, client) = fake_drive(|_, _| Reply::ok(Vec::new()));
        client.delete("file-123").await.expect("delete");

        let delete = &server.requests()[1];
        assert_eq!(delete.method, "DELETE");
        assert_eq!(delete.path(), "/drive/v3/files/file-123");
    }

    #[tokio::test]
    async fn a_refused_delete_is_reported_with_its_body() {
        let (_server, client) = fake_drive(|_, _| Reply::new(403, "insufficientFilePermissions"));
        let error = client.delete("file-123").await.unwrap_err();
        assert!(error.to_string().contains("drive delete"), "unexpected error: {error}");
        assert!(error.to_string().contains("insufficientFilePermissions"), "unexpected error: {error}");
    }

    // -- from_env ------------------------------------------------------------

    /// Touches the process-global Google env vars, so it is one `#[test]`
    /// rather than several that could interleave (same convention as
    /// `artifacts::s3`'s `from_env` test).
    #[test]
    fn from_env_is_off_without_a_refresh_token_and_names_whichever_half_is_missing() {
        let restore: Vec<(&str, Option<String>)> = [
            env_vars::GOOGLE_REFRESH_TOKEN,
            env_vars::GOOGLE_CLIENT_ID,
            env_vars::GOOGLE_CLIENT_SECRET,
        ]
        .iter()
        .map(|var| (*var, std::env::var(var).ok()))
        .collect();
        for (var, _) in &restore {
            std::env::remove_var(var);
        }

        // `DriveClient` has no `Debug`, so each case matches rather than
        // unwrapping (which would need one just for the test).
        let configured = || match DriveClient::from_env() {
            Ok(client) => client,
            Err(error) => panic!("the archive tier being off is not an error: {error}"),
        };
        let refused = || match DriveClient::from_env() {
            Ok(_) => panic!("half-configured must be an error, not a silent no-op"),
            Err(error) => error,
        };

        assert!(
            configured().is_none(),
            "the archive tier is simply off without a refresh token"
        );
        // Blank counts as absent, not as a misconfiguration.
        std::env::set_var(env_vars::GOOGLE_REFRESH_TOKEN, "   ");
        assert!(configured().is_none());

        std::env::set_var(env_vars::GOOGLE_REFRESH_TOKEN, "1//refresh");
        let error = refused();
        assert!(error.to_string().contains(env_vars::GOOGLE_CLIENT_ID), "unexpected error: {error}");

        std::env::set_var(env_vars::GOOGLE_CLIENT_ID, "client-id");
        let error = refused();
        assert!(
            error.to_string().contains(env_vars::GOOGLE_CLIENT_SECRET),
            "unexpected error: {error}"
        );

        std::env::set_var(env_vars::GOOGLE_CLIENT_SECRET, "client-secret");
        let client = configured().expect("fully configured");
        assert_eq!(client.token_url, OAUTH_TOKEN_URL);
        assert_eq!(client.files_url, DRIVE_FILES_URL);
        assert_eq!(client.upload_url, DRIVE_UPLOAD_URL);

        for (var, value) in restore {
            match value {
                Some(value) => std::env::set_var(var, value),
                None => std::env::remove_var(var),
            }
        }
    }
}

/// Live round-trip against real Drive — see `drive/tests.rs`. Kept in a
/// `tests.rs` sibling because it is `#[ignore]`d and can never run in CI (it
/// needs a real Google grant): the coverage gate excludes `tests.rs` files, so
/// permanently-unrunnable lines don't count against this module's coverage.
#[cfg(test)]
#[path = "drive/tests.rs"]
mod live;
