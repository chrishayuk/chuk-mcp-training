//! Grant-authorised blob transfer against the control plane's REST API.
//! The agent holds only a run-scoped grant token — never the API token.
//!
//! Every transfer is presign-first: the agent asks the control plane where to
//! send/get a blob (`/api/blob_url`). With an S3/R2 backend that is a presigned
//! URL the agent hits directly, so ~500 MB checkpoints never transit the
//! control plane; with the filesystem backend it points back at the control
//! plane and the agent attaches its grant token.

use anyhow::{Context, Result};
use chuk_compute_wire::{BlobMethod, BlobUrlRequest, BlobUrlResponse, API_PREFIX};

const BEARER_PREFIX: &str = "Bearer ";

#[derive(Clone)]
pub struct HttpClient {
    /// Control-plane origin, e.g. `http://127.0.0.1:8700`.
    origin: String,
    token: String,
    http: reqwest::Client,
}

impl HttpClient {
    pub fn new(origin: String, token: String) -> Self {
        Self {
            origin,
            token,
            http: reqwest::Client::new(),
        }
    }

    fn bearer(&self) -> String {
        format!("{BEARER_PREFIX}{}", self.token)
    }

    /// Ask the control plane where to transfer `key` in the given direction.
    async fn blob_url(&self, method: BlobMethod, key: &str) -> Result<BlobUrlResponse> {
        let plan = self
            .http
            .post(format!("{}{API_PREFIX}/blob_url", self.origin))
            .header(reqwest::header::AUTHORIZATION, self.bearer())
            .json(&BlobUrlRequest {
                key: key.to_owned(),
                method,
            })
            .send()
            .await
            .with_context(|| format!("requesting blob url for {key}"))?
            .error_for_status()
            .with_context(|| format!("requesting blob url for {key}"))?
            .json::<BlobUrlResponse>()
            .await
            .with_context(|| format!("parsing blob url for {key}"))?;
        Ok(plan)
    }

    /// Download a blob the grant may read (code unit, resume checkpoint).
    pub async fn fetch(&self, key: &str) -> Result<Vec<u8>> {
        let plan = self.blob_url(BlobMethod::Get, key).await?;
        let mut req = self.http.get(&plan.url);
        if plan.requires_grant_header {
            req = req.header(reqwest::header::AUTHORIZATION, self.bearer());
        }
        let response = req
            .send()
            .await
            .with_context(|| format!("fetching {key}"))?
            .error_for_status()
            .with_context(|| format!("fetching {key}"))?;
        Ok(response.bytes().await?.to_vec())
    }

    /// Upload a blob into the grant's run tree (a checkpoint file).
    pub async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        let plan = self.blob_url(BlobMethod::Put, key).await?;
        // Explicit Content-Length: for an EMPTY body hyper omits the header,
        // and S3/R2 answer a presigned PUT without it with 411 Length
        // Required — first seen live on a checkpoint's zero-byte `.ready`
        // marker. Harmless for non-empty bodies (same value hyper would set).
        let mut req = self
            .http
            .put(&plan.url)
            .header(reqwest::header::CONTENT_LENGTH, bytes.len())
            .body(bytes);
        if plan.requires_grant_header {
            req = req.header(reqwest::header::AUTHORIZATION, self.bearer());
        }
        req.send()
            .await
            .with_context(|| format!("uploading {key}"))?
            .error_for_status()
            .with_context(|| format!("uploading {key}"))?;
        Ok(())
    }
}

/// Derive the HTTP origin from the agent's websocket URL:
/// `ws://h:p/ws/agent` → `http://h:p`, `wss://h/ws/agent` → `https://h`.
pub fn origin_from_ws_url(ws_url: &str) -> Result<String> {
    let (scheme, rest) = ws_url
        .split_once("://")
        .context("agent url must include a scheme, e.g. ws://host/ws/agent")?;
    let http_scheme = match scheme {
        "ws" => "http",
        "wss" => "https",
        other => anyhow::bail!("unexpected agent url scheme: {other}"),
    };
    let authority = rest.split('/').next().unwrap_or(rest);
    Ok(format!("{http_scheme}://{authority}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // A refused loopback address, so error paths fail fast and
    // deterministically without a network (mirrors inputs.rs's convention).
    const REFUSED_ORIGIN: &str = "http://127.0.0.1:1";

    /// One HTTP/1.1 request as seen by a [`spawn_server`] handler.
    #[derive(Clone, Debug)]
    struct Received {
        method: String,
        path: String,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    }

    /// What a [`spawn_server`] handler wants written back.
    struct Reply {
        status: u16,
        reason: &'static str,
        body: Vec<u8>,
    }

    impl Reply {
        fn text(status: u16, reason: &'static str, body: &str) -> Self {
            Reply { status, reason, body: body.as_bytes().to_vec() }
        }
    }

    fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    /// Read one HTTP/1.1 request off `socket`: the request line, headers, and
    /// any `Content-Length` body.
    async fn read_request(socket: &mut tokio::net::TcpStream) -> Received {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 1024];
        let header_end = loop {
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0, "connection closed before headers completed");
            buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                break pos;
            }
        };
        let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        let mut lines = head.split("\r\n");
        let request_line = lines.next().unwrap();
        let mut parts = request_line.split(' ');
        let method = parts.next().unwrap().to_owned();
        let path = parts.next().unwrap().to_owned();
        let mut headers = HashMap::new();
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                headers.insert(k.trim().to_lowercase(), v.trim().to_owned());
            }
        }
        let mut body = buf[header_end + 4..].to_vec();
        if let Some(len) = headers.get("content-length").and_then(|v| v.parse::<usize>().ok()) {
            while body.len() < len {
                let n = socket.read(&mut chunk).await.unwrap();
                assert!(n > 0, "connection closed before body completed");
                body.extend_from_slice(&chunk[..n]);
            }
        }
        Received { method, path, headers, body }
    }

    /// Bind an ephemeral local port, returning the raw listener plus its
    /// `http://…` origin (known before any handler is attached, so a handler
    /// can build URLs that point back at its own server).
    fn bind_server() -> (std::net::TcpListener, String) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        (listener, format!("http://{addr}"))
    }

    /// Serve every connection accepted on `listener` by handing its request to
    /// `handler` and writing back the `Reply`. Runs for the rest of the test
    /// process (there is no explicit shutdown, matching inputs.rs's
    /// `serve_once` convention of a fire-and-forget background task).
    fn serve<F>(listener: std::net::TcpListener, handler: F)
    where
        F: Fn(Received) -> Reply + Send + Sync + 'static,
    {
        let listener = TcpListener::from_std(listener).unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { break };
                let handler = handler.clone();
                tokio::spawn(async move {
                    let received = read_request(&mut socket).await;
                    let reply = handler(received);
                    let header = format!(
                        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.status,
                        reply.reason,
                        reply.body.len()
                    );
                    let _ = socket.write_all(header.as_bytes()).await;
                    let _ = socket.write_all(&reply.body).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
    }

    /// [`bind_server`] + [`serve`] in one call, for a handler that does not
    /// need to know its own origin.
    fn spawn_server<F>(handler: F) -> String
    where
        F: Fn(Received) -> Reply + Send + Sync + 'static,
    {
        let (listener, origin) = bind_server();
        serve(listener, handler);
        origin
    }

    #[test]
    fn bearer_prefixes_the_token() {
        let client = HttpClient::new("http://ignored".into(), "tok123".into());
        assert_eq!(client.bearer(), "Bearer tok123");
    }

    #[tokio::test]
    async fn blob_url_posts_the_request_and_parses_a_signed_response() {
        let seen: Arc<Mutex<Option<Received>>> = Arc::new(Mutex::new(None));
        let seen2 = seen.clone();
        let origin = spawn_server(move |req| {
            *seen2.lock().unwrap() = Some(req);
            Reply::text(200, "OK", r#"{"url":"https://r2.example/put?sig=abc","requires_grant_header":false}"#)
        });

        let client = HttpClient::new(origin, "secret-tok".into());
        let plan = client.blob_url(BlobMethod::Put, "runs/j1/ckpt/model.bin").await.unwrap();
        assert_eq!(plan.url, "https://r2.example/put?sig=abc");
        assert!(!plan.requires_grant_header);

        let req = seen.lock().unwrap().take().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/api/blob_url");
        assert_eq!(req.headers.get("authorization").unwrap(), "Bearer secret-tok");
        let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(body["key"], "runs/j1/ckpt/model.bin");
        assert_eq!(body["method"], "put");
    }

    #[tokio::test]
    async fn blob_url_errors_with_context_on_a_non_2xx_status() {
        let origin = spawn_server(|_req| Reply::text(500, "Internal Server Error", "boom"));
        let client = HttpClient::new(origin, "tok".into());
        let error = client.blob_url(BlobMethod::Get, "k").await.unwrap_err();
        assert!(format!("{error:#}").contains("requesting blob url for k"));
    }

    #[tokio::test]
    async fn blob_url_errors_with_context_on_malformed_json() {
        let origin = spawn_server(|_req| Reply::text(200, "OK", "not json"));
        let client = HttpClient::new(origin, "tok".into());
        let error = client.blob_url(BlobMethod::Get, "k").await.unwrap_err();
        assert!(format!("{error:#}").contains("parsing blob url for k"));
    }

    #[tokio::test]
    async fn fetch_downloads_bytes_and_sends_the_grant_header_only_when_required() {
        for requires_grant in [false, true] {
            let log: Arc<Mutex<Vec<Received>>> = Arc::new(Mutex::new(Vec::new()));
            let log2 = log.clone();
            let (listener, origin) = bind_server();
            let origin2 = origin.clone();
            serve(listener, move |req| {
                log2.lock().unwrap().push(req.clone());
                if req.path == "/api/blob_url" {
                    Reply::text(
                        200,
                        "OK",
                        &format!(r#"{{"url":"{origin2}/blob/shard","requires_grant_header":{requires_grant}}}"#),
                    )
                } else {
                    Reply::text(200, "OK", "shard-bytes")
                }
            });

            let client = HttpClient::new(origin, "secret-tok".into());
            let bytes = client.fetch("runs/j1/shard").await.unwrap();
            assert_eq!(bytes, b"shard-bytes");

            let seen = log.lock().unwrap();
            assert_eq!(seen.len(), 2, "blob_url, then the direct GET");
            assert_eq!(seen[0].path, "/api/blob_url");
            assert_eq!(seen[1].method, "GET");
            assert_eq!(seen[1].path, "/blob/shard");
            assert_eq!(
                seen[1].headers.contains_key("authorization"),
                requires_grant,
                "second-hop grant header presence should match requires_grant_header={requires_grant}"
            );
        }
    }

    #[tokio::test]
    async fn fetch_propagates_the_blob_url_step_failure() {
        let client = HttpClient::new(REFUSED_ORIGIN.into(), "tok".into());
        let error = client.fetch("k").await.unwrap_err();
        assert!(format!("{error:#}").contains("requesting blob url for k"));
    }

    #[tokio::test]
    async fn fetch_errors_with_context_when_the_second_hop_get_fails() {
        let (listener, origin) = bind_server();
        let origin2 = origin.clone();
        serve(listener, move |req| {
            if req.path == "/api/blob_url" {
                Reply::text(200, "OK", &format!(r#"{{"url":"{origin2}/missing","requires_grant_header":false}}"#))
            } else {
                Reply::text(404, "Not Found", "nope")
            }
        });
        let client = HttpClient::new(origin, "tok".into());
        let error = client.fetch("k").await.unwrap_err();
        assert!(format!("{error:#}").contains("fetching k"));
    }

    #[tokio::test]
    async fn upload_puts_the_body_with_content_length_and_the_grant_header_when_required() {
        for requires_grant in [false, true] {
            let log: Arc<Mutex<Vec<Received>>> = Arc::new(Mutex::new(Vec::new()));
            let log2 = log.clone();
            let (listener, origin) = bind_server();
            let origin2 = origin.clone();
            serve(listener, move |req| {
                log2.lock().unwrap().push(req.clone());
                if req.path == "/api/blob_url" {
                    Reply::text(
                        200,
                        "OK",
                        &format!(r#"{{"url":"{origin2}/blob/ckpt","requires_grant_header":{requires_grant}}}"#),
                    )
                } else {
                    Reply::text(200, "OK", "")
                }
            });

            let client = HttpClient::new(origin, "secret-tok".into());
            client.upload("runs/j1/ckpt", b"payload-bytes".to_vec()).await.unwrap();

            let seen = log.lock().unwrap();
            assert_eq!(seen.len(), 2, "blob_url, then the direct PUT");
            assert_eq!(seen[1].method, "PUT");
            assert_eq!(seen[1].path, "/blob/ckpt");
            assert_eq!(seen[1].body, b"payload-bytes");
            assert_eq!(seen[1].headers.get("content-length").unwrap(), "13");
            assert_eq!(
                seen[1].headers.contains_key("authorization"),
                requires_grant,
                "second-hop grant header presence should match requires_grant_header={requires_grant}"
            );
        }
    }

    #[tokio::test]
    async fn upload_propagates_the_blob_url_step_failure() {
        let client = HttpClient::new(REFUSED_ORIGIN.into(), "tok".into());
        let error = client.upload("k", b"x".to_vec()).await.unwrap_err();
        assert!(format!("{error:#}").contains("requesting blob url for k"));
    }

    #[tokio::test]
    async fn upload_errors_with_context_when_the_second_hop_put_fails() {
        let (listener, origin) = bind_server();
        let origin2 = origin.clone();
        serve(listener, move |req| {
            if req.path == "/api/blob_url" {
                Reply::text(200, "OK", &format!(r#"{{"url":"{origin2}/blob/ckpt","requires_grant_header":false}}"#))
            } else {
                Reply::text(500, "Internal Server Error", "boom")
            }
        });
        let client = HttpClient::new(origin, "tok".into());
        let error = client.upload("k", b"x".to_vec()).await.unwrap_err();
        assert!(format!("{error:#}").contains("uploading k"));
    }

    #[test]
    fn origin_from_ws_url_maps_scheme_and_strips_the_path() {
        assert_eq!(origin_from_ws_url("ws://host:9000/ws/agent").unwrap(), "http://host:9000");
        assert_eq!(origin_from_ws_url("wss://host/ws/agent").unwrap(), "https://host");
        assert_eq!(origin_from_ws_url("wss://host").unwrap(), "https://host");
    }

    #[test]
    fn origin_from_ws_url_rejects_a_missing_scheme() {
        let error = origin_from_ws_url("host:9000/ws/agent").unwrap_err();
        assert!(format!("{error}").contains("scheme"));
    }

    #[test]
    fn origin_from_ws_url_rejects_an_unexpected_scheme() {
        let error = origin_from_ws_url("http://host/ws/agent").unwrap_err();
        assert!(format!("{error}").contains("unexpected agent url scheme: http"));
    }
}
