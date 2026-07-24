//! A loopback HTTP/1.1 server for tests, so the modules that talk to a remote
//! service over HTTP (Drive, the experiments mirror, S3/R2, Google sign-in) are
//! exercised against a real socket instead of being left untested behind a
//! network call. Same shape as `chuk-compute-worker`'s `httpclient` test
//! server, lifted here because four modules in this crate now need it.
//!
//! Each connection serves exactly one request and closes (`Connection: close`),
//! which every client we drive (reqwest, aws-sdk-s3's hyper client) handles by
//! opening a fresh connection for the next call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A refused loopback address: connecting to it fails fast and deterministically
/// without a network, for the transport-error paths.
pub(crate) const REFUSED_ORIGIN: &str = "http://127.0.0.1:1";

/// One HTTP/1.1 request as seen by a [`FakeHttp`] handler.
#[derive(Clone, Debug)]
pub(crate) struct Received {
    pub method: String,
    /// Request target including any query string, e.g. `/v1/runs/RUN-1?x=1`.
    pub target: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Received {
    /// The path with any query string stripped.
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or_default()
    }

    /// A request header by (lowercase) name, or `""` when absent.
    pub fn header(&self, name: &str) -> &str {
        self.headers.get(name).map_or("", String::as_str)
    }

    /// The body parsed as JSON; panics if it isn't.
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("request body is json")
    }
}

/// What a [`FakeHttp`] handler wants written back.
pub(crate) struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Reply {
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Reply { status, headers: Vec::new(), body: body.into() }
    }

    /// A 200 with a body — the common case.
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        Reply::new(200, body)
    }

    pub fn header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_owned(), value.into()));
        self
    }
}

/// A running fake server: its `origin` (`http://127.0.0.1:<port>`) and the log
/// of every request it has served.
pub(crate) struct FakeHttp {
    pub origin: String,
    seen: Arc<Mutex<Vec<Received>>>,
}

impl FakeHttp {
    /// Bind an ephemeral port and serve every request with `handler`. The
    /// handler is given the request and the call count so far (0 for the first
    /// request), so a test can script a failure-then-success sequence.
    pub fn start<F>(handler: F) -> Self
    where
        F: Fn(&Received, usize) -> Reply + Send + Sync + 'static,
    {
        Self::start_with_origin(move |_, req, nth| handler(req, nth))
    }

    /// [`Self::start`] for a handler that must know its own origin — a service
    /// that hands back URLs pointing at itself (Drive's resumable session URI).
    pub fn start_with_origin<F>(handler: F) -> Self
    where
        F: Fn(&str, &Received, usize) -> Reply + Send + Sync + 'static,
    {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        listener.set_nonblocking(true).expect("nonblocking");
        let origin = format!("http://{}", listener.local_addr().expect("local addr"));
        let seen: Arc<Mutex<Vec<Received>>> = Arc::new(Mutex::new(Vec::new()));

        let listener = TcpListener::from_std(listener).expect("tokio listener");
        let handler = Arc::new(handler);
        let log = seen.clone();
        let own_origin = origin.clone();
        // Fire-and-forget: the task ends when the test process does, matching
        // the worker crate's `serve` convention (there is no explicit shutdown).
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else { break };
                let handler = handler.clone();
                let log = log.clone();
                let own_origin = own_origin.clone();
                tokio::spawn(async move {
                    let received = read_request(&mut socket).await;
                    let nth = {
                        let mut log = log.lock().expect("request log");
                        log.push(received.clone());
                        log.len() - 1
                    };
                    let reply = handler(&own_origin, &received, nth);
                    let mut head = format!("HTTP/1.1 {} X\r\n", reply.status);
                    for (name, value) in &reply.headers {
                        head.push_str(&format!("{name}: {value}\r\n"));
                    }
                    head.push_str(&format!(
                        "Content-Length: {}\r\nConnection: close\r\n\r\n",
                        reply.body.len()
                    ));
                    let _ = socket.write_all(head.as_bytes()).await;
                    let _ = socket.write_all(&reply.body).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        FakeHttp { origin, seen }
    }

    /// Every request served so far, oldest first.
    pub fn requests(&self) -> Vec<Received> {
        self.seen.lock().expect("request log").clone()
    }

    /// How many requests have been served.
    pub fn hits(&self) -> usize {
        self.requests().len()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Read one HTTP/1.1 request off `socket`: the request line, headers, and any
/// `Content-Length` body.
async fn read_request(socket: &mut tokio::net::TcpStream) -> Received {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = socket.read(&mut chunk).await.expect("read request");
        assert!(n > 0, "connection closed before headers completed");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().expect("request line").split(' ');
    let method = request_line.next().expect("method").to_owned();
    let target = request_line.next().expect("target").to_owned();
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_lowercase(), v.trim().to_owned());
        }
    }
    let mut body = buf[header_end + 4..].to_vec();
    if let Some(len) = headers.get("content-length").and_then(|v| v.parse::<usize>().ok()) {
        while body.len() < len {
            let n = socket.read(&mut chunk).await.expect("read body");
            assert!(n > 0, "connection closed before body completed");
            body.extend_from_slice(&chunk[..n]);
        }
    }
    Received { method, target, headers, body }
}
