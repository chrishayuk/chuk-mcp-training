//! The agent websocket endpoint (spec §7): one outbound connection per
//! worker, `register` first, then a bidirectional message loop.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use chuk_train_proto::{AgentToCp, CpToAgent, WorkerId, REGISTER_TIMEOUT};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::AppState;

const WORKER_ID_PREFIX: &str = "w-";
const REJECT_BAD_TOKEN: &str = "bad join token";
const REJECT_NOT_REGISTER: &str = "first message must be register";

pub async fn agent_ws(State(state): State<Arc<AppState>>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| session(state, socket))
}

async fn session(state: Arc<AppState>, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();

    // Phase 1: registration, bounded by REGISTER_TIMEOUT.
    let registration =
        tokio::time::timeout(REGISTER_TIMEOUT, next_agent_message(&mut stream)).await;
    let (worker_id, labels, hardware) = match registration {
        Ok(Some(AgentToCp::Register {
            token,
            worker_id,
            labels,
            hardware,
        })) => {
            if token != state.config.join_token {
                warn!("agent presented a bad join token");
                let _ = send(
                    &mut sink,
                    &CpToAgent::Rejected {
                        reason: REJECT_BAD_TOKEN.into(),
                    },
                )
                .await;
                return;
            }
            let id =
                worker_id.unwrap_or_else(|| WorkerId(format!("{WORKER_ID_PREFIX}{}", short_id())));
            (id, labels, hardware)
        }
        Ok(Some(_)) => {
            let _ = send(
                &mut sink,
                &CpToAgent::Rejected {
                    reason: REJECT_NOT_REGISTER.into(),
                },
            )
            .await;
            return;
        }
        Ok(None) | Err(_) => {
            debug!("socket closed or timed out before registration");
            return;
        }
    };

    // Phase 2: attach to the hub; the hub writes to `tx`, we pump `rx` to the sink.
    let (tx, mut rx) = mpsc::unbounded_channel::<CpToAgent>();
    if send(
        &mut sink,
        &CpToAgent::Registered {
            worker_id: worker_id.clone(),
        },
    )
    .await
    .is_err()
    {
        return;
    }
    if let Err(error) = state.hub.attach(&worker_id, tx, &labels, &hardware).await {
        warn!(worker = %worker_id, %error, "attach failed");
        return;
    }

    loop {
        tokio::select! {
            outbound = rx.recv() => match outbound {
                Some(msg) => {
                    if send(&mut sink, &msg).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
            inbound = next_agent_message(&mut stream) => match inbound {
                Some(msg) => {
                    if let Err(error) = state.hub.on_message(&worker_id, msg).await {
                        warn!(worker = %worker_id, %error, "error handling agent message");
                    }
                }
                None => break,
            },
        }
    }

    info!(worker = %worker_id, "session ended");
    if let Err(error) = state.hub.detach(&worker_id).await {
        warn!(worker = %worker_id, %error, "detach failed");
    }
}

/// Read the next parseable agent message; `None` means the socket is gone.
/// Unparseable frames are logged and skipped rather than killing the session.
async fn next_agent_message(
    stream: &mut (impl StreamExt<Item = Result<Message, axum::Error>> + Unpin),
) -> Option<AgentToCp> {
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Text(text)) => match serde_json::from_str::<AgentToCp>(&text) {
                Ok(msg) => return Some(msg),
                Err(error) => warn!(%error, "unparseable agent message; skipping"),
            },
            Ok(Message::Close(_)) => return None,
            Ok(_) => {} // ping/pong/binary: nothing to do
            Err(_) => return None,
        }
    }
    None
}

async fn send(
    sink: &mut (impl SinkExt<Message, Error = axum::Error> + Unpin),
    msg: &CpToAgent,
) -> Result<(), axum::Error> {
    let payload = serde_json::to_string(msg).expect("CpToAgent always serialises");
    sink.send(Message::Text(payload.into())).await
}

fn short_id() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_owned()
}
