//! Shared protocol for chuk-mcp-training: the single source of truth for
//! everything that crosses a process boundary (agent websocket, REST API),
//! plus the constants both sides must agree on.

pub mod constants;
pub mod domain;
pub mod keys;
pub mod manifest;
pub mod wire;

pub use constants::*;
pub use domain::*;
pub use manifest::*;
pub use wire::*;
