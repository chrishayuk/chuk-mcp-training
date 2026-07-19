//! API-key generation + hashing and the resolved request auth context (RBAC).
//!
//! Keys are `ck_<random>`; only the sha256 hash + a short display prefix are
//! ever stored — the plaintext is returned once, at creation, and never again.
//! A bearer token is resolved to an [`AuthContext`] (role + team) by hashing it
//! and looking the hash up in `api_keys` (or matching the legacy master token).

use chuk_train_proto::{Role, API_KEY_PREFIX};
use sha2::{Digest, Sha256};

/// Chars of the plaintext kept as the human-readable display prefix
/// (`ck_` + 8), enough to recognise a key but not to use it.
const DISPLAY_PREFIX_LEN: usize = 3 + 8;

/// Generate a new key: `(plaintext, display_prefix, sha256_hex_hash)`.
pub fn generate() -> (String, String, String) {
    let random = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let plaintext = format!("{API_KEY_PREFIX}{random}");
    let prefix = plaintext[..DISPLAY_PREFIX_LEN].to_owned();
    let hash = hash_token(&plaintext);
    (plaintext, prefix, hash)
}

/// The sha256 hex of a bearer token — what's stored and looked up.
pub fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

/// Who is making a request, resolved from the bearer token or Google session.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub role: Role,
    pub team_id: String,
    /// The email (Google session / user) or key prefix (api key) — for
    /// attribution + logging.
    pub subject: String,
}

impl AuthContext {
    /// True if this context meets a minimum role.
    pub fn may(&self, min: Role) -> bool {
        self.role >= min
    }
}
