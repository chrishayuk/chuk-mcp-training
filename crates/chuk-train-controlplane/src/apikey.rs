//! API-key generation + hashing and the resolved request auth context (RBAC).
//!
//! Keys are `ck_<random>`; only the sha256 hash + a short display prefix are
//! ever stored — the plaintext is returned once, at creation, and never again.
//! A bearer token is resolved to an [`AuthContext`] (role + team) by hashing it
//! and looking the hash up in `api_keys` (or matching the legacy master token).

use chuk_train_proto::{Role, API_KEY_PREFIX, WORKER_TOKEN_PREFIX};
use sha2::{Digest, Sha256};

/// Generate a token of `<prefix><random>`, returning
/// `(plaintext, display_prefix, sha256_hex_hash)`. The display prefix keeps the
/// scheme prefix plus 8 chars of the random tail — enough to recognise a token
/// but not to use it.
fn generate_with_prefix(prefix: &str) -> (String, String, String) {
    let random = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let plaintext = format!("{prefix}{random}");
    let display_prefix = plaintext[..prefix.len() + 8].to_owned();
    let hash = hash_token(&plaintext);
    (plaintext, display_prefix, hash)
}

/// Generate a new MCP API key: `(plaintext, display_prefix, sha256_hex_hash)`.
pub fn generate() -> (String, String, String) {
    generate_with_prefix(API_KEY_PREFIX)
}

/// Generate a new persistent worker token (chuk-compute M3.1):
/// `(plaintext, display_prefix, sha256_hex_hash)`.
pub fn generate_worker_token() -> (String, String, String) {
    generate_with_prefix(WORKER_TOKEN_PREFIX)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_carries_its_prefix_and_round_trips() {
        let (plaintext, prefix, hash) = generate();
        assert!(plaintext.starts_with(API_KEY_PREFIX));
        assert_eq!(prefix, plaintext[..API_KEY_PREFIX.len() + 8]);
        assert_eq!(hash, hash_token(&plaintext));
    }

    #[test]
    fn worker_token_carries_its_prefix_and_round_trips() {
        let (plaintext, prefix, hash) = generate_worker_token();
        assert!(plaintext.starts_with(WORKER_TOKEN_PREFIX));
        assert!(prefix.starts_with(WORKER_TOKEN_PREFIX));
        assert_eq!(prefix.len(), WORKER_TOKEN_PREFIX.len() + 8);
        // The stored hash is the sha256 of the plaintext shown once at creation.
        assert_eq!(hash, hash_token(&plaintext));
    }
}
