//! API-key generation + hashing and the resolved request auth context (RBAC).
//!
//! Keys are `ck_<random>`; only the sha256 hash + a short display prefix are
//! ever stored — the plaintext is returned once, at creation, and never again.
//! A bearer token is resolved to an [`AuthContext`] (role + team) by hashing it
//! and looking the hash up in `api_keys` (or matching the legacy master token).

use chuk_train_proto::{
    Role, WorkerId, API_KEY_PREFIX, JOIN_TOKEN_PREFIX, JOIN_TOKEN_TTL, WORKER_TOKEN_PREFIX,
};
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

/// Generate a new single-use provision join token (spec §12):
/// `(plaintext, display_prefix, sha256_hex_hash)`.
pub fn generate_join_token() -> (String, String, String) {
    generate_with_prefix(JOIN_TOKEN_PREFIX)
}

/// Mint + persist a join token bound to `worker_id`, returning the plaintext
/// (handed to the provider bootstrap / Colab cell, never stored). The stored
/// hash expires for *first* use after [`JOIN_TOKEN_TTL`]; once consumed it
/// only ever readmits its own bound worker id.
pub async fn mint_join_token(
    store: &dyn crate::store::Store,
    worker_id: &WorkerId,
) -> anyhow::Result<String> {
    let (plaintext, _display_prefix, hash) = generate_join_token();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64();
    store
        .create_join_token(
            &uuid::Uuid::new_v4().simple().to_string(),
            worker_id,
            &hash,
            now + JOIN_TOKEN_TTL.as_secs_f64(),
        )
        .await?;
    Ok(plaintext)
}

/// `AuthContext.owner_email`'s value for the shared legacy master token — it
/// authenticates no particular user, so per-user features (e.g. linking a
/// personal chuk-experiments-server key) must reject this sentinel rather
/// than silently attaching state to it.
pub const MASTER_TOKEN_SENTINEL: &str = "master-token";

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
    /// Always a real email (or the master-token sentinel) regardless of auth
    /// method — for a scoped API key this is the key's owning user
    /// (`ApiKeyInfo.created_by`), not the key's prefix. Use this, not
    /// `subject`, to resolve "which user made this call" (e.g. per-user
    /// settings lookups); `subject` stays as-is for existing attribution/
    /// logging call sites.
    pub owner_email: String,
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
