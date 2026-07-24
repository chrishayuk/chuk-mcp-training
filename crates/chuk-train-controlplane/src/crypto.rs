//! Reversible encryption for the one secret this store needs back in
//! plaintext: a user's own linked chuk-experiments-server API key. Everything
//! else in this schema (`api_keys`, `worker_tokens`) is one-way hashed, since
//! we only ever need to *verify* those, never resend them — this is new
//! territory, so we mirror chuk-experiments-server's own solution to the same
//! problem (`token_crypto.py`, Fernet) one-for-one, just with AES-256-GCM
//! instead (nothing reversible was already a dependency in this workspace).

use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, Key, KeyInit, Nonce};
use anyhow::{Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chuk_train_proto::env;

/// Read + decode `CHUK_EXPERIMENTS_KEY_ENCRYPTION_KEY` (32 raw bytes, base64).
/// `None` unless it's set and decodes to exactly 32 bytes — callers treat
/// `None` as "the per-user-key feature is off", never as an error.
pub fn key_from_env() -> Option<[u8; 32]> {
    let raw = std::env::var(env::EXPERIMENTS_KEY_ENCRYPTION_KEY).ok()?;
    let bytes = STANDARD.decode(raw.trim()).ok()?;
    bytes.try_into().ok()
}

/// Encrypt `plaintext` under `key`. Self-contained: a random 96-bit nonce is
/// prepended to the ciphertext and the whole thing base64-encoded as one
/// string, so nothing else needs to be stored alongside it.
pub fn encrypt(key: &[u8; 32], plaintext: &str) -> String {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_bytes())
        .expect("AES-GCM encryption of a bounded plaintext cannot fail");
    let mut out = nonce.to_vec();
    out.extend_from_slice(&ciphertext);
    STANDARD.encode(out)
}

/// Decrypt a blob produced by [`encrypt`]. Fails on a wrong key or a corrupt/
/// truncated ciphertext — the caller (`Experiments::bearer_for`) treats any
/// error as "no usable personal key" and falls back to the shared default,
/// same as a missing key.
pub fn decrypt(key: &[u8; 32], blob_b64: &str) -> Result<String> {
    let raw = STANDARD.decode(blob_b64).context("base64 decode")?;
    if raw.len() < 12 {
        anyhow::bail!("ciphertext too short to contain a nonce");
    }
    let (nonce, ciphertext) = raw.split_at(12);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let plaintext = cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| anyhow::anyhow!("decryption failed (wrong key or corrupt ciphertext)"))?;
    String::from_utf8(plaintext).context("decrypted bytes were not valid utf-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; 32] {
        [7u8; 32]
    }

    #[test]
    fn round_trips() {
        let key = test_key();
        let ciphertext = encrypt(&key, "ck_realKeyValueHere");
        assert_eq!(decrypt(&key, &ciphertext).unwrap(), "ck_realKeyValueHere");
    }

    #[test]
    fn round_trips_empty_plaintext() {
        // Not a realistic key value, but the cipher/format shouldn't care.
        let key = test_key();
        let ciphertext = encrypt(&key, "");
        assert_eq!(decrypt(&key, &ciphertext).unwrap(), "");
    }

    #[test]
    fn wrong_key_fails() {
        let ciphertext = encrypt(&test_key(), "secret");
        let wrong = [9u8; 32];
        assert!(decrypt(&wrong, &ciphertext).is_err());
    }

    #[test]
    fn corrupt_ciphertext_fails() {
        assert!(decrypt(&test_key(), "not-valid-base64!!!").is_err());
    }

    #[test]
    fn truncated_ciphertext_too_short_for_a_nonce_fails() {
        // Valid base64, but decodes to fewer than the 12 nonce bytes `decrypt`
        // requires before it even tries to split out a ciphertext.
        let blob = STANDARD.encode([1u8, 2, 3]);
        let err = decrypt(&test_key(), &blob).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn tampered_ciphertext_fails_the_aead_tag_check() {
        let key = test_key();
        let ciphertext = encrypt(&key, "secret");
        let mut raw = STANDARD.decode(ciphertext).unwrap();
        // Flip a byte in the ciphertext body (past the 12-byte nonce prefix)
        // so the AEAD authentication tag no longer matches.
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        let tampered = STANDARD.encode(raw);
        assert!(decrypt(&key, &tampered).is_err());
    }

    /// Exercises `key_from_env` end to end (found + decodes to 32 bytes,
    /// missing, and present-but-malformed) via the real env var it reads.
    /// Runs the three cases serially inside one test (rather than three
    /// `#[test]`s) so no other test can observe a torn `set_var`/`remove_var`
    /// window on this process-global variable — this crate has no
    /// `serial_test` dependency to isolate them the usual way.
    #[test]
    fn key_from_env_reads_decodes_and_rejects_bad_config() {
        let var = env::EXPERIMENTS_KEY_ENCRYPTION_KEY;
        let saved = std::env::var(var).ok();

        std::env::remove_var(var);
        assert_eq!(key_from_env(), None, "unset: feature is off, not an error");

        let key = [4u8; 32];
        std::env::set_var(var, STANDARD.encode(key));
        assert_eq!(key_from_env(), Some(key));

        std::env::set_var(var, "not-valid-base64!!!");
        assert_eq!(key_from_env(), None, "undecodable base64 is also just \"off\"");

        std::env::set_var(var, STANDARD.encode([4u8; 16]));
        assert_eq!(key_from_env(), None, "decodes fine but isn't 32 bytes");

        match saved {
            Some(v) => std::env::set_var(var, v),
            None => std::env::remove_var(var),
        }
    }
}
