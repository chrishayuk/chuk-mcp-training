//! S3-compatible artifact backend (spec §11.5). Works against AWS S3 and,
//! preferred, Cloudflare R2 (zero egress) via an endpoint override + path-style
//! addressing.
//!
//! The big win is presigned URLs: `presign_put`/`presign_get` hand the worker a
//! time-limited URL it uses to transfer ~500 MB checkpoints **directly** to/from
//! the bucket (spec §12), so the bytes never touch the control plane.
//!
//! Not yet exercised against a live bucket — the first real checkpoint upload
//! validates it (like the Vast driver). Credentials + endpoint come from the
//! environment and live only on the control plane.

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_sdk_s3::config::{BehaviorVersion, Credentials, Region};
use aws_sdk_s3::presigning::PresigningConfig;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use chuk_train_proto::{env as env_vars, SignedUrl};

use super::ArtifactStore;

const DEFAULT_REGION: &str = "auto"; // R2 signs against "auto"
const CREDENTIAL_SOURCE: &str = "chuk-train";

#[derive(Debug)]
pub struct S3ArtifactStore {
    client: Client,
    bucket: String,
}

impl S3ArtifactStore {
    /// Build from the environment: endpoint, region, and access key / secret.
    /// R2 endpoint looks like `https://<account>.r2.cloudflarestorage.com`.
    pub fn from_env(bucket: &str) -> Result<Self> {
        let endpoint = std::env::var(env_vars::S3_ENDPOINT)
            .with_context(|| format!("{} must be set for an s3/r2 store", env_vars::S3_ENDPOINT))?;
        let region =
            std::env::var(env_vars::S3_REGION).unwrap_or_else(|_| DEFAULT_REGION.to_owned());
        let access_key = std::env::var(env_vars::S3_ACCESS_KEY_ID)
            .with_context(|| format!("{} must be set", env_vars::S3_ACCESS_KEY_ID))?;
        let secret_key = std::env::var(env_vars::S3_SECRET_ACCESS_KEY)
            .with_context(|| format!("{} must be set", env_vars::S3_SECRET_ACCESS_KEY))?;

        let creds = Credentials::new(access_key, secret_key, None, None, CREDENTIAL_SOURCE);
        let config = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(region))
            .endpoint_url(endpoint)
            .credentials_provider(creds)
            .force_path_style(true)
            // R2 rejects the default trailing-checksum trailers on presigned
            // PUTs; only sign a checksum when the caller actually requires one.
            .request_checksum_calculation(
                aws_sdk_s3::config::RequestChecksumCalculation::WhenRequired,
            )
            .build();
        Ok(Self {
            client: Client::from_conf(config),
            bucket: bucket.to_owned(),
        })
    }

    async fn presign(&self, ttl: Duration, get: bool, key: &str) -> Result<SignedUrl> {
        let cfg = PresigningConfig::expires_in(ttl)?;
        let uri = if get {
            self.client
                .get_object()
                .bucket(&self.bucket)
                .key(key)
                .presigned(cfg)
                .await?
        } else {
            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(key)
                .presigned(cfg)
                .await?
        };
        Ok(SignedUrl {
            url: uri.uri().to_string(),
            expires_at: now() + ttl.as_secs_f64(),
        })
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

/// Merge our lifecycle `rules` (`(key_prefix, days_to_expire)`) into whatever
/// rules already exist on the bucket: keep every foreign rule whose prefix
/// isn't ours (or that carries no prefix filter at all, e.g. R2's default
/// multipart-abort rule), then append ours. Pulled out of `apply_lifecycle` so
/// the merge policy is unit-testable without a live bucket.
fn merge_lifecycle_rules(
    existing: Vec<aws_sdk_s3::types::LifecycleRule>,
    rules: &[(String, i32)],
) -> Result<Vec<aws_sdk_s3::types::LifecycleRule>> {
    use aws_sdk_s3::types::{
        ExpirationStatus, LifecycleExpiration, LifecycleRule, LifecycleRuleFilter,
    };
    let ours: Vec<&str> = rules.iter().map(|(prefix, _)| prefix.as_str()).collect();
    let mut built: Vec<LifecycleRule> = existing
        .into_iter()
        .filter(|rule| {
            let prefix = rule.filter().and_then(|f| f.prefix()).unwrap_or_default();
            prefix.is_empty() || !ours.contains(&prefix)
        })
        .collect();
    for (prefix, days) in rules {
        built.push(
            LifecycleRule::builder()
                .id(format!("expire-{}", prefix.trim_end_matches('/')))
                .status(ExpirationStatus::Enabled)
                .filter(LifecycleRuleFilter::builder().prefix(prefix.clone()).build())
                .expiration(LifecycleExpiration::builder().days(*days).build())
                .build()
                .context("building lifecycle rule")?,
        );
    }
    Ok(built)
}

#[cfg(test)]
impl S3ArtifactStore {
    /// Test-only constructor: builds a client from fixed, fake config
    /// (never resolved over the network — see the presigning tests below),
    /// bypassing `from_env` so URI/presigning tests don't have to touch the
    /// process-global S3 env vars that `from_env`'s own test does.
    fn for_test(bucket: &str) -> Self {
        let creds = Credentials::new(
            "test-access-key",
            "test-secret-key",
            None,
            None,
            CREDENTIAL_SOURCE,
        );
        let config = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(DEFAULT_REGION))
            .endpoint_url("https://example.r2.cloudflarestorage.com")
            .credentials_provider(creds)
            .force_path_style(true)
            .build();
        Self {
            client: Client::from_conf(config),
            bucket: bucket.to_owned(),
        }
    }
}

#[async_trait]
impl ArtifactStore for S3ArtifactStore {
    async fn put(&self, key: &str, bytes: Vec<u8>) -> Result<String> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes))
            .send()
            .await
            .with_context(|| format!("s3 put {key}"))?;
        Ok(self.uri(key))
    }

    async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let out = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("s3 get {key}"))?;
        Ok(out.body.collect().await?.to_vec())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .is_ok())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        // DeleteObject is idempotent on S3/R2 (missing key still returns 2xx).
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("s3 delete {key}"))?;
        Ok(())
    }

    async fn copy(&self, src: &str, dst: &str) -> Result<()> {
        // x-amz-copy-source is `<bucket>/<key>`; our keys are ascii/safe (see
        // keys::is_safe_key), so no percent-encoding is needed.
        let source = format!("{}/{src}", self.bucket);
        self.client
            .copy_object()
            .bucket(&self.bucket)
            .copy_source(source)
            .key(dst)
            .send()
            .await
            .with_context(|| format!("s3 copy {src} -> {dst}"))?;
        Ok(())
    }

    async fn apply_lifecycle(&self, rules: &[(String, i32)]) -> Result<()> {
        use aws_sdk_s3::types::BucketLifecycleConfiguration;
        // PutBucketLifecycleConfiguration replaces the bucket's WHOLE config,
        // so merge with whatever already exists (R2's default multipart-abort
        // rule, rules set by hand in the dashboard/wrangler). No existing
        // config (or no read permission) merges as empty. The merge policy
        // itself lives in `merge_lifecycle_rules` (unit-tested below without a
        // live bucket); this just supplies it the current remote state.
        let existing = self
            .client
            .get_bucket_lifecycle_configuration()
            .bucket(&self.bucket)
            .send()
            .await
            .map(|existing| existing.rules.unwrap_or_default())
            .unwrap_or_default();
        let built = merge_lifecycle_rules(existing, rules)?;
        let config = BucketLifecycleConfiguration::builder()
            .set_rules(Some(built))
            .build()
            .context("building lifecycle configuration")?;
        self.client
            .put_bucket_lifecycle_configuration()
            .bucket(&self.bucket)
            .lifecycle_configuration(config)
            .send()
            .await
            .with_context(|| format!("put lifecycle on bucket {}", self.bucket))?;
        Ok(())
    }

    fn uri(&self, key: &str) -> String {
        format!("s3://{}/{key}", self.bucket)
    }

    async fn presign_get(&self, key: &str, ttl: Duration) -> Result<Option<SignedUrl>> {
        Ok(Some(self.presign(ttl, true, key).await?))
    }

    async fn presign_put(&self, key: &str, ttl: Duration) -> Result<Option<SignedUrl>> {
        Ok(Some(self.presign(ttl, false, key).await?))
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_s3::types::{ExpirationStatus, LifecycleExpiration, LifecycleRule, LifecycleRuleFilter};
    use chuk_train_proto::env;

    use super::*;

    // -- uri() / presigning: fully local, no network (see `for_test`) -------

    #[test]
    fn uri_formats_as_an_s3_scheme_url_joining_bucket_and_key() {
        let store = S3ArtifactStore::for_test("my-bucket");
        assert_eq!(store.uri("ckpt-hot/r1/step_5/model.bin"), "s3://my-bucket/ckpt-hot/r1/step_5/model.bin");
    }

    #[tokio::test]
    async fn presign_get_builds_a_time_limited_path_style_url_without_touching_the_network() {
        let store = S3ArtifactStore::for_test("my-bucket");
        // SigV4 presigning is pure local computation (the request is signed
        // but never sent — see aws-sdk-s3's `orchestrate_with_stop_point`,
        // stopped `BeforeTransmit`). Bound it anyway so a regression that
        // *did* start hitting the network fails fast instead of hanging.
        let signed = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            store.presign_get("ckpt-hot/r1/step_5/model.bin", Duration::from_secs(60)),
        )
        .await
        .expect("presigning must not touch the network")
        .unwrap()
        .expect("the S3 backend always returns Some");

        assert!(
            signed.url.contains("my-bucket/ckpt-hot/r1/step_5/model.bin"),
            "unexpected url: {}",
            signed.url
        );
        assert!(signed.url.contains("X-Amz-Expires=60"), "unexpected url: {}", signed.url);
        let expected_expiry = now() + 60.0;
        assert!(
            (signed.expires_at - expected_expiry).abs() < 5.0,
            "expires_at={} expected~={expected_expiry}",
            signed.expires_at
        );
    }

    #[tokio::test]
    async fn presign_put_signs_a_different_request_than_presign_get() {
        let store = S3ArtifactStore::for_test("my-bucket");
        let get = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            store.presign_get("k", Duration::from_secs(30)),
        )
        .await
        .expect("presigning must not touch the network")
        .unwrap()
        .expect("Some");
        let put = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            store.presign_put("k", Duration::from_secs(30)),
        )
        .await
        .expect("presigning must not touch the network")
        .unwrap()
        .expect("Some");

        assert!(put.url.contains("X-Amz-Expires=30"), "unexpected url: {}", put.url);
        // GET and PUT sign over different canonical requests (different verb),
        // so the two presigned URLs must differ even for the same key/ttl.
        assert_ne!(get.url, put.url);
    }

    // -- from_env: the one test in this crate allowed to touch the S3 env
    // vars (mirrors `artifacts::clear_s3_env`'s reasoning — duplicated here
    // rather than shared since sharing it would mean editing mod.rs). Kept as
    // a single #[test] rather than several so it can't race a sibling test
    // over the same process-global vars.
    fn clear_s3_env() {
        std::env::remove_var(env::S3_ENDPOINT);
        std::env::remove_var(env::S3_REGION);
        std::env::remove_var(env::S3_ACCESS_KEY_ID);
        std::env::remove_var(env::S3_SECRET_ACCESS_KEY);
    }

    #[test]
    fn from_env_names_each_missing_var_in_turn_then_builds_with_a_default_region() {
        clear_s3_env();

        let err = S3ArtifactStore::from_env("bucket").unwrap_err();
        assert!(err.to_string().contains(env::S3_ENDPOINT), "unexpected error: {err}");

        std::env::set_var(env::S3_ENDPOINT, "https://example.r2.cloudflarestorage.com");
        let err = S3ArtifactStore::from_env("bucket").unwrap_err();
        assert!(err.to_string().contains(env::S3_ACCESS_KEY_ID), "unexpected error: {err}");

        std::env::set_var(env::S3_ACCESS_KEY_ID, "test-access-key");
        let err = S3ArtifactStore::from_env("bucket").unwrap_err();
        assert!(err.to_string().contains(env::S3_SECRET_ACCESS_KEY), "unexpected error: {err}");

        // All required vars set, no region override: builds fine (falls back
        // to DEFAULT_REGION, "auto" — what R2 signs against).
        std::env::set_var(env::S3_SECRET_ACCESS_KEY, "test-secret-key");
        let store = S3ArtifactStore::from_env("my-bucket").unwrap();
        assert_eq!(store.uri("k"), "s3://my-bucket/k");

        // An explicit region override is read too.
        std::env::set_var(env::S3_REGION, "us-east-1");
        let store = S3ArtifactStore::from_env("my-bucket").unwrap();
        assert_eq!(store.uri("k"), "s3://my-bucket/k");

        clear_s3_env();
    }

    // -- merge_lifecycle_rules: the merge policy, pulled out of
    // apply_lifecycle specifically so it's testable without a live bucket.

    fn rule_with_prefix(id: &str, prefix: &str) -> LifecycleRule {
        LifecycleRule::builder()
            .id(id)
            .status(ExpirationStatus::Enabled)
            .filter(LifecycleRuleFilter::builder().prefix(prefix.to_owned()).build())
            .expiration(LifecycleExpiration::builder().days(7).build())
            .build()
            .unwrap()
    }

    #[test]
    fn merge_appends_ours_onto_an_empty_existing_config() {
        let built = merge_lifecycle_rules(
            Vec::new(),
            &[("ckpt-hot/".to_owned(), 1), ("ckpt-final/".to_owned(), 30)],
        )
        .unwrap();
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].id(), Some("expire-ckpt-hot"));
        assert_eq!(built[0].expiration().and_then(LifecycleExpiration::days), Some(1));
        assert_eq!(built[1].id(), Some("expire-ckpt-final"));
        assert_eq!(built[1].expiration().and_then(LifecycleExpiration::days), Some(30));
    }

    #[test]
    fn merge_keeps_foreign_rules_and_replaces_a_rule_that_shares_our_prefix() {
        let existing = vec![
            rule_with_prefix("foreign", "other-app/"),
            // Same prefix as one of ours: dropped here, re-added below with
            // our own id/days rather than kept stale.
            rule_with_prefix("stale-ckpt-hot", "ckpt-hot/"),
        ];
        let built = merge_lifecycle_rules(existing, &[("ckpt-hot/".to_owned(), 1)]).unwrap();
        let ids: Vec<_> = built.iter().filter_map(LifecycleRule::id).collect();
        assert_eq!(ids, vec!["foreign", "expire-ckpt-hot"]);
    }

    #[test]
    fn merge_keeps_a_rule_that_has_no_prefix_filter_at_all() {
        // e.g. R2's default abort-incomplete-multipart-upload rule — no
        // prefix, so it's never "ours" and must survive every merge.
        let default_rule = LifecycleRule::builder()
            .id("abort-incomplete-multipart")
            .status(ExpirationStatus::Enabled)
            .build()
            .unwrap();
        let built = merge_lifecycle_rules(vec![default_rule], &[("ckpt-hot/".to_owned(), 1)]).unwrap();
        let ids: Vec<_> = built.iter().filter_map(LifecycleRule::id).collect();
        assert_eq!(ids, vec!["abort-incomplete-multipart", "expire-ckpt-hot"]);
    }

    #[test]
    fn merge_trims_the_trailing_slash_off_the_generated_rule_id() {
        let built = merge_lifecycle_rules(Vec::new(), &[("ckpt-hot/".to_owned(), 1)]).unwrap();
        assert_eq!(built[0].id(), Some("expire-ckpt-hot"));
    }
}

/// Live check that R2 accepts + persists our lifecycle rules. Ignored by
/// default; run with `.env` sourced:
///   cargo test -p chuk-train-controlplane artifacts::s3::live::lifecycle_round_trip -- --ignored --nocapture
#[cfg(test)]
mod live {
    use super::*;

    #[ignore]
    #[tokio::test]
    async fn lifecycle_round_trip() {
        let bucket =
            std::env::var("CHUK_TRAIN_S3_BUCKET").unwrap_or_else(|_| "chuk-train".to_owned());
        let Ok(store) = S3ArtifactStore::from_env(&bucket) else {
            eprintln!("skip: no S3/R2 env");
            return;
        };
        if let Err(e) = store
            .apply_lifecycle(&[("ckpt-hot/".to_owned(), 1), ("ckpt-final/".to_owned(), 30)])
            .await
        {
            let msg = e.to_string();
            if msg.contains("AccessDenied") || msg.contains("Access Denied") {
                eprintln!("skip: R2 token lacks lifecycle permission — set the rules in the Cloudflare dashboard or use an Admin R/W token");
                return;
            }
            panic!("apply lifecycle: {e}");
        }
        let got = store
            .client
            .get_bucket_lifecycle_configuration()
            .bucket(&bucket)
            .send()
            .await
            .expect("get lifecycle");
        let rules = got.rules();
        for r in rules {
            eprintln!(
                "rule id={:?} status={:?} days={:?}",
                r.id(),
                r.status(),
                r.expiration().and_then(|e| e.days())
            );
        }
        assert!(rules.len() >= 2, "expected our two rules");
    }
}
