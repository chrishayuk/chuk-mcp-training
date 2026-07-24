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
    /// Test-only constructor: builds a client from fixed, fake config against
    /// `endpoint`, bypassing `from_env` so tests don't have to touch the
    /// process-global S3 env vars that `from_env`'s own test does. Pointed at
    /// an unroutable host for the presigning tests (which never transmit) and
    /// at a loopback fake S3 for the request-shape tests.
    fn for_test_at(endpoint: &str, bucket: &str) -> Self {
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
            .endpoint_url(endpoint)
            .credentials_provider(creds)
            .force_path_style(true)
            // Match from_env: R2 rejects trailing-checksum trailers, and the
            // fake bucket below asserts on the request bodies we really send.
            .request_checksum_calculation(
                aws_sdk_s3::config::RequestChecksumCalculation::WhenRequired,
            )
            .build();
        Self {
            client: Client::from_conf(config),
            bucket: bucket.to_owned(),
        }
    }

    fn for_test(bucket: &str) -> Self {
        Self::for_test_at("https://example.r2.cloudflarestorage.com", bucket)
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
    use crate::fakehttp::{FakeHttp, Reply};

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

    // -- from_env: touches the same process-global S3 env vars as
    // `artifacts::mod::tests::s3_backend_requires_credentials_then_selects_
    // and_strips_the_bucket` — takes the shared `S3_ENV_LOCK` (see its doc
    // comment in mod.rs) so the two can't interleave under cargo test's
    // default parallelism. Kept as a single #[test] rather than several so
    // it can't race itself either.
    fn clear_s3_env() {
        std::env::remove_var(env::S3_ENDPOINT);
        std::env::remove_var(env::S3_REGION);
        std::env::remove_var(env::S3_ACCESS_KEY_ID);
        std::env::remove_var(env::S3_SECRET_ACCESS_KEY);
    }

    #[test]
    fn from_env_names_each_missing_var_in_turn_then_builds_with_a_default_region() {
        let _guard = super::super::lock_s3_env();
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

    // -- the object operations, against a loopback fake bucket ---------------
    //
    // A live R2 endpoint is the only way to prove the *bucket's* behaviour, but
    // it is not the only way to prove ours: pointed at `fakehttp`, the real
    // aws-sdk-s3 client signs and sends real requests, so these assert the
    // request shape we actually put on the wire (verb, path-style key, body,
    // copy-source header, merged lifecycle XML) and how we read the response
    // back. The live round-trip stays in `s3/tests.rs`.

    const S3_ERROR_XML: &str =
        r#"<?xml version="1.0" encoding="UTF-8"?><Error><Code>AccessDenied</Code><Message>no</Message></Error>"#;

    /// A bucket that answers every object request successfully: GET serves
    /// `body`, everything else is an empty 2xx (CopyObject needs its result
    /// element, which S3 sends as a 200 body).
    fn fake_bucket(body: &'static str) -> FakeHttp {
        FakeHttp::start(move |req, _| match req.method.as_str() {
            "GET" => Reply::ok(body),
            "PUT" if !req.header("x-amz-copy-source").is_empty() => Reply::ok(
                r#"<?xml version="1.0" encoding="UTF-8"?><CopyObjectResult><ETag>"e"</ETag></CopyObjectResult>"#,
            ),
            "DELETE" => Reply::new(204, Vec::new()),
            _ => Reply::ok(Vec::new()),
        })
    }

    #[tokio::test]
    async fn put_sends_the_bytes_path_style_and_returns_the_object_uri() {
        let bucket = fake_bucket("");
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        let uri = store.put("ckpt-hot/r1/model.bin", b"weights".to_vec()).await.unwrap();

        assert_eq!(uri, "s3://my-bucket/ckpt-hot/r1/model.bin");
        let req = &bucket.requests()[0];
        assert_eq!(req.method, "PUT");
        assert_eq!(req.path(), "/my-bucket/ckpt-hot/r1/model.bin");
        assert_eq!(req.body, b"weights");
        assert!(req.header("authorization").starts_with("AWS4-HMAC-SHA256"));
    }

    #[tokio::test]
    async fn get_returns_the_object_body() {
        let bucket = fake_bucket("weights");
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        let got = store.get("ckpt-hot/r1/model.bin").await.unwrap();

        assert_eq!(got, b"weights");
        assert_eq!(bucket.requests()[0].method, "GET");
        assert_eq!(bucket.requests()[0].path(), "/my-bucket/ckpt-hot/r1/model.bin");
    }

    #[tokio::test]
    async fn exists_heads_the_key_and_maps_a_miss_to_false() {
        // One bucket, two keys: `there` heads 200, anything else 404.
        let bucket = FakeHttp::start(|req, _| match req.path() {
            "/my-bucket/there" => Reply::ok(Vec::new()),
            _ => Reply::new(404, Vec::new()),
        });
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");

        assert!(store.exists("there").await.unwrap());
        assert!(!store.exists("gone").await.unwrap());
        assert_eq!(bucket.requests()[0].method, "HEAD");
    }

    #[tokio::test]
    async fn delete_removes_the_key() {
        let bucket = fake_bucket("");
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        store.delete("ckpt-hot/r1/model.bin").await.unwrap();

        assert_eq!(bucket.requests()[0].method, "DELETE");
        assert_eq!(bucket.requests()[0].path(), "/my-bucket/ckpt-hot/r1/model.bin");
    }

    #[tokio::test]
    async fn copy_names_the_source_bucket_qualified_and_the_destination_as_the_key() {
        let bucket = fake_bucket("");
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        store.copy("ckpt-hot/r1/model.bin", "ckpt-final/r1/model.bin").await.unwrap();

        let req = &bucket.requests()[0];
        assert_eq!(req.method, "PUT");
        assert_eq!(req.path(), "/my-bucket/ckpt-final/r1/model.bin");
        assert_eq!(req.header("x-amz-copy-source"), "my-bucket/ckpt-hot/r1/model.bin");
    }

    #[tokio::test]
    async fn a_failing_operation_names_the_key_in_the_error_context() {
        let bucket = FakeHttp::start(|_, _| Reply::new(403, S3_ERROR_XML));
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");

        let err = store.put("ckpt-hot/r1/model.bin", b"weights".to_vec()).await.unwrap_err();
        assert!(
            err.to_string().contains("s3 put ckpt-hot/r1/model.bin"),
            "unexpected error: {err}"
        );
        let err = store.get("ckpt-hot/r1/model.bin").await.unwrap_err();
        assert!(err.to_string().contains("s3 get"), "unexpected error: {err}");
        let err = store.delete("ckpt-hot/r1/model.bin").await.unwrap_err();
        assert!(err.to_string().contains("s3 delete"), "unexpected error: {err}");
        let err = store.copy("a", "b").await.unwrap_err();
        assert!(err.to_string().contains("s3 copy a -> b"), "unexpected error: {err}");
    }

    /// What S3/R2 returns for a bucket that already has one foreign rule.
    const EXISTING_LIFECYCLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<LifecycleConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Rule><ID>foreign</ID><Filter><Prefix>other-app/</Prefix></Filter><Status>Enabled</Status><Expiration><Days>7</Days></Expiration></Rule>
</LifecycleConfiguration>"#;

    #[tokio::test]
    async fn apply_lifecycle_reads_the_bucket_config_and_puts_back_the_merge() {
        let bucket = FakeHttp::start(|req, _| match req.method.as_str() {
            "GET" => Reply::ok(EXISTING_LIFECYCLE_XML),
            _ => Reply::ok(Vec::new()),
        });
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        store
            .apply_lifecycle(&[("ckpt-hot/".to_owned(), 1), ("ckpt-final/".to_owned(), 30)])
            .await
            .unwrap();

        let requests = bucket.requests();
        assert_eq!(requests[0].method, "GET", "reads the current config first");
        assert!(requests[0].target.contains("lifecycle"), "unexpected target: {}", requests[0].target);
        let put = String::from_utf8(requests[1].body.clone()).unwrap();
        assert_eq!(requests[1].method, "PUT");
        // The foreign rule survives; ours are appended with our ids + days.
        assert!(put.contains("<ID>foreign</ID>"), "foreign rule dropped: {put}");
        assert!(put.contains("<ID>expire-ckpt-hot</ID>"), "missing our rule: {put}");
        assert!(put.contains("<ID>expire-ckpt-final</ID>"), "missing our rule: {put}");
        assert!(put.contains("<Days>30</Days>"), "missing our expiry: {put}");
    }

    #[tokio::test]
    async fn apply_lifecycle_treats_an_unreadable_existing_config_as_empty() {
        // A token without lifecycle-read permission (or a bucket with no config
        // at all) must not stop us setting ours.
        let bucket = FakeHttp::start(|req, _| match req.method.as_str() {
            "GET" => Reply::new(403, S3_ERROR_XML),
            _ => Reply::ok(Vec::new()),
        });
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");
        store.apply_lifecycle(&[("ckpt-hot/".to_owned(), 1)]).await.unwrap();

        let put = String::from_utf8(bucket.requests()[1].body.clone()).unwrap();
        assert!(put.contains("<ID>expire-ckpt-hot</ID>"), "unexpected body: {put}");
    }

    #[tokio::test]
    async fn apply_lifecycle_surfaces_a_refused_put() {
        let bucket = FakeHttp::start(|_, _| Reply::new(403, S3_ERROR_XML));
        let store = S3ArtifactStore::for_test_at(&bucket.origin, "my-bucket");

        let err = store.apply_lifecycle(&[("ckpt-hot/".to_owned(), 1)]).await.unwrap_err();
        assert!(
            err.to_string().contains("put lifecycle on bucket my-bucket"),
            "unexpected error: {err}"
        );
    }
}

/// Live check that R2 accepts + persists our lifecycle rules — see
/// `s3/tests.rs`. Kept in a `tests.rs` sibling because it is `#[ignore]`d and
/// can never run in CI (it needs real R2 credentials): the coverage gate
/// excludes `tests.rs` files, so permanently-unrunnable lines don't count
/// against this module's coverage. Same reason as `hub/tests.rs`.
#[cfg(test)]
#[path = "s3/tests.rs"]
mod live;
