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

/// Live check that R2 accepts + persists our lifecycle rules. Ignored by
/// default; run with `.env` sourced:
///   cargo test -p chuk-train-cp artifacts::s3::live::lifecycle_round_trip -- --ignored --nocapture
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
        use aws_sdk_s3::types::{
            BucketLifecycleConfiguration, ExpirationStatus, LifecycleExpiration, LifecycleRule,
            LifecycleRuleFilter,
        };
        let mut built = Vec::with_capacity(rules.len());
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
