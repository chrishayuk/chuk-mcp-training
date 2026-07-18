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
