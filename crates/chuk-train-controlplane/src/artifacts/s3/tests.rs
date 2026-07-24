//! Live S3/R2 checks for [`super`]: real credentials, real bucket, `#[ignore]`d
//! by default. They live in this file rather than inline so the per-file
//! coverage gate doesn't count code CI can never run — `tests.rs` is excluded
//! from the coverage report (see `scripts/check_coverage.py`).

use super::*;

/// Live check that R2 accepts + persists our lifecycle rules. Ignored by
/// default; run with `.env` sourced:
///   cargo test -p chuk-train-controlplane artifacts::s3::live::lifecycle_round_trip -- --ignored --nocapture
#[ignore]
#[tokio::test]
async fn lifecycle_round_trip() {
    let bucket = std::env::var("CHUK_TRAIN_S3_BUCKET").unwrap_or_else(|_| "chuk-train".to_owned());
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
