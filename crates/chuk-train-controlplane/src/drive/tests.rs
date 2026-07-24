//! Live Google Drive checks for [`super`]: a real offline grant, real files,
//! `#[ignore]`d by default. They live in this file rather than inline so the
//! per-file coverage gate doesn't count code CI can never run — `tests.rs` is
//! excluded from the coverage report (see `scripts/check_coverage.py`).

use super::*;

/// Live round-trip against real Drive. Ignored by default (needs a grant in
/// the env); run with `.env` sourced:
///   cargo test -p chuk-train-controlplane drive::live::round_trip -- --ignored --nocapture
/// Uses a >1-chunk payload so the resumable 308 → finalise path is exercised.
#[ignore]
#[tokio::test]
async fn round_trip() {
    let client = match DriveClient::from_env().expect("build client") {
        Some(c) => c,
        None => {
            eprintln!("skip: no {} in env", env_vars::GOOGLE_REFRESH_TOKEN);
            return;
        }
    };
    client.probe().await.expect("token refresh");

    // 10 MiB pattern → two chunks (8 MiB + 2 MiB): a 308 then a finalise.
    let payload: Vec<u8> = (0..10 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let folder = format!("{ARCHIVE_ROOT_FOLDER}/_smoke");
    let suffix = now() as u64;
    let name = format!("round-trip-{suffix}.bin");

    let file_id = client
        .upload_to_path(&folder, &name, None, &payload)
        .await
        .expect("upload");
    eprintln!("uploaded {name} -> {file_id}");

    let got = client.download(&file_id).await.expect("download");
    assert_eq!(got.len(), payload.len(), "size round-trips");
    assert_eq!(got, payload, "bytes round-trip");

    client.delete(&file_id).await.expect("delete");
    eprintln!("deleted {file_id} — round-trip ok");
}
