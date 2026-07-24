//! Live chuk-experiments-server checks for [`super`]: a real server, real
//! credentials, `#[ignore]`d by default. They live in this file rather than
//! inline so the per-file coverage gate doesn't count code CI can never run —
//! `tests.rs` is excluded from the coverage report (see
//! `scripts/check_coverage.py`).

use super::*;
use crate::store::SqliteStore;

/// Live proof that the outbox actually recovers from a real failure, not just
/// a simulated one. Ignored by default; run in isolation (it mutates the
/// process-wide `CHUK_EXPERIMENTS_URL` env var) against a real local
/// chuk-experiments-server:
///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
///   cargo test -p chuk-train-controlplane experiments::live::outbox_recovers_after_experiments_server_was_unreachable -- --ignored --nocapture
#[ignore]
#[tokio::test]
async fn outbox_recovers_after_experiments_server_was_unreachable() {
    let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
    let key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
    if base.is_empty() || key.is_empty() {
        eprintln!(
            "skip: need {} + {} pointed at a real local chuk-experiments-server",
            env::EXPERIMENTS_URL,
            env::EXPERIMENTS_API_KEY
        );
        return;
    }

    let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    let train: TrainSpec = serde_json::from_value(json!({
        "code": { "name": "outbox-smoke-test", "sha": "0000000000000000000000000000000000000000" },
        "entrypoint": "true",
    }))
    .expect("valid TrainSpec");
    let spec = RunSpec::Train(Box::new(train));
    // A real `runs` row, exactly as Hub::submit creates before mirroring —
    // set_experiments_run_id later needs a matching row to update.
    let run_id = store
        .create_run("outbox-smoke-test", &spec, None, None, None)
        .await
        .expect("create_run");

    // Phase 1: nothing listens here -- the create must fail and land
    // durably in the outbox rather than being silently dropped.
    std::env::set_var(env::EXPERIMENTS_URL, "http://127.0.0.1:1");
    let down = Experiments::from_env(store.clone(), "http://localhost:9").expect("down client");
    down.report_created(run_id.clone(), spec, None).await;

    assert!(
        store.experiments_run_id(&run_id).await.unwrap().is_none(),
        "must not be marked mirrored -- the create never actually landed"
    );
    let pending = store
        .due_outbox_events(now_secs() + 3_700.0, 10) // past the max backoff, just to introspect the row
        .await
        .expect("due");
    assert_eq!(pending.len(), 1, "the failed create must be sitting in the outbox, not lost");
    assert_eq!(pending[0].attempts, 1);

    // Phase 2: replay it against a client pointed at the real, healthy
    // server -- proves the *outbox row*, not the client, is what makes
    // this durable (a fresh client + the same store recovers it).
    std::env::set_var(env::EXPERIMENTS_URL, &base);
    let up = Experiments::from_env(store.clone(), "http://localhost:9").expect("up client");
    let event: OutboxEvent = serde_json::from_str(&pending[0].payload).expect("payload");
    let delivered = up.attempt(pending[0].id, &run_id, event, pending[0].attempts).await;
    assert!(delivered, "retry against the real, now-healthy server must succeed");
    assert!(
        store.experiments_run_id(&run_id).await.unwrap().is_some(),
        "now genuinely mirrored on the real server"
    );

    std::env::remove_var(env::EXPERIMENTS_URL); // don't leak into other tests
}

/// Live proof that `bearer_for` really *prefers* a linked personal key —
/// not just that each half works in isolation. The shared default stays
/// valid (so `ensure()`, which always uses it, still succeeds); the
/// user's *linked* key is deliberately garbage. If resolution ever fell
/// back to the working shared default instead of genuinely preferring the
/// (here broken) personal key, this mirror call would succeed — it must
/// not.
///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
///   cargo test -p chuk-train-controlplane experiments::live::bearer_for_prefers_the_submitting_users_own_linked_key -- --ignored --nocapture
#[ignore]
#[tokio::test]
async fn bearer_for_prefers_the_submitting_users_own_linked_key() {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    use chuk_train_proto::Role;

    let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
    let real_key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
    if base.is_empty() || real_key.is_empty() {
        eprintln!(
            "skip: need {} + {} pointed at a real local chuk-experiments-server",
            env::EXPERIMENTS_URL,
            env::EXPERIMENTS_API_KEY
        );
        return;
    }
    let _ = real_key; // the shared default stays valid throughout this test

    let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    let email = "personal-key-test@example.com";
    store
        .upsert_user(email, "default", Role::Write)
        .await
        .expect("upsert user");

    let encryption_key = [3u8; 32];
    let encrypted = crate::crypto::encrypt(&encryption_key, "deliberately-invalid-personal-key");
    store
        .set_user_experiments_key(email, Some(&encrypted))
        .await
        .expect("link key");
    std::env::set_var(env::EXPERIMENTS_KEY_ENCRYPTION_KEY, STANDARD.encode(encryption_key));

    let exp = Experiments::from_env(store.clone(), "http://localhost:9").expect("client");
    let train: TrainSpec = serde_json::from_value(json!({
        "code": { "name": "bearer-for-test", "sha": "1111111111111111111111111111111111111111" },
        "entrypoint": "true",
    }))
    .expect("valid TrainSpec");
    let spec = RunSpec::Train(Box::new(train));
    let run_id = store
        .create_run("bearer-for-test", &spec, None, Some(email), None)
        .await
        .expect("create_run");

    exp.report_created(run_id.clone(), spec, None).await;

    assert!(
        store.experiments_run_id(&run_id).await.unwrap().is_none(),
        "must have failed using the user's own (deliberately invalid) linked key, \
         not silently succeeded by falling back to the still-valid shared default"
    );
    let pending = store
        .due_outbox_events(now_secs() + 3_700.0, 10)
        .await
        .expect("due");
    assert_eq!(pending.len(), 1, "the failed create must be sitting in the outbox");

    std::env::remove_var(env::EXPERIMENTS_KEY_ENCRYPTION_KEY);
}

/// Live proof of the actual feature: seed a queued run directly on a real
/// chuk-experiments-server (exactly as `enqueue_run` would), submit it via
/// `Hub::submit_from_experiment`, and confirm the harness execution
/// *attaches* to that seeded run (via the store's local
/// `experiments_run_id` mapping, the same thing `try_attach` sets) rather
/// than a second, unrelated run being minted.
///   CHUK_EXPERIMENTS_URL=http://localhost:8123 CHUK_EXPERIMENTS_API_KEY=<a real write key> \
///   cargo test -p chuk-train-controlplane experiments::live::submit_from_experiment_attaches_to_an_existing_queued_run -- --ignored --nocapture
#[ignore]
#[tokio::test]
async fn submit_from_experiment_attaches_to_an_existing_queued_run() {
    let base = std::env::var(env::EXPERIMENTS_URL).unwrap_or_default();
    let key = std::env::var(env::EXPERIMENTS_API_KEY).unwrap_or_default();
    if base.is_empty() || key.is_empty() {
        eprintln!(
            "skip: need {} + {} pointed at a real local chuk-experiments-server",
            env::EXPERIMENTS_URL,
            env::EXPERIMENTS_API_KEY
        );
        return;
    }

    let store: Arc<dyn Store> = Arc::new(SqliteStore::open(":memory:").await.expect("store"));
    let exp = Experiments::from_env(store.clone(), "http://localhost:9").expect("client");
    exp.ensure().await.expect("ensure default experiment");

    // Seed a queued run directly via REST, exactly as an external caller
    // (e.g. chuk-experiments-server's own `enqueue_run`) would.
    let http = reqwest::Client::new();
    let resp = http
        .post(format!("{base}/v1/experiments/{DEFAULT_EXPERIMENTS_EXPERIMENT}/runs"))
        .bearer_auth(&key)
        .json(&json!({
            "slug": format!("submit-from-experiment-smoke-{}", now_secs() as i64),
            "config": {
                "entrypoint": "true",
                "code": { "name": "submit-from-experiment-smoke", "sha": "0000000000000000000000000000000000000000" },
            },
            "status": "queued",
        }))
        .send()
        .await
        .expect("seed run");
    assert!(resp.status().is_success(), "seed run: {}", resp.status());
    let created: Value = resp.json().await.expect("parse seeded run");
    let ext_id = created["id"].as_str().expect("id").to_owned();

    let artifacts: Arc<dyn crate::artifacts::ArtifactStore> =
        Arc::new(crate::artifacts::FsArtifactStore::new(std::env::temp_dir()));
    let hub = crate::hub::Hub::new(store.clone(), artifacts, Some(exp), None);

    let run_id = hub
        .submit_from_experiment(&ext_id, None, None)
        .await
        .expect("submit_from_experiment");

    // mirror_created spawns the attach call rather than awaiting it
    // inline, so poll the store's local mapping instead of a blind sleep.
    let mut attached = None;
    for _ in 0..50 {
        if let Ok(Some(got)) = store.experiments_run_id(&run_id).await {
            attached = Some(got);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        attached.as_deref(),
        Some(ext_id.as_str()),
        "must attach to the seeded run, not mint or point at a different one"
    );
}
