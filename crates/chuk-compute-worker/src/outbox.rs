//! The replay outbox: the ordered record of every streamed event the worker has
//! produced for the job currently in flight, so a reconnecting session can
//! retransmit what the control plane has not yet acknowledged (chuk-compute
//! M3.2, survive-disconnect).
//!
//! The outbox holds at most one job's events — it is cleared when a new job is
//! assigned — so its footprint is bounded by a single job's stream. Each entry
//! pairs an event with its monotonic sequence number; the control plane
//! deduplicates any replayed event against its own high-water mark, so
//! over-replaying is safe while under-replaying would lose data.

use chuk_compute_wire::WorkerToCp;

/// The origin of the sequence space (`0`): the high-water value before any
/// streamed event has been acknowledged, and the value the seq-less
/// liveness/lifecycle messages map to (they never belong in the outbox).
pub const SEQ_ORIGIN: u64 = 0;

/// The monotonic sequence stamped on a streamed [`WorkerToCp`]. The seven
/// streamed variants each carry one; the non-streamed messages (`Heartbeat`,
/// `Drained`, `Hello`) never transit the outbox and map to [`SEQ_ORIGIN`] so the
/// function stays total over the `#[non_exhaustive]` enum. Mirrors the control
/// plane's own `event_seq`.
pub fn event_seq(event: &WorkerToCp) -> u64 {
    match event {
        WorkerToCp::JobStarted { seq, .. }
        | WorkerToCp::JobExited { seq, .. }
        | WorkerToCp::JobKilled { seq, .. }
        | WorkerToCp::ServiceReady { seq, .. }
        | WorkerToCp::Log { seq, .. }
        | WorkerToCp::Metric { seq, .. }
        | WorkerToCp::Artifact { seq, .. } => *seq,
        _ => SEQ_ORIGIN,
    }
}

/// Drop, in place, the outbox entries the control plane has already applied.
///
/// `resumed_high_water` is the highest seq the control plane last acknowledged
/// for this worker (echoed in `HelloAck`). A value of [`SEQ_ORIGIN`] means it has
/// acknowledged *nothing* — a fresh join, or a control-plane restart that wiped
/// its in-memory high-water — so nothing is trimmed and the whole outbox replays
/// (seq 0 included), which is exactly what "survive a control-plane restart"
/// requires. Once the high water is positive, streamed events arrive in order,
/// so everything at or below it is already processed and is dropped, keeping only
/// events strictly after it. Any event re-sent at the boundary is deduplicated by
/// the control plane against the same mark, so a too-eager replay never
/// double-applies.
pub fn trim_to_high_water(outbox: &mut Vec<(u64, WorkerToCp)>, resumed_high_water: u64) {
    if resumed_high_water > SEQ_ORIGIN {
        outbox.retain(|(seq, _)| *seq > resumed_high_water);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_compute_wire::{ArtifactClass, JobId, KillReason};
    use std::collections::BTreeMap;

    fn job() -> JobId {
        JobId::from("j1")
    }

    #[test]
    fn event_seq_reads_the_seq_of_each_streamed_variant() {
        let cases = [
            (WorkerToCp::JobStarted { seq: 10, job_id: job() }, 10),
            (WorkerToCp::JobExited { seq: 11, job_id: job(), code: 0 }, 11),
            (
                WorkerToCp::JobKilled { seq: 12, job_id: job(), reason: KillReason::Cancel },
                12,
            ),
            (
                WorkerToCp::ServiceReady { seq: 13, job_id: job(), ports: vec![8080] },
                13,
            ),
            (WorkerToCp::Log { seq: 14, job_id: job(), line: "hi".into() }, 14),
            (
                WorkerToCp::Metric {
                    seq: 15,
                    job_id: Some(job()),
                    step: Some(1),
                    values: BTreeMap::from([("loss".into(), 0.5)]),
                },
                15,
            ),
            (
                WorkerToCp::Artifact {
                    seq: 16,
                    job_id: job(),
                    class: ArtifactClass::from("log"),
                    uri: "u".into(),
                    sha256: None,
                    bytes: None,
                    meta: serde_json::Value::Null,
                },
                16,
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(event_seq(&event), expected);
        }
    }

    #[test]
    fn event_seq_maps_non_streamed_messages_to_the_origin() {
        // These carry no seq and never enter the outbox; they map to the origin
        // so the match stays total.
        assert_eq!(event_seq(&WorkerToCp::Heartbeat), SEQ_ORIGIN);
        assert_eq!(event_seq(&WorkerToCp::Drained), SEQ_ORIGIN);
    }

    fn outbox_of(seqs: &[u64]) -> Vec<(u64, WorkerToCp)> {
        seqs.iter()
            .map(|&s| (s, WorkerToCp::Log { seq: s, job_id: job(), line: format!("l{s}") }))
            .collect()
    }

    fn seqs(outbox: &[(u64, WorkerToCp)]) -> Vec<u64> {
        outbox.iter().map(|(seq, _)| *seq).collect()
    }

    #[test]
    fn trim_keeps_only_events_after_a_positive_high_water() {
        // The control plane applied through seq 1, so seq 0 and 1 are dropped and
        // only seq 2 remains to replay.
        let mut outbox = outbox_of(&[0, 1, 2]);
        trim_to_high_water(&mut outbox, 1);
        assert_eq!(seqs(&outbox), vec![2]);
    }

    #[test]
    fn a_fresh_join_or_restart_replays_everything() {
        // resumed_high_water == SEQ_ORIGIN → the control plane has acknowledged
        // nothing (fresh join, or a restart that wiped its high-water), so the
        // entire outbox — seq 0 included — is kept for replay.
        let mut outbox = outbox_of(&[0, 1, 2]);
        trim_to_high_water(&mut outbox, SEQ_ORIGIN);
        assert_eq!(seqs(&outbox), vec![0, 1, 2]);
    }

    #[test]
    fn trimming_at_the_last_seq_empties_the_outbox() {
        // The control plane applied through the final seq, so nothing replays.
        let mut outbox = outbox_of(&[0, 1, 2]);
        trim_to_high_water(&mut outbox, 2);
        assert!(outbox.is_empty());
    }
}
