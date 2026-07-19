//! The two message enums that cross the socket, plus the handshake-resume and
//! kill-reason types. Both enums are `#[non_exhaustive]`: a peer must tolerate a
//! variant it does not know, which is how an old worker survives a newer control
//! plane and vice versa.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::capability::{Capabilities, WorkerClass};
use crate::ids::{ArtifactClass, JobId, WorkerId};
use crate::job::Job;
use crate::telemetry::TelemetryConfig;
use crate::UnixSeconds;

/// Worker → control plane. Streamed data messages carry a monotonic `seq` so a
/// reconnecting worker can replay its disk spool and the control plane can
/// deduplicate against a high-water mark (see [`Resume`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum WorkerToCp {
    /// First message on every connection; the token is exchanged for identity.
    Hello {
        protocol_version: u32,
        worker_semver: String,
        target_triple: String,
        token: String,
        capabilities: Capabilities,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resume: Option<Resume>,
    },
    /// Liveness ping.
    Heartbeat,
    JobStarted {
        seq: u64,
        job_id: JobId,
    },
    JobExited {
        seq: u64,
        job_id: JobId,
        code: i64,
    },
    JobKilled {
        seq: u64,
        job_id: JobId,
        reason: KillReason,
    },
    /// A service job has passed its readiness check and is accepting traffic.
    ServiceReady {
        seq: u64,
        job_id: JobId,
        ports: Vec<u16>,
    },
    /// One line of a job's output.
    Log {
        seq: u64,
        job_id: JobId,
        line: String,
    },
    /// A batch of metric values. `job_id` is absent for idle/host samples;
    /// `step` is absent for time-sampled (`sys/*`) metrics.
    Metric {
        seq: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        job_id: Option<JobId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<u64>,
        values: BTreeMap<String, f64>,
    },
    /// An output artifact the worker has uploaded out-of-band. `meta` is opaque
    /// to the wire — the control plane interprets it per the artifact's class.
    Artifact {
        seq: u64,
        job_id: JobId,
        class: ArtifactClass,
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes: Option<u64>,
        #[serde(default)]
        meta: Value,
    },
    /// The worker has wound down (flushed, stopped work) after a drain or wall.
    Drained,
}

/// Control plane → worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CpToWorker {
    /// Handshake accepted; the worker adopts this identity and class.
    HelloAck {
        worker_id: WorkerId,
        class: WorkerClass,
        #[serde(default)]
        telemetry: TelemetryConfig,
        /// Wall deadline for a leased worker; absent for a persistent one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wall_deadline: Option<UnixSeconds>,
    },
    /// Handshake refused. `min_protocol` plus the binary `url`/`sha256` let a
    /// self-updating worker fetch a compatible build.
    HelloReject {
        reason: String,
        min_protocol: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
    /// Run this job.
    AssignJob {
        job: Job,
    },
    /// Stop this job (SIGTERM → grace → SIGKILL).
    Cancel {
        job_id: JobId,
    },
    /// Wind down before the wall: stop taking work, flush, report `Drained`.
    /// `deadline` is the T-0 wall; the control plane reclaims at T-0 regardless.
    Drain {
        deadline: UnixSeconds,
    },
}

/// Why a job was killed. `OomGuard` is reserved for the telemetry-driven guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum KillReason {
    /// The worker's lease wall was reached.
    Wall,
    /// The job's own runtime budget was reached.
    MaxRuntime,
    /// An explicit cancel.
    Cancel,
    /// A drain the job did not finish before the wall.
    Drain,
    /// The system-telemetry OOM guard tripped.
    OomGuard,
}

/// Sent in [`WorkerToCp::Hello`] on reconnect so the control plane resynchronises
/// instead of double-assigning: the still-running jobs and the highest streamed
/// sequence the worker has produced. Replayed events carry their `seq`, so the CP
/// deduplicates against its own high-water mark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resume {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub running_jobs: Vec<JobId>,
    pub high_water: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Accelerator;
    use crate::ids::Template;
    use crate::PROTOCOL_VERSION;

    #[test]
    fn hello_round_trips_with_snake_case_tag() {
        let msg = WorkerToCp::Hello {
            protocol_version: PROTOCOL_VERSION,
            worker_semver: "0.1.0".into(),
            target_triple: "aarch64-apple-darwin".into(),
            token: "t".into(),
            capabilities: Capabilities {
                os: "macos".into(),
                arch: "aarch64".into(),
                cpu_cores: 10,
                ram_bytes: 34_359_738_368,
                free_disk_bytes: 500_000_000_000,
                preemptible: false,
                accelerator: Accelerator::Mps {
                    chip: "Apple M2".into(),
                    unified_memory_bytes: 34_359_738_368,
                },
                labels: BTreeMap::new(),
            },
            resume: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"hello""#), "{json}");
        assert_eq!(serde_json::from_str::<WorkerToCp>(&json).unwrap(), msg);
    }

    #[test]
    fn assign_job_round_trips() {
        let msg = CpToWorker::AssignJob {
            job: Job {
                id: JobId::from("j1"),
                template: Template::from("eval"),
                command: vec!["python".into(), "run.py".into()],
                env: BTreeMap::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                metrics_file: None,
                max_runtime_secs: Some(600),
                term_grace_secs: crate::DEFAULT_TERM_GRACE_SECS,
                service: None,
                needs: Vec::new(),
                campaign: None,
                placement: crate::Placement::default(),
                grant: None,
            },
        };
        let value = serde_json::to_value(&msg).unwrap();
        assert_eq!(value["type"], "assign_job");
        assert_eq!(value["job"]["template"], "eval");
        assert_eq!(serde_json::from_value::<CpToWorker>(value).unwrap(), msg);
    }

    #[test]
    fn unknown_fields_are_tolerated_forward_compat() {
        // A future control plane adds a field; today's worker must still parse.
        let json = r#"{"type":"cancel","job_id":"j9","future_field":true}"#;
        let msg: CpToWorker = serde_json::from_str(json).unwrap();
        assert_eq!(msg, CpToWorker::Cancel { job_id: JobId::from("j9") });
    }

    /// Serialise → deserialise must be the identity for any wire message.
    fn assert_worker_round_trip(msg: &WorkerToCp) {
        let json = serde_json::to_string(msg).unwrap();
        assert_eq!(&serde_json::from_str::<WorkerToCp>(&json).unwrap(), msg);
    }

    fn assert_cp_round_trip(msg: &CpToWorker) {
        let json = serde_json::to_string(msg).unwrap();
        assert_eq!(&serde_json::from_str::<CpToWorker>(&json).unwrap(), msg);
    }

    #[test]
    fn every_worker_to_cp_variant_round_trips() {
        let j = || JobId::from("j1");
        for msg in [
            WorkerToCp::Heartbeat,
            WorkerToCp::Drained,
            WorkerToCp::JobStarted { seq: 1, job_id: j() },
            WorkerToCp::JobExited { seq: 2, job_id: j(), code: 0 },
            WorkerToCp::JobKilled { seq: 3, job_id: j(), reason: KillReason::Wall },
            WorkerToCp::ServiceReady { seq: 4, job_id: j(), ports: vec![8080, 9090] },
            WorkerToCp::Log { seq: 5, job_id: j(), line: "hello".into() },
            WorkerToCp::Metric {
                seq: 6,
                job_id: Some(j()),
                step: Some(100),
                values: BTreeMap::from([("loss".into(), 0.5)]),
            },
            // A host/idle sample: no job, no step (the sys/* shape).
            WorkerToCp::Metric {
                seq: 7,
                job_id: None,
                step: None,
                values: BTreeMap::from([("sys/gpu_util".into(), 0.9)]),
            },
            WorkerToCp::Artifact {
                seq: 8,
                job_id: j(),
                class: ArtifactClass::from("report"),
                uri: "https://store/r".into(),
                sha256: Some("abc".into()),
                bytes: Some(1024),
                meta: serde_json::json!({"k": "v"}),
            },
        ] {
            assert_worker_round_trip(&msg);
        }
    }

    #[test]
    fn every_cp_to_worker_variant_round_trips() {
        for msg in [
            CpToWorker::HelloAck {
                worker_id: WorkerId::from("w1"),
                class: WorkerClass::Leased,
                telemetry: TelemetryConfig::default(),
                wall_deadline: Some(1_700_000_000.0),
            },
            CpToWorker::HelloReject {
                reason: "protocol too old".into(),
                min_protocol: PROTOCOL_VERSION,
                url: Some("https://cp/agent/x".into()),
                sha256: Some("deadbeef".into()),
            },
            CpToWorker::Cancel { job_id: JobId::from("j1") },
            CpToWorker::Drain { deadline: 1_700_000_100.0 },
        ] {
            assert_cp_round_trip(&msg);
        }
    }

    #[test]
    fn kill_reasons_and_resume_round_trip() {
        for reason in [
            KillReason::Wall,
            KillReason::MaxRuntime,
            KillReason::Cancel,
            KillReason::Drain,
            KillReason::OomGuard,
        ] {
            let round: KillReason =
                serde_json::from_str(&serde_json::to_string(&reason).unwrap()).unwrap();
            assert_eq!(round, reason);
        }
        assert_eq!(serde_json::to_string(&KillReason::OomGuard).unwrap(), r#""oom_guard""#);

        let resume = Resume {
            worker_id: WorkerId::from("w1"),
            running_jobs: vec![JobId::from("j1"), JobId::from("j2")],
            high_water: 42,
        };
        let round: Resume =
            serde_json::from_str(&serde_json::to_string(&resume).unwrap()).unwrap();
        assert_eq!(round, resume);
    }

    #[test]
    fn hello_ack_defaults_telemetry_when_absent() {
        let json = r#"{"type":"hello_ack","worker_id":"w1","class":"persistent"}"#;
        let msg: CpToWorker = serde_json::from_str(json).unwrap();
        let CpToWorker::HelloAck { class, telemetry, wall_deadline, .. } = msg else {
            panic!("expected hello_ack");
        };
        assert_eq!(class, WorkerClass::Persistent);
        assert_eq!(telemetry, TelemetryConfig::default());
        assert!(wall_deadline.is_none()); // persistent workers have no wall
    }
}
