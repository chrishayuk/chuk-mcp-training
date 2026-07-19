//! The workload model. A job is the whole contract the worker understands:
//! stage artifacts in, run one command under supervision, stream metrics,
//! collect artifacts out. Everything domain-specific rides as an opaque
//! [`Template`] tag or as artifact conventions the worker never inspects.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_TERM_GRACE_SECS;
use crate::ids::{ArtifactClass, CampaignId, JobId, Template};
use crate::WorkerId;

/// One unit of assigned work. A **batch** job (`service` unset) runs to
/// completion under a deadline; a **service** job runs until cancelled, drained,
/// or walled. The two shapes are the same type — the single `service` field is
/// what admits long-running workloads without a second job model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    /// Opaque workload-kind tag; see [`Template`]. The worker does not branch on it.
    pub template: Template,
    /// Argv of the command to run in the sandbox. `command[0]` is the program.
    pub command: Vec<String>,
    /// Environment for the command. Secret values are resolved control-plane-side
    /// at assign time, so the wire carries only plain resolved strings.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Artifacts staged into the sandbox before the command runs.
    #[serde(default)]
    pub inputs: Vec<InputArtifact>,
    /// What to collect from the sandbox, and when.
    #[serde(default)]
    pub outputs: Vec<OutputRule>,
    /// Wall-clock budget for a batch job; its effective deadline is
    /// `min(worker wall, now + max_runtime_secs)`. Unset for service jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_runtime_secs: Option<u64>,
    /// Grace between SIGTERM and SIGKILL when the job is stopped.
    #[serde(default = "default_term_grace_secs")]
    pub term_grace_secs: u64,
    /// Present iff this is a long-running service rather than a batch job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service: Option<ServiceSpec>,
    /// Service dependencies the control plane resolves to URLs and injects into
    /// `env` at assign time; the job is held until they are ready.
    #[serde(default)]
    pub needs: Vec<ServiceRef>,
    /// The campaign this job belongs to, if any (a flat fan-out group).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign: Option<CampaignId>,
    /// Scheduler placement hints.
    #[serde(default)]
    pub placement: Placement,
    /// Scoped, short-lived token the worker uses to mint upload URLs for this
    /// job's outputs. The worker never holds long-lived storage credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_grant: Option<String>,
}

fn default_term_grace_secs() -> u64 {
    DEFAULT_TERM_GRACE_SECS
}

/// An artifact to stage into the sandbox before the command runs. `uri` is a
/// presigned URL minted at assign time (reads need no worker-held credentials).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputArtifact {
    pub uri: String,
    /// Destination path within the sandbox.
    pub dest: String,
    /// Expected content hash, verified after download when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// A rule pairing an output class with a glob and when to upload matches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputRule {
    pub class: ArtifactClass,
    /// Glob, relative to the sandbox, matching the files to collect.
    pub glob: String,
    pub upload: UploadPolicy,
}

/// When the worker uploads an output matching an [`OutputRule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UploadPolicy {
    /// Once, when the command exits (final state, reports).
    OnExit,
    /// Each time a new match appears (periodic checkpoints — a preempted box
    /// loses minutes, not the whole run).
    OnAppearance,
    /// Tail the file continuously over the metric/log channel (live logs).
    Stream,
}

/// A long-running service a job exposes to the fabric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceSpec {
    /// Registry name other jobs resolve via [`ServiceRef`].
    pub name: String,
    pub ports: Vec<u16>,
    pub readiness: Readiness,
    pub restart: RestartPolicy,
}

/// How the worker decides a service is ready before announcing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Readiness {
    /// Ready as soon as the process is up.
    Immediate,
    /// Ready when an HTTP GET of `path` on `port` succeeds.
    HttpGet { path: String, port: u16 },
    /// Ready when `port` accepts a TCP connection.
    TcpOpen { port: u16 },
    /// Ready when a log line matches `pattern`.
    LogLine { pattern: String },
}

/// What to do when a service process exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RestartPolicy {
    Never,
    OnFailure,
    Always,
}

/// A dependency on a named service. The control plane resolves `name` to a URL
/// at assign time and injects it into the consuming job's `env` under `env`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceRef {
    pub name: String,
    /// Environment variable to receive the resolved service URL.
    pub env: String,
}

/// Scheduler placement hints — advisory, not guarantees.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    /// Prefer this worker (e.g. land a rollout next to its policy service).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefer_worker: Option<WorkerId>,
    /// Only place on workers whose labels match all of these.
    #[serde(default)]
    pub require_labels: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal batch job with only the required fields set in JSON, so serde
    /// defaults fill the rest (the parity path for a simple assignment).
    fn minimal_batch_json() -> &'static str {
        r#"{"id":"j1","template":"eval","command":["python","run.py"]}"#
    }

    #[test]
    fn defaults_fill_a_minimal_batch_job() {
        let job: Job = serde_json::from_str(minimal_batch_json()).unwrap();
        assert_eq!(job.term_grace_secs, DEFAULT_TERM_GRACE_SECS); // default fn hit
        assert!(job.env.is_empty());
        assert!(job.inputs.is_empty());
        assert!(job.outputs.is_empty());
        assert!(job.max_runtime_secs.is_none());
        assert!(job.service.is_none());
        assert!(job.needs.is_empty());
        assert!(job.campaign.is_none());
        assert!(job.output_grant.is_none());
        assert_eq!(job.placement, Placement::default());
    }

    #[test]
    fn full_batch_job_round_trips() {
        let mut env = BTreeMap::new();
        env.insert("SEED".into(), "81".into());
        let job = Job {
            id: JobId::from("j2"),
            template: Template::from("bench"),
            command: vec!["./run".into()],
            env,
            inputs: vec![InputArtifact {
                uri: "https://store/ds".into(),
                dest: "data/".into(),
                sha256: Some("abc".into()),
            }],
            outputs: vec![OutputRule {
                class: ArtifactClass::from("report"),
                glob: "out/*.json".into(),
                upload: UploadPolicy::OnExit,
            }],
            max_runtime_secs: Some(120),
            term_grace_secs: 10,
            service: None,
            needs: vec![ServiceRef { name: "policy".into(), env: "POLICY_URL".into() }],
            campaign: Some(CampaignId::from("camp-1")),
            placement: Placement {
                prefer_worker: Some(WorkerId::from("w-home")),
                require_labels: BTreeMap::from([("site".into(), "home".into())]),
            },
            output_grant: Some("scoped-token".into()),
        };
        let round: Job = serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
        assert_eq!(round, job);
    }

    #[test]
    fn service_job_and_its_enums_round_trip() {
        let spec = ServiceSpec {
            name: "cell-runtime".into(),
            ports: vec![8080],
            readiness: Readiness::HttpGet { path: "/healthz".into(), port: 8080 },
            restart: RestartPolicy::Always,
        };
        let round: ServiceSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(round, spec);

        // The readiness + upload + restart vocabularies serialise snake_case.
        assert_eq!(serde_json::to_value(&spec).unwrap()["readiness"]["kind"], "http_get");
        assert_eq!(serde_json::to_string(&UploadPolicy::OnAppearance).unwrap(), r#""on_appearance""#);
        assert_eq!(serde_json::to_string(&UploadPolicy::Stream).unwrap(), r#""stream""#);
        for r in [Readiness::Immediate, Readiness::TcpOpen { port: 22 }, Readiness::LogLine { pattern: "ready".into() }] {
            assert_eq!(serde_json::from_str::<Readiness>(&serde_json::to_string(&r).unwrap()).unwrap(), r);
        }
        for p in [RestartPolicy::Never, RestartPolicy::OnFailure, RestartPolicy::Always] {
            assert_eq!(serde_json::from_str::<RestartPolicy>(&serde_json::to_string(&p).unwrap()).unwrap(), p);
        }
    }
}
