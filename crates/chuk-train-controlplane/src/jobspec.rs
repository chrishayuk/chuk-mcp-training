//! Translation from the control plane's training domain into a compute-generic
//! [`wire::Job`] (chuk-compute M1.1). This is where the "training-ness" turns
//! into a plain batch job — inputs, a command, env, and outputs — so the worker
//! stays domain-free (`chuk-compute-spec.md` §1, §6).
//!
//! Pure and side-effect-free: the caller resolves the entrypoint from the code
//! unit manifest and mints the input URIs + grant, then hands them here. Paths
//! the worker only knows at run time are expressed with
//! [`wire::SANDBOX_PLACEHOLDER`], which the worker substitutes for the job's
//! sandbox root.
//!
//! Consumed by the agent-websocket assign path at the M1.2 cutover; kept a
//! standalone, fully-tested unit until then.
#![allow(dead_code)]

use std::collections::BTreeMap;

use chuk_compute_wire as wire;
use chuk_train_proto::{
    script_env, RunId, ShellSpec, TrainSpec, CHECKPOINT_DIR_PREFIX, CHECKPOINT_META_FILE,
    CHECKPOINT_MODEL_FILE, CHECKPOINT_READY_MARKER, CKPT_HOT_PREFIX,
};
use serde_json::Value;

/// The interpreter a command string is handed to (matches the worker's spawn).
const SHELL_PROGRAM: &str = "/bin/sh";
const SHELL_FLAG: &str = "-c";

/// Opaque [`wire::Template`] tags — control-plane conventions for dashboards and
/// packing; the worker never branches on them.
const TEMPLATE_SHELL: &str = "shell";
const TEMPLATE_TRAIN: &str = "train";

/// Artifact class the control plane records checkpoint outputs under.
const ARTIFACT_CHECKPOINT: &str = "checkpoint";

/// Sandbox layout the control plane encodes into a train job; the worker just
/// follows the paths (all rooted at [`wire::SANDBOX_PLACEHOLDER`]).
const UNIT_SUBDIR: &str = "unit";
const CKPT_SUBDIR: &str = "ckpt";
const RESUME_SUBDIR: &str = "resume";
const DATA_SUBDIR: &str = "data";
const METRICS_FILE: &str = "metrics.jsonl";

/// The resolved inputs a train job needs staged, supplied by the caller (which
/// owns the code-unit manifest, artifact store, and grant minting).
pub struct TrainStaging<'a> {
    /// The entrypoint command resolved from the code unit's manifest.
    pub entrypoint_cmd: &'a str,
    /// Where the worker fetches the code unit archive from.
    pub code_unit_uri: &'a str,
    /// Scoped token for input fetch + output upload through the control plane.
    pub grant: &'a str,
    /// Present when this slice resumes from a prior checkpoint.
    pub resume: Option<ResumeStaging<'a>>,
    /// Present when the run declared a `data:` block, already resolved
    /// against chuk-datasets (spec §6/§7.3).
    pub data: Option<DataStaging>,
}

/// The resolved URIs of the checkpoint a resumed slice picks up from.
pub struct ResumeStaging<'a> {
    pub model_uri: &'a str,
    pub meta_uri: &'a str,
}

/// One dataset shard the worker fetches directly from its resolved location,
/// same `wire::InputArtifact { uri, dest, sha256, unpack }` contract as the
/// code unit — hash-verified on fetch (spec §6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardInput {
    pub uri: String,
    pub sha256: String,
}

/// A run's `data:` block, resolved to a concrete identity (spec §6/§7.3): the
/// dataset's `content_sha`, the batch plan's `plan_sha` if one was declared,
/// and the manifest's shards pre-warmed with fetch URLs so the worker's own
/// `chuk-datasets-client` re-resolves only on URL expiry, not on every run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataStaging {
    pub content_sha: String,
    pub plan_sha: Option<String>,
    pub shards: Vec<ShardInput>,
}

/// A shell run → a batch job that runs the command under a timeout, nothing
/// staged, nothing collected.
pub fn shell_job(run_id: &RunId, shell: &ShellSpec) -> wire::Job {
    base_job(
        run_id,
        TEMPLATE_SHELL,
        vec![
            SHELL_PROGRAM.to_owned(),
            SHELL_FLAG.to_owned(),
            shell.command.clone(),
        ],
        shell.timeout_s,
    )
}

/// A train run → a batch job: stage the code unit (and resume checkpoint), run
/// the entrypoint under the script-contract env, collect each checkpoint dir as
/// it appears, and stream the metrics file.
pub fn train_job(run_id: &RunId, train: &TrainSpec, staging: &TrainStaging<'_>) -> wire::Job {
    let mut job = base_job(
        run_id,
        TEMPLATE_TRAIN,
        vec![
            SHELL_PROGRAM.to_owned(),
            SHELL_FLAG.to_owned(),
            format!("cd {} && {}", sandboxed(UNIT_SUBDIR), staging.entrypoint_cmd),
        ],
        train.timeout_s,
    );
    job.env = script_environment(run_id, train, staging);
    job.inputs = train_inputs(staging);
    job.outputs = vec![wire::OutputRule {
        class: wire::ArtifactClass::from(ARTIFACT_CHECKPOINT),
        glob: sandboxed(&format!("{CKPT_SUBDIR}/{CHECKPOINT_DIR_PREFIX}*")),
        upload: wire::UploadPolicy::OnAppearance,
        // Uploads land at the same ckpt-hot/<run>/step_N/ keys as today, so the
        // retrieval + archive paths are unchanged; a dir is collected only once
        // the trainer has touched its .ready marker.
        key_prefix: format!("{CKPT_HOT_PREFIX}/{}", run_id.0),
        ready_marker: Some(CHECKPOINT_READY_MARKER.to_owned()),
    }];
    job.metrics_file = Some(sandboxed(METRICS_FILE));
    job.grant = Some(staging.grant.to_owned());
    job
}

/// A batch job with the common defaults filled in; the callers layer on env,
/// inputs, outputs, and grant.
fn base_job(run_id: &RunId, template: &str, command: Vec<String>, timeout_s: u64) -> wire::Job {
    wire::Job {
        id: wire::JobId::from(run_id.0.clone()),
        template: wire::Template::from(template),
        command,
        env: BTreeMap::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        metrics_file: None,
        max_runtime_secs: Some(timeout_s),
        term_grace_secs: wire::DEFAULT_TERM_GRACE_SECS,
        service: None,
        needs: Vec::new(),
        campaign: None,
        placement: wire::Placement::default(),
        grant: None,
    }
}

/// The `$CHUK_*` script-contract environment (spec §5.1), with run-time paths as
/// sandbox-relative placeholders.
fn script_environment(run_id: &RunId, train: &TrainSpec, staging: &TrainStaging<'_>) -> BTreeMap<String, String> {
    let config = train
        .config
        .as_ref()
        .map(|rel| sandboxed(&format!("{UNIT_SUBDIR}/{rel}")))
        .unwrap_or_default();
    let seed = train
        .seed
        .or_else(|| train.overrides.get("seed").and_then(Value::as_i64));
    BTreeMap::from([
        (script_env::RUN_ID.to_owned(), run_id.0.clone()),
        (script_env::CONFIG.to_owned(), config),
        (script_env::OVERRIDES.to_owned(), overrides_json(train)),
        (script_env::METRICS.to_owned(), sandboxed(METRICS_FILE)),
        (script_env::CKPT_DIR.to_owned(), sandboxed(CKPT_SUBDIR)),
        (
            script_env::RESUME_CKPT.to_owned(),
            if staging.resume.is_some() { sandboxed(RESUME_SUBDIR) } else { String::new() },
        ),
        (
            script_env::SEED.to_owned(),
            seed.map(|s| s.to_string()).unwrap_or_default(),
        ),
        (
            script_env::DATASET.to_owned(),
            staging.data.as_ref().map(|d| d.content_sha.clone()).unwrap_or_default(),
        ),
        (
            script_env::PLAN.to_owned(),
            staging.data.as_ref().and_then(|d| d.plan_sha.clone()).unwrap_or_default(),
        ),
    ])
}

/// The code unit (unpacked) plus, when resuming, the prior checkpoint's model +
/// meta staged into the resume dir.
fn train_inputs(staging: &TrainStaging<'_>) -> Vec<wire::InputArtifact> {
    // Dests are sandbox-rooted (like the command/env/globs) so the worker
    // resolves them to the same absolute place the entrypoint cd's into.
    let mut inputs = vec![wire::InputArtifact {
        uri: staging.code_unit_uri.to_owned(),
        dest: sandboxed(UNIT_SUBDIR),
        sha256: None,
        unpack: true,
    }];
    if let Some(resume) = &staging.resume {
        inputs.push(wire::InputArtifact {
            uri: resume.model_uri.to_owned(),
            dest: sandboxed(&format!("{RESUME_SUBDIR}/{CHECKPOINT_MODEL_FILE}")),
            sha256: None,
            unpack: false,
        });
        inputs.push(wire::InputArtifact {
            uri: resume.meta_uri.to_owned(),
            dest: sandboxed(&format!("{RESUME_SUBDIR}/{CHECKPOINT_META_FILE}")),
            sha256: None,
            unpack: false,
        });
    }
    if let Some(data) = &staging.data {
        for shard in &data.shards {
            inputs.push(wire::InputArtifact {
                uri: shard.uri.clone(),
                dest: sandboxed(&format!("{DATA_SUBDIR}/{}", shard.sha256)),
                sha256: Some(shard.sha256.clone()),
                unpack: false,
            });
        }
    }
    inputs
}

fn overrides_json(train: &TrainSpec) -> String {
    if train.overrides.is_null() {
        "{}".to_owned()
    } else {
        train.overrides.to_string()
    }
}

/// A path rooted at the worker-substituted sandbox placeholder.
fn sandboxed(rel: &str) -> String {
    format!("{}/{rel}", wire::SANDBOX_PLACEHOLDER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_train_proto::{CodeRef, RunSpec};

    fn train_spec() -> TrainSpec {
        let RunSpec::Train(train) = serde_json::from_value(serde_json::json!({
            "kind": "train",
            "code": {"name": "stub-trainer", "sha": "abc123"},
            "entrypoint": "train",
            "config": "configs/demo.json",
            "overrides": {"total_steps": 20, "seed": 7},
            "timeout_s": 900
        }))
        .unwrap() else {
            unreachable!()
        };
        *train
    }

    #[test]
    fn shell_job_is_a_bare_timed_command() {
        let job = shell_job(&RunId::from("RUN-1"), &ShellSpec { command: "nvidia-smi".into(), timeout_s: 60 });
        assert_eq!(job.id.as_str(), "RUN-1");
        assert_eq!(job.template.as_str(), TEMPLATE_SHELL);
        assert_eq!(job.command, vec!["/bin/sh", "-c", "nvidia-smi"]);
        assert_eq!(job.max_runtime_secs, Some(60));
        assert!(job.inputs.is_empty() && job.outputs.is_empty());
        assert!(job.metrics_file.is_none() && job.grant.is_none());
        assert_eq!(job.term_grace_secs, wire::DEFAULT_TERM_GRACE_SECS);
    }

    #[test]
    fn train_job_command_cds_into_the_unit_and_runs_the_entrypoint() {
        let staging = TrainStaging {
            entrypoint_cmd: "uv run train.py",
            code_unit_uri: "https://store/code",
            grant: "tok",
            resume: None,
            data: None,
        };
        let job = train_job(&RunId::from("RUN-2"), &train_spec(), &staging);
        assert_eq!(job.template.as_str(), TEMPLATE_TRAIN);
        assert_eq!(
            job.command,
            vec!["/bin/sh", "-c", "cd ${SANDBOX}/unit && uv run train.py"]
        );
        assert_eq!(job.max_runtime_secs, Some(900));
        assert_eq!(job.grant.as_deref(), Some("tok"));
        assert_eq!(job.metrics_file.as_deref(), Some("${SANDBOX}/metrics.jsonl"));
    }

    #[test]
    fn train_env_follows_the_script_contract_with_sandbox_paths() {
        let staging = TrainStaging {
            entrypoint_cmd: "run",
            code_unit_uri: "u",
            grant: "g",
            resume: None,
            data: None,
        };
        let job = train_job(&RunId::from("RUN-3"), &train_spec(), &staging);
        let env = &job.env;
        assert_eq!(env[script_env::RUN_ID], "RUN-3");
        assert_eq!(env[script_env::CONFIG], "${SANDBOX}/unit/configs/demo.json");
        assert_eq!(env[script_env::CKPT_DIR], "${SANDBOX}/ckpt");
        assert_eq!(env[script_env::METRICS], "${SANDBOX}/metrics.jsonl");
        // seed falls back to the overrides value; resume dir is empty w/o resume.
        assert_eq!(env[script_env::SEED], "7");
        assert_eq!(env[script_env::RESUME_CKPT], "");
        // overrides serialise as compact JSON.
        assert!(env[script_env::OVERRIDES].contains("\"total_steps\":20"));
    }

    #[test]
    fn train_inputs_stage_the_unit_and_checkpoint_output_is_on_appearance() {
        let staging = TrainStaging {
            entrypoint_cmd: "run",
            code_unit_uri: "https://store/code",
            grant: "g",
            resume: None,
            data: None,
        };
        let job = train_job(&RunId::from("RUN-4"), &train_spec(), &staging);
        assert_eq!(job.inputs.len(), 1);
        let unit = &job.inputs[0];
        assert_eq!(unit.dest, "${SANDBOX}/unit");
        assert!(unit.unpack);
        assert_eq!(unit.uri, "https://store/code");

        assert_eq!(job.outputs.len(), 1);
        let out = &job.outputs[0];
        assert_eq!(out.class.as_str(), ARTIFACT_CHECKPOINT);
        assert_eq!(out.glob, "${SANDBOX}/ckpt/step_*");
        assert_eq!(out.upload, wire::UploadPolicy::OnAppearance);
        // Uploads keep the existing ckpt-hot key layout; .ready gates collection.
        assert_eq!(out.key_prefix, "ckpt-hot/RUN-4");
        assert_eq!(out.ready_marker.as_deref(), Some(".ready"));
    }

    #[test]
    fn resume_stages_the_prior_checkpoint_and_points_resume_env_at_it() {
        let staging = TrainStaging {
            entrypoint_cmd: "run",
            code_unit_uri: "u",
            grant: "g",
            resume: Some(ResumeStaging {
                model_uri: "https://store/ckpt/model",
                meta_uri: "https://store/ckpt/meta",
            }),
            data: None,
        };
        let job = train_job(&RunId::from("RUN-5"), &train_spec(), &staging);
        assert_eq!(job.env[script_env::RESUME_CKPT], "${SANDBOX}/resume");
        // unit + model + meta, none of the resume files unpacked.
        assert_eq!(job.inputs.len(), 3);
        assert_eq!(job.inputs[1].dest, "${SANDBOX}/resume/model.safetensors");
        assert_eq!(job.inputs[2].dest, "${SANDBOX}/resume/meta.json");
        assert!(!job.inputs[1].unpack && !job.inputs[2].unpack);
    }

    #[test]
    fn data_stages_shards_and_sets_the_dataset_plan_env() {
        let staging = TrainStaging {
            entrypoint_cmd: "run",
            code_unit_uri: "u",
            grant: "g",
            resume: None,
            data: Some(DataStaging {
                content_sha: "dsha".into(),
                plan_sha: Some("psha".into()),
                shards: vec![
                    ShardInput { uri: "https://store/shards/a".into(), sha256: "a".into() },
                    ShardInput { uri: "https://store/shards/b".into(), sha256: "b".into() },
                ],
            }),
        };
        let job = train_job(&RunId::from("RUN-7"), &train_spec(), &staging);
        assert_eq!(job.env[script_env::DATASET], "dsha");
        assert_eq!(job.env[script_env::PLAN], "psha");
        // unit + two shards, hash-verified and never unpacked.
        assert_eq!(job.inputs.len(), 3);
        assert_eq!(job.inputs[1].dest, "${SANDBOX}/data/a");
        assert_eq!(job.inputs[1].sha256.as_deref(), Some("a"));
        assert!(!job.inputs[1].unpack);
        assert_eq!(job.inputs[2].dest, "${SANDBOX}/data/b");
        assert_eq!(job.inputs[2].sha256.as_deref(), Some("b"));
    }

    #[test]
    fn data_env_is_empty_without_a_data_block() {
        let staging = TrainStaging { entrypoint_cmd: "e", code_unit_uri: "u", grant: "g", resume: None, data: None };
        let job = train_job(&RunId::from("RUN-8"), &train_spec(), &staging);
        assert_eq!(job.env[script_env::DATASET], "");
        assert_eq!(job.env[script_env::PLAN], "");
    }

    #[test]
    fn seed_and_config_are_optional() {
        // A spec with no explicit config and no seed anywhere.
        let RunSpec::Train(spec) = serde_json::from_value(serde_json::json!({
            "kind": "train",
            "code": {"name": "n", "sha": "s"},
            "entrypoint": "e"
        }))
        .unwrap() else {
            unreachable!()
        };
        let _ = CodeRef { name: "n".into(), sha: "s".into() }; // touch the type
        let staging = TrainStaging { entrypoint_cmd: "e", code_unit_uri: "u", grant: "g", resume: None, data: None };
        let job = train_job(&RunId::from("RUN-6"), &spec, &staging);
        assert_eq!(job.env[script_env::CONFIG], "");
        assert_eq!(job.env[script_env::SEED], "");
        assert_eq!(job.env[script_env::OVERRIDES], "{}");
    }
}
