//! Collect a job's output artifacts. An [`OutputRule`] pairs a class with a glob
//! and an upload policy; the collector lists matches under the glob's fixed base,
//! uploads every file of each new match, and reports one [`WorkerToCp::Artifact`]
//! per top-level match. It stays domain-free: it does not know what a match is,
//! only how to move its bytes and announce it.
//!
//! **Glob support (M1):** exactly one trailing wildcard segment, e.g.
//! `<sandbox>/ckpt/step_*`. The base is the wildcard-free prefix up to and
//! including the last `/`; the remainder is a single-segment pattern in which
//! `*` is the only wildcard (`?`/`[` are recognised for locating the base but
//! matched literally).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use chuk_compute_wire::{ArtifactClass, JobId, OutputRule, WorkerToCp};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::procio::worker_line;
use crate::sandbox::subst;
use crate::seq::Seq;

/// Characters that open a wildcard; the fixed base ends before the first one.
const WILDCARD_CHARS: [char; 3] = ['*', '?', '['];
/// The one wildcard the M1 matcher expands.
const STAR: char = '*';
/// Path and store-key separator.
const PATH_SEP: char = '/';

/// Somewhere the worker can push output bytes. Abstracted so the collection
/// logic is testable without the network; [`crate::httpclient::HttpClient`] is
/// the production implementation.
#[allow(async_fn_in_trait)]
pub trait BlobSink {
    async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()>;
}

impl BlobSink for crate::httpclient::HttpClient {
    async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
        crate::httpclient::HttpClient::upload(self, key, bytes).await
    }
}

/// Stateful collector for one [`OutputRule`]. Tracks which top-level matches it
/// has already uploaded so a repeated scan never re-sends them.
pub struct OutputCollector {
    class: ArtifactClass,
    base: PathBuf,
    pattern: String,
    key_prefix: String,
    ready_marker: Option<String>,
    collected: BTreeSet<String>,
}

impl OutputCollector {
    /// Build a collector from a rule, resolving the sandbox placeholder in its
    /// glob first.
    pub fn new(rule: &OutputRule, sandbox_path: &str) -> Self {
        let (base, pattern) = split_glob(&subst(&rule.glob, sandbox_path));
        Self {
            class: rule.class.clone(),
            base: PathBuf::from(base),
            pattern,
            key_prefix: rule.key_prefix.clone(),
            ready_marker: rule.ready_marker.clone(),
            collected: BTreeSet::new(),
        }
    }

    /// Upload every not-yet-collected match and announce each as an artifact.
    pub async fn collect<U: BlobSink>(
        &mut self,
        uploader: &U,
        job_id: &JobId,
        seq: &Seq,
        tx: &UnboundedSender<WorkerToCp>,
    ) {
        for (name, path) in self.pending() {
            let uploaded = self.upload_match(uploader, &name, &path, job_id, seq, tx).await;
            // Mark collected regardless: a match that fails to upload should not
            // be retried every tick forever.
            self.collected.insert(name.clone());
            if uploaded {
                let _ = tx.send(WorkerToCp::Artifact {
                    seq: seq.next(),
                    job_id: job_id.clone(),
                    class: self.class.clone(),
                    uri: format!("{}{PATH_SEP}{name}", self.key_prefix),
                    sha256: None,
                    bytes: None,
                    meta: Value::Null,
                });
            }
        }
    }

    /// Upload every file under one top-level match. Returns whether all files
    /// uploaded successfully (a partial failure suppresses the artifact report).
    async fn upload_match<U: BlobSink>(
        &self,
        uploader: &U,
        name: &str,
        path: &Path,
        job_id: &JobId,
        seq: &Seq,
        tx: &UnboundedSender<WorkerToCp>,
    ) -> bool {
        for file in files_under(path) {
            let rel = relpath(&self.base, &file);
            let key = format!("{}{PATH_SEP}{rel}", self.key_prefix);
            let bytes = match tokio::fs::read(&file).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    worker_line(seq, job_id, tx, &format!("output {name}: read failed: {error}"));
                    return false;
                }
            };
            if let Err(error) = uploader.upload(&key, bytes).await {
                worker_line(seq, job_id, tx, &format!("output {name}: upload failed: {error:#}"));
                return false;
            }
        }
        true
    }

    /// The top-level matches ready to collect now: entries under the base whose
    /// name matches the pattern, are not yet collected, and — for a directory
    /// gated by a ready marker — contain that marker.
    fn pending(&self) -> Vec<(String, PathBuf)> {
        let Ok(entries) = std::fs::read_dir(&self.base) else {
            return Vec::new();
        };
        let mut ready = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if self.collected.contains(&name) || !matches_pattern(&self.pattern, &name) {
                continue;
            }
            let path = entry.path();
            if !self.marker_present(&path) {
                continue;
            }
            ready.push((name, path));
        }
        ready.sort();
        ready
    }

    /// Whether a directory match's completion marker is present (always true for
    /// a file match, or when the rule sets no marker).
    fn marker_present(&self, path: &Path) -> bool {
        match &self.ready_marker {
            Some(marker) if path.is_dir() => path.join(marker).exists(),
            _ => true,
        }
    }
}

/// Split a glob into its fixed (wildcard-free) base — up to and including the
/// last `/` before the first wildcard — and the single-segment pattern that
/// follows. A wildcard-free glob splits at its final `/`.
fn split_glob(glob: &str) -> (String, String) {
    let scan_end = glob.find(|c| WILDCARD_CHARS.contains(&c)).unwrap_or(glob.len());
    match glob[..scan_end].rfind(PATH_SEP) {
        Some(slash) => (glob[..=slash].to_owned(), glob[slash + 1..].to_owned()),
        None => (String::new(), glob.to_owned()),
    }
}

/// Match a single-segment pattern against `name`. `*` matches any run of
/// characters (at most one `*` is meaningful); a pattern without `*` matches
/// exactly.
fn matches_pattern(pattern: &str, name: &str) -> bool {
    match pattern.split_once(STAR) {
        Some((prefix, suffix)) => {
            name.len() >= prefix.len() + suffix.len()
                && name.starts_with(prefix)
                && name.ends_with(suffix)
        }
        None => pattern == name,
    }
}

/// A file's path relative to `base` (which ends with a separator), as a
/// forward-slash store key fragment.
fn relpath(base: &Path, file: &Path) -> String {
    file.strip_prefix(base)
        .unwrap_or(file)
        .to_string_lossy()
        .into_owned()
}

/// Every file at or under `root`: `[root]` when it is a file, or all files in
/// its subtree (recursing into directories) when it is a directory.
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&path) {
                for entry in entries.flatten() {
                    stack.push(entry.path());
                }
            }
        } else {
            files.push(path);
        }
    }
    files.sort();
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_compute_wire::UploadPolicy;
    use std::sync::Mutex;
    use tokio::sync::mpsc;

    #[test]
    fn split_glob_isolates_the_wildcard_segment() {
        assert_eq!(
            split_glob("/box/ckpt/step_*"),
            ("/box/ckpt/".to_owned(), "step_*".to_owned())
        );
        assert_eq!(
            split_glob("/box/out/*.json"),
            ("/box/out/".to_owned(), "*.json".to_owned())
        );
        // No wildcard: split at the final separator (an exact-match rule).
        assert_eq!(
            split_glob("/box/report.json"),
            ("/box/".to_owned(), "report.json".to_owned())
        );
        // No separator before the wildcard.
        assert_eq!(split_glob("step_*"), (String::new(), "step_*".to_owned()));
    }

    #[test]
    fn matches_pattern_expands_a_single_star() {
        assert!(matches_pattern("step_*", "step_5"));
        assert!(matches_pattern("step_*", "step_"));
        assert!(!matches_pattern("step_*", "epoch_5"));
        assert!(matches_pattern("*.json", "report.json"));
        assert!(!matches_pattern("*.json", "report.txt"));
        assert!(matches_pattern("pre*suf", "preXYZsuf"));
        assert!(!matches_pattern("pre*suf", "presu")); // too short to satisfy both
        // No star: exact match only.
        assert!(matches_pattern("exact", "exact"));
        assert!(!matches_pattern("exact", "exactly"));
    }

    #[test]
    fn relpath_strips_the_base() {
        let base = PathBuf::from("/box/ckpt/");
        assert_eq!(relpath(&base, Path::new("/box/ckpt/step_5/model")), "step_5/model");
        assert_eq!(relpath(&base, Path::new("/box/ckpt/report.json")), "report.json");
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("chuk-outputs-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn files_under_lists_a_subtree_and_a_lone_file() {
        let dir = scratch("files");
        std::fs::create_dir_all(dir.join("step_1/sub")).unwrap();
        std::fs::write(dir.join("step_1/a.txt"), b"a").unwrap();
        std::fs::write(dir.join("step_1/sub/b.txt"), b"b").unwrap();
        let files = files_under(&dir.join("step_1"));
        assert_eq!(
            files,
            vec![dir.join("step_1/a.txt"), dir.join("step_1/sub/b.txt")]
        );

        std::fs::write(dir.join("lone"), b"x").unwrap();
        assert_eq!(files_under(&dir.join("lone")), vec![dir.join("lone")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Records every upload so a test can assert keys and bytes.
    #[derive(Default)]
    struct MockSink {
        uploads: Mutex<Vec<(String, Vec<u8>)>>,
        fail: bool,
    }

    impl BlobSink for MockSink {
        async fn upload(&self, key: &str, bytes: Vec<u8>) -> Result<()> {
            if self.fail {
                anyhow::bail!("sink down");
            }
            self.uploads.lock().unwrap().push((key.to_owned(), bytes));
            Ok(())
        }
    }

    fn ckpt_rule(base: &Path) -> OutputRule {
        OutputRule {
            class: ArtifactClass::from("checkpoint"),
            glob: format!("{}/step_*", base.display()),
            upload: UploadPolicy::OnAppearance,
            key_prefix: "runs/j1/ckpt".into(),
            ready_marker: Some(".ready".into()),
        }
    }

    fn artifacts(rx: &mut mpsc::UnboundedReceiver<WorkerToCp>) -> Vec<(String, String)> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let WorkerToCp::Artifact { class, uri, .. } = msg {
                out.push((class.to_string(), uri));
            }
        }
        out
    }

    #[tokio::test]
    async fn collect_uploads_ready_matches_once_and_gates_on_the_marker() {
        let dir = scratch("collect");
        // step_1 complete; step_2 not yet marked ready.
        std::fs::create_dir_all(dir.join("step_1")).unwrap();
        std::fs::write(dir.join("step_1/model.bin"), b"m1").unwrap();
        std::fs::write(dir.join("step_1/.ready"), b"").unwrap();
        std::fs::create_dir_all(dir.join("step_2")).unwrap();
        std::fs::write(dir.join("step_2/model.bin"), b"m2").unwrap();

        let sink = MockSink::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let seq = Seq::new();
        let mut collector = OutputCollector::new(&ckpt_rule(&dir), "unused");

        collector.collect(&sink, &JobId::from("j1"), &seq, &tx).await;
        // Only step_1 uploaded: its files keyed by path relative to the base.
        let uploads = sink.uploads.lock().unwrap().clone();
        let keys: Vec<&str> = uploads.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["runs/j1/ckpt/step_1/.ready", "runs/j1/ckpt/step_1/model.bin"]);
        assert_eq!(artifacts(&mut rx), vec![("checkpoint".to_owned(), "runs/j1/ckpt/step_1".to_owned())]);

        // Re-scan without changes: step_1 is not re-uploaded; step_2 still gated.
        collector.collect(&sink, &JobId::from("j1"), &seq, &tx).await;
        assert_eq!(sink.uploads.lock().unwrap().len(), 2);
        assert!(artifacts(&mut rx).is_empty());

        // Mark step_2 ready: now it collects on the next scan.
        std::fs::write(dir.join("step_2/.ready"), b"").unwrap();
        collector.collect(&sink, &JobId::from("j1"), &seq, &tx).await;
        assert_eq!(artifacts(&mut rx), vec![("checkpoint".to_owned(), "runs/j1/ckpt/step_2".to_owned())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn collect_suppresses_the_artifact_when_upload_fails() {
        let dir = scratch("failsink");
        std::fs::create_dir_all(dir.join("step_1")).unwrap();
        std::fs::write(dir.join("step_1/model.bin"), b"m1").unwrap();
        std::fs::write(dir.join("step_1/.ready"), b"").unwrap();

        let sink = MockSink { fail: true, ..MockSink::default() };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut collector = OutputCollector::new(&ckpt_rule(&dir), "unused");
        collector.collect(&sink, &JobId::from("j1"), &Seq::new(), &tx).await;

        // No artifact reported, and the match is marked so it does not retry.
        assert!(artifacts(&mut rx).is_empty());
        collector.collect(&sink, &JobId::from("j1"), &Seq::new(), &tx).await;
        assert!(sink.uploads.lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn collect_uploads_a_single_file_match() {
        let dir = scratch("single");
        std::fs::write(dir.join("report.json"), b"{}").unwrap();
        let rule = OutputRule {
            class: ArtifactClass::from("report"),
            glob: format!("{}/*.json", dir.display()),
            upload: UploadPolicy::OnExit,
            key_prefix: "runs/j1/reports".into(),
            ready_marker: None,
        };
        let sink = MockSink::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut collector = OutputCollector::new(&rule, "unused");
        collector.collect(&sink, &JobId::from("j1"), &Seq::new(), &tx).await;

        let uploads = sink.uploads.lock().unwrap().clone();
        assert_eq!(uploads, vec![("runs/j1/reports/report.json".to_owned(), b"{}".to_vec())]);
        assert_eq!(artifacts(&mut rx), vec![("report".to_owned(), "runs/j1/reports/report.json".to_owned())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn pending_on_a_missing_base_is_empty() {
        let collector = OutputCollector::new(
            &ckpt_rule(Path::new("/no/such/dir/at/all")),
            "unused",
        );
        assert!(collector.pending().is_empty());
    }
}
