//! Enforces the substrate rule from `chuk-compute-spec.md` §1: higher-layer
//! domain vocabulary never leaks into the compute-generic wire crate. The
//! canonical forbidden token is assembled from bytes below so this guard file
//! itself contains no occurrence of it.

use std::fs;
use std::path::Path;

/// The forbidden domain token (the five letters t, r, a, i, n), assembled so it
/// does not appear literally anywhere in this crate — the guard must pass on
/// itself.
fn forbidden_token() -> String {
    String::from_utf8(vec![b't', b'r', b'a', b'i', b'n']).expect("ascii")
}

fn scan(dir: &Path, needle: &str, hits: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("entry").path();
        if path.is_dir() {
            scan(&path, needle, hits);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            let text = fs::read_to_string(&path).expect("read_to_string").to_lowercase();
            for (i, line) in text.lines().enumerate() {
                if line.contains(needle) {
                    hits.push(format!("{}:{}", path.display(), i + 1));
                }
            }
        }
    }
}

#[test]
fn wire_crate_is_free_of_domain_vocabulary() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut hits = Vec::new();
    scan(&src, &forbidden_token(), &mut hits);
    assert!(
        hits.is_empty(),
        "higher-layer domain vocabulary leaked into the compute-generic wire crate at: {hits:?}"
    );
}
