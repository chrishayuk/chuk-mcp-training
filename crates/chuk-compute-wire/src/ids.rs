//! Opaque string identities that cross the wire. Each is a transparent newtype
//! so it serialises as a bare string but stays type-distinct in code.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Declare a transparent string-newtype id with the usual conversions.
macro_rules! string_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

string_id! {
    /// A worker's fabric identity, minted by the control plane at handshake and
    /// stable across reconnects.
    WorkerId
}

string_id! {
    /// One unit of assigned work.
    JobId
}

string_id! {
    /// Groups a fan-out of jobs (a config grid, rollout shards) as a flat set —
    /// no edges. Aggregation and budgets are the control plane's concern.
    CampaignId
}

string_id! {
    /// Opaque workload-kind tag on a [`crate::Job`] (e.g. the control plane's
    /// own conventions for evals, benchmarks, agents). The **worker must not
    /// branch on it** — it exists for control-plane dashboards and packing
    /// heuristics only.
    Template
}

string_id! {
    /// The kind of an output artifact — an **open string**, not an enum, so a
    /// new artifact kind never touches the protocol. Its values (e.g. `log`,
    /// `report`, `dataset`) are conventions of the control plane and the
    /// registry it reports into, opaque to both the wire and the worker.
    ArtifactClass
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn conversions_display_and_borrow() {
        let from_str = WorkerId::from("w-1");
        let from_string = WorkerId::from(String::from("w-1"));
        assert_eq!(from_str, from_string);
        assert_eq!(from_str.as_str(), "w-1");
        assert_eq!(from_str.to_string(), "w-1"); // Display

        // Every id type shares the macro-generated surface.
        assert_eq!(JobId::from("j").as_str(), "j");
        assert_eq!(CampaignId::from("c").to_string(), "c");
        assert_eq!(Template::from("eval").as_str(), "eval");
        assert_eq!(ArtifactClass::from("report").to_string(), "report");
    }

    #[test]
    fn serialises_transparently_as_a_bare_string() {
        assert_eq!(serde_json::to_string(&JobId::from("j9")).unwrap(), r#""j9""#);
        assert_eq!(
            serde_json::from_str::<Template>(r#""bench""#).unwrap(),
            Template::from("bench")
        );
    }

    #[test]
    fn ord_and_hash_make_ids_keyable() {
        let mut set = BTreeSet::new();
        set.insert(JobId::from("b"));
        set.insert(JobId::from("a"));
        // Ord sorts; Hash/Eq dedupe.
        assert!(!set.insert(JobId::from("a")));
        assert_eq!(set.iter().next().unwrap().as_str(), "a");
    }
}
