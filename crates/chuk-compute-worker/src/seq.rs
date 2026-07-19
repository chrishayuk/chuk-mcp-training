//! The per-session sequence counter. Every streamed `WorkerToCp` that carries a
//! `seq` field draws the next value from one shared counter, so the control
//! plane can order and deduplicate a worker's stream against a high-water mark.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A cheaply-cloneable handle to one session's monotonic sequence counter. Each
/// clone shares the same underlying atomic, so sequence numbers stay unique and
/// monotonic across every task that streams on behalf of the session.
#[derive(Clone, Default)]
pub struct Seq(Arc<AtomicU64>);

impl Seq {
    /// A fresh counter whose first [`Seq::next`] yields `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// The next sequence value to stamp on a streamed message. Returns the
    /// pre-increment value, so a fresh counter hands out `0, 1, 2, …`.
    pub fn next(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_counter_starts_at_zero_and_increments() {
        let seq = Seq::new();
        assert_eq!(seq.next(), 0);
        assert_eq!(seq.next(), 1);
        assert_eq!(seq.next(), 2);
    }

    #[test]
    fn clones_share_one_underlying_counter() {
        let seq = Seq::new();
        let clone = seq.clone();
        assert_eq!(seq.next(), 0);
        // The clone continues the same sequence rather than starting over.
        assert_eq!(clone.next(), 1);
        assert_eq!(seq.next(), 2);
    }

    #[test]
    fn default_matches_new() {
        let seq = Seq::default();
        assert_eq!(seq.next(), 0);
    }
}
