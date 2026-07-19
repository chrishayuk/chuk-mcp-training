//! Reconnect backoff jitter. The doubling bounds live in [`crate::constants`];
//! this module spreads each delay by a small pseudo-random amount so a fleet that
//! lost the control plane at the same instant does not reconnect in lockstep (a
//! thundering herd on the control plane as it comes back up).

use std::time::Duration;

/// The jitter added to a delay is at most `base / JITTER_DIVISOR`, i.e. up to a
/// quarter of the base. Small enough to leave the backoff curve intact, large
/// enough to de-synchronise a fleet.
const JITTER_DIVISOR: u32 = 4;

/// `base` plus up to `base / JITTER_DIVISOR` of jitter, selected by `entropy`.
///
/// Pure and deterministic in `entropy` so it can be unit-tested; the caller
/// supplies the randomness (e.g. the wall-clock subsecond) so no RNG dependency
/// is pulled in. A base too small to hold a nanosecond of jitter is returned
/// unchanged.
pub fn with_jitter(base: Duration, entropy: u64) -> Duration {
    let span_nanos = (base / JITTER_DIVISOR).as_nanos() as u64;
    if span_nanos == 0 {
        return base;
    }
    base + Duration::from_nanos(entropy % (span_nanos + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_entropy_yields_the_base_delay() {
        let base = Duration::from_secs(8);
        assert_eq!(with_jitter(base, 0), base);
    }

    #[test]
    fn jitter_stays_within_a_quarter_of_the_base() {
        let base = Duration::from_secs(8);
        let ceiling = base + base / JITTER_DIVISOR;
        // Even maximal entropy lands inside [base, base + base/4].
        for entropy in [1_u64, 7, 1_000, u64::MAX] {
            let delay = with_jitter(base, entropy);
            assert!(delay >= base, "jitter must never shorten the delay: {delay:?}");
            assert!(delay <= ceiling, "jitter must stay within a quarter: {delay:?}");
        }
    }

    #[test]
    fn entropy_moves_the_delay_inside_the_window() {
        let base = Duration::from_secs(8);
        // Roughly half of the jitter span → strictly inside the open window.
        let delay = with_jitter(base, (base / JITTER_DIVISOR).as_nanos() as u64 / 2);
        assert!(delay > base && delay < base + base / JITTER_DIVISOR);
    }

    #[test]
    fn a_base_too_small_for_jitter_is_returned_unchanged() {
        // base / 4 rounds down to zero nanoseconds → no room for jitter.
        let base = Duration::from_nanos(3);
        assert_eq!(with_jitter(base, u64::MAX), base);
    }
}
