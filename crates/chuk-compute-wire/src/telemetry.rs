//! How the control plane configures a worker's system-telemetry sampler. The
//! samples themselves ride the ordinary metric channel under the `sys/`
//! namespace ([`crate::SYS_METRIC_PREFIX`]); only their cadence is negotiated.

use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_TELEMETRY_INTERVAL_SECS;

/// Delivered in the handshake acknowledgement. An absent or default config means
/// sample at [`DEFAULT_TELEMETRY_INTERVAL_SECS`]; `enabled = false` turns the
/// sampler off entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Seconds between system-telemetry samples.
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
    /// Whether to sample at all.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            interval_secs: DEFAULT_TELEMETRY_INTERVAL_SECS,
            enabled: true,
        }
    }
}

fn default_interval_secs() -> u64 {
    DEFAULT_TELEMETRY_INTERVAL_SECS
}

fn default_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_enabled_at_the_default_interval() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.interval_secs, DEFAULT_TELEMETRY_INTERVAL_SECS);
    }

    #[test]
    fn empty_object_deserialises_to_the_defaults() {
        // A control plane that sends `{}` gets the default sampler config, via
        // both serde default fns.
        let cfg: TelemetryConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg, TelemetryConfig::default());
    }

    #[test]
    fn explicit_values_override_and_round_trip() {
        let cfg = TelemetryConfig { interval_secs: 15, enabled: false };
        let round: TelemetryConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(round, cfg);
    }
}
