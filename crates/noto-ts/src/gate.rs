//! Temporal consistency gate.
//!
//! A detection score that jumps outside what the recent history predicts is
//! held back instead of accepted. The expected range is the forecast's P10..P90
//! band widened by the SDS standard deviation of that score, so a label whose
//! scores are naturally noisy gets a wider band than a steady one.
//!
//! The gate costs one frame at first sight of an object: the first high score
//! is held, and the next consistent one is accepted because the history now
//! contains the first. That is the intended "confirm twice" behaviour.

use serde::{Deserialize, Serialize};

use crate::fusion::{forecast, HistoryMse, Predictor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateConfig {
    /// SDS standard deviations added on each side of the P10..P90 band.
    pub widen_sd: f64,
    /// Fewest history points that make the gate act; below it the gate abstains.
    pub min_history: usize,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { widen_sd: 1.0, min_history: 6 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalOutcome {
    /// Not enough history: the gate does not judge. Counts as a pass.
    Abstained,
    Accepted,
    Held,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVerdict {
    pub outcome: TemporalOutcome,
    pub observed: f64,
    pub expected: Option<f64>,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    /// Distance from the expectation in units of the widened band's half width / 1.2816.
    pub z: Option<f64>,
    pub method: String,
    pub reason: String,
}

impl TemporalVerdict {
    /// True when the observation may pass (accepted or abstained).
    pub fn passes(&self) -> bool {
        self.outcome != TemporalOutcome::Held
    }
}

/// Judges `observed` against the forecast made from `history` (oldest first).
pub fn temporal_gate(
    history: &[f64],
    observed: f64,
    sds_sd: Option<f64>,
    extra: &[&dyn Predictor],
    past_errors: &[HistoryMse],
    cfg: &GateConfig,
) -> TemporalVerdict {
    if !observed.is_finite() {
        return TemporalVerdict {
            outcome: TemporalOutcome::Held,
            observed,
            expected: None,
            lo: None,
            hi: None,
            z: None,
            method: "none".into(),
            reason: "관측값이 유한하지 않음".into(),
        };
    }
    let usable = history.iter().filter(|v| v.is_finite()).count();
    if usable < cfg.min_history {
        return TemporalVerdict {
            outcome: TemporalOutcome::Abstained,
            observed,
            expected: None,
            lo: None,
            hi: None,
            z: None,
            method: "none".into(),
            reason: format!("이력 {usable}점 < {}점이라 판정하지 않음", cfg.min_history),
        };
    }
    let f = forecast(history, 1, extra, past_errors);
    let widen = cfg.widen_sd * sds_sd.unwrap_or(0.0).max(0.0);
    let (p50, lo, hi) = (f.point[0], f.p10[0] - widen, f.p90[0] + widen);
    let half = ((hi - lo) / 2.0).max(1e-9);
    let z = (observed - p50) / (half / 1.2816);
    let inside = observed >= lo && observed <= hi;
    TemporalVerdict {
        outcome: if inside { TemporalOutcome::Accepted } else { TemporalOutcome::Held },
        observed,
        expected: Some(p50),
        lo: Some(lo),
        hi: Some(hi),
        z: Some(z),
        method: f.method,
        reason: if inside {
            "예측 띠 안".into()
        } else if observed > hi {
            format!("예측 띠 위로 벗어남 (관측 {observed:.3} > 상한 {hi:.3})")
        } else {
            format!("예측 띠 아래로 벗어남 (관측 {observed:.3} < 하한 {lo:.3})")
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(n: usize, level: f64) -> Vec<f64> {
        (0..n).map(|i| level + 0.02 * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0).collect()
    }

    #[test]
    fn abstains_without_enough_history() {
        let v = temporal_gate(&[0.5, 0.5, 0.5], 0.9, None, &[], &[], &GateConfig::default());
        assert_eq!(v.outcome, TemporalOutcome::Abstained);
        assert!(v.passes());
        assert!(v.reason.contains("3점"));
    }

    #[test]
    fn a_consistent_observation_passes_and_a_spike_is_held() {
        let h = steady(20, 0.55);
        let ok = temporal_gate(&h, 0.56, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(ok.outcome, TemporalOutcome::Accepted, "{:?}", ok);
        let spike = temporal_gate(&h, 0.95, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(spike.outcome, TemporalOutcome::Held);
        assert!(!spike.passes());
        assert!(spike.reason.contains("위로"));
        assert!(spike.z.unwrap() > 3.0, "{:?}", spike.z);
        let dip = temporal_gate(&h, 0.1, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(dip.outcome, TemporalOutcome::Held);
        assert!(dip.reason.contains("아래로"));
    }

    #[test]
    fn a_larger_sds_spread_widens_the_accepted_range() {
        let h = steady(20, 0.55);
        let obs = 0.68;
        let tight = temporal_gate(&h, obs, Some(0.0), &[], &[], &GateConfig::default());
        let wide = temporal_gate(&h, obs, Some(0.15), &[], &[], &GateConfig::default());
        assert_eq!(tight.outcome, TemporalOutcome::Held, "{:?}", tight);
        assert_eq!(wide.outcome, TemporalOutcome::Accepted, "{:?}", wide);
        assert!(wide.hi.unwrap() > tight.hi.unwrap());
    }

    #[test]
    fn the_second_consistent_high_score_is_accepted() {
        let mut h = steady(20, 0.2);
        let first = temporal_gate(&h, 0.7, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(first.outcome, TemporalOutcome::Held, "first sighting is held");
        h.push(0.7);
        let second = temporal_gate(&h, 0.71, Some(0.02), &[], &[], &GateConfig::default());
        assert_eq!(second.outcome, TemporalOutcome::Accepted, "{:?}", second);
    }

    #[test]
    fn non_finite_observations_are_held() {
        let v = temporal_gate(&steady(10, 0.5), f64::NAN, None, &[], &[], &GateConfig::default());
        assert_eq!(v.outcome, TemporalOutcome::Held);
    }
}
