//! Online statistics used by the score-dynamics store.
//!
//! Ported from logis-center `score_dynamics.rs` (original) via the adapted copy
//! in endmill-sandbox `src/sds.rs`: Welford mean/variance plus a short ring of
//! recent values, and the "decay shape" summary of a sorted score list.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Welford running mean and variance plus a ring of the latest values.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(default)]
    pub ring: Vec<f64>,
}

impl Welford {
    pub fn push(&mut self, x: f64, ring_len: usize) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
        self.ring.push(x);
        self.trim(ring_len);
    }

    fn trim(&mut self, ring_len: usize) {
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }

    /// Sample variance (n - 1); 0 below two observations.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    pub fn sd(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Mean of the ring, or the overall mean when the ring is empty.
    pub fn recent_mean(&self) -> f64 {
        if self.ring.is_empty() {
            return self.mean;
        }
        self.ring.iter().sum::<f64>() / self.ring.len() as f64
    }

    /// How many standard deviations the recent mean sits from the overall mean.
    pub fn drift_z(&self) -> f64 {
        let sd = self.sd();
        if sd <= 0.0 || self.ring.is_empty() {
            return 0.0;
        }
        (self.recent_mean() - self.mean) / sd
    }

    /// Chan's parallel update; the ring keeps the newest values of both.
    pub fn merge(&mut self, other: &Welford, ring_len: usize) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
        } else {
            let na = self.n as f64;
            let nb = other.n as f64;
            let n = na + nb;
            let delta = other.mean - self.mean;
            self.mean += delta * (nb / n);
            self.m2 += other.m2 + delta * delta * (na * nb / n);
            self.n += other.n;
            self.ring.extend(other.ring.iter().cloned());
        }
        self.trim(ring_len);
    }
}

/// Shape of a descending score list: how clearly the top item stands out.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct DecayShape {
    pub n: usize,
    pub top1: f64,
    pub top2: f64,
    pub margin: f64,
    pub top_gap_ratio: f64,
    pub tail_flatness: f64,
    pub entropy_norm: f64,
}

pub fn decay_shape(scores: &[f32]) -> Option<DecayShape> {
    let mut v: Vec<f64> = scores.iter().filter(|s| s.is_finite()).map(|s| *s as f64).collect();
    if v.len() < 2 {
        return None;
    }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let (top1, top2) = (v[0], v[1]);
    let margin = top1 - top2;
    let tail = &v[1..];
    let tail_median = tail[tail.len() / 2];
    let span = (top1 - tail_median).abs();
    let top_gap_ratio = if span > 1e-9 { (margin / span).clamp(0.0, 1.0) } else { 0.0 };
    let tail_mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let tail_var = tail.iter().map(|x| (x - tail_mean).powi(2)).sum::<f64>() / tail.len() as f64;
    let full = (v[0] - v[n - 1]).abs();
    let tail_flatness = if full > 1e-9 { (tail_var.sqrt() / full).clamp(0.0, 1.0) } else { 1.0 };
    let sd = tail_var.sqrt();
    let entropy_norm = if sd <= 1e-9 {
        1.0
    } else {
        let exps: Vec<f64> = v.iter().map(|x| ((x - v[0]) / sd).exp()).collect();
        let s: f64 = exps.iter().sum::<f64>().max(1e-12);
        let h: f64 = exps.iter().map(|e| e / s).filter(|p| *p > 1e-12).map(|p| -p * p.ln()).sum();
        let hmax = (n as f64).ln();
        if hmax > 1e-9 {
            (h / hmax).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    Some(DecayShape { n, top1, top2, margin, top_gap_ratio, tail_flatness, entropy_norm })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecayStat {
    pub margin: Welford,
    pub top_gap_ratio: Welford,
    pub tail_flatness: Welford,
    pub entropy_norm: Welford,
}

/// How often two labels were confused with each other (pair key `a|b`, a <= b).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionStat {
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub margin: Welford,
}

/// Pass/block counts of one gate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateStat {
    pub evaluated: u64,
    pub blocked: u64,
    /// Times this gate was the only one that blocked.
    pub sole: u64,
    #[serde(default)]
    pub severity: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeStat {
    #[serde(default)]
    pub baseline: HashMap<String, Welford>,
    #[serde(default)]
    pub calibration: HashMap<String, Welford>,
    #[serde(default)]
    pub decay: HashMap<String, DecayStat>,
    #[serde(default)]
    pub confusion: HashMap<String, ConfusionStat>,
    #[serde(default)]
    pub agreement: HashMap<String, Welford>,
    #[serde(default)]
    pub transition: HashMap<String, u64>,
    #[serde(default)]
    pub gates: HashMap<String, GateStat>,
    pub updated_at: i64,
}

impl ScopeStat {
    pub fn absorb(&mut self, other: ScopeStat, ring: usize, now_ms: i64) {
        for (k, v) in other.baseline {
            self.baseline.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.calibration {
            self.calibration.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.decay {
            let d = self.decay.entry(k).or_default();
            d.margin.merge(&v.margin, ring);
            d.top_gap_ratio.merge(&v.top_gap_ratio, ring);
            d.tail_flatness.merge(&v.tail_flatness, ring);
            d.entropy_norm.merge(&v.entropy_norm, ring);
        }
        for (k, v) in other.confusion {
            let c = self.confusion.entry(k).or_default();
            c.ties += v.ties;
            c.a_wins += v.a_wins;
            c.b_wins += v.b_wins;
            c.margin.merge(&v.margin, ring);
        }
        for (k, v) in other.agreement {
            self.agreement.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.transition {
            *self.transition.entry(k).or_insert(0) += v;
        }
        for (k, v) in other.gates {
            let g = self.gates.entry(k).or_default();
            g.evaluated += v.evaluated;
            g.blocked += v.blocked;
            g.sole += v.sole;
            g.severity.merge(&v.severity, ring);
        }
        self.updated_at = now_ms;
    }

    /// Total observations, used to decide which scopes to drop first when a
    /// ledger exceeds its size cap.
    pub fn observations(&self) -> u64 {
        self.baseline.values().map(|w| w.n).sum::<u64>()
            + self.calibration.values().map(|w| w.n).sum::<u64>()
            + self.confusion.values().map(|c| c.ties).sum::<u64>()
            + self.gates.values().map(|g| g.evaluated).sum::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    #[test]
    fn welford_matches_the_textbook_formulas() {
        let xs = [0.31, 0.44, 0.52, 0.38, 0.61, 0.47, 0.55, 0.29];
        let mut w = Welford::default();
        for x in xs {
            w.push(x, 4);
        }
        let (m, s) = naive(&xs);
        assert!((w.mean - m).abs() < 1e-12);
        assert!((w.sd() - s).abs() < 1e-12);
        assert_eq!(w.ring.len(), 4);
        assert_eq!(w.ring, xs[4..].to_vec(), "the ring keeps the newest values");
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let mut w = Welford::default();
        w.push(f64::NAN, 4);
        w.push(f64::INFINITY, 4);
        assert_eq!(w.n, 0);
        w.push(1.0, 4);
        assert_eq!(w.n, 1);
        assert_eq!(w.sd(), 0.0);
    }

    #[test]
    fn merge_equals_pushing_everything_in_order() {
        let a = [1.0, 2.0, 4.0, 8.0, 3.0];
        let b = [5.0, 7.0, 2.0, 9.0];
        let mut all = Welford::default();
        for x in a.iter().chain(b.iter()) {
            all.push(*x, 6);
        }
        let (mut wa, mut wb) = (Welford::default(), Welford::default());
        a.iter().for_each(|x| wa.push(*x, 6));
        b.iter().for_each(|x| wb.push(*x, 6));
        wa.merge(&wb, 6);
        assert_eq!(wa.n, all.n);
        assert!((wa.mean - all.mean).abs() < 1e-12);
        assert!((wa.sd() - all.sd()).abs() < 1e-12);
        assert_eq!(wa.ring, all.ring);
        // Merging an empty one changes nothing; merging into empty copies.
        let before = wa.clone();
        wa.merge(&Welford::default(), 6);
        assert_eq!(wa, before);
        let mut empty = Welford::default();
        empty.merge(&before, 6);
        assert_eq!(empty, before);
    }

    #[test]
    fn drift_z_flags_a_recent_shift() {
        let mut w = Welford::default();
        for _ in 0..30 {
            for x in [0.48, 0.52] {
                w.push(x, 6);
            }
        }
        assert!(w.drift_z().abs() < 0.3);
        for _ in 0..6 {
            w.push(0.9, 6);
        }
        assert!(w.drift_z() > 2.0, "drift {}", w.drift_z());
    }

    #[test]
    fn decay_shape_separates_a_clear_winner_from_a_flat_list() {
        let peaked = decay_shape(&[0.9, 0.2, 0.18, 0.15, 0.1, 0.05]).unwrap();
        let flat = decay_shape(&[0.52, 0.5, 0.49, 0.5, 0.48, 0.47]).unwrap();
        assert!(peaked.margin > flat.margin * 5.0);
        assert!(peaked.top_gap_ratio > flat.top_gap_ratio);
        assert!(flat.entropy_norm > peaked.entropy_norm);
        assert!(decay_shape(&[0.5]).is_none());
        assert!(decay_shape(&[f32::NAN, 0.5]).is_none());
    }

    #[test]
    fn absorb_adds_counts_and_merges_statistics() {
        let mut a = ScopeStat::default();
        let mut b = ScopeStat::default();
        a.baseline.entry("x".into()).or_default().push(1.0, 4);
        b.baseline.entry("x".into()).or_default().push(3.0, 4);
        b.transition.insert("a>b".into(), 2);
        a.transition.insert("a>b".into(), 1);
        b.gates.entry("g".into()).or_default().evaluated = 5;
        a.absorb(b, 4, 123);
        assert_eq!(a.baseline["x"].n, 2);
        assert!((a.baseline["x"].mean - 2.0).abs() < 1e-12);
        assert_eq!(a.transition["a>b"], 3);
        assert_eq!(a.gates["g"].evaluated, 5);
        assert_eq!(a.updated_at, 123);
        assert_eq!(a.observations(), 2 + 5);
    }
}
