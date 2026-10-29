//! How sharp the SDS's drift check and calibration factor are on series whose truth is known.

use noto_sds::{Scope, SdsStore, Track, Welford};

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The share of runs whose `drift_z` passes `limit` after `n` steady values and `k` values shifted by `shift` sd,
/// with a ring of `k`.
fn flagged(n: usize, k: usize, shift: f64, limit: f64, trials: u32) -> f64 {
    let mut rng = Rng(11 + k as u64);
    let mut hits = 0;
    for _ in 0..trials {
        let mut w = Welford::default();
        for _ in 0..n {
            w.push(rng.normal(), k);
        }
        for _ in 0..k {
            w.push(shift + rng.normal(), k);
        }
        hits += (w.drift_z().abs() > limit) as u32;
    }
    hits as f64 / trials as f64
}

#[test]
fn a_steady_scatter_seldom_drifts_and_a_two_sd_shift_does() {
    // The rings of the vision, depth and act tracks. Against the whole history in one value's spread, a 3 sd shift
    // over a ring of 8 never passed.
    for (track, k) in [("vision", 8usize), ("depth", 6), ("act", 4)] {
        let rates: Vec<String> = [0.0, 1.0, 2.0, 3.0].iter().map(|s| format!("{s:.0}σ {:.0}%", 100.0 * flagged(40, k, *s, 3.0, 2000))).collect();
        println!("[SDS] drift_z > 3, ring {k} ({track}): {}", rates.join(" · "));
        let (quiet, shifted) = (flagged(40, k, 0.0, 3.0, 2000), flagged(40, k, 2.0, 3.0, 2000));
        assert!(quiet < 0.02, "{track}: a steady scatter drifts {:.1} % of the time", quiet * 100.0);
        assert!(shifted > 0.75, "{track}: a 2 sd shift over the ring is seen {:.0} % of the time", shifted * 100.0);
    }
}

#[test]
fn a_calibration_factor_is_the_typical_ratio_not_the_mean_one() {
    // Depth readings off by a log-normal factor (median 1, ±30 %): the unbiased scale is 1. The arithmetic mean of
    // the ratios, which the factor was, sits at exp(σ²/2) = 1.046.
    let mut rng = Rng(5);
    let (runs, n) = (300, 120);
    let (mut now, mut before) = (Vec::with_capacity(runs), Vec::with_capacity(runs));
    for _ in 0..runs {
        let mut s = SdsStore::memory();
        let scope = Scope::new(Track::Depth, "probe", "");
        let mut w = Welford::default();
        for _ in 0..n {
            let r = (0.3 * rng.normal()).exp();
            w.push(r, 6);
            s.record_calibration(&scope, "scale", r * 5.0, 5.0);
        }
        now.push(s.calibration_factor(&scope, "scale").expect("enough to calibrate").0);
        before.push(1.0 + (w.mean - 1.0) * n as f64 / (n as f64 + 5.0));
    }
    let stats = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, (v.iter().map(|f| (f - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt(), v.iter().filter(|f| (**f - 1.0).abs() > 0.15).count())
    };
    let ((m, sd, off), (m0, sd0, off0)) = (stats(&now), stats(&before));
    println!("[SDS] calibration factor, {runs} runs of {n} ratios (truth 1.000): now {m:.3} ± {sd:.3} ({off} off by > 15 %) · arithmetic mean {m0:.3} ± {sd0:.3} ({off0})");
    assert!((m - 1.0).abs() < 0.01, "the factor centres on the typical ratio: {m:.3}");
    assert!(m0 - m > 0.03, "and not on the mean ratio: {m0:.3}");
    assert!(off <= 3, "{off} runs off by more than 15 %");
}
