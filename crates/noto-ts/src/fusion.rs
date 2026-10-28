use serde::{Deserialize, Serialize};

use crate::holt::holt_damped;

pub const HOLDOUT: usize = 4;
pub const MIN_POINTS: usize = 6;
pub const HISTORY_CAP: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct Band {
    pub point: Vec<f64>,
    pub lo: Option<Vec<f64>>,
    pub hi: Option<Vec<f64>>,
}

pub trait Predictor {
    fn name(&self) -> &str;
    fn min_history(&self) -> usize;
    fn max_horizon(&self) -> usize {
        usize::MAX
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String>;
}

pub struct HoltPredictor;

impl Predictor for HoltPredictor {
    fn name(&self) -> &str {
        "holt"
    }
    fn min_history(&self) -> usize {
        3
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let (point, sd, _) = holt_damped(series, horizon);
        let lo = point.iter().zip(&sd).map(|(p, s)| p - 1.2816 * s).collect();
        let hi = point.iter().zip(&sd).map(|(p, s)| p + 1.2816 * s).collect();
        Ok(Band { point, lo: Some(lo), hi: Some(hi) })
    }
}

pub struct LastPredictor;

impl Predictor for LastPredictor {
    fn name(&self) -> &str {
        "last"
    }
    fn min_history(&self) -> usize {
        1
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let last = *series.last().ok_or("empty series")?;
        Ok(Band { point: vec![last; horizon], lo: None, hi: None })
    }
}

pub struct EwmaPredictor {
    pub alpha: f64,
}

impl Predictor for EwmaPredictor {
    fn name(&self) -> &str {
        "ewma"
    }
    fn min_history(&self) -> usize {
        2
    }
    fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
        let mut it = series.iter();
        let mut level = *it.next().ok_or("empty series")?;
        for v in it {
            level = self.alpha * v + (1.0 - self.alpha) * level;
        }
        Ok(Band { point: vec![level; horizon], lo: None, hi: None })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryMse {
    pub method: String,
    pub mse: f64,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodWeight {
    pub method: String,
    pub rmse: f64,
    pub history_rmse: Option<f64>,
    pub history_n: u64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Forecast {
    pub point: Vec<f64>,
    pub p10: Vec<f64>,
    pub p90: Vec<f64>,
    pub weights: Vec<MethodWeight>,
    pub rmse_fused: f64,
    pub holdout: usize,
    pub method: String,
    pub notes: Vec<String>,
}

fn variance(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let m = y.iter().sum::<f64>() / y.len() as f64;
    y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (y.len() - 1) as f64
}

pub fn forecast(series: &[f64], horizon: usize, extra: &[&dyn Predictor], history: &[HistoryMse]) -> Forecast {
    let horizon = horizon.max(1);
    let mut notes = Vec::new();
    let clean: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.len() != series.len() {
        notes.push(format!("비유한 값 {}개를 제외했습니다", series.len() - clean.len()));
    }
    let n = clean.len();
    if n == 0 {
        return Forecast {
            point: vec![0.0; horizon],
            p10: vec![0.0; horizon],
            p90: vec![0.0; horizon],
            weights: vec![],
            rmse_fused: 0.0,
            holdout: 0,
            method: "none".into(),
            notes: vec!["이력이 없습니다".into()],
        };
    }

    let holt = HoltPredictor;
    let last = LastPredictor;
    let ewma = EwmaPredictor { alpha: 0.3 };
    let mut pool: Vec<&dyn Predictor> = vec![&last, &ewma, &holt];
    pool.extend(extra.iter().copied());

    if n < MIN_POINTS {
        notes.push(format!("이력 {n}점 < {MIN_POINTS}점이라 백테스트 없이 마지막 값을 사용했습니다"));
        let l = *clean.last().unwrap();
        let sd = variance(&clean).sqrt();
        let widen = |h: usize, sign: f64| l + sign * 1.2816 * sd * (h as f64).sqrt();
        return Forecast {
            point: vec![l; horizon],
            p10: (1..=horizon).map(|h| widen(h, -1.0)).collect(),
            p90: (1..=horizon).map(|h| widen(h, 1.0)).collect(),
            weights: vec![],
            rmse_fused: sd,
            holdout: 0,
            method: "last".into(),
            notes,
        };
    }

    let hz = HOLDOUT.min(n / 3).max(2);
    let split = n - hz;
    let eps = (0.01 * variance(&clean)).max(1e-12);

    struct Scored {
        idx: usize,
        rmse: f64,
        eff_mse: f64,
        history: Option<(f64, u64)>,
    }
    let mut scored: Vec<Scored> = Vec::new();
    for (idx, p) in pool.iter().enumerate() {
        if split < p.min_history() || hz > p.max_horizon() {
            continue;
        }
        let band = match p.predict(&clean[..split], hz) {
            Ok(b) if b.point.len() >= hz && b.point.iter().all(|v| v.is_finite()) => b,
            Ok(_) => {
                notes.push(format!("{}: 백테스트 출력이 올바르지 않아 제외했습니다", p.name()));
                continue;
            }
            Err(e) => {
                notes.push(format!("{}: 백테스트 실패({e})", p.name()));
                continue;
            }
        };
        let mse = band.point.iter().zip(&clean[split..]).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / hz as f64;
        let hist = history.iter().find(|h| h.method == p.name() && h.n > 0 && h.mse.is_finite());
        let eff_mse = match hist {
            Some(h) => {
                let nh = (h.n as f64).min(HISTORY_CAP);
                (nh * h.mse + mse) / (nh + 1.0)
            }
            None => mse,
        };
        scored.push(Scored { idx, rmse: mse.sqrt(), eff_mse, history: hist.map(|h| (h.mse.max(0.0).sqrt(), h.n)) });
    }
    if scored.is_empty() {
        notes.push("백테스트를 통과한 예측기가 없어 마지막 값을 사용했습니다".into());
        let l = *clean.last().unwrap();
        let sd = variance(&clean).sqrt();
        return Forecast {
            point: vec![l; horizon],
            p10: vec![l - 1.2816 * sd; horizon],
            p90: vec![l + 1.2816 * sd; horizon],
            weights: vec![],
            rmse_fused: sd,
            holdout: hz,
            method: "last".into(),
            notes,
        };
    }

    struct Live {
        w: MethodWeight,
        band: Band,
    }
    let mut live: Vec<Live> = Vec::new();
    for s in &scored {
        let p = pool[s.idx];
        if horizon > p.max_horizon() {
            notes.push(format!("{}: 최대 수평선 {}을 넘어 제외했습니다", p.name(), p.max_horizon()));
            continue;
        }
        match p.predict(&clean, horizon) {
            Ok(b) if b.point.len() >= horizon && b.point.iter().all(|v| v.is_finite()) => live.push(Live {
                w: MethodWeight {
                    method: p.name().to_string(),
                    rmse: s.rmse,
                    history_rmse: s.history.map(|h| h.0),
                    history_n: s.history.map_or(0, |h| h.1),
                    weight: 0.0,
                },
                band: b,
            }),
            Ok(_) => notes.push(format!("{}: 예측 출력이 올바르지 않아 제외했습니다", p.name())),
            Err(e) => notes.push(format!("{}: 예측 실패({e})", p.name())),
        }
    }
    let eff_of = |name: &str| scored.iter().find(|s| pool[s.idx].name() == name).map(|s| s.eff_mse).unwrap_or(f64::INFINITY);
    let inv: Vec<f64> = live.iter().map(|l| 1.0 / (eff_of(&l.w.method) + eps)).collect();
    let total: f64 = inv.iter().sum::<f64>().max(1e-300);
    for (l, w) in live.iter_mut().zip(&inv) {
        l.w.weight = w / total;
    }
    let rmse_fused = live.iter().map(|l| l.w.weight * eff_of(&l.w.method)).sum::<f64>().sqrt();

    let mut point = vec![0.0; horizon];
    for l in &live {
        for (i, v) in l.band.point.iter().enumerate().take(horizon) {
            point[i] += v * l.w.weight;
        }
    }
    let band_src = live
        .iter()
        .filter(|l| l.band.lo.is_some() && l.band.hi.is_some())
        .max_by(|a, b| a.w.weight.total_cmp(&b.w.weight));
    let mut p10 = vec![0.0; horizon];
    let mut p90 = vec![0.0; horizon];
    for i in 0..horizon {
        let (lo_off, hi_off) = match band_src {
            Some(l) => (
                (l.band.point[i] - l.band.lo.as_ref().unwrap()[i]).max(0.0),
                (l.band.hi.as_ref().unwrap()[i] - l.band.point[i]).max(0.0),
            ),
            None => (0.0, 0.0),
        };
        let sig = 1.2816 * rmse_fused * (((i + 1) as f64) / hz as f64).sqrt() * std::f64::consts::SQRT_2;
        p90[i] = point[i] + (hi_off * hi_off + sig * sig).sqrt();
        p10[i] = point[i] - (lo_off * lo_off + sig * sig).sqrt();
    }
    let label = live.iter().map(|l| format!("{} {:.2}", l.w.method, l.w.weight)).collect::<Vec<_>>().join(", ");
    Forecast {
        point,
        p10,
        p90,
        weights: live.into_iter().map(|l| l.w).collect(),
        rmse_fused,
        holdout: hz,
        method: format!("fusion({label})"),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Biased {
        name: &'static str,
        bias: f64,
        min: usize,
    }
    impl Predictor for Biased {
        fn name(&self) -> &str {
            self.name
        }
        fn min_history(&self) -> usize {
            self.min
        }
        fn predict(&self, series: &[f64], horizon: usize) -> Result<Band, String> {
            let l = *series.last().ok_or("empty")?;
            Ok(Band { point: vec![l + self.bias; horizon], lo: Some(vec![l + self.bias - 0.1; horizon]), hi: Some(vec![l + self.bias + 0.1; horizon]) })
        }
    }

    struct Failing;
    impl Predictor for Failing {
        fn name(&self) -> &str {
            "failing"
        }
        fn min_history(&self) -> usize {
            1
        }
        fn predict(&self, _: &[f64], _: usize) -> Result<Band, String> {
            Err("no weights".into())
        }
    }

    fn noisy_flat(n: usize, level: f64, amp: f64) -> Vec<f64> {
        (0..n).map(|i| level + amp * (((i * 7 + 3) % 5) as f64 - 2.0) / 2.0).collect()
    }

    #[test]
    fn weights_sum_to_one_and_favour_the_accurate_method() {
        let y = noisy_flat(24, 0.6, 0.04);
        let good = Biased { name: "good", bias: 0.0, min: 3 };
        let bad = Biased { name: "bad", bias: 0.4, min: 3 };
        let f = forecast(&y, 3, &[&good, &bad], &[]);
        let sum: f64 = f.weights.iter().map(|w| w.weight).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        let wg = f.weights.iter().find(|w| w.method == "good").unwrap().weight;
        let wb = f.weights.iter().find(|w| w.method == "bad").unwrap().weight;
        assert!(wg > wb * 20.0, "good {wg} bad {wb}");
        assert!((f.point[0] - 0.6).abs() < 0.08, "{:?}", f.point);
        assert!(f.method.starts_with("fusion("));
    }

    #[test]
    fn remembered_errors_pull_the_weights() {
        let y = noisy_flat(24, 0.6, 0.04);
        let a = Biased { name: "a", bias: 0.05, min: 3 };
        let b = Biased { name: "b", bias: 0.05, min: 3 };
        let neutral = forecast(&y, 2, &[&a, &b], &[]);
        let wa0 = neutral.weights.iter().find(|w| w.method == "a").unwrap().weight;
        let wb0 = neutral.weights.iter().find(|w| w.method == "b").unwrap().weight;
        assert!((wa0 - wb0).abs() < 1e-9, "same bias gives the same weight");
        let hist = [HistoryMse { method: "a".into(), mse: 0.25, n: 8 }, HistoryMse { method: "b".into(), mse: 0.0025, n: 8 }];
        let f = forecast(&y, 2, &[&a, &b], &hist);
        let wa = f.weights.iter().find(|w| w.method == "a").unwrap();
        let wb = f.weights.iter().find(|w| w.method == "b").unwrap();
        assert!(wb.weight > wa.weight * 3.0, "{} vs {}", wb.weight, wa.weight);
        assert_eq!(wa.history_n, 8);
        assert!((wa.history_rmse.unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn history_counts_at_most_the_cap_against_the_new_run() {
        let y = noisy_flat(24, 0.6, 0.04);
        let a = Biased { name: "a", bias: 0.0, min: 3 };
        let few = [HistoryMse { method: "a".into(), mse: 1.0, n: 8 }];
        let many = [HistoryMse { method: "a".into(), mse: 1.0, n: 10_000 }];
        let f_few = forecast(&y, 1, &[&a], &few);
        let f_many = forecast(&y, 1, &[&a], &many);
        let w = |f: &Forecast| f.weights.iter().find(|w| w.method == "a").unwrap().weight;
        assert!((w(&f_few) - w(&f_many)).abs() < 1e-12, "capped at {HISTORY_CAP}");
    }

    #[test]
    fn failing_and_unready_predictors_are_skipped_with_a_note() {
        let y = noisy_flat(10, 0.5, 0.02);
        let late = Biased { name: "late", bias: 0.0, min: 40 };
        let f = forecast(&y, 2, &[&Failing, &late], &[]);
        assert!(f.weights.iter().all(|w| w.method != "failing" && w.method != "late"));
        assert!(f.notes.iter().any(|n| n.contains("failing")), "{:?}", f.notes);
        assert!(!f.weights.is_empty());
    }

    #[test]
    fn short_series_use_the_last_value_without_a_backtest() {
        let f = forecast(&[0.3, 0.5, 0.4], 3, &[], &[]);
        assert_eq!(f.method, "last");
        assert_eq!(f.point, vec![0.4; 3]);
        assert!(f.p10[0] < 0.4 && f.p90[2] > f.p90[0]);
        assert!(f.notes[0].contains("백테스트 없이"));
    }

    #[test]
    fn empty_nan_and_constant_series_do_not_break() {
        let f = forecast(&[], 2, &[], &[]);
        assert_eq!(f.point, vec![0.0; 2]);
        let f = forecast(&[f64::NAN, 0.5, f64::INFINITY, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 2, &[], &[]);
        assert!(f.point.iter().all(|v| (v - 0.5).abs() < 1e-9));
        assert!(f.notes.iter().any(|n| n.contains("비유한")));
        let f = forecast(&[0.5; 20], 3, &[], &[]);
        assert!(f.point.iter().all(|v| v.is_finite() && (v - 0.5).abs() < 1e-9));
        assert!(f.weights.iter().all(|w| w.weight.is_finite()));
        assert!(f.p10.iter().zip(&f.p90).all(|(a, b)| a <= b));
    }

    #[test]
    fn bands_are_ordered_and_widen_with_the_horizon() {
        let y = noisy_flat(30, 0.5, 0.1);
        let f = forecast(&y, 4, &[], &[]);
        for i in 0..4 {
            assert!(f.p10[i] <= f.point[i] && f.point[i] <= f.p90[i]);
        }
        assert!(f.p90[3] - f.p10[3] >= f.p90[0] - f.p10[0] - 1e-12);
        assert!(f.rmse_fused > 0.0);
    }

    #[test]
    fn a_persistent_level_shift_is_followed_by_the_fused_forecast() {
        let mut y = noisy_flat(20, 0.3, 0.02);
        y.extend(noisy_flat(10, 0.7, 0.02));
        let f = forecast(&y, 1, &[], &[]);
        assert!((f.point[0] - 0.7).abs() < 0.1, "{:?} {}", f.point, f.method);
    }

    #[test]
    fn built_in_predictors_behave() {
        let y = [0.2, 0.4, 0.6, 0.8];
        assert_eq!(LastPredictor.predict(&y, 2).unwrap().point, vec![0.8, 0.8]);
        let e = EwmaPredictor { alpha: 1.0 }.predict(&y, 1).unwrap();
        assert!((e.point[0] - 0.8).abs() < 1e-12);
        let h = HoltPredictor.predict(&y, 1).unwrap();
        assert!(h.point[0] > 0.8 && h.lo.unwrap()[0] <= h.point[0]);
        assert!(LastPredictor.predict(&[], 1).is_err());
    }
}
