//! Damped Holt trend (the statistical fallback when no TTM weights exist).
//!
//! Ported from endmill-sandbox `src/timeseries.rs` (`holt_damped`): a small
//! grid search over smoothing parameters minimising one-step squared error,
//! then a damped-trend forecast with a variance that grows with the horizon.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltFit {
    pub alpha: f64,
    pub beta: f64,
    pub phi: f64,
    pub sse: f64,
    pub sigma: f64,
}

/// Returns `(point, sd, fit)` for `horizon` steps. With fewer than three
/// points it repeats the last value with zero spread.
pub fn holt_damped(y: &[f64], horizon: usize) -> (Vec<f64>, Vec<f64>, HoltFit) {
    let n = y.len();
    if n == 0 {
        return (vec![0.0; horizon], vec![0.0; horizon], HoltFit { alpha: 0.0, beta: 0.0, phi: 1.0, sse: 0.0, sigma: 0.0 });
    }
    if n < 3 {
        let last = *y.last().unwrap();
        return (vec![last; horizon], vec![0.0; horizon], HoltFit { alpha: 1.0, beta: 0.0, phi: 1.0, sse: 0.0, sigma: 0.0 });
    }
    let mut best = HoltFit { alpha: 0.5, beta: 0.1, phi: 0.98, sse: f64::INFINITY, sigma: 0.0 };
    for ai in 1..=9 {
        let alpha = ai as f64 / 10.0;
        for &beta in [0.05, 0.1, 0.2, 0.3].iter() {
            for &phi in [0.8, 0.9, 0.95, 0.98, 1.0].iter() {
                let mut l = y[0];
                let mut b = y[1] - y[0];
                let mut sse = 0.0;
                for t in 1..n {
                    let pred = l + phi * b;
                    let e = y[t] - pred;
                    sse += e * e;
                    let l_new = alpha * y[t] + (1.0 - alpha) * (l + phi * b);
                    b = beta * (l_new - l) + (1.0 - beta) * phi * b;
                    l = l_new;
                }
                if sse < best.sse {
                    best = HoltFit { alpha, beta, phi, sse, sigma: 0.0 };
                }
            }
        }
    }
    let mut l = y[0];
    let mut b = y[1] - y[0];
    for t in 1..n {
        let l_new = best.alpha * y[t] + (1.0 - best.alpha) * (l + best.phi * b);
        b = best.beta * (l_new - l) + (1.0 - best.beta) * best.phi * b;
        l = l_new;
    }
    best.sigma = (best.sse / (n - 1).max(1) as f64).sqrt();
    let mut point = Vec::with_capacity(horizon);
    let mut sd = Vec::with_capacity(horizon);
    let mut damp = 0.0;
    let mut var_acc = 1.0;
    for h in 1..=horizon {
        if h >= 2 {
            let c = best.alpha * (1.0 + best.beta * damp);
            var_acc += c * c;
        }
        damp += best.phi.powi(h as i32);
        point.push(l + damp * b);
        sd.push(best.sigma * var_acc.sqrt());
    }
    (point, sd, best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_a_linear_trend() {
        let y: Vec<f64> = (0..20).map(|i| 1.0 + 0.5 * i as f64).collect();
        let (p, sd, fit) = holt_damped(&y, 3);
        assert!(fit.sigma < 0.05, "sigma {}", fit.sigma);
        assert!((p[0] - 11.0).abs() < 0.3, "{:?}", p);
        assert!(p[2] > p[1] && p[1] > p[0]);
        assert!(sd[2] >= sd[0]);
    }

    #[test]
    fn flat_series_stays_flat_and_short_inputs_are_safe() {
        let (p, _, _) = holt_damped(&[0.4; 12], 4);
        assert!(p.iter().all(|v| (v - 0.4).abs() < 1e-9));
        let (p, sd, _) = holt_damped(&[0.7, 0.9], 3);
        assert_eq!(p, vec![0.9; 3]);
        assert_eq!(sd, vec![0.0; 3]);
        let (p, _, _) = holt_damped(&[], 2);
        assert_eq!(p, vec![0.0; 2]);
    }

    #[test]
    fn noisy_flat_series_has_a_spread_that_grows_with_the_horizon() {
        let y: Vec<f64> = (0..30).map(|i| 0.5 + if i % 2 == 0 { 0.05 } else { -0.05 }).collect();
        let (_, sd, fit) = holt_damped(&y, 5);
        assert!(fit.sigma > 0.03);
        assert!(sd.windows(2).all(|w| w[1] >= w[0] - 1e-12));
    }
}
