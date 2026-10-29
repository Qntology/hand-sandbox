//! Granite TTM-R3 (real weights, `TTM_DIR`) next to the simple predictors on series whose truth is known: its error
//! one and four steps ahead, how much of what comes next its own q10–q90 band holds, and the weight the fused
//! forecast gives it.

use std::path::PathBuf;

use candle_core::Device;
use noto_ts::fusion::{HoltPredictor, LastPredictor, Predictor};
use noto_ts::ttm::TtmForecaster;
use noto_ts::{forecast, TtmPredictor};

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

fn series(kind: &str, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut x = 0.6;
    (0..n)
        .map(|t| {
            let e = 0.05 * rng.normal();
            x = match kind {
                "iid" => 0.6 + e,
                "ar1" => 0.6 + 0.7 * (x - 0.6) + e,
                "walk" => x + e,
                _ => 0.2 + 0.01 * t as f64 + e,
            };
            x
        })
        .collect()
}

#[test]
#[ignore = "needs TTM_DIR (Granite TTM-R3 weights)"]
fn granite_ttm_next_to_the_simple_predictors() {
    let dir = PathBuf::from(std::env::var("TTM_DIR").expect("TTM_DIR"));
    let ttm = TtmForecaster::load(&dir, &Device::Cpu).expect("TTM loads");
    let (n, h, trials) = (52usize, 4usize, 200u32);
    for kind in ["iid", "ar1", "walk", "trend"] {
        let mut rng = Rng(0x7E57 ^ kind.len() as u64);
        let names = ["ttm", "holt", "last"];
        let mut sq = [[0.0f64; 2]; 3];
        let (mut inside, mut banded, mut weight, mut fused_in) = (0u32, 0u32, 0.0f64, 0u32);
        for _ in 0..trials {
            let y = series(kind, n + h, &mut rng);
            let (past, next) = (&y[..n], &y[n..]);
            let tp = TtmPredictor(&ttm);
            let preds: [&dyn Predictor; 3] = [&tp, &HoltPredictor, &LastPredictor];
            for (k, p) in preds.iter().enumerate() {
                let b = p.predict(past, h).expect("a forecast");
                sq[k][0] += (b.point[0] - next[0]).powi(2);
                sq[k][1] += (b.point[h - 1] - next[h - 1]).powi(2);
                if k == 0 {
                    if let (Some(lo), Some(hi)) = (&b.lo, &b.hi) {
                        banded += 1;
                        inside += (next[0] >= lo[0] && next[0] <= hi[0]) as u32;
                    }
                }
            }
            let f = forecast(past, 1, &[&tp], &[]);
            weight += f.weights.iter().find(|w| w.method == "ttm").map_or(0.0, |w| w.weight);
            fused_in += (next[0] >= f.p10[0] && next[0] <= f.p90[0]) as u32;
        }
        let rmse = |k: usize, s: usize| (sq[k][s] / trials as f64).sqrt();
        let line: Vec<String> = names.iter().enumerate().map(|(k, nm)| format!("{nm} {:.4}/{:.4}", rmse(k, 0), rmse(k, 1))).collect();
        let band = if banded > 0 { format!("{:.0}%", 100.0 * inside as f64 / banded as f64) } else { "없음".into() };
        println!(
            "[TTM] {kind:5}: RMSE h1/h4 {} · TTM q10–q90 holds {band} · fused weight on TTM {:.2} · fused p10–p90 holds {:.0}%",
            line.join(" · "),
            weight / trials as f64,
            100.0 * fused_in as f64 / trials as f64
        );
    }
}
