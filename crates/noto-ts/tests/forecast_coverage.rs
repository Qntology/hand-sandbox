//! How well the fused forecast's band covers what happens next, on series whose truth is known. The p10–p90 band
//! should hold about 80 % of the next values; it holds more (it adds the best model's own band to the backtest's
//! error and widens like a random walk), which the gates and the decider, reading one step ahead, can live with. The
//! check keeps it from becoming too narrow.

use noto_ts::{forecast, holt_damped};

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

/// `n` + `h` values of one kind of series: "iid" noise on a level, an "ar1" process, a "walk", a "trend".
fn series(kind: &str, n: usize, h: usize, rng: &mut Rng) -> Vec<f64> {
    let sd = 0.05;
    let mut y = Vec::with_capacity(n + h);
    let mut x = 0.6;
    for t in 0..n + h {
        let e = sd * rng.normal();
        x = match kind {
            "iid" => 0.6 + e,
            "ar1" => 0.6 + 0.7 * (x - 0.6) + e,
            "walk" => x + e,
            _ => 0.2 + 0.01 * t as f64 + e,
        };
        y.push(x);
    }
    y
}

/// The share of the next `h` values inside the band, step by step, over `trials` series.
fn coverage(kind: &str, n: usize, h: usize, trials: u32) -> Vec<f64> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let mut inside = vec![0u32; h];
    for _ in 0..trials {
        let y = series(kind, n, h, &mut rng);
        let f = forecast(&y[..n], h, &[], &[]);
        for i in 0..h {
            inside[i] += (y[n + i] >= f.p10[i] && y[n + i] <= f.p90[i]) as u32;
        }
    }
    inside.iter().map(|c| *c as f64 / trials as f64).collect()
}

#[test]
fn the_band_holds_at_least_its_eighty_percent_one_step_ahead() {
    for kind in ["iid", "ar1", "walk", "trend"] {
        for n in [12usize, 24, 52] {
            let c = coverage(kind, n, 4, 400);
            println!("[TS] {kind:5} n={n:2}: p10–p90 holds {}", c.iter().enumerate().map(|(i, v)| format!("h{} {:.0}%", i + 1, v * 100.0)).collect::<Vec<_>>().join(" · "));
            assert!(c[0] >= 0.78 && c[0] <= 0.95, "{kind} n={n}: one step ahead the band holds {:.0} %", c[0] * 100.0);
        }
    }
}

#[test]
fn holts_spread_grows_like_its_errors() {
    let mut rng = Rng(7);
    let (n, h, trials) = (40usize, 6usize, 400);
    let (mut said, mut sq) = (vec![0.0f64; h], vec![0.0f64; h]);
    for _ in 0..trials {
        let y = series("walk", n, h, &mut rng);
        let (p, sd, _) = holt_damped(&y[..n], h);
        for i in 0..h {
            said[i] += sd[i] * sd[i];
            sq[i] += (y[n + i] - p[i]).powi(2);
        }
    }
    let ratio: Vec<f64> = (0..h).map(|i| (said[i] / sq[i]).sqrt()).collect();
    println!("[TS] holt on a walk: its sd over the error it makes, h1..h{h}: {}", ratio.iter().map(|r| format!("{r:.2}")).collect::<Vec<_>>().join(" "));
    assert!(ratio.iter().all(|r| (0.85..=1.15).contains(r)), "{ratio:?}");
}
