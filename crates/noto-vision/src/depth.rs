use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::frame::{DepthKind, DepthMap, Rgb};

pub trait DepthModel: Send {
    fn mode(&self) -> &str;
    fn device(&self) -> String {
        "cpu".into()
    }
    fn max_depth_m(&self) -> Option<f32> {
        None
    }
    fn infer(&mut self, rgb: &Rgb) -> Result<DepthMap>;
    fn infer_batch(&mut self, frames: &[&Rgb]) -> Result<Vec<DepthMap>> {
        frames.iter().map(|f| self.infer(f)).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthMetrics {
    pub absrel: f32,
    pub delta1: f32,
    pub rmse_m: f32,
    pub median_scale: f32,
    pub pixels: u32,
}

pub fn depth_metrics(pred: &DepthMap, truth: &DepthMap, max_m: f32, align: bool) -> Option<DepthMetrics> {
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    for y in 0..truth.height {
        for x in 0..truth.width {
            let t = truth.at(x, y);
            if !t.is_finite() || t <= 0.0 || t > max_m {
                continue;
            }
            let p = pred.at_scaled(x as f32 + 0.5, y as f32 + 0.5, truth.width, truth.height);
            if p.is_finite() && p > 0.0 {
                pairs.push((p, t));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    let mut ratios: Vec<f32> = pairs.iter().map(|(p, t)| t / p).collect();
    ratios.sort_by(f32::total_cmp);
    let median_scale = ratios[ratios.len() / 2];
    let s = if align { median_scale } else { 1.0 };
    let (mut ar, mut d1, mut se) = (0.0f64, 0u32, 0.0f64);
    for (p, t) in &pairs {
        let p = p * s;
        ar += ((p - t).abs() / t) as f64;
        d1 += ((p / t).max(t / p) < 1.25) as u32;
        se += ((p - t) as f64).powi(2);
    }
    let n = pairs.len() as f64;
    Some(DepthMetrics { absrel: (ar / n) as f32, delta1: (d1 as f64 / n) as f32, rmse_m: (se / n).sqrt() as f32, median_scale, pixels: pairs.len() as u32 })
}

pub fn clamp_metric(map: &mut DepthMap, max_m: f32) {
    map.kind = DepthKind::Metric;
    for d in map.data.iter_mut() {
        if d.is_finite() {
            *d = d.clamp(1e-3, max_m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(v: Vec<f32>) -> DepthMap {
        DepthMap { width: v.len() as u32, height: 1, kind: DepthKind::Metric, data: v }
    }

    #[test]
    fn a_scaled_prediction_is_exact_after_alignment() {
        let truth = map(vec![2.0, 4.0, 8.0, f32::INFINITY]);
        let pred = map(vec![1.0, 2.0, 4.0, 3.0]);
        let raw = depth_metrics(&pred, &truth, 20.0, false).unwrap();
        assert!((raw.absrel - 0.5).abs() < 1e-6);
        assert_eq!(raw.delta1, 0.0);
        assert!((raw.median_scale - 2.0).abs() < 1e-6);
        let al = depth_metrics(&pred, &truth, 20.0, true).unwrap();
        assert!(al.absrel < 1e-6 && (al.delta1 - 1.0).abs() < 1e-6);
        assert_eq!(al.pixels, 3, "sky is not scored");
    }
}
