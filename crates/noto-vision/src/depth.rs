//! ML track 1: depth from one keyframe.
//!
//! The model the project uses is `google/tipsv2-b14-dpt` (TIPSv2, CVPR 2026):
//! a DINOv2-style ViT-B/14 with a DPT head that predicts metric depth in
//! metres as the expectation over 256 bins between 0.001 and 10 m. Two facts
//! about it shape the rest of the ML logic:
//!
//! * it was trained on NYUv2, indoors, so its metric scale on this outdoor
//!   voxel renderer is a guess, and
//! * it cannot say anything beyond 10 m.
//!
//! That is why the decision step triangulates before trusting a range, and
//! why the SDS learns a per-scene `depth_scale` (triangulated range over the
//! model's range) instead of believing the metres. [`DepthMetrics`] compares a
//! prediction with the renderer's depth for the evaluation columns only.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::frame::{DepthKind, DepthMap, Rgb};

pub trait DepthModel: Send {
    /// `tipsv2-b14-dpt` or the name of a stand-in.
    fn mode(&self) -> &str;
    fn device(&self) -> String {
        "cpu".into()
    }
    /// Largest depth the model can report, metres (`None` when unbounded).
    fn max_depth_m(&self) -> Option<f32> {
        None
    }
    fn infer(&mut self, rgb: &Rgb) -> Result<DepthMap>;
    /// Depth of several keyframes, in input order. A model that can batch
    /// overrides this; each map must be the one [`DepthModel::infer`] gives.
    fn infer_batch(&mut self, frames: &[&Rgb]) -> Result<Vec<DepthMap>> {
        frames.iter().map(|f| self.infer(f)).collect()
    }
}

/// Standard monocular depth errors over pixels where both maps are finite and
/// the truth is within `max_m`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthMetrics {
    /// Mean |pred - true| / true.
    pub absrel: f32,
    /// Share of pixels with max(pred/true, true/pred) < 1.25.
    pub delta1: f32,
    /// Root mean square error, metres.
    pub rmse_m: f32,
    /// Median of true / pred: the scale that would have fixed the prediction.
    pub median_scale: f32,
    pub pixels: u32,
}

/// Compares `pred` with `truth` (metric, same or other size). With `align`
/// the prediction is first scaled by the median ratio (the usual protocol for
/// relative depth).
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

/// Keeps the finite values of a metric map inside `(0, max_m]` and marks the
/// rest as unknown, the way a bounded model reports them.
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
