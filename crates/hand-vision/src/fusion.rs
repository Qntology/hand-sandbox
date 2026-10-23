//! Features that mix the ML tracks, one set per keyframe.
//!
//! Each track can look fine on its own and still disagree with the others.
//! These features say where they disagree, so the SDS `fusion` track can learn
//! what normal disagreement looks like for each label and flag the keyframes
//! that leave it:
//!
//! * `peak_in_mask`: the semantic peak (SigLIP2) lies inside the tracked mask;
//! * `peak_depth_gap`: depth at the peak against the mask's median depth
//!   (a peak on the background behind the object has a much larger depth);
//! * `heat_in_mask`: share of the label's heat above half its peak that falls
//!   inside the mask (the heat map and the mask describe the same object);
//! * `range_gap`: depth-model range times the learned scale against the
//!   triangulated range;
//! * `motion_residual_deg`: what the decision predicted against what the
//!   tracker saw after the move;
//! * `head_tilt_deg`: residual camera tilt from the balance model (body
//!   motion that reaches the camera makes the tracker's job harder).

use serde::{Deserialize, Serialize};

use crate::frame::{DepthMap, Mask};
use crate::locate::SemanticPoint;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Features {
    pub peak_in_mask: Option<bool>,
    pub peak_depth_gap: Option<f32>,
    pub heat_in_mask: Option<f32>,
    pub range_gap: Option<f32>,
    pub motion_residual_deg: Option<f32>,
    pub head_tilt_deg: Option<f32>,
}

/// Thresholds for the flags. They are starting values; the SDS baselines of
/// the same features are what the runtime compares against once it has them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlagCfg {
    pub depth_gap: f32,
    pub heat_in_mask: f32,
    pub range_gap: f32,
    pub residual_deg: f32,
    pub head_tilt_deg: f32,
}

impl Default for FlagCfg {
    fn default() -> Self {
        FlagCfg { depth_gap: 0.25, heat_in_mask: 0.25, range_gap: 0.25, residual_deg: 8.0, head_tilt_deg: 5.0 }
    }
}

pub struct FusionInput<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: Option<&'a DepthMap>,
    pub point: Option<&'a SemanticPoint>,
    pub mask: Option<&'a Mask>,
    pub depth_range_m: Option<f32>,
    pub depth_scale: Option<f32>,
    pub tri_range_m: Option<f32>,
    pub residual_deg: Option<f32>,
    pub head_tilt_deg: Option<f32>,
}

pub fn features(inp: &FusionInput<'_>) -> Features {
    let mut f = Features { motion_residual_deg: inp.residual_deg, head_tilt_deg: inp.head_tilt_deg, ..Default::default() };
    if let (Some(p), Some(m)) = (inp.point, inp.mask) {
        let to_mask = |x: f32, y: f32| -> Option<(u32, u32)> {
            let mx = (x / inp.width.max(1) as f32 * m.width as f32) as i64;
            let my = (y / inp.height.max(1) as f32 * m.height as f32) as i64;
            (mx >= 0 && my >= 0 && mx < m.width as i64 && my < m.height as i64).then_some((mx as u32, my as u32))
        };
        f.peak_in_mask = Some(to_mask(p.px, p.py).map_or(false, |(x, y)| m.get(x, y)));
        // Heat cells above half the peak, sampled at their centres.
        let h = &p.heat;
        let peak = h.values.iter().copied().fold(0.0f32, f32::max);
        let (mut inside, mut total) = (0.0f32, 0.0f32);
        for r in 0..h.rows {
            for c in 0..h.cols {
                let v = h.at(c, r);
                if v < 0.5 * peak {
                    continue;
                }
                let x = (c as f32 + 0.5) / h.cols as f32 * inp.width as f32;
                let y = (r as f32 + 0.5) / h.rows as f32 * inp.height as f32;
                total += v;
                if to_mask(x, y).map_or(false, |(mx, my)| m.get(mx, my)) {
                    inside += v;
                }
            }
        }
        if total > 0.0 {
            f.heat_in_mask = Some(inside / total);
        }
        if let Some(d) = inp.depth {
            let at_peak = d.at_scaled(p.px, p.py, inp.width, inp.height);
            if let Some(med) = d.median_in(m) {
                if at_peak.is_finite() && med > 0.0 {
                    f.peak_depth_gap = Some((at_peak - med).abs() / med);
                }
            }
        }
    }
    if let (Some(r), Some(t)) = (inp.depth_range_m, inp.tri_range_m) {
        if t > 0.1 {
            f.range_gap = Some((r * inp.depth_scale.unwrap_or(1.0) - t).abs() / t);
        }
    }
    f
}

/// Flag codes raised by fixed thresholds.
pub fn flags(f: &Features, cfg: &FlagCfg) -> Vec<&'static str> {
    let mut v = Vec::new();
    if f.peak_in_mask == Some(false) {
        v.push("peak_off_mask");
    }
    if f.peak_depth_gap.map_or(false, |g| g > cfg.depth_gap) {
        v.push("depth_edge");
    }
    if f.heat_in_mask.map_or(false, |h| h < cfg.heat_in_mask) {
        v.push("heat_spill");
    }
    if f.range_gap.map_or(false, |g| g > cfg.range_gap) {
        v.push("range_disagree");
    }
    if f.motion_residual_deg.map_or(false, |r| r.abs() > cfg.residual_deg) {
        v.push("motion_residual");
    }
    if f.head_tilt_deg.map_or(false, |t| t.abs() > cfg.head_tilt_deg) {
        v.push("head_tilt");
    }
    v
}

/// Named numeric features, for SDS baselines (`fusion|<label>|`).
pub fn numeric(f: &Features) -> Vec<(&'static str, f64)> {
    let mut v = Vec::new();
    if let Some(b) = f.peak_in_mask {
        v.push(("peak_in_mask", if b { 1.0 } else { 0.0 }));
    }
    for (k, x) in [("peak_depth_gap", f.peak_depth_gap), ("heat_in_mask", f.heat_in_mask), ("range_gap", f.range_gap), ("motion_residual_deg", f.motion_residual_deg), ("head_tilt_deg", f.head_tilt_deg)] {
        if let Some(x) = x {
            v.push((k, x as f64));
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;
    use crate::locate::Heat;

    #[test]
    fn agreeing_tracks_raise_no_flags_and_a_background_peak_does() {
        let mut m = Mask::empty(8, 4);
        for x in 2..5 {
            for y in 1..3 {
                m.set(x, y);
            }
        }
        let mut values = vec![0.0; 32];
        values[1 * 8 + 3] = 0.9;
        let heat = Heat { cols: 8, rows: 4, values };
        let depth = DepthMap { width: 8, height: 4, kind: DepthKind::Metric, data: (0..32).map(|i| if m.bits[i] { 4.0 } else { 9.0 }).collect() };
        let on = SemanticPoint { label: "stone".into(), px: 3.5, py: 1.5, score: 0.9, margin: 0.5, presence: 1.0, bbox: [3.0, 1.0, 4.0, 2.0], heat: heat.clone() };
        let inp = FusionInput { width: 8, height: 4, depth: Some(&depth), point: Some(&on), mask: Some(&m), depth_range_m: Some(5.0), depth_scale: Some(1.2), tri_range_m: Some(6.0), residual_deg: Some(1.0), head_tilt_deg: Some(0.5) };
        let f = features(&inp);
        assert_eq!(f.peak_in_mask, Some(true));
        assert_eq!(f.heat_in_mask, Some(1.0));
        assert!(f.range_gap.unwrap() < 1e-6);
        assert!(flags(&f, &FlagCfg::default()).is_empty(), "{f:?}");
        let off = SemanticPoint { px: 7.5, py: 3.5, ..on.clone() };
        let f2 = features(&FusionInput { point: Some(&off), ..inp });
        let fl = flags(&f2, &FlagCfg::default());
        assert!(fl.contains(&"peak_off_mask") && fl.contains(&"depth_edge"), "{fl:?}");
        assert_eq!(numeric(&f2).len(), 6);
    }
}
