//! Metric alignment of a monocular depth map to the ground the robot stands on.
//!
//! TIPSv2 DPT was trained indoors (NYUv2) and answers in 256 bins up to 10 m,
//! so on this outdoor renderer its metres are off by a factor that changes
//! with range. The robot does not need the model to know metres: it knows its
//! own camera. The eye height above the ground under the robot
//! ([`Pose::eye_m`]) and the camera's pitch ([`Intrinsics::pitch_deg`], the IMU
//! estimate) fix the depth at which every ray below the horizon meets the
//! ground. For a pinhole camera pitched down by θ at height h over flat ground
//! the forward depth of the row at normalised height ȳ is `Z = h / (ȳ·cos θ +
//! sin θ)`, a disparity linear in the row (standard projective geometry,
//! Hartley & Zisserman 2004; the constant-height ground prior of Tian et al.,
//! ICRA 2021). With the robot's map the ray is marched over the terrain
//! heights instead ([`ground_depth`] with a height function).
//!
//! Those expected depths are a metric reference that costs no model. The
//! model's depth on the pixels that look at open ground is fitted to them and
//! the fit is applied to the whole map:
//!
//! * [`AlignSpace::InverseAffine`] fits `1/Z_true ≈ a/Z_pred + b` (scale and
//!   shift in disparity, the MiDaS protocol: Ranftl et al., TPAMI 2022, eqs.
//!   2–4);
//! * [`AlignSpace::Scale`] fits one factor `Z_true ≈ s·Z_pred` (the median
//!   scaling of Monodepth2's evaluation, Godard et al. 2019).
//!
//! The fit is robust: RANSAC (Fischler & Bolles 1981) proposes models from
//! pairs of pixels, the best consensus is refitted by least squares and then
//! trimmed to the 80% smallest residuals and refitted once more (MiDaS's
//! trimming share). Objects standing on the ground (trees, boulders, the
//! target itself) are nearer than the ground behind them, so they fall out as
//! outliers. Pixels the model reports at its ceiling are "far/unknown", never
//! fitted and never given a depth.
//!
//! [`contact_range`] needs no depth model at all: a drop lies on the ground,
//! so the lowest row of its mask is where it touches the ground, and that
//! row's ray meets the ground at a known range. At 3 m from a 1.14 m eye a
//! 0.5° pitch error moves it by about 8 cm.

use crate::camera::{Intrinsics, Pose};
use crate::frame::{DepthMap, Mask};

/// Ground height relative to the ground under the robot, metres, at a world
/// point `(x, z)` of the robot's believed frame (its map).
pub type HeightFn<'a> = &'a dyn Fn(f32, f32) -> f32;

/// The space the depth model is fitted to the ground in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignSpace {
    /// `1/Z_true = a/Z_pred + b`: scale and shift in disparity.
    InverseAffine,
    /// `Z_true = s·Z_pred`: one metric factor.
    Scale,
}

/// Settings of the ground alignment.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlignCfg {
    pub space: AlignSpace,
    /// Ground pixels are used where the expected ground depth is below this,
    /// metres. ESTIMATE: under the 10 m ceiling of the NYU-trained head with
    /// some room for its saturation (research pack: "~9 m trusted range").
    pub max_ground_m: f32,
    /// Nearest expected ground depth used, metres (the robot's own feet and
    /// bumper are in the lowest rows). ESTIMATE.
    pub min_ground_m: f32,
    /// The model's answer at or above this share of its ceiling means
    /// "far/unknown". ESTIMATE (research pack: "~0.95·10 m").
    pub saturate_frac: f32,
    /// Largest relative residual of an inlier (in the fitted space).
    /// ESTIMATE (research pack: τ ≈ 0.05–0.10).
    pub inlier_tol: f32,
    /// Share of the smallest residuals kept in the trimmed refit (MiDaS: 0.8).
    pub trim_keep: f32,
    /// RANSAC trials: k = log(1 − p) / log(1 − wⁿ) with p = 0.99, w = 0.5 and
    /// n = 2 gives 17 (derived; w is an ESTIMATE of the ground share).
    pub trials: usize,
    /// Every `stride`-th pixel in x and y is a candidate (cost only).
    pub stride: u32,
    /// Fewest inliers for a fit to be used.
    pub min_inliers: usize,
    /// Least inlier share of the candidates for a fit to be used. ESTIMATE.
    pub min_inlier_share: f32,
    /// Model answers farther than any the ground fit was made on become
    /// "far/unknown" instead of being extrapolated (a disparity fit with a
    /// negative shift sends them towards infinity).
    #[serde(default)]
    pub clip_to_support: bool,
}

impl Default for AlignCfg {
    fn default() -> Self {
        AlignCfg {
            space: AlignSpace::InverseAffine,
            max_ground_m: 9.0,
            min_ground_m: 0.5,
            saturate_frac: 0.95,
            inlier_tol: 0.10,
            trim_keep: 0.8,
            trials: 17,
            stride: 4,
            min_inliers: 40,
            min_inlier_share: 0.30,
            clip_to_support: true,
        }
    }
}

/// The fitted correction and how well the ground agreed with it.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlignFit {
    pub space: AlignSpace,
    /// `a` of `1/Z = a/Z_pred + b`, or the factor `s` of [`AlignSpace::Scale`].
    pub a: f32,
    /// `b` of the disparity fit, 1/m (0 for [`AlignSpace::Scale`]).
    pub b: f32,
    /// Ground pixels that were candidates.
    pub candidates: u32,
    pub inliers: u32,
    pub inlier_share: f32,
    /// Median relative depth error of the inliers after the fit.
    pub median_rel_err: f32,
    /// Range of expected ground depths the inliers span, metres.
    pub span_m: (f32, f32),
    /// Model answers above this are "far/unknown" (`INFINITY`: no clip).
    #[serde(default = "no_clip")]
    pub clip_above_m: f32,
}

fn no_clip() -> f32 {
    f32::INFINITY
}

impl AlignFit {
    /// The aligned depth of one prediction, `INFINITY` where the fit puts it
    /// at or beyond the horizon.
    pub fn apply_one(&self, z_pred: f32) -> f32 {
        if !z_pred.is_finite() || z_pred <= 0.0 || z_pred > self.clip_above_m {
            return f32::INFINITY;
        }
        match self.space {
            AlignSpace::Scale => self.a * z_pred,
            AlignSpace::InverseAffine => {
                let q = self.a / z_pred + self.b;
                if q > 1e-4 {
                    1.0 / q
                } else {
                    f32::INFINITY
                }
            }
        }
    }

    /// Applies the fit to a whole map in place.
    pub fn apply(&self, map: &mut DepthMap) {
        for d in map.data.iter_mut() {
            *d = self.apply_one(*d);
        }
    }
}

/// Forward depth at which the ray through pixel `(px, py)` meets the ground,
/// metres, or `None` when it does not within `max_m` (above the horizon, or a
/// map that rises out of reach). Without a height function the ground is the
/// plane under the robot; with one, the ray is marched over it.
pub fn ground_depth(intr: &Intrinsics, pose: &Pose, px: f32, py: f32, height: Option<HeightFn>, max_m: f32) -> Option<f32> {
    let r = intr.ray(pose.yaw, px, py);
    let h = pose.eye_m;
    if h <= 0.0 {
        return None;
    }
    match height {
        None => {
            if r[1] >= -1e-6 {
                return None;
            }
            let d = h / -r[1];
            (d <= max_m).then_some(d)
        }
        Some(f) => {
            // Height of the ray above the ground at depth d, metres.
            let above = |d: f32| h + d * r[1] - f(pose.x + d * r[0], pose.z + d * r[2]);
            // March in steps short against a 0.5 m map cell, then bisect.
            let step = 0.1f32;
            let mut prev = 0.0f32;
            let mut d = step;
            while d <= max_m {
                if above(d) <= 0.0 {
                    let (mut lo, mut hi) = (prev, d);
                    for _ in 0..12 {
                        let mid = 0.5 * (lo + hi);
                        if above(mid) <= 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    return Some(hi.max(1e-3));
                }
                prev = d;
                d += step;
            }
            None
        }
    }
}

/// A small deterministic generator for the RANSAC samples (xorshift64*), so
/// the same map always gives the same fit and no world random stream is used.
struct Pick(u64);

impl Pick {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize) % n.max(1)
    }
}

/// Least squares `q ≈ a·p + b` (or `q ≈ a·p` when `scale_only`).
fn lsq(pts: &[(f32, f32)], scale_only: bool) -> Option<(f32, f32)> {
    if pts.is_empty() {
        return None;
    }
    let n = pts.len() as f64;
    let (mut sp, mut sq, mut spp, mut spq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &(p, q) in pts {
        let (p, q) = (p as f64, q as f64);
        sp += p;
        sq += q;
        spp += p * p;
        spq += p * q;
    }
    if scale_only {
        return (spp > 1e-12).then(|| ((spq / spp) as f32, 0.0));
    }
    let den = n * spp - sp * sp;
    if den.abs() < 1e-12 {
        return None;
    }
    let a = (n * spq - sp * sq) / den;
    let b = (sq - a * sp) / n;
    Some((a as f32, b as f32))
}

/// Relative residual of a point under a model, in the fitted space.
fn rel_residual(a: f32, b: f32, p: f32, q: f32) -> f32 {
    ((a * p + b) - q).abs() / q.abs().max(1e-6)
}

/// Fits the depth map to the ground and returns the fit (the map is not
/// changed). `None` when too few pixels see open ground or they disagree.
pub fn fit_ground(depth: &DepthMap, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    let ceiling = max_depth_m.map(|m| m * cfg.saturate_frac).unwrap_or(f32::INFINITY);
    let scale_only = cfg.space == AlignSpace::Scale;
    // (p, q, expected ground depth) per candidate pixel.
    let mut cand: Vec<(f32, f32, f32)> = Vec::new();
    let stride = cfg.stride.max(1);
    let (kw, kh) = (intr.width.max(1) as f32, intr.height.max(1) as f32);
    let mut y = stride / 2;
    while y < depth.height {
        let mut x = stride / 2;
        while x < depth.width {
            let z = depth.at(x, y);
            if z.is_finite() && z > 0.0 && z < ceiling {
                // Pixel centre in keyframe coordinates (the map may be smaller).
                let (px, py) = ((x as f32 + 0.5) * kw / depth.width as f32, (y as f32 + 0.5) * kh / depth.height as f32);
                if let Some(g) = ground_depth(intr, pose, px, py, height, cfg.max_ground_m) {
                    if g >= cfg.min_ground_m {
                        let (p, q) = if scale_only { (z, g) } else { (1.0 / z, 1.0 / g) };
                        cand.push((p, q, g));
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    if cand.len() < cfg.min_inliers {
        return None;
    }
    let pts: Vec<(f32, f32)> = cand.iter().map(|c| (c.0, c.1)).collect();
    let mut rng = Pick(0x9E37_79B9_7F4A_7C15 ^ cand.len() as u64);
    let mut best: Option<(usize, f32, f32)> = None;
    for _ in 0..cfg.trials.max(1) {
        let i = rng.below(pts.len());
        let model = if scale_only {
            let (p, q) = pts[i];
            (p > 1e-6).then(|| (q / p, 0.0))
        } else {
            let j = rng.below(pts.len());
            let ((p1, q1), (p2, q2)) = (pts[i], pts[j]);
            // Two pixels at nearly the same expected depth say nothing about the slope.
            if (p1 - p2).abs() < 0.05 * p1.abs().max(p2.abs()) {
                None
            } else {
                let a = (q1 - q2) / (p1 - p2);
                Some((a, q1 - a * p1))
            }
        };
        let Some((a, b)) = model else { continue };
        if a <= 0.0 {
            continue;
        }
        let n = pts.iter().filter(|&&(p, q)| rel_residual(a, b, p, q) < cfg.inlier_tol).count();
        if best.map_or(true, |(m, _, _)| n > m) {
            best = Some((n, a, b));
        }
    }
    let (_, a0, b0) = best?;
    let mut inl: Vec<(f32, f32)> = pts.iter().copied().filter(|&(p, q)| rel_residual(a0, b0, p, q) < cfg.inlier_tol).collect();
    let (mut a, mut b) = lsq(&inl, scale_only)?;
    // Trimmed refit: keep the smallest residuals of the consensus.
    inl.sort_by(|u, v| rel_residual(a, b, u.0, u.1).total_cmp(&rel_residual(a, b, v.0, v.1)));
    let keep = ((inl.len() as f32 * cfg.trim_keep).ceil() as usize).clamp(2, inl.len());
    inl.truncate(keep);
    (a, b) = lsq(&inl, scale_only)?;
    if a <= 0.0 {
        return None;
    }
    let inliers: Vec<usize> = (0..pts.len()).filter(|&i| rel_residual(a, b, pts[i].0, pts[i].1) < cfg.inlier_tol).collect();
    let share = inliers.len() as f32 / pts.len() as f32;
    if inliers.len() < cfg.min_inliers || share < cfg.min_inlier_share {
        return None;
    }
    let fit = AlignFit {
        space: cfg.space,
        a,
        b,
        candidates: pts.len() as u32,
        inliers: inliers.len() as u32,
        inlier_share: share,
        median_rel_err: 0.0,
        span_m: (0.0, 0.0),
        clip_above_m: f32::INFINITY,
    };
    // Depth error of the inliers in metres, relative.
    let mut errs: Vec<f32> = inliers
        .iter()
        .map(|&i| {
            let z_pred = if scale_only { pts[i].0 } else { 1.0 / pts[i].0 };
            let g = cand[i].2;
            (fit.apply_one(z_pred) - g).abs() / g
        })
        .collect();
    errs.sort_by(f32::total_cmp);
    let (lo, hi) = inliers.iter().fold((f32::INFINITY, 0.0f32), |(lo, hi), &i| (lo.min(cand[i].2), hi.max(cand[i].2)));
    // The farthest model answer among the inliers: the edge of the fit's support.
    let pred_max = inliers.iter().map(|&i| if scale_only { pts[i].0 } else { 1.0 / pts[i].0 }).fold(0.0f32, f32::max);
    let clip_above_m = if cfg.clip_to_support { pred_max } else { f32::INFINITY };
    Some(AlignFit { median_rel_err: errs[errs.len() / 2], span_m: (lo, hi), clip_above_m, ..fit })
}

/// Fits and applies the ground alignment; pixels at the model's ceiling
/// become far/unknown (`INFINITY`) either way. Returns the fit, `None` when
/// the ground could not be used (the map then keeps the model's metres,
/// minus the saturated pixels).
pub fn ground_align(depth: &mut DepthMap, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    let fit = fit_ground(depth, intr, pose, height, max_depth_m, cfg);
    if let Some(m) = max_depth_m {
        let ceiling = m * cfg.saturate_frac;
        for d in depth.data.iter_mut() {
            if *d >= ceiling {
                *d = f32::INFINITY;
            }
        }
    }
    if let Some(f) = &fit {
        f.apply(depth);
    }
    fit
}

/// Ground range to an object resting on the ground from the lowest row of
/// its mask, with its 1σ, metres. `pitch_sigma_deg` is how well the robot
/// knows its camera's pitch. `None` when the mask touches the bottom of the
/// frame (the contact is out of view) or its lowest row looks above the
/// horizon or past `max_m`.
pub fn contact_range(intr: &Intrinsics, pose: &Pose, mask: &Mask, height: Option<HeightFn>, pitch_sigma_deg: f32, max_m: f32) -> Option<(f32, f32)> {
    let [x0, _, x1, y1] = mask.bbox()?;
    if y1 + 1 >= mask.height {
        return None;
    }
    let (mut sx, mut n) = (0.0f32, 0u32);
    for x in x0..=x1 {
        if mask.get(x, y1) {
            sx += x as f32 + 0.5;
            n += 1;
        }
    }
    let (kw, kh) = (intr.width.max(1) as f32, intr.height.max(1) as f32);
    let px = sx / n.max(1) as f32 * kw / mask.width as f32;
    // The bottom edge of the lowest pixel row.
    let py = (y1 + 1) as f32 * kh / mask.height as f32;
    let d = ground_depth(intr, pose, px, py, height, max_m)?;
    let range = intr.ground_range(pose.yaw, px, py, d);
    // dr/dθ for a ray from height h to range r is (h² + r²)/h; one pixel of
    // the mask's row is an angle of (pixel size)/focal.
    let h = pose.eye_m.max(0.05);
    let focal = 0.5 * kw / (intr.fov_deg.to_radians() * 0.5).tan();
    let row_rad = (kh / mask.height as f32) / focal;
    let sigma = (h * h + range * range) / h * pitch_sigma_deg.to_radians().hypot(row_rad);
    Some((range, sigma))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;

    fn cam() -> Intrinsics {
        Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0 }
    }

    fn pose() -> Pose {
        Pose { x: 2.0, z: 3.0, yaw: 0.4, eye_m: 1.14 }
    }

    /// The renderer's depth of flat ground (and sky above the horizon).
    fn ground_map(intr: &Intrinsics, pose: &Pose) -> DepthMap {
        let (w, h) = (intr.width, intr.height);
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(ground_depth(intr, pose, x as f32 + 0.5, y as f32 + 0.5, None, 1e6).unwrap_or(f32::INFINITY));
            }
        }
        DepthMap { width: w, height: h, kind: DepthKind::Metric, data }
    }

    #[test]
    fn flat_ground_depth_follows_the_closed_form() {
        let (c, p) = (cam(), pose());
        // The centre row: Z = h / (ȳ cos θ + sin θ) with ȳ = 0 is h / sin 12°.
        let d = ground_depth(&c, &p, 192.0, 128.0, None, 100.0).unwrap();
        assert!((d - 1.14 / 12f32.to_radians().sin()).abs() < 1e-3, "{d}");
        // Above the horizon there is no ground.
        assert!(ground_depth(&c, &p, 192.0, 2.0, None, 100.0).is_none());
        // A map that is flat gives the same answer as the plane.
        let flat = |_: f32, _: f32| 0.0f32;
        let m = ground_depth(&c, &p, 100.0, 200.0, Some(&flat), 30.0).unwrap();
        let e = ground_depth(&c, &p, 100.0, 200.0, None, 30.0).unwrap();
        assert!((m - e).abs() < 0.01, "{m} vs {e}");
        // A step of 0.5 m starting 2 m ahead along the ray is met sooner.
        let (yaw, x0, z0) = (p.yaw, p.x, p.z);
        let step = move |x: f32, z: f32| if ((x - x0) * yaw.cos() + (z - z0) * yaw.sin()) > 2.0 { 0.5 } else { 0.0 };
        let s = ground_depth(&c, &p, 192.0, 140.0, Some(&step), 30.0).unwrap();
        let open = ground_depth(&c, &p, 192.0, 140.0, None, 30.0).unwrap();
        assert!(s < open, "{s} < {open}");
    }

    #[test]
    fn a_known_disparity_distortion_is_recovered_despite_an_object() {
        let (c, p) = (cam(), pose());
        let truth = ground_map(&c, &p);
        // The model: 1/Z_pred = (1/Z - b) / a with a = 0.7, b = 0.05, capped at 10 m like TIPS.
        let (a, b) = (0.7f32, 0.05f32);
        let mut pred = truth.clone();
        for d in pred.data.iter_mut() {
            *d = if d.is_finite() { (1.0 / ((1.0 / *d - b) / a)).clamp(1e-3, 10.0) } else { 10.0 };
        }
        // An object 2 m away covers a block of pixels: nearer than the ground behind it.
        for y in 150..200 {
            for x in 150..230 {
                pred.data[(y * 384 + x) as usize] = 1.5;
            }
        }
        let mut aligned = pred.clone();
        let fit = ground_align(&mut aligned, &c, &p, None, Some(10.0), &AlignCfg::default()).expect("fit");
        assert!((fit.a - a).abs() < 0.02 * a && (fit.b - b).abs() < 0.02, "{fit:?}");
        assert!(fit.inlier_share > 0.5 && fit.median_rel_err < 0.01, "{fit:?}");
        // Ground pixels come back to their metres within 2%.
        let i = (230 * 384 + 300) as usize;
        assert!((aligned.data[i] - truth.data[i]).abs() < 0.02 * truth.data[i], "{} vs {}", aligned.data[i], truth.data[i]);
        // Saturated pixels (sky here) are far/unknown, not 10 m.
        assert!(aligned.data[5 * 384 + 10].is_infinite());
        // Answers past the farthest ground the fit saw are not extrapolated.
        assert!(fit.clip_above_m.is_finite() && fit.apply_one(fit.clip_above_m * 1.01).is_infinite());
        let open = fit_ground(&pred, &c, &p, None, Some(10.0), &AlignCfg { clip_to_support: false, ..AlignCfg::default() }).unwrap();
        assert!(open.clip_above_m.is_infinite() && open.apply_one(fit.clip_above_m * 1.01).is_finite());
    }

    #[test]
    fn a_scale_only_fit_recovers_a_single_factor() {
        let (c, p) = (cam(), pose());
        let truth = ground_map(&c, &p);
        let mut pred = truth.clone();
        for d in pred.data.iter_mut() {
            *d = if d.is_finite() { (*d * 0.68).min(10.0) } else { 10.0 };
        }
        let cfg = AlignCfg { space: AlignSpace::Scale, ..AlignCfg::default() };
        let fit = fit_ground(&pred, &c, &p, None, Some(10.0), &cfg).unwrap();
        assert!((fit.a - 1.0 / 0.68).abs() < 0.01, "{fit:?}");
        assert_eq!(fit.b, 0.0);
    }

    #[test]
    fn no_ground_in_view_gives_no_fit() {
        let c = Intrinsics { pitch_deg: 20.0, ..cam() };
        let p = pose();
        let pred = DepthMap { width: 384, height: 256, kind: DepthKind::Metric, data: vec![4.0; 384 * 256] };
        assert!(fit_ground(&pred, &c, &p, None, Some(10.0), &AlignCfg::default()).is_none());
    }

    #[test]
    fn the_contact_row_gives_the_range_of_a_drop_without_depth() {
        let (c, p) = (cam(), pose());
        // A 0.2 m object 3 m ahead, straight on: its contact row from the camera geometry.
        let range = 3.0f32;
        let (px, py, _) = c.project(&p, p.x + range * p.yaw.cos(), -p.eye_m, p.z + range * p.yaw.sin()).unwrap();
        let mut m = Mask::empty(384, 256);
        let yb = (py - 0.5).round() as u32 - 1;
        for y in yb - 12..=yb {
            for x in (px as u32 - 8)..(px as u32 + 8) {
                m.set(x, y);
            }
        }
        let (r, s) = contact_range(&c, &p, &m, None, 0.5, 20.0).unwrap();
        assert!((r - range).abs() < 0.05, "{r}");
        // 0.5° of pitch at 3 m from 1.14 m: (h² + r²)/h · 0.0087 ≈ 8 cm, plus a pixel row.
        assert!(s > 0.07 && s < 0.12, "{s}");
        // A pitch that is wrong by 0.5° moves the answer by about that sigma.
        let tilted = Intrinsics { pitch_deg: -12.5, ..c };
        let (r2, _) = contact_range(&tilted, &p, &m, None, 0.5, 20.0).unwrap();
        assert!((r2 - r).abs() < 0.10 && (r2 - r).abs() > 0.04, "{r2} vs {r}");
        // A mask cut by the bottom of the frame has no visible contact.
        let mut low = Mask::empty(384, 256);
        low.set(100, 255);
        assert!(contact_range(&c, &p, &low, None, 0.5, 20.0).is_none());
    }
}
