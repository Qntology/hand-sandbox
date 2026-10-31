use crate::camera::{Intrinsics, Pose};
use crate::frame::{DepthMap, Mask, NormalMap};

pub type HeightFn<'a> = &'a dyn Fn(f32, f32) -> f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignSpace {
    InverseAffine,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlignCfg {
    pub space: AlignSpace,
    pub max_ground_m: f32,
    pub min_ground_m: f32,
    pub saturate_frac: f32,
    pub inlier_tol: f32,
    pub trim_keep: f32,
    pub trials: usize,
    pub stride: u32,
    pub min_inliers: usize,
    pub min_inlier_share: f32,
    #[serde(default)]
    pub clip_to_support: bool,
    #[serde(default = "default_normal_max_deg")]
    pub normal_max_deg: f32,
}

fn default_normal_max_deg() -> f32 {
    35.0
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
            normal_max_deg: default_normal_max_deg(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AlignFit {
    pub space: AlignSpace,
    pub a: f32,
    pub b: f32,
    pub candidates: u32,
    pub inliers: u32,
    pub inlier_share: f32,
    pub median_rel_err: f32,
    pub span_m: (f32, f32),
    #[serde(default = "no_clip")]
    pub clip_above_m: f32,
    #[serde(default)]
    pub off_ground_normals: u32,
}

fn no_clip() -> f32 {
    f32::INFINITY
}

impl AlignFit {
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

    pub fn apply(&self, map: &mut DepthMap) {
        for d in map.data.iter_mut() {
            *d = self.apply_one(*d);
        }
    }
}

/// How far a ray over uneven ground is followed (m): past any map, it has gone over the horizon.
const GROUND_RAY_MAX_M: f32 = 200.0;

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
            let above = |d: f32| h + d * r[1] - f(pose.x + d * r[0], pose.z + d * r[2]);
            let step = 0.1f32;
            let mut prev = 0.0f32;
            let mut d = step;
            while d <= max_m.min(GROUND_RAY_MAX_M) {
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

struct Pick(u64);

impl Pick {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize) % n.max(1)
    }
}

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

fn rel_residual(a: f32, b: f32, p: f32, q: f32) -> f32 {
    ((a * p + b) - q).abs() / q.abs().max(1e-6)
}

pub fn fit_ground(depth: &DepthMap, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    fit_ground_with(depth, None, intr, pose, height, max_depth_m, cfg)
}

pub fn faces_up(n: [f32; 3], up: [f32; 3], max_deg: f32) -> bool {
    n[0] * up[0] + n[1] * up[1] + n[2] * up[2] >= max_deg.to_radians().cos()
}

pub const GROUND_SEED_DEG: f32 = 60.0;

/// The ground's normal as the model sees it in this frame: the mean of the predicted normals within
/// `GROUND_SEED_DEG` of the camera's up, over the pixels whose rays meet the ground (one mean-shift step from the
/// geometric up). A model trained on other cameras tilts its normals by a near-constant angle; this absorbs it.
pub fn ground_normal(depth: &DepthMap, normals: &NormalMap, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, cfg: &AlignCfg) -> Option<[f32; 3]> {
    let up = intr.world_up_in_camera();
    let (kw, kh) = (intr.width.max(1) as f32, intr.height.max(1) as f32);
    let stride = (cfg.stride.max(1) * 2).max(2);
    let (mut s, mut n) = ([0.0f64; 3], 0u32);
    let mut y = stride / 2;
    while y < depth.height {
        let mut x = stride / 2;
        while x < depth.width {
            let (px, py) = ((x as f32 + 0.5) * kw / depth.width as f32, (y as f32 + 0.5) * kh / depth.height as f32);
            if ground_depth(intr, pose, px, py, height, cfg.max_ground_m).is_some_and(|g| g >= cfg.min_ground_m) {
                let v = normals.at_scaled(x as f32 + 0.5, y as f32 + 0.5, depth.width, depth.height);
                if faces_up(v, up, GROUND_SEED_DEG) {
                    for k in 0..3 {
                        s[k] += v[k] as f64;
                    }
                    n += 1;
                }
            }
            x += stride;
        }
        y += stride;
    }
    let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    (n >= 20 && len > 1e-6).then(|| [(s[0] / len) as f32, (s[1] / len) as f32, (s[2] / len) as f32])
}

pub fn fit_ground_with(depth: &DepthMap, normals: Option<&NormalMap>, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    let ceiling = max_depth_m.map(|m| m * cfg.saturate_frac).unwrap_or(f32::INFINITY);
    let scale_only = cfg.space == AlignSpace::Scale;
    let use_normals = normals.is_some() && cfg.normal_max_deg > 0.0 && cfg.normal_max_deg < 90.0;
    let up = normals.filter(|_| use_normals).and_then(|n| ground_normal(depth, n, intr, pose, height, cfg)).unwrap_or_else(|| intr.world_up_in_camera());
    let (mut cand, mut off): (Vec<(f32, f32, f32)>, Vec<(f32, f32, f32)>) = (Vec::new(), Vec::new());
    let stride = cfg.stride.max(1);
    let (kw, kh) = (intr.width.max(1) as f32, intr.height.max(1) as f32);
    let mut y = stride / 2;
    while y < depth.height {
        let mut x = stride / 2;
        while x < depth.width {
            let z = depth.at(x, y);
            if z.is_finite() && z > 0.0 && z < ceiling {
                let (px, py) = ((x as f32 + 0.5) * kw / depth.width as f32, (y as f32 + 0.5) * kh / depth.height as f32);
                if let Some(g) = ground_depth(intr, pose, px, py, height, cfg.max_ground_m) {
                    if g >= cfg.min_ground_m {
                        let (p, q) = if scale_only { (z, g) } else { (1.0 / z, 1.0 / g) };
                        let ground_facing = match normals.filter(|_| use_normals) {
                            Some(n) => faces_up(n.at_scaled(x as f32 + 0.5, y as f32 + 0.5, depth.width, depth.height), up, cfg.normal_max_deg),
                            None => true,
                        };
                        if ground_facing {
                            cand.push((p, q, g));
                        } else {
                            off.push((p, q, g));
                        }
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    let off_ground = if cand.len() < cfg.min_inliers {
        cand.append(&mut off);
        0
    } else {
        off.len() as u32
    };
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
        off_ground_normals: off_ground,
    };
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
    let pred_max = inliers.iter().map(|&i| if scale_only { pts[i].0 } else { 1.0 / pts[i].0 }).fold(0.0f32, f32::max);
    let clip_above_m = if cfg.clip_to_support { pred_max } else { f32::INFINITY };
    Some(AlignFit { median_rel_err: errs[errs.len() / 2], span_m: (lo, hi), clip_above_m, ..fit })
}

pub fn ground_align(depth: &mut DepthMap, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    ground_align_with(depth, None, intr, pose, height, max_depth_m, cfg)
}

pub fn ground_align_with(depth: &mut DepthMap, normals: Option<&NormalMap>, intr: &Intrinsics, pose: &Pose, height: Option<HeightFn>, max_depth_m: Option<f32>, cfg: &AlignCfg) -> Option<AlignFit> {
    let fit = fit_ground_with(depth, normals, intr, pose, height, max_depth_m, cfg);
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
    let py = (y1 + 1) as f32 * kh / mask.height as f32;
    let d = ground_depth(intr, pose, px, py, height, max_m)?;
    let range = intr.ground_range(pose.yaw, px, py, d);
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
        Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0, roll_deg: 0.0 }
    }

    fn distort(truth: &DepthMap, a: f32, b: f32) -> DepthMap {
        let mut pred = truth.clone();
        for d in pred.data.iter_mut() {
            *d = if d.is_finite() { (1.0 / ((1.0 / *d - b) / a)).clamp(1e-3, 10.0) } else { 10.0 };
        }
        pred
    }

    #[test]
    fn normals_keep_a_big_object_out_of_the_ground_fit() {
        let (c, p) = (Intrinsics { pitch_deg: -40.0, ..cam() }, pose());
        let truth = ground_map(&c, &p);
        let (a, b) = (0.7f32, 0.05f32);
        let mut pred = distort(&truth, a, b);
        let up = c.world_up_in_camera();
        let tilted = |deg: f32| {
            let (s, k) = deg.to_radians().sin_cos();
            [up[0], up[1] * k - up[2] * s, up[1] * s + up[2] * k]
        };
        let face = [0.0, -40f32.to_radians().sin(), 40f32.to_radians().cos()];
        let mut normals = NormalMap { width: 384, height: 256, data: vec![tilted(25.0); 384 * 256] };
        for y in 40..256u32 {
            for x in 30..300u32 {
                let i = (y * 384 + x) as usize;
                pred.data[i] = (pred.data[i] * 0.55).max(0.3);
                normals.data[i] = face;
            }
        }
        let seen_up = ground_normal(&pred, &normals, &c, &p, None, &AlignCfg::default()).unwrap();
        let bias = (seen_up[0] * up[0] + seen_up[1] * up[1] + seen_up[2] * up[2]).acos().to_degrees();
        assert!((bias - 25.0).abs() < 1.0, "the model's own ground normal is found, tilt and all: {bias}");
        let cfg = AlignCfg::default();
        let blind = fit_ground(&pred, &c, &p, None, Some(10.0), &cfg);
        let seen = fit_ground_with(&pred, Some(&normals), &c, &p, None, Some(10.0), &cfg).expect("the ground beside the object is enough");
        assert!((seen.a - a).abs() < 0.03 * a && (seen.b - b).abs() < 0.02, "{seen:?}");
        assert!(seen.off_ground_normals > 1000, "{seen:?}");
        let blind_err = blind.map_or(f32::INFINITY, |f| (f.a - a).abs() / a);
        assert!(blind_err > 0.05, "without normals the object's own (wrong) depth wins the fit: {blind:?}");
        let none_up = NormalMap { width: 384, height: 256, data: vec![[1.0, 0.0, 0.0]; 384 * 256] };
        assert!(fit_ground_with(&distort(&truth, a, b), Some(&none_up), &c, &p, None, Some(10.0), &cfg).is_some(), "too few ground normals falls back to all candidates");
    }

    #[test]
    fn a_rolled_chassis_is_aligned_with_its_roll() {
        let rolled = Intrinsics { roll_deg: 6.0, ..cam() };
        let p = pose();
        let truth = ground_map(&rolled, &p);
        let (a, b) = (0.8f32, 0.03f32);
        let pred = distort(&truth, a, b);
        let cfg = AlignCfg::default();
        let right = fit_ground(&pred, &rolled, &p, None, Some(10.0), &cfg).unwrap();
        assert!((right.a - a).abs() < 0.02 * a && right.median_rel_err < 0.01, "{right:?}");
        let level = fit_ground(&pred, &cam(), &p, None, Some(10.0), &cfg);
        assert!(level.map_or(true, |f| f.median_rel_err > 2.0 * right.median_rel_err.max(0.005)), "ignoring the roll fits worse: {level:?} vs {right:?}");
    }

    fn pose() -> Pose {
        Pose { x: 2.0, z: 3.0, yaw: 0.4, eye_m: 1.14 }
    }

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
        let d = ground_depth(&c, &p, 192.0, 128.0, None, 100.0).unwrap();
        assert!((d - 1.14 / 12f32.to_radians().sin()).abs() < 1e-3, "{d}");
        assert!(ground_depth(&c, &p, 192.0, 2.0, None, 100.0).is_none());
        let flat = |_: f32, _: f32| 0.0f32;
        let m = ground_depth(&c, &p, 100.0, 200.0, Some(&flat), 30.0).unwrap();
        let e = ground_depth(&c, &p, 100.0, 200.0, None, 30.0).unwrap();
        assert!((m - e).abs() < 0.01, "{m} vs {e}");
        assert!(ground_depth(&c, &p, 192.0, 2.0, Some(&flat), f32::INFINITY).is_none(), "a ray over the horizon ends");
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
        let (a, b) = (0.7f32, 0.05f32);
        let mut pred = truth.clone();
        for d in pred.data.iter_mut() {
            *d = if d.is_finite() { (1.0 / ((1.0 / *d - b) / a)).clamp(1e-3, 10.0) } else { 10.0 };
        }
        for y in 150..200 {
            for x in 150..230 {
                pred.data[(y * 384 + x) as usize] = 1.5;
            }
        }
        let mut aligned = pred.clone();
        let fit = ground_align(&mut aligned, &c, &p, None, Some(10.0), &AlignCfg::default()).expect("fit");
        assert!((fit.a - a).abs() < 0.02 * a && (fit.b - b).abs() < 0.02, "{fit:?}");
        assert!(fit.inlier_share > 0.5 && fit.median_rel_err < 0.01, "{fit:?}");
        let i = (230 * 384 + 300) as usize;
        assert!((aligned.data[i] - truth.data[i]).abs() < 0.02 * truth.data[i], "{} vs {}", aligned.data[i], truth.data[i]);
        assert!(aligned.data[5 * 384 + 10].is_infinite());
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
        assert!(s > 0.07 && s < 0.12, "{s}");
        let tilted = Intrinsics { pitch_deg: -12.5, ..c };
        let (r2, _) = contact_range(&tilted, &p, &m, None, 0.5, 20.0).unwrap();
        assert!((r2 - r).abs() < 0.10 && (r2 - r).abs() > 0.04, "{r2} vs {r}");
        let mut low = Mask::empty(384, 256);
        low.set(100, 255);
        assert!(contact_range(&c, &p, &low, None, 0.5, 20.0).is_none());
    }
}
