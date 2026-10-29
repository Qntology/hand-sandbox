//! Stereo depth on a rectified left/right pair by semi-global block matching (SGBM): the L2 slot of the 1 : 2 : 4
//! pyramid when the robot's eye is a stereo pair instead of one camera. It triangulates, Z = f·b/d, so its depth is
//! metric as it comes out and needs no ground alignment (`align`, the monocular model's 1/Z = a/Z + b), and every
//! pixel it cannot match with confidence — no texture (a plain board, the sky), a highlight, a strip one eye cannot
//! see, an ambiguous repeat — stays empty (NaN) instead of being filled with a guess.
//!
//! Matching runs on a grid that is a whole multiple of the pyramid: `per_l2` matching pixels per L2 patch along each
//! axis, so every SigLIP2 patch and every L2 patch is a whole block of depth pixels (a 384 × 256 frame: 18 × 12 :
//! 36 × 24 : 288 × 192 with `per_l2` 8, which is the SAM mask's own grid).
//!
//! Steps (Hirschmüller 2008, with a census cost): both images to grey on that grid; 5 × 5 census of each; matching
//! cost = Hamming distance summed over a small block; the cost aggregated along 8 paths with the P1/P2 smoothness
//! penalties; winner-take-all disparity with a uniqueness ratio; parabola sub-pixel; left–right consistency from the
//! same aggregated costs; small speckles removed.

use std::time::Instant;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::depth::DepthModel;
use crate::frame::{DepthKind, DepthMap, Rgb};
use crate::pyramid::{Pyramid, PyramidCfg};

pub const MODE: &str = "sgbm-stereo";

/// A calibrated, rectified pair: the right camera sits `baseline_m` along the left camera's right axis, with the same
/// orientation and intrinsics. The left camera is the reference; the semantic layer and the masks read its image.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StereoRig {
    /// Focal length in pixels at the frame's full width.
    pub focal_px: f32,
    pub baseline_m: f32,
    /// The matcher's disparity error, px RMS on its matching grid: the sub-pixel term of σ_Z = Z²·σ_d / (f·b).
    pub subpixel_px: f32,
}

impl StereoRig {
    /// The focal length on a matching grid `grid_w` pixels wide, for a frame `frame_w` wide.
    pub fn focal_on(&self, frame_w: u32, grid_w: u32) -> f32 {
        self.focal_px * grid_w as f32 / frame_w.max(1) as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SgbmCfg {
    /// Matching pixels per L2 patch along each axis, a power of two; halved until the grid fits in the frame.
    pub per_l2: u32,
    /// The nearest depth matched (m): it sets the disparity range, f·b / `min_depth_m`.
    pub min_depth_m: f32,
    /// A confident match farther than this (m) reads as infinitely far: real, but beyond what the rig can range.
    pub max_depth_m: f32,
    /// Half size of the block the per-pixel census costs are summed over (1: 3 × 3).
    pub block: u32,
    /// 4 (left, right, up, down) or 8 (and the diagonals) aggregation paths.
    pub paths: u8,
    /// Penalty for a one-pixel disparity step between neighbours (slanted surfaces), in census bits.
    pub p1: u16,
    /// Penalty for a larger step (an edge between surfaces), in census bits.
    pub p2: u16,
    /// The best aggregated cost must beat every disparity more than one away by this share (OpenCV's
    /// uniquenessRatio / 100).
    pub uniqueness: f32,
    /// Largest disagreement (px) between the left image's disparity and the right image's at the matched pixel.
    pub lr_max_diff: f32,
    /// Least grey-level standard deviation in the census window: below it there is nothing to match.
    pub min_texture: f32,
    /// Grey level from which a pixel is blown out (a specular highlight) in either image.
    pub saturated: u8,
    /// Connected regions of fewer pixels than this, whose neighbours lie within `speckle_range` px, are dropped.
    pub speckle_px: u32,
    pub speckle_range: f32,
    #[serde(default)]
    pub subpixel: SubpixelFit,
    /// Matches whose depth error σ_Z / Z (= Z·σ_d / (f·b), the rig's `subpixel_px` as σ_d) would pass this read as
    /// beyond range (infinity): the disparity there is a pixel or two, and a fraction of a pixel moves it metres.
    pub max_rel_sigma: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubpixelFit {
    Parabola,
    #[default]
    Equiangular,
}

impl Default for SgbmCfg {
    fn default() -> Self {
        SgbmCfg {
            per_l2: 8,
            min_depth_m: 0.25,
            max_depth_m: 12.0,
            block: 1,
            paths: 8,
            p1: 10,
            p2: 120,
            uniqueness: 0.10,
            lr_max_diff: 1.0,
            min_texture: 2.0,
            saturated: 250,
            speckle_px: 24,
            speckle_range: 1.0,
            subpixel: SubpixelFit::Equiangular,
            max_rel_sigma: 0.25,
        }
    }
}

/// Why pixels hold no depth, and how much of the grid does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StereoStats {
    pub width: u32,
    pub height: u32,
    pub disparities: u32,
    /// The focal length on the matching grid (px), for σ_Z.
    pub focal_px: f32,
    pub baseline_m: f32,
    pub valid: u32,
    /// Matched, but farther than `max_depth_m`: infinity, not unknown.
    pub far: u32,
    pub border: u32,
    pub saturated: u32,
    pub no_texture: u32,
    pub not_unique: u32,
    pub lr_mismatch: u32,
    pub speckle: u32,
    pub ms: f32,
}

impl StereoStats {
    pub fn pixels(&self) -> u32 {
        self.width * self.height
    }

    /// The share of the grid with a finite depth.
    pub fn valid_frac(&self) -> f32 {
        self.valid as f32 / self.pixels().max(1) as f32
    }
}

pub struct StereoOut {
    pub depth: DepthMap,
    /// Disparity on the matching grid (px), NaN where rejected.
    pub disparity: Vec<f32>,
    pub stats: StereoStats,
}

/// The matching grid for a `width` × `height` frame: the pyramid's L2 grid times the largest power of two up to
/// `per_l2` that still fits in the frame (no upsampling).
pub fn stereo_grid(width: u32, height: u32, pyramid: Option<&PyramidCfg>, per_l2: u32) -> (u32, u32) {
    let cfg = pyramid.copied().unwrap_or_default();
    let (l2w, l2h) = Pyramid::for_frame(width, height, &cfg).l2();
    let mut k = per_l2.max(1).next_power_of_two();
    if k > per_l2.max(1) {
        k /= 2;
    }
    while k > 1 && (l2w * k > width.max(1) || l2h * k > height.max(1)) {
        k /= 2;
    }
    (l2w * k, l2h * k)
}

/// For each of `dst` cells over `src` pixels: the source pixels it covers and their overlap weights (summing to 1).
fn area_weights(src: u32, dst: u32) -> Vec<Vec<(u32, f32)>> {
    let s = src as f32 / dst.max(1) as f32;
    (0..dst)
        .map(|i| {
            let (a, b) = (i as f32 * s, (i + 1) as f32 * s);
            let mut v = Vec::new();
            let mut j = a.floor() as u32;
            while (j as f32) < b && j < src {
                let w = ((j + 1) as f32).min(b) - (j as f32).max(a);
                if w > 1e-6 {
                    v.push((j, w / s));
                }
                j += 1;
            }
            if v.is_empty() {
                v.push(((a as u32).min(src - 1), 1.0));
            }
            v
        })
        .collect()
}

/// Grey (Rec. 601 luma) on the matching grid, averaged over the area each grid pixel covers, and the brightest
/// covered pixel (for the highlight test).
fn grey_on(rgb: &Rgb, gw: u32, gh: u32) -> (Vec<f32>, Vec<u8>) {
    let (w, h) = (rgb.width, rgb.height);
    let luma = |x: u32, y: u32| {
        let p = rgb.px(x, y);
        (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32, p[0].max(p[1]).max(p[2]))
    };
    if (gw, gh) == (w, h) {
        let mut g = Vec::with_capacity((w * h) as usize);
        let mut m = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let (l, b) = luma(x, y);
                g.push(l);
                m.push(b);
            }
        }
        return (g, m);
    }
    let (wx, wy) = (area_weights(w, gw), area_weights(h, gh));
    let mut g = vec![0.0f32; (gw * gh) as usize];
    let mut m = vec![0u8; (gw * gh) as usize];
    for (gy, ys) in wy.iter().enumerate() {
        for (gx, xs) in wx.iter().enumerate() {
            let (mut acc, mut hi) = (0.0f32, 0u8);
            for &(sy, ky) in ys {
                for &(sx, kx) in xs {
                    let (l, b) = luma(sx, sy);
                    acc += l * kx * ky;
                    hi = hi.max(b);
                }
            }
            g[gy * gw as usize + gx] = acc;
            m[gy * gw as usize + gx] = hi;
        }
    }
    (g, m)
}

const CENSUS_R: i64 = 2;
const CENSUS_BITS: u16 = 24;

/// 5 × 5 census (a bit per neighbour darker than the centre; the border repeats the edge pixels) and the grey-level
/// standard deviation over the same window.
fn census(g: &[f32], w: usize, h: usize) -> (Vec<u32>, Vec<f32>) {
    let mut code = vec![0u32; w * h];
    let mut sd = vec![0.0f32; w * h];
    let at = |x: i64, y: i64| g[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let c = at(x, y);
            let (mut bits, mut s, mut ss) = (0u32, 0.0f32, 0.0f32);
            for dy in -CENSUS_R..=CENSUS_R {
                for dx in -CENSUS_R..=CENSUS_R {
                    let v = at(x + dx, y + dy);
                    s += v;
                    ss += v * v;
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    bits = (bits << 1) | (v < c) as u32;
                }
            }
            let n = ((2 * CENSUS_R + 1) * (2 * CENSUS_R + 1)) as f32;
            let i = y as usize * w + x as usize;
            code[i] = bits;
            sd[i] = (ss / n - (s / n).powi(2)).max(0.0).sqrt();
        }
    }
    (code, sd)
}

/// Matching cost C(x, y, d): the census Hamming distance between the left pixel and the right pixel d to its left,
/// summed over a (2·block + 1)² block. A disparity that would reach past the right image's left edge says nothing:
/// it costs what the pixel's best one does. (Costing it the most instead leaves a penalty that the paths from the
/// left edge carry across the whole row, and that alone then picks among the equal matches of a repeating pattern.)
fn costs(cl: &[u32], cr: &[u32], w: usize, h: usize, nd: usize, block: usize) -> Vec<u16> {
    let mut ham = vec![0u16; w * h * nd];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * nd;
            let a = cl[y * w + x];
            let top = nd.min(x + 1);
            let mut least = CENSUS_BITS;
            for d in 0..top {
                let c = (a ^ cr[y * w + x - d]).count_ones() as u16;
                ham[base + d] = c;
                least = least.min(c);
            }
            for d in top..nd {
                ham[base + d] = least;
            }
        }
    }
    if block == 0 {
        return ham;
    }
    // Box sum, rows then columns (the border repeats the edge).
    let mut rows = vec![0u16; w * h * nd];
    for y in 0..h {
        for x in 0..w {
            let out = (y * w + x) * nd;
            for k in 0..=2 * block {
                let xx = (x + k).saturating_sub(block).min(w - 1);
                let src = (y * w + xx) * nd;
                for d in 0..nd {
                    rows[out + d] += ham[src + d];
                }
            }
        }
    }
    let mut out = vec![0u16; w * h * nd];
    for y in 0..h {
        for k in 0..=2 * block {
            let yy = (y + k).saturating_sub(block).min(h - 1);
            for x in 0..w {
                let (o, s) = ((y * w + x) * nd, (yy * w + x) * nd);
                for d in 0..nd {
                    out[o + d] += rows[s + d];
                }
            }
        }
    }
    out
}

/// One step along an aggregation path: L(p, d) = C(p, d) + min(L(p−r, d), L(p−r, d±1) + P1, min L(p−r) + P2) −
/// min L(p−r). Writes L(p, ·) into `out` and returns its minimum.
#[inline]
fn path_step(c: &[u16], prev: Option<(&[u16], u16)>, p1: u16, p2: u16, out: &mut [u16]) -> u16 {
    let nd = c.len();
    match prev {
        None => {
            out.copy_from_slice(c);
            c.iter().copied().min().unwrap_or(0)
        }
        Some((lp, mp)) => {
            let jump = mp + p2;
            let mut m = u16::MAX;
            for d in 0..nd {
                let mut best = lp[d].min(jump);
                if d > 0 {
                    best = best.min(lp[d - 1] + p1);
                }
                if d + 1 < nd {
                    best = best.min(lp[d + 1] + p1);
                }
                let v = c[d] + best - mp;
                out[d] = v;
                m = m.min(v);
            }
            m
        }
    }
}

/// S(p, d) = Σ_r L_r(p, d) over 4 or 8 paths, in two sweeps: rows top to bottom with the paths from the left, the
/// top-left, the top and the top-right; then rows bottom to top with the four opposite ones.
fn aggregate(cost: &[u16], w: usize, h: usize, nd: usize, cfg: &SgbmCfg) -> Vec<u16> {
    let (p1, p2) = (cfg.p1, cfg.p2.max(cfg.p1 + 1));
    let diagonals = cfg.paths >= 8;
    let mut sum = vec![0u16; w * h * nd];
    let mut tmp = vec![0u16; nd];
    for forward in [true, false] {
        // [from x − s, from x, from x + s] of the previous row, s = 1 going forward and −1 going back.
        let mut prev: [Vec<u16>; 3] = [vec![0u16; w * nd], vec![0u16; w * nd], vec![0u16; w * nd]];
        let mut prev_min: [Vec<u16>; 3] = [vec![0u16; w], vec![0u16; w], vec![0u16; w]];
        let mut cur: [Vec<u16>; 3] = [vec![0u16; w * nd], vec![0u16; w * nd], vec![0u16; w * nd]];
        let mut cur_min: [Vec<u16>; 3] = [vec![0u16; w], vec![0u16; w], vec![0u16; w]];
        let mut along = vec![0u16; nd];
        let mut along_min = 0u16;
        for yi in 0..h {
            let y = if forward { yi } else { h - 1 - yi };
            for xi in 0..w {
                let x = if forward { xi } else { w - 1 - xi };
                let p = (y * w + x) * nd;
                let c = &cost[p..p + nd];
                // Along the row.
                along_min = if xi == 0 {
                    path_step(c, None, p1, p2, &mut tmp)
                } else {
                    let m = path_step(c, Some((&along, along_min)), p1, p2, &mut tmp);
                    m
                };
                along.copy_from_slice(&tmp);
                for d in 0..nd {
                    sum[p + d] += tmp[d];
                }
                // From the previous row: straight, and the two diagonals.
                for k in 0..3 {
                    if k != 1 && !diagonals {
                        continue;
                    }
                    let from: Option<usize> = if yi == 0 {
                        None
                    } else {
                        let s: i64 = if forward { 1 } else { -1 };
                        let fx = x as i64 + (k as i64 - 1) * s;
                        (0..w as i64).contains(&fx).then_some(fx as usize)
                    };
                    let m = match from {
                        Some(fx) => {
                            let lp = &prev[k][fx * nd..fx * nd + nd];
                            path_step(c, Some((lp, prev_min[k][fx])), p1, p2, &mut tmp)
                        }
                        None => path_step(c, None, p1, p2, &mut tmp),
                    };
                    cur[k][x * nd..x * nd + nd].copy_from_slice(&tmp);
                    cur_min[k][x] = m;
                    for d in 0..nd {
                        sum[p + d] += tmp[d];
                    }
                }
            }
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut prev_min, &mut cur_min);
        }
    }
    sum
}

/// How far (−0.5…0.5 px) the cost minimum lies from the best whole disparity, from the costs one below (`a`), at
/// (`b`) and one above (`c`) it.
fn subpixel(a: f32, b: f32, c: f32, fit: SubpixelFit) -> f32 {
    let off = match fit {
        SubpixelFit::Parabola => {
            let den = a + c - 2.0 * b;
            if den > 0.0 {
                (a - c) / (2.0 * den)
            } else {
                0.0
            }
        }
        // Two lines of equal and opposite slope through the three costs: the shape of a cost that adds up absolute
        // differences (a census distance is one), where a parabola pulls every match towards the whole pixel.
        SubpixelFit::Equiangular => {
            let den = (a - b).max(c - b);
            if den > 0.0 {
                (a - c) / (2.0 * den)
            } else {
                0.0
            }
        }
    };
    off.clamp(-0.5, 0.5)
}

#[derive(Clone, Copy, PartialEq)]
enum Why {
    Valid,
    Border,
    Saturated,
    NoTexture,
    NotUnique,
    LrMismatch,
    Speckle,
}

/// Drops connected regions of fewer than `min_px` pixels (4-neighbours within `range` px of disparity).
fn remove_speckles(disp: &mut [f32], why: &mut [Why], w: usize, h: usize, min_px: u32, range: f32) {
    if min_px == 0 {
        return;
    }
    let mut label = vec![0u32; w * h];
    let mut next = 1u32;
    let mut stack: Vec<usize> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if label[start] != 0 || !disp[start].is_finite() {
            continue;
        }
        label[start] = next;
        stack.push(start);
        members.clear();
        while let Some(i) = stack.pop() {
            members.push(i);
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if label[j] == 0 && disp[j].is_finite() && (disp[j] - disp[i]).abs() <= range {
                    label[j] = next;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
        if (members.len() as u32) < min_px {
            for &i in &members {
                disp[i] = f32::NAN;
                why[i] = Why::Speckle;
            }
        }
        next += 1;
    }
}

/// Depth of the left image of a rectified pair on the 1 : 2 : 4 matching grid. Pixels without a confident match are
/// NaN; matches farther than `max_depth_m` are infinity.
pub fn sgbm(left: &Rgb, right: &Rgb, rig: &StereoRig, cfg: &SgbmCfg, pyramid: Option<&PyramidCfg>) -> Result<StereoOut> {
    if (left.width, left.height) != (right.width, right.height) {
        bail!("스테레오 쌍의 크기가 다릅니다: 왼쪽 {}×{}, 오른쪽 {}×{}", left.width, left.height, right.width, right.height);
    }
    if rig.baseline_m <= 0.0 || rig.focal_px <= 0.0 {
        bail!("스테레오 보정값이 잘못되었습니다: 초점 {} px, 기선 {} m", rig.focal_px, rig.baseline_m);
    }
    let t0 = Instant::now();
    let (gw, gh) = stereo_grid(left.width, left.height, pyramid, cfg.per_l2);
    let (w, h) = (gw as usize, gh as usize);
    let f = rig.focal_on(left.width, gw);
    let fb = f * rig.baseline_m;
    let nd = ((fb / cfg.min_depth_m.max(0.05)).ceil() as usize + 2).next_multiple_of(8).clamp(8, (w / 2).max(8));
    let (gl, ml) = grey_on(left, gw, gh);
    let (gr, mr) = grey_on(right, gw, gh);
    let (cl, sd) = census(&gl, w, h);
    let (cr, _) = census(&gr, w, h);
    let block = cfg.block as usize;
    let cost = costs(&cl, &cr, w, h, nd, block);
    let sum = aggregate(&cost, w, h, nd, cfg);

    // The right image's own best disparity at each of its pixels, from the same aggregated costs: S_R(xr, d) =
    // S(xr + d, d).
    let mut right_disp = vec![-1i32; w * h];
    for y in 0..h {
        for xr in 0..w {
            let (mut best, mut bd) = (u16::MAX, -1i32);
            for d in 0..nd.min(w - xr) {
                let s = sum[(y * w + xr + d) * nd + d];
                if s < best {
                    best = s;
                    bd = d as i32;
                }
            }
            right_disp[y * w + xr] = bd;
        }
    }

    let edge = CENSUS_R as usize + block;
    let mut disp = vec![f32::NAN; w * h];
    let mut why = vec![Why::Valid; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x < edge || y < edge || x + edge >= w || y + edge >= h {
                why[i] = Why::Border;
                continue;
            }
            if ml[i] >= cfg.saturated {
                why[i] = Why::Saturated;
                continue;
            }
            if sd[i] < cfg.min_texture {
                why[i] = Why::NoTexture;
                continue;
            }
            let s = &sum[i * nd..i * nd + nd];
            let top = nd.min(x + 1);
            let (mut best, mut bd) = (u16::MAX, 0usize);
            for (d, &v) in s.iter().enumerate().take(top) {
                if v < best {
                    best = v;
                    bd = d;
                }
            }
            let second = s.iter().enumerate().take(top).filter(|(d, _)| d.abs_diff(bd) > 1).map(|(_, &v)| v).min().unwrap_or(u16::MAX);
            // A tie is not a win, even at zero cost (a pattern that repeats matches perfectly at every period).
            if (second as f32) * (1.0 - cfg.uniqueness) <= best as f32 {
                why[i] = Why::NotUnique;
                continue;
            }
            // The pixel's own matching cost must single the disparity out too. Where another disparity more than one
            // away matches as well or better, the paths' smoothness alone made the choice — on a repeating pattern
            // it carries whatever the image edge said across the whole row — and the depth would be a guess.
            let c = &cost[i * nd..i * nd + nd];
            if c.iter().enumerate().take(top).any(|(d, &v)| d.abs_diff(bd) > 1 && v <= c[bd]) {
                why[i] = Why::NotUnique;
                continue;
            }
            let mut d = bd as f32;
            if bd > 0 && bd + 1 < top {
                d += subpixel(s[bd - 1] as f32, s[bd] as f32, s[bd + 1] as f32, cfg.subpixel);
            }
            let xr = x as i64 - d.round() as i64;
            let ok = xr >= 0 && right_disp[y * w + xr as usize] >= 0 && (d - right_disp[y * w + xr as usize] as f32).abs() <= cfg.lr_max_diff;
            if !ok {
                why[i] = Why::LrMismatch;
                continue;
            }
            if xr >= 0 && mr[y * w + xr as usize] >= cfg.saturated {
                why[i] = Why::Saturated;
                continue;
            }
            disp[i] = d;
        }
    }
    remove_speckles(&mut disp, &mut why, w, h, cfg.speckle_px, cfg.speckle_range);

    let mut stats = StereoStats { width: gw, height: gh, disparities: nd as u32, focal_px: f, baseline_m: rig.baseline_m, ..StereoStats::default() };
    // Beyond range: past `max_depth_m`, or where the rig's own disparity error would move the depth by more than
    // `max_rel_sigma` of it.
    let z_far = cfg.max_depth_m.min(cfg.max_rel_sigma * fb / rig.subpixel_px.max(0.01));
    let d_far = fb / z_far.max(1e-3);
    let mut data = Vec::with_capacity(w * h);
    for (d, why) in disp.iter().zip(&why) {
        match why {
            Why::Valid if *d > d_far => {
                stats.valid += 1;
                data.push(fb / d);
            }
            Why::Valid => {
                stats.far += 1;
                data.push(f32::INFINITY);
            }
            other => {
                match other {
                    Why::Border => stats.border += 1,
                    Why::Saturated => stats.saturated += 1,
                    Why::NoTexture => stats.no_texture += 1,
                    Why::NotUnique => stats.not_unique += 1,
                    Why::LrMismatch => stats.lr_mismatch += 1,
                    Why::Speckle => stats.speckle += 1,
                    Why::Valid => {}
                }
                data.push(f32::NAN);
            }
        }
    }
    stats.ms = t0.elapsed().as_secs_f32() * 1000.0;
    Ok(StereoOut { depth: DepthMap { width: gw, height: gh, kind: DepthKind::Metric, data }, disparity: disp, stats })
}

/// The SGBM engine in the depth slot: a `DepthModel` that measures from a pair (`infer_pair`) and refuses a single
/// image, which is the monocular model's job.
pub struct SgbmDepth {
    pub rig: StereoRig,
    pub cfg: SgbmCfg,
    pyramid: Option<PyramidCfg>,
    last: Option<StereoStats>,
}

impl SgbmDepth {
    pub fn new(rig: StereoRig, cfg: SgbmCfg) -> SgbmDepth {
        SgbmDepth { rig, cfg, pyramid: Some(PyramidCfg::default()), last: None }
    }

    /// The matching grid for a frame of this size.
    pub fn grid_for(&self, width: u32, height: u32) -> (u32, u32) {
        stereo_grid(width, height, self.pyramid.as_ref(), self.cfg.per_l2)
    }

    pub fn compute(&mut self, left: &Rgb, right: &Rgb) -> Result<StereoOut> {
        let out = sgbm(left, right, &self.rig, &self.cfg, self.pyramid.as_ref())?;
        self.last = Some(out.stats);
        Ok(out)
    }

    /// What the last pair gave: the grid, the share with depth, and why the rest has none.
    pub fn last_stats(&self) -> Option<StereoStats> {
        self.last
    }
}

impl DepthModel for SgbmDepth {
    fn mode(&self) -> &str {
        MODE
    }

    fn max_depth_m(&self) -> Option<f32> {
        Some(self.cfg.max_depth_m)
    }

    fn infer(&mut self, _rgb: &Rgb) -> Result<DepthMap> {
        bail!("SGBM 스테레오 깊이는 왼쪽·오른쪽 두 화면이 필요합니다 (화면 하나는 단안 모델 TIPSv2의 몫)")
    }

    fn set_pyramid(&mut self, cfg: Option<&PyramidCfg>) {
        // Off the pyramid the grid still snaps to the default one, so the depth stays a whole multiple of it.
        self.pyramid = cfg.copied();
    }

    fn stereo_rig(&self) -> Option<StereoRig> {
        Some(self.rig)
    }

    fn infer_pair(&mut self, left: &Rgb, right: &Rgb) -> Result<DepthMap> {
        Ok(self.compute(left, right)?.depth)
    }

    fn stereo_stats(&self) -> Option<StereoStats> {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 288;
    const H: u32 = 192;

    fn rig() -> StereoRig {
        StereoRig { focal_px: 205.0, baseline_m: 0.05, subpixel_px: 0.25 }
    }

    /// Random dots, blurred a little so that a fractional shift has something to interpolate.
    fn texture(seed: u64, w: u32, h: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut raw = vec![0.0f32; (w * h) as usize];
        for v in raw.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = 40.0 + 170.0 * ((s >> 33) as f32 / (1u64 << 31) as f32);
        }
        let mut out = raw.clone();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = (y * w + x) as usize;
                out[i] = 0.5 * raw[i] + 0.125 * (raw[i - 1] + raw[i + 1] + raw[i - w as usize] + raw[i + w as usize]);
            }
        }
        out
    }

    fn sample(t: &[f32], w: u32, x: f32, y: u32) -> f32 {
        let x = x.clamp(0.0, (w - 1) as f32);
        let (x0, k) = (x.floor() as u32, x - x.floor());
        let x1 = (x0 + 1).min(w - 1);
        t[(y * w + x0) as usize] * (1.0 - k) + t[(y * w + x1) as usize] * k
    }

    fn rgb_of(g: impl Fn(u32, u32) -> f32, w: u32, h: u32) -> Rgb {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = g(x, y).round().clamp(0.0, 255.0) as u8;
                data.extend_from_slice(&[v, v, v]);
            }
        }
        Rgb::new(w, h, data)
    }

    /// A pair whose left pixel (x, y) is the right pixel (x − d(x, y), y): the right image samples the scene at
    /// x + d.
    fn pair(scene: &[f32], disp: impl Fn(u32, u32) -> f32) -> (Rgb, Rgb) {
        let left = rgb_of(|x, y| scene[(y * W + x) as usize], W, H);
        // Right pixel xr shows the scene point whose left pixel is xr + d; find d at that point by a short search.
        let right = rgb_of(
            |xr, y| {
                let mut x = xr as f32;
                for _ in 0..4 {
                    x = xr as f32 + disp(x.round().clamp(0.0, (W - 1) as f32) as u32, y);
                }
                sample(scene, W, x, y)
            },
            W,
            H,
        );
        (left, right)
    }

    fn inner(x: u32, y: u32) -> bool {
        x >= 40 && x < W - 8 && y >= 8 && y < H - 8
    }

    #[test]
    fn the_grid_is_a_whole_multiple_of_the_pyramid() {
        assert_eq!(stereo_grid(384, 256, None, 8), (288, 192), "L2 36 x 24 times 8: the SAM mask's grid");
        assert_eq!(stereo_grid(384, 256, None, 4), (144, 96));
        assert_eq!(stereo_grid(384, 256, None, 6), (144, 96), "rounded down to a power of two");
        for (w, h) in [(384u32, 256u32), (640, 480), (192, 128), (288, 192)] {
            let pyr = Pyramid::for_frame(w, h, &PyramidCfg::default());
            let (gw, gh) = stereo_grid(w, h, None, 8);
            let ((c1, r1), (c2, r2)) = (pyr.l1(), pyr.l2());
            assert!(gw <= w && gh <= h, "no upsampling: {gw}x{gh} in {w}x{h}");
            assert!(gw % c2 == 0 && gh % r2 == 0 && gw % c1 == 0 && gh % r1 == 0, "{w}x{h}: {gw}x{gh} over L1 {c1}x{r1}, L2 {c2}x{r2}");
            assert_eq!(gw / c2, gh / r2, "square blocks");
        }
    }

    #[test]
    fn a_shifted_texture_gives_back_its_disparity_and_metric_depth() {
        let scene = texture(3, W, H);
        let (l, r) = pair(&scene, |_, _| 12.0);
        let out = sgbm(&l, &r, &rig(), &SgbmCfg::default(), None).unwrap();
        assert_eq!((out.depth.width, out.depth.height), (W, H));
        assert_eq!(out.depth.kind, DepthKind::Metric);
        let (mut n, mut good, mut valid) = (0, 0, 0);
        for y in 0..H {
            for x in 0..W {
                if !inner(x, y) {
                    continue;
                }
                n += 1;
                let d = out.disparity[(y * W + x) as usize];
                if d.is_finite() {
                    valid += 1;
                    good += ((d - 12.0).abs() <= 0.25) as u32;
                    let z = out.depth.at(x, y);
                    assert!((z - 205.0 * 0.05 / d).abs() < 1e-4, "Z = f·b/d: {z} at d {d}");
                }
            }
        }
        assert!(valid as f32 >= 0.95 * n as f32, "{valid} of {n} matched");
        assert!(good as f32 >= 0.95 * valid as f32, "{good} of {valid} within 0.25 px");
        assert!(out.stats.ms > 0.0 && out.stats.disparities >= 48, "{:?}", out.stats);
    }

    #[test]
    fn a_slanted_surface_is_matched_to_a_fraction_of_a_pixel() {
        let scene = texture(5, W, H);
        let disp = |x: u32, _y: u32| 6.0 + 10.0 * x as f32 / W as f32;
        let (l, r) = pair(&scene, disp);
        let out = sgbm(&l, &r, &rig(), &SgbmCfg::default(), None).unwrap();
        let (mut se, mut n) = (0.0f32, 0u32);
        for y in 0..H {
            for x in 0..W {
                let d = out.disparity[(y * W + x) as usize];
                if inner(x, y) && d.is_finite() {
                    se += (d - disp(x, y)).powi(2);
                    n += 1;
                }
            }
        }
        let rms = (se / n.max(1) as f32).sqrt();
        assert!(n > 30_000, "{n}");
        assert!(rms < 0.35, "sub-pixel RMS {rms}");
    }

    #[test]
    fn a_plain_board_and_a_highlight_are_left_empty() {
        let mut scene = texture(7, W, H);
        for y in 60..120 {
            for x in 120..200 {
                scene[(y * W + x) as usize] = 128.0;
            }
            for x in 220..260 {
                scene[(y * W + x) as usize] = 255.0;
            }
        }
        let (l, r) = pair(&scene, |_, _| 10.0);
        let out = sgbm(&l, &r, &rig(), &SgbmCfg::default(), None).unwrap();
        // Inside by more than the census window: a pixel whose window reaches the textured border still matches on
        // that texture, and its depth is the true one, not a guess.
        let filled = |x0: u32, x1: u32| (64..116).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| out.depth.at(x, y).is_finite()).count();
        assert_eq!(filled(126, 194), 0, "nothing to match on the board: no depth is made up");
        assert_eq!(filled(222, 258), 0, "blown out: left empty");
        assert!(out.stats.no_texture > 3000 && out.stats.saturated > 1500, "{:?}", out.stats);
        assert!(out.depth.data.iter().all(|d| d.is_nan() || (d.is_finite() && *d > 0.0) || d.is_infinite()));
    }

    #[test]
    fn the_strip_one_eye_cannot_see_is_not_given_the_near_surface() {
        let back = texture(11, W, H);
        let front = texture(12, W, H);
        let (bx0, bx1, by0, by1) = (140u32, 200u32, 50u32, 140u32);
        let in_box = |x: u32, y: u32| (bx0..bx1).contains(&x) && (by0..by1).contains(&y);
        let (db, df) = (4.0f32, 20.0f32);
        let left = rgb_of(|x, y| if in_box(x, y) { front[(y * W + x) as usize] } else { back[(y * W + x) as usize] }, W, H);
        let right = rgb_of(
            |xr, y| {
                // The front box moves df to the left in the right image and covers the background there.
                let xf = xr as f32 + df;
                if in_box(xf as u32, y) {
                    sample(&front, W, xf, y)
                } else {
                    sample(&back, W, xr as f32 + db, y)
                }
            },
            W,
            H,
        );
        let out = sgbm(&left, &right, &rig(), &SgbmCfg::default(), None).unwrap();
        // The background just left of the box: the right eye sees the box there instead.
        let strip: Vec<f32> = (by0 + 4..by1 - 4).flat_map(|y| (bx0 - 14..bx0 - 2).map(move |x| (x, y))).map(|(x, y)| out.disparity[(y * W + x) as usize]).collect();
        let near = strip.iter().filter(|d| d.is_finite() && (**d - df).abs() < 3.0).count();
        assert!(near * 20 < strip.len(), "{near} of {} occluded pixels took the box's disparity", strip.len());
        let inside: Vec<f32> = (by0 + 6..by1 - 6).flat_map(|y| (bx0 + 6..bx1 - 6).map(move |x| (x, y))).map(|(x, y)| out.disparity[(y * W + x) as usize]).collect();
        let on_box = inside.iter().filter(|d| d.is_finite() && (**d - df).abs() <= 0.5).count();
        assert!(on_box * 10 >= inside.len() * 8, "{on_box} of {} on the box", inside.len());
        assert!(out.stats.lr_mismatch + out.stats.not_unique > 100, "{:?}", out.stats);
    }

    #[test]
    fn a_repeating_pattern_has_no_unique_match() {
        let scene: Vec<f32> = (0..W * H).map(|i| if (i % W) / 3 % 2 == 0 { 60.0 } else { 190.0 }).collect();
        let (l, r) = pair(&scene, |_, _| 9.0);
        let out = sgbm(&l, &r, &rig(), &SgbmCfg::default(), None).unwrap();
        let valid = (0..H).flat_map(|y| (0..W).map(move |x| (x, y))).filter(|&(x, y)| inner(x, y) && out.depth.at(x, y).is_finite()).count();
        assert!(valid < 500, "{valid} pixels claim a depth on stripes that repeat every 6 px");
        assert!(out.stats.not_unique > 20_000, "{:?}", out.stats);
    }

    #[test]
    fn the_engine_measures_from_a_pair_and_refuses_one_image() {
        let scene = texture(13, W, H);
        let (l, r) = pair(&scene, |_, _| 8.0);
        let mut m = SgbmDepth::new(rig(), SgbmCfg::default());
        assert!(m.infer(&l).is_err(), "one image is the monocular model's job");
        assert_eq!(m.stereo_rig(), Some(rig()));
        let d = m.infer_pair(&l, &r).unwrap();
        assert_eq!(d.kind, DepthKind::Metric);
        let st = m.stereo_stats().unwrap();
        assert!(st.valid_frac() > 0.8, "{st:?}");
        let mid = d.at(W / 2, H / 2);
        assert!((mid - 205.0 * 0.05 / 8.0).abs() < 0.05, "{mid}");
        assert!(sgbm(&l, &Rgb::new(4, 4, vec![0; 48]), &rig(), &SgbmCfg::default(), None).is_err());
    }

    #[test]
    fn a_frame_larger_than_the_grid_is_averaged_down_onto_it() {
        let (fw, fh) = (384u32, 256u32);
        let scene = texture(17, fw, fh);
        let shift = 16.0f32;
        let left = rgb_of(|x, y| scene[(y * fw + x) as usize], fw, fh);
        let right = rgb_of(|x, y| sample(&scene, fw, x as f32 + shift, y), fw, fh);
        let rig = StereoRig { focal_px: 274.0, baseline_m: 0.05, subpixel_px: 0.25 };
        let out = sgbm(&left, &right, &rig, &SgbmCfg::default(), None).unwrap();
        assert_eq!((out.stats.width, out.stats.height), (288, 192));
        assert!((out.stats.focal_px - 274.0 * 0.75).abs() < 1e-3);
        let want = shift * 0.75;
        let ok = out.disparity.iter().filter(|d| d.is_finite() && (**d - want).abs() < 0.4).count();
        assert!(ok > 40_000, "{ok} of {} near {want} px", out.disparity.len());
    }
}
