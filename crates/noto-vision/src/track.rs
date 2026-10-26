//! ML track 3: mask tracking from point prompts.
//!
//! The interface is the one SAM 3.1 offers (point prompts in, masks with
//! object ids out, objects kept across frames), so the real model can replace
//! the stand-in without touching the callers. The SAM 3.1 weights the project
//! points at (`avencera/sam3.1-mlx-bf16`) are for Apple's MLX and do not run
//! on this Windows/CUDA build; see `docs/PLAN-v0.3.md` for what it takes.
//!
//! [`RegionTracker`] is the stand-in and an honest one: it uses only the
//! keyframe and the depth map, never the renderer's object ids. From each
//! prompt it grows a region over pixels whose colour stays close to the
//! region's mean and whose depth does not jump (an object ends where the depth
//! steps to what is behind it). A track is re-found in the next keyframe by
//! growing again from the pixel nearest the predicted centroid whose colour
//! matches the track's colour model; prompts that land inside a track's mask
//! refresh it instead of making a new one.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::frame::{DepthMap, Mask, Rgb};

/// A point prompt, as SAM takes it: a pixel and the label it stands for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointPrompt {
    pub label: String,
    pub px: f32,
    pub py: f32,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackedMask {
    pub id: u32,
    pub label: String,
    /// Mask at the tracker's working resolution.
    pub mask: Mask,
    /// Centroid in pixels of the keyframe.
    pub cx: f32,
    pub cy: f32,
    /// Box in pixels of the keyframe.
    pub bbox: [f32; 4],
    pub area_frac: f32,
    /// Mask quality 0..1 (colour coherence times prompt agreement).
    pub score: f32,
    /// IoU with this track's mask in the previous keyframe (no motion compensation).
    pub iou_prev: Option<f32>,
    /// Keyframes this track has been followed.
    pub age: u32,
    /// True when it was re-found from the previous keyframe without a fresh prompt.
    pub propagated: bool,
}

pub trait MaskTracker: Send {
    /// `region-grow` (stand-in) or `sam3.1`.
    fn mode(&self) -> &str;
    fn reset(&mut self);
    /// Follows the existing tracks into this keyframe and starts tracks for
    /// prompts that do not fall on one. Returns the live tracks.
    fn update(&mut self, rgb: &Rgb, depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>>;
    /// Called with keyframes that [`MaskTracker::update`] will see next, so a
    /// model with an image encoder can encode them in one batch ahead of the
    /// sequential tracking. The default does nothing.
    fn prepare(&mut self, _frames: &[&Rgb]) -> Result<()> {
        Ok(())
    }
    /// Device the tracker runs on.
    fn device(&self) -> String {
        "cpu".into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionCfg {
    /// Working resolution divisor (2 means half width and height).
    pub downscale: u32,
    /// Largest RGB distance from the running region mean for a pixel to join.
    pub color_tol: f32,
    /// A pixel whose depth differs from its neighbour's by more than this
    /// fraction is across an object edge.
    pub depth_jump: f32,
    pub min_area_frac: f32,
    pub max_area_frac: f32,
    /// Search radius around the predicted centroid when re-finding a track, working pixels.
    pub search_px: i32,
    /// Colour distance accepted when re-finding a track's seed.
    pub refind_tol: f32,
    pub max_tracks: usize,
    /// Keyframes a track may be carried without a prompt landing on it. SAM
    /// keeps an object through short detection gaps; past this the track is
    /// dropped instead of drifting onto the background.
    pub max_misses: u32,
}

impl Default for RegionCfg {
    fn default() -> Self {
        RegionCfg {
            downscale: 2,
            color_tol: 40.0,
            depth_jump: 0.06,
            min_area_frac: 0.0004,
            max_area_frac: 0.25,
            search_px: 14,
            refind_tol: 46.0,
            max_tracks: 8,
            max_misses: 2,
        }
    }
}

struct Live {
    id: u32,
    label: String,
    mask: Mask,
    color: [f32; 3],
    cx: f32,
    cy: f32,
    age: u32,
    misses: u32,
}

pub struct RegionTracker {
    pub cfg: RegionCfg,
    tracks: Vec<Live>,
    next_id: u32,
}

impl Default for RegionTracker {
    fn default() -> Self {
        RegionTracker::new(RegionCfg::default())
    }
}

struct Work {
    w: u32,
    h: u32,
    rgb: Vec<[f32; 3]>,
    depth: Option<Vec<f32>>,
}

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// 3×3 per-channel median: removes one-pixel flecks and lines (ore texture)
/// that would otherwise wall a region off, and keeps object edges.
fn median3(px: &[[f32; 3]], w: u32, h: u32) -> Vec<[f32; 3]> {
    let mut out = px.to_vec();
    let mut win: [Vec<f32>; 3] = [Vec::with_capacity(9), Vec::with_capacity(9), Vec::with_capacity(9)];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            for c in win.iter_mut() {
                c.clear();
            }
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (sx, sy) = ((x + dx).clamp(0, w as i32 - 1), (y + dy).clamp(0, h as i32 - 1));
                    let p = px[(sy as u32 * w + sx as u32) as usize];
                    for k in 0..3 {
                        win[k].push(p[k]);
                    }
                }
            }
            let mut m = [0.0f32; 3];
            for k in 0..3 {
                win[k].sort_by(f32::total_cmp);
                m[k] = win[k][4];
            }
            out[(y as u32 * w + x as u32) as usize] = m;
        }
    }
    out
}

impl Work {
    fn new(rgb: &Rgb, depth: Option<&DepthMap>, k: u32) -> Work {
        let k = k.max(1);
        let (w, h) = ((rgb.width / k).max(1), (rgb.height / k).max(1));
        let mut px = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                // Box filter over the k x k block: flecks average out a little.
                let mut s = [0.0f32; 3];
                let mut n = 0.0;
                for dy in 0..k {
                    for dx in 0..k {
                        let (sx, sy) = ((x * k + dx).min(rgb.width - 1), (y * k + dy).min(rgb.height - 1));
                        let p = rgb.px(sx, sy);
                        s[0] += p[0] as f32;
                        s[1] += p[1] as f32;
                        s[2] += p[2] as f32;
                        n += 1.0;
                    }
                }
                px.push([s[0] / n, s[1] / n, s[2] / n]);
            }
        }
        let px = median3(&px, w, h);
        let depth = depth.map(|d| {
            let mut v = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    v.push(d.at_scaled((x * k) as f32 + k as f32 * 0.5, (y * k) as f32 + k as f32 * 0.5, rgb.width, rgb.height));
                }
            }
            v
        });
        Work { w, h, rgb: px, depth }
    }

    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.w + x) as usize
    }

    /// Grows a region from a seed. Returns the mask, its mean colour, and
    /// whether it stopped at the size cap (then it is ground or sky, not an
    /// object bounded by colour and depth edges).
    fn grow(&self, seed: (u32, u32), cfg: &RegionCfg) -> (Mask, [f32; 3], bool) {
        let mut mask = Mask::empty(self.w, self.h);
        let max_px = (cfg.max_area_frac * (self.w * self.h) as f32) as u32 + 1;
        let mut mean = self.rgb[self.idx(seed.0, seed.1)];
        let mut n = 1.0f32;
        let mut stack = vec![seed];
        mask.set(seed.0, seed.1);
        let mut count = 1;
        while let Some((x, y)) = stack.pop() {
            let here = self.idx(x, y);
            let neigh = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
            for (nx, ny) in neigh {
                if nx >= self.w || ny >= self.h || mask.get(nx, ny) {
                    continue;
                }
                let j = self.idx(nx, ny);
                let c = self.rgb[j];
                if let Some(d) = &self.depth {
                    let (a, b) = (d[here], d[j]);
                    if !b.is_finite() {
                        continue; // sky
                    }
                    if a.is_finite() && (a - b).abs() > cfg.depth_jump * a.min(b).max(0.1) {
                        continue;
                    }
                }
                if dist(c, mean) > cfg.color_tol {
                    continue;
                }
                mask.set(nx, ny);
                count += 1;
                n += 1.0;
                for k in 0..3 {
                    mean[k] += (c[k] - mean[k]) / n;
                }
                if count >= max_px {
                    return (mask, mean, true);
                }
                stack.push((nx, ny));
            }
        }
        (mask, mean, false)
    }

    /// Per-channel median colour of the window of radius `r` around `(x, y)`.
    /// A prompt that lands on a fleck of ore still seeds the stone around it.
    fn local_median(&self, x: f32, y: f32, r: i32) -> Option<[f32; 3]> {
        let (cx, cy) = (x as i32, y as i32);
        let mut ch: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || py < 0 || px >= self.w as i32 || py >= self.h as i32 {
                    continue;
                }
                let c = self.rgb[self.idx(px as u32, py as u32)];
                for k in 0..3 {
                    ch[k].push(c[k]);
                }
            }
        }
        if ch[0].is_empty() {
            return None;
        }
        let mut out = [0.0f32; 3];
        for k in 0..3 {
            ch[k].sort_by(f32::total_cmp);
            out[k] = ch[k][ch[k].len() / 2];
        }
        Some(out)
    }

    /// Pixel within `r` of `(x, y)` whose colour is nearest to `color`.
    fn best_seed(&self, x: f32, y: f32, color: Option<[f32; 3]>, r: i32) -> Option<((u32, u32), f32)> {
        let (cx, cy) = (x as i32, y as i32);
        let mut best: Option<((u32, u32), f32)> = None;
        for dy in -r..=r {
            for dx in -r..=r {
                let (px, py) = (cx + dx, cy + dy);
                if px < 0 || py < 0 || px >= self.w as i32 || py >= self.h as i32 {
                    continue;
                }
                let i = self.idx(px as u32, py as u32);
                if self.depth.as_ref().map_or(false, |d| !d[i].is_finite()) {
                    continue;
                }
                let colour_cost = color.map_or(0.0, |c| dist(self.rgb[i], c));
                let cost = colour_cost + 0.6 * ((dx * dx + dy * dy) as f32).sqrt();
                if best.map_or(true, |(_, b)| cost < b) {
                    best = Some(((px as u32, py as u32), cost));
                }
            }
        }
        best
    }
}

impl RegionTracker {
    pub fn new(cfg: RegionCfg) -> RegionTracker {
        RegionTracker { cfg, tracks: Vec::new(), next_id: 1 }
    }

    fn accept(&self, work: &Work, m: &Mask, capped: bool) -> bool {
        let frac = m.area() as f32 / (work.w * work.h) as f32;
        !capped && frac >= self.cfg.min_area_frac && frac <= self.cfg.max_area_frac
    }

    fn out(&self, t: &Live, k: u32, frame: (u32, u32), iou_prev: Option<f32>, propagated: bool, score: f32) -> TrackedMask {
        let b = t.mask.bbox().unwrap_or([0, 0, 0, 0]);
        let kf = k as f32;
        TrackedMask {
            id: t.id,
            label: t.label.clone(),
            mask: t.mask.clone(),
            cx: t.cx * kf,
            cy: t.cy * kf,
            bbox: [b[0] as f32 * kf, b[1] as f32 * kf, (b[2] + 1) as f32 * kf, (b[3] + 1) as f32 * kf],
            area_frac: t.mask.area() as f32 / (frame.0 / k * (frame.1 / k)).max(1) as f32,
            score,
            iou_prev,
            age: t.age,
            propagated,
        }
    }
}

impl MaskTracker for RegionTracker {
    fn mode(&self) -> &str {
        "region-grow"
    }

    fn reset(&mut self) {
        self.tracks.clear();
    }

    fn update(&mut self, rgb: &Rgb, depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>> {
        let k = self.cfg.downscale.max(1);
        let work = Work::new(rgb, depth, k);
        let kf = k as f32;
        let mut out = Vec::new();
        let mut used = vec![false; prompts.len()];
        let old = std::mem::take(&mut self.tracks);
        // 1. Follow the existing tracks.
        for mut t in old {
            // A prompt of the same label on the mask, or near its box (the
            // object moved in the image between keyframes), refreshes the track.
            let near = |p: &PointPrompt| -> bool {
                let (x, y) = (p.px / kf, p.py / kf);
                let (xi, yi) = (x as u32, y as u32);
                if xi < t.mask.width && yi < t.mask.height && t.mask.get(xi, yi) {
                    return true;
                }
                let Some(b) = t.mask.bbox() else { return false };
                let (bw, bh) = ((b[2] - b[0] + 1) as f32, (b[3] - b[1] + 1) as f32);
                let (gx, gy) = ((0.5 * bw).max(6.0), (0.5 * bh).max(6.0));
                x >= b[0] as f32 - gx && x <= b[2] as f32 + gx && y >= b[1] as f32 - gy && y <= b[3] as f32 + gy
            };
            let prompt_hit = prompts.iter().enumerate().find(|(i, p)| !used[*i] && p.label == t.label && near(p)).map(|(i, p)| (i, p.clone()));
            let (seed, propagated) = match &prompt_hit {
                Some((i, p)) => {
                    used[*i] = true;
                    (work.best_seed(p.px / kf, p.py / kf, Some(t.color), 2).map(|s| s.0), false)
                }
                None => {
                    let s = work.best_seed(t.cx, t.cy, Some(t.color), self.cfg.search_px);
                    (s.filter(|(seed, _)| dist(work.rgb[work.idx(seed.0, seed.1)], t.color) <= self.cfg.refind_tol).map(|s| s.0), true)
                }
            };
            let Some(seed) = seed else { continue };
            if propagated && t.misses >= self.cfg.max_misses {
                continue;
            }
            let (mask, color, capped) = work.grow(seed, &self.cfg);
            if !self.accept(&work, &mask, capped) {
                continue;
            }
            t.misses = if propagated { t.misses + 1 } else { 0 };
            let iou = t.mask.iou(&mask);
            let coherence = (1.0 - dist(color, t.color) / (2.0 * self.cfg.refind_tol)).clamp(0.0, 1.0);
            let score = if propagated { 0.6 * coherence } else { 0.4 + 0.6 * coherence };
            let (cx, cy) = mask.centroid().expect("accepted masks are not empty");
            t.mask = mask;
            t.color = [0.7 * t.color[0] + 0.3 * color[0], 0.7 * t.color[1] + 0.3 * color[1], 0.7 * t.color[2] + 0.3 * color[2]];
            t.cx = cx;
            t.cy = cy;
            t.age += 1;
            out.push(self.out(&t, k, (rgb.width, rgb.height), Some(iou), propagated, score));
            self.tracks.push(t);
        }
        // 2. New tracks from the prompts that did not land on one.
        for (i, p) in prompts.iter().enumerate() {
            if used[i] || self.tracks.len() >= self.cfg.max_tracks {
                continue;
            }
            let on_other = self.tracks.iter().any(|t| {
                let (x, y) = ((p.px / kf) as u32, (p.py / kf) as u32);
                x < t.mask.width && y < t.mask.height && t.mask.get(x, y)
            });
            if on_other {
                continue;
            }
            let around = work.local_median(p.px / kf, p.py / kf, 2);
            let Some((seed, _)) = work.best_seed(p.px / kf, p.py / kf, around, 2) else { continue };
            let (mask, color, capped) = work.grow(seed, &self.cfg);
            if !self.accept(&work, &mask, capped) {
                continue;
            }
            let (cx, cy) = mask.centroid().expect("accepted masks are not empty");
            let t = Live { id: self.next_id, label: p.label.clone(), mask, color, cx, cy, age: 1, misses: 0 };
            self.next_id += 1;
            out.push(self.out(&t, k, (rgb.width, rgb.height), None, false, 0.4 + 0.6 * p.score.clamp(0.0, 1.0)));
            self.tracks.push(t);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;

    /// Grass background with a grey block at `x0..x0+20, 20..40`, nearer than the ground.
    fn scene(x0: u32) -> (Rgb, DepthMap) {
        let (w, h) = (80u32, 60u32);
        let mut rgb = Vec::new();
        let mut depth = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let block = x >= x0 && x < x0 + 20 && (20..40).contains(&y);
                let fleck = block && (x + y) % 7 == 0;
                rgb.extend_from_slice(&if fleck { [180, 120, 90] } else if block { [128, 128, 132] } else { [70, 130, 60] });
                depth.push(if block { 4.0 } else { 6.0 + y as f32 * 0.01 });
            }
        }
        (Rgb::new(w, h, rgb), DepthMap { width: w, height: h, kind: DepthKind::Metric, data: depth })
    }

    #[test]
    fn a_prompt_on_the_block_segments_the_block_and_follows_it() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let first = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 40.0, py: 30.0, score: 0.8 }]).unwrap();
        assert_eq!(first.len(), 1);
        let m = &first[0];
        assert!(m.mask.area() >= 300 && m.mask.area() <= 400, "area {}", m.mask.area());
        assert!((m.cx - 40.0).abs() < 2.0 && (m.cy - 30.0).abs() < 2.0, "{} {}", m.cx, m.cy);
        // Next keyframe: the block moved 6 px right, no prompt. The track is re-found.
        let (rgb2, depth2) = scene(36);
        let next = t.update(&rgb2, Some(&depth2), &[]).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].id, m.id);
        assert!(next[0].propagated);
        assert!((next[0].cx - 46.0).abs() < 2.0, "{}", next[0].cx);
        let iou = next[0].iou_prev.unwrap();
        assert!(iou > 0.4 && iou < 0.9, "moved block overlaps partly: {iou}");
    }

    #[test]
    fn a_track_without_prompts_is_dropped_after_its_misses_and_a_nearby_prompt_keeps_it() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let first = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 40.0, py: 30.0, score: 0.8 }]).unwrap();
        let id = first[0].id;
        // A prompt just outside the moved block's old box still refreshes the same track.
        let (rgb2, depth2) = scene(44);
        let near = t.update(&rgb2, Some(&depth2), &[PointPrompt { label: "stone".into(), px: 53.0, py: 30.0, score: 0.8 }]).unwrap();
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].id, id, "same object, same id");
        assert!(!near[0].propagated);
        for _ in 0..2 {
            assert_eq!(t.update(&rgb2, Some(&depth2), &[]).unwrap().len(), 1);
        }
        assert!(t.update(&rgb2, Some(&depth2), &[]).unwrap().is_empty(), "dropped after two keyframes without support");
    }

    #[test]
    fn a_prompt_on_the_ground_alone_is_too_big_to_be_an_object() {
        let mut t = RegionTracker::new(RegionCfg { downscale: 1, ..RegionCfg::default() });
        let (rgb, depth) = scene(30);
        let got = t.update(&rgb, Some(&depth), &[PointPrompt { label: "stone".into(), px: 5.0, py: 55.0, score: 0.8 }]).unwrap();
        assert!(got.is_empty(), "the grass region covers most of the frame and is refused");
    }
}
