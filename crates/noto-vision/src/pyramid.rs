//! The 1 : 2 : 4 pyramid that lines the vision layers up on one grid. SigLIP2 (L1, 16 px patches, NaFlex budget
//! 256) takes `l1_short_side` patches across the frame's short side; TIPSv2 (L2, 14 px patches) takes exactly
//! `l2_ratio` times that grid, so every L1 patch covers exactly `l2_ratio` × `l2_ratio` L2 patches; the SAM mask (L3)
//! is sampled at `mask_per_cell` pixels per L1 patch. A 384 × 256 keyframe gives 18 × 12 : 36 × 24 : 288 × 192, a
//! 640 × 480 frame 16 × 12 : 32 × 24 : 256 × 192 (SAM's own 64 × 48 embedding and 256 × 192 low-res logits).
//!
//! On that grid the layers check each other: the depth layer's sky and far cells take the semantic layer's heat away
//! (`gate_points`), and the depth inside a semantic peak's patch moves the peak onto the nearer surface it straddles
//! (`snap_points`).

use noto_ml::preprocess::{pyramid_grid, PYRAMID_SHORT_SIDE};
use serde::{Deserialize, Serialize};

use crate::align::{ground_depth, HeightFn};
use crate::camera::{Intrinsics, Pose};
use crate::frame::{DepthMap, SegMap};
use crate::locate::{points_with_presence, Heat, SemanticPoint};

/// SigLIP2 base patch16 NaFlex: its patch budget (`num_patches`).
pub const L1_BUDGET: usize = 256;
/// TIPSv2 patches per SigLIP2 patch along each axis.
pub const L2_RATIO: usize = 2;
/// SAM mask pixels per SigLIP2 patch along each axis: 4 per SAM embedding cell of a 4:3 frame (64 × 48), i.e. its
/// 256 × 192 low-res logits one to one.
pub const MASK_PER_CELL: u32 = 16;

/// L2 → L1: which semantic patches may hold a peak at all. A patch that is mostly sky, or whose measured content is
/// mostly farther than anything the robot acts on, only ever produces look-alike heat (the same limits as the tile
/// scan's `RoiCfg`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellGateCfg {
    pub enabled: bool,
    pub sky_max_frac: f32,
    pub far_m: f32,
    pub min_near_frac: f32,
}

impl Default for CellGateCfg {
    fn default() -> Self {
        CellGateCfg { enabled: true, sky_max_frac: 0.6, far_m: 18.0, min_near_frac: 0.25 }
    }
}

/// L1 → L2: a semantic peak sits somewhere in a 16 px patch; when the depth patch under it reads the background behind
/// a nearer surface in the same semantic patch, the peak moves onto that surface. `edge_gap` is the relative depth
/// step that counts as an edge (the fusion flag's `depth_gap`), measured against the ground plane's own depth at each
/// sub-patch so that the ground's slope across a patch is not an edge.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapCfg {
    pub enabled: bool,
    pub edge_gap: f32,
    pub max_rel_spread: f32,
    pub min_valid_frac: f32,
}

impl Default for SnapCfg {
    fn default() -> Self {
        SnapCfg { enabled: true, edge_gap: 0.25, max_rel_spread: 0.25, min_valid_frac: 0.5 }
    }
}

fn on() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PyramidCfg {
    pub l1_short_side: usize,
    pub l1_budget: usize,
    /// 2 for the 1 : 2 pyramid; 1 runs TIPSv2 on the SigLIP2 grid itself (a quarter of its tokens, no sub-patches to
    /// snap to).
    pub l2_ratio: usize,
    /// Mask pixels per L1 patch along each axis, at most one per frame pixel.
    pub mask_per_cell: u32,
    #[serde(default)]
    pub gate: CellGateCfg,
    #[serde(default)]
    pub snap: SnapCfg,
    /// Encode a frame with SAM only when it has a prompt that passed the semantic and geometric gates or a live track.
    #[serde(default = "on")]
    pub lazy_sam: bool,
}

impl Default for PyramidCfg {
    fn default() -> Self {
        PyramidCfg {
            l1_short_side: PYRAMID_SHORT_SIDE,
            l1_budget: L1_BUDGET,
            l2_ratio: L2_RATIO,
            mask_per_cell: MASK_PER_CELL,
            gate: CellGateCfg::default(),
            snap: SnapCfg::default(),
            lazy_sam: true,
        }
    }
}

impl PyramidCfg {
    /// TIPSv2 on the SigLIP2 grid: 75 % fewer depth tokens, no peak snapping.
    pub fn fast() -> PyramidCfg {
        PyramidCfg { l2_ratio: 1, ..PyramidCfg::default() }
    }
}

/// The pyramid of one frame size, in frame pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pyramid {
    pub width: u32,
    pub height: u32,
    pub cols: u32,
    pub rows: u32,
    pub l2_ratio: u32,
    pub mask_per_cell: u32,
}

impl Pyramid {
    pub fn for_frame(width: u32, height: u32, cfg: &PyramidCfg) -> Pyramid {
        let (rows, cols) = pyramid_grid(width, height, cfg.l1_short_side, cfg.l1_budget);
        let (cols, rows) = (cols as u32, rows as u32);
        let finest = (width / cols).min(height / rows).max(1);
        Pyramid { width: width.max(1), height: height.max(1), cols, rows, l2_ratio: cfg.l2_ratio.max(1) as u32, mask_per_cell: cfg.mask_per_cell.clamp(1, finest) }
    }

    /// SigLIP2 grid (cols, rows).
    pub fn l1(&self) -> (u32, u32) {
        (self.cols, self.rows)
    }

    /// TIPSv2 grid (cols, rows).
    pub fn l2(&self) -> (u32, u32) {
        (self.cols * self.l2_ratio, self.rows * self.l2_ratio)
    }

    /// SAM mask size (width, height).
    pub fn mask(&self) -> (u32, u32) {
        (self.cols * self.mask_per_cell, self.rows * self.mask_per_cell)
    }

    /// Frame pixels of one L1 patch (width, height).
    pub fn cell_px(&self) -> (f32, f32) {
        (self.width as f32 / self.cols as f32, self.height as f32 / self.rows as f32)
    }

    pub fn cell_of(&self, x: f32, y: f32) -> (u32, u32) {
        let (cw, ch) = self.cell_px();
        (((x / cw).floor().max(0.0) as u32).min(self.cols - 1), ((y / ch).floor().max(0.0) as u32).min(self.rows - 1))
    }

    /// [x0, y0, x1, y1] of the L1 patch (c, r).
    pub fn cell_rect(&self, c: u32, r: u32) -> [f32; 4] {
        let (cw, ch) = self.cell_px();
        [c as f32 * cw, r as f32 * ch, (c + 1) as f32 * cw, (r + 1) as f32 * ch]
    }

    /// [x0, y0, x1, y1] of the L2 patch (i, j) inside the L1 patch (c, r), i and j in 0..l2_ratio.
    pub fn sub_rect(&self, c: u32, r: u32, i: u32, j: u32) -> [f32; 4] {
        let k = self.l2_ratio as f32;
        let (cw, ch) = self.cell_px();
        let (sw, sh) = (cw / k, ch / k);
        let (x0, y0) = (c as f32 * cw + i as f32 * sw, r as f32 * ch + j as f32 * sh);
        [x0, y0, x0 + sw, y0 + sh]
    }

    fn l1_index(&self, c: u32, r: u32) -> usize {
        (r * self.cols + c) as usize
    }
}

/// The pixels of a map of `mw` × `mh` whose centres fall inside the frame rectangle `rect` of a `fw` × `fh` frame
/// (at least the one under the rectangle's centre).
fn pixels_in(rect: [f32; 4], fw: u32, fh: u32, mw: u32, mh: u32) -> impl Iterator<Item = (u32, u32)> {
    let (sx, sy) = (mw as f32 / fw.max(1) as f32, mh as f32 / fh.max(1) as f32);
    let span = |a: f32, b: f32, s: f32, n: u32| -> (u32, u32) {
        let lo = ((a * s - 0.5).ceil().max(0.0) as u32).min(n);
        let hi = ((b * s - 0.5).ceil().max(0.0) as u32).min(n);
        if hi > lo {
            (lo, hi)
        } else {
            let mid = ((0.5 * (a + b) * s).floor().max(0.0) as u32).min(n - 1);
            (mid, mid + 1)
        }
    };
    let (x0, x1) = span(rect[0], rect[2], sx, mw.max(1));
    let (y0, y1) = span(rect[1], rect[3], sy, mh.max(1));
    (y0..y1).flat_map(move |y| (x0..x1).map(move |x| (x, y)))
}

/// What the depth layer says about one L1 patch: the share of sky, of measured depth within `far_m`, and of measured
/// depth beyond it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CellGeo {
    pub sky: f32,
    pub near: f32,
    pub far: f32,
}

impl CellGeo {
    pub fn blocked(&self, cfg: &CellGateCfg) -> bool {
        let measured = self.near + self.far;
        self.sky > cfg.sky_max_frac || (measured > 0.0 && self.near / measured < cfg.min_near_frac)
    }
}

/// The camera that took the frame, for the ground plane and the horizon.
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub intr: &'a Intrinsics,
    pub pose: &'a Pose,
    pub height: Option<HeightFn<'a>>,
}

/// Pools the depth (and the ADE20K sky, when the depth model ran its segmentation head) over each L2 patch and the
/// L2 patches of each L1 patch. Without a segmentation, a pixel with no depth is sky when its ray points above the
/// horizon (with no camera, whenever it has no depth); below the horizon it is unknown, not sky.
pub fn cell_geometry(pyr: &Pyramid, depth: &DepthMap, seg: Option<&SegMap>, view: Option<View<'_>>, far_m: f32) -> Vec<CellGeo> {
    let (fw, fh) = (pyr.width, pyr.height);
    let k = pyr.l2_ratio;
    let mut out = Vec::with_capacity((pyr.cols * pyr.rows) as usize);
    for r in 0..pyr.rows {
        for c in 0..pyr.cols {
            let mut acc = CellGeo::default();
            for j in 0..k {
                for i in 0..k {
                    let rect = pyr.sub_rect(c, r, i, j);
                    let (mut n, mut near, mut far, mut void_sky) = (0u32, 0u32, 0u32, 0u32);
                    for (x, y) in pixels_in(rect, fw, fh, depth.width, depth.height) {
                        n += 1;
                        let d = depth.at(x, y);
                        if d.is_finite() && d > 0.0 {
                            if d <= far_m {
                                near += 1;
                            } else {
                                far += 1;
                            }
                        } else if seg.is_none() {
                            let (px, py) = ((x as f32 + 0.5) * fw as f32 / depth.width as f32, (y as f32 + 0.5) * fh as f32 / depth.height as f32);
                            if view.map_or(true, |v| v.intr.ray(v.pose.yaw, px, py)[1] >= 0.0) {
                                void_sky += 1;
                            }
                        }
                    }
                    let sky = match seg {
                        Some(s) => {
                            let (mut m, mut sky) = (0u32, 0u32);
                            for (x, y) in pixels_in(rect, fw, fh, s.width, s.height) {
                                m += 1;
                                sky += s.is_sky((y * s.width + x) as usize) as u32;
                            }
                            sky as f32 / m.max(1) as f32
                        }
                        None => void_sky as f32 / n.max(1) as f32,
                    };
                    acc.sky += sky;
                    acc.near += near as f32 / n.max(1) as f32;
                    acc.far += far as f32 / n.max(1) as f32;
                }
            }
            let kk = (k * k) as f32;
            out.push(CellGeo { sky: acc.sky / kk, near: acc.near / kk, far: acc.far / kk });
        }
    }
    out
}

/// L2 → L1 gating: the heat of every patch the depth layer rules out drops to the heat's floor (0 for the
/// probability scorings), and the points are found again on what is left. A label whose heat lies only on ruled-out
/// patches has no point any more. Returns how many patches were ruled out.
pub fn gate_points(points: &mut Vec<SemanticPoint>, pyr: &Pyramid, geo: &[CellGeo], cfg: &CellGateCfg) -> u32 {
    let blocked: Vec<bool> = geo.iter().map(|g| g.blocked(cfg)).collect();
    let n = blocked.iter().filter(|b| **b).count() as u32;
    if n == 0 || points.is_empty() {
        return n;
    }
    let heats: Vec<(String, Heat, f32)> = points
        .iter()
        .map(|p| {
            let h = &p.heat;
            let floor = h.values.iter().copied().filter(|v| v.is_finite()).fold(0.0f32, f32::min);
            let mut values = h.values.clone();
            for r in 0..h.rows {
                for c in 0..h.cols {
                    let (x, y) = ((c as f32 + 0.5) / h.cols as f32 * pyr.width as f32, (r as f32 + 0.5) / h.rows as f32 * pyr.height as f32);
                    let (lc, lr) = pyr.cell_of(x, y);
                    if blocked[pyr.l1_index(lc, lr)] {
                        values[(r * h.cols + c) as usize] = floor;
                    }
                }
            }
            (p.label.clone(), Heat { cols: h.cols, rows: h.rows, values }, p.presence)
        })
        .collect();
    *points = points_with_presence(pyr.width, pyr.height, heats);
    n
}

/// One L2 patch: its median depth, relative spread (interquartile range over the median), the share of its pixels
/// with depth, its foreground pixels' centroid, and the ground plane's depth at its centre.
#[derive(Clone, Copy, Debug)]
struct Sub {
    median: f32,
    spread: f32,
    valid: f32,
    at: (f32, f32),
    ground: Option<f32>,
}

fn sub_patch(pyr: &Pyramid, depth: &DepthMap, rect: [f32; 4], view: Option<View<'_>>, edge_gap: f32) -> Sub {
    let (fw, fh) = (pyr.width, pyr.height);
    let mut vals: Vec<(f32, u32, u32)> = Vec::new();
    let mut n = 0u32;
    for (x, y) in pixels_in(rect, fw, fh, depth.width, depth.height) {
        n += 1;
        let d = depth.at(x, y);
        if d.is_finite() && d > 0.0 {
            vals.push((d, x, y));
        }
    }
    let centre = (0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3]));
    let ground = view.and_then(|v| ground_depth(v.intr, v.pose, centre.0, centre.1, v.height, f32::INFINITY));
    if vals.is_empty() {
        return Sub { median: f32::INFINITY, spread: f32::INFINITY, valid: 0.0, at: centre, ground };
    }
    let mut ds: Vec<f32> = vals.iter().map(|v| v.0).collect();
    ds.sort_by(f32::total_cmp);
    let q = |f: f32| ds[((ds.len() - 1) as f32 * f).round() as usize];
    let median = q(0.5);
    let band = median * (1.0 + 0.5 * edge_gap);
    let (mut sx, mut sy, mut m) = (0.0f32, 0.0f32, 0u32);
    for &(d, x, y) in &vals {
        if d <= band {
            sx += (x as f32 + 0.5) * fw as f32 / depth.width as f32;
            sy += (y as f32 + 0.5) * fh as f32 / depth.height as f32;
            m += 1;
        }
    }
    let at = if m > 0 { (sx / m as f32, sy / m as f32) } else { centre };
    Sub { median, spread: (q(0.75) - q(0.25)) / median.max(1e-6), valid: vals.len() as f32 / n.max(1) as f32, at, ground }
}

/// L1 → L2 peak snapping: for each point, the L2 patches inside its L1 patch are compared by their depth relative
/// to the ground plane's depth there (raw depth when any of them looks above the horizon, or without a camera). When
/// the point's own L2 patch has no usable depth, or reads farther than the nearest steady sub-patch by more than
/// `edge_gap`, the point moves onto that sub-patch's foreground pixels. Returns how many points moved.
pub fn snap_points(points: &mut [SemanticPoint], pyr: &Pyramid, depth: &DepthMap, view: Option<View<'_>>, cfg: &SnapCfg) -> u32 {
    let k = pyr.l2_ratio;
    if k < 2 {
        return 0;
    }
    let mut moved = 0;
    for p in points.iter_mut() {
        let (c, r) = pyr.cell_of(p.px, p.py);
        let subs: Vec<Sub> = (0..k * k).map(|s| sub_patch(pyr, depth, pyr.sub_rect(c, r, s % k, s / k), view, cfg.edge_gap)).collect();
        let relative = subs.iter().all(|s| s.ground.is_some_and(|g| g > 0.0));
        let key = |s: &Sub| if relative { s.median / s.ground.unwrap_or(1.0) } else { s.median };
        let own = {
            let [x0, y0, ..] = pyr.cell_rect(c, r);
            let (sw, sh) = (pyr.cell_px().0 / k as f32, pyr.cell_px().1 / k as f32);
            let (i, j) = ((((p.px - x0) / sw).floor().max(0.0) as u32).min(k - 1), (((p.py - y0) / sh).floor().max(0.0) as u32).min(k - 1));
            (j * k + i) as usize
        };
        let steady = |s: &Sub| s.valid >= cfg.min_valid_frac && s.spread <= cfg.max_rel_spread && s.median.is_finite();
        let Some((fg, best)) = subs.iter().enumerate().filter(|(_, s)| steady(s)).min_by(|a, b| key(a.1).total_cmp(&key(b.1))) else { continue };
        if fg == own {
            continue;
        }
        let mine = &subs[own];
        let off_edge = mine.valid < cfg.min_valid_frac || !mine.median.is_finite() || key(mine) > key(best) * (1.0 + cfg.edge_gap);
        if off_edge {
            (p.px, p.py) = best.at;
            moved += 1;
        }
    }
    moved
}

/// What the pyramid did to one frame, for the keyframe record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PyramidReport {
    pub l1: (u32, u32),
    pub l2: (u32, u32),
    pub mask: (u32, u32),
    pub gated_cells: u32,
    pub snapped: u32,
}

impl PyramidReport {
    pub fn of(pyr: &Pyramid) -> PyramidReport {
        PyramidReport { l1: pyr.l1(), l2: pyr.l2(), mask: pyr.mask(), gated_cells: 0, snapped: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;

    fn heat_point(label: &str, pyr: &Pyramid, peak: (u32, u32), elsewhere: f32) -> SemanticPoint {
        let (cols, rows) = pyr.l1();
        let mut values = vec![elsewhere; (cols * rows) as usize];
        values[(peak.1 * cols + peak.0) as usize] = 0.9;
        let heat = Heat { cols, rows, values };
        points_with_presence(pyr.width, pyr.height, vec![(label.into(), heat, 0.9)]).pop().unwrap()
    }

    fn cam() -> Intrinsics {
        Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0, roll_deg: 0.0 }
    }

    fn pose() -> Pose {
        Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 1.14 }
    }

    /// The ground plane's depth at every pixel (sky above the horizon), `box_` drawn nearer at `box_m`.
    fn scene(intr: &Intrinsics, pose: &Pose, box_: Option<([u32; 4], f32)>) -> DepthMap {
        let (w, h) = (intr.width, intr.height);
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let g = ground_depth(intr, pose, x as f32 + 0.5, y as f32 + 0.5, None, f32::INFINITY).unwrap_or(f32::INFINITY);
                let d = match box_ {
                    Some((b, m)) if x >= b[0] && x < b[2] && y >= b[1] && y < b[3] => m,
                    _ => g,
                };
                data.push(d);
            }
        }
        DepthMap { width: w, height: h, kind: DepthKind::Metric, data }
    }

    #[test]
    fn the_three_layers_are_whole_multiples_of_one_grid() {
        let cfg = PyramidCfg::default();
        let p = Pyramid::for_frame(384, 256, &cfg);
        assert_eq!((p.l1(), p.l2(), p.mask()), ((18, 12), (36, 24), (288, 192)));
        let (tc, tr) = crate::tips::InputGrid::default().grid_for(384, 256);
        assert_eq!(p.l2(), (tc as u32, tr as u32), "TIPSv2 keeps its own keyframe grid");
        let q = Pyramid::for_frame(640, 480, &cfg);
        assert_eq!((q.l1(), q.l2(), q.mask()), ((16, 12), (32, 24), (256, 192)));
        assert_eq!(q.l1().0 * 4, crate::sam::EMBED_SIDE as u32, "16 x 12 : 32 x 24 : SAM's 64 x 48");
        assert_eq!(Pyramid::for_frame(384, 256, &PyramidCfg::fast()).l2(), (18, 12));
        assert_eq!(Pyramid::for_frame(96, 64, &cfg).mask(), (90, 60), "a small frame's mask stays at most one pixel per frame pixel");
        let (cw, ch) = p.cell_px();
        assert!((cw - 384.0 / 18.0).abs() < 1e-4 && (ch - 256.0 / 12.0).abs() < 1e-4);
        for (c, r) in [(0, 0), (17, 11), (5, 7)] {
            let cell = p.cell_rect(c, r);
            let (a, b) = (p.sub_rect(c, r, 0, 0), p.sub_rect(c, r, 1, 1));
            assert!((a[0] - cell[0]).abs() < 1e-3 && (b[3] - cell[3]).abs() < 1e-3, "the 2 x 2 L2 patches tile the L1 patch exactly");
            let (x, y) = (0.5 * (cell[0] + cell[2]), 0.5 * (cell[1] + cell[3]));
            assert_eq!(p.cell_of(x, y), (c, r));
            let mpc = p.mask_per_cell as f32;
            assert!((cell[0] / p.width as f32 * p.mask().0 as f32 - c as f32 * mpc).abs() < 1e-3, "a patch is a whole mask block");
        }
    }

    #[test]
    fn sky_patches_lose_their_heat_and_the_peak_moves_to_the_ground() {
        let (intr, pose) = (cam(), pose());
        let pyr = Pyramid::for_frame(384, 256, &PyramidCfg::default());
        let depth = scene(&intr, &pose, None);
        let view = View { intr: &intr, pose: &pose, height: None };
        let geo = cell_geometry(&pyr, &depth, None, Some(view), 18.0);
        let cfg = CellGateCfg::default();
        assert!(geo[pyr.l1_index(9, 0)].blocked(&cfg), "the top row looks at the sky: {:?}", geo[pyr.l1_index(9, 0)]);
        assert!(!geo[pyr.l1_index(9, 11)].blocked(&cfg), "the bottom row is ground at a metre or two");
        let mut pts = vec![heat_point("tree", &pyr, (9, 0), 0.1)];
        pts[0].heat.values[(8 * 18 + 4) as usize] = 0.5;
        let n = gate_points(&mut pts, &pyr, &geo, &cfg);
        assert!(n > 0);
        assert_eq!(pts.len(), 1);
        assert_eq!(pyr.cell_of(pts[0].px, pts[0].py), (4, 8), "the next best patch on the ground wins");
        assert!(pts[0].heat.values[9] < 1e-6, "the sky patch's heat is gone");
        let mut only_sky = vec![heat_point("cloud", &pyr, (3, 0), 0.0)];
        gate_points(&mut only_sky, &pyr, &geo, &cfg);
        assert!(only_sky.is_empty(), "a label seen only in the sky has no point");
        let off = CellGateCfg { enabled: true, sky_max_frac: 1.1, min_near_frac: 0.0, ..cfg };
        let mut kept = vec![heat_point("tree", &pyr, (9, 0), 0.1)];
        assert_eq!(gate_points(&mut kept, &pyr, &geo, &off), 0);
        assert_eq!(pyr.cell_of(kept[0].px, kept[0].py), (9, 0));
    }

    #[test]
    fn a_segmentation_names_the_sky_and_unknown_ground_is_not_sky() {
        let (intr, pose) = (cam(), pose());
        let pyr = Pyramid::for_frame(384, 256, &PyramidCfg::default());
        let mut depth = scene(&intr, &pose, None);
        for y in 100..140 {
            for x in 0..384 {
                depth.data[(y * 384 + x) as usize] = f32::INFINITY;
            }
        }
        let view = View { intr: &intr, pose: &pose, height: None };
        let geo = cell_geometry(&pyr, &depth, None, Some(view), 18.0);
        let (_, r) = pyr.cell_of(10.0, 120.0);
        assert!(geo[pyr.l1_index(0, r)].sky < 0.01, "saturated depth below the horizon is far ground, not sky");
        assert!(!geo[pyr.l1_index(0, r)].blocked(&CellGateCfg::default()));
        let seg = SegMap { width: 2, height: 2, labels: vec![crate::tips::ADE_SKY as u8, 9, 9, 9], sky: vec![0.9, 0.0, 0.0, 0.0] };
        let geo = cell_geometry(&pyr, &depth, Some(&seg), Some(view), 18.0);
        assert!(geo[pyr.l1_index(0, 0)].sky > 0.99 && geo[pyr.l1_index(17, 0)].sky < 0.01, "the segmentation, not the depth, says where the sky is");
    }

    #[test]
    fn far_patches_are_ruled_out_beyond_far_m() {
        let pyr = Pyramid::for_frame(96, 64, &PyramidCfg::default());
        let depth = DepthMap { width: 96, height: 64, kind: DepthKind::Metric, data: (0..96 * 64).map(|i| if i % 96 < 48 { 30.0 } else { 4.0 }).collect() };
        let geo = cell_geometry(&pyr, &depth, None, None, 18.0);
        let cfg = CellGateCfg::default();
        assert!(geo[0].blocked(&cfg) && geo[0].far > 0.99);
        assert!(!geo[(pyr.cols - 1) as usize].blocked(&cfg));
    }

    #[test]
    fn a_peak_on_the_background_beside_an_object_snaps_onto_it() {
        let (intr, pose) = (cam(), pose());
        let pyr = Pyramid::for_frame(384, 256, &PyramidCfg::default());
        let (c, r) = (9u32, 8u32);
        let cell = pyr.cell_rect(c, r);
        let left = pyr.sub_rect(c, r, 0, 0);
        let b = [left[0] as u32 + 1, left[1] as u32 + 1, left[2] as u32, (cell[3] as u32).min(255)];
        let g = ground_depth(&intr, &pose, b[0] as f32, b[1] as f32, None, f32::INFINITY).unwrap();
        let depth = scene(&intr, &pose, Some((b, 0.6 * g)));
        let view = View { intr: &intr, pose: &pose, height: None };
        let right = pyr.sub_rect(c, r, 1, 0);
        let mut pts = vec![SemanticPoint { px: 0.5 * (right[0] + right[2]), py: 0.5 * (right[1] + right[3]), ..heat_point("stone", &pyr, (c, r), 0.1) }];
        let n = snap_points(&mut pts, &pyr, &depth, Some(view), &SnapCfg::default());
        assert_eq!(n, 1);
        let (x, y) = (pts[0].px, pts[0].py);
        assert!(x >= b[0] as f32 && x < b[2] as f32 && y >= b[1] as f32 && y < b[3] as f32, "the point lies on the stone now: {x}, {y} in {b:?}");
        assert!(depth.at_scaled(x, y, 384, 256) < 0.7 * g);
    }

    #[test]
    fn the_ground_slope_inside_a_patch_is_not_an_edge() {
        let (intr, pose) = (cam(), pose());
        let pyr = Pyramid::for_frame(384, 256, &PyramidCfg::default());
        let depth = scene(&intr, &pose, None);
        let view = View { intr: &intr, pose: &pose, height: None };
        let mut moved = 0;
        for r in 6..12 {
            for c in [2u32, 9, 15] {
                let rect = pyr.sub_rect(c, r, 1, 0);
                let mut pts = vec![SemanticPoint { px: 0.5 * (rect[0] + rect[2]), py: 0.5 * (rect[1] + rect[3]), ..heat_point("stone", &pyr, (c, r), 0.1) }];
                moved += snap_points(&mut pts, &pyr, &depth, Some(view), &SnapCfg::default());
            }
        }
        assert_eq!(moved, 0, "no ground-only patch pulls its peak downhill");
        let mut pts = vec![heat_point("stone", &pyr, (9, 9), 0.1)];
        assert_eq!(snap_points(&mut pts, &pyr, &depth, Some(view), &SnapCfg::default()), 0);
        let fast = Pyramid::for_frame(384, 256, &PyramidCfg::fast());
        assert_eq!(snap_points(&mut pts, &fast, &depth, Some(view), &SnapCfg::default()), 0, "1:1 has no sub-patches");
    }
}
