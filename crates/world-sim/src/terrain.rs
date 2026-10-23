//! Heightfield terrain on a square cell grid.
//!
//! Heights are stored in centimetres, quantised to 5 cm, and rendered as flat
//! topped columns (a blocky look). Smooth height changes between neighbouring
//! cells are judged as slopes. Sharp obstacles are separate "curbs": a curb
//! sits on top of a cell and is judged as a step. That mirrors the two kinds of
//! terrain segment in `hand_core::Terrain` (`Slope` and `Step`).

use serde::{Deserialize, Serialize};

use crate::kinds::Ground;
use crate::rng::{fbm, hash2};

/// Edge length of one cell in metres.
pub const CELL_M: f32 = 0.5;
/// Height quantisation in centimetres.
pub const HEIGHT_QUANT_CM: i32 = 5;
/// Largest height difference allowed between neighbouring cells, centimetres.
pub const MAX_NEIGHBOR_DIFF_CM: i32 = 20;

/// A circular area flattened to ground level, in metres: `(x, z, radius)`.
pub type FlatZone = (f32, f32, f32);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terrain {
    pub w: usize,
    pub d: usize,
    heights_cm: Vec<i16>,
    curb_cm: Vec<u8>,
    ground: Vec<Ground>,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Terrain {
    /// Builds terrain from `seed`. Cells inside a flat zone are at height 0 and
    /// blend into the hills over 0.6 radii.
    pub fn generate(seed: u64, w: usize, d: usize, zones: &[FlatZone]) -> Terrain {
        assert!(w >= 8 && d >= 8, "terrain must be at least 8x8 cells");
        let mut raw = vec![0i32; w * d];
        for cz in 0..d {
            for cx in 0..w {
                let n = fbm(seed, cx as f32 * 0.06, cz as f32 * 0.06, 3);
                let mut h = ((n - 0.35).max(0.0) * 160.0) as f32;
                let (x, z) = cell_center(cx, cz);
                for &(zx, zz, r) in zones {
                    let dist = ((x - zx).powi(2) + (z - zz).powi(2)).sqrt();
                    h *= smoothstep((dist - r) / (0.6 * r.max(0.1)));
                }
                raw[cz * w + cx] = (h / HEIGHT_QUANT_CM as f32).round() as i32 * HEIGHT_QUANT_CM;
            }
        }
        // Relax so neighbouring cells never differ by more than the limit.
        for _ in 0..8 {
            let mut changed = false;
            for cz in 0..d {
                for cx in 0..w {
                    let i = cz * w + cx;
                    for (nx, nz) in neighbors4(cx, cz, w, d) {
                        let j = nz * w + nx;
                        let diff = raw[i] - raw[j];
                        if diff > MAX_NEIGHBOR_DIFF_CM {
                            raw[i] = raw[j] + MAX_NEIGHBOR_DIFF_CM;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let heights_cm: Vec<i16> = raw.iter().map(|&h| h.max(0) as i16).collect();
        let mut ground = Vec::with_capacity(w * d);
        for cz in 0..d {
            for cx in 0..w {
                let h = heights_cm[cz * w + cx] as i32;
                let jitter = (hash2(seed, cx as i32, cz as i32) & 0xF) as i32;
                // Sand is only used for the waypoint pads (set by the world).
                ground.push(if h < 70 && jitter == 0 {
                    Ground::Dirt
                } else if h < 70 {
                    Ground::Grass
                } else if h < 100 {
                    Ground::Dirt
                } else {
                    Ground::Rock
                });
            }
        }
        Terrain {
            w,
            d,
            heights_cm,
            curb_cm: vec![0; w * d],
            ground,
        }
    }

    pub fn idx(&self, cx: usize, cz: usize) -> usize {
        cz * self.w + cx
    }

    pub fn in_bounds(&self, cx: i32, cz: i32) -> bool {
        cx >= 0 && cz >= 0 && (cx as usize) < self.w && (cz as usize) < self.d
    }

    pub fn size_m(&self) -> (f32, f32) {
        (self.w as f32 * CELL_M, self.d as f32 * CELL_M)
    }

    pub fn height_cm(&self, cx: usize, cz: usize) -> i32 {
        self.heights_cm[self.idx(cx, cz)] as i32
    }

    pub fn height_m(&self, cx: usize, cz: usize) -> f32 {
        self.height_cm(cx, cz) as f32 / 100.0
    }

    pub fn curb_m(&self, cx: usize, cz: usize) -> f32 {
        self.curb_cm[self.idx(cx, cz)] as f32 / 100.0
    }

    pub fn ground_at(&self, cx: usize, cz: usize) -> Ground {
        self.ground[self.idx(cx, cz)]
    }

    pub fn set_curb_cm(&mut self, cx: usize, cz: usize, cm: u8) {
        let i = self.idx(cx, cz);
        self.curb_cm[i] = cm;
    }

    pub fn set_ground(&mut self, cx: usize, cz: usize, g: Ground) {
        let i = self.idx(cx, cz);
        self.ground[i] = g;
    }

    /// Sets a cell height, quantised to the grid step. Used by scenario setup
    /// and tests; generated terrain already respects the neighbour limit.
    pub fn set_height_cm(&mut self, cx: usize, cz: usize, cm: i32) {
        let q = (cm.max(0) as f32 / HEIGHT_QUANT_CM as f32).round() as i32 * HEIGHT_QUANT_CM;
        let i = self.idx(cx, cz);
        self.heights_cm[i] = q as i16;
    }

    /// Adds a curb of `cm` centimetres along the border of the inclusive cell
    /// rectangle, leaving the inside open. Cells that already carry a taller
    /// curb keep it.
    pub fn add_curb_ring(&mut self, x0: usize, z0: usize, x1: usize, z1: usize, cm: u8) {
        for cz in z0..=z1.min(self.d - 1) {
            for cx in x0..=x1.min(self.w - 1) {
                let border = cx == x0 || cx == x1 || cz == z0 || cz == z1;
                if border {
                    let i = self.idx(cx, cz);
                    self.curb_cm[i] = self.curb_cm[i].max(cm);
                }
            }
        }
    }

    /// Flattens the inclusive cell rectangle to ground level and clears curbs.
    pub fn flatten_rect(&mut self, x0: usize, z0: usize, x1: usize, z1: usize) {
        for cz in z0..=z1.min(self.d - 1) {
            for cx in x0..=x1.min(self.w - 1) {
                let i = self.idx(cx, cz);
                self.heights_cm[i] = 0;
                self.curb_cm[i] = 0;
            }
        }
    }

    pub fn cell_of(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        if x < 0.0 || z < 0.0 {
            return None;
        }
        let (cx, cz) = ((x / CELL_M) as usize, (z / CELL_M) as usize);
        (cx < self.w && cz < self.d).then_some((cx, cz))
    }

    /// Ground height in metres under the point (the containing cell's height).
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        match self.cell_of(x, z) {
            Some((cx, cz)) => self.height_m(cx, cz),
            None => 0.0,
        }
    }

    /// Slope in degrees between the centres of two cells.
    pub fn slope_deg(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let (ax, az) = cell_center(a.0, a.1);
        let (bx, bz) = cell_center(b.0, b.1);
        let run = ((ax - bx).powi(2) + (az - bz).powi(2)).sqrt();
        let rise = (self.height_m(a.0, a.1) - self.height_m(b.0, b.1)).abs();
        rise.atan2(run).to_degrees()
    }
}

/// Centre of a cell in metres.
pub fn cell_center(cx: usize, cz: usize) -> (f32, f32) {
    ((cx as f32 + 0.5) * CELL_M, (cz as f32 + 0.5) * CELL_M)
}

fn neighbors4(cx: usize, cz: usize, w: usize, d: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut v = Vec::with_capacity(4);
    if cx > 0 {
        v.push((cx - 1, cz));
    }
    if cx + 1 < w {
        v.push((cx + 1, cz));
    }
    if cz > 0 {
        v.push((cx, cz - 1));
    }
    if cz + 1 < d {
        v.push((cx, cz + 1));
    }
    v.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Terrain {
        Terrain::generate(11, 64, 64, &[(4.5, 4.5, 4.0)])
    }

    #[test]
    fn same_seed_same_terrain_different_seed_differs() {
        let a = sample();
        let b = sample();
        assert_eq!(a.heights_cm, b.heights_cm);
        let c = Terrain::generate(12, 64, 64, &[(4.5, 4.5, 4.0)]);
        assert_ne!(a.heights_cm, c.heights_cm);
    }

    #[test]
    fn heights_are_quantised_and_neighbors_limited() {
        let t = sample();
        for cz in 0..t.d {
            for cx in 0..t.w {
                assert_eq!(t.height_cm(cx, cz) % HEIGHT_QUANT_CM, 0);
                if cx + 1 < t.w {
                    assert!((t.height_cm(cx, cz) - t.height_cm(cx + 1, cz)).abs() <= MAX_NEIGHBOR_DIFF_CM);
                }
                if cz + 1 < t.d {
                    assert!((t.height_cm(cx, cz) - t.height_cm(cx, cz + 1)).abs() <= MAX_NEIGHBOR_DIFF_CM);
                }
            }
        }
    }

    #[test]
    fn flat_zone_is_at_ground_level() {
        let t = sample();
        for cz in 0..t.d {
            for cx in 0..t.w {
                let (x, z) = cell_center(cx, cz);
                if ((x - 4.5).powi(2) + (z - 4.5).powi(2)).sqrt() <= 4.0 {
                    assert_eq!(t.height_cm(cx, cz), 0, "cell {cx},{cz} inside the zone is not flat");
                }
            }
        }
    }

    #[test]
    fn terrain_has_some_relief_outside_the_zone() {
        let t = sample();
        let max = (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).map(|(x, z)| t.height_cm(x, z)).max().unwrap();
        assert!(max >= 20, "terrain is too flat (max {max} cm)");
    }

    #[test]
    fn cell_lookup_round_trips() {
        let t = sample();
        let (x, z) = cell_center(10, 20);
        assert_eq!(t.cell_of(x, z), Some((10, 20)));
        assert_eq!(t.cell_of(-0.1, 1.0), None);
        let (mx, mz) = t.size_m();
        assert_eq!(t.cell_of(mx + 0.1, 1.0), None);
        assert_eq!(t.cell_of(1.0, mz + 0.1), None);
    }

    #[test]
    fn curb_ring_marks_only_the_border() {
        let mut t = sample();
        t.add_curb_ring(30, 30, 34, 34, 15);
        assert_eq!(t.curb_m(30, 32), 0.15);
        assert_eq!(t.curb_m(34, 34), 0.15);
        assert_eq!(t.curb_m(32, 32), 0.0);
        t.add_curb_ring(30, 30, 34, 34, 5);
        assert_eq!(t.curb_m(30, 32), 0.15, "a lower curb must not replace a taller one");
    }

    #[test]
    fn slope_between_cells_uses_centre_distance() {
        let mut t = sample();
        t.heights_cm[0] = 0;
        t.heights_cm[1] = 20;
        let deg = t.slope_deg((0, 0), (1, 0));
        // 0.20 m rise over 0.5 m run
        assert!((deg - (0.2f32 / 0.5).atan().to_degrees()).abs() < 1e-3);
    }
}
