use serde::{Deserialize, Serialize};

use crate::kinds::Ground;
use crate::rng::{fbm, hash2, Rng};

pub const CELL_M: f32 = 0.5;
pub const HEIGHT_QUANT_CM: i32 = 5;
pub const MAX_NEIGHBOR_DIFF_CM: i32 = 20;

pub type FlatZone = (f32, f32, f32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerrainKind {
    #[default]
    Open,
    Rugged,
}

impl TerrainKind {
    pub fn id(self) -> &'static str {
        match self {
            TerrainKind::Open => "open",
            TerrainKind::Rugged => "rugged",
        }
    }

    pub fn from_id(id: &str) -> Option<TerrainKind> {
        match id {
            "open" => Some(TerrainKind::Open),
            "rugged" => Some(TerrainKind::Rugged),
            _ => None,
        }
    }
}

pub const MESA_CM: i32 = 160;
pub const RIDGE_CM: i32 = 140;
pub const RAMP_RISE_CM: i32 = 15;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RuggedLayout {
    pub mesas: Vec<(f32, f32, f32, bool)>,
    pub ridges: Vec<((f32, f32), (f32, f32))>,
    pub water_cells: usize,
}

pub const RAMP_SMOOTH_MAX_CM: i32 = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CurbKind {
    #[default]
    Curb = 0,
    Step = 1,
    Floor = 2,
    Ramp = 3,
}

impl CurbKind {
    pub fn from_u8(v: u8) -> CurbKind {
        match v {
            1 => CurbKind::Step,
            2 => CurbKind::Floor,
            3 => CurbKind::Ramp,
            _ => CurbKind::Curb,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Terrain {
    pub w: usize,
    pub d: usize,
    heights_cm: Vec<i16>,
    curb_cm: Vec<u8>,
    ground: Vec<Ground>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    curb_kind: Vec<u8>,
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Terrain {
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
            curb_kind: Vec::new(),
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

    pub fn curb_kind(&self, cx: usize, cz: usize) -> CurbKind {
        if self.curb_kind.is_empty() {
            return CurbKind::Curb;
        }
        CurbKind::from_u8(self.curb_kind[self.idx(cx, cz)])
    }

    pub fn set_curb_kind(&mut self, cx: usize, cz: usize, k: CurbKind) {
        if self.curb_kind.is_empty() {
            if k == CurbKind::Curb {
                return;
            }
            self.curb_kind = vec![CurbKind::Curb as u8; self.w * self.d];
        }
        let i = self.idx(cx, cz);
        self.curb_kind[i] = k as u8;
    }

    pub fn curb_kind_codes(&self) -> &[u8] {
        &self.curb_kind
    }

    pub fn ramp_rise_m(&self, a: (usize, usize), b: (usize, usize)) -> Option<f32> {
        if self.curb_kind.is_empty() {
            return None;
        }
        let ramp = self.curb_kind(a.0, a.1) == CurbKind::Ramp || self.curb_kind(b.0, b.1) == CurbKind::Ramp;
        let d = self.curb_cm[self.idx(b.0, b.1)] as i32 - self.curb_cm[self.idx(a.0, a.1)] as i32;
        (ramp && d != 0 && d.abs() <= RAMP_SMOOTH_MAX_CM).then(|| d as f32 / 100.0)
    }

    pub fn set_height_cm(&mut self, cx: usize, cz: usize, cm: i32) {
        let q = (cm.max(0) as f32 / HEIGHT_QUANT_CM as f32).round() as i32 * HEIGHT_QUANT_CM;
        let i = self.idx(cx, cz);
        self.heights_cm[i] = q as i16;
    }

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

    pub fn flatten_rect(&mut self, x0: usize, z0: usize, x1: usize, z1: usize) {
        for cz in z0..=z1.min(self.d - 1) {
            for cx in x0..=x1.min(self.w - 1) {
                let i = self.idx(cx, cz);
                self.heights_cm[i] = 0;
                self.curb_cm[i] = 0;
            }
        }
    }

    pub fn max_top_m(&self) -> f32 {
        (0..self.w * self.d).map(|i| (self.heights_cm[i] as i32 + self.curb_cm[i] as i32) as f32 / 100.0).fold(0.0, f32::max)
    }

    pub fn is_water(&self, cx: usize, cz: usize) -> bool {
        self.ground[self.idx(cx, cz)].is_water()
    }

    pub fn stamp_rugged(&mut self, seed: u64, clear: &[FlatZone], polyline: &[(f32, f32)], corridor_m: f32) -> RuggedLayout {
        let (w, d) = (self.w, self.d);
        let (size_x, size_z) = self.size_m();
        let span = size_x.min(size_z);
        let mut layout = RuggedLayout::default();
        let mut fixed = vec![false; w * d];
        let near_polyline = |x: f32, z: f32, r: f32| polyline.windows(2).any(|s| seg_dist(x, z, s[0], s[1]) < r);
        let near_clear = |x: f32, z: f32, r: f32| clear.iter().any(|&(cx, cz, cr)| ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() < cr + r);

        let corner = (size_x * 1.02, -0.02 * size_z);
        let reach = 0.36 * span;
        for cz in 0..d {
            for cx in 0..w {
                let (x, z) = cell_center(cx, cz);
                let coast = reach + 0.12 * span * (fbm(seed ^ 0x5EA, x * 0.18, z * 0.18, 2) - 0.5);
                let dist = ((x - corner.0).powi(2) + (z - corner.1).powi(2)).sqrt();
                if dist < coast && !near_clear(x, z, 1.0) && !near_polyline(x, z, corridor_m) {
                    let i = self.idx(cx, cz);
                    self.heights_cm[i] = 0;
                    self.curb_cm[i] = 0;
                    self.ground[i] = Ground::Water;
                    fixed[i] = true;
                    layout.water_cells += 1;
                }
            }
        }
        let shore = self.water_distance_m(8.0);
        for i in 0..w * d {
            let (x, z) = cell_center(i % w, i / w);
            if fixed[i] || shore[i] > 2.5 || near_polyline(x, z, corridor_m) {
                continue;
            }
            let h = self.heights_cm[i] as f32 * (shore[i] / 2.5);
            self.heights_cm[i] = ((h / HEIGHT_QUANT_CM as f32).round() as i32 * HEIGHT_QUANT_CM) as i16;
            if self.curb_cm[i] == 0 {
                self.ground[i] = if shore[i] <= 1.05 { Ground::WetSand } else { Ground::Sand };
            }
        }
        let shore_at = |x: f32, z: f32| -> f32 {
            let (cx, cz) = (((x / CELL_M) as usize).min(w - 1), ((z / CELL_M) as usize).min(d - 1));
            shore[cz * w + cx]
        };
        let mut rng = Rng::new(seed ^ 0xC11F_F5EA_0BAD_CAFE);
        let mut tries = 0;
        while layout.mesas.len() < 2 && tries < 400 {
            tries += 1;
            let r = rng.range_f32(2.2, 3.4);
            let (x, z) = (rng.range_f32(0.18, 0.82) * size_x, rng.range_f32(0.18, 0.82) * size_z);
            let water_near = shore_at(x, z) < r + 2.0;
            let other_near = layout.mesas.iter().any(|&(mx, mz, mr, _)| ((x - mx).powi(2) + (z - mz).powi(2)).sqrt() < r + mr + 3.0);
            if water_near || other_near || near_clear(x, z, r + 2.5) || near_polyline(x, z, r + corridor_m) {
                continue;
            }
            let ramp = layout.mesas.is_empty();
            for cz in 0..d {
                for cx in 0..w {
                    let (px, pz) = cell_center(cx, cz);
                    let rim = r + 0.45 * (hash2(seed ^ 0x3E5A, cx as i32, cz as i32) & 0xFF) as f32 / 255.0;
                    if ((px - x).powi(2) + (pz - z).powi(2)).sqrt() <= rim {
                        let i = self.idx(cx, cz);
                        self.heights_cm[i] = MESA_CM as i16;
                        self.curb_cm[i] = 0;
                        self.ground[i] = Ground::Rock;
                        fixed[i] = true;
                    }
                }
            }
            if ramp {
                self.stamp_ramp(x, z, r, rng.range_f32(0.0, std::f32::consts::TAU), &mut fixed);
            }
            layout.mesas.push((x, z, r, ramp));
        }
        let mut tries = 0;
        while layout.ridges.len() < 3 && tries < 600 {
            tries += 1;
            let (x0, z0) = (rng.range_f32(0.12, 0.88) * size_x, rng.range_f32(0.12, 0.88) * size_z);
            let a = rng.range_f32(0.0, std::f32::consts::TAU);
            let len = rng.range_f32(5.0, 9.0);
            let (x1, z1) = (x0 + len * a.cos(), z0 + len * a.sin());
            let cells = line_cells(x0, z0, x1, z1);
            let ok = !cells.is_empty()
                && cells.iter().all(|&(cx, cz)| {
                    if cx >= w || cz >= d {
                        return false;
                    }
                    let (px, pz) = cell_center(cx, cz);
                    let i = cz * w + cx;
                    !fixed[i]
                        && self.curb_cm[i] == 0
                        && !near_clear(px, pz, 2.0)
                        && !near_polyline(px, pz, corridor_m)
                        && layout.mesas.iter().all(|&(mx, mz, mr, _)| ((px - mx).powi(2) + (pz - mz).powi(2)).sqrt() > mr + 2.0)
                        && shore[i] > 2.0
                });
            if !ok {
                continue;
            }
            for (cx, cz) in cells {
                let i = cz * w + cx;
                self.heights_cm[i] = (self.heights_cm[i] as i32 + RIDGE_CM) as i16;
                self.ground[i] = Ground::Rock;
                fixed[i] = true;
            }
            layout.ridges.push(((x0, z0), (x1, z1)));
        }
        self.relax_except(&fixed);
        layout
    }

    pub fn water_distance_m(&self, cap_m: f32) -> Vec<f32> {
        let (w, d) = (self.w, self.d);
        let mut dist = vec![cap_m; w * d];
        let mut queue = std::collections::VecDeque::new();
        for i in 0..w * d {
            if self.ground[i].is_water() {
                dist[i] = 0.0;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            let (cx, cz) = ((i % w) as i32, (i / w) as i32);
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (nx, nz) = (cx + dx, cz + dz);
                if !self.in_bounds(nx, nz) {
                    continue;
                }
                let j = nz as usize * w + nx as usize;
                let step = if dx != 0 && dz != 0 { CELL_M * std::f32::consts::SQRT_2 } else { CELL_M };
                let nd = dist[i] + step;
                if nd < dist[j] {
                    dist[j] = nd;
                    queue.push_back(j);
                }
            }
        }
        dist
    }

    fn stamp_ramp(&mut self, x: f32, z: f32, r: f32, heading: f32, fixed: &mut [bool]) {
        let (dx, dz) = (heading.cos(), heading.sin());
        let (nx, nz) = (-dz, dx);
        let mut h = MESA_CM;
        let mut s = r + 0.6;
        while h > 0 {
            h -= RAMP_RISE_CM;
            for off in [-0.25f32, 0.25] {
                let (px, pz) = (x + dx * s + nx * off, z + dz * s + nz * off);
                let Some((cx, cz)) = self.cell_of(px, pz) else { return };
                let i = self.idx(cx, cz);
                if self.ground[i].is_water() {
                    return;
                }
                let ground = self.heights_cm[i] as i32;
                if h <= ground {
                    continue;
                }
                self.heights_cm[i] = h as i16;
                self.curb_cm[i] = 0;
                self.ground[i] = Ground::Rock;
                fixed[i] = true;
            }
            s += CELL_M;
        }
    }

    fn relax_except(&mut self, fixed: &[bool]) {
        let (w, d) = (self.w, self.d);
        for _ in 0..8 {
            let mut changed = false;
            for cz in 0..d {
                for cx in 0..w {
                    let i = cz * w + cx;
                    if fixed[i] {
                        continue;
                    }
                    for (nx, nz) in neighbors4(cx, cz, w, d) {
                        let j = nz * w + nx;
                        if fixed[j] {
                            continue;
                        }
                        if self.heights_cm[i] as i32 - self.heights_cm[j] as i32 > MAX_NEIGHBOR_DIFF_CM {
                            self.heights_cm[i] = (self.heights_cm[j] as i32 + MAX_NEIGHBOR_DIFF_CM) as i16;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
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

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        match self.cell_of(x, z) {
            Some((cx, cz)) => self.height_m(cx, cz),
            None => 0.0,
        }
    }

    pub fn surface_smooth_m(&self, x: f32, z: f32, base_m: f32, step_m: f32) -> f32 {
        let fx = (x / CELL_M - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fz = (z / CELL_M - 0.5).clamp(0.0, (self.d - 1) as f32);
        let (x0, z0) = (fx.floor() as usize, fz.floor() as usize);
        let (x1, z1) = ((x0 + 1).min(self.w - 1), (z0 + 1).min(self.d - 1));
        let (u, v) = (fx - x0 as f32, fz - z0 as f32);
        let h = |a: usize, b: usize| {
            let s = self.height_m(a, b) + self.curb_m(a, b);
            if (s - base_m).abs() > step_m {
                base_m
            } else {
                s
            }
        };
        let near = h(x0, z0) * (1.0 - u) + h(x1, z0) * u;
        let far = h(x0, z1) * (1.0 - u) + h(x1, z1) * u;
        near * (1.0 - v) + far * v
    }

    pub fn slope_deg(&self, a: (usize, usize), b: (usize, usize)) -> f32 {
        let (ax, az) = cell_center(a.0, a.1);
        let (bx, bz) = cell_center(b.0, b.1);
        let run = ((ax - bx).powi(2) + (az - bz).powi(2)).sqrt();
        let rise = (self.height_m(a.0, a.1) - self.height_m(b.0, b.1)).abs();
        rise.atan2(run).to_degrees()
    }
}

pub fn cell_center(cx: usize, cz: usize) -> (f32, f32) {
    ((cx as f32 + 0.5) * CELL_M, (cz as f32 + 0.5) * CELL_M)
}

pub fn seg_dist(x: f32, z: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (vx, vz) = (b.0 - a.0, b.1 - a.1);
    let len2 = vx * vx + vz * vz;
    let t = if len2 < 1e-9 { 0.0 } else { (((x - a.0) * vx + (z - a.1) * vz) / len2).clamp(0.0, 1.0) };
    ((x - a.0 - t * vx).powi(2) + (z - a.1 - t * vz).powi(2)).sqrt()
}

fn line_cells(x0: f32, z0: f32, x1: f32, z1: f32) -> Vec<(usize, usize)> {
    if x0.min(x1) < 0.0 || z0.min(z1) < 0.0 {
        return vec![];
    }
    let n = (((x1 - x0).abs().max((z1 - z0).abs())) / (CELL_M * 0.25)).ceil().max(1.0) as usize;
    let mut out: Vec<(usize, usize)> = Vec::new();
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let c = (((x0 + (x1 - x0) * t) / CELL_M) as usize, ((z0 + (z1 - z0) * t) / CELL_M) as usize);
        if let Some(&last) = out.last() {
            if last == c {
                continue;
            }
            if last.0 != c.0 && last.1 != c.1 {
                out.push((c.0, last.1));
            }
        }
        out.push(c);
    }
    out
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
        assert!((deg - (0.2f32 / 0.5).atan().to_degrees()).abs() < 1e-3);
    }
}
