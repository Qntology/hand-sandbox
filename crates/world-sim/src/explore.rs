use noto_core::BaseSpec;

use crate::nav::{self, Cell};
use crate::terrain::{cell_center, Terrain, CELL_M};

pub const SEARCH_RANGE_M: f32 = 14.0;
pub const TARGET_H_M: f32 = 0.3;
pub const LOOK_COST_S: f32 = 12.0;
pub const MIN_GAIN: usize = 16;
const RAY_STEP_M: f32 = 0.3;
const RAY_STEP_RAD: f32 = 0.035;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub x: f32,
    pub z: f32,
    pub height_m: f32,
    pub pitch_deg: f32,
    pub vfov_deg: f32,
    pub range_m: f32,
}

impl Eye {
    pub fn at(t: &Terrain, c: Cell, eye_above_m: f32, pitch_deg: f32, vfov_deg: f32) -> Eye {
        let (x, z) = cell_center(c.0, c.1);
        Eye { x, z, height_m: t.height_m(c.0, c.1) + t.curb_m(c.0, c.1) + eye_above_m, pitch_deg, vfov_deg, range_m: SEARCH_RANGE_M }
    }
}

pub fn viewshed(t: &Terrain, eye: &Eye, a0: f32, a1: f32, visit: &mut dyn FnMut(usize)) {
    let span = (a1 - a0).abs().max(1e-3);
    let n = (span / RAY_STEP_RAD).ceil().max(1.0) as usize;
    let lo = (eye.pitch_deg - eye.vfov_deg * 0.5).to_radians().tan();
    let hi = (eye.pitch_deg + eye.vfov_deg * 0.5).to_radians().tan();
    for k in 0..=n {
        let a = a0 + (a1 - a0) * k as f32 / n as f32;
        let (dx, dz) = (a.cos(), a.sin());
        let mut horizon = f32::NEG_INFINITY;
        let mut last = usize::MAX;
        let mut r = RAY_STEP_M;
        while r <= eye.range_m {
            let Some((cx, cz)) = t.cell_of(eye.x + dx * r, eye.z + dz * r) else { break };
            let i = t.idx(cx, cz);
            let top = if t.is_water(cx, cz) { 0.0 } else { t.height_m(cx, cz) + t.curb_m(cx, cz) };
            let low = (top - eye.height_m) / r;
            let high = (top + TARGET_H_M - eye.height_m) / r;
            if i != last && high >= horizon && low <= hi && high >= lo {
                visit(i);
            }
            horizon = horizon.max(low);
            last = i;
            r += RAY_STEP_M;
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Coverage {
    w: usize,
    d: usize,
    seen: Vec<bool>,
    land: usize,
    pub looks: u32,
}

impl Coverage {
    pub fn new(t: &Terrain) -> Coverage {
        let land = (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).filter(|&(x, z)| !t.is_water(x, z)).count();
        Coverage { w: t.w, d: t.d, seen: vec![false; t.w * t.d], land: land.max(1), looks: 0 }
    }

    pub fn is_seen(&self, cx: usize, cz: usize) -> bool {
        cx < self.w && cz < self.d && self.seen[cz * self.w + cx]
    }

    pub fn seen_cells(&self) -> usize {
        self.seen.iter().filter(|s| **s).count()
    }

    pub fn land_frac(&self, t: &Terrain) -> f32 {
        let n = (0..self.seen.len()).filter(|&i| self.seen[i] && !t.is_water(i % self.w, i / self.w)).count();
        n as f32 / self.land as f32
    }

    pub fn look(&mut self, t: &Terrain, eye: &Eye, yaw: f32, hfov_deg: f32) {
        let h = hfov_deg.to_radians() * 0.5;
        let seen = &mut self.seen;
        viewshed(t, eye, yaw - h, yaw + h, &mut |i| seen[i] = true);
        self.looks += 1;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Viewpoint {
    pub x: f32,
    pub z: f32,
    pub cell: Cell,
    pub gain: usize,
    pub weighted: f32,
    pub time_s: f32,
    pub score: f32,
    pub face_yaw: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn next_viewpoint(
    t: &Terrain,
    spec: &BaseSpec,
    blocked: &dyn Fn(usize, usize) -> bool,
    start: Cell,
    cov: &Coverage,
    eye_above_m: f32,
    pitch_deg: f32,
    vfov_deg: f32,
    range_m: f32,
    avoid: &[(f32, f32)],
) -> Option<Viewpoint> {
    next_viewpoint_with(t, spec, blocked, start, cov, eye_above_m, pitch_deg, vfov_deg, range_m, avoid, None)
}

#[allow(clippy::too_many_arguments)]
pub fn next_viewpoint_with(
    t: &Terrain,
    spec: &BaseSpec,
    blocked: &dyn Fn(usize, usize) -> bool,
    start: Cell,
    cov: &Coverage,
    eye_above_m: f32,
    pitch_deg: f32,
    vfov_deg: f32,
    range_m: f32,
    avoid: &[(f32, f32)],
    prior: Option<&[f32]>,
) -> Option<Viewpoint> {
    let weight = |i: usize| prior.and_then(|p| p.get(i)).copied().unwrap_or(1.0);
    let times = nav::travel_times(spec, t, blocked, start, 900.0);
    let mut cands: Vec<(usize, usize, f32)> = Vec::new();
    for cz in (1..t.d).step_by(3) {
        for cx in (1..t.w).step_by(3) {
            let tt = times[t.idx(cx, cz)];
            if !tt.is_finite() || tt < 1.0 || t.is_water(cx, cz) || blocked(cx, cz) {
                continue;
            }
            let (x, z) = cell_center(cx, cz);
            if avoid.iter().any(|&(ax, az)| (x - ax).hypot(z - az) < 2.0) {
                continue;
            }
            cands.push((cx, cz, tt));
        }
    }
    cands.sort_by(|a, b| a.2.total_cmp(&b.2));
    cands.truncate(220);
    let mut stamp = vec![u32::MAX; t.w * t.d];
    let mut best: Option<Viewpoint> = None;
    for (k, &(cx, cz, tt)) in cands.iter().enumerate() {
        let eye = Eye { range_m: range_m.min(SEARCH_RANGE_M), ..Eye::at(t, (cx, cz), eye_above_m, pitch_deg, vfov_deg) };
        let mut gain = 0usize;
        let mut weighted = 0.0f32;
        let mut sectors = [0.0f32; 8];
        let tag = k as u32;
        let w = t.w;
        viewshed(t, &eye, 0.0, std::f32::consts::TAU, &mut |i| {
            if stamp[i] != tag && !cov.seen[i] && !t.is_water(i % w, i / w) {
                stamp[i] = tag;
                gain += 1;
                let wi = weight(i);
                weighted += wi;
                let (px, pz) = cell_center(i % w, i / w);
                let a = (pz - eye.z).atan2(px - eye.x).rem_euclid(std::f32::consts::TAU);
                sectors[((a / std::f32::consts::FRAC_PI_4) as usize).min(7)] += wi;
            }
        });
        if gain < MIN_GAIN {
            continue;
        }
        let score = weighted / (tt + LOOK_COST_S);
        if best.as_ref().map_or(true, |b| score > b.score) {
            let s = (0..8).fold(0usize, |b, i| if sectors[i] >= sectors[b] { i } else { b });
            best = Some(Viewpoint {
                x: eye.x,
                z: eye.z,
                cell: (cx, cz),
                gain,
                weighted,
                time_s: tt,
                score,
                face_yaw: (s as f32 + 0.5) * std::f32::consts::FRAC_PI_4,
            });
        }
    }
    best
}

pub fn line_of_sight(t: &Terrain, a: (f32, f32), eye_above_m: f32, b: (f32, f32), target_h_m: f32) -> bool {
    let Some(ca) = t.cell_of(a.0, a.1) else { return false };
    let Some(cb) = t.cell_of(b.0, b.1) else { return false };
    let top = |c: Cell| if t.is_water(c.0, c.1) { 0.0 } else { t.height_m(c.0, c.1) + t.curb_m(c.0, c.1) };
    let (ha, hb) = (top(ca) + eye_above_m, top(cb) + target_h_m);
    let len = (b.0 - a.0).hypot(b.1 - a.1);
    let n = (len / (CELL_M * 0.5)).ceil().max(1.0) as usize;
    for k in 1..n {
        let f = k as f32 / n as f32;
        let Some(c) = t.cell_of(a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f) else { return false };
        if c == ca || c == cb {
            continue;
        }
        if top(c) > ha + (hb - ha) * f {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::BaseKind;

    fn flat(n: usize) -> Terrain {
        Terrain::generate(1, n, n, &[(n as f32 * 0.25, n as f32 * 0.25, n as f32)])
    }

    fn eye(x: f32, z: f32) -> Eye {
        Eye { x, z, height_m: 0.5, pitch_deg: -12.0, vfov_deg: 50.0, range_m: SEARCH_RANGE_M }
    }

    #[test]
    fn a_ridge_hides_what_is_behind_it() {
        let mut t = flat(48);
        let (cx, cz) = t.cell_of(4.0, 12.0).unwrap();
        let far = t.cell_of(10.0, 12.0).unwrap();
        let mut cov = Coverage::new(&t);
        cov.look(&t, &eye(4.0, 12.0), 0.0, 70.0);
        assert!(cov.is_seen(far.0, far.1), "open ground ahead is seen");
        for z in 0..48 {
            t.set_height_cm(cx + 6, z, 140);
        }
        let mut cov = Coverage::new(&t);
        cov.look(&t, &eye(4.0, 12.0), 0.0, 70.0);
        assert!(!cov.is_seen(far.0, far.1), "a 1.4 m ridge in between hides it");
        assert!(cov.is_seen(cx + 3, cz), "the near side is still seen");
        assert!(!line_of_sight(&t, (4.0, 12.0), 0.5, (10.0, 12.0), 0.3));
        assert!(line_of_sight(&t, (4.0, 12.0), 0.5, (6.0, 12.0), 0.3));
    }

    #[test]
    fn the_range_and_the_field_of_view_bound_a_look() {
        let t = flat(80);
        let mut cov = Coverage::new(&t);
        cov.look(&t, &eye(2.0, 20.0), 0.0, 70.0);
        let (bx, bz) = t.cell_of(2.0 + SEARCH_RANGE_M + 3.0, 20.0).unwrap();
        assert!(!cov.is_seen(bx, bz), "beyond the range");
        let (sx, sz) = t.cell_of(2.0, 26.0).unwrap();
        assert!(!cov.is_seen(sx, sz), "outside the field of view (90° to the side)");
        assert!(cov.land_frac(&t) > 0.0 && cov.land_frac(&t) < 0.2);
    }

    #[test]
    fn the_next_viewpoint_goes_around_a_ridge_to_unseen_ground() {
        let mut t = flat(48);
        for z in 4..48 {
            t.set_height_cm(24, z, 140);
        }
        let start = t.cell_of(6.0, 12.0).unwrap();
        let mut cov = Coverage::new(&t);
        for k in 0..6 {
            cov.look(&t, &Eye::at(&t, start, 0.5, -12.0, 50.0), k as f32 * std::f32::consts::FRAC_PI_3, 70.0);
        }
        let spec = BaseKind::Track.spec();
        let none = |_: usize, _: usize| false;
        let v = next_viewpoint(&t, &spec, &none, start, &cov, 0.5, -12.0, 50.0, SEARCH_RANGE_M, &[]).expect("unseen ground is left");
        assert!(v.gain >= MIN_GAIN && v.time_s > 0.0);
        let mut all = Coverage::new(&t);
        all.seen.iter_mut().for_each(|s| *s = true);
        assert!(next_viewpoint(&t, &spec, &none, start, &all, 0.5, -12.0, 50.0, SEARCH_RANGE_M, &[]).is_none());
    }
}
