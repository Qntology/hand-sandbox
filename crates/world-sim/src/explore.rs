//! Where to look from next: terrain line of sight, what the camera has
//! covered, and the next viewpoint when the target is hidden.
//!
//! The robot knows the terrain (a surveyed site map, the same one its routes
//! are planned on), not where the objects are. What it has already looked at
//! is a coverage grid: the cells its camera could see from each pose, with the
//! terrain occluding what lies behind a ridge or below a cliff edge and the
//! camera's own field of view and range. When a full turn shows nothing, the
//! next viewpoint is the reachable cell that would show the most unseen
//! ground per second spent getting there and looking (next-best-view:
//! Connolly 1985; frontier exploration: Yamauchi 1997). On the open map that
//! is roughly "the next unseen patch"; on the rugged map it walks around a
//! mesa or a ridge instead of turning on the spot.
//!
//! Objects (trees, boulders) do not occlude here: the planner does not know
//! where they are, only the terrain.

use hand_core::BaseSpec;

use crate::nav::{self, Cell};
use crate::terrain::{cell_center, Terrain, CELL_M};

/// How far a search look counts as covering the ground, metres (est: a
/// 0.5 m node at 14 m is about 10 px wide on the 384 px camera, above the
/// recogniser's 6 px floor).
pub const SEARCH_RANGE_M: f32 = 14.0;
/// A cell counts as seen when an object this tall standing on it would show,
/// metres (est: a boulder is 0.6 m, a tree 1.6 m, a dropped piece 0.1 m).
pub const TARGET_H_M: f32 = 0.3;
/// The time a look-around costs on top of the drive, seconds (est: six 60°
/// turns with two scans each).
pub const LOOK_COST_S: f32 = 12.0;
/// A viewpoint must show at least this many unseen cells (4 m²) to be worth the trip.
pub const MIN_GAIN: usize = 16;
const RAY_STEP_M: f32 = 0.3;
const RAY_STEP_RAD: f32 = 0.035;

/// The camera as the planner sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub x: f32,
    pub z: f32,
    /// Height of the camera above sea level, metres.
    pub height_m: f32,
    pub pitch_deg: f32,
    /// Vertical field of view, degrees.
    pub vfov_deg: f32,
    pub range_m: f32,
}

impl Eye {
    /// The robot's camera standing on cell `c` (ground plus `eye_above_m`).
    pub fn at(t: &Terrain, c: Cell, eye_above_m: f32, pitch_deg: f32, vfov_deg: f32) -> Eye {
        let (x, z) = cell_center(c.0, c.1);
        Eye { x, z, height_m: t.height_m(c.0, c.1) + t.curb_m(c.0, c.1) + eye_above_m, pitch_deg, vfov_deg, range_m: SEARCH_RANGE_M }
    }
}

/// Visits every cell index the eye can see in the heading sector `[a0, a1]`
/// (radians, world frame: 0 = +x, π/2 = +z). A cell can be visited more than once.
pub fn viewshed(t: &Terrain, eye: &Eye, a0: f32, a1: f32, visit: &mut dyn FnMut(usize)) {
    let span = (a1 - a0).abs().max(1e-3);
    let n = (span / RAY_STEP_RAD).ceil().max(1.0) as usize;
    let lo = (eye.pitch_deg - eye.vfov_deg * 0.5).to_radians().tan();
    let hi = (eye.pitch_deg + eye.vfov_deg * 0.5).to_radians().tan();
    for k in 0..=n {
        let a = a0 + (a1 - a0) * k as f32 / n as f32;
        let (dx, dz) = (a.cos(), a.sin());
        // Steepest terrain seen so far along the ray (tangent of its elevation).
        let mut horizon = f32::NEG_INFINITY;
        let mut last = usize::MAX;
        let mut r = RAY_STEP_M;
        while r <= eye.range_m {
            let Some((cx, cz)) = t.cell_of(eye.x + dx * r, eye.z + dz * r) else { break };
            let i = t.idx(cx, cz);
            let top = if t.is_water(cx, cz) { 0.0 } else { t.height_m(cx, cz) + t.curb_m(cx, cz) };
            let low = (top - eye.height_m) / r;
            let high = (top + TARGET_H_M - eye.height_m) / r;
            // An object here shows when its top clears the horizon and it is inside the camera's vertical band.
            if i != last && high >= horizon && low <= hi && high >= lo {
                visit(i);
            }
            horizon = horizon.max(low);
            last = i;
            r += RAY_STEP_M;
        }
    }
}

/// The cells the camera has looked at so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Coverage {
    w: usize,
    d: usize,
    seen: Vec<bool>,
    land: usize,
    /// Looks marked so far.
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

    /// Share of the land cells seen (water is never searched for objects).
    pub fn land_frac(&self, t: &Terrain) -> f32 {
        let n = (0..self.seen.len()).filter(|&i| self.seen[i] && !t.is_water(i % self.w, i / self.w)).count();
        n as f32 / self.land as f32
    }

    /// Marks what a camera facing `yaw` (radians) with a horizontal field of view of `hfov_deg` sees.
    pub fn look(&mut self, t: &Terrain, eye: &Eye, yaw: f32, hfov_deg: f32) {
        let h = hfov_deg.to_radians() * 0.5;
        let seen = &mut self.seen;
        viewshed(t, eye, yaw - h, yaw + h, &mut |i| seen[i] = true);
        self.looks += 1;
    }
}

/// A place to look from next.
#[derive(Clone, Debug, PartialEq)]
pub struct Viewpoint {
    pub x: f32,
    pub z: f32,
    pub cell: Cell,
    /// Unseen cells a full turn there would show.
    pub gain: usize,
    pub time_s: f32,
    /// Gain per second of driving plus [`LOOK_COST_S`].
    pub score: f32,
    /// Heading of the 45° sector with the most unseen cells, radians.
    pub face_yaw: f32,
}

/// The reachable cell that would show the most unseen ground per second.
/// `eye_above_m`, `pitch_deg` and `vfov_deg` describe the camera; `avoid` are
/// points already tried (a viewpoint within 2 m of one is skipped). `None`
/// when nothing reachable would show [`MIN_GAIN`] new cells.
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
    avoid: &[(f32, f32)],
) -> Option<Viewpoint> {
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
        let eye = Eye::at(t, (cx, cz), eye_above_m, pitch_deg, vfov_deg);
        let mut gain = 0usize;
        let mut sectors = [0usize; 8];
        let tag = k as u32;
        let w = t.w;
        viewshed(t, &eye, 0.0, std::f32::consts::TAU, &mut |i| {
            if stamp[i] != tag && !cov.seen[i] && !t.is_water(i % w, i / w) {
                stamp[i] = tag;
                gain += 1;
                let (px, pz) = cell_center(i % w, i / w);
                let a = (pz - eye.z).atan2(px - eye.x).rem_euclid(std::f32::consts::TAU);
                sectors[((a / std::f32::consts::FRAC_PI_4) as usize).min(7)] += 1;
            }
        });
        if gain < MIN_GAIN {
            continue;
        }
        let score = gain as f32 / (tt + LOOK_COST_S);
        if best.as_ref().map_or(true, |b| score > b.score) {
            let s = (0..8).max_by_key(|&i| sectors[i]).unwrap_or(0);
            best = Some(Viewpoint {
                x: eye.x,
                z: eye.z,
                cell: (cx, cz),
                gain,
                time_s: tt,
                score,
                face_yaw: (s as f32 + 0.5) * std::f32::consts::FRAC_PI_4,
            });
        }
    }
    best
}

/// Whether the straight sight line from `a` (eye height above its ground)
/// to the ground point `b` clears the terrain, with `b` raised by `target_h_m`.
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
    use hand_core::BaseKind;

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
        // A wall across the middle with a gap at one end.
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
        let v = next_viewpoint(&t, &spec, &none, start, &cov, 0.5, -12.0, 50.0, &[]).expect("unseen ground is left");
        assert!(v.gain >= MIN_GAIN && v.time_s > 0.0);
        // Everything seen: nowhere to go.
        let mut all = Coverage::new(&t);
        all.seen.iter_mut().for_each(|s| *s = true);
        assert!(next_viewpoint(&t, &spec, &none, start, &all, 0.5, -12.0, 50.0, &[]).is_none());
    }
}
