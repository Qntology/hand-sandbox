//! Room around the body. The route planner steps the robot from cell centre to cell centre, but the body is a box (a
//! lift truck with its tines out front, a walker as wide as its shoulders) and so are the things it can hit: a tree's
//! crown, a boulder, an ore block, a wall, a door jamb, a shelf, a table, the water's edge. The clearance map gives,
//! for any point on the ground, how far it is to the nearest of their footprints. Routes keep the body's half width
//! and a gap from them (a hard limit) and pay for passing closer than a comfortable berth (a soft cost that falls
//! with distance), as a navigation costmap's inflated obstacles do (Lu, Hershberger & Smart, "Layered costmaps for
//! context-sensitive navigation", IROS 2014); the route is then pushed off obstacles to the middle of each gap, the
//! follower checks the body's outline at every step, and a turn in place is taken only where its sweep is clear.

use serde::Serialize;

use crate::fixture::{self, Fixture, FixtureKind};
use crate::nav::Cell;
use crate::terrain::{cell_center, Terrain, CELL_M};
use crate::world::Node;

/// The gap a route keeps between the body's side and an obstacle (m) [추정: a hand's width, more than the follower
/// drifts off its line].
pub const MIN_GAP_M: f32 = 0.06;
/// The gap a route makes do with when nothing roomier gets there (m): the body still fits [추정].
pub const TIGHT_GAP_M: f32 = 0.015;
/// The berth a route prefers (m) [추정: a person keeps about 0.3 m off walls and posts, Fruin's body ellipse].
pub const COMFORT_GAP_M: f32 = 0.30;
/// How many times its own travel time a cell costs at the hard limit (falling to nothing at the comfortable berth) [추정].
pub const CLEAR_COST: f32 = 3.0;
/// The gap the follower lets the body's outline come to an obstacle before it stops cutting the corner (m) [추정].
pub const TOUCH_GAP_M: f32 = 0.02;
/// The half size of a node's footprint (m): a tree's crown, a boulder's base and an ore block, as drawn (`render`).
pub const NODE_HALF_M: f32 = 0.24;
/// The farthest an obstacle is looked for (m); clearances are capped there.
pub const REACH_M: f32 = 1.2;
const CELL_SPOTS: [f32; 3] = [-0.2, 0.0, 0.2];

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Rect {
    pub x0: f32,
    pub z0: f32,
    pub x1: f32,
    pub z1: f32,
}

impl Rect {
    pub fn dist(&self, x: f32, z: f32) -> f32 {
        let dx = (self.x0 - x).max(x - self.x1).max(0.0);
        let dz = (self.z0 - z).max(z - self.z1).max(0.0);
        dx.hypot(dz)
    }
}

#[derive(Clone, Debug)]
pub struct ClearMap {
    w: usize,
    d: usize,
    rects: Vec<Rect>,
    by_cell: Vec<Vec<u32>>,
    best: Vec<f32>,
}

impl ClearMap {
    /// The obstacles a body up to `top_m` tall can hit: nodes, the blocking fixtures' boxes that reach between ankle
    /// and head height (a door's lintel does not), and the water along the shore.
    pub fn build(t: &Terrain, nodes: &[Node], fixtures: &[Fixture], top_m: f32) -> ClearMap {
        let mut rects = Vec::new();
        for n in nodes {
            let Some((cx, cz)) = t.cell_of(n.x, n.z) else { continue };
            let (x, z) = cell_center(cx, cz);
            rects.push(Rect { x0: x - NODE_HALF_M, z0: z - NODE_HALF_M, x1: x + NODE_HALF_M, z1: z + NODE_HALF_M });
        }
        for b in fixture::render_boxes(fixtures, t) {
            let walkable = fixtures.iter().find(|f| f.id == b.entity).map_or(true, |f| matches!(f.kind, FixtureKind::Floor | FixtureKind::Landing | FixtureKind::Step { .. } | FixtureKind::Ramp { .. } | FixtureKind::Roof { .. }));
            let ground = t.height_m(b.cell.0, b.cell.1);
            if walkable || b.max[1] <= ground + 0.05 || b.min[1] >= ground + top_m {
                continue;
            }
            rects.push(Rect { x0: b.min[0], z0: b.min[2], x1: b.max[0], z1: b.max[2] });
        }
        for cz in 0..t.d {
            for cx in 0..t.w {
                let shore = t.is_water(cx, cz)
                    && (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| t.in_bounds(cx as i32 + dx, cz as i32 + dz) && !t.is_water((cx as i32 + dx) as usize, (cz as i32 + dz) as usize)));
                if shore {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    rects.push(Rect { x0, z0, x1: x0 + CELL_M, z1: z0 + CELL_M });
                }
            }
        }
        let mut by_cell = vec![Vec::new(); t.w * t.d];
        let reach = (REACH_M / CELL_M).ceil() as i32 + 1;
        for (i, r) in rects.iter().enumerate() {
            let (c0, c1) = (((r.x0 / CELL_M).floor() as i32 - reach).max(0), ((r.x1 / CELL_M).floor() as i32 + reach).min(t.w as i32 - 1));
            let (r0, r1) = (((r.z0 / CELL_M).floor() as i32 - reach).max(0), ((r.z1 / CELL_M).floor() as i32 + reach).min(t.d as i32 - 1));
            for cz in r0..=r1 {
                for cx in c0..=c1 {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    let gap = Rect { x0, z0, x1: x0 + CELL_M, z1: z0 + CELL_M };
                    let far = (r.x0 - gap.x1).max(gap.x0 - r.x1).max(0.0).hypot((r.z0 - gap.z1).max(gap.z0 - r.z1).max(0.0));
                    if far <= REACH_M {
                        by_cell[cz as usize * t.w + cx as usize].push(i as u32);
                    }
                }
            }
        }
        let mut map = ClearMap { w: t.w, d: t.d, rects, by_cell, best: Vec::new() };
        map.best = (0..t.w * t.d)
            .map(|i| {
                let (x, z) = cell_center(i % t.w, i / t.w);
                CELL_SPOTS.iter().flat_map(|dx| CELL_SPOTS.iter().map(move |dz| (x + dx, z + dz))).map(|(x, z)| map.at(x, z)).fold(0.0f32, f32::max)
            })
            .collect();
        map
    }

    pub fn obstacles(&self) -> &[Rect] {
        &self.rects
    }

    /// How far the point is from the nearest obstacle or the map's edge (m, 0 inside one), capped at `REACH_M`.
    pub fn at(&self, x: f32, z: f32) -> f32 {
        let (wm, dm) = (self.w as f32 * CELL_M, self.d as f32 * CELL_M);
        if !(0.0..wm).contains(&x) || !(0.0..dm).contains(&z) {
            return 0.0;
        }
        let edge = x.min(wm - x).min(z).min(dm - z).min(REACH_M);
        let (cx, cz) = (((x / CELL_M) as usize).min(self.w - 1), ((z / CELL_M) as usize).min(self.d - 1));
        self.by_cell[cz * self.w + cx].iter().map(|&i| self.rects[i as usize].dist(x, z)).fold(edge, f32::min)
    }

    /// The most room anywhere in the cell (m): where the body would stand in it.
    pub fn best_in(&self, c: Cell) -> f32 {
        self.best.get(c.1 * self.w + c.0).copied().unwrap_or(0.0)
    }

    /// The most room on the way from cell `a` to the neighbouring cell `b` (m): across their shared side, or at the
    /// corner a diagonal step passes.
    pub fn gate(&self, a: Cell, b: Cell) -> f32 {
        let (ax, az) = cell_center(a.0, a.1);
        let (bx, bz) = cell_center(b.0, b.1);
        let (mx, mz) = (0.5 * (ax + bx), 0.5 * (az + bz));
        if a.0 != b.0 && a.1 != b.1 {
            return self.at(mx, mz);
        }
        let (ux, uz) = if a.0 == b.0 { (1.0, 0.0) } else { (0.0, 1.0) };
        CELL_SPOTS.iter().map(|s| self.at(mx + ux * s, mz + uz * s)).fold(0.0f32, f32::max)
    }

    /// The least room along the straight line from `a` to `b` (m), sampled every 5 cm.
    pub fn along(&self, a: (f32, f32), b: (f32, f32)) -> f32 {
        let n = ((a.0 - b.0).hypot(a.1 - b.1) / 0.05).ceil().max(1.0) as usize;
        (0..=n).map(|i| i as f32 / n as f32).map(|u| self.at(a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)).fold(f32::INFINITY, f32::min)
    }
}

/// The body's outline on the ground, about its centre: as far forward as the tine tips (or the hands), as far back as
/// the counterweight (or the heels), and its half width at the widest (the chassis, or the shoulders and arms).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Footprint {
    pub front_m: f32,
    pub rear_m: f32,
    pub half_w_m: f32,
}

impl Footprint {
    pub fn of(body: &noto_core::balance::Body) -> Footprint {
        if body.legged() {
            // Shoulders plus an arm's thickness each side; the hands and what they hold a little ahead of the chest.
            let k = body.dims.k;
            Footprint { front_m: 0.22 * k, rear_m: 0.15 * k, half_w_m: body.dims.shoulder_half_m + 0.06 * k }
        } else {
            use noto_core::act::fork;
            Footprint { front_m: fork::heel_x(body) + body.dims.tine_len_m, rear_m: 0.5 * body.support_len_m + 1.15 * body.dims.wheel_r_m + 0.02, half_w_m: 0.5 * body.support_width_m }
        }
    }

    /// The radius the outline sweeps turning in place about its centre.
    pub fn sweep_m(&self) -> f32 {
        self.front_m.max(self.rear_m).hypot(self.half_w_m)
    }

    /// Points along the outline, at most 10 cm apart, for the body at (`x`, `z`) facing `yaw`.
    pub fn outline(&self, x: f32, z: f32, yaw: f32) -> Vec<(f32, f32)> {
        let (s, c) = yaw.sin_cos();
        let corners = [(self.front_m, self.half_w_m), (-self.rear_m, self.half_w_m), (-self.rear_m, -self.half_w_m), (self.front_m, -self.half_w_m)];
        let mut v = Vec::with_capacity(28);
        for k in 0..4 {
            let (a, b) = (corners[k], corners[(k + 1) % 4]);
            let n = ((a.0 - b.0).hypot(a.1 - b.1) / 0.1).ceil().max(1.0) as usize;
            for i in 0..n {
                let u = i as f32 / n as f32;
                let (f, l) = (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u);
                v.push((x + c * f - s * l, z + s * f + c * l));
            }
        }
        v
    }

    /// The least room around the outline at that pose (m).
    pub fn room(&self, map: &ClearMap, x: f32, z: f32, yaw: f32) -> f32 {
        self.outline(x, z, yaw).into_iter().map(|(px, pz)| map.at(px, pz)).fold(f32::INFINITY, f32::min)
    }
}

/// Pushes the route's corner points off obstacles, each within its own cell, to where there is the most room up to the
/// comfortable berth (a point in the open stays at its cell's centre), and keeps a point at its centre where the move
/// would bring a leg of the route closer to an obstacle than `need_m`. The first point (where the body stands) and the
/// last (the goal) stay.
pub fn centre_path(map: &ClearMap, path: &mut [(f32, f32)], need_m: f32, soft_m: f32) {
    if path.len() < 3 {
        return;
    }
    let spots: Vec<(f32, f32)> = (-2..=2).flat_map(|i| (-2..=2).map(move |j| (0.1 * i as f32, 0.1 * j as f32))).collect();
    let n = path.len();
    for i in 1..n - 1 {
        let c = path[i];
        let here = map.at(c.0, c.1);
        if here >= soft_m {
            continue;
        }
        let score = |q: (f32, f32)| map.at(q.0, q.1).min(soft_m) - 0.15 * (q.0 - c.0).hypot(q.1 - c.1);
        let best = spots.iter().map(|(dx, dz)| (c.0 + dx, c.1 + dz)).filter(|q| map.at(q.0, q.1) >= need_m.min(here)).max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap_or(c);
        let before = map.along(path[i - 1], c).min(map.along(c, path[i + 1]));
        let after = map.along(path[i - 1], best).min(map.along(best, path[i + 1]));
        if after >= need_m.min(before) - 1e-4 {
            path[i] = best;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(w: usize, d: usize) -> Terrain {
        Terrain::generate(1, w, d, &[(w as f32 * 0.25, d as f32 * 0.25, w as f32)])
    }

    fn tree(id: u32, c: Cell) -> Node {
        let (x, z) = cell_center(c.0, c.1);
        Node { id, kind: crate::kinds::NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 }
    }

    #[test]
    fn room_is_the_distance_to_the_nearest_footprint_and_the_edge() {
        let t = flat(32, 32);
        let map = ClearMap::build(&t, &[tree(1, (10, 10))], &[], 1.4);
        let (x, z) = cell_center(10, 10);
        assert_eq!(map.at(x, z), 0.0, "inside the crown");
        assert!((map.at(x + 0.74, z) - 0.5).abs() < 1e-4, "{}", map.at(x + 0.74, z));
        assert!((map.at(x + 3.0, z) - REACH_M).abs() < 1e-5, "capped far away");
        assert!((map.at(0.2, 8.0) - 0.2).abs() < 1e-5, "the map's edge counts");
        assert!(map.best_in((11, 10)) > map.at(cell_center(11, 10).0, cell_center(11, 10).1), "the best spot of a cell is off its centre, away from the tree");
    }

    #[test]
    fn a_route_is_pushed_to_the_middle_of_a_doorway_two_cells_wide() {
        let t = flat(32, 32);
        // Two rows of trees leave a gap of two cells; the planner's centres run 0.25 m off the gap's middle.
        let mut nodes = Vec::new();
        for (k, cz) in [6usize, 7, 8, 9, 12, 13, 14, 15].into_iter().enumerate() {
            nodes.push(tree(k as u32, (10, cz)));
        }
        let map = ClearMap::build(&t, &nodes, &[], 1.4);
        let row = |cx: usize| cell_center(cx, 10);
        let mut path = vec![row(7), row(8), row(9), row(10), row(11), row(12), row(13)];
        let before = path.clone();
        let need = 0.2 + MIN_GAP_M;
        centre_path(&map, &mut path, need, 0.2 + COMFORT_GAP_M);
        let mid = 5.5f32;
        assert!((before[3].1 - mid).abs() > 0.2 && (path[3].1 - mid).abs() < 0.06, "the point in the gap moves to its middle: {:?} -> {:?}", before[3], path[3]);
        assert!(map.at(path[3].0, path[3].1) > map.at(before[3].0, before[3].1) + 0.15);
        assert_eq!(path[0], before[0]);
        assert_eq!(path[6], before[6]);
    }

    #[test]
    fn the_outline_of_a_lift_truck_reaches_its_tine_tips() {
        use noto_core::{balance::Body, BaseKind, BodyPlan, FingerCount};
        let truck = Footprint::of(&Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five));
        let walker = Footprint::of(&Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five));
        assert!(truck.front_m > 0.4 && truck.half_w_m > 0.15 && truck.sweep_m() > walker.sweep_m(), "{truck:?} {walker:?}");
        let pts = truck.outline(1.0, 2.0, 0.0);
        let xmax = pts.iter().map(|p| p.0).fold(f32::MIN, f32::max);
        assert!((xmax - (1.0 + truck.front_m)).abs() < 1e-5);
        let turned = truck.outline(1.0, 2.0, std::f32::consts::FRAC_PI_2);
        let zmax = turned.iter().map(|p| p.1).fold(f32::MIN, f32::max);
        assert!((zmax - (2.0 + truck.front_m)).abs() < 1e-5, "yaw turns the outline toward +z");
    }
}
