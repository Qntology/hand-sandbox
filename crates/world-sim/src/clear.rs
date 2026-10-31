//! Room around the body. The route planner steps the robot from cell centre to cell centre, but the body is a box (a
//! lift truck with its tines out front, a walker as wide as its shoulders) and so are the things it can hit: a tree's
//! crown, a boulder, an ore block, a wall, a door jamb, a shelf, a table, the water's edge. The clearance map gives,
//! for any point on the ground, how far it is to the nearest of their footprints. Routes keep the body's half width
//! and a gap from them (a hard limit) and pay for passing closer than a comfortable berth (a soft cost that falls
//! with distance), as a navigation costmap's inflated obstacles do (Lu, Hershberger & Smart, "Layered costmaps for
//! context-sensitive navigation", IROS 2014); the route is then pushed off obstacles to the middle of each gap, the
//! follower checks the body's outline at every step, and a turn in place is taken only where its sweep is clear.
//!
//! What can be hit has a height, and so does the body: every solid here is a box above the ground and a body is a box
//! for its hull and another for its tines (`Footprint`). Tines at travel height pass over a curb the wheels cannot
//! climb, and nothing at knee height stops a crown overhead. A line of curb or a cliff too high for this base is a
//! solid along the cell edge (`ledges`), seen from either side, whatever the base's own climbing limit makes of it: a
//! walker steps over what stops a wheel, so each kind of body has its own map (`World::clear_map`). Boxes are tested
//! with the separating axes, so a thin curb between two points of an outline is not missed, and the gap goes negative
//! once a body is inside something, which is how a body caught inside can still be told to get out and not further in.

use noto_core::BaseSpec;
use serde::Serialize;

use crate::fixture::{self, Fixture, FixtureKind};
use crate::nav::{self, Cell};
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
/// The farthest an obstacle is looked for (m); clearances are capped there.
pub const REACH_M: f32 = 1.2;
/// A ledge's half thickness on the cell edge (m), and how far it stands above its top: enough to stop a body up there.
const LEDGE_HALF_M: f32 = 0.01;
const LEDGE_LIP_M: f32 = 0.03;
/// A height change between two cells that is a face of rock rather than a slope (m) [추정].
const CLIFF_M: f32 = 0.30;
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

/// What a solid is, for the contact it makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolidKind {
    Tree,
    Rock,
    Fixture,
    Ledge,
    Water,
}

/// A box of the world: its footprint and how high it stands (m, in world height).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solid {
    pub rect: Rect,
    pub y0: f32,
    pub y1: f32,
    pub kind: SolidKind,
}

/// A box on the ground, turned: its centre, its long axis and its half sizes along and across that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plate {
    pub c: (f32, f32),
    pub u: (f32, f32),
    pub half: (f32, f32),
}

impl Plate {
    pub fn of_rect(r: &Rect) -> Plate {
        Plate { c: (0.5 * (r.x0 + r.x1), 0.5 * (r.z0 + r.z1)), u: (1.0, 0.0), half: (0.5 * (r.x1 - r.x0), 0.5 * (r.z1 - r.z0)) }
    }

    /// The box from `rear` behind to `front` ahead of (`x`, `z`) facing `yaw`, `half_w` to each side.
    pub fn of_pose(x: f32, z: f32, yaw: f32, front: f32, rear: f32, half_w: f32) -> Plate {
        let (s, c) = yaw.sin_cos();
        let mid = 0.5 * (front - rear);
        Plate { c: (x + c * mid, z + s * mid), u: (c, s), half: (0.5 * (front + rear), half_w) }
    }

    /// How far apart two plates are along the axis that separates them best (m): negative when they overlap, by how
    /// little it would take to part them.
    pub fn gap(&self, o: &Plate) -> f32 {
        let d = (o.c.0 - self.c.0, o.c.1 - self.c.1);
        let v = (-self.u.1, self.u.0);
        let w = (-o.u.1, o.u.0);
        let dot = |a: (f32, f32), b: (f32, f32)| a.0 * b.0 + a.1 * b.1;
        [self.u, v, o.u, w]
            .into_iter()
            .map(|n| {
                let ra = self.half.0 * dot(self.u, n).abs() + self.half.1 * dot(v, n).abs();
                let rb = o.half.0 * dot(o.u, n).abs() + o.half.1 * dot(w, n).abs();
                dot(d, n).abs() - ra - rb
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

#[derive(Clone, Debug)]
pub struct ClearMap {
    w: usize,
    d: usize,
    solids: Vec<Solid>,
    by_cell: Vec<Vec<u32>>,
    best: Vec<f32>,
    ground: Vec<f32>,
}

impl ClearMap {
    /// The obstacles a body up to `top_m` tall can hit: nodes, the blocking fixtures' boxes that reach between ankle
    /// and head height (a door's lintel does not), and the water along the shore.
    pub fn build(t: &Terrain, nodes: &[Node], fixtures: &[Fixture], top_m: f32) -> ClearMap {
        ClearMap::build_for(t, nodes, fixtures, top_m, None)
    }

    /// `build`, with the ledges `spec` cannot take as solids too (`ledges`).
    pub fn build_for(t: &Terrain, nodes: &[Node], fixtures: &[Fixture], top_m: f32, spec: Option<&BaseSpec>) -> ClearMap {
        let mut solids = Vec::new();
        for n in nodes {
            let Some((cx, cz)) = t.cell_of(n.x, n.z) else { continue };
            let (x, z) = cell_center(cx, cz);
            let g = t.height_m(cx, cz) + t.curb_m(cx, cz);
            let kind = if n.kind == crate::kinds::NodeKind::Tree { SolidKind::Tree } else { SolidKind::Rock };
            for &(y0, y1, half) in n.kind.solids() {
                if y0 < top_m {
                    solids.push(Solid { rect: Rect { x0: x - half, z0: z - half, x1: x + half, z1: z + half }, y0: g + y0, y1: g + y1, kind });
                }
            }
        }
        for b in fixture::render_boxes(fixtures, t) {
            let walkable = fixtures.iter().find(|f| f.id == b.entity).map_or(true, |f| matches!(f.kind, FixtureKind::Floor | FixtureKind::Landing | FixtureKind::Step { .. } | FixtureKind::Ramp { .. } | FixtureKind::Roof { .. }));
            let ground = t.height_m(b.cell.0, b.cell.1);
            if walkable || b.max[1] <= ground + 0.05 || b.min[1] >= ground + top_m {
                continue;
            }
            solids.push(Solid { rect: Rect { x0: b.min[0], z0: b.min[2], x1: b.max[0], z1: b.max[2] }, y0: b.min[1], y1: b.max[1], kind: SolidKind::Fixture });
        }
        for cz in 0..t.d {
            for cx in 0..t.w {
                let shore = t.is_water(cx, cz)
                    && (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| t.in_bounds(cx as i32 + dx, cz as i32 + dz) && !t.is_water((cx as i32 + dx) as usize, (cz as i32 + dz) as usize)));
                if shore {
                    let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
                    solids.push(Solid { rect: Rect { x0, z0, x1: x0 + CELL_M, z1: z0 + CELL_M }, y0: -1.0, y1: t.max_top_m() + top_m, kind: SolidKind::Water });
                }
            }
        }
        if let Some(spec) = spec {
            solids.extend(ledges(t, spec));
        }
        let mut by_cell = vec![Vec::new(); t.w * t.d];
        let reach = (REACH_M / CELL_M).ceil() as i32 + 1;
        for (i, s) in solids.iter().enumerate() {
            let r = &s.rect;
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
        let ground = (0..t.w * t.d).map(|i| t.height_m(i % t.w, i / t.w) + t.curb_m(i % t.w, i / t.w)).collect();
        let mut map = ClearMap { w: t.w, d: t.d, solids, by_cell, best: Vec::new(), ground };
        map.best = (0..t.w * t.d)
            .map(|i| {
                let (x, z) = cell_center(i % t.w, i / t.w);
                CELL_SPOTS.iter().flat_map(|dx| CELL_SPOTS.iter().map(move |dz| (x + dx, z + dz))).map(|(x, z)| map.at(x, z)).fold(0.0f32, f32::max)
            })
            .collect();
        map
    }

    pub fn solids(&self) -> &[Solid] {
        &self.solids
    }

    /// The height of the ground under a point (m).
    pub fn ground_at(&self, x: f32, z: f32) -> f32 {
        let (cx, cz) = (((x.max(0.0) / CELL_M) as usize).min(self.w - 1), ((z.max(0.0) / CELL_M) as usize).min(self.d - 1));
        self.ground[cz * self.w + cx]
    }

    fn near(&self, x: f32, z: f32) -> impl Iterator<Item = &Solid> {
        let (cx, cz) = (((x.max(0.0) / CELL_M) as usize).min(self.w - 1), ((z.max(0.0) / CELL_M) as usize).min(self.d - 1));
        self.by_cell[cz * self.w + cx].iter().map(|&i| &self.solids[i as usize])
    }

    /// The nearest solid to `p` that reaches into the heights `y0..y1`, and the gap to it (m; negative inside it). Looked
    /// for from points along the plate, since each cell lists only what is within `REACH_M` of it.
    pub fn nearest(&self, p: &Plate, y0: f32, y1: f32) -> Option<(f32, Solid)> {
        let n = ((2.0 * p.half.0 / CELL_M).ceil() as i32).max(1);
        let mut best: Option<(f32, Solid)> = None;
        for i in 0..=n {
            let along = -p.half.0 + 2.0 * p.half.0 * i as f32 / n as f32;
            let (x, z) = (p.c.0 + p.u.0 * along, p.c.1 + p.u.1 * along);
            for s in self.near(x, z) {
                if s.y0 >= y1 || s.y1 <= y0 {
                    continue;
                }
                let gap = p.gap(&Plate::of_rect(&s.rect));
                if best.as_ref().map_or(true, |b| gap < b.0) {
                    best = Some((gap, *s));
                }
            }
        }
        best
    }

    /// How far the point is from the nearest obstacle or the map's edge (m, 0 inside one), capped at `REACH_M`.
    pub fn at(&self, x: f32, z: f32) -> f32 {
        let (wm, dm) = (self.w as f32 * CELL_M, self.d as f32 * CELL_M);
        if !(0.0..wm).contains(&x) || !(0.0..dm).contains(&z) {
            return 0.0;
        }
        let edge = x.min(wm - x).min(z).min(dm - z).min(REACH_M);
        let (cx, cz) = (((x / CELL_M) as usize).min(self.w - 1), ((z / CELL_M) as usize).min(self.d - 1));
        self.by_cell[cz * self.w + cx].iter().map(|&i| self.solids[i as usize].rect.dist(x, z)).fold(edge, f32::min)
    }

    /// How far a point is from the edge of the map (m).
    fn to_edge(&self, x: f32, z: f32) -> f32 {
        let (wm, dm) = (self.w as f32 * CELL_M, self.d as f32 * CELL_M);
        x.min(wm - x).min(z).min(dm - z)
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

/// A cell edge too high for a base to cross, as a thin solid. Which edges depends on the base: a step a leg takes in its
/// stride stops a wheel, and the planner's own `nav::edge` says so. An edge it can take in one direction is no ledge.
fn ledges(t: &Terrain, spec: &BaseSpec) -> Vec<Solid> {
    let top = |c: Cell| t.height_m(c.0, c.1) + t.curb_m(c.0, c.1);
    let mut out = Vec::new();
    for cz in 0..t.d {
        for cx in 0..t.w {
            let a = (cx, cz);
            for (dx, dz) in [(1usize, 0usize), (0, 1)] {
                let b = (cx + dx, cz + dz);
                if b.0 >= t.w || b.1 >= t.d || t.is_water(a.0, a.1) || t.is_water(b.0, b.1) {
                    continue;
                }
                let (ha, hb) = (top(a), top(b));
                // A hill too steep for a step between two cells is a slope, not a wall: the planner keeps off it.
                let sheer = (t.curb_m(a.0, a.1) - t.curb_m(b.0, b.1)).abs() > 1e-4 || (t.height_m(a.0, a.1) - t.height_m(b.0, b.1)).abs() >= CLIFF_M;
                if !sheer || nav::edge(spec, t, a, b).is_some() || nav::edge(spec, t, b, a).is_some() {
                    continue;
                }
                let (x, z) = ((cx + dx) as f32 * CELL_M, (cz + dz) as f32 * CELL_M);
                let rect = if dx == 1 {
                    Rect { x0: x - LEDGE_HALF_M, z0: cz as f32 * CELL_M, x1: x + LEDGE_HALF_M, z1: (cz + 1) as f32 * CELL_M }
                } else {
                    Rect { x0: cx as f32 * CELL_M, z0: z - LEDGE_HALF_M, x1: (cx + 1) as f32 * CELL_M, z1: z + LEDGE_HALF_M }
                };
                out.push(Solid { rect, y0: ha.min(hb) - 0.5, y1: ha.max(hb) + LEDGE_LIP_M, kind: SolidKind::Ledge });
            }
        }
    }
    out
}

/// The body's outline on the ground, about its centre: as far forward as the tine tips (or the hands), as far back as
/// the counterweight (or the heels), and its half width at the widest (the chassis, or the shoulders and arms). Above
/// that the hull stands from the ground to its top, and a lift truck's tines ride out in front of the mast at their
/// travel height, narrower than the truck.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Footprint {
    pub front_m: f32,
    pub rear_m: f32,
    pub half_w_m: f32,
    /// Where the hull ends, ahead of the centre: the mast of a truck, the outline of a walker.
    pub hull_m: f32,
    pub top_m: f32,
    /// Across the tines at their widest, and from the tines' underside to the top of what rides on them (m above the
    /// ground); `None` on legs.
    pub tine_half_w_m: f32,
    pub tines_z_m: Option<[f32; 2]>,
    /// What a body steps over rather than runs into, lying on the ground (m).
    pub over_m: f32,
}

impl Footprint {
    pub fn of(body: &noto_core::balance::Body) -> Footprint {
        if body.legged() {
            // Shoulders plus an arm's thickness each side; the hands and what they hold a little ahead of the chest.
            let k = body.dims.k;
            let front_m = 0.22 * k;
            let spec = body.base.spec();
            Footprint { front_m, rear_m: 0.15 * k, half_w_m: body.dims.shoulder_half_m + 0.06 * k, hull_m: front_m, top_m: 2.1 * body.dims.hip_m, tine_half_w_m: 0.0, tines_z_m: None, over_m: spec.max_step_m.min(0.5 * body.dims.hip_m) }
        } else {
            use noto_core::act::fork;
            let d = &body.dims;
            let z = d.fork_carry_m;
            Footprint {
                front_m: fork::heel_x(body) + d.tine_len_m,
                rear_m: 0.5 * body.support_len_m + 1.15 * d.wheel_r_m + 0.02,
                half_w_m: 0.5 * body.support_width_m,
                hull_m: fork::heel_x(body),
                top_m: d.roof_m.unwrap_or(d.mast_top_m).max(d.mast_top_m),
                tine_half_w_m: 0.5 * (d.tine_spread_m + d.tine_w_m),
                tines_z_m: Some([z, z + d.tine_t_m]),
                over_m: 0.0,
            }
        }
    }

    /// The same body with `height_m` of load on its tines.
    pub fn laden(mut self, height_m: f32) -> Footprint {
        if let Some(z) = self.tines_z_m.as_mut() {
            z[1] += height_m;
        }
        self
    }

    /// The radius the outline sweeps turning in place about its centre.
    pub fn sweep_m(&self) -> f32 {
        self.front_m.max(self.rear_m).hypot(self.half_w_m)
    }

    /// The hull at a pose.
    pub fn hull(&self, x: f32, z: f32, yaw: f32) -> Plate {
        Plate::of_pose(x, z, yaw, self.hull_m, self.rear_m, self.half_w_m)
    }

    /// The tines at a pose, and how high they ride.
    pub fn tines(&self, x: f32, z: f32, yaw: f32) -> Option<(Plate, [f32; 2])> {
        let band = self.tines_z_m?;
        let (s, c) = yaw.sin_cos();
        let mid = 0.5 * (self.hull_m + self.front_m);
        Some((Plate { c: (x + c * mid, z + s * mid), u: (c, s), half: (0.5 * (self.front_m - self.hull_m), self.tine_half_w_m) }, band))
    }

    /// The solid nearest the body at that pose, and the gap to it (m; negative inside it): the hull from the ground to
    /// its top, and the tines where they ride.
    pub fn nearest(&self, map: &ClearMap, x: f32, z: f32, yaw: f32) -> Option<(f32, Solid)> {
        let g = map.ground_at(x, z);
        let hull = map.nearest(&self.hull(x, z, yaw), g, g + self.top_m);
        // Going up a rise the body tips with it: the tines ride that far above the ground they are over.
        let tines = self.tines(x, z, yaw).and_then(|(p, band)| {
            let up = g.max(map.ground_at(p.c.0 + p.u.0 * p.half.0, p.c.1 + p.u.1 * p.half.0));
            map.nearest(&p, up + band[0], up + band[1])
        });
        match (hull, tines) {
            (Some(a), Some(b)) => Some(if b.0 < a.0 { b } else { a }),
            (a, b) => a.or(b),
        }
    }

    /// The least room around the body at that pose (m), negative inside something, and capped at `REACH_M` as the map
    /// is. Its corners count against the edge of the map.
    pub fn room(&self, map: &ClearMap, x: f32, z: f32, yaw: f32) -> f32 {
        let solid = self.nearest(map, x, z, yaw).map_or(REACH_M, |n| n.0.min(REACH_M));
        let (s, c) = yaw.sin_cos();
        let corners = [(self.front_m, self.half_w_m), (self.front_m, -self.half_w_m), (-self.rear_m, self.half_w_m), (-self.rear_m, -self.half_w_m)];
        corners.into_iter().map(|(f, l)| map.to_edge(x + c * f - s * l, z + s * f + c * l)).fold(solid, f32::min)
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

/// Shifts the route's inner points, a few centimetres at a time, to where the body fits the legs on both sides of them:
/// a corner pushed to the middle of its cell can still leave the straight between two points against a rock. The ends
/// stay. Returns the least room left along the route (m), for the body facing along it.
pub fn fit_path(map: &ClearMap, fp: &Footprint, path: &mut [(f32, f32)], want_m: f32) -> f32 {
    let leg = |a: (f32, f32), b: (f32, f32)| {
        let yaw = (b.1 - a.1).atan2(b.0 - a.0);
        let n = ((a.0 - b.0).hypot(a.1 - b.1) / 0.1).ceil().max(1.0) as usize;
        (0..=n).map(|i| i as f32 / n as f32).map(|u| fp.room(map, a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u, yaw)).fold(f32::INFINITY, f32::min)
    };
    if path.len() >= 3 {
        const STEPS: [(f32, f32); 8] = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7)];
        for _ in 0..4 {
            let mut moved = false;
            for i in 1..path.len() - 1 {
                let here = leg(path[i - 1], path[i]).min(leg(path[i], path[i + 1]));
                if here >= want_m {
                    continue;
                }
                let best = STEPS
                    .iter()
                    .flat_map(|d| [0.06f32, 0.12].map(|r| (path[i].0 + d.0 * r, path[i].1 + d.1 * r)))
                    .map(|q| (q, leg(path[i - 1], q).min(leg(q, path[i + 1]))))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((q, _)) = best.filter(|b| b.1 > here + 1e-3) {
                    path[i] = q;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }
    path.windows(2).map(|w| leg(w[0], w[1])).fold(f32::INFINITY, f32::min)
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
        let half = crate::kinds::NodeKind::Tree.solids()[0].2;
        assert!((map.at(x + 0.74, z) - (0.74 - half)).abs() < 1e-4, "{}", map.at(x + 0.74, z));
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
        let (tines, band) = truck.tines(1.0, 2.0, 0.0).unwrap();
        assert!((tines.c.0 + tines.half.0 - (1.0 + truck.front_m)).abs() < 1e-5 && band[0] > 0.04);
        let (turned, _) = truck.tines(1.0, 2.0, std::f32::consts::FRAC_PI_2).unwrap();
        assert!((turned.c.1 + turned.half.0 - (2.0 + truck.front_m)).abs() < 1e-5, "yaw turns the outline toward +z");
        assert!(walker.tines(0.0, 0.0, 0.0).is_none());
    }

    fn body(base: noto_core::BaseKind) -> noto_core::balance::Body {
        noto_core::balance::Body::new(&noto_core::BodyPlan::standard(), base, noto_core::FingerCount::Five)
    }

    #[test]
    fn two_plates_part_along_the_axis_between_them() {
        let a = Plate::of_pose(0.0, 0.0, 0.0, 1.0, 1.0, 0.5);
        let b = Plate::of_rect(&Rect { x0: 1.5, z0: -0.5, x1: 2.5, z1: 0.5 });
        assert!((a.gap(&b) - 0.5).abs() < 1e-5);
        let inside = Plate::of_rect(&Rect { x0: 0.5, z0: -0.1, x1: 2.0, z1: 0.1 });
        assert!(a.gap(&inside) < -0.3, "overlapping goes negative: {}", a.gap(&inside));
        let turned = Plate::of_pose(0.0, 0.0, std::f32::consts::FRAC_PI_4, 1.0, 1.0, 0.1);
        let corner = Plate::of_rect(&Rect { x0: 0.75, z0: 0.0, x1: 1.0, z1: 0.1 });
        assert!(turned.gap(&corner) > 0.0, "the long thin plate misses a box at its side");
    }

    #[test]
    fn a_thin_curb_cannot_slip_between_two_points_of_the_outline() {
        let mut t = flat(32, 32);
        for cz in 0..32 {
            t.set_curb_cm(16, cz, 15);
        }
        let truck = body(noto_core::BaseKind::Wheel);
        let spec = truck.base.spec();
        let map = ClearMap::build_for(&t, &[], &[], 2.3, Some(&spec));
        let fp = Footprint::of(&truck);
        let z = cell_center(0, 10).1;
        assert!(fp.room(&map, 5.0, z, 0.0) > 0.0);
        assert!(fp.room(&map, 8.0, z, 0.0) < 0.0, "the hull is across the curb");
        assert!(map.solids().iter().any(|s| s.kind == SolidKind::Ledge));
        let walker = body(noto_core::BaseKind::HumanFoot);
        let foot = ClearMap::build_for(&t, &[], &[], 1.9, Some(&walker.base.spec()));
        assert!(foot.solids().iter().all(|s| s.kind != SolidKind::Ledge), "a leg steps over a curb a wheel cannot climb");
    }

    #[test]
    fn tines_at_travel_height_pass_over_a_curb_that_stops_the_hull() {
        let mut t = flat(32, 32);
        for cz in 0..32 {
            t.set_curb_cm(16, cz, 15);
        }
        let truck = body(noto_core::BaseKind::Wheel);
        let map = ClearMap::build_for(&t, &[], &[], 2.3, Some(&truck.base.spec()));
        let fp = Footprint::of(&truck);
        // Tine tips past the curb line, the hull short of it.
        let z = cell_center(0, 10).1;
        let x = 8.0 - fp.hull_m - 0.1;
        assert!(fp.room(&map, x, z, 0.0) > 0.0, "the tines ride over it: {}", fp.room(&map, x, z, 0.0));
        let low = Footprint { tines_z_m: Some([0.02, 0.05]), ..fp };
        assert!(low.room(&map, x, z, 0.0) < 0.0, "tines on the floor run into it");
    }

    #[test]
    fn a_rock_stops_the_tines_but_the_crown_overhead_stops_nothing() {
        let t = flat(32, 32);
        let rock = Node { id: 1, kind: crate::kinds::NodeKind::Boulder, x: cell_center(16, 10).0, z: cell_center(16, 10).1, hp: 1.0, max_hp: 1.0 };
        let truck = body(noto_core::BaseKind::Wheel);
        let fp = Footprint::of(&truck);
        let map = ClearMap::build(&t, &[rock.clone()], &[], fp.top_m);
        let x = rock.x - fp.front_m + 0.1;
        assert!(fp.room(&map, x, rock.z, 0.0) < 0.0, "the tine tips are in the rock");
        let tree = Node { kind: crate::kinds::NodeKind::Tree, ..rock.clone() };
        let map = ClearMap::build(&t, &[tree.clone()], &[], fp.top_m);
        let y = tree.z + 0.14 + fp.half_w_m + 0.2;
        assert!(fp.room(&map, tree.x, y, 0.0) > 0.0, "under the crown, clear of the trunk");
    }

    #[test]
    fn the_route_is_shifted_off_a_rock_the_straight_would_clip() {
        let t = flat(32, 32);
        let c = cell_center(16, 10);
        let rock = Node { id: 1, kind: crate::kinds::NodeKind::Boulder, x: c.0, z: c.1, hp: 1.0, max_hp: 1.0 };
        let walker = body(noto_core::BaseKind::HumanFoot);
        let fp = Footprint::of(&walker);
        let map = ClearMap::build(&t, &[rock], &[], fp.top_m);
        // Along the row of cells beside the rock: the centres are 0.26 m off its face, closer than the shoulders.
        let row = |cx: usize| cell_center(cx, 11);
        let mut path = vec![row(13), row(14), row(15), row(16), row(17), row(18), row(19)];
        assert!(fp.room(&map, row(16).0, row(16).1, 0.0) < 0.0, "the cell centres clip the rock");
        let room = fit_path(&map, &fp, &mut path, TOUCH_GAP_M);
        assert!(room >= TOUCH_GAP_M - 1e-3, "{room}: {path:?}");
    }
}
