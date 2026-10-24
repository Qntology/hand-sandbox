//! Route planning on the cell grid.
//!
//! Every edge between neighbouring cells is judged by the same drive model the
//! rest of the project uses, `hand_core::simulate_drive`: a smooth height
//! difference becomes a `Slope` segment and a curb difference becomes a `Step`
//! segment. So what the planner calls passable is exactly what the "foot"
//! preview calls passable, and edge time and energy come from the same formula.
//! On top of that the ground of the cell being entered sets its own limit:
//! the friction its contacts get and, for a known drive, the force left after
//! rolling resistance (`hand_core::surface::climb_limit_deg`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use hand_core::surface::{climb_limit_deg, energy_factor};
use hand_core::{simulate_drive, BaseSpec, Terrain as Segment};

use crate::terrain::{cell_center, Terrain, CELL_M};

pub type Cell = (usize, usize);

/// One move between two neighbouring cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgePlan {
    pub from: Cell,
    pub to: Cell,
    /// Straight-line distance between the cell centres, metres.
    pub horizontal_m: f32,
    /// Driving time on this edge without the curb pause, seconds.
    pub move_time_s: f32,
    /// Extra time spent climbing a curb at the start of the edge, seconds.
    pub pause_s: f32,
    /// Curb step height on this edge, metres (0 when none).
    pub step_m: f32,
    pub slope_deg: f32,
    /// Smallest remaining margin to the slope and step limits, 0..1.
    pub margin: f32,
    /// Steepest slope the base holds on the ground of the cell entered, degrees.
    pub ground_limit_deg: f32,
    /// Driving energy on that ground over firm ground (rolling resistance, walking terrain factor).
    pub energy_factor: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub cells: Vec<Cell>,
    pub edges: Vec<EdgePlan>,
    pub time_s: f32,
    pub energy_j: f32,
    pub distance_m: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum Goal {
    /// Stand on this exact cell.
    Cell(Cell),
    /// Stand on any free cell whose centre is within `reach_m` of the point.
    Within { x: f32, z: f32, reach_m: f32 },
}

/// Something the base pushes along the ground ahead of it: a pushed item's
/// mass and what it is made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PushLoad {
    pub mass_kg: f32,
    pub material: hand_core::act::material::ItemMaterial,
}

/// Judges one move. `None` when the base cannot pass.
pub fn edge(spec: &BaseSpec, t: &Terrain, a: Cell, b: Cell) -> Option<EdgePlan> {
    edge_with(spec, t, a, b, None)
}

/// [`edge`] while pushing a load: the item's static friction on the ground
/// it is pushed onto and its share of the slope go on top of the drive's own
/// need (a move the drive cannot make with it is not passable).
pub fn edge_with(spec: &BaseSpec, t: &Terrain, a: Cell, b: Cell, load: Option<PushLoad>) -> Option<EdgePlan> {
    // No base drives or walks into the sea.
    if t.is_water(b.0, b.1) {
        return None;
    }
    let (ax, az) = cell_center(a.0, a.1);
    let (bx, bz) = cell_center(b.0, b.1);
    let horizontal = ((ax - bx).powi(2) + (az - bz).powi(2)).sqrt();
    let rise = t.height_m(b.0, b.1) - t.height_m(a.0, a.1);
    let step = (t.curb_m(b.0, b.1) - t.curb_m(a.0, a.1)).abs();

    let ground = if rise.abs() < 1e-6 {
        Segment::Flat { length_m: horizontal }
    } else {
        Segment::Slope {
            length_m: horizontal.hypot(rise),
            deg: rise.abs().atan2(horizontal).to_degrees(),
        }
    };
    let slope_deg = match ground {
        Segment::Slope { deg, .. } => deg,
        _ => 0.0,
    };
    let ground_only = simulate_drive(spec, &[ground]);
    if !ground_only.passable {
        return None;
    }
    // The ground of the cell entered: its friction and the drive's force left after rolling resistance.
    let props = t.ground_at(b.0, b.1).surface().props();
    let ground_limit = climb_limit_deg(spec.kind, props, spec.drive_accel_mps2);
    if slope_deg > ground_limit {
        return None;
    }
    // How much of the cruise speed this ground leaves: shorter steps on slippery
    // ground, wheel or track slip on soft ground (`hand_core::act`).
    let signed = if rise >= 0.0 { slope_deg } else { -slope_deg };
    let extra = match (load, spec.mass_kg) {
        (Some(l), Some(m)) => {
            let th = signed.to_radians();
            let mu = hand_core::act::material::item_on(l.material, t.ground_at(b.0, b.1).surface()).0.mean;
            (l.mass_kg * hand_core::balance::G * (mu * th.cos() + th.sin())).max(0.0) / m.max(1e-3)
        }
        _ => 0.0,
    };
    let factor = hand_core::act::locomotion_factor_with(spec, props, signed, extra)?;
    let move_time = ground_only.time_s / factor;
    // The drive turns at its cruise speed whatever the slip: the power is the
    // cruise power, and the slower ground speed makes the stretch cost more.
    let speed = horizontal / ground_only.time_s.max(1e-6);
    let energy_factor = energy_factor(spec.kind, props, spec.mass_kg.unwrap_or(0.0), speed, spec.power_w);
    let mut pause = 0.0;
    if step > 1e-6 {
        let full = simulate_drive(spec, &[ground, Segment::Step { height_m: step }]);
        if !full.passable {
            return None;
        }
        pause = full.time_s - ground_only.time_s;
    }
    let slope_margin = 1.0 - slope_deg / spec.max_slope_deg.min(ground_limit).max(1e-3);
    let step_margin = if step > 1e-6 { 1.0 - step / spec.max_step_m } else { 1.0 };
    Some(EdgePlan {
        from: a,
        to: b,
        horizontal_m: horizontal,
        move_time_s: move_time,
        pause_s: pause,
        step_m: step,
        slope_deg,
        margin: slope_margin.min(step_margin).clamp(0.0, 1.0),
        ground_limit_deg: ground_limit,
        energy_factor,
    })
}

const NEIGHBORS: [(i32, i32); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

fn reached(goal: &Goal, c: Cell, blocked: &dyn Fn(usize, usize) -> bool) -> bool {
    match *goal {
        Goal::Cell(g) => c == g,
        Goal::Within { x, z, reach_m } => {
            if blocked(c.0, c.1) {
                return false;
            }
            let (cx, cz) = cell_center(c.0, c.1);
            ((cx - x).powi(2) + (cz - z).powi(2)).sqrt() <= reach_m
        }
    }
}

fn heuristic(goal: &Goal, c: Cell, speed: f32) -> f32 {
    let (cx, cz) = cell_center(c.0, c.1);
    let (gx, gz, reach) = match *goal {
        Goal::Cell(g) => {
            let (x, z) = cell_center(g.0, g.1);
            (x, z, 0.0)
        }
        Goal::Within { x, z, reach_m } => (x, z, reach_m),
    };
    (((cx - gx).powi(2) + (cz - gz).powi(2)).sqrt() - reach).max(0.0) / speed
}

/// A* over the 8-neighbour grid, minimising drive time. `blocked` marks cells
/// the base cannot enter (resource nodes). Returns `None` when no route exists.
pub fn plan(spec: &BaseSpec, t: &Terrain, blocked: &dyn Fn(usize, usize) -> bool, start: Cell, goal: Goal) -> Option<Plan> {
    plan_with(spec, t, blocked, start, goal, None)
}

/// [`plan`] for a base pushing a load ahead of it ([`edge_with`]).
pub fn plan_with(spec: &BaseSpec, t: &Terrain, blocked: &dyn Fn(usize, usize) -> bool, start: Cell, goal: Goal, load: Option<PushLoad>) -> Option<Plan> {
    if let Goal::Cell(g) = goal {
        if blocked(g.0, g.1) {
            return None;
        }
    }
    let n = t.w * t.d;
    let mut best = vec![f32::INFINITY; n];
    let mut came: Vec<Option<(usize, EdgePlan)>> = vec![None; n];
    let mut heap: BinaryHeap<(Reverse<u32>, usize)> = BinaryHeap::new();
    let key = |f: f32| f.max(0.0).to_bits();

    let si = t.idx(start.0, start.1);
    best[si] = 0.0;
    heap.push((Reverse(key(heuristic(&goal, start, spec.max_speed_mps))), si));

    while let Some((_, ci)) = heap.pop() {
        let c = (ci % t.w, ci / t.w);
        if reached(&goal, c, blocked) {
            return Some(rebuild(&came, t, ci, si, spec));
        }
        let g = best[ci];
        for (dx, dz) in NEIGHBORS {
            let (nx, nz) = (c.0 as i32 + dx, c.1 as i32 + dz);
            if !t.in_bounds(nx, nz) {
                continue;
            }
            let nc = (nx as usize, nz as usize);
            if blocked(nc.0, nc.1) {
                continue;
            }
            if dx != 0 && dz != 0 {
                // No corner cutting past a blocked cell.
                if blocked((c.0 as i32 + dx) as usize, c.1) || blocked(c.0, (c.1 as i32 + dz) as usize) {
                    continue;
                }
            }
            let Some(e) = edge_with(spec, t, c, nc, load) else { continue };
            let cost = g + e.move_time_s + e.pause_s;
            let ni = t.idx(nc.0, nc.1);
            if cost < best[ni] {
                best[ni] = cost;
                came[ni] = Some((ci, e));
                heap.push((Reverse(key(cost + heuristic(&goal, nc, spec.max_speed_mps))), ni));
            }
        }
    }
    None
}

fn rebuild(came: &[Option<(usize, EdgePlan)>], t: &Terrain, goal_idx: usize, start_idx: usize, spec: &BaseSpec) -> Plan {
    let mut edges = Vec::new();
    let mut i = goal_idx;
    while i != start_idx {
        let (prev, e) = came[i].expect("path is connected back to the start");
        edges.push(e);
        i = prev;
    }
    edges.reverse();
    let mut cells = vec![(start_idx % t.w, start_idx / t.w)];
    cells.extend(edges.iter().map(|e| e.to));
    let time_s: f32 = edges.iter().map(|e| e.move_time_s + e.pause_s).sum();
    let distance_m: f32 = edges.iter().map(|e| e.horizontal_m).sum();
    let energy_j = spec.power_w * edges.iter().map(|e| e.move_time_s * e.energy_factor + e.pause_s).sum::<f32>();
    Plan { cells, edges, time_s, energy_j, distance_m }
}

/// Drive time from `start` to every cell (Dijkstra over the same edges as
/// [`plan`]), seconds; infinite where the base cannot get. Cells costlier
/// than `max_s` are left infinite.
pub fn travel_times(spec: &BaseSpec, t: &Terrain, blocked: &dyn Fn(usize, usize) -> bool, start: Cell, max_s: f32) -> Vec<f32> {
    let n = t.w * t.d;
    let mut best = vec![f32::INFINITY; n];
    let mut heap: BinaryHeap<(Reverse<u32>, usize)> = BinaryHeap::new();
    let si = t.idx(start.0, start.1);
    best[si] = 0.0;
    heap.push((Reverse(0), si));
    while let Some((Reverse(k), ci)) = heap.pop() {
        let g = f32::from_bits(k);
        if g > best[ci] {
            continue;
        }
        let c = (ci % t.w, ci / t.w);
        for (dx, dz) in NEIGHBORS {
            let (nx, nz) = (c.0 as i32 + dx, c.1 as i32 + dz);
            if !t.in_bounds(nx, nz) {
                continue;
            }
            let nc = (nx as usize, nz as usize);
            if blocked(nc.0, nc.1) {
                continue;
            }
            if dx != 0 && dz != 0 && (blocked((c.0 as i32 + dx) as usize, c.1) || blocked(c.0, (c.1 as i32 + dz) as usize)) {
                continue;
            }
            let Some(e) = edge(spec, t, c, nc) else { continue };
            let cost = g + e.move_time_s + e.pause_s;
            let ni = t.idx(nc.0, nc.1);
            if cost < best[ni] && cost <= max_s {
                best[ni] = cost;
                heap.push((Reverse(cost.max(0.0).to_bits()), ni));
            }
        }
    }
    best
}

/// Straight-line distance in metres between the centres of two cells.
pub fn cell_distance_m(a: Cell, b: Cell) -> f32 {
    (((a.0 as f32 - b.0 as f32).powi(2) + (a.1 as f32 - b.1 as f32).powi(2)).sqrt()) * CELL_M
}

#[cfg(test)]
mod tests {
    use super::*;
    use hand_core::BaseKind;

    fn flat(w: usize, d: usize) -> Terrain {
        // A zone covering everything keeps the terrain at height 0.
        Terrain::generate(1, w, d, &[(w as f32 * 0.25, d as f32 * 0.25, w as f32)])
    }

    fn none(_: usize, _: usize) -> bool {
        false
    }

    #[test]
    fn flat_route_time_matches_the_drive_model() {
        use crate::kinds::Ground;
        let mut t = flat(32, 32);
        for x in 0..32 {
            t.set_ground(x, 2, Ground::Rock);
        }
        let spec = BaseKind::Wheel.spec();
        let p = plan(&spec, &t, &none, (2, 2), Goal::Cell((10, 2))).unwrap();
        assert_eq!(p.cells.len(), 9);
        let r = simulate_drive(&spec, &[Segment::Flat { length_m: 4.0 }]);
        // On rock the tyre's slip is a fraction of a percent of the drive model's time.
        let f = hand_core::act::locomotion_factor(&spec, Ground::Rock.surface().props(), 0.0).unwrap();
        assert!(f > 0.99 && f <= 1.0, "{f}");
        assert!((p.time_s - r.time_s / f).abs() < 1e-3, "{} vs {}", p.time_s, r.time_s / f);
        let ef = p.edges[0].energy_factor;
        assert!((p.energy_j - r.energy_j / f * ef).abs() < 1.0, "{} vs {}", p.energy_j, r.energy_j / f * ef);
    }

    #[test]
    fn diagonals_are_used_when_free() {
        let t = flat(32, 32);
        let spec = BaseKind::Wheel.spec();
        let p = plan(&spec, &t, &none, (2, 2), Goal::Cell((8, 8))).unwrap();
        assert_eq!(p.cells.len(), 7, "six diagonal steps");
    }

    #[test]
    fn wheel_detours_around_a_curb_wall_that_a_track_crosses() {
        let mut t = flat(32, 32);
        // A 10 cm wall across the whole column except a gap near the bottom.
        for cz in 0..28 {
            t.set_curb_cm(16, cz, 10);
        }
        let wheel = BaseKind::Wheel.spec();
        let track = BaseKind::Track.spec();
        let direct = Goal::Cell((24, 4));
        let w = plan(&wheel, &t, &none, (8, 4), direct).unwrap();
        let tr = plan(&track, &t, &none, (8, 4), direct).unwrap();
        assert!(w.distance_m > tr.distance_m + 15.0, "wheel {} vs track {}", w.distance_m, tr.distance_m);
        assert!(tr.edges.iter().any(|e| e.step_m > 0.0), "the track climbs the curb");
        assert!(w.edges.iter().all(|e| e.step_m == 0.0), "the wheel never meets a curb");
    }

    #[test]
    fn no_route_when_the_wall_is_closed_for_a_wheel() {
        let mut t = flat(32, 32);
        for cz in 0..32 {
            t.set_curb_cm(16, cz, 10);
        }
        let wheel = BaseKind::Wheel.spec();
        assert!(plan(&wheel, &t, &none, (8, 4), Goal::Cell((24, 4))).is_none());
        let foot = BaseKind::HumanFoot.spec();
        assert!(plan(&foot, &t, &none, (8, 4), Goal::Cell((24, 4))).is_some());
    }

    #[test]
    fn curb_pause_matches_the_drive_model_step_time() {
        let mut t = flat(16, 16);
        t.set_curb_cm(5, 4, 10);
        let spec = BaseKind::Track.spec();
        let e = edge(&spec, &t, (4, 4), (5, 4)).unwrap();
        let step = simulate_drive(&spec, &[Segment::Step { height_m: 0.10 }]);
        assert!((e.pause_s - step.time_s).abs() < 1e-5);
        assert!((e.margin - (1.0 - 0.10 / spec.max_step_m)).abs() < 1e-5);
    }

    #[test]
    fn blocked_cells_are_avoided_and_goal_within_reach_stops_early() {
        let t = flat(32, 32);
        let spec = BaseKind::Wheel.spec();
        let blocked = |x: usize, z: usize| (x, z) == (10, 10);
        let (tx, tz) = cell_center(10, 10);
        let p = plan(&spec, &t, &blocked, (4, 10), Goal::Within { x: tx, z: tz, reach_m: 0.9 }).unwrap();
        let last = *p.cells.last().unwrap();
        assert_ne!(last, (10, 10));
        let (lx, lz) = cell_center(last.0, last.1);
        assert!(((lx - tx).powi(2) + (lz - tz).powi(2)).sqrt() <= 0.9);
        assert!(p.cells.len() <= 8, "stops about 2 cells short instead of walking to the target");
        assert!(plan(&spec, &t, &blocked, (4, 10), Goal::Cell((10, 10))).is_none());
    }

    #[test]
    fn the_ground_entered_sets_its_own_climb_limit_and_energy() {
        use crate::kinds::Ground;
        let mut t = flat(16, 16);
        // 10 cm over 0.5 m: an 11.3° slope.
        t.set_height_cm(8, 4, 10);
        let weak = BaseSpec { drive_accel_mps2: Some(4.6), mass_kg: Some(10.0), ..BaseKind::Wheel.spec() };
        t.set_ground(8, 4, Ground::Rock);
        let rock = edge(&weak, &t, (7, 4), (8, 4)).expect("rock: friction and drive suffice");
        t.set_ground(8, 4, Ground::Sand);
        // On sand the thrust a wheel's short contact gets at full slip (Janosi–Hanamoto,
        // 80% of the friction limit) less the rolling resistance climbs only about 8°,
        // under the friction limit alone (atan 0.25 = 14°).
        assert!(edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).is_none(), "sand: 11° is past the traction at full slip");
        assert!(edge(&weak, &t, (7, 4), (8, 4)).is_none(), "a weak drive spends its force on the sand's rolling resistance");
        // A gentler 5.7° slope: the wheel climbs the sand, slipping, slower than over rock.
        t.set_height_cm(8, 4, 5);
        let sand = edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).expect("sand: 5.7° is within the traction");
        t.set_ground(8, 4, Ground::Rock);
        let firm = edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).unwrap();
        t.set_ground(8, 4, Ground::Sand);
        assert!(sand.ground_limit_deg < 15.0 && sand.margin < firm.margin && sand.move_time_s > firm.move_time_s, "sand {} s vs rock {} s", sand.move_time_s, firm.move_time_s);
        assert!(rock.margin > 0.0);
        // Level sand: passable, but the rolling resistance costs energy; legs pay the walking factor.
        t.set_height_cm(8, 4, 0);
        let level = edge(&weak, &t, (7, 4), (8, 4)).unwrap();
        assert!(level.energy_factor > 1.2, "{}", level.energy_factor);
        let walk = edge(&BaseKind::HumanFoot.spec(), &t, (7, 4), (8, 4)).unwrap();
        assert!((walk.energy_factor - 2.1).abs() < 1e-6);
        t.set_ground(8, 4, Ground::Gravel);
        assert!(edge(&weak, &t, (7, 4), (8, 4)).unwrap().energy_factor < level.energy_factor);
    }

    #[test]
    fn a_pushed_item_closes_the_slopes_the_drive_cannot_push_it_up() {
        use crate::kinds::Ground;
        let mut t = flat(16, 16);
        // 11.3° of grass.
        t.set_height_cm(8, 4, 10);
        t.set_ground(8, 4, Ground::Grass);
        let spec = BaseSpec { mass_kg: Some(19.3), drive_accel_mps2: Some(240.0 / 19.3), ..BaseKind::Wheel.spec() };
        assert!(edge(&spec, &t, (7, 4), (8, 4)).is_some(), "alone the wheels climb it");
        let ore = PushLoad { mass_kg: 4.0, material: hand_core::act::material::ItemMaterial::IronOre };
        assert!(edge_with(&spec, &t, (7, 4), (8, 4), Some(ore)).is_none(), "pushing 4 kg of ore up it is past the grass's traction");
        assert!(edge_with(&spec, &t, (8, 4), (7, 4), Some(ore)).is_some(), "down it the slope helps");
    }

    #[test]
    fn steep_hills_block_weak_bases_first() {
        let mut t = flat(16, 16);
        // Raise a 15 cm ridge: 0.15 / 0.5 m is about 16.7 degrees.
        for cz in 0..16 {
            t.set_height_cm(8, cz, 15);
        }
        let wheel = BaseKind::Wheel.spec();
        let track = BaseKind::Track.spec();
        assert!(edge(&wheel, &t, (7, 4), (8, 4)).is_none());
        assert!(edge(&track, &t, (7, 4), (8, 4)).is_some());
    }
}
