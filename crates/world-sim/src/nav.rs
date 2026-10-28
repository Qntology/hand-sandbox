use std::cmp::Reverse;
use std::collections::BinaryHeap;

use noto_core::surface::{climb_limit_deg, energy_factor};
use noto_core::{simulate_drive, BaseSpec, Terrain as Segment};

use crate::terrain::{cell_center, Terrain, CELL_M};

pub type Cell = (usize, usize);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgePlan {
    pub from: Cell,
    pub to: Cell,
    pub horizontal_m: f32,
    pub move_time_s: f32,
    pub pause_s: f32,
    pub step_m: f32,
    pub slope_deg: f32,
    pub margin: f32,
    pub ground_limit_deg: f32,
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
    Cell(Cell),
    Within { x: f32, z: f32, reach_m: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PushLoad {
    pub mass_kg: f32,
    pub material: noto_core::act::material::ItemMaterial,
}

pub fn edge(spec: &BaseSpec, t: &Terrain, a: Cell, b: Cell) -> Option<EdgePlan> {
    edge_with(spec, t, a, b, None)
}

pub fn edge_with(spec: &BaseSpec, t: &Terrain, a: Cell, b: Cell, load: Option<PushLoad>) -> Option<EdgePlan> {
    if t.is_water(b.0, b.1) {
        return None;
    }
    let (ax, az) = cell_center(a.0, a.1);
    let (bx, bz) = cell_center(b.0, b.1);
    let horizontal = ((ax - bx).powi(2) + (az - bz).powi(2)).sqrt();
    let ramp = t.ramp_rise_m(a, b);
    let rise = t.height_m(b.0, b.1) - t.height_m(a.0, a.1) + ramp.unwrap_or(0.0);
    let step = if ramp.is_some() { 0.0 } else { (t.curb_m(b.0, b.1) - t.curb_m(a.0, a.1)).abs() };

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
    let props = t.ground_at(b.0, b.1).surface().props();
    let ground_limit = climb_limit_deg(spec.kind, props, spec.drive_accel_mps2);
    if slope_deg > ground_limit {
        return None;
    }
    let signed = if rise >= 0.0 { slope_deg } else { -slope_deg };
    let extra = match (load, spec.mass_kg) {
        (Some(l), Some(m)) => {
            let th = signed.to_radians();
            let mu = noto_core::act::material::item_on(l.material, t.ground_at(b.0, b.1).surface()).0.mean;
            (l.mass_kg * noto_core::balance::G * (mu * th.cos() + th.sin())).max(0.0) / m.max(1e-3)
        }
        _ => 0.0,
    };
    let factor = noto_core::act::locomotion_factor_with(spec, props, signed, extra)?;
    let move_time = ground_only.time_s / factor;
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

pub fn plan(spec: &BaseSpec, t: &Terrain, blocked: &dyn Fn(usize, usize) -> bool, start: Cell, goal: Goal) -> Option<Plan> {
    plan_with(spec, t, blocked, start, goal, None)
}

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

pub fn cell_distance_m(a: Cell, b: Cell) -> f32 {
    (((a.0 as f32 - b.0 as f32).powi(2) + (a.1 as f32 - b.1 as f32).powi(2)).sqrt()) * CELL_M
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::BaseKind;

    fn flat(w: usize, d: usize) -> Terrain {
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
        let f = noto_core::act::locomotion_factor(&spec, Ground::Rock.surface().props(), 0.0).unwrap();
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
        t.set_height_cm(8, 4, 10);
        let weak = BaseSpec { drive_accel_mps2: Some(4.6), mass_kg: Some(10.0), ..BaseKind::Wheel.spec() };
        t.set_ground(8, 4, Ground::Rock);
        let rock = edge(&weak, &t, (7, 4), (8, 4)).expect("rock: friction and drive suffice");
        t.set_ground(8, 4, Ground::Sand);
        assert!(edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).is_none(), "sand: 11° is past the traction at full slip");
        assert!(edge(&weak, &t, (7, 4), (8, 4)).is_none(), "a weak drive spends its force on the sand's rolling resistance");
        t.set_height_cm(8, 4, 5);
        let sand = edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).expect("sand: 5.7° is within the traction");
        t.set_ground(8, 4, Ground::Rock);
        let firm = edge(&BaseKind::Wheel.spec(), &t, (7, 4), (8, 4)).unwrap();
        t.set_ground(8, 4, Ground::Sand);
        assert!(sand.ground_limit_deg < 15.0 && sand.margin < firm.margin && sand.move_time_s > firm.move_time_s, "sand {} s vs rock {} s", sand.move_time_s, firm.move_time_s);
        assert!(rock.margin > 0.0);
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
        t.set_height_cm(8, 4, 10);
        t.set_ground(8, 4, Ground::Grass);
        let spec = BaseSpec { mass_kg: Some(19.3), drive_accel_mps2: Some(240.0 / 19.3), ..BaseKind::Wheel.spec() };
        assert!(edge(&spec, &t, (7, 4), (8, 4)).is_some(), "alone the wheels climb it");
        let ore = PushLoad { mass_kg: 4.0, material: noto_core::act::material::ItemMaterial::IronOre };
        assert!(edge_with(&spec, &t, (7, 4), (8, 4), Some(ore)).is_none(), "pushing 4 kg of ore up it is past the grass's traction");
        assert!(edge_with(&spec, &t, (8, 4), (7, 4), Some(ore)).is_some(), "down it the slope helps");
    }

    #[test]
    fn steep_hills_block_weak_bases_first() {
        let mut t = flat(16, 16);
        for cz in 0..16 {
            t.set_height_cm(8, cz, 15);
        }
        let wheel = BaseKind::Wheel.spec();
        let track = BaseKind::Track.spec();
        assert!(edge(&wheel, &t, (7, 4), (8, 4)).is_none());
        assert!(edge(&track, &t, (7, 4), (8, 4)).is_some());
    }
}
