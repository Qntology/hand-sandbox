//! Driving the planned cell path like a vehicle or a walker, not like a cursor: the base follows the path with a
//! pure-pursuit steering law (Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm", CMU-RI-TR-92-01,
//! 1992), so it rounds corners on arcs at the speed the arc allows (`steer::TurnCaps`), stops to turn only where the
//! path bends more than it can steer, turns there the way the base can (scrub pivot, K-turn, step turn), and reaches
//! a goal close behind it in reverse. The cell edges keep their physics (speed on the ground and slope, curb pauses,
//! footsteps, slip): the edge under the base sets the speed at each moment.

use super::*;
use crate::steer::{self, Seg, TurnCaps, TurnKind};

pub const LOOKAHEAD_MIN_M: f32 = 0.35;
pub const LOOKAHEAD_MAX_M: f32 = 1.0;
pub const LOOKAHEAD_S: f32 = 0.6;
pub const ALPHA_STOP_SKID_RAD: f32 = 1.2;
pub const ALPHA_STOP_LEGS_RAD: f32 = 1.05;
pub const REVERSE_BEHIND_RAD: f32 = 2.0;
pub const REVERSE_STRAIGHT_RAD: f32 = 0.8;
pub const SUBSTEP_S: f32 = 0.05;
pub const CREEP_RAD_S: f32 = 0.25;
pub const MIN_CURVE_MPS: f32 = 0.08;
pub const GOAL_TOL_M: f32 = 0.06;
pub const MAX_STOP_TURNS: u32 = 6;
pub const STALL_S: f32 = 2.5;
pub const FINAL_APPROACH_M: f32 = 0.45;
pub const STEER_MAX_RAD: f32 = 0.75;
pub const STEER_RATE_RAD_S: f32 = 1.0;

#[derive(Clone, Debug)]
pub(crate) struct SegRun {
    pub kind: &'static str,
    pub label_ko: String,
    pub segs: Vec<Seg>,
    pub i: usize,
    pub u: f32,
    pub start: (f32, f32, f32),
    pub wait_left: f32,
    pub forced: bool,
}

impl SegRun {
    pub fn new(m: steer::Maneuver, at: (f32, f32, f32)) -> SegRun {
        SegRun { kind: m.kind, label_ko: m.label_ko, segs: m.segs, i: 0, u: 0.0, start: at, wait_left: 0.0, forced: false }
    }

    pub fn done(&self) -> bool {
        self.i >= self.segs.len()
    }

    pub fn reversing(&self) -> bool {
        matches!(self.segs.get(self.i), Some(Seg::Drive { dir, .. }) if *dir < 0.0)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Follow {
    pub path: Vec<(f32, f32)>,
    pub cum: Vec<f32>,
    pub edge_at: Vec<usize>,
    pub s: f32,
    pub reverse: bool,
    pub man: Option<SegRun>,
    pub note: Option<(&'static str, String)>,
    pub turns: u32,
    pub on_rails: bool,
    pub best_s: f32,
    pub stall_s: f32,
    pub stop_within: Option<((f32, f32), f32)>,
    pub steer: f32,
}

impl Follow {
    pub fn total_m(&self) -> f32 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    pub fn point_at(&self, s: f32) -> (f32, f32) {
        let s = s.clamp(0.0, self.total_m());
        let k = self.cum.partition_point(|c| *c <= s).clamp(1, self.path.len().max(2) - 1);
        let (a, b) = (self.path[k - 1], self.path[k]);
        let len = (self.cum[k] - self.cum[k - 1]).max(1e-6);
        let u = ((s - self.cum[k - 1]) / len).clamp(0.0, 1.0);
        (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
    }

    pub fn edge_index(&self, s: f32) -> usize {
        let k = self.cum.partition_point(|c| *c <= s.max(0.0)).clamp(1, self.path.len().max(2) - 1);
        self.edge_at.get(k - 1).copied().unwrap_or(0)
    }

    fn project(&self, p: (f32, f32), from: f32, ahead: f32) -> f32 {
        let (mut best, mut best_d) = (from, f32::INFINITY);
        for k in 1..self.path.len() {
            if self.cum[k] < from - 1e-4 || self.cum[k - 1] > from + ahead {
                continue;
            }
            let (a, b) = (self.path[k - 1], self.path[k]);
            let (dx, dz) = (b.0 - a.0, b.1 - a.1);
            let l2 = (dx * dx + dz * dz).max(1e-9);
            let u = (((p.0 - a.0) * dx + (p.1 - a.1) * dz) / l2).clamp(0.0, 1.0);
            let (qx, qz) = (a.0 + dx * u, a.1 + dz * u);
            let d = (p.0 - qx).hypot(p.1 - qz);
            let s = self.cum[k - 1] + u * l2.sqrt();
            if d < best_d && s >= from - 1e-4 {
                (best, best_d) = (s, d);
            }
        }
        best.max(from)
    }

    /// How much the path turns in total (rad), corner by corner.
    pub fn turning(&self) -> f32 {
        self.path.windows(3).map(|w| steer::wrap((w[2].1 - w[1].1).atan2(w[2].0 - w[1].0) - (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0)).abs()).sum()
    }
}

pub(crate) fn follow_of(start: (f32, f32), edges: &[EdgePlan]) -> Follow {
    let mut path = vec![start];
    let mut edge_at = Vec::new();
    for (i, e) in edges.iter().enumerate() {
        let p = cell_center(e.to.0, e.to.1);
        if (p.0 - path.last().unwrap().0).hypot(p.1 - path.last().unwrap().1) < 1e-4 {
            continue;
        }
        path.push(p);
        edge_at.push(i);
    }
    let mut cum = vec![0.0];
    for w in path.windows(2) {
        cum.push(cum.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
    }
    Follow { path, cum, edge_at, ..Follow::default() }
}

impl World {
    pub fn turn_caps(&self) -> TurnCaps {
        steer::caps(&self.drive_spec(), self.body.support_len_m, self.body.support_width_m, self.body.chassis.drive_force_n, self.ground_here().surface().props())
    }

    pub(crate) fn is_track(&self) -> bool {
        self.hand.base.kind == noto_core::BaseKind::Track
    }

    /// Ground the base may stand on while it manoeuvres: on the map, dry, unobstructed, and no curb or ledge away
    /// from where it stands now.
    pub(crate) fn free_at(&self, x: f32, z: f32) -> bool {
        let here = self.agent_cell();
        let blocked = self.blocked_cells();
        let h0 = self.terrain.height_m(here.0, here.1) + self.terrain.curb_m(here.0, here.1);
        let step = self.drive_spec().max_step_m;
        match self.terrain.cell_of(x, z) {
            Some(c) if c == here => true,
            Some(c) => !self.terrain.is_water(c.0, c.1) && !blocked.contains(&c) && (self.terrain.height_m(c.0, c.1) + self.terrain.curb_m(c.0, c.1) - h0).abs() <= step + 0.01,
            None => false,
        }
    }

    pub(crate) fn turn_run(&self, target: f32) -> SegRun {
        let pose = (self.agent.x, self.agent.z, self.agent.yaw);
        let delta = wrap_angle(target - self.agent.yaw);
        let caps = self.turn_caps();
        let v = self.drive_spec().max_speed_mps;
        match steer::turn_in_place(pose, delta, &caps, self.is_track(), v, &|x, z| self.free_at(x, z)) {
            Some(m) => SegRun::new(m, pose),
            None => SegRun {
                forced: true,
                ..SegRun::new(steer::Maneuver { kind: "pivot", label_ko: format!("자리가 좁아 제자리에서 억지로 {:.0}° 비벼 돎 (스키드 조향 힘 부족)", delta.abs().to_degrees()), segs: vec![Seg::Pivot { angle: delta }], time_s: 0.0 }, pose)
            },
        }
    }

    /// Plays `run` for up to `remaining` s. Returns the time used.
    pub(crate) fn advance_run(&mut self, run: &mut SegRun, remaining: f32) -> f32 {
        if run.wait_left > 0.0 {
            let t = run.wait_left.min(remaining);
            run.wait_left -= t;
            self.agent.speed_mps = 0.0;
            self.add_energy(0.2 * self.drive_spec().power_w as f64 * t as f64);
            return t;
        }
        let Some(seg) = run.segs.get(run.i).copied() else { return 0.0 };
        let caps = self.turn_caps();
        let v_fwd = self.drive_spec().max_speed_mps * if run.kind == "kturn" { 0.5 } else { 1.0 };
        let (total, speed) = match seg {
            Seg::Pivot { angle } => {
                let rate = if caps.can_pivot() { caps.pivot_rad_s } else { CREEP_RAD_S };
                (angle.abs() / rate.max(0.05), 0.0)
            }
            Seg::Drive { .. } => {
                let t = steer::seg_time(&seg, &caps, v_fwd);
                (t, seg.length_m() / t.max(1e-6))
            }
        };
        let left = (1.0 - run.u) * total;
        let t = left.min(remaining).max(0.0);
        run.u = if total > 1e-6 { (run.u + t / total).min(1.0) } else { 1.0 };
        let (x, z, yaw) = steer::along(run.start, &seg, run.u);
        let moved = (x - self.agent.x).hypot(z - self.agent.z);
        (self.agent.x, self.agent.z, self.agent.yaw) = (x, z, wrap_angle(yaw));
        self.agent.speed_mps = speed;
        let scrub = match (seg, caps.kind) {
            (Seg::Pivot { .. }, TurnKind::Skid) => if run.forced { 2.0 } else { 1.3 },
            _ => 1.0,
        };
        self.add_energy(self.drive_spec().power_w as f64 * scrub * t as f64);
        if moved > 0.0 && self.body.legged() {
            self.walk.since_step_m += moved;
        }
        if run.u >= 1.0 - 1e-6 {
            run.i += 1;
            run.u = 0.0;
            run.start = (self.agent.x, self.agent.z, self.agent.yaw);
            let flips = match (seg, run.segs.get(run.i)) {
                (Seg::Drive { dir: a, .. }, Some(Seg::Drive { dir: b, .. })) => a != *b,
                _ => false,
            };
            if flips {
                run.wait_left = steer::SHIFT_S;
            }
        }
        t
    }

    pub(crate) fn start_follow(&self, edges: &[EdgePlan]) -> Follow {
        let mut f = follow_of((self.agent.x, self.agent.z), edges);
        let caps = self.turn_caps();
        if f.path.len() < 2 || caps.kind == TurnKind::Holonomic {
            return f;
        }
        let first = (f.path[1].1 - f.path[0].1).atan2(f.path[1].0 - f.path[0].0);
        let delta = wrap_angle(first - self.agent.yaw);
        if delta.abs() > REVERSE_BEHIND_RAD && f.total_m() <= caps.reverse_max_m && f.turning() <= REVERSE_STRAIGHT_RAD {
            f.reverse = true;
            let how = if caps.kind == TurnKind::Legs { "뒷걸음" } else { "후진" };
            f.note = Some(("reverse", format!("목표가 뒤쪽 {:.1} m: 돌지 않고 {how}", f.total_m())));
        }
        f
    }

    /// One step of path following. Returns the time used.
    pub(super) fn advance_follow(&mut self, m: &mut MoveState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        if let Some(run) = m.follow.man.as_mut() {
            let mut r = run.clone();
            let t = self.advance_run(&mut r, remaining);
            m.follow.man = (!r.done() || r.wait_left > 0.0).then_some(r);
            if m.follow.man.is_none() {
                m.follow.s = m.follow.project((self.agent.x, self.agent.z), m.follow.s, 1.5);
            }
            m.elapsed_s += t;
            return t.max(1e-4);
        }
        let total = m.follow.total_m();
        if m.edges.is_empty() || total <= GOAL_TOL_M {
            *done = Some((true, None));
            return 0.0;
        }
        let k = m.follow.edge_index(m.follow.s).min(m.edges.len() - 1);
        if k != m.idx {
            m.idx = k;
            m.announced_pause = false;
            m.progress_s = 0.0;
        }
        let e = m.edges[k];
        self.agent.terrain_margin = Some(e.margin);
        let power = self.drive_spec().power_w as f64;
        if e.pause_s > 0.0 && !m.announced_pause {
            m.announced_pause = true;
            m.pause_left = e.pause_s;
            events.push(self.stamp(WorldEvent::CurbClimbed { height_m: e.step_m }));
        }
        if m.pause_left > 0.0 {
            let t = m.pause_left.min(remaining);
            m.pause_left -= t;
            self.agent.speed_mps = 0.0;
            self.add_energy(power * t as f64);
            return t;
        }
        let goal = *m.follow.path.last().expect("a path has an end");
        let pos = (self.agent.x, self.agent.z);
        if let Some((p, reach)) = m.follow.stop_within {
            if (pos.0 - p.0).hypot(pos.1 - p.1) <= reach {
                self.agent.speed_mps = 0.0;
                *done = Some((true, None));
                return 0.0;
            }
        }
        let left = (total - m.follow.s).max(0.0).min(total);
        if left <= GOAL_TOL_M || (pos.0 - goal.0).hypot(pos.1 - goal.1) <= GOAL_TOL_M {
            (self.agent.x, self.agent.z) = goal;
            m.idx = m.edges.len() - 1;
            *done = Some((true, None));
            return 0.0;
        }
        let caps = self.turn_caps();
        let v_edge = e.horizontal_m / e.move_time_s.max(1e-6);
        if m.follow.on_rails {
            return self.advance_on_rails(m, &e, remaining, left, v_edge, events, done);
        }
        let to_goal = (pos.0 - goal.0).hypot(pos.1 - goal.1);
        let ld = (v_edge * LOOKAHEAD_S).clamp(LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M).min(if caps.kind == TurnKind::Legs { 0.6 } else { LOOKAHEAD_MAX_M });
        let target = if to_goal < FINAL_APPROACH_M { goal } else { m.follow.point_at(m.follow.s + ld) };
        let (tx, tz) = (target.0 - pos.0, target.1 - pos.1);
        let dist = tx.hypot(tz).max(1e-4);
        let heading = if m.follow.reverse { self.agent.yaw + std::f32::consts::PI } else { self.agent.yaw };
        let alpha = wrap_angle(tz.atan2(tx) - heading);
        let stop_at = if caps.kind == TurnKind::Legs { ALPHA_STOP_LEGS_RAD } else { ALPHA_STOP_SKID_RAD };
        let near_end = to_goal < FINAL_APPROACH_M && alpha.abs() > 0.6;
        let stalled = m.follow.stall_s > STALL_S;
        if caps.kind != TurnKind::Holonomic && (alpha.abs() > stop_at || near_end || stalled) && dist > 0.03 {
            m.follow.stall_s = 0.0;
            m.follow.steer = 0.0;
            if m.follow.turns >= MAX_STOP_TURNS {
                m.follow.on_rails = true;
                return 1e-4;
            }
            m.follow.turns += 1;
            let want = tz.atan2(tx) - if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
            let run = self.turn_run(wrap_angle(want));
            m.follow.note = Some((run.kind, run.label_ko.clone()));
            self.agent.speed_mps = 0.0;
            m.follow.man = Some(run);
            return 1e-4;
        }
        let dt = remaining.min(SUBSTEP_S);
        let mut v = if m.follow.reverse { v_edge.min(caps.reverse_mps) } else { v_edge };
        let mut kappa = 2.0 * alpha.sin() / dist;
        if caps.kind == TurnKind::Skid && caps.min_radius_m > 0.0 {
            kappa = kappa.clamp(-1.0 / caps.min_radius_m, 1.0 / caps.min_radius_m);
        }
        if caps.kind == TurnKind::Skid {
            // The steering lock swings at a finite rate and stops at a limit, as a hydraulic tractor's does, so the
            // curvature ramps instead of jumping to whatever the pursuit law asks for in one tick.
            let l = self.body.support_len_m.max(0.1);
            let want = (kappa * l).atan().clamp(-STEER_MAX_RAD, STEER_MAX_RAD);
            let step = STEER_RATE_RAD_S * dt;
            m.follow.steer += (want - m.follow.steer).clamp(-step, step);
            kappa = m.follow.steer.tan() / l;
        }
        if caps.kind != TurnKind::Holonomic && kappa.abs() > 1e-3 {
            v = v.min(caps.speed_on_radius(1.0 / kappa.abs(), v).max(MIN_CURVE_MPS));
        }
        let brake = self.planned_accel(e.slope_deg, self.ground_at(e.to), -1.0).max(0.3);
        v = v.min((2.0 * brake * left).sqrt() + 0.05);
        let ds = (v * dt).min(left + 0.01);
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
        let (nx, nz, nyaw) = match caps.kind {
            TurnKind::Holonomic => {
                let turn = (caps.pivot_rad_s * dt).min(alpha.abs()) * alpha.signum();
                (pos.0 + ds * tx / dist, pos.1 + ds * tz / dist, self.agent.yaw + turn)
            }
            _ if m.follow.reverse => steer::along(before, &Seg::Drive { dir: -1.0, kappa: -kappa, len: ds }, 1.0),
            _ => steer::along(before, &Seg::Drive { dir: 1.0, kappa, len: ds }, 1.0),
        };
        let (nx, nz, nyaw) = if self.free_ahead(nx, nz) {
            (nx, nz, nyaw)
        } else {
            let p = m.follow.point_at(m.follow.s + ds);
            let ahead = m.follow.point_at(m.follow.s + ds + 0.1);
            let yaw = (ahead.1 - p.1).atan2(ahead.0 - p.0) + if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
            (p.0, p.1, yaw)
        };
        (self.agent.x, self.agent.z, self.agent.yaw) = (nx, nz, wrap_angle(nyaw));
        m.follow.s = m.follow.project((nx, nz), m.follow.s, ds + 0.3);
        if m.follow.s > m.follow.best_s + 0.02 {
            (m.follow.best_s, m.follow.stall_s) = (m.follow.s, 0.0);
        } else {
            m.follow.stall_s += dt;
        }
        let moved = (nx - pos.0).hypot(nz - pos.1);
        self.agent.speed_mps = moved / dt.max(1e-6);
        m.progress_s += dt;
        m.elapsed_s += dt;
        if !self.body.legged() {
            let props = self.ground_at(e.to).surface().props();
            self.odom_over = noto_core::act::cruise_slip(&self.drive_spec(), props, e.slope_deg).map_or(0.0, |s| s / (1.0 - s).max(0.05));
        }
        if let Some(why) = self.walk_steps(moved, &e, events) {
            *done = Some((false, Some(format!("fall: {why}"))));
            return dt;
        }
        self.add_energy(power * e.energy_factor as f64 * dt as f64);
        dt
    }

    /// Fallback after too many stops to turn: slide along the path line itself, facing along it.
    #[allow(clippy::too_many_arguments)]
    fn advance_on_rails(&mut self, m: &mut MoveState, e: &EdgePlan, remaining: f32, left: f32, v_edge: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let dt = remaining.min(SUBSTEP_S);
        let ds = (v_edge * dt).min(left);
        let pos = (self.agent.x, self.agent.z);
        let p = m.follow.point_at(m.follow.s + ds);
        let ahead = m.follow.point_at(m.follow.s + ds + 0.1);
        let flip = if m.follow.reverse { std::f32::consts::PI } else { 0.0 };
        if (ahead.0 - p.0).hypot(ahead.1 - p.1) > 1e-4 {
            self.agent.yaw = wrap_angle((ahead.1 - p.1).atan2(ahead.0 - p.0) + flip);
        }
        (self.agent.x, self.agent.z) = p;
        m.follow.s += ds;
        let moved = (p.0 - pos.0).hypot(p.1 - pos.1);
        self.agent.speed_mps = moved / dt.max(1e-6);
        m.elapsed_s += dt;
        if let Some(why) = self.walk_steps(moved, e, events) {
            *done = Some((false, Some(format!("fall: {why}"))));
        }
        self.add_energy(self.drive_spec().power_w as f64 * e.energy_factor as f64 * dt as f64);
        dt
    }

    fn free_ahead(&self, x: f32, z: f32) -> bool {
        match self.terrain.cell_of(x, z) {
            Some(c) => !self.terrain.is_water(c.0, c.1) && !self.blocked_cells().contains(&c),
            None => false,
        }
    }

    pub fn motion_view(&self) -> Option<MotionView> {
        match &self.active {
            Some((_, Active::Move(m))) => {
                if let Some(r) = &m.follow.man {
                    return Some(MotionView { kind: r.kind.into(), label_ko: r.label_ko.clone(), reversing: r.reversing() });
                }
                if m.follow.reverse {
                    return Some(MotionView { kind: "reverse".into(), label_ko: m.follow.note.as_ref().map(|n| n.1.clone()).unwrap_or_default(), reversing: true });
                }
                (self.agent.speed_mps > 0.0).then(|| MotionView { kind: "follow".into(), label_ko: String::new(), reversing: false })
            }
            Some((_, Active::Turn { run, .. })) => Some(MotionView { kind: run.kind.into(), label_ko: run.label_ko.clone(), reversing: run.reversing() }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainKind;
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        let mut w = World::new(WorldConfig { seed: 11, cells: 64, noto: hand, body: BodyPlan::default(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let (cx, cz) = w.agent_cell();
        (w.agent.x, w.agent.z) = cell_center(cx, cz);
        w.agent.yaw = 0.0;
        w
    }

    fn run(w: &mut World, max_ticks: usize) -> Vec<(f32, f32, f32, f32, bool)> {
        let mut trace = Vec::new();
        for _ in 0..max_ticks {
            w.tick();
            let rev = w.motion_view().is_some_and(|m| m.reversing);
            trace.push((w.agent.x, w.agent.z, w.agent.yaw, w.agent.speed_mps, rev));
            if !w.busy() {
                break;
            }
        }
        trace
    }

    #[test]
    fn a_wheeled_base_rounds_a_corner_without_stopping() {
        let mut w = world(BaseKind::Wheel);
        let (x0, z0) = (w.agent.x, w.agent.z);
        let goal = (x0 + 3.0, z0 + 3.0);
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let tr = run(&mut w, 400);
        assert!(!w.busy(), "arrived");
        assert!((w.agent.x - goal.0).hypot(w.agent.z - goal.1) < 0.3);
        let moving: Vec<&(f32, f32, f32, f32, bool)> = tr.iter().skip(3).take(tr.len().saturating_sub(8)).collect();
        assert!(moving.iter().all(|t| t.3 > 0.05), "it never halts mid-route: {:?}", moving.iter().map(|t| t.3).collect::<Vec<_>>());
        let jumps = tr.windows(2).map(|p| wrap_angle(p[1].2 - p[0].2).abs()).fold(0.0f32, f32::max);
        assert!(jumps < 0.35, "the heading turns smoothly, at most {jumps} rad a tick");
    }

    #[test]
    fn the_steering_lock_swings_at_a_finite_rate() {
        let mut w = world(BaseKind::Track);
        let (x0, z0) = (w.agent.x, w.agent.z);
        w.command(Action::MoveTo { x: x0 + 3.0, z: z0 + 3.0, reach_m: 0.0 }).unwrap();
        let mut steer = 0.0f32;
        for _ in 0..400 {
            w.tick();
            if let Some(Active::Move(m)) = w.active.as_ref().map(|a| &a.1) {
                assert!((m.follow.steer - steer).abs() < STEER_RATE_RAD_S * 0.2, "the lock jumped {} rad", m.follow.steer - steer);
                steer = m.follow.steer;
            }
            if !w.busy() {
                break;
            }
        }
        assert!(!w.busy());
    }

    #[test]
    fn a_goal_close_behind_is_reached_in_reverse() {
        for (base, back_m) in [(BaseKind::Wheel, 1.5), (BaseKind::Track, 1.5), (BaseKind::HumanFoot, 1.0)] {
            let mut w = world(base);
            let (x0, z0) = (w.agent.x, w.agent.z);
            w.command(Action::MoveTo { x: x0 - back_m, z: z0, reach_m: 0.0 }).unwrap();
            let tr = run(&mut w, 300);
            assert!(!w.busy(), "{base:?} arrived");
            assert!(tr.iter().any(|t| t.4), "{base:?} backs up");
            assert!(wrap_angle(w.agent.yaw).abs() < 0.2, "{base:?} still faces the way it did: {}", w.agent.yaw);
        }
    }

    #[test]
    fn a_track_on_soft_ground_turns_round_with_a_k_turn() {
        let mut w = world(BaseKind::Track);
        let (cx, cz) = w.agent_cell();
        for dz in -4i32..=4 {
            for dx in -4i32..=4 {
                let (x, z) = ((cx as i32 + dx) as usize, (cz as i32 + dz) as usize);
                w.terrain.set_ground(x, z, crate::kinds::Ground::Dirt);
            }
        }
        assert!(w.turn_caps().prefers_kturn(true), "{:?}", w.turn_caps());
        w.command(Action::Turn { yaw_deg: 150.0 }).unwrap();
        assert_eq!(w.motion_view().map(|m| m.kind), Some("kturn".into()));
        let tr = run(&mut w, 400);
        assert!(!w.busy());
        assert!(tr.iter().any(|t| t.4) && tr.iter().any(|t| !t.4 && t.3 > 0.01), "forward and reverse legs");
        assert!((wrap_angle(w.agent.yaw) - 150f32.to_radians()).abs() < 0.05, "{}", w.agent.yaw.to_degrees());
        let mut walker = world(BaseKind::HumanFoot);
        walker.command(Action::Turn { yaw_deg: 150.0 }).unwrap();
        assert_eq!(walker.motion_view().map(|m| m.kind), Some("pivot".into()), "legs step round where they stand");
    }
}
