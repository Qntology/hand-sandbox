//! Robots sharing one map: who stands where (the first worker, its rider when it has stepped off, the sub-workers),
//! what each one's eye sees ahead on its way, and how they keep clear of each other. Routes go round the robots that
//! stand still; a robot on the move stops for a robot its camera sees in its way — inside the camera's horizontal
//! field of view, within its stopping distance, not hidden behind a tree or a boulder — and when the way does not
//! clear it routes round that robot. A humanoid gives way sooner than a vehicle, and both give way to the first
//! worker, so two robots meeting head on do not wait for each other forever.
//!
//! A lift truck's rider fells a node on foot (`Rider`): the truck stops short of the node, the rider steps down, walks
//! over, swings at it, walks back and climbs on again; the head camera goes with the rider.

use std::collections::HashSet;

use serde::Serialize;

use super::*;

/// Half the head camera's horizontal field of view, which the way-ahead check looks over (deg) [직접: 70° camera].
pub const SIGHT_HALF_FOV_DEG: f32 = 35.0;
/// The farthest the way-ahead check looks (m).
pub const SIGHT_RANGE_M: f32 = 6.0;
/// Clearance kept between two robots' outlines when one stops for the other (m) [추정: a hand's breadth beyond the
/// bodies], plus the distance it covers in a second at its speed (reaction and braking).
pub const KEEP_CLEAR_M: f32 = 0.45;
/// Seconds a robot waits for its way to clear before it routes round: humanoids give way first.
pub const YIELD_HUMANOID_S: f32 = 1.5;
pub const YIELD_VEHICLE_S: f32 = 3.0;
/// A rider getting off and on a lift truck (s), and walking (m/s) [추정: three points of contact take 3–4 s; a
/// careful walk on site 0.8 m/s].
pub const DISMOUNT_S: f32 = 3.0;
pub const REMOUNT_S: f32 = 3.0;
pub const RIDER_WALK_MPS: f32 = 0.8;
/// The radius a standing rider keeps others at (m).
pub const RIDER_R_M: f32 = 0.28;
/// How far from a node a lift truck stops for its rider to walk over and fell it, and the farthest it may (m): the
/// tines' tips stay clear of the trunk.
pub const RIDER_STOP_M: f32 = 1.8;
pub const RIDER_REACH_M: f32 = 3.5;

/// The distance `RIDER_STOP_M` for this truck: its tine tips and a margin, if they reach further.
pub fn rider_stop_m(body: &noto_core::balance::Body) -> f32 {
    RIDER_STOP_M.max(crate::clear::Footprint::of(body).front_m + 0.3)
}

/// The distance `RIDER_REACH_M` for this truck.
pub fn rider_reach_m(body: &noto_core::balance::Body) -> f32 {
    RIDER_REACH_M + rider_stop_m(body) - RIDER_STOP_M
}

/// The person who rides a truck, on foot.
pub fn rider_body(body: &noto_core::balance::Body) -> noto_core::balance::Body {
    use noto_core::{BaseKind, BodyPlan, Morphology};
    noto_core::balance::Body::new(&BodyPlan { morph: Morphology::humanoid(BaseKind::HumanFoot), ..body.plan }, BaseKind::HumanFoot, body.fingers)
}

/// A standing rider's eye height (m).
pub fn rider_eye_m(body: &noto_core::balance::Body) -> f32 {
    let on_foot = rider_body(body);
    noto_core::kin::head_camera(&on_foot, &noto_core::kin::KinPose::stand(&on_foot, 0.0)).pos[2]
}
/// The id the first worker's dismounted rider goes by among the robots.
pub const RIDER_ID: u32 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RiderPhase {
    Dismount,
    WalkOut,
    Mine,
    WalkBack,
    Remount,
}

impl RiderPhase {
    pub fn label_ko(self) -> &'static str {
        match self {
            RiderPhase::Dismount => "하차",
            RiderPhase::WalkOut => "걸어가기",
            RiderPhase::Mine => "채취",
            RiderPhase::WalkBack => "돌아오기",
            RiderPhase::Remount => "재탑승",
        }
    }
}

/// A lift truck's rider on foot, felling a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rider {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub phase: RiderPhase,
    pub t: f32,
    /// Where it steps down and back on (beside the seat), where it stands to swing, and the node.
    pub seat: (f32, f32),
    pub stand: (f32, f32),
    pub node: (f32, f32),
    pub failed: bool,
    /// A step aside round a robot standing in its way, walked before the rest; and how long it has waited.
    pub via: Option<(f32, f32)>,
    pub held_s: f32,
}

impl Rider {
    /// The rider of a truck at (x, z) facing `yaw`, about to step down and fell the node at `node`.
    pub fn start(x: f32, z: f32, yaw: f32, node: (f32, f32)) -> Rider {
        let (s, c) = yaw.sin_cos();
        // Steps down on the side nearer the node.
        let side = if (node.0 - x) * -s + (node.1 - z) * c >= 0.0 { 1.0 } else { -1.0 };
        let seat = (x - s * 0.7 * side, z + c * 0.7 * side);
        let (dx, dz) = (node.0 - seat.0, node.1 - seat.1);
        let d = dx.hypot(dz).max(1e-3);
        let keep = (MINE_REACH_M - 0.25).min(d);
        let stand = (node.0 - dx / d * keep, node.1 - dz / d * keep);
        Rider { x, z, yaw, phase: RiderPhase::Dismount, t: 0.0, seat, stand, node, failed: false, via: None, held_s: 0.0 }
    }

    /// On foot, away from the seat.
    pub fn off(&self) -> bool {
        matches!(self.phase, RiderPhase::WalkOut | RiderPhase::Mine | RiderPhase::WalkBack)
    }

    /// Walks toward `to` for `dt`; true once there.
    pub fn walk_toward(&mut self, to: (f32, f32), dt: f32) -> bool {
        let (dx, dz) = (to.0 - self.x, to.1 - self.z);
        let d = dx.hypot(dz);
        if d < 1e-3 {
            return true;
        }
        let step = (RIDER_WALK_MPS * dt).min(d);
        self.yaw = dz.atan2(dx);
        self.x += dx / d * step;
        self.z += dz / d * step;
        step >= d - 1e-4
    }

    /// The rider's body as solids in the world, standing (on foot) at its place: the same person as on the seat.
    pub fn boxes(&self, ground_m: f32, body: &noto_core::balance::Body) -> Vec<crate::render::Obb> {
        use noto_core::kin::{self, KinLoads, KinPose};
        let on_foot = rider_body(body);
        let links = kin::solve(&on_foot, &KinPose::stand(&on_foot, 0.0), &KinLoads::default()).boxes;
        let parts = noto_core::rig::dress(&on_foot, &links, &noto_core::rig::Palette { accent: [236, 190, 40], ..Default::default() });
        let (s, c) = self.yaw.sin_cos();
        let at = |p: [f32; 3]| [self.x + c * p[0] + s * p[1], ground_m + p[2], self.z + s * p[0] - c * p[1]];
        let dir = |v: [f32; 3]| [c * v[0] + s * v[1], v[2], s * v[0] - c * v[1]];
        parts.iter().map(|p| crate::render::Obb::placed(p, at, dir)).collect()
    }
}

/// A robot as the others see it: its outline on the ground. A lift truck is long and narrow, and a disc round it
/// would cover the bay it waits at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobotOutline {
    pub id: u32,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub front: f32,
    pub rear: f32,
    pub half_w: f32,
    pub moving: bool,
    pub humanoid: bool,
}

impl RobotOutline {
    /// A round outline, for someone on foot.
    fn round(id: u32, x: f32, z: f32, r: f32, moving: bool) -> RobotOutline {
        RobotOutline { id, x, z, yaw: 0.0, front: r, rear: r, half_w: r, moving, humanoid: true }
    }

    fn of(id: u32, x: f32, z: f32, yaw: f32, fp: crate::clear::Footprint, moving: bool, humanoid: bool) -> RobotOutline {
        RobotOutline { id, x, z, yaw, front: fp.front_m, rear: fp.rear_m, half_w: fp.half_w_m, moving, humanoid }
    }

    /// Whether a point lies within `pad` of the outline.
    pub fn near(&self, p: (f32, f32), pad: f32) -> bool {
        let (s, c) = self.yaw.sin_cos();
        let (dx, dz) = (p.0 - self.x, p.1 - self.z);
        let (along, across) = (dx * c + dz * s, dz * c - dx * s);
        along <= self.front + pad && along >= -self.rear - pad && across.abs() <= self.half_w + pad
    }

    /// The farthest the outline reaches from the robot's centre.
    pub fn reach_m(&self) -> f32 {
        self.front.max(self.rear).hypot(self.half_w)
    }
}

/// The robot looking: where it is, which way it goes and how fast, how wide its way is (half its width) and how far
/// its body reaches ahead along it, and its eye height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Looker {
    pub id: u32,
    pub at: (f32, f32),
    pub dir: f32,
    pub half_w: f32,
    pub front: f32,
    pub speed: f32,
    pub eye_m: f32,
}

/// Whether a robot that has stopped for `other` routes round it (the other waits): the lower rank does, and of two
/// of one rank the higher id — so of two meeting head on, exactly one goes round.
pub fn gives_way(me: u32, me_humanoid: bool, other: u32, other_humanoid: bool) -> bool {
    let (a, b) = (rank(me, me_humanoid), rank(other, other_humanoid));
    a > b || (a == b && me > other)
}

/// Which of two robots gives way: the lower rank waits first (the first worker 0, vehicles 1, humanoids 2).
pub fn rank(id: u32, humanoid: bool) -> u8 {
    match id {
        1 | RIDER_ID => 0,
        _ if humanoid => 2,
        _ => 1,
    }
}

impl World {
    /// The radius the first worker keeps others at: its outline's half width and half its length.
    pub fn agent_radius_m(&self) -> f32 {
        let fp = self.footprint();
        fp.half_w_m.max(0.5 * (fp.front_m + fp.rear_m)).max(0.3)
    }

    /// Every robot on the map: the first worker (and its rider while on foot) and the sub-workers.
    pub fn robot_outlines(&self) -> Vec<RobotOutline> {
        // On the move: going somewhere, even while stopped for another robot (so that two which stopped for each
        // other still tell who gives way).
        let agent_moving = matches!(self.active, Some((_, Active::Move(_))));
        let mut v = vec![RobotOutline::of(1, self.agent.x, self.agent.z, self.agent.yaw, self.footprint(), agent_moving, self.body.legged())];
        if let Some(r) = self.rider.filter(|r| r.off()) {
            v.push(RobotOutline::round(RIDER_ID, r.x, r.z, RIDER_R_M, !matches!(r.phase, RiderPhase::Mine)));
        }
        for w in &self.crew {
            v.push(RobotOutline::of(w.id, w.x, w.z, w.yaw, w.footprint, matches!(w.job, crate::crew::Job::Go { .. }), w.form == crate::crew::Form::Humanoid));
            if let Some(r) = w.rider.filter(|r| r.off()) {
                v.push(RobotOutline::round(RIDER_ID + w.id, r.x, r.z, RIDER_R_M, !matches!(r.phase, RiderPhase::Mine)));
            }
        }
        v
    }

    /// Whether the straight line from `a` to `b` (both at `h` above the ground) passes clear of every node.
    pub fn line_of_sight(&self, a: (f32, f32), b: (f32, f32), h: f32) -> bool {
        let (dx, dz) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dz);
        if len < 1e-3 {
            return true;
        }
        let base = self.surface_m(a.0, a.1) + h;
        self.nodes.iter().all(|n| {
            let r = if n.kind == crate::kinds::NodeKind::Tree { 0.35 } else { 0.55 };
            let t = (((n.x - a.0) * dx + (n.z - a.1) * dz) / (len * len)).clamp(0.0, 1.0);
            let (px, pz) = (a.0 + dx * t, a.1 + dz * t);
            // A node at either end is what is looked at, not what hides it.
            let at_end = t <= 1e-3 || t >= 1.0 - 1e-3;
            at_end || (n.x - px).hypot(n.z - pz) > r || self.surface_m(n.x, n.z) + 1.0 < base - 0.6
        })
    }

    /// The nearest robot the eye of `me` sees in its way: in the camera's horizontal field of view about the way it
    /// goes (or close enough to touch), not hidden behind a node, and standing on the stretch of its path (`path`,
    /// the points it goes through next) it covers before it could stop — its body's reach ahead and the stopping
    /// distance. (its id, the path length to it in m). A robot the path passes clear of is passed, not waited for.
    pub fn robot_in_way(&self, me: &Looker, path: &[(f32, f32)]) -> Option<(u32, f32)> {
        let look = me.front + KEEP_CLEAR_M + me.speed.max(0.3) * 1.0;
        let half = SIGHT_HALF_FOV_DEG.to_radians();
        let own = |o: u32| o == me.id || (me.id == 1 && o == RIDER_ID) || (me.id == RIDER_ID && o == 1) || (me.id > 1 && me.id < RIDER_ID && o == RIDER_ID + me.id) || (me.id > RIDER_ID && o == me.id - RIDER_ID);
        // The path ahead every 0.15 m, with the distance along it.
        let mut samples: Vec<(f32, f32, f32)> = Vec::new();
        let (mut prev, mut run) = (me.at, 0.0f32);
        for &p in path.iter().chain(std::iter::once(&(me.at.0 + me.dir.cos() * 1e-3, me.at.1 + me.dir.sin() * 1e-3)).filter(|_| path.is_empty())) {
            let seg = (p.0 - prev.0).hypot(p.1 - prev.1);
            let n = (seg / 0.15).ceil().max(1.0) as usize;
            for k in 1..=n {
                let u = k as f32 / n as f32;
                let s = run + seg * u;
                if s > look {
                    break;
                }
                samples.push((prev.0 + (p.0 - prev.0) * u, prev.1 + (p.1 - prev.1) * u, s));
            }
            run += seg;
            prev = p;
            if run > look {
                break;
            }
        }
        if path.is_empty() || samples.is_empty() {
            // Standing, or no path: only a robot pressing on it.
            samples = vec![(me.at.0, me.at.1, 0.0)];
        }
        self.robot_outlines()
            .into_iter()
            .filter(|o| !own(o.id))
            .filter_map(|o| {
                let (dx, dz) = (o.x - me.at.0, o.z - me.at.1);
                let d = dx.hypot(dz);
                let bearing = wrap_angle(dz.atan2(dx) - me.dir).abs();
                let touching = o.near(me.at, me.half_w + 0.05);
                let seen = (bearing <= half && d <= SIGHT_RANGE_M) || touching;
                // Bodies that would overlap: the path's points within both half widths of the robot's centre.
                let hit = samples.iter().find(|s| o.near((s.0, s.1), me.half_w - 0.05)).map(|s| s.2);
                let hit = hit.or(touching.then_some(0.0).filter(|_| bearing <= half));
                hit.filter(|_| seen && self.line_of_sight(me.at, (o.x, o.z), me.eye_m.min(1.2))).map(|s| (o.id, s))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The cells robots standing still occupy, widened by `pad` (the half width of the robot routing round them and a
    /// margin), for a route to go round; all but `except`, and never the cell `keep` (where the router stands).
    pub fn still_robot_cells(&self, except: &[u32], keep: Option<(usize, usize)>, pad: f32) -> HashSet<(usize, usize)> {
        self.robot_cells(except, keep, pad, true)
    }

    /// `still_robot_cells` with moving robots too when `still_only` is false, widened by `pad`.
    pub fn robot_cells(&self, except: &[u32], keep: Option<(usize, usize)>, pad: f32, still_only: bool) -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        for o in self.robot_outlines().into_iter().filter(|o| !except.contains(&o.id) && (!still_only || !o.moving)) {
            let (cx, cz) = match self.terrain.cell_of(o.x, o.z) {
                Some(c) => c,
                None => continue,
            };
            let span = ((o.reach_m() + pad) / CELL_M).ceil() as i32 + 1;
            for dz in -span..=span {
                for dx in -span..=span {
                    let (x, z) = (cx as i32 + dx, cz as i32 + dz);
                    if x < 0 || z < 0 || x as usize >= self.terrain.w || z as usize >= self.terrain.d {
                        continue;
                    }
                    let c = (x as usize, z as usize);
                    let (px, pz) = crate::terrain::cell_center(c.0, c.1);
                    if o.near((px, pz), pad) && Some(c) != keep {
                        out.insert(c);
                    }
                }
            }
        }
        out
    }

    /// The first worker's way ahead: its heading, or backwards when it reverses.
    pub(crate) fn travel_dir(&self) -> f32 {
        let reversing = self.motion_view().is_some_and(|m| m.reversing);
        if reversing {
            wrap_angle(self.agent.yaw + std::f32::consts::PI)
        } else {
            self.agent.yaw
        }
    }

    /// A rider walking out to the node or back to the seat for `dt`, stopping for a robot it sees in its way and
    /// stepping round one that stays there. True once it has arrived.
    pub(crate) fn rider_walk(&self, id: u32, r: &mut Rider, dt: f32) -> bool {
        let goal = if r.phase == RiderPhase::WalkOut { r.stand } else { r.seat };
        let to = r.via.unwrap_or(goal);
        let me = Looker { id, at: (r.x, r.z), dir: (to.1 - r.z).atan2(to.0 - r.x), half_w: RIDER_R_M, front: RIDER_R_M, speed: RIDER_WALK_MPS, eye_m: rider_eye_m(&self.body) };
        let path: Vec<(f32, f32)> = r.via.into_iter().chain(std::iter::once(goal)).collect();
        if let Some((other, _)) = self.robot_in_way(&me, &path) {
            r.held_s += dt;
            if r.held_s >= YIELD_HUMANOID_S {
                // Step round it on the side nearer the goal.
                if let Some(o) = self.robot_outlines().into_iter().find(|o| o.id == other) {
                    let (dx, dz) = (goal.0 - r.x, goal.1 - r.z);
                    let n = dx.hypot(dz).max(1e-3);
                    let (px, pz) = (-dz / n, dx / n);
                    let side = if (o.x - r.x) * px + (o.z - r.z) * pz > 0.0 { -1.0 } else { 1.0 };
                    let off = o.reach_m() + RIDER_R_M + 0.3;
                    r.via = Some((o.x + px * off * side, o.z + pz * off * side));
                }
                r.held_s = 0.0;
            }
            return false;
        }
        r.held_s = 0.0;
        if r.walk_toward(to, dt) {
            if r.via.take().is_some() {
                return false;
            }
            return true;
        }
        false
    }

    /// The first worker's rider steps down, walks to the node, fells it, walks back and climbs on: one tick of it.
    pub(super) fn advance_rider(&mut self, node: u32, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        let Some(mut r) = self.rider else { return 0.0 };
        let t = remaining;
        self.agent.speed_mps = 0.0;
        match r.phase {
            RiderPhase::Dismount => {
                r.t += t;
                self.add_energy(MANIPULATION_POWER_W * t as f64);
                if r.t >= DISMOUNT_S {
                    (r.phase, r.t, r.x, r.z) = (RiderPhase::WalkOut, 0.0, r.seat.0, r.seat.1);
                    events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "dismount".into(), label: format!("탑승자 하차: 지게차를 노드에서 {:.1} m에 세우고 내림", (self.agent.x - r.node.0).hypot(self.agent.z - r.node.1)) }));
                }
            }
            RiderPhase::WalkOut | RiderPhase::WalkBack => {
                if self.rider_walk(RIDER_ID, &mut r, t) {
                    if r.phase == RiderPhase::WalkOut {
                        r.phase = RiderPhase::Mine;
                        events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "walk".into(), label: "탑승자가 노드 앞까지 걸어감".into() }));
                    } else {
                        (r.phase, r.t) = (RiderPhase::Remount, 0.0);
                    }
                }
                self.add_energy(MANIPULATION_POWER_W * t as f64);
            }
            RiderPhase::Mine => match self.node(node).map(|n| (n.x, n.z)) {
                None => {
                    (r.phase, r.failed) = (RiderPhase::WalkBack, true);
                }
                Some((nx, nz)) => {
                    r.yaw = (nz - r.z).atan2(nx - r.x);
                    let rate = self.mining_rate();
                    self.add_energy(MANIPULATION_POWER_W * t as f64);
                    self.agent.strike_acc += t;
                    while self.agent.strike_acc >= 1.0 {
                        self.agent.strike_acc -= 1.0;
                        self.agent.strikes += 1;
                    }
                    let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                    n.hp -= (t * rate).min(n.hp);
                    if n.hp <= 1e-5 {
                        self.break_node(node, events);
                        r.phase = RiderPhase::WalkBack;
                        events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "mine".into(), label: "탑승자가 걸어서 채취를 마침 — 지게차로 돌아감".into() }));
                    }
                }
            },
            RiderPhase::Remount => {
                r.t += t;
                self.add_energy(MANIPULATION_POWER_W * t as f64);
                if r.t >= REMOUNT_S {
                    self.rider = None;
                    events.push(self.stamp(WorldEvent::Rider { worker: 1, phase: "remount".into(), label: "탑승자 재탑승 — 포크 작업으로".into() }));
                    *done = Some(if r.failed { (false, Some("unknown_target".into())) } else { (true, None) });
                    return t;
                }
            }
        }
        self.rider = Some(r);
        t
    }
}
