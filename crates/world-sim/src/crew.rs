//! Sub-workers (서브 작업자): robots added around the first worker (the world's `agent`, worker 1) that share its world —
//! the trees they fell, the drops they pick up, the loading bay they stage at, the waypoints they stock. Each runs a
//! lighter simulation than the first worker: it plans on the same terrain and clearance map with its own base
//! (`BaseKind::spec`, its body's footprint), moves along that plan at the plan's own edge times, swings at a node at
//! the first worker's mining rate per hand, and picks up, stages and delivers in fixed times — no joint dynamics, no
//! camera frames of its own. The runtime's crew dispatcher (`noto-runtime::crew`) hands out the jobs; the heavy vision
//! pipeline runs for one robot at a time, so a crew costs about one robot's vision.
//!
//! A loading bay (상차장) is a waypoint whose deliveries become drops on the ground in a row of slots in front of it:
//! where humanoids put down what they harvested and a lift truck's tines take it from.

use noto_core::balance::Body;
use noto_core::{BaseKind, BaseSpec, BodyPlan, FingerCount};
use serde::{Deserialize, Serialize};

use crate::clear::{self, Footprint};
use crate::kinds::ItemKind;
use crate::nav::{self, Goal};
use crate::render::Obb;
use crate::snapshot::{drop_id, node_id, waypoint_id, CarryView};
use crate::terrain::cell_center;
use crate::world::traffic::{self, Rider, RiderPhase, RIDER_REACH_M};
use crate::world::{add_stock, DropItem, TimedEvent, Waypoint, World, WorldEvent, AMOUNT_EPS, DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};

/// At most this many sub-workers (with the first worker, nine robots on one map).
pub const CREW_MAX: usize = 8;
/// How many times a sub-worker routes round robots in its way on one move before the move fails.
pub const MAX_REROUTES: u32 = 6;
/// Seconds to pick an item up by hand, to put it down at a bay slot, to hand it over or deliver it.
pub const HAND_TIME_S: f32 = 1.0;
/// Seconds for a lift truck to put its tines under a staged item and lift it [추정: the first worker's fork act runs
/// 2–4 s from lowering to raising].
pub const FORK_TIME_S: f32 = 2.5;
/// Seconds for a lift truck to set its load down at a waypoint.
pub const UNLOAD_TIME_S: f32 = 2.0;
/// Power while working in place (W), as the first worker's `MANIPULATION_POWER_W`.
pub const WORK_POWER_W: f64 = 30.0;
/// The slots a bay keeps its staged items in: a row across its front, this far apart (m) [추정: a log is 0.15 m
/// across; a hand's width between them leaves room for the tines].
pub const BAY_SLOT_M: f32 = 0.35;
pub const BAY_SLOTS: usize = 6;
/// How far in front of the bay the row of slots lies (m).
pub const BAY_ROW_AHEAD_M: f32 = 0.6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Form {
    /// Legs and two hands: fells, picks up, carries, stages.
    Humanoid,
    /// A lift truck (wheels, mast, tines) with a rider: hauls what is staged, and can fell and pick up as the first
    /// worker's wheeled bodies do.
    Forklift,
}

impl Form {
    pub fn base(self) -> BaseKind {
        match self {
            Form::Humanoid => BaseKind::HumanFoot,
            Form::Forklift => BaseKind::Wheel,
        }
    }

    pub fn of_base(b: BaseKind) -> Form {
        match b {
            BaseKind::HumanFoot => Form::Humanoid,
            _ => Form::Forklift,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Form::Humanoid => "humanoid",
            Form::Forklift => "forklift",
        }
    }

    pub fn from_id(s: &str) -> Option<Form> {
        match s {
            "humanoid" | "human" | "휴머노이드" => Some(Form::Humanoid),
            "forklift" | "lift_truck" | "지게차" => Some(Form::Forklift),
            _ => None,
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Form::Humanoid => "휴머노이드",
            Form::Forklift => "지게차",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Job {
    Idle,
    /// Walk or drive until within `reach_m` of (x, z).
    Go { x: f32, z: f32, reach_m: f32 },
    /// Swing at a node until it breaks.
    Mine { node: u32 },
    /// Pick a drop up (a lift truck: with its tines).
    Pick { drop: u32 },
    /// Put everything carried down in the bay's free slots.
    Stage { bay: u32 },
    /// Put everything carried into a waypoint's stock.
    Deliver { waypoint: u32 },
    /// Stand facing (x, z) until given another job.
    Park { x: f32, z: f32 },
}

impl Job {
    pub fn label_ko(&self) -> String {
        match self {
            Job::Idle => "대기".into(),
            Job::Go { .. } => "이동".into(),
            Job::Mine { node } => format!("{} 채취", node_id(*node)),
            Job::Pick { drop } => format!("{} 줍기", drop_id(*drop)),
            Job::Stage { .. } => "상차장에 내려놓기".into(),
            Job::Deliver { waypoint } => format!("{} 인도", waypoint_id(*waypoint)),
            Job::Park { .. } => "정차 대기".into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Leg {
    x: f32,
    z: f32,
    s: f32,
}

#[derive(Clone, Debug)]
pub struct Worker {
    pub id: u32,
    pub form: Form,
    /// Cameras on its eye: 1 (monocular, TIPSv2 range) or 2 (stereo pair, SGBM range).
    pub cameras: u8,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub speed_mps: f32,
    pub carried: Vec<(ItemKind, f32)>,
    pub energy_j: f64,
    pub capacity_kg: f32,
    pub job: Job,
    /// The last job's outcome (ok, what happened), for the dispatcher to take.
    pub outcome: Option<(bool, String)>,
    /// How much its last delivery or staging put down.
    pub last_moved: f32,
    /// A lift truck's rider on foot, felling a node (`traffic::Rider`).
    pub rider: Option<Rider>,
    /// Stopped for a robot its eye sees in its way: for how long, for which; and how many times it routed round.
    pub held_s: f32,
    pub held_by: Option<u32>,
    pub reroutes: u32,
    felled: Option<String>,
    spec: BaseSpec,
    footprint: Footprint,
    route: Vec<Leg>,
    leg: usize,
    leg_t: f32,
    from: (f32, f32),
    work_left_s: f32,
}

impl Worker {
    fn new(id: u32, form: Form, cameras: u8, x: f32, z: f32, yaw: f32) -> Worker {
        let body = Body::new(&BodyPlan::default(), form.base(), FingerCount::Five);
        let spec = form.base().spec();
        let capacity_kg = body.drive_limits(&spec, 0.0).capacity_kg;
        Worker {
            id,
            form,
            cameras: cameras.clamp(1, 2),
            x,
            z,
            yaw,
            speed_mps: 0.0,
            carried: Vec::new(),
            energy_j: 0.0,
            capacity_kg,
            job: Job::Idle,
            outcome: None,
            last_moved: 0.0,
            rider: None,
            held_s: 0.0,
            held_by: None,
            reroutes: 0,
            felled: None,
            spec,
            footprint: Footprint::of(&body),
            route: Vec::new(),
            leg: 0,
            leg_t: 0.0,
            from: (x, z),
            work_left_s: 0.0,
        }
    }

    pub fn carried_kg(&self) -> f32 {
        self.carried.iter().map(|(k, n)| k.spec().mass_kg * n).sum()
    }

    pub fn carried_count(&self) -> f32 {
        self.carried.iter().map(|(_, n)| n).sum()
    }

    /// Whether it can take one more `item` on its hands or tines.
    pub fn room_for(&self, item: ItemKind) -> bool {
        self.carried_kg() + item.spec().mass_kg <= self.capacity_kg + 1e-3
    }

    pub fn busy(&self) -> bool {
        !matches!(self.job, Job::Idle | Job::Park { .. })
    }

    /// The radius it keeps others at: its outline's half width and half its length.
    pub fn radius_m(&self) -> f32 {
        self.footprint.half_w_m.max(0.5 * (self.footprint.front_m + self.footprint.rear_m)).max(0.3)
    }

    /// Its eye height (a humanoid's head, a lift truck rider's seat) [추정].
    pub fn eye_m(&self) -> f32 {
        match self.form {
            Form::Humanoid => 1.5,
            Form::Forklift => 1.8,
        }
    }

    /// Where a lift truck's tines take a load from, or where a humanoid holds it: a little ahead of its body.
    pub fn front(&self) -> (f32, f32) {
        let ahead = self.footprint.front_m.max(0.3);
        (self.x + ahead * self.yaw.cos(), self.z + ahead * self.yaw.sin())
    }

    pub fn carry_views(&self) -> Vec<CarryView> {
        World::carry_views(&self.carried)
    }

    pub fn status_ko(&self) -> String {
        if let Some(r) = self.rider {
            return format!("탑승자 {} (채취)", r.phase.label_ko());
        }
        let load = if self.carried.is_empty() {
            String::new()
        } else {
            format!(" · 적재 {:.1} kg", self.carried_kg())
        };
        format!("{}{load}", self.job.label_ko())
    }
}

fn wrap(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// A distinct colour per sub-worker (the first worker keeps the robot's own).
fn colour(id: u32, form: Form) -> [u8; 3] {
    const HUMANOID: [[u8; 3]; 4] = [[236, 128, 52], [52, 170, 190], [168, 96, 210], [96, 186, 82]];
    const FORKLIFT: [[u8; 3]; 2] = [[232, 192, 40], [214, 150, 32]];
    match form {
        Form::Humanoid => HUMANOID[id as usize % HUMANOID.len()],
        Form::Forklift => FORKLIFT[id as usize % FORKLIFT.len()],
    }
}

impl World {
    /// Adds sub-workers of the given forms around the first worker, each on a free cell its base can reach from
    /// the first worker's. Returns their ids (2, 3, …).
    pub fn add_crew(&mut self, forms: &[Form], cameras: u8) -> Vec<u32> {
        let mut ids = Vec::new();
        let blocked = self.blocked_cells();
        let start = self.terrain.cell_of(self.agent.x, self.agent.z).unwrap_or((0, 0));
        for &form in forms {
            if self.crew.len() >= CREW_MAX {
                break;
            }
            let id = self.crew.iter().map(|w| w.id).max().unwrap_or(1) + 1;
            let spec = form.base().spec();
            let taken: Vec<(f32, f32)> = self.crew.iter().map(|w| (w.x, w.z)).chain(std::iter::once((self.agent.x, self.agent.z))).collect();
            let spot = [1.6f32, 2.2, 2.9, 3.6].into_iter().find_map(|r| {
                (0..12).find_map(|k| {
                    // Behind and beside the first worker first, so the way ahead stays open.
                    let a = self.agent.yaw + std::f32::consts::PI + (k as f32 * 0.5 + 0.25) * if k % 2 == 0 { 1.0 } else { -1.0 } * 0.5;
                    let (x, z) = (self.agent.x + r * a.cos(), self.agent.z + r * a.sin());
                    let c = self.terrain.cell_of(x, z)?;
                    if blocked.contains(&c) || taken.iter().any(|(tx, tz)| (tx - x).hypot(tz - z) < 1.0) {
                        return None;
                    }
                    nav::plan(&spec, &self.terrain, &|cx, cz| blocked.contains(&(cx, cz)), start, Goal::Cell(c))?;
                    let (cx, cz) = cell_center(c.0, c.1);
                    Some((cx, cz))
                })
            });
            let Some((x, z)) = spot else { continue };
            let yaw = self.agent.yaw;
            self.crew.push(Worker::new(id, form, cameras, x, z, yaw));
            ids.push(id);
        }
        ids
    }

    /// Takes every sub-worker off the map; what they carried goes down where they stood.
    pub fn remove_crew(&mut self) -> usize {
        let crew = std::mem::take(&mut self.crew);
        for w in &crew {
            for (k, n) in &w.carried {
                if *n > AMOUNT_EPS {
                    let d = DropItem { amount: *n, ..DropItem::whole(self.next_drop_id(), *k, w.x, w.z) };
                    self.drops.push(d);
                }
            }
        }
        crew.len()
    }

    pub fn crew_worker(&self, id: u32) -> Option<&Worker> {
        self.crew.iter().find(|w| w.id == id)
    }

    pub fn crew_worker_mut(&mut self, id: u32) -> Option<&mut Worker> {
        self.crew.iter_mut().find(|w| w.id == id)
    }

    /// Gives a sub-worker a job; a move plans its route now (an error when there is none).
    pub fn crew_command(&mut self, id: u32, job: Job) -> Result<(), String> {
        let route = match &job {
            Job::Go { x, z, reach_m } => {
                let w = self.crew_worker(id).ok_or_else(|| format!("작업자 {id}가 없습니다"))?;
                if (w.x - x).hypot(w.z - z) <= *reach_m {
                    Some(Vec::new())
                } else {
                    Some(self.crew_route(w, *x, *z, *reach_m).ok_or_else(|| format!("작업자 {id}: ({x:.1}, {z:.1})로 가는 길이 없습니다"))?)
                }
            }
            _ => None,
        };
        let w = self.crew_worker(id).ok_or_else(|| format!("작업자 {id}가 없습니다"))?;
        let (wx, wz) = (w.x, w.z);
        let check = match &job {
            Job::Mine { node } => self.node(*node).map(|n| (n.x - wx).hypot(n.z - wz) <= if w.form == Form::Forklift { RIDER_REACH_M } else { MINE_REACH_M + 0.35 }).ok_or("unknown_target"),
            Job::Pick { drop } => self.drop_item(*drop).map(|d| (d.x - wx).hypot(d.z - wz) <= PICKUP_REACH_M + if w.form == Form::Forklift { 1.2 } else { 0.35 }).ok_or("unknown_target"),
            Job::Deliver { waypoint } | Job::Stage { bay: waypoint } => self.waypoint(*waypoint).map(|p| (p.x - wx).hypot(p.z - wz) <= DELIVER_REACH_M + 0.6).ok_or("unknown_target"),
            _ => Ok(true),
        };
        match check {
            Err(code) => return Err(format!("작업자 {id}: 대상이 없습니다 ({code})")),
            Ok(false) => return Err(format!("작업자 {id}: 대상에 손이 닿지 않습니다")),
            Ok(true) => {}
        }
        let work = match &job {
            Job::Pick { .. } if w.form == Form::Forklift => FORK_TIME_S,
            Job::Pick { .. } | Job::Stage { .. } => HAND_TIME_S,
            Job::Deliver { .. } if w.form == Form::Forklift => UNLOAD_TIME_S,
            Job::Deliver { .. } => HAND_TIME_S,
            _ => 0.0,
        };
        let node_at = match &job {
            Job::Mine { node } => self.node(*node).map(|n| (n.x, n.z)),
            _ => None,
        };
        let w = self.crew_worker_mut(id).expect("looked up above");
        w.route = route.unwrap_or_default();
        w.leg = 0;
        w.leg_t = 0.0;
        w.from = (w.x, w.z);
        w.work_left_s = work;
        w.outcome = None;
        w.held_s = 0.0;
        w.held_by = None;
        w.reroutes = 0;
        // A lift truck's rider steps down and fells it on foot.
        w.rider = match (w.form, node_at) {
            (Form::Forklift, Some(at)) => Some(Rider::start(w.x, w.z, w.yaw, at)),
            _ => None,
        };
        w.felled = None;
        w.job = job;
        Ok(())
    }

    /// The outcome of a sub-worker's last job, once (ok, what happened).
    pub fn crew_take_outcome(&mut self, id: u32) -> Option<(bool, String)> {
        self.crew_worker_mut(id)?.outcome.take()
    }

    fn crew_route(&self, w: &Worker, x: f32, z: f32, reach_m: f32) -> Option<Vec<Leg>> {
        let start = self.terrain.cell_of(w.x, w.z)?;
        let robots = self.still_robot_cells(&[w.id, traffic::RIDER_ID + w.id], Some(start), w.footprint.half_w_m + 0.15);
        self.crew_route_avoiding(w, x, z, reach_m, &robots).or_else(|| self.crew_route_avoiding(w, x, z, reach_m, &std::collections::HashSet::new()))
    }

    fn crew_route_avoiding(&self, w: &Worker, x: f32, z: f32, reach_m: f32, extra: &std::collections::HashSet<(usize, usize)>) -> Option<Vec<Leg>> {
        let mut blocked = self.blocked_cells();
        blocked.extend(extra.iter().copied());
        let map = self.clear_map();
        let start = self.terrain.cell_of(w.x, w.z)?;
        let plan = [clear::MIN_GAP_M, clear::TIGHT_GAP_M].into_iter().find_map(|gap| {
            let room = nav::Room { map: &map, need_m: w.footprint.half_w_m + gap, soft_m: w.footprint.half_w_m + clear::COMFORT_GAP_M };
            nav::plan_room(&w.spec, &self.terrain, &|cx, cz| blocked.contains(&(cx, cz)), Some(&room), start, Goal::Within { x, z, reach_m }, None)
        })?;
        let mut legs: Vec<Leg> = plan
            .edges
            .iter()
            .map(|e| {
                let (cx, cz) = cell_center(e.to.0, e.to.1);
                Leg { x: cx, z: cz, s: (e.move_time_s + e.pause_s).max(0.05) }
            })
            .collect();
        // The last leg ends at the reach circle around the goal rather than at the cell's centre.
        if let Some(last) = legs.last_mut() {
            let (dx, dz) = (last.x - x, last.z - z);
            let d = dx.hypot(dz);
            if d > reach_m && d > 1e-3 {
                let k = (d - reach_m * 0.9) / d;
                let (px, pz) = (x + dx * (1.0 - k), z + dz * (1.0 - k));
                last.x = px;
                last.z = pz;
            }
        }
        Some(legs)
    }

    /// Advances every sub-worker by `dt`.
    pub(crate) fn step_crew(&mut self, dt: f32, events: &mut Vec<TimedEvent>) {
        for i in 0..self.crew.len() {
            self.step_worker(i, dt, events);
        }
    }

    fn crew_event(&self, id: u32, kind: &str, label: String, item: Option<ItemKind>, amount: f32, ok: bool) -> TimedEvent {
        self.stamp(WorldEvent::Crew { worker: id, kind: kind.into(), label, item, amount, ok })
    }

    fn step_worker(&mut self, i: usize, dt: f32, events: &mut Vec<TimedEvent>) {
        let job = self.crew[i].job.clone();
        let id = self.crew[i].id;
        let finish = |w: &mut Worker, ok: bool, what: String| {
            w.outcome = Some((ok, what));
            w.job = Job::Idle;
            w.speed_mps = 0.0;
            w.route.clear();
        };
        match job {
            Job::Idle => self.crew[i].speed_mps = 0.0,
            Job::Park { x, z } => {
                let w = &mut self.crew[i];
                w.speed_mps = 0.0;
                let want = (z - w.z).atan2(x - w.x);
                let turn = wrap(want - w.yaw).clamp(-1.2 * dt, 1.2 * dt);
                w.yaw = wrap(w.yaw + turn);
            }
            Job::Go { x, z, reach_m } => {
                if self.crew_hold(i, dt, (x, z, reach_m), events) {
                    return;
                }
                let w = &mut self.crew[i];
                let mut left = dt;
                while left > 1e-6 && w.leg < w.route.len() {
                    let leg = w.route[w.leg];
                    let step = (leg.s - w.leg_t).min(left);
                    w.leg_t += step;
                    left -= step;
                    let k = (w.leg_t / leg.s).clamp(0.0, 1.0);
                    let (nx, nz) = (w.from.0 + (leg.x - w.from.0) * k, w.from.1 + (leg.z - w.from.1) * k);
                    let moved = (nx - w.x).hypot(nz - w.z);
                    if moved > 1e-4 {
                        w.yaw = (nz - w.z).atan2(nx - w.x);
                    }
                    w.x = nx;
                    w.z = nz;
                    w.energy_j += w.spec.power_w as f64 * step as f64;
                    w.speed_mps = moved / step.max(1e-3);
                    if w.leg_t >= leg.s - 1e-6 {
                        w.leg += 1;
                        w.leg_t = 0.0;
                        w.from = (leg.x, leg.z);
                    }
                }
                if w.leg >= w.route.len() {
                    let want = (z - w.z).atan2(x - w.x);
                    if (x - w.x).hypot(z - w.z) > 0.05 {
                        w.yaw = want;
                    }
                    finish(w, true, "도착".into());
                }
            }
            Job::Mine { node } if self.crew[i].rider.is_some() => self.crew_rider_step(i, node, dt, events),
            Job::Mine { node } => {
                let Some(n) = self.node(node) else {
                    finish(&mut self.crew[i], false, format!("{} 없음 (이미 쓰러짐)", node_id(node)));
                    return;
                };
                let (nx, nz) = (n.x, n.z);
                let w = &mut self.crew[i];
                w.yaw = (nz - w.z).atan2(nx - w.x);
                w.energy_j += WORK_POWER_W * dt as f64;
                // The first worker's rate for five fingers, a hand per robot.
                let rate = 1.0f32;
                let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                n.hp -= (rate * dt).min(n.hp);
                if n.hp <= 1e-5 {
                    let (kind, drops) = self.fell(node);
                    let label = format!("{} {} 쓰러뜨림 → {}", kind.spec().label_ko, node_id(node), drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                    events.push(self.crew_event(id, "node_broken", label, Some(kind.spec().drop), drops.len() as f32, true));
                    finish(&mut self.crew[i], true, drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                }
            }
            Job::Pick { drop } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let Some(pos) = self.drops.iter().position(|d| d.id == drop) else {
                    finish(&mut self.crew[i], false, format!("{} 없음", drop_id(drop)));
                    return;
                };
                if !self.crew[i].room_for(self.drops[pos].kind) {
                    finish(&mut self.crew[i], false, "적재 한계".into());
                    return;
                }
                let d = self.drops.remove(pos);
                let w = &mut self.crew[i];
                add_stock(&mut w.carried, d.kind, d.amount);
                let how = if w.form == Form::Forklift { "포크로 들어 올림" } else { "주움" };
                let label = format!("{} {} {how}", d.kind.spec().label_ko, drop_id(drop));
                finish(w, true, drop_id(drop));
                events.push(self.crew_event(id, "picked", label, Some(d.kind), d.amount, true));
            }
            Job::Stage { bay } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let load = std::mem::take(&mut self.crew[i].carried);
                self.crew[i].last_moved = load.iter().map(|(_, n)| n).sum();
                let mut ids = Vec::new();
                for (k, n) in load {
                    ids.extend(self.stage_at_bay(bay, k, n));
                }
                let label = format!("상차장에 {}개 내려놓음 ({})", ids.len(), ids.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                finish(&mut self.crew[i], true, ids.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                events.push(self.crew_event(id, "staged", label, None, ids.len() as f32, true));
            }
            Job::Deliver { waypoint } => {
                let w = &mut self.crew[i];
                w.work_left_s -= dt;
                w.energy_j += WORK_POWER_W * dt as f64;
                if w.work_left_s > 0.0 {
                    return;
                }
                let load = std::mem::take(&mut self.crew[i].carried);
                self.crew[i].last_moved = load.iter().map(|(_, n)| n).sum();
                let name = self.waypoint(waypoint).map(|p| p.label_ko.clone()).unwrap_or_default();
                for (k, n) in load.into_iter().filter(|(_, n)| *n > AMOUNT_EPS) {
                    if let Some(p) = self.waypoints.iter_mut().find(|p| p.id == waypoint) {
                        add_stock(&mut p.stock, k, n);
                    }
                    let label = format!("{name}에 {} {} 인도", k.spec().label_ko, crate::world::amount_text(n));
                    events.push(self.crew_event(id, "delivered", label, Some(k), n, true));
                }
                finish(&mut self.crew[i], true, name);
            }
        }
    }

    /// The way-ahead check of a sub-worker on the move: a robot its eye sees in its way stops it; one that stays has
    /// it route round (a humanoid sooner than a lift truck). True while it waits.
    fn crew_hold(&mut self, i: usize, dt: f32, goal: (f32, f32, f32), events: &mut Vec<TimedEvent>) -> bool {
        let w = &self.crew[i];
        let (id, form) = (w.id, w.form);
        let dir = match w.route.get(w.leg) {
            Some(l) if (l.x - w.x).hypot(l.z - w.z) > 1e-3 => (l.z - w.z).atan2(l.x - w.x),
            _ => w.yaw,
        };
        let me = traffic::Looker { id, at: (w.x, w.z), dir, half_w: w.footprint.half_w_m, front: w.footprint.front_m, speed: w.spec.max_speed_mps.min(1.0), eye_m: w.eye_m() };
        let path: Vec<(f32, f32)> = w.route[w.leg.min(w.route.len())..].iter().map(|l| (l.x, l.z)).collect();
        let Some((other, gap)) = self.robot_in_way(&me, &path) else {
            let w = &mut self.crew[i];
            w.held_s = 0.0;
            w.held_by = None;
            return false;
        };
        let w = &mut self.crew[i];
        w.speed_mps = 0.0;
        if w.held_by != Some(other) {
            w.held_by = Some(other);
            w.held_s = 0.0;
            let label = format!("시야에 작업자 {} {gap:.1} m 앞 — 멈춰 기다림", if other >= traffic::RIDER_ID { format!("{}의 탑승자", other - traffic::RIDER_ID) } else { other.to_string() });
            events.push(self.stamp(WorldEvent::Hold { worker: id, other, gap_m: gap, label }));
        }
        let blocker = self.robot_discs().into_iter().find(|o| o.id == other);
        let (other_humanoid, other_moving) = blocker.map_or((false, false), |o| (o.humanoid, o.moving));
        // An idle sub-worker in the way steps aside.
        if !other_moving && self.crew_make_way(other, dir) {
            events.push(self.stamp(WorldEvent::Hold { worker: other, other: id, gap_m: gap, label: format!("작업자 {id}의 길에서 비켜 섬") }));
        }
        let w = &mut self.crew[i];
        w.held_s += dt;
        let patience = if form == Form::Humanoid { traffic::YIELD_HUMANOID_S } else { traffic::YIELD_VEHICLE_S };
        // Of two on the move only the one that gives way routes round, the other waits for it to pass; a robot that
        // stands still is routed round by whoever it is in the way of.
        if w.held_s < patience || (other_moving && !traffic::gives_way(id, form == Form::Humanoid, other, other_humanoid)) {
            return true;
        }
        if w.reroutes >= MAX_REROUTES {
            // Blocked every way round: the move fails, and whoever gave it picks another target.
            w.outcome = Some((false, format!("작업자 {other}에 막혀 갈 길이 없음")));
            w.job = Job::Idle;
            w.route.clear();
            return true;
        }
        let snapshot = w.clone();
        let start = self.terrain.cell_of(snapshot.x, snapshot.z);
        let mut extra = self.still_robot_cells(&[id, traffic::RIDER_ID + id], start, snapshot.footprint.half_w_m + 0.15);
        let others: Vec<u32> = self.robot_discs().iter().map(|o| o.id).filter(|o| *o != other).collect();
        extra.extend(self.robot_cells(&others, start, snapshot.footprint.half_w_m + 0.15, false));
        let route = self.crew_route_avoiding(&snapshot, goal.0, goal.1, goal.2, &extra);
        let w = &mut self.crew[i];
        w.held_s = 0.0;
        if let Some(r) = route {
            (w.route, w.leg, w.leg_t, w.from) = (r, 0, 0.0, (w.x, w.z));
            w.reroutes += 1;
            events.push(self.stamp(WorldEvent::Hold { worker: id, other, gap_m: gap, label: format!("작업자 {other}이(가) 비키지 않아 돌아가는 길로 다시 계획") }));
        }
        true
    }

    /// Asks a sub-worker standing idle in another robot's way (that robot going along `dir`) to step aside, across
    /// that way; true when it goes.
    pub fn crew_make_way(&mut self, id: u32, dir: f32) -> bool {
        let Some(w) = self.crew_worker(id) else { return false };
        if !matches!(w.job, Job::Idle) {
            return false;
        }
        let (x, z) = (w.x, w.z);
        for d in [1.3f32, 2.0] {
            for side in [1.0f32, -1.0] {
                let a = dir + side * std::f32::consts::FRAC_PI_2;
                if self.crew_command(id, Job::Go { x: x + d * a.cos(), z: z + d * a.sin(), reach_m: 0.3 }).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// A sub-worker lift truck's rider fells a node on foot: one tick of stepping down, walking, swinging, walking
    /// back, climbing on.
    fn crew_rider_step(&mut self, i: usize, node: u32, dt: f32, events: &mut Vec<TimedEvent>) {
        let Some(mut r) = self.crew[i].rider else { return };
        let id = self.crew[i].id;
        self.crew[i].speed_mps = 0.0;
        self.crew[i].energy_j += WORK_POWER_W * dt as f64;
        match r.phase {
            RiderPhase::Dismount => {
                r.t += dt;
                if r.t >= traffic::DISMOUNT_S {
                    (r.phase, r.t, r.x, r.z) = (RiderPhase::WalkOut, 0.0, r.seat.0, r.seat.1);
                    events.push(self.stamp(WorldEvent::Rider { worker: id, phase: "dismount".into(), label: "탑승자 하차: 지게차를 세우고 내림".into() }));
                }
            }
            RiderPhase::WalkOut | RiderPhase::WalkBack => {
                if self.rider_walk(traffic::RIDER_ID + id, &mut r, dt) {
                    if r.phase == RiderPhase::WalkOut {
                        r.phase = RiderPhase::Mine;
                    } else {
                        (r.phase, r.t) = (RiderPhase::Remount, 0.0);
                    }
                }
            }
            RiderPhase::Mine => match self.node(node).map(|n| (n.x, n.z)) {
                None => {
                    (r.phase, r.failed) = (RiderPhase::WalkBack, true);
                }
                Some((nx, nz)) => {
                    r.yaw = (nz - r.z).atan2(nx - r.x);
                    let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                    n.hp -= dt.min(n.hp);
                    if n.hp <= 1e-5 {
                        let (kind, drops) = self.fell(node);
                        let label = format!("탑승자가 걸어가 {} {} 쓰러뜨림 → {}", kind.spec().label_ko, node_id(node), drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(", "));
                        events.push(self.crew_event(id, "node_broken", label, Some(kind.spec().drop), drops.len() as f32, true));
                        self.crew[i].felled = Some(drops.iter().map(|d| drop_id(*d)).collect::<Vec<_>>().join(","));
                        r.phase = RiderPhase::WalkBack;
                    }
                }
            },
            RiderPhase::Remount => {
                r.t += dt;
                if r.t >= traffic::REMOUNT_S {
                    events.push(self.stamp(WorldEvent::Rider { worker: id, phase: "remount".into(), label: "탑승자 재탑승".into() }));
                    let w = &mut self.crew[i];
                    w.rider = None;
                    let felled = w.felled.take();
                    w.outcome = Some(match (r.failed, felled) {
                        (false, Some(d)) => (true, d),
                        _ => (false, format!("{} 없음 (이미 쓰러짐)", node_id(node))),
                    });
                    w.job = Job::Idle;
                    return;
                }
            }
        }
        self.crew[i].rider = Some(r);
    }

    /// A loading bay: a waypoint whose deliveries become drops in the row of slots in front of it, facing `face`.
    pub fn add_bay(&mut self, x: f32, z: f32, face: (f32, f32), label_ko: &str) -> u32 {
        let id = self.waypoints.iter().map(|p| p.id).max().unwrap_or(0) + 1;
        let n = self.waypoints.iter().filter(|p| p.key.starts_with("bay")).count() + 1;
        self.waypoints.push(Waypoint { id, key: format!("bay{n}"), label_ko: label_ko.into(), x, z, stock: Vec::new() });
        self.bay_faces.push((id, (face.1 - z).atan2(face.0 - x)));
        id
    }

    pub fn is_bay(&self, waypoint: u32) -> bool {
        self.waypoint(waypoint).is_some_and(|p| p.key.starts_with("bay"))
    }

    /// The `k`-th slot of a bay's row.
    pub fn bay_slot(&self, bay: u32, k: usize) -> Option<(f32, f32)> {
        let p = self.waypoint(bay)?;
        let face = self.bay_faces.iter().find(|(id, _)| *id == bay).map_or(0.0, |(_, a)| *a);
        let (fx, fz) = (face.cos(), face.sin());
        let (rx, rz) = (-fz, fx);
        let off = (k as f32 - (BAY_SLOTS as f32 - 1.0) * 0.5) * BAY_SLOT_M;
        Some((p.x + fx * BAY_ROW_AHEAD_M + rx * off, p.z + fz * BAY_ROW_AHEAD_M + rz * off))
    }

    /// The drops lying in a bay's slots now.
    pub fn staged_at(&self, bay: u32) -> Vec<u32> {
        let slots: Vec<(f32, f32)> = (0..BAY_SLOTS).filter_map(|k| self.bay_slot(bay, k)).collect();
        self.drops.iter().filter(|d| slots.iter().any(|(x, z)| (d.x - x).hypot(d.z - z) < 0.5 * BAY_SLOT_M)).map(|d| d.id).collect()
    }

    /// Puts `amount` of `item` down at a bay, a whole one per free slot (a fraction in one); the rest goes to the
    /// bay's stock when the row is full.
    pub fn stage_at_bay(&mut self, bay: u32, item: ItemKind, amount: f32) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut left = amount;
        while left > AMOUNT_EPS {
            let piece = left.min(1.0);
            let occupied = |w: &World, x: f32, z: f32| w.drops.iter().any(|d| (d.x - x).hypot(d.z - z) < 0.5 * BAY_SLOT_M);
            let Some((x, z)) = (0..BAY_SLOTS).filter_map(|k| self.bay_slot(bay, k)).find(|&(x, z)| !occupied(self, x, z)) else {
                if let Some(p) = self.waypoints.iter_mut().find(|p| p.id == bay) {
                    add_stock(&mut p.stock, item, left);
                }
                break;
            };
            let id = self.next_drop_id();
            self.drops.push(DropItem { amount: piece, ..DropItem::whole(id, item, x, z) });
            ids.push(id);
            left -= piece;
        }
        ids
    }

    /// The bodies of the sub-workers as boxes, for every camera to draw (id 0: not an object the scan names).
    pub fn crew_boxes(&self) -> Vec<Obb> {
        let mut v = Vec::new();
        for w in &self.crew {
            let g = self.surface_m(w.x, w.z);
            let (s, c) = w.yaw.sin_cos();
            let axes = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
            let at = |fwd: f32, up: f32| [w.x + c * fwd, g + up, w.z + s * fwd];
            let col = colour(w.id, w.form);
            let dark = [col[0] / 2, col[1] / 2, col[2] / 2];
            let load = [150u8, 104, 62];
            match w.form {
                Form::Humanoid => {
                    v.push(Obb { center: at(0.0, 0.42), axes, half: [0.09, 0.42, 0.15], color: dark });
                    v.push(Obb { center: at(0.0, 1.11), axes, half: [0.11, 0.27, 0.19], color: col });
                    v.push(Obb { center: at(0.0, 1.50), axes, half: [0.10, 0.12, 0.09], color: [210, 210, 214] });
                    if !w.carried.is_empty() {
                        v.push(Obb { center: at(0.30, 0.95), axes, half: [0.08, 0.08, 0.30], color: load });
                    }
                }
                Form::Forklift => {
                    v.push(Obb { center: at(-0.15, 0.32), axes, half: [0.55, 0.22, 0.38], color: col });
                    if !w.rider.is_some_and(|r| r.off()) {
                        v.push(Obb { center: at(-0.35, 1.05), axes, half: [0.25, 0.30, 0.25], color: dark });
                    }
                    v.push(Obb { center: at(-0.20, 1.95), axes, half: [0.45, 0.03, 0.40], color: dark });
                    v.push(Obb { center: at(0.45, 1.05), axes, half: [0.04, 1.00, 0.32], color: [70, 70, 76] });
                    let lift = if w.carried.is_empty() { 0.08 } else { 0.30 };
                    for side in [-0.18f32, 0.18] {
                        let p = at(0.95, lift);
                        v.push(Obb { center: [p[0] - s * side, p[1], p[2] + c * side], axes, half: [0.45, 0.02, 0.05], color: [60, 60, 64] });
                    }
                    if !w.carried.is_empty() {
                        let n = w.carried_count().ceil().clamp(1.0, 4.0);
                        v.push(Obb { center: at(0.95, lift + 0.08 * n), axes, half: [0.35, 0.08 * n, 0.30], color: load });
                    }
                }
            }
        }
        v
    }

    pub fn crew_views(&self) -> Vec<crate::snapshot::CrewView> {
        self.crew
            .iter()
            .map(|w| crate::snapshot::CrewView {
                id: w.id,
                form: w.form,
                cameras: w.cameras,
                pos: crate::snapshot::Pos3 { x: w.x, y: self.surface_m(w.x, w.z), z: w.z },
                yaw_deg: w.yaw.to_degrees(),
                speed_mps: w.speed_mps,
                carrying: w.carry_views(),
                carried_kg: w.carried_kg(),
                capacity_kg: w.capacity_kg,
                job: w.job.clone(),
                status_ko: w.status_ko(),
                energy_j: w.energy_j,
                rider: w.rider.filter(|r| r.off()).map(|r| crate::snapshot::Pos3 { x: r.x, y: self.surface_m(r.x, r.z), z: r.z }),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinds::NodeKind;
    use crate::terrain::TerrainKind;
    use crate::world::{Node, WorldConfig};

    fn world() -> World {
        World::new(WorldConfig { seed: 3, parcels: Some(0), terrain: TerrainKind::Open, ..WorldConfig::default() }).unwrap()
    }

    fn run(w: &mut World, id: u32, max_s: f32) -> (bool, String) {
        let mut t = 0.0;
        while t < max_s {
            w.tick();
            t += crate::world::TICK_S;
            if let Some(o) = w.crew_take_outcome(id) {
                return o;
            }
        }
        panic!("worker {id} still on {:?} after {max_s} s", w.crew_worker(id).map(|w| &w.job));
    }

    #[test]
    fn sub_workers_stand_around_the_first_worker_on_free_cells() {
        let mut w = world();
        let ids = w.add_crew(&[Form::Humanoid, Form::Humanoid, Form::Forklift], 1);
        assert_eq!(ids, vec![2, 3, 4]);
        for c in &w.crew {
            let d = (c.x - w.agent.x).hypot(c.z - w.agent.z);
            assert!((1.0..4.5).contains(&d), "worker {} stands {d:.2} m from the first", c.id);
            assert!(!w.blocked_cells().contains(&w.terrain.cell_of(c.x, c.z).unwrap()));
        }
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            assert!((w.crew[a].x - w.crew[b].x).hypot(w.crew[a].z - w.crew[b].z) >= 1.0, "apart");
        }
        assert!(w.crew[2].capacity_kg > w.crew[0].capacity_kg, "a lift truck takes more than two hands");
        assert_eq!(w.crew_boxes().len(), 3 + 3 + 6, "a humanoid is three boxes, a lift truck six (its rider aboard)");
        let many = w.add_crew(&[Form::Humanoid; 9], 2);
        assert_eq!(w.crew.len(), CREW_MAX, "{many:?}");
        assert_eq!(w.remove_crew(), CREW_MAX);
    }

    #[test]
    fn a_humanoid_fells_a_tree_and_stages_its_logs_at_a_bay_for_the_tines() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let (ax, az, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        let tree = (ax + 3.0 * yaw.cos(), az + 3.0 * yaw.sin());
        w.nodes.push(Node { id: 500, kind: NodeKind::Tree, x: tree.0, z: tree.1, hp: 3.0, max_hp: 3.0 });
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        w.crew_command(id, Job::Go { x: tree.0, z: tree.1, reach_m: MINE_REACH_M - 0.1 }).unwrap();
        assert!(run(&mut w, id, 60.0).0);
        w.crew_command(id, Job::Mine { node: 500 }).unwrap();
        let (ok, drops) = run(&mut w, id, 20.0);
        assert!(ok && w.node(500).is_none(), "felled: {drops}");
        let logs: Vec<u32> = w.drops.iter().filter(|d| d.kind == ItemKind::Log).map(|d| d.id).collect();
        assert_eq!(logs.len(), 3);
        let bay = w.add_bay(ax + 1.0, az, (tree.0, tree.1), "상차장");
        assert!(w.is_bay(bay));
        for &d in &logs {
            let p = w.drop_item(d).map(|d| (d.x, d.z)).unwrap();
            w.crew_command(id, Job::Go { x: p.0, z: p.1, reach_m: PICKUP_REACH_M - 0.1 }).unwrap();
            assert!(run(&mut w, id, 30.0).0);
            w.crew_command(id, Job::Pick { drop: d }).unwrap();
            let (ok, why) = run(&mut w, id, 5.0);
            assert!(ok, "{why}");
            let b = w.waypoint(bay).map(|p| (p.x, p.z)).unwrap();
            w.crew_command(id, Job::Go { x: b.0, z: b.1, reach_m: DELIVER_REACH_M - 0.3 }).unwrap();
            assert!(run(&mut w, id, 30.0).0);
            w.crew_command(id, Job::Stage { bay }).unwrap();
            assert!(run(&mut w, id, 5.0).0);
        }
        let staged = w.staged_at(bay);
        assert_eq!(staged.len(), 3, "three logs in the bay's row");
        let slots: Vec<(f32, f32)> = staged.iter().map(|d| w.drop_item(*d).map(|d| (d.x, d.z)).unwrap()).collect();
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            assert!((slots[a].0 - slots[b].0).hypot(slots[a].1 - slots[b].1) >= BAY_SLOT_M - 1e-3, "one log a slot");
        }
        assert!(w.crew[0].energy_j > 0.0);
    }

    #[test]
    fn a_lift_truck_takes_staged_logs_on_its_tines_and_delivers_them() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let (ax, az) = (w.agent.x, w.agent.z);
        let bay = w.add_bay(ax + 2.0, az + 1.0, (ax + 4.0, az + 1.0), "상차장");
        let staged = w.stage_at_bay(bay, ItemKind::Log, 2.0);
        assert_eq!(staged.len(), 2);
        let id = w.add_crew(&[Form::Forklift], 2)[0];
        for d in staged {
            let p = w.drop_item(d).map(|d| (d.x, d.z)).unwrap();
            w.crew_command(id, Job::Go { x: p.0, z: p.1, reach_m: 1.1 }).unwrap();
            assert!(run(&mut w, id, 40.0).0);
            w.crew_command(id, Job::Pick { drop: d }).unwrap();
            let (ok, why) = run(&mut w, id, 5.0);
            assert!(ok, "{why}");
        }
        assert_eq!(w.crew[0].carried_count(), 2.0);
        let wh = w.waypoint_by_key("warehouse").map(|p| (p.id, p.x, p.z)).unwrap();
        w.crew_command(id, Job::Go { x: wh.1, z: wh.2, reach_m: DELIVER_REACH_M - 0.3 }).unwrap();
        assert!(run(&mut w, id, 200.0).0, "drove to the warehouse");
        w.crew_command(id, Job::Deliver { waypoint: wh.0 }).unwrap();
        assert!(run(&mut w, id, 5.0).0);
        assert_eq!(w.waypoint(wh.0).unwrap().stock_of(ItemKind::Log), 2.0);
        assert!(w.crew[0].carried.is_empty());
        let mut ev = Vec::new();
        w.step_crew(0.1, &mut ev);
        assert!(w.crew_command(id, Job::Mine { node: 999 }).is_err(), "no such node");
    }

    #[test]
    fn a_lift_truck_rider_steps_down_fells_the_tree_on_foot_and_climbs_on_again() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let id = w.add_crew(&[Form::Forklift], 1)[0];
        let (x, z) = { let c = w.crew_worker(id).unwrap(); (c.x, c.z) };
        // Two metres off, on the side away from the first worker.
        let away = (z - w.agent.z).atan2(x - w.agent.x);
        let tree = (x + 2.0 * away.cos(), z + 2.0 * away.sin());
        w.nodes.push(Node { id: 600, kind: NodeKind::Tree, x: tree.0, z: tree.1, hp: 2.0, max_hp: 2.0 });
        assert!(w.crew_command(id, Job::Mine { node: 600 }).is_ok(), "2 m away: the rider walks the rest");
        let (mut phases, mut off_far) = (Vec::new(), 0.0f32);
        let mut events = Vec::new();
        for _ in 0..400 {
            events.extend(w.tick());
            let c = w.crew_worker(id).unwrap();
            assert!((c.x - x).abs() < 1e-4 && (c.z - z).abs() < 1e-4, "the truck stays where it stopped");
            if let Some(r) = c.rider {
                if phases.last() != Some(&r.phase) {
                    phases.push(r.phase);
                }
                if r.off() {
                    off_far = off_far.max((r.x - x).hypot(r.z - z));
                    assert!(w.rider_boxes(None).len() == 4, "the rider on foot is drawn");
                }
            }
            if let Some(o) = w.crew_take_outcome(id) {
                assert!(o.0, "{o:?}");
                break;
            }
        }
        assert_eq!(phases, vec![RiderPhase::Dismount, RiderPhase::WalkOut, RiderPhase::Mine, RiderPhase::WalkBack, RiderPhase::Remount]);
        assert!(off_far > 0.9, "the rider walked over to the tree: {off_far:.2} m");
        assert!(w.node(600).is_none() && w.crew_worker(id).unwrap().rider.is_none());
        let said: Vec<String> = events.iter().filter_map(|e| match &e.event { WorldEvent::Rider { label, .. } => Some(label.clone()), _ => None }).collect();
        assert!(said.iter().any(|l| l.contains("하차")) && said.iter().any(|l| l.contains("재탑승")), "{said:?}");
    }

    #[test]
    fn two_robots_meeting_head_on_stop_for_each_other_and_route_round() {
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        let ids = w.add_crew(&[Form::Humanoid, Form::Humanoid], 1);
        let (ax, az, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        // Put them 6 m apart on one line ahead of the first worker, each walking to where the other stands.
        let p = |d: f32| (ax + d * yaw.cos(), az + d * yaw.sin());
        let (a, b) = (p(4.0), p(10.0));
        for (id, at) in [(ids[0], a), (ids[1], b)] {
            let c = w.crew_worker_mut(id).unwrap();
            (c.x, c.z) = at;
        }
        w.crew_command(ids[0], Job::Go { x: b.0, z: b.1, reach_m: 0.3 }).unwrap();
        w.crew_command(ids[1], Job::Go { x: a.0, z: a.1, reach_m: 0.3 }).unwrap();
        let (mut closest, mut held, mut rerouted) = (f32::INFINITY, false, false);
        let mut arrived = (false, false);
        for _ in 0..1500 {
            for e in w.tick() {
                if let WorldEvent::Hold { label, .. } = &e.event {
                    held = true;
                    rerouted |= label.contains("돌아가는 길");
                }
            }
            let (c0, c1) = (w.crew_worker(ids[0]).unwrap(), w.crew_worker(ids[1]).unwrap());
            closest = closest.min((c0.x - c1.x).hypot(c0.z - c1.z) - c0.radius_m() - c1.radius_m());
            arrived.0 |= w.crew_take_outcome(ids[0]).is_some_and(|o| o.0);
            arrived.1 |= w.crew_take_outcome(ids[1]).is_some_and(|o| o.0);
            if arrived.0 && arrived.1 {
                break;
            }
        }
        assert!(held, "one saw the other in its way and stopped");
        assert!(rerouted, "the one giving way routed round");
        assert!(closest > 0.0, "their outlines never touched: closest gap {closest:.2} m");
        assert!(arrived.0 && arrived.1, "both got where they were going");
    }

    #[test]
    fn a_robot_hidden_behind_a_tree_is_not_seen_and_one_in_the_open_is() {
        let mut w = world();
        w.nodes.clear();
        let id = w.add_crew(&[Form::Humanoid], 1)[0];
        let (x, z) = (w.agent.x, w.agent.z);
        {
            let c = w.crew_worker_mut(id).unwrap();
            (c.x, c.z) = (x + 1.6, z);
        }
        let fp = w.footprint();
        let me = |dir: f32| traffic::Looker { id: 1, at: (x, z), dir, half_w: fp.half_w_m, front: fp.front_m, speed: 1.0, eye_m: 1.4 };
        let line = |dir: f32| vec![(x + 3.0 * dir.cos(), z + 3.0 * dir.sin())];
        assert_eq!(w.robot_in_way(&me(0.0), &line(0.0)).map(|s| s.0), Some(id), "in the open, on the way");
        assert_eq!(w.robot_in_way(&me(std::f32::consts::PI), &line(std::f32::consts::PI)), None, "behind the eye: not seen");
        assert_eq!(w.robot_in_way(&me(1.2), &line(1.2)), None, "the path passes clear of it: passed, not waited for");
        w.nodes.push(Node { id: 700, kind: NodeKind::Tree, x: x + 0.9, z, hp: 3.0, max_hp: 3.0 });
        assert!(!w.line_of_sight((x, z), (x + 1.6, z), 1.2), "the trunk is in the line");
    }

    #[test]
    fn the_first_worker_delivering_to_a_bay_stages_its_load_on_the_ground() {
        let mut w = world();
        w.drops.clear();
        let (ax, az) = (w.agent.x, w.agent.z);
        let bay = w.add_bay(ax + 0.8, az, (ax + 2.0, az), "상차장");
        crate::world::add_stock(&mut w.agent.carried, ItemKind::Log, 2.0);
        let ev = w.command(crate::world::Action::Deliver { waypoint: bay, item: None }).unwrap();
        let mut all = ev;
        for _ in 0..60 {
            all.extend(w.tick());
        }
        assert!(all.iter().any(|e| matches!(e.event, WorldEvent::Delivered { .. })), "the first worker's driver still hears the delivery");
        assert_eq!(w.staged_at(bay).len(), 2, "two logs on the ground in the bay's row");
        assert_eq!(w.waypoint(bay).unwrap().stock_of(ItemKind::Log), 0.0);
    }
}
