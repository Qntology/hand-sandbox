//! The simulated open world and the robot living in it.
//!
//! The robot is the project's own hand and foot model: the base spec decides
//! where it can drive and how fast (through `hand_core::simulate_drive`), the
//! finger count decides how fast it mines, and `hand_core::grasp` decides
//! whether it can pick an item up at all. Nothing here is tuned to make a
//! scenario succeed; a weak configuration fails for the same reason it would
//! fail in the standalone previews.
//!
//! Assumptions that are not in the hand/foot models are named constants below
//! and listed in `data/README.md`.

use std::collections::HashSet;

use hand_core::{
    battery_capacity_j, compute_body, grasp, BodyInputs, BodyStatus, ConfigError, GraspSetup, Hand,
    HandConfig, Payload,
};
use serde::{Deserialize, Serialize};

use crate::kinds::{Ground, ItemKind, NodeKind};
use crate::nav::{self, Cell, EdgePlan, Goal};
use crate::rng::Rng;
use crate::snapshot::*;
use crate::terrain::{cell_center, FlatZone, Terrain, CELL_M};

/// Length of one simulation tick, seconds.
pub const TICK_S: f32 = 0.1;
/// Camera height above the ground the robot stands on, metres.
pub const EYE_HEIGHT_M: f32 = 0.5;
/// How close the robot must be to mine a node, metres.
pub const MINE_REACH_M: f32 = 0.9;
/// How close the robot must be to pick up a dropped item, metres.
pub const PICKUP_REACH_M: f32 = 0.7;
/// How close the robot must be to hand over cargo at a waypoint, metres.
pub const DELIVER_REACH_M: f32 = 1.6;
/// Time to close the hand on an item and lift it, seconds (assumption).
pub const PICKUP_TIME_S: f32 = 1.0;
/// Time to put cargo down, seconds (assumption).
pub const DELIVER_TIME_S: f32 = 1.0;
/// Electrical power while mining, picking or putting down, watts (assumption).
pub const MANIPULATION_POWER_W: f64 = 30.0;
/// Turning speed, radians per second (assumption).
pub const TURN_RATE_RAD_S: f32 = 2.5;
/// Friction between finger pads and every item (assumption).
pub const ITEM_FRICTION: f32 = 0.6;
/// Heading error above which a non-holonomic base turns in place, radians.
const TURN_IN_PLACE_RAD: f32 = 0.6;
/// Heading error at which the in-place turn stops, radians.
const TURN_DONE_RAD: f32 = 0.2;
/// Heading error accepted before manipulation starts, radians.
const FACE_DONE_RAD: f32 = 0.08;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    pub seed: u64,
    /// Cells per side (square world). Each cell is 0.5 m.
    pub cells: usize,
    pub hand: HandConfig,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig { seed: 7, cells: 64, hand: HandConfig::preset(hand_core::FingerCount::Five) }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u32,
    pub kind: NodeKind,
    pub x: f32,
    pub z: f32,
    /// Remaining work in full-hand seconds.
    pub hp: f32,
    pub max_hp: f32,
}

#[derive(Clone, Debug)]
pub struct DropItem {
    pub id: u32,
    pub kind: ItemKind,
    pub x: f32,
    pub z: f32,
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub id: u32,
    pub key: String,
    pub label_ko: String,
    pub x: f32,
    pub z: f32,
    pub stock: Vec<(ItemKind, u32)>,
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub carried: Vec<(ItemKind, u32)>,
    pub energy_used_j: f64,
    pub strikes: u32,
    pub speed_mps: f32,
    pub grip_margin: Option<f32>,
    pub terrain_margin: Option<f32>,
    strike_acc: f32,
}

/// A command for the robot. One action runs at a time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Drive until a free cell within `reach_m` of the point (0 = that cell).
    MoveTo { x: f32, z: f32, reach_m: f32 },
    /// Turn in place to a heading, degrees (0 = +x, 90 = +z).
    Turn { yaw_deg: f32 },
    /// Turn to face a point.
    Face { x: f32, z: f32 },
    Mine { node: u32 },
    PickUp { drop: u32 },
    /// Put carried items into a waypoint's stock; `item` limits it to one kind.
    Deliver { waypoint: u32, item: Option<ItemKind> },
}

impl Action {
    pub fn label(&self) -> String {
        match self {
            Action::MoveTo { x, z, reach_m } => format!("move_to({x:.1},{z:.1},reach={reach_m:.1})"),
            Action::Turn { yaw_deg } => format!("turn({yaw_deg:.0})"),
            Action::Face { x, z } => format!("face({x:.1},{z:.1})"),
            Action::Mine { node } => format!("mine({})", node_id(*node)),
            Action::PickUp { drop } => format!("pick_up({})", drop_id(*drop)),
            Action::Deliver { waypoint, item } => match item {
                Some(i) => format!("deliver({},{})", waypoint_id(*waypoint), i.id()),
                None => format!("deliver({})", waypoint_id(*waypoint)),
            },
        }
    }
}

/// Why an action could not start. `code()` is stable and goes into logs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum ActionError {
    Busy,
    BatteryDepleted,
    UnknownTarget,
    NoPath,
    OutOfReach { distance_m: f32, reach_m: f32 },
    GraspUnstable { margin: f32 },
    CapacityExceeded { carried_kg: f32, item_kg: f32, capacity_kg: f32 },
    NothingToDeliver,
}

impl ActionError {
    pub fn code(&self) -> &'static str {
        match self {
            ActionError::Busy => "busy",
            ActionError::BatteryDepleted => "battery_depleted",
            ActionError::UnknownTarget => "unknown_target",
            ActionError::NoPath => "no_path",
            ActionError::OutOfReach { .. } => "out_of_reach",
            ActionError::GraspUnstable { .. } => "grasp_unstable",
            ActionError::CapacityExceeded { .. } => "capacity_exceeded",
            ActionError::NothingToDeliver => "nothing_to_deliver",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorldEvent {
    ActionStarted { action: String },
    ActionDone { action: String, ok: bool, reason: Option<String> },
    CurbClimbed { height_m: f32 },
    NodeBroken { node: String, kind: NodeKind, drops: Vec<String> },
    ItemPicked { drop: String, item: ItemKind, mass_kg: f32, grip_margin: f32 },
    Delivered { waypoint: String, item: ItemKind, count: u32 },
    BatteryDepleted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedEvent {
    pub tick: u64,
    pub time_s: f64,
    pub event: WorldEvent,
}

#[derive(Clone, Debug)]
struct MoveState {
    edges: Vec<EdgePlan>,
    idx: usize,
    /// Seconds of curb climbing left on the current edge.
    pause_left: f32,
    /// Seconds of driving done on the current edge.
    progress_s: f32,
    announced_pause: bool,
}

#[derive(Clone, Debug)]
enum WorkKind {
    Mine { node: u32 },
    Pick { drop: u32 },
    Deliver { waypoint: u32, item: Option<ItemKind> },
}

#[derive(Clone, Debug)]
struct WorkState {
    kind: WorkKind,
    remaining_s: f32,
}

#[derive(Clone, Debug)]
enum Active {
    Move(MoveState),
    Turn { target: f32 },
    Work(WorkState),
}

pub struct World {
    pub cfg: WorldConfig,
    pub hand: Hand,
    pub terrain: Terrain,
    pub nodes: Vec<Node>,
    pub drops: Vec<DropItem>,
    pub waypoints: Vec<Waypoint>,
    pub agent: Agent,
    pub tick: u64,
    pub time_s: f64,
    grasps: Vec<GraspInfo>,
    active: Option<(Action, Active)>,
    rng: Rng,
    next_drop: u32,
}

fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

fn add_stock(list: &mut Vec<(ItemKind, u32)>, kind: ItemKind, n: u32) {
    if let Some(e) = list.iter_mut().find(|(k, _)| *k == kind) {
        e.1 += n;
    } else {
        list.push((kind, n));
    }
}

impl World {
    pub fn new(cfg: WorldConfig) -> Result<World, ConfigError> {
        let hand = cfg.hand.build()?;
        let cells = cfg.cells.clamp(32, 160);
        let size = cells as f32 * CELL_M;

        let base_pos = (0.14 * size, 0.14 * size);
        let wh_pos = (0.78 * size, 0.66 * size);
        let wall_cx = (0.46 * cells as f32).round() as usize;
        let wall_x = (wall_cx as f32 + 0.5) * CELL_M;

        // Flat zones: the two waypoints, plus a chain along the curb wall so
        // the wall stands on level ground.
        let mut zones: Vec<FlatZone> = vec![(base_pos.0, base_pos.1, 3.5), (wh_pos.0, wh_pos.1, 3.5)];
        let mut zc = 1.0;
        while zc < size {
            zones.push((wall_x, zc, 1.4));
            zc += 2.0;
        }
        let mut terrain = Terrain::generate(cfg.seed, cells, cells, &zones);

        // Sand pads under the waypoints.
        for (px, pz) in [base_pos, wh_pos] {
            for cz in 0..cells {
                for cx in 0..cells {
                    let (x, z) = cell_center(cx, cz);
                    if ((x - px).powi(2) + (z - pz).powi(2)).sqrt() <= 2.0 {
                        terrain.set_ground(cx, cz, Ground::Sand);
                    }
                }
            }
        }

        // The curb wall: 5, 10 and 15 cm sections and one open gap.
        let gap_from = (0.86 * cells as f32) as usize;
        let gap_to = gap_from + 2;
        for cz in 0..cells {
            if (gap_from..gap_to).contains(&cz) {
                continue;
            }
            let f = cz as f32 / cells as f32;
            let cm = if f < 0.30 {
                5
            } else if f < 0.55 {
                10
            } else {
                15
            };
            terrain.set_curb_cm(wall_cx, cz, cm);
        }

        let mut rng = Rng::new(cfg.seed ^ 0xA5A5_5A5A_1234_8765);
        let mut occupied: HashSet<Cell> = HashSet::new();
        let mut nodes = Vec::new();
        let mut next_node = 1u32;
        let wanted = [
            (NodeKind::Tree, (cells * cells / 140).max(4)),
            (NodeKind::Boulder, (cells * cells / 300).max(3)),
            (NodeKind::CoalOre, (cells * cells / 520).max(2)),
            (NodeKind::IronOre, (cells * cells / 700).max(2)),
        ];
        let in_zone = |x: f32, z: f32| {
            [base_pos, wh_pos].iter().any(|&(px, pz)| ((x - px).powi(2) + (z - pz).powi(2)).sqrt() < 3.8)
        };
        for (kind, count) in wanted {
            let mut placed = 0;
            let mut tries = 0;
            while placed < count && tries < 6000 {
                tries += 1;
                let cx = rng.below(cells as u32) as usize;
                let cz = rng.below(cells as u32) as usize;
                let (x, z) = cell_center(cx, cz);
                let h = terrain.height_cm(cx, cz);
                let strict = tries < 3000;
                let height_ok = match kind {
                    NodeKind::Tree => h < 70,
                    NodeKind::Boulder => true,
                    NodeKind::CoalOre | NodeKind::IronOre => !strict || h >= 20,
                };
                let near_wall = cx.abs_diff(wall_cx) <= 1;
                if !height_ok
                    || near_wall
                    || in_zone(x, z)
                    || terrain.curb_m(cx, cz) > 0.0
                    || occupied.contains(&(cx, cz))
                {
                    continue;
                }
                occupied.insert((cx, cz));
                let hp = kind.break_time_s();
                nodes.push(Node { id: next_node, kind, x, z, hp, max_hp: hp });
                next_node += 1;
                placed += 1;
            }
        }

        let waypoints = vec![
            Waypoint { id: 1, key: "base_camp".into(), label_ko: "베이스캠프".into(), x: base_pos.0, z: base_pos.1, stock: vec![] },
            Waypoint { id: 2, key: "warehouse".into(), label_ko: "창고".into(), x: wh_pos.0, z: wh_pos.1, stock: vec![] },
        ];
        let yaw = (wh_pos.1 - base_pos.1).atan2(wh_pos.0 - base_pos.0);
        let agent = Agent {
            x: base_pos.0,
            z: base_pos.1,
            yaw,
            carried: vec![],
            energy_used_j: 0.0,
            strikes: 0,
            speed_mps: 0.0,
            grip_margin: None,
            terrain_margin: None,
            strike_acc: 0.0,
        };

        let grasps = ItemKind::ALL
            .iter()
            .map(|&k| {
                let s = k.spec();
                let out = grasp(
                    &hand,
                    &Payload { name: s.label_en.into(), size_m: s.size_m, mass_kg: s.mass_kg, friction: ITEM_FRICTION },
                    &GraspSetup::default(),
                );
                let r = &out.result;
                let margin = if r.required_n > 1e-6 { r.friction_capacity_n / r.required_n } else { 99.0 };
                GraspInfo {
                    item: k,
                    stable: r.stable,
                    margin: margin.min(99.0),
                    max_payload_kg: r.max_payload_kg,
                    grasp_type: format!("{:?}", r.grasp_type).to_lowercase(),
                }
            })
            .collect();

        Ok(World {
            cfg,
            hand,
            terrain,
            nodes,
            drops: vec![],
            waypoints,
            agent,
            tick: 0,
            time_s: 0.0,
            grasps,
            active: None,
            rng,
            next_drop: 1,
        })
    }

    // ----- lookups -----

    pub fn grasp_info(&self, item: ItemKind) -> &GraspInfo {
        self.grasps.iter().find(|g| g.item == item).expect("every item has a grasp entry")
    }

    pub fn grasp_table(&self) -> &[GraspInfo] {
        &self.grasps
    }

    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn drop_item(&self, id: u32) -> Option<&DropItem> {
        self.drops.iter().find(|d| d.id == id)
    }

    pub fn waypoint(&self, id: u32) -> Option<&Waypoint> {
        self.waypoints.iter().find(|w| w.id == id)
    }

    pub fn waypoint_by_key(&self, key: &str) -> Option<&Waypoint> {
        self.waypoints.iter().find(|w| w.key == key)
    }

    pub fn base_spec(&self) -> &hand_core::BaseSpec {
        &self.hand.base
    }

    pub fn carried_kg(&self) -> f32 {
        self.agent.carried.iter().map(|(k, n)| k.spec().mass_kg * *n as f32).sum()
    }

    pub fn carried_count(&self, item: ItemKind) -> u32 {
        self.agent.carried.iter().find(|(k, _)| *k == item).map_or(0, |e| e.1)
    }

    pub fn capacity_kg(&self) -> f32 {
        self.hand.base.kind.carry_capacity_kg()
    }

    pub fn battery_capacity_j(&self) -> f64 {
        battery_capacity_j(&self.hand.base)
    }

    pub fn battery_ratio(&self) -> f32 {
        (1.0 - self.agent.energy_used_j / self.battery_capacity_j()).clamp(0.0, 1.0) as f32
    }

    pub fn battery_depleted(&self) -> bool {
        self.agent.energy_used_j >= self.battery_capacity_j()
    }

    /// Mining progress per second: a full five-finger hand works at 1.0
    /// (assumption: proportional to the finger count).
    pub fn mining_rate(&self) -> f32 {
        u8::from(self.hand.config.fingers) as f32 / 5.0
    }

    pub fn busy(&self) -> bool {
        self.active.is_some()
    }

    pub fn current_action(&self) -> Option<String> {
        self.active.as_ref().map(|(a, _)| a.label())
    }

    fn agent_cell(&self) -> Cell {
        self.terrain.cell_of(self.agent.x, self.agent.z).unwrap_or((0, 0))
    }

    /// Cells that hold a resource node (the base cannot drive through them).
    fn blocked_cells(&self) -> HashSet<Cell> {
        self.nodes.iter().filter_map(|n| self.terrain.cell_of(n.x, n.z)).collect()
    }

    /// Ground height under the robot, curb included.
    pub fn agent_ground_m(&self) -> f32 {
        match self.terrain.cell_of(self.agent.x, self.agent.z) {
            Some((cx, cz)) => self.terrain.height_m(cx, cz) + self.terrain.curb_m(cx, cz),
            None => 0.0,
        }
    }

    /// Ground height including any curb at a point.
    pub fn surface_m(&self, x: f32, z: f32) -> f32 {
        match self.terrain.cell_of(x, z) {
            Some((cx, cz)) => self.terrain.height_m(cx, cz) + self.terrain.curb_m(cx, cz),
            None => 0.0,
        }
    }

    /// Estimates a route for the current base without moving. `None` when no
    /// route exists.
    pub fn plan_route(&self, goal: Goal) -> Option<nav::Plan> {
        let blocked = self.blocked_cells();
        nav::plan(&self.hand.base, &self.terrain, &|x, z| blocked.contains(&(x, z)), self.agent_cell(), goal)
    }

    // ----- commands -----

    /// Starts an action. Fails right away when it cannot start.
    pub fn command(&mut self, action: Action) -> Result<Vec<TimedEvent>, ActionError> {
        if self.active.is_some() {
            return Err(ActionError::Busy);
        }
        if self.battery_depleted() {
            return Err(ActionError::BatteryDepleted);
        }
        let state = match &action {
            Action::MoveTo { x, z, reach_m } => {
                let goal = if *reach_m <= 0.0 {
                    match self.terrain.cell_of(*x, *z) {
                        Some(c) => Goal::Cell(c),
                        None => return Err(ActionError::UnknownTarget),
                    }
                } else {
                    Goal::Within { x: *x, z: *z, reach_m: *reach_m }
                };
                let plan = self.plan_route(goal).ok_or(ActionError::NoPath)?;
                Active::Move(MoveState { edges: plan.edges, idx: 0, pause_left: 0.0, progress_s: 0.0, announced_pause: false })
            }
            Action::Turn { yaw_deg } => Active::Turn { target: wrap_angle(yaw_deg.to_radians()) },
            Action::Face { x, z } => Active::Turn { target: (z - self.agent.z).atan2(x - self.agent.x) },
            Action::Mine { node } => {
                let n = self.node(*node).ok_or(ActionError::UnknownTarget)?;
                self.check_reach(n.x, n.z, MINE_REACH_M)?;
                Active::Work(WorkState { kind: WorkKind::Mine { node: *node }, remaining_s: n.hp })
            }
            Action::PickUp { drop } => {
                let (dx, dz, kind) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind)
                };
                self.check_reach(dx, dz, PICKUP_REACH_M)?;
                let info = self.grasp_info(kind).clone();
                self.agent.grip_margin = Some(info.margin);
                if !info.stable {
                    return Err(ActionError::GraspUnstable { margin: info.margin });
                }
                let item_kg = kind.spec().mass_kg;
                if self.carried_kg() + item_kg > self.capacity_kg() + 1e-4 {
                    return Err(ActionError::CapacityExceeded {
                        carried_kg: self.carried_kg(),
                        item_kg,
                        capacity_kg: self.capacity_kg(),
                    });
                }
                Active::Work(WorkState { kind: WorkKind::Pick { drop: *drop }, remaining_s: PICKUP_TIME_S })
            }
            Action::Deliver { waypoint, item } => {
                let w = self.waypoint(*waypoint).ok_or(ActionError::UnknownTarget)?;
                self.check_reach(w.x, w.z, DELIVER_REACH_M)?;
                let any = self.agent.carried.iter().any(|(k, n)| *n > 0 && item.map_or(true, |i| i == *k));
                if !any {
                    return Err(ActionError::NothingToDeliver);
                }
                Active::Work(WorkState { kind: WorkKind::Deliver { waypoint: *waypoint, item: *item }, remaining_s: DELIVER_TIME_S })
            }
        };
        let label = action.label();
        self.active = Some((action, state));
        Ok(vec![self.stamp(WorldEvent::ActionStarted { action: label })])
    }

    /// Stops the current action.
    pub fn cancel(&mut self) -> Vec<TimedEvent> {
        match self.active.take() {
            Some((a, _)) => {
                self.agent.speed_mps = 0.0;
                self.agent.terrain_margin = None;
                vec![self.stamp(WorldEvent::ActionDone { action: a.label(), ok: false, reason: Some("cancelled".into()) })]
            }
            None => vec![],
        }
    }

    fn check_reach(&self, x: f32, z: f32, reach: f32) -> Result<(), ActionError> {
        let d = ((x - self.agent.x).powi(2) + (z - self.agent.z).powi(2)).sqrt();
        if d > reach {
            Err(ActionError::OutOfReach { distance_m: d, reach_m: reach })
        } else {
            Ok(())
        }
    }

    fn stamp(&self, event: WorldEvent) -> TimedEvent {
        TimedEvent { tick: self.tick, time_s: self.time_s, event }
    }

    // ----- stepping -----

    /// Advances one tick of [`TICK_S`].
    pub fn tick(&mut self) -> Vec<TimedEvent> {
        let ev = self.step(TICK_S);
        self.tick += 1;
        ev
    }

    /// Advances the world by `dt` seconds.
    pub fn step(&mut self, dt: f32) -> Vec<TimedEvent> {
        let mut events = Vec::new();
        let mut remaining = dt;
        let mut guard = 0;
        while remaining > 1e-6 && self.active.is_some() && guard < 64 {
            guard += 1;
            if self.battery_depleted() {
                events.push(self.stamp(WorldEvent::BatteryDepleted));
                self.finish(false, Some("battery_depleted".into()), &mut events);
                break;
            }
            let used = self.advance(remaining, &mut events);
            remaining -= used.max(1e-6);
        }
        if self.active.is_none() {
            self.agent.speed_mps = 0.0;
        }
        self.time_s += dt as f64;
        events
    }

    fn finish(&mut self, ok: bool, reason: Option<String>, events: &mut Vec<TimedEvent>) {
        if let Some((a, _)) = self.active.take() {
            self.agent.speed_mps = 0.0;
            self.agent.terrain_margin = None;
            events.push(self.stamp(WorldEvent::ActionDone { action: a.label(), ok, reason }));
        }
    }

    fn rotate_toward(&mut self, target: f32, max_dt: f32) -> f32 {
        let err = wrap_angle(target - self.agent.yaw);
        let turn = (TURN_RATE_RAD_S * max_dt).min(err.abs());
        self.agent.yaw = wrap_angle(self.agent.yaw + turn * err.signum());
        turn / TURN_RATE_RAD_S
    }

    /// Runs the active action for up to `remaining` seconds; returns time used.
    fn advance(&mut self, remaining: f32, events: &mut Vec<TimedEvent>) -> f32 {
        let (action, mut state) = self.active.take().expect("checked by the caller");
        let used;
        let mut done: Option<(bool, Option<String>)> = None;
        match &mut state {
            Active::Turn { target } => {
                let err = wrap_angle(*target - self.agent.yaw).abs();
                if err <= FACE_DONE_RAD * 0.5 {
                    done = Some((true, None));
                    used = 0.0;
                } else {
                    let t = self.rotate_toward(*target, remaining);
                    self.add_energy(self.hand.base.power_w as f64 * t as f64);
                    used = t;
                }
            }
            Active::Move(m) => {
                used = self.advance_move(m, remaining, events, &mut done);
            }
            Active::Work(w) => {
                used = self.advance_work(w, remaining, events, &mut done);
            }
        }
        self.active = Some((action, state));
        if let Some((ok, reason)) = done {
            self.finish(ok, reason, events);
        }
        used
    }

    fn add_energy(&mut self, j: f64) {
        self.agent.energy_used_j += j;
    }

    fn advance_move(&mut self, m: &mut MoveState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        if m.idx >= m.edges.len() {
            *done = Some((true, None));
            return 0.0;
        }
        let e = m.edges[m.idx];
        let (fx, fz) = cell_center(e.from.0, e.from.1);
        let (tx, tz) = cell_center(e.to.0, e.to.1);
        self.agent.terrain_margin = Some(e.margin);
        let power = self.hand.base.power_w as f64;

        // Curb climb pause at the start of the edge.
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

        let heading = (tz - fz).atan2(tx - fx);
        let err = wrap_angle(heading - self.agent.yaw).abs();
        if !self.hand.base.holonomic && err > TURN_IN_PLACE_RAD {
            // Turn until the heading error is small, then drive.
            let need = (err - TURN_DONE_RAD) / TURN_RATE_RAD_S;
            let t = need.min(remaining);
            self.rotate_toward(heading, t);
            self.agent.speed_mps = 0.0;
            self.add_energy(power * t as f64);
            return t;
        }

        let left = e.move_time_s - m.progress_s;
        let t = left.min(remaining);
        m.progress_s += t;
        self.rotate_toward(heading, t);
        let f = (m.progress_s / e.move_time_s).clamp(0.0, 1.0);
        self.agent.x = fx + (tx - fx) * f;
        self.agent.z = fz + (tz - fz) * f;
        self.agent.speed_mps = e.horizontal_m / e.move_time_s;
        self.add_energy(power * t as f64);
        if m.progress_s >= e.move_time_s - 1e-6 {
            self.agent.x = tx;
            self.agent.z = tz;
            m.idx += 1;
            m.progress_s = 0.0;
            m.announced_pause = false;
            if m.idx >= m.edges.len() {
                *done = Some((true, None));
            }
        }
        t
    }

    fn advance_work(&mut self, w: &mut WorkState, remaining: f32, events: &mut Vec<TimedEvent>, done: &mut Option<(bool, Option<String>)>) -> f32 {
        // Face the target first.
        let (tx, tz) = match w.kind {
            WorkKind::Mine { node } => match self.node(node) {
                Some(n) => (n.x, n.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Pick { drop } => match self.drop_item(drop) {
                Some(d) => (d.x, d.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            WorkKind::Deliver { waypoint, .. } => match self.waypoint(waypoint) {
                Some(p) => (p.x, p.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
        };
        let heading = (tz - self.agent.z).atan2(tx - self.agent.x);
        let err = wrap_angle(heading - self.agent.yaw).abs();
        if err > FACE_DONE_RAD {
            let t = self.rotate_toward(heading, remaining);
            self.add_energy(MANIPULATION_POWER_W * t as f64);
            return t;
        }

        let t = w.remaining_s.min(remaining);
        self.add_energy(MANIPULATION_POWER_W * t as f64);
        match w.kind.clone() {
            WorkKind::Mine { node } => {
                let rate = self.mining_rate();
                let work = (t * rate).max(1e-6);
                // Every full second of work counts as one strike on the joints.
                self.agent.strike_acc += t;
                while self.agent.strike_acc >= 1.0 {
                    self.agent.strike_acc -= 1.0;
                    self.agent.strikes += 1;
                }
                let n = self.nodes.iter_mut().find(|n| n.id == node).expect("looked up above");
                n.hp -= work.min(n.hp);
                // Remaining time at the current rate.
                w.remaining_s = n.hp / rate;
                if n.hp <= 1e-5 {
                    self.break_node(node, events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Pick { drop } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.complete_pick(drop, events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Deliver { waypoint, item } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.complete_deliver(waypoint, item, events);
                    *done = Some((true, None));
                }
            }
        }
        t
    }

    fn break_node(&mut self, id: u32, events: &mut Vec<TimedEvent>) {
        let pos = self.nodes.iter().position(|n| n.id == id).expect("node exists");
        let node = self.nodes.remove(pos);
        let spec = node.kind.spec();
        let mut ids = Vec::new();
        for i in 0..spec.drop_count {
            // Spread the drops on a small ring inside the node's cell.
            let a = (i as f32 / spec.drop_count as f32) * std::f32::consts::TAU + self.rng.range_f32(0.0, 1.0);
            let r = if spec.drop_count == 1 { 0.0 } else { 0.13 };
            let (x, z) = (node.x + r * a.cos(), node.z + r * a.sin());
            let d = DropItem { id: self.next_drop, kind: spec.drop, x, z };
            self.next_drop += 1;
            ids.push(drop_id(d.id));
            self.drops.push(d);
        }
        events.push(self.stamp(WorldEvent::NodeBroken { node: node_id(id), kind: node.kind, drops: ids }));
    }

    fn complete_pick(&mut self, id: u32, events: &mut Vec<TimedEvent>) {
        if let Some(pos) = self.drops.iter().position(|d| d.id == id) {
            let d = self.drops.remove(pos);
            add_stock(&mut self.agent.carried, d.kind, 1);
            let margin = self.agent.grip_margin.unwrap_or(0.0);
            events.push(self.stamp(WorldEvent::ItemPicked { drop: drop_id(id), item: d.kind, mass_kg: d.kind.spec().mass_kg, grip_margin: margin }));
        }
    }

    fn complete_deliver(&mut self, waypoint: u32, item: Option<ItemKind>, events: &mut Vec<TimedEvent>) {
        let mut moved: Vec<(ItemKind, u32)> = Vec::new();
        self.agent.carried.retain_mut(|(k, n)| {
            if *n > 0 && item.map_or(true, |i| i == *k) {
                moved.push((*k, *n));
                false
            } else {
                *n > 0
            }
        });
        for (k, n) in moved {
            if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == waypoint) {
                add_stock(&mut w.stock, k, n);
            }
            events.push(self.stamp(WorldEvent::Delivered { waypoint: waypoint_id(waypoint), item: k, count: n }));
        }
    }

    // ----- views -----

    pub fn terrain_grid(&self) -> TerrainGrid {
        let t = &self.terrain;
        let mut heights = Vec::with_capacity(t.w * t.d);
        let mut curbs = Vec::with_capacity(t.w * t.d);
        let mut ground = Vec::with_capacity(t.w * t.d);
        for cz in 0..t.d {
            for cx in 0..t.w {
                heights.push(t.height_cm(cx, cz) as i16);
                curbs.push((t.curb_m(cx, cz) * 100.0).round() as u8);
                ground.push(match t.ground_at(cx, cz) {
                    Ground::Sand => 0,
                    Ground::Grass => 1,
                    Ground::Dirt => 2,
                    Ground::Rock => 3,
                });
            }
        }
        TerrainGrid { cells_w: t.w, cells_d: t.d, cell_m: CELL_M, heights_cm: heights, curb_cm: curbs, ground }
    }

    fn carry_views(list: &[(ItemKind, u32)]) -> Vec<CarryView> {
        list.iter()
            .filter(|(_, n)| *n > 0)
            .map(|(k, n)| CarryView { item: *k, label_ko: k.spec().label_ko.into(), count: *n, mass_kg: k.spec().mass_kg * *n as f32 })
            .collect()
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        let (w, d) = self.terrain.size_m();
        let at = |x: f32, z: f32| Pos3 { x, y: self.surface_m(x, z), z };
        WorldSnapshot {
            tick: self.tick,
            time_s: self.time_s,
            seed: self.cfg.seed,
            bounds: Bounds { width_m: w, depth_m: d, cells_w: self.terrain.w, cells_d: self.terrain.d, cell_m: CELL_M },
            agent: AgentView {
                pos: at(self.agent.x, self.agent.z),
                yaw_deg: self.agent.yaw.to_degrees(),
                speed_mps: self.agent.speed_mps,
                base: format!("{:?}", self.hand.base.kind).to_lowercase(),
                fingers: u8::from(self.hand.config.fingers),
                carrying: Self::carry_views(&self.agent.carried),
                carried_kg: self.carried_kg(),
                capacity_kg: self.capacity_kg(),
                battery_ratio: self.battery_ratio(),
                action: self.current_action(),
            },
            waypoints: self
                .waypoints
                .iter()
                .map(|p| WaypointView {
                    id: waypoint_id(p.id),
                    key: p.key.clone(),
                    label_ko: p.label_ko.clone(),
                    pos: at(p.x, p.z),
                    stock: Self::carry_views(&p.stock),
                })
                .collect(),
            nodes: self
                .nodes
                .iter()
                .map(|n| NodeView { id: node_id(n.id), kind: n.kind, label_ko: n.kind.spec().label_ko.into(), pos: at(n.x, n.z), hp: (n.hp / n.max_hp).clamp(0.0, 1.0) })
                .collect(),
            drops: self
                .drops
                .iter()
                .map(|dr| DropView { id: drop_id(dr.id), item: dr.kind, label_ko: dr.kind.spec().label_ko.into(), pos: at(dr.x, dr.z) })
                .collect(),
        }
    }

    /// Inputs for the body-status display. Perception numbers come from the
    /// caller because the world does not know what the recogniser saw.
    pub fn body_inputs(&self, vision_confidence: Option<f32>, frame_ms: Option<f32>) -> BodyInputs {
        BodyInputs {
            base: self.hand.base.kind,
            fingers: u8::from(self.hand.config.fingers),
            vision_confidence,
            frame_ms,
            energy_used_j: self.agent.energy_used_j,
            carried_kg: self.carried_kg(),
            grip_margin: self.agent.grip_margin,
            joint_strikes: self.agent.strikes,
            speed_factor: (self.agent.speed_mps / self.hand.base.max_speed_mps).clamp(0.0, 1.0),
            terrain_margin: self.agent.terrain_margin,
        }
    }

    pub fn body_status(&self, vision_confidence: Option<f32>, frame_ms: Option<f32>) -> BodyStatus {
        compute_body(&self.body_inputs(vision_confidence, frame_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hand_core::{BaseKind, FingerCount};

    fn world(fingers: FingerCount, base: BaseKind) -> World {
        let mut hand = HandConfig::preset(fingers);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, hand }).unwrap()
    }

    fn run(w: &mut World, max_s: f32) -> Vec<TimedEvent> {
        let mut all = Vec::new();
        let mut t = 0.0;
        while w.busy() && t < max_s {
            all.extend(w.tick());
            t += TICK_S;
        }
        all
    }

    fn ok_done(events: &[TimedEvent]) -> bool {
        events.iter().any(|e| matches!(&e.event, WorldEvent::ActionDone { ok: true, .. }))
    }

    /// Nearest tree that the given world can reach, as a node id.
    fn nearest_node(w: &World, kind: NodeKind) -> u32 {
        w.nodes
            .iter()
            .filter(|n| n.kind == kind)
            .min_by(|a, b| {
                let da = (a.x - w.agent.x).hypot(a.z - w.agent.z);
                let db = (b.x - w.agent.x).hypot(b.z - w.agent.z);
                da.total_cmp(&db)
            })
            .unwrap()
            .id
    }

    fn mine_nearest(w: &mut World, kind: NodeKind) -> Vec<TimedEvent> {
        let id = nearest_node(w, kind);
        let (x, z) = {
            let n = w.node(id).unwrap();
            (n.x, n.z)
        };
        w.command(Action::MoveTo { x, z, reach_m: 0.8 }).expect("route to the node");
        let mut ev = run(w, 600.0);
        w.command(Action::Mine { node: id }).expect("in reach");
        ev.extend(run(w, 600.0));
        ev
    }

    #[test]
    fn same_seed_gives_the_same_world() {
        let a = world(FingerCount::Five, BaseKind::Track);
        let b = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(a.nodes.len(), b.nodes.len());
        for (x, y) in a.nodes.iter().zip(&b.nodes) {
            assert_eq!((x.kind, x.x, x.z), (y.kind, y.x, y.z));
        }
    }

    #[test]
    fn world_has_every_resource_and_two_waypoints() {
        let w = world(FingerCount::Five, BaseKind::Track);
        for k in NodeKind::ALL {
            assert!(w.nodes.iter().filter(|n| n.kind == k).count() >= 2, "{k:?}");
        }
        assert_eq!(w.waypoints.len(), 2);
        let cells: HashSet<_> = w.nodes.iter().map(|n| w.terrain.cell_of(n.x, n.z).unwrap()).collect();
        assert_eq!(cells.len(), w.nodes.len(), "one node per cell");
    }

    #[test]
    fn robot_starts_at_the_base_camp_on_level_ground() {
        let w = world(FingerCount::Five, BaseKind::Wheel);
        let camp = w.waypoint_by_key("base_camp").unwrap();
        assert!((w.agent.x - camp.x).abs() < 1e-4 && (w.agent.z - camp.z).abs() < 1e-4);
        assert!(w.agent_ground_m().abs() < 1e-6);
    }

    #[test]
    fn mining_a_tree_takes_the_time_the_finger_count_implies() {
        for (fingers, expect) in [(FingerCount::Five, 3.0f32), (FingerCount::Two, 7.5)] {
            let mut w = world(fingers, BaseKind::Track);
            let id = nearest_node(&w, NodeKind::Tree);
            let n = w.node(id).unwrap().clone();
            // Teleport next to the tree so only mining time is measured.
            w.agent.x = n.x - 0.6;
            w.agent.z = n.z;
            w.agent.yaw = 0.0;
            let t0 = w.time_s;
            w.command(Action::Mine { node: id }).unwrap();
            let ev = run(&mut w, 60.0);
            assert!(ok_done(&ev), "{ev:?}");
            let dt = (w.time_s - t0) as f32;
            assert!((dt - expect).abs() < 0.25, "{fingers:?}: {dt} s vs {expect} s");
            assert!(w.node(id).is_none());
            assert_eq!(w.drops.len(), 3);
        }
    }

    #[test]
    fn out_of_reach_actions_fail_immediately() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let id = w.nodes[0].id;
        w.agent.x = 0.0;
        w.agent.z = 0.0;
        match w.command(Action::Mine { node: id }) {
            Err(ActionError::OutOfReach { distance_m, reach_m }) => assert!(distance_m > reach_m),
            other => panic!("{other:?}"),
        }
        assert!(matches!(w.command(Action::Mine { node: 9999 }), Err(ActionError::UnknownTarget)));
    }

    #[test]
    fn a_second_command_while_busy_is_refused() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.command(Action::Turn { yaw_deg: 170.0 }).unwrap();
        assert_eq!(w.command(Action::Turn { yaw_deg: 0.0 }), Err(ActionError::Busy));
        let ev = w.cancel();
        assert!(matches!(&ev[0].event, WorldEvent::ActionDone { ok: false, .. }));
        assert!(!w.busy());
    }

    #[test]
    fn driving_uses_the_energy_the_drive_model_predicts() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        let goal = (camp.x + 3.0, camp.z);
        let plan = w.plan_route(Goal::Cell(w.terrain.cell_of(goal.0, goal.1).unwrap())).unwrap();
        w.agent.yaw = 0.0;
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        let ev = run(&mut w, 120.0);
        assert!(ok_done(&ev));
        assert!((w.agent.x - cell_center(w.terrain.cell_of(goal.0, goal.1).unwrap().0, 0).0).abs() < 0.3);
        // Turning is extra, so the real energy is at least the plan and close to it.
        assert!(w.agent.energy_used_j >= plan.energy_j as f64 - 1.0);
        assert!(w.agent.energy_used_j <= plan.energy_j as f64 * 1.15 + 20.0, "{} vs {}", w.agent.energy_used_j, plan.energy_j);
    }

    #[test]
    fn five_finger_track_carries_a_log_to_the_warehouse() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let ev = mine_nearest(&mut w, NodeKind::Tree);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::NodeBroken { .. })));
        let drop = w.drops[0].clone();
        w.command(Action::MoveTo { x: drop.x, z: drop.z, reach_m: 0.5 }).unwrap();
        run(&mut w, 60.0);
        w.command(Action::PickUp { drop: drop.id }).unwrap();
        let ev = run(&mut w, 10.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })));
        assert_eq!(w.carried_count(ItemKind::Log), 1);
        let wh = w.waypoint_by_key("warehouse").unwrap().clone();
        w.command(Action::MoveTo { x: wh.x, z: wh.z, reach_m: 1.0 }).unwrap();
        let ev = run(&mut w, 600.0);
        assert!(ok_done(&ev));
        w.command(Action::Deliver { waypoint: wh.id, item: Some(ItemKind::Log) }).unwrap();
        run(&mut w, 10.0);
        assert_eq!(w.carried_count(ItemKind::Log), 0);
        let stock = &w.waypoint(wh.id).unwrap().stock;
        assert_eq!(stock, &vec![(ItemKind::Log, 1)]);
    }

    #[test]
    fn delivery_without_cargo_is_refused() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(w.command(Action::Deliver { waypoint: 1, item: None }), Err(ActionError::NothingToDeliver));
    }

    #[test]
    fn the_curb_wall_costs_a_wheel_a_long_detour_but_a_foot_crosses_it() {
        let wheel = world(FingerCount::Five, BaseKind::Wheel);
        let foot = world(FingerCount::Five, BaseKind::HumanFoot);
        let wh = wheel.waypoint_by_key("warehouse").unwrap();
        let goal = Goal::Within { x: wh.x, z: wh.z, reach_m: 1.0 };
        let pw = wheel.plan_route(goal).expect("wheel reaches the warehouse through the gap");
        let pf = foot.plan_route(goal).expect("foot route");
        assert!(pw.distance_m > pf.distance_m + 8.0, "wheel {} m, foot {} m", pw.distance_m, pf.distance_m);
        assert!(pf.edges.iter().any(|e| e.step_m > 0.0));
        assert!(pw.edges.iter().all(|e| e.step_m == 0.0));
    }

    #[test]
    fn grasp_table_reflects_the_hand() {
        let five = world(FingerCount::Five, BaseKind::Track);
        let two = world(FingerCount::Two, BaseKind::Track);
        for k in ItemKind::ALL {
            let a = five.grasp_info(k);
            let b = two.grasp_info(k);
            assert!(a.margin.is_finite() && b.margin.is_finite());
        }
        // The lightest item must be easier to hold than the heaviest one.
        let light = five.grasp_info(ItemKind::Coal).margin;
        let heavy = five.grasp_info(ItemKind::Stone).margin;
        assert!(light >= heavy * 0.99, "coal {light} vs stone {heavy}");
    }

    #[test]
    fn item_catalogue_spans_what_the_hands_can_hold() {
        // The item sizes and masses are assumptions chosen so that hands of
        // different finger counts give different answers; check they still do.
        let holdable = |f: FingerCount| {
            let w = world(f, BaseKind::Track);
            ItemKind::ALL.into_iter().filter(|&k| w.grasp_info(k).stable).count()
        };
        let (two, five) = (holdable(FingerCount::Two), holdable(FingerCount::Five));
        assert!(five >= 3, "a five-finger hand should hold at least three of four items, holds {five}");
        assert!(two < five, "a two-finger hand should hold fewer items ({two} vs {five})");
        assert!(two >= 1, "the lightest item should be holdable even with two fingers");
    }

    #[test]
    fn an_unstable_grasp_blocks_the_pickup() {
        let mut w = world(FingerCount::Two, BaseKind::Track);
        // Force the situation: whichever item the hand cannot hold.
        let bad = ItemKind::ALL
            .into_iter()
            .find(|&k| !w.grasp_info(k).stable)
            .expect("the two-finger hand cannot hold every item (see item_catalogue_spans_what_the_hands_can_hold)");
        w.drops.push(DropItem { id: 500, kind: bad, x: w.agent.x + 0.4, z: w.agent.z });
        match w.command(Action::PickUp { drop: 500 }) {
            Err(ActionError::GraspUnstable { margin }) => assert!(margin < 1.0 + 1e-3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn capacity_limit_stops_the_pickup() {
        let mut w = world(FingerCount::Five, BaseKind::Mecanum);
        let item = ItemKind::ALL.into_iter().find(|&k| w.grasp_info(k).stable).expect("some item is holdable");
        let cap = w.capacity_kg();
        add_stock(&mut w.agent.carried, item, (cap / item.spec().mass_kg) as u32);
        w.drops.push(DropItem { id: 501, kind: item, x: w.agent.x + 0.4, z: w.agent.z });
        assert!(matches!(w.command(Action::PickUp { drop: 501 }), Err(ActionError::CapacityExceeded { .. })));
    }

    #[test]
    fn mining_counts_joint_strikes_and_drains_the_battery() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let before = w.battery_ratio();
        mine_nearest(&mut w, NodeKind::Tree);
        assert!(w.agent.strikes >= 3);
        assert!(w.battery_ratio() < before);
        let body = w.body_status(Some(0.8), Some(12.0));
        assert_eq!(body.parts.len(), 7);
    }

    #[test]
    fn a_depleted_battery_stops_everything() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.agent.energy_used_j = w.battery_capacity_j() + 1.0;
        assert_eq!(w.command(Action::Turn { yaw_deg: 10.0 }), Err(ActionError::BatteryDepleted));
        // And mid-action:
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.command(Action::Turn { yaw_deg: 170.0 }).unwrap();
        w.agent.energy_used_j = w.battery_capacity_j() + 1.0;
        let ev = w.tick();
        assert!(ev.iter().any(|e| e.event == WorldEvent::BatteryDepleted));
        assert!(!w.busy());
    }

    #[test]
    fn snapshot_uses_prefixed_string_ids_and_serialises() {
        let w = world(FingerCount::Five, BaseKind::Track);
        let s = w.snapshot();
        assert!(s.nodes.iter().all(|n| n.id.starts_with('n')));
        assert!(s.waypoints.iter().all(|p| p.id.starts_with('w')));
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"base\":\"track\""));
        let g = w.terrain_grid();
        assert_eq!(g.heights_cm.len(), g.cells_w * g.cells_d);
    }

    #[test]
    fn facing_a_point_turns_the_agent_toward_it() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        w.agent.yaw = 0.0;
        w.command(Action::Face { x: w.agent.x, z: w.agent.z + 5.0 }).unwrap();
        run(&mut w, 10.0);
        assert!((w.agent.yaw - std::f32::consts::FRAC_PI_2).abs() < 0.05);
    }
}
