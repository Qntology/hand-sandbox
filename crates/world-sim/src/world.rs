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

use std::sync::Mutex;
use std::collections::{BTreeMap, HashSet};

use hand_core::balance::{BalanceReport, Body, DriveLimits, FeedForward, Posture, Source, GRIP_SAFETY};
use hand_core::dynamics::{self, BodyPose, DynCfg, DynDigest, Rollout};
use hand_core::{compute_body, BalanceSummary, BodyInputs, BodyLimits, BodyPlan, BodyStatus, ConfigError, Hand, HandConfig};
use serde::{Deserialize, Serialize};

use crate::kinds::{Ground, ItemKind, NodeKind, MIN_PIECE};
use crate::nav::{self, Cell, EdgePlan, Goal};
use crate::odometry::{OdomCfg, Odometry};
use crate::render;
use crate::rng::Rng;
use crate::snapshot::*;
use crate::terrain::{cell_center, FlatZone, RuggedLayout, Terrain, TerrainKind, CELL_M};

mod acts;
mod eye;
pub use acts::{camera_of, material_of, ActPlay, PushState, WalkState, GET_UP_S, PUSH_START_SLACK_M, SIDE_SHUFFLE_MAX_M};
pub use eye::{chassis_accel_rms, exposure_s, profile, roughness_gd, sensor_timing, strike_peak_mps2, strike_scale, EyeState, HeadView, SensorTiming, Shake, SCENE_LUX};

/// Length of one simulation tick, seconds.
pub const TICK_S: f32 = 0.1;
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
/// Heading error above which a non-holonomic base turns in place, radians.
const TURN_IN_PLACE_RAD: f32 = 0.6;
/// Heading error at which the in-place turn stops, radians.
const TURN_DONE_RAD: f32 = 0.2;
/// Heading error accepted before manipulation starts, radians.
const FACE_DONE_RAD: f32 = 0.08;
/// A lift shared by both arms takes this much longer (est).
pub const TWO_ARM_LIFT_FACTOR: f32 = 1.5;
/// Time to reach cruise speed from standing, seconds (est); the first part of every move is an `accel` balance state.
pub const ACCEL_TIME_S: f32 = 0.5;
/// A balance state that lasts is recorded again every this many ticks.
const BALANCE_RESAMPLE_TICKS: u64 = 50;
/// Time to set carried items down on the ground or a waypoint's pile, seconds (assumption, as [`DELIVER_TIME_S`]).
pub const DROP_TIME_S: f32 = 1.0;
/// Items set down on the ground land this far in front of the robot, metres (est: forearm reach).
pub const DROP_AHEAD_M: f32 = 0.45;
/// Amounts within this of a whole number count as whole.
pub const AMOUNT_EPS: f32 = 1e-3;
/// The base camp's parcel lies this far ahead of the robot's start along the way
/// to the warehouse, and this far to its right, metres (a scenario: seen in
/// front at the start, picked from the side in passing).
pub const CAMP_PARCEL_AHEAD_M: f32 = 2.0;
pub const CAMP_PARCEL_RIGHT_M: f32 = 0.30;
/// Parcels per cell of the map by default (about one per 175 m²).
pub const PARCELS_PER_CELL: f32 = 1.0 / 700.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    pub seed: u64,
    /// Cells per side (square world). Each cell is 0.5 m.
    pub cells: usize,
    pub hand: HandConfig,
    /// Part presets of the robot's body (`hand_core::parts`).
    #[serde(default)]
    pub body: BodyPlan,
    /// The landscape: the open map, or the rugged one with sea, cliffs and ridges.
    #[serde(default)]
    pub terrain: TerrainKind,
    /// The robot's dead reckoning (`odometry`): on, its believed pose drifts as a real one's does.
    #[serde(default)]
    pub odometry: OdomCfg,
    /// Parcels lying in the world: `None` the default (scattered over the map
    /// and one in front of the base camp), `Some(0)` none.
    #[serde(default)]
    pub parcels: Option<usize>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig { seed: 7, cells: 64, hand: HandConfig::preset(hand_core::FingerCount::Five), body: BodyPlan::default(), terrain: TerrainKind::Open, odometry: OdomCfg::default(), parcels: None }
    }
}

/// A balance state the robot entered (or stayed in long enough to sample again).
#[derive(Clone, Debug)]
pub struct BalanceEvent {
    pub tick: u64,
    pub time_s: f64,
    /// Terrain and phase bucket, such as `flat|mine` or `up_hi|move`.
    pub state: String,
    pub report: BalanceReport,
    /// The joint dynamics of the change into this state (`None` for a re-sample).
    pub dynamics: Option<DynDigest>,
    /// The posture itself (every cause acting, for the composite N:N phases).
    pub posture: Posture,
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

/// An item lying on the ground. `amount` is how much of one item the piece
/// is (1 = a whole stone; a stone broken in half leaves two pieces of 0.5).
#[derive(Clone, Debug)]
pub struct DropItem {
    pub id: u32,
    pub kind: ItemKind,
    pub x: f32,
    pub z: f32,
    pub amount: f32,
}

impl DropItem {
    /// A whole item.
    pub fn whole(id: u32, kind: ItemKind, x: f32, z: f32) -> DropItem {
        DropItem { id, kind, x, z, amount: 1.0 }
    }

    pub fn mass_kg(&self) -> f32 {
        self.kind.spec().mass_kg * self.amount
    }
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub id: u32,
    pub key: String,
    pub label_ko: String,
    pub x: f32,
    pub z: f32,
    /// What has been put down here, in items (fractions are broken pieces).
    pub stock: Vec<(ItemKind, f32)>,
}

impl Waypoint {
    pub fn stock_of(&self, item: ItemKind) -> f32 {
        self.stock.iter().filter(|(k, _)| *k == item).map(|(_, n)| *n).sum()
    }
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    /// Items in the cargo bay, in items (fractions are broken pieces).
    pub carried: Vec<(ItemKind, f32)>,
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
    /// Pick an item up. The hand goes where the robot believes it is (`aim`,
    /// in its believed frame; `None`: where it lies, seen as it reaches), in
    /// front after facing it, or beside the right foot without turning
    /// (`side`: the camera does not see it there). `memory_sigma_m` is the
    /// 1σ the robot gives that memory (0 when it sees the item).
    PickUp {
        drop: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aim: Option<(f32, f32)>,
        #[serde(default, skip_serializing_if = "is_false")]
        side: bool,
        #[serde(default, skip_serializing_if = "is_zero")]
        memory_sigma_m: f32,
    },
    /// Put carried items into a waypoint's stock; `item` limits it to one kind.
    Deliver { waypoint: u32, item: Option<ItemKind> },
    /// Set carried items down here: into the pile of a waypoint within
    /// [`DELIVER_REACH_M`], else on the ground in front of the robot. `item`
    /// limits it to one kind, `amount` to that many items (all when `None`).
    Drop { item: Option<ItemKind>, amount: Option<f32> },
    /// Break a divisible piece lying on the ground: a new piece with
    /// `portion` of it appears beside it, the rest stays.
    Split { drop: u32, portion: f32 },
    /// Push an item lying on the ground with the bumper (wheeled and tracked
    /// bases) to within reach of a point; the robot must stand lined up behind
    /// it (`World::push_route`).
    Push { drop: u32, x: f32, z: f32 },
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_zero(x: &f32) -> bool {
    *x == 0.0
}

impl Action {
    /// A pick-up of an item the robot sees, in front of it (the pick before v0.7).
    pub fn pick(drop: u32) -> Action {
        Action::PickUp { drop, aim: None, side: false, memory_sigma_m: 0.0 }
    }

    pub fn label(&self) -> String {
        match self {
            Action::MoveTo { x, z, reach_m } => format!("move_to({x:.1},{z:.1},reach={reach_m:.1})"),
            Action::Turn { yaw_deg } => format!("turn({yaw_deg:.0})"),
            Action::Face { x, z } => format!("face({x:.1},{z:.1})"),
            Action::Mine { node } => format!("mine({})", node_id(*node)),
            Action::PickUp { drop, side, .. } => format!("{}({})", if *side { "pick_up_side" } else { "pick_up" }, drop_id(*drop)),
            Action::Deliver { waypoint, item } => match item {
                Some(i) => format!("deliver({},{})", waypoint_id(*waypoint), i.id()),
                None => format!("deliver({})", waypoint_id(*waypoint)),
            },
            Action::Drop { item, amount } => match (item, amount) {
                (Some(i), Some(a)) => format!("drop({}×{a:.2})", i.id()),
                (Some(i), None) => format!("drop({})", i.id()),
                (None, _) => "drop(all)".to_string(),
            },
            Action::Split { drop, portion } => format!("split({},{portion:.2})", drop_id(*drop)),
            Action::Push { drop, x, z } => format!("push({}→{x:.1},{z:.1})", drop_id(*drop)),
        }
    }
}

/// An amount as people say it: `2`, `0.5`, `1.25`.
pub fn amount_text(a: f32) -> String {
    if (a - a.round()).abs() < AMOUNT_EPS {
        format!("{}", a.round() as i64)
    } else {
        let s = format!("{a:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
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
    /// The arms (or the balance) cannot lift the item: a joint over its peak, or a negative margin.
    LiftOverload { limiting: String, margin: f32 },
    NothingToDrop,
    /// The item cannot be broken by hand (a log).
    NotDivisible,
    /// The piece, or what would be left of it, is below [`MIN_PIECE`].
    PieceTooSmall { amount: f32 },
    /// The body cannot bring its palm down to the item (`hand_core::act::reach`).
    CannotReach { lowest_m: f32, need_m: f32, reason: String },
    /// This base cannot push (legs), or the item cannot be pushed there.
    CannotPush { reason: String },
    /// The robot lies on the ground after a fall and is getting up.
    Fallen,
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
            ActionError::LiftOverload { .. } => "lift_overload",
            ActionError::NothingToDrop => "nothing_to_drop",
            ActionError::NotDivisible => "not_divisible",
            ActionError::PieceTooSmall { .. } => "piece_too_small",
            ActionError::CannotReach { .. } => "reach_ground",
            ActionError::CannotPush { .. } => "push_impossible",
            ActionError::Fallen => "fallen",
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
    /// `amount` of one item was lifted into the bay (`mass_kg` is that piece's mass).
    ItemPicked { drop: String, item: ItemKind, mass_kg: f32, grip_margin: f32, amount: f32 },
    Delivered { waypoint: String, item: ItemKind, amount: f32 },
    /// Carried items set down: on a waypoint's pile (`waypoint`) or on the ground as pieces (`drops`).
    ItemDropped { item: ItemKind, amount: f32, waypoint: Option<String>, drops: Vec<String> },
    /// A piece broken off `drop`: the new piece `piece` holds `amount` items.
    ItemSplit { drop: String, piece: String, item: ItemKind, amount: f32, left: f32 },
    BatteryDepleted,
    /// A moment of an action's physics (`hand_core::act`): a contact, a lift-off,
    /// a slip, an impact, a heel strike, a fall. `parts` are the focus views it concerns.
    Act { act: String, kind: String, label: String, ok: bool, value: Option<f32>, parts: Vec<String> },
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
    /// Seconds since the move started (the first part is acceleration).
    elapsed_s: f32,
}

#[derive(Clone, Debug)]
enum WorkKind {
    Mine { node: u32 },
    /// `face`: where the robot turns before it bends (the item where it believes it is); `None` for a reach to the side.
    Pick { drop: u32, two_arms: bool, face: Option<(f32, f32)> },
    Deliver { waypoint: u32, item: Option<ItemKind> },
    Drop { item: Option<ItemKind>, amount: Option<f32>, waypoint: Option<u32> },
    Split { drop: u32, portion: f32 },
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
    Push(PushState),
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
    /// The robot's body built from the part presets.
    pub body: Body,
    limits: Mutex<Option<(u32, DriveLimits)>>,
    /// Balance of the posture held right now, keyed `source|terrain|phase`.
    pub balance: Option<(String, BalanceReport)>,
    balance_since: u64,
    balance_events: Vec<BalanceEvent>,
    /// Feed-forward gains per balance source, set by the runtime from the SDS.
    pub ff_gains: BTreeMap<Source, f32>,
    pub ff_enabled: bool,
    /// Heat state per joint, `(τ/τ_cont)²` low-passed with the actuator's time constant.
    pub heat: BTreeMap<String, f32>,
    /// Life used per joint (Miner damage over the reducer's rating, `hand_core::wear`; 1 = worn out).
    pub wear: BTreeMap<String, f32>,
    /// Joint dynamics of the last posture change (`hand_core::dynamics`): the
    /// presets' masses and actuators moving the joints, for the third-person view.
    pub motion: Option<Motion>,
    last_posture: Posture,
    /// Where the rugged features went (`None` on the open map).
    pub layout: Option<RuggedLayout>,
    /// The action being played out (`hand_core::act`), and the last one, kept for the focus views.
    pub act: Option<ActPlay>,
    pub last_act: Option<ActPlay>,
    /// The actions' own draws (friction, aim, slips), apart from the world's stream.
    act_rng: Rng,
    /// The robot lies on the ground after a fall until this time.
    pub fallen_until: Option<f64>,
    /// A legged robot's steps along the current move.
    pub walk: WalkState,
    /// The robot's dead reckoning: the pose it believes it has.
    pub odom: Odometry,
    /// Share of this tick's distance the wheels counted without travelling it (their slip).
    odom_over: f32,
    /// What the head camera tracks between ticks (`eye`).
    pub eye: EyeState,
    /// Side reaches planned so far, by item, piece (thousandths) and cargo (tenths of a kg).
    side_cache: Mutex<std::collections::HashMap<(ItemKind, i32, i32), Option<(f32, f32)>>>,
}

impl Clone for World {
    /// A copy that runs on its own: the integrated test replays a recorded
    /// start state in it while the live world waits.
    fn clone(&self) -> World {
        World {
            cfg: self.cfg.clone(),
            hand: self.hand.clone(),
            terrain: self.terrain.clone(),
            nodes: self.nodes.clone(),
            drops: self.drops.clone(),
            waypoints: self.waypoints.clone(),
            agent: self.agent.clone(),
            tick: self.tick,
            time_s: self.time_s,
            grasps: self.grasps.clone(),
            active: self.active.clone(),
            rng: self.rng.clone(),
            next_drop: self.next_drop,
            body: self.body.clone(),
            limits: Mutex::new(self.limits.lock().unwrap_or_else(|e| e.into_inner()).clone()),
            balance: self.balance.clone(),
            balance_since: self.balance_since,
            balance_events: self.balance_events.clone(),
            ff_gains: self.ff_gains.clone(),
            ff_enabled: self.ff_enabled,
            heat: self.heat.clone(),
            wear: self.wear.clone(),
            motion: self.motion.clone(),
            last_posture: self.last_posture,
            layout: self.layout.clone(),
            act: self.act.clone(),
            last_act: self.last_act.clone(),
            act_rng: self.act_rng.clone(),
            fallen_until: self.fallen_until,
            walk: self.walk.clone(),
            odom: self.odom.clone(),
            odom_over: self.odom_over,
            eye: self.eye.clone(),
            side_cache: Mutex::new(self.side_cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
        }
    }
}

/// A simulated posture change and when it started.
#[derive(Clone, Debug)]
pub struct Motion {
    pub rollout: Rollout,
    pub posture: Posture,
    pub start_s: f64,
    pub state: String,
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

fn add_stock(list: &mut Vec<(ItemKind, f32)>, kind: ItemKind, n: f32) {
    if let Some(e) = list.iter_mut().find(|(k, _)| *k == kind) {
        e.1 += n;
        // Two halves make a whole: keep sums free of float dust.
        if (e.1 - e.1.round()).abs() < AMOUNT_EPS {
            e.1 = e.1.round();
        }
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

        // Compacted gravel pads under the waypoints.
        for (px, pz) in [base_pos, wh_pos] {
            for cz in 0..cells {
                for cx in 0..cells {
                    let (x, z) = cell_center(cx, cz);
                    if ((x - px).powi(2) + (z - pz).powi(2)).sqrt() <= 2.0 {
                        terrain.set_ground(cx, cz, Ground::Gravel);
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

        // The rugged map: sea, cliffs and ridges away from the way a wheel
        // takes between the waypoints (through the gap in the curb wall).
        let layout = (cfg.terrain == TerrainKind::Rugged).then(|| {
            let gap = (wall_x, (gap_from as f32 + 1.0) * CELL_M);
            let clear: Vec<FlatZone> = vec![(base_pos.0, base_pos.1, 3.5), (wh_pos.0, wh_pos.1, 3.5)];
            terrain.stamp_rugged(cfg.seed, &clear, &[base_pos, (base_pos.0, gap.1), gap, wh_pos], 2.2)
        });

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
                    || terrain.is_water(cx, cz)
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
        // On the rugged map some nodes end up on mesas or behind water: seen
        // from afar, out of reach. Every kind keeps two a wheel can get to, so
        // a quest has a way to succeed (a decoy on a mesa is the hard part).
        if layout.is_some() {
            let wheel = hand_core::BaseKind::Wheel.spec();
            let start = terrain.cell_of(base_pos.0, base_pos.1).unwrap_or((0, 0));
            let taken: HashSet<Cell> = occupied.clone();
            let times = nav::travel_times(&wheel, &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let reachable = |cx: usize, cz: usize| -> bool {
                (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| {
                    let (nx, nz) = (cx as i32 + dx, cz as i32 + dz);
                    terrain.in_bounds(nx, nz) && times[terrain.idx(nx as usize, nz as usize)].is_finite()
                }))
            };
            for kind in NodeKind::ALL {
                let mut ok = nodes.iter().filter(|n| n.kind == kind).filter(|n| terrain.cell_of(n.x, n.z).map_or(false, |(cx, cz)| reachable(cx, cz))).count();
                let mut tries = 0;
                while ok < 2 && tries < 4000 {
                    tries += 1;
                    let (cx, cz) = (rng.below(cells as u32) as usize, rng.below(cells as u32) as usize);
                    let (x, z) = cell_center(cx, cz);
                    let i = terrain.idx(cx, cz);
                    if !times[i].is_finite() || occupied.contains(&(cx, cz)) || in_zone(x, z) || cx.abs_diff(wall_cx) <= 1 || terrain.curb_m(cx, cz) > 0.0 {
                        continue;
                    }
                    if kind == NodeKind::Tree && terrain.height_cm(cx, cz) >= 70 {
                        continue;
                    }
                    occupied.insert((cx, cz));
                    let hp = kind.break_time_s();
                    nodes.push(Node { id: next_node, kind, x, z, hp, max_hp: hp });
                    next_node += 1;
                    ok += 1;
                }
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

        // Parcels: light boxes lying about, found by looking (no node yields them).
        // Their own random stream, so the nodes and the game's draws stay as they were.
        let mut drops = Vec::new();
        let mut next_drop = 1u32;
        let n_parcels = cfg.parcels.unwrap_or(((cells * cells) as f32 * PARCELS_PER_CELL).round().max(3.0) as usize);
        if n_parcels > 0 {
            let mut prng = Rng::new(cfg.seed ^ 0x9A2C_E1B0_77D3_4F15);
            // One in front of the base camp, a little right of the way to the warehouse.
            let (s, c) = yaw.sin_cos();
            let (px, pz) = (base_pos.0 + CAMP_PARCEL_AHEAD_M * c - CAMP_PARCEL_RIGHT_M * s, base_pos.1 + CAMP_PARCEL_AHEAD_M * s + CAMP_PARCEL_RIGHT_M * c);
            drops.push(DropItem::whole(next_drop, ItemKind::Parcel, px, pz));
            next_drop += 1;
            let wheel = hand_core::BaseKind::Wheel.spec();
            let start = terrain.cell_of(base_pos.0, base_pos.1).unwrap_or((0, 0));
            let taken: HashSet<Cell> = occupied.clone();
            let times = nav::travel_times(&wheel, &terrain, &|x, z| taken.contains(&(x, z)), start, f32::INFINITY);
            let mut tries = 0;
            while drops.len() < n_parcels && tries < 4000 {
                tries += 1;
                let (cx, cz) = (prng.below(cells as u32) as usize, prng.below(cells as u32) as usize);
                let (x, z) = cell_center(cx, cz);
                // On land a wheel can get to, off the waypoints' pads and the curb wall, one per cell.
                if !times[terrain.idx(cx, cz)].is_finite() || occupied.contains(&(cx, cz)) || in_zone(x, z) || cx.abs_diff(wall_cx) <= 1 || terrain.curb_m(cx, cz) > 0.0 {
                    continue;
                }
                occupied.insert((cx, cz));
                let (jx, jz) = (prng.range_f32(-0.12, 0.12), prng.range_f32(-0.12, 0.12));
                drops.push(DropItem::whole(next_drop, ItemKind::Parcel, x + jx, z + jz));
                next_drop += 1;
            }
        }

        let grasps = ItemKind::ALL.iter().map(|&k| top_grasp_info(&hand, k, 1.0)).collect();

        let (cfg_body, base_kind, fingers, cfg_seed) = (cfg.body, cfg.hand.base, cfg.hand.fingers, cfg.seed);
        let odom = Odometry::new(cfg.odometry, cfg_seed, base_kind == hand_core::BaseKind::HumanFoot, (agent.x, agent.z, agent.yaw));
        Ok(World {
            cfg,
            hand,
            terrain,
            nodes,
            drops,
            waypoints,
            agent,
            tick: 0,
            time_s: 0.0,
            grasps,
            active: None,
            rng,
            next_drop,
            body: Body::new(&cfg_body, base_kind, fingers),
            limits: Mutex::new(None),
            balance: None,
            balance_since: 0,
            balance_events: Vec::new(),
            ff_gains: BTreeMap::new(),
            ff_enabled: true,
            heat: BTreeMap::new(),
            wear: BTreeMap::new(),
            motion: None,
            last_posture: Posture::quiet(0.0),
            layout,
            act: None,
            last_act: None,
            act_rng: Rng::new(cfg_seed ^ 0x51D3_A6C2_7F10_88E5),
            fallen_until: None,
            walk: WalkState::default(),
            odom,
            odom_over: 0.0,
            eye: EyeState::default(),
            side_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// The world as the seed made it (terrain, nodes, nothing on the ground,
    /// empty piles, the robot at the base camp), with the robot's body as it
    /// is now: the battery charge, joint heat and wear stay. Nothing changes
    /// if the world cannot be rebuilt.
    pub fn reset_world_keep_body(&mut self) -> Result<(), ConfigError> {
        let mut fresh = World::new(self.cfg.clone())?;
        fresh.agent.energy_used_j = self.agent.energy_used_j;
        fresh.agent.strikes = self.agent.strikes;
        fresh.agent.strike_acc = self.agent.strike_acc;
        fresh.heat = std::mem::take(&mut self.heat);
        fresh.wear = std::mem::take(&mut self.wear);
        fresh.ff_gains = std::mem::take(&mut self.ff_gains);
        fresh.ff_enabled = self.ff_enabled;
        // The clock keeps running: logs and records stay in order.
        fresh.tick = self.tick;
        fresh.time_s = self.time_s;
        *self = fresh;
        Ok(())
    }

    /// The body back to rested: full battery, cold joints, no wear, the last
    /// grasp and terrain readings forgotten. The robot stays where it is with
    /// what it carries, and a running action goes on.
    pub fn reset_body(&mut self) {
        self.agent.energy_used_j = 0.0;
        self.agent.strikes = 0;
        self.agent.strike_acc = 0.0;
        self.agent.grip_margin = None;
        self.agent.terrain_margin = None;
        self.heat.clear();
        self.wear.clear();
        self.balance = None;
        self.motion = None;
        self.last_posture = Posture::quiet(self.carried_kg());
        self.fallen_until = None;
    }

    // ----- lookups -----

    pub fn grasp_info(&self, item: ItemKind) -> &GraspInfo {
        self.grasps.iter().find(|g| g.item == item).expect("every item has a grasp entry")
    }

    pub fn grasp_table(&self) -> &[GraspInfo] {
        &self.grasps
    }

    // ----- the robot's own sense of where it is (`odometry`) -----

    /// Where the robot believes it is: x, z, yaw (its dead reckoning).
    pub fn believed_pose(&self) -> (f32, f32, f32) {
        self.odom.pose()
    }

    fn true_pose(&self) -> (f32, f32, f32) {
        (self.agent.x, self.agent.z, self.agent.yaw)
    }

    /// A point of the world in the robot's believed frame: where the robot
    /// puts something its camera or its hand finds there.
    pub fn belief_of(&self, x: f32, z: f32) -> (f32, f32) {
        self.odom.belief_of(self.true_pose(), (x, z))
    }

    /// A point of the believed frame in the world: where the robot's body
    /// goes when it goes where it believes that point is.
    pub fn true_of(&self, x: f32, z: f32) -> (f32, f32) {
        self.odom.true_of(self.true_pose(), (x, z))
    }

    /// A believed heading as the world's.
    pub fn true_yaw_of(&self, yaw: f32) -> f32 {
        self.odom.true_yaw(self.agent.yaw, yaw)
    }

    /// How far the robot's belief is off its true position, metres. For the
    /// record and the benchmark only: nothing that decides may read it.
    pub fn belief_error_m(&self) -> f32 {
        (self.odom.x - self.agent.x).hypot(self.odom.z - self.agent.z)
    }

    /// The robot moves its believed position by what a landmark of the map
    /// showed it (a waypoint seen where the map has it).
    pub fn relocalize(&mut self, dx: f32, dz: f32) {
        self.odom.correct(dx, dz);
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
        self.agent.carried.iter().map(|(k, n)| k.spec().mass_kg * *n).sum()
    }

    /// Whole items of a kind in the bay (a half stone does not count).
    pub fn carried_count(&self, item: ItemKind) -> u32 {
        (self.carried_amount(item) + AMOUNT_EPS).floor() as u32
    }

    /// Items of a kind in the bay, broken pieces included.
    pub fn carried_amount(&self, item: ItemKind) -> f32 {
        self.agent.carried.iter().filter(|(k, _)| *k == item).map(|e| e.1).sum()
    }

    /// What one hand's grasp test says about a piece of `amount` items:
    /// smaller pieces of a brittle item weigh less and are smaller.
    pub fn grasp_for(&self, item: ItemKind, amount: f32) -> GraspInfo {
        if (amount - 1.0).abs() < AMOUNT_EPS {
            return self.grasp_info(item).clone();
        }
        top_grasp_info(&self.hand, item, amount)
    }

    /// Friction capacity of one hand for a piece, newtons.
    pub fn grip_capacity_for(&self, item: ItemKind, amount: f32) -> f32 {
        let g = self.grasp_for(item, amount);
        g.margin * item.spec().mass_kg * amount * hand_core::balance::G * GRIP_SAFETY
    }

    /// Time to break a piece of `amount` items in two with this hand (est:
    /// the fracture surface grows as the piece's cross-section, `amount^(2/3)`;
    /// the hand strikes at the mining rate).
    pub fn split_time_s(&self, item: ItemKind, amount: f32) -> f32 {
        item.split_time_s() * amount.max(MIN_PIECE).powf(2.0 / 3.0) / self.mining_rate().max(0.05)
    }

    /// The waypoint whose pile a drop here would go on.
    pub fn waypoint_in_reach(&self) -> Option<u32> {
        self.waypoints
            .iter()
            .map(|w| (w.id, ((w.x - self.agent.x).powi(2) + (w.z - self.agent.z).powi(2)).sqrt()))
            .filter(|(_, d)| *d <= DELIVER_REACH_M)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)
    }

    /// Cargo limit: the base's own limit, lowered by what the legs or the
    /// chassis can carry with this body (`Body::drive_limits`).
    pub fn capacity_kg(&self) -> f32 {
        self.limits().capacity_kg
    }

    /// Battery energy of the chest preset.
    pub fn battery_capacity_j(&self) -> f64 {
        self.body.battery_j()
    }

    /// Speed, slope, step and cargo limits for the current load (cached per load).
    pub fn limits(&self) -> DriveLimits {
        let key = self.carried_kg().to_bits();
        let mut cache = self.limits.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((k, l)) = cache.as_ref() {
            if *k == key {
                return l.clone();
            }
        }
        let l = self.body.drive_limits(&self.hand.base, self.carried_kg());
        *cache = Some((key, l.clone()));
        l
    }

    /// The base spec the routes and the drive use: the mechanism's spec with the
    /// body's limits applied, the drive power scaled with the mass, and the
    /// chassis force over the loaded mass (each ground's climb limit).
    pub fn drive_spec(&self) -> hand_core::BaseSpec {
        self.drive_spec_for(self.carried_kg())
    }

    /// The drive spec with `cargo_kg` aboard (the cargo decision asks before loading).
    pub fn drive_spec_for(&self, cargo_kg: f32) -> hand_core::BaseSpec {
        let l = if (cargo_kg - self.carried_kg()).abs() < 1e-6 { self.limits() } else { self.body.drive_limits(&self.hand.base, cargo_kg) };
        let mass = self.body.mass_kg() + cargo_kg;
        hand_core::BaseSpec {
            max_speed_mps: l.max_speed_mps,
            max_slope_deg: l.max_slope_deg,
            max_step_m: l.max_step_m,
            power_w: self.hand.base.power_w * l.power_scale,
            drive_accel_mps2: (!self.body.legged).then(|| self.body.chassis.drive_force_n / mass),
            mass_kg: Some(mass),
            walk_com_z_m: self.body.legged.then(|| hand_core::act::gait::walking_com_z(&self.body)),
            walk_foot_m: self.body.legged.then_some(self.body.leg.foot_len_m),
            ..self.hand.base
        }
    }

    /// The ground of a cell (gravel, as the pads, outside the map).
    pub fn ground_at(&self, cell: Cell) -> Ground {
        if cell.0 < self.terrain.w && cell.1 < self.terrain.d {
            self.terrain.ground_at(cell.0, cell.1)
        } else {
            Ground::Gravel
        }
    }

    /// The ground under the robot.
    pub fn ground_here(&self) -> Ground {
        self.ground_at(self.agent_cell())
    }

    /// Time to reach cruise speed: at least [`ACCEL_TIME_S`], longer when the
    /// body's acceleration limit (ZMP inside the support) is lower.
    pub fn accel_time_s(&self) -> f32 {
        let l = self.limits();
        (l.max_speed_mps / l.accel_limit_mps2.max(0.05)).max(ACCEL_TIME_S)
    }

    /// The drive planner's start (`dir > 0`) or stop (`dir < 0`) acceleration on
    /// `slope_deg` and `ground` with the current cargo (`Body::planned_accel`).
    pub fn planned_accel(&self, slope_deg: f32, ground: Ground, dir: f32) -> f32 {
        self.body.planned_accel_within(&self.limits(), slope_deg, self.carried_kg(), ground.surface(), dir, ACCEL_TIME_S)
    }

    /// Distance to stop from cruise speed with the planned deceleration, metres.
    pub fn braking_distance_m(&self, slope_deg: f32, ground: Ground) -> f32 {
        let v = self.drive_spec().max_speed_mps;
        v * v / (2.0 * self.planned_accel(slope_deg, ground, -1.0))
    }

    /// Path left on a move, metres.
    fn remaining_m(&self, m: &MoveState) -> f32 {
        let rest: f32 = m.edges.iter().skip(m.idx).map(|e| e.horizontal_m).sum();
        let done = m.edges.get(m.idx).map_or(0.0, |e| e.horizontal_m * (m.progress_s / e.move_time_s.max(1e-6)).clamp(0.0, 1.0));
        rest - done
    }

    fn ff_for(&self, s: Source) -> FeedForward {
        FeedForward { enabled: self.ff_enabled, gain: self.ff_gains.get(&s).copied().unwrap_or(1.0) }
    }

    /// Friction capacity of one hand for an item, newtons (from the grasp table).
    pub fn grip_capacity_n(&self, item: ItemKind) -> f32 {
        let g = self.grasp_info(item);
        g.margin * item.spec().mass_kg * hand_core::balance::G * GRIP_SAFETY
    }

    /// Balance states entered since the last call.
    pub fn take_balance_events(&mut self) -> Vec<BalanceEvent> {
        std::mem::take(&mut self.balance_events)
    }

    pub fn battery_ratio(&self) -> f32 {
        (1.0 - self.agent.energy_used_j / self.battery_capacity_j()).clamp(0.0, 1.0) as f32
    }

    pub fn battery_depleted(&self) -> bool {
        self.agent.energy_used_j >= self.battery_capacity_j()
    }

    /// Mining progress per second: a full five-finger hand works at 1.0
    /// (assumption: proportional to the finger count), slowed when the
    /// working arm's joints run hot (heat above 1 derates by 1/heat).
    pub fn mining_rate(&self) -> f32 {
        let hot = ["shoulder_r", "elbow_r"].iter().map(|j| self.heat.get(*j).copied().unwrap_or(0.0)).fold(0.0f32, f32::max);
        u8::from(self.hand.config.fingers) as f32 / 5.0 / hot.max(1.0)
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
    pub fn blocked_cells(&self) -> HashSet<Cell> {
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
        nav::plan(&self.drive_spec(), &self.terrain, &|x, z| blocked.contains(&(x, z)), self.agent_cell(), goal)
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
        if self.fallen_until.is_some_and(|t| self.time_s < t) {
            return Err(ActionError::Fallen);
        }
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
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
                self.odom.new_move();
                Active::Move(MoveState { edges: plan.edges, idx: 0, pause_left: 0.0, progress_s: 0.0, announced_pause: false, elapsed_s: 0.0 })
            }
            Action::Turn { yaw_deg } => Active::Turn { target: wrap_angle(yaw_deg.to_radians()) },
            Action::Face { x, z } => Active::Turn { target: (z - self.agent.z).atan2(x - self.agent.x) },
            Action::Mine { node } => {
                let n = self.node(*node).ok_or(ActionError::UnknownTarget)?;
                self.check_reach(n.x, n.z, MINE_REACH_M)?;
                Active::Work(WorkState { kind: WorkKind::Mine { node: *node }, remaining_s: n.hp })
            }
            Action::PickUp { drop, aim, side, memory_sigma_m } => {
                let (dx, dz, kind, amount) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind, d.amount)
                };
                self.check_reach(dx, dz, PICKUP_REACH_M)?;
                let info = self.grasp_for(kind, amount);
                self.agent.grip_margin = Some(info.margin);
                // From above, the fingers' friction alone carries it: one hand, or both sharing it.
                if !info.stable && 2.0 * info.margin < 1.0 {
                    return Err(ActionError::GraspUnstable { margin: info.margin });
                }
                let item_kg = kind.spec().mass_kg * amount;
                if self.carried_kg() + item_kg > self.capacity_kg() + 1e-4 {
                    return Err(ActionError::CapacityExceeded {
                        carried_kg: self.carried_kg(),
                        item_kg,
                        capacity_kg: self.capacity_kg(),
                    });
                }
                // Can the arms lift it, and does the robot stay balanced while they do?
                let (rep, mut two_arms) = self.body.lift(item_kg, self.carried_kg(), &self.ff_for(Source::ArmLift), self.grip_capacity_for(kind, amount));
                if !rep.feasible {
                    return Err(ActionError::LiftOverload { limiting: rep.limiting.clone().unwrap_or_default(), margin: rep.margin });
                }
                // A grip one hand cannot hold with its safety factor takes both hands.
                if !info.stable {
                    two_arms = true;
                }
                // Bend down, close, lift, stow: the action's own time series (`hand_core::act::grasp`).
                let (t, face) = self.start_grasp(*drop, two_arms, *aim, *side, *memory_sigma_m)?;
                Active::Work(WorkState { kind: WorkKind::Pick { drop: *drop, two_arms, face }, remaining_s: t })
            }
            Action::Drop { item, amount } => {
                let have: f32 = self.agent.carried.iter().filter(|(k, n)| *n > AMOUNT_EPS && item.map_or(true, |i| i == *k)).map(|(_, n)| *n).sum();
                if have <= AMOUNT_EPS || amount.is_some_and(|a| a <= AMOUNT_EPS) {
                    return Err(ActionError::NothingToDrop);
                }
                let waypoint = self.waypoint_in_reach();
                let pieces = self.pieces_to_set_down(*item, *amount);
                let t = self.start_release(&pieces, waypoint);
                Active::Work(WorkState { kind: WorkKind::Drop { item: *item, amount: *amount, waypoint }, remaining_s: t })
            }
            Action::Split { drop, portion } => {
                let (dx, dz, kind, amount) = {
                    let d = self.drop_item(*drop).ok_or(ActionError::UnknownTarget)?;
                    (d.x, d.z, d.kind, d.amount)
                };
                self.check_reach(dx, dz, PICKUP_REACH_M)?;
                if !kind.spec().divisible {
                    return Err(ActionError::NotDivisible);
                }
                let piece = amount * portion;
                if !(0.0..1.0).contains(portion) || piece < MIN_PIECE - AMOUNT_EPS || amount - piece < MIN_PIECE - AMOUNT_EPS {
                    return Err(ActionError::PieceTooSmall { amount: piece.min(amount - piece) });
                }
                Active::Work(WorkState { kind: WorkKind::Split { drop: *drop, portion: *portion }, remaining_s: self.split_time_s(kind, amount) })
            }
            Action::Deliver { waypoint, item } => {
                let w = self.waypoint(*waypoint).ok_or(ActionError::UnknownTarget)?;
                self.check_reach(w.x, w.z, DELIVER_REACH_M)?;
                let any = self.agent.carried.iter().any(|(k, n)| *n > AMOUNT_EPS && item.map_or(true, |i| i == *k));
                if !any {
                    return Err(ActionError::NothingToDeliver);
                }
                let pieces = self.pieces_to_set_down(*item, None);
                let t = self.start_release(&pieces, Some(*waypoint));
                Active::Work(WorkState { kind: WorkKind::Deliver { waypoint: *waypoint, item: *item }, remaining_s: t })
            }
            Action::Push { drop, x, z } => Active::Push(self.start_push(*drop, *x, *z)?),
        };
        // A start can move the body at once (a pick's shuffle to its stand-off,
        // a push lining up its heading): the odometry counts that too.
        let after = (self.agent.x, self.agent.z, self.agent.yaw);
        if after != before {
            self.odom.update(before, after, 0.0, 0.0);
        }
        let label = action.label();
        self.active = Some((action, state));
        Ok(vec![self.stamp(WorldEvent::ActionStarted { action: label })])
    }

    /// Stops the current action.
    pub fn cancel(&mut self) -> Vec<TimedEvent> {
        self.end_act();
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
        self.update_balance(dt);
        let before = (self.agent.x, self.agent.z, self.agent.yaw);
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
        // The robot's own sensors count the motion of this tick, with their errors.
        let after = (self.agent.x, self.agent.z, self.agent.yaw);
        self.odom.update(before, after, dt, self.odom_over);
        self.odom_over = 0.0;
        self.eye.advance(before, after, dt);
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
                    self.add_energy(self.drive_spec().power_w as f64 * t as f64);
                    used = t;
                }
            }
            Active::Move(m) => {
                used = self.advance_move(m, remaining, events, &mut done);
            }
            Active::Work(w) => {
                used = self.advance_work(w, remaining, events, &mut done);
            }
            Active::Push(ps) => {
                used = self.advance_push(ps, remaining, events, &mut done);
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
        let power = self.drive_spec().power_w as f64;

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

        // A slipping wheel or track counts distance it did not travel (the drive model's cruise slip on this ground).
        if !self.body.legged {
            let props = self.ground_at(e.to).surface().props();
            self.odom_over = hand_core::act::cruise_slip(&self.drive_spec(), props, e.slope_deg).map_or(0.0, |s| s / (1.0 - s).max(0.05));
        }
        let left = e.move_time_s - m.progress_s;
        let t = left.min(remaining);
        m.progress_s += t;
        m.elapsed_s += t;
        self.rotate_toward(heading, t);
        let f = (m.progress_s / e.move_time_s).clamp(0.0, 1.0);
        let (px, pz) = (self.agent.x, self.agent.z);
        self.agent.x = fx + (tx - fx) * f;
        self.agent.z = fz + (tz - fz) * f;
        self.agent.speed_mps = e.horizontal_m / e.move_time_s;
        // Legs: every step that comes down can slip; a slip the robot cannot catch is a fall.
        if let Some(why) = self.walk_steps((self.agent.x - px).hypot(self.agent.z - pz), &e, events) {
            *done = Some((false, Some(format!("fall: {why}"))));
            return t;
        }
        // The ground being entered: rolling resistance or the walking terrain factor.
        self.add_energy(power * e.energy_factor as f64 * t as f64);
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
            // Toward the item where the robot believes it is; a reach to the side does not turn.
            WorkKind::Pick { drop, face, .. } => match (self.drop_item(drop), face) {
                (Some(_), Some(p)) => p,
                (Some(_), None) => (self.agent.x + self.agent.yaw.cos(), self.agent.z + self.agent.yaw.sin()),
                (None, _) => {
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
            WorkKind::Split { drop, .. } => match self.drop_item(drop) {
                Some(d) => (d.x, d.z),
                None => {
                    *done = Some((false, Some("unknown_target".into())));
                    return 0.0;
                }
            },
            // Onto a pile: face it; onto the ground: set down ahead, no turn.
            WorkKind::Drop { waypoint, .. } => match waypoint.and_then(|w| self.waypoint(w)) {
                Some(p) => (p.x, p.z),
                None => (self.agent.x + self.agent.yaw.cos(), self.agent.z + self.agent.yaw.sin()),
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
        self.act_advance(t, events);
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
            WorkKind::Pick { drop, .. } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    let play = self.end_act();
                    match play.as_ref().map(|p| (p.rollout.outcome.ok, p.rollout.outcome.code)) {
                        Some((false, code)) => {
                            if let Some(p) = &play {
                                self.grasp_failed(drop, p);
                            }
                            *done = Some((false, Some(code.unwrap_or("grasp_failed").to_string())));
                        }
                        _ => {
                            self.complete_pick(drop, events);
                            *done = Some((true, None));
                        }
                    }
                }
            }
            WorkKind::Deliver { waypoint, item } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    self.end_act();
                    self.complete_deliver(waypoint, item, events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Drop { item, amount, waypoint } => {
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.act_finish(events);
                    let play = self.end_act();
                    self.complete_drop(item, amount, waypoint, play.as_ref(), events);
                    *done = Some((true, None));
                }
            }
            WorkKind::Split { drop, portion } => {
                // Breaking a piece is striking: it wears the joints like mining.
                self.agent.strike_acc += t;
                while self.agent.strike_acc >= 1.0 {
                    self.agent.strike_acc -= 1.0;
                    self.agent.strikes += 1;
                }
                w.remaining_s -= t;
                if w.remaining_s <= 1e-6 {
                    self.complete_split(drop, portion, events);
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
            let d = DropItem::whole(self.next_drop, spec.drop, x, z);
            self.next_drop += 1;
            ids.push(drop_id(d.id));
            self.drops.push(d);
        }
        events.push(self.stamp(WorldEvent::NodeBroken { node: node_id(id), kind: node.kind, drops: ids }));
    }

    fn complete_pick(&mut self, id: u32, events: &mut Vec<TimedEvent>) {
        if let Some(pos) = self.drops.iter().position(|d| d.id == id) {
            let d = self.drops.remove(pos);
            add_stock(&mut self.agent.carried, d.kind, d.amount);
            let margin = self.agent.grip_margin.unwrap_or(0.0);
            events.push(self.stamp(WorldEvent::ItemPicked { drop: drop_id(id), item: d.kind, mass_kg: d.mass_kg(), grip_margin: margin, amount: d.amount }));
        }
    }

    fn complete_deliver(&mut self, waypoint: u32, item: Option<ItemKind>, events: &mut Vec<TimedEvent>) {
        let mut moved: Vec<(ItemKind, f32)> = Vec::new();
        self.agent.carried.retain_mut(|(k, n)| {
            if *n > AMOUNT_EPS && item.map_or(true, |i| i == *k) {
                moved.push((*k, *n));
                false
            } else {
                *n > AMOUNT_EPS
            }
        });
        for (k, n) in moved {
            if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == waypoint) {
                add_stock(&mut w.stock, k, n);
            }
            events.push(self.stamp(WorldEvent::Delivered { waypoint: waypoint_id(waypoint), item: k, amount: n }));
        }
    }

    /// Takes up to `amount` (all when `None`) of the matching kinds out of the bay.
    fn take_cargo(&mut self, item: Option<ItemKind>, amount: Option<f32>) -> Vec<(ItemKind, f32)> {
        let mut left = amount.unwrap_or(f32::INFINITY);
        let mut taken: Vec<(ItemKind, f32)> = Vec::new();
        for (k, n) in self.agent.carried.iter_mut() {
            if left <= AMOUNT_EPS || *n <= AMOUNT_EPS || !item.map_or(true, |i| i == *k) {
                continue;
            }
            let a = n.min(left);
            *n -= a;
            left -= a;
            taken.push((*k, a));
        }
        self.agent.carried.retain(|(_, n)| *n > AMOUNT_EPS);
        taken
    }

    fn complete_drop(&mut self, item: Option<ItemKind>, amount: Option<f32>, waypoint: Option<u32>, play: Option<&ActPlay>, events: &mut Vec<TimedEvent>) {
        let taken = self.take_cargo(item, amount);
        let mut piece_index = 0usize;
        for (k, a) in taken {
            if let Some(wid) = waypoint {
                if let Some(w) = self.waypoints.iter_mut().find(|w| w.id == wid) {
                    add_stock(&mut w.stock, k, a);
                }
                events.push(self.stamp(WorldEvent::ItemDropped { item: k, amount: a, waypoint: Some(waypoint_id(wid)), drops: vec![] }));
                continue;
            }
            // On the ground ahead: one piece per whole item and one for the rest.
            let ahead = (self.agent.x + DROP_AHEAD_M * self.agent.yaw.cos(), self.agent.z + DROP_AHEAD_M * self.agent.yaw.sin());
            let on_land = |w: &World, x: f32, z: f32| w.terrain.cell_of(x, z).is_some_and(|(cx, cz)| !w.terrain.is_water(cx, cz) && !w.blocked_cells().contains(&(cx, cz)));
            let (bx, bz) = if on_land(self, ahead.0, ahead.1) { ahead } else { (self.agent.x, self.agent.z) };
            let mut pieces = Vec::new();
            let mut rest = a;
            while rest > AMOUNT_EPS {
                pieces.push(rest.min(1.0));
                rest -= 1.0;
            }
            let (sx, sz) = (-self.agent.yaw.sin(), self.agent.yaw.cos());
            let mut ids = Vec::new();
            for (j, p) in pieces.iter().enumerate() {
                let off = (j as f32 - (pieces.len() as f32 - 1.0) * 0.5) * 0.13;
                // Where the release let it fall, rebound included; the old spread when there was no release.
                let (lx, lz) = play
                    .and_then(|pl| self.release_landing(pl, piece_index))
                    .filter(|&(x, z)| on_land(self, x, z))
                    .unwrap_or((bx + sx * off, bz + sz * off));
                piece_index += 1;
                let d = DropItem { id: self.next_drop, kind: k, x: lx, z: lz, amount: *p };
                self.next_drop += 1;
                ids.push(drop_id(d.id));
                self.drops.push(d);
            }
            events.push(self.stamp(WorldEvent::ItemDropped { item: k, amount: a, waypoint: None, drops: ids }));
        }
    }

    fn complete_split(&mut self, id: u32, portion: f32, events: &mut Vec<TimedEvent>) {
        let Some(d) = self.drops.iter_mut().find(|d| d.id == id) else { return };
        let piece = d.amount * portion;
        d.amount -= piece;
        let (kind, left, x, z) = (d.kind, d.amount, d.x, d.z);
        // The broken-off piece lands beside the rest, toward the robot.
        let (ax, az) = (self.agent.x - x, self.agent.z - z);
        let n = ax.hypot(az).max(1e-3);
        let new = DropItem { id: self.next_drop, kind, x: x + 0.12 * ax / n, z: z + 0.12 * az / n, amount: piece };
        self.next_drop += 1;
        events.push(self.stamp(WorldEvent::ItemSplit { drop: drop_id(id), piece: drop_id(new.id), item: kind, amount: piece, left }));
        self.drops.push(new);
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
                ground.push(t.ground_at(cx, cz).code());
            }
        }
        TerrainGrid { cells_w: t.w, cells_d: t.d, cell_m: CELL_M, heights_cm: heights, curb_cm: curbs, ground, kind: self.cfg.terrain }
    }

    fn carry_views(list: &[(ItemKind, f32)]) -> Vec<CarryView> {
        list.iter()
            .filter(|(_, n)| *n > AMOUNT_EPS)
            .map(|(k, n)| CarryView {
                item: *k,
                label_ko: k.spec().label_ko.into(),
                count: (*n + AMOUNT_EPS).floor() as u32,
                amount: *n,
                mass_kg: k.spec().mass_kg * *n,
                unit_kg: k.spec().mass_kg,
            })
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
                belief: self.odom.cfg.enabled.then(|| {
                    let (x, z, yaw) = self.believed_pose();
                    BeliefView { x, z, yaw_deg: yaw.to_degrees(), error_m: self.belief_error_m(), since_fix_m: self.odom.since_fix_m }
                }),
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
                .map(|dr| DropView { id: drop_id(dr.id), item: dr.kind, label_ko: dr.kind.spec().label_ko.into(), pos: at(dr.x, dr.z), amount: dr.amount })
                .collect(),
            terrain: self.cfg.terrain,
        }
    }

    /// The posture the current action asks for, and its terrain/phase bucket.
    pub fn posture_now(&self) -> (Posture, String) {
        let (post, state) = self.posture_action();
        (Posture { ground: self.ground_here().surface(), ..post }, state)
    }

    fn posture_action(&self) -> (Posture, String) {
        let cargo = self.carried_kg();
        let bucket = |deg: f32, step: f32| -> String {
            if step > 0.0 {
                "step".into()
            } else if deg.abs() < 3.0 {
                "flat".into()
            } else {
                format!("{}_{}", if deg > 0.0 { "up" } else { "down" }, if deg.abs() < 10.0 { "lo" } else { "hi" })
            }
        };
        match &self.active {
            Some((_, Active::Move(m))) => {
                let Some(e) = m.edges.get(m.idx) else { return (Posture::quiet(cargo), "flat|move".into()) };
                let rise = self.terrain.height_m(e.to.0, e.to.1) - self.terrain.height_m(e.from.0, e.from.1);
                let slope = if rise >= 0.0 { e.slope_deg } else { -e.slope_deg };
                let ground = self.ground_here();
                let mut p = Posture { slope_deg: slope, moving: true, ..Posture::quiet(cargo) };
                let b = bucket(slope, if m.pause_left > 0.0 { e.step_m } else { 0.0 });
                let mut b_brake = false;
                if m.pause_left > 0.0 {
                    p.source = Source::StepUp;
                    p.step_m = e.step_m;
                    p.single_support = self.body.legged;
                } else if m.elapsed_s < self.drive_spec().max_speed_mps / self.planned_accel(slope, ground, 1.0) {
                    // The planner starts no harder than the ZMP's room toward the back edge and the ground's friction allow on this slope.
                    p.source = Source::Accel;
                    p.accel_mps2 = self.planned_accel(slope, ground, 1.0);
                } else if self.remaining_m(m) < self.braking_distance_m(slope, ground) {
                    // The last stretch of a move is the stop, planned against the front edge.
                    p.source = Source::Accel;
                    p.accel_mps2 = -self.planned_accel(slope, ground, -1.0);
                    b_brake = true;
                } else if self.body.legged {
                    p.source = Source::FootLift;
                    p.single_support = true;
                } else if slope.abs() >= 3.0 {
                    p.source = Source::Slope;
                }
                (p, if b_brake { format!("{b}|brake") } else { format!("{b}|move") })
            }
            Some((_, Active::Work(w))) => match &w.kind {
                WorkKind::Mine { .. } => (Posture::arm_raise(cargo), "flat|mine".into()),
                WorkKind::Pick { drop, two_arms, .. } => {
                    let kg = self.drop_item(*drop).map_or(0.0, |d| d.mass_kg());
                    (Posture::arm_lift(kg, *two_arms, cargo), "flat|pick".into())
                }
                WorkKind::Deliver { .. } | WorkKind::Drop { .. } => {
                    // The heaviest single piece in the bay is the one the arms handle.
                    let kg = self.agent.carried.iter().map(|(k, n)| k.spec().mass_kg * n.min(1.0)).fold(0.0f32, f32::max);
                    let tag = if matches!(w.kind, WorkKind::Drop { .. }) { "flat|drop" } else { "flat|deliver" };
                    (Posture::arm_lift(kg, false, (cargo - kg).max(0.0)), tag.into())
                }
                WorkKind::Split { .. } => (Posture::arm_raise(cargo), "flat|split".into()),
            },
            Some((_, Active::Push(_))) => (Posture { moving: true, ..Posture::quiet(cargo) }, "flat|push".into()),
            _ => (Posture::quiet(cargo), "flat|idle".into()),
        }
    }

    /// Solves the balance of the current posture, heats the joints and notes
    /// a balance event when the state changed or lasted long enough to sample.
    fn update_balance(&mut self, dt: f32) {
        let (post, state) = self.posture_now();
        let grip = match &self.active {
            Some((_, Active::Work(WorkState { kind: WorkKind::Pick { drop, .. }, .. }))) => self.drop_item(*drop).map(|d| self.grip_capacity_for(d.kind, d.amount)),
            _ => None,
        }
        .unwrap_or(1e3);
        let rep = self.body.solve(&post, &self.ff_for(post.source), grip);
        for j in &rep.joints {
            let Some(act) = self.joint_actuator(&j.joint) else { continue };
            // Short term: heat follows effort² with the actuator's time constant.
            let h = self.heat.entry(j.joint.clone()).or_insert(0.0);
            *h += (dt / act.thermal_tau_s).min(1.0) * (j.effort * j.effort - *h);
            // Long term: the reducer's life used at this torque.
            *self.wear.entry(j.joint.clone()).or_insert(0.0) += hand_core::wear::damage(&act, j.torque_nm, dt);
        }
        let key = format!("{}|{state}", post.source.id());
        let changed = self.balance.as_ref().map_or(true, |(k, _)| *k != key);
        // A new posture: move the joints there with the presets' actuators.
        let digest = if changed {
            let roll = dynamics::simulate(&self.body, &self.last_posture, &post, &self.ff_for(post.source), grip, &DynCfg::default());
            let d = roll.digest();
            self.motion = Some(Motion { rollout: roll, posture: post, start_s: self.time_s, state: state.clone() });
            self.last_posture = post;
            Some(d)
        } else {
            None
        };
        if changed || self.tick.saturating_sub(self.balance_since) >= BALANCE_RESAMPLE_TICKS {
            self.balance_since = self.tick;
            if self.balance_events.len() < 2000 {
                self.balance_events.push(BalanceEvent { tick: self.tick, time_s: self.time_s, state: state.clone(), report: rep.clone(), dynamics: digest, posture: post });
            }
        }
        self.balance = Some((key, rep));
    }

    /// The whole body at this instant, as the joint dynamics move it (base
    /// frame: x forward, y left, z up from the ground under the robot).
    pub fn body_pose(&self) -> Option<BodyPose> {
        if let Some(p) = self.act_body_pose() {
            return Some(p);
        }
        match &self.motion {
            Some(m) => m.rollout.pose_at(&self.body, &m.posture, (self.time_s - m.start_s) as f32),
            None => {
                let q = Posture::quiet(self.carried_kg());
                let r = dynamics::simulate(&self.body, &q, &q, &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.05, ..DynCfg::default() });
                r.pose_at(&self.body, &q, 0.0)
            }
        }
    }

    /// The preset actuator that drives a joint of the balance model.
    pub fn joint_actuator(&self, joint: &str) -> Option<hand_core::parts::Actuator> {
        match joint {
            n if n.starts_with("shoulder") => Some(self.body.arm.shoulder),
            n if n.starts_with("elbow") => Some(self.body.arm.elbow),
            "waist" => self.body.belly.pitch,
            "neck" => Some(self.body.head.neck),
            "knee" | "hip_roll" => Some(self.body.leg.knee),
            "ankle" => Some(self.body.leg.ankle),
            _ => None,
        }
    }

    /// The joint closest to a limit: the larger of its torque over the peak
    /// rating and its RMS over the continuous rating (`√heat`, the motor
    /// sizing rules) — joint, that index, peak ratio, RMS ratio.
    pub fn joint_load(&self) -> Option<(String, f32, f32, f32)> {
        let (_, r) = self.balance.as_ref()?;
        r.joints
            .iter()
            .filter(|j| !j.joint.starts_with("grip"))
            .map(|j| {
                let rms = self.heat.get(&j.joint).copied().unwrap_or(0.0).max(0.0).sqrt();
                (j.joint.clone(), j.peak_ratio.max(rms), j.peak_ratio, rms)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The most worn joint and its Miner damage.
    pub fn most_worn(&self) -> Option<(String, f32)> {
        self.wear.iter().max_by(|a, b| a.1.total_cmp(b.1)).map(|(k, v)| (k.clone(), *v))
    }

    /// Summary of the current balance for the body status.
    pub fn balance_summary(&self) -> Option<BalanceSummary> {
        let (key, r) = self.balance.as_ref()?;
        let worst = r.worst();
        Some(BalanceSummary {
            plan: self.cfg.body.id(),
            source: r.source.id().into(),
            source_ko: r.source.label_ko().into(),
            state: key.splitn(2, '|').nth(1).unwrap_or("").to_string(),
            margin: r.margin,
            margin_noff: r.margin_noff,
            trunk_lean_deg: r.trunk_lean_deg,
            head_tilt_deg: r.head_pitch_deg.abs().max(r.head_roll_deg.abs()),
            worst_joint: worst.map(|w| w.joint.clone()),
            worst_ratio: worst.map_or(0.0, |w| w.peak_ratio),
            effort_sum: r.effort_sum,
            mass_kg: r.mass_kg,
            com_height_m: r.com[2],
            heat_max: self.heat.values().copied().fold(0.0, f32::max),
            load_joint: self.joint_load().map(|l| l.0),
            load_index: self.joint_load().map_or(0.0, |l| l.1),
            load_rms: self.joint_load().map_or(0.0, |l| l.3),
            joints: r.joints.clone(),
            couplings: r.couplings.clone(),
            ground: self.ground_here().label_ko().into(),
            contact: Some(r.contact),
        })
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
            arm_wear: ["shoulder_r", "elbow_r"].iter().filter_map(|j| self.wear.get(*j)).copied().fold(0.0, f32::max),
            wear_max: self.most_worn().map_or(0.0, |w| w.1),
            wear_joint: self.most_worn().map(|w| w.0),
            speed_factor: (self.agent.speed_mps / self.drive_spec().max_speed_mps).clamp(0.0, 1.0),
            terrain_margin: self.agent.terrain_margin,
            limits: Some({
                let l = self.limits();
                BodyLimits { battery_j: self.battery_capacity_j(), capacity_kg: l.capacity_kg, max_speed_mps: l.max_speed_mps, max_slope_deg: l.max_slope_deg, max_step_m: l.max_step_m }
            }),
            body_mass_kg: Some(self.body.mass_kg()),
            capacity_here: Some((self.capacity_here(), self.ground_here().label_ko().to_string())),
            balance: self.balance_summary(),
        }
    }

    /// Cargo the base carries on the ground and slope under it (`Body::capacity_on`).
    pub fn capacity_here(&self) -> f32 {
        let slope = self.posture_now().0.slope_deg;
        self.body.capacity_on(&self.hand.base, self.ground_here().surface(), slope).min(self.capacity_kg())
    }

    pub fn body_status(&self, vision_confidence: Option<f32>, frame_ms: Option<f32>) -> BodyStatus {
        compute_body(&self.body_inputs(vision_confidence, frame_ms))
    }

    /// The pieces a set-down takes out of the bay, in the order the hand takes them.
    fn pieces_to_set_down(&self, item: Option<ItemKind>, amount: Option<f32>) -> Vec<(ItemKind, f32)> {
        let mut left = amount.unwrap_or(f32::INFINITY);
        let mut out = Vec::new();
        for (k, n) in &self.agent.carried {
            if !item.map_or(true, |i| i == *k) {
                continue;
            }
            let mut rest = n.min(left);
            left -= rest;
            while rest > AMOUNT_EPS {
                out.push((*k, rest.min(1.0)));
                rest -= 1.0;
            }
        }
        out
    }
}

/// The grasp table entry of a hand for a piece lifted off the ground from above:
/// the fingers' squeeze times the friction the controller believes, over the
/// weight with the safety factor (`hand_core::act::grasp::top_grasp_ok`).
fn top_grasp_info(hand: &Hand, item: ItemKind, amount: f32) -> GraspInfo {
    let s = item.spec();
    let mass = s.mass_kg * amount;
    let mu = hand_core::act::material::pad_on(material_of(item)).mean;
    let cap = hand_core::act::grasp::capacity(hand, s.piece_size_m(amount), mass, mu);
    let need = mass * hand_core::balance::G * GRIP_SAFETY;
    let margin = if need > 1e-6 { mu * cap.squeeze_n / need } else { 99.0 };
    GraspInfo { item, stable: cap.contacts >= 2 && margin >= 1.0, margin: margin.min(99.0), max_payload_kg: mu * cap.squeeze_n / (hand_core::balance::G * GRIP_SAFETY), grasp_type: cap.grasp_type }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hand_core::{BaseKind, FingerCount};

    /// A world without the placed parcels: these tests are about the nodes and their drops.
    fn world(fingers: FingerCount, base: BaseKind) -> World {
        let mut hand = HandConfig::preset(fingers);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, hand, body: BodyPlan::default(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn a_pick_never_shuffles_the_body_onto_ground_it_cannot_drive_onto() {
        let mut w = world(FingerCount::Five, BaseKind::HumanFoot);
        let (cx, cz) = w.agent_cell();
        let (ax, az) = (w.agent.x, w.agent.z);
        assert!(w.can_step_to(ax + 0.1, az), "within its own cell");
        assert!(w.can_step_to(ax + CELL_M, az), "onto a level neighbour");
        // A rock column 0.9 m above the cell ahead: no step onto it, and not two cells away either.
        let h = (w.terrain.height_m(cx, cz) * 100.0).round() as i32;
        w.terrain.set_height_cm(cx + 1, cz, h + 90);
        assert!(!w.can_step_to(ax + CELL_M, az));
        assert!(!w.can_step_to(ax + 2.0 * CELL_M, az));
    }

    #[test]
    fn parcels_lie_about_and_one_waits_in_front_of_the_base_camp() {
        let mut hand = HandConfig::preset(FingerCount::Five);
        hand.base = BaseKind::HumanFoot;
        let w = World::new(WorldConfig { hand, ..WorldConfig::default() }).unwrap();
        let parcels: Vec<&DropItem> = w.drops.iter().filter(|d| d.kind == ItemKind::Parcel).collect();
        assert_eq!(parcels.len(), w.drops.len(), "only parcels lie about at the start");
        assert!(parcels.len() >= 3, "{}", parcels.len());
        // The first one: ahead of the robot on its way to the warehouse, a little to its right.
        let p = parcels[0];
        let (s, c) = w.agent.yaw.sin_cos();
        let (rx, rz) = (p.x - w.agent.x, p.z - w.agent.z);
        let (ahead, right) = (rx * c + rz * s, -rx * s + rz * c);
        assert!((ahead - CAMP_PARCEL_AHEAD_M).abs() < 1e-3 && (right - CAMP_PARCEL_RIGHT_M).abs() < 1e-3, "{ahead} {right}");
        // None on a node's cell or in the water; the same seed puts them in the same places.
        let blocked = w.blocked_cells();
        assert!(parcels.iter().all(|d| w.terrain.cell_of(d.x, d.z).is_some_and(|c| !blocked.contains(&c) && !w.terrain.is_water(c.0, c.1))));
        let again = World::new(WorldConfig { hand, ..WorldConfig::default() }).unwrap();
        assert!(w.drops.iter().zip(&again.drops).all(|(a, b)| (a.x, a.z, a.id) == (b.x, b.z, b.id)));
        // New drops get ids after them.
        assert!(w.next_drop > parcels.iter().map(|d| d.id).max().unwrap());
        // The nodes do not move for them (their own random stream).
        let bare = World::new(WorldConfig { hand, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        assert!(w.nodes.iter().zip(&bare.nodes).all(|(a, b)| (a.x, a.z) == (b.x, b.z)));
    }

    #[test]
    fn the_belief_drifts_while_the_robot_drives_and_maps_back() {
        let mut w = World::new(WorldConfig { parcels: Some(0), ..WorldConfig::default() }).unwrap();
        assert!(w.belief_error_m() < 1e-6, "it starts on its pad");
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        let goal = (camp.x + 6.0, camp.z + 2.0);
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        run(&mut w, 120.0);
        let err = w.belief_error_m();
        assert!(err > 1e-4 && err < 0.5, "{err}");
        assert!((w.odom.travelled_m - 6.3).abs() < 1.5, "{}", w.odom.travelled_m);
        // What it believes about itself is where it is in its own frame, and a point maps back.
        let (bx, bz, _) = w.believed_pose();
        let me = w.belief_of(w.agent.x, w.agent.z);
        assert!((me.0 - bx).abs() < 1e-4 && (me.1 - bz).abs() < 1e-4);
        let p = w.true_of(10.0, 9.0);
        let q = w.belief_of(p.0, p.1);
        assert!((q.0 - 10.0).abs() < 1e-3 && (q.1 - 9.0).abs() < 1e-3);
        // Exact odometry: the belief is the truth.
        let mut e = World::new(WorldConfig { parcels: Some(0), odometry: OdomCfg::exact(), ..WorldConfig::default() }).unwrap();
        e.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.0 }).unwrap();
        run(&mut e, 120.0);
        assert!(e.belief_error_m() < 1e-3);
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
    fn five_finger_legs_carry_a_log_to_the_warehouse() {
        // Legs: the body bends to the ground (a wheeled or tracked mid body cannot).
        let mut w = world(FingerCount::Five, BaseKind::HumanFoot);
        let ev = mine_nearest(&mut w, NodeKind::Tree);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::NodeBroken { .. })));
        let drop = w.drops[0].clone();
        w.command(Action::MoveTo { x: drop.x, z: drop.z, reach_m: 0.5 }).unwrap();
        run(&mut w, 60.0);
        w.command(Action::pick(drop.id)).unwrap();
        let ev = run(&mut w, 20.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })), "{ev:?}");
        assert_eq!(w.carried_count(ItemKind::Log), 1);
        // The pick played its physics: the bend, the contact and the lift-off came out as events.
        for k in ["bend_bottom", "contact", "lift_off", "stowed"] {
            assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::Act { kind, .. } if kind == k)), "no {k}");
        }
        let wh = w.waypoint_by_key("warehouse").unwrap().clone();
        w.command(Action::MoveTo { x: wh.x, z: wh.z, reach_m: 1.0 }).unwrap();
        let ev = run(&mut w, 600.0);
        assert!(ok_done(&ev));
        w.command(Action::Deliver { waypoint: wh.id, item: Some(ItemKind::Log) }).unwrap();
        run(&mut w, 20.0);
        assert_eq!(w.carried_count(ItemKind::Log), 0);
        let stock = &w.waypoint(wh.id).unwrap().stock;
        assert_eq!(stock, &vec![(ItemKind::Log, 1.0)]);
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
        // From above only the fingers' friction holds: one hand, or two sharing the item.
        let holdable = |f: FingerCount| {
            let w = world(f, BaseKind::Track);
            ItemKind::ALL.into_iter().filter(|&k| w.grasp_info(k).stable || 2.0 * w.grasp_info(k).margin >= 1.0).count()
        };
        let (two, five) = (holdable(FingerCount::Two), holdable(FingerCount::Five));
        assert!(five >= 3, "a five-finger hand should hold at least three of four items, holds {five}");
        assert!(two < five, "a two-finger hand should hold fewer items ({two} vs {five})");
        assert!(two >= 1, "the lightest item should be holdable even with two fingers");
    }

    #[test]
    fn an_unstable_grasp_blocks_the_pickup() {
        let mut w = world(FingerCount::Two, BaseKind::HumanFoot);
        // Force the situation: whichever item even two such hands cannot hold.
        let bad = ItemKind::ALL
            .into_iter()
            .find(|&k| 2.0 * w.grasp_info(k).margin < 1.0)
            .expect("the two-finger hand cannot hold every item (see item_catalogue_spans_what_the_hands_can_hold)");
        w.drops.push(DropItem::whole(500, bad, w.agent.x + 0.4, w.agent.z));
        match w.command(Action::pick(500)) {
            Err(ActionError::GraspUnstable { margin }) => assert!(margin < 1.0 + 1e-3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn capacity_limit_stops_the_pickup() {
        let mut w = world(FingerCount::Five, BaseKind::Mecanum);
        let item = ItemKind::ALL.into_iter().find(|&k| w.grasp_info(k).stable).expect("some item is holdable");
        let cap = w.capacity_kg();
        add_stock(&mut w.agent.carried, item, (cap / item.spec().mass_kg).floor());
        w.drops.push(DropItem::whole(501, item, w.agent.x + 0.4, w.agent.z));
        assert!(matches!(w.command(Action::pick(501)), Err(ActionError::CapacityExceeded { .. })));
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

    fn world_with(body: BodyPlan, base: BaseKind) -> World {
        let mut hand = HandConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, hand, body, terrain: TerrainKind::Open, ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_arm_preset_decides_what_can_be_picked_up() {
        let place = |w: &mut World, kind: ItemKind| {
            w.drops.push(DropItem::whole(600, kind, w.agent.x + 0.4, w.agent.z));
            w.command(Action::pick(600))
        };
        let mut budget = world_with(BodyPlan { arm: hand_core::Tier::Budget, ..BodyPlan::default() }, BaseKind::HumanFoot);
        match place(&mut budget, ItemKind::Log) {
            Err(ActionError::LiftOverload { limiting, .. }) => assert!(limiting.contains("elbow"), "{limiting}"),
            other => panic!("{other:?}"),
        }
        let mut mid = world_with(BodyPlan::default(), BaseKind::HumanFoot);
        assert!(place(&mut mid, ItemKind::Log).is_ok());
        let ev = run(&mut mid, 20.0);
        assert!(ev.iter().any(|e| matches!(e.event, WorldEvent::ItemPicked { .. })), "{ev:?}");
        // Iron ore (more than mid legs carry): a bend, a squeeze and a lift take seconds, not the old fixed second.
        let mut iron = world_with(BodyPlan::uniform(hand_core::Tier::Pro), BaseKind::HumanFoot);
        let t0 = iron.time_s;
        place(&mut iron, ItemKind::IronOre).unwrap();
        run(&mut iron, 20.0);
        assert!((iron.time_s - t0) as f32 >= PICKUP_TIME_S * TWO_ARM_LIFT_FACTOR - 0.15);
        // A tracked mid body has no waist pitch: it cannot reach the ground at all.
        let mut track = world_with(BodyPlan::default(), BaseKind::Track);
        match place(&mut track, ItemKind::Log) {
            Err(ActionError::CannotReach { lowest_m, need_m, .. }) => assert!(lowest_m > need_m + 0.3, "{lowest_m} {need_m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mining_records_an_arm_raise_and_a_hot_budget_arm_mines_slower() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        mine_nearest(&mut w, NodeKind::Tree);
        let events = w.take_balance_events();
        let raise = events.iter().find(|e| e.report.source == Source::ArmRaise).expect("an arm-raise state");
        assert_eq!(raise.state, "flat|mine");
        assert!(raise.report.couplings.iter().any(|c| c.target == hand_core::Segment::Foot));
        let mut cold = world_with(BodyPlan { arm: hand_core::Tier::Budget, ..BodyPlan::default() }, BaseKind::Track);
        cold.heat.insert("elbow_r".into(), 2.0);
        assert!((cold.mining_rate() - 0.5).abs() < 1e-6, "a joint at heat 2 halves the strike rate");
    }

    fn rugged(seed: u64, base: BaseKind) -> World {
        let mut hand = HandConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed, cells: 64, hand, body: BodyPlan::default(), terrain: TerrainKind::Rugged, ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_rugged_map_has_sea_cliffs_and_ridges_and_stays_connected() {
        for seed in 1..=12u64 {
            let w = rugged(seed, BaseKind::Wheel);
            let l = w.layout.as_ref().expect("rugged layout");
            assert!(l.water_cells > 20, "seed {seed}: a sea");
            assert!(!l.mesas.is_empty() && !l.ridges.is_empty(), "seed {seed}: {l:?}");
            // The waypoints stay connected for the weakest common base.
            let wh = w.waypoint_by_key("warehouse").unwrap();
            assert!(w.plan_route(Goal::Within { x: wh.x, z: wh.z, reach_m: 1.0 }).is_some(), "seed {seed}: no wheel route to the warehouse");
            // A cliff is a height jump no base climbs.
            let t = &w.terrain;
            let cliff = (0..t.d).any(|z| (0..t.w - 1).any(|x| (t.height_cm(x, z) - t.height_cm(x + 1, z)).abs() >= 100));
            assert!(cliff, "seed {seed}: no cliff");
            // Every kind keeps two nodes a wheel can reach.
            let start = t.cell_of(w.agent.x, w.agent.z).unwrap();
            let blocked = w.blocked_cells();
            let times = nav::travel_times(&BaseKind::Wheel.spec(), t, &|x, z| blocked.contains(&(x, z)), start, f32::INFINITY);
            for k in NodeKind::ALL {
                let ok = w.nodes.iter().filter(|n| n.kind == k).filter(|n| {
                    let (cx, cz) = t.cell_of(n.x, n.z).unwrap();
                    (-1i32..=1).any(|dz| (-1i32..=1).any(|dx| t.in_bounds(cx as i32 + dx, cz as i32 + dz) && times[t.idx((cx as i32 + dx) as usize, (cz as i32 + dz) as usize)].is_finite()))
                }).count();
                assert!(ok >= 2, "seed {seed}: {k:?} has {ok} reachable nodes");
                assert!(w.nodes.iter().all(|n| !t.is_water(t.cell_of(n.x, n.z).unwrap().0, t.cell_of(n.x, n.z).unwrap().1)), "no node in the sea");
            }
        }
        // Same seed, same map; the open map is untouched by the option.
        let (a, b) = (rugged(5, BaseKind::Track), rugged(5, BaseKind::Track));
        assert_eq!(a.terrain_grid(), b.terrain_grid());
        assert!(world(FingerCount::Five, BaseKind::Track).layout.is_none());
    }

    #[test]
    fn pads_are_gravel_the_beach_is_dry_sand_behind_wet_sand_and_the_ground_reaches_the_posture() {
        let w = rugged(4, BaseKind::Wheel);
        let t = &w.terrain;
        let camp = w.waypoint_by_key("base_camp").unwrap();
        assert_eq!(w.ground_at(t.cell_of(camp.x, camp.z).unwrap()), Ground::Gravel);
        let count = |g: Ground| (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).filter(|&(x, z)| t.ground_at(x, z) == g).count();
        assert!(count(Ground::WetSand) > 0 && count(Ground::Sand) > 0, "a beach with both bands");
        let shore = t.water_distance_m(8.0);
        for z in 0..t.d {
            for x in 0..t.w {
                match t.ground_at(x, z) {
                    Ground::WetSand => assert!(shore[t.idx(x, z)] <= 1.05 + 1e-3),
                    Ground::Sand => assert!(shore[t.idx(x, z)] > 1.05 && shore[t.idx(x, z)] <= 2.5 + 1e-3),
                    _ => {}
                }
            }
        }
        // Standing on the pad, the posture knows its ground and the friction it gets.
        let (p, _) = w.posture_now();
        assert_eq!(p.ground, hand_core::Surface::Gravel);
        // The same move costs a wheel more on sand than on gravel.
        let mut flat = world(FingerCount::Five, BaseKind::Wheel);
        let cell = flat.terrain.cell_of(flat.agent.x, flat.agent.z).unwrap();
        let ahead = (cell.0 + 2, cell.1);
        let cost = |g: Ground| {
            let mut w = flat.clone();
            for c in [(cell.0 + 1, cell.1), ahead] {
                w.terrain.set_ground(c.0, c.1, g);
            }
            w.plan_route(Goal::Cell(ahead)).map(|p| p.energy_j / p.time_s.max(1e-6))
        };
        let (gravel, sand) = (cost(Ground::Gravel).unwrap(), cost(Ground::Sand).unwrap());
        assert!(sand > gravel * 1.2, "sand {sand} W vs gravel {gravel} W");
        flat.terrain.set_ground(ahead.0, ahead.1, Ground::Sand);
        flat.command(Action::MoveTo { x: cell_center(ahead.0, ahead.1).0, z: cell_center(ahead.0, ahead.1).1, reach_m: 0.0 }).unwrap();
        for _ in 0..200 {
            flat.tick();
            if flat.ground_here() == Ground::Sand {
                break;
            }
        }
        let s = flat.body_status(None, None);
        let slip = s.effects.iter().find(|e| e.id == "slip").unwrap();
        assert!(slip.detail.contains("모래") && slip.detail.contains("마찰 이용률"), "{}", slip.detail);
    }

    #[test]
    fn nothing_drives_into_the_sea_and_the_ramp_is_for_tracks_and_legs() {
        let w = rugged(3, BaseKind::Track);
        let t = &w.terrain;
        let water = (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).find(|&(x, z)| t.is_water(x, z)).unwrap();
        assert!(w.plan_route(Goal::Cell(water)).is_none());
        // The ramp rises 15 cm a cell: a track climbs it, a wheel's 15° does not.
        let e = (0..t.d).flat_map(|z| (0..t.w - 1).map(move |x| (x, z))).find(|&(x, z)| {
            let d = t.height_cm(x + 1, z) - t.height_cm(x, z);
            d == crate::terrain::RAMP_RISE_CM && t.ground_at(x + 1, z) == Ground::Rock && t.height_cm(x + 1, z) > 0
        });
        if let Some((x, z)) = e {
            assert!(nav::edge(&BaseKind::Track.spec(), t, (x, z), (x + 1, z)).is_some());
            assert!(nav::edge(&BaseKind::Wheel.spec(), t, (x, z), (x + 1, z)).is_none());
        }
    }

    #[test]
    fn carried_items_can_be_set_down_on_the_ground_or_on_a_pile() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        assert_eq!(w.command(Action::Drop { item: None, amount: None }), Err(ActionError::NothingToDrop));
        add_stock(&mut w.agent.carried, ItemKind::Log, 2.0);
        add_stock(&mut w.agent.carried, ItemKind::Coal, 1.0);
        // At the base camp: onto its pile.
        w.command(Action::Drop { item: Some(ItemKind::Log), amount: Some(1.0) }).unwrap();
        let ev = run(&mut w, 15.0);
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::ItemDropped { waypoint: Some(_), .. })));
        assert_eq!(w.waypoint_by_key("base_camp").unwrap().stock_of(ItemKind::Log), 1.0);
        assert_eq!(w.carried_count(ItemKind::Log), 1);
        // Away from the waypoints: on the ground ahead, one piece per item.
        w.agent.x += 5.0;
        w.agent.yaw = 0.0;
        let before = w.drops.len();
        w.command(Action::Drop { item: None, amount: None }).unwrap();
        let ev = run(&mut w, 30.0);
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::ItemDropped { waypoint: None, .. })));
        assert_eq!(w.drops.len(), before + 2, "a log and a coal lump");
        assert!(w.agent.carried.is_empty() && w.carried_kg() == 0.0);
        let d = w.drops.last().unwrap();
        assert!((d.x - w.agent.x).abs() < DROP_AHEAD_M + 0.3 && d.x > w.agent.x, "ahead of the robot: {} vs {}", d.x, w.agent.x);
        // A tracked body cannot bend: it lets go from hand height and the piece falls the rest.
        assert!(ev.iter().any(|e| matches!(&e.event, WorldEvent::Act { kind, .. } if kind == "impact")));
    }

    #[test]
    fn a_stone_breaks_in_half_and_the_half_is_easier_to_hold() {
        let mut w = world(FingerCount::Two, BaseKind::HumanFoot);
        w.agent.yaw = 0.0;
        w.drops.push(DropItem::whole(700, ItemKind::Stone, w.agent.x + 0.4, w.agent.z));
        w.drops.push(DropItem::whole(701, ItemKind::Log, w.agent.x - 0.4, w.agent.z));
        assert_eq!(w.command(Action::Split { drop: 701, portion: 0.5 }), Err(ActionError::NotDivisible));
        assert!(matches!(w.command(Action::Split { drop: 700, portion: 0.05 }), Err(ActionError::PieceTooSmall { .. })));
        let whole = w.grasp_for(ItemKind::Stone, 1.0);
        let half = w.grasp_for(ItemKind::Stone, 0.5);
        assert!(half.margin > whole.margin, "half {} vs whole {}", half.margin, whole.margin);
        let t0 = w.time_s;
        w.command(Action::Split { drop: 700, portion: 0.5 }).unwrap();
        let ev = run(&mut w, 30.0);
        let split = ev.iter().find_map(|e| match &e.event { WorldEvent::ItemSplit { piece, amount, left, .. } => Some((piece.clone(), *amount, *left)), _ => None }).expect("split");
        assert!((split.1 - 0.5).abs() < 1e-5 && (split.2 - 0.5).abs() < 1e-5);
        // Two fingers strike at 2/5 of the full hand's rate.
        assert!(((w.time_s - t0) as f32 - w.split_time_s(ItemKind::Stone, 1.0)).abs() < 0.3);
        let piece: u32 = split.0[1..].parse().unwrap();
        assert!((w.drop_item(piece).unwrap().amount - 0.5).abs() < 1e-5);
        assert!((w.drop_item(700).unwrap().mass_kg() - 1.3).abs() < 1e-4);
        if half.stable {
            w.command(Action::pick(piece)).unwrap();
            run(&mut w, 20.0);
            assert!((w.carried_amount(ItemKind::Stone) - 0.5).abs() < 1e-5);
            assert_eq!(w.carried_count(ItemKind::Stone), 0, "half a stone is not a whole one");
        }
    }

    #[test]
    fn a_bus_servo_arm_wears_in_hours_and_a_qdd_arm_barely() {
        let mine = |plan: BodyPlan| {
            let mut w = world_with(plan, BaseKind::Track);
            mine_nearest(&mut w, NodeKind::Boulder);
            let s = w.body_status(None, None);
            (w.wear.get("shoulder_r").copied().unwrap_or(0.0), w.agent.strikes, s)
        };
        let (budget, strikes, s) = mine(BodyPlan::uniform(hand_core::Tier::Budget));
        let (mid, _, _) = mine(BodyPlan::default());
        assert!(strikes > 0 && budget > 0.0 && mid > 0.0);
        assert!(budget > 100.0 * mid, "spur-gear servos against a planetary QDD: {budget} vs {mid}");
        let arm = s.parts.iter().find(|p| p.id == "arm_r").unwrap();
        assert!(arm.detail.contains("수명") && arm.ratio < 1.0, "{}", arm.detail);
        let strain = s.effects.iter().find(|e| e.id == "strain").unwrap();
        assert!(strain.detail.contains("RMS"), "{}", strain.detail);
    }

    #[test]
    fn the_world_and_the_body_reset_on_their_own() {
        let mut w = world(FingerCount::Five, BaseKind::Track);
        let nodes = w.nodes.len();
        mine_nearest(&mut w, NodeKind::Tree);
        add_stock(&mut w.agent.carried, ItemKind::Log, 1.0);
        w.heat.insert("elbow_r".into(), 1.4);
        let (energy, strikes) = (w.agent.energy_used_j, w.agent.strikes);
        let worn = w.wear.get("shoulder_r").copied().unwrap_or(0.0);
        assert!(energy > 0.0 && strikes > 0 && w.nodes.len() < nodes && worn > 0.0);
        w.reset_world_keep_body().unwrap();
        assert_eq!(w.wear.get("shoulder_r"), Some(&worn), "the reducers keep their wear");
        assert_eq!(w.nodes.len(), nodes, "the tree is back");
        assert!(w.drops.is_empty() && w.agent.carried.is_empty());
        assert_eq!((w.agent.energy_used_j, w.agent.strikes), (energy, strikes), "the body keeps its wear");
        assert_eq!(w.heat.get("elbow_r"), Some(&1.4));
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        assert!((w.agent.x - camp.x).abs() < 1e-4);
        add_stock(&mut w.agent.carried, ItemKind::Coal, 1.0);
        w.agent.x += 2.0;
        w.reset_body();
        assert_eq!((w.agent.energy_used_j, w.agent.strikes), (0.0, 0));
        assert!(w.heat.is_empty() && w.wear.is_empty() && w.battery_ratio() == 1.0);
        assert_eq!(w.carried_count(ItemKind::Coal), 1, "the body reset keeps the cargo");
        assert!((w.agent.x - camp.x - 2.0).abs() < 1e-4, "and the place");
    }

    #[test]
    fn a_cloned_world_runs_on_its_own() {
        let mut a = world(FingerCount::Five, BaseKind::Track);
        let mut b = a.clone();
        mine_nearest(&mut b, NodeKind::Tree);
        assert!(b.nodes.len() < a.nodes.len() && b.tick > a.tick);
        a.tick();
        assert_eq!(a.tick, 1);
        assert_eq!(a.capacity_kg(), b.capacity_kg());
    }

    #[test]
    fn a_walking_mid_robot_carries_little_and_a_tracked_one_carries_more() {
        let foot = world_with(BodyPlan::default(), BaseKind::HumanFoot);
        let track = world_with(BodyPlan::default(), BaseKind::Track);
        assert!(foot.capacity_kg() < 6.0, "mid legs hold ~24 kg in total: {}", foot.capacity_kg());
        assert!(track.capacity_kg() > foot.capacity_kg());
        // The chest preset sets the battery: 324 Wh for mid.
        assert!((foot.battery_capacity_j() - 324.0 * 3600.0).abs() < 1.0);
        // Cargo makes the base slower and uses more power.
        let mut loaded = world_with(BodyPlan::default(), BaseKind::Track);
        let (v0, p0) = (loaded.drive_spec().max_speed_mps, loaded.drive_spec().power_w);
        add_stock(&mut loaded.agent.carried, ItemKind::IronOre, 3.0);
        assert!(loaded.drive_spec().power_w > p0);
        assert!(loaded.drive_spec().max_speed_mps <= v0);
        let s = loaded.body_status(None, None);
        assert!(s.balance.is_none() || s.balance.as_ref().unwrap().mass_kg > 0.0);
    }
}
