//! Carries a quest out in the world, one tick at a time.
//!
//! The driver never reads the world's object list to find what it should go
//! for. It looks (scan), keeps what the gates confirmed as landmarks, drives
//! up to one, and then touches: a contact sensor reports the nearest node in
//! reach, and breaking it tells which kind it really was. That outcome feeds
//! the SDS, which is how the gates learn from mistakes.
//!
//! Three things it is told rather than perceives, all assumptions of this
//! simulator: where the waypoints are (they are on the map), the terrain (a
//! surveyed site map: routes and lines of sight are planned on it, objects
//! are not on it), and which drops a node just released (the event says so,
//! and they land within reach).
//!
//! When nothing is in sight after a full turn, the next place to look from is
//! the reachable viewpoint that shows the most ground not seen yet
//! (`world_sim::explore`): behind a mesa or a ridge rather than the nearest
//! grid point. Before each pick-up the cargo decision (`crate::cargo`) says
//! whether to take the piece, make a trip with what is aboard first, or break
//! the piece and take part of it.
//!
//! The actions play their physics (`hand_core::act`), so they can fail the
//! way the real ones do, and the driver handles it the way a careful operator
//! would: a grasp that slips or misses is tried again (`MAX_GRASP_TRIES`), a
//! body that cannot bring its hand down to the ground pushes the item to the
//! destination with its bumper instead (`MAX_PUSH_TRIES` lost pushes), and a
//! fall is waited out and the move tried again (`MAX_FALLS`).
//!
//! Since v0.7 everything it decides is in its believed frame (its odometry
//! drifts, `world_sim::odometry`) and it does not look all the time
//! (`crate::vision`): it walks and acts on memories, looks on a pattern while
//! a memory is too unsure for the next action, glances down at a piece before
//! the hand goes, turns to look at one it cannot see, fixes its pose from a
//! waypoint's flag near the place it delivers to, walks on until its belief
//! says it has arrived, and steps closer when the hand does not find the pile.

use std::collections::HashMap;

use hand_quest::{QuestResult, State};
use hand_sds::{Level, Scope, Track};
use hand_search::{DropTarget, Plan};
use serde_json::json;
use world_sim::explore::{Coverage, Viewpoint};
use world_sim::nav::Goal;
use world_sim::world::{AMOUNT_EPS, DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{amount_text, Action, ActionError, ItemKind, NodeKind, TimedEvent, WorldEvent};

use hand_search::text::eul_reul;

use crate::cargo::{self, CargoAction, CargoCtx, CargoDecision};
use crate::vision::{gaze_to_see, memory_ok, Memory, EYE_LEAD_S};

/// Grasps of one piece the driver tries before it leaves the piece (est: an operator's retry policy).
pub(crate) const MAX_GRASP_TRIES: u32 = 3;
/// Pushes of one piece that may lose it before the driver leaves it (est).
pub(crate) const MAX_PUSH_TRIES: u32 = 3;
/// Falls in one quest before the driver gives up (est).
pub(crate) const MAX_FALLS: u32 = 3;
/// The hand finds a piece within this of where the robot believes it lies,
/// metres (est: the open hand's span over the spot).
pub(crate) const TOUCH_PIECE_M: f32 = 0.25;
/// A side pick's spot is reached when the robot stands within this of it,
/// metres (a cell centre is within 0.354 m of any point; the reach's own
/// shuffle closes the rest, `SIDE_SHUFFLE_MAX_M`).
pub(crate) const SIDE_SPOT_REACH_M: f32 = 0.36;
/// A move to a place of the map stops this far from it, metres: the hand's
/// reach of the pile less half a metre of slack.
pub(crate) const PLACE_REACH_M: f32 = DELIVER_REACH_M - 0.5;
/// How far from a faintly seen item the robot stands to look down at it,
/// metres: the head camera (1.14 m) then sees it about 44° below level,
/// inside the 12° mount plus the neck's 45° tilt (est).
pub(crate) const INSPECT_STAND_M: f32 = 1.2;
/// How far a parcel is recognised from the head camera, metres: SigLIP2 base
/// with the 6×4 tiles picks a 0.30 m parcel out of the grass up to 3 m away
/// with the head pitched down 30°, never at 4 m (tests/siglip_parcel.rs).
pub(crate) const SMALL_ITEM_SEE_M: f32 = 3.0;
/// Head tilt of the look down that sweeps the ground near the feet while
/// searching for a small item, degrees (the probe's best view at 2–3 m).
pub(crate) const GROUND_SWEEP_TILT_DEG: f32 = 30.0;
/// Moves the driver adds toward a place when the planned one ended short of
/// it in the robot's belief (est).
pub(crate) const MAX_PLACE_FIXES: u32 = 2;
use crate::perception::EntKind;
use crate::precision::LmState;
use crate::runtime::Runtime;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Phase {
    Locate,
    Approach,
    Mine,
    Collect,
    Carry,
    Deliver,
    Goto,
    Unload,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MoveFor {
    Approach(u32),
    Search,
    Drop,
    Carry,
    Goto,
    /// A round trip: out to the destination with the load that is aboard.
    Shuttle,
    /// A round trip: back to where the rest is lying.
    Return,
    /// Lining up behind a piece to push it.
    PushLineUp(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Act {
    Move(MoveFor),
    Turn,
    Mine(u32),
    Pick(u32),
    Deliver(ItemKind),
    Split(u32),
    Unload,
    Push(u32),
}

#[derive(Clone, Debug)]
pub(crate) struct TaskTarget {
    pub resource: String,
    pub item: ItemKind,
    pub wanted: u32,
    /// Items lifted into the bay so far (half a stone is 0.5).
    pub picked: f32,
    pub label_ko: String,
}

/// A piece being pushed to the destination: which one and how many pushes lost it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PushJob {
    pub drop: u32,
    pub tries: u32,
}

/// Where a round trip stands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TripStage {
    Out,
    Unload,
    Back,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Trip {
    /// Where it stood within reach of the rest (believed), and its odometer then.
    pub back: (f32, f32),
    pub back_odo_m: f32,
    pub stage: TripStage,
}

pub(crate) enum Flow {
    Continue,
    Done,
    Fail { code: String, msg: String },
}

fn fail(code: &str, msg: impl Into<String>) -> Flow {
    Flow::Fail { code: code.into(), msg: msg.into() }
}

pub(crate) struct Task {
    pub quest: String,
    pub kind: &'static str,
    pub targets: Vec<TaskTarget>,
    pub destination: Option<String>,
    pub cur: usize,
    pub phase: Phase,
    pub act: Option<Act>,
    pub done: Option<(bool, Option<String>)>,
    pub broken: Option<(String, NodeKind, Vec<String>)>,
    /// The piece the last split broke off.
    pub split_piece: Option<u32>,
    pub started_tick: u64,
    pub started_time_s: f64,
    pub started_energy_j: f64,
    pub act_started_s: f64,
    pub act_predicted_s: Option<f32>,
    pub act_distance0: f32,
    // locating
    pub locate_since_s: f64,
    pub scan_cooldown: u32,
    pub scans_here: u32,
    pub rotations: u32,
    pub relocations: u32,
    pub visited: Vec<(f32, f32)>,
    pub target_lm: Option<u32>,
    pub approach_tries: u32,
    pub approach_scan_cd: u32,
    /// What the camera has looked at during this quest (terrain occludes).
    pub coverage: Coverage,
    /// Heading to turn to after reaching a viewpoint.
    pub face_after: Option<f32>,
    /// Faint sightings of the target already walked to and looked down at.
    pub inspected: Vec<u32>,
    // collecting
    pub drops: Vec<u32>,
    pub drop_idx: usize,
    pub rounds: u32,
    pub pick_retry: bool,
    pub trip: Option<Trip>,
    /// Round trips made before the last carry.
    pub trips: u32,
    pub last_cargo: Option<(u32, String)>,
    // delivering
    pub deliver_idx: usize,
    // setting down (drop quests)
    pub drop_items: Vec<DropTarget>,
    pub unload_idx: usize,
    pub unloaded: Vec<(String, f32)>,
    pub unload_place: Option<String>,
    /// Times each quest step was entered (index as in the quest).
    pub attempts: Vec<u32>,
    // result
    pub touch_ok: u32,
    pub touch_miss: u32,
    pub wasted: u32,
    pub delivered: Vec<(String, f32)>,
    // the actions' physics
    /// Failed grasps of the current piece.
    pub grasp_tries: u32,
    pub falls: u32,
    pub push: Option<PushJob>,
    /// Pieces that reached the destination by pushing.
    pub pushed: Vec<(String, f32)>,
    // memory and looking (`vision`)
    /// Where the robot believes each piece it knows of lies (its believed frame).
    pub mem: HashMap<u32, Memory>,
    /// Ticks to the pattern's next look while approaching.
    pub look_cd: u32,
    /// The look just before the hand acts has been taken on this approach.
    pub pre_looked: bool,
    /// A pick from the side is planned for this piece, facing this (believed) heading.
    pub side: Option<(u32, f32)>,
    /// Looks the vision schedule took, and actions taken on memory alone.
    pub looks: u32,
    pub blind_acts: u32,
    /// The last pick went on memory alone (no look at the piece from where it stood).
    pub last_pick_blind: bool,
    /// Moves added toward the place after the planned one ended short of it (`MAX_PLACE_FIXES`).
    pub place_fixes: u32,
    /// The hand did not find the pile once on this delivery: it looked and walked up close.
    pub deliver_retry: bool,
    /// The piece the robot turned to so that it could look at it (once per piece).
    pub look_turn: Option<u32>,
}

fn add_amount(list: &mut Vec<(String, f32)>, id: &str, a: f32) {
    match list.iter_mut().find(|(k, _)| k == id) {
        Some(e) => e.1 += a,
        None => list.push((id.to_string(), a)),
    }
}

fn amount_of(list: &[(String, f32)], id: &str) -> f32 {
    list.iter().filter(|(k, _)| k == id).map(|(_, a)| *a).sum()
}

impl Task {
    pub(crate) fn from_plan(quest: &str, plan: &Plan, world: &world_sim::World) -> Result<Task, (String, String)> {
        let mut targets = Vec::new();
        let mut drop_items = Vec::new();
        let (kind, destination) = match plan {
            Plan::Fetch { targets: ts, destination } => {
                for t in ts {
                    targets.push(make_target(t)?);
                }
                ("fetch", Some(destination.place.clone()))
            }
            Plan::Collect { targets: ts } => {
                for t in ts {
                    targets.push(make_target(t)?);
                }
                ("collect", None)
            }
            Plan::Goto { place } => ("goto", Some(place.place.clone())),
            Plan::Drop { items, place } => {
                for t in items {
                    ItemKind::from_id(&t.resource).ok_or_else(|| ("unknown_target".to_string(), format!("'{}'은(는) 이 월드에 없는 물건입니다", t.resource)))?;
                }
                drop_items = items.clone();
                ("drop", place.as_ref().map(|p| p.place.clone()))
            }
            Plan::Stop => return Err(("stop".into(), "중지 명령은 퀘스트가 아닙니다".into())),
        };
        if let Some(d) = &destination {
            if world.waypoint_by_key(d).is_none() {
                return Err(("unknown_target".into(), format!("지도에 '{d}' 지점이 없습니다")));
            }
        }
        let phase = match kind {
            "goto" => Phase::Goto,
            "drop" if destination.is_some() => Phase::Goto,
            "drop" => Phase::Unload,
            _ => Phase::Locate,
        };
        let n_steps = match kind {
            "fetch" => 4 * targets.len() + 2,
            "collect" => 4 * targets.len(),
            "drop" if destination.is_some() => 2,
            _ => 1,
        };
        Ok(Task {
            quest: quest.to_string(),
            kind,
            targets,
            destination,
            cur: 0,
            phase,
            act: None,
            done: None,
            broken: None,
            split_piece: None,
            started_tick: world.tick,
            started_time_s: world.time_s,
            started_energy_j: world.agent.energy_used_j,
            act_started_s: world.time_s,
            act_predicted_s: None,
            act_distance0: 0.0,
            locate_since_s: world.time_s,
            scan_cooldown: 0,
            scans_here: 0,
            rotations: 0,
            relocations: 0,
            visited: Vec::new(),
            target_lm: None,
            approach_tries: 0,
            approach_scan_cd: 0,
            coverage: Coverage::new(&world.terrain),
            face_after: None,
            inspected: Vec::new(),
            drops: Vec::new(),
            drop_idx: 0,
            rounds: 0,
            pick_retry: false,
            trip: None,
            trips: 0,
            last_cargo: None,
            deliver_idx: 0,
            drop_items,
            unload_idx: 0,
            unloaded: Vec::new(),
            unload_place: None,
            attempts: {
                let mut a = vec![0u32; n_steps];
                a[0] = 1;
                a
            },
            touch_ok: 0,
            touch_miss: 0,
            wasted: 0,
            delivered: Vec::new(),
            grasp_tries: 0,
            falls: 0,
            push: None,
            pushed: Vec::new(),
            mem: HashMap::new(),
            look_cd: 0,
            pre_looked: false,
            side: None,
            looks: 0,
            blind_acts: 0,
            last_pick_blind: false,
            place_fixes: 0,
            deliver_retry: false,
            look_turn: None,
        })
    }

    fn step_of(&self, phase: &Phase) -> usize {
        let n = self.targets.len();
        match phase {
            Phase::Locate => 4 * self.cur,
            Phase::Approach => 4 * self.cur + 1,
            Phase::Mine => 4 * self.cur + 2,
            Phase::Collect => 4 * self.cur + 3,
            Phase::Carry => 4 * n,
            Phase::Deliver => 4 * n + 1,
            Phase::Goto => 0,
            Phase::Unload => usize::from(self.destination.is_some()),
        }
    }

    fn target(&self) -> &TaskTarget {
        &self.targets[self.cur]
    }

    fn reset_locate(&mut self) {
        self.scan_cooldown = 0;
        self.scans_here = 0;
        self.rotations = 0;
        self.target_lm = None;
        self.approach_tries = 0;
        self.pre_looked = false;
        self.side = None;
    }

    /// The target lies where the world placed it (a parcel), not in a node.
    fn placed(&self) -> bool {
        !self.target().item.from_node()
    }

    fn precision(&self) -> Option<f64> {
        let n = self.touch_ok + self.touch_miss;
        (n > 0).then(|| self.touch_ok as f64 / n as f64)
    }
}

fn make_target(t: &hand_search::Target) -> Result<TaskTarget, (String, String)> {
    let item = ItemKind::from_id(&t.resource).ok_or_else(|| ("unknown_target".to_string(), format!("'{}'은(는) 이 월드에 없는 자원입니다", t.resource)))?;
    Ok(TaskTarget { resource: t.resource.clone(), item, wanted: t.quantity.max(1), picked: 0.0, label_ko: item.spec().label_ko.to_string() })
}

pub(crate) fn action_error_ko(e: &ActionError) -> String {
    match e {
        ActionError::Busy => "로봇이 다른 동작을 하고 있습니다".into(),
        ActionError::BatteryDepleted => "배터리가 방전되었습니다".into(),
        ActionError::UnknownTarget => "대상이 이미 사라졌거나 알 수 없습니다".into(),
        ActionError::NoPath => "갈 수 있는 경로가 없습니다 (이 베이스로는 지나갈 수 없는 지형)".into(),
        ActionError::OutOfReach { distance_m, reach_m } => format!("손이 닿지 않습니다 ({distance_m:.2} m > {reach_m:.2} m)"),
        ActionError::GraspUnstable { margin } => format!("이 손으로는 안정적으로 쥘 수 없습니다 (마찰 여유 {margin:.2})"),
        ActionError::CapacityExceeded { carried_kg, item_kg, capacity_kg } => {
            format!("적재 한도를 넘습니다 (실은 {carried_kg:.1} kg + {item_kg:.1} kg > {capacity_kg:.1} kg)")
        }
        ActionError::NothingToDeliver => "인도할 물건이 없습니다".into(),
        ActionError::LiftOverload { limiting, margin } => format!("팔·자세가 들 수 없습니다 ({limiting}, 균형 여유 {margin:.2})"),
        ActionError::NothingToDrop => "적재함에 내려놓을 물건이 없습니다".into(),
        ActionError::NotDivisible => "이 물건은 손으로 쪼갤 수 없습니다".into(),
        ActionError::PieceTooSmall { amount } => format!("조각이 너무 작아 더 쪼갤 수 없습니다 ({} 개)", amount_text(*amount)),
        ActionError::CannotReach { reason, .. } => format!("바닥에 손이 닿지 않습니다: {reason}"),
        ActionError::CannotPush { reason } => format!("밀 수 없습니다: {reason}"),
        ActionError::Fallen => "넘어져 일어나는 중입니다".into(),
    }
}

fn parse_id(s: &str) -> Option<u32> {
    s.get(1..)?.parse().ok()
}

impl Runtime {
    /// Starts the next pending quest, if nothing is running.
    pub(crate) fn begin_next(&mut self) {
        let Some(id) = self.quests.current_id() else { return };
        let Some(q) = self.quests.get(&id) else { return };
        let (state, plan) = (q.state, q.plan.clone());
        if state == State::Active {
            // Active without a task can only be a leftover; end it.
            if let Some(q) = self.quests.get_mut(&id) {
                let _ = q.fail("lost_task", "실행 정보를 잃어 중단했습니다", None);
            }
            self.log("quest", Level::Warn, format!("퀘스트 {id}: 실행 정보를 잃어 중단"), None);
            return;
        }
        match Task::from_plan(&id, &plan, &self.world) {
            Ok(task) => {
                let tick = self.world.tick;
                if let Some(q) = self.quests.get_mut(&id) {
                    let _ = q.start(tick);
                }
                let title = self.quests.get(&id).map(|q| q.title.clone()).unwrap_or_default();
                self.log("quest", Level::Info, format!("퀘스트 {id} 시작: {title}"), Some(json!({"quest": id, "tick": tick})));
                self.task = Some(task);
            }
            Err((code, msg)) => {
                if let Some(q) = self.quests.get_mut(&id) {
                    let _ = q.fail(&code, &msg, None);
                }
                self.log("quest", Level::Warn, format!("퀘스트 {id} 시작 실패: {msg}"), Some(json!({"quest": id, "code": code})));
            }
        }
    }

    /// One tick of the driver. Called after the world has stepped.
    pub(crate) fn drive(&mut self, events: &[TimedEvent]) {
        if self.pass == "ml" {
            self.ml_drive(events);
            return;
        }
        if self.task.is_none() {
            self.begin_next();
        }
        let Some(mut t) = self.task.take() else {
            self.apply_fix();
            self.focus = None;
            return;
        };
        self.take_fix(&mut t);
        self.focus = t.targets.get(t.cur).map(|x| x.resource.clone());
        match self.step_task(&mut t, events) {
            Flow::Continue => self.task = Some(t),
            Flow::Done => self.finish_task(t),
            Flow::Fail { code, msg } => self.fail_task(t, &code, &msg),
        }
    }

    fn step_task(&mut self, t: &mut Task, events: &[TimedEvent]) -> Flow {
        for te in events {
            match &te.event {
                WorldEvent::ActionDone { ok, reason, .. } => t.done = Some((*ok, reason.clone())),
                WorldEvent::NodeBroken { node, kind, drops } => t.broken = Some((node.clone(), *kind, drops.clone())),
                WorldEvent::ItemPicked { item, amount, .. } => {
                    if let Some(x) = t.targets.iter_mut().find(|x| x.item == *item) {
                        x.picked += *amount;
                    }
                }
                WorldEvent::Delivered { item, amount, .. } => add_amount(&mut t.delivered, item.id(), *amount),
                WorldEvent::ItemDropped { item, amount, waypoint, .. } => {
                    add_amount(&mut t.unloaded, item.id(), *amount);
                    t.unload_place = waypoint.clone();
                }
                WorldEvent::ItemSplit { piece, .. } => t.split_piece = parse_id(piece),
                WorldEvent::BatteryDepleted => return fail("battery_depleted", "배터리가 방전되어 멈췄습니다"),
                _ => {}
            }
        }
        if let Some(act) = t.act.clone() {
            return match t.done.take() {
                Some((ok, reason)) => {
                    t.act = None;
                    self.on_act_done(t, act, ok, reason)
                }
                None => {
                    self.while_running(t, &act);
                    Flow::Continue
                }
            };
        }
        // After a fall the robot gets up before anything else.
        if self.world.fallen_until.is_some_and(|f| self.world.time_s < f) {
            return Flow::Continue;
        }
        self.next_action(t)
    }

    // ----- quest bookkeeping -----

    fn goto_phase(&mut self, t: &mut Task, to: Phase, note: &str) {
        let (from_idx, to_idx) = (t.step_of(&t.phase), t.step_of(&to));
        if let Some(q) = self.quests.get_mut(&t.quest) {
            if to_idx > from_idx {
                for i in from_idx..to_idx {
                    let _ = q.complete_step(i);
                }
            } else if to_idx < from_idx {
                let _ = q.rewind(to_idx, note);
            } else if !note.is_empty() {
                let _ = q.set_progress(to_idx, 0.0, Some(note.to_string()));
            }
        }
        if from_idx != to_idx {
            if let Some(a) = t.attempts.get_mut(to_idx) {
                *a += 1;
            }
        }
        if !note.is_empty() || from_idx != to_idx {
            let label = self.quests.get(&t.quest).and_then(|q| q.steps.get(to_idx)).map(|s| s.label_ko.clone()).unwrap_or_default();
            let level = if to_idx < from_idx { Level::Warn } else { Level::Info };
            self.log("quest", level, format!("{} · 단계 {}: {}{}", t.quest, to_idx + 1, label, if note.is_empty() { String::new() } else { format!(" ({note})") }), None);
        }
        t.phase = to;
    }

    fn progress(&mut self, t: &Task, progress: f32, note: Option<String>) {
        let idx = t.step_of(&t.phase);
        if let Some(q) = self.quests.get_mut(&t.quest) {
            let _ = q.set_progress(idx, progress, note);
        }
    }

    // ----- deciding -----

    fn next_action(&mut self, t: &mut Task) -> Flow {
        match t.phase {
            Phase::Locate => self.do_locate(t),
            Phase::Approach => self.do_approach(t),
            Phase::Mine => self.do_mine(t),
            Phase::Collect => self.do_collect(t),
            Phase::Carry => self.do_carry(t),
            Phase::Deliver => self.do_deliver(t),
            Phase::Goto => self.do_goto(t),
            Phase::Unload => self.do_unload(t),
        }
    }

    /// A fix of the pose from the last step's looks: the pose and the
    /// landmarks move (`Runtime::apply_fix`), and so do the task's memories
    /// and the place a round trip comes back to.
    fn take_fix(&mut self, t: &mut Task) {
        let Some(f) = self.apply_fix() else { return };
        let rate = self.drift_rate();
        for m in t.mem.values_mut() {
            (m.x, m.z) = f.shift(m.x, m.z, m.odo_m, rate);
        }
        if let Some(tr) = t.trip.as_mut() {
            tr.back = f.shift(tr.back.0, tr.back.1, tr.back_odo_m, rate);
        }
    }

    fn issue(&mut self, t: &mut Task, action: Action, act: Act) -> Result<(), ActionError> {
        match self.world.command(self.to_world(action.clone())) {
            Ok(ev) => {
                self.log_world_events(&ev);
                t.act = Some(act);
                t.done = None;
                t.act_started_s = self.world.time_s;
                Ok(())
            }
            Err(e) => {
                self.log("world", Level::Warn, format!("동작 거부 {}: {}", action.label(), action_error_ko(&e)), Some(json!({"action": action.label(), "code": e.code()})));
                Err(e)
            }
        }
    }

    fn best_landmark(&self, t: &Task) -> Option<u32> {
        let want = &t.target().resource;
        // A mined item comes from a node it sees; a placed one is what it sees lying there.
        let kind = if t.placed() { EntKind::Drop } else { EntKind::Source };
        let (ax, az, _) = self.me();
        self.gates
            .landmarks()
            .iter()
            .filter(|l| l.confirmed && l.state == LmState::Active && l.kind == kind && &l.link == want)
            .min_by(|a, b| a.distance_to(ax, az).total_cmp(&b.distance_to(ax, az)))
            .map(|l| l.id)
    }

    /// Marks what the camera sees from here into the quest's coverage.
    pub(crate) fn mark_coverage(&self, cov: &mut Coverage) {
        self.mark_coverage_as(cov, 0.0, world_sim::explore::SEARCH_RANGE_M);
    }

    /// Marks the ground the camera saw with the head pitched down `tilt_deg`,
    /// as far as `range_m`: past that the target would not have been
    /// recognised even in view, so that ground is still unsearched.
    pub(crate) fn mark_coverage_as(&self, cov: &mut Coverage, tilt_deg: f32, range_m: f32) {
        let mut eye = self.cam.eye(&self.world);
        eye.pitch_deg -= tilt_deg;
        eye.range_m = eye.range_m.min(range_m);
        cov.look(&self.world.terrain, &eye, self.world.agent.yaw, self.cam.fov_deg);
    }

    /// How far the target can be recognised, and the head tilt of the sweep
    /// that searches the ground near the feet: a small item lying on the
    /// ground (a parcel) only within [`SMALL_ITEM_SEE_M`], with a look down.
    fn search_reach(&self, t: &Task) -> (f32, f32) {
        if t.placed() {
            (SMALL_ITEM_SEE_M, GROUND_SWEEP_TILT_DEG)
        } else {
            (world_sim::explore::SEARCH_RANGE_M, 0.0)
        }
    }

    /// The next viewpoint for a search (see `world_sim::explore`): the
    /// camera's level view and its look down together, as far as the target
    /// can be recognised.
    pub(crate) fn search_viewpoint(&self, t: &Task) -> Option<Viewpoint> {
        let (range, tilt) = self.search_reach(t);
        self.viewpoint_with(&t.coverage, &t.visited, tilt, range)
    }

    /// The next viewpoint with the level camera at its full range (the ML pass's search).
    pub(crate) fn search_viewpoint_for(&self, cov: &Coverage, avoid: &[(f32, f32)]) -> Option<Viewpoint> {
        self.viewpoint_with(cov, avoid, 0.0, world_sim::explore::SEARCH_RANGE_M)
    }

    fn viewpoint_with(&self, cov: &Coverage, avoid: &[(f32, f32)], tilt: f32, range: f32) -> Option<Viewpoint> {
        let blocked = self.world.blocked_cells();
        let start = self.world.terrain.cell_of(self.world.agent.x, self.world.agent.z)?;
        world_sim::explore::next_viewpoint(&self.world.terrain, &self.world.drive_spec(), &|x, z| blocked.contains(&(x, z)), start, cov, self.world.standing_eye_m(), self.cam.pitch_deg - 0.5 * tilt, self.cam.vfov_deg() + tilt, range, avoid)
    }

    fn do_locate(&mut self, t: &mut Task) -> Flow {
        if self.world.time_s - t.locate_since_s > self.cfg.locate_timeout_s {
            let name = t.target().label_ko.clone();
            return fail("locate_timeout", format!("{name}{} {:.0}초 안에 찾지 못했습니다", eul_reul(&name), self.cfg.locate_timeout_s));
        }
        if let Some(id) = self.best_landmark(t) {
            let (label, x, z, score) = self.gates.landmark(id).map(|l| (l.label.clone(), l.x, l.z, l.score)).unwrap();
            t.target_lm = Some(id);
            t.approach_tries = 0;
            t.approach_scan_cd = self.cfg.approach_scan_every_ticks;
            self.log(
                "quest",
                Level::Info,
                format!("{} 찾음: 랜드마크 L{id} {label} @ ({x:.1}, {z:.1}) 점수 {score:.2}", t.target().label_ko),
                Some(json!({"landmark": id, "label": label, "x": x, "z": z, "score": score})),
            );
            self.goto_phase(t, Phase::Approach, "");
            return Flow::Continue;
        }
        if t.placed() {
            if let Some(flow) = self.inspect_faint(t) {
                return flow;
            }
        }
        if t.scan_cooldown > 0 {
            t.scan_cooldown -= 1;
            return Flow::Continue;
        }
        let need = self.cfg.gates.confirm_sightings.max(1);
        if t.scans_here < need {
            if let Err(e) = self.scan_now("locate") {
                self.log("vision", Level::Error, format!("인식 실패: {e}"), None);
                return fail("perception_failed", format!("인식기가 실패했습니다: {e}"));
            }
            let (range, tilt) = self.search_reach(t);
            self.mark_coverage_as(&mut t.coverage, 0.0, range);
            // A small item: the same heading again with the head pitched down over the ground near the feet.
            if tilt > 0.0 && self.scan_gaze("locate_down", 0.0, tilt).is_ok() {
                self.mark_coverage_as(&mut t.coverage, tilt, range);
            }
            t.scans_here += 1;
            t.scan_cooldown = self.cfg.scan_every_ticks;
            return Flow::Continue;
        }
        // Looked here enough: turn, or go somewhere else.
        let rot_max = ((360.0 / self.cfg.rotate_deg.max(15.0)).ceil() as u32).saturating_sub(1);
        if t.rotations < rot_max {
            let yaw = self.world.agent.yaw.to_degrees() + self.cfg.rotate_deg;
            t.rotations += 1;
            t.scans_here = 0;
            let note = format!("탐색 중 · 회전 {}/{}회 · 이동 {}회 · 지형 {:.0}% 확인", t.rotations, rot_max, t.relocations, t.coverage.land_frac(&self.world.terrain) * 100.0);
            self.progress(t, t.rotations as f32 / (rot_max + 1) as f32, Some(note));
            return match self.issue(t, Action::Turn { yaw_deg: yaw }, Act::Turn) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        // A full turn without finding it: go where the most unseen ground is in view.
        t.visited.push((self.world.agent.x, self.world.agent.z));
        t.relocations += 1;
        if t.relocations > self.cfg.max_relocations {
            let name = t.target().label_ko.clone();
            return fail("locate_timeout", format!("{name}{} 찾지 못했습니다 (장소 {}곳을 둘러봄)", eul_reul(&name), t.relocations - 1));
        }
        t.rotations = 0;
        t.scans_here = 0;
        loop {
            let Some(v) = self.search_viewpoint(t) else {
                let name = t.target().label_ko.clone();
                let seen = t.coverage.land_frac(&self.world.terrain) * 100.0;
                return fail("locate_timeout", format!("{name}{} 찾지 못했습니다 (갈 수 있는 곳에서 볼 수 있는 땅을 다 봄 · {seen:.0}%)", eul_reul(&name)));
            };
            t.visited.push((v.x, v.z));
            t.face_after = Some(v.face_yaw);
            self.log(
                "quest",
                Level::Info,
                format!("{} 탐색: 시점 ({:.1}, {:.1})로 이동 — 안 본 곳 {}칸, 약 {:.0}초", t.target().label_ko, v.x, v.z, v.gain, v.time_s),
                Some(json!({"viewpoint": [v.x, v.z], "gain": v.gain, "time_s": v.time_s, "seen_frac": t.coverage.land_frac(&self.world.terrain)})),
            );
            match self.issue(t, Action::MoveTo { x: v.x, z: v.z, reach_m: 0.0 }, Act::Move(MoveFor::Search)) {
                Ok(()) => return Flow::Continue,
                Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => continue,
                Err(e) => return fail(e.code(), action_error_ko(&e)),
            }
        }
    }

    /// Something that looked like the wanted item from afar but did not pass
    /// the gates (a parcel a few pixels wide in a tile of grass) is checked as
    /// a person would: walk to about [`INSPECT_STAND_M`] from it, pitch and
    /// turn the head down at it so it fills the view, and look again. The
    /// close look confirms it (the next step approaches it) or leaves it
    /// unconfirmed, and the search goes on. `None` when there is nothing faint
    /// left to check.
    fn inspect_faint(&mut self, t: &mut Task) -> Option<Flow> {
        let want = t.target().resource.clone();
        let (mx, mz, myaw) = self.me();
        let (id, label, x, z, score) = self
            .gates
            .landmarks()
            .iter()
            .filter(|l| !l.confirmed && l.state == LmState::Active && l.kind == EntKind::Drop && l.link == want && l.seen >= 1 && !t.inspected.contains(&l.id))
            .min_by(|a, b| a.distance_to(mx, mz).total_cmp(&b.distance_to(mx, mz)))
            .map(|l| (l.id, l.label.clone(), l.x, l.z, l.score))?;
        let name = t.target().label_ko.clone();
        let d = (x - mx).hypot(z - mz);
        if d > INSPECT_STAND_M + 0.4 {
            self.log("quest", Level::Info, format!("멀리서 {name}처럼 보이는 것을 봄 (L{id} {label}, 점수 {score:.2}, {d:.1} m) — 가까이 가서 내려다보며 확인"), Some(json!({"landmark": id, "x": x, "z": z, "score": score})));
            t.face_after = Some((z - mz).atan2(x - mx));
            return match self.issue(t, Action::MoveTo { x, z, reach_m: INSPECT_STAND_M }, Act::Move(MoveFor::Search)) {
                Ok(()) => Some(Flow::Continue),
                Err(_) => {
                    t.inspected.push(id);
                    None
                }
            };
        }
        // Here: the head pitched down (and turned) until the spot is in the middle of the view.
        let b = (z - mz).atan2(x - mx) - myaw;
        let bearing = b.sin().atan2(b.cos()).to_degrees();
        let pan = (-bearing).clamp(-hand_core::kin::NECK_PAN_MAX_DEG, hand_core::kin::NECK_PAN_MAX_DEG);
        let depression = self.world.standing_eye_m().atan2(d.max(0.1)).to_degrees();
        let tilt = (depression + self.cam.pitch_deg).clamp(0.0, hand_core::kin::NECK_PITCH_MAX_DEG);
        t.inspected.push(id);
        for _ in 0..self.cfg.gates.confirm_sightings.max(1) {
            if self.scan_gaze("inspect", pan, tilt).is_err() {
                break;
            }
        }
        t.looks += 1;
        let ok = self.gates.landmark(id).is_some_and(|l| l.confirmed);
        let msg = if ok {
            format!("내려다보고 확인: L{id}는 {name} (목 숙임 {tilt:.0}°, 돌림 {pan:.0}°)")
        } else {
            format!("내려다봤지만 L{id}는 {name}로 확인되지 않음 (목 숙임 {tilt:.0}°) — 다른 곳을 찾음")
        };
        self.log("quest", Level::Info, msg, Some(json!({"landmark": id, "confirmed": ok, "tilt_deg": tilt, "pan_deg": pan})));
        self.note_look("inspect", format!("L{id}"), None, None, None);
        Some(Flow::Continue)
    }

    fn do_approach(&mut self, t: &mut Task) -> Flow {
        let Some(id) = t.target_lm else {
            self.goto_phase(t, Phase::Locate, "랜드마크를 잃음");
            t.reset_locate();
            return Flow::Continue;
        };
        let Some((x, z)) = self.gates.landmark(id).filter(|l| l.state == LmState::Active).map(|l| (l.x, l.z)) else {
            self.goto_phase(t, Phase::Locate, "랜드마크가 배제됨");
            t.reset_locate();
            return Flow::Continue;
        };
        let dist = self.dist_me(x, z);
        if t.placed() {
            return self.approach_placed(t, id, (x, z), dist);
        }
        if dist <= MINE_REACH_M - 0.1 {
            self.goto_phase(t, Phase::Mine, "");
            return Flow::Continue;
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: self.cfg.approach_reach_m }).map(|p| p.time_s);
        t.act_distance0 = dist;
        t.look_cd = self.cfg.vision.look_every_ticks;
        t.pre_looked = false;
        match self.issue(t, Action::MoveTo { x, z, reach_m: self.cfg.approach_reach_m }, Act::Move(MoveFor::Approach(id))) {
            Ok(()) => Flow::Continue,
            Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => {
                self.gates.note_unreachable(id);
                self.goto_phase(t, Phase::Locate, "경로 없음, 다른 대상을 찾음");
                t.reset_locate();
                Flow::Continue
            }
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    /// The node the hand feels, nearest to where the target is believed to be
    /// (the hand goes where the robot believes it is: its believed frame
    /// carried into the world, `World::true_of`).
    fn touch_node(&self, believed: (f32, f32)) -> Option<u32> {
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        let at = self.world.true_of(believed.0, believed.1);
        self.world
            .nodes
            .iter()
            .filter(|n| ((n.x - ax).powi(2) + (n.z - az).powi(2)).sqrt() <= MINE_REACH_M)
            .min_by(|a, b| {
                let da = (a.x - at.0).powi(2) + (a.z - at.1).powi(2);
                let db = (b.x - at.0).powi(2) + (b.z - at.1).powi(2);
                da.total_cmp(&db)
            })
            .map(|n| n.id)
    }

    /// The piece of `kind` the hand feels where the robot believes one lies:
    /// within the hand's reach of the robot and `TOUCH_PIECE_M` of the spot.
    pub(crate) fn touch_piece(&self, believed: (f32, f32), kind: ItemKind) -> Option<u32> {
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        let at = self.world.true_of(believed.0, believed.1);
        self.world
            .drops
            .iter()
            .filter(|d| d.kind == kind && (d.x - ax).hypot(d.z - az) <= PICKUP_REACH_M && (d.x - at.0).hypot(d.z - at.1) <= TOUCH_PIECE_M)
            .min_by(|a, b| (a.x - at.0).hypot(a.z - at.1).total_cmp(&(b.x - at.0).hypot(b.z - at.1)))
            .map(|d| d.id)
    }

    /// Approaching an item that lies where the world placed it (a parcel):
    /// to a spot beside it where the right hand reaches it without turning
    /// (on the way to the destination), or in front of it. On arrival the hand
    /// touches it where memory says (`touch_piece`) and the pick follows.
    fn approach_placed(&mut self, t: &mut Task, id: u32, at: (f32, f32), dist: f32) -> Flow {
        let item = t.target().item;
        if t.side.is_none() && dist > PICKUP_REACH_M {
            // Facing on toward where it goes next; a side reach the body can make.
            let (mx, mz, myaw) = self.me();
            let heading = match self.waypoint_of(t) {
                Some((_, wx, wz, _)) => (wz - at.1).atan2(wx - at.0),
                None => (at.1 - mz).atan2(at.0 - mx),
            };
            let _ = myaw;
            if let Some((ox, oy)) = self.world.side_reach(item, 1.0) {
                t.side = Some((id, heading));
                let (s, c) = heading.sin_cos();
                // The item at (forward ox, left oy) of the robot: it stands back and to the left of it.
                let spot = (at.0 - ox * c - oy * s, at.1 - ox * s + oy * c);
                t.act_distance0 = self.dist_me(spot.0, spot.1);
                t.look_cd = self.cfg.vision.look_every_ticks;
                t.pre_looked = false;
                self.log("quest", Level::Info, format!("{}: 지나가며 오른쪽으로 집음 — 옆 자리 ({:.1}, {:.1})로", t.target().label_ko, spot.0, spot.1), Some(json!({"landmark": id, "spot": [spot.0, spot.1], "heading_deg": heading.to_degrees()})));
                return match self.issue(t, Action::MoveTo { x: spot.0, z: spot.1, reach_m: SIDE_SPOT_REACH_M }, Act::Move(MoveFor::Approach(id))) {
                    Ok(()) => Flow::Continue,
                    Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => {
                        // No way to the side spot: from in front then.
                        t.side = None;
                        self.issue_front_approach(t, id, at)
                    }
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                };
            }
        }
        if t.side.is_some() || dist <= PICKUP_REACH_M - 0.1 {
            // Arrived: the hand feels for it (a side pick faces on first).
            self.goto_phase(t, Phase::Mine, "");
            return Flow::Continue;
        }
        self.issue_front_approach(t, id, at)
    }

    fn issue_front_approach(&mut self, t: &mut Task, id: u32, at: (f32, f32)) -> Flow {
        t.act_predicted_s = self.world.plan_route(Goal::Within { x: at.0, z: at.1, reach_m: 0.4 }).map(|p| p.time_s);
        t.act_distance0 = self.dist_me(at.0, at.1);
        t.look_cd = self.cfg.vision.look_every_ticks;
        t.pre_looked = false;
        match self.issue(t, Action::MoveTo { x: at.0, z: at.1, reach_m: 0.4 }, Act::Move(MoveFor::Approach(id))) {
            Ok(()) => Flow::Continue,
            Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => {
                self.gates.note_unreachable(id);
                self.goto_phase(t, Phase::Locate, "경로 없음, 다른 대상을 찾음");
                t.reset_locate();
                Flow::Continue
            }
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    /// The hand feels for a placed item where memory says it is: found, it is
    /// the piece to pick; not there, the memory was wrong (look again, or rule
    /// the sighting out).
    fn touch_placed(&mut self, t: &mut Task) -> Flow {
        let Some(id) = t.target_lm else {
            self.goto_phase(t, Phase::Locate, "랜드마크를 잃음");
            t.reset_locate();
            return Flow::Continue;
        };
        let Some(l) = self.gates.landmark(id).cloned() else {
            self.goto_phase(t, Phase::Locate, "랜드마크가 배제됨");
            t.reset_locate();
            return Flow::Continue;
        };
        // A pick in passing: face on the way it goes next, the piece beside the right foot.
        if let Some((_, heading)) = t.side {
            let err = (heading - self.me().2 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            if err.abs() > 0.15 {
                return match self.issue(t, Action::Turn { yaw_deg: heading.to_degrees() }, Act::Turn) {
                    Ok(()) => Flow::Continue,
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                };
            }
        }
        let item = t.target().item;
        match self.touch_piece((l.x, l.z), item) {
            Some(piece) => {
                t.side = t.side.map(|(_, h)| (piece, h));
                let vscope = Scope::new(Track::Vision, &l.label, "");
                self.sds.record_agreement(&vscope, "touch", true);
                self.sds.record_baseline(&vscope, "score", l.score as f64);
                t.touch_ok += 1;
                t.mem.insert(piece, l.memory());
                self.gates.consume(id);
                t.drops = vec![piece];
                t.drop_idx = 0;
                t.pick_retry = false;
                self.log("vision", Level::Info, format!("손으로 확인: {} 맞음 (d{piece})", l.label), Some(json!({"landmark": id, "drop": piece})));
                self.goto_phase(t, Phase::Collect, "");
                Flow::Continue
            }
            None => {
                t.approach_tries += 1;
                if t.approach_tries < 2 {
                    self.log("quest", Level::Warn, format!("기억한 자리에 {}이 손에 닿지 않음 — 다시 보고 접근 ({}/2)", t.target().label_ko, t.approach_tries), None);
                    let _ = self.scan_now("retouch");
                    t.side = None;
                    self.goto_phase(t, Phase::Approach, "위치 재확인");
                    return Flow::Continue;
                }
                let vscope = Scope::new(Track::Vision, &l.label, "");
                self.sds.record_agreement(&vscope, "touch", false);
                self.gates.reject(id);
                t.touch_miss += 1;
                t.wasted += 1;
                self.goto_phase(t, Phase::Locate, "헛감지였음, 다시 탐색");
                t.reset_locate();
                Flow::Continue
            }
        }
    }

    fn do_mine(&mut self, t: &mut Task) -> Flow {
        if t.placed() {
            return self.touch_placed(t);
        }
        let believed = t.target_lm.and_then(|id| self.gates.landmark(id)).map(|l| (l.x, l.z)).unwrap_or_else(|| {
            let (x, z, _) = self.me();
            (x, z)
        });
        match self.touch_node(believed) {
            Some(node) => {
                let (hp, rate) = self.world.node(node).map(|n| (n.hp, self.world.mining_rate())).unwrap_or((0.0, 1.0));
                t.act_predicted_s = Some(hp / rate.max(0.05));
                match self.issue(t, Action::Mine { node }, Act::Mine(node)) {
                    Ok(()) => {
                        self.progress(t, 0.0, Some("채취 중".into()));
                        Flow::Continue
                    }
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                }
            }
            None => {
                // Nothing within reach: the position was off, or nothing is there.
                t.approach_tries += 1;
                let id = t.target_lm;
                if t.approach_tries < 2 {
                    self.log("quest", Level::Warn, format!("손이 닿는 곳에 아무것도 없음 — 다시 보고 접근 ({}/2)", t.approach_tries), None);
                    let _ = self.scan_now("retouch");
                    self.goto_phase(t, Phase::Approach, "위치 재확인");
                    return Flow::Continue;
                }
                if let Some(id) = id {
                    if let Some(l) = self.gates.landmark(id).cloned() {
                        let vscope = Scope::new(Track::Vision, &l.label, "");
                        self.sds.record_agreement(&vscope, "touch", false);
                        self.sds.record_confusion(&vscope, &l.label, "background", l.margin as f64);
                        self.log(
                            "vision",
                            Level::Warn,
                            format!("헛감지: L{id} {} @ ({:.1}, {:.1})에는 아무것도 없었음", l.label, l.x, l.z),
                            Some(json!({"landmark": id, "label": l.label, "score": l.score, "margin": l.margin})),
                        );
                    }
                    self.gates.reject(id);
                }
                t.touch_miss += 1;
                t.wasted += 1;
                self.goto_phase(t, Phase::Locate, "헛감지였음, 다시 탐색");
                t.reset_locate();
                Flow::Continue
            }
        }
    }

    /// Mass of this quest's items in the bay (what a trip would deliver).
    fn quest_cargo_kg(&self, t: &Task) -> f32 {
        t.targets
            .iter()
            .map(|x| {
                let mine = (x.picked - amount_of(&t.delivered, &x.resource)).max(0.0);
                self.world.carried_amount(x.item).min(mine) * x.item.spec().mass_kg
            })
            .sum()
    }

    /// Logs a cargo decision once per piece and action, and records it in the SDS.
    fn note_cargo(&mut self, t: &mut Task, id: u32, dec: &CargoDecision) {
        let key = (id, format!("{}:{:.3}", dec.action.key(), dec.portion));
        if t.last_cargo.as_ref() == Some(&key) {
            return;
        }
        t.last_cargo = Some(key);
        let level = if dec.action == CargoAction::Take { Level::Debug } else { Level::Info };
        let gates: Vec<serde_json::Value> = dec.gates.iter().map(|g| json!({"id": g.id, "value": g.value, "limit": g.limit, "severity": g.severity, "action": g.action.key(), "message": g.message})).collect();
        self.log("quest", level, format!("적재 판단 · {} ({})", dec.summary_ko, dec.action.label_ko()), Some(json!({"drop": id, "action": dec.action, "rule": dec.rule_action, "gates": gates})));
        let scope = Scope::new(Track::Motion, "cargo", &dec.item);
        self.sds.record_baseline(&scope, "portion", dec.portion as f64);
        self.sds.record_agreement(&scope, "whole", dec.action == CargoAction::Take);
    }

    fn do_collect(&mut self, t: &mut Task) -> Flow {
        if let Some(flow) = self.trip_step(t) {
            return flow;
        }
        if t.push.is_some() {
            return self.push_step(t);
        }
        loop {
            let (need_more, name) = {
                let x = t.target();
                (x.picked + AMOUNT_EPS < x.wanted as f32, x.label_ko.clone())
            };
            if !need_more || t.drop_idx >= t.drops.len() {
                let (picked, wanted) = (t.target().picked, t.target().wanted);
                if picked + AMOUNT_EPS >= wanted as f32 {
                    return self.next_target(t);
                }
                t.rounds += 1;
                if t.rounds > self.cfg.max_rounds {
                    return fail("not_enough", format!("{name} {wanted}개 중 {}개만 모았습니다 (더 캘 곳을 찾지 못함)", amount_text(picked)));
                }
                self.goto_phase(t, Phase::Locate, &format!("{name} {}/{wanted}개, 더 찾음", amount_text(picked)));
                t.reset_locate();
                t.locate_since_s = self.world.time_s;
                return Flow::Continue;
            }
            let id = t.drops[t.drop_idx];
            let Some(d) = self.world.drop_item(id).cloned() else {
                t.drop_idx += 1;
                continue;
            };
            // Where the robot believes it lies: its memory (the piece's own place only as a last resort, when it never saw it).
            let m = *t.mem.entry(id).or_insert_with(|| {
                let (x, z) = self.world.belief_of(d.x, d.z);
                Memory::seen(x, z, self.world.aim_sigma_m(), self.world.odom.travelled_m, self.world.tick)
            });
            // A pick from the side starts where the robot stands (its own shuffle closes the rest).
            let side_here = t.side.is_some_and(|(sid, _)| sid == id);
            let dist = self.dist_me(m.x, m.z);
            if dist > PICKUP_REACH_M - 0.1 && !side_here {
                t.act_predicted_s = self.world.plan_route(Goal::Within { x: m.x, z: m.z, reach_m: 0.4 }).map(|p| p.time_s);
                // Up to the piece; where no route ends that close (it lies on a ledge, or the memory drifted onto one), as close as the hand's reach allows.
                let r = match self.issue(t, Action::MoveTo { x: m.x, z: m.z, reach_m: 0.4 }, Act::Move(MoveFor::Drop)) {
                    Err(ActionError::NoPath) => self.issue(t, Action::MoveTo { x: m.x, z: m.z, reach_m: PICKUP_REACH_M - 0.1 }, Act::Move(MoveFor::Drop)),
                    r => r,
                };
                return match r {
                    Ok(()) => Flow::Continue,
                    Err(ActionError::NoPath) => {
                        self.log("quest", Level::Warn, format!("떨어진 {name}에게 갈 경로가 없어 건너뜀"), None);
                        t.drop_idx += 1;
                        continue;
                    }
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                };
            }
            // A body that cannot bring its hand to the ground pushes the piece there instead (or cannot fetch it).
            if let Err(why) = self.world.reach_ground(d.kind, d.amount) {
                if t.destination.is_none() {
                    return fail("reach_ground", format!("{name}{} 집을 수 없습니다: {why}", eul_reul(&name)));
                }
                if self.world.body.legged {
                    return fail("reach_ground", format!("{name}{} 집을 수 없습니다: {why}", eul_reul(&name)));
                }
                self.log("quest", Level::Info, format!("{name}: 바닥에 손이 닿지 않아 범퍼로 밀어 옮김 — {why}"), Some(json!({"drop": id})));
                t.push = Some(PushJob { drop: id, tries: 0 });
                return self.push_step(t);
            }
            // The cargo decision: take it, make a trip first, or break it.
            let ctx = CargoCtx { quest_cargo_kg: self.quest_cargo_kg(t), destination: self.waypoint_of(t).map(|(_, x, z, _)| (x, z)) };
            let dec = cargo::decide(&self.world, &d, &ctx);
            self.note_cargo(t, id, &dec);
            match dec.action {
                CargoAction::Take => {}
                CargoAction::Split { portion } => {
                    let (picked, wanted) = (t.target().picked, t.target().wanted);
                    self.progress(t, picked / wanted as f32, Some(format!("{}개 · {name} 쪼개는 중 ({:.0}%만 들 수 있음)", amount_text(picked), portion * 100.0)));
                    return match self.issue(t, Action::Split { drop: id, portion }, Act::Split(id)) {
                        Ok(()) => Flow::Continue,
                        Err(e) => fail(e.code(), action_error_ko(&e)),
                    };
                }
                CargoAction::Shuttle => {
                    if t.trips >= self.cfg.max_trips {
                        return fail("too_many_trips", format!("{}회 왕복해도 다 나르지 못했습니다", t.trips));
                    }
                    t.trips += 1;
                    // It comes back to where it stands now, within the hand's reach of the rest.
                    let (sx, sz, _) = self.me();
                    t.trip = Some(Trip { back: (sx, sz), back_odo_m: self.world.odom.travelled_m, stage: TripStage::Out });
                    let dest = self.waypoint_of(t).map(|w| w.3).unwrap_or_default();
                    self.log("quest", Level::Info, format!("{} · 왕복 {}회차: 실은 짐을 {dest}에 내려놓고 {name}에게 돌아옴", t.quest, t.trips), Some(json!({"trip": t.trips, "carried_kg": self.world.carried_kg(), "capacity_kg": self.world.capacity_kg()})));
                    return self.trip_step(t).unwrap_or(Flow::Continue);
                }
                CargoAction::Refuse => return fail(dec.code.unwrap_or("lift_overload"), dec.summary_ko.clone()),
            }
            // Before the hand goes: is the memory good enough for it (2σ inside half the top face)?
            let pick = match self.pick_from_memory(t, id, d.kind, d.amount) {
                Ok(a) => a,
                Err(flow) => return flow,
            };
            return match self.issue(t, pick, Act::Pick(id)) {
                Ok(()) => Flow::Continue,
                Err(ActionError::CannotReach { reason, .. }) if t.side.is_some() => {
                    // The side reach does not hold: from in front, facing it.
                    self.log("quest", Level::Info, format!("옆으로는 못 집음 ({reason}) — 앞에서 집음"), None);
                    t.side = None;
                    Flow::Continue
                }
                Err(ActionError::OutOfReach { .. }) if !t.pick_retry => {
                    t.pick_retry = true;
                    t.side = None;
                    let m = t.mem.get(&id).copied().unwrap_or(m);
                    match self.issue(t, Action::MoveTo { x: m.x, z: m.z, reach_m: 0.2 }, Act::Move(MoveFor::Drop)) {
                        Ok(()) => Flow::Continue,
                        Err(ActionError::NoPath) => {
                            self.log("quest", Level::Warn, format!("{name}에 손이 닿지 않고 더 다가갈 길도 없어 건너뜀"), None);
                            t.pick_retry = false;
                            t.drop_idx += 1;
                            Flow::Continue
                        }
                        Err(e) => fail(e.code(), action_error_ko(&e)),
                    }
                }
                Err(ActionError::CapacityExceeded { carried_kg, item_kg, capacity_kg }) => {
                    let fit = ((capacity_kg - carried_kg + 1e-3) / item_kg).floor().max(0.0);
                    let picked = t.target().picked;
                    fail(
                        "capacity_exceeded",
                        format!("적재 한도를 넘습니다 — 이 베이스는 {name}{} {}개까지 실을 수 있습니다 (현재 {}개 실음)", eul_reul(&name), amount_text(picked + fit), amount_text(picked)),
                    )
                }
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
    }

    /// The pick of piece `id` from what the robot remembers of it: the hand
    /// goes where memory says, from the side (a pick in passing, the camera
    /// cannot see there) or in front. When the memory's 2σ is wider than half
    /// the piece's top face the robot looks first if it can see the spot from
    /// where it stands; a side pick it cannot look for becomes a front pick
    /// (it turns, and the turn shows it the piece).
    fn pick_from_memory(&mut self, t: &mut Task, id: u32, kind: ItemKind, amount: f32) -> Result<Action, Flow> {
        let rate = self.drift_rate();
        let odo = self.world.odom.travelled_m;
        let size = kind.spec().piece_size_m(amount);
        let tolerance = 0.5 * size[0].min(size[2]);
        let mut m = t.mem.get(&id).copied().ok_or(Flow::Continue)?;
        let mut sigma = m.sigma_m(odo, rate);
        let me = self.me();
        let side = t.side.is_some_and(|(sid, _)| sid == id);
        let intermittent = self.cfg.vision.intermittent;
        if !intermittent || !memory_ok(sigma, tolerance, &self.cfg.vision) {
            if let Some((pan, tilt)) = gaze_to_see(&self.cam, self.world.standing_eye_m(), me, m.x, m.z) {
                // A glance (the head turned to the spot, pitched down for the ground at the feet): it refreshes the memory from what the gates saw there.
                let before = sigma;
                if self.scan_gaze(if intermittent { "glance" } else { "pre_act" }, pan, tilt).is_ok() {
                    t.looks += 1;
                    self.vstats.looks += 1;
                    if let Some((fused, innov)) = self.refresh_memory(&m, kind, odo) {
                        m = fused;
                        t.mem.insert(id, m);
                        sigma = m.sigma_m(odo, rate);
                        self.note_look("glance", format!("d{id}"), Some(before), Some(sigma), Some(innov));
                    } else {
                        self.log("vision", Level::Debug, format!("응시했지만 d{id}가 보이지 않음 — 기억(±{:.0} cm)으로 집음", 2.0 * sigma * 100.0), None);
                    }
                }
            } else if intermittent && (side || t.look_turn != Some(id)) {
                // Too unsure for a reach it cannot look at (beside the foot, behind it): turn to it, look, and pick in front.
                self.log("quest", Level::Info, format!("기억이 흐림 (±{:.0} cm > {:.0} cm) — 돌아서 보고 앞에서 집음", 2.0 * sigma * 100.0, tolerance * 100.0), None);
                t.side = None;
                t.look_turn = Some(id);
                return Err(match self.issue(t, Action::Face { x: m.x, z: m.z }, Act::Turn) {
                    Ok(()) => Flow::Continue,
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                });
            }
        }
        let side = side && t.side.is_some();
        // Seen just now (a look from here found it and refreshed the memory), or on memory alone.
        let looked = m.tick == self.world.tick;
        let blind = side || !looked;
        if blind {
            t.blind_acts += 1;
            self.vstats.blind_acts += 1;
            self.note_look("blind", format!("d{id} {}", if side { "옆" } else { "앞" }), Some(sigma), None, None);
        }
        self.vstats.picks += 1;
        if side {
            self.vstats.side_picks += 1;
        }
        t.last_pick_blind = blind;
        Ok(Action::PickUp { drop: id, aim: Some((m.x, m.z)), side, memory_sigma_m: sigma })
    }

    /// On the way to a remembered piece (up to it, or back to it after a
    /// trip): the pattern's look while the memory is too unsure for the grasp
    /// (2σ wider than half the top face) and its spot is in view; a sighting
    /// of the piece refreshes the memory. With the pattern off it looks as
    /// often as on an approach.
    fn piece_look(&mut self, t: &mut Task) {
        let intermittent = self.cfg.vision.intermittent;
        if intermittent {
            self.vstats.approach_s += world_sim::TICK_S as f64;
        }
        if t.look_cd > 0 {
            t.look_cd -= 1;
            return;
        }
        let Some(&id) = t.drops.get(t.drop_idx) else { return };
        let (Some(m), Some((kind, amount))) = (t.mem.get(&id).copied(), self.world.drop_item(id).map(|d| (d.kind, d.amount))) else { return };
        let (rate, odo) = (self.drift_rate(), self.world.odom.travelled_m);
        let size = kind.spec().piece_size_m(amount);
        let sigma = m.sigma_m(odo, rate);
        if intermittent && memory_ok(sigma, 0.5 * size[0].min(size[2]), &self.cfg.vision) {
            return;
        }
        // The head turns to the piece when the level camera does not reach it.
        let Some((pan, tilt)) = gaze_to_see(&self.cam, self.world.standing_eye_m(), self.me(), m.x, m.z) else { return };
        t.look_cd = if intermittent { self.cfg.vision.look_every_ticks } else { self.cfg.approach_scan_every_ticks };
        if self.scan_gaze("pattern", pan, tilt).is_ok() {
            t.looks += 1;
            self.vstats.looks += 1;
            if intermittent {
                self.vstats.approach_looks += 1;
            }
            if let Some((fused, innov)) = self.refresh_memory(&m, kind, odo) {
                t.mem.insert(id, fused);
                self.log("vision", Level::Debug, format!("가는 길에 d{id}를 다시 봄: 기억이 {innov:.2} m 어긋나 있었음 (±{:.0} → ±{:.0} cm)", 2.0 * sigma * 100.0, 2.0 * fused.sigma_m(odo, rate) * 100.0), None);
                self.note_look("pattern", format!("d{id}"), Some(sigma), Some(fused.sigma_m(odo, rate)), Some(innov));
            }
        }
    }

    /// What the last scan saw of a remembered piece: the sighting of its kind
    /// nearest to the memory (within 3σ, at least 0.3 m) that the gates
    /// accepted or that holds by itself (`Judged::sound`: the piece is known
    /// to be there), fused with it.
    pub(crate) fn refresh_memory(&self, m: &Memory, kind: ItemKind, odo: f32) -> Option<(Memory, f32)> {
        let rate = self.drift_rate();
        let reach = (3.0 * m.sigma_m(odo, rate)).max(0.3);
        let scan = self.last_scan.as_ref()?;
        let mem = self.mem_ctx();
        let best = scan
            .judged
            .iter()
            .filter(|j| (j.accepted || j.sound()) && j.cand.kind == EntKind::Drop && j.cand.link == kind.id())
            .filter_map(|j| j.cand.world.map(|w| (w, j.cand.depth_m)))
            .filter(|((x, z), _)| (x - m.x).hypot(z - m.z) <= reach)
            .min_by(|a, b| (a.0 .0 - m.x).hypot(a.0 .1 - m.z).total_cmp(&(b.0 .0 - m.x).hypot(b.0 .1 - m.z)))?;
        let ((x, z), depth) = best;
        Some(m.fuse(x, z, mem.sigma_obs(depth), odo, self.world.tick, rate))
    }

    /// Advances a push: line up behind the piece, face the way, push it to the
    /// destination; when it lies there it goes onto the pile and counts as
    /// delivered. A push that lost the piece starts over from where it lies.
    fn push_step(&mut self, t: &mut Task) -> Flow {
        let Some(job) = t.push else { return Flow::Continue };
        let Some((wid, wx, wz, place)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let name = t.target().label_ko.clone();
        let Some(d) = self.world.drop_item(job.drop).cloned() else {
            t.push = None;
            t.drop_idx += 1;
            return Flow::Continue;
        };
        // The piece against the bumper: where it is relative to the robot is known by touch.
        let (px, pz) = self.world.belief_of(d.x, d.z);
        let mut stocked = false;
        if (px - wx).hypot(pz - wz) <= DELIVER_REACH_M {
            let mut ev = Vec::new();
            if let Some(a) = self.world.stock_pushed(job.drop, wid, &mut ev) {
                stocked = true;
                self.log_world_events(&ev);
                if let Some(x) = t.targets.iter_mut().find(|x| x.item == d.kind) {
                    x.picked += a;
                }
                add_amount(&mut t.delivered, d.kind.id(), a);
                add_amount(&mut t.pushed, d.kind.id(), a);
                self.log("quest", Level::Info, format!("{name}{} {place}까지 밀어 옮김", eul_reul(&name)), Some(json!({"drop": job.drop, "amount": a})));
            }
            if stocked {
                t.push = None;
                t.drop_idx += 1;
                let (picked, wanted) = (t.target().picked, t.target().wanted);
                self.progress(t, (picked / wanted as f32).min(1.0), Some(format!("{}/{wanted}개 (밀어서)", amount_text(picked))));
                return Flow::Continue;
            }
            // It believed the piece was there; the pile is not where it believed: push on.
        }
        let (tx, tz) = self.world.true_of(wx, wz);
        let (_, (lx, lz, yaw)) = match self.world.push_route(job.drop, tx, tz) {
            Ok(r) => r,
            Err(e) => return fail(e.code(), format!("{name}{} {place}까지 밀 수 없습니다: {}", eul_reul(&name), action_error_ko(&e))),
        };
        // The line-up behind it, in the robot's frame.
        let (lx, lz) = self.world.belief_of(lx, lz);
        let yaw = yaw + (self.me().2 - self.world.agent.yaw);
        let off = self.dist_me(lx, lz);
        // The nearest cell centre is within half a cell's diagonal of any point.
        if off > world_sim::world::PUSH_START_SLACK_M {
            return match self.issue(t, Action::MoveTo { x: lx, z: lz, reach_m: 0.36 }, Act::Move(MoveFor::PushLineUp(job.drop))) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        let err = (yaw - self.me().2 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        if err.abs() > 0.12 {
            return match self.issue(t, Action::Turn { yaw_deg: yaw.to_degrees() }, Act::Turn) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        self.progress(t, (t.target().picked / t.target().wanted as f32).min(1.0), Some(format!("{name} 미는 중 → {place}")));
        match self.issue(t, Action::Push { drop: job.drop, x: wx, z: wz }, Act::Push(job.drop)) {
            Ok(()) => Flow::Continue,
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    /// Advances a round trip: out to the destination, the quest's items onto
    /// its pile, back to the rest. `None` when no trip is going on.
    fn trip_step(&mut self, t: &mut Task) -> Option<Flow> {
        let trip = t.trip?;
        let (wid, x, z, place) = self.waypoint_of(t)?;
        let reach = PLACE_REACH_M;
        let name = t.target().label_ko.clone();
        match trip.stage {
            TripStage::Out => {
                if self.dist_me(x, z) <= reach + 0.2 {
                    t.trip = Some(Trip { stage: TripStage::Unload, ..trip });
                    return self.trip_step(t);
                }
                t.place_fixes = 0;
                t.act_distance0 = self.dist_me(x, z);
                self.progress(t, t.target().picked / t.target().wanted as f32, Some(format!("왕복 {}회차 · {place}로 운반 중", t.trips)));
                Some(match self.issue(t, Action::MoveTo { x, z, reach_m: reach }, Act::Move(MoveFor::Shuttle)) {
                    Ok(()) => Flow::Continue,
                    Err(ActionError::NoPath) => fail("no_path", format!("{place}까지 갈 경로가 없습니다 (짐을 실은 채로는 지나갈 수 없는 지형)")),
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                })
            }
            TripStage::Unload => {
                let next = t.targets.iter().find(|x| self.world.carried_amount(x.item) > AMOUNT_EPS && x.picked > amount_of(&t.delivered, &x.resource) + AMOUNT_EPS).map(|x| x.item);
                match next {
                    Some(item) => Some(match self.issue(t, Action::Deliver { waypoint: wid, item: Some(item) }, Act::Deliver(item)) {
                        Ok(()) => Flow::Continue,
                        Err(ActionError::OutOfReach { .. }) if !t.deliver_retry => self.reach_for_place(t, MoveFor::Shuttle),
                        Err(e) => fail(e.code(), action_error_ko(&e)),
                    }),
                    None => {
                        t.trip = Some(Trip { stage: TripStage::Back, ..trip });
                        self.trip_step(t)
                    }
                }
            }
            TripStage::Back => {
                let (bx, bz) = trip.back;
                if self.dist_me(bx, bz) <= PICKUP_REACH_M - 0.1 {
                    t.trip = None;
                    return None;
                }
                self.progress(t, t.target().picked / t.target().wanted as f32, Some(format!("왕복 {}회차 · {name}에게 돌아가는 중", t.trips)));
                // Where it believes the rest lies; a spot it cannot stand on (the belief drifted onto a slope) from the nearest one in reach of it.
                let r = match self.issue(t, Action::MoveTo { x: bx, z: bz, reach_m: 0.4 }, Act::Move(MoveFor::Return)) {
                    Err(ActionError::NoPath) => self.issue(t, Action::MoveTo { x: bx, z: bz, reach_m: PICKUP_REACH_M - 0.1 }, Act::Move(MoveFor::Return)),
                    r => r,
                };
                Some(match r {
                    Ok(()) => Flow::Continue,
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                })
            }
        }
    }

    /// The target is done: the next target, or carrying, or the end.
    fn next_target(&mut self, t: &mut Task) -> Flow {
        let k = t.cur;
        let last_collect_step = 4 * k + 3;
        let more = k + 1 < t.targets.len();
        if let Some(q) = self.quests.get_mut(&t.quest) {
            let _ = q.complete_step(last_collect_step);
        }
        if more {
            t.cur += 1;
            t.phase = Phase::Locate;
            t.reset_locate();
            t.locate_since_s = self.world.time_s;
            t.drops.clear();
            t.drop_idx = 0;
            t.rounds = 0;
            let label = t.target().label_ko.clone();
            self.log("quest", Level::Info, format!("{} · 다음 대상: {label}", t.quest), None);
            Flow::Continue
        } else if t.destination.is_some() {
            t.phase = Phase::Carry;
            let dest = t.destination.clone().unwrap_or_default();
            self.log("quest", Level::Info, format!("{} · 운반 시작 → {dest}", t.quest), None);
            Flow::Continue
        } else {
            Flow::Done
        }
    }

    fn waypoint_of(&self, t: &Task) -> Option<(u32, f32, f32, String)> {
        t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| (w.id, w.x, w.z, w.label_ko.clone()))
    }

    fn do_carry(&mut self, t: &mut Task) -> Flow {
        let Some((_, x, z, name)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let reach = PLACE_REACH_M;
        let dist = self.dist_me(x, z);
        if dist <= reach + 0.2 {
            self.goto_phase(t, Phase::Deliver, "");
            return Flow::Continue;
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: reach }).map(|p| p.time_s);
        t.act_distance0 = dist;
        t.place_fixes = 0;
        match self.issue(t, Action::MoveTo { x, z, reach_m: reach }, Act::Move(MoveFor::Carry)) {
            Ok(()) => Flow::Continue,
            Err(ActionError::NoPath) => fail("no_path", format!("{name}까지 갈 경로가 없습니다 (이 베이스로는 지나갈 수 없는 지형)")),
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    fn do_deliver(&mut self, t: &mut Task) -> Flow {
        let Some((wid, ..)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        while t.deliver_idx < t.targets.len() {
            let item = t.targets[t.deliver_idx].item;
            if self.world.carried_amount(item) <= AMOUNT_EPS {
                t.deliver_idx += 1;
                continue;
            }
            return match self.issue(t, Action::Deliver { waypoint: wid, item: Some(item) }, Act::Deliver(item)) {
                Ok(()) => Flow::Continue,
                Err(ActionError::OutOfReach { .. }) if !t.deliver_retry => self.reach_for_place(t, MoveFor::Carry),
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        Flow::Done
    }

    /// A move to a place ended where the robot does not believe the place is
    /// in reach: a look on the way moved its belief, or its odometer counted
    /// the walk short of where the plan put the end. It walks the rest, as one
    /// walks on until one sees one has arrived, a few times at most.
    fn closer_to_place(&mut self, t: &mut Task, why: MoveFor) -> Option<Flow> {
        let (_, x, z, name) = self.waypoint_of(t)?;
        let d = self.dist_me(x, z);
        if d <= PLACE_REACH_M + 0.2 || t.place_fixes >= MAX_PLACE_FIXES {
            return None;
        }
        t.place_fixes += 1;
        self.log("quest", Level::Info, format!("{name}까지 아직 {d:.1} m — 마저 걸어감"), None);
        match self.issue(t, Action::MoveTo { x, z, reach_m: PLACE_REACH_M }, Act::Move(why)) {
            Ok(()) => Some(Flow::Continue),
            // No way closer: the hand tries from here.
            Err(ActionError::NoPath) => None,
            Err(e) => Some(fail(e.code(), action_error_ko(&e))),
        }
    }

    /// The hand did not find the pile where the robot believed it stood in
    /// reach of it: it looks at the place (a sighting of the flag fixes its
    /// belief) and walks up close, once per delivery.
    fn reach_for_place(&mut self, t: &mut Task, why: MoveFor) -> Flow {
        let Some((_, x, z, name)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        t.deliver_retry = true;
        if self.scan_now("place").is_ok() {
            t.looks += 1;
            self.vstats.looks += 1;
        }
        // The look's fix first: the place is on the map, and the walk up to it goes by the corrected pose.
        self.take_fix(t);
        self.log("quest", Level::Warn, format!("{name}에 손이 닿지 않음 — 보고 더 다가감"), None);
        let r = match self.issue(t, Action::MoveTo { x, z, reach_m: PLACE_REACH_M - 0.5 }, Act::Move(why.clone())) {
            Err(ActionError::NoPath) => self.issue(t, Action::MoveTo { x, z, reach_m: PLACE_REACH_M }, Act::Move(why)),
            r => r,
        };
        match r {
            Ok(()) => Flow::Continue,
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    fn do_goto(&mut self, t: &mut Task) -> Flow {
        let Some((_, x, z, name)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let reach = PLACE_REACH_M;
        let dist = self.dist_me(x, z);
        if dist <= reach + 0.2 {
            return self.after_goto(t);
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: reach }).map(|p| p.time_s);
        t.act_distance0 = dist;
        t.place_fixes = 0;
        match self.issue(t, Action::MoveTo { x, z, reach_m: reach }, Act::Move(MoveFor::Goto)) {
            Ok(()) => Flow::Continue,
            Err(ActionError::NoPath) => fail("no_path", format!("{name}까지 갈 경로가 없습니다 (이 베이스로는 지나갈 수 없는 지형)")),
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    /// At the place: a walk ends, a drop quest sets its items down.
    fn after_goto(&mut self, t: &mut Task) -> Flow {
        if t.kind == "drop" {
            self.goto_phase(t, Phase::Unload, "");
            return Flow::Continue;
        }
        Flow::Done
    }

    /// Sets down what a drop quest names (everything when it names nothing).
    fn do_unload(&mut self, t: &mut Task) -> Flow {
        if t.drop_items.is_empty() {
            if t.unload_idx > 0 {
                return Flow::Done;
            }
            t.unload_idx = 1;
            return match self.issue(t, Action::Drop { item: None, amount: None }, Act::Unload) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        while t.unload_idx < t.drop_items.len() {
            let it = t.drop_items[t.unload_idx].clone();
            t.unload_idx += 1;
            let Some(item) = ItemKind::from_id(&it.resource) else { continue };
            let have = self.world.carried_amount(item);
            if have <= AMOUNT_EPS {
                self.log("quest", Level::Warn, format!("적재함에 {}{} 없어 건너뜀", item.spec().label_ko, hand_search::text::i_ga(item.spec().label_ko)), None);
                continue;
            }
            let amount = it.quantity.map(|q| (q as f32).min(have));
            if let Some(q) = it.quantity.filter(|q| (*q as f32) > have + AMOUNT_EPS) {
                self.log("quest", Level::Warn, format!("{} {q}개를 내려놓으라 했지만 {}개만 있어 그만큼 내려놓음", item.spec().label_ko, amount_text(have)), None);
            }
            return match self.issue(t, Action::Drop { item: Some(item), amount }, Act::Unload) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        if t.unloaded.is_empty() {
            let names: Vec<&str> = t.drop_items.iter().filter_map(|x| ItemKind::from_id(&x.resource)).map(|k| k.spec().label_ko).collect();
            return fail("nothing_to_drop", format!("적재함에 {}{} 없습니다", names.join(", "), hand_search::text::i_ga(names.last().copied().unwrap_or("물건"))));
        }
        Flow::Done
    }

    // ----- while an action runs -----

    fn while_running(&mut self, t: &mut Task, act: &Act) {
        match act {
            Act::Move(MoveFor::Approach(_)) => {
                let (mx, mz, _) = self.me();
                let target = t.target_lm.and_then(|id| self.gates.landmark(id)).map(|l| (l.x, l.z, l.distance_to(mx, mz)));
                if !self.cfg.vision.intermittent {
                    // As before v0.7: a look every few ticks.
                    if t.approach_scan_cd == 0 {
                        t.approach_scan_cd = self.cfg.approach_scan_every_ticks;
                        if self.scan_now("approach").is_ok() {
                            self.mark_coverage(&mut t.coverage);
                        }
                    } else {
                        t.approach_scan_cd -= 1;
                    }
                } else {
                    self.vstats.approach_s += world_sim::TICK_S as f64;
                    // The pattern: a look per look-ahead horizon, and one about a second before the hand acts.
                    let lead_m = self.world.agent.speed_mps.max(0.1) * EYE_LEAD_S + PICKUP_REACH_M;
                    // The look before the hand turns the head to the target when the level camera does not reach it (at the feet, to the side).
                    let eye = self.world.standing_eye_m();
                    let gaze = if t.pre_looked { None } else { target.filter(|&(_, _, d)| d <= lead_m).and_then(|(x, z, _)| gaze_to_see(&self.cam, eye, self.me(), x, z)) };
                    let pre = gaze.is_some();
                    let due = t.look_cd == 0;
                    if pre || due {
                        let kind = if pre { "pre_act" } else { "pattern" };
                        if pre {
                            t.pre_looked = true;
                        }
                        t.look_cd = self.cfg.vision.look_every_ticks;
                        let (pan, tilt) = gaze.unwrap_or((0.0, 0.0));
                        if self.scan_gaze(kind, pan, tilt).is_ok() {
                            t.looks += 1;
                            self.vstats.looks += 1;
                            self.vstats.approach_looks += 1;
                            self.mark_coverage(&mut t.coverage);
                        }
                    } else {
                        t.look_cd -= 1;
                    }
                }
                if let Some((_, _, d)) = target {
                    let p = 1.0 - (d / t.act_distance0.max(0.1)).clamp(0.0, 1.0);
                    self.progress(t, p, Some(format!("남은 거리 {d:.1} m")));
                }
            }
            Act::Move(MoveFor::Drop) | Act::Move(MoveFor::Return) => self.piece_look(t),
            Act::Move(MoveFor::Carry) | Act::Move(MoveFor::Goto) | Act::Move(MoveFor::Shuttle) => {
                if let Some((_, x, z, _)) = self.waypoint_of(t) {
                    // Near the place a look at its flag fixes where the robot believes it is.
                    if self.place_look(&mut t.look_cd, x, z) {
                        t.looks += 1;
                    }
                    if !matches!(act, Act::Move(MoveFor::Shuttle)) {
                        let d = self.dist_me(x, z);
                        let p = 1.0 - (d / t.act_distance0.max(0.1)).clamp(0.0, 1.0);
                        let trips = if t.trips > 0 { format!(" · 왕복 {}회 뒤 마지막 운반", t.trips) } else { String::new() };
                        self.progress(t, p, Some(format!("남은 거리 {d:.1} m{trips}")));
                    }
                }
            }
            Act::Mine(node) => {
                if let Some(n) = self.world.node(*node) {
                    let p = 1.0 - (n.hp / n.max_hp.max(1e-3)).clamp(0.0, 1.0);
                    self.progress(t, p, None);
                }
            }
            _ => {}
        }
    }

    // ----- when an action ends -----

    fn on_act_done(&mut self, t: &mut Task, act: Act, ok: bool, reason: Option<String>) -> Flow {
        let elapsed = self.world.time_s - t.act_started_s;
        if matches!(act, Act::Pick(_)) {
            if let Some(e) = self.world.last_act.as_ref().and_then(|p| p.rollout.metric("aim_err_m")) {
                if t.last_pick_blind {
                    self.vstats.memory_aim_m.push(e);
                } else {
                    self.vstats.seen_aim_m.push(e);
                }
            }
        }
        if !ok {
            let why = reason.clone().unwrap_or_else(|| "알 수 없음".into());
            if why == "cancelled" {
                return fail("cancelled", "중지되었습니다");
            }
            // A fall: the robot gets up (the world holds it down for a while) and the phase moves again.
            if why.starts_with("fall") {
                t.falls += 1;
                let ground = self.world.ground_here().id();
                self.sds.record_agreement(&Scope::new(Track::Act, "walk", ground), "no_fall", false);
                if t.falls > MAX_FALLS {
                    return fail("fall", format!("{}번 넘어져 그만둠 — {}", t.falls, why.trim_start_matches("fall: ")));
                }
                self.log("quest", Level::Warn, format!("넘어짐 {}/{MAX_FALLS} — 일어나서 다시 감: {}", t.falls, why.trim_start_matches("fall: ")), None);
                return Flow::Continue;
            }
            match act {
                Act::Pick(id) => {
                    // The hand closed on air: the memory was off by more than half the top face, so the next try looks first.
                    if why == "grasp_missed" {
                        if let (Some(m), Some(d)) = (t.mem.get_mut(&id), self.world.drop_item(id)) {
                            let size = d.kind.spec().piece_size_m(d.amount);
                            m.sigma0_m = m.sigma0_m.max(0.5 * size[0].min(size[2]));
                        }
                    }
                    // A grasp that slipped or missed: the piece fell back; try again, then leave it.
                    t.grasp_tries += 1;
                    if t.grasp_tries < MAX_GRASP_TRIES {
                        self.log("quest", Level::Warn, format!("잡기 실패 ({why}) — 다시 잡기 {}/{MAX_GRASP_TRIES}", t.grasp_tries + 1), Some(json!({"drop": id, "code": why})));
                    } else {
                        self.log("quest", Level::Warn, format!("{MAX_GRASP_TRIES}번 잡지 못해 이 조각은 두고 감 ({why})"), Some(json!({"drop": id, "code": why})));
                        t.grasp_tries = 0;
                        t.drop_idx += 1;
                        t.wasted += 1;
                    }
                    t.pick_retry = false;
                    return Flow::Continue;
                }
                Act::Push(id) => {
                    if let Some(j) = t.push.as_mut() {
                        j.tries += 1;
                        if j.tries >= MAX_PUSH_TRIES {
                            self.log("quest", Level::Warn, format!("{MAX_PUSH_TRIES}번 밀다 놓쳐 이 조각은 두고 감 ({why})"), Some(json!({"drop": id, "code": why})));
                            t.push = None;
                            t.drop_idx += 1;
                            t.wasted += 1;
                        } else {
                            self.log("quest", Level::Warn, format!("밀다 놓침 ({why}) — 다시 줄 서기 {}/{MAX_PUSH_TRIES}", j.tries + 1), Some(json!({"drop": id, "code": why})));
                        }
                    }
                    return Flow::Continue;
                }
                Act::Move(MoveFor::PushLineUp(_)) => return Flow::Continue,
                _ => {}
            }
            // A move that fails midway (a blocked cell, for example) is retried from the top.
            return match act {
                Act::Move(MoveFor::Approach(id)) => {
                    self.gates.note_unreachable(id);
                    self.goto_phase(t, Phase::Locate, &format!("접근 실패({why}), 다른 대상을 찾음"));
                    t.reset_locate();
                    Flow::Continue
                }
                Act::Move(MoveFor::Search) | Act::Turn => Flow::Continue,
                _ => fail(&why, format!("동작이 실패했습니다: {why}")),
            };
        }
        match act {
            Act::Turn => {
                t.scan_cooldown = 0;
                Flow::Continue
            }
            Act::Move(MoveFor::Search) => {
                t.scan_cooldown = 0;
                t.scans_here = 0;
                // Look first where the planner expected the most unseen ground.
                if let Some(yaw) = t.face_after.take() {
                    if let Err(e) = self.issue(t, Action::Turn { yaw_deg: yaw.to_degrees() }, Act::Turn) {
                        return fail(e.code(), action_error_ko(&e));
                    }
                }
                Flow::Continue
            }
            Act::Move(MoveFor::Approach(_)) => {
                self.calibrate_move(t, elapsed);
                self.goto_phase(t, Phase::Mine, "");
                Flow::Continue
            }
            Act::Move(MoveFor::Drop) => {
                self.calibrate_move(t, elapsed);
                Flow::Continue
            }
            Act::Move(MoveFor::Shuttle) => {
                if let Some(flow) = self.closer_to_place(t, MoveFor::Shuttle) {
                    return flow;
                }
                if let Some(tr) = t.trip {
                    t.trip = Some(Trip { stage: TripStage::Unload, ..tr });
                }
                Flow::Continue
            }
            Act::Move(MoveFor::Return) => {
                t.trip = None;
                Flow::Continue
            }
            Act::Move(MoveFor::Carry) => {
                self.calibrate_move(t, elapsed);
                if let Some(flow) = self.closer_to_place(t, MoveFor::Carry) {
                    return flow;
                }
                self.goto_phase(t, Phase::Deliver, "");
                Flow::Continue
            }
            Act::Move(MoveFor::Goto) => {
                self.calibrate_move(t, elapsed);
                if let Some(flow) = self.closer_to_place(t, MoveFor::Goto) {
                    return flow;
                }
                self.after_goto(t)
            }
            Act::Mine(node) => self.after_mine(t, node, elapsed),
            Act::Pick(_) => {
                t.drop_idx += 1;
                t.pick_retry = false;
                t.grasp_tries = 0;
                let (picked, wanted) = (t.target().picked, t.target().wanted);
                self.progress(t, (picked / wanted as f32).min(1.0), Some(format!("{}/{wanted}개", amount_text(picked))));
                Flow::Continue
            }
            Act::Split(_) => {
                // The broken-off piece is picked next; the rest stays in the list after it.
                if let Some(p) = t.split_piece.take() {
                    t.drops.insert(t.drop_idx, p);
                }
                Flow::Continue
            }
            Act::Deliver(_) => {
                t.deliver_retry = false;
                if t.trip.is_none() {
                    t.deliver_idx += 1;
                }
                Flow::Continue
            }
            Act::Unload => Flow::Continue,
            Act::Push(_) | Act::Move(MoveFor::PushLineUp(_)) => Flow::Continue,
        }
    }

    fn calibrate_move(&mut self, t: &mut Task, elapsed: f64) {
        if let Some(pred) = t.act_predicted_s.take() {
            if pred > 0.5 {
                let base = format!("{:?}", self.world.hand.base.kind).to_lowercase();
                self.sds.record_calibration(&Scope::new(Track::Motion, &base, "move"), "route_time_s", elapsed, pred as f64);
            }
        }
    }

    fn after_mine(&mut self, t: &mut Task, node: u32, elapsed: f64) -> Flow {
        if let Some(pred) = t.act_predicted_s.take() {
            if pred > 0.5 {
                if let Some((_, kind, _)) = &t.broken {
                    self.sds.record_calibration(&Scope::new(Track::Motion, kind.id(), "mine"), "mine_time_s", elapsed, pred as f64);
                }
            }
        }
        let Some((_, kind, drops)) = t.broken.take() else {
            return fail("mine_failed", format!("노드 n{node}를 캐지 못했습니다"));
        };
        let wanted = t.target().item;
        let hit = kind.spec().drop == wanted;
        let lm = t.target_lm.take().and_then(|id| self.gates.landmark(id).cloned());
        if let Some(l) = &lm {
            let vscope = Scope::new(Track::Vision, &l.label, "");
            self.sds.record_agreement(&vscope, "touch", hit);
            if hit {
                self.sds.record_baseline(&vscope, "score", l.score as f64);
            } else {
                self.sds.record_confusion(&vscope, &l.label, kind.id(), l.margin as f64);
            }
            self.gates.consume(l.id);
        }
        if hit {
            t.touch_ok += 1;
            let ids: Vec<u32> = drops.iter().filter_map(|d| parse_id(d)).filter(|id| self.world.drop_item(*id).map_or(false, |d| d.kind == wanted)).collect();
            self.log(
                "vision",
                Level::Info,
                format!("접촉 확인: {} 맞음 ({}개 떨어짐)", kind.spec().label_ko, ids.len()),
                lm.as_ref().map(|l| json!({"landmark": l.id, "label": l.label, "score": l.score, "margin": l.margin, "agreed": true})),
            );
            // It sees them fall: where they lie goes into memory, in its believed frame.
            let sigma = self.world.aim_sigma_m();
            for &id in &ids {
                if let Some(d) = self.world.drop_item(id) {
                    let (x, z) = self.world.belief_of(d.x, d.z);
                    t.mem.insert(id, Memory::seen(x, z, sigma, self.world.odom.travelled_m, self.world.tick));
                }
            }
            t.drops = ids;
            t.drop_idx = 0;
            t.pick_retry = false;
            self.goto_phase(t, Phase::Collect, "");
        } else {
            t.touch_miss += 1;
            t.wasted += 1;
            let said = lm.as_ref().map(|l| l.label.clone()).unwrap_or_else(|| "?".into());
            self.log(
                "vision",
                Level::Warn,
                format!("접촉 불일치: '{said}'로 인식했지만 {}였음 — 헛수고 (총 {}회)", kind.spec().label_ko, t.wasted),
                lm.as_ref().map(|l| json!({"landmark": l.id, "label": l.label, "actual": kind.id(), "score": l.score, "margin": l.margin, "agreed": false})),
            );
            self.goto_phase(t, Phase::Locate, &format!("인식 오류({said}→{}), 다시 탐색", kind.id()));
            t.reset_locate();
        }
        Flow::Continue
    }

    // ----- ending -----

    fn result_of(&self, t: &Task, summary: String) -> QuestResult {
        let whole = |list: &[(String, f32)]| list.iter().map(|(k, a)| (k.clone(), (a + AMOUNT_EPS).floor() as u32)).collect::<Vec<_>>();
        let items: Vec<(String, u32)> = match t.kind {
            "fetch" => whole(&t.delivered),
            "drop" => whole(&t.unloaded),
            _ => t.targets.iter().map(|x| (x.resource.clone(), (x.picked + AMOUNT_EPS).floor() as u32)).collect(),
        };
        QuestResult {
            ticks: self.world.tick - t.started_tick,
            time_s: self.world.time_s - t.started_time_s,
            energy_j: self.world.agent.energy_used_j - t.started_energy_j,
            precision: t.precision(),
            summary,
            items,
            wasted_actions: t.wasted,
            trips: t.trips + u32::from(t.kind == "fetch" && !t.delivered.is_empty()),
        }
    }

    fn finish_task(&mut self, t: Task) {
        let items_ko = |items: &[(String, f32)]| -> String {
            items
                .iter()
                .map(|(r, n)| format!("{} {}개", ItemKind::from_id(r).map(|k| k.spec().label_ko.to_string()).unwrap_or_else(|| r.clone()), amount_text(*n)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let dest_ko = t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| w.label_ko.clone()).unwrap_or_default();
        let mut r = self.result_of(&t, String::new());
        let tail = format!(
            "{:.1}초 · {:.0} J{}{}{}",
            r.time_s,
            r.energy_j,
            r.precision.map(|p| format!(" · 접촉 확인 정밀도 {:.0}%", p * 100.0)).unwrap_or_default(),
            if r.wasted_actions > 0 { format!(" · 헛수고 {}회", r.wasted_actions) } else { String::new() },
            if r.trips > 1 { format!(" · {}회 왕복", r.trips) } else { String::new() }
        ) + &if t.pushed.is_empty() { String::new() } else { format!(" · 밀어서 옮김 {}", items_ko(&t.pushed)) } + &if t.falls > 0 { format!(" · 넘어짐 {}회", t.falls) } else { String::new() };
        let picked: Vec<(String, f32)> = t.targets.iter().map(|x| (x.resource.clone(), x.picked)).collect();
        r.summary = match t.kind {
            "fetch" => format!("{}{} {dest_ko}에 인도했습니다 — {tail}", items_ko(&t.delivered), eul_reul(&items_ko(&t.delivered))),
            "collect" => format!("{} 채취 완료 — {tail}", items_ko(&picked)),
            "drop" => {
                let at = match &t.unload_place {
                    Some(w) => self.world.waypoints.iter().find(|p| world_sim::snapshot::waypoint_id(p.id) == *w).map(|p| format!("{}에 쌓음", p.label_ko)).unwrap_or_default(),
                    None => "땅에 내려놓음".to_string(),
                };
                format!("{}{} 내려놓았습니다 ({at}) — {tail}", items_ko(&t.unloaded), eul_reul(&items_ko(&t.unloaded)))
            }
            _ => format!("{dest_ko}에 도착했습니다 — {tail}"),
        };
        let last = t.step_of(&t.phase);
        let tick = self.world.tick;
        if let Some(q) = self.quests.get_mut(&t.quest) {
            let _ = q.complete_step(last);
            if let Err(e) = q.finish(r.clone()) {
                self.audit.warn("quest", format!("퀘스트 {} 종료 처리 실패: {e}", t.quest));
            }
        }
        let scope = Scope::new(Track::Motion, t.kind, "");
        self.sds.record_baseline(&scope, "duration_s", r.time_s);
        self.sds.record_baseline(&scope, "energy_j", r.energy_j);
        if t.kind == "fetch" {
            self.sds.record_baseline(&scope, "trips", r.trips as f64);
        }
        self.log("quest", Level::Info, format!("퀘스트 {} 완료: {}", t.quest, r.summary), Some(json!({"quest": t.quest, "tick": tick, "result": r})));
        self.record_quest_run(&t.quest, &t.attempts);
        self.flush_sds();
    }

    pub(crate) fn fail_task(&mut self, t: Task, code: &str, msg: &str) {
        if self.world.busy() {
            let ev = self.world.cancel();
            self.log_world_events(&ev);
        }
        let r = self.result_of(&t, format!("실패: {msg}"));
        if let Some(q) = self.quests.get_mut(&t.quest) {
            let _ = q.fail(code, msg, Some(r.clone()));
        }
        self.log("quest", Level::Warn, format!("퀘스트 {} 실패 [{code}]: {msg}", t.quest), Some(json!({"quest": t.quest, "code": code, "result": r})));
        self.record_quest_run(&t.quest, &t.attempts);
        self.flush_sds();
    }

    /// Stops the running quest and drops the queue.
    pub fn cancel(&mut self) {
        if let Some(m) = self.ml_task.take() {
            self.ml_end(m, Some(("cancelled".into(), "사용자가 중지했습니다".into())));
        }
        let t = self.task.take();
        if self.world.busy() {
            let ev = self.world.cancel();
            self.log_world_events(&ev);
        }
        match t {
            Some(t) => self.fail_task(t, "cancelled", "사용자가 중지했습니다"),
            None => {}
        }
        let ids: Vec<String> = self.quests.all().iter().filter(|q| !q.is_finished()).map(|q| q.id.clone()).collect();
        for id in ids {
            if let Some(q) = self.quests.get_mut(&id) {
                let _ = q.fail("cancelled", "사용자가 중지했습니다", None);
            }
            self.log("quest", Level::Warn, format!("퀘스트 {id} 취소"), None);
        }
    }
}
