use std::collections::HashMap;

use noto_quest::{QuestResult, State};
use noto_sds::{Level, Scope, Track};
use noto_search::{DropTarget, Plan};
use serde_json::json;
use world_sim::explore::{Coverage, Viewpoint};
use world_sim::nav::Goal;
use world_sim::world::{AMOUNT_EPS, DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{amount_text, Action, ActionError, ItemKind, NodeKind, TimedEvent, WorldEvent};

use noto_search::text::eul_reul;

use crate::cargo::{self, CargoAction, CargoCtx, CargoDecision};
use crate::vision::{gaze_to_see, memory_ok, Memory, EYE_LEAD_S};

pub(crate) const MAX_GRASP_TRIES: u32 = 3;
pub(crate) const MAX_PUSH_TRIES: u32 = 3;
pub(crate) const MAX_FALLS: u32 = 3;
pub(crate) const TOUCH_PIECE_M: f32 = 0.25;
pub(crate) const SIDE_SPOT_REACH_M: f32 = 0.36;
pub(crate) const PLACE_REACH_M: f32 = DELIVER_REACH_M - 0.5;
pub(crate) const INSPECT_STAND_M: f32 = 1.2;
/// Half the width of the lane the tines and the base sweep going in (m): the base's half width and a margin.
pub(crate) const FORK_LANE_HALF_M: f32 = 0.30;
/// How near a lift truck drives to the spot it lines up at (m): half a cell's diagonal, so a cell centre always is.
pub(crate) const LINEUP_REACH_M: f32 = 0.36;
pub(crate) const SMALL_ITEM_SEE_M: f32 = 3.0;
pub(crate) const GROUND_SWEEP_TILT_DEG: f32 = 30.0;
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
    Shuttle,
    Return,
    PushLineUp(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Act {
    Move(MoveFor),
    Turn,
    Look(u32),
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
    pub picked: f32,
    pub label_ko: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PushJob {
    pub drop: u32,
    pub tries: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TripStage {
    Out,
    Unload,
    Back,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Trip {
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
    pub split_piece: Option<u32>,
    pub started_tick: u64,
    pub started_time_s: f64,
    pub started_energy_j: f64,
    pub act_started_s: f64,
    pub act_predicted_s: Option<f32>,
    pub act_distance0: f32,
    pub locate_since_s: f64,
    pub scan_cooldown: u32,
    pub scans_here: u32,
    pub rotations: u32,
    pub relocations: u32,
    pub visited: Vec<(f32, f32)>,
    pub target_lm: Option<u32>,
    pub approach_tries: u32,
    pub approach_scan_cd: u32,
    pub coverage: Coverage,
    pub face_after: Option<f32>,
    pub inspected: Vec<u32>,
    pub drops: Vec<u32>,
    pub drop_idx: usize,
    pub rounds: u32,
    pub pick_retry: bool,
    pub trip: Option<Trip>,
    pub trips: u32,
    pub last_cargo: Option<(u32, String)>,
    pub deliver_idx: usize,
    pub drop_items: Vec<DropTarget>,
    pub unload_idx: usize,
    pub unloaded: Vec<(String, f32)>,
    pub unload_place: Option<String>,
    pub attempts: Vec<u32>,
    pub touch_ok: u32,
    pub touch_miss: u32,
    pub wasted: u32,
    pub delivered: Vec<(String, f32)>,
    pub grasp_tries: u32,
    pub falls: u32,
    pub push: Option<PushJob>,
    pub pushed: Vec<(String, f32)>,
    pub mem: HashMap<u32, Memory>,
    pub look_cd: u32,
    pub pre_looked: bool,
    pub side: Option<(u32, f32)>,
    pub looks: u32,
    pub blind_acts: u32,
    pub last_pick_blind: bool,
    pub place_fixes: u32,
    pub deliver_retry: bool,
    pub look_turn: Option<u32>,
    pub gazed: Option<u32>,
    /// The drop the forks last lined up in front of (once per drop).
    pub lined_up: Option<u32>,
    /// The act (by its start time) whose rider look and whose re-check a lift truck already made: once per act.
    pub rider_looked: Option<f64>,
    pub rechecked: Option<f64>,
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
            rider_looked: None,
            rechecked: None,
            gazed: None,
            lined_up: None,
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

    fn placed(&self) -> bool {
        !self.target().item.from_node()
    }

    fn precision(&self) -> Option<f64> {
        let n = self.touch_ok + self.touch_miss;
        (n > 0).then(|| self.touch_ok as f64 / n as f64)
    }
}

fn make_target(t: &noto_search::Target) -> Result<TaskTarget, (String, String)> {
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
    pub(crate) fn begin_next(&mut self) {
        let Some(id) = self.quests.current_id() else { return };
        let Some(q) = self.quests.get(&id) else { return };
        let (state, plan) = (q.state, q.plan.clone());
        if state == State::Active {
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
        if self.world.fallen_until.is_some_and(|f| self.world.time_s < f) {
            return Flow::Continue;
        }
        self.next_action(t)
    }

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
                if matches!(act, Act::Look(_)) {
                    // A new look at an item: the two eyes start over on it.
                    self.fork_cross = None;
                }
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
        let kind = if t.placed() { EntKind::Drop } else { EntKind::Source };
        let (ax, az, _) = self.me();
        self.gates
            .landmarks()
            .iter()
            .filter(|l| l.confirmed && l.state == LmState::Active && l.kind == kind && &l.link == want)
            .min_by(|a, b| a.distance_to(ax, az).total_cmp(&b.distance_to(ax, az)))
            .map(|l| l.id)
    }

    pub(crate) fn mark_coverage(&self, cov: &mut Coverage) {
        self.mark_coverage_as(cov, 0.0, world_sim::explore::SEARCH_RANGE_M);
    }

    pub(crate) fn mark_coverage_as(&self, cov: &mut Coverage, tilt_deg: f32, range_m: f32) {
        let mut eye = self.cam.eye(&self.world);
        eye.pitch_deg -= tilt_deg;
        eye.range_m = eye.range_m.min(range_m);
        cov.look(&self.world.terrain, &eye, self.world.agent.yaw, self.cam.fov_deg);
    }

    fn search_reach(&self, t: &Task) -> (f32, f32) {
        if t.placed() {
            (SMALL_ITEM_SEE_M, GROUND_SWEEP_TILT_DEG)
        } else {
            (world_sim::explore::SEARCH_RANGE_M, 0.0)
        }
    }

    pub(crate) fn search_viewpoint(&self, t: &Task) -> Option<Viewpoint> {
        let (range, tilt) = self.search_reach(t);
        self.viewpoint_with(&t.coverage, &t.visited, tilt, range)
    }

    pub(crate) fn search_viewpoint_for(&self, cov: &Coverage, avoid: &[(f32, f32)]) -> Option<Viewpoint> {
        self.viewpoint_with(cov, avoid, 0.0, world_sim::explore::SEARCH_RANGE_M)
    }

    pub(crate) fn viewpoint_with(&self, cov: &Coverage, avoid: &[(f32, f32)], tilt: f32, range: f32) -> Option<Viewpoint> {
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
            if tilt > 0.0 && self.scan_gaze("locate_down", 0.0, tilt).is_ok() {
                self.mark_coverage_as(&mut t.coverage, tilt, range);
            }
            t.scans_here += 1;
            t.scan_cooldown = self.cfg.scan_every_ticks;
            return Flow::Continue;
        }
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
        let b = (z - mz).atan2(x - mx) - myaw;
        let bearing = b.sin().atan2(b.cos()).to_degrees();
        let pan = (-bearing).clamp(-noto_core::kin::NECK_PAN_MAX_DEG, noto_core::kin::NECK_PAN_MAX_DEG);
        let depression = self.world.standing_eye_m().atan2(d.max(0.1)).to_degrees();
        let base_tilt = (depression + self.cam.pitch_deg).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
        t.inspected.push(id);
        let need = self.cfg.gates.confirm_sightings.max(1);
        let mut tilt = base_tilt;
        for i in 0..need {
            let seen_before = self.gates.landmark(id).map_or(0, |l| l.seen);
            tilt = if i > 0 && !self.gates.landmark(id).is_some_and(|l| l.confirmed) {
                (base_tilt + 12.0).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG)
            } else {
                base_tilt
            };
            if self.scan_gaze("inspect", pan, tilt).is_err() {
                break;
            }
            let seen_after = self.gates.landmark(id).map_or(0, |l| l.seen);
            if seen_after == seen_before && tilt < noto_core::kin::NECK_PITCH_MAX_DEG {
                let deeper = (tilt + 12.0).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
                let _ = self.scan_gaze("inspect", pan, deeper);
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

    fn approach_placed(&mut self, t: &mut Task, id: u32, at: (f32, f32), dist: f32) -> Flow {
        let item = t.target().item;
        if t.side.is_none() && dist > PICKUP_REACH_M {
            let (mx, mz, myaw) = self.me();
            let heading = match self.waypoint_of(t) {
                Some((_, wx, wz, _)) => (wz - at.1).atan2(wx - at.0),
                None => (at.1 - mz).atan2(at.0 - mx),
            };
            let _ = myaw;
            if let Some((ox, oy)) = self.world.side_reach(item, 1.0) {
                t.side = Some((id, heading));
                let (s, c) = heading.sin_cos();
                let spot = (at.0 - ox * c - oy * s, at.1 - ox * s + oy * c);
                t.act_distance0 = self.dist_me(spot.0, spot.1);
                t.look_cd = self.cfg.vision.look_every_ticks;
                t.pre_looked = false;
                self.log("quest", Level::Info, format!("{}: 지나가며 오른쪽으로 집음 — 옆 자리 ({:.1}, {:.1})로", t.target().label_ko, spot.0, spot.1), Some(json!({"landmark": id, "spot": [spot.0, spot.1], "heading_deg": heading.to_degrees()})));
                return match self.issue(t, Action::MoveTo { x: spot.0, z: spot.1, reach_m: SIDE_SPOT_REACH_M }, Act::Move(MoveFor::Approach(id))) {
                    Ok(()) => Flow::Continue,
                    Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => {
                        t.side = None;
                        self.issue_front_approach(t, id, at)
                    }
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                };
            }
        }
        if t.side.is_some() || dist <= PICKUP_REACH_M - 0.1 {
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
                let vscope = self.gates.vision_scope(&l.label);
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
                    self.log("quest", Level::Warn, format!("기억한 자리에 {}이 손에 닿지 않음 — 내려다보며 다시 확인 후 접근 ({}/2)", t.target().label_ko, t.approach_tries), None);
                    let me = self.me();
                    let eye = self.world.standing_eye_m();
                    if let Some((pan, tilt)) = gaze_to_see(&self.cam, eye, me, l.x, l.z) {
                        let _ = self.scan_gaze("retouch", pan, tilt);
                    } else {
                        let d = (l.x - me.0).hypot(l.z - me.1).max(0.15);
                        let dep = eye.atan2(d).to_degrees();
                        let tilt = (dep + self.cam.pitch_deg).clamp(GROUND_SWEEP_TILT_DEG, noto_core::kin::NECK_PITCH_MAX_DEG);
                        let _ = self.scan_gaze("retouch", 0.0, tilt);
                    }
                    t.looks += 1;
                    self.vstats.looks += 1;
                    self.take_fix(t);
                    t.side = None;
                    self.goto_phase(t, Phase::Approach, "위치 재확인");
                    return Flow::Continue;
                }
                let vscope = self.gates.vision_scope(&l.label);
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
                        let vscope = self.gates.vision_scope(&l.label);
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

    fn quest_cargo_kg(&self, t: &Task) -> f32 {
        t.targets
            .iter()
            .map(|x| {
                let mine = (x.picked - amount_of(&t.delivered, &x.resource)).max(0.0);
                self.world.carried_amount(x.item).min(mine) * x.item.spec().mass_kg
            })
            .sum()
    }

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
            let m = *t.mem.entry(id).or_insert_with(|| {
                let (x, z) = self.world.belief_of(d.x, d.z);
                Memory::seen(x, z, self.world.aim_sigma_m(), self.world.odom.travelled_m, self.world.tick)
            });
            let side_here = t.side.is_some_and(|(sid, _)| sid == id);
            let dist = self.dist_me(m.x, m.z);
            // Wheels and tracks lift with forks and stop with the tine tips short of the item; legs bend down and
            // pick by hand from close by.
            let forks = noto_core::act::fork::applies(&self.world.body);
            let size = d.kind.spec().piece_size_m(d.amount);
            let stop_m = if forks { noto_core::act::fork::standoff_x(&self.world.body, size[0].max(size[2])) } else { 0.4 };
            let far_m = if forks { stop_m + 0.15 } else { PICKUP_REACH_M - 0.1 };
            // A lift truck takes the wanted item nearest in front first, as an operator does, rather than run its tines
            // through it to the one behind.
            if forks {
                if let Some(k) = self.fork_front_pick(t, id, (m.x, m.z)) {
                    let other = t.drops[k];
                    self.log("quest", Level::Info, format!("{name}: 포크 길 앞에 있는 {name}(d{other})부터 듦 — 뒤의 것(d{id})에 포크를 밀어 넣지 않음"), Some(json!({"drop": other, "instead_of": id})));
                    t.drops.swap(t.drop_idx, k);
                    continue;
                }
            }
            // Tines go straight in: line up on a side where no other item lies in their way.
            if forks && t.lined_up != Some(id) && (dist > far_m || self.fork_path_blocked(id, (m.x, m.z), self.me())) {
                t.lined_up = Some(id);
                if let Some((lx, lz)) = self.fork_lineup(id, (m.x, m.z), stop_m) {
                    self.log("quest", Level::Info, format!("{name}: 포크가 곧게 들어갈 자리로 정렬 ({lx:.1}, {lz:.1})"), Some(json!({"drop": id, "x": lx, "z": lz})));
                    // Within half a cell's diagonal a cell centre always qualifies; the pick turns the base to the item.
                    match self.issue(t, Action::MoveTo { x: lx, z: lz, reach_m: LINEUP_REACH_M }, Act::Move(MoveFor::Drop)) {
                        Ok(()) => return Flow::Continue,
                        Err(_) => {}
                    }
                }
            }
            if dist > far_m && !side_here {
                t.act_predicted_s = self.world.plan_route(Goal::Within { x: m.x, z: m.z, reach_m: stop_m }).map(|p| p.time_s);
                let r = match self.issue(t, Action::MoveTo { x: m.x, z: m.z, reach_m: stop_m }, Act::Move(MoveFor::Drop)) {
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
            let can_take =if forks { self.world.fork_check(d.kind, d.amount).map(|_| ()) } else { self.world.reach_ground(d.kind, d.amount) };
            if let Err(why) = can_take {
                if t.destination.is_none() || !forks {
                    return fail("reach_ground", format!("{name}{} 집을 수 없습니다: {why}", eul_reul(&name)));
                }
                self.log("quest", Level::Info, format!("{name}: 포크로 들 수 없어 범퍼로 밀어 옮김 — {why}"), Some(json!({"drop": id})));
                t.push = Some(PushJob { drop: id, tries: 0 });
                if self.cfg.vision.gaze_before_pick && t.gazed != Some(id) {
                    t.gazed = Some(id);
                    if self.issue(t, Action::Look { x: m.x, z: m.z }, Act::Look(id)).is_ok() {
                        return Flow::Continue;
                    }
                }
                return self.push_step(t);
            }
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
                    let (sx, sz, _) = self.me();
                    t.trip = Some(Trip { back: (sx, sz), back_odo_m: self.world.odom.travelled_m, stage: TripStage::Out });
                    let dest = self.waypoint_of(t).map(|w| w.3).unwrap_or_default();
                    self.log("quest", Level::Info, format!("{} · 왕복 {}회차: 실은 짐을 {dest}에 내려놓고 {name}에게 돌아옴", t.quest, t.trips), Some(json!({"trip": t.trips, "carried_kg": self.world.carried_kg(), "capacity_kg": self.world.capacity_kg()})));
                    return self.trip_step(t).unwrap_or(Flow::Continue);
                }
                CargoAction::Refuse => return fail(dec.code.unwrap_or("lift_overload"), dec.summary_ko.clone()),
            }
            let pick = match self.pick_from_memory(t, id, d.kind, d.amount) {
                Ok(a) => a,
                Err(flow) => return flow,
            };
            return match self.issue(t, pick, Act::Pick(id)) {
                Ok(()) => Flow::Continue,
                Err(ActionError::CannotReach { reason, .. }) if t.side.is_some() => {
                    self.log("quest", Level::Info, format!("옆으로는 못 집음 ({reason}) — 앞에서 집음"), None);
                    t.side = None;
                    Flow::Continue
                }
                Err(ActionError::OutOfReach { .. }) if !t.pick_retry => {
                    t.pick_retry = true;
                    t.side = None;
                    t.gazed = None;
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

    /// The index in the task's drops of a wanted item lying in the tines' way from where the base stands to the item
    /// `id` at `at` (the nearest one), which a lift truck takes first.
    fn fork_front_pick(&self, t: &Task, id: u32, at: (f32, f32)) -> Option<usize> {
        let from = self.me();
        let (dx, dz) = (at.0 - from.0, at.1 - from.1);
        let len = dx.hypot(dz).max(1e-3);
        let (ux, uz) = (dx / len, dz / len);
        t.drops
            .iter()
            .enumerate()
            .skip(t.drop_idx + 1)
            .filter(|(_, &o)| o != id)
            .filter_map(|(k, &o)| {
                let d = self.world.drop_item(o)?;
                let (px, pz) = self.world.belief_of(d.x, d.z);
                let (rx, rz) = (px - from.0, pz - from.1);
                let along = rx * ux + rz * uz;
                let off = (rx * uz - rz * ux).abs();
                (along > 0.0 && along < len && off < FORK_LANE_HALF_M).then_some((k, along))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(k, _)| k)
    }

    /// Whether another item lies in the way of the tines going from where the base stands (`from`) to the item at `at`.
    fn fork_path_blocked(&self, id: u32, at: (f32, f32), from: (f32, f32, f32)) -> bool {
        let (dx, dz) = (at.0 - from.0, at.1 - from.1);
        let len = dx.hypot(dz).max(1e-3);
        let (ux, uz) = (dx / len, dz / len);
        self.world.drops.iter().filter(|d| d.id != id).any(|d| {
            let (px, pz) = self.world.belief_of(d.x, d.z);
            let (rx, rz) = (px - from.0, pz - from.1);
            let along = rx * ux + rz * uz;
            let off = (rx * uz - rz * ux).abs();
            along > 0.0 && along < len + 0.1 && off < FORK_LANE_HALF_M
        })
    }

    /// A place `stop_m` from the item at `at` from which the tines go in with nothing else in their way, nearest the
    /// side the base comes from first.
    fn fork_lineup(&self, id: u32, at: (f32, f32), stop_m: f32) -> Option<(f32, f32)> {
        let me = self.me();
        let a0 = (me.1 - at.1).atan2(me.0 - at.0);
        (0..8)
            .map(|k| a0 + std::f32::consts::FRAC_PI_4 * if k % 2 == 0 { (k / 2) as f32 } else { -((k / 2 + 1) as f32) })
            .map(|a| (at.0 + stop_m * a.cos(), at.1 + stop_m * a.sin(), a))
            .find(|&(x, z, a)| {
                let from = (x, z, a + std::f32::consts::PI);
                let (tx, tz) = self.world.true_of(x, z);
                self.world.terrain.cell_of(tx, tz).is_some_and(|c| !self.world.terrain.is_water(c.0, c.1) && !self.world.blocked_cells().contains(&c)) && !self.fork_path_blocked(id, at, from)
            })
            .map(|(x, z, _)| (x, z))
    }

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
        let can_look = noto_core::act::gaze::can_aim(&self.world.body) || noto_core::act::fork::applies(&self.world.body);
        if !side && t.gazed != Some(id) && self.cfg.vision.gaze_before_pick && can_look {
            t.gazed = Some(id);
            if self.issue(t, Action::Look { x: m.x, z: m.z }, Act::Look(id)).is_ok() {
                return Err(Flow::Continue);
            }
        }
        let looked_down = t.gazed == Some(id) && self.gaze_check.as_ref().is_some_and(|g| g.drop == id && g.ok && self.world.tick.saturating_sub(g.tick) <= 3);
        if !looked_down && (!intermittent || !memory_ok(sigma, tolerance, &self.cfg.vision)) {
            if let Some((pan, tilt)) = gaze_to_see(&self.cam, self.world.standing_eye_m(), me, m.x, m.z) {
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
        let looked = m.tick == self.world.tick || looked_down;
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

    /// The look before the hand: the head is down over the item now, so the frame the scan sees is the close-up the
    /// pick-up needs. The scan names it and fixes where it lies; the depth model, aligned to the ground under that
    /// camera, gives its range inside the scan's box. A lift truck looks with two eyes instead (`fork_cam_check`).
    fn check_gaze(&mut self, t: &mut Task, id: u32) {
        if noto_core::act::fork::applies(&self.world.body) {
            self.fork_cam_check(t, id);
            return;
        }
        let Some(s) = self.sight(t, id, "head") else { return };
        let how = format!("고개 {:.0}° 숙여", s.neck_deg);
        self.fuse_sighting(t, id, &s, &how);
        self.gaze_check = Some(s.gaze_check());
    }

    /// One look at the drop `id` through the camera the world shows now (the head camera, or the carriage camera
    /// when `world.fork_cam`): the scan's match of its kind nearest the memory, where it puts it, and the depth
    /// model's range inside its box.
    fn sight(&mut self, t: &mut Task, id: u32, camera: &'static str) -> Option<crate::vision::Sighting> {
        let m = t.mem.get(&id).copied()?;
        let kind = self.world.drop_item(id).map(|d| d.kind)?;
        let (rate, odo) = (self.drift_rate(), self.world.odom.travelled_m);
        let before = m.sigma_m(odo, rate);
        let name = kind.spec().label_ko.to_string();
        let hv = self.world.head_view(&self.cam);
        let (neck_deg, pitch_deg) = (self.world.current_kin_pose().neck_deg, hv.view.pitch.to_degrees());
        let scanned = self.scan_now("gaze").is_ok();
        t.looks += 1;
        self.vstats.looks += 1;
        let mem = self.mem_ctx();
        let seen: Vec<(f32, [f32; 4], (f32, f32), f32)> = self
            .last_scan
            .as_ref()
            .filter(|_| scanned)
            .map(|s| {
                s.judged
                    .iter()
                    .filter(|j| (j.accepted || j.sound()) && j.cand.kind == EntKind::Drop && j.cand.link == kind.id())
                    .filter_map(|j| j.cand.world.map(|w| (j.cand.score, j.cand.bbox, w, j.cand.depth_m)))
                    .collect()
            })
            .unwrap_or_default();
        let off = |w: (f32, f32)| (w.0 - m.x).hypot(w.1 - m.z);
        let nearest = seen.iter().copied().min_by(|a, b| off(a.2).total_cmp(&off(b.2)));
        // Data association: the look fixes the memory only with what lies inside the memory's and the look's own
        // uncertainty. Several of one kind lie close together (a felled tree drops its logs a hand apart); the
        // nearest one outside that gate is a neighbour, and the memory stays as it was.
        let matched = nearest.filter(|c| off(c.2) <= (3.0 * before.hypot(mem.sigma_obs(c.3))).max(0.06));
        let range_m = nearest.map(|c| self.dist_me(c.2 .0, c.2 .1));
        let depth_m = nearest.and_then(|c| self.depth_in_box(c.1));
        let screen = match camera {
            "fork" => "포크 카메라 화면",
            _ if noto_core::act::fork::applies(&self.world.body) => "탑승자가 숙여 본 화면",
            _ => "숙인 화면",
        };
        let reason = match (nearest, matched) {
            (None, _) => Some(format!("{screen}에서 {name}을 찾지 못함")),
            (Some(c), None) => Some(format!("{name}이 보이지만 기억과 {:.0} cm 떨어져 이웃 것으로 봄 — 기억 위치로 집음", off(c.2) * 100.0)),
            _ => None,
        };
        Some(crate::vision::Sighting {
            tick: self.world.tick,
            camera,
            drop: id,
            label_ko: name,
            ok: nearest.is_some(),
            matched: matched.is_some(),
            score: nearest.map(|c| c.0),
            at: nearest.map(|c| c.2),
            sigma_m: nearest.map(|c| mem.sigma_obs(c.3)),
            range_m,
            depth_m,
            bbox: nearest.map(|c| c.1),
            reason,
            neck_deg,
            pitch_deg,
        })
    }

    /// `sight`, and once more when the first look does not find the item itself (missed, mistaken for another kind,
    /// or only a neighbour of its kind in view), as one looks again before trusting a single glance.
    fn sight_again(&mut self, t: &mut Task, id: u32, camera: &'static str) -> Option<crate::vision::Sighting> {
        let first = self.sight(t, id, camera)?;
        if first.matched {
            return Some(first);
        }
        let second = self.sight(t, id, camera)?;
        self.log("vision", Level::Debug, format!("{}: {} — 한 번 더 봄 → {}", if camera == "fork" { "포크 카메라" } else { "탑승자 시선" }, first.reason.clone().unwrap_or_default(), if second.matched { "찾음" } else { "여전히 못 찾음" }), Some(json!({"drop": id, "camera": camera})));
        Some(if second.matched || (second.ok && !first.ok) { second } else { first })
    }

    /// Fixes the memory of `id` with a sighting that matched it, and logs the look (`how` says how it looked).
    fn fuse_sighting(&mut self, t: &mut Task, id: u32, s: &crate::vision::Sighting, how: &str) {
        let Some(m) = t.mem.get(&id).copied() else { return };
        let (rate, odo) = (self.drift_rate(), self.world.odom.travelled_m);
        let before = m.sigma_m(odo, rate);
        let depth = s.depth_m.map(|d| format!(" · 깊이 모델 {d:.2} m")).unwrap_or_default();
        let data = json!({"drop": id, "camera": s.camera, "neck_deg": s.neck_deg, "pitch_deg": s.pitch_deg, "range_m": s.range_m, "depth_m": s.depth_m});
        match (s.matched, s.at, s.sigma_m) {
            (true, Some((x, z)), Some(sigma)) => {
                let (f, innov) = m.fuse(x, z, sigma, odo, self.world.tick, rate);
                t.mem.insert(id, f);
                self.note_look("gaze", format!("d{id}"), Some(before), Some(f.sigma_m(odo, rate)), Some(innov));
                let msg = format!("비전 확인 ✓ {}: {how} {:.2} m에서 봄{depth} — 기억 ±{:.0} → ±{:.0} cm", s.label_ko, s.range_m.unwrap_or(f32::NAN), 2.0 * before * 100.0, 2.0 * f.sigma_m(odo, rate) * 100.0);
                self.log("vision", Level::Info, msg, Some(data));
            }
            _ if s.ok => {
                let msg = format!("비전 확인 ✓ {} 있음: {how} 봄{depth} · {}", s.label_ko, s.reason.clone().unwrap_or_default());
                self.log("vision", Level::Info, msg, Some(data));
            }
            _ => {
                let msg = format!("비전 확인 실패: {} — 기억(±{:.0} cm)으로 집음", s.reason.clone().unwrap_or_default(), 2.0 * before * 100.0);
                self.log("vision", Level::Warn, msg, Some(data));
            }
        }
    }

    /// A lift truck's first eye on the item: the rider looks down at it over the dashboard while the tines are still
    /// up (the fork look's "rider_check"), with the head camera.
    fn rider_check(&mut self, t: &mut Task, id: u32) {
        self.world.fork_cam = false;
        let Some(s) = self.sight_again(t, id, "head") else { return };
        let how = format!("탑승자가 고개 {:.0}° 숙여 (포크 {:.2} m)", s.neck_deg, self.world.fork_m_now());
        self.fuse_sighting(t, id, &s, &how);
        self.gaze_check = Some(s.gaze_check());
        self.fork_cross = Some(crate::vision::CrossCheck { rider: Some(s), ..crate::vision::CrossCheck::new(id) });
    }

    /// A lift truck's second eye on the item: the tines are down and the carriage camera between them looks at it
    /// (the end of the fork look), checked against what the rider saw. The carriage camera, close and low, fixes the
    /// memory; when the two disagree it is the one believed, and the log says so.
    fn fork_cam_check(&mut self, t: &mut Task, id: u32) {
        self.world.fork_cam = true;
        let s = self.sight_again(t, id, "fork");
        self.world.fork_cam = false;
        let Some(s) = s else { return };
        let mut x = self.fork_cross.take().filter(|c| c.drop == id).unwrap_or_else(|| crate::vision::CrossCheck::new(id));
        x.fork_vs_rider = x.rider.as_ref().filter(|r| r.ok && r.matched && s.ok && s.matched).and_then(|r| crate::vision::cross_gap(r, &s));
        x.fork = Some(s.clone());
        let how = format!("포크 {:.2} m · 포크 카메라로 (탑승자 시선 정면)", self.world.fork_m_now());
        self.fuse_sighting(t, id, &s, &how);
        let (level, msg) = match (x.verdict(), x.fork_vs_rider) {
            ("agree", Some((d, g))) => (Level::Info, format!("교차 검증 ✓ {}: 탑승자 시선과 포크 카메라가 같은 자리 ({:.0} cm 차이 ≤ {:.0} cm)", s.label_ko, d * 100.0, g * 100.0)),
            ("agree", None) => (Level::Info, format!("교차 검증 ✓ {}: 탑승자 시선과 포크 카메라 모두 봄", s.label_ko)),
            ("disagree", Some((d, g))) => (Level::Warn, format!("교차 검증 ✗ {}: 두 시점이 {:.0} cm 어긋남 (> {:.0} cm) — 가까운 포크 카메라를 믿음", s.label_ko, d * 100.0, g * 100.0)),
            ("fork_only", _) => (Level::Warn, format!("교차 검증 △ {}: 탑승자 시선에는 없고 포크 카메라만 봄 — 포크 카메라를 믿음", s.label_ko)),
            ("rider_only", _) => (Level::Warn, format!("교차 검증 △ {}: 포크 카메라에는 없고 탑승자만 봄 — 탑승자 시선으로 끼움", s.label_ko)),
            _ => (Level::Warn, format!("교차 검증 ✗ {}: 두 시점 모두 보지 못함 — 기억으로 끼움", s.label_ko)),
        };
        self.log("vision", level, msg, Some(json!({"drop": id, "verdict": x.verdict(), "apart_m": x.fork_vs_rider.map(|v| v.0), "gate_m": x.fork_vs_rider.map(|v| v.1)})));
        self.gaze_check = Some(s.gaze_check());
        self.fork_cross = Some(x);
    }

    /// A lift truck's third look: the tines went in until the item filled the carriage camera's view, the base
    /// stopped, and the rider looks down at it again (the fork pick's "recheck"). The item must lie where the carriage
    /// camera last saw it (or the rider, when it saw nothing).
    fn rider_recheck(&mut self, t: &mut Task, id: u32) {
        let blocked = self.fork_blocked_share(id);
        self.world.fork_cam = false;
        let Some(s) = self.sight_again(t, id, "head") else { return };
        let mut x = self.fork_cross.take().filter(|c| c.drop == id).unwrap_or_else(|| crate::vision::CrossCheck::new(id));
        let last = x.fork.as_ref().filter(|f| f.ok && f.matched).or(x.rider.as_ref().filter(|r| r.ok && r.matched));
        x.recheck_vs_fork = last.filter(|_| s.ok && s.matched).and_then(|f| crate::vision::cross_gap(f, &s));
        x.blocked_share = Some(blocked);
        x.recheck = Some(s.clone());
        let name = s.label_ko.clone();
        let (level, msg) = match (s.ok && s.matched, x.recheck_vs_fork) {
            (true, Some((d, g))) if d <= g => (Level::Info, format!("재확인 ✓ {name}: 포크 카메라가 {:.0}% 가려져 탑승자가 고개 {:.0}° 숙여 봄 — 포크 위로 오는 중, 포크 카메라와 {:.0} cm 차이 ≤ {:.0} cm", blocked * 100.0, s.neck_deg, d * 100.0, g * 100.0)),
            (true, Some((d, g))) => (Level::Warn, format!("재확인 ✗ {name}: 탑승자가 본 자리가 포크 카메라와 {:.0} cm 어긋남 (> {:.0} cm)", d * 100.0, g * 100.0)),
            (true, None) => (Level::Info, format!("재확인 ✓ {name}: 포크 카메라가 {:.0}% 가려져 탑승자가 고개 {:.0}° 숙여 봄", blocked * 100.0, s.neck_deg)),
            (false, _) if s.ok => (Level::Warn, format!("재확인 △ {name}: 포크 카메라가 {:.0}% 가려져 탑승자가 봤지만 {}", blocked * 100.0, s.reason.clone().unwrap_or_default())),
            (false, _) => (Level::Warn, format!("재확인 실패: 포크 카메라가 {:.0}% 가려졌는데 탑승자 시선에도 {name}이 보이지 않음", blocked * 100.0)),
        };
        self.log("vision", level, msg, Some(json!({"drop": id, "blocked_share": blocked, "neck_deg": s.neck_deg, "apart_m": x.recheck_vs_fork.map(|v| v.0), "gate_m": x.recheck_vs_fork.map(|v| v.1)})));
        self.gaze_check = Some(s.gaze_check());
        self.fork_cross = Some(x);
    }

    /// The share of the carriage camera's image the drop `id` fills now, from a frame of that camera.
    fn fork_blocked_share(&mut self, id: u32) -> f32 {
        self.world.fork_cam = true;
        let f = self.world.render(&self.cam);
        self.world.fork_cam = false;
        let want = world_sim::render::pack_id(world_sim::ObjectClass::Drop, id);
        f.ids.iter().filter(|&&i| i == want).count() as f32 / f.ids.len().max(1) as f32
    }

    /// How far from the robot's feet the thing filling the middle of `bbox` (normalised) lies, by the depth model
    /// aligned under the current camera pose.
    fn depth_in_box(&mut self, bbox: [f32; 4]) -> Option<f32> {
        let frame = self.world.render(&self.cam);
        let rgb = noto_vision::Rgb::new(frame.width as u32, frame.height as u32, frame.rgb.clone());
        let intr = crate::record::intrinsics(&self.world, &self.cam);
        let pose = crate::record::odometry(&self.world, &self.cam);
        let truth = crate::ml::KeyTruth::capture(&self.world, &frame);
        let fix = self.ml.fix.clone();
        let depth = match &mut self.ml.depth {
            crate::ml::DepthEngine::Model(m) => {
                let max = m.max_depth_m();
                let s = m.infer_scene(&rgb, crate::ml::scene_heads(&fix)).ok()?;
                let mut d = s.depth;
                if let Some(cfg) = &fix.align {
                    noto_vision::ground_align_with(&mut d, s.normals.as_ref().filter(|_| fix.normals), &intr, &pose, None, max, cfg)?;
                }
                d
            }
            crate::ml::DepthEngine::StandIn(s) => s.run(&truth),
        };
        let (w, h) = (frame.width as f32, frame.height as f32);
        let mut mask = noto_vision::Mask::empty(depth.width, depth.height);
        let (sx, sy) = (depth.width as f32 / w, depth.height as f32 / h);
        let (x0, x1) = ((((bbox[0] * 0.75 + bbox[2] * 0.25) * w) * sx) as u32, (((bbox[0] * 0.25 + bbox[2] * 0.75) * w) * sx) as u32);
        let (y0, y1) = ((((bbox[1] * 0.75 + bbox[3] * 0.25) * h) * sy) as u32, (((bbox[1] * 0.25 + bbox[3] * 0.75) * h) * sy) as u32);
        for y in y0..=y1.min(depth.height - 1) {
            for x in x0..=x1.min(depth.width - 1) {
                mask.set(x, y);
            }
        }
        let d = depth.median_in(&mask)?;
        let (cx, cy) = ((bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h);
        let (x, z) = intr.ground_point(&pose, cx, cy, d)?;
        Some(self.dist_me(x, z))
    }

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

    fn push_step(&mut self, t: &mut Task) -> Flow {
        let Some(job) = t.push else { return Flow::Continue };
        let Some((wid, wx, wz, place)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let name = t.target().label_ko.clone();
        let Some(d) = self.world.drop_item(job.drop).cloned() else {
            t.push = None;
            t.drop_idx += 1;
            return Flow::Continue;
        };
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
        }
        let (tx, tz) = self.world.true_of(wx, wz);
        let (_, (lx, lz, yaw)) = match self.world.push_route(job.drop, tx, tz) {
            Ok(r) => r,
            Err(e) => return fail(e.code(), format!("{name}{} {place}까지 밀 수 없습니다: {}", eul_reul(&name), action_error_ko(&e))),
        };
        let (lx, lz) = self.world.belief_of(lx, lz);
        let yaw = yaw + (self.me().2 - self.world.agent.yaw);
        let off = self.dist_me(lx, lz);
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
            Err(ActionError::NoPath) => None,
            Err(e) => Some(fail(e.code(), action_error_ko(&e))),
        }
    }

    fn reach_for_place(&mut self, t: &mut Task, why: MoveFor) -> Flow {
        let Some((_, x, z, name)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        t.deliver_retry = true;
        if self.scan_now("place").is_ok() {
            t.looks += 1;
            self.vstats.looks += 1;
        }
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

    fn after_goto(&mut self, t: &mut Task) -> Flow {
        if t.kind == "drop" {
            self.goto_phase(t, Phase::Unload, "");
            return Flow::Continue;
        }
        Flow::Done
    }

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
                self.log("quest", Level::Warn, format!("적재함에 {}{} 없어 건너뜀", item.spec().label_ko, noto_search::text::i_ga(item.spec().label_ko)), None);
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
            return fail("nothing_to_drop", format!("적재함에 {}{} 없습니다", names.join(", "), noto_search::text::i_ga(names.last().copied().unwrap_or("물건"))));
        }
        Flow::Done
    }

    fn while_running(&mut self, t: &mut Task, act: &Act) {
        match act {
            Act::Move(MoveFor::Approach(_)) => {
                let me = self.me();
                let eye = self.world.standing_eye_m();
                let target = t.target_lm.and_then(|id| self.gates.landmark(id)).map(|l| (l.x, l.z, l.distance_to(me.0, me.1)));
                let target_gaze = target.and_then(|(x, z, _)| gaze_to_see(&self.cam, eye, me, x, z));
                if !self.cfg.vision.intermittent {
                    if t.approach_scan_cd == 0 {
                        t.approach_scan_cd = self.cfg.approach_scan_every_ticks;
                        let (pan, tilt) = target_gaze.unwrap_or((0.0, 0.0));
                        if self.scan_gaze("approach", pan, tilt).is_ok() {
                            let (range, _) = self.search_reach(t);
                            self.mark_coverage_as(&mut t.coverage, tilt, range);
                        }
                    } else {
                        t.approach_scan_cd -= 1;
                    }
                } else {
                    self.vstats.approach_s += world_sim::TICK_S as f64;
                    let lead_m = self.world.agent.speed_mps.max(0.1) * EYE_LEAD_S + PICKUP_REACH_M;
                    let needs_down = target_gaze.is_some_and(|(_, tilt)| tilt > 0.0);
                    let pre = !t.pre_looked && target.is_some_and(|(_, _, d)| d <= lead_m) && (needs_down || target.is_some_and(|(_, _, d)| d <= PICKUP_REACH_M + 0.4));
                    let due = t.look_cd == 0;
                    if pre || due {
                        let kind = if pre { "pre_act" } else { "pattern" };
                        if pre {
                            t.pre_looked = true;
                        }
                        t.look_cd = self.cfg.vision.look_every_ticks;
                        let (pan, tilt) = target_gaze.unwrap_or((0.0, 0.0));
                        if self.scan_gaze(kind, pan, tilt).is_ok() {
                            t.looks += 1;
                            self.vstats.looks += 1;
                            self.vstats.approach_looks += 1;
                            let (range, _) = self.search_reach(t);
                            self.mark_coverage_as(&mut t.coverage, tilt, range);
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
            // A lift truck looks with its rider's eyes while the act holds still for them: before the tines go down
            // ("rider_check"), and once the item fills the carriage camera's view ("recheck").
            Act::Look(id) | Act::Pick(id) if noto_core::act::fork::applies(&self.world.body) => {
                let id = *id;
                match self.world.act_phase().map(|p| p.1) {
                    Some("rider_check") if t.rider_looked != Some(t.act_started_s) => {
                        t.rider_looked = Some(t.act_started_s);
                        self.rider_check(t, id);
                    }
                    Some("recheck") if t.rechecked != Some(t.act_started_s) => {
                        t.rechecked = Some(t.act_started_s);
                        self.rider_recheck(t, id);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

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
                    t.gazed = None;
                    if why == "grasp_missed" {
                        if let (Some(m), Some(d)) = (t.mem.get_mut(&id), self.world.drop_item(id)) {
                            let size = d.kind.spec().piece_size_m(d.amount);
                            m.sigma0_m = m.sigma0_m.max(0.5 * size[0].min(size[2]));
                        }
                    }
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
            return match act {
                Act::Move(MoveFor::Approach(id)) => {
                    self.gates.note_unreachable(id);
                    self.goto_phase(t, Phase::Locate, &format!("접근 실패({why}), 다른 대상을 찾음"));
                    t.reset_locate();
                    Flow::Continue
                }
                Act::Move(MoveFor::Search) | Act::Turn | Act::Look(_) => Flow::Continue,
                _ => fail(&why, format!("동작이 실패했습니다: {why}")),
            };
        }
        if matches!(act, Act::Move(_)) && self.world.body.base == noto_core::BaseKind::HumanFoot {
            self.sds.record_agreement(&Scope::new(Track::Act, "walk", self.world.ground_here().id()), "no_fall", true);
        }
        match act {
            Act::Turn => {
                t.scan_cooldown = 0;
                Flow::Continue
            }
            Act::Look(id) => {
                self.check_gaze(t, id);
                Flow::Continue
            }
            Act::Move(MoveFor::Search) => {
                t.scan_cooldown = 0;
                t.scans_here = 0;
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
            let vscope = self.gates.vision_scope(&l.label);
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
