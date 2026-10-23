//! Carries a quest out in the world, one tick at a time.
//!
//! The driver never reads the world's object list to find what it should go
//! for. It looks (scan), keeps what the gates confirmed as landmarks, drives
//! up to one, and then touches: a contact sensor reports the nearest node in
//! reach, and breaking it tells which kind it really was. That outcome feeds
//! the SDS, which is how the gates learn from mistakes.
//!
//! Two things it is told rather than perceives, both assumptions of this
//! simulator: where the waypoints are (they are on the map), and which drops a
//! node just released (the event says so, and they land within reach).

use hand_quest::{QuestResult, State};
use hand_sds::{Level, Scope, Track};
use hand_search::Plan;
use serde_json::json;
use world_sim::nav::Goal;
use world_sim::world::{DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{Action, ActionError, ItemKind, NodeKind, TimedEvent, WorldEvent};

use hand_search::text::eul_reul;

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
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MoveFor {
    Approach(u32),
    Search,
    Drop,
    Carry,
    Goto,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Act {
    Move(MoveFor),
    Turn,
    Mine(u32),
    Pick(u32),
    Deliver(ItemKind),
}

#[derive(Clone, Debug)]
pub(crate) struct TaskTarget {
    pub resource: String,
    pub item: ItemKind,
    pub wanted: u32,
    pub picked: u32,
    pub label_ko: String,
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
    // collecting
    pub drops: Vec<u32>,
    pub drop_idx: usize,
    pub rounds: u32,
    pub pick_retry: bool,
    // delivering
    pub deliver_idx: usize,
    /// Times each quest step was entered (index as in the quest).
    pub attempts: Vec<u32>,
    // result
    pub touch_ok: u32,
    pub touch_miss: u32,
    pub wasted: u32,
    pub delivered: Vec<(String, u32)>,
}

impl Task {
    pub(crate) fn from_plan(quest: &str, plan: &Plan, world: &world_sim::World) -> Result<Task, (String, String)> {
        let mut targets = Vec::new();
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
            Plan::Stop => return Err(("stop".into(), "중지 명령은 퀘스트가 아닙니다".into())),
        };
        if let Some(d) = &destination {
            if world.waypoint_by_key(d).is_none() {
                return Err(("unknown_target".into(), format!("지도에 '{d}' 지점이 없습니다")));
            }
        }
        let phase = if kind == "goto" { Phase::Goto } else { Phase::Locate };
        let n_steps = match kind {
            "fetch" => 4 * targets.len() + 2,
            "collect" => 4 * targets.len(),
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
            drops: Vec::new(),
            drop_idx: 0,
            rounds: 0,
            pick_retry: false,
            deliver_idx: 0,
            attempts: {
                let mut a = vec![0u32; n_steps];
                a[0] = 1;
                a
            },
            touch_ok: 0,
            touch_miss: 0,
            wasted: 0,
            delivered: Vec::new(),
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
    }

    fn precision(&self) -> Option<f64> {
        let n = self.touch_ok + self.touch_miss;
        (n > 0).then(|| self.touch_ok as f64 / n as f64)
    }
}

fn make_target(t: &hand_search::Target) -> Result<TaskTarget, (String, String)> {
    let item = ItemKind::from_id(&t.resource).ok_or_else(|| ("unknown_target".to_string(), format!("'{}'은(는) 이 월드에 없는 자원입니다", t.resource)))?;
    Ok(TaskTarget { resource: t.resource.clone(), item, wanted: t.quantity.max(1), picked: 0, label_ko: item.spec().label_ko.to_string() })
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
            self.focus = None;
            return;
        };
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
                WorldEvent::ItemPicked { item, .. } => {
                    if let Some(x) = t.targets.iter_mut().find(|x| x.item == *item) {
                        x.picked += 1;
                    }
                }
                WorldEvent::Delivered { item, count, .. } => {
                    let id = item.id().to_string();
                    match t.delivered.iter_mut().find(|(k, _)| *k == id) {
                        Some(e) => e.1 += *count,
                        None => t.delivered.push((id, *count)),
                    }
                }
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
        }
    }

    fn issue(&mut self, t: &mut Task, action: Action, act: Act) -> Result<(), ActionError> {
        match self.world.command(action.clone()) {
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
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        self.gates
            .landmarks()
            .iter()
            .filter(|l| l.confirmed && l.state == LmState::Active && l.kind == EntKind::Source && &l.link == want)
            .min_by(|a, b| a.distance_to(ax, az).total_cmp(&b.distance_to(ax, az)))
            .map(|l| l.id)
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
            let note = format!("탐색 중 · 회전 {}/{}회 · 이동 {}회", t.rotations, rot_max, t.relocations);
            self.progress(t, t.rotations as f32 / (rot_max + 1) as f32, Some(note));
            return match self.issue(t, Action::Turn { yaw_deg: yaw }, Act::Turn) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        // A full turn without finding it: move to the nearest unvisited spot.
        t.visited.push((self.world.agent.x, self.world.agent.z));
        t.relocations += 1;
        if t.relocations > self.cfg.max_relocations {
            let name = t.target().label_ko.clone();
            return fail("locate_timeout", format!("{name}{} 찾지 못했습니다 (장소 {}곳을 둘러봄)", eul_reul(&name), t.relocations - 1));
        }
        t.rotations = 0;
        t.scans_here = 0;
        loop {
            let Some((x, z)) = self.next_search_point(t) else {
                let name = t.target().label_ko.clone();
                return fail("locate_timeout", format!("{name}{} 찾지 못했습니다 (더 갈 곳이 없음)", eul_reul(&name)));
            };
            t.visited.push((x, z));
            self.log("quest", Level::Info, format!("{} 탐색: ({x:.0}, {z:.0})로 이동", t.target().label_ko), None);
            match self.issue(t, Action::MoveTo { x, z, reach_m: 1.0 }, Act::Move(MoveFor::Search)) {
                Ok(()) => return Flow::Continue,
                Err(ActionError::NoPath) | Err(ActionError::UnknownTarget) => continue,
                Err(e) => return fail(e.code(), action_error_ko(&e)),
            }
        }
    }

    /// Nearest point of an 8 m grid that has not been visited, nearest first.
    fn next_search_point(&self, t: &Task) -> Option<(f32, f32)> {
        let (w, d) = self.world.terrain.size_m();
        let step = 8.0;
        let mut best: Option<((f32, f32), f32)> = None;
        let mut gx = step * 0.5;
        while gx < w - 2.0 {
            let mut gz = step * 0.5;
            while gz < d - 2.0 {
                if t.visited.iter().all(|(vx, vz)| ((vx - gx).powi(2) + (vz - gz).powi(2)).sqrt() > step * 0.6) {
                    let dist = ((gx - self.world.agent.x).powi(2) + (gz - self.world.agent.z).powi(2)).sqrt();
                    if best.map_or(true, |(_, bd)| dist < bd) {
                        best = Some(((gx, gz), dist));
                    }
                }
                gz += step;
            }
            gx += step;
        }
        best.map(|(p, _)| p)
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
        let dist = ((x - self.world.agent.x).powi(2) + (z - self.world.agent.z).powi(2)).sqrt();
        if dist <= MINE_REACH_M - 0.1 {
            self.goto_phase(t, Phase::Mine, "");
            return Flow::Continue;
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: self.cfg.approach_reach_m }).map(|p| p.time_s);
        t.act_distance0 = dist;
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

    /// The node the hand feels, nearest to where the target is believed to be.
    fn touch_node(&self, believed: (f32, f32)) -> Option<u32> {
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        self.world
            .nodes
            .iter()
            .filter(|n| ((n.x - ax).powi(2) + (n.z - az).powi(2)).sqrt() <= MINE_REACH_M)
            .min_by(|a, b| {
                let da = (a.x - believed.0).powi(2) + (a.z - believed.1).powi(2);
                let db = (b.x - believed.0).powi(2) + (b.z - believed.1).powi(2);
                da.total_cmp(&db)
            })
            .map(|n| n.id)
    }

    fn do_mine(&mut self, t: &mut Task) -> Flow {
        let believed = t.target_lm.and_then(|id| self.gates.landmark(id)).map(|l| (l.x, l.z)).unwrap_or((self.world.agent.x, self.world.agent.z));
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

    fn do_collect(&mut self, t: &mut Task) -> Flow {
        loop {
            let (need_more, name) = {
                let x = t.target();
                (x.picked < x.wanted, x.label_ko.clone())
            };
            if !need_more || t.drop_idx >= t.drops.len() {
                let (picked, wanted) = (t.target().picked, t.target().wanted);
                if picked >= wanted {
                    return self.next_target(t);
                }
                t.rounds += 1;
                if t.rounds > self.cfg.max_rounds {
                    return fail("not_enough", format!("{name} {wanted}개 중 {picked}개만 모았습니다 (더 캘 곳을 찾지 못함)"));
                }
                self.goto_phase(t, Phase::Locate, &format!("{name} {picked}/{wanted}개, 더 찾음"));
                t.reset_locate();
                t.locate_since_s = self.world.time_s;
                return Flow::Continue;
            }
            let id = t.drops[t.drop_idx];
            let Some((dx, dz)) = self.world.drop_item(id).map(|d| (d.x, d.z)) else {
                t.drop_idx += 1;
                continue;
            };
            let dist = ((dx - self.world.agent.x).powi(2) + (dz - self.world.agent.z).powi(2)).sqrt();
            if dist > PICKUP_REACH_M - 0.1 {
                t.act_predicted_s = self.world.plan_route(Goal::Within { x: dx, z: dz, reach_m: 0.4 }).map(|p| p.time_s);
                return match self.issue(t, Action::MoveTo { x: dx, z: dz, reach_m: 0.4 }, Act::Move(MoveFor::Drop)) {
                    Ok(()) => Flow::Continue,
                    Err(ActionError::NoPath) => {
                        self.log("quest", Level::Warn, format!("떨어진 {name}에게 갈 경로가 없어 건너뜀"), None);
                        t.drop_idx += 1;
                        continue;
                    }
                    Err(e) => fail(e.code(), action_error_ko(&e)),
                };
            }
            return match self.issue(t, Action::PickUp { drop: id }, Act::Pick(id)) {
                Ok(()) => Flow::Continue,
                Err(ActionError::OutOfReach { .. }) if !t.pick_retry => {
                    t.pick_retry = true;
                    match self.issue(t, Action::MoveTo { x: dx, z: dz, reach_m: 0.2 }, Act::Move(MoveFor::Drop)) {
                        Ok(()) => Flow::Continue,
                        Err(e) => fail(e.code(), action_error_ko(&e)),
                    }
                }
                Err(ActionError::CapacityExceeded { carried_kg, item_kg, capacity_kg }) => {
                    let fit = ((capacity_kg - carried_kg + 1e-3) / item_kg).floor().max(0.0);
                    let picked = t.target().picked;
                    fail(
                        "capacity_exceeded",
                        format!("적재 한도를 넘습니다 — 이 베이스는 {name}{} {}개까지 실을 수 있습니다 (현재 {picked}개 실음)", eul_reul(&name), picked + fit as u32),
                    )
                }
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
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
        let reach = DELIVER_REACH_M - 0.5;
        let dist = ((x - self.world.agent.x).powi(2) + (z - self.world.agent.z).powi(2)).sqrt();
        if dist <= reach + 0.2 {
            self.goto_phase(t, Phase::Deliver, "");
            return Flow::Continue;
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: reach }).map(|p| p.time_s);
        t.act_distance0 = dist;
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
            if self.world.carried_count(item) == 0 {
                t.deliver_idx += 1;
                continue;
            }
            return match self.issue(t, Action::Deliver { waypoint: wid, item: Some(item) }, Act::Deliver(item)) {
                Ok(()) => Flow::Continue,
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        Flow::Done
    }

    fn do_goto(&mut self, t: &mut Task) -> Flow {
        let Some((_, x, z, name)) = self.waypoint_of(t) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let reach = DELIVER_REACH_M - 0.5;
        let dist = ((x - self.world.agent.x).powi(2) + (z - self.world.agent.z).powi(2)).sqrt();
        if dist <= reach + 0.2 {
            return Flow::Done;
        }
        t.act_predicted_s = self.world.plan_route(Goal::Within { x, z, reach_m: reach }).map(|p| p.time_s);
        t.act_distance0 = dist;
        match self.issue(t, Action::MoveTo { x, z, reach_m: reach }, Act::Move(MoveFor::Goto)) {
            Ok(()) => Flow::Continue,
            Err(ActionError::NoPath) => fail("no_path", format!("{name}까지 갈 경로가 없습니다 (이 베이스로는 지나갈 수 없는 지형)")),
            Err(e) => fail(e.code(), action_error_ko(&e)),
        }
    }

    // ----- while an action runs -----

    fn while_running(&mut self, t: &mut Task, act: &Act) {
        match act {
            Act::Move(MoveFor::Approach(_)) => {
                if t.approach_scan_cd == 0 {
                    t.approach_scan_cd = self.cfg.approach_scan_every_ticks;
                    let _ = self.scan_now("approach");
                } else {
                    t.approach_scan_cd -= 1;
                }
                if let Some(id) = t.target_lm {
                    if let Some(l) = self.gates.landmark(id) {
                        let d = l.distance_to(self.world.agent.x, self.world.agent.z);
                        let p = 1.0 - (d / t.act_distance0.max(0.1)).clamp(0.0, 1.0);
                        self.progress(t, p, Some(format!("남은 거리 {d:.1} m")));
                    }
                }
            }
            Act::Move(MoveFor::Carry) | Act::Move(MoveFor::Goto) => {
                if let Some((_, x, z, _)) = self.waypoint_of(t) {
                    let d = ((x - self.world.agent.x).powi(2) + (z - self.world.agent.z).powi(2)).sqrt();
                    let p = 1.0 - (d / t.act_distance0.max(0.1)).clamp(0.0, 1.0);
                    self.progress(t, p, Some(format!("남은 거리 {d:.1} m")));
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
        if !ok {
            let why = reason.clone().unwrap_or_else(|| "알 수 없음".into());
            if why == "cancelled" {
                return fail("cancelled", "중지되었습니다");
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
            Act::Move(MoveFor::Carry) => {
                self.calibrate_move(t, elapsed);
                self.goto_phase(t, Phase::Deliver, "");
                Flow::Continue
            }
            Act::Move(MoveFor::Goto) => {
                self.calibrate_move(t, elapsed);
                Flow::Done
            }
            Act::Mine(node) => self.after_mine(t, node, elapsed),
            Act::Pick(_) => {
                t.drop_idx += 1;
                t.pick_retry = false;
                let (picked, wanted) = (t.target().picked, t.target().wanted);
                self.progress(t, picked as f32 / wanted as f32, Some(format!("{picked}/{wanted}개")));
                Flow::Continue
            }
            Act::Deliver(_) => {
                t.deliver_idx += 1;
                Flow::Continue
            }
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
        let items: Vec<(String, u32)> = if t.kind == "fetch" { t.delivered.clone() } else { t.targets.iter().map(|x| (x.resource.clone(), x.picked)).collect() };
        QuestResult {
            ticks: self.world.tick - t.started_tick,
            time_s: self.world.time_s - t.started_time_s,
            energy_j: self.world.agent.energy_used_j - t.started_energy_j,
            precision: t.precision(),
            summary,
            items,
            wasted_actions: t.wasted,
        }
    }

    fn finish_task(&mut self, t: Task) {
        let items_ko = |items: &[(String, u32)]| -> String {
            items
                .iter()
                .map(|(r, n)| format!("{} {n}개", ItemKind::from_id(r).map(|k| k.spec().label_ko.to_string()).unwrap_or_else(|| r.clone())))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let dest_ko = t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| w.label_ko.clone()).unwrap_or_default();
        let mut r = self.result_of(&t, String::new());
        let tail = format!(
            "{:.1}초 · {:.0} J{}{}",
            r.time_s,
            r.energy_j,
            r.precision.map(|p| format!(" · 접촉 확인 정밀도 {:.0}%", p * 100.0)).unwrap_or_default(),
            if r.wasted_actions > 0 { format!(" · 헛수고 {}회", r.wasted_actions) } else { String::new() }
        );
        r.summary = match t.kind {
            "fetch" => format!("{}{} {dest_ko}에 인도했습니다 — {tail}", items_ko(&r.items), eul_reul(&items_ko(&r.items))),
            "collect" => format!("{} 채취 완료 — {tail}", items_ko(&r.items)),
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
