use std::collections::{HashMap, VecDeque};

use noto_quest::{QuestResult, State};
use noto_sds::{Level, Scope, Track};
use noto_search::{DropTarget, Plan};
use noto_ts::{Predictor, TtmPredictor};
use noto_vision::{DecideCfg, Decider, Decision, TriStep};
use serde_json::json;
use world_sim::explore::{line_of_sight, Coverage, TARGET_H_M};
use world_sim::world::{AMOUNT_EPS, DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{amount_text, Action, ActionError, ItemKind, NodeKind, TimedEvent, WorldEvent};

use crate::cargo::{self, CargoAction, CargoCtx};
use crate::driver::{action_error_ko, MAX_FALLS, MAX_GRASP_TRIES, MAX_PUSH_TRIES, PLACE_REACH_M};
use crate::vision::{gaze_to_see, memory_ok, Memory};

const TOUCH_FROM_ESTIMATE_M: f32 = 0.8;
const ARRIVE_SURFACE_M: f32 = MINE_REACH_M - 0.4;
use crate::record::{capture_keyframe, DecisionView};
use crate::runtime::Runtime;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MlPhase {
    Look,
    Mine,
    Collect,
    Carry,
    Deliver,
    Goto,
    Unload,
}

pub(crate) struct MlTarget {
    pub resource: String,
    pub item: ItemKind,
    pub wanted: u32,
    pub picked: f32,
    pub label_ko: String,
    pub labels: Vec<String>,
}

pub(crate) struct MlTask {
    pub quest: String,
    pub kind: &'static str,
    pub targets: Vec<MlTarget>,
    pub destination: Option<String>,
    pub cur: usize,
    pub phase: MlPhase,
    pub decider: Decider,
    queue: VecDeque<Action>,
    running: bool,
    done: Option<(bool, Option<String>)>,
    broken: Option<(String, NodeKind, Vec<String>)>,
    drops: Vec<u32>,
    drop_idx: usize,
    deliver_idx: usize,
    pub decisions: u32,
    holds: u32,
    arrive_tries: u32,
    fixed: bool,
    look_since_s: f64,
    pub attempts: Vec<u32>,
    started_tick: u64,
    started_time_s: f64,
    started_energy_j: f64,
    wasted: u32,
    touch_ok: u32,
    touch_miss: u32,
    delivered: Vec<(String, f32)>,
    pending_kf: Option<usize>,
    coverage: Coverage,
    visited: Vec<(f32, f32)>,
    split_piece: Option<u32>,
    trips: u32,
    drop_items: Vec<DropTarget>,
    unloaded: Vec<(String, f32)>,
    unload_started: bool,
    grasp_tries: u32,
    falls: u32,
    push: Option<(u32, u32)>,
    last_pick: Option<u32>,
    mem: HashMap<u32, Memory>,
    blind_run: u32,
    pub blind: u32,
    look_cd: u32,
    deliver_retry: bool,
}

enum Flow {
    Continue,
    Done,
    Fail(String, String),
}

fn fail(code: &str, msg: impl Into<String>) -> Flow {
    Flow::Fail(code.into(), msg.into())
}

impl MlTask {
    fn step_of(&self, phase: MlPhase) -> usize {
        let n = self.targets.len();
        match phase {
            MlPhase::Look if self.fixed => 4 * self.cur + 1,
            MlPhase::Look => 4 * self.cur,
            MlPhase::Mine => 4 * self.cur + 2,
            MlPhase::Collect => 4 * self.cur + 3,
            MlPhase::Carry => 4 * n,
            MlPhase::Deliver => 4 * n + 1,
            MlPhase::Goto => 0,
            MlPhase::Unload => usize::from(self.destination.is_some()),
        }
    }

    fn target(&self) -> &MlTarget {
        &self.targets[self.cur.min(self.targets.len().saturating_sub(1))]
    }
}

impl Runtime {
    fn ml_task_from(&self, quest: &str, plan: &Plan) -> Result<MlTask, (String, String)> {
        let mut targets = Vec::new();
        let mut drop_items = Vec::new();
        let (kind, destination, list) = match plan {
            Plan::Fetch { targets: ts, destination } => ("fetch", Some(destination.place.clone()), ts.clone()),
            Plan::Collect { targets: ts } => ("collect", None, ts.clone()),
            Plan::Goto { place } => ("goto", Some(place.place.clone()), vec![]),
            Plan::Drop { items, place } => {
                drop_items = items.clone();
                ("drop", place.as_ref().map(|p| p.place.clone()), vec![])
            }
            Plan::Stop => return Err(("stop".into(), "중지 명령은 퀘스트가 아닙니다".into())),
        };
        for t in &list {
            let item = ItemKind::from_id(&t.resource).ok_or_else(|| ("unknown_target".to_string(), format!("'{}'은(는) 이 월드에 없는 자원입니다", t.resource)))?;
            let labels = if item.from_node() { self.ml.source_labels_for(&t.resource) } else { vec![t.resource.clone()] };
            targets.push(MlTarget { resource: t.resource.clone(), item, wanted: t.quantity.max(1), picked: 0.0, label_ko: item.spec().label_ko.into(), labels });
        }
        let n_steps = self.quests.get(quest).map_or(0, |q| q.steps.len());
        let phase = match kind {
            "goto" => MlPhase::Goto,
            "drop" if destination.is_some() => MlPhase::Goto,
            "drop" => MlPhase::Unload,
            _ => MlPhase::Look,
        };
        Ok(MlTask {
            quest: quest.into(),
            kind,
            targets,
            destination,
            cur: 0,
            phase,
            decider: Decider::new(DecideCfg { reach_m: ARRIVE_SURFACE_M, ..DecideCfg::default() }),
            queue: VecDeque::new(),
            running: false,
            done: None,
            broken: None,
            drops: vec![],
            drop_idx: 0,
            deliver_idx: 0,
            decisions: 0,
            holds: 0,
            arrive_tries: 0,
            fixed: false,
            look_since_s: self.world.time_s,
            attempts: {
                let mut a = vec![0; n_steps];
                if let Some(x) = a.first_mut() {
                    *x = 1;
                }
                a
            },
            started_tick: self.world.tick,
            started_time_s: self.world.time_s,
            started_energy_j: self.world.agent.energy_used_j,
            wasted: 0,
            touch_ok: 0,
            touch_miss: 0,
            delivered: vec![],
            pending_kf: None,
            coverage: Coverage::new(&self.world.terrain),
            visited: vec![],
            split_piece: None,
            trips: 0,
            drop_items,
            unloaded: vec![],
            unload_started: false,
            grasp_tries: 0,
            falls: 0,
            push: None,
            last_pick: None,
            mem: HashMap::new(),
            blind_run: 0,
            blind: 0,
            look_cd: 0,
            deliver_retry: false,
        })
    }

    pub(crate) fn ml_drive(&mut self, events: &[TimedEvent]) {
        if self.ml_task.is_none() {
            let Some(id) = self.quests.current_id() else { return };
            let Some(plan) = self.quests.get(&id).map(|q| q.plan.clone()) else { return };
            match self.ml_task_from(&id, &plan) {
                Ok(t) => {
                    let tick = self.world.tick;
                    if let Some(q) = self.quests.get_mut(&id) {
                        let _ = q.start(tick);
                    }
                    self.log("quest", Level::Info, format!("[ML] 퀘스트 {id} 시작: ML 결정(삼각 측량·시계열)으로 탐색·접근"), Some(json!({"quest": id, "pass": "ml"})));
                    self.ml_task = Some(t);
                }
                Err((code, msg)) => {
                    if let Some(q) = self.quests.get_mut(&id) {
                        let _ = q.fail(&code, &msg, None);
                    }
                    return;
                }
            }
        }
        let Some(mut t) = self.ml_task.take() else {
            self.apply_fix();
            return;
        };
        if let Some(f) = self.apply_fix() {
            let rate = self.drift_rate();
            for m in t.mem.values_mut() {
                (m.x, m.z) = f.shift(m.x, m.z, m.odo_m, rate);
            }
        }
        match self.ml_step(&mut t, events) {
            Flow::Continue => self.ml_task = Some(t),
            Flow::Done => self.ml_end(t, None),
            Flow::Fail(code, msg) => self.ml_end(t, Some((code, msg))),
        }
    }

    fn ml_goto(&mut self, t: &mut MlTask, to: MlPhase, note: &str) {
        let (a, b) = (t.step_of(t.phase), t.step_of(to));
        if let Some(q) = self.quests.get_mut(&t.quest) {
            if b > a {
                for i in a..b {
                    let _ = q.complete_step(i);
                }
            } else if b < a {
                let _ = q.rewind(b, note);
            }
        }
        if a != b {
            if let Some(x) = t.attempts.get_mut(b) {
                *x += 1;
            }
            self.log("quest", if b < a { Level::Warn } else { Level::Info }, format!("[ML] {} · 단계 {}{}", t.quest, b + 1, if note.is_empty() { String::new() } else { format!(" ({note})") }), None);
        }
        t.phase = to;
    }

    fn ml_issue(&mut self, t: &mut MlTask, a: Action) -> Result<(), ActionError> {
        match self.world.command(self.to_world(a.clone())) {
            Ok(ev) => {
                self.log_world_events(&ev);
                t.running = true;
                t.done = None;
                Ok(())
            }
            Err(e) => {
                self.log("world", Level::Warn, format!("[ML] 동작 거부 {}: {}", a.label(), action_error_ko(&e)), Some(json!({"action": a.label(), "code": e.code()})));
                Err(e)
            }
        }
    }

    fn mark_executed(&mut self, t: &MlTask, what: String, ok: bool) {
        if let (Some(i), Some(rec)) = (t.pending_kf, self.recording.as_mut()) {
            if let Some(d) = rec.keyframes.get_mut(i).and_then(|k| k.decision.as_mut()) {
                if d.executed.is_empty() {
                    d.executed = what;
                } else {
                    d.executed = format!("{} → {what}", d.executed);
                }
                d.ok = d.ok && ok;
            }
        }
    }

    fn ml_step(&mut self, t: &mut MlTask, events: &[TimedEvent]) -> Flow {
        for te in events {
            match &te.event {
                WorldEvent::ActionDone { ok, reason, .. } => t.done = Some((*ok, reason.clone())),
                WorldEvent::NodeBroken { node, kind, drops } => t.broken = Some((node.clone(), *kind, drops.clone())),
                WorldEvent::ItemPicked { item, amount, .. } => {
                    if let Some(x) = t.targets.iter_mut().find(|x| x.item == *item) {
                        x.picked += *amount;
                    }
                }
                WorldEvent::Delivered { item, amount, .. } => {
                    t.deliver_retry = false;
                    let id = item.id().to_string();
                    match t.delivered.iter_mut().find(|(k, _)| *k == id) {
                        Some(e) => e.1 += *amount,
                        None => t.delivered.push((id, *amount)),
                    }
                }
                WorldEvent::ItemDropped { item, amount, .. } => {
                    let id = item.id().to_string();
                    match t.unloaded.iter_mut().find(|(k, _)| *k == id) {
                        Some(e) => e.1 += *amount,
                        None => t.unloaded.push((id, *amount)),
                    }
                }
                WorldEvent::ItemSplit { piece, .. } => t.split_piece = piece.get(1..).and_then(|s| s.parse().ok()),
                WorldEvent::BatteryDepleted => return fail("battery_depleted", "배터리가 방전되어 멈췄습니다"),
                _ => {}
            }
        }
        if t.running {
            match t.done.take() {
                None => {
                    let to_place = matches!(t.phase, MlPhase::Carry | MlPhase::Goto) || matches!(t.queue.front(), Some(Action::Deliver { .. }));
                    if let Some((x, z)) = t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| (w.x, w.z)).filter(|_| to_place) {
                        self.place_look(&mut t.look_cd, x, z);
                    }
                    return Flow::Continue;
                }
                Some((ok, reason)) => {
                    t.running = false;
                    if !ok {
                        let why = reason.unwrap_or_else(|| "unknown".into());
                        if why == "cancelled" {
                            return fail("cancelled", "중지되었습니다");
                        }
                        if why.starts_with("fall") {
                            t.falls += 1;
                            t.queue.clear();
                            if t.falls > MAX_FALLS {
                                return fail("fall", format!("[ML] {}번 넘어져 그만둠", t.falls));
                            }
                            self.log("quest", Level::Warn, format!("[ML] 넘어짐 {}/{MAX_FALLS} — 일어나서 다시 감", t.falls), None);
                            if t.phase == MlPhase::Look {
                                t.decider.move_failed();
                            }
                            return Flow::Continue;
                        }
                        if t.phase == MlPhase::Collect && (why.starts_with("grasp") || why == "lift_overload") {
                            if why == "grasp_missed" {
                                if let Some(id) = t.last_pick {
                                    if let (Some(m), Some(d)) = (t.mem.get_mut(&id), self.world.drop_item(id)) {
                                        let size = d.kind.spec().piece_size_m(d.amount);
                                        m.sigma0_m = m.sigma0_m.max(0.5 * size[0].min(size[2]));
                                    }
                                }
                            }
                            t.grasp_tries += 1;
                            t.queue.clear();
                            if t.grasp_tries < MAX_GRASP_TRIES {
                                if let Some(id) = t.last_pick {
                                    if let Some(i) = t.drops.iter().position(|d| *d == id) {
                                        t.drop_idx = i;
                                    }
                                }
                                self.log("quest", Level::Warn, format!("[ML] 잡기 실패 ({why}) — 다시 잡기 {}/{MAX_GRASP_TRIES}", t.grasp_tries + 1), None);
                            } else {
                                t.grasp_tries = 0;
                                t.wasted += 1;
                                self.log("quest", Level::Warn, format!("[ML] {MAX_GRASP_TRIES}번 잡지 못해 이 조각은 두고 감 ({why})"), None);
                            }
                            return Flow::Continue;
                        }
                        if t.phase == MlPhase::Collect && why.starts_with("push") {
                            t.queue.clear();
                            if let Some((id, n)) = t.push {
                                if n + 1 >= MAX_PUSH_TRIES {
                                    t.push = None;
                                    t.drop_idx += 1;
                                    t.wasted += 1;
                                    self.log("quest", Level::Warn, format!("[ML] {MAX_PUSH_TRIES}번 밀다 놓쳐 d{id}는 두고 감 ({why})"), None);
                                } else {
                                    t.push = Some((id, n + 1));
                                }
                            }
                            return Flow::Continue;
                        }
                        if t.phase == MlPhase::Look {
                            t.decider.move_failed();
                            t.queue.clear();
                            self.mark_executed(t, format!("실패({why})"), false);
                            return Flow::Continue;
                        }
                        return fail(&why, format!("동작이 실패했습니다: {why}"));
                    }
                    if t.phase == MlPhase::Mine && t.queue.is_empty() {
                        return self.ml_after_mine(t);
                    }
                }
            }
        }
        if self.world.fallen_until.is_some_and(|f| self.world.time_s < f) {
            return Flow::Continue;
        }
        if let Some(a) = t.queue.pop_front() {
            let label = a.label();
            return match self.ml_issue(t, a.clone()) {
                Ok(()) => {
                    if t.phase == MlPhase::Look {
                        self.mark_executed(t, label, true);
                    }
                    Flow::Continue
                }
                Err(e) if t.phase == MlPhase::Look => {
                    t.decider.move_failed();
                    t.queue.clear();
                    self.mark_executed(t, format!("{label} 거부({})", e.code()), false);
                    Flow::Continue
                }
                Err(ActionError::CapacityExceeded { carried_kg, item_kg, capacity_kg }) => {
                    fail("capacity_exceeded", format!("적재 한도를 넘습니다 ({carried_kg:.1} + {item_kg:.1} > {capacity_kg:.1} kg)"))
                }
                Err(ActionError::OutOfReach { .. }) if matches!(a, Action::Deliver { .. }) && !t.deliver_retry => {
                    t.deliver_retry = true;
                    let _ = self.scan_now("place");
                    match t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| (w.x, w.z)) {
                        Some((x, z)) => {
                            self.log("quest", Level::Warn, "[ML] 목적지에 손이 닿지 않음 — 보고 더 다가감", None);
                            t.queue.push_front(a);
                            t.queue.push_front(Action::MoveTo { x, z, reach_m: PLACE_REACH_M - 0.5 });
                            Flow::Continue
                        }
                        None => fail("unknown_target", "목적지를 찾을 수 없습니다"),
                    }
                }
                Err(ActionError::NoPath) if matches!(a, Action::MoveTo { reach_m, .. } if reach_m < PICKUP_REACH_M - 0.1) => {
                    if let Action::MoveTo { x, z, .. } = a {
                        t.queue.push_front(Action::MoveTo { x, z, reach_m: PICKUP_REACH_M - 0.1 });
                    }
                    Flow::Continue
                }
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        match t.phase {
            MlPhase::Look => self.ml_look(t),
            MlPhase::Mine => self.ml_after_mine(t),
            MlPhase::Collect => self.ml_collect(t),
            MlPhase::Carry | MlPhase::Goto => self.ml_carry(t),
            MlPhase::Deliver => self.ml_deliver(t),
            MlPhase::Unload => self.ml_unload(t),
        }
    }

    fn ml_look(&mut self, t: &mut MlTask) -> Flow {
        if self.world.time_s - t.look_since_s > self.cfg.locate_timeout_s {
            let name = t.target().label_ko.clone();
            return fail("locate_timeout", format!("[ML] {name}을(를) {:.0}초 안에 찾지 못했습니다", self.cfg.locate_timeout_s));
        }
        if let Some(flow) = self.ml_blind(t) {
            return flow;
        }
        t.decisions += 1;
        if t.decisions > self.test_cfg.max_ml_decisions {
            return fail("ml_budget", format!("[ML] 결정 {}회 안에 끝내지 못했습니다", self.test_cfg.max_ml_decisions));
        }
        let placed = !t.target().item.from_node();
        let mut look_cam = self.cam;
        let mut applied_tilt = 0.0f32;
        if placed {
            let eye = self.world.standing_eye_m();
            let ttm = self.gates.ttm();
            let tp = ttm.as_deref().map(TtmPredictor);
            let extra: Vec<&dyn Predictor> = tp.iter().map(|p| p as &dyn Predictor).collect();
            let odo_pose = crate::record::odometry(&self.world, &self.cam);
            if let Some(pred) = t.decider.predict(&odo_pose, &extra, 0.0).and_then(|r| r.estimate) {
                if let Some((_, tilt)) = gaze_to_see(&self.cam, eye, (odo_pose.x, odo_pose.z, odo_pose.yaw), pred.x, pred.z) {
                    applied_tilt = tilt;
                } else if pred.range_m <= 1.8 {
                    let dep = eye.atan2(pred.range_m.max(0.15)).to_degrees();
                    applied_tilt = (dep + self.cam.pitch_deg).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
                }
            } else if t.holds > 0 {
                applied_tilt = crate::driver::GROUND_SWEEP_TILT_DEG;
            }
            look_cam.pitch_deg -= applied_tilt;
        }
        let index = self.recording.as_ref().map_or(0, |r| r.keyframes.len()) as u32;
        let mut kf = match capture_keyframe(&self.world, &look_cam, self.test_cfg.record.quality, index) {
            Ok(k) => k,
            Err(e) => return fail("capture_failed", e.to_string()),
        };
        kf.step = self.quests.get(&t.quest).and_then(|q| q.steps.get(t.step_of(t.phase))).map(|s| s.label_ko.clone());
        let wanted = t.target().labels.clone();
        let out = match self.ml_analyse(&mut kf, &wanted, t.decider_track()) {
            Ok(o) => o,
            Err(e) => return fail("perception_failed", format!("ML 파이프라인 실패: {e}")),
        };
        if placed {
            self.mark_coverage_as(&mut t.coverage, applied_tilt, crate::driver::SMALL_ITEM_SEE_M);
        } else {
            self.mark_coverage(&mut t.coverage);
        }
        let label = out.frame.target.as_ref().map(|x| x.label.clone()).or_else(|| wanted.first().cloned()).unwrap_or_default();
        let ttm = self.gates.ttm();
        let tp = ttm.as_deref().map(TtmPredictor);
        let extra: Vec<&dyn Predictor> = tp.iter().map(|p| p as &dyn Predictor).collect();
        let holonomic = self.world.hand.base.holonomic;
        let mut rep = t.decider.observe(&kf.pose, out.sighting, holonomic, &extra, self.depth_scale(&label));
        let off_mask = kf.ml.as_ref().map_or(false, |m| m.flags.iter().any(|f| f == "peak_off_mask"));
        if matches!(rep.decision, Decision::Arrive { .. }) && off_mask {
            t.decider.reset();
            self.ml.reset();
            t.fixed = false;
            rep.decision = Decision::Hold { reason: "semantic_mismatch".into() };
            rep.note = "도달 직전 의미 피크가 추적 마스크 밖: 추적을 다시 시작".into();
            self.sds.record_agreement(&Scope::new(Track::Fusion, &label, ""), "commit_ok", false);
        } else if matches!(rep.decision, Decision::Arrive { .. }) {
            self.sds.record_agreement(&Scope::new(Track::Fusion, &label, ""), "commit_ok", true);
        }

        if let (Some(fix), Some(s)) = (&rep.fix, &out.sighting) {
            if let Some(d) = s.depth_range_m.filter(|_| noto_vision::scale_usable(fix, s)) {
                let scope = Scope::new(Track::Depth, &self.ml.depth_mode(), &label);
                self.sds.record_calibration(&scope, "scale", fix.range_m as f64, d as f64);
            }
        }
        let kind = match &rep.decision {
            Decision::Search { .. } => "search",
            Decision::LookDown { .. } => "look_down",
            Decision::Triangulate(_) => "triangulate",
            Decision::Approach { .. } => "approach",
            Decision::Arrive { .. } => "arrive",
            Decision::Hold { .. } => "hold",
            Decision::Explore { .. } => "explore",
        };
        let base = format!("{:?}", self.world.hand.base.kind).to_lowercase();
        let dscope = Scope::new(Track::Decide, kind, &base);
        if let Some(r) = rep.residual_deg {
            if let Some((m, sd, _)) = self.sds.adaptive_baseline(&dscope, "residual_deg") {
                kf.sds.insert("decide".into(), ((r as f64 - m) / sd.max(1e-6)) as f32);
            }
            self.sds.record_baseline(&dscope, "residual_deg", r as f64);
        }
        if let Some(f) = &rep.fix {
            self.sds.record_baseline(&dscope, "tri_sigma_frac", (f.sigma_m / f.range_m.max(0.1)) as f64);
        }
        if let Some(ml) = kf.ml.as_mut() {
            if let Some(tv) = ml.target.as_mut() {
                tv.tri_range_m = rep.fix.as_ref().map(|f| f.range_m);
                tv.tri_parallax_deg = rep.fix.as_ref().map(|f| f.parallax_deg);
                tv.tri_sigma_m = rep.fix.as_ref().map(|f| f.sigma_m);
                if rep.depth_scale.is_some() {
                    tv.depth_scale = rep.depth_scale;
                }
                tv.range_used_m = rep.estimate.as_ref().map(|e| e.range_m);
            }
        }
        kf.decision = Some(DecisionView {
            decision: rep.decision.label(),
            kind: kind.into(),
            note: rep.note.clone(),
            residual_deg: rep.residual_deg,
            forecast_method: rep.forecast_method.clone(),
            band: rep.band,
            estimate: rep.estimate.as_ref().map(|e| (e.x, e.z, e.range_m, e.sigma_m)),
            fix: rep.fix.as_ref().map(|f| (f.x, f.z, f.range_m, f.parallax_deg, f.sigma_m)),
            executed: String::new(),
            ok: true,
        });
        let target_label = label.clone();
        self.record_keyframe_row("ml", &kf, &target_label);
        let yaw_deg = self.world.agent.yaw.to_degrees();
        let mut arrive_at = None;
        match &rep.decision {
            Decision::Search { turn_deg } => t.queue.push_back(Action::Turn { yaw_deg: yaw_deg + turn_deg }),
            Decision::LookDown { pitch_deg, turn_deg } => {
                if let Err(e) = self.scan_gaze("ml_look_down", 0.0, *pitch_deg) {
                    return fail("perception_failed", format!("인식기가 실패했습니다: {e}"));
                }
                if *turn_deg != 0.0 {
                    t.queue.push_back(Action::Turn { yaw_deg: yaw_deg + turn_deg });
                }
            }
            Decision::Triangulate(s) => {
                let (step, how) = self.feasible_step(s, (kf.pose.x, kf.pose.z));
                if how == "mirrored" || how == "mirrored_short" {
                    t.decider.set_side(-t.decider.side());
                }
                if !how.is_empty() {
                    if let Some(d) = kf.decision.as_mut() {
                        d.note = format!("{} · 기선 {}", d.note, match how {
                            "mirrored" => "반대쪽으로 (지형이 막음)",
                            "short" => "짧게 (지형이 막음)",
                            "mirrored_short" => "반대쪽으로 짧게 (지형이 막음)",
                            _ => "확인 못 함",
                        });
                    }
                }
                t.queue.push_back(Action::MoveTo { x: step.to.0, z: step.to.1, reach_m: 0.0 });
                t.queue.push_back(Action::Face { x: step.face.0, z: step.face.1 });
            }
            Decision::Explore { .. } => {
                let vp = if !t.target().item.from_node() {
                    self.viewpoint_with(&t.coverage, &t.visited, crate::driver::GROUND_SWEEP_TILT_DEG, crate::driver::SMALL_ITEM_SEE_M)
                } else {
                    self.search_viewpoint_for(&t.coverage, &t.visited)
                };
                match vp {
                Some(v) => {
                    t.visited.push((self.world.agent.x, self.world.agent.z));
                    t.visited.push((v.x, v.z));
                    if let Some(d) = kf.decision.as_mut() {
                        d.note = format!("{} · 시점 ({:.1}, {:.1}) 안 본 곳 {}칸 · 약 {:.0}초", d.note, v.x, v.z, v.gain, v.time_s);
                    }
                    t.queue.push_back(Action::MoveTo { x: v.x, z: v.z, reach_m: 0.0 });
                    t.queue.push_back(Action::Turn { yaw_deg: v.face_yaw.to_degrees() });
                }
                None => {
                    if let Some(d) = kf.decision.as_mut() {
                        d.note = format!("{} · 갈 수 있는 곳에서 볼 땅을 다 봄", d.note);
                    }
                    t.queue.push_back(Action::Turn { yaw_deg: yaw_deg + 50.0 });
                }
            }
        },
            Decision::Approach { to, face, .. } => {
                t.queue.push_back(Action::MoveTo { x: to.0, z: to.1, reach_m: 0.3 });
                t.queue.push_back(Action::Face { x: face.0, z: face.1 });
            }
            Decision::Arrive { at, .. } => arrive_at = Some(*at),
            Decision::Hold { .. } => {
                t.holds += 1;
                if t.holds > 3 {
                    t.decider.reset();
                    t.holds = 0;
                }
            }
        }
        let newly_fixed = !t.fixed && t.decider.has_fix();
        if let Some(rec) = self.recording.as_mut() {
            if rec.keyframes.len() < self.test_cfg.record.max_keyframes {
                kf.truth.shed_pixels();
                rec.keyframes.push(kf);
                t.pending_kf = Some(rec.keyframes.len() - 1);
            } else {
                rec.dropped_keyframes += 1;
                t.pending_kf = None;
            }
        }
        if newly_fixed {
            self.ml_goto(t, MlPhase::Look, "");
            let (a, b) = (4 * t.cur, 4 * t.cur + 1);
            if let Some(q) = self.quests.get_mut(&t.quest) {
                let _ = q.complete_step(a);
            }
            if let Some(x) = t.attempts.get_mut(b) {
                *x += 1;
            }
            t.fixed = true;
        }
        if let Some(at) = arrive_at {
            return self.ml_arrive(t, at);
        }
        Flow::Continue
    }

    fn ml_blind(&mut self, t: &mut MlTask) -> Option<Flow> {
        if !self.cfg.vision.intermittent || t.blind_run >= 1 || !t.fixed {
            t.blind_run = 0;
            return None;
        }
        let rate = self.drift_rate();
        let since = t.decider.fix_sigma_m()?;
        let hop = t.decider.cfg.hop_max_m;
        let drift = Memory::seen(0.0, 0.0, 0.0, 0.0, 0).sigma_m(hop, rate);
        if self.cfg.vision.sigmas * since.hypot(drift) > t.decider.cfg.reach_m {
            t.blind_run = 0;
            return None;
        }
        let pose = crate::record::odometry(&self.world, &self.cam);
        let ttm = self.gates.ttm();
        let tp = ttm.as_deref().map(TtmPredictor);
        let extra: Vec<&dyn Predictor> = tp.iter().map(|p| p as &dyn Predictor).collect();
        let rep = t.decider.predict(&pose, &extra, drift)?;
        t.blind_run += 1;
        t.blind += 1;
        t.decisions += 1;
        let what = rep.estimate.as_ref().map(|e| format!("{:.1},{:.1}", e.x, e.z)).unwrap_or_default();
        self.note_look("blind", format!("ML {what}"), rep.estimate.as_ref().map(|e| e.sigma_m), None, None);
        self.log("quest", Level::Info, format!("[ML] {}", rep.note), None);
        match rep.decision {
            Decision::Approach { to, face, .. } => {
                t.queue.push_back(Action::MoveTo { x: to.0, z: to.1, reach_m: 0.3 });
                t.queue.push_back(Action::Face { x: face.0, z: face.1 });
                Some(Flow::Continue)
            }
            Decision::Arrive { at, .. } => Some(self.ml_arrive(t, at)),
            _ => None,
        }
    }

    fn ml_arrive(&mut self, t: &mut MlTask, at: (f32, f32)) -> Flow {
        if !t.target().item.from_node() {
            let item = t.target().item;
            return match self.touch_piece(at, item) {
                Some(piece) => {
                    t.arrive_tries = 0;
                    t.touch_ok += 1;
                    t.mem.insert(piece, Memory::seen(at.0, at.1, t.decider.fix_sigma_m().unwrap_or(0.1), self.world.odom.travelled_m, self.world.tick));
                    t.decider.reset();
                    self.ml.reset();
                    t.fixed = false;
                    t.drops = vec![piece];
                    t.drop_idx = 0;
                    self.ml_goto(t, MlPhase::Collect, "");
                    self.mark_executed(t, format!("접촉 d{piece} → 줍기"), true);
                    Flow::Continue
                }
                None if t.arrive_tries < 2 => {
                    t.arrive_tries += 1;
                    t.queue.push_back(Action::MoveTo { x: at.0, z: at.1, reach_m: 0.4 });
                    Flow::Continue
                }
                None => {
                    t.arrive_tries = 0;
                    t.decider.reset();
                    self.ml.reset();
                    t.fixed = false;
                    t.wasted += 1;
                    self.log("vision", Level::Warn, "[ML] 추정 위치에 손에 닿는 물건이 없음: 다시 탐색", None);
                    Flow::Continue
                }
            };
        }
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        let at_w = self.world.true_of(at.0, at.1);
        let near = self
            .world
            .nodes
            .iter()
            .filter(|n| (n.x - ax).hypot(n.z - az) <= MINE_REACH_M)
            .map(|n| (n.id, n.x, n.z, (n.x - at_w.0).hypot(n.z - at_w.1)))
            .filter(|n| n.3 <= TOUCH_FROM_ESTIMATE_M)
            .min_by(|a, b| a.3.total_cmp(&b.3));
        match near {
            Some((id, x, z, _)) => {
                t.arrive_tries = 0;
                self.ml_goto(t, MlPhase::Mine, "");
                let (fx, fz) = self.world.belief_of(x, z);
                t.queue.push_back(Action::Face { x: fx, z: fz });
                t.queue.push_back(Action::Mine { node: id });
                self.mark_executed(t, format!("접촉 n{id} → 채취"), true);
                Flow::Continue
            }
            None if t.arrive_tries < 2 => {
                t.arrive_tries += 1;
                t.queue.push_back(Action::MoveTo { x: at.0, z: at.1, reach_m: 0.6 });
                Flow::Continue
            }
            None => {
                t.arrive_tries = 0;
                t.decider.reset();
                self.ml.reset();
                t.fixed = false;
                t.wasted += 1;
                self.log("vision", Level::Warn, "[ML] 추정 위치에 닿을 물체가 없음: 다시 탐색", None);
                Flow::Continue
            }
        }
    }

    fn ml_after_mine(&mut self, t: &mut MlTask) -> Flow {
        let Some((_, kind, drops)) = t.broken.take() else {
            if t.running || !t.queue.is_empty() {
                return Flow::Continue;
            }
            return fail("mine_failed", "[ML] 채취하지 못했습니다");
        };
        let want = t.target().item;
        let hit = kind.spec().drop == want;
        let lmode = self.ml.locate_mode();
        for l in t.target().labels.clone() {
            self.sds.record_agreement(&Scope::new(Track::Locate, &l, &lmode), "touch", hit);
        }
        t.decider.reset();
        self.ml.reset();
        t.fixed = false;
        if hit {
            t.touch_ok += 1;
            t.drops = drops.iter().filter_map(|d| d.get(1..)?.parse().ok()).filter(|id| self.world.drop_item(*id).map_or(false, |d| d.kind == want)).collect();
            t.drop_idx = 0;
            let sigma = self.world.aim_sigma_m();
            for &id in &t.drops {
                if let Some(d) = self.world.drop_item(id) {
                    let (x, z) = self.world.belief_of(d.x, d.z);
                    t.mem.insert(id, Memory::seen(x, z, sigma, self.world.odom.travelled_m, self.world.tick));
                }
            }
            self.ml_goto(t, MlPhase::Collect, "");
        } else {
            t.touch_miss += 1;
            t.wasted += 1;
            self.ml_goto(t, MlPhase::Look, &format!("접촉 불일치({})", kind.id()));
            t.look_since_s = self.world.time_s;
        }
        Flow::Continue
    }

    fn feasible_step(&self, s: &TriStep, from: (f32, f32)) -> (TriStep, &'static str) {
        let ok = |st: &TriStep| -> bool {
            let reach = self.world.terrain.cell_of(st.to.0, st.to.1).is_some_and(|c| self.world.plan_route(world_sim::nav::Goal::Cell(c)).is_some());
            reach && line_of_sight(&self.world.terrain, st.to, self.world.standing_eye_m(), st.face, TARGET_H_M)
        };
        if ok(s) {
            return (s.clone(), "");
        }
        let (ux, uz) = {
            let (dx, dz) = (s.face.0 - from.0, s.face.1 - from.1);
            let n = dx.hypot(dz).max(1e-3);
            (dx / n, dz / n)
        };
        let mirror = |st: &TriStep| -> TriStep {
            let (vx, vz) = (st.to.0 - from.0, st.to.1 - from.1);
            let along = vx * ux + vz * uz;
            TriStep { to: (from.0 + 2.0 * along * ux - vx, from.1 + 2.0 * along * uz - vz), ..st.clone() }
        };
        let shrink = |st: &TriStep, k: f32| -> TriStep {
            TriStep { to: (from.0 + k * (st.to.0 - from.0), from.1 + k * (st.to.1 - from.1)), step_m: st.step_m * k, baseline_perp_m: st.baseline_perp_m * k, ..st.clone() }
        };
        let m = mirror(s);
        if ok(&m) {
            return (m, "mirrored");
        }
        for k in [0.6f32, 0.35] {
            let a = shrink(s, k);
            if ok(&a) {
                return (a, "short");
            }
            let b = shrink(&m, k);
            if ok(&b) {
                return (b, "mirrored_short");
            }
        }
        (s.clone(), "unchecked")
    }

    fn ml_quest_cargo_kg(&self, t: &MlTask) -> f32 {
        t.targets
            .iter()
            .map(|x| {
                let given: f32 = t.delivered.iter().filter(|(k, _)| *k == x.resource).map(|(_, a)| *a).sum();
                self.world.carried_amount(x.item).min((x.picked - given).max(0.0)) * x.item.spec().mass_kg
            })
            .sum()
    }

    fn ml_collect(&mut self, t: &mut MlTask) -> Flow {
        if let Some(p) = t.split_piece.take() {
            t.drops.insert(t.drop_idx, p);
        }
        if t.push.is_some() {
            return self.ml_push(t);
        }
        let (picked, wanted) = (t.target().picked, t.target().wanted);
        if picked + AMOUNT_EPS >= wanted as f32 || t.drop_idx >= t.drops.len() {
            if picked + AMOUNT_EPS >= wanted as f32 {
                if t.cur + 1 < t.targets.len() {
                    let step = 4 * t.cur + 3;
                    if let Some(q) = self.quests.get_mut(&t.quest) {
                        let _ = q.complete_step(step);
                    }
                    t.cur += 1;
                    t.phase = MlPhase::Look;
                    t.look_since_s = self.world.time_s;
                    if let Some(x) = t.attempts.get_mut(4 * t.cur) {
                        *x += 1;
                    }
                    return Flow::Continue;
                }
                if t.kind == "collect" {
                    return Flow::Done;
                }
                self.ml_goto(t, MlPhase::Carry, "");
                return Flow::Continue;
            }
            self.ml_goto(t, MlPhase::Look, &format!("{}/{wanted}개, 더 찾음", amount_text(picked)));
            t.look_since_s = self.world.time_s;
            return Flow::Continue;
        }
        let id = t.drops[t.drop_idx];
        let Some(d) = self.world.drop_item(id).cloned() else {
            t.drop_idx += 1;
            return Flow::Continue;
        };
        let m = *t.mem.entry(id).or_insert_with(|| {
            let (x, z) = self.world.belief_of(d.x, d.z);
            Memory::seen(x, z, self.world.aim_sigma_m(), self.world.odom.travelled_m, self.world.tick)
        });
        let dist = self.dist_me(m.x, m.z);
        // Wheels and tracks lift with forks and stop with the tine tips short of the item.
        let forks = noto_core::act::fork::applies(&self.world.body);
        let size = d.kind.spec().piece_size_m(d.amount);
        let stop_m = if forks { noto_core::act::fork::standoff_x(&self.world.body, size[0].max(size[2])) } else { 0.4 };
        if dist > if forks { stop_m + 0.15 } else { PICKUP_REACH_M - 0.1 } {
            t.queue.push_back(Action::MoveTo { x: m.x, z: m.z, reach_m: stop_m });
            return Flow::Continue;
        }
        let dest = t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| (w.id, w.x, w.z));
        let can_take = if forks { self.world.fork_check(d.kind, d.amount).map(|_| ()) } else { self.world.reach_ground(d.kind, d.amount) };
        if let Err(why) = can_take {
            if dest.is_none() || self.world.body.legged() {
                return fail("reach_ground", format!("[ML] {}을(를) 집을 수 없습니다: {why}", d.kind.spec().label_ko));
            }
            t.push = Some((id, 0));
            return self.ml_push(t);
        }
        let ctx = CargoCtx { quest_cargo_kg: self.ml_quest_cargo_kg(t), destination: dest.map(|w| (w.1, w.2)) };
        let dec = cargo::decide(&self.world, &d, &ctx);
        if dec.action != CargoAction::Take {
            self.log("quest", Level::Info, format!("[ML] 적재 판단 · {} ({})", dec.summary_ko, dec.action.label_ko()), Some(json!({"drop": id, "action": dec.action})));
        }
        match dec.action {
            CargoAction::Take => {
                t.drop_idx += 1;
                t.last_pick = Some(id);
                let (rate, odo) = (self.drift_rate(), self.world.odom.travelled_m);
                let size = d.kind.spec().piece_size_m(d.amount);
                let mut m = m;
                if !memory_ok(m.sigma_m(odo, rate), 0.5 * size[0].min(size[2]), &self.cfg.vision) {
                    if let Some((pan, tilt)) = gaze_to_see(&self.cam, self.world.standing_eye_m(), self.me(), m.x, m.z) {
                        if self.scan_gaze("glance", pan, tilt).is_ok() {
                            if let Some((fused, innov)) = self.refresh_memory(&m, d.kind, odo) {
                                let before = m.sigma_m(odo, rate);
                                m = fused;
                                t.mem.insert(id, m);
                                self.note_look("glance", format!("ML d{id}"), Some(before), Some(m.sigma_m(odo, rate)), Some(innov));
                            }
                        }
                    }
                }
                let sigma = m.sigma_m(odo, rate);
                if gaze_to_see(&self.cam, self.world.standing_eye_m(), self.me(), m.x, m.z).is_none() {
                    self.note_look("blind", format!("ML d{id}"), Some(sigma), None, None);
                }
                t.queue.push_back(Action::PickUp { drop: id, aim: Some((m.x, m.z)), side: false, memory_sigma_m: sigma });
            }
            CargoAction::Split { portion } => t.queue.push_back(Action::Split { drop: id, portion }),
            CargoAction::Shuttle => {
                let Some((wid, wx, wz)) = dest else { return fail("capacity_exceeded", dec.summary_ko) };
                if t.trips >= self.cfg.max_trips {
                    return fail("too_many_trips", format!("[ML] {}회 왕복해도 다 나르지 못했습니다", t.trips));
                }
                t.trips += 1;
                t.queue.push_back(Action::MoveTo { x: wx, z: wz, reach_m: DELIVER_REACH_M - 0.5 });
                for x in &t.targets {
                    if self.world.carried_amount(x.item) > AMOUNT_EPS {
                        t.queue.push_back(Action::Deliver { waypoint: wid, item: Some(x.item) });
                    }
                }
                let (sx, sz, _) = self.me();
                t.queue.push_back(Action::MoveTo { x: sx, z: sz, reach_m: 0.4 });
            }
            CargoAction::Refuse => return fail(dec.code.unwrap_or("lift_overload"), format!("[ML] {}", dec.summary_ko)),
        }
        Flow::Continue
    }

    fn ml_push(&mut self, t: &mut MlTask) -> Flow {
        let Some((id, _)) = t.push else { return Flow::Continue };
        let Some((wid, wx, wz)) = t.destination.as_deref().and_then(|k| self.world.waypoint_by_key(k)).map(|w| (w.id, w.x, w.z)) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let Some(d) = self.world.drop_item(id).cloned() else {
            t.push = None;
            t.drop_idx += 1;
            return Flow::Continue;
        };
        let (px, pz) = self.world.belief_of(d.x, d.z);
        if (px - wx).hypot(pz - wz) <= DELIVER_REACH_M {
            let mut ev = Vec::new();
            if let Some(a) = self.world.stock_pushed(id, wid, &mut ev) {
                self.log_world_events(&ev);
                if let Some(x) = t.targets.iter_mut().find(|x| x.item == d.kind) {
                    x.picked += a;
                }
                let k = d.kind.id().to_string();
                match t.delivered.iter_mut().find(|(n, _)| *n == k) {
                    Some(e) => e.1 += a,
                    None => t.delivered.push((k, a)),
                }
                t.push = None;
                t.drop_idx += 1;
                return Flow::Continue;
            }
        }
        let (tx, tz) = self.world.true_of(wx, wz);
        let (_, (lx, lz, yaw)) = match self.world.push_route(id, tx, tz) {
            Ok(r) => r,
            Err(e) => return fail(e.code(), format!("[ML] 밀 수 없습니다: {}", action_error_ko(&e))),
        };
        let (lx, lz) = self.world.belief_of(lx, lz);
        let yaw = yaw + (self.me().2 - self.world.agent.yaw);
        if self.dist_me(lx, lz) > world_sim::world::PUSH_START_SLACK_M {
            t.queue.push_back(Action::MoveTo { x: lx, z: lz, reach_m: 0.36 });
        }
        t.queue.push_back(Action::Turn { yaw_deg: yaw.to_degrees() });
        t.queue.push_back(Action::Push { drop: id, x: wx, z: wz });
        Flow::Continue
    }

    fn ml_carry(&mut self, t: &mut MlTask) -> Flow {
        let Some(dest) = t.destination.clone() else { return Flow::Done };
        let Some((x, z)) = self.world.waypoint_by_key(&dest).map(|w| (w.x, w.z)) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let reach = DELIVER_REACH_M - 0.5;
        if self.dist_me(x, z) <= reach + 0.2 {
            if t.phase == MlPhase::Goto && t.kind == "drop" {
                self.ml_goto(t, MlPhase::Unload, "");
                return Flow::Continue;
            }
            if t.phase == MlPhase::Goto {
                return Flow::Done;
            }
            self.ml_goto(t, MlPhase::Deliver, "");
            return Flow::Continue;
        }
        t.queue.push_back(Action::MoveTo { x, z, reach_m: reach });
        Flow::Continue
    }

    fn ml_deliver(&mut self, t: &mut MlTask) -> Flow {
        if t.deliver_idx >= t.targets.len() {
            return Flow::Done;
        }
        let Some(dest) = t.destination.clone() else { return Flow::Done };
        let Some(wid) = self.world.waypoint_by_key(&dest).map(|w| w.id) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let item = t.targets[t.deliver_idx].item;
        t.deliver_idx += 1;
        if self.world.carried_amount(item) <= AMOUNT_EPS {
            return Flow::Continue;
        }
        t.queue.push_back(Action::Deliver { waypoint: wid, item: Some(item) });
        Flow::Continue
    }

    fn ml_unload(&mut self, t: &mut MlTask) -> Flow {
        if t.unload_started {
            return if t.unloaded.is_empty() { fail("nothing_to_drop", "[ML] 적재함에 내려놓을 물건이 없습니다") } else { Flow::Done };
        }
        t.unload_started = true;
        if t.drop_items.is_empty() {
            if self.world.agent.carried.is_empty() {
                return fail("nothing_to_drop", "[ML] 적재함에 내려놓을 물건이 없습니다");
            }
            t.queue.push_back(Action::Drop { item: None, amount: None });
            return Flow::Continue;
        }
        for it in &t.drop_items {
            let Some(item) = ItemKind::from_id(&it.resource) else { continue };
            let have = self.world.carried_amount(item);
            if have > AMOUNT_EPS {
                t.queue.push_back(Action::Drop { item: Some(item), amount: it.quantity.map(|q| (q as f32).min(have)) });
            }
        }
        if t.queue.is_empty() {
            return fail("nothing_to_drop", "[ML] 적재함에 내려놓으라는 물건이 없습니다");
        }
        Flow::Continue
    }

    pub(crate) fn ml_end(&mut self, t: MlTask, failure: Option<(String, String)>) {
        if self.world.busy() {
            let ev = self.world.cancel();
            self.log_world_events(&ev);
        }
        let result = QuestResult {
            ticks: self.world.tick - t.started_tick,
            time_s: self.world.time_s - t.started_time_s,
            energy_j: self.world.agent.energy_used_j - t.started_energy_j,
            precision: (t.touch_ok + t.touch_miss > 0).then(|| t.touch_ok as f64 / (t.touch_ok + t.touch_miss) as f64),
            summary: match &failure {
                None => format!("[ML] 완료 — 결정 {}회 · {:.1}초", t.decisions, self.world.time_s - t.started_time_s),
                Some((_, m)) => format!("[ML] 실패: {m}"),
            },
            items: {
                let whole = |list: &[(String, f32)]| list.iter().map(|(k, a)| (k.clone(), (a + AMOUNT_EPS).floor() as u32)).collect::<Vec<_>>();
                match t.kind {
                    "fetch" => whole(&t.delivered),
                    "drop" => whole(&t.unloaded),
                    _ => t.targets.iter().map(|x| (x.resource.clone(), (x.picked + AMOUNT_EPS).floor() as u32)).collect(),
                }
            },
            wasted_actions: t.wasted,
            trips: t.trips + u32::from(t.kind == "fetch" && !t.delivered.is_empty()),
        };
        let tick = self.world.tick;
        if let Some(q) = self.quests.get_mut(&t.quest) {
            match &failure {
                None => {
                    for i in 0..q.steps.len() {
                        if q.steps[i].state != State::Done {
                            let _ = q.complete_step(i);
                        }
                    }
                    let _ = q.finish(result.clone());
                }
                Some((code, msg)) => {
                    let _ = q.fail(code, msg, Some(result.clone()));
                }
            }
        }
        match &failure {
            None => self.log("quest", Level::Info, format!("[ML] 퀘스트 {} 완료: {}", t.quest, result.summary), Some(json!({"quest": t.quest, "tick": tick, "decisions": t.decisions}))),
            Some((code, msg)) => self.log("quest", Level::Warn, format!("[ML] 퀘스트 {} 실패 [{code}]: {msg}", t.quest), Some(json!({"quest": t.quest, "code": code}))),
        }
        self.record_quest_run(&t.quest.clone(), &t.attempts);
        self.flush_sds();
    }
}

impl MlTask {
    fn decider_track(&self) -> Option<u32> {
        self.decider.track_id()
    }
}
