//! The ML pass: the same quest, but finding and approaching the target is
//! the ML logic's job. One decision per keyframe:
//!
//! capture → depth / semantic location / mask tracking → triangulation and
//! time-series forecast (`hand_vision::Decider`) → the game carries out the
//! move it asked for → capture again.
//!
//! The decision's first move toward a new target is always a triangulation
//! step: a monocular range is not trusted. Once the target is reached, the
//! rest (touch and mine, pick up, carry, deliver) uses the game's actions,
//! which still have to pass the physics (lift, balance, capacity). Waypoints
//! are on the map, as in the game pass; the drops a broken node released are
//! told by the event, as in the game pass.

use std::collections::VecDeque;

use hand_quest::{QuestResult, State};
use hand_sds::{Level, Scope, Track};
use hand_search::Plan;
use hand_ts::{Predictor, TtmPredictor};
use hand_vision::{DecideCfg, Decider, Decision};
use serde_json::json;
use world_sim::world::{DELIVER_REACH_M, MINE_REACH_M, PICKUP_REACH_M};
use world_sim::{Action, ActionError, ItemKind, NodeKind, TimedEvent, WorldEvent};

use crate::driver::action_error_ko;

/// A node counts as the one the ML logic meant when it is this close to the
/// estimated target point (the estimate is the visible surface; the node's
/// centre is behind it).
const TOUCH_FROM_ESTIMATE_M: f32 = 0.8;
/// The decision arrives when the visible surface is this close; the node's
/// centre is then within the hand's reach.
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
}

pub(crate) struct MlTarget {
    pub resource: String,
    pub item: ItemKind,
    pub wanted: u32,
    pub picked: u32,
    pub label_ko: String,
    /// Vision labels of the nodes that yield it.
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
    delivered: Vec<(String, u32)>,
    /// Keyframe whose decision the queued actions carry out.
    pending_kf: Option<usize>,
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
        }
    }

    fn target(&self) -> &MlTarget {
        &self.targets[self.cur.min(self.targets.len().saturating_sub(1))]
    }
}

impl Runtime {
    fn ml_task_from(&self, quest: &str, plan: &Plan) -> Result<MlTask, (String, String)> {
        let mut targets = Vec::new();
        let (kind, destination, list) = match plan {
            Plan::Fetch { targets: ts, destination } => ("fetch", Some(destination.place.clone()), ts.clone()),
            Plan::Collect { targets: ts } => ("collect", None, ts.clone()),
            Plan::Goto { place } => ("goto", Some(place.place.clone()), vec![]),
            Plan::Stop => return Err(("stop".into(), "중지 명령은 퀘스트가 아닙니다".into())),
        };
        for t in &list {
            let item = ItemKind::from_id(&t.resource).ok_or_else(|| ("unknown_target".to_string(), format!("'{}'은(는) 이 월드에 없는 자원입니다", t.resource)))?;
            targets.push(MlTarget { resource: t.resource.clone(), item, wanted: t.quantity.max(1), picked: 0, label_ko: item.spec().label_ko.into(), labels: self.ml.source_labels_for(&t.resource) });
        }
        let n_steps = self.quests.get(quest).map_or(0, |q| q.steps.len());
        Ok(MlTask {
            quest: quest.into(),
            kind,
            targets,
            destination,
            cur: 0,
            phase: if kind == "goto" { MlPhase::Goto } else { MlPhase::Look },
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
        })
    }

    /// The ML pass's counterpart of the quest driver's tick.
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
        let Some(mut t) = self.ml_task.take() else { return };
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
        match self.world.command(a.clone()) {
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
        if t.running {
            match t.done.take() {
                None => return Flow::Continue,
                Some((ok, reason)) => {
                    t.running = false;
                    if !ok {
                        let why = reason.unwrap_or_else(|| "unknown".into());
                        if why == "cancelled" {
                            return fail("cancelled", "중지되었습니다");
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
                Err(e) => fail(e.code(), action_error_ko(&e)),
            };
        }
        match t.phase {
            MlPhase::Look => self.ml_look(t),
            MlPhase::Mine => self.ml_after_mine(t),
            MlPhase::Collect => self.ml_collect(t),
            MlPhase::Carry | MlPhase::Goto => self.ml_carry(t),
            MlPhase::Deliver => self.ml_deliver(t),
        }
    }

    fn ml_look(&mut self, t: &mut MlTask) -> Flow {
        if self.world.time_s - t.look_since_s > self.cfg.locate_timeout_s {
            let name = t.target().label_ko.clone();
            return fail("locate_timeout", format!("[ML] {name}을(를) {:.0}초 안에 찾지 못했습니다", self.cfg.locate_timeout_s));
        }
        t.decisions += 1;
        if t.decisions > self.test_cfg.max_ml_decisions {
            return fail("ml_budget", format!("[ML] 결정 {}회 안에 끝내지 못했습니다", self.test_cfg.max_ml_decisions));
        }
        let index = self.recording.as_ref().map_or(0, |r| r.keyframes.len()) as u32;
        let mut kf = match capture_keyframe(&self.world, &self.cam, self.test_cfg.record.quality, index) {
            Ok(k) => k,
            Err(e) => return fail("capture_failed", e.to_string()),
        };
        kf.step = self.quests.get(&t.quest).and_then(|q| q.steps.get(t.step_of(t.phase))).map(|s| s.label_ko.clone());
        let wanted = t.target().labels.clone();
        let out = match self.ml_analyse(&mut kf, &wanted, t.decider_track()) {
            Ok(o) => o,
            Err(e) => return fail("perception_failed", format!("ML 파이프라인 실패: {e}")),
        };
        let label = out.frame.target.as_ref().map(|x| x.label.clone()).or_else(|| wanted.first().cloned()).unwrap_or_default();
        let ttm = self.gates.ttm();
        let tp = ttm.as_deref().map(TtmPredictor);
        let extra: Vec<&dyn Predictor> = tp.iter().map(|p| p as &dyn Predictor).collect();
        let holonomic = self.world.hand.base.holonomic;
        let mut rep = t.decider.observe(&kf.pose, out.sighting, holonomic, &extra, self.depth_scale(&label));
        // Mixed features gate the commitment: before touching, the semantic
        // peak must still lie on the tracked mask. A mask that drifted onto a
        // neighbouring object (the region tracker's typical failure) would
        // otherwise send the hand to the wrong thing.
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

        // Triangulation teaches the depth model its scale, without ground truth.
        if let (Some(fix), Some(tgt)) = (&rep.fix, &out.frame.target) {
            if let Some(d) = tgt.depth_range_m.filter(|d| hand_vision::scale_usable(fix, *d)) {
                let scope = Scope::new(Track::Depth, &self.ml.depth_mode(), &label);
                self.sds.record_calibration(&scope, "scale", fix.range_m as f64, d as f64);
            }
        }
        let kind = match &rep.decision {
            Decision::Search { .. } => "search",
            Decision::Triangulate(_) => "triangulate",
            Decision::Approach { .. } => "approach",
            Decision::Arrive { .. } => "arrive",
            Decision::Hold { .. } => "hold",
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
            Decision::Triangulate(s) => {
                t.queue.push_back(Action::MoveTo { x: s.to.0, z: s.to.1, reach_m: 0.0 });
                t.queue.push_back(Action::Face { x: s.face.0, z: s.face.1 });
            }
            Decision::Approach { to, face, .. } => {
                t.queue.push_back(Action::MoveTo { x: to.0, z: to.1, reach_m: 0.3 });
                t.queue.push_back(Action::Face { x: face.0, z: face.1 });
            }
            Decision::Arrive { at } => arrive_at = Some(*at),
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

    fn ml_arrive(&mut self, t: &mut MlTask, at: (f32, f32)) -> Flow {
        // Touch: of the nodes within reach of the hand, the one nearest to
        // where the ML logic placed the target (the hand reaches toward what
        // was seen, not toward whatever happens to be closest).
        let (ax, az) = (self.world.agent.x, self.world.agent.z);
        let near = self
            .world
            .nodes
            .iter()
            .filter(|n| (n.x - ax).hypot(n.z - az) <= MINE_REACH_M)
            .map(|n| (n.id, n.x, n.z, (n.x - at.0).hypot(n.z - at.1)))
            .filter(|n| n.3 <= TOUCH_FROM_ESTIMATE_M)
            .min_by(|a, b| a.3.total_cmp(&b.3));
        match near {
            Some((id, x, z, _)) => {
                t.arrive_tries = 0;
                self.ml_goto(t, MlPhase::Mine, "");
                t.queue.push_back(Action::Face { x, z });
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
            self.ml_goto(t, MlPhase::Collect, "");
        } else {
            t.touch_miss += 1;
            t.wasted += 1;
            self.ml_goto(t, MlPhase::Look, &format!("접촉 불일치({})", kind.id()));
            t.look_since_s = self.world.time_s;
        }
        Flow::Continue
    }

    fn ml_collect(&mut self, t: &mut MlTask) -> Flow {
        let (picked, wanted) = (t.target().picked, t.target().wanted);
        if picked >= wanted || t.drop_idx >= t.drops.len() {
            if picked >= wanted {
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
            self.ml_goto(t, MlPhase::Look, &format!("{picked}/{wanted}개, 더 찾음"));
            t.look_since_s = self.world.time_s;
            return Flow::Continue;
        }
        let id = t.drops[t.drop_idx];
        let Some((dx, dz)) = self.world.drop_item(id).map(|d| (d.x, d.z)) else {
            t.drop_idx += 1;
            return Flow::Continue;
        };
        let dist = (dx - self.world.agent.x).hypot(dz - self.world.agent.z);
        if dist > PICKUP_REACH_M - 0.1 {
            t.queue.push_back(Action::MoveTo { x: dx, z: dz, reach_m: 0.4 });
            return Flow::Continue;
        }
        t.drop_idx += 1;
        t.queue.push_back(Action::PickUp { drop: id });
        Flow::Continue
    }

    fn ml_carry(&mut self, t: &mut MlTask) -> Flow {
        let Some(dest) = t.destination.clone() else { return Flow::Done };
        let Some((x, z)) = self.world.waypoint_by_key(&dest).map(|w| (w.x, w.z)) else { return fail("unknown_target", "목적지를 찾을 수 없습니다") };
        let reach = DELIVER_REACH_M - 0.5;
        if (x - self.world.agent.x).hypot(z - self.world.agent.z) <= reach + 0.2 {
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
        if self.world.carried_count(item) == 0 {
            return Flow::Continue;
        }
        t.queue.push_back(Action::Deliver { waypoint: wid, item: Some(item) });
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
            items: if t.kind == "fetch" { t.delivered.clone() } else { t.targets.iter().map(|x| (x.resource.clone(), x.picked)).collect() },
            wasted_actions: t.wasted,
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
