//! The integrated test: one command, three passes, one timeline.
//!
//! 1. **game**: the quest driver (game logic) carries the command out while
//!    the recorder takes a small video and fixed-interval keyframes;
//! 2. **perceive**: every game keyframe goes through the ML tracks (depth,
//!    semantic location, mask tracking), with opportunistic triangulation
//!    between keyframes where the robot moved across the line of sight;
//! 3. **decide**: the same world again (same seed and presets), and the ML
//!    logic decides where to go: capture, analyse, triangulate, forecast,
//!    move, capture.
//!
//! The UI advances the session in small steps (`test_step`) so the timeline
//! fills while it runs.
//!
//! A **live** session (`live_start`) records a command typed while the world
//! plays: the game pass runs in the live world from where it is (no reset),
//! the start state is kept as a copy, and the decision pass replays that copy
//! while the live world waits; afterwards the live world comes back as the
//! game pass left it. The play loop (`sim_step`) advances it.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use hand_core::balance::{reference, Source};
use hand_core::parts::Segment;
use hand_core::{BaseKind, BodyPlan};
use hand_quest::{Quest, QuestBook};
use hand_sds::{Level, Scope, Track};
use hand_vision::{triangulate, Bearing, Rgb};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use world_sim::World;

use crate::precision::PrecisionGates;
use crate::record::{capture_keyframe, capture_video, Keyframe, RecordCfg, Recording};
use crate::runtime::{Runtime, ScanReport, Submission};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TestCfg {
    pub record: RecordCfg,
    /// Ticks the game pass and the decision pass may each use.
    pub max_pass_ticks: u64,
    pub max_ml_decisions: u32,
    /// Keyframes the perception pass handles per `test_step` call.
    pub perceive_per_step: usize,
}

impl Default for TestCfg {
    fn default() -> Self {
        TestCfg { record: RecordCfg::default(), max_pass_ticks: 30_000, max_ml_decisions: 160, perceive_per_step: 6 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TestPhase {
    Game,
    Perceive,
    Decide,
    Done,
}

pub struct TestSession {
    pub id: String,
    pub command: String,
    pub seed: u64,
    pub body: BodyPlan,
    pub base: BaseKind,
    pub phase: TestPhase,
    pub game: Option<Recording>,
    pub decide: Option<Recording>,
    pub perceive_next: usize,
    pub game_quest: Option<Quest>,
    pub ml_quest: Option<Quest>,
    pub notes: Vec<String>,
    pub error: Option<String>,
    started_tick: u64,
    rays: HashMap<u32, Vec<Bearing>>,
    /// Recorded from the live world while it plays (see the module comment).
    pub live: bool,
    /// The quest needs finding things, so the ML decision pass is worth running.
    pub ml_needed: bool,
    start_world: Option<World>,
    stash: Option<LiveStash>,
}

/// The live world, its quests and landmarks, kept aside while the decision
/// pass replays the start state.
struct LiveStash {
    world: World,
    quests: QuestBook,
    gates: PrecisionGates,
    last_scan: Option<ScanReport>,
}

fn b64(v: &[u8]) -> String {
    STANDARD.encode(v)
}

fn r3(x: f32) -> f64 {
    ((x as f64) * 1000.0).round() / 1000.0
}

impl Runtime {
    pub fn test_session(&self) -> Option<&TestSession> {
        self.test.as_ref()
    }

    /// Starts a test: a fresh world with `seed`, the presets and the base, and the game pass.
    pub fn test_start(&mut self, command: &str, seed: u64, body: BodyPlan, base: Option<BaseKind>) -> Result<()> {
        self.cancel();
        let mut hand = self.cfg.hand;
        if let Some(b) = base {
            hand.base = b;
        }
        self.cfg.body = body;
        self.reset_with(Some(seed), None, Some(hand))?;
        self.test_seq += 1;
        let id = format!("{}-t{}", self.session_id, self.test_seq);
        self.begin_pass("game", &id);
        let s = self.submit(command);
        if !s.ok {
            self.recording = None;
            self.pass = "game".into();
            return Err(anyhow!("{}", s.error.unwrap_or_else(|| "명령을 이해하지 못했습니다".into())));
        }
        self.test = Some(TestSession {
            id: id.clone(),
            command: command.into(),
            seed,
            body,
            base: hand.base,
            phase: TestPhase::Game,
            game: None,
            decide: None,
            perceive_next: 0,
            game_quest: None,
            ml_quest: None,
            notes: vec![],
            error: None,
            started_tick: self.world.tick,
            rays: HashMap::new(),
            live: false,
            ml_needed: true,
            start_world: None,
            stash: None,
        });
        self.log("quest", Level::Info, format!("통합 테스트 {id} 시작: \"{command}\" · 시드 {seed} · 프리셋 {} · 베이스 {:?}", body.id(), hand.base), None);
        Ok(())
    }

    /// Starts recording a command in the live world, from where it is now:
    /// the game pass begins at once, perception and the ML decision follow by
    /// themselves as the play loop advances. Only when nothing else runs (the
    /// caller queues the command otherwise). A command that makes no quest
    /// (a stop, an error) records nothing.
    pub fn live_start(&mut self, command: &str) -> Result<Submission> {
        let start = self.world.clone();
        self.test = None;
        self.test_seq += 1;
        let id = format!("{}-t{}", self.session_id, self.test_seq);
        self.begin_pass("game", &id);
        let s = self.submit(command);
        let plan = s.quest_id.as_deref().and_then(|q| self.quests.get(q)).map(|q| q.plan.clone());
        let Some(plan) = plan.filter(|_| s.ok && !s.stopped) else {
            self.recording = None;
            self.pass = "game".into();
            return Ok(s);
        };
        let ml_needed = matches!(plan, hand_search::Plan::Fetch { .. } | hand_search::Plan::Collect { .. });
        self.test = Some(TestSession {
            id: id.clone(),
            command: command.into(),
            seed: self.cfg.seed,
            body: self.cfg.body,
            base: self.cfg.hand.base,
            phase: TestPhase::Game,
            game: None,
            decide: None,
            perceive_next: 0,
            game_quest: None,
            ml_quest: None,
            notes: vec![],
            error: None,
            started_tick: self.world.tick,
            rays: HashMap::new(),
            live: true,
            ml_needed,
            start_world: Some(start),
            stash: None,
        });
        self.log(
            "quest",
            Level::Info,
            format!("기록 시작 {id}: \"{command}\" — 1 게임(지금 월드) → 2 ML 인식 → 3 ML 결정{}", if ml_needed { "(같은 시작 상태 복제 월드)" } else { " 생략: 찾을 대상이 없는 명령" }),
            None,
        );
        Ok(s)
    }

    /// Ends a live session early. The live world comes back if the decision
    /// pass had set it aside.
    pub(crate) fn live_restore(&mut self) {
        if let Some(st) = self.test.as_mut().and_then(|t| t.stash.take()) {
            self.ml_task = None;
            self.task = None;
            self.world = st.world;
            self.quests = st.quests;
            self.gates = st.gates;
            self.last_scan = st.last_scan;
            self.log("world", Level::Info, "ML 결정 통과를 멈추고 라이브 월드로 돌아옴", None);
        }
    }

    fn begin_pass(&mut self, pass: &str, test_id: &str) {
        self.pass = pass.into();
        self.run_id = format!("{test_id}-{pass}");
        self.ml.reset();
        self.ml_task = None;
        let mut rec = Recording::new(pass, &self.run_id, self.world.tick);
        if pass == "game" {
            if let Ok(kf) = capture_keyframe(&self.world, &self.cam, self.test_cfg.record.quality, 0) {
                rec.keyframes.push(kf);
            }
        }
        self.recording = Some(rec);
    }

    /// Video frames and the game pass's fixed keyframes, after each tick.
    pub(crate) fn record_tick(&mut self) {
        let Some((started, is_game, n_video, n_kf)) = self.recording.as_ref().map(|r| (r.started_tick, r.pass == "game", r.video.len(), r.keyframes.len())) else { return };
        let cfg = self.test_cfg.record.clone();
        let since = self.world.tick.saturating_sub(started);
        let video = since % cfg.video_every_ticks.max(1) as u64 == 0 && n_video < cfg.max_video_frames;
        let key = is_game && since % cfg.keyframe_every_ticks.max(1) as u64 == 0;
        if video {
            if let Ok(v) = capture_video(&self.world, &self.cam, &cfg) {
                if let Some(r) = self.recording.as_mut() {
                    r.video.push(v);
                }
            }
        }
        // How far the robot's belief is off the truth (the record's, never the robot's).
        if since % BELIEF_EVERY_TICKS == 0 {
            let e = self.world.belief_error_m();
            if let Some(r) = self.recording.as_mut() {
                if r.belief_err.len() < MAX_BELIEF_POINTS {
                    r.belief_err.push((self.world.tick, e));
                }
            }
        }
        if key {
            let index = n_kf as u32;
            if n_kf >= cfg.max_keyframes {
                if let Some(r) = self.recording.as_mut() {
                    r.dropped_keyframes += 1;
                }
                return;
            }
            if let Ok(mut kf) = capture_keyframe(&self.world, &self.cam, cfg.quality, index) {
                kf.step = self.current_step_label();
                if let Some(r) = self.recording.as_mut() {
                    r.keyframes.push(kf);
                }
            }
        }
    }

    pub(crate) fn current_step_label(&self) -> Option<String> {
        let id = self.quests.current_id()?;
        let q = self.quests.get(&id)?;
        q.steps.iter().find(|s| s.state == hand_quest::State::Active).map(|s| s.label_ko.clone())
    }

    /// Advances the running test by up to `budget_ticks` world ticks (or a
    /// few keyframes in the perception pass). Returns the phase after.
    pub fn test_advance(&mut self, budget_ticks: u32) -> Result<TestPhase> {
        let Some(phase) = self.test.as_ref().map(|t| t.phase) else { return Err(anyhow!("진행 중인 통합 테스트가 없습니다")) };
        match phase {
            TestPhase::Game | TestPhase::Decide => {
                let start = self.test.as_ref().map_or(0, |t| t.started_tick);
                for _ in 0..budget_ticks.max(1) {
                    if !self.busy() {
                        break;
                    }
                    self.tick();
                    if self.world.tick.saturating_sub(start) > self.test_cfg.max_pass_ticks {
                        self.log("quest", Level::Warn, format!("통합 테스트: 통과당 틱 한도 {}를 넘어 중지", self.test_cfg.max_pass_ticks), None);
                        self.cancel();
                        break;
                    }
                }
                if !self.busy() {
                    self.end_pass(phase)?;
                }
            }
            TestPhase::Perceive => self.perceive_some()?,
            TestPhase::Done => {}
        }
        Ok(self.test.as_ref().map_or(TestPhase::Done, |t| t.phase))
    }

    fn end_pass(&mut self, phase: TestPhase) -> Result<()> {
        let mut rec = self.recording.take();
        if let Some(r) = rec.as_mut() {
            r.ended_tick = Some(self.world.tick);
        }
        let quest = self.quests.all().last().cloned();
        let Some(t) = self.test.as_mut() else { return Ok(()) };
        match phase {
            TestPhase::Game => {
                t.game = rec;
                t.game_quest = quest;
                t.phase = TestPhase::Perceive;
                t.perceive_next = 0;
                self.ml.reset();
            }
            TestPhase::Decide => {
                t.decide = rec;
                t.ml_quest = quest;
                t.phase = TestPhase::Done;
                self.pass = "game".into();
                if let Some(st) = t.stash.take() {
                    // Back to the live world as the game pass left it.
                    self.world = st.world;
                    self.quests = st.quests;
                    self.gates = st.gates;
                    self.last_scan = st.last_scan;
                    self.task = None;
                    self.ml_task = None;
                }
                let id = t.id.clone();
                let ok = |q: &Option<Quest>| q.as_ref().map_or("—".to_string(), |q| format!("{:?}", q.state).to_lowercase());
                let msg = format!("통합 테스트 {id} 끝: 게임 {} · ML {}", ok(&t.game_quest), ok(&t.ml_quest));
                self.log("quest", Level::Info, msg, None);
                for l in self.db.flush() {
                    self.audit.info("sds", format!("[DB] {l}"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Wanted vision labels of a quest: the source nodes of every target (a
    /// placed item, a parcel, is seen as itself).
    fn quest_labels(&self, q: Option<&Quest>) -> Vec<String> {
        let Some(q) = q else { return vec![] };
        let resources: Vec<String> = match &q.plan {
            hand_search::Plan::Fetch { targets, .. } | hand_search::Plan::Collect { targets } => targets.iter().map(|t| t.resource.clone()).collect(),
            _ => vec![],
        };
        resources
            .iter()
            .flat_map(|r| match world_sim::ItemKind::from_id(r) {
                Some(k) if !k.from_node() => vec![r.clone()],
                _ => self.ml.source_labels_for(r),
            })
            .collect()
    }

    fn perceive_some(&mut self) -> Result<()> {
        let Some(mut t) = self.test.take() else { return Ok(()) };
        let wanted = self.quest_labels(t.game_quest.as_ref());
        let n = t.game.as_ref().map_or(0, |g| g.keyframes.len());
        let end = (t.perceive_next + self.test_cfg.perceive_per_step).min(n);
        let prev_pass = std::mem::replace(&mut self.pass, "game".into());
        let run = t.game.as_ref().map(|g| g.run_id.clone()).unwrap_or_default();
        let prev_run = std::mem::replace(&mut self.run_id, run);
        // Depth, semantic location and the mask encoder for this step's
        // keyframes in batched calls; the loop below then only tracks.
        if let Some(g) = t.game.as_ref() {
            let rgbs: Vec<Rgb> = (t.perceive_next..end).filter_map(|i| g.keyframes.get(i).and_then(|k| k.rgb().ok())).collect();
            let refs: Vec<&Rgb> = rgbs.iter().collect();
            if let Err(e) = self.ml.prefetch(&refs) {
                t.notes.push(format!("일괄 추론 실패, 키프레임마다 다시 계산: {e}"));
            }
        }
        for i in t.perceive_next..end {
            let Some(mut kf) = t.game.as_mut().and_then(|g| g.keyframes.get_mut(i)).map(|k| std::mem::replace(k, placeholder(k))) else { continue };
            let res = self.ml_analyse(&mut kf, &wanted, None);
            if let Ok(out) = res {
                // Opportunistic triangulation over the keyframes of the same track.
                if let Some(s) = out.sighting {
                    let rays = t.rays.entry(s.track_id).or_default();
                    if rays.last().map_or(true, |r| (r.x - s.pose.x).hypot(r.z - s.pose.z) > 0.2) {
                        rays.push(Bearing { x: s.pose.x, z: s.pose.z, heading: s.heading, sigma: s.sigma });
                        if rays.len() > 4 {
                            rays.remove(0);
                        }
                    }
                    if let Ok(fix) = triangulate(rays, 6.0) {
                        let label = out.frame.target.as_ref().map(|x| x.label.clone()).unwrap_or_default();
                        if let Some(d) = s.depth_range_m.filter(|d| hand_vision::scale_usable(&fix, *d)) {
                            let scope = Scope::new(Track::Depth, &self.ml.depth_mode(), &label);
                            self.sds.record_calibration(&scope, "scale", fix.range_m as f64, d as f64);
                        }
                        if let Some(tv) = kf.ml.as_mut().and_then(|m| m.target.as_mut()) {
                            tv.tri_range_m = Some(fix.range_m);
                            tv.tri_parallax_deg = Some(fix.parallax_deg);
                            tv.tri_sigma_m = Some(fix.sigma_m);
                            tv.depth_scale = s.depth_range_m.filter(|d| hand_vision::scale_usable(&fix, *d)).map(|d| fix.range_m / d);
                            tv.range_used_m = Some(fix.range_m);
                        }
                    }
                }
                let target = wanted.first().cloned().unwrap_or_default();
                self.record_keyframe_row("game", &kf, &target);
            } else if let Err(e) = res {
                t.notes.push(format!("키프레임 {i}: {e}"));
            }
            if let Some(slot) = t.game.as_mut().and_then(|g| g.keyframes.get_mut(i)) {
                *slot = kf;
            }
        }
        t.perceive_next = end;
        self.pass = prev_pass;
        self.run_id = prev_run;
        if end >= n && t.live && !t.ml_needed {
            // A command with nothing to find: the game and perception rows are the record.
            t.phase = TestPhase::Done;
            self.log("quest", Level::Info, format!("기록 {} 끝: ML 결정 통과는 찾을 대상이 없어 생략", t.id), None);
            for l in self.db.flush() {
                self.audit.info("sds", format!("[DB] {l}"));
            }
            self.test = Some(t);
            return Ok(());
        }
        if end >= n {
            // Decision pass: the same world again, the ML logic finds and approaches.
            let (seed, body, base, cmd, id) = (t.seed, t.body, t.base, t.command.clone(), t.id.clone());
            if t.live {
                // The start state the game pass began from, in a copy; the live world waits.
                let start = t.start_world.take().unwrap_or_else(|| self.world.clone());
                let world = std::mem::replace(&mut self.world, start);
                let quests = std::mem::replace(&mut self.quests, QuestBook::new());
                let gates = std::mem::replace(&mut self.gates, PrecisionGates::new(self.cfg.gates.clone()));
                t.stash = Some(LiveStash { world, quests, gates, last_scan: self.last_scan.take() });
                self.task = None;
                self.ml_task = None;
                self.focus = None;
            } else {
                let mut hand = self.cfg.hand;
                hand.base = base;
                self.cfg.body = body;
                self.reset_with(Some(seed), None, Some(hand))?;
            }
            t.phase = TestPhase::Decide;
            t.started_tick = self.world.tick;
            self.test = Some(t);
            self.begin_pass("ml", &id);
            let s = self.submit(&cmd);
            if !s.ok {
                if let Some(t) = self.test.as_mut() {
                    t.error = s.error.clone();
                    t.phase = TestPhase::Done;
                }
                self.pass = "game".into();
            }
            return Ok(());
        }
        self.test = Some(t);
        Ok(())
    }

    // ----- views -----

    fn kf_brief(kf: &Keyframe) -> Value {
        let ml = kf.ml.as_ref();
        json!({
            "i": kf.index, "tick": kf.tick, "t": r3(kf.time_s as f32),
            "step": kf.step, "action": kf.action,
            "pose": [r3(kf.pose.x), r3(kf.pose.z), r3(kf.pose.yaw.to_degrees())],
            "balance": kf.balance.as_ref().map(|b| json!({"source": b.source, "source_ko": b.source_ko, "margin": r3(b.margin), "worst": b.worst_joint, "worst_ratio": r3(b.worst_ratio), "head_tilt_deg": r3(b.head_tilt_deg)})),
            "ml": ml.map(|m| json!({
                "absrel": m.depth_metrics.map(|d| r3(d.absrel)),
                "iou_truth": m.track_iou_truth.map(r3),
                "locate_err_px": m.locate_err_px.map(r3),
                "flags": m.flags,
                "target": m.target.as_ref().map(|t| json!({"track": t.track_id, "label": t.label, "bearing": r3(t.rel_bearing_deg), "depth_range": t.depth_range_m.map(r3), "tri_range": t.tri_range_m.map(r3), "true_range": t.range_true_m.map(r3), "scale": t.depth_scale.map(r3)})),
            })),
            "decision": kf.decision.as_ref().map(|d| json!({"label": d.decision, "kind": d.kind, "residual": d.residual_deg.map(r3), "executed": d.executed, "ok": d.ok})),
            "sds": kf.sds.iter().map(|(k, v)| (k.clone(), json!(r3(*v)))).collect::<serde_json::Map<String, Value>>(),
            // The action moment that made the keyframe, and what its focus views are about.
            "event": kf.event.as_ref().map(|e| json!({"act": e.act, "kind": e.kind, "label": e.label, "ok": e.ok, "parts": e.parts})),
            "focus": kf.focus.as_ref().map(|f| json!({"action": f.action, "action_ko": f.action_ko, "phase": f.phase, "parts": f.parts, "margin": r3(f.body.margin), "trunk_deg": r3(f.body.trunk_deg), "slip_ratio": f.hand.grip.map(|g| r3(g.slip_ratio)), "closure": r3(f.hand.closure)})),
        })
    }

    fn quest_json(q: &Option<Quest>) -> Value {
        match q {
            None => Value::Null,
            Some(q) => json!({
                "id": q.id, "title": q.title, "state": q.state,
                "steps": q.steps.iter().map(|s| json!({"kind": s.kind, "label_ko": s.label_ko, "state": s.state})).collect::<Vec<_>>(),
                "failure": q.failure, "result": q.result,
            }),
        }
    }

    /// `test_status()`: phase, progress and the keyframe briefs of every row.
    pub fn api_test_status(&self) -> Result<Value, String> {
        let Some(t) = &self.test else { return Ok(json!({"active": false})) };
        let live = self.recording.as_ref();
        let rows = |r: Option<&Recording>| -> Value {
            match r {
                None => json!({"keyframes": [], "video": 0, "start_tick": null, "end_tick": null}),
                Some(r) => json!({
                    "run_id": r.run_id, "start_tick": r.started_tick, "end_tick": r.ended_tick,
                    "video": r.video.len(), "dropped": r.dropped_keyframes,
                    "keyframes": r.keyframes.iter().map(Self::kf_brief).collect::<Vec<_>>(),
                    // When the robot looked and what it did on memory alone (the vision lane).
                    "vision": r.vision.iter().map(|m| json!({"tick": m.tick, "kind": m.kind, "what": m.what, "s0": m.sigma_before_m.map(r3), "s1": m.sigma_after_m.map(r3), "innov": m.innovation_m.map(r3)})).collect::<Vec<_>>(),
                    "belief_err": r.belief_err.iter().map(|(t, e)| json!([t, r3(*e)])).collect::<Vec<_>>(),
                }),
            }
        };
        let game = if t.phase == TestPhase::Game { live } else { t.game.as_ref() };
        let decide = if t.phase == TestPhase::Decide { live } else { t.decide.as_ref() };
        let game_quest = if t.phase == TestPhase::Game { self.quests.all().last().cloned() } else { t.game_quest.clone() };
        let ml_quest = if t.phase == TestPhase::Decide { self.quests.all().last().cloned() } else { t.ml_quest.clone() };
        Ok(json!({
            "active": true,
            "live": t.live, "ml_needed": t.ml_needed,
            "id": t.id, "command": t.command, "seed": t.seed, "preset": t.body.id(), "body": t.body, "base": t.base,
            "phase": t.phase, "tick": self.world.tick,
            "perceive": {"done": t.perceive_next, "total": t.game.as_ref().map_or(0, |g| g.keyframes.len())},
            "modes": {"depth": self.ml.depth_mode(), "locate": self.ml.locate_mode(), "track": self.ml.track_mode(), "ttm": self.has_ttm()},
            "game": rows(game), "decide": rows(decide),
            "game_quest": Self::quest_json(&game_quest), "ml_quest": Self::quest_json(&ml_quest),
            "notes": t.notes, "error": t.error,
            "keyframe_every_ticks": self.test_cfg.record.keyframe_every_ticks,
        }))
    }

    fn rec_of(&self, pass: &str) -> Option<&Recording> {
        let t = self.test.as_ref()?;
        match (pass, t.phase) {
            ("game", TestPhase::Game) | ("ml", TestPhase::Decide) => self.recording.as_ref(),
            ("game", _) => t.game.as_ref(),
            _ => t.decide.as_ref(),
        }
    }

    /// `test_keyframe({pass, index})`: the keyframe's pictures and everything the tracks said.
    pub fn api_test_keyframe(&self, pass: &str, index: usize) -> Result<Value, String> {
        self.api_test_keyframe_with(pass, index, true)
    }

    /// `test_keyframe({pass, index, close_ups})`: without the focus close-ups
    /// (`close_ups: false`) for the timeline's tiles, which draw only the frame
    /// or the focus numbers.
    pub fn api_test_keyframe_with(&self, pass: &str, index: usize, close_ups: bool) -> Result<Value, String> {
        let rec = self.rec_of(pass).ok_or("이 통과의 기록이 없습니다")?;
        let kf = rec.keyframes.get(index).ok_or_else(|| format!("키프레임 {index}이(가) 없습니다"))?;
        let mut v = json!({
            "brief": Self::kf_brief(kf),
            "frame_webp": b64(&kf.webp),
            "ml_webp": kf.ml_webp.as_deref().map(b64),
            "width": kf.intr.width, "height": kf.intr.height,
            "ml": kf.ml, "decision": kf.decision, "balance": kf.balance,
            // Every focus view of the instant (hand, finger joints, feet, leg joints, waist, neck, wheels, tracks).
            "focus": kf.focus, "event": kf.event,
            "truth": kf.truth.objects.iter().map(|o| json!({"entity": o.entity, "label": o.label, "label_ko": o.label_ko, "bbox": [o.bbox[0] as f32 / kf.truth.src_width as f32, o.bbox[1] as f32 / kf.truth.src_height as f32, o.bbox[2] as f32 / kf.truth.src_width as f32, o.bbox[3] as f32 / kf.truth.src_height as f32], "depth_m": r3(o.depth_m)})).collect::<Vec<_>>(),
        });
        // The close-ups: the hand, the body from the side, low by the feet or wheels.
        if close_ups {
            v["focus_webp"] = Value::Object(kf.focus_webp.iter().map(|(k, w)| (k.clone(), json!(b64(w)))).collect());
        }
        Ok(v)
    }

    /// `test_video({pass, from, count})`: video frames as WebP.
    pub fn api_test_video(&self, pass: &str, from: usize, count: usize) -> Result<Value, String> {
        let rec = self.rec_of(pass).ok_or("이 통과의 기록이 없습니다")?;
        let frames: Vec<Value> = rec.video.iter().skip(from).take(count.clamp(1, 120)).map(|v| json!({"tick": v.tick, "t": r3(v.time_s as f32), "webp": b64(&v.webp)})).collect();
        Ok(json!({"from": from, "total": rec.video.len(), "frames": frames}))
    }

    /// `test_stats()`: success per pass, kind, target, preset and step; the N:N
    /// balance matrix from the SDS (value, reference, deviation, share in band).
    pub fn api_test_stats(&self) -> Result<Value, String> {
        let stats = self.db.stats();
        let mut nn: Vec<Value> = Vec::new();
        for s in Source::ALL.into_iter().filter(|s| *s != Source::Quiet) {
            for g in Segment::ALL {
                let scope = Scope::new(Track::Balance, &format!("{}>{}", s.id(), g.id()), "");
                for axis in ["tilt_deg", "lean_deg", "restore_frac", "shift_frac", "peak_ratio", "grip_effort", "com_shift_frac", "ankle_rise", "margin", "cop_shift_frac"] {
                    let rf = reference(s, g, axis, &hand_core::balance::RefCtx { load_ratio: 0.0, load_dist_m: 0.0, slope_deg: 10.0 });
                    let rep = self.sds.exact_reference(&scope, axis);
                    let base = self.sds.exact_baseline(&scope, axis);
                    if rep.is_none() && base.is_none() {
                        continue;
                    }
                    nn.push(json!({
                        "source": s.id(), "source_ko": s.label_ko(), "target": g.id(), "target_ko": g.label_ko(), "axis": axis,
                        "mean": rep.as_ref().map(|r| r.mean).or(base.map(|b| b.0)),
                        "n": rep.as_ref().map(|r| r.n).or(base.map(|b| b.2)),
                        "dev": rep.as_ref().map(|r| r.mean_dev), "in_band": rep.as_ref().map(|r| r.in_band_rate),
                        // The reference recorded with the observations (it scales with load and slope);
                        // the context-free one only when nothing was recorded against a reference.
                        "reference": rep.as_ref().map(|r| r.mean_ref).filter(|x| x.is_finite()).or(rf.as_ref().map(|r| r.value as f64)),
                        "tolerance": rep.as_ref().map(|r| r.mean_tol).filter(|x| x.is_finite()).or(rf.as_ref().map(|r| r.tolerance as f64)),
                        "cite": rf.as_ref().map(|r| r.cite),
                    }));
                }
            }
        }
        let mut gains: BTreeMap<String, f64> = BTreeMap::new();
        for (s, g) in &self.world.ff_gains {
            gains.insert(s.id().into(), *g as f64);
        }
        let runs: Vec<Value> = self.db.runs().iter().rev().take(20).map(|r| json!({"run_id": r.run_id, "pass": r.pass, "kind": r.kind, "target": r.target, "preset": r.preset, "ok": r.ok, "failure": r.failure, "time_s": r.time_s, "steps": r.steps})).collect();
        Ok(json!({"stats": stats, "nn": nn, "ff_gains": gains, "runs": runs, "db_dir": self.db.dir().map(|d| d.display().to_string())}))
    }
}

/// The belief's error is recorded every this many ticks, up to this many points.
const BELIEF_EVERY_TICKS: u64 = 5;
const MAX_BELIEF_POINTS: usize = 6000;

/// A cheap stand-in left in a slot while its keyframe is being analysed.
fn placeholder(k: &Keyframe) -> Keyframe {
    Keyframe {
        index: k.index,
        tick: k.tick,
        time_s: k.time_s,
        pose: k.pose,
        intr: k.intr,
        webp: Vec::new(),
        truth: crate::ml::KeyTruth { width: 0, height: 0, ids: vec![], depth: vec![], objects: vec![], positions: vec![], src_width: 0, src_height: 0 },
        step: None,
        action: None,
        balance: None,
        ml: None,
        ml_webp: None,
        decision: None,
        sds: BTreeMap::new(),
        event: k.event.clone(),
        focus: None,
        focus_webp: Vec::new(),
    }
}
