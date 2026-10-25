//! The runtime: one world, one robot, one set of gates, one SDS, one log.
//!
//! Nothing here depends on a UI. The Tauri commands (and the dev server in
//! `examples/serve.rs`) call these methods and turn the results into JSON.

use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use hand_core::{BodyStatus, HandConfig};
use hand_quest::{Labels, Quest, QuestBook};
use hand_sds::{AuditLog, Level, LogLine, SdsStatus, SdsStore};
use hand_search::semantic::PhraseEmbedder;
use hand_search::{Lexicon, ParsedCommand, Parser, Plan};
use hand_ts::Forecast;
use serde::Serialize;
use serde_json::{json, Value};
use world_sim::{amount_text, encode_webp, Action, Camera, ItemKind, NodeKind, TimedEvent, World, WorldConfig, WorldEvent, WorldSnapshot};

use hand_search::text::eul_reul;
use crate::config::{GateCfg, NoiseCfg, RuntimeConfig};
use crate::driver::Task;
use crate::perception::{OraclePerception, Recognizer, ScanInput};
use crate::perception::EntKind;
use crate::precision::{Judged, MemCtx, PrecisionGates};
use crate::vision::{in_view, memory_ok, DriftRate, VisionMark};
use world_sim::world::{DELIVER_REACH_M, EYE_HEIGHT_M};
use hand_sds::{Scope, Track};

/// What one scan saw and decided.
#[derive(Clone, Debug, Serialize)]
pub struct ScanReport {
    pub seq: u64,
    pub tick: u64,
    pub trigger: String,
    pub mode: String,
    pub device: String,
    pub tiles: usize,
    pub embed_ms: f32,
    pub total_ms: f32,
    pub judged: Vec<Judged>,
    pub scene: Vec<(String, f32)>,
}

impl ScanReport {
    pub fn accepted(&self) -> impl Iterator<Item = &Judged> {
        self.judged.iter().filter(|j| j.accepted)
    }

    /// Mean score of the accepted detections, if any.
    pub fn confidence(&self) -> Option<f32> {
        let v: Vec<f32> = self.accepted().map(|j| j.cand.score).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
    }
}

/// Counts against the renderer's ground truth. Only the stand-in recogniser
/// can fill these in; a real recogniser leaves them at zero.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EvalCounters {
    pub scans: u64,
    pub raw_total: u64,
    pub raw_correct: u64,
    pub accepted_total: u64,
    pub accepted_correct: u64,
    /// Objects in view, summed over scans.
    pub truths: u64,
    pub truths_found_raw: u64,
    pub truths_found_accepted: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct EvalSummary {
    pub raw_precision: Option<f64>,
    pub raw_recall: Option<f64>,
    pub gated_precision: Option<f64>,
    pub gated_recall: Option<f64>,
}

fn ratio(a: u64, b: u64) -> Option<f64> {
    (b > 0).then(|| a as f64 / b as f64)
}

impl EvalCounters {
    pub fn summary(&self) -> EvalSummary {
        EvalSummary {
            raw_precision: ratio(self.raw_correct, self.raw_total),
            raw_recall: ratio(self.truths_found_raw, self.truths),
            gated_precision: ratio(self.accepted_correct, self.accepted_total),
            gated_recall: ratio(self.truths_found_accepted, self.truths),
        }
    }

    fn add(&mut self, truths: &[String], judged: &[Judged]) {
        self.scans += 1;
        self.truths += truths.len() as u64;
        for j in judged {
            let correct = j.cand.is_correct() == Some(true);
            self.raw_total += 1;
            self.raw_correct += correct as u64;
            if j.accepted {
                self.accepted_total += 1;
                self.accepted_correct += correct as u64;
            }
        }
        for t in truths {
            let hit = |only_accepted: bool| judged.iter().any(|j| j.cand.truth.as_deref() == Some(t) && j.cand.is_correct() == Some(true) && (!only_accepted || j.accepted));
            self.truths_found_raw += hit(false) as u64;
            self.truths_found_accepted += hit(true) as u64;
        }
    }
}

/// What the vision schedule did over the robot's quests (the benchmark reads it).
#[derive(Clone, Debug, Default, Serialize)]
pub struct VisionStats {
    /// Seconds spent approaching something remembered, and the looks taken meanwhile.
    pub approach_s: f64,
    pub approach_looks: u32,
    /// Looks the schedule took in all (pattern, before acting, glances), and actions on memory alone.
    pub looks: u32,
    pub blind_acts: u32,
    /// Picks tried from the side, and in all.
    pub side_picks: u32,
    pub picks: u32,
    /// Aim error of the picks made on memory alone and of those just looked at, metres (the record's).
    pub memory_aim_m: Vec<f32>,
    pub seen_aim_m: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Submission {
    pub ok: bool,
    pub stopped: bool,
    pub quest_id: Option<String>,
    pub error: Option<String>,
    pub error_code: Option<String>,
    /// `lexical` or `lexical+embedding`.
    pub mode: String,
    pub parsed: ParsedCommand,
}

#[derive(Clone, Debug, Serialize)]
pub struct TsView {
    pub label: String,
    pub history: Vec<f64>,
    pub forecast: Option<Forecast>,
    /// Labels that have a history, longest first.
    pub labels: Vec<(String, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub label: String,
    pub label_ko: String,
    pub score: f32,
    pub margin: f32,
    pub accepted: bool,
    pub blocked_by: Option<String>,
    /// Why it was accepted or held, in Korean.
    pub reason: String,
    /// `[x0, y0, x1, y1]` as fractions of the frame.
    pub bbox: [f32; 4],
    pub landmark: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FrameOut {
    #[serde(skip)]
    pub webp: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bytes: usize,
    pub encode_ms: f32,
    pub lossless: bool,
    pub tick: u64,
    pub detections: Vec<Detection>,
    /// Joint markers, CoM, ZMP and support of the third-person view (pixels).
    pub overlay: Option<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameView {
    /// The robot's first-person camera.
    Agent,
    /// A top-down map of the surroundings.
    Top,
    /// Third person: a chase camera `orbit_deg` around the robot, with its
    /// links and joints as the joint dynamics move them.
    Chase { orbit_deg: i16 },
    /// A close-up on the part an action works with: the hand, the body from
    /// the side (waist, neck, legs), or low by the feet, wheels or tracks.
    Focus(world_sim::render::FocusCam),
}

struct WorldLabels<'a>(&'a World);

impl Labels for WorldLabels<'_> {
    fn resource_ko(&self, id: &str) -> String {
        ItemKind::from_id(id).map(|k| k.spec().label_ko.to_string()).unwrap_or_else(|| id.to_string())
    }
    fn place_ko(&self, id: &str) -> String {
        self.0.waypoint_by_key(id).map(|w| w.label_ko.clone()).unwrap_or_else(|| id.to_string())
    }
    fn resource_mined(&self, id: &str) -> bool {
        ItemKind::from_id(id).map_or(true, |k| k.from_node())
    }
}

/// Korean name of a dictionary label.
pub fn label_ko(world: &World, label: &str) -> String {
    if let Some(k) = NodeKind::ALL.into_iter().find(|k| k.id() == label) {
        return k.spec().label_ko.to_string();
    }
    if let Some(k) = ItemKind::from_id(label) {
        return k.spec().label_ko.to_string();
    }
    if let Some(w) = world.waypoint_by_key(label) {
        return w.label_ko.clone();
    }
    label.to_string()
}

/// Farthest a waypoint's flag is trusted to fix the robot's position, metres (est: the flag fills a few pixels there).
pub(crate) const RELOCALIZE_RANGE_M: f32 = 8.0;
/// Validation gate of a fix: a sighting whose squared innovation exceeds this
/// many times its variance (the pose's and the sighting's) is not the flag the
/// map has there (a misread one) and is left out: the 99% point of χ² with two
/// degrees of freedom (the validation region of Bar-Shalom & Fortmann 1988,
/// *Tracking and Data Association*).
const FIX_GATE_CHI2: f32 = 9.21;

/// A fix of the robot's pose a flag gave, and the stretch of odometry it
/// corrects (from the fix before it to this one).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PoseFix {
    pub dx: f32,
    pub dz: f32,
    /// The odometer at this fix and at the one before it.
    pub odo_m: f32,
    pub prev_odo_m: f32,
}

impl PoseFix {
    /// The share of the fix a memory seen at odometer `odo` moves by: the
    /// drift's variance it had taken in when it was seen over the variance at
    /// the fix (all of it for what was seen just before the fix, none for
    /// what was seen at or before the last one).
    pub fn share(&self, odo: f32, rate: DriftRate) -> f32 {
        if odo <= self.prev_odo_m {
            return 0.0;
        }
        let var = |d: f32| (rate.per_m * d).powi(2) + rate.walk_m * rate.walk_m * d;
        (var(odo.min(self.odo_m) - self.prev_odo_m) / var(self.odo_m - self.prev_odo_m).max(1e-12)).clamp(0.0, 1.0)
    }

    /// Where a memory seen at odometer `odo` lies after the fix.
    pub fn shift(&self, x: f32, z: f32, odo: f32, rate: DriftRate) -> (f32, f32) {
        let k = self.share(odo, rate);
        (x + k * self.dx, z + k * self.dz)
    }
}

fn new_world(cfg: &RuntimeConfig) -> Result<World> {
    World::new(WorldConfig { seed: cfg.seed, cells: cfg.cells, hand: cfg.hand, body: cfg.body, terrain: cfg.terrain, odometry: cfg.odometry, parcels: cfg.parcels }).map_err(|e| anyhow!("월드를 만들 수 없습니다: {e}"))
}

pub struct Runtime {
    pub cfg: RuntimeConfig,
    /// Composite N:N phases scored as the quests produce them.
    pub(crate) nn_live: crate::nnlive::NnLive,
    pub(crate) world: World,
    pub(crate) cam: Camera,
    parser: Parser,
    embedder: Option<Arc<dyn PhraseEmbedder + Send + Sync>>,
    pub(crate) recognizer: Box<dyn Recognizer>,
    pub(crate) gates: PrecisionGates,
    pub(crate) sds: SdsStore,
    pub(crate) audit: AuditLog,
    pub(crate) quests: QuestBook,
    pub(crate) task: Option<Task>,
    pub(crate) last_scan: Option<ScanReport>,
    /// A fix of the pose a flag gave, for the next driver step (`apply_fix`).
    pub(crate) pending_fix: Option<PoseFix>,
    scan_seq: u64,
    last_device: String,
    pub(crate) eval: EvalCounters,
    last_flush_tick: u64,
    /// Resource the running quest is after; only its landmarks are logged loudly.
    pub(crate) focus: Option<String>,
    /// Test records: quest runs, ML keyframes, balance effort, success table.
    pub(crate) db: hand_sds::TestDb,
    /// `game` (the quest driver decides) or `ml` (the ML logic decides).
    pub(crate) pass: String,
    pub(crate) run_id: String,
    pub(crate) session_id: String,
    pub(crate) test_seq: u32,
    pub(crate) recording: Option<crate::record::Recording>,
    pub(crate) ml: crate::ml::MlPipeline,
    pub(crate) ml_task: Option<crate::mldrive::MlTask>,
    pub(crate) test: Option<crate::testrun::TestSession>,
    pub test_cfg: crate::testrun::TestCfg,
    /// Commands typed while a live recording runs, started one after another.
    pub(crate) live_queue: std::collections::VecDeque<String>,
    /// What the vision schedule did (`vision`).
    pub(crate) vstats: VisionStats,
}

impl Runtime {
    /// A runtime over a fresh world. With `data_dir` the SDS ledgers and the
    /// audit log are kept on disk under it.
    pub fn new(cfg: RuntimeConfig, lexicon_json: &str, data_dir: Option<&Path>) -> Result<Runtime> {
        let lex = Lexicon::from_json(lexicon_json)?;
        let recognizer: Box<dyn Recognizer> = Box::new(OraclePerception::new(&lex, cfg.noise.clone(), cfg.min_pixels));
        let ml = crate::ml::MlPipeline::stand_in(&lex, cfg.seed);
        let parser = Parser::new(lex);
        let world = new_world(&cfg)?;
        let ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
        let (sds, audit, db) = match data_dir {
            Some(dir) => (SdsStore::open(&dir.join("sds")), AuditLog::open(&dir.join("logs"), &format!("session_{ms}"), 4000)?, hand_sds::TestDb::open(&dir.join("db"))),
            None => (SdsStore::memory(), AuditLog::memory(4000), hand_sds::TestDb::memory()),
        };
        let cam = cfg.camera.camera();
        let gates = PrecisionGates::new(cfg.gates.clone());
        let mut rt = Runtime {
            cfg,
            world,
            cam,
            parser,
            embedder: None,
            recognizer,
            gates,
            sds,
            audit,
            quests: QuestBook::new(),
            task: None,
            last_scan: None,
            pending_fix: None,
            scan_seq: 0,
            last_device: String::new(),
            nn_live: crate::nnlive::NnLive::default(),
            eval: EvalCounters::default(),
            last_flush_tick: 0,
            focus: None,
            db,
            pass: "game".into(),
            run_id: format!("s{ms}-live"),
            session_id: format!("s{ms}"),
            test_seq: 0,
            recording: None,
            ml,
            ml_task: None,
            test: None,
            test_cfg: crate::testrun::TestCfg::default(),
            live_queue: std::collections::VecDeque::new(),
            vstats: VisionStats::default(),
        };
        let msg = format!(
            "월드 시작: seed {} · {}×{}칸 · 베이스 {:?} · 손가락 {} · 인식기 {}",
            rt.cfg.seed,
            rt.world.terrain.w,
            rt.world.terrain.d,
            rt.world.hand.base.kind,
            u8::from(rt.world.hand.config.fingers),
            rt.recognizer.mode()
        );
        rt.log("world", Level::Info, msg, Some(json!({"seed": rt.cfg.seed, "cells": rt.cfg.cells})));
        Ok(rt)
    }

    // ----- plumbing -----

    pub(crate) fn log(&mut self, cat: &str, level: Level, msg: impl Into<String>, data: Option<Value>) {
        self.audit.log(cat, level, msg, data.map(tidy));
    }

    /// A line in the `device` category, for things the host noticed (a folder it could not use).
    pub fn note_device(&mut self, msg: &str) {
        self.log("device", Level::Warn, msg, None);
    }

    pub(crate) fn log_world_events(&mut self, events: &[TimedEvent]) {
        for te in events {
            let (level, msg) = match &te.event {
                WorldEvent::ActionStarted { action } => (Level::Debug, format!("동작 시작 {action}")),
                WorldEvent::ActionDone { action, ok, reason } => (
                    if *ok { Level::Debug } else { Level::Warn },
                    format!("동작 {} {action}{}", if *ok { "완료" } else { "실패" }, reason.as_ref().map(|r| format!(" ({r})")).unwrap_or_default()),
                ),
                WorldEvent::CurbClimbed { height_m } => (Level::Info, format!("연석 {:.0} cm를 넘음", height_m * 100.0)),
                WorldEvent::NodeBroken { node, kind, drops } => {
                    (Level::Info, format!("{}({node}){} 캠 → 아이템 {}개 떨어짐", kind.spec().label_ko, eul_reul(kind.spec().label_ko), drops.len()))
                }
                WorldEvent::ItemPicked { drop, item, mass_kg, grip_margin, amount } => {
                    (Level::Info, format!("{} {drop} 줍기 ({}개, {mass_kg:.1} kg, 마찰 여유 {grip_margin:.2})", item.spec().label_ko, amount_text(*amount)))
                }
                WorldEvent::Delivered { waypoint, item, amount } => (Level::Info, format!("{waypoint}에 {} {}개 인도", item.spec().label_ko, amount_text(*amount))),
                WorldEvent::ItemDropped { item, amount, waypoint, drops } => (
                    Level::Info,
                    match waypoint {
                        Some(w) => format!("{} {}개를 {w} 더미에 내려놓음", item.spec().label_ko, amount_text(*amount)),
                        None => format!("{} {}개를 땅에 내려놓음 ({})", item.spec().label_ko, amount_text(*amount), drops.join(", ")),
                    },
                ),
                WorldEvent::ItemSplit { drop, piece, item, amount, left } => {
                    (Level::Info, format!("{} {drop} 쪼갬 → 조각 {piece} {}개 + 남은 {}개", item.spec().label_ko, amount_text(*amount), amount_text(*left)))
                }
                WorldEvent::BatteryDepleted => (Level::Error, "배터리 방전".to_string()),
                WorldEvent::Act { act, kind, label, ok, .. } => (if *ok { Level::Debug } else { Level::Warn }, format!("[{act}:{kind}] {label}")),
            };
            self.audit.log("world", level, msg, Some(tidy(json!({"tick": te.tick, "time_s": te.time_s, "event": te.event}))));
        }
    }

    pub(crate) fn flush_sds(&mut self) {
        self.last_flush_tick = self.world.tick;
        for l in self.sds.flush() {
            self.audit.info("sds", l);
        }
    }

    // ----- access -----

    pub fn world(&self) -> &World {
        &self.world
    }

    /// For scenarios and tests that need to place things.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn camera(&self) -> &Camera {
        &self.cam
    }

    pub fn gates(&self) -> &PrecisionGates {
        &self.gates
    }

    pub fn sds(&self) -> &SdsStore {
        &self.sds
    }

    pub fn sds_mut(&mut self) -> &mut SdsStore {
        &mut self.sds
    }

    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    pub fn quest_book(&self) -> &QuestBook {
        &self.quests
    }

    pub fn quests(&self) -> &[Quest] {
        self.quests.all()
    }

    pub fn eval(&self) -> &EvalCounters {
        &self.eval
    }

    /// What the vision schedule did so far.
    pub fn vision_stats(&self) -> &VisionStats {
        &self.vstats
    }

    pub fn last_scan(&self) -> Option<&ScanReport> {
        self.last_scan.as_ref()
    }

    pub fn parser(&self) -> &Parser {
        &self.parser
    }

    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    pub fn recognizer_mode(&self) -> &str {
        self.recognizer.mode()
    }

    pub fn busy(&self) -> bool {
        self.task.is_some() || self.ml_task.is_some() || self.quests.current_id().is_some()
    }

    /// The ML pipeline (depth, semantic location, tracking).
    pub fn ml_pipeline(&self) -> &crate::ml::MlPipeline {
        &self.ml
    }

    pub fn test_db(&self) -> &hand_sds::TestDb {
        &self.db
    }

    pub fn test_db_mut(&mut self) -> &mut hand_sds::TestDb {
        &mut self.db
    }

    /// `game` or `ml`: who decides the running quest.
    pub fn pass(&self) -> &str {
        &self.pass
    }

    /// Loads TIPSv2-B/14 DPT from `dir` as the ML depth model.
    pub fn use_tips(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        let t0 = Instant::now();
        let m = hand_vision::TipsDepth::load(dir, device)?;
        self.ml.depth = crate::ml::DepthEngine::Model(Box::new(m));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("TIPSv2 DPT 깊이 모델 로드 ({ms:.0} ms) · {} · {}", hand_ml::device_label(device), dir.display()), None);
        Ok(())
    }

    /// Loads SAM (MobileSAM or ViT-B, whichever file `dir` has) as the mask
    /// tracker. Without weights the region-growing tracker stays.
    pub fn use_sam(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        let t0 = Instant::now();
        let m = hand_vision::SamModel::load(dir, device)?;
        let mode = m.variant().mode();
        self.ml.tracker = Box::new(hand_vision::SamTracker::new(Arc::new(m)));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("{mode} 마스크 모델 로드 ({ms:.0} ms) · {} · {}", hand_ml::device_label(device), dir.display()), None);
        Ok(())
    }

    /// Back to the depth stand-in (no TIPS weights in use).
    pub fn use_depth_stand_in(&mut self) {
        self.ml.depth = crate::ml::DepthEngine::StandIn(crate::ml::DepthStandIn::new(self.cfg.seed));
        self.log("embed", Level::Info, "깊이: 대역 모델(stand-in)", None);
    }

    /// Back to the region-growing mask tracker.
    pub fn use_region_tracker(&mut self) {
        self.ml.tracker = Box::new(hand_vision::RegionTracker::default());
        self.log("embed", Level::Info, "마스크 추적: 영역 확장(region-grow)", None);
    }

    // ----- configuration -----

    pub fn set_gates(&mut self, gates: GateCfg) {
        self.cfg.gates = gates.clone();
        self.gates.set_cfg(gates);
        self.log("vision", Level::Info, "정밀도 게이트 설정이 바뀜", Some(serde_json::to_value(&self.cfg.gates).unwrap_or(Value::Null)));
    }

    pub fn set_noise(&mut self, noise: NoiseCfg) {
        self.recognizer.set_noise(&noise);
        self.cfg.noise = noise;
        self.log("vision", Level::Info, "모의 인식기 오류 설정이 바뀜", Some(serde_json::to_value(&self.cfg.noise).unwrap_or(Value::Null)));
    }

    /// Replaces the recogniser (for example with SigLIP2).
    pub fn set_recognizer(&mut self, r: Box<dyn Recognizer>) {
        let mode = r.mode().to_string();
        self.recognizer = r;
        self.eval = EvalCounters::default();
        self.gates.clear();
        self.log("vision", Level::Info, format!("인식기 교체: {mode}"), None);
    }

    /// SigLIP2 as the recogniser, and for the command parser's embedding
    /// fallback. The dictionary's prompts are embedded here, once; with
    /// `bank_dir` the vectors are kept between runs.
    pub fn use_siglip(&mut self, model: Arc<hand_ml::Siglip2>, bank_dir: Option<&Path>) -> Result<()> {
        let t0 = Instant::now();
        let phrases = Arc::new(crate::embedding::BankedPhrases::new(Arc::clone(&model), bank_dir));
        self.ml.locate = crate::ml::LocateEngine::Model(Box::new(hand_vision::SiglipLocator::new(Arc::clone(&model))));
        let rec = crate::embedding::EmbeddingPerception::siglip(model, self.parser.lexicon(), bank_dir, Default::default())?;
        self.attach_embedder(phrases)?;
        self.set_recognizer(Box::new(rec));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("SigLIP2 인식기 준비 ({ms:.0} ms)"), Some(json!({"ms": ms})));
        Ok(())
    }

    /// Lets the forecasts use Granite TTM, or stop using it.
    pub fn attach_ttm(&mut self, ttm: Option<Arc<hand_ts::ttm::TtmForecaster>>) {
        let msg = match &ttm {
            Some(t) => format!("Granite TTM 연결: 문맥 {} · 예측 {} 스텝", t.context_length(), t.prediction_length()),
            None => "Granite TTM 해제: last/ewma/holt 융합만 사용".to_string(),
        };
        self.gates.set_ttm(ttm);
        self.log("ts", Level::Info, msg, None);
    }

    pub fn has_ttm(&self) -> bool {
        self.gates.has_ttm()
    }

    /// Loads Granite TTM from `dir` (a `config.json` and the safetensors next
    /// to it) onto `device` and lets the forecasts use it. If the files are not
    /// there or do not fit, nothing changes and the error says why.
    pub fn load_ttm(&mut self, dir: &Path, device: &candle_core::Device) -> Result<()> {
        let t0 = Instant::now();
        let ttm = hand_ts::ttm::TtmForecaster::load(dir, device).map_err(|e| anyhow!("Granite TTM 을 {} 에서 읽지 못했습니다: {e}", dir.display()))?;
        self.attach_ttm(Some(Arc::new(ttm)));
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("ts", Level::Info, format!("Granite TTM 로드 완료 ({ms:.0} ms) · {}", dir.display()), Some(json!({"ms": ms})));
        Ok(())
    }

    /// Back to the stand-in recogniser.
    pub fn use_oracle(&mut self) {
        let r = OraclePerception::new(self.parser.lexicon(), self.cfg.noise.clone(), self.cfg.min_pixels);
        self.set_recognizer(Box::new(r));
    }

    /// Lets the command parser use embeddings for words the dictionary misses.
    pub fn attach_embedder(&mut self, emb: Arc<dyn PhraseEmbedder + Send + Sync>) -> Result<()> {
        let t0 = Instant::now();
        self.parser.attach_embedder(&*emb)?;
        self.embedder = Some(emb);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.log("embed", Level::Info, format!("명령어 사전 임베딩 완료 ({ms:.0} ms)"), Some(json!({"ms": ms})));
        Ok(())
    }

    /// A new world. The SDS keeps what it learned.
    pub fn reset(&mut self, seed: Option<u64>) -> Result<()> {
        self.reset_with(seed, None, None)
    }

    /// A new world, optionally with another size or another hand and base.
    /// Nothing changes if the world cannot be built.
    pub fn reset_with(&mut self, seed: Option<u64>, cells: Option<usize>, hand: Option<HandConfig>) -> Result<()> {
        let mut next = self.cfg.clone();
        if let Some(s) = seed {
            next.seed = s;
        }
        if let Some(c) = cells {
            next.cells = c;
        }
        if let Some(h) = hand {
            next.hand = h;
        }
        let world = new_world(&next)?;
        self.cfg = next;
        self.world = world;
        self.task = None;
        self.ml_task = None;
        self.quests.clear();
        self.gates.clear();
        self.last_scan = None;
        self.focus = None;
        self.eval = EvalCounters::default();
        self.flush_sds();
        let msg = format!(
            "월드 다시 만듦: seed {} · {}×{}칸 · 베이스 {:?} · 손가락 {}",
            self.cfg.seed,
            self.world.terrain.w,
            self.world.terrain.d,
            self.world.hand.base.kind,
            u8::from(self.world.hand.config.fingers)
        );
        self.log("world", Level::Info, msg, Some(json!({"seed": self.cfg.seed, "cells": self.cfg.cells})));
        Ok(())
    }

    /// The world back to how its seed made it (nodes, empty ground and piles,
    /// the robot at the base camp), keeping the robot's body as it is: battery
    /// charge, joint heat and wear stay. Quests and landmarks are cleared; the
    /// SDS keeps what it learned. Use [`Runtime::reset_body`] for the body.
    pub fn reset_world(&mut self) -> Result<()> {
        self.cancel();
        self.world.reset_world_keep_body().map_err(|e| anyhow!("월드를 되돌릴 수 없습니다: {e}"))?;
        self.task = None;
        self.ml_task = None;
        self.quests.clear();
        self.gates.clear();
        self.last_scan = None;
        self.focus = None;
        self.eval = EvalCounters::default();
        self.flush_sds();
        let msg = format!("월드 리셋: seed {} 처음 상태로 (신체 상태는 유지 · 배터리 {:.0}%)", self.cfg.seed, self.world.battery_ratio() * 100.0);
        self.log("world", Level::Info, msg, Some(json!({"seed": self.cfg.seed, "kept_energy_j": self.world.agent.energy_used_j})));
        Ok(())
    }

    /// The robot's body rested: full battery, cold joints, no joint wear. The
    /// world, the robot's place and its cargo stay; a running quest goes on.
    pub fn reset_body(&mut self) {
        let (battery, strikes) = (self.world.battery_ratio(), self.world.agent.strikes);
        let worn = self.world.most_worn().map(|(j, d)| format!(", 가장 닳은 관절 {j} 수명 {:.2}% → 0", d * 100.0)).unwrap_or_default();
        self.world.reset_body();
        self.log("world", Level::Info, format!("신체 상태 리셋: 배터리 {:.0}% → 100%, 채취 타격 {strikes} → 0, 관절 발열 초기화{worn}", battery * 100.0), None);
    }

    /// Queues a quest for a plan the UI built itself (the inventory buttons),
    /// as `submit` does for a typed command. Returns the quest id.
    pub fn submit_plan(&mut self, plan: Plan, what: &str) -> std::result::Result<String, String> {
        let tick = self.world.tick;
        match self.quests.add(&plan, &WorldLabels(&self.world), tick) {
            Ok(id) => {
                let title = self.quests.get(&id).map(|q| q.title.clone()).unwrap_or_default();
                self.log("quest", Level::Info, format!("퀘스트 {id} 접수 ({what}): {title}"), Some(json!({"quest": id})));
                Ok(id)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    // ----- running -----

    /// One world tick (0.1 s) and one driver step.
    pub fn tick(&mut self) -> Vec<TimedEvent> {
        let events = self.world.tick();
        self.log_world_events(&events);
        self.drive(&events);
        self.record_balance_events();
        self.record_tick();
        self.record_act_keyframes(&events);
        self.record_act_sds(&events);
        if self.world.tick.saturating_sub(self.last_flush_tick) >= 600 {
            self.flush_sds();
        }
        events
    }

    /// `n` ticks. Returns the log lines written meanwhile.
    pub fn step(&mut self, n: u32) -> Vec<LogLine> {
        let before = self.audit.last_seq();
        for _ in 0..n {
            self.tick();
        }
        self.audit.tail(before, 500, None).0
    }

    /// Runs until no quest is left or `max_ticks` passed. Returns ticks run.
    pub fn run_until_idle(&mut self, max_ticks: u64) -> u64 {
        let mut n = 0;
        while n < max_ticks && self.busy() {
            self.tick();
            n += 1;
        }
        n
    }

    // ----- commands -----

    /// Parses a typed command and queues the quest it describes.
    pub fn submit(&mut self, text: &str) -> Submission {
        let emb = self.embedder.clone();
        let parsed = self.parser.parse(text, emb.as_deref().map(|e| e as &dyn PhraseEmbedder));
        let mode = if emb.is_some() { "lexical+embedding" } else { "lexical" }.to_string();
        let plan_json = parsed.plan.as_ref().and_then(|p| serde_json::to_value(p).ok());
        self.log("search", Level::Info, format!("명령: \"{text}\" → {}", plan_brief(&parsed)), Some(json!({"text": text, "plan": plan_json, "mode": mode})));
        for tl in &parsed.trace {
            self.log("search", Level::Debug, format!("{}: {}", tl.stage, tl.detail), None);
        }
        for w in &parsed.warnings {
            self.log("search", Level::Warn, w.clone(), None);
        }
        if let Some(Plan::Fetch { targets, .. }) | Some(Plan::Collect { targets }) = &parsed.plan {
            for t in targets {
                let scope = Scope::new(Track::Search, &t.resource, &format!("{:?}", t.via));
                self.sds.record_baseline(&scope, "score", t.score as f64);
            }
        }
        let mut out = Submission { ok: false, stopped: false, quest_id: None, error: None, error_code: None, mode, parsed };
        match out.parsed.plan.clone() {
            None => {
                let (code, msg) = out.parsed.problem.as_ref().map(|p| (p.code.clone(), p.message.clone())).unwrap_or(("unknown".into(), "명령을 이해하지 못했습니다".into()));
                self.log("search", Level::Warn, format!("해석 실패 [{code}]: {msg}"), None);
                out.error = Some(msg);
                out.error_code = Some(code);
            }
            Some(Plan::Stop) => {
                self.cancel();
                out.ok = true;
                out.stopped = true;
            }
            Some(plan) => {
                let tick = self.world.tick;
                match self.quests.add(&plan, &WorldLabels(&self.world), tick) {
                    Ok(id) => {
                        let title = self.quests.get(&id).map(|q| q.title.clone()).unwrap_or_default();
                        self.log("quest", Level::Info, format!("퀘스트 {id} 접수: {title}"), Some(json!({"quest": id})));
                        out.ok = true;
                        out.quest_id = Some(id);
                    }
                    Err(e) => {
                        out.error = Some(e.to_string());
                        out.error_code = Some("quest".into());
                    }
                }
            }
        }
        out
    }

    /// What [`Runtime::submit`] would answer, without queueing a quest (for a
    /// command that waits for the running recording to end).
    pub fn submit_preview(&self, text: &str) -> Submission {
        let emb = self.embedder.clone();
        let parsed = self.parser.parse(text, emb.as_deref().map(|e| e as &dyn PhraseEmbedder));
        let mode = if emb.is_some() { "lexical+embedding" } else { "lexical" }.to_string();
        let (error, error_code) = match (&parsed.plan, &parsed.problem) {
            (None, Some(p)) => (Some(p.message.clone()), Some(p.code.clone())),
            _ => (None, None),
        };
        Submission { ok: parsed.plan.is_some(), stopped: false, quest_id: None, error, error_code, mode, parsed }
    }

    // ----- looking -----

    /// Renders what the robot sees, recognises it, judges it. Returns the report.
    pub fn scan_now(&mut self, trigger: &str) -> Result<&ScanReport> {
        let t0 = Instant::now();
        let frame = self.world.render(&self.cam);
        let mut out = self.recognizer.scan(ScanInput { world: &self.world, frame: &frame, camera: &self.cam })?;
        let tick = self.world.tick;
        let oracle = self.recognizer.mode() == "oracle";
        let truths: Vec<String> = if oracle {
            self.world.visible_objects(&frame).into_iter().filter(|v| v.pixels >= self.cfg.min_pixels).map(|v| v.entity).collect()
        } else {
            Vec::new()
        };
        let mut cands = std::mem::take(&mut out.candidates);
        // The camera measures where things are from the robot; the robot puts
        // them in its own (believed) frame, which its odometry drifts.
        for c in cands.iter_mut() {
            c.world = c.world.map(|(x, z)| self.world.belief_of(x, z));
        }
        let mem = self.mem_ctx();
        let judged = self.gates.judge_with(cands, tick, &mut self.sds, &mem);
        self.relocalize_from(&judged);
        if oracle {
            self.eval.add(&truths, &judged);
        }
        self.note_look(trigger, String::new(), None, None, None);
        self.scan_seq += 1;
        let report = ScanReport {
            seq: self.scan_seq,
            tick,
            trigger: trigger.to_string(),
            mode: self.recognizer.mode().to_string(),
            device: out.device.clone(),
            tiles: out.tiles,
            embed_ms: out.embed_ms,
            total_ms: t0.elapsed().as_secs_f32() * 1000.0,
            judged,
            scene: out.scene,
        };
        self.note_scan(&report);
        self.last_scan = Some(report);
        Ok(self.last_scan.as_ref().expect("just stored"))
    }

    /// The robot's believed pose (its odometry): x, z, yaw.
    pub(crate) fn me(&self) -> (f32, f32, f32) {
        self.world.believed_pose()
    }

    /// Distance from the robot's believed position to a believed point.
    pub(crate) fn dist_me(&self, x: f32, z: f32) -> f32 {
        let (mx, mz, _) = self.me();
        (x - mx).hypot(z - mz)
    }

    pub(crate) fn base_id(&self) -> String {
        format!("{:?}", self.world.hand.base.kind).to_lowercase()
    }

    /// How fast the robot's memories spread per metre: what the SDS learned, else its odometry's specification.
    pub(crate) fn drift_rate(&self) -> DriftRate {
        DriftRate::learned(&self.sds, &self.base_id(), self.world.body.legged)
    }

    /// 1σ of the robot's own position against the map: its odometry's drift
    /// since the last fix, at the same rate as a memory's (Smith, Self &
    /// Cheeseman 1986, §3.2).
    pub(crate) fn pose_sigma_m(&self) -> f32 {
        let r = self.drift_rate();
        let d = self.world.odom.since_fix_m;
        ((r.per_m * d).powi(2) + r.walk_m * r.walk_m * d).sqrt()
    }

    /// What the gates need to keep landmarks as memories.
    pub(crate) fn mem_ctx(&self) -> MemCtx {
        let base = self.base_id();
        let floor = self.sds.adaptive_baseline(&Scope::new(Track::Memory, &base, "obs"), "scatter_var_m2").map_or(0.0, |b| (b.0 as f32).max(0.0).sqrt());
        MemCtx { odo_m: self.world.odom.travelled_m, rate: self.drift_rate(), base, camera: world_sim::world::camera_of(self.world.body.head.sensor), obs_floor_m: floor }
    }

    /// A waypoint seen where the map does not have it shows how far the
    /// robot's belief has drifted. The surest such sighting (the nearest flag)
    /// that passes the validation gate moves the believed pose by the Kalman
    /// share of the difference, the drift since the last fix against the
    /// sighting's own error (Matthies & Shafer 1987, Eqs. 9-10). The fix waits
    /// for the start of the next driver step and moves the memories with the
    /// pose there (`apply_fix`): what one step decides stays in one frame.
    fn relocalize_from(&mut self, judged: &[Judged]) {
        if self.pending_fix.is_some() {
            return;
        }
        let mem = self.mem_ctx();
        let p_pose = self.pose_sigma_m().powi(2);
        let mut best: Option<(f32, f32, f32, String, f32)> = None;
        for j in judged.iter().filter(|j| j.accepted && j.cand.kind == EntKind::Waypoint && j.cand.depth_m <= RELOCALIZE_RANGE_M) {
            let (Some((sx, sz)), Some(w)) = (j.cand.world, self.world.waypoint_by_key(&j.cand.link)) else { continue };
            let (ex, ez) = (w.x - sx, w.z - sz);
            let p_obs = mem.sigma_obs(j.cand.depth_m).powi(2);
            if ex * ex + ez * ez > FIX_GATE_CHI2 * (p_pose + p_obs) {
                let msg = format!("위치 보정 안 함: 지점 깃발 {}이 지도와 {:.2} m 달라 보임 — 예상 오차 밖 (잘못 본 깃발)", j.cand.label, ex.hypot(ez));
                self.log("vision", Level::Debug, msg, None);
                continue;
            }
            if best.as_ref().map_or(true, |b| p_obs < b.0) {
                best = Some((p_obs, ex, ez, j.cand.label.clone(), 0.0));
            }
        }
        let Some((p_obs, ex, ez, label, _)) = best else { return };
        let k = p_pose / (p_pose + p_obs).max(1e-9);
        if k * ex.hypot(ez) > 1e-3 {
            let odo = self.world.odom.travelled_m;
            self.pending_fix = Some(PoseFix { dx: k * ex, dz: k * ez, odo_m: odo, prev_odo_m: odo - self.world.odom.since_fix_m });
            let msg = format!("위치 보정: 지점 깃발 {label}이 지도와 {:.2} m 달라 보여 {:.0}%만큼 고침", ex.hypot(ez), k * 100.0);
            self.log("vision", Level::Debug, msg, None);
        }
    }

    /// The pending fix of the pose, applied: the believed pose moves, and so
    /// does every landmark, by the share of the drift it had taken in when it
    /// was last seen (`PoseFix::share`): the first-order update of a landmark
    /// whose error is correlated with the pose's (the stochastic map of Smith,
    /// Self & Cheeseman 1986). Returns the fix, for the task's own memories.
    pub(crate) fn apply_fix(&mut self) -> Option<PoseFix> {
        let f = self.pending_fix.take()?;
        self.world.relocalize(f.dx, f.dz);
        let rate = self.drift_rate();
        self.gates.shift_landmarks(|x, z, odo| f.shift(x, z, odo, rate));
        Some(f)
    }

    /// An action in the robot's believed frame as the world's: where its body
    /// goes when it goes where it believes that is (its odometry's drift is
    /// in the difference). Ids (a node, a piece, a waypoint) need nothing.
    pub fn to_world(&self, a: Action) -> Action {
        match a {
            Action::MoveTo { x, z, reach_m } => {
                let (x, z) = self.world.true_of(x, z);
                Action::MoveTo { x, z, reach_m }
            }
            Action::Face { x, z } => {
                let (x, z) = self.world.true_of(x, z);
                Action::Face { x, z }
            }
            Action::Turn { yaw_deg } => Action::Turn { yaw_deg: self.world.true_yaw_of(yaw_deg.to_radians()).to_degrees() },
            Action::Push { drop, x, z } => {
                let (x, z) = self.world.true_of(x, z);
                Action::Push { drop, x, z }
            }
            other => other,
        }
    }

    /// Notes a look (or an action taken on memory without one) for the timeline's vision lane.
    pub(crate) fn note_look(&mut self, kind: &str, what: String, sigma_before_m: Option<f32>, sigma_after_m: Option<f32>, innovation_m: Option<f32>) {
        let mark = VisionMark { tick: self.world.tick, kind: kind.to_string(), what, sigma_before_m, sigma_after_m, innovation_m };
        if let Some(rec) = self.recording.as_mut() {
            if rec.vision.len() < crate::record::MAX_VISION_MARKS {
                rec.vision.push(mark);
            }
        }
    }

    /// A look with the head pitched down by `tilt_deg` (the neck), to see the
    /// ground in front of the feet; the camera goes back level after.
    pub fn scan_tilted(&mut self, trigger: &str, tilt_deg: f32) -> Result<()> {
        let level = self.cam;
        self.cam.pitch_deg -= tilt_deg;
        let r = self.scan_now(trigger).map(|_| ());
        self.cam = level;
        r
    }

    /// On the way to a place of the map. Once the place is near enough for a
    /// sighting of its flag to fix the pose (`RELOCALIZE_RANGE_M`) and in
    /// view, the robot looks on the pattern while its pose is too unsure for
    /// the hand to find the pile there: 2σ of the drift since the last fix
    /// against the slack between where the move stops and the hand's reach.
    /// With the pattern off it looks as often as on an approach. Whether it looked.
    pub(crate) fn place_look(&mut self, cd: &mut u32, x: f32, z: f32) -> bool {
        if *cd > 0 {
            *cd -= 1;
            return false;
        }
        if self.dist_me(x, z) > RELOCALIZE_RANGE_M || !in_view(&self.cam, EYE_HEIGHT_M, self.me(), x, z) {
            return false;
        }
        let intermittent = self.cfg.vision.intermittent;
        if intermittent && memory_ok(self.pose_sigma_m(), DELIVER_REACH_M - crate::driver::PLACE_REACH_M, &self.cfg.vision) {
            return false;
        }
        *cd = if intermittent { self.cfg.vision.look_every_ticks } else { self.cfg.approach_scan_every_ticks };
        let looked = self.scan_now("place").is_ok();
        if looked {
            self.vstats.looks += 1;
        }
        looked
    }

    fn note_scan(&mut self, r: &ScanReport) {
        if r.device != self.last_device {
            self.last_device = r.device.clone();
            self.log("device", Level::Info, format!("인식 장치: {} (모드 {})", r.device, r.mode), None);
        }
        if r.embed_ms > 0.0 {
            self.log("embed", Level::Debug, format!("스캔 #{}: 이미지 조각 {}개 임베딩 {:.1} ms", r.seq, r.tiles, r.embed_ms), None);
        }
        let blocked: Vec<&Judged> = r.judged.iter().filter(|j| !j.accepted).collect();
        let accepted = r.judged.len() - blocked.len();
        let mut why: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
        for j in &blocked {
            if let Some(g) = j.blocked_by() {
                *why.entry(g).or_insert(0) += 1;
            }
        }
        let why_txt = if why.is_empty() { String::new() } else { format!(" (차단: {})", why.iter().map(|(g, n)| format!("{g}×{n}")).collect::<Vec<_>>().join(" ")) };
        let level = if r.judged.is_empty() { Level::Debug } else { Level::Info };
        let list: Vec<Value> = r
            .judged
            .iter()
            .take(12)
            .map(|j| json!({"label": j.cand.label, "score": j.cand.score, "margin": j.cand.margin, "accepted": j.accepted, "blocked_by": j.blocked_by(), "landmark": j.landmark}))
            .collect();
        self.log(
            "vision",
            level,
            format!("스캔 #{} [{}]: 후보 {}건 → 수락 {accepted}건{why_txt}", r.seq, r.trigger, r.judged.len()),
            Some(json!({"seq": r.seq, "tick": r.tick, "total_ms": r.total_ms, "candidates": list})),
        );
        let need = self.cfg.gates.confirm_sightings;
        for j in &r.judged {
            if let (true, Some(id)) = (j.accepted, j.landmark) {
                if self.gates.landmark(id).map_or(false, |l| l.good == need) {
                    if let Some(l) = self.gates.landmark(id) {
                        let msg = format!("랜드마크 L{id} 확인: {} ({}) @ ({:.1}, {:.1}) 점수 {:.2}", l.label, l.link, l.x, l.z, l.score);
                        let level = if self.focus.as_deref() == Some(l.link.as_str()) { Level::Info } else { Level::Debug };
                        self.audit.log("vision", level, msg, None);
                    }
                }
            }
            if let Some(v) = &j.temporal {
                if v.outcome == hand_ts::TemporalOutcome::Held {
                    let msg = format!("시계열 게이트가 {}의 점수 {:.2}를 보류: {}", j.cand.label, v.observed, v.reason);
                    self.audit.log(
                        "ts",
                        Level::Warn,
                        msg,
                        Some(tidy(json!({"label": j.cand.label, "observed": v.observed, "expected": v.expected, "lo": v.lo, "hi": v.hi, "method": v.method}))),
                    );
                }
            }
        }
    }

    // ----- views -----

    pub fn snapshot(&self) -> WorldSnapshot {
        self.world.snapshot()
    }

    pub fn body_status(&self) -> BodyStatus {
        let (conf, ms) = match &self.last_scan {
            Some(r) => (r.confidence(), Some(r.total_ms)),
            None => (None, None),
        };
        self.world.body_status(conf, ms)
    }

    pub fn sds_status(&self) -> SdsStatus {
        self.sds.status(80)
    }

    /// Score history and forecast of one label (the longest history if none is named).
    pub fn ts_view(&self, label: Option<&str>, horizon: usize) -> TsView {
        let labels = self.gates.series_labels();
        let label = label.map(String::from).or_else(|| labels.first().map(|l| l.0.clone())).unwrap_or_default();
        TsView { history: self.gates.series(&label).to_vec(), forecast: self.gates.forecast_for(&label, horizon, &self.sds), label, labels }
    }

    pub fn log_tail(&self, after_seq: u64, limit: usize, cat: Option<&str>) -> (Vec<LogLine>, u64) {
        self.audit.tail(after_seq, limit, cat)
    }

    /// A WebP screenshot of a view at the camera's own size.
    pub fn frame(&mut self, view: FrameView, quality: f32) -> Result<FrameOut> {
        let (w, h) = (self.cam.width as u32, self.cam.height as u32);
        self.frame_sized(view, w, h, quality)
    }

    /// A WebP screenshot of a view, with the last scan's boxes for the robot's
    /// camera. The boxes are fractions of the frame, so any size fits them.
    /// The top view is square and uses `height` for its side.
    pub fn frame_sized(&mut self, view: FrameView, width: u32, height: u32, quality: f32) -> Result<FrameOut> {
        let mut overlay = None;
        let (frame, detections) = match view {
            FrameView::Agent => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, ..self.cam };
                let f = self.world.render(&cam);
                let det = self
                    .last_scan
                    .as_ref()
                    .filter(|r| self.world.tick.saturating_sub(r.tick) <= 6)
                    .map(|r| {
                        r.judged
                            .iter()
                            .map(|j| Detection {
                                label: j.cand.label.clone(),
                                label_ko: label_ko(&self.world, &j.cand.label),
                                score: j.cand.score,
                                margin: j.cand.margin,
                                accepted: j.accepted,
                                blocked_by: j.blocked_by().map(String::from),
                                reason: j.reason_ko(),
                                bbox: j.cand.bbox,
                                landmark: j.landmark,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (f, det)
            }
            FrameView::Top => (self.world.render_top(height.clamp(32, 2048) as usize, 24.0), Vec::new()),
            FrameView::Chase { orbit_deg } => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, fov_deg: 64.0, max_dist_m: 30.0, ..self.cam };
                let (f, o) = crate::chase::render(&self.world, &cam, orbit_deg as f32);
                overlay = o;
                (f, Vec::new())
            }
            FrameView::Focus(which) => {
                let cam = Camera { width: width.clamp(32, 2048) as usize, height: height.clamp(32, 2048) as usize, fov_deg: 50.0, max_dist_m: 12.0, ..self.cam };
                (world_sim::render::render_focus(&self.world, &cam, which), Vec::new())
            }
        };
        let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
        Ok(FrameOut {
            width: enc.width,
            height: enc.height,
            bytes: enc.bytes.len(),
            encode_ms: enc.encode_ms,
            lossless: enc.lossless,
            tick: frame.tick,
            detections,
            overlay,
            webp: enc.bytes,
        })
    }
}

/// Rounds every fractional number in a log payload to four decimals. Scores
/// come from `f32`, and `0.98f32` as an `f64` prints as `0.9800000190734863`.
pub(crate) fn tidy(v: Value) -> Value {
    match v {
        Value::Number(n) => match n.as_f64() {
            Some(x) if n.is_f64() => serde_json::Number::from_f64((x * 1e4).round() / 1e4).map(Value::Number).unwrap_or(Value::Null),
            _ => Value::Number(n),
        },
        Value::Array(a) => Value::Array(a.into_iter().map(tidy).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, tidy(v))).collect()),
        other => other,
    }
}

fn plan_brief(p: &ParsedCommand) -> String {
    match &p.plan {
        Some(Plan::Fetch { targets, destination }) => format!(
            "가져오기 {} → {}",
            targets.iter().map(|t| format!("{}×{}", t.resource, t.quantity)).collect::<Vec<_>>().join(", "),
            destination.place
        ),
        Some(Plan::Collect { targets }) => format!("채취 {}", targets.iter().map(|t| format!("{}×{}", t.resource, t.quantity)).collect::<Vec<_>>().join(", ")),
        Some(Plan::Goto { place }) => format!("이동 {}", place.place),
        Some(Plan::Drop { items, place }) => format!(
            "내려놓기 {}{}",
            if items.is_empty() { "전부".to_string() } else { items.iter().map(|t| format!("{}×{}", t.resource, t.quantity.map_or("전부".to_string(), |q| q.to_string()))).collect::<Vec<_>>().join(", ") },
            place.as_ref().map(|p| format!(" @ {}", p.place)).unwrap_or_default()
        ),
        Some(Plan::Stop) => "중지".to_string(),
        None => p.problem.as_ref().map(|x| format!("해석 불가({})", x.code)).unwrap_or_else(|| "해석 불가".into()),
    }
}
