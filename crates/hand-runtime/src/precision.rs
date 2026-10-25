//! Precision gates between the recogniser and the robot's decisions.
//!
//! A detection is only acted on when it passes every gate that applies:
//!
//! 1. `score`: at least the floor. The floor is the fixed `min_score`, raised
//!    when the SDS has learned that touching objects of this label confirmed
//!    only high-scoring detections, and raised again while the label's touch
//!    agreement rate is low. It is never lowered below the fixed value.
//! 2. `margin`: the label beat its runner-up by enough.
//! 3. `temporal`: the score fits the forecast made from the earlier scores of
//!    the same landmark (Granite TTM when loaded, otherwise Holt, EWMA and
//!    persistence fused by their remembered errors). Needs enough history.
//! 4. `conflict`: no other label already holds the same spot with a better score.
//! 5. `touch`: the spot was not ruled out by touching it earlier.
//! 6. `confirm`: the landmark has been seen well enough several times.
//!
//! Every outcome is written to the SDS (`record_gates`) so the UI can show how
//! often each gate blocks and how often it was the only one that did.

use std::collections::HashMap;
use std::sync::Arc;

use hand_sds::{Scope, SdsStore, Track};
use hand_ts::ttm::TtmForecaster;
use hand_ts::{forecast, temporal_gate, Forecast, GateConfig, HistoryMse, Predictor, TemporalOutcome, TemporalVerdict, TtmPredictor};
use serde::Serialize;

use crate::config::GateCfg;
use crate::perception::{Candidate, EntKind};
use crate::vision::{DriftRate, Memory};

const CONFLICT_RADIUS_M: f32 = 0.4;
const STALE_TICKS: u64 = 400;
const MAX_LANDMARKS: usize = 256;
const SERIES_CAP: usize = 64;
const HISTORY_CAP: usize = 32;
const METHODS: [&str; 4] = ["last", "ewma", "holt", "ttm"];
/// Held scores in a row that agree within `SHIFT_SPREAD` count as a level change.
const SHIFT_RUN: usize = 3;
const SHIFT_SPREAD: f64 = 0.08;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LmState {
    Active,
    /// Touching the spot showed it was not this label.
    Rejected,
}

/// Something the robot believes is out there.
#[derive(Clone, Debug, Serialize)]
pub struct Landmark {
    pub id: u32,
    pub label: String,
    pub kind: EntKind,
    pub link: String,
    pub x: f32,
    pub z: f32,
    /// Smoothed score.
    pub score: f32,
    /// Smoothed margin over the runner-up.
    pub margin: f32,
    /// Sightings that passed the earlier gates.
    pub good: u32,
    pub seen: u32,
    pub first_tick: u64,
    pub last_tick: u64,
    pub confirmed: bool,
    pub state: LmState,
    /// Times the robot failed to reach it.
    pub unreachable: u32,
    pub history: Vec<f64>,
    /// Consecutive scores the temporal gate held. Three that agree with each
    /// other are taken as a real change of level (the robot got closer, the
    /// light changed) and replace the history.
    pub held_run: Vec<f64>,
    /// 1σ of the position when it was last fused, metres, and the odometer
    /// then: the landmark is a memory that spreads as the robot moves (`vision`).
    pub sigma_m: f32,
    pub odo_m: f32,
}

impl Landmark {
    pub fn distance_to(&self, x: f32, z: f32) -> f32 {
        ((self.x - x).powi(2) + (self.z - z).powi(2)).sqrt()
    }

    /// The landmark as a memory.
    pub fn memory(&self) -> Memory {
        Memory::seen(self.x, self.z, self.sigma_m, self.odo_m, self.last_tick)
    }
}

/// What the gates need to keep landmarks as memories (`vision`).
#[derive(Clone, Debug)]
pub struct MemCtx {
    /// The odometer now, metres.
    pub odo_m: f32,
    pub rate: DriftRate,
    /// The base, for the SDS scope of what re-sightings show.
    pub base: String,
    /// The head camera: focal length (px), stereo baseline (m), subpixel error.
    pub camera: (f32, f32, f32),
    /// The scatter re-sightings without travel showed (SDS), metres per axis; 0 before any.
    pub obs_floor_m: f32,
}

impl MemCtx {
    /// No motion and the D435's camera: sightings at one range weigh the same (as before v0.7).
    pub fn still() -> MemCtx {
        MemCtx { odo_m: 0.0, rate: DriftRate { per_m: 0.0, walk_m: 0.0, source: "spec" }, base: String::new(), camera: (424.0, 0.050, 0.08), obs_floor_m: 0.0 }
    }

    /// 1σ of a sighting at `depth_m`, metres per axis: the stereo depth law
    /// along the ray and one pixel across it, or what re-sightings showed.
    pub fn sigma_obs(&self, depth_m: f32) -> f32 {
        let (f, b, sub) = self.camera;
        let along = hand_core::act::material::stereo_depth_sigma(depth_m, f, b, sub);
        let across = depth_m / f.max(1.0);
        (along * along + across * across).sqrt().max(self.obs_floor_m).max(0.005)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GateResult {
    pub gate: &'static str,
    pub passed: bool,
    /// False when the gate had nothing to judge with (for example no history yet).
    pub applied: bool,
    pub detail: String,
    /// 0 (passed) to 100 (far off).
    pub severity: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub cand: Candidate,
    pub accepted: bool,
    pub gates: Vec<GateResult>,
    pub landmark: Option<u32>,
    pub floor: f32,
    pub floor_source: String,
    pub temporal: Option<TemporalVerdict>,
}

impl Judged {
    /// One line on why the detection was accepted or held.
    pub fn reason_ko(&self) -> String {
        match self.gates.iter().find(|g| g.applied && !g.passed) {
            Some(g) => format!("보류 · {} 게이트: {}", gate_ko(g.gate), g.detail),
            None => format!("수락 · 점수 {:.2} ≥ 바닥 {:.2} ({})", self.cand.score, self.floor, self.floor_source),
        }
    }

    /// Name of the first gate that blocked, if any.
    pub fn blocked_by(&self) -> Option<&'static str> {
        self.gates.iter().find(|g| g.applied && !g.passed).map(|g| g.gate)
    }

    /// The detection itself holds (its score and its margin pass), whatever
    /// the gates that need a history said (repeated sightings, the score's
    /// time series, a landmark of another label nearby): enough to find again
    /// a piece the robot already knows is there, not to believe a new one.
    pub fn sound(&self) -> bool {
        self.gates.iter().filter(|g| g.applied && !g.passed).all(|g| matches!(g.gate, "confirm" | "temporal" | "conflict"))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GateCounters {
    pub evaluated: u64,
    pub accepted: u64,
    pub blocked_by: HashMap<String, u64>,
}

pub struct PrecisionGates {
    cfg: GateCfg,
    next_id: u32,
    landmarks: Vec<Landmark>,
    series: HashMap<String, Vec<f64>>,
    /// Margin over the runner-up, kept next to `series` so the UI can chart both.
    margins: HashMap<String, Vec<f64>>,
    /// Granite TTM, when its weights are loaded. It joins the fusion as one more predictor.
    ttm: Option<Arc<TtmForecaster>>,
    pub counters: GateCounters,
}

/// Runs `f` with the extra predictors that are available (Granite TTM, if loaded).
fn with_extra<T>(ttm: &Option<Arc<TtmForecaster>>, f: impl FnOnce(&[&dyn Predictor]) -> T) -> T {
    match ttm {
        Some(t) => {
            let p = TtmPredictor(t.as_ref());
            f(&[&p])
        }
        None => f(&[]),
    }
}

/// Korean name of a gate.
pub fn gate_ko(gate: &str) -> &str {
    match gate {
        "score" => "점수",
        "margin" => "여유폭",
        "temporal" => "시계열",
        "conflict" => "라벨 충돌",
        "touch" => "접촉 확인",
        "confirm" => "반복 관측",
        other => other,
    }
}

fn severity(shortfall: f64, scale: f64) -> u8 {
    ((shortfall / scale.max(1e-6)).clamp(0.0, 1.0) * 100.0).round() as u8
}

/// Remembered per-method forecast errors of a label, from the SDS.
pub fn past_errors(sds: &SdsStore, label: &str) -> Vec<HistoryMse> {
    let scope = Scope::new(Track::Forecast, label, "");
    METHODS.iter().filter_map(|m| sds.fusion_mse(&scope, m).map(|(mse, n)| HistoryMse { method: (*m).to_string(), mse, n })).collect()
}

/// What a re-sighting says about the robot's memory (SDS `memory` track).
/// The squared 2-D innovation is, on average, the two sightings' scatter in
/// both axes (`2·σ0² + 2·σ_obs²`), the random walk in both (`2·w²·d`) and the
/// odometer's one-dimensional scale drift `(r·d)²` (the memory's model,
/// `vision::Memory::sigma_m`). So a sighting after a walk of `d` (far enough
/// to tell drift from scatter) gives an unbiased sample of `r²`, and one
/// without travel a sample of the camera's per-axis variance; both are kept
/// as variances (a square root per sighting would bias the mean low), and the
/// innovation over the spread the memory predicted is about 1 when the model is right.
fn record_memory(sds: &mut SdsStore, mem: &MemCtx, label: &str, travelled: f32, innov: f32, predicted: f32, sigma0: f32, sigma_obs: f32) {
    if mem.base.is_empty() {
        return;
    }
    let sq = innov * innov;
    let scatter = 2.0 * (sigma0 * sigma0 + sigma_obs * sigma_obs);
    let walk_var = mem.rate.walk_m * mem.rate.walk_m * travelled.max(0.0);
    if travelled >= 2.0 {
        sds.record_baseline(&Scope::new(Track::Memory, &mem.base, "drift"), "drift_var_per_m2", ((sq - scatter - 2.0 * walk_var) / (travelled * travelled)) as f64);
    } else if travelled < 0.2 {
        sds.record_baseline(&Scope::new(Track::Memory, &mem.base, "obs"), "scatter_var_m2", (0.5 * sq - sigma0 * sigma0) as f64);
    }
    // The spread the memory predicted, as the same sum (its σ already holds σ0, one walk and the drift).
    let spread = (sigma0 * sigma0 + predicted * predicted + 2.0 * sigma_obs * sigma_obs + walk_var).sqrt().max(1e-4);
    sds.record_baseline(&Scope::new(Track::Memory, label, &mem.base), "innovation_z", (innov / spread) as f64);
}

impl PrecisionGates {
    pub fn new(cfg: GateCfg) -> PrecisionGates {
        PrecisionGates { cfg, next_id: 1, landmarks: Vec::new(), series: HashMap::new(), margins: HashMap::new(), ttm: None, counters: GateCounters::default() }
    }

    pub fn cfg(&self) -> &GateCfg {
        &self.cfg
    }

    pub fn set_cfg(&mut self, cfg: GateCfg) {
        self.cfg = cfg;
    }

    /// Lets the forecasts use Granite TTM (or stop using it with `None`).
    pub fn set_ttm(&mut self, ttm: Option<Arc<TtmForecaster>>) {
        self.ttm = ttm;
    }

    /// The attached Granite TTM, for other forecasts (the ML decision) to use.
    pub fn ttm(&self) -> Option<Arc<TtmForecaster>> {
        self.ttm.clone()
    }

    pub fn has_ttm(&self) -> bool {
        self.ttm.is_some()
    }

    pub fn landmarks(&self) -> &[Landmark] {
        &self.landmarks
    }

    /// Moves every landmark by `f` (from its position and its odometer: a fix of the robot's pose).
    pub fn shift_landmarks(&mut self, f: impl Fn(f32, f32, f32) -> (f32, f32)) {
        for l in self.landmarks.iter_mut() {
            (l.x, l.z) = f(l.x, l.z, l.odo_m);
        }
    }

    pub fn landmark(&self, id: u32) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }

    pub fn clear(&mut self) {
        self.landmarks.clear();
        self.series.clear();
        self.margins.clear();
        self.counters = GateCounters::default();
    }

    /// The landmark is gone (the robot broke the node).
    pub fn consume(&mut self, id: u32) {
        self.landmarks.retain(|l| l.id != id);
    }

    /// Touch showed the spot is not this label.
    pub fn reject(&mut self, id: u32) {
        if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
            l.state = LmState::Rejected;
        }
    }

    /// The robot could not get there. Two failures rule the landmark out.
    pub fn note_unreachable(&mut self, id: u32) {
        if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
            l.unreachable += 1;
            if l.unreachable >= 2 {
                l.state = LmState::Rejected;
            }
        }
    }

    /// Score history of a label, oldest first.
    pub fn series(&self, label: &str) -> &[f64] {
        self.series.get(label).map_or(&[], Vec::as_slice)
    }

    /// Margin history of a label, oldest first.
    pub fn margin_series(&self, label: &str) -> &[f64] {
        self.margins.get(label).map_or(&[], Vec::as_slice)
    }

    /// Forecast of the next margins. No remembered errors: the SDS keeps those
    /// for scores only.
    pub fn forecast_margin(&self, label: &str, horizon: usize) -> Option<Forecast> {
        let m = self.margins.get(label)?;
        if m.is_empty() {
            return None;
        }
        Some(with_extra(&self.ttm, |extra| forecast(m, horizon, extra, &[])))
    }

    /// Labels that have a score history, longest first.
    pub fn series_labels(&self) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.series.iter().map(|(k, s)| (k.clone(), s.len())).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }

    /// Forecast of the next scores of a label, with the SDS's remembered errors.
    pub fn forecast_for(&self, label: &str, horizon: usize, sds: &SdsStore) -> Option<Forecast> {
        let s = self.series.get(label)?;
        if s.is_empty() {
            return None;
        }
        let past = past_errors(sds, label);
        Some(with_extra(&self.ttm, |extra| forecast(s, horizon, extra, &past)))
    }

    /// Score floor for a label and where it came from.
    pub fn floor_for(&self, label: &str, sds: &SdsStore) -> (f32, String) {
        let fixed = self.cfg.min_score as f64;
        let mut floor = fixed;
        let mut source = format!("고정 {fixed:.2}");
        if self.cfg.adaptive {
            let scope = Scope::new(Track::Vision, label, "");
            if let Some((mean, sd, n)) = sds.adaptive_baseline(&scope, "score") {
                let learned = (mean - self.cfg.adaptive_k * sd).min(fixed + 0.25);
                if learned > floor {
                    floor = learned;
                    source = format!("SDS 학습 {learned:.2} (접촉 확인 {n}건, 평균 {mean:.2}±{sd:.2})");
                }
            }
            if let Some((rate, n)) = sds.agreement_rate(&scope, "touch") {
                if rate < 0.8 {
                    let extra = (0.8 - rate) * 0.5;
                    floor += extra;
                    source = format!("{source} + 접촉 일치율 {:.0}% ({n}건)로 +{extra:.2}", rate * 100.0);
                }
            }
        }
        (floor.min(0.95) as f32, source)
    }

    fn associate(&mut self, c: &Candidate, x: f32, z: f32, tick: u64) -> u32 {
        let r = self.cfg.track_radius_m;
        let best = self
            .landmarks
            .iter()
            .filter(|l| l.label == c.label && l.kind == c.kind)
            .map(|l| (l.id, l.distance_to(x, z)))
            .filter(|(_, d)| *d <= r)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((id, _)) = best {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.landmarks.push(Landmark {
            id,
            label: c.label.clone(),
            kind: c.kind,
            link: c.link.clone(),
            x,
            z,
            score: c.score,
            margin: c.margin,
            good: 0,
            seen: 0,
            first_tick: tick,
            last_tick: tick,
            confirmed: false,
            state: LmState::Active,
            unreachable: 0,
            history: Vec::new(),
            held_run: Vec::new(),
            sigma_m: 0.0,
            odo_m: 0.0,
        });
        id
    }

    fn prune(&mut self, tick: u64) {
        self.landmarks.retain(|l| l.confirmed || l.state == LmState::Rejected || tick.saturating_sub(l.last_tick) <= STALE_TICKS);
        if self.landmarks.len() > MAX_LANDMARKS {
            self.landmarks.sort_by(|a, b| b.last_tick.cmp(&a.last_tick));
            self.landmarks.truncate(MAX_LANDMARKS);
        }
    }

    /// Judges the candidates of one scan, in order, with the robot standing still.
    pub fn judge(&mut self, cands: Vec<Candidate>, tick: u64, sds: &mut SdsStore) -> Vec<Judged> {
        self.judge_with(cands, tick, sds, &MemCtx::still())
    }

    /// Judges the candidates of one scan; a landmark seen again is fused with
    /// its memory by their variances (`mem`: how far the robot has moved since).
    pub fn judge_with(&mut self, cands: Vec<Candidate>, tick: u64, sds: &mut SdsStore, mem: &MemCtx) -> Vec<Judged> {
        self.prune(tick);
        cands.into_iter().map(|c| self.judge_one(c, tick, sds, mem)).collect()
    }

    fn judge_one(&mut self, c: Candidate, tick: u64, sds: &mut SdsStore, mem: &MemCtx) -> Judged {
        let vscope = Scope::new(Track::Vision, &c.label, "");
        sds.record_baseline(&vscope, "score_all", c.score as f64);
        sds.record_baseline(&vscope, "margin", c.margin as f64);

        let mut gates: Vec<GateResult> = Vec::new();

        // 1. score
        let (floor, floor_source) = self.floor_for(&c.label, sds);
        let score_ok = c.score >= floor;
        gates.push(GateResult {
            gate: "score",
            passed: score_ok,
            applied: true,
            severity: if score_ok { 0 } else { severity((floor - c.score) as f64, floor as f64) },
            detail: format!("점수 {:.2} {} 하한 {floor:.2} ({floor_source})", c.score, if score_ok { "≥" } else { "<" }),
        });

        // 2. margin
        let margin_ok = c.margin >= self.cfg.min_margin;
        gates.push(GateResult {
            gate: "margin",
            passed: margin_ok,
            applied: true,
            severity: if margin_ok { 0 } else { severity((self.cfg.min_margin - c.margin) as f64, self.cfg.min_margin as f64) },
            detail: format!(
                "차점{} 대비 여유 {:.2} {} {:.2}",
                c.runner_up.as_ref().map(|r| format!("({r})")).unwrap_or_default(),
                c.margin,
                if margin_ok { "≥" } else { "<" },
                self.cfg.min_margin
            ),
        });
        let plausible = score_ok && margin_ok;

        // The spot as a landmark.
        let tid = c.world.map(|(x, z)| self.associate(&c, x, z, tick));
        let state = tid.and_then(|id| self.landmark(id)).map(|l| l.state);

        // 3. temporal
        let mut temporal = None;
        match tid {
            Some(id) => {
                let hist = self.landmark(id).map(|l| l.history.clone()).unwrap_or_default();
                let need = self.cfg.temporal_min_history;
                if plausible && hist.len() >= need {
                    let past = past_errors(sds, &c.label);
                    let sd = sds.adaptive_baseline(&vscope, "score_all").map(|b| b.1);
                    let gcfg = GateConfig { widen_sd: self.cfg.temporal_widen_sd, min_history: need };
                    let (v, f) = with_extra(&self.ttm, |extra| (temporal_gate(&hist, c.score as f64, sd, extra, &past, &gcfg), forecast(&hist, 1, extra, &past)));
                    let fscope = Scope::new(Track::Forecast, &c.label, "");
                    for w in &f.weights {
                        sds.record_baseline(&fscope, &format!("fusion_mse:{}", w.method), w.rmse * w.rmse);
                    }
                    let mut held = v.outcome == TemporalOutcome::Held;
                    let mut detail = format!("{} · {}", v.method, v.reason);
                    if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                        if held {
                            l.held_run.push(c.score as f64);
                            let run = &l.held_run;
                            let spread = run.iter().cloned().fold(f64::MIN, f64::max) - run.iter().cloned().fold(f64::MAX, f64::min);
                            if run.len() >= SHIFT_RUN && spread <= SHIFT_SPREAD {
                                l.history = std::mem::take(&mut l.held_run);
                                held = false;
                                detail = format!("{detail} → 일치하는 {SHIFT_RUN}회가 이어져 수준 변화로 받아들임");
                            }
                        } else {
                            l.held_run.clear();
                        }
                    }
                    gates.push(GateResult {
                        gate: "temporal",
                        passed: !held,
                        applied: v.outcome != TemporalOutcome::Abstained,
                        severity: if held { v.z.map_or(50, |z| severity(z.abs() - 1.0, 3.0)) } else { 0 },
                        detail,
                    });
                    temporal = Some(v);
                } else {
                    gates.push(GateResult {
                        gate: "temporal",
                        passed: true,
                        applied: false,
                        severity: 0,
                        detail: format!("이력 {}점 (판정에 {need}점 필요)", hist.len()),
                    });
                }
            }
            None => gates.push(GateResult { gate: "temporal", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 건너뜀".into() }),
        }

        // 4. conflict
        if let Some((x, z)) = c.world {
            let rival = self
                .landmarks
                .iter()
                .filter(|l| l.state == LmState::Active && l.kind == c.kind && l.label != c.label && l.good >= 1 && l.distance_to(x, z) <= CONFLICT_RADIUS_M)
                .max_by(|a, b| a.score.total_cmp(&b.score));
            match rival {
                Some(r) if r.score >= c.score => gates.push(GateResult {
                    gate: "conflict",
                    passed: false,
                    applied: true,
                    severity: severity((r.score - c.score) as f64, 0.5),
                    detail: format!("같은 자리에 '{}'(점수 {:.2})가 더 확실함", r.label, r.score),
                }),
                _ => gates.push(GateResult { gate: "conflict", passed: true, applied: true, severity: 0, detail: "겹치는 다른 라벨 없음".into() }),
            }
        } else {
            gates.push(GateResult { gate: "conflict", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 건너뜀".into() });
        }

        // 5. touch
        let ruled_out = state == Some(LmState::Rejected);
        gates.push(GateResult {
            gate: "touch",
            passed: !ruled_out,
            applied: tid.is_some(),
            severity: if ruled_out { 100 } else { 0 },
            detail: if ruled_out { "이 자리는 접촉으로 아니라고 확인됨".into() } else { "접촉으로 배제된 적 없음".into() },
        });

        let base_ok = gates.iter().all(|g| g.passed);
        let temporal_ok = gates.iter().find(|g| g.gate == "temporal").map_or(true, |g| g.passed);
        let record_ok = plausible && temporal_ok;

        // Update the landmark before the confirmation gate reads it.
        if let Some(id) = tid {
            let (x, z) = c.world.unwrap_or((0.0, 0.0));
            let sigma_obs = mem.sigma_obs(c.depth_m);
            if let Some(l) = self.landmarks.iter_mut().find(|l| l.id == id) {
                let first = l.seen == 0;
                l.seen += 1;
                if record_ok {
                    l.history.push(c.score as f64);
                    if l.history.len() > HISTORY_CAP {
                        l.history.remove(0);
                    }
                    l.score = if first { c.score } else { 0.5 * l.score + 0.5 * c.score };
                    l.margin = if first { c.margin } else { 0.5 * l.margin + 0.5 * c.margin };
                    if first || l.sigma_m <= 0.0 {
                        l.x = x;
                        l.z = z;
                        l.sigma_m = sigma_obs;
                    } else {
                        // The memory and the new sighting, each by its variance: after a walk the sighting counts for more.
                        let before = l.memory();
                        let travelled = mem.odo_m - before.odo_m;
                        let predicted = before.sigma_m(mem.odo_m, mem.rate);
                        let (fused, innov) = before.fuse(x, z, sigma_obs, mem.odo_m, tick, mem.rate);
                        record_memory(sds, mem, &c.label, travelled, innov, predicted, before.sigma0_m, sigma_obs);
                        l.x = fused.x;
                        l.z = fused.z;
                        // The camera's own bias does not average away (est: at most four sightings' worth).
                        l.sigma_m = fused.sigma0_m.max(0.5 * sigma_obs);
                    }
                    l.odo_m = mem.odo_m;
                }
                l.last_tick = tick;
                if base_ok {
                    l.good += 1;
                }
            }
            if record_ok {
                let s = self.series.entry(c.label.clone()).or_default();
                s.push(c.score as f64);
                if s.len() > SERIES_CAP {
                    s.remove(0);
                }
                let m = self.margins.entry(c.label.clone()).or_default();
                m.push(c.margin as f64);
                if m.len() > SERIES_CAP {
                    m.remove(0);
                }
            }
        }

        // 6. confirm
        let need = self.cfg.confirm_sightings;
        match tid.and_then(|id| self.landmark(id)) {
            Some(l) => {
                let ok = l.good >= need;
                gates.push(GateResult {
                    gate: "confirm",
                    passed: ok,
                    applied: true,
                    severity: if ok { 0 } else { severity((need - l.good.min(need)) as f64, need as f64) },
                    detail: format!("통과한 관측 {}/{need}", l.good),
                });
            }
            None => gates.push(GateResult { gate: "confirm", passed: true, applied: false, severity: 0, detail: "위치를 알 수 없어 확인 불가".into() }),
        }

        let by_gates = gates.iter().all(|g| g.passed) && tid.is_some();
        let accepted = if self.cfg.enabled { by_gates } else { tid.is_some() };
        if accepted {
            if let Some(l) = tid.and_then(|id| self.landmarks.iter_mut().find(|l| l.id == id)) {
                l.confirmed = true;
            }
        }

        // Bookkeeping: counters and SDS gate record.
        self.counters.evaluated += 1;
        if accepted {
            self.counters.accepted += 1;
        }
        let failed = gates.iter().filter(|g| g.applied && !g.passed).count();
        if let Some(first) = gates.iter().find(|g| g.applied && !g.passed) {
            *self.counters.blocked_by.entry(first.gate.to_string()).or_insert(0) += 1;
        }
        let items: Vec<(String, u8, bool, bool)> =
            gates.iter().filter(|g| g.applied).map(|g| (g.gate.to_string(), g.severity, !g.passed, !g.passed && failed == 1)).collect();
        sds.record_gates(&vscope, &items);

        Judged { cand: c, accepted, gates, landmark: tid, floor, floor_source, temporal }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn cand(label: &str, score: f32, margin: f32, x: f32, z: f32) -> Candidate {
        let (kind, link) = match label {
            "tree" => (EntKind::Source, "log"),
            "boulder" => (EntKind::Source, "stone"),
            "coal_ore" => (EntKind::Source, "coal"),
            "log" => (EntKind::Drop, "log"),
            "warehouse" => (EntKind::Waypoint, "warehouse"),
            _ => (EntKind::Source, "x"),
        };
        Candidate {
            label: label.into(),
            kind,
            link: link.into(),
            score,
            margin,
            runner_up: None,
            bbox: [0.4, 0.4, 0.6, 0.6],
            pixels: 100,
            depth_m: 3.0,
            world: Some((x, z)),
            truth: None,
            truth_label: None,
        }
    }

    fn judge1(g: &mut PrecisionGates, sds: &mut SdsStore, c: Candidate, tick: u64) -> Judged {
        g.judge(vec![c], tick, sds).pop().unwrap()
    }

    #[test]
    fn a_landmark_needs_two_good_sightings() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let a = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1);
        assert!(!a.accepted);
        assert_eq!(a.blocked_by(), Some("confirm"));
        let b = judge1(&mut g, &mut sds, cand("tree", 0.82, 0.2, 5.1, 5.0), 3);
        assert!(b.accepted, "{:?}", b.gates);
        assert_eq!(g.landmarks().len(), 1, "both sightings are one landmark");
        let lm = &g.landmarks()[0];
        assert!(lm.confirmed && lm.good == 2);
        assert!((lm.x - 5.05).abs() < 0.01, "position is averaged: {}", lm.x);
    }

    #[test]
    fn low_score_and_low_margin_are_blocked_and_do_not_confirm() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..4 {
            let low = judge1(&mut g, &mut sds, cand("tree", 0.2, 0.2, 5.0, 5.0), t);
            assert_eq!(low.blocked_by(), Some("score"));
            assert!(!low.accepted);
        }
        for t in 4..8 {
            let thin = judge1(&mut g, &mut sds, cand("boulder", 0.7, 0.01, 8.0, 5.0), t);
            assert_eq!(thin.blocked_by(), Some("margin"));
        }
        assert!(g.landmarks().iter().all(|l| !l.confirmed && l.good == 0));
        assert!(g.series("tree").is_empty(), "implausible scores do not enter the forecast series");
    }

    #[test]
    fn the_sds_can_raise_the_floor_but_never_lower_it() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        assert_eq!(g.floor_for("tree", &sds).0, 0.30);
        let scope = Scope::new(Track::Vision, "tree", "");
        for i in 0..40 {
            sds.record_baseline(&scope, "score", 0.85 + 0.01 * ((i % 3) as f64 - 1.0));
        }
        let (floor, src) = g.floor_for("tree", &sds);
        // mean 0.85 - 2 sd would be about 0.83, but the learned floor is capped 0.25 above the fixed one
        assert!((floor - 0.55).abs() < 1e-6, "{floor}");
        assert!(src.contains("SDS"), "{src}");
        let weak = judge1(&mut g, &mut sds, cand("tree", 0.50, 0.2, 5.0, 5.0), 1);
        assert_eq!(weak.blocked_by(), Some("score"), "0.50 passes the fixed floor but not the learned one");
        let fine = judge1(&mut g, &mut sds, cand("tree", 0.70, 0.2, 5.0, 5.0), 2);
        assert!(fine.gates.iter().find(|x| x.gate == "score").unwrap().passed);
        // low scores learned (mean 0.20): the learned bound is below the fixed floor, which stays
        let mut sds2 = SdsStore::memory();
        for i in 0..40 {
            sds2.record_baseline(&scope, "score", 0.20 + 0.02 * ((i % 3) as f64 - 1.0));
        }
        assert_eq!(g.floor_for("tree", &sds2).0, 0.30);
        // adaptive off ignores the SDS
        g.set_cfg(GateCfg { adaptive: false, ..GateCfg::default() });
        assert_eq!(g.floor_for("tree", &sds).0, 0.30);
    }

    #[test]
    fn a_low_touch_agreement_rate_tightens_the_floor() {
        let g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let scope = Scope::new(Track::Vision, "boulder", "");
        for i in 0..20 {
            sds.record_agreement(&scope, "touch", i % 2 == 0);
        }
        let (floor, src) = g.floor_for("boulder", &sds);
        assert!(floor > 0.30 + 0.1, "agreement 50% should add (0.8-0.5)*0.5=0.15, got {floor}");
        assert!(src.contains("접촉 일치율"), "{src}");
    }

    #[test]
    fn the_temporal_gate_holds_a_jump_once_there_is_history() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            let s = 0.80 + 0.02 * ((t % 3) as f32 - 1.0);
            let j = judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
            let temporal = j.gates.iter().find(|x| x.gate == "temporal").unwrap();
            if t < 6 {
                assert!(!temporal.applied, "no history yet at {t}");
            }
        }
        let jump = judge1(&mut g, &mut sds, cand("tree", 0.40, 0.2, 5.0, 5.0), 11);
        let t = jump.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(t.applied && !t.passed, "{:?}", jump.gates);
        assert!(!jump.accepted);
        assert!(jump.temporal.as_ref().unwrap().expected.unwrap() > 0.7);
        let steady = judge1(&mut g, &mut sds, cand("tree", 0.81, 0.2, 5.0, 5.0), 12);
        assert!(steady.accepted, "{:?}", steady.gates);
        assert!(g.landmark(steady.landmark.unwrap()).unwrap().history.iter().all(|v| *v > 0.7), "the held score did not enter the history");
        assert!(g.series("tree").iter().all(|v| *v > 0.7), "nor the label series");
    }

    #[test]
    fn three_agreeing_held_scores_count_as_a_level_change() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..10u64 {
            judge1(&mut g, &mut sds, cand("tree", 0.80 + 0.02 * ((t % 3) as f32 - 1.0), 0.2, 5.0, 5.0), t);
        }
        // the robot's view changed and the score now sits near 0.45
        let a = judge1(&mut g, &mut sds, cand("tree", 0.45, 0.2, 5.0, 5.0), 20);
        let b = judge1(&mut g, &mut sds, cand("tree", 0.46, 0.2, 5.0, 5.0), 21);
        assert!(!a.accepted && !b.accepted, "held at first");
        let c = judge1(&mut g, &mut sds, cand("tree", 0.44, 0.2, 5.0, 5.0), 22);
        let tg = c.gates.iter().find(|x| x.gate == "temporal").unwrap();
        assert!(tg.passed && tg.detail.contains("수준 변화"), "{:?}", tg);
        let lm = g.landmark(c.landmark.unwrap()).unwrap();
        assert_eq!(lm.history.len(), 4, "the three held scores replaced the history, then the current one was added: {:?}", lm.history);
    }

    #[test]
    fn forecast_errors_flow_into_the_sds() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..40 {
            let s = 0.80 + 0.02 * (((t * 7) % 5) as f32 - 2.0) / 2.0;
            judge1(&mut g, &mut sds, cand("tree", s, 0.2, 5.0, 5.0), t);
        }
        let past = past_errors(&sds, "tree");
        assert!(!past.is_empty(), "after many judged sightings the SDS remembers method errors");
        assert!(past.iter().all(|h| h.mse.is_finite() && h.mse >= 0.0 && h.n > 0));
        let f = g.forecast_for("tree", 3, &sds).unwrap();
        assert_eq!(f.point.len(), 3);
        assert!(f.point[0] > 0.7 && f.point[0] < 0.9, "{:?}", f.point);
        assert!(f.p10[0] <= f.point[0] && f.point[0] <= f.p90[0]);
    }

    #[test]
    fn a_weaker_label_on_the_same_spot_is_blocked_as_a_conflict() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("boulder", 0.85, 0.2, 6.0, 6.0), 1);
        judge1(&mut g, &mut sds, cand("boulder", 0.86, 0.2, 6.0, 6.0), 2);
        let alt = judge1(&mut g, &mut sds, cand("coal_ore", 0.60, 0.2, 6.1, 6.0), 3);
        assert_eq!(alt.blocked_by(), Some("conflict"), "{:?}", alt.gates);
        // a distant object of the other label is fine
        let far = judge1(&mut g, &mut sds, cand("coal_ore", 0.60, 0.2, 9.0, 6.0), 4);
        assert!(far.gates.iter().find(|x| x.gate == "conflict").unwrap().passed);
        // and a clearly stronger label on the spot is not blocked
        let strong = judge1(&mut g, &mut sds, cand("coal_ore", 0.95, 0.3, 6.1, 6.0), 5);
        assert!(strong.gates.iter().find(|x| x.gate == "conflict").unwrap().passed);
    }

    #[test]
    fn a_rejected_spot_blocks_the_same_label_but_not_other_labels() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1);
        let id = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 2).landmark.unwrap();
        g.reject(id);
        let again = judge1(&mut g, &mut sds, cand("tree", 0.9, 0.3, 5.1, 5.0), 3);
        assert_eq!(again.blocked_by(), Some("touch"));
        assert!(!again.accepted);
        let other = judge1(&mut g, &mut sds, cand("boulder", 0.9, 0.3, 5.1, 5.0), 4);
        assert!(other.gates.iter().find(|x| x.gate == "touch").unwrap().passed);
        g.consume(id);
        assert!(g.landmark(id).is_none());
    }

    #[test]
    fn two_failed_approaches_rule_a_landmark_out() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        let id = judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1).landmark.unwrap();
        g.note_unreachable(id);
        assert_eq!(g.landmark(id).unwrap().state, LmState::Active);
        g.note_unreachable(id);
        assert_eq!(g.landmark(id).unwrap().state, LmState::Rejected);
    }

    #[test]
    fn with_the_gates_off_everything_with_a_position_is_accepted() {
        let mut g = PrecisionGates::new(GateCfg { enabled: false, ..GateCfg::default() });
        let mut sds = SdsStore::memory();
        let j = judge1(&mut g, &mut sds, cand("tree", 0.05, 0.0, 5.0, 5.0), 1);
        assert!(j.accepted);
        assert!(g.landmarks()[0].confirmed);
        let mut c = cand("tree", 0.9, 0.3, 0.0, 0.0);
        c.world = None;
        assert!(!judge1(&mut g, &mut sds, c, 2).accepted, "without a position there is nothing to act on");
    }

    #[test]
    fn unconfirmed_phantoms_expire_but_confirmed_landmarks_stay() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        judge1(&mut g, &mut sds, cand("tree", 0.8, 0.2, 5.0, 5.0), 1); // one sighting only
        judge1(&mut g, &mut sds, cand("boulder", 0.8, 0.2, 9.0, 5.0), 1);
        judge1(&mut g, &mut sds, cand("boulder", 0.8, 0.2, 9.0, 5.0), 2); // confirmed
        assert_eq!(g.landmarks().len(), 2);
        judge1(&mut g, &mut sds, cand("log", 0.8, 0.2, 2.0, 2.0), 2 + STALE_TICKS + 5);
        let labels: Vec<&str> = g.landmarks().iter().map(|l| l.label.as_str()).collect();
        assert!(labels.contains(&"boulder") && !labels.contains(&"tree"), "{labels:?}");
    }

    #[test]
    fn every_applied_gate_is_recorded_in_the_sds() {
        let mut g = PrecisionGates::new(GateCfg::default());
        let mut sds = SdsStore::memory();
        for t in 0..6 {
            judge1(&mut g, &mut sds, cand("tree", if t % 2 == 0 { 0.8 } else { 0.1 }, 0.2, 5.0, 5.0), t);
        }
        let rates = sds.gate_rates(&Scope::new(Track::Vision, "tree", ""));
        let score = rates.iter().find(|r| r.gate == "score").expect("score gate recorded");
        assert_eq!(score.evaluated, 6);
        assert!((score.blocked_rate - 0.5).abs() < 1e-9, "{}", score.blocked_rate);
        assert!(g.counters.evaluated == 6 && g.counters.blocked_by.contains_key("score"));
    }
}
