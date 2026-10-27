//! The action N:N benchmark: the four actions — grasp & release, push,
//! walk, bend — played out by `crate::act` for a robot, and every answer
//! compared with a measurement from a source the engine does not use
//! (`docs/research/actions.md` §5–§7). It is a reference-score test, not a
//! timing run.
//!
//! Each scenario is scored in four layers, the same ones the SDS keeps live:
//!
//! 1. **outcome** — how often the action succeeds over a fixed set of draws
//!    (Monte Carlo seeds 0..n; the draws are the engine's own scatter of
//!    friction, aim and slips), and why it fails;
//! 2. **time series** — the phases' timing and the shape of the force and
//!    angle trajectories (grip leading load, the vertical ground reaction of a
//!    step, when the bend's peak moment comes);
//! 3. **part** — each part's posture and load at the action's key instant
//!    (trunk, hip and knee at the bottom of a floor reach);
//! 4. **coupling** — what one part's action asks of another (N:N): the grasp's
//!    ZMP margin at the feet, the push's traction at the wheels, the step's
//!    friction at the sole.
//!
//! Scoring is `effects`': a likeness cell `100 − 10·|z|`, an "at least" cell
//! full above its floor, gates first (a robot that cannot do the action at
//! all scores 0 and says why), a cell the engine does not model scores 0 and
//! says "미구현", raw cells are reported only. Layers weigh the same; the
//! scenario is the mean of its scored layers.
//!
//! What keeps it from being tuned to: the engine files never name a scoring
//! source (a test scans them), every scenario runs the same engine with the
//! same parameters, and the held-out draws (`holdout`) use items, grounds and
//! loads the development scenarios do not.

use serde::Serialize;

use crate::act::drive;
use crate::act::gait;
use crate::act::grasp::{self as g, GraspInput, ReleaseInput};
use crate::act::material::ItemMaterial;
use crate::act::push::{self as p, PushInput, PushLeg};
use crate::act::reach::{self, ReachGoal};
use crate::balance::{reference, Body, RefCtx, Source, G};
use crate::effects::Rule;
use crate::hand::{Hand, NotoConfig};
use crate::parts::Segment;
use crate::surface::{Surface, SurfaceProps};

/// The four actions the benchmark asks about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Act4 {
    GraspRelease,
    Push,
    Walk,
    Bend,
}

impl Act4 {
    pub fn id(self) -> &'static str {
        match self {
            Act4::GraspRelease => "grasp_release",
            Act4::Push => "push",
            Act4::Walk => "walk",
            Act4::Bend => "bend",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Act4::GraspRelease => "잡기·놓기",
            Act4::Push => "밀기",
            Act4::Walk => "걷기",
            Act4::Bend => "굽히기",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Outcome,
    TimeSeries,
    Part,
    Coupling,
}

impl Layer {
    pub const ALL: [Layer; 4] = [Layer::Outcome, Layer::TimeSeries, Layer::Part, Layer::Coupling];

    pub fn id(self) -> &'static str {
        match self {
            Layer::Outcome => "outcome",
            Layer::TimeSeries => "time_series",
            Layer::Part => "part",
            Layer::Coupling => "coupling",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Layer::Outcome => "결과",
            Layer::TimeSeries => "시계열",
            Layer::Part => "부위",
            Layer::Coupling => "N:N 결합",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AxCell {
    pub layer: Layer,
    pub axis: &'static str,
    /// The body part the cell is about (`act::Part` id), if one.
    pub part: Option<&'static str>,
    /// `None`: the engine does not model this.
    pub value: Option<f32>,
    pub rule: Rule,
    pub cite: &'static str,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub gate: Option<String>,
}

/// The object a grasp, a push or a bend handles.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Obj {
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub size_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AxScenario {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub action: Act4,
    pub obj: Option<Obj>,
    /// The ground: a world ground id, or a laboratory floor with its friction.
    pub ground: Surface,
    /// Friction under the base when it differs from the ground's own (a contaminated floor), and what the controller believes.
    pub mu_actual: Option<f32>,
    pub mu_belief: Option<f32>,
    pub slope_deg: f32,
    /// Cargo in the bay, as a share of the body mass.
    pub cargo_frac: f32,
    /// Turns of a push path (degrees) at the planned radius.
    pub turns: &'static [f32],
    pub push_m: f32,
    /// Monte Carlo draws.
    pub trials: u32,
    pub holdout: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AxResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub action: Act4,
    pub holdout: bool,
    pub skipped: Option<String>,
    pub cells: Vec<AxCell>,
    /// Mean cell score per layer (`Layer::ALL` order).
    pub layer_scores: [Option<f32>; 4],
    pub score: f32,
    pub gates: Vec<String>,
    /// No gate and every scored cell within two tolerances (the held-out rule).
    pub passed: bool,
    /// The success rate over the draws and the failure codes counted.
    pub success_rate: f32,
    pub failures: Vec<(String, u32)>,
    /// The first reason the engine gave for each failure code.
    pub reasons: Vec<(String, String)>,
}

// ----- references (benchmark side only; the engine never reads these) -----

/// Single, known objects grasped by robots: 93% (8 known objects), 93% (dense
/// clutter, 288 attempts), 96% (unseen objects, 580k attempts), 98% (single
/// objects of the heap study): mean and SD across the four.
pub const GRASP_SUCCESS: (f32, f32) = (0.95, 0.024);
pub const CITE_GRASP_SUCCESS: &str = "로봇 파지 성공률 93% (Dex-Net 2.0, Mahler 2017) · 93% (GPD, ten Pas 2017) · 96% (QT-Opt, Kalashnikov 2018) · 98% (Dex-Net 4.0 단일 물체, Mahler 2019) — 평균 ± SD";
/// Human precision lift of a small object: preload 230 ± 91 ms, loading 214 ±
/// 51 ms, grip leads load by 26 ± 17 ms, hold safety margin 0.53 ± 0.06.
pub const PRELOAD_S: (f32, f32) = (0.230, 0.091);
pub const LOADING_S: (f32, f32) = (0.214, 0.051);
pub const GF_LEAD_S: (f32, f32) = (0.026, 0.017);
pub const SAFETY_MARGIN: (f32, f32) = (0.53, 0.06);
pub const GF_LF_CORR: (f32, f32) = (0.92, 0.05);
pub const CITE_GUILLERY: &str = "사람 정밀 들기: 예압 230±91 ms, 하중기 214±51 ms, 쥠이 하중보다 26±17 ms 앞섬, 쥠–하중 r 0.92, 유지 안전 여유 0.53±0.06 (Guillery 2013, n = 18)";
/// Placement error of a learned pick-and-place on unknown objects: 5.9 (1.1) mm.
pub const PLACE_ERR_MM: (f32, f32) = (5.9, 1.1);
pub const CITE_PLACE: &str = "놓기 위치 오차 5.9 (1.1) mm, 처음 보는 물체 (Berscheid 2020)";
/// Freestyle floor lift at the peak moment: torso 56.1 (13.0)° from vertical,
/// hip flexion 106.6 (9.2)°, knee flexion 78.5 (13.5)°; the peak comes 0.321
/// (0.122) s into the lift (n = 43).
pub const LIFT_TRUNK_DEG: (f32, f32) = (56.1, 13.0);
pub const LIFT_HIP_DEG: (f32, f32) = (106.6, 9.2);
pub const LIFT_KNEE_DEG: (f32, f32) = (78.5, 13.5);
pub const LIFT_PEAK_T_S: (f32, f32) = (0.321, 0.122);
pub const CITE_FLOOR_LIFT: &str = "바닥 들기(자유 자세) 최대 모멘트 때 몸통 56.1 (13.0)°, 고관절 굴곡 106.6 (9.2)°, 무릎 굴곡 78.5 (13.5)°, 들기 시작 후 0.321 (0.122) s (Menzer & Reiser 2005, n = 43)";
/// Initial push force half of men accept, 133 N; typical hand forces 133–267 N.
pub const PUSH_FORCE_N: (f32, f32) = (133.0, 67.0);
pub const CITE_PUSH: &str = "초기 미는 힘 남성 50% 수용 133 N (Liberty Mutual 표, Steele 2014 인용), 전형적 손 힘 133–267 N → 허용폭 (267−133)/2";
/// Share of unexpected exposures to a 0.164 ACOF floor that ended in a slip: 38% (n = 31).
pub const LAB_ACOF: f32 = 0.164;
pub const SLIP_FRACTION: (f32, f32) = (0.38, 0.087);
pub const CITE_SLIP: &str = "글리세롤 바닥(ACOF 0.149–0.179) 예고 없는 노출의 38%가 미끄러짐 (Beschorner 2016, n = 31)";
/// Share of first unexpected walking slips that ended in a fall, young adults:
/// 0.19 (n = 35; tolerance the binomial standard error).
pub const FALL_GIVEN_SLIP: (f32, f32) = (0.19, 0.066);
pub const CITE_FALL: &str = "예고 없는 첫 보행 미끄러짐의 넘어짐 비율 젊은 성인 0.19 (Pai 2010, n = 35; 허용폭 이항 표준오차)";
/// Slip distance of young adults' heel on an oily floor: 4.25 (0.87) cm (n = 14).
pub const SLIP_DIST_CM: (f32, f32) = (4.25, 0.87);
pub const CITE_SLIP_DIST: &str = "기름 바닥 젊은 성인 발뒤꿈치 미끄럼 거리 4.25 (0.87) cm (Lockhart 2003 표 4, n = 14) — 다리 길이로 무차원화";

/// Adult leg length the dimensionless comparisons assume for people, metres (est).
pub const HUMAN_LEG_M: f32 = 0.88;
/// Barefoot walking at 1.1 / 1.4 / 1.7 m/s (n = 30): vertical ground reaction
/// first peak, valley and second peak (BW, mean ± SD), and the stance share
/// of the stride (the men's and the women's rows pooled). Compared at the
/// robot's Froude number on a [`HUMAN_LEG_M`] leg (held at the ends).
pub const GRF_SPEEDS_MPS: [f32; 3] = [1.1, 1.4, 1.7];
pub const GRF_F1: [(f32, f32); 3] = [(1.17, 0.44), (1.37, 0.42), (1.52, 0.53)];
pub const GRF_VALLEY: [(f32, f32); 3] = [(0.86, 0.22), (0.76, 0.23), (0.62, 0.26)];
pub const GRF_F2: [(f32, f32); 3] = [(1.10, 0.23), (1.08, 0.18), (1.09, 0.16)];
pub const STANCE_SHARE: [(f32, f32); 3] = [(0.600, 0.031), (0.571, 0.039), (0.547, 0.038)];
pub const CITE_GRF: &str = "맨발 보행 1.1/1.4/1.7 m/s 수직 반력 첫 봉우리 1.17±0.44/1.37±0.42/1.52±0.53, 골 0.86±0.22/0.76±0.23/0.62±0.26, 둘째 봉우리 1.10±0.23/1.08±0.18/1.09±0.16 BW, 입각 59.9–60.2/56.8–57.3/54.3–55.0% (Wang 2017, n = 30) — 다리 0.88 m 프루드 수로 맞춤";
/// Mean loading rate from heel contact to the first peak, barefoot at 1.3 m/s: 7.96 ± 1.79 BW/s (n = 10).
pub const LOADING_RATE_BW_S: (f32, f32) = (7.96, 1.79);
pub const CITE_LOADING: &str = "맨발 1.3 m/s 첫 봉우리까지 평균 부하율 7.96 ± 1.79 BW/s (Zhang 2013, n = 10) — √(g/ℓ) 단위로 무차원화";
/// Heel-strike transient, barefoot: 0.48 BW at 30.2 BW/s (least-squares means
/// ± s.e. over 50 steps; the tolerance is the s.e. × √50, derived).
pub const IMPACT_BW: (f32, f32) = (0.48, 0.141);
pub const IMPACT_RATE_BW_S: (f32, f32) = (30.2, 17.0);
pub const CITE_IMPACT: &str = "맨발 보행 충격 봉우리 0.48 BW, 부하율 30.2 BW/s (Wallace 2018, 50걸음 최소제곱 평균; 허용폭은 표준오차 × √50, 유도)";
/// Walking heel impact, barefoot at about 1.3 m/s (n = 12, inter-subject SD):
/// time to the impact peak 17.5 (3.6) ms, heel vertical speed at contact 0.57
/// (0.12) m/s, effective mass 11.1 (2.8)% of the body.
pub const IMPACT_PEAK_MS: (f32, f32) = (17.5, 3.6);
pub const HEEL_SPEED_MPS: (f32, f32) = (0.57, 0.12);
pub const IMPACT_MASS_SHARE: (f32, f32) = (0.111, 0.028);
pub const CITE_HEEL: &str = "맨발 보행 충격 봉우리까지 17.5 (3.6) ms, 접지 순간 뒤꿈치 수직 속도 0.57 (0.12) m/s, 유효 질량 체중의 11.1 (2.8)% (Baines 2018, n = 12) — 시간·속도는 다리 길이로 무차원화";
/// Drawbar pull of a smooth rigid wheel (r 130 mm, 130 N) on dry sand at slip 0.1 / 0.3 / 0.7, digitised (ten repeats, spread shown only as box plots).
pub const SAND_DP_W: [(f32, f32); 3] = [(0.1, 0.050), (0.3, 0.138), (0.7, 0.192)];
pub const CITE_SAND_DP: &str = "마른 모래 매끈한 강체 바퀴 (r 130 mm, 130 N) 견인력/하중 0.050 @0.1 · 0.138 @0.3 · 0.192 @0.7 미끄럼 (Agarwal 2019 그림 12a 판독, 10회 반복 상자 그림뿐) — 표준편차 수치가 없어 기록만";

// ----- scenarios -----

const LOG: Obj = Obj { material: ItemMaterial::Wood, mass_kg: 2.4, size_m: 0.15 };
const STONE: Obj = Obj { material: ItemMaterial::Stone, mass_kg: 2.6, size_m: 0.10 };
const COAL: Obj = Obj { material: ItemMaterial::Coal, mass_kg: 1.4, size_m: 0.10 };
const ORE: Obj = Obj { material: ItemMaterial::IronOre, mass_kg: 4.0, size_m: 0.10 };

fn scen(id: &'static str, name_ko: &'static str, action: Act4, obj: Option<Obj>, ground: Surface) -> AxScenario {
    AxScenario { id, name_ko, action, obj, ground, mu_actual: None, mu_belief: None, slope_deg: 0.0, cargo_frac: 0.0, turns: &[], push_m: 0.0, trials: 60, holdout: false }
}

/// The development scenarios (research §6).
pub fn library() -> Vec<AxScenario> {
    vec![
        scen("G1", "자갈 위 원목 집어 적재함에", Act4::GraspRelease, Some(LOG), Surface::Gravel),
        scen("G2", "바위 지면 돌 집기", Act4::GraspRelease, Some(STONE), Surface::Rock),
        scen("G3", "흙 위 석탄 덩이 (먼지 낀 손 마찰)", Act4::GraspRelease, Some(COAL), Surface::Dirt),
        scen("G4", "철광석 4 kg 두 손", Act4::GraspRelease, Some(ORE), Surface::Gravel),
        AxScenario { push_m: 3.0, ..scen("P1", "자갈 위 돌 3 m 곧게 밀기", Act4::Push, Some(STONE), Surface::Gravel) },
        AxScenario { push_m: 3.0, turns: &[45.0, -45.0], ..scen("P2", "흙 위 원목 밀며 두 번 꺾기", Act4::Push, Some(LOG), Surface::Dirt) },
        AxScenario { push_m: 2.0, ..scen("P3", "마른 모래 위 철광석 밀기", Act4::Push, Some(ORE), Surface::Sand) },
        // Moving about: walking on legs, driving on wheels or tracks, over the same grounds.
        AxScenario { trials: 30, ..scen("W1", "단단한 바닥 이동", Act4::Walk, None, Surface::Firm) },
        AxScenario { trials: 30, slope_deg: 8.0, ..scen("W2", "젖은 모래 8° 오르기", Act4::Walk, None, Surface::WetSand) },
        AxScenario { trials: 30, mu_actual: Some(LAB_ACOF), mu_belief: Some(Surface::Firm.props().mu_foot), ..scen("W3", "예고 없는 오염 바닥", Act4::Walk, None, Surface::Firm) },
        AxScenario { trials: 30, cargo_frac: 0.10, ..scen("W4", "짐 10% 싣고 풀밭 이동", Act4::Walk, None, Surface::Grass) },
        AxScenario { trials: 30, ..scen("W5", "마른 모래 평지 이동", Act4::Walk, None, Surface::Sand) },
        AxScenario { trials: 30, slope_deg: 12.0, ..scen("W6", "마른 모래 12° 오르기", Act4::Walk, None, Surface::Sand) },
        AxScenario { trials: 1, ..scen("B1", "바닥의 원목으로 굽히기", Act4::Bend, Some(LOG), Surface::Gravel) },
        AxScenario { trials: 1, cargo_frac: 0.10, ..scen("B2", "짐 실은 채 바닥 돌로 굽히기", Act4::Bend, Some(STONE), Surface::Rock) },
    ]
}

/// Grounds, objects and loads the held-out draws use; none is a development condition (a test checks it).
pub const HOLDOUT_GROUNDS: [Surface; 3] = [Surface::Grass, Surface::WetSand, Surface::Dirt];
pub const HOLDOUT_AMOUNTS: [f32; 2] = [0.5, 0.75];
pub const HOLDOUT_CARGO: (f32, f32) = (0.03, 0.07);

/// Held-out scenarios for one evaluation seed: broken pieces of the items on
/// other grounds with a little cargo. Counted pass / fail, never tuned against.
pub fn holdout(seed: u64, count: usize) -> Vec<AxScenario> {
    let mut s = seed ^ 0x6A09_E667_F3BC_C908;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let objs = [LOG, STONE, COAL, ORE];
    (0..count)
        .map(|i| {
            let mut o = objs[(next() * 4.0) as usize % 4];
            let amount = HOLDOUT_AMOUNTS[(next() * 2.0) as usize % 2];
            if o.material != ItemMaterial::Wood {
                o = Obj { mass_kg: o.mass_kg * amount, size_m: o.size_m * amount.cbrt(), ..o };
            }
            let ground = HOLDOUT_GROUNDS[(next() * 3.0) as usize % 3];
            let cargo = HOLDOUT_CARGO.0 + (HOLDOUT_CARGO.1 - HOLDOUT_CARGO.0) * next();
            let action = if i % 2 == 0 { Act4::GraspRelease } else { Act4::Push };
            let id: &'static str = Box::leak(format!("Y{}-{i}", seed % 1000).into_boxed_str());
            let name: &'static str = Box::leak(format!("{} {} {:.1} kg · {} · 짐 {:.0}%", action.label_ko(), o.material.label_ko(), o.mass_kg, ground.label_ko(), cargo * 100.0).into_boxed_str());
            AxScenario { id, name_ko: name, action, obj: Some(o), ground, mu_actual: None, mu_belief: None, slope_deg: 0.0, cargo_frac: cargo, turns: &[], push_m: 2.0, trials: 30, holdout: true }
        })
        .collect()
}

// ----- scoring -----

/// The shared judge (`crate::bench::judge`); the gate is worded as this benchmark prints it.
fn judge(c: &mut AxCell) {
    let v = crate::bench::judge(c.value, c.rule.into());
    c.z = v.z;
    c.score = v.score;
    c.gate = v.broken.map(|(x, limit)| format!("{} {x:.2} > {limit:.2}", c.axis));
}

fn cell(layer: Layer, axis: &'static str, part: Option<&'static str>, value: Option<f32>, rule: Rule, cite: &'static str) -> AxCell {
    let mut c = AxCell { layer, axis, part, value, rule, cite, z: None, score: None, gate: None };
    judge(&mut c);
    c
}

fn like(r: (f32, f32)) -> Rule {
    Rule::Like { reference: r.0, tolerance: r.1 }
}

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}

fn finish(s: &AxScenario, cells: Vec<AxCell>, success_rate: f32, failures: Vec<(String, u32)>, skipped: Option<String>) -> AxResult {
    let gates: Vec<String> = cells.iter().filter_map(|c| c.gate.clone()).collect();
    let layer_scores: [Option<f32>; 4] = std::array::from_fn(|i| {
        let sc: Vec<f32> = cells.iter().filter(|c| c.layer == Layer::ALL[i]).filter_map(|c| c.score).collect();
        mean(&sc)
    });
    let scored: Vec<f32> = layer_scores.iter().flatten().copied().collect();
    // Nothing to compare with is no score, not a full one.
    let score = if skipped.is_some() || !gates.is_empty() { 0.0 } else { mean(&scored).unwrap_or(0.0) };
    let within = cells.iter().all(|c| c.score.is_none() || c.z.map_or(c.score.unwrap_or(0.0) > 0.0, |z| z.abs() <= 2.0));
    AxResult { id: s.id, name_ko: s.name_ko, action: s.action, holdout: s.holdout, skipped, passed: gates.is_empty() && within && !scored.is_empty(), gates, layer_scores, score, cells, success_rate, failures, reasons: vec![] }
}

/// The first reason of each failure code over a set of rollouts.
fn reasons_of(runs: &[crate::act::ActRollout]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Vec::new();
    for r in runs {
        if let (Some(c), Some(why)) = (r.outcome.code, &r.outcome.reason_ko) {
            if !v.iter().any(|(k, _)| k == c) {
                v.push((c.to_string(), why.clone()));
            }
        }
    }
    v
}

fn count_codes(codes: &[Option<&'static str>]) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = Vec::new();
    for c in codes.iter().flatten() {
        match v.iter_mut().find(|(k, _)| k == c) {
            Some(e) => e.1 += 1,
            None => v.push((c.to_string(), 1)),
        }
    }
    v
}

/// The hand the robot has, as `NotoConfig::preset` builds it for the body's finger count and base.
pub fn hand_of(body: &Body) -> Hand {
    let mut cfg = NotoConfig::preset(body.fingers);
    cfg.base = body.base;
    cfg.build().expect("preset hands build")
}

/// Why a pick from above is refused: two hands do not hold even the object's weight (`grasp::top_hands`).
pub fn refused_ko(th: &g::TopHands) -> String {
    format!("두 손으로도 무게를 못 버팀 (한 손 {:.0} N × 2 < 무게 {:.0} N, 손가락 {}개 닿음): 집지 않음", th.hold_n, th.weight_n, th.touching)
}

/// The aim error the benchmark's robot has: a stereo head 1.05 m from an object at its feet (D435 law) and a 5 mm arm.
fn aim_sigma(body: &Body) -> f32 {
    let depth = crate::act::material::stereo_depth_sigma(1.09, 424.0, 0.05, 0.08);
    let arm = if body.arm.shoulder.backdrivable { 0.005 } else { 0.010 };
    (depth * depth + arm * arm).sqrt()
}

/// Runs one scenario on a robot.
pub fn evaluate(body: &Body, s: &AxScenario) -> AxResult {
    match s.action {
        Act4::GraspRelease => eval_grasp(body, s),
        Act4::Push => eval_push(body, s),
        Act4::Walk => eval_walk(body, s),
        Act4::Bend => eval_bend(body, s),
    }
}

fn eval_grasp(body: &Body, s: &AxScenario) -> AxResult {
    let o = s.obj.expect("a grasp scenario has an object");
    let hand = hand_of(body);
    let cargo = s.cargo_frac * body.mass_kg();
    // One hand when it holds the object with its safety factor, else both, and none when two hands do not
    // hold even its weight: the contacts the rollout holds with (`top_hands`; the world's grasp table still
    // reads the v0.9 `grasp::capacity`).
    let th = g::top_hands(&hand, [o.size_m; 3], o.mass_kg, crate::act::material::pad_on(o.material).mean);
    if th.hands == 0 {
        return finish(s, vec![], 0.0, vec![], Some(refused_ko(&th)));
    }
    let two = th.hands == 2;
    let mut runs = Vec::new();
    for seed in 0..s.trials as u64 {
        runs.push(g::grasp(&GraspInput::front(body, &hand, o.material, o.mass_kg, [o.size_m; 3], s.ground, two, cargo, aim_sigma(body), seed)));
    }
    let ok: Vec<&crate::act::ActRollout> = runs.iter().filter(|r| r.outcome.ok).collect();
    let rate = ok.len() as f32 / runs.len().max(1) as f32;
    let failures = count_codes(&runs.iter().map(|r| r.outcome.code).collect::<Vec<_>>());
    let reach = runs.iter().any(|r| r.outcome.code == Some("reach_ground"));
    let m = |k: &str| mean(&ok.iter().filter_map(|r| r.metric(k)).collect::<Vec<_>>());
    let mut cells = Vec::new();
    // Outcome.
    if reach {
        cells.push(cell(Layer::Outcome, "reach_ground", Some("waist"), Some(1.0), Rule::AtMost { limit: 0.0 }, "바닥의 물체에 손이 닿아야 집을 수 있음 (기하·관절 범위)"));
    }
    cells.push(cell(Layer::Outcome, "success_rate", Some("hand"), Some(rate), like(GRASP_SUCCESS), CITE_GRASP_SUCCESS));
    let rate_of = |code: &str| runs.iter().filter(|r| r.outcome.code == Some(code)).count() as f32 / runs.len().max(1) as f32;
    cells.push(cell(Layer::Outcome, "miss_rate", Some("hand"), Some(rate_of("grasp_missed")), Rule::Raw, ""));
    cells.push(cell(Layer::Outcome, "slip_rate", Some("hand"), Some(rate_of("grasp_slip")), Rule::Raw, ""));
    cells.push(cell(Layer::Outcome, "overload_rate", Some("hand_joints"), Some(rate_of("lift_overload")), Rule::Raw, ""));
    // Time series.
    cells.push(cell(Layer::TimeSeries, "preload_s", Some("hand"), m("preload_s"), like(PRELOAD_S), CITE_GUILLERY));
    cells.push(cell(Layer::TimeSeries, "loading_s", Some("hand"), m("loading_s"), like(LOADING_S), CITE_GUILLERY));
    cells.push(cell(Layer::TimeSeries, "gf_lead_s", Some("hand"), m("gf_lead_s"), like(GF_LEAD_S), CITE_GUILLERY));
    cells.push(cell(Layer::TimeSeries, "gf_lf_corr", Some("hand"), m("gf_lf_corr"), like(GF_LF_CORR), CITE_GUILLERY));
    cells.push(cell(Layer::TimeSeries, "safety_margin", Some("hand"), m("safety_margin"), like(SAFETY_MARGIN), CITE_GUILLERY));
    cells.push(cell(Layer::TimeSeries, "duration_s", None, m("duration_s"), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "lift_s", None, m("lift_s"), Rule::Raw, ""));
    // The release of each successful grasp: where it lands.
    let place: Vec<f32> = (0..ok.len().min(30) as u64)
        .map(|seed| g::release(&ReleaseInput { body, material: o.material, mass_kg: o.mass_kg, size_m: [o.size_m; 3], ground: s.ground, surface_z_m: 0.0, cargo_kg: cargo, aim_sigma_m: aim_sigma(body), seed }))
        .filter_map(|r| r.metric("place_err_m"))
        .map(|e| e * 1000.0)
        .collect();
    cells.push(cell(Layer::TimeSeries, "place_err_mm", Some("hand"), mean(&place), like(PLACE_ERR_MM), CITE_PLACE));
    // Part: the bottom of the bend that reached the object (the engine's own plan: in front, or a wide squat over a load it cannot hold there).
    let plan = reach::plan_or_straddle(body, &ReachGoal::ground_object(o.size_m, o.mass_kg, two, cargo), 0.5 * o.size_m);
    part_cells(body, &plan, &mut cells);
    // Coupling: the feet's margin while the hands lift, the neck's camera pitch.
    let margin = m("min_margin");
    let mr = reference(Source::ArmLift, Segment::Foot, "margin", &RefCtx::default());
    match mr {
        Some(r) => cells.push(cell(Layer::Coupling, "hand>foot:margin", Some("foot"), margin, Rule::Like { reference: r.value, tolerance: r.tolerance }, r.cite)),
        None => cells.push(cell(Layer::Coupling, "hand>foot:margin", Some("foot"), margin, Rule::Raw, "")),
    }
    cells.push(cell(Layer::Coupling, "hand>joints:peak_ratio", Some("hand_joints"), m("peak_ratio"), Rule::Raw, ""));
    AxResult { reasons: reasons_of(&runs), ..finish(s, cells, rate, failures, None) }
}

/// The trunk, hip and knee at the bottom of a floor reach against the human's (legs only for the hip and knee).
fn part_cells(body: &Body, plan: &reach::ReachPlan, cells: &mut Vec<AxCell>) {
    let ok = plan.ok;
    let r = &plan.loaded;
    let trunk = ok.then(|| r.pose.trunk_deg());
    cells.push(cell(Layer::Part, "trunk_deg", Some("waist"), trunk, like(LIFT_TRUNK_DEG), CITE_FLOOR_LIFT));
    if body.legged() {
        let hip = if ok { r.joint("hip_pitch_r").and_then(|j| j.angle_deg) } else { None };
        let knee = if ok { r.joint("knee_r").and_then(|j| j.angle_deg) } else { None };
        cells.push(cell(Layer::Part, "hip_flex_deg", Some("foot_joints"), hip, like(LIFT_HIP_DEG), CITE_FLOOR_LIFT));
        cells.push(cell(Layer::Part, "knee_flex_deg", Some("foot_joints"), knee, like(LIFT_KNEE_DEG), CITE_FLOOR_LIFT));
    }
    let worst = if ok { r.worst().map(|j| j.ratio) } else { None };
    cells.push(cell(Layer::Part, "peak_ratio_bottom", None, worst, Rule::Raw, ""));
    let moment = if ok { r.joint(if body.legged() { "hip_pitch_r" } else { "waist_pitch" }).map(|j| j.torque.abs() * if body.legged() { 2.0 } else { 1.0 } / body.mass_kg()) } else { None };
    cells.push(cell(Layer::Part, "trunk_moment_nm_per_kg", Some("waist"), moment, Rule::Raw, ""));
}

fn eval_bend(body: &Body, s: &AxScenario) -> AxResult {
    let o = s.obj.expect("a bend scenario has an object");
    let hand = hand_of(body);
    let cargo = s.cargo_frac * body.mass_kg();
    let th = g::top_hands(&hand, [o.size_m; 3], o.mass_kg, crate::act::material::pad_on(o.material).mean);
    if th.hands == 0 {
        return finish(s, vec![], 0.0, vec![], Some(refused_ko(&th)));
    }
    let two = th.hands == 2;
    let plan = reach::plan_or_straddle(body, &ReachGoal::ground_object(o.size_m, o.mass_kg, two, cargo), 0.5 * o.size_m);
    let mut cells = Vec::new();
    if !plan.ok {
        cells.push(cell(Layer::Outcome, "reach_ground", Some("waist"), Some(1.0), Rule::AtMost { limit: 0.0 }, "바닥의 물체에 손이 닿아야 굽혀 집을 수 있음"));
    }
    cells.push(cell(Layer::Outcome, "lowest_palm_m", Some("hand"), Some(plan.lowest_palm_z), Rule::Raw, ""));
    part_cells(body, &plan, &mut cells);
    // When the bend's peak moment comes: the rise's samples of one grasp.
    let r = g::grasp(&GraspInput::front(body, &hand, o.material, o.mass_kg, [o.size_m; 3], s.ground, two, cargo, 0.0, 1));
    let peak_t = (|| {
        let lift: Vec<&crate::act::ActSample> = r.samples.iter().filter(|x| x.phase == "lift").collect();
        let t0 = lift.first()?.t;
        let torque = |x: &crate::act::ActSample| {
            let k = crate::kin::solve_fast(body, &x.pose, &x.loads);
            k.joint(if body.legged() { "hip_pitch_r" } else { "waist_pitch" }).map_or(0.0, |j| j.torque.abs())
        };
        let best = lift.iter().max_by(|a, b| torque(a).total_cmp(&torque(b)))?;
        Some(best.t - t0)
    })();
    cells.push(cell(Layer::TimeSeries, "peak_moment_t_s", Some("waist"), plan.ok.then_some(peak_t).flatten(), like(LIFT_PEAK_T_S), CITE_FLOOR_LIFT));
    cells.push(cell(Layer::TimeSeries, "descend_s", None, plan.ok.then_some(plan.descend_s), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "rise_s", None, plan.ok.then_some(plan.rise_s), Rule::Raw, ""));
    // Coupling: the bend at the feet (margin) and at the camera (pitch).
    let mr = reference(Source::ArmLift, Segment::Foot, "margin", &RefCtx::default());
    let margin = plan.ok.then_some(plan.loaded.margin);
    match mr {
        Some(rf) => cells.push(cell(Layer::Coupling, "waist>foot:margin", Some("foot"), margin, Rule::Like { reference: rf.value, tolerance: rf.tolerance }, rf.cite)),
        None => cells.push(cell(Layer::Coupling, "waist>foot:margin", Some("foot"), margin, Rule::Raw, "")),
    }
    cells.push(cell(Layer::Coupling, "waist>neck:camera_pitch_deg", Some("neck"), plan.ok.then(|| plan.loaded.pose.trunk_deg() + plan.loaded.pose.neck_deg), Rule::Raw, ""));
    let rate = if plan.ok { 1.0 } else { 0.0 };
    finish(s, cells, rate, vec![], None)
}

fn eval_push(body: &Body, s: &AxScenario) -> AxResult {
    let o = s.obj.expect("a push scenario has an object");
    let cargo = s.cargo_frac * body.mass_kg();
    if body.legged() {
        return finish(s, vec![], 0.0, vec![], Some("다리형은 범퍼로 밀지 않음 (들어서 옮김)".into()));
    }
    let r_plan = p::planned_radius_m(body, o.material, o.size_m);
    let mut legs = vec![PushLeg { turn_deg: 0.0, radius_m: 0.0, len_m: s.push_m / (s.turns.len() + 1) as f32, ground: s.ground, slope_deg: s.slope_deg, blocked: false }];
    for t in s.turns {
        legs.push(PushLeg { turn_deg: *t, radius_m: r_plan, len_m: s.push_m / (s.turns.len() + 1) as f32, ground: s.ground, slope_deg: s.slope_deg, blocked: false });
    }
    let runs: Vec<crate::act::ActRollout> = (0..s.trials as u64)
        .map(|seed| p::push(&PushInput { body, material: o.material, mass_kg: o.mass_kg, size_m: [o.size_m; 3], cargo_kg: cargo, legs: legs.clone(), aim_sigma_m: aim_sigma(body), seed }))
        .collect();
    let ok: Vec<&crate::act::ActRollout> = runs.iter().filter(|r| r.outcome.ok).collect();
    let rate = ok.len() as f32 / runs.len().max(1) as f32;
    let failures = count_codes(&runs.iter().map(|r| r.outcome.code).collect::<Vec<_>>());
    let m = |k: &str| mean(&runs.iter().filter_map(|r| r.metric(k)).collect::<Vec<_>>());
    let mut cells = Vec::new();
    cells.push(cell(Layer::Outcome, "success_rate", Some("wheel"), Some(rate), Rule::Raw, ""));
    let stuck = runs.iter().filter(|r| r.outcome.code == Some("push_stuck")).count();
    cells.push(cell(Layer::Outcome, "stuck", Some("wheel"), Some(stuck as f32 / runs.len().max(1) as f32), Rule::AtMost { limit: 0.5 }, "구동부가 물체를 밀지 못하고 헛돎"));
    cells.push(cell(Layer::TimeSeries, "push_peak_n", Some("wheel"), m("push_peak_n"), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "duration_s", None, m("duration_s"), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "contact_impact_n", Some("wheel"), m("contact_impact_n"), Rule::Raw, ""));
    // Part: the largest push the drive and the ground give at the bumper (traction less the rolling resistance).
    let props: SurfaceProps = *s.ground.props();
    let m_tot = body.mass_kg() + cargo;
    let mu = crate::surface::available_mu(body.base, &props);
    let crr = crate::surface::rolling_resistance(body.base, &props);
    let cap = (mu * m_tot * G).min(body.chassis.drive_force_n) - crr * m_tot * G;
    cells.push(cell(Layer::Part, "max_push_n", Some("wheel"), Some(cap.max(0.0)), Rule::AtLeast { reference: PUSH_FORCE_N.0, tolerance: PUSH_FORCE_N.1 }, CITE_PUSH));
    cells.push(cell(Layer::Part, "stable_radius_m", Some("wheel"), m("stable_radius_m"), Rule::Raw, ""));
    // Coupling: the push at the bumper loads the drive (slip) and moves the ZMP back.
    cells.push(cell(Layer::Coupling, "bumper>wheel:slip_max", Some("wheel"), m("slip_max"), Rule::Raw, ""));
    let st = crate::act::drive::state(body, &props, s.slope_deg, 0.0, cargo, m("push_peak_n").unwrap_or(0.0));
    cells.push(cell(Layer::Coupling, "bumper>wheel:front_share", Some("wheel"), Some(st.front_n / (st.front_n + st.rear_n).max(1e-3)), Rule::Raw, ""));
    AxResult { reasons: reasons_of(&runs), ..finish(s, cells, rate, failures, None) }
}

/// The stance foot's vertical force over every whole stance of a walk: per
/// stance, (seconds from its landing, force over the weight). The samples are
/// solved again; a stance cut by the walk's start or end is left out.
fn stances(body: &Body, r: &crate::act::ActRollout, weight_n: f32) -> Vec<Vec<(f32, f32)>> {
    let mut out = Vec::new();
    for (s, name) in [(0usize, "foot_r"), (1, "foot_l")] {
        let mut cur: Vec<(f32, f32)> = Vec::new();
        for x in &r.samples {
            if x.pose.feet[s][1] <= 1e-4 {
                let k = crate::kin::solve_fast(body, &x.pose, &x.loads);
                cur.push((x.t, k.contact(name).map_or(0.0, |c| c.normal_n) / weight_n));
            } else if !cur.is_empty() {
                // A stance that starts already loaded began before the walk did;
                // the unloaded instant at lift-off (the swing foot still at
                // height 0) is not stance.
                let last = cur.iter().rposition(|p| p.1 > 0.01);
                if let (true, Some(last)) = (cur[0].1 < 0.5, last) {
                    let t0 = cur[0].0;
                    out.push(cur[..=last].iter().map(|(t, f)| (t - t0, *f)).collect());
                }
                cur.clear();
            }
        }
    }
    out
}

/// A stance's vertical force shape: first peak, valley and second peak (BW),
/// the time to the first peak and the stance's duration (s).
fn grf_shape(st: &[(f32, f32)], dt: f32) -> Option<(f32, f32, f32, f32, f32)> {
    let dur = st.last()?.0 + dt;
    let part = |a: f32, b: f32| st.iter().filter(move |(t, _)| *t >= a * dur && *t <= b * dur).map(|p| p.1);
    let f1 = part(0.0, 0.5).fold(f32::NEG_INFINITY, f32::max);
    let valley = part(0.3, 0.7).fold(f32::INFINITY, f32::min);
    let f2 = part(0.5, 1.0).fold(f32::NEG_INFINITY, f32::max);
    let t1 = st.iter().find(|p| p.1 >= f1 - 1e-3)?.0;
    (f1.is_finite() && valley.is_finite() && f2.is_finite()).then_some((f1, valley, f2, t1.max(dt), dur))
}

/// A reference over walking speed at the robot's Froude number on a human leg (held at the table's ends).
fn at_froude(table: &[(f32, f32); 3], froude: f32) -> (f32, f32) {
    let fr = GRF_SPEEDS_MPS.map(|v| v * v / (G * HUMAN_LEG_M));
    if froude <= fr[0] {
        return table[0];
    }
    if froude >= fr[2] {
        return table[2];
    }
    let i = if froude <= fr[1] { 0 } else { 1 };
    let u = (froude - fr[i]) / (fr[i + 1] - fr[i]);
    (table[i].0 + (table[i + 1].0 - table[i].0) * u, table[i].1 + (table[i + 1].1 - table[i].1) * u)
}

/// A rate per second in units of √(g/ℓ), a time in units of √(ℓ/g), a speed in units of √(g·ℓ): dynamic similarity.
fn per_dimless_t(x_per_s: f32, leg_m: f32) -> f32 {
    x_per_s * (leg_m / G).sqrt()
}

fn dimless_t(seconds: f32, leg_m: f32) -> f32 {
    seconds * (G / leg_m).sqrt()
}

fn dimless_v(mps: f32, leg_m: f32) -> f32 {
    mps / (G * leg_m).sqrt()
}

/// A human (mean, SD) carried into dimensionless units by `f`.
fn human(r: (f32, f32), f: fn(f32, f32) -> f32) -> Rule {
    Rule::Like { reference: f(r.0, HUMAN_LEG_M), tolerance: f(r.1, HUMAN_LEG_M) }
}

fn eval_walk(body: &Body, s: &AxScenario) -> AxResult {
    if !body.legged() {
        return eval_drive(body, s);
    }
    let props = SurfaceProps { mu_foot: s.mu_actual.unwrap_or(s.ground.props().mu_foot), ..*s.ground.props() };
    let cargo = s.cargo_frac * body.mass_kg();
    let speed = body.drive_limits(&body.base.spec(), cargo).max_speed_mps;
    let weight = (body.mass_kg() + cargo) * G;
    let leg = crate::parts::HIP_M;
    let (mut slipped_steps, mut steps, mut falls) = (0u32, 0u32, 0u32);
    let mut runs = Vec::new();
    let mut slides = Vec::new();
    let mut shapes = Vec::new();
    for seed in 0..s.trials as u64 {
        let (r, st) = gait::walk(body, &props, s.mu_belief, s.slope_deg, cargo, speed, 40, seed);
        steps += st.len() as u32;
        slipped_steps += st.iter().filter(|x| x.slipped).count() as u32;
        falls += st.iter().filter(|x| x.fell).count() as u32;
        slides.extend(st.iter().filter(|x| x.slipped).map(|x| x.slide_m));
        if seed < 2 {
            // The force shape of every whole stance (the steps repeat; two walks are plenty).
            shapes.extend(stances(body, &r, weight).iter().filter_map(|x| grf_shape(x, crate::act::SAMPLE_S)));
        }
        runs.push(r);
    }
    let rate = 1.0 - falls as f32 / s.trials.max(1) as f32;
    let failures = if falls > 0 { vec![("fall".to_string(), falls)] } else { vec![] };
    let m = |k: &str| mean(&runs.iter().filter_map(|r| r.metric(k)).collect::<Vec<_>>());
    let shape = |f: fn(&(f32, f32, f32, f32, f32)) -> f32| mean(&shapes.iter().map(f).collect::<Vec<_>>());
    let froude = m("froude").unwrap_or(0.0);
    let mut cells = Vec::new();
    // Outcome.
    cells.push(cell(Layer::Outcome, "trial_no_fall_rate", Some("foot"), Some(rate), Rule::Raw, ""));
    cells.push(cell(Layer::Outcome, "slip_per_step", Some("foot"), Some(slipped_steps as f32 / steps.max(1) as f32), Rule::Raw, ""));
    if s.mu_actual.is_some() {
        // An unexpected slippery floor, first exposures (the first step on it): the share that slip, and of those the share that fall.
        let first: Vec<gait::StepStat> = (0..s.trials as u64).filter_map(|seed| gait::walk(body, &props, s.mu_belief, 0.0, cargo, speed, 1, 1000 + seed).1.into_iter().next()).collect();
        let slipped: Vec<&gait::StepStat> = first.iter().filter(|x| x.slipped).collect();
        cells.push(cell(Layer::Outcome, "slip_fraction", Some("foot"), Some(slipped.len() as f32 / first.len().max(1) as f32), like(SLIP_FRACTION), CITE_SLIP));
        let fell = (!slipped.is_empty()).then(|| slipped.iter().filter(|x| x.fell).count() as f32 / slipped.len() as f32);
        cells.push(cell(Layer::Outcome, "fall_given_slip", Some("foot"), fell, if fell.is_some() { like(FALL_GIVEN_SLIP) } else { Rule::Raw }, CITE_FALL));
    }
    // Time series: the stance foot's vertical force, its loading, the heel-strike
    // transient. A walk that falls before one whole stance has no force shape
    // to compare (recorded empty, not scored).
    let scored = |r: Rule| if shapes.is_empty() { Rule::Raw } else { r };
    cells.push(cell(Layer::TimeSeries, "grf_f1_bw", Some("foot"), shape(|x| x.0), scored(like(at_froude(&GRF_F1, froude))), CITE_GRF));
    cells.push(cell(Layer::TimeSeries, "grf_valley_bw", Some("foot"), shape(|x| x.1), scored(like(at_froude(&GRF_VALLEY, froude))), CITE_GRF));
    cells.push(cell(Layer::TimeSeries, "grf_f2_bw", Some("foot"), shape(|x| x.2), scored(like(at_froude(&GRF_F2, froude))), CITE_GRF));
    let stance = shape(|x| x.4).zip(m("step_s")).map(|(d, st)| d / (2.0 * st).max(1e-3));
    cells.push(cell(Layer::TimeSeries, "stance_share", Some("foot"), stance, scored(like(at_froude(&STANCE_SHARE, froude))), CITE_GRF));
    cells.push(cell(Layer::TimeSeries, "loading_rate_f1", Some("foot"), shape(|x| x.0 / x.3).map(|v| per_dimless_t(v, leg)), scored(human(LOADING_RATE_BW_S, per_dimless_t)), CITE_LOADING));
    cells.push(cell(Layer::TimeSeries, "impact_bw", Some("foot"), m("impact_bw"), like(IMPACT_BW), CITE_IMPACT));
    cells.push(cell(Layer::TimeSeries, "impact_rate", Some("foot"), m("loading_rate_bw_s").map(|v| per_dimless_t(v, leg)), human(IMPACT_RATE_BW_S, per_dimless_t), CITE_IMPACT));
    cells.push(cell(Layer::TimeSeries, "impact_peak_t", Some("foot"), m("impact_ms").map(|ms| dimless_t(0.5e-3 * ms, leg)), human((IMPACT_PEAK_MS.0 * 1e-3, IMPACT_PEAK_MS.1 * 1e-3), dimless_t), CITE_HEEL));
    cells.push(cell(Layer::TimeSeries, "relative_step", Some("foot"), m("relative_step"), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "froude", None, Some(froude), Rule::Raw, ""));
    // Part: how the leg lands.
    cells.push(cell(Layer::Part, "touchdown_speed", Some("foot_joints"), m("touchdown_mps").map(|v| dimless_v(v, leg)), human(HEEL_SPEED_MPS, dimless_v), CITE_HEEL));
    cells.push(cell(Layer::Part, "impact_mass_share", Some("foot_joints"), m("impact_mass_kg").map(|k| k / body.mass_kg()), like(IMPACT_MASS_SHARE), CITE_HEEL));
    // Coupling: what the ground and the gait ask of the sole, and a slip of the foot.
    cells.push(cell(Layer::Coupling, "foot>sole:rcof", Some("foot"), m("rcof"), Rule::Raw, ""));
    cells.push(cell(Layer::Coupling, "speed_over_cmd", Some("foot_joints"), m("speed_mps").zip(m("speed_cmd_mps")).map(|(a, b)| a / b.max(1e-3)), Rule::Raw, ""));
    if s.mu_actual.is_some() {
        // How far a slipping foot slides (none recorded when no step slipped).
        let rule = if slides.is_empty() { Rule::Raw } else { human((SLIP_DIST_CM.0 / 100.0, SLIP_DIST_CM.1 / 100.0), |d, l| d / l) };
        cells.push(cell(Layer::Coupling, "slip>foot:slide_per_leg", Some("foot"), mean(&slides).map(|d| d / leg), rule, CITE_SLIP_DIST));
    }
    AxResult { reasons: reasons_of(&runs), ..finish(s, cells, rate, failures, None) }
}

/// Moving about on wheels or tracks: the drive's slip and loads at the start,
/// the cruise and the stop over the scenario's ground, slope and cargo
/// (`act::drive`), with the accelerations planned for the ground it believes.
fn eval_drive(body: &Body, s: &AxScenario) -> AxResult {
    let mut props: SurfaceProps = *s.ground.props();
    if let Some(mu) = s.mu_actual {
        props.mu_wheel = mu;
        props.mu_track = mu;
    }
    let part = if body.base == crate::base::BaseKind::Track { "track" } else { "wheel" };
    let cargo = s.cargo_frac * body.mass_kg();
    let spec = body.base.spec();
    let a_up = body.planned_accel(&spec, s.slope_deg, cargo, s.ground, 1.0, 0.5);
    let a_dn = body.planned_accel(&spec, s.slope_deg, cargo, s.ground, -1.0, 0.5);
    let start = drive::state(body, &props, s.slope_deg, a_up, cargo, 0.0);
    let cruise = drive::state(body, &props, s.slope_deg, 0.0, cargo, 0.0);
    let stop = drive::state(body, &props, s.slope_deg, -a_dn, cargo, 0.0);
    let stuck = cruise.slip >= 0.999;
    let share = |st: &drive::DriveState| st.front_n / (st.front_n + st.rear_n).max(1e-3);
    let mut cells = Vec::new();
    cells.push(cell(Layer::Outcome, "stuck", Some(part), Some(if stuck { 1.0 } else { 0.0 }), Rule::AtMost { limit: 0.5 }, "구동부가 헛돌아 나아가지 못함"));
    cells.push(cell(Layer::Outcome, "start_spins", Some(part), Some(if start.slip >= 0.999 { 1.0 } else { 0.0 }), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "slip_start", Some(part), Some(start.slip), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "slip_cruise", Some(part), Some(cruise.slip), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "slip_stop", Some(part), Some(stop.slip), Rule::Raw, ""));
    cells.push(cell(Layer::TimeSeries, "speed_factor", Some(part), Some(cruise.speed_factor), Rule::Raw, ""));
    cells.push(cell(Layer::Part, "front_share_cruise", Some(part), Some(share(&cruise)), Rule::Raw, ""));
    if let Some(p) = cruise.pressure_kpa {
        cells.push(cell(Layer::Part, "track_pressure_max_kpa", Some("track"), Some(p[0].max(p[1])), Rule::Raw, ""));
    }
    if s.ground == Surface::Sand && s.slope_deg == 0.0 {
        // Traction over slip on dry sand, recorded beside a rigid wheel's (no spread to score with).
        for (axis, slip) in [("dp_w@0.1", 0.1), ("dp_w@0.3", 0.3), ("dp_w@0.7", 0.7)] {
            cells.push(cell(Layer::Part, axis, Some(part), Some(drive::drawbar_over_weight(body, &props, slip)), Rule::Raw, CITE_SAND_DP));
        }
    }
    cells.push(cell(Layer::Coupling, "slope>drive:need_over_limit", Some(part), Some(cruise.need_n / cruise.limit_n.max(1e-3)), Rule::Raw, ""));
    cells.push(cell(Layer::Coupling, "brake>axle:front_share", Some(part), Some(share(&stop)), Rule::Raw, ""));
    let failures = if stuck { vec![("stuck".to_string(), 1)] } else { vec![] };
    let mut r = finish(s, cells, if stuck { 0.0 } else { 1.0 }, failures, None);
    if stuck {
        r.reasons = vec![("stuck".into(), format!("{} 경사 {:.0}°: 필요 {:.0} N > 지면 한계 {:.0} N", s.ground.label_ko(), s.slope_deg, cruise.need_n, cruise.limit_n))];
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    #[test]
    fn holdout_conditions_never_touch_the_development_ones() {
        for s in library() {
            assert!(!HOLDOUT_GROUNDS.contains(&s.ground) || !matches!(s.action, Act4::GraspRelease | Act4::Push) || s.cargo_frac == 0.0, "{}: a development grasp or push on a held-out ground with cargo", s.id);
            assert!(!(HOLDOUT_CARGO.0..=HOLDOUT_CARGO.1).contains(&s.cargo_frac), "{}", s.id);
        }
        let h = holdout(20261024, 6);
        assert_eq!(h, holdout(20261024, 6), "a seed gives the same draws");
        assert!(h.iter().all(|s| s.holdout && HOLDOUT_GROUNDS.contains(&s.ground)));
    }

    #[test]
    fn a_robot_that_cannot_reach_scores_zero_and_says_why() {
        let wheel = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let g1 = library().into_iter().find(|s| s.id == "G1").unwrap();
        let r = evaluate(&wheel, &g1);
        assert_eq!(r.score, 0.0);
        assert!(r.gates.iter().any(|g| g.contains("reach_ground")), "{:?}", r.gates);
        let legs = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        let r = evaluate(&legs, &g1);
        assert!(r.gates.is_empty() && r.score > 0.0, "{:?}", r.gates);
        assert_eq!(evaluate(&legs, &g1).score, r.score, "deterministic");
    }
}
