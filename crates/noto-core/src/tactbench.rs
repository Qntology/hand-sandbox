//! The tactile N:N benchmark (v1.0, `docs/research/tactile.md` §5): the
//! finger pads, the grip loop and the soles of `contact` played out by the
//! engine — a lift, a pad pressed and pulled, a held object pulled by an
//! outside load, a walk's stance — and every answer compared with a
//! measurement from a source the engine does not use. Like `actbench`, it is
//! a reference-score test, not a timing run.
//!
//! Each scenario is scored in the four layers of `actbench` (outcome, time
//! series, part, coupling) with the same scoring: a likeness cell
//! `100 − 10·|z|`, an "at least" cell full above its floor, a cell the
//! engine does not model scores 0 and says "미구현", raw cells are reported
//! only; the scenario is the mean of its scored layers.
//!
//! The references compare the robot's silicone pads and soles with human
//! fingertips and feet: the human numbers are what "uncanny" is measured
//! against, not what the engine is set to. What keeps them from becoming
//! tuning targets: the engine files never name a scoring source (a test
//! scans them), every scenario runs the same engine constants, and the
//! held-out draws (`holdout`) use objects, rates and grounds the
//! development scenarios do not.

use serde::Serialize;

use crate::act::gait;
use crate::act::grasp::{self as g, GraspInput, LOOP_RISE_S};
use crate::act::material::{self, ItemMaterial};
use crate::actbench::{hand_of, refused_ko, AxCell, Layer};
use crate::balance::{Body, G, GRIP_SAFETY};
use crate::contact::{self, PadSurface};
use crate::effects::Rule;
use crate::surface::{Surface, SurfaceProps};

/// What a tactile scenario plays out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TxKind {
    /// The robot bends, closes its hand on an object and lifts it (`act::grasp`).
    Lift,
    /// One finger pad pressed on a face and dragged along it (`contact`).
    Pad,
    /// A pinch holds an object still while an outside load ramps up (`contact::ramp_hold`).
    Ramp,
    /// The soles over a walk's stances (`act::gait`, `contact::sole_contact`).
    Sole,
}

impl TxKind {
    pub fn id(self) -> &'static str {
        match self {
            TxKind::Lift => "lift",
            TxKind::Pad => "pad",
            TxKind::Ramp => "ramp",
            TxKind::Sole => "sole",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TxScenario {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub kind: TxKind,
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub size_m: f32,
    pub ground: Surface,
    pub slope_deg: f32,
    /// Cargo in the bay, as a share of the body mass.
    pub cargo_frac: f32,
    /// Load rates of a ramp, N/s, and the plateaus held, N.
    pub rates: &'static [f32],
    pub plateaus: &'static [f32],
    /// Monte Carlo draws.
    pub trials: u32,
    pub holdout: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TxResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub kind: TxKind,
    pub holdout: bool,
    pub skipped: Option<String>,
    pub cells: Vec<AxCell>,
    /// Mean cell score per layer (`Layer::ALL` order).
    pub layer_scores: [Option<f32>; 4],
    pub score: f32,
    pub gates: Vec<String>,
    /// No gate and every scored cell within two tolerances (the held-out rule).
    pub passed: bool,
    pub success_rate: f32,
}

// ----- references (benchmark side only; the engine never reads these) -----

/// Contact to lift-off of a 165 g or 330 g object in a thumb–index lift:
/// median 166 ms, interquartile range 90 ms (one participant's trials; the
/// tolerance is the IQR over 1.349, a normal's SD) [derived].
pub const LIFT_OFF_S: (f32, f32) = (0.166, 0.090 / 1.349);
pub const CITE_LIFT_OFF: &str = "엄지–검지 들기 접촉→들림 중앙값 166 ms, IQR 90 ms (Luciw, Jarocka & Edin 2014 WAY-EEG-GAL, 참가자 3; 허용폭 IQR/1.349, 유도)";
/// Fingertip contact area on flat glass at 1 N: 124–151 mm² over four
/// subjects (the middle ± half the range) [derived].
pub const AREA_1N_MM2: (f32, f32) = (137.5, 13.5);
/// Contact area grows as the normal force to the 0.4; the spread is what the
/// 0.5 N and 2 N area ranges allow (0.24–0.59) [derived].
pub const AREA_EXPONENT: (f32, f32) = (0.40, 0.17);
pub const CITE_AREA: &str = "손끝 접촉 면적 1 N에서 124–151 mm², 면적 ∝ 힘^0.4 (Delhaye, Lefèvre & Thonnard 2014, 유리판, n = 4; 허용폭은 피험자 범위에서 유도)";
/// Dynamic fingertip friction on glass at 2 N over that at 0.5 N: subject
/// means 1.29–1.54 against 1.92–2.41, the ratio of the middles with the
/// ranges' relative spread [derived].
pub const MU_RATIO_2N_05N: (f32, f32) = (0.65, 0.094);
pub const CITE_MU_RATIO: &str = "유리판 손끝 동마찰 0.5 N 1.92–2.41 → 2 N 1.29–1.54: 비 0.65 (Delhaye 2014; 허용폭은 범위에서 유도)";
/// Skin displacement before full slip: about 2 mm proximal–distal and 3 mm
/// radial–ulnar (the middle ± half the difference) [derived].
pub const PRESLIP_MM: (f32, f32) = (2.5, 0.5);
/// Stick-to-slip phase at 5–20 mm/s: 90–980 ms (the middle ± half the range) [derived].
pub const PARTIAL_SLIP_S: (f32, f32) = (0.535, 0.445);
/// Platform speed the partial-slip phase is compared at (the middle of 5, 10, 20 mm/s).
pub const DRAG_MPS: f32 = 0.010;
pub const CITE_SLIP_ONSET: &str = "완전 미끄럼 전 피부 변위 약 2 mm (근–원위)·3 mm (요–척측), 고착→미끄럼 90–980 ms (5–20 mm/s), 고착 비율은 접선력/미끄럼 힘에 거의 선형으로 감소 (Delhaye 2014, 380회)";
/// Grip response latency after the onset of an unexpected pull, by load
/// rate (N/s, s); no spread was given — the tolerance is a quarter of each
/// value (est).
pub const RAMP_LATENCY: [(f32, f32); 4] = [(2.0, 0.174), (4.0, 0.138), (8.0, 0.108), (32.0, 0.080)];
pub const LATENCY_TOL_SHARE: f32 = 0.25;

/// The latency reference at any load rate: the reported ones, linear in the
/// rate's inverse between them (a threshold plus a fixed delay is linear in
/// `1/rate`), held at the ends [derived].
pub fn latency_ref(rate_n_s: f32) -> (f32, f32) {
    let t = &RAMP_LATENCY;
    let x = 1.0 / rate_n_s.max(1e-3);
    let v = if x >= 1.0 / t[0].0 {
        t[0].1
    } else if x <= 1.0 / t[t.len() - 1].0 {
        t[t.len() - 1].1
    } else {
        let i = t.iter().position(|p| 1.0 / p.0 <= x).unwrap_or(t.len() - 1).max(1);
        let (a, b) = (t[i - 1], t[i]);
        let u = (x - 1.0 / a.0) / (1.0 / b.0 - 1.0 / a.0);
        a.1 + (b.1 - a.1) * u
    };
    (v, LATENCY_TOL_SHARE * v)
}
/// Safety margin `(GF − slip force)/GF` at the end of 1, 2 and 4 N load
/// plateaus; individual means ranged 18–68% (a quarter of the range as the
/// tolerance) [derived].
pub const PLATEAU_MARGIN: [(f32, f32); 3] = [(1.0, 0.44), (2.0, 0.40), (4.0, 0.39)];
pub const PLATEAU_MARGIN_TOL: f32 = 0.125;
pub const CITE_RAMP: &str = "예고 없는 당김 하중에 쥠 반응 지연 174/138/108/80 ms (2/4/8/32 N/s), 1/2/4 N 유지 끝 안전 여유 44/40/39% (개인 평균 18–68%) (Häger-Ross 1995 학위논문, Johansson 등 1992 요약; 지연 허용폭은 값의 25% 추정, 여유 허용폭은 범위/4)";
/// Load rate and hold of the plateau-margin trials (est: a middle rate of the ramps, a one-second hold).
pub const PLATEAU_RATE_N_S: f32 = 4.0;
pub const PLATEAU_HOLD_S: f32 = 1.0;
/// Regional peak plantar pressure in barefoot walking at 4.3 km/h (kPa):
/// heel 286 ± 45, medial forefoot 332 ± 90, lateral forefoot 214 ± 68,
/// medial midfoot 182 ± 85, lateral midfoot 150 ± 65. The robot's mass and
/// sole differ from a person's, so the comparison is the forefoot and
/// midfoot peaks over the heel's (the regions' means, the SDs carried by
/// error propagation) [derived].
pub const FOREFOOT_OVER_HEEL: (f32, f32) = (0.955, 0.25);
pub const MIDFOOT_OVER_HEEL: (f32, f32) = (0.58, 0.21);
pub const CITE_PLANTAR: &str = "맨발 4.3 km/h 최대 족저압 뒤꿈치 286±45, 앞발 안쪽 332±90·바깥 214±68, 중족 안쪽 182±85·바깥 150±65 kPa (Koo 2019, n = 20) — 뒤꿈치 대비 비로 비교 (질량·발 크기 무관)";
/// Medial-heel maximum force in walking: 45.6 ± 8.5% of body weight
/// (in-shoe, slippers). The engine's heel patch is the whole heel, medial
/// and lateral together: its half is compared (an even split, est).
pub const HEEL_FORCE_BW: (f32, f32) = (0.456, 0.085);
pub const HEEL_MEDIAL_SHARE: f32 = 0.5;
pub const CITE_HEEL_FORCE: &str = "보행 뒤꿈치 안쪽 최대 힘 체중의 45.6 ± 8.5% (Hessert 2005, 젊은 성인 9명, 슬리퍼 안 Pedar 깔창) — 엔진 뒤꿈치 패치는 안·바깥 한 덩어리라 절반과 비교 [추정]";
/// Peak force under the great-toe pad in walking about 30% BW and under the
/// second toe about 6% BW: the toes together carry at least the great toe's
/// share (the second toe's share as the tolerance) [derived].
pub const TOE_FORCE_BW: (f32, f32) = (0.30, 0.06);
pub const CITE_TOE_FORCE: &str = "보행 엄지발가락 패드 최대 힘 약 30% BW, 둘째 발가락 약 6% BW (Jacob 1989 학위논문) — 발가락 전체는 적어도 엄지 몫";
/// Toes in ground contact for about three quarters of the stance (no spread
/// given; the tolerance is est).
pub const TOE_CONTACT_SHARE: (f32, f32) = (0.75, 0.10);
pub const CITE_TOE_CONTACT: &str = "발가락이 입각기의 약 3/4 동안 지면 접촉 (Hughes, Clark & Klenerman 1990, n = 160; 허용폭 추정)";

// ----- scenarios -----

const NO_RATES: &[f32] = &[];

fn scen(id: &'static str, name_ko: &'static str, kind: TxKind) -> TxScenario {
    TxScenario { id, name_ko, kind, material: ItemMaterial::Cardboard, mass_kg: 0.33, size_m: 0.10, ground: Surface::Firm, slope_deg: 0.0, cargo_frac: 0.0, rates: NO_RATES, plateaus: NO_RATES, trials: 20, holdout: false }
}

/// The development scenarios.
pub fn library() -> Vec<TxScenario> {
    vec![
        scen("T1", "골판지 상자 0.33 kg 손으로 들기", TxKind::Lift),
        TxScenario { mass_kg: 2.4, size_m: 0.15, material: ItemMaterial::Wood, ground: Surface::Gravel, ..scen("T2", "원목 2.4 kg 들기 (쥠–하중 결합)", TxKind::Lift) },
        TxScenario { trials: 1, ..scen("T3", "손가락 패드 누르고 끌기 (1 N)", TxKind::Pad) },
        TxScenario { trials: 1, rates: &[2.0, 4.0, 8.0, 32.0], plateaus: &[1.0, 2.0, 4.0], ..scen("T4", "쥔 물체를 당기는 하중에 쥠 반응", TxKind::Ramp) },
        TxScenario { trials: 6, ..scen("T5", "단단한 바닥 보행 발바닥", TxKind::Sole) },
        TxScenario { trials: 6, ground: Surface::WetSand, slope_deg: 8.0, ..scen("T6", "젖은 모래 8° 오르기 발바닥", TxKind::Sole) },
    ]
}

/// Objects and grounds the held-out draws use (and load rates of 3 and 16 N/s, scored against the
/// reference's interpolation); none is a development condition (a test checks it).
pub const HOLDOUT_MATERIALS: [ItemMaterial; 3] = [ItemMaterial::Stone, ItemMaterial::Coal, ItemMaterial::IronOre];
pub const HOLDOUT_GROUNDS: [Surface; 3] = [Surface::Grass, Surface::Dirt, Surface::Sand];
pub const HOLDOUT_CARGO: (f32, f32) = (0.03, 0.07);

/// Held-out scenarios for one evaluation seed: lifts of other objects, pulls
/// on other materials, walks on other grounds with a little cargo. Counted
/// pass / fail, never tuned against.
pub fn holdout(seed: u64, count: usize) -> Vec<TxScenario> {
    let mut s = seed ^ 0xBB67_AE85_84CA_A73B;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..count)
        .map(|i| {
            let m = HOLDOUT_MATERIALS[(next() * 3.0) as usize % 3];
            let ground = HOLDOUT_GROUNDS[(next() * 3.0) as usize % 3];
            let cargo = HOLDOUT_CARGO.0 + (HOLDOUT_CARGO.1 - HOLDOUT_CARGO.0) * next();
            let mass = 0.5 + 1.5 * next();
            let kind = [TxKind::Lift, TxKind::Ramp, TxKind::Sole][i % 3];
            let id: &'static str = Box::leak(format!("Z{}-{i}", seed % 1000).into_boxed_str());
            let name: &'static str = Box::leak(match kind {
                TxKind::Lift => format!("들기 {} {mass:.1} kg · {}", m.label_ko(), ground.label_ko()),
                TxKind::Ramp => format!("당김 하중 · {}", m.label_ko()),
                _ => format!("보행 발바닥 · {} · 짐 {:.0}%", ground.label_ko(), cargo * 100.0),
            }
            .into_boxed_str());
            TxScenario { id, name_ko: name, kind, material: m, mass_kg: mass, size_m: 0.10, ground, slope_deg: 0.0, cargo_frac: if kind == TxKind::Sole { cargo } else { 0.0 }, rates: &[3.0, 16.0], plateaus: &[2.0], trials: if kind == TxKind::Sole { 4 } else { 10 }, holdout: true }
        })
        .collect()
}

// ----- scoring (the `actbench` rules) -----

fn judge(c: &mut AxCell) {
    let Some(v) = c.value else {
        if !matches!(c.rule, Rule::Raw) {
            c.score = Some(0.0);
        }
        return;
    };
    match c.rule {
        Rule::Like { reference, tolerance } => {
            let z = (v - reference) / tolerance.max(1e-6);
            c.z = Some(z);
            c.score = Some((100.0 - 10.0 * z.abs()).max(0.0));
        }
        Rule::AtLeast { reference, tolerance } => {
            let z = ((v - reference) / tolerance.max(1e-6)).min(0.0);
            c.z = Some(z);
            c.score = Some((100.0 - 10.0 * z.abs()).max(0.0));
        }
        Rule::AtMost { limit } => {
            if v > limit {
                c.gate = Some(format!("{} {v:.2} > {limit:.2}", c.axis));
            }
        }
        Rule::Raw => {}
    }
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

fn finish(s: &TxScenario, cells: Vec<AxCell>, success_rate: f32, skipped: Option<String>) -> TxResult {
    let gates: Vec<String> = cells.iter().filter_map(|c| c.gate.clone()).collect();
    let layer_scores: [Option<f32>; 4] = std::array::from_fn(|i| {
        let sc: Vec<f32> = cells.iter().filter(|c| c.layer == Layer::ALL[i]).filter_map(|c| c.score).collect();
        mean(&sc)
    });
    let scored: Vec<f32> = layer_scores.iter().flatten().copied().collect();
    let score = if skipped.is_some() || !gates.is_empty() { 0.0 } else { mean(&scored).unwrap_or(0.0) };
    let within = cells.iter().all(|c| c.score.is_none() || c.z.map_or(c.score.unwrap_or(0.0) > 0.0, |z| z.abs() <= 2.0));
    TxResult { id: s.id, name_ko: s.name_ko, kind: s.kind, holdout: s.holdout, passed: skipped.is_none() && gates.is_empty() && within && !scored.is_empty(), skipped, gates, layer_scores, score, cells, success_rate }
}

/// Runs one scenario on a robot.
pub fn evaluate(body: &Body, s: &TxScenario) -> TxResult {
    match s.kind {
        TxKind::Lift => eval_lift(body, s),
        TxKind::Pad => eval_pad(s),
        TxKind::Ramp => eval_ramp(s),
        TxKind::Sole => eval_sole(body, s),
    }
}

/// The aim error the benchmark's robot has (as `actbench`: a stereo head 1.05 m from the object, a 5 or 10 mm arm).
fn aim_sigma(body: &Body) -> f32 {
    let depth = material::stereo_depth_sigma(1.09, 424.0, 0.05, 0.08);
    let arm = if body.arm.shoulder.backdrivable { 0.005 } else { 0.010 };
    (depth * depth + arm * arm).sqrt()
}

/// A lift's tactile cells. The only scored one, contact to lift-off, is the
/// grasp's timing by construction (the sensor's two-frame confirmation and
/// two loop rises: 0.208 s whatever the object, body or draw), so a lift
/// scenario's score is effectively a constant; the grip–load coupling, the
/// safety margin and the finger shares are recorded only — the research pack
/// has no reference for them apart from the ones `actbench` already scores
/// in its grasp scenarios.
fn eval_lift(body: &Body, s: &TxScenario) -> TxResult {
    let hand = hand_of(body);
    let size = [s.size_m; 3];
    // The hands the planner takes (`top_hands`), and no rollout when two hands do not hold even the weight.
    let th = g::top_hands(&hand, size, s.mass_kg, material::pad_on(s.material).mean);
    if th.hands == 0 {
        return finish(s, vec![], 0.0, Some(refused_ko(&th)));
    }
    let runs: Vec<crate::act::ActRollout> = (0..s.trials as u64).map(|seed| g::grasp(&GraspInput::front(body, &hand, s.material, s.mass_kg, size, s.ground, th.hands == 2, 0.0, aim_sigma(body), seed))).collect();
    if runs.iter().all(|r| matches!(r.outcome.code, Some("reach_ground") | Some("reach_side"))) {
        return finish(s, vec![], 0.0, Some("바닥의 물체에 손이 닿지 않음 (actbench reach_ground)".into()));
    }
    // The tactile part starts at the contact: the trials whose fingers closed on the object.
    let held: Vec<&crate::act::ActRollout> = runs.iter().filter(|r| r.metric("lift_off_s").is_some()).collect();
    let ok = runs.iter().filter(|r| r.outcome.ok).count() as f32 / runs.len().max(1) as f32;
    let m = |k: &str| mean(&held.iter().filter_map(|r| r.metric(k)).collect::<Vec<_>>());
    let share = |code: &str| runs.iter().filter(|r| r.outcome.code == Some(code)).count() as f32 / runs.len().max(1) as f32;
    let incipient = held.iter().filter(|r| r.metric("incipient_slips").unwrap_or(0.0) > 0.0).count() as f32 / held.len().max(1) as f32;
    let cells = vec![
        cell(Layer::Outcome, "success_rate", Some("hand"), Some(ok), Rule::Raw, ""),
        cell(Layer::Outcome, "slip_rate", Some("hand"), Some(share("grasp_slip")), Rule::Raw, ""),
        cell(Layer::Outcome, "incipient_share", Some("hand"), Some(incipient), Rule::Raw, ""),
        cell(Layer::TimeSeries, "lift_off_s", Some("hand"), m("lift_off_s"), like(LIFT_OFF_S), CITE_LIFT_OFF),
        cell(Layer::TimeSeries, "gf_lf_corr", Some("hand"), m("gf_lf_corr"), Rule::Raw, ""),
        cell(Layer::TimeSeries, "gf_lead_s", Some("hand"), m("gf_lead_s"), Rule::Raw, ""),
        cell(Layer::Part, "mu_pad", Some("hand"), m("mu_pad"), Rule::Raw, ""),
        cell(Layer::Part, "stick_min", Some("hand"), m("stick_min"), Rule::Raw, ""),
        cell(Layer::Part, "squeeze_cap_n", Some("hand_joints"), m("squeeze_cap_n"), Rule::Raw, ""),
        cell(Layer::Coupling, "grip>load:safety_margin", Some("hand"), m("safety_margin"), Rule::Raw, ""),
        cell(Layer::Coupling, "hand>finger:share_max", Some("hand_joints"), m("finger_share_max"), Rule::Raw, ""),
        cell(Layer::Coupling, "hand>finger:share_min", Some("hand_joints"), m("finger_share_min"), Rule::Raw, ""),
    ];
    finish(s, cells, ok, None)
}

/// One pad of the robot's fingers on the scenario's material: its patch,
/// its friction over the force, and its creep before it slides.
fn eval_pad(s: &TxScenario) -> TxResult {
    let mu0 = material::pad_on(s.material).mean;
    let area = |n: f32| contact::patch_area_m2(n) * 1e6;
    let exponent = (area(2.0) / area(0.5)).ln() / 4f32.ln();
    let mu = |n: f32| contact::pad_mu(mu0, n, PadSurface::Dry);
    let creep = contact::preslip_displacement_m(mu(1.0) * 1.0, mu(1.0), 1.0);
    let stick_half = contact::stick_ratio(0.5, 1.0);
    let cells = vec![
        cell(Layer::Outcome, "slip_force_1n", Some("hand"), Some(mu(1.0)), Rule::Raw, ""),
        cell(Layer::Part, "contact_area_1n_mm2", Some("hand"), Some(area(1.0)), like(AREA_1N_MM2), CITE_AREA),
        cell(Layer::Part, "area_exponent", Some("hand"), Some(exponent), like(AREA_EXPONENT), CITE_AREA),
        cell(Layer::Part, "mu_2n_over_05n", Some("hand"), Some(mu(2.0) / mu(0.5)), like(MU_RATIO_2N_05N), CITE_MU_RATIO),
        cell(Layer::Part, "preslip_mm", Some("hand"), Some(creep * 1000.0), like(PRESLIP_MM), CITE_SLIP_ONSET),
        cell(Layer::Part, "stick_at_half_slip_force", Some("hand"), Some(stick_half), Rule::Raw, CITE_SLIP_ONSET),
        cell(Layer::TimeSeries, "partial_slip_s", Some("hand"), Some(creep / DRAG_MPS), like(PARTIAL_SLIP_S), CITE_SLIP_ONSET),
    ];
    finish(s, cells, 1.0, None)
}

/// A two-pad pinch (one pad on each face) at the robot's seating preload,
/// on the scenario's material, pulled by ramps and held at plateaus.
fn eval_ramp(s: &TxScenario) -> TxResult {
    let mu0 = material::pad_on(s.material).mean;
    let hold = |rate: f32, plateau: f32, hold_s: f32| contact::ramp_hold(mu0, PadSurface::Dry, mu0, GRIP_SAFETY, contact::PRELOAD_N, 1, rate, plateau, hold_s, LOOP_RISE_S);
    let mut cells = Vec::new();
    let mut slipped = 0u32;
    let mut runs = 0u32;
    let mut stick_min = 1.0f32;
    const LAT_AXES: [&str; 4] = ["latency_s@2", "latency_s@4", "latency_s@8", "latency_s@32"];
    for &rate in s.rates {
        let h = hold(rate, s.plateaus.last().copied().unwrap_or(2.0), 0.3);
        runs += 1;
        slipped += h.slipped as u32;
        stick_min = stick_min.min(h.stick_min);
        let axis = RAMP_LATENCY.iter().position(|x| (x.0 - rate).abs() < 1e-3).map_or("latency_s", |i| LAT_AXES[i]);
        cells.push(cell(Layer::TimeSeries, axis, Some("hand"), h.latency_s, like(latency_ref(rate)), CITE_RAMP));
    }
    const SM_AXES: [&str; 3] = ["load>grip:safety_margin@1n", "load>grip:safety_margin@2n", "load>grip:safety_margin@4n"];
    for &p in s.plateaus {
        let h = hold(PLATEAU_RATE_N_S, p, PLATEAU_HOLD_S);
        runs += 1;
        slipped += h.slipped as u32;
        stick_min = stick_min.min(h.stick_min);
        let r = PLATEAU_MARGIN.iter().position(|x| (x.0 - p).abs() < 1e-3);
        let axis = r.map_or("load>grip:safety_margin", |i| SM_AXES[i]);
        let rule = r.map_or(Rule::Raw, |i| Rule::Like { reference: PLATEAU_MARGIN[i].1, tolerance: PLATEAU_MARGIN_TOL });
        cells.push(cell(Layer::Coupling, axis, Some("hand"), (!h.slipped).then_some(h.safety_margin), rule, CITE_RAMP));
    }
    let rate_ok = 1.0 - slipped as f32 / runs.max(1) as f32;
    cells.insert(0, cell(Layer::Outcome, "held_share", Some("hand"), Some(rate_ok), Rule::Raw, ""));
    cells.push(cell(Layer::Part, "stick_min", Some("hand"), Some(stick_min), Rule::Raw, ""));
    finish(s, cells, rate_ok, None)
}

fn eval_sole(body: &Body, s: &TxScenario) -> TxResult {
    if !body.legged {
        return finish(s, vec![], 0.0, Some("발이 없음 (바퀴·궤도)".into()));
    }
    let props: SurfaceProps = *s.ground.props();
    let cargo = s.cargo_frac * body.mass_kg();
    let speed = body.drive_limits(&body.base.spec(), cargo).max_speed_mps;
    let runs: Vec<crate::act::ActRollout> = (0..s.trials as u64).map(|seed| gait::walk(body, &props, None, s.slope_deg, cargo, speed, 20, seed).0).collect();
    let falls = runs.iter().filter(|r| !r.outcome.ok).count() as f32;
    let rate = 1.0 - falls / runs.len().max(1) as f32;
    let m = |k: &str| mean(&runs.iter().filter_map(|r| r.metric(k)).collect::<Vec<_>>());
    let heel = m("heel_peak_kpa");
    let over_heel = |k: &str| m(k).zip(heel).filter(|(_, h)| *h > 1e-6).map(|(x, h)| x / h);
    let cells = vec![
        cell(Layer::Outcome, "no_fall_rate", Some("foot"), Some(rate), Rule::Raw, ""),
        cell(Layer::TimeSeries, "toe_contact_share", Some("foot"), m("toe_contact_share"), like(TOE_CONTACT_SHARE), CITE_TOE_CONTACT),
        cell(Layer::TimeSeries, "heel_off_frac", Some("foot"), m("heel_off_frac"), Rule::Raw, ""),
        cell(Layer::TimeSeries, "toe_off_ds_frac", Some("foot"), m("toe_off_ds_frac"), Rule::Raw, ""),
        cell(Layer::Part, "forefoot_over_heel_peak", Some("foot"), over_heel("meta_peak_kpa"), like(FOREFOOT_OVER_HEEL), CITE_PLANTAR),
        cell(Layer::Part, "midfoot_over_heel_peak", Some("foot"), over_heel("midfoot_peak_kpa"), like(MIDFOOT_OVER_HEEL), CITE_PLANTAR),
        cell(Layer::Part, "heel_medial_peak_bw", Some("foot"), m("heel_peak_bw").map(|x| HEEL_MEDIAL_SHARE * x), like(HEEL_FORCE_BW), CITE_HEEL_FORCE),
        cell(Layer::Part, "toe_peak_bw", Some("foot"), m("toe_peak_bw"), Rule::AtLeast { reference: TOE_FORCE_BW.0, tolerance: TOE_FORCE_BW.1 }, CITE_TOE_FORCE),
        cell(Layer::Part, "heel_peak_kpa", Some("foot"), heel, Rule::Raw, ""),
        cell(Layer::Part, "toe_deg_max", Some("foot_joints"), m("toe_deg_max"), Rule::Raw, ""),
        cell(Layer::Coupling, "ground>sole:cone_use_max", Some("foot"), m("cone_use_max"), Rule::Raw, ""),
        cell(Layer::Coupling, "sole>toe:cop_reach_fl", Some("foot"), m("cop_reach_fl"), Rule::Raw, ""),
        cell(Layer::Coupling, "sole:stick_min", Some("foot"), m("sole_stick_min"), Rule::Raw, ""),
        cell(Layer::Coupling, "toe>push:force_bw", Some("foot"), m("toe_push_n").map(|n| n / ((body.mass_kg() + cargo) * G)), Rule::Raw, ""),
    ];
    finish(s, cells, rate, None)
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
            assert!(!HOLDOUT_MATERIALS.contains(&s.material), "{}: a development scenario on a held-out material", s.id);
            assert!(!HOLDOUT_GROUNDS.contains(&s.ground), "{}: a development scenario on a held-out ground", s.id);
            assert!(!(HOLDOUT_CARGO.0..=HOLDOUT_CARGO.1).contains(&s.cargo_frac), "{}", s.id);
        }
        let h = holdout(20261027, 6);
        assert_eq!(h, holdout(20261027, 6), "a seed gives the same draws");
        assert!(h.iter().all(|s| s.holdout && HOLDOUT_MATERIALS.contains(&s.material)));
        let dev_rates: Vec<f32> = library().iter().flat_map(|s| s.rates.iter().copied()).collect();
        assert!(h.iter().flat_map(|s| s.rates.iter()).all(|r| !dev_rates.contains(r)), "held-out load rates are not the development ones");
        // Between the reported rates the latency reference follows 1/rate.
        assert!((latency_ref(4.0).0 - 0.138).abs() < 1e-6 && latency_ref(16.0).0 < 0.108 && latency_ref(16.0).0 > 0.080);
    }

    #[test]
    fn pads_and_pulls_score_and_a_wheeled_body_skips_the_soles() {
        let wheel = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let lib = library();
        let sole = lib.iter().find(|s| s.kind == TxKind::Sole).unwrap();
        assert!(evaluate(&wheel, sole).skipped.is_some());
        for s in lib.iter().filter(|s| matches!(s.kind, TxKind::Pad | TxKind::Ramp)) {
            let r = evaluate(&wheel, s);
            assert!(r.skipped.is_none() && r.layer_scores.iter().any(|x| x.is_some()), "{}: {:?}", s.id, r.layer_scores);
            assert!(r.cells.iter().all(|c| c.value.map_or(true, f32::is_finite)));
            assert_eq!(evaluate(&wheel, s), r, "deterministic");
        }
    }
}
