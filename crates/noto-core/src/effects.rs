//! Environment N:N benchmark of the five body effects (slip, load, fatigue,
//! joint load, posture stability): scenarios on the world's grounds and one
//! laboratory floor, each cell compared with a measurement from a source the
//! engine does not use. Sources, derivations and the improvement record are
//! in `docs/research/effects.md` §4–§6.
//!
//! Scoring follows [`crate::nn`]:
//!
//! 1. A likeness cell scores `100 − 10·|z|`, `z = (value − ref) / tol`.
//! 2. An "at least" cell scores 100 at or above its reference and like a
//!    likeness cell below it (a robot that replaces manual work should not tire
//!    sooner than the person).
//! 3. Gates come first: sliding, tipping, a joint over its peak or its
//!    continuous rating (maxon sizing rules), cargo over the capacity, or a
//!    traction the experiments never measured on that ground zero the scenario.
//! 4. A cell the engine does not model scores 0 and says so ("미구현"): a
//!    missing model is not a pass.
//! 5. The five effects weigh the same (no source ranks them); an effect is the
//!    mean of its scored cells, the scenario the mean of its scored effects.
//!
//! What keeps it from being tuned to (research §5): the engine never names a
//! benchmark source (a test scans its files), every scenario runs the same
//! engine with the same parameters, and the held-out draws ([`holdout`]) use
//! grounds, slopes and loads the development scenarios do not.

use serde::Serialize;

use crate::balance::{reference, Body, FeedForward, Posture, RefCtx, Source};
use crate::nn::PLANNER_MIN_RAMP_S;
use crate::parts::Segment;
use crate::surface::{self, Surface, SurfaceProps};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Slip,
    Load,
    Fatigue,
    JointLoad,
    Stability,
}

impl Effect {
    pub const ALL: [Effect; 5] = [Effect::Slip, Effect::Load, Effect::Fatigue, Effect::JointLoad, Effect::Stability];

    pub fn id(self) -> &'static str {
        match self {
            Effect::Slip => "slip",
            Effect::Load => "load",
            Effect::Fatigue => "fatigue",
            Effect::JointLoad => "joint_load",
            Effect::Stability => "stability",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Effect::Slip => "미끄러짐",
            Effect::Load => "하중",
            Effect::Fatigue => "피로",
            Effect::JointLoad => "관절 부하",
            Effect::Stability => "자세 안정",
        }
    }
}

/// How a cell is judged.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    /// Like the reference: `100 − 10·|z|`.
    Like { reference: f32, tolerance: f32 },
    /// At least the reference: 100 above it, `100 − 10·|z|` below.
    AtLeast { reference: f32, tolerance: f32 },
    /// An engineering or physical limit: above it the scenario is gated.
    AtMost { limit: f32 },
    /// Reported only.
    Raw,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FxCell {
    pub effect: Effect,
    pub axis: &'static str,
    /// `None`: the engine does not model this.
    pub value: Option<f32>,
    pub rule: Rule,
    pub cite: &'static str,
    pub z: Option<f32>,
    pub score: Option<f32>,
    /// Set when an `AtMost` limit is broken.
    pub gate: Option<String>,
}

/// What the robot does in a scenario.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Walk or drive along the slope with the cargo.
    Travel,
    /// Strike a rock with the working arm for this many minutes.
    Mine { minutes: f32 },
}

/// One ground the robot stands on: a world ground id, or a laboratory floor
/// with a measured available friction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ground {
    World { id: &'static str },
    Lab { id: &'static str, acof: f32 },
}

impl Ground {
    pub fn id(&self) -> &'static str {
        match self {
            Ground::World { id } | Ground::Lab { id, .. } => id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FxScenario {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub ground: Ground,
    /// Along the travel direction, positive uphill.
    pub slope_deg: f32,
    /// Cargo as a fraction of the body mass.
    pub cargo_frac: f32,
    pub task: Task,
    /// Only bases with legs can do it (a walking-gait measurement).
    pub needs_legs: bool,
    pub holdout: bool,
}

/// What the engine says about one scenario. `None`: not modelled.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Probe {
    /// Steepest slope the robot can climb on this ground, degrees (negative:
    /// it cannot move on level ground).
    pub climb_deg: f32,
    /// Steepest slope it can go down without sliding (friction only), degrees.
    pub descend_deg: f32,
    /// Friction needed at the task's most demanding instant over friction available.
    pub utilisation: Option<f32>,
    /// Friction needed on this slope and on level ground (legs: shear over normal at the foot).
    pub required_mu: Option<f32>,
    pub required_mu_level: Option<f32>,
    pub slip_probability: Option<f32>,
    /// Forward force left over weight once rolling resistance or the gait's own shear is paid.
    pub net_traction: Option<f32>,
    /// Locomotion energy on this ground over firm level ground.
    pub energy_factor: Option<f32>,
    pub capacity_kg: f32,
    /// Top speed with the scenario's cargo over empty.
    pub speed_ratio: f32,
    pub margin: f32,
    /// The robot slides before it tips on this slope.
    pub slide_first: Option<bool>,
    /// Time until the working arm derates, seconds (infinite: never).
    pub endurance_s: Option<f32>,
    /// Share of the working shoulder's peak torque the task uses.
    pub work_frac: f32,
    /// Life used per hour of the task (1 = worn out).
    pub wear_per_h: Option<f32>,
    pub peak_ratio: f32,
    /// Torque over the continuous rating of the worst joint (the steady RMS of a held task).
    pub effort: f32,
    pub worst_joint: String,
    /// Arm or leg peak ratio of the task posture, for the human comparison.
    pub limb_peak_ratio: f32,
    pub posture_source: Source,
    pub single_support: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FxResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub holdout: bool,
    pub skipped: Option<String>,
    pub cells: Vec<FxCell>,
    /// Mean cell score per effect (`Effect::ALL` order), `None` without a scored cell.
    pub effect_scores: [Option<f32>; 5],
    pub score: f32,
    pub gates: Vec<String>,
    /// No gate and every scored cell within two tolerances (the held-out rule).
    pub passed: bool,
    pub probe: Option<Probe>,
}

// ----- references (benchmark side only; the engine never reads these) -----

/// RCOF rise of walking on an incline over level walking: +0.14 at 10°,
/// +0.25 at 20° (Hanson, Redfern & Mazumdar 1999, Ergonomics 42:1619, as
/// cited by Beschorner 2016); 15° interpolated. Tolerance: SD of the level
/// peak RCOF, 0.027 (Beschorner, Albert & Redfern 2016, n = 31).
pub fn ramp_rcof_rise(slope_deg: f32) -> Option<(f32, f32)> {
    let s = slope_deg.abs();
    if !(5.0..=20.0).contains(&s) {
        return None;
    }
    let v = if s <= 10.0 { 0.14 * s / 10.0 } else { 0.14 + (0.25 - 0.14) * (s - 10.0) / 10.0 };
    Some((v, 0.027))
}

pub const CITE_RAMP: &str = "경사 보행 RCOF 상승 +0.14 (10°), +0.25 (20°) (Hanson 1999, Beschorner 2016 인용), 허용폭 수평 RCOF SD 0.027 (Beschorner 2016)";
/// Available friction of the contaminated laboratory floor: ACOF 0.149–0.179
/// (work boots on vinyl with 50% glycerol, Beschorner 2016), the midpoint.
pub const LAB_GLYCEROL_ACOF: f32 = 0.164;
/// Share of unexpected exposures to that floor that ended in a slip: 38%
/// (Beschorner 2016, n = 31); tolerance the binomial SD √(0.38·0.62/31).
pub const SLIP_FRACTION: (f32, f32) = (0.38, 0.087);
pub const CITE_SLIP: &str = "글리세롤 바닥(ACOF 0.149–0.179) 예고 없는 노출의 38%가 미끄러짐 (Beschorner 2016, n = 31), 허용폭 이항 SD";
/// Drawbar pull over weight of a rigid grousered rover wheel on loose sand:
/// always below 0.3, and 0.052 / 0.080 at 20% / 60% slip at 1 m/s
/// (Rodríguez-Martínez, Buse, Van Winnendael & Yoshida 2019, ISTVS).
pub const SAND_DPW_CEILING: f32 = 0.3;
pub const SAND_DPW: (f32, f32) = (0.066, 0.028);
pub const CITE_SAND_DPW: &str = "모래 위 강성 바퀴 견인/하중 < 0.3, 1 m/s에서 0.052–0.080 (Rodríguez-Martínez 2019)";
/// Walking on sand costs 2.1–2.7× the energy of a hard surface (Lejeune,
/// Willems & Heglund 1998, J Exp Biol 201:2071): midpoint ± half-range.
pub const SAND_ENERGY: (f32, f32) = (2.4, 0.3);
pub const CITE_SAND_ENERGY: &str = "모래 보행 에너지 2.1–2.7배 (Lejeune 1998)";
/// Load carried in a sustained march: fighting load 30% of body mass,
/// approach march 45% (US Army doctrine, Knapik & Reynolds, Borden Institute).
pub const CARRY_FRAC: (f32, f32) = (0.30, 0.15);
pub const CITE_CARRY: &str = "지속 운반 하중 체중의 30% (전투), 45% (접근 행군) (Knapik & Reynolds)";
/// Self-selected walking speed with 40% of body mass over unloaded: 4.6 /
/// 4.75 km/h, SD 0.5 km/h (Middleton 2022, IJERPH 19:3927, n = 30).
pub const LOAD_SPEED: (f32, f32) = (0.968, 0.105);
pub const CITE_LOAD_SPEED: &str = "체중 40% 짐의 보행 속도 4.6 / 4.75 km/h, SD 0.5 (Middleton 2022, n = 30)";
pub const CITE_ENDURANCE: &str = "어깨 지구 시간 ET = 14.86·f^-1.83 s (Frey Law & Avin 2010), 예상 변동 29–47% (Frey-Law 2012) → 허용폭 38%";
pub const CITE_SIZING: &str = "모터 선정 규칙: 연속 정격 > RMS 토크, 정지 토크 > 피크 (maxon Product Range 2026/27 p. 95)";

/// Endurance time of a person holding `f` of the shoulder's maximum, seconds
/// (Frey Law & Avin 2010 power model, shoulder).
pub fn shoulder_endurance_s(f: f32) -> f32 {
    14.86 * f.clamp(0.01, 1.0).powf(-1.83)
}

// ----- scenarios -----

fn scen(id: &'static str, name_ko: &'static str, ground: Ground, slope_deg: f32, cargo_frac: f32, task: Task, needs_legs: bool) -> FxScenario {
    FxScenario { id, name_ko, ground, slope_deg, cargo_frac, task, needs_legs, holdout: false }
}

/// The development scenarios (research §4.2).
pub fn library() -> Vec<FxScenario> {
    vec![
        scen("E1", "모래 해변 가로지르기 (짐 10%)", Ground::World { id: "sand" }, 0.0, 0.10, Task::Travel, false),
        scen("E2", "젖은 모래 10° 오르기 (짐 10%)", Ground::World { id: "wet_sand" }, 10.0, 0.10, Task::Travel, false),
        scen("E3", "오염된 실험실 바닥 걷기", Ground::Lab { id: "lab_glycerol", acof: LAB_GLYCEROL_ACOF }, 0.0, 0.0, Task::Travel, true),
        scen("E4", "체중 40% 짐 들고 바위 지면 걷기", Ground::World { id: "rock" }, 0.0, 0.40, Task::Travel, false),
        scen("E5", "바위 채취 10분", Ground::World { id: "rock" }, 0.0, 0.0, Task::Mine { minutes: 10.0 }, false),
        scen("E6", "풀밭 15° 오르기", Ground::World { id: "grass" }, 15.0, 0.0, Task::Travel, false),
    ]
}

/// World grounds the held-out draws use, and the ranges they draw from. None
/// of them is a development condition (slope 0 / 10 / 15°, cargo 0 / 10 /
/// 40%); a test checks it.
pub const HOLDOUT_GROUNDS: [&str; 4] = ["dirt", "gravel", "grass", "wet_sand"];
pub const HOLDOUT_SLOPE_DEG: [(f32, f32); 2] = [(4.0, 8.0), (11.5, 13.5)];
pub const HOLDOUT_CARGO: [(f32, f32); 2] = [(0.15, 0.25), (0.45, 0.50)];

/// Held-out scenarios for one evaluation seed: random ground, slope (up or
/// down) and cargo. Counted pass / fail, never tuned against.
pub fn holdout(seed: u64, count: usize) -> Vec<FxScenario> {
    let mut s = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let pick = |r: (f32, f32), u: f32| r.0 + (r.1 - r.0) * u;
    (0..count)
        .map(|i| {
            let ground = HOLDOUT_GROUNDS[(next() * 4.0) as usize % 4];
            let slope = pick(HOLDOUT_SLOPE_DEG[(next() * 2.0) as usize % 2], next()) * if next() < 0.5 { 1.0 } else { -1.0 };
            let cargo = pick(HOLDOUT_CARGO[(next() * 2.0) as usize % 2], next());
            let id: &'static str = Box::leak(format!("X{}-{i}", seed % 1000).into_boxed_str());
            let name: &'static str = Box::leak(format!("{ground} {slope:+.1}° 짐 {:.0}%", cargo * 100.0).into_boxed_str());
            FxScenario { id, name_ko: name, ground: Ground::World { id: ground }, slope_deg: slope, cargo_frac: cargo, task: Task::Travel, needs_legs: false, holdout: true }
        })
        .collect()
}

/// The world's condition as a scenario: the ground, slope and cargo of a
/// posture; mining when the arm is raised to strike, travelling while the base
/// walks or drives. `None` for the other states (standing, lifting).
pub fn live_scenario(body: &Body, post: &Posture) -> Option<FxScenario> {
    let task = match post.source {
        Source::ArmRaise => Task::Mine { minutes: 1.0 },
        _ if post.propelling() => Task::Travel,
        _ => return None,
    };
    Some(FxScenario { id: "live", name_ko: "월드 상태", ground: Ground::World { id: post.ground.id() }, slope_deg: post.slope_deg, cargo_frac: post.cargo_kg / body.mass_kg().max(1e-3), task, needs_legs: false, holdout: false })
}

// ----- the engine side -----

/// The ground's properties: a world ground from the table, a laboratory
/// floor as the firm floor with its measured friction.
pub fn ground_props(g: &Ground) -> (Surface, SurfaceProps) {
    match *g {
        Ground::World { id } => {
            let s = Surface::from_id(id).unwrap_or_default();
            (s, *s.props())
        }
        Ground::Lab { id, acof } => (Surface::Firm, SurfaceProps { id, label_ko: "실험실 바닥", mu_foot: acof, mu_wheel: acof, mu_track: acof, ..*Surface::Firm.props() }),
    }
}

/// The task's most demanding instant and its sustained posture: driving off
/// from a stop and cruising for a wheeled base, single support for legs (both),
/// the strike for mining (both).
fn task_postures(body: &Body, s: &FxScenario, cargo: f32, ground: Surface) -> (Posture, Posture) {
    match s.task {
        Task::Mine { .. } => {
            let p = Posture { slope_deg: s.slope_deg, ground, ..Posture::arm_raise(cargo) };
            (p, p)
        }
        Task::Travel if body.legged() => {
            let p = Posture { source: Source::FootLift, single_support: true, slope_deg: s.slope_deg, ground, moving: true, ..Posture::quiet(cargo) };
            (p, p)
        }
        Task::Travel => {
            let a = body.planned_accel(&body.base.spec(), s.slope_deg, cargo, ground, 1.0, PLANNER_MIN_RAMP_S);
            let q = Posture::quiet(cargo);
            let cruise = Posture { source: if s.slope_deg.abs() >= 3.0 { Source::Slope } else { q.source }, slope_deg: s.slope_deg, ground, moving: true, ..q };
            (Posture { source: Source::Accel, accel_mps2: a, ..cruise }, cruise)
        }
    }
}

/// Asks the engine about a scenario.
pub fn probe(body: &Body, s: &FxScenario) -> Probe {
    let spec = body.base.spec();
    let cargo = s.cargo_frac * body.mass_kg();
    let limits = body.drive_limits(&spec, cargo);
    let empty = body.drive_limits(&spec, 0.0);
    let (ground, props) = ground_props(&s.ground);
    let (post, steady) = task_postures(body, s, cargo, ground);
    let rep = body.solve(&post, &FeedForward::default(), 1e3);
    let held = body.solve(&steady, &FeedForward::default(), 1e3);
    let limbs = |r: &crate::balance::BalanceReport| r.joints.iter().filter(|j| !j.joint.starts_with("grip")).cloned().collect::<Vec<_>>();
    let worst = limbs(&rep).into_iter().max_by(|a, b| a.peak_ratio.total_cmp(&b.peak_ratio));
    let hot = limbs(&held).into_iter().max_by(|a, b| a.effort.total_cmp(&b.effort));
    let m = body.mass_kg() + cargo;
    let drive = (!body.legged()).then(|| body.chassis.drive_force_n / m);
    let travel = matches!(s.task, Task::Travel);
    let c = surface::contact(body.base, &props, post.slope_deg, post.accel_mps2, travel);
    let level = surface::contact(body.base, &props, 0.0, if body.legged() { 0.0 } else { post.accel_mps2 }, travel);
    // Statics: on a slope the robot tips at atan(d / h) and slides past the friction limit.
    let tip_deg = (0.5 * (rep.support[1] - rep.support[0]) / rep.com[2].max(0.1)).atan().to_degrees();
    let slide_first = surface::friction_slope_limit_deg(body.base, &props) < tip_deg;
    let shoulder = rep.joint("shoulder_r");
    let (work_frac, endurance) = match (s.task, shoulder) {
        (Task::Mine { .. }, Some(j)) => {
            let e = j.effort;
            let tau = body.arm.shoulder.thermal_tau_s;
            // Heat follows effort² with the joint's time constant; above 1 the joint derates.
            let t = if e > 1.0 { -tau * (1.0 - 1.0 / (e * e)).ln() } else { f32::INFINITY };
            (j.peak_ratio, Some(t))
        }
        _ => (shoulder.map_or(0.0, |j| j.peak_ratio), None),
    };
    // The working shoulder's reducer life used per hour at the strike torque (Miner, `wear`).
    let wear_per_h = match (s.task, shoulder) {
        (Task::Mine { .. }, Some(j)) => Some(crate::wear::damage(&body.arm.shoulder, j.torque_nm, 3600.0)),
        _ => None,
    };
    let limb = match s.task {
        Task::Mine { .. } => shoulder.map_or(0.0, |j| j.peak_ratio),
        Task::Travel => rep.joint(if post.single_support { "hip_roll" } else { "knee" }).or(rep.joint("drive")).map_or(0.0, |j| j.peak_ratio),
    };
    Probe {
        climb_deg: limits.max_slope_deg.min(surface::climb_limit_deg(body.base, &props, drive)),
        descend_deg: limits.max_slope_deg.min(surface::friction_slope_limit_deg(body.base, &props)),
        utilisation: Some(c.utilisation),
        required_mu: Some(c.required_mu),
        required_mu_level: Some(level.required_mu),
        slip_probability: c.slip_probability,
        net_traction: Some(c.net_traction),
        energy_factor: Some(surface::energy_factor(body.base, &props, m, limits.max_speed_mps, spec.power_w * limits.power_scale)),
        capacity_kg: body.capacity_on(&spec, ground, s.slope_deg),
        speed_ratio: limits.max_speed_mps / empty.max_speed_mps.max(1e-3),
        margin: rep.margin,
        slide_first: Some(slide_first),
        endurance_s: endurance,
        work_frac,
        wear_per_h,
        peak_ratio: worst.as_ref().map_or(0.0, |j| j.peak_ratio),
        effort: hot.as_ref().map_or(0.0, |j| j.effort),
        worst_joint: format!("피크 {} · 연속 {}", worst.as_ref().map_or("-", |j| j.joint.as_str()), hot.as_ref().map_or("-", |j| j.joint.as_str())),
        limb_peak_ratio: limb,
        posture_source: post.source,
        single_support: post.single_support,
    }
}

// ----- cells and scoring -----

/// The shared judge (`crate::bench::judge`); the gate is worded as this benchmark prints it.
fn judge(c: &mut FxCell) {
    let v = crate::bench::judge(c.value, c.rule.into());
    c.z = v.z;
    c.score = v.score;
    c.gate = v.broken.map(|(x, limit)| format!("{} {} {x:.2} > {limit:.2}", c.effect.label_ko(), c.axis));
}

fn cell(effect: Effect, axis: &'static str, value: Option<f32>, rule: Rule, cite: &'static str) -> FxCell {
    let mut c = FxCell { effect, axis, value, rule, cite, z: None, score: None, gate: None };
    judge(&mut c);
    c
}

/// The cells of a scenario from the engine's answers.
pub fn cells(body: &Body, s: &FxScenario, p: &Probe) -> Vec<FxCell> {
    let mut v = Vec::new();
    let wheels = !body.legged() && body.base != crate::base::BaseKind::Track;
    let mass = body.mass_kg();
    let cargo = s.cargo_frac * mass;
    // Slip.
    // The laboratory floor is chosen to be slippery: there the slip fraction is the question, not a gate.
    let lab = matches!(s.ground, Ground::Lab { .. });
    v.push(cell(Effect::Slip, "utilisation", p.utilisation, if lab { Rule::Raw } else { Rule::AtMost { limit: 1.0 } }, "마찰 이용률 > 1이면 미끄러짐 (정역학)"));
    v.push(cell(Effect::Slip, "slip_probability", p.slip_probability, Rule::Raw, ""));
    if body.legged() {
        if let Some((rise, tol)) = ramp_rcof_rise(s.slope_deg) {
            let value = p.required_mu.zip(p.required_mu_level).map(|(a, b)| a - b);
            v.push(cell(Effect::Slip, "ramp_rcof_rise", value, Rule::Like { reference: rise, tolerance: tol }, CITE_RAMP));
        }
        if lab {
            v.push(cell(Effect::Slip, "slip_fraction", p.slip_probability, Rule::Like { reference: SLIP_FRACTION.0, tolerance: SLIP_FRACTION.1 }, CITE_SLIP));
        }
    }
    if s.ground.id() == "sand" && wheels {
        v.push(cell(Effect::Slip, "net_traction_ceiling", p.net_traction, Rule::AtMost { limit: SAND_DPW_CEILING }, CITE_SAND_DPW));
        v.push(cell(Effect::Slip, "net_traction", p.net_traction, Rule::Like { reference: SAND_DPW.0, tolerance: SAND_DPW.1 }, CITE_SAND_DPW));
    } else {
        v.push(cell(Effect::Slip, "net_traction", p.net_traction, Rule::Raw, ""));
    }
    // Load.
    v.push(cell(Effect::Load, "cargo_over_capacity", Some(cargo / p.capacity_kg.max(1e-3)), Rule::AtMost { limit: 1.0 }, "적재 한도"));
    if body.legged() {
        v.push(cell(Effect::Load, "capacity_frac", Some(p.capacity_kg / mass), Rule::Like { reference: CARRY_FRAC.0, tolerance: CARRY_FRAC.1 }, CITE_CARRY));
        if (s.cargo_frac - 0.40).abs() < 1e-3 {
            v.push(cell(Effect::Load, "speed_ratio", Some(p.speed_ratio), Rule::Like { reference: LOAD_SPEED.0, tolerance: LOAD_SPEED.1 }, CITE_LOAD_SPEED));
        }
    } else {
        v.push(cell(Effect::Load, "capacity_frac", Some(p.capacity_kg / mass), Rule::Raw, ""));
    }
    // Fatigue.
    if let Task::Mine { .. } = s.task {
        let human = shoulder_endurance_s(p.work_frac);
        let ratio = p.endurance_s.map(|t| (t / human).min(100.0));
        v.push(cell(Effect::Fatigue, "endurance_ratio", ratio, Rule::AtLeast { reference: 1.0, tolerance: 0.38 }, CITE_ENDURANCE));
        v.push(cell(Effect::Fatigue, "wear_per_h", p.wear_per_h, Rule::Raw, ""));
    }
    if body.legged() && s.ground.id() == "sand" {
        v.push(cell(Effect::Fatigue, "energy_factor", p.energy_factor, Rule::Like { reference: SAND_ENERGY.0, tolerance: SAND_ENERGY.1 }, CITE_SAND_ENERGY));
    } else {
        v.push(cell(Effect::Fatigue, "energy_factor", p.energy_factor, Rule::Raw, ""));
    }
    // Joint load: the sizing rules gate; the limb's share of its peak against the person's.
    v.push(cell(Effect::JointLoad, "peak_ratio", Some(p.peak_ratio), Rule::AtMost { limit: 1.0 }, CITE_SIZING));
    v.push(cell(Effect::JointLoad, "rms_over_cont", Some(p.effort), Rule::AtMost { limit: 1.0 }, CITE_SIZING));
    let ctx = RefCtx { load_ratio: 0.0, load_dist_m: 0.0, slope_deg: s.slope_deg };
    let limb_ref = match s.task {
        Task::Mine { .. } => reference(Source::ArmRaise, Segment::Arm, "peak_ratio", &ctx),
        Task::Travel if p.single_support => reference(Source::FootLift, Segment::Leg, "peak_ratio", &ctx),
        Task::Travel => None,
    };
    match limb_ref {
        Some(r) => v.push(cell(Effect::JointLoad, "limb_peak_ratio", Some(p.limb_peak_ratio), Rule::Like { reference: r.value, tolerance: r.tolerance }, r.cite)),
        None => v.push(cell(Effect::JointLoad, "limb_peak_ratio", Some(p.limb_peak_ratio), Rule::Raw, "")),
    }
    // Stability.
    let mr = reference(p.posture_source, Segment::Foot, "margin", &ctx);
    match mr {
        Some(r) => v.push(cell(Effect::Stability, "margin", Some(p.margin), Rule::Like { reference: r.value, tolerance: r.tolerance }, r.cite)),
        None => v.push(cell(Effect::Stability, "margin", Some(p.margin), Rule::Raw, "")),
    }
    v.push(cell(Effect::Stability, "tip", Some(-p.margin), Rule::AtMost { limit: 0.0 }, "ZMP가 지지면 밖이면 넘어짐"));
    v.push(cell(Effect::Stability, "slide_first", p.slide_first.map(|b| if b { 1.0 } else { 0.0 }), Rule::Raw, "μ < d/h 이면 넘어지기 전에 미끄러짐 (정역학)"));
    v
}

/// Scores a scenario's cells: gates first, then the effect means.
pub fn score(s: &FxScenario, cells: Vec<FxCell>, probe: Option<Probe>) -> FxResult {
    let gates: Vec<String> = cells.iter().filter_map(|c| c.gate.clone()).collect();
    let effect_scores: [Option<f32>; 5] = std::array::from_fn(|i| {
        let e = Effect::ALL[i];
        let sc: Vec<f32> = cells.iter().filter(|c| c.effect == e).filter_map(|c| c.score).collect();
        (!sc.is_empty()).then(|| sc.iter().sum::<f32>() / sc.len() as f32)
    });
    let scored: Vec<f32> = effect_scores.iter().flatten().copied().collect();
    let score = if !gates.is_empty() { 0.0 } else if scored.is_empty() { 100.0 } else { scored.iter().sum::<f32>() / scored.len() as f32 };
    let within = cells.iter().all(|c| c.score.is_none() || c.z.map_or(c.score.unwrap_or(0.0) > 0.0, |z| z.abs() <= 2.0));
    FxResult { id: s.id, name_ko: s.name_ko, holdout: s.holdout, skipped: None, effect_scores, score, passed: gates.is_empty() && within, gates, cells, probe }
}

/// Runs one scenario on a robot.
pub fn evaluate(body: &Body, s: &FxScenario) -> FxResult {
    if s.needs_legs && !body.legged() {
        return FxResult { id: s.id, name_ko: s.name_ko, holdout: s.holdout, skipped: Some("보행 측정이라 다리가 없는 베이스는 건너뜀".into()), cells: vec![], effect_scores: [None; 5], score: 0.0, gates: vec![], passed: false, probe: None };
    }
    let p = probe(body, s);
    // The laboratory floor is meant to be too slippery; elsewhere a ground the base cannot take is a skip.
    let limit = if s.slope_deg >= 0.0 { p.climb_deg } else { p.descend_deg };
    if matches!(s.ground, Ground::World { .. }) && limit + 1e-3 < s.slope_deg.abs() {
        let why = if p.climb_deg < 0.0 {
            "이 지면에서는 구동력이 구름 저항을 넘지 못해 움직일 수 없음".to_string()
        } else if s.slope_deg >= 0.0 {
            format!("이 베이스가 오를 수 있는 비탈 {:.1}° < {:.1}°", limit, s.slope_deg)
        } else {
            format!("이 베이스가 미끄러지지 않고 내려갈 수 있는 비탈 {:.1}° < {:.1}°", limit, -s.slope_deg)
        };
        return FxResult { id: s.id, name_ko: s.name_ko, holdout: s.holdout, skipped: Some(why), cells: vec![], effect_scores: [None; 5], score: 0.0, gates: vec![], passed: false, probe: Some(p) };
    }
    let cs = cells(body, s, &p);
    score(s, cs, Some(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    #[test]
    fn the_reference_curves_reproduce_their_sources() {
        assert_eq!(ramp_rcof_rise(10.0), Some((0.14, 0.027)));
        assert!((ramp_rcof_rise(20.0).unwrap().0 - 0.25).abs() < 1e-6);
        assert!((ramp_rcof_rise(15.0).unwrap().0 - 0.195).abs() < 1e-6);
        assert!(ramp_rcof_rise(0.0).is_none());
        // Frey Law & Avin's shoulder curve: 52.8 s at half of the maximum.
        assert!((shoulder_endurance_s(0.5) - 52.8).abs() < 0.5);
        // The binomial tolerance of 38% of 31 people.
        assert!(((0.38f32 * 0.62 / 31.0).sqrt() - SLIP_FRACTION.1).abs() < 1e-3);
    }

    #[test]
    fn rules_score_likeness_one_sided_and_gates() {
        let mut c = FxCell { effect: Effect::Slip, axis: "x", value: Some(1.3), rule: Rule::Like { reference: 1.0, tolerance: 0.1 }, cite: "", z: None, score: None, gate: None };
        judge(&mut c);
        assert!((c.score.unwrap() - 70.0).abs() < 1e-3);
        let mut a = FxCell { rule: Rule::AtLeast { reference: 1.0, tolerance: 0.5 }, value: Some(4.0), z: None, score: None, ..c.clone() };
        judge(&mut a);
        assert_eq!(a.score, Some(100.0), "above the floor is full marks");
        let mut g = FxCell { rule: Rule::AtMost { limit: 1.0 }, value: Some(1.2), z: None, score: None, ..c.clone() };
        judge(&mut g);
        assert!(g.gate.is_some() && g.score.is_none());
        let mut m = FxCell { value: None, z: None, score: None, ..c };
        judge(&mut m);
        assert_eq!(m.score, Some(0.0), "a missing model is not a pass");
    }

    #[test]
    fn holdout_ranges_never_touch_the_development_conditions() {
        for s in library() {
            assert!(!HOLDOUT_SLOPE_DEG.iter().any(|r| (r.0..=r.1).contains(&s.slope_deg.abs())), "{}", s.id);
            assert!(!HOLDOUT_CARGO.iter().any(|r| (r.0..=r.1).contains(&s.cargo_frac)), "{}", s.id);
        }
        let h = holdout(20261024, 8);
        assert_eq!(h.len(), 8);
        assert_eq!(holdout(20261024, 8), h, "a seed gives the same draws");
        assert!(h.iter().all(|s| s.holdout && HOLDOUT_GROUNDS.contains(&s.ground.id())));
    }

    #[test]
    fn walking_scenarios_skip_bases_without_legs() {
        let b = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
        let e3 = library().into_iter().find(|s| s.id == "E3").unwrap();
        assert!(evaluate(&b, &e3).skipped.is_some());
    }
}
