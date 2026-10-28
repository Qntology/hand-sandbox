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

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "rule", rename_all = "snake_case")]
pub enum Rule {
    Like { reference: f32, tolerance: f32 },
    AtLeast { reference: f32, tolerance: f32 },
    AtMost { limit: f32 },
    Raw,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FxCell {
    pub effect: Effect,
    pub axis: &'static str,
    pub value: Option<f32>,
    pub rule: Rule,
    pub cite: &'static str,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub gate: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Travel,
    Mine { minutes: f32 },
}

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
    pub slope_deg: f32,
    pub cargo_frac: f32,
    pub task: Task,
    pub needs_legs: bool,
    pub holdout: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Probe {
    pub climb_deg: f32,
    pub descend_deg: f32,
    pub utilisation: Option<f32>,
    pub required_mu: Option<f32>,
    pub required_mu_level: Option<f32>,
    pub slip_probability: Option<f32>,
    pub net_traction: Option<f32>,
    pub energy_factor: Option<f32>,
    pub capacity_kg: f32,
    pub speed_ratio: f32,
    pub margin: f32,
    pub slide_first: Option<bool>,
    pub endurance_s: Option<f32>,
    pub work_frac: f32,
    pub wear_per_h: Option<f32>,
    pub peak_ratio: f32,
    pub effort: f32,
    pub worst_joint: String,
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
    pub effect_scores: [Option<f32>; 5],
    pub score: f32,
    pub gates: Vec<String>,
    pub passed: bool,
    pub probe: Option<Probe>,
}

pub fn ramp_rcof_rise(slope_deg: f32) -> Option<(f32, f32)> {
    let s = slope_deg.abs();
    if !(5.0..=20.0).contains(&s) {
        return None;
    }
    let v = if s <= 10.0 { 0.14 * s / 10.0 } else { 0.14 + (0.25 - 0.14) * (s - 10.0) / 10.0 };
    Some((v, 0.027))
}

pub const CITE_RAMP: &str = "경사 보행 RCOF 상승 +0.14 (10°), +0.25 (20°) (Hanson 1999, Beschorner 2016 인용), 허용폭 수평 RCOF SD 0.027 (Beschorner 2016)";
pub const LAB_GLYCEROL_ACOF: f32 = 0.164;
pub const SLIP_FRACTION: (f32, f32) = (0.38, 0.087);
pub const CITE_SLIP: &str = "글리세롤 바닥(ACOF 0.149–0.179) 예고 없는 노출의 38%가 미끄러짐 (Beschorner 2016, n = 31), 허용폭 이항 SD";
pub const SAND_DPW_CEILING: f32 = 0.3;
pub const SAND_DPW: (f32, f32) = (0.066, 0.028);
pub const CITE_SAND_DPW: &str = "모래 위 강성 바퀴 견인/하중 < 0.3, 1 m/s에서 0.052–0.080 (Rodríguez-Martínez 2019)";
pub const SAND_ENERGY: (f32, f32) = (2.4, 0.3);
pub const CITE_SAND_ENERGY: &str = "모래 보행 에너지 2.1–2.7배 (Lejeune 1998)";
pub const CARRY_FRAC: (f32, f32) = (0.30, 0.15);
pub const CITE_CARRY: &str = "지속 운반 하중 체중의 30% (전투), 45% (접근 행군) (Knapik & Reynolds)";
pub const LOAD_SPEED: (f32, f32) = (0.968, 0.105);
pub const CITE_LOAD_SPEED: &str = "체중 40% 짐의 보행 속도 4.6 / 4.75 km/h, SD 0.5 (Middleton 2022, n = 30)";
pub const CITE_ENDURANCE: &str = "어깨 지구 시간 ET = 14.86·f^-1.83 s (Frey Law & Avin 2010), 예상 변동 29–47% (Frey-Law 2012) → 허용폭 38%";
pub const CITE_SIZING: &str = "모터 선정 규칙: 연속 정격 > RMS 토크, 정지 토크 > 피크 (maxon Product Range 2026/27 p. 95)";

pub fn shoulder_endurance_s(f: f32) -> f32 {
    14.86 * f.clamp(0.01, 1.0).powf(-1.83)
}

fn scen(id: &'static str, name_ko: &'static str, ground: Ground, slope_deg: f32, cargo_frac: f32, task: Task, needs_legs: bool) -> FxScenario {
    FxScenario { id, name_ko, ground, slope_deg, cargo_frac, task, needs_legs, holdout: false }
}

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

pub const HOLDOUT_GROUNDS: [&str; 4] = ["dirt", "gravel", "grass", "wet_sand"];
pub const HOLDOUT_SLOPE_DEG: [(f32, f32); 2] = [(4.0, 8.0), (11.5, 13.5)];
pub const HOLDOUT_CARGO: [(f32, f32); 2] = [(0.15, 0.25), (0.45, 0.50)];

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

pub fn live_scenario(body: &Body, post: &Posture) -> Option<FxScenario> {
    let task = match post.source {
        Source::ArmRaise => Task::Mine { minutes: 1.0 },
        _ if post.propelling() => Task::Travel,
        _ => return None,
    };
    Some(FxScenario { id: "live", name_ko: "월드 상태", ground: Ground::World { id: post.ground.id() }, slope_deg: post.slope_deg, cargo_frac: post.cargo_kg / body.mass_kg().max(1e-3), task, needs_legs: false, holdout: false })
}

pub fn ground_props(g: &Ground) -> (Surface, SurfaceProps) {
    match *g {
        Ground::World { id } => {
            let s = Surface::from_id(id).unwrap_or_default();
            (s, *s.props())
        }
        Ground::Lab { id, acof } => (Surface::Firm, SurfaceProps { id, label_ko: "실험실 바닥", mu_foot: acof, mu_wheel: acof, mu_track: acof, ..*Surface::Firm.props() }),
    }
}

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
    let tip_deg = (0.5 * (rep.support[1] - rep.support[0]) / rep.com[2].max(0.1)).atan().to_degrees();
    let slide_first = surface::friction_slope_limit_deg(body.base, &props) < tip_deg;
    let shoulder = rep.joint("shoulder_r");
    let (work_frac, endurance) = match (s.task, shoulder) {
        (Task::Mine { .. }, Some(j)) => {
            let e = j.effort;
            let tau = body.arm.shoulder.thermal_tau_s;
            let t = if e > 1.0 { -tau * (1.0 - 1.0 / (e * e)).ln() } else { f32::INFINITY };
            (j.peak_ratio, Some(t))
        }
        _ => (shoulder.map_or(0.0, |j| j.peak_ratio), None),
    };
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

pub fn cells(body: &Body, s: &FxScenario, p: &Probe) -> Vec<FxCell> {
    let mut v = Vec::new();
    let wheels = !body.legged() && body.base != crate::base::BaseKind::Track;
    let mass = body.mass_kg();
    let cargo = s.cargo_frac * mass;
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
    v.push(cell(Effect::Load, "cargo_over_capacity", Some(cargo / p.capacity_kg.max(1e-3)), Rule::AtMost { limit: 1.0 }, "적재 한도"));
    if body.legged() {
        v.push(cell(Effect::Load, "capacity_frac", Some(p.capacity_kg / mass), Rule::Like { reference: CARRY_FRAC.0, tolerance: CARRY_FRAC.1 }, CITE_CARRY));
        if (s.cargo_frac - 0.40).abs() < 1e-3 {
            v.push(cell(Effect::Load, "speed_ratio", Some(p.speed_ratio), Rule::Like { reference: LOAD_SPEED.0, tolerance: LOAD_SPEED.1 }, CITE_LOAD_SPEED));
        }
    } else {
        v.push(cell(Effect::Load, "capacity_frac", Some(p.capacity_kg / mass), Rule::Raw, ""));
    }
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
    let mr = reference(p.posture_source, Segment::Foot, "margin", &ctx);
    match mr {
        Some(r) => v.push(cell(Effect::Stability, "margin", Some(p.margin), Rule::Like { reference: r.value, tolerance: r.tolerance }, r.cite)),
        None => v.push(cell(Effect::Stability, "margin", Some(p.margin), Rule::Raw, "")),
    }
    v.push(cell(Effect::Stability, "tip", Some(-p.margin), Rule::AtMost { limit: 0.0 }, "ZMP가 지지면 밖이면 넘어짐"));
    v.push(cell(Effect::Stability, "slide_first", p.slide_first.map(|b| if b { 1.0 } else { 0.0 }), Rule::Raw, "μ < d/h 이면 넘어지기 전에 미끄러짐 (정역학)"));
    v
}

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

pub fn evaluate(body: &Body, s: &FxScenario) -> FxResult {
    if s.needs_legs && !body.legged() {
        return FxResult { id: s.id, name_ko: s.name_ko, holdout: s.holdout, skipped: Some("보행 측정이라 다리가 없는 베이스는 건너뜀".into()), cells: vec![], effect_scores: [None; 5], score: 0.0, gates: vec![], passed: false, probe: None };
    }
    let p = probe(body, s);
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
        assert!((shoulder_endurance_s(0.5) - 52.8).abs() < 0.5);
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
