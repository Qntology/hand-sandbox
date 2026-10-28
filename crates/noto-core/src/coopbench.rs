use serde::Serialize;

use crate::act::cocarry::{self, Carrier, CarryOpts, CoCarry, GripKind, Load, TeamRule};
use crate::act::material::ItemMaterial;
use crate::actbench::Layer;
use crate::balance::Body;
use crate::base::BaseKind;
use crate::collbench::{cell, finish, like, raw, BenchCell, BenchResult};
use crate::effects::Rule;
use crate::hand::FingerCount;
use crate::surface::Surface;

pub const TEAM_STRENGTH: (f32, f32) = (0.862, 0.056);
pub const CITE_TEAM: &str = "팀 최대 들기 힘 / 개인 합: 남성 87.3%, 여성 91.1%, 혼성 80.2%, 2–4명에서 거의 같음 (Sharp, Rice, Nindl & Williamson 1997) — 세 값의 평균 ± SD";
pub const STAIR_LOWER_OVER_UPPER: (f32, f32) = (1.37, 0.37);
pub const STAIR_RISE_RUN_M: (f32, f32) = (0.180, 0.305);
pub const CITE_STAIR: &str = "계단(180/305 mm)으로 의자형 들것을 내릴 때 아래쪽(앞장선) 사람의 L4/L5 압박이 37% 큼 (Lavender 2000) · 두 사람 들기에서 척추 하중 차이는 짐 무게 차이보다 작음 (Dennis & Barrett 2003) → 아래/위 손 하중 비 ≥ 1.37 (허용폭은 초과분 0.37, 유도)";
pub const QUAD_PAIR_PAYLOAD: f32 = 0.7467;
pub const CITE_QUAD: &str = "사족 로봇 두 대(각 약 15 kg)가 함께 11.2 kg 운반: 한 대 질량의 74.67% (Sambhus 2026) — 몸 구조가 달라 기록만";
pub const PACE_TROT_SHARE: (f32, f32) = (0.409, 0.364);
pub const CITE_PHASE: &str = "막대형 물체를 함께 들고 걸을 때 동기화된 걸음의 40.9%가 같은 위상, 36.4%가 반대 위상 (Lanini 2017) — 엔진은 위상을 고르지 않아 두 경우의 하중 변동을 기록만";

pub const LOG_GRIP_M: f32 = 1.6;
pub const LOG_LEN_M: f32 = 2.0;
pub const LOG_DIA_M: f32 = 0.15;
pub const SLOPE_DEG: f32 = 10.0;
pub const CHAIR_GRIP_M: f32 = 1.0;
pub const CHAIR_COM_UP_M: f32 = 0.35;
pub const CHAIR_HANDLE_M: f32 = 0.04;
pub const ROCK_SIZE_M: f32 = 0.5;
pub const ROCK_GRIP_R_M: f32 = 0.3;
pub const LOAD_OVER_CAP: f32 = 1.4;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KxKind {
    LevelPair,
    SlopePair,
    StairChair,
    Mixed,
    RockThree,
    LineFour,
    LetGo,
    Conservative,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct KxScenario {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub kind: KxKind,
}

pub fn library() -> Vec<KxScenario> {
    let s = |id, name_ko, kind| KxScenario { id, name_ko, kind };
    vec![
        s("K1", "평지에서 원목을 둘이 받쳐 듦", KxKind::LevelPair),
        s("K2", "10° 비탈을 둘이 원목 들고 오름", KxKind::SlopePair),
        s("K3", "계단(30.5°)에서 의자형 들것을 둘이 듦", KxKind::StairChair),
        s("K4", "다리형과 바퀴형이 함께 원목을 듦", KxKind::Mixed),
        s("K5", "바위를 셋이 둘러 듦", KxKind::RockThree),
        s("K6", "긴 원목을 넷이 한 줄로 듦", KxKind::LineFour),
        s("K7", "둘 중 하나가 놓침", KxKind::LetGo),
        s("K8", "같은 짐을 보수 규칙(2/3)으로 셈", KxKind::Conservative),
    ]
}

pub fn member_kg(body: &Body) -> f32 {
    cocarry::member_capacity_kg(body, Surface::Firm, 0.0, cocarry::carry_hand_z(body))
}

pub fn sized_kg(body: &Body) -> f32 {
    LOAD_OVER_CAP * member_kg(body).max(0.1)
}

fn base_part(body: &Body) -> &'static str {
    match body.base {
        BaseKind::HumanFoot => "foot",
        BaseKind::Track => "track",
        _ => "wheel",
    }
}

fn pair<'a>(a: &'a Body, b: &'a Body, along: f32, dz: f32, slope_deg: f32) -> Vec<Carrier<'a>> {
    let mut c0 = Carrier::at(a, 0.0, 0.0, 0.0, Surface::Firm);
    let mut c1 = Carrier::at(b, along, 0.0, dz, Surface::Firm);
    c0.slope_deg = slope_deg;
    c1.slope_deg = slope_deg;
    vec![c0, c1]
}

fn carry_cells(cc: &CoCarry, load: &Load, cells: &mut Vec<BenchCell>, part: &'static str) {
    cells.push(raw(Layer::Outcome, "team_ok", Some("hand"), Some(if cc.ok { 1.0 } else { 0.0 })));
    cells.push(raw(Layer::Outcome, "load_over_limit", Some("hand"), Some(load.mass_kg / cc.limit_kg.max(1e-3))));
    cells.push(raw(Layer::Outcome, "balanced", Some("hand"), Some(if cc.statics.balanced { 1.0 } else { 0.0 })));
    let shares: Vec<f32> = cc.members.iter().map(|m| m.share).collect();
    cells.push(raw(Layer::Part, "share_max", Some("hand"), shares.iter().copied().reduce(f32::max)));
    cells.push(raw(Layer::Part, "share_min", Some("hand"), shares.iter().copied().reduce(f32::min)));
    cells.push(raw(Layer::Part, "peak_ratio_max", Some("hand_joints"), cc.members.iter().map(|m| m.peak_ratio).reduce(f32::max)));
    cells.push(raw(Layer::Coupling, "share>support:margin_min", Some(part), cc.members.iter().map(|m| m.margin).reduce(f32::min)));
    cells.push(raw(Layer::Coupling, "share>grip:use_max", Some("hand"), cc.members.iter().map(|m| m.grip_use).filter(|g| g.is_finite()).reduce(f32::max)));
    cells.push(raw(Layer::TimeSeries, "speed_mps", Some(part), Some(cc.speed_mps)));
}

fn carry_notes(cc: &CoCarry) -> Vec<String> {
    let mut v: Vec<String> = cc.members.iter().map(|m| format!("{}번 몫 {:.1}% ({:.2} kg, 걸음 최고 {:.2} kg / 능력 {:.2} kg){}", m.index + 1, m.share * 100.0, m.load_kg, m.peak_kg, m.capacity_kg, m.why_ko.as_ref().map_or(String::new(), |w| format!(" — {w}")))).collect();
    v.extend(cc.events.iter().map(|e| format!("{}: {}", e.kind, e.label_ko)));
    v.extend(cc.notes.iter().cloned());
    v
}

fn team_factor(cc: &CoCarry) -> Option<f32> {
    let sum: f32 = cc.members.iter().map(|m| m.capacity_kg).sum();
    (sum > 1e-3).then(|| cc.limit_kg / sum)
}

pub fn evaluate(body: &Body, s: &KxScenario) -> BenchResult {
    let opts = CarryOpts::default();
    let part = base_part(body);
    let mass = sized_kg(body);
    let mut cells = Vec::new();
    let notes;
    match s.kind {
        KxKind::LevelPair | KxKind::Conservative => {
            let log = Load::log(mass, LOG_LEN_M, LOG_DIA_M, true);
            let cs = pair(body, body, LOG_GRIP_M, 0.0, 0.0);
            let rule = if s.kind == KxKind::Conservative { TeamRule::Conservative } else { TeamRule::Hse2016 };
            let cc = cocarry::share(&log, &cs, &CarryOpts { rule, ..opts });
            carry_cells(&cc, &log, &mut cells, part);
            cells.push(cell(Layer::Coupling, "team_factor", Some("hand"), team_factor(&cc), like(TEAM_STRENGTH), CITE_TEAM));
            let alone = cocarry::share(&log, &cs[..1], &opts);
            cells.push(raw(Layer::Outcome, "alone_ok", Some("hand"), Some(if alone.ok { 1.0 } else { 0.0 })));
            if let Some(d) = &cc.dynamic {
                cells.push(cell(Layer::TimeSeries, "in_phase_n", Some("hand"), Some(d.in_phase_n[0]), Rule::Raw, CITE_PHASE));
                cells.push(raw(Layer::TimeSeries, "anti_phase_n", Some("hand"), Some(d.anti_phase_n[0])));
                cells.push(raw(Layer::TimeSeries, "step_hz", Some(part), Some(d.step_hz)));
            }
            cells.push(cell(Layer::Coupling, "team_limit_over_robot_mass", None, Some(cc.limit_kg / body.mass_kg()), Rule::Raw, CITE_QUAD));
            notes = carry_notes(&cc);
        }
        KxKind::SlopePair => {
            let log = Load::log(mass, LOG_LEN_M, LOG_DIA_M, true);
            let dz = LOG_GRIP_M * SLOPE_DEG.to_radians().tan();
            let cs = pair(body, body, LOG_GRIP_M, dz, SLOPE_DEG);
            let cc = cocarry::share(&log, &cs, &opts);
            carry_cells(&cc, &log, &mut cells, part);
            let (lo, up) = (cc.members[0].f_n, cc.members[1].f_n);
            cells.push(raw(Layer::Part, "lower_share", Some("hand"), Some(cc.members[0].share)));
            cells.push(raw(Layer::Part, "lower_over_upper", Some("hand"), Some(lo / up.max(1e-6))));
            cells.push(raw(Layer::Coupling, "axial_even_n", Some(part), Some(cc.axial_even_n)));
            cells.push(raw(Layer::Coupling, "axial_even_held", Some(part), Some(if cc.axial_even_ok { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Coupling, "team_factor", Some("hand"), team_factor(&cc)));
            notes = carry_notes(&cc);
        }
        KxKind::StairChair => {
            if body.base != BaseKind::HumanFoot {
                return finish(s.id, s.name_ko, vec![], vec![], Some("계단은 다리형만 오르내림".into()));
            }
            let th = STAIR_RISE_RUN_M.0.atan2(STAIR_RISE_RUN_M.1);
            let chair = Load { material: ItemMaterial::Wood, mass_kg: mass, length_m: CHAIR_GRIP_M, thick_m: CHAIR_HANDLE_M, com_along: 0.5, com_up_m: CHAIR_COM_UP_M, grip: GripKind::Squeeze };
            let cs = pair(body, body, CHAIR_GRIP_M * th.cos(), CHAIR_GRIP_M * th.sin(), 0.0);
            let cc = cocarry::share(&chair, &cs, &opts);
            carry_cells(&cc, &chair, &mut cells, part);
            let (lo, up) = (cc.members[0].f_n, cc.members[1].f_n);
            cells.push(cell(Layer::Part, "lower_over_upper", Some("hand"), Some(lo / up.max(1e-6)), Rule::AtLeast { reference: STAIR_LOWER_OVER_UPPER.0, tolerance: STAIR_LOWER_OVER_UPPER.1 }, CITE_STAIR));
            cells.push(raw(Layer::Part, "lower_share", Some("hand"), Some(cc.members[0].share)));
            cells.push(raw(Layer::Coupling, "axial_even_n", Some(part), Some(cc.axial_even_n)));
            cells.push(raw(Layer::Coupling, "incline_deg", None, Some(cc.statics.incline_deg)));
            notes = carry_notes(&cc);
        }
        KxKind::Mixed => {
            let wheel = Body::new(&body.plan, BaseKind::Wheel, FingerCount::Five);
            let legs = Body::new(&body.plan, BaseKind::HumanFoot, FingerCount::Five);
            let m = LOAD_OVER_CAP * member_kg(&legs).min(member_kg(&wheel)).max(0.1);
            let log = Load::log(m, LOG_LEN_M, LOG_DIA_M, true);
            let cs = pair(&legs, &wheel, LOG_GRIP_M, 0.0, 0.0);
            let cc = cocarry::share(&log, &cs, &opts);
            carry_cells(&cc, &log, &mut cells, part);
            cells.push(raw(Layer::Coupling, "incline_deg", None, Some(cc.statics.incline_deg)));
            if let Some(d) = &cc.dynamic {
                cells.push(raw(Layer::TimeSeries, "legs_in_phase_n", Some("foot"), Some(d.in_phase_n[0])));
                cells.push(raw(Layer::TimeSeries, "wheel_in_phase_n", Some("wheel"), Some(d.in_phase_n[1])));
            }
            cells.push(raw(Layer::Coupling, "team_factor", Some("hand"), team_factor(&cc)));
            notes = carry_notes(&cc);
        }
        KxKind::RockThree => {
            let rock = Load::rock(1.5 * mass, ROCK_SIZE_M);
            let cs: Vec<Carrier> = (0..3)
                .map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 3.0;
                    Carrier::at(body, ROCK_GRIP_R_M * a.cos(), ROCK_GRIP_R_M * a.sin(), 0.0, Surface::Firm)
                })
                .collect();
            let cc = cocarry::share(&rock, &cs, &opts);
            carry_cells(&cc, &rock, &mut cells, part);
            cells.push(cell(Layer::Coupling, "team_factor", Some("hand"), team_factor(&cc), like(TEAM_STRENGTH), CITE_TEAM));
            cells.push(raw(Layer::Outcome, "determinate", None, Some(if cc.statics.determinate { 1.0 } else { 0.0 })));
            notes = carry_notes(&cc);
        }
        KxKind::LineFour => {
            let log = Load::log(2.0 * mass, 3.6, 0.2, true);
            let cs: Vec<Carrier> = (0..4).map(|i| Carrier::at(body, 1.0 * i as f32, 0.0, 0.0, Surface::Firm)).collect();
            let cc = cocarry::share(&log, &cs, &opts);
            carry_cells(&cc, &log, &mut cells, part);
            cells.push(cell(Layer::Coupling, "team_factor", Some("hand"), team_factor(&cc), like(TEAM_STRENGTH), CITE_TEAM));
            cells.push(raw(Layer::Outcome, "determinate", None, Some(if cc.statics.determinate { 1.0 } else { 0.0 })));
            notes = carry_notes(&cc);
        }
        KxKind::LetGo => {
            let log = Load::log(mass, LOG_LEN_M, LOG_DIA_M, true);
            let cs = pair(body, body, LOG_GRIP_M, 0.0, 0.0);
            let g = cocarry::let_go(&log, &cs, 1, &opts);
            cells.push(raw(Layer::Outcome, "dropped", Some("hand"), Some(if g.dropped { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Outcome, "end_held", Some("hand"), Some(if g.end_held { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Part, "instant_share", Some("hand"), g.instant_n.first().map(|f| f / log.weight_n())));
            cells.push(raw(Layer::Part, "end_on_ground_share", Some("hand"), g.end_on_ground_share));
            cells.push(raw(Layer::Part, "end_drop_m", None, Some(g.end_drop_m)));
            notes = g.events.iter().map(|e| format!("{}: {}", e.kind, e.label_ko)).collect();
        }
    }
    finish(s.id, s.name_ko, cells, notes, None)
}
