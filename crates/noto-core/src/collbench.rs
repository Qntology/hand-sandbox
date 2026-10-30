use serde::Serialize;

use crate::act::collide::{self, BumpInput, Collision, FallingObject, HitInput, Law, Obstacle, Region};
use crate::act::fall::{self, Fall, FallCause, FallInput};
use crate::act::material::ItemMaterial;
pub use crate::actbench::HUMAN_LEG_M;
use crate::actbench::Layer;
use crate::balance::Body;
use crate::base::BaseKind;
use crate::effects::Rule;
use crate::kin::KinPose;
use crate::surface::Surface;

pub const FALL_PELVIS_MPS: (f32, f32) = (2.14, 0.63);
pub const FALL_HEAD_MPS: (f32, f32) = (2.91, 0.86);
pub const FALL_HAND_MPS: (f32, f32) = (2.87, 1.60);
pub const FALL_ONSET_S: (f32, f32) = (1.271, 0.648);
pub const FALL_DESCENT_S: (f32, f32) = (0.583, 0.255);
pub const CITE_CHOI_PELVIS: &str = "실제 낙상 골반 수직 충돌 속도 2.14 (0.63) m/s (Choi, Wakeling & Robinovitch 2015, 노인 25건 영상) — 다리 0.88 m 프루드 수로 맞춤";
pub const CITE_CHOI_HEAD: &str = "실제 낙상 머리 수직 충돌 속도 2.91 (0.86) m/s (Choi 2015) — 프루드 수로 맞춤";
pub const CITE_CHOI_HAND: &str = "실제 낙상 손 수직 충돌 속도 2.87 (1.60) m/s (Choi 2015) — 프루드 수로 맞춤";
pub const CITE_CHOI_ONSET: &str = "균형을 잃은 순간부터 골반 충돌까지 1271 (648) ms (Choi 2015) — 엔진이 두지 않는 넘어지기 전 단계(헛디딤, 버티는 걸음)가 들어 있어 기록만";
pub const CITE_CHOI_DESCENT: &str = "하강 시작부터 골반 충돌까지 583 (255) ms (Choi 2015) — 엔진의 넘어짐은 무게중심이 지지 모서리 위(호의 꼭대기)에서 시작해 골반이 곧바로 내려가므로 같은 시계; 첫 강한 타격까지, 프루드 수로 맞춤";
pub const SIDE_HIP_MPS: (f32, f32) = (2.75, 0.42);
pub const CITE_SIDE_HIP: &str = "옆으로 넘어질 때 엉덩이 수직 충돌 속도 2.75 (0.42) m/s (van den Kroonenberg, Hayes & McMahon 1996, n = 6) — 프루드 수로 맞춤";
pub const WRIST_FWD_MPS: f32 = 1.33;
pub const CITE_WRIST: &str = "앞으로 넘어질 때 손목 충돌 속도 1.33 m/s (Tan, Eng, Robinovitch & Warnick 2006; 분산 없음) — 기록만 (로봇은 팔로 짚지 않음)";
pub const HAND_LEAD_S: (f32, f32) = (0.050, 0.040);
pub const CITE_HAND_LEAD: &str = "옆 낙상 때 손이 골반보다 50 (40) ms 먼저 닿음 (Feldman & Robinovitch 2007), 손목–골반 38 ms (Hsiao & Robinovitch 1998) — 관절을 고정한 로봇은 짚지 않아 기록만";
pub const HUMANOID_FALL_G: (f32, f32) = (97.5, 15.5);
pub const CITE_KAJITA: &str = "사람 크기 휴머노이드(HRP-2Kai 65 kg) 앞으로 넘어짐, 스포츠 매트 위 최대 충격 82–113 g (Kajita 2016) — 가운데 ± 반폭, 매트 대신 엔진의 가장 무른 땅(풀밭)";
pub const WALL_COR: (f32, f32) = (0.521, 0.063);
pub const WALL_CONTACT_MS: (f32, f32) = (57.5, 26.3);
pub const CITE_WALL_COR: &str = "로봇 몸체가 벽·매트에 부딪힐 때 반발계수 0.526/0.603/0.451/0.504 (Liu & Karydis 2023 표 III, 1.9 kg 비행 로봇 약 2 m/s) — 네 값의 평균 ± SD";
pub const CITE_WALL_CONTACT: &str = "같은 충돌의 접촉 시간 31/37/79/83 ms (Liu & Karydis 2023 표 III) — 평균 ± SD, 질량이 다른 몸체의 대리값";
pub const DETECT_MS: (f32, f32) = (82.5, 26.3);
pub const DUMMY_KG: [f32; 2] = [9.08, 13.62];
pub const CITE_DETECT: &str = "구동 토크로 충돌 감지까지 45/95/85/105 ms (Kim, Llado & Sentis 2015, 9.08 kg 더미 0.5 m/s · 로봇 0.22 m/s로 13.62 kg 더미) — 평균 ± SD";
pub const HELMET_N: (f32, f32) = (3490.0, 960.0);
pub const HELMET_DROP: (f32, f32) = (3.6, 1.54);
pub const CITE_HELMET: &str = "3.6 kg 충격체 1.54 m 낙하 안전모 전달 최대 힘 새 것 2.53 kN ~ 반복 뒤 4.45 kN 넘음 (Wu, Pan & Wimer 2019) — 범위의 가운데 ± 반폭 (유도); 로봇 머리는 안전모가 아님";
pub const PELVIS_DROP_N: (f32, f32) = (1004.0, 115.0);
pub const PELVIS_DROP_M: f32 = 0.05;
pub const CITE_PELVIS: &str = "골반을 5 cm에서 떨어뜨린 옆 충돌 최대 힘 1004 (115) N (Laing & Robinovitch 2010) — 엔진의 ISO/TS 15066 골반 스프링·질량으로";

pub const FALL_OMEGA0: f32 = 0.3;
pub const CURB_M: f32 = 0.10;
pub const WALK_INTO_MPS: f32 = 0.8;
pub const SHELF_EDGE_M: f32 = 0.75;
pub const PARCEL: (f32, f32, f32) = (2.0, 0.30, 1.8);
pub const LOG: (f32, f32, f32) = (2.4, 0.15, 2.0);
pub const EDGE_DROP_M: f32 = 0.5;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchCell {
    pub layer: Layer,
    pub axis: &'static str,
    pub part: Option<&'static str>,
    pub value: Option<f32>,
    pub rule: Rule,
    pub cite: &'static str,
    pub z: Option<f32>,
    pub score: Option<f32>,
    pub gate: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchResult {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub skipped: Option<String>,
    pub cells: Vec<BenchCell>,
    pub layer_scores: [Option<f32>; 4],
    pub score: f32,
    pub gates: Vec<String>,
    pub passed: bool,
    pub notes: Vec<String>,
}

impl BenchResult {
    pub fn cell(&self, axis: &str) -> Option<&BenchCell> {
        self.cells.iter().find(|c| c.axis == axis)
    }

    pub fn scored(&self) -> bool {
        self.layer_scores.iter().any(|s| s.is_some())
    }
}

pub(crate) fn judge(c: &mut BenchCell) {
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

pub(crate) fn cell(layer: Layer, axis: &'static str, part: Option<&'static str>, value: Option<f32>, rule: Rule, cite: &'static str) -> BenchCell {
    let mut c = BenchCell { layer, axis, part, value, rule, cite, z: None, score: None, gate: None };
    judge(&mut c);
    c
}

pub(crate) fn raw(layer: Layer, axis: &'static str, part: Option<&'static str>, value: Option<f32>) -> BenchCell {
    cell(layer, axis, part, value, Rule::Raw, "")
}

pub(crate) fn like(r: (f32, f32)) -> Rule {
    Rule::Like { reference: r.0, tolerance: r.1 }
}

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}

pub(crate) fn finish(id: &'static str, name_ko: &'static str, cells: Vec<BenchCell>, notes: Vec<String>, skipped: Option<String>) -> BenchResult {
    let gates: Vec<String> = cells.iter().filter_map(|c| c.gate.clone()).collect();
    let layer_scores: [Option<f32>; 4] = std::array::from_fn(|i| {
        let sc: Vec<f32> = cells.iter().filter(|c| c.layer == Layer::ALL[i]).filter_map(|c| c.score).collect();
        mean(&sc)
    });
    let scored: Vec<f32> = layer_scores.iter().flatten().copied().collect();
    let score = if skipped.is_some() || !gates.is_empty() { 0.0 } else { mean(&scored).unwrap_or(0.0) };
    let within = cells.iter().all(|c| c.score.is_none() || c.z.map_or(c.score.unwrap_or(0.0) > 0.0, |z| z.abs() <= 2.0));
    BenchResult { id, name_ko, skipped, passed: gates.is_empty() && within && !scored.is_empty(), gates, layer_scores, score, cells, notes }
}

fn skip(id: &'static str, name_ko: &'static str, why: &str) -> BenchResult {
    finish(id, name_ko, vec![], vec![], Some(why.to_string()))
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CxKind {
    WallBump,
    Detect { moving: bool },
    ShelfEdge,
    Curb,
    Fall { dir_deg: f32, ground: Surface },
    EdgeFall,
    Dropped { material: ItemMaterial, mass_kg: f32, size_m: f32, h_m: f32 },
    PelvisDrop,
    PersonLeg,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CxScenario {
    pub id: &'static str,
    pub name_ko: &'static str,
    pub kind: CxKind,
}

pub fn library() -> Vec<CxScenario> {
    let s = |id, name_ko, kind| CxScenario { id, name_ko, kind };
    vec![
        s("C1", "콘크리트 벽에 최고 속도로 부딪힘", CxKind::WallBump),
        s("C2", "느린 더미가 서 있는 로봇에 부딪힘 (감지)", CxKind::Detect { moving: false }),
        s("C3", "0.22 m/s로 더미에 부딪힘 (감지)", CxKind::Detect { moving: true }),
        s("C4", "가슴 높이 석고보드 선반 모서리", CxKind::ShelfEdge),
        s("C5", "10 cm 연석에 걸림", CxKind::Curb),
        s("F1", "서 있다 앞으로 넘어짐 (단단한 바닥)", CxKind::Fall { dir_deg: 0.0, ground: Surface::Firm }),
        s("F2", "서 있다 뒤로 넘어짐 (단단한 바닥)", CxKind::Fall { dir_deg: 180.0, ground: Surface::Firm }),
        s("F3", "서 있다 옆으로 넘어짐 (단단한 바닥)", CxKind::Fall { dir_deg: 90.0, ground: Surface::Firm }),
        s("F4", "앞으로 넘어짐 (풀밭: 매트 대리)", CxKind::Fall { dir_deg: 0.0, ground: Surface::Grass }),
        s("F5", "0.5 m 모서리에서 뒤로 떨어짐", CxKind::EdgeFall),
        s("H1", "안전모 시험 충격체가 머리에 떨어짐", CxKind::Dropped { material: ItemMaterial::Stone, mass_kg: HELMET_DROP.0, size_m: 0.10, h_m: HELMET_DROP.1 }),
        s("H2", "선반에서 택배 상자가 떨어짐", CxKind::Dropped { material: ItemMaterial::Cardboard, mass_kg: PARCEL.0, size_m: PARCEL.1, h_m: PARCEL.2 }),
        s("H3", "더미에서 원목이 떨어짐", CxKind::Dropped { material: ItemMaterial::Wood, mass_kg: LOG.0, size_m: LOG.1, h_m: LOG.2 }),
        s("P1", "사람 골반 5 cm 낙하 (ISO 몸 모델)", CxKind::PelvisDrop),
        s("P2", "걷는 속도로 사람 정강이에 닿음", CxKind::PersonLeg),
    ]
}

pub fn leg_m(_body: &Body) -> f32 {
    crate::dims::Dims::COMPACT.hip_m
}

fn froude_v(v: f32, leg: f32) -> f32 {
    v * (HUMAN_LEG_M / leg.max(0.05)).sqrt()
}

fn froude_t(t: f32, leg: f32) -> f32 {
    t * (HUMAN_LEG_M / leg.max(0.05)).sqrt()
}

pub fn top_speed(body: &Body) -> f32 {
    body.drive_limits(&body.base.spec(), 0.0).max_speed_mps
}

fn part_of(body: &Body) -> &'static str {
    match body.base {
        BaseKind::HumanFoot => "foot",
        BaseKind::Track => "track",
        _ => "wheel",
    }
}

fn notes_of(c: &Collision) -> Vec<String> {
    c.rollout.events.iter().map(|e| format!("{:.3} s {}: {}", e.t, e.kind, e.label_ko)).collect()
}

fn fall_notes(f: &Fall) -> Vec<String> {
    let mut v: Vec<String> = f.rollout.events.iter().filter(|e| e.kind != "impact" || e.value.unwrap_or(0.0) > 0.0).take(8).map(|e| format!("{:.3} s {}: {}", e.t, e.kind, e.label_ko)).collect();
    v.push(f.get_up.why_ko.clone());
    v
}

fn bump_cells(body: &Body, c: &Collision, cells: &mut Vec<BenchCell>) {
    let part = Some(part_of(body));
    cells.push(raw(Layer::Outcome, "detected", part, Some(if c.detected { 1.0 } else { 0.0 })));
    cells.push(raw(Layer::Outcome, "tipped", part, Some(if c.tip.is_some() { 1.0 } else { 0.0 })));
    cells.push(raw(Layer::Part, "peak_n", part, c.pulse.as_ref().map(|p| p.peak_n)));
    cells.push(raw(Layer::Part, "m_eff_kg", part, Some(c.m_eff_kg)));
    cells.push(raw(Layer::Part, "a_peak_g", Some("waist"), Some(c.a_peak_g)));
    cells.push(raw(Layer::Coupling, "contact>support:margin_min", part, Some(c.margin_min)));
    cells.push(raw(Layer::Coupling, "contact>support:force_angle_min", part, Some(c.beta_min)));
    let worst = c.damage.iter().map(|h| h.ratio).fold(0.0f32, f32::max);
    cells.push(raw(Layer::Coupling, "contact>joints:peak_ratio", None, Some(worst)));
}

pub fn evaluate(body: &Body, s: &CxScenario) -> BenchResult {
    let legged = body.base == BaseKind::HumanFoot;
    let stand = KinPose::stand(body, 0.0);
    let mut cells = Vec::new();
    let notes;
    match s.kind {
        CxKind::WallBump => {
            let v = if legged { WALK_INTO_MPS } else { top_speed(body) };
            let c = collide::bump(&BumpInput::wall(body, v, Obstacle::Concrete, Surface::Firm));
            bump_cells(body, &c, &mut cells);
            let p = c.pulse.as_ref();
            if legged {
                cells.push(raw(Layer::TimeSeries, "contact_ms", Some("foot"), p.map(|p| p.contact_s * 1000.0)));
                cells.push(raw(Layer::Part, "restitution", Some("foot"), p.map(|p| p.e)));
                cells.push(raw(Layer::Part, "second_peak_n", Some("waist"), c.second.as_ref().and_then(|s| s.pulse.as_ref()).map(|p| p.peak_n)));
            } else {
                cells.push(cell(Layer::TimeSeries, "contact_ms", Some(part_of(body)), p.map(|p| p.contact_s * 1000.0), like(WALL_CONTACT_MS), CITE_WALL_CONTACT));
                cells.push(cell(Layer::Part, "restitution", Some(part_of(body)), p.map(|p| p.e), like(WALL_COR), CITE_WALL_COR));
            }
            cells.push(raw(Layer::Outcome, "speed_mps", None, Some(v)));
            cells.push(raw(Layer::TimeSeries, "detect_ms", Some("waist"), c.detect_s.map(|t| t * 1000.0)));
            notes = notes_of(&c);
        }
        CxKind::Detect { moving } => {
            let (speed, ob_speed, m) = if moving { (0.22, 0.0, DUMMY_KG[1]) } else { (0.0, 0.5, DUMMY_KG[0]) };
            let c = collide::bump(&BumpInput { speed_mps: speed, obstacle_speed_mps: ob_speed, ..BumpInput::wall(body, speed, Obstacle::Robot { mass_kg: m }, Surface::Firm) });
            bump_cells(body, &c, &mut cells);
            cells.push(cell(Layer::TimeSeries, "detect_ms", Some("waist"), c.detect_s.map(|t| t * 1000.0), like(DETECT_MS), CITE_DETECT));
            notes = notes_of(&c);
        }
        CxKind::ShelfEdge => {
            let v = if legged { WALK_INTO_MPS } else { top_speed(body) };
            let c = collide::bump(&BumpInput { obstacle_bottom_m: SHELF_EDGE_M, ..BumpInput::wall(body, v, Obstacle::Drywall, Surface::Firm) });
            bump_cells(body, &c, &mut cells);
            cells.push(raw(Layer::Part, "dent_mpa", Some("waist"), Some(c.dent_pa.map_or(0.0, |p| p * 1e-6))));
            cells.push(raw(Layer::TimeSeries, "contact_ms", Some("waist"), c.pulse.as_ref().map(|p| p.contact_s * 1000.0)));
            cells.push(raw(Layer::Coupling, "tilt_max_deg", Some(part_of(body)), Some(c.tilt_max_deg)));
            notes = notes_of(&c);
        }
        CxKind::Curb => {
            let v = if legged { WALK_INTO_MPS } else { top_speed(body) };
            let c = collide::bump(&BumpInput::curb(body, v, CURB_M, Surface::Firm));
            bump_cells(body, &c, &mut cells);
            cells.push(raw(Layer::TimeSeries, "contact_ms", Some(part_of(body)), c.pulse.as_ref().map(|p| p.contact_s * 1000.0)));
            cells.push(raw(Layer::Coupling, "tilt_max_deg", Some(part_of(body)), Some(c.tilt_max_deg)));
            cells.push(raw(Layer::Coupling, "energy_ratio", Some(part_of(body)), Some(c.energy_ratio)));
            notes = notes_of(&c);
        }
        CxKind::Fall { dir_deg, ground } => {
            let f = fall::fall(&FallInput::lost_balance(body, stand, FallCause::Pushed, dir_deg, FALL_OMEGA0, ground));
            let leg = leg_m(body);
            cells.push(raw(Layer::Outcome, "fell", Some("waist"), Some(if f.fell { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Outcome, "gets_up", Some(part_of(body)), Some(if f.get_up.can { 1.0 } else { 0.0 })));
            let main = f.main.as_ref();
            cells.push(raw(Layer::Part, "v_impact_mps", None, main.map(|s| s.v_n)));
            cells.push(raw(Layer::Part, "peak_n", None, main.map(|s| s.peak_n)));
            cells.push(raw(Layer::TimeSeries, "t_ground_s", None, Some(f.t_ground_s)));
            cells.push(raw(Layer::TimeSeries, "t_first_s", None, Some(f.t_first_s)));
            if legged {
                let side = (dir_deg.rem_euclid(180.0) - 90.0).abs() < 1.0;
                cells.push(cell(Layer::TimeSeries, "t_descent_s_froude", None, Some(froude_t(f.t_first_s, leg)), like(FALL_DESCENT_S), CITE_CHOI_DESCENT));
                cells.push(cell(Layer::TimeSeries, "t_onset_s_froude", None, Some(froude_t(f.t_first_s, leg)), Rule::Raw, CITE_CHOI_ONSET));
                if side {
                    cells.push(cell(Layer::Part, "pelvis_v_froude", Some("waist"), f.pelvis_v.map(|v| froude_v(v, leg)), like(SIDE_HIP_MPS), CITE_SIDE_HIP));
                } else {
                    cells.push(cell(Layer::Part, "pelvis_v_froude", Some("waist"), f.pelvis_v.map(|v| froude_v(v, leg)), like(FALL_PELVIS_MPS), CITE_CHOI_PELVIS));
                }
                cells.push(cell(Layer::Part, "head_v_froude", Some("neck"), f.head_v.map(|v| froude_v(v, leg)), like(FALL_HEAD_MPS), CITE_CHOI_HEAD));
                cells.push(cell(Layer::Part, "hand_v_froude", Some("hand"), f.hand_v.map(|v| froude_v(v, leg)), like(FALL_HAND_MPS), CITE_CHOI_HAND));
                if dir_deg.abs() < 1.0 {
                    cells.push(cell(Layer::Part, "wrist_v_froude", Some("hand"), f.hand_v.map(|v| froude_v(v, leg)), Rule::Raw, CITE_WRIST));
                }
                if side {
                    let t_hand = f.strikes.iter().find(|s| s.link.starts_with("palm") || s.link.starts_with("forearm")).map(|s| s.t);
                    let t_pelvis = f.strikes.iter().find(|s| s.link == "pelvis").map(|s| s.t);
                    cells.push(cell(Layer::TimeSeries, "hand_lead_s", Some("hand"), t_hand.zip(t_pelvis).map(|(h, p)| p - h), Rule::Raw, CITE_HAND_LEAD));
                }
            } else {
                cells.push(raw(Layer::Part, "head_v_mps", Some("neck"), f.head_v));
            }
            let trunk_g = main.map(|s| s.a_trunk_g);
            if legged && dir_deg.abs() < 1.0 && ground == Surface::Grass {
                cells.push(cell(Layer::Coupling, "ground>trunk:a_peak_g", Some("waist"), trunk_g, like(HUMANOID_FALL_G), CITE_KAJITA));
            } else {
                cells.push(raw(Layer::Coupling, "ground>trunk:a_peak_g", Some("waist"), trunk_g));
            }
            cells.push(raw(Layer::Coupling, "fall>joints:peak_ratio", None, Some(f.damage.iter().map(|h| h.ratio).fold(0.0f32, f32::max))));
            cells.push(raw(Layer::Coupling, "righting_force_n", Some(part_of(body)), Some(f.get_up.righting_force_n)));
            notes = fall_notes(&f);
        }
        CxKind::EdgeFall => {
            let f = fall::fall(&FallInput { drop_m: EDGE_DROP_M, ..FallInput::lost_balance(body, stand, FallCause::Edge, 180.0, FALL_OMEGA0, Surface::Firm) });
            cells.push(raw(Layer::Outcome, "fell", Some("waist"), Some(if f.fell { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Outcome, "gets_up", Some(part_of(body)), Some(if f.get_up.can { 1.0 } else { 0.0 })));
            let main = f.main.as_ref();
            cells.push(raw(Layer::Part, "v_impact_mps", None, main.map(|s| s.v_n)));
            cells.push(raw(Layer::Part, "peak_n", None, main.map(|s| s.peak_n)));
            cells.push(raw(Layer::Part, "energy_j", None, main.map(|s| s.energy_j)));
            cells.push(raw(Layer::TimeSeries, "t_ground_s", None, Some(f.t_ground_s)));
            cells.push(raw(Layer::Coupling, "ground>trunk:a_peak_g", Some("waist"), main.map(|s| s.a_trunk_g)));
            cells.push(raw(Layer::Coupling, "fall>joints:peak_ratio", None, Some(f.damage.iter().map(|h| h.ratio).fold(0.0f32, f32::max))));
            notes = fall_notes(&f);
        }
        CxKind::Dropped { material, mass_kg, size_m, h_m } => {
            let c = collide::struck_by(&HitInput { body, pose: stand, object: FallingObject::dropped(material, mass_kg, size_m, h_m), point: None, ground: Surface::Firm });
            let on_head = c.contact.is_some_and(|p| p.link == "head");
            let peak = c.pulse.as_ref().map(|p| p.peak_n);
            if s.id == "H1" && on_head {
                cells.push(cell(Layer::Part, "head_peak_n", Some("neck"), peak, like(HELMET_N), CITE_HELMET));
            } else {
                cells.push(raw(Layer::Part, "peak_n", Some("neck"), peak));
            }
            cells.push(raw(Layer::Outcome, "tipped", Some(part_of(body)), Some(if c.tip.is_some() { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Outcome, "serious_energy", None, c.rollout.metric("serious")));
            cells.push(raw(Layer::Part, "energy_j", None, c.rollout.metric("energy_j")));
            cells.push(raw(Layer::TimeSeries, "contact_ms", Some("neck"), c.pulse.as_ref().map(|p| p.contact_s * 1000.0)));
            cells.push(raw(Layer::Coupling, "hit>support:margin_min", Some(part_of(body)), Some(c.margin_min)));
            let neck = c.damage.iter().find(|h| h.joint == "neck").map(|h| h.ratio);
            cells.push(raw(Layer::Coupling, "hit>neck:peak_ratio", Some("neck"), neck));
            notes = notes_of(&c);
        }
        CxKind::PelvisDrop => {
            let l = Region::Pelvis.limit();
            let v = collide::drop_speed(PELVIS_DROP_M);
            let p = collide::pulse(&Law::Linear { k: l.k_n_per_m }, 0.99, l.m_kg, v);
            cells.push(cell(Layer::Part, "pelvis_peak_n", Some("waist"), Some(p.peak_n), like(PELVIS_DROP_N), CITE_PELVIS));
            cells.push(raw(Layer::TimeSeries, "contact_ms", Some("waist"), Some(p.contact_s * 1000.0)));
            cells.push(raw(Layer::Outcome, "v_mps", None, Some(v)));
            notes = vec![format!("ISO 골반 k {:.0} N/mm, m {:.0} kg, {:.2} m/s", l.k_n_per_m / 1000.0, l.m_kg, v)];
        }
        CxKind::PersonLeg => {
            let v = if legged { WALK_INTO_MPS } else { top_speed(body).min(WALK_INTO_MPS) };
            let c = collide::bump(&BumpInput::wall(body, v, Obstacle::Person(Region::LowerLeg), Surface::Firm));
            bump_cells(body, &c, &mut cells);
            let iso = c.iso.as_ref();
            cells.push(raw(Layer::Outcome, "iso_ok", None, iso.map(|i| if i.ok { 1.0 } else { 0.0 })));
            cells.push(raw(Layer::Part, "iso_peak_n", None, iso.map(|i| i.f_peak_n)));
            cells.push(raw(Layer::Part, "iso_v_max_mps", None, iso.and_then(|i| i.v_max_mps)));
            cells.push(raw(Layer::Part, "en1525_leg_ratio", None, c.pulse.as_ref().map(|p| p.peak_n / collide::EN1525_LEG_N)));
            notes = notes_of(&c);
        }
    }
    finish(s.id, s.name_ko, cells, notes, None)
}

pub fn evaluate_or_skip(body: &Body, s: &CxScenario) -> BenchResult {
    match s.kind {
        CxKind::Fall { ground: Surface::Grass, .. } if body.base != BaseKind::HumanFoot => skip(s.id, s.name_ko, "매트 대리 낙상은 다리형(사람 크기 휴머노이드 기준)만"),
        CxKind::PelvisDrop if body.base != BaseKind::HumanFoot => skip(s.id, s.name_ko, "몸 모델 확인은 한 번만 (다리형 줄에서)"),
        _ => evaluate(body, s),
    }
}
