use noto_core::act::gait;
use noto_core::bench::{form_id, BenchCell, BenchScenario, Rule};
use noto_core::{BaseKind, Body, BodyPlan, FingerCount, Surface};
use world_sim::world::{chassis_accel_rms, strike_peak_mps2};

const CHASSIS_SPEED_MPS: f32 = 0.8;
const CHASSIS: [(Surface, &str, f32); 3] = [(Surface::Grass, "grass", 0.050), (Surface::Dirt, "dirt (ploughed)", 0.065), (Surface::Gravel, "gravel (stone paved)", 0.085)];
const CHASSIS_SPREAD: f32 = 0.0176;
const CITE_CHASSIS: &str = "4륜 스키드 스티어 0.8 m/s 수직 가속 RMS: 풀밭 0.050 · 갈아엎은 흙 0.065 · 돌 포장 0.085 m/s² (Milella 2019)";
const STRIKE_RATIO: (f32, f32) = (1.035, 0.065);
const CITE_STRIKE: &str = "디딤 충격 잔디/단단한 바닥: 0.97 (Moon 2025, 아스팔트) · 1.10 (Waite 2021, 콘크리트) — 사람은 무른 땅에서 다리를 굳힘";
const CITE_HEAD_IRREGULAR: &str = "불규칙한 바닥/평평한 바닥의 머리 가속 비 ≈ 1, 골반은 커짐 (Menz 2003, 초록만 — 값 없음, 미구현)";

fn cell(scenario: &str, form: &str, axis: &str, value: Option<f32>, rule: Rule, cite: &str) -> BenchCell {
    BenchCell::new("CAM", scenario, "", axis, None, form, value, rule, cite, false)
}

pub fn c1() -> BenchScenario {
    let wheel = Body::new(&BodyPlan::uniform(noto_core::Tier::Mid), BaseKind::Wheel, FingerCount::Five);
    let (id, form) = ("C1-mid-Wheel", form_id(&wheel));
    let d = 2.0 * noto_core::kin::WHEEL_R_M;
    let mut cells: Vec<BenchCell> = CHASSIS
        .iter()
        .map(|&(g, name, r)| cell(id, &form, &format!("{name}_rms_mps2"), Some(chassis_accel_rms(11, g, CHASSIS_SPEED_MPS, wheel.support_len_m, wheel.support_width_m, d)), Rule::Like { reference: r, tolerance: CHASSIS_SPREAD }, CITE_CHASSIS))
        .collect();
    let v: Vec<f32> = CHASSIS.iter().map(|&(g, _, _)| chassis_accel_rms(11, g, CHASSIS_SPEED_MPS, wheel.support_len_m, wheel.support_width_m, d)).collect();
    cells.push(cell(id, &form, "order_grass_lt_dirt_lt_gravel", Some(f32::from(u8::from(v[0] < v[1] && v[1] < v[2]))), Rule::Raw, CITE_CHASSIS));
    BenchScenario::from_cells("CAM", id, "바퀴형 섀시 진동 (지면별, 0.8 m/s)", &form, false, cells)
}

pub fn c2() -> BenchScenario {
    let legs = Body::new(&BodyPlan::uniform(noto_core::Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
    let (id, form) = ("C2-mid-HumanFoot", form_id(&legs));
    let g = gait::plan(&legs, 0.6, 0.7, 0.0);
    let ds = g.ds_frac * g.step_s;
    let ratio = strike_peak_mps2(Surface::Grass, ds) / strike_peak_mps2(Surface::Firm, ds).max(1e-6);
    let cells = vec![
        cell(id, &form, "strike_grass_over_firm", Some(ratio), Rule::Like { reference: STRIKE_RATIO.0, tolerance: STRIKE_RATIO.1 }, CITE_STRIKE),
        cell(id, &form, "head_irregular_over_level", None, Rule::Raw, CITE_HEAD_IRREGULAR),
    ];
    BenchScenario::from_cells("CAM", id, "다리형 디딤 충격의 지면 비", &form, false, cells)
}

pub fn scenarios() -> Vec<BenchScenario> {
    vec![c1(), c2()]
}
