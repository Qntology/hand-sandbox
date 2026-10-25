//! Head-camera benchmark: how the robot's camera shakes against what people's
//! heads and other robots' chassis measured (research `docs/research/camera.md`
//! §7, group B). None of these sources is an engine input
//! (`the_engine_never_names_a_camera_benchmark_source` checks the engine's
//! files), and no engine constant is set from them. The lines it prints are
//! `[CAM]` cells and `[CAM-SUM]` scenarios, scored as the other benchmarks:
//! `like` scores `100 − 10·|z|` against a reference and its spread, a `raw`
//! cell is recorded only.

use hand_core::act::gait;
use hand_core::{BaseKind, Body, BodyPlan, FingerCount, Surface};
use world_sim::world::{chassis_accel_rms, strike_peak_mps2};

// ----- references (benchmark side only; the engine never reads these) -----

/// A four-wheel skid-steer robot at about 0.8 m/s: vertical acceleration RMS
/// on grass 0.050, ploughed soil 0.065, stone pavement 0.085 m/s² (Milella,
/// Reina & Nielsen 2019). Spread: the SD of the three.
const CHASSIS_SPEED_MPS: f32 = 0.8;
const CHASSIS: [(Surface, &str, f32); 3] = [(Surface::Grass, "grass", 0.050), (Surface::Dirt, "dirt (ploughed)", 0.065), (Surface::Gravel, "gravel (stone paved)", 0.085)];
const CHASSIS_SPREAD: f32 = 0.0176;
const CITE_CHASSIS: &str = "4륜 스키드 스티어 0.8 m/s 수직 가속 RMS: 풀밭 0.050 · 갈아엎은 흙 0.065 · 돌 포장 0.085 m/s² (Milella 2019)";
/// Peak tibial acceleration grass over hard ground: 10.7 / 11.0 g = 0.97
/// (Moon et al. 2025, grass vs asphalt) and 8.22 / 7.47 g = 1.10 (Waite et
/// al. 2021, grass vs concrete): people stiffen their legs on soft ground.
const STRIKE_RATIO: (f32, f32) = (1.035, 0.065);
const CITE_STRIKE: &str = "디딤 충격 잔디/단단한 바닥: 0.97 (Moon 2025, 아스팔트) · 1.10 (Waite 2021, 콘크리트) — 사람은 무른 땅에서 다리를 굳힘";
/// Head acceleration on an irregular floor over a level one: about 1 for
/// young adults, while the pelvis's rises (Menz, Lord & Fitzpatrick 2003).
const CITE_HEAD_IRREGULAR: &str = "불규칙한 바닥/평평한 바닥의 머리 가속 비 ≈ 1, 골반은 커짐 (Menz 2003, 초록만 — 값 없음, 미구현)";

#[derive(Clone, Copy, Debug)]
enum Rule {
    Like { reference: f32, tolerance: f32 },
    Raw,
}

struct Cell {
    axis: String,
    value: Option<f32>,
    rule: Rule,
    cite: &'static str,
    score: Option<f32>,
}

fn cell(axis: impl Into<String>, value: Option<f32>, rule: Rule, cite: &'static str) -> Cell {
    let score = match (value, rule) {
        (None, Rule::Raw) => None,
        (None, _) => Some(0.0),
        (Some(v), Rule::Like { reference, tolerance }) => Some((100.0 - 10.0 * ((v - reference) / tolerance.max(1e-6)).abs()).max(0.0)),
        (Some(_), Rule::Raw) => None,
    };
    Cell { axis: axis.into(), value, rule, cite, score }
}

fn print_scenario(id: &str, name: &str, cells: &[Cell]) -> f32 {
    for c in cells {
        let v = c.value.map_or("미구현".to_string(), |v| format!("{v:.3}"));
        let s = c.score.map_or(String::new(), |s| format!(" · 점수 {s:.0}"));
        println!("[CAM] {id}/{} {v} ({:?}){s} — {}", c.axis, c.rule, c.cite);
    }
    let scored: Vec<f32> = cells.iter().filter_map(|c| c.score).collect();
    if scored.is_empty() {
        println!("[CAM-SUM] {id} {name} 기록만 (비교할 기준 없음)");
        return f32::NAN;
    }
    let score = scored.iter().sum::<f32>() / scored.len() as f32;
    println!("[CAM-SUM] {id} {name} 점수 {score:.1}");
    score
}

#[test]
fn camera_benchmark_prints_its_cells() {
    // C1: a mid wheeled base's chassis (rigid, small wheels) on the three grounds.
    let wheel = Body::new(&BodyPlan::uniform(hand_core::Tier::Mid), BaseKind::Wheel, FingerCount::Five);
    let d = 2.0 * hand_core::kin::WHEEL_R_M;
    let mut c1: Vec<Cell> = CHASSIS
        .iter()
        .map(|&(g, name, r)| cell(format!("{name}_rms_mps2"), Some(chassis_accel_rms(11, g, CHASSIS_SPEED_MPS, wheel.support_len_m, wheel.support_width_m, d)), Rule::Like { reference: r, tolerance: CHASSIS_SPREAD }, CITE_CHASSIS))
        .collect();
    let v: Vec<f32> = CHASSIS.iter().map(|&(g, _, _)| chassis_accel_rms(11, g, CHASSIS_SPEED_MPS, wheel.support_len_m, wheel.support_width_m, d)).collect();
    c1.push(cell("order_grass_lt_dirt_lt_gravel", Some(f32::from(u8::from(v[0] < v[1] && v[1] < v[2]))), Rule::Raw, CITE_CHASSIS));
    let s1 = print_scenario("C1-mid-Wheel", "바퀴형 섀시 진동 (지면별, 0.8 m/s)", &c1);

    // C2: the heel strike on grass against a hard floor, mid legs at a walk.
    let legs = Body::new(&BodyPlan::uniform(hand_core::Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
    let g = gait::plan(&legs, 0.6, 0.7, 0.0);
    let ds = g.ds_frac * g.step_s;
    let ratio = strike_peak_mps2(Surface::Grass, ds) / strike_peak_mps2(Surface::Firm, ds).max(1e-6);
    let c2 = vec![
        cell("strike_grass_over_firm", Some(ratio), Rule::Like { reference: STRIKE_RATIO.0, tolerance: STRIKE_RATIO.1 }, CITE_STRIKE),
        cell("head_irregular_over_level", None, Rule::Raw, CITE_HEAD_IRREGULAR),
    ];
    let s2 = print_scenario("C2-mid-HumanFoot", "다리형 디딤 충격의 지면 비", &c2);
    assert!(s1.is_finite() && s2.is_finite());
}

#[test]
fn the_engine_never_names_a_camera_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/world-sim/src/world/eye.rs",
        "crates/world-sim/src/render.rs",
        "crates/hand-core/src/kin.rs",
        "crates/hand-core/src/act/reach.rs",
        "crates/hand-runtime/src/vision.rs",
        "crates/hand-runtime/src/record.rs",
    ];
    let banned = ["Milella", "Reina", "Moon, Leary", "Moon 2025", "Waite", "Menz", "Kavanagh"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
