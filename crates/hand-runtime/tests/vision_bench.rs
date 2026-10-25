//! Spatial-memory benchmark: the robot's intermittent vision and the memory
//! it acts on in between, against people's and other robots' (research
//! `docs/research/perception.md` §4, group B). None of these sources is an
//! engine input (`the_engine_never_names_a_vision_benchmark_source` checks the
//! engine's files). The lines it prints are `[VX]` cells and `[VX-SUM]` scenarios.
//!
//! Scoring as the other benchmarks: `like` scores `100 − 10·|z|` against a
//! reference and its spread, `no_worse` scores 100 at or below a reference
//! error and `100 − 10·z` above it, `at_most` gates the scenario when a limit is
//! broken, and a `raw` cell is recorded only.

use hand_core::{BaseKind, BodyPlan};
use hand_runtime::{NoiseCfg, Runtime, RuntimeConfig};
use world_sim::world::DropItem;
use world_sim::{Action, ItemKind};

const LEX: &str = include_str!("../../../data/world_lexicon.json");

// ----- references (benchmark side only; the engine never reads these) -----

/// Blind walking to a target seen 4–12 m away: absolute error 92 cm over the
/// 8 m mean, 11.5% of the distance (Loomis et al. 1992, Exp. 1, n = 10, p. 911);
/// the variable error is about 8% of the distance (Rieser et al. 1990).
const BLIND_ABS_FRAC: f32 = 0.115;
const BLIND_SPREAD_FRAC: f32 = 0.08;
const CITE_BLIND: &str = "눈 감고 걷기 4–12 m: 절대 오차 92 cm / 평균 8 m = 거리의 11.5% (Loomis 1992 실험 1, n = 10), 가변 오차 거리의 약 8% (Rieser 1990)";
/// Signed error +55 cm over 8 m (+7%), same study.
const CITE_BLIND_SIGNED: &str = "부호 있는 오차 +55 cm / 8 m = +7% (Loomis 1992 실험 1)";
/// Waiting before walking: no effect within 30 s (Steenhuis & Goodale 1988),
/// none at all (Rieser et al. 1990), against Thomson 1983's 8 s.
const CITE_DELAY: &str = "출발 전 기다림: 30 s 안에는 영향 없음 (Steenhuis & Goodale 1988), 없음 (Rieser 1990) — Thomson 1983의 8 s 효과는 재현되지 않음";
/// Self-chosen sampling of less than half the walking time keeps walking safe (Patla et al. 1996).
const VISION_SHARE_MAX: f32 = 0.5;
const CITE_SHARE: &str = "스스로 고른 시각 표집이 걷는 시간의 50% 미만이어도 안전하게 걸음 (Patla 1996)";
/// Robots picking objects off the floor: 96.2% of the localised objects
/// (TidyBot, Wu et al. 2023 §4.5), 98.4% (Dusty, King et al. 2012), 94.7% of
/// 680 trials (Xu, Deyle & Kemp 2009): mean and SD of the three.
const FLOOR_PICK: (f32, f32) = (0.964, 0.019);
const CITE_FLOOR_PICK: &str = "로봇의 바닥 물건 집기 성공률 96.2% (TidyBot, Wu 2023) · 98.4% (Dusty, King 2012) · 94.7% (Xu 2009) — 평균 ± SD";
/// Pointing from memory: the error about doubles after a 2 s delay (Elliott & Madalena 1987).
const CITE_MEMORY_REACH: &str = "기억으로 가리키기: 2 s 지연 뒤 오차가 약 2배 (Elliott & Madalena 1987, 초록만 — SD 없음, 기록만)";

// ----- cells -----

#[derive(Clone, Copy, Debug)]
enum Rule {
    Like { reference: f32, tolerance: f32 },
    NoWorse { reference: f32, tolerance: f32 },
    AtMost { limit: f32 },
    Raw,
}

struct Cell {
    axis: &'static str,
    value: Option<f32>,
    rule: Rule,
    cite: &'static str,
    score: Option<f32>,
    gate: Option<String>,
}

fn cell(axis: &'static str, value: Option<f32>, rule: Rule, cite: &'static str) -> Cell {
    let (mut score, mut gate) = (None, None);
    match (value, rule) {
        (None, Rule::Raw) => {}
        (None, _) => score = Some(0.0),
        (Some(v), Rule::Like { reference, tolerance }) => score = Some((100.0 - 10.0 * ((v - reference) / tolerance.max(1e-6)).abs()).max(0.0)),
        (Some(v), Rule::NoWorse { reference, tolerance }) => score = Some((100.0 - 10.0 * ((v - reference) / tolerance.max(1e-6)).max(0.0)).max(0.0)),
        (Some(v), Rule::AtMost { limit }) => {
            if v > limit {
                gate = Some(format!("{axis} {v:.3} > {limit:.3}"));
            }
        }
        (Some(_), Rule::Raw) => {}
    }
    Cell { axis, value, rule, cite, score, gate }
}

fn print_scenario(id: &str, name: &str, cells: &[Cell]) -> f32 {
    for c in cells {
        let v = c.value.map_or("미구현".to_string(), |v| format!("{v:.3}"));
        let s = c.score.map_or(String::new(), |s| format!(" · 점수 {s:.0}"));
        let g = c.gate.as_ref().map_or(String::new(), |g| format!(" · 게이트 {g}"));
        println!("[VX] {id}/{} {v} ({:?}){s}{g} — {}", c.axis, c.rule, c.cite);
    }
    let scored: Vec<f32> = cells.iter().filter_map(|c| c.score).collect();
    let gated = cells.iter().any(|c| c.gate.is_some());
    // A scenario with nothing to score against (no reference) is a record, not a zero.
    if scored.is_empty() && !gated {
        println!("[VX-SUM] {id} {name} 기록만 (비교할 기준 없음)");
        return f32::NAN;
    }
    let score = if gated { 0.0 } else { scored.iter().sum::<f32>() / scored.len() as f32 };
    println!("[VX-SUM] {id} {name} 점수 {score:.1}{}", if gated { " (게이트)" } else { "" });
    score
}

fn mean(xs: &[f32]) -> Option<f32> {
    (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
}

fn runtime(seed: u64, base: BaseKind, plan: &str, parcels: Option<usize>, intermittent: bool, noise: NoiseCfg) -> Runtime {
    let mut cfg = RuntimeConfig { seed, noise, parcels, ..RuntimeConfig::default() };
    cfg.hand.base = base;
    cfg.body = BodyPlan::named(plan).expect("a body plan");
    cfg.vision.intermittent = intermittent;
    Runtime::new(cfg, LEX, None).expect("runtime")
}

/// One blind walk: the robot sees a parcel `d` ahead, waits `delay_s`, then
/// walks to it without looking; how far its memory of the parcel is off when
/// it gets there (the memory carried into the world against the parcel), and
/// along the way (positive: it believes it is further than it is).
fn blind_walk(seed: u64, base: BaseKind, plan: &str, d: f32, delay_s: f32) -> Option<(f32, f32)> {
    let mut rt = runtime(seed, base, plan, Some(0), true, NoiseCfg::none());
    let (x0, z0, yaw) = {
        let w = rt.world();
        (w.agent.x, w.agent.z, w.agent.yaw)
    };
    let (tx, tz) = (x0 + d * yaw.cos(), z0 + d * yaw.sin());
    rt.world_mut().drops.push(DropItem::whole(9001, ItemKind::Parcel, tx, tz));
    for _ in 0..2 {
        rt.scan_now("bench").ok()?;
    }
    let lm = rt.gates().landmarks().iter().find(|l| l.link == "parcel" && l.confirmed)?.clone();
    for _ in 0..((delay_s / world_sim::TICK_S).round() as u32) {
        rt.world_mut().tick();
    }
    let go = rt.to_world(Action::MoveTo { x: lm.x, z: lm.z, reach_m: 0.6 });
    rt.world_mut().command(go).ok()?;
    let mut n = 0;
    while rt.world().busy() && n < 3000 {
        rt.world_mut().tick();
        n += 1;
    }
    let (px, pz) = rt.world().true_of(lm.x, lm.z);
    let err = (px - tx).hypot(pz - tz);
    let along = (px - tx) * yaw.cos() + (pz - tz) * yaw.sin();
    Some((err, along))
}

fn eval_blind(base: BaseKind, plan: &str) -> Vec<Cell> {
    let mut abs = Vec::new();
    let mut signed = Vec::new();
    for d in [4.0f32, 6.0, 8.0, 10.0] {
        for seed in 1..=4u64 {
            if let Some((e, along)) = blind_walk(seed, base, plan, d, 0.0) {
                abs.push(e / d);
                signed.push(-along / d);
            }
        }
    }
    let delayed: Vec<f32> = (1..=4u64).filter_map(|s| blind_walk(s, base, plan, 8.0, 10.0)).map(|(e, _)| e).collect();
    let now: Vec<f32> = (1..=4u64).filter_map(|s| blind_walk(s, base, plan, 8.0, 0.0)).map(|(e, _)| e).collect();
    let ratio = match (mean(&delayed), mean(&now)) {
        (Some(a), Some(b)) if b > 1e-4 => Some(a / b),
        _ => None,
    };
    vec![
        cell("abs_error_frac", mean(&abs), Rule::NoWorse { reference: BLIND_ABS_FRAC, tolerance: BLIND_SPREAD_FRAC }, CITE_BLIND),
        cell("signed_error_frac", mean(&signed), Rule::Raw, CITE_BLIND_SIGNED),
        cell("delay_10s_over_none", ratio, Rule::Raw, CITE_DELAY),
        cell("walks", Some(abs.len() as f32), Rule::Raw, ""),
    ]
}

/// The base camp's parcel to the warehouse: seen in front at the start,
/// picked from the side in passing (or in front when the side does not hold).
fn eval_pick_in_passing(plan: &str) -> Vec<Cell> {
    let mut ok = 0u32;
    let mut runs = 0u32;
    let (mut side, mut picks, mut share_s, mut share_l) = (0u32, 0u32, 0.0f64, 0u32);
    let (mut mem_aim, mut seen_aim) = (Vec::new(), Vec::new());
    for seed in 1..=6u64 {
        let mut rt = runtime(seed, BaseKind::HumanFoot, plan, None, true, NoiseCfg::demo());
        let s = rt.submit("택배 상자 1개를 창고로 가져와");
        assert!(s.ok, "{:?}", s.error);
        rt.run_until_idle(9000);
        runs += 1;
        let q = rt.quests().last().cloned().expect("a quest");
        if q.state == hand_quest::State::Done {
            ok += 1;
        }
        let v = rt.vision_stats();
        side += v.side_picks;
        picks += v.picks;
        share_s += v.approach_s;
        share_l += v.approach_looks;
        mem_aim.extend(v.memory_aim_m.iter().copied());
        seen_aim.extend(v.seen_aim_m.iter().copied());
    }
    let rate = ok as f32 / runs.max(1) as f32;
    let share = (share_s > 0.0).then(|| (share_l as f64 * world_sim::TICK_S as f64 / share_s) as f32);
    let ratio = match (mean(&mem_aim), mean(&seen_aim)) {
        (Some(a), Some(b)) if b > 1e-5 => Some(a / b),
        _ => None,
    };
    vec![
        cell("success_rate", Some(rate), Rule::Like { reference: FLOOR_PICK.0, tolerance: FLOOR_PICK.1 }, CITE_FLOOR_PICK),
        cell("side_pick_share", (picks > 0).then(|| side as f32 / picks as f32), Rule::Raw, ""),
        cell("approach_vision_share", share, Rule::AtMost { limit: VISION_SHARE_MAX }, CITE_SHARE),
        cell("memory_over_seen_aim", ratio, Rule::Raw, CITE_MEMORY_REACH),
        cell("memory_aim_mm", mean(&mem_aim).map(|e| e * 1000.0), Rule::Raw, ""),
    ]
}

/// The same fetch with the camera on a pattern and always on: what looking
/// less costs in success and time, and what it saves in looks (a design check
/// against the robot's own always-on runs, not a research reference).
fn eval_cost(plan: &str) -> Vec<Cell> {
    let run = |intermittent: bool| -> (f32, f32, f32) {
        let (mut ok, mut scans, mut time) = (0u32, 0u64, 0.0f64);
        for seed in [3u64, 7] {
            let mut rt = runtime(seed, BaseKind::HumanFoot, plan, None, intermittent, NoiseCfg::demo());
            let s = rt.submit("원목 1개를 창고로 가져와");
            assert!(s.ok);
            rt.run_until_idle(12000);
            let q = rt.quests().last().cloned().expect("a quest");
            if q.state == hand_quest::State::Done {
                ok += 1;
                time += q.result.as_ref().map_or(0.0, |r| r.time_s);
            }
            scans += rt.eval().scans;
        }
        (ok as f32 / 2.0, scans as f32 / 2.0, if ok > 0 { (time / ok as f64) as f32 } else { f32::NAN })
    };
    let (on_ok, on_scans, on_t) = run(false);
    let (pat_ok, pat_scans, pat_t) = run(true);
    vec![
        cell("success_pattern", Some(pat_ok), Rule::Raw, "패턴 시각의 성공률"),
        cell("success_always", Some(on_ok), Rule::Raw, "늘 켠 시각의 성공률"),
        cell("success_kept", Some(pat_ok - on_ok), Rule::Raw, "패턴 − 늘 켬 (설계 목표: 정밀도 유지)"),
        cell("scans_pattern", Some(pat_scans), Rule::Raw, "퀘스트당 스캔 (계산 비용)"),
        cell("scans_always", Some(on_scans), Rule::Raw, "퀘스트당 스캔"),
        cell("time_pattern_s", Some(pat_t), Rule::Raw, ""),
        cell("time_always_s", Some(on_t), Rule::Raw, ""),
    ]
}

#[test]
fn vision_benchmark_prints_its_cells() {
    let mut total = Vec::new();
    for (plan, base) in [("mid", BaseKind::HumanFoot), ("mid", BaseKind::Wheel)] {
        let id = format!("V1-{plan}-{base:?}");
        total.push(print_scenario(&id, "본 물건까지 눈 감고 걷기", &eval_blind(base, plan)));
    }
    total.push(print_scenario("V2-mid", "베이스캠프 택배 상자를 지나가며 오른쪽으로 집기", &eval_pick_in_passing("mid")));
    // V3 has no reference to score against: the cost and the success of the two schedules are recorded.
    assert!(print_scenario("V3-pro", "패턴 시각과 늘 켠 시각 (원목 1개)", &eval_cost("pro")).is_nan());
    assert!(total.iter().all(|s| s.is_finite()));
}

#[test]
fn the_engine_never_names_a_vision_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/hand-runtime/src/vision.rs",
        "crates/hand-runtime/src/driver.rs",
        "crates/hand-runtime/src/mldrive.rs",
        "crates/hand-runtime/src/precision.rs",
        "crates/hand-runtime/src/runtime.rs",
        "crates/world-sim/src/odometry.rs",
        "crates/world-sim/src/world.rs",
        "crates/world-sim/src/world/acts.rs",
        "crates/hand-core/src/act/reach.rs",
        "crates/hand-core/src/act/grasp.rs",
        "crates/hand-core/src/kin.rs",
        "crates/hand-vision/src/decide.rs",
    ];
    let banned = ["Loomis", "Rieser", "Thomson", "Steenhuis", "Elliott", "Madalena", "Westwood", "Hesse", "Patla", "Hollands", "Laurent", "Assaiante", "TidyBot", "Dusty", "Deyle", "Ramanan", "Holbein", "Wysog", "King, Judge", "Toussaint", "van der Burg"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}
