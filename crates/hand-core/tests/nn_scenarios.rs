//! The composite N:N scenarios (`hand_core::nn`) for the named presets on a
//! walking and a wheeled base, compared with the human references. The lines
//! this prints (`[NN]` cells, `[NN-SUM]` scenarios, `[NN-HELD]` held-out) are
//! the record of how far the robot is from the references; the assertions
//! check the scoring machinery and the rules that keep the references from
//! becoming tuning targets, never a score threshold. The world's own quests
//! score the same phases as they happen (`hand-runtime` `nn_status`).

use hand_core::balance::{Body, FeedForward};
use hand_core::nn::{evaluate, holdout, library, shift_index, ScenarioResult};
use hand_core::parts::BodyPlan;
use hand_core::{BaseKind, FingerCount};

const PLANS: [&str; 4] = ["budget", "mid", "pro", "pro_arms"];
/// The evaluation seed of the held-out scenarios. Changing it is allowed
/// (research §4 rule 3: draw new ones per release); tuning against it is not.
const HOLDOUT_SEED: u64 = 20261024;

fn print_result(plan: &str, base: BaseKind, r: &ScenarioResult, cells: bool) {
    if let Some(why) = &r.skipped {
        println!("[NN-SUM] {plan:<8} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let items: Vec<String> = r.phases.iter().map(|p| format!("{}={:.0}({})", p.id, p.score, p.item)).collect();
    let beyond: Vec<String> = r.phases.iter().flat_map(|p| p.beyond_human.iter().map(move |x| format!("{}: {x}", p.id))).collect();
    println!(
        "[NN-SUM] {plan:<8} {base:?} {} {} 점수 {:.1}{}{} · 위상 {} · 가중: {}",
        r.id,
        r.name_ko,
        r.score,
        if r.gated { format!(" (게이트: {})", r.gates.join("; ")) } else { String::new() },
        if beyond.is_empty() { String::new() } else { format!(" (사람도 내딛는 강도: {})", beyond.join("; ")) },
        items.join(" "),
        r.weights_from
    );
    if !cells {
        return;
    }
    for p in &r.phases {
        for c in p.cells.iter().filter(|c| c.z.is_some()) {
            println!(
                "[NN] {plan} {base:?} {}/{} {}>{} {}{}: {:.3} (기준 {:.3} ± {:.3}, z {:+.1}, 점수 {:.0}) — {}",
                r.id,
                p.id,
                c.source.id(),
                c.target.id(),
                c.axis,
                if c.dynamic { " [동역학]" } else { "" },
                c.value,
                c.reference.unwrap_or(f32::NAN),
                c.tolerance.unwrap_or(f32::NAN),
                c.z.unwrap_or(f32::NAN),
                c.score.unwrap_or(f32::NAN),
                c.cite.unwrap_or("")
            );
        }
    }
}

#[test]
fn composite_scenarios_against_the_references() {
    let ff = FeedForward::default();
    for plan in PLANS {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let lib = library(&body);
            let results: Vec<ScenarioResult> = lib.iter().map(|s| evaluate(&body, s, &ff)).collect();
            for r in &results {
                print_result(plan, base, r, matches!(plan, "mid" | "pro"));
                assert!(r.score.is_finite() && (0.0..=100.0).contains(&r.score), "{} {}", r.id, r.score);
                for p in &r.phases {
                    assert!(p.cells.iter().all(|c| c.value.is_finite()), "{} {}", r.id, p.id);
                    assert!(p.cells.iter().any(|c| c.z.is_some()), "{} {}: every phase has referenced cells", r.id, p.id);
                }
            }
            let by = |id: &str| results.iter().find(|r| r.id == id).unwrap();
            assert_eq!(by("S2").skipped.is_some(), base != BaseKind::HumanFoot, "the 17 cm step needs legs");
            // S7: the work shift of S1 (120 min), S2 (60 min) and S4 (60 min), NIOSH SLI.
            let sli = shift_index(&[(by("S1"), 120.0), (by("S2"), 60.0), (by("S4"), 60.0)]);
            println!("[NN-SUM] {plan:<8} {base:?} S7 근무 교대 합성 SLI {} (1 넘으면 교대를 버티지 못함; NIOSH SLI, Waters 2007 — 토크비 대응은 이 프로젝트의 적용)", sli.map_or("—".into(), |v| format!("{v:.2}")));
        }
    }
}

#[test]
fn held_out_scenarios_are_counted_not_tuned() {
    let ff = FeedForward::default();
    for plan in ["mid", "pro"] {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let held = holdout(&body, HOLDOUT_SEED, 8);
            let results: Vec<ScenarioResult> = held.iter().map(|s| evaluate(&body, s, &ff)).collect();
            let run: Vec<&ScenarioResult> = results.iter().filter(|r| r.skipped.is_none()).collect();
            let passed = run.iter().filter(|r| r.passed).count();
            println!("[NN-HELD] {plan:<8} {base:?} 시드 {HOLDOUT_SEED}: 통과 {passed} / {} (DARPA 방식, 과제당 1점)", run.len());
            for r in &run {
                let p = &r.phases[0];
                println!("[NN-HELD]   {} {} {} 점수 {:.0} · 항목 {} · 동적 여유 {:.2} · 피크 {:.0}%{}", r.id, r.name_ko, p.id, p.score, p.item, p.min_margin, p.peak_ratio * 100.0, if r.gated { format!(" · {}", r.gates.join("; ")) } else { String::new() });
            }
            // Deterministic for a seed: the same draw, the same verdicts.
            let again: Vec<bool> = holdout(&body, HOLDOUT_SEED, 8).iter().map(|s| evaluate(&body, s, &ff).passed).collect();
            assert_eq!(again, results.iter().map(|r| r.passed).collect::<Vec<_>>());
        }
    }
}

#[test]
fn the_controller_never_reads_the_references() {
    // The balance controller and the joint dynamics must not see the human
    // references: they are for scoring only (research §4 rules 4 and 6).
    let balance = include_str!("../src/balance.rs");
    // Everything before the reference section (its types and table) is the controller.
    let controller = balance.split("/// Context the references depend on.").next().unwrap();
    for (name, src) in [("balance.rs (controller part)", controller), ("dynamics.rs", include_str!("../src/dynamics.rs"))] {
        for needle in ["reference(", "composite_reference", "nn::", "Reference {"] {
            assert!(!src.contains(needle), "{name} mentions `{needle}`");
        }
    }
}
