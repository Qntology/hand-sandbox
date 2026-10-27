//! The tactile N:N benchmark (`noto_core::tactbench`) for the named presets
//! on legs, wheels and tracks. The lines this prints (`[TX]` cells,
//! `[TX-SUM]` scenarios, `[TX-HELD]` held-out) are the record of how far the
//! robot's pads, grip loop and soles are from the measurements; the
//! assertions check the scoring machinery and the rules that keep the
//! references from becoming tuning targets, never a score threshold.

use noto_core::actbench::Layer;
use noto_core::balance::Body;
use noto_core::effects::Rule;
use noto_core::parts::BodyPlan;
use noto_core::tactbench::{evaluate, holdout, library, TxResult};
use noto_core::{BaseKind, FingerCount};

const PLANS: [&str; 3] = ["mid", "pro", "wheel_lift"];
const BASES: [BaseKind; 3] = [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track];
/// The evaluation seed of the held-out draws (draw new ones per release, never tune against them).
const HOLDOUT_SEED: u64 = 20261027;

fn num(v: f32) -> String {
    if v != 0.0 && v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn print_result(plan: &str, base: BaseKind, r: &TxResult) {
    if let Some(why) = &r.skipped {
        println!("[TX-SUM] {plan:<6} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let layers: Vec<String> = Layer::ALL.iter().zip(r.layer_scores).map(|(l, s)| format!("{}={}", l.id(), s.map_or("—".into(), |v| format!("{v:.0}")))).collect();
    let scored = r.layer_scores.iter().any(|s| s.is_some());
    println!(
        "[TX-SUM] {plan:<6} {base:?} {} {} 점수 {} · 성공 {:.0}%{} · {}",
        r.id,
        r.name_ko,
        if scored { format!("{:.1}", r.score) } else { "— (기준 칸 없음)".into() },
        r.success_rate * 100.0,
        if r.gates.is_empty() { String::new() } else { format!(" (게이트: {})", r.gates.join("; ")) },
        layers.join(" ")
    );
    for c in &r.cells {
        let rule = match c.rule {
            Rule::Like { reference, tolerance } => format!("기준 {reference:.3} ± {tolerance:.3}"),
            Rule::AtLeast { reference, tolerance } => format!("하한 {reference:.3} (허용폭 {tolerance:.3})"),
            Rule::AtMost { limit } => format!("한계 ≤ {limit:.2}"),
            Rule::Raw => "기록".into(),
        };
        println!(
            "[TX] {plan} {base:?} {}/{}:{}{} {} ({rule}{}{}) — {}",
            r.id,
            c.layer.id(),
            c.axis,
            c.part.map(|p| format!(" [{p}]")).unwrap_or_default(),
            c.value.map_or("미구현".into(), num),
            c.z.map_or(String::new(), |z| format!(", z {z:+.1}")),
            c.score.map_or(String::new(), |s| format!(", 점수 {s:.0}")),
            c.cite
        );
    }
}

#[test]
fn tactile_answers_against_the_measurements() {
    for plan in PLANS {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            for s in library() {
                let r = evaluate(&body, &s);
                print_result(plan, base, &r);
                assert!(r.score.is_finite() && (0.0..=100.0).contains(&r.score), "{} {}", r.id, r.score);
                assert!(r.cells.iter().all(|c| c.value.map_or(true, f32::is_finite)), "{}", r.id);
                if r.skipped.is_none() {
                    // Every scenario asks about the outcome and at least one other layer, and scores something.
                    assert!(r.cells.iter().any(|c| c.layer == Layer::Outcome), "{} lacks an outcome", r.id);
                    assert!(r.cells.iter().any(|c| c.layer != Layer::Outcome), "{} has only an outcome", r.id);
                    assert!(r.layer_scores.iter().any(|s| s.is_some()), "{} scores nothing", r.id);
                }
            }
        }
    }
}

#[test]
fn the_same_engine_and_draws_give_the_same_answer() {
    let body = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
    for s in library() {
        assert_eq!(evaluate(&body, &s), evaluate(&body, &s), "{}", s.id);
    }
}

#[test]
fn held_out_draws_are_counted_not_tuned() {
    for plan in ["mid", "pro"] {
        for base in [BaseKind::HumanFoot, BaseKind::Track] {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let results: Vec<TxResult> = holdout(HOLDOUT_SEED, 6).iter().map(|s| evaluate(&body, s)).collect();
            let run: Vec<&TxResult> = results.iter().filter(|r| r.skipped.is_none()).collect();
            let passed = run.iter().filter(|r| r.passed).count();
            println!("[TX-HELD] {plan:<6} {base:?} 시드 {HOLDOUT_SEED}: 통과 {passed} / {} (건너뜀 {})", run.len(), results.len() - run.len());
            for r in &results {
                println!(
                    "[TX-HELD]   {} {} {}{}",
                    r.id,
                    r.name_ko,
                    r.skipped.clone().unwrap_or_else(|| format!("점수 {:.0} · 성공 {:.0}%", r.score, r.success_rate * 100.0)),
                    if r.gates.is_empty() { String::new() } else { format!(" · {}", r.gates.join("; ")) }
                );
            }
        }
    }
}

#[test]
fn the_engine_never_names_a_tactile_benchmark_source() {
    // The model files cite where their own numbers come from, never a source
    // the tactile benchmark scores with (`docs/research/tactile.md` §5): a
    // constant set from the reference would make the score say nothing.
    let files = [
        ("contact.rs", include_str!("../src/contact.rs")),
        ("sim.rs", include_str!("../src/sim.rs")),
        ("kin.rs", include_str!("../src/kin.rs")),
        ("balance.rs", include_str!("../src/balance.rs")),
        ("act/mod.rs", include_str!("../src/act/mod.rs")),
        ("act/grasp.rs", include_str!("../src/act/grasp.rs")),
        ("act/gait.rs", include_str!("../src/act/gait.rs")),
        ("act/material.rs", include_str!("../src/act/material.rs")),
        ("act/focus.rs", include_str!("../src/act/focus.rs")),
        ("world-sim world/acts.rs", include_str!("../../world-sim/src/world/acts.rs")),
    ];
    let sources = [
        // fingers and the grip loop
        "Delhaye", "Lefèvre", "Lefevre", "Thonnard", "Häger", "Hager", "Luciw", "Jarocka", "WAY-EEG-GAL", "Westling", "Johansson", "Flanagan", "Riso",
        // soles and toes
        "Jacob 1989", "Jacob H", "Hughes", "Klenerman", "Koo ", "Koo,", "Hessert", "Becker", "Chou", "Pedar", "tactbench",
    ];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the tactile benchmark source `{needle}`");
        }
    }
}
