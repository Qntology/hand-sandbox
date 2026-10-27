//! The environment N:N benchmark of the five body effects
//! (`noto_core::effects`) for the named presets on legs, wheels and tracks.
//! The lines this prints (`[FX]` cells, `[FX-SUM]` scenarios, `[FX-HELD]`
//! held-out) are the record of how far the engine is from the measurements;
//! the assertions check the scoring machinery and the rules that keep the
//! references from becoming tuning targets, never a score threshold.

use noto_core::balance::Body;
use noto_core::effects::{evaluate, holdout, library, Effect, FxResult, Rule};
use noto_core::parts::BodyPlan;
use noto_core::{BaseKind, FingerCount};

const PLANS: [&str; 4] = ["budget", "mid", "pro", "pro_arms"];
const BASES: [BaseKind; 3] = [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track];
/// The evaluation seed of the held-out draws (research §5: draw new ones per
/// release, never tune against them).
const HOLDOUT_SEED: u64 = 20261024;

/// Three decimals, or scientific notation for the small numbers (wear per hour).
fn num(v: f32) -> String {
    if v != 0.0 && v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn print_result(plan: &str, base: BaseKind, r: &FxResult, cells: bool) {
    if let Some(why) = &r.skipped {
        println!("[FX-SUM] {plan:<8} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let eff: Vec<String> = Effect::ALL.iter().zip(r.effect_scores).map(|(e, s)| format!("{}={}", e.id(), s.map_or("—".into(), |v| format!("{v:.0}")))).collect();
    println!(
        "[FX-SUM] {plan:<8} {base:?} {} {} 점수 {:.1}{} · {}",
        r.id,
        r.name_ko,
        r.score,
        if r.gates.is_empty() { String::new() } else { format!(" (게이트: {})", r.gates.join("; ")) },
        eff.join(" ")
    );
    if !cells {
        return;
    }
    for c in &r.cells {
        let rule = match c.rule {
            Rule::Like { reference, tolerance } => format!("기준 {reference:.3} ± {tolerance:.3}"),
            Rule::AtLeast { reference, tolerance } => format!("하한 {reference:.3} (허용폭 {tolerance:.3})"),
            Rule::AtMost { limit } => format!("한계 ≤ {limit:.2}"),
            Rule::Raw => "기록".into(),
        };
        println!(
            "[FX] {plan} {base:?} {}/{}:{} {} ({rule}{}{}) — {}",
            r.id,
            c.effect.id(),
            c.axis,
            c.value.map_or("미구현".into(), num),
            c.z.map_or(String::new(), |z| format!(", z {z:+.1}")),
            c.score.map_or(String::new(), |s| format!(", 점수 {s:.0}")),
            c.cite
        );
    }
}

#[test]
fn environment_effects_against_the_measurements() {
    for plan in PLANS {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            for s in library() {
                let r = evaluate(&body, &s);
                print_result(plan, base, &r, matches!(plan, "mid" | "pro"));
                assert!(r.score.is_finite() && (0.0..=100.0).contains(&r.score), "{} {}", r.id, r.score);
                if r.skipped.is_none() {
                    assert!(r.cells.iter().all(|c| c.value.map_or(true, f32::is_finite)), "{}", r.id);
                    // Every effect is asked about in every scenario.
                    for e in Effect::ALL {
                        assert!(r.cells.iter().any(|c| c.effect == e), "{} lacks {}", r.id, e.id());
                    }
                }
                // The same engine and scenario give the same answer.
                assert_eq!(evaluate(&body, &s).score, r.score);
            }
        }
    }
}

#[test]
fn held_out_draws_are_counted_not_tuned() {
    for plan in ["mid", "pro"] {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let results: Vec<FxResult> = holdout(HOLDOUT_SEED, 8).iter().map(|s| evaluate(&body, s)).collect();
            let run: Vec<&FxResult> = results.iter().filter(|r| r.skipped.is_none()).collect();
            let passed = run.iter().filter(|r| r.passed).count();
            println!("[FX-HELD] {plan:<8} {base:?} 시드 {HOLDOUT_SEED}: 통과 {passed} / {} (건너뜀 {})", run.len(), results.len() - run.len());
            for r in &results {
                println!(
                    "[FX-HELD]   {} {} {}{}",
                    r.id,
                    r.name_ko,
                    r.skipped.clone().unwrap_or_else(|| format!("점수 {:.0}", r.score)),
                    if r.gates.is_empty() { String::new() } else { format!(" · {}", r.gates.join("; ")) }
                );
            }
        }
    }
}

#[test]
fn the_engine_never_names_a_benchmark_source() {
    // The model files may cite where their own numbers come from, but never a
    // source the benchmark scores against (research §5): a constant set from
    // the reference would make the score say nothing.
    let balance = include_str!("../src/balance.rs");
    let controller = balance.split("/// Context the references depend on.").next().unwrap();
    let files = [
        ("balance.rs (controller part)", controller),
        ("dynamics.rs", include_str!("../src/dynamics.rs")),
        ("body.rs", include_str!("../src/body.rs")),
        ("parts.rs", include_str!("../src/parts.rs")),
        ("morph.rs", include_str!("../src/morph.rs")),
        ("surface.rs", include_str!("../src/surface.rs")),
        ("contact.rs", include_str!("../src/contact.rs")),
        ("act/collide.rs", include_str!("../src/act/collide.rs")),
        ("act/fall.rs", include_str!("../src/act/fall.rs")),
        ("act/cocarry.rs", include_str!("../src/act/cocarry.rs")),
        ("world-sim world.rs", include_str!("../../world-sim/src/world.rs")),
        ("world-sim nav.rs", include_str!("../../world-sim/src/nav.rs")),
        ("world-sim terrain.rs", include_str!("../../world-sim/src/terrain.rs")),
        ("world-sim site.rs", include_str!("../../world-sim/src/site.rs")),
        ("world-sim fixture.rs", include_str!("../../world-sim/src/fixture.rs")),
    ];
    let sources = ["Hanson", "Beschorner", "Rodríguez-Martínez", "Lejeune", "Knapik", "Middleton", "Frey Law", "Frey-Law", "maxon", "effects::"];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the benchmark source `{needle}`");
        }
    }
}
