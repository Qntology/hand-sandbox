//! The action N:N benchmark (`noto_core::actbench`) for the named presets on
//! legs, wheels and tracks. The lines this prints (`[AX]` cells, `[AX-SUM]`
//! scenarios, `[AX-HELD]` held-out) are the record of how far the robot's
//! actions are from the measurements; the assertions check the scoring
//! machinery and the rules that keep the references from becoming tuning
//! targets, never a score threshold.

use noto_core::actbench::{evaluate, holdout, library, AxResult, Layer};
use noto_core::balance::Body;
use noto_core::effects::Rule;
use noto_core::parts::BodyPlan;
use noto_core::{BaseKind, FingerCount};

/// The named presets benchmarked: the three tiers, and the wheeled base with a torso lift (v0.7), the one wheeled body that reaches the floor.
const PLANS: [&str; 4] = ["budget", "mid", "pro", "wheel_lift"];
const BASES: [BaseKind; 3] = [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track];
/// The evaluation seed of the held-out draws (draw new ones per release, never tune against them).
const HOLDOUT_SEED: u64 = 20261024;

fn num(v: f32) -> String {
    if v != 0.0 && v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn print_result(plan: &str, base: BaseKind, r: &AxResult, cells: bool) {
    if let Some(why) = &r.skipped {
        println!("[AX-SUM] {plan:<6} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let layers: Vec<String> = Layer::ALL.iter().zip(r.layer_scores).map(|(l, s)| format!("{}={}", l.id(), s.map_or("—".into(), |v| format!("{v:.0}")))).collect();
    let fails: Vec<String> = r.failures.iter().map(|(k, n)| format!("{k}×{n}")).collect();
    // A scenario with nothing to compare against has no score (only recorded cells).
    let scored = r.layer_scores.iter().any(|s| s.is_some());
    println!(
        "[AX-SUM] {plan:<6} {base:?} {} {} 점수 {} · 성공 {:.0}%{}{} · {}",
        r.id,
        r.name_ko,
        if scored { format!("{:.1}", r.score) } else { "— (기준 칸 없음)".into() },
        r.success_rate * 100.0,
        if fails.is_empty() { String::new() } else { format!(" (실패 {})", fails.join(", ")) },
        if r.gates.is_empty() { String::new() } else { format!(" (게이트: {})", r.gates.join("; ")) },
        layers.join(" ")
    );
    for (code, why) in &r.reasons {
        println!("[AX-WHY] {plan:<6} {base:?} {} {code}: {why}", r.id);
    }
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
            "[AX] {plan} {base:?} {}/{}:{}{} {} ({rule}{}{}) — {}",
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
fn actions_against_the_measurements() {
    for plan in PLANS {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            for s in library() {
                let r = evaluate(&body, &s);
                print_result(plan, base, &r, matches!(plan, "mid" | "pro" | "wheel_lift"));
                assert!(r.score.is_finite() && (0.0..=100.0).contains(&r.score), "{} {}", r.id, r.score);
                assert!(r.cells.iter().all(|c| c.value.map_or(true, f32::is_finite)), "{}", r.id);
                if r.skipped.is_none() {
                    // Every scenario asks about the outcome and at least one other layer.
                    assert!(r.cells.iter().any(|c| c.layer == Layer::Outcome), "{} lacks an outcome", r.id);
                    assert!(r.cells.iter().any(|c| c.layer != Layer::Outcome), "{} has only an outcome", r.id);
                }
            }
        }
    }
}

#[test]
fn the_same_engine_and_draws_give_the_same_answer() {
    let body = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
    for s in library().into_iter().filter(|s| s.trials <= 30) {
        assert_eq!(evaluate(&body, &s), evaluate(&body, &s), "{}", s.id);
    }
}

#[test]
fn held_out_draws_are_counted_not_tuned() {
    for plan in ["mid", "pro"] {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let results: Vec<AxResult> = holdout(HOLDOUT_SEED, 6).iter().map(|s| evaluate(&body, s)).collect();
            let run: Vec<&AxResult> = results.iter().filter(|r| r.skipped.is_none()).collect();
            let passed = run.iter().filter(|r| r.passed).count();
            println!("[AX-HELD] {plan:<6} {base:?} 시드 {HOLDOUT_SEED}: 통과 {passed} / {} (건너뜀 {})", run.len(), results.len() - run.len());
            for r in &results {
                println!(
                    "[AX-HELD]   {} {} {}{}",
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
fn the_engine_never_names_an_action_benchmark_source() {
    // The model files cite where their own numbers come from, never a source
    // the action benchmark scores with (research §7): a constant set from the
    // reference would make the score say nothing.
    let files = [
        ("kin.rs", include_str!("../src/kin.rs")),
        ("morph.rs", include_str!("../src/morph.rs")),
        ("act/mod.rs", include_str!("../src/act/mod.rs")),
        ("act/reach.rs", include_str!("../src/act/reach.rs")),
        ("act/grasp.rs", include_str!("../src/act/grasp.rs")),
        ("act/push.rs", include_str!("../src/act/push.rs")),
        ("act/gait.rs", include_str!("../src/act/gait.rs")),
        ("act/drive.rs", include_str!("../src/act/drive.rs")),
        ("act/material.rs", include_str!("../src/act/material.rs")),
        ("act/focus.rs", include_str!("../src/act/focus.rs")),
        ("contact.rs", include_str!("../src/contact.rs")),
        ("sim.rs", include_str!("../src/sim.rs")),
        ("act/collide.rs", include_str!("../src/act/collide.rs")),
        ("act/fall.rs", include_str!("../src/act/fall.rs")),
        ("act/cocarry.rs", include_str!("../src/act/cocarry.rs")),
        ("world-sim world.rs", include_str!("../../world-sim/src/world.rs")),
        ("world-sim world/acts.rs", include_str!("../../world-sim/src/world/acts.rs")),
        ("world-sim nav.rs", include_str!("../../world-sim/src/nav.rs")),
        ("world-sim site.rs", include_str!("../../world-sim/src/site.rs")),
        ("world-sim fixture.rs", include_str!("../../world-sim/src/fixture.rs")),
    ];
    let sources = [
        "Guillery", "Johansson", "Westling", "Flanagan", "Menzer", "Reiser", "Bazrgari", "Kingma", "Plamondon", "Hwang", "von Arx", "Dolan", "NIOSH", "Waters", "Liberty Mutual", "Snook", "Steele",
        "Mahler", "Dex-Net", "QT-Opt", "Kalashnikov", "Levine", "ten Pas", "Morrison", "Berscheid", "Yu, Bauza", "Bauza", "Beschorner", "actbench",
        // walking and driving
        "Wang 2017", "Jafarnezhadgero", "Zhang 2013", "Wallace", "Baines", "Lockhart", "Pai 2010", "Yang 2012", "Brady", "Stansfield", "Vaughan", "Nilsson", "Keller", "Holowka", "Warmerdam", "Agarwal", "Yoshida",
    ];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the benchmark source `{needle}`");
        }
    }
}
