use noto_core::act::collide::{self, BumpInput, Obstacle};
use noto_core::act::fall::{self, FallCause, FallInput};
use noto_core::actbench::Layer;
use noto_core::balance::Body;
use noto_core::collbench::{evaluate, evaluate_or_skip, library, BenchResult, CxKind};
use noto_core::effects::Rule;
use noto_core::kin::{self, KinLoads, KinPose};
use noto_core::parts::BodyPlan;
use noto_core::{BaseKind, FingerCount, Surface};

const PLANS: [&str; 3] = ["budget", "mid", "pro"];
const BASES: [BaseKind; 3] = [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track];

fn num(v: f32) -> String {
    if v != 0.0 && v.abs() < 0.01 {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn print_result(tag: &str, plan: &str, base: BaseKind, r: &BenchResult, cells: bool) {
    if let Some(why) = &r.skipped {
        println!("[{tag}-SUM] {plan:<6} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let layers: Vec<String> = Layer::ALL.iter().zip(r.layer_scores).map(|(l, s)| format!("{}={}", l.id(), s.map_or("—".into(), |v| format!("{v:.0}")))).collect();
    println!(
        "[{tag}-SUM] {plan:<6} {base:?} {} {} 점수 {}{} · {}",
        r.id,
        r.name_ko,
        if r.scored() { format!("{:.1}", r.score) } else { "— (기준 칸 없음)".into() },
        if r.gates.is_empty() { String::new() } else { format!(" (게이트: {})", r.gates.join("; ")) },
        layers.join(" ")
    );
    if !cells {
        return;
    }
    for n in &r.notes {
        println!("[{tag}-WHY] {plan:<6} {base:?} {} {n}", r.id);
    }
    for c in &r.cells {
        let rule = match c.rule {
            Rule::Like { reference, tolerance } => format!("기준 {reference:.3} ± {tolerance:.3}"),
            Rule::AtLeast { reference, tolerance } => format!("하한 {reference:.3} (허용폭 {tolerance:.3})"),
            Rule::AtMost { limit } => format!("한계 ≤ {limit:.2}"),
            Rule::Raw => "기록".into(),
        };
        println!(
            "[{tag}] {plan} {base:?} {}/{}:{}{} {} ({rule}{}{}){}",
            r.id,
            c.layer.id(),
            c.axis,
            c.part.map(|p| format!(" [{p}]")).unwrap_or_default(),
            c.value.map_or("답 없음".into(), num),
            c.z.map_or(String::new(), |z| format!(", z {z:+.1}")),
            c.score.map_or(String::new(), |s| format!(", 점수 {s:.0}")),
            if c.cite.is_empty() { String::new() } else { format!(" — {}", c.cite) }
        );
    }
}

#[test]
fn collisions_and_falls_against_the_measurements() {
    for plan in PLANS {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            for s in library() {
                let r = evaluate_or_skip(&body, &s);
                print_result("CX", plan, base, &r, plan != "budget");
                assert!(r.score.is_finite() && (0.0..=100.0).contains(&r.score), "{} {}", r.id, r.score);
                assert!(r.cells.iter().all(|c| c.value.map_or(true, f32::is_finite)), "{} {:?}", r.id, r.cells);
                if r.skipped.is_none() {
                    assert!(r.cells.iter().any(|c| c.layer == Layer::Outcome), "{} lacks an outcome", r.id);
                }
            }
        }
    }
}

#[test]
fn the_same_engine_gives_the_same_answer() {
    let body = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
    for s in library() {
        assert_eq!(evaluate(&body, &s), evaluate(&body, &s), "{}", s.id);
    }
}

#[test]
fn the_physics_holds_its_own_laws_across_the_presets() {
    for plan in PLANS {
        for base in BASES {
            let b = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            let slow = collide::bump(&BumpInput::wall(&b, 0.3, Obstacle::Concrete, Surface::Firm));
            let fast = collide::bump(&BumpInput::wall(&b, 0.9, Obstacle::Concrete, Surface::Firm));
            let pk = |c: &collide::Collision| c.pulse.as_ref().map_or(0.0, |p| p.peak_n);
            assert!(pk(&fast) > pk(&slow), "{plan} {base:?}: {} vs {}", pk(&fast), pk(&slow));
            assert!(fast.v_after[0] < 0.9, "{plan} {base:?}: the wall takes speed off");
            let f = fall::fall(&FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Pushed, 180.0, 0.3, Surface::Firm));
            assert!(f.fell && f.lying.down(), "{plan} {base:?}: {:?}", f.rollout.events);
            if base == BaseKind::HumanFoot {
                assert_eq!(f.get_up.can, !f.get_up.needs_help, "{plan}: {:?}", f.get_up);
            } else {
                assert!(!f.get_up.can && f.get_up.needs_help && f.get_up.righting_force_n > 0.0, "{plan} {base:?}: {:?}", f.get_up);
            }
            let main = f.main.as_ref().expect("a fall strikes");
            let com_z = kin::solve(&b, &KinPose::stand_compact(0.0), &KinLoads::default()).com[2];
            assert!(main.energy_j <= b.mass_kg() * 9.81 * com_z * 1.05 + 1.0, "{plan} {base:?}: {} J from a {com_z} m CoM", main.energy_j);
        }
    }
}

#[test]
fn every_scenario_kind_runs_on_every_base() {
    let kinds: Vec<CxKind> = library().iter().map(|s| s.kind).collect();
    assert!(kinds.iter().any(|k| matches!(k, CxKind::WallBump)));
    assert!(kinds.iter().any(|k| matches!(k, CxKind::Fall { .. })));
    assert!(kinds.iter().any(|k| matches!(k, CxKind::Dropped { .. })));
    for base in BASES {
        let b = Body::new(&BodyPlan::named("mid").unwrap(), base, FingerCount::Five);
        let run = library().iter().filter(|s| evaluate_or_skip(&b, s).skipped.is_none()).count();
        assert!(run >= library().len() - 2, "{base:?} ran {run}");
    }
}

#[test]
fn the_engine_never_names_a_collision_benchmark_source() {
    let files = [
        ("kin.rs", include_str!("../src/kin.rs")),
        ("act/mod.rs", include_str!("../src/act/mod.rs")),
        ("act/collide.rs", include_str!("../src/act/collide.rs")),
        ("act/fall.rs", include_str!("../src/act/fall.rs")),
        ("act/cocarry.rs", include_str!("../src/act/cocarry.rs")),
        ("act/material.rs", include_str!("../src/act/material.rs")),
        ("act/gait.rs", include_str!("../src/act/gait.rs")),
        ("act/drive.rs", include_str!("../src/act/drive.rs")),
        ("act/push.rs", include_str!("../src/act/push.rs")),
    ];
    let sources = [
        "Choi", "Wakeling", "Robinovitch", "Kroonenberg", "Hsiao", "Feldman", "Laing", "Kajita", "HRP-2", "Bioloid", "Karydis", "Wimer", "Llado", "Sentis", "Trikey", "Kumar", "Warnick", "Tan, Eng", "collbench",
    ];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the collision benchmark source `{needle}`");
        }
    }
}
