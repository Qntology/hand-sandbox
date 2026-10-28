use noto_core::act::cocarry::{self, Carrier, CarryOpts, Load};
use noto_core::actbench::Layer;
use noto_core::balance::Body;
use noto_core::collbench::BenchResult;
use noto_core::coopbench::{evaluate, library, sized_kg, LOG_DIA_M, LOG_GRIP_M, LOG_LEN_M};
use noto_core::effects::Rule;
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

fn print_result(plan: &str, base: BaseKind, r: &BenchResult, cells: bool) {
    if let Some(why) = &r.skipped {
        println!("[KX-SUM] {plan:<6} {base:?} {} {} — 건너뜀: {why}", r.id, r.name_ko);
        return;
    }
    let layers: Vec<String> = Layer::ALL.iter().zip(r.layer_scores).map(|(l, s)| format!("{}={}", l.id(), s.map_or("—".into(), |v| format!("{v:.0}")))).collect();
    println!("[KX-SUM] {plan:<6} {base:?} {} {} 점수 {} · {}", r.id, r.name_ko, if r.scored() { format!("{:.1}", r.score) } else { "— (기준 칸 없음)".into() }, layers.join(" "));
    if !cells {
        return;
    }
    for n in &r.notes {
        println!("[KX-WHY] {plan:<6} {base:?} {} {n}", r.id);
    }
    for c in &r.cells {
        let rule = match c.rule {
            Rule::Like { reference, tolerance } => format!("기준 {reference:.3} ± {tolerance:.3}"),
            Rule::AtLeast { reference, tolerance } => format!("하한 {reference:.3} (허용폭 {tolerance:.3})"),
            Rule::AtMost { limit } => format!("한계 ≤ {limit:.2}"),
            Rule::Raw => "기록".into(),
        };
        println!(
            "[KX] {plan} {base:?} {}/{}:{}{} {} ({rule}{}{}){}",
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
fn shared_carries_against_the_measurements() {
    for plan in PLANS {
        for base in BASES {
            let body = Body::new(&BodyPlan::named(plan).unwrap(), base, FingerCount::Five);
            for s in library() {
                let r = evaluate(&body, &s);
                print_result(plan, base, &r, plan != "budget");
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
    let body = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
    for s in library() {
        assert_eq!(evaluate(&body, &s), evaluate(&body, &s), "{}", s.id);
    }
}

#[test]
fn the_statics_hold_their_own_laws() {
    for plan in PLANS {
        let b = Body::new(&BodyPlan::named(plan).unwrap(), BaseKind::HumanFoot, FingerCount::Five);
        let m = sized_kg(&b);
        let level = [Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, LOG_GRIP_M, 0.0, 0.0, Surface::Firm)];
        let log = Load::log(m, LOG_LEN_M, LOG_DIA_M, true);
        let cc = cocarry::share(&log, &level, &CarryOpts::default());
        let sum: f32 = cc.members.iter().map(|x| x.f_n).sum();
        assert!((sum - log.weight_n()).abs() < 1e-3 * log.weight_n(), "{plan}: {sum}");
        assert!((cc.members[0].share - 0.5).abs() < 1e-3, "{plan}: {}", cc.members[0].share);
        let alone = cocarry::share(&log, &level[..1], &CarryOpts::default());
        assert!(!alone.ok, "{plan}: one robot carries {m} kg alone");
        let dz = LOG_GRIP_M * 10f32.to_radians().tan();
        let slope = |up: f32| {
            let l = Load { com_up_m: up, ..log };
            cocarry::share(&l, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, LOG_GRIP_M, 0.0, dz, Surface::Firm)], &CarryOpts::default()).members[0].share
        };
        assert!((slope(0.0) - 0.5).abs() < 1e-3 && slope(0.1) > 0.5 && slope(0.3) > slope(0.1), "{plan}");
    }
}

#[test]
fn the_engine_never_names_a_cooperation_benchmark_source() {
    let files = [
        ("kin.rs", include_str!("../src/kin.rs")),
        ("act/mod.rs", include_str!("../src/act/mod.rs")),
        ("act/cocarry.rs", include_str!("../src/act/cocarry.rs")),
        ("act/collide.rs", include_str!("../src/act/collide.rs")),
        ("act/fall.rs", include_str!("../src/act/fall.rs")),
        ("act/grasp.rs", include_str!("../src/act/grasp.rs")),
        ("act/gait.rs", include_str!("../src/act/gait.rs")),
        ("balance.rs", include_str!("../src/balance.rs")),
    ];
    let sources = ["Lavender", "Dennis", "Barrett", "Sharp", "Visser", "Lanini", "Ijspeert", "Kluth", "Sambhus", "DGUV", "Loiu", "coopbench"];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the cooperation benchmark source `{needle}`");
        }
    }
}
