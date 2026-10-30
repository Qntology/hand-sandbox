// The standard build: a person's size on legs, a 1-tonne lift truck on wheels. The same balance, kinematics and action
// models as the compact robot, on the build's own lengths and parts.
use noto_core::act::{fork, gait, grasp};
use noto_core::dims::Build;
use noto_core::kin::{self, KinLoads, KinPose};
use noto_core::{BaseKind, Body, BodyPlan, FingerCount, Tier};

fn body(tier: Tier, base: BaseKind) -> Body {
    Body::new(&BodyPlan { build: Build::Standard, ..BodyPlan::uniform(tier) }, base, FingerCount::Five)
}

#[test]
fn a_standard_humanoid_has_a_persons_size_and_stands() {
    for tier in Tier::ALL {
        let b = body(tier, BaseKind::HumanFoot);
        let r = kin::solve(&b, &KinPose::stand(&b, 0.0), &KinLoads::default());
        let eye = kin::head_camera(&b, &KinPose::stand(&b, 0.0)).pos[2];
        let top = r.boxes.iter().filter(|x| x.name == "head").map(|x| x.center[2] + x.half[2]).fold(0.0, f32::max);
        println!("{}: {:.1} kg · 눈 {eye:.2} m · 머리 끝 {top:.2} m · CoM {:.2} m · 균형 {:.2} · 무릎 {} ({:.0} N·m)", tier.id(), r.mass_kg, r.com[2], r.margin, b.leg.knee.label(), b.leg.knee.peak_nm);
        assert!(r.out_of_range.is_empty(), "{:?}", r.out_of_range);
        assert!((1.45..1.60).contains(&eye), "{eye}");
        assert!((1.55..1.75).contains(&top), "{top}");
        assert!(r.margin > 0.3, "{}", r.margin);
    }
    let mid = body(Tier::Mid, BaseKind::HumanFoot);
    assert!((40.0..70.0).contains(&mid.mass_kg()), "{}", mid.mass_kg());
    assert_ne!(mid.leg.knee.label(), mid.leg.knee.name, "a scaled actuator says so");
}

#[test]
fn a_standard_humanoid_walks_at_a_persons_pace() {
    let b = body(Tier::Mid, BaseKind::HumanFoot);
    let g = gait::plan(&b, 1.1, 0.6, 0.0);
    println!("{g:?}");
    assert!((0.9..1.3).contains(&g.speed), "{}", g.speed);
    assert!((0.35..0.85).contains(&g.step_m), "{}", g.step_m);
    let limits = b.drive_limits(&BaseKind::HumanFoot.spec(), 0.0);
    println!("{limits:?}");
    assert!(limits.max_speed_mps > 0.9, "{limits:?}");
    assert!(limits.capacity_kg > 4.2, "{limits:?}");
}

#[test]
fn a_standard_humanoid_picks_a_log_off_the_ground() {
    let b = body(Tier::Mid, BaseKind::HumanFoot);
    let size = [0.50, 0.14, 0.14];
    let hand = noto_core::NotoConfig::preset(FingerCount::Five).build().unwrap().scaled(b.dims.k);
    let r = grasp::grasp(&grasp::GraspInput::front(&b, &hand, noto_core::act::material::ItemMaterial::Wood, 4.2, size, noto_core::Surface::Firm, true, 0.0, 0.01, 3));
    println!("{:?} · {:?} · {:.1} s · {:?}", r.outcome, r.metrics.iter().take(5).collect::<Vec<_>>(), r.duration_s, r.notes);
    assert!(r.outcome.ok, "{:?}", r.outcome);
}

#[test]
fn a_standard_lift_truck_is_a_truck_and_lifts_a_log() {
    for base in [BaseKind::Wheel, BaseKind::Mecanum, BaseKind::Track] {
        let b = body(Tier::Mid, base);
        assert!(fork::applies(&b));
        let r = kin::solve(&b, &KinPose::stand(&b, 0.0), &KinLoads::default());
        let eye = kin::head_camera(&b, &KinPose::stand(&b, 0.0)).pos[2];
        println!("{base:?}: {:.0} kg · 눈 {eye:.2} m · 축거 {:.2} × 폭 {:.2} m · 포크 뿌리 {:.2} m · 끝 {:.2} m · 균형 {:.2}", r.mass_kg, b.support_len_m, b.support_width_m, fork::heel_x(&b), fork::heel_x(&b) + b.dims.tine_len_m, r.margin);
        assert!((2000.0..3500.0).contains(&r.mass_kg), "{}", r.mass_kg);
        assert!((1.5..2.0).contains(&eye), "{eye}");
        let fit = fork::check(&b, 4.2, [0.50, 0.14, 0.14], 0.0, 0.0).unwrap();
        println!("  적재 {fit:?}");
        assert!(fit.capacity_kg >= 900.0, "a 1-tonne truck: {fit:?}");
        let pick = fork::pick(&b, b.dims.fork_rest_m, fork::standoff_x(&b, 0.50), 4.2, [0.50, 0.14, 0.14], 0.0, 0.0, 0.0);
        println!("  {:?} · {:.1} s", pick.outcome, pick.duration_s);
        assert!(pick.outcome.ok, "{:?}", pick.outcome);
    }
}
