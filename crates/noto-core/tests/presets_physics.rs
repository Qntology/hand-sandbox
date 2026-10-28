use noto_core::balance::{Body, FeedForward, Posture, Source};
use noto_core::dynamics::{simulate, DynCfg, Rollout};
use noto_core::parts::*;
use noto_core::{BaseKind, FingerCount};

fn plan_with(seg: &str, tier: Tier) -> BodyPlan {
    let mut p = BodyPlan::default();
    match seg {
        "head" => p.head = tier,
        "chest" => p.chest = tier,
        "belly" => p.belly = tier,
        "arm" => p.arm = tier,
        "leg" => p.leg = tier,
        _ => unreachable!(),
    }
    p
}

fn body(plan: &BodyPlan, base: BaseKind) -> Body {
    Body::new(plan, base, FingerCount::Five)
}

fn joint_peak(r: &Rollout, name: &str) -> f32 {
    r.joints.iter().find(|j| j.name == name).map_or(f32::NAN, |j| j.peak_nm)
}

fn summary(r: &Rollout, name: &str) -> (f32, f32, f32) {
    r.summary.iter().find(|s| s.name == name).map_or((f32::NAN, 0.0, 0.0), |s| (s.peak_ratio, s.saturated_s, s.stalled_s))
}

fn hold(b: &Body, p: &Posture) -> Rollout {
    simulate(b, p, p, &FeedForward::default(), 500.0, &DynCfg { settle_s: 0.3, ..DynCfg::default() })
}

#[test]
fn every_preset_reaches_its_joints_in_the_dynamics() {
    for tier in Tier::ALL {
        let p = BodyPlan::uniform(tier);
        let r = hold(&body(&p, BaseKind::HumanFoot), &Posture::quiet(0.0));
        let (arm, belly, head, leg) = (p.arm_preset(), p.belly_preset(), p.head_preset(), p.leg_preset());
        for (joint, act) in [("shoulder_r", arm.shoulder), ("elbow_l", arm.elbow), ("waist_yaw", belly.waist), ("neck", head.neck), ("knee_r", leg.knee), ("ankle_l", leg.ankle)] {
            assert_eq!(joint_peak(&r, joint), act.peak_nm, "{tier:?} {joint}");
        }
        let pitch = r.joints.iter().find(|j| j.name == "waist_pitch").unwrap();
        assert_eq!(pitch.kind, if belly.pitch.is_some() { "sim" } else { "structure" }, "{tier:?}: waist pitch is an actuator only when the belly preset has one");
        let wheeled = hold(&body(&p, BaseKind::Wheel), &Posture::quiet(0.0));
        assert_eq!(joint_peak(&wheeled, "drive"), 2.0 * p.chassis_preset().drive_force_n);
        println!("[PRESET] {:<6} joints: shoulder {} · elbow {} · waist pitch {} · neck {} · knee {} · ankle {} · drive {:.0} N", tier.id(), arm.shoulder.name, arm.elbow.name, belly.pitch.map_or("구조", |a| a.name), head.neck.name, leg.knee.name, leg.ankle.name, p.chassis_preset().drive_force_n);
    }
}

#[test]
fn each_segment_changes_what_it_should() {
    let ff = FeedForward::default();
    let base = body(&BodyPlan::default(), BaseKind::HumanFoot);
    let base_w = body(&BodyPlan::default(), BaseKind::Wheel);
    for tier in [Tier::Budget, Tier::Pro] {
        let h = body(&plan_with("head", tier), BaseKind::Wheel);
        let slope = Posture { source: Source::Slope, slope_deg: 8.0, ..Posture::quiet(0.0) };
        let (a, b) = (hold(&h, &slope), hold(&base_w, &slope));
        let (ha, hb) = (a.samples.last().unwrap().head_pitch_deg.abs(), b.samples.last().unwrap().head_pitch_deg.abs());
        if tier == Tier::Pro {
            assert!(ha < hb, "a pro neck keeps the camera more level on a slope: {ha} vs {hb}");
        } else {
            assert!(ha > hb, "{ha} vs {hb}");
        }
        println!("[PRESET] head  {:<6} camera tilt on 8° slope {ha:.2}° (mid {hb:.2}°), neck {}", tier.id(), h.head.neck.name);

        let c = body(&plan_with("chest", tier), BaseKind::HumanFoot);
        assert_ne!(c.battery_j(), base.battery_j());
        assert_ne!(c.mass_kg(), base.mass_kg());
        println!("[PRESET] chest {:<6} battery {:.0} Wh · mass {:.1} kg (mid {:.1}) · CoM {:.3} m (mid {:.3})", tier.id(), c.chest.battery_wh, c.mass_kg(), base.mass_kg(), c.com_height_m(), base.com_height_m());

        let bl = body(&plan_with("belly", tier), BaseKind::HumanFoot);
        let lift = Posture::arm_lift(4.0, true, 0.0);
        let (rb, r0) = (bl.solve(&lift, &ff, 500.0), base.solve(&lift, &ff, 500.0));
        if tier == Tier::Pro {
            assert!(rb.trunk_lean_deg.abs() > 0.1 && r0.trunk_lean_deg == 0.0, "{} vs {}", rb.trunk_lean_deg, r0.trunk_lean_deg);
            assert!(rb.restore_frac > r0.restore_frac);
        }
        let lim_b = bl.drive_limits(&BaseKind::Wheel.spec(), 0.0);
        let _ = lim_b;
        println!("[PRESET] belly {:<6} lean for a 4 kg lift {:.1}° (mid {:.1}°), restore {:.2} (mid {:.2})", tier.id(), rb.trunk_lean_deg, r0.trunk_lean_deg, rb.restore_frac, r0.restore_frac);

        let ar = body(&plan_with("arm", tier), BaseKind::Wheel);
        let one = Posture::arm_lift(2.4, false, 0.0);
        let (sa, s0) = (ar.solve(&one, &ff, 500.0), base_w.solve(&one, &ff, 500.0));
        let (ea, e0) = (sa.joint("elbow_r").unwrap().peak_ratio, s0.joint("elbow_r").unwrap().peak_ratio);
        let (sh_a, sh_0) = (sa.joint("shoulder_r").unwrap().peak_ratio, s0.joint("shoulder_r").unwrap().peak_ratio);
        if tier == Tier::Budget {
            assert!(ea > e0 && sh_a > sh_0, "budget: elbow {ea} vs {e0}, shoulder {sh_a} vs {sh_0}");
        } else {
            assert!(sh_a < sh_0, "pro shoulder {sh_a} vs mid {sh_0}");
            assert!(ea >= e0 * 0.95, "pro elbow {ea} vs mid {e0}");
        }
        println!("[PRESET] arm   {:<6} 2.4 kg one-arm lift: shoulder {:.0}% (mid {:.0}%) · elbow {:.0}% (mid {:.0}%)", tier.id(), sh_a * 100.0, sh_0 * 100.0, ea * 100.0, e0 * 100.0);

        let lg = body(&plan_with("leg", tier), BaseKind::HumanFoot);
        let (ll, l0) = (lg.drive_limits(&BaseKind::HumanFoot.spec(), 0.0), base.drive_limits(&BaseKind::HumanFoot.spec(), 0.0));
        assert!(if tier == Tier::Budget { ll.max_step_m < l0.max_step_m } else { ll.max_step_m >= l0.max_step_m }, "{} vs {}", ll.max_step_m, l0.max_step_m);
        println!("[PRESET] leg   {:<6} step {:.0} cm (mid {:.0}) · speed {:.2} m/s (mid {:.2})", tier.id(), ll.max_step_m * 100.0, l0.max_step_m * 100.0, ll.max_speed_mps, l0.max_speed_mps);

        let ch = body(&plan_with("leg", tier), BaseKind::Wheel);
        let (cl, c0) = (ch.drive_limits(&BaseKind::Wheel.spec(), 0.0), base_w.drive_limits(&BaseKind::Wheel.spec(), 0.0));
        println!("[PRESET] chassis {:<6} slope {:.1}° (mid {:.1}°) · speed {:.2} m/s (mid {:.2}) · {}", tier.id(), cl.max_slope_deg, c0.max_slope_deg, cl.max_speed_mps, c0.max_speed_mps, ch.chassis.name);
        assert!(cl.max_speed_mps != c0.max_speed_mps || cl.max_slope_deg != c0.max_slope_deg);
    }
}

#[test]
fn hand_and_battery_options_reach_the_physics() {
    let two = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Two);
    let five = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
    let p = Posture::arm_raise(0.0);
    let (a, b) = (hold(&two, &p), hold(&five, &p));
    let (ta, tb) = (summary(&a, "shoulder_r").0, summary(&b, "shoulder_r").0);
    assert!(ta < tb, "a five-finger hand is heavier: {ta} vs {tb}");
    let low = Body::new(&BodyPlan { battery_low: true, ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
    let high = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
    assert!(low.com_height_m() < high.com_height_m());
    let (rl, rh) = (hold(&low, &Posture::quiet(0.0)), hold(&high, &Posture::quiet(0.0)));
    println!(
        "[PRESET] hand  2 vs 5 fingers: shoulder at strike {:.0}% vs {:.0}% · battery low CoM {:.3} m vs {:.3} m (dynamic CoM {:.3} vs {:.3})",
        ta * 100.0,
        tb * 100.0,
        low.com_height_m(),
        high.com_height_m(),
        rl.samples.last().unwrap().com[2],
        rh.samples.last().unwrap().com[2]
    );
}

#[test]
fn the_dynamics_never_allow_a_lift_the_statics_refuse() {
    let ff = FeedForward::default();
    for tier in Tier::ALL {
        let b = body(&plan_with("arm", tier), BaseKind::Wheel);
        let mut line = Vec::new();
        for kg in [1.4f32, 2.4, 2.6, 4.0] {
            let st = b.solve(&Posture::arm_lift(kg, false, 0.0), &ff, 500.0).feasible;
            let from = Posture { hand_load_kg: [kg, 0.0], ..Posture::quiet(0.0) };
            let r = simulate(&b, &from, &Posture::arm_lift(kg, false, 0.0), &ff, 500.0, &DynCfg::default());
            assert!(!r.feasible || st, "{tier:?} {kg} kg: dynamics lift what the statics refuse ({:?})", r.notes);
            let (pk, sat, stall) = summary(&r, "elbow_r");
            line.push(format!("{kg} kg 정적 {} / 동적 {} (팔꿈치 {:.0}% · 포화 {sat:.2} s · 멈춤 {stall:.2} s)", if st { "가능" } else { "불가" }, if r.feasible { "가능" } else { "불가" }, pk * 100.0));
        }
        println!("[PRESET] lift  {:<6} {}", tier.id(), line.join(" | "));
    }
}
