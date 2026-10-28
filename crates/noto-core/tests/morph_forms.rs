use noto_core::act::focus::{self, FocusCtx};
use noto_core::act::grasp::{self, GraspInput, ReleaseInput};
use noto_core::act::material::ItemMaterial;
use noto_core::act::reach::{self, ReachGoal};
use noto_core::act::ActKind;
use noto_core::balance::{Body, FeedForward, Posture, Source, G};
use noto_core::body::{compute, BodyInputs, BodyLimits, PART_IDS};
use noto_core::dynamics::{simulate, DynCfg};
use noto_core::kin::{self, KinLoads, KinPose};
use noto_core::morph::{HeadMount, Morphology, FORMS, STRETCH_CAMERA_H_MM};
use noto_core::parts::*;
use noto_core::{BaseKind, FingerCount, NotoConfig, Surface};

fn body_of(form: &str, tier: Tier) -> Body {
    let m = Morphology::named(form).unwrap();
    Body::new(&BodyPlan { morph: m, ..BodyPlan::uniform(tier) }, m.lower.base(), FingerCount::Five)
}

fn hand_for(base: BaseKind) -> noto_core::Hand {
    let mut cfg = NotoConfig::preset(FingerCount::Five);
    cfg.base = base;
    cfg.build().unwrap()
}

fn analogue_kg(form: &str) -> Option<(f32, Tier)> {
    match form {
        "mast_gripper" => Some((23.0, Tier::Mid)),
        "deck_gripper" => Some((1.7 + 0.8, Tier::Budget)),
        "truck" => Some((30.0, Tier::Pro)),
        _ => None,
    }
}

#[test]
fn every_form_builds_stands_and_moves_in_all_three_models() {
    for f in &FORMS {
        for tier in Tier::ALL {
            let b = body_of(f.id, tier);
            let stand = b.stand(0.0);
            let k = kin::solve(&b, &stand, &KinLoads::default());
            let q = b.solve(&b.quiet(0.0), &FeedForward { enabled: false, gain: 1.0 }, 1e3);
            assert!((k.mass_kg - b.mass_kg()).abs() < 1e-3, "{} {tier:?}: kin {} vs balance {}", f.id, k.mass_kg, b.mass_kg());
            assert!((k.com[2] - q.com[2]).abs() < 0.01 && (k.com[0] - q.com[0]).abs() < 0.01, "{} {tier:?}: CoM kin {:?} vs balance {:?}", f.id, k.com, q.com);
            assert!(k.feasible && k.margin > 0.3, "{} {tier:?}: standing {:?} margin {}", f.id, k.limiting, k.margin);
            assert!((k.grf[2] - k.mass_kg * G).abs() < 1e-2);
            assert!(q.com[2] > 0.05 && q.com[2] < kin::camera(&b, &stand).pos[2].max(0.3), "{} {tier:?}: CoM {}", f.id, q.com[2]);
            let r = simulate(&b, &b.quiet(0.0), &b.quiet(0.0), &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.2, ..DynCfg::default() });
            let last = r.samples.last().unwrap();
            assert!(r.reached && r.feasible, "{} {tier:?}: {:?}", f.id, r.notes);
            assert!((last.grf[2] - b.mass_kg() * G).abs() < 0.005 * b.mass_kg() * G, "{} {tier:?}: dynamics GRF {} vs {}", f.id, last.grf[2], b.mass_kg() * G);
            assert!((last.com[2] - q.com[2]).abs() < 0.01, "{} {tier:?}: dynamics CoM {} vs balance {}", f.id, last.com[2], q.com[2]);
            let lim = b.drive_limits(&b.base.spec(), 0.0);
            assert!(lim.max_speed_mps > 0.0 && lim.max_slope_deg >= 0.0 && lim.capacity_kg >= 0.0);
            for j in &r.joints {
                assert!(!j.name.ends_with("_l") || b.arm_count() == 2 || j.name.starts_with("hip") || j.name.starts_with("knee") || j.name.starts_with("ankle"), "{} {tier:?}: {} without a left arm", f.id, j.name);
            }
            println!(
                "[MORPH] {:<15} {:<6} {:.1} kg · CoM {:.2} m · 카메라 {:.2} m · 여유 {:.2} · 경사 {:.1}° · 속도 {:.2} m/s · 적재 {:.1} kg · 관절 {}",
                f.id,
                tier.id(),
                b.mass_kg(),
                q.com[2],
                kin::camera(&b, &stand).pos[2],
                k.margin,
                lim.max_slope_deg,
                lim.max_speed_mps,
                lim.capacity_kg,
                k.joints.iter().map(|j| j.name.as_str()).collect::<Vec<_>>().join(",")
            );
        }
        if let Some((kg, tier)) = analogue_kg(f.id) {
            let m = body_of(f.id, tier).mass_kg();
            assert!(m > 0.5 * kg && m < 2.0 * kg, "{}: {m} kg vs analogue {kg} kg", f.id);
            println!("[MORPH] {:<15} 질량 {:.1} kg ({}) vs 원형 {kg:.1} kg ({:+.0}%) — {}", f.id, m, tier.id(), 100.0 * (m - kg) / kg, f.analogue);
        }
    }
}

#[test]
fn every_layer_combination_builds_in_all_three_models() {
    use noto_core::morph::{Cargo, Lower, Manip, Torso, AMR_CAMERA_H_MM, FIXED_HEAD_TILT_DEG, TRUCK_BED};
    let heads = [
        HeadMount::Neck,
        HeadMount::Fixed { tilt_deg: FIXED_HEAD_TILT_DEG },
        HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: true },
        HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: false },
        HeadMount::ChassisFront { height_mm: AMR_CAMERA_H_MM },
        HeadMount::None,
    ];
    let cargos = [Cargo::Bay, TRUCK_BED, Cargo::Tote, Cargo::None];
    let mut seen = std::collections::HashSet::new();
    let (mut asked_n, mut fitted, mut with_arm, mut grasped, mut unstowable) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for lower in Lower::ALL {
        let base = lower.base();
        let hand = hand_for(base);
        for torso in Torso::ALL {
            for head in heads {
                for manip in Manip::ALL {
                    for cargo in cargos {
                        let asked = Morphology { lower, torso, head, manip, cargo };
                        let m = asked.fit(base);
                        assert!(m.validate(base).is_ok(), "{asked:?} fits to {m:?}: {:?}", m.issues(base));
                        assert_eq!(asked.validate(base).is_ok(), asked == m, "{asked:?}: a valid form fits unchanged, an invalid one changes");
                        asked_n += 1;
                        fitted += usize::from(asked != m);
                        for lift in [false, true] {
                            if !seen.insert((m, lift)) {
                                continue;
                            }
                            let plan = BodyPlan { morph: m, lift, ..BodyPlan::default() };
                            let b = Body::new(&plan, base, FingerCount::Five);
                            let tag = format!("{} lift={lift}", plan.id());
                            assert_eq!(b.morph, m, "{tag}");
                            assert_eq!(b.plan, plan, "{tag}: the body's plan is the plan as built");
                            assert_eq!(plan.id().contains('+'), !m.humanoid_upper(), "{tag}");
                            let back: BodyPlan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
                            assert_eq!(back, plan, "{tag}");
                            let stand = b.stand(0.0);
                            let k = kin::solve(&b, &stand, &KinLoads::default());
                            let q = b.solve(&b.quiet(0.0), &FeedForward { enabled: false, gain: 1.0 }, 1e3);
                            assert!((k.mass_kg - b.mass_kg()).abs() < 1e-3, "{tag}: kin {} vs balance {}", k.mass_kg, b.mass_kg());
                            assert!((k.com[2] - q.com[2]).abs() < 0.01 && (k.com[0] - q.com[0]).abs() < 0.01, "{tag}: CoM kin {:?} vs balance {:?}", k.com, q.com);
                            assert!(k.feasible && k.margin > 0.0, "{tag}: standing {:?} {:?}", k.limiting, k.out_of_range);
                            let r = simulate(&b, &b.quiet(0.0), &b.quiet(0.0), &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.2, ..DynCfg::default() });
                            let last = r.samples.last().unwrap();
                            assert!((last.grf[2] - b.mass_kg() * G).abs() < 0.005 * b.mass_kg() * G, "{tag}: dynamics GRF {} vs {}", last.grf[2], b.mass_kg() * G);
                            assert!((last.com[2] - q.com[2]).abs() < 0.01, "{tag}: dynamics CoM {} vs balance {}", last.com[2], q.com[2]);
                            let has = |name: &str| match name {
                                "neck" => b.has_neck(),
                                "waist_yaw" => b.has_torso(),
                                "shoulder_r" | "elbow_r" => b.arm_count() >= 1,
                                "shoulder_l" | "elbow_l" => b.arm_count() == 2,
                                _ => true,
                            };
                            for name in ["neck", "waist_yaw", "shoulder_r", "elbow_r", "shoulder_l", "elbow_l"] {
                                let in_kin = k.joint(name).is_some();
                                let in_dyn = r.joints.iter().any(|j| j.name == name);
                                let in_bal = q.joint(name).is_some();
                                assert_eq!(in_dyn, has(name), "{tag}: dynamics {name}");
                                if name != "waist_yaw" {
                                    assert_eq!(in_kin, has(name), "{tag}: kin {name}");
                                    assert_eq!(in_bal, has(name), "{tag}: balance {name}");
                                }
                            }
                            assert!(r.pose_at(&b, &b.quiet(0.0), 0.1).is_some(), "{tag}");
                            let l = b.drive_limits(&base.spec(), 0.0);
                            assert!(l.max_speed_mps > 0.0 && l.capacity_kg >= 0.0 && l.capacity_kg.is_finite(), "{tag}: {l:?}");
                            assert!(b.cargo_geom().is_some() || l.capacity_kg == 0.0, "{tag}: no cargo place carries nothing");
                            let s = compute(&BodyInputs {
                                base,
                                morph: b.morph,
                                fingers: 5,
                                vision_confidence: None,
                                frame_ms: None,
                                energy_used_j: 0.0,
                                carried_kg: 0.0,
                                grip_margin: None,
                                joint_strikes: 0,
                                arm_wear: 0.0,
                                wear_max: 0.0,
                                wear_joint: None,
                                speed_factor: 0.0,
                                terrain_margin: None,
                                limits: Some(BodyLimits { battery_j: b.battery_j(), capacity_kg: l.capacity_kg, max_speed_mps: l.max_speed_mps, max_slope_deg: l.max_slope_deg, max_step_m: l.max_step_m }),
                                body_mass_kg: Some(b.mass_kg()),
                                capacity_here: None,
                                balance: None,
                            });
                            assert_eq!(s.parts.iter().any(|p| p.id == "arm_l"), b.has_manipulator(), "{tag}");
                            assert_eq!(s.parts.iter().any(|p| p.id == "head"), m.head != HeadMount::None, "{tag}");
                            for kind in ActKind::ALL {
                                let f = focus::frame(&FocusCtx { body: &b, hand: &hand, report: &k, sample: None, action: Some(kind), event: None, ground_mu: 0.6, drive: None, object: None, step_mu: None });
                                assert!(b.has_manipulator() || !f.parts.contains(&"hand"), "{tag} {kind:?}: {:?}", f.parts);
                                assert!(b.has_neck() || !f.parts.contains(&"neck"), "{tag} {kind:?}: {:?}", f.parts);
                            }
                            let p = reach::plan(&b, &ReachGoal::ground_object(0.15, 1.0, false, 0.0));
                            assert_eq!(p.code == Some("no_manipulator"), !b.has_manipulator(), "{tag}: {:?}", p.code);
                            let g = grasp::grasp(&GraspInput::front(&b, &hand, ItemMaterial::Wood, 1.0, [0.15; 3], Surface::Firm, false, 0.0, 0.0, 5));
                            let rel = grasp::release(&ReleaseInput { body: &b, material: ItemMaterial::Wood, mass_kg: 1.0, size_m: [0.15; 3], ground: Surface::Firm, surface_z_m: 0.0, cargo_kg: 1.0, aim_sigma_m: 0.0, seed: 5 });
                            let off_pelvis = b.cargo_geom().is_some_and(|c| !c.on_pelvis);
                            let stowable = !b.has_manipulator() || b.cargo_geom().is_none() || grasp::stow_pose(&b, false, 0.0).is_some();
                            assert_eq!(plan.validate(base).is_ok(), stowable, "{tag}: {:?}", plan.validate(base));
                            unstowable += usize::from(!stowable);
                            assert!(stowable || off_pelvis, "{tag}: the bay is always in reach");
                            if b.has_manipulator() {
                                with_arm += 1;
                                grasped += usize::from(g.outcome.ok);
                                assert!(g.outcome.ok || g.outcome.code.is_some_and(|c| c.starts_with("reach")), "{tag}: {:?}", g.outcome);
                                assert!(rel.outcome.ok, "{tag}: {:?}", rel.outcome);
                                if g.outcome.ok {
                                    let has = |k: &str| g.events.iter().any(|e| e.kind == k);
                                    assert_eq!(has("stowed"), b.cargo_geom().is_some() && stowable, "{tag}");
                                    assert_eq!(has("stow_unreachable"), !stowable, "{tag}");
                                }
                            } else {
                                assert_eq!((g.outcome.code, rel.outcome.code), (Some("no_manipulator"), Some("no_manipulator")), "{tag}");
                            }
                        }
                    }
                }
            }
        }
    }
    let forms = seen.iter().map(|(m, _)| *m).collect::<std::collections::HashSet<_>>().len();
    println!("[MORPH] 층 조합 {asked_n}개 (하체 4 × 몸통 3 × 카메라 6 × 조작기 4 × 적재 4) 가운데 만들 수 없는 {fitted}개는 가까운 형태로 고침 → 서로 다른 형태 {forms}개 × 리프트 유무 = {}개 모두 세 모델 질량·CoM 일치, 서 있음, 관절 목록 일치 · 조작기가 있는 {with_arm}개 중 바닥 물건(1 kg)을 집은 것 {grasped}개 (나머지는 reach 실패로 정직하게 거절) · 손이 자기 토트·짐칸에 닿지 않는 {unstowable}개는 계획 검사(BodyPlan::validate)가 거절하고 집기는 stow_unreachable로 알림", seen.len());
}

#[test]
fn the_default_forms_are_todays_robots() {
    for (form, base) in [("legs_humanoid", BaseKind::HumanFoot), ("wheel_humanoid", BaseKind::Wheel), ("track_humanoid", BaseKind::Track)] {
        let today = Body::new(&BodyPlan::default(), base, FingerCount::Five);
        let f = body_of(form, Tier::Mid);
        assert_eq!(f.mass_kg(), today.mass_kg(), "{form}");
        assert_eq!(f.com_height_m(), today.com_height_m(), "{form}");
        assert_eq!(kin::camera(&f, &KinPose::stand(0.0)), kin::head_camera(&today, &KinPose::stand(0.0)), "{form}");
        assert_eq!(f.plan.id(), "mid");
        let p = Posture::arm_lift(2.4, false, 0.0);
        assert_eq!(f.solve(&p, &FeedForward::default(), 200.0), today.solve(&p, &FeedForward::default(), 200.0), "{form}");
    }
}

#[test]
fn forms_have_only_the_joints_they_have() {
    let names = |b: &Body| kin::solve(b, &b.stand(0.0), &KinLoads::default()).joints.into_iter().map(|j| j.name).collect::<Vec<_>>();
    let noneck = body_of("wheel_noneck", Tier::Mid);
    let n = names(&noneck);
    assert!(!n.iter().any(|j| j == "neck") && n.iter().any(|j| j == "waist_pitch") && n.iter().any(|j| j == "shoulder_l"), "{n:?}");
    let truck = body_of("truck", Tier::Mid);
    let t = names(&truck);
    assert_eq!(t, vec!["drive_front", "drive_rear"], "a truck has nothing but its drive");
    let mast = body_of("mast_gripper", Tier::Mid);
    let m = names(&mast);
    assert!(m.iter().any(|j| j == "neck") && m.iter().any(|j| j == "shoulder_r") && !m.iter().any(|j| j == "shoulder_l" || j == "waist_pitch"), "{m:?}");
    let deck = body_of("deck_gripper", Tier::Mid);
    let d = names(&deck);
    assert!(!d.iter().any(|j| j == "neck" || j == "waist_pitch" || j.ends_with("_l")), "{d:?}");
    for b in [&noneck, &truck, &mast, &deck] {
        let rep = b.solve(&Posture::arm_raise(0.0), &FeedForward::default(), 1e3);
        assert_eq!(rep.joint("neck").is_some(), b.has_neck());
        assert_eq!(rep.joint("shoulder_r").is_some(), b.has_manipulator());
        assert!(rep.joint("shoulder_l").is_none() || b.arm_count() == 2);
        let r = simulate(b, &b.quiet(0.0), &Posture::arm_raise(0.0), &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.2, ..DynCfg::default() });
        assert_eq!(r.joints.iter().any(|j| j.name == "neck"), b.has_neck());
        assert_eq!(r.joints.iter().any(|j| j.name == "waist_yaw"), b.has_torso());
        assert!(r.pose_at(b, &Posture::arm_raise(0.0), 0.3).is_some());
    }
    let today = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
    assert!((today.head.mass_kg - noneck.head.mass_kg - 2.0 * today.head.neck.mass_kg).abs() < 1e-6);
    assert_eq!(noneck.head.gaze_stabilization, 0.0);
    assert_eq!(truck.belly.mass_kg, 0.0);
    assert_eq!(truck.belly.tier, Tier::Mid, "the plan's tier stays");
}

#[test]
fn the_camera_follows_the_mount() {
    let stand = KinPose::stand(0.0);
    let legs = body_of("legs_humanoid", Tier::Mid);
    assert!((kin::camera(&legs, &stand).pos[2] - 1.14).abs() < 0.01);
    let noneck = body_of("wheel_noneck", Tier::Mid);
    let level = kin::camera(&noneck, &stand);
    let asked = kin::camera(&noneck, &KinPose { neck_deg: 30.0, neck_pan_deg: 40.0, ..stand });
    assert_eq!(level, asked, "no neck: the camera does not pitch or pan with a neck angle");
    assert!((level.pitch_down_deg - 15.0).abs() < 1e-4 && level.pan_deg == 0.0);
    let pro_noneck = Body::new(&BodyPlan { morph: Morphology::named("wheel_noneck").unwrap(), ..BodyPlan::uniform(Tier::Pro) }, BaseKind::Wheel, FingerCount::Five);
    let leaned = kin::camera(&pro_noneck, &KinPose { waist_deg: 20.0, ..stand });
    assert!((leaned.pitch_down_deg - 35.0).abs() < 1e-3, "{leaned:?}");
    let mast = body_of("mast_gripper", Tier::Mid);
    let c = kin::camera(&mast, &stand);
    assert!((c.pos[2] - STRETCH_CAMERA_H_MM as f32 / 1000.0).abs() < 1e-4 && (1.061..=1.447).contains(&c.pos[2]), "{c:?}");
    let down = kin::camera(&mast, &KinPose { neck_deg: 60.0, ..stand });
    assert!((down.pitch_down_deg - 60.0).abs() < 1e-4 && down.pos[2] < c.pos[2], "{down:?}");
    assert!(kin::solve(&mast, &KinPose { neck_deg: 60.0, ..mast.stand(0.0) }, &KinLoads::default()).feasible, "a mast pan-tilt looks 60° down (Fetch: 83°)");
    assert!(!kin::solve(&noneck, &KinPose { neck_deg: 10.0, ..noneck.stand(0.0) }, &KinLoads::default()).feasible, "no neck joint to ask");
    for (form, h) in [("deck_gripper", 0.20), ("truck", 0.25)] {
        let b = body_of(form, Tier::Mid);
        let c = kin::camera(&b, &stand);
        assert!((c.pos[2] - h).abs() < 1e-4 && c.pos[2] < 0.30 && c.pos[2] > noto_core::morph::DECK_Z_M, "{form}: {c:?}");
        let up = kin::camera(&b, &KinPose { base_pitch_deg: 10.0, neck_deg: 20.0, ..stand });
        assert!((up.pitch_down_deg + 10.0).abs() < 1e-4, "{form}: the base's pitch only: {up:?}");
        println!("[MORPH] {form:<15} 카메라 {:.2} m (차체 앞 {:.2} m)", c.pos[2], c.pos[0]);
    }
    println!("[MORPH] mast_gripper    카메라 {:.2} m (Stretch RE1 1.31 m, Fetch 1.061–1.447 m) · 고정 머리 {:.2} m 아래로 {:.0}°", c.pos[2], level.pos[2], level.pitch_down_deg);
}

#[test]
fn a_neckless_camera_feels_the_whole_tilt_and_a_pan_tilt_levels_it() {
    let slope = Posture { source: Source::Slope, slope_deg: 8.0, ..Posture::quiet(0.0) };
    let hold = |b: &Body| {
        let p = Posture { arm_elev_deg: b.quiet(0.0).arm_elev_deg, elbow_deg: b.quiet(0.0).elbow_deg, ..slope };
        simulate(b, &p, &p, &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.3, ..DynCfg::default() }).samples.last().unwrap().head_pitch_deg.abs()
    };
    let (neck, noneck, mast, deck) = (hold(&body_of("wheel_humanoid", Tier::Mid)), hold(&body_of("wheel_noneck", Tier::Mid)), hold(&body_of("mast_gripper", Tier::Mid)), hold(&body_of("deck_gripper", Tier::Mid)));
    assert!((noneck - 8.0).abs() < 0.2 && (deck - 8.0).abs() < 0.2, "{noneck} {deck}");
    assert!(neck < noneck - 1.0 && mast < noneck - 1.0, "{neck} {mast} vs {noneck}");
    println!("[MORPH] 8° 경사의 카메라 기울기: 목 {neck:.2}° · 목 없음 {noneck:.2}° · 마스트 팬틸트 {mast:.2}° · 차체 카메라 {deck:.2}°");
}

#[test]
fn the_truck_refuses_a_pick_and_carries_more_on_its_bed() {
    let truck = body_of("truck", Tier::Mid);
    let hand = hand_for(BaseKind::Wheel);
    let p = reach::plan(&truck, &ReachGoal::ground_object(0.15, 1.4, false, 0.0));
    assert!(!p.ok && p.code == Some("no_manipulator"), "{p:?}");
    let g = grasp::grasp(&GraspInput::front(&truck, &hand, ItemMaterial::Wood, 1.4, [0.15; 3], Surface::Firm, false, 0.0, 0.0, 1));
    assert_eq!(g.outcome.code, Some("no_manipulator"));
    let r = grasp::release(&ReleaseInput { body: &truck, material: ItemMaterial::Wood, mass_kg: 1.4, size_m: [0.15; 3], ground: Surface::Firm, surface_z_m: 0.0, cargo_kg: 1.4, aim_sigma_m: 0.0, seed: 1 });
    assert_eq!(r.outcome.code, Some("no_manipulator"));
    let (lift, two) = truck.lift(1.4, 0.0, &FeedForward::default(), 200.0);
    assert!(!lift.feasible && !two && lift.limiting.as_deref().is_some_and(|l| l.contains("no_manipulator")));
    for tier in [Tier::Mid, Tier::Pro] {
        let t = body_of("truck", tier);
        let h = Body::new(&BodyPlan::uniform(tier), BaseKind::Wheel, FingerCount::Five);
        let (ct, ch) = (t.drive_limits(&BaseKind::Wheel.spec(), 0.0).capacity_kg, h.drive_limits(&BaseKind::Wheel.spec(), 0.0).capacity_kg);
        assert!(ct > ch && ct <= 100.0 + 1e-3, "{tier:?}: truck {ct} vs humanoid {ch}");
        let k = kin::solve(&t, &t.stand(ct), &KinLoads::default());
        assert!(k.feasible && k.com[2] < h.com_height_m(), "{tier:?}: {:?} CoM {}", k.limiting, k.com[2]);
        println!("[MORPH] truck {:<6} 짐칸 적재 {ct:.1} kg (휴머노이드 배 적재 {ch:.1} kg) · 만재 CoM {:.2} m · 여유 {:.2}", tier.id(), k.com[2], k.margin);
    }
    let f = focus::frame(&FocusCtx { body: &truck, hand: &hand, report: &kin::solve(&truck, &truck.stand(0.0), &KinLoads::default()), sample: None, action: Some(ActKind::Grasp), event: None, ground_mu: 0.6, drive: None, object: None, step_mu: None });
    assert!(f.hand.fingers.is_empty() && f.neck.is_none() && f.waist.is_none() && !f.parts.contains(&"hand"), "{:?}", f.parts);
    serde_json::to_string(&f).unwrap();
}

#[test]
fn a_deck_arm_reaches_the_floor_and_stows_in_its_tote() {
    let deck = body_of("deck_gripper", Tier::Mid);
    let hand = hand_for(BaseKind::Wheel);
    let goal = ReachGoal::ground_object(0.15, 1.4, false, 0.0);
    let p = reach::plan(&deck, &goal);
    assert!(p.ok && p.strategy == "stand", "{p:?}");
    assert!((p.empty.hands[0][2] - goal.palm_z).abs() < 3e-3 && p.empty.hands[0][0] > reach::min_object_x(&deck, goal.half_len_m) - 1e-3, "{:?}", p.empty.hands[0]);
    let g = grasp::grasp(&GraspInput::front(&deck, &hand, ItemMaterial::Wood, 1.4, [0.15; 3], Surface::Firm, false, 0.0, 0.0, 3));
    assert!(g.outcome.ok, "{:?} {:?}", g.outcome, g.notes);
    let stowed = g.events.iter().find(|e| e.kind == "stowed").expect("stowed");
    assert!(stowed.label_ko.contains("토트"), "{}", stowed.label_ko);
    let drop = g.metric("stow_drop_m").unwrap();
    assert!(drop > 0.0 && drop < 0.3, "{drop}");
    let at = g.samples.last().unwrap().object.unwrap();
    assert!((at[0] - noto_core::morph::TOTE_X_M).abs() < 1e-4 && at[2] < 0.3, "{at:?}");
    let two = grasp::grasp(&GraspInput::front(&deck, &hand, ItemMaterial::Wood, 1.4, [0.15; 3], Surface::Firm, true, 0.0, 0.0, 3));
    assert_eq!(two.outcome.code, Some("one_arm"));
    assert!(!deck.lift(4.0, 0.0, &FeedForward::default(), 1e3).1, "one arm never lifts with two");
    let rel = grasp::release(&ReleaseInput { body: &deck, material: ItemMaterial::Wood, mass_kg: 1.4, size_m: [0.15; 3], ground: Surface::Firm, surface_z_m: 0.0, cargo_kg: 0.0, aim_sigma_m: 0.0, seed: 1 });
    assert!(rel.outcome.ok && rel.metric("drop_height_m").unwrap() < 0.05, "{:?} {:?}", rel.outcome, rel.metric("drop_height_m"));
    println!("[MORPH] deck_gripper    바닥 집기 {} · {:.1} s · 토트 낙하 {drop:.2} m · 내려놓기 낙하 {:.3} m", p.strategy_ko, g.duration_s, rel.metric("drop_height_m").unwrap());
}

#[test]
fn a_deck_arm_tips_where_the_moment_balance_says() {
    for tier in [Tier::Mid, Tier::Pro] {
        let deck = body_of("deck_gripper", tier);
        let mut pose = deck.stand(0.0);
        pose.shoulder_deg = [90.0, pose.shoulder_deg[1]];
        pose.elbow_deg = [0.0, pose.elbow_deg[1]];
        let empty = kin::solve(&deck, &pose, &KinLoads::default());
        let front = empty.support[1];
        let d_p = empty.hands[0][0] - front;
        let m_tip = empty.mass_kg * (front - empty.com[0]) / d_p;
        let at = |kg: f32| kin::solve(&deck, &pose, &KinLoads::holding(kg, false, 0.0)).margin;
        assert!(at(0.97 * m_tip) > 0.0 && at(1.03 * m_tip) < 0.0, "{tier:?}: tip at {m_tip} kg: {} / {}", at(0.97 * m_tip), at(1.03 * m_tip));
        let shoulder = kin::solve(&deck, &pose, &KinLoads::holding(m_tip, false, 0.0)).joint("shoulder_r").unwrap().ratio;
        println!("[MORPH] deck_gripper {:<6} 팔을 수평으로 뻗고 앞바퀴 위로 넘어지는 하중 {m_tip:.1} kg (손 {d_p:.2} m 앞) · 그때 어깨 {:.0}%", tier.id(), shoulder * 100.0);
    }
}

#[test]
fn a_mast_robot_rides_its_lift_to_the_floor_and_to_a_table_top() {
    let mast = body_of("mast_gripper", Tier::Mid);
    let lift = mast.lift.as_ref().expect("a mast with an arm has its lift");
    let (top, stroke) = mast.morph.mast_lift_m().unwrap();
    assert!((top - noto_core::morph::MAST_LIFT_TOP_M).abs() < 1e-6 && (lift.stroke_m - stroke).abs() < 1e-6 && stroke > 0.8, "{top} {stroke}");
    assert!((mast.shoulder_at(0, stroke)[2] - (noto_core::morph::DECK_Z_M + noto_core::morph::DECK_ARM_BASE_H_M)).abs() < 1e-5);
    let floor = reach::plan(&mast, &ReachGoal::ground_object(0.10, 1.0, false, 0.0));
    assert!(floor.ok && floor.strategy == "lower" && floor.pose.lift_m > 0.4, "{floor:?}");
    assert!(floor.pose.neck_deg > 30.0 && floor.pose.neck_deg <= mast.head_range_deg().0 + 1e-3, "{}", floor.pose.neck_deg);
    let neck = floor.loaded.joint("neck").unwrap();
    assert!(neck.in_range && neck.ratio < 0.2, "{neck:?}");
    let carriage = floor.loaded.joint("torso_lift").expect("the mast's lift is a joint");
    assert!(carriage.in_range && carriage.ratio < 0.2, "{carriage:?}");
    assert!(floor.descend_s >= noto_core::act::MIN_JERK_PEAK * floor.pose.lift_m / lift.speed_mps - 1e-3, "{}", floor.descend_s);
    let table = ReachGoal { palm_z: 0.75 + 0.15 + reach::PALM_CLEARANCE_M, ..ReachGoal::ground_object(0.15, 1.0, false, 0.0) };
    let high = reach::plan(&mast, &table);
    assert!(high.ok && (high.empty.hands[0][2] - table.palm_z).abs() < 3e-3, "{high:?}");
    let deck = reach::plan(&body_of("deck_gripper", Tier::Mid), &table);
    assert!(!deck.ok && deck.reason_ko.as_deref().is_some_and(|r| r.contains("갑판")), "{deck:?}");
    println!(
        "[MORPH] mast_gripper    리프트 {:.2} m 행정 · 바닥 물건: 리프트 {:.2} m 내림, 팬틸트 {:.0}° (한계 {:.0}°), {:.1} s · 0.75 m 테이블: 리프트 {:.2} m · 갑판 팔은 불가 ({})",
        stroke,
        floor.pose.lift_m,
        floor.pose.neck_deg,
        mast.head_range_deg().0,
        floor.descend_s,
        high.pose.lift_m,
        deck.reason_ko.unwrap_or_default()
    );
}

#[test]
fn a_mast_robot_stows_in_its_tote_with_the_carriage_lowered() {
    let mast = body_of("mast_gripper", Tier::Mid);
    let hand = hand_for(BaseKind::Wheel);
    let g = grasp::grasp(&GraspInput::front(&mast, &hand, ItemMaterial::Wood, 1.0, [0.10; 3], Surface::Firm, false, 0.0, 0.0, 3));
    assert!(g.outcome.ok, "{:?} {:?}", g.outcome, g.notes);
    let stowed = g.events.iter().find(|e| e.kind == "stowed").expect("stowed");
    assert!(stowed.label_ko.contains("토트"), "{}", stowed.label_ko);
    let stow = g.samples.iter().filter(|s| s.phase == "stow").last().unwrap();
    assert!(stow.pose.lift_m > 0.05, "{:?}", stow.pose);
    let highest = g.samples.iter().filter(|s| s.phase == "lift" || s.phase == "stow").map(|s| s.pose.lift_m).fold(f32::INFINITY, f32::min);
    assert!(highest >= stow.pose.lift_m - 1e-3, "carriage went up to {highest} m below the top, stows at {}", stow.pose.lift_m);
    let drop = g.metric("stow_drop_m").unwrap();
    assert!(drop > 0.0 && drop <= noto_core::morph::RIM_RELEASE_M + 1e-4, "{drop}");
    let at = g.samples.last().unwrap().object.unwrap();
    assert!((at[0] - noto_core::morph::TOTE_X_M).abs() < 1e-4, "{at:?}");
    let rel = grasp::release(&ReleaseInput { body: &mast, material: ItemMaterial::Wood, mass_kg: 1.0, size_m: [0.10; 3], ground: Surface::Firm, surface_z_m: 0.0, cargo_kg: 0.0, aim_sigma_m: 0.0, seed: 1 });
    assert!(rel.outcome.ok && rel.metric("drop_height_m").unwrap() < 0.05, "{:?} {:?}", rel.outcome, rel.metric("drop_height_m"));
    println!("[MORPH] mast_gripper    바닥 집어 토트에: {:.1} s · 리프트 {:.2} m · 토트 낙하 {drop:.2} m · 내려놓기 {:.1} s", g.duration_s, stow.pose.lift_m, rel.duration_s);
}

#[test]
fn body_status_draws_the_form() {
    let inputs = |b: &Body| BodyInputs {
        base: b.base,
        morph: b.morph,
        fingers: 5,
        vision_confidence: Some(0.8),
        frame_ms: None,
        energy_used_j: 0.0,
        carried_kg: 1.0,
        grip_margin: Some(1.5),
        joint_strikes: 0,
        arm_wear: 0.0,
        wear_max: 0.0,
        wear_joint: None,
        speed_factor: 0.5,
        terrain_margin: Some(0.8),
        limits: None,
        body_mass_kg: Some(b.mass_kg()),
        capacity_here: None,
        balance: None,
    };
    for f in &FORMS {
        let b = body_of(f.id, Tier::Mid);
        let s = compute(&inputs(&b));
        let ids: Vec<&str> = s.parts.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(s.form, f.id);
        assert_eq!(s.morph, f.morph);
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"form_ko\"") && json.contains("\"morph\""));
        match f.id {
            "truck" => assert_eq!(ids, ["head", "thorax", "bed", "leg_l", "leg_r"]),
            "deck_gripper" | "mast_gripper" => assert_eq!(ids, ["head", "thorax", "tote", "arm_l", "arm_r", "leg_l", "leg_r"]),
            _ => assert_eq!(ids, PART_IDS),
        }
        let leg = s.parts.iter().find(|p| p.id == "leg_l").unwrap();
        assert!(leg.label_ko.starts_with(if b.legged() { "발" } else if b.base == BaseKind::Track { "트랙" } else { "바퀴" }), "{}: {}", f.id, leg.label_ko);
        println!("[MORPH] {:<15} 상태 영역 {} · {}", f.id, s.parts.iter().map(|p| p.label_ko.as_str()).collect::<Vec<_>>().join(" / "), s.form_ko);
    }
    let mast = compute(&inputs(&body_of("mast_gripper", Tier::Mid)));
    assert!(mast.parts[0].label_ko.contains("마스트 카메라") && mast.parts[3].label_ko.contains("집게"));
    assert!(mast.effects.iter().any(|e| e.id == "load"));
    let none = Body::new(&BodyPlan { morph: Morphology { head: HeadMount::None, ..Morphology::named("truck").unwrap() }, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
    let s = compute(&inputs(&none));
    assert!(s.parts.iter().all(|p| p.id != "head") && s.form == "custom");
}

#[test]
fn drive_and_push_draw_each_form_in_its_own_rest_pose() {
    use noto_core::act::drive;
    use noto_core::act::push::{self, PushInput, PushLeg};
    let deck = body_of("deck_gripper", Tier::Mid);
    let hanging = kin::solve(&deck, &KinPose::stand(0.0), &KinLoads::default());
    assert!(hanging.hands[0][2] < 0.0, "{:?}", hanging.hands[0]);
    for form in ["deck_gripper", "mast_gripper", "truck", "wheel_noneck", "wheel_humanoid", "track_humanoid"] {
        let b = body_of(form, Tier::Mid);
        let d = drive::drive(&b, Surface::Firm, 5.0, 1.0, 1.0, 2.0, 0.0);
        let leg = PushLeg { turn_deg: 0.0, radius_m: 0.0, len_m: 1.0, ground: Surface::Gravel, slope_deg: 0.0, blocked: false };
        let p = push::push(&PushInput { body: &b, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], cargo_kg: 1.0, legs: vec![leg], aim_sigma_m: 0.0, seed: 1 });
        assert!(d.outcome.ok && p.outcome.ok && !d.samples.is_empty() && !p.samples.is_empty(), "{form}: {:?} {:?}", d.outcome, p.outcome);
        let mut low = f32::INFINITY;
        for s in d.samples.iter().chain(&p.samples) {
            assert_eq!((s.pose.shoulder_deg, s.pose.elbow_deg), b.rest_arms_deg(), "{form} {}", s.phase);
            let r = kin::solve(&b, &s.pose, &s.loads);
            assert!(r.out_of_range.is_empty(), "{form} {}: {:?}", s.phase, r.out_of_range);
            for k in 0..b.arm_count() {
                low = low.min(r.hands[k][2]);
            }
        }
        assert!(b.arm_count() == 0 || low > 0.1, "{form}: a hand at {low} m");
        if b.morph.humanoid_upper() {
            assert_eq!(d.samples[0].pose, KinPose { base_pitch_deg: 5.0, ..KinPose::stand(1.0) }, "{form}: today's pose, bit for bit");
        }
        let st = drive::state(&b, Surface::Firm.props(), 0.0, 0.0, 0.0, 0.0);
        let r = kin::solve_fast(&b, &b.stand(0.0), &KinLoads::default());
        if let (Some(f), Some(rr)) = (r.contact("axle_front"), r.contact("axle_rear")) {
            assert!((st.front_n - f.normal_n).abs() < 1e-3 && (st.rear_n - rr.normal_n).abs() < 1e-3, "{form}: {st:?}");
        }
        if form == "deck_gripper" {
            println!("[MORPH] deck_gripper    주행·밀기 표본의 손 최저 {low:.2} m (사람형 선 자세였다면 {:.2} m, 땅 아래) · 앞축 {:.0} N / 뒷축 {:.0} N", hanging.hands[0][2], st.front_n, st.rear_n);
        }
    }
}

#[test]
fn a_gripper_squeezes_with_its_jaws_and_is_drawn_with_two() {
    let hand = hand_for(BaseKind::Wheel);
    let size = [0.06; 3];
    let two = {
        let mut cfg = NotoConfig::preset(FingerCount::Two);
        cfg.base = BaseKind::Wheel;
        cfg.build().unwrap()
    };
    let budget = body_of("deck_gripper", Tier::Budget);
    let pro = body_of("deck_gripper", Tier::Pro);
    let mid = body_of("deck_gripper", Tier::Mid);
    let cap = |b: &Body| grasp::capacity_for(b, &hand, size, 0.5, 0.6);
    assert_eq!((cap(&budget).squeeze_n, cap(&budget).contacts), (42.0, 2));
    assert_eq!((cap(&pro).squeeze_n, cap(&pro).contacts), (470.0, 2));
    assert_eq!(cap(&mid), grasp::capacity(&two, size, 0.5, 0.6));
    let today = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
    assert_eq!(grasp::capacity_for(&today, &hand, size, 0.5, 0.6), grasp::capacity(&hand, size, 0.5, 0.6));
    let g = grasp::grasp(&GraspInput::front(&pro, &hand, ItemMaterial::Wood, 1.0, [0.08; 3], Surface::Firm, false, 0.0, 0.0, 3));
    assert!(g.outcome.ok && g.metric("squeeze_cap_n") == Some(470.0), "{:?} {:?}", g.outcome, g.metric("squeeze_cap_n"));
    for b in [&budget, &mid, &pro, &body_of("mast_gripper", Tier::Mid)] {
        let k = kin::solve(b, &b.stand(0.0), &KinLoads::default());
        assert_eq!(k.boxes.iter().filter(|x| x.name == "finger_r").count(), 4, "two jaws of two links");
        let f = focus::frame(&FocusCtx { body: b, hand: &hand, report: &k, sample: None, action: Some(ActKind::Grasp), event: None, ground_mu: 0.6, drive: None, object: None, step_mu: None });
        assert_eq!(f.hand.fingers.len(), 2);
    }
    println!("[MORPH] 집게 조임 (양쪽 턱 2·F): SO-101 {:.0} N · XM430 2지 {:.0} N (손 모델) · Robotiq 2F-85 {:.0} N — 설정 손(5지) {:.0} N", cap(&budget).squeeze_n, cap(&mid).squeeze_n, cap(&pro).squeeze_n, grasp::capacity(&hand, size, 0.5, 0.6).squeeze_n);
}

#[test]
fn a_load_on_an_arm_the_form_does_not_have_is_never_dropped() {
    let deck = body_of("deck_gripper", Tier::Mid);
    let truck = body_of("truck", Tier::Mid);
    let both = kin::solve(&deck, &deck.stand(0.0), &KinLoads::holding(1.0, true, 0.0));
    assert!(!both.feasible && both.out_of_range.iter().any(|r| r.contains("왼팔 없음")), "{:?}", both.out_of_range);
    let one = kin::solve(&deck, &deck.stand(0.0), &KinLoads::holding(1.0, false, 0.0));
    assert!(one.feasible && one.out_of_range.is_empty(), "{:?}", one.out_of_range);
    let t = kin::solve(&truck, &truck.stand(0.0), &KinLoads::holding(1.0, false, 0.0));
    assert!(!t.feasible && t.out_of_range.iter().any(|r| r.contains("오른팔 없음")), "{:?}", t.out_of_range);
    let left = deck.solve(&Posture { hand_load_kg: [0.0, 1.0], ..deck.quiet(0.0) }, &FeedForward::default(), 1e3);
    assert!(!left.feasible && left.limiting.as_deref().is_some_and(|l| l.contains("왼팔 없음")), "{:?}", left.limiting);
    assert!(deck.solve(&Posture { hand_load_kg: [1.0, 0.0], ..deck.quiet(0.0) }, &FeedForward::default(), 1e3).feasible);
    let today = Body::new(&BodyPlan::default(), BaseKind::Wheel, FingerCount::Five);
    assert!(kin::solve(&today, &KinPose::stand(0.0), &KinLoads::holding(1.0, true, 0.0)).out_of_range.is_empty());
}

#[test]
fn a_mast_cameras_mass_sits_where_all_three_models_swing_it() {
    for pan_tilt in [true, false] {
        let m = Morphology { head: HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt }, ..Morphology::named("mast_gripper").unwrap() };
        let mut b = Body::new(&BodyPlan { morph: m, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        b.head.com_up_m = 0.15;
        let com = b.head_com().unwrap();
        assert!((com[2] - (STRETCH_CAMERA_H_MM as f32 / 1000.0 - kin::CAMERA_UP_M + 0.15)).abs() < 1e-6, "{com:?}");
        let k = kin::solve(&b, &b.stand(0.0), &KinLoads::default());
        let q = b.solve(&b.quiet(0.0), &FeedForward { enabled: false, gain: 1.0 }, 1e3);
        assert!((k.com[2] - q.com[2]).abs() < 1e-4 && (k.com[0] - q.com[0]).abs() < 1e-4, "pan_tilt {pan_tilt}: kin {:?} vs balance {:?}", k.com, q.com);
        let r = simulate(&b, &b.quiet(0.0), &b.quiet(0.0), &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.2, ..DynCfg::default() });
        let last = r.samples.last().unwrap().com[2];
        assert!((last - q.com[2]).abs() < 1e-3, "pan_tilt {pan_tilt}: dynamics {last} vs balance {}", q.com[2]);
    }
}

#[test]
fn kin_and_dynamics_draw_the_same_deck_and_nothing_on_it_intersects() {
    type Aabb = ([f32; 3], [f32; 3]);
    let aabb = |c: [f32; 3], h: [f32; 3]| -> Aabb { ([c[0] - h[0], c[1] - h[1], c[2] - h[2]], [c[0] + h[0], c[1] + h[1], c[2] + h[2]]) };
    let overlap = |a: Aabb, b: Aabb| (0..3).all(|i| a.0[i] < b.1[i] - 1e-4 && b.0[i] < a.1[i] - 1e-4);
    for f in &FORMS {
        let b = body_of(f.id, Tier::Mid);
        let k = kin::solve(&b, &b.stand(0.0), &KinLoads::default());
        let r = simulate(&b, &b.quiet(0.0), &b.quiet(0.0), &FeedForward::default(), 1e3, &DynCfg { settle_s: 0.2, ..DynCfg::default() });
        let drawn = r.pose_at(&b, &b.quiet(0.0), 0.1).unwrap();
        let deck = b.deck_boxes();
        for d in &deck {
            let kb = k.boxes.iter().find(|x| x.name == d.name).unwrap_or_else(|| panic!("{}: kin has no {}", f.id, d.name));
            let db = drawn.boxes.iter().find(|x| x.name == d.name).unwrap_or_else(|| panic!("{}: dynamics has no {}", f.id, d.name));
            assert_eq!((kb.half, db.half), (d.half, d.half), "{} {}", f.id, d.name);
            for i in 0..3 {
                assert!((kb.center[i] - d.center[i]).abs() < 1e-5 && (db.center[i] - d.center[i]).abs() < 0.01, "{} {}: {:?} {:?} {:?}", f.id, d.name, kb.center, db.center, d.center);
            }
        }
        assert!(!k.boxes.iter().any(|x| x.name == "electronics"), "the electronics ride inside the chassis");
        let mut parts: Vec<(&str, Aabb)> = deck.iter().filter(|d| d.name != "camera").map(|d| (d.name, aabb(d.center, d.half))).collect();
        for x in k.boxes.iter().filter(|x| matches!(x.name, "pole" | "column" | "carriage")) {
            parts.push((x.name, aabb(x.center, x.half)));
        }
        if let Some((_, stroke)) = b.morph.mast_lift_m() {
            let down = kin::solve(&b, &KinPose { lift_m: stroke, ..b.stand(0.0) }, &KinLoads::default());
            let c = down.boxes.iter().find(|x| x.name == "carriage").unwrap();
            parts.push(("carriage_down", aabb(c.center, c.half)));
        }
        for (i, a) in parts.iter().enumerate() {
            for bb in &parts[i + 1..] {
                let rides = (a.0 == "pole" && bb.0.starts_with("carriage")) || (bb.0 == "pole" && a.0.starts_with("carriage"));
                if rides {
                    continue;
                }
                assert!(!overlap(a.1, bb.1), "{}: {} intersects {} ({:?} / {:?})", f.id, a.0, bb.0, a.1, bb.1);
            }
        }
    }
}

#[test]
fn the_body_keeps_the_plan_it_built() {
    use noto_core::morph::{Cargo, Torso};
    let asked = BodyPlan { morph: Morphology { torso: Torso::None, ..Morphology::default() }, ..BodyPlan::default() };
    assert!(asked.validate(BaseKind::Wheel).is_err());
    let b = Body::new(&asked, BaseKind::Wheel, FingerCount::Five);
    assert_eq!(b.plan.morph, Morphology { lower: asked.morph.lower, ..b.morph });
    assert_ne!(b.plan.id(), asked.id());
    assert!(b.plan.id().contains("mast1310pt") && b.plan.id().contains("c:tote"), "{} (asked {})", b.plan.id(), asked.id());
    assert!(b.plan.validate(BaseKind::Wheel).is_ok(), "{:?}", b.plan.validate(BaseKind::Wheel));
    for base in BaseKind::ALL {
        assert_eq!(Body::new(&BodyPlan::default(), base, FingerCount::Five).plan, BodyPlan::default(), "{base:?}");
    }
    let tote = BodyPlan { morph: Morphology { cargo: Cargo::Tote, ..Morphology::default() }, ..BodyPlan::default() };
    let why = tote.validate(BaseKind::Wheel).unwrap_err();
    assert!(why.contains("닿지 않음"), "{why}");
    let lifted = BodyPlan { lift: true, ..tote };
    println!("[MORPH] 휴머노이드 + 토트: 리프트 없음 → {why} · 리프트 있음 → {:?}", lifted.validate(BaseKind::Wheel));
}

#[test]
fn the_engine_never_names_a_morphology_benchmark_source() {
    let files = [
        ("morph.rs", include_str!("../src/morph.rs")),
        ("parts.rs", include_str!("../src/parts.rs")),
        ("balance.rs", include_str!("../src/balance.rs")),
        ("kin.rs", include_str!("../src/kin.rs")),
        ("body.rs", include_str!("../src/body.rs")),
        ("dynamics.rs", include_str!("../src/dynamics.rs")),
        ("act/reach.rs", include_str!("../src/act/reach.rs")),
        ("act/grasp.rs", include_str!("../src/act/grasp.rs")),
        ("act/focus.rs", include_str!("../src/act/focus.rs")),
        ("act/drive.rs", include_str!("../src/act/drive.rs")),
        ("act/push.rs", include_str!("../src/act/push.rs")),
    ];
    let sources = ["Wevolver", "Hardware Guide", "hardware_guide", "Ackerman", "IEEE Spectrum", "Konica", "New Equipment", "GIGAZINE", "iFixit", "Techzine"];
    for (name, src) in files {
        for needle in sources {
            assert!(!src.contains(needle), "{name} names the morphology benchmark source `{needle}`");
        }
    }
}
