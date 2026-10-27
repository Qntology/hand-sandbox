//! Robot status shown like a body-part health display (head, thorax, stomach,
//! arms, legs).
//!
//! Every value is computed from simulation variables. The two "arms" are two
//! metrics of the working (right) hand or gripper (grip and joints) and the
//! two "legs" are two metrics of the base (drive and terrain margin). They are
//! not anatomical left and right.
//!
//! The zones follow the form (`morph`): the drive and terrain zones are named
//! after the lower body (발, 바퀴, 메카넘 바퀴, 트랙), the vision zone after the
//! camera (a head, a mast camera, a chassis camera, none without one), the
//! cargo zone after the cargo (the bay `stomach`, a bed `bed`, a tote `tote`,
//! none without one), and the grip and arm zones exist only with a
//! manipulator. Today's humanoid keeps its seven zones in their order.
//!
//! Modelling assumptions (not measured values) are named constants here and
//! listed in `data/README.md`:
//! * [`BATTERY_RUNTIME_S`]: battery energy is cruise power times this runtime.
//! * [`BaseKind::carry_capacity_kg`]: cargo limit per base.
//!
//! Joint wear is the reducers' Miner damage (`wear`), joint heat the
//! world's thermal state; both come in through [`BodyInputs`].

use serde::{Deserialize, Serialize};

use crate::balance::{Coupling, JointLoad};
use crate::base::{BaseKind, BaseSpec};
use crate::morph::{Cargo, HeadMount, Lower, Manip, Morphology, Torso};
use crate::surface::{slip_probability, Contact};

/// Battery energy equals cruise power times this many seconds (assumption).
pub const BATTERY_RUNTIME_S: f64 = 1800.0;

impl BaseKind {
    /// Cargo limit in kilograms (assumption; see `data/README.md`).
    pub fn carry_capacity_kg(self) -> f32 {
        match self {
            BaseKind::Wheel => 20.0,
            BaseKind::Mecanum => 15.0,
            BaseKind::Track => 30.0,
            BaseKind::HumanFoot => 25.0,
        }
    }
}

/// Battery capacity in joules for a base: cruise power times [`BATTERY_RUNTIME_S`].
pub fn battery_capacity_j(spec: &BaseSpec) -> f64 {
    spec.power_w as f64 * BATTERY_RUNTIME_S
}

/// Ordered from best to worst, so `max` is the worse of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BodyLevel {
    Ok,
    Warn,
    Bad,
    Crit,
}

impl BodyLevel {
    /// Maps a 0..1 health ratio to a level: >= 0.6 ok, >= 0.3 warn, >= 0.1 bad, else crit.
    pub fn from_ratio(ratio: f32) -> BodyLevel {
        if ratio >= 0.6 {
            BodyLevel::Ok
        } else if ratio >= 0.3 {
            BodyLevel::Warn
        } else if ratio >= 0.1 {
            BodyLevel::Bad
        } else {
            BodyLevel::Crit
        }
    }
}

/// Identifiers of the seven body zones, in display order.
pub const PART_IDS: [&str; 7] = [
    "head", "thorax", "stomach", "arm_l", "arm_r", "leg_l", "leg_r",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyPart {
    pub id: String,
    pub label_ko: String,
    pub value: f32,
    pub max: f32,
    /// Health ratio, 0 (worst) to 1 (best).
    pub ratio: f32,
    pub level: BodyLevel,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyEffect {
    pub id: String,
    pub label_ko: String,
    pub level: BodyLevel,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandSummary {
    pub fingers: u8,
    /// Friction capacity over required force (>= 1 means the last grasp was stable).
    pub grip_margin: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootSummary {
    pub base: BaseKind,
    pub speed_factor: f32,
    pub terrain_margin: Option<f32>,
}

/// Balance and joint load of the current posture (see `balance`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BalanceSummary {
    /// Body preset id.
    pub plan: String,
    pub source: String,
    pub source_ko: String,
    /// Terrain and phase bucket, such as `flat|mine`.
    pub state: String,
    pub margin: f32,
    pub margin_noff: f32,
    pub trunk_lean_deg: f32,
    pub head_tilt_deg: f32,
    pub worst_joint: Option<String>,
    /// Largest torque over peak rating.
    pub worst_ratio: f32,
    pub effort_sum: f32,
    pub mass_kg: f32,
    pub com_height_m: f32,
    /// Hottest joint's heat state (above 1 the joint is derated).
    pub heat_max: f32,
    /// The joint closest to a limit and its index: the larger of torque over
    /// peak and RMS (`√heat`) over the continuous rating.
    #[serde(default)]
    pub load_joint: Option<String>,
    #[serde(default)]
    pub load_index: f32,
    #[serde(default)]
    pub load_rms: f32,
    pub joints: Vec<JointLoad>,
    /// The N:N cells of this state.
    pub couplings: Vec<Coupling>,
    /// Ground under the robot.
    #[serde(default)]
    pub ground: String,
    /// Friction the contacts need on that ground against what it gives.
    #[serde(default)]
    pub contact: Option<Contact>,
}

/// Limits that the part presets set; without them the base's defaults apply.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyLimits {
    pub battery_j: f64,
    pub capacity_kg: f32,
    pub max_speed_mps: f32,
    pub max_slope_deg: f32,
    pub max_step_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyStatus {
    pub parts: Vec<BodyPart>,
    pub effects: Vec<BodyEffect>,
    pub hand: HandSummary,
    pub foot: FootSummary,
    #[serde(default)]
    pub balance: Option<BalanceSummary>,
    /// The robot's form, for the silhouette (`morph`).
    #[serde(default)]
    pub morph: Morphology,
    /// The named form's id (`truck`, ...) or `custom`, and its Korean name.
    #[serde(default)]
    pub form: String,
    #[serde(default)]
    pub form_ko: String,
}

/// Simulation variables the status is computed from.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyInputs {
    pub base: BaseKind,
    /// The robot's form (its lower body follows `base`).
    pub morph: Morphology,
    pub fingers: u8,
    /// Recent perception confidence 0..1. `None` when nothing was perceived yet.
    pub vision_confidence: Option<f32>,
    /// Latest frame pipeline time in milliseconds, for the detail text.
    pub frame_ms: Option<f32>,
    pub energy_used_j: f64,
    pub carried_kg: f32,
    /// Friction capacity over required force of the latest grasp, if any.
    pub grip_margin: Option<f32>,
    pub joint_strikes: u32,
    /// Life used by the working arm's joints and by the most worn joint (Miner damage, 1 = worn out).
    pub arm_wear: f32,
    pub wear_max: f32,
    pub wear_joint: Option<String>,
    /// Current speed over the base's top speed, 0..1.
    pub speed_factor: f32,
    /// Smallest remaining margin to the base's slope or step limit on the
    /// current route segment, 0..1. `None` without a route.
    pub terrain_margin: Option<f32>,
    /// Limits from the part presets (battery, cargo, speed, slope, step).
    pub limits: Option<BodyLimits>,
    /// Robot mass without cargo, for the load over body mass.
    pub body_mass_kg: Option<f32>,
    /// Cargo the drive still pushes on the ground and slope here, when that
    /// is less than the continuous limit (soft ground, a slope), and the ground.
    pub capacity_here: Option<(f32, String)>,
    pub balance: Option<BalanceSummary>,
}

fn part(id: &str, label_ko: &str, value: f32, max: f32, ratio: f32, detail: String) -> BodyPart {
    let ratio = if ratio.is_finite() { ratio.clamp(0.0, 1.0) } else { 0.0 };
    BodyPart {
        id: id.into(),
        label_ko: label_ko.into(),
        value,
        max,
        ratio,
        level: BodyLevel::from_ratio(ratio),
        detail,
    }
}

/// Level of a friction margin (available − required): the 1%, 5% and 50%
/// points of the slip fit (`surface::slip_probability`). Applied to wheels and
/// tracks too, as a convention.
pub fn slip_level(margin: f32) -> BodyLevel {
    match slip_probability(margin) {
        p if p < 0.01 => BodyLevel::Ok,
        p if p < 0.05 => BodyLevel::Warn,
        p if p < 0.5 => BodyLevel::Bad,
        _ => BodyLevel::Crit,
    }
}

/// The zone names the form gives: vision, battery, cargo (id and name), grip,
/// arm joints and the lower body's word (`None` where the form has no such zone).
struct ZoneNames {
    head: Option<&'static str>,
    thorax: &'static str,
    cargo: Option<(&'static str, &'static str)>,
    arms: Option<(&'static str, &'static str)>,
    lower: &'static str,
}

fn zone_names(m: &Morphology) -> ZoneNames {
    ZoneNames {
        head: match m.head {
            HeadMount::Neck | HeadMount::Fixed { .. } => Some("머리 · 비전"),
            HeadMount::MastCamera { .. } => Some("마스트 카메라 · 비전"),
            HeadMount::ChassisFront { .. } => Some("전방 카메라 · 비전"),
            HeadMount::None => None,
        },
        thorax: if m.torso == Torso::Humanoid { "가슴 · 배터리" } else { "본체 · 배터리" },
        cargo: match m.cargo {
            Cargo::Bay => Some(("stomach", "배 · 적재")),
            Cargo::Bed { .. } => Some(("bed", "적재함 · 짐칸")),
            Cargo::Tote => Some(("tote", "토트 · 적재")),
            Cargo::None => None,
        },
        arms: match m.manip {
            Manip::Arms2 | Manip::Arm1 => Some(("손 · 파지", "손 · 관절")),
            Manip::Gripper1 => Some(("집게 · 파지", "팔 · 관절")),
            Manip::None => None,
        },
        lower: match m.lower {
            Lower::Legs => "발",
            Lower::Wheels => "바퀴",
            Lower::Mecanum => "메카넘 바퀴",
            Lower::Track => "트랙",
        },
    }
}

/// Computes the zones of the form (today's humanoid: seven) and the active effects from `inp`.
pub fn compute(inp: &BodyInputs) -> BodyStatus {
    let morph = inp.morph.fit(inp.base);
    let names = zone_names(&morph);
    let mut spec = inp.base.spec();
    let (cap_j, cap_kg, runtime_note) = match &inp.limits {
        Some(l) => {
            spec.max_speed_mps = l.max_speed_mps;
            spec.max_slope_deg = l.max_slope_deg;
            spec.max_step_m = l.max_step_m;
            (l.battery_j, l.capacity_kg, "가슴 프리셋 배터리".to_string())
        }
        None => (battery_capacity_j(&spec), inp.base.carry_capacity_kg(), format!("순항 전력 × {:.0}분 가정", BATTERY_RUNTIME_S / 60.0)),
    };

    // head: vision (the camera wherever the form has it)
    let head = names.head.map(|label| match inp.vision_confidence {
        Some(c) => {
            let ms = inp
                .frame_ms
                .map(|m| format!(", 프레임 {m:.0} ms"))
                .unwrap_or_default();
            part(
                "head",
                label,
                c * 100.0,
                100.0,
                c,
                format!("최근 인식 신뢰도 {:.0}%{ms}", c * 100.0),
            )
        }
        None => part("head", label, 100.0, 100.0, 1.0, "인식 기록 없음".into()),
    });

    // thorax: battery
    let used = inp.energy_used_j.max(0.0).min(cap_j);
    let battery = (1.0 - used / cap_j) as f32;
    let thorax = part(
        "thorax",
        names.thorax,
        battery * 100.0,
        100.0,
        battery,
        format!("잔량 {:.0} / {:.0} kJ ({runtime_note})", (cap_j - used) / 1000.0, cap_j / 1000.0),
    );

    // stomach: cargo (ratio is the remaining capacity, on this ground when the drive limits it)
    let carried = inp.carried_kg.max(0.0);
    let here = inp.capacity_here.as_ref().filter(|(kg, _)| *kg < cap_kg);
    let room = here.map_or(cap_kg, |(kg, _)| *kg);
    let load_ratio = 1.0 - carried / room.max(1e-3);
    let frac = inp.body_mass_kg.filter(|m| *m > 0.0).map(|m| format!(" · 몸무게의 {:.0}%", 100.0 * carried / m)).unwrap_or_default();
    let stomach = names.cargo.map(|(id, label)| {
        part(
            id,
            label,
            carried,
            cap_kg,
            load_ratio,
            match here {
                Some((kg, ground)) => format!("적재 {carried:.1} / {cap_kg:.0} kg{frac} · {ground}에서는 구동력이 {kg:.1} kg까지"),
                None => format!("적재 {carried:.1} / {cap_kg:.0} kg{frac}"),
            },
        )
    });

    // arm_l: grip margin (2.0 or more counts as full)
    let arm_l = names.arms.map(|(grip, _)| match inp.grip_margin {
        Some(m) => part(
            "arm_l",
            grip,
            m,
            2.0,
            m / 2.0,
            format!("마찰 용량 / 필요 힘 = {m:.2} (안전계수 포함, 1 이상이면 안정)"),
        ),
        None => part("arm_l", grip, 2.0, 2.0, 1.0, "파지 대기".into()),
    });

    // arm_r: the working arm's wear (reducer life used).
    let wear = inp.arm_wear.clamp(0.0, 1.0);
    let arm_r = names.arms.map(|(_, joints)| {
        part(
            "arm_r",
            joints,
            (1.0 - wear) * 100.0,
            100.0,
            1.0 - wear,
            format!("작업 팔 감속기 수명 {:.2}% 사용 · 채취 타격 {}회", inp.arm_wear * 100.0, inp.joint_strikes),
        )
    });

    // leg_l: drive. This is a speed gauge, not a health bar: a robot standing
    // still is not hurt, so the level stays "ok" unless the battery is empty and
    // the base cannot move at all.
    let sf = inp.speed_factor.clamp(0.0, 1.0);
    let speed = sf * spec.max_speed_mps;
    let depleted = used >= cap_j;
    let mut leg_l = part(
        "leg_l",
        &format!("{} · 구동", names.lower),
        speed,
        spec.max_speed_mps,
        sf,
        if depleted {
            "배터리가 바닥나 움직일 수 없음".to_string()
        } else if speed < 0.01 {
            format!("정지 · 최고 {:.2} m/s (속도 계기: 멈춰 있어도 이상이 아님)", spec.max_speed_mps)
        } else {
            format!("속도 {:.2} / {:.2} m/s (속도 계기)", speed, spec.max_speed_mps)
        },
    );
    leg_l.level = if depleted { BodyLevel::Crit } else { BodyLevel::Ok };

    // leg_r: terrain margin (the route's slope and step limits, the ground's friction and drive limits included)
    let ground = inp.balance.as_ref().map(|b| b.ground.as_str()).filter(|g| !g.is_empty());
    let terrain_label = format!("{} · 지형", names.lower);
    let leg_r = match inp.terrain_margin {
        Some(m) => part(
            "leg_r",
            &terrain_label,
            m * 100.0,
            100.0,
            m,
            format!(
                "경사·턱 한계까지 여유 {:.0}% (기하 한계 {:.0}°, {:.0} cm{})",
                m.clamp(0.0, 1.0) * 100.0,
                spec.max_slope_deg,
                spec.max_step_m * 100.0,
                ground.map(|g| format!(" · {g}의 마찰·구동력 한계 포함")).unwrap_or_default()
            ),
        ),
        None => part("leg_r", &terrain_label, 100.0, 100.0, 1.0, "경로 없음".into()),
    };

    let parts: Vec<BodyPart> = [head, Some(thorax), stomach, arm_l, arm_r, Some(leg_l), Some(leg_r)].into_iter().flatten().collect();

    let level_of = |id: &str| -> Option<(BodyLevel, f32)> { parts.iter().find(|p| p.id == id).map(|p| (p.level, p.ratio)) };
    let mut effects = Vec::new();
    // Fatigue: short term the hottest joint's heat (above 1 it is derated),
    // long term the most worn reducer.
    let heat = inp.balance.as_ref().map_or(0.0, |b| b.heat_max);
    let worn = inp.wear_max.max(inp.arm_wear);
    let fatigue_level = BodyLevel::from_ratio(1.0 - heat.min(1.0)).max(BodyLevel::from_ratio(1.0 - worn.min(1.0)));
    effects.push(BodyEffect {
        id: "fatigue".into(),
        label_ko: "피로".into(),
        level: fatigue_level,
        detail: format!(
            "발열 {heat:.2} (1 넘으면 감액) · 수명 사용 {:.2}%{}",
            worn * 100.0,
            inp.wear_joint.as_ref().map(|j| format!(" ({j})")).unwrap_or_default()
        ),
    });
    // Load: what is left of the cargo place (a form without one has no load effect).
    if let Some((load_level, load_left)) = names.cargo.and_then(|(id, _)| level_of(id)) {
        effects.push(BodyEffect {
            id: "load".into(),
            label_ko: "하중".into(),
            level: load_level,
            detail: format!(
                "적재 여유 {:.0}%{}",
                load_left * 100.0,
                inp.body_mass_kg.filter(|m| *m > 0.0).map(|m| format!(" · 몸무게의 {:.0}%", 100.0 * carried / m)).unwrap_or_default()
            ),
        });
    }
    // Slip: the friction picture of the ground under the robot when the
    // balance model has one, else the route's margin to the slope limit.
    match inp.balance.as_ref().and_then(|b| b.contact.map(|c| (b, c))) {
        Some((b, c)) => effects.push(BodyEffect {
            id: "slip".into(),
            label_ko: "미끄러짐".into(),
            level: slip_level(c.margin),
            detail: format!(
                "{} · 마찰 이용률 {:.0}% (필요 {:.2} / 지면 {:.2}){}",
                if b.ground.is_empty() { "지면" } else { b.ground.as_str() },
                (c.utilisation * 100.0).min(999.0),
                c.required_mu,
                c.available_mu,
                c.slip_probability.map(|p| format!(" · 한 걸음 미끄럼 확률 {:.1}%", p * 100.0)).unwrap_or_default()
            ),
        }),
        None => {
            let (slip_level, slip_ratio) = level_of("leg_r").unwrap_or((BodyLevel::Ok, 1.0));
            effects.push(BodyEffect {
                id: "slip".into(),
                label_ko: "미끄러짐".into(),
                level: slip_level,
                detail: format!("지형 한계 여유 {:.0}%", slip_ratio * 100.0),
            });
        }
    }
    // Balance and joint strain, when the part presets are in play.
    if let Some(b) = &inp.balance {
        // Stability: tipping (the ZMP's margin) and sliding (the friction
        // margin) — whichever is closer to failing decides the level.
        let tip = BodyLevel::from_ratio(b.margin);
        let (level, slide) = match &b.contact {
            Some(c) => (tip.max(slip_level(c.margin)), format!(" · 미끄러짐 여유 {:.0}%{}", (1.0 - c.utilisation).max(-9.99) * 100.0, if slip_level(c.margin) > tip { " — 미끄러짐이 먼저" } else { "" })),
            None => (tip, String::new()),
        };
        effects.push(BodyEffect {
            id: "balance".into(),
            label_ko: "자세 안정".into(),
            level,
            detail: format!("{} · 넘어짐 여유 {:.0}% (피드포워드 없으면 {:.0}%){slide}", b.source_ko, b.margin * 100.0, b.margin_noff * 100.0),
        });
        // Joint load: the motor sizing rules, peak against the peak rating and
        // RMS against the continuous one, whichever is closer to its limit.
        let index = b.load_index.max(b.worst_ratio);
        effects.push(BodyEffect {
            id: "strain".into(),
            label_ko: "관절 부하".into(),
            level: BodyLevel::from_ratio(1.0 - index),
            detail: match (&b.load_joint, &b.worst_joint) {
                (Some(j), _) => format!("{j} 한계 대비 {:.0}% (지금 피크 {:.0}%, RMS/연속 {:.2})", index * 100.0, b.worst_ratio * 100.0, b.load_rms),
                (None, Some(j)) => format!("{j} 피크 대비 {:.0}%", b.worst_ratio * 100.0),
                (None, None) => "부하 없음".into(),
            },
        });
    }

    BodyStatus {
        balance: inp.balance.clone(),
        parts,
        effects,
        hand: HandSummary {
            fingers: inp.fingers,
            grip_margin: inp.grip_margin,
        },
        foot: FootSummary {
            base: inp.base,
            speed_factor: sf,
            terrain_margin: inp.terrain_margin,
        },
        morph,
        form: morph.form().map_or("custom", |f| f.id).to_string(),
        form_ko: morph.label_ko(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle(base: BaseKind) -> BodyInputs {
        BodyInputs {
            base,
            morph: Morphology::humanoid(base),
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
            speed_factor: 1.0,
            terrain_margin: None,
            limits: None,
            body_mass_kg: None,
            capacity_here: None,
            balance: None,
        }
    }

    #[test]
    fn level_thresholds() {
        assert_eq!(BodyLevel::from_ratio(1.0), BodyLevel::Ok);
        assert_eq!(BodyLevel::from_ratio(0.6), BodyLevel::Ok);
        assert_eq!(BodyLevel::from_ratio(0.59), BodyLevel::Warn);
        assert_eq!(BodyLevel::from_ratio(0.3), BodyLevel::Warn);
        assert_eq!(BodyLevel::from_ratio(0.29), BodyLevel::Bad);
        assert_eq!(BodyLevel::from_ratio(0.1), BodyLevel::Bad);
        assert_eq!(BodyLevel::from_ratio(0.09), BodyLevel::Crit);
        assert_eq!(BodyLevel::from_ratio(0.0), BodyLevel::Crit);
    }

    #[test]
    fn idle_robot_is_all_ok_with_seven_parts_in_order() {
        let s = compute(&idle(BaseKind::Wheel));
        let ids: Vec<&str> = s.parts.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, PART_IDS);
        assert!(s.parts.iter().all(|p| p.level == BodyLevel::Ok));
        assert!(s.effects.iter().all(|e| e.level == BodyLevel::Ok));
        assert!(s.effects.iter().all(|e| e.id != "heat"));
    }

    #[test]
    fn battery_follows_energy_used() {
        let spec = BaseKind::Wheel.spec();
        let cap = battery_capacity_j(&spec);
        let mut inp = idle(BaseKind::Wheel);
        inp.energy_used_j = cap * 0.75;
        let s = compute(&inp);
        let thorax = s.parts.iter().find(|p| p.id == "thorax").unwrap();
        assert!((thorax.ratio - 0.25).abs() < 1e-4);
        assert_eq!(thorax.level, BodyLevel::Bad);
        // Over-use clamps to empty instead of going negative.
        inp.energy_used_j = cap * 3.0;
        let s = compute(&inp);
        let thorax = s.parts.iter().find(|p| p.id == "thorax").unwrap();
        assert_eq!(thorax.ratio, 0.0);
        assert_eq!(thorax.level, BodyLevel::Crit);
    }

    #[test]
    fn cargo_ratio_is_remaining_capacity_per_base() {
        let mut inp = idle(BaseKind::Track);
        inp.carried_kg = 15.0;
        let s = compute(&inp);
        let stomach = s.parts.iter().find(|p| p.id == "stomach").unwrap();
        assert!((stomach.ratio - 0.5).abs() < 1e-5);
        assert_eq!(stomach.max, 30.0);
        // The same load is worse on a base with a smaller limit.
        let mut small = idle(BaseKind::Mecanum);
        small.carried_kg = 15.0;
        let s = compute(&small);
        let stomach = s.parts.iter().find(|p| p.id == "stomach").unwrap();
        assert_eq!(stomach.ratio, 0.0);
        assert_eq!(stomach.level, BodyLevel::Crit);
    }

    #[test]
    fn grip_margin_two_is_full_and_below_one_is_poor() {
        let mut inp = idle(BaseKind::Wheel);
        inp.grip_margin = Some(2.5);
        let s = compute(&inp);
        assert_eq!(s.parts[3].ratio, 1.0);
        inp.grip_margin = Some(1.0);
        let s = compute(&inp);
        assert!((s.parts[3].ratio - 0.5).abs() < 1e-6);
        // 0.9 / 2 = 0.45 -> warn; 0.4 / 2 = 0.2 -> bad.
        inp.grip_margin = Some(0.9);
        let s = compute(&inp);
        assert_eq!(s.parts[3].level, BodyLevel::Warn);
        inp.grip_margin = Some(0.4);
        let s = compute(&inp);
        assert_eq!(s.parts[3].level, BodyLevel::Bad);
        assert_eq!(s.hand.grip_margin, Some(0.4));
    }

    #[test]
    fn joint_wear_raises_fatigue_effect() {
        let mut inp = idle(BaseKind::Wheel);
        inp.arm_wear = 1.0;
        inp.joint_strikes = 9000;
        let s = compute(&inp);
        let fatigue = s.effects.iter().find(|e| e.id == "fatigue").unwrap();
        assert_eq!(fatigue.level, BodyLevel::Crit);
        assert!(s.parts[4].detail.contains("9000"));
        inp.arm_wear = 2.0;
        // Life used past the rating clamps to fully worn, never negative.
        let s = compute(&inp);
        assert_eq!(s.parts[4].ratio, 0.0);
        // A worn joint elsewhere also tires the robot.
        let mut leg = idle(BaseKind::HumanFoot);
        leg.wear_max = 0.75;
        leg.wear_joint = Some("knee".into());
        let f = compute(&leg).effects.into_iter().find(|e| e.id == "fatigue").unwrap();
        assert_eq!(f.level, BodyLevel::Bad);
        assert!(f.detail.contains("knee"));
        assert!(BodyLevel::Crit > BodyLevel::Ok, "max is the worse level");
    }

    fn balance(margin: f32, contact: Option<Contact>) -> BalanceSummary {
        BalanceSummary {
            plan: "mid".into(),
            source: "slope".into(),
            source_ko: "경사".into(),
            state: "up_hi|move".into(),
            margin,
            margin_noff: margin,
            trunk_lean_deg: 0.0,
            head_tilt_deg: 0.0,
            worst_joint: Some("drive".into()),
            worst_ratio: 0.3,
            effort_sum: 0.1,
            mass_kg: 20.0,
            com_height_m: 0.5,
            heat_max: 0.2,
            load_joint: Some("drive".into()),
            load_index: 0.4,
            load_rms: 0.4,
            joints: vec![],
            couplings: vec![],
            ground: "풀밭".into(),
            contact,
        }
    }

    #[test]
    fn the_ground_lowers_what_the_drive_carries_and_sliding_can_come_first() {
        let mut inp = idle(BaseKind::Wheel);
        inp.carried_kg = 4.0;
        inp.body_mass_kg = Some(20.0);
        inp.capacity_here = Some((5.0, "모래".into()));
        let s = compute(&inp);
        let stomach = s.parts.iter().find(|p| p.id == "stomach").unwrap();
        assert!((stomach.ratio - 0.2).abs() < 1e-5, "4 of the 5 kg the drive pushes on sand");
        assert!(stomach.detail.contains("모래") && stomach.detail.contains("몸무게의 20%"), "{}", stomach.detail);
        // A wheel on a grass slope: far from tipping, close to sliding.
        let c = crate::surface::contact(BaseKind::Wheel, crate::surface::Surface::Grass.props(), 13.0, 0.0, true);
        inp.balance = Some(balance(0.8, Some(c)));
        let s = compute(&inp);
        let bal = s.effects.iter().find(|e| e.id == "balance").unwrap();
        assert!(bal.level > BodyLevel::Ok && bal.detail.contains("미끄러짐이 먼저"), "{bal:?}");
        let slip = s.effects.iter().find(|e| e.id == "slip").unwrap();
        assert!(slip.detail.contains("풀밭") && slip.detail.contains("마찰 이용률"), "{}", slip.detail);
        let strain = s.effects.iter().find(|e| e.id == "strain").unwrap();
        assert!(strain.detail.contains("RMS/연속 0.40"), "{}", strain.detail);
    }

    #[test]
    fn terrain_margin_drives_slip_and_leg_r() {
        let mut inp = idle(BaseKind::Wheel);
        inp.terrain_margin = Some(0.2);
        let s = compute(&inp);
        assert_eq!(s.parts[6].level, BodyLevel::Bad);
        let slip = s.effects.iter().find(|e| e.id == "slip").unwrap();
        assert_eq!(slip.level, BodyLevel::Bad);
        assert_eq!(s.foot.terrain_margin, Some(0.2));
    }

    #[test]
    fn non_finite_inputs_do_not_panic_and_read_as_worst() {
        let mut inp = idle(BaseKind::Wheel);
        inp.vision_confidence = Some(f32::NAN);
        let s = compute(&inp);
        assert_eq!(s.parts[0].ratio, 0.0);
        assert_eq!(s.parts[0].level, BodyLevel::Crit);
    }

    #[test]
    fn serializes_levels_in_snake_case() {
        let s = compute(&idle(BaseKind::HumanFoot));
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"level\":\"ok\""));
        assert!(json.contains("\"base\":\"human_foot\""));
    }

    #[test]
    fn a_parked_robot_is_not_hurt_but_a_flat_battery_is() {
        let mut i = idle(BaseKind::Wheel);
        i.speed_factor = 0.0;
        let leg = |inp: &BodyInputs| compute(inp).parts.into_iter().find(|p| p.id == "leg_l").unwrap();
        let parked = leg(&i);
        assert_eq!(parked.level, BodyLevel::Ok);
        assert_eq!(parked.ratio, 0.0);
        assert!(parked.detail.contains("정지"));
        i.energy_used_j = 1e12;
        let dead = leg(&i);
        assert_eq!(dead.level, BodyLevel::Crit);
        assert!(dead.detail.contains("배터리"));
    }
}
