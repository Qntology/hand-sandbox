//! Robot status shown like a body-part health display (head, thorax, stomach,
//! arms, legs).
//!
//! Every value is computed from simulation variables. The robot has ONE hand
//! and one locomotion base, so the two "arms" are two metrics of that hand
//! (grip and joints) and the two "legs" are two metrics of the base (drive and
//! terrain margin). They are not anatomical left and right.
//!
//! Modelling assumptions (not measured values) are named constants here and
//! listed in `data/README.md`:
//! * [`BATTERY_RUNTIME_S`]: battery energy is cruise power times this runtime.
//! * [`JOINT_ENDURANCE_STRIKES`]: mining strikes before the hand joints count
//!   as fully worn.
//! * [`BaseKind::carry_capacity_kg`]: cargo limit per base.

use serde::{Deserialize, Serialize};

use crate::balance::{Coupling, JointLoad};
use crate::base::{BaseKind, BaseSpec};

/// Battery energy equals cruise power times this many seconds (assumption).
pub const BATTERY_RUNTIME_S: f64 = 1800.0;

/// Mining strikes after which the hand joints count as fully worn (assumption).
pub const JOINT_ENDURANCE_STRIKES: u32 = 400;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    pub joints: Vec<JointLoad>,
    /// The N:N cells of this state.
    pub couplings: Vec<Coupling>,
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
}

/// Simulation variables the status is computed from.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyInputs {
    pub base: BaseKind,
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
    /// Current speed over the base's top speed, 0..1.
    pub speed_factor: f32,
    /// Smallest remaining margin to the base's slope or step limit on the
    /// current route segment, 0..1. `None` without a route.
    pub terrain_margin: Option<f32>,
    /// Limits from the part presets (battery, cargo, speed, slope, step).
    pub limits: Option<BodyLimits>,
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

/// Computes the seven zones and the active effects from `inp`.
pub fn compute(inp: &BodyInputs) -> BodyStatus {
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

    // head: vision
    let head = match inp.vision_confidence {
        Some(c) => {
            let ms = inp
                .frame_ms
                .map(|m| format!(", 프레임 {m:.0} ms"))
                .unwrap_or_default();
            part(
                "head",
                "머리 · 비전",
                c * 100.0,
                100.0,
                c,
                format!("최근 인식 신뢰도 {:.0}%{ms}", c * 100.0),
            )
        }
        None => part("head", "머리 · 비전", 100.0, 100.0, 1.0, "인식 기록 없음".into()),
    };

    // thorax: battery
    let used = inp.energy_used_j.max(0.0).min(cap_j);
    let battery = (1.0 - used / cap_j) as f32;
    let thorax = part(
        "thorax",
        "가슴 · 배터리",
        battery * 100.0,
        100.0,
        battery,
        format!("잔량 {:.0} / {:.0} kJ ({runtime_note})", (cap_j - used) / 1000.0, cap_j / 1000.0),
    );

    // stomach: cargo (ratio is the remaining capacity)
    let carried = inp.carried_kg.max(0.0);
    let load_ratio = 1.0 - carried / cap_kg;
    let stomach = part(
        "stomach",
        "배 · 적재",
        carried,
        cap_kg,
        load_ratio,
        format!("적재 {carried:.1} / {cap_kg:.0} kg"),
    );

    // arm_l: grip margin (2.0 or more counts as full)
    let arm_l = match inp.grip_margin {
        Some(m) => part(
            "arm_l",
            "손 · 파지",
            m,
            2.0,
            m / 2.0,
            format!("마찰 용량 / 필요 힘 = {m:.2} (안전계수 포함, 1 이상이면 안정)"),
        ),
        None => part("arm_l", "손 · 파지", 2.0, 2.0, 1.0, "파지 대기".into()),
    };

    // arm_r: joint wear
    let wear = (inp.joint_strikes as f32 / JOINT_ENDURANCE_STRIKES as f32).clamp(0.0, 1.0);
    let arm_r = part(
        "arm_r",
        "손 · 관절",
        (1.0 - wear) * 100.0,
        100.0,
        1.0 - wear,
        format!(
            "채취 타격 {} / {} (내구 가정값)",
            inp.joint_strikes, JOINT_ENDURANCE_STRIKES
        ),
    );

    // leg_l: drive. This is a speed gauge, not a health bar: a robot standing
    // still is not hurt, so the level stays "ok" unless the battery is empty and
    // the base cannot move at all.
    let sf = inp.speed_factor.clamp(0.0, 1.0);
    let speed = sf * spec.max_speed_mps;
    let depleted = used >= cap_j;
    let mut leg_l = part(
        "leg_l",
        "발 · 구동",
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

    // leg_r: terrain margin
    let leg_r = match inp.terrain_margin {
        Some(m) => part(
            "leg_r",
            "발 · 지형",
            m * 100.0,
            100.0,
            m,
            format!(
                "경사·턱 한계까지 여유 {:.0}% (한계: {:.0}°, {:.0} cm)",
                m.clamp(0.0, 1.0) * 100.0,
                spec.max_slope_deg,
                spec.max_step_m * 100.0
            ),
        ),
        None => part("leg_r", "발 · 지형", 100.0, 100.0, 1.0, "경로 없음".into()),
    };

    let parts = vec![head, thorax, stomach, arm_l, arm_r, leg_l, leg_r];

    let level_of = |id: &str| -> (BodyLevel, f32) {
        let p = parts.iter().find(|p| p.id == id).expect("known part id");
        (p.level, p.ratio)
    };
    let mut effects = Vec::new();
    let (joint_level, joint_ratio) = level_of("arm_r");
    effects.push(BodyEffect {
        id: "fatigue".into(),
        label_ko: "피로".into(),
        level: joint_level,
        detail: format!("손 관절 내구 {:.0}% 남음", joint_ratio * 100.0),
    });
    let (load_level, load_left) = level_of("stomach");
    effects.push(BodyEffect {
        id: "load".into(),
        label_ko: "하중".into(),
        level: load_level,
        detail: format!("적재 여유 {:.0}%", load_left * 100.0),
    });
    let (slip_level, slip_ratio) = level_of("leg_r");
    effects.push(BodyEffect {
        id: "slip".into(),
        label_ko: "미끄러짐".into(),
        level: slip_level,
        detail: format!("지형 한계 여유 {:.0}%", slip_ratio * 100.0),
    });
    // Balance and joint strain, when the part presets are in play.
    if let Some(b) = &inp.balance {
        effects.push(BodyEffect {
            id: "balance".into(),
            label_ko: "자세 안정".into(),
            level: BodyLevel::from_ratio(b.margin),
            detail: format!("{} · 안정 여유 {:.0}% (피드포워드 없으면 {:.0}%)", b.source_ko, b.margin * 100.0, b.margin_noff * 100.0),
        });
        effects.push(BodyEffect {
            id: "strain".into(),
            label_ko: "관절 부하".into(),
            level: BodyLevel::from_ratio(1.0 - b.worst_ratio),
            detail: match &b.worst_joint {
                Some(j) => format!("{j} 피크 대비 {:.0}% · 발열 {:.2}", b.worst_ratio * 100.0, b.heat_max),
                None => "부하 없음".into(),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle(base: BaseKind) -> BodyInputs {
        BodyInputs {
            base,
            fingers: 5,
            vision_confidence: None,
            frame_ms: None,
            energy_used_j: 0.0,
            carried_kg: 0.0,
            grip_margin: None,
            joint_strikes: 0,
            speed_factor: 1.0,
            terrain_margin: None,
            limits: None,
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
        inp.joint_strikes = JOINT_ENDURANCE_STRIKES;
        let s = compute(&inp);
        let fatigue = s.effects.iter().find(|e| e.id == "fatigue").unwrap();
        assert_eq!(fatigue.level, BodyLevel::Crit);
        inp.joint_strikes = JOINT_ENDURANCE_STRIKES * 2;
        // Strikes past the endurance clamp to fully worn, never negative.
        let s = compute(&inp);
        assert_eq!(s.parts[4].ratio, 0.0);
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
