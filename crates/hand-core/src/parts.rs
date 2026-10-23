//! Part presets for a ~1.2 m logistics robot that a small team can build.
//!
//! Every segment (head, chest, belly/pelvis, arm, leg or chassis, foot) comes
//! in tiers built from off-the-shelf actuators: bus servos and printed PLA
//! (budget), small quasi-direct-drive (QDD) motors and aluminium (mid), large
//! QDDs and carbon fibre (pro). The numbers are datasheet values with sources
//! in `docs/research/parts.md` (the `src` keys below), or estimates marked
//! "est" there with the arithmetic shown. They are what makes a preset matter
//! to the physics: total mass, centre-of-mass height, joint peak and
//! continuous torque, joint speed, drive force and battery energy decide what
//! the robot can lift, how steep it can go and how fast it moves.
//!
//! Geometry of the reference robot (est, research report §5): hip 0.58 m,
//! waist 0.70 m, shoulder 1.00 m, upper arm 0.22 m, forearm 0.20 m.

use serde::{Deserialize, Serialize};

use crate::base::BaseKind;
use crate::hand::FingerCount;

/// Hip height above the ground (or the chassis mast top for wheeled bases), metres.
pub const HIP_M: f32 = 0.58;
/// Waist joint height, metres.
pub const WAIST_M: f32 = 0.70;
/// Shoulder joint height, metres.
pub const SHOULDER_M: f32 = 1.00;
/// Half the distance between the two shoulders, metres (est).
pub const SHOULDER_HALF_WIDTH_M: f32 = 0.18;
pub const UPPER_ARM_M: f32 = 0.22;
pub const FOREARM_M: f32 = 0.20;
/// Centre-to-centre distance of the two feet in stance, metres (est, about a human's).
pub const STANCE_WIDTH_M: f32 = 0.18;
/// Track width of the wheeled chassis, metres (est).
pub const CHASSIS_WIDTH_M: f32 = 0.40;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Budget,
    /// Only for legs: DIY printed cycloidal BLDC actuators (Berkeley Humanoid Lite).
    BudgetPlus,
    Mid,
    Pro,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Budget, Tier::Mid, Tier::Pro];
    pub const LEGS: [Tier; 4] = [Tier::Budget, Tier::BudgetPlus, Tier::Mid, Tier::Pro];

    pub fn id(self) -> &'static str {
        match self {
            Tier::Budget => "budget",
            Tier::BudgetPlus => "budget_plus",
            Tier::Mid => "mid",
            Tier::Pro => "pro",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Tier::Budget => "보급형",
            Tier::BudgetPlus => "보급형+ (자작 BLDC)",
            Tier::Mid => "중급형",
            Tier::Pro => "고급형",
        }
    }

    /// `BudgetPlus` exists only for legs; elsewhere it means `Budget`.
    fn no_plus(self) -> Tier {
        if self == Tier::BudgetPlus {
            Tier::Budget
        } else {
            self
        }
    }
}

/// One actuator, or several ganged on one joint.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Actuator {
    pub name: &'static str,
    pub peak_nm: f32,
    pub cont_nm: f32,
    /// No-load output speed, rad/s.
    pub speed_rad_s: f32,
    pub mass_kg: f32,
    pub price_usd: f32,
    pub backdrivable: bool,
    /// Thermal time constant of the heat model, seconds (est: research §1).
    pub thermal_tau_s: f32,
    /// Source keys in `docs/research/parts.md`.
    pub src: &'static str,
}

impl Actuator {
    /// Two of the same actuator driving one joint.
    pub const fn ganged(self, name: &'static str) -> Actuator {
        Actuator {
            name,
            peak_nm: self.peak_nm * 2.0,
            cont_nm: self.cont_nm * 2.0,
            mass_kg: self.mass_kg * 2.0,
            price_usd: self.price_usd * 2.0,
            ..self
        }
    }

    /// Torque available at joint speed `omega` (linear torque-speed line).
    pub fn torque_at(&self, omega: f32) -> f32 {
        self.peak_nm * (1.0 - (omega.abs() / self.speed_rad_s.max(1e-3))).clamp(0.0, 1.0)
    }
}

/// Feetech STS3215-C018, 12 V [1].
pub const STS3215: Actuator = Actuator { name: "Feetech STS3215 (12 V)", peak_nm: 2.94, cont_nm: 0.98, speed_rad_s: 4.7, mass_kg: 0.055, price_usd: 32.0, backdrivable: false, thermal_tau_s: 200.0, src: "1" };
/// Hiwonder HTD-45H, 11.1 V [4]; continuous is an estimate (a third of peak, like Feetech).
pub const HTD45H: Actuator = Actuator { name: "Hiwonder HTD-45H", peak_nm: 4.41, cont_nm: 1.5, speed_rad_s: 5.8, mass_kg: 0.064, price_usd: 30.0, backdrivable: false, thermal_tau_s: 200.0, src: "4" };
/// Dynamixel XM430-W350 [7][10].
pub const XM430: Actuator = Actuator { name: "Dynamixel XM430-W350", peak_nm: 4.1, cont_nm: 0.82, speed_rad_s: 4.8, mass_kg: 0.082, price_usd: 310.0, backdrivable: false, thermal_tau_s: 250.0, src: "7,10" };
/// Robstride 01 [15].
pub const RS01: Actuator = Actuator { name: "Robstride 01", peak_nm: 17.0, cont_nm: 6.0, speed_rad_s: 33.0, mass_kg: 0.38, price_usd: 130.0, backdrivable: true, thermal_tau_s: 600.0, src: "15" };
/// Robstride 02 [15].
pub const RS02: Actuator = Actuator { name: "Robstride 02", peak_nm: 17.0, cont_nm: 6.0, speed_rad_s: 42.9, mass_kg: 0.405, price_usd: 145.0, backdrivable: true, thermal_tau_s: 600.0, src: "15" };
/// Robstride 03 [16].
pub const RS03: Actuator = Actuator { name: "Robstride 03", peak_nm: 60.0, cont_nm: 20.0, speed_rad_s: 20.4, mass_kg: 0.88, price_usd: 225.0, backdrivable: true, thermal_tau_s: 800.0, src: "16" };
/// Robstride 04 [17]; 40 N·m continuous needs the large heatsink.
pub const RS04: Actuator = Actuator { name: "Robstride 04", peak_nm: 120.0, cont_nm: 40.0, speed_rad_s: 20.9, mass_kg: 1.42, price_usd: 255.0, backdrivable: true, thermal_tau_s: 900.0, src: "17" };
/// Berkeley Humanoid Lite 6512 printed cycloidal actuator [22]; continuous and speed est.
pub const BHL6512: Actuator = Actuator { name: "BHL 6512 (자작 사이클로이드)", peak_nm: 20.0, cont_nm: 7.0, speed_rad_s: 15.0, mass_kg: 0.55, price_usd: 170.0, backdrivable: true, thermal_tau_s: 500.0, src: "22" };

/// The seven segments of the body, as the N:N balance tables name them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Segment {
    Head,
    Chest,
    Belly,
    Arm,
    Hand,
    Leg,
    Foot,
}

impl Segment {
    pub const ALL: [Segment; 7] = [Segment::Head, Segment::Chest, Segment::Belly, Segment::Arm, Segment::Hand, Segment::Leg, Segment::Foot];

    pub fn id(self) -> &'static str {
        match self {
            Segment::Head => "head",
            Segment::Chest => "chest",
            Segment::Belly => "belly",
            Segment::Arm => "arm",
            Segment::Hand => "hand",
            Segment::Leg => "leg",
            Segment::Foot => "foot",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Segment::Head => "머리",
            Segment::Chest => "가슴",
            Segment::Belly => "배(골반)",
            Segment::Arm => "팔",
            Segment::Hand => "손",
            Segment::Leg => "다리",
            Segment::Foot => "발",
        }
    }

    pub fn from_id(id: &str) -> Option<Segment> {
        Segment::ALL.into_iter().find(|s| s.id() == id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeadPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    /// CoM above the neck joint, metres.
    pub com_up_m: f32,
    pub neck: Actuator,
    pub sensor: &'static str,
    /// Share of a body tilt the neck can cancel to keep the camera level (est:
    /// servo backlash and speed; a QDD neck would do better).
    pub gaze_stabilization: f32,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChestPreset {
    pub tier: Tier,
    /// Structure and computer, without the battery.
    pub mass_kg: f32,
    pub battery_kg: f32,
    pub battery_wh: f32,
    /// CoM above the waist joint, metres.
    pub com_up_m: f32,
    pub computer: &'static str,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BellyPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    /// CoM below the waist joint, metres.
    pub com_down_m: f32,
    /// Waist yaw actuator (turns the trunk; carries no pitch load).
    pub waist: Actuator,
    /// Waist pitch actuator, if the tier has one. Without it the trunk is
    /// rigid in pitch: it cannot lean to restore balance, and the pitch load
    /// goes into the structure (research §5: a pitch joint needs ≥12 N·m
    /// continuous, which only the pro tier's Robstride 03 has).
    pub pitch: Option<Actuator>,
    /// Largest trunk lean the waist can hold, degrees (0 without a pitch joint).
    pub max_lean_deg: f32,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ArmPreset {
    pub tier: Tier,
    /// One arm without the hand.
    pub mass_kg: f32,
    /// CoM along the arm from the shoulder, metres (actuators near the shoulder pull it in).
    pub com_m: f32,
    pub shoulder: Actuator,
    pub elbow: Actuator,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LegPreset {
    pub tier: Tier,
    /// One leg without the foot.
    pub mass_kg: f32,
    /// CoM below the hip, metres.
    pub com_down_m: f32,
    pub knee: Actuator,
    pub ankle: Actuator,
    /// Walking speed limit as a Froude number v² / (g · leg length) (research §5 tier totals).
    pub froude: f32,
    pub foot_mass_kg: f32,
    pub foot_len_m: f32,
    pub foot_width_m: f32,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChassisPreset {
    pub tier: Tier,
    pub name: &'static str,
    pub mass_kg: f32,
    /// Height of the chassis' own CoM, metres.
    pub com_m: f32,
    pub max_speed_mps: f32,
    /// Continuous traction force of all drive wheels together, N.
    pub drive_force_n: f32,
    pub power_w: f32,
    pub cost_usd: f32,
    pub src: &'static str,
}

impl HeadPreset {
    pub fn of(tier: Tier) -> HeadPreset {
        match tier.no_plus() {
            Tier::Budget => HeadPreset { tier: Tier::Budget, mass_kg: 0.45, com_up_m: 0.07, neck: STS3215, sensor: "OAK-D Lite", gaze_stabilization: 0.45, power_w: 5.0, cost_usd: 215.0 },
            Tier::Mid => HeadPreset { tier: Tier::Mid, mass_kg: 0.44, com_up_m: 0.07, neck: STS3215, sensor: "RealSense D435i", gaze_stabilization: 0.55, power_w: 4.0, cost_usd: 415.0 },
            _ => HeadPreset { tier: Tier::Pro, mass_kg: 0.45, com_up_m: 0.07, neck: XM430, sensor: "RealSense D435i + RGB", gaze_stabilization: 0.75, power_w: 6.0, cost_usd: 1050.0 },
        }
    }
}

impl ChestPreset {
    pub fn of(tier: Tier) -> ChestPreset {
        match tier.no_plus() {
            Tier::Budget => ChestPreset { tier: Tier::Budget, mass_kg: 1.2, battery_kg: 0.72, battery_wh: 160.0, com_up_m: 0.15, computer: "Raspberry Pi 5", power_w: 6.0, cost_usd: 150.0 },
            Tier::Mid => ChestPreset { tier: Tier::Mid, mass_kg: 1.6, battery_kg: 1.6, battery_wh: 324.0, com_up_m: 0.16, computer: "Jetson Orin Nano Super", power_w: 15.0, cost_usd: 410.0 },
            _ => ChestPreset { tier: Tier::Pro, mass_kg: 1.5, battery_kg: 2.5, battery_wh: 420.0, com_up_m: 0.17, computer: "Orin 급", power_w: 25.0, cost_usd: 1500.0 },
        }
    }
}

impl BellyPreset {
    pub fn of(tier: Tier) -> BellyPreset {
        match tier.no_plus() {
            Tier::Budget => BellyPreset { tier: Tier::Budget, mass_kg: 0.7, com_down_m: 0.06, waist: HTD45H.ganged("2× HTD-45H"), pitch: None, max_lean_deg: 0.0, power_w: 3.0, cost_usd: 70.0 },
            Tier::Mid => BellyPreset { tier: Tier::Mid, mass_kg: 1.3, com_down_m: 0.06, waist: RS02, pitch: None, max_lean_deg: 0.0, power_w: 5.0, cost_usd: 185.0 },
            _ => BellyPreset { tier: Tier::Pro, mass_kg: 1.9, com_down_m: 0.06, waist: RS03, pitch: Some(RS03), max_lean_deg: 25.0, power_w: 5.0, cost_usd: 285.0 },
        }
    }
}

impl ArmPreset {
    pub fn of(tier: Tier) -> ArmPreset {
        match tier.no_plus() {
            Tier::Budget => ArmPreset { tier: Tier::Budget, mass_kg: 0.64, com_m: 0.17, shoulder: HTD45H.ganged("2× HTD-45H"), elbow: STS3215, power_w: 10.0, cost_usd: 135.0 },
            Tier::Mid => ArmPreset { tier: Tier::Mid, mass_kg: 2.1, com_m: 0.11, shoulder: RS02, elbow: RS01, power_w: 15.0, cost_usd: 600.0 },
            _ => ArmPreset { tier: Tier::Pro, mass_kg: 3.0, com_m: 0.08, shoulder: RS03, elbow: RS02, power_w: 20.0, cost_usd: 800.0 },
        }
    }
}

impl LegPreset {
    pub fn of(tier: Tier) -> LegPreset {
        match tier {
            Tier::Budget => LegPreset { tier, mass_kg: 1.45, com_down_m: 0.23, knee: HTD45H.ganged("2× HTD-45H"), ankle: HTD45H.ganged("2× HTD-45H"), froude: 0.02, foot_mass_kg: 0.10, foot_len_m: 0.16, foot_width_m: 0.08, power_w: 50.0, cost_usd: 315.0 },
            Tier::BudgetPlus => LegPreset { tier, mass_kg: 3.0, com_down_m: 0.15, knee: BHL6512, ankle: BHL6512, froude: 0.08, foot_mass_kg: 0.20, foot_len_m: 0.18, foot_width_m: 0.09, power_w: 60.0, cost_usd: 970.0 },
            Tier::Mid => LegPreset { tier, mass_kg: 4.8, com_down_m: 0.14, knee: RS03, ankle: RS02.ganged("2× Robstride 02 (평행 링크)"), froude: 0.15, foot_mass_kg: 0.30, foot_len_m: 0.20, foot_width_m: 0.10, power_w: 70.0, cost_usd: 1190.0 },
            Tier::Pro => LegPreset { tier, mass_kg: 7.1, com_down_m: 0.14, knee: RS04, ankle: RS03.ganged("2× Robstride 03 (평행 링크)"), froude: 0.40, foot_mass_kg: 0.60, foot_len_m: 0.22, foot_width_m: 0.10, power_w: 70.0, cost_usd: 1510.0 },
        }
    }
}

impl ChassisPreset {
    pub fn of(tier: Tier) -> ChassisPreset {
        match tier.no_plus() {
            // LeKiwi: 3× STS3215 on 100 mm omni wheels [30]; force est 3 × 0.98 N·m × 0.9 / 0.05 m × cos 30°.
            Tier::Budget => ChassisPreset { tier: Tier::Budget, name: "LeKiwi 급 (STS3215 ×3)", mass_kg: 1.7, com_m: 0.06, max_speed_mps: 0.24, drive_force_n: 46.0, power_w: 30.0, cost_usd: 226.0, src: "30" },
            // Hub-motor base (est, between LeKiwi and Scout Mini): 4 × 100 W, force = 400 W × 0.9 / 1.5 m/s.
            Tier::Mid => ChassisPreset { tier: Tier::Mid, name: "허브 모터 4륜 (추정)", mass_kg: 8.0, com_m: 0.10, max_speed_mps: 1.5, drive_force_n: 240.0, power_w: 120.0, cost_usd: 900.0, src: "est" },
            // AgileX Tracer: 2 × 150 W, 1.6 m/s, 100 kg payload [33]; force est 300 W × 0.9 / 1.6 m/s.
            _ => ChassisPreset { tier: Tier::Pro, name: "AgileX Tracer 급", mass_kg: 30.0, com_m: 0.12, max_speed_mps: 1.6, drive_force_n: 170.0, power_w: 300.0, cost_usd: 7590.0, src: "33" },
        }
    }
}

/// Mass of the hand for a finger count (est from the hands in research §2:
/// a two-finger XM430 jaw ~0.35 kg, an underactuated three-finger ~0.40 kg,
/// five-finger dexterous hands 0.54–0.60 kg).
pub fn hand_mass_kg(fingers: FingerCount) -> f32 {
    match fingers {
        FingerCount::Two => 0.35,
        FingerCount::Three => 0.40,
        FingerCount::Four => 0.55,
        FingerCount::Five => 0.60,
    }
}

/// The preset choice for every segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyPlan {
    pub head: Tier,
    pub chest: Tier,
    pub belly: Tier,
    pub arm: Tier,
    /// Legs for a walking base, the chassis for a wheeled or tracked one.
    pub leg: Tier,
    /// Battery in the pelvis instead of the chest (lowers the CoM a little and
    /// takes inertia off the waist; research §6).
    #[serde(default)]
    pub battery_low: bool,
}

impl Default for BodyPlan {
    fn default() -> Self {
        BodyPlan::uniform(Tier::Mid)
    }
}

impl BodyPlan {
    pub fn uniform(tier: Tier) -> BodyPlan {
        BodyPlan { head: tier, chest: tier, belly: tier, arm: tier, leg: tier, battery_low: false }
    }

    /// Short id such as `mid` or `h:mid c:pro b:mid a:budget l:mid bat:low`.
    pub fn id(&self) -> String {
        let u = [self.head, self.chest, self.belly, self.arm].iter().all(|t| *t == self.head) && self.leg == self.head;
        if u && !self.battery_low {
            return self.head.id().to_string();
        }
        format!(
            "h:{} c:{} b:{} a:{} l:{}{}",
            self.head.id(),
            self.chest.id(),
            self.belly.id(),
            self.arm.id(),
            self.leg.id(),
            if self.battery_low { " bat:low" } else { "" }
        )
    }

    /// Named starting points that make sense together (research §6).
    pub fn named(id: &str) -> Option<BodyPlan> {
        Some(match id {
            "budget" => BodyPlan::uniform(Tier::Budget),
            "mid" => BodyPlan::uniform(Tier::Mid),
            "pro" => BodyPlan::uniform(Tier::Pro),
            // Budget arms and head on a wheeled base: an SO-101-like logistics robot.
            "budget_arms_mid_base" => BodyPlan { arm: Tier::Budget, head: Tier::Budget, ..BodyPlan::uniform(Tier::Mid) },
            // Strong shoulders on mid legs: more payload, but the ankles must follow.
            "pro_arms" => BodyPlan { arm: Tier::Pro, ..BodyPlan::uniform(Tier::Mid) },
            "battery_low" => BodyPlan { battery_low: true, ..BodyPlan::uniform(Tier::Mid) },
            _ => return None,
        })
    }

    pub const NAMED: [&'static str; 6] = ["budget", "mid", "pro", "budget_arms_mid_base", "pro_arms", "battery_low"];

    pub fn head_preset(&self) -> HeadPreset {
        HeadPreset::of(self.head)
    }
    pub fn chest_preset(&self) -> ChestPreset {
        ChestPreset::of(self.chest)
    }
    pub fn belly_preset(&self) -> BellyPreset {
        BellyPreset::of(self.belly)
    }
    pub fn arm_preset(&self) -> ArmPreset {
        ArmPreset::of(self.arm)
    }
    pub fn leg_preset(&self) -> LegPreset {
        LegPreset::of(self.leg)
    }
    pub fn chassis_preset(&self) -> ChassisPreset {
        ChassisPreset::of(self.leg)
    }

    /// Approximate parts cost, US dollars, for two arms and two legs (or one chassis).
    pub fn cost_usd(&self, base: BaseKind) -> f32 {
        let lower = if base == BaseKind::HumanFoot { 2.0 * self.leg_preset().cost_usd } else { self.chassis_preset().cost_usd };
        self.head_preset().cost_usd + self.chest_preset().cost_usd + self.belly_preset().cost_usd + 2.0 * self.arm_preset().cost_usd + lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_trade_torque_for_mass_and_money() {
        let (b, m, p) = (ArmPreset::of(Tier::Budget), ArmPreset::of(Tier::Mid), ArmPreset::of(Tier::Pro));
        assert!(b.mass_kg < m.mass_kg && m.mass_kg < p.mass_kg);
        assert!(b.shoulder.peak_nm < m.shoulder.peak_nm && m.shoulder.peak_nm < p.shoulder.peak_nm);
        // Bus servos are not weak on static torque per kilogram, only on continuous torque and power.
        let per_kg = |a: Actuator| a.peak_nm / a.mass_kg;
        assert!(per_kg(STS3215) > per_kg(RS02), "{} vs {}", per_kg(STS3215), per_kg(RS02));
        assert!(STS3215.cont_nm / STS3215.peak_nm < 0.4);
        assert!(BodyPlan::uniform(Tier::Budget).cost_usd(BaseKind::HumanFoot) < BodyPlan::uniform(Tier::Pro).cost_usd(BaseKind::HumanFoot));
    }

    #[test]
    fn plans_have_stable_ids_and_named_presets_parse() {
        assert_eq!(BodyPlan::default().id(), "mid");
        for n in BodyPlan::NAMED {
            let p = BodyPlan::named(n).unwrap();
            assert!(!p.id().is_empty());
        }
        assert_eq!(BodyPlan::named("battery_low").unwrap().id(), "h:mid c:mid b:mid a:mid l:mid bat:low");
        let json = serde_json::to_string(&BodyPlan::named("pro_arms").unwrap()).unwrap();
        assert!(json.contains("\"arm\":\"pro\""));
        let back: BodyPlan = serde_json::from_str(r#"{"head":"mid","chest":"mid","belly":"mid","arm":"mid","leg":"budget_plus"}"#).unwrap();
        assert_eq!(back.leg, Tier::BudgetPlus);
        assert!(!back.battery_low, "the battery position defaults to the chest");
    }

    #[test]
    fn torque_falls_with_speed() {
        assert_eq!(RS02.torque_at(0.0), RS02.peak_nm);
        assert!((RS02.torque_at(RS02.speed_rad_s / 2.0) - RS02.peak_nm / 2.0).abs() < 1e-4);
        assert_eq!(RS02.torque_at(1e6), 0.0);
    }
}
