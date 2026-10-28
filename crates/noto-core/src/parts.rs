use serde::{Deserialize, Serialize};

use crate::base::BaseKind;
use crate::hand::FingerCount;
use crate::morph::{HeadMount, Morphology, Torso};
use crate::wear::Reducer;

pub const HIP_M: f32 = 0.58;
pub const WAIST_M: f32 = 0.70;
pub const SHOULDER_M: f32 = 1.00;
pub const SHOULDER_HALF_WIDTH_M: f32 = 0.18;
pub const UPPER_ARM_M: f32 = 0.22;
pub const FOREARM_M: f32 = 0.20;
pub const STANCE_WIDTH_M: f32 = 0.18;
pub const CHASSIS_WIDTH_M: f32 = 0.40;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Budget,
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

    fn no_plus(self) -> Tier {
        if self == Tier::BudgetPlus {
            Tier::Budget
        } else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Actuator {
    pub name: &'static str,
    pub peak_nm: f32,
    pub cont_nm: f32,
    pub speed_rad_s: f32,
    pub mass_kg: f32,
    pub price_usd: f32,
    pub backdrivable: bool,
    pub thermal_tau_s: f32,
    pub reducer: Reducer,
    pub src: &'static str,
}

impl Actuator {
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

    pub fn torque_at(&self, omega: f32) -> f32 {
        self.peak_nm * (1.0 - (omega.abs() / self.speed_rad_s.max(1e-3))).clamp(0.0, 1.0)
    }
}

pub const STS3215: Actuator = Actuator { name: "Feetech STS3215 (12 V)", peak_nm: 2.94, cont_nm: 0.98, speed_rad_s: 4.7, mass_kg: 0.055, price_usd: 32.0, backdrivable: false, thermal_tau_s: 200.0, reducer: Reducer::Spur, src: "1" };
pub const HTD45H: Actuator = Actuator { name: "Hiwonder HTD-45H", peak_nm: 4.41, cont_nm: 1.5, speed_rad_s: 5.8, mass_kg: 0.064, price_usd: 30.0, backdrivable: false, thermal_tau_s: 200.0, reducer: Reducer::Spur, src: "4" };
pub const XM430: Actuator = Actuator { name: "Dynamixel XM430-W350", peak_nm: 4.1, cont_nm: 0.82, speed_rad_s: 4.8, mass_kg: 0.082, price_usd: 310.0, backdrivable: false, thermal_tau_s: 250.0, reducer: Reducer::Spur, src: "7,10" };
pub const RS01: Actuator = Actuator { name: "Robstride 01", peak_nm: 17.0, cont_nm: 6.0, speed_rad_s: 33.0, mass_kg: 0.38, price_usd: 130.0, backdrivable: true, thermal_tau_s: 600.0, reducer: Reducer::Planetary, src: "15" };
pub const RS02: Actuator = Actuator { name: "Robstride 02", peak_nm: 17.0, cont_nm: 6.0, speed_rad_s: 42.9, mass_kg: 0.405, price_usd: 145.0, backdrivable: true, thermal_tau_s: 600.0, reducer: Reducer::Planetary, src: "15" };
pub const RS03: Actuator = Actuator { name: "Robstride 03", peak_nm: 60.0, cont_nm: 20.0, speed_rad_s: 20.4, mass_kg: 0.88, price_usd: 225.0, backdrivable: true, thermal_tau_s: 800.0, reducer: Reducer::Planetary, src: "16" };
pub const RS04: Actuator = Actuator { name: "Robstride 04", peak_nm: 120.0, cont_nm: 40.0, speed_rad_s: 20.9, mass_kg: 1.42, price_usd: 255.0, backdrivable: true, thermal_tau_s: 900.0, reducer: Reducer::Planetary, src: "17" };
pub const BHL6512: Actuator = Actuator { name: "BHL 6512 (자작 사이클로이드)", peak_nm: 20.0, cont_nm: 7.0, speed_rad_s: 15.0, mass_kg: 0.55, price_usd: 170.0, backdrivable: true, thermal_tau_s: 500.0, reducer: Reducer::Cycloid, src: "22" };

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
    pub com_up_m: f32,
    pub neck: Actuator,
    pub sensor: &'static str,
    pub gaze_stabilization: f32,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChestPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    pub battery_kg: f32,
    pub battery_wh: f32,
    pub com_up_m: f32,
    pub computer: &'static str,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BellyPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    pub com_down_m: f32,
    pub waist: Actuator,
    pub pitch: Option<Actuator>,
    pub max_lean_deg: f32,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ArmPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    pub com_m: f32,
    pub shoulder: Actuator,
    pub elbow: Actuator,
    pub power_w: f32,
    pub cost_usd: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LegPreset {
    pub tier: Tier,
    pub mass_kg: f32,
    pub com_down_m: f32,
    pub knee: Actuator,
    pub ankle: Actuator,
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
    pub com_m: f32,
    pub max_speed_mps: f32,
    pub drive_force_n: f32,
    pub power_w: f32,
    pub cost_usd: f32,
    pub src: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LiftPreset {
    pub name: &'static str,
    pub stroke_m: f32,
    pub speed_mps: f32,
    pub force_n: f32,
    pub mass_kg: f32,
    pub power_w: f32,
    pub cost_usd: f32,
    pub src: &'static str,
}

pub const TORSO_LIFT: LiftPreset = LiftPreset { name: "몸통 리프트 (Fetch 급 0.40 m · 450 N)", stroke_m: 0.40, speed_mps: 0.10, force_n: 450.0, mass_kg: 2.5, power_w: 30.0, cost_usd: 600.0, src: "Fetch 사양 · 질량·가격 추정" };

pub const fn lift_mass_kg(stroke_m: f32) -> f32 {
    1.0 + 1.4 * stroke_m + 0.9
}

pub const MAST_LIFT: LiftPreset = LiftPreset { name: "마스트 리프트 (Stretch 급)", stroke_m: 0.84, speed_mps: 0.10, force_n: 450.0, mass_kg: lift_mass_kg(0.84), power_w: 30.0, cost_usd: 600.0, src: "행정: Stretch RE1 리프트 · 속도·힘·질량·가격 추정" };

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
            Tier::Budget => ChassisPreset { tier: Tier::Budget, name: "LeKiwi 급 (STS3215 ×3)", mass_kg: 1.7, com_m: 0.06, max_speed_mps: 0.24, drive_force_n: 46.0, power_w: 30.0, cost_usd: 226.0, src: "30" },
            Tier::Mid => ChassisPreset { tier: Tier::Mid, name: "허브 모터 4륜 (추정)", mass_kg: 8.0, com_m: 0.10, max_speed_mps: 1.5, drive_force_n: 240.0, power_w: 120.0, cost_usd: 900.0, src: "est" },
            _ => ChassisPreset { tier: Tier::Pro, name: "AgileX Tracer 급", mass_kg: 30.0, com_m: 0.12, max_speed_mps: 1.6, drive_force_n: 170.0, power_w: 300.0, cost_usd: 7590.0, src: "33" },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GripperPreset {
    pub tier: Tier,
    pub name: &'static str,
    pub mass_kg: f32,
    pub peak_n: Option<f32>,
    pub cont_n: Option<f32>,
    pub src: &'static str,
}

impl GripperPreset {
    pub fn of(tier: Tier) -> GripperPreset {
        match tier.no_plus() {
            Tier::Budget => GripperPreset { tier: Tier::Budget, name: "SO-101 집게 (STS3215 ×1)", mass_kg: 0.13, peak_n: Some(21.0), cont_n: Some(7.0), src: "29 · 힘 추정" },
            Tier::Mid => GripperPreset { tier: Tier::Mid, name: "2지 XM430 집게 (추정)", mass_kg: 0.35, peak_n: None, cont_n: None, src: "est" },
            _ => GripperPreset { tier: Tier::Pro, name: "Robotiq 2F-85", mass_kg: 0.9, peak_n: Some(235.0), cont_n: Some(20.0), src: "37" },
        }
    }
}

pub fn hand_mass_kg(fingers: FingerCount) -> f32 {
    match fingers {
        FingerCount::Two => 0.35,
        FingerCount::Three => 0.40,
        FingerCount::Four => 0.55,
        FingerCount::Five => 0.60,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyPlan {
    pub head: Tier,
    pub chest: Tier,
    pub belly: Tier,
    pub arm: Tier,
    pub leg: Tier,
    #[serde(default)]
    pub battery_low: bool,
    #[serde(default)]
    pub lift: bool,
    #[serde(default)]
    pub morph: Morphology,
}

impl Default for BodyPlan {
    fn default() -> Self {
        BodyPlan::uniform(Tier::Mid)
    }
}

impl BodyPlan {
    pub fn uniform(tier: Tier) -> BodyPlan {
        BodyPlan { head: tier, chest: tier, belly: tier, arm: tier, leg: tier, battery_low: false, lift: false, morph: Morphology::default() }
    }

    pub fn id(&self) -> String {
        match self.morph.upper_key() {
            Some(k) => format!("{}+{k}", self.tier_id()),
            None => self.tier_id(),
        }
    }

    pub fn validate(&self, base: BaseKind) -> Result<(), String> {
        let mut v = self.morph.issues(base);
        let body = crate::balance::Body::new(self, base, FingerCount::Five);
        if let Some(c) = body.cargo_geom().filter(|c| body.has_manipulator() && !c.on_pelvis) {
            if crate::act::grasp::stow_pose(&body, false, 0.0).is_none() {
                v.push(format!("팔이 {}에 닿지 않음 (내려놓을 점 {:.2} m): 집은 물건을 직접 실을 수 없음 — 리프트를 달거나 배 앞 적재함을 쓸 것", c.cargo.label_ko(), c.release[1]));
            }
        }
        if v.is_empty() {
            Ok(())
        } else {
            Err(v.join("; "))
        }
    }

    fn tier_id(&self) -> String {
        let u = [self.head, self.chest, self.belly, self.arm].iter().all(|t| *t == self.head) && self.leg == self.head;
        if u && !self.battery_low && !self.lift {
            return self.head.id().to_string();
        }
        format!(
            "h:{} c:{} b:{} a:{} l:{}{}{}",
            self.head.id(),
            self.chest.id(),
            self.belly.id(),
            self.arm.id(),
            self.leg.id(),
            if self.battery_low { " bat:low" } else { "" },
            if self.lift { " lift" } else { "" }
        )
    }

    pub fn named(id: &str) -> Option<BodyPlan> {
        Some(match id {
            "budget" => BodyPlan::uniform(Tier::Budget),
            "mid" => BodyPlan::uniform(Tier::Mid),
            "pro" => BodyPlan::uniform(Tier::Pro),
            "budget_arms_mid_base" => BodyPlan { arm: Tier::Budget, head: Tier::Budget, ..BodyPlan::uniform(Tier::Mid) },
            "pro_arms" => BodyPlan { arm: Tier::Pro, ..BodyPlan::uniform(Tier::Mid) },
            "battery_low" => BodyPlan { battery_low: true, ..BodyPlan::uniform(Tier::Mid) },
            "wheel_lift" => BodyPlan { lift: true, belly: Tier::Pro, ..BodyPlan::uniform(Tier::Mid) },
            _ => return None,
        })
    }

    pub const NAMED: [&'static str; 7] = ["budget", "mid", "pro", "budget_arms_mid_base", "pro_arms", "battery_low", "wheel_lift"];

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
    pub fn lift_preset(&self, base: BaseKind) -> Option<LiftPreset> {
        if base == BaseKind::HumanFoot {
            return None;
        }
        if let Some((_, stroke_m)) = self.morph.mast_lift_m() {
            return Some(LiftPreset { stroke_m, mass_kg: lift_mass_kg(stroke_m), ..MAST_LIFT });
        }
        (self.lift && self.morph.torso != Torso::None).then_some(TORSO_LIFT)
    }

    pub fn gripper_preset(&self) -> GripperPreset {
        GripperPreset::of(self.arm)
    }

    pub fn cost_usd(&self, base: BaseKind) -> f32 {
        let m = self.morph.fit(base);
        let lower = if base == BaseKind::HumanFoot { 2.0 * self.leg_preset().cost_usd } else { self.chassis_preset().cost_usd + BodyPlan { morph: m, ..*self }.lift_preset(base).map_or(0.0, |l| l.cost_usd) };
        let h = self.head_preset();
        let head = match m.head {
            HeadMount::Neck | HeadMount::MastCamera { pan_tilt: true, .. } => h.cost_usd,
            HeadMount::Fixed { .. } | HeadMount::MastCamera { .. } | HeadMount::ChassisFront { .. } => h.cost_usd - 2.0 * h.neck.price_usd,
            HeadMount::None => 0.0,
        };
        let belly = if m.torso == Torso::Humanoid { self.belly_preset().cost_usd } else { 0.0 };
        head + self.chest_preset().cost_usd + belly + m.arm_count() as f32 * self.arm_preset().cost_usd + lower
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
        assert_eq!(back.morph, Morphology::default(), "a plan without a form is today's humanoid");
    }

    #[test]
    fn a_form_suffixes_the_id_and_changes_the_cost() {
        let truck = BodyPlan { morph: Morphology::named("truck").unwrap(), ..BodyPlan::default() };
        assert_eq!(truck.id(), "mid+truck");
        let lifted = BodyPlan { lift: true, morph: Morphology::named("mast_gripper").unwrap(), ..BodyPlan::default() };
        assert_eq!(lifted.id(), "h:mid c:mid b:mid a:mid l:mid lift+mast_gripper");
        for f in ["legs_humanoid", "wheel_humanoid", "track_humanoid"] {
            assert_eq!(BodyPlan { morph: Morphology::named(f).unwrap(), ..BodyPlan::default() }.id(), "mid");
        }
        assert!(truck.cost_usd(BaseKind::Wheel) < BodyPlan::default().cost_usd(BaseKind::Wheel));
        let deck = BodyPlan { lift: true, morph: Morphology::named("deck_gripper").unwrap(), ..BodyPlan::default() };
        assert!(deck.lift_preset(BaseKind::Wheel).is_none());
        let mast = BodyPlan { morph: Morphology::named("mast_gripper").unwrap(), ..BodyPlan::default() };
        assert_eq!(mast.lift_preset(BaseKind::Wheel).map(|l| l.name), Some(MAST_LIFT.name));
        assert_eq!(lifted.lift_preset(BaseKind::Wheel), mast.lift_preset(BaseKind::Wheel));
        assert!(mast.lift_preset(BaseKind::HumanFoot).is_none());
        let json = serde_json::to_string(&truck).unwrap();
        let back: BodyPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(back, truck);
        assert_eq!(GripperPreset::of(Tier::Budget).mass_kg, 0.13);
    }

    #[test]
    fn torque_falls_with_speed() {
        assert_eq!(RS02.torque_at(0.0), RS02.peak_nm);
        assert!((RS02.torque_at(RS02.speed_rad_s / 2.0) - RS02.peak_nm / 2.0).abs() < 1e-4);
        assert_eq!(RS02.torque_at(1e6), 0.0);
    }
}
