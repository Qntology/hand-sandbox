use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::balance::{Body, Posture, MAST_KG};
use crate::base::BaseKind;
use crate::dims::Dims;
use crate::dynamics::LinkBox;
use crate::hand::{FingerCount, Hand, NotoConfig};
use crate::kin::KinPose;
use crate::parts::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Lower {
    Legs,
    #[default]
    Wheels,
    Mecanum,
    Track,
}

impl Lower {
    pub const ALL: [Lower; 4] = [Lower::Legs, Lower::Wheels, Lower::Mecanum, Lower::Track];

    pub fn from_base(base: BaseKind) -> Lower {
        match base {
            BaseKind::HumanFoot => Lower::Legs,
            BaseKind::Wheel => Lower::Wheels,
            BaseKind::Mecanum => Lower::Mecanum,
            BaseKind::Track => Lower::Track,
        }
    }

    pub fn base(self) -> BaseKind {
        match self {
            Lower::Legs => BaseKind::HumanFoot,
            Lower::Wheels => BaseKind::Wheel,
            Lower::Mecanum => BaseKind::Mecanum,
            Lower::Track => BaseKind::Track,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Lower::Legs => "legs",
            Lower::Wheels => "wheels",
            Lower::Mecanum => "mecanum",
            Lower::Track => "track",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Lower::Legs => "다리",
            Lower::Wheels => "바퀴",
            Lower::Mecanum => "메카넘 바퀴",
            Lower::Track => "트랙",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Torso {
    #[default]
    Humanoid,
    Mast,
    None,
}

impl Torso {
    pub const ALL: [Torso; 3] = [Torso::Humanoid, Torso::Mast, Torso::None];

    pub fn id(self) -> &'static str {
        match self {
            Torso::Humanoid => "humanoid",
            Torso::Mast => "mast",
            Torso::None => "none",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Torso::Humanoid => "휴머노이드 몸통 (골반·허리·가슴)",
            Torso::Mast => "마스트 기둥",
            Torso::None => "몸통 없음 (갑판)",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadMount {
    #[default]
    Neck,
    Fixed { tilt_deg: i16 },
    MastCamera { height_mm: u16, pan_tilt: bool },
    ChassisFront { height_mm: u16 },
    None,
}

impl HeadMount {
    pub fn id(self) -> &'static str {
        match self {
            HeadMount::Neck => "neck",
            HeadMount::Fixed { .. } => "fixed",
            HeadMount::MastCamera { .. } => "mast_camera",
            HeadMount::ChassisFront { .. } => "chassis_front",
            HeadMount::None => "none",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            HeadMount::Neck => "목 (팬·틸트) 위 머리",
            HeadMount::Fixed { .. } => "몸통에 고정된 머리 (목 없음)",
            HeadMount::MastCamera { pan_tilt: true, .. } => "마스트 카메라 (팬·틸트)",
            HeadMount::MastCamera { .. } => "마스트 카메라 (고정)",
            HeadMount::ChassisFront { .. } => "차체 앞 카메라",
            HeadMount::None => "카메라 없음",
        }
    }

    pub fn pans_and_tilts(self) -> bool {
        matches!(self, HeadMount::Neck | HeadMount::MastCamera { pan_tilt: true, .. })
    }

    pub fn on_trunk(self) -> bool {
        matches!(self, HeadMount::Neck | HeadMount::Fixed { .. })
    }

    fn code(self) -> String {
        match self {
            HeadMount::Neck => "neck".into(),
            HeadMount::Fixed { tilt_deg } => format!("fixed{tilt_deg}"),
            HeadMount::MastCamera { height_mm, pan_tilt } => format!("mast{height_mm}{}", if pan_tilt { "pt" } else { "" }),
            HeadMount::ChassisFront { height_mm } => format!("front{height_mm}"),
            HeadMount::None => "none".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Manip {
    #[default]
    Arms2,
    Arm1,
    Gripper1,
    None,
}

impl Manip {
    pub const ALL: [Manip; 4] = [Manip::Arms2, Manip::Arm1, Manip::Gripper1, Manip::None];

    pub fn id(self) -> &'static str {
        match self {
            Manip::Arms2 => "arms2",
            Manip::Arm1 => "arm1",
            Manip::Gripper1 => "gripper1",
            Manip::None => "none",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Manip::Arms2 => "두 팔 (손)",
            Manip::Arm1 => "한 팔 (손)",
            Manip::Gripper1 => "집게 팔 하나",
            Manip::None => "조작기 없음",
        }
    }

    pub fn arm_count(self) -> usize {
        match self {
            Manip::Arms2 => 2,
            Manip::Arm1 | Manip::Gripper1 => 1,
            Manip::None => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cargo {
    #[default]
    Bay,
    Bed { len_mm: u16, width_mm: u16, height_mm: u16, payload_kg: u16 },
    Tote,
    None,
}

impl Cargo {
    pub fn id(self) -> &'static str {
        match self {
            Cargo::Bay => "bay",
            Cargo::Bed { .. } => "bed",
            Cargo::Tote => "tote",
            Cargo::None => "none",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Cargo::Bay => "배 앞 적재함",
            Cargo::Bed { .. } => "짐칸 (적재 베드)",
            Cargo::Tote => "갑판 토트",
            Cargo::None => "적재 공간 없음",
        }
    }

    fn code(self) -> String {
        match self {
            Cargo::Bed { len_mm, width_mm, height_mm, payload_kg } => format!("bed{len_mm}x{width_mm}x{height_mm}/{payload_kg}"),
            c => c.id().into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Morphology {
    #[serde(default)]
    pub lower: Lower,
    #[serde(default)]
    pub torso: Torso,
    #[serde(default)]
    pub head: HeadMount,
    #[serde(default)]
    pub manip: Manip,
    #[serde(default)]
    pub cargo: Cargo,
}

/// The compact chassis deck the small forms mount on (`Dims::deck_z_m` for a body's own).
pub const DECK_Z_M: f32 = 0.16;
pub const DECK_BOX_KG: f32 = 0.30;
pub const DECK_ARM_BASE_HALF_M: f32 = 0.03;
pub const DECK_ARM_X_M: f32 = -0.06;
pub const DECK_ARM_BASE_H_M: f32 = 0.10;
pub const DECK_ARMS_HALF_SPACING_M: f32 = 0.10;
pub const DECK_REST_DEG: [f32; 2] = [120.0, 120.0];
pub const DECK_LIFT_DEG: [f32; 2] = [70.0, 30.0];
pub const EXTRUSION_KG_PER_M: f32 = 0.50;
pub const POLE_MOUNT_KG: f32 = 0.20;
pub const CHASSIS_CAMERA_KG: f32 = 0.10;
pub const BED_KG_PER_M2: f32 = 8.1;
pub const CARGO_ABOVE_FLOOR_M: f32 = 0.10;
pub const TOTE_SIZE_M: [f32; 3] = [0.16, 0.24, 0.12];
pub const TOTE_KG: f32 = 0.4;
pub const TOTE_X_M: f32 = 0.06;
pub const RIM_RELEASE_M: f32 = 0.20;
pub const MAST_TILT_MAX_DEG: f32 = 83.0;
/// A pan-tilt head's pan each way (a robot unit, not a neck).
pub const MAST_PAN_MAX_DEG: f32 = 90.0;
pub const STRETCH_CAMERA_H_MM: u16 = 1310;
pub const DECK_CAMERA_H_MM: u16 = 200;
pub const AMR_CAMERA_H_MM: u16 = 250;
pub const FIXED_HEAD_TILT_DEG: i16 = 15;
pub const TRUCK_BED: Cargo = Cargo::Bed { len_mm: 500, width_mm: 400, height_mm: 250, payload_kg: 100 };

pub const MAST_LIFT_TOP_M: f32 = 1.10;
pub const MAST_CAMERA_CLEAR_M: f32 = 0.15;
pub const MAST_ARM_X_M: f32 = 0.05;

pub const FIXED_TILT_RANGE_DEG: [i16; 2] = [-45, 60];
pub const MAST_CAMERA_MAX_MM: u16 = 2000;
pub const CHASSIS_CAMERA_RANGE_MM: [u16; 2] = [50, 600];
pub const BED_MAX_MM: [u16; 3] = [1000, 800, 800];

pub fn pole_x_m(support_len_m: f32) -> f32 {
    -0.5 * support_len_m + 0.02
}

pub fn column_top_m(d: &Dims, head: HeadMount) -> f32 {
    match head {
        HeadMount::MastCamera { height_mm, .. } => d.column_top_m().max(height_mm as f32 / 1000.0 - d.camera_up_m),
        _ => d.column_top_m(),
    }
}

const HUMANOID_UPPER: (Torso, HeadMount, Manip, Cargo) = (Torso::Humanoid, HeadMount::Neck, Manip::Arms2, Cargo::Bay);

pub struct Form {
    pub id: &'static str,
    pub label_ko: &'static str,
    pub key: Option<&'static str>,
    pub morph: Morphology,
    pub analogue: &'static str,
}

const fn m(lower: Lower, torso: Torso, head: HeadMount, manip: Manip, cargo: Cargo) -> Morphology {
    Morphology { lower, torso, head, manip, cargo }
}

pub const FORMS: [Form; 7] = [
    Form { id: "legs_humanoid", label_ko: "두 다리 휴머노이드", key: None, morph: m(Lower::Legs, Torso::Humanoid, HeadMount::Neck, Manip::Arms2, Cargo::Bay), analogue: "이족 휴머노이드 (Unitree G1·K-Scale K-Bot 급, parts.md §2)" },
    Form { id: "wheel_humanoid", label_ko: "바퀴 휴머노이드", key: None, morph: m(Lower::Wheels, Torso::Humanoid, HeadMount::Neck, Manip::Arms2, Cargo::Bay), analogue: "바퀴 위 상체 (오늘의 기본 로봇)" },
    Form { id: "wheel_noneck", label_ko: "바퀴 휴머노이드 (목 없음)", key: Some("noneck"), morph: m(Lower::Wheels, Torso::Humanoid, HeadMount::Fixed { tilt_deg: FIXED_HEAD_TILT_DEG }, Manip::Arms2, Cargo::Bay), analogue: "목 자유도 없는 머리 (Unitree H1, 미확인)" },
    Form { id: "mast_gripper", label_ko: "마스트 카메라 + 집게 팔 (Stretch 형)", key: Some("mast_gripper"), morph: m(Lower::Wheels, Torso::None, HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: true }, Manip::Gripper1, Cargo::Tote), analogue: "Hello Robot Stretch RE1 (마스트 카메라 1.31 m, 팔 리프트 0–1.10 m, 23 kg; Kemp 2022)" },
    Form { id: "deck_gripper", label_ko: "갑판 집게 팔 (LeKiwi 형)", key: Some("deck_gripper"), morph: m(Lower::Wheels, Torso::None, HeadMount::ChassisFront { height_mm: DECK_CAMERA_H_MM }, Manip::Gripper1, Cargo::Tote), analogue: "LeKiwi 베이스 + SO-101 팔 (parts.md [29][30])" },
    Form { id: "truck", label_ko: "운반 트럭 (조작기 없음)", key: Some("truck"), morph: m(Lower::Wheels, Torso::None, HeadMount::ChassisFront { height_mm: AMR_CAMERA_H_MM }, Manip::None, TRUCK_BED), analogue: "AMR·카트 운반 로봇 (MiR250, AgileX Tracer; Amazon Proteus는 수치 없음)" },
    Form { id: "track_humanoid", label_ko: "트랙 휴머노이드", key: None, morph: m(Lower::Track, Torso::Humanoid, HeadMount::Neck, Manip::Arms2, Cargo::Bay), analogue: "트랙 위 상체 (오늘의 트랙 로봇)" },
];

impl Morphology {
    pub fn humanoid(base: BaseKind) -> Morphology {
        Morphology { lower: Lower::from_base(base), ..Morphology::default() }
    }

    pub const NAMED: [&'static str; 7] = ["legs_humanoid", "wheel_humanoid", "wheel_noneck", "mast_gripper", "deck_gripper", "truck", "track_humanoid"];

    pub fn named(id: &str) -> Option<Morphology> {
        FORMS.iter().find(|f| f.id == id).map(|f| f.morph)
    }

    pub fn form(&self) -> Option<&'static Form> {
        let wheels = Morphology { lower: if self.lower == Lower::Mecanum { Lower::Wheels } else { self.lower }, ..*self };
        FORMS
            .iter()
            .find(|f| f.morph == *self)
            .or_else(|| FORMS.iter().find(|f| f.morph == wheels))
            .or_else(|| FORMS.iter().find(|f| f.morph.upper() == self.upper() && f.key.is_some()))
    }

    fn upper(&self) -> (Torso, HeadMount, Manip, Cargo) {
        (self.torso, self.head, self.manip, self.cargo)
    }

    pub fn humanoid_upper(&self) -> bool {
        self.upper() == HUMANOID_UPPER
    }

    pub fn upper_key(&self) -> Option<String> {
        if self.humanoid_upper() {
            return None;
        }
        if let Some(k) = FORMS.iter().find(|f| f.morph.upper() == self.upper()).and_then(|f| f.key) {
            return Some(k.to_string());
        }
        Some(format!("t:{} h:{} m:{} c:{}", self.torso.id(), self.head.code(), self.manip.id(), self.cargo.code()))
    }

    pub fn label_ko(&self) -> String {
        if let Some(f) = FORMS.iter().find(|f| f.morph == *self) {
            return f.label_ko.to_string();
        }
        if self.humanoid_upper() {
            return format!("{} 휴머노이드", self.lower.label_ko());
        }
        match self.form() {
            Some(f) => format!("{} · {}", f.label_ko, self.lower.label_ko()),
            None => format!("{} · {} · {} · {} · {}", self.lower.label_ko(), self.torso.label_ko(), self.head.label_ko(), self.manip.label_ko(), self.cargo.label_ko()),
        }
    }

    pub fn legged(&self) -> bool {
        self.lower == Lower::Legs
    }

    pub fn has_torso(&self) -> bool {
        self.torso == Torso::Humanoid
    }

    pub fn has_neck(&self) -> bool {
        self.head.pans_and_tilts()
    }

    pub fn arm_count(&self) -> usize {
        self.manip.arm_count()
    }

    pub fn mast_arms(&self) -> bool {
        self.torso == Torso::None && matches!(self.head, HeadMount::MastCamera { .. }) && self.arm_count() > 0
    }

    pub fn mast_lift_m(&self) -> Option<(f32, f32)> {
        let HeadMount::MastCamera { height_mm, .. } = self.head else { return None };
        if !self.mast_arms() {
            return None;
        }
        let bottom = DECK_Z_M + DECK_ARM_BASE_H_M;
        let top = MAST_LIFT_TOP_M.min(height_mm as f32 / 1000.0 - MAST_CAMERA_CLEAR_M).max(bottom);
        Some((top, top - bottom))
    }

    pub fn issues(&self, base: BaseKind) -> Vec<String> {
        let mut v = Vec::new();
        if self.lower != Lower::from_base(base) {
            v.push(format!("하체 '{}'가 이동 베이스 '{:?}'와 다름", self.lower.label_ko(), base));
        }
        let legs = Lower::from_base(base) == Lower::Legs;
        if legs && self.torso != Torso::Humanoid {
            v.push("두 다리는 골반이 있는 휴머노이드 몸통이 필요함 (균형 모델이 골반 위 고관절을 씀)".into());
        }
        match self.head {
            HeadMount::Neck | HeadMount::Fixed { .. } if self.torso == Torso::None => v.push("몸통이 없으면 머리를 얹을 곳이 없음 (마스트 카메라나 차체 카메라를 쓸 것)".into()),
            HeadMount::MastCamera { .. } | HeadMount::ChassisFront { .. } if legs => v.push("마스트·차체 카메라는 바퀴·트랙 차체에만 달 수 있음".into()),
            HeadMount::MastCamera { .. } if self.torso == Torso::Humanoid => v.push("마스트 카메라는 몸통 없는 갑판이나 마스트 기둥에 달 것 (휴머노이드 몸통에는 목)".into()),
            HeadMount::MastCamera { height_mm, .. } if height_mm < min_mast_camera_mm() => v.push(format!("마스트 카메라 높이 {height_mm} mm가 갑판에 너무 가까움")),
            HeadMount::MastCamera { height_mm, .. } if height_mm > MAST_CAMERA_MAX_MM => v.push(format!("마스트 카메라 높이 {height_mm} mm > {MAST_CAMERA_MAX_MM} mm")),
            HeadMount::Fixed { tilt_deg } if !(FIXED_TILT_RANGE_DEG[0]..=FIXED_TILT_RANGE_DEG[1]).contains(&tilt_deg) => v.push(format!("고정 머리 기울기 {tilt_deg}°가 범위 {}…{}° 밖", FIXED_TILT_RANGE_DEG[0], FIXED_TILT_RANGE_DEG[1])),
            HeadMount::ChassisFront { height_mm } if !(CHASSIS_CAMERA_RANGE_MM[0]..=CHASSIS_CAMERA_RANGE_MM[1]).contains(&height_mm) => v.push(format!("차체 카메라 높이 {height_mm} mm가 범위 {}…{} mm 밖", CHASSIS_CAMERA_RANGE_MM[0], CHASSIS_CAMERA_RANGE_MM[1])),
            _ => {}
        }
        match self.cargo {
            Cargo::Bay if self.torso != Torso::Humanoid => v.push("배 앞 적재함은 골반에 달림: 몸통이 없으면 토트나 짐칸을 쓸 것".into()),
            Cargo::Bed { .. } | Cargo::Tote if legs => v.push("짐칸·토트는 바퀴·트랙 차체에만 얹을 수 있음".into()),
            Cargo::Bed { len_mm, width_mm, height_mm, payload_kg } if len_mm == 0 || width_mm == 0 || payload_kg == 0 || height_mm < deck_mm() => v.push("짐칸의 크기·높이·정격 하중이 맞지 않음".into()),
            Cargo::Bed { len_mm, width_mm, height_mm, .. } if len_mm > BED_MAX_MM[0] || width_mm > BED_MAX_MM[1] || height_mm > BED_MAX_MM[2] => {
                v.push(format!("짐칸 {len_mm}×{width_mm} mm, 높이 {height_mm} mm가 한도 {}×{} mm, {} mm를 넘음", BED_MAX_MM[0], BED_MAX_MM[1], BED_MAX_MM[2]))
            }
            _ => {}
        }
        v
    }

    pub fn validate(&self, base: BaseKind) -> Result<(), String> {
        let v = self.issues(base);
        if v.is_empty() {
            Ok(())
        } else {
            Err(v.join("; "))
        }
    }

    pub fn fit(self, base: BaseKind) -> Morphology {
        let mut m = self;
        m.lower = Lower::from_base(base);
        let legs = m.lower == Lower::Legs;
        if legs {
            m.torso = Torso::Humanoid;
        }
        m.head = match m.head {
            HeadMount::Neck if m.torso == Torso::None => HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: true },
            HeadMount::Fixed { .. } if m.torso == Torso::None => HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: false },
            HeadMount::MastCamera { .. } | HeadMount::ChassisFront { .. } if legs => HeadMount::Neck,
            HeadMount::MastCamera { pan_tilt, .. } if m.torso == Torso::Humanoid => if pan_tilt { HeadMount::Neck } else { HeadMount::Fixed { tilt_deg: 0 } },
            HeadMount::MastCamera { height_mm, pan_tilt } => HeadMount::MastCamera { height_mm: height_mm.clamp(min_mast_camera_mm(), MAST_CAMERA_MAX_MM), pan_tilt },
            HeadMount::Fixed { tilt_deg } => HeadMount::Fixed { tilt_deg: tilt_deg.clamp(FIXED_TILT_RANGE_DEG[0], FIXED_TILT_RANGE_DEG[1]) },
            HeadMount::ChassisFront { height_mm } => HeadMount::ChassisFront { height_mm: height_mm.clamp(CHASSIS_CAMERA_RANGE_MM[0], CHASSIS_CAMERA_RANGE_MM[1]) },
            h => h,
        };
        m.cargo = match m.cargo {
            Cargo::Bay if m.torso != Torso::Humanoid => Cargo::Tote,
            Cargo::Bed { .. } | Cargo::Tote if legs => Cargo::Bay,
            Cargo::Bed { len_mm, width_mm, height_mm, payload_kg } => Cargo::Bed {
                len_mm: len_mm.clamp(1, BED_MAX_MM[0]),
                width_mm: width_mm.clamp(1, BED_MAX_MM[1]),
                height_mm: height_mm.clamp(deck_mm(), BED_MAX_MM[2]),
                payload_kg: payload_kg.max(1),
            },
            c => c,
        };
        m
    }
}

fn deck_mm() -> u16 {
    (1000.0 * DECK_Z_M).round() as u16
}

fn min_mast_camera_mm() -> u16 {
    (1000.0 * (DECK_Z_M + 0.1)).round() as u16
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CargoGeom {
    pub cargo: Cargo,
    pub at: [f32; 3],
    pub on_pelvis: bool,
    pub release: [f32; 2],
    pub rim_z_m: f32,
    pub floor_z_m: f32,
    pub payload_kg: Option<f32>,
    pub struct_kg: f32,
    pub struct_at: [f32; 3],
    pub center: [f32; 3],
    pub half: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MountMass {
    pub name: &'static str,
    pub segment: Segment,
    pub kg: f32,
    pub at: [f32; 3],
}

impl Body {
    pub fn legged(&self) -> bool {
        self.morph.legged()
    }

    pub fn has_torso(&self) -> bool {
        self.morph.has_torso()
    }

    pub fn has_column(&self) -> bool {
        self.morph.torso == Torso::Mast
    }

    pub fn has_neck(&self) -> bool {
        self.morph.has_neck()
    }

    pub fn head_on_trunk(&self) -> bool {
        self.morph.head.on_trunk()
    }

    pub fn arm_count(&self) -> usize {
        self.morph.arm_count()
    }

    pub fn has_manipulator(&self) -> bool {
        self.arm_count() > 0
    }

    pub fn gripper_only(&self) -> bool {
        self.morph.manip == Manip::Gripper1
    }

    pub fn deck_arms(&self) -> bool {
        self.morph.torso == Torso::None && self.has_manipulator() && !self.mast_arms()
    }

    pub fn mast_arms(&self) -> bool {
        self.morph.mast_arms()
    }

    pub fn camera_mount(&self) -> HeadMount {
        self.morph.head
    }

    pub fn head_range_deg(&self) -> (f32, f32) {
        match self.morph.head {
            HeadMount::Neck => (crate::kin::NECK_PITCH_MAX_DEG, crate::kin::NECK_PAN_MAX_DEG),
            HeadMount::MastCamera { pan_tilt: true, .. } => (MAST_TILT_MAX_DEG, MAST_PAN_MAX_DEG),
            _ => (0.0, 0.0),
        }
    }

    pub fn shoulder_mount(&self, side: usize) -> [f32; 3] {
        let y = if side == 0 { -self.dims.shoulder_half_m } else { self.dims.shoulder_half_m };
        match self.morph.torso {
            Torso::Humanoid | Torso::Mast => [0.0, y, self.dims.shoulder_m],
            Torso::None => {
                let yd = if self.arm_count() >= 2 { if side == 0 { -DECK_ARMS_HALF_SPACING_M } else { DECK_ARMS_HALF_SPACING_M } } else { 0.0 };
                match self.morph.mast_lift_m() {
                    Some((top, _)) => [self.mast_camera_x() + MAST_ARM_X_M, yd, top],
                    None => [DECK_ARM_X_M, yd, DECK_Z_M + DECK_ARM_BASE_H_M],
                }
            }
        }
    }

    pub fn shoulder_at(&self, side: usize, lift_m: f32) -> [f32; 3] {
        let mut s = self.shoulder_mount(side);
        if self.mast_arms() {
            s[2] -= lift_m;
        }
        s
    }

    pub fn rest_arms_deg(&self) -> ([f32; 2], [f32; 2]) {
        if self.deck_arms() {
            ([DECK_REST_DEG[0]; 2], [DECK_REST_DEG[1]; 2])
        } else {
            ([0.0, 0.0], [10.0, 10.0])
        }
    }

    pub fn stand(&self, cargo_kg: f32) -> KinPose {
        let (s, e) = self.rest_arms_deg();
        KinPose { shoulder_deg: s, elbow_deg: e, ..KinPose::stand(self, cargo_kg) }
    }

    pub fn quiet(&self, cargo_kg: f32) -> Posture {
        let (s, e) = self.rest_arms_deg();
        Posture { arm_elev_deg: s, elbow_deg: e, ..Posture::quiet(cargo_kg) }
    }

    pub fn lift_posture(&self, load_kg: f32, two_arms: bool, cargo_kg: f32) -> Posture {
        let p = Posture::arm_lift(load_kg, two_arms, cargo_kg);
        if !self.deck_arms() {
            return p;
        }
        let (rs, re) = self.rest_arms_deg();
        let (s1, e1) = if two_arms { (DECK_LIFT_DEG[0], DECK_LIFT_DEG[1]) } else { (rs[1], re[1]) };
        Posture { arm_elev_deg: [DECK_LIFT_DEG[0], s1], elbow_deg: [DECK_LIFT_DEG[1], e1], ..p }
    }

    pub fn neck_pivot(&self) -> Option<[f32; 3]> {
        match self.morph.head {
            HeadMount::Neck => Some([0.0, 0.0, self.dims.shoulder_m + self.dims.neck_up_m]),
            HeadMount::MastCamera { height_mm, pan_tilt: true } => Some([self.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - self.dims.camera_up_m]),
            _ => None,
        }
    }

    pub fn mast_camera_x(&self) -> f32 {
        if self.morph.torso == Torso::None {
            pole_x_m(self.support_len_m)
        } else {
            0.0
        }
    }

    pub fn head_com(&self) -> Option<[f32; 3]> {
        match self.morph.head {
            HeadMount::Neck => Some([0.0, 0.0, self.dims.shoulder_m + self.dims.neck_up_m + self.head.com_up_m]),
            HeadMount::Fixed { tilt_deg } => {
                let t = (tilt_deg as f32).to_radians();
                Some([self.head.com_up_m * t.sin(), 0.0, self.dims.shoulder_m + self.dims.neck_up_m + self.head.com_up_m * t.cos()])
            }
            HeadMount::MastCamera { height_mm, .. } => Some([self.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - self.dims.camera_up_m + self.head.com_up_m]),
            HeadMount::ChassisFront { height_mm } => Some([0.5 * self.support_len_m, 0.0, height_mm as f32 / 1000.0]),
            HeadMount::None => None,
        }
    }

    pub fn deck_point(&self) -> [f32; 3] {
        [0.0, 0.0, self.dims.wheel_r_m + 0.05]
    }

    pub fn deck_boxes(&self) -> Vec<LinkBox> {
        const UP: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut v = Vec::new();
        if self.deck_arms() {
            for s in 0..self.arm_count() {
                let sh = self.shoulder_mount(s);
                let h = 0.5 * (sh[2] - DECK_Z_M);
                v.push(LinkBox { name: if s == 0 { "arm_base_r" } else { "arm_base_l" }, segment: Segment::Arm, center: [sh[0], sh[1], DECK_Z_M + h], axes: UP, half: [DECK_ARM_BASE_HALF_M, DECK_ARM_BASE_HALF_M, h] });
            }
        }
        if let Some(c) = self.cargo_geom().filter(|c| !c.on_pelvis) {
            v.push(LinkBox { name: c.cargo.id(), segment: Segment::Belly, center: c.center, axes: UP, half: c.half });
        }
        if let HeadMount::ChassisFront { height_mm } = self.morph.head {
            v.push(LinkBox { name: "camera", segment: Segment::Head, center: [0.5 * self.support_len_m - 0.02, 0.0, height_mm as f32 / 1000.0], axes: UP, half: [0.02, 0.045, 0.015] });
        }
        v
    }

    pub fn working_hand<'a>(&self, hand: &'a Hand) -> Cow<'a, Hand> {
        if !self.gripper_only() || hand.config.fingers == FingerCount::Two {
            return Cow::Borrowed(hand);
        }
        let mut cfg = NotoConfig::preset(FingerCount::Two);
        cfg.base = hand.config.base;
        cfg.build().map_or(Cow::Borrowed(hand), Cow::Owned)
    }

    pub fn column(&self) -> Option<(f32, [f32; 3])> {
        if self.legged() {
            return None;
        }
        match (self.morph.torso, self.morph.head) {
            (Torso::Humanoid, _) => Some((self.mast_kg(), [0.0, 0.0, 0.35])),
            (Torso::Mast, h) => Some((self.mast_kg(), [0.0, 0.0, 0.5 * (DECK_Z_M + column_top_m(&self.dims, h))])),
            (Torso::None, HeadMount::MastCamera { height_mm, .. }) => Some((self.mast_kg(), [self.mast_camera_x(), 0.0, 0.5 * (DECK_Z_M + height_mm as f32 / 1000.0)])),
            _ => None,
        }
    }

    pub fn cargo_geom(&self) -> Option<CargoGeom> {
        match self.morph.cargo {
            Cargo::Bay => {
                let rim = self.dims.cargo_z_m() + 0.08;
                Some(CargoGeom { cargo: Cargo::Bay, at: [self.dims.cargo_x_m, 0.0, self.dims.cargo_z_m()], on_pelvis: true, release: [self.dims.cargo_x_m + 0.12, rim + 0.03 + 0.05], rim_z_m: rim, floor_z_m: self.dims.cargo_z_m() - CARGO_ABOVE_FLOOR_M, payload_kg: None, struct_kg: 0.0, struct_at: [self.dims.cargo_x_m, 0.0, self.dims.cargo_z_m()], center: [self.dims.cargo_x_m, 0.0, self.dims.cargo_z_m()], half: [0.08, 0.12, 0.08] })
            }
            Cargo::Bed { len_mm, width_mm, height_mm, payload_kg } => {
                let (l, w, h) = (len_mm as f32 / 1000.0, width_mm as f32 / 1000.0, height_mm as f32 / 1000.0);
                Some(CargoGeom {
                    cargo: self.morph.cargo,
                    at: [0.0, 0.0, h + CARGO_ABOVE_FLOOR_M],
                    on_pelvis: false,
                    release: [0.0, h + RIM_RELEASE_M],
                    rim_z_m: h,
                    floor_z_m: h,
                    payload_kg: Some(payload_kg as f32),
                    struct_kg: BED_KG_PER_M2 * l * w,
                    struct_at: [0.0, 0.0, h],
                    center: [0.0, 0.0, h - 0.01],
                    half: [0.5 * l, 0.5 * w, 0.01],
                })
            }
            Cargo::Tote => {
                let [l, w, h] = TOTE_SIZE_M;
                let rim = DECK_Z_M + h;
                Some(CargoGeom {
                    cargo: Cargo::Tote,
                    at: [TOTE_X_M, 0.0, DECK_Z_M + 0.5 * h],
                    on_pelvis: false,
                    release: [TOTE_X_M, rim + RIM_RELEASE_M],
                    rim_z_m: rim,
                    floor_z_m: DECK_Z_M,
                    payload_kg: None,
                    struct_kg: TOTE_KG,
                    struct_at: [TOTE_X_M, 0.0, DECK_Z_M + 0.5 * h],
                    center: [TOTE_X_M, 0.0, DECK_Z_M + 0.5 * h],
                    half: [0.5 * l, 0.5 * w, 0.5 * h],
                })
            }
            Cargo::None => None,
        }
    }

    pub fn cargo_point(&self) -> ([f32; 3], bool) {
        match self.cargo_geom() {
            Some(c) => (c.at, c.on_pelvis),
            None => ([0.0, 0.0, DECK_Z_M + CARGO_ABOVE_FLOOR_M], false),
        }
    }

    pub fn mount_masses(&self) -> Vec<MountMass> {
        let mut v = Vec::new();
        if self.morph.torso != Torso::Humanoid {
            let d = self.deck_point();
            v.push(MountMass { name: "electronics", segment: Segment::Chest, kg: self.chest.mass_kg, at: d });
            v.push(MountMass { name: "battery", segment: Segment::Chest, kg: self.chest.battery_kg, at: d });
        }
        if !self.head_on_trunk() {
            if let Some(h) = self.head_com() {
                v.push(MountMass { name: "camera", segment: Segment::Head, kg: self.head.mass_kg, at: h });
            }
        }
        if let Some(c) = self.cargo_geom().filter(|c| c.struct_kg > 0.0) {
            v.push(MountMass { name: c.cargo.id(), segment: Segment::Foot, kg: c.struct_kg, at: c.struct_at });
        }
        v
    }

    pub fn form_id(&self) -> Option<&'static str> {
        self.morph.form().map(|f| f.id)
    }
}

pub fn mounted_head(mut head: HeadPreset, mount: HeadMount) -> HeadPreset {
    let fixed = head.mass_kg - 2.0 * head.neck.mass_kg;
    match mount {
        HeadMount::Neck | HeadMount::MastCamera { pan_tilt: true, .. } => {}
        HeadMount::Fixed { .. } | HeadMount::MastCamera { pan_tilt: false, .. } => head.mass_kg = fixed,
        HeadMount::ChassisFront { .. } => head.mass_kg = CHASSIS_CAMERA_KG,
        HeadMount::None => head.mass_kg = 0.0,
    }
    if !mount.pans_and_tilts() {
        head.gaze_stabilization = 0.0;
    }
    head
}

pub fn computer_kg(tier: Tier) -> f32 {
    match tier {
        Tier::Budget | Tier::BudgetPlus => 0.054,
        Tier::Mid | Tier::Pro => 0.175,
    }
}

pub fn mounted_chest(mut chest: ChestPreset, torso: Torso) -> ChestPreset {
    if torso != Torso::Humanoid {
        chest.mass_kg = computer_kg(chest.tier) + DECK_BOX_KG;
    }
    chest
}

pub fn mounted_belly(mut belly: BellyPreset, torso: Torso) -> BellyPreset {
    if torso != Torso::Humanoid {
        belly.mass_kg = 0.0;
        belly.pitch = None;
        belly.max_lean_deg = 0.0;
        belly.power_w = 0.0;
        belly.cost_usd = 0.0;
    }
    belly
}

pub fn column_kg(d: &Dims, torso: Torso, head: HeadMount) -> f32 {
    match (torso, head) {
        (Torso::Humanoid, _) => MAST_KG,
        (Torso::Mast, h) => MAST_KG + EXTRUSION_KG_PER_M * (column_top_m(d, h) - d.hip_m),
        (Torso::None, HeadMount::MastCamera { height_mm, .. }) => POLE_MOUNT_KG + EXTRUSION_KG_PER_M * (height_mm as f32 / 1000.0 - DECK_Z_M).max(0.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_todays_humanoid_and_serialises_by_layer() {
        let d = Morphology::default();
        assert!(d.humanoid_upper() && d.upper_key().is_none());
        assert_eq!(Morphology::humanoid(BaseKind::HumanFoot), Morphology::named("legs_humanoid").unwrap());
        assert_eq!(Morphology::humanoid(BaseKind::Wheel), Morphology::named("wheel_humanoid").unwrap());
        let json = serde_json::to_string(&Morphology::named("truck").unwrap()).unwrap();
        assert!(json.contains("\"torso\":\"none\"") && json.contains("\"kind\":\"bed\"") && json.contains("\"payload_kg\":100"), "{json}");
        let back: Morphology = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Morphology::named("truck").unwrap());
        let partial: Morphology = serde_json::from_str(r#"{"head":{"kind":"fixed","tilt_deg":10}}"#).unwrap();
        assert_eq!(partial.head, HeadMount::Fixed { tilt_deg: 10 });
        assert_eq!((partial.torso, partial.manip, partial.cargo), (Torso::Humanoid, Manip::Arms2, Cargo::Bay));
    }

    #[test]
    fn named_forms_are_valid_on_their_own_base_and_keyed() {
        for f in &FORMS {
            assert!(f.morph.validate(f.morph.lower.base()).is_ok(), "{}: {:?}", f.id, f.morph.issues(f.morph.lower.base()));
            assert_eq!(f.morph.fit(f.morph.lower.base()), f.morph, "{}: a valid form fits unchanged", f.id);
            assert_eq!(f.morph.form().map(|x| x.id), Some(f.id));
            assert_eq!(f.morph.upper_key().as_deref(), f.key);
        }
        assert_eq!(Morphology::NAMED.len(), FORMS.len());
        let t = Morphology { lower: Lower::Track, ..Morphology::named("truck").unwrap() };
        assert_eq!(t.form().map(|f| f.id), Some("truck"));
        assert!(t.label_ko().contains("트랙"));
    }

    #[test]
    fn fit_repairs_what_cannot_be_built() {
        let deck = Morphology { torso: Torso::None, ..Morphology::default() };
        assert!(deck.validate(BaseKind::Wheel).is_err(), "a neck and a bay need a trunk");
        let f = deck.fit(BaseKind::Wheel);
        assert!(f.validate(BaseKind::Wheel).is_ok(), "{:?}", f.issues(BaseKind::Wheel));
        assert!(matches!(f.head, HeadMount::MastCamera { pan_tilt: true, .. }) && f.cargo == Cargo::Tote);
        let truck_on_legs = Morphology::named("truck").unwrap().fit(BaseKind::HumanFoot);
        assert_eq!(truck_on_legs.lower, Lower::Legs);
        assert!(truck_on_legs.validate(BaseKind::HumanFoot).is_ok(), "{:?}", truck_on_legs.issues(BaseKind::HumanFoot));
        assert_eq!((truck_on_legs.torso, truck_on_legs.head, truck_on_legs.cargo), (Torso::Humanoid, HeadMount::Neck, Cargo::Bay));
        assert!(Morphology::default().validate(BaseKind::Track).is_err(), "the lower body must follow the drive");
    }

    #[test]
    fn a_mecanum_humanoid_is_the_wheel_humanoid_and_only_a_mast_with_an_arm_has_a_lift() {
        let m = Morphology::humanoid(BaseKind::Mecanum);
        assert_eq!(m.form().map(|f| f.id), Some("wheel_humanoid"));
        assert!(m.label_ko().contains("메카넘"), "{}", m.label_ko());
        let mast = Morphology::named("mast_gripper").unwrap();
        let (top, stroke) = mast.mast_lift_m().unwrap();
        assert!((top - MAST_LIFT_TOP_M).abs() < 1e-6 && (stroke - (MAST_LIFT_TOP_M - DECK_Z_M - DECK_ARM_BASE_H_M)).abs() < 1e-6, "{top} {stroke}");
        let low = Morphology { head: HeadMount::MastCamera { height_mm: 900, pan_tilt: true }, ..mast };
        assert!((low.mast_lift_m().unwrap().0 - (0.9 - MAST_CAMERA_CLEAR_M)).abs() < 1e-6);
        for id in ["deck_gripper", "truck", "wheel_humanoid", "wheel_noneck"] {
            assert!(Morphology::named(id).unwrap().mast_lift_m().is_none(), "{id}");
        }
        assert!(Morphology { manip: Manip::None, ..mast }.mast_lift_m().is_none(), "a camera pole without an arm has no lift");
    }

    #[test]
    fn a_column_runs_up_to_a_mast_camera_above_it() {
        let cam = HeadMount::MastCamera { height_mm: STRETCH_CAMERA_H_MM, pan_tilt: true };
        let d = Dims::COMPACT;
        assert!((column_top_m(&d, cam) - (STRETCH_CAMERA_H_MM as f32 / 1000.0 - d.camera_up_m)).abs() < 1e-6);
        assert!(column_kg(&d, Torso::Mast, cam) > column_kg(&d, Torso::Mast, HeadMount::Neck));
        assert_eq!(column_top_m(&d, HeadMount::MastCamera { height_mm: 900, pan_tilt: false }), d.column_top_m());
        assert_eq!(column_top_m(&d, HeadMount::Neck), d.column_top_m());
    }

    #[test]
    fn a_custom_form_gets_a_compact_key() {
        let c = Morphology { torso: Torso::Mast, head: HeadMount::Fixed { tilt_deg: 20 }, manip: Manip::Arm1, ..Morphology::default() }.fit(BaseKind::Wheel);
        let k = c.upper_key().unwrap();
        assert!(k.contains("t:mast") && k.contains("fixed20") && k.contains("arm1"), "{k}");
        assert!(c.form().is_none());
        assert!(c.label_ko().contains("마스트 기둥"));
    }

    #[test]
    fn sizes_out_of_range_are_named_and_clamped() {
        let w = BaseKind::Wheel;
        let truck = Morphology::named("truck").unwrap();
        let mast = Morphology::named("mast_gripper").unwrap();
        let cases = [
            Morphology { head: HeadMount::Fixed { tilt_deg: 180 }, ..Morphology::default() },
            Morphology { head: HeadMount::Fixed { tilt_deg: -90 }, ..Morphology::default() },
            Morphology { head: HeadMount::MastCamera { height_mm: u16::MAX, pan_tilt: true }, ..mast },
            Morphology { head: HeadMount::ChassisFront { height_mm: 0 }, ..truck },
            Morphology { head: HeadMount::ChassisFront { height_mm: 5000 }, ..truck },
            Morphology { cargo: Cargo::Bed { len_mm: u16::MAX, width_mm: 400, height_mm: 250, payload_kg: 100 }, ..truck },
            Morphology { cargo: Cargo::Bed { len_mm: 500, width_mm: u16::MAX, height_mm: u16::MAX, payload_kg: 100 }, ..truck },
        ];
        for c in cases {
            assert!(c.validate(w).is_err(), "{c:?}");
            let f = c.fit(w);
            assert!(f.validate(w).is_ok(), "{c:?} -> {f:?}: {:?}", f.issues(w));
            assert_eq!((f.torso, f.manip), (c.torso, c.manip), "only the size changes");
        }
        assert_eq!(cases[0].fit(w).head, HeadMount::Fixed { tilt_deg: FIXED_TILT_RANGE_DEG[1] });
        assert_eq!(cases[2].fit(w).head, HeadMount::MastCamera { height_mm: MAST_CAMERA_MAX_MM, pan_tilt: true });
        assert_eq!(cases[3].fit(w).head, HeadMount::ChassisFront { height_mm: CHASSIS_CAMERA_RANGE_MM[0] });
        let Cargo::Bed { len_mm, .. } = cases[5].fit(w).cargo else { panic!() };
        assert_eq!(len_mm, BED_MAX_MM[0]);
        for f in &FORMS {
            assert_eq!(f.morph.fit(f.morph.lower.base()), f.morph, "{}", f.id);
        }
    }
}
