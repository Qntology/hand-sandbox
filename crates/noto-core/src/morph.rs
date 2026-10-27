//! Layered robot morphology: what the robot is built from, layer by layer.
//!
//! Until v0.9 only the lower body varied (`BaseKind`): everything above it was
//! one humanoid, a pelvis with a waist, a trunk, a pan-tilt neck carrying the
//! head camera, two arms with hands and a cargo bay in front of the belly. A
//! [`Morphology`] names each layer on its own:
//!
//! * [`Lower`]: legs, wheels, mecanum wheels or tracks. It follows the drive
//!   (`NotoConfig.base` stays the source of truth, [`Lower::from_base`]).
//! * [`Torso`]: the humanoid pelvis-waist-trunk, a fixed column (a mast that
//!   carries the arms and the head on a carriage), or none (a deck; with a
//!   camera on a mast, the arm rides a lift on that mast, Stretch-like).
//! * [`HeadMount`]: where the camera is: a head on a pan-tilt neck, a head
//!   fixed to the trunk, a camera on a mast or a pole, a low camera on the
//!   chassis front, or none.
//! * [`Manip`]: two arms with hands, one arm with a hand, one arm with a
//!   parallel-jaw gripper, or none.
//! * [`Cargo`]: the bay in front of the pelvis, a bed on the chassis (a
//!   truck), a tote on the deck, or none.
//!
//! The default is today's humanoid, so a plan without a `morph` field builds
//! the same robot as before, to the last digit. The named forms
//! ([`Morphology::named`]) are combinations that make sense together, each
//! after a real robot (`docs/research/parts.md` §7).
//!
//! The capability methods on [`Body`] (`legged()`, `has_neck()`,
//! `arm_count()`, `cargo_geom()`, `camera_mount()`, ...) are what the physics
//! asks, and the mount geometry here is the one `balance`, `kin` and
//! `dynamics` share, so the three models place the same masses. Every length
//! and mass below is cited (in `docs/research/parts.md`) or marked as an
//! estimate (est) in both places.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::balance::{Body, Posture, CARGO_X_M, CARGO_Z_M, MAST_KG};
use crate::base::BaseKind;
use crate::dynamics::LinkBox;
use crate::hand::{FingerCount, Hand, NotoConfig};
use crate::kin::{KinPose, CAMERA_UP_M, NECK_UP_M, WHEEL_R_M};
use crate::parts::*;

// ----- the layers -----

/// The lower body. It mirrors the drive (`BaseKind`), which stays the source of truth.
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

    /// The drive this lower body is.
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

/// What sits between the lower body and the arms and the head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Torso {
    /// Pelvis, waist (yaw, and pitch when the belly preset has it) and trunk: today's robot.
    #[default]
    Humanoid,
    /// A fixed column on the chassis whose carriage carries the shoulders and the
    /// head (the torso lift of `BodyPlan.lift` slides it down): Fetch- and TIAGo-like.
    Mast,
    /// No torso: the chassis deck carries the arm and the cargo, the chassis
    /// holds the electronics (with a camera on a mast, the arm rides a lift on
    /// that mast).
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

/// Where the camera is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HeadMount {
    /// A head on a pan-tilt neck at the top of the trunk (or of the column): today's.
    #[default]
    Neck,
    /// A head fixed to the trunk (or the column top), its camera tilted down by
    /// `tilt_deg`: no neck joint, the camera turns only with the body.
    Fixed { tilt_deg: i16 },
    /// A camera on a pole or a mast at `height_mm` above the ground (optical
    /// centre), on a pan-tilt unit or fixed: Stretch- and Fetch-like.
    MastCamera { height_mm: u16, pan_tilt: bool },
    /// A low camera on the chassis front at `height_mm`: AMR- and truck-like.
    ChassisFront { height_mm: u16 },
    /// No camera.
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

    /// A pan-tilt joint carries the camera: the neck, or a pan-tilt unit on a mast.
    pub fn pans_and_tilts(self) -> bool {
        matches!(self, HeadMount::Neck | HeadMount::MastCamera { pan_tilt: true, .. })
    }

    /// The head sits on the trunk (or on the column's carriage) and moves with it.
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

/// What handles objects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Manip {
    /// Two arms with the configured hand (`NotoConfig.fingers`): today's.
    #[default]
    Arms2,
    /// One arm (the right one) with the configured hand.
    Arm1,
    /// One arm with a parallel-jaw gripper (`GripperPreset`), on the trunk, the
    /// column, a camera mast's lift or the deck: SO-101-, Stretch-class.
    Gripper1,
    /// Nothing: the robot carries what others load onto it.
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

/// Where carried things go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Cargo {
    /// The bay in front of the pelvis (`balance::CARGO_X_M`, `CARGO_Z_M`): today's.
    #[default]
    Bay,
    /// A flat bed on the chassis, `len_mm` along the travel, `width_mm` across,
    /// its surface `height_mm` above the ground, rated `payload_kg`: a truck.
    Bed { len_mm: u16, width_mm: u16, height_mm: u16, payload_kg: u16 },
    /// A small bin on the deck front ([`TOTE_SIZE_M`]).
    Tote,
    /// Nothing: the gripper holds what it carries.
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

/// The robot's form: one choice per layer.
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

// ----- mount geometry (shared by balance, kin and dynamics) -----

/// Top of the chassis deck above the ground, metres: the chassis box `kin`
/// draws (centre `WHEEL_R_M + 0.05`, 0.05 half high).
pub const DECK_Z_M: f32 = WHEEL_R_M + 0.10;
/// Centre height of the electronics (computer and battery) of a robot without
/// a humanoid trunk, metres: inside the chassis body, under the deck, at the
/// chassis box's centre (est: an AMR's or a mobile manipulator's base holds
/// its battery and computer; on the deck they would stand where the tote, the
/// arm's base and the column are).
pub const ELECTRONICS_Z_M: f32 = WHEEL_R_M + 0.05;
/// The box itself that holds the computer and the battery's electronics in the
/// chassis of a robot without a humanoid trunk, kg (est: a small printed or
/// sheet-metal case).
pub const DECK_BOX_KG: f32 = 0.30;
/// Half the footprint of a deck arm's base (the shoulder pan servo and its
/// bracket, between the deck and the shoulder), metres (est: an SO-101-class
/// base is about 0.06 m across). Drawn only: its mass is the arm preset's.
pub const DECK_ARM_BASE_HALF_M: f32 = 0.03;
/// A deck arm's shoulder (its first pitch axis), metres: behind the deck's
/// centre, which leaves the front of the deck to a tote, and above the deck by
/// the arm's base (est: an SO-101-class arm's shoulder-lift axis sits about
/// 0.1 m above its mounting plate).
pub const DECK_ARM_X_M: f32 = -0.06;
pub const DECK_ARM_BASE_H_M: f32 = 0.10;
/// Half the spacing of two arms on a deck, metres (est, inside the 0.24 m deck).
pub const DECK_ARMS_HALF_SPACING_M: f32 = 0.10;
/// Rest of a deck arm, degrees (shoulder from hanging, elbow): it cannot hang
/// below its deck, so it folds above it, the upper arm 30° above horizontal
/// and the forearm back over the shoulder (est).
pub const DECK_REST_DEG: [f32; 2] = [120.0, 120.0];
/// A deck arm reaching forward and down to an object in front of the bumper,
/// degrees, for the quasi-static lift check of `balance` (est: the palm about
/// 0.40 m ahead of the shoulder, 0.23 m above the ground).
pub const DECK_LIFT_DEG: [f32; 2] = [70.0, 30.0];
/// Mass of 2020 aluminium extrusion per metre, kg/m (parts.md §4 [46]).
pub const EXTRUSION_KG_PER_M: f32 = 0.50;
/// Clamps and the camera plate of a camera pole, kg (est).
pub const POLE_MOUNT_KG: f32 = 0.20;
/// A camera module on the chassis front with its bracket, kg (est: a D435i is
/// about 72 g, parts.md §3 [43], plus the bracket).
pub const CHASSIS_CAMERA_KG: f32 = 0.10;
/// Mass of a bed plate per area, kg/m²: 3 mm of aluminium at 2.70 g/cm³
/// (parts.md §4); its posts and rails are left out (est).
pub const BED_KG_PER_M2: f32 = 8.1;
/// Height of a load's centre of mass above the bed or the tote floor, metres
/// (est: half of a 0.2 m parcel).
pub const CARGO_ABOVE_FLOOR_M: f32 = 0.10;
/// The tote on a deck: length, width and height, metres, and its own mass, kg (est).
pub const TOTE_SIZE_M: [f32; 3] = [0.16, 0.24, 0.12];
pub const TOTE_KG: f32 = 0.4;
/// The tote's centre along x on the deck, metres (est: in front of the deck arm).
pub const TOTE_X_M: f32 = 0.06;
/// The palm lets an item go this far above a tote's or a bed's rim, metres
/// (est: an item up to 0.15 m hangs under the palm and clears the rim; the
/// arm comes over it from above instead of through the deck).
pub const RIM_RELEASE_M: f32 = 0.20;
/// Top of the column of a `Torso::Mast`, metres: its carriage holds the
/// shoulders at `SHOULDER_M` and the neck where the humanoid's is.
pub const COLUMN_TOP_M: f32 = SHOULDER_M + NECK_UP_M;
/// Down-tilt limit of a pan-tilt camera on a mast, degrees: Fetch's
/// head_tilt_joint upper limit, 1.45 rad (fetch_description URDF).
pub const MAST_TILT_MAX_DEG: f32 = 83.0;
/// Camera height of the Stretch-like form, mm: the head camera of Stretch RE1
/// at 131 cm (Kemp et al., ICRA 2022, Sec. II).
pub const STRETCH_CAMERA_H_MM: u16 = 1310;
/// Camera height of the LeKiwi-like deck form, mm (est: on the deck's front
/// edge, 0.04 m above the 0.16 m deck).
pub const DECK_CAMERA_H_MM: u16 = 200;
/// Camera height of the truck form, mm (est: under the top of an AMR body,
/// MiR250 is 300 mm high with its 3D cameras low at the front, MiR250
/// Specifications 2023).
pub const AMR_CAMERA_H_MM: u16 = 250;
/// Down-tilt of a head fixed to the trunk, degrees (est: without a neck to
/// look down, 15° more brings the floor from about 0.9 m ahead into the
/// render camera's 50° tall view at the 1.14 m eye, where a level head with
/// that camera's own 12° down-pitch sees it from 1.5 m; geometry only).
pub const FIXED_HEAD_TILT_DEG: i16 = 15;
/// The truck's bed: 0.50 × 0.40 m, its surface 0.25 m up, rated 100 kg (the
/// AgileX Tracer's payload, parts.md [33]; size and height est, under the
/// 300 mm top of a MiR250).
pub const TRUCK_BED: Cargo = Cargo::Bed { len_mm: 500, width_mm: 400, height_mm: 250, payload_kg: 100 };

/// Top of the carriage on a mast that carries the arm as well as the camera
/// (no torso: Stretch-like, [`Morphology::mast_arms`]), metres: the shoulder's
/// highest point. Stretch RE1's lift takes its arm from the floor up to 1.10 m
/// (Kemp et al., ICRA 2022, Sec. II).
pub const MAST_LIFT_TOP_M: f32 = 1.10;
/// Room kept between the carriage's top and the mast camera's lens, metres
/// (est: the pan-tilt head and its bracket).
pub const MAST_CAMERA_CLEAR_M: f32 = 0.15;
/// The shoulder of an arm on a mast's carriage, ahead of the mast, metres
/// (est: the carriage's bracket).
pub const MAST_ARM_X_M: f32 = 0.05;

// The sizes a custom form must stay within (est: what this project's
// 0.25–0.36 m chassis can carry). `Morphology::issues` names a size outside
// them and `Morphology::fit` clamps it in.

/// A fixed head's down-tilt, degrees (est): from 45° up, where the render
/// camera's 50° tall view (with its own 12° down-pitch) holds no floor any
/// more, to 60° down, where it sees only the robot's own front.
pub const FIXED_TILT_RANGE_DEG: [i16; 2] = [-45, 60];
/// The highest camera on a mast or a pole, mm (est: above Fetch's highest
/// head, 1.447 m, fetch_description URDF; a taller pole on this chassis is
/// another machine).
pub const MAST_CAMERA_MAX_MM: u16 = 2000;
/// A chassis-front camera's height, mm (est: from clear of the ground to
/// twice the 300 mm top of an AMR body, MiR250 Specifications 2023).
pub const CHASSIS_CAMERA_RANGE_MM: [u16; 2] = [50, 600];
/// The largest bed: length, width and surface height, mm (est: a bed past
/// these on a 0.25–0.36 m chassis is a trailer, and its plate alone weighs
/// several kilograms).
pub const BED_MAX_MM: [u16; 3] = [1000, 800, 800];

/// Where a camera pole stands on a deck, x in metres: at its rear edge (est).
pub fn pole_x_m(support_len_m: f32) -> f32 {
    -0.5 * support_len_m + 0.02
}

/// Top of a `Torso::Mast` column, metres: [`COLUMN_TOP_M`], or the pivot of a
/// camera on a mast above it (the column runs on up to the camera's pan-tilt).
pub fn column_top_m(head: HeadMount) -> f32 {
    match head {
        HeadMount::MastCamera { height_mm, .. } => COLUMN_TOP_M.max(height_mm as f32 / 1000.0 - CAMERA_UP_M),
        _ => COLUMN_TOP_M,
    }
}

/// The upper layers of today's humanoid.
const HUMANOID_UPPER: (Torso, HeadMount, Manip, Cargo) = (Torso::Humanoid, HeadMount::Neck, Manip::Arms2, Cargo::Bay);

/// A named form: id, Korean label, the key its upper layers go by in a plan
/// id (`None`: the humanoid's, no suffix), the layers, and the real robot it
/// is after.
pub struct Form {
    pub id: &'static str,
    pub label_ko: &'static str,
    pub key: Option<&'static str>,
    pub morph: Morphology,
    /// The real robot the form follows, for the docs and the UI.
    pub analogue: &'static str,
}

const fn m(lower: Lower, torso: Torso, head: HeadMount, manip: Manip, cargo: Cargo) -> Morphology {
    Morphology { lower, torso, head, manip, cargo }
}

/// The named forms.
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
    /// Today's humanoid on the lower body of `base`.
    pub fn humanoid(base: BaseKind) -> Morphology {
        Morphology { lower: Lower::from_base(base), ..Morphology::default() }
    }

    pub const NAMED: [&'static str; 7] = ["legs_humanoid", "wheel_humanoid", "wheel_noneck", "mast_gripper", "deck_gripper", "truck", "track_humanoid"];

    /// A named form (its lower body included: apply it with the base [`Lower::base`]).
    pub fn named(id: &str) -> Option<Morphology> {
        FORMS.iter().find(|f| f.id == id).map(|f| f.morph)
    }

    /// The named form this is, if any: the exact one, else the one on wheels
    /// when this one rolls on mecanum wheels (today's mecanum robot is the
    /// wheel humanoid), else the one with the same upper layers (a truck on
    /// tracks is still the truck).
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

    /// The upper layers are today's humanoid's.
    pub fn humanoid_upper(&self) -> bool {
        self.upper() == HUMANOID_UPPER
    }

    /// How the upper layers show in a plan id: `None` for the humanoid (so
    /// today's ids stay), a named form's key, or a compact code.
    pub fn upper_key(&self) -> Option<String> {
        if self.humanoid_upper() {
            return None;
        }
        if let Some(k) = FORMS.iter().find(|f| f.morph.upper() == self.upper()).and_then(|f| f.key) {
            return Some(k.to_string());
        }
        Some(format!("t:{} h:{} m:{} c:{}", self.torso.id(), self.head.code(), self.manip.id(), self.cargo.code()))
    }

    /// A Korean name: the named form's, or the layers one by one.
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

    /// The pelvis-waist-trunk exists (waist joints, a trunk that leans).
    pub fn has_torso(&self) -> bool {
        self.torso == Torso::Humanoid
    }

    /// A pan-tilt joint carries the camera (the neck, or a pan-tilt unit on a mast).
    pub fn has_neck(&self) -> bool {
        self.head.pans_and_tilts()
    }

    pub fn arm_count(&self) -> usize {
        self.manip.arm_count()
    }

    /// The arms ride a lift carriage on the camera's mast: no torso, a camera
    /// on a mast and at least one arm (Stretch-like). Without a mast camera, the
    /// arms of a torso-less form stand on the deck.
    pub fn mast_arms(&self) -> bool {
        self.torso == Torso::None && matches!(self.head, HeadMount::MastCamera { .. }) && self.arm_count() > 0
    }

    /// The carriage of a mast with arms: its top (the shoulder's highest point,
    /// [`MAST_LIFT_TOP_M`] or [`MAST_CAMERA_CLEAR_M`] under the lens, whichever
    /// is lower) and its travel down to a deck arm's shoulder height, metres.
    pub fn mast_lift_m(&self) -> Option<(f32, f32)> {
        let HeadMount::MastCamera { height_mm, .. } = self.head else { return None };
        if !self.mast_arms() {
            return None;
        }
        let bottom = DECK_Z_M + DECK_ARM_BASE_H_M;
        let top = MAST_LIFT_TOP_M.min(height_mm as f32 / 1000.0 - MAST_CAMERA_CLEAR_M).max(bottom);
        Some((top, top - bottom))
    }

    /// What is wrong with this form on `base`, in Korean (empty: it can be built).
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

    /// `Ok` when the form can be built on `base`, else the reasons.
    pub fn validate(&self, base: BaseKind) -> Result<(), String> {
        let v = self.issues(base);
        if v.is_empty() {
            Ok(())
        } else {
            Err(v.join("; "))
        }
    }

    /// The nearest form that can be built on `base`: the lower body follows
    /// the drive, legs get the humanoid torso, a head without a trunk to sit on
    /// moves to a mast (pan-tilt kept), a camera that needs a chassis goes back
    /// on a neck, a bay without a pelvis becomes a tote, a bed or a tote on
    /// legs becomes the bay, and sizes come into their ranges
    /// ([`FIXED_TILT_RANGE_DEG`], [`MAST_CAMERA_MAX_MM`],
    /// [`CHASSIS_CAMERA_RANGE_MM`], [`BED_MAX_MM`]). A valid form comes back
    /// unchanged.
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

/// The deck's top, mm (a bed cannot be lower).
fn deck_mm() -> u16 {
    (1000.0 * DECK_Z_M).round() as u16
}

/// The lowest a mast camera may sit, mm: 0.1 m above the deck.
fn min_mast_camera_mm() -> u16 {
    (1000.0 * (DECK_Z_M + 0.1)).round() as u16
}

/// Where the cargo sits and how the hand puts things in it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CargoGeom {
    pub cargo: Cargo,
    /// The load's centre of mass in the base frame at rest (no lift, upright), metres.
    pub at: [f32; 3],
    /// It rides on the pelvis (moves with the torso lift and the pelvis pitch)
    /// rather than on the chassis.
    pub on_pelvis: bool,
    /// The point (x, z) over which the hand lets an item go, metres.
    pub release: [f32; 2],
    /// Height of the rim (the bay's, the tote's) or of the bed's surface, metres.
    pub rim_z_m: f32,
    /// Height of the floor an item lands on (the tote's bottom, the bed), metres.
    pub floor_z_m: f32,
    /// Rated load of the structure, kg (`None`: the base's own limit applies).
    pub payload_kg: Option<f32>,
    /// The structure itself (a bed plate, a tote), kg, with its centre of mass,
    /// and the box it is drawn as (centre and half sizes), metres.
    pub struct_kg: f32,
    pub struct_at: [f32; 3],
    pub center: [f32; 3],
    pub half: [f32; 3],
}

/// A mass fixed to the chassis frame besides the chassis itself: the
/// electronics in the chassis of a robot without a humanoid trunk, a camera
/// that is not on the trunk, a bed plate or a tote.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MountMass {
    pub name: &'static str,
    pub segment: Segment,
    pub kg: f32,
    pub at: [f32; 3],
}

impl Body {
    /// Walks on legs (the only switch before v1.0).
    pub fn legged(&self) -> bool {
        self.morph.legged()
    }

    /// Has the humanoid pelvis-waist-trunk (waist joints, a trunk that leans).
    pub fn has_torso(&self) -> bool {
        self.morph.has_torso()
    }

    /// A fixed column (`Torso::Mast`) carries the arms and the head.
    pub fn has_column(&self) -> bool {
        self.morph.torso == Torso::Mast
    }

    /// A pan-tilt joint (`neck`) carries the camera.
    pub fn has_neck(&self) -> bool {
        self.morph.has_neck()
    }

    /// The head rides on the trunk or the column carriage.
    pub fn head_on_trunk(&self) -> bool {
        self.morph.head.on_trunk()
    }

    pub fn arm_count(&self) -> usize {
        self.morph.arm_count()
    }

    pub fn has_manipulator(&self) -> bool {
        self.arm_count() > 0
    }

    /// One arm with a parallel-jaw gripper and no hand.
    pub fn gripper_only(&self) -> bool {
        self.morph.manip == Manip::Gripper1
    }

    /// The arms stand on the deck (no torso, no mast to carry them).
    pub fn deck_arms(&self) -> bool {
        self.morph.torso == Torso::None && self.has_manipulator() && !self.mast_arms()
    }

    /// The arms ride a lift carriage on the camera's mast ([`Morphology::mast_arms`]).
    pub fn mast_arms(&self) -> bool {
        self.morph.mast_arms()
    }

    pub fn camera_mount(&self) -> HeadMount {
        self.morph.head
    }

    /// The camera's pitch range below level and pan range either side, degrees
    /// (0 without a pan-tilt joint).
    pub fn head_range_deg(&self) -> (f32, f32) {
        match self.morph.head {
            HeadMount::Neck => (crate::kin::NECK_PITCH_MAX_DEG, crate::kin::NECK_PAN_MAX_DEG),
            // Fetch's head pan is ±1.57 rad, as the neck's 90° (fetch_description URDF).
            HeadMount::MastCamera { pan_tilt: true, .. } => (MAST_TILT_MAX_DEG, crate::kin::NECK_PAN_MAX_DEG),
            _ => (0.0, 0.0),
        }
    }

    /// The shoulder of arm `side` (0 right, 1 left) in the base frame at rest
    /// (no lift, upright), metres: on the trunk or the column carriage at
    /// shoulder height, on a mast's carriage at its top, or on the deck. A
    /// single arm is the right one; on a deck or a mast it stands in the middle.
    pub fn shoulder_mount(&self, side: usize) -> [f32; 3] {
        let y = if side == 0 { -SHOULDER_HALF_WIDTH_M } else { SHOULDER_HALF_WIDTH_M };
        match self.morph.torso {
            Torso::Humanoid | Torso::Mast => [0.0, y, SHOULDER_M],
            Torso::None => {
                let yd = if self.arm_count() >= 2 { if side == 0 { -DECK_ARMS_HALF_SPACING_M } else { DECK_ARMS_HALF_SPACING_M } } else { 0.0 };
                match self.morph.mast_lift_m() {
                    Some((top, _)) => [self.mast_camera_x() + MAST_ARM_X_M, yd, top],
                    None => [DECK_ARM_X_M, yd, DECK_Z_M + DECK_ARM_BASE_H_M],
                }
            }
        }
    }

    /// [`Body::shoulder_mount`] on a pose's lift: an arm on a mast's carriage
    /// comes down with it (`KinPose::lift_m`); a deck arm stays where it
    /// stands. A trunk's and a column's shoulders follow their own chains
    /// (`kin::solve`, `act::reach::shoulder_point`).
    pub fn shoulder_at(&self, side: usize, lift_m: f32) -> [f32; 3] {
        let mut s = self.shoulder_mount(side);
        if self.mast_arms() {
            s[2] -= lift_m;
        }
        s
    }

    /// The arms at rest, degrees: shoulder flexion from hanging and elbow
    /// flexion per arm. Hanging on a trunk, a column or a mast's carriage;
    /// folded above the deck on a deck ([`DECK_REST_DEG`]).
    pub fn rest_arms_deg(&self) -> ([f32; 2], [f32; 2]) {
        if self.deck_arms() {
            ([DECK_REST_DEG[0]; 2], [DECK_REST_DEG[1]; 2])
        } else {
            ([0.0, 0.0], [10.0, 10.0])
        }
    }

    /// Standing still with the arms at rest (`KinPose::stand` for the humanoid).
    pub fn stand(&self, cargo_kg: f32) -> KinPose {
        let (s, e) = self.rest_arms_deg();
        KinPose { shoulder_deg: s, elbow_deg: e, ..KinPose::stand(cargo_kg) }
    }

    /// The quiet posture with the arms at rest (`Posture::quiet` for the humanoid).
    pub fn quiet(&self, cargo_kg: f32) -> Posture {
        let (s, e) = self.rest_arms_deg();
        Posture { arm_elev_deg: s, elbow_deg: e, ..Posture::quiet(cargo_kg) }
    }

    /// The posture of lifting `load_kg` (`Posture::arm_lift` on a trunk, a
    /// column or a mast; a deck arm reaches forward and down, [`DECK_LIFT_DEG`]).
    pub fn lift_posture(&self, load_kg: f32, two_arms: bool, cargo_kg: f32) -> Posture {
        let p = Posture::arm_lift(load_kg, two_arms, cargo_kg);
        if !self.deck_arms() {
            return p;
        }
        let (rs, re) = self.rest_arms_deg();
        let (s1, e1) = if two_arms { (DECK_LIFT_DEG[0], DECK_LIFT_DEG[1]) } else { (rs[1], re[1]) };
        Posture { arm_elev_deg: [DECK_LIFT_DEG[0], s1], elbow_deg: [DECK_LIFT_DEG[1], e1], ..p }
    }

    /// The pivot of the camera's pan-tilt at rest (the neck joint, or the
    /// pan-tilt on a mast), base frame, metres; `None` without one.
    pub fn neck_pivot(&self) -> Option<[f32; 3]> {
        match self.morph.head {
            HeadMount::Neck => Some([0.0, 0.0, SHOULDER_M + NECK_UP_M]),
            HeadMount::MastCamera { height_mm, pan_tilt: true } => Some([self.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - CAMERA_UP_M]),
            _ => None,
        }
    }

    /// Where the camera's pole (or mast) stands along x: on the column (x = 0)
    /// or at the deck's rear edge, which leaves the deck's front to the tote
    /// and the arm's carriage ([`MAST_ARM_X_M`] ahead of it).
    pub fn mast_camera_x(&self) -> f32 {
        if self.morph.torso == Torso::None {
            pole_x_m(self.support_len_m)
        } else {
            0.0
        }
    }

    /// The head's (or the camera module's) centre of mass at rest, base frame,
    /// metres; `None` without a camera. A head on a pan-tilt (a neck, or the
    /// pan-tilt unit of a mast camera) has it `head.com_up_m` above the pivot,
    /// where `kin` and `dynamics` swing it about that pivot; a fixed mast
    /// camera's pivot is the same bracket, `CAMERA_UP_M` under the lens.
    pub fn head_com(&self) -> Option<[f32; 3]> {
        match self.morph.head {
            HeadMount::Neck => Some([0.0, 0.0, SHOULDER_M + NECK_UP_M + self.head.com_up_m]),
            // Pitched down by its fixed tilt about the top of the trunk.
            HeadMount::Fixed { tilt_deg } => {
                let t = (tilt_deg as f32).to_radians();
                Some([self.head.com_up_m * t.sin(), 0.0, SHOULDER_M + NECK_UP_M + self.head.com_up_m * t.cos()])
            }
            HeadMount::MastCamera { height_mm, .. } => Some([self.mast_camera_x(), 0.0, height_mm as f32 / 1000.0 - CAMERA_UP_M + self.head.com_up_m]),
            HeadMount::ChassisFront { height_mm } => Some([0.5 * self.support_len_m, 0.0, height_mm as f32 / 1000.0]),
            HeadMount::None => None,
        }
    }

    /// The electronics (computer and battery in their box) of a robot without
    /// a humanoid trunk: inside the chassis ([`ELECTRONICS_Z_M`]), base frame,
    /// metres.
    pub fn deck_point(&self) -> [f32; 3] {
        [0.0, 0.0, ELECTRONICS_Z_M]
    }

    /// What the form puts on the chassis besides the chassis, its drive and
    /// the mast, column or pole: the base of each deck arm, the cargo
    /// structure (a bed plate, a tote; not the bay, which rides on the
    /// pelvis) and a chassis-front camera, as boxes in the base frame at rest
    /// (upright axes). `kin` and `dynamics` draw the same ones, pitched with
    /// the base.
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

    /// The hand at the end of the arm: the configured one (`NotoConfig`), or,
    /// for a parallel-jaw gripper (`Manip::Gripper1`), the two-finger jaw
    /// (`FingerCount::Two`, the hand the mid gripper's mass is taken from,
    /// `parts::hand_mass_kg`) whatever the configured hand is. Its squeeze is
    /// `act::grasp::capacity_for`'s.
    pub fn working_hand<'a>(&self, hand: &'a Hand) -> Cow<'a, Hand> {
        if !self.gripper_only() || hand.config.fingers == FingerCount::Two {
            return Cow::Borrowed(hand);
        }
        let mut cfg = NotoConfig::preset(FingerCount::Two);
        cfg.base = hand.config.base;
        cfg.build().map_or(Cow::Borrowed(hand), Cow::Owned)
    }

    /// The mast, column or camera pole fixed to the chassis with the torso
    /// lift when there is one: kg and centre at rest. `None` on legs and on a
    /// bare deck.
    pub fn column(&self) -> Option<(f32, [f32; 3])> {
        if self.legged() {
            return None;
        }
        match (self.morph.torso, self.morph.head) {
            (Torso::Humanoid, _) => Some((self.mast_kg(), [0.0, 0.0, 0.35])),
            (Torso::Mast, h) => Some((self.mast_kg(), [0.0, 0.0, 0.5 * (DECK_Z_M + column_top_m(h))])),
            (Torso::None, HeadMount::MastCamera { height_mm, .. }) => Some((self.mast_kg(), [self.mast_camera_x(), 0.0, 0.5 * (DECK_Z_M + height_mm as f32 / 1000.0)])),
            _ => None,
        }
    }

    /// The cargo's place (`None`: no cargo space).
    pub fn cargo_geom(&self) -> Option<CargoGeom> {
        match self.morph.cargo {
            Cargo::Bay => {
                let rim = CARGO_Z_M + 0.08;
                // Its floor: the load's centre is a 0.2 m parcel's half above it (est, as a bed's).
                Some(CargoGeom { cargo: Cargo::Bay, at: [CARGO_X_M, 0.0, CARGO_Z_M], on_pelvis: true, release: [CARGO_X_M + 0.12, rim + 0.03 + 0.05], rim_z_m: rim, floor_z_m: CARGO_Z_M - CARGO_ABOVE_FLOOR_M, payload_kg: None, struct_kg: 0.0, struct_at: [CARGO_X_M, 0.0, CARGO_Z_M], center: [CARGO_X_M, 0.0, CARGO_Z_M], half: [0.08, 0.12, 0.08] })
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

    /// Where the load sits at rest: the cargo's place, or on the deck's
    /// middle when there is none (strapped on, est: its centre
    /// [`CARGO_ABOVE_FLOOR_M`] up), base frame.
    pub fn cargo_point(&self) -> ([f32; 3], bool) {
        match self.cargo_geom() {
            Some(c) => (c.at, c.on_pelvis),
            None => ([0.0, 0.0, DECK_Z_M + CARGO_ABOVE_FLOOR_M], false),
        }
    }

    /// The masses fixed to the chassis other than the chassis and the column:
    /// the electronics of a deck or a column robot (in the chassis), a camera
    /// that is not on the trunk, the cargo structure. Empty for today's humanoid.
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

    /// The named form this body is (`None` for a custom one).
    pub fn form_id(&self) -> Option<&'static str> {
        self.morph.form().map(|f| f.id)
    }
}

/// The head preset as mounted: its mass without the two neck servos when the
/// head is fixed (parts.md §5: every head tier carries its camera on two
/// neck actuators), the camera and its bracket alone on the chassis front,
/// nothing without a camera; no gaze stabilisation without a pan-tilt.
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

/// The computer alone, kg, by the chest tier (parts.md §3): a Raspberry Pi 5
/// weighs 45–62 g [41] (the middle, 54 g), a Jetson Orin Nano Super 175 g
/// [42]; the pro tier's "Orin-class" module is taken as the Orin Nano (est).
pub fn computer_kg(tier: Tier) -> f32 {
    match tier {
        Tier::Budget | Tier::BudgetPlus => 0.054,
        Tier::Mid | Tier::Pro => 0.175,
    }
}

/// The chest preset as built: without a humanoid torso there is no trunk
/// structure (the chest tier's spine and shell), only the computer in its box
/// in the chassis ([`computer_kg`] + [`DECK_BOX_KG`]); the battery stays.
pub fn mounted_chest(mut chest: ChestPreset, torso: Torso) -> ChestPreset {
    if torso != Torso::Humanoid {
        chest.mass_kg = computer_kg(chest.tier) + DECK_BOX_KG;
    }
    chest
}

/// The belly preset as built: without a humanoid torso there is no pelvis
/// and no waist, so no mass, no pitch joint and no lean (the tier is kept).
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

/// Mass of the mast, the column or the camera pole on a wheeled or tracked
/// base without the torso lift, kg: today's mast under a humanoid
/// (`MAST_KG`); a column to [`column_top_m`] is that mast with the extra
/// extrusion above the hip height; a camera pole is extrusion from the deck to
/// the camera with its clamps.
pub fn column_kg(torso: Torso, head: HeadMount) -> f32 {
    match (torso, head) {
        (Torso::Humanoid, _) => MAST_KG,
        (Torso::Mast, h) => MAST_KG + EXTRUSION_KG_PER_M * (column_top_m(h) - HIP_M),
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
        // Missing layers default to the humanoid's.
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
        // The truck on tracks is still the truck.
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
        // Legs take the humanoid torso and a neck; the lower body follows the drive.
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
        // A low mast camera keeps the carriage under its lens.
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
        assert!((column_top_m(cam) - (STRETCH_CAMERA_H_MM as f32 / 1000.0 - CAMERA_UP_M)).abs() < 1e-6);
        assert!(column_kg(Torso::Mast, cam) > column_kg(Torso::Mast, HeadMount::Neck));
        // A camera lower than the column's top sits on it: the column keeps its height.
        assert_eq!(column_top_m(HeadMount::MastCamera { height_mm: 900, pan_tilt: false }), COLUMN_TOP_M);
        assert_eq!(column_top_m(HeadMount::Neck), COLUMN_TOP_M);
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
        // The named forms are inside every range.
        for f in &FORMS {
            assert_eq!(f.morph.fit(f.morph.lower.base()), f.morph, "{}", f.id);
        }
    }
}
