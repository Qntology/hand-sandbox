//! Hand configuration: finger count presets, joint options and base choice,
//! and the geometry each configuration builds.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::base::{BaseKind, BaseSpec};
use crate::finger::FingerSpec;
use crate::joint::{Actuation, JointKind, JointSpec};
use crate::math::Vec3;

/// Torque of the proximal (knuckle) joint, N·m.
const KNUCKLE_TORQUE_NM: f32 = 1.2;
/// Torque of every distal joint, N·m.
const DISTAL_TORQUE_NM: f32 = 0.4;
/// Drive speed of every joint, rad/s (360 deg/s).
const JOINT_SPEED_RAD_S: f32 = std::f32::consts::TAU;
/// Lateral width of the palm for each finger count, metres.
const PALM_WIDTH_M: f32 = 0.09;
const PALM_LENGTH_M: f32 = 0.10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    FingerCount(u8),
    Segments(u8),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::FingerCount(n) => {
                write!(f, "finger count must be 2, 3, 4 or 5 (got {n})")
            }
            ConfigError::Segments(n) => {
                write!(f, "joints per finger must be 2, 3 or 4 (got {n})")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Number of fingers, including the thumb when the preset has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub enum FingerCount {
    Two,
    Three,
    Four,
    Five,
}

impl FingerCount {
    pub const ALL: [FingerCount; 4] = [
        FingerCount::Two,
        FingerCount::Three,
        FingerCount::Four,
        FingerCount::Five,
    ];

    pub fn count(self) -> u8 {
        match self {
            FingerCount::Two => 2,
            FingerCount::Three => 3,
            FingerCount::Four => 4,
            FingerCount::Five => 5,
        }
    }
}

impl TryFrom<u8> for FingerCount {
    type Error = ConfigError;

    fn try_from(n: u8) -> Result<Self, Self::Error> {
        match n {
            2 => Ok(FingerCount::Two),
            3 => Ok(FingerCount::Three),
            4 => Ok(FingerCount::Four),
            5 => Ok(FingerCount::Five),
            other => Err(ConfigError::FingerCount(other)),
        }
    }
}

impl From<FingerCount> for u8 {
    fn from(c: FingerCount) -> u8 {
        c.count()
    }
}

/// Joint options applied to every finger of the hand.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointOptions {
    /// Joints per finger, 2 to 4.
    pub segments: u8,
    /// Kind of the proximal knuckle joint.
    pub base_kind: JointKind,
    /// Kind of every other joint in the finger.
    pub distal_kind: JointKind,
    pub actuation: Actuation,
}

impl Default for JointOptions {
    /// Human-like finger: universal knuckle, hinge joints, tendon-style coupling.
    fn default() -> Self {
        JointOptions {
            segments: 3,
            base_kind: JointKind::Universal,
            distal_kind: JointKind::Hinge,
            actuation: Actuation::Coupled { ratio: 0.67 },
        }
    }
}

/// Full choice for one robot: fingers, joints and the locomotion base.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HandConfig {
    pub fingers: FingerCount,
    pub joints: JointOptions,
    pub base: BaseKind,
}

impl HandConfig {
    /// Default configuration for a finger count.
    ///
    /// * 2 fingers: parallel-jaw style pinch gripper, hinge joints, direct drive.
    /// * 3 fingers: thumb and two fingers (tripod).
    /// * 4 fingers: thumb and three fingers.
    /// * 5 fingers: human-like hand with tendon-coupled fingers.
    pub fn preset(fingers: FingerCount) -> Self {
        let joints = match fingers {
            FingerCount::Two => JointOptions {
                segments: 2,
                base_kind: JointKind::Hinge,
                distal_kind: JointKind::Hinge,
                actuation: Actuation::Direct,
            },
            FingerCount::Three => JointOptions {
                segments: 3,
                ..JointOptions::default()
            },
            FingerCount::Four | FingerCount::Five => JointOptions::default(),
        };
        HandConfig {
            fingers,
            joints,
            base: BaseKind::Wheel,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(2..=4).contains(&self.joints.segments) {
            return Err(ConfigError::Segments(self.joints.segments));
        }
        Ok(())
    }

    /// Builds the hand geometry. Fails when the options are out of range.
    pub fn build(&self) -> Result<Hand, ConfigError> {
        self.validate()?;
        let fingers = layout(self.fingers)
            .iter()
            .map(|slot| slot.build(&self.joints))
            .collect();
        Ok(Hand {
            config: *self,
            palm: PalmSpec {
                width: PALM_WIDTH_M,
                length: PALM_LENGTH_M,
            },
            fingers,
            base: self.base.spec(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PalmSpec {
    pub width: f32,
    pub length: f32,
}

/// A built hand: geometry for every finger plus the locomotion base.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hand {
    pub config: HandConfig,
    pub palm: PalmSpec,
    pub fingers: Vec<FingerSpec>,
    pub base: BaseSpec,
}

/// Where one finger sits on the palm.
struct Slot {
    name: &'static str,
    thumb: bool,
    x: f32,
    yaw: f32,
}

impl Slot {
    fn build(&self, options: &JointOptions) -> FingerSpec {
        let n = options.segments as usize;
        let joints = (0..n)
            .map(|i| joint_spec(options, i, self.thumb))
            .collect();
        FingerSpec {
            name: self.name.to_string(),
            is_thumb: self.thumb,
            mount: Vec3::new(self.x, 0.0, 0.0),
            mount_yaw: self.yaw,
            segments: segment_lengths(n, self.thumb),
            joints,
        }
    }
}

/// Finger placement per finger count. Fingers flex toward +Y (palm side).
fn layout(count: FingerCount) -> Vec<Slot> {
    match count {
        FingerCount::Two => vec![
            Slot { name: "left jaw", thumb: false, x: -0.035, yaw: 0.20 },
            Slot { name: "right jaw", thumb: false, x: 0.035, yaw: -0.20 },
        ],
        FingerCount::Three => vec![
            Slot { name: "thumb", thumb: true, x: -0.035, yaw: 0.50 },
            Slot { name: "index", thumb: false, x: 0.0, yaw: -0.05 },
            Slot { name: "middle", thumb: false, x: 0.03, yaw: -0.10 },
        ],
        FingerCount::Four => vec![
            Slot { name: "thumb", thumb: true, x: -0.040, yaw: 0.50 },
            Slot { name: "index", thumb: false, x: -0.012, yaw: 0.05 },
            Slot { name: "middle", thumb: false, x: 0.012, yaw: -0.05 },
            Slot { name: "ring", thumb: false, x: 0.035, yaw: -0.15 },
        ],
        FingerCount::Five => vec![
            Slot { name: "thumb", thumb: true, x: -0.045, yaw: 0.60 },
            Slot { name: "index", thumb: false, x: -0.025, yaw: 0.08 },
            Slot { name: "middle", thumb: false, x: -0.005, yaw: 0.0 },
            Slot { name: "ring", thumb: false, x: 0.015, yaw: -0.06 },
            Slot { name: "pinky", thumb: false, x: 0.035, yaw: -0.14 },
        ],
    }
}

fn segment_lengths(n: usize, thumb: bool) -> Vec<f32> {
    // Thumbs are a little shorter than fingers in the proximal segments.
    let base: &[f32] = match (n, thumb) {
        (2, _) => &[0.050, 0.035],
        (3, true) => &[0.040, 0.028, 0.020],
        (3, false) => &[0.045, 0.028, 0.020],
        (_, _) => &[0.040, 0.025, 0.018, 0.012],
    };
    base.to_vec()
}

/// Limits for joint `index` of a finger. Index 0 is the knuckle.
fn joint_spec(options: &JointOptions, index: usize, thumb: bool) -> JointSpec {
    let (kind, torque) = if index == 0 {
        (options.base_kind, KNUCKLE_TORQUE_NM)
    } else {
        (options.distal_kind, DISTAL_TORQUE_NM)
    };
    // Approximate human flexion ranges per joint, editable in the config.
    let flex_max_deg: f32 = match index {
        0 => 90.0,
        1 => 100.0,
        2 => 80.0,
        _ => 70.0,
    };
    let flex_max = if thumb && index == 0 {
        60.0_f32.to_radians()
    } else {
        flex_max_deg.to_radians()
    };
    JointSpec {
        kind,
        flex_min: 0.0,
        flex_max,
        abd_limit: 15.0_f32.to_radians(),
        roll_limit: 25.0_f32.to_radians(),
        max_speed: JOINT_SPEED_RAD_S,
        max_torque: torque,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finger_count_round_trips_through_u8() {
        for n in 2u8..=5 {
            let c = FingerCount::try_from(n).unwrap();
            assert_eq!(u8::from(c), n);
        }
        assert_eq!(
            FingerCount::try_from(6),
            Err(ConfigError::FingerCount(6))
        );
    }

    #[test]
    fn presets_build_expected_finger_counts() {
        for count in FingerCount::ALL {
            let hand = HandConfig::preset(count).build().unwrap();
            assert_eq!(hand.fingers.len(), count.count() as usize);
        }
    }

    #[test]
    fn thumb_is_present_from_three_fingers_up() {
        let two = HandConfig::preset(FingerCount::Two).build().unwrap();
        assert!(two.fingers.iter().all(|f| !f.is_thumb));
        let five = HandConfig::preset(FingerCount::Five).build().unwrap();
        assert_eq!(five.fingers.iter().filter(|f| f.is_thumb).count(), 1);
    }

    #[test]
    fn joint_option_controls_segments_and_kinds() {
        let mut cfg = HandConfig::preset(FingerCount::Five);
        cfg.joints.segments = 4;
        cfg.joints.base_kind = JointKind::Ball;
        cfg.joints.distal_kind = JointKind::Hinge;
        let hand = cfg.build().unwrap();
        for f in &hand.fingers {
            assert_eq!(f.joint_count(), 4);
            assert_eq!(f.segments.len(), 4);
            assert_eq!(f.joints[0].kind, JointKind::Ball);
            assert!(f.joints[1..].iter().all(|j| j.kind == JointKind::Hinge));
        }
    }

    #[test]
    fn out_of_range_segments_are_rejected() {
        let mut cfg = HandConfig::preset(FingerCount::Three);
        cfg.joints.segments = 5;
        assert_eq!(cfg.build().unwrap_err(), ConfigError::Segments(5));
        cfg.joints.segments = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn distal_joints_have_lower_torque_than_knuckle() {
        let hand = HandConfig::preset(FingerCount::Five).build().unwrap();
        let f = &hand.fingers[1];
        assert!(f.joints[0].max_torque > f.joints[1].max_torque);
    }
}
