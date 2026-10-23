//! Joint kinds, joint limits and actuation options for one finger joint.

use serde::{Deserialize, Serialize};

/// Degrees of freedom a joint offers.
///
/// * `Hinge`     — flexion only (1 DoF), like a PIP or DIP knuckle.
/// * `Universal` — flexion plus abduction (2 DoF), like an MCP knuckle.
/// * `Ball`      — flexion, abduction and roll about the segment (3 DoF).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    Hinge,
    Universal,
    Ball,
}

impl JointKind {
    pub const ALL: [JointKind; 3] = [JointKind::Hinge, JointKind::Universal, JointKind::Ball];

    pub fn dof(self) -> u8 {
        match self {
            JointKind::Hinge => 1,
            JointKind::Universal => 2,
            JointKind::Ball => 3,
        }
    }

    pub fn has_abduction(self) -> bool {
        !matches!(self, JointKind::Hinge)
    }

    pub fn has_roll(self) -> bool {
        matches!(self, JointKind::Ball)
    }
}

/// How the flexion of the joints in one finger is driven.
///
/// * `Direct`  — every joint has its own actuator.
/// * `Coupled` — one actuator drives the proximal joint and each distal joint
///   follows the joint before it at `ratio` (tendon-style coupling). A
///   ratio near 0.67 mirrors the usual PIP-to-DIP relation of the human hand.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Actuation {
    Direct,
    Coupled { ratio: f32 },
}

/// Limits and drive capability of one joint. Angles are in radians.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointSpec {
    pub kind: JointKind,
    /// Flexion range (min, max).
    pub flex_min: f32,
    pub flex_max: f32,
    /// Half-range of abduction (only used when the kind has abduction).
    pub abd_limit: f32,
    /// Half-range of roll (only used by `Ball`).
    pub roll_limit: f32,
    /// Maximum angular speed of the drive, rad/s.
    pub max_speed: f32,
    /// Maximum drive torque, N·m.
    pub max_torque: f32,
}

impl JointSpec {
    pub fn clamp_flex(&self, flex: f32) -> f32 {
        flex.clamp(self.flex_min, self.flex_max)
    }
}

/// Angles of one joint. Unused axes stay at zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointAngles {
    pub flex: f32,
    pub abd: f32,
    pub roll: f32,
}
