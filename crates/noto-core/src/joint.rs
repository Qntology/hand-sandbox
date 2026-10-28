use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Actuation {
    Direct,
    Coupled { ratio: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointSpec {
    pub kind: JointKind,
    pub flex_min: f32,
    pub flex_max: f32,
    pub abd_limit: f32,
    pub roll_limit: f32,
    pub max_speed: f32,
    pub max_torque: f32,
}

impl JointSpec {
    pub fn clamp_flex(&self, flex: f32) -> f32 {
        flex.clamp(self.flex_min, self.flex_max)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointAngles {
    pub flex: f32,
    pub abd: f32,
    pub roll: f32,
}
