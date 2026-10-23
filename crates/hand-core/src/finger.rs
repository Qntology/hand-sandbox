//! One finger as a serial chain of joints and rigid segments.

use serde::{Deserialize, Serialize};

use crate::joint::{JointAngles, JointSpec};
use crate::math::{Mat3, Vec3};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FingerSpec {
    pub name: String,
    pub is_thumb: bool,
    /// Position of the first joint in the hand frame (metres).
    pub mount: Vec3,
    /// Fan-out of the finger about the palm normal (radians).
    pub mount_yaw: f32,
    /// Length of each segment; one per joint, the last one ends at the tip.
    pub segments: Vec<f32>,
    /// Joint specs, same length as `segments`.
    pub joints: Vec<JointSpec>,
}

impl FingerSpec {
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }

    pub fn reach(&self) -> f32 {
        self.segments.iter().sum()
    }

    /// Rotation of one joint for the given angles. Axes the joint does not
    /// have are ignored, so a `Hinge` never abducts or rolls.
    pub fn joint_rotation(spec: &JointSpec, angles: JointAngles) -> Mat3 {
        let abd = if spec.kind.has_abduction() { angles.abd } else { 0.0 };
        let roll = if spec.kind.has_roll() { angles.roll } else { 0.0 };
        Mat3::rot_y(abd)
            .mul(Mat3::rot_x(-angles.flex))
            .mul(Mat3::rot_z(roll))
    }

    /// Forward kinematics: joint and tip positions in the hand frame, as
    /// `[mount, joint 1, ..., tip]`. `angles` must have one entry per joint.
    pub fn points(&self, angles: &[JointAngles]) -> Vec<Vec3> {
        let mut pos = self.mount;
        let mut rot = Mat3::rot_y(self.mount_yaw);
        let mut out = Vec::with_capacity(self.segments.len() + 1);
        out.push(pos);
        for ((spec, len), a) in self.joints.iter().zip(&self.segments).zip(angles) {
            rot = rot.mul(Self::joint_rotation(spec, *a));
            pos = pos + rot.apply(Vec3::new(0.0, 0.0, *len));
            out.push(pos);
        }
        out
    }
}

/// Points along the finger used for contact tests: every joint point plus
/// the midpoint of each segment, so a thin object cannot slip between joints.
pub fn sample_points(points: &[Vec3]) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(points.len() * 2);
    for pair in points.windows(2) {
        out.push(pair[0]);
        out.push((pair[0] + pair[1]).scale(0.5));
    }
    if let Some(last) = points.last() {
        out.push(*last);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::JointKind;
    use std::f32::consts::FRAC_PI_2;

    fn hinge(max_flex: f32) -> JointSpec {
        JointSpec {
            kind: JointKind::Hinge,
            flex_min: 0.0,
            flex_max: max_flex,
            abd_limit: 0.0,
            roll_limit: 0.0,
            max_speed: 6.0,
            max_torque: 0.5,
        }
    }

    fn straight_finger() -> FingerSpec {
        FingerSpec {
            name: "test".into(),
            is_thumb: false,
            mount: Vec3::ZERO,
            mount_yaw: 0.0,
            segments: vec![0.04, 0.03],
            joints: vec![hinge(90.0_f32.to_radians()), hinge(90.0_f32.to_radians())],
        }
    }

    #[test]
    fn straight_finger_tip_is_total_reach_along_z() {
        let f = straight_finger();
        let pts = f.points(&[JointAngles::default(); 2]);
        let tip = *pts.last().unwrap();
        assert!((tip.z - 0.07).abs() < 1e-5);
        assert!(tip.y.abs() < 1e-5);
    }

    #[test]
    fn quarter_flexion_bends_tip_toward_palm_normal() {
        let f = straight_finger();
        let a = JointAngles {
            flex: FRAC_PI_2,
            ..Default::default()
        };
        let pts = f.points(&[a, JointAngles::default()]);
        // First segment points up (+Y) after a 90 degree flexion.
        assert!((pts[1].y - 0.04).abs() < 1e-5);
        assert!(pts[1].z.abs() < 1e-5);
    }

    #[test]
    fn sample_points_interleave_midpoints() {
        let pts = vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0)];
        let s = sample_points(&pts);
        assert_eq!(s.len(), 5);
        assert!((s[1].z - 0.5).abs() < 1e-6);
        assert!((s[4].z - 2.0).abs() < 1e-6);
    }
}
