//! The camera as the ML logic knows it: intrinsics (size, field of view,
//! pitch) and the robot's odometry pose. Nothing about the world.
//!
//! The conventions match `world_sim::Camera` so a pixel means the same ray on
//! both sides: x and z on the ground, y up, heading 0 along +x and 90 degrees
//! along +z, the camera's right is `(-sin yaw, 0, cos yaw)`. Depth is the
//! distance along the camera's forward axis, as the renderer writes it.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    pub width: u32,
    pub height: u32,
    /// Horizontal field of view, degrees.
    pub fov_deg: f32,
    /// Pitch, degrees; negative looks down.
    pub pitch_deg: f32,
}

/// Where the camera was when a keyframe was taken, from the robot's own
/// odometry (not from the world).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub x: f32,
    pub z: f32,
    /// Heading, radians.
    pub yaw: f32,
    /// Camera height above the ground under the robot, metres.
    pub eye_m: f32,
}

fn basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    ([cy * cp, sp, sy * cp], [-sy, 0.0, cy], [-cy * sp, cp, -sy * sp])
}

pub fn wrap_angle(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

impl Intrinsics {
    fn screen(&self, px: f32, py: f32) -> (f32, f32) {
        let (w, h) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        let half_h = half_w * h / w;
        ((2.0 * px / w - 1.0) * half_w, (1.0 - 2.0 * py / h) * half_h)
    }

    /// Ray through pixel `(px, py)` (continuous pixel coordinates) scaled so
    /// that its forward component is 1: `depth * ray` is the hit point
    /// relative to the camera.
    pub fn ray(&self, yaw: f32, px: f32, py: f32) -> [f32; 3] {
        let (fwd, right, up) = basis(yaw, self.pitch_deg.to_radians());
        let (sx, sy) = self.screen(px, py);
        [fwd[0] + right[0] * sx + up[0] * sy, fwd[1] + right[1] * sx + up[1] * sy, fwd[2] + right[2] * sx + up[2] * sy]
    }

    /// World heading of the ground-plane direction seen at a pixel, radians.
    /// It depends a little on the row because the camera is pitched.
    pub fn bearing(&self, pose: &Pose, px: f32, py: f32) -> f32 {
        let r = self.ray(pose.yaw, px, py);
        r[2].atan2(r[0])
    }

    /// Bearing relative to the robot's heading, radians (positive to the camera's right).
    pub fn relative_bearing(&self, pose: &Pose, px: f32, py: f32) -> f32 {
        wrap_angle(self.bearing(pose, px, py) - pose.yaw)
    }

    /// Ground-plane distance to the point seen at a pixel with forward depth `depth_m`.
    pub fn ground_range(&self, yaw: f32, px: f32, py: f32, depth_m: f32) -> f32 {
        let r = self.ray(yaw, px, py);
        depth_m * r[0].hypot(r[2])
    }

    pub fn with_pitch(&self, pitch_deg: f32) -> Intrinsics {
        Intrinsics { pitch_deg, ..*self }
    }

    pub fn vfov_deg(&self) -> f32 {
        let (w, h) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        2.0 * (half_w * h / w).atan().to_degrees()
    }

    pub fn look_down_pitch_deg(&self, eye_m: f32, range_m: f32, target_h_m: f32) -> f32 {
        let dz = (eye_m - target_h_m).max(0.05);
        let direct_deg = -dz.atan2(range_m.max(0.15)).to_degrees();
        let half_vfov = self.vfov_deg() * 0.5;
        let min_needed = direct_deg + half_vfov * 0.65;
        if self.pitch_deg > min_needed {
            direct_deg.clamp(-65.0, self.pitch_deg)
        } else {
            self.pitch_deg
        }
    }

    pub fn is_ground_target_visible(&self, eye_m: f32, range_m: f32, target_h_m: f32) -> bool {
        let dz = (eye_m - target_h_m).max(0.0);
        let target_pitch_deg = -dz.atan2(range_m.max(0.05)).to_degrees();
        let half_vfov = self.vfov_deg() * 0.5;
        let bottom_edge_deg = self.pitch_deg - half_vfov * 0.85;
        let top_edge_deg = self.pitch_deg + half_vfov * 0.85;
        target_pitch_deg >= bottom_edge_deg && target_pitch_deg <= top_edge_deg
    }

    pub fn depth_for_range(&self, yaw: f32, px: f32, py: f32, range_m: f32) -> f32 {
        let r = self.ray(yaw, px, py);
        range_m / r[0].hypot(r[2]).max(1e-6)
    }

    pub fn ground_point(&self, pose: &Pose, px: f32, py: f32, depth_m: f32) -> Option<(f32, f32)> {
        if !depth_m.is_finite() || depth_m <= 0.0 {
            return None;
        }
        let r = self.ray(pose.yaw, px, py);
        Some((pose.x + depth_m * r[0], pose.z + depth_m * r[2]))
    }

    /// Pixel of a ground-level point at height `y_rel` relative to the camera,
    /// or `None` behind the camera.
    pub fn project(&self, pose: &Pose, x: f32, y_rel: f32, z: f32) -> Option<(f32, f32, f32)> {
        let (w, h) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let (fwd, right, up) = basis(pose.yaw, self.pitch_deg.to_radians());
        let rel = [x - pose.x, y_rel, z - pose.z];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let depth = dot(rel, fwd);
        if depth <= 1e-4 {
            return None;
        }
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        let half_h = half_w * h / w;
        Some(((dot(rel, right) / depth / half_w + 1.0) * 0.5 * w, (1.0 - dot(rel, up) / depth / half_h) * 0.5 * h, depth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Intrinsics {
        Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0 }
    }

    #[test]
    fn the_centre_column_looks_straight_ahead() {
        let pose = Pose { x: 1.0, z: 2.0, yaw: 0.7, eye_m: 0.5 };
        let b = cam().bearing(&pose, 192.0, 128.0);
        assert!((b - 0.7).abs() < 1e-5);
        assert!(cam().relative_bearing(&pose, 300.0, 128.0) > 0.0, "the right half of the frame is to the right");
    }

    #[test]
    fn project_and_ground_point_are_inverse() {
        let pose = Pose { x: 3.0, z: -1.0, yaw: -0.4, eye_m: 0.5 };
        let c = cam();
        let (px, py, depth) = c.project(&pose, 7.0, -0.5, 1.5).unwrap();
        let (x, z) = c.ground_point(&pose, px, py, depth).unwrap();
        assert!((x - 7.0).abs() < 1e-3 && (z - 1.5).abs() < 1e-3, "{x} {z}");
        let range = c.ground_range(pose.yaw, px, py, depth);
        assert!((range - (4.0f32).hypot(2.5)).abs() < 1e-3);
        assert!((c.depth_for_range(pose.yaw, px, py, range) - depth).abs() < 1e-3);
    }
}
