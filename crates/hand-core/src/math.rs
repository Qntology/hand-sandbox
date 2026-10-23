//! Minimal 3D math used by the kinematics: vectors and rotation matrices.
//!
//! Conventions: right-handed frame, angles in radians, rotations are active
//! (they rotate points counter-clockwise about the named axis).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn distance(self, other: Vec3) -> f32 {
        (self - other).length()
    }

    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn normalized(self) -> Vec3 {
        let len = self.length();
        if len <= f32::EPSILON {
            Vec3::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/// Row-major 3x3 rotation matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub m: [[f32; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub fn rot_x(angle: f32) -> Mat3 {
        let (s, c) = angle.sin_cos();
        Mat3 {
            m: [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        }
    }

    pub fn rot_y(angle: f32) -> Mat3 {
        let (s, c) = angle.sin_cos();
        Mat3 {
            m: [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        }
    }

    pub fn rot_z(angle: f32) -> Mat3 {
        let (s, c) = angle.sin_cos();
        Mat3 {
            m: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn mul(self, rhs: Mat3) -> Mat3 {
        let mut out = [[0.0f32; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.m[i][k] * rhs.m[k][j]).sum();
            }
        }
        Mat3 { m: out }
    }

    pub fn apply(self, v: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-5
    }

    #[test]
    fn rotations_map_axes_as_expected() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        // Abduction: positive angle swings the +Z axis toward +X.
        assert!(approx(
            Mat3::rot_y(std::f32::consts::FRAC_PI_2).apply(z),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        // Flexion (rot_x with negative angle) swings +Z toward +Y.
        assert!(approx(
            Mat3::rot_x(-std::f32::consts::FRAC_PI_2).apply(z),
            Vec3::new(0.0, 1.0, 0.0)
        ));
    }

    #[test]
    fn matrix_product_composes_rotations() {
        let a = Mat3::rot_z(0.3).mul(Mat3::rot_z(0.4));
        let b = Mat3::rot_z(0.7);
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(approx(a.apply(v), b.apply(v)));
    }
}
