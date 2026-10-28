use serde::{Deserialize, Serialize};

use crate::rng::Rng;

pub const WHEEL_SCALE_SD: f32 = 0.0058;
pub const RANDOM_WALK_M: f32 = 0.025 / 4.0;
pub const LEG_SCALE_SD: f32 = 0.0167;
pub const GYRO_ARW_RAD: f32 = 0.0069 * std::f32::consts::PI / 180.0;
pub const GYRO_BIAS_RAD_S: f32 = 3.5 / 3600.0 * std::f32::consts::PI / 180.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OdomCfg {
    pub enabled: bool,
    pub scale: f32,
}

impl Default for OdomCfg {
    fn default() -> Self {
        OdomCfg { enabled: true, scale: 1.0 }
    }
}

impl OdomCfg {
    pub fn exact() -> OdomCfg {
        OdomCfg { enabled: false, scale: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct Odometry {
    pub cfg: OdomCfg,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub since_fix_m: f32,
    pub since_fix_s: f32,
    pub travelled_m: f32,
    legged: bool,
    scale_err: f32,
    gyro_bias: f32,
    rng: Rng,
}

fn gauss(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

impl Odometry {
    pub fn new(cfg: OdomCfg, seed: u64, legged: bool, start: (f32, f32, f32)) -> Odometry {
        let mut rng = Rng::new(seed ^ 0x0D0_E7E7_5EED_2B1D);
        let k = if cfg.enabled { cfg.scale } else { 0.0 };
        let scale_err = k * if legged { LEG_SCALE_SD } else { WHEEL_SCALE_SD } * gauss(&mut rng);
        let gyro_bias = k * GYRO_BIAS_RAD_S * gauss(&mut rng);
        Odometry { cfg, x: start.0, z: start.1, yaw: start.2, since_fix_m: 0.0, since_fix_s: 0.0, travelled_m: 0.0, legged, scale_err, gyro_bias, rng }
    }

    fn k(&self) -> f32 {
        if self.cfg.enabled {
            self.cfg.scale
        } else {
            0.0
        }
    }

    pub fn new_move(&mut self) {
        self.scale_err = self.k() * if self.legged { LEG_SCALE_SD } else { WHEEL_SCALE_SD } * gauss(&mut self.rng);
    }

    pub fn update(&mut self, from: (f32, f32, f32), to: (f32, f32, f32), dt: f32, over: f32) {
        let (dx, dz) = (to.0 - from.0, to.1 - from.1);
        let d = dx.hypot(dz);
        let dpsi = wrap(to.2 - from.2);
        let k = self.k();
        if k <= 0.0 {
            self.x += dx;
            self.z += dz;
            self.yaw = wrap(self.yaw + dpsi);
        } else {
            let (s, c) = from.2.sin_cos();
            let fwd = dx * c + dz * s;
            let lat = -dx * s + dz * c;
            let rw = k * RANDOM_WALK_M * d.sqrt();
            let fwd_b = fwd * (1.0 + self.scale_err + over.max(0.0)) + rw * gauss(&mut self.rng);
            let lat_b = lat + rw * gauss(&mut self.rng);
            let psi = self.yaw + 0.5 * dpsi;
            let (sb, cb) = psi.sin_cos();
            self.x += fwd_b * cb - lat_b * sb;
            self.z += fwd_b * sb + lat_b * cb;
            let gyro = self.gyro_bias * dt + k * GYRO_ARW_RAD * dt.max(0.0).sqrt() * gauss(&mut self.rng);
            self.yaw = wrap(self.yaw + dpsi + gyro);
        }
        self.since_fix_m += d;
        self.since_fix_s += dt;
        self.travelled_m += d;
    }

    pub fn foot_slid(&mut self, slide_m: f32) {
        if self.k() <= 0.0 {
            return;
        }
        let (s, c) = self.yaw.sin_cos();
        self.x += slide_m * c;
        self.z += slide_m * s;
    }

    pub fn pose(&self) -> (f32, f32, f32) {
        (self.x, self.z, self.yaw)
    }

    pub fn belief_of(&self, truth: (f32, f32, f32), p: (f32, f32)) -> (f32, f32) {
        let th = wrap(self.yaw - truth.2);
        let (s, c) = th.sin_cos();
        let (rx, rz) = (p.0 - truth.0, p.1 - truth.1);
        (self.x + rx * c - rz * s, self.z + rx * s + rz * c)
    }

    pub fn true_of(&self, truth: (f32, f32, f32), q: (f32, f32)) -> (f32, f32) {
        let th = wrap(truth.2 - self.yaw);
        let (s, c) = th.sin_cos();
        let (rx, rz) = (q.0 - self.x, q.1 - self.z);
        (truth.0 + rx * c - rz * s, truth.1 + rx * s + rz * c)
    }

    pub fn true_yaw(&self, truth_yaw: f32, believed_yaw: f32) -> f32 {
        wrap(believed_yaw - (self.yaw - truth_yaw))
    }

    pub fn correct(&mut self, dx: f32, dz: f32) {
        self.x += dx;
        self.z += dz;
        self.since_fix_m = 0.0;
        self.since_fix_s = 0.0;
    }
}

fn wrap(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walk(o: &mut Odometry, legs: usize, len: f32, over: f32) -> (f32, f32, f32) {
        let mut t = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..legs {
            o.new_move();
            let n = (len / 0.05).round() as usize;
            for _ in 0..n {
                let next = (t.0 + 0.05 * t.2.cos(), t.1 + 0.05 * t.2.sin(), t.2);
                o.update(t, next, 0.1, over);
                t = next;
            }
            let next = (t.0, t.1, wrap(t.2 + std::f32::consts::FRAC_PI_2));
            o.update(t, next, 0.6, 0.0);
            t = next;
        }
        t
    }

    #[test]
    fn exact_odometry_follows_the_truth() {
        let mut o = Odometry::new(OdomCfg::exact(), 1, false, (0.0, 0.0, 0.0));
        let t = walk(&mut o, 4, 4.0, 0.0);
        assert!((o.x - t.0).abs() < 1e-4 && (o.z - t.1).abs() < 1e-4 && (o.yaw - t.2).abs() < 1e-5);
        assert!((o.travelled_m - 16.0).abs() < 1e-3);
    }

    #[test]
    fn wheel_drift_over_the_umbmark_square_is_centimetres() {
        let errs: Vec<f32> = (0..200u64)
            .map(|seed| {
                let mut o = Odometry::new(OdomCfg::default(), seed, false, (0.0, 0.0, 0.0));
                let t = walk(&mut o, 4, 4.0, 0.0);
                (o.x - t.0).hypot(o.z - t.1)
            })
            .collect();
        let mean = errs.iter().sum::<f32>() / errs.len() as f32;
        assert!(mean > 0.005 && mean < 0.08, "{mean}");
        let mut o = Odometry::new(OdomCfg::exact(), 1, false, (0.0, 0.0, 0.0));
        o.cfg = OdomCfg::default();
        let mut t = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..100 {
            let next = (t.0 + 0.05, t.1, 0.0);
            o.update(t, next, 0.1, 0.1 / 0.9);
            t = next;
        }
        assert!((o.x - 5.0 * (1.0 + 0.1 / 0.9)).abs() < 0.1, "{}", o.x);
    }

    #[test]
    fn legs_drift_more_than_wheels_and_the_frames_invert() {
        let spread = |legged: bool| {
            let errs: Vec<f32> = (0..200u64)
                .map(|seed| {
                    let mut o = Odometry::new(OdomCfg::default(), seed, legged, (0.0, 0.0, 0.0));
                    let t = walk(&mut o, 1, 10.0, 0.0);
                    (o.x - t.0).hypot(o.z - t.1)
                })
                .collect();
            errs.iter().sum::<f32>() / errs.len() as f32
        };
        let (w, l) = (spread(false), spread(true));
        assert!(l > 1.8 * w && l < 5.0 * w, "legs {l} wheels {w}");
        let mut o = Odometry::new(OdomCfg::default(), 3, true, (1.0, 2.0, 0.3));
        let truth = walk(&mut o, 3, 5.0, 0.0);
        let p = (4.0, -1.0);
        let b = o.belief_of(truth, p);
        let back = o.true_of(truth, b);
        assert!((back.0 - p.0).abs() < 1e-4 && (back.1 - p.1).abs() < 1e-4);
        let me = o.belief_of(truth, (truth.0, truth.1));
        assert!((me.0 - o.x).abs() < 1e-4 && (me.1 - o.z).abs() < 1e-4);
    }
}
