//! What the robot's own motion sensors tell it about where it is.
//!
//! The world moves the robot exactly along its route. The robot only knows its
//! pose by dead reckoning (wheel encoders or the legs' kinematics, and a gyro
//! for the heading), which drifts. [`Odometry`] keeps the pose the robot
//! believes it has and grows its error with every bit of motion:
//!
//! * along the path, a scale error drawn per move (the wheel odometer's or
//!   the leg kinematics' own error), plus the slip the drive physics gives on
//!   that ground (a spinning tyre or track counts distance it did not travel,
//!   `noto_core::act::cruise_slip`) and the foot slides a gait draws;
//! * a random walk in position that grows with the square root of the distance;
//! * in heading, the gyro's angle random walk and the bias left after its
//!   start-up calibration.
//!
//! Everything the robot remembers seeing is stored in this believed frame
//! (`World::belief_of`), and everything it decides to do there is carried out
//! by moving its real body as far as it thinks it must (`World::true_of`).
//! The error of a remembered position relative to the robot is therefore the
//! drift since it was seen, which is what a look takes away again.
//!
//! Numbers (research §1, group A; none of them is a benchmark source):
//! Wheel-INS (Niu et al., arXiv:1912.07805, Table III) for a wheel odometer
//! with a MEMS IMU outdoors, Borenstein & Feng 1996 (UMBmark, IEEE T-RA 12,
//! p. 21) for its random part, Camurri et al. 2020 (Pronto, Front. Robot. AI
//! 7:68, §7.2) for leg kinematics with an IMU, and the Allan analysis of a
//! head camera's IMU in MILUV (arXiv 2504.14376, Tables 3–4). Slip that
//! over-counts distance follows González, Rodríguez & Guzmán 2015 (RIAI 12,
//! §3.1: i = 1 − v/(rω)) through the drive model.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// Scale error of a wheel odometer fused with a MEMS gyro, share of the
/// distance, drawn per move (Wheel-INS Table III: mean drift 0.58% (SD 0.45)
/// per 100 m on campus roads; the 2.2% of bumpy farm roads is mostly slip,
/// which the drive model adds on its own).
pub const WHEEL_SCALE_SD: f32 = 0.0058;
/// Non-systematic position error per √metre, metres (Borenstein & Feng 1996
/// p. 21: run-to-run SD ≈ 25 mm over the 16 m path, as a random walk).
pub const RANDOM_WALK_M: f32 = 0.025 / 4.0;
/// Leg kinematics with an IMU: position error over the distance walked
/// (Camurri et al. 2020 §7.2: Atlas without lidar, ≈1.67% of the distance).
pub const LEG_SCALE_SD: f32 = 0.0167;
/// The head IMU's gyro angle random walk, rad/√s (MILUV Tables 3–4, a D435i:
/// 0.0053–0.0084 °/√s; the middle).
pub const GYRO_ARW_RAD: f32 = 0.0069 * std::f32::consts::PI / 180.0;
/// Bias left after the start-up calibration, rad/s, 1σ (MILUV: bias
/// instability 1.6–5.4 °/h; the middle).
pub const GYRO_BIAS_RAD_S: f32 = 3.5 / 3600.0 * std::f32::consts::PI / 180.0;

/// Odometry settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OdomCfg {
    /// Off: the robot knows its pose exactly (as before v0.7).
    pub enabled: bool,
    /// Scales every error source (1 = the numbers above).
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

/// The pose the robot believes it has, and how it drifts.
#[derive(Clone, Debug)]
pub struct Odometry {
    pub cfg: OdomCfg,
    /// Believed position and heading (x, z, yaw).
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    /// Distance and time travelled since the last fix (a start on a pad or a correction).
    pub since_fix_m: f32,
    pub since_fix_s: f32,
    /// Total distance travelled.
    pub travelled_m: f32,
    legged: bool,
    /// The odometer's scale error over the current move.
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
    /// Odometry that starts where the robot really is (it starts on a surveyed pad).
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

    /// A new move starts: the odometer's scale error for it is drawn.
    pub fn new_move(&mut self) {
        self.scale_err = self.k() * if self.legged { LEG_SCALE_SD } else { WHEEL_SCALE_SD } * gauss(&mut self.rng);
    }

    /// Integrates one step of the true motion from `from` to `to` (x, z, yaw)
    /// over `dt` seconds. `over` is the share of the distance the wheels
    /// counted without travelling it (their slip ratio over one minus it).
    pub fn update(&mut self, from: (f32, f32, f32), to: (f32, f32, f32), dt: f32, over: f32) {
        let (dx, dz) = (to.0 - from.0, to.1 - from.1);
        let d = dx.hypot(dz);
        let dpsi = wrap(to.2 - from.2);
        let k = self.k();
        if k <= 0.0 {
            // Exact: the belief follows the truth.
            self.x += dx;
            self.z += dz;
            self.yaw = wrap(self.yaw + dpsi);
        } else {
            // The motion in the robot's own frame, as its sensors count it.
            let (s, c) = from.2.sin_cos();
            let fwd = dx * c + dz * s;
            let lat = -dx * s + dz * c;
            let rw = k * RANDOM_WALK_M * d.sqrt();
            let fwd_b = fwd * (1.0 + self.scale_err + over.max(0.0)) + rw * gauss(&mut self.rng);
            let lat_b = lat + rw * gauss(&mut self.rng);
            // Laid out along the heading it believes it had over the step.
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

    /// A foot slid `slide_m` back under the body without the legs' kinematics
    /// seeing it: the robot believes it went that much further.
    pub fn foot_slid(&mut self, slide_m: f32) {
        if self.k() <= 0.0 {
            return;
        }
        let (s, c) = self.yaw.sin_cos();
        self.x += slide_m * c;
        self.z += slide_m * s;
    }

    /// The believed pose.
    pub fn pose(&self) -> (f32, f32, f32) {
        (self.x, self.z, self.yaw)
    }

    /// A point of the true world in the believed frame, given the true pose:
    /// where the robot puts what it sees there.
    pub fn belief_of(&self, truth: (f32, f32, f32), p: (f32, f32)) -> (f32, f32) {
        let th = wrap(self.yaw - truth.2);
        let (s, c) = th.sin_cos();
        let (rx, rz) = (p.0 - truth.0, p.1 - truth.1);
        (self.x + rx * c - rz * s, self.z + rx * s + rz * c)
    }

    /// A believed point in the true world, given the true pose: where the
    /// robot's body goes when it goes where it believes that point is.
    pub fn true_of(&self, truth: (f32, f32, f32), q: (f32, f32)) -> (f32, f32) {
        let th = wrap(truth.2 - self.yaw);
        let (s, c) = th.sin_cos();
        let (rx, rz) = (q.0 - self.x, q.1 - self.z);
        (truth.0 + rx * c - rz * s, truth.1 + rx * s + rz * c)
    }

    /// A believed heading as a true one.
    pub fn true_yaw(&self, truth_yaw: f32, believed_yaw: f32) -> f32 {
        wrap(believed_yaw - (self.yaw - truth_yaw))
    }

    /// The robot corrects its believed position by what a known landmark
    /// showed (the map's place of a waypoint against where it saw it).
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
        // Square loops: `legs` straight runs of `len` m with a 90° turn after each, at 0.5 m/s.
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
        // The 4 m square of UMBmark, many robots: the return error is centimetres, as after calibration.
        let errs: Vec<f32> = (0..200u64)
            .map(|seed| {
                let mut o = Odometry::new(OdomCfg::default(), seed, false, (0.0, 0.0, 0.0));
                let t = walk(&mut o, 4, 4.0, 0.0);
                (o.x - t.0).hypot(o.z - t.1)
            })
            .collect();
        let mean = errs.iter().sum::<f32>() / errs.len() as f32;
        assert!(mean > 0.005 && mean < 0.08, "{mean}");
        // Slip over-counts: a 10% slip makes the robot believe it went about 11% further.
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
        // Leg kinematics drift about three times the wheel odometer per metre (1.67% against 0.58%).
        assert!(l > 1.8 * w && l < 5.0 * w, "legs {l} wheels {w}");
        // A true point seen now and put back is where it was.
        let mut o = Odometry::new(OdomCfg::default(), 3, true, (1.0, 2.0, 0.3));
        let truth = walk(&mut o, 3, 5.0, 0.0);
        let p = (4.0, -1.0);
        let b = o.belief_of(truth, p);
        let back = o.true_of(truth, b);
        assert!((back.0 - p.0).abs() < 1e-4 && (back.1 - p.1).abs() < 1e-4);
        // The robot's own believed position is its true one in the believed frame.
        let me = o.belief_of(truth, (truth.0, truth.1));
        assert!((me.0 - o.x).abs() < 1e-4 && (me.1 - o.z).abs() < 1e-4);
    }
}
