//! The head camera at this instant (v0.8): where the robot's eyes are, how
//! they are turned, how they shake, and what the robot believes about it.
//! Numbers and their sources: `docs/research/camera.md` (§6 lists them).
//!
//! * Pose. The camera rides on the head (`noto_core::kin::head_camera`) of
//!   the pose the robot holds now: the playing action's sample (a bend, a
//!   reach that looks at the hand, a pan toward an item beside the foot), the
//!   walking step (the CoM's rise and fall, the pelvis's sway: the head's bob
//!   needs no constant of its own), or standing. A wheeled base pitches with
//!   the slope it stands on; the neck takes back its preset's share of that
//!   (`HeadPreset::gaze_stabilization`).
//! * Torso sway (legs). The trunk pitches at the step frequency and rolls
//!   at the stride frequency, 4° peak to peak each (a person's trunk pitch,
//!   Cromwell & Wellmon 2001; the roll taken equal), less the share the neck
//!   cancels.
//! * Heel strike (legs). The head takes a half-sine pulse of 0.5 g when a
//!   heel lands on hard ground (a person's skull with hard heels, Light 1980)
//!   lasting 33 ms (the middle of the 9–20 Hz impact band). A softer ground
//!   lowers the peak and stretches the pulse by its strike scale `S` (the
//!   foot pad and the ground in series; the velocity change stays); a pulse
//!   longer than the double support is not felt as a strike. The camera's
//!   mount on the head rings with it (its relative motion, the response of a
//!   damped oscillator to the pulse), and because the heel lands half a step
//!   ahead of the CoM and half the stance to the side, the kick also pitches
//!   the body back and rolls it away from the landing foot.
//! * Rough ground. Each surface has an ISO 8608 roughness class (a spatial
//!   PSD of height), drawn as two parallel profiles, left and right. A rigid
//!   wheeled chassis follows them at its four wheels (heave, pitch over the
//!   wheelbase, roll over the track) down to the bumps its wheel diameter
//!   bridges; a track bridges everything shorter than its contact; each foot
//!   lands on the profile where it steps, which tilts the body by the height
//!   between this footfall and the last one.
//! * Sensor. The head preset's RGB sensor has a rolling shutter (rows read
//!   one after another over the readout) and an exposure set by the light
//!   (a reflected-light meter's law, `t = 0.5 / L`); `render` turns the
//!   camera's angular rate into the skew and the blur they make.
//! * Belief. The robot knows its joints (encoders) and measures the shake
//!   with the head camera's own IMU: the gyro integrates it, so the believed
//!   view carries the gyro's angle random walk over the time the shake has
//!   been integrated. Back-projection (`noto-runtime`) uses the believed view.

use noto_core::kin::{self as hkin, KinPose};
use noto_core::surface::Surface;
use serde::Serialize;

use super::World;
use crate::render::{Camera, Exposure, View};
use crate::rng::hash2;

/// What the head camera tracks between ticks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EyeState {
    /// Distance the base has travelled over the ground, metres (true, not the odometer).
    pub travelled_m: f32,
    /// Heading rate over the last tick, rad/s (turning in place or on a curve).
    pub yaw_rate: f32,
}

impl EyeState {
    pub(super) fn advance(&mut self, before: (f32, f32, f32), after: (f32, f32, f32), dt: f32) {
        self.travelled_m += (after.0 - before.0).hypot(after.1 - before.1);
        let mut d = (after.2 - before.2) % std::f32::consts::TAU;
        if d > std::f32::consts::PI {
            d -= std::f32::consts::TAU;
        } else if d < -std::f32::consts::PI {
            d += std::f32::consts::TAU;
        }
        self.yaw_rate = if dt > 0.0 { d / dt } else { 0.0 };
    }
}

/// The camera's motion on top of the body's pose now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Shake {
    /// The largest part: `sway`, `heel_strike`, `footfall`, `rolling`, or `still`.
    pub source: &'static str,
    /// Vertical displacement, metres; pitch (nose up positive) and roll (right side down positive), radians.
    pub dz_m: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Rates of the three (m/s, rad/s, rad/s).
    pub rate: [f32; 3],
}

impl Shake {
    fn still() -> Shake {
        Shake { source: "still", ..Shake::default() }
    }

    fn size(&self) -> f32 {
        self.pitch.abs() + self.roll.abs() + self.dz_m.abs()
    }

    /// The sum of two parts, named after the larger.
    fn add(self, o: Shake) -> Shake {
        let source = if o.size() > self.size() || self.source == "still" { o.source } else { self.source };
        Shake {
            source,
            dz_m: self.dz_m + o.dz_m,
            pitch: self.pitch + o.pitch,
            roll: self.roll + o.roll,
            rate: [self.rate[0] + o.rate[0], self.rate[1] + o.rate[1], self.rate[2] + o.rate[2]],
        }
    }
}

/// The head camera now, as it is and as the robot believes it is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct HeadView {
    pub view: View,
    pub exposure: Exposure,
    /// The view the robot back-projects with (joints exact, the shake as its gyro integrated it).
    pub est: View,
    pub shake: Shake,
    /// Eye height above the ground under the robot, metres.
    pub eye_m: f32,
    /// Head pitch below level from the joints (bend and neck), degrees, and the neck's pan, degrees (left positive).
    pub head_pitch_deg: f32,
    pub pan_deg: f32,
    /// Largest motion blur in this frame, pixels.
    pub blur_px: f32,
    /// Pitch (nose up) and roll (right side down) of a wheeled or tracked
    /// chassis on the ground under it, degrees (0 for legs: the gait keeps
    /// the torso level).
    pub tilt_deg: [f32; 2],
}

// ----- the sensor of each head preset (research camera.md §4) -----

/// The head camera's RGB sensor and IMU as the simulator uses them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SensorTiming {
    pub rgb: &'static str,
    /// Exposure at the scene's light, s.
    pub exposure_s: f32,
    /// Rolling-shutter readout of a frame, s.
    pub readout_s: f32,
    /// Gyro angle random walk, degrees per √s (its noise density).
    pub gyro_arw_deg: f32,
}

/// Illuminance of the simulated scene, lux (est: bright overcast daylight,
/// the renderer's soft sky without hard shadows; research §4.5's table).
pub const SCENE_LUX: f32 = 10_000.0;
/// Readout of a rolling-shutter frame, s (est: at most the 16.7 ms frame
/// period of the 1080p 60 fps modes both RGB sensors have; research §4.4).
pub const RGB_READOUT_S: f32 = 0.0167;
/// Longest exposure, s: the 33 ms frame period at 30 fps (research §4.5).
pub const EXPOSURE_MAX_S: f32 = 1.0 / 30.0;

/// Auto-exposure time at an illuminance, s: `N²/t = L·S/K` with K = 12.5
/// (ISO 2720), f/2.0, ISO 100, and the luminance of an 18% grey Lambertian
/// face `L = 0.18·E/π`, which gives `t = 0.5/L` (research §4.5).
pub fn exposure_s(lux: f32) -> f32 {
    let luminance = 0.18 * lux.max(1.0) / std::f32::consts::PI;
    (0.5 / luminance).min(EXPOSURE_MAX_S)
}

/// The head camera's RGB timing and gyro (research §4.1–4.2: the D435i's
/// OV2740 and BMI055, the OAK-D Lite's IMX214 and BMI270).
pub fn sensor_timing(sensor: &str) -> SensorTiming {
    let exposure = exposure_s(SCENE_LUX);
    if sensor.contains("OAK") {
        SensorTiming { rgb: "IMX214 (rolling)", exposure_s: exposure, readout_s: RGB_READOUT_S, gyro_arw_deg: 0.010 }
    } else {
        SensorTiming { rgb: "OV2740 (rolling)", exposure_s: exposure, readout_s: RGB_READOUT_S, gyro_arw_deg: 0.014 }
    }
}

// ----- walking (research camera.md §1, §2) -----

/// Trunk pitch over a step and roll over a stride, peak to peak, degrees
/// (pitch: a person's trunk, Cromwell & Wellmon 2001; roll est equal).
pub const SWAY_PITCH_PP_DEG: f32 = 4.0;
pub const SWAY_ROLL_PP_DEG: f32 = 4.0;
/// Head acceleration of a heel strike on hard ground, m/s² (0.5 g: a
/// person's skull with hard heels, Light 1980), and its half-sine length, s (est).
pub const STRIKE_PEAK_MPS2: f32 = 0.5 * 9.81;
pub const STRIKE_PULSE_S: f32 = 0.033;
/// The camera mount's ringing on the head after a strike: frequency, Hz
/// (est: the geometric middle of the 9–20 Hz impact band) and damping ratio (est).
pub const MOUNT_HZ: f32 = 13.4;
pub const MOUNT_ZETA: f32 = 0.30;
/// Whole-body radius of gyration about the CoM for pitch and roll, as a share of the eye height (est).
pub const GYRATION_SHARE: f32 = 0.25;
/// How long the IMU integrates the shake of a rolling base before the
/// gravity reference takes it back, s (est: a complementary filter's constant).
pub const IMU_WINDOW_S: f32 = 1.0;

/// Strike scale of a ground: the peak of the heel-strike pulse against a
/// hard floor's, and the inverse of its length (research §2.3).
pub fn strike_scale(ground: Surface) -> f32 {
    match ground {
        Surface::Firm | Surface::Rock => 1.00,
        Surface::Gravel => 0.92,
        Surface::Dirt => 0.75,
        Surface::WetSand => 0.68,
        Surface::Grass => 0.36,
        Surface::Sand | Surface::Water => 0.17,
    }
}

// ----- ground roughness (research camera.md §3) -----

/// ISO 8608 displacement PSD at `n0 = 0.1` cycles/m, m³ (the class's
/// geometric mean, `16·4^k × 10⁻⁶`; the class of each ground is est, §3.3).
pub fn roughness_gd(ground: Surface) -> f32 {
    let class = |k: i32| 16.0e-6 * 4f32.powi(k);
    match ground {
        Surface::Firm | Surface::WetSand => class(1),
        Surface::Gravel | Surface::Grass | Surface::Water => class(2),
        Surface::Dirt | Surface::Sand => class(3),
        Surface::Rock => class(5),
    }
}

/// Reference spatial frequency of ISO 8608, cycles/m.
const N0: f32 = 0.1;
/// Spatial frequencies the profile is drawn over, cycles/m: from 10 m (the
/// terrain grid already holds anything longer) to 2 cm.
const N_LO: f32 = 0.1;
const N_HI: f32 = 50.0;
/// Sinusoids per profile.
const N_WAVES: usize = 48;

/// Height of the ground profile (left `side` 0, right 1) at distance `s`
/// along the path, metres, and its slope dh/ds, with the bumps shorter than
/// `bridge_m` averaged out (the contact bridges them: `sinc(π·n·ℓ)`, whose
/// first zero is the `1/ℓ` cut-off of research §3.2).
pub fn profile(seed: u64, gd: f32, side: u32, s: f32, bridge_m: f32) -> (f32, f32) {
    let ratio = (N_HI / N_LO).powf(1.0 / (N_WAVES - 1) as f32);
    let (mut h, mut dh) = (0.0f32, 0.0f32);
    let mut n = N_LO;
    for k in 0..N_WAVES {
        let dn = n * (ratio.sqrt() - 1.0 / ratio.sqrt());
        let g = gd * (n / N0).powi(-2);
        let x = std::f32::consts::PI * n * bridge_m;
        let bridge = if x < 1e-4 { 1.0 } else { (x.sin() / x).abs() };
        let a = (2.0 * g * dn).sqrt() * bridge;
        let phase = (hash2(seed ^ 0x6E7A_10F3, side as i32, k as i32) % 10_000) as f32 / 10_000.0 * std::f32::consts::TAU;
        let w = std::f32::consts::TAU * n;
        h += a * (w * s + phase).sin();
        dh += a * w * (w * s + phase).cos();
        n *= ratio;
    }
    (h, dh)
}

/// The camera mount's displacement and speed `t` seconds after the start of
/// a half-sine base pulse of `peak` m/s² lasting `pulse_s`: the damped
/// oscillator's response to it (Duhamel's integral over the pulse).
fn ring(t: f32, peak: f32, pulse_s: f32) -> (f32, f32) {
    let wn = std::f32::consts::TAU * MOUNT_HZ;
    let wd = wn * (1.0 - MOUNT_ZETA * MOUNT_ZETA).sqrt();
    let end = t.min(pulse_s);
    let steps = 24;
    let ds = end / steps as f32;
    let (mut x, mut v) = (0.0f32, 0.0f32);
    for i in 0..steps {
        let s = (i as f32 + 0.5) * ds;
        let a = peak * (std::f32::consts::PI * s / pulse_s).sin();
        let tau = t - s;
        let e = (-MOUNT_ZETA * wn * tau).exp();
        x += a * e * (wd * tau).sin() / wd * ds;
        v += a * e * ((wd * tau).cos() - MOUNT_ZETA * wn / wd * (wd * tau).sin()) * ds;
    }
    (x, v)
}

/// A standard normal draw from a hash (Box–Muller).
fn normal(seed: u64, a: i32, b: i32) -> f32 {
    let u1 = ((hash2(seed, a, b) % 1_000_000) as f32 + 0.5) / 1_000_000.0;
    let u2 = ((hash2(seed ^ 0x9E37_79B9, a, b) % 1_000_000) as f32 + 0.5) / 1_000_000.0;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

impl World {
    /// Eye height of the robot standing still on level ground, metres (the
    /// search planner's and the look geometry's camera height).
    pub fn standing_eye_m(&self) -> f32 {
        hkin::head_camera(&self.body, &KinPose::stand(0.0)).pos[2]
    }

    /// The pose the camera rides on now, and whether it is a walking step
    /// (whose base frame sits on the stance ankle, not under the body).
    fn camera_pose(&self) -> (KinPose, bool) {
        if let Some(p) = &self.act {
            if let Some(s) = p.rollout.sample_at(p.t()) {
                return (s.pose, false);
            }
        }
        if let Some((pose, _, src)) = self.act_pose() {
            return (pose, src == noto_core::Source::FootLift);
        }
        let slope = if self.body.legged { 0.0 } else { self.posture_now().0.slope_deg };
        (KinPose { base_pitch_deg: slope, ..KinPose::stand(self.carried_kg()) }, false)
    }

    /// The head camera now: the body's pose, the shake, the sensor.
    pub fn head_view(&self, cam: &Camera) -> HeadView {
        let (mut pose, walking) = self.camera_pose();
        // A rigid chassis lies on the ground under its contacts; legs keep the torso level.
        let (tilt, tilt_rate) = if self.body.legged {
            ((0.0, 0.0), [0.0, 0.0])
        } else {
            let (x, z, yaw) = (self.agent.x, self.agent.z, self.agent.yaw);
            let now = self.chassis_attitude(x, z, yaw);
            let v = self.agent.speed_mps;
            let dt = 0.05;
            let next = if v > 1e-3 { self.chassis_attitude(x + v * dt * yaw.cos(), z + v * dt * yaw.sin(), yaw) } else { now };
            pose.base_pitch_deg = now.0.to_degrees();
            (now, [(next.0 - now.0) / dt, (next.1 - now.1) / dt])
        };
        let hc = hkin::head_camera(&self.body, &pose);
        let mut pos = hc.pos;
        if walking {
            pos[0] -= pose.hip[0];
        }
        // The neck takes back its share of the base's tilt on a slope.
        let base = if self.body.legged { 0.0 } else { pose.base_pitch_deg };
        let head_pitch_deg = hc.pitch_down_deg + base * self.body.head.gaze_stabilization;
        let timing = sensor_timing(self.body.head.sensor);
        let (shake, window_s) = self.shake();
        // What the gyro got wrong integrating the shake: its angle random walk over the window.
        let arw = (timing.gyro_arw_deg * window_s.max(0.0).sqrt()).to_radians();
        let (ep, er) = if shake.source == "still" { (0.0, 0.0) } else { (arw * normal(self.cfg.seed, self.tick as i32, 1), arw * normal(self.cfg.seed, self.tick as i32, 2)) };
        let at = self.body_to_world(pos);
        let pan_deg = hc.pan_deg + cam.pan_deg;
        let yaw = self.agent.yaw - pan_deg.to_radians();
        let pitch = cam.pitch_deg.to_radians() - head_pitch_deg.to_radians();
        // The neck has no roll joint: the chassis's roll reaches the camera whole.
        let view = View { origin: [at[0], at[1] + shake.dz_m, at[2]], yaw, pitch: pitch + shake.pitch, roll: shake.roll + tilt.1 };
        let est = View { pitch: view.pitch + ep, roll: view.roll + er, ..view };
        let keep = 1.0 - self.body.head.gaze_stabilization;
        let rate = [self.eye.yaw_rate, shake.rate[1] + tilt_rate[0] * keep, shake.rate[2] + tilt_rate[1]];
        let exposure = Exposure { rate, exposure_s: timing.exposure_s, readout_s: timing.readout_s };
        let tilt_deg = [tilt.0.to_degrees(), tilt.1.to_degrees()];
        HeadView { view, exposure, est, shake, eye_m: pos[2] + shake.dz_m, head_pitch_deg, pan_deg, blur_px: exposure.blur_px(cam), tilt_deg }
    }

    /// Pitch (nose up) and roll (right side down), radians, of a rigid
    /// wheeled or tracked chassis at `(x, z)` heading `yaw`: the plane of the
    /// ground (`Terrain::surface_smooth_m`) under its four contacts, the
    /// wheelbase (a track's contact length) apart along the heading and the
    /// track width across it. A cell taller than the base climbs (by step or
    /// by slope) is a wall beside it, not ground under it.
    pub fn chassis_attitude(&self, x: f32, z: f32, yaw: f32) -> (f32, f32) {
        let spec = self.drive_spec();
        let reach = spec.max_step_m.max(spec.max_slope_deg.to_radians().tan() * crate::terrain::CELL_M) + 0.01;
        let base = self.surface_m(x, z);
        let (len, wid) = (self.body.support_len_m.max(0.2), self.body.support_width_m.max(0.2));
        let (s, c) = yaw.sin_cos();
        // Forward (c, s); the robot's right is (-s, c) as the camera's.
        let h = |f: f32, r: f32| self.terrain.surface_smooth_m(x + c * f - s * r, z + s * f + c * r, base, reach);
        let (fr, fl) = (h(0.5 * len, 0.5 * wid), h(0.5 * len, -0.5 * wid));
        let (rr, rl) = (h(-0.5 * len, 0.5 * wid), h(-0.5 * len, -0.5 * wid));
        let pitch = (0.5 * (fr + fl) - 0.5 * (rr + rl)).atan2(len);
        let roll = (0.5 * (fl + rl) - 0.5 * (fr + rr)).atan2(wid);
        (pitch, roll)
    }

    /// The shake now and how long the IMU has been integrating it, s.
    fn shake(&self) -> (Shake, f32) {
        let ground = self.ground_here().surface();
        let speed = self.agent.speed_mps;
        if speed <= 1e-3 {
            return (Shake::still(), 0.0);
        }
        let gd = roughness_gd(ground);
        let seed = self.cfg.seed;
        let stab = self.body.head.gaze_stabilization;
        if self.body.legged {
            let Some((g, _, _)) = self.walk.plan.as_ref() else { return (Shake::still(), 0.0) };
            let t = self.walk.since_step_m / speed;
            let u = (self.walk.since_step_m / g.step_m.max(1e-3)).clamp(0.0, 1.0);
            let s = sway(u, self.walk.stance, g.step_s, stab)
                .add(self.heel_strike(g, ground, t))
                .add(footfall(seed, gd, g.step_m, self.walk.steps, stab));
            (s, t)
        } else {
            let tracked = matches!(self.body.base, noto_core::BaseKind::Track);
            let bridge = if tracked { self.body.support_len_m } else { 2.0 * hkin::WHEEL_R_M };
            (rolling(seed, gd, self.eye.travelled_m, speed, self.body.support_len_m.max(0.2), self.body.support_width_m.max(0.2), bridge), IMU_WINDOW_S)
        }
    }

    /// The mount's ringing `t` seconds after the last heel strike.
    fn heel_strike(&self, g: &noto_core::act::gait::GaitPlan, ground: Surface, t: f32) -> Shake {
        let s = strike_scale(ground);
        let pulse = STRIKE_PULSE_S / s;
        // A pulse longer than both feet are down is part of the step, not a strike.
        if pulse > g.ds_frac * g.step_s {
            return Shake::still();
        }
        let (x, v) = ring(t, STRIKE_PEAK_MPS2 * s, pulse);
        let eye = self.standing_eye_m();
        let r2 = (GYRATION_SHARE * eye).powi(2);
        // The heel lands half a step ahead of the CoM (the body pitches back, nose up)
        // and half the stance to the side of the landing foot (the body rolls away from it).
        let pitch_arm = 0.5 * g.step_m / r2;
        let side = if self.walk.stance == 0 { -1.0 } else { 1.0 };
        let roll_arm = side * 0.5 * noto_core::parts::STANCE_WIDTH_M / r2;
        Shake { source: "heel_strike", dz_m: x, pitch: pitch_arm * x, roll: roll_arm * x, rate: [v, pitch_arm * v, roll_arm * v] }
    }
}

/// The trunk's sway `u` (0..1) into a step whose stance foot is `stance`:
/// pitch at the step frequency, nose up at the strike and down at
/// mid-stance; roll at the stride frequency, toward the stance foot; each
/// less the share the neck cancels.
fn sway(u: f32, stance: usize, step_s: f32, stab: f32) -> Shake {
    let keep = 1.0 - stab;
    let (ap, ar) = ((0.5 * SWAY_PITCH_PP_DEG).to_radians() * keep, (0.5 * SWAY_ROLL_PP_DEG).to_radians() * keep);
    let w = std::f32::consts::TAU / step_s.max(0.1);
    let phase = std::f32::consts::TAU * u;
    // The right foot (0) in stance leans the body right: right side down.
    let side = if stance == 0 { 1.0 } else { -1.0 };
    let half = std::f32::consts::PI * u;
    Shake {
        source: "sway",
        dz_m: 0.0,
        pitch: ap * phase.cos(),
        roll: side * ar * half.sin(),
        rate: [0.0, -ap * w * phase.sin(), side * ar * 0.5 * w * half.cos()],
    }
}

/// The body's tilt from where the feet landed on the rough ground: this
/// footfall against the last one (along the step: pitch; across the stance:
/// roll), less the neck's stabilisation. Constant over the step.
fn footfall(seed: u64, gd: f32, step_m: f32, steps: u32, stab: f32) -> Shake {
    let s_now = steps as f32 * step_m;
    let s_last = s_now - step_m;
    // Footfalls alternate sides: this foot on one profile, the last on the other.
    let (side_now, side_last) = if steps % 2 == 0 { (0, 1) } else { (1, 0) };
    let foot = noto_core::act::gait::NOMINAL_FOOT_M;
    let (h_now, _) = profile(seed, gd, side_now, s_now, foot);
    let (h_last, _) = profile(seed, gd, side_last, s_last, foot);
    let dh = h_now - h_last;
    let keep = 1.0 - stab;
    let roll_sign = if side_now == 1 { 1.0 } else { -1.0 };
    Shake {
        source: "footfall",
        dz_m: 0.5 * dh,
        pitch: keep * (dh / step_m.max(0.05)).atan() * 0.5,
        roll: keep * roll_sign * (dh / noto_core::parts::STANCE_WIDTH_M).atan() * 0.5,
        rate: [0.0; 3],
    }
}

/// Vertical acceleration RMS of a rigid chassis rolling at `speed` over a
/// ground, m/s² (the second difference of its heave over 60 m of profile).
pub fn chassis_accel_rms(seed: u64, ground: Surface, speed: f32, base_m: f32, track_m: f32, bridge_m: f32) -> f32 {
    let gd = roughness_gd(ground);
    let ds = 0.005f32;
    let n = (60.0 / ds) as usize;
    let heave: Vec<f32> = (0..n).map(|i| rolling(seed, gd, i as f32 * ds, speed, base_m, track_m, bridge_m).dz_m).collect();
    let acc: Vec<f32> = heave.windows(3).map(|w| (w[0] - 2.0 * w[1] + w[2]) / (ds * ds) * speed * speed).collect();
    (acc.iter().map(|a| a * a).sum::<f32>() / acc.len().max(1) as f32).sqrt()
}

/// Peak head acceleration of a heel strike on a ground, m/s² (0 when the
/// pulse outlasts a double support of `ds_s`).
pub fn strike_peak_mps2(ground: Surface, ds_s: f32) -> f32 {
    let s = strike_scale(ground);
    if STRIKE_PULSE_S / s > ds_s {
        0.0
    } else {
        STRIKE_PEAK_MPS2 * s
    }
}

/// A rigid chassis on two profiles at its four contacts: heave, pitch over
/// the wheelbase, roll over the track, and their rates at `speed`.
fn rolling(seed: u64, gd: f32, s: f32, speed: f32, base_m: f32, track_m: f32, bridge_m: f32) -> Shake {
    let half = 0.5 * base_m;
    let at = |side: u32, x: f32| profile(seed, gd, side, x, bridge_m);
    let (fl, dfl) = at(0, s + half);
    let (fr, dfr) = at(1, s + half);
    let (rl, drl) = at(0, s - half);
    let (rr, drr) = at(1, s - half);
    let heave = 0.25 * (fl + fr + rl + rr);
    let pitch = ((0.5 * (fl + fr) - 0.5 * (rl + rr)) / base_m).atan();
    let roll = ((0.5 * (fl + rl) - 0.5 * (fr + rr)) / track_m).atan();
    let d_heave = 0.25 * (dfl + dfr + drl + drr) * speed;
    let d_pitch = (0.5 * (dfl + dfr) - 0.5 * (drl + drr)) / base_m * speed;
    let d_roll = (0.5 * (dfl + drl) - 0.5 * (dfr + drr)) / track_m * speed;
    Shake { source: "rolling", dz_m: heave, pitch, roll, rate: [d_heave, d_pitch, d_roll] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::TerrainKind;
    use crate::world::{Action, WorldConfig};
    use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 64, noto: hand, body: BodyPlan::default(), terrain: TerrainKind::Open, parcels: Some(0), ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn the_camera_sits_on_the_head_and_stands_still_at_rest() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let w = world(base);
            let eye = w.standing_eye_m();
            // Hip 0.58 + pelvis 0.12 + trunk 0.30 + neck 0.07 + lens 0.07 (kin's reference geometry).
            assert!((eye - 1.14).abs() < 0.02, "{base:?}: eye {eye}");
            let cam = Camera::default();
            let hv = w.head_view(&cam);
            assert!((hv.view.pitch - cam.pitch_deg.to_radians()).abs() < 1e-4 && hv.view.roll == 0.0, "{base:?}: {:?}", hv.view);
            assert_eq!(hv.shake.source, "still");
            assert_eq!(hv.view, hv.est);
            assert!((hv.view.origin[1] - (w.agent_ground_m() + eye)).abs() < 1e-4);
        }
    }

    #[test]
    fn walking_shakes_the_camera_within_a_persons_range_and_the_gyro_reads_it() {
        let mut w = world(BaseKind::HumanFoot);
        // Somewhere a walk of a few metres away that the planner reaches.
        let (ax, az) = (w.agent.x, w.agent.z);
        let goal = (0..16)
            .map(|k| (k as f32 * std::f32::consts::TAU / 16.0).sin_cos())
            .map(|(s, c)| (ax + 6.0 * c, az + 6.0 * s))
            .find(|&(x, z)| w.clone().command(Action::MoveTo { x, z, reach_m: 0.3 }).is_ok())
            .expect("a reachable point 6 m away");
        w.command(Action::MoveTo { x: goal.0, z: goal.1, reach_m: 0.3 }).unwrap();
        let cam = Camera::default();
        let (mut peak_pitch, mut peak_rate, mut seen, mut peak_err) = (0.0f32, 0.0f32, 0, 0.0f32);
        for _ in 0..40 {
            w.tick();
            let hv = w.head_view(&cam);
            if hv.shake.source == "still" {
                continue;
            }
            seen += 1;
            peak_pitch = peak_pitch.max(hv.shake.pitch.abs());
            peak_rate = peak_rate.max(hv.shake.rate[1].abs());
            peak_err = peak_err.max((hv.view.pitch - hv.est.pitch).abs());
        }
        assert!(seen > 20, "walking ticks {seen}");
        // A person's head pitches 8–9° peak to peak at up to 30 °/s (research §1.1); the neck keeps the robot's camera inside that.
        assert!(peak_pitch.to_degrees() > 0.2 && peak_pitch.to_degrees() < 4.5, "pitch shake {}°", peak_pitch.to_degrees());
        assert!(peak_rate.to_degrees() < 90.0, "pitch rate {}°/s", peak_rate.to_degrees());
        // The gyro's random walk over a step is a few hundredths of a degree.
        assert!(peak_err.to_degrees() < 0.1, "IMU error {}°", peak_err.to_degrees());
    }

    #[test]
    fn harder_ground_strikes_harder_and_rougher_ground_tilts_more() {
        let w = world(BaseKind::HumanFoot);
        let g = noto_core::act::gait::plan(&w.body, 0.6, 0.7, 0.0);
        let peak = |s: Surface| (0..300).map(|i| w.heel_strike(&g, s, i as f32 * 0.001).pitch.abs()).fold(0.0, f32::max);
        assert!(peak(Surface::Firm) > peak(Surface::Grass) && peak(Surface::Grass) > peak(Surface::Sand), "firm {} grass {} sand {}", peak(Surface::Firm), peak(Surface::Grass), peak(Surface::Sand));
        // Dry sand's 194 ms pulse outlasts a short double support: no strike then.
        assert_eq!(strike_peak_mps2(Surface::Sand, 0.15), 0.0);
        assert!(strike_peak_mps2(Surface::Firm, 0.15) > 4.0);
        let tilt = |s: Surface| ((0..400).map(|k| footfall(5, roughness_gd(s), 0.3, k, 0.55).roll.powi(2)).sum::<f32>() / 400.0).sqrt();
        assert!(tilt(Surface::Rock) > 4.0 * tilt(Surface::Firm), "rock {} firm {}", tilt(Surface::Rock), tilt(Surface::Firm));
        let roll = |s: Surface| ((0..400).map(|i| rolling(5, roughness_gd(s), i as f32 * 0.07, 1.0, 0.4, 0.35, 0.12).pitch.powi(2)).sum::<f32>() / 400.0).sqrt();
        assert!(roll(Surface::Gravel) > 1.5 * roll(Surface::Firm), "gravel {} firm {}", roll(Surface::Gravel), roll(Surface::Firm));
    }

    #[test]
    fn a_chassis_tilts_with_the_ground_and_legs_keep_the_torso_level() {
        for base in [BaseKind::Wheel, BaseKind::Track, BaseKind::HumanFoot] {
            let mut w = world(base);
            w.agent.yaw = 0.0;
            let (cx, cz) = w.agent_cell();
            (w.agent.x, w.agent.z) = crate::terrain::cell_center(cx, cz);
            let h0 = 50;
            // A side slope: the ground falls 5 cm per cell toward the robot's right (+z at yaw 0).
            for dz in -2i32..=2 {
                for dx in -2i32..=2 {
                    let (x, z) = ((cx as i32 + dx) as usize, (cz as i32 + dz) as usize);
                    w.terrain.set_height_cm(x, z, h0 - 5 * dz);
                }
            }
            let hv = w.head_view(&Camera::default());
            if base == BaseKind::HumanFoot {
                assert_eq!(hv.tilt_deg, [0.0, 0.0], "legs keep the torso level");
                assert!(hv.view.roll.abs() < 1e-6);
            } else {
                // 5 cm per 0.5 m is a 5.7° slope: the chassis rolls right side down by about that.
                assert!(hv.tilt_deg[1] > 3.0 && hv.tilt_deg[1] < 7.0, "{base:?} roll {:?}", hv.tilt_deg);
                assert!(hv.tilt_deg[0].abs() < 0.5, "{base:?} pitch {:?}", hv.tilt_deg);
                assert!((hv.view.roll.to_degrees() - hv.tilt_deg[1]).abs() < 1e-3, "the neck has no roll joint");
            }
        }
        // A wall beside the robot is not ground under its wheels.
        let mut w = world(BaseKind::Wheel);
        w.agent.yaw = 0.0;
        let (cx, cz) = w.agent_cell();
        (w.agent.x, w.agent.z) = crate::terrain::cell_center(cx, cz);
        let h0 = (w.terrain.height_m(cx, cz) * 100.0).round() as i32;
        for dx in -2i32..=2 {
            w.terrain.set_height_cm((cx as i32 + dx) as usize, cz + 1, h0 + 120);
        }
        assert!(w.head_view(&Camera::default()).tilt_deg[1].abs() < 0.5);
    }

    #[test]
    fn turning_blurs_in_dim_light_and_hardly_in_daylight() {
        let mut w = world(BaseKind::Wheel);
        let cam = Camera::default();
        assert!(w.head_view(&cam).blur_px < 0.1);
        let yaw = w.agent.yaw + 1.5;
        w.command(Action::Turn { yaw_deg: yaw.to_degrees() }).unwrap();
        w.tick();
        let hv = w.head_view(&cam);
        assert!(hv.exposure.rate[0].abs() > 1.0, "yaw rate {}", hv.exposure.rate[0]);
        // Daylight (10 000 lx): under a millisecond of exposure, about a pixel of smear.
        assert!(hv.blur_px > 0.1 && hv.blur_px < 2.0, "daylight blur {} px", hv.blur_px);
        // A warehouse aisle (200 lx) holds the shutter for a whole 33 ms frame.
        let dim = Exposure { exposure_s: exposure_s(200.0), ..hv.exposure };
        assert!(dim.blur_px(&cam) > 15.0, "dim blur {} px", dim.blur_px(&cam));
        // Research §4.5: 10 000 lx → 0.87 ms; 500 lx → 17.5 ms.
        assert!((exposure_s(10_000.0) - 0.00087).abs() < 2e-5 && (exposure_s(500.0) - 0.0175).abs() < 3e-4);
    }

    #[test]
    fn rougher_classes_give_taller_profiles() {
        let rms = |gd: f32| {
            let n = 2000;
            ((0..n).map(|i| profile(7, gd, 0, i as f32 * 0.05, 0.0).0.powi(2)).sum::<f32>() / n as f32).sqrt()
        };
        let (a, d) = (rms(16.0e-6), rms(1024.0e-6));
        // Height scales with √Gd: class D is 8× class A.
        assert!((d / a - 8.0).abs() < 0.5, "{a} {d}");
        // ISO 8608 class A over 0.1–50 cycles/m: σ² = Gd·n0²·(1/n_lo − 1/n_hi) ≈ 1.6e-6 m².
        assert!(a > 0.7e-3 && a < 2.0e-3, "class A rms {a}");
    }

    #[test]
    fn a_long_contact_bridges_short_bumps() {
        let rough = |bridge: f32| {
            let n = 2000;
            (0..n).map(|i| profile(3, 512e-6, 1, i as f32 * 0.01, bridge).1.abs()).sum::<f32>() / n as f32
        };
        assert!(rough(0.4) < 0.5 * rough(0.0), "a track's contact smooths the slope it feels");
    }

    #[test]
    fn the_mount_rings_and_settles() {
        let (x0, _) = ring(0.04, STRIKE_PEAK_MPS2, STRIKE_PULSE_S);
        let (x1, _) = ring(0.40, STRIKE_PEAK_MPS2, STRIKE_PULSE_S);
        assert!(x0.abs() > 10.0 * x1.abs(), "{x0} {x1}");
        // A softer ground's longer, lower pulse moves the mount less.
        let peak = |s: f32| (0..200).map(|i| ring(i as f32 * 0.001, STRIKE_PEAK_MPS2 * s, STRIKE_PULSE_S / s).0.abs()).fold(0.0, f32::max);
        assert!(peak(0.36) < peak(1.0));
    }
}
