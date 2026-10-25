//! The robot's actions as time series of physics: bending to reach the
//! ground (`reach`), grasping, lifting and setting an object down (`grasp`),
//! pushing an object along the ground (`push`), walking step by step
//! (`gait`) and driving on wheels or tracks (`drive`).
//!
//! The world used to treat an action as a timer that always succeeds once a
//! static check passed: the hand picked an item off the ground without
//! bending, held anything its friction table allowed, and walked without
//! steps. Each module here plays the action out instead (`kin` places the
//! body, the presets' actuators and masses load it) with the four physical
//! causes the research names (`docs/research/actions.md`):
//!
//! * **friction**: Coulomb limits at every contact (finger pads on the object,
//!   the object on the ground, feet, wheels and tracks on the ground), with the
//!   scatter of measured friction drawn per contact;
//! * **mass**: weight and inertia of the robot's links and of the object,
//!   in the joint torques, the grip the hand needs and the ZMP;
//! * **elasticity**: soft finger pads (contact area, torsional friction),
//!   restitution of an object hitting the ground, compliance at heel strike;
//! * **collision**: the finger, foot or bumper meeting a surface, the object
//!   landing, a pushed object meeting an obstacle.
//!
//! Every function takes the random draw it needs as a seed, so a world (and
//! a test) replays the same outcome; nothing here is tuned to a success rate.
//! Outcomes come from the physics and the scatter of its inputs.

use serde::Serialize;

use crate::kin::{KinLoads, KinPose};

pub mod drive;
pub mod focus;
pub mod gait;
pub mod grasp;
pub mod material;
pub mod push;
pub mod reach;

/// Sample interval of the action time series, seconds (50 Hz).
pub const SAMPLE_S: f32 = 0.02;

/// What the robot does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActKind {
    /// Reach down, close the hand, lift, stow in the bay.
    Grasp,
    /// Take from the bay (or hold), bring it down, open the hand.
    Release,
    Push,
    Walk,
    Drive,
    /// Bend to reach low and come back up (part of a grasp or a release).
    Bend,
}

impl ActKind {
    pub const ALL: [ActKind; 6] = [ActKind::Grasp, ActKind::Release, ActKind::Push, ActKind::Walk, ActKind::Drive, ActKind::Bend];

    pub fn id(self) -> &'static str {
        match self {
            ActKind::Grasp => "grasp",
            ActKind::Release => "release",
            ActKind::Push => "push",
            ActKind::Walk => "walk",
            ActKind::Drive => "drive",
            ActKind::Bend => "bend",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            ActKind::Grasp => "잡기",
            ActKind::Release => "놓기",
            ActKind::Push => "밀기",
            ActKind::Walk => "걷기",
            ActKind::Drive => "주행",
            ActKind::Bend => "굽히기",
        }
    }
}

/// The body parts the focus views follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// The hand's shape: fingers, contacts, grip and load force.
    Hand,
    /// The finger joints: angle and torque per joint.
    HandJoints,
    /// The foot's sole: contact, centre of pressure, ground reaction.
    Foot,
    /// Hip, knee and ankle: angle, torque, ratio.
    FootJoints,
    Waist,
    Neck,
    Wheel,
    Track,
}

impl Part {
    pub const ALL: [Part; 8] = [Part::Hand, Part::HandJoints, Part::Foot, Part::FootJoints, Part::Waist, Part::Neck, Part::Wheel, Part::Track];

    pub fn id(self) -> &'static str {
        match self {
            Part::Hand => "hand",
            Part::HandJoints => "hand_joints",
            Part::Foot => "foot",
            Part::FootJoints => "foot_joints",
            Part::Waist => "waist",
            Part::Neck => "neck",
            Part::Wheel => "wheel",
            Part::Track => "track",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Part::Hand => "손 모양",
            Part::HandJoints => "손 관절",
            Part::Foot => "발 모양",
            Part::FootJoints => "발 관절",
            Part::Waist => "허리",
            Part::Neck => "목 관절",
            Part::Wheel => "바퀴",
            Part::Track => "트랙",
        }
    }

    pub fn from_id(id: &str) -> Option<Part> {
        Part::ALL.into_iter().find(|p| p.id() == id)
    }
}

/// A moment worth a keyframe: a contact made or lost, a slip, an impact.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActEvent {
    /// Seconds from the start of the action.
    pub t: f32,
    /// `contact`, `lift_off`, `slip`, `drop`, `release`, `impact`, `bend_bottom`, `stand`, `heel_strike`, `toe_off`, `foot_slip`, `fall`, `recover`, `breakaway`, `push_lost`, `collision`, `wheel_slip`, `stuck`.
    pub kind: &'static str,
    pub label_ko: String,
    /// The parts whose focus views show it.
    pub parts: Vec<Part>,
    /// The number the event is about (a force, a slip ratio, a margin), if any.
    pub value: Option<f32>,
    /// False for a failure (a slip that drops the object, a fall).
    pub ok: bool,
}

impl ActEvent {
    pub fn new(t: f32, kind: &'static str, label_ko: impl Into<String>, parts: &[Part], value: Option<f32>, ok: bool) -> ActEvent {
        ActEvent { t, kind, label_ko: label_ko.into(), parts: parts.to_vec(), value, ok }
    }
}

/// The hand's grip at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Grip {
    /// Squeeze of the fingers, total normal force, N.
    pub grip_n: f32,
    /// Force the object puts on the hand along the fingers' contact plane (weight and inertia), N.
    pub load_n: f32,
    /// Friction the pads actually have on this object, and what the controller believes.
    pub mu: f32,
    pub mu_est: f32,
    /// `mu · grip / load` (slips below 1).
    pub slip_ratio: f32,
    /// Torsional friction left against the object's turn about the grasp axis, N·m (soft-finger limit minus need).
    pub twist_margin_nm: f32,
    /// Fingers touching the object.
    pub contacts: u8,
    pub closure: f32,
}

/// One sample of an action.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActSample {
    pub t: f32,
    pub phase: &'static str,
    pub pose: KinPose,
    pub loads: KinLoads,
    pub margin: f32,
    pub zmp: [f32; 2],
    pub com: [f32; 3],
    /// Worst actuated joint's torque over its peak.
    pub peak_ratio: f32,
    pub grip: Option<Grip>,
    /// The handled object's centre in the base frame (in the hand, on the ground, sliding).
    pub object: Option<[f32; 3]>,
}

/// How an action ended.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Outcome {
    pub ok: bool,
    /// Stable failure code (`reach_ground`, `grasp_missed`, `grasp_slip`, `lift_overload`, `push_lost`, `push_stuck`, `fall`, ...).
    pub code: Option<&'static str>,
    pub reason_ko: Option<String>,
}

impl Outcome {
    pub fn ok() -> Outcome {
        Outcome { ok: true, code: None, reason_ko: None }
    }

    pub fn fail(code: &'static str, reason_ko: impl Into<String>) -> Outcome {
        Outcome { ok: false, code: Some(code), reason_ko: Some(reason_ko.into()) }
    }
}

/// A played-out action.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActRollout {
    pub kind: ActKind,
    pub samples: Vec<ActSample>,
    pub events: Vec<ActEvent>,
    pub outcome: Outcome,
    pub duration_s: f32,
    /// Summary numbers for the SDS and the benchmark (each documented where it is set).
    pub metrics: Vec<(&'static str, f32)>,
    pub notes: Vec<String>,
}

impl ActRollout {
    /// The sample at or before `t` (the last one after the end).
    pub fn sample_at(&self, t: f32) -> Option<&ActSample> {
        let i = self.samples.partition_point(|s| s.t <= t);
        self.samples.get(i.saturating_sub(1))
    }

    pub fn metric(&self, key: &str) -> Option<f32> {
        self.metrics.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }

    /// Events in `(from, to]`.
    pub fn events_between(&self, from: f32, to: f32) -> impl Iterator<Item = &ActEvent> {
        self.events.iter().filter(move |e| e.t > from && e.t <= to)
    }
}

/// A small deterministic generator for the draws (xorshift64*, seeded).
#[derive(Clone, Debug)]
pub struct Draw(u64);

impl Draw {
    pub fn new(seed: u64) -> Draw {
        Draw(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    /// A positive quantity with mean `mean` and coefficient of variation `cov`
    /// (log-normal, so friction never goes negative).
    pub fn lognormal(&mut self, mean: f32, cov: f32) -> f32 {
        if cov <= 0.0 {
            return mean;
        }
        let s2 = (1.0 + cov * cov).ln();
        let mu = mean.ln() - 0.5 * s2;
        (mu + s2.sqrt() * self.normal()).exp()
    }
}

/// The error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    s * y as f32
}

/// The standard normal distribution function.
pub fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// The chance a log-normal draw of mean `mean` and variation `cov` is at least `need`.
pub fn lognormal_above(mean: f32, cov: f32, need: f32) -> f32 {
    if need <= 0.0 {
        return 1.0;
    }
    if cov <= 0.0 {
        return if mean >= need { 1.0 } else { 0.0 };
    }
    let s2 = (1.0 + cov * cov).ln();
    let mu = mean.ln() - 0.5 * s2;
    1.0 - normal_cdf((need.ln() - mu) / s2.sqrt())
}

/// Minimum-jerk blend 0 → 1 over `u` in [0, 1] (as `dynamics`).
pub fn min_jerk(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    let u3 = u * u * u;
    10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u * u
}

/// Peak speed of a minimum-jerk move over `dur` covering `delta`: 1.875·Δ/T.
pub const MIN_JERK_PEAK: f32 = 1.875;

/// How fast a base actually goes on a ground and a slope (positive uphill)
/// over its flat-ground cruise speed, for the route planner: legs shorten
/// their step to keep the friction margin (`gait::plan_z`, when the walking
/// CoM height is known), wheels and tracks lose speed to the slip their
/// cruise traction needs (`drive::slip_on`; the mass cancels at cruise).
/// `None` where the drive spins in place.
pub fn locomotion_factor(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32) -> Option<f32> {
    locomotion_factor_with(spec, ground, slope_deg, 0.0)
}

/// [`locomotion_factor`] with more for the drive to push: `extra_n_per_kg`,
/// newtons per kilogram of the robot (a pushed object's friction and its
/// share of the slope). Past the drive's own force (`drive_accel_mps2`) it
/// does not move.
pub fn locomotion_factor_with(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32, extra_n_per_kg: f32) -> Option<f32> {
    use crate::base::BaseKind;
    use crate::balance::G;
    match spec.kind {
        BaseKind::HumanFoot => {
            let Some(z) = spec.walk_com_z_m else { return Some(1.0) };
            let g = gait::plan_z(z, spec.walk_foot_m.unwrap_or(gait::NOMINAL_FOOT_M), spec.max_speed_mps, crate::surface::available_mu(spec.kind, ground), slope_deg.max(0.0));
            Some((g.speed / g.speed_cmd.max(1e-3)).clamp(0.05, 1.0))
        }
        _ => {
            let th = slope_deg.to_radians();
            let mu = crate::surface::available_mu(spec.kind, ground);
            let per_kg = G * (crate::surface::rolling_resistance(spec.kind, ground) * th.cos() + th.sin()) + extra_n_per_kg;
            if extra_n_per_kg > 0.0 && spec.drive_accel_mps2.is_some_and(|a| per_kg > a) {
                return None;
            }
            let need = per_kg / (G * mu * th.cos()).max(1e-3);
            let gs = crate::surface::Surface::from_id(ground.id).unwrap_or_default();
            let l = if spec.kind == BaseKind::Track { spec.footprint_m } else { drive::wheel_contact_m(gs) };
            drive::slip_on(gs, need, l).map(|s| (1.0 - s).max(0.02))
        }
    }
}

/// The slip ratio a wheeled or tracked base cruises at up `slope_deg`
/// (downhill counts as flat) on `ground`: the traction its rolling resistance
/// and the slope need, as [`locomotion_factor_with`] finds it
/// (`drive::slip_on`). `None` for legs and where the drive spins. Wheel
/// odometry counts this slip as distance it did not travel.
pub fn cruise_slip(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32) -> Option<f32> {
    use crate::balance::G;
    use crate::base::BaseKind;
    if spec.kind == BaseKind::HumanFoot {
        return None;
    }
    let th = slope_deg.max(0.0).to_radians();
    let mu = crate::surface::available_mu(spec.kind, ground);
    let per_kg = G * (crate::surface::rolling_resistance(spec.kind, ground) * th.cos() + th.sin());
    let need = per_kg / (G * mu * th.cos()).max(1e-3);
    let gs = crate::surface::Surface::from_id(ground.id).unwrap_or_default();
    let l = if spec.kind == BaseKind::Track { spec.footprint_m } else { drive::wheel_contact_m(gs) };
    drive::slip_on(gs, need, l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_repeat_per_seed_and_have_the_asked_moments() {
        let mut a = Draw::new(7);
        let mut b = Draw::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut d = Draw::new(11);
        let xs: Vec<f32> = (0..20000).map(|_| d.lognormal(0.6, 0.15)).collect();
        let m = xs.iter().sum::<f32>() / xs.len() as f32;
        let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32).sqrt();
        assert!((m - 0.6).abs() < 0.01 && (sd / m - 0.15).abs() < 0.01, "{m} {sd}");
        assert!(xs.iter().all(|x| *x > 0.0));
        assert!((min_jerk(0.5) - 0.5).abs() < 1e-6 && min_jerk(1.0) == 1.0);
    }
}
