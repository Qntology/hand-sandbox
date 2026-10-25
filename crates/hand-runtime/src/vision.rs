//! When the robot looks, and what it believes while it does not.
//!
//! The camera and the recogniser do not run all the time. While the robot
//! moves toward something it remembers, it looks on a pattern, once per
//! look-ahead horizon, and once more just before its hand acts. In between it
//! moves and acts on memory: every remembered position (a landmark, a piece it
//! saw fall) lives in the robot's believed frame (`world_sim::odometry`), and
//! its uncertainty grows with the distance the robot has dead-reckoned since
//! it last saw it. A look is skipped while the predicted 2σ of the memory is
//! inside what the next action tolerates (half the object's width for a
//! grasp, the reach of the hand for a touch); an object the camera cannot see
//! from where the robot stands (beside its foot, behind it) is reached for on
//! memory, as a person reaches for a cup they saw a moment ago.
//!
//! The pattern's numbers come from how people use their eyes (research
//! `docs/research/perception.md` §2, group A): gaze looks about 1.5 s ahead
//! while walking (Matthis, Yates & Hayhoe 2018, Fig. 6B) and reaches the grasp
//! site about a second before the fingertip (Johansson et al. 2001, Fig. 8C:
//! median 1.13 s). The memory's growth follows the odometry error of the base
//! (the robot knows its own sensors' specification, `world_sim::odometry`) and
//! the first-order propagation of an uncertain pose to a landmark (Smith, Self
//! & Cheeseman 1986, §3.2); a fresh sighting is fused with the memory by the
//! Kalman gain of the two variances (Matthies & Shafer 1987, Eqs. 9–10). The
//! SDS `memory` track learns how far memories had really drifted when they
//! were seen again, and the robot uses that rate once it has enough of them.

use hand_sds::{Scope, SdsStore, Track};
use serde::{Deserialize, Serialize};
use world_sim::odometry::{LEG_SCALE_SD, RANDOM_WALK_M, WHEEL_SCALE_SD};
use world_sim::Camera;

/// Gaze look-ahead while walking, seconds (Matthis, Yates & Hayhoe 2018: ≈1.5 s on all terrains).
pub const LOOK_AHEAD_S: f32 = 1.5;
/// Gaze reaches the grasp site this long before the hand, seconds (Johansson et al. 2001: median 1.13 s, about a second).
pub const EYE_LEAD_S: f32 = 1.0;

/// When to look.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionCfg {
    /// Off: look as often as before v0.7 (`approach_scan_every_ticks` on a move,
    /// a look before every pick).
    pub intermittent: bool,
    /// Ticks between the pattern's looks while moving toward something remembered.
    pub look_every_ticks: u32,
    /// A memory is good enough for an action while this many σ stay inside its tolerance.
    pub sigmas: f32,
}

impl Default for VisionCfg {
    fn default() -> Self {
        VisionCfg { intermittent: true, look_every_ticks: (LOOK_AHEAD_S / world_sim::TICK_S).round() as u32, sigmas: 2.0 }
    }
}

/// Something the robot remembers: where it believes it is, how sure it was
/// when it saw it, and its odometer then.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Memory {
    pub x: f32,
    pub z: f32,
    /// 1σ of the position when it was seen, metres.
    pub sigma0_m: f32,
    /// The odometer when it was seen, metres.
    pub odo_m: f32,
    pub tick: u64,
}

impl Memory {
    pub fn seen(x: f32, z: f32, sigma0_m: f32, odo_m: f32, tick: u64) -> Memory {
        Memory { x, z, sigma0_m, odo_m, tick }
    }

    /// 1σ now: what it was when seen, and the drift of `rate` per metre (and
    /// the random walk) over what the robot has travelled since.
    pub fn sigma_m(&self, odo_now: f32, rate: DriftRate) -> f32 {
        let d = (odo_now - self.odo_m).max(0.0);
        (self.sigma0_m * self.sigma0_m + (rate.per_m * d).powi(2) + rate.walk_m * rate.walk_m * d).sqrt()
    }

    /// The memory updated by a new sighting at `(x, z)` with 1σ `sigma_obs`:
    /// the Kalman blend of the two (the more the robot moved since, the more
    /// the new sighting counts). Returns it and the innovation, metres.
    pub fn fuse(&self, x: f32, z: f32, sigma_obs: f32, odo_now: f32, tick: u64, rate: DriftRate) -> (Memory, f32) {
        let pm = self.sigma_m(odo_now, rate).powi(2);
        let po = sigma_obs * sigma_obs;
        let k = pm / (pm + po).max(1e-9);
        let innov = (x - self.x).hypot(z - self.z);
        let fused = Memory { x: self.x + k * (x - self.x), z: self.z + k * (z - self.z), sigma0_m: (pm * po / (pm + po).max(1e-9)).sqrt(), odo_m: odo_now, tick };
        (fused, innov)
    }
}

/// How fast a memory spreads per metre the robot dead-reckons.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DriftRate {
    /// Proportional part, metres per metre.
    pub per_m: f32,
    /// Random walk, metres per √metre.
    pub walk_m: f32,
    /// `spec` (the odometry's specification) or `sds` (what re-sightings showed).
    pub source: &'static str,
}

impl DriftRate {
    /// The odometry's own specification for this base.
    pub fn spec(legged: bool) -> DriftRate {
        DriftRate { per_m: if legged { LEG_SCALE_SD } else { WHEEL_SCALE_SD }, walk_m: RANDOM_WALK_M, source: "spec" }
    }

    /// What the SDS learned from re-sightings (`memory|<base>|drift`: the mean
    /// of the drift's variance per square metre), else the specification.
    pub fn learned(sds: &SdsStore, base: &str, legged: bool) -> DriftRate {
        let spec = DriftRate::spec(legged);
        match sds.adaptive_baseline(&Scope::new(Track::Memory, base, "drift"), "drift_var_per_m2") {
            Some((mean, _, _)) if mean.is_finite() && mean > 0.0 => DriftRate { per_m: (mean as f32).sqrt(), source: "sds", ..spec },
            _ => spec,
        }
    }
}

/// Whether a memory of 1σ `sigma_m` is good enough for an action that tolerates `tolerance_m`.
pub fn memory_ok(sigma_m: f32, tolerance_m: f32, cfg: &VisionCfg) -> bool {
    cfg.sigmas * sigma_m <= tolerance_m
}

/// Whether the camera on the robot at `pose` (x, z, yaw; believed) can see a
/// point on the ground at `(x, z)`: inside the horizontal field of view, beyond
/// the near edge of the view of the ground and within the camera's range.
pub fn in_view(cam: &Camera, eye_m: f32, pose: (f32, f32, f32), x: f32, z: f32) -> bool {
    let (dx, dz) = (x - pose.0, z - pose.1);
    let d = dx.hypot(dz);
    let bearing = wrap(dz.atan2(dx) - pose.2).abs();
    let half_h = 0.5 * cam.fov_deg.to_radians() - 0.05;
    // The lowest ray of the camera meets the ground this far ahead.
    let low = (-cam.pitch_deg).to_radians() + 0.5 * cam.vfov_deg().to_radians();
    let near = if low > 1e-3 { eye_m / low.tan() } else { 0.0 };
    bearing <= half_h && d >= near && d <= cam.max_dist_m
}

/// Whether the camera sees the point with the head pitched down by the neck
/// as far as it goes (`hand_core::kin::NECK_PITCH_MAX_DEG`): a glance at the
/// ground in front of the feet, where the level camera does not reach. The
/// tilt it takes, degrees, or `None` (beside the robot or behind it no tilt helps).
pub fn tilt_to_see(cam: &Camera, eye_m: f32, pose: (f32, f32, f32), x: f32, z: f32) -> Option<f32> {
    if in_view(cam, eye_m, pose, x, z) {
        return Some(0.0);
    }
    let tilt = hand_core::kin::NECK_PITCH_MAX_DEG;
    let down = Camera { pitch_deg: cam.pitch_deg - tilt, ..*cam };
    in_view(&down, eye_m, pose, x, z).then_some(tilt)
}

/// A record of when the robot looked and why, for the timeline's vision lane.
#[derive(Clone, Debug, Serialize)]
pub struct VisionMark {
    pub tick: u64,
    /// `pattern` (the look-ahead), `pre_act` (just before the hand acts),
    /// `glance` (a memory too unsure for the next action), `search`, `locate`
    /// and the like, or `blind` (an action taken on memory without a look).
    pub kind: String,
    /// What it was about (a landmark, a piece), and the memory's 1σ before and after, metres.
    pub what: String,
    pub sigma_before_m: Option<f32>,
    pub sigma_after_m: Option<f32>,
    /// How far the memory was off when seen again, metres (the truth is the record's, never the robot's).
    pub innovation_m: Option<f32>,
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

    #[test]
    fn memory_spreads_with_travel_and_a_look_shrinks_it() {
        let rate = DriftRate::spec(true);
        let m = Memory::seen(3.0, 1.0, 0.03, 10.0, 0);
        assert!((m.sigma_m(10.0, rate) - 0.03).abs() < 1e-6);
        let after = m.sigma_m(20.0, rate);
        assert!(after > 0.15 && after < 0.25, "10 m of legs: {after}");
        // A fresh sighting after 10 m counts for almost everything; right after the last one, for half.
        let (f, innov) = m.fuse(3.2, 1.0, 0.03, 20.0, 5, rate);
        assert!((f.x - 3.2).abs() < 0.01 && (innov - 0.2).abs() < 1e-5 && f.sigma0_m < 0.031);
        let (g, _) = m.fuse(3.2, 1.0, 0.03, 10.0, 5, rate);
        assert!((g.x - 3.1).abs() < 1e-4, "{}", g.x);
        // Wheels drift less than legs.
        assert!(m.sigma_m(20.0, DriftRate::spec(false)) < after);
    }

    #[test]
    fn what_the_camera_can_see_from_where_it_stands() {
        let cam = Camera::default();
        let me = (0.0, 0.0, 0.0);
        assert!(in_view(&cam, 0.5, me, 3.0, 0.2), "ahead");
        assert!(!in_view(&cam, 0.5, me, 0.1, 0.3), "beside the foot");
        assert!(!in_view(&cam, 0.5, me, -2.0, 0.0), "behind");
        assert!(!in_view(&cam, 0.5, me, 0.4, 0.0), "too close in front: under the view");
        // The head pitched down sees the ground at the feet in front, not beside them.
        assert_eq!(tilt_to_see(&cam, 0.5, me, 0.4, 0.0), Some(hand_core::kin::NECK_PITCH_MAX_DEG));
        assert_eq!(tilt_to_see(&cam, 0.5, me, 3.0, 0.2), Some(0.0));
        assert_eq!(tilt_to_see(&cam, 0.5, me, 0.1, 0.3), None);
        let cfg = VisionCfg::default();
        assert_eq!(cfg.look_every_ticks, 15);
        assert!(memory_ok(0.02, 0.06, &cfg) && !memory_ok(0.04, 0.06, &cfg));
    }
}
