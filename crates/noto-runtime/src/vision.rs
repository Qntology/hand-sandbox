use noto_sds::{Scope, SdsStore, Track};
use serde::{Deserialize, Serialize};
use world_sim::odometry::{LEG_SCALE_SD, RANDOM_WALK_M, WHEEL_SCALE_SD};
use world_sim::Camera;

pub const LOOK_AHEAD_S: f32 = 1.5;
pub const EYE_LEAD_S: f32 = 1.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionCfg {
    pub intermittent: bool,
    pub look_every_ticks: u32,
    pub sigmas: f32,
    #[serde(default = "gaze_on")]
    pub gaze_before_pick: bool,
}

fn gaze_on() -> bool {
    true
}

impl Default for VisionCfg {
    fn default() -> Self {
        VisionCfg { intermittent: true, look_every_ticks: (LOOK_AHEAD_S / world_sim::TICK_S).round() as u32, sigmas: 2.0, gaze_before_pick: true }
    }
}

/// What the robot saw when it bent its head over an item before picking it up: the scan's match (label, score, box),
/// how far away the scan put it, and the range the depth model gives inside that box.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GazeCheck {
    pub tick: u64,
    pub drop: u32,
    pub label_ko: String,
    pub ok: bool,
    pub score: Option<f32>,
    pub range_m: Option<f32>,
    pub depth_m: Option<f32>,
    pub bbox: Option<[f32; 4]>,
    pub reason: Option<String>,
    pub neck_deg: f32,
    pub pitch_deg: f32,
}

/// One look at an item through one camera: the scan's match of its kind nearest the memory (its label, score and
/// box), where it puts the item on the ground and how sure that is, and the depth model's range inside the box.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sighting {
    pub tick: u64,
    /// "head" (the walker's head camera, or a lift truck rider's) or "fork" (the carriage camera between the tines).
    pub camera: &'static str,
    pub drop: u32,
    pub label_ko: String,
    pub ok: bool,
    /// The match lies inside the memory's gate (not a neighbour of the same kind).
    pub matched: bool,
    pub score: Option<f32>,
    pub at: Option<(f32, f32)>,
    pub sigma_m: Option<f32>,
    pub range_m: Option<f32>,
    pub depth_m: Option<f32>,
    pub bbox: Option<[f32; 4]>,
    pub reason: Option<String>,
    pub neck_deg: f32,
    pub pitch_deg: f32,
}

impl Sighting {
    pub fn gaze_check(&self) -> GazeCheck {
        GazeCheck { tick: self.tick, drop: self.drop, label_ko: self.label_ko.clone(), ok: self.ok, score: self.score, range_m: self.range_m, depth_m: self.depth_m, bbox: self.bbox, reason: self.reason.clone(), neck_deg: self.neck_deg, pitch_deg: self.pitch_deg }
    }
}

/// The least two sightings of one item may be apart and still be the same (m) [추정: the 0.10 m items' half size
/// and the depth model's range error at a metre].
pub const CROSS_MIN_M: f32 = 0.08;

/// How far apart two sightings put the item and how far they may be (3σ of both, at least `CROSS_MIN_M`).
pub fn cross_gap(a: &Sighting, b: &Sighting) -> Option<(f32, f32)> {
    let (pa, pb) = (a.at?, b.at?);
    let gate = (3.0 * a.sigma_m.unwrap_or(0.05).hypot(b.sigma_m.unwrap_or(0.05))).max(CROSS_MIN_M);
    Some(((pa.0 - pb.0).hypot(pa.1 - pb.1), gate))
}

/// A lift truck's two eyes on one item, each checked against the other, as an operator and a fork camera do: the
/// rider looks down at it before the tines go down, the carriage camera looks at it with them down, and once the item
/// fills the carriage camera's view the rider looks down at it again.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CrossCheck {
    pub drop: u32,
    pub rider: Option<Sighting>,
    pub fork: Option<Sighting>,
    pub recheck: Option<Sighting>,
    /// The share of the carriage camera's image the item filled when the base stopped for the re-check.
    pub blocked_share: Option<f32>,
    /// Rider vs carriage camera, and the re-check vs the carriage camera: (apart, allowed) in m.
    pub fork_vs_rider: Option<(f32, f32)>,
    pub recheck_vs_fork: Option<(f32, f32)>,
}

impl CrossCheck {
    pub fn new(drop: u32) -> CrossCheck {
        CrossCheck { drop, ..CrossCheck::default() }
    }

    /// "agree", "disagree", "fork_only", "rider_only" or "none": what the two looks before the tines go in say. A
    /// look sees the item when its match lies inside the memory's gate (a neighbour of the same kind does not count).
    pub fn verdict(&self) -> &'static str {
        let seen = |s: &Option<Sighting>| s.as_ref().is_some_and(|s| s.ok && s.matched);
        match (seen(&self.rider), seen(&self.fork), self.fork_vs_rider) {
            (true, true, Some((d, g))) if d <= g => "agree",
            (true, true, Some(_)) => "disagree",
            (true, true, None) => "agree",
            (false, true, _) => "fork_only",
            (true, false, _) => "rider_only",
            (false, false, _) => "none",
        }
    }

    /// Whether the re-check found the item where the carriage camera last saw it (`None` before the re-check).
    pub fn seated(&self) -> Option<bool> {
        let r = self.recheck.as_ref()?;
        Some(r.ok && r.matched && self.recheck_vs_fork.map_or(true, |(d, g)| d <= g))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Memory {
    pub x: f32,
    pub z: f32,
    pub sigma0_m: f32,
    pub odo_m: f32,
    pub tick: u64,
}

impl Memory {
    pub fn seen(x: f32, z: f32, sigma0_m: f32, odo_m: f32, tick: u64) -> Memory {
        Memory { x, z, sigma0_m, odo_m, tick }
    }

    pub fn sigma_m(&self, odo_now: f32, rate: DriftRate) -> f32 {
        let d = (odo_now - self.odo_m).max(0.0);
        (self.sigma0_m * self.sigma0_m + (rate.per_m * d).powi(2) + rate.walk_m * rate.walk_m * d).sqrt()
    }

    pub fn fuse(&self, x: f32, z: f32, sigma_obs: f32, odo_now: f32, tick: u64, rate: DriftRate) -> (Memory, f32) {
        let pm = self.sigma_m(odo_now, rate).powi(2);
        let po = sigma_obs * sigma_obs;
        let k = pm / (pm + po).max(1e-9);
        let innov = (x - self.x).hypot(z - self.z);
        let fused = Memory { x: self.x + k * (x - self.x), z: self.z + k * (z - self.z), sigma0_m: (pm * po / (pm + po).max(1e-9)).sqrt(), odo_m: odo_now, tick };
        (fused, innov)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DriftRate {
    pub per_m: f32,
    pub walk_m: f32,
    pub source: &'static str,
}

impl DriftRate {
    pub fn spec(legged: bool) -> DriftRate {
        DriftRate { per_m: if legged { LEG_SCALE_SD } else { WHEEL_SCALE_SD }, walk_m: RANDOM_WALK_M, source: "spec" }
    }

    pub fn learned(sds: &SdsStore, base: &str, legged: bool) -> DriftRate {
        let spec = DriftRate::spec(legged);
        match sds.adaptive_baseline(&Scope::new(Track::Memory, base, "drift"), "drift_var_per_m2") {
            Some((mean, _, _)) if mean.is_finite() && mean > 0.0 => DriftRate { per_m: (mean as f32).sqrt(), source: "sds", ..spec },
            _ => spec,
        }
    }
}

pub fn memory_ok(sigma_m: f32, tolerance_m: f32, cfg: &VisionCfg) -> bool {
    cfg.sigmas * sigma_m <= tolerance_m
}

pub fn in_view(cam: &Camera, eye_m: f32, pose: (f32, f32, f32), x: f32, z: f32) -> bool {
    let (dx, dz) = (x - pose.0, z - pose.1);
    let d = dx.hypot(dz);
    let bearing = wrap(dz.atan2(dx) - pose.2).abs();
    let half_h = 0.5 * cam.fov_deg.to_radians() - 0.05;
    let half_v = (0.5 * cam.vfov_deg().to_radians() - 0.05).max(0.05);
    let center_dep = (-cam.pitch_deg).to_radians();
    let low = center_dep + half_v;
    let high = center_dep - half_v;
    let near = if low > 1e-3 { eye_m / low.tan() } else { 0.0 };
    let far = if high > 1e-3 { (eye_m / high.tan()).min(cam.max_dist_m) } else { cam.max_dist_m };
    bearing <= half_h && d >= near && d <= far
}

pub fn tilt_to_see(cam: &Camera, eye_m: f32, pose: (f32, f32, f32), x: f32, z: f32) -> Option<f32> {
    if in_view(cam, eye_m, pose, x, z) {
        return Some(0.0);
    }
    let d = (x - pose.0).hypot(z - pose.1).max(0.1);
    let dep_deg = eye_m.atan2(d).to_degrees();
    let ideal_tilt = (dep_deg + cam.pitch_deg).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
    for tilt in [ideal_tilt, noto_core::kin::NECK_PITCH_MAX_DEG] {
        let down = Camera { pitch_deg: cam.pitch_deg - tilt, ..*cam };
        if in_view(&down, eye_m, pose, x, z) {
            return Some(tilt);
        }
    }
    None
}

pub fn gaze_to_see(cam: &Camera, eye_m: f32, pose: (f32, f32, f32), x: f32, z: f32) -> Option<(f32, f32)> {
    let d = (x - pose.0).hypot(z - pose.1).max(0.1);
    let dep_deg = eye_m.atan2(d).to_degrees();
    let comfortable_low = -cam.pitch_deg + 0.30 * cam.vfov_deg();
    if in_view(cam, eye_m, pose, x, z) && dep_deg <= comfortable_low {
        return Some((0.0, 0.0));
    }
    let max_pan = noto_core::kin::NECK_PAN_MAX_DEG;
    let bearing = wrap((z - pose.1).atan2(x - pose.0) - pose.2).to_degrees();
    let pan = (-bearing).clamp(-max_pan, max_pan);
    let turned = (pose.0, pose.1, pose.2 - pan.to_radians());
    let ideal_tilt = (dep_deg + cam.pitch_deg).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
    for tilt in [0.0, ideal_tilt, noto_core::kin::NECK_PITCH_MAX_DEG] {
        let c = Camera { pitch_deg: cam.pitch_deg - tilt, ..*cam };
        if in_view(&c, eye_m, turned, x, z) {
            if tilt == 0.0 && dep_deg > comfortable_low {
                continue;
            }
            return Some((pan, tilt));
        }
    }
    None
}

#[derive(Clone, Debug, Serialize)]
pub struct VisionMark {
    pub tick: u64,
    pub kind: String,
    pub what: String,
    pub sigma_before_m: Option<f32>,
    pub sigma_after_m: Option<f32>,
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
        let (f, innov) = m.fuse(3.2, 1.0, 0.03, 20.0, 5, rate);
        assert!((f.x - 3.2).abs() < 0.01 && (innov - 0.2).abs() < 1e-5 && f.sigma0_m < 0.031);
        let (g, _) = m.fuse(3.2, 1.0, 0.03, 10.0, 5, rate);
        assert!((g.x - 3.1).abs() < 1e-4, "{}", g.x);
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
        let ideal = (0.5f32.atan2(0.4).to_degrees() + cam.pitch_deg).clamp(0.0, noto_core::kin::NECK_PITCH_MAX_DEG);
        assert_eq!(tilt_to_see(&cam, 0.5, me, 0.4, 0.0), Some(ideal), "just enough tilt to put the point in view");
        assert_eq!(tilt_to_see(&cam, 0.5, me, 3.0, 0.2), Some(0.0));
        assert_eq!(tilt_to_see(&cam, 0.5, me, 0.1, 0.3), None);
        let (pan, tilt) = gaze_to_see(&cam, 1.14, me, 1.0, 2.5).expect("to the right, ahead");
        assert!(pan < -30.0 && tilt == 0.0, "pan {pan} tilt {tilt}");
        let (pan, tilt) = gaze_to_see(&cam, 1.14, me, 0.15, -0.35).expect("beside the left foot");
        assert!(pan > 30.0 && tilt > 0.0, "pan {pan} tilt {tilt}");
        assert_eq!(gaze_to_see(&cam, 1.14, me, -2.0, 0.0), None);
        assert_eq!(gaze_to_see(&cam, 1.14, me, 3.0, 0.2), Some((0.0, 0.0)));
        let cfg = VisionCfg::default();
        assert_eq!(cfg.look_every_ticks, 15);
        assert!(memory_ok(0.02, 0.06, &cfg) && !memory_ok(0.04, 0.06, &cfg));
    }
}
