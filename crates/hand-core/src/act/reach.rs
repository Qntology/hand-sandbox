//! Bending to reach low: how far down this body can bring its palm, and the
//! least-effort pose that brings it to an object lying in front of it.
//!
//! An object on the ground is grasped from above: the palm centre must come
//! down to the object's top plus a clearance while the fingers wrap its sides
//! (`PALM_CLEARANCE_M`). With the arms of the reference robot (shoulder
//! 1.00 m, upper arm 0.22 m, forearm and palm 0.26 m) a hanging hand stops at
//! 0.52 m, so anything lower needs the body to bend:
//!
//! * a wheeled or tracked base can only pitch its trunk at the waist, and only
//!   when the belly preset has a pitch joint, within that joint's range;
//! * legs add hip flexion (the pelvis pitches over the thighs) and the squat
//!   (knees and ankles), each within its range (`kin::Rom`).
//!
//! The planner searches those joints (coarse grid, then a finer one around the
//! best) and the stand-off distance, solves the arm by two-link inverse
//! kinematics for every candidate, and keeps the pose with the least effort
//! (Σ (τ/τ_peak)², `balance`'s effort metric) whose statics hold with the load
//! in the hand: every joint within its peak and its range, the ZMP inside the
//! support. Nothing is preferred by name: a stoop wins when it costs less than
//! a squat on this body, and the other way round.
//!
//! The bend's duration follows the actuators: a minimum-jerk move's peak speed
//! `1.875·Δq/T` must leave the joint the torque it needs on its torque–speed
//! line, `ω ≤ ω₀·(1 − τ/τ_peak)`, and no posture change is shorter than the
//! arm-lift time `balance` uses (1.0 s).

use serde::Serialize;

use crate::balance::Body;
use crate::kin::{self, KinLoads, KinPose, KinReport, Rom, ANKLE_H_M, PALM_OFFSET_M, SHANK_M, THIGH_M, WHEEL_R_M, BUMPER_GAP_M};
use crate::parts::*;

use super::MIN_JERK_PEAK;

/// The palm stays this far above the object's top while the fingers wrap down its sides, metres (est).
pub const PALM_CLEARANCE_M: f32 = 0.02;
/// Shortest posture change, seconds (`balance::Source::ArmLift` duration).
pub const BEND_MIN_S: f32 = 1.0;
/// Gap kept between the object and the toes or the bumper, metres (est).
pub const STANDOFF_GAP_M: f32 = 0.02;

/// What the hand must reach.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ReachGoal {
    /// Height the palm centre must come down to, metres.
    pub palm_z: f32,
    /// Half the object's length along x (it must sit in front of the toes or the bumper), metres.
    pub half_len_m: f32,
    /// Load lifted off when the hand comes back up, kg, and whether both arms share it.
    pub load_kg: f32,
    pub two_arms: bool,
    pub cargo_kg: f32,
}

impl ReachGoal {
    /// An object of edge `height_m` (and the same length) lying on the ground.
    pub fn ground_object(height_m: f32, load_kg: f32, two_arms: bool, cargo_kg: f32) -> ReachGoal {
        ReachGoal { palm_z: height_m + PALM_CLEARANCE_M, half_len_m: 0.5 * height_m, load_kg, two_arms, cargo_kg }
    }
}

/// The bottom of a bend and how to get there and back.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReachPlan {
    pub ok: bool,
    /// `stand`, `stoop` (waist), `hinge` (hip flexion), `squat`, `squat_hinge`, `squat_stoop`.
    pub strategy: &'static str,
    pub strategy_ko: &'static str,
    /// The pose at the bottom, the palm at the object.
    pub pose: KinPose,
    /// Statics at the bottom with an empty hand and with the load lifted off.
    pub empty: KinReport,
    pub loaded: KinReport,
    /// Where the object's centre is along x from the base origin, metres.
    pub object_x: f32,
    /// The lowest the palm gets with this body at that stand-off, metres (the reason when it is not low enough).
    pub lowest_palm_z: f32,
    pub reason_ko: Option<String>,
    /// Down with an empty hand and back up with the load, seconds.
    pub descend_s: f32,
    pub rise_s: f32,
    /// Candidates the search evaluated.
    pub evaluated: u32,
}

/// The palm in the sagittal plane from the shoulder by two-link inverse
/// kinematics: shoulder flexion relative to the trunk and elbow flexion,
/// degrees, or `None` out of reach. The elbow points back and down, as when
/// reaching to the floor.
pub fn arm_ik(shoulder: [f32; 2], target: [f32; 2], trunk_deg: f32) -> Option<(f32, f32)> {
    let (l1, l2) = (UPPER_ARM_M, FOREARM_M + PALM_OFFSET_M);
    let (dx, dz) = (target[0] - shoulder[0], target[1] - shoulder[1]);
    let d = (dx * dx + dz * dz).sqrt();
    if d > l1 + l2 - 1e-4 || d < (l1 - l2).abs() + 1e-4 {
        return None;
    }
    let elbow = std::f32::consts::PI - ((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)).clamp(-1.0, 1.0).acos();
    let toward = dx.atan2(-dz);
    let beta = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0).acos();
    let upper = toward - beta;
    Some((upper.to_degrees() + trunk_deg, elbow.to_degrees()))
}

/// Ankle dorsiflexion (the shank from vertical) and the thigh's angle from
/// vertical, degrees, of a pose with both feet planted at x = 0 (as the bend
/// candidates are built).
pub fn leg_angles(pose: &KinPose) -> (f32, f32) {
    let knee = kin::knee_ik(pose.hip, [0.0, ANKLE_H_M]).map(|k| k.0).unwrap_or([0.0, ANKLE_H_M + SHANK_M]);
    (knee[0].atan2(knee[1] - ANKLE_H_M).to_degrees(), (knee[0] - pose.hip[0]).atan2(pose.hip[1] - knee[1]).to_degrees())
}

/// The hip of planted legs at an ankle and a thigh angle, degrees.
pub fn hip_at(shank_deg: f32, thigh_deg: f32) -> [f32; 2] {
    let (s, t) = (shank_deg.to_radians(), thigh_deg.to_radians());
    let knee = [SHANK_M * s.sin(), ANKLE_H_M + SHANK_M * s.cos()];
    [knee[0] - THIGH_M * t.sin(), knee[1] + THIGH_M * t.cos()]
}

/// [`KinPose::lerp`] with planted legs blended in joint space: the ankle and
/// the knee move steadily between their two ends as a bend or a rise moves
/// them (a straight hip path can carry the ankle past both ends).
pub fn blend(body: &Body, a: &KinPose, b: &KinPose, u: f32) -> KinPose {
    let mut p = KinPose::lerp(a, b, u);
    let planted = |q: &KinPose| q.feet.iter().all(|f| f[0].abs() < 1e-4 && f[1].abs() < 1e-4);
    if body.legged && planted(a) && planted(b) {
        let ((sa, ta), (sb, tb)) = (leg_angles(a), leg_angles(b));
        p.hip = hip_at(sa + (sb - sa) * u, ta + (tb - ta) * u);
    }
    p
}

/// The shoulder joint in the sagittal plane (x forward, z up) for a pose.
pub fn shoulder_point(body: &Body, pose: &KinPose) -> [f32; 2] {
    let hip = if body.legged { pose.hip } else { [0.0, HIP_M] };
    let p = pose.pelvis_deg.to_radians();
    let t = pose.trunk_deg().to_radians();
    let waist = [hip[0] + kin::PELVIS_M * p.sin(), hip[1] + kin::PELVIS_M * p.cos()];
    [waist[0] + kin::TRUNK_M * t.sin(), waist[1] + kin::TRUNK_M * t.cos()]
}

/// Nearest the object's centre can be to the base origin along x: in front of the toes or the bumper.
pub fn min_object_x(body: &Body, half_len_m: f32) -> f32 {
    let front = if body.legged { 0.75 * body.leg.foot_len_m } else { 0.5 * body.support_len_m + WHEEL_R_M + BUMPER_GAP_M };
    front + half_len_m + STANDOFF_GAP_M
}

struct Cand {
    pose: KinPose,
    strategy: &'static str,
    loaded: KinReport,
    score: f32,
}

/// Classifies a bottom pose by the joints it bends.
fn strategy_of(body: &Body, pose: &KinPose) -> &'static str {
    let squat = body.legged && pose.hip[1] < HIP_M - 0.05;
    let hinge = body.legged && pose.pelvis_deg > 10.0;
    let stoop = pose.waist_deg > 10.0;
    match (squat, hinge, stoop) {
        (false, false, false) => "stand",
        (false, false, true) => "stoop",
        (false, true, _) => "hinge",
        (true, false, false) => "squat",
        (true, true, _) => "squat_hinge",
        (true, false, true) => "squat_stoop",
    }
}

pub fn strategy_ko(s: &str) -> &'static str {
    match s {
        "stand" => "선 채로",
        "stoop" => "허리 굽힘",
        "hinge" => "고관절 굽힘",
        "squat" => "쪼그려 앉기",
        "squat_hinge" => "쪼그려 앉아 고관절 굽힘",
        "squat_stoop" => "쪼그려 앉아 허리 굽힘",
        _ => "—",
    }
}

/// A pose that reaches but does not hold: how far it is from holding, and why.
struct Miss {
    violation: f32,
    reason: String,
}

/// Builds and scores one candidate: the legs (or the base), the trunk, then the arms by IK.
fn candidate(body: &Body, goal: &ReachGoal, x_obj: f32, shank_deg: f32, thigh_deg: f32, pelvis_deg: f32, waist_deg: f32) -> Option<Cand> {
    candidate_or_miss(body, goal, x_obj, shank_deg, thigh_deg, pelvis_deg, waist_deg).ok()
}

fn violation(r: &KinReport) -> f32 {
    r.worst().map_or(0.0, |j| (j.ratio - 1.0).max(0.0)) + (-r.margin).max(0.0) + 0.5 * r.out_of_range.len() as f32
}

fn candidate_or_miss(body: &Body, goal: &ReachGoal, x_obj: f32, shank_deg: f32, thigh_deg: f32, pelvis_deg: f32, waist_deg: f32) -> Result<Cand, Option<Miss>> {
    let mut pose = KinPose::stand(goal.cargo_kg);
    if body.legged {
        // Feet flat at x = 0; the shank leans forward by the dorsiflexion, the thigh back from the knee.
        let (s, t) = (shank_deg.to_radians(), thigh_deg.to_radians());
        let knee = [SHANK_M * s.sin(), ANKLE_H_M + SHANK_M * s.cos()];
        pose.hip = [knee[0] - THIGH_M * t.sin(), knee[1] + THIGH_M * t.cos()];
        pose.pelvis_deg = pelvis_deg;
    }
    pose.waist_deg = waist_deg;
    let sh = shoulder_point(body, &pose);
    let (sd, ed) = arm_ik(sh, [x_obj, goal.palm_z], pose.trunk_deg()).ok_or(None)?;
    pose.shoulder_deg = [sd, if goal.two_arms { sd } else { 0.0 }];
    pose.elbow_deg = [ed, if goal.two_arms { ed } else { 10.0 }];
    // Look at the hand: the neck tilts the camera down toward the object, within its range.
    let eye = [sh[0], sh[1] + kin::NECK_UP_M + 0.07];
    let look = (eye[1] - goal.palm_z).atan2(x_obj - eye[0]).to_degrees();
    pose.neck_deg = (look - pose.trunk_deg()).clamp(-kin::NECK_PITCH_MAX_DEG, kin::NECK_PITCH_MAX_DEG);
    pose.closure = 0.0;
    let loaded = kin::solve_fast(body, &pose, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0));
    if !loaded.feasible {
        return Err(Some(Miss { violation: violation(&loaded), reason: loaded.limiting.clone().unwrap_or_default() }));
    }
    let empty = kin::solve_fast(body, &pose, &KinLoads::default());
    if !empty.feasible {
        return Err(Some(Miss { violation: violation(&empty), reason: empty.limiting.clone().unwrap_or_default() }));
    }
    // Least effort with the load; a low margin costs too (the same weight as one joint at its peak at zero margin).
    let score = loaded.effort_sum + (1.0 - loaded.margin.clamp(0.0, 1.0)).powi(2);
    Ok(Cand { strategy: strategy_of(body, &pose), pose, loaded, score })
}

/// The lowest palm height over the body's ranges at a stand-off (no load, no torque limits): for the reason text.
fn lowest_palm(body: &Body, rom: &Rom, x_obj: f32) -> f32 {
    let mut best = f32::INFINITY;
    let steps = |max: f32, n: usize| (0..=n).map(move |i| max * i as f32 / n as f32);
    let shank_max = if body.legged { rom.ankle_dorsi_max } else { 0.0 };
    let thigh_max = if body.legged { (rom.knee_max - shank_max).max(0.0).min(110.0) } else { 0.0 };
    let pelvis_max = if body.legged { rom.hip_flex_max } else { 0.0 };
    for s in steps(shank_max, 3) {
        for t in steps(thigh_max, 11) {
            for p in steps(pelvis_max, 12) {
                for w in steps(rom.waist_max, 9) {
                    let mut pose = KinPose::stand(0.0);
                    if body.legged {
                        let (sr, tr) = (s.to_radians(), t.to_radians());
                        let knee = [SHANK_M * sr.sin(), ANKLE_H_M + SHANK_M * sr.cos()];
                        pose.hip = [knee[0] - THIGH_M * tr.sin(), knee[1] + THIGH_M * tr.cos()];
                        pose.pelvis_deg = p;
                    }
                    pose.waist_deg = w;
                    let sh = shoulder_point(body, &pose);
                    // The palm reaches the arm's length from the shoulder; at this x the lowest point is straight below if in reach.
                    let reach = UPPER_ARM_M + FOREARM_M + PALM_OFFSET_M;
                    let dx = x_obj - sh[0];
                    if dx.abs() <= reach {
                        best = best.min(sh[1] - (reach * reach - dx * dx).sqrt());
                    }
                }
            }
        }
    }
    best
}

/// Plans the bend that brings the palm to `goal` with the least effort.
pub fn plan(body: &Body, goal: &ReachGoal) -> ReachPlan {
    let rom = Rom::of(body);
    let x0 = min_object_x(body, goal.half_len_m);
    let xs = [x0, x0 + 0.06, x0 + 0.12, x0 + 0.20];
    let mut evaluated = 0u32;
    let mut best: Option<(Cand, f32)> = None;
    let mut nearest: Option<Miss> = None;
    let consider = |c: Option<Cand>, x: f32, best: &mut Option<(Cand, f32)>| {
        if let Some(c) = c {
            if best.as_ref().map_or(true, |(b, _)| c.score < b.score) {
                *best = Some((c, x));
            }
        }
    };
    // Standing reach first: nothing to bend when the palm already gets there.
    let grid = |max: f32, n: usize| -> Vec<f32> { (0..=n).map(|i| max * i as f32 / n.max(1) as f32).collect() };
    let shank_max = if body.legged { rom.ankle_dorsi_max } else { 0.0 };
    let thigh_max = if body.legged { (rom.knee_max - shank_max).max(0.0).min(110.0) } else { 0.0 };
    let pelvis_max = if body.legged { rom.hip_flex_max } else { 0.0 };
    let (ss, ts, ps, ws) = (grid(shank_max, if body.legged { 3 } else { 0 }), grid(thigh_max, if body.legged { 11 } else { 0 }), grid(pelvis_max, if body.legged { 12 } else { 0 }), grid(rom.waist_max, if rom.waist_max > 0.0 { 9 } else { 0 }));
    for &x in &xs {
        for &s in &ss {
            for &t in &ts {
                for &p in &ps {
                    for &w in &ws {
                        evaluated += 1;
                        match candidate_or_miss(body, goal, x, s, t, p, w) {
                            Ok(c) => consider(Some(c), x, &mut best),
                            Err(Some(m)) => {
                                if nearest.as_ref().map_or(true, |n| m.violation < n.violation) {
                                    nearest = Some(m);
                                }
                            }
                            Err(None) => {}
                        }
                    }
                }
            }
        }
    }
    // Refine around the best on half the grid steps.
    if let Some((b, x)) = best.as_ref().map(|(c, x)| (c.pose, *x)) {
        let (s0, t0) = if body.legged {
            let knee = kin::knee_ik(b.hip, [0.0, ANKLE_H_M]).map(|k| k.0).unwrap_or([0.0, ANKLE_H_M + SHANK_M]);
            ((knee[0]).atan2(knee[1] - ANKLE_H_M).to_degrees(), (knee[0] - b.hip[0]).atan2(b.hip[1] - knee[1]).to_degrees())
        } else {
            (0.0, 0.0)
        };
        let d = |max: f32, n: usize| if n == 0 { 0.0 } else { 0.5 * max / n as f32 };
        let (ds, dt, dp, dw) = (d(shank_max, 3), d(thigh_max, 11), d(pelvis_max, 12), d(rom.waist_max, 9));
        for k in [-1.0f32, 0.0, 1.0] {
            for l in [-1.0f32, 0.0, 1.0] {
                for m in [-1.0f32, 0.0, 1.0] {
                    for n in [-1.0f32, 0.0, 1.0] {
                        for dx in [-0.03f32, 0.0, 0.03] {
                            let xx = (x + dx).max(x0);
                            let c = candidate(body, goal, xx, (s0 + k * ds).clamp(0.0, shank_max), (t0 + l * dt).clamp(0.0, thigh_max), (b.pelvis_deg + m * dp).clamp(0.0, pelvis_max), (b.waist_deg + n * dw).clamp(0.0, rom.waist_max));
                            evaluated += 1;
                            consider(c, xx, &mut best);
                        }
                    }
                }
            }
        }
    }
    let stand = KinPose::stand(goal.cargo_kg);
    match best {
        Some((c, x)) => {
            let empty = kin::solve(body, &c.pose, &KinLoads::default());
            let loaded = kin::solve(body, &c.pose, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0));
            let up = kin::solve_fast(body, &stand, &KinLoads::default());
            let descend_s = move_time(&up, &empty);
            let rise_s = move_time(&loaded, &kin::solve_fast(body, &KinPose { cargo_kg: goal.cargo_kg, ..stand }, &KinLoads::holding(goal.load_kg, goal.two_arms, 0.0)));
            let _ = c.loaded;
            ReachPlan { ok: true, strategy: c.strategy, strategy_ko: strategy_ko(c.strategy), pose: c.pose, empty, loaded, object_x: x, lowest_palm_z: goal.palm_z, reason_ko: None, descend_s, rise_s, evaluated }
        }
        None => {
            let low = lowest_palm(body, &rom, x0).max(0.0);
            let why = if low > goal.palm_z + 1e-3 {
                if body.legged {
                    format!("다리를 굽혀도 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m)", low, goal.palm_z)
                } else if rom.waist_max <= 0.0 {
                    format!("허리 피치 관절이 없어 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m) — 바닥의 물건을 집을 수 없음", low, goal.palm_z)
                } else {
                    format!("허리를 {:.0}°까지 굽혀도 손바닥이 {:.2} m 아래로 내려가지 않음 (필요 {:.2} m)", rom.waist_max, low, goal.palm_z)
                }
            } else {
                // Low enough in geometry, but no candidate held: the nearest one's limit (torque, balance or range).
                format!("손은 닿지만 굽힌 자세를 버티지 못함{}", nearest.as_ref().map(|m| format!(": {}", m.reason)).unwrap_or_default())
            };
            let empty = kin::solve(body, &stand, &KinLoads::default());
            ReachPlan { ok: false, strategy: "none", strategy_ko: "불가", pose: stand, loaded: empty.clone(), empty, object_x: x0, lowest_palm_z: low, reason_ko: Some(why), descend_s: 0.0, rise_s: 0.0, evaluated }
        }
    }
}

/// Time to move between two solved poses: every actuated joint's minimum-jerk
/// peak speed within what its torque–speed line leaves at the torque it holds,
/// and no shorter than [`BEND_MIN_S`].
pub fn move_time(a: &KinReport, b: &KinReport) -> f32 {
    let body_speed = |name: &str, r: &KinReport| r.joint(name).map(|j| (j.angle_deg.unwrap_or(0.0), j.ratio, j.peak));
    let mut t = BEND_MIN_S;
    for ja in &a.joints {
        if ja.structural || ja.angle_deg.is_none() {
            continue;
        }
        let Some((qb, rb, _)) = body_speed(&ja.name, b) else { continue };
        let dq = (qb - ja.angle_deg.unwrap_or(0.0)).abs().to_radians();
        if dq < 1e-3 {
            continue;
        }
        let ratio = ja.ratio.max(rb).min(0.95);
        let w0 = joint_speed(&ja.name, ja);
        let allowed = w0 * (1.0 - ratio);
        t = t.max(MIN_JERK_PEAK * dq / allowed.max(1e-3));
    }
    t
}

/// No-load speed of a joint's actuator, rad/s (from its name and actuator string).
fn joint_speed(_name: &str, j: &kin::KinJoint) -> f32 {
    // The actuator table: every preset actuator by name.
    const ALL: [&Actuator; 8] = [&STS3215, &HTD45H, &XM430, &RS01, &RS02, &RS03, &RS04, &BHL6512];
    ALL.iter().find(|a| j.actuator.contains(a.name) || a.name.contains(&j.actuator)).map(|a| a.speed_rad_s).unwrap_or_else(|| {
        // Ganged actuators keep the speed of their unit ("2× HTD-45H", "2× Robstride 02 (평행 링크)").
        ALL.iter().find(|a| j.actuator.contains(a.name.split(' ').last().unwrap_or(""))).map(|a| a.speed_rad_s).unwrap_or(5.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;

    fn body(plan: BodyPlan, base: BaseKind) -> Body {
        Body::new(&plan, base, FingerCount::Five)
    }

    #[test]
    fn the_arm_ik_puts_the_palm_on_the_target() {
        let b = body(BodyPlan::default(), BaseKind::Wheel);
        let mut pose = KinPose::stand(0.0);
        let (sd, ed) = arm_ik([0.0, SHOULDER_M], [0.30, 0.70], 0.0).unwrap();
        pose.shoulder_deg = [sd, 0.0];
        pose.elbow_deg = [ed, 10.0];
        let r = kin::solve(&b, &pose, &KinLoads::default());
        assert!((r.hands[0][0] - 0.30).abs() < 2e-3 && (r.hands[0][2] - 0.70).abs() < 2e-3, "{:?}", r.hands[0]);
        assert!(arm_ik([0.0, SHOULDER_M], [0.0, 0.40], 0.0).is_none(), "below the hanging hand");
    }

    #[test]
    fn a_rigid_trunk_on_wheels_cannot_reach_the_ground() {
        let b = body(BodyPlan::default(), BaseKind::Wheel);
        let p = plan(&b, &ReachGoal::ground_object(0.15, 2.4, false, 0.0));
        assert!(!p.ok, "{p:?}");
        assert!(p.lowest_palm_z > 0.4, "{}", p.lowest_palm_z);
        assert!(p.reason_ko.as_deref().unwrap_or("").contains("허리 피치"), "{:?}", p.reason_ko);
    }

    #[test]
    fn mid_legs_squat_and_hinge_to_reach_a_log() {
        let b = body(BodyPlan::default(), BaseKind::HumanFoot);
        let p = plan(&b, &ReachGoal::ground_object(0.15, 2.4, false, 0.0));
        assert!(p.ok, "{:?} lowest {}", p.reason_ko, p.lowest_palm_z);
        assert!(p.strategy.starts_with("squat") || p.strategy == "hinge", "{}", p.strategy);
        assert!((p.loaded.hands[0][2] - p.loaded.pose.hip[1]).is_finite());
        assert!((p.empty.hands[0][2] - (0.15 + PALM_CLEARANCE_M)).abs() < 3e-3, "palm at {:?}", p.empty.hands[0]);
        assert!(p.descend_s >= BEND_MIN_S && p.rise_s >= BEND_MIN_S);
        assert!(p.loaded.margin >= 0.0 && p.loaded.feasible);
    }

    #[test]
    fn a_pro_waist_pitch_alone_does_not_get_a_wheeled_hand_to_a_stone() {
        // 90° of waist pitch puts the shoulder 0.70 m up and 0.30 m ahead: the palm stops near 0.22 m.
        let b = body(BodyPlan::uniform(Tier::Pro), BaseKind::Wheel);
        let p = plan(&b, &ReachGoal::ground_object(0.10, 2.6, false, 0.0));
        assert!(!p.ok);
        assert!(p.lowest_palm_z > 0.12 && p.lowest_palm_z < 0.35, "{}", p.lowest_palm_z);
        // Something at table height it reaches by stooping.
        let t = plan(&b, &ReachGoal { palm_z: 0.40, half_len_m: 0.05, load_kg: 1.0, two_arms: false, cargo_kg: 0.0 });
        assert!(t.ok && t.strategy == "stoop", "{t:?}");
    }

    #[test]
    #[ignore]
    fn print_plans() {
        for plan_id in ["budget", "mid", "pro", "pro_arms"] {
            for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track] {
                let b = body(BodyPlan::named(plan_id).unwrap(), base);
                for (h, kg) in [(0.15f32, 2.4f32), (0.10, 2.6), (0.10, 4.0)] {
                    let p = plan(&b, &ReachGoal::ground_object(h, kg, false, 0.0));
                    let j = |n: &str| p.loaded.joint(n).map(|j| format!("{n} {:.0}° {:.1}N·m r{:.2}", j.angle_deg.unwrap_or(0.0), j.torque.abs(), j.ratio)).unwrap_or_default();
                    println!("[REACH] {plan_id:<8} {base:?} h{h} {kg}kg ok={} {} x={:.2} low={:.2} down {:.2}s up {:.2}s margin {:.2} eval {} | {} | {} | {} | {} | {} | {:?}",
                        p.ok, p.strategy, p.object_x, p.lowest_palm_z, p.descend_s, p.rise_s, p.loaded.margin, p.evaluated,
                        j("knee_r"), j("hip_pitch_r"), j("ankle_r"), j("waist_pitch"), j("shoulder_r"), p.reason_ko);
                }
            }
        }
    }
}
