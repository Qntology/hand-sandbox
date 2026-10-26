//! Grasping an object off the ground and setting one down, as time series.
//!
//! **Grasp** (`grasp`): bend down (`reach`), close the fingers, squeeze,
//! lift off, rise with the object, stow it in the bay.
//!
//! * The hand arrives where the robot believes the object is; the belief is
//!   off by a drawn aim error (the head camera's and the arm's own errors).
//!   A palm that misses the object's top face closes on air.
//! * The fingers meet the object at their closing speed: a collision whose
//!   peak force follows the soft pad's stiffness (`material::impact`).
//! * The controller squeezes `GRIP_SAFETY` times the load it expects over the
//!   friction it believes (the table mean), capped by what the fingers can
//!   press (`sim::grasp`'s finger contacts; from above, the palm carries none
//!   of the weight). The pads' real friction is drawn from the measured scatter.
//! * Lift-off comes once the arm's pull reaches the weight; then the hand
//!   rises with the body on a minimum-jerk path and the load is the object's
//!   weight plus its inertia. The object slides out when the load passes
//!   friction times squeeze. When its weight's moment about the grasp (the
//!   aim error is the lever) passes the soft contacts' torsional friction, a
//!   pinch lets it pivot until its centre of mass hangs under the grasp axis
//!   (it stays held, tilted); a power grasp's palm on top blocks the turn.
//! * Every sample is solved by `kin` with the object's force on the hand:
//!   joints over their peak or a ZMP outside the support stop the lift.
//!
//! **Release** (`release`): take the object from the bay, bend down as far as
//! the body allows over the ground or a pile, open the hand. The object falls
//! the rest of the way, hits with the ground's stiffness, rebounds with its
//! restitution and lands a little away. A body that cannot bend drops it from
//! hand height.

use serde::Serialize;

use crate::balance::{Body, CARGO_X_M, CARGO_Z_M, G, GRIP_SAFETY};
use crate::hand::Hand;
use crate::kin::{self, KinLoads, KinPose};
use crate::parts::*;
use crate::sim::{grasp as close_hand, ContactSource, GraspSetup, Payload};
use crate::surface::Surface;

use super::material::{self, ItemMaterial};
use super::reach::{self, ReachGoal, ReachMode};
use super::{min_jerk, ActEvent, ActKind, ActRollout, ActSample, Draw, Grip, Outcome, Part, SAMPLE_S};

/// Closed-loop rise time of a joint position loop (0.35 / bandwidth, the 4 Hz
/// of `dynamics::DynCfg`), seconds: how fast the squeeze and the pull build.
pub const LOOP_RISE_S: f32 = 0.35 / 4.0;
/// Finger closing from open to contact: the knuckle swings this much, radians (est, half its 90° range).
pub const CLOSE_SWING_RAD: f32 = 0.8;
/// Lever of the fingertip about the knuckle, metres (the 0.083 m chain at mid closure, est).
pub const TIP_LEVER_M: f32 = 0.06;
/// Height of the bay's rim the hand releases over, and how far above it, metres (est).
pub const BAY_RIM_Z_M: f32 = CARGO_Z_M + 0.08;
pub const BAY_DROP_M: f32 = 0.03;
/// The slowest lift the controller plans when the squeeze is near its limit, seconds (est: three arm-lift times).
pub const LIFT_SLOWEST_S: f32 = 3.0;
/// Peak acceleration of a minimum-jerk move per unit of `Δ / T²` (10/√3).
pub const MIN_JERK_ACC: f32 = 5.77;

/// What is grasped and where.
#[derive(Clone, Copy, Debug)]
pub struct GraspInput<'a> {
    pub body: &'a Body,
    pub hand: &'a Hand,
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub size_m: [f32; 3],
    /// The ground under the object.
    pub ground: Surface,
    pub two_arms: bool,
    /// Cargo already in the bay, kg.
    pub cargo_kg: f32,
    /// 1σ of the hand's aim error along each horizontal axis, metres.
    pub aim_sigma_m: f32,
    pub seed: u64,
    /// Where the hand meets the object: in front, or beside the right foot
    /// without facing it (`reach::ReachMode`).
    pub mode: ReachMode,
    /// Where the robot believes the object is minus where it is, in the base
    /// frame (x forward, y left), metres: what its memory got wrong since it
    /// last saw the object (odometry drift). Zero when it sees it as it reaches.
    pub aim_offset_m: [f32; 2],
    /// 1σ the robot itself gives that memory, metres (0 when it sees the object):
    /// it expects its aim to be off by this as well.
    pub memory_sigma_m: f32,
}

impl<'a> GraspInput<'a> {
    /// A grasp of an object the robot sees in front of it (the pick before v0.7).
    pub fn front(body: &'a Body, hand: &'a Hand, material: ItemMaterial, mass_kg: f32, size_m: [f32; 3], ground: Surface, two_arms: bool, cargo_kg: f32, aim_sigma_m: f32, seed: u64) -> GraspInput<'a> {
        GraspInput { body, hand, material, mass_kg, size_m, ground, two_arms, cargo_kg, aim_sigma_m, seed, mode: ReachMode::Front, aim_offset_m: [0.0; 2], memory_sigma_m: 0.0 }
    }
}

/// What the fingers can do on this object: total finger squeeze (no palm), contacts, finger joints.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HandCapacity {
    pub squeeze_n: f32,
    pub contacts: u8,
    /// Per finger: contact force at the knuckle's maximum torque, N.
    pub finger_n: Vec<f32>,
    pub grasp_type: String,
}

/// Whether a hand lifts an object off the ground from above at the friction
/// the controller believes, with its safety factor: the fingers' squeeze times
/// the friction must carry `GRIP_SAFETY` times the weight (the palm on top
/// carries none of it). The squeeze and the answer.
pub fn top_grasp_ok(hand: &Hand, size_m: [f32; 3], mass_kg: f32, mu_est: f32) -> (f32, bool) {
    let c = capacity(hand, size_m, mass_kg, mu_est);
    (c.squeeze_n, c.contacts >= 2 && mu_est * c.squeeze_n >= GRIP_SAFETY * mass_kg * G)
}

/// The fingers closed on the object at the friction the controller believes.
pub fn capacity(hand: &Hand, size_m: [f32; 3], mass_kg: f32, mu_est: f32) -> HandCapacity {
    let out = close_hand(hand, &Payload { name: "object".into(), size_m, mass_kg, friction: mu_est }, &GraspSetup::default());
    let mut finger_n = vec![0.0f32; hand.fingers.len()];
    for c in &out.result.contacts {
        if let ContactSource::Finger { index } = c.source {
            finger_n[index] = c.force_n;
        }
    }
    let contacts = finger_n.iter().filter(|f| **f > 0.0).count() as u8;
    HandCapacity { squeeze_n: finger_n.iter().sum(), contacts, finger_n, grasp_type: format!("{:?}", out.result.grasp_type).to_lowercase() }
}

/// The pose that holds the object in front of the chest after the lift (the
/// `balance` arm-lift: upper arm 30° forward, forearm horizontal).
pub fn hold_pose(two_arms: bool, cargo_kg: f32) -> KinPose {
    KinPose { shoulder_deg: [30.0, if two_arms { 30.0 } else { 0.0 }], elbow_deg: [60.0, if two_arms { 60.0 } else { 10.0 }], closure: 1.0, ..KinPose::stand(cargo_kg) }
}

/// The arms that hold the object.
fn holding_arms(two_arms: bool) -> &'static [usize] {
    if two_arms {
        &[0, 1]
    } else {
        &[0]
    }
}

/// `p` with the holding arms reaching the hand point `at` (x forward, z up,
/// by the arm's inverse kinematics from the pose's shoulder). A point past
/// the arm's length is reached as far as the straight arm goes along the line
/// to it, so the hand moves on without a jump.
pub fn arms_to(body: &Body, p: &KinPose, at: [f32; 2], two_arms: bool) -> KinPose {
    let mut out = *p;
    let sh = reach::shoulder_point(body, p);
    let (dx, dz) = (at[0] - sh[0], at[1] - sh[1]);
    let k = ((UPPER_ARM_M + FOREARM_M + kin::PALM_OFFSET_M - 0.002) / (dx * dx + dz * dz).sqrt().max(1e-6)).min(1.0);
    if let Some((sd, ed)) = reach::arm_ik(sh, [sh[0] + dx * k, sh[1] + dz * k], p.trunk_deg()) {
        for &k in holding_arms(two_arms) {
            out.shoulder_deg[k] = sd;
            out.elbow_deg[k] = ed;
        }
    }
    out
}

/// `p` with the holding arms reaching the palm point `at` (x forward, y left,
/// z up) by [`reach::arm_ik3`]: the right palm at `at`, the left (with both
/// arms) at its mirror across the middle. A point past an arm's length is
/// reached as far as the straight arm goes along the line to it, as in
/// [`arms_to`].
pub fn arms_to3(body: &Body, p: &KinPose, at: [f32; 3], two_arms: bool) -> KinPose {
    let mut out = *p;
    let sh = reach::shoulder_point(body, p);
    let len = UPPER_ARM_M + FOREARM_M + kin::PALM_OFFSET_M - 0.002;
    for &k in holding_arms(two_arms) {
        let (sy, ty) = if k == 0 { (-SHOULDER_HALF_WIDTH_M, at[1]) } else { (SHOULDER_HALF_WIDTH_M, -at[1]) };
        let s3 = [sh[0], sy, sh[1]];
        let d = [at[0] - s3[0], ty - s3[1], at[2] - s3[2]];
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        let f = (len / n).min(1.0);
        if let Some((sd, ed, ad)) = reach::arm_ik3(s3, [s3[0] + d[0] * f, s3[1] + d[1] * f, s3[2] + d[2] * f], p.trunk_deg(), k) {
            out.shoulder_deg[k] = sd;
            out.elbow_deg[k] = ed;
            out.shoulder_abd_deg[k] = ad;
        }
    }
    out
}

/// The path of a lift: the body straightens from `bottom` to `hold`'s over
/// `rise_s` while the hand carries the load straight up from where it took it
/// to the hold's height (the load stays over its spot, close to the body,
/// instead of swinging out with the arms), then the arms bring it to `hold`
/// over `curl_s`. Played backwards it sets a load down. An arm turned out at
/// the bottom (a reach to the side, hands between the feet) is solved in
/// three dimensions; a sagittal one as before.
struct LiftPath {
    bottom: KinPose,
    top: KinPose,
    hold: KinPose,
    from: [f32; 3],
    up_z: f32,
    two_arms: bool,
    rise_s: f32,
    curl_s: f32,
    spatial: bool,
}

impl LiftPath {
    fn new(body: &Body, bottom: &KinPose, hold: &KinPose, two_arms: bool, rise_s: f32) -> LiftPath {
        let start = kin::solve_fast(body, bottom, &KinLoads::default()).hands[0];
        let end = kin::solve_fast(body, hold, &KinLoads::default()).hands[0];
        let spatial = bottom.shoulder_abd_deg.iter().any(|a| a.abs() > 1e-6);
        let held = KinPose { closure: hold.closure, ..*hold };
        let top = if spatial { arms_to3(body, &held, [start[0], start[1], end[2]], two_arms) } else { arms_to(body, &held, [start[0], end[2]], two_arms) };
        let curl_s = reach::move_time(&kin::solve_fast(body, &top, &KinLoads::default()), &kin::solve_fast(body, hold, &KinLoads::default()));
        LiftPath { bottom: *bottom, top, hold: *hold, from: start, up_z: end[2], two_arms, rise_s, curl_s, spatial }
    }

    fn duration(&self) -> f32 {
        self.rise_s + self.curl_s
    }

    /// The pose `t` seconds into the lift.
    fn at(&self, body: &Body, t: f32) -> KinPose {
        if t <= self.rise_s {
            let u = min_jerk((t / self.rise_s.max(1e-3)).clamp(0.0, 1.0));
            let p = reach::blend(body, &self.bottom, &self.top, u);
            let z = self.from[2] + (self.up_z - self.from[2]) * u;
            if self.spatial {
                arms_to3(body, &p, [self.from[0], self.from[1], z], self.two_arms)
            } else {
                arms_to(body, &p, [self.from[0], z], self.two_arms)
            }
        } else {
            KinPose::lerp(&self.top, &self.hold, min_jerk(((t - self.rise_s) / self.curl_s.max(1e-3)).clamp(0.0, 1.0)))
        }
    }
}

/// The pose that puts the hand over the bay.
pub fn bay_pose(two_arms: bool, cargo_kg: f32) -> KinPose {
    let target = [CARGO_X_M + 0.12, BAY_RIM_Z_M + BAY_DROP_M + 0.05];
    let (sd, ed) = reach::arm_ik([0.0, SHOULDER_M], target, 0.0).unwrap_or((30.0, 60.0));
    KinPose { shoulder_deg: [sd, if two_arms { sd } else { 0.0 }], elbow_deg: [ed, if two_arms { ed } else { 10.0 }], closure: 1.0, ..KinPose::stand(cargo_kg) }
}

struct Series {
    samples: Vec<ActSample>,
    events: Vec<ActEvent>,
    peak_ratio: f32,
    min_margin: f32,
    limiting: Option<String>,
}

impl Series {
    fn new() -> Series {
        Series { samples: Vec::new(), events: Vec::new(), peak_ratio: 0.0, min_margin: f32::INFINITY, limiting: None }
    }

    /// Solves and keeps one sample; returns whether the statics hold.
    fn push(&mut self, body: &Body, t: f32, phase: &'static str, pose: KinPose, loads: KinLoads, grip: Option<Grip>, object: Option<[f32; 3]>) -> bool {
        let r = kin::solve_fast(body, &pose, &loads);
        let pr = r.worst().map_or(0.0, |j| j.ratio);
        self.peak_ratio = self.peak_ratio.max(pr);
        self.min_margin = self.min_margin.min(r.margin);
        if !r.feasible && self.limiting.is_none() {
            self.limiting = r.limiting.clone();
        }
        self.samples.push(ActSample { t, phase, pose, loads, margin: r.margin, zmp: r.zmp, com: r.com, peak_ratio: pr, grip, object });
        r.feasible
    }

    fn end(&self) -> f32 {
        self.samples.last().map_or(0.0, |s| s.t)
    }
}

fn hand_force(two_arms: bool, f: [f32; 3]) -> [[f32; 3]; 2] {
    if two_arms {
        [[0.5 * f[0], 0.5 * f[1], 0.5 * f[2]], [0.5 * f[0], 0.5 * f[1], 0.5 * f[2]]]
    } else {
        [f, [0.0; 3]]
    }
}

/// Plays a grasp out.
pub fn grasp(inp: &GraspInput) -> ActRollout {
    let body = inp.body;
    let mut d = Draw::new(inp.seed);
    let mut s = Series::new();
    let mut notes = Vec::new();
    let mut metrics: Vec<(&'static str, f32)> = Vec::new();
    let half = [0.5 * inp.size_m[0], 0.5 * inp.size_m[2]];
    let goal = ReachGoal::ground_object(inp.size_m[1], inp.mass_kg, inp.two_arms, inp.cargo_kg);
    // In front: the bend, or a wide squat over a load the bend cannot hold. Beside the foot: the arm turned out.
    let plan = match inp.mode {
        ReachMode::Front => reach::plan_or_straddle(body, &goal, half[1]),
        ReachMode::Side => reach::plan_side(body, &goal, half[1]),
    };
    let stand = KinPose::stand(inp.cargo_kg);
    if !plan.ok {
        let at = if inp.mode == ReachMode::Side { [plan.object_x, plan.object_y, 0.5 * inp.size_m[1]] } else { [reach::min_object_x(body, half[0]), 0.0, 0.5 * inp.size_m[1]] };
        s.push(body, 0.0, "reach", stand, KinLoads::default(), None, Some(at));
        let why = plan.reason_ko.clone().unwrap_or_default();
        s.events.push(ActEvent::new(0.0, "reach_fail", why.clone(), &[Part::Waist, Part::Hand, Part::FootJoints], Some(plan.lowest_palm_z), false));
        metrics.push(("lowest_palm_z", plan.lowest_palm_z));
        let code = if inp.mode == ReachMode::Side { "reach_side" } else { "reach_ground" };
        return ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::fail(code, why), duration_s: 0.0, metrics, notes };
    }
    metrics.push(("lowest_palm_z", plan.lowest_palm_z));
    // A load held between the feet takes both hands; a reach to the side, the right one.
    let two_arms = match plan.approach {
        "straddle" => true,
        "side" => false,
        _ => inp.two_arms,
    };
    if plan.approach != "front" {
        metrics.push(("stance_out_m", plan.pose.stance_out_m));
        metrics.push(("reach_y_m", plan.object_y));
        metrics.push(("abduction_deg", plan.pose.shoulder_abd_deg[0]));
    }
    let ox = plan.object_x;
    let obj_ground = [ox, plan.object_y, 0.5 * inp.size_m[1]];
    let mu = material::pad_on(inp.material);
    let mu_act = d.lognormal(mu.mean, mu.cov);
    // The hand goes where the robot believes the object is: its memory's error plus the camera's and the arm's.
    let aim = [inp.aim_offset_m[0] + inp.aim_sigma_m * d.normal(), inp.aim_offset_m[1] + inp.aim_sigma_m * d.normal()];
    if inp.memory_sigma_m > 0.0 || inp.aim_offset_m != [0.0; 2] {
        metrics.push(("memory_offset_m", inp.aim_offset_m[0].hypot(inp.aim_offset_m[1])));
        metrics.push(("memory_sigma_m", inp.memory_sigma_m));
    }
    let per_hand_kg = if two_arms { 0.5 * inp.mass_kg } else { inp.mass_kg };
    let cap = capacity(inp.hand, inp.size_m, per_hand_kg, mu.mean);
    // The controller expects the minimum-jerk lift's peak acceleration, 10/√3 · Δz / T².
    let rise_dz = (kin::solve_fast(body, &hold_pose(two_arms, inp.cargo_kg), &KinLoads::default()).hands[0][2] - plan.empty.hands[0][2]).max(0.05);
    let a_plan = MIN_JERK_ACC * rise_dz / (plan.rise_s * plan.rise_s);
    let grip_target = GRIP_SAFETY * per_hand_kg * (G + a_plan) / mu.mean.max(0.05);
    let grip = grip_target.min(cap.squeeze_n);
    // A squeeze at its limit: lift slower so the expected load stays within the safety factor.
    let a_room = mu.mean * grip / (GRIP_SAFETY * per_hand_kg.max(1e-3)) - G;
    let rise_s = if a_room >= a_plan {
        plan.rise_s
    } else {
        // A plan already slower than that (a torso lift's travel) keeps its own pace.
        let slowest = LIFT_SLOWEST_S.max(plan.rise_s);
        let t = if a_room > 1e-3 { (MIN_JERK_ACC * rise_dz / a_room).sqrt() } else { slowest };
        notes.push(format!("목표 쥠 {:.0} N > 손가락 한계 {:.0} N: 한계로 쥐고 {:.1} s에 걸쳐 천천히 듦", grip_target, cap.squeeze_n, t.clamp(plan.rise_s, slowest)));
        t.clamp(plan.rise_s, slowest)
    };
    metrics.push(("lift_s", rise_s));
    // What the robot expects before it tries, from its own model: the palm lands
    // on the top face (the aim error against the half widths) and the pads'
    // friction holds the planned lift's peak load (their scatter).
    let sigma_expect = inp.aim_sigma_m.hypot(inp.memory_sigma_m);
    let hit = |half: f32| if sigma_expect > 1e-6 { super::erf(half / (sigma_expect * std::f32::consts::SQRT_2)) } else { 1.0 };
    let need_mu = per_hand_kg * (G + MIN_JERK_ACC * rise_dz / (rise_s * rise_s)) / grip.max(1e-3);
    let p_success = if cap.contacts >= 2 { hit(half[0]) * hit(half[1]) * super::lognormal_above(mu.mean, mu.cov, need_mu) } else { 0.0 };
    metrics.push(("p_success", p_success));

    // 1. Reach down with the hand open, eyes on the object.
    let bottom = KinPose { closure: 0.0, ..plan.pose };
    let n = (plan.descend_s / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 0..=n {
        let u = min_jerk(i as f32 / n as f32);
        s.push(body, i as f32 * SAMPLE_S, "reach", reach::blend(body, &stand, &bottom, u), KinLoads::default(), None, Some(obj_ground));
    }
    let t_bottom = s.end();
    let how = match plan.approach {
        "side" => format!("오른팔 옆으로 {:.0}° 뻗음 (보지 않고) · ", plan.pose.shoulder_abd_deg[0]),
        "straddle" => format!("발 {:.0} cm 벌려 물건을 사이에 · ", plan.pose.stance_out_m * 100.0),
        _ => String::new(),
    };
    s.events.push(ActEvent::new(t_bottom, "bend_bottom", format!("{how}{} · 손바닥 {:.2} m", plan.strategy_ko, plan.empty.hands[0][2]), &[Part::Waist, Part::FootJoints, Part::Neck], Some(plan.loaded.margin), true));

    // 2. Close the fingers: on the object, or on air when the palm missed its top face.
    let w_close = inp.hand.fingers.first().map_or(std::f32::consts::TAU, |f| f.joints[0].max_speed);
    let t_close = CLOSE_SWING_RAD / w_close.max(0.1);
    let missed = aim[0].abs() > half[0] || aim[1].abs() > half[1] || cap.contacts < 2;
    let m_close = (t_close / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m_close {
        let u = i as f32 / m_close as f32;
        s.push(body, t_bottom + i as f32 * SAMPLE_S, "close", KinPose { closure: if missed { u } else { 0.7 * u }, ..bottom }, KinLoads::default(), None, Some(obj_ground));
    }
    let t_contact = s.end();
    metrics.push(("aim_err_m", (aim[0] * aim[0] + aim[1] * aim[1]).sqrt()));
    metrics.push(("mu", mu_act));
    metrics.push(("mu_est", mu.mean));
    metrics.push(("squeeze_cap_n", cap.squeeze_n));
    if missed {
        let why = if cap.contacts < 2 {
            format!("손가락이 {:.2} m 물체를 감싸지 못함 (접촉 {}개)", inp.size_m[0], cap.contacts)
        } else {
            format!("조준 오차 ({:+.3}, {:+.3}) m가 물체 윗면 반폭 {:.3} m를 넘어 빈손으로 쥠", aim[0], aim[1], half[0])
        };
        s.events.push(ActEvent::new(t_contact, "miss", why.clone(), &[Part::Hand, Part::HandJoints], Some((aim[0] * aim[0] + aim[1] * aim[1]).sqrt()), false));
        let n = (plan.descend_s / SAMPLE_S).ceil().max(1.0) as usize;
        for i in 1..=n {
            let u = min_jerk(i as f32 / n as f32);
            s.push(body, t_contact + i as f32 * SAMPLE_S, "rise", reach::blend(body, &bottom, &stand, u), KinLoads::default(), None, Some(obj_ground));
        }
        let dur = s.end();
        metrics.push(("duration_s", dur));
        return ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::fail("grasp_missed", why), duration_s: dur, metrics, notes };
    }
    // The fingertips meet the object at their closing speed.
    let v_tip = w_close * TIP_LEVER_M;
    let (f_imp, dt_imp) = material::impact(v_tip, material::FINGER_MASS_KG, material::pad_stiffness(5.0));
    metrics.push(("contact_impact_n", f_imp));
    s.events.push(ActEvent::new(t_contact, "contact", format!("손가락 {}개 접촉 · 충돌 {:.1} N ({:.1} ms)", cap.contacts, f_imp, dt_imp * 1000.0), &[Part::Hand, Part::HandJoints], Some(f_imp), true));

    // 3. Squeeze, then pull until the object leaves the ground.
    let torsion = |g: f32| cap.contacts as f32 * material::torsion_capacity_nm(mu_act, g / cap.contacts.max(1) as f32);
    let lever = (aim[0] * aim[0] + aim[1] * aim[1]).sqrt();
    let mk_grip = |g: f32, load: f32, closure: f32| Grip { grip_n: g, load_n: load, mu: mu_act, mu_est: mu.mean, slip_ratio: if load > 1e-6 { mu_act * g / load } else { 99.0 }, twist_margin_nm: torsion(g) - load * lever, contacts: cap.contacts, closure };
    let m_load = (LOOP_RISE_S / SAMPLE_S).ceil().max(1.0) as usize;
    let mut slip_min = f32::INFINITY;
    let mut twist_max = 0.0f32;
    let mut failed: Option<(&'static str, String, f32)> = None;
    for i in 1..=m_load {
        let u = i as f32 / m_load as f32;
        let g = grip * u;
        let t = t_contact + i as f32 * SAMPLE_S;
        s.push(body, t, "squeeze", KinPose { closure: 0.7, ..bottom }, KinLoads::default(), Some(mk_grip(g, 0.0, 0.7)), Some(obj_ground));
    }
    let t_squeezed = s.end();
    for i in 1..=m_load {
        let u = i as f32 / m_load as f32;
        let load = per_hand_kg * G * u;
        let t = t_squeezed + i as f32 * SAMPLE_S;
        let gr = mk_grip(grip, load, 0.7);
        slip_min = slip_min.min(gr.slip_ratio);
        if gr.slip_ratio < 1.0 && failed.is_none() {
            failed = Some(("grasp_slip", format!("들기 전에 미끄러짐: 마찰 {:.2} × 쥠 {:.0} N < 하중 {:.0} N", mu_act, grip, load), t));
        }
        s.push(body, t, "load", KinPose { closure: 0.7, ..bottom }, KinLoads { hand_force: hand_force(two_arms, [0.0, 0.0, -inp.mass_kg * G * u]), ..KinLoads::default() }, Some(gr), Some(obj_ground));
    }
    let t_lift = s.end();
    metrics.push(("lift_off_s", t_lift - t_contact));
    // The squeeze builds before the pull (the grip leads the load by the whole squeeze).
    metrics.push(("preload_s", t_squeezed - t_contact));
    metrics.push(("loading_s", t_lift - t_squeezed));
    metrics.push(("gf_lead_s", t_squeezed - t_contact));
    if failed.is_none() {
        s.events.push(ActEvent::new(t_lift, "lift_off", format!("들림 · 쥠 {:.0} N / 하중 {:.0} N", grip, per_hand_kg * G), &[Part::Hand, Part::Waist, Part::FootJoints], Some(grip / (per_hand_kg * G).max(1e-3)), true));
    }

    // 4. Rise with the object, then bring it up to the hold pose; the load is
    //    weight plus inertia. The controller keeps the load hanging under the
    //    shoulder while the body straightens, then curls it to the chest.
    let hold = hold_pose(two_arms, inp.cargo_kg);
    let path = LiftPath::new(body, &bottom, &hold, two_arms, rise_s);
    let lift_s = path.duration();
    metrics.push(("curl_s", path.curl_s));
    let at = |t: f32| path.at(body, t);
    let n = (lift_s / SAMPLE_S).ceil().max(2.0) as usize;
    let palm = |u: f32| kin::solve_fast(body, &at(u * lift_s), &KinLoads::default()).hands[0];
    let mut prev: Option<([f32; 3], [f32; 3])> = None;
    let mut dropped_at: Option<[f32; 3]> = None;
    // Three or more fingers close the object against the palm: the palm on its top face stops a turn about the grasp axis.
    let palm_blocks = cap.contacts >= 3;
    let mut pivoted = false;
    for i in 1..=n {
        if failed.is_some() {
            break;
        }
        let u = i as f32 / n as f32;
        let p = palm(u);
        let p_next = palm(((i + 1) as f32 / n as f32).min(1.0));
        let p_prev = prev.map(|x| x.0).unwrap_or(palm((i - 1) as f32 / n as f32));
        let dt = lift_s / n as f32;
        // Across, only one hand's swing in from the side moves the object (two hands mirror each other).
        let ay = if two_arms { 0.0 } else { (p_next[1] - 2.0 * p[1] + p_prev[1]) / (dt * dt) };
        let acc = [(p_next[0] - 2.0 * p[0] + p_prev[0]) / (dt * dt), ay, (p_next[2] - 2.0 * p[2] + p_prev[2]) / (dt * dt)];
        prev = Some((p, acc));
        let f_obj = [-inp.mass_kg * acc[0], -inp.mass_kg * acc[1], -inp.mass_kg * (G + acc[2])];
        let load = per_hand_kg * ((G + acc[2]).powi(2) + acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
        let gr = mk_grip(grip, load, 0.7 + 0.3 * u);
        slip_min = slip_min.min(gr.slip_ratio);
        let tw = if torsion(grip) > 1e-6 { load * lever / torsion(grip) } else { 99.0 };
        twist_max = twist_max.max(tw);
        let t = t_lift + i as f32 * dt;
        let pose = KinPose { closure: 0.7 + 0.3 * u, ..at(u * lift_s) };
        let ok = s.push(body, t, "lift", pose, KinLoads { hand_force: hand_force(two_arms, f_obj), ..KinLoads::default() }, Some(gr), Some([p[0], p[1], p[2] - 0.5 * inp.size_m[1] - reach::PALM_CLEARANCE_M]));
        if gr.slip_ratio < 1.0 {
            failed = Some(("grasp_slip", format!("들어 올리다 미끄러짐: 마찰 {:.2} × 쥠 {:.0} N < 하중 {:.0} N (가속 {:+.1} m/s²)", mu_act, grip, load, acc[2]), t));
            dropped_at = Some(p);
        } else if tw > 1.0 && !palm_blocks && !pivoted {
            // The pinch lets it swing until the centre of mass hangs under the grasp axis.
            pivoted = true;
            let tilt = lever.atan2(0.5 * inp.size_m[1]).to_degrees();
            s.events.push(ActEvent::new(t, "pivot", format!("물체가 손 안에서 {tilt:.0}° 돌아 매달림: 무게 모멘트 {:.2} N·m > 비틀림 마찰 {:.2} N·m", load * lever, torsion(grip)), &[Part::Hand, Part::HandJoints], Some(tilt), true));
            metrics.push(("pivot_deg", tilt));
        } else if !ok {
            failed = Some(("lift_overload", format!("들어 올리는 동안 {}", s.limiting.clone().unwrap_or_default()), t));
            dropped_at = Some(p);
        }
    }
    metrics.push(("slip_ratio_min", slip_min));
    metrics.push(("twist_ratio_max", twist_max));
    // How the grip follows the load over the squeeze, the pull and the lift (Pearson r).
    let pairs: Vec<(f32, f32)> = s.samples.iter().filter_map(|x| x.grip).map(|g| (g.grip_n, g.load_n)).collect();
    if pairs.len() > 3 {
        let n = pairs.len() as f32;
        let (mg, ml) = (pairs.iter().map(|p| p.0).sum::<f32>() / n, pairs.iter().map(|p| p.1).sum::<f32>() / n);
        let cov: f32 = pairs.iter().map(|p| (p.0 - mg) * (p.1 - ml)).sum();
        let (vg, vl): (f32, f32) = (pairs.iter().map(|p| (p.0 - mg).powi(2)).sum(), pairs.iter().map(|p| (p.1 - ml).powi(2)).sum());
        if vg > 1e-9 && vl > 1e-9 {
            metrics.push(("gf_lf_corr", cov / (vg * vl).sqrt()));
        }
    }
    metrics.push(("grip_n", grip));
    metrics.push(("gf_lf_hold", grip / (per_hand_kg * G).max(1e-3)));
    metrics.push(("safety_margin", (grip - per_hand_kg * G / mu_act) / grip.max(1e-3)));
    metrics.push(("reach_s", plan.descend_s));
    metrics.push(("rise_s", plan.rise_s));
    if let Some((code, why, t)) = failed {
        // The object falls back from where the hand was.
        let at = dropped_at.unwrap_or(plan.empty.hands[0]);
        let h = (at[2] - inp.size_m[1] - reach::PALM_CLEARANCE_M).max(0.0);
        let (f, dt) = material::impact((2.0 * G * h).sqrt(), inp.mass_kg, material::ground_stiffness(inp.ground));
        let e = material::restitution(inp.material, inp.ground);
        s.events.push(ActEvent::new(t, if code == "lift_overload" { "stall" } else { "slip" }, why.clone(), &[Part::Hand, Part::HandJoints], Some(slip_min), false));
        s.events.push(ActEvent::new(t + (2.0 * h / G).sqrt(), "drop", format!("{:.2} m에서 떨어짐 · 충격 {:.0} N ({:.1} ms) · 튀어 오름 {:.3} m", h, f, dt * 1000.0, e * e * h), &[Part::Hand], Some(f), false));
        metrics.push(("drop_height_m", h));
        metrics.push(("impact_n", f));
        let t0 = s.end();
        let back = s.samples.last().map(|x| x.pose).unwrap_or(stand);
        let n = (plan.rise_s / SAMPLE_S).ceil().max(1.0) as usize;
        for i in 1..=n {
            let u = min_jerk(i as f32 / n as f32);
            s.push(body, t0 + i as f32 * SAMPLE_S, "recover", KinPose { closure: 0.0, ..reach::blend(body, &back, &stand, u) }, KinLoads::default(), None, Some(obj_ground));
        }
        let dur = s.end();
        metrics.push(("duration_s", dur));
        metrics.push(("peak_ratio", s.peak_ratio));
        metrics.push(("min_margin", s.min_margin));
        return ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::fail(code, why), duration_s: dur, metrics, notes };
    }
    let t_hold = s.end();
    s.events.push(ActEvent::new(t_hold, "hold", format!("들어 올림 · 최소 미끄럼 여유 {:.2}", slip_min), &[Part::Hand, Part::Waist, Part::Neck], Some(slip_min), true));

    // 5. Stow it in the bay.
    let bay = bay_pose(two_arms, inp.cargo_kg);
    let r_hold = kin::solve_fast(body, &hold, &KinLoads::holding(inp.mass_kg, two_arms, 0.0));
    let r_bay = kin::solve_fast(body, &bay, &KinLoads::holding(inp.mass_kg, two_arms, 0.0));
    let t_stow = reach::move_time(&r_hold, &r_bay);
    let n = (t_stow / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        let pose = KinPose::lerp(&hold, &bay, u);
        let p = kin::solve_fast(body, &pose, &KinLoads::default()).hands[0];
        s.push(body, t_hold + i as f32 * SAMPLE_S, "stow", pose, KinLoads::holding(inp.mass_kg, two_arms, 0.0), Some(mk_grip(grip, per_hand_kg * G, 1.0)), Some([p[0], p[1], p[2] - 0.5 * inp.size_m[1]]));
    }
    let t_open = s.end();
    let (f_bay, _) = material::impact((2.0 * G * BAY_DROP_M).sqrt(), inp.mass_kg, material::ground_stiffness(Surface::Firm));
    let m_open = (t_close / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m_open {
        let u = i as f32 / m_open as f32;
        s.push(body, t_open + i as f32 * SAMPLE_S, "release", KinPose { closure: 1.0 - u, ..bay }, KinLoads::default(), None, Some([CARGO_X_M, 0.0, CARGO_Z_M]));
    }
    s.events.push(ActEvent::new(s.end(), "stowed", format!("적재함에 놓음 · {:.0} N", f_bay), &[Part::Hand], Some(f_bay), true));
    let dur = s.end();
    metrics.push(("duration_s", dur));
    metrics.push(("peak_ratio", s.peak_ratio));
    metrics.push(("min_margin", s.min_margin));
    ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::ok(), duration_s: dur, metrics, notes }
}

/// What is set down and where.
#[derive(Clone, Copy, Debug)]
pub struct ReleaseInput<'a> {
    pub body: &'a Body,
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub size_m: [f32; 3],
    /// The surface it lands on.
    pub ground: Surface,
    /// Height of what it lands on above the ground under the robot (a pile's top), metres.
    pub surface_z_m: f32,
    /// Cargo left in the bay after this object is taken out, kg.
    pub cargo_kg: f32,
    /// 1σ of where the hand lets go along each horizontal axis, metres (the aim error of a grasp).
    pub aim_sigma_m: f32,
    pub seed: u64,
}

/// Plays a release out: from the bay down to the ground or a pile, and open.
pub fn release(inp: &ReleaseInput) -> ActRollout {
    let body = inp.body;
    let mut d = Draw::new(inp.seed);
    let mut s = Series::new();
    let mut metrics: Vec<(&'static str, f32)> = Vec::new();
    let notes = Vec::new();
    let stand = KinPose::stand(inp.cargo_kg);
    let bay = bay_pose(false, inp.cargo_kg);
    // Take it out of the bay.
    let r_stand = kin::solve_fast(body, &stand, &KinLoads::default());
    let r_bay = kin::solve_fast(body, &bay, &KinLoads::default());
    let t_take = reach::move_time(&r_stand, &r_bay);
    let n = (t_take / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 0..=n {
        let u = min_jerk(i as f32 / n as f32);
        s.push(body, i as f32 * SAMPLE_S, "take", KinPose { closure: u, ..KinPose::lerp(&stand, &bay, u) }, KinLoads::default(), None, Some([CARGO_X_M, 0.0, CARGO_Z_M]));
    }
    // Down as far as the body allows: onto the surface, or the lowest it can.
    let want = inp.surface_z_m + inp.size_m[1] + reach::PALM_CLEARANCE_M;
    let mut plan = reach::plan(body, &ReachGoal { palm_z: want, half_len_m: 0.5 * inp.size_m[0], load_kg: inp.mass_kg, two_arms: false, cargo_kg: inp.cargo_kg });
    if !plan.ok {
        // Release from as low as it gets (a little above its lowest palm height).
        let low = plan.lowest_palm_z.max(want) + 0.02;
        plan = reach::plan(body, &ReachGoal { palm_z: low, half_len_m: 0.5 * inp.size_m[0], load_kg: inp.mass_kg, two_arms: false, cargo_kg: inp.cargo_kg });
    }
    let t0 = s.end();
    let bottom = if plan.ok { KinPose { closure: 1.0, ..plan.pose } } else { KinPose { closure: 1.0, ..grasp_hold_low(inp.cargo_kg) } };
    let held = KinLoads::holding(inp.mass_kg, false, 0.0);
    let r_bottom = kin::solve_fast(body, &bottom, &held);
    // The lift backwards: the arm brings the object from the bay over the
    // spot, then it goes straight down while the body bends.
    let path = LiftPath::new(body, &bottom, &bay, false, reach::move_time(&r_bay, &r_bottom));
    let t_down = path.duration();
    let n = (t_down / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let t = t_down * i as f32 / n as f32;
        let pose = path.at(body, t_down - t);
        let p = kin::solve_fast(body, &pose, &KinLoads::default()).hands[0];
        s.push(body, t0 + t, "lower", pose, held, None, Some([p[0], p[1], p[2] - 0.5 * inp.size_m[1] - reach::PALM_CLEARANCE_M]));
    }
    let t_open = s.end();
    let palm = r_bottom.hands[0];
    let h = (palm[2] - reach::PALM_CLEARANCE_M - inp.size_m[1] - inp.surface_z_m).max(0.0);
    let v = (2.0 * G * h).sqrt();
    let (f, dt) = material::impact(v, inp.mass_kg, material::ground_stiffness(inp.ground));
    let e = material::restitution(inp.material, inp.ground);
    // A cube lands on an edge and rebounds a little to the side (deflection 0–30°, est).
    let theta = d.uniform() * 30f32.to_radians();
    let dir = d.uniform() * std::f32::consts::TAU;
    let scatter = 2.0 * e * e * h * (2.0 * theta).sin();
    // The hand lets go where it believes the spot is, off by its aim error.
    let aim = [inp.aim_sigma_m * d.normal(), inp.aim_sigma_m * d.normal()];
    let land = [palm[0] + aim[0] + scatter * dir.cos(), palm[1] + aim[1] + scatter * dir.sin()];
    let off = (land[0] - palm[0]).hypot(land[1] - palm[1]);
    let m_open = 12usize;
    for i in 1..=m_open {
        let u = i as f32 / m_open as f32;
        s.push(body, t_open + i as f32 * SAMPLE_S, "open", KinPose { closure: 1.0 - u, ..bottom }, KinLoads::default(), None, Some([land[0], land[1], inp.surface_z_m + 0.5 * inp.size_m[1]]));
    }
    s.events.push(ActEvent::new(t_open, "release", format!("손을 폄 · 낙하 {:.2} m", h), &[Part::Hand, Part::HandJoints, Part::Waist], Some(h), true));
    s.events.push(ActEvent::new(t_open + (2.0 * h / G).sqrt(), "impact", format!("착지 충격 {:.0} N ({:.1} ms) · 튀어 오름 {:.3} m · {:.3} m 옆으로", f, dt * 1000.0, e * e * h, scatter), &[Part::Hand], Some(f), true));
    let t_up = s.end();
    let n = (plan.rise_s.max(reach::BEND_MIN_S) / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        s.push(body, t_up + i as f32 * SAMPLE_S, "rise", KinPose { closure: 0.0, ..reach::blend(body, &bottom, &stand, u) }, KinLoads::default(), None, Some([land[0], land[1], inp.surface_z_m + 0.5 * inp.size_m[1]]));
    }
    let dur = s.end();
    metrics.push(("drop_height_m", h));
    metrics.push(("impact_n", f));
    metrics.push(("impact_ms", dt * 1000.0));
    metrics.push(("bounce_m", e * e * h));
    metrics.push(("scatter_m", scatter));
    metrics.push(("place_err_m", off));
    metrics.push(("land_x", land[0]));
    metrics.push(("land_y", land[1]));
    metrics.push(("bent", if plan.ok { 1.0 } else { 0.0 }));
    metrics.push(("duration_s", dur));
    metrics.push(("peak_ratio", s.peak_ratio));
    metrics.push(("min_margin", s.min_margin));
    ActRollout { kind: ActKind::Release, samples: s.samples, events: s.events, outcome: Outcome::ok(), duration_s: dur, metrics, notes }
}

/// The lowest a body without any bend holds an object: the arm straight down in front.
fn grasp_hold_low(cargo_kg: f32) -> KinPose {
    KinPose { shoulder_deg: [15.0, 0.0], elbow_deg: [5.0, 10.0], ..KinPose::stand(cargo_kg) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::{FingerCount, NotoConfig};

    fn setup(base: BaseKind, fingers: FingerCount) -> (Body, Hand) {
        let mut cfg = NotoConfig::preset(fingers);
        cfg.base = base;
        (Body::new(&BodyPlan::default(), base, fingers), cfg.build().unwrap())
    }

    fn input<'a>(b: &'a Body, h: &'a Hand, seed: u64, sigma: f32) -> GraspInput<'a> {
        GraspInput::front(b, h, ItemMaterial::Wood, 2.4, [0.15, 0.15, 0.15], Surface::Gravel, false, 0.0, sigma, seed)
    }

    #[test]
    fn a_legged_robot_picks_a_log_with_a_full_time_series() {
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let r = grasp(&input(&b, &h, 3, 0.0));
        assert!(r.outcome.ok, "{:?} {:?}", r.outcome, r.notes);
        for e in ["bend_bottom", "contact", "lift_off", "hold", "stowed"] {
            assert!(r.events.iter().any(|x| x.kind == e), "missing {e}: {:?}", r.events.iter().map(|x| x.kind).collect::<Vec<_>>());
        }
        assert!(r.duration_s > 2.5 && r.duration_s < 8.0, "{}", r.duration_s);
        assert!(r.samples.windows(2).all(|w| w[1].t > w[0].t));
        let gf = r.metric("gf_lf_hold").unwrap();
        assert!(gf > 1.0 / r.metric("mu").unwrap(), "squeeze above the slip limit: {gf}");
        assert!(r.metric("slip_ratio_min").unwrap() > 1.0);
    }

    #[test]
    fn a_wheeled_rigid_trunk_cannot_reach_and_says_why() {
        let (b, h) = setup(BaseKind::Wheel, FingerCount::Five);
        let r = grasp(&input(&b, &h, 3, 0.0));
        assert!(!r.outcome.ok && r.outcome.code == Some("reach_ground"), "{:?}", r.outcome);
    }

    #[test]
    fn aim_error_misses_and_the_draw_repeats() {
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let a = grasp(&input(&b, &h, 17, 0.06));
        let a2 = grasp(&input(&b, &h, 17, 0.06));
        assert_eq!(a.outcome, a2.outcome, "the same seed replays");
        // Over many draws a large aim error misses more often than a small one.
        let misses = |sigma: f32| (0..60).filter(|s| grasp(&input(&b, &h, *s, sigma)).outcome.code == Some("grasp_missed")).count();
        assert_eq!(misses(0.0), 0);
        assert!(misses(0.08) > misses(0.01) + 5, "large aim errors miss");
    }

    #[test]
    fn a_release_from_hand_height_bounces_more_than_one_set_down() {
        let (legs, _) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let (wheels, _) = setup(BaseKind::Wheel, FingerCount::Five);
        let inp = |b: &Body| release(&ReleaseInput { body: b, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], ground: Surface::Firm, surface_z_m: 0.0, cargo_kg: 0.0, aim_sigma_m: 0.005, seed: 5 });
        let a = inp(&legs);
        let b = inp(&wheels);
        assert!(a.metric("drop_height_m").unwrap() < 0.05, "{:?}", a.metric("drop_height_m"));
        assert!(b.metric("drop_height_m").unwrap() > 0.3, "{:?}", b.metric("drop_height_m"));
        assert!(b.metric("impact_n").unwrap() > a.metric("impact_n").unwrap());
    }

    #[test]
    #[ignore]
    fn print_grasps() {
        for (base, fingers) in [(BaseKind::HumanFoot, FingerCount::Five), (BaseKind::HumanFoot, FingerCount::Two), (BaseKind::HumanFoot, FingerCount::Three)] {
            let (b, h) = setup(base, fingers);
            for (mat, kg, size) in [(ItemMaterial::Wood, 2.4f32, 0.15f32), (ItemMaterial::Stone, 2.6, 0.10), (ItemMaterial::Coal, 1.4, 0.10), (ItemMaterial::IronOre, 4.0, 0.10)] {
                for sigma in [0.0f32, 0.01, 0.02] {
                    let mut ok = 0;
                    let mut codes = std::collections::BTreeMap::new();
                    for seed in 0..100u64 {
                        let r = grasp(&GraspInput::front(&b, &h, mat, kg, [size; 3], Surface::Gravel, false, 0.0, sigma, seed));
                        if r.outcome.ok { ok += 1 } else { *codes.entry(r.outcome.code.unwrap_or("?")).or_insert(0) += 1 }
                        if seed == 0 {
                            println!("[GRASP1] {fingers:?} {mat:?} σ{sigma}: {:?} dur {:.2} gf/lf {:?} slip {:?} twist {:?} grip {:?} cap {:?} mu {:?} notes {:?}", r.outcome.code, r.duration_s, r.metric("gf_lf_hold"), r.metric("slip_ratio_min"), r.metric("twist_ratio_max"), r.metric("grip_n"), r.metric("squeeze_cap_n"), r.metric("mu"), r.notes);
                        }
                    }
                    println!("[GRASP] {fingers:?} {mat:?} {kg}kg σ{sigma}: ok {ok}/100 {codes:?}");
                }
            }
        }
    }
}
