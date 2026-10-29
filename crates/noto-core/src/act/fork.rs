//! Forklift handling for wheeled and tracked bases. A rider sits on the base and looks ahead; the work at the floor is
//! done by a pair of tines on a mast at the front, as on a counterbalanced lift truck: the tines go down to the floor,
//! a camera on the carriage between them checks the item, the base drives forward until the item sits on the tines,
//! and the mast lifts it to travel height. Setting down runs the other way: lower, back out, raise the empty tines.
//!
//! Two eyes check the item, as an operator and a fork camera do on a real truck. Before the tines go down the rider
//! glances down over the dashboard at it (the head camera), then looks ahead again while the carriage camera takes
//! over and follows it as the tines slide in. Close up the item fills the carriage camera's view (`BLOCKED_SHARE`),
//! so the base stops and the rider looks down once more to see the load ride onto the tines before the mast lifts it.
//! Each look holds still for one fixation, so the world can check what each camera sees against the other.
//!
//! The load may not tip the base over its front axle. A counterbalanced truck stays upright while the moment of the
//! load about the front axle is less than the moment of the truck's own weight about it (the longitudinal stability
//! of the "stability triangle"; ISO 22915-2 tests it on a tilted platform); here the base's weight must hold the load
//! with `TIP_MIN` to spare.

use crate::balance::{Body, G};
use crate::dynamics::LinkBox;
use crate::kin::{self, KinLoads, KinPose, BUMPER_GAP_M, WHEEL_R_M};
use crate::parts::Segment;

use super::{gaze, min_jerk, ActEvent, ActKind, ActRollout, ActSample, Outcome, Part, SAMPLE_S};

/// Tine length (m) [추정] (a small robot's tines: twice the 0.10 m items they carry).
pub const TINE_LEN_M: f32 = 0.20;
/// Tine centre-to-centre spread (m) [추정]: the smallest item (0.10 m) rests on both tines.
pub const TINE_SPREAD_M: f32 = 0.07;
pub const TINE_W_M: f32 = 0.025;
pub const TINE_T_M: f32 = 0.012;
pub const MAST_TOP_M: f32 = 0.95;
/// The highest the tines go (m): the mast top less the carriage.
pub const FORK_MAX_M: f32 = MAST_TOP_M - 0.10;
/// Tine heights (m above the ground under the base): down on the floor, empty travel, loaded travel [추정].
pub const FORK_FLOOR_M: f32 = 0.0;
pub const FORK_REST_M: f32 = 0.05;
pub const FORK_CARRY_M: f32 = 0.10;
/// Mast speed (m/s), the same 0.10 m/s as the torso and mast lifts (`parts::TORSO_LIFT`) [추정].
pub const MAST_MPS: f32 = 0.10;
/// Mast push (N), as `parts::TORSO_LIFT` (450 N) [추정].
pub const MAST_FORCE_N: f32 = 450.0;
/// Creep speed while the tines slide under or out of the item (m/s) [추정].
pub const INSERT_MPS: f32 = 0.12;
/// How long the carriage camera holds still to see (s): one fixation, as `gaze::FIXATE_S`.
pub const CHECK_S: f32 = super::gaze::FIXATE_S;
/// Restoring moment over overturning moment about the front axle that a load may not go under [추정].
pub const TIP_MIN: f32 = 1.5;
/// The carriage camera: above the tines, looking ahead and down (deg below level) [추정].
pub const CAMERA_UP_M: f32 = 0.06;
pub const CAMERA_PITCH_DEG: f32 = 25.0;
/// The carriage camera's lens: the head camera's 70° across a 4:3 frame (the HUD's small fork view) [추정].
pub const CAMERA_HFOV_DEG: f32 = 70.0;
pub const CAMERA_ASPECT: f32 = 0.75;
/// The share of the carriage camera's image the item may fill while the tines slide in. Past it the camera sees only
/// the item's near face, not where the tines go, so the base stops and the rider looks down at the load [추정].
pub const BLOCKED_SHARE: f32 = 0.6;
/// Clearance between the tine tips and the item before the tines go down (m) [추정].
pub const STANDOFF_GAP_M: f32 = 0.04;
/// Turns shorter than this are not worth a separate alignment (s).
pub const ALIGN_MIN_S: f32 = 0.05;

/// Whether this body handles items with forks: every wheeled, mecanum and tracked base does; legs pick by hand.
pub fn applies(body: &Body) -> bool {
    !body.legged()
}

/// The bumper face (m forward of the base centre), where the mast stands.
pub fn front_x(body: &Body) -> f32 {
    0.5 * body.support_len_m + WHEEL_R_M + BUMPER_GAP_M
}

/// Where the tines start (m forward of the base centre).
pub fn heel_x(body: &Body) -> f32 {
    front_x(body) + 0.035
}

/// Where an item `len_m` long sits on the tines, pushed back to the carriage (m forward of the base centre).
pub fn load_x(body: &Body, len_m: f32) -> f32 {
    heel_x(body) + 0.5 * len_m
}

/// How far ahead an item `len_m` long must lie (its centre) for the tines to go down in front of it, not on it.
pub fn standoff_x(body: &Body, len_m: f32) -> f32 {
    heel_x(body) + TINE_LEN_M + 0.5 * len_m + STANDOFF_GAP_M
}

/// The carriage camera's place on the body for tines at `fork_m` (x forward, y left, z up).
pub fn camera_pos(body: &Body, fork_m: f32) -> [f32; 3] {
    [heel_x(body) - 0.01, 0.0, fork_m + TINE_T_M + CAMERA_UP_M]
}

/// How much of the carriage camera's image an item fills: the share of a ray grid over the camera's view that meets
/// the item's box. `object` is its centre in the body frame and `size_m` its length along the tines, height and width.
pub fn camera_blocked_share(body: &Body, fork_m: f32, object: [f32; 3], size_m: [f32; 3]) -> f32 {
    let cam = camera_pos(body, fork_m);
    let (sp, cp) = (-CAMERA_PITCH_DEG.to_radians()).sin_cos();
    let half_w = (0.5 * CAMERA_HFOV_DEG.to_radians()).tan();
    let half_h = half_w * CAMERA_ASPECT;
    let lo = [object[0] - 0.5 * size_m[0], object[1] - 0.5 * size_m[2], object[2] - 0.5 * size_m[1]];
    let hi = [object[0] + 0.5 * size_m[0], object[1] + 0.5 * size_m[2], object[2] + 0.5 * size_m[1]];
    let (nx, ny) = (16usize, 12usize);
    let mut hit = 0usize;
    for j in 0..ny {
        for i in 0..nx {
            let right = ((i as f32 + 0.5) / nx as f32 * 2.0 - 1.0) * half_w;
            let up = (1.0 - (j as f32 + 0.5) / ny as f32 * 2.0) * half_h;
            // Camera frame (forward, left, up), pitched down about the left axis.
            let d = [cp - up * sp, -right, sp + up * cp];
            let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
            let inside = (0..3).all(|k| {
                if d[k].abs() < 1e-9 {
                    return cam[k] >= lo[k] && cam[k] <= hi[k];
                }
                let (a, b) = ((lo[k] - cam[k]) / d[k], (hi[k] - cam[k]) / d[k]);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
                t0 <= t1
            });
            if inside {
                hit += 1;
            }
        }
    }
    hit as f32 / (nx * ny) as f32
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct ForkFit {
    /// Restoring over overturning moment about the front axle with this load on the tines.
    pub tip_ratio: f32,
    /// The heaviest load the tines take here (kg): the mast's push or the tip limit, whichever is less.
    pub capacity_kg: f32,
    pub load_x_m: f32,
}

impl ForkFit {
    pub fn margin(&self) -> f32 {
        ((self.tip_ratio - 1.0) / self.tip_ratio.max(1e-3)).clamp(0.0, 1.0)
    }
}

/// Whether the tines can take an item of `mass_kg` and `size_m` (x, height, z) lying `base_m` above the ground under
/// the base, with `load_kg` already on them.
pub fn check(body: &Body, mass_kg: f32, size_m: [f32; 3], base_m: f32, load_kg: f32) -> Result<ForkFit, String> {
    let len = size_m[0].max(size_m[2]);
    let narrow = size_m[0].min(size_m[2]);
    if narrow < TINE_SPREAD_M + TINE_W_M - 1e-4 {
        return Err(format!("물건 폭 {:.0} cm가 포크 간격 {:.0} cm보다 좁아 포크 사이로 빠짐", narrow * 100.0, (TINE_SPREAD_M + TINE_W_M) * 100.0));
    }
    if len > 2.0 * TINE_LEN_M {
        return Err(format!("물건 길이 {:.0} cm가 포크 길이 {:.0} cm의 2배를 넘음", len * 100.0, TINE_LEN_M * 100.0));
    }
    if base_m > FORK_MAX_M {
        return Err(format!("물건이 {:.2} m 높이에 있어 마스트 최고 {:.2} m보다 높음", base_m, FORK_MAX_M));
    }
    let axle = 0.5 * body.support_len_m;
    let lx = load_x(body, len.min(2.0 * TINE_LEN_M));
    let com_x = kin::solve_fast(body, &KinPose::stand(0.0), &KinLoads::default()).com[0];
    let restoring = body.mass_kg() * (axle - com_x).max(0.01);
    let total = mass_kg + load_kg;
    let tip_ratio = restoring / (total * (lx - axle).max(0.01)).max(1e-6);
    let tip_kg = restoring / (TIP_MIN * (lx - axle).max(0.01));
    let capacity_kg = tip_kg.min(MAST_FORCE_N / G);
    if total > capacity_kg {
        return Err(format!(
            "포크 하중 {total:.1} kg > 한계 {capacity_kg:.1} kg (앞바퀴 축 기준 전도 여유 {tip_ratio:.2} < {TIP_MIN:.1}, 마스트 {:.0} kg)",
            MAST_FORCE_N / G
        ));
    }
    Ok(ForkFit { tip_ratio, capacity_kg, load_x_m: lx })
}

fn sample(body: &Body, t: f32, phase: &'static str, fork_m: f32, cargo_kg: f32, object: [f32; 3]) -> ActSample {
    posed(body, t, phase, &KinPose::stand(cargo_kg), fork_m, object)
}

/// A sample with the rider's head (and lean) as in `head`, the tines at `fork_m`.
fn posed(body: &Body, t: f32, phase: &'static str, head: &KinPose, fork_m: f32, object: [f32; 3]) -> ActSample {
    let pose = KinPose { fork_m, ..*head };
    let r = kin::solve_fast(body, &pose, &KinLoads::default());
    ActSample { t, phase, pose, loads: KinLoads::default(), margin: r.margin, zmp: r.zmp, com: r.com, peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: Some(object) }
}

/// Moves the tines from `from_m` to `to_m` on the mast, `phase` by phase, appending samples from `t0`. Returns the end.
#[allow(clippy::too_many_arguments)]
fn travel(body: &Body, samples: &mut Vec<ActSample>, t0: f32, phase: &'static str, from_m: f32, to_m: f32, cargo_kg: f32, object: impl Fn(f32) -> [f32; 3]) -> f32 {
    let ahead = KinPose::stand(cargo_kg);
    travel_with(body, samples, t0, phase, from_m, to_m, |_| ahead, object)
}

/// `travel` with the rider's head as `head` gives it at each time (s from `t0`).
#[allow(clippy::too_many_arguments)]
fn travel_with(body: &Body, samples: &mut Vec<ActSample>, t0: f32, phase: &'static str, from_m: f32, to_m: f32, head: impl Fn(f32) -> KinPose, object: impl Fn(f32) -> [f32; 3]) -> f32 {
    let dt = ((to_m - from_m).abs() / MAST_MPS).max(SAMPLE_S);
    let n = (dt / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        let h = from_m + (to_m - from_m) * u;
        let t = dt * i as f32 / n as f32;
        samples.push(posed(body, t0 + t, phase, &head(t), h, object(h)));
    }
    t0 + dt
}

/// The rider's head moving from `from` to `to` over a head shift that starts at `t0` (s): ahead of it `from`, after it `to`.
fn head_shift(body: &Body, from: KinPose, to: KinPose, t0: f32) -> impl Fn(f32) -> KinPose {
    let dur = gaze::shift_s(body, &from, &to);
    move |t: f32| KinPose::lerp(&from, &to, min_jerk((t - t0) / dur.max(1e-3)))
}

/// The rider looks down at the item (`at`, its centre in the body frame) from the seat and holds still to see it, as an
/// operator glances over the dashboard at the tines: the neck first, the waist for the rest (`gaze::look_pose`). The
/// tines stay at `fork_m`; `phases` names the head shift and the hold. Returns the pose that looks and the end time.
#[allow(clippy::too_many_arguments)]
fn rider_look(body: &Body, from: &KinPose, at: [f32; 3], fork_m: f32, t0: f32, phases: (&'static str, &'static str), why: &str, samples: &mut Vec<ActSample>, events: &mut Vec<ActEvent>) -> (KinPose, f32) {
    let to = KinPose { fork_m, ..gaze::look_pose(body, from, at[0], at[2]) };
    let t_look = gaze::shift_s(body, from, &to);
    let n = (t_look / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        samples.push(posed(body, t0 + t_look * i as f32 / n as f32, phases.0, &KinPose::lerp(from, &to, u), fork_m, at));
    }
    let t_fix = t0 + t_look;
    let how = if (to.trunk_deg() - from.trunk_deg()).abs() > 0.5 { format!("목 {:.0}° · 허리 {:.0}° 숙여", to.neck_deg, to.trunk_deg()) } else { format!("목 {:.0}° 숙여", to.neck_deg) };
    events.push(ActEvent::new(t_fix, "rider_fix", format!("탑승자 시선: {how} {why}"), &[Part::Neck], Some(to.neck_deg), true));
    let m = (CHECK_S / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m {
        samples.push(posed(body, t_fix + CHECK_S * i as f32 / m as f32, phases.1, &to, fork_m, at));
    }
    let end = t_fix + CHECK_S;
    events.push(ActEvent::new(end, "rider_check", format!("탑승자 머리 카메라로 {CHECK_S:.2} s 확인"), &[Part::Neck], Some(CHECK_S), true));
    (to, end)
}

/// Where along the slide the carriage camera is blocked: the first item centre (m forward of the base centre), going
/// from `from_x` down to `to_x`, at which the item fills `BLOCKED_SHARE` of its image, and that share.
fn blocked_at(body: &Body, fork_m: f32, from_x: f32, to_x: f32, z: f32, size_m: [f32; 3]) -> (f32, f32) {
    let len = size_m[0].max(size_m[2]);
    let box_m = [len, size_m[1], size_m[0].min(size_m[2])];
    let steps = ((from_x - to_x) / 0.005).ceil().max(1.0) as usize;
    (0..=steps)
        .map(|i| from_x + (to_x - from_x) * i as f32 / steps as f32)
        .map(|x| (x, camera_blocked_share(body, fork_m, [x, 0.0, z], box_m)))
        .find(|(_, s)| *s >= BLOCKED_SHARE)
        .unwrap_or((to_x, camera_blocked_share(body, fork_m, [to_x, 0.0, z], box_m)))
}

/// Turns the base to face the item for `turn_s` and backs it off until the item is `to_x` ahead ("align"), tines held
/// at `fork_m`. Returns the end time; nothing is added when there is nothing to do.
#[allow(clippy::too_many_arguments)]
fn align(body: &Body, samples: &mut Vec<ActSample>, turn_s: f32, fork_m: f32, cargo_kg: f32, from_x: f32, to_x: f32, z: f32) -> f32 {
    let back = (to_x - from_x).max(0.0);
    let dur = turn_s.max(back / INSERT_MPS);
    if dur < ALIGN_MIN_S {
        return 0.0;
    }
    let n = (dur / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        samples.push(sample(body, dur * i as f32 / n as f32, "align", fork_m, cargo_kg, [from_x + back * u, 0.0, z]));
    }
    dur
}

/// The look before the tines: the base turns to the item for `turn_s` (and backs off if the tines would come down on
/// it), the rider looks down at it over the dashboard and holds still ("rider_check": the head camera's look), then
/// looks ahead again while the tines go down to the item's base (`base_m`), and the tines hold while the carriage
/// camera sees the item ("check"). `size_m` is the item's length, height and width.
#[allow(clippy::too_many_arguments)]
pub fn look(body: &Body, from_m: f32, dist_m: f32, size_m: [f32; 3], base_m: f32, cargo_kg: f32, turn_s: f32) -> ActRollout {
    let floor = base_m.max(FORK_FLOOR_M);
    let x = dist_m.max(standoff_x(body, size_m[0].max(size_m[2])));
    let obj = [x, 0.0, base_m + 0.5 * size_m[1]];
    let ahead = KinPose::stand(cargo_kg);
    let mut samples = vec![sample(body, 0.0, "align", from_m, cargo_kg, [dist_m, 0.0, obj[2]])];
    let mut events = Vec::new();
    let t0 = align(body, &mut samples, turn_s, from_m, cargo_kg, dist_m, x, obj[2]);
    let rider = gaze::can_aim(body);
    if t0 == 0.0 {
        samples[0].phase = if rider { "rider_look" } else { "lower" };
    }
    let (eyes, t) = if rider { rider_look(body, &ahead, obj, from_m, t0, ("rider_look", "rider_check"), "물건을 화면 가운데에", &mut samples, &mut events) } else { (ahead, t0) };
    let head = head_shift(body, eyes, ahead, t);
    let t1 = travel_with(body, &mut samples, t, "lower", from_m, floor, |s| head(t + s), |_| obj);
    events.push(ActEvent::new(t1, "fork_down", format!("포크를 {floor:.2} m로 내림: 탑승자는 다시 정면, 포크 카메라가 물건을 봄"), &[Part::Wheel, Part::Track], Some(floor), true));
    let m = (CHECK_S / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m {
        let at = t1 + CHECK_S * i as f32 / m as f32;
        samples.push(posed(body, at, "check", &head(at), floor, obj));
    }
    let end = t1 + CHECK_S;
    events.push(ActEvent::new(end, "vision_check", format!("포크 카메라로 {CHECK_S:.2} s 확인"), &[Part::Wheel, Part::Track], Some(CHECK_S), true));
    let metrics = vec![("fork_m", floor), ("rider_neck_deg", eyes.neck_deg), ("rider_check_s", t), ("look_s", t1), ("fixate_s", CHECK_S), ("duration_s", end)];
    ActRollout { kind: ActKind::Fork, samples, events, outcome: Outcome::ok(), duration_s: end, metrics, notes: vec![] }
}

/// Picks an item `dist_m` ahead (its centre) up on the tines: turn to it for `turn_s`, down (if not already), slide in
/// (back off first when it is closer than the tines reach) with the carriage camera on it until the item blocks the
/// camera, stop while the rider looks down at it ("recheck"), slide the rest of the way with the rider watching it
/// ride on ("seat"), and lift to travel height while the rider looks ahead again. `object` in each sample is the
/// item's centre in the body frame; the world drives the base so that it is.
#[allow(clippy::too_many_arguments)]
pub fn pick(body: &Body, from_m: f32, dist_m: f32, mass_kg: f32, size_m: [f32; 3], base_m: f32, load_kg: f32, turn_s: f32) -> ActRollout {
    let fit = match check(body, mass_kg, size_m, base_m, load_kg) {
        Ok(f) => f,
        Err(why) => {
            let s = vec![sample(body, 0.0, "lower", from_m, load_kg, [dist_m, 0.0, base_m + 0.5 * size_m[1]])];
            return ActRollout { kind: ActKind::Fork, samples: s, events: vec![], outcome: Outcome::fail("fork_overload", why), duration_s: 0.0, metrics: vec![], notes: vec![] };
        }
    };
    let floor = base_m.max(FORK_FLOOR_M);
    let half_h = 0.5 * size_m[1];
    let at_rest = |x: f32| [x, 0.0, base_m + half_h];
    let ahead = KinPose::stand(load_kg);
    // Tines still up: back off first so they come down in front of the item, not on it.
    let x0 = if (from_m - floor).abs() > 1e-3 { dist_m.max(standoff_x(body, size_m[0].max(size_m[2]))) } else { dist_m };
    let mut samples = vec![sample(body, 0.0, "align", from_m, load_kg, at_rest(dist_m))];
    let mut events = Vec::new();
    let mut t = align(body, &mut samples, turn_s, from_m, load_kg, dist_m, x0, base_m + half_h);
    if t == 0.0 {
        samples[0].phase = if (from_m - floor).abs() > 1e-3 { "lower" } else { "insert" };
    }
    let dist_m = x0;
    if (from_m - floor).abs() > 1e-3 {
        t = travel(body, &mut samples, t, "lower", from_m, floor, load_kg, |_| at_rest(dist_m));
        events.push(ActEvent::new(t, "fork_down", format!("포크를 {floor:.2} m로 내림"), &[Part::Wheel, Part::Track], Some(floor), true));
    }
    let slide = dist_m - fit.load_x_m;
    let rider = gaze::can_aim(body);
    // In with the carriage camera on the item, as far as it still sees past it.
    let (stop_x, share) = if slide > 0.0 && rider { blocked_at(body, floor, dist_m, fit.load_x_m, base_m + half_h, size_m) } else { (fit.load_x_m, 0.0) };
    let creep = |samples: &mut Vec<ActSample>, t0: f32, phase: &'static str, from_x: f32, to_x: f32, head: &dyn Fn(f32, f32) -> KinPose| {
        let dt = ((from_x - to_x).abs() / INSERT_MPS).max(SAMPLE_S);
        let n = (dt / SAMPLE_S).ceil().max(1.0) as usize;
        for i in 1..=n {
            let x = from_x + (to_x - from_x) * min_jerk(i as f32 / n as f32);
            let at = t0 + dt * i as f32 / n as f32;
            samples.push(posed(body, at, phase, &head(at, x), floor, at_rest(x)));
        }
        t0 + dt
    };
    let ahead_now = |_: f32, _: f32| ahead;
    t = creep(&mut samples, t, "insert", dist_m, stop_x, &ahead_now);
    let mut eyes = ahead;
    if rider && slide > 0.0 {
        events.push(ActEvent::new(t, "fork_cam_blocked", format!("포크 카메라 화면 {:.0}%가 물건에 가려짐: 멈추고 탑승자가 고개를 숙여 다시 확인", share * 100.0), &[Part::Wheel, Part::Track], Some(share), true));
        let (_, t_end) = rider_look(body, &ahead, at_rest(stop_x), floor, t, ("recheck_look", "recheck"), "포크 위로 오는 물건을 봄", &mut samples, &mut events);
        // The rest of the way with the eyes on the item as it rides onto the tines.
        let follow = |_: f32, x: f32| KinPose { fork_m: floor, ..gaze::look_pose(body, &ahead, x, base_m + half_h) };
        t = creep(&mut samples, t_end, "seat", stop_x, fit.load_x_m, &follow);
        eyes = samples.last().map_or(eyes, |s| s.pose);
    } else if (stop_x - fit.load_x_m).abs() > 1e-4 {
        t = creep(&mut samples, t, "seat", stop_x, fit.load_x_m, &ahead_now);
    }
    events.push(ActEvent::new(t, "fork_in", if slide >= 0.0 { format!("{:.0} cm 전진해 포크를 물건 밑으로 넣음", slide * 100.0) } else { format!("{:.0} cm 물러나 포크 끝을 물건에 맞춤", -slide * 100.0) }, &[Part::Wheel, Part::Track], Some(slide), true));
    let carry = (floor + FORK_CARRY_M).min(FORK_MAX_M).max(FORK_CARRY_M);
    let on = |h: f32| [fit.load_x_m, 0.0, h + TINE_T_M + half_h];
    let head = head_shift(body, eyes, KinPose::stand(load_kg + mass_kg), t);
    let t_raise = t;
    t = travel_with(body, &mut samples, t, "raise", floor, carry, |s| head(t_raise + s), on);
    events.push(ActEvent::new(t, "fork_up", format!("마스트로 {:.1} kg를 {carry:.2} m까지 들어올림 (전도 여유 {:.2})", mass_kg, fit.tip_ratio), &[Part::Wheel, Part::Track], Some(fit.tip_ratio), true));
    let mut metrics = vec![("fork_m", carry), ("slide_m", slide), ("tip_ratio", fit.tip_ratio), ("capacity_kg", fit.capacity_kg), ("duration_s", t)];
    if rider && slide > 0.0 {
        metrics.extend([("blocked_x_m", stop_x), ("blocked_share", share)]);
    }
    ActRollout { kind: ActKind::Fork, samples, events, outcome: Outcome::ok(), duration_s: t, metrics, notes: vec![] }
}

/// Sets the load down at `surface_m` (the pile top): lower, back out from under it, raise the empty tines to rest.
pub fn set_down(body: &Body, from_m: f32, mass_kg: f32, size_m: [f32; 3], surface_m: f32, load_kg: f32) -> ActRollout {
    let len = size_m[0].max(size_m[2]).min(2.0 * TINE_LEN_M);
    let lx = load_x(body, len);
    let half_h = 0.5 * size_m[1];
    let floor = surface_m.max(FORK_FLOOR_M).min(FORK_MAX_M);
    let mut samples = vec![sample(body, 0.0, "lower_load", from_m, load_kg, [lx, 0.0, from_m + TINE_T_M + half_h])];
    let mut t = travel(body, &mut samples, 0.0, "lower_load", from_m, floor, load_kg, |h| [lx, 0.0, h + TINE_T_M + half_h]);
    let mut events = vec![ActEvent::new(t, "fork_down", format!("짐을 {floor:.2} m에 내려놓음"), &[Part::Wheel, Part::Track], Some(floor), true)];
    let back = heel_x(body) + TINE_LEN_M + 0.02 - (lx - 0.5 * len);
    let dt = (back / INSERT_MPS).max(SAMPLE_S);
    let n = (dt / SAMPLE_S).ceil().max(1.0) as usize;
    let left = (load_kg - mass_kg).max(0.0);
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        samples.push(sample(body, t + dt * i as f32 / n as f32, "back_out", floor, left, [lx + back * u, 0.0, floor + half_h]));
    }
    t += dt;
    events.push(ActEvent::new(t, "fork_out", format!("{:.0} cm 후진해 포크를 뺌", back * 100.0), &[Part::Wheel, Part::Track], Some(back), true));
    let end_obj = [lx + back, 0.0, floor + half_h];
    t = travel(body, &mut samples, t, "rest", floor, FORK_REST_M, left, |_| end_obj);
    let metrics = vec![("fork_m", FORK_REST_M), ("back_m", back), ("duration_s", t)];
    ActRollout { kind: ActKind::Fork, samples, events, outcome: Outcome::ok(), duration_s: t, metrics, notes: vec![] }
}

/// What a wheeled or tracked base looks like as a lift truck, over its own chassis and wheels or tracks: a truck body
/// with a counterweight at the back, a raised seat with a seated rider whose eyes are at `eye_m` (the head camera),
/// a dashboard and steering wheel, and an overhead guard on four posts. Body frame; the names say what each box is.
pub fn truck_boxes(body: &Body, eye_m: f32) -> Vec<LinkBox> {
    truck_boxes_posed(body, eye_m, &KinPose::stand(0.0))
}

/// `truck_boxes` with the rider leaning by `pose`'s trunk angle and bending the head by its neck, as the head camera
/// does when the rider looks down at the tines.
pub fn truck_boxes_posed(body: &Body, eye_m: f32, pose: &KinPose) -> Vec<LinkBox> {
    let eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (l, w) = (body.support_len_m, body.support_width_m);
    let b = |name: &'static str, segment: Segment, c: [f32; 3], half: [f32; 3]| LinkBox { name, segment, center: c, axes: eye3, half };
    let deck = WHEEL_R_M + 0.04;
    let top = deck + 0.20;
    let seat = (eye_m - 0.315).max(top + 0.04);
    let roof = seat + 0.315 + 0.13;
    // Pitch forward by `deg` about the left axis: up tips toward the front.
    let pitched = |deg: f32, v: [f32; 3]| {
        let (s, c) = deg.to_radians().sin_cos();
        [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
    };
    let (lean, head) = (pose.trunk_deg(), pose.trunk_deg() + pose.neck_deg);
    let hip = [-0.07, 0.0, seat];
    let add = |a: [f32; 3], d: [f32; 3]| [a[0] + d[0], a[1] + d[1], a[2] + d[2]];
    let neck = add(hip, pitched(lean, [0.0, 0.0, 0.24]));
    let tilted = |name: &'static str, segment: Segment, deg: f32, c: [f32; 3], half: [f32; 3]| LinkBox { name, segment, center: c, axes: [pitched(deg, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], pitched(deg, [0.0, 0.0, 1.0])], half };
    let mut v = vec![
        b("truck_body", Segment::Leg, [-0.01, 0.0, 0.5 * (deck + top)], [0.5 * l + 0.01, 0.5 * w - 0.01, 0.5 * (top - deck)]),
        b("truck_counterweight", Segment::Foot, [-0.5 * l - 0.045, 0.0, 0.5 * (deck + top) + 0.01], [0.045, 0.5 * w - 0.02, 0.5 * (top - deck) + 0.01]),
        b("truck_seat", Segment::Foot, [-0.07, 0.0, seat - 0.02], [0.08, 0.11, 0.02]),
        b("truck_seat", Segment::Foot, [-0.15, 0.0, seat + 0.11], [0.015, 0.11, 0.12]),
        b("truck_dash", Segment::Leg, [0.5 * l - 0.03, 0.0, 0.5 * (top + seat + 0.08)], [0.03, 0.5 * w - 0.05, 0.5 * (seat + 0.08 - top)]),
        b("truck_steer", Segment::Foot, [0.5 * l - 0.07, 0.0, seat + 0.13], [0.008, 0.07, 0.07]),
        b("truck_roof", Segment::Foot, [-0.02, 0.0, roof], [0.5 * l + 0.02, 0.5 * w - 0.01, 0.012]),
        tilted("rider_torso", Segment::Chest, lean, add(hip, pitched(lean, [0.0, 0.0, 0.12])), [0.06, 0.09, 0.12]),
        tilted("rider_head", Segment::Head, head, add(neck, pitched(head, [0.01, 0.0, 0.075])), [0.055, 0.055, 0.065]),
        tilted("rider_helmet", Segment::Leg, head, add(neck, pitched(head, [0.01, 0.0, 0.15])), [0.06, 0.06, 0.015]),
    ];
    if seat - 0.04 > top {
        v.push(b("truck_cab", Segment::Leg, [-0.07, 0.0, 0.5 * (top + seat - 0.04)], [0.09, 0.12, 0.5 * (seat - 0.04 - top)]));
    }
    for y in [-0.05f32, 0.05] {
        v.push(b("rider_leg", Segment::Belly, [0.0, y, seat + 0.03], [0.08, 0.035, 0.03]));
        v.push(b("rider_leg", Segment::Belly, [0.08, y, seat - 0.05], [0.03, 0.035, 0.08]));
    }
    for y in [-0.1f32, 0.1] {
        v.push(b("rider_arm", Segment::Chest, [0.0, y, seat + 0.16], [0.08, 0.025, 0.025]));
    }
    for x in [0.5 * l - 0.01, -0.5 * l - 0.03] {
        for y in [-0.5 * w + 0.03, 0.5 * w - 0.03] {
            v.push(b("truck_guard", Segment::Foot, [x, y, 0.5 * (top + roof)], [0.012, 0.012, 0.5 * (roof - top)]));
        }
    }
    v
}

/// The mast, carriage and tines at `fork_m`, in the body frame.
pub fn boxes(body: &Body, fork_m: f32) -> Vec<LinkBox> {
    let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let fx = front_x(body);
    let hx = heel_x(body);
    let h = fork_m.clamp(FORK_FLOOR_M, FORK_MAX_M);
    let mut v = Vec::new();
    for y in [-0.5 * TINE_SPREAD_M - 0.03, 0.5 * TINE_SPREAD_M + 0.03] {
        v.push(LinkBox { name: "fork_mast", segment: Segment::Leg, center: [fx + 0.012, y, 0.5 * (0.02 + MAST_TOP_M)], axes: eye, half: [0.012, 0.012, 0.5 * (MAST_TOP_M - 0.02)] });
    }
    v.push(LinkBox { name: "fork_carriage", segment: Segment::Foot, center: [hx - 0.008, 0.0, h + 0.05], axes: eye, half: [0.008, 0.5 * TINE_SPREAD_M + 0.04, 0.05] });
    for y in [-0.5 * TINE_SPREAD_M, 0.5 * TINE_SPREAD_M] {
        v.push(LinkBox { name: "fork_tine", segment: Segment::Foot, center: [hx + 0.5 * TINE_LEN_M, y, h + 0.5 * TINE_T_M], axes: eye, half: [0.5 * TINE_LEN_M, 0.5 * TINE_W_M, 0.5 * TINE_T_M] });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    fn body(base: BaseKind) -> Body {
        Body::new(&BodyPlan::default(), base, FingerCount::Five)
    }

    #[test]
    fn wheels_and_tracks_use_forks_and_legs_do_not() {
        assert!(applies(&body(BaseKind::Wheel)) && applies(&body(BaseKind::Track)) && applies(&body(BaseKind::Mecanum)));
        assert!(!applies(&body(BaseKind::HumanFoot)));
    }

    #[test]
    fn a_pick_lowers_slides_in_then_lifts_and_the_rider_keeps_looking_ahead() {
        let b = body(BaseKind::Wheel);
        let r = pick(&b, FORK_REST_M, 0.55, 2.4, [0.15; 3], 0.0, 0.0, 0.4);
        assert!(r.outcome.ok, "{:?}", r.outcome);
        let phases: Vec<&str> = r.samples.iter().map(|s| s.phase).collect::<Vec<_>>();
        let first = |p: &str| phases.iter().position(|q| *q == p).unwrap();
        assert!(first("align") < first("lower") && first("lower") < first("insert") && first("insert") < first("raise"), "{phases:?}");
        assert!(r.samples.iter().filter(|s| matches!(s.phase, "align" | "lower" | "insert")).all(|s| s.pose.neck_deg == 0.0), "the rider looks ahead while the carriage camera sees the item");
        let low = r.samples.iter().filter(|s| s.phase == "insert").map(|s| s.pose.fork_m).fold(f32::INFINITY, f32::min);
        assert!(low <= 1e-3, "the tines are on the floor while they slide in: {low}");
        let end = r.samples.last().unwrap();
        assert!((end.pose.fork_m - FORK_CARRY_M).abs() < 1e-3, "{}", end.pose.fork_m);
        assert!(end.pose.neck_deg.abs() < 0.5, "the rider looks ahead again with the load up: {}", end.pose.neck_deg);
        let obj = end.object.unwrap();
        assert!((obj[0] - load_x(&b, 0.15)).abs() < 1e-4 && obj[2] > FORK_CARRY_M, "the log rides on the tines: {obj:?}");
        let slide = r.metric("slide_m").unwrap();
        assert!((slide - (0.55f32.max(standoff_x(&b, 0.15)) - load_x(&b, 0.15))).abs() < 1e-4, "the tines come down in front of the log, then slide in: {slide}");
    }

    #[test]
    fn the_item_blocks_the_carriage_camera_and_the_rider_looks_down_at_it_before_the_lift() {
        let b = body(BaseKind::Wheel);
        let size = [0.15; 3];
        let r = pick(&b, FORK_FLOOR_M, standoff_x(&b, 0.15), 2.4, size, 0.0, 0.0, 0.0);
        assert!(r.outcome.ok, "{:?}", r.outcome);
        let phases: Vec<&str> = r.samples.iter().map(|s| s.phase).fold(Vec::new(), |mut v, p| {
            if v.last() != Some(&p) {
                v.push(p);
            }
            v
        });
        assert_eq!(phases, ["insert", "recheck_look", "recheck", "seat", "raise"], "{phases:?}");
        let at_stop = r.samples.iter().filter(|s| s.phase == "insert").last().unwrap().object.unwrap();
        let share = camera_blocked_share(&b, 0.0, at_stop, size);
        assert!(share >= BLOCKED_SHARE && (r.metric("blocked_share").unwrap() - share).abs() < 1e-4, "the base stops once the log fills the carriage camera: {share}");
        let far = camera_blocked_share(&b, 0.0, [standoff_x(&b, 0.15), 0.0, 0.075], size);
        assert!(far < 0.4, "from the standoff the carriage camera sees past the log: {far}");
        assert!(at_stop[0] > load_x(&b, 0.15) + 0.01, "it stops before the log is on: {at_stop:?}");
        let hold: Vec<&ActSample> = r.samples.iter().filter(|s| s.phase == "recheck").collect();
        assert!(hold.iter().all(|s| s.object.unwrap() == at_stop), "the base holds still while the rider looks");
        let neck = hold.last().unwrap().pose.neck_deg;
        assert!(neck > 25.0, "the rider bends the head to the load: {neck}");
        let cam = kin::head_camera(&b, &hold.last().unwrap().pose);
        let to = (cam.pos[2] - at_stop[2]).atan2(at_stop[0] - cam.pos[0]).to_degrees();
        let short = gaze::aim_shortfall_deg(&b, &hold.last().unwrap().pose, at_stop[0], at_stop[2]);
        assert!((cam.pitch_down_deg - gaze::HEAD_CAMERA_MOUNT_DEG - to).abs() < 1.5 + short, "the head camera centres the log: axis {:.1}° vs {to:.1}° (short {short:.1}°)", cam.pitch_down_deg - gaze::HEAD_CAMERA_MOUNT_DEG);
        let seat: Vec<&ActSample> = r.samples.iter().filter(|s| s.phase == "seat").collect();
        assert!(seat.windows(2).all(|w| w[1].object.unwrap()[0] <= w[0].object.unwrap()[0] + 1e-5) && seat.last().unwrap().object.unwrap()[0] - load_x(&b, 0.15) < 1e-4, "it rolls the rest of the way on");
        assert!(r.events.iter().any(|e| e.kind == "fork_cam_blocked") && r.events.iter().filter(|e| e.kind == "rider_check").count() == 1);
    }

    #[test]
    fn a_look_has_the_rider_check_first_then_the_carriage_camera() {
        let b = body(BaseKind::Track);
        let r = look(&b, FORK_REST_M, 0.62, [0.15; 3], 0.0, 0.0, 0.0);
        let phases: Vec<&str> = r.samples.iter().map(|s| s.phase).fold(Vec::new(), |mut v, p| {
            if v.last() != Some(&p) {
                v.push(p);
            }
            v
        });
        assert_eq!(phases, ["rider_look", "rider_check", "lower", "check"], "{phases:?}");
        let rider = r.samples.iter().filter(|s| s.phase == "rider_check").last().unwrap();
        assert!(rider.pose.neck_deg > 20.0 && (rider.pose.fork_m - FORK_REST_M).abs() < 1e-4, "the rider looks down while the tines are still up: {:?}", rider.pose);
        let fork = r.samples.last().unwrap();
        assert!(fork.pose.neck_deg.abs() < 0.5 && fork.pose.fork_m <= 1e-3, "looking ahead again, tines down for the carriage camera: {:?}", fork.pose);
        let kinds: Vec<&str> = r.events.iter().map(|e| e.kind).collect();
        let at = |k: &str| kinds.iter().position(|x| *x == k).unwrap();
        assert!(at("rider_check") < at("fork_down") && at("fork_down") < at("vision_check"), "{kinds:?}");
    }

    #[test]
    fn the_rider_leans_and_bends_the_head_in_the_truck_drawing() {
        let b = body(BaseKind::Wheel);
        let ahead = truck_boxes(&b, 1.0);
        let down = truck_boxes_posed(&b, 1.0, &KinPose { neck_deg: 40.0, ..KinPose::stand(0.0) });
        let head = |v: &[LinkBox]| v.iter().find(|x| x.name == "rider_head").unwrap().clone();
        let (h0, h1) = (head(&ahead), head(&down));
        assert!(h1.center[0] > h0.center[0] && h1.center[2] < h0.center[2], "the head tips forward and down: {:?} -> {:?}", h0.center, h1.center);
        assert!(h1.axes[2][0] > 0.5, "the head box turns with the neck: {:?}", h1.axes);
        let torso = |v: &[LinkBox]| v.iter().find(|x| x.name == "rider_torso").unwrap().center;
        assert_eq!(torso(&ahead), torso(&down), "the neck alone, no lean");
    }

    #[test]
    fn a_load_that_would_tip_the_base_or_falls_between_the_tines_is_refused() {
        let b = body(BaseKind::Wheel);
        assert!(check(&b, 2.4, [0.15; 3], 0.0, 0.0).is_ok());
        let heavy = check(&b, 200.0, [0.15; 3], 0.0, 0.0).unwrap_err();
        assert!(heavy.contains("포크 하중"), "{heavy}");
        let thin = check(&b, 0.3, [0.05, 0.05, 0.05], 0.0, 0.0).unwrap_err();
        assert!(thin.contains("포크 사이"), "{thin}");
        let far = check(&b, 1.0, [0.15; 3], 0.0, 0.0).unwrap();
        assert!(far.tip_ratio > TIP_MIN && far.capacity_kg > 2.4, "{far:?}");
    }

    #[test]
    fn a_set_down_lowers_backs_out_and_raises_the_empty_tines() {
        let b = body(BaseKind::Track);
        let r = set_down(&b, FORK_CARRY_M, 2.4, [0.15; 3], 0.0, 2.4);
        assert!(r.outcome.ok);
        let objs: Vec<f32> = r.samples.iter().filter(|s| s.phase == "back_out").map(|s| s.object.unwrap()[0]).collect();
        assert!(objs.windows(2).all(|w| w[1] >= w[0] - 1e-6) && objs.last().unwrap() > &(heel_x(&b) + TINE_LEN_M), "the base backs off the load: {objs:?}");
        assert!((r.samples.last().unwrap().pose.fork_m - FORK_REST_M).abs() < 1e-3);
    }

    #[test]
    fn a_look_lowers_the_tines_and_holds_for_the_camera() {
        let b = body(BaseKind::Wheel);
        let r = look(&b, FORK_REST_M, 0.3, [0.15; 3], 0.0, 0.0, 0.0);
        assert!(r.samples.last().unwrap().object.unwrap()[0] >= standoff_x(&b, 0.15) - 1e-4, "it backs off so the tines come down in front of the item");
        assert!(r.samples.iter().any(|s| s.phase == "check") && r.samples.last().unwrap().pose.fork_m <= 1e-3);
        let cam = camera_pos(&b, 0.0);
        assert!(cam[2] < 0.1 && cam[0] > front_x(&b), "the carriage camera rides low at the front: {cam:?}");
    }
}
