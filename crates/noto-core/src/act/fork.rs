//! Forklift handling for wheeled and tracked bases. A rider sits on the base and looks ahead; the work at the floor is
//! done by a pair of tines on a mast at the front, as on a counterbalanced lift truck: the tines go down to the floor,
//! a camera on the carriage between them checks the item, the base drives forward until the item sits on the tines,
//! and the mast lifts it to travel height. Setting down runs the other way: lower, back out, raise the empty tines.
//!
//! The load may not tip the base over its front axle. A counterbalanced truck stays upright while the moment of the
//! load about the front axle is less than the moment of the truck's own weight about it (the longitudinal stability
//! of the "stability triangle"; ISO 22915-2 tests it on a tilted platform); here the base's weight must hold the load
//! with `TIP_MIN` to spare.

use crate::balance::{Body, G};
use crate::dynamics::LinkBox;
use crate::kin::{self, KinLoads, KinPose, BUMPER_GAP_M, WHEEL_R_M};
use crate::parts::Segment;

use super::{min_jerk, ActEvent, ActKind, ActRollout, ActSample, Outcome, Part, SAMPLE_S};

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
    let pose = KinPose { fork_m, ..KinPose::stand(cargo_kg) };
    let r = kin::solve_fast(body, &pose, &KinLoads::default());
    ActSample { t, phase, pose, loads: KinLoads::default(), margin: r.margin, zmp: r.zmp, com: r.com, peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: Some(object) }
}

/// Moves the tines from `from_m` to `to_m` on the mast, `phase` by phase, appending samples from `t0`. Returns the end.
fn travel(body: &Body, samples: &mut Vec<ActSample>, t0: f32, phase: &'static str, from_m: f32, to_m: f32, cargo_kg: f32, object: impl Fn(f32) -> [f32; 3]) -> f32 {
    let dt = ((to_m - from_m).abs() / MAST_MPS).max(SAMPLE_S);
    let n = (dt / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        let h = from_m + (to_m - from_m) * u;
        samples.push(sample(body, t0 + dt * i as f32 / n as f32, phase, h, cargo_kg, object(h)));
    }
    t0 + dt
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

/// The look before the tines: the base turns to the item for `turn_s` (and backs off if the tines would come down
/// on it), the tines go down to the item's base (`base_m`), and they hold while the carriage camera sees the item.
/// The rider keeps looking ahead.
#[allow(clippy::too_many_arguments)]
pub fn look(body: &Body, from_m: f32, dist_m: f32, len_m: f32, base_m: f32, cargo_kg: f32, turn_s: f32) -> ActRollout {
    let floor = base_m.max(FORK_FLOOR_M);
    let x = dist_m.max(standoff_x(body, len_m));
    let obj = [x, 0.0, base_m];
    let mut samples = vec![sample(body, 0.0, "align", from_m, cargo_kg, [dist_m, 0.0, base_m])];
    let t0 = align(body, &mut samples, turn_s, from_m, cargo_kg, dist_m, x, base_m);
    if t0 == 0.0 {
        samples[0].phase = "lower";
    }
    let t = travel(body, &mut samples, t0, "lower", from_m, floor, cargo_kg, |_| obj);
    let mut events = vec![ActEvent::new(t, "fork_down", format!("포크를 {:.2} m로 내림: 탑승자는 정면, 포크 카메라가 물건을 봄", floor), &[Part::Wheel, Part::Track], Some(floor), true)];
    let m = (CHECK_S / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m {
        samples.push(sample(body, t + CHECK_S * i as f32 / m as f32, "check", floor, cargo_kg, obj));
    }
    let end = t + CHECK_S;
    events.push(ActEvent::new(end, "vision_check", format!("포크 카메라로 {:.2} s 확인", CHECK_S), &[Part::Wheel, Part::Track], Some(CHECK_S), true));
    let metrics = vec![("fork_m", floor), ("look_s", t), ("fixate_s", CHECK_S), ("duration_s", end)];
    ActRollout { kind: ActKind::Fork, samples, events, outcome: Outcome::ok(), duration_s: end, metrics, notes: vec![] }
}

/// Picks an item `dist_m` ahead (its centre) up on the tines: turn to it for `turn_s`, down (if not already), slide in
/// (back off first when it is closer than the tines reach), lift to travel height. `object` in each sample is the
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
    let dt = (slide.abs() / INSERT_MPS).max(SAMPLE_S);
    let n = (dt / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        samples.push(sample(body, t + dt * i as f32 / n as f32, "insert", floor, load_kg, at_rest(dist_m - slide * u)));
    }
    t += dt;
    events.push(ActEvent::new(t, "fork_in", if slide >= 0.0 { format!("{:.0} cm 전진해 포크를 물건 밑으로 넣음", slide * 100.0) } else { format!("{:.0} cm 물러나 포크 끝을 물건에 맞춤", -slide * 100.0) }, &[Part::Wheel, Part::Track], Some(slide), true));
    let carry = (floor + FORK_CARRY_M).min(FORK_MAX_M).max(FORK_CARRY_M);
    let on = |h: f32| [fit.load_x_m, 0.0, h + TINE_T_M + half_h];
    t = travel(body, &mut samples, t, "raise", floor, carry, load_kg + mass_kg, on);
    events.push(ActEvent::new(t, "fork_up", format!("마스트로 {:.1} kg를 {carry:.2} m까지 들어올림 (전도 여유 {:.2})", mass_kg, fit.tip_ratio), &[Part::Wheel, Part::Track], Some(fit.tip_ratio), true));
    let metrics = vec![("fork_m", carry), ("slide_m", slide), ("tip_ratio", fit.tip_ratio), ("capacity_kg", fit.capacity_kg), ("duration_s", t)];
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
    let eye3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (l, w) = (body.support_len_m, body.support_width_m);
    let b = |name: &'static str, segment: Segment, c: [f32; 3], half: [f32; 3]| LinkBox { name, segment, center: c, axes: eye3, half };
    let deck = WHEEL_R_M + 0.04;
    let top = deck + 0.20;
    let seat = (eye_m - 0.315).max(top + 0.04);
    let roof = seat + 0.315 + 0.13;
    let mut v = vec![
        b("truck_body", Segment::Leg, [-0.01, 0.0, 0.5 * (deck + top)], [0.5 * l + 0.01, 0.5 * w - 0.01, 0.5 * (top - deck)]),
        b("truck_counterweight", Segment::Foot, [-0.5 * l - 0.045, 0.0, 0.5 * (deck + top) + 0.01], [0.045, 0.5 * w - 0.02, 0.5 * (top - deck) + 0.01]),
        b("truck_seat", Segment::Foot, [-0.07, 0.0, seat - 0.02], [0.08, 0.11, 0.02]),
        b("truck_seat", Segment::Foot, [-0.15, 0.0, seat + 0.11], [0.015, 0.11, 0.12]),
        b("truck_dash", Segment::Leg, [0.5 * l - 0.03, 0.0, 0.5 * (top + seat + 0.08)], [0.03, 0.5 * w - 0.05, 0.5 * (seat + 0.08 - top)]),
        b("truck_steer", Segment::Foot, [0.5 * l - 0.07, 0.0, seat + 0.13], [0.008, 0.07, 0.07]),
        b("truck_roof", Segment::Foot, [-0.02, 0.0, roof], [0.5 * l + 0.02, 0.5 * w - 0.01, 0.012]),
        b("rider_torso", Segment::Chest, [-0.07, 0.0, seat + 0.12], [0.06, 0.09, 0.12]),
        b("rider_head", Segment::Head, [-0.06, 0.0, seat + 0.315], [0.055, 0.055, 0.065]),
        b("rider_helmet", Segment::Leg, [-0.06, 0.0, seat + 0.39], [0.06, 0.06, 0.015]),
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
        assert!(r.samples.iter().all(|s| s.pose.neck_deg == 0.0), "the rider does not bend to the floor");
        let low = r.samples.iter().filter(|s| s.phase == "insert").map(|s| s.pose.fork_m).fold(f32::INFINITY, f32::min);
        assert!(low <= 1e-3, "the tines are on the floor while they slide in: {low}");
        let end = r.samples.last().unwrap();
        assert!((end.pose.fork_m - FORK_CARRY_M).abs() < 1e-3, "{}", end.pose.fork_m);
        let obj = end.object.unwrap();
        assert!((obj[0] - load_x(&b, 0.15)).abs() < 1e-4 && obj[2] > FORK_CARRY_M, "the log rides on the tines: {obj:?}");
        let slide = r.metric("slide_m").unwrap();
        assert!((slide - (0.55f32.max(standoff_x(&b, 0.15)) - load_x(&b, 0.15))).abs() < 1e-4, "the tines come down in front of the log, then slide in: {slide}");
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
        let r = look(&b, FORK_REST_M, 0.3, 0.15, 0.0, 0.0, 0.0);
        assert!(r.samples.last().unwrap().object.unwrap()[0] >= standoff_x(&b, 0.15) - 1e-4, "it backs off so the tines come down in front of the item");
        assert!(r.samples.iter().any(|s| s.phase == "check") && r.samples.last().unwrap().pose.fork_m <= 1e-3);
        let cam = camera_pos(&b, 0.0);
        assert!(cam[2] < 0.1 && cam[0] > front_x(&b), "the carriage camera rides low at the front: {cam:?}");
    }
}
