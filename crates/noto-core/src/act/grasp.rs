//! Grasping an object off the ground and setting one down, as time series.
//!
//! **Grasp** (`grasp`): bend down (`reach`), close the fingers, squeeze,
//! lift off, rise with the object, stow it in the bay.
//!
//! * The hand arrives where the robot believes the object is; the belief is
//!   off by a drawn aim error (the head camera's and the arm's own errors).
//!   A palm that misses the object's top face closes on air.
//! * The fingers meet the object at their closing speed: a collision whose
//!   peak force follows the soft pad's stiffness (`material::impact`). Each
//!   finger stops where a phalanx touches (`sim::top_grasp`: per-phalanx
//!   contacts on the box's faces, the force each finger's drive presses with
//!   by the contact statics `τ = Jᵀλ`); the hand holds at those angles.
//!   One hand lifts when those contacts hold the load with the safety
//!   factor, else both (`top_hands`, which also says when two hands do not
//!   hold even the weight, so the planners refuse); a one-hand request in
//!   front that one hand does not hold takes both.
//! * The grip follows the load (v1.0, `docs/research/tactile.md`): after
//!   the tactile sensor confirms the contacts the fingers seat with a small
//!   preload, then the controller commands `GF = GF_pre + S·LF/μ_est`
//!   (S the grip safety factor, `GRIP_SAFETY`; LF the hand's share of the
//!   load, held by its fingers' face alone, `SQUEEZE_SURFACES`; `μ_est` the
//!   table friction it believes), reading the planned load `GRIP_LEAD_S`
//!   ahead, and keeps tracking the load through the lift. The squeeze is
//!   capped by what the fingers can press.
//! * Every instant the load is shared over the finger pads
//!   (`contact::share_load`): each touching finger presses its share of the
//!   grip, with its own pad friction (the object's friction drawn from the
//!   measured scatter, a finger-to-finger scatter on a separate draw, and
//!   the pad's fall with force `contact::pad_mu`); a pad under the box's
//!   edge holds part of the weight with its normal force. A pad whose stick
//!   ratio falls below the alarm is an incipient slip: after the detection
//!   delay the controller reads the friction off the tactile signal
//!   (`contact::mu_from_stick`) and raises the grip over the finger loop's
//!   rise time.
//! * Lift-off comes once the arm's pull reaches the weight; then the hand
//!   rises with the body on a minimum-jerk path and the load is the object's
//!   weight plus its inertia. The object slides out when the load passes what
//!   the pads hold together. When its weight's moment about the squeeze axis
//!   (the aim error across the fingers' row is the lever) takes the pads
//!   past their limit surfaces while the load alone would not, a pinch lets
//!   it pivot until its centre of mass hangs under the grasp axis (it stays
//!   held, tilted); a power grasp's palm on top blocks the turn.
//! * Every sample is solved by `kin` with the object's force on the hand:
//!   joints over their peak or a ZMP outside the support stop the lift.
//!
//! **Release** (`release`): take the object from the bay, bend down as far as
//! the body allows over the ground or a pile, open the hand (held at the
//! contact-stopped closure until it opens). The object falls
//! the rest of the way, hits with the ground's stiffness, rebounds with its
//! restitution and lands a little away. A body that cannot bend drops it from
//! hand height.
//!
//! The form (`morph`) decides where things go: the bay on the pelvis, a tote
//! on the deck or a bed (`Body::cargo_geom`); a form without a cargo place
//! keeps the object in the hand (and says so, `stow_unreachable`, when its
//! hand cannot get over the place it has). A form without a manipulator
//! refuses at once (`no_manipulator`), a single arm refuses a two-handed lift
//! (`one_arm`), and a parallel-jaw gripper squeezes with its two jaws
//! (`capacity_for`).

use serde::Serialize;

use crate::balance::{Body, CARGO_X_M, CARGO_Z_M, G, GRIP_SAFETY};
use crate::contact::{self, FingerContact, FingerContacts, PadSurface};
use crate::hand::Hand;
use crate::kin::{self, KinLoads, KinPose};
use crate::math::Vec3;
use crate::parts::*;
use crate::sim::{grasp as close_hand, top_grasp, ContactSource, GraspSetup, Payload, TopGrasp};
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
    /// Both hands. One hand (false) is a request: a pick in front whose one
    /// hand's contacts do not hold the load with the safety factor takes both
    /// ([`top_hands`]; the metric `arms` says how many it took), a load held
    /// between the feet takes both and a reach to the side the right one.
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

/// Whether one hand lifts an object off the ground from above at the
/// friction the controller believes, with its safety factor: the same
/// contacts the rollout holds with (`sim::top_grasp`, the fingers stopped
/// where they touch and pressing what their joints allow by `τ = Jᵀλ`), the
/// fingers' face alone carrying the weight by friction (`SQUEEZE_SURFACES`)
/// plus what a pad under the box's edge holds ([`top_hold_n`]), must carry
/// `GRIP_SAFETY` times the weight, with at least two fingers touching (the
/// palm on top carries none of it). The squeeze and the answer.
pub fn top_grasp_ok(hand: &Hand, size_m: [f32; 3], mass_kg: f32, mu_est: f32) -> (f32, bool) {
    let closed = top_grasp(hand, size_m);
    (closed.squeeze_max_n(), closed.touching >= 2 && top_hold_n(&closed, mu_est) >= GRIP_SAFETY * mass_kg * G)
}

/// The load a hand closed from above holds at its fingers' squeeze limit and
/// the friction `mu`, N: per touching finger, its force times the friction on
/// its face and, for a pad under the box's edge, the share of that force the
/// face's normal turns against the load (as `HandPads::state`; no pad
/// scatter, no fall of the friction with force).
pub fn top_hold_n(closed: &TopGrasp, mu: f32) -> f32 {
    closed
        .finger_max_n
        .iter()
        .enumerate()
        .map(|(fi, &fmax)| {
            let edge = closed.contacts.iter().filter(|c| c.finger == fi).max_by(|a, b| a.max_n.total_cmp(&b.max_n)).map_or(0.0, |c| c.normal.y.max(0.0));
            fmax.max(0.0) * (SQUEEZE_SURFACES * mu.max(0.0) + edge)
        })
        .sum()
}

/// How many hands a grasp from above takes, read off the contacts the
/// rollout holds with ([`top_hold_n`] per hand; two hands each carry half
/// the load, each by its own fingers' face).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TopHands {
    /// What one hand holds at its fingers' squeeze limit and the believed friction, N.
    pub hold_n: f32,
    /// The object's weight, N.
    pub weight_n: f32,
    /// Fingers of one hand that touch the object.
    pub touching: usize,
    /// The hands to take: 1 when one hand holds `GRIP_SAFETY` times the
    /// weight; 2 when one does not and two hands hold the weight — with the
    /// safety factor, or below it at their squeeze limit (the rollout then
    /// lifts slowly, `LIFT_SLOWEST_S`, and the pads' friction scatter can
    /// let a few slip); 0 when two hands do not hold even the weight at the
    /// believed friction: the object would slide out of them, so the caller
    /// refuses the pick instead of playing a doomed rollout.
    pub hands: u8,
}

impl TopHands {
    /// The answer for a hand already closed on the object (`sim::top_grasp`).
    pub fn of(closed: &TopGrasp, mass_kg: f32, mu_est: f32) -> TopHands {
        let hold_n = if closed.touching >= 2 { top_hold_n(closed, mu_est) } else { 0.0 };
        let weight_n = mass_kg.max(0.0) * G;
        let hands = if closed.touching < 2 {
            0
        } else if hold_n >= GRIP_SAFETY * weight_n {
            1
        } else if 2.0 * hold_n >= weight_n {
            2
        } else {
            0
        };
        TopHands { hold_n, weight_n, touching: closed.touching, hands }
    }

    /// What `hands` hands hold over `GRIP_SAFETY` times the weight (≥ 1: the full safety factor).
    pub fn margin(&self, hands: u8) -> f32 {
        if self.weight_n > 1e-6 {
            hands as f32 * self.hold_n / (GRIP_SAFETY * self.weight_n)
        } else {
            99.0
        }
    }

    /// The hands it takes hold the object with the full safety factor.
    pub fn safe(&self) -> bool {
        self.hands > 0 && self.margin(self.hands) >= 1.0
    }
}

/// The planner on the rollout's contacts: one hand, two, or none (see
/// [`TopHands::hands`]). One hand is [`top_grasp_ok`]'s answer; the second
/// check is whether two hands then suffice. Fewer than two touching fingers
/// hold nothing. The rollout ([`grasp`]) reads the same answer: a pick in
/// front asked with one hand that this says needs two takes both.
pub fn top_hands(hand: &Hand, size_m: [f32; 3], mass_kg: f32, mu_est: f32) -> TopHands {
    TopHands::of(&top_grasp(hand, size_m), mass_kg, mu_est)
}

/// The v0.9 table of what the fingers can do on the object (`sim::grasp`:
/// the object resting on the palm, each finger's force its knuckle torque
/// over the lever to its contact). The world's grasp table (`GraspInfo`:
/// which items a hand holds, one hand or two) still reads it; the rollout,
/// [`top_grasp_ok`] and [`top_hands`] use the contact statics of a grasp
/// from above (`sim::top_grasp`), whose distal joints let the fingers press
/// about half as hard (`squeeze_plan_n` against `squeeze_cap_n` in the
/// metrics). The rollout takes both hands where one hand's contacts do not
/// hold the load ([`GraspInput::two_arms`]), but until the world reads
/// [`top_hands`] (its `margin(1)` is this table's margin on the rollout's
/// contacts) its labels and the items it lets a hand try differ from what
/// the rollout holds (`docs/research/tactile.md` §7: a three-finger hand's
/// two-hand log): that switch changes which items the hands can hold, a
/// world re-baseline.
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

/// [`capacity`] of the hand at the end of the arm (`Body::working_hand`):
/// the configured hand's fingers, or a parallel-jaw gripper's two jaws. A
/// gripper whose jaw force a source gives (`GripperPreset::peak_n`) presses
/// the object with it from both sides, so the friction acts on two faces with
/// `2·F` (the squeeze at the actuator's peak, as the fingers' knuckle peak);
/// without one (the mid tier's XM430 jaw) the two-finger hand's own contacts
/// stand (est).
pub fn capacity_for(body: &Body, hand: &Hand, size_m: [f32; 3], mass_kg: f32, mu_est: f32) -> HandCapacity {
    let working = body.working_hand(hand);
    let mut c = capacity(&working, size_m, mass_kg, mu_est);
    if let Some(f) = body.gripper.as_ref().filter(|_| body.gripper_only()).and_then(|g| g.peak_n) {
        if c.contacts >= 2 {
            c.finger_n = vec![f; c.finger_n.len().max(2)];
            c.squeeze_n = 2.0 * f;
            c.contacts = 2;
        }
    }
    c
}

/// Salt of the per-finger friction draw: a stream of its own, so the
/// grasp's own draws (the object's friction, the aim) stay as they were.
pub const FINGER_MU_STREAM: u64 = 0xF16E_C0A7_5EED_0001;

/// The object's surface under the pads: an object lifted off wet sand or
/// out of water is wet; the dust of coal is already in its table friction.
pub fn pad_surface(ground: Surface) -> PadSurface {
    match ground {
        Surface::WetSand | Surface::Water => PadSurface::Wet,
        _ => PadSurface::Dry,
    }
}

/// One touching finger of a hand closed on an object from above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadFinger {
    pub finger: usize,
    /// The phalanx pressing hardest, where it touches and the face's normal there (hand frame).
    pub phalanx: usize,
    pub point: Vec3,
    pub normal: Vec3,
    /// The finger's share of the squeeze: its drive's force over the hand's,
    /// so every finger reaches its limit together.
    pub share: f32,
    /// This finger's pad friction over the object's (the finger-to-finger scatter).
    pub mu_scale: f32,
}

/// The faces whose friction carries a hand's share of the load: one, the
/// face its fingers' pads press. The hand closed from above has nothing on
/// the near face (`sim::top_grasp`: the thumb sits in the fingers' row), so
/// what presses back against the squeeze is assumed and its friction not
/// counted (est) — with one hand an unplaced reaction (the palm's heel or
/// the wrist), with two hands the other hand's fingers, whose own pads hold
/// the other half of the load (counting them again per hand would count
/// that face twice). An opposed pinch with a pad on each face counts both
/// (`contact::PINCH_FACES`).
pub const SQUEEZE_SURFACES: f32 = 1.0;
/// How far ahead of the arm's planned pull and lift the grip command reads
/// the load, seconds (est: the tactile loop's two-frame confirmation,
/// `contact::detect_delay_s`, reused, so a correction from the sensed load
/// never chases a command already behind it; no robot source was found).
/// The grip's lead over the load (`gf_lead_s`) follows from it by design
/// while the squeeze is below its limit.
pub const GRIP_LEAD_S: f32 = contact::DETECT_FRAMES / contact::TACTILE_FPS;

/// The fingers of a hand closed on an object from above (`sim::top_grasp`)
/// and what their pads do under a grip, a load and a twist.
#[derive(Clone, Debug, PartialEq)]
pub struct HandPads {
    pub closed: TopGrasp,
    pub fingers: Vec<PadFinger>,
    pub surface: PadSurface,
}

/// What the pads do at one instant.
#[derive(Clone, Debug, PartialEq)]
pub struct PadState {
    /// Every finger pad's share of the load (in `HandPads::fingers` order).
    pub shared: contact::Shared,
    /// The smallest stick ratio of a loaded finger pad (what the fingertip sensors feel).
    pub finger_stick_min: f32,
    /// The part of the load the pads under the box's edge hold with their normal force, N.
    pub support_n: f32,
    /// The load the pads hold together (friction with the twist, plus the edge support), N.
    pub capacity_n: f32,
    /// The twist the pads resist together, N·m.
    pub torsion_cap_nm: f32,
    /// The pads' mean friction, weighted by their normal force.
    pub mu_mean: f32,
    pub contacts: FingerContacts,
}

impl HandPads {
    /// Closes `hand` on a box of `size_m` and draws each finger's pad
    /// friction scatter on the grasp's finger stream (`seed`).
    pub fn new(hand: &Hand, size_m: [f32; 3], surface: PadSurface, seed: u64) -> HandPads {
        let closed = top_grasp(hand, size_m);
        let mut d = Draw::new(seed ^ FINGER_MU_STREAM);
        let scales: Vec<f32> = (0..hand.fingers.len()).map(|_| d.lognormal(1.0, contact::FINGER_MU_COV)).collect();
        let total = closed.squeeze_max_n().max(1e-6);
        let mut fingers = Vec::new();
        for (fi, &fmax) in closed.finger_max_n.iter().enumerate() {
            if fmax <= 1e-6 {
                continue;
            }
            if let Some(c) = closed.contacts.iter().filter(|c| c.finger == fi).max_by(|a, b| a.max_n.total_cmp(&b.max_n)) {
                fingers.push(PadFinger { finger: fi, phalanx: c.phalanx, point: c.point, normal: c.normal, share: fmax / total, mu_scale: scales[fi] });
            }
        }
        HandPads { closed, fingers, surface }
    }

    pub fn touching(&self) -> usize {
        self.fingers.len()
    }

    /// The most the fingers squeeze together, N.
    pub fn squeeze_max_n(&self) -> f32 {
        self.closed.squeeze_max_n()
    }

    /// The pads under a grip `grip_n` (the fingers' normal forces together),
    /// a load `load_n` pulling the object out of the hand (its weight and
    /// inertia, toward the fingers' curl side) and a twist `twist_nm` about
    /// the squeeze axis, on an object of friction `mu_obj`: the load and the
    /// twist share over the finger pads, the one face whose friction counts
    /// (`SQUEEZE_SURFACES`).
    pub fn state(&self, mu_obj: f32, grip_n: f32, load_n: f32, twist_nm: f32) -> PadState {
        let n: Vec<f32> = self.fingers.iter().map(|f| grip_n.max(0.0) * f.share).collect();
        let mus: Vec<f32> = self.fingers.iter().zip(&n).map(|(f, &ni)| contact::pad_mu(mu_obj * f.mu_scale, ni, self.surface)).collect();
        // A pad under the box's edge (its normal points down the load) holds that much with its normal force.
        let support_n = self.fingers.iter().zip(&n).map(|(f, &ni)| ni * f.normal.y.max(0.0)).sum::<f32>().min(load_n.max(0.0));
        let shared = contact::share_load(&n, &mus, (load_n - support_n).max(0.0), twist_nm);
        let torsion_cap_nm = shared.pads.iter().map(|p| contact::torsion_radius_m(p.n) * p.mu * p.n).sum();
        let n_sum: f32 = n.iter().sum();
        let mu_mean = if n_sum > 1e-6 { shared.pads.iter().map(|p| p.mu * p.n).sum::<f32>() / n_sum } else { mus.iter().sum::<f32>() / mus.len().max(1) as f32 };
        let finger_stick_min = shared.pads.iter().filter(|p| p.n > 1e-6).map(|p| p.stick).fold(1.0f32, f32::min);
        let mut contacts = FingerContacts::default();
        let up = Vec3::new(0.0, -1.0, 0.0);
        for (f, p) in self.fingers.iter().zip(&shared.pads) {
            // The pad's friction holds the object up the face (−y in the hand frame), in the face's plane.
            let t = up - f.normal.scale(up.dot(f.normal));
            let t = if t.length() > 1e-6 { t.normalized() } else { up };
            contacts.push(FingerContact {
                finger: f.finger as u8,
                phalanx: f.phalanx as u8,
                point: [f.point.x, f.point.y, f.point.z],
                normal: [f.normal.x, f.normal.y, f.normal.z],
                fn_n: p.n,
                ft: [t.x * p.ft, t.y * p.ft, t.z * p.ft],
                ft_n: p.ft,
                torsion_nm: p.torsion_nm,
                mu: p.mu,
                cone_use: p.cone_use,
                stick: p.stick,
                state: p.state,
                radius_m: p.radius_m,
                slip_dir: p.slip_dir,
            });
        }
        PadState { capacity_n: shared.capacity_n + support_n, support_n, torsion_cap_nm, mu_mean, finger_stick_min, shared, contacts }
    }
}

/// The friction a tactile signal reports for the loaded finger pads (the
/// sensors are in the fingertips): the lowest `contact::mu_from_stick` over
/// the finger pads that carry load. A twist shrinks the stick zone too
/// (`contact::share_load`), so the friction read under a twist is lower and
/// the grip rises against it.
fn mu_seen(st: &PadState, fingers: usize) -> f32 {
    st.shared.pads[..fingers.min(st.shared.pads.len())].iter().filter(|p| p.n > 1e-6 && p.ft > 1e-6).map(|p| contact::mu_from_stick(p.ft, p.n, p.stick).min(p.mu.max(p.ft / p.n))).fold(f32::INFINITY, f32::min)
}

/// The grip controller: grip follows the load it expects `GRIP_LEAD_S`
/// ahead, `GF = GF_pre + S·LF/μ_est` (the fingers' face alone holds it,
/// `SQUEEZE_SURFACES`), and an incipient slip lowers `μ_est` to what the
/// tactile signal reads, the grip rising to the new command over the finger
/// loop's rise time.
#[derive(Clone, Copy, Debug)]
struct GripCtl {
    preload_n: f32,
    cap_n: f32,
    mu_est: f32,
    /// Since when the controller acts on a new friction estimate, and the old one.
    react: Option<(f32, f32)>,
}

impl GripCtl {
    fn ff(&self, mu: f32, load_ahead: f32, seat: f32) -> f32 {
        (self.preload_n * seat + GRIP_SAFETY * load_ahead / (SQUEEZE_SURFACES * mu.max(0.05))).min(self.cap_n)
    }

    /// The grip at `t` with the load expected `GRIP_LEAD_S` ahead and the seat's share of the preload.
    fn grip(&self, t: f32, load_ahead: f32, seat: f32) -> f32 {
        match self.react {
            Some((t0, mu_old)) => {
                let u = ((t - t0) / LOOP_RISE_S).clamp(0.0, 1.0);
                let a = self.ff(mu_old, load_ahead, seat);
                a + (self.ff(self.mu_est, load_ahead, seat) - a) * u
            }
            None => self.ff(self.mu_est, load_ahead, seat),
        }
    }

    /// A pad's stick ratio fell below the alarm at `t`: from a detection delay later the controller grips for the friction it read.
    fn incipient(&mut self, t: f32, seen: f32) -> bool {
        let busy = self.react.is_some_and(|(t0, _)| t < t0 + LOOP_RISE_S);
        if busy || !seen.is_finite() || seen >= self.mu_est {
            return false;
        }
        self.react = Some((t + contact::detect_delay_s(), self.mu_est));
        self.mu_est = seen.max(0.05);
        true
    }
}

/// The pose that holds the object in front of the chest after the lift (the
/// `balance` arm-lift: upper arm 30° forward, forearm horizontal).
pub fn hold_pose(two_arms: bool, cargo_kg: f32) -> KinPose {
    KinPose { shoulder_deg: [30.0, if two_arms { 30.0 } else { 0.0 }], elbow_deg: [60.0, if two_arms { 60.0 } else { 10.0 }], closure: 1.0, ..KinPose::stand(cargo_kg) }
}

/// [`hold_pose`] for the form: an arm on a deck holds the object folded above
/// the deck, in its rest pose (`morph::DECK_REST_DEG`), since it cannot hang it
/// in front of a chest it does not have. An arm on a mast's carriage holds it
/// with the carriage where it will stow it ([`stow_pose`]'s lift), so the
/// carriage does not climb to the top of the mast and come down again.
pub fn hold_pose_for(body: &Body, two_arms: bool, cargo_kg: f32) -> KinPose {
    if body.deck_arms() {
        return KinPose { closure: 1.0, ..body.stand(cargo_kg) };
    }
    let hold = hold_pose(two_arms, cargo_kg);
    if body.mast_arms() {
        if let Some(s) = stow_pose(body, two_arms, cargo_kg) {
            return KinPose { lift_m: s.lift_m, ..hold };
        }
    }
    hold
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
        let (sy, ty) = (body.shoulder_mount(k)[1], if k == 0 { at[1] } else { -at[1] });
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

/// The pose that puts the hand over the form's cargo place
/// (`Body::cargo_geom`): [`bay_pose`] for the bay; over a tote or a bed, the
/// arm's inverse kinematics to the release point, with the lift (a mast's
/// carriage, a torso lift) lowered no further than it takes to get there.
/// `None` without a cargo place, or when the hand cannot get over it: then the
/// hand keeps the object.
pub fn stow_pose(body: &Body, two_arms: bool, cargo_kg: f32) -> Option<KinPose> {
    let c = body.cargo_geom()?;
    if c.on_pelvis {
        return Some(bay_pose(two_arms, cargo_kg));
    }
    let stand = body.stand(cargo_kg);
    let lift_max = kin::Rom::of(body).lift_max_m;
    let n = if lift_max > 0.0 { 40 } else { 0 };
    (0..=n).find_map(|i| {
        let mut p = KinPose { lift_m: if n == 0 { 0.0 } else { lift_max * i as f32 / n as f32 }, closure: 1.0, ..stand };
        let (sd, ed) = reach::arm_ik(reach::shoulder_point(body, &p), c.release, p.trunk_deg())?;
        for &k in holding_arms(two_arms) {
            p.shoulder_deg[k] = sd;
            p.elbow_deg[k] = ed;
        }
        Some(p)
    })
}

/// Seconds a lift's carriage takes between two poses (a minimum-jerk travel
/// at the lift's top speed); 0 without a lift. `reach::move_time` covers the
/// rotary joints only.
fn lift_travel_s(body: &Body, a: &KinPose, b: &KinPose) -> f32 {
    body.lift.as_ref().map_or(0.0, |l| super::MIN_JERK_PEAK * (a.lift_m - b.lift_m).abs() / l.speed_mps.max(1e-3))
}

/// Where a stowed object ends up: the cargo place's load point, or the hand.
fn stowed_at(body: &Body) -> [f32; 3] {
    body.cargo_point().0
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
    let mu = material::pad_on(inp.material);
    // The fingers (or a gripper's jaws, `Body::working_hand`) closed on the object (their friction scatter on a
    // stream of its own, no main draw).
    let working = body.working_hand(inp.hand);
    let pads = HandPads::new(&working, inp.size_m, pad_surface(inp.ground), inp.seed);
    // A pick in front asked with one hand whose contacts do not hold the load with the safety factor
    // takes both (`TopHands`: the caller's table may still be the v0.9 `capacity`), as a load held
    // between the feet does.
    let planned = TopHands::of(&pads.closed, inp.mass_kg, mu.mean);
    let both_needed = inp.mode == ReachMode::Front && !inp.two_arms && planned.hands != 1;
    let two_asked = inp.two_arms || both_needed;
    let goal = ReachGoal::ground_object(inp.size_m[1], inp.mass_kg, two_asked, inp.cargo_kg);
    // In front: the bend, or a wide squat over a load the bend cannot hold. Beside the foot: the arm turned out.
    // A form without a manipulator, or a two-handed lift asked of one arm, is refused by the plan at once.
    let plan = match inp.mode {
        ReachMode::Front => reach::plan_or_straddle(body, &goal, half[1]),
        ReachMode::Side => reach::plan_side(body, &goal, half[1]),
    };
    let stand = body.stand(inp.cargo_kg);
    if !plan.ok {
        let at = if inp.mode == ReachMode::Side { [plan.object_x, plan.object_y, 0.5 * inp.size_m[1]] } else { [reach::min_object_x(body, half[0]), 0.0, 0.5 * inp.size_m[1]] };
        s.push(body, 0.0, "reach", stand, KinLoads::default(), None, Some(at));
        let why = plan.reason_ko.clone().unwrap_or_default();
        let parts: &[Part] = if body.has_manipulator() { &[Part::Waist, Part::Hand, Part::FootJoints] } else { &[] };
        s.events.push(ActEvent::new(0.0, "reach_fail", why.clone(), parts, Some(plan.lowest_palm_z), false));
        metrics.push(("lowest_palm_z", plan.lowest_palm_z));
        let code = plan.code.unwrap_or(if inp.mode == ReachMode::Side { "reach_side" } else { "reach_ground" });
        return ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::fail(code, why), duration_s: 0.0, metrics, notes };
    }
    metrics.push(("lowest_palm_z", plan.lowest_palm_z));
    // A load held between the feet takes both hands; a reach to the side, the right one.
    let two_arms = match plan.approach {
        "straddle" => true,
        "side" => false,
        _ => two_asked,
    };
    metrics.push(("arms", if two_arms { 2.0 } else { 1.0 }));
    if both_needed && two_arms {
        notes.push(format!("한 손 접촉으로는 {:.0} N까지 버팀 < 안전계수 {GRIP_SAFETY:.0} × 무게 {:.0} N: 두 손으로 잡음", planned.hold_n, planned.weight_n));
    }
    if plan.approach != "front" {
        metrics.push(("stance_out_m", plan.pose.stance_out_m));
        metrics.push(("reach_y_m", plan.object_y));
        metrics.push(("abduction_deg", plan.pose.shoulder_abd_deg[0]));
    }
    let ox = plan.object_x;
    let obj_ground = [ox, plan.object_y, 0.5 * inp.size_m[1]];
    let mu_act = d.lognormal(mu.mean, mu.cov);
    // The hand goes where the robot believes the object is: its memory's error plus the camera's and the arm's.
    let aim = [inp.aim_offset_m[0] + inp.aim_sigma_m * d.normal(), inp.aim_offset_m[1] + inp.aim_sigma_m * d.normal()];
    if inp.memory_sigma_m > 0.0 || inp.aim_offset_m != [0.0; 2] {
        metrics.push(("memory_offset_m", inp.aim_offset_m[0].hypot(inp.aim_offset_m[1])));
        metrics.push(("memory_sigma_m", inp.memory_sigma_m));
    }
    let per_hand_kg = if two_arms { 0.5 * inp.mass_kg } else { inp.mass_kg };
    // The planner's table (`capacity_for`: the fingers, or a gripper's jaws) and what the closed fingers can
    // press (the contact statics). A gripper with a published jaw force squeezes with it (`capacity_for`).
    let cap = capacity_for(body, &working, inp.size_m, per_hand_kg, mu.mean);
    let touching = pads.touching();
    let jaw_force = body.gripper.as_ref().filter(|_| body.gripper_only()).and_then(|g| g.peak_n).is_some() && cap.contacts == 2;
    let squeeze_max = if jaw_force { cap.squeeze_n } else { pads.squeeze_max_n() };
    let preload_n = contact::PRELOAD_N * touching as f32;
    let mut ctl = GripCtl { preload_n, cap_n: squeeze_max, mu_est: mu.mean, react: None };
    // The rise ends in the hold pose. An arm on a mast's carriage holds the
    // object at the height it stows it (`hold_pose_for`): its rise is the
    // carriage's travel to there, not the plan's rise to the top of the mast.
    let hold = hold_pose_for(body, two_arms, inp.cargo_kg);
    let plan_rise_s = if body.mast_arms() {
        let up = kin::solve_fast(body, &KinPose { cargo_kg: inp.cargo_kg, lift_m: hold.lift_m, ..stand }, &KinLoads::holding(inp.mass_kg, two_arms, 0.0));
        reach::move_time(&plan.loaded, &up).max(lift_travel_s(body, &plan.pose, &hold))
    } else {
        plan.rise_s
    };
    // The controller expects the minimum-jerk lift's peak acceleration, 10/√3 · Δz / T².
    let rise_dz = (kin::solve_fast(body, &hold, &KinLoads::default()).hands[0][2] - plan.empty.hands[0][2]).max(0.05);
    let a_plan = MIN_JERK_ACC * rise_dz / (plan_rise_s * plan_rise_s);
    let grip_target = preload_n + GRIP_SAFETY * per_hand_kg * (G + a_plan) / (SQUEEZE_SURFACES * mu.mean.max(0.05));
    let grip_peak = grip_target.min(squeeze_max);
    // A squeeze at its limit: lift slower so the expected load stays within the safety factor.
    let a_room = SQUEEZE_SURFACES * mu.mean * (squeeze_max - preload_n) / (GRIP_SAFETY * per_hand_kg.max(1e-3)) - G;
    let rise_s = if a_room >= a_plan {
        plan_rise_s
    } else {
        // A plan already slower than that (a torso lift's travel) keeps its own pace.
        let slowest = LIFT_SLOWEST_S.max(plan_rise_s);
        let t = if a_room > 1e-3 { (MIN_JERK_ACC * rise_dz / a_room).sqrt() } else { slowest };
        notes.push(format!("목표 쥠 {:.0} N > 손가락 한계 {:.0} N: 한계로 쥐고 {:.1} s에 걸쳐 천천히 듦", grip_target, squeeze_max, t.clamp(plan_rise_s, slowest)));
        t.clamp(plan_rise_s, slowest)
    };
    metrics.push(("lift_s", rise_s));
    // What the robot expects before it tries, from its own model: the palm lands
    // on the top face (the aim error against the half widths) and the pads'
    // friction holds the planned lift's peak load (their scatter).
    let sigma_expect = inp.aim_sigma_m.hypot(inp.memory_sigma_m);
    let hit = |half: f32| if sigma_expect > 1e-6 { super::erf(half / (sigma_expect * std::f32::consts::SQRT_2)) } else { 1.0 };
    let need_mu = per_hand_kg * (G + MIN_JERK_ACC * rise_dz / (rise_s * rise_s)) / (SQUEEZE_SURFACES * grip_peak.max(1e-3));
    let p_success = if touching >= 2 { hit(half[0]) * hit(half[1]) * super::lognormal_above(mu.mean, mu.cov, need_mu) } else { 0.0 };
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

    // 2. Close the fingers (or a gripper's jaws): on the object (each stops where it touches), or on air when the palm missed its top face.
    let w_close = working.fingers.first().map_or(std::f32::consts::TAU, |f| f.joints[0].max_speed);
    let t_close = CLOSE_SWING_RAD / w_close.max(0.1);
    let missed = aim[0].abs() > half[0] || aim[1].abs() > half[1] || touching < 2;
    let held_closure = pads.closed.mean_closure();
    let m_close = (t_close / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m_close {
        let u = i as f32 / m_close as f32;
        s.push(body, t_bottom + i as f32 * SAMPLE_S, "close", KinPose { closure: if missed { u } else { held_closure * u }, ..bottom }, KinLoads::default(), None, Some(obj_ground));
    }
    let t_contact = s.end();
    metrics.push(("aim_err_m", (aim[0] * aim[0] + aim[1] * aim[1]).sqrt()));
    metrics.push(("mu", mu_act));
    metrics.push(("mu_est", mu.mean));
    metrics.push(("squeeze_cap_n", squeeze_max));
    metrics.push(("squeeze_plan_n", cap.squeeze_n));
    metrics.push(("touching", touching as f32));
    if missed {
        let why = if touching < 2 {
            format!("손가락이 {:.2} m 물체를 감싸지 못함 (접촉 {}개)", inp.size_m[0], touching)
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
    s.events.push(ActEvent::new(t_contact, "contact", format!("손가락 {}개 접촉 · 충돌 {:.1} N ({:.1} ms)", touching, f_imp, dt_imp * 1000.0), &[Part::Hand, Part::HandJoints], Some(f_imp), true));

    // 3. Seat the fingers once the sensor confirms the contacts, then pull
    //    until the object leaves the ground; the grip reads the pull
    //    `GRIP_LEAD_S` ahead.
    let delay = contact::detect_delay_s();
    let t_seat = t_contact + delay;
    let t_pull = t_seat + LOOP_RISE_S;
    let t_lift = t_pull + LOOP_RISE_S;
    let weight = per_hand_kg * G;
    let seat_at = |t: f32| ((t - t_seat) / LOOP_RISE_S).clamp(0.0, 1.0);
    let pull_at = |t: f32| weight * ((t - t_pull) / LOOP_RISE_S).clamp(0.0, 1.0);
    // The weight twists the pads about the squeeze axis (the far face's
    // normal, along the fingers) by the centre of mass's offset across the
    // fingers' row; an offset along the squeeze axis tilts the object about
    // an axis in the face, which the pads' pressure and the palm resist, not
    // their torsion. The hand reaches with its fingers forward (est), so the
    // row lies across the body: the aim error's sideways part is the lever.
    let lever = aim[1].abs();
    // Three or more fingers close the object against the palm: the palm on its top face stops a turn about the grasp axis.
    let palm_blocks = touching >= 3;
    let mut pivoted = false;
    let mut slip_min = f32::INFINITY;
    let mut twist_max = 0.0f32;
    let mut stick_min = 1.0f32;
    let mut incipients = 0u32;
    let mut seen_mu: Option<f32> = None;
    let mut failed: Option<(&'static str, String, f32)> = None;
    // The pads at one instant, the grip sample, and whether the controller saw an incipient slip.
    let mut feel = |t: f32, grip_n: f32, load: f32, closure: f32, ctl: &mut GripCtl, s: &mut Series, pivoted: bool| -> (Grip, PadState) {
        // Once it has turned, the centre of mass hangs under the grasp axis: no moment is left.
        let need = if pivoted { 0.0 } else { load * lever };
        let twist = if palm_blocks { 0.0 } else { need };
        let st = pads.state(mu_act, grip_n, load, twist);
        if load > 1e-6 {
            stick_min = stick_min.min(st.finger_stick_min);
            if st.finger_stick_min < contact::STICK_ALARM && !st.shared.gross_slip {
                let seen = mu_seen(&st, touching);
                if ctl.incipient(t, seen) {
                    incipients += 1;
                    seen_mu = Some(seen);
                    let worst = st.contacts.as_slice().iter().min_by(|a, b| a.stick.total_cmp(&b.stick)).map_or(0, |c| c.finger as usize);
                    let name = inp.hand.fingers.get(worst).map_or("?", |f| f.name.as_str());
                    s.events.push(ActEvent::new(t, "incipient_slip", format!("미세 미끄럼: {name} 패드 고착 {:.2} < {:.2} → {:.0} ms 뒤 마찰 {:.2}로 다시 쥠", st.finger_stick_min, contact::STICK_ALARM, delay * 1000.0, seen.max(0.05)), &[Part::Hand, Part::HandJoints], Some(st.finger_stick_min), true));
                }
            }
        }
        let gr = Grip {
            grip_n,
            load_n: load,
            mu: st.mu_mean,
            mu_est: ctl.mu_est,
            slip_ratio: if load > 1e-6 { st.capacity_n / load } else { 99.0 },
            twist_margin_nm: st.torsion_cap_nm - need,
            contacts: touching as u8,
            closure,
            detail: Some(st.contacts),
        };
        (gr, st)
    };
    // A pinch turns rather than lets go when the twist is what breaks the
    // hold: the load alone stays inside the pads' limit surfaces, the load
    // with the weight's moment does not (the slip runs along the surface's
    // normal, mostly a turn, and the turn takes the moment away). Before
    // lift-off the object tips on the ground about the pads as the pull comes on.
    let turns = |g: f32, load: f32, pivoted: bool| {
        !palm_blocks && !pivoted && load > 1e-6 && {
            let free = pads.state(mu_act, g, load, 0.0);
            !free.shared.gross_slip && (load * lever > free.torsion_cap_nm || pads.state(mu_act, g, load, load * lever).shared.gross_slip)
        }
    };
    let tilt = lever.atan2(0.5 * inp.size_m[1]).to_degrees();
    let pivot_label = |g: f32, load: f32| format!("물체가 손 안에서 {tilt:.0}° 돌아 매달림: 무게 모멘트 {:.2} N·m가 패드 한계면을 넘음 (비틀림 마찰 {:.2} N·m)", load * lever, pads.state(mu_act, g, load, 0.0).torsion_cap_nm);
    let m_pre = ((t_lift - t_contact) / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m_pre {
        let t = t_contact + i as f32 * SAMPLE_S;
        let load = pull_at(t);
        let g = ctl.grip(t, pull_at(t + GRIP_LEAD_S), seat_at(t));
        let pivot_now = turns(g, load, pivoted);
        let (gr, st) = feel(t, g, load, held_closure, &mut ctl, &mut s, pivoted || pivot_now);
        if load > 1e-6 {
            slip_min = slip_min.min(gr.slip_ratio);
        }
        if load > 1e-6 && st.shared.gross_slip && failed.is_none() {
            failed = Some(("grasp_slip", format!("들기 전에 미끄러짐: 패드 마찰 {:.2} × 쥠 {:.0} N로 하중 {:.0} N를 못 버팀", st.mu_mean, g, load), t));
        } else if pivot_now {
            pivoted = true;
            s.events.push(ActEvent::new(t, "pivot", pivot_label(g, load), &[Part::Hand, Part::HandJoints], Some(tilt), true));
            metrics.push(("pivot_deg", tilt));
        }
        let phase = if t < t_pull { "squeeze" } else { "load" };
        s.push(body, t, phase, KinPose { closure: held_closure, ..bottom }, KinLoads { hand_force: hand_force(two_arms, [0.0, 0.0, -inp.mass_kg * G * load / weight.max(1e-6)]), ..KinLoads::default() }, Some(gr), Some(obj_ground));
    }
    let t_grid = s.end();
    metrics.push(("lift_off_s", t_lift - t_contact));
    // The fingers seat for the sensor's confirmation and the loop's rise; the pull builds over the arm loop's rise.
    metrics.push(("preload_s", t_pull - t_contact));
    metrics.push(("loading_s", t_lift - t_pull));
    // How far the grip leads the load, measured on the samples: the instant
    // the load reaches half the weight less the instant the grip has made
    // half its rise over the preload. The controller reads the planned pull
    // `GRIP_LEAD_S` ahead, so below the squeeze limit the lead is that
    // constant by design; a squeeze at its limit or a slip reaction moves it.
    let pre: Vec<(f32, f32, f32)> = s.samples.iter().filter(|x| x.t > t_contact && x.t <= t_grid + 1e-6).filter_map(|x| x.grip.map(|g| (x.t, g.grip_n, g.load_n))).collect();
    let gf_end = pre.last().map_or(ctl.preload_n, |p| p.1);
    let t_lf = rise_time(&pre.iter().map(|p| (p.0, p.2)).collect::<Vec<_>>(), 0.5 * weight);
    let t_gf = rise_time(&pre.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>(), ctl.preload_n + 0.5 * (gf_end - ctl.preload_n));
    if let (Some(tl), Some(tg)) = (t_lf, t_gf) {
        metrics.push(("gf_lead_s", tl - tg));
    }
    if failed.is_none() {
        s.events.push(ActEvent::new(t_lift, "lift_off", format!("들림 · 쥠 {:.0} N / 하중 {:.0} N", ctl.grip(t_lift, weight, 1.0), weight), &[Part::Hand, Part::Waist, Part::FootJoints], Some(ctl.grip(t_lift, weight, 1.0) / weight.max(1e-3)), true));
    }

    // 4. Rise with the object, then bring it up to the hold pose; the load is
    //    weight plus inertia. The controller keeps the load hanging under the
    //    shoulder while the body straightens, then curls it to the chest.
    let hold = KinPose { closure: held_closure, ..hold };
    let path = LiftPath::new(body, &bottom, &hold, two_arms, rise_s);
    let lift_s = path.duration();
    metrics.push(("curl_s", path.curl_s));
    let at = |t: f32| path.at(body, t);
    let n = (lift_s / SAMPLE_S).ceil().max(2.0) as usize;
    let dt = lift_s / n as f32;
    let palm = |u: f32| kin::solve_fast(body, &at(u * lift_s), &KinLoads::default()).hands[0];
    // The load over the lift (weight and the hand's acceleration), so the grip can lead it.
    let mut prof: Vec<([f32; 3], [f32; 3], f32)> = Vec::with_capacity(n + 1);
    let mut prev_p: Option<[f32; 3]> = None;
    for i in 1..=n {
        let u = i as f32 / n as f32;
        let p = palm(u);
        let p_next = palm(((i + 1) as f32 / n as f32).min(1.0));
        let p_prev = prev_p.unwrap_or(palm((i - 1) as f32 / n as f32));
        prev_p = Some(p);
        // Across, only one hand's swing in from the side moves the object (two hands mirror each other).
        let ay = if two_arms { 0.0 } else { (p_next[1] - 2.0 * p[1] + p_prev[1]) / (dt * dt) };
        let acc = [(p_next[0] - 2.0 * p[0] + p_prev[0]) / (dt * dt), ay, (p_next[2] - 2.0 * p[2] + p_prev[2]) / (dt * dt)];
        let load = per_hand_kg * ((G + acc[2]).powi(2) + acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
        prof.push((p, acc, load));
    }
    let load_ahead = |i: usize| {
        // The load `GRIP_LEAD_S` after sample i (linear between samples, the last one held).
        let x = i as f32 + GRIP_LEAD_S / dt;
        let k = (x.floor() as usize).min(prof.len() - 1);
        let k1 = (k + 1).min(prof.len() - 1);
        let f = (x - x.floor()).clamp(0.0, 1.0);
        prof[k].2 + (prof[k1].2 - prof[k].2) * f
    };
    let mut dropped_at: Option<[f32; 3]> = None;
    let t0 = t_grid;
    for i in 1..=n {
        if failed.is_some() {
            break;
        }
        let (p, acc, load) = prof[i - 1];
        let t = t0 + i as f32 * dt;
        let g = ctl.grip(t, load_ahead(i - 1), 1.0);
        let pivot_now = turns(g, load, pivoted);
        let (gr, st) = feel(t, g, load, held_closure, &mut ctl, &mut s, pivoted || pivot_now);
        slip_min = slip_min.min(gr.slip_ratio);
        if st.torsion_cap_nm > 1e-6 {
            twist_max = twist_max.max(load * lever / st.torsion_cap_nm);
        }
        let f_obj = [-inp.mass_kg * acc[0], -inp.mass_kg * acc[1], -inp.mass_kg * (G + acc[2])];
        let pose = at(i as f32 / n as f32 * lift_s);
        let ok = s.push(body, t, "lift", KinPose { closure: held_closure, ..pose }, KinLoads { hand_force: hand_force(two_arms, f_obj), ..KinLoads::default() }, Some(gr), Some([p[0], p[1], p[2] - 0.5 * inp.size_m[1] - reach::PALM_CLEARANCE_M]));
        if st.shared.gross_slip || gr.slip_ratio < 1.0 {
            failed = Some(("grasp_slip", format!("들어 올리다 미끄러짐: 패드 마찰 {:.2} × 쥠 {:.0} N로 하중 {:.0} N를 못 버팀 (가속 {:+.1} m/s²)", st.mu_mean, g, load, acc[2]), t));
            dropped_at = Some(p);
        } else if pivot_now {
            // The pinch lets it swing until the centre of mass hangs under the grasp axis.
            pivoted = true;
            s.events.push(ActEvent::new(t, "pivot", pivot_label(g, load), &[Part::Hand, Part::HandJoints], Some(tilt), true));
            metrics.push(("pivot_deg", tilt));
        } else if !ok {
            failed = Some(("lift_overload", format!("들어 올리는 동안 {}", s.limiting.clone().unwrap_or_default()), t));
            dropped_at = Some(p);
        }
    }
    metrics.push(("slip_ratio_min", slip_min));
    metrics.push(("twist_ratio_max", twist_max));
    metrics.push(("stick_min", stick_min));
    metrics.push(("incipient_slips", incipients as f32));
    if let Some(m) = seen_mu {
        metrics.push(("mu_seen", m));
        metrics.push(("react_s", delay + LOOP_RISE_S));
    }
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
    // The hold: the grip on the object's weight, and the pads' answer to it.
    let grip_hold = ctl.grip(f32::INFINITY, weight, 1.0);
    let st_hold = pads.state(mu_act, grip_hold, weight, 0.0);
    metrics.push(("grip_n", grip_hold));
    metrics.push(("gf_lf_hold", grip_hold / weight.max(1e-3)));
    // (GF − the grip at which it would slip) / GF: the pads' capacity over the weight.
    metrics.push(("safety_margin", 1.0 - weight / st_hold.capacity_n.max(1e-6)));
    metrics.push(("hook_share", st_hold.support_n / weight.max(1e-6)));
    metrics.push(("mu_pad", st_hold.mu_mean));
    let shares: Vec<f32> = pads.fingers.iter().map(|f| f.share).collect();
    metrics.push(("finger_share_max", shares.iter().copied().fold(0.0, f32::max)));
    metrics.push(("finger_share_min", shares.iter().copied().fold(1.0, f32::min)));
    metrics.push(("reach_s", plan.descend_s));
    metrics.push(("rise_s", plan_rise_s));
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

    // 5. Stow it in the cargo place (the bay, a tote, a bed), or keep it in the hand without one.
    let Some(bay) = stow_pose(body, two_arms, inp.cargo_kg) else {
        if let Some(c) = body.cargo_geom() {
            // Never quietly: the form has a cargo place its hand cannot get over (`BodyPlan::validate` says so too).
            s.events.push(ActEvent::new(t_hold, "stow_unreachable", format!("손이 {} 위로 가지 못함 (내려놓을 점 {:.2} m): 손에 든 채 운반", c.cargo.label_ko(), c.release[1]), &[Part::Hand], Some(c.release[1]), false));
        }
        notes.push(if body.cargo_geom().is_some() { "적재 위치 위로 손이 가지 못함: 물건을 손(집게)에 든 채 운반" } else { "적재 공간 없음: 물건을 손(집게)에 든 채 운반" }.to_string());
        let dur = s.end();
        metrics.push(("duration_s", dur));
        metrics.push(("peak_ratio", s.peak_ratio));
        metrics.push(("min_margin", s.min_margin));
        return ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::ok(), duration_s: dur, metrics, notes };
    };
    let bay = KinPose { closure: held_closure, ..bay };
    let r_hold = kin::solve_fast(body, &hold, &KinLoads::holding(inp.mass_kg, two_arms, 0.0));
    let r_bay = kin::solve_fast(body, &bay, &KinLoads::holding(inp.mass_kg, two_arms, 0.0));
    // The arm's joints and, on a mast or a torso lift, the carriage's travel to the cargo place.
    let t_stow = reach::move_time(&r_hold, &r_bay).max(lift_travel_s(body, &hold, &bay));
    let n = (t_stow / SAMPLE_S).ceil().max(1.0) as usize;
    let grip_stow = held_grip(&pads, &ctl, mu_act, grip_hold, weight, touching);
    for i in 1..=n {
        let u = min_jerk(i as f32 / n as f32);
        let pose = KinPose::lerp(&hold, &bay, u);
        let p = kin::solve_fast(body, &pose, &KinLoads::default()).hands[0];
        s.push(body, t_hold + i as f32 * SAMPLE_S, "stow", pose, KinLoads::holding(inp.mass_kg, two_arms, 0.0), Some(grip_stow), Some([p[0], p[1], p[2] - 0.5 * inp.size_m[1]]));
    }
    let t_open = s.end();
    // Into the bay from just over its rim; into a tote or onto a bed from the
    // palm's release point down to its floor.
    let (drop_m, place_ko) = match body.cargo_geom() {
        Some(c) if !c.on_pelvis => {
            let h = (c.release[1] - reach::PALM_CLEARANCE_M - inp.size_m[1] - c.floor_z_m).max(0.0);
            metrics.push(("stow_drop_m", h));
            (h, c.cargo.label_ko())
        }
        _ => (BAY_DROP_M, "적재함"),
    };
    let (f_bay, _) = material::impact((2.0 * G * drop_m).sqrt(), inp.mass_kg, material::ground_stiffness(Surface::Firm));
    let m_open = (t_close / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 1..=m_open {
        let u = i as f32 / m_open as f32;
        s.push(body, t_open + i as f32 * SAMPLE_S, "release", KinPose { closure: held_closure * (1.0 - u), ..bay }, KinLoads::default(), None, Some(stowed_at(body)));
    }
    s.events.push(ActEvent::new(s.end(), "stowed", format!("{place_ko}에 놓음 · {:.0} N", f_bay), &[Part::Hand], Some(f_bay), true));
    let dur = s.end();
    metrics.push(("duration_s", dur));
    metrics.push(("peak_ratio", s.peak_ratio));
    metrics.push(("min_margin", s.min_margin));
    ActRollout { kind: ActKind::Grasp, samples: s.samples, events: s.events, outcome: Outcome::ok(), duration_s: dur, metrics, notes }
}

/// The first instant a sampled curve `(t, v)` rises through `level`, linear between samples.
fn rise_time(pts: &[(f32, f32)], level: f32) -> Option<f32> {
    pts.windows(2).find(|w| w[0].1 < level && w[1].1 >= level).map(|w| {
        let u = (level - w[0].1) / (w[1].1 - w[0].1).max(1e-9);
        w[0].0 + u * (w[1].0 - w[0].0)
    })
}

/// The grip of a hand holding an object still: the controller's grip on its
/// weight and the pads' answer (no twist).
fn held_grip(pads: &HandPads, ctl: &GripCtl, mu_obj: f32, grip_n: f32, load_n: f32, touching: usize) -> Grip {
    let st = pads.state(mu_obj, grip_n, load_n, 0.0);
    Grip { grip_n, load_n, mu: st.mu_mean, mu_est: ctl.mu_est, slip_ratio: if load_n > 1e-6 { st.capacity_n / load_n } else { 99.0 }, twist_margin_nm: st.torsion_cap_nm, contacts: touching as u8, closure: pads.closed.mean_closure(), detail: Some(st.contacts) }
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

/// The hand a body's preset builds (`NotoConfig::preset` for its finger
/// count and base), when the caller has no hand of its own.
fn preset_hand(body: &Body) -> Option<Hand> {
    let mut cfg = crate::hand::NotoConfig::preset(body.fingers);
    cfg.base = body.base;
    cfg.build().ok()
}

/// Plays a release out: from the bay down to the ground or a pile, and open
/// ([`release_by`] with the body's preset hand, which `ReleaseInput` does
/// not carry; a world whose hand is edited away from its preset draws the
/// closure of the preset's fingers).
pub fn release(inp: &ReleaseInput) -> ActRollout {
    match preset_hand(inp.body) {
        Some(h) => release_by(inp, &h),
        None => release_closed(inp, 1.0),
    }
}

/// Plays a release out with `hand`: it holds the object at the closure its
/// fingers stopped at on it (`sim::top_grasp`) until it opens.
pub fn release_by(inp: &ReleaseInput, hand: &Hand) -> ActRollout {
    release_closed(inp, top_grasp(hand, inp.size_m).mean_closure())
}

fn release_closed(inp: &ReleaseInput, closed: f32) -> ActRollout {
    let body = inp.body;
    let mut d = Draw::new(inp.seed);
    let mut s = Series::new();
    let mut metrics: Vec<(&'static str, f32)> = Vec::new();
    let notes = Vec::new();
    let stand = body.stand(inp.cargo_kg);
    if !body.has_manipulator() {
        s.push(body, 0.0, "take", stand, KinLoads::default(), None, Some(stowed_at(body)));
        let why = "조작기(팔·집게)가 없어 물건을 내려놓을 수 없음";
        s.events.push(ActEvent::new(0.0, "release_fail", why, &[], None, false));
        return ActRollout { kind: ActKind::Release, samples: s.samples, events: s.events, outcome: Outcome::fail("no_manipulator", why), duration_s: 0.0, metrics, notes };
    }
    // From the cargo place (or the hand that holds it, without one).
    let bay = stow_pose(body, false, inp.cargo_kg).unwrap_or_else(|| hold_pose_for(body, false, inp.cargo_kg));
    // Take it out of the bay.
    let r_stand = kin::solve_fast(body, &stand, &KinLoads::default());
    let r_bay = kin::solve_fast(body, &bay, &KinLoads::default());
    let t_take = reach::move_time(&r_stand, &r_bay).max(lift_travel_s(body, &stand, &bay));
    let n = (t_take / SAMPLE_S).ceil().max(1.0) as usize;
    for i in 0..=n {
        let u = min_jerk(i as f32 / n as f32);
        s.push(body, i as f32 * SAMPLE_S, "take", KinPose { closure: closed * u, ..KinPose::lerp(&stand, &bay, u) }, KinLoads::default(), None, Some(stowed_at(body)));
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
    let bottom = if plan.ok { KinPose { closure: closed, ..plan.pose } } else { KinPose { closure: closed, ..grasp_hold_low(body, inp.cargo_kg) } };
    let held = KinLoads::holding(inp.mass_kg, false, 0.0);
    let r_bottom = kin::solve_fast(body, &bottom, &held);
    // The lift backwards: the arm brings the object from the bay over the
    // spot, then it goes straight down while the body bends.
    // A mast's carriage takes its own time down (the torso lift's travel is
    // not timed here yet, which keeps today's release of a lifted torso).
    let lower_s = reach::move_time(&r_bay, &r_bottom);
    let lower_s = if body.mast_arms() { lower_s.max(lift_travel_s(body, &bay, &bottom)) } else { lower_s };
    let path = LiftPath::new(body, &bottom, &KinPose { closure: closed, ..bay }, false, lower_s);
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
        s.push(body, t_open + i as f32 * SAMPLE_S, "open", KinPose { closure: closed * (1.0 - u), ..bottom }, KinLoads::default(), None, Some([land[0], land[1], inp.surface_z_m + 0.5 * inp.size_m[1]]));
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

/// The lowest a body without any bend holds an object: the arm straight down
/// in front (an arm on a deck, which cannot hang, keeps its rest).
fn grasp_hold_low(body: &Body, cargo_kg: f32) -> KinPose {
    if body.deck_arms() {
        return body.stand(cargo_kg);
    }
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

    /// The 2.4 kg log with the hands the planner picks for it (`top_grasp_ok`).
    fn log_input<'a>(b: &'a Body, h: &'a Hand, seed: u64) -> GraspInput<'a> {
        let (_, one) = top_grasp_ok(h, [0.15; 3], 2.4, material::pad_on(ItemMaterial::Wood).mean);
        GraspInput { two_arms: !one, ..input(b, h, seed, 0.0) }
    }

    #[test]
    fn a_legged_robot_picks_a_log_with_a_full_time_series() {
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Five);
        // One hand's fingers, pressing what their joints allow on the log's one face, do not hold it
        // with the safety factor: the planner takes both hands (as v0.9's table did for this log).
        let inp = log_input(&b, &h, 3);
        assert!(inp.two_arms);
        let r = grasp(&inp);
        assert!(r.outcome.ok, "{:?} {:?}", r.outcome, r.notes);
        for e in ["bend_bottom", "contact", "lift_off", "hold", "stowed"] {
            assert!(r.events.iter().any(|x| x.kind == e), "missing {e}: {:?}", r.events.iter().map(|x| x.kind).collect::<Vec<_>>());
        }
        assert!(r.duration_s > 2.5 && r.duration_s < 8.0, "{}", r.duration_s);
        assert!(r.samples.windows(2).all(|w| w[1].t > w[0].t));
        // Above the slip limit: each hand's grip times the friction carries its share of the weight.
        let gf = r.metric("gf_lf_hold").unwrap();
        assert!(gf > 1.0 / r.metric("mu").unwrap(), "squeeze above the slip limit: {gf}");
        assert!(r.metric("slip_ratio_min").unwrap() > 1.0 && r.metric("safety_margin").unwrap() > 0.0);
        // Every sample of the hold carries the pads' detail, and the fingers touch without sinking in.
        let held: Vec<&Grip> = r.samples.iter().filter_map(|s| s.grip.as_ref()).collect();
        assert!(!held.is_empty() && held.iter().all(|g| g.detail.is_some_and(|d| d.n as u8 == g.contacts)));
        let pads = HandPads::new(&h, [0.15; 3], PadSurface::Dry, 3);
        assert!((r.samples.iter().find(|s| s.phase == "lift").unwrap().pose.closure - pads.closed.mean_closure()).abs() < 1e-6, "the hand holds at the contact-stopped closure");
    }

    #[test]
    fn the_grip_follows_the_load_and_an_incipient_slip_raises_it() {
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let r = grasp(&log_input(&b, &h, 3));
        // The controller reads the planned pull `GRIP_LEAD_S` ahead (a set lead, not the whole squeeze;
        // measured on the samples). A light box leaves the squeeze below its limit: the lead is that constant.
        let light = grasp(&GraspInput::front(&b, &h, ItemMaterial::Cardboard, 0.33, [0.10; 3], Surface::Firm, false, 0.0, 0.0, 3));
        let lead = light.metric("gf_lead_s").unwrap();
        assert!((lead - GRIP_LEAD_S).abs() < 0.002, "{lead}");
        // The 2.4 kg log asks for more than the fingers press: the grip reaches its (capped) half rise
        // sooner, and still leads by less than the whole preload.
        let lead = r.metric("gf_lead_s").unwrap();
        assert!(lead >= GRIP_LEAD_S - 0.002 && lead < r.metric("preload_s").unwrap(), "{lead}");
        assert!(r.metric("gf_lf_corr").unwrap() > 0.8, "{:?}", r.metric("gf_lf_corr"));
        // The pads: the finger pads alone share the load (one face); a lighter-pressed pad sticks less.
        let pads = HandPads::new(&h, [0.10; 3], PadSurface::Dry, 1);
        let st = pads.state(0.6, 20.0, 8.0, 0.0);
        assert_eq!(st.shared.pads.len(), pads.touching());
        let held: f32 = st.shared.pads.iter().map(|p| p.ft).sum();
        assert!((held - 8.0).abs() < 1e-3 && !st.shared.gross_slip, "{held}");
        assert!(st.capacity_n > 8.0 && st.capacity_n < 0.6 * 20.0 * 1.3, "{}", st.capacity_n);
        // The planner reads the same contacts: what holds at the squeeze limit is the fingers' force times the friction.
        assert!((top_hold_n(&pads.closed, 0.6) - 0.6 * pads.squeeze_max_n()).abs() < 1e-3);
        // A wet object on a light grip: an incipient slip is felt, the controller reads a lower friction and grips harder.
        let mut ctl = GripCtl { preload_n: 4.0, cap_n: 60.0, mu_est: 0.6, react: None };
        let before = ctl.grip(0.0, 10.0, 1.0);
        let wet = HandPads::new(&h, [0.10; 3], PadSurface::Wet, 1);
        let st = wet.state(0.35, before, 10.0, 0.0);
        assert!(st.finger_stick_min < contact::STICK_ALARM && !st.shared.gross_slip, "{:?}", st.finger_stick_min);
        let seen = mu_seen(&st, wet.touching());
        assert!(seen < 0.6 && ctl.incipient(0.0, seen));
        let after = ctl.grip(1.0, 10.0, 1.0);
        assert!(after > before * 1.2, "{before} → {after}");
        assert!(wet.state(0.35, after, 10.0, 0.0).finger_stick_min > st.finger_stick_min);
    }

    #[test]
    fn a_pinch_off_centre_turns_in_the_hand_instead_of_letting_go() {
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Two);
        let mut inp = GraspInput::front(&b, &h, ItemMaterial::Wood, 0.6, [0.10; 3], Surface::Firm, false, 0.0, 0.0, 3);
        inp.aim_offset_m = [0.0, 0.03];
        let r = grasp(&inp);
        // Two pads cannot hold the weight's moment about a 3 cm lever across the fingers' row: the box swings
        // to hang under the grasp axis and stays held.
        assert!(r.outcome.ok, "{:?} {:?}", r.outcome, r.notes);
        assert!(r.events.iter().any(|e| e.kind == "pivot") && r.metric("pivot_deg").unwrap() > 10.0, "{:?}", r.events.iter().map(|e| e.kind).collect::<Vec<_>>());
        // Centred, the same pinch holds it straight.
        inp.aim_offset_m = [0.0; 2];
        let r = grasp(&inp);
        assert!(r.outcome.ok && !r.events.iter().any(|e| e.kind == "pivot"), "{:?}", r.outcome);
        // Off along the squeeze axis (forward, the way the fingers point), the weight tilts the box about
        // an axis in the gripped face, not about the pads' normal: no twist, no turn.
        inp.aim_offset_m = [0.03, 0.0];
        let r = grasp(&inp);
        assert!(r.outcome.ok && !r.events.iter().any(|e| e.kind == "pivot"), "{:?}", r.outcome);
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
        // The hand holds the object where its fingers stopped on it until it opens.
        let (_, hand) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let held = top_grasp(&hand, [0.1; 3]).mean_closure();
        assert!(held < 0.95 && a.samples.iter().filter(|s| s.phase == "lower").all(|s| (s.pose.closure - held).abs() < 1e-5), "{held}");
        assert!(a.samples.iter().all(|s| s.pose.closure <= held + 1e-5));
        assert!(a.metric("drop_height_m").unwrap() < 0.05, "{:?}", a.metric("drop_height_m"));
        assert!(b.metric("drop_height_m").unwrap() > 0.3, "{:?}", b.metric("drop_height_m"));
        assert!(b.metric("impact_n").unwrap() > a.metric("impact_n").unwrap());
    }

    /// The world's items (`world-sim` `ItemKind::spec`: log, stone, coal, iron ore, parcel) as material, kg, size.
    const WORLD_ITEMS: [(ItemMaterial, f32, [f32; 3]); 5] = [(ItemMaterial::Wood, 2.4, [0.15; 3]), (ItemMaterial::Stone, 2.6, [0.10; 3]), (ItemMaterial::Coal, 1.4, [0.10; 3]), (ItemMaterial::IronOre, 4.0, [0.10; 3]), (ItemMaterial::Cardboard, 0.7, [0.16, 0.10, 0.12])];

    #[test]
    fn the_planner_takes_two_hands_or_refuses_on_the_rollouts_contacts() {
        let (_, five) = setup(BaseKind::HumanFoot, FingerCount::Five);
        let (_, three) = setup(BaseKind::HumanFoot, FingerCount::Three);
        let wood = material::pad_on(ItemMaterial::Wood).mean;
        // One hand: the same answer as `top_grasp_ok`, and its margin is the hold over the safety factor's need.
        let parcel = top_hands(&five, [0.16, 0.10, 0.12], 0.7, wood);
        assert!(parcel.hands == 1 && parcel.safe(), "{parcel:?}");
        assert!(top_grasp_ok(&five, [0.16, 0.10, 0.12], 0.7, wood).1);
        let pads = HandPads::new(&five, [0.15; 3], PadSurface::Dry, 1);
        let log = top_hands(&five, [0.15; 3], 2.4, wood);
        assert!((log.hold_n - top_hold_n(&pads.closed, wood)).abs() < 1e-4 && (log.margin(1) - log.hold_n / (GRIP_SAFETY * 2.4 * G)).abs() < 1e-6);
        // The 2.4 kg log: one five-finger hand does not hold it with the factor; two hold its weight,
        // just below the factor (the rollout lifts slowly at the squeeze limit).
        assert_eq!(log.hands, 2, "{log:?}");
        assert!(!top_grasp_ok(&five, [0.15; 3], 2.4, wood).1 && log.margin(2) < 1.0 && log.margin(2) > 0.9 && !log.safe(), "{log:?}");
        // Three fingers (the thumb meets nothing): two hands do not hold even the log's weight. Refused.
        let weak = top_hands(&three, [0.15; 3], 2.4, wood);
        assert_eq!(weak.touching, 2);
        assert!(weak.hands == 0 && 2.0 * weak.hold_n < weak.weight_n, "{weak:?}");
    }

    #[test]
    fn a_one_hand_request_its_contacts_do_not_hold_takes_both_hands() {
        // The world's v0.9 table gives a four-finger hand the 2.6 kg stone in one hand; its contacts do not hold it so.
        let (b, h) = setup(BaseKind::HumanFoot, FingerCount::Four);
        let stone = material::pad_on(ItemMaterial::Stone).mean;
        let cap = capacity(&h, [0.10; 3], 2.6, stone);
        assert!(stone * cap.squeeze_n >= GRIP_SAFETY * 2.6 * G, "the v0.9 table holds it in one hand: {cap:?}");
        assert_eq!(top_hands(&h, [0.10; 3], 2.6, stone).hands, 2);
        let r = grasp(&GraspInput::front(&b, &h, ItemMaterial::Stone, 2.6, [0.10; 3], Surface::Gravel, false, 0.0, 0.0, 4));
        assert!(r.outcome.ok && r.metric("arms") == Some(2.0), "{:?} {:?}", r.outcome, r.notes);
        assert!(r.notes.iter().any(|n| n.contains("두 손으로 잡음")), "{:?}", r.notes);
        // A parcel one hand holds stays in one hand.
        let p = grasp(&GraspInput::front(&b, &h, ItemMaterial::Cardboard, 0.7, [0.16, 0.10, 0.12], Surface::Gravel, false, 0.0, 0.0, 4));
        assert!(p.outcome.ok && p.metric("arms") == Some(1.0), "{:?}", p.outcome);
    }

    /// The planner and the rollout read the same contacts: a pick it sends with the full safety factor
    /// never slips, one below the factor rarely, and one it refuses would slide out of both hands.
    #[test]
    fn the_planner_sends_only_picks_the_rollout_holds() {
        const SEEDS: u64 = 20;
        for fingers in FingerCount::ALL {
            let (b, h) = setup(BaseKind::HumanFoot, fingers);
            for (mat, kg, size) in WORLD_ITEMS {
                let th = top_hands(&h, size, kg, material::pad_on(mat).mean);
                // The hold's own failure (the object slides out), not the body's (reach, joint limits).
                let codes = |two: bool| (0..SEEDS).map(|seed| grasp(&GraspInput::front(&b, &h, mat, kg, size, Surface::Gravel, two, 0.0, 0.0, seed)).outcome.code).collect::<Vec<_>>();
                let slips = |c: &[Option<&str>]| c.iter().filter(|c| **c == Some("grasp_slip")).count() as u64;
                let what = format!("{fingers:?} {mat:?} {kg} kg: {th:?}");
                match th.hands {
                    1 => assert_eq!(slips(&codes(false)), 0, "{what}"),
                    2 if th.safe() => assert_eq!(slips(&codes(true)), 0, "{what}"),
                    2 => assert!(slips(&codes(true)) * 5 <= SEEDS, "{what}: {:?}", codes(true)),
                    _ => {
                        let c = codes(true);
                        let tried = c.iter().filter(|c| !matches!(c, Some("reach_ground") | Some("lift_overload"))).count() as u64;
                        assert!(tried == 0 || slips(&c) * 3 >= tried, "{what}: refused, yet two hands hold: {c:?}");
                    }
                }
            }
        }
    }

    /// The world's grasp table (`capacity`, v0.9) against the rollout's own planner (`top_hands`): the hands each
    /// picks and how often they hold (in brackets the failures that are not a slip: reach, joint limits). A
    /// one-hand pick the contacts do not hold plays with both hands (`GraspInput::two_arms`), as in the world.
    #[test]
    #[ignore]
    fn print_world_table_vs_rollout() {
        for fingers in FingerCount::ALL {
            let (b, h) = setup(BaseKind::HumanFoot, fingers);
            for (mat, kg, size) in WORLD_ITEMS {
                let mu = material::pad_on(mat).mean;
                let cap = capacity(&h, size, kg, mu);
                let margin = mu * cap.squeeze_n / (kg * G * GRIP_SAFETY);
                let world = if cap.contacts >= 2 && margin >= 1.0 { Some(false) } else if 2.0 * margin >= 1.0 { Some(true) } else { None };
                let th = top_hands(&h, size, kg, mu);
                let ok = |two: bool| {
                    let runs: Vec<ActRollout> = (0..100u64).map(|seed| grasp(&GraspInput::front(&b, &h, mat, kg, size, Surface::Gravel, two, 0.0, 0.0, seed))).collect();
                    let other = runs.iter().filter(|r| !r.outcome.ok && r.outcome.code != Some("grasp_slip")).count();
                    format!("{}% [{other}]", runs.iter().filter(|r| r.outcome.ok).count())
                };
                let w = world.map_or("거부".to_string(), |two| format!("{} {}", if two { "두 손" } else if th.hands != 1 { "한 손(실행은 두 손)" } else { "한 손" }, ok(two)));
                let t = match th.hands {
                    1 => format!("한 손 {}", ok(false)),
                    2 => format!("두 손 (여유 {:.2}) {}", th.margin(2), ok(true)),
                    _ => format!("거부 (두 손 {:.2}×무게, 두 손이면 {})", 2.0 * th.hold_n / th.weight_n.max(1e-6), ok(true)),
                };
                println!("[WORLD] {fingers:?} {mat:?} {kg} kg: 월드 표 여유 {margin:.2} → {w} · top_hands 여유 {:.2} → {t}", th.margin(1));
            }
        }
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
