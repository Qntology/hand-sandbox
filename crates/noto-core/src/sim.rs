//! Hand simulation: rate-limited joint motion for animation, and a quasi-static
//! grasp model that closes every finger until it touches a logistics object.
//!
//! The grasp model is kinematic, not a rigid-body physics engine:
//!
//! 1. The object rests on the palm, centred over it at `GraspSetup::z_offset`.
//! 2. Each finger closes in small steps. A step is rejected for a finger when
//!    any of its sample points would enter the object (signed distance below
//!    the contact margin), so the finger stops at the first contact.
//! 3. The palm is a contact too when the object's footprint overlaps it.
//! 4. Contact normals point from the object centre to the contact point, the
//!    convex-object approximation. Force closure (normals not all in one
//!    half-space) is reported as a diagnostic.
//! 5. Contact force is the knuckle torque divided by the lever arm to the
//!    contact point. Friction at every contact (palm included, with the palm
//!    squeeze equal to the finger forces) must exceed `mass * g * safety`.
//! 6. The object counts as supported when the palm touches it or it is
//!    force-closed. The grasp is stable when it is supported, at least two
//!    fingers touch it and the friction capacity covers the weight.

use serde::{Deserialize, Serialize};

use crate::finger::{sample_points, FingerSpec};
use crate::hand::Hand;
use crate::joint::{Actuation, JointAngles};
use crate::math::Vec3;

pub const GRAVITY_MPS2: f32 = 9.81;

/// Number of closing steps when a finger is driven to full closure.
const CLOSE_STEPS: usize = 100;
/// Smallest lever arm used for force estimates, metres.
const MIN_LEVER_M: f32 = 0.02;
/// Directions sampled when testing force closure.
const FORCE_CLOSURE_DIRS: usize = 512;

/// A rectangular logistics object: carton, tote, bag, etc.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Payload {
    pub name: String,
    /// Width x height x depth, metres.
    pub size_m: [f32; 3],
    pub mass_kg: f32,
    /// Coulomb friction coefficient between finger pad and surface.
    pub friction: f32,
}

/// Where the object sits on the palm, and how strict the test is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspSetup {
    /// Position of the object's centre along the palm, metres. Its bottom
    /// face always rests on the palm plane (y = 0).
    pub z_offset: f32,
    /// Distance at which a sample point counts as touching, metres.
    pub contact_margin_m: f32,
    /// Required friction margin over the object weight.
    pub safety_factor: f32,
}

impl Default for GraspSetup {
    fn default() -> Self {
        GraspSetup {
            z_offset: 0.03,
            contact_margin_m: 0.004,
            safety_factor: 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ContactSource {
    Finger { index: usize },
    Palm,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub source: ContactSource,
    /// Contact point in the hand frame.
    pub point: Vec3,
    /// Unit normal pointing from the object centre to the contact point.
    pub normal: Vec3,
    pub force_n: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraspType {
    None,
    /// Two or fewer fingers in contact.
    Pinch,
    /// Three or more fingers in contact.
    Power,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspResult {
    pub contacts: Vec<Contact>,
    pub grasp_type: GraspType,
    /// True when the contact normals cannot all lie in one half-space.
    pub force_closed: bool,
    /// Friction force the contacts can resist, N.
    pub friction_capacity_n: f32,
    /// Weight times safety factor, N.
    pub required_n: f32,
    pub stable: bool,
    /// Heaviest object this hand can hold with the configured safety factor.
    pub max_payload_kg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspOutcome {
    pub result: GraspResult,
    /// Joint angles of every finger at the moment it stopped.
    pub angles: Vec<Vec<JointAngles>>,
}

/// Joint angles that a closure value in `[0, 1]` commands for one finger.
pub fn closure_angles(finger: &FingerSpec, actuation: Actuation, closure: f32) -> Vec<JointAngles> {
    let closure = closure.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(finger.joint_count());
    let mut prev_flex = 0.0;
    for (i, joint) in finger.joints.iter().enumerate() {
        let target = match actuation {
            Actuation::Direct => closure * joint.flex_max,
            Actuation::Coupled { ratio } => {
                if i == 0 {
                    closure * joint.flex_max
                } else {
                    ratio * prev_flex
                }
            }
        };
        let flex = joint.clamp_flex(target);
        out.push(JointAngles {
            flex,
            ..Default::default()
        });
        prev_flex = flex;
    }
    out
}

/// Joint points of every finger for the given angles, hand frame.
pub fn fk_all(hand: &Hand, angles: &[Vec<JointAngles>]) -> Vec<Vec<Vec3>> {
    hand.fingers
        .iter()
        .zip(angles)
        .map(|(f, a)| f.points(a))
        .collect()
}

/// Signed distance from `p` to an axis-aligned box centred at `c` with half
/// sizes `h`. Negative inside the box.
fn box_sdf(p: Vec3, c: Vec3, h: Vec3) -> f32 {
    let d = p - c;
    let q = Vec3::new(d.x.abs() - h.x, d.y.abs() - h.y, d.z.abs() - h.z);
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y).max(q.z).min(0.0);
    outside + inside
}

/// Unit direction from the object centre to `p`: the normal of a convex
/// object at that point.
fn radial_normal(p: Vec3, center: Vec3) -> Vec3 {
    (p - center).normalized()
}

/// True when no direction sees all normals on the same side, i.e. the
/// normals positively span space and can resist any load. Checked by sampling
/// directions on a Fibonacci sphere.
fn force_closed(normals: &[Vec3]) -> bool {
    if normals.len() < 2 {
        return false;
    }
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    for i in 0..FORCE_CLOSURE_DIRS {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / FORCE_CLOSURE_DIRS as f32;
        let r = (1.0 - y * y).sqrt();
        let theta = golden_angle * i as f32;
        let d = Vec3::new(theta.cos() * r, y, theta.sin() * r);
        if normals.iter().all(|n| n.dot(d) > 1e-3) {
            return false;
        }
    }
    true
}

/// Closes every finger on the object and reports whether it can be held.
pub fn grasp(hand: &Hand, payload: &Payload, setup: &GraspSetup) -> GraspOutcome {
    let actuation = hand.config.joints.actuation;
    let half = Vec3::new(
        payload.size_m[0] * 0.5,
        payload.size_m[1] * 0.5,
        payload.size_m[2] * 0.5,
    );
    // The object rests on the palm, so its bottom face sits on the palm plane.
    let center = Vec3::new(0.0, half.y, setup.z_offset);
    let margin = setup.contact_margin_m;

    let mut angles: Vec<Vec<JointAngles>> = hand
        .fingers
        .iter()
        .map(|f| vec![JointAngles::default(); f.joint_count()])
        .collect();
    let mut contacts: Vec<Contact> = Vec::new();
    let mut blocked = vec![false; hand.fingers.len()];

    for step in 1..=CLOSE_STEPS {
        let closure = step as f32 / CLOSE_STEPS as f32;
        for (fi, finger) in hand.fingers.iter().enumerate() {
            if blocked[fi] {
                continue;
            }
            let candidate = closure_angles(finger, actuation, closure);
            let points = finger.points(&candidate);
            // Points at palm level belong to the palm, not to the finger.
            let touch = sample_points(&points)
                .into_iter()
                .filter(|p| p.y >= margin)
                .map(|p| (p, box_sdf(p, center, half)))
                .filter(|(_, d)| *d <= margin)
                .min_by(|a, b| a.1.total_cmp(&b.1));

            match touch {
                Some((p, _)) => {
                    blocked[fi] = true;
                    let lever = p.distance(finger.mount).max(MIN_LEVER_M);
                    contacts.push(Contact {
                        source: ContactSource::Finger { index: fi },
                        point: p,
                        normal: radial_normal(p, center),
                        force_n: finger.joints[0].max_torque / lever,
                    });
                }
                None => angles[fi] = candidate,
            }
        }
        if blocked.iter().all(|&b| b) {
            break;
        }
    }

    // The palm supports the object when its footprint overlaps the palm. The
    // palm squeeze balances the finger forces.
    let footprint_on_palm = center.x - half.x <= hand.palm.width * 0.5
        && center.x + half.x >= -hand.palm.width * 0.5
        && center.z - half.z <= 0.0
        && center.z + half.z >= -hand.palm.length;
    if footprint_on_palm {
        let palm_point = Vec3::new(
            center.x,
            0.0,
            center.z.clamp(-hand.palm.length, 0.0),
        );
        let squeeze_n: f32 = contacts.iter().map(|c| c.force_n).sum();
        contacts.push(Contact {
            source: ContactSource::Palm,
            point: palm_point,
            normal: radial_normal(palm_point, center),
            force_n: squeeze_n,
        });
    }

    let result = score(&contacts, payload, setup);
    GraspOutcome { result, angles }
}

fn score(contacts: &[Contact], payload: &Payload, setup: &GraspSetup) -> GraspResult {
    let normals: Vec<Vec3> = contacts.iter().map(|c| c.normal).collect();
    let closed = force_closed(&normals);
    let palm_support = contacts
        .iter()
        .any(|c| matches!(c.source, ContactSource::Palm));
    let supported = palm_support || closed;
    let finger_contacts = contacts
        .iter()
        .filter(|c| matches!(c.source, ContactSource::Finger { .. }))
        .count();
    let friction_capacity_n: f32 = contacts
        .iter()
        .map(|c| c.force_n * payload.friction)
        .sum();
    let required_n = payload.mass_kg * GRAVITY_MPS2 * setup.safety_factor;
    let holdable = supported && finger_contacts >= 2;
    let max_payload_kg = if holdable {
        friction_capacity_n / (GRAVITY_MPS2 * setup.safety_factor)
    } else {
        0.0
    };
    let grasp_type = match finger_contacts {
        0 => GraspType::None,
        1 | 2 => GraspType::Pinch,
        _ => GraspType::Power,
    };
    GraspResult {
        contacts: contacts.to_vec(),
        grasp_type,
        force_closed: closed,
        friction_capacity_n,
        required_n,
        stable: holdable && friction_capacity_n >= required_n,
        max_payload_kg,
    }
}

/// Animated hand: each finger's closure command is tracked by rate-limited
/// joints, so the motion respects each joint's speed and limits.
#[derive(Clone, Debug)]
pub struct HandSim {
    hand: Hand,
    closure: Vec<f32>,
    angles: Vec<Vec<JointAngles>>,
}

impl HandSim {
    pub fn new(hand: Hand) -> Self {
        let n = hand.fingers.len();
        let angles = hand
            .fingers
            .iter()
            .map(|f| vec![JointAngles::default(); f.joint_count()])
            .collect();
        HandSim {
            hand,
            closure: vec![0.0; n],
            angles,
        }
    }

    pub fn hand(&self) -> &Hand {
        &self.hand
    }

    /// Sets the closure command per finger, each clamped to `[0, 1]`.
    /// Entries beyond the finger count are ignored.
    pub fn set_closure(&mut self, closure: &[f32]) {
        for (dst, src) in self.closure.iter_mut().zip(closure) {
            *dst = src.clamp(0.0, 1.0);
        }
    }

    pub fn closure(&self) -> &[f32] {
        &self.closure
    }

    /// Advances joint angles by `dt` seconds toward the commanded closure.
    pub fn step(&mut self, dt: f32) {
        let actuation = self.hand.config.joints.actuation;
        for ((finger, angles), &c) in self
            .hand
            .fingers
            .iter()
            .zip(self.angles.iter_mut())
            .zip(&self.closure)
        {
            let targets = closure_angles(finger, actuation, c);
            for ((joint, a), t) in finger.joints.iter().zip(angles.iter_mut()).zip(&targets) {
                let max_delta = joint.max_speed * dt;
                let delta = (t.flex - a.flex).clamp(-max_delta, max_delta);
                a.flex = joint.clamp_flex(a.flex + delta);
            }
        }
    }

    pub fn angles(&self) -> &[Vec<JointAngles>] {
        &self.angles
    }

    pub fn points(&self) -> Vec<Vec<Vec3>> {
        fk_all(&self.hand, &self.angles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::{FingerCount, NotoConfig};

    fn box_payload(mass_kg: f32) -> Payload {
        Payload {
            name: "test carton".into(),
            size_m: [0.08, 0.06, 0.06],
            mass_kg,
            friction: 0.6,
        }
    }

    fn build(count: FingerCount) -> Hand {
        NotoConfig::preset(count).build().unwrap()
    }

    #[test]
    fn box_sdf_is_negative_inside_and_positive_outside() {
        let h = Vec3::new(0.5, 0.5, 0.5);
        assert!(box_sdf(Vec3::ZERO, Vec3::ZERO, h) < 0.0);
        assert!((box_sdf(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, h) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn coupled_distal_joints_follow_ratio() {
        let hand = build(FingerCount::Five);
        let f = &hand.fingers[1];
        let a = closure_angles(f, Actuation::Coupled { ratio: 0.5 }, 1.0);
        let expected = 0.5 * a[0].flex;
        assert!((a[1].flex - expected).abs() < 1e-6);
    }

    #[test]
    fn closure_angles_respect_joint_limits() {
        let hand = build(FingerCount::Four);
        for f in &hand.fingers {
            for c in [0.0, 0.5, 1.0] {
                for (j, a) in f
                    .joints
                    .iter()
                    .zip(closure_angles(f, Actuation::Direct, c))
                {
                    assert!(a.flex >= j.flex_min && a.flex <= j.flex_max + 1e-6);
                }
            }
        }
    }

    #[test]
    fn five_finger_hand_holds_a_light_carton() {
        let hand = build(FingerCount::Five);
        let out = grasp(&hand, &box_payload(0.5), &GraspSetup::default());
        let r = &out.result;
        assert!(r.contacts.len() >= 2, "contacts: {}", r.contacts.len());
        assert!(r.stable, "capacity {} N vs required {} N", r.friction_capacity_n, r.required_n);
    }

    #[test]
    fn heavy_object_exceeds_capacity() {
        let hand = build(FingerCount::Two);
        let out = grasp(&hand, &box_payload(50.0), &GraspSetup::default());
        assert!(!out.result.stable);
        assert!(out.result.max_payload_kg < 50.0);
    }

    #[test]
    fn grasp_outcome_angles_stay_within_limits() {
        let hand = build(FingerCount::Three);
        let out = grasp(&hand, &box_payload(0.5), &GraspSetup::default());
        for (f, a) in hand.fingers.iter().zip(&out.angles) {
            for (j, x) in f.joints.iter().zip(a) {
                assert!(x.flex >= j.flex_min - 1e-6 && x.flex <= j.flex_max + 1e-6);
            }
        }
    }

    #[test]
    fn hand_sim_moves_toward_command_without_overshoot() {
        let mut sim = HandSim::new(build(FingerCount::Five));
        sim.set_closure(&[1.0; 5]);
        sim.step(0.01);
        let finger = sim.hand().fingers[0].clone();
        let actuation = sim.hand().config.joints.actuation;
        let first = sim.angles()[0][0].flex;
        assert!(first > 0.0 && first <= finger.joints[0].max_speed * 0.01 + 1e-6);
        for _ in 0..1000 {
            sim.step(0.01);
        }
        let target = closure_angles(&finger, actuation, 1.0);
        assert!((sim.angles()[0][0].flex - target[0].flex).abs() < 1e-5);
    }
}
