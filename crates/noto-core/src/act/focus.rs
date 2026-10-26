//! What the eight focus views show at one instant: the hand's shape and its
//! finger joints, each foot's sole and the leg joints, the waist, the neck,
//! the wheels and the tracks. Built from one solved instant (`kin`) and what
//! the action knew then (grip, object, slip), so a keyframe can be drawn part
//! by part long after the world has moved on.

use serde::Serialize;

use crate::balance::Body;
use crate::hand::Hand;
use crate::joint::Actuation;
use crate::kin::{KinJoint, KinReport};
use crate::sim::closure_angles;

use super::drive::DriveState;
use super::{ActEvent, ActKind, ActSample, Grip, Part};

/// A joint for a focus view.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointView {
    pub name: String,
    pub angle_deg: Option<f32>,
    pub torque_nm: f32,
    pub peak_nm: f32,
    pub ratio: f32,
    pub effort: f32,
    pub in_range: bool,
    pub structural: bool,
    pub actuator: String,
}

impl JointView {
    fn of(j: &KinJoint) -> JointView {
        JointView { name: j.name.clone(), angle_deg: j.angle_deg, torque_nm: j.torque.abs(), peak_nm: j.peak, ratio: j.ratio, effort: j.effort, in_range: j.in_range, structural: j.structural, actuator: j.actuator.clone() }
    }
}

/// One finger joint.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FingerJointView {
    pub finger: String,
    pub joint: usize,
    pub angle_deg: f32,
    pub torque_nm: f32,
    pub max_nm: f32,
}

/// The working hand: finger polylines in the hand frame (`sim`: z along the
/// palm, y toward the palm side), the object, the grip.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HandView {
    pub closure: f32,
    pub fingers: Vec<Vec<[f32; 3]>>,
    pub thumbs: Vec<bool>,
    pub palm: [f32; 2],
    pub grip: Option<Grip>,
    /// The object's half sizes in the hand frame (`None` when the hand is empty).
    pub object_half: Option<[f32; 3]>,
    pub object_kg: Option<f32>,
    /// Palm centre in the base frame and its height above the ground.
    pub palm_at: [f32; 3],
    /// Force on the hand from the object or the world, N (base frame).
    pub force_n: [f32; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FootView {
    pub side: &'static str,
    pub in_air: bool,
    pub normal_n: f32,
    pub shear_n: f32,
    /// Centre of pressure and the sole `[x0, x1, y0, y1]` in the base frame.
    pub cop: [f32; 2],
    pub rect: [f32; 4],
    /// Shear over normal at this foot, and the ground's friction under it.
    pub rcof: f32,
    pub mu: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WheelView {
    pub axle: String,
    pub normal_n: f32,
    pub shear_n: f32,
    pub slip: f32,
    pub mu: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrackView {
    pub pressure_kpa: [f32; 2],
    pub loaded_len_m: f32,
    pub normal_n: f32,
    pub shear_n: f32,
    pub slip: f32,
    pub mu: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NeckView {
    pub joint: JointView,
    /// Camera pitch below horizontal, degrees (trunk plus neck).
    pub camera_pitch_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BodyView {
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    pub support: [f32; 4],
    pub margin: f32,
    pub trunk_deg: f32,
    pub mass_kg: f32,
    pub grf: [f32; 3],
    pub feasible: bool,
    pub limiting: Option<String>,
}

/// Every focus view of one instant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FocusFrame {
    pub t: f32,
    pub action: Option<&'static str>,
    pub action_ko: Option<&'static str>,
    pub phase: Option<&'static str>,
    pub event: Option<String>,
    pub event_ok: Option<bool>,
    /// The parts this instant is about (an event's, or the action's).
    pub parts: Vec<&'static str>,
    pub hand: HandView,
    pub hand_joints: Vec<FingerJointView>,
    pub foot: Vec<FootView>,
    pub foot_joints: Vec<JointView>,
    pub waist: Option<JointView>,
    pub neck: Option<NeckView>,
    pub wheel: Vec<WheelView>,
    pub track: Option<TrackView>,
    pub body: BodyView,
}

/// What the focus frame is built from.
pub struct FocusCtx<'a> {
    pub body: &'a Body,
    pub hand: &'a Hand,
    pub report: &'a KinReport,
    pub sample: Option<&'a ActSample>,
    pub action: Option<ActKind>,
    pub event: Option<&'a ActEvent>,
    /// Friction under the base (feet, wheels or tracks).
    pub ground_mu: f32,
    pub drive: Option<DriveState>,
    /// The handled object's half sizes and mass, when there is one.
    pub object: Option<([f32; 3], f32)>,
}

/// The parts an action moves, for the views to open on.
pub fn parts_of(kind: ActKind, legged: bool) -> Vec<Part> {
    let base = if legged { vec![Part::Foot, Part::FootJoints] } else { vec![Part::Wheel, Part::Track] };
    let mut v = match kind {
        ActKind::Grasp | ActKind::Release => vec![Part::Hand, Part::HandJoints, Part::Waist, Part::Neck],
        ActKind::Bend => vec![Part::Waist, Part::Neck, Part::Hand],
        ActKind::Push => vec![Part::Wheel, Part::Track],
        ActKind::Walk => vec![Part::Foot, Part::FootJoints, Part::Waist, Part::Neck],
        ActKind::Drive => vec![Part::Wheel, Part::Track, Part::Neck],
    };
    for p in base {
        if !v.contains(&p) {
            v.push(p);
        }
    }
    v
}

/// Builds the focus frame of one instant.
pub fn frame(ctx: &FocusCtx) -> FocusFrame {
    let r = ctx.report;
    let body = ctx.body;
    let closure = ctx.sample.map_or(r.pose.closure, |s| s.pose.closure);
    let grip = ctx.sample.and_then(|s| s.grip);
    let force = ctx.sample.map_or([0.0; 3], |s| s.loads.hand_force[0]);
    // Fingers in the hand frame at this closure.
    let act = ctx.hand.config.joints.actuation;
    let mut fingers = Vec::new();
    let mut thumbs = Vec::new();
    let mut hand_joints = Vec::new();
    let share = grip.map_or(0.0, |g| (g.grip_n / ctx.hand.fingers.len().max(1) as f32).max(0.0));
    for f in &ctx.hand.fingers {
        let angles = closure_angles(f, act, closure.clamp(0.0, 1.0));
        fingers.push(f.points(&angles).iter().map(|p| [p.x, p.y, p.z]).collect());
        thumbs.push(f.is_thumb);
        // The joints hold the finger's share of the squeeze at their lever (the contact near the tip).
        let lever_tip: f32 = f.segments.iter().sum::<f32>() * 0.8;
        for (k, (j, a)) in f.joints.iter().zip(&angles).enumerate() {
            let lever = (lever_tip - f.segments.iter().take(k).sum::<f32>()).max(0.005);
            let coupled = matches!(act, Actuation::Coupled { .. }) && k > 0;
            let tau = if grip.is_some() { (share * lever).min(j.max_torque) * if coupled { 0.7 } else { 1.0 } } else { 0.0 };
            hand_joints.push(FingerJointView { finger: f.name.clone(), joint: k, angle_deg: a.flex.to_degrees(), torque_nm: tau, max_nm: j.max_torque });
        }
    }
    let hand = HandView {
        closure,
        fingers,
        thumbs,
        palm: [ctx.hand.palm.width, ctx.hand.palm.length],
        grip,
        object_half: ctx.object.map(|o| o.0),
        object_kg: ctx.object.map(|o| o.1),
        palm_at: r.hands[0],
        force_n: force,
    };
    let foot = r
        .contacts
        .iter()
        .filter(|c| c.name.starts_with("foot"))
        .map(|c| FootView { side: if c.name.ends_with('r') { "r" } else { "l" }, in_air: c.normal_n <= 1e-3, normal_n: c.normal_n, shear_n: c.shear_n, cop: c.cop, rect: c.rect, rcof: if c.normal_n > 1e-3 { (c.shear_n / c.normal_n).abs() } else { 0.0 }, mu: ctx.ground_mu })
        .collect();
    let foot_joints = r.joints.iter().filter(|j| j.name.starts_with("hip_") || j.name.starts_with("knee") || j.name.starts_with("ankle")).map(JointView::of).collect();
    let slip = ctx.drive.map_or(0.0, |d| d.slip);
    let wheel = r.contacts.iter().filter(|c| c.name.starts_with("axle")).map(|c| WheelView { axle: c.name.clone(), normal_n: c.normal_n, shear_n: c.shear_n, slip, mu: ctx.ground_mu }).collect();
    let track = r.contact("track").map(|c| TrackView { pressure_kpa: c.pressure_kpa.unwrap_or([0.0; 2]), loaded_len_m: c.loaded_len_m.unwrap_or(0.0), normal_n: c.normal_n, shear_n: c.shear_n, slip, mu: ctx.ground_mu });
    let neck = r.joint("neck").map(|j| NeckView { joint: JointView::of(j), camera_pitch_deg: r.pose.trunk_deg() + r.pose.neck_deg });
    let parts = match (ctx.event, ctx.action) {
        (Some(e), _) if !e.parts.is_empty() => e.parts.iter().map(|p| p.id()).collect(),
        (_, Some(k)) => parts_of(k, body.legged).iter().map(|p| p.id()).collect(),
        _ => vec![],
    };
    FocusFrame {
        t: ctx.sample.map_or(0.0, |s| s.t),
        action: ctx.action.map(|k| k.id()),
        action_ko: ctx.action.map(|k| k.label_ko()),
        phase: ctx.sample.map(|s| s.phase),
        event: ctx.event.map(|e| e.label_ko.clone()),
        event_ok: ctx.event.map(|e| e.ok),
        parts,
        hand,
        hand_joints,
        foot,
        foot_joints,
        waist: r.joint("waist_pitch").map(JointView::of),
        neck,
        wheel,
        track,
        body: BodyView { com: r.com, zmp: r.zmp, support: r.support, margin: r.margin, trunk_deg: r.pose.trunk_deg(), mass_kg: r.mass_kg, grf: r.grf, feasible: r.feasible, limiting: r.limiting.clone() },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::{FingerCount, NotoConfig};
    use crate::kin::{self, KinLoads, KinPose};
    use crate::parts::BodyPlan;

    #[test]
    fn a_frame_has_every_view_for_the_base_it_has() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track] {
            let body = Body::new(&BodyPlan::default(), base, FingerCount::Five);
            let mut cfg = NotoConfig::preset(FingerCount::Five);
            cfg.base = base;
            let hand = cfg.build().unwrap();
            let r = kin::solve(&body, &KinPose { closure: 0.6, ..KinPose::stand(0.0) }, &KinLoads::default());
            let f = frame(&FocusCtx { body: &body, hand: &hand, report: &r, sample: None, action: Some(ActKind::Grasp), event: None, ground_mu: 0.6, drive: None, object: None });
            assert_eq!(f.hand.fingers.len(), 5);
            assert!(f.hand_joints.len() >= 10);
            assert!(f.neck.is_some() && f.waist.is_some());
            match base {
                BaseKind::HumanFoot => assert!(f.foot.len() == 2 && f.foot_joints.len() == 8 && f.wheel.is_empty()),
                BaseKind::Track => assert!(f.track.is_some() && f.foot.is_empty()),
                _ => assert!(f.wheel.len() == 2 && f.track.is_none()),
            }
            assert!(f.parts.contains(&"hand"));
            serde_json::to_string(&f).unwrap();
        }
    }
}
