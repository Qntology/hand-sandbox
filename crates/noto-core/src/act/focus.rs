use serde::Serialize;

use crate::balance::Body;
use crate::contact::{self, FingerContact, SoleContact, SoleModel};
use crate::hand::Hand;
use crate::joint::Actuation;
use crate::kin::{KinJoint, KinReport};
use crate::math::Vec3;
use crate::morph::Morphology;
use crate::sim::{closure_angles, finger_joint_torques, top_grasp};

use super::drive::DriveState;
use super::{ActEvent, ActKind, ActSample, Grip, Part};

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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FingerJointView {
    pub finger: String,
    pub joint: usize,
    pub angle_deg: f32,
    pub torque_nm: f32,
    pub max_nm: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HandView {
    pub closure: f32,
    pub fingers: Vec<Vec<[f32; 3]>>,
    pub thumbs: Vec<bool>,
    pub palm: [f32; 2],
    pub grip: Option<Grip>,
    pub object_half: Option<[f32; 3]>,
    pub object_kg: Option<f32>,
    pub palm_at: [f32; 3],
    pub force_n: [f32; 3],
    pub contacts: Vec<FingerContact>,
    pub object_center: Option<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FootView {
    pub side: &'static str,
    pub in_air: bool,
    pub normal_n: f32,
    pub shear_n: f32,
    pub cop: [f32; 2],
    pub rect: [f32; 4],
    pub rcof: f32,
    pub mu: f32,
    pub mu_mean: f32,
    pub shear_y_n: f32,
    pub toe_deg: f32,
    pub heel_off: bool,
    pub cone_use: f32,
    pub slipping: bool,
    pub sole: Option<SoleContact>,
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FocusFrame {
    pub t: f32,
    pub action: Option<&'static str>,
    pub action_ko: Option<&'static str>,
    pub phase: Option<&'static str>,
    pub event: Option<String>,
    pub event_ok: Option<bool>,
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

pub struct FocusCtx<'a> {
    pub body: &'a Body,
    pub hand: &'a Hand,
    pub report: &'a KinReport,
    pub sample: Option<&'a ActSample>,
    pub action: Option<ActKind>,
    pub event: Option<&'a ActEvent>,
    pub ground_mu: f32,
    pub drive: Option<DriveState>,
    pub object: Option<([f32; 3], f32)>,
    pub step_mu: Option<f32>,
}

pub fn parts_of(kind: ActKind, morph: &Morphology) -> Vec<Part> {
    let base = if morph.legged() { vec![Part::Foot, Part::FootJoints] } else { vec![Part::Wheel, Part::Track] };
    let mut v = match kind {
        ActKind::Grasp | ActKind::Release => vec![Part::Hand, Part::HandJoints, Part::Waist, Part::Neck],
        ActKind::Bend => vec![Part::Waist, Part::Neck, Part::Hand],
        ActKind::Look => vec![Part::Neck, Part::Waist],
        ActKind::Push => vec![Part::Wheel, Part::Track],
        ActKind::Walk => vec![Part::Foot, Part::FootJoints, Part::Waist, Part::Neck],
        ActKind::Drive => vec![Part::Wheel, Part::Track, Part::Neck],
    };
    for p in base {
        if !v.contains(&p) {
            v.push(p);
        }
    }
    v.retain(|p| match p {
        Part::Hand | Part::HandJoints => morph.arm_count() > 0,
        Part::Waist => morph.has_torso(),
        Part::Neck => morph.has_neck(),
        _ => true,
    });
    v
}

pub fn frame(ctx: &FocusCtx) -> FocusFrame {
    let r = ctx.report;
    let body = ctx.body;
    let closure = ctx.sample.map_or(r.pose.closure, |s| s.pose.closure);
    let grip = ctx.sample.and_then(|s| s.grip);
    let force = ctx.sample.map_or([0.0; 3], |s| s.loads.hand_force[0]);
    let working = body.working_hand(ctx.hand);
    let act = working.config.joints.actuation;
    let closed = ctx.object.filter(|_| body.has_manipulator()).map(|(h, _)| top_grasp(&working, [2.0 * h[0], 2.0 * h[1], 2.0 * h[2]]));
    let contacts: Vec<FingerContact> = grip.and_then(|g| g.detail).map_or(Vec::new(), |d| d.as_slice().to_vec());
    let mut fingers = Vec::new();
    let mut thumbs = Vec::new();
    let mut hand_joints = Vec::new();
    let share = grip.map_or(0.0, |g| (g.grip_n / working.fingers.len().max(1) as f32).max(0.0));
    let hand_fingers = if body.has_manipulator() { &working.fingers[..] } else { &[] };
    for (fi, f) in hand_fingers.iter().enumerate() {
        let angles = match &closed {
            Some(tg) if grip.is_some() => tg.angles[fi].clone(),
            Some(tg) => closure_angles(f, act, closure.clamp(0.0, 1.0).min(tg.closure[fi].max(0.0))),
            None => closure_angles(f, act, closure.clamp(0.0, 1.0)),
        };
        fingers.push(f.points(&angles).iter().map(|p| [p.x, p.y, p.z]).collect());
        thumbs.push(f.is_thumb);
        let pad = contacts.iter().find(|c| c.finger as usize == fi);
        let at = closed.as_ref().and_then(|tg| tg.contacts.iter().filter(|c| c.finger == fi).max_by(|a, b| a.max_n.total_cmp(&b.max_n)));
        let statics: Option<Vec<f32>> = match (pad, at) {
            (Some(p), Some(c)) => {
                let n = Vec3::new(p.normal[0], p.normal[1], p.normal[2]);
                let ft = Vec3::new(p.ft[0], p.ft[1], p.ft[2]);
                Some(finger_joint_torques(f, &angles, c.phalanx, c.along, n.scale(p.fn_n) - ft))
            }
            _ => None,
        };
        let lever_tip: f32 = f.segments.iter().sum::<f32>() * 0.8;
        for (k, (j, a)) in f.joints.iter().zip(&angles).enumerate() {
            let tau = match (&statics, grip) {
                (Some(t), _) => t[k].abs(),
                (None, Some(g)) if g.detail.is_some() => 0.0,
                (None, Some(_)) => {
                    let lever = (lever_tip - f.segments.iter().take(k).sum::<f32>()).max(0.005);
                    let coupled = matches!(act, Actuation::Coupled { .. }) && k > 0;
                    (share * lever).min(j.max_torque) * if coupled { 0.7 } else { 1.0 }
                }
                (None, None) => 0.0,
            };
            hand_joints.push(FingerJointView { finger: f.name.clone(), joint: k, angle_deg: a.flex.to_degrees(), torque_nm: tau, max_nm: j.max_torque });
        }
    }
    let hand = HandView {
        closure,
        fingers,
        thumbs,
        palm: [working.palm.width, working.palm.length],
        grip,
        object_half: ctx.object.map(|o| o.0),
        object_kg: ctx.object.map(|o| o.1),
        palm_at: r.hands[0],
        force_n: force,
        contacts,
        object_center: closed.as_ref().map(|tg| [tg.center.x, tg.center.y, tg.center.z]),
    };
    let mu_now = ctx.step_mu.unwrap_or(ctx.ground_mu);
    let sole_model = body.legged.then(|| SoleModel::of(body));
    let foot = r
        .contacts
        .iter()
        .filter(|c| c.name.starts_with("foot"))
        .map(|c| {
            let s = if c.name.ends_with('r') { 0 } else { 1 };
            let toe = r.pose.toe_deg[s];
            let sole = sole_model.as_ref().filter(|_| c.normal_n > 1e-3).map(|m| contact::sole_contact(m, r.pose.feet[s][0], c.normal_n, [c.shear_n, c.shear_y_n], c.torsion_nm, c.cop[0], toe, mu_now));
            FootView {
                side: if s == 0 { "r" } else { "l" },
                in_air: c.normal_n <= 1e-3,
                normal_n: c.normal_n,
                shear_n: c.shear_n,
                cop: c.cop,
                rect: c.rect,
                rcof: if c.normal_n > 1e-3 { (c.shear_n / c.normal_n).abs() } else { 0.0 },
                mu: mu_now,
                mu_mean: ctx.ground_mu,
                shear_y_n: c.shear_y_n,
                toe_deg: sole.map_or(0.0, |x| x.toe_deg),
                heel_off: sole.is_some_and(|x| x.rocker),
                cone_use: sole.map_or(0.0, |x| x.cone_use),
                slipping: sole.is_some_and(|x| x.patches.iter().any(|p| p.slipping)),
                sole,
            }
        })
        .collect();
    let foot_joints = r.joints.iter().filter(|j| j.name.starts_with("hip_") || j.name.starts_with("knee") || j.name.starts_with("ankle")).map(JointView::of).collect();
    let slip = ctx.drive.map_or(0.0, |d| d.slip);
    let wheel = r.contacts.iter().filter(|c| c.name.starts_with("axle")).map(|c| WheelView { axle: c.name.clone(), normal_n: c.normal_n, shear_n: c.shear_n, slip, mu: ctx.ground_mu }).collect();
    let track = r.contact("track").map(|c| TrackView { pressure_kpa: c.pressure_kpa.unwrap_or([0.0; 2]), loaded_len_m: c.loaded_len_m.unwrap_or(0.0), normal_n: c.normal_n, shear_n: c.shear_n, slip, mu: ctx.ground_mu });
    let neck = r.joint("neck").map(|j| NeckView { joint: JointView::of(j), camera_pitch_deg: r.pose.trunk_deg() + r.pose.neck_deg });
    let parts = match (ctx.event, ctx.action) {
        (Some(e), _) if !e.parts.is_empty() => e.parts.iter().map(|p| p.id()).collect(),
        (_, Some(k)) => parts_of(k, &body.morph).iter().map(|p| p.id()).collect(),
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
            let f = frame(&FocusCtx { body: &body, hand: &hand, report: &r, sample: None, action: Some(ActKind::Grasp), event: None, ground_mu: 0.6, drive: None, object: None, step_mu: None });
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

    #[test]
    fn a_held_object_shows_its_pad_contacts_and_a_walking_foot_its_drawn_friction() {
        use crate::act::grasp::{grasp, GraspInput};
        use crate::act::material::ItemMaterial;
        use crate::surface::Surface;
        let body = Body::new(&BodyPlan::default(), BaseKind::HumanFoot, FingerCount::Five);
        let mut cfg = NotoConfig::preset(FingerCount::Five);
        cfg.base = BaseKind::HumanFoot;
        let hand = cfg.build().unwrap();
        let size = [0.15f32; 3];
        let ro = grasp(&GraspInput::front(&body, &hand, ItemMaterial::Wood, 2.4, size, Surface::Gravel, true, 0.0, 0.0, 3));
        let s = ro.samples.iter().find(|s| s.phase == "lift").unwrap();
        let r = kin::solve(&body, &s.pose, &s.loads);
        let f = frame(&FocusCtx { body: &body, hand: &hand, report: &r, sample: Some(s), action: Some(ActKind::Grasp), event: None, ground_mu: 0.6, drive: None, object: Some(([0.075; 3], 2.4)), step_mu: None });
        let g = s.grip.unwrap();
        assert_eq!(f.hand.contacts.len(), g.contacts as usize);
        assert!(f.hand.contacts.iter().all(|c| c.fn_n > 0.0 && c.cone_use < 1.0 && c.stick > 0.0), "{:?}", f.hand.contacts);
        assert!(f.hand.object_center.is_some());
        let touched: Vec<String> = f.hand.contacts.iter().map(|c| hand.fingers[c.finger as usize].name.clone()).collect();
        assert!(f.hand_joints.iter().filter(|j| touched.contains(&j.finger) && j.joint == 0).all(|j| j.torque_nm > 0.0));
        let walk = crate::act::gait::walk(&body, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 4, 1).0;
        let st = walk.samples.iter().find(|x| x.phase == "single_support").unwrap();
        let r = kin::solve(&body, &st.pose, &st.loads);
        let f = frame(&FocusCtx { body: &body, hand: &hand, report: &r, sample: Some(st), action: Some(ActKind::Walk), event: None, ground_mu: 0.725, drive: None, object: None, step_mu: Some(0.61) });
        let foot = f.foot.iter().find(|x| !x.in_air).unwrap();
        assert!(foot.mu == 0.61 && foot.mu_mean == 0.725, "{foot:?}");
        let sole = foot.sole.unwrap();
        assert!((sole.patches.iter().map(|p| p.normal_n).sum::<f32>() - foot.normal_n).abs() < 0.01 * foot.normal_n.max(1.0), "{sole:?}");
    }
}
