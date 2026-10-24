//! Whole-body balance and joint effort of a robot built from part presets.
//!
//! The robot is a set of point masses (head, chest, battery, belly, cargo bay,
//! two arms with hands and loads, two legs and feet or a chassis). For one
//! posture and terrain the model works out, quasi-statically:
//!
//! * the centre of mass and the zero-moment point (the CoM shifted by the
//!   acceleration: `x_zmp = x_com - z_com · a / g`),
//! * the stability margin: how far the ZMP is from the edge of the support
//!   polygon, as a fraction of the half-length (1 at the centre, 0 at the
//!   edge, negative outside, where a person would have to step: Hof 2005),
//! * the torque at every joint and its effort (torque over the actuator's
//!   continuous and peak rating),
//! * what the other segments did to compensate (the N:N coupling).
//!
//! The controller follows the human strategies (Horak & Nashner 1986): a small
//! CoM shift is held by the ankles (or simply tolerated by a wheeled base), a
//! larger one is pulled back by leaning the trunk at the waist, as far as the
//! belly preset can lean; a lifted foot needs the pelvis over the stance foot.
//! A feed-forward term (the robot's anticipatory postural adjustment) predicts
//! the disturbance before it happens; without it the reaction comes one
//! electromechanical delay late and the margin dips. The SDS learns how wrong
//! the feed-forward's prediction is and scales it (see [`FeedForward`]).
//!
//! References (human, made dimensionless) are in [`reference`] with sources in
//! `docs/research/balance.md`. Simplifications: quasi-static, sagittal plus a
//! lateral check, no foot rotation, no joint-angle limits other than the trunk
//! lean, the camera stays at the renderer's eye height.

use serde::{Deserialize, Serialize};

use crate::base::{BaseKind, BaseSpec};
use crate::hand::FingerCount;
use crate::parts::*;

pub const G: f32 = 9.81;
/// Delay of a purely reactive correction, seconds (electromechanical delay 80–100 ms, research §3).
pub const REACTION_DELAY_S: f32 = 0.10;
/// Margin the controller and the speed limit keep (research §2: 10–20% of the half-length, design choice).
pub const MARGIN_KEEP: f32 = 0.2;
/// Knee lever while walking, metres (research §5 sizing rule, d ≈ 0.085–0.09 m).
pub const KNEE_LEVER_WALK_M: f32 = 0.085;
/// Knee lever while standing still (est).
pub const KNEE_LEVER_STAND_M: f32 = 0.03;
/// Knee peak torque must exceed the need by this factor (research §5 sizing rule).
pub const KNEE_SAFETY: f32 = 1.5;
/// Lateral distance from the pelvis centre to a hip joint, metres (est, 0.191·H/2 scaled to 1.2 m).
pub const HIP_HALF_WIDTH_M: f32 = 0.09;
/// Foot sole friction (est, rubber on dirt) for the slip limit on slopes.
pub const FOOT_FRICTION: f32 = 0.6;
/// Rolling resistance of the chassis wheels (est).
pub const ROLLING_RESISTANCE: f32 = 0.03;
/// Mass of the mast that carries the upper body on a wheeled chassis, kg (est: 0.5 m of 2020 extrusion plus mounts).
pub const MAST_KG: f32 = 1.0;
/// Where the cargo bay sits: in front of the belly, metres (est).
pub const CARGO_X_M: f32 = 0.12;
pub const CARGO_Z_M: f32 = WAIST_M - 0.10;
/// Grip safety factor, as in `sim::GraspSetup`.
pub const GRIP_SAFETY: f32 = 2.0;
/// Distance a wheeled base must be able to stop in, metres (one cell).
pub const STOP_DISTANCE_M: f32 = 0.5;

/// What disturbs the balance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Quiet,
    /// Working arm raised forward to strike (mining).
    ArmRaise,
    /// Reaching to lift an item into the cargo bay.
    ArmLift,
    /// One foot off the ground while walking.
    FootLift,
    /// Climbing a curb or step.
    StepUp,
    /// Standing or driving on a slope.
    Slope,
    /// Cargo in the bay in front of the belly.
    CarryFront,
    /// Starting or braking.
    Accel,
}

impl Source {
    pub const ALL: [Source; 8] = [Source::Quiet, Source::ArmRaise, Source::ArmLift, Source::FootLift, Source::StepUp, Source::Slope, Source::CarryFront, Source::Accel];

    pub fn id(self) -> &'static str {
        match self {
            Source::Quiet => "quiet",
            Source::ArmRaise => "arm_raise",
            Source::ArmLift => "arm_lift",
            Source::FootLift => "foot_lift",
            Source::StepUp => "step_up",
            Source::Slope => "slope",
            Source::CarryFront => "carry_front",
            Source::Accel => "accel",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Source::Quiet => "정지 자세",
            Source::ArmRaise => "팔 들기 (채취)",
            Source::ArmLift => "팔로 들어올리기",
            Source::FootLift => "발 들기 (보행)",
            Source::StepUp => "턱 오르기",
            Source::Slope => "경사",
            Source::CarryFront => "앞쪽 적재",
            Source::Accel => "가감속",
        }
    }

    /// The segment whose action this is.
    pub fn segment(self) -> Segment {
        match self {
            Source::ArmRaise | Source::ArmLift => Segment::Arm,
            Source::FootLift | Source::Slope => Segment::Foot,
            Source::StepUp | Source::Accel => Segment::Leg,
            Source::CarryFront | Source::Quiet => Segment::Belly,
        }
    }

    /// How long the disturbance takes to build up, seconds (research §3–4:
    /// arm raise ~0.45 s, weight shift 0.4–0.8 s; the rest est).
    pub fn duration_s(self) -> f32 {
        match self {
            Source::Quiet => 10.0,
            Source::ArmRaise => 0.45,
            Source::ArmLift => 1.0,
            Source::FootLift => 0.6,
            Source::StepUp => 0.6,
            Source::Slope => 1.0,
            Source::CarryFront => 2.0,
            Source::Accel => 0.5,
        }
    }

    pub fn from_id(id: &str) -> Option<Source> {
        Source::ALL.into_iter().find(|s| s.id() == id)
    }
}

/// Posture and situation for one evaluation. Arm 0 is the working (right)
/// arm, arm 1 the other one.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Posture {
    pub source: Source,
    /// Upper-arm elevation from hanging (0°) to forward horizontal (90°).
    pub arm_elev_deg: [f32; 2],
    /// Elbow flexion: forearm angle from the upper-arm line, degrees.
    pub elbow_deg: [f32; 2],
    pub hand_load_kg: [f32; 2],
    pub cargo_kg: f32,
    /// Ground slope along the heading, degrees, positive uphill.
    pub slope_deg: f32,
    /// Step being climbed, metres.
    pub step_m: f32,
    /// Walking with one foot off the ground (legged bases only).
    pub single_support: bool,
    /// Forward acceleration, m/s².
    pub accel_mps2: f32,
}

impl Posture {
    pub fn quiet(cargo_kg: f32) -> Posture {
        Posture {
            source: if cargo_kg > 0.0 { Source::CarryFront } else { Source::Quiet },
            arm_elev_deg: [0.0, 0.0],
            elbow_deg: [10.0, 10.0],
            hand_load_kg: [0.0, 0.0],
            cargo_kg,
            slope_deg: 0.0,
            step_m: 0.0,
            single_support: false,
            accel_mps2: 0.0,
        }
    }

    /// Working arm raised to strike (75° elevation, est for hitting a block at chest height).
    pub fn arm_raise(cargo_kg: f32) -> Posture {
        Posture { source: Source::ArmRaise, arm_elev_deg: [75.0, 0.0], elbow_deg: [15.0, 10.0], ..Posture::quiet(cargo_kg) }
    }

    /// Lifting `load_kg` into the bay: upper arm 30° forward, forearm horizontal
    /// (the worst instant for the elbow). `two_arms` shares the load.
    pub fn arm_lift(load_kg: f32, two_arms: bool, cargo_kg: f32) -> Posture {
        let (l0, l1, e1) = if two_arms { (load_kg / 2.0, load_kg / 2.0, 30.0) } else { (load_kg, 0.0, 0.0) };
        Posture { source: Source::ArmLift, arm_elev_deg: [30.0, e1], elbow_deg: [60.0, if two_arms { 60.0 } else { 10.0 }], hand_load_kg: [l0, l1], ..Posture::quiet(cargo_kg) }
    }
}

/// The feed-forward (anticipatory) compensation for one source.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedForward {
    pub enabled: bool,
    /// Learned scale of the nominal prediction (SDS calibration measured / predicted; 1 untrained).
    pub gain: f32,
}

impl Default for FeedForward {
    fn default() -> Self {
        FeedForward { enabled: true, gain: 1.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointLoad {
    /// `shoulder_r`, `elbow_l`, `waist`, `neck`, `knee`, `hip_roll`, `ankle`, `drive`, `grip_r`.
    pub joint: String,
    pub segment: Segment,
    pub actuator: String,
    pub torque_nm: f32,
    pub cont_nm: f32,
    pub peak_nm: f32,
    /// Torque over continuous rating (heats above 1).
    pub effort: f32,
    /// Torque over peak rating (cannot do it above 1).
    pub peak_ratio: f32,
}

/// One N:N cell: what `target` did when `source` acted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Coupling {
    pub source: Source,
    pub target: Segment,
    /// Axis name, see [`reference`].
    pub axis: String,
    pub value: f32,
    pub unit: String,
    /// Same without the feed-forward.
    pub value_noff: f32,
    /// Effort (peak ratio) of the target's joint in this state, if it has one.
    pub effort: Option<f32>,
    pub effort_noff: Option<f32>,
    pub torque_nm: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BalanceReport {
    pub source: Source,
    pub mass_kg: f32,
    /// Load in the hands, cargo in the bay and ground slope of the posture.
    pub hand_load_kg: f32,
    pub cargo_kg: f32,
    pub slope_deg: f32,
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    /// Support polygon `[x0, x1, y0, y1]` (horizontal projection).
    pub support: [f32; 4],
    pub margin_x: f32,
    pub margin_y: f32,
    /// Smaller of the two, after the transient of this source.
    pub margin: f32,
    /// The same if the reaction had no feed-forward.
    pub margin_noff: f32,
    /// CoM shift the source caused before compensation, forward, metres.
    pub shift_uncomp_m: f32,
    /// Share of that shift the compensation took back.
    pub restore_frac: f32,
    pub trunk_lean_deg: f32,
    /// Pelvis displacement toward the stance foot, metres.
    pub lateral_shift_m: f32,
    /// How far that moved the whole-body CoM sideways, metres.
    pub lateral_com_shift_m: f32,
    pub head_pitch_deg: f32,
    pub head_roll_deg: f32,
    pub joints: Vec<JointLoad>,
    /// Σ (τ / τ_peak)² over the joints (research §7 effort metric).
    pub effort_sum: f32,
    pub feasible: bool,
    pub limiting: Option<String>,
    /// Feed-forward prediction of the shift and the gain used.
    pub ff_predicted_m: f32,
    pub ff_gain: f32,
    pub couplings: Vec<Coupling>,
}

impl BalanceReport {
    /// The context the references of this report's cells depend on.
    pub fn ref_ctx(&self, body_mass_kg: f32) -> RefCtx {
        let m = body_mass_kg.max(1.0);
        match self.source {
            Source::CarryFront => RefCtx { load_ratio: self.cargo_kg / m, load_dist_m: CARGO_X_M, slope_deg: self.slope_deg },
            _ => RefCtx { load_ratio: self.hand_load_kg / m, load_dist_m: 0.0, slope_deg: self.slope_deg },
        }
    }

    pub fn joint(&self, name: &str) -> Option<&JointLoad> {
        self.joints.iter().find(|j| j.joint == name)
    }

    pub fn worst(&self) -> Option<&JointLoad> {
        self.joints.iter().max_by(|a, b| a.peak_ratio.total_cmp(&b.peak_ratio))
    }
}

/// Speed and terrain limits that follow from the presets and the load.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLimits {
    pub max_speed_mps: f32,
    pub max_slope_deg: f32,
    pub max_step_m: f32,
    /// Drive power relative to the empty default robot (heavier costs more).
    pub power_scale: f32,
    pub accel_limit_mps2: f32,
    /// Cargo the base can carry continuously, kg.
    pub capacity_kg: f32,
    /// Why each limit is what it is, in Korean.
    pub reasons: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct Pt {
    seg: Segment,
    group: Group,
    m: f32,
    p: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Group {
    /// Rotates with the trunk lean about the waist.
    Upper,
    /// Moves with the pelvis shift.
    Pelvis,
    Lower,
    /// The swing leg in single support: moves with the pelvis and lifts.
    Swing,
}

/// A robot assembled from presets.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Body {
    pub plan: BodyPlan,
    pub base: BaseKind,
    pub fingers: FingerCount,
    pub legged: bool,
    pub head: HeadPreset,
    pub chest: ChestPreset,
    pub belly: BellyPreset,
    pub arm: ArmPreset,
    pub leg: LegPreset,
    pub chassis: ChassisPreset,
    pub hand_kg: f32,
    /// Support length along travel and width, metres.
    pub support_len_m: f32,
    pub support_width_m: f32,
}

struct Eval {
    pts: Vec<Pt>,
    mass: f32,
    com: [f32; 3],
    support: [f32; 4],
    pitch: f32,
}

fn rot_xz(p: [f32; 3], pivot: [f32; 3], a: f32) -> [f32; 3] {
    // Positive `a` moves points above the pivot forward (+x).
    let (s, c) = a.sin_cos();
    let (x, z) = (p[0] - pivot[0], p[2] - pivot[2]);
    [pivot[0] + x * c + z * s, p[1], pivot[2] - x * s + z * c]
}

impl Body {
    pub fn new(plan: &BodyPlan, base: BaseKind, fingers: FingerCount) -> Body {
        let leg = plan.leg_preset();
        let legged = base == BaseKind::HumanFoot;
        let spec = base.spec();
        let (support_len_m, support_width_m) = if legged {
            (leg.foot_len_m, STANCE_WIDTH_M + leg.foot_width_m)
        } else {
            (spec.footprint_m, CHASSIS_WIDTH_M)
        };
        Body {
            plan: *plan,
            base,
            fingers,
            legged,
            head: plan.head_preset(),
            chest: plan.chest_preset(),
            belly: plan.belly_preset(),
            arm: plan.arm_preset(),
            leg,
            chassis: plan.chassis_preset(),
            hand_kg: hand_mass_kg(fingers),
            support_len_m,
            support_width_m,
        }
    }

    /// Robot mass without cargo.
    pub fn mass_kg(&self) -> f32 {
        let lower = if self.legged { 2.0 * (self.leg.mass_kg + self.leg.foot_mass_kg) } else { self.chassis.mass_kg + MAST_KG };
        self.head.mass_kg + self.chest.mass_kg + self.chest.battery_kg + self.belly.mass_kg + 2.0 * (self.arm.mass_kg + self.hand_kg) + lower
    }

    pub fn battery_j(&self) -> f64 {
        self.chest.battery_wh as f64 * 3600.0
    }

    /// Height of the CoM standing quietly, metres.
    pub fn com_height_m(&self) -> f32 {
        self.build(&Posture::quiet(0.0)).com[2]
    }

    /// How one arm's mass splits: upper arm mass, its CoM distance from the
    /// shoulder, forearm mass (CoM at 45% of the forearm). The upper part is
    /// placed so a straight arm has the preset's CoM.
    pub fn arm_split(&self) -> (f32, f32, f32) {
        let m_fore = 0.25 * self.arm.mass_kg;
        let m_up = self.arm.mass_kg - m_fore;
        let r_up = ((self.arm.com_m * self.arm.mass_kg - m_fore * (UPPER_ARM_M + 0.45 * FOREARM_M)) / m_up).max(0.02);
        (m_up, r_up, m_fore)
    }

    /// Support polygon `[x0, x1, y0, y1]` of a posture, before any base pitch
    /// (`pitch` is the base's pitch for a wheeled robot, 0 for legs).
    pub fn support_rect(&self, post: &Posture, pitch: f32) -> [f32; 4] {
        if self.legged {
            let c = post.slope_deg.to_radians().cos();
            let (x0, x1) = (-0.25 * self.leg.foot_len_m * c, 0.75 * self.leg.foot_len_m * c);
            if post.single_support {
                let ys = -STANCE_WIDTH_M / 2.0;
                [x0, x1, ys - self.leg.foot_width_m / 2.0, ys + self.leg.foot_width_m / 2.0]
            } else {
                [x0, x1, -self.support_width_m / 2.0, self.support_width_m / 2.0]
            }
        } else {
            let h = 0.5 * self.support_len_m * pitch.cos();
            [-h, h, -self.support_width_m / 2.0, self.support_width_m / 2.0]
        }
    }

    /// Pitch of a wheeled base on the posture's slope and step, radians (0 for legs).
    pub fn base_pitch(&self, post: &Posture) -> f32 {
        if self.legged {
            0.0
        } else {
            post.slope_deg.to_radians() + (post.step_m / self.support_len_m.max(0.1)).atan()
        }
    }

    fn arm_points(&self, side: usize, elev_deg: f32, elbow_deg: f32, load: f32) -> (Vec<Pt>, [f32; 3], [f32; 3]) {
        let y = if side == 0 { -SHOULDER_HALF_WIDTH_M } else { SHOULDER_HALF_WIDTH_M };
        let sh = [0.0, y, SHOULDER_M];
        let (m_up, r_up, m_fore) = self.arm_split();
        let a = elev_deg.to_radians();
        let b = (elev_deg + elbow_deg).to_radians();
        let du = [a.sin(), 0.0, -a.cos()];
        let df = [b.sin(), 0.0, -b.cos()];
        let at = |o: [f32; 3], d: [f32; 3], r: f32| [o[0] + d[0] * r, o[1] + d[1] * r, o[2] + d[2] * r];
        let elbow = at(sh, du, UPPER_ARM_M);
        let hand = at(elbow, df, FOREARM_M + 0.06);
        let mut v = vec![
            Pt { seg: Segment::Arm, group: Group::Upper, m: m_up, p: at(sh, du, r_up) },
            Pt { seg: Segment::Arm, group: Group::Upper, m: m_fore, p: at(elbow, df, 0.45 * FOREARM_M) },
            Pt { seg: Segment::Hand, group: Group::Upper, m: self.hand_kg, p: hand },
        ];
        if load > 0.0 {
            v.push(Pt { seg: Segment::Hand, group: Group::Upper, m: load, p: hand });
        }
        (v, sh, elbow)
    }

    fn points(&self, post: &Posture) -> Vec<Pt> {
        let mut v = Vec::new();
        let up = |seg, m, p| Pt { seg, group: Group::Upper, m, p };
        v.push(up(Segment::Head, self.head.mass_kg, [0.0, 0.0, SHOULDER_M + 0.07 + self.head.com_up_m]));
        v.push(up(Segment::Chest, self.chest.mass_kg, [0.0, 0.0, WAIST_M + self.chest.com_up_m]));
        let belly_z = WAIST_M - self.belly.com_down_m;
        if self.plan.battery_low {
            v.push(Pt { seg: Segment::Belly, group: Group::Pelvis, m: self.chest.battery_kg, p: [0.0, 0.0, belly_z] });
        } else {
            v.push(up(Segment::Chest, self.chest.battery_kg, [0.0, 0.0, WAIST_M + self.chest.com_up_m]));
        }
        v.push(Pt { seg: Segment::Belly, group: Group::Pelvis, m: self.belly.mass_kg, p: [0.0, 0.0, belly_z] });
        if post.cargo_kg > 0.0 {
            v.push(Pt { seg: Segment::Belly, group: Group::Pelvis, m: post.cargo_kg, p: [CARGO_X_M, 0.0, CARGO_Z_M] });
        }
        for side in 0..2 {
            let (pts, _, _) = self.arm_points(side, post.arm_elev_deg[side], post.elbow_deg[side], post.hand_load_kg[side]);
            v.extend(pts);
        }
        if self.legged {
            for (i, y) in [-STANCE_WIDTH_M / 2.0, STANCE_WIDTH_M / 2.0].into_iter().enumerate() {
                // The swing leg is the left one (index 1) in single support; it lifts the step height.
                let swing = post.single_support && i == 1;
                let lift = if swing { post.step_m.max(0.05) } else { 0.0 };
                let g = if swing { Group::Swing } else { Group::Lower };
                v.push(Pt { seg: Segment::Leg, group: g, m: self.leg.mass_kg, p: [if swing { 0.10 } else { 0.0 }, y, HIP_M - self.leg.com_down_m + lift * 0.5] });
                v.push(Pt { seg: Segment::Foot, group: g, m: self.leg.foot_mass_kg, p: [0.25 * self.leg.foot_len_m + if swing { 0.15 } else { 0.0 }, y, 0.04 + lift] });
            }
        } else {
            v.push(Pt { seg: Segment::Leg, group: Group::Lower, m: MAST_KG, p: [0.0, 0.0, 0.35] });
            v.push(Pt { seg: Segment::Foot, group: Group::Lower, m: self.chassis.mass_kg, p: [0.0, 0.0, self.chassis.com_m] });
        }
        v
    }

    /// Positions after the trunk lean `lean` (rad, + forward), pelvis shift
    /// `shift_y` (m) and the whole-body pitch of a wheeled base.
    fn evaluate(&self, post: &Posture, lean: f32, shift_y: f32) -> Eval {
        let mut pts = self.points(post);
        let pivot = [0.0, 0.0, WAIST_M];
        for p in pts.iter_mut() {
            if p.group == Group::Upper {
                p.p = rot_xz(p.p, pivot, lean);
            }
            if matches!(p.group, Group::Upper | Group::Pelvis | Group::Swing) {
                p.p[1] += shift_y;
            }
        }
        // A wheeled base tilts with the ground; a step tilts it by the step over the wheelbase.
        let pitch = self.base_pitch(post);
        if pitch != 0.0 {
            for p in pts.iter_mut() {
                // Nose up moves points above the ground backward.
                p.p = rot_xz(p.p, [0.0, 0.0, 0.0], -pitch);
            }
        }
        let mass: f32 = pts.iter().map(|p| p.m).sum();
        let mut com = [0.0f32; 3];
        for p in &pts {
            for k in 0..3 {
                com[k] += p.m * p.p[k] / mass;
            }
        }
        let support = self.support_rect(post, pitch);
        Eval { pts, mass, com, support, pitch }
    }

    /// Where the controller wants the CoM along x: a little ahead of the
    /// ankles for a legged base (people stand ~5 cm ahead, Loram 2007), at the
    /// centre for a wheeled one.
    fn home_x(&self) -> f32 {
        if self.legged {
            0.15 * self.leg.foot_len_m
        } else {
            0.0
        }
    }

    /// Shift held without leaning: the ankle band of a legged base (est 10% of
    /// the foot), a small offset for a wheeled one.
    fn ankle_band(&self) -> f32 {
        if self.legged {
            0.10 * self.leg.foot_len_m
        } else {
            0.02
        }
    }

    /// Nominal feed-forward prediction of the CoM shift a posture causes: the
    /// controller's own model, which assumes human proportions (arm 5% of the
    /// body mass, CoM at 53% of the arm, Winter 2009) and the payload it was
    /// told. The SDS learns the ratio between this and what happens.
    pub fn nominal_shift(&self, post: &Posture) -> f32 {
        let m = self.mass_kg() + post.cargo_kg + post.hand_load_kg.iter().sum::<f32>();
        let arm_len = UPPER_ARM_M + FOREARM_M + 0.06;
        let mut s = 0.0;
        for i in 0..2 {
            let a = post.arm_elev_deg[i].to_radians();
            let reach = arm_len * a.sin();
            s += 0.05 * m * 0.53 * reach + post.hand_load_kg[i] * reach;
        }
        s += post.cargo_kg * CARGO_X_M;
        s / m
    }

    /// Builds the posture without compensation.
    fn build(&self, post: &Posture) -> Eval {
        self.evaluate(post, 0.0, 0.0)
    }

    /// Solves the compensation for `post` and reports balance, joints and the N:N cells.
    /// `grip_capacity_n` is the friction capacity of one hand (from `sim::grasp`).
    pub fn solve(&self, post: &Posture, ff: &FeedForward, grip_capacity_n: f32) -> BalanceReport {
        // The disturbance against the same posture with the source removed.
        let base_post = Posture { arm_elev_deg: [0.0, 0.0], elbow_deg: [10.0, 10.0], hand_load_kg: [0.0, 0.0], cargo_kg: if post.source == Source::CarryFront { 0.0 } else { post.cargo_kg }, ..*post };
        let uncomp = self.build(post);
        let reference = self.build(&base_post);
        let home = self.home_x();
        // The robot knows its own planned acceleration and leans into it, as
        // standing passengers do. The lean is in place before the acceleration
        // starts (posture leads movement, `dynamics`), so the ZMP is at the CoM
        // before and at `x_com − z·a/g` during it; the CoM target that keeps the
        // worse of the two margins largest centres that pair in the support.
        // A waist without a pitch joint cannot lean (max_lean 0).
        let target = if post.accel_mps2 != 0.0 {
            let s = uncomp.support;
            0.5 * (s[0] + s[1]) + 0.5 * uncomp.com[2] * post.accel_mps2 / G
        } else {
            home
        };
        let delta = uncomp.com[0] - target;
        let band = self.ankle_band();
        let excess = delta - delta.clamp(-band, band);
        // Lean: the upper group's moment arm decides how much a degree buys.
        let upper_m: f32 = uncomp.pts.iter().filter(|p| p.group == Group::Upper).map(|p| p.m).sum();
        let upper_h: f32 = uncomp.pts.iter().filter(|p| p.group == Group::Upper).map(|p| p.m * (p.p[2] - WAIST_M)).sum::<f32>() / upper_m.max(1e-3);
        let max_lean = self.belly.max_lean_deg.to_radians();
        let lean = if excess.abs() > 1e-5 && max_lean > 0.0 {
            (-(excess * uncomp.mass) / (upper_m * upper_h.max(0.05))).clamp(-1.0, 1.0).asin().clamp(-max_lean, max_lean)
        } else {
            0.0
        };
        // Pelvis over the stance foot: only what is above the stance leg moves.
        let shift_y = if self.legged && post.single_support {
            let moved: f32 = uncomp.pts.iter().filter(|p| p.group != Group::Lower).map(|p| p.m).sum();
            (-STANCE_WIDTH_M / 2.0 - uncomp.com[1]) * uncomp.mass / moved.max(1e-3)
        } else {
            0.0
        };
        let fin = self.evaluate(post, lean, shift_y);

        let shift_uncomp = uncomp.com[0] - reference.com[0];
        let restore = if (uncomp.com[0] - home).abs() > 1e-4 { ((uncomp.com[0] - fin.com[0]) / (uncomp.com[0] - home)).clamp(-1.0, 2.0) } else { 0.0 };

        // Feed-forward: how much of the disturbance was anticipated.
        let predicted = self.nominal_shift(post) - self.nominal_shift(&base_post);
        let anticipated = if ff.enabled && shift_uncomp.abs() > 1e-5 { (ff.gain * predicted / shift_uncomp).clamp(0.0, 2.0) } else { 0.0 };
        let lag = (REACTION_DELAY_S / post.source.duration_s()).min(1.0);
        let eps_ff = lag * (1.0 - anticipated).abs();
        let eps_noff = lag;

        let a = post.accel_mps2;
        let zmp = [fin.com[0] - fin.com[2] * a / G, fin.com[1]];
        let half_x = 0.5 * (fin.support[1] - fin.support[0]);
        let half_y = 0.5 * (fin.support[3] - fin.support[2]);
        let cx = 0.5 * (fin.support[0] + fin.support[1]);
        let cy = 0.5 * (fin.support[2] + fin.support[3]);
        let mx = 1.0 - (zmp[0] - cx).abs() / half_x.max(1e-3);
        let my = 1.0 - (zmp[1] - cy).abs() / half_y.max(1e-3);
        let dip = |eps: f32| eps * shift_uncomp.abs() / half_x.max(1e-3);
        let margin = mx.min(my) - dip(eps_ff);
        let margin_noff = mx.min(my) - dip(eps_noff);

        // Head: body pitch and lean reach the camera, the neck cancels part of it.
        let keep = 1.0 - self.head.gaze_stabilization;
        let head_pitch = (fin.pitch + lean).to_degrees() * keep;
        let head_roll = if self.legged && post.single_support { (shift_y / HIP_M).atan().to_degrees() * keep } else { 0.0 };

        let joints = self.joint_loads(post, &fin, lean, shift_y, grip_capacity_n);
        let comp_joint = |j: &JointLoad, eps: f32| {
            if matches!(j.joint.as_str(), "waist" | "ankle" | "hip_roll" | "drive") {
                j.peak_ratio * (1.0 + eps)
            } else {
                j.peak_ratio
            }
        };
        let effort_sum: f32 = joints.iter().map(|j| comp_joint(j, eps_ff).powi(2)).sum();
        let worst = joints.iter().max_by(|a, b| comp_joint(a, eps_ff).total_cmp(&comp_joint(b, eps_ff)));
        let (mut feasible, mut limiting) = (true, None);
        if let Some(w) = worst {
            if comp_joint(w, eps_ff) > 1.0 {
                feasible = false;
                limiting = Some(format!("{} {:.1}/{:.1} N·m ({})", w.joint, w.torque_nm * (1.0 + eps_ff), w.peak_nm, w.actuator));
            }
        }
        if margin < 0.0 {
            feasible = false;
            limiting = Some(format!("균형 여유 {:.2} < 0", margin));
        }

        let mut rep = BalanceReport {
            source: post.source,
            mass_kg: fin.mass,
            hand_load_kg: post.hand_load_kg.iter().sum(),
            cargo_kg: post.cargo_kg,
            slope_deg: post.slope_deg,
            com: fin.com,
            zmp,
            support: fin.support,
            margin_x: mx,
            margin_y: my,
            margin,
            margin_noff,
            shift_uncomp_m: shift_uncomp,
            restore_frac: restore,
            trunk_lean_deg: lean.to_degrees(),
            lateral_shift_m: shift_y,
            lateral_com_shift_m: fin.com[1] - uncomp.com[1],
            head_pitch_deg: head_pitch,
            head_roll_deg: head_roll,
            joints,
            effort_sum,
            feasible,
            limiting,
            ff_predicted_m: predicted,
            ff_gain: ff.gain,
            couplings: Vec::new(),
        };
        rep.couplings = self.couplings(post, &rep, eps_ff, eps_noff, &reference);
        rep
    }

    fn joint_loads(&self, post: &Posture, e: &Eval, lean: f32, shift_y: f32, grip_capacity_n: f32) -> Vec<JointLoad> {
        let a = post.accel_mps2;
        let lever = |p: &Pt, j: [f32; 3]| (p.p[0] - j[0]) - (a / G) * (p.p[2] - j[2]);
        let moment = |pts: &mut dyn Iterator<Item = &Pt>, j: [f32; 3]| -> f32 { G * pts.map(|p| p.m * lever(p, j)).sum::<f32>() };
        let jl = |joint: &str, seg: Segment, act: &Actuator, tau: f32| {
            let t = tau.abs();
            JointLoad { joint: joint.into(), segment: seg, actuator: act.name.into(), torque_nm: t, cont_nm: act.cont_nm, peak_nm: act.peak_nm, effort: t / act.cont_nm.max(1e-3), peak_ratio: t / act.peak_nm.max(1e-3) }
        };
        let tilt = |p: [f32; 3]| {
            let mut q = rot_xz(p, [0.0, 0.0, WAIST_M], lean);
            q[1] += shift_y;
            if e.pitch != 0.0 {
                q = rot_xz(q, [0.0, 0.0, 0.0], -e.pitch);
            }
            q
        };
        let mut out = Vec::new();
        for side in 0..2 {
            let (pts, sh, el) = self.arm_points(side, post.arm_elev_deg[side], post.elbow_deg[side], post.hand_load_kg[side]);
            let pts: Vec<Pt> = pts.into_iter().map(|p| Pt { p: tilt(p.p), ..p }).collect();
            let (sh, el) = (tilt(sh), tilt(el));
            let tag = if side == 0 { "r" } else { "l" };
            out.push(jl(&format!("shoulder_{tag}"), Segment::Arm, &self.arm.shoulder, moment(&mut pts.iter(), sh)));
            out.push(jl(&format!("elbow_{tag}"), Segment::Arm, &self.arm.elbow, moment(&mut pts.iter().skip(1), el)));
            let load = post.hand_load_kg[side];
            if load > 0.0 {
                let need = load * (G + a.abs()) * GRIP_SAFETY;
                let cap = grip_capacity_n.max(1e-3);
                out.push(JointLoad { joint: format!("grip_{tag}"), segment: Segment::Hand, actuator: "손가락 마찰".into(), torque_nm: need, cont_nm: cap, peak_nm: cap, effort: need / cap, peak_ratio: need / cap });
            }
        }
        let waist = tilt([0.0, 0.0, WAIST_M]);
        if let Some(pitch) = &self.belly.pitch {
            out.push(jl("waist", Segment::Belly, pitch, moment(&mut e.pts.iter().filter(|p| p.group == Group::Upper), waist)));
        }
        let neck = tilt([0.0, 0.0, SHOULDER_M + 0.07]);
        out.push(jl("neck", Segment::Head, &self.head.neck, moment(&mut e.pts.iter().filter(|p| p.seg == Segment::Head), neck)));
        if self.legged {
            let n_support = if post.single_support { 1.0 } else { 2.0 };
            let above_feet: f32 = e.mass - e.pts.iter().filter(|p| p.seg == Segment::Foot).map(|p| p.m).sum::<f32>();
            let x_ank = 0.0;
            let tau_ankle = G * e.pts.iter().filter(|p| p.seg != Segment::Foot && p.group != Group::Swing).map(|p| p.m * lever(p, [x_ank, 0.0, 0.08])).sum::<f32>() / n_support;
            out.push(jl("ankle", Segment::Foot, &self.leg.ankle, tau_ankle));
            let d = if post.step_m > 0.0 { step_knee_lever(post.step_m) } else if post.single_support { KNEE_LEVER_WALK_M } else { KNEE_LEVER_STAND_M };
            let leg_m = 2.0 * self.leg.mass_kg;
            out.push(jl("knee", Segment::Leg, &self.leg.knee, (above_feet - leg_m * 0.5) * G * d / n_support));
            if post.single_support {
                let upper: f32 = e.mass - e.pts.iter().filter(|p| matches!(p.seg, Segment::Leg | Segment::Foot)).map(|p| p.m).sum::<f32>();
                // Hip roll uses the knee's actuator class (research §5: same family per tier).
                out.push(jl("hip_roll", Segment::Leg, &self.leg.knee, upper * G * HIP_HALF_WIDTH_M));
            }
        } else {
            let slope = post.slope_deg.to_radians();
            let force = e.mass * (G * (slope.sin() + ROLLING_RESISTANCE * slope.cos()) + a.abs());
            let f = self.chassis.drive_force_n.max(1e-3);
            out.push(JointLoad { joint: "drive".into(), segment: Segment::Foot, actuator: self.chassis.name.into(), torque_nm: force, cont_nm: f, peak_nm: 2.0 * f, effort: force / f, peak_ratio: force / (2.0 * f) });
        }
        out
    }

    fn couplings(&self, post: &Posture, r: &BalanceReport, eps_ff: f32, eps_noff: f32, reference: &Eval) -> Vec<Coupling> {
        let src = post.source;
        let foot_len = if self.legged { self.leg.foot_len_m } else { self.support_len_m };
        let j = |name: &str| r.joint(name);
        let cell = |target: Segment, axis: &str, unit: &str, value: f32, value_noff: f32, joint: Option<&JointLoad>, comp: bool| {
            let k = |eps: f32| joint.map(|j| if comp { j.peak_ratio * (1.0 + eps) } else { j.peak_ratio });
            Coupling { source: src, target, axis: axis.into(), value, unit: unit.into(), value_noff, effort: k(eps_ff), effort_noff: k(eps_noff), torque_nm: joint.map(|j| j.torque_nm) }
        };
        let quiet_ankle = {
            let q = Posture { source: Source::Quiet, ..Posture::quiet(post.cargo_kg) };
            let e = self.evaluate(&q, 0.0, 0.0);
            self.joint_loads(&q, &e, 0.0, 0.0, 1.0).into_iter().find(|j| j.joint == "ankle" || j.joint == "drive").map(|j| j.torque_nm).unwrap_or(0.0)
        };
        let ankle = j("ankle").or(j("drive"));
        let ankle_rise = ankle.map(|a| if quiet_ankle > 1e-3 { a.torque_nm / quiet_ankle - 1.0 } else { 0.0 }).unwrap_or(0.0);
        let mut v = Vec::new();
        let head_tilt = r.head_pitch_deg.abs().max(r.head_roll_deg.abs());
        v.push(cell(Segment::Head, "tilt_deg", "°", head_tilt, head_tilt, j("neck"), false));
        v.push(cell(Segment::Chest, "lean_deg", "°", r.trunk_lean_deg, r.trunk_lean_deg, None, false));
        v.push(cell(Segment::Belly, "restore_frac", "", r.restore_frac, r.restore_frac * (1.0 - eps_noff), j("waist"), true));
        match src {
            Source::ArmRaise | Source::ArmLift => {
                v.push(cell(Segment::Arm, "peak_ratio", "", j("shoulder_r").map_or(0.0, |x| x.peak_ratio), j("shoulder_r").map_or(0.0, |x| x.peak_ratio), j("shoulder_r"), false));
            }
            _ => {}
        }
        if let Some(g) = j("grip_r") {
            v.push(cell(Segment::Hand, "grip_effort", "", g.effort, g.effort, Some(g), false));
        }
        if self.legged {
            let leg_joint = if post.single_support && post.step_m == 0.0 { j("hip_roll") } else { j("knee") };
            if let Some(lj) = leg_joint {
                v.push(cell(Segment::Leg, "peak_ratio", "", lj.peak_ratio, lj.peak_ratio, Some(lj), lj.joint == "hip_roll"));
            }
            if post.single_support {
                let frac = r.lateral_com_shift_m.abs() / (STANCE_WIDTH_M / 2.0);
                v.push(cell(Segment::Belly, "shift_frac", "", frac, frac, None, false));
            }
        }
        let shift = (r.shift_uncomp_m) / foot_len;
        v.push(cell(Segment::Foot, "com_shift_frac", "", shift, shift, ankle, true));
        v.push(cell(Segment::Foot, "ankle_rise", "", ankle_rise, ankle_rise * (1.0 + eps_noff), ankle, true));
        v.push(cell(Segment::Foot, "margin", "", r.margin, r.margin_noff, None, false));
        if src == Source::Slope {
            let x_ref = reference.com[0];
            let cop = (r.zmp[0] - x_ref) / foot_len;
            v.push(cell(Segment::Foot, "cop_shift_frac", "", cop, cop, ankle, true));
        }
        v
    }

    /// Speed, slope, step and cargo limits for `cargo_kg` on top of the
    /// mechanism's own limits (`spec`). Never above what the mechanism allows.
    pub fn drive_limits(&self, spec: &BaseSpec, cargo_kg: f32) -> DriveLimits {
        let m = self.mass_kg() + cargo_kg;
        let ref_m = Body::new(&BodyPlan::default(), self.base, FingerCount::Five).mass_kg();
        let mut reasons = Vec::new();
        let mut speed = spec.max_speed_mps;
        let mut slope = spec.max_slope_deg;
        let mut step = spec.max_step_m;
        let quiet = self.solve(&Posture::quiet(cargo_kg), &FeedForward::default(), 1.0);
        let half = 0.5 * (quiet.support[1] - quiet.support[0]);
        let accel = (G * (quiet.margin_x - MARGIN_KEEP).max(0.02) * half / quiet.com[2].max(0.1)).min(3.0);
        let capacity;
        if self.legged {
            let v_fr = (self.leg.froude * G * HIP_M).sqrt();
            if v_fr < speed {
                speed = v_fr;
                reasons.push(format!("보행 속도 한계 Froude {:.2} → {:.2} m/s", self.leg.froude, v_fr));
            }
            // Single support on one knee while walking.
            let need = m * G * KNEE_LEVER_WALK_M;
            let s = self.leg.knee.cont_nm / need;
            if s < 1.0 {
                let f = s * s;
                speed *= f;
                reasons.push(format!("무릎 연속 토크 {:.1} < 필요 {:.1} N·m → 속도 ×{:.2}", self.leg.knee.cont_nm, need, f));
            }
            if self.leg.knee.peak_nm < KNEE_SAFETY * need {
                speed = speed.min(0.02);
                reasons.push(format!("무릎 피크 {:.1} < {:.1}×필요 {:.1} N·m: 거의 걸을 수 없음", self.leg.knee.peak_nm, KNEE_SAFETY, need));
            }
            // Highest step the knee can push up: 1.5 · m · g · d(h) <= peak.
            let d_max = self.leg.knee.peak_nm / (KNEE_SAFETY * m * G);
            let h_max = ((d_max / HIP_M - 0.10) / 0.40 * HIP_M).max(0.0);
            if h_max < step {
                step = h_max;
                reasons.push(format!("무릎 피크로 오를 수 있는 턱 {:.0} cm", h_max * 100.0));
            }
            let slip = FOOT_FRICTION.atan().to_degrees();
            if slip < slope {
                slope = slip;
                reasons.push(format!("발바닥 마찰 {FOOT_FRICTION} 미끄럼 한계 {slip:.0}°"));
            }
            capacity = (self.leg.knee.cont_nm / (G * KNEE_LEVER_WALK_M) - self.mass_kg()).max(0.0);
        } else {
            if self.chassis.max_speed_mps < speed {
                speed = self.chassis.max_speed_mps;
                reasons.push(format!("섀시 최고 속도 {:.2} m/s ({})", speed, self.chassis.name));
            }
            let trac = (self.chassis.drive_force_n / (m * G) - ROLLING_RESISTANCE).clamp(-1.0, 1.0).asin().to_degrees();
            if trac < slope {
                slope = trac.max(0.0);
                reasons.push(format!("구동력 {:.0} N으로 오를 수 있는 경사 {:.0}°", self.chassis.drive_force_n, slope));
            }
            // Steepest slope where the margin (with the waist lean) stays above MARGIN_KEEP.
            let mut tip = 0.0;
            let mut d = 0.5;
            while d <= 45.0 {
                let p = Posture { source: Source::Slope, slope_deg: d, ..Posture::quiet(cargo_kg) };
                if self.solve(&p, &FeedForward::default(), 1.0).margin_x < MARGIN_KEEP {
                    break;
                }
                tip = d;
                d += 0.5;
            }
            if tip < slope {
                slope = tip;
                reasons.push(format!("키 {:.2} m 무게중심과 바퀴 간 거리 {:.2} m로 넘어지지 않는 경사 {tip:.1}°", quiet.com[2], self.support_len_m));
            }
            let v_bal = (2.0 * accel * STOP_DISTANCE_M).sqrt();
            if v_bal < speed {
                speed = v_bal;
                reasons.push(format!("{STOP_DISTANCE_M} m 안에 서려면(감속 {accel:.2} m/s²) {v_bal:.2} m/s"));
            }
            let structural = match self.chassis.tier {
                Tier::Budget | Tier::BudgetPlus => 8.0,
                Tier::Mid => 30.0,
                Tier::Pro => 100.0,
            };
            capacity = (structural - (self.mass_kg() - self.chassis.mass_kg)).max(0.0);
        }
        let capacity_kg = capacity.min(self.base.carry_capacity_kg());
        DriveLimits { max_speed_mps: speed.max(0.01), max_slope_deg: slope.max(0.0), max_step_m: step.max(0.0), power_scale: m / ref_m, accel_limit_mps2: accel, capacity_kg, reasons }
    }

    /// The acceleration the drive planner commands to start (`dir > 0`) or to
    /// stop (`dir < 0`) on `slope_deg` with `cargo_kg`: the ramp to cruise speed
    /// in at least `min_ramp_s`, never harder than the balance allows.
    ///
    /// A start pushes the ZMP backward (`x_zmp = x_com − z·a/g`), a stop
    /// forward, so each is limited by the room between the ZMP and the support
    /// edge it moves toward, less the kept margin, on this slope (a wheeled
    /// base pitches with it and its CoM moves downhill) with this cargo. An
    /// uphill start is also limited by traction. The flat, symmetric limit of
    /// `drive_limits` stays as the upper bound.
    pub fn planned_accel(&self, spec: &BaseSpec, slope_deg: f32, cargo_kg: f32, dir: f32, min_ramp_s: f32) -> f32 {
        self.planned_accel_within(&self.drive_limits(spec, cargo_kg), slope_deg, cargo_kg, dir, min_ramp_s)
    }

    /// [`Body::planned_accel`] with the drive limits already computed (the world caches them).
    pub fn planned_accel_within(&self, l: &DriveLimits, slope_deg: f32, cargo_kg: f32, dir: f32, min_ramp_s: f32) -> f32 {
        let p = Posture { source: if slope_deg.abs() >= 3.0 { Source::Slope } else { Source::Quiet }, slope_deg, ..Posture::quiet(cargo_kg) };
        let r = self.solve(&p, &FeedForward::default(), 1e3);
        let s = r.support;
        let keep = MARGIN_KEEP * 0.5 * (s[1] - s[0]);
        let room = if dir > 0.0 { r.zmp[0] - (s[0] + keep) } else { (s[1] - keep) - r.zmp[0] };
        let mut a = (G * room.max(0.0) / r.com[2].max(0.1)).min(l.accel_limit_mps2);
        if !self.legged && dir > 0.0 {
            let m = self.mass_kg() + cargo_kg;
            let slope = slope_deg.to_radians();
            let trac = self.chassis.drive_force_n / m - G * (slope.sin() + ROLLING_RESISTANCE * slope.cos());
            a = a.min(trac.max(0.0));
        }
        (l.max_speed_mps / min_ramp_s.max(1e-3)).min(a).max(0.05)
    }

    /// Lifts `load_kg` with one arm, or with both when one is not enough.
    /// Returns the report of the posture that worked, or the one-arm attempt.
    pub fn lift(&self, load_kg: f32, cargo_kg: f32, ff: &FeedForward, grip_capacity_n: f32) -> (BalanceReport, bool) {
        let one = self.solve(&Posture::arm_lift(load_kg, false, cargo_kg), ff, grip_capacity_n);
        if one.feasible {
            return (one, false);
        }
        let two = self.solve(&Posture::arm_lift(load_kg, true, cargo_kg), ff, grip_capacity_n);
        if two.feasible {
            (two, true)
        } else {
            (one, false)
        }
    }
}

/// Knee lever for climbing a step of height `h`, metres (est: scaled from the
/// 18 cm stair data of Lin 2004 to a 0.58 m leg, `L · (0.10 + 0.40 · h / L)`).
pub fn step_knee_lever(h: f32) -> f32 {
    HIP_M * (0.10 + 0.40 * h / HIP_M)
}

/// Context the references depend on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RefCtx {
    /// Load over body mass.
    pub load_ratio: f32,
    /// Load moment arm, metres (front carry).
    pub load_dist_m: f32,
    pub slope_deg: f32,
}

/// A research reference for one N:N cell: expected value, tolerance and source.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reference {
    pub value: f32,
    pub tolerance: f32,
    pub cite: &'static str,
}

fn r(value: f32, tolerance: f32, cite: &'static str) -> Option<Reference> {
    Some(Reference { value, tolerance: tolerance.max(1e-3), cite })
}

/// The human reference for an N:N cell, made dimensionless
/// (`docs/research/balance.md` §8 and "로봇에 옮기는 방법"). `None` where
/// the literature has no number; those cells are recorded as baselines only.
pub fn reference(source: Source, target: Segment, axis: &str, ctx: &RefCtx) -> Option<Reference> {
    // Human load ratio of the 5 kg experiments (5 / 75 kg).
    const R5: f32 = 0.067;
    let lr = (ctx.load_ratio / R5).max(0.0);
    match (source, target, axis) {
        (_, Segment::Head, "tilt_deg") => r(0.0, if matches!(source, Source::FootLift | Source::StepUp) { 8.5 } else { 5.0 }, "Pozzo 1990 (머리 <20°), 보행 피치 1.5–8.5°"),
        (Source::ArmRaise, Segment::Arm, "peak_ratio") => r(0.15, 0.05, "어깨 ~15% (유도: Winter 질량, Marcondes 2019)"),
        (Source::ArmRaise, Segment::Foot, "com_shift_frac") => r(0.056, 0.017, "CoM +1.5 cm / 발 26.6 cm (유도)"),
        (Source::ArmRaise, Segment::Foot, "ankle_rise") => r(0.33, 0.10, "발목 +33% (유도)"),
        (Source::ArmRaise, Segment::Belly, "restore_frac") => r(0.2, 0.2, "무부하: 지지 분절이 뒤로 보상하나 다 막지 못함 (Aruin & Latash 1995)"),
        (Source::ArmLift, Segment::Arm, "peak_ratio") => {
            let v = (0.15 + 0.44 * lr).min(1.0);
            r(v, 0.3 * v, "어깨 무부하 15% → 5 kg 59%, 하중비에 선형 (유도)")
        }
        (Source::ArmLift, Segment::Foot, "com_shift_frac") => {
            let v = (1.5 + 4.2 * lr) / 26.6;
            r(v, 0.3 * v, "CoM +1.5 → +5.7 cm (5 kg), 하중비에 선형 (유도)")
        }
        (Source::ArmLift, Segment::Belly, "restore_frac") => {
            let v = (0.2 + 0.7 * lr.min(1.0)).min(0.9);
            r(v, 0.3, "5 kg: 골반·몸통 ≈ 6 cm 뒤로 = 거의 전부 회복 (유도)")
        }
        (Source::FootLift, Segment::Belly, "shift_frac") => r(1.0, 0.2, "좌우 CoM 이동 = 발 중심 거리의 절반 (유도)"),
        (Source::FootLift, Segment::Leg, "peak_ratio") => r(0.55, 0.10, "지지 고관절 외전 최대의 50–60% (유도)"),
        (Source::FootLift, Segment::Foot, "margin") => r(0.32, 0.12, "한 발 b_min 15.5 ± 6 mm / 발 반폭 48 mm (Hof 2005, 유도)"),
        (Source::StepUp, Segment::Leg, "peak_ratio") => r(0.46, 0.10, "무릎 최대의 42–50% (Lin 2004, 유도)"),
        (Source::Slope, Segment::Foot, "cop_shift_frac") => {
            // Toes-up (uphill) moves the CoP back ~0.6% of the foot per degree, toes-down forward ~0.8%.
            let v = if ctx.slope_deg >= 0.0 { -0.006 * ctx.slope_deg } else { -0.008 * ctx.slope_deg };
            r(v, 0.3 * v.abs().max(0.01), "Atsawakaewmongkhon 2024 CoP 이동 (유도)")
        }
        (Source::CarryFront, Segment::Belly, "restore_frac") => r(0.9, 0.3, "앞쪽 짐: 뒤로 기울여 CoM 회복 (Hudson 2018 원리, 유도)"),
        (Source::CarryFront, Segment::Foot, "com_shift_frac") => {
            let v = 0.133 * (ctx.load_ratio * ctx.load_dist_m) / (0.133 * 0.30);
            r(v, 0.3 * v.max(0.01), "10 kg @ 30 cm: 보상 없으면 CoM 3.5 cm (유도)")
        }
        (_, Segment::Foot, "margin") => r(0.48, 0.15, "두 발 b_min 64 mm / 지지 반폭 133 mm (Hof 2005, 유도)"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn tier_masses_and_com_heights_match_the_research_totals() {
        // Research §5 tier totals for the legged robot: 7.7 / 20.1 / 29.6 kg (est), CoM ~0.6 m.
        for (tier, want) in [(Tier::Budget, 7.7), (Tier::Mid, 20.1), (Tier::Pro, 29.6)] {
            let b = body(tier, BaseKind::HumanFoot);
            let m = b.mass_kg();
            assert!((m - want).abs() / want < 0.15, "{tier:?}: {m} kg vs {want}");
            let h = b.com_height_m();
            assert!((0.5..0.75).contains(&h), "{tier:?}: CoM {h} m");
        }
        // Moving the battery to the pelvis lowers the CoM by about 2 cm (research §6).
        let chest = body(Tier::Mid, BaseKind::HumanFoot);
        let low = Body::new(&BodyPlan { battery_low: true, ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        let d = chest.com_height_m() - low.com_height_m();
        assert!(d > 0.01 && d < 0.04, "{d}");
    }

    #[test]
    fn an_arm_raise_moves_the_com_as_the_lever_rule_says() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let ff = FeedForward { enabled: false, gain: 1.0 };
        let r = b.solve(&Posture::arm_raise(0.0), &ff, 50.0);
        // Hand-computed: the raised arm's points moved forward by their sine over the total mass.
        let (pts, _, _) = b.arm_points(0, 75.0, 15.0, 0.0);
        let (down, _, _) = b.arm_points(0, 0.0, 10.0, 0.0);
        let dm: f32 = pts.iter().zip(&down).map(|(a, d)| a.m * (a.p[0] - d.p[0])).sum();
        assert!((r.shift_uncomp_m - dm / b.mass_kg()).abs() < 1e-4, "{} vs {}", r.shift_uncomp_m, dm / b.mass_kg());
        assert!(r.shift_uncomp_m > 0.0);
        assert!(r.joint("shoulder_r").unwrap().torque_nm > r.joint("shoulder_l").unwrap().torque_nm);
    }

    #[test]
    fn what_can_be_lifted_follows_the_arm_preset() {
        // Items of the world: coal 1.4, log 2.4, stone 2.6, iron ore 4.0 kg.
        let grip = 200.0;
        let ff = FeedForward::default();
        let budget = body(Tier::Budget, BaseKind::Wheel);
        let mid = body(Tier::Mid, BaseKind::Wheel);
        let pro = body(Tier::Pro, BaseKind::Wheel);
        assert!(!budget.lift(2.4, 0.0, &ff, grip).0.feasible, "a bus-servo elbow cannot lift a log");
        assert!(mid.lift(2.4, 0.0, &ff, grip).0.feasible);
        let (iron_mid, two) = mid.lift(4.0, 0.0, &ff, grip);
        assert!(iron_mid.feasible && two, "mid arms lift iron ore only with both arms: {:?}", iron_mid.limiting);
        let (iron_pro, two_pro) = pro.lift(4.0, 0.0, &ff, grip);
        assert!(iron_pro.feasible && !two_pro, "pro shoulder and elbow lift iron ore one-handed: {:?}", iron_pro.limiting);
        // Lift capacity falls with reach roughly as 1/H, like the NIOSH horizontal multiplier.
        let cap = |elev: f32| {
            let mut kg = 0.0;
            while kg < 20.0 {
                let p = Posture { arm_elev_deg: [elev, 0.0], elbow_deg: [0.0, 10.0], hand_load_kg: [kg + 0.1, 0.0], ..Posture::arm_lift(0.0, false, 0.0) };
                if !mid.solve(&p, &ff, 1e6).joint("shoulder_r").map_or(false, |j| j.peak_ratio <= 1.0) {
                    break;
                }
                kg += 0.1;
            }
            kg
        };
        let (near, far) = (cap(30.0), cap(90.0));
        let ratio = near / far;
        assert!(ratio > 1.5 && ratio < 2.6, "reach 0.24 m vs 0.48 m: {near} kg vs {far} kg");
    }

    #[test]
    fn a_tall_robot_on_a_short_wheelbase_tips_early_and_a_waist_lean_helps() {
        let spec = BaseKind::Wheel.spec();
        let stiff = Body::new(&BodyPlan { belly: Tier::Budget, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        let lean = body(Tier::Pro, BaseKind::Wheel);
        let a = stiff.drive_limits(&spec, 0.0);
        let b = Body::new(&BodyPlan { belly: Tier::Pro, ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five).drive_limits(&spec, 0.0);
        assert!(a.max_slope_deg < spec.max_slope_deg, "{:?}", a.reasons);
        assert!(b.max_slope_deg > a.max_slope_deg, "a waist that can lean holds a steeper slope: {} vs {}", b.max_slope_deg, a.max_slope_deg);
        // Static tip angle without a lean is about atan(half wheelbase / CoM height) minus the kept margin.
        let h = stiff.com_height_m();
        let tip = ((1.0 - MARGIN_KEEP) * 0.5 * spec.footprint_m / h).atan().to_degrees();
        assert!((a.max_slope_deg - tip).abs() < 2.0, "{} vs {tip}", a.max_slope_deg);
        let _ = lean;
    }

    #[test]
    fn walking_speed_and_steps_follow_the_leg_preset_and_the_load() {
        let spec = BaseKind::HumanFoot.spec();
        let mid = body(Tier::Mid, BaseKind::HumanFoot);
        let empty = mid.drive_limits(&spec, 0.0);
        let loaded = mid.drive_limits(&spec, 6.0);
        assert!(empty.max_step_m >= 0.15, "a mid leg climbs the 15 cm curb: {:?}", empty.reasons);
        assert!(loaded.max_speed_mps <= empty.max_speed_mps);
        // DIY BLDC knees: the 10 cm curb is already too much, and cargo lowers the step further.
        let plus = Body::new(&BodyPlan { leg: Tier::BudgetPlus, ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        let (p0, p2) = (plus.drive_limits(&spec, 0.0), plus.drive_limits(&spec, 2.0));
        assert!(p0.max_step_m < 0.10 && p0.max_step_m > 0.03, "{:?}", p0.reasons);
        assert!(p2.max_step_m < p0.max_step_m);
        let budget = body(Tier::Budget, BaseKind::HumanFoot).drive_limits(&spec, 0.0);
        assert!(budget.max_speed_mps < 0.05, "bus-servo knees cannot carry the robot: {:?}", budget.reasons);
        let pro = body(Tier::Pro, BaseKind::HumanFoot).drive_limits(&spec, 0.0);
        assert!(pro.capacity_kg > empty.capacity_kg);
    }

    #[test]
    fn a_foot_lift_shifts_the_pelvis_over_the_stance_foot() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let p = Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) };
        let r = b.solve(&p, &FeedForward::default(), 50.0);
        assert!((r.com[1] + STANCE_WIDTH_M / 2.0).abs() < 0.02, "CoM over the right foot: {:?}", r.com);
        assert!(r.margin > 0.0, "{r:?}");
        let hip = r.joint("hip_roll").unwrap();
        assert!(hip.peak_ratio > 0.05 && hip.peak_ratio < 1.0, "{}", hip.peak_ratio);
        // A person uses 50–60% of the hip abductor here; a Robstride 03 hip on a
        // 20 kg robot uses far less. The reference check says so instead of hiding it:
        // the hip is over-provisioned (a cheaper actuator would do).
        let hr = reference(Source::FootLift, Segment::Leg, "peak_ratio", &RefCtx::default()).unwrap();
        assert!(hip.peak_ratio < hr.value - hr.tolerance, "{} vs {hr:?}", hip.peak_ratio);
        let frac = r.couplings.iter().find(|c| c.axis == "shift_frac").unwrap().value;
        let rf = reference(Source::FootLift, Segment::Belly, "shift_frac", &RefCtx::default()).unwrap();
        assert!((frac - rf.value).abs() <= rf.tolerance, "{frac}");
    }

    #[test]
    fn the_feed_forward_reduces_the_dip_and_learning_the_gain_removes_it() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let p = Posture::arm_lift(2.0, false, 0.0);
        let off = b.solve(&p, &FeedForward { enabled: false, gain: 1.0 }, 100.0);
        let nominal = b.solve(&p, &FeedForward::default(), 100.0);
        // The nominal model assumes human proportions, so it is off; the SDS would learn measured / predicted.
        let learned_gain = nominal.shift_uncomp_m / nominal.ff_predicted_m;
        let learned = b.solve(&p, &FeedForward { enabled: true, gain: learned_gain }, 100.0);
        assert!(nominal.margin >= off.margin - 1e-6);
        assert!((learned.margin - learned.margin_noff) >= (nominal.margin - nominal.margin_noff) - 1e-6);
        assert!((learned.margin - (learned.margin_x.min(learned.margin_y))).abs() < 1e-4, "a learned gain leaves no dip");
        let c = learned.couplings.iter().find(|c| c.target == Segment::Foot && c.axis == "ankle_rise").unwrap();
        assert!(c.effort.unwrap() <= c.effort_noff.unwrap());
    }

    #[test]
    fn references_cover_the_named_cells_and_scale_with_load() {
        let ctx = RefCtx { load_ratio: 0.067, ..Default::default() };
        let r5 = reference(Source::ArmLift, Segment::Arm, "peak_ratio", &ctx).unwrap();
        assert!((r5.value - 0.59).abs() < 1e-3);
        let r0 = reference(Source::ArmLift, Segment::Arm, "peak_ratio", &RefCtx::default()).unwrap();
        assert!((r0.value - 0.15).abs() < 1e-3);
        assert!(reference(Source::ArmRaise, Segment::Hand, "grip_effort", &ctx).is_none(), "no human number for grip: baseline only");
        let up = reference(Source::Slope, Segment::Foot, "cop_shift_frac", &RefCtx { slope_deg: 10.0, ..Default::default() }).unwrap();
        assert!(up.value < 0.0, "toes-up moves the CoP back");
    }
}
