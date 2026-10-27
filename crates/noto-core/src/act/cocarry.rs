//! Carrying one object together: how the weight of a log, a rock or a
//! stretcher-like load splits between the robots holding it, what each robot's
//! joints, feet and grip make of its share, how the share moves with their
//! steps, whether the team can carry it at all, and what happens when one of
//! them lets go. Pure functions: the world places the robots and the object,
//! this module does the statics (`docs/research/coop.md`).
//!
//! * **Statics.** The object is rigid and the hands push vertically (they do
//!   not squeeze it along its length unless asked). Its CoM sits a share of
//!   the way along the grip line and `h_c` above it, perpendicular to the line;
//!   on an incline (the carriers' grounds at different heights) the CoM moves
//!   downhill of the line's middle by `h_c·sin θ`, so the lower carrier takes
//!   `W·((1 − a) + (h_c/L)·tan θ)` (moment balance about the lower grip). Two
//!   grips in a line, or three not in a line, are statically determinate; with
//!   more the hands are springs (their relative stiffness) and the object
//!   heaves and pitches (and rolls) on them: the compliance-weighted minimum
//!   norm, which also takes a hand held lower than the others.
//! * **Internal force.** A force `T` along an inclined grip line, equal and
//!   opposite at the two hands, keeps the object in equilibrium and moves
//!   `T·sin θ` of the vertical load from the lower carrier to the upper; the
//!   `T` that would even the shares is reported with whether the carriers'
//!   feet hold its horizontal part, and it is applied only on request.
//! * **Steps.** A walking carrier's hands rise and fall with its CoM (the
//!   gait's own excursion and cadence; a wheel does not bob). With rigid arms
//!   the beam heaves and pitches with the hands, so each hand feels
//!   `m·(g + z̈_G)` shared and `I_G·θ̈` about the CoM: walking in phase adds the
//!   heave to both, in anti-phase mostly a pitch.
//! * **Team.** The team carries at most `k_n·Σ m_lim` (the HSE guidance 0.85
//!   for any team, or the older 2/3 for two and 1/2 for three) and no more
//!   than its weakest member allows for its share (`m_lim,i / s_i`); four is
//!   the most that work together.
//! * **Members.** Each robot's share goes to `kin` as the force on both hands
//!   at the grip's height (joint peaks and the ZMP margin, at the gait's peak),
//!   to its continuous carry capacity on its ground and slope (its base's,
//!   `Body::capacity_on`, and its arms' continuous hold at the carry pose,
//!   the smaller), and to its grip: hands under the object hold the
//!   vertical by contact but slide along an inclined surface past
//!   `tan θ > μ/GRIP_SAFETY`; a squeezing grip must hold the whole share by
//!   friction with the safety factor.
//! * **Letting go.** When a member fails, the rest share again; if they cannot
//!   hold the object in equilibrium (a long object on one robot, a rock on two)
//!   the free end falls: the robot still holding feels `W·I_G/(I_G + m·d²)` at
//!   the instant, then its share of the object resting on the dropped end.
//!
//! Engine inputs only (Group A); the cooperation benchmark scores
//! with other sources.

use serde::Serialize;

use crate::balance::{Body, G, GRIP_SAFETY};
use crate::base::BaseKind;
use crate::hand::NotoConfig;
use crate::kin::{self, KinLoads, KinPose};
use crate::surface::{self, Surface};

use super::material::{self, ItemMaterial};
use super::{gait, grasp, ActEvent, Part};

/// Fraction of the summed individual capabilities a team handles, for any
/// team size (HSE, L23 Manual Handling Operations Regulations guidance, 4th
/// ed. 2016, para 148).
pub const HSE_TEAM_FACTOR: f32 = 0.85;
/// The older, conservative rule: 2/3 of the sum for two, 1/2 for three
/// (Network Rail's team handling page quoting HSE; ISO/TR 12295 team
/// multipliers 0.67 and 0.5, via a secondary summary).
pub const TEAM_FACTOR_2: f32 = 2.0 / 3.0;
pub const TEAM_FACTOR_3: f32 = 0.5;
/// Teams of more than four are unlikely to work (HSE L23 para 147).
pub const TEAM_MAX: usize = 4;
/// How far ahead of the hip the hands hold a carried object, metres (est:
/// the forearm about level in front of the trunk, as `grasp::hold_pose`).
pub const CARRY_REACH_M: f32 = 0.35;
/// Span between one robot's two hands along the object, metres (est: about
/// the shoulder width); a robot alone can hold a long object only with its
/// CoM between them.
pub const HANDS_SPAN_M: f32 = 0.30;
/// The hands are where the grip says within this, metres, else out of reach
/// (est: a numerical tolerance of the pose search, about a palm's half width).
const REACH_TOL_M: f32 = 0.04;
/// Two grips closer than this along the line hold the object at one point, metres (est: a numerical tolerance).
const SAME_GRIP_M: f32 = 1e-3;
/// Half the width of a palm under the object, metres (the drawn palm box of
/// `kin`, 0.07 m across): two or more hands under an object hold the roll
/// about their line only while its CoM stays above their palms.
pub const PALM_HALF_W_M: f32 = 0.035;

/// How a team's capacity is counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamRule {
    /// 0.85 of the sum for any team (HSE 2016).
    Hse2016,
    /// 2/3 for two, 1/2 for three or more (the older guidance).
    Conservative,
}

impl TeamRule {
    pub fn factor(self, n: usize) -> f32 {
        match (self, n) {
            (_, 0 | 1) => 1.0,
            (TeamRule::Hse2016, _) => HSE_TEAM_FACTOR,
            (TeamRule::Conservative, 2) => TEAM_FACTOR_2,
            (TeamRule::Conservative, _) => TEAM_FACTOR_3,
        }
    }
}

/// How the hands hold the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GripKind {
    /// Hands under it: the vertical load is carried by contact, friction only along an inclined surface.
    Under,
    /// Fingers round it: the whole share is held by friction.
    Squeeze,
}

/// The object carried together.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Load {
    pub material: ItemMaterial,
    pub mass_kg: f32,
    /// Length along the grip line, metres (the pitch inertia, the free end).
    pub length_m: f32,
    /// Thickness where the hands hold it (a log's diameter), metres.
    pub thick_m: f32,
    /// Where the CoM lies along the grip line, from the first grip toward the
    /// last (0.5: the middle); not used for grips around the object.
    pub com_along: f32,
    /// The CoM above the grip line (or the grips' plane), perpendicular to it, metres (`h_c`).
    pub com_up_m: f32,
    pub grip: GripKind,
}

impl Load {
    /// A log held at its ends: under it (the CoM a radius above the hands) or squeezed round it (at its axis).
    pub fn log(mass_kg: f32, length_m: f32, diameter_m: f32, under: bool) -> Load {
        Load { material: ItemMaterial::Wood, mass_kg, length_m, thick_m: diameter_m, com_along: 0.5, com_up_m: if under { 0.5 * diameter_m } else { 0.0 }, grip: if under { GripKind::Under } else { GripKind::Squeeze } }
    }

    /// A rock lifted from under its sides (the hands about a quarter of its size below its centre: est).
    pub fn rock(mass_kg: f32, size_m: f32) -> Load {
        Load { material: ItemMaterial::Stone, mass_kg, length_m: size_m, thick_m: size_m, com_along: 0.5, com_up_m: 0.25 * size_m, grip: GripKind::Under }
    }

    pub fn weight_n(&self) -> f32 {
        self.mass_kg * G
    }

    /// Pitch inertia about the CoM, kg·m² (a uniform bar of its length and thickness).
    pub fn pitch_inertia(&self) -> f32 {
        self.mass_kg * (self.length_m * self.length_m + self.thick_m * self.thick_m) / 12.0
    }
}

/// One robot holding the object.
#[derive(Clone, Debug)]
pub struct Carrier<'a> {
    pub body: &'a Body,
    /// Where its hands hold the object (their middle), world frame (x, y horizontal, z up), metres.
    pub grip: [f32; 3],
    /// The ground under it: its height, kind and slope along its way, degrees (uphill positive).
    pub ground_z_m: f32,
    pub ground: Surface,
    pub slope_deg: f32,
    /// Relative vertical stiffness of its hold (only the ratios matter, and only when the team is indeterminate).
    pub stiffness: f32,
    /// How far below the object's surface its hands would sit unloaded, metres (a shorter robot at full reach).
    pub hand_low_m: f32,
}

impl<'a> Carrier<'a> {
    /// A robot standing on the ground at `ground_z_m` holding at its own carry height at (`x`, `y`).
    pub fn at(body: &'a Body, x: f32, y: f32, ground_z_m: f32, ground: Surface) -> Carrier<'a> {
        Carrier { body, grip: [x, y, ground_z_m + carry_hand_z(body)], ground_z_m, ground, slope_deg: 0.0, stiffness: 1.0, hand_low_m: 0.0 }
    }

    /// Its hands' height above its own ground, metres.
    pub fn hand_z(&self) -> f32 {
        self.grip[2] - self.ground_z_m
    }
}

/// Height of the hands in the two-arm hold above the robot's ground, metres.
pub fn carry_hand_z(body: &Body) -> f32 {
    kin::solve_fast(body, &grasp::hold_pose(true, 0.0), &KinLoads::default()).hands[0][2]
}

/// The pose that holds a shared object at `hand_z` above the robot's ground, [`CARRY_REACH_M`] ahead, both arms.
pub fn carry_pose(body: &Body, hand_z: f32) -> KinPose {
    grasp::arms_to(body, &grasp::hold_pose(true, 0.0), [CARRY_REACH_M, hand_z], true)
}

/// The joints that hold a load in the hands against the trunk: shoulders,
/// elbows, the waist and a torso lift (the legs or the drive carry it on as
/// part of the body's own capacity).
fn holds_in_hands(name: &str) -> bool {
    name.starts_with("shoulder") || name.starts_with("elbow") || name == "waist_pitch" || name == "torso_lift"
}

/// What a robot holds in both hands at the carry pose continuously, kg: the
/// largest load that keeps every arm, waist and lift joint within its
/// continuous rating (bisection on `kin`).
pub fn hold_capacity_kg(body: &Body, hand_z: f32) -> f32 {
    let pose = carry_pose(body, hand_z);
    let ok = |kg: f32| kin::solve_fast(body, &pose, &KinLoads::holding(kg, true, 0.0)).joints.iter().filter(|j| !j.structural && holds_in_hands(&j.name)).all(|j| j.effort <= 1.0);
    if !ok(0.0) {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0f32, 200.0f32);
    if ok(hi) {
        return hi;
    }
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// What one member carries continuously in its hands on its ground and
/// slope, kg: its base's capacity (`Body::capacity_on`: the legs' knees, the
/// drive) and its arms' continuous hold ([`hold_capacity_kg`]), the smaller.
pub fn member_capacity_kg(body: &Body, ground: Surface, slope_deg: f32, hand_z: f32) -> f32 {
    body.capacity_on(&body.base.spec(), ground, slope_deg).min(hold_capacity_kg(body, hand_z))
}

/// The rigid-object statics of a set of grips.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Statics {
    /// Vertical force each grip holds, N (negative: it must pull the object down).
    pub forces: Vec<f32>,
    /// The CoM, world frame.
    pub com: [f32; 3],
    /// The grip line's incline, degrees (positive rising from the first grip toward the last), 0 for grips around the object.
    pub incline_deg: f32,
    /// Grips in a line (a long object) or around it.
    pub collinear: bool,
    /// The shares follow from equilibrium alone (two in a line, three around).
    pub determinate: bool,
    /// The grips can hold the object in equilibrium (the CoM within their support).
    pub balanced: bool,
    /// The roll moment a grip line must hold when the CoM is off it, N·m.
    pub roll_nm: f32,
    /// Distance of each grip along the line from the CoM (collinear), metres.
    pub along: Vec<f32>,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt().max(1e-9);
    [a[0] / n, a[1] / n, a[2] / n]
}

fn solve_lin(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let s: f32 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

/// The first and last grip of the grip line: the two furthest apart, in order.
fn line_ends(grips: &[[f32; 3]]) -> (usize, usize) {
    let (mut ia, mut ib, mut best) = (0usize, 0usize, -1.0f32);
    for i in 0..grips.len() {
        for j in i + 1..grips.len() {
            let d = dot(sub(grips[j], grips[i]), sub(grips[j], grips[i]));
            if d > best {
                best = d;
                ia = i;
                ib = j;
            }
        }
    }
    (ia.min(ib), ia.max(ib))
}

/// Grips off the line of the two furthest apart by no more than this lie in
/// a line, metres (est: a geometric tolerance, about a hand's width).
const LINE_TOL_M: f32 = 0.05;

/// Whether `grips` lie in a line (a long object) rather than around it.
fn in_line(grips: &[[f32; 3]]) -> bool {
    if grips.len() <= 2 {
        return true;
    }
    let (ia, ib) = line_ends(grips);
    let e = unit(sub(grips[ib], grips[ia]));
    grips.iter().all(|p| {
        let d = sub(*p, grips[ia]);
        let a = dot(d, e);
        let r = sub(d, [e[0] * a, e[1] * a, e[2] * a]);
        dot(r, r).sqrt() < LINE_TOL_M
    })
}

/// Where the object's CoM is when all of `grips` hold it (world frame): a
/// share [`Load::com_along`] along the grip line and [`Load::com_up_m`]
/// perpendicular above it; for grips around the object, above their
/// centroid along the up normal of their plane; for one grip, above it.
pub fn object_com(load: &Load, grips: &[[f32; 3]]) -> [f32; 3] {
    let n = grips.len();
    let up = [0.0f32, 0.0, 1.0];
    if n == 0 {
        return [0.0; 3];
    }
    if n == 1 {
        let g = grips[0];
        return [g[0], g[1], g[2] + load.com_up_m];
    }
    let off = |p: [f32; 3], nrm: [f32; 3]| [p[0] + load.com_up_m * nrm[0], p[1] + load.com_up_m * nrm[1], p[2] + load.com_up_m * nrm[2]];
    if in_line(grips) {
        let (ia, ib) = line_ends(grips);
        let line = sub(grips[ib], grips[ia]);
        let e = unit(line);
        let nrm = unit(sub(up, [e[0] * e[2], e[1] * e[2], e[2] * e[2]]));
        let a = load.com_along;
        off([grips[ia][0] + a * line[0], grips[ia][1] + a * line[1], grips[ia][2] + a * line[2]], nrm)
    } else {
        // The grips' plane (Newell's normal), up side; the CoM over their centroid.
        let mut nv = [0.0f32; 3];
        for i in 0..n {
            let (p, q) = (grips[i], grips[(i + 1) % n]);
            nv[0] += (p[1] - q[1]) * (p[2] + q[2]);
            nv[1] += (p[2] - q[2]) * (p[0] + q[0]);
            nv[2] += (p[0] - q[0]) * (p[1] + q[1]);
        }
        let mut nrm = unit(nv);
        if nrm[2] < 0.0 {
            nrm = [-nrm[0], -nrm[1], -nrm[2]];
        }
        let c = grips.iter().fold([0.0f32; 3], |s, p| [s[0] + p[0] / n as f32, s[1] + p[1] / n as f32, s[2] + p[2] / n as f32]);
        off(c, nrm)
    }
}

/// The vertical forces at `grips` (world frame) holding `load`, with the
/// hands as springs of relative stiffness `k` sitting `low` below the object
/// unloaded (they matter only when the grips are more than equilibrium fixes).
pub fn statics(load: &Load, grips: &[[f32; 3]], k: &[f32], low: &[f32]) -> Statics {
    statics_at(load, grips, object_com(load, grips), k, low)
}

/// [`statics`] with the object's CoM at `com` (world frame): after a member
/// lets go, the object's CoM stays where the whole team held it.
pub fn statics_at(load: &Load, grips: &[[f32; 3]], com: [f32; 3], k: &[f32], low: &[f32]) -> Statics {
    let w = load.weight_n();
    let n = grips.len();
    if n == 0 {
        return Statics { forces: vec![], com, incline_deg: 0.0, collinear: true, determinate: true, balanced: false, roll_nm: 0.0, along: vec![] };
    }
    let (ia, ib) = line_ends(grips);
    let e = unit(sub(grips[ib], grips[ia]));
    let horiz = [e[0], e[1], 0.0];
    let hlen = dot(horiz, horiz).sqrt();
    let collinear = in_line(grips);
    let incline_deg = if n >= 2 && collinear { e[2].atan2(hlen).to_degrees() } else { 0.0 };
    // A lone robot holds it between its two hands.
    if n == 1 {
        let d = ((com[0] - grips[0][0]).powi(2) + (com[1] - grips[0][1]).powi(2)).sqrt();
        let balanced = d <= 0.5 * HANDS_SPAN_M;
        return Statics { forces: vec![w], com, incline_deg, collinear: true, determinate: true, balanced, roll_nm: w * d, along: vec![0.0] };
    }
    let kk: Vec<f32> = (0..n).map(|i| k.get(i).copied().unwrap_or(1.0).max(1e-6)).collect();
    let dl: Vec<f32> = (0..n).map(|i| -low.get(i).copied().unwrap_or(0.0)).collect();
    let (forces, determinate, roll, along) = if collinear {
        // Along the line's horizontal direction from the CoM.
        let h = if hlen > 1e-6 { [horiz[0] / hlen, horiz[1] / hlen, 0.0] } else { [1.0, 0.0, 0.0] };
        let s: Vec<f32> = grips.iter().map(|p| dot(sub(*p, com), h)).collect();
        let side = [-h[1], h[0], 0.0];
        let roll = w * dot(sub(com, grips[ia]), side);
        // F_i = k_i·(Δ_i − z0 − a·s_i): Σ F = W, Σ F·s = 0.
        let (sk, sks, skss) = (kk.iter().sum::<f32>(), (0..n).map(|i| kk[i] * s[i]).sum::<f32>(), (0..n).map(|i| kk[i] * s[i] * s[i]).sum::<f32>());
        let (skd, skds) = ((0..n).map(|i| kk[i] * dl[i]).sum::<f32>(), (0..n).map(|i| kk[i] * dl[i] * s[i]).sum::<f32>());
        let x = solve_lin(vec![vec![sk, sks], vec![sks, skss]], vec![skd - w, skds]);
        let f = match x {
            Some(x) => (0..n).map(|i| kk[i] * (dl[i] - x[0] - x[1] * s[i])).collect(),
            None => vec![w / n as f32; n],
        };
        (f, n == 2, roll, s)
    } else {
        let d: Vec<[f32; 2]> = grips.iter().map(|p| [p[0] - com[0], p[1] - com[1]]).collect();
        let sum = |f: &dyn Fn(usize) -> f32| (0..n).map(f).sum::<f32>();
        let a = vec![
            vec![sum(&|i| kk[i]), sum(&|i| kk[i] * d[i][0]), sum(&|i| kk[i] * d[i][1])],
            vec![sum(&|i| kk[i] * d[i][0]), sum(&|i| kk[i] * d[i][0] * d[i][0]), sum(&|i| kk[i] * d[i][0] * d[i][1])],
            vec![sum(&|i| kk[i] * d[i][1]), sum(&|i| kk[i] * d[i][0] * d[i][1]), sum(&|i| kk[i] * d[i][1] * d[i][1])],
        ];
        let b = vec![sum(&|i| kk[i] * dl[i]) - w, sum(&|i| kk[i] * dl[i] * d[i][0]), sum(&|i| kk[i] * dl[i] * d[i][1])];
        let f = match solve_lin(a, b) {
            Some(x) => (0..n).map(|i| kk[i] * (dl[i] - x[0] - x[1] * d[i][0] - x[2] * d[i][1])).collect(),
            None => vec![w / n as f32; n],
        };
        (f, n == 3, 0.0, vec![])
    };
    // Hands under the object cannot pull it down: a negative force means it
    // tips off them; a CoM beside their line rolls it off their palms. Fingers
    // round it hold both (a squeeze pulls and twists).
    let rolls_off = collinear && roll.abs() > w * PALM_HALF_W_M;
    let balanced = load.grip == GripKind::Squeeze || (forces.iter().all(|f| *f >= -1e-3 * w) && !rolls_off);
    Statics { forces, com, incline_deg, collinear, determinate, balanced, roll_nm: roll, along }
}

/// Walking in step with each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Both CoMs rise together (a pace).
    InPhase,
    /// Half a step apart (a trot).
    AntiPhase,
    /// The larger of the two (the controller does not sync the gaits).
    Worst,
}

/// The gait-phase load fluctuation of two carriers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Dynamic {
    /// Step frequency of the walking carriers, Hz.
    pub step_hz: f32,
    /// Each carrier's hand bob amplitude, metres (0 for wheels).
    pub amp_m: Vec<f32>,
    /// Largest change of each member's force over a cycle, in phase and half a step apart, N.
    pub in_phase_n: Vec<f32>,
    pub anti_phase_n: Vec<f32>,
}

/// Options of a shared carry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CarryOpts {
    pub rule: TeamRule,
    pub phase: Phase,
    /// Speed of the formation, m/s (`None`: the slowest member's limit with its share).
    pub speed_mps: Option<f32>,
    /// Apply the internal force that evens two carriers' shares on an incline (as far as their feet hold it).
    pub even_axial: bool,
}

impl Default for CarryOpts {
    fn default() -> Self {
        CarryOpts { rule: TeamRule::Hse2016, phase: Phase::Worst, speed_mps: None, even_axial: false }
    }
}

/// One member's share and what it makes of it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MemberShare {
    pub index: usize,
    /// Vertical force it holds, N, and its share of the weight.
    pub f_n: f32,
    pub share: f32,
    /// The force with the gait's peak on top, N.
    pub f_peak_n: f32,
    /// Its load and peak load as masses, kg.
    pub load_kg: f32,
    pub peak_kg: f32,
    /// What it carries continuously in its hands on its ground and slope, kg ([`member_capacity_kg`]).
    pub capacity_kg: f32,
    /// Friction its hold needs (with the safety factor) over what the pads give (≤ 1 holds).
    pub grip_use: f32,
    /// The worst joint's torque over its peak and the ZMP margin, at the peak load.
    pub peak_ratio: f32,
    pub margin: f32,
    /// Its hands reach the grip.
    pub reaches: bool,
    pub ok: bool,
    pub why_ko: Option<String>,
    /// The pose and the loads it holds its share with (for `kin`, the world, the views).
    pub pose: KinPose,
    pub loads: KinLoads,
}

/// A shared carry worked out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoCarry {
    pub statics: Statics,
    pub members: Vec<MemberShare>,
    /// The internal force along the grip line that would even two carriers' vertical shares, N, and whether their feet hold its horizontal part.
    pub axial_even_n: f32,
    pub axial_even_ok: bool,
    /// The internal force applied, N (0 unless `even_axial`).
    pub axial_n: f32,
    pub dynamic: Option<Dynamic>,
    pub speed_mps: f32,
    /// The team's limit by the rule, by its weakest member for its share, and the smaller, kg.
    pub team_limit_kg: f32,
    pub weakest_limit_kg: f32,
    pub limit_kg: f32,
    pub rule: TeamRule,
    pub ok: bool,
    pub events: Vec<ActEvent>,
    pub notes: Vec<String>,
}

impl CoCarry {
    /// The lower carrier's share over the upper's (two carriers), the larger over the smaller otherwise.
    pub fn share_ratio(&self) -> f32 {
        let f: Vec<f32> = self.members.iter().map(|m| m.f_n).collect();
        let hi = f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lo = f.iter().copied().fold(f32::INFINITY, f32::min);
        hi / lo.max(1e-6)
    }
}

fn hand_of(body: &Body) -> Option<crate::hand::Hand> {
    let mut cfg = NotoConfig::preset(body.fingers);
    cfg.base = body.base;
    cfg.build().ok()
}

/// Each walking carrier's hand bob (amplitude, m) and its step frequency at `speed`.
fn bob(c: &Carrier, speed: f32) -> (f32, f32) {
    if c.body.base != BaseKind::HumanFoot {
        return (0.0, 0.0);
    }
    let mu = surface::available_mu(c.body.base, c.ground.props());
    let g = gait::plan(c.body, speed, mu, c.slope_deg);
    (0.5 * g.com_amp_m, 1.0 / g.step_s.max(1e-3))
}

/// Peak change of two carriers' forces over a step cycle, the hands bobbing
/// with amplitudes `amp` at `hz` with phase lag `lag` (radians), the grips at
/// `s` along the line from the CoM. Two grips at one point (closer than
/// [`SAME_GRIP_M`]) cannot turn the object: they move it as one, each taking
/// half of `m·a` (the least-norm split of the indeterminate pair).
fn fluctuation(load: &Load, s: [f32; 2], amp: [f32; 2], hz: f32, lag: f32) -> [f32; 2] {
    if hz <= 0.0 || (amp[0] == 0.0 && amp[1] == 0.0) {
        return [0.0; 2];
    }
    let m = load.mass_kg;
    let i_g = load.pitch_inertia();
    let w = std::f32::consts::TAU * hz;
    if (s[1] - s[0]).abs() < SAME_GRIP_M {
        let peak = (0..64)
            .map(|k| {
                let t = k as f32 / 64.0 / hz;
                0.5 * m * (0.5 * (amp[0] * (w * t).sin() + amp[1] * (w * t + lag).sin()) * w * w).abs()
            })
            .fold(0.0f32, f32::max);
        return [peak, peak];
    }
    let lam = (0.0 - s[0]) / (s[1] - s[0]);
    let mut peak = [0.0f32; 2];
    for k in 0..64 {
        let t = k as f32 / 64.0 / hz;
        let a1 = -amp[0] * w * w * (w * t).sin();
        let a2 = -amp[1] * w * w * (w * t + lag).sin();
        let a_g = a1 * (1.0 - lam) + a2 * lam;
        let th_dd = (a2 - a1) / (s[1] - s[0]);
        // F1 + F2 = m·a_G and F2·s2 + F1·s1 = I_G·θ̈ (moments about the CoM, dynamic parts only).
        let f2 = (i_g * th_dd - s[0] * m * a_g) / (s[1] - s[0]);
        let f1 = m * a_g - f2;
        peak[0] = peak[0].max(f1.abs());
        peak[1] = peak[1].max(f2.abs());
    }
    peak
}

/// Works out a shared carry: the statics, each member's share and checks, the
/// gait fluctuation (two carriers), the internal force, and the team's limit.
pub fn share(load: &Load, carriers: &[Carrier], opts: &CarryOpts) -> CoCarry {
    let grips: Vec<[f32; 3]> = carriers.iter().map(|c| c.grip).collect();
    share_at(load, carriers, object_com(load, &grips), opts)
}

/// [`share`] with the object's CoM at `com` (world frame).
pub fn share_at(load: &Load, carriers: &[Carrier], com: [f32; 3], opts: &CarryOpts) -> CoCarry {
    let n = carriers.len();
    let w = load.weight_n();
    let grips: Vec<[f32; 3]> = carriers.iter().map(|c| c.grip).collect();
    let k: Vec<f32> = carriers.iter().map(|c| c.stiffness).collect();
    let low: Vec<f32> = carriers.iter().map(|c| c.hand_low_m).collect();
    let st = statics_at(load, &grips, com, &k, &low);
    let mut notes = Vec::new();
    let mut events = Vec::new();
    let mut forces = st.forces.clone();
    // The internal force that evens two carriers' shares on an incline.
    let th = st.incline_deg.to_radians();
    let (mut axial_even, mut axial_ok, mut axial) = (0.0f32, true, 0.0f32);
    if n == 2 && th.abs() > 1e-3 {
        axial_even = (forces[0] - forces[1]) / (2.0 * th.sin());
        // Its horizontal part pushes each carrier's feet (or wheels) over the
        // ground, which holds it by friction under the robot's weight and its share.
        let hold = |i: usize| {
            let c = &carriers[i];
            surface::available_mu(c.body.base, c.ground.props()) * (c.body.mass_kg() * G + forces[i].max(0.0))
        };
        let need = (axial_even * th.cos()).abs();
        axial_ok = need <= hold(0).min(hold(1));
        if opts.even_axial {
            let cap = hold(0).min(hold(1)) / th.cos().abs().max(1e-3);
            axial = axial_even.clamp(-cap, cap);
            forces[0] -= axial * th.sin();
            forces[1] += axial * th.sin();
            notes.push(format!("축 방향 내력 {:.0} N으로 아래쪽 하중을 위쪽으로 넘김", axial));
        }
    }
    // Speed: the slowest member with its share.
    let speed = opts.speed_mps.unwrap_or_else(|| {
        carriers
            .iter()
            .zip(&forces)
            .map(|(c, f)| c.body.drive_limits(&c.body.base.spec(), (f / G).max(0.0)).max_speed_mps)
            .fold(f32::INFINITY, f32::min)
            .min(1.0)
    });
    // The gait's fluctuation (two carriers in a line).
    let dynamic = (n == 2 && st.collinear).then(|| {
        let b: Vec<(f32, f32)> = carriers.iter().map(|c| bob(c, speed)).collect();
        let hz = b.iter().map(|x| x.1).fold(0.0f32, f32::max);
        let amp = [b[0].0, b[1].0];
        let s = [st.along[0], st.along[1]];
        let ip = fluctuation(load, s, amp, hz, 0.0);
        let ap = fluctuation(load, s, amp, hz, std::f32::consts::PI);
        Dynamic { step_hz: hz, amp_m: amp.to_vec(), in_phase_n: ip.to_vec(), anti_phase_n: ap.to_vec() }
    });
    let extra = |i: usize| -> f32 {
        dynamic.as_ref().map_or(0.0, |d| match opts.phase {
            Phase::InPhase => d.in_phase_n[i],
            Phase::AntiPhase => d.anti_phase_n[i],
            Phase::Worst => d.in_phase_n[i].max(d.anti_phase_n[i]),
        })
    };
    // Each member.
    let mu_pad = material::pad_on(load.material).mean;
    let mut members = Vec::new();
    for (i, c) in carriers.iter().enumerate() {
        let f = forces[i];
        let f_peak = f + extra(i);
        let hz = c.hand_z();
        let pose = carry_pose(c.body, hz);
        let loads = KinLoads { hand_force: [[0.0, 0.0, -0.5 * f_peak], [0.0, 0.0, -0.5 * f_peak]], ..KinLoads::default() };
        let r = kin::solve(c.body, &pose, &loads);
        let reaches = (r.hands[0][2] - hz).abs() <= REACH_TOL_M;
        let capacity = member_capacity_kg(c.body, c.ground, c.slope_deg, hz);
        let grip_use = match load.grip {
            // Hands under an inclined surface: the share's component along it against the pads' friction.
            GripKind::Under => GRIP_SAFETY * th.tan().abs() / mu_pad.max(1e-3),
            GripKind::Squeeze => match hand_of(c.body) {
                Some(h) => {
                    let cap = grasp::capacity(&h, [load.thick_m; 3], (f_peak / G).max(0.0), mu_pad);
                    GRIP_SAFETY * f_peak.abs() / (mu_pad * 2.0 * cap.squeeze_n).max(1e-3)
                }
                None => f32::INFINITY,
            },
        };
        let peak_ratio = r.worst().map_or(0.0, |j| j.ratio);
        let why = if f < -1e-3 * w && load.grip == GripKind::Under {
            Some(format!("물체를 아래로 눌러야 함 ({:.0} N): 손을 받친 채로는 반대쪽으로 넘어감", f))
        } else if !reaches {
            Some(format!("손이 {:.2} m 높이에 닿지 않음 ({:.2} m까지)", hz, r.hands[0][2]))
        } else if f / G > capacity {
            Some(format!("몫 {:.1} kg > 계속 나를 수 있는 {:.1} kg", f / G, capacity))
        } else if peak_ratio > 1.0 {
            Some(format!("걸음 최고 하중 {:.1} kg에서 {}", f_peak / G, r.limiting.clone().unwrap_or_default()))
        } else if r.margin < 0.0 {
            Some(format!("균형 여유 {:.2} < 0", r.margin))
        } else if grip_use > 1.0 {
            Some(format!("손이 미끄러짐: 필요 마찰 {:.2} > 패드 {:.2}", grip_use * mu_pad, mu_pad))
        } else {
            None
        };
        if let Some(wk) = &why {
            events.push(ActEvent::new(0.0, "overload_share", format!("{}번 로봇: {wk}", i + 1), &[Part::Hand, Part::Waist], Some(f / G), false));
        }
        members.push(MemberShare { index: i, f_n: f, share: f / w.max(1e-6), f_peak_n: f_peak, load_kg: f / G, peak_kg: f_peak / G, capacity_kg: capacity, grip_use, peak_ratio, margin: r.margin, reaches, ok: why.is_none(), why_ko: why, pose, loads });
    }
    // The team.
    let rule_limit = opts.rule.factor(n) * members.iter().map(|m| m.capacity_kg).sum::<f32>();
    let weakest = members.iter().filter(|m| m.share > 1e-3).map(|m| m.capacity_kg / m.share).fold(f32::INFINITY, f32::min);
    let limit = rule_limit.min(weakest);
    if n > TEAM_MAX {
        notes.push(format!("{n}대: 넷을 넘는 팀은 잘 맞추기 어려움"));
    }
    if !st.balanced {
        events.push(ActEvent::new(0.0, "co_drop", format!("잡은 곳들이 무게중심을 받치지 못해 물체가 넘어감 (몫 {:?})", st.forces.iter().map(|f| (f / G * 10.0).round() / 10.0).collect::<Vec<_>>()), &[Part::Hand], None, false));
    }
    if load.mass_kg > limit + 1e-3 {
        events.push(ActEvent::new(0.0, "overload_share", format!("팀 한계 {:.1} kg (규칙 {:.1} · 가장 약한 로봇 {:.1}) < 물체 {:.1} kg", limit, rule_limit, weakest, load.mass_kg), &[Part::Hand, Part::Waist], Some(load.mass_kg / limit.max(1e-3)), false));
    }
    let ok = st.balanced && members.iter().all(|m| m.ok) && load.mass_kg <= limit + 1e-3;
    if let Some(d) = &dynamic {
        notes.push(format!("걸음 {:.2} Hz: 같은 위상 ±{:.0}/{:.0} N, 반대 위상 ±{:.0}/{:.0} N", d.step_hz, d.in_phase_n[0], d.in_phase_n[1], d.anti_phase_n[0], d.anti_phase_n[1]));
    }
    CoCarry { statics: st, members, axial_even_n: axial_even, axial_even_ok: axial_ok, axial_n: axial, dynamic, speed_mps: speed, team_limit_kg: rule_limit, weakest_limit_kg: weakest, limit_kg: limit, rule: opts.rule, ok, events, notes }
}

/// What happens when member `i` lets go (or falls).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LetGo {
    /// The rest's carry when they can still hold the object.
    pub after: Option<CoCarry>,
    /// The object (or one end of it) falls.
    pub dropped: bool,
    /// The force each remaining member feels at the instant of the release, N (a member holding a long object alone: `W·I_G/(I_G + m·d²)`).
    pub instant_n: Vec<f32>,
    /// With the free end on the ground: the share the remaining holder carries, and whether it can.
    pub end_on_ground_share: Option<f32>,
    pub end_held: bool,
    /// Height the free end falls from, metres.
    pub end_drop_m: f32,
    pub events: Vec<ActEvent>,
}

/// Member `i` lets go: the rest share again; if they cannot hold the object
/// in equilibrium the free end falls (a long object left on one robot rests
/// on its dropped end with the robot holding the other).
pub fn let_go(load: &Load, carriers: &[Carrier], i: usize, opts: &CarryOpts) -> LetGo {
    let rest: Vec<Carrier> = carriers.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, c)| c.clone()).collect();
    let mut events = vec![ActEvent::new(0.0, "co_drop", format!("{}번 로봇이 놓음: 나머지 {}대가 다시 나눔", i + 1, rest.len()), &[Part::Hand], None, false)];
    let w = load.weight_n();
    let end_drop = carriers.get(i).map_or(0.0, |c| c.hand_z());
    if rest.is_empty() {
        events.push(ActEvent::new(0.0, "co_drop", "잡은 로봇이 없어 물체가 떨어짐", &[Part::Hand], Some(end_drop), false));
        return LetGo { after: None, dropped: true, instant_n: vec![], end_on_ground_share: None, end_held: false, end_drop_m: end_drop, events };
    }
    // The object stays where the whole team held it.
    let com = object_com(load, &carriers.iter().map(|c| c.grip).collect::<Vec<_>>());
    let after = share_at(load, &rest, com, opts);
    if after.statics.balanced {
        let held = after.ok;
        if !held {
            events.push(ActEvent::new(0.0, "overload_share", format!("나머지가 받치지만 한계를 넘음 (한계 {:.1} kg)", after.limit_kg), &[Part::Hand, Part::Waist], Some(load.mass_kg), false));
        }
        let instant = after.members.iter().map(|m| m.f_n).collect();
        return LetGo { after: Some(after), dropped: false, instant_n: instant, end_on_ground_share: None, end_held: held, end_drop_m: 0.0, events };
    }
    // The object turns off the rest: a long object held by one robot pivots on its hands until the free end lands.
    let mut instant = Vec::new();
    let mut end_share = None;
    let mut end_held = false;
    if rest.len() == 1 && in_line(&carriers.iter().map(|c| c.grip).collect::<Vec<_>>()) {
        let h = &rest[0];
        let full = statics(load, &carriers.iter().map(|c| c.grip).collect::<Vec<_>>(), &[], &[]);
        let holder = (0..carriers.len()).find(|j| *j != i).unwrap_or(0);
        let s_h = full.along.get(holder).copied().unwrap_or(0.0);
        let s_free = full.along.get(i).copied().unwrap_or(0.0);
        let d = s_h.abs();
        let i_g = load.pitch_inertia();
        instant.push(w * i_g / (i_g + load.mass_kg * d * d));
        // The free end: past the dropped grip by the object's overhang.
        let span = (full.along.iter().copied().fold(f32::NEG_INFINITY, f32::max) - full.along.iter().copied().fold(f32::INFINITY, f32::min)).max(1e-3);
        let over = 0.5 * (load.length_m - span).max(0.0);
        let s_end = s_free + s_free.signum() * over;
        let share_h = (s_end / (s_end - s_h)).abs().min(1.0);
        end_share = Some(share_h);
        let cap = member_capacity_kg(h.body, h.ground, h.slope_deg, h.hand_z());
        end_held = share_h * load.mass_kg <= cap;
        events.push(ActEvent::new(
            0.0,
            "co_drop",
            format!("한쪽 끝이 {:.2} m에서 떨어져 땅에 닿음: 남은 로봇이 순간 {:.0} N, 그 뒤 {:.0}% ({:.1} kg){}", end_drop, instant[0], share_h * 100.0, share_h * load.mass_kg, if end_held { "를 받침" } else { "도 못 받쳐 놓침" }),
            &[Part::Hand, Part::Waist],
            Some(share_h * load.mass_kg),
            false,
        ));
    } else {
        events.push(ActEvent::new(0.0, "co_drop", format!("나머지 {}대가 무게중심을 받치지 못해 물체가 떨어짐", rest.len()), &[Part::Hand], Some(end_drop), false));
    }
    LetGo { after: None, dropped: true, instant_n: instant, end_on_ground_share: end_share, end_held, end_drop_m: end_drop, events }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn a_level_log_splits_evenly_and_a_slope_loads_the_lower_carrier() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let l = 2.0f32;
        let log = Load::log(8.0, 2.4, 0.2, false);
        let level = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!((level.members[0].share - 0.5).abs() < 1e-3 && level.statics.determinate, "{:?}", level.members.iter().map(|m| m.share).collect::<Vec<_>>());
        // A 10° slope: the upper carrier stands L·tan θ higher. Held at its axis (h_c = 0) the split stays even.
        let dz = l * 10f32.to_radians().tan();
        let slope = |h_c: f32| {
            let load = Load { com_up_m: h_c, ..log };
            share(&load, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, dz, Surface::Firm)], &CarryOpts::default())
        };
        let s0 = slope(0.0);
        assert!((s0.members[0].share - 0.5).abs() < 1e-3, "{}", s0.members[0].share);
        assert!((s0.statics.incline_deg - 10.0).abs() < 0.1);
        // With the CoM h_c above the line, the lower takes ½ + h_c·tan θ / L (L along the line).
        for h_c in [0.1f32, 0.3, 0.5] {
            let s = slope(h_c);
            let along = l / 10f32.to_radians().cos();
            let expect = 0.5 + h_c * 10f32.to_radians().tan() / along;
            assert!((s.members[0].share - expect).abs() < 2e-3, "h_c {h_c}: {} vs {expect}", s.members[0].share);
        }
        assert!(slope(0.5).members[0].share > slope(0.1).members[0].share);
        // The internal force that would even them, and it does when applied.
        let s = slope(0.5);
        assert!(s.axial_even_n > 0.0 && s.axial_even_ok);
        let ev = share(&Load { com_up_m: 0.5, ..log }, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, dz, Surface::Firm)], &CarryOpts { even_axial: true, ..CarryOpts::default() });
        assert!((ev.members[0].share - 0.5).abs() < 1e-3, "{}", ev.members[0].share);
    }

    #[test]
    fn two_walkers_gripping_one_point_share_its_bob_without_nan() {
        // Both grip the rock's top at the same point: the pair cannot turn it, the
        // gait's bob is shared, and every number stays finite (a check that
        // compares with NaN would pass anything).
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let rock = Load::rock(4.0, 0.4);
        let s = share(&rock, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        for m in &s.members {
            assert!(m.f_n.is_finite() && m.f_peak_n.is_finite() && m.f_peak_n >= m.f_n - 1e-3, "{m:?}");
        }
        let d = s.dynamic.as_ref().expect("two walkers in a line bob");
        assert!(d.in_phase_n.iter().chain(&d.anti_phase_n).all(|f| f.is_finite()), "{d:?}");
        assert!((d.in_phase_n[0] - d.in_phase_n[1]).abs() < 1e-3);
    }

    #[test]
    fn three_around_a_rock_are_determinate_and_four_in_a_line_share_by_stiffness() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let rock = Load::rock(12.0, 0.5);
        let r = 0.3f32;
        let around: Vec<Carrier> = (0..3).map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 3.0;
            Carrier::at(&b, r * a.cos(), r * a.sin(), 0.0, Surface::Firm)
        }).collect();
        let s = share(&rock, &around, &CarryOpts::default());
        assert!(!s.statics.collinear && s.statics.determinate);
        assert!(s.members.iter().all(|m| (m.share - 1.0 / 3.0).abs() < 1e-3), "{:?}", s.members.iter().map(|m| m.share).collect::<Vec<_>>());
        // Four on a log in a line: equal springs share evenly; a stiffer pair at the ends takes more.
        let log = Load::log(16.0, 4.0, 0.2, false);
        let line = |k: [f32; 4]| -> Vec<f32> {
            let cs: Vec<Carrier> = (0..4).map(|i| Carrier { stiffness: k[i], ..Carrier::at(&b, i as f32, 0.0, 0.0, Surface::Firm) }).collect();
            share(&log, &cs, &CarryOpts::default()).members.iter().map(|m| m.share).collect()
        };
        let eq = line([1.0; 4]);
        assert!(eq.iter().all(|x| (x - 0.25).abs() < 1e-3), "{eq:?}");
        let stiff = line([2.0, 1.0, 1.0, 2.0]);
        assert!(stiff[0] > 0.25 && stiff[1] < 0.25 && (stiff.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{stiff:?}");
        // A hand held lower than the rest unloads that robot.
        let cs: Vec<Carrier> = (0..4).map(|i| Carrier { hand_low_m: if i == 1 { 0.01 } else { 0.0 }, stiffness: 2.0e4, ..Carrier::at(&b, i as f32, 0.0, 0.0, Surface::Firm) }).collect();
        let low = share(&log, &cs, &CarryOpts::default());
        assert!(low.members[1].share < 0.25, "{:?}", low.members.iter().map(|m| m.share).collect::<Vec<_>>());
    }

    #[test]
    fn a_legged_pair_carries_what_neither_can_alone() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let cap = member_capacity_kg(&b, Surface::Firm, 0.0, carry_hand_z(&b));
        assert!(cap < 6.0 && cap > 1.0, "mid legs carry {cap} kg");
        // A log a little heavier than one robot carries continuously.
        let log = Load::log(1.4 * cap, 2.0, 0.15, false);
        let alone = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!(!alone.ok, "one robot: {:?}", alone.members[0].why_ko);
        let pair = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!(pair.ok, "{:?} {:?}", pair.members.iter().map(|m| m.why_ko.clone()).collect::<Vec<_>>(), pair.events);
        // The team rule: 0.85 of the sum, or 2/3 conservatively.
        assert!((pair.team_limit_kg - 0.85 * 2.0 * cap).abs() < 1e-3);
        let cons = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts { rule: TeamRule::Conservative, ..CarryOpts::default() });
        assert!(cons.limit_kg < pair.limit_kg);
        // Walking in phase heaves the log: more than half a step apart, which mostly pitches it.
        let d = pair.dynamic.as_ref().unwrap();
        assert!(d.amp_m[0] > 0.0 && d.in_phase_n[0] > d.anti_phase_n[0], "{d:?}");
        // A wheeled partner does not bob.
        let w = body(Tier::Mid, BaseKind::Wheel);
        let mixed = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&w, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert_eq!(mixed.dynamic.as_ref().unwrap().amp_m[1], 0.0);
    }

    #[test]
    fn letting_go_drops_the_free_end_onto_the_ground() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let log = Load::log(6.0, 2.0, 0.15, false);
        let cs = [Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)];
        let g = let_go(&log, &cs, 1, &CarryOpts::default());
        assert!(g.dropped && g.end_drop_m > 0.5);
        // A uniform bar held at one end feels a quarter of its weight at the instant the other end is let go.
        let i_g = log.pitch_inertia();
        let d = 0.8f32;
        assert!((g.instant_n[0] - log.weight_n() * i_g / (i_g + log.mass_kg * d * d)).abs() < 0.5, "{:?}", g.instant_n);
        // Resting on its dropped end, the holder carries about half.
        let s = g.end_on_ground_share.unwrap();
        assert!(s > 0.4 && s < 0.6, "{s}");
        assert!(g.events.iter().any(|e| e.kind == "co_drop"));
        // Three around a rock: one lets go, the two left cannot balance it.
        let rock = Load::rock(9.0, 0.5);
        let around: Vec<Carrier> = (0..3).map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 3.0;
            Carrier::at(&b, 0.3 * a.cos(), 0.3 * a.sin(), 0.0, Surface::Firm)
        }).collect();
        assert!(let_go(&rock, &around, 0, &CarryOpts::default()).dropped);
    }
}
