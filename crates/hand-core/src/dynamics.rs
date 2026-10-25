//! Joint-level rigid-body dynamics of a robot built from the part presets.
//!
//! [`crate::balance`] answers, for one posture, where the centre of mass is
//! and what torque each joint holds (quasi-static). This module moves the
//! joints between two postures and reports every joint at every instant:
//!
//! * The upper body is a tree of rigid links rooted at the pelvis: waist yaw,
//!   waist pitch (or a rigid trunk when the belly preset has no pitch
//!   actuator), neck, and per arm a shoulder and an elbow with the hand and
//!   its load welded to the forearm. Masses and centres of mass are the
//!   presets' (the same split as `balance`); inertias are those of solid
//!   boxes of the drawn link sizes (est).
//! * Forward dynamics `M(q) q̈ = τ − h(q, q̇)`: `h` and the columns of `M` come
//!   from the recursive Newton–Euler algorithm (the CRBA-by-RNEA route of
//!   Featherstone 2008, *Rigid Body Dynamics Algorithms*, ch. 5–6), solved by
//!   Cholesky and integrated by semi-implicit Euler (2 ms).
//! * Each joint is driven by its preset's actuator: computed-torque control
//!   around a minimum-jerk reference (Flash & Hogan 1985), clipped to the
//!   torque the actuator has at that speed (`Actuator::torque_at`, the linear
//!   torque–speed line). A non-backdrivable actuator (the bus servos) stalls
//!   instead of being driven back by an overload; a backdrivable QDD yields.
//! * The root moves as the base makes it: the pelvis shifts over the stance
//!   foot, rises on a step, accelerates with a start, and a wheeled base
//!   pitches on a slope. Gravity enters as an upward root acceleration.
//! * The root wrench of the Newton–Euler pass plus the lower body gives the
//!   ground reaction and the dynamic zero-moment point (Vukobratović &
//!   Borovac 2004). The leg and drive torques are its statics on the stance
//!   feet (Jacobian transpose, knee lever as in `balance`): the legs are not
//!   simulated as moving links.
//! * The balance strategy (trunk lean, pelvis shift) is the one `balance`
//!   solves for the target posture. With the feed-forward its anticipated
//!   share starts with the motion; the rest, and everything without it,
//!   starts one electromechanical delay late (`REACTION_DELAY_S`), so the
//!   margin dip of a late reaction comes out of the simulation.
//!
//! Simplifications, all marked in the code: legs are statics, no joint
//! friction or rotor inertia (armature), no current-loop lag, a stalled
//! servo's gearbox holds any load (est), box inertias.

use serde::Serialize;

use crate::balance::{step_knee_lever, Body, FeedForward, Posture, Source, CARGO_X_M, CARGO_Z_M, G, HIP_HALF_WIDTH_M, KNEE_LEVER_STAND_M, KNEE_LEVER_WALK_M, REACTION_DELAY_S};
use crate::parts::*;

type V = [f64; 3];
type M = [[f64; 3]; 3];

fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn mv(m: &M, v: V) -> V {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
fn mm(a: &M, b: &M) -> M {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}
fn tr(a: &M) -> M {
    [[a[0][0], a[1][0], a[2][0]], [a[0][1], a[1][1], a[2][1]], [a[0][2], a[1][2], a[2][2]]]
}
const EYE: M = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// Rotation by `ang` about the unit axis `a` (Rodrigues).
fn rot(a: V, ang: f64) -> M {
    let (s, c) = ang.sin_cos();
    let t = 1.0 - c;
    [
        [c + t * a[0] * a[0], t * a[0] * a[1] - s * a[2], t * a[0] * a[2] + s * a[1]],
        [t * a[0] * a[1] + s * a[2], c + t * a[1] * a[1], t * a[1] * a[2] - s * a[0]],
        [t * a[0] * a[2] - s * a[1], t * a[1] * a[2] + s * a[0], c + t * a[2] * a[2]],
    ]
}
fn f3(v: V) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}
fn d3(v: [f32; 3]) -> V {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

/// Inertia of a solid box of half sizes `h` and mass `m` about its centre.
fn box_inertia(m: f64, h: V) -> M {
    let (a, b, c) = (2.0 * h[0], 2.0 * h[1], 2.0 * h[2]);
    [[m * (b * b + c * c) / 12.0, 0.0, 0.0], [0.0, m * (a * a + c * c) / 12.0, 0.0], [0.0, 0.0, m * (a * a + b * b) / 12.0]]
}

/// Raising axis of the arm joints: rotation about −y turns a hanging limb forward.
const ARM_AXIS: V = [0.0, -1.0, 0.0];
/// Pitch axis of the waist and neck: positive leans forward.
const PITCH_AXIS: V = [0.0, 1.0, 0.0];
const YAW_AXIS: V = [0.0, 0.0, 1.0];

#[derive(Clone, Debug)]
struct Link {
    name: &'static str,
    segment: Segment,
    parent: Option<usize>,
    /// Joint origin in the parent's frame.
    origin: V,
    /// Joint axis, in the parent's frame (and this link's).
    axis: V,
    /// Degree of freedom of the joint, `None` when welded.
    dof: Option<usize>,
    /// A welded joint whose structural moment about `axis` is reported.
    locked: Option<usize>,
    mass: f64,
    com: V,
    inertia: M,
    /// Drawn box: centre in the link frame and half sizes.
    shape: (V, V),
}

/// One joint of the report, simulated or not.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynJoint {
    pub name: String,
    pub segment: Segment,
    pub actuator: String,
    pub peak_nm: f32,
    pub cont_nm: f32,
    pub speed_rad_s: f32,
    pub backdrivable: bool,
    /// `sim` (moved by the dynamics), `statics` (leg and drive torques from
    /// the ground reaction) or `structure` (a rigid joint without an actuator).
    pub kind: &'static str,
    pub q_min_deg: f32,
    pub q_max_deg: f32,
}

/// The upper-body tree of one robot with one load case.
#[derive(Clone, Debug)]
pub struct Robot {
    links: Vec<Link>,
    /// Simulated joints, in degree-of-freedom order.
    dof_joints: Vec<(usize, Actuator, f64, f64)>,
    /// The report's joint list: simulated, structural, then statics.
    pub joints: Vec<DynJoint>,
}

/// Where the base puts the pelvis at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RootState {
    /// Pelvis shift toward the stance foot, metres.
    pub shift_y: f32,
    /// Pelvis rise (stepping up), metres, and its vertical acceleration.
    pub rise_m: f32,
    pub rise_acc: f32,
    /// Forward acceleration of the base, m/s².
    pub accel: f32,
    /// Pitch of a wheeled base, radians (nose up positive).
    pub pitch: f32,
}

impl Robot {
    /// Builds the tree from the presets with the hand loads and cargo of `post`.
    pub fn new(body: &Body, post: &Posture) -> Robot {
        let mut links: Vec<Link> = Vec::new();
        let mut joints = Vec::new();
        let mut dof_joints = Vec::new();
        let sim = |name: &str, seg: Segment, a: &Actuator, lo: f64, hi: f64| DynJoint {
            name: name.into(),
            segment: seg,
            actuator: a.name.into(),
            peak_nm: a.peak_nm,
            cont_nm: a.cont_nm,
            speed_rad_s: a.speed_rad_s,
            backdrivable: a.backdrivable,
            kind: "sim",
            q_min_deg: lo as f32,
            q_max_deg: hi as f32,
        };
        // Pelvis (root): belly, the battery when it sits low, and the cargo bay.
        let pelvis_half = [0.09, 0.14, 0.07];
        let mut pm = body.belly.mass_kg as f64;
        let mut pc = scale([0.0, 0.0, -(body.belly.com_down_m as f64)], pm);
        if body.plan.battery_low {
            pm += body.chest.battery_kg as f64;
            pc = add(pc, scale([0.0, 0.0, -(body.belly.com_down_m as f64)], body.chest.battery_kg as f64));
        }
        if post.cargo_kg > 0.0 {
            pm += post.cargo_kg as f64;
            pc = add(pc, scale([CARGO_X_M as f64, 0.0, (CARGO_Z_M - WAIST_M) as f64], post.cargo_kg as f64));
        }
        links.push(Link { name: "pelvis", segment: Segment::Belly, parent: None, origin: [0.0; 3], axis: YAW_AXIS, dof: None, locked: None, mass: pm, com: scale(pc, 1.0 / pm.max(1e-6)), inertia: box_inertia(pm, pelvis_half), shape: ([0.0, 0.0, -0.06], pelvis_half) });
        // Waist yaw: carries the whole upper body about the vertical.
        dof_joints.push((1, body.belly.waist, -90f64.to_radians(), 90f64.to_radians()));
        joints.push(sim("waist_yaw", Segment::Belly, &body.belly.waist, -90.0, 90.0));
        links.push(Link { name: "waist", segment: Segment::Belly, parent: Some(0), origin: [0.0; 3], axis: YAW_AXIS, dof: Some(0), locked: None, mass: 0.0, com: [0.0; 3], inertia: [[0.0; 3]; 3], shape: ([0.0; 3], [0.0; 3]) });
        // Trunk: chest structure and computer, the battery when it sits high.
        let trunk_half = [0.10, 0.16, 0.16];
        let mut tm = body.chest.mass_kg as f64;
        if !body.plan.battery_low {
            tm += body.chest.battery_kg as f64;
        }
        let (dof, locked) = match &body.belly.pitch {
            Some(a) => {
                let lim = (body.belly.max_lean_deg as f64).max(1.0);
                dof_joints.push((2, *a, -lim.to_radians(), lim.to_radians()));
                joints.push(sim("waist_pitch", Segment::Belly, a, -lim, lim));
                (Some(dof_joints.len() - 1), None)
            }
            None => (None, Some(0)),
        };
        links.push(Link { name: "trunk", segment: Segment::Chest, parent: Some(1), origin: [0.0; 3], axis: PITCH_AXIS, dof, locked, mass: tm, com: [0.0, 0.0, body.chest.com_up_m as f64], inertia: box_inertia(tm, trunk_half), shape: ([0.0, 0.0, 0.16], trunk_half) });
        let neck_h = (SHOULDER_M + 0.07 - WAIST_M) as f64;
        dof_joints.push((3, body.head.neck, -45f64.to_radians(), 45f64.to_radians()));
        joints.push(sim("neck", Segment::Head, &body.head.neck, -45.0, 45.0));
        let head_half = [0.07, 0.08, 0.07];
        links.push(Link { name: "head", segment: Segment::Head, parent: Some(2), origin: [0.0, 0.0, neck_h], axis: PITCH_AXIS, dof: Some(dof_joints.len() - 1), locked: None, mass: body.head.mass_kg as f64, com: [0.0, 0.0, body.head.com_up_m as f64], inertia: box_inertia(body.head.mass_kg as f64, head_half), shape: ([0.0, 0.0, 0.07], head_half) });
        let (m_up, r_up, m_fore) = body.arm_split();
        let (m_up, r_up, m_fore) = (m_up as f64, r_up as f64, m_fore as f64);
        let (up_l, fo_l) = (UPPER_ARM_M as f64, FOREARM_M as f64);
        for (side, tag) in [(0usize, "r"), (1, "l")] {
            let y = if side == 0 { -SHOULDER_HALF_WIDTH_M } else { SHOULDER_HALF_WIDTH_M } as f64;
            let sh = links.len();
            dof_joints.push((sh, body.arm.shoulder, -60f64.to_radians(), 180f64.to_radians()));
            joints.push(sim(&format!("shoulder_{tag}"), Segment::Arm, &body.arm.shoulder, -60.0, 180.0));
            let uh = [0.035, 0.035, up_l / 2.0];
            links.push(Link { name: if side == 0 { "upper_arm_r" } else { "upper_arm_l" }, segment: Segment::Arm, parent: Some(2), origin: [0.0, y, (SHOULDER_M - WAIST_M) as f64], axis: ARM_AXIS, dof: Some(dof_joints.len() - 1), locked: None, mass: m_up, com: [0.0, 0.0, -r_up], inertia: box_inertia(m_up, uh), shape: ([0.0, 0.0, -up_l / 2.0], uh) });
            let el = links.len();
            dof_joints.push((el, body.arm.elbow, 0.0, 150f64.to_radians()));
            joints.push(sim(&format!("elbow_{tag}"), Segment::Arm, &body.arm.elbow, 0.0, 150.0));
            let fh = [0.03, 0.03, fo_l / 2.0];
            links.push(Link { name: if side == 0 { "forearm_r" } else { "forearm_l" }, segment: Segment::Arm, parent: Some(sh), origin: [0.0, 0.0, -up_l], axis: ARM_AXIS, dof: Some(dof_joints.len() - 1), locked: None, mass: m_fore, com: [0.0, 0.0, -0.45 * fo_l], inertia: box_inertia(m_fore, fh), shape: ([0.0, 0.0, -fo_l / 2.0], fh) });
            let hm = body.hand_kg as f64 + post.hand_load_kg[side] as f64;
            let hh = [0.04, 0.05, 0.04];
            links.push(Link { name: if side == 0 { "hand_r" } else { "hand_l" }, segment: Segment::Hand, parent: Some(el), origin: [0.0, 0.0, -fo_l], axis: ARM_AXIS, dof: None, locked: None, mass: hm, com: [0.0, 0.0, -0.06], inertia: box_inertia(hm, hh), shape: ([0.0, 0.0, -0.05], hh) });
        }
        if body.belly.pitch.is_none() {
            let w = body.belly.waist;
            joints.push(DynJoint { name: "waist_pitch".into(), segment: Segment::Belly, actuator: "구조 (피치 관절 없음)".into(), peak_nm: 0.0, cont_nm: 0.0, speed_rad_s: w.speed_rad_s, backdrivable: false, kind: "structure", q_min_deg: 0.0, q_max_deg: 0.0 });
        }
        let stat = |name: &str, seg: Segment, a: &Actuator| DynJoint { kind: "statics", ..sim(name, seg, a, 0.0, 0.0) };
        if body.legged {
            for tag in ["r", "l"] {
                joints.push(stat(&format!("hip_pitch_{tag}"), Segment::Leg, &body.leg.knee));
                joints.push(stat(&format!("hip_roll_{tag}"), Segment::Leg, &body.leg.knee));
                joints.push(stat(&format!("knee_{tag}"), Segment::Leg, &body.leg.knee));
                joints.push(stat(&format!("ankle_{tag}"), Segment::Foot, &body.leg.ankle));
            }
        } else {
            let f = body.chassis.drive_force_n;
            joints.push(DynJoint { name: "drive".into(), segment: Segment::Foot, actuator: body.chassis.name.into(), peak_nm: 2.0 * f, cont_nm: f, speed_rad_s: 0.0, backdrivable: true, kind: "statics", q_min_deg: 0.0, q_max_deg: 0.0 });
        }
        Robot { links, dof_joints, joints }
    }

    pub fn dofs(&self) -> usize {
        self.dof_joints.len()
    }

    /// Index of a joint of the report by name.
    pub fn joint_index(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }

    /// Root pose: rotation and position of the pelvis frame.
    fn root_pose(&self, root: &RootState) -> (M, V) {
        let r = rot(PITCH_AXIS, -(root.pitch as f64));
        let p = mv(&r, [0.0, root.shift_y as f64, WAIST_M as f64 + root.rise_m as f64]);
        (r, p)
    }

    /// Recursive Newton–Euler: joint torques for `q, qd, qdd`, the wrench the
    /// root takes (force, moment about the ground origin), the structural
    /// moments of locked joints, and the link frames for drawing.
    fn rnea(&self, q: &[f64], qd: &[f64], qdd: &[f64], root: &RootState, gravity: bool) -> Rnea {
        let n = self.links.len();
        let (r0, p0) = self.root_pose(root);
        let mut rot_w = vec![EYE; n];
        let mut pos = vec![[0.0; 3]; n];
        let mut w = vec![[0.0; 3]; n];
        let mut wd = vec![[0.0; 3]; n];
        let mut acc = vec![[0.0; 3]; n];
        let mut f = vec![[0.0; 3]; n];
        let mut nn = vec![[0.0; 3]; n];
        let mut com_w = vec![[0.0; 3]; n];
        let g = if gravity { G as f64 } else { 0.0 };
        let base_acc = [root.accel as f64, 0.0, root.rise_acc as f64 + g];
        for (i, l) in self.links.iter().enumerate() {
            match l.parent {
                None => {
                    rot_w[i] = r0;
                    pos[i] = p0;
                    acc[i] = if gravity || root.accel != 0.0 || root.rise_acc != 0.0 { base_acc } else { [0.0; 3] };
                }
                Some(p) => {
                    let o = add(pos[p], mv(&rot_w[p], l.origin));
                    let z = mv(&rot_w[p], l.axis);
                    let (qi, qdi, qddi) = l.dof.map_or((0.0, 0.0, 0.0), |k| (q[k], qd[k], qdd[k]));
                    rot_w[i] = mm(&rot_w[p], &rot(l.axis, qi));
                    pos[i] = o;
                    w[i] = add(w[p], scale(z, qdi));
                    wd[i] = add(add(wd[p], scale(z, qddi)), cross(w[p], scale(z, qdi)));
                    let d = sub(o, pos[p]);
                    acc[i] = add(add(acc[p], cross(wd[p], d)), cross(w[p], cross(w[p], d)));
                }
            }
            let rc = mv(&rot_w[i], l.com);
            com_w[i] = add(pos[i], rc);
            let ac = add(add(acc[i], cross(wd[i], rc)), cross(w[i], cross(w[i], rc)));
            let iw = mm(&mm(&rot_w[i], &l.inertia), &tr(&rot_w[i]));
            f[i] = scale(ac, l.mass);
            nn[i] = add(add(mv(&iw, wd[i]), cross(w[i], mv(&iw, w[i]))), cross(rc, f[i]));
        }
        let mut tau = vec![0.0; self.dof_joints.len()];
        let mut locked = vec![0.0; 1];
        for i in (0..n).rev() {
            let l = &self.links[i];
            if let Some(p) = l.parent {
                let z = mv(&rot_w[p], l.axis);
                if let Some(k) = l.dof {
                    tau[k] = dot(z, nn[i]);
                }
                if let Some(k) = l.locked {
                    locked[k] = dot(z, nn[i]);
                }
                let d = sub(pos[i], pos[p]);
                let (fi, ni) = (f[i], nn[i]);
                f[p] = add(f[p], fi);
                nn[p] = add(add(nn[p], ni), cross(d, fi));
            }
        }
        let f0 = f[0];
        let n0 = add(nn[0], cross(pos[0], f0));
        Rnea { tau, f0, n0, locked, rot: rot_w, pos, com: com_w }
    }

    /// Joint-space inertia matrix (columns by RNEA with unit accelerations).
    fn mass_matrix(&self, q: &[f64], root: &RootState) -> Vec<Vec<f64>> {
        let n = self.dofs();
        let zero = vec![0.0; n];
        let still = RootState { pitch: root.pitch, shift_y: root.shift_y, rise_m: root.rise_m, ..RootState::default() };
        let cols: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                self.rnea(q, &zero, &e, &still, false).tau
            })
            .collect();
        (0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect()
    }
}

struct Rnea {
    tau: Vec<f64>,
    f0: V,
    /// About the ground origin.
    n0: V,
    locked: Vec<f64>,
    rot: Vec<M>,
    pos: Vec<V>,
    com: Vec<V>,
}

/// Solves `a x = b` for a symmetric positive definite `a` (Cholesky).
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d <= 1e-12 {
                    return None;
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    Some(x)
}

/// Minimum-jerk blend 0 → 1 over `u` in [0, 1], with its first two derivatives (per unit `u`).
fn min_jerk(u: f64) -> (f64, f64, f64) {
    if u <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if u >= 1.0 {
        return (1.0, 0.0, 0.0);
    }
    let (u2, u3) = (u * u, u * u * u);
    (10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u2, 30.0 * u2 - 60.0 * u3 + 30.0 * u2 * u2, 60.0 * u - 180.0 * u2 + 120.0 * u3)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DynCfg {
    pub dt_s: f32,
    /// Report every this many seconds.
    pub sample_s: f32,
    /// Time after the motion to settle.
    pub settle_s: f32,
    /// Closed-loop bandwidth of the joint controller, Hz.
    pub bandwidth_hz: f32,
    /// A joint has "reached" its target within this error, degrees.
    pub reach_tol_deg: f32,
}

impl Default for DynCfg {
    fn default() -> Self {
        // 4 Hz with critical damping is a moderate position loop for these
        // actuators (est); 2 ms keeps ω·dt ≈ 0.05.
        DynCfg { dt_s: 0.002, sample_s: 0.02, settle_s: 0.6, bandwidth_hz: 4.0, reach_tol_deg: 3.0 }
    }
}

/// One reported instant.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynSample {
    pub t: f32,
    /// Per simulated joint (degree-of-freedom order).
    pub q_deg: Vec<f32>,
    pub qd: Vec<f32>,
    /// Per report joint ([`Robot::joints`]): torque, torque available, peak ratio.
    pub tau_nm: Vec<f32>,
    pub avail_nm: Vec<f32>,
    pub ratio: Vec<f32>,
    pub saturated: Vec<bool>,
    pub stalled: Vec<bool>,
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    pub margin: f32,
    pub grf: [f32; 3],
    /// Camera pitch left after the neck's correction, degrees.
    pub head_pitch_deg: f32,
    pub root: RootState,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointSummary {
    pub name: String,
    pub kind: &'static str,
    pub peak_ratio: f32,
    /// Root mean square of torque over the continuous rating.
    pub rms_effort: f32,
    pub saturated_s: f32,
    pub stalled_s: f32,
    /// Largest |reference − actual| during the motion, degrees (simulated joints).
    pub max_lag_deg: f32,
    /// |target − actual| at the end, degrees (simulated joints).
    pub final_err_deg: f32,
}

/// A simulated transition between two postures.
#[derive(Clone, Debug, Serialize)]
pub struct Rollout {
    pub source: Source,
    pub duration_s: f32,
    pub feed_forward: bool,
    /// Share of the compensation started with the motion (the rest starts late).
    pub anticipated: f32,
    pub joints: Vec<DynJoint>,
    pub samples: Vec<DynSample>,
    pub summary: Vec<JointSummary>,
    pub min_margin: f32,
    pub min_margin_t: f32,
    pub final_margin: f32,
    /// The quasi-static margin `balance` gives for the target posture.
    pub static_margin: f32,
    pub peak_ratio: f32,
    pub worst_joint: Option<String>,
    /// Every simulated joint ended within the tolerance of its target.
    pub reached: bool,
    /// When every simulated joint stayed within the tolerance, seconds.
    pub settle_s: Option<f32>,
    /// Reached, nothing over its peak or stalled, margin never below zero.
    pub feasible: bool,
    pub notes: Vec<String>,
    /// Largest camera pitch left after the neck, degrees.
    pub head_pitch_max_deg: f32,
    /// Largest sideways acceleration of the pelvis (the weight shift), m/s².
    pub pelvis_lat_acc_max: f32,
    #[serde(skip)]
    robot: Robot,
    #[serde(skip)]
    support: [f32; 4],
    #[serde(skip)]
    support_before: [f32; 4],
    #[serde(skip)]
    lift_at: f32,
}

/// Simulated joint targets of a posture: waist yaw, waist pitch (if any),
/// neck, shoulders and elbows, radians, in degree-of-freedom order.
fn targets(body: &Body, post: &Posture, lean_rad: f64, pitch: f64, robot: &Robot) -> Vec<f64> {
    let mut v = Vec::with_capacity(robot.dofs());
    v.push(0.0);
    if body.belly.pitch.is_some() {
        v.push(lean_rad);
    }
    // The neck cancels its share of the body's tilt to keep the camera level.
    let tilt = if body.belly.pitch.is_some() { lean_rad } else { 0.0 } - pitch;
    v.push(-tilt * body.head.gaze_stabilization as f64);
    for s in 0..2 {
        v.push((post.arm_elev_deg[s] as f64).to_radians());
        v.push((post.elbow_deg[s] as f64).to_radians());
    }
    v
}

/// Simulates the move from posture `from` to posture `to` (the motion of
/// `to.source`) and reports every joint. `grip_capacity_n` is the friction
/// capacity of one hand, as for `Body::solve`.
pub fn simulate(body: &Body, from: &Posture, to: &Posture, ff: &FeedForward, grip_capacity_n: f32, cfg: &DynCfg) -> Rollout {
    let robot = Robot::new(body, to);
    let n = robot.dofs();
    let start = body.solve(from, ff, grip_capacity_n);
    let goal = body.solve(to, ff, grip_capacity_n);
    let dur = to.source.duration_s().clamp(0.2, 3.0) as f64;
    let dt = cfg.dt_s as f64;
    let wn = std::f64::consts::TAU * cfg.bandwidth_hz as f64;
    let (kp, kd) = (wn * wn, 2.0 * wn);
    let lean0 = (start.trunk_lean_deg as f64).to_radians();
    let lean1 = (goal.trunk_lean_deg as f64).to_radians();
    let pitch0 = body.base_pitch(from) as f64;
    let pitch1 = body.base_pitch(to) as f64;
    let q_from = targets(body, from, lean0, pitch0, &robot);
    let q_to = targets(body, to, lean1, pitch1, &robot);
    // Posture leads movement (APA before the push-off: Callegari 2018, Honeine
    // 2016): a start or a stop begins its acceleration once the posture change
    // that goes with it (arms, lean, neck) has settled, so the two inertial
    // disturbances do not add up.
    let posture_moves = q_from.iter().zip(&q_to).any(|(a, b)| (a - b).abs() > 1f64.to_radians());
    let accel_t0 = if to.source == Source::Accel && posture_moves { dur } else { 0.0 };
    let total = dur + accel_t0 + cfg.settle_s as f64;
    // How much of the balance correction is anticipated (as in `Body::solve`).
    let anticipated = if ff.enabled && goal.shift_uncomp_m.abs() > 1e-5 { (ff.gain * goal.ff_predicted_m / goal.shift_uncomp_m).clamp(0.0, 2.0) } else { 0.0 } as f64;
    let delay = REACTION_DELAY_S as f64;
    let comp = |t: f64| -> (f64, f64, f64) {
        let (a, ad, add) = min_jerk(t / dur);
        let (b, bd, bdd) = min_jerk((t - delay) / dur);
        (anticipated * a + (1.0 - anticipated) * b, (anticipated * ad + (1.0 - anticipated) * bd) / dur, (anticipated * add + (1.0 - anticipated) * bdd) / (dur * dur))
    };
    let comp_dof = if body.belly.pitch.is_some() { Some(1usize) } else { None };
    let (shift0, shift1) = (start.lateral_shift_m as f64, goal.lateral_shift_m as f64);
    let step = to.step_m as f64;
    let support = body.support_rect(to, pitch1 as f32);
    // Going from two feet to one, the weight shifts first and the foot lifts
    // when the move ends (APA before foot-off, Honeine 2016): until then the
    // support is still both feet.
    let lift_at = if to.single_support && !from.single_support { dur } else { 0.0 };
    let support_before = body.support_rect(from, pitch0 as f32);
    let root_at = |t: f64| -> RootState {
        let (c, cd, cdd) = comp(t);
        let _ = cd;
        let (s, _, sdd) = min_jerk(t / dur);
        let _ = cdd;
        // A start or stop holds its acceleration for the whole move, with 0.1 s ramps.
        let ramp = |x: f64| (x / 0.1).clamp(0.0, 1.0);
        let acc = if to.source == Source::Accel { to.accel_mps2 as f64 * ramp(t - accel_t0).min(ramp(accel_t0 + dur - t)) } else { 0.0 };
        RootState {
            shift_y: (shift0 + (shift1 - shift0) * c) as f32,
            rise_m: (step * s) as f32,
            rise_acc: (step * sdd / (dur * dur)) as f32,
            accel: acc as f32,
            pitch: (pitch0 + (pitch1 - pitch0) * s) as f32,
        }
    };

    let mut q = q_from.clone();
    let mut qd = vec![0.0; n];
    let nj = robot.joints.len();
    let mut samples = Vec::new();
    let mut peak = vec![0.0f32; nj];
    let mut eff2 = vec![0.0f64; nj];
    let mut sat_s = vec![0.0f32; nj];
    let mut stall_s = vec![0.0f32; nj];
    let mut lag = vec![0.0f32; n];
    let (mut min_margin, mut min_t) = (f32::INFINITY, 0.0f32);
    let mut settle: Option<f32> = None;
    let mut steps = 0usize;
    let mut next_sample = 0.0;
    let mut t = 0.0;
    let (mut head_max, mut lat_acc_max) = (0.0f32, 0.0f32);
    let mut notes = Vec::new();
    while t <= total + 1e-9 {
        let root = root_at(t);
        // Reference: the arms follow min-jerk to their targets; the
        // compensating waist pitch follows the (partly late) balance profile.
        let (s, sd, sdd) = min_jerk(t / dur);
        let (c, cd, cdd) = comp(t);
        lat_acc_max = lat_acc_max.max(((shift1 - shift0) * cdd).abs() as f32);
        let mut r = vec![0.0; n];
        let mut rd = vec![0.0; n];
        let mut rdd = vec![0.0; n];
        for k in 0..n {
            let (b, bd, bdd) = if Some(k) == comp_dof { (c, cd, cdd) } else { (s, sd / dur, sdd / (dur * dur)) };
            let dq = q_to[k] - q_from[k];
            r[k] = q_from[k] + dq * b;
            rd[k] = dq * bd;
            rdd[k] = dq * bdd;
        }
        let mmat = robot.mass_matrix(&q, &root);
        let h = robot.rnea(&q, &qd, &vec![0.0; n], &root, true).tau;
        let mut tau = vec![0.0; n];
        let mut sat = vec![false; n];
        let mut avail = vec![0.0; n];
        for k in 0..n {
            let a = rdd[k] + kp * (r[k] - q[k]) + kd * (rd[k] - qd[k]);
            let cmd: f64 = (0..n).map(|j| mmat[k][j] * if j == k { a } else { rdd[j] + kp * (r[j] - q[j]) + kd * (rd[j] - qd[j]) }).sum::<f64>() + h[k];
            let act = &robot.dof_joints[k].1;
            avail[k] = act.torque_at(qd[k] as f32) as f64;
            tau[k] = cmd.clamp(-avail[k], avail[k]);
            sat[k] = cmd.abs() > avail[k] + 1e-9;
        }
        let rhs: Vec<f64> = (0..n).map(|k| tau[k] - h[k]).collect();
        let qdd = cholesky_solve(&mmat, &rhs).unwrap_or_else(|| vec![0.0; n]);
        let mut stalled = vec![false; n];
        for k in 0..n {
            let (_, act, lo, hi) = robot.dof_joints[k];
            let mut v = qd[k] + qdd[k] * dt;
            // A self-locking gearbox does not let an overload drive the joint back (est).
            if !act.backdrivable && sat[k] && v * tau[k] < 0.0 {
                v = 0.0;
                stalled[k] = true;
            }
            q[k] += v * dt;
            if q[k] < lo || q[k] > hi {
                q[k] = q[k].clamp(lo, hi);
                v = 0.0;
            }
            qd[k] = v;
            if t <= dur {
                lag[k] = lag[k].max(((r[k] - q[k]).abs().to_degrees()) as f32);
            }
        }
        // Whole-body wrench: the tree's root wrench plus the lower body.
        let full = robot.rnea(&q, &qd, &qdd, &root, true);
        // One foot carries the robot only once the swing foot has lifted.
        let single = to.single_support && t >= lift_at;
        let (grf, zmp, com) = whole_body(body, to, single, &robot, &full, &root);
        let margin = margin_of(zmp, if t < lift_at { &support_before } else { &support });
        if margin < min_margin {
            min_margin = margin;
            min_t = t as f32;
        }
        let mut all_tau = vec![0.0f32; nj];
        let mut all_avail = vec![0.0f32; nj];
        let mut all_sat = vec![false; nj];
        let mut all_stall = vec![false; nj];
        for k in 0..n {
            all_tau[k] = tau[k].abs() as f32;
            all_avail[k] = avail[k] as f32;
            all_sat[k] = sat[k];
            all_stall[k] = stalled[k];
        }
        let mut idx = n;
        if body.belly.pitch.is_none() {
            all_tau[idx] = full.locked[0].abs() as f32;
            idx += 1;
        }
        for (k, v) in lower_torques(body, to, single, grf, zmp, &root).into_iter().enumerate() {
            all_tau[idx + k] = v.abs();
            all_avail[idx + k] = robot.joints[idx + k].peak_nm;
            all_sat[idx + k] = v.abs() > robot.joints[idx + k].peak_nm;
        }
        for j in 0..nj {
            let pk = robot.joints[j].peak_nm.max(1e-3);
            if robot.joints[j].kind != "structure" {
                peak[j] = peak[j].max(all_tau[j] / pk);
            }
            eff2[j] += ((all_tau[j] / robot.joints[j].cont_nm.max(1e-3)) as f64).powi(2);
            if all_sat[j] {
                sat_s[j] += dt as f32;
            }
            if all_stall[j] {
                stall_s[j] += dt as f32;
            }
        }
        let within = (0..n).all(|k| (q_to[k] - q[k]).abs().to_degrees() <= cfg.reach_tol_deg as f64);
        if within && settle.is_none() && t >= dur * 0.5 {
            settle = Some(t as f32);
        } else if !within {
            settle = None;
        }
        if t + 1e-9 >= next_sample {
            next_sample += cfg.sample_s as f64;
            let tilt = if body.belly.pitch.is_some() { q[1] } else { 0.0 } - root.pitch as f64;
            let neck = q[if body.belly.pitch.is_some() { 2 } else { 1 }];
            head_max = head_max.max(((tilt + neck).to_degrees() as f32).abs());
            samples.push(DynSample {
                t: t as f32,
                q_deg: q.iter().map(|x| x.to_degrees() as f32).collect(),
                qd: qd.iter().map(|x| *x as f32).collect(),
                tau_nm: all_tau.clone(),
                avail_nm: all_avail.clone(),
                ratio: (0..nj).map(|j| if robot.joints[j].kind == "structure" { 0.0 } else { all_tau[j] / robot.joints[j].peak_nm.max(1e-3) }).collect(),
                saturated: all_sat,
                stalled: all_stall,
                com,
                zmp,
                margin,
                grf,
                head_pitch_deg: (tilt + neck).to_degrees() as f32,
                root,
            });
        }
        steps += 1;
        t = steps as f64 * dt;
    }
    let last = samples.last().cloned();
    let summary: Vec<JointSummary> = (0..nj)
        .map(|j| {
            let (lag_j, err) = if j < n { (lag[j], last.as_ref().map_or(0.0, |s| (q_to[j].to_degrees() as f32 - s.q_deg[j]).abs())) } else { (0.0, 0.0) };
            JointSummary { name: robot.joints[j].name.clone(), kind: robot.joints[j].kind, peak_ratio: peak[j], rms_effort: (eff2[j] / steps.max(1) as f64).sqrt() as f32, saturated_s: sat_s[j], stalled_s: stall_s[j], max_lag_deg: lag_j, final_err_deg: err }
        })
        .collect();
    let reached = summary.iter().take(n).all(|s| s.final_err_deg <= cfg.reach_tol_deg);
    let worst = summary.iter().filter(|s| s.kind != "structure").max_by(|a, b| a.peak_ratio.total_cmp(&b.peak_ratio)).map(|w| (w.name.clone(), w.peak_ratio));
    let peak_ratio = worst.as_ref().map_or(0.0, |w| w.1);
    for s in &summary {
        if s.stalled_s > 0.0 {
            notes.push(format!("{} 멈춤 {:.2} s (역구동 불가 기어가 버팀)", s.name, s.stalled_s));
        } else if s.saturated_s > 0.05 {
            notes.push(format!("{} 토크 포화 {:.2} s, 최대 지연 {:.1}°", s.name, s.saturated_s, s.max_lag_deg));
        }
        if s.kind == "sim" && s.final_err_deg > cfg.reach_tol_deg {
            notes.push(format!("{} 목표에 못 미침 ({:.1}° 남음)", s.name, s.final_err_deg));
        }
    }
    if min_margin < 0.0 {
        notes.push(format!("동적 ZMP가 지지면 밖 (여유 {min_margin:.2}, t = {min_t:.2} s)"));
    }
    let final_margin = last.as_ref().map_or(0.0, |s| s.margin);
    let feasible = reached && min_margin >= 0.0 && summary.iter().filter(|s| s.kind != "structure").all(|s| s.stalled_s == 0.0 && s.peak_ratio <= 1.0);
    Rollout {
        source: to.source,
        duration_s: dur as f32,
        feed_forward: ff.enabled,
        anticipated: anticipated as f32,
        joints: robot.joints.clone(),
        samples,
        summary,
        min_margin,
        min_margin_t: min_t,
        final_margin,
        static_margin: goal.margin,
        peak_ratio,
        worst_joint: worst.map(|w| w.0),
        reached,
        settle_s: settle,
        feasible,
        notes,
        head_pitch_max_deg: head_max,
        pelvis_lat_acc_max: lat_acc_max,
        robot,
        support,
        support_before,
        lift_at: lift_at as f32,
    }
}

/// Ground reaction force, dynamic ZMP and whole-body CoM: the tree's root
/// wrench plus the legs and feet (or the mast and chassis) moving with the base.
fn whole_body(body: &Body, post: &Posture, single: bool, robot: &Robot, r: &Rnea, root: &RootState) -> ([f32; 3], [f32; 2], [f32; 3]) {
    let mut f = r.f0;
    let mut m_o = r.n0;
    let g = G as f64;
    let a = [root.accel as f64, 0.0, g];
    let mut mass_sum = 0.0;
    let mut mc = [0.0; 3];
    for (i, l) in robot.links.iter().enumerate() {
        mass_sum += l.mass;
        mc = add(mc, scale(r.com[i], l.mass));
    }
    for (m, p) in lower_masses(body, post, single, root) {
        let fi = scale(a, m);
        f = add(f, fi);
        m_o = add(m_o, cross(p, fi));
        mass_sum += m;
        mc = add(mc, scale(p, m));
    }
    let fz = f[2].max(1e-6);
    let zmp = [(-m_o[1] / fz) as f32, (m_o[0] / fz) as f32];
    (f3(f), zmp, f3(scale(mc, 1.0 / mass_sum.max(1e-6))))
}

/// Point masses of the lower body in the base frame (as `balance` places them).
fn lower_masses(body: &Body, post: &Posture, single: bool, root: &RootState) -> Vec<(f64, V)> {
    let mut v = Vec::new();
    if body.legged {
        for (i, y) in [-STANCE_WIDTH_M / 2.0, STANCE_WIDTH_M / 2.0].into_iter().enumerate() {
            let swing = single && i == 1;
            let lift = if swing { post.step_m.max(0.05) } else { 0.0 } as f64;
            let ys = if swing { root.shift_y as f64 } else { 0.0 };
            v.push((body.leg.mass_kg as f64, [if swing { 0.10 } else { 0.0 }, y as f64 + ys, (HIP_M - body.leg.com_down_m) as f64 + lift * 0.5]));
            v.push((body.leg.foot_mass_kg as f64, [0.25 * body.leg.foot_len_m as f64 + if swing { 0.15 } else { 0.0 }, y as f64 + ys, 0.04 + lift]));
        }
    } else {
        let r = rot(PITCH_AXIS, -(root.pitch as f64));
        v.push((body.mast_kg() as f64, mv(&r, [0.0, 0.0, 0.35])));
        v.push((body.chassis.mass_kg as f64, mv(&r, [0.0, 0.0, body.chassis.com_m as f64])));
    }
    v
}

/// Leg (hip pitch, hip roll, knee, ankle per side) or drive torques: the
/// statics of the ground reaction on the stance feet.
fn lower_torques(body: &Body, post: &Posture, single: bool, grf: [f32; 3], zmp: [f32; 2], root: &RootState) -> Vec<f32> {
    let f = d3(grf);
    if !body.legged {
        let slope = (post.slope_deg as f64).to_radians();
        let m = f[2] / G as f64;
        let crr = if post.propelling() { crate::surface::rolling_resistance(body.base, post.ground.props()) as f64 } else { 0.0 };
        let force = m * (G as f64 * (slope.sin() + crr * slope.cos()) + root.accel.abs() as f64);
        return vec![force as f32];
    }
    let half = STANCE_WIDTH_M as f64 / 2.0;
    let feet = [-half, half];
    // Vertical load share of each foot from the lateral ZMP (all on the stance foot in single support).
    let share_r = if single { 1.0 } else { ((half - zmp[1] as f64) / (2.0 * half)).clamp(0.0, 1.0) };
    let d = if post.step_m > 0.0 { step_knee_lever(post.step_m) } else if single { KNEE_LEVER_WALK_M } else { KNEE_LEVER_STAND_M } as f64;
    let mut out = Vec::with_capacity(8);
    for (k, fy) in feet.iter().enumerate() {
        let share = if k == 0 { share_r } else { 1.0 - share_r };
        let fk = scale(f, share);
        let x_lo = -0.25 * body.leg.foot_len_m as f64;
        let x_hi = 0.75 * body.leg.foot_len_m as f64;
        let cop = [(zmp[0] as f64).clamp(x_lo, x_hi), *fy, 0.0];
        let ankle = [0.0, *fy, 0.08];
        // The knee lever of the sizing rule is the horizontal distance from the reaction to the knee.
        let knee = [cop[0] + d, *fy, 0.5 * (HIP_M as f64 + 0.08)];
        let side = if k == 0 { -1.0 } else { 1.0 };
        let hip = [0.0, root.shift_y as f64 + side * HIP_HALF_WIDTH_M as f64, HIP_M as f64 + root.rise_m as f64];
        let m_at = |p: V| cross(sub(cop, p), fk);
        out.push(m_at(hip)[1].abs() as f32);
        out.push(m_at(hip)[0].abs() as f32);
        out.push(m_at(knee)[1].abs() as f32);
        out.push(m_at(ankle)[1].abs() as f32);
    }
    out
}

/// `balance`'s margin of a ZMP in a support rectangle.
fn margin_of(zmp: [f32; 2], s: &[f32; 4]) -> f32 {
    let (hx, hy) = (0.5 * (s[1] - s[0]), 0.5 * (s[3] - s[2]));
    let (cx, cy) = (0.5 * (s[0] + s[1]), 0.5 * (s[2] + s[3]));
    (1.0 - (zmp[0] - cx).abs() / hx.max(1e-3)).min(1.0 - (zmp[1] - cy).abs() / hy.max(1e-3))
}

/// A drawn link: centre, orientation (columns are the box axes) and half sizes, base frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LinkBox {
    pub name: &'static str,
    pub segment: Segment,
    pub center: [f32; 3],
    pub axes: [[f32; 3]; 3],
    pub half: [f32; 3],
}

/// A joint as the third-person view draws it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointPoint {
    pub name: String,
    pub kind: &'static str,
    pub pos: [f32; 3],
    pub q_deg: Option<f32>,
    pub tau_nm: f32,
    pub peak_nm: f32,
    pub ratio: f32,
    pub saturated: bool,
    pub stalled: bool,
}

/// The whole body at one instant, base frame (x forward, y left, z up, ground at z = 0).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BodyPose {
    pub t: f32,
    pub source: Source,
    pub boxes: Vec<LinkBox>,
    pub joints: Vec<JointPoint>,
    pub com: [f32; 3],
    pub zmp: [f32; 2],
    pub support: [f32; 4],
    pub margin: f32,
    pub grf: [f32; 3],
}

fn axes_of(r: &M) -> [[f32; 3]; 3] {
    [[r[0][0] as f32, r[1][0] as f32, r[2][0] as f32], [r[0][1] as f32, r[1][1] as f32, r[2][1] as f32], [r[0][2] as f32, r[1][2] as f32, r[2][2] as f32]]
}

/// What a rollout says, without the time series: for event logs and the SDS.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynDigest {
    pub source: Source,
    pub feed_forward: bool,
    pub anticipated: f32,
    pub min_margin: f32,
    pub min_margin_t: f32,
    pub static_margin: f32,
    pub peak_ratio: f32,
    pub worst_joint: Option<String>,
    pub reached: bool,
    pub settle_s: Option<f32>,
    pub feasible: bool,
    pub joints: Vec<JointSummary>,
    pub notes: Vec<String>,
    pub head_pitch_max_deg: f32,
    pub pelvis_lat_acc_max: f32,
}

impl Rollout {
    pub fn digest(&self) -> DynDigest {
        DynDigest {
            source: self.source,
            feed_forward: self.feed_forward,
            anticipated: self.anticipated,
            min_margin: self.min_margin,
            min_margin_t: self.min_margin_t,
            static_margin: self.static_margin,
            peak_ratio: self.peak_ratio,
            worst_joint: self.worst_joint.clone(),
            reached: self.reached,
            settle_s: self.settle_s,
            feasible: self.feasible,
            joints: self.summary.clone(),
            notes: self.notes.clone(),
            head_pitch_max_deg: self.head_pitch_max_deg,
            pelvis_lat_acc_max: self.pelvis_lat_acc_max,
        }
    }

    /// The sample at or before `t` (the last one after the end).
    pub fn sample_at(&self, t: f32) -> Option<&DynSample> {
        let i = self.samples.partition_point(|s| s.t <= t);
        self.samples.get(i.saturating_sub(1))
    }

    /// The pose at time `t` of the rollout, for drawing.
    pub fn pose_at(&self, body: &Body, post: &Posture, t: f32) -> Option<BodyPose> {
        let s = self.sample_at(t)?;
        let q: Vec<f64> = s.q_deg.iter().map(|d| (*d as f64).to_radians()).collect();
        let n = q.len();
        let k = self.robot.rnea(&q, &vec![0.0; n], &vec![0.0; n], &s.root, false);
        let mut boxes = Vec::new();
        let mut joints = Vec::new();
        for (i, l) in self.robot.links.iter().enumerate() {
            if l.shape.1 != [0.0; 3] {
                boxes.push(LinkBox { name: l.name, segment: l.segment, center: f3(add(k.pos[i], mv(&k.rot[i], l.shape.0))), axes: axes_of(&k.rot[i]), half: f3(l.shape.1) });
            }
        }
        let jp = |name: &str, pos: V, q_deg: Option<f32>, j: usize| JointPoint {
            name: name.into(),
            kind: self.joints[j].kind,
            pos: f3(pos),
            q_deg,
            tau_nm: s.tau_nm[j],
            peak_nm: self.joints[j].peak_nm,
            ratio: s.ratio[j],
            saturated: s.saturated[j],
            stalled: s.stalled[j],
        };
        for (d, (li, _, _, _)) in self.robot.dof_joints.iter().enumerate() {
            joints.push(jp(&self.joints[d].name, k.pos[*li], Some(s.q_deg[d]), d));
        }
        if let Some(j) = self.robot.joint_index("waist_pitch").filter(|j| self.joints[*j].kind == "structure") {
            joints.push(jp("waist_pitch", k.pos[2], None, j));
        }
        let pelvis = k.pos[0];
        if body.legged {
            let half = STANCE_WIDTH_M / 2.0;
            for (side, tag) in [(0usize, "r"), (1, "l")] {
                let y = if side == 0 { -half } else { half };
                let swing = post.single_support && side == 1 && s.t >= self.lift_at;
                let lift = if swing { post.step_m.max(0.05) } else { 0.0 };
                let sign = if side == 0 { -1.0 } else { 1.0 };
                let hip = [pelvis[0] as f32, pelvis[1] as f32 + sign * HIP_HALF_WIDTH_M, HIP_M + s.root.rise_m];
                let ankle = [if swing { 0.15 } else { 0.0 }, y + if swing { s.root.shift_y } else { 0.0 }, 0.08 + lift];
                let bend = if post.step_m > 0.0 { step_knee_lever(post.step_m) } else if post.single_support && s.t >= self.lift_at { KNEE_LEVER_WALK_M } else { KNEE_LEVER_STAND_M };
                let knee = [0.5 * (hip[0] + ankle[0]) + bend, 0.5 * (hip[1] + ankle[1]), 0.5 * (hip[2] + ankle[2])];
                for (name, a, b, w) in [("thigh", hip, knee, 0.05f32), ("shank", knee, ankle, 0.045)] {
                    boxes.push(limb_box(if side == 0 { if name == "thigh" { "thigh_r" } else { "shank_r" } } else if name == "thigh" { "thigh_l" } else { "shank_l" }, Segment::Leg, a, b, w));
                }
                let fl = body.leg.foot_len_m;
                boxes.push(LinkBox { name: if side == 0 { "foot_r" } else { "foot_l" }, segment: Segment::Foot, center: [0.25 * fl + ankle[0], ankle[1], ankle[2] - 0.05], axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], half: [fl / 2.0, body.leg.foot_width_m / 2.0, 0.03] });
                let base = self.robot.joint_index(&format!("hip_pitch_{tag}")).unwrap_or(0);
                joints.push(jp(&format!("hip_pitch_{tag}"), d3(hip), None, base));
                joints.push(jp(&format!("hip_roll_{tag}"), d3(hip), None, base + 1));
                joints.push(jp(&format!("knee_{tag}"), d3(knee), None, base + 2));
                joints.push(jp(&format!("ankle_{tag}"), d3(ankle), None, base + 3));
            }
        } else {
            let r = rot(PITCH_AXIS, -(s.root.pitch as f64));
            let len = body.support_len_m;
            boxes.push(LinkBox { name: "chassis", segment: Segment::Foot, center: f3(mv(&r, [0.0, 0.0, 0.08])), axes: axes_of(&r), half: [len / 2.0, CHASSIS_WIDTH_M / 2.0, 0.06] });
            boxes.push(LinkBox { name: "mast", segment: Segment::Leg, center: f3(mv(&r, [0.0, 0.0, (0.14 + WAIST_M as f64 - 0.13) / 2.0])), axes: axes_of(&r), half: [0.03, 0.03, (WAIST_M - 0.27) / 2.0] });
            if let Some(j) = self.robot.joint_index("drive") {
                joints.push(jp("drive", mv(&r, [0.0, 0.0, 0.08]), None, j));
            }
        }
        Some(BodyPose { t: s.t, source: self.source, boxes, joints, com: s.com, zmp: s.zmp, support: if s.t < self.lift_at { self.support_before } else { self.support }, margin: s.margin, grf: s.grf })
    }
}

/// A box along the segment from `a` to `b`, `w` half wide.
fn limb_box(name: &'static str, segment: Segment, a: [f32; 3], b: [f32; 3], w: f32) -> LinkBox {
    let (a, b) = (d3(a), d3(b));
    let d = sub(b, a);
    let len = dot(d, d).sqrt().max(1e-6);
    let z = scale(d, 1.0 / len);
    let x0 = if z[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let y = {
        let c = cross(z, x0);
        scale(c, 1.0 / dot(c, c).sqrt())
    };
    let x = cross(y, z);
    LinkBox { name, segment, center: f3(scale(add(a, b), 0.5)), axes: [f3(x), f3(y), f3(z)], half: [w, w, (len / 2.0) as f32] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn holding_torques_match_the_quasi_static_model() {
        // Holding still, the dynamics' shoulder and elbow torques are the statics of `balance`.
        let b = body(Tier::Mid, BaseKind::Wheel);
        let p = Posture::arm_lift(2.0, false, 0.0);
        let r = simulate(&b, &p, &p, &FeedForward::default(), 200.0, &DynCfg::default());
        let st = b.solve(&p, &FeedForward::default(), 200.0);
        for name in ["shoulder_r", "elbow_r"] {
            let j = r.joints.iter().position(|j| j.name == name).unwrap();
            let dyn_tau = r.samples.last().unwrap().tau_nm[j];
            let st_tau = st.joint(name).unwrap().torque_nm;
            assert!((dyn_tau - st_tau).abs() < 0.03 * st_tau.max(0.5), "{name}: dynamics {dyn_tau} vs statics {st_tau}");
        }
        assert!(r.reached && r.feasible, "{:?}", r.notes);
        // The ZMP of a still robot is the CoM's ground projection.
        let s = r.samples.last().unwrap();
        assert!((s.zmp[0] - s.com[0]).abs() < 2e-3 && (s.zmp[1] - s.com[1]).abs() < 2e-3, "{:?} vs {:?}", s.zmp, s.com);
        assert!((s.margin - st.margin_x.min(st.margin_y)).abs() < 0.02, "{} vs {}", s.margin, st.margin);
    }

    #[test]
    fn the_mass_matrix_is_symmetric_positive_and_matches_a_hand_count() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let r = Robot::new(&b, &Posture::quiet(0.0));
        let q = vec![0.1, 0.2, 1.0, 0.3, 0.5, 0.2];
        let m = r.mass_matrix(&q[..r.dofs()], &RootState::default());
        for i in 0..r.dofs() {
            assert!(m[i][i] > 0.0);
            for j in 0..r.dofs() {
                assert!((m[i][j] - m[j][i]).abs() < 1e-9, "M[{i}][{j}]");
            }
        }
        // Elbow: the forearm and hand about the elbow axis (point masses plus box terms).
        let e = r.joint_index("elbow_r").unwrap();
        let (_, _, m_fore) = b.arm_split();
        let fo = FOREARM_M as f64;
        let hand = b.hand_kg as f64;
        let want = m_fore as f64 * (0.45 * fo).powi(2) + m_fore as f64 * (0.06f64.powi(2) + fo.powi(2)) / 12.0 + hand * (fo + 0.06).powi(2) + hand * (0.08f64.powi(2) + 0.08f64.powi(2)) / 12.0;
        assert!((m[e][e] - want).abs() < 1e-6, "{} vs {want}", m[e][e]);
    }

    #[test]
    fn a_free_arm_swings_without_gaining_energy() {
        // With the actuators off, the arm is a passive pendulum: semi-implicit Euler keeps its energy within 2%.
        let b = body(Tier::Mid, BaseKind::Wheel);
        let r = Robot::new(&b, &Posture::quiet(0.0));
        let n = r.dofs();
        let sh = r.joint_index("shoulder_r").unwrap();
        let mut q = vec![0.0; n];
        q[sh] = 1.2;
        let mut qd = vec![0.0; n];
        let root = RootState::default();
        let energy = |q: &[f64], qd: &[f64]| -> f64 {
            let m = r.mass_matrix(q, &root);
            let ke: f64 = (0..n).map(|i| (0..n).map(|j| 0.5 * qd[i] * m[i][j] * qd[j]).sum::<f64>()).sum();
            let k = r.rnea(q, &vec![0.0; n], &vec![0.0; n], &root, false);
            let pe: f64 = r.links.iter().enumerate().map(|(i, l)| l.mass * G as f64 * k.com[i][2]).sum();
            ke + pe
        };
        let e0 = energy(&q, &qd);
        let dt = 0.001;
        let mut swing = 0.0f64;
        for _ in 0..2000 {
            let m = r.mass_matrix(&q, &root);
            let h = r.rnea(&q, &qd, &vec![0.0; n], &root, true).tau;
            let qdd = cholesky_solve(&m, &h.iter().map(|x| -x).collect::<Vec<_>>()).unwrap();
            for k in 0..n {
                qd[k] += qdd[k] * dt;
                q[k] += qd[k] * dt;
            }
            swing = swing.min(q[sh]);
        }
        let e1 = energy(&q, &qd);
        assert!(swing < -0.8, "the arm swung through the bottom: {swing}");
        assert!((e1 - e0).abs() < 0.02 * e0.abs().max(1.0), "{e0} -> {e1}");
    }

    #[test]
    fn a_weak_servo_elbow_stalls_where_a_qdd_elbow_lifts() {
        // A log (2.4 kg) with bus-servo arms: the non-backdrivable elbow cannot lift it and stalls.
        let ff = FeedForward::default();
        let from = Posture { hand_load_kg: [2.4, 0.0], ..Posture::quiet(0.0) };
        let to = Posture::arm_lift(2.4, false, 0.0);
        let budget = simulate(&body(Tier::Budget, BaseKind::Wheel), &from, &to, &ff, 200.0, &DynCfg::default());
        let mid = simulate(&body(Tier::Mid, BaseKind::Wheel), &from, &to, &ff, 200.0, &DynCfg::default());
        assert!(!budget.feasible, "{:?}", budget.notes);
        let elbow = budget.summary.iter().find(|s| s.name == "elbow_r").unwrap();
        assert!(elbow.saturated_s > 0.0, "{elbow:?}");
        assert!(mid.reached && mid.feasible, "{:?}", mid.notes);
        // The static check says the same (balance: the budget arm cannot lift a log).
        assert!(!body(Tier::Budget, BaseKind::Wheel).lift(2.4, 0.0, &ff, 200.0).0.feasible);
    }

    #[test]
    fn a_late_reaction_dips_the_dynamic_margin() {
        // Pro belly (waist pitch) lifting 4 kg: the feed-forward starts the lean with the arm.
        let b = body(Tier::Pro, BaseKind::HumanFoot);
        let from = Posture { hand_load_kg: [4.0, 0.0], ..Posture::quiet(0.0) };
        let to = Posture::arm_lift(4.0, false, 0.0);
        let cfg = DynCfg::default();
        let with = simulate(&b, &from, &to, &FeedForward::default(), 400.0, &cfg);
        let without = simulate(&b, &from, &to, &FeedForward { enabled: false, gain: 1.0 }, 400.0, &cfg);
        assert!(with.anticipated > 0.0 && without.anticipated == 0.0);
        assert!(with.min_margin >= without.min_margin - 1e-4, "{} vs {}", with.min_margin, without.min_margin);
        assert!((with.final_margin - without.final_margin).abs() < 0.02, "both settle to the same posture");
    }

    #[test]
    fn the_weight_shifts_before_the_foot_lifts() {
        // From two feet to one: the pelvis moves over the stance foot first, so the
        // dynamic ZMP never leaves the support (it did when the foot lifted at t = 0).
        for tier in [Tier::Mid, Tier::Pro] {
            let b = body(tier, BaseKind::HumanFoot);
            let to = Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) };
            let r = simulate(&b, &Posture::quiet(0.0), &to, &FeedForward::default(), 1e3, &DynCfg::default());
            assert!(r.min_margin > 0.0, "{tier:?}: {} at {} s, {:?}", r.min_margin, r.min_margin_t, r.notes);
            let end = r.samples.last().unwrap();
            assert!((end.com[1] + STANCE_WIDTH_M / 2.0).abs() < 0.02, "CoM over the right foot: {:?}", end.com);
            let early = r.pose_at(&b, &to, 0.05).unwrap();
            let late = r.pose_at(&b, &to, 10.0).unwrap();
            assert!(early.support[3] - early.support[2] > late.support[3] - late.support[2], "two feet, then one");
        }
    }

    #[test]
    fn a_start_moves_the_zmp_back_by_the_inertial_term() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let p = Posture { source: Source::Accel, accel_mps2: 1.0, ..Posture::quiet(0.0) };
        let r = simulate(&b, &Posture::quiet(0.0), &p, &FeedForward::default(), 1e3, &DynCfg::default());
        let mid = r.sample_at(0.25).unwrap();
        // x_zmp = x_com - z_com * a / g for a rigid body moving with the base.
        let want = mid.com[0] - mid.com[2] * 1.0 / G;
        assert!((mid.zmp[0] - want).abs() < 0.01, "{} vs {want}", mid.zmp[0]);
        let drive = r.summary.iter().find(|s| s.name == "drive").unwrap();
        assert!(drive.peak_ratio > 0.0);
    }

    #[test]
    fn poses_have_boxes_and_joints_for_every_part() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let b = body(Tier::Mid, base);
            let p = Posture::arm_raise(0.0);
            let r = simulate(&b, &Posture::quiet(0.0), &p, &FeedForward::default(), 1e3, &DynCfg::default());
            let pose = r.pose_at(&b, &p, 0.3).unwrap();
            let names: Vec<&str> = pose.boxes.iter().map(|x| x.name).collect();
            for n in ["pelvis", "trunk", "head", "upper_arm_r", "forearm_r", "hand_r"] {
                assert!(names.contains(&n), "{base:?}: {names:?}");
            }
            assert!(pose.joints.iter().any(|j| j.name == "shoulder_r" && j.q_deg.is_some()));
            if base == BaseKind::HumanFoot {
                assert!(names.contains(&"thigh_l") && pose.joints.iter().any(|j| j.name == "knee_r"));
            } else {
                assert!(names.contains(&"chassis"));
            }
            // The raised arm's hand is in front of the shoulder at the end.
            let end = r.pose_at(&b, &p, 10.0).unwrap();
            let hand = end.boxes.iter().find(|x| x.name == "hand_r").unwrap();
            let sh = end.joints.iter().find(|j| j.name == "shoulder_r").unwrap();
            assert!(hand.center[0] > sh.pos[0] + 0.2, "{:?} vs {:?}", hand.center, sh.pos);
        }
    }
}
