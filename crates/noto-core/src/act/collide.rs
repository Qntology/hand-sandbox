//! The body meeting the world: a wall, a curb, another robot or a person met
//! at speed (`bump`), and an object falling onto one of the robot's links
//! (`struck_by`). Pure functions: the world finds the contact, this module
//! plays the physics of it out (`docs/research/collision.md`).
//!
//! * **Effective mass.** The robot is a rigid body built from the presets'
//!   masses on the pose's links ([`mass_props`]), reduced to the vertical
//!   plane of the impulse ([`Planar`]). Its support (wheel axles, a track's
//!   ends, a foot's heel and toe) are unilateral ground contacts; the obstacle
//!   is one more contact, with the other body's mass for a robot, a person or
//!   a falling object. The contact impulses solve Newton's restitution law
//!   with Coulomb friction (a small complementarity problem by enumeration), so
//!   the mass the obstacle feels comes out of the geometry: the whole robot
//!   when it is hit low and square, less when the hit also turns it. Stuck
//!   friction that asks a point for another speed than a contact's normal
//!   does is tried sliding before the set is given up.
//! * **Force and time.** The contact's force over time follows its law: a
//!   linear spring with viscous damping for soft faces (the bumper's rubber,
//!   a sole, a person's tissue), a Hertz contact with Hunt–Crossley damping
//!   (`F = K·x^1.5·(1 + c·ẋ/v₀)`) for hard ones; the damping is found so the
//!   played contact, which never pulls, rebounds at the restitution it is
//!   given ([`damping_for`]; the closed forms hold only near `e = 1`). A moulded housing is a thin shell: it dents as a shallow spherical
//!   shell under a point load (`k = 4·E·t²/(√(3(1 − ν²))·R)`) in series with
//!   the Hertz contact of its face, far softer than the face alone; on soil
//!   the ground's own stiffness comes in series too (`ground_pair`). The
//!   peak and the contact time come from integrating it with the effective mass.
//! * **Material pairs.** Concrete (`E = 4700·√f'c`, ACI 318-19), plasterboard
//!   (E 2.3 GPa, crushing above 3.02 MPa, Petrone 2016), wood and rock
//!   (estimates), another robot's bumper; restitution 0.55 for every robot
//!   surface (a soft-bumpered light robot on a wall, Abazari 2022, a proxy).
//! * **Kin.** During the pulse the force goes to `kin` as
//!   `KinLoads::body_force` at the contact: the ZMP and margin of the impulse.
//!   The force-angle measure of tip-over stability (Papadopoulos & Rey 1996)
//!   is reported beside it (and a disagreement with the impulse's turn noted).
//! * **Tip.** Wheels and tracks: after the impulse the base turns about the
//!   support edge the solver leaves loaded; it goes over when the turning
//!   energy beats the barrier `M·g·(√(d² + h²) − h)`. The turn is integrated
//!   with `fall`'s rigid body, so a wall that catches the top, a bumper that
//!   strikes the ground ahead of the wheels, and the lifted end landing back
//!   are all seen; going over (on an edge, or past the tip angle in the air
//!   and not back) hands the fall to `fall::fall` ([`TipStart`]).
//!   Legs: the capture point `x + ẋ/ω` (`ω = √(g/z)`) must be reached by a
//!   step that lands one reaction delay and one swing later, within the legs'
//!   reach; a toe caught on a curb (a trip) cannot step until it is lifted
//!   over it, and before a wall the body goes on into it with the speed the
//!   stubbed foot left (a second hit, of the upper body).
//! * **Classification.** ISO/TS 15066's body model (transient contact ≤ 0.5 s,
//!   twice the quasi-static limit except head, face and neck; the robot's own
//!   share of the effective mass, the person's region being the second body
//!   of the impulse already) and the AGV
//!   bumper limits for a person; the trunk accelerometer's 3.52 g impact
//!   threshold for whether the robot itself notices the hit ("collision")
//!   or not ("bump").
//!
//! Engine inputs only (Group A); the collision benchmark scores
//! with other sources.

use serde::Serialize;

use crate::balance::{Body, G, REACTION_DELAY_S};
use crate::base::BaseKind;
use crate::kin::{self, KinLoads, KinPose, KinReport, WHEEL_R_M};
use crate::parts::*;
use crate::surface::{self, Surface};
use crate::wear;

use super::material::{self, ItemMaterial};
use super::{ActEvent, ActKind, ActRollout, ActSample, Outcome, Part};

// ----- what a rollout of this module is -----

/// What a physics rollout of `collide`, `fall` and `cocarry` plays out. These
/// happen to the robot rather than being chosen by it, so they are kept apart
/// from [`ActKind`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysKind {
    /// The robot meets an obstacle (`bump`).
    Collide,
    /// Something falls onto the robot (`struck_by`).
    Hit,
    /// The robot topples (`fall`).
    Fall,
    /// Several robots carry one object (`cocarry`).
    CoCarry,
}

impl PhysKind {
    pub fn id(self) -> &'static str {
        match self {
            PhysKind::Collide => "collide",
            PhysKind::Hit => "hit",
            PhysKind::Fall => "fall",
            PhysKind::CoCarry => "cocarry",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            PhysKind::Collide => "충돌",
            PhysKind::Hit => "맞음",
            PhysKind::Fall => "넘어짐",
            PhysKind::CoCarry => "함께 나르기",
        }
    }
}

/// A played-out physical event: the same parts as an [`ActRollout`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PhysRollout {
    pub kind: PhysKind,
    pub samples: Vec<ActSample>,
    pub events: Vec<ActEvent>,
    pub outcome: Outcome,
    pub duration_s: f32,
    /// Summary numbers for the SDS and the benchmark (each documented where it is set).
    pub metrics: Vec<(&'static str, f32)>,
    pub notes: Vec<String>,
}

impl PhysRollout {
    /// The sample at or before `t` (the last one after the end).
    pub fn sample_at(&self, t: f32) -> Option<&ActSample> {
        let i = self.samples.partition_point(|s| s.t <= t);
        self.samples.get(i.saturating_sub(1))
    }

    pub fn metric(&self, key: &str) -> Option<f32> {
        self.metrics.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }

    /// Events in `(from, to]`.
    pub fn events_between(&self, from: f32, to: f32) -> impl Iterator<Item = &ActEvent> {
        self.events.iter().filter(move |e| e.t > from && e.t <= to)
    }

    /// The same time series as an [`ActRollout`] of `host`'s kind, for a world
    /// that plays actions back (the focus views open on `host`'s parts).
    pub fn into_act(self, host: ActKind) -> ActRollout {
        ActRollout { kind: host, samples: self.samples, events: self.events, outcome: self.outcome, duration_s: self.duration_s, metrics: self.metrics, notes: self.notes }
    }
}

// ----- engine inputs -----

/// Restitution of the robot's surfaces against a wall or the ground: a light,
/// soft-bumpered robot rebounds 0.5–0.6 from a wall (Abazari 2022, a
/// collision-resilient foldable micro quadcopter, visco-elastic model and
/// experiment): the middle. A proxy for every robot surface (bumper, shell,
/// sole): no measured restitution of a plastic shell or bumper on concrete,
/// wood or plasterboard was found (research §1.3).
pub const ROBOT_RESTITUTION: f32 = 0.55;
/// Compressive strength of the concrete walls, MPa (est: a common normal-weight class).
pub const CONCRETE_FC_MPA: f32 = 30.0;
/// Gypsum plasterboard, 12.5 mm standard board in compression: median Young's
/// modulus 2303 MPa (Petrone, Magliulo & Manfredi 2016, 302 tests).
pub const DRYWALL_E_PA: f32 = 2.303e9;
/// The weakest compressive strength of the five boards, Pa (Petrone 2016:
/// 3.02–8.14 MPa): a contact pressing harder dents the board.
pub const DRYWALL_CRUSH_PA: f32 = 3.02e6;
/// Wood across the grain (a stud's or a panel's face), Pa (est: the research
/// found no wood modulus; a softwood's radial modulus is a few percent of
/// its ~10 GPa along the grain).
pub const WOOD_E_PA: f32 = 1.0e9;
/// A rock face, Pa (est: granite-like; no rock modulus in the research).
pub const ROCK_E_PA: f32 = 50.0e9;
/// A hard plastic housing (ABS or PC), Pa (est: no shell modulus in the research).
pub const SHELL_E_PA: f32 = 2.2e9;
/// Radius of a housing's edge or corner where it meets a flat face, metres
/// (est: the drawn trunk and head boxes with rounded edges).
pub const SHELL_RADIUS_M: f32 = 0.10;
/// Wall thickness of a moulded housing, metres (est: injection-moulded robot
/// covers are 2–3 mm thick; the thicker end).
pub const SHELL_T_M: f32 = 0.003;
/// Poisson's ratios (est, typical): concrete, plasterboard, wood, rock, plastic.
pub const NU_CONCRETE: f32 = 0.2;
pub const NU_DRYWALL: f32 = 0.2;
pub const NU_WOOD: f32 = 0.3;
pub const NU_ROCK: f32 = 0.25;
pub const NU_SHELL: f32 = 0.35;

/// Young's modulus of the concrete walls, Pa: `E_c = 4700·√f'c` MPa (ACI 318-19 §19.2.2.1.b).
pub fn concrete_e_pa() -> f32 {
    4700.0 * CONCRETE_FC_MPA.sqrt() * 1e6
}

/// Point-load stiffness of a housing of curvature radius `r_m`, N/m: a
/// shallow spherical shell of [`SHELL_T_M`] wall dents `w = √(3(1 − ν²))/4 · F·R/(E·t²)`
/// under a point force (Reissner's shallow-shell solution, classical thin-shell
/// theory), so `k = 4·E·t²/(√(3(1 − ν²))·R)`.
pub fn shell_k(r_m: f32) -> f32 {
    4.0 * SHELL_E_PA * SHELL_T_M * SHELL_T_M / ((3.0 * (1.0 - NU_SHELL * NU_SHELL)).sqrt() * r_m.max(1e-3))
}

/// The trunk accelerometer's impact threshold, g: falls are told from daily
/// activity at 3.52 g (Bourke, O'Brien & Lyons 2007, upper fall threshold;
/// confirmed through a secondary source). A hit the robot's own IMU reads at
/// or above it is a detected "collision"; below it the robot only "bumps".
pub const IMPACT_DETECT_G: f32 = 3.52;
/// The free-fall dip of the same accelerometer, g (lower fall threshold 0.41 g).
pub const FREE_FALL_G: f32 = 0.41;
/// The base accelerometer's low-pass corner, Hz (Hello Robot stretch_body
/// factory parameters, accel_LPF 20 Hz): the reading lags the force by one
/// filter time constant.
pub const ACCEL_LPF_HZ: f32 = 20.0;
/// Over-tilt alert, radians (stretch_body over_tilt_alert 0.17; the unit is
/// not documented and read as radians, about 9.7°).
pub const OVER_TILT_RAD: f32 = 0.17;

/// Energy of a dropped object that seriously hurts an unprotected person, J
/// (the DROPS rule of thumb, Britannia P&I 2024): flags a hit.
pub const DROPPED_SERIOUS_J: f32 = 40.0;
/// Force an ANSI/ISEA Z89.1 Type I helmet may pass to the head in its
/// 54.5 J striker test, N (ISEA 2024): a hit on the robot's head above it is
/// flagged as harmful to the head's sensors.
pub const HEAD_FORCE_LIMIT_N: f32 = 4450.0;

/// AGV bumper force limits on a person, N (Norcross, Bostelman & Falco 2015,
/// NISTIR 8029): ANSI/ITSDF B56.5-2012 134 N; EN 1525 750 N on the
/// horizontal test piece (torso, thigh), 250 N on the vertical one (lower
/// leg), 400 N at the compression of a stop from full speed.
pub const B56_BUMPER_N: f32 = 134.0;
pub const EN1525_TORSO_N: f32 = 750.0;
pub const EN1525_LEG_N: f32 = 250.0;
pub const EN1525_STOP_N: f32 = 400.0;

// ----- ISO/TS 15066 body model -----

/// A body region of ISO/TS 15066's body model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    Skull,
    Face,
    Neck,
    BackShoulders,
    Chest,
    Abdomen,
    Pelvis,
    UpperArm,
    LowerArm,
    Hand,
    Thigh,
    LowerLeg,
}

/// A region's quasi-static force limit, effective spring constant and effective mass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RegionLimit {
    pub f_qs_n: f32,
    pub k_n_per_m: f32,
    pub m_kg: f32,
}

/// Contacts up to this long are transient, the body part free to recoil;
/// longer ones are quasi-static, the part clamped (Kirschner, Mansfeld et al.
/// 2022 on ISO/TS 15066's inconsistent definitions).
pub const TRANSIENT_MAX_S: f32 = 0.5;
/// Transient over quasi-static force limit (an industry rule of thumb, Robotiq
/// 2016); none for the skull, the face and the neck, where a transient
/// contact is not allowed at all.
pub const TRANSIENT_FACTOR: f32 = 2.0;

impl Region {
    pub const ALL: [Region; 12] = [Region::Skull, Region::Face, Region::Neck, Region::BackShoulders, Region::Chest, Region::Abdomen, Region::Pelvis, Region::UpperArm, Region::LowerArm, Region::Hand, Region::Thigh, Region::LowerLeg];

    /// ISO/TS 15066:2016 Annex A as reproduced by Ghanbarzadeh & Najafi 2023
    /// (arXiv:2311.13814, Table 1): N, N/mm (here N/m), kg.
    pub fn limit(self) -> RegionLimit {
        let (f, k_mm, m) = match self {
            Region::Skull => (130.0, 150.0, 4.4),
            Region::Face => (65.0, 75.0, 4.4),
            Region::Neck => (150.0, 50.0, 1.2),
            Region::BackShoulders => (210.0, 35.0, 40.0),
            Region::Chest => (140.0, 25.0, 40.0),
            Region::Abdomen => (110.0, 10.0, 40.0),
            Region::Pelvis => (180.0, 25.0, 40.0),
            Region::UpperArm => (150.0, 30.0, 3.0),
            Region::LowerArm => (160.0, 40.0, 2.0),
            Region::Hand => (140.0, 75.0, 0.6),
            Region::Thigh => (220.0, 50.0, 75.0),
            Region::LowerLeg => (130.0, 60.0, 75.0),
        };
        RegionLimit { f_qs_n: f, k_n_per_m: k_mm * 1000.0, m_kg: m }
    }

    /// The transient force limit, N (`None`: no transient contact allowed).
    pub fn transient_n(self) -> Option<f32> {
        match self {
            Region::Skull | Region::Face | Region::Neck => None,
            _ => Some(TRANSIENT_FACTOR * self.limit().f_qs_n),
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Region::Skull => "skull",
            Region::Face => "face",
            Region::Neck => "neck",
            Region::BackShoulders => "back_shoulders",
            Region::Chest => "chest",
            Region::Abdomen => "abdomen",
            Region::Pelvis => "pelvis",
            Region::UpperArm => "upper_arm",
            Region::LowerArm => "lower_arm",
            Region::Hand => "hand",
            Region::Thigh => "thigh",
            Region::LowerLeg => "lower_leg",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Region::Skull => "머리(이마)",
            Region::Face => "얼굴",
            Region::Neck => "목",
            Region::BackShoulders => "등·어깨",
            Region::Chest => "가슴",
            Region::Abdomen => "배",
            Region::Pelvis => "골반",
            Region::UpperArm => "위팔",
            Region::LowerArm => "아래팔",
            Region::Hand => "손",
            Region::Thigh => "허벅지",
            Region::LowerLeg => "종아리",
        }
    }
}

/// ISO/TS 15066's power-and-force-limiting check of a contact with a person.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PflCheck {
    pub region: Region,
    /// The robot's and the person's reduced mass, kg: `(1/m_H + 1/m_R)⁻¹`.
    pub reduced_mass_kg: f32,
    /// The peak force by the standard's spring model, `v·√(μ·k)`, N.
    pub f_peak_n: f32,
    /// The limit it is judged against (transient or quasi-static), N; `None` where no contact is allowed.
    pub f_limit_n: Option<f32>,
    pub transient: bool,
    /// The largest relative speed under the limit, `F_max/√(μ·k)`, m/s.
    pub v_max_mps: Option<f32>,
    pub ok: bool,
}

/// The PFL check of a robot of effective mass `m_robot_kg` (its own share,
/// not yet reduced with the person's) meeting `region` at `v_rel` m/s for
/// `contact_s` seconds.
pub fn pfl(region: Region, m_robot_kg: f32, v_rel: f32, contact_s: f32) -> PflCheck {
    let l = region.limit();
    let mu = 1.0 / (1.0 / l.m_kg + 1.0 / m_robot_kg.max(1e-3));
    let f = v_rel.abs() * (mu * l.k_n_per_m).sqrt();
    let transient = contact_s <= TRANSIENT_MAX_S;
    let lim = if transient { region.transient_n() } else { Some(l.f_qs_n) };
    let v_max = lim.map(|f_max| f_max / (mu * l.k_n_per_m).sqrt());
    PflCheck { region, reduced_mass_kg: mu, f_peak_n: f, f_limit_n: lim, transient, v_max_mps: v_max, ok: lim.is_some_and(|f_max| f <= f_max) }
}

/// The AGV bumper limit for a person's `region`, N (EN 1525: the vertical
/// test piece for the lower leg, the horizontal one otherwise).
pub fn agv_bumper_limit_n(region: Region) -> f32 {
    match region {
        Region::LowerLeg => EN1525_LEG_N,
        _ => EN1525_TORSO_N,
    }
}

// ----- materials of a contact -----

/// The part of the robot that meets the obstacle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Contactor {
    /// The rubber bumper of a wheeled or tracked base.
    Bumper,
    /// A hard housing: trunk, head, arm, chassis.
    Shell,
    /// A foot's sole or a shin.
    Sole,
    /// A wheel's tyre (a curb).
    Tyre,
}

impl Contactor {
    pub fn id(self) -> &'static str {
        match self {
            Contactor::Bumper => "bumper",
            Contactor::Shell => "shell",
            Contactor::Sole => "sole",
            Contactor::Tyre => "tyre",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Contactor::Bumper => "범퍼",
            Contactor::Shell => "외피",
            Contactor::Sole => "발바닥",
            Contactor::Tyre => "바퀴",
        }
    }

    /// The linear stiffness of a soft contactor, N/m (`None` for the hard shell).
    pub fn linear_k(self) -> Option<f32> {
        match self {
            Contactor::Bumper => Some(super::push::BUMPER_STIFFNESS),
            Contactor::Sole => Some(super::gait::SOLE_STIFFNESS),
            Contactor::Tyre => Some(super::drive::TYRE_STIFFNESS),
            Contactor::Shell => None,
        }
    }
}

/// What the robot meets.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Obstacle {
    Concrete,
    Drywall,
    Wood,
    Rock,
    /// Another robot (its bumper or shell), as a body of this mass.
    Robot { mass_kg: f32 },
    /// A person's body region (ISO/TS 15066).
    Person(Region),
}

impl Obstacle {
    pub fn id(&self) -> &'static str {
        match self {
            Obstacle::Concrete => "concrete",
            Obstacle::Drywall => "drywall",
            Obstacle::Wood => "wood",
            Obstacle::Rock => "rock",
            Obstacle::Robot { .. } => "robot",
            Obstacle::Person(_) => "person",
        }
    }

    pub fn label_ko(&self) -> &'static str {
        match self {
            Obstacle::Concrete => "콘크리트 벽",
            Obstacle::Drywall => "석고보드 벽",
            Obstacle::Wood => "나무 벽",
            Obstacle::Rock => "바위",
            Obstacle::Robot { .. } => "다른 로봇",
            Obstacle::Person(_) => "사람",
        }
    }

    /// Young's modulus and Poisson's ratio of a rigid obstacle's face.
    fn elastic(&self) -> Option<(f32, f32)> {
        match self {
            Obstacle::Concrete => Some((concrete_e_pa(), NU_CONCRETE)),
            Obstacle::Drywall => Some((DRYWALL_E_PA, NU_DRYWALL)),
            Obstacle::Wood => Some((WOOD_E_PA, NU_WOOD)),
            Obstacle::Rock => Some((ROCK_E_PA, NU_ROCK)),
            Obstacle::Robot { .. } => Some((SHELL_E_PA, NU_SHELL)),
            Obstacle::Person(_) => None,
        }
    }

    /// Mass of the obstacle when it moves, kg (`None`: fixed).
    pub fn mass_kg(&self) -> Option<f32> {
        match self {
            Obstacle::Robot { mass_kg } => Some(*mass_kg),
            Obstacle::Person(r) => Some(r.limit().m_kg),
            _ => None,
        }
    }

    /// The crushing strength of the obstacle's face, Pa.
    fn crush_pa(&self) -> Option<f32> {
        matches!(self, Obstacle::Drywall).then_some(DRYWALL_CRUSH_PA)
    }
}

/// How a contact's force grows with its deflection.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Law {
    /// A linear spring `F = k·x`, N/m (rubber, a sole, tissue).
    Linear { k: f32 },
    /// A Hertz contact `F = K·x^1.5`, N/m^1.5 (two hard faces).
    Hertz { k: f32 },
    /// A linear spring (soil, a soft face) in series with a Hertz contact (a hard corner).
    Series { k_lin: f32, k_hertz: f32 },
}

impl Law {
    /// Elastic force at deflection `x`, N.
    pub fn force(&self, x: f32) -> f32 {
        let x = x.max(0.0);
        match *self {
            Law::Linear { k } => k * x,
            Law::Hertz { k } => k * x * x.sqrt(),
            Law::Series { k_lin, k_hertz } => {
                // The deflections add: x = F/k + (F/K)^(2/3). With y = (F/K)^(1/3) that is
                // the cubic a·y³ + y² = x (a = K/k), convex and rising in y: Newton from
                // above (the smaller of the two one-part roots) falls onto its root.
                if x <= 0.0 {
                    return 0.0;
                }
                let a = k_hertz / k_lin.max(1e-6);
                let mut y = x.sqrt().min((x / a.max(1e-12)).cbrt());
                for _ in 0..40 {
                    let h = (a * y + 1.0) * y * y - x;
                    let d = (3.0 * a * y + 2.0) * y;
                    if d <= 0.0 {
                        break;
                    }
                    let next = y - h / d;
                    let done = (y - next).abs() <= 1e-7 * y;
                    y = next.max(0.0);
                    if done {
                        break;
                    }
                }
                k_hertz * y * y * y
            }
        }
    }

    /// The Hertz part's stiffness, N/m^1.5 (`None` for a linear law).
    pub fn hertz_part(&self) -> Option<f32> {
        match *self {
            Law::Hertz { k } => Some(k),
            Law::Series { k_hertz, .. } => Some(k_hertz),
            Law::Linear { .. } => None,
        }
    }

    /// A time scale of a contact of mass `m` at speed `v`, s (for the integration step).
    fn time_scale(&self, m: f32, v: f32) -> f32 {
        let lin = |k: f32| std::f32::consts::PI * (m / k.max(1.0)).sqrt();
        let hz = |k: f32| {
            let xm = (5.0 * m * v * v / (4.0 * k.max(1.0))).powf(0.4);
            2.94 * xm / v.max(1e-3)
        };
        match *self {
            Law::Linear { k } => lin(k),
            Law::Hertz { k } => hz(k),
            Law::Series { k_lin, k_hertz } => lin(k_lin) + hz(k_hertz),
        }
    }
}

/// Hertz contact modulus of two faces: `1/E* = (1 − ν₁²)/E₁ + (1 − ν₂²)/E₂`.
pub fn e_star(e1: f32, nu1: f32, e2: f32, nu2: f32) -> f32 {
    1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)
}

/// Hertz stiffness of a sphere of radius `r` on a flat face, `K = 4/3·E*·√R`, N/m^1.5.
pub fn hertz_k(e_star: f32, r: f32) -> f32 {
    4.0 / 3.0 * e_star * r.sqrt()
}

/// Springs in series, N/m.
pub fn series_k(a: f32, b: f32) -> f32 {
    1.0 / (1.0 / a.max(1e-6) + 1.0 / b.max(1e-6))
}

/// The contact law and restitution of a robot part on an obstacle.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PairLaw {
    pub law: Law,
    pub e: f32,
}

/// The law of a contactor meeting an obstacle: a soft contactor is a linear
/// spring (the obstacle's face far stiffer; another robot's bumper in series;
/// a person's region in series), the hard housing a shell ([`shell_k`]) in
/// series with the Hertz contact of its corner on the face (on a person, the
/// region's spring in series with the shell).
pub fn pair(contactor: Contactor, obstacle: &Obstacle) -> PairLaw {
    let law = match (contactor.linear_k(), obstacle) {
        (Some(k), Obstacle::Person(r)) => Law::Linear { k: series_k(k, r.limit().k_n_per_m) },
        (None, Obstacle::Person(r)) => Law::Linear { k: series_k(shell_k(SHELL_RADIUS_M), r.limit().k_n_per_m) },
        (Some(k), Obstacle::Robot { .. }) => Law::Linear { k: series_k(k, super::push::BUMPER_STIFFNESS) },
        (Some(k), _) => Law::Linear { k },
        (None, o) => {
            let (e, nu) = o.elastic().unwrap_or((concrete_e_pa(), NU_CONCRETE));
            // Two shells meet corner to corner: the radii and the shells in series.
            let two = matches!(o, Obstacle::Robot { .. });
            let r = if two { 0.5 * SHELL_RADIUS_M } else { SHELL_RADIUS_M };
            let k_lin = if two { 0.5 * shell_k(SHELL_RADIUS_M) } else { shell_k(SHELL_RADIUS_M) };
            Law::Series { k_lin, k_hertz: hertz_k(e_star(SHELL_E_PA, NU_SHELL, e, nu), r) }
        }
    };
    PairLaw { law, e: ROBOT_RESTITUTION }
}

/// The law of a robot part striking the ground (a fall, a wheel landing
/// back): a soft contactor on a hard ground is its own spring, on soil in
/// series with the ground's ([`material::ground_stiffness`]); the housing is
/// the shell in series with its Hertz contact on concrete or rock, or with the
/// soil's spring. Restitution is the robot's on a hard ground, lowered on
/// soft ones in the proportion a rock block's landing is ([`material::restitution`]).
pub fn ground_pair(contactor: Contactor, ground: Surface) -> PairLaw {
    let hard = matches!(ground, Surface::Firm | Surface::Rock);
    let e = if hard { ROBOT_RESTITUTION } else { ROBOT_RESTITUTION * material::restitution(ItemMaterial::Stone, ground) / material::restitution(ItemMaterial::Stone, Surface::Rock) };
    let soil = material::ground_stiffness(ground);
    let law = match (contactor.linear_k(), hard) {
        (Some(k), true) => Law::Linear { k },
        (Some(k), false) => Law::Linear { k: series_k(k, soil) },
        (None, true) => {
            let (eg, nu) = if ground == Surface::Rock { (ROCK_E_PA, NU_ROCK) } else { (concrete_e_pa(), NU_CONCRETE) };
            Law::Series { k_lin: shell_k(SHELL_RADIUS_M), k_hertz: hertz_k(e_star(SHELL_E_PA, NU_SHELL, eg, nu), SHELL_RADIUS_M) }
        }
        (None, false) => Law::Linear { k: series_k(shell_k(SHELL_RADIUS_M), soil) },
    };
    PairLaw { law, e: e.max(0.02) }
}

/// The force of one contact over time.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Pulse {
    pub peak_n: f32,
    pub t_peak_s: f32,
    pub contact_s: f32,
    /// Rebound over approach speed the law gives.
    pub e: f32,
    pub max_defl_m: f32,
    /// `(t, F)` over the contact.
    pub series: Vec<(f32, f32)>,
}

/// Plays a contact of mass `m` closing at `v` through `law` with restitution
/// `e` out: a linear spring with viscous damping (`F = k·x + 2ζ·√(k·m)·ẋ`), a
/// Hertz (or series) contact with Hunt–Crossley damping
/// (`F = f(x)·(1 + c·ẋ/v₀)`). The contact never pulls: it ends when the force
/// returns to zero. The closed forms of the damping (`ζ = −ln e/√(π² + ln² e)`
/// for a spring that may pull; Hunt & Crossley's `c = 3(1 − e)/2`, which holds
/// for `e` near 1) give back more than `e` once the force is cut at zero
/// (0.25 for 0.1 on the linear law, 0.69 for 0.55 on a Hertz contact), so the
/// damping is found by a root search on the played rebound
/// ([`damping_for`]): the pulse gives back the restitution it is given, and
/// its force over time is the one of that restitution.
pub fn pulse(law: &Law, e: f32, m: f32, v: f32) -> Pulse {
    let v0 = v.abs().max(1e-6);
    let e = e.clamp(0.01, 0.999);
    let m = m.max(1e-4);
    let ts = law.time_scale(m, v0);
    let c = damping_for(law, e, m, v0, ts);
    play_contact(law, c, m, v0, (ts / 1500.0).max(1e-7), true)
}

/// The damping (the linear law's ratio ζ, the Hunt–Crossley factor `c` of the
/// others) with which a contact of mass `m` closing at `v0` through `law`
/// rebounds at `e`: Illinois regula falsi on the played rebound (a coarser
/// step than the pulse's), bracketed by no damping (all given back) and the
/// damping whose force cut caps the rebound at `e` (`v₀/c`; `v₀/(2ζ)` for the
/// linear law). Deterministic: the same inputs give the same damping.
pub fn damping_for(law: &Law, e: f32, m: f32, v0: f32, ts: f32) -> f32 {
    if e >= 0.999 {
        return 0.0;
    }
    let dt = (ts / 400.0).max(1e-7);
    let g = |c: f32| play_contact(law, c, m, v0, dt, false).e - e;
    let (mut a, mut fa) = (0.0f32, g(0.0));
    if fa <= 0.0 {
        return 0.0;
    }
    let mut b = 1.05 * if matches!(law, Law::Linear { .. }) { 0.5 / e } else { 1.0 / e };
    let mut fb = g(b);
    for _ in 0..8 {
        if fb <= 0.0 {
            break;
        }
        b *= 2.0;
        fb = g(b);
    }
    if fb > 0.0 {
        return b;
    }
    let mut c = b;
    let mut side = 0i8;
    for _ in 0..40 {
        c = (a * fb - b * fa) / (fb - fa);
        let fc = g(c);
        if fc.abs() < 2e-4 || (b - a).abs() < 1e-6 * b.max(1e-6) {
            break;
        }
        if fc * fb > 0.0 {
            b = c;
            fb = fc;
            if side == -1 {
                fa *= 0.5;
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb *= 0.5;
            }
            side = 1;
        }
    }
    c
}

/// One contact played out with the damping `c` ([`damping_for`]) at the step
/// `dt` (RK4); the force series only when `record`.
fn play_contact(law: &Law, c: f32, m: f32, v0: f32, dt: f32, record: bool) -> Pulse {
    let force = |x: f32, xd: f32| -> f32 {
        match *law {
            Law::Linear { k } => (k * x + 2.0 * c * (k * m).sqrt() * xd).max(0.0),
            _ => (law.force(x) * (1.0 + c * xd / v0)).max(0.0),
        }
    };
    let (mut x, mut xd, mut t) = (0.0f32, v0, 0.0f32);
    let mut peak = (0.0f32, 0.0f32);
    let mut series = vec![(0.0, 0.0)];
    let mut max_x = 0.0f32;
    let every = 25usize;
    for i in 0..60_000usize {
        let acc = |x: f32, xd: f32| -force(x, xd) / m;
        let (k1x, k1v) = (xd, acc(x, xd));
        let (k2x, k2v) = (xd + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, xd + 0.5 * dt * k1v));
        let (k3x, k3v) = (xd + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, xd + 0.5 * dt * k2v));
        let (k4x, k4v) = (xd + dt * k3v, acc(x + dt * k3x, xd + dt * k3v));
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        xd += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        t += dt;
        max_x = max_x.max(x);
        let f = force(x, xd);
        if f > peak.0 {
            peak = (f, t);
        }
        if record && i % every == 0 {
            series.push((t, f));
        }
        if x <= 0.0 || (xd < 0.0 && f <= 0.0) {
            break;
        }
    }
    if record {
        series.push((t, 0.0));
    }
    Pulse { peak_n: peak.0, t_peak_s: peak.1, contact_s: t, e: (-xd / v0).clamp(0.0, 1.0), max_defl_m: max_x, series }
}

// ----- mass properties -----

/// Mass, CoM and inertia of the robot in a pose (base frame).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MassProps {
    pub mass_kg: f32,
    pub com: [f32; 3],
    /// Inertia tensor about the CoM, kg·m².
    pub inertia: [[f32; 3]; 3],
}

impl MassProps {
    /// Moment of inertia about an axis through the CoM.
    pub fn about(&self, axis: [f32; 3]) -> f32 {
        let a = unit3(axis);
        (0..3).map(|i| (0..3).map(|j| a[i] * self.inertia[i][j] * a[j]).sum::<f32>()).sum()
    }
}

/// The mass a link box of `kin` carries: the presets' segment masses on the
/// boxes that draw them (the hand on the palm, the battery on the trunk or
/// the pelvis).
fn box_mass(body: &Body, name: &str) -> f32 {
    let (m_up, _, m_fore) = body.arm_split();
    let share = kin::thigh_share(body);
    match name {
        "head" => body.head.mass_kg,
        "trunk" => body.chest.mass_kg + if body.plan.battery_low { 0.0 } else { body.chest.battery_kg },
        "pelvis" => body.belly.mass_kg + if body.plan.battery_low { body.chest.battery_kg } else { 0.0 },
        "upper_arm_r" | "upper_arm_l" => m_up,
        "forearm_r" | "forearm_l" => m_fore,
        "palm_r" | "palm_l" => body.hand_kg,
        "thigh_r" | "thigh_l" => body.leg.mass_kg * share,
        "shank_r" | "shank_l" => body.leg.mass_kg * (1.0 - share),
        "foot_r" | "foot_l" => body.leg.foot_mass_kg,
        "chassis" => body.chassis.mass_kg,
        "mast" => body.mast_kg(),
        _ => 0.0,
    }
}

/// The robot's mass properties in a pose, from the link boxes `kin` draws
/// (each a uniform solid box of its segment's mass) and whatever mass the
/// boxes do not carry (cargo, parts a box does not draw) as one point that
/// puts the CoM where `kin` has it.
pub fn mass_props(body: &Body, pose: &KinPose) -> MassProps {
    mass_props_of(body, &kin::solve(body, pose, &KinLoads::default()))
}

/// [`mass_props`] of a solved pose (its boxes drawn).
pub fn mass_props_of(body: &Body, r: &KinReport) -> MassProps {
    let mut parts: Vec<(f32, [f32; 3], [[f32; 3]; 3])> = Vec::new();
    for b in &r.boxes {
        let m = box_mass(body, b.name);
        if m <= 0.0 {
            continue;
        }
        let h = b.half;
        let own = [m * (h[1] * h[1] + h[2] * h[2]) / 3.0, m * (h[0] * h[0] + h[2] * h[2]) / 3.0, m * (h[0] * h[0] + h[1] * h[1]) / 3.0];
        let mut i3 = [[0.0f32; 3]; 3];
        for k in 0..3 {
            let a = b.axes[k];
            for p in 0..3 {
                for q in 0..3 {
                    i3[p][q] += own[k] * a[p] * a[q];
                }
            }
        }
        parts.push((m, b.center, i3));
    }
    let boxed: f32 = parts.iter().map(|p| p.0).sum();
    let total = r.mass_kg;
    if boxed > total && boxed > 0.0 {
        let s = total / boxed;
        for p in parts.iter_mut() {
            p.0 *= s;
            for row in p.2.iter_mut() {
                for v in row.iter_mut() {
                    *v *= s;
                }
            }
        }
    }
    let boxed: f32 = parts.iter().map(|p| p.0).sum();
    let rest = (total - boxed).max(0.0);
    if rest > 1e-4 {
        let mut c = scale3(r.com, total);
        for p in &parts {
            c = sub3(c, scale3(p.1, p.0));
        }
        parts.push((rest, scale3(c, 1.0 / rest), [[0.0; 3]; 3]));
    }
    let com = r.com;
    let mut inertia = [[0.0f32; 3]; 3];
    for (m, c, own) in &parts {
        let d = sub3(*c, com);
        let dd = dot3(d, d);
        for p in 0..3 {
            for q in 0..3 {
                inertia[p][q] += own[p][q] + m * (if p == q { dd } else { 0.0 } - d[p] * d[q]);
            }
        }
    }
    MassProps { mass_kg: total, com, inertia }
}

// ----- the body in the plane of an impulse -----

/// The robot in the vertical plane of a horizontal direction `dir`: `u` along
/// `dir`, `z` up, both from the CoM.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Planar {
    pub mass_kg: f32,
    /// Moment of inertia about the CoM about the horizontal axis square to `dir` (`ẑ × dir`), kg·m².
    pub inertia: f32,
    /// CoM height above the ground, metres.
    pub com_z: f32,
    /// Horizontal distance from the CoM to the support's edge along `dir` and against it, metres.
    pub ahead_m: f32,
    pub behind_m: f32,
    /// Unit horizontal direction (base frame x, y).
    pub dir: [f32; 2],
    /// The CoM in the base frame.
    pub com: [f32; 3],
}

impl Planar {
    /// The body `mp` on the support rectangle `[x0, x1, y0, y1]`, in the plane of `dir`.
    pub fn of(mp: &MassProps, support: [f32; 4], dir: [f32; 2]) -> Planar {
        let n = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-6);
        let u = [dir[0] / n, dir[1] / n];
        let corners = [[support[0], support[2]], [support[0], support[3]], [support[1], support[2]], [support[1], support[3]]];
        let proj = |c: &[f32; 2]| c[0] * u[0] + c[1] * u[1];
        let hi = corners.iter().map(proj).fold(f32::NEG_INFINITY, f32::max);
        let lo = corners.iter().map(proj).fold(f32::INFINITY, f32::min);
        let cu = mp.com[0] * u[0] + mp.com[1] * u[1];
        Planar { mass_kg: mp.mass_kg, inertia: mp.about(Planar::axis_of(u)), com_z: mp.com[2], ahead_m: hi - cu, behind_m: cu - lo, dir: u, com: mp.com }
    }

    /// The tip axis `ẑ × dir`: turning about it moves the top along `dir`.
    pub fn axis_of(u: [f32; 2]) -> [f32; 3] {
        [-u[1], u[0], 0.0]
    }

    pub fn axis(&self) -> [f32; 3] {
        Planar::axis_of(self.dir)
    }

    /// A base-frame point in the plane, from the CoM: `(u, z)`.
    pub fn to_plane(&self, p: [f32; 3]) -> [f32; 2] {
        [(p[0] - self.com[0]) * self.dir[0] + (p[1] - self.com[1]) * self.dir[1], p[2] - self.com[2]]
    }

    /// A plane vector `(u, z)` as a base-frame vector.
    pub fn to_base(&self, v: [f32; 2]) -> [f32; 3] {
        [v[0] * self.dir[0], v[0] * self.dir[1], v[1]]
    }

    /// The barrier the CoM must climb to go over the edge `d` from it horizontally, J.
    pub fn barrier_j(&self, d: f32) -> f32 {
        self.mass_kg * G * ((d * d + self.com_z * self.com_z).sqrt() - self.com_z)
    }

    /// The tilt at which the CoM is over the edge `d` from it, radians.
    pub fn tip_angle(&self, d: f32) -> f32 {
        d.max(0.0).atan2(self.com_z.max(1e-3))
    }
}

/// A body point's offset from the CoM after the body turns `th` radians about
/// the plane's axis (the top moving `+u`).
pub fn rot2(q: [f32; 2], th: f32) -> [f32; 2] {
    let (s, c) = th.sin_cos();
    [q[0] * c + q[1] * s, -q[0] * s + q[1] * c]
}

// ----- impulses -----

/// One contact of the planar impulse problem.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Row {
    /// The contact point from the CoM, `(u, z)`.
    pub r: [f32; 2],
    /// Unit direction of the impulse on the robot (into the robot from the other side).
    pub n: [f32; 2],
    pub e: f32,
    /// A friction row: the index of its normal row; it holds the point still along `n` up to `mu` of the normal impulse.
    pub tangent_of: Option<usize>,
    pub mu: f32,
    /// The other body's inverse mass (0: the ground, a wall) and its velocity along `n`.
    pub inv_m_other: f32,
    pub v_other: f32,
}

impl Row {
    pub fn normal(r: [f32; 2], n: [f32; 2], e: f32) -> Row {
        Row { r, n, e, tangent_of: None, mu: 0.0, inv_m_other: 0.0, v_other: 0.0 }
    }

    pub fn tangent(of: usize, r: [f32; 2], n: [f32; 2], mu: f32) -> Row {
        Row { r, n, e: 0.0, tangent_of: Some(of), mu, inv_m_other: 0.0, v_other: 0.0 }
    }

    /// The ω coefficient of the point's velocity along `n` (and the torque of a unit impulse along `n`).
    fn c(&self, d: [f32; 2]) -> f32 {
        self.r[1] * d[0] - self.r[0] * d[1]
    }

    /// Relative velocity along `n` (negative: closing) of a body state `[v_u, v_z, ω]`.
    pub fn vel(&self, s: [f32; 3]) -> f32 {
        self.n[0] * s[0] + self.n[1] * s[1] + self.c(self.n) * s[2] - self.v_other
    }
}

/// The impulses of a set of contacts.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Impulses {
    /// The body state after: `[v_u, v_z, ω]`.
    pub s: [f32; 3],
    pub lambda: Vec<f32>,
    pub sliding: Vec<bool>,
    /// A consistent set of contacts was found (otherwise no impulse at all: `s` is the state before).
    pub found: bool,
}

fn gauss(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
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

/// Constraint-force mixing of the impulse problem, a fraction of its largest
/// diagonal term: small enough to leave a well-posed set unchanged (the
/// constraints then hold to about this fraction of the closing speed).
const SOFT_EPS: f64 = 1e-7;

/// Solves the contact rows' system `A·λ = b` in double precision with a
/// little constraint-force mixing (`(A + ε·I)·λ = b`): contacts that hold the
/// same motion twice (both support edges of a level base gripping by
/// friction, a body landing flat on two corners at one height) then share it
/// as the minimum-norm split instead of leaving the system singular. A set
/// that asks one motion for two different speeds has no such split: its
/// residual stays large and it is rejected (`None`).
fn solve_soft(a: &[Vec<f32>], b: &[f32]) -> Option<Vec<f32>> {
    let n = b.len();
    let dmax = (0..n).map(|i| a[i][i].abs() as f64).fold(0.0f64, f64::max).max(1e-12);
    let eps = SOFT_EPS * dmax;
    let mut m: Vec<Vec<f64>> = a.iter().map(|r| r.iter().map(|&v| v as f64).collect()).collect();
    let mut y: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] += eps;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        y.swap(col, piv);
        for row in col + 1..n {
            let f = m[row][col] / m[col][col];
            for k in col..n {
                m[row][k] -= f * m[col][k];
            }
            y[row] -= f * y[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| m[i][k] * x[k]).sum();
        x[i] = (y[i] - s) / m[i][i];
    }
    // The unmixed residual: the rows must hold what they were asked to.
    let scale = b.iter().map(|v| v.abs() as f64).fold(1e-3f64, f64::max);
    let worst = (0..n).map(|i| ((0..n).map(|k| a[i][k] as f64 * x[k]).sum::<f64>() - b[i] as f64).abs()).fold(0.0f64, f64::max);
    (worst <= 1e-3 * scale && x.iter().all(|v| v.is_finite())).then(|| x.iter().map(|&v| v as f32).collect())
}

/// [`solve_impulses`] of `rows`, or, when no set with the other contacts is
/// consistent, of the first row alone (the obstacle's contact; the support
/// then takes no impulse). The flag says whether it fell back.
pub(crate) fn solve_or_alone(m: f32, inertia: f32, s0: [f32; 3], rows: &[Row]) -> (Impulses, bool) {
    let imp = solve_impulses(m, inertia, s0, rows);
    if imp.found || rows.is_empty() {
        return (imp, false);
    }
    let mut alone = solve_impulses(m, inertia, s0, &rows[..1]);
    alone.lambda.resize(rows.len(), 0.0);
    alone.sliding.resize(rows.len(), false);
    (alone, true)
}

/// Solves the contacts' impulses on a body of mass `m` and inertia `inertia`
/// moving at `s0`: every closing contact rebounds by its restitution
/// (Newton), resting ones do not sink, friction rows hold within their cone
/// or slide at its edge. The active set is found by enumeration, smallest
/// first.
pub(crate) fn solve_impulses(m: f32, inertia: f32, s0: [f32; 3], rows: &[Row]) -> Impulses {
    let n_rows = rows.len();
    let normals: Vec<usize> = (0..n_rows).filter(|&i| rows[i].tangent_of.is_none()).collect();
    let pre: Vec<f32> = rows.iter().map(|r| r.vel(s0)).collect();
    let scale = rows.iter().map(|r| r.vel(s0).abs()).fold(1e-3f32, f32::max);
    let mut subsets: Vec<Vec<usize>> = (0u32..(1u32 << normals.len())).map(|mask| normals.iter().enumerate().filter(|(k, _)| mask & (1 << k) != 0).map(|(_, &i)| i).collect()).collect();
    subsets.sort_by_key(|s| s.len());
    for set in &subsets {
        let mut sliding = vec![false; n_rows];
        let mut sigma = vec![0.0f32; n_rows];
        'retry: for _ in 0..=n_rows {
            let active: Vec<usize> = (0..n_rows).filter(|&i| match rows[i].tangent_of { None => set.contains(&i), Some(o) => set.contains(&o) && !sliding[i] }).collect();
            // The direction each row's impulse acts along: a normal row carries
            // its sliding friction with it.
            let effects: Vec<[f32; 3]> = (0..n_rows)
                .map(|i| {
                    let mut d = rows[i].n;
                    if rows[i].tangent_of.is_none() {
                        for t in 0..n_rows {
                            if rows[t].tangent_of == Some(i) && sliding[t] {
                                d = [d[0] + sigma[t] * rows[t].mu * rows[t].n[0], d[1] + sigma[t] * rows[t].mu * rows[t].n[1]];
                            }
                        }
                    }
                    [d[0] / m, d[1] / m, rows[i].c(d) / inertia.max(1e-6)]
                })
                .collect();
            let effect = |j: usize| effects[j];
            let k = active.len();
            let mut a = vec![vec![0.0f32; k]; k];
            let mut b = vec![0.0f32; k];
            for (ii, &i) in active.iter().enumerate() {
                for (jj, &j) in active.iter().enumerate() {
                    let ef = effect(j);
                    a[ii][jj] = rows[i].n[0] * ef[0] + rows[i].n[1] * ef[1] + rows[i].c(rows[i].n) * ef[2] + if i == j { rows[i].inv_m_other } else { 0.0 };
                }
                let target = if rows[i].tangent_of.is_none() { -rows[i].e * pre[i].min(0.0) } else { 0.0 };
                b[ii] = target - pre[i];
            }
            let lam_active = if k == 0 { Some(vec![]) } else { solve_soft(&a, &b) };
            let Some(lam_active) = lam_active else {
                // Stuck friction can ask a point for another speed than a normal row
                // does (a toe hit at the support edge it stands on): the set may still
                // hold with those rows sliding, against the way the closing contacts
                // alone would move their points.
                let stuck: Vec<usize> = active.iter().copied().filter(|&t| rows[t].tangent_of.is_some()).collect();
                if stuck.is_empty() {
                    break 'retry;
                }
                for t in stuck {
                    let push: f32 = active
                        .iter()
                        .filter(|&&j| rows[j].tangent_of.is_none() && pre[j] < 0.0)
                        .map(|&j| {
                            let ef = effect(j);
                            let own = rows[j].n[0] * ef[0] + rows[j].n[1] * ef[1] + rows[j].c(rows[j].n) * ef[2];
                            -pre[j] / own.max(1e-9) * (rows[t].n[0] * ef[0] + rows[t].n[1] * ef[1] + rows[t].c(rows[t].n) * ef[2])
                        })
                        .sum();
                    let slip = pre[t] + push;
                    sliding[t] = true;
                    sigma[t] = if slip.abs() > 1e-6 { -slip.signum() } else { 1.0 };
                }
                continue 'retry;
            };
            let mut lambda = vec![0.0f32; n_rows];
            for (ii, &i) in active.iter().enumerate() {
                lambda[i] = lam_active[ii];
            }
            // Friction within its cone, or slide at its edge: a contact whose
            // sticking would need more than its cone (or a pull on its normal)
            // slides, against the sliding it already has, else against the
            // stick impulse.
            let mut slid = false;
            for &t in &active {
                if let Some(o) = rows[t].tangent_of {
                    if lambda[t].abs() > rows[t].mu * lambda[o].max(0.0) * (1.0 + 1e-4) + 1e-7 {
                        sliding[t] = true;
                        sigma[t] = if pre[t].abs() > 1e-3 { -pre[t].signum() } else { lambda[t].signum() };
                        slid = true;
                    }
                }
            }
            if slid {
                continue 'retry;
            }
            if active.iter().any(|&i| rows[i].tangent_of.is_none() && lambda[i] < -1e-5 * scale) {
                break 'retry;
            }
            let mut s = s0;
            for &j in &active {
                let ef = effect(j);
                s = [s[0] + lambda[j] * ef[0], s[1] + lambda[j] * ef[1], s[2] + lambda[j] * ef[2]];
            }
            // Contacts left out must not close.
            let ok = normals.iter().filter(|i| !set.contains(i)).all(|&i| rows[i].vel(s) >= -1e-4 * scale);
            if !ok {
                break 'retry;
            }
            // A sliding row's impulse shows as the friction share of its normal one.
            for t in 0..n_rows {
                if let (Some(o), true) = (rows[t].tangent_of, sliding[t]) {
                    lambda[t] = sigma[t] * rows[t].mu * lambda[o];
                }
            }
            return Impulses { s, lambda, sliding, found: true };
        }
    }
    Impulses { s: s0, lambda: vec![0.0; n_rows], sliding: vec![false; n_rows], found: false }
}

// ----- the support: ground rows -----

/// How the support holds along the impulse's plane during the impulse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Hold {
    /// Rolling wheels or tracks along their travel, or a moving leg: no tangential impulse.
    Rolling,
    /// Tyres across their travel, a standing foot: friction holds the edge.
    Friction,
}

/// How a base's support holds an impulse in the plane of `dir` (base frame):
/// standing still, every base holds by friction (feet grip, the drive holds
/// its wheels or tracks still); moving, wheels and tracks roll along x and grip
/// across it, and a walking leg gives no tangential impulse.
pub fn hold_of(body: &Body, dir: [f32; 2], moving: bool) -> Hold {
    if !moving {
        return Hold::Friction;
    }
    if body.base == BaseKind::HumanFoot {
        return Hold::Rolling;
    }
    // Mecanum rollers roll either way.
    if body.base == BaseKind::Mecanum || dir[0].abs() >= dir[1].abs() {
        Hold::Rolling
    } else {
        Hold::Friction
    }
}

/// The two ground rows (support edges ahead and behind) and their friction rows.
pub(crate) fn ground_rows(p: &Planar, hold: Hold, mu: f32) -> Vec<Row> {
    let h = p.com_z;
    let mut rows = vec![Row::normal([p.ahead_m, -h], [0.0, 1.0], 0.0), Row::normal([-p.behind_m, -h], [0.0, 1.0], 0.0)];
    if hold == Hold::Friction {
        rows.push(Row::tangent(0, [p.ahead_m, -h], [1.0, 0.0], mu));
        rows.push(Row::tangent(1, [-p.behind_m, -h], [1.0, 0.0], mu));
    }
    rows
}

// ----- turning about a support edge -----

/// How the pivot holds while the body turns about it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) enum PivotMode {
    Stick,
    /// Sliding against the friction (the sign of the pivot's velocity along the surface's tangent).
    Slide(f32),
    /// Off the surface.
    Free,
}

/// A planar rigid body: CoM position and velocity in the plane, tilt and its rate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Rigid2 {
    pub c: [f32; 2],
    pub th: f32,
    pub v: [f32; 2],
    pub w: f32,
}

/// The tangent of a surface of outward normal `n` in the plane: `(n_z, −n_u)` (along `+u` on the ground).
pub(crate) fn tangent_of(n: [f32; 2]) -> [f32; 2] {
    [n[1], -n[0]]
}

/// Accelerations of a body of mass `m`, inertia `i` resting on the body point
/// `r0` (its offset from the CoM at zero tilt) against a surface of outward
/// normal `n` (up: the ground; `(±1, 0)`: a wall or a step's face) under
/// `mode`, with friction `mu` along it: `(a_u, a_z, α, N, F_t)`, `N` along `n`
/// and `F_t` along [`tangent_of`]. The pivot's acceleration is gravity's, the
/// turn's centripetal `−ω²·r`, and the response to the contact force
/// (`f/m` and `α × r`, `α = (r × f)/I`): held at zero (stuck), or along `n`
/// only with `F_t = −σ·μ·N` (sliding).
pub(crate) fn pivot_accel(m: f32, i: f32, st: &Rigid2, r0: [f32; 2], n: [f32; 2], mode: PivotMode, mu: f32) -> (f32, f32, f32, f32, f32) {
    let r = rot2(r0, st.th);
    let w2 = st.w * st.w;
    let t = tangent_of(n);
    // The pivot's acceleration without the contact, and per unit of force along `d`.
    let free = [-w2 * r[0], -G - w2 * r[1]];
    let resp = |d: [f32; 2]| {
        let c = (r[1] * d[0] - r[0] * d[1]) / i;
        [d[0] / m + c * r[1], d[1] / m - c * r[0]]
    };
    let out = |nn: f32, ft: f32| {
        let f = [nn * n[0] + ft * t[0], nn * n[1] + ft * t[1]];
        (f[0] / m, f[1] / m - G, (r[1] * f[0] - r[0] * f[1]) / i, nn, ft)
    };
    let (jn, jt) = (resp(n), resp(t));
    match mode {
        PivotMode::Free => (0.0, -G, 0.0, 0.0, 0.0),
        PivotMode::Stick => {
            let a = vec![vec![jn[0], jt[0]], vec![jn[1], jt[1]]];
            let x = gauss(a, vec![-free[0], -free[1]]).unwrap_or(vec![m * G * n[1].max(0.0), 0.0]);
            out(x[0], x[1])
        }
        PivotMode::Slide(sigma) => {
            let dot = |a: [f32; 2], b: [f32; 2]| a[0] * b[0] + a[1] * b[1];
            let nn = -dot(n, free) / (dot(n, jn) - sigma * mu * dot(n, jt));
            out(nn, -sigma * mu * nn)
        }
    }
}

/// The mode a pivot is in: sliding while it slides (kinetic friction until it
/// stops), stuck while friction holds it at rest, sliding past that, free when
/// the surface would have to pull.
pub(crate) fn pivot_mode(m: f32, i: f32, st: &Rigid2, r0: [f32; 2], n: [f32; 2], mu: f32, prev: PivotMode) -> PivotMode {
    let r = rot2(r0, st.th);
    let t = tangent_of(n);
    let slip_v = (st.v[0] + st.w * r[1]) * t[0] + (st.v[1] - st.w * r[0]) * t[1];
    let slide = |s: f32| {
        let (.., n_s, _) = pivot_accel(m, i, st, r0, n, PivotMode::Slide(s), mu);
        if n_s <= 0.0 {
            PivotMode::Free
        } else {
            PivotMode::Slide(s)
        }
    };
    if let PivotMode::Slide(s) = prev {
        if slip_v * s > 1e-4 {
            return slide(s);
        }
    }
    if slip_v.abs() > 1e-3 {
        return slide(slip_v.signum());
    }
    let (.., nn, ft) = pivot_accel(m, i, st, r0, n, PivotMode::Stick, mu);
    if nn <= 0.0 {
        return PivotMode::Free;
    }
    if ft.abs() <= mu * nn {
        return PivotMode::Stick;
    }
    slide(if slip_v.abs() > 1e-4 { slip_v.signum() } else { -ft.signum() })
}

/// One RK4 step of the body on its pivot (against a surface of outward normal `n`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pivot_step(m: f32, i: f32, st: &Rigid2, r0: [f32; 2], n: [f32; 2], mode: PivotMode, mu: f32, dt: f32) -> Rigid2 {
    let d = |s: &Rigid2| -> [f32; 6] {
        let (au, az, al, ..) = pivot_accel(m, i, s, r0, n, mode, mu);
        [s.v[0], s.v[1], s.w, au, az, al]
    };
    let add = |s: &Rigid2, k: [f32; 6], h: f32| Rigid2 { c: [s.c[0] + h * k[0], s.c[1] + h * k[1]], th: s.th + h * k[2], v: [s.v[0] + h * k[3], s.v[1] + h * k[4]], w: s.w + h * k[5] };
    let k1 = d(st);
    let k2 = d(&add(st, k1, 0.5 * dt));
    let k3 = d(&add(st, k2, 0.5 * dt));
    let k4 = d(&add(st, k3, dt));
    let k: [f32; 6] = std::array::from_fn(|j| (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0);
    let mut out = add(st, k, dt);
    if mode != PivotMode::Free {
        // Hold the pivot on the surface (and, stuck, in place) against the integration's drift.
        let r_then = rot2(r0, st.th);
        let r_now = rot2(r0, out.th);
        let dp = [out.c[0] + r_now[0] - st.c[0] - r_then[0], out.c[1] + r_now[1] - st.c[1] - r_then[1]];
        let mut hold = |e: [f32; 2]| {
            let x = dp[0] * e[0] + dp[1] * e[1];
            out.c[0] -= x * e[0];
            out.c[1] -= x * e[1];
        };
        hold(n);
        if mode == PivotMode::Stick {
            hold(tangent_of(n));
        }
    }
    out
}

// ----- the force-angle measure -----

/// The force-angle tip-over stability measure (Papadopoulos & Rey 1996) of a
/// body with its CoM at `com` on the support rectangle `[x0, x1, y0, y1]`
/// (ground z = 0) under the net force `f_r` on the CoM (gravity, external
/// forces, less the inertial `m·a`) and the net moment `n_r` about the CoM
/// (external moments, less `I·α`): for each edge, the angle θ between the
/// force (the moment added as a force couple) and the normal from the CoM to
/// the edge, positive while the force line passes inside;
/// `β = min θ·‖d‖·‖f‖`, `d` the distance from the edge to the force line.
/// Returns `(β, θ of that edge in degrees, edge: 0 front, 1 rear, 2 left, 3 right)`.
pub fn force_angle(support: [f32; 4], com: [f32; 3], f_r: [f32; 3], n_r: [f32; 3]) -> (f32, f32, usize) {
    let edges: [([f32; 2], f32); 4] = [([1.0, 0.0], support[1]), ([-1.0, 0.0], -support[0]), ([0.0, 1.0], support[3]), ([0.0, -1.0], -support[2])];
    let mut best = (f32::INFINITY, 0.0f32, 0usize);
    for (k, (o, at)) in edges.iter().enumerate() {
        let d_out = at - (com[0] * o[0] + com[1] * o[1]);
        let h = com[2];
        let l = [d_out, -h];
        let ll = (d_out * d_out + h * h).sqrt().max(1e-6);
        let axis = Planar::axis_of(*o);
        let m_out = dot3(n_r, axis);
        let f = [f_r[0] * o[0] + f_r[1] * o[1] + m_out / ll * h / ll, f_r[2] + m_out / ll * d_out / ll];
        let cross = l[0] * f[1] - l[1] * f[0];
        let th = (-cross).atan2(l[0] * f[0] + l[1] * f[1]);
        let fm = (f[0] * f[0] + f[1] * f[1]).sqrt();
        let beta = th * ll * th.sin().abs() * fm;
        if beta < best.0 {
            best = (beta, th.to_degrees(), k);
        }
    }
    best
}

// ----- what the world hands this module -----

/// A body driving or walking into an obstacle.
#[derive(Clone, Debug)]
pub struct BumpInput<'a> {
    pub body: &'a Body,
    /// The pose at the contact (standing, a walking step, holding something).
    pub pose: KinPose,
    /// Speed toward the obstacle, m/s, and the direction of travel in the base frame (degrees, 0 ahead, 90 left).
    pub speed_mps: f32,
    pub heading_deg: f32,
    pub obstacle: Obstacle,
    /// Height of the obstacle's top, metres (a wall: `f32::INFINITY`; a curb: its height).
    pub obstacle_top_m: f32,
    /// Height of the obstacle's underside, metres (0 for a wall or a curb; a shelf's or a table's edge above the floor).
    pub obstacle_bottom_m: f32,
    /// The obstacle's own speed toward the robot (another robot, a person walking in), m/s.
    pub obstacle_speed_mps: f32,
    pub ground: Surface,
    /// The base's step limit: a curb this low the wheel climbs (`BaseSpec::max_step_m`).
    pub max_step_m: f32,
}

impl<'a> BumpInput<'a> {
    /// A body at `speed` straight ahead into a wall of `obstacle`.
    pub fn wall(body: &'a Body, speed: f32, obstacle: Obstacle, ground: Surface) -> BumpInput<'a> {
        BumpInput { body, pose: KinPose::stand(0.0), speed_mps: speed, heading_deg: 0.0, obstacle, obstacle_top_m: f32::INFINITY, obstacle_bottom_m: 0.0, obstacle_speed_mps: 0.0, ground, max_step_m: body.base.spec().max_step_m }
    }

    /// A body at `speed` straight ahead into a concrete curb `height_m` high.
    pub fn curb(body: &'a Body, speed: f32, height_m: f32, ground: Surface) -> BumpInput<'a> {
        BumpInput { obstacle_top_m: height_m, ..BumpInput::wall(body, speed, Obstacle::Concrete, ground) }
    }
}

/// The first point of the body an obstacle ahead meets.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ContactPoint {
    /// Base frame, metres.
    pub point: [f32; 3],
    /// The link box it is on.
    pub link: &'static str,
    pub segment: Segment,
    pub contactor: Contactor,
    /// Unit direction of the obstacle's push on the robot, base frame.
    pub normal: [f32; 3],
}

/// The robot's frontmost point along `dir` within the obstacle's height band
/// `[bottom_m, top_m]`: box corners and edge midpoints of every link (a box
/// that crosses the band's edge counts at that height). A wheel meets a low
/// curb on its rim (the push along the line to the axle); a bumper face meets
/// it square.
pub fn first_contact(body: &Body, r: &KinReport, dir: [f32; 2], bottom_m: f32, top_m: f32) -> Option<ContactPoint> {
    let n = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-6);
    let u = [dir[0] / n, dir[1] / n];
    let mut best: Option<(f32, [f32; 3], &'static str, Segment)> = None;
    for b in &r.boxes {
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for sx in [-1.0f32, 0.0, 1.0] {
            for sy in [-1.0f32, 0.0, 1.0] {
                for sz in [-1.0f32, 0.0, 1.0] {
                    let mut p = b.center;
                    for (k, s) in [sx, sy, sz].iter().enumerate() {
                        p = add3(p, scale3(b.axes[k], s * b.half[k]));
                    }
                    pts.push(p);
                }
            }
        }
        let low = pts.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min);
        let high = pts.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max);
        if low > top_m || high < bottom_m {
            continue;
        }
        for p in pts {
            let p = [p[0], p[1], p[2].clamp(bottom_m, top_m)];
            let a = p[0] * u[0] + p[1] * u[1];
            if best.map_or(true, |bb| a > bb.0 + 1e-4) {
                best = Some((a, p, b.name, b.segment));
            }
        }
    }
    let (_, p, link, segment) = best?;
    let back = [-u[0], -u[1], 0.0];
    let (contactor, normal, point) = match link {
        "bumper" => (Contactor::Bumper, back, p),
        "wheel" | "track" => {
            // A curb's edge pushes the rim along the line to the axle.
            let h = top_m.min(WHEEL_R_M).max(0.0);
            let reach = (WHEEL_R_M * WHEEL_R_M - (WHEEL_R_M - h) * (WHEEL_R_M - h)).max(0.0).sqrt();
            let axle_u = 0.5 * body.support_len_m;
            let d = [-reach, WHEEL_R_M - h];
            let dl = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-6);
            let nrm = [u[0] * d[0] / dl, u[1] * d[0] / dl, d[1] / dl];
            let pt = [u[0] * (axle_u + reach), u[1] * (axle_u + reach), h];
            (Contactor::Tyre, nrm, pt)
        }
        _ if matches!(segment, Segment::Foot | Segment::Leg) && body.base == BaseKind::HumanFoot => (Contactor::Sole, back, p),
        _ => (Contactor::Shell, back, p),
    };
    Some(ContactPoint { point, link, segment, contactor, normal })
}

// ----- the outcome -----

/// Where a fall starts, for `fall::fall` (`fall::FallInput::from_tip`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TipStart {
    /// Horizontal direction the robot goes over, base frame, degrees (0 ahead, 90 left).
    pub dir_deg: f32,
    /// Tilt past upright about the edge when the fall takes over, degrees, and the rate then, rad/s.
    pub tilt_deg: f32,
    pub omega: f32,
    /// The edge it goes over: from the CoM along `dir_deg` and up (negative:
    /// below), at zero tilt, metres (the support's edge, or a bumper that caught it).
    pub pivot: [f32; 2],
}

/// A joint the hit loads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointHit {
    pub joint: String,
    pub actuator: String,
    /// The moment of the hit's force about the joint, N·m, over the joint's peak.
    pub torque_nm: f32,
    pub peak_nm: f32,
    pub ratio: f32,
    /// Share of the reducer's life the shock uses (Miner, `wear::damage` over the contact time).
    pub miner: f32,
    /// A backdrivable actuator (a quasi-direct drive) gives way past its peak
    /// instead of holding: the joint is knocked out of position, the gears
    /// are not loaded past their rating.
    pub backdrives: bool,
    /// Past the peak on a joint that does not give way: a shock the reducer
    /// (a bus servo's spur train) was not rated for.
    pub overload: bool,
}

/// What a collision did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Collision {
    pub rollout: PhysRollout,
    pub contact: Option<ContactPoint>,
    pub law: Option<PairLaw>,
    /// Closing speed along the contact's normal, m/s.
    pub v_close_mps: f32,
    /// The mass the obstacle feels, kg (`impulse / ((1 + e)·v)`).
    pub m_eff_kg: f32,
    pub impulse_ns: f32,
    pub pulse: Option<Pulse>,
    /// Peak deceleration of the robot's CoM, g, and whether the trunk IMU would read the impact.
    pub a_peak_g: f32,
    pub detected: bool,
    /// When the IMU reading crosses the threshold, from the first touch (with the filter lag), s.
    pub detect_s: Option<f32>,
    /// Pressure that crushed the obstacle's face, Pa (a dented plasterboard).
    pub dent_pa: Option<f32>,
    /// The CoM's horizontal velocity after the impulse (base frame) and the tilt rate.
    pub v_after: [f32; 2],
    pub omega_after: f32,
    /// The lowest ZMP margin and force-angle measure (over the static value) during the pulse.
    pub margin_min: f32,
    pub beta_min: f32,
    /// The largest tilt reached afterwards, degrees, and its turning energy over the barrier.
    pub tilt_max_deg: f32,
    pub energy_ratio: f32,
    pub tip: Option<TipStart>,
    pub iso: Option<PflCheck>,
    pub damage: Vec<JointHit>,
    /// A second hit the first one led to (a stubbed toe, then the upper body into the wall).
    pub second: Option<Box<Collision>>,
}

/// The upper body's band for the second hit after a stubbed toe: everything
/// above the soles, metres (est: a little above the drawn sole and ankle).
pub const UPPER_BAND_M: f32 = 0.07;
/// The robot's share of the effective mass when its support holds the
/// contact rigidly (the obstacle's row fully braced by the ground's), kg:
/// effectively infinite (a numerical stand-in).
pub const ROBOT_SIDE_RIGID_KG: f32 = 1.0e6;

impl Collision {
    /// The contact's force on the body at its peak as `kin` takes it
    /// (`KinLoads::body_force` at the contact point): the ZMP and the joints
    /// of the hit's worst instant.
    pub fn peak_loads(&self) -> KinLoads {
        match (&self.contact, &self.pulse) {
            (Some(c), Some(p)) => KinLoads { body_force: scale3(c.normal, p.peak_n), body_point: c.point, ..KinLoads::default() },
            _ => KinLoads::default(),
        }
    }
}

/// The focus parts of a base: the feet and their joints, a track, or the wheels.
pub(crate) fn base_parts(body: &Body) -> Vec<Part> {
    match body.base {
        BaseKind::HumanFoot => vec![Part::Foot, Part::FootJoints],
        BaseKind::Track => vec![Part::Track],
        _ => vec![Part::Wheel],
    }
}

/// The body's step reach from its stance along `dir` (base frame), metres:
/// two legs swung 0.8 rad apart either way (as `gait::slip_outcome`) ahead or
/// behind, the hip abduction's stance room to the side.
pub fn step_reach_m(body: &Body, dir: [f32; 2]) -> f32 {
    let fore = 2.0 * (kin::THIGH_M + kin::SHANK_M) * 0.8f32.sin();
    let side = kin::Rom::of(body).stance_out_max_m + 0.5 * STANCE_WIDTH_M;
    let n = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-6);
    let (c, s) = ((dir[0] / n).abs(), (dir[1] / n).abs());
    // The reach ellipse.
    1.0 / ((c / fore).powi(2) + (s / side.max(1e-3)).powi(2)).sqrt()
}

/// Time a legged robot needs to put a foot down after a disturbance, s: the
/// reaction delay and one swing of its gait at its top walking speed.
pub fn step_time_s(body: &Body) -> f32 {
    let spec = body.base.spec();
    let speed = body.drive_limits(&spec, 0.0).max_speed_mps;
    let g = super::gait::plan(body, speed, surface::available_mu(body.base, Surface::Firm.props()), 0.0);
    REACTION_DELAY_S + g.step_s * (1.0 - g.ds_frac)
}

/// The joints a force `f` at `point` (base frame) loads, from the link it
/// strikes back toward the pelvis: its moment about each joint's pitch axis
/// (`y`; the hip roll about `x`) over the joint's peak, and the Miner damage of
/// holding it for `dt_s`.
pub fn joint_hits(body: &Body, r: &KinReport, link: &str, point: [f32; 3], f: [f32; 3], dt_s: f32) -> Vec<JointHit> {
    let side = if link.ends_with("_l") { "l" } else { "r" };
    let chain: Vec<String> = match link {
        "head" => vec!["neck".into()],
        "trunk" => vec!["waist_pitch".into()],
        "upper_arm_r" | "upper_arm_l" => vec![format!("shoulder_{side}"), "waist_pitch".into()],
        "forearm_r" | "forearm_l" | "palm_r" | "palm_l" | "finger_r" | "finger_l" => vec![format!("elbow_{side}"), format!("shoulder_{side}"), "waist_pitch".into()],
        "thigh_r" | "thigh_l" => vec![format!("hip_pitch_{side}"), format!("hip_roll_{side}")],
        "shank_r" | "shank_l" => vec![format!("knee_{side}"), format!("hip_pitch_{side}")],
        "foot_r" | "foot_l" => vec![format!("ankle_{side}"), format!("knee_{side}")],
        _ => vec![],
    };
    let mut out = Vec::new();
    for name in chain {
        let Some(j) = r.joint(&name) else { continue };
        if j.structural || j.peak <= 0.0 {
            continue;
        }
        let axis = if name.starts_with("hip_roll") { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let tau = dot3(cross3(sub3(point, j.pos), f), axis).abs();
        let act = actuator_of(body, &name);
        let backdrives = act.is_some_and(|a| a.backdrivable);
        // A joint that gives way holds no more than its peak.
        let held = if backdrives { tau.min(j.peak) } else { tau };
        let miner = act.map_or(0.0, |a| wear::damage(&a, held, dt_s));
        out.push(JointHit { joint: name, actuator: j.actuator.clone(), torque_nm: tau, peak_nm: j.peak, ratio: tau / j.peak.max(1e-3), miner, backdrives, overload: tau > j.peak && !backdrives });
    }
    out
}

/// Events of the joints a hit loads past their peak: a shock the gears take
/// (`joint_shock`), or a backdrivable joint knocked out of position (`joint_yield`).
pub fn shock_events(hits: &[JointHit], t: f32, parts: &[Part]) -> Vec<ActEvent> {
    hits.iter()
        .filter(|h| h.ratio > 1.0)
        .map(|h| {
            if h.overload {
                ActEvent::new(t, "joint_shock", format!("{} 충격 토크 {:.1} N·m > 피크 {:.1} N·m (감속기 손상 위험)", h.joint, h.torque_nm, h.peak_nm), parts, Some(h.ratio), false)
            } else {
                ActEvent::new(t, "joint_yield", format!("{} 충격 토크 {:.1} N·m > 피크 {:.1} N·m: 역구동으로 밀려 자세가 흐트러짐", h.joint, h.torque_nm, h.peak_nm), parts, Some(h.ratio), true)
            }
        })
        .collect()
}

/// The actuator a `kin` joint is driven by.
pub fn actuator_of(body: &Body, joint: &str) -> Option<Actuator> {
    if joint == "neck" {
        Some(body.head.neck)
    } else if joint == "waist_pitch" {
        body.belly.pitch
    } else if joint.starts_with("shoulder") {
        Some(body.arm.shoulder)
    } else if joint.starts_with("elbow") {
        Some(body.arm.elbow)
    } else if joint.starts_with("hip") || joint.starts_with("knee") {
        Some(body.leg.knee)
    } else if joint.starts_with("ankle") {
        Some(body.leg.ankle)
    } else {
        None
    }
}

/// What the robot meets, in Korean: a fixed obstacle lower than the hips is a curb or a step.
fn obstacle_ko(inp: &BumpInput) -> String {
    let low = inp.obstacle.mass_kg().is_none() && inp.obstacle_top_m.is_finite() && inp.obstacle_top_m < HIP_M;
    if low {
        let mat = match inp.obstacle {
            Obstacle::Drywall => "석고보드",
            Obstacle::Wood => "나무",
            Obstacle::Rock => "바위",
            _ => "콘크리트",
        };
        format!("{mat} 턱 {:.0} cm", inp.obstacle_top_m * 100.0)
    } else {
        inp.obstacle.label_ko().to_string()
    }
}

/// The 3-vector of a direction in degrees on the ground plane.
fn dir_of(deg: f32) -> [f32; 2] {
    let a = deg.to_radians();
    [a.cos(), a.sin()]
}

/// A body meets an obstacle: the impulse, the force over time, what the
/// support does after it, and whether the robot goes over.
pub fn bump(inp: &BumpInput) -> Collision {
    let body = inp.body;
    let legged = body.base == BaseKind::HumanFoot;
    let dir = dir_of(inp.heading_deg);
    let solved = kin::solve(body, &inp.pose, &KinLoads::default());
    let mp = mass_props_of(body, &solved);
    let pl = Planar::of(&mp, solved.support, dir);
    let parts = base_parts(body);
    let mut events = Vec::new();
    let mut samples = Vec::new();
    let mut notes = Vec::new();
    let v_close = inp.speed_mps + inp.obstacle_speed_mps;
    let stand = |t: f32, phase: &'static str, loads: KinLoads| -> ActSample {
        let k = kin::solve_fast(body, &inp.pose, &loads);
        ActSample { t, phase, pose: inp.pose, loads, margin: k.margin, zmp: k.zmp, com: k.com, peak_ratio: k.worst().map_or(0.0, |j| j.ratio), grip: None, object: None }
    };
    samples.push(stand(0.0, "approach", KinLoads::default()));
    let Some(cp) = first_contact(body, &solved, dir, inp.obstacle_bottom_m, inp.obstacle_top_m) else {
        let out = Outcome::ok();
        let rollout = PhysRollout { kind: PhysKind::Collide, samples, events, outcome: out, duration_s: 0.0, metrics: vec![], notes: vec!["장애물보다 낮은 곳에 몸이 없음".into()] };
        return Collision { rollout, contact: None, law: None, v_close_mps: v_close, m_eff_kg: 0.0, impulse_ns: 0.0, pulse: None, a_peak_g: 0.0, detected: false, detect_s: None, dent_pa: None, v_after: [0.0; 2], omega_after: 0.0, margin_min: solved.margin, beta_min: 1.0, tilt_max_deg: 0.0, energy_ratio: 0.0, tip: None, iso: None, damage: vec![], second: None };
    };
    let pairlaw = pair(cp.contactor, &inp.obstacle);
    let what_ko = obstacle_ko(inp);
    let hold = hold_of(body, dir, inp.speed_mps > 0.05);
    let mu = surface::available_mu(body.base, inp.ground.props());
    // The body's state along the plane before the touch, and the obstacle as a row.
    let s0 = [inp.speed_mps, 0.0, 0.0];
    let rc = pl.to_plane(cp.point);
    let nrm = [cp.normal[0] * pl.dir[0] + cp.normal[1] * pl.dir[1], cp.normal[2]];
    let nl = (nrm[0] * nrm[0] + nrm[1] * nrm[1]).sqrt().max(1e-6);
    let nrm = [nrm[0] / nl, nrm[1] / nl];
    let mut obs = Row::normal(rc, nrm, pairlaw.e);
    if let Some(m_o) = inp.obstacle.mass_kg() {
        obs.inv_m_other = 1.0 / m_o.max(1e-3);
        // The other body comes on along the direction it pushes the robot.
        obs.v_other = inp.obstacle_speed_mps;
    }
    let rows_with = |e: f32| {
        let mut rows = vec![Row { e, ..obs }];
        for mut g in ground_rows(&pl, hold, mu) {
            if let Some(o) = g.tangent_of.as_mut() {
                *o += 1;
            }
            rows.push(g);
        }
        rows
    };
    let pre = obs.vel(s0);
    let closing = (-pre).max(0.0);
    // A walking robot's toe caught on a curb (a trip) stops the swing leg, not
    // the body: the leg's foot and shank take the hit (the heel-strike
    // effective mass of `gait`), and the body loses only the impulse the leg
    // passes on. A standing robot's planted foot passes the hit to the body.
    let trip = legged && cp.contactor == Contactor::Sole && inp.speed_mps > 0.05;
    // `m_eff` the mass the obstacle feels (with a body for an obstacle, the
    // pair's reduced mass), `m_robot` the robot's own share of it.
    let (m_eff, m_robot, pulse_r, impulse, s_after) = if trip {
        // The leg against the obstacle: with a body for an obstacle (a person's
        // region, another robot) the pair's reduced mass; the leg is the robot's share.
        let m_leg = super::gait::impact_mass_kg(body);
        let m = 1.0 / (1.0 / m_leg.max(1e-3) + obs.inv_m_other);
        let p = pulse(&pairlaw.law, pairlaw.e, m, closing);
        let j = m * (1.0 + p.e) * closing;
        (m, m_leg, p, j, [inp.speed_mps - j / pl.mass_kg.max(1e-3), 0.0, 0.0])
    } else {
        let (first, alone) = solve_or_alone(pl.mass_kg, pl.inertia, s0, &rows_with(pairlaw.e));
        if alone {
            notes.push("지지 접촉과 함께 풀리지 않아 장애물 접촉만으로 충격량을 계산".into());
        }
        let m = if closing > 1e-6 { first.lambda[0] / ((1.0 + pairlaw.e) * closing) } else { 0.0 };
        // Over the same contacts the inverse masses add (`1/m = 1/m_R + 1/m_o`): the robot's share.
        let inv_r = 1.0 / m.max(1e-6) - obs.inv_m_other;
        let m_r = if obs.inv_m_other <= 0.0 { m } else if inv_r > 1e-6 { 1.0 / inv_r } else { ROBOT_SIDE_RIGID_KG };
        let p = pulse(&pairlaw.law, pairlaw.e, m.max(1e-3), closing);
        // The kinematics with the restitution the contact law actually gives.
        let (imp, _) = solve_or_alone(pl.mass_kg, pl.inertia, s0, &rows_with(p.e));
        (m, m_r, p, imp.lambda[0], imp.s)
    };
    // Low curbs a wheel climbs: a bump on the way up, the drive goes on.
    let climbable = cp.contactor == Contactor::Tyre && inp.obstacle_top_m <= inp.max_step_m + 1e-6;
    let a_peak = pulse_r.peak_n / pl.mass_kg.max(1e-3) / G;
    let detected = a_peak >= IMPACT_DETECT_G;
    let tau_f = 1.0 / (std::f32::consts::TAU * ACCEL_LPF_HZ);
    let detect_s = pulse_r.series.iter().find(|(_, f)| f / pl.mass_kg.max(1e-3) / G >= IMPACT_DETECT_G).map(|(t, _)| t + tau_f);
    let force3 = scale3(cp.normal, 1.0);
    // The pulse through kin: the ZMP and the force-angle measure as the force rises and falls.
    let mut margin_min = solved.margin;
    let beta0 = force_angle(solved.support, mp.com, [0.0, 0.0, -mp.mass_kg * G], [0.0; 3]).0.max(1e-6);
    let mut beta_min = 1.0f32;
    let step = (pulse_r.series.len() / 12).max(1);
    for (t, f) in pulse_r.series.iter().step_by(step).chain(std::iter::once(&(pulse_r.t_peak_s, pulse_r.peak_n))) {
        let fv = scale3(force3, *f);
        // Rolling: the base gives no tangential force, the body decelerates as a whole.
        let acc = if hold == Hold::Rolling { [fv[0] / mp.mass_kg, fv[1] / mp.mass_kg, 0.0] } else { [0.0; 3] };
        let loads = KinLoads { body_force: fv, body_point: cp.point, accel: acc, ..KinLoads::default() };
        let s = stand(*t, "impact", loads);
        margin_min = margin_min.min(s.margin);
        let f_r = add3([0.0, 0.0, -mp.mass_kg * G], sub3(fv, scale3(acc, mp.mass_kg)));
        let n_r = cross3(sub3(cp.point, mp.com), fv);
        beta_min = beta_min.min(force_angle(solved.support, mp.com, f_r, n_r).0 / beta0);
        samples.push(s);
    }
    samples.sort_by(|a, b| a.t.total_cmp(&b.t));
    // Plasterboard dents under a hard corner pressing past its strength: the
    // mean Hertz pressure over the contact circle (`a² = R·δ`, `δ = (F/K)^(2/3)`
    // the face's own share of the deflection).
    let dent_pa = inp.obstacle.crush_pa().and_then(|crush| {
        let k_h = pairlaw.law.hertz_part()?;
        let a2 = SHELL_RADIUS_M * (pulse_r.peak_n / k_h).powf(2.0 / 3.0);
        let p = pulse_r.peak_n / (std::f32::consts::PI * a2).max(1e-9);
        (p > crush).then_some(p)
    });
    let iso = match inp.obstacle {
        Obstacle::Person(region) => Some(pfl(region, m_robot.max(1e-3), closing, pulse_r.contact_s)),
        _ => None,
    };
    let kind = if detected { "collision" } else { "bump" };
    let label = format!(
        "{} {} {:.0} N ({:.0} ms) · 유효 질량 {:.1} kg · 감속 {:.1} g · 반발 {:.2}{}",
        what_ko,
        cp.contactor.label_ko(),
        pulse_r.peak_n,
        pulse_r.contact_s * 1000.0,
        m_eff,
        a_peak,
        pulse_r.e,
        if detected { " · 충돌 감지" } else { " · 감지 문턱 아래" }
    );
    let mut ev_parts = parts.clone();
    if matches!(cp.segment, Segment::Chest | Segment::Belly) {
        ev_parts.push(Part::Waist);
    }
    if cp.segment == Segment::Head {
        ev_parts.push(Part::Neck);
    }
    events.push(ActEvent::new(0.0, kind, label, &ev_parts, Some(pulse_r.peak_n), !detected));
    if let Some(p) = dent_pa {
        events.push(ActEvent::new(pulse_r.t_peak_s, "dent", format!("석고보드 면이 눌려 파임: 접촉 압력 {:.1} MPa > {:.2} MPa", p * 1e-6, DRYWALL_CRUSH_PA * 1e-6), &ev_parts, Some(p), true));
    }
    if let Some(c) = &iso {
        let lim = c.f_limit_n.map_or("허용 안 됨".to_string(), |f| format!("{f:.0} N"));
        events.push(ActEvent::new(0.0, "iso_contact", format!("ISO/TS 15066 {} {} 접촉: {:.0} N vs 한계 {lim}", c.region.label_ko(), if c.transient { "과도" } else { "준정적" }, c.f_peak_n), &ev_parts, Some(c.f_peak_n), c.ok));
    }
    let damage = joint_hits(body, &solved, cp.link, cp.point, scale3(force3, pulse_r.peak_n), pulse_r.contact_s);
    events.extend(shock_events(&damage, pulse_r.t_peak_s, &ev_parts));
    let t_end = pulse_r.contact_s;
    let mut tip = None;
    let mut tilt_max = 0.0f32;
    let mut rest_tilt = 0.0f32;
    let mut energy_ratio = 0.0f32;
    let mut second: Option<Box<Collision>> = None;
    let mut outcome = if climbable { Outcome::ok() } else if detected { Outcome::fail("collision", format!("{}에 {:.0} N으로 부딪혀 멈춤", what_ko, pulse_r.peak_n)) } else { Outcome::fail("bump", format!("{}에 닿아 막힘 ({:.0} N, 감지 문턱 아래)", what_ko, pulse_r.peak_n)) };
    // A fixed obstacle (not another body) that stands past the hips: no step lands beyond its face.
    let fixed = inp.obstacle.mass_kg().is_none();
    let wall_high = fixed && inp.obstacle_top_m >= HIP_M;
    if legged {
        // The capture point after the impulse, from the CoM's line over the support.
        let v_u = s_after[0];
        let w0 = (G / pl.com_z.max(0.1)).sqrt();
        let xi0 = v_u / w0;
        let inside = xi0 <= pl.ahead_m && xi0 >= -pl.behind_m;
        if inside {
            events.push(ActEvent::new(t_end, "recover", format!("발목으로 버팀: 포착점 {:+.2} m가 발 안", xi0), &parts, Some(xi0), true));
        } else {
            // The capture point runs away from the support edge until a foot lands on it.
            let edge = if xi0 >= 0.0 { pl.ahead_m } else { -pl.behind_m };
            let ts = step_time_s(body);
            let xi_t = edge + (xi0 - edge) * (w0 * ts).exp();
            let mut need = xi_t.abs();
            let fl = body.leg.foot_len_m;
            // Room for a step that lands short of the obstacle's face.
            let short_of = rc[0] - (1.0 - super::gait::ANKLE_FL) * fl;
            let mut into_wall = false;
            if xi0 > 0.0 && need > short_of && (trip || wall_high) {
                if wall_high {
                    into_wall = true;
                } else {
                    // A low curb: the caught foot lands past it with its heel clear of the face.
                    need = need.max(rc[0] + super::gait::ANKLE_FL * fl);
                }
            }
            let sdir = if xi0 >= 0.0 { dir } else { [-dir[0], -dir[1]] };
            let reach = step_reach_m(body, sdir);
            if into_wall && trip && inp.obstacle_bottom_m < UPPER_BAND_M {
                // The stubbed foot slowed the body little: it goes on and its upper body
                // meets the wall (once: the second hit is above the soles, not a trip again).
                let upper = bump(&BumpInput { speed_mps: v_u.max(0.0), obstacle_bottom_m: inp.obstacle_bottom_m.max(UPPER_BAND_M), ..inp.clone() });
                if let Some(c2) = &upper.contact {
                    let gap = (rc[0] - pl.to_plane(c2.point)[0]).max(0.0);
                    let t2 = t_end + gap / v_u.max(0.05);
                    notes.push(format!("발끝이 먼저 닿고 {:.2} s 뒤 {}가 벽에 닿음 ({:.2} m 더 감)", t2, c2.link, gap));
                    for s in &upper.rollout.samples {
                        samples.push(ActSample { t: s.t + t2, ..s.clone() });
                    }
                    for e in &upper.rollout.events {
                        events.push(ActEvent { t: e.t + t2, ..e.clone() });
                    }
                    if !upper.rollout.outcome.ok {
                        outcome = upper.rollout.outcome.clone();
                    }
                    tip = upper.tip;
                    second = Some(Box::new(upper));
                }
            } else if into_wall {
                events.push(ActEvent::new(t_end + ts, "recover", format!("벽 앞에서 걸음을 못 내딛고 벽에 기댐 (포착점 {:.2} m > 벽까지 {:.2} m)", need, short_of.max(0.0)), &[Part::Foot, Part::Waist], Some(need), true));
            } else if need <= reach {
                events.push(ActEvent::new(t_end + ts, "recover", format!("{:.2} s 뒤 한 걸음 {:.2} m로 잡음 (한계 {:.2} m)", ts, need, reach), &parts, Some(need), true));
                notes.push(format!("포착점 {:+.2} m → 내딛을 때 {:+.2} m", xi0, xi_t * xi0.signum()));
            } else {
                let tdir = if xi0 >= 0.0 { inp.heading_deg } else { inp.heading_deg + 180.0 };
                let d = if xi0 >= 0.0 { pl.ahead_m } else { pl.behind_m };
                let th_star = pl.tip_angle(d);
                // The tilt when the step fails: the CoM rolled over the edge on the LIP.
                let omega = (v_u.abs() - w0 * d).max(0.0) / pl.com_z.max(0.1) + 1e-3;
                tip = Some(TipStart { dir_deg: tdir, tilt_deg: th_star.to_degrees(), omega, pivot: [d, -pl.com_z] });
                events.push(ActEvent::new(t_end + ts, "tip", format!("{}: 필요한 걸음 {:.2} m > 다리 한계 {:.2} m → 넘어짐", if trip { "턱에 발이 걸림" } else { "밀려남" }, need, reach), &[Part::Foot, Part::FootJoints, Part::Waist], Some(need), false));
                outcome = Outcome::fail("tip", format!("{} 뒤 걸음이 포착점({:.2} m)에 못 미쳐 넘어짐", if trip { "발이 턱에 걸린" } else { "부딪힌" }, need));
            }
        }
    } else {
        // Wheels and tracks: the base turns about the edge the impulse leaves loaded.
        let w = s_after[2];
        if w.abs() > 1e-3 && !climbable {
            let ahead = w > 0.0;
            let sgn = if ahead { 1.0f32 } else { -1.0 };
            // The plane of the turn, +u toward the edge it turns about (the top moving that way).
            let tdir = [pl.dir[0] * sgn, pl.dir[1] * sgn];
            let tpl = Planar::of(&mp, solved.support, tdir);
            let probes = super::fall::probes(body, &solved, &tpl);
            let q0 = [tpl.ahead_m, -tpl.com_z];
            let st0 = Rigid2 { c: [-q0[0], -q0[1]], th: 0.0, v: [s_after[0] * sgn, s_after[1]], w: w.abs() };
            let d = tpl.ahead_m;
            let th_star = tpl.tip_angle(d);
            let j0 = tpl.inertia + tpl.mass_kg * if hold == Hold::Friction { d * d + tpl.com_z * tpl.com_z } else { d * d };
            energy_ratio = 0.5 * j0 * w * w / tpl.barrier_j(d).max(1e-6);
            // The obstacle's face, where the top may lean on it (the edge's frame: u from the edge, z from the ground).
            let wall = (ahead && fixed).then_some(([rc[0] - q0[0], inp.obstacle_bottom_m, inp.obstacle_top_m], inp.obstacle));
            let sc = super::fall::Scene { ground: inp.ground, drop_m: 0.0, edge_u: 0.0, wall, mu_edge: if hold == Hold::Friction { mu } else { 0.0 }, base: body.base };
            let rk = super::fall::rock(tpl.mass_kg, tpl.inertia, &probes, &[], &sc, q0, st0, true, 2.0);
            // `over` sees the CoM cross an edge the body turns about on the ground; one
            // that crosses the tip angle in the air (thrown up off the obstacle, or still
            // turning when the time runs out) and does not come back has gone over too.
            let th_back = tpl.tip_angle(tpl.behind_m);
            let crossed = rk.track.iter().find(|(_, s)| (s.th > th_star && s.w > 0.0) || (s.th < -th_back && s.w < 0.0)).copied();
            let leaned = rk.hits.iter().any(|h| h.wall);
            let past = rk.last.th > th_star || rk.last.th < -th_back;
            // Lying (more than half way over), not leaning on the obstacle's face: over a
            // low obstacle the top lands on the ground beyond it.
            let lying = rk.last.th.abs() >= std::f32::consts::FRAC_PI_4;
            // Still past it at the end of the time (not held up by the obstacle), or come to rest lying.
            let flipped = match (rk.end, crossed) {
                (super::fall::RockEnd::Timeout, Some(c)) if past && (lying || !leaned) => Some(c),
                (super::fall::RockEnd::Rest, Some(c)) if past && lying => Some(c),
                _ => None,
            };
            for (t, st) in rk.track.iter().skip(1) {
                tilt_max = tilt_max.max(st.th.to_degrees());
                let shift = st.c[0] + q0[0];
                let pose = KinPose { base_pitch_deg: if dir[0].abs() >= dir[1].abs() { -sgn * dir[0].signum() * st.th.to_degrees() } else { 0.0 }, ..inp.pose };
                let com = [mp.com[0] + tdir[0] * shift, mp.com[1] + tdir[1] * shift, st.c[1]];
                samples.push(ActSample { t: t_end + t, phase: "tilt", pose, loads: KinLoads::default(), margin: 1.0 - st.th / th_star.max(1e-3), zmp: [com[0], com[1]], com, peak_ratio: 0.0, grip: None, object: None });
            }
            for h in rk.hits.iter().filter(|h| h.v_n >= 0.05) {
                let p = &h.at;
                let what = if h.wall { format!("{}가 {}에 기댐", p.link, inp.obstacle.label_ko()) } else if p.link == "support" { "들린 쪽 바퀴가 다시 닿음".to_string() } else { format!("{}가 땅에 닿아 받침", p.link) };
                events.push(ActEvent::new(t_end + h.t, "impact", format!("{what}: {:.2} m/s · {:.0} N ({:.0} ms)", h.v_n, h.pulse.peak_n, h.pulse.contact_s * 1000.0), &[parts[0], Part::Waist], Some(h.pulse.peak_n), true));
            }
            if rk.end == super::fall::RockEnd::Timeout {
                notes.push(if rk.capped { "타격 횟수 한도에서 아직 움직임: 마지막 자세를 믿지 말 것".into() } else { "기울어진 채 시간 끝".into() });
            }
            match (rk.end, flipped) {
                (super::fall::RockEnd::Timeout | super::fall::RockEnd::Rest, Some((t_over, st))) => {
                    let turn_deg = if ahead { inp.heading_deg } else { inp.heading_deg + 180.0 };
                    let fwd = st.th > 0.0;
                    tip = Some(if fwd {
                        TipStart { dir_deg: turn_deg, tilt_deg: st.th.to_degrees(), omega: st.w, pivot: q0 }
                    } else {
                        TipStart { dir_deg: turn_deg + 180.0, tilt_deg: -st.th.to_degrees(), omega: -st.w, pivot: [tpl.behind_m, -tpl.com_z] }
                    });
                    rest_tilt = rk.last.th.to_degrees();
                    events.push(ActEvent::new(t_end + t_over, "tip", format!("{:.0}° 기울어 무게중심이 {} 지지 모서리를 넘음 (공중에서, 회전 에너지/장벽 {:.2})", st.th.to_degrees().abs(), if fwd { "앞" } else { "뒤" }, energy_ratio), &[parts[0], Part::Waist], Some(energy_ratio), false));
                    outcome = Outcome::fail("tip", format!("{}에 부딪혀 {} 쪽으로 넘어감 (회전 에너지/장벽 {:.2})", what_ko, if ahead == fwd { "앞" } else { "뒤" }, energy_ratio));
                }
                (super::fall::RockEnd::Over, _) => {
                    let (t_over, st, piv) = rk.over.expect("over has its state");
                    // Over an edge ahead of the support's middle it goes on the way it turned;
                    // over one behind it (a rebound off the obstacle) the other way.
                    let fwd = piv[0] >= 0.5 * (tpl.ahead_m - tpl.behind_m);
                    let turn_deg = if ahead { inp.heading_deg } else { inp.heading_deg + 180.0 };
                    tip = Some(if fwd {
                        TipStart { dir_deg: turn_deg, tilt_deg: st.th.to_degrees(), omega: st.w, pivot: piv }
                    } else {
                        TipStart { dir_deg: turn_deg + 180.0, tilt_deg: -st.th.to_degrees(), omega: -st.w, pivot: [-piv[0], piv[1]] }
                    });
                    let on = probes.iter().find(|p| (p.q[0] - piv[0]).abs() < 3e-3 && (p.q[1] - piv[1]).abs() < 3e-3).map_or("받친 곳".to_string(), |p| if p.link == "support" { "지지 모서리".into() } else { format!("{} 모서리", p.link) });
                    events.push(ActEvent::new(t_end + t_over, "tip", format!("{:.0}° 기울어 무게중심이 {on}를 넘음 (회전 에너지/장벽 {:.2})", st.th.to_degrees().abs(), energy_ratio), &[parts[0], Part::Waist], Some(energy_ratio), false));
                    outcome = Outcome::fail("tip", format!("{}에 부딪혀 {} 쪽으로 넘어감 (회전 에너지/장벽 {:.2})", what_ko, if ahead == fwd { "앞" } else { "뒤" }, energy_ratio));
                }
                (super::fall::RockEnd::Rest, None) => {
                    let th = rk.last.th;
                    rest_tilt = th.to_degrees();
                    if th.abs() <= 0.5 * th_star {
                        events.push(ActEvent::new(t_end + rk.t, "recover", format!("{:.1}° 들렸다 다시 섬", tilt_max), &parts, Some(tilt_max), true));
                    } else {
                        // Rests off its own support (a bumper or the obstacle holds it): the lifted wheels do not drive.
                        let label = if leaned && th > 0.0 { format!("{what_ko}에 기대 {:.0}° 기운 채 멈춤: 바퀴가 들려 도움이 필요", th.to_degrees()) } else { format!("{:.0}° 기운 채 받쳐 멈춤: 바퀴가 들려 도움이 필요", th.to_degrees().abs()) };
                        events.push(ActEvent::new(t_end + rk.t, "stuck", label.clone(), &[parts[0], Part::Waist], Some(th.to_degrees().abs()), false));
                        outcome = Outcome::fail("stuck", label);
                    }
                }
                (super::fall::RockEnd::Timeout, None) => {}
            }
        } else if climbable {
            events.push(ActEvent::new(t_end, "recover", format!("턱 {:.0} mm를 넘음", inp.obstacle_top_m * 1000.0), &parts, Some(inp.obstacle_top_m), true));
        }
    }
    // The force-angle measure going below zero says the pulse's resultant left
    // the support (an edge unloads); the impulse's turn says the base lifted.
    // The two are the same physics by different routes: a disagreement is noted.
    let beta_lifts = beta_min < 0.0;
    let turned = !legged && !trip && s_after[2].abs() > 1e-3 && !climbable;
    if !legged && beta_lifts != turned {
        notes.push(format!("힘-각 척도(최소 {:.2})와 충격량의 회전({})이 다름", beta_min, if turned { "들림" } else { "안 들림" }));
    }
    samples.sort_by(|a, b| a.t.total_cmp(&b.t));
    events.sort_by(|a, b| a.t.total_cmp(&b.t));
    let duration = samples.last().map_or(t_end, |s| s.t).max(t_end);
    let v_after = [s_after[0] * pl.dir[0], s_after[0] * pl.dir[1]];
    let mut metrics = vec![
        ("peak_n", pulse_r.peak_n),
        ("contact_ms", pulse_r.contact_s * 1000.0),
        ("t_peak_ms", pulse_r.t_peak_s * 1000.0),
        ("restitution", pulse_r.e),
        ("m_eff_kg", m_eff),
        ("impulse_ns", impulse),
        ("a_peak_g", a_peak),
        ("v_close_mps", closing),
        ("margin_min", margin_min),
        ("beta_min", beta_min),
        ("beta_lifts", if beta_lifts { 1.0 } else { 0.0 }),
        ("tilt_max_deg", tilt_max),
        ("energy_ratio", energy_ratio),
        ("rest_tilt_deg", rest_tilt),
        ("v_after_mps", s_after[0]),
        ("omega_after", s_after[2]),
        ("detected", if detected { 1.0 } else { 0.0 }),
        ("tipped", if tip.is_some() { 1.0 } else { 0.0 }),
    ];
    if let Some(d) = detect_s {
        metrics.push(("detect_ms", d * 1000.0));
    }
    if let Some(s2) = &second {
        if let Some(p2) = &s2.pulse {
            metrics.push(("second_peak_n", p2.peak_n));
            metrics.push(("second_v_mps", s2.v_close_mps));
        }
    }
    let rollout = PhysRollout { kind: PhysKind::Collide, samples, events, outcome, duration_s: duration, metrics, notes };
    Collision { rollout, contact: Some(cp), law: Some(pairlaw), v_close_mps: closing, m_eff_kg: m_eff, impulse_ns: impulse, pulse: Some(pulse_r), a_peak_g: a_peak, detected, detect_s, dent_pa, v_after, omega_after: s_after[2], margin_min, beta_min, tilt_max_deg: tilt_max, energy_ratio, tip, iso, damage, second }
}

// ----- objects falling onto the robot -----

/// What falls onto the robot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FallingObject {
    pub material: ItemMaterial,
    pub mass_kg: f32,
    /// Its size (edge or diameter), metres: the radius of its corner on the robot is half of it.
    pub size_m: f32,
    /// Its velocity when it touches, base frame, m/s.
    pub vel: [f32; 3],
}

impl FallingObject {
    /// An object dropped from `h_m` above the point it strikes: `v = √(2·g·h)` straight down.
    pub fn dropped(material: ItemMaterial, mass_kg: f32, size_m: f32, h_m: f32) -> FallingObject {
        FallingObject { material, mass_kg, size_m, vel: [0.0, 0.0, -drop_speed(h_m)] }
    }

    /// Kinetic energy, J.
    pub fn energy_j(&self) -> f32 {
        0.5 * self.mass_kg * dot3(self.vel, self.vel)
    }
}

/// Speed of a fall from `h_m`, m/s.
pub fn drop_speed(h_m: f32) -> f32 {
    (2.0 * G * h_m.max(0.0)).sqrt()
}

/// Young's modulus and Poisson's ratio of a falling object's material, for its
/// corner on the robot's shell (est: wood across the grain, rock; a parcel is
/// corrugated board, far softer than the shell).
fn object_elastic(m: ItemMaterial) -> (f32, f32) {
    match m {
        ItemMaterial::Wood => (WOOD_E_PA, NU_WOOD),
        ItemMaterial::Stone | ItemMaterial::IronOre => (ROCK_E_PA, NU_ROCK),
        ItemMaterial::Coal => (0.2 * ROCK_E_PA, NU_ROCK),
        ItemMaterial::Cardboard => (CARDBOARD_E_PA, NU_WOOD),
    }
}

/// Corrugated board's effective modulus at a box's corner, Pa (est: the
/// corner crushes like a soft foam; no board modulus in the research).
pub const CARDBOARD_E_PA: f32 = 5.0e6;

/// An object falling onto the robot.
#[derive(Clone, Debug)]
pub struct HitInput<'a> {
    pub body: &'a Body,
    pub pose: KinPose,
    pub object: FallingObject,
    /// Where it strikes, base frame; `None`: the top of the head (or of the highest link).
    pub point: Option<[f32; 3]>,
    pub ground: Surface,
}

/// The highest point of the robot (the top of its highest link).
pub fn top_point(r: &KinReport) -> Option<([f32; 3], &'static str, Segment)> {
    let mut best: Option<([f32; 3], &'static str, Segment)> = None;
    for b in &r.boxes {
        let top = b.center[2] + (0..3).map(|i| b.axes[i][2].abs() * b.half[i]).sum::<f32>();
        if best.map_or(true, |x| top > x.0[2]) {
            best = Some(([b.center[0], b.center[1], top], b.name, b.segment));
        }
    }
    best
}

/// The link whose box holds (or is nearest to) a point.
fn link_at(r: &KinReport, p: [f32; 3]) -> Option<(&'static str, Segment)> {
    r.boxes
        .iter()
        .map(|b| {
            let d = sub3(p, b.center);
            let out: f32 = (0..3).map(|k| (dot3(d, b.axes[k]).abs() - b.half[k]).max(0.0).powi(2)).sum();
            (out, b.name, b.segment)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|x| (x.1, x.2))
}

/// An object strikes a link: the impulse (the robot standing on its support,
/// the object a second body), the force over time through the object's corner
/// and the robot's shell in series (Hunt–Crossley, the object's landing
/// restitution on a hard face), the energy flag, the joints between the
/// struck link and the ground, and whether the hit tips the robot over.
pub fn struck_by(inp: &HitInput) -> Collision {
    let body = inp.body;
    let solved = kin::solve(body, &inp.pose, &KinLoads::default());
    let mp = mass_props_of(body, &solved);
    let parts = base_parts(body);
    let (point, link, segment) = match inp.point {
        Some(p) => {
            let (l, s) = link_at(&solved, p).unwrap_or(("trunk", Segment::Chest));
            (p, l, s)
        }
        None => top_point(&solved).unwrap_or((mp.com, "trunk", Segment::Chest)),
    };
    let o = inp.object;
    let horiz = [o.vel[0], o.vel[1]];
    let hn = (horiz[0] * horiz[0] + horiz[1] * horiz[1]).sqrt();
    let dir = if hn > 1e-3 { [horiz[0] / hn, horiz[1] / hn] } else { [1.0, 0.0] };
    let pl = Planar::of(&mp, solved.support, dir);
    // The contact's normal: along the object's velocity (it pushes the robot the way it moves).
    let speed = dot3(o.vel, o.vel).sqrt().max(1e-6);
    let nrm = [hn / speed, o.vel[2] / speed];
    let rc = pl.to_plane(point);
    let (eo, nuo) = object_elastic(o.material);
    // The object's corner on the housing's face (Hertz, the radii in series), in series with the housing denting as a shell.
    let k_h = hertz_k(e_star(SHELL_E_PA, NU_SHELL, eo, nuo), series_r(SHELL_RADIUS_M, 0.5 * o.size_m));
    let law = Law::Series { k_lin: shell_k(SHELL_RADIUS_M), k_hertz: k_h };
    let e = material::restitution(o.material, Surface::Firm);
    let mut row = Row::normal(rc, nrm, e);
    row.inv_m_other = 1.0 / o.mass_kg.max(1e-3);
    row.v_other = speed;
    let mu = surface::available_mu(body.base, inp.ground.props());
    let hold = hold_of(body, dir, false);
    let rows_with = |e: f32| {
        let mut rows = vec![Row { e, ..row }];
        for mut g in ground_rows(&pl, hold, mu) {
            if let Some(t) = g.tangent_of.as_mut() {
                *t += 1;
            }
            rows.push(g);
        }
        rows
    };
    let s0 = [0.0f32; 3];
    let pre = row.vel(s0);
    let closing = (-pre).max(0.0);
    let (first, alone) = solve_or_alone(pl.mass_kg, pl.inertia, s0, &rows_with(e));
    let m_eff = if closing > 1e-6 { first.lambda[0] / ((1.0 + e) * closing) } else { 0.0 };
    let pulse_r = pulse(&law, e, m_eff.max(1e-4), closing);
    let (imp, _) = solve_or_alone(pl.mass_kg, pl.inertia, s0, &rows_with(pulse_r.e));
    let impulse = imp.lambda[0];
    let energy = o.energy_j();
    let force3 = [o.vel[0] / speed, o.vel[1] / speed, o.vel[2] / speed];
    let mut samples = Vec::new();
    let mut margin_min = solved.margin;
    let stand = |t: f32, loads: KinLoads| {
        let k = kin::solve_fast(body, &inp.pose, &loads);
        ActSample { t, phase: "impact", pose: inp.pose, loads, margin: k.margin, zmp: k.zmp, com: k.com, peak_ratio: k.worst().map_or(0.0, |j| j.ratio), grip: None, object: Some(point) }
    };
    let step = (pulse_r.series.len() / 12).max(1);
    for (t, f) in pulse_r.series.iter().step_by(step).chain(std::iter::once(&(pulse_r.t_peak_s, pulse_r.peak_n))) {
        let s = stand(*t, KinLoads { body_force: scale3(force3, *f), body_point: point, ..KinLoads::default() });
        margin_min = margin_min.min(s.margin);
        samples.push(s);
    }
    samples.sort_by(|a, b| a.t.total_cmp(&b.t));
    let a_peak = pulse_r.peak_n / pl.mass_kg.max(1e-3) / G;
    let mut ev_parts = parts.clone();
    ev_parts.push(match segment {
        Segment::Head => Part::Neck,
        Segment::Arm | Segment::Hand => Part::HandJoints,
        _ => Part::Waist,
    });
    let mut events = vec![ActEvent::new(
        0.0,
        "hit",
        format!("{} {:.1} kg가 {}에 {:.1} m/s로 떨어짐 · {:.0} N ({:.1} ms) · {:.0} J{}", o.material.label_ko(), o.mass_kg, link, closing, pulse_r.peak_n, pulse_r.contact_s * 1000.0, energy, if energy >= DROPPED_SERIOUS_J { " (40 J 이상: 중상 위험 에너지)" } else { "" }),
        &ev_parts,
        Some(pulse_r.peak_n),
        energy < DROPPED_SERIOUS_J,
    )];
    if segment == Segment::Head && pulse_r.peak_n > HEAD_FORCE_LIMIT_N {
        events.push(ActEvent::new(pulse_r.t_peak_s, "head_overload", format!("머리에 {:.0} N > 안전모 시험 한계 {:.0} N: 카메라·센서 손상 위험", pulse_r.peak_n, HEAD_FORCE_LIMIT_N), &[Part::Neck], Some(pulse_r.peak_n), false));
    }
    let damage = joint_hits(body, &solved, link, point, scale3(force3, pulse_r.peak_n), pulse_r.contact_s);
    events.extend(shock_events(&damage, pulse_r.t_peak_s, &ev_parts));
    // A hit that turns the robot: over the edge or back.
    let w = imp.s[2];
    let mut tip = None;
    let mut energy_ratio = 0.0;
    let mut outcome = Outcome::ok();
    let heading = dir[1].atan2(dir[0]).to_degrees();
    if body.base == BaseKind::HumanFoot && imp.s[0].abs() > 1e-4 {
        // Legs step: the capture point after the hit must be reached by a step within the legs' reach.
        let w0 = (G / pl.com_z.max(0.1)).sqrt();
        let xi0 = imp.s[0] / w0;
        let (edge, d) = if xi0 >= 0.0 { (pl.ahead_m, pl.ahead_m) } else { (-pl.behind_m, pl.behind_m) };
        if xi0 > pl.ahead_m || xi0 < -pl.behind_m {
            let need = (edge + (xi0 - edge) * (w0 * step_time_s(body)).exp()).abs();
            let sdir = if xi0 >= 0.0 { dir } else { [-dir[0], -dir[1]] };
            let reach = step_reach_m(body, sdir);
            energy_ratio = need / reach.max(1e-3);
            if need > reach {
                let omega = (imp.s[0].abs() - w0 * d).max(0.0) / pl.com_z.max(0.1) + 1e-3;
                tip = Some(TipStart { dir_deg: heading + if xi0 >= 0.0 { 0.0 } else { 180.0 }, tilt_deg: pl.tip_angle(d).to_degrees(), omega, pivot: [d, -pl.com_z] });
                events.push(ActEvent::new(pulse_r.contact_s, "tip", format!("맞아서 밀려 넘어감: 필요한 걸음 {:.2} m > 다리 한계 {:.2} m", need, reach), &ev_parts, Some(energy_ratio), false));
                outcome = Outcome::fail("tip", format!("떨어진 {}에 맞아 넘어감", o.material.label_ko()));
            } else {
                events.push(ActEvent::new(pulse_r.contact_s, "recover", format!("한 걸음 {:.2} m로 버팀", need), &ev_parts, Some(need), true));
            }
        }
    } else if w.abs() > 1e-4 {
        let ahead = w > 0.0;
        let d = if ahead { pl.ahead_m } else { pl.behind_m };
        let j0 = pl.inertia + pl.mass_kg * (d * d + pl.com_z * pl.com_z);
        energy_ratio = 0.5 * j0 * w * w / pl.barrier_j(d).max(1e-6);
        if energy_ratio > 1.0 {
            let omega = (2.0 * (0.5 * j0 * w * w - pl.barrier_j(d)) / j0).max(0.0).sqrt() + 1e-3;
            tip = Some(TipStart { dir_deg: heading + if ahead { 0.0 } else { 180.0 }, tilt_deg: pl.tip_angle(d).to_degrees(), omega, pivot: [d, -pl.com_z] });
            events.push(ActEvent::new(pulse_r.contact_s, "tip", format!("맞아서 넘어감 (회전 에너지/장벽 {:.2})", energy_ratio), &ev_parts, Some(energy_ratio), false));
            outcome = Outcome::fail("tip", format!("떨어진 {}에 맞아 넘어감", o.material.label_ko()));
        }
    }
    if outcome.ok && damage.iter().any(|h| h.overload) {
        outcome = Outcome::fail("hit_damage", format!("떨어진 {}의 충격이 관절 피크를 넘음", o.material.label_ko()));
    }
    let duration = samples.last().map_or(0.0, |s| s.t);
    let metrics = vec![
        ("peak_n", pulse_r.peak_n),
        ("contact_ms", pulse_r.contact_s * 1000.0),
        ("restitution", pulse_r.e),
        ("m_eff_kg", m_eff),
        ("impulse_ns", impulse),
        ("energy_j", energy),
        ("v_close_mps", closing),
        ("a_peak_g", a_peak),
        ("margin_min", margin_min),
        ("energy_ratio", energy_ratio),
        ("serious", if energy >= DROPPED_SERIOUS_J { 1.0 } else { 0.0 }),
        ("tipped", if tip.is_some() { 1.0 } else { 0.0 }),
    ];
    let cp = ContactPoint { point, link, segment, contactor: Contactor::Shell, normal: force3 };
    let notes = if alone { vec!["지지 접촉과 함께 풀리지 않아 맞은 접촉만으로 충격량을 계산".to_string()] } else { vec![] };
    let rollout = PhysRollout { kind: PhysKind::Hit, samples, events, outcome, duration_s: duration, metrics, notes };
    Collision { rollout, contact: Some(cp), law: Some(PairLaw { law, e }), v_close_mps: closing, m_eff_kg: m_eff, impulse_ns: impulse, pulse: Some(pulse_r), a_peak_g: a_peak, detected: a_peak >= IMPACT_DETECT_G, detect_s: None, dent_pa: None, v_after: [imp.s[0] * dir[0], imp.s[0] * dir[1]], omega_after: w, margin_min, beta_min: 1.0, tilt_max_deg: 0.0, energy_ratio, tip, iso: None, damage, second: None }
}

/// Radii in series (two curved faces), metres.
pub fn series_r(a: f32, b: f32) -> f32 {
    1.0 / (1.0 / a.max(1e-4) + 1.0 / b.max(1e-4))
}

// ----- small vector helpers -----

pub(crate) fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(crate) fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub(crate) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub(crate) fn unit3(a: [f32; 3]) -> [f32; 3] {
    let n = dot3(a, a).sqrt().max(1e-9);
    scale3(a, 1.0 / n)
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
    fn the_peak_follows_v_sqrt_km_and_the_restitution_comes_back() {
        // Nearly elastic: the linear law's peak is v·√(k·m) and its time π·√(m/k).
        let p = pulse(&Law::Linear { k: 5.0e4 }, 0.99, 20.0, 1.0);
        let (f, dt) = material::impact(1.0, 20.0, 5.0e4);
        assert!((p.peak_n - f).abs() / f < 0.02 && (p.contact_s - dt).abs() / dt < 0.03, "{} vs {f}, {} vs {dt}", p.peak_n, p.contact_s);
        // Twice the speed, twice the force; four times the mass, twice the force.
        let p2 = pulse(&Law::Linear { k: 5.0e4 }, 0.55, 20.0, 2.0);
        let p4 = pulse(&Law::Linear { k: 5.0e4 }, 0.55, 80.0, 1.0);
        let p1 = pulse(&Law::Linear { k: 5.0e4 }, 0.55, 20.0, 1.0);
        assert!((p2.peak_n / p1.peak_n - 2.0).abs() < 0.05 && (p4.peak_n / p1.peak_n - 2.0).abs() < 0.05);
        // The linear law gives back about the restitution it was given (no tension: a little more).
        assert!((p1.e - 0.55).abs() < 0.1, "{}", p1.e);
        // Hertz: the undamped peak of a sphere, F = K·x_m^1.5 with x_m = (5mv²/4K)^0.4.
        let k = 1.0e9;
        let h = pulse(&Law::Hertz { k }, 0.99, 10.0, 1.0);
        let xm = (5.0 * 10.0 / (4.0 * k)).powf(0.4);
        assert!((h.peak_n - k * xm.powf(1.5)).abs() / (k * xm.powf(1.5)) < 0.03, "{}", h.peak_n);
        let hd = pulse(&Law::Hertz { k }, 0.5, 10.0, 1.0);
        assert!(hd.e < 0.8 && hd.e > 0.3 && hd.peak_n < h.peak_n, "{} {}", hd.e, hd.peak_n);
        // Series: softer than either alone, and its deflections add up.
        let s = Law::Series { k_lin: 2.0e6, k_hertz: 1.0e9 };
        let f = s.force(0.001);
        assert!(f < 2.0e6 * 0.001 && f < 1.0e9 * 0.001f32.powf(1.5));
        assert!((f / 2.0e6 + (f / 1.0e9).powf(2.0 / 3.0) - 0.001).abs() < 1e-8, "{f}");
    }

    #[test]
    fn the_pulse_gives_back_the_restitution_it_is_given() {
        // The force is cut at zero (a contact never pulls), which the closed-form
        // dampings do not allow for; the searched damping gives back the asked
        // restitution on every law, over masses and speeds.
        let laws = [Law::Linear { k: 5.0e4 }, Law::Linear { k: series_k(shell_k(SHELL_RADIUS_M), 1.0e4) }, Law::Hertz { k: 1.0e9 }, Law::Series { k_lin: shell_k(SHELL_RADIUS_M), k_hertz: hertz_k(e_star(SHELL_E_PA, NU_SHELL, concrete_e_pa(), NU_CONCRETE), SHELL_RADIUS_M) }];
        for law in laws {
            for e in [0.02f32, 0.1, 0.25, 0.55, 0.8] {
                for (m, v) in [(2.0f32, 6.0f32), (8.0, 1.5), (40.0, 0.3)] {
                    let p = pulse(&law, e, m, v);
                    assert!((p.e - e).abs() < 0.03, "{law:?} e {e} m {m} v {v}: {}", p.e);
                }
            }
        }
        // The force over time is the one of that restitution: its impulse is (1 + e)·m·v.
        for law in laws {
            for e in [0.1f32, 0.55] {
                let p = pulse(&law, e, 8.0, 1.5);
                let j: f32 = p.series.windows(2).map(|w| 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0)).sum();
                let want = (1.0 + e) * 8.0 * 1.5;
                assert!((j - want).abs() < 0.03 * want, "{law:?} e {e}: {j} vs {want} N·s");
            }
        }
    }

    #[test]
    fn mass_props_match_the_body() {
        for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
            let b = body(Tier::Mid, base);
            let mp = mass_props(&b, &KinPose::stand(0.0));
            assert!((mp.mass_kg - b.mass_kg()).abs() < 1e-3);
            let iy = mp.about([0.0, 1.0, 0.0]);
            // A body about 1.2 m tall: the radius of gyration about the CoM is a fraction of its height.
            let k = (iy / mp.mass_kg).sqrt();
            assert!(k > 0.15 && k < 0.45, "{base:?} k {k}");
        }
    }

    #[test]
    fn a_square_low_impulse_feels_the_whole_mass_and_a_high_one_less() {
        // A box of mass 10, I 1 on two edges, hit low at the CoM's height: no turn, all the mass.
        let pl = Planar { mass_kg: 10.0, inertia: 1.0, com_z: 0.5, ahead_m: 0.2, behind_m: 0.2, dir: [1.0, 0.0], com: [0.0, 0.0, 0.5] };
        let mut rows = vec![Row::normal([0.3, 0.0], [-1.0, 0.0], 0.0)];
        rows.extend(ground_rows(&pl, Hold::Rolling, 0.0).into_iter().map(|mut r| {
            r.tangent_of = r.tangent_of.map(|t| t + 1);
            r
        }));
        let imp = solve_impulses(10.0, 1.0, [1.0, 0.0, 0.0], &rows);
        assert!((imp.lambda[0] - 10.0).abs() < 1e-3 && imp.s[2].abs() < 1e-4, "{imp:?}");
        // Hit below the CoM: the base pitches toward the wall (top forward), the rear lifts.
        rows[0].r = [0.3, -0.4];
        let low = solve_impulses(10.0, 1.0, [1.0, 0.0, 0.0], &rows);
        assert!(low.s[2] > 0.0 && low.lambda[0] < 10.0, "{low:?}");
        assert!(rows[2].vel(low.s) > 0.0, "the rear edge lifts");
    }

    #[test]
    fn a_tall_narrow_base_tips_at_a_lower_bump_speed_than_a_low_wide_one() {
        let wide = body(Tier::Mid, BaseKind::Wheel);
        let mut narrow = wide.clone();
        narrow.support_len_m = 0.18;
        // A shelf edge at chest height: the impulse above the CoM turns the top back over the rear axle.
        let edge = |b: &Body, v: f32| bump(&BumpInput { obstacle_bottom_m: 0.75, ..BumpInput::wall(b, v, Obstacle::Concrete, Surface::Firm) });
        let tip_speed = |b: &Body| (1..=40).map(|i| i as f32 * 0.1).find(|v| edge(b, *v).tip.is_some()).unwrap_or(f32::INFINITY);
        let (vn, vw) = (tip_speed(&narrow), tip_speed(&wide));
        assert!(vn.is_finite() && vn < vw, "narrow tips at {vn} m/s, wide at {vw} m/s");
        let t = edge(&narrow, vn).tip.unwrap();
        assert!((t.dir_deg - 180.0).abs() < 1e-3 && t.pivot[1] < 0.0, "backward over the rear axle: {t:?}");
        // A curb within the step limit is climbed with a bump, never tipped over.
        let low = bump(&BumpInput { max_step_m: 0.03, ..BumpInput::curb(&wide, 1.0, 0.02, Surface::Firm) });
        assert!(low.rollout.outcome.ok && low.tip.is_none(), "{:?}", low.rollout.events);
    }

    #[test]
    fn a_wall_bump_stops_the_robot_and_hard_walls_hit_harder_than_soft_ones() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let slow = bump(&BumpInput::wall(&b, 0.2, Obstacle::Concrete, Surface::Firm));
        let fast = bump(&BumpInput::wall(&b, 1.5, Obstacle::Concrete, Surface::Firm));
        assert_eq!(slow.contact.unwrap().contactor, Contactor::Bumper);
        assert!(fast.pulse.as_ref().unwrap().peak_n > 5.0 * slow.pulse.as_ref().unwrap().peak_n);
        assert!(!slow.detected && slow.rollout.events[0].kind == "bump", "{:?}", slow.rollout.events);
        assert!(fast.detected && fast.rollout.events[0].kind == "collision", "{:?}", fast.rollout.events);
        // The bumper rebounds (the impulse is (1 + e) of the mass the wall feels); hit below the CoM,
        // the base keeps part of its speed as a forward pitch rather than bouncing back whole.
        let e = fast.pulse.as_ref().unwrap().e;
        assert!(e > 0.3 && e < 0.8 && (fast.impulse_ns - (1.0 + e) * fast.m_eff_kg * 1.5).abs() < 0.05 * fast.impulse_ns, "{e} {} {}", fast.impulse_ns, fast.m_eff_kg);
        assert!(fast.v_after[0] < 1.5 && fast.omega_after > 0.0, "{:?} {}", fast.v_after, fast.omega_after);
        assert!(fast.margin_min < slow.margin_min, "the impulse moves the ZMP toward the wall");
        let pk = fast.peak_loads();
        assert!(pk.body_force[0] < 0.0 && (pk.body_force[0] + fast.pulse.as_ref().unwrap().peak_n).abs() < 1e-2);
        // A legged robot walks its chest into a shelf edge of concrete or plasterboard: the softer board gives a lower peak, and a dent.
        let legs = body(Tier::Mid, BaseKind::HumanFoot);
        let shell = |o: Obstacle| bump(&BumpInput { obstacle_bottom_m: 0.7, ..BumpInput::wall(&legs, 0.6, o, Surface::Firm) });
        let (c, d) = (shell(Obstacle::Concrete), shell(Obstacle::Drywall));
        assert_eq!(c.contact.unwrap().contactor, Contactor::Shell, "{:?}", c.contact);
        assert!(d.pulse.as_ref().unwrap().peak_n < c.pulse.as_ref().unwrap().peak_n);
        assert!(d.dent_pa.is_some(), "a hard corner dents plasterboard");
        // The shell's own give: far softer than the Hertz contact of the face alone.
        assert!(shell_k(SHELL_RADIUS_M) > 1e5 && shell_k(SHELL_RADIUS_M) < 2e6, "{}", shell_k(SHELL_RADIUS_M));
    }

    #[test]
    fn walking_into_a_wall_stubs_the_toe_then_the_chest_meets_it() {
        let legs = body(Tier::Mid, BaseKind::HumanFoot);
        let c = bump(&BumpInput::wall(&legs, 0.8, Obstacle::Concrete, Surface::Firm));
        assert_eq!(c.contact.unwrap().contactor, Contactor::Sole, "{:?}", c.contact);
        let s = c.second.as_ref().expect("the upper body goes on into the wall");
        assert!(matches!(s.contact.unwrap().segment, Segment::Chest | Segment::Head | Segment::Arm), "{:?}", s.contact);
        assert!(c.rollout.metric("second_peak_n").unwrap() > c.pulse.as_ref().unwrap().peak_n, "the trunk carries more of the body than the toe");
        let ts: Vec<f32> = c.rollout.events.iter().map(|e| e.t).collect();
        assert!(ts.windows(2).all(|w| w[1] >= w[0]), "{ts:?}");
    }

    #[test]
    fn a_top_heavy_base_rebounds_off_a_wall_and_goes_over_backward() {
        // At full speed the bumper's hit below the CoM pitches the base onto the
        // wall; the head's rebound off it (at the robot's restitution, 0.55)
        // turns the base back. On the mid base's 0.30 m wheelbase it rocks back
        // onto its wheels; on a short one (0.20 m) the rebound carries it past
        // its rear axle.
        let mut b = body(Tier::Mid, BaseKind::Wheel);
        let c = bump(&BumpInput::wall(&b, 1.5, Obstacle::Concrete, Surface::Firm));
        assert!(c.rollout.events.iter().any(|e| e.kind == "impact" && e.label_ko.contains("기댐")), "{:?}", c.rollout.events);
        assert!(c.tip.is_none() && c.rollout.events.iter().any(|e| e.kind == "recover"), "{:?}", c.rollout.events);
        b.support_len_m = 0.20;
        let c = bump(&BumpInput::wall(&b, 1.5, Obstacle::Concrete, Surface::Firm));
        assert!(c.rollout.events.iter().any(|e| e.kind == "impact" && e.label_ko.contains("기댐")), "{:?}", c.rollout.events);
        let t = c.tip.expect("it goes over");
        assert!((t.dir_deg - 180.0).abs() < 1e-3 && t.omega > 0.0 && t.pivot[0] > 0.0, "{t:?}");
        assert_eq!(c.rollout.outcome.code, Some("tip"));
        let f = super::super::fall::fall(&super::super::fall::FallInput::from_tip(&b, KinPose::stand(0.0), &t, super::super::fall::FallCause::Bumped, Surface::Firm));
        assert!(f.fell && f.lying.down() && f.get_up.needs_help, "{:?}", f.rollout.events);
    }

    #[test]
    fn a_standing_foot_passes_a_hit_to_the_body_and_a_swinging_one_trips() {
        // A dummy sliding into a standing legged robot's toe meets the planted
        // foot and through it the body; walking into the same dummy, the swing
        // leg alone takes the hit.
        let legs = body(Tier::Mid, BaseKind::HumanFoot);
        let dummy = Obstacle::Robot { mass_kg: 9.0 };
        let standing = bump(&BumpInput { obstacle_speed_mps: 0.5, ..BumpInput::wall(&legs, 0.0, dummy, Surface::Firm) });
        let walking = bump(&BumpInput::wall(&legs, 0.5, dummy, Surface::Firm));
        assert_eq!(standing.contact.unwrap().contactor, Contactor::Sole);
        let leg = super::super::gait::impact_mass_kg(&legs);
        // The swinging leg against the 9 kg dummy: the pair's reduced mass.
        let pair = 1.0 / (1.0 / leg + 1.0 / 9.0);
        assert!((walking.m_eff_kg - pair).abs() < 1e-3, "{} vs {pair}", walking.m_eff_kg);
        assert!(standing.m_eff_kg > 2.0 * leg, "{} vs {leg}", standing.m_eff_kg);
        // The planted foot braces the body: the support takes part in the impulse
        // (its friction slides at the toe the dummy pushes, rather than the solve
        // giving up and leaving the body unsupported).
        assert!(!standing.rollout.notes.iter().any(|n| n.contains("장애물 접촉만으로")), "{:?}", standing.rollout.notes);
        // A person's hand met by the swinging toe: the ISO check reduces the leg (the robot's share) with the hand.
        let hand = bump(&BumpInput { obstacle: Obstacle::Person(Region::Hand), ..BumpInput::wall(&legs, 0.5, Obstacle::Concrete, Surface::Firm) });
        let iso = hand.iso.expect("a person's region is checked");
        assert!((iso.reduced_mass_kg - hand.m_eff_kg).abs() < 1e-3 * hand.m_eff_kg, "{} vs {}", iso.reduced_mass_kg, hand.m_eff_kg);
    }

    #[test]
    fn iso_limits_classify_a_contact_with_a_person() {
        // The standard's spring model: F = v·√(μ·k); the pelvis at 1 m/s with a 40 kg robot.
        let c = pfl(Region::Pelvis, 40.0, 1.0, 0.1);
        assert!((c.reduced_mass_kg - 20.0).abs() < 1e-3);
        assert!((c.f_peak_n - (20.0f32 * 25_000.0).sqrt()).abs() < 1.0 && c.transient);
        assert!(c.f_limit_n == Some(360.0) && !c.ok);
        assert!((c.v_max_mps.unwrap() - 360.0 / (20.0f32 * 25_000.0).sqrt()).abs() < 1e-3);
        // No transient contact with the head at all.
        assert!(!pfl(Region::Skull, 5.0, 0.05, 0.05).ok);
        // A slow wheeled robot bumping a thigh stays under the AGV limit at walking pace or not: the check runs.
        let b = body(Tier::Mid, BaseKind::Wheel);
        let r = bump(&BumpInput { obstacle_top_m: f32::INFINITY, ..BumpInput::wall(&b, 0.3, Obstacle::Person(Region::LowerLeg), Surface::Firm) });
        assert!(r.iso.is_some() && r.rollout.events.iter().any(|e| e.kind == "iso_contact"));
        // A light region (the hand, 0.6 kg): the impulse already feels the pair's
        // reduced mass, which is the standard's μ; it is not reduced a second time.
        let v = 0.8f32;
        let hand = bump(&BumpInput::wall(&b, v, Obstacle::Person(Region::Hand), Surface::Firm));
        let iso = hand.iso.expect("a person's region is checked");
        let k = Region::Hand.limit().k_n_per_m;
        assert!(hand.m_eff_kg < Region::Hand.limit().m_kg && (iso.reduced_mass_kg - hand.m_eff_kg).abs() < 1e-3 * hand.m_eff_kg, "{} vs {}", iso.reduced_mass_kg, hand.m_eff_kg);
        assert!((iso.f_peak_n - v * (hand.m_eff_kg * k).sqrt()).abs() < 1e-2 * iso.f_peak_n, "{} vs {}", iso.f_peak_n, v * (hand.m_eff_kg * k).sqrt());
        assert!((iso.v_max_mps.unwrap() - 2.0 * 140.0 / (hand.m_eff_kg * k).sqrt()).abs() < 1e-2, "{:?}", iso.v_max_mps);
    }

    #[test]
    fn a_falling_object_hits_the_head_with_its_drop_energy() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let hit = |h: f32, m: f32| struck_by(&HitInput { body: &b, pose: KinPose::stand(0.0), object: FallingObject::dropped(ItemMaterial::Wood, m, 0.15, h), point: None, ground: Surface::Firm });
        let low = hit(0.5, 2.4);
        let high = hit(2.0, 2.4);
        assert!((high.v_close_mps - (2.0 * G * 2.0f32).sqrt()).abs() < 1e-3);
        assert!(high.pulse.as_ref().unwrap().peak_n > low.pulse.as_ref().unwrap().peak_n);
        assert_eq!(high.contact.unwrap().link, "head");
        // A top hit on a standing robot: the support holds, the object's own mass is what stops.
        assert!(high.tip.is_none() && (high.m_eff_kg - 2.4).abs() < 0.3, "{}", high.m_eff_kg);
        // 40 J flags a hit: 2.4 kg from 2 m is 47 J.
        assert!(high.rollout.metric("serious") == Some(1.0) && low.rollout.metric("serious") == Some(0.0));
        assert!(high.damage.iter().any(|h| h.joint == "neck"));
    }

    #[test]
    fn the_force_angle_measure_falls_to_zero_at_the_edge() {
        let sup = [-0.15, 0.15, -0.2, 0.2];
        let com = [0.0, 0.0, 0.5];
        let (b0, th0, _) = force_angle(sup, com, [0.0, 0.0, -100.0], [0.0; 3]);
        assert!(b0 > 0.0 && (th0 - (0.15f32 / 0.5).atan().to_degrees()).abs() < 0.1, "{b0} {th0}");
        // A forward force that tilts the resultant to the front edge.
        let (b1, th1, e1) = force_angle(sup, com, [30.0, 0.0, -100.0], [0.0; 3]);
        assert!(e1 == 0 && th1.abs() < 0.1 && b1.abs() < 1e-2 * b0, "{b1} {th1}");
        let (b2, ..) = force_angle(sup, com, [60.0, 0.0, -100.0], [0.0; 3]);
        assert!(b2 < 0.0);
    }

    #[test]
    fn a_kin_body_force_moves_the_zmp_and_zero_changes_nothing() {
        let b = body(Tier::Mid, BaseKind::Wheel);
        let p = KinPose::stand(0.0);
        let r0 = kin::solve(&b, &p, &KinLoads::default());
        let r1 = kin::solve(&b, &p, &KinLoads { body_point: [0.1, 0.0, 1.0], ..KinLoads::default() });
        assert_eq!(r0, r1);
        let push = kin::solve(&b, &p, &KinLoads { body_force: [-50.0, 0.0, 0.0], body_point: [0.1, 0.0, 1.0], ..KinLoads::default() });
        assert!(push.zmp[0] < r0.zmp[0] - 0.1 && (push.grf[0] - 50.0).abs() < 1e-3, "{:?}", push.zmp);
    }
}
