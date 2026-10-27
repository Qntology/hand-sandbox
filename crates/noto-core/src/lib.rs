//! Noto robot model for logistics robots (hands, feet and bases, body, balance, actions).
//!
//! * [`NotoConfig`] picks the finger count (2-5), the joint options and the
//!   locomotion base ("foot": wheel, mecanum, track or human foot).
//! * [`NotoConfig::build`] turns that choice into a [`Hand`] with geometry.
//! * [`sim`] animates the hand and runs a quasi-static grasp test on a
//!   [`sim::Payload`]. [`base::simulate_drive`] checks a route for the base.
//! * [`body`] turns simulation variables into a body-part status display.
//! * [`parts`] holds researched part presets (head, chest, belly, arm, leg or
//!   chassis) and [`balance`] the whole-body balance and joint effort they give.
//! * [`morph`] names the robot's form layer by layer (lower body, torso,
//!   head mount, manipulator, cargo) and the named forms after real robots.
//! * [`surface`] holds the grounds (friction, rolling resistance, walking
//!   cost) the balance, the drive and the routes use, [`wear`] the reducers'
//!   life, and [`effects`] the environment N:N benchmark of the five body
//!   effects (slip, load, fatigue, joint load, posture stability).
//! * [`kin`] solves a whole-body pose (joints, contacts, ZMP), [`act`] plays
//!   the actions out on it as time series (grasp and release, push, walk,
//!   drive, bend) with friction, mass, elasticity and collisions, and
//!   [`actbench`] scores those actions against measurements the engine does
//!   not use.
//! * [`contact`] holds the soft finger pads and the toe-jointed soles the
//!   grasp and the gait touch the world with, and [`tactbench`] scores the
//!   tactile answers (pads, grip loop, soles) against measurements the
//!   engine does not use.
//! * [`bench`] judges every benchmark area's cells one way, sums each area's
//!   deviation from its references for the cross-area report, and keeps the
//!   registry of engine files and benchmark-only sources the source-split
//!   test checks.
//!
//! Units: metres, seconds, kilograms, newtons, and radians inside the model
//! (degrees only appear in the UI).

pub mod act;
pub mod actbench;
pub mod balance;
pub mod base;
pub mod bench;
pub mod body;
pub mod collbench;
pub mod contact;
pub mod coopbench;
pub mod dynamics;
pub mod effects;
pub mod finger;
pub mod hand;
pub mod joint;
pub mod kin;
pub mod math;
pub mod morph;
pub mod nn;
pub mod parts;
pub mod sim;
pub mod surface;
pub mod tactbench;
pub mod wear;

pub use balance::{reference, Body, BalanceReport, Coupling, DriveLimits, FeedForward, JointLoad, Posture, RefCtx, Reference, Source};
pub use parts::{BodyPlan, Segment, Tier};
pub use base::{simulate_drive, BaseKind, BaseSpec, DriveReport, Terrain};
pub use bench::{BenchCell, BenchScenario, Rule as BenchRule, Summary as BenchSummary};
pub use body::{
    battery_capacity_j, compute as compute_body, BalanceSummary, BodyEffect, BodyInputs, BodyLevel, BodyLimits, BodyPart,
    BodyStatus,
};
pub use finger::FingerSpec;
pub use hand::{ConfigError, FingerCount, Hand, NotoConfig, JointOptions};
pub use joint::{Actuation, JointAngles, JointKind, JointSpec};
pub use math::Vec3;
pub use morph::{Cargo, HeadMount, Lower, Manip, Morphology, Torso};
pub use surface::{Surface, SurfaceProps};
pub use sim::{grasp, Contact, ContactSource, GraspOutcome, GraspResult, GraspSetup, GraspType, HandSim, Payload};
