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
//! * [`surface`] holds the grounds (friction, rolling resistance, walking
//!   cost) the balance, the drive and the routes use, [`wear`] the reducers'
//!   life, and [`effects`] the environment N:N benchmark of the five body
//!   effects (slip, load, fatigue, joint load, posture stability).
//! * [`kin`] solves a whole-body pose (joints, contacts, ZMP), [`act`] plays
//!   the actions out on it as time series (grasp and release, push, walk,
//!   drive, bend) with friction, mass, elasticity and collisions, and
//!   [`actbench`] scores those actions against measurements the engine does
//!   not use.
//!
//! Units: metres, seconds, kilograms, newtons, and radians inside the model
//! (degrees only appear in the UI).

pub mod act;
pub mod actbench;
pub mod balance;
pub mod base;
pub mod body;
pub mod dynamics;
pub mod effects;
pub mod finger;
pub mod hand;
pub mod joint;
pub mod kin;
pub mod math;
pub mod nn;
pub mod parts;
pub mod sim;
pub mod surface;
pub mod wear;

pub use balance::{reference, Body, BalanceReport, Coupling, DriveLimits, FeedForward, JointLoad, Posture, RefCtx, Reference, Source};
pub use parts::{BodyPlan, Segment, Tier};
pub use base::{simulate_drive, BaseKind, BaseSpec, DriveReport, Terrain};
pub use body::{
    battery_capacity_j, compute as compute_body, BalanceSummary, BodyEffect, BodyInputs, BodyLevel, BodyLimits, BodyPart,
    BodyStatus,
};
pub use finger::FingerSpec;
pub use hand::{ConfigError, FingerCount, Hand, NotoConfig, JointOptions};
pub use joint::{Actuation, JointAngles, JointKind, JointSpec};
pub use math::Vec3;
pub use surface::{Surface, SurfaceProps};
pub use sim::{grasp, Contact, ContactSource, GraspOutcome, GraspResult, GraspSetup, GraspType, HandSim, Payload};
