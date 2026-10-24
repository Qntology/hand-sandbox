//! Robot hand model for logistics robots.
//!
//! * [`HandConfig`] picks the finger count (2-5), the joint options and the
//!   locomotion base ("foot": wheel, mecanum, track or human foot).
//! * [`HandConfig::build`] turns that choice into a [`Hand`] with geometry.
//! * [`sim`] animates the hand and runs a quasi-static grasp test on a
//!   [`sim::Payload`]. [`base::simulate_drive`] checks a route for the base.
//! * [`body`] turns simulation variables into a body-part status display.
//! * [`parts`] holds researched part presets (head, chest, belly, arm, leg or
//!   chassis) and [`balance`] the whole-body balance and joint effort they give.
//!
//! Units: metres, seconds, kilograms, newtons, and radians inside the model
//! (degrees only appear in the UI).

pub mod balance;
pub mod base;
pub mod body;
pub mod dynamics;
pub mod finger;
pub mod hand;
pub mod joint;
pub mod math;
pub mod nn;
pub mod parts;
pub mod sim;

pub use balance::{reference, Body, BalanceReport, Coupling, DriveLimits, FeedForward, JointLoad, Posture, RefCtx, Reference, Source};
pub use parts::{BodyPlan, Segment, Tier};
pub use base::{simulate_drive, BaseKind, BaseSpec, DriveReport, Terrain};
pub use body::{
    battery_capacity_j, compute as compute_body, BalanceSummary, BodyEffect, BodyInputs, BodyLevel, BodyLimits, BodyPart,
    BodyStatus, JOINT_ENDURANCE_STRIKES,
};
pub use finger::FingerSpec;
pub use hand::{ConfigError, FingerCount, Hand, HandConfig, JointOptions};
pub use joint::{Actuation, JointAngles, JointKind, JointSpec};
pub use math::Vec3;
pub use sim::{grasp, Contact, ContactSource, GraspOutcome, GraspResult, GraspSetup, GraspType, HandSim, Payload};
