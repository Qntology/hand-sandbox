//! Robot hand model for logistics robots.
//!
//! * [`HandConfig`] picks the finger count (2-5), the joint options and the
//!   locomotion base ("foot": wheel, mecanum, track or human foot).
//! * [`HandConfig::build`] turns that choice into a [`Hand`] with geometry.
//! * [`sim`] animates the hand and runs a quasi-static grasp test on a
//!   [`sim::Payload`]. [`base::simulate_drive`] checks a route for the base.
//! * [`body`] turns simulation variables into a body-part status display.
//!
//! Units: metres, seconds, kilograms, newtons, and radians inside the model
//! (degrees only appear in the UI).

pub mod base;
pub mod body;
pub mod finger;
pub mod hand;
pub mod joint;
pub mod math;
pub mod sim;

pub use base::{simulate_drive, BaseKind, BaseSpec, DriveReport, Terrain};
pub use body::{
    battery_capacity_j, compute as compute_body, BodyEffect, BodyInputs, BodyLevel, BodyPart,
    BodyStatus, JOINT_ENDURANCE_STRIKES,
};
pub use finger::FingerSpec;
pub use hand::{ConfigError, FingerCount, Hand, HandConfig, JointOptions};
pub use joint::{Actuation, JointAngles, JointKind, JointSpec};
pub use math::Vec3;
pub use sim::{grasp, Contact, ContactSource, GraspOutcome, GraspResult, GraspSetup, GraspType, HandSim, Payload};
