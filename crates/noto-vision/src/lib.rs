//! ML logic on keyframes. See `docs/PLAN-v0.3.md`.
//!
//! The game logic (world, quest driver, physics) records a play-through; this
//! crate only ever sees what a real robot would have: a keyframe image, the
//! camera intrinsics and the robot's odometry pose. Its tracks, in pipeline
//! order:
//!
//! 1. [`depth`]: monocular depth, TIPSv2-B/14 DPT ([`tips`]).
//! 2. [`locate`]: SigLIP2 dense patch heat maps, peak = semantic position.
//! 3. [`track`]: masks from the semantic points as prompts, kept across
//!    keyframes (SAM interface: [`sam`] runs SAM ViT-B or MobileSAM on the
//!    selected device, the region-growing stand-in runs without weights).
//! 4. [`triangulate`]: range from two bearings over an odometry baseline, the
//!    mandatory step before trusting a distance.
//! 5. [`decide`]: the time-series decision (forecast the target, choose the
//!    move, predict what the next keyframe should show).
//!
//! [`fusion`] mixes the tracks into per-keyframe features for the SDS.

pub mod camera;
pub mod decide;
pub mod depth;
pub mod frame;
pub mod fusion;
pub mod locate;
pub mod sam;
mod sam_decoder;
pub mod tips;
pub mod track;
pub mod triangulate;

pub use camera::{Intrinsics, Pose};
pub use decide::{centroid_sigma, scale_usable, DecideCfg, Decider, Decision, DecisionReport, Estimate, Prediction, Sighting};
pub use fusion::{features, flags, FlagCfg, Features, FusionInput};
pub use depth::{depth_metrics, DepthMetrics, DepthModel};
pub use frame::{DepthKind, DepthMap, Mask, Rgb};
pub use locate::{Heat, LabelPrompt, SemanticLocator, SemanticPoint, SiglipLocator};
pub use tips::{InputGrid, TipsConfig, TipsDepth};
pub use sam::{SamCfg, SamModel, SamTracker, SamVariant};
pub use track::{MaskTracker, PointPrompt, RegionCfg, RegionTracker, TrackedMask};
pub use triangulate::{plan_step, triangulate, Bearing, Fix, TriCfg, TriError, TriStep};
