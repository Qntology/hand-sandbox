//! Lightweight deterministic open-world resource-gathering simulator.
//!
//! * [`terrain`], [`kinds`], [`world`]: a block-style heightfield world with
//!   trees, boulders and ore nodes, dropped items, two waypoints and a curb
//!   wall, plus a robot bound to the project's hand and foot models.
//! * [`nav`]: A* routes judged by `noto_core::simulate_drive`.
//! * [`render`]: a software ray-caster that outputs an RGB frame, an object-id
//!   buffer and a depth buffer, so the ground truth of every pixel is known.
//! * [`capture`]: screenshots as WebP (lossy through libwebp when available).
//! * [`explore`]: terrain line of sight, what the camera has covered, and the
//!   next viewpoint to look from when the target is hidden.
//!
//! It is a stand-in for a full voxel game: it reproduces the part that the
//! robot tests need (harvest, pick up, carry, deliver, seen through a camera),
//! not Minecraft itself. See `docs/PLAN.md` for why.

pub mod capture;
pub mod explore;
pub mod kinds;
pub mod nav;
pub mod odometry;
pub mod render;
pub mod rng;
pub mod snapshot;
pub mod terrain;
pub mod world;

pub use capture::{decode_webp, encode_webp, Encoded};
pub use kinds::{Ground, ItemKind, ItemSpec, NodeKind, NodeSpec};
pub use odometry::{OdomCfg, Odometry};
pub use render::{Camera, Frame, ObjectClass, VisibleObject};
pub use snapshot::{GraspInfo, TerrainGrid, WorldSnapshot};
pub use terrain::TerrainKind;
pub use world::{amount_text, Action, ActionError, BalanceEvent, TimedEvent, World, WorldConfig, WorldEvent, TICK_S};
