//! Lightweight deterministic open-world resource-gathering simulator.
//!
//! * [`terrain`], [`kinds`], [`world`]: a block-style heightfield world with
//!   trees, boulders and ore nodes, dropped items, two waypoints and a curb
//!   wall, plus a robot bound to the project's hand and foot models.
//! * [`nav`]: A* routes judged by `noto_core::simulate_drive`.
//! * [`render`]: a software ray-caster that outputs an RGB frame, an object-id
//!   buffer and a depth buffer, so the ground truth of every pixel is known.
//! * [`light`]: the lighting of the scene (v1.0): the flat pre-v1.0 shading, or
//!   a sun placed by the NOAA solar position series that lights faces, casts
//!   shadows and sets the camera's exposure (IESNA daylight model).
//! * [`iso`]: the isometric third-person view (orthographic rays from a
//!   world-fixed diagonal, cutaway, tilted robot links).
//! * [`capture`]: screenshots as WebP (lossy through libwebp when available).
//! * [`explore`]: terrain line of sight, what the camera has covered, and the
//!   next viewpoint to look from when the target is hidden.
//! * [`site`], [`fixture`]: houses with porch stairs and ramps, a logistics
//!   centre with shelves and packing tables, and the parcels' places on them
//!   (`WorldConfig::site`, off by default).
//!
//! It is a stand-in for a full voxel game: it reproduces the part that the
//! robot tests need (harvest, pick up, carry, deliver, seen through a camera),
//! not Minecraft itself. See `docs/PLAN.md` for why.

pub mod capture;
pub mod explore;
pub mod fixture;
pub mod iso;
pub mod kinds;
pub mod light;
pub mod nav;
pub mod odometry;
pub mod render;
pub mod rng;
pub mod site;
pub mod snapshot;
pub mod terrain;
pub mod world;

pub use capture::{decode_webp, encode_webp, Encoded};
pub use kinds::{Ground, ItemKind, ItemSpec, NodeKind, NodeSpec};
pub use light::Lighting;
pub use odometry::{OdomCfg, Odometry};
pub use fixture::{Fixture, FixtureKind};
pub use render::{Camera, Frame, ObjectClass, VisibleObject};
pub use site::{Site, SiteCfg, Support};
pub use snapshot::{GraspInfo, SiteView, TerrainGrid, WorldSnapshot};
pub use terrain::{CurbKind, TerrainKind};
pub use world::{amount_text, Action, ActionError, BalanceEvent, TimedEvent, World, WorldConfig, WorldEvent, TICK_S};
