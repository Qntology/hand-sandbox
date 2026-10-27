//! Serializable views of the world for the UI and for logs. Ids are strings
//! with a one-letter prefix (`n12` node, `d3` drop, `w1` waypoint) so they stay
//! unique across kinds.

use serde::{Deserialize, Serialize};

use crate::kinds::{ItemKind, NodeKind};
use crate::site::Support;
use crate::terrain::TerrainKind;

pub fn node_id(id: u32) -> String {
    format!("n{id}")
}
pub fn drop_id(id: u32) -> String {
    format!("d{id}")
}
pub fn waypoint_id(id: u32) -> String {
    format!("w{id}")
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pos3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bounds {
    pub width_m: f32,
    pub depth_m: f32,
    pub cells_w: usize,
    pub cells_d: usize,
    pub cell_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarryView {
    pub item: ItemKind,
    pub label_ko: String,
    /// Whole items.
    pub count: u32,
    /// Items, broken pieces included (`0.5` is half a stone).
    pub amount: f32,
    /// Mass of everything in this entry.
    pub mass_kg: f32,
    /// Mass of one whole item.
    pub unit_kg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentView {
    pub pos: Pos3,
    pub yaw_deg: f32,
    pub speed_mps: f32,
    pub base: String,
    pub fingers: u8,
    pub carrying: Vec<CarryView>,
    pub carried_kg: f32,
    pub capacity_kg: f32,
    pub battery_ratio: f32,
    pub action: Option<String>,
    /// Where the robot believes it is (its odometry), and how far that is off
    /// the truth (the view shows it; the robot does not know the error).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub belief: Option<BeliefView>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeliefView {
    pub x: f32,
    pub z: f32,
    pub yaw_deg: f32,
    pub error_m: f32,
    /// Distance travelled since the robot last fixed its position.
    pub since_fix_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaypointView {
    pub id: String,
    pub key: String,
    pub label_ko: String,
    pub pos: Pos3,
    pub stock: Vec<CarryView>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeView {
    pub id: String,
    pub kind: NodeKind,
    pub label_ko: String,
    pub pos: Pos3,
    /// Remaining work over total work, 0..1.
    pub hp: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropView {
    pub id: String,
    pub item: ItemKind,
    pub label_ko: String,
    /// `y` is the top of what it rests on: the ground, a porch, a shelf deck, a table.
    pub pos: Pos3,
    /// How much of one item the piece is.
    pub amount: f32,
    /// What it rests on when that is not the ground (`site::Support`).
    #[serde(default, skip_serializing_if = "Support::is_ground")]
    pub support: Support,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub time_s: f64,
    pub seed: u64,
    pub bounds: Bounds,
    pub agent: AgentView,
    pub waypoints: Vec<WaypointView>,
    pub nodes: Vec<NodeView>,
    pub drops: Vec<DropView>,
    pub terrain: TerrainKind,
    /// Houses, the logistics centre and their fixtures (`crate::site`), when the world has a site.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<SiteView>,
}

/// Terrain as flat arrays, sent once per world because it never changes while
/// the world runs. `ground` holds 0 sand, 1 grass, 2 dirt, 3 rock, 4 water, 5 wet sand,
/// 6 gravel, 7 a site's concrete floor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainGrid {
    pub cells_w: usize,
    pub cells_d: usize,
    pub cell_m: f32,
    pub heights_cm: Vec<i16>,
    pub curb_cm: Vec<u8>,
    pub ground: Vec<u8>,
    pub kind: TerrainKind,
    /// `terrain::CurbKind` per cell (0 kerb, 1 stair tread, 2 raised floor, 3 ramp); empty without a site.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curb_kind: Vec<u8>,
}

/// A site for the UI (metres, world frame: x, z; heights are world y).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteView {
    pub houses: Vec<HouseView>,
    pub warehouse: Option<WarehouseView>,
    /// Walls, doors, landings, stairs, ramps, shelves and tables (not the floors and roofs).
    pub fixtures: Vec<FixtureView>,
    /// Places where parcels can lie.
    pub slots: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HouseView {
    pub id: u32,
    pub label_ko: String,
    /// The body, walls included: x0, z0, x1, z1.
    pub rect_m: [f32; 4],
    /// Heading of the front (inside to outside), degrees: 0 = +x, 90 = +z.
    pub front_deg: f32,
    /// Floor and porch height above the ground.
    pub floor_m: f32,
    pub door: [f32; 2],
    pub porch_m: [f32; 4],
    pub stairs: StairView,
    pub ramp: Option<RampView>,
    pub reachable_legs: bool,
    pub reachable_wheels: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StairView {
    pub risers: u8,
    pub riser_m: f32,
    pub tread_m: f32,
    pub width_m: f32,
    /// Tread cells: x, z, top.
    pub treads: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RampView {
    /// Ramp cells from the porch down: x, z, top.
    pub cells: Vec<[f32; 3]>,
    pub rise_m: f32,
    pub run_m: f32,
    pub run_per_rise: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WarehouseView {
    pub label_ko: String,
    pub rect_m: [f32; 4],
    pub door: [f32; 2],
    pub door_width_m: f32,
    pub door_front_deg: f32,
    pub aisle_m: f32,
    pub shelves: usize,
    pub tables: usize,
    /// Task zones with their maintained illuminance (EN 12464-1).
    pub zones: Vec<ZoneView>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoneView {
    pub kind: String,
    pub label_ko: String,
    pub rect_m: [f32; 4],
    pub em_lx: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixtureView {
    pub id: u32,
    /// `wall`, `door`, `landing`, `step`, `ramp`, `shelf`, `table`.
    pub kind: String,
    pub label_ko: String,
    /// Cell centres: x, z.
    pub cells: Vec<[f32; 2]>,
    pub yaw_deg: f32,
    pub base_m: f32,
    pub top_m: f32,
    /// Heights items rest at (shelf decks, a table top), world y.
    pub levels_m: Vec<f32>,
    /// Whether it blocks the cells it stands on.
    pub blocks: bool,
}

/// What one grasp test says about holding an item with the current hand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspInfo {
    pub item: ItemKind,
    pub stable: bool,
    /// Friction capacity over required force.
    pub margin: f32,
    pub max_payload_kg: f32,
    pub grasp_type: String,
}
