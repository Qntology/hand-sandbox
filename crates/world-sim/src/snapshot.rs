//! Serializable views of the world for the UI and for logs. Ids are strings
//! with a one-letter prefix (`n12` node, `d3` drop, `w1` waypoint) so they stay
//! unique across kinds.

use serde::{Deserialize, Serialize};

use crate::kinds::{ItemKind, NodeKind};

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
    pub count: u32,
    pub mass_kg: f32,
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
    pub pos: Pos3,
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
}

/// Terrain as flat arrays, sent once per world because it never changes while
/// the world runs. `ground` holds 0 sand, 1 grass, 2 dirt, 3 rock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainGrid {
    pub cells_w: usize,
    pub cells_d: usize,
    pub cell_m: f32,
    pub heights_cm: Vec<i16>,
    pub curb_cm: Vec<u8>,
    pub ground: Vec<u8>,
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
