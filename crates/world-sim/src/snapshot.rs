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
    pub count: u32,
    pub amount: f32,
    pub mass_kg: f32,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motion: Option<MotionView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub belief: Option<BeliefView>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionView {
    pub kind: String,
    pub label_ko: String,
    pub reversing: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeliefView {
    pub x: f32,
    pub z: f32,
    pub yaw_deg: f32,
    pub error_m: f32,
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
    pub hp: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropView {
    pub id: String,
    pub item: ItemKind,
    pub label_ko: String,
    pub pos: Pos3,
    pub amount: f32,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<SiteView>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerrainGrid {
    pub cells_w: usize,
    pub cells_d: usize,
    pub cell_m: f32,
    pub heights_cm: Vec<i16>,
    pub curb_cm: Vec<u8>,
    pub ground: Vec<u8>,
    pub kind: TerrainKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub curb_kind: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteView {
    pub houses: Vec<HouseView>,
    pub warehouse: Option<WarehouseView>,
    pub fixtures: Vec<FixtureView>,
    pub slots: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HouseView {
    pub id: u32,
    pub label_ko: String,
    pub rect_m: [f32; 4],
    pub front_deg: f32,
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
    pub treads: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RampView {
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
    pub kind: String,
    pub label_ko: String,
    pub cells: Vec<[f32; 2]>,
    pub yaw_deg: f32,
    pub base_m: f32,
    pub top_m: f32,
    pub levels_m: Vec<f32>,
    pub blocks: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraspInfo {
    pub item: ItemKind,
    pub stable: bool,
    pub margin: f32,
    pub max_payload_kg: f32,
    pub grasp_type: String,
}
