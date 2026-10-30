use noto_core::{BodyPlan, FingerCount, NotoConfig};
use serde::{Deserialize, Serialize};
use world_sim::Camera;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseCfg {
    pub seed: u64,
    pub score_sd: f32,
    pub confusion_rate: f32,
    pub false_positive_rate: f32,
    pub miss_rate: f32,
}

impl NoiseCfg {
    pub fn none() -> NoiseCfg {
        NoiseCfg { seed: 1, score_sd: 0.0, confusion_rate: 0.0, false_positive_rate: 0.0, miss_rate: 0.0 }
    }

    pub fn demo() -> NoiseCfg {
        NoiseCfg { seed: 99, score_sd: 0.05, confusion_rate: 0.12, false_positive_rate: 0.30, miss_rate: 0.05 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateCfg {
    pub enabled: bool,
    pub min_score: f32,
    pub min_margin: f32,
    pub confirm_sightings: u32,
    pub track_radius_m: f32,
    pub temporal_min_history: usize,
    pub temporal_widen_sd: f64,
    pub adaptive: bool,
    pub adaptive_k: f64,
}

impl Default for GateCfg {
    fn default() -> Self {
        GateCfg {
            enabled: true,
            min_score: 0.30,
            min_margin: 0.04,
            confirm_sightings: 2,
            track_radius_m: 1.0,
            temporal_min_history: 6,
            temporal_widen_sd: 1.0,
            adaptive: true,
            adaptive_k: 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraCfg {
    pub width: usize,
    pub height: usize,
    pub fov_deg: f32,
    pub pitch_deg: f32,
    pub max_dist_m: f32,
    /// Cameras per eye (the head, and a lift truck's carriage camera): 1 is a monocular RGB camera whose depth comes
    /// from TIPSv2, 2 a rectified stereo pair whose depth comes from SGBM (TIPSv2 is then neither loaded nor run).
    #[serde(default = "one_camera")]
    pub cameras: u8,
}

fn one_camera() -> u8 {
    1
}

impl Default for CameraCfg {
    fn default() -> Self {
        let c = Camera::default();
        CameraCfg { width: c.width, height: c.height, fov_deg: c.fov_deg, pitch_deg: c.pitch_deg, max_dist_m: c.max_dist_m, cameras: one_camera() }
    }
}

impl CameraCfg {
    pub fn camera(&self) -> Camera {
        Camera { width: self.width, height: self.height, fov_deg: self.fov_deg, pitch_deg: self.pitch_deg, max_dist_m: self.max_dist_m, pan_deg: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub seed: u64,
    pub cells: usize,
    #[serde(default = "default_noto")]
    #[serde(alias = "hand")]
    pub noto: NotoConfig,
    #[serde(default)]
    pub body: BodyPlan,
    #[serde(default)]
    pub terrain: world_sim::TerrainKind,
    pub camera: CameraCfg,
    pub gates: GateCfg,
    pub noise: NoiseCfg,
    pub scan_every_ticks: u32,
    pub approach_scan_every_ticks: u32,
    pub rotate_deg: f32,
    pub locate_timeout_s: f64,
    pub max_relocations: u32,
    pub max_rounds: u32,
    #[serde(default = "default_max_trips")]
    pub max_trips: u32,
    pub approach_reach_m: f32,
    pub min_pixels: u32,
    #[serde(default)]
    pub vision: crate::vision::VisionCfg,
    #[serde(default)]
    pub odometry: world_sim::OdomCfg,
    #[serde(default)]
    pub parcels: Option<usize>,
    #[serde(default)]
    pub lighting: world_sim::Lighting,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<world_sim::SiteCfg>,
}

fn default_noto() -> NotoConfig {
    NotoConfig::preset(FingerCount::Five)
}

fn default_max_trips() -> u32 {
    16
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            seed: 7,
            cells: 64,
            noto: default_noto(),
            body: BodyPlan::standard(),
            terrain: world_sim::TerrainKind::Open,
            camera: CameraCfg::default(),
            gates: GateCfg::default(),
            noise: NoiseCfg::demo(),
            scan_every_ticks: 2,
            approach_scan_every_ticks: 8,
            rotate_deg: 60.0,
            locate_timeout_s: 240.0,
            max_relocations: 10,
            max_rounds: 8,
            max_trips: default_max_trips(),
            approach_reach_m: 0.65,
            min_pixels: 6,
            vision: crate::vision::VisionCfg::default(),
            odometry: world_sim::OdomCfg::default(),
            parcels: None,
            lighting: world_sim::Lighting::Flat,
            site: None,
        }
    }
}
