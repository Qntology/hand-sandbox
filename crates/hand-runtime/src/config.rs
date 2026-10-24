//! Settings of the runtime. Every number here is a default that can be changed
//! from the UI or a config file; none of the thresholds has been tuned on real
//! SigLIP2 scores (see `docs/PLAN.md`, section on what is unverified).

use hand_core::{BodyPlan, FingerCount, HandConfig};
use serde::{Deserialize, Serialize};
use world_sim::Camera;

/// Errors injected into the stand-in recogniser so the gates have something to do.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseCfg {
    pub seed: u64,
    /// Standard deviation of the score noise.
    pub score_sd: f32,
    /// Chance that a visible object is reported as a look-alike.
    pub confusion_rate: f32,
    /// Chance per scan of one detection where nothing is.
    pub false_positive_rate: f32,
    /// Chance that a visible object is not reported at all.
    pub miss_rate: f32,
}

impl NoiseCfg {
    /// A perfect recogniser: the truth, with fixed scores.
    pub fn none() -> NoiseCfg {
        NoiseCfg { seed: 1, score_sd: 0.0, confusion_rate: 0.0, false_positive_rate: 0.0, miss_rate: 0.0 }
    }

    /// Noise that makes the gates visible in the demo.
    pub fn demo() -> NoiseCfg {
        NoiseCfg { seed: 99, score_sd: 0.05, confusion_rate: 0.12, false_positive_rate: 0.30, miss_rate: 0.05 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateCfg {
    /// With the gates off every detection is accepted (for before/after tests).
    pub enabled: bool,
    pub min_score: f32,
    pub min_margin: f32,
    /// Good sightings of one landmark before it is trusted.
    pub confirm_sightings: u32,
    pub track_radius_m: f32,
    pub temporal_min_history: usize,
    pub temporal_widen_sd: f64,
    /// Let the SDS raise the score floor from what touching objects confirmed.
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
}

impl Default for CameraCfg {
    fn default() -> Self {
        let c = Camera::default();
        CameraCfg { width: c.width, height: c.height, fov_deg: c.fov_deg, pitch_deg: c.pitch_deg, max_dist_m: c.max_dist_m }
    }
}

impl CameraCfg {
    pub fn camera(&self) -> Camera {
        Camera { width: self.width, height: self.height, fov_deg: self.fov_deg, pitch_deg: self.pitch_deg, max_dist_m: self.max_dist_m }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub seed: u64,
    pub cells: usize,
    /// Hand and base the robot has in the world.
    #[serde(default = "default_hand")]
    pub hand: HandConfig,
    /// Part presets of the body (`hand_core::parts`).
    #[serde(default)]
    pub body: BodyPlan,
    /// The landscape of new worlds (open, or rugged with sea, cliffs and ridges).
    #[serde(default)]
    pub terrain: world_sim::TerrainKind,
    pub camera: CameraCfg,
    pub gates: GateCfg,
    pub noise: NoiseCfg,
    /// Ticks between scans while searching (a scan renders and recognises one frame).
    pub scan_every_ticks: u32,
    /// Ticks between scans while driving toward a landmark.
    pub approach_scan_every_ticks: u32,
    pub rotate_deg: f32,
    /// Simulated seconds allowed for finding one target.
    pub locate_timeout_s: f64,
    pub max_relocations: u32,
    /// Times a quest may go back to searching for more of a resource.
    pub max_rounds: u32,
    /// Round trips a quest may make when the load does not fit in one.
    #[serde(default = "default_max_trips")]
    pub max_trips: u32,
    pub approach_reach_m: f32,
    /// Objects covering fewer pixels than this are not reported.
    pub min_pixels: u32,
}

fn default_hand() -> HandConfig {
    HandConfig::preset(FingerCount::Five)
}

fn default_max_trips() -> u32 {
    16
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            seed: 7,
            cells: 64,
            hand: default_hand(),
            body: BodyPlan::default(),
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
        }
    }
}
