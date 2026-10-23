//! Recording a play-through: a low-resolution video and fixed-interval
//! keyframes, each keyframe with what the ML tracks and the decision made of it.
//!
//! The game pass records at a fixed keyframe interval (the timeline's first
//! row). The ML decision pass records one keyframe per decision (the third
//! row): the picture taken after the move the previous decision asked for.

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use hand_core::BalanceSummary;
use hand_vision::{DepthMetrics, Features, Intrinsics, Pose, Rgb};
use serde::{Deserialize, Serialize};
use world_sim::world::EYE_HEIGHT_M;
use world_sim::{decode_webp, encode_webp, Camera, World};

use crate::ml::KeyTruth;

/// Recording settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordCfg {
    /// Ticks between keyframes of the game pass (10 ticks = 1 s).
    pub keyframe_every_ticks: u32,
    /// Ticks between video frames (2 ticks = 5 fps).
    pub video_every_ticks: u32,
    pub video_width: u32,
    pub video_height: u32,
    pub quality: f32,
    pub max_keyframes: usize,
    pub max_video_frames: usize,
}

impl Default for RecordCfg {
    fn default() -> Self {
        RecordCfg { keyframe_every_ticks: 10, video_every_ticks: 2, video_width: 192, video_height: 128, quality: 70.0, max_keyframes: 400, max_video_frames: 3000 }
    }
}

pub struct VideoFrame {
    pub tick: u64,
    pub time_s: f64,
    pub webp: Vec<u8>,
}

/// What the ML tracks saw, for the UI (coordinates are fractions of the frame).
#[derive(Clone, Debug, Default, Serialize)]
pub struct MlView {
    pub depth_mode: String,
    pub locate_mode: String,
    pub track_mode: String,
    pub depth_metrics: Option<DepthMetrics>,
    pub points: Vec<PointView>,
    pub tracks: Vec<TrackView>,
    pub target: Option<TargetView>,
    pub locate_err_px: Option<f32>,
    pub track_iou_truth: Option<f32>,
    pub features: Features,
    pub flags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PointView {
    pub label: String,
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub margin: f32,
    pub presence: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrackView {
    pub id: u32,
    pub label: String,
    pub bbox: [f32; 4],
    pub area_frac: f32,
    pub score: f32,
    pub iou_prev: Option<f32>,
    pub propagated: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TargetView {
    pub track_id: u32,
    pub label: String,
    pub rel_bearing_deg: f32,
    pub depth_range_m: Option<f32>,
    pub tri_range_m: Option<f32>,
    pub tri_parallax_deg: Option<f32>,
    pub tri_sigma_m: Option<f32>,
    pub depth_scale: Option<f32>,
    pub range_used_m: Option<f32>,
    pub range_true_m: Option<f32>,
}

/// What the decision chose on a keyframe and how it went.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DecisionView {
    pub decision: String,
    pub kind: String,
    pub note: String,
    pub residual_deg: Option<f32>,
    pub forecast_method: String,
    pub band: Option<[f32; 4]>,
    pub estimate: Option<(f32, f32, f32, f32)>,
    pub fix: Option<(f32, f32, f32, f32, f32)>,
    /// What the game executed for it (`move_to(...)`, `turn(...)`), or why it could not.
    pub executed: String,
    pub ok: bool,
}

pub struct Keyframe {
    pub index: u32,
    pub tick: u64,
    pub time_s: f64,
    pub pose: Pose,
    pub intr: Intrinsics,
    pub webp: Vec<u8>,
    pub truth: KeyTruth,
    /// Quest step and world action at the time, for the game row's colour band.
    pub step: Option<String>,
    pub action: Option<String>,
    pub balance: Option<BalanceSummary>,
    pub ml: Option<MlView>,
    pub ml_webp: Option<Vec<u8>>,
    pub decision: Option<DecisionView>,
    /// Per-track SDS z-scores (or flags) for the SDS lane: `depth`, `locate`, `tracker`, `decide`, `fusion`, `balance`.
    pub sds: BTreeMap<String, f32>,
}

impl Keyframe {
    pub fn rgb(&self) -> Result<Rgb> {
        let (w, h, data) = decode_webp(&self.webp).map_err(|e| anyhow!("키프레임 WebP 해독 실패: {e}"))?;
        Ok(Rgb::new(w, h, data))
    }
}

/// One pass's recording.
pub struct Recording {
    pub pass: String,
    pub run_id: String,
    pub keyframes: Vec<Keyframe>,
    pub video: Vec<VideoFrame>,
    pub started_tick: u64,
    pub ended_tick: Option<u64>,
    pub dropped_keyframes: usize,
}

impl Recording {
    pub fn new(pass: &str, run_id: &str, tick: u64) -> Recording {
        Recording { pass: pass.into(), run_id: run_id.into(), keyframes: Vec::new(), video: Vec::new(), started_tick: tick, ended_tick: None, dropped_keyframes: 0 }
    }
}

/// The robot's pose from its own odometry. The simulator's odometry is exact;
/// a real robot's would drift, which the triangulation residuals would show.
pub fn odometry(world: &World) -> Pose {
    Pose { x: world.agent.x, z: world.agent.z, yaw: world.agent.yaw, eye_m: EYE_HEIGHT_M }
}

pub fn intrinsics(cam: &Camera) -> Intrinsics {
    Intrinsics { width: cam.width as u32, height: cam.height as u32, fov_deg: cam.fov_deg, pitch_deg: cam.pitch_deg }
}

/// Renders a keyframe with its ground truth.
pub fn capture_keyframe(world: &World, cam: &Camera, quality: f32, index: u32) -> Result<Keyframe> {
    let frame = world.render(cam);
    let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    Ok(Keyframe {
        index,
        tick: world.tick,
        time_s: world.time_s,
        pose: odometry(world),
        intr: intrinsics(cam),
        webp: enc.bytes,
        truth: KeyTruth::capture(world, &frame),
        step: None,
        action: world.current_action(),
        balance: world.balance_summary(),
        ml: None,
        ml_webp: None,
        decision: None,
        sds: BTreeMap::new(),
    })
}

/// A small video frame.
pub fn capture_video(world: &World, cam: &Camera, cfg: &RecordCfg) -> Result<VideoFrame> {
    let c = Camera { width: cfg.video_width as usize, height: cfg.video_height as usize, ..*cam };
    let f = world.render(&c);
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, cfg.quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    Ok(VideoFrame { tick: world.tick, time_s: world.time_s, webp: enc.bytes })
}
