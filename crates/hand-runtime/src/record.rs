//! Recording a play-through: a low-resolution video and fixed-interval
//! keyframes, each keyframe with what the ML tracks and the decision made of it.
//!
//! The game pass records at a fixed keyframe interval (the timeline's first
//! row). The ML decision pass records one keyframe per decision (the third
//! row): the picture taken after the move the previous decision asked for.
//! Both rows also record a keyframe at each moment of an action's physics
//! (a contact, a lift-off, a slip, an impact, a fall: `KeyEvent`), and every
//! keyframe carries the focus views of that instant (`FocusFrame`: hand,
//! finger joints, feet, leg joints, waist, neck, wheels, tracks) with three
//! close-ups (the hand, the body from the side, low by the feet or wheels).

use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use hand_core::act::focus::FocusFrame;
use hand_core::BalanceSummary;
use hand_vision::{DepthMetrics, Features, Intrinsics, Pose, Rgb};
use serde::{Deserialize, Serialize};
use world_sim::render::{render_focus, FocusCam};
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
    /// Per-track SDS z-scores (or flags) for the SDS lane: `depth`, `locate`, `tracker`, `decide`, `fusion`, `balance`, `act`.
    pub sds: BTreeMap<String, f32>,
    /// The action moment that made this keyframe (`None` for a fixed-interval one).
    pub event: Option<KeyEvent>,
    /// The focus views of this instant.
    pub focus: Option<FocusFrame>,
    /// Focus close-ups as WebP: `hand`, `side`, `low`.
    pub focus_webp: Vec<(String, Vec<u8>)>,
}

/// An action moment that made a keyframe.
#[derive(Clone, Debug, Serialize)]
pub struct KeyEvent {
    pub act: String,
    pub kind: String,
    pub label: String,
    pub ok: bool,
    pub parts: Vec<String>,
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
    /// When the robot looked, and the actions it took on memory (the vision lane).
    pub vision: Vec<crate::vision::VisionMark>,
    /// The belief's true error over time (tick, metres): the record's, never the robot's.
    pub belief_err: Vec<(u64, f32)>,
}

/// Most vision marks kept per recording.
pub const MAX_VISION_MARKS: usize = 4000;

impl Recording {
    pub fn new(pass: &str, run_id: &str, tick: u64) -> Recording {
        Recording { pass: pass.into(), run_id: run_id.into(), keyframes: Vec::new(), video: Vec::new(), started_tick: tick, ended_tick: None, dropped_keyframes: 0, vision: Vec::new(), belief_err: Vec::new() }
    }
}

/// The robot's pose from its own odometry (`world_sim::odometry`): it drifts
/// as a real robot's does, which the triangulation residuals show. The camera
/// is where the robot believes its head camera is on that pose: the head's
/// offset from the body (its joints) turned into the believed heading, the
/// neck's pan on top, and the eye's height above the ground.
pub fn odometry(world: &World, cam: &Camera) -> Pose {
    let (x, z, yaw) = world.believed_pose();
    let est = world.head_view(cam).est;
    let (ox, oz) = (est.origin[0] - world.agent.x, est.origin[2] - world.agent.z);
    let (s, c) = (yaw - world.agent.yaw).sin_cos();
    Pose { x: x + c * ox - s * oz, z: z + s * ox + c * oz, yaw: yaw + (est.yaw - world.agent.yaw), eye_m: est.origin[1] - world.agent_ground_m() }
}

/// The camera's intrinsics with the pitch the robot believes its head camera
/// has now (the mount, the bend and the neck, the shake its IMU reads).
pub fn intrinsics(world: &World, cam: &Camera) -> Intrinsics {
    let est = world.head_view(cam).est;
    Intrinsics { width: cam.width as u32, height: cam.height as u32, fov_deg: cam.fov_deg, pitch_deg: est.pitch.to_degrees() }
}

/// Width and height of the focus close-ups, pixels.
pub const FOCUS_W: usize = 192;
pub const FOCUS_H: usize = 128;

/// The three focus close-ups of the world now, as WebP.
pub fn capture_focus(world: &World, quality: f32) -> Vec<(String, Vec<u8>)> {
    let cam = Camera { width: FOCUS_W, height: FOCUS_H, fov_deg: 50.0, max_dist_m: 12.0, ..Camera::default() };
    FocusCam::ALL
        .into_iter()
        .filter_map(|c| {
            let f = render_focus(world, &cam, c);
            encode_webp(&f.rgb, f.width as u32, f.height as u32, quality).ok().map(|e| (c.id().to_string(), e.bytes))
        })
        .collect()
}

/// Renders a keyframe with its ground truth and the focus views of the instant.
pub fn capture_keyframe(world: &World, cam: &Camera, quality: f32, index: u32) -> Result<Keyframe> {
    let frame = world.render(cam);
    let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    Ok(Keyframe {
        index,
        tick: world.tick,
        time_s: world.time_s,
        pose: odometry(world, cam),
        intr: intrinsics(world, cam),
        webp: enc.bytes,
        truth: KeyTruth::capture(world, &frame),
        step: None,
        action: world.current_action(),
        balance: world.balance_summary(),
        ml: None,
        ml_webp: None,
        decision: None,
        sds: BTreeMap::new(),
        event: None,
        focus: Some(world.focus_frame()),
        focus_webp: capture_focus(world, quality),
    })
}

/// A small video frame.
pub fn capture_video(world: &World, cam: &Camera, cfg: &RecordCfg) -> Result<VideoFrame> {
    let c = Camera { width: cfg.video_width as usize, height: cfg.video_height as usize, ..*cam };
    let f = world.render(&c);
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, cfg.quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    Ok(VideoFrame { tick: world.tick, time_s: world.time_s, webp: enc.bytes })
}
