use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use noto_core::act::focus::FocusFrame;
use noto_core::BalanceSummary;
use noto_vision::{DepthMetrics, Features, Intrinsics, Pose, Rgb};
use serde::{Deserialize, Serialize};
use world_sim::render::{render_focus, FocusCam};
use world_sim::{decode_webp, encode_webp, Camera, World};

use crate::ml::KeyTruth;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordCfg {
    pub keyframe_every_ticks: u32,
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
    /// The 1 : 2 : 4 grids this keyframe ran on and what the cross-layer checks changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pyramid: Option<noto_vision::PyramidReport>,
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
    /// The right camera's frame when the eye is a stereo pair (`webp` is then the left one); SGBM matches the two.
    pub webp_right: Option<Vec<u8>>,
    pub truth: KeyTruth,
    pub step: Option<String>,
    pub action: Option<String>,
    pub balance: Option<BalanceSummary>,
    pub ml: Option<MlView>,
    pub ml_webp: Option<Vec<u8>>,
    pub decision: Option<DecisionView>,
    pub sds: BTreeMap<String, f32>,
    pub event: Option<KeyEvent>,
    pub focus: Option<FocusFrame>,
    pub focus_webp: Vec<(String, Vec<u8>)>,
}

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

    /// The stereo pair's right image, when the keyframe has one.
    pub fn rgb_right(&self) -> Result<Option<Rgb>> {
        let Some(webp) = &self.webp_right else { return Ok(None) };
        let (w, h, data) = decode_webp(webp).map_err(|e| anyhow!("키프레임 오른쪽 화면 WebP 해독 실패: {e}"))?;
        Ok(Some(Rgb::new(w, h, data)))
    }
}

pub struct Recording {
    pub pass: String,
    pub run_id: String,
    pub keyframes: Vec<Keyframe>,
    pub video: Vec<VideoFrame>,
    pub started_tick: u64,
    pub ended_tick: Option<u64>,
    pub dropped_keyframes: usize,
    pub vision: Vec<crate::vision::VisionMark>,
    pub belief_err: Vec<(u64, f32)>,
}

pub const MAX_VISION_MARKS: usize = 4000;

impl Recording {
    pub fn new(pass: &str, run_id: &str, tick: u64) -> Recording {
        Recording { pass: pass.into(), run_id: run_id.into(), keyframes: Vec::new(), video: Vec::new(), started_tick: tick, ended_tick: None, dropped_keyframes: 0, vision: Vec::new(), belief_err: Vec::new() }
    }

    /// The bytes the recording holds: the keyframes' truth pixels (until each is analysed), their images (the frame,
    /// the depth-and-mask overlay, the close-ups) and the video.
    pub fn memory(&self) -> (usize, usize, usize) {
        let pixels = self.keyframes.iter().map(|k| k.truth.pixel_bytes()).sum();
        let images = self.keyframes.iter().map(|k| k.webp.len() + k.webp_right.as_ref().map_or(0, Vec::len) + k.ml_webp.as_ref().map_or(0, Vec::len) + k.focus_webp.iter().map(|(_, w)| w.len()).sum::<usize>()).sum();
        let video = self.video.iter().map(|v| v.webp.len()).sum();
        (pixels, images, video)
    }
}

pub fn odometry(world: &World, cam: &Camera) -> Pose {
    let (x, z, yaw) = world.believed_pose();
    let est = world.head_view(cam).est;
    let (ox, oz) = (est.origin[0] - world.agent.x, est.origin[2] - world.agent.z);
    let (s, c) = (yaw - world.agent.yaw).sin_cos();
    Pose { x: x + c * ox - s * oz, z: z + s * ox + c * oz, yaw: yaw + (est.yaw - world.agent.yaw), eye_m: est.origin[1] - world.agent_ground_m() }
}

pub fn intrinsics(world: &World, cam: &Camera) -> Intrinsics {
    let est = world.head_view(cam).est;
    Intrinsics { width: cam.width as u32, height: cam.height as u32, fov_deg: cam.fov_deg, pitch_deg: est.pitch.to_degrees(), roll_deg: est.roll.to_degrees() }
}

pub const FOCUS_W: usize = 192;
pub const FOCUS_H: usize = 128;

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

/// A keyframe of the eye's frame; with `stereo_baseline_m` (a stereo eye) the right camera's frame too, both kept at
/// `STEREO_KEYFRAME_QUALITY` at least, for SGBM to match what the cameras saw rather than the codec's artefacts.
pub fn capture_keyframe(world: &World, cam: &Camera, quality: f32, index: u32, stereo_baseline_m: Option<f32>) -> Result<Keyframe> {
    let frame = world.render(cam);
    let quality = if stereo_baseline_m.is_some() { quality.max(crate::stereo::STEREO_KEYFRAME_QUALITY) } else { quality };
    let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    let webp_right = match stereo_baseline_m {
        Some(b) => {
            let right = world.render_right(cam, b);
            Some(encode_webp(&right.rgb, right.width as u32, right.height as u32, quality).map_err(|e| anyhow!("오른쪽 화면 WebP 인코딩 실패: {e}"))?.bytes)
        }
        None => None,
    };
    Ok(Keyframe {
        index,
        tick: world.tick,
        time_s: world.time_s,
        pose: odometry(world, cam),
        intr: intrinsics(world, cam),
        webp: enc.bytes,
        webp_right,
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

pub fn capture_video(world: &World, cam: &Camera, cfg: &RecordCfg) -> Result<VideoFrame> {
    let c = Camera { width: cfg.video_width as usize, height: cfg.video_height as usize, ..*cam };
    let f = world.render(&c);
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, cfg.quality).map_err(|e| anyhow!("WebP 인코딩 실패: {e}"))?;
    Ok(VideoFrame { tick: world.tick, time_s: world.time_s, webp: enc.bytes })
}
