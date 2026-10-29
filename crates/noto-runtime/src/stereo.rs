//! The robot's eye as one camera or a stereo pair (`CameraCfg::cameras`), and how a look measures the range inside a
//! box with each: the two depth procedures.
//!
//! One camera — the TIPSv2 procedure: the monocular model predicts depth from the left image alone, the ground under
//! the camera's own pose turns it into metres (`AlignFit`, 1/Z = a/Z_model + b), and its error is the fit's median
//! relative error. No ground fit, no range.
//!
//! Two cameras — the SGBM procedure: the renderer draws the right camera one baseline along the eye's right axis,
//! SGBM matches the pair on the 1 : 2 : 4 grid and triangulates, Z = f·b/d — metres as it comes out, no alignment —
//! and its error is the stereo law σ_Z = Z²·σ_d/(f·b) (`noto_core::act::material::stereo_depth_sigma`, the same one
//! `MemCtx::sigma_obs` uses). Where it could not match (no texture, a highlight, a strip one eye cannot see, nearer
//! than its disparity range) the box has no depth, and the look gives no range rather than a guessed one.

use noto_vision::{stereo_grid, DepthMap, Intrinsics, Pose, PyramidCfg, SgbmCfg, StereoRig};
use serde::Serialize;
use world_sim::Camera;

/// WebP quality of a stereo keyframe's two images. The codec's artefacts differ between the eyes and are what SGBM
/// then matches: σ_d 0.57 px at 70, 0.40 at 85, 0.27 at 95 against 0.24 px uncompressed, for 11 / 21 / 45 KB a pair
/// [측정: noto-vision `sgbm_render_eval::compare_sources_of_disparity_error`, 4 head-camera scenes].
pub const STEREO_KEYFRAME_QUALITY: f32 = 95.0;

/// σ_d of SGBM on the renderer's pairs at `STEREO_KEYFRAME_QUALITY` [측정: 0.29 px MAD over 8 scenes, head and
/// carriage cameras; 0.24 px on the uncompressed frames the looks use], in pixels of its matching grid.
pub const SGBM_SUBPIXEL_PX: f32 = 0.27;

/// The least share of a box with stereo depth, and the least pixels, for the box to have a range at all.
pub const BOX_MIN_VALID: f32 = 0.05;
pub const BOX_MIN_PIXELS: u32 = 8;

/// The rig of an eye for a camera: the renderer's focal length at the frame's width, the head sensor's baseline
/// (OAK-D Lite 7.5 cm, RealSense D435i 5 cm; the carriage camera is taken to be the same module) [직접: 센서 사양].
pub fn rig_for(cam: &Camera, sensor: &str) -> StereoRig {
    let (_, b, _) = world_sim::world::camera_of(sensor);
    StereoRig { focal_px: world_sim::render::focal_px(cam), baseline_m: b, subpixel_px: SGBM_SUBPIXEL_PX }
}

/// (f, b, σ_d) of the rig on its matching grid: the camera `MemCtx::sigma_obs` puts into `stereo_depth_sigma`.
pub fn sigma_camera(rig: &StereoRig, cam: &Camera, cfg: &SgbmCfg) -> (f32, f32, f32) {
    let (gw, _) = stereo_grid(cam.width as u32, cam.height as u32, Some(&PyramidCfg::default()), cfg.per_l2);
    (rig.focal_on(cam.width as u32, gw), rig.baseline_m, rig.subpixel_px)
}

/// The range inside a box of the frame, measured by the eye's depth procedure.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct BoxDepth {
    /// "sgbm" (stereo, metric), "tipsv2+ground" (monocular, aligned to the ground) or "stand-in".
    pub source: String,
    /// Median depth (along the optical axis) of the box's pixels that have one.
    pub depth_m: Option<f32>,
    /// Where that puts the box's centre on the ground, and how far that is from the robot.
    pub at: Option<(f32, f32)>,
    pub range_m: Option<f32>,
    pub sigma_m: Option<f32>,
    /// The share of the box's pixels with depth.
    pub valid_frac: f32,
    pub reason: Option<String>,
}

impl BoxDepth {
    pub fn none(source: &str, reason: impl Into<String>) -> BoxDepth {
        BoxDepth { source: source.into(), reason: Some(reason.into()), ..BoxDepth::default() }
    }

    pub fn stereo(&self) -> bool {
        self.source == "sgbm"
    }

    pub fn source_ko(&self) -> &'static str {
        match self.source.as_str() {
            "sgbm" => "SGBM 스테레오",
            "tipsv2+ground" => "TIPSv2 + 지면 정렬",
            _ => "깊이 대역",
        }
    }
}

/// The depth inside the central half of `bbox` (normalised to the frame) of `depth`: its median over the pixels
/// that have one, the share that do, and where that puts the item — the surface point on the box's centre ray at that
/// depth, moved on to the centre of an item `centre_m` deep, as the scan places what it sees
/// (`perception::centre_from_surface`).
pub fn in_box(depth: &DepthMap, intr: &Intrinsics, pose: &Pose, bbox: [f32; 4], centre_m: f32) -> (Option<f32>, f32, Option<(f32, f32)>) {
    let (w, h) = (intr.width as f32, intr.height as f32);
    let (sx, sy) = (depth.width as f32 / w, depth.height as f32 / h);
    let (x0, x1) = ((((bbox[0] * 0.75 + bbox[2] * 0.25) * w) * sx) as u32, (((bbox[0] * 0.25 + bbox[2] * 0.75) * w) * sx) as u32);
    let (y0, y1) = ((((bbox[1] * 0.75 + bbox[3] * 0.25) * h) * sy) as u32, (((bbox[1] * 0.25 + bbox[3] * 0.75) * h) * sy) as u32);
    let mut vals = Vec::new();
    let mut n = 0u32;
    for y in y0..=y1.min(depth.height.saturating_sub(1)) {
        for x in x0..=x1.min(depth.width.saturating_sub(1)) {
            n += 1;
            let d = depth.at(x, y);
            if d.is_finite() && d > 0.0 {
                vals.push(d);
            }
        }
    }
    let frac = vals.len() as f32 / n.max(1) as f32;
    if vals.is_empty() {
        return (None, frac, None);
    }
    vals.sort_by(f32::total_cmp);
    let d = vals[vals.len() / 2];
    let (cx, cy) = ((bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h);
    let ray = intr.ray(pose.yaw, cx, cy);
    let surface = [pose.x + d * ray[0], pose.eye_m + d * ray[1], pose.z + d * ray[2]];
    (Some(d), frac, Some(crate::perception::centre_from_surface_at([pose.x, pose.eye_m, pose.z], surface, centre_m)))
}

/// Whether a box has enough stereo depth to give a range: the fail-safe that keeps a look from turning a few stray
/// matches into a distance.
pub fn enough_stereo(valid_frac: f32, pixels: u32) -> bool {
    valid_frac >= BOX_MIN_VALID && pixels >= BOX_MIN_PIXELS
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_vision::DepthKind;

    fn intr() -> Intrinsics {
        Intrinsics { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0, roll_deg: 0.0 }
    }

    #[test]
    fn the_rig_follows_the_head_sensor_and_the_grid() {
        let cam = Camera::default();
        let d435 = rig_for(&cam, "RealSense D435i");
        let oak = rig_for(&cam, "OAK-D Lite");
        assert!((d435.baseline_m - 0.05).abs() < 1e-6 && (oak.baseline_m - 0.075).abs() < 1e-6);
        assert!((d435.focal_px - world_sim::render::focal_px(&cam)).abs() < 1e-4);
        let (f, b, sub) = sigma_camera(&d435, &cam, &SgbmCfg::default());
        assert!((f - d435.focal_px * 0.75).abs() < 1e-3, "the 288-px grid of a 384-px frame: {f}");
        assert_eq!((b, sub), (0.05, SGBM_SUBPIXEL_PX));
    }

    #[test]
    fn a_box_reads_only_the_pixels_with_depth() {
        let i = intr();
        let pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 1.0 };
        let mut data = vec![f32::NAN; 288 * 192];
        for y in 80..112 {
            for x in 128..160 {
                if (x + y) % 3 == 0 {
                    data[y * 288 + x] = 2.0;
                }
            }
        }
        let map = DepthMap { width: 288, height: 192, kind: DepthKind::Metric, data };
        let bbox = [128.0 / 288.0 - 0.02, 80.0 / 192.0, 160.0 / 288.0 + 0.02, 112.0 / 192.0];
        let (d, frac, at) = in_box(&map, &i, &pose, bbox, 0.0);
        assert_eq!(d, Some(2.0));
        assert!(frac > 0.2 && frac < 0.4, "{frac}");
        assert!(at.is_some_and(|(x, _)| x > 1.5), "{at:?}");
        // An item 5 cm deep: its centre lies on behind the surface the eye sees, along the line of sight.
        let (_, _, deep) = in_box(&map, &i, &pose, bbox, 0.05);
        let (s, c) = (at.unwrap(), deep.unwrap());
        assert!(c.0 - s.0 > 0.02 && c.0 - s.0 <= 0.05, "{s:?} → {c:?}");
        let empty = DepthMap { width: 288, height: 192, kind: DepthKind::Metric, data: vec![f32::NAN; 288 * 192] };
        let (d, frac, at) = in_box(&empty, &i, &pose, bbox, 0.05);
        assert!(d.is_none() && frac == 0.0 && at.is_none(), "no match, no range");
        assert!(!enough_stereo(0.04, 100) && !enough_stereo(0.5, 7) && enough_stereo(0.05, 8));
    }
}
