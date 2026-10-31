//! Reading a step off a depth map. The robot looks at the ground a stride or two ahead of its feet; each pixel of that
//! strip, put back through the camera, is a point with a distance along the way and a height above the ground the robot
//! stands on. Binned by distance and taken at the median, the strip is a profile of the ground, and a step is where the
//! profile jumps and stays there. Works on whichever depth the robot has: TIPSv2 aligned to the ground, or SGBM from a
//! stereo pair, whose unmatched pixels are simply missing.

use crate::camera::{Intrinsics, Pose};
use crate::frame::DepthMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCfg {
    /// How far along the way to read (m).
    pub near_m: f32,
    pub far_m: f32,
    /// How wide a strip, each side of the way (m).
    pub half_width_m: f32,
    pub bin_m: f32,
    /// The least height change that is a step (m): under this a depth map's own patches pass for one.
    pub min_step_m: f32,
    /// A bin takes this many points to count.
    pub min_points: usize,
    /// How many bins each side of a point are taken together.
    pub window: usize,
}

impl Default for StepCfg {
    fn default() -> StepCfg {
        StepCfg { near_m: 0.55, far_m: 1.7, half_width_m: 0.28, bin_m: 0.05, min_step_m: 0.06, min_points: 3, window: 4 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepRead {
    /// How far ahead the edge is (m).
    pub dist_m: f32,
    /// Up is positive (m).
    pub height_m: f32,
    /// The share of the profile's bins that had a reading.
    pub coverage: f32,
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// The first step in the strip ahead, if there is one.
pub fn read_step(depth: &DepthMap, intr: &Intrinsics, pose: &Pose, cfg: &StepCfg) -> Option<StepRead> {
    let n = ((cfg.far_m - cfg.near_m) / cfg.bin_m).ceil() as usize;
    let mut bins: Vec<Vec<f32>> = vec![Vec::new(); n];
    let (sx, sy) = (intr.width as f32 / depth.width.max(1) as f32, intr.height as f32 / depth.height.max(1) as f32);
    for y in 0..depth.height {
        for x in 0..depth.width {
            let d = depth.at(x, y);
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            // With yaw 0 the ray's x runs along the way, its z across and its y up.
            let r = intr.ray(0.0, (x as f32 + 0.5) * sx, (y as f32 + 0.5) * sy);
            let (along, across, up) = (r[0] * d, r[2] * d, r[1] * d);
            if across.abs() > cfg.half_width_m || along < cfg.near_m || along >= cfg.far_m {
                continue;
            }
            bins[((along - cfg.near_m) / cfg.bin_m) as usize].push(pose.eye_m + up);
        }
    }
    let profile: Vec<Option<f32>> = bins.iter_mut().map(|b| (b.len() >= cfg.min_points).then(|| median(b))).collect();
    let read = profile.iter().flatten().count();
    if read < 2 * cfg.window {
        return None;
    }
    let coverage = read as f32 / n as f32;
    // A step is a jump between what lies before a point and what lies after it. Each side is taken as a median, because
    // a depth map's error comes in patches larger than a bin; the jump's height does not depend on the depth's scale
    // being right, only its proportion, which an unaligned model gets close enough.
    let side = |r: std::ops::Range<usize>| {
        let mut v: Vec<f32> = profile[r].iter().flatten().copied().collect();
        (v.len() * 2 > cfg.window).then(|| median(&mut v))
    };
    let mut peak: Option<(usize, f32)> = None;
    for i in cfg.window..=n - cfg.window {
        let (Some(a), Some(b)) = (side(i - cfg.window..i), side(i..i + cfg.window)) else { continue };
        let jump = b - a;
        match peak {
            None if jump.abs() >= cfg.min_step_m => peak = Some((i, jump)),
            Some((_, p)) if jump * p > 0.0 && jump.abs() > p.abs() => peak = Some((i, jump)),
            // Past the first edge: anything further on is the next step.
            Some((_, p)) if jump.abs() < 0.5 * p.abs() => break,
            _ => {}
        }
    }
    peak.map(|(i, jump)| StepRead { dist_m: cfg.near_m + i as f32 * cfg.bin_m, height_m: jump, coverage })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::DepthKind;

    /// The depth a camera sees of level ground with a step `h` high starting `edge` ahead.
    fn scene(eye: f32, pitch_deg: f32, edge: f32, h: f32, noise: f32) -> (DepthMap, Intrinsics, Pose) {
        let intr = Intrinsics { width: 192, height: 128, fov_deg: 70.0, pitch_deg, roll_deg: 0.0 };
        let mut data = Vec::new();
        for y in 0..intr.height {
            for x in 0..intr.width {
                let r = intr.ray(0.0, x as f32 + 0.5, y as f32 + 0.5);
                let mut d = f32::NAN;
                if r[1] < -1e-4 {
                    // Onto the upper level, past the edge; else the riser, or the lower level.
                    let top = -(eye - h) / r[1];
                    let low = -eye / r[1];
                    d = if h >= 0.0 {
                        if r[0] * top >= edge {
                            top
                        } else if r[0] * low >= edge {
                            edge / r[0]
                        } else {
                            low
                        }
                    } else if r[0] * top < edge {
                        low.min(edge / r[0]).max(0.0).min(top).max(low.min(top))
                    } else {
                        top
                    };
                    let u = ((x * 73 + y * 37) % 101) as f32 / 100.0 - 0.5;
                    d *= 1.0 + noise * u;
                }
                data.push(d);
            }
        }
        (DepthMap { width: intr.width, height: intr.height, kind: DepthKind::Metric, data }, intr, Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: eye })
    }

    #[test]
    fn level_ground_is_no_step() {
        let (d, i, p) = scene(1.5, -40.0, 9.0, 0.0, 0.01);
        assert_eq!(read_step(&d, &i, &p, &StepCfg::default()), None);
    }

    #[test]
    fn a_step_up_is_read_for_its_height_and_where_it_starts() {
        for (edge, h) in [(1.2f32, 0.15f32), (0.95, 0.08), (1.3, 0.18)] {
            let (d, i, p) = scene(1.5, -40.0, edge, h, 0.005);
            let r = read_step(&d, &i, &p, &StepCfg::default()).expect("a step");
            assert!((r.height_m - h).abs() < 0.03, "{r:?} for {h}");
            assert!((r.dist_m - edge).abs() < 0.15, "{r:?} for {edge}");
        }
    }

    #[test]
    fn a_depth_off_in_scale_reads_the_same_step() {
        let (mut d, i, p) = scene(1.5, -40.0, 1.2, 0.15, 0.0);
        for v in d.data.iter_mut() {
            *v *= 0.85;
        }
        let r = read_step(&d, &i, &p, &StepCfg::default()).expect("a step");
        assert!((r.height_m - 0.15).abs() < 0.04, "{r:?}");
    }

    #[test]
    fn missing_pixels_do_not_make_a_step() {
        let (mut d, i, p) = scene(1.5, -40.0, 9.0, 0.0, 0.0);
        for (k, v) in d.data.iter_mut().enumerate() {
            if k % 3 == 0 {
                *v = f32::NAN;
            }
        }
        assert_eq!(read_step(&d, &i, &p, &StepCfg::default()), None);
    }
}
