//! ML track 4: the decision. Forecast where the target is, choose the move,
//! predict what the next keyframe should show, then look again.
//!
//! One decision per keyframe, in this order:
//!
//! 1. Compare what the tracker sees with what the previous decision predicted
//!    (the bearing residual). That is the "interpretation" of the move just
//!    made: a large residual means the target, the track or the odometry is
//!    not what the decision assumed.
//! 2. Triangulate from the sightings of the same track. Until a fix exists the
//!    decision is a mandatory sideways step ([`crate::triangulate::plan_step`]);
//!    a monocular depth model cannot be trusted for metres on this renderer.
//! 3. Feed the fixes into a time series of target positions and forecast it
//!    with `noto_ts::forecast` (last, EWMA, Holt and Granite TTM when the
//!    runtime passes it in). The fused forecast is the position used; a fix
//!    outside the forecast band is held and re-triangulated rather than
//!    followed.
//! 4. Move a bounded hop toward the forecast and face it, or arrive.
//!
//! With nothing tracked the decision turns to look; after a full turn with
//! nothing in sight it asks for another viewpoint (`Explore`): on a map where
//! ridges and mesas hide the target, turning on the spot cannot find it. Where
//! to go is the executor's call (it knows the terrain, not the target).

use noto_ts::{forecast, Predictor};
use serde::{Deserialize, Serialize};

use crate::camera::{wrap_angle, Pose};
use crate::triangulate::{plan_step, triangulate, Bearing, Fix, TriCfg, TriStep};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecideCfg {
    /// Distance at which the target counts as reached, metres.
    pub reach_m: f32,
    /// Longest move between two keyframes, metres.
    pub hop_max_m: f32,
    /// Turn used while searching, degrees.
    pub search_turn_deg: f32,
    /// A fix is good enough when its range sigma is below this share of the range.
    pub max_sigma_frac: f32,
    /// Sightings kept for triangulation.
    pub max_rays: usize,
    /// A fix further than this many sigmas outside the forecast band is held.
    pub band_widen_sigma: f32,
    /// Range guess when there is no depth either, metres.
    pub default_range_m: f32,
    /// Below this depth range (times the learned scale) the target is close
    /// enough that depth alone decides: a metre-scale error is centimetres here.
    pub close_range_m: f32,
    pub tri: TriCfg,
}

impl Default for DecideCfg {
    fn default() -> Self {
        DecideCfg {
            reach_m: 0.8,
            hop_max_m: 3.0,
            search_turn_deg: 50.0,
            max_sigma_frac: 0.15,
            max_rays: 4,
            band_widen_sigma: 2.0,
            default_range_m: 6.0,
            close_range_m: 1.6,
            tri: TriCfg::default(),
        }
    }
}

/// What the ML tracks saw of the target in one keyframe.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sighting {
    pub keyframe: u32,
    pub pose: Pose,
    /// World heading of the target's mask centroid, radians.
    pub heading: f32,
    /// Bearing noise, radians.
    pub sigma: f32,
    /// Ground range the depth model gives at the mask, metres.
    pub depth_range_m: Option<f32>,
    pub track_id: u32,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Decision {
    Search { turn_deg: f32 },
    LookDown { pitch_deg: f32, turn_deg: f32 },
    Triangulate(TriStep),
    Approach { to: (f32, f32), hop_m: f32, face: (f32, f32), look_pitch_deg: f32 },
    Arrive { at: (f32, f32), look_pitch_deg: f32 },
    Hold { reason: String },
    Explore { turned_deg: f32 },
}

impl Decision {
    pub fn label(&self) -> String {
        match self {
            Decision::Search { turn_deg } => format!("search(turn {turn_deg:+.0}°)"),
            Decision::LookDown { pitch_deg, turn_deg } => format!("look_down(pitch {pitch_deg:+.0}°, turn {turn_deg:+.0}°)"),
            Decision::Triangulate(t) => format!("tri_step({}, {:.2} m)", t.kind, t.step_m),
            Decision::Approach { hop_m, look_pitch_deg, .. } => format!("move({hop_m:.1} m, pitch {look_pitch_deg:+.0}°)"),
            Decision::Arrive { look_pitch_deg, .. } => format!("arrive(pitch {look_pitch_deg:+.0}°)"),
            Decision::Hold { reason } => format!("hold({reason})"),
            Decision::Explore { turned_deg } => format!("explore(after {turned_deg:.0}°)"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub bearing_rel_deg: f32,
    pub range_m: f32,
    pub pitch_deg: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub x: f32,
    pub z: f32,
    /// Range from the current pose, metres.
    pub range_m: f32,
    pub sigma_m: f32,
    /// `triangulation`, `forecast` or `depth`.
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionReport {
    pub decision: Decision,
    pub estimate: Option<Estimate>,
    /// The fix made at this keyframe, if any.
    pub fix: Option<Fix>,
    /// Why a fix could not be made (`low_parallax`, ...).
    pub fix_error: Option<String>,
    /// Triangulated range over the depth model's range at this keyframe.
    pub depth_scale: Option<f32>,
    /// Observed minus predicted bearing at this keyframe, degrees.
    pub residual_deg: Option<f32>,
    pub prediction: Option<Prediction>,
    /// Forecast method and the band `[x_lo, x_hi, z_lo, z_hi]` of the target position.
    pub forecast_method: String,
    pub band: Option<[f32; 4]>,
    pub note: String,
}

pub struct Decider {
    pub cfg: DecideCfg,
    rays: Vec<Sighting>,
    est_x: Vec<f64>,
    est_z: Vec<f64>,
    last_fix: Option<Fix>,
    prediction: Option<Prediction>,
    side: f32,
    lost: u32,
    /// Search turning since the last sighting or viewpoint change, degrees.
    turned_deg: f32,
}

impl Decider {
    pub fn new(cfg: DecideCfg) -> Decider {
        Decider { cfg, rays: Vec::new(), est_x: Vec::new(), est_z: Vec::new(), last_fix: None, prediction: None, side: 1.0, lost: 0, turned_deg: 0.0 }
    }

    /// The side the next triangulation step goes to (+1 the camera's right, −1 its left).
    pub fn side(&self) -> f32 {
        self.side
    }

    /// Keeps stepping to `side` from now on (the executor found the other
    /// side blocked by terrain and took this one).
    pub fn set_side(&mut self, side: f32) {
        self.side = if side < 0.0 { -1.0 } else { 1.0 };
    }

    /// Forgets the target (a new one is wanted, or the last one was wrong).
    pub fn reset(&mut self) {
        *self = Decider::new(self.cfg);
    }

    pub fn has_fix(&self) -> bool {
        self.last_fix.is_some()
    }

    /// Track id of the target the sightings belong to, if any.
    pub fn track_id(&self) -> Option<u32> {
        self.rays.last().map(|r| r.track_id)
    }

    /// A planned move could not be made (no path, refused): try the other side next time.
    pub fn move_failed(&mut self) {
        self.side = -self.side;
        self.prediction = None;
    }

    fn forecast_position(&self, extra: &[&dyn Predictor]) -> Option<(f32, f32, [f32; 4], String)> {
        if self.est_x.is_empty() {
            return None;
        }
        let fx = forecast(&self.est_x, 1, extra, &[]);
        let fz = forecast(&self.est_z, 1, extra, &[]);
        Some((fx.point[0] as f32, fz.point[0] as f32, [fx.p10[0] as f32, fx.p90[0] as f32, fz.p10[0] as f32, fz.p90[0] as f32], fx.method))
    }

    /// One decision for the keyframe just analysed. `sighting` is `None` when
    /// the target is not tracked in this keyframe. `extra` are additional
    /// forecasters (Granite TTM). `depth_scale` is the SDS's learned scale of
    /// the depth model, if any.
    pub fn observe(&mut self, pose: &Pose, sighting: Option<Sighting>, holonomic: bool, extra: &[&dyn Predictor], depth_scale: Option<f32>) -> DecisionReport {
        let mut rep = DecisionReport {
            decision: Decision::Hold { reason: "start".into() },
            estimate: None,
            fix: None,
            fix_error: None,
            depth_scale: None,
            residual_deg: None,
            prediction: None,
            forecast_method: String::new(),
            band: None,
            note: String::new(),
        };
        let predicted = self.prediction.take();
        let calc_pitch = |eye_m: f32, dist_m: f32| -> f32 {
            let dz = (eye_m - 0.05).max(0.1);
            let direct = -dz.atan2(dist_m.max(0.2)).to_degrees();
            if dist_m <= 1.6 {
                direct.clamp(-60.0, -12.0)
            } else {
                -12.0
            }
        };

        let Some(s) = sighting else {
            self.lost += 1;
            if let Some((x, z, band, method)) = self.forecast_position(extra) {
                let dist = (x - pose.x).hypot(z - pose.z);
                let rel = wrap_angle((z - pose.z).atan2(x - pose.x) - pose.yaw).to_degrees();
                rep.forecast_method = method;
                rep.band = Some(band);
                if self.lost == 1 && dist <= 1.8 {
                    let pitch_deg = calc_pitch(pose.eye_m, dist);
                    let turn_deg = if rel.abs() > 5.0 { rel } else { 0.0 };
                    rep.decision = Decision::LookDown { pitch_deg, turn_deg };
                    rep.note = format!("근거리({dist:.2} m) 대상 시야 이탈: 고개를 {pitch_deg:+.0}° 숙여 바닥 확인");
                    return rep;
                }
                if self.lost <= 2 && rel.abs() > 8.0 {
                    rep.decision = Decision::Search { turn_deg: rel };
                    rep.note = "추적을 놓쳐 예측 위치 쪽으로 회전".into();
                    return rep;
                }
            }
            if self.turned_deg + self.cfg.search_turn_deg * 0.5 >= 360.0 {
                rep.decision = Decision::Explore { turned_deg: self.turned_deg };
                rep.note = format!("{:.0}° 돌아도 대상 없음: 다른 시점으로 이동", self.turned_deg);
                self.turned_deg = 0.0;
                return rep;
            }
            self.turned_deg += self.cfg.search_turn_deg.abs();
            rep.decision = Decision::Search { turn_deg: self.cfg.search_turn_deg };
            rep.note = "대상 없음: 회전 탐색".into();
            return rep;
        };
        self.lost = 0;
        self.turned_deg = 0.0;
        if let Some(p) = predicted {
            rep.residual_deg = Some(wrap_angle(s.heading - pose.yaw).to_degrees() - p.bearing_rel_deg);
        }
        // Rays of the same track only; a new track id starts over.
        if self.rays.last().map_or(false, |r| r.track_id != s.track_id) {
            self.rays.clear();
        }
        self.rays.push(s);
        if self.rays.len() > self.cfg.max_rays {
            self.rays.remove(0);
        }
        let bearings: Vec<Bearing> = self.rays.iter().map(|r| Bearing { x: r.pose.x, z: r.pose.z, heading: r.heading, sigma: r.sigma }).collect();
        match triangulate(&bearings, self.cfg.tri.min_parallax_deg) {
            Ok(fix) => {
                // Only a good fix in the range where the depth model can be
                // trusted says anything about its scale.
                if let Some(d) = s.depth_range_m.filter(|d| *d > 0.1) {
                    if scale_usable(&fix, d) {
                        rep.depth_scale = Some(fix.range_m / d);
                    }
                }
                // A fix far outside the forecast band is not followed blindly.
                if let Some((x, z, band, _)) = self.forecast_position(extra) {
                    let w = self.cfg.band_widen_sigma * fix.sigma_m;
                    let outside = fix.x < band[0] - w || fix.x > band[1] + w || fix.z < band[2] - w || fix.z > band[3] + w;
                    if outside && self.est_x.len() >= 2 {
                        rep.note = format!("새 측량점이 예측 띠 밖 ({:.1}, {:.1}) vs ({x:.1}, {z:.1})", fix.x, fix.z);
                        rep.fix = Some(fix);
                        self.rays.clear();
                        self.rays.push(s);
                        rep.decision = Decision::Hold { reason: "band".into() };
                        return rep;
                    }
                }
                self.est_x.push(fix.x as f64);
                self.est_z.push(fix.z as f64);
                self.last_fix = Some(fix.clone());
                rep.fix = Some(fix);
            }
            Err(e) => rep.fix_error = Some(e.code().to_string()),
        }

        // Close in, the depth range decides (the triangulation step already
        // happened on the way, or the target was close from the start).
        let near = s.depth_range_m.map(|d| d * depth_scale.unwrap_or(1.0)).filter(|r| *r <= self.cfg.close_range_m);
        if let Some(r) = near {
            let at = (pose.x + r * s.heading.cos(), pose.z + r * s.heading.sin());
            rep.estimate = Some(Estimate { x: at.0, z: at.1, range_m: r, sigma_m: 0.25 * r, source: "depth".into() });
            if r <= self.cfg.reach_m + 0.25 {
                let look_pitch_deg = calc_pitch(pose.eye_m, r);
                rep.decision = Decision::Arrive { at, look_pitch_deg };
                rep.note = format!("가까움: 깊이 거리 {r:.2} m로 도달 (시선 {look_pitch_deg:+.0}°)");
            } else {
                let hop = r - self.cfg.reach_m;
                let next_r = r - hop;
                let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
                let to = (pose.x + hop * s.heading.cos(), pose.z + hop * s.heading.sin());
                self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
                rep.prediction = self.prediction;
                rep.decision = Decision::Approach { to, hop_m: hop, face: at, look_pitch_deg };
                rep.note = format!("가까움: 깊이 거리 {r:.2} m, {hop:.2} m 다가감 (시선 {look_pitch_deg:+.0}°)");
            }
            return rep;
        }

        let good = self.last_fix.as_ref().filter(|f| f.sigma_m <= (self.cfg.max_sigma_frac * f.range_m).max(0.25));
        if good.is_none() {
            let guess = s.depth_range_m.map(|d| d * depth_scale.unwrap_or(1.0)).unwrap_or(self.cfg.default_range_m);
            let step = plan_step(pose, s.heading, guess, holonomic, self.side, &self.cfg.tri);
            if let Some(f) = &self.last_fix {
                rep.note = format!("측량 오차 {:.2} m / 거리 {:.1} m: 기선 추가", f.sigma_m, f.range_m);
            } else {
                rep.note = format!("거리 미확정 (깊이 추정 {guess:.1} m): 삼각 측량 이동");
            }
            if matches!(rep.fix_error.as_deref(), Some("behind")) {
                self.rays.clear();
                self.rays.push(s);
            }
            let next_r = (step.face.0 - step.to.0).hypot(step.face.1 - step.to.1);
            self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: calc_pitch(pose.eye_m, next_r) });
            rep.prediction = self.prediction;
            rep.decision = Decision::Triangulate(step);
            return rep;
        }

        let (x, z, band, method) = self.forecast_position(extra).expect("a good fix implies a series");
        let sigma = good.map(|f| f.sigma_m).unwrap_or(1.0);
        let range = (x - pose.x).hypot(z - pose.z);
        rep.forecast_method = method;
        rep.band = Some(band);
        rep.estimate = Some(Estimate { x, z, range_m: range, sigma_m: sigma, source: if self.est_x.len() > 1 { "forecast".into() } else { "triangulation".into() } });
        if range <= self.cfg.reach_m + 0.25 {
            let look_pitch_deg = calc_pitch(pose.eye_m, range);
            rep.decision = Decision::Arrive { at: (x, z), look_pitch_deg };
            rep.note = format!("도달: 거리 {range:.2} m (시선 {look_pitch_deg:+.0}°)");
            return rep;
        }
        let hop = (range - self.cfg.reach_m).min(self.cfg.hop_max_m);
        let next_r = range - hop;
        let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
        let h = (z - pose.z).atan2(x - pose.x);
        let to = (pose.x + hop * h.cos(), pose.z + hop * h.sin());
        self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
        rep.prediction = self.prediction;
        rep.decision = Decision::Approach { to, hop_m: hop, face: (x, z), look_pitch_deg };
        rep.note = format!("예측 위치 ({x:.1}, {z:.1}) ±{sigma:.2} m 쪽으로 {hop:.1} m (시선 {look_pitch_deg:+.0}°)");
        rep
    }
}

impl Decider {
    /// The 1σ of the target's position the decision would act on now, metres
    /// (its last good fix), if it has one.
    pub fn fix_sigma_m(&self) -> Option<f32> {
        self.last_fix.as_ref().map(|f| f.sigma_m)
    }

    /// A decision without a keyframe (the ML tracks are off): the move the
    /// forecast of the fixes asks for, from the pose the robot believes it
    /// has. `extra_sigma_m` is what the robot's own drift since the last look
    /// adds to the fix's spread. `None` without a good fix (then it must look).
    pub fn predict(&mut self, pose: &Pose, extra: &[&dyn Predictor], extra_sigma_m: f32) -> Option<DecisionReport> {
        let good = self.last_fix.clone().filter(|f| f.sigma_m <= (self.cfg.max_sigma_frac * f.range_m).max(0.25))?;
        let (x, z, band, method) = self.forecast_position(extra)?;
        let sigma = good.sigma_m.hypot(extra_sigma_m);
        let range = (x - pose.x).hypot(z - pose.z);
        let mut rep = DecisionReport {
            decision: Decision::Hold { reason: "blind".into() },
            estimate: Some(Estimate { x, z, range_m: range, sigma_m: sigma, source: "forecast".into() }),
            fix: None,
            fix_error: None,
            depth_scale: None,
            residual_deg: None,
            prediction: None,
            forecast_method: method,
            band: Some(band),
            note: String::new(),
        };
        self.prediction = None;
        let calc_pitch = |eye_m: f32, dist_m: f32| -> f32 {
            let dz = (eye_m - 0.05).max(0.1);
            let direct = -dz.atan2(dist_m.max(0.2)).to_degrees();
            if dist_m <= 1.6 { direct.clamp(-60.0, -12.0) } else { -12.0 }
        };
        if range <= self.cfg.reach_m + 0.25 {
            let look_pitch_deg = calc_pitch(pose.eye_m, range);
            rep.decision = Decision::Arrive { at: (x, z), look_pitch_deg };
            rep.note = format!("눈 감고 예측으로 도달: 거리 {range:.2} m (±{sigma:.2} m, 시선 {look_pitch_deg:+.0}°)");
            return Some(rep);
        }
        let hop = (range - self.cfg.reach_m).min(self.cfg.hop_max_m);
        let next_r = range - hop;
        let look_pitch_deg = calc_pitch(pose.eye_m, next_r);
        let h = (z - pose.z).atan2(x - pose.x);
        let to = (pose.x + hop * h.cos(), pose.z + hop * h.sin());
        self.prediction = Some(Prediction { bearing_rel_deg: 0.0, range_m: next_r, pitch_deg: look_pitch_deg });
        rep.prediction = self.prediction;
        rep.decision = Decision::Approach { to, hop_m: hop, face: (x, z), look_pitch_deg };
        rep.note = format!("눈 감고 예측 위치 ({x:.1}, {z:.1}) ±{sigma:.2} m 쪽으로 {hop:.1} m (ML 꺼짐)");
        Some(rep)
    }
}

/// A fix says something about the depth model's scale when at least three
/// rays agree (two always meet, even if one is on another object), its range
/// error is small (at most 15%), the target is neither at the feet nor near
/// the model's 10 m limit, and the depth model's own range is not tiny.
pub fn scale_usable(fix: &Fix, depth_range_m: f32) -> bool {
    fix.rays >= 3 && fix.residual_m <= 0.15 && fix.sigma_m <= 0.15 * fix.range_m && (1.5..=9.0).contains(&fix.range_m) && depth_range_m >= 0.5
}

/// Bearing noise of a mask centroid: a couple of pixels over the focal length.
pub fn centroid_sigma(width: u32, fov_deg: f32, px_err: f32) -> f32 {
    let f = width as f32 * 0.5 / (fov_deg.to_radians() * 0.5).tan();
    (px_err / f).max(1e-4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sight(k: u32, pose: Pose, target: (f32, f32), id: u32) -> Sighting {
        Sighting {
            keyframe: k,
            pose,
            heading: (target.1 - pose.z).atan2(target.0 - pose.x),
            sigma: 0.006,
            depth_range_m: Some(((target.0 - pose.x).hypot(target.1 - pose.z)) * 0.6),
            track_id: id,
            score: 0.8,
        }
    }

    /// Plays the decisions out with perfect execution and returns them.
    fn play(target: (f32, f32), holonomic: bool, steps: usize) -> Vec<DecisionReport> {
        let mut d = Decider::new(DecideCfg::default());
        let mut pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut out = Vec::new();
        for k in 0..steps as u32 {
            let rep = d.observe(&pose, Some(sight(k, pose, target, 1)), holonomic, &[], None);
            match &rep.decision {
                Decision::Triangulate(t) => {
                    pose = Pose { x: t.to.0, z: t.to.1, yaw: (t.face.1 - t.to.1).atan2(t.face.0 - t.to.0), ..pose };
                }
                Decision::Approach { to, face, .. } => {
                    pose = Pose { x: to.0, z: to.1, yaw: (face.1 - to.1).atan2(face.0 - to.0), ..pose };
                }
                Decision::Arrive { .. } => {
                    out.push(rep);
                    break;
                }
                _ => {}
            }
            out.push(rep);
        }
        out
    }

    #[test]
    fn the_first_decision_is_a_triangulation_step_and_the_target_is_reached() {
        for holonomic in [true, false] {
            let reps = play((9.0, 2.0), holonomic, 20);
            assert!(matches!(reps[0].decision, Decision::Triangulate(_)), "{:?}", reps[0].decision);
            assert!(reps[0].fix.is_none(), "one sighting cannot give a range");
            let fix = reps.iter().find_map(|r| r.fix.clone()).expect("a fix after the step");
            assert!((fix.x - 9.0).abs() < 0.05 && (fix.z - 2.0).abs() < 0.05, "{fix:?}");
            // The depth model said 0.6 of the truth, so the scale it needed is about 1/0.6.
            let scale = reps.iter().find_map(|r| r.depth_scale).unwrap();
            assert!((scale - 1.0 / 0.6).abs() < 0.05, "{scale}");
            let last = reps.last().unwrap();
            assert!(matches!(last.decision, Decision::Arrive { .. }), "{holonomic}: {:?}", reps.iter().map(|r| r.decision.label()).collect::<Vec<_>>());
            // Every approach predicted the target dead ahead and it was. (After a
            // triangulation step the prediction rests on the rough depth guess, so
            // its residual is large; that is what tells the decision the guess was off.)
            for w in reps.windows(2) {
                if matches!(w[0].decision, Decision::Approach { .. }) {
                    let res = w[1].residual_deg.expect("an approach leaves a prediction");
                    assert!(res.abs() < 1.0, "{res}");
                }
            }
        }
    }

    #[test]
    fn a_lost_target_turns_toward_its_forecast_then_searches() {
        let mut d = Decider::new(DecideCfg::default());
        let reps = play((6.0, 0.0), true, 4);
        assert!(reps.iter().any(|r| r.fix.is_some()));
        // Rebuild the same state in `d`, then lose the target behind the robot.
        let mut pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut r = d.observe(&pose, Some(sight(0, pose, (6.0, 0.0), 1)), true, &[], None);
        for k in 1..4 {
            if let Decision::Triangulate(t) = &r.decision {
                pose = Pose { x: t.to.0, z: t.to.1, yaw: 0.0, eye_m: 0.5 };
            }
            r = d.observe(&pose, Some(sight(k, pose, (6.0, 0.0), 1)), true, &[], None);
            if d.has_fix() {
                break;
            }
        }
        assert!(d.has_fix());
        let turned = Pose { yaw: std::f32::consts::PI, ..pose };
        let lost = d.observe(&turned, None, true, &[], None);
        match lost.decision {
            Decision::Search { turn_deg } => assert!(turn_deg.abs() > 150.0, "turn back toward the target: {turn_deg}"),
            other => panic!("{other:?}"),
        }
        d.observe(&turned, None, true, &[], None);
        let third = d.observe(&turned, None, true, &[], None);
        assert_eq!(third.decision, Decision::Search { turn_deg: DecideCfg::default().search_turn_deg });
    }

    #[test]
    fn a_full_turn_without_a_sighting_asks_for_another_viewpoint() {
        let mut d = Decider::new(DecideCfg::default());
        let pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let mut turns = 0.0;
        let explore = loop {
            match d.observe(&pose, None, false, &[], None).decision {
                Decision::Search { turn_deg } => turns += turn_deg,
                Decision::Explore { turned_deg } => break turned_deg,
                other => panic!("{other:?}"),
            }
            assert!(turns < 720.0, "never asks to move");
        };
        assert!(explore + DecideCfg::default().search_turn_deg >= 360.0 - 1e-3, "after about a full turn: {explore}");
        // A sighting starts the count again.
        let t = (5.0, 1.0);
        let r = d.observe(&pose, Some(sight(0, pose, t, 3)), false, &[], None);
        assert!(matches!(r.decision, Decision::Triangulate(_)));
        assert!(matches!(d.observe(&pose, None, false, &[], None).decision, Decision::Search { .. }));
    }
}
