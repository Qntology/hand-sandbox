//! The actions' physics in the live record.
//!
//! * Keyframes: every moment of an action's physics the world reports (a
//!   contact, a lift-off, a slip, an impact, a heel slip, a fall, a push lost)
//!   becomes a keyframe of the row being recorded, beside the fixed-interval
//!   ones, with the focus views of that instant (`record::capture_keyframe`).
//! * SDS (`act` track), in four layers per finished action:
//!   1. outcome — `act|<action>|<preset>:<base>`: `ok` agreement, the failure
//!      code, and the calibration of the success the action predicted
//!      (`p_success`, measured 0/1 over predicted): the time-series check of
//!      the robot's own model;
//!   2. time series — `act|<action>:ts|<item or ground>`: every number of the
//!      rollout as a baseline, and against the benchmark's measurement where
//!      it has one (`hand_core::actbench`);
//!   3. part — `act|<action>>part|<state>`: the joints' peak ratios and the
//!      bend's angles at the bottom, the floor-lift angles as references;
//!   4. coupling — `act|nn:<action>|<part>`: what the action asked of another
//!      part (the feet's margin while the hands lift, the camera's pitch).
//!   The engine never reads any of this back.

use hand_core::act::ActKind;
use hand_core::actbench as ab;
use hand_core::balance::{reference, RefCtx, Source};
use hand_core::kin;
use hand_core::parts::Segment;
use hand_sds::{Scope, Track};
use world_sim::{TimedEvent, WorldEvent};

use crate::record::{capture_keyframe, KeyEvent};
use crate::runtime::Runtime;

/// Action moments that make a keyframe.
pub const KEY_KINDS: [&str; 22] = [
    "bend_bottom", "contact", "lift_off", "hold", "slip", "pivot", "drop", "stowed", "release", "impact", "stall", "miss", "reach_fail", "foot_slip", "fall", "breakaway", "turn_slide", "push_lost", "collision", "arrive", "stuck", "wheel_slip",
];
/// Action keyframes come no closer together than this, seconds (one world tick).
pub const KEY_GAP_S: f64 = 0.1;

/// The benchmark's measurement for a rollout number, when it has one: (reference, tolerance).
fn ts_reference(action: ActKind, metric: &str) -> Option<(f32, f32)> {
    match (action, metric) {
        (ActKind::Grasp, "preload_s") => Some(ab::PRELOAD_S),
        (ActKind::Grasp, "loading_s") => Some(ab::LOADING_S),
        (ActKind::Grasp, "gf_lead_s") => Some(ab::GF_LEAD_S),
        (ActKind::Grasp, "gf_lf_corr") => Some(ab::GF_LF_CORR),
        (ActKind::Grasp, "safety_margin") => Some(ab::SAFETY_MARGIN),
        (ActKind::Release, "place_err_m") => Some((ab::PLACE_ERR_MM.0 / 1000.0, ab::PLACE_ERR_MM.1 / 1000.0)),
        _ => None,
    }
}

impl Runtime {
    /// Action moments of this tick become keyframes of the row being recorded.
    pub(crate) fn record_act_keyframes(&mut self, events: &[TimedEvent]) {
        let Some(rec) = self.recording.as_ref() else { return };
        let last = rec.keyframes.iter().rev().find(|k| k.event.is_some()).map_or(f64::NEG_INFINITY, |k| k.time_s);
        let n = rec.keyframes.len();
        if n >= self.test_cfg.record.max_keyframes {
            return;
        }
        // One keyframe per tick: a failure first, else the tick's first moment.
        let moments: Vec<(String, String, String, bool, Vec<String>)> = events
            .iter()
            .filter_map(|e| match &e.event {
                WorldEvent::Act { act, kind, label, ok, parts, .. } if KEY_KINDS.contains(&kind.as_str()) => Some((act.clone(), kind.clone(), label.clone(), *ok, parts.clone())),
                _ => None,
            })
            .collect();
        let Some((act, kind, label, ok, parts)) = moments.iter().find(|m| !m.3).or(moments.first()).cloned() else {
            return;
        };
        // A failure always gets its keyframe; the rest keep a little apart.
        if ok && self.world.time_s - last < KEY_GAP_S {
            return;
        }
        let quality = self.test_cfg.record.quality;
        if let Ok(mut kf) = capture_keyframe(&self.world, &self.cam, quality, n as u32) {
            kf.step = self.current_step_label();
            kf.event = Some(KeyEvent { act: act.clone(), kind: kind.clone(), label, ok, parts });
            // The SDS lane: the moment's number against what this kind of moment usually shows.
            let value = events.iter().find_map(|e| match &e.event {
                WorldEvent::Act { kind: k, value, .. } if *k == kind => *value,
                _ => None,
            });
            if let Some(v) = value {
                let scope = Scope::new(Track::Act, &format!("{act}:{kind}"), "");
                if let Some((m, sd, _)) = self.sds.adaptive_baseline(&scope, "value") {
                    kf.sds.insert("act".into(), ((v as f64 - m) / sd.max(1e-6)) as f32);
                }
                self.sds.record_baseline(&scope, "value", v as f64);
            }
            if let Some(r) = self.recording.as_mut() {
                r.keyframes.push(kf);
            }
        }
    }

    /// A finished action (pick, set-down, push) into the SDS `act` track, layer by layer.
    pub(crate) fn record_act_sds(&mut self, events: &[TimedEvent]) {
        let done = events.iter().any(|e| matches!(&e.event, WorldEvent::ActionDone { action, .. } if action.starts_with("pick_up") || action.starts_with("drop") || action.starts_with("deliver") || action.starts_with("push")));
        if !done {
            return;
        }
        let Some(p) = self.world.last_act.clone() else { return };
        let r = &p.rollout;
        let action = r.kind;
        let preset = format!("{}:{:?}", self.world.cfg.body.id(), self.world.hand.base.kind).to_lowercase();
        let what = p.object.map(|o| o.0.id().to_string()).unwrap_or_default();
        // 1. outcome
        let s1 = Scope::new(Track::Act, action.id(), &preset);
        self.sds.record_agreement(&s1, "ok", r.outcome.ok);
        if let Some(c) = r.outcome.code {
            self.sds.record_agreement(&s1, &format!("code:{c}"), true);
        }
        if let Some(pred) = r.metric("p_success") {
            self.sds.record_calibration(&s1, "p_success", if r.outcome.ok { 1.0 } else { 0.0 }, pred as f64);
        }
        // 2. time series
        let s2 = Scope::new(Track::Act, &format!("{}:ts", action.id()), &what);
        for (k, v) in &r.metrics {
            if !v.is_finite() {
                continue;
            }
            match ts_reference(action, k) {
                Some((rf, tol)) if r.outcome.ok => self.sds.record_reference(&s2, k, *v as f64, rf as f64, tol as f64),
                _ => self.sds.record_baseline(&s2, k, *v as f64),
            }
        }
        // 3. part: the deepest sample of the action (lowest palm), its joints and angles.
        let bottom = r.samples.iter().filter(|s| matches!(s.phase, "reach" | "close" | "squeeze" | "load" | "lower" | "open")).min_by(|a, b| {
            let za = kin::solve_fast(&self.world.body, &a.pose, &a.loads).hands[0][2];
            let zb = kin::solve_fast(&self.world.body, &b.pose, &b.loads).hands[0][2];
            za.total_cmp(&zb)
        });
        if let Some(b) = bottom {
            let k = kin::solve_fast(&self.world.body, &b.pose, &b.loads);
            let s3 = Scope::new(Track::Act, &format!("{}>part", action.id()), b.phase);
            for j in k.joints.iter().filter(|j| !j.structural) {
                self.sds.record_baseline(&s3, &format!("{}:ratio", j.name), j.ratio as f64);
            }
            let trunk = k.pose.trunk_deg();
            self.sds.record_reference(&s3, "trunk_deg", trunk as f64, ab::LIFT_TRUNK_DEG.0 as f64, ab::LIFT_TRUNK_DEG.1 as f64);
            if self.world.body.legged {
                if let Some(h) = k.joint("hip_pitch_r").and_then(|j| j.angle_deg) {
                    self.sds.record_reference(&s3, "hip_flex_deg", h as f64, ab::LIFT_HIP_DEG.0 as f64, ab::LIFT_HIP_DEG.1 as f64);
                }
                if let Some(n) = k.joint("knee_r").and_then(|j| j.angle_deg) {
                    self.sds.record_reference(&s3, "knee_flex_deg", n as f64, ab::LIFT_KNEE_DEG.0 as f64, ab::LIFT_KNEE_DEG.1 as f64);
                }
            }
            // 4. coupling: the feet's margin while the hands work, the camera's pitch.
            let s4 = Scope::new(Track::Act, &format!("nn:{}", action.id()), "foot");
            match reference(Source::ArmLift, Segment::Foot, "margin", &RefCtx::default()) {
                Some(rf) => self.sds.record_reference(&s4, "margin", k.margin as f64, rf.value as f64, rf.tolerance as f64),
                None => self.sds.record_baseline(&s4, "margin", k.margin as f64),
            }
            let s4n = Scope::new(Track::Act, &format!("nn:{}", action.id()), "neck");
            self.sds.record_baseline(&s4n, "camera_pitch_deg", (k.pose.trunk_deg() + k.pose.neck_deg) as f64);
        }
    }
}
