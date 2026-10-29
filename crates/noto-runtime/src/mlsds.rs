use std::collections::BTreeMap;

use anyhow::Result;
use noto_core::balance::{reference, Source};
use noto_sds::{BalanceRow, KeyframeRow, Level, QuestRun, Scope, StepRecord, Track};
use noto_vision::{features, flags, FlagCfg, FusionInput, Sighting};
use serde_json::json;
use world_sim::encode_webp;

use crate::ml::{overlay, MlFrame};
use crate::record::{Keyframe, MlView, PointView, TargetView, TrackView};
use crate::runtime::Runtime;

fn default_floor(mode: &str) -> f32 {
    if mode.starts_with("stand-in") && !mode.contains("measured") {
        0.35
    } else {
        0.0
    }
}

fn is_stand_in(mode: &str) -> bool {
    mode.starts_with("stand-in")
}

fn z_of(rt: &Runtime, scope: &Scope, axis: &str, x: f64) -> Option<f64> {
    rt.sds.adaptive_baseline(scope, axis).map(|(m, sd, _)| (x - m) / sd.max(1e-6))
}

pub(crate) struct MlOut {
    pub frame: MlFrame,
    pub sighting: Option<Sighting>,
}

impl Runtime {
    pub(crate) fn depth_scale(&self, label: &str) -> Option<f32> {
        let mode = self.ml.depth_mode();
        let scope = Scope::new(Track::Depth, &mode, label);
        self.sds.calibration_factor(&scope, "scale").filter(|(_, _, _, key)| is_stand_in(&mode) || *key != scope.key_global()).map(|(f, _, _, _)| f as f32)
    }

    pub(crate) fn ml_analyse(&mut self, kf: &mut Keyframe, wanted: &[String], prefer: Option<u32>) -> Result<MlOut> {
        let rgb = kf.rgb()?;
        let lmode = self.ml.locate_mode();
        let floor_default = default_floor(&lmode);
        let floors: BTreeMap<String, f32> = wanted
            .iter()
            .map(|l| {
                let s = Scope::new(Track::Locate, l, &lmode);
                let learned = self.sds.adaptive_baseline_scoped(&s, "score").filter(|(_, key)| is_stand_in(&lmode) || *key == s.key_secondary()).map(|(b, _)| b);
                let f = learned.map(|(m, sd, _)| (m - 2.0 * sd) as f32).unwrap_or(floor_default).max(floor_default.min(0.2));
                (l.clone(), f)
            })
            .collect();
        let ground0 = self.world.surface_m(kf.pose.x, kf.pose.z);
        let world = &self.world;
        let height = move |x: f32, z: f32| world.surface_m(x, z) - ground0;
        let frame = self.ml.analyse_with(&rgb, &kf.intr, &kf.pose, Some(&kf.truth), wanted, &|l| floors.get(l).copied().unwrap_or(floor_default), prefer, Some(&height))?;
        let (dmode, tmode) = (self.ml.depth_mode(), self.ml.track_mode());
        let label = frame.target.as_ref().map(|t| t.label.clone()).or_else(|| wanted.first().cloned()).unwrap_or_default();

        let dscope = Scope::new(Track::Depth, &dmode, &label);
        if let Some(m) = &frame.depth_metrics {
            self.sds.record_baseline(&dscope, "absrel", m.absrel as f64);
            self.sds.record_baseline(&dscope, "delta1", m.delta1 as f64);
        }
        if let Some(f) = &frame.align {
            self.sds.record_baseline(&dscope, "align_a", f.a as f64);
            self.sds.record_baseline(&dscope, "align_b", f.b as f64);
            self.sds.record_baseline(&dscope, "align_inliers", f.inlier_share as f64);
            self.sds.record_baseline(&dscope, "align_err", f.median_rel_err as f64);
        }
        let point = frame.points.iter().find(|p| wanted.contains(&p.label)).cloned();
        let lscope = Scope::new(Track::Locate, &label, &lmode);
        let mut sds_lane: BTreeMap<String, f32> = BTreeMap::new();
        if let Some(p) = &point {
            if let Some(z) = z_of(self, &lscope, "score", p.score as f64) {
                sds_lane.insert("locate".into(), z as f32);
            }
            self.sds.record_baseline(&lscope, "score", p.score as f64);
            self.sds.record_baseline(&lscope, "margin", p.margin as f64);
            self.sds.record_baseline(&lscope, "presence", p.presence as f64);
            if let Some(e) = frame.locate_err_px {
                self.sds.record_baseline(&lscope, "err_px", e as f64);
            }
        }
        let tscope = Scope::new(Track::Tracker, &label, &tmode);
        let target_track = frame.target.as_ref().and_then(|t| frame.tracks.iter().find(|m| m.id == t.track_id));
        self.sds.record_agreement(&tscope, "alive", target_track.is_some());
        if let Some(t) = target_track {
            if let Some(iou) = t.iou_prev {
                if let Some(z) = z_of(self, &tscope, "iou_prev", iou as f64) {
                    sds_lane.insert("tracker".into(), z as f32);
                }
                self.sds.record_baseline(&tscope, "iou_prev", iou as f64);
            }
            self.sds.record_baseline(&tscope, "area_frac", t.area_frac as f64);
            self.sds.record_baseline(&tscope, "score", t.score as f64);
            if let Some(iou) = frame.track_iou_truth {
                self.sds.record_baseline(&tscope, "iou_truth", iou as f64);
            }
        }
        if let Some(m) = &frame.depth_metrics {
            if let Some(z) = z_of(self, &dscope, "absrel", m.absrel as f64) {
                sds_lane.insert("depth".into(), z as f32);
            }
        }

        let scale = self.depth_scale(&label);
        let sighting = frame.target.as_ref().map(|t| Sighting {
            keyframe: kf.index,
            pose: kf.pose,
            heading: t.heading,
            sigma: noto_vision::centroid_sigma(kf.intr.width, kf.intr.fov_deg, 2.0),
            extent: target_track.map_or(0.0, |m| (kf.intr.relative_bearing(&kf.pose, m.bbox[2], m.cy) - kf.intr.relative_bearing(&kf.pose, m.bbox[0], m.cy)).abs()),
            depth_range_m: t.depth_range_m,
            track_id: t.track_id,
            score: t.score,
        });
        let head_tilt = kf.balance.as_ref().map(|b| b.head_tilt_deg);
        let feats = features(&FusionInput {
            width: kf.intr.width,
            height: kf.intr.height,
            depth: Some(&frame.depth),
            point: point.as_ref(),
            mask: target_track.map(|t| &t.mask),
            depth_range_m: frame.target.as_ref().and_then(|t| t.depth_range_m),
            depth_scale: scale,
            tri_range_m: None,
            residual_deg: None,
            head_tilt_deg: head_tilt,
        });
        let mut fl: Vec<String> = flags(&feats, &FlagCfg::default()).into_iter().map(String::from).collect();
        let fscope = Scope::new(Track::Fusion, &label, "");
        let mut zmax = 0.0f64;
        for (axis, x) in noto_vision::fusion::numeric(&feats) {
            if let Some(z) = z_of(self, &fscope, axis, x) {
                if z.abs() > zmax.abs() {
                    zmax = z;
                }
                if z.abs() > 2.5 {
                    fl.push(format!("{axis}_z"));
                }
            }
            self.sds.record_baseline(&fscope, axis, x);
        }
        sds_lane.insert("fusion".into(), zmax as f32);
        if let Some(b) = &kf.balance {
            sds_lane.insert("balance".into(), ((noto_core::balance::MARGIN_KEEP - b.margin) / 0.1).max(0.0));
        }

        let (w, h) = (kf.intr.width as f32, kf.intr.height as f32);
        let view = MlView {
            depth_mode: dmode.clone(),
            locate_mode: lmode.clone(),
            track_mode: tmode.clone(),
            depth_metrics: frame.depth_metrics,
            points: frame.points.iter().filter(|p| wanted.contains(&p.label) || p.score > 0.5).map(|p| PointView { label: p.label.clone(), x: p.px / w, y: p.py / h, score: p.score, margin: p.margin, presence: p.presence }).collect(),
            tracks: frame
                .tracks
                .iter()
                .map(|t| TrackView { id: t.id, label: t.label.clone(), bbox: [t.bbox[0] / w, t.bbox[1] / h, t.bbox[2] / w, t.bbox[3] / h], area_frac: t.area_frac, score: t.score, iou_prev: t.iou_prev, propagated: t.propagated })
                .collect(),
            target: frame.target.as_ref().map(|t| TargetView {
                track_id: t.track_id,
                label: t.label.clone(),
                rel_bearing_deg: t.rel_bearing_deg,
                depth_range_m: t.depth_range_m,
                depth_scale: scale,
                range_true_m: frame.range_true_m,
                ..Default::default()
            }),
            locate_err_px: frame.locate_err_px,
            track_iou_truth: frame.track_iou_truth,
            features: feats,
            flags: fl,
            pyramid: frame.pyramid,
        };
        let ov = overlay(&frame, kf.intr.width, kf.intr.height, wanted);
        kf.ml_webp = encode_webp(&ov.data, ov.width, ov.height, 70.0).ok().map(|e| e.bytes);
        kf.ml = Some(view);
        kf.sds.extend(sds_lane);
        Ok(MlOut { frame, sighting })
    }

    pub(crate) fn record_keyframe_row(&mut self, pass: &str, kf: &Keyframe, target: &str) {
        let ml = kf.ml.as_ref();
        let t = ml.and_then(|m| m.target.as_ref());
        let d = kf.decision.as_ref();
        let row = KeyframeRow {
            run_id: self.run_id.clone(),
            pass: pass.into(),
            index: kf.index,
            tick: kf.tick,
            time_s: kf.time_s,
            target: target.into(),
            depth_mode: ml.map(|m| m.depth_mode.clone()).unwrap_or_default(),
            depth_absrel: ml.and_then(|m| m.depth_metrics.map(|x| x.absrel as f64)),
            depth_delta1: ml.and_then(|m| m.depth_metrics.map(|x| x.delta1 as f64)),
            locate_mode: ml.map(|m| m.locate_mode.clone()).unwrap_or_default(),
            locate_score: ml.and_then(|m| m.points.iter().find(|p| p.label == target).map(|p| p.score as f64)),
            locate_err_px: ml.and_then(|m| m.locate_err_px.map(|x| x as f64)),
            track_mode: ml.map(|m| m.track_mode.clone()).unwrap_or_default(),
            track_id: t.map(|t| t.track_id),
            track_iou_prev: ml.and_then(|m| t.and_then(|t| m.tracks.iter().find(|x| x.id == t.track_id)).and_then(|x| x.iou_prev.map(|v| v as f64))),
            track_iou_truth: ml.and_then(|m| m.track_iou_truth.map(|x| x as f64)),
            tri_range_m: t.and_then(|t| t.tri_range_m.map(|x| x as f64)),
            tri_parallax_deg: t.and_then(|t| t.tri_parallax_deg.map(|x| x as f64)),
            depth_scale: t.and_then(|t| t.depth_scale.map(|x| x as f64)),
            range_used_m: t.and_then(|t| t.range_used_m.map(|x| x as f64)),
            range_true_m: t.and_then(|t| t.range_true_m.map(|x| x as f64)),
            decision: d.map(|d| d.decision.clone()).unwrap_or_default(),
            ts_method: d.map(|d| d.forecast_method.clone()).unwrap_or_default(),
            ts_residual: d.and_then(|d| d.residual_deg.map(|x| x as f64)),
            fusion_flags: ml.map(|m| m.flags.join(",")).unwrap_or_default(),
            balance_margin: kf.balance.as_ref().map(|b| b.margin as f64),
            effort_max: kf.balance.as_ref().map(|b| b.worst_ratio as f64),
        };
        self.db.record_keyframe(row);
    }

    fn record_effects(&mut self, post: &noto_core::Posture, state: &str) {
        use noto_core::effects::{evaluate, live_scenario, Rule};
        let Some(s) = live_scenario(&self.world.body, post) else { return };
        let fx = evaluate(&self.world.body, &s);
        if fx.skipped.is_some() {
            return;
        }
        let at = format!("{}|{state}", post.ground.id());
        for c in &fx.cells {
            let Some(v) = c.value else { continue };
            let scope = Scope::new(Track::Balance, &format!("fx:{}", c.effect.id()), &at);
            match c.rule {
                Rule::Like { reference, tolerance } | Rule::AtLeast { reference, tolerance } => self.sds.record_reference(&scope, c.axis, v as f64, reference as f64, tolerance as f64),
                Rule::AtMost { limit } => {
                    self.sds.record_baseline(&scope, c.axis, v as f64);
                    self.sds.record_agreement(&scope, &format!("{}_ok", c.axis), v <= limit);
                }
                Rule::Raw => self.sds.record_baseline(&scope, c.axis, v as f64),
            }
        }
        let scope = Scope::new(Track::Effort, "wear", state);
        for (joint, d) in self.world.wear.clone() {
            self.sds.record_baseline(&scope, &format!("{joint}:life_used"), d as f64);
        }
    }

    pub(crate) fn record_balance_events(&mut self) {
        let events = self.world.take_balance_events();
        if events.is_empty() {
            return;
        }
        let preset = self.world.cfg.body.id();
        let body_kg = self.world.body.mass_kg();
        let nn_lib = noto_core::nn::library(&self.world.body);
        for ev in &events {
            let Some(d) = &ev.dynamics else { continue };
            self.record_effects(&ev.posture, &ev.state);
            for (spec, phase) in noto_core::nn::matching_phases(&nn_lib, &ev.posture) {
                let cells = noto_core::nn::cells(&self.world.body, &ev.posture, &ev.report, Some(d));
                let pr = noto_core::nn::score_phase(phase, &spec.seg_w, cells, &ev.report, Some(d));
                let scope = Scope::new(Track::Balance, &format!("nn:{}", spec.id), &phase.id);
                self.sds.record_baseline(&scope, "score", pr.score as f64);
                self.sds.record_agreement(&scope, "item_ok", pr.item >= 1);
                for c in pr.cells.iter().filter(|c| c.z.is_some()) {
                    let axis = format!("{}>{}:{}", c.source.id(), c.target.id(), c.axis);
                    self.sds.record_reference(&scope, &axis, c.value as f64, c.reference.unwrap_or(0.0) as f64, c.tolerance.unwrap_or(1.0) as f64);
                }
                self.nn_live.record(spec.id, spec.name_ko, &pr);
            }
        }
        for ev in events {
            let r = &ev.report;
            if r.source == Source::Quiet {
                continue;
            }
            let src = r.source.id();
            let ff_scope = Scope::new(Track::Balance, &format!("{src}>foot"), &ev.state);
            if r.ff_predicted_m.abs() > 1e-4 && r.shift_uncomp_m.abs() > 1e-4 {
                self.sds.record_calibration(&ff_scope, "ff", r.shift_uncomp_m as f64, r.ff_predicted_m as f64);
            }
            let ctx = r.ref_ctx(body_kg);
            for c in &r.couplings {
                let scope = Scope::new(Track::Balance, &format!("{src}>{}", c.target.id()), &ev.state);
                let rf = reference(r.source, c.target, &c.axis, &ctx);
                match &rf {
                    Some(rf) => self.sds.record_reference(&scope, &c.axis, c.value as f64, rf.value as f64, rf.tolerance as f64),
                    None => self.sds.record_baseline(&scope, &c.axis, c.value as f64),
                }
                self.db.record_balance(BalanceRow {
                    run_id: self.run_id.clone(),
                    pass: self.pass.clone(),
                    tick: ev.tick,
                    source: src.into(),
                    source_seg: r.source.segment().id().into(),
                    target: c.target.id().into(),
                    state: format!("{}|{}", ev.state, c.axis),
                    preset: preset.clone(),
                    comp: c.value as f64,
                    comp_unit: c.unit.clone(),
                    reference: rf.as_ref().map(|x| x.value as f64),
                    tolerance: rf.as_ref().map(|x| x.tolerance as f64),
                    torque_nm: c.torque_nm.unwrap_or(0.0) as f64,
                    effort: c.effort.unwrap_or(0.0) as f64,
                    effort_noff: c.effort_noff.unwrap_or(0.0) as f64,
                    margin: r.margin as f64,
                    margin_noff: r.margin_noff as f64,
                    ff_gain: r.ff_gain as f64,
                });
            }
            for j in &r.joints {
                let scope = Scope::new(Track::Effort, j.segment.id(), &format!("{src}|{}", ev.state));
                self.sds.record_baseline(&scope, &format!("{}:peak_ratio", j.joint), j.peak_ratio as f64);
                self.sds.record_baseline(&scope, &format!("{}:effort", j.joint), j.effort as f64);
            }
            if let Some(d) = &ev.dynamics {
                let scope = Scope::new(Track::Effort, "dynamics", &format!("{src}|{}", ev.state));
                self.sds.record_baseline(&scope, "min_margin", d.min_margin as f64);
                self.sds.record_baseline(&scope, "margin_dip", (d.static_margin - d.min_margin) as f64);
                self.sds.record_baseline(&scope, "peak_ratio", d.peak_ratio as f64);
                self.sds.record_agreement(&scope, "feasible", d.feasible);
                self.sds.record_agreement(&scope, "reached", d.reached);
                for j in d.joints.iter().filter(|j| j.kind == "sim") {
                    self.sds.record_baseline(&scope, &format!("{}:peak_ratio", j.name), j.peak_ratio as f64);
                    self.sds.record_baseline(&scope, &format!("{}:lag_deg", j.name), j.max_lag_deg as f64);
                    if j.saturated_s > 0.0 || j.stalled_s > 0.0 {
                        self.sds.record_baseline(&scope, &format!("{}:sat_s", j.name), j.saturated_s as f64);
                    }
                }
            }
        }
        for s in Source::ALL {
            let scope = Scope::new(Track::Balance, &format!("{}>foot", s.id()), "");
            if let Some((f, _, _, key)) = self.sds.calibration_factor(&scope, "ff") {
                if key.starts_with(&format!("balance|{}>foot|", s.id())) {
                    self.world.ff_gains.insert(s, f as f32);
                }
            }
        }
    }

    pub(crate) fn record_quest_run(&mut self, quest_id: &str, attempts: &[u32]) {
        let Some(q) = self.quests.get(quest_id).cloned() else { return };
        let kind = match &q.plan {
            noto_search::Plan::Fetch { .. } => "fetch",
            noto_search::Plan::Collect { .. } => "collect",
            noto_search::Plan::Goto { .. } => "goto",
            noto_search::Plan::Drop { .. } => "drop",
            noto_search::Plan::Stop => "stop",
        };
        let (target, destination) = match &q.plan {
            noto_search::Plan::Fetch { targets, destination } => (targets.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), Some(destination.place.clone())),
            noto_search::Plan::Collect { targets } => (targets.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), None),
            noto_search::Plan::Goto { place } => (place.place.clone(), Some(place.place.clone())),
            noto_search::Plan::Drop { items, place } => (items.iter().map(|t| t.resource.clone()).collect::<Vec<_>>().join("+"), place.as_ref().map(|p| p.place.clone())),
            noto_search::Plan::Stop => (String::new(), None),
        };
        let ok = q.state == noto_quest::State::Done;
        let steps: Vec<StepRecord> = q
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state != noto_quest::State::Pending)
            .map(|(i, s)| StepRecord {
                step: format!("{:?}", s.kind).to_lowercase(),
                attempts: attempts.get(i).copied().unwrap_or(1).max(1),
                ok: s.state == noto_quest::State::Done,
                fail_code: (s.state == noto_quest::State::Failed).then(|| q.failure.as_ref().map(|f| f.code.clone()).unwrap_or_default()),
                time_s: 0.0,
            })
            .collect();
        let r = q.result.clone();
        let run = QuestRun {
            run_id: self.run_id.clone(),
            session: self.session_id.clone(),
            pass: self.pass.clone(),
            quest_id: q.id.clone(),
            kind: kind.into(),
            target: target.clone(),
            destination,
            preset: self.world.cfg.body.id(),
            seed: self.cfg.seed,
            ok,
            failure: q.failure.as_ref().map(|f| f.code.clone()),
            ticks: r.as_ref().map_or(0, |r| r.ticks),
            time_s: r.as_ref().map_or(0.0, |r| r.time_s),
            energy_j: r.as_ref().map_or(0.0, |r| r.energy_j),
            steps: steps.clone(),
            created_ms: noto_sds::store::now_ms(),
        };
        for s in &steps {
            let scope = Scope::new(Track::Quest, &format!("{}|{kind}", self.pass), &s.step);
            for i in 0..s.attempts.max(1) {
                self.sds.record_agreement(&scope, "success", s.ok && i + 1 == s.attempts.max(1));
            }
        }
        self.sds.record_agreement(&Scope::new(Track::Quest, &format!("{}|{kind}", self.pass), "quest"), "success", ok);
        let stat = {
            self.db.record_run(run);
            self.db.stat(&self.pass, kind, &target, &self.world.cfg.body.id(), noto_sds::WHOLE_QUEST).cloned()
        };
        if let Some(st) = stat {
            let msg = format!("[{}] {kind} {target}: 성공 {}/{} ({:.0}%) · 프리셋 {}", self.pass, st.successes, st.attempts, st.rate * 100.0, st.preset);
            self.log("quest", Level::Info, msg, Some(json!({"pass": self.pass, "kind": kind, "target": target, "attempts": st.attempts, "successes": st.successes})));
        }
        for l in self.db.flush() {
            self.audit.info("sds", format!("[DB] {l}"));
        }
    }
}
