//! JSON calls of the integrated test and the part presets (see `api.rs` for
//! the conventions). Kept apart from the ten world calls of v0.2.

use hand_core::balance::{reference, Body, FeedForward, Posture, RefCtx, Source};
use hand_core::parts::{ArmPreset, BellyPreset, ChassisPreset, ChestPreset, HeadPreset, LegPreset, Segment, Tier};
use hand_core::{BaseKind, BodyPlan, FingerCount};
use serde_json::{json, Value};
use world_sim::ItemKind;

use crate::runtime::Runtime;

fn r3(x: f32) -> f64 {
    let v = ((x as f64) * 1000.0).round() / 1000.0;
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

/// A body plan from `{body}` (full object) or `{preset}` (a named plan).
pub fn plan_from(args: &Value) -> Result<Option<BodyPlan>, String> {
    if let Some(b) = args.get("body").filter(|b| !b.is_null()) {
        return serde_json::from_value::<BodyPlan>(b.clone()).map(Some).map_err(|e| format!("body(BodyPlan)를 읽을 수 없습니다: {e}"));
    }
    if let Some(p) = args.get("preset").and_then(Value::as_str).filter(|p| !p.is_empty()) {
        return BodyPlan::named(p).map(Some).ok_or_else(|| format!("알 수 없는 프리셋 '{p}' (가능: {})", BodyPlan::NAMED.join(", ")));
    }
    Ok(None)
}

fn base_from(args: &Value) -> Result<Option<BaseKind>, String> {
    match args.get("base").and_then(Value::as_str).filter(|b| !b.is_empty()) {
        Some(b) => serde_json::from_value::<BaseKind>(Value::String(b.into())).map(Some).map_err(|_| format!("알 수 없는 베이스 '{b}'")),
        None => Ok(None),
    }
}

/// Every preset of every segment, for the preset panel.
pub fn presets_json() -> Value {
    let act = |a: &hand_core::parts::Actuator| json!({"name": a.name, "peak_nm": a.peak_nm, "cont_nm": a.cont_nm, "speed_rad_s": a.speed_rad_s, "mass_kg": a.mass_kg, "price_usd": a.price_usd, "backdrivable": a.backdrivable, "src": a.src});
    let tiers: Vec<Value> = Tier::ALL.iter().map(|t| json!({"id": t.id(), "label_ko": t.label_ko()})).collect();
    let leg_tiers: Vec<Value> = Tier::LEGS.iter().map(|t| json!({"id": t.id(), "label_ko": t.label_ko()})).collect();
    json!({
        "tiers": tiers, "leg_tiers": leg_tiers, "named": BodyPlan::NAMED,
        "segments": {
            "head": Tier::ALL.iter().map(|t| { let p = HeadPreset::of(*t); json!({"tier": t.id(), "mass_kg": p.mass_kg, "sensor": p.sensor, "neck": act(&p.neck), "gaze_stabilization": p.gaze_stabilization, "power_w": p.power_w, "cost_usd": p.cost_usd}) }).collect::<Vec<_>>(),
            "chest": Tier::ALL.iter().map(|t| { let p = ChestPreset::of(*t); json!({"tier": t.id(), "mass_kg": p.mass_kg, "battery_kg": p.battery_kg, "battery_wh": p.battery_wh, "computer": p.computer, "power_w": p.power_w, "cost_usd": p.cost_usd}) }).collect::<Vec<_>>(),
            "belly": Tier::ALL.iter().map(|t| { let p = BellyPreset::of(*t); json!({"tier": t.id(), "mass_kg": p.mass_kg, "waist": act(&p.waist), "pitch": p.pitch.as_ref().map(act), "max_lean_deg": p.max_lean_deg, "cost_usd": p.cost_usd}) }).collect::<Vec<_>>(),
            "arm": Tier::ALL.iter().map(|t| { let p = ArmPreset::of(*t); json!({"tier": t.id(), "mass_kg": p.mass_kg, "com_m": p.com_m, "shoulder": act(&p.shoulder), "elbow": act(&p.elbow), "power_w": p.power_w, "cost_usd": p.cost_usd}) }).collect::<Vec<_>>(),
            "leg": Tier::LEGS.iter().map(|t| { let p = LegPreset::of(*t); json!({"tier": t.id(), "mass_kg": p.mass_kg, "knee": act(&p.knee), "ankle": act(&p.ankle), "froude": p.froude, "foot": [p.foot_len_m, p.foot_width_m], "cost_usd": p.cost_usd}) }).collect::<Vec<_>>(),
            "chassis": Tier::ALL.iter().map(|t| { let p = ChassisPreset::of(*t); json!({"tier": t.id(), "name": p.name, "mass_kg": p.mass_kg, "max_speed_mps": p.max_speed_mps, "drive_force_n": p.drive_force_n, "cost_usd": p.cost_usd, "src": p.src}) }).collect::<Vec<_>>(),
        },
        "sources": Source::ALL.iter().map(|s| json!({"id": s.id(), "label_ko": s.label_ko(), "segment": s.segment().id()})).collect::<Vec<_>>(),
        "segments_order": Segment::ALL.iter().map(|s| json!({"id": s.id(), "label_ko": s.label_ko()})).collect::<Vec<_>>(),
    })
}

/// What a plan means physically: mass, CoM, limits empty and loaded, which
/// items it can lift (one or two arms), and the N:N cells of typical states
/// next to their references.
pub fn plan_eval_json(plan: &BodyPlan, base: BaseKind, fingers: FingerCount) -> Value {
    let body = Body::new(plan, base, fingers);
    let spec = base.spec();
    let limits = |kg: f32| {
        let l = body.drive_limits(&spec, kg);
        json!({"max_speed_mps": r3(l.max_speed_mps), "max_slope_deg": r3(l.max_slope_deg), "max_step_m": r3(l.max_step_m), "capacity_kg": r3(l.capacity_kg), "accel_mps2": r3(l.accel_limit_mps2), "power_scale": r3(l.power_scale), "reasons": l.reasons})
    };
    let ff = FeedForward::default();
    let lifts: Vec<Value> = ItemKind::ALL
        .iter()
        .map(|k| {
            let s = k.spec();
            let (rep, two) = body.lift(s.mass_kg, 0.0, &ff, 1e4);
            let worst = rep.worst();
            json!({"item": s.id, "label_ko": s.label_ko, "mass_kg": s.mass_kg, "ok": rep.feasible, "two_arms": two, "limiting": rep.limiting, "worst": worst.map(|w| w.joint.clone()), "worst_ratio": worst.map(|w| r3(w.peak_ratio)), "margin": r3(rep.margin)})
        })
        .collect();
    let m = body.mass_kg();
    let states: Vec<(Posture, &str)> = vec![
        (Posture::arm_raise(0.0), "flat"),
        (Posture::arm_lift(ItemKind::Log.spec().mass_kg, false, 0.0), "flat"),
        (Posture { source: Source::FootLift, single_support: true, ..Posture::quiet(0.0) }, "flat"),
        // The step this base can actually climb (a wheel never meets a 10 cm curb).
        (Posture { source: Source::StepUp, step_m: base.spec().max_step_m.min(0.10), single_support: base == BaseKind::HumanFoot, ..Posture::quiet(0.0) }, "step"),
        (Posture { source: Source::Slope, slope_deg: 10.0, ..Posture::quiet(0.0) }, "up_hi"),
        (Posture::quiet(4.0), "flat"),
        (Posture { source: Source::Accel, accel_mps2: 1.5, ..Posture::quiet(0.0) }, "flat"),
    ];
    let cells: Vec<Value> = states
        .iter()
        .filter(|(p, _)| p.source != Source::FootLift || base == BaseKind::HumanFoot)
        .flat_map(|(p, state)| {
            let rep = body.solve(p, &ff, 1e4);
            let ctx: RefCtx = rep.ref_ctx(m);
            rep.couplings
                .iter()
                .map(|c| {
                    let rf = reference(rep.source, c.target, &c.axis, &ctx);
                    json!({
                        "source": rep.source.id(), "target": c.target.id(), "axis": c.axis, "state": state,
                        "value": r3(c.value), "value_noff": r3(c.value_noff), "effort": c.effort.map(r3), "effort_noff": c.effort_noff.map(r3),
                        "reference": rf.as_ref().map(|r| r3(r.value)), "tolerance": rf.as_ref().map(|r| r3(r.tolerance)), "cite": rf.as_ref().map(|r| r.cite),
                        "dev": rf.as_ref().map(|r| r3((c.value - r.value) / r.tolerance)),
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let quiet = body.solve(&Posture::quiet(0.0), &ff, 1e4);
    json!({
        "plan": plan, "id": plan.id(), "base": base,
        "mass_kg": r3(m), "com_height_m": r3(body.com_height_m()), "battery_wh": body.chest.battery_wh, "cost_usd": r3(plan.cost_usd(base)),
        "support": [r3(body.support_len_m), r3(body.support_width_m)],
        "limits_empty": limits(0.0), "limits_loaded": limits(8.0),
        "lifts": lifts,
        "quiet": {"margin": r3(quiet.margin), "joints": quiet.joints},
        "nn": cells,
    })
}

impl Runtime {
    pub(crate) fn api_test_call(&mut self, cmd: &str, args: &Value) -> Option<Result<Value, String>> {
        let num = |k: &str| args.get(k).and_then(Value::as_f64);
        let text = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("");
        Some(match cmd {
            "test_start" => (|| {
                let command = text("command");
                if command.trim().is_empty() {
                    return Err("command 가 필요합니다".to_string());
                }
                let seed = num("seed").map(|s| s.max(0.0) as u64).unwrap_or(self.cfg.seed);
                let body = plan_from(args)?.unwrap_or(self.cfg.body);
                let base = base_from(args)?;
                self.test_start(command, seed, body, base).map_err(|e| e.to_string())?;
                self.api_test_status()
            })(),
            "test_step" => (|| {
                let ticks = num("ticks").unwrap_or(40.0).clamp(1.0, 600.0) as u32;
                self.test_advance(ticks).map_err(|e| e.to_string())?;
                self.api_test_status()
            })(),
            "test_status" => self.api_test_status(),
            "test_keyframe" => self.api_test_keyframe(if text("pass").is_empty() { "game" } else { text("pass") }, num("index").unwrap_or(0.0).max(0.0) as usize),
            "test_video" => self.api_test_video(if text("pass").is_empty() { "game" } else { text("pass") }, num("from").unwrap_or(0.0).max(0.0) as usize, num("count").unwrap_or(30.0) as usize),
            "test_stats" => self.api_test_stats(),
            "test_cancel" => {
                self.cancel();
                self.test = None;
                self.recording = None;
                self.pass = "game".into();
                Ok(json!({"active": false}))
            }
            "body_presets" => Ok(json!({"presets": presets_json(), "current": self.cfg.body, "current_id": self.cfg.body.id()})),
            "body_plan_eval" => (|| {
                let plan = plan_from(args)?.unwrap_or(self.cfg.body);
                let base = base_from(args)?.unwrap_or(self.cfg.hand.base);
                Ok(plan_eval_json(&plan, base, self.cfg.hand.fingers))
            })(),
            "body_plan_set" => (|| {
                let plan = plan_from(args)?.ok_or("body 또는 preset 이 필요합니다")?;
                self.cfg.body = plan;
                self.reset_with(None, None, None).map_err(|e| e.to_string())?;
                Ok(plan_eval_json(&plan, self.cfg.hand.base, self.cfg.hand.fingers))
            })(),
            _ => return None,
        })
    }
}
