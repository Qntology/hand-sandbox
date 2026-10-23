//! The calls the UI makes, as JSON in and JSON out.
//!
//! `ui/js/api.js` talks to a backend that answers ten commands. The browser
//! mock (`ui/js/mock-backend.js`) was written first and fixed the shapes; this
//! module answers with the same shapes from the real [`Runtime`]. The Tauri
//! commands and the development server (`examples/serve.rs`) are thin wrappers
//! around it, so what the UI sees does not depend on which one runs.
//!
//! Two conventions differ between the simulator and the UI, and are converted
//! here and nowhere else:
//!
//! * The simulator has `x` and `z` on the ground and `y` up. The UI draws a map
//!   with `x` to the right and `y` up, so a position goes out as
//!   `[x, depth - z, height]`. The simulator's right-hand side is `+z`, which
//!   would mirror the map, hence the flip.
//! * The same flip turns the yaw: `yaw_ui = -yaw_sim`.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use hand_core::{BaseKind, HandConfig};
use hand_search::{Category, ParsedCommand, Plan, Via};
use serde_json::{json, Value};
use world_sim::snapshot::{CarryView, Pos3};
use world_sim::WorldSnapshot;

use hand_sds::LogLine;

use crate::precision::Judged;
use crate::runtime::{label_ko, FrameView, Runtime, Submission};

/// Smallest and largest world the UI may ask for, in cells of 0.5 m.
pub const MIN_CELLS: usize = 32;
pub const MAX_CELLS: usize = 160;

/// Names of the commands [`Runtime::api_call`] answers.
pub const COMMANDS: [&str; 11] = [
    "world_new",
    "sim_step",
    "world_frame",
    "command_submit",
    "quest_list",
    "perception_scan",
    "body_status",
    "sds_status",
    "ts_forecast",
    "log_tail",
    "models_status",
];

/// How many steps ahead the time-series tab forecasts.
const FORECAST_HORIZON: usize = 16;

fn r3(x: f32) -> f64 {
    let v = ((x as f64) * 1000.0).round() / 1000.0;
    // no "-0.0" in the JSON
    if v == 0.0 { 0.0 } else { v }
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// Calendar date and time of a Unix time in milliseconds, as ISO 8601 in UTC.
/// The logs keep integers; the UI wants a string it can cut the clock out of.
pub fn iso_ms(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let milli = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    // Days since 1970-01-01 to a civil date (H. Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{milli:03}Z", sod / 3600, sod % 3600 / 60, sod % 60)
}

fn wrap_deg(a: f32) -> f32 {
    let mut a = a % 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a <= -180.0 {
        a += 360.0;
    }
    a
}

fn pos(p: &Pos3, depth_m: f32) -> Value {
    json!([r3(p.x), r3(depth_m - p.z), r3(p.y)])
}

/// `mass_kg` in a carry entry is the mass of one item, as the UI reads it; the
/// simulator reports the mass of the whole stack.
fn carry(c: &CarryView) -> Value {
    json!({"item_id": c.item.id(), "label_ko": c.label_ko, "count": c.count, "mass_kg": r3(c.mass_kg / c.count.max(1) as f32)})
}

pub fn snapshot_json(s: &WorldSnapshot) -> Value {
    let d = s.bounds.depth_m;
    json!({
        "tick": s.tick,
        "time_s": r3(s.time_s as f32),
        "seed": s.seed,
        "bounds": [r3(s.bounds.width_m), r3(s.bounds.depth_m)],
        "agent": {
            "pos": pos(&s.agent.pos, d),
            "yaw_deg": r3(wrap_deg(-s.agent.yaw_deg)),
            "speed_mps": r3(s.agent.speed_mps),
            "base": s.agent.base,
            "fingers": s.agent.fingers,
            "carrying": s.agent.carrying.iter().map(carry).collect::<Vec<_>>(),
            "carried_kg": r3(s.agent.carried_kg),
            "capacity_kg": r3(s.agent.capacity_kg),
            "battery_ratio": r3(s.agent.battery_ratio),
            "action": s.agent.action,
        },
        "waypoints": s.waypoints.iter().map(|w| json!({
            "id": w.key,
            "node_id": w.id,
            "label_ko": w.label_ko,
            "pos": pos(&w.pos, d),
            "stock": w.stock.iter().map(carry).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "nodes": s.nodes.iter().map(|n| json!({
            "id": n.id, "kind": n.kind.id(), "label_ko": n.label_ko, "pos": pos(&n.pos, d), "hp": r3(n.hp),
        })).collect::<Vec<_>>(),
        "drops": s.drops.iter().map(|x| json!({
            "id": x.id, "item_id": x.item.id(), "label_ko": x.label_ko, "pos": pos(&x.pos, d),
        })).collect::<Vec<_>>(),
    })
}

pub fn log_json(l: &LogLine) -> Value {
    let mut v = json!({"seq": l.seq, "ts": iso_ms(l.ts), "cat": l.cat, "level": l.level, "msg": l.msg});
    if let Some(d) = &l.data {
        v["data"] = d.clone();
    }
    v
}

fn role_ko(c: Category) -> &'static str {
    match c {
        Category::Resource => "아이템",
        Category::Place => "장소",
        Category::Negation => "부정",
        Category::Direction => "도착 표지",
        Category::Source => "출발 표지",
        Category::Verb => "동작",
        Category::Filler => "군더더기",
    }
}

fn stage_of(v: Via) -> &'static str {
    match v {
        Via::Exact => "lexical",
        Via::Tail => "tail",
        Via::Prefix => "prefix",
        Via::Embedding => "embedding",
    }
}

/// What the UI's command line shows about a parsed command.
fn parsed_json(rt: &Runtime, p: &ParsedCommand) -> Value {
    let w = rt.world();
    let place = |id: &str| json!({"kind": "waypoint", "id": id, "label_ko": label_ko(w, id)});
    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
    let targets_json = |targets: &[hand_search::Target]| {
        targets.iter().map(|t| json!({"item_id": t.resource, "label_ko": label_ko(w, &t.resource), "count": t.quantity, "score": r3(t.score)})).collect::<Vec<_>>()
    };
    let (action, item_id, item_ko, count, dest, conf, targets) = match &p.plan {
        Some(Plan::Fetch { targets, destination }) => {
            let scores: Vec<f32> = targets.iter().map(|t| t.score).chain(std::iter::once(destination.score)).collect();
            summary_of(targets, "mine_and_deliver", Some(place(&destination.place)), mean(&scores), w, targets_json(targets))
        }
        Some(Plan::Collect { targets }) => {
            let scores: Vec<f32> = targets.iter().map(|t| t.score).collect();
            summary_of(targets, "mine", None, mean(&scores), w, targets_json(targets))
        }
        Some(Plan::Goto { place: pl }) => ("goto", None, None, None, Some(place(&pl.place)), pl.score, Vec::new()),
        Some(Plan::Stop) => ("stop", None, None, None, None, 1.0, Vec::new()),
        None => ("unknown", None, None, None, None, 0.0, Vec::new()),
    };
    json!({
        "action": action,
        "item_id": item_id,
        "item_label_ko": item_ko,
        "count": count,
        "destination": dest,
        "confidence": r3(conf),
        "targets": targets,
    })
}

type Summary = (&'static str, Option<String>, Option<String>, Option<u32>, Option<Value>, f32, Vec<Value>);

fn summary_of(targets: &[hand_search::Target], action: &'static str, dest: Option<Value>, conf: f32, w: &world_sim::World, tj: Vec<Value>) -> Summary {
    let first = targets.first();
    let one = targets.len() == 1;
    let item_ko = if one {
        first.map(|t| label_ko(w, &t.resource))
    } else {
        Some(targets.iter().map(|t| format!("{} {}개", label_ko(w, &t.resource), t.quantity)).collect::<Vec<_>>().join(" + "))
    };
    (action, first.map(|t| t.resource.clone()), item_ko, if one { first.map(|t| t.quantity) } else { None }, dest, conf, tj)
}

/// The steps of the parse as table rows for the analysis tab.
fn trace_json(p: &ParsedCommand, conf: f64) -> Vec<Value> {
    let mut rows = vec![json!({"stage": "normalize", "text": p.normalized, "score": null, "role": null, "accepted": true})];
    for h in &p.hits {
        let ruled_out = p.excluded.iter().any(|e| *e == h.text);
        rows.push(json!({"stage": stage_of(h.via), "text": h.text, "score": r3(h.score), "role": role_ko(h.category), "accepted": !ruled_out}));
    }
    for (value, start, end) in &p.quantities {
        let text = p.tokens.get(*start..*end).map(|t| t.join(" ")).unwrap_or_default();
        rows.push(json!({"stage": "quantity", "text": format!("{text} → {value}"), "score": 1.0, "role": "수량", "accepted": true}));
    }
    for w in &p.unmatched {
        rows.push(json!({"stage": "lexical", "text": w, "score": 0.0, "role": null, "accepted": false}));
    }
    for t in &p.trace {
        rows.push(json!({"stage": t.stage, "text": t.detail, "score": null, "role": null, "accepted": true}));
    }
    for w in &p.warnings {
        rows.push(json!({"stage": "warning", "text": w, "score": null, "role": null, "accepted": false}));
    }
    let end = match (&p.plan, &p.problem) {
        (Some(_), _) => json!({"stage": "assign", "text": "역할 배정 완료", "score": conf, "role": null, "accepted": true}),
        (None, Some(pr)) => json!({"stage": "assign", "text": pr.message, "score": 0.0, "role": null, "accepted": false}),
        (None, None) => json!({"stage": "assign", "text": "해석하지 못함", "score": 0.0, "role": null, "accepted": false}),
    };
    rows.push(end);
    rows
}

fn submission_json(rt: &Runtime, text: &str, s: &Submission) -> Value {
    let parsed = parsed_json(rt, &s.parsed);
    let conf = parsed["confidence"].as_f64().unwrap_or(0.0);
    json!({
        "ok": s.ok,
        "stopped": s.stopped,
        "parsed": parsed,
        "quest_id": s.quest_id,
        "trace": trace_json(&s.parsed, conf),
        "error": s.error,
        "error_code": s.error_code,
        "mode": s.mode,
        "text": text,
    })
}

/// One row per label of a scan: the best candidate of the label.
fn scan_labels(rt: &Runtime, judged: &[Judged], top_k: usize) -> Vec<Value> {
    use std::collections::BTreeMap;
    let mut best: BTreeMap<&str, (&Judged, u32, bool)> = BTreeMap::new();
    for j in judged {
        let e = best.entry(j.cand.label.as_str()).or_insert((j, 0, false));
        e.1 += 1;
        e.2 |= j.accepted;
        if j.cand.score > e.0.cand.score {
            e.0 = j;
        }
    }
    let mut rows: Vec<(&str, (&Judged, u32, bool))> = best.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.cand.score.total_cmp(&a.1 .0.cand.score));
    rows.into_iter()
        .take(top_k.max(1))
        .map(|(label, (j, n, any_ok))| {
            json!({
                "id": label,
                "label_ko": label_ko(rt.world(), label),
                "score": r3(j.cand.score),
                "margin": r3(j.cand.margin),
                "accepted": any_ok,
                "reason": if any_ok && !j.accepted { "다른 개체가 수락됨".to_string() } else { j.reason_ko() },
                "count": n,
                "blocked_by": j.blocked_by(),
                "floor": r3(j.floor),
                "truth": j.cand.truth_label,
            })
        })
        .collect()
}

fn num(args: &Value, key: &str) -> Option<f64> {
    args.get(key).and_then(Value::as_f64)
}

fn text<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

impl Runtime {
    // ----- the ten commands -----

    /// `world_new({seed, size, config, base})`: a fresh world, and its snapshot.
    pub fn api_world_new(&mut self, seed: u64, size: usize, config: Option<HandConfig>, base: Option<&str>) -> Result<Value, String> {
        if !(MIN_CELLS..=MAX_CELLS).contains(&size) {
            return Err(format!("size 는 {MIN_CELLS}~{MAX_CELLS} 셀이어야 합니다 (받은 값 {size})"));
        }
        let mut hand = config;
        if let Some(b) = base.filter(|b| !b.is_empty()) {
            let kind: BaseKind = serde_json::from_value(Value::String(b.to_string())).map_err(|_| format!("알 수 없는 베이스 '{b}'"))?;
            let mut h = hand.unwrap_or(self.cfg.hand);
            h.base = kind;
            hand = Some(h);
        }
        self.reset_with(Some(seed), Some(size), hand).map_err(|e| e.to_string())?;
        Ok(snapshot_json(&self.snapshot()))
    }

    /// `sim_step({ticks})`: runs ticks, answers the new state and the notable log lines.
    pub fn api_sim_step(&mut self, ticks: u32) -> Result<Value, String> {
        let lines = self.step(ticks.clamp(1, 200));
        let notable: Vec<&LogLine> = lines.iter().filter(|l| l.level != hand_sds::Level::Debug).collect();
        let events: Vec<Value> = notable[notable.len().saturating_sub(50)..].iter().map(|l| log_json(l)).collect();
        Ok(json!({
            "snapshot": snapshot_json(&self.snapshot()),
            "quests": self.api_quest_list()?,
            "body": self.api_body_status()?,
            "events": events,
        }))
    }

    /// `world_frame({width, height, quality, view})`: a WebP screenshot and the boxes of the last scan.
    pub fn api_world_frame(&mut self, width: u32, height: u32, quality: f32, view: &str) -> Result<Value, String> {
        let view = match view {
            "top" => FrameView::Top,
            _ => FrameView::Agent,
        };
        let f = self.frame_sized(view, width, height, quality).map_err(|e| e.to_string())?;
        let dets: Vec<Value> = f
            .detections
            .iter()
            .map(|d| {
                json!({
                    "label_id": d.label, "label_ko": d.label_ko, "score": r3(d.score), "margin": r3(d.margin),
                    "accepted": d.accepted, "blocked_by": d.blocked_by, "reason": d.reason,
                    "bbox": d.bbox.map(r3), "landmark": d.landmark,
                })
            })
            .collect();
        Ok(json!({
            "webp_base64": STANDARD.encode(&f.webp),
            "width": f.width, "height": f.height, "bytes": f.bytes,
            "encode_ms": r3(f.encode_ms), "lossless": f.lossless, "tick": f.tick,
            "detections": dets,
        }))
    }

    /// `command_submit({text})`: parses a typed command and queues its quest.
    pub fn api_command_submit(&mut self, input: &str) -> Result<Value, String> {
        let s = self.submit(input);
        Ok(submission_json(self, input, &s))
    }

    pub fn api_quest_list(&self) -> Result<Value, String> {
        serde_json::to_value(self.quests()).map_err(|e| e.to_string())
    }

    /// `perception_scan({topK})`: looks now and answers one row per label.
    pub fn api_perception_scan(&mut self, top_k: usize) -> Result<Value, String> {
        self.scan_now("manual").map_err(|e| e.to_string())?;
        let r = self.last_scan().ok_or("스캔 결과가 없습니다")?;
        Ok(json!({
            "labels": scan_labels(self, &r.judged, top_k),
            "tiles": r.tiles,
            "embed_ms": r3(r.embed_ms),
            "total_ms": r3(r.total_ms),
            "device": r.device,
            "mode": r.mode,
            "seq": r.seq,
            "tick": r.tick,
            "candidates": r.judged.len(),
        }))
    }

    pub fn api_body_status(&self) -> Result<Value, String> {
        serde_json::to_value(self.body_status()).map_err(|e| e.to_string())
    }

    /// `sds_status()`: the SDS tables plus what the precision gates have done.
    pub fn api_sds_status(&self) -> Result<Value, String> {
        let mut v = serde_json::to_value(self.sds_status()).map_err(|e| e.to_string())?;
        let g = self.gates();
        let landmarks = g.landmarks();
        // The pairs the time-series tab can chart: only these have a history.
        let forecastable: Vec<Value> = g
            .series_labels()
            .into_iter()
            .filter(|(_, n)| *n >= 3)
            .flat_map(|(l, n)| ["score", "margin"].map(|axis| json!({"scope": format!("vision|{l}|"), "axis": axis, "n": n})))
            .collect();
        v["forecastable"] = Value::Array(forecastable);
        v["precision"] = json!({
            "mode": self.recognizer_mode(),
            "gates_enabled": g.cfg().enabled,
            "evaluated": g.counters.evaluated,
            "accepted": g.counters.accepted,
            "blocked_by": g.counters.blocked_by,
            "landmarks": {
                "total": landmarks.len(),
                "confirmed": landmarks.iter().filter(|l| l.confirmed).count(),
                "rejected": landmarks.iter().filter(|l| l.state == crate::precision::LmState::Rejected).count(),
            },
            "eval": self.eval().summary(),
            "counts": self.eval(),
        });
        Ok(v)
    }

    /// `ts_forecast({scope, axis})`: history and forecast of a score or margin series.
    ///
    /// The SDS keeps running statistics, not histories. The only histories are
    /// the per-label score and margin series of the precision gates, so those
    /// are the axes that can be forecast; anything else answers with a reason.
    pub fn api_ts_forecast(&self, scope: &str, axis: &str) -> Result<Value, String> {
        let label = scope.split('|').nth(1).unwrap_or("").trim();
        if label.is_empty() {
            return Err(format!("범위 '{scope}'에는 라벨이 없어 예측할 수 없습니다"));
        }
        let (history, fc) = match axis {
            "score" | "score_all" => (self.gates().series(label).to_vec(), self.gates().forecast_for(label, FORECAST_HORIZON, self.sds())),
            "margin" => (self.gates().margin_series(label).to_vec(), self.gates().forecast_margin(label, FORECAST_HORIZON)),
            other => return Err(format!("축 '{other}'은 이력을 따로 두지 않습니다. 이력이 있는 축: score, margin")),
        };
        let Some(fc) = fc else {
            let known: Vec<String> = self.gates().series_labels().into_iter().take(6).map(|(l, n)| format!("{l}({n})")).collect();
            return Err(format!("'{label}'의 {axis} 이력이 아직 없습니다. 인식이 쌓이면 예측할 수 있습니다{}", if known.is_empty() { String::new() } else { format!(" (이력 있는 라벨: {})", known.join(", ")) }));
        };
        Ok(json!({
            "scope": scope, "axis": axis, "label": label,
            "history": history.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "point": fc.point.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "p10": fc.p10.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "p90": fc.p90.iter().map(|x| r4(*x)).collect::<Vec<_>>(),
            "method": fc.method,
            "weights": fc.weights.iter().map(|w| json!({"method": w.method, "weight": r4(w.weight), "rmse": r4(w.rmse)})).collect::<Vec<_>>(),
            "rmse_fused": r4(fc.rmse_fused),
            "holdout": fc.holdout,
            "notes": fc.notes,
        }))
    }

    /// `log_tail({afterSeq, limit, category})`.
    pub fn api_log_tail(&self, after_seq: u64, limit: usize, category: Option<&str>) -> Result<Value, String> {
        let cat = category.filter(|c| !c.is_empty());
        let (lines, next) = self.log_tail(after_seq, limit.clamp(1, 2000), cat);
        Ok(json!({"lines": lines.iter().map(log_json).collect::<Vec<_>>(), "next_seq": next}))
    }

    // ----- by name -----

    /// Answers a command by name with the JSON arguments the UI sends (camelCase keys).
    /// Which recogniser and which time-series model are in use right now. The
    /// host (Tauri) adds the model folders and whether their files exist.
    pub fn api_models_status(&self) -> Result<Value, String> {
        let mode = self.recognizer_mode();
        Ok(json!({
            "recognizer": mode,
            "embedding": mode != "oracle",
            "ttm": self.has_ttm(),
            "parser_embedding": self.has_embedder(),
        }))
    }

    pub fn api_call(&mut self, cmd: &str, args: &Value) -> Result<Value, String> {
        match cmd {
            "world_new" => {
                let seed = num(args, "seed").filter(|s| *s >= 0.0).ok_or("seed 는 0 이상의 숫자여야 합니다")? as u64;
                let size = num(args, "size").ok_or("size 가 필요합니다")?.max(0.0) as usize;
                let config = match args.get("config") {
                    Some(c) if !c.is_null() => Some(serde_json::from_value::<HandConfig>(c.clone()).map_err(|e| format!("config(HandConfig)를 읽을 수 없습니다: {e}"))?),
                    _ => None,
                };
                self.api_world_new(seed, size, config, text(args, "base"))
            }
            "sim_step" => self.api_sim_step(num(args, "ticks").unwrap_or(1.0).max(1.0) as u32),
            "world_frame" => {
                let (w, h) = (num(args, "width").unwrap_or(480.0) as u32, num(args, "height").unwrap_or(320.0) as u32);
                self.api_world_frame(w, h, num(args, "quality").unwrap_or(75.0) as f32, text(args, "view").unwrap_or("agent"))
            }
            "command_submit" => self.api_command_submit(text(args, "text").unwrap_or("")),
            "quest_list" => self.api_quest_list(),
            "perception_scan" => self.api_perception_scan(num(args, "topK").unwrap_or(8.0).max(1.0) as usize),
            "body_status" => self.api_body_status(),
            "sds_status" => self.api_sds_status(),
            "ts_forecast" => self.api_ts_forecast(text(args, "scope").unwrap_or(""), text(args, "axis").unwrap_or("")),
            "models_status" => self.api_models_status(),
            "log_tail" => self.api_log_tail(num(args, "afterSeq").unwrap_or(0.0).max(0.0) as u64, num(args, "limit").unwrap_or(200.0) as usize, text(args, "category")),
            other => Err(format!("unknown command {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use hand_core::FingerCount;
    use std::sync::Arc;

    fn lexicon() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file")
    }

    fn rt() -> Runtime {
        Runtime::new(RuntimeConfig::default(), &lexicon(), None).unwrap()
    }

    #[test]
    fn iso_time_matches_known_instants() {
        assert_eq!(iso_ms(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso_ms(1_700_000_000_123), "2023-11-14T22:13:20.123Z");
        assert_eq!(iso_ms(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(iso_ms(-1), "1969-12-31T23:59:59.999Z");
    }

    #[test]
    fn yaw_is_wrapped_and_flipped() {
        assert_eq!(wrap_deg(190.0), -170.0);
        assert_eq!(wrap_deg(-180.0), 180.0);
        assert_eq!(wrap_deg(-0.0), 0.0);
    }

    #[test]
    fn snapshot_has_the_shape_the_ui_reads() {
        let rt = rt();
        let v = snapshot_json(&rt.snapshot());
        assert_eq!(v["bounds"].as_array().unwrap().len(), 2);
        assert_eq!(v["agent"]["pos"].as_array().unwrap().len(), 3);
        assert!(v["agent"]["yaw_deg"].is_number());
        assert!(v["agent"]["carrying"].as_array().unwrap().is_empty());
        let w0 = &v["waypoints"][0];
        assert!(w0["id"].is_string() && w0["label_ko"].is_string() && w0["pos"].as_array().unwrap().len() == 3);
        let n0 = &v["nodes"][0];
        assert!(n0["kind"].is_string() && n0["hp"].is_number());
        let depth = v["bounds"][1].as_f64().unwrap();
        for n in v["nodes"].as_array().unwrap() {
            let y = n["pos"][1].as_f64().unwrap();
            assert!((0.0..=depth).contains(&y), "node inside the map after the flip: {y}");
        }
    }

    #[test]
    fn the_flip_keeps_the_map_unmirrored() {
        // A node on the robot's right-hand side (+z when facing +x) must be below the robot on the UI map (smaller y).
        let mut rt = rt();
        let (ax, az) = (rt.world().agent.x, rt.world().agent.z);
        rt.world_mut().agent.yaw = 0.0;
        let d = rt.snapshot().bounds.depth_m;
        let p = pos(&Pos3 { x: ax, y: 0.0, z: az + 2.0 }, d);
        let me = pos(&Pos3 { x: ax, y: 0.0, z: az }, d);
        assert!(p[1].as_f64().unwrap() < me[1].as_f64().unwrap());
        let v = snapshot_json(&rt.snapshot());
        assert_eq!(v["agent"]["yaw_deg"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn world_new_validates_and_applies_the_hand() {
        let mut rt = rt();
        assert!(rt.api_world_new(1, 8, None, None).is_err());
        assert!(rt.api_world_new(1, 64, None, Some("hoverboard")).is_err());
        let two = HandConfig::preset(FingerCount::Two);
        let v = rt.api_world_new(5, 48, Some(two), Some("track")).unwrap();
        assert_eq!(v["seed"], 5);
        assert_eq!(v["agent"]["fingers"], 2);
        assert_eq!(v["agent"]["base"], "track");
        assert_eq!(v["bounds"][0].as_f64().unwrap(), 24.0);
        // a refused request leaves the world as it was
        assert!(rt.api_world_new(9, 8, None, None).is_err());
        assert_eq!(rt.snapshot().seed, 5);
    }

    #[test]
    fn command_submit_answers_what_the_command_bar_prints() {
        let mut rt = rt();
        let v = rt.api_command_submit("철광석 2개 캐서 창고로 가져와").unwrap();
        assert_eq!(v["ok"], true);
        assert_eq!(v["parsed"]["action"], "mine_and_deliver");
        assert_eq!(v["parsed"]["item_id"], "iron_ore");
        assert_eq!(v["parsed"]["item_label_ko"], "철광석");
        assert_eq!(v["parsed"]["count"], 2);
        assert_eq!(v["parsed"]["destination"]["kind"], "waypoint");
        assert_eq!(v["parsed"]["destination"]["id"], "warehouse");
        assert!(v["quest_id"].is_string());
        let trace = v["trace"].as_array().unwrap();
        assert_eq!(trace.first().unwrap()["stage"], "normalize");
        assert_eq!(trace.last().unwrap()["stage"], "assign");
        assert!(trace.iter().any(|r| r["role"] == "아이템" && r["accepted"] == true));
        assert!(rt.api_quest_list().unwrap().as_array().unwrap().len() == 1);

        let bad = rt.api_command_submit("춤춰").unwrap();
        assert_eq!(bad["ok"], false);
        assert!(bad["error"].is_string());
        assert_eq!(bad["parsed"]["action"], "unknown");
        assert!(bad["quest_id"].is_null());
    }

    #[test]
    fn several_targets_are_summarised_for_the_one_line_display() {
        let mut rt = rt();
        let v = rt.api_command_submit("원목 2개랑 석탄 1개 창고로 가져와").unwrap();
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["parsed"]["targets"].as_array().unwrap().len(), 2);
        assert!(v["parsed"]["count"].is_null());
        assert!(v["parsed"]["item_label_ko"].as_str().unwrap().contains(" + "));
    }

    #[test]
    fn a_run_through_the_api_moves_the_quest_and_the_log() {
        let mut rt = rt();
        rt.api_command_submit("돌 1개 캐서 베이스캠프로 가져와").unwrap();
        let mut last = Value::Null;
        for _ in 0..40 {
            last = rt.api_sim_step(25).unwrap();
        }
        let q = &last["quests"][0];
        assert!(matches!(q["state"].as_str(), Some("active" | "done")), "{q}");
        assert!(last["snapshot"]["tick"].as_u64().unwrap() >= 1000);
        assert!(last["body"]["parts"].as_array().unwrap().len() == 7);
        let tail = rt.api_log_tail(0, 5, Some("quest")).unwrap();
        let lines = tail["lines"].as_array().unwrap();
        assert!(!lines.is_empty() && lines.iter().all(|l| l["cat"] == "quest"));
        assert!(lines[0]["ts"].as_str().unwrap().contains('T'));
        let next = tail["next_seq"].as_u64().unwrap();
        let again = rt.api_log_tail(next, 5, Some("quest")).unwrap();
        assert!(again["next_seq"].as_u64().unwrap() >= next);
    }

    #[test]
    fn frames_and_scans_carry_boxes_and_reasons() {
        let mut rt = rt();
        let scan = rt.api_perception_scan(5).unwrap();
        assert_eq!(scan["mode"], "oracle");
        let labels = scan["labels"].as_array().unwrap();
        assert!(labels.len() <= 5);
        for l in labels {
            assert!(l["reason"].as_str().unwrap().len() > 3);
            assert!(l["score"].as_f64().unwrap() <= 1.0);
        }
        let f = rt.api_world_frame(320, 200, 70.0, "agent").unwrap();
        assert_eq!(f["width"], 320);
        assert_eq!(f["height"], 200);
        let b64 = f["webp_base64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        assert_eq!(&bytes[..4], b"RIFF");
        for d in f["detections"].as_array().unwrap() {
            let b = d["bbox"].as_array().unwrap();
            assert!(b.len() == 4 && b.iter().all(|x| (0.0..=1.0).contains(&x.as_f64().unwrap())));
            assert!(d["label_id"].is_string() && d["reason"].is_string());
        }
        let top = rt.api_world_frame(300, 256, 70.0, "top").unwrap();
        assert_eq!(top["width"], top["height"]);
    }

    #[test]
    fn forecast_asks_for_a_history_that_exists() {
        let mut rt = rt();
        assert!(rt.api_ts_forecast("vision|tree|", "score").is_err());
        assert!(rt.api_ts_forecast("motion|wheel|move", "route_time_s").unwrap_err().contains("이력"));
        for _ in 0..40 {
            rt.scan_now("test").unwrap();
            rt.tick();
        }
        let labels = rt.gates().series_labels();
        let (label, n) = labels.first().cloned().expect("a label has a history after forty scans");
        assert!(n >= 3);
        let v = rt.api_ts_forecast(&format!("vision|{label}|"), "score").unwrap();
        assert_eq!(v["history"].as_array().unwrap().len(), n);
        assert_eq!(v["point"].as_array().unwrap().len(), v["p10"].as_array().unwrap().len());
        assert_eq!(v["point"].as_array().unwrap().len(), v["p90"].as_array().unwrap().len());
        let m = rt.api_ts_forecast(&format!("vision|{label}|"), "margin").unwrap();
        assert_eq!(m["history"].as_array().unwrap().len(), n);
    }

    #[test]
    fn sds_status_tells_what_the_gates_did() {
        let mut rt = rt();
        for _ in 0..20 {
            rt.scan_now("test").unwrap();
        }
        let v = rt.api_sds_status().unwrap();
        assert!(v["ledgers"].is_array() && v["axes"].is_array());
        assert!(v["precision"]["evaluated"].as_u64().unwrap() > 0);
        assert!(v["precision"]["eval"].is_object());
        let fc = v["forecastable"].as_array().unwrap();
        assert!(!fc.is_empty(), "twenty scans leave a history to forecast");
        let first = &fc[0];
        let r = rt.api_ts_forecast(first["scope"].as_str().unwrap(), first["axis"].as_str().unwrap());
        assert!(r.is_ok(), "every listed pair can be forecast: {r:?}");
    }

    #[test]
    fn the_dispatcher_reads_the_ui_argument_names() {
        let mut rt = rt();
        let config = serde_json::to_value(HandConfig::preset(FingerCount::Three)).unwrap();
        let v = rt.api_call("world_new", &json!({"seed": 3, "size": 64, "config": config, "base": "mecanum"})).unwrap();
        assert_eq!(v["agent"]["fingers"], 3);
        assert_eq!(v["agent"]["base"], "mecanum");
        assert!(rt.api_call("sim_step", &json!({"ticks": 3})).is_ok());
        assert!(rt.api_call("log_tail", &json!({"afterSeq": 0, "limit": 3, "category": null})).is_ok());
        assert!(rt.api_call("perception_scan", &json!({"topK": 3})).is_ok());
        let m = rt.api_call("models_status", &json!({})).unwrap();
        assert_eq!((m["recognizer"].as_str(), m["embedding"].as_bool(), m["ttm"].as_bool()), (Some("oracle"), Some(false), Some(false)));
        assert!(rt.api_call("world_new", &json!({"seed": -1, "size": 64})).is_err());
        assert_eq!(rt.api_call("nope", &json!({})).unwrap_err(), "unknown command nope");
    }

    #[test]
    fn granite_ttm_joins_the_fusion_once_it_is_attached() {
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use hand_ts::ttm::{TtmConfig, TtmForecaster};

        let cfg: TtmConfig = serde_json::from_value(json!({
            "context_length": 52, "prediction_length": 16, "patch_length": 4, "patch_stride": 4, "num_input_channels": 1,
            "d_model": 16, "num_layers": 2, "expansion_factor": 2, "mode": "common_channel", "gated_attn": true,
            "norm_mlp": "LayerNorm", "scaling": "std", "decoder_num_layers": 2, "decoder_d_model": 8,
            "decoder_mode": "common_channel", "use_decoder": true, "adaptive_patching_levels": 0
        }))
        .unwrap();
        // Weights are all zero: this checks the wiring, not forecast quality.
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let ttm = Arc::new(TtmForecaster::from_varbuilder(cfg, vb, std::path::PathBuf::from("zeros")).expect("model builds"));

        let mut rt = rt();
        for _ in 0..60 {
            rt.scan_now("test").unwrap();
            rt.tick();
        }
        let (label, n) = rt.gates().series_labels().first().cloned().unwrap();
        assert!(n >= 12);
        let scope = format!("vision|{label}|");
        let before = rt.api_ts_forecast(&scope, "score").unwrap();
        assert!(!before["method"].as_str().unwrap().contains("ttm"), "{}", before["method"]);
        assert!(!rt.has_ttm());

        rt.attach_ttm(Some(Arc::clone(&ttm)));
        assert!(rt.has_ttm());
        let after = rt.api_ts_forecast(&scope, "score").unwrap();
        let weights = after["weights"].as_array().unwrap();
        assert!(weights.iter().any(|w| w["method"] == "ttm"), "ttm takes part: {after}");
        assert_eq!(after["point"].as_array().unwrap().len(), 16);
        assert!(after["point"].as_array().unwrap().iter().all(|v| v.as_f64().is_some_and(|x| x.is_finite())));
        // the log says so, and detaching goes back
        let ts = rt.api_log_tail(0, 50, Some("ts")).unwrap();
        assert!(ts["lines"].as_array().unwrap().iter().any(|l| l["msg"].as_str().unwrap().contains("Granite TTM 연결")));
        rt.attach_ttm(None);
        assert!(!rt.has_ttm());
        let again = rt.api_ts_forecast(&scope, "score").unwrap();
        assert!(!again["weights"].as_array().unwrap().iter().any(|w| w["method"] == "ttm"));
    }
}
