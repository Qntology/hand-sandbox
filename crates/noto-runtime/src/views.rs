use noto_sds::Level;
use serde_json::{json, Value};
use world_sim::iso::{IsoView, RobotExtent};
use world_sim::light::{Lighting, PRESETS, PRESET_DAY, PRESET_LAT_DEG};
use world_sim::render::{Cutaway, FocusCam};
use world_sim::{Camera, Frame, World};

use crate::runtime::{FrameView, Runtime};

pub const ISO_FOLLOW_S: f32 = 0.5;
pub const ISO_SNAP_S: f64 = 5.0 * ISO_FOLLOW_S as f64;

fn r3(x: f32) -> f64 {
    let v = ((x as f64) * 1000.0).round() / 1000.0;
    if v == 0.0 { 0.0 } else { v }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IsoFollow {
    state: Option<FollowState>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FollowState {
    pos: [f32; 3],
    vel: [f32; 3],
    t_s: f64,
}

impl IsoFollow {
    pub fn advance(&mut self, target: [f32; 3], now_s: f64) -> [f32; 3] {
        let st = match self.state {
            Some(s) if now_s >= s.t_s && now_s - s.t_s <= ISO_SNAP_S => {
                let dt = (now_s - s.t_s) as f32;
                let w = 2.0 / ISO_FOLLOW_S;
                let e = (-w * dt).exp();
                let mut next = FollowState { pos: s.pos, vel: s.vel, t_s: now_s };
                for k in 0..3 {
                    let c0 = s.pos[k] - target[k];
                    let c1 = s.vel[k] + w * c0;
                    next.pos[k] = target[k] + (c0 + c1 * dt) * e;
                    next.vel[k] = (s.vel[k] - w * c1 * dt) * e;
                }
                next
            }
            _ => FollowState { pos: target, vel: [0.0; 3], t_s: now_s },
        };
        self.state = Some(st);
        st.pos
    }

    pub fn centre(&self) -> Option<[f32; 3]> {
        self.state.map(|s| s.pos)
    }
}

pub fn parse_view(view: &str) -> Result<FrameView, String> {
    let mut parts = view.trim().split(':');
    let id = parts.next().unwrap_or("");
    let args: Vec<&str> = parts.collect();
    let num = |s: &str, what: &str| -> Result<f32, String> {
        s.trim().parse::<f32>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("시점 '{view}'의 {what} '{s}'를 읽을 수 없습니다"))
    };
    match (id, args.as_slice()) {
        ("agent", []) => Ok(FrameView::Agent),
        ("top", []) => Ok(FrameView::Top),
        ("chase", []) => Ok(FrameView::Chase { orbit_deg: 0 }),
        ("chase", [o]) => Ok(FrameView::Chase { orbit_deg: num(o, "궤도 각")?.clamp(-180.0, 180.0) as i16 }),
        ("focus", []) => Ok(FrameView::Focus(FocusCam::Hand)),
        ("focus", [c]) => FocusCam::from_id(c.trim()).map(FrameView::Focus).ok_or_else(|| format!("알 수 없는 근접 카메라 '{c}' (hand · side · low)")),
        ("iso", rest) if rest.len() <= 3 => {
            let az_step = match rest.first() {
                None => 0,
                Some(a) => iso_step(num(a, "방향")?),
            };
            let zoom = match rest.get(1) {
                None => 1.0,
                Some(z) => num(z, "배율")?.clamp(0.25, 4.0),
            };
            let dimetric = match rest.get(2).map(|p| p.trim()) {
                None | Some("iso") | Some("35.264") | Some("35") => false,
                Some("dimetric") | Some("30") => true,
                Some(p) => return Err(format!("시점 '{view}'의 기울기 '{p}'는 35.264(등각) 또는 30(2:1)이어야 합니다")),
            };
            Ok(FrameView::Iso { az_step, zoom_pct: (zoom * 100.0).round() as u16, dimetric })
        }
        _ => Err(format!("알 수 없는 시점 '{view}' (agent · top · chase[:궤도°] · focus:hand|side|low · iso[:방향[:배율[:30]]])")),
    }
}

fn iso_step(a: f32) -> u8 {
    if a >= 0.0 && a <= 3.0 && a.fract() == 0.0 {
        a as u8
    } else {
        ((((a - 45.0) / 90.0).round() as i32).rem_euclid(4)) as u8
    }
}

pub fn render_iso(world: &World, cam: &Camera, follow: &mut IsoFollow, az_step: u8, zoom: f32, dimetric: bool) -> (Frame, Option<Value>) {
    let pose = world.body_pose();
    let tilt = pose.as_ref().map(|p| world.base_tilt(p)).unwrap_or_default();
    let boxes = pose.as_ref().map(|p| world.tilted_robot_boxes(p, &tilt)).unwrap_or_default();
    let ext = RobotExtent::of(world, &boxes);
    let ground = world.agent_ground_m();
    let centre = follow.advance([world.agent.x, ground + 0.5 * ext.height_m, world.agent.z], world.time_s);
    let view = IsoView::new(centre, az_step, ext.iso_half_h_m(zoom), dimetric);
    let cut = Cutaway::around(&view, [world.agent.x, world.agent.z], ground, &ext);
    let frame = world_sim::iso::render_iso(world, cam, &view, &boxes, Some(&cut));
    let look_compass = (90.0 + view.yaw.to_degrees()).rem_euclid(360.0);
    let camera = json!({
        "kind": "iso", "centre": view.centre.map(r3), "az_step": az_step, "look_compass_deg": r3(look_compass),
        "yaw_deg": r3(view.yaw.to_degrees()), "pitch_deg": r3(view.pitch.to_degrees()), "half_h_m": r3(view.half_h_m), "zoom": r3(zoom),
        "dimetric": dimetric, "follow_s": ISO_FOLLOW_S,
    });
    let overlay = pose.map(|p| crate::chase::overlay(world, &p, &frame, &|q| world.tilted_body_to_world(q, &tilt), &|q| view.project(cam, q), camera));
    (frame, overlay)
}

pub fn parse_light(args: &Value, now: Lighting) -> Result<Option<Lighting>, String> {
    let num = |k: &str| args.get(k).and_then(Value::as_f64).map(|v| v as f32);
    let overcast = match num("overcast") {
        None => None,
        Some(o) if (0.0..=1.0).contains(&o) => Some(o),
        Some(o) => return Err(format!("overcast 는 0(맑음)~1(흐림)이어야 합니다 (받은 값 {o})")),
    };
    let with_clouds = |l: Lighting| match (l, overcast) {
        (Lighting::Sun { elevation_deg, azimuth_deg, .. }, Some(o)) => Lighting::Sun { elevation_deg, azimuth_deg, overcast: o },
        (l, _) => l,
    };
    if let Some(p) = args.get("preset").and_then(Value::as_str).filter(|p| !p.is_empty()) {
        let l = Lighting::preset(p).ok_or_else(|| format!("알 수 없는 시간대 '{p}' ({})", PRESETS.iter().map(|q| format!("{}/{}", q.id, q.label_ko)).collect::<Vec<_>>().join(" · ")))?;
        return Ok(Some(with_clouds(l)));
    }
    if let Some(c) = args.get("clock").filter(|c| !c.is_null()) {
        let (hour, day, lat) = match c {
            Value::Number(n) => (n.as_f64().map(|h| h as f32), None, None),
            Value::Object(_) => (
                c.get("hour").and_then(Value::as_f64).map(|h| h as f32),
                c.get("day").and_then(Value::as_f64),
                c.get("lat_deg").and_then(Value::as_f64).map(|v| v as f32),
            ),
            _ => (None, None, None),
        };
        let hour = hour.filter(|h| (0.0..=24.0).contains(h)).ok_or("clock.hour 는 0~24 시여야 합니다")?;
        let day = match day {
            None => PRESET_DAY,
            Some(d) if (1.0..=366.0).contains(&d) => d as u32,
            Some(d) => return Err(format!("clock.day 는 1~366 이어야 합니다 (받은 값 {d})")),
        };
        let lat = match lat {
            None => PRESET_LAT_DEG,
            Some(l) if (-90.0..=90.0).contains(&l) => l,
            Some(l) => return Err(format!("clock.lat_deg 는 -90~90 이어야 합니다 (받은 값 {l})")),
        };
        return Ok(Some(with_clouds(Lighting::from_clock(lat, day, hour))));
    }
    let mode = args.get("mode").and_then(Value::as_str);
    let (elev, azim) = (num("elevation").or(num("elevation_deg")), num("azimuth").or(num("azimuth_deg")));
    match mode {
        Some("flat") => Ok(Some(Lighting::Flat)),
        Some("sun") | None if elev.is_some() || azim.is_some() || mode.is_some() => {
            let (e0, a0, o0) = match now {
                Lighting::Sun { elevation_deg, azimuth_deg, overcast } => (elevation_deg, azimuth_deg, overcast),
                Lighting::Flat => (45.0, 180.0, 0.0),
            };
            let e = elev.unwrap_or(e0);
            if !(-90.0..=90.0).contains(&e) {
                return Err(format!("elevation 은 -90~90° 여야 합니다 (받은 값 {e})"));
            }
            let a = azim.unwrap_or(a0);
            if !a.is_finite() {
                return Err("azimuth 를 읽을 수 없습니다".into());
            }
            Ok(Some(Lighting::Sun { elevation_deg: e, azimuth_deg: a.rem_euclid(360.0), overcast: overcast.unwrap_or(o0) }))
        }
        Some(m) if m != "sun" => Err(format!("알 수 없는 조명 모드 '{m}' (flat 또는 sun)")),
        _ => Ok(overcast.map(|_| with_clouds(now))),
    }
}

impl Runtime {
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.cfg.lighting = lighting;
        self.world.set_lighting(lighting);
        let s = lighting.summary();
        self.log("world", Level::Info, format!("조명: {} · 조도 {:.0} lx · 노출 {:.2} ms", s.label_ko, s.lux, s.exposure_ms), Some(json!({"lighting": lighting})));
    }

    pub fn api_world_light(&mut self, args: &Value) -> Result<Value, String> {
        if let Some(l) = parse_light(args, self.cfg.lighting)? {
            self.set_lighting(l);
        }
        Ok(light_json(&self.world))
    }
}

pub fn light_json(world: &World) -> Value {
    let mut v = serde_json::to_value(world.lighting().summary()).unwrap_or(Value::Null);
    if let Value::Object(o) = &mut v {
        o.insert("lighting".into(), serde_json::to_value(world.lighting()).unwrap_or(Value::Null));
        o.insert("presets".into(), json!(PRESETS.iter().map(|p| json!({"id": p.id, "label_ko": p.label_ko, "hour": p.hour})).collect::<Vec<_>>()));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;

    fn rt() -> Runtime {
        let lex = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json")).expect("lexicon file");
        Runtime::new(RuntimeConfig { cells: 40, ..RuntimeConfig::default() }, &lex, None).unwrap()
    }

    #[test]
    fn view_ids_are_parsed_and_unknown_ones_rejected() {
        assert_eq!(parse_view("agent"), Ok(FrameView::Agent));
        assert_eq!(parse_view("top"), Ok(FrameView::Top));
        assert_eq!(parse_view("chase"), Ok(FrameView::Chase { orbit_deg: 0 }));
        assert_eq!(parse_view("chase:-35"), Ok(FrameView::Chase { orbit_deg: -35 }));
        assert_eq!(parse_view("focus:side"), Ok(FrameView::Focus(FocusCam::Side)));
        assert_eq!(parse_view("iso"), Ok(FrameView::Iso { az_step: 0, zoom_pct: 100, dimetric: false }));
        assert_eq!(parse_view("iso:2:1.5"), Ok(FrameView::Iso { az_step: 2, zoom_pct: 150, dimetric: false }));
        assert_eq!(parse_view("iso:135:0.5:30"), Ok(FrameView::Iso { az_step: 1, zoom_pct: 50, dimetric: true }));
        assert_eq!(parse_view("iso:315:9"), Ok(FrameView::Iso { az_step: 3, zoom_pct: 400, dimetric: false }));
        for bad in ["", "agnet", "iso:x", "iso:0:1:45", "chase:abc", "focus:knee", "top:2", "iso:0:1:30:9"] {
            assert!(parse_view(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn the_iso_follow_runs_on_simulated_time_and_settles() {
        let mut f = IsoFollow::default();
        assert_eq!(f.advance([1.0, 2.0, 3.0], 10.0), [1.0, 2.0, 3.0]);
        assert_eq!(f.advance([1.0, 2.0, 3.0], 10.5), [1.0, 2.0, 3.0]);
        let (mut a, mut b) = (f.clone(), f.clone());
        let target = [2.0, 2.0, 3.0];
        let mut last = 1.0f32;
        for k in 1..=10 {
            let x = a.advance(target, 10.5 + 0.1 * k as f64)[0];
            assert!(x >= last - 1e-6 && x <= 2.0 + 1e-6, "critically damped: no overshoot ({x})");
            last = x;
        }
        let once = b.advance(target, 11.5)[0];
        assert!((once - last).abs() < 1e-4, "{once} vs {last}");
        assert_eq!(b.advance(target, 11.5)[0], once);
        let mut c = f.clone();
        let half = c.advance(target, 10.5 + ISO_FOLLOW_S as f64)[0] - 1.0;
        assert!(half > 0.5 && half < 0.7, "{half}");
        let settled = c.advance(target, 10.5 + 4.0 * ISO_FOLLOW_S as f64)[0];
        assert!((settled - 2.0).abs() < 0.01, "{settled}");
        assert_eq!(c.advance([9.0, 0.0, 9.0], 1.0), [9.0, 0.0, 9.0]);
        assert_eq!(c.advance([5.0, 0.0, 5.0], 1.0 + 2.0 * ISO_SNAP_S), [5.0, 0.0, 5.0]);
    }

    #[test]
    fn the_iso_frame_keeps_the_robot_in_view_while_it_drives() {
        let mut rt = rt();
        let cam = Camera { width: 192, height: 128, max_dist_m: 60.0, ..Camera::default() };
        let robot_centre = |f: &Frame| {
            let px: Vec<usize> = (0..f.width * f.height).filter(|&i| f.ids[i] == 0 && f.depth[i].is_finite()).collect();
            assert!(px.len() > 100, "{} robot pixels", px.len());
            (px.iter().map(|&i| (i % f.width) as f32).sum::<f32>() / px.len() as f32, px.iter().map(|&i| (i / f.width) as f32).sum::<f32>() / px.len() as f32)
        };
        let (f, o) = render_iso(&rt.world, &cam, &mut rt.iso, 0, 1.0, false);
        let o = o.expect("an overlay");
        assert_eq!(o["camera"]["kind"], "iso");
        assert!(o["joints"].as_array().unwrap().iter().filter(|j| j["at"]["visible"] == true).count() >= 3);
        let (x0, y0) = robot_centre(&f);
        assert!((x0 - 96.0).abs() < 15.0 && (y0 - 64.0).abs() < 18.0, "({x0}, {y0})");
        let (ax, az) = (rt.world.agent.x, rt.world.agent.z);
        let goal = (0..16)
            .map(|k| (k as f32 * std::f32::consts::TAU / 16.0).sin_cos())
            .map(|(s, c)| world_sim::Action::MoveTo { x: ax + 4.0 * c, z: az + 4.0 * s, reach_m: 0.3 })
            .find(|a| rt.world.clone().command(a.clone()).is_ok())
            .expect("a reachable point");
        rt.world.command(goal).unwrap();
        for _ in 0..30 {
            rt.world.tick();
            let (f, _) = render_iso(&rt.world, &cam, &mut rt.iso, 1, 1.0, false);
            let (x, y) = robot_centre(&f);
            assert!(x > 20.0 && x < 172.0 && y > 15.0 && y < 113.0, "robot at ({x}, {y})");
        }
    }

    #[test]
    fn the_frame_api_draws_iso_and_rejects_unknown_views() {
        let mut rt = rt();
        let f = rt.api_call("world_frame", &json!({"width": 240, "height": 160, "view": "iso:1:1.5"})).unwrap();
        assert!(f["webp_base64"].as_str().unwrap().len() > 100);
        assert_eq!(f["overlay"]["camera"]["kind"], "iso");
        assert_eq!(f["overlay"]["camera"]["az_step"], 1);
        assert!(f["camera"].is_null(), "the iso view is not the robot's camera");
        assert_eq!(f["light"]["mode"], "flat");
        let err = rt.api_call("world_frame", &json!({"view": "isometric"})).unwrap_err();
        assert!(err.contains("isometric"), "{err}");
        rt.api_call("world_light", &json!({"preset": "dusk"})).unwrap();
        let a = rt.api_call("world_frame", &json!({"width": 160, "height": 96, "view": "agent"})).unwrap();
        let lux = a["camera"]["sensor"]["scene_lux"].as_f64().unwrap();
        assert!(lux > 1_000.0 && lux < 20_000.0, "dusk lux {lux}");
        assert_eq!(a["camera"]["light"]["mode"], "sun");
        assert!(crate::api::COMMANDS.contains(&"world_light"));
    }

    #[test]
    fn world_light_sets_and_reports_the_sun() {
        let mut rt = rt();
        let q = rt.api_world_light(&Value::Null).unwrap();
        assert_eq!(q["mode"], "flat");
        assert_eq!(q["lux"].as_f64().unwrap(), 10_000.0);
        assert_eq!(q["presets"].as_array().unwrap().len(), 5);
        let noon = rt.api_world_light(&json!({"preset": "정오"})).unwrap();
        assert_eq!(noon["mode"], "sun");
        assert!(noon["lux"].as_f64().unwrap() > 50_000.0 && noon["sun"]["elevation_deg"].as_f64().unwrap() > 40.0);
        let set = rt.api_world_light(&json!({"mode": "sun", "elevation": 10.0, "azimuth": 450.0, "overcast": 0.5})).unwrap();
        assert_eq!(set["lighting"], json!({"mode": "sun", "elevation_deg": 10.0, "azimuth_deg": 90.0, "overcast": 0.5}));
        assert!(set["exposure_ms"].as_f64().unwrap() > noon["exposure_ms"].as_f64().unwrap());
        let clock = rt.api_world_light(&json!({"clock": {"hour": 15.0, "day": 172}})).unwrap();
        assert!(clock["sun"]["azimuth_deg"].as_f64().unwrap() > 180.0);
        let cloudy = rt.api_world_light(&json!({"overcast": 1.0})).unwrap();
        assert_eq!(cloudy["sun"]["azimuth_deg"], clock["sun"]["azimuth_deg"]);
        assert_eq!(cloudy["sun"]["direct_normal_lux"].as_f64().unwrap(), 0.0);
        for bad in [json!({"preset": "midnight"}), json!({"mode": "moon"}), json!({"elevation": 95.0}), json!({"clock": {"hour": 25}}), json!({"overcast": 2.0})] {
            assert!(rt.api_world_light(&bad).is_err(), "{bad}");
        }
        rt.reset(Some(3)).unwrap();
        assert_eq!(rt.world().lighting(), rt.cfg.lighting);
        assert!(matches!(rt.world().lighting(), Lighting::Sun { .. }));
        rt.api_world_light(&json!({"mode": "flat"})).unwrap();
        assert_eq!(rt.world().lighting(), Lighting::Flat);
        assert_eq!(rt.world().scene_lux(), 10_000.0);
    }
}
