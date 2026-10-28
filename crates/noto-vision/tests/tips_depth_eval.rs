// Real TIPSv2 on the head camera: how far each model-side step (ground alignment, ADE20K sky mask,
// predicted ground normals, flip averaging, input size) moves the depth from the renderer's truth.
//   set TIPS_DIR=...\models\tipsv2-b14-dpt
//   test_cuda.bat -p noto-vision --test tips_depth_eval -- --ignored --nocapture
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use noto_core::{BaseKind, NotoConfig};
use noto_vision::{ade_name, ground_align_with, AlignCfg, DepthMap, DepthModel, InputGrid, Intrinsics, Pose, Rgb, SceneHeads, TipsDepth};
use world_sim::render::{pack_id, unpack_id};
use world_sim::world::{DropItem, Node, WorldConfig};
use world_sim::{decode_webp, encode_webp, Camera, Frame, ItemKind, NodeKind, ObjectClass, TerrainKind, VisibleObject, World};

/// The renderer's answer at half resolution, as the runtime's `KeyTruth` keeps it.
struct KeyTruth {
    width: u32,
    height: u32,
    ids: Vec<u32>,
    depth: Vec<f32>,
    objects: Vec<VisibleObject>,
}

impl KeyTruth {
    fn capture(world: &World, frame: &Frame) -> KeyTruth {
        let (w, h) = ((frame.width / 2).max(1) as u32, (frame.height / 2).max(1) as u32);
        let (mut ids, mut depth) = (Vec::new(), Vec::new());
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * 2) * frame.width + x * 2;
                ids.push(frame.ids[i]);
                depth.push(frame.depth[i]);
            }
        }
        KeyTruth { width: w, height: h, ids, depth, objects: world.visible_objects(frame) }
    }
}

fn intrinsics(world: &World, cam: &Camera) -> Intrinsics {
    let est = world.head_view(cam).est;
    Intrinsics { width: cam.width as u32, height: cam.height as u32, fov_deg: cam.fov_deg, pitch_deg: est.pitch.to_degrees(), roll_deg: est.roll.to_degrees() }
}

fn odometry(world: &World, cam: &Camera) -> Pose {
    let est = world.head_view(cam).est;
    Pose { x: est.origin[0], z: est.origin[2], yaw: est.yaw, eye_m: est.origin[1] - world.agent_ground_m() }
}

struct Shot {
    name: String,
    rgb: Rgb,
    intr: Intrinsics,
    pose: Pose,
    truth: KeyTruth,
    target_raw: u32,
}

fn base_world(seed: u64, terrain: TerrainKind, base: BaseKind) -> World {
    let mut noto = NotoConfig::preset(noto_core::FingerCount::Five);
    noto.base = base;
    World::new(WorldConfig { seed, parcels: Some(0), terrain, noto, ..WorldConfig::default() }).expect("world")
}

fn place(w: &mut World, label: &str, id: u32, x: f32, z: f32) -> u32 {
    let node = |k| Node { id, kind: k, x, z, hp: 10.0, max_hp: 10.0 };
    match label {
        "tree" => w.nodes.push(node(NodeKind::Tree)),
        "boulder" => w.nodes.push(node(NodeKind::Boulder)),
        "log" => w.drops.push(DropItem::whole(id, ItemKind::Log, x, z)),
        _ => w.drops.push(DropItem::whole(id, ItemKind::Parcel, x, z)),
    }
    pack_id(if ["tree", "boulder"].contains(&label) { ObjectClass::Node } else { ObjectClass::Drop }, id)
}

fn shoot(base: &World, label: &str, dist_m: f32, tilt_deg: f32, tag: &str) -> Shot {
    let mut w = base.clone();
    w.nodes.clear();
    w.drops.clear();
    let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
    let at = |d: f32, lat: f32| (x + d * yaw.cos() - lat * yaw.sin(), z + d * yaw.sin() + lat * yaw.cos());
    let (px, pz) = at(dist_m, 0.0);
    let target_raw = place(&mut w, label, 9001, px, pz);
    let (qx, qz) = at(dist_m + 2.5, -1.4);
    place(&mut w, if label == "tree" { "boulder" } else { "tree" }, 9002, qx, qz);
    let cam = Camera { pitch_deg: Camera::default().pitch_deg - tilt_deg, ..Camera::default() };
    let frame = w.render(&cam);
    let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, 70.0).expect("webp");
    let (dw, dh, data) = decode_webp(&enc.bytes).expect("webp");
    Shot {
        name: format!("{tag} {label} {dist_m:.1} m tilt {tilt_deg:.0}°"),
        rgb: Rgb::new(dw, dh, data),
        intr: intrinsics(&w, &cam),
        pose: odometry(&w, &cam),
        truth: KeyTruth::capture(&w, &frame),
        target_raw,
    }
}

#[derive(Default)]
struct Book(BTreeMap<String, Vec<f32>>);

impl Book {
    fn add(&mut self, k: impl Into<String>, v: f32) {
        if v.is_finite() {
            self.0.entry(k.into()).or_default().push(v);
        }
    }
    fn mean(&self, k: &str) -> Option<f32> {
        self.0.get(k).filter(|v| !v.is_empty()).map(|v| v.iter().sum::<f32>() / v.len() as f32)
    }
}

struct Eval {
    absrel: f32,
    delta1: f32,
    rmse: f32,
    coverage: f32,
    target: Option<f32>,
    sky_leak: f32,
}

fn eval(pred: &DepthMap, t: &KeyTruth, cap: f32, target_raw: u32) -> Eval {
    let (mut ar, mut d1, mut se, mut n, mut all) = (0.0f64, 0u32, 0.0f64, 0u32, 0u32);
    let (mut tar, mut tn, mut sky, mut sky_n) = (0.0f64, 0u32, 0u32, 0u32);
    for y in 0..t.height {
        for x in 0..t.width {
            let i = (y * t.width + x) as usize;
            let truth = t.depth[i];
            let p = pred.at_scaled(x as f32 + 0.5, y as f32 + 0.5, t.width, t.height);
            let ok = p.is_finite() && p > 0.0;
            if !truth.is_finite() {
                sky_n += 1;
                sky += ok as u32;
                continue;
            }
            if ok && t.ids[i] == target_raw {
                tar += ((p - truth).abs() / truth) as f64;
                tn += 1;
            }
            if truth > cap {
                continue;
            }
            all += 1;
            if ok {
                ar += ((p - truth).abs() / truth) as f64;
                d1 += ((p / truth).max(truth / p) < 1.25) as u32;
                se += ((p - truth) as f64).powi(2);
                n += 1;
            }
        }
    }
    let n1 = n.max(1) as f64;
    Eval {
        absrel: (ar / n1) as f32,
        delta1: d1 as f32 / n.max(1) as f32,
        rmse: (se / n1).sqrt() as f32,
        coverage: n as f32 / all.max(1) as f32,
        target: (tn > 0).then(|| (tar / tn as f64) as f32),
        sky_leak: sky as f32 / sky_n.max(1) as f32,
    }
}

fn record(book: &mut Book, arm: &str, shot: &Shot, pred: &DepthMap) {
    for (cap, tag) in [(10.0f32, "10m"), (24.0, "24m")] {
        let e = eval(pred, &shot.truth, cap, shot.target_raw);
        book.add(format!("{arm}/{tag}/absrel"), e.absrel);
        book.add(format!("{arm}/{tag}/delta1"), e.delta1);
        book.add(format!("{arm}/{tag}/rmse"), e.rmse);
        book.add(format!("{arm}/{tag}/coverage"), e.coverage);
        if tag == "10m" {
            if let Some(v) = e.target {
                book.add(format!("{arm}/target"), v);
                book.add(format!("{arm}/target/{}", if shot.name.contains("tilt 0°") { "level" } else { "look_down" }), v);
            }
            book.add(format!("{arm}/sky_leak"), e.sky_leak);
        }
    }
}

fn class_of(raw: u32, shot: &Shot) -> &'static str {
    let (class, id) = unpack_id(raw);
    match class {
        ObjectClass::Sky => "sky",
        ObjectClass::Ground | ObjectClass::Curb => "ground",
        ObjectClass::Node | ObjectClass::Drop => {
            let label = shot.truth.objects.iter().find(|o| o.entity.get(1..).and_then(|s| s.parse::<u32>().ok()) == Some(id)).map(|o| o.label.as_str()).unwrap_or("");
            match label {
                "tree" => "tree",
                "boulder" => "boulder",
                "log" => "log",
                "parcel" => "parcel",
                _ => "object",
            }
        }
        _ => "other",
    }
}

#[test]
#[ignore]
fn tips_depth_steps_on_the_head_camera() {
    let dir = PathBuf::from(std::env::var("TIPS_DIR").expect("TIPS_DIR"));
    let device = noto_ml::preferred().backend.device(0).expect("device");
    let mut model = TipsDepth::load_with(&dir, &device, SceneHeads::ALL).expect("TIPSv2");
    println!("[DEPTH] device {} · heads {:?}", noto_ml::device_label(&device), model.heads());
    let legs = base_world(1, TerrainKind::Open, BaseKind::HumanFoot);
    let wheels = base_world(3, TerrainKind::Rugged, BaseKind::Wheel);
    let mut shots = Vec::new();
    for label in ["tree", "boulder", "log", "parcel"] {
        for (d, tilt) in [(1.0f32, 40.0f32), (1.5, 30.0), (2.0, 0.0), (2.0, 30.0), (4.0, 0.0), (7.0, 0.0)] {
            shots.push(shoot(&legs, label, d, tilt, "legs"));
        }
        for (d, tilt) in [(1.2f32, 35.0f32), (3.0, 0.0)] {
            shots.push(shoot(&wheels, label, d, tilt, "wheels"));
        }
    }
    let cfg = AlignCfg::default();
    let no_normals = AlignCfg { normal_max_deg: 0.0, ..cfg };
    let sweeps: [(&str, f32); 4] = [("n20", 20.0), ("n30", 30.0), ("n40", 40.0), ("n55", 55.0)];
    let mut book = Book::default();
    let mut timing: BTreeMap<String, f64> = BTreeMap::new();
    let (mut up_sum, mut raw_sum, mut up_n) = ([0.0f64; 3], [0.0f64; 3], 0u32);
    let mut seg_hist: BTreeMap<(&str, String), u32> = BTreeMap::new();
    let mut seg_tot: BTreeMap<&str, u32> = BTreeMap::new();
    let variants: [(&str, bool, usize); 1] = [("base", false, 24)];
    for (vname, flip, side) in variants {
        model.flip_tta = flip;
        model.input_grid = InputGrid::ShortSide(side);
        for shot in &shots {
            let t0 = Instant::now();
            let raw = model.predict(&shot.rgb).expect("depth");
            *timing.entry(format!("{vname} 깊이만")).or_default() += t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            let scene = model.infer_scene(&shot.rgb, SceneHeads::ALL).expect("scene");
            *timing.entry(format!("{vname} 깊이+법선+분할")).or_default() += t1.elapsed().as_secs_f64() * 1000.0;
            if vname == "base" {
                record(&mut book, "raw", shot, &raw);
                let mut ground = raw.clone();
                let fit = ground_align_with(&mut ground, None, &shot.intr, &shot.pose, None, Some(10.0), &no_normals);
                record(&mut book, "ground", shot, &ground);
                book.add("ground/fit", fit.is_some() as u8 as f32);
                let mut sky = scene.depth.clone();
                ground_align_with(&mut sky, None, &shot.intr, &shot.pose, None, Some(10.0), &no_normals);
                record(&mut book, "ground+sky", shot, &sky);
                let mut level = scene.depth.clone();
                ground_align_with(&mut level, None, &Intrinsics { roll_deg: 0.0, ..shot.intr }, &shot.pose, None, Some(10.0), &no_normals);
                record(&mut book, "ground+sky(no roll)", shot, &level);
                if shot.intr.roll_deg.abs() > 1.0 {
                    record(&mut book, "rolled/ground+sky", shot, &sky);
                    record(&mut book, "rolled/ground+sky(no roll)", shot, &level);
                }
                for (tag, deg) in sweeps {
                    let mut d = scene.depth.clone();
                    let fit = ground_align_with(&mut d, scene.normals.as_ref(), &shot.intr, &shot.pose, None, Some(10.0), &AlignCfg { normal_max_deg: deg, ..cfg });
                    record(&mut book, &format!("ground+sky+{tag}"), shot, &d);
                    book.add(format!("{tag}/off_ground"), fit.map_or(0.0, |f| f.off_ground_normals as f32));
                }
                if let (Some(n), Some(seg)) = (&scene.normals, &scene.segmentation) {
                    for y in 0..shot.truth.height {
                        for x in 0..shot.truth.width {
                            let raw_id = shot.truth.ids[(y * shot.truth.width + x) as usize];
                            let c = class_of(raw_id, shot);
                            let l = seg.label_scaled(x as f32 + 0.5, y as f32 + 0.5, shot.truth.width, shot.truth.height);
                            *seg_hist.entry((c, ade_name(l as usize).to_string())).or_default() += 1;
                            *seg_tot.entry(c).or_default() += 1;
                            if c == "ground" && y > shot.truth.height / 2 {
                                let v = n.at_scaled(x as f32 + 0.5, y as f32 + 0.5, shot.truth.width, shot.truth.height);
                                let up = shot.intr.world_up_in_camera();
                                for k in 0..3 {
                                    raw_sum[k] += v[k] as f64;
                                    up_sum[k] += up[k] as f64;
                                }
                                up_n += 1;
                            }
                        }
                    }
                }
            }
            let mut full = scene.depth.clone();
            let fit = ground_align_with(&mut full, scene.normals.as_ref(), &shot.intr, &shot.pose, None, Some(10.0), &cfg);
            record(&mut book, &format!("{vname}/ground+sky+normals"), shot, &full);
            book.add(format!("{vname}/fit"), fit.is_some() as u8 as f32);
            book.add(format!("{vname}/off_ground"), fit.map_or(0.0, |f| f.off_ground_normals as f32));
            let mut sky_only = scene.depth.clone();
            ground_align_with(&mut sky_only, None, &shot.intr, &shot.pose, None, Some(10.0), &no_normals);
            record(&mut book, &format!("{vname}/ground+sky"), shot, &sky_only);
        }
    }
    let n = shots.len() as f64;
    for (k, ms) in &timing {
        println!("[DEPTH] 시간 {k}: {:.0} ms/프레임", ms / n);
    }
    let f = |k: &str| book.mean(k).map_or("   -  ".to_string(), |v| format!("{v:6.3}"));
    println!("[DEPTH] {:<34} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}", "팔", "AbsRel", "δ1", "RMSE", "덮음", "≤24 AR", "목표", "목표숙임", "하늘누출");
    let mut arms: Vec<String> = vec!["raw".into(), "ground".into(), "ground+sky".into(), "ground+sky(no roll)".into(), "rolled/ground+sky".into(), "rolled/ground+sky(no roll)".into()];
    arms.extend(sweeps.iter().map(|(t, _)| format!("ground+sky+{t}")));
    for a in &arms {
        println!(
            "[DEPTH] {a:<34} {} {} {} {} {} {} {} {}",
            f(&format!("{a}/10m/absrel")),
            f(&format!("{a}/10m/delta1")),
            f(&format!("{a}/10m/rmse")),
            f(&format!("{a}/10m/coverage")),
            f(&format!("{a}/24m/absrel")),
            f(&format!("{a}/target")),
            f(&format!("{a}/target/look_down")),
            f(&format!("{a}/sky_leak"))
        );
    }
    for v in ["ground"].into_iter().chain(variants.iter().map(|x| x.0)) {
        println!("[DEPTH] 정렬 성공률 {v}: {}", f(&format!("{v}/fit")));
    }
    if up_n > 0 {
        let m = |s: [f64; 3]| [s[0] / up_n as f64, s[1] / up_n as f64, s[2] / up_n as f64];
        println!("[DEPTH] 법선 규약: 지면 화소의 모델 법선 평균 {:?} · 기대 위쪽(오른·위·뒤) {:?} (n {up_n})", m(raw_sum), m(up_sum));
    }
    for (c, tot) in &seg_tot {
        let mut top: Vec<(&String, &u32)> = seg_hist.iter().filter(|((k, _), _)| k == c).map(|((_, l), n)| (l, n)).collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        let line: Vec<String> = top.iter().take(4).map(|(l, n)| format!("{l} {:.0}%", **n as f32 * 100.0 / *tot as f32)).collect();
        println!("[DEPTH] 분할 {c:<7} ({tot} px): {}", line.join(" · "));
    }
}
