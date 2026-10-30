// SGBM on the renderer's stereo pairs: the head camera and the lift truck's carriage camera render a left frame
// (the eye itself) and a right frame one baseline along the camera's right axis; both go through WebP like a
// keyframe; SGBM measures the left frame's depth. Scored against the renderer's depth on the pixels SGBM kept, and
// on the target item's pixels: how much of the grid has depth, how close it is, and the disparity error that the
// runtime's σ_Z = Z²·σ_d / (f·b) takes as σ_d.
//   cargo test -p noto-vision --test sgbm_render_eval -- --nocapture
use noto_core::{BaseKind, NotoConfig};
use noto_vision::stereo::SubpixelFit;
use noto_vision::{sgbm, SgbmCfg, StereoRig};
use world_sim::render::{focal_px, pack_id};
use world_sim::world::{camera_of, DropItem, Node, WorldConfig};
use world_sim::{decode_webp, encode_webp, Camera, Frame, ItemKind, NodeKind, ObjectClass, TerrainKind, World};

fn base_world(seed: u64, terrain: TerrainKind, base: BaseKind) -> World {
    let mut noto = NotoConfig::preset(noto_core::FingerCount::Five);
    noto.base = base;
    World::new(WorldConfig { seed, parcels: Some(0), terrain, noto, ..WorldConfig::default() }).expect("world")
}

/// Stereo keyframes are kept at this WebP quality (the runtime's `STEREO_KEYFRAME_QUALITY`): at 70 the codec's
/// artefacts, different in each eye, more than double the disparity error (see `compare_sources_of_disparity_error`).
const PAIR_QUALITY: f32 = 95.0;

fn webp(f: &Frame) -> noto_vision::Rgb {
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, PAIR_QUALITY).expect("webp");
    let (w, h, data) = decode_webp(&enc.bytes).expect("webp");
    noto_vision::Rgb::new(w, h, data)
}

struct Score {
    name: String,
    valid: f32,
    absrel: f32,
    delta1: f32,
    disp_rms: f32,
    /// Disparity errors (px) of every kept pixel, for the robust spread and the outlier share.
    disp_err: Vec<f32>,
    /// absrel within 2 m, 2–5 m and beyond 5 m.
    bands: [(f32, u32); 3],
    target_rel: Option<f32>,
    target_valid: f32,
    ms: f32,
    /// Grid pixels next to a jump in the true disparity, and how many of them are more than a pixel off.
    edge: (u32, u32),
    /// The error's change from one row to the next (px): summed, and over how many pairs.
    streak: (f32, u32),
}

/// Puts `label` `dist_m` ahead (a tree a little to the side behind it, for clutter), renders the pair with the head
/// tilted down `tilt_deg` (or through the carriage camera) and scores SGBM on it.
fn shot(base: &World, label: &str, dist_m: f32, tilt_deg: f32, fork_cam: bool, tag: &str) -> Score {
    shot_with(base, label, dist_m, tilt_deg, fork_cam, tag, &SgbmCfg::default())
}

fn shot_with(base: &World, label: &str, dist_m: f32, tilt_deg: f32, fork_cam: bool, tag: &str, cfg: &SgbmCfg) -> Score {
    let mut w = base.clone();
    w.nodes.clear();
    w.drops.clear();
    let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
    let at = |d: f32, lat: f32| (x + d * yaw.cos() - lat * yaw.sin(), z + d * yaw.sin() + lat * yaw.cos());
    let (px, pz) = at(dist_m, 0.0);
    let id = 9001;
    let raw = match label {
        "log" => {
            w.drops.push(DropItem::whole(id, ItemKind::Log, px, pz));
            pack_id(ObjectClass::Drop, id)
        }
        "stone" => {
            w.drops.push(DropItem::whole(id, ItemKind::Stone, px, pz));
            pack_id(ObjectClass::Drop, id)
        }
        "parcel" => {
            w.drops.push(DropItem::whole(id, ItemKind::Parcel, px, pz));
            pack_id(ObjectClass::Drop, id)
        }
        _ => {
            w.nodes.push(Node { id, kind: NodeKind::Tree, x: px, z: pz, hp: 10.0, max_hp: 10.0 });
            pack_id(ObjectClass::Node, id)
        }
    };
    let (qx, qz) = at(dist_m + 2.0, -1.2);
    w.nodes.push(Node { id: 9002, kind: NodeKind::Tree, x: qx, z: qz, hp: 10.0, max_hp: 10.0 });
    w.fork_cam = fork_cam;
    let cam = Camera { pitch_deg: Camera::default().pitch_deg - tilt_deg, ..Camera::default() };
    let (_, b, _) = camera_of(w.body.head.sensor);
    let left = w.render(&cam);
    let right = w.render_right(&cam, b);
    let rig = StereoRig { focal_px: focal_px(&cam), baseline_m: b, subpixel_px: 0.27 };
    let out = sgbm(&webp(&left), &webp(&right), &rig, cfg, None).expect("sgbm");
    let st = out.stats;
    let (gw, gh) = (st.width as usize, st.height as usize);
    let fb = st.focal_px * b;
    let (mut n, mut ar, mut d1, mut se) = (0u32, 0.0f32, 0u32, 0.0f32);
    let mut disp_err = Vec::new();
    let mut bands = [(0.0f32, 0u32); 3];
    let (mut tgt_pred, mut tgt_true, mut tgt_n) = (Vec::new(), Vec::new(), 0u32);
    let mut err_grid = vec![f32::NAN; gw * gh];
    let mut true_grid = vec![f32::NAN; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            // The renderer's pixel under the grid pixel's centre.
            let fx = (((gx as f32 + 0.5) * left.width as f32 / gw as f32) as usize).min(left.width - 1);
            let fy = (((gy as f32 + 0.5) * left.height as f32 / gh as f32) as usize).min(left.height - 1);
            let truth = left.depth[fy * left.width + fx];
            let pred = out.depth.data[gy * gw + gx];
            let on_target = left.ids[fy * left.width + fx] == raw;
            if truth.is_finite() && truth > 0.0 {
                true_grid[gy * gw + gx] = fb / truth;
            }
            if on_target {
                tgt_n += 1;
                if truth.is_finite() {
                    tgt_true.push(truth);
                }
                if pred.is_finite() {
                    tgt_pred.push(pred);
                }
            }
            if !(pred.is_finite() && truth.is_finite() && truth > 0.0) {
                continue;
            }
            n += 1;
            ar += (pred - truth).abs() / truth;
            d1 += ((pred / truth).max(truth / pred) < 1.25) as u32;
            se += (fb / pred - fb / truth).powi(2);
            disp_err.push(fb / pred - fb / truth);
            err_grid[gy * gw + gx] = fb / pred - fb / truth;
            let band = if truth < 2.0 { 0 } else if truth < 5.0 { 1 } else { 2 };
            bands[band].0 += (pred - truth).abs() / truth;
            bands[band].1 += 1;
        }
    }
    let (mut edge, mut streak) = ((0u32, 0u32), (0.0f32, 0u32));
    for gy in 2..gh - 2 {
        for gx in 2..gw - 2 {
            let i = gy * gw + gx;
            let e = err_grid[i];
            if !e.is_finite() {
                continue;
            }
            let below = err_grid[i + gw];
            if below.is_finite() {
                streak = (streak.0 + (below - e).abs(), streak.1 + 1);
            }
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for dy in 0..5 {
                for dx in 0..5 {
                    let v = true_grid[(gy + dy - 2) * gw + gx + dx - 2];
                    if v.is_finite() {
                        (lo, hi) = (lo.min(v), hi.max(v));
                    }
                }
            }
            if hi - lo > 3.0 {
                edge = (edge.0 + 1, edge.1 + (e.abs() > 1.0) as u32);
            }
        }
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let target_rel = match (median(&mut tgt_pred), median(&mut tgt_true)) {
        (Some(p), Some(t)) => Some((p - t).abs() / t),
        _ => None,
    };
    Score {
        name: format!("{tag} {label} {dist_m:.1} m{}", if fork_cam { " (포크 카메라)".to_string() } else { format!(" 숙임 {tilt_deg:.0}°") }),
        valid: st.valid_frac(),
        absrel: ar / n.max(1) as f32,
        delta1: d1 as f32 / n.max(1) as f32,
        disp_rms: (se / n.max(1) as f32).sqrt(),
        disp_err,
        bands,
        target_rel,
        target_valid: tgt_pred.len() as f32 / tgt_n.max(1) as f32,
        ms: st.ms,
        edge,
        streak,
    }
}

/// Median absolute deviation as a standard deviation, and the share beyond three of them.
fn robust(v: &[f32]) -> (f32, f32) {
    let mut a: Vec<f32> = v.iter().map(|e| e.abs()).collect();
    a.sort_by(f32::total_cmp);
    let sd = 1.4826 * a.get(a.len() / 2).copied().unwrap_or(0.0);
    (sd, v.iter().filter(|e| e.abs() > 3.0 * sd.max(1e-3)).count() as f32 / v.len().max(1) as f32)
}

#[test]
fn sgbm_on_rendered_pairs_is_metric_where_it_answers() {
    let walker = base_world(1, TerrainKind::Open, BaseKind::HumanFoot);
    let rugged = base_world(2, TerrainKind::Rugged, BaseKind::HumanFoot);
    let truck = base_world(1, TerrainKind::Open, BaseKind::Wheel);
    let shots = vec![
        shot(&walker, "log", 1.0, 30.0, false, "보행"),
        shot(&walker, "stone", 1.5, 25.0, false, "보행"),
        shot(&walker, "parcel", 2.5, 10.0, false, "보행"),
        shot(&walker, "tree", 4.0, 0.0, false, "보행"),
        shot(&rugged, "log", 1.2, 30.0, false, "험지"),
        shot(&rugged, "tree", 6.0, 0.0, false, "험지"),
        shot(&truck, "log", 1.6, 30.0, false, "지게차 탑승자"),
        // The carriage camera stands about a metre ahead of the truck's centre.
        shot(&truck, "log", 2.4, 0.0, true, "지게차"),
    ];
    println!("{:<30} {:>5} {:>7} {:>6} {:>7} {:>7} {:>7} {:>7} {:>6} {:>8} {:>7} {:>5}", "장면", "유효", "absrel", "δ1", "<2m", "2-5m", ">5m", "σd", "이상%", "대상오차", "대상유효", "ms");
    let mut all = Vec::new();
    for s in &shots {
        let (sd, out) = robust(&s.disp_err);
        let band = |b: usize| if s.bands[b].1 > 0 { format!("{:.3}", s.bands[b].0 / s.bands[b].1 as f32) } else { "—".into() };
        println!(
            "{:<30} {:>4.0}% {:>7.3} {:>6.3} {:>7} {:>7} {:>7} {:>5.2}px {:>5.1}% {:>8} {:>6.0}% {:>5.0}",
            s.name,
            s.valid * 100.0,
            s.absrel,
            s.delta1,
            band(0),
            band(1),
            band(2),
            sd,
            out * 100.0,
            s.target_rel.map_or("—".to_string(), |r| format!("{:.1}%", r * 100.0)),
            s.target_valid * 100.0,
            s.ms
        );
        all.extend_from_slice(&s.disp_err);
    }
    let (sd, out) = robust(&all);
    println!("모든 장면: 시차 오차 σ_d {sd:.2} px (MAD), 3σ 밖 {:.1}%", out * 100.0);
    for s in &shots {
        // The ground holds a stereo match everywhere the eye looks down at it; what looks at the sky leaves that empty.
        let floor = if s.name.contains("tree") { 0.45 } else { 0.55 };
        assert!(s.valid > floor, "{}: SGBM answers somewhere ({:.0}% of the grid)", s.name, s.valid * 100.0);
        if s.bands[0].1 > 200 {
            let near = s.bands[0].0 / s.bands[0].1 as f32;
            // σ_Z / Z = Z·σ_d / (f·b): about 4 % at 1.5 m with the 5 cm baseline on the 288-px grid.
            assert!(near < 0.04, "{}: within 2 m the depth is metric: absrel {near:.3}", s.name);
        }
    }
    assert!(sd < 0.22 && out < 0.04, "disparity error {sd:.2} px, {:.1}% beyond 3σ", out * 100.0);
    let near: Vec<&Score> = shots.iter().filter(|s| s.target_rel.is_some()).collect();
    assert!(near.len() >= 6, "most targets get a depth");
    for s in near {
        assert!(s.target_rel.unwrap() < 0.10, "{}: the target's median depth is off by {:.1}%", s.name, s.target_rel.unwrap() * 100.0);
    }
}

/// The disparity error with P2 falling at the image's edges and without, and the speed at the frame sizes a stereo
/// eye could render.
#[test]
#[ignore]
fn compare_grids_and_edge_penalty() {
    let walker = base_world(1, TerrainKind::Open, BaseKind::HumanFoot);
    let rugged = base_world(2, TerrainKind::Rugged, BaseKind::HumanFoot);
    let truck = base_world(1, TerrainKind::Open, BaseKind::Wheel);
    let cfgs = [("적응 P2", SgbmCfg::default()), ("고정 P2", SgbmCfg { p2_edge: 0.0, ..SgbmCfg::default() })];
    for (name, cfg) in cfgs {
        let scenes: [(&World, &str, f32, f32, bool); 8] = [
            (&walker, "log", 1.0, 30.0, false),
            (&walker, "stone", 1.5, 25.0, false),
            (&walker, "parcel", 2.5, 10.0, false),
            (&walker, "tree", 4.0, 0.0, false),
            (&rugged, "log", 1.2, 30.0, false),
            (&rugged, "tree", 6.0, 0.0, false),
            (&truck, "log", 1.6, 30.0, false),
            (&truck, "log", 2.4, 0.0, true),
        ];
        let (mut all, mut valid, mut ms) = (Vec::new(), 0.0, 0.0);
        let (mut edge, mut streak) = ((0u32, 0u32), (0.0f32, 0u32));
        for (w, label, d, t, fork) in scenes {
            let s = shot_with(w, label, d, t, fork, "", &cfg);
            all.extend_from_slice(&s.disp_err);
            valid += s.valid / 8.0;
            ms += s.ms / 8.0;
            edge = (edge.0 + s.edge.0, edge.1 + s.edge.1);
            streak = (streak.0 + s.streak.0, streak.1 + s.streak.1);
        }
        let (sd, out) = robust(&all);
        println!(
            "{name}: 유효 {:.1}% · σ_d {sd:.3} px · 3σ 밖 {:.2}% · 경계 화소 {}개 중 1 px 넘게 틀림 {:.1}% · 행간 오차 변화 {:.3} px · {ms:.0} ms",
            valid * 100.0,
            out * 100.0,
            edge.0,
            edge.1 as f32 / edge.0.max(1) as f32 * 100.0,
            streak.0 / streak.1.max(1) as f32
        );
    }
    for (w_px, h_px, per) in [(384usize, 256usize, 8u32), (576, 384, 16)] {
        let mut w = walker.clone();
        w.nodes.clear();
        w.drops.clear();
        let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        w.nodes.push(Node { id: 9002, kind: NodeKind::Tree, x: x + 5.0 * yaw.cos(), z: z + 5.0 * yaw.sin(), hp: 10.0, max_hp: 10.0 });
        let cam = Camera { width: w_px, height: h_px, ..Camera::default() };
        let (_, b, _) = camera_of(w.body.head.sensor);
        let (l, r) = (w.render(&cam), w.render_right(&cam, b));
        let rig = StereoRig { focal_px: focal_px(&cam), baseline_m: b, subpixel_px: 0.2 };
        let out = sgbm(&webp(&l), &webp(&r), &rig, &SgbmCfg { per_l2: per, ..SgbmCfg::default() }, None).expect("sgbm");
        let st = out.stats;
        let fb = st.focal_px * b;
        let (gw, gh) = (st.width as usize, st.height as usize);
        let mut e = Vec::new();
        for gy in 0..gh {
            for gx in 0..gw {
                let fx = (((gx as f32 + 0.5) * l.width as f32 / gw as f32) as usize).min(l.width - 1);
                let fy = (((gy as f32 + 0.5) * l.height as f32 / gh as f32) as usize).min(l.height - 1);
                let (t, p) = (l.depth[fy * l.width + fx], out.depth.data[gy * gw + gx]);
                if t.is_finite() && p.is_finite() && t > 3.0 {
                    e.push((p - t).abs());
                }
            }
        }
        e.sort_by(f32::total_cmp);
        println!("{gw}x{gh}: f·b {fb:.1} · 시차 {} · {:.0} ms · 3 m 밖 깊이 오차 중앙값 {:.3} m ({} 화소)", st.disparities, st.ms, e.get(e.len() / 2).copied().unwrap_or(0.0), e.len());
    }
}

#[test]
#[ignore]
fn compare_sources_of_disparity_error() {
    // Same scenes; the grid matched at the frame's own size (no resampling) by asking for a pyramid whose L2 grid
    // divides the frame, and with or without WebP.
    let walker = base_world(1, TerrainKind::Open, BaseKind::HumanFoot);
    for (q, tag) in [(70.0f32, "WebP q70"), (85.0, "WebP q85"), (95.0, "WebP q95"), (-1.0, "원본")] {
        let raw = q < 0.0;
        let mut bytes = 0usize;
        let mut all = Vec::new();
        for (label, d, t) in [("log", 1.0, 30.0), ("stone", 1.5, 25.0), ("parcel", 2.5, 10.0), ("tree", 4.0, 0.0)] {
            let mut w = walker.clone();
            w.nodes.clear();
            w.drops.clear();
            let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
            let _ = label;
            w.drops.push(DropItem::whole(9001, ItemKind::Log, x + d * yaw.cos(), z + d * yaw.sin()));
            let cam = Camera { pitch_deg: Camera::default().pitch_deg - t, ..Camera::default() };
            let (_, b, _) = camera_of(w.body.head.sensor);
            let (l, r) = (w.render(&cam), w.render_right(&cam, b));
            let conv = |f: &Frame| {
                if raw {
                    return (noto_vision::Rgb::new(f.width as u32, f.height as u32, f.rgb.clone()), 0usize);
                }
                let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, q).expect("webp");
                let (w, h, data) = decode_webp(&enc.bytes).expect("webp");
                (noto_vision::Rgb::new(w, h, data), enc.bytes.len())
            };
            let ((lr, lb), (rr, rb)) = (conv(&l), conv(&r));
            bytes += lb + rb;
            let rig = StereoRig { focal_px: focal_px(&cam), baseline_m: b, subpixel_px: 0.25 };
            for per in [8u32] {
                let cfg = SgbmCfg { per_l2: per, max_rel_sigma: 10.0, ..SgbmCfg::default() };
                let out = sgbm(&lr, &rr, &rig, &cfg, None).unwrap();
                let st = out.stats;
                let (gw, gh) = (st.width as usize, st.height as usize);
                let fb = st.focal_px * b;
                for gy in 0..gh {
                    for gx in 0..gw {
                        let fx = (((gx as f32 + 0.5) * l.width as f32 / gw as f32) as usize).min(l.width - 1);
                        let fy = (((gy as f32 + 0.5) * l.height as f32 / gh as f32) as usize).min(l.height - 1);
                        let (t, p) = (l.depth[fy * l.width + fx], out.depth.data[gy * gw + gx]);
                        if t.is_finite() && p.is_finite() {
                            all.push(fb / p - fb / t);
                        }
                    }
                }
            }
        }
        let (sd, out) = robust(&all);
        println!("{tag}: σ_d {sd:.3} px, 3σ 밖 {:.1}%, n {}, 쌍 평균 {:.1} KB", out * 100.0, all.len(), bytes as f32 / 4.0 / 1024.0);
    }
}

/// Writes the left frame, the true disparity and SGBM's disparity side by side (holes black) and prints why pixels are
/// empty, for looking at what the matcher keeps. `SGBM_DUMP_DIR` names the output folder.
#[test]
#[ignore]
fn dump_depth_views() {
    let dir = std::env::var("SGBM_DUMP_DIR").expect("SGBM_DUMP_DIR");
    let walker = base_world(1, TerrainKind::Open, BaseKind::HumanFoot);
    let truck = base_world(1, TerrainKind::Open, BaseKind::Wheel);
    for (name, base, label, dist_m, tilt, fork) in [("log_1m", &walker, "log", 1.0f32, 30.0f32, false), ("tree_4m", &walker, "tree", 4.0, 0.0, false), ("fork_log", &truck, "log", 0.9, 0.0, true)] {
        let mut w = base.clone();
        w.nodes.clear();
        w.drops.clear();
        let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
        if label == "log" {
            w.drops.push(DropItem::whole(9001, ItemKind::Log, x + dist_m * yaw.cos(), z + dist_m * yaw.sin()));
        } else {
            w.nodes.push(Node { id: 9001, kind: NodeKind::Tree, x: x + dist_m * yaw.cos(), z: z + dist_m * yaw.sin(), hp: 10.0, max_hp: 10.0 });
        }
        w.fork_cam = fork;
        let cam = Camera { pitch_deg: Camera::default().pitch_deg - tilt, ..Camera::default() };
        let (_, b, _) = camera_of(w.body.head.sensor);
        let (l, r) = (w.render(&cam), w.render_right(&cam, b));
        let rig = StereoRig { focal_px: focal_px(&cam), baseline_m: b, subpixel_px: 0.27 };
        let out = sgbm(&webp(&l), &webp(&r), &rig, &SgbmCfg::default(), None).expect("sgbm");
        let st = out.stats;
        let (gw, gh) = (st.width as usize, st.height as usize);
        let fb = st.focal_px * b;
        let pct = |n: u32| 100.0 * n as f32 / st.pixels() as f32;
        println!(
            "{name}: 격자 {gw}x{gh} · 시차 {} 단계 · f·b {fb:.1} · 유효 {:.0}% 멂 {:.0}% · 민무늬 {:.0}% 고유성 {:.0}% 좌우 {:.0}% 얼룩 {:.0}% 가장자리 {:.0}%",
            st.disparities,
            pct(st.valid),
            pct(st.far),
            pct(st.no_texture),
            pct(st.not_unique),
            pct(st.lr_mismatch),
            pct(st.speckle),
            pct(st.border)
        );
        let colour = |d: f32| -> [u8; 3] {
            if !d.is_finite() {
                return [0, 0, 0];
            }
            let t = (d / 16.0).clamp(0.0, 1.0);
            [(255.0 * t) as u8, (255.0 * (1.0 - (2.0 * t - 1.0).abs())) as u8, (255.0 * (1.0 - t)) as u8]
        };
        let mut img = image::RgbImage::new(3 * gw as u32 + 8, gh as u32);
        for gy in 0..gh {
            for gx in 0..gw {
                let fx = (((gx as f32 + 0.5) * l.width as f32 / gw as f32) as usize).min(l.width - 1);
                let fy = (((gy as f32 + 0.5) * l.height as f32 / gh as f32) as usize).min(l.height - 1);
                let c = fy * l.width + fx;
                img.put_pixel(gx as u32, gy as u32, image::Rgb([l.rgb[c * 3], l.rgb[c * 3 + 1], l.rgb[c * 3 + 2]]));
                let t = l.depth[c];
                img.put_pixel((gw + 4 + gx) as u32, gy as u32, image::Rgb(colour(if t.is_finite() { fb / t } else { f32::NAN })));
                img.put_pixel((2 * gw + 8 + gx) as u32, gy as u32, image::Rgb(colour(out.disparity[gy * gw + gx])));
            }
        }
        img.save(format!("{dir}/{name}.png")).expect("png");
        image::save_buffer(format!("{dir}/{name}_l.png"), &l.rgb, l.width as u32, l.height as u32, image::ColorType::Rgb8).expect("png");
        image::save_buffer(format!("{dir}/{name}_r.png"), &r.rgb, r.width as u32, r.height as u32, image::ColorType::Rgb8).expect("png");
    }
}
