//! [MLAB] The controlled A/B of the ML tracks: the stand-ins against the real
//! models (TIPSv2 DPT, SigLIP2, MobileSAM) on identical keyframes, one factor
//! at a time, before and after the v1.0 fixes (`docs/research/ml-ab.md`).
//!
//!     set SIGLIP2_DIR=...\models\siglip2-base-patch16-naflex
//!     set TIPS_DIR=...\models\tipsv2-b14-dpt
//!     set SAM_DIR=...\models\sam
//!     set RUNTIME_DIR=...\runtime            (its `bank` keeps the phrase vectors)
//!     test_cuda.bat -p noto-runtime --test ml_ab -- --ignored --nocapture
//!
//! Every scene is the seed's world with its nodes and drops cleared, one
//! target placed in front of the robot (plus a distractor), seen from the
//! head camera at a distance and a head tilt; some scenes have no target. The
//! keyframe is made by the runtime's own capture code (`record::odometry`,
//! `record::intrinsics`, `KeyTruth::capture`, WebP q70), so every arm sees the
//! same bytes. The arms:
//!
//! | arm | depth | semantic location | masks |
//! |---|---|---|---|
//! | A0 | stand-in | stand-in | region-grow |
//! | A1 | TIPS | stand-in | region-grow |
//! | A2 | stand-in | SigLIP2 dense | region-grow |
//! | A3 | stand-in | stand-in | MobileSAM |
//! | A4 | TIPS | SigLIP2 dense | MobileSAM |
//! | A5 | TIPS + v1.0 | SigLIP2 + v1.0 | MobileSAM |
//!
//! A0–A4 run the port as it was before v1.0 (plain bilinear SigLIP2
//! positions, exact-GELU TIPS readout); A5 runs the v1.0 port and fixes.
//! Per frame it measures depth (raw, median-aligned, oracle least-squares in
//! disparity, ground-aligned; banded by true range; on the target), the
//! locator's presence AUROC/AP per label, the peak inside the true box, the
//! tracker's IoU with the truth and the target range error, and the tile
//! recogniser's precision/recall after the score and margin gates against the
//! oracle. The calibration of A5's gate is fitted on the fit seeds and scored
//! on the others; the fitted table is written to `$MLAB_OUT/ml_calib.json`.
//!
//! Group B references below are for scoring only; the engine never reads them
//! (`the_engine_never_names_an_ml_benchmark_source`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use noto_core::{BaseKind, NotoConfig};
use noto_runtime::ml::{DepthErrModel, DepthStandIn, KeyTruth, LocateErrModel, LocateStandIn, MlFixCfg, MlPipeline};
use noto_runtime::record::{intrinsics, odometry};
use noto_runtime::{DepthEngine, EmbeddingCfg, EmbeddingPerception, LocateEngine, NoiseCfg, OraclePerception, Recognizer, Runtime, RuntimeConfig, ScanInput};
use noto_search::{Lexicon, VisionPrompting};
use noto_vision::calib::{auroc, average_precision, fit_label};
use noto_vision::locate::points_with_presence;
use noto_vision::{
    depth_metrics, label_heats, AlignCfg, AlignSpace, CalibTable, DepthKind, DepthMap, DepthModel, Intrinsics, LabelPrompt, LabelTexts, LocateCfg, LocateScoring, MaskTracker, Pose, RegionTracker, Rgb,
    InputGrid, SamModel, SamTracker, SemanticLocator, SemanticPoint, TipsDepth,
};
use world_sim::render::pack_id;
use world_sim::world::{DropItem, Node};
use world_sim::{decode_webp, encode_webp, Camera, ItemKind, NodeKind, ObjectClass, TerrainKind, World};

const LEX: &str = include_str!("../../../data/world_lexicon.json");

// ----- references (benchmark side only; the engine never reads these) -----

/// Zero-shot relative depth on a rendered movie (Sintel) after per-image
/// affine alignment: DPT-Large AbsRel 0.270 (Ranftl, Bochkovskiy & Koltun,
/// ICCV 2021, Table 1); Depth Anything ViT-L 0.458, MiDaS v3.1 0.587 (Yang et
/// al., CVPR 2024, Table 2). The renderer's oracle-aligned TIPS AbsRel is read
/// against the spread of these three.
const SINTEL_ALIGNED_ABSREL: (f32, f32) = (0.438, 0.160);
const CITE_SINTEL: &str = "렌더 영화(Sintel) 제로샷 깊이, 이미지별 아핀 정렬 후 AbsRel: DPT-L 0.270 (Ranftl 2021) · Depth Anything 0.458 · MiDaS 3.1 0.587 (Yang 2024)";
/// Scale from the known camera height over the ground plane against the
/// ground-truth median scale on KITTI: AbsRel 0.118 vs 0.113 (Xue et al.,
/// DNet, IROS 2020): a ratio of 1.044.
const DNET_GROUND_OVER_MEDIAN: f32 = 0.118 / 0.113;
const CITE_DNET: &str = "카메라 높이(지면)로 정한 축척 대 정답 중앙값 축척: KITTI AbsRel 0.118 / 0.113 = 1.04 (Xue 2020, DNet)";
/// Metric depth outside the training domain (no alignment): an NYU-finetuned
/// head on Hypersim AbsRel 0.363, delta1 0.361 (Yang et al. 2024, ZoeDepth framework).
const OFFDOMAIN_METRIC_ABSREL: f32 = 0.363;
const CITE_OFFDOMAIN: &str = "학습 영역 밖 미터 깊이(정렬 없음): NYU 미세조정 헤드를 Hypersim에 AbsRel 0.363 · δ1 0.361 (Yang 2024)";
/// Training-free patch-text location quality: raw CLIP patch tokens 14.1 mIoU
/// against 30.3 for the MaskCLIP value path (Wang, Mei & Yuille, SCLIP, ECCV
/// 2024, 8-benchmark average) — the value path more than doubles a raw readout.
const CITE_SCLIP: &str = "학습 없는 패치-텍스트 위치: CLIP 패치 원시 14.1 대 MaskCLIP 값 경로 30.3 mIoU (Wang 2024, SCLIP 8개 평균)";
/// Prompt ensembling: about +3.5 points of zero-shot accuracy on top of a
/// single template (Radford et al., CLIP, ICML 2021).
const CITE_ENSEMBLE: &str = "프롬프트 앙상블: 단일 템플릿보다 제로샷 정확도 약 +3.5점 (Radford 2021, CLIP)";
/// Zero-shot calibration: ECE 6.3–6.7% before temperature scaling, 1.4–2.2%
/// after (LeVine et al. 2023, LAION ViT-B/16 and ViT-L/14).
const CITE_CALIB: &str = "제로샷 보정: 온도 보정 전 ECE 6.3–6.7% → 후 1.4–2.2% (LeVine 2023)";

// ----- the grid -----

const TARGETS: [&str; 9] = ["tree", "boulder", "coal_ore", "iron_ore_node", "log", "stone", "coal", "iron_ore", "parcel"];
/// A label counts as in view from this many pixels (full keyframe), and as
/// absent at zero; in between it is not scored for presence.
const PRESENT_PX: u32 = 16;
/// Calibration targets of the A5 gate: a threshold at this precision on the
/// fit seeds, else at this false-positive rate. Engineering choices.
const CALIB_PRECISION: f32 = 0.9;
const CALIB_FPR: f32 = 0.05;
/// Scan gates as the runtime's defaults (`GateCfg`).
const GATE_SCORE: f32 = 0.30;
const GATE_MARGIN: f32 = 0.04;
/// One locator patch is about 20 px of a 384-px keyframe.
const PATCH_PX: f32 = 20.0;

#[derive(Clone, Debug)]
struct Scene {
    seed: u64,
    terrain: TerrainKind,
    base: BaseKind,
    /// What is placed and what the decision asks for (`query`).
    target: Option<&'static str>,
    query: &'static str,
    distractor: Option<&'static str>,
    dist_m: f32,
    tilt_deg: f32,
    lateral_m: f32,
}

impl Scene {
    fn fit(&self) -> bool {
        self.seed % 2 == 1
    }

    fn id(&self) -> String {
        format!("s{}-{:?}-{:?}-{}-{:.0}m-{:.0}deg", self.seed, self.terrain, self.base, self.target.unwrap_or("none"), self.dist_m, self.tilt_deg)
    }
}

fn env_list<T: std::str::FromStr>(key: &str, default: &[T]) -> Vec<T>
where
    T: Clone,
{
    match std::env::var(key) {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => default.to_vec(),
    }
}

fn look_alike_of(label: &str) -> &'static str {
    noto_runtime::perception::look_alike(label).unwrap_or("stone")
}

fn scenes() -> Vec<Scene> {
    let seeds: Vec<u64> = env_list("MLAB_SEEDS", &[1, 2, 3, 4]);
    let dists: Vec<f32> = env_list("MLAB_DISTS", &[2.0, 4.0, 7.0]);
    let tilts: Vec<f32> = env_list("MLAB_TILTS", &[0.0, 30.0]);
    let mut out = Vec::new();
    for &seed in &seeds {
        // Seeds 1–2 legs, 3–4 wheels; odd seeds fit the calibration, even ones score it.
        let base = if seed <= 2 { BaseKind::HumanFoot } else { BaseKind::Wheel };
        for terrain in [TerrainKind::Open, TerrainKind::Rugged] {
            for (k, &t) in TARGETS.iter().enumerate() {
                for &d in &dists {
                    for &tilt in &tilts {
                        let lateral = [-0.6f32, 0.0, 0.6][(k + seed as usize) % 3];
                        out.push(Scene { seed, terrain, base, target: Some(t), query: t, distractor: Some(look_alike_of(t)), dist_m: d, tilt_deg: tilt, lateral_m: lateral });
                    }
                }
            }
            // Nothing asked for is there: only a look-alike, or nothing at all.
            for (k, &tilt) in tilts.iter().enumerate() {
                for (j, q) in ["parcel", "log", "stone"].into_iter().enumerate() {
                    let distractor = if j == 2 { None } else { Some(look_alike_of(q)) };
                    out.push(Scene { seed, terrain, base, target: None, query: q, distractor, dist_m: 3.0 + j as f32, tilt_deg: tilt, lateral_m: 0.3 * k as f32 });
                }
            }
        }
    }
    out
}

fn base_world(seed: u64, terrain: TerrainKind, base: BaseKind) -> World {
    let cfg = RuntimeConfig { seed, noise: NoiseCfg::none(), parcels: Some(0), terrain, noto: NotoConfig { base, ..RuntimeConfig::default().noto }, ..RuntimeConfig::default() };
    Runtime::new(cfg, LEX, None).expect("runtime").world().clone()
}

/// Places a vision label at `(x, z)`; returns its packed id.
fn place(w: &mut World, label: &str, id: u32, x: f32, z: f32) -> u32 {
    let node = |k| Node { id, kind: k, x, z, hp: 10.0, max_hp: 10.0 };
    let item = |k| DropItem::whole(id, k, x, z);
    match label {
        "tree" => w.nodes.push(node(NodeKind::Tree)),
        "boulder" => w.nodes.push(node(NodeKind::Boulder)),
        "coal_ore" => w.nodes.push(node(NodeKind::CoalOre)),
        "iron_ore_node" => w.nodes.push(node(NodeKind::IronOre)),
        "log" => w.drops.push(item(ItemKind::Log)),
        "stone" => w.drops.push(item(ItemKind::Stone)),
        "coal" => w.drops.push(item(ItemKind::Coal)),
        "iron_ore" => w.drops.push(item(ItemKind::IronOre)),
        _ => w.drops.push(item(ItemKind::Parcel)),
    }
    let class = if ["tree", "boulder", "coal_ore", "iron_ore_node"].contains(&label) { ObjectClass::Node } else { ObjectClass::Drop };
    pack_id(class, id)
}

/// One captured scene: the keyframe as the runtime records it, plus what the
/// scan path needs (the full frame, the world, the camera).
struct Shot {
    rgb: Rgb,
    intr: Intrinsics,
    pose: Pose,
    truth: KeyTruth,
    world: World,
    frame: world_sim::Frame,
    cam: Camera,
    target_raw: Option<u32>,
}

fn shoot(bases: &mut BTreeMap<(u64, u8, u8), World>, s: &Scene) -> Shot {
    let key = (s.seed, matches!(s.terrain, TerrainKind::Rugged) as u8, matches!(s.base, BaseKind::Wheel) as u8);
    let mut w = bases.entry(key).or_insert_with(|| base_world(s.seed, s.terrain, s.base)).clone();
    w.nodes.clear();
    w.drops.clear();
    let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
    let at = |d: f32, lat: f32| (x + d * yaw.cos() - lat * yaw.sin(), z + d * yaw.sin() + lat * yaw.cos());
    let target_raw = s.target.map(|t| {
        let (px, pz) = at(s.dist_m, s.lateral_m);
        place(&mut w, t, 9001, px, pz)
    });
    if let Some(d) = s.distractor {
        let (px, pz) = at(s.dist_m + 1.5, s.lateral_m - 1.2);
        place(&mut w, d, 9002, px, pz);
    }
    let cam = Camera { pitch_deg: Camera::default().pitch_deg - s.tilt_deg, ..Camera::default() };
    // The runtime's capture: the renderer's frame, WebP q70 for the models, the truth at half size.
    let frame = w.render(&cam);
    let enc = encode_webp(&frame.rgb, frame.width as u32, frame.height as u32, 70.0).expect("webp");
    let (dw, dh, data) = decode_webp(&enc.bytes).expect("webp");
    let rgb = Rgb::new(dw, dh, data);
    let truth = KeyTruth::capture(&w, &frame);
    Shot { rgb, intr: intrinsics(&w, &cam), pose: odometry(&w, &cam), truth, world: w, frame, cam, target_raw }
}

// ----- the bookkeeping -----

#[derive(Default)]
struct Book {
    vals: BTreeMap<String, Vec<f32>>,
    pairs: BTreeMap<String, (Vec<f32>, Vec<bool>)>,
}

impl Book {
    fn add(&mut self, key: impl Into<String>, v: f32) {
        if v.is_finite() {
            self.vals.entry(key.into()).or_default().push(v);
        }
    }

    fn flag(&mut self, key: impl Into<String>, b: bool) {
        self.add(key, b as u8 as f32);
    }

    fn pair(&mut self, key: impl Into<String>, s: f32, hit: bool) {
        let e = self.pairs.entry(key.into()).or_default();
        e.0.push(s);
        e.1.push(hit);
    }

    fn mean(&self, key: &str) -> Option<(f32, usize)> {
        self.vals.get(key).filter(|v| !v.is_empty()).map(|v| (v.iter().sum::<f32>() / v.len() as f32, v.len()))
    }

    fn sd(&self, key: &str) -> Option<f32> {
        let v = self.vals.get(key).filter(|v| v.len() > 1)?;
        let m = v.iter().sum::<f32>() / v.len() as f32;
        Some((v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32).sqrt())
    }

    fn median(&self, key: &str) -> Option<f32> {
        let mut v = self.vals.get(key)?.clone();
        if v.is_empty() {
            return None;
        }
        v.sort_by(f32::total_cmp);
        Some(v[v.len() / 2])
    }

    fn line(&self, key: &str, note: &str) {
        match self.mean(key) {
            Some((m, n)) => println!("[MLAB] {key} {m:.3} (n {n}, 중앙값 {:.3}) — {note}", self.median(key).unwrap_or(f32::NAN)),
            None => println!("[MLAB] {key} 없음 — {note}"),
        }
    }

    fn rank(&self, key: &str) -> (Option<f32>, Option<f32>, usize, usize) {
        match self.pairs.get(key) {
            Some((s, h)) => (auroc(s, h), average_precision(s, h), h.iter().filter(|x| **x).count(), h.len()),
            None => (None, None, 0, 0),
        }
    }
}

// ----- depth -----

/// AbsRel, δ1, RMSE and coverage of `pred` on the truth pixels within `cap`,
/// plus AbsRel per true-range band and on the target's pixels.
struct DepthEval {
    absrel: f32,
    delta1: f32,
    rmse: f32,
    coverage: f32,
    bands: [Option<f32>; 5],
    target: Option<f32>,
}

const BANDS: [(f32, f32, &str); 5] = [(0.0, 2.0, "0-2"), (2.0, 4.0, "2-4"), (4.0, 6.0, "4-6"), (6.0, 10.0, "6-10"), (10.0, 1e9, ">10")];

fn depth_eval(pred: &DepthMap, truth: &KeyTruth, cap: f32, target_raw: Option<u32>) -> Option<DepthEval> {
    let (mut ar, mut d1, mut se, mut n, mut all) = (0.0f64, 0u32, 0.0f64, 0u32, 0u32);
    let mut band = [(0.0f64, 0u32); 5];
    let (mut tar, mut tn) = (0.0f64, 0u32);
    for y in 0..truth.height {
        for x in 0..truth.width {
            let i = (y * truth.width + x) as usize;
            let t = truth.depth[i];
            if !t.is_finite() || t <= 0.0 {
                continue;
            }
            let p = pred.at_scaled(x as f32 + 0.5, y as f32 + 0.5, truth.width, truth.height);
            let ok = p.is_finite() && p > 0.0;
            if ok {
                let e = ((p - t).abs() / t) as f64;
                if let Some(b) = BANDS.iter().position(|(lo, hi, _)| t >= *lo && t < *hi) {
                    band[b].0 += e;
                    band[b].1 += 1;
                }
                if target_raw == Some(truth.ids[i]) {
                    tar += e;
                    tn += 1;
                }
            }
            if t > cap {
                continue;
            }
            all += 1;
            if ok {
                ar += ((p - t).abs() / t) as f64;
                d1 += ((p / t).max(t / p) < 1.25) as u32;
                se += ((p - t) as f64).powi(2);
                n += 1;
            }
        }
    }
    (n > 0).then(|| DepthEval {
        absrel: (ar / n as f64) as f32,
        delta1: d1 as f32 / n as f32,
        rmse: (se / n as f64).sqrt() as f32,
        coverage: n as f32 / all.max(1) as f32,
        bands: band.map(|(s, c)| (c > 0).then(|| (s / c as f64) as f32)),
        target: (tn > 0).then(|| (tar / tn as f64) as f32),
    })
}

/// The best a per-frame scale and shift in disparity could do (truth in the loop: harness only).
fn oracle_lsq(pred: &DepthMap, truth: &KeyTruth, cap: f32) -> DepthMap {
    let (mut sp, mut sq, mut spp, mut spq, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for y in 0..truth.height {
        for x in 0..truth.width {
            let t = truth.depth[(y * truth.width + x) as usize];
            let p = pred.at_scaled(x as f32 + 0.5, y as f32 + 0.5, truth.width, truth.height);
            if t.is_finite() && t > 0.0 && t <= cap && p.is_finite() && p > 0.0 {
                let (p, q) = (1.0 / p as f64, 1.0 / t as f64);
                sp += p;
                sq += q;
                spp += p * p;
                spq += p * q;
                n += 1.0;
            }
        }
    }
    let den = n * spp - sp * sp;
    let (a, b) = if den.abs() > 1e-12 { let a = (n * spq - sp * sq) / den; (a, (sq - a * sp) / n) } else { (1.0, 0.0) };
    let mut out = pred.clone();
    for d in out.data.iter_mut() {
        let q = a / *d as f64 + b;
        *d = if d.is_finite() && q > 1e-4 { (1.0 / q) as f32 } else { f32::INFINITY };
    }
    out
}

fn record_depth(book: &mut Book, arm: &str, pred: &DepthMap, truth: &KeyTruth, target_raw: Option<u32>) {
    if let Some(e) = depth_eval(pred, truth, 10.0, target_raw) {
        book.add(format!("{arm}/absrel"), e.absrel);
        book.add(format!("{arm}/delta1"), e.delta1);
        book.add(format!("{arm}/rmse_m"), e.rmse);
        book.add(format!("{arm}/coverage"), e.coverage);
        for (b, v) in BANDS.iter().zip(e.bands) {
            if let Some(v) = v {
                book.add(format!("{arm}/band{}", b.2), v);
            }
        }
        if let Some(t) = e.target {
            book.add(format!("{arm}/target_absrel"), t);
        }
    }
}

/// Pixel samples of `ln(pred/true)` against `ln(true)`, summed per frame, for
/// an error model of the depth model: `pred = e^(a + η + ε) · true^(1+b)`
/// with a per-frame offset η and a per-pixel residual ε (the honest depth
/// stand-in, `DepthErrModel`). Also what the model answers where the truth
/// is sky.
#[derive(Default)]
struct ErrFit {
    /// Per frame: n, Σx, Σy, Σxx, Σxy, Σyy with x = ln t, y = ln(p/t).
    frames: Vec<[f64; 6]>,
    sky: Vec<f32>,
    /// Per metre of true depth (0–40 m): pixels answered, pixels "unknown".
    known: Vec<(u32, u32)>,
}

impl ErrFit {
    fn add_frame(&mut self, pred: &DepthMap, truth: &KeyTruth, cap: f32) {
        let mut s = [0.0f64; 6];
        for y in (0..truth.height).step_by(2) {
            for x in (0..truth.width).step_by(2) {
                let t = truth.depth[(y * truth.width + x) as usize];
                let p = pred.at_scaled(x as f32 + 0.5, y as f32 + 0.5, truth.width, truth.height);
                if !t.is_finite() {
                    // What the model says where the truth is sky (INFINITY: unknown).
                    if p > 0.0 {
                        self.sky.push(p);
                    }
                    continue;
                }
                if self.known.is_empty() {
                    self.known = vec![(0, 0); 40];
                }
                if let Some(k) = self.known.get_mut(t as usize) {
                    if p.is_finite() && p > 0.0 {
                        k.0 += 1;
                    } else {
                        k.1 += 1;
                    }
                }
                if !(p.is_finite() && p > 0.0) {
                    continue;
                }
                if t <= 0.05 || t > cap {
                    continue;
                }
                let (lx, ly) = ((t as f64).ln(), ((p / t) as f64).ln());
                for (k, v) in [1.0, lx, ly, lx * lx, lx * ly, ly * ly].into_iter().enumerate() {
                    s[k] += v;
                }
            }
        }
        if s[0] >= 50.0 {
            self.frames.push(s);
        }
    }

    /// `(a, b, frame sd, pixel sd, median sky answer)`.
    fn fit(&self) -> Option<(f64, f64, f64, f64, f32)> {
        let mut t = [0.0f64; 6];
        for f in &self.frames {
            for k in 0..6 {
                t[k] += f[k];
            }
        }
        let n = t[0];
        let den = n * t[3] - t[1] * t[1];
        if self.frames.len() < 2 || den.abs() < 1e-9 {
            return None;
        }
        let b = (n * t[4] - t[1] * t[2]) / den;
        let a = (t[2] - b * t[1]) / n;
        let (mut between, mut within, mut wn) = (Vec::new(), 0.0f64, 0.0f64);
        for f in &self.frames {
            let (nf, mx, my) = (f[0], f[1] / f[0], f[2] / f[0]);
            between.push(my - a - b * mx);
            // Σ(y − b·x − (ȳ − b·x̄))² from the sums.
            within += (f[5] - 2.0 * b * f[4] + b * b * f[3]) - nf * (my - b * mx).powi(2);
            wn += nf;
        }
        let m = between.iter().sum::<f64>() / between.len() as f64;
        let tau = (between.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (between.len() - 1) as f64).sqrt();
        let mut sky = self.sky.clone();
        sky.sort_by(f32::total_cmp);
        Some((a, b, tau, (within / wn.max(1.0)).max(0.0).sqrt(), sky.get(sky.len() / 2).copied().unwrap_or(f32::INFINITY)))
    }

    /// The nearest true depth (whole metres, at least 50 pixels) from which
    /// most pixels are "unknown".
    fn unknown_beyond(&self) -> f32 {
        self.known.iter().enumerate().find(|(_, (k, u))| k + u >= 50 && u > k).map_or(f32::INFINITY, |(m, _)| m as f32)
    }

    /// The fit as the honest stand-in's error model (`cap_m`: the model's
    /// ceiling; the blur radius is the stand-in's estimate).
    fn model(&self, cap_m: f32) -> Option<DepthErrModel> {
        let (a, b, tau, sigma, sky) = self.fit()?;
        Some(DepthErrModel { a: a as f32, b: b as f32, frame_sd: tau as f32, patch_sd: sigma as f32, sky_m: sky, cap_m, unknown_beyond_m: self.unknown_beyond(), ..DepthErrModel::TIPS_GROUND })
    }
}

// ----- given models: one computation shared by several arms -----

/// A depth "model" that returns the map the harness computed for this frame.
struct GivenDepth {
    mode: String,
    max: Option<f32>,
    map: Arc<Mutex<Option<DepthMap>>>,
}

impl DepthModel for GivenDepth {
    fn mode(&self) -> &str {
        &self.mode
    }
    fn max_depth_m(&self) -> Option<f32> {
        self.max
    }
    fn infer(&mut self, _rgb: &Rgb) -> anyhow::Result<DepthMap> {
        Ok(self.map.lock().unwrap().clone().expect("depth set for this frame"))
    }
}

/// A locator that returns the points the harness computed for this frame.
struct GivenPoints {
    mode: String,
    points: Arc<Mutex<Vec<SemanticPoint>>>,
}

impl SemanticLocator for GivenPoints {
    fn mode(&self) -> &str {
        &self.mode
    }
    fn locate(&mut self, _rgb: &Rgb, _labels: &[LabelPrompt]) -> anyhow::Result<Vec<SemanticPoint>> {
        Ok(self.points.lock().unwrap().clone())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Src {
    StandIn,
    Model,
}

struct Arm {
    name: &'static str,
    pipe: MlPipeline,
    depth: Src,
    locate: Src,
    /// Error models of honest stand-ins (fitted on the fit seeds).
    honest: Option<(DepthErrModel, LocateErrModel)>,
}

fn arm(name: &'static str, lex: &Lexicon, depth: Src, locate: Src, tracker: Box<dyn MaskTracker>, d: &Arc<Mutex<Option<DepthMap>>>, p: &Arc<Mutex<Vec<SemanticPoint>>>, dmode: &str, lmode: &str, fix: MlFixCfg) -> Arm {
    let mut pipe = MlPipeline::stand_in(lex, 7);
    if depth == Src::Model {
        pipe.depth = DepthEngine::Model(Box::new(GivenDepth { mode: dmode.into(), max: Some(10.0), map: Arc::clone(d) }));
    }
    if locate == Src::Model {
        pipe.locate = LocateEngine::Model(Box::new(GivenPoints { mode: lmode.into(), points: Arc::clone(p) }));
    }
    pipe.tracker = tracker;
    pipe.fix = fix;
    Arm { name, pipe, depth, locate, honest: None }
}

/// Runs one arm's chain on a shot and books the tracker and range numbers.
fn run_arm(book: &mut Book, a: &mut Arm, shot: &Shot, scene: &Scene, height: Option<noto_vision::align::HeightFn>) {
    a.pipe.reset();
    // The stand-ins are re-seeded per scene so every arm gets the same draws.
    if a.depth == Src::StandIn {
        let mut d = DepthStandIn::new(scene.seed * 1000 + scene.dist_m as u64);
        d.err = a.honest.map(|h| h.0);
        a.pipe.depth = DepthEngine::StandIn(d);
    }
    if a.locate == Src::StandIn {
        let mut l = LocateStandIn::new(scene.seed * 1000 + scene.dist_m as u64);
        l.err = a.honest.map(|h| h.1);
        a.pipe.locate = LocateEngine::StandIn(l);
    }
    // The runtime's default floors (`mlsds::default_floor`): the chosen stand-in's
    // 0.35; the real locators and the honest stand-in are gated by calibration.
    let floor_default = if a.locate == Src::StandIn && a.honest.is_none() { 0.35 } else { 0.0 };
    let wanted = vec![scene.query.to_string()];
    let out = a.pipe.analyse_with(&shot.rgb, &shot.intr, &shot.pose, Some(&shot.truth), &wanted, &|_| floor_default, None, height).expect("analyse");
    let name = a.name;
    let present = shot.target_raw.and_then(|raw| shot.truth.objects.iter().find(|o| pack_raw(o) == Some(raw))).map_or(0, |o| o.pixels);
    let tracked = out.target.is_some();
    if scene.target.is_some() && present >= PRESENT_PX {
        book.flag(format!("{name}/track/target_found"), tracked);
        let right = out.truth_entity.as_deref() == Some("n9001") || out.truth_entity.as_deref() == Some("d9001");
        book.flag(format!("{name}/track/right_object"), tracked && right);
        if let (true, Some(iou)) = (right, out.track_iou_truth) {
            book.add(format!("{name}/track/iou_truth"), iou);
        }
        if let (true, Some(t), Some(r)) = (right, out.target.as_ref(), out.range_true_m) {
            if let Some(d) = t.depth_range_m {
                book.add(format!("{name}/range/rel_err"), (d - r).abs() / r);
                book.add(format!("{name}/range/signed_rel"), (d - r) / r);
                if let Some(sg) = t.range_sigma_m {
                    book.add(format!("{name}/range/z"), (d - r).abs() / sg.max(1e-3));
                }
            }
        }
    }
    if scene.target.is_none() {
        // Asked for something that is not there: any track is a phantom.
        book.flag(format!("{name}/track/phantom_when_absent"), tracked);
    }
    if let Some(f) = out.align {
        book.add(format!("{name}/align/inlier_share"), f.inlier_share);
        book.add(format!("{name}/align/a"), f.a);
        book.add(format!("{name}/align/b"), f.b);
    }
    if a.depth == Src::Model {
        book.flag(format!("{name}/align/fitted"), out.align.is_some());
    }
}

fn pack_raw(o: &world_sim::VisibleObject) -> Option<u32> {
    let id: u32 = o.entity.get(1..)?.parse().ok()?;
    let class = match o.class {
        ObjectClass::Node => ObjectClass::Node,
        ObjectClass::Drop => ObjectClass::Drop,
        _ => return None,
    };
    Some(pack_id(class, id))
}

// ----- semantic location -----

/// Presence label of every target label in a shot: `Some(true)` in view
/// (≥ PRESENT_PX), `Some(false)` absent, `None` too small to say.
fn presence(shot: &Shot, label: &str) -> Option<bool> {
    let px = shot.truth.objects.iter().filter(|o| o.label == label).map(|o| o.pixels).max().unwrap_or(0);
    if px >= PRESENT_PX {
        Some(true)
    } else if px == 0 {
        Some(false)
    } else {
        None
    }
}

fn peak_in_box(shot: &Shot, p: &SemanticPoint) -> bool {
    shot.truth.objects.iter().filter(|o| o.label == p.label && o.pixels >= PRESENT_PX).any(|o| {
        let b = o.bbox;
        p.px >= b[0] as f32 - PATCH_PX && p.px <= b[2] as f32 + 1.0 + PATCH_PX && p.py >= b[1] as f32 - PATCH_PX && p.py <= b[3] as f32 + 1.0 + PATCH_PX
    })
}

fn locate_err(shot: &Shot, p: &SemanticPoint) -> Option<f32> {
    shot.truth
        .objects
        .iter()
        .filter(|o| o.label == p.label && o.pixels >= PRESENT_PX)
        .map(|o| {
            let (cx, cy) = ((o.bbox[0] + o.bbox[2]) as f32 * 0.5, (o.bbox[1] + o.bbox[3]) as f32 * 0.5);
            (cx - p.px).hypot(cy - p.py)
        })
        .min_by(f32::total_cmp)
}

/// Books the presence statistics and the peak hit of every label for one locator variant.
fn record_locate(book: &mut Book, variant: &str, shot: &Shot, scene: &Scene, points: &[SemanticPoint], per_frame: &mut Vec<serde_json::Value>) {
    for label in TARGETS {
        let Some(present) = presence(shot, label) else { continue };
        let p = points.iter().find(|p| p.label == label);
        let (peak, pres) = p.map_or((0.0, 0.0), |p| (p.score, p.presence));
        let split = if scene.fit() { "fit" } else { "eval" };
        for (stat, v) in [("peak", peak), ("presence", pres), ("peak_x_presence", peak * pres)] {
            book.pair(format!("{variant}/{split}/{stat}/{label}"), v, present);
            book.pair(format!("{variant}/{split}/{stat}/all"), v, present);
        }
        if present {
            if let Some(p) = p {
                book.flag(format!("{variant}/peak_in_box"), peak_in_box(shot, p));
                if let Some(e) = locate_err(shot, p) {
                    book.add(format!("{variant}/locate_err_px"), e);
                }
            }
        }
        let px = shot.truth.objects.iter().filter(|o| o.label == label).map(|o| o.pixels).max().unwrap_or(0);
        per_frame.push(serde_json::json!({"variant": variant, "scene": scene.id(), "fit": scene.fit(), "label": label, "present": present, "px": px, "peak": peak, "presence": pres, "in_box": p.map(|p| peak_in_box(shot, p))}));
    }
}

// ----- the tile scan -----

/// One scan candidate as the calibration sees it: its label, raw (or
/// calibrated) score, margin, and whether it lies within the gates' 1 m of a
/// visible object of that label.
struct ScanCand {
    label: String,
    score: f32,
    tp: bool,
}

/// What one scan did against the truth: every candidate, every visible
/// target object (its pixels and whether a passing candidate named it within
/// 1 m), and the passing candidates' outcomes.
#[derive(Default)]
struct ScanRec {
    cands: Vec<ScanCand>,
    objects: Vec<(u32, bool)>,
    false_pass: u32,
    pos_err: Vec<f32>,
    /// (score, margin) of passing candidates that were right, and wrong.
    pass_true: Vec<(f32, f32)>,
    pass_false: Vec<(f32, f32)>,
}

fn record_scan(book: &mut Book, name: &str, shot: &Shot, rec: &mut dyn Recognizer) -> ScanRec {
    let out = rec.scan(ScanInput { world: &shot.world, frame: &shot.frame, camera: &shot.cam }).expect("scan");
    let present: Vec<(&String, f32, f32, u32)> = shot
        .truth
        .positions
        .iter()
        .filter_map(|(e, l, x, z)| {
            let px = shot.truth.objects.iter().filter(|o| &o.entity == e).map(|o| o.pixels).max()?;
            (TARGETS.contains(&l.as_str()) && px >= PRESENT_PX).then_some((l, *x, *z, px))
        })
        .collect();
    let near = |label: &str, w: Option<(f32, f32)>| -> Option<(usize, f32)> {
        let (cx, cz) = w?;
        present.iter().enumerate().filter(|(_, (l, _, _, _))| l.as_str() == label).map(|(i, (_, x, z, _))| (i, (x - cx).hypot(z - cz))).min_by(|a, b| a.1.total_cmp(&b.1)).filter(|(_, d)| *d <= 1.0)
    };
    let mut r = ScanRec::default();
    let mut matched = vec![false; present.len()];
    for c in out.candidates.iter().filter(|c| TARGETS.contains(&c.label.as_str())) {
        let hit = near(&c.label, c.world);
        r.cands.push(ScanCand { label: c.label.clone(), score: c.score, tp: hit.is_some() });
        if c.score < GATE_SCORE || c.margin < GATE_MARGIN {
            continue;
        }
        book.flag(format!("{name}/scan/precision"), hit.is_some());
        match hit {
            Some((i, d)) => {
                book.add(format!("{name}/scan/pos_err_m"), d);
                matched[i] = true;
                r.pos_err.push(d);
                r.pass_true.push((c.score, c.margin));
            }
            None => {
                r.false_pass += 1;
                r.pass_false.push((c.score, c.margin));
            }
        }
    }
    book.add(format!("{name}/scan/false_per_scan"), r.false_pass as f32);
    for (m, p) in matched.iter().zip(&present) {
        book.flag(format!("{name}/scan/recall"), *m);
        r.objects.push((p.3, *m));
    }
    r
}

/// The honest scan stand-in's error model from scans of the real recogniser.
#[derive(Default)]
struct ScanFit {
    lnpx: Vec<f32>,
    det: Vec<bool>,
    scans: u32,
    false_pass: u32,
    pos: Vec<f32>,
    pass_true: Vec<(f32, f32)>,
    pass_false: Vec<(f32, f32)>,
}

impl ScanFit {
    fn add(&mut self, r: &ScanRec) {
        self.scans += 1;
        self.false_pass += r.false_pass;
        for (px, d) in &r.objects {
            self.lnpx.push((*px.max(&1) as f32).ln());
            self.det.push(*d);
        }
        self.pos.extend(&r.pos_err);
        self.pass_true.extend(&r.pass_true);
        self.pass_false.extend(&r.pass_false);
    }

    fn model(&self) -> Option<noto_runtime::perception::ScanErrModel> {
        if self.scans == 0 || self.det.is_empty() {
            return None;
        }
        let pos = self.det.iter().filter(|d| **d).count();
        // Never or always detected: a flat logistic at the observed (smoothed) rate.
        let (det_a, det_b) = match noto_vision::calib::platt_fit(&self.lnpx, &self.det) {
            Some(f) => f,
            None => {
                let p = (pos as f32 + 0.5) / (self.det.len() as f32 + 1.0);
                (0.0, (p / (1.0 - p)).ln())
            }
        };
        let ms = |v: &[f32], dflt: (f32, f32)| -> (f32, f32) {
            if v.is_empty() {
                return dflt;
            }
            let m = v.iter().sum::<f32>() / v.len() as f32;
            let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32).sqrt() } else { 0.0 };
            (m, sd)
        };
        let st: Vec<f32> = self.pass_true.iter().map(|x| x.0).collect();
        let sf: Vec<f32> = self.pass_false.iter().map(|x| x.0).collect();
        let mg: Vec<f32> = self.pass_true.iter().chain(&self.pass_false).map(|x| x.1).collect();
        // Radial errors of a 2-D Gaussian: E[d²] = 2σ².
        let pos_sd_m = if self.pos.is_empty() { 0.0 } else { (self.pos.iter().map(|d| d * d).sum::<f32>() / (2.0 * self.pos.len() as f32)).sqrt() };
        Some(noto_runtime::perception::ScanErrModel {
            det_a,
            det_b,
            false_per_scan: self.false_pass as f32 / self.scans as f32,
            pos_sd_m,
            score_true: ms(&st, (GATE_SCORE, 0.0)),
            score_false: ms(&sf, (GATE_SCORE, 0.0)),
            margin: ms(&mg, (GATE_MARGIN, 0.0)),
            score_floor: GATE_SCORE,
            margin_floor: GATE_MARGIN,
        })
    }
}

/// Per-label Platt maps of the tile score, with a pooled `*` entry for labels
/// with too few candidates of either kind.
fn fit_scan_calibration(cands: &[ScanCand]) -> CalibTable {
    let mut t = CalibTable { mode: noto_runtime::embedding::SCAN_CALIB_MODE.into(), statistic: "tile_score".into(), note: "ml_ab.rs 맞춤 시드: 후보가 1 m 안의 같은 라벨 물체인 확률의 Platt 맞춤".into(), ..CalibTable::default() };
    let logit = |p: f32| (p / (1.0 - p)).ln();
    let mut add = |key: &str, s: Vec<f32>, h: Vec<bool>| {
        let pos = h.iter().filter(|x| **x).count();
        if h.len() < 10 {
            return;
        }
        // A score that does not rank right candidates above wrong ones (no
        // positive slope), or too few right ones to tell, says nothing: the
        // label's base rate, flat.
        let rate = (pos as f32 + 0.5) / (h.len() as f32 + 1.0);
        let flat = (0.0f32, logit(rate));
        let fit = if pos >= 5 { noto_vision::calib::platt_fit(&s, &h).filter(|f| f.0 > 0.0) } else { None };
        if let Some((a, b)) = fit.or(Some(flat)) {
            // The raw score at which the probability reaches the gates' floor
            // (JSON has no infinity: a fit that never gets there says f32::MAX).
            let threshold = if a > 0.0 { (logit(GATE_SCORE) - b) / a } else { f32::MAX };
            t.labels.insert(key.into(), noto_vision::LabelCalib { threshold, platt: Some((a, b)), positives: pos as u32, negatives: (h.len() - pos) as u32, precision: None, recall: None });
        }
    };
    add("*", cands.iter().map(|c| c.score).collect(), cands.iter().map(|c| c.tp).collect());
    for l in TARGETS {
        let (s, h): (Vec<f32>, Vec<bool>) = cands.iter().filter(|c| c.label == l).map(|c| (c.score, c.tp)).unzip();
        add(l, s, h);
    }
    t
}

/// The honest locate stand-in's outcome model from the fit seeds' records of
/// one locator variant, gated by that variant's table: how often a label in
/// view gets a passing point on its object (logistic in ln pixels), how often
/// a passing point lands elsewhere or appears for an absent label, and the
/// calibrated scores of passing and blocked points.
fn locate_errmodel(per_frame: &[serde_json::Value], variant: &str, table: &CalibTable) -> Option<LocateErrModel> {
    let (mut lnpx, mut hits) = (Vec::new(), Vec::new());
    let (mut miss_pass, mut absent_pass) = (Book::default(), Book::default());
    let (mut pass, mut block) = (Book::default(), Book::default());
    for r in per_frame.iter().filter(|r| r["variant"] == variant && r["fit"] == true) {
        let label = r["label"].as_str()?;
        let Some(c) = table.labels.get(label) else { continue };
        let (peak, pres) = (r["peak"].as_f64()? as f32, r["presence"].as_f64()? as f32);
        let stat = match table.statistic.as_str() {
            "presence" => pres,
            "peak_x_presence" => peak * pres,
            _ => peak,
        };
        let passed = stat >= c.threshold;
        if let Some(p) = c.probability(stat) {
            if passed { pass.add("p", p) } else { block.add("p", p) }
        }
        if r["present"] == true {
            let on = passed && r["in_box"].as_bool().unwrap_or(false);
            lnpx.push((r["px"].as_u64()?.max(1) as f32).ln());
            hits.push(on);
            if !on {
                miss_pass.flag("p", passed);
            }
        } else {
            absent_pass.flag("p", passed);
        }
    }
    let pos = hits.iter().filter(|h| **h).count();
    let (hit_a, hit_b) = match noto_vision::calib::platt_fit(&lnpx, &hits) {
        Some(f) => f,
        None if hits.is_empty() => return None,
        None => {
            let p = (pos as f32 + 0.5) / (hits.len() as f32 + 1.0);
            (0.0, (p / (1.0 - p)).ln())
        }
    };
    let ms = |b: &Book, dflt: (f32, f32)| b.mean("p").map_or(dflt, |(m, _)| (m, b.sd("p").unwrap_or(0.0)));
    // The probability each label's threshold stands for, averaged.
    let at: Vec<f32> = table.labels.values().filter(|c| c.threshold.is_finite() && c.threshold < f32::MAX).filter_map(|c| c.probability(c.threshold)).collect();
    let gate_p = if at.is_empty() { 0.5 } else { at.iter().sum::<f32>() / at.len() as f32 };
    Some(LocateErrModel {
        hit_a,
        hit_b,
        pass_miss: miss_pass.mean("p").map_or(0.0, |x| x.0),
        pass_absent: absent_pass.mean("p").map_or(0.0, |x| x.0),
        score_pass: ms(&pass, (gate_p, 0.0)),
        score_block: ms(&block, (0.0, 0.0)),
        gate_p,
    })
}

// ----- the harness -----

fn device() -> candle_core::Device {
    noto_ml::preferred().backend.device(0).expect("device")
}

fn bank_dir() -> PathBuf {
    match std::env::var("RUNTIME_DIR") {
        Ok(d) => PathBuf::from(d).join("bank"),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/runtime/bank"),
    }
}

fn out_dir() -> Option<PathBuf> {
    std::env::var("MLAB_OUT").ok().map(PathBuf::from)
}

fn print_rank(book: &Book, variant: &str, stat: &str) {
    for split in ["fit", "eval"] {
        let (au, ap, pos, n) = book.rank(&format!("{variant}/{split}/{stat}/all"));
        println!("[MLAB] {variant}/{split}/{stat} AUROC {} · AP {} (양성 {pos}/{n}) — 라벨 합산", au.map_or("-".into(), |v| format!("{v:.3}")), ap.map_or("-".into(), |v| format!("{v:.3}")));
    }
    let per: Vec<String> = TARGETS
        .iter()
        .filter_map(|l| {
            let (au, _, pos, _) = book.rank(&format!("{variant}/eval/{stat}/{l}"));
            au.map(|a| format!("{l} {a:.2}({pos})"))
        })
        .collect();
    println!("[MLAB] {variant}/eval/{stat} 라벨별 AUROC: {}", per.join(" · "));
}

/// SigLIP2 behind a lock, so one loaded model serves the old and the v1.0
/// port (the position filter is switched between them on the same frame).
struct Sig(Arc<RwLock<noto_ml::Siglip2>>);

impl noto_ml::ImageEmbedder for Sig {
    fn embedding_dim(&self) -> usize {
        self.0.read().unwrap().embedding_dim()
    }
    fn embed_images_batch(&self, images: &[image::DynamicImage]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.0.read().unwrap().embed_images(images, 0)
    }
}

impl noto_ml::TextEmbedder for Sig {
    fn embedding_dim(&self) -> usize {
        self.0.read().unwrap().embedding_dim()
    }
    fn fingerprint(&self) -> u64 {
        self.0.read().unwrap().fingerprint()
    }
    fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.0.read().unwrap().embed_texts(&refs)
    }
}

/// Wall time per stage, seconds.
#[derive(Default)]
struct Timers(BTreeMap<&'static str, f64>);

impl Timers {
    fn time<T>(&mut self, k: &'static str, f: impl FnOnce() -> T) -> T {
        let t = std::time::Instant::now();
        let out = f();
        *self.0.entry(k).or_default() += t.elapsed().as_secs_f64();
        out
    }
}

#[test]
#[ignore]
fn ml_ab_benchmark() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let lex = Lexicon::from_json(LEX).unwrap();
    let prompting = VisionPrompting::from_lexicon_json(LEX).unwrap();
    let siglip = noto_ml::Siglip2::load(&PathBuf::from(std::env::var("SIGLIP2_DIR").expect("SIGLIP2_DIR")), &dev).expect("SigLIP2");
    let mut tips = TipsDepth::load(&PathBuf::from(std::env::var("TIPS_DIR").expect("TIPS_DIR")), &dev).expect("TIPS");
    let sam = Arc::new(SamModel::load(&PathBuf::from(std::env::var("SAM_DIR").expect("SAM_DIR")), &dev).expect("SAM"));
    let (scale, bias) = siglip.logit_scale_bias();
    println!("[MLAB] 장치 {} · SigLIP2 exp(t) {scale:.1}, b {bias:.2} (짝 확률 0.5인 코사인 {:.3})", noto_ml::device_label(&dev), -bias / scale);
    let mut scenes = scenes();
    // Fit seeds first: the gate and the scan calibration are fitted on them before the others are scored.
    scenes.sort_by_key(|s| !s.fit());
    let n_fit = scenes.iter().filter(|s| s.fit()).count();
    println!("[MLAB] 장면 {}개 (맞춤 {n_fit}, 평가 {}) · 시드 {:?}", scenes.len(), scenes.len() - n_fit, env_list::<u64>("MLAB_SEEDS", &[1, 2, 3, 4]));
    let labels: Vec<LabelPrompt> = MlPipeline::stand_in(&lex, 7).labels().to_vec();

    // The text side of every locator variant (text vectors do not depend on the image tower's switch).
    let variants: Vec<(&str, LocateCfg)> = vec![
        ("loc/old_pairprob", LocateCfg::default()),
        ("loc/new_pairprob", LocateCfg::default()),
        ("loc/new_pairprob_tpl", LocateCfg { templates: prompting.templates.clone(), ..LocateCfg::default() }),
        ("loc/new_contrast", LocateCfg { scoring: LocateScoring::Contrast, background: prompting.background.clone(), ..LocateCfg::default() }),
        ("loc/new_contrast_tpl", LocateCfg { scoring: LocateScoring::Contrast, background: prompting.background.clone(), templates: prompting.templates.clone(), ..LocateCfg::default() }),
        ("loc/new_contrast_nobg", LocateCfg { scoring: LocateScoring::Contrast, ..LocateCfg::default() }),
        ("loc/new_z", LocateCfg { scoring: LocateScoring::ZScore, ..LocateCfg::default() }),
    ];
    let siglip = Arc::new(siglip);
    let texts: Vec<LabelTexts> = {
        let mut loc = noto_vision::SiglipLocator::with_cfg(Arc::clone(&siglip), LocateCfg::default());
        variants.iter().map(|(_, c)| loc.label_texts(&labels, c).unwrap()).collect()
    };
    let sig = Arc::new(Sig(Arc::new(RwLock::new(Arc::try_unwrap(siglip).ok().expect("the locator is gone")))));
    let set_port = |sig: &Sig, tips: &mut TipsDepth, v1: bool| {
        sig.0.write().unwrap().set_position_antialias(v1);
        tips.set_readout_gelu_tanh(v1);
    };
    let bank = bank_dir();
    let phrases = noto_runtime::BankedPhrases::new(Arc::clone(&sig), Some(&bank));
    let calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync> = Arc::new(move |c| 1.0 / (1.0 + (-(scale * c + bias)).exp()));
    let image: Arc<dyn noto_ml::ImageEmbedder + Send + Sync> = sig.clone();
    let mut scan = EmbeddingPerception::new_with(&lex, &phrases, Arc::clone(&image), Arc::clone(&calibrate), "gpu", EmbeddingCfg::default(), None).unwrap();
    let mut scan_tpl = EmbeddingPerception::new_with(&lex, &phrases, Arc::clone(&image), Arc::clone(&calibrate), "gpu", EmbeddingCfg::default(), Some(&prompting)).unwrap();
    let mut scan_cal = EmbeddingPerception::new_with(&lex, &phrases, Arc::clone(&image), Arc::clone(&calibrate), "gpu", EmbeddingCfg::default(), None).unwrap();
    let mut oracle = OraclePerception::new(&lex, NoiseCfg::demo(), 6);

    let d_cell: Arc<Mutex<Option<DepthMap>>> = Arc::new(Mutex::new(None));
    let p_cell: Arc<Mutex<Vec<SemanticPoint>>> = Arc::new(Mutex::new(Vec::new()));
    let legacy = MlFixCfg::legacy();
    let mut arms = vec![
        arm("A0", &lex, Src::StandIn, Src::StandIn, Box::new(RegionTracker::default()), &d_cell, &p_cell, "", "", legacy.clone()),
        arm("A1", &lex, Src::Model, Src::StandIn, Box::new(RegionTracker::default()), &d_cell, &p_cell, "tipsv2-b14-dpt", "", legacy.clone()),
        arm("A2", &lex, Src::StandIn, Src::Model, Box::new(RegionTracker::default()), &d_cell, &p_cell, "", "siglip2-dense", legacy.clone()),
        arm("A3", &lex, Src::StandIn, Src::StandIn, Box::new(SamTracker::new(Arc::clone(&sam))), &d_cell, &p_cell, "", "", legacy.clone()),
        arm("A4", &lex, Src::Model, Src::Model, Box::new(SamTracker::new(Arc::clone(&sam))), &d_cell, &p_cell, "tipsv2-b14-dpt", "siglip2-dense", legacy.clone()),
    ];
    let mut a4n = arm("A4new", &lex, Src::Model, Src::Model, Box::new(SamTracker::new(Arc::clone(&sam))), &d_cell, &p_cell, "tipsv2-b14-dpt", "siglip2-dense", legacy.clone());
    // A shipped calibration (data/ml_calib.json with the dense gate and the
    // scan table) is used from the first scene; `MLAB_REFIT=1` or an empty
    // file fits both on the fit seeds instead.
    let shipped: Option<Vec<CalibTable>> = {
        let t = CalibTable::parse(noto_runtime::ml::ML_CALIB_JSON).unwrap_or_default();
        let dense = t.iter().any(|x| x.mode.starts_with("siglip2-dense"));
        let scan_t = t.iter().any(|x| x.mode == noto_runtime::embedding::SCAN_CALIB_MODE);
        (dense && scan_t && std::env::var("MLAB_REFIT").is_err()).then_some(t)
    };
    let mut a5: Option<(Arm, Arm, usize)> = None;
    let mut tables: Vec<CalibTable> = Vec::new();
    if let Some(t) = &shipped {
        let dense = t.iter().find(|x| x.mode.starts_with("siglip2-dense")).unwrap().clone();
        let scan_t = t.iter().find(|x| x.mode == noto_runtime::embedding::SCAN_CALIB_MODE).unwrap().clone();
        // Variant 0 is the old port; the shipped table belongs to the v1.0 port.
        let vk = (1..variants.len()).find(|k| variants[*k].1.mode() == dense.mode).unwrap_or_else(|| panic!("no locator variant scores like the shipped table {}", dense.mode));
        println!("[MLAB] 실린 보정 사용: 위치 {} · {} / 스캔 라벨 {}개", dense.mode, dense.statistic, scan_t.labels.len());
        scan_cal.set_calibration(Some(scan_t.clone()));
        tables = vec![dense, scan_t];
        a5 = Some(a5_arms(&lex, &prompting, &tables[0], vk, &sam, &d_cell, &p_cell));
    }
    let mut scan_honest_fit = ScanFit::default();
    let mut honest_scan: Option<OraclePerception> = None;

    let mut book = Book::default();
    let mut timers = Timers::default();
    let mut per_frame: Vec<serde_json::Value> = Vec::new();
    let mut bases: BTreeMap<(u64, u8, u8), World> = BTreeMap::new();
    let mut scan_fit: Vec<ScanCand> = Vec::new();
    let mut errfit = ErrFit::default();
    // The honest stand-ins' depth model: v1.0 TIPS after the ground alignment, fit seeds only.
    let mut errfit_ground = ErrFit::default();
    let mut honest: Vec<Arm> = Vec::new();
    for (i, s) in scenes.iter().enumerate() {
        // The fit stage, once the fit seeds are done: the gate and the scan
        // calibration (unless shipped) and the honest stand-ins.
        if i == n_fit {
            if a5.is_none() {
                let mut best: Option<(f32, usize, &str)> = None;
                for (k, (name, _)) in variants.iter().enumerate().skip(1) {
                    for stat in ["peak", "presence", "peak_x_presence"] {
                        if let (_, Some(ap), _, _) = book.rank(&format!("{name}/fit/{stat}/all")) {
                            if best.map_or(true, |b| ap > b.0) {
                                best = Some((ap, k, stat));
                            }
                        }
                    }
                }
                let (best_ap, vk, vstat) = best.expect("a locator statistic");
                let (vname, vcfg) = variants[vk].clone();
                println!("[MLAB] 게이트 통계: {vname} · {vstat} (맞춤 시드 AP {best_ap:.3})");
                let mut table = CalibTable { mode: vcfg.mode(), statistic: vstat.into(), note: format!("ml_ab.rs 맞춤 시드, 정밀도 {CALIB_PRECISION} (없으면 FPR {CALIB_FPR})"), ..CalibTable::default() };
                for l in TARGETS {
                    if let Some((sc, h)) = book.pairs.get(&format!("{vname}/fit/{vstat}/{l}")) {
                        if let Some(c) = fit_label(sc, h, CALIB_PRECISION, CALIB_FPR) {
                            table.labels.insert(l.to_string(), c);
                        }
                    }
                }
                let scan_table = fit_scan_calibration(&scan_fit);
                println!("[MLAB] 스캔 보정: 맞춤 후보 {}개 (참 {}), 라벨 {}개", scan_fit.len(), scan_fit.iter().filter(|c| c.tp).count(), scan_table.labels.len());
                scan_cal.set_calibration(Some(scan_table.clone()));
                tables = vec![table, scan_table];
                a5 = Some(a5_arms(&lex, &prompting, &tables[0], vk, &sam, &d_cell, &p_cell));
            }
            // The honest stand-ins, fitted on the fit seeds: the depth the v1.0
            // path hands on (TIPS after the ground alignment), where the gated
            // locator's peaks landed and how they scored through its table,
            // and (with a shipped scan table) what the calibrated scan did.
            let vk = a5.as_ref().map(|a| a.2).unwrap();
            let dmodel = errfit_ground.model(f32::INFINITY);
            let lmodel = locate_errmodel(&per_frame, variants[vk].0, &tables[0]);
            println!("[MLAB] 정직한 대역 (맞춤 시드): 깊이 {}", dmodel.map_or("없음".into(), |m| depth_model_rs(&m)));
            println!("[MLAB] 정직한 대역 (맞춤 시드): 위치 {lmodel:?}");
            if let (Some(dm), Some(lm)) = (dmodel, lmodel) {
                for (name, tracker) in [("A0h", Box::new(RegionTracker::default()) as Box<dyn MaskTracker>), ("A0hsam", Box::new(SamTracker::new(Arc::clone(&sam))))] {
                    let mut h = arm(name, &lex, Src::StandIn, Src::StandIn, tracker, &d_cell, &p_cell, "", "", legacy.clone());
                    h.honest = Some((dm, lm));
                    honest.push(h);
                }
            }
            if shipped.is_some() {
                let sm = scan_honest_fit.model();
                println!("[MLAB] 정직한 대역 (맞춤 시드): 스캔 {sm:?}");
                honest_scan = sm.map(|m| {
                    let mut o = OraclePerception::new(&lex, NoiseCfg::none(), 1);
                    o.set_honest(Some(m));
                    o
                });
            }
        }
        let shot = timers.time("render", || shoot(&mut bases, s));
        // ---------- the port as it was (A0–A4) ----------
        set_port(&*sig, &mut tips, false);
        let (raw, spread) = timers.time("tips", || tips.predict_batch_spread(&[&shot.rgb]).unwrap().pop().unwrap());
        timers.time("metrics", || {
            record_depth(&mut book, "depth/tips_raw", &raw, &shot.truth, shot.target_raw);
            if let Some(m) = depth_metrics(&raw, &shot.truth.depth_map(), 10.0, false) {
                book.add("depth/tips_raw/median_scale", m.median_scale);
                let mut d = raw.clone();
                d.data.iter_mut().for_each(|x| *x *= m.median_scale);
                record_depth(&mut book, "depth/tips_median", &d, &shot.truth, shot.target_raw);
            }
            record_depth(&mut book, "depth/tips_oracle_lsq", &oracle_lsq(&raw, &shot.truth, 10.0), &shot.truth, shot.target_raw);
            let si = DepthStandIn::new(s.seed * 1000 + s.dist_m as u64).run(&shot.truth);
            record_depth(&mut book, "depth/standin", &si, &shot.truth, shot.target_raw);
            // Does the bin spread flag the large errors (a pixel sample)?
            for y in (0..shot.truth.height).step_by(4) {
                for x in (0..shot.truth.width).step_by(4) {
                    let t = shot.truth.depth[(y * shot.truth.width + x) as usize];
                    let p = raw.at_scaled(x as f32 + 0.5, y as f32 + 0.5, shot.truth.width, shot.truth.height);
                    let sd = spread.at_scaled(x as f32 + 0.5, y as f32 + 0.5, shot.truth.width, shot.truth.height);
                    if t.is_finite() && t <= 10.0 && p > 0.0 {
                        book.pair("depth/tips_spread_flags_err", sd / p, (p - t).abs() / t > 0.25);
                    }
                }
            }
        });
        let dense_old = timers.time("siglip_dense", || sig.0.read().unwrap().embed_dense_batch(&[shot.rgb.to_dynamic()]).unwrap().pop().unwrap());
        let pts_old = points_with_presence(shot.rgb.width, shot.rgb.height, label_heats(&dense_old, &texts[0], &variants[0].1, scale, bias));
        record_locate(&mut book, "loc/old_pairprob", &shot, s, &pts_old, &mut per_frame);
        let pts_si = LocateStandIn::new(s.seed * 1000 + s.dist_m as u64).run(&shot.truth, &labels);
        record_locate(&mut book, "loc/standin", &shot, s, &pts_si, &mut per_frame);
        *d_cell.lock().unwrap() = Some(raw.clone());
        *p_cell.lock().unwrap() = pts_old.clone();
        for a in arms.iter_mut() {
            let k: &'static str = if a.name == "A3" || a.name == "A4" { "arm_sam" } else { "arm_region" };
            timers.time(k, || run_arm(&mut book, a, &shot, s, None));
        }
        timers.time("scan_oracle", || record_scan(&mut book, "A0", &shot, &mut oracle));
        timers.time("scan_siglip", || record_scan(&mut book, "A4", &shot, &mut scan));
        timers.time("scan_siglip", || record_scan(&mut book, "A4tpl", &shot, &mut scan_tpl));

        // ---------- the v1.0 port and fixes ----------
        set_port(&*sig, &mut tips, true);
        let raw1 = timers.time("tips", || tips.predict(&shot.rgb).unwrap());
        let ground0 = shot.world.surface_m(shot.pose.x, shot.pose.z);
        let w = &shot.world;
        let dem = move |x: f32, z: f32| w.surface_m(x, z) - ground0;
        timers.time("metrics", || {
            record_depth(&mut book, "depth/tips_v1_raw", &raw1, &shot.truth, shot.target_raw);
            record_depth(&mut book, "depth/tips_v1_oracle_lsq", &oracle_lsq(&raw1, &shot.truth, 10.0), &shot.truth, shot.target_raw);
            if let Some(m) = depth_metrics(&raw1, &shot.truth.depth_map(), 10.0, false) {
                book.add("depth/tips_v1_raw/median_scale", m.median_scale);
            }
            errfit.add_frame(&raw1, &shot.truth, 10.0);
        });
        // The same frame without the WebP q70 round trip (cause 9: what the
        // lossy keyframe costs the depth model).
        let lossless = Rgb::new(shot.frame.width as u32, shot.frame.height as u32, shot.frame.rgb.clone());
        let raw_ll = timers.time("tips", || tips.predict(&lossless).unwrap());
        timers.time("metrics", || {
            record_depth(&mut book, "depth/tips_v1_lossless", &raw_ll, &shot.truth, shot.target_raw);
            record_depth(&mut book, "depth/tips_v1_lossless_oracle_lsq", &oracle_lsq(&raw_ll, &shot.truth, 10.0), &shot.truth, shot.target_raw);
        });
        // The input grid: the checkpoint's own square 448 x 448 (32 x 32 patches,
        // the aspect squeezed) instead of the 504 x 336 that keeps it.
        tips.input_grid = InputGrid::Fixed { cols: 32, rows: 32 };
        let sq = timers.time("tips", || tips.predict(&shot.rgb).unwrap());
        tips.input_grid = InputGrid::default();
        timers.time("metrics", || {
            record_depth(&mut book, "depth/tips_v1_sq448", &sq, &shot.truth, shot.target_raw);
            record_depth(&mut book, "depth/tips_v1_sq448_oracle_lsq", &oracle_lsq(&sq, &shot.truth, 10.0), &shot.truth, shot.target_raw);
            if let Some(m) = depth_metrics(&sq, &shot.truth.depth_map(), 10.0, false) {
                book.add("depth/tips_v1_sq448/median_scale", m.median_scale);
            }
            let mut d = sq.clone();
            noto_vision::ground_align(&mut d, &shot.intr, &shot.pose, Some(&dem as noto_vision::align::HeightFn), Some(10.0), &AlignCfg::default());
            record_depth(&mut book, "depth/tips_v1_sq448_ground_map", &d, &shot.truth, shot.target_raw);
        });
        for (name, h, cfg) in [
            ("depth/tips_v1_ground_flat", None, AlignCfg::default()),
            ("depth/tips_v1_ground_map", Some(&dem as noto_vision::align::HeightFn), AlignCfg::default()),
            ("depth/tips_v1_ground_scale", Some(&dem as noto_vision::align::HeightFn), AlignCfg { space: AlignSpace::Scale, ..AlignCfg::default() }),
            ("depth/tips_v1_ground_map_noclip", Some(&dem as noto_vision::align::HeightFn), AlignCfg { clip_to_support: false, ..AlignCfg::default() }),
        ] {
            let mut d = raw1.clone();
            let fit = timers.time("align", || noto_vision::ground_align(&mut d, &shot.intr, &shot.pose, h, Some(10.0), &cfg));
            book.flag(format!("{name}/fitted"), fit.is_some());
            record_depth(&mut book, name, &d, &shot.truth, shot.target_raw);
            if name == "depth/tips_v1_ground_map" && s.fit() {
                errfit_ground.add_frame(&d, &shot.truth, 10.0);
            }
            if let Some(f) = fit {
                book.add(format!("{name}/fit_a"), f.a);
                book.add(format!("{name}/fit_b"), f.b);
                book.add(format!("{name}/inlier_share"), f.inlier_share);
                book.add(format!("{name}/median_rel_err"), f.median_rel_err);
            }
        }
        let dense = timers.time("siglip_dense", || sig.0.read().unwrap().embed_dense_batch(&[shot.rgb.to_dynamic()]).unwrap().pop().unwrap());
        let pts_new: Vec<Vec<SemanticPoint>> = variants
            .iter()
            .zip(&texts)
            .skip(1)
            .map(|((name, cfg), tx)| {
                let pts = points_with_presence(shot.rgb.width, shot.rgb.height, label_heats(&dense, tx, cfg, scale, bias));
                record_locate(&mut book, name, &shot, s, &pts, &mut per_frame);
                pts
            })
            .collect();
        // The contrast locator on the lossless frame (cause 9 for the locator).
        let dense_ll = timers.time("siglip_dense", || sig.0.read().unwrap().embed_dense_batch(&[lossless.to_dynamic()]).unwrap().pop().unwrap());
        let pts_ll = points_with_presence(lossless.width, lossless.height, label_heats(&dense_ll, &texts[3], &variants[3].1, scale, bias));
        record_locate(&mut book, "loc/new_contrast_lossless", &shot, s, &pts_ll, &mut per_frame);
        let raw_scan = timers.time("scan_siglip", || record_scan(&mut book, "A5raw", &shot, &mut scan));
        if s.fit() {
            scan_fit.extend(raw_scan.cands);
        } else {
            for c in &raw_scan.cands {
                book.pair("A5raw/scan/eval_score", c.score, c.tp);
            }
        }
        // The calibrated scan: on every scene with a shipped table (the fit
        // seeds feed the honest scan stand-in), else on the held-out seeds.
        if scan_cal.calibration().is_some() {
            let name = if s.fit() { "A5cal_fit" } else { "A5cal" };
            let cal = timers.time("scan_siglip", || record_scan(&mut book, name, &shot, &mut scan_cal));
            if !s.fit() {
                for c in &cal.cands {
                    book.pair("A5cal/scan/platt", c.score, c.tp);
                }
            }
            if s.fit() == shipped.is_some() {
                scan_honest_fit.add(&cal);
            }
        }
        if !s.fit() {
            if let Some((x, y, vk)) = a5.as_mut() {
                let pts = &pts_new[*vk - 1];
                // The gate on the held-out seeds, point by point.
                let t = &tables[0];
                for p in pts.iter().filter(|p| TARGETS.contains(&p.label.as_str())) {
                    let (Some(c), Some(present)) = (t.labels.get(&p.label), presence(&shot, &p.label)) else { continue };
                    let stat = match t.statistic.as_str() {
                        "presence" => p.presence,
                        "peak_x_presence" => p.score * p.presence,
                        _ => p.score,
                    };
                    let pass = stat >= c.threshold;
                    if pass {
                        book.flag("A5gate/eval/precision", present);
                    }
                    if present {
                        book.flag("A5gate/eval/recall", pass);
                    }
                    if let Some(pr) = c.probability(stat) {
                        book.pair("A5gate/eval/platt", pr, present);
                    }
                }
                *d_cell.lock().unwrap() = Some(raw1.clone());
                *p_cell.lock().unwrap() = pts.clone();
                timers.time("arm_sam", || run_arm(&mut book, x, &shot, s, Some(&dem)));
                timers.time("arm_region", || run_arm(&mut book, y, &shot, s, Some(&dem)));
            }
            *p_cell.lock().unwrap() = pts_new[0].clone();
            *d_cell.lock().unwrap() = Some(raw1.clone());
            timers.time("arm_sam", || run_arm(&mut book, &mut a4n, &shot, s, None));
            for h in honest.iter_mut() {
                timers.time("arm_honest", || run_arm(&mut book, h, &shot, s, None));
                if h.name == "A0h" {
                    let mut d = DepthStandIn::new(s.seed * 1000 + s.dist_m as u64);
                    d.err = h.honest.map(|x| x.0);
                    record_depth(&mut book, "depth/standin_honest", &d.run(&shot.truth), &shot.truth, shot.target_raw);
                }
            }
            if let Some(o) = honest_scan.as_mut() {
                timers.time("scan_oracle", || record_scan(&mut book, "A0h", &shot, o));
            }
        }
        if i % 25 == 0 {
            println!("[MLAB] {}/{} ({:.0} s)", i + 1, scenes.len(), t0.elapsed().as_secs_f32());
        }
    }
    if let Some(dir) = out_dir() {
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("ml_calib.json"), serde_json::to_string_pretty(&tables).unwrap()).unwrap();
        let lines: Vec<String> = per_frame.iter().map(|v| v.to_string()).collect();
        std::fs::write(dir.join("mlab_frames.jsonl"), lines.join("\n")).unwrap();
    }

    // ---------- report ----------
    let times: Vec<String> = timers.0.iter().map(|(k, v)| format!("{k} {v:.0} s")).collect();
    println!("[MLAB] 시간: {}", times.join(" · "));
    println!("[MLAB] ===== 깊이 (진실 10 m 이하 화소; NYU 상한과 같은 범위) =====");
    for d in [
        "depth/standin",
        "depth/tips_raw",
        "depth/tips_median",
        "depth/tips_oracle_lsq",
        "depth/tips_v1_raw",
        "depth/tips_v1_oracle_lsq",
        "depth/tips_v1_ground_flat",
        "depth/tips_v1_ground_map",
        "depth/tips_v1_ground_scale",
        "depth/tips_v1_ground_map_noclip",
        "depth/tips_v1_sq448",
        "depth/tips_v1_sq448_oracle_lsq",
        "depth/tips_v1_sq448_ground_map",
        "depth/tips_v1_lossless",
        "depth/tips_v1_lossless_oracle_lsq",
        "depth/standin_honest",
    ] {
        for m in ["absrel", "delta1", "rmse_m", "coverage", "target_absrel"] {
            book.line(&format!("{d}/{m}"), "");
        }
        let bands: Vec<String> = BANDS.iter().filter_map(|b| book.mean(&format!("{d}/band{}", b.2)).map(|(v, n)| format!("{} m {v:.2} ({n})", b.2))).collect();
        println!("[MLAB] {d}/bands AbsRel {}", bands.join(" · "));
    }
    book.line("depth/tips_raw/median_scale", "참/예측 중앙값: 1보다 크면 모델이 가깝게 봄");
    book.line("depth/tips_v1_raw/median_scale", "");
    book.line("depth/tips_v1_sq448/median_scale", "정사각 448 입력");
    for (name, f, cap) in [("depth/errmodel_raw", &errfit, 10.0f32), ("depth/errmodel_ground", &errfit_ground, f32::INFINITY)] {
        if let Some((a, b, tau, sigma, sky)) = f.fit() {
            println!("[MLAB] {name} ln(예측/참) = {a:.4} + {b:.4}·ln(참) · 프레임 sd {tau:.4} · 화소 sd {sigma:.4} · 하늘 답 중앙값 {sky:.2} m (프레임 {})", f.frames.len());
            if let Some(m) = f.model(cap) {
                println!("[MLAB] {name} = {}", depth_model_rs(&m));
            }
        }
    }
    for k in ["depth/tips_v1_ground_flat/fitted", "depth/tips_v1_ground_map/fitted", "depth/tips_v1_ground_map/fit_a", "depth/tips_v1_ground_map/fit_b", "depth/tips_v1_ground_map/inlier_share", "depth/tips_v1_ground_map/median_rel_err", "depth/tips_v1_ground_scale/fit_a"] {
        book.line(k, "");
    }
    let (au, _, pos, n) = book.rank("depth/tips_spread_flags_err");
    println!("[MLAB] depth/tips_spread_flags_err AUROC {} (오차>25% 화소 {pos}/{n}) — 빈 분포 폭/깊이가 큰 오차를 가려내는가", au.map_or("-".into(), |v| format!("{v:.3}")));
    println!("[MLAB] ===== 의미 위치 (존재 AUROC/AP, 상자 안 피크) =====");
    for v in ["loc/standin", "loc/old_pairprob", "loc/new_pairprob", "loc/new_pairprob_tpl", "loc/new_contrast", "loc/new_contrast_tpl", "loc/new_contrast_nobg", "loc/new_z", "loc/new_contrast_lossless"] {
        for stat in ["peak", "presence"] {
            print_rank(&book, v, stat);
        }
        book.line(&format!("{v}/peak_in_box"), "보이는 라벨의 피크가 참 상자(한 패치 확장) 안");
        book.line(&format!("{v}/locate_err_px"), "");
    }
    book.line("A5gate/eval/precision", "보정 게이트를 통과한 점의 정밀도 (평가 시드)");
    book.line("A5gate/eval/recall", "보이는 라벨 중 게이트 통과");
    if let Some((s, h)) = book.pairs.get("A5gate/eval/platt") {
        println!("[MLAB] A5gate/eval/ece {:.3} (n {}) — {CITE_CALIB}", noto_vision::calib::ece(s, h, noto_vision::calib::ECE_BINS).unwrap_or(f32::NAN), s.len());
    }
    println!("[MLAB] ===== 추적과 거리 (사슬 전체, 같은 키프레임) =====");
    for a in ["A0", "A1", "A2", "A3", "A4", "A4new", "A5", "A5region", "A0h", "A0hsam"] {
        for m in ["track/target_found", "track/right_object", "track/iou_truth", "track/phantom_when_absent", "range/rel_err", "range/signed_rel", "range/z", "align/fitted"] {
            book.line(&format!("{a}/{m}"), "");
        }
    }
    println!("[MLAB] ===== 타일 스캔 (게이트 점수 {GATE_SCORE} · 여유 {GATE_MARGIN}, 1 m 안) =====");
    for a in ["A0", "A4", "A4tpl", "A5raw", "A5cal_fit", "A5cal", "A0h"] {
        for m in ["scan/precision", "scan/recall", "scan/pos_err_m", "scan/false_per_scan"] {
            book.line(&format!("{a}/{m}"), if a == "A0" { "대역(정답 버퍼 + 주입 오류)" } else if a == "A0h" { "정직한 대역" } else { "" });
        }
    }
    if shipped.is_none() {
        // Without a shipped table the calibrated scan ran on the held-out seeds only: in-sample.
        println!("[MLAB] scan/errmodel (평가 시드, 표본 안) {:?}", scan_honest_fit.model());
    }
    for k in ["A5raw/scan/eval_score", "A5cal/scan/platt"] {
        if let Some((s, h)) = book.pairs.get(k) {
            let probs: Vec<f32> = s.iter().map(|x| x.clamp(0.0, 1.0)).collect();
            println!("[MLAB] {k} AUROC {} · ECE {:.3} (후보 {}, 참 {})", auroc(s, h).map_or("-".into(), |v| format!("{v:.3}")), noto_vision::calib::ece(&probs, h, noto_vision::calib::ECE_BINS).unwrap_or(f32::NAN), s.len(), h.iter().filter(|x| **x).count());
        }
    }

    // ---------- scored cells (Group B) ----------
    println!("[MLAB] ===== 기준 대비 =====");
    if let Some((v, _)) = book.mean("depth/tips_v1_oracle_lsq/absrel") {
        let z = (v - SINTEL_ALIGNED_ABSREL.0) / SINTEL_ALIGNED_ABSREL.1;
        println!("[MLAB] M1/aligned_absrel {v:.3} (Like {{ reference: {:.3}, tolerance: {:.3} }}) · 점수 {:.0} — {CITE_SINTEL}", SINTEL_ALIGNED_ABSREL.0, SINTEL_ALIGNED_ABSREL.1, (100.0 - 10.0 * z.abs()).max(0.0));
    }
    if let (Some((g, _)), Some((o, _))) = (book.mean("depth/tips_v1_ground_map/absrel"), book.mean("depth/tips_v1_oracle_lsq/absrel")) {
        let r = g / o.max(1e-6);
        println!("[MLAB] M2/ground_over_oracle_absrel {r:.3} (NoWorse {{ reference: {DNET_GROUND_OVER_MEDIAN:.3}, tolerance: 0.1 }}) · 점수 {:.0} — {CITE_DNET}", (100.0 - 10.0 * ((r - DNET_GROUND_OVER_MEDIAN) / 0.1).max(0.0)).max(0.0));
    }
    if let Some((v, _)) = book.mean("depth/tips_raw/absrel") {
        println!("[MLAB] M3/raw_metric_absrel {v:.3} (Raw, 기준 {OFFDOMAIN_METRIC_ABSREL}) — {CITE_OFFDOMAIN}");
    }
    println!("[MLAB] M4 기록 — {CITE_SCLIP}; {CITE_ENSEMBLE}");
    println!("[MLAB-SUM] {}개 장면 · {:.0} s", scenes.len(), t0.elapsed().as_secs_f32());
}

/// The v1.0 arms (chain with MobileSAM, and with the region tracker) for the
/// locator variant `vk` gated by `table`.
#[allow(clippy::too_many_arguments)]
fn a5_arms(lex: &Lexicon, prompting: &VisionPrompting, table: &CalibTable, vk: usize, sam: &Arc<SamModel>, d: &Arc<Mutex<Option<DepthMap>>>, p: &Arc<Mutex<Vec<SemanticPoint>>>) -> (Arm, Arm, usize) {
    let vcfg = LocateCfg::from_mode(&table.mode, &prompting.background, &prompting.templates).expect("a dense locator mode");
    let mut fix = MlFixCfg::v1(prompting);
    fix.locate = vcfg.clone();
    let mut x = arm("A5", lex, Src::Model, Src::Model, Box::new(SamTracker::new(Arc::clone(sam))), d, p, "tipsv2-b14-dpt", &vcfg.mode(), fix.clone());
    x.pipe.calib = vec![table.clone()];
    let mut y = arm("A5region", lex, Src::Model, Src::Model, Box::new(RegionTracker::default()), d, p, "tipsv2-b14-dpt", &vcfg.mode(), fix);
    y.pipe.calib = vec![table.clone()];
    (x, y, vk)
}

/// A depth error model as Rust source (for `DepthErrModel::TIPS_GROUND`).
fn depth_model_rs(m: &DepthErrModel) -> String {
    let f = |v: f32| if v.is_finite() { format!("{v:.4}") } else { "f32::INFINITY".into() };
    format!("DepthErrModel {{ a: {}, b: {}, frame_sd: {}, patch_sd: {}, sky_m: {}, cap_m: {}, unknown_beyond_m: {}, blur_px: {} }}", f(m.a), f(m.b), f(m.frame_sd), f(m.patch_sd), f(m.sky_m), f(m.cap_m), f(m.unknown_beyond_m), m.blur_px)
}

/// One closed-loop quest: the game pass, perception of its keyframes and the
/// ML decision pass, as `tests/integrated.rs` runs them.
fn run_quest(rt: &mut Runtime, cmd: &str, seed: u64, base: BaseKind) -> serde_json::Value {
    rt.test_start(cmd, seed, noto_core::BodyPlan::default(), Some(base)).expect("test starts");
    let mut guard = 0;
    while rt.test_session().map(|t| t.phase) != Some(noto_runtime::TestPhase::Done) {
        rt.test_advance(400).expect("test advances");
        guard += 1;
        assert!(guard < 4000, "the test does not end");
    }
    rt.api_test_status().unwrap()
}

/// [MLAB] level 2: whole quests (game pass, perception, ML decision pass)
/// with the stand-ins, the honest stand-ins, the real models as they were
/// before v1.0 and the real models on the v1.0 path. `MLAB_QUESTS` (commands
/// separated by `;`) and `MLAB_QUEST_SEEDS` choose the grid.
#[test]
#[ignore]
fn ml_ab_quests() {
    let t0 = std::time::Instant::now();
    let dev = device();
    let cmds: Vec<String> = std::env::var("MLAB_QUESTS").unwrap_or_else(|_| "원목 1개를 창고로 가져와;택배 상자 1개를 창고로 가져와".into()).split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let seeds: Vec<u64> = env_list("MLAB_QUEST_SEEDS", &[7]);
    let sig_dir = PathBuf::from(std::env::var("SIGLIP2_DIR").expect("SIGLIP2_DIR"));
    let tips_dir = PathBuf::from(std::env::var("TIPS_DIR").expect("TIPS_DIR"));
    let sam_dir = PathBuf::from(std::env::var("SAM_DIR").expect("SAM_DIR"));
    let bank = bank_dir();
    let mut rows: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for arm in ["standin", "honest", "real_v0.9", "real_v1.0"] {
        // One SigLIP2 at a time on the 4 GB card; the old port resizes positions without antialias.
        let siglip = arm.starts_with("real").then(|| {
            let mut s = noto_ml::Siglip2::load(&sig_dir, &dev).expect("SigLIP2");
            s.set_position_antialias(arm == "real_v1.0");
            Arc::new(s)
        });
        for cmd in &cmds {
            for &seed in &seeds {
                let cfg = RuntimeConfig { seed, noise: NoiseCfg::none(), ..RuntimeConfig::default() };
                let mut rt = Runtime::new(cfg, LEX, None).expect("runtime");
                match arm {
                    "honest" => rt.use_honest_stand_ins(),
                    "real_v0.9" | "real_v1.0" => {
                        if arm == "real_v0.9" {
                            rt.ml_pipeline_mut().fix = MlFixCfg::legacy();
                        }
                        rt.use_tips(&tips_dir, &dev).expect("TIPS");
                        rt.use_sam(&sam_dir, &dev).expect("SAM");
                        rt.use_siglip(Arc::clone(siglip.as_ref().unwrap()), Some(&bank)).expect("SigLIP2");
                    }
                    _ => {}
                }
                let t = std::time::Instant::now();
                let st = run_quest(&mut rt, cmd, seed, BaseKind::HumanFoot);
                let dec = st["decide"]["keyframes"].as_array().cloned().unwrap_or_default();
                let mut kinds: BTreeMap<String, u32> = BTreeMap::new();
                for k in &dec {
                    if let Some(x) = k["decision"]["kind"].as_str() {
                        *kinds.entry(x.to_string()).or_default() += 1;
                    }
                }
                let tracked = dec.iter().filter(|k| !k["ml"]["target"].is_null()).count();
                let q = &st["ml_quest"];
                let row = serde_json::json!({
                    "cmd": cmd, "seed": seed,
                    "game": st["game_quest"]["state"], "ml": q["state"], "failure": q["failure"]["code"],
                    "time_s": q["result"]["time_s"], "wasted": q["result"]["wasted_actions"], "precision": q["result"]["precision"],
                    "keyframes": dec.len(), "tracked": tracked, "decisions": kinds, "modes": st["modes"], "wall_s": t.elapsed().as_secs_f32(),
                });
                println!("[MLAB-Q] {arm} {row}");
                rows.entry(arm).or_default().push(row);
            }
        }
    }
    for (arm, r) in &rows {
        let ok = r.iter().filter(|x| x["ml"] == "done").count();
        let game = r.iter().filter(|x| x["game"] == "done").count();
        let wasted: u64 = r.iter().filter_map(|x| x["wasted"].as_u64()).sum();
        let time: Vec<f64> = r.iter().filter(|x| x["ml"] == "done").filter_map(|x| x["time_s"].as_f64()).collect();
        let mean_t = if time.is_empty() { f64::NAN } else { time.iter().sum::<f64>() / time.len() as f64 };
        println!("[MLAB-Q-SUM] {arm}: ML 패스 성공 {ok}/{} · 게임 패스 {game}/{} · 헛수고 {wasted} · 성공한 퀘스트 평균 {mean_t:.0} s", r.len(), r.len());
    }
    println!("[MLAB-SUM] 퀘스트 {:.0} s", t0.elapsed().as_secs_f32());
}

fn write_f32(path: PathBuf, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// The port against its references (cause 10 of `docs/research/ml-ab.md`).
/// For a few rendered keyframes it writes the exact pixels TIPS and SigLIP2
/// were given (after this port's own resize) and what the port made of them,
/// plus the token ids and text vectors of a few prompts, to `$MLAB_OUT/parity`.
/// `tests/ml_parity.py` runs the Hugging Face implementations on the same
/// inputs (CPU) and prints the differences.
#[test]
#[ignore]
fn ml_port_dump() {
    let dir = out_dir().expect("MLAB_OUT").join("parity");
    std::fs::create_dir_all(&dir).unwrap();
    let dev = device();
    let sig_dir = PathBuf::from(std::env::var("SIGLIP2_DIR").expect("SIGLIP2_DIR"));
    let mut siglip = noto_ml::Siglip2::load(&sig_dir, &dev).expect("SigLIP2");
    let mut tips = TipsDepth::load(&PathBuf::from(std::env::var("TIPS_DIR").expect("TIPS_DIR")), &dev).expect("TIPS");
    let mut bases: BTreeMap<(u64, u8, u8), World> = BTreeMap::new();
    let scene = |target: &'static str, dist_m: f32, tilt_deg: f32| Scene {
        seed: 1,
        terrain: TerrainKind::Open,
        base: BaseKind::HumanFoot,
        target: Some(target),
        query: target,
        distractor: Some(look_alike_of(target)),
        dist_m,
        tilt_deg,
        lateral_m: 0.0,
    };
    let scenes = [scene("parcel", 3.0, 30.0), scene("log", 2.0, 0.0), scene("tree", 6.0, 0.0)];
    let cfg = siglip.config().clone();
    let mut manifest = Vec::new();
    for (k, s) in scenes.iter().enumerate() {
        let shot = shoot(&mut bases, s);
        let src = image::RgbImage::from_raw(shot.rgb.width, shot.rgb.height, shot.rgb.data.clone()).unwrap();
        // TIPS: the keyframe on its patch grid, as `model_input` resizes it.
        let (cols, rows) = tips.input_grid.grid_for(shot.rgb.width, shot.rgb.height);
        let p = tips.config().patch_size as u32;
        let t_img = image::imageops::resize(&src, cols as u32 * p, rows as u32 * p, image::imageops::FilterType::Triangle);
        t_img.save(dir.join(format!("kf{k}_tips.png"))).unwrap();
        let t_rgb = Rgb::new(t_img.width(), t_img.height(), t_img.clone().into_raw());
        for (tag, tanh) in [("tanh", true), ("erf", false)] {
            tips.set_readout_gelu_tanh(tanh);
            write_f32(dir.join(format!("kf{k}_tips_{tag}.f32")), &tips.predict(&t_rgb).unwrap().data);
        }
        // SigLIP2: the keyframe on its NaFlex grid, as `preprocess` resizes it.
        let (gh, gw) = noto_ml::preprocess::naflex_grid(shot.rgb.width, shot.rgb.height, cfg.patch_size, cfg.max_patches);
        let s_img = shot.rgb.to_dynamic().resize_exact((gw * cfg.patch_size) as u32, (gh * cfg.patch_size) as u32, image::imageops::FilterType::Triangle).to_rgb8();
        s_img.save(dir.join(format!("kf{k}_siglip.png"))).unwrap();
        for (tag, aa) in [("aa", true), ("bilinear", false)] {
            siglip.set_position_antialias(aa);
            write_f32(dir.join(format!("kf{k}_siglip_{tag}.f32")), &siglip.embed_image(&image::DynamicImage::ImageRgb8(s_img.clone())).unwrap());
        }
        manifest.push(serde_json::json!({"k": k, "scene": s.id(), "tips": [t_img.width(), t_img.height()], "siglip_grid": [gh, gw]}));
    }
    siglip.set_position_antialias(true);
    // Text: this port's tokens (canonicalised: lowercase, punctuation to spaces) and vectors.
    let prompts = ["a cardboard parcel", "a photo of a cardboard parcel.", "a wooden log", "a grey boulder", "green grass", "a rendering of a stone.", "a tree in a video game."];
    let tok = noto_ml::tokenizer::TextTokenizer::from_file(&sig_dir.join("tokenizer.json"), cfg.text_pad_id, cfg.text_max_len).unwrap();
    let (ids, mask) = tok.encode(&prompts).unwrap();
    let vecs = siglip.embed_texts(&prompts).unwrap();
    write_f32(dir.join("text_rust.f32"), &vecs.concat());
    let (scale, bias) = siglip.logit_scale_bias();
    let meta = serde_json::json!({
        "frames": manifest,
        "prompts": prompts,
        "ids": ids.chunks(cfg.text_max_len).map(|c| c.to_vec()).collect::<Vec<_>>(),
        "mask_sum": mask.chunks(cfg.text_max_len).map(|c| c.iter().sum::<f32>()).collect::<Vec<_>>(),
        "logit_scale": scale,
        "logit_bias": bias,
        "dim": siglip.embedding_dim(),
        "tips_max_depth": tips.config().max_depth,
    });
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    println!("[MLAB] 비교 입력 {}", dir.display());
}

#[test]
fn the_engine_never_names_an_ml_benchmark_source() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let engine = [
        "crates/noto-vision/src/align.rs",
        "crates/noto-vision/src/calib.rs",
        "crates/noto-vision/src/locate.rs",
        "crates/noto-vision/src/tips.rs",
        "crates/noto-vision/src/depth.rs",
        "crates/noto-vision/src/decide.rs",
        "crates/noto-ml/src/vision.rs",
        "crates/noto-search/src/vision.rs",
        "crates/noto-runtime/src/ml.rs",
        "crates/noto-runtime/src/mlsds.rs",
        "crates/noto-runtime/src/embedding.rs",
        "crates/noto-runtime/src/precision.rs",
        "crates/noto-runtime/src/perception.rs",
        "crates/noto-runtime/src/runtime.rs",
        "crates/noto-ml/src/model.rs",
        "data/world_lexicon.json",
        "data/ml_calib.json",
    ];
    let banned = ["Sintel", "Depth Anything", "Yang et al", "DNet", "Xue", "SCLIP", "Wang, Mei", "LeVine", "Hypersim", "ZoeDepth", "Marigold", "PointCLIP", "CoOp", "Bochkovskiy", "KITTI", "Radford", "Cao et al"];
    for f in engine {
        let text = std::fs::read_to_string(format!("{root}/{f}")).unwrap_or_else(|e| panic!("{f}: {e}"));
        for b in banned {
            assert!(!text.contains(b), "{f} names the benchmark source {b}");
        }
    }
}

#[test]
fn the_shipped_calibration_parses_and_names_known_labels() {
    let t = CalibTable::parse(noto_runtime::ml::ML_CALIB_JSON).expect("data/ml_calib.json");
    for table in &t {
        assert!(table.mode.starts_with("siglip2-dense") || table.mode == noto_runtime::embedding::SCAN_CALIB_MODE, "{}", table.mode);
        for (l, c) in &table.labels {
            assert!(TARGETS.contains(&l.as_str()) || l == "*", "{l}");
            assert!(c.threshold.is_finite() || c.platt.is_some());
        }
    }
    let _ = DepthKind::Metric;
}
