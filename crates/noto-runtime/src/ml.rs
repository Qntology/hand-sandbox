//! The ML logic's pipeline as the runtime drives it: one keyframe in, depth,
//! semantic points, tracked masks and the target observation out.
//!
//! The models live in `noto-vision`. Two of them need weights this repository
//! does not ship (TIPSv2 DPT, SigLIP2). Without weights the runtime plugs in a
//! stand-in that is honest about being one: it reads the renderer's ground
//! truth of the keyframe and spoils it the way the real model would be wrong
//! (depth clamped at 10 m, an unknown scale, patch-wise errors; semantic heat
//! on a coarse patch grid with noise and look-alike confusion). Its mode name
//! says `stand-in`. The mask tracker has no stand-in of that kind: the
//! region-growing tracker sees only the image and the depth map.
//!
//! Ground truth is also used to score the ML tracks (depth error, location
//! error, mask IoU, true range). Those numbers go to evaluation columns and
//! SDS evaluation axes only; the decision never reads them.

use std::collections::{BTreeMap, HashMap, VecDeque};

use anyhow::Result;
use noto_search::{EntityKind, Lexicon};
use noto_vision::{
    depth_metrics, DepthKind, DepthMap, DepthMetrics, DepthModel, Heat, Intrinsics, LabelPrompt, Mask, MaskTracker, PointPrompt, Pose, RegionTracker, Rgb, SemanticLocator,
    SemanticPoint, TrackedMask,
};
use serde::Serialize;
use world_sim::render::unpack_id;
use world_sim::rng::Rng;
use world_sim::{Frame, ObjectClass, VisibleObject, World};

/// TIPSv2 DPT cannot report depth beyond this, metres.
pub const TIPS_MAX_DEPTH_M: f32 = 10.0;
/// Scale error of the depth stand-in: an indoor-trained model on an outdoor
/// voxel scene has some unknown metric scale. The value is chosen, not
/// measured; triangulation is what finds it (the SDS `depth` scale axis).
pub const STANDIN_DEPTH_SCALE: f32 = 0.8;

/// Ground truth of a keyframe at half resolution, for the stand-ins and the
/// evaluation columns.
#[derive(Clone, Debug)]
pub struct KeyTruth {
    pub width: u32,
    pub height: u32,
    pub ids: Vec<u32>,
    pub depth: Vec<f32>,
    pub objects: Vec<VisibleObject>,
    /// Ground position of every visible entity: (entity, label, x, z).
    pub positions: Vec<(String, String, f32, f32)>,
    /// Size of the keyframe the truth was taken from.
    pub src_width: u32,
    pub src_height: u32,
}

impl KeyTruth {
    pub fn capture(world: &World, frame: &Frame) -> KeyTruth {
        let (w, h) = ((frame.width / 2).max(1) as u32, (frame.height / 2).max(1) as u32);
        let mut ids = Vec::with_capacity((w * h) as usize);
        let mut depth = Vec::with_capacity((w * h) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * 2) * frame.width + x * 2;
                ids.push(frame.ids[i]);
                depth.push(frame.depth[i]);
            }
        }
        let objects = world.visible_objects(frame);
        let positions = objects
            .iter()
            .filter_map(|o| {
                let id: u32 = o.entity.get(1..)?.parse().ok()?;
                let (x, z) = match o.class {
                    ObjectClass::Node => world.node(id).map(|n| (n.x, n.z))?,
                    ObjectClass::Drop => world.drop_item(id).map(|d| (d.x, d.z))?,
                    ObjectClass::Waypoint => world.waypoint(id).map(|p| (p.x, p.z))?,
                    _ => return None,
                };
                Some((o.entity.clone(), o.label.clone(), x, z))
            })
            .collect();
        KeyTruth { width: w, height: h, ids, depth, objects, positions, src_width: frame.width as u32, src_height: frame.height as u32 }
    }

    pub fn depth_map(&self) -> DepthMap {
        DepthMap { width: self.width, height: self.height, kind: DepthKind::Metric, data: self.depth.clone() }
    }

    /// Entity under the most pixels of a mask (same half resolution).
    pub fn entity_under(&self, mask: &Mask) -> Option<(u32, f32)> {
        let mut count: HashMap<u32, u32> = HashMap::new();
        for y in 0..mask.height.min(self.height) {
            for x in 0..mask.width.min(self.width) {
                if mask.get(x, y) {
                    *count.entry(self.ids[(y * self.width + x) as usize]).or_insert(0) += 1;
                }
            }
        }
        let (raw, n) = count.into_iter().filter(|(raw, _)| matches!(unpack_id(*raw).0, ObjectClass::Node | ObjectClass::Drop)).max_by_key(|(_, n)| *n)?;
        Some((raw, n as f32 / mask.area().max(1) as f32))
    }

    /// IoU of a mask with the true pixels of entity `raw`.
    pub fn iou(&self, mask: &Mask, raw: u32) -> f32 {
        let (mut inter, mut uni) = (0u32, 0u32);
        for y in 0..self.height.min(mask.height) {
            for x in 0..self.width.min(mask.width) {
                let a = mask.get(x, y);
                let b = self.ids[(y * self.width + x) as usize] == raw;
                inter += (a && b) as u32;
                uni += (a || b) as u32;
            }
        }
        if uni == 0 {
            0.0
        } else {
            inter as f32 / uni as f32
        }
    }
}

fn gauss(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f32().max(1e-6);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Depth stand-in shaped like TIPSv2 DPT's output (research report: ground
/// truth clamped to 10 m and seen through the model's patch grid).
pub struct DepthStandIn {
    rng: Rng,
    /// Multiplicative error per 14-pixel patch (one standard deviation).
    pub patch_sd: f32,
}

impl DepthStandIn {
    pub fn new(seed: u64) -> DepthStandIn {
        DepthStandIn { rng: Rng::new(seed ^ 0xDE97), patch_sd: 0.06 }
    }

    pub fn run(&mut self, truth: &KeyTruth) -> DepthMap {
        // Patch grid of a 504x336 input is 36x24; on the half-resolution truth that is about 5 px per patch.
        let (pw, ph) = (36u32, 24u32);
        let errs: Vec<f32> = (0..pw * ph).map(|_| (self.patch_sd * gauss(&mut self.rng)).exp()).collect();
        let mut data = Vec::with_capacity(truth.depth.len());
        for y in 0..truth.height {
            for x in 0..truth.width {
                let d = truth.depth[(y * truth.width + x) as usize];
                // NYU has no sky: an indoor model reports its far limit there.
                let d = if d.is_finite() { d } else { TIPS_MAX_DEPTH_M };
                let e = errs[((y * ph / truth.height).min(ph - 1) * pw + (x * pw / truth.width).min(pw - 1)) as usize];
                data.push((d * STANDIN_DEPTH_SCALE * e).clamp(1e-3, TIPS_MAX_DEPTH_M));
            }
        }
        DepthMap { width: truth.width, height: truth.height, kind: DepthKind::Metric, data }
    }
}

/// Semantic-location stand-in shaped like SigLIP2 dense heat (a coarse patch
/// grid, soft scores, noise, and heat leaking to look-alike labels).
pub struct LocateStandIn {
    rng: Rng,
    pub noise_sd: f32,
    /// Share of a label's heat that also shows up on its look-alike.
    pub confusion: f32,
}

impl LocateStandIn {
    pub fn new(seed: u64) -> LocateStandIn {
        LocateStandIn { rng: Rng::new(seed ^ 0x51C1), noise_sd: 0.04, confusion: 0.35 }
    }

    pub fn run(&mut self, truth: &KeyTruth, labels: &[LabelPrompt]) -> Vec<SemanticPoint> {
        // A 384x256 frame through NaFlex at 256 patches is about 19 x 13 patches.
        let (cols, rows) = (19u32, 13u32);
        let mut frac: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        let label_of: HashMap<u32, String> = truth
            .objects
            .iter()
            .filter_map(|o| {
                let id: u32 = o.entity.get(1..)?.parse().ok()?;
                let class = match o.class {
                    ObjectClass::Node => ObjectClass::Node,
                    ObjectClass::Drop => ObjectClass::Drop,
                    ObjectClass::Waypoint => ObjectClass::Waypoint,
                    _ => return None,
                };
                Some((world_sim::render::pack_id(class, id), vision_label(o)))
            })
            .collect();
        let cell = |x: u32, y: u32| ((y * rows / truth.height).min(rows - 1) * cols + (x * cols / truth.width).min(cols - 1)) as usize;
        let mut area = vec![0u32; (cols * rows) as usize];
        for y in 0..truth.height {
            for x in 0..truth.width {
                area[cell(x, y)] += 1;
                if let Some(l) = label_of.get(&truth.ids[(y * truth.width + x) as usize]) {
                    frac.entry(l.clone()).or_insert_with(|| vec![0.0; (cols * rows) as usize])[cell(x, y)] += 1.0;
                }
            }
        }
        for v in frac.values_mut() {
            for (f, a) in v.iter_mut().zip(&area) {
                *f /= (*a).max(1) as f32;
            }
        }
        let heats: Vec<(String, Heat, f32)> = labels
            .iter()
            .map(|l| {
                let own = frac.get(&l.id);
                let alike = crate::perception::look_alike(&l.id).and_then(|a| frac.get(a));
                let values: Vec<f32> = (0..(cols * rows) as usize)
                    .map(|i| {
                        let f = own.map_or(0.0, |v| v[i]).max(self.confusion * alike.map_or(0.0, |v| v[i]));
                        (0.12 + 0.75 * f.sqrt() + self.noise_sd * gauss(&mut self.rng)).clamp(0.0, 1.0)
                    })
                    .collect();
                let presence = values.iter().copied().fold(0.0f32, f32::max);
                (l.id.clone(), Heat { cols, rows, values }, presence)
            })
            .collect();
        noto_vision::locate::points_with_presence(truth.src_width, truth.src_height, heats)
    }
}

/// The dictionary's vision label of a visible object. The world's kind ids
/// are the dictionary's ids (`iron_ore_node` for an iron node, `iron_ore` for
/// a dropped piece).
pub fn vision_label(o: &VisibleObject) -> String {
    o.label.clone()
}

pub enum DepthEngine {
    Model(Box<dyn DepthModel>),
    StandIn(DepthStandIn),
}

pub enum LocateEngine {
    Model(Box<dyn SemanticLocator>),
    StandIn(LocateStandIn),
}

/// What the ML tracks made of one keyframe.
pub struct MlFrame {
    pub depth: DepthMap,
    pub depth_metrics: Option<DepthMetrics>,
    pub points: Vec<SemanticPoint>,
    pub tracks: Vec<TrackedMask>,
    pub target: Option<TargetObs>,
    pub locate_err_px: Option<f32>,
    pub track_iou_truth: Option<f32>,
    pub range_true_m: Option<f32>,
    pub truth_entity: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetObs {
    pub track_id: u32,
    pub label: String,
    pub cx: f32,
    pub cy: f32,
    /// World heading of the mask centroid, radians.
    pub heading: f32,
    pub rel_bearing_deg: f32,
    /// Ground range from the depth model at the mask, metres.
    pub depth_range_m: Option<f32>,
    pub score: f32,
}

pub struct MlPipeline {
    pub depth: DepthEngine,
    pub locate: LocateEngine,
    pub tracker: Box<dyn MaskTracker>,
    labels: Vec<LabelPrompt>,
    /// Vision label to resource id (`iron_ore_node` → `iron_ore`).
    link: HashMap<String, String>,
    /// Depth and semantic points that [`MlPipeline::prefetch`] computed in
    /// batches, by image hash, waiting for [`MlPipeline::analyse`].
    pre: HashMap<u64, Prefetched>,
    pre_order: VecDeque<u64>,
}

/// Most keyframes whose batched results wait for `analyse`.
pub const PREFETCH_MAX: usize = 32;

#[derive(Default)]
struct Prefetched {
    depth: Option<DepthMap>,
    points: Option<Vec<SemanticPoint>>,
}

/// FNV-1a over the size and the pixels of a keyframe.
pub fn frame_key(rgb: &Rgb) -> u64 {
    use noto_ml::parallel::{fnv1a, FNV_OFFSET};
    let h = fnv1a(fnv1a(FNV_OFFSET, &rgb.width.to_le_bytes()), &rgb.height.to_le_bytes());
    fnv1a(h, &rgb.data)
}

impl MlPipeline {
    pub fn stand_in(lex: &Lexicon, seed: u64) -> MlPipeline {
        let labels: Vec<LabelPrompt> = lex
            .vision
            .iter()
            .filter(|e| matches!(e.kind, EntityKind::Source | EntityKind::Drop))
            .map(|e| LabelPrompt {
                id: e.id.clone(),
                prompts: std::iter::once(e.semantic.clone()).chain(e.bias.iter().map(|p| p.text.clone())).collect(),
                negatives: e.prejudice.iter().map(|p| p.text.clone()).collect(),
            })
            .collect();
        let link = lex.vision.iter().map(|e| (e.id.clone(), e.link.clone())).collect();
        MlPipeline {
            depth: DepthEngine::StandIn(DepthStandIn::new(seed)),
            locate: LocateEngine::StandIn(LocateStandIn::new(seed)),
            tracker: Box::new(RegionTracker::default()),
            labels,
            link,
            pre: HashMap::new(),
            pre_order: VecDeque::new(),
        }
    }

    /// Runs the stages that batch (depth, semantic location, the mask
    /// model's image encoder) for keyframes [`MlPipeline::analyse`] will see
    /// next: one batched call per model instead of one call per keyframe, on
    /// whatever device each model was loaded to. Tracking itself stays
    /// sequential because each keyframe's prompts depend on the last one's
    /// tracks. The stand-ins need the ground truth and cost nothing, so they
    /// are left to `analyse`. Returns how many keyframes were computed.
    pub fn prefetch(&mut self, frames: &[&Rgb]) -> Result<usize> {
        let mut seen = std::collections::HashSet::new();
        let todo: Vec<(u64, &Rgb)> = frames
            .iter()
            .map(|f| (frame_key(f), *f))
            .filter(|(k, _)| !self.pre.contains_key(k) && seen.insert(*k))
            .take(PREFETCH_MAX)
            .collect();
        let sel: Vec<&Rgb> = todo.iter().map(|(_, f)| *f).collect();
        if sel.is_empty() {
            return Ok(0);
        }
        self.tracker.prepare(&sel)?;
        let depths = match &mut self.depth {
            DepthEngine::Model(m) => Some(m.infer_batch(&sel)?),
            DepthEngine::StandIn(_) => None,
        };
        let points = match &mut self.locate {
            LocateEngine::Model(m) => Some(m.locate_batch(&sel, &self.labels)?),
            LocateEngine::StandIn(_) => None,
        };
        if depths.is_none() && points.is_none() {
            return Ok(0);
        }
        let mut depths = depths.map(|v| v.into_iter().map(Some).collect::<Vec<_>>());
        let mut points = points.map(|v| v.into_iter().map(Some).collect::<Vec<_>>());
        for (i, (key, _)) in todo.iter().enumerate() {
            let p = Prefetched { depth: depths.as_mut().and_then(|d| d[i].take()), points: points.as_mut().and_then(|p| p[i].take()) };
            if self.pre.insert(*key, p).is_none() {
                self.pre_order.push_back(*key);
            }
        }
        while self.pre_order.len() > PREFETCH_MAX {
            if let Some(old) = self.pre_order.pop_front() {
                self.pre.remove(&old);
            }
        }
        Ok(todo.len())
    }

    /// Keyframes waiting with prefetched results.
    pub fn prefetched(&self) -> usize {
        self.pre.len()
    }

    pub fn depth_mode(&self) -> String {
        match &self.depth {
            DepthEngine::Model(m) => m.mode().to_string(),
            DepthEngine::StandIn(_) => "stand-in(tipsv2-like)".into(),
        }
    }

    pub fn locate_mode(&self) -> String {
        match &self.locate {
            LocateEngine::Model(m) => m.mode().to_string(),
            LocateEngine::StandIn(_) => "stand-in(siglip2-like)".into(),
        }
    }

    pub fn track_mode(&self) -> String {
        self.tracker.mode().to_string()
    }

    /// Device each model runs on (`cpu` for the stand-ins).
    pub fn depth_device(&self) -> String {
        match &self.depth {
            DepthEngine::Model(m) => m.device(),
            DepthEngine::StandIn(_) => "cpu".into(),
        }
    }

    pub fn locate_device(&self) -> String {
        match &self.locate {
            LocateEngine::Model(m) => m.device(),
            LocateEngine::StandIn(_) => "cpu".into(),
        }
    }

    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    /// Source labels (nodes in the world) that yield `resource`.
    pub fn source_labels_for(&self, resource: &str) -> Vec<String> {
        self.labels.iter().filter(|l| self.link.get(&l.id).map_or(false, |r| r == resource) && l.id != resource).map(|l| l.id.clone()).collect()
    }

    /// Runs the tracks on one keyframe. `wanted` are the labels the decision
    /// is after; `floor` maps a label to the score below which its semantic
    /// point is not used as a prompt (the SDS-learned floor, or a default).
    pub fn analyse(&mut self, rgb: &Rgb, intr: &Intrinsics, pose: &Pose, truth: Option<&KeyTruth>, wanted: &[String], floor: &dyn Fn(&str) -> f32, prefer_track: Option<u32>) -> Result<MlFrame> {
        let key = frame_key(rgb);
        let pre = self.pre.remove(&key).unwrap_or_default();
        self.pre_order.retain(|k| *k != key);
        let depth = match (pre.depth, &mut self.depth, truth) {
            (Some(d), DepthEngine::Model(_), _) => d,
            (_, DepthEngine::Model(m), _) => m.infer(rgb)?,
            (_, DepthEngine::StandIn(s), Some(t)) => s.run(t),
            (_, DepthEngine::StandIn(_), None) => anyhow::bail!("깊이 대역 모델에는 정답 버퍼가 필요합니다"),
        };
        let points = match (pre.points, &mut self.locate, truth) {
            (Some(p), LocateEngine::Model(_), _) => p,
            (_, LocateEngine::Model(m), _) => m.locate(rgb, &self.labels)?,
            (_, LocateEngine::StandIn(s), Some(t)) => s.run(t, &self.labels),
            (_, LocateEngine::StandIn(_), None) => anyhow::bail!("의미 위치 대역 모델에는 정답 버퍼가 필요합니다"),
        };
        let prompts: Vec<PointPrompt> = points
            .iter()
            .filter(|p| wanted.contains(&p.label) && p.score >= floor(&p.label))
            .map(|p| PointPrompt { label: p.label.clone(), px: p.px, py: p.py, score: p.score })
            .collect();
        let tracks = self.tracker.update(rgb, Some(&depth), &prompts)?;
        let pick = tracks
            .iter()
            .filter(|t| wanted.contains(&t.label))
            .max_by(|a, b| {
                let k = |t: &TrackedMask| (prefer_track == Some(t.id)) as u8 as f32 * 10.0 + t.score;
                k(a).total_cmp(&k(b))
            });
        let target = pick.map(|t| {
            let d = depth.median_in(&t.mask);
            TargetObs {
                track_id: t.id,
                label: t.label.clone(),
                cx: t.cx,
                cy: t.cy,
                heading: intr.bearing(pose, t.cx, t.cy),
                rel_bearing_deg: intr.relative_bearing(pose, t.cx, t.cy).to_degrees(),
                depth_range_m: d.map(|d| intr.ground_range(pose.yaw, t.cx, t.cy, d)),
                score: t.score,
            }
        });
        // Evaluation against the renderer (never read by the decision).
        let (mut depth_m, mut loc_err, mut iou, mut range_true, mut entity) = (None, None, None, None, None);
        if let Some(t) = truth {
            depth_m = depth_metrics(&depth, &t.depth_map(), 24.0, false);
            if let Some(p) = points.iter().find(|p| wanted.contains(&p.label)) {
                loc_err = t
                    .objects
                    .iter()
                    .filter(|o| vision_label(o) == p.label)
                    .map(|o| {
                        let (cx, cy) = ((o.bbox[0] + o.bbox[2]) as f32 * 0.5, (o.bbox[1] + o.bbox[3]) as f32 * 0.5);
                        (cx - p.px).hypot(cy - p.py)
                    })
                    .min_by(f32::total_cmp);
            }
            if let Some(tr) = pick {
                if let Some((raw, _)) = t.entity_under(&tr.mask) {
                    iou = Some(t.iou(&tr.mask, raw));
                    let (class, id) = unpack_id(raw);
                    let name = match class {
                        ObjectClass::Node => format!("n{id}"),
                        _ => format!("d{id}"),
                    };
                    if let Some((_, _, x, z)) = t.positions.iter().find(|(e, _, _, _)| *e == name) {
                        range_true = Some((x - pose.x).hypot(z - pose.z));
                    }
                    entity = Some(name);
                }
            }
        }
        Ok(MlFrame { depth, depth_metrics: depth_m, points, tracks, target, locate_err_px: loc_err, track_iou_truth: iou, range_true_m: range_true, truth_entity: entity })
    }
}

/// The depth map in colour with the target's mask outlined and the semantic
/// peaks marked: the picture of timeline row 2.
pub fn overlay(f: &MlFrame, src_w: u32, src_h: u32, wanted: &[String]) -> Rgb {
    let mut img = f.depth.colorize();
    let (w, h) = (img.width, img.height);
    let put = |img: &mut Rgb, x: i64, y: i64, c: [u8; 3]| {
        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
            let i = ((y as u32 * w + x as u32) * 3) as usize;
            img.data[i..i + 3].copy_from_slice(&c);
        }
    };
    for t in &f.tracks {
        let target = f.target.as_ref().map_or(false, |x| x.track_id == t.id);
        let c = if target { [255, 255, 255] } else { [160, 160, 170] };
        let m = &t.mask;
        for y in 0..m.height {
            for x in 0..m.width {
                if !m.get(x, y) {
                    continue;
                }
                let edge = x == 0 || y == 0 || x + 1 == m.width || y + 1 == m.height || !m.get(x - 1, y) || !m.get(x + 1, y) || !m.get(x, y - 1) || !m.get(x, y + 1);
                if edge {
                    let (px, py) = ((x * w / m.width) as i64, (y * h / m.height) as i64);
                    put(&mut img, px, py, c);
                }
            }
        }
    }
    for p in f.points.iter().filter(|p| wanted.contains(&p.label)) {
        let (cx, cy) = ((p.px / src_w as f32 * w as f32) as i64, (p.py / src_h as f32 * h as f32) as i64);
        for d in -4i64..=4 {
            put(&mut img, cx + d, cy, [255, 60, 200]);
            put(&mut img, cx, cy + d, [255, 60, 200]);
        }
    }
    img
}
