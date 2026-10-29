use std::collections::{BTreeMap, HashMap, VecDeque};

use anyhow::Result;
use noto_search::{EntityKind, Lexicon, VisionPrompting};
use noto_vision::{
    depth_metrics, AlignCfg, AlignFit, CalibTable, DepthKind, DepthMap, DepthMetrics, DepthModel, Heat, Intrinsics, LabelPrompt, LocateCfg, LocateScoring, Mask, MaskTracker,
    PointPrompt, Pose, Pyramid, PyramidCfg, PyramidReport, RegionTracker, Rgb, SemanticLocator, SemanticPoint, TrackedMask,
};
use serde::{Deserialize, Serialize};
use world_sim::render::unpack_id;
use world_sim::rng::Rng;
use world_sim::{Frame, ObjectClass, VisibleObject, World};

pub const TIPS_MAX_DEPTH_M: f32 = 10.0;
pub const STANDIN_DEPTH_SCALE: f32 = 0.8;

pub const ML_CALIB_JSON: &str = include_str!("../../../data/ml_calib.json");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlFixCfg {
    #[serde(default)]
    pub align: Option<AlignCfg>,
    #[serde(default)]
    pub contact_range: bool,
    #[serde(default = "default_pitch_sigma")]
    pub pitch_sigma_deg: f32,
    #[serde(default = "default_depth_rel_sigma")]
    pub depth_rel_sigma: f32,
    #[serde(default)]
    pub calibrated_gate: bool,
    #[serde(default)]
    pub locate: LocateCfg,
    #[serde(default)]
    pub scan_prompting: bool,
    #[serde(default)]
    pub scan_calibrated: bool,
    #[serde(default)]
    pub sky_mask: bool,
    #[serde(default)]
    pub normals: bool,
    /// The 1 : 2 : 4 pyramid: SigLIP2, TIPSv2 and the SAM mask on whole multiples of one grid, the depth's sky and
    /// far patches gating the semantic heat, peaks snapped onto the nearer surface, SAM encoding only frames with a
    /// gated prompt or a live track.
    #[serde(default)]
    pub pyramid: Option<PyramidCfg>,
}

/// The depth model's extra heads a configuration asks for: the ADE20K segmentation to take the sky out of the
/// depth, the surface normals to keep objects out of the ground fit.
pub fn scene_heads(fix: &MlFixCfg) -> noto_vision::SceneHeads {
    noto_vision::SceneHeads { normals: fix.normals && fix.align.is_some(), segmentation: fix.sky_mask }
}

fn default_pitch_sigma() -> f32 {
    0.5
}

fn default_depth_rel_sigma() -> f32 {
    0.05
}

impl MlFixCfg {
    pub fn legacy() -> MlFixCfg {
        MlFixCfg {
            align: None,
            contact_range: false,
            pitch_sigma_deg: default_pitch_sigma(),
            depth_rel_sigma: default_depth_rel_sigma(),
            calibrated_gate: false,
            locate: LocateCfg::default(),
            scan_prompting: false,
            scan_calibrated: false,
            sky_mask: false,
            normals: false,
            pyramid: None,
        }
    }

    pub fn v1(prompting: &VisionPrompting) -> MlFixCfg {
        let shipped = CalibTable::parse(ML_CALIB_JSON).unwrap_or_default();
        let locate = shipped
            .iter()
            .find(|t| t.mode.starts_with("siglip2-dense"))
            .and_then(|t| LocateCfg::from_mode(&t.mode, &prompting.background, &prompting.templates))
            .unwrap_or_else(|| LocateCfg { scoring: LocateScoring::Contrast, background: prompting.background.clone(), ..LocateCfg::default() });
        MlFixCfg {
            align: Some(AlignCfg::default()),
            contact_range: true,
            pitch_sigma_deg: default_pitch_sigma(),
            depth_rel_sigma: default_depth_rel_sigma(),
            calibrated_gate: true,
            locate,
            scan_prompting: false,
            scan_calibrated: true,
            sky_mask: true,
            normals: false,
            pyramid: Some(PyramidCfg::default()),
        }
    }
}

impl Default for MlFixCfg {
    fn default() -> Self {
        MlFixCfg::v1(&VisionPrompting::default())
    }
}

#[derive(Clone, Debug)]
pub struct KeyTruth {
    pub width: u32,
    pub height: u32,
    pub ids: Vec<u32>,
    pub depth: Vec<f32>,
    pub objects: Vec<VisibleObject>,
    pub positions: Vec<(String, String, f32, f32)>,
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

    /// The mask's bit over the truth pixel (x, y): the same pixel when both have one size, else the mask pixel under
    /// the truth pixel's centre (a pyramid mask is 288 × 192 on a 384 × 256 frame, the truth 192 × 128).
    fn mask_at(&self, mask: &Mask, x: u32, y: u32) -> bool {
        if (mask.width, mask.height) == (self.width, self.height) {
            return mask.get(x, y);
        }
        let mx = (((x as f32 + 0.5) * mask.width as f32 / self.width.max(1) as f32) as u32).min(mask.width.saturating_sub(1));
        let my = (((y as f32 + 0.5) * mask.height as f32 / self.height.max(1) as f32) as u32).min(mask.height.saturating_sub(1));
        mask.get(mx, my)
    }

    pub fn entity_under(&self, mask: &Mask) -> Option<(u32, f32)> {
        if mask.width == 0 || mask.height == 0 {
            return None;
        }
        let mut count: HashMap<u32, u32> = HashMap::new();
        let mut area = 0u32;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.mask_at(mask, x, y) {
                    area += 1;
                    *count.entry(self.ids[(y * self.width + x) as usize]).or_insert(0) += 1;
                }
            }
        }
        let (raw, n) = count.into_iter().filter(|(raw, _)| matches!(unpack_id(*raw).0, ObjectClass::Node | ObjectClass::Drop)).max_by_key(|(_, n)| *n)?;
        Some((raw, n as f32 / area.max(1) as f32))
    }

    pub fn iou(&self, mask: &Mask, raw: u32) -> f32 {
        if mask.width == 0 || mask.height == 0 {
            return 0.0;
        }
        let (mut inter, mut uni) = (0u32, 0u32);
        for y in 0..self.height {
            for x in 0..self.width {
                let a = self.mask_at(mask, x, y);
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthErrModel {
    pub a: f32,
    pub b: f32,
    pub frame_sd: f32,
    pub patch_sd: f32,
    pub sky_m: f32,
    pub cap_m: f32,
    #[serde(default = "never_unknown")]
    pub unknown_beyond_m: f32,
    pub blur_px: u32,
}

fn never_unknown() -> f32 {
    f32::INFINITY
}

impl DepthErrModel {
    pub const TIPS_RAW: DepthErrModel = DepthErrModel { a: 0.0, b: 0.0, frame_sd: 0.0, patch_sd: 0.0, sky_m: TIPS_MAX_DEPTH_M, cap_m: TIPS_MAX_DEPTH_M, unknown_beyond_m: f32::INFINITY, blur_px: 3 };
    pub const TIPS_GROUND: DepthErrModel = DepthErrModel { a: 0.0, b: 0.0, frame_sd: 0.0, patch_sd: 0.0, sky_m: f32::INFINITY, cap_m: f32::INFINITY, unknown_beyond_m: f32::INFINITY, blur_px: 3 };
}

pub struct DepthStandIn {
    rng: Rng,
    pub patch_sd: f32,
    pub err: Option<DepthErrModel>,
    /// The depth patches follow the pyramid's L2 grid while set (else TIPSv2's 36 × 24 keyframe grid).
    pub pyramid: Option<PyramidCfg>,
}

impl DepthStandIn {
    pub fn new(seed: u64) -> DepthStandIn {
        DepthStandIn { rng: Rng::new(seed ^ 0xDE97), patch_sd: 0.06, err: None, pyramid: None }
    }

    pub fn honest(seed: u64) -> DepthStandIn {
        DepthStandIn { err: Some(DepthErrModel::TIPS_GROUND), ..DepthStandIn::new(seed) }
    }

    fn patch_grid(&self, truth: &KeyTruth) -> (u32, u32) {
        match &self.pyramid {
            Some(p) => Pyramid::for_frame(truth.src_width, truth.src_height, p).l2(),
            None => (36, 24),
        }
    }

    pub fn run(&mut self, truth: &KeyTruth) -> DepthMap {
        if let Some(m) = self.err {
            return self.run_measured(truth, &m);
        }
        let (pw, ph) = self.patch_grid(truth);
        let errs: Vec<f32> = (0..pw * ph).map(|_| (self.patch_sd * gauss(&mut self.rng)).exp()).collect();
        let mut data = Vec::with_capacity(truth.depth.len());
        for y in 0..truth.height {
            for x in 0..truth.width {
                let d = truth.depth[(y * truth.width + x) as usize];
                let d = if d.is_finite() { d } else { TIPS_MAX_DEPTH_M };
                let e = errs[((y * ph / truth.height).min(ph - 1) * pw + (x * pw / truth.width).min(pw - 1)) as usize];
                data.push((d * STANDIN_DEPTH_SCALE * e).clamp(1e-3, TIPS_MAX_DEPTH_M));
            }
        }
        DepthMap { width: truth.width, height: truth.height, kind: DepthKind::Metric, data }
    }

    fn run_measured(&mut self, truth: &KeyTruth, m: &DepthErrModel) -> DepthMap {
        let (pw, ph) = self.patch_grid(truth);
        let eta = m.frame_sd * gauss(&mut self.rng);
        let errs: Vec<f32> = (0..pw * ph).map(|_| m.patch_sd * gauss(&mut self.rng)).collect();
        let (w, h) = (truth.width, truth.height);
        let mut ln: Vec<f32> = Vec::with_capacity(truth.depth.len());
        for y in 0..h {
            for x in 0..w {
                let t = truth.depth[(y * w + x) as usize];
                let e = errs[((y * ph / h).min(ph - 1) * pw + (x * pw / w).min(pw - 1)) as usize];
                ln.push(if t.is_finite() && t > 0.0 && t < m.unknown_beyond_m {
                    m.a + eta + e + (1.0 + m.b) * t.ln()
                } else if t.is_finite() && t > 0.0 {
                    f32::INFINITY
                } else {
                    m.sky_m.ln()
                });
            }
        }
        let ln = box_blur(&ln, w, h, m.blur_px);
        let data = ln.into_iter().map(|l| if l.is_finite() { l.exp().clamp(1e-3, m.cap_m) } else { f32::INFINITY }).collect();
        DepthMap { width: w, height: h, kind: DepthKind::Metric, data }
    }
}

fn box_blur(v: &[f32], w: u32, h: u32, r: u32) -> Vec<f32> {
    if r == 0 {
        return v.to_vec();
    }
    let (w, h, r) = (w as i64, h as i64, r as i64);
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for y in 0..h {
            for x in 0..w {
                let here = src[(y * w + x) as usize];
                if !here.is_finite() {
                    out[(y * w + x) as usize] = here;
                    continue;
                }
                let (mut s, mut n) = (0.0f32, 0u32);
                for k in -r..=r {
                    let (xx, yy) = if horizontal { ((x + k).clamp(0, w - 1), y) } else { (x, (y + k).clamp(0, h - 1)) };
                    let v = src[(yy * w + xx) as usize];
                    if v.is_finite() {
                        s += v;
                        n += 1;
                    }
                }
                out[(y * w + x) as usize] = s / n.max(1) as f32;
            }
        }
        out
    };
    pass(&pass(v, true), false)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocateErrModel {
    pub hit_a: f32,
    pub hit_b: f32,
    pub pass_miss: f32,
    pub pass_absent: f32,
    pub score_pass: (f32, f32),
    pub score_block: (f32, f32),
    pub gate_p: f32,
}

impl LocateErrModel {
    pub const SIGLIP_MEASURED: LocateErrModel = LocateErrModel { hit_a: 0.0, hit_b: 0.0, pass_miss: 0.0, pass_absent: 0.0, score_pass: (0.6, 0.1), score_block: (0.1, 0.1), gate_p: 0.5 };
}

pub struct LocateStandIn {
    rng: Rng,
    pub noise_sd: f32,
    pub confusion: f32,
    pub err: Option<LocateErrModel>,
    /// The heat follows the pyramid's L1 grid while set (else SigLIP2 NaFlex's 19 × 13 keyframe grid).
    pub pyramid: Option<PyramidCfg>,
}

impl LocateStandIn {
    pub fn new(seed: u64) -> LocateStandIn {
        LocateStandIn { rng: Rng::new(seed ^ 0x51C1), noise_sd: 0.04, confusion: 0.35, err: None, pyramid: None }
    }

    pub fn honest(seed: u64) -> LocateStandIn {
        LocateStandIn { err: Some(LocateErrModel::SIGLIP_MEASURED), ..LocateStandIn::new(seed) }
    }

    fn heat_grid(&self, truth: &KeyTruth) -> (u32, u32) {
        match &self.pyramid {
            Some(p) => Pyramid::for_frame(truth.src_width, truth.src_height, p).l1(),
            None => (19, 13),
        }
    }

    pub fn run(&mut self, truth: &KeyTruth, labels: &[LabelPrompt]) -> Vec<SemanticPoint> {
        if let Some(m) = self.err {
            return self.run_measured(truth, labels, &m);
        }
        let (cols, rows) = self.heat_grid(truth);
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

    fn run_measured(&mut self, truth: &KeyTruth, labels: &[LabelPrompt], m: &LocateErrModel) -> Vec<SemanticPoint> {
        let (cols, rows) = self.heat_grid(truth);
        let n = (cols * rows) as usize;
        let cell_of = |o: &VisibleObject| {
            let (cx, cy) = ((o.bbox[0] + o.bbox[2]) as f32 * 0.5, (o.bbox[1] + o.bbox[3]) as f32 * 0.5);
            let gx = ((cx / truth.src_width.max(1) as f32) * cols as f32).clamp(0.0, cols as f32 - 1.0) as u32;
            let gy = ((cy / truth.src_height.max(1) as f32) * rows as f32).clamp(0.0, rows as f32 - 1.0) as u32;
            (gy * cols + gx) as usize
        };
        let gate = m.gate_p.clamp(1e-3, 1.0);
        let pass = |rng: &mut Rng| (m.score_pass.0 + m.score_pass.1 * gauss(rng)).clamp(gate, 1.0);
        let block = |rng: &mut Rng| (m.score_block.0 + m.score_block.1 * gauss(rng)).clamp(0.0, gate - 1e-3);
        let heats: Vec<(String, Heat, f32)> = labels
            .iter()
            .map(|l| {
                let own = truth.objects.iter().filter(|o| vision_label(o) == l.id).max_by_key(|o| o.pixels);
                let u = self.rng.next_f32();
                let random_cell = self.rng.below(n as u32) as usize;
                let (cell, score) = match own {
                    Some(o) => {
                        let p_hit = 1.0 / (1.0 + (-(m.hit_a * (o.pixels.max(1) as f32).ln() + m.hit_b)).exp());
                        if u < p_hit {
                            (cell_of(o), pass(&mut self.rng))
                        } else if self.rng.next_f32() < m.pass_miss {
                            (random_cell, pass(&mut self.rng))
                        } else {
                            (cell_of(o), block(&mut self.rng))
                        }
                    }
                    None if u < m.pass_absent => (random_cell, pass(&mut self.rng)),
                    None => (random_cell, block(&mut self.rng)),
                };
                let values: Vec<f32> = (0..n).map(|i| if i == cell { score } else { score * (0.2 + 0.6 * self.rng.next_f32()) }).collect();
                (l.id.clone(), Heat { cols, rows, values }, score)
            })
            .collect();
        noto_vision::locate::points_with_presence(truth.src_width, truth.src_height, heats)
    }
}

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

pub struct MlFrame {
    pub depth: DepthMap,
    pub align: Option<AlignFit>,
    pub depth_metrics: Option<DepthMetrics>,
    pub points: Vec<SemanticPoint>,
    pub tracks: Vec<TrackedMask>,
    pub target: Option<TargetObs>,
    pub locate_err_px: Option<f32>,
    pub track_iou_truth: Option<f32>,
    pub range_true_m: Option<f32>,
    pub truth_entity: Option<String>,
    pub pyramid: Option<PyramidReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetObs {
    pub track_id: u32,
    pub label: String,
    pub cx: f32,
    pub cy: f32,
    pub heading: f32,
    pub rel_bearing_deg: f32,
    pub depth_range_m: Option<f32>,
    pub range_sigma_m: Option<f32>,
    pub score: f32,
}

pub struct MlPipeline {
    pub depth: DepthEngine,
    pub locate: LocateEngine,
    pub tracker: Box<dyn MaskTracker>,
    labels: Vec<LabelPrompt>,
    link: HashMap<String, String>,
    drops: std::collections::HashSet<String>,
    pub fix: MlFixCfg,
    pub prompting: VisionPrompting,
    pub calib: Vec<CalibTable>,
    pre: HashMap<u64, Prefetched>,
    pre_order: VecDeque<u64>,
}

pub const PREFETCH_MAX: usize = 32;

#[derive(Default)]
struct Prefetched {
    depth: Option<noto_vision::DepthScene>,
    points: Option<Vec<SemanticPoint>>,
}

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
        let drops = lex.vision.iter().filter(|e| e.kind == EntityKind::Drop).map(|e| e.id.clone()).collect();
        MlPipeline {
            depth: DepthEngine::StandIn(DepthStandIn::new(seed)),
            locate: LocateEngine::StandIn(LocateStandIn::new(seed)),
            tracker: Box::new(RegionTracker::default()),
            labels,
            link,
            drops,
            fix: MlFixCfg::default(),
            prompting: VisionPrompting::default(),
            calib: CalibTable::parse(ML_CALIB_JSON).unwrap_or_default(),
            pre: HashMap::new(),
            pre_order: VecDeque::new(),
        }
    }

    pub fn use_honest_stand_ins(&mut self, seed: u64) {
        if let DepthEngine::StandIn(_) = self.depth {
            self.depth = DepthEngine::StandIn(DepthStandIn::honest(seed));
        }
        if let LocateEngine::StandIn(_) = self.locate {
            self.locate = LocateEngine::StandIn(LocateStandIn::honest(seed));
        }
        self.sync_pyramid();
    }

    /// Hands the configuration's pyramid (or none) to every engine: SigLIP2's dense grid, TIPSv2's input grid, the
    /// SAM mask grid, the stand-ins' heat and depth patches. Call after swapping an engine; `prefetch` and `analyse`
    /// call it too.
    pub fn sync_pyramid(&mut self) {
        let p = self.fix.pyramid;
        match &mut self.depth {
            DepthEngine::Model(m) => m.set_pyramid(p.as_ref()),
            DepthEngine::StandIn(s) => s.pyramid = p,
        }
        match &mut self.locate {
            LocateEngine::Model(m) => m.set_pyramid(p.as_ref()),
            LocateEngine::StandIn(s) => s.pyramid = p,
        }
        self.tracker.set_pyramid(p.as_ref());
    }

    /// The prompts a frame's points give the mask tracker: wanted labels above their floors that pass the
    /// calibrated gate.
    fn prompts_for(&self, points: &[SemanticPoint], wanted: Option<&[String]>, floor: &dyn Fn(&str) -> f32) -> Vec<PointPrompt> {
        points
            .iter()
            .filter(|p| wanted.map_or(true, |w| w.contains(&p.label)) && p.score >= floor(&p.label) && self.passes_calibration(p))
            .map(|p| PointPrompt { label: p.label.clone(), px: p.px, py: p.py, score: p.score })
            .collect()
    }

    pub fn labels(&self) -> &[LabelPrompt] {
        &self.labels
    }

    pub fn is_drop(&self, label: &str) -> bool {
        self.drops.contains(label)
    }

    pub fn calib_table(&self) -> Option<&CalibTable> {
        if !self.fix.calibrated_gate || matches!(self.locate, LocateEngine::StandIn(_)) {
            return None;
        }
        let mode = self.locate_mode();
        self.calib.iter().find(|t| t.mode == mode)
    }

    pub fn passes_calibration(&self, p: &SemanticPoint) -> bool {
        if let LocateEngine::StandIn(LocateStandIn { err: Some(m), .. }) = &self.locate {
            return p.score >= m.gate_p;
        }
        let Some(t) = self.calib_table() else { return true };
        let Some(c) = t.labels.get(&p.label) else { return true };
        let stat = match t.statistic.as_str() {
            "presence" => p.presence,
            "peak_x_presence" => p.score * p.presence,
            _ => p.score,
        };
        stat >= c.threshold
    }

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
        self.sync_pyramid();
        let lazy = self.fix.pyramid.is_some_and(|p| p.lazy_sam) && matches!(self.locate, LocateEngine::Model(_));
        if !lazy {
            self.tracker.prepare(&sel)?;
        }
        let heads = scene_heads(&self.fix);
        let depths = match &mut self.depth {
            DepthEngine::Model(m) => Some(m.infer_scene_batch(&sel, heads)?),
            DepthEngine::StandIn(_) => None,
        };
        let points = match &mut self.locate {
            LocateEngine::Model(m) => Some(m.locate_batch(&sel, &self.labels)?),
            LocateEngine::StandIn(_) => None,
        };
        if lazy {
            let any: Vec<&Rgb> = match &points {
                Some(pts) => sel.iter().zip(pts).filter(|(_, p)| !self.prompts_for(p, None, &|_| f32::NEG_INFINITY).is_empty()).map(|(f, _)| *f).collect(),
                None => sel.clone(),
            };
            if !any.is_empty() {
                self.tracker.prepare(&any)?;
            }
        }
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

    pub fn prefetched(&self) -> usize {
        self.pre.len()
    }

    pub fn depth_mode(&self) -> String {
        match &self.depth {
            DepthEngine::Model(m) if self.fix.align.is_some() => format!("{}+ground", m.mode()),
            DepthEngine::Model(m) => m.mode().to_string(),
            DepthEngine::StandIn(s) if s.err.is_some() => "stand-in(tipsv2-measured)".into(),
            DepthEngine::StandIn(_) => "stand-in(tipsv2-like)".into(),
        }
    }

    pub fn locate_mode(&self) -> String {
        match &self.locate {
            LocateEngine::Model(m) => m.mode().to_string(),
            LocateEngine::StandIn(s) if s.err.is_some() => "stand-in(siglip2-measured)".into(),
            LocateEngine::StandIn(_) => "stand-in(siglip2-like)".into(),
        }
    }

    pub fn track_mode(&self) -> String {
        self.tracker.mode().to_string()
    }

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

    pub fn source_labels_for(&self, resource: &str) -> Vec<String> {
        self.labels.iter().filter(|l| self.link.get(&l.id).map_or(false, |r| r == resource) && l.id != resource).map(|l| l.id.clone()).collect()
    }

    pub fn analyse(&mut self, rgb: &Rgb, intr: &Intrinsics, pose: &Pose, truth: Option<&KeyTruth>, wanted: &[String], floor: &dyn Fn(&str) -> f32, prefer_track: Option<u32>) -> Result<MlFrame> {
        self.analyse_with(rgb, intr, pose, truth, wanted, floor, prefer_track, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn analyse_with(
        &mut self,
        rgb: &Rgb,
        intr: &Intrinsics,
        pose: &Pose,
        truth: Option<&KeyTruth>,
        wanted: &[String],
        floor: &dyn Fn(&str) -> f32,
        prefer_track: Option<u32>,
        height: Option<noto_vision::align::HeightFn>,
    ) -> Result<MlFrame> {
        self.sync_pyramid();
        let key = frame_key(rgb);
        let pre = self.pre.remove(&key).unwrap_or_default();
        self.pre_order.retain(|k| *k != key);
        let heads = scene_heads(&self.fix);
        let scene = match (pre.depth, &mut self.depth, truth) {
            (Some(d), DepthEngine::Model(_), _) => d,
            (_, DepthEngine::Model(m), _) => m.infer_scene(rgb, heads)?,
            (_, DepthEngine::StandIn(s), Some(t)) => noto_vision::DepthScene::depth_only(s.run(t)),
            (_, DepthEngine::StandIn(_), None) => anyhow::bail!("깊이 대역 모델에는 정답 버퍼가 필요합니다"),
        };
        let mut depth = scene.depth;
        let seg = scene.segmentation;
        let normals = scene.normals.filter(|_| self.fix.normals);
        let align = match (&self.depth, &self.fix.align) {
            (DepthEngine::Model(m), Some(cfg)) => noto_vision::ground_align_with(&mut depth, normals.as_ref(), intr, pose, height, m.max_depth_m(), cfg),
            _ => None,
        };
        let mut points = match (pre.points, &mut self.locate, truth) {
            (Some(p), LocateEngine::Model(_), _) => p,
            (_, LocateEngine::Model(m), _) => m.locate(rgb, &self.labels)?,
            (_, LocateEngine::StandIn(s), Some(t)) => s.run(t, &self.labels),
            (_, LocateEngine::StandIn(_), None) => anyhow::bail!("의미 위치 대역 모델에는 정답 버퍼가 필요합니다"),
        };
        let pyramid = self.fix.pyramid.map(|cfg| {
            let pyr = Pyramid::for_frame(rgb.width, rgb.height, &cfg);
            let view = noto_vision::pyramid::View { intr, pose, height };
            let mut rep = PyramidReport::of(&pyr);
            if cfg.gate.enabled {
                let geo = noto_vision::cell_geometry(&pyr, &depth, seg.as_ref(), Some(view), cfg.gate.far_m);
                rep.gated_cells = noto_vision::gate_points(&mut points, &pyr, &geo, &cfg.gate);
            }
            if cfg.snap.enabled {
                rep.snapped = noto_vision::snap_points(&mut points, &pyr, &depth, Some(view), &cfg.snap);
            }
            rep
        });
        let prompts = self.prompts_for(&points, Some(wanted), floor);
        let tracks = self.tracker.update(rgb, Some(&depth), &prompts)?;
        let pick = tracks
            .iter()
            .filter(|t| wanted.contains(&t.label))
            .max_by(|a, b| {
                let k = |t: &TrackedMask| (prefer_track == Some(t.id)) as u8 as f32 * 10.0 + t.score;
                k(a).total_cmp(&k(b))
            });
        let model_depth = matches!(self.depth, DepthEngine::Model(_));
        let target = pick.map(|t| {
            let d = depth.median_in(&t.mask);
            let mut range = d.map(|d| intr.ground_range(pose.yaw, t.cx, t.cy, d));
            let mut sigma = None;
            if model_depth && (self.fix.align.is_some() || self.fix.contact_range) {
                let rel = align.map_or(0.5, |f| f.median_rel_err.max(self.fix.depth_rel_sigma));
                sigma = range.map(|r| rel * r);
                if self.fix.contact_range && self.drops.contains(&t.label) {
                    if let Some((rc, sc)) = noto_vision::contact_range(intr, pose, &t.mask, height, self.fix.pitch_sigma_deg, 20.0) {
                        (range, sigma) = match (range, sigma) {
                            (Some(ra), Some(sa)) if sa > 0.0 => {
                                let (wa, wc) = (1.0 / (sa * sa), 1.0 / (sc * sc));
                                (Some((wa * ra + wc * rc) / (wa + wc)), Some((wa + wc).sqrt().recip()))
                            }
                            _ => (Some(rc), Some(sc)),
                        };
                    }
                }
            }
            TargetObs {
                track_id: t.id,
                label: t.label.clone(),
                cx: t.cx,
                cy: t.cy,
                heading: intr.bearing(pose, t.cx, t.cy),
                rel_bearing_deg: intr.relative_bearing(pose, t.cx, t.cy).to_degrees(),
                depth_range_m: range,
                range_sigma_m: sigma,
                score: t.score,
            }
        });
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
        Ok(MlFrame { depth, align, depth_metrics: depth_m, points, tracks, target, locate_err_px: loc_err, track_iou_truth: iou, range_true_m: range_true, truth_entity: entity, pyramid })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use noto_vision::{CalibTable, LabelCalib};

    const LEX: &str = include_str!("../../../data/world_lexicon.json");

    fn truth_with(objects: Vec<VisibleObject>) -> KeyTruth {
        let (w, h) = (96u32, 64u32);
        let mut depth = Vec::new();
        for y in 0..h {
            for _ in 0..w {
                depth.push(if y < 8 { f32::INFINITY } else { 1.0 + 9.0 * (h - 1 - y) as f32 / (h - 9) as f32 });
            }
        }
        KeyTruth { width: w, height: h, ids: vec![0; (w * h) as usize], depth, objects, positions: Vec::new(), src_width: 2 * w, src_height: 2 * h }
    }

    fn object(label: &str, pixels: u32, bbox: [u32; 4]) -> VisibleObject {
        VisibleObject { class: ObjectClass::Drop, entity: "d1".into(), label: label.into(), label_ko: String::new(), pixels, bbox, depth_m: 3.0 }
    }

    #[test]
    fn the_honest_depth_stand_in_follows_its_error_model() {
        let t = truth_with(Vec::new());
        let m = DepthErrModel { a: 0.1, b: -0.2, frame_sd: 0.0, patch_sd: 0.0, sky_m: f32::INFINITY, cap_m: 100.0, unknown_beyond_m: f32::INFINITY, blur_px: 0 };
        let mut s = DepthStandIn { err: Some(m), ..DepthStandIn::new(3) };
        let d = s.run(&t);
        for (p, q) in d.data.iter().zip(&t.depth) {
            if q.is_finite() {
                let want = (0.1f32).exp() * q.powf(0.8);
                assert!((p - want).abs() < 1e-4 * want, "{p} vs {want}");
            } else {
                assert!(p.is_infinite(), "sky stays unknown");
            }
        }
        let mut s = DepthStandIn { err: Some(DepthErrModel { blur_px: 2, cap_m: 4.0, sky_m: 10.0, ..m }), ..DepthStandIn::new(3) };
        let d = s.run(&t);
        assert!(d.data.iter().all(|v| v.is_finite() && *v <= 4.0 + 1e-6));
        let mut s = DepthStandIn { err: Some(DepthErrModel { unknown_beyond_m: 5.0, ..m }), ..DepthStandIn::new(3) };
        let d = s.run(&t);
        for (p, q) in d.data.iter().zip(&t.depth) {
            assert_eq!(p.is_finite(), q.is_finite() && *q < 5.0, "{q} -> {p}");
        }
        assert_eq!(MlPipeline::stand_in(&Lexicon::from_json(LEX).unwrap(), 1).depth_mode(), "stand-in(tipsv2-like)");
    }

    #[test]
    fn the_honest_locate_stand_in_lands_as_its_outcome_model_says() {
        let lex = Lexicon::from_json(LEX).unwrap();
        let mut pipe = MlPipeline::stand_in(&lex, 1);
        pipe.use_honest_stand_ins(5);
        assert_eq!(pipe.locate_mode(), "stand-in(siglip2-measured)");
        assert_eq!(pipe.depth_mode(), "stand-in(tipsv2-measured)");
        let labels: Vec<LabelPrompt> = pipe.labels().iter().filter(|l| l.id == "parcel").cloned().collect();
        let t = truth_with(vec![object("parcel", 400, [100, 80, 120, 100])]);
        let sure = LocateErrModel { hit_a: 0.0, hit_b: 50.0, pass_miss: 0.0, pass_absent: 0.0, score_pass: (0.9, 0.0), score_block: (0.1, 0.0), gate_p: 0.5 };
        let mut s = LocateStandIn { err: Some(sure), ..LocateStandIn::new(2) };
        for _ in 0..5 {
            let p = s.run(&t, &labels);
            let p = p.iter().find(|p| p.label == "parcel").unwrap();
            assert!((p.score - 0.9).abs() < 1e-6 && (p.presence - 0.9).abs() < 1e-6);
            assert!((p.px - 110.0).abs() <= 192.0 / 19.0 && (p.py - 90.0).abs() <= 128.0 / 13.0, "{} {}", p.px, p.py);
        }
        let p = s.run(&truth_with(Vec::new()), &labels);
        assert!((p[0].score - 0.1).abs() < 1e-6, "absent: blocked");
        let never = LocateErrModel { hit_b: -50.0, ..sure };
        let mut s = LocateStandIn { err: Some(never), ..LocateStandIn::new(2) };
        assert!((s.run(&t, &labels)[0].score - 0.1).abs() < 1e-6);
        let phantom = LocateErrModel { pass_absent: 0.3, score_pass: (0.4, 0.3), score_block: (0.6, 0.3), ..sure };
        let mut s = LocateStandIn { err: Some(phantom), ..LocateStandIn::new(4) };
        let empty = truth_with(Vec::new());
        let passed = (0..1000).filter(|_| s.run(&empty, &labels).first().is_some_and(|p| p.score >= 0.5)).count();
        assert!((250..350).contains(&passed), "{passed} of 1000");
        let half = LocateErrModel { hit_a: 1.0, hit_b: -(400f32).ln(), ..sure };
        let mut s = LocateStandIn { err: Some(half), ..LocateStandIn::new(9) };
        let hits = (0..400).filter(|_| (s.run(&t, &labels)[0].score - 0.9).abs() < 1e-6).count();
        assert!((150..250).contains(&hits), "{hits} of 400");
        pipe.locate = LocateEngine::StandIn(LocateStandIn { err: Some(sure), ..LocateStandIn::new(1) });
        assert!(pipe.passes_calibration(&point("parcel", 0.5, 0.5)) && !pipe.passes_calibration(&point("parcel", 0.49, 0.9)));
    }

    struct Fixed;

    impl SemanticLocator for Fixed {
        fn mode(&self) -> &str {
            "siglip2-dense+bg"
        }
        fn locate(&mut self, _rgb: &Rgb, _labels: &[LabelPrompt]) -> Result<Vec<SemanticPoint>> {
            Ok(Vec::new())
        }
    }

    fn point(label: &str, score: f32, presence: f32) -> SemanticPoint {
        SemanticPoint { label: label.into(), px: 0.0, py: 0.0, score, margin: 0.0, presence, bbox: [0.0; 4], heat: Heat { cols: 1, rows: 1, values: vec![score] } }
    }

    #[test]
    fn the_calibrated_gate_reads_its_table_and_never_gates_a_stand_in() {
        let lex = Lexicon::from_json(LEX).unwrap();
        let mut pipe = MlPipeline::stand_in(&lex, 1);
        let mut t = CalibTable { mode: "siglip2-dense+bg".into(), statistic: "peak_x_presence".into(), ..CalibTable::default() };
        t.labels.insert("parcel".into(), LabelCalib { threshold: 0.3, platt: None, positives: 1, negatives: 1, precision: None, recall: None });
        pipe.calib = vec![t];
        let low = point("parcel", 0.5, 0.5);
        assert!(pipe.passes_calibration(&low));
        pipe.locate = LocateEngine::Model(Box::new(Fixed));
        assert!(!pipe.passes_calibration(&low), "0.25 < 0.3");
        assert!(pipe.passes_calibration(&point("parcel", 0.6, 0.6)));
        assert!(pipe.passes_calibration(&point("log", 0.0, 0.0)), "a label without calibration passes");
        pipe.fix = MlFixCfg::legacy();
        assert!(pipe.passes_calibration(&low), "the old path has no gate");
    }

    #[test]
    fn the_truth_scores_a_mask_of_any_grid_on_its_own_pixels() {
        let mut t = truth_with(Vec::new());
        let raw = world_sim::render::pack_id(ObjectClass::Drop, 7);
        for y in 20..40 {
            for x in 30..54 {
                t.ids[(y * t.width + x) as usize] = raw;
            }
        }
        let mut same = Mask::empty(96, 64);
        let mut fine = Mask::empty(144, 96);
        for y in 20..40 {
            for x in 30..54 {
                same.set(x, y);
            }
        }
        for y in 30..60 {
            for x in 45..81 {
                fine.set(x, y);
            }
        }
        assert_eq!(t.iou(&same, raw), 1.0);
        assert_eq!(t.iou(&fine, raw), 1.0, "a 1.5x mask of the same box scores the same");
        assert_eq!(t.entity_under(&fine), Some((raw, 1.0)));
        assert_eq!(t.iou(&Mask::empty(0, 0), raw), 0.0);
    }

    #[test]
    fn the_pipeline_hands_its_pyramid_to_the_stand_ins() {
        let lex = Lexicon::from_json(LEX).unwrap();
        let mut pipe = MlPipeline::stand_in(&lex, 1);
        pipe.sync_pyramid();
        let t = truth_with(Vec::new());
        let LocateEngine::StandIn(s) = &mut pipe.locate else { panic!() };
        assert!(s.pyramid.is_some());
        let labels: Vec<LabelPrompt> = lex_labels(&lex);
        let pts = s.run(&t, &labels);
        assert_eq!((pts[0].heat.cols, pts[0].heat.rows), (18, 12), "the 192 x 128 frame's SigLIP2 pyramid grid");
        let DepthEngine::StandIn(d) = &pipe.depth else { panic!() };
        assert_eq!(d.patch_grid(&t), (36, 24));
        pipe.fix = MlFixCfg::legacy();
        pipe.sync_pyramid();
        let LocateEngine::StandIn(s) = &mut pipe.locate else { panic!() };
        let pts = s.run(&t, &labels);
        assert_eq!((pts[0].heat.cols, pts[0].heat.rows), (19, 13), "the old path keeps NaFlex's own grid");
    }

    fn lex_labels(lex: &Lexicon) -> Vec<LabelPrompt> {
        MlPipeline::stand_in(lex, 1).labels().iter().take(2).cloned().collect()
    }

    #[test]
    fn the_legacy_fix_is_the_old_path_and_v1_switches_every_fix_on() {
        let l = MlFixCfg::legacy();
        assert!(l.align.is_none() && !l.contact_range && !l.calibrated_gate && !l.scan_calibrated && l.locate == LocateCfg::default());
        let p = VisionPrompting { templates: Vec::new(), background: vec!["green grass".into()] };
        let v = MlFixCfg::v1(&p);
        assert!(v.align.is_some() && v.contact_range && v.calibrated_gate && v.scan_calibrated);
        assert_eq!(v.locate.background, p.background);
        assert!(l.pyramid.is_none(), "the old path runs every model on its own grid");
        assert_eq!(v.pyramid, Some(PyramidCfg::default()), "v1 lines the models up 1 : 2 : 4");
        let back: MlFixCfg = serde_json::from_str("{}").unwrap();
        assert!(back.align.is_none() && back.pyramid.is_none() && (back.pitch_sigma_deg - 0.5).abs() < 1e-6);
        assert!(CalibTable::parse(ML_CALIB_JSON).is_ok());
    }
}
