use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use noto_ml::parallel::{aligned_tile_grids, embed_frame_tiles, pool_dense_tiles, BatchOpts, ImageEmbedder, PhraseBank, TextEmbedder, Tile};
use noto_ml::preprocess::{pyramid_grid, DenseGrid, PYRAMID_SHORT_SIDE};
use noto_ml::Siglip2;
use noto_search::semantic::PhraseEmbedder;
use noto_search::vision::EntityScore;
use noto_search::{Lexicon, VisionBank};
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use world_sim::render::View;
use world_sim::{decode_webp, encode_webp, Camera, Frame};

use crate::perception::{box_iou_and_iomin, centre_from_surface, centre_offset_m, Candidate, EntKind, EntityTable, Recognizer, ScanFilter, ScanInput, ScanOutput, SizePrior};

pub const SCAN_CALIB_MODE: &str = "embedding";

/// Which tiles are worth asking SigLIP2 about: a tile that is mostly sky, or whose nearest solid content is farther
/// than any object the robot could act on, only ever produces look-alike guesses, so it is not embedded and cannot
/// win (or lose) a non-maximum suppression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoiCfg {
    pub enabled: bool,
    pub sky_max_frac: f32,
    pub far_m: f32,
    pub min_valid_frac: f32,
}

impl Default for RoiCfg {
    fn default() -> Self {
        RoiCfg { enabled: true, sky_max_frac: 0.6, far_m: 18.0, min_valid_frac: 0.25 }
    }
}

/// Checks a tile's label against what the depth says about the thing in it: how tall it stands above the ground it
/// is on, and whether it touches that ground. A tree is metres tall, a boulder under a metre and a half, a log or a
/// stone on the ground a few centimetres; SigLIP2 mixes them up on blocky renders, the geometry does not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyCfg {
    pub enabled: bool,
    pub top_k: usize,
    pub sigma_ln: f32,
    pub lift_m: f32,
    pub drop_below: f32,
    pub cross_iou: f32,
    pub cross_iomin: f32,
    pub same_object_m: f32,
}

impl Default for VerifyCfg {
    fn default() -> Self {
        VerifyCfg { enabled: true, top_k: 3, sigma_ln: 0.35, lift_m: 0.3, drop_below: 0.15, cross_iou: 0.5, cross_iomin: 0.7, same_object_m: 0.8 }
    }
}

/// The tile scan on the 1 : 2 : 4 pyramid (L0): the tile grids snap to whole blocks of SigLIP2's pyramid grid, one
/// dense pass over the frame gives every coarser tile by pooling its patches, and only the finest level's tiles that
/// passed the ROI are cropped and encoded. Off by default: pooled tiles score differently from crops, so the scan's
/// calibration has to be measured for them first.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DenseTileCfg {
    pub enabled: bool,
    pub crop_finest: bool,
}

impl Default for DenseTileCfg {
    fn default() -> Self {
        DenseTileCfg { enabled: false, crop_finest: true }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCfg {
    pub grids: Vec<(u32, u32)>,
    pub overlap: f32,
    pub prejudice_weight: f32,
    pub report_floor: f32,
    pub nms_iou: f32,
    pub webp_quality: f32,
    #[serde(default)]
    pub roi: RoiCfg,
    #[serde(default)]
    pub verify: VerifyCfg,
    #[serde(default)]
    pub dense: DenseTileCfg,
}

impl Default for EmbeddingCfg {
    fn default() -> Self {
        EmbeddingCfg { grids: vec![(2, 2), (4, 3), (6, 4)], overlap: 0.15, prejudice_weight: 0.5, report_floor: 0.10, nms_iou: 0.4, webp_quality: 80.0, roi: RoiCfg::default(), verify: VerifyCfg::default(), dense: DenseTileCfg::default() }
    }
}

pub struct BankedPhrases<E: TextEmbedder + Send + Sync> {
    embedder: Arc<E>,
    bank: PhraseBank,
    opts: BatchOpts,
}

impl<E: TextEmbedder + Send + Sync> BankedPhrases<E> {
    pub fn new(embedder: Arc<E>, dir: Option<&Path>) -> Self {
        let bank = match dir {
            Some(d) => PhraseBank::open(d),
            None => PhraseBank::memory(),
        };
        BankedPhrases { embedder, bank, opts: BatchOpts::default() }
    }

    pub fn bank(&self) -> &PhraseBank {
        &self.bank
    }
}

impl<E: TextEmbedder + Send + Sync> PhraseEmbedder for BankedPhrases<E> {
    fn embed_phrases(&self, phrases: &[String]) -> Result<Vec<Arc<[f32]>>> {
        self.bank.embed(&*self.embedder, phrases, &self.opts).map(|(v, _)| v)
    }
}

pub type SiglipPhrases = BankedPhrases<Siglip2>;

pub struct EmbeddingPerception {
    image: Arc<dyn ImageEmbedder + Send + Sync>,
    bank: VisionBank,
    table: EntityTable,
    calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
    cfg: EmbeddingCfg,
    device: String,
    calib: Option<noto_vision::CalibTable>,
}

impl EmbeddingPerception {
    pub fn new(
        lex: &Lexicon,
        text: &dyn PhraseEmbedder,
        image: Arc<dyn ImageEmbedder + Send + Sync>,
        calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
        device: &str,
        cfg: EmbeddingCfg,
    ) -> Result<EmbeddingPerception> {
        EmbeddingPerception::new_with(lex, text, image, calibrate, device, cfg, None)
    }

    pub fn new_with(
        lex: &Lexicon,
        text: &dyn PhraseEmbedder,
        image: Arc<dyn ImageEmbedder + Send + Sync>,
        calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
        device: &str,
        cfg: EmbeddingCfg,
        prompting: Option<&noto_search::VisionPrompting>,
    ) -> Result<EmbeddingPerception> {
        let bank = VisionBank::build_with(lex, text, prompting)?;
        Ok(EmbeddingPerception { image, bank, table: EntityTable::from_lexicon(lex), calibrate, cfg, device: device.to_string(), calib: None })
    }

    pub fn set_calibration(&mut self, table: Option<noto_vision::CalibTable>) {
        self.calib = table;
    }

    pub fn calibration(&self) -> Option<&noto_vision::CalibTable> {
        self.calib.as_ref()
    }

    fn calibrated(&self, label: &str, kind: &noto_search::EntityKind, score: f32) -> f32 {
        let Some(t) = &self.calib else { return score };
        if *kind == noto_search::EntityKind::Waypoint {
            return score;
        }
        t.labels.get(label).or_else(|| t.labels.get("*")).and_then(|c| c.probability(score)).unwrap_or(score)
    }

    pub fn siglip(model: Arc<Siglip2>, lex: &Lexicon, bank_dir: Option<&Path>, cfg: EmbeddingCfg) -> Result<EmbeddingPerception> {
        EmbeddingPerception::siglip_with(model, lex, bank_dir, cfg, None)
    }

    pub fn siglip_with(model: Arc<Siglip2>, lex: &Lexicon, bank_dir: Option<&Path>, cfg: EmbeddingCfg, prompting: Option<&noto_search::VisionPrompting>) -> Result<EmbeddingPerception> {
        let phrases = BankedPhrases::new(Arc::clone(&model), bank_dir);
        let device = noto_ml::device_label(model.device());
        let m = Arc::clone(&model);
        EmbeddingPerception::new_with(lex, &phrases, model, Arc::new(move |c| m.pair_probability(c)), &device, cfg, prompting)
    }

    pub fn cfg(&self) -> &EmbeddingCfg {
        &self.cfg
    }

    /// The tile grids a `width` × `height` frame is cut into: the configured ones, snapped to the pyramid's SigLIP2
    /// grid when tiles are pooled from the dense pass.
    pub fn tile_grids(&self, width: u32, height: u32) -> Vec<(u32, u32)> {
        if !self.cfg.dense.enabled {
            return self.cfg.grids.clone();
        }
        let (rows, cols) = pyramid_grid(width, height, PYRAMID_SHORT_SIDE, noto_ml::Siglip2Config::default().max_patches);
        aligned_tile_grids(&self.cfg.grids, cols as u32, rows as u32)
    }

    /// Tile vectors from one dense pass (coarser levels pooled, the finest cropped), or every tile cropped when the
    /// embedder has no dense pass.
    fn dense_tiles(&self, img: &DynamicImage, tiles: &[Tile]) -> Result<Vec<Vec<f32>>> {
        let dense = self.image.embed_dense_images(std::slice::from_ref(img), DenseGrid::pyramid())?.and_then(|mut v| v.pop());
        let Some(dense) = dense else {
            return Ok(embed_frame_tiles(&*self.image, &[(img, tiles)])?.pop().unwrap_or_default());
        };
        let finest = tiles.iter().map(|t| t.level).max().unwrap_or(0);
        let crop = |t: &Tile| self.cfg.dense.crop_finest && finest > 0 && t.level == finest;
        let crops: Vec<Tile> = tiles.iter().filter(|t| crop(t)).copied().collect();
        let pooled: Vec<Tile> = tiles.iter().filter(|t| !crop(t)).copied().collect();
        let mut cropped = if crops.is_empty() { Vec::new() } else { embed_frame_tiles(&*self.image, &[(img, &crops)])?.pop().unwrap_or_default() }.into_iter();
        let mut pooled = pool_dense_tiles(&dense, img.width(), img.height(), &pooled).into_iter();
        Ok(tiles.iter().filter_map(|t| if crop(t) { cropped.next() } else { pooled.next() }).collect())
    }
}

impl Recognizer for EmbeddingPerception {
    fn mode(&self) -> &str {
        "embedding"
    }

    fn scope(&self) -> String {
        if self.calib.is_some() {
            "embedding+cal".into()
        } else {
            "embedding".into()
        }
    }

    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput> {
        let f = input.frame;
        let (w, h) = (f.width as u32, f.height as u32);
        let enc = encode_webp(&f.rgb, w, h, self.cfg.webp_quality).map_err(|e| anyhow!("스크린샷 WebP 인코딩 실패: {e}"))?;
        let (dw, dh, rgb) = decode_webp(&enc.bytes).map_err(|e| anyhow!("스크린샷 WebP 디코딩 실패: {e}"))?;
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(dw, dh, rgb).ok_or_else(|| anyhow!("디코딩한 프레임의 크기가 맞지 않습니다"))?);

        let hv = input.world.head_view(input.camera);
        let view = hv.est;
        let sc = TileScene { frame: f, camera: input.camera, view: &view, ground_y: view.origin[1] - hv.eye_m };
        let all = noto_ml::parallel::tile_layout(dw, dh, &self.tile_grids(dw, dh), self.cfg.overlap);
        let (tiles, mut filter) = roi_tiles(&all, &sc, &self.cfg.roi);
        let t0 = Instant::now();
        let vecs = if self.cfg.dense.enabled { self.dense_tiles(&img, &tiles)? } else { embed_frame_tiles(&*self.image, &[(&img, &tiles)])?.pop().unwrap_or_default() };
        let embed_ms = t0.elapsed().as_secs_f32() * 1000.0;
        if vecs.len() != tiles.len() {
            return Err(anyhow!("타일 {}개에 벡터 {}개가 돌아왔습니다", tiles.len(), vecs.len()));
        }
        let mut scores: Vec<Vec<EntityScore>> = vecs.iter().map(|v| self.bank.score(v, &*self.calibrate, self.cfg.prejudice_weight)).collect();
        if self.calib.is_some() {
            for sc in scores.iter_mut() {
                for e in sc.iter_mut() {
                    e.score = self.calibrated(&e.id, &e.kind, e.score);
                }
                sc.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }

        let scene = scores.first().map(|s| s.iter().map(|e| (e.id.clone(), e.score)).collect()).unwrap_or_default();
        let (candidates, verified) = candidates_from_tiles(&tiles, &scores, &self.table, &sc, &self.cfg);
        filter.relabeled = verified.relabeled;
        filter.dropped = verified.dropped;
        filter.merged = verified.merged;
        Ok(ScanOutput { candidates, tiles: tiles.len(), embed_ms, device: self.device.clone(), scene, filter })
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (ix0, iy0, ix1, iy1) = (a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3]));
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// What the scan knows about the frame besides its pixels: the head sensor's depth, the camera, and the height of
/// the ground the robot stands on (eye height from the joint chain).
pub struct TileScene<'a> {
    pub frame: &'a Frame,
    pub camera: &'a Camera,
    pub view: &'a View,
    pub ground_y: f32,
}

impl TileScene<'_> {
    fn depth(&self, x: u32, y: u32) -> f32 {
        self.frame.depth[y as usize * self.frame.width + x as usize]
    }

    fn point(&self, x: u32, y: u32) -> Option<[f32; 3]> {
        self.camera.world_point3_in(self.view, x as f32 + 0.5, y as f32 + 0.5, self.depth(x, y))
    }

    fn flat(&self, x: u32, y: u32) -> Option<f32> {
        let p = self.camera.world_point3_in(self.view, x as f32 + 0.5, y as f32 + 0.5, 1.0)?;
        let dy = p[1] - self.view.origin[1];
        (dy < -1e-4).then(|| (self.ground_y - self.view.origin[1]) / dy)
    }

    /// A surface rising at 45° or more between this pixel and the one below it: the face of something standing,
    /// not ground (a far slope of the terrain stays under 45°).
    fn steep(&self, x: u32, y: u32) -> bool {
        if y + 1 >= self.frame.height as u32 {
            return false;
        }
        let (Some(a), Some(b)) = (self.point(x, y), self.point(x, y + 1)) else { return false };
        let (dy, dh) = (a[1] - b[1], (a[0] - b[0]).hypot(a[2] - b[2]));
        dy > 0.0 && dy >= dh && dh < 0.3
    }

    /// Stands up out of the ground: above the horizon, clearly nearer than the robot's ground plane along the ray,
    /// or on a steep face.
    fn solid(&self, x: u32, y: u32) -> bool {
        let d = self.depth(x, y);
        d.is_finite() && self.flat(x, y).map_or(true, |g| d < 0.8 * g - 0.05 || self.steep(x, y))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    pub sky_frac: f32,
    pub valid_frac: f32,
    pub near_m: f32,
}

pub fn tile_roi(sc: &TileScene, t: &Tile) -> Roi {
    let (w, h) = (sc.frame.width as u32, sc.frame.height as u32);
    let (mut sky, mut n) = (0u32, 0u32);
    let mut finite: Vec<f32> = Vec::new();
    let step = (t.w.min(t.h) / 24).max(1);
    let mut y = t.y;
    while y < (t.y + t.h).min(h) {
        let mut x = t.x;
        while x < (t.x + t.w).min(w) {
            n += 1;
            let d = sc.depth(x, y);
            if d.is_finite() {
                finite.push(d);
            } else {
                sky += 1;
            }
            x += step;
        }
        y += step;
    }
    finite.sort_by(f32::total_cmp);
    let near_m = if finite.is_empty() { f32::INFINITY } else { finite[finite.len() * 3 / 10] };
    Roi { sky_frac: sky as f32 / n.max(1) as f32, valid_frac: finite.len() as f32 / n.max(1) as f32, near_m }
}

/// Keeps the whole frame (it scores the scene) and the tiles with something near enough to matter in them.
pub fn roi_tiles(tiles: &[Tile], sc: &TileScene, cfg: &RoiCfg) -> (Vec<Tile>, ScanFilter) {
    let mut f = ScanFilter::default();
    if !cfg.enabled {
        return (tiles.to_vec(), f);
    }
    let kept = tiles
        .iter()
        .filter(|t| {
            if t.level == 0 {
                return true;
            }
            let r = tile_roi(sc, t);
            if r.sky_frac > cfg.sky_max_frac || r.valid_frac < cfg.min_valid_frac {
                f.skipped_sky += 1;
                false
            } else if r.near_m > cfg.far_m {
                f.skipped_far += 1;
                false
            } else {
                true
            }
        })
        .copied()
        .collect();
    (kept, f)
}

fn median_depth(f: &Frame, t: &Tile) -> Option<f32> {
    let (x0, x1) = (t.x + t.w / 4, (t.x + t.w * 3 / 4).max(t.x + t.w / 4 + 1));
    let (y0, y1) = (t.y + t.h / 4, (t.y + t.h * 3 / 4).max(t.y + t.h / 4 + 1));
    let mut v: Vec<f32> = Vec::new();
    for y in y0..y1.min(f.height as u32) {
        for x in x0..x1.min(f.width as u32) {
            let d = f.depth[y as usize * f.width + x as usize];
            if d.is_finite() {
                v.push(d);
            }
        }
    }
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

/// The thing in a tile, measured from the depth: how far its top stands above the ground it is on (`top_m`), how
/// far the bottom of its typical column floats above that ground (`lift_m`: most of a tree's columns see only the
/// crown, a flag floats on its pole, a boulder or a log sits on the ground), and whether the frame cut its top off
/// (`clipped`, so `top_m` is only a lower bound).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Extent {
    pub top_m: f32,
    pub lift_m: f32,
    pub clipped: bool,
    pub columns: u32,
}

pub fn object_extent(sc: &TileScene, t: &Tile) -> Option<Extent> {
    let (w, h) = (sc.frame.width as u32, sc.frame.height as u32);
    let near = |a: f32, b: f32| (a - b).abs() < 0.25 + 0.1 * a.min(b);
    let (x0, x1) = (t.x + t.w / 5, (t.x + t.w * 4 / 5).min(w));
    let (ty0, ty1) = (t.y, (t.y + t.h).min(h));
    let mut solid_d: Vec<f32> = Vec::new();
    for x in (x0..x1).step_by(2) {
        for y in (ty0..ty1).step_by(2) {
            if sc.solid(x, y) {
                solid_d.push(sc.depth(x, y));
            }
        }
    }
    if solid_d.len() < 8 {
        return None;
    }
    solid_d.sort_by(f32::total_cmp);
    let reference = solid_d[solid_d.len() / 4];
    let tol = 0.35 + 0.15 * reference;
    let on = |x: u32, y: u32| sc.solid(x, y) && (sc.depth(x, y) - reference).abs() <= tol;
    let mut cols: Vec<(f32, f32, Option<f32>, bool)> = Vec::new();
    for x in (x0..x1).step_by(2) {
        let mut best: Option<(u32, u32)> = None;
        let mut y = ty0;
        while y < ty1 {
            if !on(x, y) {
                y += 1;
                continue;
            }
            let start = y;
            while y + 1 < ty1 && on(x, y + 1) && near(sc.depth(x, y), sc.depth(x, y + 1)) {
                y += 1;
            }
            if best.map_or(true, |(a, b)| y - start > b - a) {
                best = Some((start, y));
            }
            y += 1;
        }
        let Some((mut top, mut bot)) = best else { continue };
        while top > 0 && on(x, top - 1) && near(sc.depth(x, top), sc.depth(x, top - 1)) {
            top -= 1;
        }
        while bot + 1 < h && on(x, bot + 1) && near(sc.depth(x, bot), sc.depth(x, bot + 1)) {
            bot += 1;
        }
        let (Some(pt), Some(pb)) = (sc.point(x, top), sc.point(x, bot)) else { continue };
        let below = if bot + 1 < h { sc.point(x, bot + 1).map(|g| g[1]) } else { None };
        cols.push((pt[1], pb[1], below, top == 0));
    }
    if cols.len() < 3 {
        return None;
    }
    let mut contact: Vec<f32> = cols.iter().filter_map(|c| c.2.filter(|g| (c.1 - g).abs() < 0.06)).collect();
    contact.sort_by(f32::total_cmp);
    let base = match contact.get(contact.len() / 2) {
        Some(g) => *g,
        None => cols.iter().filter_map(|c| c.2).fold(sc.ground_y, f32::min),
    };
    let mut tops: Vec<f32> = cols.iter().map(|c| c.0 - base).collect();
    let mut lifts: Vec<f32> = cols.iter().map(|c| (c.1 - base).max(0.0)).collect();
    tops.sort_by(f32::total_cmp);
    lifts.sort_by(f32::total_cmp);
    let top_m = tops[(tops.len() * 4 / 5).min(tops.len() - 1)].max(0.0);
    Some(Extent { top_m, lift_m: lifts[lifts.len() / 2], clipped: cols.iter().any(|c| c.3), columns: cols.len() as u32 })
}

/// How well a size prior fits a measured extent, in 0..1: 1 inside the prior's height band, falling off as a
/// log-normal outside it; a floating blob fits only things that may float (a crown, a flag).
pub fn plausibility(p: &SizePrior, e: &Extent, cfg: &VerifyCfg) -> f32 {
    let ln = if e.top_m > p.hi_m {
        (e.top_m / p.hi_m).ln()
    } else if e.top_m < p.lo_m && !e.clipped {
        (p.lo_m / e.top_m.max(0.01)).ln()
    } else {
        0.0
    };
    let mut g = (-0.5 * (ln / cfg.sigma_ln.max(0.05)).powi(2)).exp();
    if !p.elevated && e.lift_m > cfg.lift_m {
        g *= (-0.5 * ((e.lift_m - cfg.lift_m) / 0.15).powi(2)).exp();
    }
    g
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verified {
    pub relabeled: u32,
    pub dropped: u32,
    pub merged: u32,
}

/// One candidate per tile (its best label, checked against the geometry), then non-maximum suppression: first among
/// tiles of one label, then across labels for boxes on the same object, so a boulder is not also reported as a log.
pub fn candidates_from_tiles(tiles: &[Tile], scores: &[Vec<EntityScore>], table: &EntityTable, sc: &TileScene, cfg: &EmbeddingCfg) -> (Vec<Candidate>, Verified) {
    let frame = sc.frame;
    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let mut out: Vec<Candidate> = Vec::new();
    let mut v = Verified::default();
    for (tile, sc_tile) in tiles.iter().zip(scores) {
        if tile.level == 0 {
            continue;
        }
        let Some(first) = sc_tile.first() else { continue };
        if first.score < cfg.report_floor {
            continue;
        }
        let ranked: Vec<(&EntityScore, f32)> = match (cfg.verify.enabled, object_extent(sc, tile)) {
            (true, Some(ext)) => {
                let mut r: Vec<(&EntityScore, f32)> = sc_tile.iter().take(cfg.verify.top_k.max(1)).map(|e| (e, e.score * table.prior(&e.id).map_or(1.0, |p| plausibility(p, &ext, &cfg.verify)))).collect();
                r.sort_by(|a, b| b.1.total_cmp(&a.1));
                if r.iter().all(|(e, s)| *s < cfg.verify.drop_below * e.score.max(1e-6)) {
                    v.dropped += 1;
                    continue;
                }
                r
            }
            _ => sc_tile.iter().take(2).map(|e| (e, e.score)).collect(),
        };
        let (best, score) = ranked[0];
        if score < cfg.report_floor {
            v.dropped += 1;
            continue;
        }
        if best.id != first.id {
            v.relabeled += 1;
        }
        let Some((kind, link)) = table.get(&best.id).cloned() else { continue };
        let runner = ranked.get(1);
        let bbox = [tile.x as f32 / fw, tile.y as f32 / fh, (tile.x + tile.w) as f32 / fw, (tile.y + tile.h) as f32 / fh];
        let depth = median_depth(frame, tile);
        let (cx, cy) = tile.center();
        let world = depth.and_then(|d| sc.camera.world_point3_in(sc.view, cx, cy, d)).map(|p| centre_from_surface(sc.view, p, centre_offset_m(kind)));
        out.push(Candidate {
            label: best.id.clone(),
            kind,
            link,
            score,
            margin: score - runner.map_or(0.0, |r| r.1),
            runner_up: runner.map(|r| r.0.id.clone()),
            bbox,
            pixels: tile.area(),
            depth_m: depth.unwrap_or(f32::INFINITY),
            world,
            truth: None,
            truth_label: None,
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in out {
        if kept.iter().all(|k| k.label != c.label || iou(&k.bbox, &c.bbox) < cfg.nms_iou) {
            kept.push(c);
        }
    }
    if cfg.verify.enabled {
        let mut cross: Vec<Candidate> = Vec::new();
        for c in kept {
            let same = cross.iter_mut().find(|k| {
                if k.label == c.label || k.kind == EntKind::Waypoint || c.kind == EntKind::Waypoint {
                    return false;
                }
                let (u, m) = box_iou_and_iomin(&k.bbox, &c.bbox);
                let close = match (k.world, c.world) {
                    (Some(a), Some(b)) => (a.0 - b.0).hypot(a.1 - b.1) <= cfg.verify.same_object_m,
                    _ => u >= cfg.verify.cross_iou,
                };
                close && (u >= cfg.verify.cross_iou || m >= cfg.verify.cross_iomin)
            });
            match same {
                Some(k) => {
                    v.merged += 1;
                    if k.margin > k.score - c.score {
                        k.margin = (k.score - c.score).max(0.0);
                        k.runner_up = Some(c.label.clone());
                    }
                }
                None => cross.push(c),
            }
        }
        kept = cross;
    }
    kept.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.label.cmp(&b.label)));
    (kept, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_search::semantic::cosine;
    use world_sim::kinds::NodeKind;
    use world_sim::world::Node;
    use world_sim::{World, WorldConfig};

    fn lexicon() -> Lexicon {
        Lexicon::from_json(include_str!("../../../data/world_lexicon.json")).expect("lexicon loads")
    }

    fn world_with_tree_ahead() -> (World, Camera) {
        world_with_ahead(NodeKind::Tree, 3.2)
    }

    fn world_with_ahead(kind: NodeKind, dist_m: f32) -> (World, Camera) {
        let mut w = World::new(WorldConfig::default()).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let (cx, cz) = w.terrain.cell_of(w.agent.x + dist_m, w.agent.z + 0.2).unwrap();
        let (x, z) = world_sim::terrain::cell_center(cx, cz);
        w.nodes.push(Node { id: 1, kind, x, z, hp: 3.0, max_hp: 3.0 });
        w.agent.yaw = 0.0;
        (w, Camera::default())
    }

    fn with_scene<R>(w: &World, f: &Frame, cam: &Camera, run: impl FnOnce(&TileScene) -> R) -> R {
        let hv = w.head_view(cam);
        let sc = TileScene { frame: f, camera: cam, view: &hv.est, ground_y: hv.est.origin[1] - hv.eye_m };
        run(&sc)
    }

    fn score(id: &str, s: f32) -> EntityScore {
        EntityScore { id: id.into(), kind: noto_search::EntityKind::Source, link: "log".into(), bias: s, prejudice: 0.0, score: s }
    }

    fn tile_with_most(tiles: &[Tile], f: &Frame, w: &World, label: &str) -> usize {
        let target = w.visible_objects(f).into_iter().find(|v| v.label == label).expect("in view").bbox;
        let (cx, cy) = (((target[0] + target[2]) / 2) as f32, ((target[1] + target[3]) / 2) as f32);
        tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.level > 0 && cx >= t.x as f32 && cx < (t.x + t.w) as f32 && cy >= t.y as f32 && cy < (t.y + t.h) as f32)
            .min_by_key(|(_, t)| t.area())
            .map(|(i, _)| i)
            .expect("a tile holds the centre")
    }

    #[test]
    fn a_tile_that_names_the_tree_becomes_a_candidate_on_the_tree() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(2, 2), (4, 3)], 0.15);
        let table = EntityTable::from_lexicon(&lexicon());
        let hit = tile_with_most(&tiles, &f, &w, "tree");
        let scores: Vec<Vec<EntityScore>> =
            (0..tiles.len()).map(|i| if i == hit { vec![score("tree", 0.82), score("boulder", 0.30)] } else { vec![score("boulder", 0.05)] }).collect();
        let (cands, _) = with_scene(&w, &f, &cam, |sc| candidates_from_tiles(&tiles, &scores, &table, sc, &EmbeddingCfg::default()));
        assert_eq!(cands.len(), 1, "only the named tile reports: {cands:?}");
        let c = &cands[0];
        assert_eq!(c.label, "tree");
        assert_eq!(c.runner_up.as_deref(), Some("boulder"));
        assert!(c.margin >= 0.52 - 1e-5, "the crown floats, which no boulder does: margin {}", c.margin);
        let t = &tiles[hit];
        assert_eq!(c.bbox, [t.x as f32 / f.width as f32, t.y as f32 / f.height as f32, (t.x + t.w) as f32 / f.width as f32, (t.y + t.h) as f32 / f.height as f32]);
        let node = &w.nodes[0];
        let (x, z) = c.world.expect("depth known");
        let err = ((x - node.x).powi(2) + (z - node.z).powi(2)).sqrt();
        assert!(err < 1.5, "estimated {x:.2},{z:.2} vs tree {:.2},{:.2}", node.x, node.z);
        assert!(c.truth.is_none() && c.truth_label.is_none(), "a real recogniser knows no ground truth");
    }

    #[test]
    fn overlapping_tiles_of_one_label_merge_and_a_rival_label_on_the_same_tree_folds_in() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(4, 3)], 0.5);
        let table = EntityTable::from_lexicon(&lexicon());
        let hit = tile_with_most(&tiles, &f, &w, "tree");
        let scores: Vec<Vec<EntityScore>> = (0..tiles.len())
            .map(|i| if i == hit { vec![score("boulder", 0.7), score("tree", 0.1)] } else { vec![score("tree", 0.9 - 0.01 * i as f32), score("boulder", 0.1)] })
            .collect();
        let plain = EmbeddingCfg { verify: VerifyCfg { enabled: false, ..VerifyCfg::default() }, ..EmbeddingCfg::default() };
        let (old, _) = with_scene(&w, &f, &cam, |sc| candidates_from_tiles(&tiles, &scores, &table, sc, &plain));
        let trees: Vec<&Candidate> = old.iter().filter(|c| c.label == "tree").collect();
        assert!(trees.len() < tiles.len() - 1, "NMS merged some of the {} tree tiles, left {}", tiles.len() - 2, trees.len());
        for (i, a) in trees.iter().enumerate() {
            for b in &trees[i + 1..] {
                assert!(iou(&a.bbox, &b.bbox) < 0.4);
            }
        }
        assert!(old.iter().any(|c| c.label == "boulder"), "per-label suppression alone keeps a boulder on the tree");
        let (new, v) = with_scene(&w, &f, &cam, |sc| candidates_from_tiles(&tiles, &scores, &table, sc, &EmbeddingCfg::default()));
        let node = &w.nodes[0];
        let on_tree = |c: &&Candidate| c.world.is_some_and(|(x, z)| (x - node.x).hypot(z - node.z) < 1.0);
        assert!(new.iter().filter(on_tree).all(|c| c.label == "tree"), "{:?}", new.iter().map(|c| (&c.label, c.score, c.world)).collect::<Vec<_>>());
        assert!(v.relabeled + v.merged > 0, "{v:?}");
    }

    #[test]
    fn tiles_below_the_report_floor_say_nothing() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(2, 2)], 0.0);
        let table = EntityTable::from_lexicon(&lexicon());
        let scores: Vec<Vec<EntityScore>> = tiles.iter().map(|_| vec![score("tree", 0.09)]).collect();
        let (cands, _) = with_scene(&w, &f, &cam, |sc| candidates_from_tiles(&tiles, &scores, &table, sc, &EmbeddingCfg::default()));
        assert!(cands.is_empty());
    }

    #[test]
    fn sky_and_far_tiles_are_left_out_before_they_are_embedded() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &EmbeddingCfg::default().grids, 0.15);
        with_scene(&w, &f, &cam, |sc| {
            let (kept, filter) = roi_tiles(&tiles, sc, &RoiCfg::default());
            assert!(filter.skipped_sky > 0, "the top row of tiles is sky: {filter:?}");
            assert_eq!(kept.len() + (filter.skipped_sky + filter.skipped_far) as usize, tiles.len());
            assert!(kept.iter().any(|t| t.level == 0), "the whole frame always scores the scene");
            assert!(kept.iter().filter(|t| t.level > 0).all(|t| tile_roi(sc, t).sky_frac <= 0.6));
            let near = RoiCfg { far_m: 2.5, ..RoiCfg::default() };
            let (_, f2) = roi_tiles(&tiles, sc, &near);
            assert!(f2.skipped_far > 0, "a 2.5 m reach leaves the distant ground out: {f2:?}");
            let (all, f0) = roi_tiles(&tiles, sc, &RoiCfg { enabled: false, ..RoiCfg::default() });
            assert!(all.len() == tiles.len() && f0.is_empty());
        });
    }

    #[test]
    fn the_depth_tells_a_tree_from_a_boulder_and_a_boulder_from_a_log() {
        let table = EntityTable::from_lexicon(&lexicon());
        let cfg = EmbeddingCfg::default();
        let probe = |kind: NodeKind, label: &str| {
            let (w, cam) = world_with_ahead(kind, 2.6);
            let f = w.render(&cam);
            let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(4, 3)], 0.15);
            let hit = tile_with_most(&tiles, &f, &w, label);
            with_scene(&w, &f, &cam, |sc| object_extent(sc, &tiles[hit]).expect("the object stands up out of the ground"))
        };
        let (tree, boulder) = (probe(NodeKind::Tree, "tree"), probe(NodeKind::Boulder, "boulder"));
        assert!(tree.top_m > 1.3 && tree.top_m < 1.9, "{tree:?}");
        assert!(boulder.top_m > 0.4 && boulder.top_m < 0.8 && boulder.lift_m < 0.1, "{boulder:?}");
        let g = |label: &str, e: &Extent| plausibility(table.prior(label).unwrap(), e, &cfg.verify);
        assert!(g("tree", &tree) > 0.9 && g("boulder", &tree) < 0.3, "a 1.6 m thing with a floating crown is no boulder: {} {}", g("tree", &tree), g("boulder", &tree));
        assert!(g("boulder", &boulder) > 0.9 && g("log", &boulder) < 0.6 && g("tree", &boulder) < 0.2, "{} {} {}", g("boulder", &boulder), g("log", &boulder), g("tree", &boulder));
        let (w, cam) = world_with_ahead(NodeKind::Tree, 2.6);
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(4, 3)], 0.15);
        let hit = tile_with_most(&tiles, &f, &w, "tree");
        let scores: Vec<Vec<EntityScore>> = (0..tiles.len()).map(|i| if i == hit { vec![score("boulder", 0.62), score("tree", 0.48), score("log", 0.2)] } else { vec![] }).collect();
        let (cands, v) = with_scene(&w, &f, &cam, |sc| candidates_from_tiles(&tiles, &scores, &table, sc, &cfg));
        assert_eq!(cands.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(), vec!["tree"], "SigLIP2's boulder on a tree is relabelled");
        assert_eq!(v.relabeled, 1);
        assert_eq!(cands[0].runner_up.as_deref(), Some("boulder"));
    }

    struct WordAxes;

    const AXES: [&str; 4] = ["tree", "boulder", "log", "warehouse"];

    impl TextEmbedder for WordAxes {
        fn embedding_dim(&self) -> usize {
            AXES.len() + 1
        }
        fn fingerprint(&self) -> u64 {
            7
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v: Vec<f32> = AXES.iter().map(|a| if t.contains(a) { 1.0 } else { 0.0 }).collect();
                    v.push(0.05);
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter().map(|x| x / n).collect()
                })
                .collect())
        }
    }

    struct Greens;

    impl ImageEmbedder for Greens {
        fn embedding_dim(&self) -> usize {
            AXES.len() + 1
        }
        fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
            Ok(images
                .iter()
                .map(|im| {
                    let rgb = im.to_rgb8();
                    let n = (rgb.width() * rgb.height()) as f32;
                    let (mut green, mut grey) = (0.0f32, 0.0f32);
                    for p in rgb.pixels() {
                        let [r, g, b] = p.0;
                        if g > r.saturating_add(20) && g > b.saturating_add(20) {
                            green += 1.0;
                        } else if (r as i32 - g as i32).abs() < 12 && (g as i32 - b as i32).abs() < 12 {
                            grey += 1.0;
                        }
                    }
                    let mut v = vec![green / n, grey / n, 0.0, 0.0, 0.05];
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter_mut().for_each(|x| *x /= norm);
                    v
                })
                .collect())
        }
    }

    /// `Greens` with a dense pass: every patch is the whole frame's `Greens` vector, the attention uniform.
    struct DenseGreens {
        crops: std::sync::atomic::AtomicUsize,
    }

    impl ImageEmbedder for DenseGreens {
        fn embedding_dim(&self) -> usize {
            Greens.embedding_dim()
        }
        fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
            self.crops.fetch_add(images.len(), std::sync::atomic::Ordering::Relaxed);
            Greens.embed_images_batch(images)
        }
        fn embed_dense_images(&self, images: &[DynamicImage], grid: DenseGrid) -> Result<Option<Vec<noto_ml::DenseEmbedding>>> {
            let whole = Greens.embed_images_batch(images)?;
            Ok(Some(
                images
                    .iter()
                    .zip(whole)
                    .map(|(im, v)| {
                        let (gh, gw) = grid.grid_for(im.width(), im.height(), 16, 256);
                        noto_ml::DenseEmbedding { grid_h: gh, grid_w: gw, vectors: vec![v.clone(); gh * gw], attention: vec![1.0 / (gh * gw) as f32; gh * gw], pooled: v }
                    })
                    .collect(),
            ))
        }
    }

    #[test]
    fn dense_tiles_pool_the_coarse_levels_and_crop_only_the_finest() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let lex = lexicon();
        let phrases = BankedPhrases::new(Arc::new(WordAxes), None);
        let cfg = EmbeddingCfg { webp_quality: 100.0, report_floor: 0.2, dense: DenseTileCfg { enabled: true, crop_finest: true }, ..EmbeddingCfg::default() };
        let emb = Arc::new(DenseGreens { crops: Default::default() });
        let mut p = EmbeddingPerception::new(&lex, &phrases, emb.clone(), Arc::new(|c| c), "cpu (stand-in)", cfg).unwrap();
        assert_eq!(p.tile_grids(384, 256), vec![(2, 2), (3, 3), (6, 4)], "whole blocks of the 18 x 12 SigLIP2 grid");
        let out = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(out.tiles + (out.filter.skipped_sky + out.filter.skipped_far) as usize, 1 + 4 + 9 + 24);
        let crops = emb.crops.load(std::sync::atomic::Ordering::Relaxed);
        assert!(crops > 0 && crops <= 24 && crops < out.tiles, "only finest tiles that passed the ROI are encoded: {crops} of {}", out.tiles);
        assert!(!out.scene.is_empty());
        let plain = perception();
        assert_eq!(plain.tile_grids(384, 256), EmbeddingCfg::default().grids, "off by default");
    }

    fn perception() -> EmbeddingPerception {
        let lex = lexicon();
        let phrases = BankedPhrases::new(Arc::new(WordAxes), None);
        let cfg = EmbeddingCfg { webp_quality: 100.0, report_floor: 0.2, ..EmbeddingCfg::default() };
        EmbeddingPerception::new(&lex, &phrases, Arc::new(Greens), Arc::new(|c| c), "cpu (stand-in)", cfg).unwrap()
    }

    #[test]
    fn a_whole_scan_goes_screenshot_tiles_scores_candidates() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let mut p = perception();
        assert_eq!(p.mode(), "embedding");
        let out = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(out.tiles + (out.filter.skipped_sky + out.filter.skipped_far) as usize, 1 + 4 + 12 + 24);
        assert!(out.filter.skipped_sky > 0, "sky tiles are not embedded: {:?}", out.filter);
        assert_eq!(out.device, "cpu (stand-in)");
        assert!(!out.scene.is_empty(), "the whole-frame scores are reported");
        assert!(out.candidates.iter().all(|c| c.truth.is_none()));
        assert!(out.candidates.iter().any(|c| c.label == "tree"), "grass and leaves are green: {:?}", out.candidates.iter().map(|c| (&c.label, c.score)).collect::<Vec<_>>());
        for c in &out.candidates {
            assert!(c.bbox.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(c.score >= 0.2);
        }
        let again = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(again.candidates, out.candidates);
    }

    #[test]
    fn a_calibration_maps_entity_scores_but_leaves_places_raw() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let mut p = perception();
        let raw = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(Recognizer::scope(&p), "embedding");
        let mut t = noto_vision::CalibTable { mode: SCAN_CALIB_MODE.into(), statistic: "tile_score".into(), ..Default::default() };
        t.labels.insert("tree".into(), noto_vision::LabelCalib { threshold: 1.0, platt: Some((10.0, -10.0)), positives: 5, negatives: 5, precision: None, recall: None });
        p.set_calibration(Some(t));
        assert_eq!(Recognizer::scope(&p), "embedding+cal", "calibrated statistics stay apart");
        let cal = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        let top = |o: &ScanOutput, id: &str| o.scene.iter().find(|(l, _)| l == id).map(|x| x.1).unwrap();
        let (r, c) = (top(&raw, "tree"), top(&cal, "tree"));
        assert!((c - 1.0 / (1.0 + (-(10.0 * r - 10.0)).exp())).abs() < 1e-5, "{r} -> {c}");
        assert_eq!(top(&raw, "warehouse"), top(&cal, "warehouse"));
        assert_eq!(top(&raw, "boulder"), top(&cal, "boulder"));
    }

    #[test]
    fn banked_phrases_embed_each_phrase_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting(AtomicUsize);
        impl TextEmbedder for Counting {
            fn embedding_dim(&self) -> usize {
                3
            }
            fn fingerprint(&self) -> u64 {
                1
            }
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.0.fetch_add(texts.len(), Ordering::SeqCst);
                Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0, 0.0]).map(|v| {
                    let n = (v[0] * v[0] + 1.0).sqrt();
                    vec![v[0] / n, 1.0 / n, 0.0]
                }).collect())
            }
        }
        let e = Arc::new(Counting(AtomicUsize::new(0)));
        let b = BankedPhrases::new(Arc::clone(&e), None);
        let texts: Vec<String> = ["tree", "tree", "  tree ", "rock"].iter().map(|s| s.to_string()).collect();
        let v = b.embed_phrases(&texts).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(e.0.load(Ordering::SeqCst), 2, "four requests, two distinct phrases");
        assert!((cosine(&v[0], &v[2]) - 1.0).abs() < 1e-6);
        b.embed_phrases(&texts).unwrap();
        assert_eq!(e.0.load(Ordering::SeqCst), 2, "the second call is served from the bank");
    }
}
