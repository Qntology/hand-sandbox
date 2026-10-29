use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::image_encoder::ImageEncoderViT;
use crate::sam_decoder::MaskDecoder;
use candle_transformers::models::segment_anything::prompt_encoder::PromptEncoder;
use candle_transformers::models::segment_anything::tiny_vit::{tiny_vit_5m, TinyViT};

use crate::frame::{DepthMap, Mask, Rgb};
use crate::track::{MaskTracker, PointPrompt, TrackedMask};

pub const IMAGE_SIZE: usize = 1024;
pub const EMBED_SIDE: usize = 64;
pub const LOW_RES: usize = 256;
const PROMPT_DIM: usize = 256;
pub const DIR_ENV: &str = "SAM_DIR";
pub const DEFAULT_DIR: &str = "models/sam";
pub const TINY_FILE: &str = "mobile_sam-tiny-vitt.safetensors";
pub const VIT_B_FILE: &str = "sam_vit_b_01ec64.safetensors";
pub const BATCH_ENV: &str = "NOTO_SAM_BATCH";
/// The least margin around a track's box (frame px) within which a prompt of its label follows it.
const FOLLOW_GUARD_PX: f32 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamVariant {
    MobileSam,
    VitB,
    TestVit { dim: usize, depth: usize, heads: usize },
}

impl SamVariant {
    pub fn mode(self) -> &'static str {
        match self {
            SamVariant::MobileSam => "mobile-sam",
            SamVariant::VitB => "sam-vit-b",
            SamVariant::TestVit { .. } => "sam-test",
        }
    }
}

enum Encoder {
    Vit(Box<ImageEncoderViT>),
    Tiny(Box<TinyViT>),
}

impl Encoder {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Encoder::Vit(e) => e.forward(x),
            Encoder::Tiny(e) => e.forward(x),
        }
    }
}

pub struct SamModel {
    variant: SamVariant,
    encoder: Encoder,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    image_pe: Tensor,
    mean: Tensor,
    std: Tensor,
    device: Device,
}

pub struct Decoded {
    pub logits: Vec<f32>,
    pub iou_pred: f32,
}

impl SamModel {
    pub fn find_weights(dir: &Path) -> Option<(PathBuf, SamVariant)> {
        let tiny = dir.join(TINY_FILE);
        let vit = dir.join(VIT_B_FILE);
        let want_vit = std::env::var("NOTO_SAM").map(|v| v.trim().eq_ignore_ascii_case("vit_b")).unwrap_or(false);
        match (tiny.is_file(), vit.is_file()) {
            (_, true) if want_vit => Some((vit, SamVariant::VitB)),
            (true, _) => Some((tiny, SamVariant::MobileSam)),
            (false, true) => Some((vit, SamVariant::VitB)),
            (false, false) => None,
        }
    }

    pub fn load(dir: &Path, device: &Device) -> Result<SamModel> {
        let Some((file, variant)) = Self::find_weights(dir) else {
            bail!(
                "SAM weights not found in {}. Put {TINY_FILE} (MobileSAM) or {VIT_B_FILE} (SAM ViT-B) from lmz/candle-sam there, \
                 or point ${DIR_ENV} at the folder that has them (default {DEFAULT_DIR}).",
                dir.display()
            );
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&file], DType::F32, device)? };
        Self::from_var_builder(variant, vb, device).with_context(|| format!("loading {}", file.display()))
    }

    pub fn from_var_builder(variant: SamVariant, vb: VarBuilder, device: &Device) -> Result<SamModel> {
        let enc = vb.pp("image_encoder");
        let encoder = match variant {
            SamVariant::MobileSam => Encoder::Tiny(Box::new(tiny_vit_5m(enc)?)),
            SamVariant::VitB => Encoder::Vit(Box::new(ImageEncoderViT::new(IMAGE_SIZE, 16, 3, 768, 12, 12, PROMPT_DIM, true, true, true, 14, &[2, 5, 8, 11], enc)?)),
            SamVariant::TestVit { dim, depth, heads } => {
                Encoder::Vit(Box::new(ImageEncoderViT::new(IMAGE_SIZE, 16, 3, dim, depth, heads, PROMPT_DIM, true, true, true, 14, &[depth.saturating_sub(1)], enc)?))
            }
        };
        let prompt = PromptEncoder::new(PROMPT_DIM, (EMBED_SIDE, EMBED_SIDE), (IMAGE_SIZE, IMAGE_SIZE), 16, vb.pp("prompt_encoder"))?;
        let decoder = MaskDecoder::new(PROMPT_DIM, 3, 3, 256, vb.pp("mask_decoder"))?;
        let image_pe = prompt.get_dense_pe()?;
        let mean = Tensor::new(&[123.675f32, 116.28, 103.53], device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&[58.395f32, 57.12, 57.375], device)?.reshape((3, 1, 1))?;
        Ok(SamModel { variant, encoder, prompt, decoder, image_pe, mean, std, device: device.clone() })
    }

    pub fn variant(&self) -> SamVariant {
        self.variant
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn scale(width: u32, height: u32) -> f32 {
        IMAGE_SIZE as f32 / width.max(height).max(1) as f32
    }

    fn input(&self, rgb: &Rgb) -> Result<Tensor> {
        let s = Self::scale(rgb.width, rgb.height);
        let (w, h) = (((rgb.width as f32 * s).round() as u32).clamp(1, IMAGE_SIZE as u32), ((rgb.height as f32 * s).round() as u32).clamp(1, IMAGE_SIZE as u32));
        let src = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.data.clone()).context("RGB buffer does not match its size")?;
        let resized = image::imageops::resize(&src, w, h, image::imageops::FilterType::Triangle);
        let plane = (w * h) as usize;
        let mut chw = vec![0f32; 3 * plane];
        for (i, px) in resized.as_raw().chunks_exact(3).enumerate() {
            for c in 0..3 {
                chw[c * plane + i] = px[c] as f32;
            }
        }
        let t = Tensor::from_vec(chw, (3, h as usize, w as usize), &self.device)?.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?;
        Ok(t.pad_with_zeros(1, 0, IMAGE_SIZE - h as usize)?.pad_with_zeros(2, 0, IMAGE_SIZE - w as usize)?)
    }

    pub fn encoder_rows(&self) -> usize {
        std::env::var(BATCH_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or_else(|| noto_ml::BatchPolicy::for_device(&self.device).image_rows(EMBED_SIDE * EMBED_SIDE))
    }

    pub fn encode_batch(&self, frames: &[&Rgb]) -> Result<Vec<Tensor>> {
        let mut out = Vec::with_capacity(frames.len());
        for chunk in frames.chunks(self.encoder_rows()) {
            let inputs = chunk.iter().map(|f| self.input(f)).collect::<Result<Vec<_>>>()?;
            let emb = self.encoder.forward(&Tensor::stack(&inputs, 0)?)?;
            for k in 0..chunk.len() {
                out.push(emb.narrow(0, k, 1)?);
            }
        }
        noto_ml::trim_idle_pool(&self.device);
        Ok(out)
    }

    pub fn decode(&self, embedding: &Tensor, width: u32, height: u32, points: &[(f32, f32)]) -> Result<Vec<Decoded>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let s = Self::scale(width, height);
        let n = points.len();
        let xy: Vec<f32> = points.iter().flat_map(|(x, y)| [x * s, y * s]).collect();
        let coords = Tensor::from_vec(xy, (n, 1, 2), &self.device)?;
        let labels = Tensor::ones((n, 1), DType::F32, &self.device)?;
        let (sparse, dense) = self.prompt.forward(Some((&coords, &labels)), None, None)?;
        let (masks, iou) = self.decoder.forward(embedding, &self.image_pe, &sparse, &dense, true)?;
        let iou: Vec<Vec<f32>> = iou.to_vec2()?;
        let k = masks.dim(1)?;
        let best: Vec<u32> = iou.iter().enumerate().map(|(i, row)| (i * k + row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |b| b.0)) as u32).collect();
        let idx = Tensor::new(best.as_slice(), &self.device)?;
        let chosen: Vec<Vec<f32>> = masks.contiguous()?.reshape((n * k, LOW_RES * LOW_RES))?.index_select(&idx, 0)?.to_vec2()?;
        Ok(chosen
            .into_iter()
            .zip(&iou)
            .map(|(logits, row)| Decoded { logits, iou_pred: row.iter().copied().fold(f32::MIN, f32::max) })
            .collect())
    }
}

pub fn sample_mask(logits: &[f32], frame_w: u32, frame_h: u32, k: u32) -> Mask {
    let k = k.max(1);
    sample_scaled(logits, frame_w, frame_h, (frame_w / k).max(1), (frame_h / k).max(1), k as f32, k as f32)
}

/// The decoder's low-res logits as a `mask_w` × `mask_h` mask of the frame (the pyramid's L1 grid times
/// `mask_per_cell`, so every SigLIP2 patch is a whole block of mask pixels).
pub fn sample_mask_to(logits: &[f32], frame_w: u32, frame_h: u32, mask_w: u32, mask_h: u32) -> Mask {
    let (w, h) = (mask_w.max(1), mask_h.max(1));
    sample_scaled(logits, frame_w, frame_h, w, h, frame_w.max(1) as f32 / w as f32, frame_h.max(1) as f32 / h as f32)
}

fn sample_scaled(logits: &[f32], frame_w: u32, frame_h: u32, w: u32, h: u32, kx: f32, ky: f32) -> Mask {
    let s = SamModel::scale(frame_w, frame_h);
    let r = LOW_RES as f32 / IMAGE_SIZE as f32;
    let at = |x: i64, y: i64| logits[(y.clamp(0, LOW_RES as i64 - 1) as usize) * LOW_RES + x.clamp(0, LOW_RES as i64 - 1) as usize];
    let mut m = Mask::empty(w, h);
    for y in 0..h {
        for x in 0..w {
            let lx = ((x as f32 + 0.5) * kx) * s * r - 0.5;
            let ly = ((y as f32 + 0.5) * ky) * s * r - 0.5;
            let (x0, y0) = (lx.floor() as i64, ly.floor() as i64);
            let (fx, fy) = (lx - x0 as f32, ly - y0 as f32);
            let v = at(x0, y0) * (1.0 - fx) * (1.0 - fy) + at(x0 + 1, y0) * fx * (1.0 - fy) + at(x0, y0 + 1) * (1.0 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
            if v > 0.0 {
                m.set(x, y);
            }
        }
    }
    m
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamCfg {
    pub downscale: u32,
    pub min_iou_pred: f32,
    pub min_area_frac: f32,
    pub max_area_frac: f32,
    pub max_tracks: usize,
    pub max_misses: u32,
    pub cache_frames: usize,
    /// The mask grid: the pyramid's (L1 × `mask_per_cell`) while set, else the frame over `downscale`.
    pub pyramid: Option<crate::pyramid::PyramidCfg>,
}

impl Default for SamCfg {
    fn default() -> Self {
        SamCfg { downscale: 2, min_iou_pred: 0.5, min_area_frac: 0.0004, max_area_frac: 0.25, max_tracks: 8, max_misses: 2, cache_frames: 16, pyramid: None }
    }
}

impl SamCfg {
    /// The mask size for a `width` × `height` frame.
    pub fn mask_size(&self, width: u32, height: u32) -> (u32, u32) {
        match &self.pyramid {
            Some(p) => crate::pyramid::Pyramid::for_frame(width, height, p).mask(),
            None => {
                let k = self.downscale.max(1);
                ((width / k).max(1), (height / k).max(1))
            }
        }
    }
}

/// A followed object; its mask and centre are in mask pixels, `sx` × `sy` frame pixels each.
struct Live {
    id: u32,
    label: String,
    mask: Mask,
    cx: f32,
    cy: f32,
    sx: f32,
    sy: f32,
    age: u32,
    misses: u32,
}

enum Query {
    Follow { track: usize, prompted: bool },
    New { prompt: usize },
}

pub struct SamTracker {
    model: Arc<SamModel>,
    pub cfg: SamCfg,
    tracks: Vec<Live>,
    next_id: u32,
    cache: HashMap<u64, Tensor>,
    order: VecDeque<u64>,
    encoded: usize,
}

fn image_key(rgb: &Rgb) -> u64 {
    let h = noto_ml::parallel::fnv1a(noto_ml::parallel::FNV_OFFSET, &rgb.width.to_le_bytes());
    let h = noto_ml::parallel::fnv1a(h, &rgb.height.to_le_bytes());
    noto_ml::parallel::fnv1a(h, &rgb.data)
}

impl SamTracker {
    pub fn new(model: Arc<SamModel>) -> SamTracker {
        SamTracker { model, cfg: SamCfg::default(), tracks: Vec::new(), next_id: 1, cache: HashMap::new(), order: VecDeque::new(), encoded: 0 }
    }

    fn remember(&mut self, key: u64, emb: Tensor) {
        if self.cache.insert(key, emb).is_none() {
            self.order.push_back(key);
        }
        while self.order.len() > self.cfg.cache_frames.max(1) {
            if let Some(old) = self.order.pop_front() {
                self.cache.remove(&old);
            }
        }
    }

    fn embedding(&mut self, rgb: &Rgb) -> Result<Tensor> {
        let key = image_key(rgb);
        if let Some(e) = self.cache.get(&key) {
            return Ok(e.clone());
        }
        let e = self.model.encode_batch(&[rgb])?.pop().expect("one embedding per frame");
        self.encoded += 1;
        self.remember(key, e.clone());
        Ok(e)
    }

    pub fn cached_frames(&self) -> usize {
        self.cache.len()
    }

    /// How many frames went through the image encoder.
    pub fn encoded_frames(&self) -> usize {
        self.encoded
    }

    /// Whether `update` looks at the image at all: not when there is nothing to prompt and nothing to follow, so a
    /// search frame without a gated prompt costs no encoder pass.
    pub fn wants_frame(&self, prompts: usize) -> bool {
        prompts > 0 || !self.tracks.is_empty()
    }

    fn accept(&self, m: &Mask, iou_pred: f32) -> bool {
        let area = m.area();
        let frac = area as f32 / (m.width * m.height).max(1) as f32;
        area > 0 && iou_pred >= self.cfg.min_iou_pred && frac >= self.cfg.min_area_frac && frac <= self.cfg.max_area_frac
    }

    fn out(&self, t: &Live, iou_prev: Option<f32>, propagated: bool, score: f32) -> TrackedMask {
        let (kx, ky) = (t.sx, t.sy);
        let b = t.mask.bbox().unwrap_or([0, 0, 0, 0]);
        TrackedMask {
            id: t.id,
            label: t.label.clone(),
            mask: t.mask.clone(),
            cx: t.cx * kx,
            cy: t.cy * ky,
            bbox: [b[0] as f32 * kx, b[1] as f32 * ky, (b[2] + 1) as f32 * kx, (b[3] + 1) as f32 * ky],
            area_frac: t.mask.area() as f32 / (t.mask.width * t.mask.height).max(1) as f32,
            score,
            iou_prev,
            age: t.age,
            propagated,
        }
    }
}

impl MaskTracker for SamTracker {
    fn mode(&self) -> &str {
        self.model.variant.mode()
    }

    fn device(&self) -> String {
        noto_ml::device_label(&self.model.device)
    }

    fn reset(&mut self) {
        self.tracks.clear();
    }

    fn prepare(&mut self, frames: &[&Rgb]) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        let missing: Vec<(u64, &Rgb)> = frames.iter().map(|f| (image_key(f), *f)).filter(|(k, _)| !self.cache.contains_key(k) && seen.insert(*k)).collect();
        let missing = &missing[..missing.len().min(self.cfg.cache_frames.max(1))];
        if missing.is_empty() {
            return Ok(());
        }
        let refs: Vec<&Rgb> = missing.iter().map(|(_, f)| *f).collect();
        let embs = self.model.encode_batch(&refs)?;
        self.encoded += refs.len();
        for ((key, _), e) in missing.iter().zip(embs) {
            self.remember(*key, e);
        }
        Ok(())
    }

    fn set_pyramid(&mut self, cfg: Option<&crate::pyramid::PyramidCfg>) {
        self.cfg.pyramid = cfg.copied();
    }

    fn update(&mut self, rgb: &Rgb, _depth: Option<&DepthMap>, prompts: &[PointPrompt]) -> Result<Vec<TrackedMask>> {
        if !self.wants_frame(prompts.len()) {
            return Ok(Vec::new());
        }
        let emb = self.embedding(rgb)?;
        let (mw, mh) = self.cfg.mask_size(rgb.width, rgb.height);
        let (kx, ky) = match self.cfg.pyramid {
            Some(_) => (rgb.width.max(1) as f32 / mw as f32, rgb.height.max(1) as f32 / mh as f32),
            None => (self.cfg.downscale.max(1) as f32, self.cfg.downscale.max(1) as f32),
        };
        let mut used = vec![false; prompts.len()];
        let mut queries: Vec<(Query, (f32, f32))> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            let near = |p: &PointPrompt| -> bool {
                let (x, y) = (p.px / t.sx, p.py / t.sy);
                if (x as u32) < t.mask.width && (y as u32) < t.mask.height && t.mask.get(x as u32, y as u32) {
                    return true;
                }
                let Some(b) = t.mask.bbox() else { return false };
                let (gx, gy) = ((0.5 * (b[2] - b[0] + 1) as f32).max(FOLLOW_GUARD_PX / t.sx), (0.5 * (b[3] - b[1] + 1) as f32).max(FOLLOW_GUARD_PX / t.sy));
                x >= b[0] as f32 - gx && x <= b[2] as f32 + gx && y >= b[1] as f32 - gy && y <= b[3] as f32 + gy
            };
            match prompts.iter().enumerate().find(|(i, p)| !used[*i] && p.label == t.label && near(p)) {
                Some((i, p)) => {
                    used[i] = true;
                    queries.push((Query::Follow { track: ti, prompted: true }, (p.px, p.py)));
                }
                None if t.misses < self.cfg.max_misses => queries.push((Query::Follow { track: ti, prompted: false }, (t.cx * t.sx, t.cy * t.sy))),
                None => {}
            }
        }
        let mut room = self.cfg.max_tracks.saturating_sub(self.tracks.len());
        for (i, p) in prompts.iter().enumerate() {
            if used[i] || room == 0 {
                continue;
            }
            let on_track = self.tracks.iter().any(|t| {
                let (x, y) = ((p.px / t.sx) as u32, (p.py / t.sy) as u32);
                x < t.mask.width && y < t.mask.height && t.mask.get(x, y)
            });
            if !on_track {
                used[i] = true;
                room -= 1;
                queries.push((Query::New { prompt: i }, (p.px, p.py)));
            }
        }
        let points: Vec<(f32, f32)> = queries.iter().map(|(_, p)| *p).collect();
        let decoded = self.model.decode(&emb, rgb.width, rgb.height, &points)?;

        let old = std::mem::take(&mut self.tracks);
        let mut kept: Vec<Option<Live>> = old.into_iter().map(Some).collect();
        let mut out = Vec::new();
        let mut next = Vec::new();
        for ((q, _), d) in queries.into_iter().zip(decoded) {
            let mask = match self.cfg.pyramid {
                Some(_) => sample_mask_to(&d.logits, rgb.width, rgb.height, mw, mh),
                None => sample_mask(&d.logits, rgb.width, rgb.height, self.cfg.downscale.max(1)),
            };
            if !self.accept(&mask, d.iou_pred) {
                continue;
            }
            let (cx, cy) = mask.centroid().expect("accepted masks are not empty");
            match q {
                Query::Follow { track, prompted } => {
                    let Some(mut t) = kept[track].take() else { continue };
                    let iou_prev = t.mask.iou(&mask);
                    t.misses = if prompted { 0 } else { t.misses + 1 };
                    t.mask = mask;
                    t.cx = cx;
                    t.cy = cy;
                    (t.sx, t.sy) = (kx, ky);
                    t.age += 1;
                    let score = if prompted { 0.4 + 0.6 * d.iou_pred.clamp(0.0, 1.0) } else { 0.6 * d.iou_pred.clamp(0.0, 1.0) };
                    out.push(self.out(&t, Some(iou_prev), !prompted, score));
                    next.push(t);
                }
                Query::New { prompt } => {
                    let t = Live { id: self.next_id, label: prompts[prompt].label.clone(), mask, cx, cy, sx: kx, sy: ky, age: 1, misses: 0 };
                    self.next_id += 1;
                    let score = 0.4 + 0.6 * (d.iou_pred.clamp(0.0, 1.0) * prompts[prompt].score.clamp(0.0, 1.0)).sqrt();
                    out.push(self.out(&t, None, false, score));
                    next.push(t);
                }
            }
        }
        self.tracks = next;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn frame(w: u32, h: u32, shift: u8) -> Rgb {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let inside = (w / 3..w / 2).contains(&x) && (h / 3..2 * h / 3).contains(&y);
                data.extend_from_slice(&if inside { [200, 60 + shift, 40] } else { [40, 120, (x % 50) as u8] });
            }
        }
        Rgb::new(w, h, data)
    }

    #[test]
    fn mask_sampling_follows_the_resize_and_the_padding() {
        let s = SamModel::scale(384, 256) * LOW_RES as f32 / IMAGE_SIZE as f32;
        let mut logits = vec![-5.0f32; LOW_RES * LOW_RES];
        for y in 0..LOW_RES {
            for x in 0..LOW_RES {
                let (fx, fy) = ((x as f32 + 0.5) / s, (y as f32 + 0.5) / s);
                if (96.0..192.0).contains(&fx) && (64.0..128.0).contains(&fy) {
                    logits[y * LOW_RES + x] = 5.0;
                }
            }
        }
        let m = sample_mask(&logits, 384, 256, 2);
        assert_eq!((m.width, m.height), (192, 128));
        let b = m.bbox().unwrap();
        assert!((b[0] as i32 - 48).abs() <= 1 && (b[2] as i32 - 95).abs() <= 1, "{b:?}");
        assert!((b[1] as i32 - 32).abs() <= 1 && (b[3] as i32 - 63).abs() <= 1, "{b:?}");
        assert!(!m.get(0, 127));
    }

    fn test_model() -> SamModel {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        SamModel::from_var_builder(SamVariant::TestVit { dim: 32, depth: 1, heads: 2 }, vb, &dev).unwrap()
    }

    #[test]
    fn batched_encoding_matches_one_by_one_and_one_decoder_call_serves_every_query() {
        let m = test_model();
        let (a, b) = (frame(96, 64, 0), frame(96, 64, 30));
        let both = m.encode_batch(&[&a, &b]).unwrap();
        let one = m.encode_batch(&[&b]).unwrap();
        assert_eq!(both[0].dims(), &[1, PROMPT_DIM, EMBED_SIDE, EMBED_SIDE]);
        let diff = (&both[1] - &one[0]).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(diff < 1e-4, "batch rows do not mix: {diff}");
        let d = m.decode(&both[0], 96, 64, &[(40.0, 30.0), (10.0, 10.0), (80.0, 50.0)]).unwrap();
        assert_eq!(d.len(), 3);
        assert!(d.iter().all(|q| q.logits.len() == LOW_RES * LOW_RES && q.iou_pred.is_finite()));
        assert!(m.decode(&both[0], 96, 64, &[]).unwrap().is_empty());
    }

    #[test]
    fn a_pyramid_mask_is_a_whole_number_of_blocks_per_siglip_patch() {
        let s = SamModel::scale(384, 256) * LOW_RES as f32 / IMAGE_SIZE as f32;
        let mut logits = vec![-5.0f32; LOW_RES * LOW_RES];
        for y in 0..LOW_RES {
            for x in 0..LOW_RES {
                let (fx, fy) = ((x as f32 + 0.5) / s, (y as f32 + 0.5) / s);
                if (96.0..192.0).contains(&fx) && (64.0..128.0).contains(&fy) {
                    logits[y * LOW_RES + x] = 5.0;
                }
            }
        }
        let cfg = SamCfg { pyramid: Some(crate::pyramid::PyramidCfg::default()), ..SamCfg::default() };
        assert_eq!(cfg.mask_size(384, 256), (288, 192), "18 x 12 SigLIP2 patches of 16 x 16 mask pixels");
        assert_eq!(cfg.mask_size(640, 480), (256, 192), "SAM's 256 x 192 low-res logits of a 4:3 frame");
        assert_eq!(SamCfg::default().mask_size(384, 256), (192, 128));
        let m = sample_mask_to(&logits, 384, 256, 288, 192);
        let b = m.bbox().unwrap();
        let k = 288.0 / 384.0;
        assert!((b[0] as f32 - 96.0 * k).abs() <= 1.0 && (b[2] as f32 - (192.0 * k - 1.0)).abs() <= 1.0, "{b:?}");
        assert!((b[1] as f32 - 64.0 * k).abs() <= 1.0 && (b[3] as f32 - (128.0 * k - 1.0)).abs() <= 1.0, "{b:?}");
        assert_eq!(sample_mask_to(&logits, 384, 256, 192, 128), sample_mask(&logits, 384, 256, 2), "the old grid is one case of the new");
    }

    #[test]
    fn nothing_to_prompt_and_nothing_to_follow_skips_the_encoder() {
        let mut t = SamTracker::new(Arc::new(test_model()));
        t.cfg = SamCfg { min_iou_pred: f32::MIN, min_area_frac: 0.0, max_area_frac: 1.0, pyramid: Some(crate::pyramid::PyramidCfg::default()), ..SamCfg::default() };
        let (a, b) = (frame(96, 64, 0), frame(96, 64, 30));
        assert!(!t.wants_frame(0));
        assert!(t.update(&a, None, &[]).unwrap().is_empty());
        assert_eq!(t.encoded_frames(), 0, "an empty search frame costs no encoder pass");
        let p = PointPrompt { label: "iron_ore_node".into(), px: 40.0, py: 30.0, score: 0.9 };
        let first = t.update(&a, None, &[p]).unwrap();
        assert_eq!(t.encoded_frames(), 1);
        if let Some(f) = first.first() {
            assert_eq!((f.mask.width, f.mask.height), t.cfg.mask_size(96, 64));
            assert!(f.cx > 0.0 && f.cx < 96.0 && f.bbox[2] <= 96.0 + 1e-3, "track centres come back in frame pixels");
            assert!(t.wants_frame(0), "a live track is followed without a prompt");
            t.update(&b, None, &[]).unwrap();
            assert_eq!(t.encoded_frames(), 2);
        }
    }

    #[test]
    fn the_tracker_encodes_each_keyframe_once() {
        let mut t = SamTracker::new(Arc::new(test_model()));
        t.cfg = SamCfg { min_iou_pred: f32::MIN, min_area_frac: 0.0, max_area_frac: 1.0, ..SamCfg::default() };
        let (a, b) = (frame(96, 64, 0), frame(96, 64, 30));
        t.prepare(&[&a, &b, &a]).unwrap();
        assert_eq!(t.cached_frames(), 2);
        let p = PointPrompt { label: "iron_ore_node".into(), px: 40.0, py: 30.0, score: 0.9 };
        let first = t.update(&a, None, &[p.clone()]).unwrap();
        let second = t.update(&b, None, &[]).unwrap();
        assert_eq!(t.cached_frames(), 2, "update used the prepared embeddings");
        assert!(first.len() <= 1 && second.len() <= first.len());
        if let (Some(f), Some(s)) = (first.first(), second.first()) {
            assert_eq!(f.id, s.id, "a followed track keeps its id");
            assert!(s.propagated && s.iou_prev.is_some());
        }
        assert_eq!(t.mode(), "sam-test");
    }
}
