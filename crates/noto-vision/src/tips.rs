use std::path::Path;

use anyhow::{bail, Context};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{
    conv2d, conv2d_no_bias, conv_transpose2d, layer_norm, linear, Conv2d, Conv2dConfig, ConvTranspose2d,
    ConvTranspose2dConfig, LayerNorm, Linear, VarBuilder,
};

use noto_ml::vision::antialias_bilinear_weights;

use crate::depth::{clamp_metric, DepthModel};
use crate::frame::{DepthKind, DepthMap, Rgb};

pub const MODE: &str = "tipsv2-b14-dpt";
pub const DEFAULT_DIR: &str = "models/tipsv2-b14-dpt";
pub const DIR_ENV: &str = "TIPS_DIR";
pub const WEIGHTS_FILE: &str = "model.safetensors";
pub const CONFIG_FILE: &str = "config.json";

pub const DEFAULT_SHORT_SIDE_PATCHES: usize = 24;

const DPT_LEVELS: usize = 4;
const FUSION_UPSAMPLE: usize = 2;
const DOWN_KERNEL: usize = 3;
const SAME_PAD_3X3: usize = 1;

#[derive(Clone, Copy, Debug)]
enum LevelScale {
    Up(usize),
    Keep,
    Down(usize),
}

const LEVEL_SCALES: [LevelScale; DPT_LEVELS] = [LevelScale::Up(4), LevelScale::Up(2), LevelScale::Keep, LevelScale::Down(2)];

#[derive(Clone, Debug, PartialEq)]
pub struct TipsConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_hidden: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub register_tokens: usize,
    pub layer_norm_eps: f64,
    pub tap_blocks: [usize; DPT_LEVELS],
    pub post_process_channels: [usize; DPT_LEVELS],
    pub dpt_channels: usize,
    pub depth_bins: usize,
    pub min_depth: f32,
    pub max_depth: f32,
    pub readout_gelu_tanh: bool,
}

impl Default for TipsConfig {
    fn default() -> Self {
        TipsConfig {
            hidden: 768,
            layers: 12,
            heads: 12,
            mlp_hidden: 3072,
            patch_size: 14,
            image_size: 448,
            register_tokens: 1,
            layer_norm_eps: 1e-6,
            tap_blocks: [2, 5, 8, 11],
            post_process_channels: [96, 192, 384, 768],
            dpt_channels: 256,
            depth_bins: 256,
            min_depth: 1e-3,
            max_depth: 10.0,
            readout_gelu_tanh: true,
        }
    }
}

impl TipsConfig {
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn from_json(raw: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        if let Some(f) = v["vision_fn"].as_str() {
            if f != "vit_base" {
                bail!("only the ViT-B/14 backbone (vision_fn \"vit_base\") is ported, the config asks for {f:?}");
            }
        }
        if let Some(f) = v["ffn_layer"].as_str() {
            if f != "mlp" {
                bail!("only the GELU MLP feed-forward is ported, the config asks for {f:?}");
            }
        }
        let d = TipsConfig::default();
        let us = |keys: &[&str], fallback: usize| keys.iter().find_map(|k| v[*k].as_u64()).map_or(fallback, |x| x as usize);
        let fl = |keys: &[&str], fallback: f64| keys.iter().find_map(|k| v[*k].as_f64()).unwrap_or(fallback);
        let levels = |key: &str| -> anyhow::Result<Option<[usize; DPT_LEVELS]>> {
            let Some(a) = v[key].as_array() else { return Ok(None) };
            let n: Vec<usize> = a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect();
            match <[usize; DPT_LEVELS]>::try_from(n) {
                Ok(n) if n.len() == a.len() => Ok(Some(n)),
                _ => bail!("{key} must hold {DPT_LEVELS} non-negative integers"),
            }
        };
        let hidden = us(&["embed_dim", "hidden_size"], d.hidden);
        let mlp_hidden = match v["mlp_ratio"].as_f64() {
            Some(r) => (hidden as f64 * r) as usize,
            None => us(&["intermediate_size"], d.mlp_hidden),
        };
        let tap_blocks = match levels("out_indices")? {
            Some(stages) if stages.contains(&0) => bail!("out_indices counts stages from the patch embedding (0); DPT needs block outputs (1 and up)"),
            Some(stages) => stages.map(|s| s - 1),
            None => d.tap_blocks,
        };
        Ok(TipsConfig {
            hidden,
            layers: us(&["num_layers", "num_hidden_layers"], d.layers),
            heads: us(&["num_heads", "num_attention_heads"], d.heads),
            mlp_hidden,
            patch_size: us(&["patch_size"], d.patch_size),
            image_size: us(&["img_size", "image_size"], d.image_size),
            register_tokens: us(&["num_register_tokens"], d.register_tokens),
            layer_norm_eps: fl(&["layer_norm_eps"], d.layer_norm_eps),
            tap_blocks,
            post_process_channels: levels("post_process_channels")?.unwrap_or(d.post_process_channels),
            dpt_channels: us(&["channels", "dpt_channels"], d.dpt_channels),
            depth_bins: us(&["num_depth_bins"], d.depth_bins),
            min_depth: fl(&["min_depth"], d.min_depth as f64) as f32,
            max_depth: fl(&["max_depth"], d.max_depth as f64) as f32,
            readout_gelu_tanh: match ["readout_act", "readout_activation"].iter().find_map(|k| v[*k].as_str()) {
                Some(a) => a.contains("tanh"),
                None => d.readout_gelu_tanh,
            },
        })
    }

    pub fn position_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.heads == 0 || !self.hidden.is_multiple_of(self.heads) {
            bail!("hidden size {} does not split into {} heads", self.hidden, self.heads);
        }
        if self.patch_size == 0 || !self.image_size.is_multiple_of(self.patch_size) {
            bail!("training size {} is not a whole number of {} px patches", self.image_size, self.patch_size);
        }
        if !self.tap_blocks.windows(2).all(|w| w[0] < w[1]) || self.tap_blocks[DPT_LEVELS - 1] >= self.layers {
            bail!("tap blocks {:?} must increase and stay below {} layers", self.tap_blocks, self.layers);
        }
        if self.depth_bins < 2 || !(self.min_depth > 0.0 && self.min_depth < self.max_depth) {
            bail!("depth bins need at least two centres with 0 < min < max, got {} in {}..{}", self.depth_bins, self.min_depth, self.max_depth);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputGrid {
    ShortSide(usize),
    Fixed { cols: usize, rows: usize },
}

impl Default for InputGrid {
    fn default() -> Self {
        InputGrid::ShortSide(DEFAULT_SHORT_SIDE_PATCHES)
    }
}

impl InputGrid {
    pub fn grid_for(self, width: u32, height: u32) -> (usize, usize) {
        match self {
            InputGrid::Fixed { cols, rows } => (cols.max(1), rows.max(1)),
            InputGrid::ShortSide(n) => {
                let n = n.max(1) as f64;
                let (w, h) = (width.max(1) as f64, height.max(1) as f64);
                let long = |a: f64, b: f64| ((n * a / b).round() as usize).max(1);
                if width <= height {
                    (n as usize, long(h, w))
                } else {
                    (long(w, h), n as usize)
                }
            }
        }
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &TipsConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let d = cfg.hidden;
        Ok(Attention { qkv: linear(d, 3 * d, vb.pp("qkv"))?, proj: linear(d, d, vb.pp("proj"))?, heads: cfg.heads, head_dim: d / cfg.heads })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let qkv = self.qkv.forward(x)?;
        let split = |i: usize| -> candle_core::Result<Tensor> {
            qkv.narrow(D::Minus1, i * d, d)?.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * (self.head_dim as f64).powf(-0.5))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b, n, d))?;
        self.proj.forward(&ctx)
    }
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ls2: Tensor,
}

impl Block {
    fn new(cfg: &TipsConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let (d, eps) = (cfg.hidden, cfg.layer_norm_eps);
        let mlp = vb.pp("mlp");
        Ok(Block {
            norm1: layer_norm(d, eps, vb.pp("norm1"))?,
            attn: Attention::new(cfg, vb.pp("attn"))?,
            ls1: vb.pp("ls1").get(d, "gamma")?,
            norm2: layer_norm(d, eps, vb.pp("norm2"))?,
            fc1: linear(d, cfg.mlp_hidden, mlp.pp("fc1"))?,
            fc2: linear(cfg.mlp_hidden, d, mlp.pp("fc2"))?,
            ls2: vb.pp("ls2").get(d, "gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (x + self.attn.forward(&self.norm1.forward(x)?)?.broadcast_mul(&self.ls1)?)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        &x + mlp.broadcast_mul(&self.ls2)?
    }
}

struct Tap {
    cls: Tensor,
    patches: Tensor,
    rows: usize,
    cols: usize,
}

struct Encoder {
    patch_embed: Conv2d,
    cls_token: Tensor,
    register_tokens: Option<Tensor>,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    side: usize,
    hidden: usize,
    patch_size: usize,
    tap_blocks: [usize; DPT_LEVELS],
}

impl Encoder {
    fn new(cfg: &TipsConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let (d, side) = (cfg.hidden, cfg.position_side());
        let patch_cfg = Conv2dConfig { stride: cfg.patch_size, ..Default::default() };
        let register_tokens = match cfg.register_tokens {
            0 => None,
            r => Some(vb.get((1, r, d), "register_tokens")?),
        };
        Ok(Encoder {
            patch_embed: conv2d(3, d, cfg.patch_size, patch_cfg, vb.pp("patch_embed").pp("proj"))?,
            cls_token: vb.get((1, 1, d), "cls_token")?,
            register_tokens,
            pos_embed: vb.get((1, 1 + side * side, d), "pos_embed")?,
            blocks: (0..cfg.layers).map(|i| Block::new(cfg, vb.pp(format!("blocks.{i}")))).collect::<candle_core::Result<_>>()?,
            norm: layer_norm(d, cfg.layer_norm_eps, vb.pp("norm"))?,
            side,
            hidden: d,
            patch_size: cfg.patch_size,
            tap_blocks: cfg.tap_blocks,
        })
    }

    fn positions(&self, rows: usize, cols: usize) -> candle_core::Result<Tensor> {
        let (side, d) = (self.side, self.hidden);
        if (rows, cols) == (side, side) {
            return Ok(self.pos_embed.clone());
        }
        let dev = self.pos_embed.device();
        let wy = Tensor::from_vec(antialias_bilinear_weights(side, rows), (rows, side), dev)?;
        let wx = Tensor::from_vec(antialias_bilinear_weights(side, cols), (cols, side), dev)?;
        let table = self.pos_embed.narrow(1, 1, side * side)?.reshape((side, side * d))?;
        let by_rows = wy.matmul(&table)?.reshape((rows, side, d))?;
        let grid = wx.broadcast_matmul(&by_rows)?.reshape((1, rows * cols, d))?;
        Tensor::cat(&[&self.pos_embed.narrow(1, 0, 1)?, &grid], 1)
    }

    fn taps(&self, pixels: &Tensor) -> candle_core::Result<Vec<Tap>> {
        let (b, _, h, w) = pixels.dims4()?;
        let (rows, cols, d) = (h / self.patch_size, w / self.patch_size, self.hidden);
        let n = rows * cols;
        let patches = self.patch_embed.forward(pixels)?.flatten_from(2)?.transpose(1, 2)?;
        let cls = self.cls_token.broadcast_as((b, 1, d))?;
        let mut x = Tensor::cat(&[&cls, &patches], 1)?.broadcast_add(&self.positions(rows, cols)?)?;
        let registers = match &self.register_tokens {
            Some(r) => {
                let r = r.broadcast_as((b, r.dim(1)?, d))?;
                x = Tensor::cat(&[&x.narrow(1, 0, 1)?, &r, &x.narrow(1, 1, n)?], 1)?;
                r.dim(1)?
            }
            None => 0,
        };
        let last = self.tap_blocks[DPT_LEVELS - 1];
        let mut taps = Vec::with_capacity(DPT_LEVELS);
        for (i, block) in self.blocks.iter().enumerate().take(last + 1) {
            x = block.forward(&x)?;
            if self.tap_blocks.contains(&i) {
                let normed = self.norm.forward(&x)?;
                taps.push(Tap {
                    cls: normed.narrow(1, 0, 1)?.squeeze(1)?,
                    patches: normed.narrow(1, 1 + registers, n)?,
                    rows,
                    cols,
                });
            }
        }
        Ok(taps)
    }
}

struct ReassembleLevel {
    readout: Linear,
    out_projection: Conv2d,
    resize: Resize,
    gelu_tanh: bool,
}

enum Resize {
    Up(ConvTranspose2d),
    Keep,
    Down(Conv2d),
}

impl ReassembleLevel {
    fn new(level: usize, cfg: &TipsConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let (d, ch) = (cfg.hidden, cfg.post_process_channels[level]);
        let resize_vb = vb.pp("resize_layers").pp(level);
        let resize = match LEVEL_SCALES[level] {
            LevelScale::Up(s) => Resize::Up(conv_transpose2d(ch, ch, s, ConvTranspose2dConfig { stride: s, ..Default::default() }, resize_vb)?),
            LevelScale::Keep => Resize::Keep,
            LevelScale::Down(s) => {
                let down = Conv2dConfig { stride: s, padding: SAME_PAD_3X3, ..Default::default() };
                Resize::Down(conv2d(ch, ch, DOWN_KERNEL, down, resize_vb)?)
            }
        };
        Ok(ReassembleLevel {
            readout: linear(2 * d, d, vb.pp("readout_projects").pp(level))?,
            out_projection: conv2d(d, ch, 1, Conv2dConfig::default(), vb.pp("out_projections").pp(level))?,
            resize,
            gelu_tanh: cfg.readout_gelu_tanh,
        })
    }

    fn forward(&self, tap: &Tap) -> candle_core::Result<Tensor> {
        let (b, n, d) = tap.patches.dims3()?;
        let readout = tap.cls.unsqueeze(1)?.broadcast_as((b, n, d))?;
        let x = self.readout.forward(&Tensor::cat(&[&tap.patches, &readout], 2)?.contiguous()?)?;
        let x = if self.gelu_tanh { x.gelu()? } else { x.gelu_erf()? };
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b, d, tap.rows, tap.cols))?;
        let x = self.out_projection.forward(&x)?;
        match &self.resize {
            Resize::Up(up) => up.forward(&x),
            Resize::Keep => Ok(x),
            Resize::Down(down) => down.forward(&x),
        }
    }
}

struct ResidualUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualUnit {
    fn new(ch: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let same = Conv2dConfig { padding: SAME_PAD_3X3, ..Default::default() };
        Ok(ResidualUnit { conv1: conv2d_no_bias(ch, ch, 3, same, vb.pp("conv1"))?, conv2: conv2d_no_bias(ch, ch, 3, same, vb.pp("conv2"))? })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.conv2.forward(&self.conv1.forward(&x.relu()?)?.relu()?)?;
        x + y
    }
}

struct FusionBlock {
    residual_unit: Option<ResidualUnit>,
    main_unit: ResidualUnit,
    out_conv: Conv2d,
}

impl FusionBlock {
    fn new(ch: usize, has_skip: bool, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(FusionBlock {
            residual_unit: if has_skip { Some(ResidualUnit::new(ch, vb.pp("residual_unit"))?) } else { None },
            main_unit: ResidualUnit::new(ch, vb.pp("main_unit"))?,
            out_conv: conv2d(ch, ch, 1, Conv2dConfig::default(), vb.pp("out_conv"))?,
        })
    }

    fn forward(&self, x: &Tensor, skip: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        if let (Some(unit), Some(skip)) = (&self.residual_unit, skip) {
            let (_, _, h, w) = x.dims4()?;
            let skip = if skip.dims() != x.dims() { skip.upsample_bilinear2d(h, w, false)? } else { skip.clone() };
            x = (x + unit.forward(&skip)?)?;
        }
        let x = self.main_unit.forward(&x)?;
        let (_, _, h, w) = x.dims4()?;
        let x = x.upsample_bilinear2d(h * FUSION_UPSAMPLE, w * FUSION_UPSAMPLE, true)?;
        self.out_conv.forward(&x)
    }
}

struct DepthHead {
    reassemble: Vec<ReassembleLevel>,
    convs: Vec<Conv2d>,
    fusion: Vec<FusionBlock>,
    project: Conv2d,
    classifier: Linear,
    centers: Tensor,
    min_depth: f64,
}

impl DepthHead {
    fn new(cfg: &TipsConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let ch = cfg.dpt_channels;
        let same = Conv2dConfig { padding: SAME_PAD_3X3, ..Default::default() };
        let reassemble = (0..DPT_LEVELS).map(|i| ReassembleLevel::new(i, cfg, vb.pp("reassemble"))).collect::<candle_core::Result<_>>()?;
        let convs = (0..DPT_LEVELS)
            .map(|i| conv2d_no_bias(cfg.post_process_channels[i], ch, 3, same, vb.pp("convs").pp(i)))
            .collect::<candle_core::Result<_>>()?;
        let fusion = (0..DPT_LEVELS).map(|i| FusionBlock::new(ch, i > 0, vb.pp("fusion_blocks").pp(i))).collect::<candle_core::Result<_>>()?;
        Ok(DepthHead {
            reassemble,
            convs,
            fusion,
            project: conv2d(ch, ch, 3, same, vb.pp("project"))?,
            classifier: linear(ch, cfg.depth_bins, vb.pp("depth_head"))?,
            centers: bin_centers(cfg.min_depth, cfg.max_depth, cfg.depth_bins, vb.device())?,
            min_depth: cfg.min_depth as f64,
        })
    }

    fn forward(&self, taps: &[Tap], spread: bool) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let levels: Vec<Tensor> = taps
            .iter()
            .zip(&self.reassemble)
            .zip(&self.convs)
            .map(|((tap, level), conv)| conv.forward(&level.forward(tap)?))
            .collect::<candle_core::Result<_>>()?;
        let mut out = self.fusion[0].forward(&levels[DPT_LEVELS - 1], None)?;
        for (i, block) in self.fusion.iter().enumerate().skip(1) {
            out = block.forward(&out, Some(&levels[DPT_LEVELS - 1 - i]))?;
        }
        let out = self.project.forward(&out)?.relu()?;
        let logits = self.classifier.forward(&out.permute((0, 2, 3, 1))?.contiguous()?)?;
        if spread {
            let (mean, sd) = decode_bins_spread(&logits, &self.centers, self.min_depth)?;
            Ok((mean.unsqueeze(1)?, Some(sd.unsqueeze(1)?)))
        } else {
            Ok((decode_bins(&logits, &self.centers, self.min_depth)?.unsqueeze(1)?, None))
        }
    }
}

fn bin_centers(min: f32, max: f32, bins: usize, device: &Device) -> candle_core::Result<Tensor> {
    let step = (max as f64 - min as f64) / (bins - 1) as f64;
    let centers: Vec<f32> = (0..bins).map(|i| (min as f64 + step * i as f64) as f32).collect();
    Tensor::from_vec(centers, bins, device)
}

fn decode_bins(logits: &Tensor, centers: &Tensor, min_depth: f64) -> candle_core::Result<Tensor> {
    let p = (logits.relu()? + min_depth)?;
    let p = p.broadcast_div(&p.sum_keepdim(D::Minus1)?)?;
    p.broadcast_mul(centers)?.sum(D::Minus1)
}

fn decode_bins_spread(logits: &Tensor, centers: &Tensor, min_depth: f64) -> candle_core::Result<(Tensor, Tensor)> {
    let p = (logits.relu()? + min_depth)?;
    let p = p.broadcast_div(&p.sum_keepdim(D::Minus1)?)?;
    let mean = p.broadcast_mul(centers)?.sum(D::Minus1)?;
    let second = p.broadcast_mul(&centers.sqr()?)?.sum(D::Minus1)?;
    let var = (second - mean.sqr()?)?.relu()?;
    Ok((mean, var.sqrt()?))
}

pub struct TipsDepth {
    cfg: TipsConfig,
    encoder: Encoder,
    head: DepthHead,
    device: Device,
    pub input_grid: InputGrid,
    pub max_rel_spread: Option<f32>,
}

impl TipsDepth {
    pub fn load(dir: &Path, device: &Device) -> anyhow::Result<Self> {
        let weights = dir.join(WEIGHTS_FILE);
        if !weights.exists() {
            bail!(
                "TIPSv2 DPT weights not found at {}. Download google/tipsv2-b14-dpt ({WEIGHTS_FILE}, optionally \
                 {CONFIG_FILE}) into that folder, or point ${DIR_ENV} at the folder that has them (default {DEFAULT_DIR}).",
                weights.display()
            );
        }
        let config_path = dir.join(CONFIG_FILE);
        let cfg = if config_path.exists() { TipsConfig::from_file(&config_path)? } else { TipsConfig::default() };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights], DType::F32, device)? };
        Self::from_var_builder(cfg, vb, device).with_context(|| format!("loading {}", weights.display()))
    }

    pub fn from_var_builder(cfg: TipsConfig, vb: VarBuilder, device: &Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        let encoder = Encoder::new(&cfg, vb.pp("vision_encoder"))?;
        let head = DepthHead::new(&cfg, vb.pp("depth_head"))?;
        Ok(TipsDepth { cfg, encoder, head, device: device.clone(), input_grid: InputGrid::default(), max_rel_spread: None })
    }

    pub fn config(&self) -> &TipsConfig {
        &self.cfg
    }

    pub fn set_readout_gelu_tanh(&mut self, on: bool) {
        self.cfg.readout_gelu_tanh = on;
        for level in &mut self.head.reassemble {
            level.gelu_tanh = on;
        }
    }

    pub fn predict(&self, rgb: &Rgb) -> anyhow::Result<DepthMap> {
        let mut out = self.predict_batch(&[rgb])?;
        Ok(out.pop().expect("one map per frame"))
    }

    pub fn predict_batch(&self, frames: &[&Rgb]) -> anyhow::Result<Vec<DepthMap>> {
        Ok(self.run_batch(frames, false)?.into_iter().map(|(d, _)| d).collect())
    }

    pub fn predict_batch_spread(&self, frames: &[&Rgb]) -> anyhow::Result<Vec<(DepthMap, DepthMap)>> {
        Ok(self.run_batch(frames, true)?.into_iter().map(|(d, s)| (d, s.expect("spread was asked for"))).collect())
    }

    fn run_batch(&self, frames: &[&Rgb], want_spread: bool) -> anyhow::Result<Vec<(DepthMap, Option<DepthMap>)>> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<(u32, u32), Vec<usize>> = BTreeMap::new();
        for (i, f) in frames.iter().enumerate() {
            if f.width == 0 || f.height == 0 {
                bail!("empty keyframe ({}x{})", f.width, f.height);
            }
            groups.entry((f.width, f.height)).or_default().push(i);
        }
        let policy = noto_ml::BatchPolicy::for_device(&self.device);
        let p = self.cfg.patch_size;
        let mut out: Vec<Option<(DepthMap, Option<DepthMap>)>> = (0..frames.len()).map(|_| None).collect();
        let spread = want_spread || self.max_rel_spread.is_some();
        for (&(w, h), idx) in &groups {
            let (cols, rows) = self.input_grid.grid_for(w, h);
            for chunk in idx.chunks(policy.image_rows(cols * rows)) {
                let inputs = chunk.iter().map(|&i| model_input(frames[i], (cols * p) as u32, (rows * p) as u32, &self.device)).collect::<anyhow::Result<Vec<_>>>()?;
                let pixels = Tensor::cat(&inputs, 0)?;
                let taps = self.encoder.taps(&pixels)?;
                let (depth, sd) = self.head.forward(&taps, spread)?;
                let depth = depth.upsample_bilinear2d(h as usize, w as usize, false)?;
                let sd = sd.map(|t| t.upsample_bilinear2d(h as usize, w as usize, false)).transpose()?;
                for (k, &i) in chunk.iter().enumerate() {
                    let mut map = DepthMap { width: w, height: h, kind: DepthKind::Metric, data: depth.get(k)?.flatten_all()?.to_vec1()? };
                    clamp_metric(&mut map, self.cfg.max_depth);
                    let sd = match &sd {
                        Some(t) => Some(DepthMap { width: w, height: h, kind: DepthKind::Metric, data: t.get(k)?.flatten_all()?.to_vec1()? }),
                        None => None,
                    };
                    if let (Some(limit), Some(s)) = (self.max_rel_spread, &sd) {
                        for (d, e) in map.data.iter_mut().zip(&s.data) {
                            if *e > limit * *d {
                                *d = f32::INFINITY;
                            }
                        }
                    }
                    out[i] = Some((map, if want_spread { sd } else { None }));
                }
            }
        }
        noto_ml::trim_idle_pool(&self.device);
        Ok(out.into_iter().map(|m| m.expect("every frame is in one group")).collect())
    }
}

fn model_input(rgb: &Rgb, width: u32, height: u32, device: &Device) -> anyhow::Result<Tensor> {
    let resized;
    let data: &[u8] = if (rgb.width, rgb.height) == (width, height) {
        &rgb.data
    } else {
        let src = image::RgbImage::from_raw(rgb.width, rgb.height, rgb.data.clone()).context("RGB buffer does not match its size")?;
        resized = image::imageops::resize(&src, width, height, image::imageops::FilterType::Triangle);
        resized.as_raw()
    };
    let plane = (width * height) as usize;
    let mut chw = vec![0f32; 3 * plane];
    for (i, px) in data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            chw[c * plane + i] = px[c] as f32 / 255.0;
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, height as usize, width as usize), device)?)
}

impl DepthModel for TipsDepth {
    fn mode(&self) -> &str {
        MODE
    }

    fn device(&self) -> String {
        noto_ml::device_label(&self.device)
    }

    fn max_depth_m(&self) -> Option<f32> {
        Some(self.cfg.max_depth)
    }

    fn infer(&mut self, rgb: &Rgb) -> anyhow::Result<DepthMap> {
        self.predict(rgb)
    }

    fn infer_batch(&mut self, frames: &[&Rgb]) -> anyhow::Result<Vec<DepthMap>> {
        self.predict_batch(frames)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::{Arc, Mutex};

    use candle_core::Shape;
    use candle_nn::var_builder::SimpleBackend;

    use super::*;

    const DEPTH_PREFIXES: [&str; 2] = ["vision_encoder.", "depth_head."];
    const UNUSED_AT_INFERENCE: [&str; 1] = ["vision_encoder.mask_token"];
    const WEIGHT_RANGE: f32 = 0.05;

    type Shapes = BTreeMap<String, Vec<usize>>;
    type Requested = Arc<Mutex<BTreeSet<String>>>;

    fn checkpoint_shapes() -> Shapes {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tipsv2_b14_dpt_header.json");
        let raw = std::fs::read_to_string(&path).unwrap();
        let header: BTreeMap<String, serde_json::Value> = serde_json::from_str(&raw).unwrap();
        header
            .into_iter()
            .map(|(name, meta)| {
                assert_eq!(meta["dtype"], "F32", "{name}");
                let shape = meta["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                (name, shape)
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    enum Fill {
        Random,
        Zero,
    }

    struct HeaderBackend {
        shapes: Shapes,
        fill: Fill,
        requested: Requested,
    }

    impl HeaderBackend {
        fn tensor(&self, name: &str, want: Option<&Shape>, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            let Some(shape) = self.shapes.get(name) else { candle_core::bail!("{name} is not a depth-path tensor of the checkpoint") };
            if let Some(want) = want {
                if want.dims() != shape.as_slice() {
                    candle_core::bail!("{name}: the port asks for {:?}, the checkpoint has {shape:?}", want.dims());
                }
            }
            self.requested.lock().unwrap().insert(name.to_string());
            match self.fill {
                Fill::Random => Tensor::rand(-WEIGHT_RANGE, WEIGHT_RANGE, shape.as_slice(), dev)?.to_dtype(dtype),
                Fill::Zero => Tensor::zeros((), dtype, dev)?.broadcast_as(shape.as_slice()),
            }
        }
    }

    impl SimpleBackend for HeaderBackend {
        fn get(&self, s: Shape, name: &str, _: candle_nn::Init, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            self.tensor(name, Some(&s), dtype, dev)
        }

        fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
            self.tensor(name, None, dtype, dev)
        }

        fn contains_tensor(&self, name: &str) -> bool {
            self.shapes.contains_key(name)
        }
    }

    fn header_model(fill: Fill) -> (TipsDepth, Shapes, Requested) {
        let shapes: Shapes = checkpoint_shapes().into_iter().filter(|(n, _)| DEPTH_PREFIXES.iter().any(|p| n.starts_with(p))).collect();
        let requested = Arc::new(Mutex::new(BTreeSet::new()));
        let backend = HeaderBackend { shapes: shapes.clone(), fill, requested: requested.clone() };
        let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, Device::Cpu);
        let model = TipsDepth::from_var_builder(TipsConfig::default(), vb, &Device::Cpu).unwrap();
        (model, shapes, requested)
    }

    fn gradient(width: u32, height: u32) -> Rgb {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, ((x + y) * 7 % 256) as u8]);
            }
        }
        Rgb::new(width, height, data)
    }

    #[test]
    fn runs_on_every_name_and_shape_of_the_real_checkpoint() {
        let (mut model, _, _) = header_model(Fill::Random);
        model.input_grid = InputGrid::Fixed { cols: 4, rows: 3 };
        let cfg = model.config().clone();
        let map = model.infer(&gradient(50, 37)).unwrap();
        assert_eq!((map.width, map.height), (50, 37));
        assert_eq!(map.kind, DepthKind::Metric);
        assert_eq!(map.data.len(), 50 * 37);
        assert!(map.data.iter().all(|d| d.is_finite() && (cfg.min_depth..=cfg.max_depth).contains(d)), "depth outside the bins");
        assert_eq!(model.mode(), "tipsv2-b14-dpt");
        assert_eq!(model.max_depth_m(), Some(10.0));
        assert_eq!(model.device(), "CPU");
    }

    #[test]
    fn every_depth_path_tensor_of_the_checkpoint_is_read() {
        let (_model, shapes, requested) = header_model(Fill::Zero);
        let requested = requested.lock().unwrap();
        let expected: BTreeSet<String> = shapes.keys().filter(|n| !UNUSED_AT_INFERENCE.contains(&n.as_str())).cloned().collect();
        let unread: Vec<&String> = expected.difference(&requested).collect();
        assert!(unread.is_empty(), "checkpoint tensors the port ignores: {unread:?}");
        assert!(requested.iter().all(|n| expected.contains(n)), "the port read a tensor it documents as unused");
        assert_eq!(requested.len(), 175 + 52, "176 encoder tensors minus mask_token, 52 depth-head tensors");
    }

    #[test]
    fn bins_decode_to_metres() {
        let cfg = TipsConfig::default();
        let (m, n) = (cfg.min_depth as f64, cfg.depth_bins);
        let centers = bin_centers(cfg.min_depth, cfg.max_depth, n, &Device::Cpu).unwrap();
        let c = centers.to_vec1::<f32>().unwrap();
        assert!((c[0] as f64 - 0.001).abs() < 1e-7 && (c[n - 1] as f64 - 10.0).abs() < 1e-6);
        let sum_c: f64 = c.iter().map(|&x| x as f64).sum();

        let (big, k) = (1000.0f32, 200usize);
        let mut logits = vec![-3.0f32; n];
        let mut peaked = vec![0.0f32; n];
        peaked[k] = big;
        logits.extend(peaked);
        let t = Tensor::from_vec(logits, (1, 2, n), &Device::Cpu).unwrap();
        let depth = decode_bins(&t, &centers, m).unwrap().to_vec2::<f32>().unwrap();

        let uniform = sum_c / n as f64;
        assert!((uniform - (0.001 + 10.0) / 2.0).abs() < 1e-5);
        assert!((depth[0][0] as f64 - uniform).abs() < 1e-4, "{} vs {uniform}", depth[0][0]);
        let expected = (m * sum_c + big as f64 * c[k] as f64) / (n as f64 * m + big as f64);
        assert!((depth[0][1] as f64 - expected).abs() < 1e-4, "{} vs {expected}", depth[0][1]);
        assert!((depth[0][1] - c[k]).abs() < 0.01, "a confident bin dominates");
    }

    #[test]
    fn position_resampling_matches_torch_antialias() {
        let w = |src, dst| antialias_bilinear_weights(src, dst);
        assert_eq!(w(3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(w(2, 4), vec![1.0, 0.0, 0.75, 0.25, 0.25, 0.75, 0.0, 1.0]);
        let down = w(4, 2);
        let expect = [3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0, 0.0, 0.0, 1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0];
        assert!(down.iter().zip(expect).all(|(a, b)| (a - b).abs() < 1e-6), "{down:?}");
        for row in w(32, 24).chunks(32) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn default_grid_turns_keyframes_into_504_by_336() {
        assert_eq!(InputGrid::default().grid_for(384, 256), (36, 24));
        assert_eq!(InputGrid::ShortSide(24).grid_for(256, 384), (24, 36));
        assert_eq!(InputGrid::Fixed { cols: 4, rows: 3 }.grid_for(1000, 10), (4, 3));
    }

    #[test]
    fn config_reads_python_names_and_backbone_stages() {
        let c = TipsConfig::from_json(r#"{"out_indices": [3, 6, 9, 12], "num_depth_bins": 128, "img_size": 448}"#).unwrap();
        assert_eq!(c.tap_blocks, [2, 5, 8, 11]);
        assert_eq!(c.depth_bins, 128);
        assert_eq!(c.hidden, 768);
        assert!(TipsConfig::from_json(r#"{"vision_fn": "vit_large"}"#).is_err());
        assert!(TipsConfig::from_json(r#"{"out_indices": [0, 6, 9, 12]}"#).is_err());
    }

    #[test]
    fn missing_weights_name_the_folder_and_files() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/no-such-model");
        let err = TipsDepth::load(&dir, &Device::Cpu).err().unwrap().to_string();
        assert!(err.contains("model.safetensors") && err.contains("no-such-model") && err.contains("TIPS_DIR"), "{err}");
    }
}
