//! Granite TTM-R3 (TinyTimeMixer) forecaster on candle.
//!
//! Copied from endmill-sandbox `src/ml/ttm.rs` (same author, Apache-2.0). Only
//! the three calls into that project's `ml` module were replaced by the local
//! helpers in `io.rs`; the model code is unchanged. It reads `config.json` and
//! `model.safetensors` from the `granite-timeseries-ttm-r3/52-16-dec-52-r3`
//! folder of `ibm-granite/granite-timeseries-ttm-r3`.
//!
//! Unsupported configuration options are rejected in [`TtmForecaster::validate`]
//! instead of being run with wrong results.

use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn d1() -> usize {
    1
}
fn d2() -> usize {
    2
}
fn d3() -> usize {
    3
}
fn d4() -> usize {
    4
}
fn d8() -> usize {
    8
}
fn d16() -> usize {
    16
}
fn d64() -> usize {
    64
}
fn dtrue() -> bool {
    true
}
fn d_eps() -> f64 {
    1e-5
}
fn d_mq_eps() -> f64 {
    1e-6
}
fn s_common() -> String {
    "common_channel".into()
}
fn s_layernorm() -> String {
    "LayerNorm".into()
}
fn s_median() -> String {
    "median".into()
}
fn s_pool() -> String {
    "pool".into()
}
fn s_add() -> String {
    "add".into()
}
fn s_sincos() -> String {
    "sincos".into()
}
fn s_std() -> serde_json::Value {
    serde_json::Value::String("std".into())
}
fn default_q() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}

#[derive(Debug, Clone, Deserialize)]
pub struct TtmConfig {
    #[serde(default = "d64")]
    pub context_length: usize,
    #[serde(default = "d8")]
    pub patch_length: usize,
    #[serde(default = "d8")]
    pub patch_stride: usize,
    #[serde(default = "d1")]
    pub num_input_channels: usize,
    #[serde(default = "d16")]
    pub prediction_length: usize,
    #[serde(default = "d16")]
    pub d_model: usize,
    #[serde(default = "d2")]
    pub expansion_factor: usize,
    #[serde(default = "d3")]
    pub num_layers: usize,
    #[serde(default = "s_common")]
    pub mode: String,
    #[serde(default = "dtrue")]
    pub gated_attn: bool,
    #[serde(default = "s_layernorm")]
    pub norm_mlp: String,
    #[serde(default)]
    pub self_attn: bool,
    #[serde(default)]
    pub use_positional_encoding: bool,
    #[serde(default = "s_sincos")]
    pub positional_encoding_type: String,
    #[serde(default = "s_std")]
    pub scaling: serde_json::Value,
    #[serde(default = "d_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub adaptive_patching_levels: usize,
    #[serde(default)]
    pub resolution_prefix_tuning: bool,
    #[serde(default = "d8")]
    pub decoder_num_layers: usize,
    #[serde(default = "d8")]
    pub decoder_d_model: usize,
    #[serde(default)]
    pub decoder_adaptive_patching_levels: usize,
    #[serde(default)]
    pub decoder_raw_residual: bool,
    #[serde(default = "s_common")]
    pub decoder_mode: String,
    #[serde(default = "dtrue")]
    pub use_decoder: bool,
    #[serde(default)]
    pub enable_forecast_channel_mixing: bool,
    #[serde(default)]
    pub categorical_vocab_size_list: Option<Vec<usize>>,
    #[serde(default)]
    pub prediction_filter_length: Option<usize>,
    #[serde(default)]
    pub multi_scale: bool,
    #[serde(default)]
    pub register_tokens: usize,
    #[serde(default)]
    pub fft_length: usize,
    #[serde(default = "dtrue")]
    pub use_fft_embedding: bool,
    #[serde(default)]
    pub get_one_freq_emb: bool,
    #[serde(default)]
    pub fft_ignore_dc: bool,
    #[serde(default = "d4")]
    pub fft_d_freq_min: usize,
    #[serde(default = "d64")]
    pub fft_d_freq_max: usize,
    #[serde(default)]
    pub multi_quantile_head: bool,
    #[serde(default)]
    pub residual_context_length: Option<usize>,
    #[serde(default)]
    pub trend_patch_length: Option<usize>,
    #[serde(default)]
    pub trend_patch_stride: Option<usize>,
    #[serde(default)]
    pub trend_d_model: Option<usize>,
    #[serde(default)]
    pub trend_decoder_d_model: Option<usize>,
    #[serde(default)]
    pub trend_num_layers: Option<usize>,
    #[serde(default)]
    pub trend_decoder_num_layers: Option<usize>,
    #[serde(default)]
    pub trend_register_tokens: Option<usize>,
    #[serde(default)]
    pub trend_fft_length: Option<usize>,
    #[serde(default)]
    pub trend_multi_scale: Option<bool>,
    #[serde(default)]
    pub trend_adaptive_patching_levels: Option<usize>,
    #[serde(default)]
    pub decompose: bool,
    #[serde(default = "d8")]
    pub mq_hidden: usize,
    #[serde(default = "d3")]
    pub mq_kernel_size: usize,
    #[serde(default = "d_mq_eps")]
    pub mq_eps: f64,
    #[serde(default)]
    pub mq_use_decoder_pool: bool,
    #[serde(default = "s_median")]
    pub mq_q50_type: String,
    #[serde(default = "s_pool")]
    pub mq_cond_path: String,
    #[serde(default = "s_add")]
    pub mq_cond_mode: String,
    #[serde(default = "d8")]
    pub mq_decoder_d_model: usize,
    #[serde(default)]
    pub combine_quantiles_via_variance: bool,
    #[serde(default = "default_q")]
    pub quantile_levels: Vec<f64>,
    #[serde(default)]
    pub enable_base_norm_always: bool,
    #[serde(default)]
    pub gate_mode: Option<String>,
    #[serde(default)]
    pub gate_groups: Option<usize>,
    #[serde(default)]
    pub use_register_context_gating: bool,
    #[serde(default)]
    pub num_patches: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Scaling {
    Std,
    Mean,
    Nop,
}

impl Scaling {
    fn from_value(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::String(s) if s == "std" => Scaling::Std,
            serde_json::Value::String(s) if s == "mean" => Scaling::Mean,
            serde_json::Value::Bool(true) => Scaling::Std,
            _ => Scaling::Nop,
        }
    }

    fn apply(&self, data: &[f64], observed: &[bool]) -> (Vec<f64>, f64, f64) {
        match self {
            Scaling::Nop => (data.to_vec(), 0.0, 1.0),
            Scaling::Std => {
                let cnt = observed.iter().filter(|o| **o).count().max(1) as f64;
                let loc = data.iter().zip(observed).filter(|(_, o)| **o).map(|(x, _)| *x).sum::<f64>() / cnt;
                let var = data
                    .iter()
                    .zip(observed)
                    .map(|(x, o)| if *o { (x - loc) * (x - loc) } else { 0.0 })
                    .sum::<f64>()
                    / cnt;
                let scale = (var + 1e-5).sqrt();
                (data.iter().map(|x| (x - loc) / scale).collect(), loc, scale)
            }
            Scaling::Mean => {
                let cnt = observed.iter().filter(|o| **o).count();
                let s: f64 = data.iter().zip(observed).filter(|(_, o)| **o).map(|(x, _)| x.abs()).sum();
                let mut scale = if cnt > 0 { s / cnt as f64 } else { s };
                if scale < 1e-10 {
                    scale = 1e-10;
                }
                (data.iter().map(|x| x / scale).collect(), 0.0, scale)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SubConfig {
    context_length: usize,
    patch_length: usize,
    d_model: usize,
    decoder_d_model: usize,
    num_layers: usize,
    decoder_num_layers: usize,
    register_tokens: usize,
    fft_length: usize,
    multi_scale: bool,
    scaling: Scaling,
    num_patches: usize,
    n_base_patches: usize,
}

fn multiscale_patches(context_length: usize, patch_length: usize) -> usize {
    let mut total = 0;
    let mut i = 0u32;
    loop {
        let factor = 1usize << i;
        let ds = context_length / factor;
        if ds < patch_length {
            break;
        }
        total += ds / patch_length;
        i += 1;
    }
    total
}

impl SubConfig {
    fn derive(cfg: &TtmConfig, which: &str) -> std::result::Result<Self, String> {
        let mut s = SubConfig {
            context_length: cfg.context_length,
            patch_length: cfg.patch_length,
            d_model: cfg.d_model,
            decoder_d_model: cfg.decoder_d_model,
            num_layers: cfg.num_layers,
            decoder_num_layers: cfg.decoder_num_layers,
            register_tokens: cfg.register_tokens,
            fft_length: cfg.fft_length,
            multi_scale: cfg.multi_scale,
            scaling: Scaling::from_value(&cfg.scaling),
            num_patches: 0,
            n_base_patches: 0,
        };
        let mut stride = cfg.patch_stride;
        if which == "trend" {
            if let Some(v) = cfg.trend_patch_length {
                s.patch_length = v;
            }
            if let Some(v) = cfg.trend_patch_stride {
                stride = v;
            }
            if let Some(v) = cfg.trend_d_model {
                s.d_model = v;
            }
            if let Some(v) = cfg.trend_decoder_d_model {
                s.decoder_d_model = v;
            }
            if let Some(v) = cfg.trend_num_layers {
                s.num_layers = v;
            }
            if let Some(v) = cfg.trend_decoder_num_layers {
                s.decoder_num_layers = v;
            }
            if let Some(v) = cfg.trend_register_tokens {
                s.register_tokens = v;
            }
            if let Some(v) = cfg.trend_fft_length {
                s.fft_length = v;
            }
            if let Some(v) = cfg.trend_multi_scale {
                s.multi_scale = v;
            }
            if cfg.trend_adaptive_patching_levels.unwrap_or(0) > 0 {
                return Err("TTM: trend_adaptive_patching_levels > 0 은 지원하지 않습니다".into());
            }
            s.scaling = Scaling::Nop;
        } else if which == "residual" {
            if let Some(v) = cfg.residual_context_length {
                s.context_length = v;
            }
            s.scaling = Scaling::Nop;
        }
        if s.patch_length != stride {
            return Err("TTM: patch_length 와 patch_stride 가 달라 지원하지 않습니다".into());
        }
        if s.context_length <= s.patch_length {
            return Err("TTM: context_length 가 patch_length 이하입니다".into());
        }
        s.n_base_patches = (s.context_length - s.patch_length) / s.patch_length + 1;
        let mut np = if s.multi_scale {
            multiscale_patches(s.context_length, s.patch_length)
        } else {
            s.n_base_patches
        };
        np += s.register_tokens;
        if s.fft_length > 0 {
            np += if cfg.get_one_freq_emb { 1 } else { s.fft_length };
        }
        s.num_patches = match (which, cfg.num_patches) {
            ("trend", _) => np,
            (_, Some(given)) => given,
            _ => np,
        };
        Ok(s)
    }
}

#[derive(Debug, Clone)]
enum GateKind {
    Softmax,
    Sigmoid,
    Glu,
    GroupSigmoid(usize),
}

#[derive(Debug, Clone)]
struct Gate {
    layer: Linear,
    kind: GateKind,
    size: usize,
}

impl Gate {
    fn load(cfg: &TtmConfig, size: usize, vb: VarBuilder) -> Result<Self> {
        let mode = cfg.gate_mode.clone().unwrap_or_else(|| "softmax".into());
        let kind = match mode.as_str() {
            "sigmoid" => GateKind::Sigmoid,
            "glu" => GateKind::Glu,
            "group_sigmoid" => {
                let req = cfg.gate_groups.unwrap_or(8);
                let mut g = 1;
                if req > 1 {
                    if size % req == 0 {
                        g = req;
                    } else {
                        for cand in (1..=req.min(size)).rev() {
                            if size % cand == 0 {
                                g = cand;
                                break;
                            }
                        }
                    }
                }
                GateKind::GroupSigmoid(g)
            }
            _ => GateKind::Softmax,
        };
        let out = if matches!(kind, GateKind::Glu) { 2 * size } else { size };
        Ok(Self {
            layer: candle_nn::linear(size, out, vb.pp("attn_layer"))?,
            kind,
            size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = x.apply(&self.layer)?;
        match &self.kind {
            GateKind::Softmax => x * candle_nn::ops::softmax_last_dim(&logits)?,
            GateKind::Sigmoid => x * candle_nn::ops::sigmoid(&logits)?,
            GateKind::Glu => {
                let parts = logits.chunk(2, D::Minus1)?;
                &parts[0] * candle_nn::ops::sigmoid(&parts[1])?
            }
            GateKind::GroupSigmoid(g) => {
                let dims = logits.dims().to_vec();
                let mut shape = dims.clone();
                let last = shape.pop().unwrap_or(self.size);
                shape.push(*g);
                shape.push(last / g);
                let grouped = logits.reshape(shape.clone())?.mean_keepdim(D::Minus1)?;
                let gate = candle_nn::ops::sigmoid(&grouped)?.broadcast_as(shape)?.reshape(dims)?;
                x * gate
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Norm {
    Layer(LayerNorm),
    Batch { w: Tensor, b: Tensor, mean: Tensor, var: Tensor, eps: f64 },
}

impl Norm {
    fn load(cfg: &TtmConfig, d: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.norm_mlp.to_lowercase().contains("batch") {
            let bv = vb.pp("norm").pp("batchnorm");
            Ok(Norm::Batch {
                w: bv.get(d, "weight")?,
                b: bv.get(d, "bias")?,
                mean: bv.get(d, "running_mean")?,
                var: bv.get(d, "running_var")?,
                eps: cfg.norm_eps,
            })
        } else {
            Ok(Norm::Layer(candle_nn::layer_norm(d, cfg.norm_eps, vb.pp("norm"))?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Norm::Layer(l) => x.apply(l),
            Norm::Batch { w, b, mean, var, eps } => {
                let denom = (var + *eps)?.sqrt()?;
                x.broadcast_sub(mean)?.broadcast_div(&denom)?.broadcast_mul(w)?.broadcast_add(b)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(i: usize, o: usize, expansion: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(i, i * expansion, vb.pp("fc1"))?,
            fc2: candle_nn::linear(i * expansion, o, vb.pp("fc2"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct MixerLayer {
    patch_norm: Option<Norm>,
    patch_mlp: Option<Mlp>,
    patch_gate: Option<Gate>,
    feat_norm: Norm,
    feat_mlp: Mlp,
    feat_gate: Option<Gate>,
}

impl MixerLayer {
    fn load(cfg: &TtmConfig, num_patches: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let (patch_norm, patch_mlp, patch_gate) = if num_patches > 1 {
            let pv = vb.pp("patch_mixer");
            (
                Some(Norm::load(cfg, d, pv.pp("norm"))?),
                Some(Mlp::load(num_patches, num_patches, cfg.expansion_factor, pv.pp("mlp"))?),
                if cfg.gated_attn {
                    Some(Gate::load(cfg, num_patches, pv.pp("gating_block"))?)
                } else {
                    None
                },
            )
        } else {
            (None, None, None)
        };
        let fv = vb.pp("feature_mixer");
        Ok(Self {
            patch_norm,
            patch_mlp,
            patch_gate,
            feat_norm: Norm::load(cfg, d, fv.pp("norm"))?,
            feat_mlp: Mlp::load(d, d, cfg.expansion_factor, fv.pp("mlp"))?,
            feat_gate: if cfg.gated_attn {
                Some(Gate::load(cfg, d, fv.pp("gating_block"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        if let (Some(norm), Some(mlp)) = (&self.patch_norm, &self.patch_mlp) {
            let r = h.clone();
            let mut y = norm.forward(&h)?.transpose(2, 3)?.contiguous()?;
            y = y.apply(mlp)?;
            if let Some(g) = &self.patch_gate {
                y = g.forward(&y)?;
            }
            h = (y.transpose(2, 3)? + r)?;
        }
        let r = h.clone();
        let mut y = self.feat_norm.forward(&h)?.apply(&self.feat_mlp)?;
        if let Some(g) = &self.feat_gate {
            y = g.forward(&y)?;
        }
        y + r
    }
}

#[derive(Debug, Clone)]
struct Block {
    layers: Vec<MixerLayer>,
}

impl Block {
    fn load(cfg: &TtmConfig, n: usize, num_patches: usize, d: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(MixerLayer::load(cfg, num_patches, d, vb.pp("mixers").pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for l in self.layers.iter() {
            h = l.forward(&h)?;
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
enum Patcher {
    Single(Linear),
    Multi(Vec<Linear>),
}

#[derive(Debug, Clone)]
enum FftEmbed {
    Table(Tensor),
    Mlp(Linear, Linear),
}

#[derive(Debug, Clone)]
struct FftTokens {
    k: usize,
    max_bins: usize,
    embed: FftEmbed,
    one: Option<(Linear, usize)>,
    ignore_dc: bool,
}

#[derive(Debug, Clone)]
struct MqHead {
    dw_w: Vec<f32>,
    dw_b: f32,
    in_w: f32,
    in_b: f32,
    pw1_w: Vec<f32>,
    pw1_b: Vec<f32>,
    pool: Option<Linear>,
    flat: Option<(Linear, Linear)>,
    pw2_w: Vec<Vec<f32>>,
    pw2_b: Vec<f32>,
    hidden: usize,
    concat: bool,
    use_cond: bool,
    q50_median: bool,
    k_side: usize,
    eps: f64,
    num_q: usize,
}

impl MqHead {
    fn load(cfg: &TtmConfig, decoder_d_model: usize, num_patches: usize, vb: VarBuilder) -> Result<Self> {
        let q = cfg.quantile_levels.len();
        let h = cfg.mq_hidden;
        let ks = cfg.mq_kernel_size;
        let use_cond = cfg.mq_use_decoder_pool;
        let concat = use_cond && cfg.mq_cond_mode == "concat";
        let pw2_in = if concat { 2 * h } else { h };
        let dw_w = vb.get((1, 1, ks), "dw.weight")?.flatten_all()?.to_vec1::<f32>()?;
        let dw_b = vb.get(1, "dw.bias")?.to_vec1::<f32>()?[0];
        let in_w = vb.get(1, "inorm.weight")?.to_vec1::<f32>()?[0];
        let in_b = vb.get(1, "inorm.bias")?.to_vec1::<f32>()?[0];
        let pw1_w = vb.get((h, 1, 1), "pw1.weight")?.flatten_all()?.to_vec1::<f32>()?;
        let pw1_b = vb.get(h, "pw1.bias")?.to_vec1::<f32>()?;
        let pw2_w = vb.get((q, pw2_in, 1), "pw2.weight")?.reshape((q, pw2_in))?.to_vec2::<f32>()?;
        let pw2_b = vb.get(q, "pw2.bias")?.to_vec1::<f32>()?;
        let (pool, flat) = if use_cond {
            if cfg.mq_cond_path == "pool" {
                (Some(candle_nn::linear(decoder_d_model, h, vb.pp("dec_proj"))?), None)
            } else {
                let md = cfg.mq_decoder_d_model;
                (
                    None,
                    Some((
                        candle_nn::linear(decoder_d_model, md, vb.pp("dec_tok_proj"))?,
                        candle_nn::linear(num_patches * md, h, vb.pp("dec_flat_to_hidden"))?,
                    )),
                )
            }
        } else {
            (None, None)
        };
        Ok(Self {
            dw_w,
            dw_b,
            in_w,
            in_b,
            pw1_w,
            pw1_b,
            pool,
            flat,
            pw2_w,
            pw2_b,
            hidden: h,
            concat,
            use_cond,
            q50_median: cfg.mq_q50_type == "median",
            k_side: q / 2,
            eps: cfg.mq_eps,
            num_q: q,
        })
    }

    fn forward(&self, mean_hat: &[f64], dec: &Tensor) -> Result<Vec<Vec<f64>>> {
        let t = mean_hat.len();
        let ks = self.dw_w.len();
        let pad = ks / 2;
        let mut x1 = vec![0f64; t];
        for i in 0..t {
            let mut s = self.dw_b as f64;
            for j in 0..ks {
                let idx = i as i64 + j as i64 - pad as i64;
                if idx >= 0 && (idx as usize) < t {
                    s += self.dw_w[j] as f64 * mean_hat[idx as usize];
                }
            }
            x1[i] = s;
        }
        let m = x1.iter().sum::<f64>() / t as f64;
        let v = x1.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / t as f64;
        let x2: Vec<f64> = x1
            .iter()
            .map(|a| (a - m) / (v + 1e-5).sqrt() * self.in_w as f64 + self.in_b as f64)
            .collect();
        let gelu = |z: f64| 0.5 * z * (1.0 + erf(z / std::f64::consts::SQRT_2));
        let mut hmat: Vec<Vec<f64>> = (0..self.hidden)
            .map(|j| x2.iter().map(|a| gelu(self.pw1_w[j] as f64 * a + self.pw1_b[j] as f64)).collect())
            .collect();
        if self.use_cond {
            let cond: Vec<f32> = if let Some(p) = &self.pool {
                dec.mean(0)?.unsqueeze(0)?.apply(p)?.flatten_all()?.to_vec1::<f32>()?
            } else if let Some((tok, fl)) = &self.flat {
                let dt = dec.apply(tok)?.flatten_all()?.unsqueeze(0)?;
                dt.apply(fl)?.flatten_all()?.to_vec1::<f32>()?
            } else {
                vec![0.0; self.hidden]
            };
            if self.concat {
                for j in 0..self.hidden {
                    hmat.push(vec![cond[j] as f64; t]);
                }
            } else {
                for j in 0..self.hidden {
                    for i in 0..t {
                        hmat[j][i] += cond[j] as f64;
                    }
                }
            }
        }
        let o: Vec<Vec<f64>> = (0..self.num_q)
            .map(|qi| {
                (0..t)
                    .map(|i| {
                        let mut s = self.pw2_b[qi] as f64;
                        for (c, row) in hmat.iter().enumerate() {
                            s += self.pw2_w[qi][c] as f64 * row[i];
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        let softplus = |z: f64| if z > 20.0 { z } else { z.exp().ln_1p() };
        let k = self.k_side;
        let mut q50: Vec<f64> = mean_hat.to_vec();
        if self.q50_median {
            for i in 0..t {
                q50[i] += o[0][i];
            }
        }
        let mut down = vec![vec![0f64; t]; k];
        let mut up = vec![vec![0f64; t]; k];
        for i in 0..t {
            let mut dc = 0.0;
            let mut uc = 0.0;
            for j in 0..k {
                dc += softplus(o[1 + j][i]) + self.eps;
                uc += softplus(o[1 + k + j][i]) + self.eps;
                down[j][i] = q50[i] - dc;
                up[j][i] = q50[i] + uc;
            }
        }
        let mut out = Vec::with_capacity(2 * k + 1);
        for j in (0..k).rev() {
            out.push(down[j].clone());
        }
        out.push(q50);
        for j in 0..k {
            out.push(up[j].clone());
        }
        Ok(out)
    }
}

fn erf(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x > 6.0 {
        return s;
    }
    let t = 1.0 / (1.0 + 0.5 * x);
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    s * y
}

#[derive(Debug, Clone)]
struct Predictor {
    sc: SubConfig,
    patcher: Patcher,
    register: Option<Tensor>,
    fft: Option<FftTokens>,
    pos: Option<Tensor>,
    base_norm: Option<LayerNorm>,
    encoder: Block,
    adapter: Option<Linear>,
    decoder: Option<Block>,
    head: Linear,
    mq: Option<MqHead>,
    device: Device,
}

impl Predictor {
    fn load(cfg: &TtmConfig, sc: SubConfig, vb: VarBuilder) -> Result<Self> {
        let ev = vb.pp("backbone").pp("encoder");
        let d = sc.d_model;
        let patcher = if sc.multi_scale {
            let mut n = 0;
            let mut s = sc.context_length;
            while s >= sc.patch_length {
                n += 1;
                s /= 2;
            }
            let mut ps = Vec::with_capacity(n);
            for i in 0..n {
                ps.push(candle_nn::linear(sc.patch_length, d, ev.pp("patcher").pp("projectors").pp(i))?);
            }
            Patcher::Multi(ps)
        } else {
            Patcher::Single(candle_nn::linear(sc.patch_length, d, ev.pp("patcher"))?)
        };
        let register = if sc.register_tokens > 0 {
            Some(ev.get((sc.register_tokens, d), "add_tokens.patch_tokens")?)
        } else {
            None
        };
        let fft = if sc.fft_length > 0 {
            let fv = ev.pp("add_fft_tokens");
            let max_bins = sc.context_length / 2 + 1;
            let k = sc.fft_length.min(max_bins);
            let (dim, one) = if cfg.get_one_freq_emb {
                let raw = (d / k.max(1)).max(1);
                let df = raw.min(cfg.fft_d_freq_max).max(cfg.fft_d_freq_min);
                (df, Some((candle_nn::linear(k * df, d, fv.pp("freq_concat_proj"))?, df)))
            } else {
                (d, None)
            };
            let embed = if cfg.use_fft_embedding {
                FftEmbed::Table(fv.get((max_bins, dim), "freq_embedding.weight")?)
            } else {
                FftEmbed::Mlp(
                    candle_nn::linear(1, dim, fv.pp("freq_index_mlp").pp(0))?,
                    candle_nn::linear(dim, dim, fv.pp("freq_index_mlp").pp(2))?,
                )
            };
            Some(FftTokens {
                k,
                max_bins,
                embed,
                one,
                ignore_dc: cfg.fft_ignore_dc,
            })
        } else {
            None
        };
        let pos = if cfg.use_positional_encoding {
            let pv = ev.pp("positional_encoder");
            if pv.contains_tensor("position_enc") {
                Some(pv.get((sc.num_patches, d), "position_enc")?)
            } else {
                Some(sincos_pe(sc.num_patches, d, vb.device())?)
            }
        } else {
            None
        };
        let base_norm = if sc.multi_scale || cfg.enable_base_norm_always {
            Some(candle_nn::layer_norm(sc.num_patches * d, cfg.norm_eps, ev.pp("base_norm"))?)
        } else {
            None
        };
        let encoder = Block::load(cfg, sc.num_layers, sc.num_patches, d, ev.pp("mlp_mixer_encoder"))?;
        let (adapter, decoder) = if cfg.use_decoder {
            let dv = vb.pp("decoder");
            let adapter = if d != sc.decoder_d_model {
                Some(candle_nn::linear(d, sc.decoder_d_model, dv.pp("adapter"))?)
            } else {
                None
            };
            (
                adapter,
                Some(Block::load(
                    cfg,
                    sc.decoder_num_layers,
                    sc.num_patches,
                    sc.decoder_d_model,
                    dv.pp("decoder_block"),
                )?),
            )
        } else {
            (None, None)
        };
        let head_d = if cfg.use_decoder { sc.decoder_d_model } else { d };
        let head = candle_nn::linear(
            sc.num_patches * head_d,
            cfg.prediction_length,
            vb.pp("head").pp("base_forecast_block"),
        )?;
        let mq = if cfg.multi_quantile_head {
            Some(MqHead::load(cfg, head_d, sc.num_patches, vb.pp("multi_quantile_head_block"))?)
        } else {
            None
        };
        Ok(Self {
            sc,
            patcher,
            register,
            fft,
            pos,
            base_norm,
            encoder,
            adapter,
            decoder,
            head,
            mq,
            device: vb.device().clone(),
        })
    }

    fn forward(&self, series: &[f64]) -> Result<(Vec<f64>, Option<Vec<Vec<f64>>>)> {
        let sl = self.sc.context_length;
        let mut vals = vec![0f64; sl];
        let mut obs = vec![false; sl];
        if series.len() >= sl {
            vals.copy_from_slice(&series[series.len() - sl..]);
            obs.iter_mut().for_each(|o| *o = true);
        } else {
            let off = sl - series.len();
            vals[off..].copy_from_slice(series);
            for o in obs[off..].iter_mut() {
                *o = true;
            }
        }
        let (scaled, loc, scale) = self.sc.scaling.apply(&vals, &obs);
        let pl = self.sc.patch_length;
        let n = self.sc.n_base_patches;
        let start = sl - n * pl;
        let body: Vec<f64> = scaled[start..].to_vec();
        let d = self.sc.d_model;
        let mut tokens: Vec<Tensor> = Vec::new();
        match &self.patcher {
            Patcher::Single(lin) => {
                let t = Tensor::from_vec(body.iter().map(|x| *x as f32).collect::<Vec<f32>>(), (1, 1, n, pl), &self.device)?;
                tokens.push(t.apply(lin)?);
            }
            Patcher::Multi(projs) => {
                let s = body.len();
                let mut outs: Vec<Tensor> = Vec::new();
                let mut i = 0u32;
                loop {
                    let factor = 1usize << i;
                    if s < factor || s / factor < pl {
                        break;
                    }
                    let ds: Vec<f64> = (0..s / factor)
                        .map(|w| body[w * factor..(w + 1) * factor].iter().sum::<f64>() / factor as f64)
                        .collect();
                    let np = ds.len() / pl;
                    if np == 0 {
                        break;
                    }
                    let tail: Vec<f32> = ds[ds.len() - np * pl..].iter().map(|x| *x as f32).collect();
                    let t = Tensor::from_vec(tail, (1, 1, np, pl), &self.device)?;
                    let proj = projs
                        .get(i as usize)
                        .ok_or_else(|| candle_core::Error::Msg("TTM multi-scale projector 부족".into()))?;
                    outs.push(t.apply(proj)?);
                    i += 1;
                }
                outs.reverse();
                tokens.push(Tensor::cat(&outs, 2)?);
            }
        }
        let mut x = tokens.remove(0);
        if let Some(r) = &self.register {
            x = Tensor::cat(&[&r.unsqueeze(0)?.unsqueeze(0)?, &x], 2)?;
        }
        if let Some(f) = &self.fft {
            let raw: Vec<f64> = scaled[..sl].to_vec();
            let bins = f.max_bins;
            let mut mag: Vec<(usize, f64)> = (0..bins)
                .map(|kf| {
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for (tt, v) in raw.iter().enumerate() {
                        let a = -2.0 * std::f64::consts::PI * (kf * tt) as f64 / sl as f64;
                        re += v * a.cos();
                        im += v * a.sin();
                    }
                    let m = (re * re + im * im).sqrt();
                    (kf, if f.ignore_dc && kf == 0 { 0.0 } else { m })
                })
                .collect();
            mag.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
            let idx: Vec<u32> = mag.iter().take(f.k).map(|(i, _)| *i as u32).collect();
            let emb = match &f.embed {
                FftEmbed::Table(tab) => tab.index_select(&Tensor::from_vec(idx.clone(), idx.len(), &self.device)?, 0)?,
                FftEmbed::Mlp(l1, l2) => {
                    let v: Vec<f32> = idx.iter().map(|i| *i as f32 / bins as f32).collect();
                    Tensor::from_vec(v, (idx.len(), 1), &self.device)?.apply(l1)?.relu()?.apply(l2)?
                }
            };
            let tok = match &f.one {
                Some((proj, df)) => emb.reshape((1, f.k * df))?.apply(proj)?,
                None => emb,
            };
            x = Tensor::cat(&[&tok.unsqueeze(0)?.unsqueeze(0)?, &x], 2)?;
        }
        if let Some(p) = &self.pos {
            x = x.broadcast_add(p)?;
        }
        let p_total = x.dim(2)?;
        if p_total != self.sc.num_patches {
            return Err(candle_core::Error::Msg(format!(
                "TTM 패치 수 불일치: {} != {}",
                p_total, self.sc.num_patches
            )));
        }
        if let Some(bn) = &self.base_norm {
            x = x.reshape((1, 1, p_total * d))?.apply(bn)?.reshape((1, 1, p_total, d))?;
        }
        let mut h = self.encoder.forward(&x)?;
        if let Some(a) = &self.adapter {
            h = h.apply(a)?;
        }
        if let Some(dec) = &self.decoder {
            h = dec.forward(&h)?;
        }
        let dd = h.dim(3)?;
        let flat = h.reshape((1, 1, p_total * dd))?;
        let y = flat.apply(&self.head)?.flatten_all()?.to_vec1::<f32>()?;
        let y: Vec<f64> = y.into_iter().map(|v| v as f64).collect();
        let (point, quant) = match &self.mq {
            Some(mq) => {
                let dec2 = h.i((0, 0))?;
                let q = mq.forward(&y, &dec2)?;
                let med = q[q.len() / 2].clone();
                (med, Some(q))
            }
            None => (y, None),
        };
        let inv = |v: &Vec<f64>| -> Vec<f64> { v.iter().map(|a| a * scale + loc).collect() };
        Ok((inv(&point), quant.map(|q| q.iter().map(|r| inv(r)).collect())))
    }
}

fn sincos_pe(n: usize, d: usize, device: &Device) -> Result<Tensor> {
    let mut pe = vec![0f64; n * d];
    for p in 0..n {
        for i in (0..d).step_by(2) {
            let div = (-(10000f64.ln()) * i as f64 / d as f64).exp();
            pe[p * d + i] = (p as f64 * div).sin();
            if i + 1 < d {
                pe[p * d + i + 1] = (p as f64 * div).cos();
            }
        }
    }
    let mean = pe.iter().sum::<f64>() / pe.len() as f64;
    let var = pe.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (pe.len() as f64 - 1.0).max(1.0);
    let sd = var.sqrt() * 10.0;
    let v: Vec<f32> = pe.iter().map(|x| ((x - mean) / sd) as f32).collect();
    Tensor::from_vec(v, (n, d), device)
}

fn lower_median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s[(s.len() - 1) / 2]
}

pub fn robust_lowess_like(x: &[f64], frac: f64, iters: usize) -> Vec<f64> {
    let t = x.len();
    if t < 3 {
        return x.to_vec();
    }
    let eps = 1e-8;
    let mut w = ((frac * t as f64).round() as usize).max(5);
    if w % 2 == 0 {
        w += 1;
    }
    let cap = if t % 2 == 1 { t } else { t - 1 };
    w = w.min(cap);
    let r = (w - 1) / 2;
    let sigma = 0.25 * w as f64;
    let mut k: Vec<f64> = (0..w).map(|i| (-0.5 * ((i as f64 - r as f64) / sigma).powi(2)).exp()).collect();
    let ks: f64 = k.iter().sum::<f64>() + 1e-12;
    for v in k.iter_mut() {
        *v /= ks;
    }
    let reflect = |src: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(t + 2 * r);
        for i in (1..=r).rev() {
            out.push(src[i]);
        }
        out.extend_from_slice(src);
        for i in 0..r {
            out.push(src[t - 2 - i]);
        }
        out
    };
    let conv = |y: &[f64], wts: Option<&[f64]>| -> Vec<f64> {
        let ones = vec![1.0; t];
        let wv: &[f64] = wts.unwrap_or(&ones);
        let yw: Vec<f64> = y.iter().zip(wv).map(|(a, b)| a * b).collect();
        let yp = reflect(&yw);
        let wp = reflect(wv);
        (0..t)
            .map(|i| {
                let mut num = 0.0;
                let mut den = 0.0;
                for j in 0..w {
                    num += k[j] * yp[i + j];
                    den += k[j] * wp[i + j];
                }
                num / (den + eps)
            })
            .collect()
    };
    let xm = x.iter().sum::<f64>() / t as f64;
    let recenter = |tr: Vec<f64>| -> Vec<f64> {
        let tm = tr.iter().sum::<f64>() / t as f64;
        tr.into_iter().map(|v| v - tm + xm).collect()
    };
    let mut trend = recenter(conv(x, None));
    for _ in 0..iters {
        let resid: Vec<f64> = x.iter().zip(trend.iter()).map(|(a, b)| a - b).collect();
        let med = lower_median(&resid);
        let rc: Vec<f64> = resid.iter().map(|v| v - med).collect();
        let abs: Vec<f64> = rc.iter().map(|v| v.abs()).collect();
        let mad = lower_median(&abs) + eps;
        let wts: Vec<f64> = rc
            .iter()
            .map(|v| {
                let u = v / (6.0 * mad);
                let c = (u * u).clamp(0.0, 1.0);
                (1.0 - c) * (1.0 - c)
            })
            .collect();
        trend = recenter(conv(x, Some(&wts)));
    }
    trend
}

#[derive(Debug, Clone)]
enum Kind {
    Plain(Predictor),
    Decomposed {
        trend: Predictor,
        resid: Predictor,
        scaling: Scaling,
        residual_context: Option<usize>,
        via_variance: bool,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TtmOutput {
    pub point: Vec<f64>,
    pub quantile_levels: Vec<f64>,
    pub quantiles: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone)]
pub struct TtmForecaster {
    pub config: TtmConfig,
    kind: Kind,
    pub dir: PathBuf,
}

impl TtmForecaster {
    pub fn validate(cfg: &TtmConfig) -> std::result::Result<(), String> {
        if cfg.mode != "common_channel" || cfg.decoder_mode != "common_channel" {
            return Err("TTM: mix_channel 모드는 지원하지 않습니다".into());
        }
        if cfg.self_attn {
            return Err("TTM: self_attn 은 지원하지 않습니다".into());
        }
        if cfg.adaptive_patching_levels > 0 || cfg.decoder_adaptive_patching_levels > 0 {
            return Err("TTM: adaptive patching 은 지원하지 않습니다".into());
        }
        if cfg.resolution_prefix_tuning {
            return Err("TTM: resolution_prefix_tuning 은 지원하지 않습니다".into());
        }
        if cfg.decoder_raw_residual {
            return Err("TTM: decoder_raw_residual 은 지원하지 않습니다".into());
        }
        if cfg.enable_forecast_channel_mixing || cfg.categorical_vocab_size_list.is_some() {
            return Err("TTM: forecast channel mixing / categorical 입력은 지원하지 않습니다".into());
        }
        if cfg.use_register_context_gating {
            return Err("TTM: register context gating 은 지원하지 않습니다".into());
        }
        if cfg.multi_quantile_head {
            let q = &cfg.quantile_levels;
            if q.len() % 2 != 1 || (q[q.len() / 2] - 0.5).abs() > 1e-9 {
                return Err("TTM: quantile_levels 는 0.5 를 중심으로 한 홀수 개여야 합니다".into());
            }
        }
        Ok(())
    }

    pub fn from_varbuilder(cfg: TtmConfig, vb: VarBuilder, dir: PathBuf) -> std::result::Result<Self, String> {
        Self::validate(&cfg)?;
        let kind = if cfg.decompose {
            let tsc = SubConfig::derive(&cfg, "trend")?;
            let rsc = SubConfig::derive(&cfg, "residual")?;
            Kind::Decomposed {
                trend: Predictor::load(&cfg, tsc, vb.pp("trend_forecaster")).map_err(|e| format!("TTM trend 로드 실패: {}", e))?,
                resid: Predictor::load(&cfg, rsc, vb.pp("residual_forecaster")).map_err(|e| format!("TTM residual 로드 실패: {}", e))?,
                scaling: Scaling::from_value(&cfg.scaling),
                residual_context: cfg.residual_context_length,
                via_variance: cfg.combine_quantiles_via_variance,
            }
        } else {
            let sc = SubConfig::derive(&cfg, "plain")?;
            Kind::Plain(Predictor::load(&cfg, sc, vb).map_err(|e| format!("TTM 로드 실패: {}", e))?)
        };
        Ok(Self { config: cfg, kind, dir })
    }

    pub fn load(dir: &Path, device: &Device) -> std::result::Result<Self, String> {
        let v = crate::io::read_json(&dir.join("config.json"))?;
        let cfg: TtmConfig = serde_json::from_value(v).map_err(|e| format!("TTM config: {}", e))?;
        let vb = crate::io::load_varbuilder(dir, device)?;
        Self::from_varbuilder(cfg, vb, dir.to_path_buf())
    }

    pub fn context_length(&self) -> usize {
        self.config.context_length
    }

    pub fn prediction_length(&self) -> usize {
        self.config.prediction_length
    }

    pub fn min_history(&self) -> usize {
        8
    }

    pub fn forecast(&self, series: &[f64]) -> std::result::Result<TtmOutput, String> {
        let out = self.forecast_inner(series);
        out
    }

    fn forecast_inner(&self, series: &[f64]) -> std::result::Result<TtmOutput, String> {
        if series.len() < self.min_history() {
            return Err(format!("TTM 입력이 너무 짧습니다 ({} < {})", series.len(), self.min_history()));
        }
        if series.iter().any(|v| !v.is_finite()) {
            return Err("TTM 입력에 NaN/Inf 가 있습니다".into());
        }
        let levels = self.config.quantile_levels.clone();
        match &self.kind {
            Kind::Plain(p) => {
                let (point, q) = p.forward(series).map_err(|e| e.to_string())?;
                Ok(TtmOutput {
                    point,
                    quantile_levels: levels,
                    quantiles: q,
                })
            }
            Kind::Decomposed {
                trend,
                resid,
                scaling,
                residual_context,
                via_variance,
            } => {
                let ctx = self.config.context_length.max(1);
                let mut past = vec![0f64; ctx];
                let mut obs = vec![false; ctx];
                if series.len() >= ctx {
                    past.copy_from_slice(&series[series.len() - ctx..]);
                    obs.iter_mut().for_each(|o| *o = true);
                } else {
                    let pad = ctx - series.len();
                    past[pad..].copy_from_slice(series);
                    obs[pad..].iter_mut().for_each(|o| *o = true);
                }
                let (scaled, loc, scale) = scaling.apply(&past, &obs);
                let tr = robust_lowess_like(&scaled, 0.15, 2);
                let l_res = residual_context.map(|r| r.min(ctx)).unwrap_or(ctx);
                let off = ctx - l_res;
                let rsig: Vec<f64> = (off..ctx).map(|i| scaled[i] - tr[i]).collect();
                let (tp, tq) = trend.forward(&scaled).map_err(|e| e.to_string())?;
                let (rp, rq) = resid.forward(&rsig).map_err(|e| e.to_string())?;
                let point: Vec<f64> = tp.iter().zip(rp.iter()).map(|(a, b)| (a + b) * scale + loc).collect();
                let quantiles = match (tq, rq) {
                    (Some(tq), Some(rq)) => {
                        let qn = tq.len();
                        let k = qn / 2;
                        let t = tq[0].len();
                        let mut out = vec![vec![0f64; t]; qn];
                        for i in 0..t {
                            let t50 = tq[k][i];
                            let r50 = rq[k][i];
                            for qi in 0..qn {
                                let v = if *via_variance {
                                    if qi == k {
                                        t50 + r50
                                    } else {
                                        let tw = tq[qi][i] - t50;
                                        let rw = rq[qi][i] - r50;
                                        let mag = (tw * tw + rw * rw + 1e-8).sqrt();
                                        let sgn = (tw + rw).signum() * if tw + rw == 0.0 { 0.0 } else { 1.0 };
                                        t50 + r50 + sgn * mag
                                    }
                                } else {
                                    tq[qi][i] + rq[qi][i]
                                };
                                out[qi][i] = v * scale + loc;
                            }
                        }
                        Some(out)
                    }
                    _ => None,
                };
                Ok(TtmOutput {
                    point,
                    quantile_levels: levels,
                    quantiles,
                })
            }
        }
    }
}