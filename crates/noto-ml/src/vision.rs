//! SigLIP2 vision tower with NaFlex patches and a multihead attention-pooling
//! (MAP) head that turns the patch tokens into one image embedding.
//!
//! The learned position table is a 16x16 grid; a NaFlex image of another
//! grid gets it resized. Hugging Face's `Siglip2VisionEmbeddings.
//! resize_positional_embeddings` does that with `F.interpolate(mode=
//! "bilinear", align_corners=False, antialias=True)`, so a 13-row grid (a
//! 384x256 keyframe) averages the 16 table rows instead of sampling them.
//! [`VisionTower::set_position_antialias`] chooses between that reference
//! filter (the default) and the plain bilinear resize this port used before.

use candle_core::{Device, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::config::Siglip2Config;
use crate::encoder::{EncoderLayer, Mlp};
use crate::preprocess::NaflexInput;

/// Learned pooling query that attends over all patch tokens.
pub struct MapHead {
    probe: Tensor,
    in_w: Tensor,
    in_b: Tensor,
    out: Linear,
    ln: LayerNorm,
    mlp: Mlp,
    heads: usize,
    hidden: usize,
}

impl MapHead {
    fn new(cfg: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.vision_hidden;
        let attn = vb.pp("attention");
        Ok(MapHead {
            probe: vb.get((1, 1, d), "probe")?,
            in_w: attn.get((3 * d, d), "in_proj_weight")?,
            in_b: attn.get(3 * d, "in_proj_bias")?,
            out: linear(d, d, attn.pp("out_proj"))?,
            ln: layer_norm(d, cfg.vision_eps, vb.pp("layernorm"))?,
            mlp: Mlp::new(d, cfg.vision_intermediate, vb.pp("mlp"))?,
            heads: cfg.vision_heads,
            hidden: d,
        })
    }

    /// `tokens` is `(1, n, hidden)`; returns `(1, hidden)`.
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let d = self.hidden;
        let (b, n, _) = tokens.dims3()?;
        let hd = d / self.heads;
        let w = |i: usize| self.in_w.narrow(0, i * d, d);
        let bias = |i: usize| self.in_b.narrow(0, i * d, d);

        let probe = self.probe.broadcast_as((b, 1, d))?;
        let q = probe
            .broadcast_matmul(&w(0)?.t()?.contiguous()?)?
            .broadcast_add(&bias(0)?)?;
        let k = tokens
            .broadcast_matmul(&w(1)?.t()?.contiguous()?)?
            .broadcast_add(&bias(1)?)?;
        let v = tokens
            .broadcast_matmul(&w(2)?.t()?.contiguous()?)?
            .broadcast_add(&bias(2)?)?;

        let split = |t: Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.heads, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = split(q, 1)?;
        let k = split(k, n)?;
        let v = split(v, n)?;
        let scores = q
            .matmul(&k.transpose(2, 3)?)?
            .affine((hd as f64).powf(-0.5), 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, 1, d))?;

        let attended = self.out.forward(&ctx)?;
        let normed = self.ln.forward(&attended)?;
        let pooled = (&attended + self.mlp.forward(&normed)?)?;
        pooled.reshape((b, d))
    }

    /// Every patch token mapped into the image-text space on its own:
    /// the value projection, output projection and MLP of the head, as if the
    /// probe attended to that one token (the MaskCLIP trick for an attention
    /// pool). `tokens` is `(b, n, hidden)`; returns `(b, n, hidden)`.
    fn dense(&self, tokens: &Tensor) -> Result<Tensor> {
        let d = self.hidden;
        let v = tokens
            .broadcast_matmul(&self.in_w.narrow(0, 2 * d, d)?.t()?.contiguous()?)?
            .broadcast_add(&self.in_b.narrow(0, 2 * d, d)?)?;
        let attended = self.out.forward(&v)?;
        let normed = self.ln.forward(&attended)?;
        &attended + self.mlp.forward(&normed)?
    }

    /// The probe's attention over the patch tokens, averaged over heads:
    /// `(b, n)`, each row sums to one.
    fn attention(&self, tokens: &Tensor) -> Result<Tensor> {
        let d = self.hidden;
        let (b, n, _) = tokens.dims3()?;
        let hd = d / self.heads;
        let probe = self.probe.broadcast_as((b, 1, d))?;
        let q = probe
            .broadcast_matmul(&self.in_w.narrow(0, 0, d)?.t()?.contiguous()?)?
            .broadcast_add(&self.in_b.narrow(0, 0, d)?)?;
        let k = tokens
            .broadcast_matmul(&self.in_w.narrow(0, d, d)?.t()?.contiguous()?)?
            .broadcast_add(&self.in_b.narrow(0, d, d)?)?;
        let q = q.reshape((b, 1, self.heads, hd))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, n, self.heads, hd))?.transpose(1, 2)?.contiguous()?;
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine((hd as f64).powf(-0.5), 0.0)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs.mean(1)?.reshape((b, n))
    }
}

pub struct VisionTower {
    patch_proj: Linear,
    /// Learned positions, `(max_patches, hidden)`, laid out as a square grid.
    position: Tensor,
    layers: Vec<EncoderLayer>,
    post_ln: LayerNorm,
    head: MapHead,
    side: usize,
    hidden: usize,
    device: Device,
    /// Resize the position table with the antialiased filter of the reference.
    pos_antialias: bool,
}

/// Weights of PyTorch's antialiased bilinear resize along one axis
/// (`_compute_indices_min_size_weights_aa` with `align_corners=False`), as a
/// row-major `dst x src` matrix. Upsampling reduces to ordinary bilinear with
/// half-pixel centres; downsampling widens the triangle to the scale factor so
/// every input sample contributes.
pub fn antialias_bilinear_weights(src: usize, dst: usize) -> Vec<f32> {
    let scale = src as f64 / dst as f64;
    let (support, inv_scale) = if scale >= 1.0 { (scale, 1.0 / scale) } else { (1.0, 1.0) };
    let triangle = |x: f64| (1.0 - x.abs()).max(0.0);
    let mut weights = vec![0f32; dst * src];
    for o in 0..dst {
        let center = scale * (o as f64 + 0.5);
        // `as i64` truncates toward zero like the C++ `static_cast<int64_t>`.
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as i64).min(src as i64).max(lo as i64) as usize;
        let taps: Vec<f64> = (lo..hi).map(|j| triangle((j as f64 - center + 0.5) * inv_scale)).collect();
        let total: f64 = taps.iter().sum();
        if total > 0.0 {
            for (k, t) in taps.iter().enumerate() {
                weights[o * src + lo + k] = (t / total) as f32;
            }
        }
    }
    weights
}

impl VisionTower {
    /// `vb` must point at the `vision_model` prefix of the checkpoint.
    pub fn new(cfg: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.vision_hidden;
        let emb = vb.pp("embeddings");
        let patch_dim = cfg.patch_size * cfg.patch_size * 3;
        let mut layers = Vec::with_capacity(cfg.vision_layers);
        for i in 0..cfg.vision_layers {
            layers.push(EncoderLayer::new(
                d,
                cfg.vision_intermediate,
                cfg.vision_heads,
                cfg.vision_eps,
                vb.pp("encoder").pp(format!("layers.{i}")),
            )?);
        }
        Ok(VisionTower {
            patch_proj: linear(patch_dim, d, emb.pp("patch_embedding"))?,
            position: emb
                .pp("position_embedding")
                .get((cfg.max_patches, d), "weight")?,
            layers,
            post_ln: layer_norm(d, cfg.vision_eps, vb.pp("post_layernorm"))?,
            head: MapHead::new(cfg, vb.pp("head"))?,
            side: cfg.position_side(),
            hidden: d,
            device: vb.device().clone(),
            pos_antialias: true,
        })
    }

    /// `true` (default): the reference's antialiased resize of the position
    /// table; `false`: plain bilinear (this port before v1.0).
    pub fn set_position_antialias(&mut self, on: bool) {
        self.pos_antialias = on;
    }

    pub fn position_antialias(&self) -> bool {
        self.pos_antialias
    }

    /// Positions for a `grid_h x grid_w` patch grid. The square table is
    /// resized when the grid is not square-shaped like the table.
    fn positions(&self, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let d = self.hidden;
        let table = self.position.reshape((self.side, self.side, d))?;
        if grid_h == self.side && grid_w == self.side {
            return table.reshape((1, grid_h * grid_w, d));
        }
        if self.pos_antialias {
            // Separable: rows, then columns, as two small matrix products.
            let side = self.side;
            let wy = Tensor::from_vec(antialias_bilinear_weights(side, grid_h), (grid_h, side), &self.device)?;
            let wx = Tensor::from_vec(antialias_bilinear_weights(side, grid_w), (grid_w, side), &self.device)?;
            let by_rows = wy.matmul(&table.reshape((side, side * d))?)?.reshape((grid_h, side, d))?;
            return wx.broadcast_matmul(&by_rows)?.reshape((1, grid_h * grid_w, d));
        }
        table
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .upsample_bilinear2d(grid_h, grid_w, false)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .reshape((1, grid_h * grid_w, d))
    }

    /// Patch tokens after the encoder and the final layer norm, `(1, n, hidden)`.
    fn tokens(&self, input: &NaflexInput) -> Result<Tensor> {
        let (gh, gw, pd) = (input.grid_h, input.grid_w, input.patch_dim);
        let patches = Tensor::from_slice(&input.patches, (1, gh * gw, pd), &self.device)?;
        let mut x = self.patch_proj.forward(&patches)?;
        x = x.broadcast_add(&self.positions(gh, gw)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, None)?;
        }
        self.post_ln.forward(&x)
    }

    /// Dense output for one image: per-patch embeddings in the image-text
    /// space `(n, hidden)` before normalization, the pooling head's attention
    /// over the patches `(n,)`, and the pooled image embedding `(hidden,)`.
    /// Patches are row-major on the `grid_h x grid_w` grid of `input`.
    pub fn forward_dense(&self, input: &NaflexInput) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.tokens(input)?;
        let dense = self.head.dense(&x)?.squeeze(0)?;
        let attn = self.head.attention(&x)?.squeeze(0)?;
        let pooled = self.head.forward(&x)?.squeeze(0)?;
        Ok((dense, attn, pooled))
    }

    /// Image embedding `(1, hidden)` before normalization.
    pub fn forward(&self, input: &NaflexInput) -> Result<Tensor> {
        self.forward_batch(&[input])
    }

    /// [`VisionTower::forward_dense`] for images that share one patch grid:
    /// dense `(b, n, hidden)`, attention `(b, n)`, pooled `(b, hidden)`.
    pub fn forward_dense_batch(&self, inputs: &[&NaflexInput]) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.batch_tokens(inputs)?;
        Ok((self.head.dense(&x)?, self.head.attention(&x)?, self.head.forward(&x)?))
    }

    /// Patch tokens `(b, n, hidden)` after the encoder and the final norm.
    fn batch_tokens(&self, inputs: &[&NaflexInput]) -> Result<Tensor> {
        let first = match inputs.first() {
            Some(f) => f,
            None => candle_core::bail!("a vision batch needs at least one image"),
        };
        let (gh, gw, pd) = (first.grid_h, first.grid_w, first.patch_dim);
        let n = gh * gw;
        let mut flat: Vec<f32> = Vec::with_capacity(inputs.len() * n * pd);
        for i in inputs {
            if (i.grid_h, i.grid_w, i.patch_dim) != (gh, gw, pd) {
                candle_core::bail!("batched images must share one patch grid ({gh}x{gw}), got {}x{}", i.grid_h, i.grid_w);
            }
            flat.extend_from_slice(&i.patches);
        }
        let patches = Tensor::from_vec(flat, (inputs.len(), n, pd), &self.device)?;
        let mut x = self.patch_proj.forward(&patches)?;
        x = x.broadcast_add(&self.positions(gh, gw)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, None)?;
        }
        self.post_ln.forward(&x)
    }

    /// Embeddings `(batch, hidden)` before normalization for images that share
    /// one patch grid. Rows are independent: attention never mixes images.
    pub fn forward_batch(&self, inputs: &[&NaflexInput]) -> Result<Tensor> {
        self.head.forward(&self.batch_tokens(inputs)?)
    }
}
