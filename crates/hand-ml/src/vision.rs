//! SigLIP2 vision tower with NaFlex patches and a multihead attention-pooling
//! (MAP) head that turns the patch tokens into one image embedding.

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
        })
    }

    /// Positions for a `grid_h x grid_w` patch grid. The square table is
    /// bilinearly resized when the grid is not square-shaped like the table.
    fn positions(&self, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let d = self.hidden;
        let table = self.position.reshape((self.side, self.side, d))?;
        if grid_h == self.side && grid_w == self.side {
            return table.reshape((1, grid_h * grid_w, d));
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

    /// Image embedding `(1, hidden)` before normalization.
    pub fn forward(&self, input: &NaflexInput) -> Result<Tensor> {
        let n = input.grid_h * input.grid_w;
        let patches = Tensor::from_slice(&input.patches, (1, n, input.patch_dim), &self.device)?;
        let mut x = self.patch_proj.forward(&patches)?;
        x = x.broadcast_add(&self.positions(input.grid_h, input.grid_w)?)?;
        for layer in &self.layers {
            x = layer.forward(&x, None)?;
        }
        let x = self.post_ln.forward(&x)?;
        self.head.forward(&x)
    }
}
