use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::config::Siglip2Config;
use crate::encoder::EncoderLayer;

pub struct TextTower {
    tokens: Embedding,
    position: Tensor,
    layers: Vec<EncoderLayer>,
    final_ln: LayerNorm,
    head: Linear,
    max_len: usize,
    hidden: usize,
    device: Device,
}

impl TextTower {
    pub fn new(cfg: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.text_hidden;
        let emb = vb.pp("embeddings");
        let mut layers = Vec::with_capacity(cfg.text_layers);
        for i in 0..cfg.text_layers {
            layers.push(EncoderLayer::new(
                d,
                cfg.text_intermediate,
                cfg.text_heads,
                cfg.text_eps,
                vb.pp("encoder").pp(format!("layers.{i}")),
            )?);
        }
        Ok(TextTower {
            tokens: embedding(cfg.text_vocab, d, emb.pp("token_embedding"))?,
            position: emb
                .pp("position_embedding")
                .get((cfg.text_max_len, d), "weight")?,
            layers,
            final_ln: layer_norm(d, cfg.text_eps, vb.pp("final_layer_norm"))?,
            head: linear(d, d, vb.pp("head"))?,
            max_len: cfg.text_max_len,
            hidden: d,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, len) = ids.dims2()?;
        let pos = self.position.narrow(0, 0, len)?.unsqueeze(0)?;
        let mut x = self.tokens.forward(ids)?.broadcast_add(&pos)?;

        let additive = mask
            .affine(-1.0, 1.0)?
            .affine(-1e9, 0.0)?
            .reshape((b, 1, 1, len))?;
        for layer in &self.layers {
            x = layer.forward(&x, Some(&additive))?;
        }
        let x = self.final_ln.forward(&x)?;
        let pooled = x.narrow(1, len - 1, 1)?.reshape((b, self.hidden))?;
        self.head.forward(&pooled)
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
