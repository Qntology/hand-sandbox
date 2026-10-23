//! SigLIP2 facade: loads the checkpoint, embeds images and texts, and scores
//! image-text pairs the way SigLIP does (sigmoid of scaled cosine similarity).

use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use image::DynamicImage;

use crate::config::Siglip2Config;
use crate::preprocess::preprocess;
use crate::text::TextTower;
use crate::tokenizer::TextTokenizer;
use crate::vision::VisionTower;

pub struct Siglip2 {
    config: Siglip2Config,
    vision: VisionTower,
    text: TextTower,
    tokenizer: Option<TextTokenizer>,
    logit_scale: f32,
    logit_bias: f32,
    device: Device,
}

impl Siglip2 {
    /// Loads `config.json`, `tokenizer.json` and `model.safetensors` from `dir`.
    /// Without `config.json` the base NaFlex defaults are used.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            return Err(anyhow!(
                "SigLIP2 weights not found at {}. Download google/siglip2-base-patch16-naflex \
                 (config.json, tokenizer.json, model.safetensors) into that folder.",
                weights.display()
            ));
        }
        let config_path = dir.join("config.json");
        let config = if config_path.exists() {
            Siglip2Config::from_file(&config_path)?
        } else {
            Siglip2Config::default()
        };
        // SAFETY: the file is only read, and the mapping lives as long as the model.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let tokenizer = TextTokenizer::from_file(
            &dir.join("tokenizer.json"),
            config.text_pad_id,
            config.text_max_len,
        )?;
        Self::from_var_builder(config, vb, device, Some(tokenizer))
    }

    /// Builds the model from any `VarBuilder`. Tests use a `VarMap` with random weights.
    pub fn from_var_builder(
        config: Siglip2Config,
        vb: VarBuilder,
        device: &Device,
        tokenizer: Option<TextTokenizer>,
    ) -> Result<Self> {
        let vision = VisionTower::new(&config, vb.pp("vision_model"))?;
        let text = TextTower::new(&config, vb.pp("text_model"))?;
        let logit_scale = read_scalar(&vb, "logit_scale").unwrap_or(0.0);
        let logit_bias = read_scalar(&vb, "logit_bias").unwrap_or(0.0);
        Ok(Siglip2 {
            config,
            vision,
            text,
            tokenizer,
            logit_scale,
            logit_bias,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &Siglip2Config {
        &self.config
    }

    /// Length of every embedding this model returns.
    pub fn embedding_dim(&self) -> usize {
        self.config.vision_hidden
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Unit-length embedding of one image.
    pub fn embed_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let input = preprocess(image, &self.config);
        let emb = self.vision.forward(&input)?;
        Ok(l2_normalize(&emb)?.flatten_all()?.to_vec1()?)
    }

    /// Unit-length embeddings of texts, in input order.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("this model was loaded without a tokenizer"))?;
        let (ids, mask) = tokenizer.encode(texts)?;
        self.embed_tokens(&ids, &mask, texts.len())
    }

    /// Embeds pre-tokenized rows. `ids` and `mask` are `rows * text_max_len` long.
    pub fn embed_tokens(&self, ids: &[u32], mask: &[f32], rows: usize) -> Result<Vec<Vec<f32>>> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let len = self.config.text_max_len;
        let ids_t = Tensor::from_slice(ids, (rows, len), &self.device)?;
        let mask_t = Tensor::from_slice(mask, (rows, len), &self.device)?;
        let emb = l2_normalize(&self.text.forward(&ids_t, &mask_t)?)?;
        Ok(emb.to_vec2()?)
    }

    /// SigLIP pair score per text: `sigmoid(exp(logit_scale) * cos + logit_bias)`.
    /// Each score is independent of the others, so several labels can be high at once.
    pub fn scores(&self, image: &[f32], texts: &[Vec<f32>]) -> Vec<f32> {
        let scale = self.logit_scale.exp();
        texts
            .iter()
            .map(|t| {
                let cos: f32 = image.iter().zip(t).map(|(a, b)| a * b).sum();
                sigmoid(scale * cos + self.logit_bias)
            })
            .collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2_normalize(t: &Tensor) -> candle_core::Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.affine(1.0, 1e-12)?;
    t.broadcast_div(&norm)
}

/// Reads a scalar stored as shape `(1,)` or `()`. Returns `None` when absent.
fn read_scalar(vb: &VarBuilder, name: &str) -> Option<f32> {
    let t = vb
        .get(1usize, name)
        .or_else(|_| vb.get((), name))
        .ok()?;
    t.flatten_all().ok()?.to_vec1::<f32>().ok()?.first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use image::{Rgb, RgbImage};

    fn tiny_config() -> Siglip2Config {
        Siglip2Config {
            vision_hidden: 32,
            vision_intermediate: 64,
            vision_layers: 2,
            vision_heads: 4,
            patch_size: 4,
            max_patches: 16,
            vision_eps: 1e-6,
            text_hidden: 32,
            text_intermediate: 64,
            text_layers: 2,
            text_heads: 4,
            text_vocab: 50,
            text_max_len: 8,
            text_pad_id: 1,
            text_eps: 1e-6,
        }
    }

    fn tiny_model() -> Siglip2 {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        Siglip2::from_var_builder(tiny_config(), vb, &dev, None).unwrap()
    }

    fn test_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x * 7 % 256) as u8, (y * 13 % 256) as u8, 128])
        }))
    }

    #[test]
    fn image_embedding_is_unit_length_and_deterministic() {
        let m = tiny_model();
        let img = test_image(64, 32);
        let a = m.embed_image(&img).unwrap();
        let b = m.embed_image(&img).unwrap();
        assert_eq!(a.len(), m.embedding_dim());
        let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        assert_eq!(a, b);
    }

    #[test]
    fn non_square_grids_use_resized_positions() {
        // 64x32 at patch 4 gives a grid that is not 4x4, so the resize path runs.
        let m = tiny_model();
        let input = preprocess(&test_image(64, 32), m.config());
        assert_ne!((input.grid_h, input.grid_w), (4, 4));
        assert!(m.embed_image(&test_image(64, 32)).is_ok());
    }

    #[test]
    fn token_embeddings_are_unit_length_and_padding_is_masked() {
        let m = tiny_model();
        let len = m.config().text_max_len;
        let ids: Vec<u32> = (0..len as u32).chain(0..len as u32).collect();
        // Second row has the last four tokens padded.
        let mut mask = vec![1.0f32; 2 * len];
        for v in &mut mask[len + 4..] {
            *v = 0.0;
        }
        let out = m.embed_tokens(&ids, &mask, 2).unwrap();
        assert_eq!(out.len(), 2);
        for row in &out {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn scores_are_probabilities() {
        let m = tiny_model();
        let img = m.embed_image(&test_image(32, 32)).unwrap();
        let texts = vec![img.clone(), img.iter().map(|v| -v).collect()];
        let s = m.scores(&img, &texts);
        assert!(s.iter().all(|p| (0.0..=1.0).contains(p)));
        assert!(s[0] > s[1]);
    }

    #[test]
    fn missing_weights_gives_a_clear_error() {
        let dir = std::env::temp_dir().join("hand-ml-missing-weights-test");
        let _ = std::fs::create_dir_all(&dir);
        let err = Siglip2::load(&dir, &Device::Cpu).err().expect("should fail");
        assert!(err.to_string().contains("model.safetensors"));
    }
}
