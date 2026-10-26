//! SigLIP2 model configuration. Defaults match `google/siglip2-base-patch16-naflex`
//! and are overridden by `config.json` when the file has the fields.

use std::path::Path;

use anyhow::{Context, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct Siglip2Config {
    pub vision_hidden: usize,
    pub vision_intermediate: usize,
    pub vision_layers: usize,
    pub vision_heads: usize,
    pub patch_size: usize,
    /// Largest number of patches per image (NaFlex). Also the length of the
    /// learned position table, which must be a square.
    pub max_patches: usize,
    pub vision_eps: f64,
    pub text_hidden: usize,
    pub text_intermediate: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub text_vocab: usize,
    /// Text is padded or truncated to this many tokens.
    pub text_max_len: usize,
    pub text_pad_id: u32,
    pub text_eps: f64,
}

impl Default for Siglip2Config {
    fn default() -> Self {
        Siglip2Config {
            vision_hidden: 768,
            vision_intermediate: 3072,
            vision_layers: 12,
            vision_heads: 12,
            patch_size: 16,
            max_patches: 256,
            vision_eps: 1e-6,
            text_hidden: 768,
            text_intermediate: 3072,
            text_layers: 12,
            text_heads: 12,
            text_vocab: 256_000,
            text_max_len: 64,
            text_pad_id: 1,
            text_eps: 1e-6,
        }
    }
}

impl Siglip2Config {
    /// Reads a Hugging Face `config.json`. Missing fields keep their defaults.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)?;
        let d = Siglip2Config::default();
        let vc = &v["vision_config"];
        let tc = &v["text_config"];
        let us = |node: &serde_json::Value, key: &str, fallback: usize| -> usize {
            node[key].as_u64().map(|x| x as usize).unwrap_or(fallback)
        };
        let fl = |node: &serde_json::Value, key: &str, fallback: f64| -> f64 {
            node[key].as_f64().unwrap_or(fallback)
        };
        Ok(Siglip2Config {
            vision_hidden: us(vc, "hidden_size", d.vision_hidden),
            vision_intermediate: us(vc, "intermediate_size", d.vision_intermediate),
            vision_layers: us(vc, "num_hidden_layers", d.vision_layers),
            vision_heads: us(vc, "num_attention_heads", d.vision_heads),
            patch_size: us(vc, "patch_size", d.patch_size),
            max_patches: us(vc, "num_patches", d.max_patches),
            vision_eps: fl(vc, "layer_norm_eps", d.vision_eps),
            text_hidden: us(tc, "hidden_size", d.text_hidden),
            text_intermediate: us(tc, "intermediate_size", d.text_intermediate),
            text_layers: us(tc, "num_hidden_layers", d.text_layers),
            text_heads: us(tc, "num_attention_heads", d.text_heads),
            text_vocab: us(tc, "vocab_size", d.text_vocab),
            text_max_len: us(tc, "max_position_embeddings", d.text_max_len),
            text_pad_id: us(tc, "pad_token_id", d.text_pad_id as usize) as u32,
            text_eps: fl(tc, "layer_norm_eps", d.text_eps),
        })
    }

    /// Side of the square position table (256 -> 16).
    pub fn position_side(&self) -> usize {
        (self.max_patches as f64).sqrt().round().max(1.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_describe_base_naflex() {
        let c = Siglip2Config::default();
        assert_eq!(c.position_side(), 16);
        assert_eq!(c.text_max_len, 64);
    }

    #[test]
    fn json_overrides_only_present_fields() {
        let c = Siglip2Config::from_json(
            r#"{"vision_config": {"hidden_size": 1152, "num_hidden_layers": 27}}"#,
        )
        .unwrap();
        assert_eq!(c.vision_hidden, 1152);
        assert_eq!(c.vision_layers, 27);
        assert_eq!(c.text_hidden, 768);
    }
}
