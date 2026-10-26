//! SigLIP2 facade: loads the checkpoint, embeds images and texts, and scores
//! image-text pairs the way SigLIP does (sigmoid of scaled cosine similarity).

use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use image::DynamicImage;

use crate::config::Siglip2Config;
use crate::device::{device_family, trim_idle_pool, BatchPolicy};
use crate::parallel::{fnv1a, FNV_OFFSET};
use crate::preprocess::{preprocess, NaflexInput};
use crate::text::TextTower;
use crate::tokenizer::TextTokenizer;
use crate::vision::VisionTower;

/// Per-patch output of [`Siglip2::embed_dense`].
#[derive(Clone, Debug)]
pub struct DenseEmbedding {
    pub grid_h: usize,
    pub grid_w: usize,
    /// `grid_h * grid_w` unit vectors, row-major.
    pub vectors: Vec<Vec<f32>>,
    /// Pooling attention per patch (sums to one).
    pub attention: Vec<f32>,
    /// The whole-image embedding (unit length), same as [`Siglip2::embed_image`].
    pub pooled: Vec<f32>,
}

pub struct Siglip2 {
    config: Siglip2Config,
    vision: VisionTower,
    text: TextTower,
    tokenizer: Option<TextTokenizer>,
    logit_scale: f32,
    logit_bias: f32,
    device: Device,
    /// Rows per forward pass on this device.
    policy: BatchPolicy,
    /// Hash of the checkpoint alone (config, file sizes and times).
    checkpoint: u64,
    /// Identifies the checkpoint and the device family, so cached text
    /// vectors never cross models or backends.
    fingerprint: u64,
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
        let mut model = Self::from_var_builder(config, vb, device, Some(tokenizer))?;
        model.checkpoint = checkpoint_fingerprint(dir, &model.config);
        model.fingerprint = device_salt(model.checkpoint, device, &model.policy);
        Ok(model)
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
        let policy = BatchPolicy::for_device(device);
        let checkpoint = fnv1a(FNV_OFFSET, format!("{config:?}").as_bytes());
        let fingerprint = device_salt(checkpoint, device, &policy);
        Ok(Siglip2 {
            config,
            vision,
            text,
            tokenizer,
            logit_scale,
            logit_bias,
            device: device.clone(),
            policy,
            checkpoint,
            fingerprint,
        })
    }

    /// Rows per forward pass on this model's device.
    pub fn batch_policy(&self) -> BatchPolicy {
        self.policy
    }

    /// Replaces the batch policy (tests, or a caller with its own memory budget).
    /// The fingerprint follows, because fixed-shape text batches change the vectors.
    pub fn set_batch_policy(&mut self, policy: BatchPolicy) {
        self.fingerprint = device_salt(self.checkpoint, &self.device, &policy);
        self.policy = policy;
    }

    /// Tokens one image of this patch grid sends through the tower.
    fn tokens_of(input: &NaflexInput) -> usize {
        input.grid_h * input.grid_w
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

    /// Hash of the config, weight file size and modification time. Two
    /// checkpoints that differ in any of these never share cached vectors.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Image-text pair probability for a cosine similarity.
    pub fn pair_probability(&self, cosine: f32) -> f32 {
        sigmoid(self.logit_scale.exp() * cosine + self.logit_bias)
    }

    /// Unit-length embedding of one image.
    pub fn embed_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let input = preprocess(image, &self.config);
        let emb = self.vision.forward(&input)?;
        Ok(l2_normalize(&emb)?.flatten_all()?.to_vec1()?)
    }

    /// Unit-length embeddings of many images. Images are preprocessed in
    /// parallel, grouped by patch grid, and each group runs through the vision
    /// tower in micro-batches of at most `max_batch` images (`0`: as many as
    /// the device's [`BatchPolicy`] allows for that grid). Output order
    /// matches the input order.
    pub fn embed_images(&self, images: &[DynamicImage], max_batch: usize) -> Result<Vec<Vec<f32>>> {
        let inputs = self.preprocess_all(images);
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); images.len()];
        for chunk in self.image_chunks(&inputs, max_batch) {
            let refs: Vec<&NaflexInput> = chunk.iter().map(|&i| &inputs[i]).collect();
            let emb = l2_normalize(&self.vision.forward_batch(&refs)?)?;
            let rows: Vec<Vec<f32>> = emb.to_vec2()?;
            for (row, &i) in rows.into_iter().zip(&chunk) {
                out[i] = row;
            }
        }
        trim_idle_pool(&self.device);
        Ok(out)
    }

    fn preprocess_all(&self, images: &[DynamicImage]) -> Vec<NaflexInput> {
        use rayon::prelude::*;
        images.par_iter().map(|image| preprocess(image, &self.config)).collect()
    }

    /// Input indices grouped by patch grid and cut into forward passes.
    fn image_chunks(&self, inputs: &[NaflexInput], max_batch: usize) -> Vec<Vec<usize>> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (i, input) in inputs.iter().enumerate() {
            groups.entry((input.grid_h, input.grid_w)).or_default().push(i);
        }
        let mut out = Vec::new();
        for indices in groups.values() {
            let rows = match max_batch {
                0 => self.policy.image_rows(Self::tokens_of(&inputs[indices[0]])),
                n => n,
            };
            out.extend(indices.chunks(rows.max(1)).map(|c| c.to_vec()));
        }
        out
    }

    /// [`Siglip2::embed_dense`] for many images, batched like
    /// [`Siglip2::embed_images`]: images that share a patch grid go through the
    /// tower together, and each result is the same as the single-image call
    /// (rows never mix inside the tower). Output order matches the input.
    pub fn embed_dense_batch(&self, images: &[DynamicImage]) -> Result<Vec<DenseEmbedding>> {
        let inputs = self.preprocess_all(images);
        let mut out: Vec<Option<DenseEmbedding>> = (0..images.len()).map(|_| None).collect();
        for chunk in self.image_chunks(&inputs, 0) {
            let refs: Vec<&NaflexInput> = chunk.iter().map(|&i| &inputs[i]).collect();
            let (dense, attn, pooled) = self.vision.forward_dense_batch(&refs)?;
            let dense = l2_normalize(&dense)?;
            let pooled = l2_normalize(&pooled)?.to_vec2::<f32>()?;
            let attn = attn.to_vec2::<f32>()?;
            for (k, &i) in chunk.iter().enumerate() {
                out[i] = Some(DenseEmbedding {
                    grid_h: inputs[i].grid_h,
                    grid_w: inputs[i].grid_w,
                    vectors: dense.get(k)?.to_vec2()?,
                    attention: attn[k].clone(),
                    pooled: pooled[k].clone(),
                });
            }
        }
        trim_idle_pool(&self.device);
        Ok(out.into_iter().map(|d| d.expect("every input is in one chunk")).collect())
    }

    /// Dense embedding of one image: a unit vector per patch in the same
    /// space as the text embeddings, and where the pooling head looks. This is
    /// what semantic location uses: the cosine of each patch with a label's
    /// text vector is a heat map from one forward pass.
    pub fn embed_dense(&self, image: &DynamicImage) -> Result<DenseEmbedding> {
        let input = preprocess(image, &self.config);
        let (dense, attn, pooled) = self.vision.forward_dense(&input)?;
        let vectors = l2_normalize(&dense)?.to_vec2()?;
        let pooled = l2_normalize(&pooled.unsqueeze(0)?)?.flatten_all()?.to_vec1()?;
        Ok(DenseEmbedding { grid_h: input.grid_h, grid_w: input.grid_w, vectors, attention: attn.to_vec1()?, pooled })
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
        if !self.policy.fixed_shape {
            let ids_t = Tensor::from_slice(ids, (rows, len), &self.device)?;
            let mask_t = Tensor::from_slice(mask, (rows, len), &self.device)?;
            let emb = l2_normalize(&self.text.forward(&ids_t, &mask_t)?)?;
            trim_idle_pool(&self.device);
            return Ok(emb.to_vec2()?);
        }
        // CUDA and ROCm: every pass is exactly `max_rows x len`; the last one is
        // padded with copies of its first row, whose outputs are dropped. The
        // vector of a phrase then does not depend on what shares its call
        // (see `BatchPolicy`).
        let b = self.policy.max_rows.max(1);
        let mut out = Vec::with_capacity(rows);
        for start in (0..rows).step_by(b) {
            let n = b.min(rows - start);
            let mut i = ids[start * len..(start + n) * len].to_vec();
            let mut m = mask[start * len..(start + n) * len].to_vec();
            for _ in n..b {
                i.extend_from_slice(&ids[start * len..(start + 1) * len]);
                m.extend_from_slice(&mask[start * len..(start + 1) * len]);
            }
            let ids_t = Tensor::from_vec(i, (b, len), &self.device)?;
            let mask_t = Tensor::from_vec(m, (b, len), &self.device)?;
            let emb: Vec<Vec<f32>> = l2_normalize(&self.text.forward(&ids_t, &mask_t)?)?.to_vec2()?;
            out.extend(emb.into_iter().take(n));
        }
        trim_idle_pool(&self.device);
        Ok(out)
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

fn checkpoint_fingerprint(dir: &Path, config: &Siglip2Config) -> u64 {
    let mut h = fnv1a(FNV_OFFSET, format!("{config:?}").as_bytes());
    for name in ["model.safetensors", "tokenizer.json", "config.json"] {
        match std::fs::metadata(dir.join(name)) {
            Ok(meta) => {
                h = fnv1a(h, name.as_bytes());
                h = fnv1a(h, &meta.len().to_le_bytes());
                if let Ok(d) = meta.modified().and_then(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                }) {
                    h = fnv1a(h, &d.as_nanos().to_le_bytes());
                }
            }
            Err(_) => h = fnv1a(h, b"<missing>"),
        }
    }
    h
}

/// Mixes the device family and the text batch shape into a fingerprint, so
/// the phrase bank keeps CPU, CUDA, ROCm and Vulkan vectors apart.
fn device_salt(h: u64, device: &Device, policy: &BatchPolicy) -> u64 {
    let h = fnv1a(h, device_family(device).as_bytes());
    if policy.fixed_shape {
        fnv1a(h, &(policy.max_rows as u64).to_le_bytes())
    } else {
        h
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
    fn dense_embedding_has_one_unit_vector_per_patch() {
        let m = tiny_model();
        let d = m.embed_dense(&test_image(32, 16)).unwrap();
        assert_eq!(d.vectors.len(), d.grid_h * d.grid_w);
        assert!(d.grid_w > d.grid_h, "a wide image keeps a wide grid: {}x{}", d.grid_h, d.grid_w);
        for v in &d.vectors {
            assert_eq!(v.len(), 32);
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4);
        }
        let s: f32 = d.attention.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "attention sums to one: {s}");
        let whole = m.embed_image(&test_image(32, 16)).unwrap();
        let same: f32 = whole.iter().zip(&d.pooled).map(|(a, b)| a * b).sum();
        assert!((same - 1.0).abs() < 1e-4, "the pooled vector is the image embedding");
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
    fn batched_image_embeddings_match_one_by_one_in_input_order() {
        let m = tiny_model();
        // Two shapes, interleaved, so grouping by patch grid has to restore the order.
        let images = vec![test_image(64, 32), test_image(32, 32), test_image(64, 32), test_image(32, 32)];
        let batched = m.embed_images(&images, 2).unwrap();
        assert_eq!(batched.len(), 4);
        for (img, row) in images.iter().zip(&batched) {
            let single = m.embed_image(img).unwrap();
            for (a, b) in single.iter().zip(row) {
                assert!((a - b).abs() < 1e-4, "batched row differs from single: {a} vs {b}");
            }
        }
        assert!(m.embed_images(&[], 4).unwrap().is_empty());
    }

    #[test]
    fn fingerprint_depends_on_the_config() {
        let a = tiny_model();
        let mut cfg = tiny_config();
        cfg.text_layers = 3;
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let b = Siglip2::from_var_builder(cfg, vb, &dev, None).unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.fingerprint(), tiny_model().fingerprint());
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
    fn batched_dense_embeddings_match_one_by_one() {
        let m = tiny_model();
        let images = vec![test_image(32, 16), test_image(32, 32), test_image(32, 16)];
        let batched = m.embed_dense_batch(&images).unwrap();
        assert_eq!(batched.len(), 3);
        for (img, b) in images.iter().zip(&batched) {
            let one = m.embed_dense(img).unwrap();
            assert_eq!((one.grid_h, one.grid_w), (b.grid_h, b.grid_w));
            for (u, v) in one.vectors.iter().zip(&b.vectors) {
                assert!(u.iter().zip(v).all(|(a, c)| (a - c).abs() < 1e-4));
            }
            assert!(one.pooled.iter().zip(&b.pooled).all(|(a, c)| (a - c).abs() < 1e-4));
            assert!(one.attention.iter().zip(&b.attention).all(|(a, c)| (a - c).abs() < 1e-5));
        }
    }

    #[test]
    fn fixed_shape_text_batches_give_the_dynamic_vectors() {
        let mut m = tiny_model();
        let len = m.config().text_max_len;
        let rows = 5;
        let ids: Vec<u32> = (0..rows * len).map(|i| (i * 7 % 50) as u32).collect();
        let mask = vec![1.0f32; rows * len];
        m.set_batch_policy(BatchPolicy { max_rows: 64, token_budget: 4096, fixed_shape: false });
        let dynamic = m.embed_tokens(&ids, &mask, rows).unwrap();
        let fp_dynamic = m.fingerprint();
        m.set_batch_policy(BatchPolicy { max_rows: 4, token_budget: 4096, fixed_shape: true });
        let fixed = m.embed_tokens(&ids, &mask, rows).unwrap();
        assert_eq!(fixed.len(), rows, "padding rows are dropped");
        for (a, b) in dynamic.iter().zip(&fixed) {
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5));
        }
        assert_ne!(fp_dynamic, m.fingerprint(), "the phrase bank keeps the two shapes apart");
    }

    #[test]
    fn missing_weights_gives_a_clear_error() {
        let dir = std::env::temp_dir().join("noto-ml-missing-weights-test");
        let _ = std::fs::create_dir_all(&dir);
        let err = Siglip2::load(&dir, &Device::Cpu).err().expect("should fail");
        assert!(err.to_string().contains("model.safetensors"));
    }
}
