use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use image::DynamicImage;

use crate::config::Siglip2Config;
use crate::device::{device_family, trim_idle_pool, BatchPolicy};
use crate::parallel::{fnv1a, FNV_OFFSET};
use crate::preprocess::{preprocess, preprocess_grid, DenseGrid, NaflexInput};
use crate::text::TextTower;
use crate::tokenizer::TextTokenizer;
use crate::vision::VisionTower;

#[derive(Clone, Debug)]
pub struct DenseEmbedding {
    pub grid_h: usize,
    pub grid_w: usize,
    pub vectors: Vec<Vec<f32>>,
    pub attention: Vec<f32>,
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
    policy: BatchPolicy,
    checkpoint: u64,
    fingerprint: u64,
}

impl Siglip2 {
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

    pub fn batch_policy(&self) -> BatchPolicy {
        self.policy
    }

    pub fn set_position_antialias(&mut self, on: bool) {
        self.vision.set_position_antialias(on);
    }

    pub fn position_antialias(&self) -> bool {
        self.vision.position_antialias()
    }

    pub fn set_batch_policy(&mut self, policy: BatchPolicy) {
        self.fingerprint = device_salt(self.checkpoint, &self.device, &policy);
        self.policy = policy;
    }

    fn tokens_of(input: &NaflexInput) -> usize {
        input.grid_h * input.grid_w
    }

    pub fn config(&self) -> &Siglip2Config {
        &self.config
    }

    pub fn embedding_dim(&self) -> usize {
        self.config.vision_hidden
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn pair_probability(&self, cosine: f32) -> f32 {
        sigmoid(self.logit_scale.exp() * cosine + self.logit_bias)
    }

    pub fn logit_scale_bias(&self) -> (f32, f32) {
        (self.logit_scale.exp(), self.logit_bias)
    }

    pub fn embed_image(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        let input = preprocess(image, &self.config);
        let emb = self.vision.forward(&input)?;
        Ok(l2_normalize(&emb)?.flatten_all()?.to_vec1()?)
    }

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

    pub fn embed_dense_batch(&self, images: &[DynamicImage]) -> Result<Vec<DenseEmbedding>> {
        self.embed_dense_batch_grid(images, DenseGrid::Naflex)
    }

    /// The dense pass on the grid `grid` picks for each image (`DenseGrid::Pyramid` lines the patches up with the
    /// depth and mask layers; the position table is resampled to any grid either way).
    pub fn embed_dense_batch_grid(&self, images: &[DynamicImage], grid: DenseGrid) -> Result<Vec<DenseEmbedding>> {
        let inputs = match grid {
            DenseGrid::Naflex => self.preprocess_all(images),
            _ => {
                use rayon::prelude::*;
                let (p, budget) = (self.config.patch_size, self.config.max_patches);
                images
                    .par_iter()
                    .map(|image| {
                        let (gh, gw) = grid.grid_for(image.width(), image.height(), p, budget);
                        preprocess_grid(image, &self.config, gh, gw)
                    })
                    .collect()
            }
        };
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

    pub fn embed_dense(&self, image: &DynamicImage) -> Result<DenseEmbedding> {
        let input = preprocess(image, &self.config);
        let (dense, attn, pooled) = self.vision.forward_dense(&input)?;
        let vectors = l2_normalize(&dense)?.to_vec2()?;
        let pooled = l2_normalize(&pooled.unsqueeze(0)?)?.flatten_all()?.to_vec1()?;
        Ok(DenseEmbedding { grid_h: input.grid_h, grid_w: input.grid_w, vectors, attention: attn.to_vec1()?, pooled })
    }

    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow!("this model was loaded without a tokenizer"))?;
        let (ids, mask) = tokenizer.encode(texts)?;
        self.embed_tokens(&ids, &mask, texts.len())
    }

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
    fn the_pyramid_grid_is_the_dense_grid_it_asks_for() {
        let m = tiny_model();
        let img = test_image(48, 32);
        let grid = DenseGrid::Pyramid { short_side: 2 };
        let want = grid.grid_for(48, 32, m.config().patch_size, m.config().max_patches);
        let d = m.embed_dense_batch_grid(&[img.clone()], grid).unwrap().pop().unwrap();
        assert_eq!((d.grid_h, d.grid_w), want);
        assert_eq!((d.grid_h, d.grid_w), (2, 3), "3:2 keeps a 3:2 grid");
        assert_eq!(d.vectors.len(), 6);
        let naflex = m.embed_dense_batch_grid(&[img.clone()], DenseGrid::Naflex).unwrap().pop().unwrap();
        let plain = m.embed_dense(&img).unwrap();
        assert_eq!((naflex.grid_h, naflex.grid_w), (plain.grid_h, plain.grid_w), "NaFlex is the default dense grid");
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
    fn the_position_filter_changes_only_grids_the_table_is_shrunk_for() {
        let mut m = tiny_model();
        assert!(m.position_antialias(), "the reference's antialiased filter is the default");
        let wide = test_image(64, 32);
        let square = test_image(16, 16);
        let (a_wide, a_sq) = (m.embed_image(&wide).unwrap(), m.embed_image(&square).unwrap());
        m.set_position_antialias(false);
        let (b_wide, b_sq) = (m.embed_image(&wide).unwrap(), m.embed_image(&square).unwrap());
        assert_eq!(a_sq, b_sq, "a 4x4 grid takes the table as it is");
        assert!(a_wide.iter().zip(&b_wide).any(|(x, y)| (x - y).abs() > 1e-5), "shrinking averages rows instead of sampling them");
    }

    #[test]
    fn non_square_grids_use_resized_positions() {
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
