//! GPU backend parity (features `cuda`, `rocm`, `vulkan`, `metal`): the models
//! of the ML logic (SigLIP2 vision and text towers, TIPSv2 DPT depth, SAM mask
//! encoder and decoder) run with the same random weights on the CPU and on
//! the GPU this build opens first (`hand_ml::preferred`), batched as the
//! runtime batches them, and the results are compared. Step 07 of
//! `test_runner.bat` runs this and keeps the `[GPU]` lines (largest error per
//! model). It checks numbers, not speed.
//!
//! The physics and the game logic never run here: they are CPU code for every
//! backend and must give the same results everywhere (steps 02-06).
#![cfg(any(feature = "cuda", feature = "rocm", feature = "vulkan", feature = "metal"))]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use hand_ml::{Siglip2, Siglip2Config};
use hand_vision::{MaskTracker, PointPrompt, Rgb, SamModel, SamTracker, SamVariant, TipsConfig, TipsDepth};

fn gpu() -> Device {
    let s = hand_ml::preferred();
    assert_ne!(
        s.backend,
        hand_ml::Backend::Cpu,
        "[GPU] no GPU device opened ({}) - check the driver / runtime (CUDA toolkit, HIP SDK, vulkan-1.dll) and HAND_DEVICE. \
         A Vulkan build without a GPU can use CANDLE_VULKAN_ALLOW_CPU=1",
        s.reason
    );
    let d = s.backend.device(s.ordinal).expect("the selected device opens");
    println!("[GPU] device {} ({})", hand_ml::device_label(&d), s.reason);
    d
}

/// The weights a CPU model was built with, copied to `dev`.
fn weights_on(vm: &VarMap, dev: &Device) -> HashMap<String, Tensor> {
    vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.as_tensor().to_device(dev).unwrap())).collect()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| {
        assert!(x.is_finite() && y.is_finite(), "non-finite value: {x} vs {y}");
        (x - y).abs() / x.abs().max(y.abs()).max(1.0)
    }).fold(0.0, f32::max)
}

fn frame(w: u32, h: u32, k: u32) -> Rgb {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.extend_from_slice(&[((x * 7 + k * 13) % 256) as u8, ((y * 5 + k) % 256) as u8, (((x + y) * 3) % 256) as u8]);
        }
    }
    Rgb::new(w, h, data)
}

fn tiny_siglip() -> Siglip2Config {
    Siglip2Config {
        vision_hidden: 64,
        vision_intermediate: 128,
        vision_layers: 2,
        vision_heads: 4,
        patch_size: 8,
        max_patches: 64,
        vision_eps: 1e-6,
        text_hidden: 64,
        text_intermediate: 128,
        text_layers: 2,
        text_heads: 4,
        text_vocab: 100,
        text_max_len: 16,
        text_pad_id: 1,
        text_eps: 1e-6,
    }
}

#[test]
fn siglip2_vision_dense_and_text_match_the_cpu() {
    let dev = gpu();
    let vm = VarMap::new();
    let cpu = Siglip2::from_var_builder(tiny_siglip(), VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), &Device::Cpu, None).unwrap();
    let g = Siglip2::from_var_builder(tiny_siglip(), VarBuilder::from_tensors(weights_on(&vm, &dev), DType::F32, &dev), &dev, None).unwrap();
    let images: Vec<_> = (0..3).map(|k| frame(96, 64, k).to_dynamic()).collect();
    let (a, b) = (cpu.embed_images(&images, 0).unwrap(), g.embed_images(&images, 0).unwrap());
    let e_img = a.iter().zip(&b).map(|(x, y)| worst(x, y)).fold(0.0, f32::max);
    let (da, db) = (cpu.embed_dense_batch(&images).unwrap(), g.embed_dense_batch(&images).unwrap());
    let e_dense = da.iter().zip(&db).map(|(x, y)| x.vectors.iter().zip(&y.vectors).map(|(u, v)| worst(u, v)).fold(0.0, f32::max)).fold(0.0, f32::max);
    let len = tiny_siglip().text_max_len;
    let rows = 5;
    let ids: Vec<u32> = (0..rows * len).map(|i| (i * 11 % 100) as u32).collect();
    let mut mask = vec![1.0f32; rows * len];
    for v in &mut mask[len + 6..2 * len] {
        *v = 0.0;
    }
    let (ta, tb) = (cpu.embed_tokens(&ids, &mask, rows).unwrap(), g.embed_tokens(&ids, &mask, rows).unwrap());
    let e_text = ta.iter().zip(&tb).map(|(x, y)| worst(x, y)).fold(0.0, f32::max);
    println!("[GPU] siglip2 · image {e_img:.2e} · dense {e_dense:.2e} · text {e_text:.2e} (fixed-shape text: {})", g.batch_policy().fixed_shape);
    assert!(e_img < 1e-3 && e_dense < 1e-3 && e_text < 1e-3);
}

fn tiny_tips() -> TipsConfig {
    TipsConfig {
        hidden: 64,
        layers: 4,
        heads: 4,
        mlp_hidden: 128,
        tap_blocks: [0, 1, 2, 3],
        post_process_channels: [16, 32, 64, 64],
        dpt_channels: 32,
        depth_bins: 16,
        ..TipsConfig::default()
    }
}

#[test]
fn tips_depth_batch_matches_the_cpu() {
    let dev = gpu();
    let vm = VarMap::new();
    let cpu = TipsDepth::from_var_builder(tiny_tips(), VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), &Device::Cpu).unwrap();
    let g = TipsDepth::from_var_builder(tiny_tips(), VarBuilder::from_tensors(weights_on(&vm, &dev), DType::F32, &dev), &dev).unwrap();
    let frames: Vec<Rgb> = (0..3).map(|k| frame(192, 128, k)).collect();
    let refs: Vec<&Rgb> = frames.iter().collect();
    let (a, b) = (cpu.predict_batch(&refs).unwrap(), g.predict_batch(&refs).unwrap());
    let e = a.iter().zip(&b).map(|(x, y)| worst(&x.data, &y.data)).fold(0.0, f32::max);
    // One frame alone on the GPU gives the batched row.
    let one = g.predict(&frames[1]).unwrap();
    let e_row = worst(&one.data, &b[1].data);
    println!("[GPU] tips depth · batch vs cpu {e:.2e} · single vs batched row {e_row:.2e}");
    assert!(e < 1e-3 && e_row < 1e-4);
}

#[test]
fn sam_encoder_and_decoder_match_the_cpu() {
    let dev = gpu();
    let vm = VarMap::new();
    let variant = SamVariant::TestVit { dim: 32, depth: 2, heads: 2 };
    let cpu = SamModel::from_var_builder(variant, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu), &Device::Cpu).unwrap();
    // SAM's LayerNorm2d reads its weights with a zero default: without this the
    // encoder output would be all zeros and the comparison would prove nothing.
    for (name, var) in vm.data().lock().unwrap().iter() {
        if name.ends_with("weight") && var.as_tensor().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap() == 0.0 {
            let shape = var.as_tensor().dims().to_vec();
            var.set(&(Tensor::randn(0f32, 0.1, shape, &Device::Cpu).unwrap() + 1.0).unwrap()).unwrap();
        }
    }
    let g = SamModel::from_var_builder(variant, VarBuilder::from_tensors(weights_on(&vm, &dev), DType::F32, &dev), &dev).unwrap();
    let frames: Vec<Rgb> = (0..2).map(|k| frame(192, 128, k)).collect();
    let refs: Vec<&Rgb> = frames.iter().collect();
    let (ea, eb) = (cpu.encode_batch(&refs).unwrap(), g.encode_batch(&refs).unwrap());
    let e_enc = ea.iter().zip(&eb).map(|(x, y)| worst(&x.flatten_all().unwrap().to_vec1().unwrap(), &y.flatten_all().unwrap().to_vec1().unwrap())).fold(0.0, f32::max);
    let spread = ea[0].abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(spread > 1e-3, "the encoder output is not trivially zero: {spread}");
    let pts = [(60.0, 50.0), (150.0, 90.0), (20.0, 20.0)];
    let (qa, qb) = (cpu.decode(&ea[0], 192, 128, &pts).unwrap(), g.decode(&eb[0], 192, 128, &pts).unwrap());
    let e_dec = qa.iter().zip(&qb).map(|(x, y)| worst(&x.logits, &y.logits)).fold(0.0, f32::max);
    let e_iou = qa.iter().zip(&qb).map(|(x, y)| (x.iou_pred - y.iou_pred).abs()).fold(0.0, f32::max);
    println!("[GPU] sam · encoder {e_enc:.2e} · decoder logits {e_dec:.2e} · iou {e_iou:.2e} · {} images per encoder pass", g.encoder_rows());
    assert!(e_enc < 1e-3 && e_dec < 2e-3 && e_iou < 1e-3);
    // The tracker on the GPU runs end to end on the prepared embeddings.
    let mut t = SamTracker::new(Arc::new(g));
    t.prepare(&refs).unwrap();
    let p = PointPrompt { label: "x".into(), px: 60.0, py: 50.0, score: 0.9 };
    let out = t.update(&frames[0], None, &[p]).unwrap();
    assert!(out.len() <= 1);
    assert_eq!(t.cached_frames(), 2);
}
