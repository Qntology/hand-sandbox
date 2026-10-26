//! Smoke tests for the Granite TTM port with random weights.
//!
//! The real checkpoint (`ibm-granite/granite-timeseries-ttm-r3`, folder
//! `52-16-dec-52-r3`) is not available in the development sandbox, so these
//! tests only prove what random weights can prove: the tensor shapes of the
//! copied model line up for a config of the same family, outputs are finite
//! and deterministic, bad inputs are rejected, and the fusion accepts the
//! model as one of its predictors. They say nothing about forecast accuracy.

use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use noto_ts::ttm::{TtmConfig, TtmForecaster};
use noto_ts::{forecast, TtmPredictor};

struct RandomBackend {
    seed: u64,
}

fn fnv(name: &str, seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl SimpleBackend for RandomBackend {
    fn get(&self, s: Shape, name: &str, _h: Init, dtype: DType, dev: &Device) -> Result<Tensor> {
        let dims = s.dims().to_vec();
        let n: usize = dims.iter().product();
        let mut state = fnv(name, self.seed);
        let fan_in = dims.last().copied().unwrap_or(1).max(1) as f32;
        let scale = 1.0 / fan_in.sqrt();
        let norm_weight = dims.len() == 1 && name.ends_with("weight");
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                if norm_weight {
                    1.0 + 0.05 * u
                } else {
                    scale * u
                }
            })
            .collect();
        Tensor::from_vec(vals, s, dev)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, name: &str, _dtype: DType, _dev: &Device) -> Result<Tensor> {
        Err(candle_core::Error::Msg(format!("get_unchecked({name}) is not supported by the random backend")))
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn config(extra: serde_json::Value) -> TtmConfig {
    let mut v = serde_json::json!({
        "context_length": 52,
        "prediction_length": 16,
        "patch_length": 4,
        "patch_stride": 4,
        "num_input_channels": 1,
        "d_model": 16,
        "num_layers": 2,
        "expansion_factor": 2,
        "mode": "common_channel",
        "gated_attn": true,
        "norm_mlp": "LayerNorm",
        "scaling": "std",
        "decoder_num_layers": 2,
        "decoder_d_model": 8,
        "decoder_mode": "common_channel",
        "use_decoder": true,
        "adaptive_patching_levels": 0
    });
    if let (Some(base), Some(add)) = (v.as_object_mut(), extra.as_object()) {
        for (k, val) in add {
            base.insert(k.clone(), val.clone());
        }
    }
    serde_json::from_value(v).expect("config parses")
}

fn model(seed: u64, extra: serde_json::Value) -> TtmForecaster {
    let vb = VarBuilder::from_backend(Box::new(RandomBackend { seed }), DType::F32, Device::Cpu);
    TtmForecaster::from_varbuilder(config(extra), vb, std::path::PathBuf::from("random")).expect("model builds")
}

fn series(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.1 * (i as f64 * 0.4).sin() + 0.01 * (i % 3) as f64).collect()
}

#[test]
fn random_weight_model_forecasts_the_configured_horizon_with_finite_values() {
    let m = model(1, serde_json::json!({}));
    assert_eq!((m.context_length(), m.prediction_length(), m.min_history()), (52, 16, 8));
    let out = m.forecast(&series(52)).expect("forecast");
    assert_eq!(out.point.len(), 16);
    assert!(out.point.iter().all(|v| v.is_finite()), "{:?}", out.point);
    // Same weights, same input, same output.
    let again = m.forecast(&series(52)).unwrap();
    assert_eq!(out.point, again.point);
    // Different weights give a different forecast, so the output depends on the weights.
    let other = model(2, serde_json::json!({})).forecast(&series(52)).unwrap();
    assert_ne!(out.point, other.point);
}

#[test]
fn shorter_histories_are_padded_and_still_finite() {
    let m = model(3, serde_json::json!({}));
    for n in [8, 20, 51, 80] {
        let out = m.forecast(&series(n)).unwrap_or_else(|e| panic!("n={n}: {e}"));
        assert_eq!(out.point.len(), 16, "n={n}");
        assert!(out.point.iter().all(|v| v.is_finite()), "n={n}: {:?}", out.point);
    }
}

#[test]
fn bad_inputs_and_unsupported_configs_are_refused() {
    let m = model(4, serde_json::json!({}));
    assert!(m.forecast(&series(7)).unwrap_err().contains("너무 짧습니다"));
    let mut bad = series(30);
    bad[5] = f64::NAN;
    assert!(m.forecast(&bad).unwrap_err().contains("NaN"));
    assert!(TtmForecaster::validate(&config(serde_json::json!({"self_attn": true}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({"adaptive_patching_levels": 1}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({"mode": "mix_channel"}))).is_err());
    assert!(TtmForecaster::validate(&config(serde_json::json!({}))).is_ok());
}

#[test]
fn the_fusion_can_use_the_model_as_one_predictor() {
    let m = model(5, serde_json::json!({}));
    let ttm = TtmPredictor(&m);
    let y = series(60);
    let f = forecast(&y, 3, &[&ttm], &[]);
    assert!(f.weights.iter().any(|w| w.method == "ttm"), "{:?} / {:?}", f.weights, f.notes);
    assert!(f.point.iter().all(|v| v.is_finite()));
    let sum: f64 = f.weights.iter().map(|w| w.weight).sum();
    assert!((sum - 1.0).abs() < 1e-9);
    // Random weights forecast badly, so a good built-in predictor outweighs it.
    let wt = f.weights.iter().find(|w| w.method == "ttm").unwrap().weight;
    let best_other = f.weights.iter().filter(|w| w.method != "ttm").map(|w| w.weight).fold(0.0, f64::max);
    assert!(best_other >= wt, "ttm {wt} vs best other {best_other}");
    // A horizon beyond the model's prediction length leaves it out instead of failing.
    let long = forecast(&y, 20, &[&ttm], &[]);
    assert!(long.weights.iter().all(|w| w.method != "ttm"));
    assert!(long.notes.iter().any(|n| n.contains("ttm")), "{:?}", long.notes);
    assert_eq!(long.point.len(), 20);
}

#[test]
fn a_missing_weight_folder_gives_a_readable_error() {
    let err = TtmForecaster::load(std::path::Path::new("/nonexistent/ttm"), &Device::Cpu).unwrap_err();
    assert!(err.contains("config.json") || err.contains("없습니다"), "{err}");
}
