//! Loading of TTM files (config.json and safetensors weights).

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path.display(), e))
}

/// Weight files of a model folder: `model.safetensors`, or the shards named by
/// `model.safetensors.index.json`, or every `*.safetensors` file. ONNX-only
/// folders are refused with a clear message.
pub fn safetensor_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !dir.exists() {
        return Err(format!("모델 디렉터리가 없습니다: {}", dir.display()));
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let txt = std::fs::read_to_string(&index).map_err(|e| format!("{}: {}", index.display(), e))?;
        let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        let mut files: Vec<String> = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| m.values().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        files.sort();
        files.dedup();
        if !files.is_empty() {
            return Ok(files.into_iter().map(|f| dir.join(f)).collect());
        }
    }
    let entries: Vec<PathBuf> =
        std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    let mut found: Vec<PathBuf> = entries.iter().filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false)).cloned().collect();
    found.sort();
    if found.is_empty() {
        if entries.iter().any(|p| p.extension().map(|x| x == "onnx").unwrap_or(false)) {
            return Err(format!("ONNX 가중치만 있습니다. safetensors 가중치가 필요합니다: {}", dir.display()));
        }
        return Err(format!("safetensors 가중치를 찾을 수 없습니다: {}", dir.display()));
    }
    Ok(found)
}

pub fn load_varbuilder(dir: &Path, device: &Device) -> Result<VarBuilder<'static>, String> {
    let files = safetensor_files(dir)?;
    // SAFETY: the weight files are only read, and are not modified while mapped.
    unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, device) }.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hand_ts_io_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn finds_single_sharded_and_loose_weight_files() {
        let d = tmp("single");
        std::fs::write(d.join("model.safetensors"), b"x").unwrap();
        assert_eq!(safetensor_files(&d).unwrap(), vec![d.join("model.safetensors")]);

        let d = tmp("shards");
        std::fs::write(d.join("model.safetensors.index.json"), r#"{"weight_map":{"a":"m-2.safetensors","b":"m-1.safetensors","c":"m-1.safetensors"}}"#).unwrap();
        assert_eq!(safetensor_files(&d).unwrap(), vec![d.join("m-1.safetensors"), d.join("m-2.safetensors")]);

        let d = tmp("loose");
        std::fs::write(d.join("b.safetensors"), b"x").unwrap();
        std::fs::write(d.join("a.safetensors"), b"x").unwrap();
        assert_eq!(safetensor_files(&d).unwrap(), vec![d.join("a.safetensors"), d.join("b.safetensors")]);
    }

    #[test]
    fn explains_missing_and_onnx_only_folders() {
        let d = tmp("onnx");
        std::fs::write(d.join("model.onnx"), b"x").unwrap();
        assert!(safetensor_files(&d).unwrap_err().contains("ONNX"));
        let d = tmp("empty");
        assert!(safetensor_files(&d).unwrap_err().contains("찾을 수 없습니다"));
        assert!(safetensor_files(&d.join("nope")).unwrap_err().contains("없습니다"));
        assert!(read_json(&d.join("config.json")).is_err());
    }
}
