//! Application state shared by the commands: the selected compute device and
//! the lazily loaded SigLIP2 engine.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use hand_ml::{probe_backends, Anchors, Backend, BackendStatus, Dictionary, Siglip2};
use hand_runtime::{Runtime, RuntimeConfig};

/// Logistics dictionary compiled into the app.
pub const DICTIONARY_JSON: &str = include_str!("../../data/logistics_dictionary.json");
pub const MODEL_FOLDER: &str = "siglip2-base-patch16-naflex";
pub const TTM_FOLDER: &str = "granite-timeseries-ttm-r3";
pub const TTM_SUBFOLDER: &str = "52-16-dec-52-r3";
/// Word dictionary of the world commands (logis-center `bias.json` node format).
pub const WORLD_LEXICON_JSON: &str = include_str!("../../data/world_lexicon.json");

/// A loaded SigLIP2 model with the dictionary embedded as text anchors.
pub struct VisionEngine {
    pub model: Arc<Siglip2>,
    pub anchors: Anchors,
    pub dict: Dictionary,
    pub backend: Backend,
    pub ordinal: usize,
}

pub struct AppState {
    device: Mutex<(Backend, usize)>,
    probed: OnceLock<Vec<BackendStatus>>,
    vision: Mutex<Option<Arc<VisionEngine>>>,
    /// The simulated world and robot. Built on the first world command.
    world: Mutex<Option<Runtime>>,
}

impl AppState {
    pub fn new() -> Self {
        AppState {
            device: Mutex::new((Backend::Cpu, 0)),
            probed: OnceLock::new(),
            vision: Mutex::new(None),
            world: Mutex::new(None),
        }
    }

    /// Runs `f` on the world runtime, building it first if needed. The lock is
    /// held for the whole call, so calls take turns.
    pub fn with_world<T>(&self, f: impl FnOnce(&mut Runtime) -> Result<T, String>) -> Result<T, String> {
        let mut slot = self.world.lock().map_err(|_| "월드 잠금이 깨졌습니다. 앱을 다시 시작하세요".to_string())?;
        if slot.is_none() {
            *slot = Some(build_runtime()?);
        }
        f(slot.as_mut().expect("just built"))
    }

    /// Backends compiled in and whether each one opens on this machine. Probed once.
    pub fn backends(&self) -> &[BackendStatus] {
        self.probed.get_or_init(probe_backends)
    }

    pub fn selected(&self) -> (Backend, usize) {
        *self.device.lock().expect("device lock")
    }

    /// Selects a device. The vision engine reloads on the next classification.
    pub fn select(&self, backend: Backend, ordinal: usize) -> Result<()> {
        if !backend.compiled() {
            anyhow::bail!("{} is not compiled into this build", backend.name());
        }
        backend.open(ordinal)?;
        *self.device.lock().expect("device lock") = (backend, ordinal);
        *self.vision.lock().expect("vision lock") = None;
        Ok(())
    }

    /// Returns the vision engine, loading it when missing or when the device changed.
    pub fn vision_engine(&self) -> Result<Arc<VisionEngine>> {
        let (backend, ordinal) = self.selected();
        let mut slot = self.vision.lock().expect("vision lock");
        if let Some(engine) = slot.as_ref() {
            if engine.backend == backend && engine.ordinal == ordinal {
                return Ok(Arc::clone(engine));
            }
        }
        let dir = model_dir()?;
        let device = backend.open(ordinal)?;
        let model = Arc::new(
            Siglip2::load(&dir, &device).with_context(|| format!("loading SigLIP2 from {}", dir.display()))?,
        );
        let dict = Dictionary::from_json(DICTIONARY_JSON)?;
        let anchors = Anchors::build(&model, &dict)?;
        let engine = Arc::new(VisionEngine {
            model,
            anchors,
            dict,
            backend,
            ordinal,
        });
        *slot = Some(Arc::clone(&engine));
        Ok(engine)
    }
}

/// Where the SDS ledgers and the hash-chained logs of the world are kept:
/// `data/runtime` at the project root, or `RUNTIME_DIR`.
pub fn runtime_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUNTIME_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("data").join("runtime")
}

/// A runtime that keeps its files under [`runtime_dir`]. If that folder cannot
/// be used (read-only install, say) it runs with memory only, and says so in
/// its log.
fn build_runtime() -> Result<Runtime, String> {
    let dir = runtime_dir();
    match Runtime::new(RuntimeConfig::default(), WORLD_LEXICON_JSON, Some(&dir)) {
        Ok(rt) => Ok(rt),
        Err(first) => {
            let mut rt = Runtime::new(RuntimeConfig::default(), WORLD_LEXICON_JSON, None).map_err(|e| e.to_string())?;
            rt.note_device(&format!("로그·SDS 폴더 {}를 쓸 수 없어 메모리에만 기록합니다: {first}", dir.display()));
            Ok(rt)
        }
    }
}

/// Folder with the SigLIP2 files. `SIGLIP2_DIR` wins; otherwise `models/` next
/// to the project root, which is where `cargo tauri dev` runs from.
pub fn model_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SIGLIP2_DIR") {
        return Ok(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    Ok(root.join("models").join(MODEL_FOLDER))
}

/// Folder with the Granite TTM files (`config.json` and the safetensors).
/// `TTM_DIR` wins; otherwise `models/granite-timeseries-ttm-r3`, or its
/// `52-16-dec-52-r3` subfolder when the files were downloaded with that layout.
pub fn ttm_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TTM_DIR") {
        return PathBuf::from(p);
    }
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join(TTM_FOLDER);
    let nested = base.join(TTM_SUBFOLDER);
    if !base.join("config.json").is_file() && nested.join("config.json").is_file() {
        return nested;
    }
    base
}

/// Folder with the TIPSv2-B/14 DPT files (`model.safetensors`, `config.json`).
/// `TIPS_DIR` wins; otherwise `models/tipsv2-b14-dpt`.
pub fn tips_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TIPS_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("tipsv2-b14-dpt")
}

/// Where the embedded phrase vectors of the world's recogniser are kept between runs.
pub fn bank_dir() -> PathBuf {
    runtime_dir().join("bank")
}

/// Default Lance store: `data/lance` at the project root, or `LANCE_DIR`.
pub fn default_store_dir() -> PathBuf {
    if let Ok(p) = std::env::var("LANCE_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
        .join("lance")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_constant_parses() {
        let d = Dictionary::from_json(DICTIONARY_JSON).unwrap();
        assert_eq!(d.entries.len(), 28);
    }

    #[test]
    fn selecting_uncompiled_backend_fails_cleanly() {
        let s = AppState::new();
        if !Backend::Vulkan.compiled() {
            assert!(s.select(Backend::Vulkan, 0).is_err());
            assert_eq!(s.selected(), (Backend::Cpu, 0));
        }
    }
}
