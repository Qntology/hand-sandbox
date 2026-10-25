//! Application state shared by the commands: the selected compute device and
//! the lazily loaded SigLIP2 engine.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use hand_ml::{probe_backends, Anchors, Backend, BackendStatus, Dictionary, Selection, Siglip2};
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
    /// How the start-up device was chosen (`HAND_DEVICE`, first GPU, CPU fallback).
    startup: Selection,
    probed: OnceLock<Vec<BackendStatus>>,
    vision: Mutex<Option<Arc<VisionEngine>>>,
    /// The simulated world and robot. Built on the first world command.
    world: Mutex<Option<Runtime>>,
    /// Model downloads (`models`).
    pub downloads: Arc<crate::models::Downloads>,
    /// All four models are loaded on the selected device and attached to the world.
    models_ready: AtomicBool,
    /// The model being loaded now, and why the last load failed.
    loading: Mutex<Option<String>>,
    load_error: Mutex<Option<String>>,
    live_devices: Mutex<std::collections::BTreeMap<String, String>>,
}

impl AppState {
    pub fn new() -> Self {
        // The first GPU this build has and that opens, unless HAND_DEVICE says otherwise
        // (logis-center `new_gpu_device`). The probe leaves the device in the cache.
        let startup = hand_ml::preferred();
        AppState {
            device: Mutex::new((startup.backend, startup.ordinal)),
            startup,
            probed: OnceLock::new(),
            vision: Mutex::new(None),
            world: Mutex::new(None),
            downloads: Arc::default(),
            models_ready: AtomicBool::new(false),
            loading: Mutex::new(None),
            load_error: Mutex::new(None),
            live_devices: Mutex::new(Default::default()),
        }
    }

    /// Whether the world may run: every model downloaded and loaded. The world
    /// commands call this first, so nothing runs on a stand-in.
    pub fn require_models(&self) -> Result<(), String> {
        if self.models_ready() {
            Ok(())
        } else {
            Err(crate::models::not_ready_message())
        }
    }

    pub fn models_ready(&self) -> bool {
        self.models_ready.load(Ordering::Acquire)
    }

    pub fn set_models_ready(&self, ready: bool) {
        self.models_ready.store(ready, Ordering::Release);
    }

    pub fn loading(&self) -> Option<String> {
        self.loading.lock().ok().and_then(|l| l.clone())
    }

    pub fn set_loading(&self, id: Option<&str>) {
        if let Ok(mut l) = self.loading.lock() {
            *l = id.map(String::from);
        }
    }

    pub fn load_error(&self) -> Option<String> {
        self.load_error.lock().ok().and_then(|l| l.clone())
    }

    pub fn set_load_error(&self, e: Option<String>) {
        if let Ok(mut l) = self.load_error.lock() {
            *l = e;
        }
    }

    /// The device the models go on: the selected one, unless that is the CPU
    /// while this build has a GPU that opens — then the first such GPU (the
    /// start-up order), which becomes the selection. `HAND_DEVICE=cpu` keeps
    /// the CPU. Returns whether the selection moved.
    pub fn prefer_gpu(&self) -> bool {
        let (backend, _) = self.selected();
        if backend != Backend::Cpu || std::env::var(hand_ml::device::DEVICE_ENV).is_ok_and(|v| v.trim().eq_ignore_ascii_case("cpu")) {
            return false;
        }
        let picked = hand_ml::device::select(None);
        if picked.backend == Backend::Cpu {
            return false;
        }
        self.select(picked.backend, picked.ordinal).is_ok()
    }

    /// Where each loaded model runs, by model id (set when they are loaded).
    pub fn live_devices(&self) -> std::collections::BTreeMap<String, String> {
        self.live_devices.lock().map(|d| d.clone()).unwrap_or_default()
    }

    pub fn set_live_devices(&self, d: std::collections::BTreeMap<String, String>) {
        if let Ok(mut l) = self.live_devices.lock() {
            *l = d;
        }
    }

    /// Forgets the loaded SigLIP2 (its files are about to go).
    pub fn drop_vision(&self) {
        *self.vision.lock().expect("vision lock") = None;
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

    pub fn startup(&self) -> &Selection {
        &self.startup
    }

    /// The selected device, from the shared cache (one handle per backend).
    pub fn device(&self) -> Result<candle_core::Device> {
        let (backend, ordinal) = self.selected();
        backend.device(ordinal)
    }

    /// Selects a device. The vision engine reloads on the next classification,
    /// and the world waits until the models are loaded again on this device.
    pub fn select(&self, backend: Backend, ordinal: usize) -> Result<()> {
        if !backend.compiled() {
            anyhow::bail!("{} is not compiled into this build", backend.name());
        }
        backend.device(ordinal)?;
        let changed = *self.device.lock().expect("device lock") != (backend, ordinal);
        *self.device.lock().expect("device lock") = (backend, ordinal);
        if changed {
            *self.vision.lock().expect("vision lock") = None;
            self.set_models_ready(false);
            self.set_live_devices(Default::default());
        }
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
        let device = backend.device(ordinal)?;
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

/// Where downloaded TTM files go: `TTM_DIR`, else the folder that already has
/// a `config.json`, else the `52-16-dec-52-r3` subfolder (the branch the files
/// come from).
pub fn ttm_download_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TTM_DIR") {
        return PathBuf::from(p);
    }
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join(TTM_FOLDER);
    if base.join("config.json").is_file() {
        base
    } else {
        base.join(TTM_SUBFOLDER)
    }
}

/// Folder with the TIPSv2-B/14 DPT files (`model.safetensors`, `config.json`).
/// `TIPS_DIR` wins; otherwise `models/tipsv2-b14-dpt`.
pub fn tips_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TIPS_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("tipsv2-b14-dpt")
}

/// Folder with the SAM weights (`mobile_sam-tiny-vitt.safetensors` or
/// `sam_vit_b_01ec64.safetensors`). `SAM_DIR` wins; otherwise `models/sam`.
pub fn sam_dir() -> PathBuf {
    if let Ok(p) = std::env::var(hand_vision::sam::DIR_ENV) {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("sam")
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
        let before = s.selected();
        if !Backend::Vulkan.compiled() {
            assert!(s.select(Backend::Vulkan, 0).is_err());
            assert_eq!(s.selected(), before);
        }
    }
}
