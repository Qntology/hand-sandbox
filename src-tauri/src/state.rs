use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use noto_ml::{probe_backends, Anchors, Backend, BackendStatus, Dictionary, Selection, Siglip2};
use noto_runtime::{Runtime, RuntimeConfig};

pub const DICTIONARY_JSON: &str = include_str!("../../data/logistics_dictionary.json");
pub const MODEL_FOLDER: &str = "siglip2-base-patch16-naflex";
pub const TTM_FOLDER: &str = "granite-timeseries-ttm-r3";
pub const TTM_SUBFOLDER: &str = "52-16-dec-52-r3";
pub const WORLD_LEXICON_JSON: &str = include_str!("../../data/world_lexicon.json");

pub struct VisionEngine {
    pub model: Arc<Siglip2>,
    pub anchors: Anchors,
    pub dict: Dictionary,
    pub backend: Backend,
    pub ordinal: usize,
}

pub struct AppState {
    device: Mutex<(Backend, usize)>,
    startup: Selection,
    probed: OnceLock<Vec<BackendStatus>>,
    vision: Mutex<Option<Arc<VisionEngine>>>,
    world: Mutex<Option<Runtime>>,
    pub downloads: Arc<crate::models::Downloads>,
    models_ready: AtomicBool,
    /// The robot's eye is a stereo pair (two cameras): its depth is SGBM's, and TIPSv2 is neither needed nor loaded.
    stereo: AtomicBool,
    loading: Mutex<Option<String>>,
    load_error: Mutex<Option<String>>,
    live_devices: Mutex<std::collections::BTreeMap<String, String>>,
}

impl AppState {
    pub fn new() -> Self {
        let startup = noto_ml::preferred();
        AppState {
            device: Mutex::new((startup.backend, startup.ordinal)),
            startup,
            probed: OnceLock::new(),
            vision: Mutex::new(None),
            world: Mutex::new(None),
            downloads: Arc::default(),
            models_ready: AtomicBool::new(false),
            stereo: AtomicBool::new(false),
            loading: Mutex::new(None),
            load_error: Mutex::new(None),
            live_devices: Mutex::new(Default::default()),
        }
    }

    pub fn require_models(&self) -> Result<(), String> {
        if self.models_ready() {
            Ok(())
        } else {
            Err(crate::models::not_ready_message(self.stereo()))
        }
    }

    pub fn stereo(&self) -> bool {
        self.stereo.load(Ordering::Acquire)
    }

    pub fn set_stereo(&self, on: bool) {
        self.stereo.store(on, Ordering::Release);
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

    pub fn prefer_gpu(&self) -> bool {
        let (backend, _) = self.selected();
        if backend != Backend::Cpu || std::env::var(noto_ml::device::DEVICE_ENV).is_ok_and(|v| v.trim().eq_ignore_ascii_case("cpu")) {
            return false;
        }
        let picked = noto_ml::device::select(None);
        if picked.backend == Backend::Cpu {
            return false;
        }
        self.select(picked.backend, picked.ordinal).is_ok()
    }

    pub fn live_devices(&self) -> std::collections::BTreeMap<String, String> {
        self.live_devices.lock().map(|d| d.clone()).unwrap_or_default()
    }

    pub fn set_live_devices(&self, d: std::collections::BTreeMap<String, String>) {
        if let Ok(mut l) = self.live_devices.lock() {
            *l = d;
        }
    }

    pub fn drop_vision(&self) {
        *self.vision.lock().expect("vision lock") = None;
    }

    pub fn with_world<T>(&self, f: impl FnOnce(&mut Runtime) -> Result<T, String>) -> Result<T, String> {
        let mut slot = self.world.lock().map_err(|_| "월드 잠금이 깨졌습니다. 앱을 다시 시작하세요".to_string())?;
        if slot.is_none() {
            *slot = Some(build_runtime()?);
        }
        f(slot.as_mut().expect("just built"))
    }

    pub fn backends(&self) -> &[BackendStatus] {
        self.probed.get_or_init(probe_backends)
    }

    pub fn selected(&self) -> (Backend, usize) {
        *self.device.lock().expect("device lock")
    }

    pub fn startup(&self) -> &Selection {
        &self.startup
    }

    pub fn device(&self) -> Result<candle_core::Device> {
        let (backend, ordinal) = self.selected();
        backend.device(ordinal)
    }

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

pub fn runtime_dir() -> PathBuf {
    if let Ok(p) = std::env::var("RUNTIME_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("data").join("runtime")
}

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

pub fn model_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SIGLIP2_DIR") {
        return Ok(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    Ok(root.join("models").join(MODEL_FOLDER))
}

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

pub fn tips_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TIPS_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("tipsv2-b14-dpt")
}

pub fn sam_dir() -> PathBuf {
    if let Ok(p) = std::env::var(noto_vision::sam::DIR_ENV) {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models").join("sam")
}

pub fn bank_dir() -> PathBuf {
    runtime_dir().join("bank")
}

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
