//! Application state shared by the commands: the selected compute device and
//! the lazily loaded SigLIP2 engine.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use hand_ml::{probe_backends, Anchors, Backend, BackendStatus, Dictionary, Siglip2};

/// Logistics dictionary compiled into the app.
pub const DICTIONARY_JSON: &str = include_str!("../../data/logistics_dictionary.json");
pub const MODEL_FOLDER: &str = "siglip2-base-patch16-naflex";

/// A loaded SigLIP2 model with the dictionary embedded as text anchors.
pub struct VisionEngine {
    pub model: Siglip2,
    pub anchors: Anchors,
    pub dict: Dictionary,
    pub backend: Backend,
    pub ordinal: usize,
}

pub struct AppState {
    device: Mutex<(Backend, usize)>,
    probed: OnceLock<Vec<BackendStatus>>,
    vision: Mutex<Option<Arc<VisionEngine>>>,
}

impl AppState {
    pub fn new() -> Self {
        AppState {
            device: Mutex::new((Backend::Cpu, 0)),
            probed: OnceLock::new(),
            vision: Mutex::new(None),
        }
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
        let model = Siglip2::load(&dir, &device)
            .with_context(|| format!("loading SigLIP2 from {}", dir.display()))?;
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

/// Folder with the SigLIP2 files. `SIGLIP2_DIR` wins; otherwise `models/` next
/// to the project root, which is where `cargo tauri dev` runs from.
pub fn model_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SIGLIP2_DIR") {
        return Ok(PathBuf::from(p));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    Ok(root.join("models").join(MODEL_FOLDER))
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
