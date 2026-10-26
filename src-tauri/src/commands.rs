//! Tauri commands. Each one is a thin wrapper over `views`, `noto-ml` or
//! `noto-data`. Heavy work (model load, inference, Lance writes) runs off the
//! UI thread.

use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine;
use noto_core::{GraspSetup, NotoConfig, Payload, Terrain};
use noto_data::schema::{
    TABLE_DICTIONARY, TABLE_CONTACTS, TABLE_EMBEDDINGS, TABLE_JOINT_STATES, TABLE_POINT_CLOUD,
    TABLE_SAMPLES, TABLE_TACTILE, TABLE_VISION,
};
use noto_data::{dictionary_batch, random_scene, simulate_scene, LanceStore};
use noto_ml::classify::{classify, Prediction};
use noto_ml::{Backend, BackendStatus, DictEntry, Dictionary};
use serde::Serialize;
use tauri::State;

use crate::state::{default_store_dir, model_dir, AppState, DICTIONARY_JSON};
use crate::views::{self, DriveView, GraspView, HandView, PresetsView};

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[tauri::command]
pub fn presets() -> PresetsView {
    views::presets_view()
}

#[tauri::command]
pub fn hand_geometry(config: NotoConfig, closure: f32) -> Result<HandView, String> {
    views::hand_view(&config, closure)
}

#[tauri::command]
pub fn grasp_test(
    config: NotoConfig,
    payload: Payload,
    setup: GraspSetup,
) -> Result<GraspView, String> {
    views::grasp_view(&config, &payload, &setup)
}

#[tauri::command]
pub fn drive_courses(course: Option<Vec<Terrain>>) -> Vec<DriveView> {
    let course = course.unwrap_or_else(views::default_course);
    views::drive_matrix(&course)
}

#[tauri::command]
pub fn dictionary() -> Result<Vec<DictEntry>, String> {
    Ok(Dictionary::from_json(DICTIONARY_JSON).map_err(err)?.entries)
}

#[derive(Serialize)]
pub struct DeviceState {
    pub backends: Vec<BackendStatus>,
    pub selected: Backend,
    pub ordinal: usize,
    /// How the start-up device was chosen.
    pub reason: String,
}

fn device_state(state: &AppState) -> DeviceState {
    let (selected, ordinal) = state.selected();
    DeviceState {
        backends: state.backends().to_vec(),
        selected,
        ordinal,
        reason: state.startup().reason.clone(),
    }
}

#[tauri::command]
pub fn devices(state: State<'_, Arc<AppState>>) -> DeviceState {
    device_state(&state)
}

#[tauri::command]
pub fn select_device(
    state: State<'_, Arc<AppState>>,
    backend: String,
    ordinal: usize,
) -> Result<DeviceState, String> {
    let b = Backend::parse(&backend).ok_or_else(|| format!("unknown backend {backend:?}"))?;
    state.select(b, ordinal).map_err(err)?;
    Ok(device_state(&state))
}

#[tauri::command]
pub async fn classify_image(
    state: State<'_, Arc<AppState>>,
    image_base64: String,
    top_k: usize,
) -> Result<Vec<Prediction>, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Vec<Prediction>, String> {
        let data = image_base64
            .split(',')
            .next_back()
            .unwrap_or_default()
            .trim();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(err)?;
        let img = image::load_from_memory(&bytes).map_err(err)?;
        let engine = shared.vision_engine().map_err(err)?;
        classify(&engine.model, &engine.anchors, &engine.dict, &img, top_k.clamp(1, 28)).map_err(err)
    })
    .await
    .map_err(err)?
}

#[derive(Serialize)]
pub struct PathsView {
    pub models_dir: String,
    pub store_dir: String,
}

#[tauri::command]
pub fn default_paths() -> Result<PathsView, String> {
    Ok(PathsView {
        models_dir: model_dir().map_err(err)?.display().to_string(),
        store_dir: default_store_dir().display().to_string(),
    })
}

fn store_path(store_dir: Option<String>) -> PathBuf {
    store_dir
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_store_dir)
}

#[derive(Serialize)]
pub struct SynthSummary {
    pub episodes: usize,
    pub stable: usize,
    pub store_dir: String,
}

/// Simulates `count` randomized episodes and appends them to the Lance store.
#[tauri::command]
pub async fn synthesize(
    count: usize,
    seed: u64,
    label: String,
    store_dir: Option<String>,
) -> Result<SynthSummary, String> {
    let dir = store_path(store_dir);
    let count = count.clamp(1, 5000);
    let (samples, contacts, joints, stable) = tauri::async_runtime::spawn_blocking(
        move || -> Result<_, String> {
            let (mut samples, mut contacts, mut joints) = (Vec::new(), Vec::new(), Vec::new());
            let mut stable = 0;
            for i in 0..count {
                let scene = random_scene(seed.wrapping_add(i as u64), &label);
                let rows = simulate_scene(&scene).map_err(err)?;
                stable += usize::from(rows.stable);
                samples.push(rows.samples);
                contacts.push(rows.contacts);
                joints.push(rows.joint_states);
            }
            Ok((samples, contacts, joints, stable))
        },
    )
    .await
    .map_err(err)??;

    let store = LanceStore::open(dir.to_str().ok_or("store path is not UTF-8")?)
        .await
        .map_err(err)?;
    store.append_all(TABLE_SAMPLES, samples).await.map_err(err)?;
    store.append_all(TABLE_CONTACTS, contacts).await.map_err(err)?;
    store.append_all(TABLE_JOINT_STATES, joints).await.map_err(err)?;
    Ok(SynthSummary {
        episodes: count,
        stable,
        store_dir: dir.display().to_string(),
    })
}

/// Writes the shipped logistics dictionary to the Lance `dictionary` table, replacing it.
#[tauri::command]
pub async fn load_dictionary(store_dir: Option<String>) -> Result<usize, String> {
    let dir = store_path(store_dir);
    let batch = dictionary_batch(DICTIONARY_JSON).map_err(err)?;
    let rows = batch.num_rows();
    let store = LanceStore::open(dir.to_str().ok_or("store path is not UTF-8")?)
        .await
        .map_err(err)?;
    store
        .replace_with(TABLE_DICTIONARY, vec![batch])
        .await
        .map_err(err)?;
    Ok(rows)
}

#[derive(Serialize)]
pub struct TableStatus {
    pub name: &'static str,
    pub rows: usize,
    /// `recorded` when the simulator writes it now, `planned` otherwise.
    pub status: &'static str,
}

#[tauri::command]
pub async fn dataset_summary(store_dir: Option<String>) -> Result<Vec<TableStatus>, String> {
    let dir = store_path(store_dir);
    let store = LanceStore::open(dir.to_str().ok_or("store path is not UTF-8")?)
        .await
        .map_err(err)?;
    let recorded = [TABLE_SAMPLES, TABLE_CONTACTS, TABLE_JOINT_STATES, TABLE_DICTIONARY];
    let planned = [TABLE_POINT_CLOUD, TABLE_VISION, TABLE_TACTILE, TABLE_EMBEDDINGS];
    let mut out = Vec::new();
    for name in recorded {
        out.push(TableStatus {
            name,
            rows: store.row_count(name).await.map_err(err)?,
            status: "recorded",
        });
    }
    for name in planned {
        out.push(TableStatus {
            name,
            rows: 0,
            status: "planned",
        });
    }
    Ok(out)
}
