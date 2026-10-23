//! Tauri commands of the simulated world. Each one hands its arguments to
//! `hand_runtime::Runtime::api_*`, which builds the JSON the UI reads. Nothing
//! is decided here.
//!
//! The commands are `async` so that a long call (a few hundred ticks, a frame
//! encode) runs on the async runtime and not on the window's main thread.

use std::sync::Arc;

use hand_core::HandConfig;
use hand_runtime::Runtime;
use serde_json::{json, Value};
use tauri::State;

use crate::state::{bank_dir, model_dir, ttm_dir, AppState};

type App<'a> = State<'a, Arc<AppState>>;

#[tauri::command]
pub async fn world_new(state: App<'_>, seed: u64, size: usize, config: Option<HandConfig>, base: Option<String>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_world_new(seed, size, config, base.as_deref()))
}

#[tauri::command]
pub async fn sim_step(state: App<'_>, ticks: u32) -> Result<Value, String> {
    state.with_world(|rt| rt.api_sim_step(ticks))
}

#[tauri::command]
pub async fn world_frame(state: App<'_>, width: u32, height: u32, quality: f32, view: Option<String>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_world_frame(width, height, quality, view.as_deref().unwrap_or("agent")))
}

#[tauri::command]
pub async fn command_submit(state: App<'_>, text: String) -> Result<Value, String> {
    state.with_world(|rt| rt.api_command_submit(&text))
}

#[tauri::command]
pub async fn quest_list(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_quest_list())
}

#[tauri::command]
pub async fn perception_scan(state: App<'_>, top_k: usize) -> Result<Value, String> {
    state.with_world(|rt| rt.api_perception_scan(top_k))
}

#[tauri::command]
pub async fn body_status(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_body_status())
}

#[tauri::command]
pub async fn sds_status(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_sds_status())
}

#[tauri::command]
pub async fn ts_forecast(state: App<'_>, scope: String, axis: String) -> Result<Value, String> {
    state.with_world(|rt| rt.api_ts_forecast(&scope, &axis))
}

#[tauri::command]
pub async fn log_tail(state: App<'_>, after_seq: u64, limit: usize, category: Option<String>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_log_tail(after_seq, limit, category.as_deref()))
}

/// What `models_status` and the two model switches answer: the runtime's own
/// view plus what only the host knows (which device, which folders, whether the
/// files are there).
fn models_json(state: &AppState, rt: &Runtime) -> Result<Value, String> {
    let mut v = rt.api_models_status()?;
    let (backend, ordinal) = state.selected();
    let siglip = model_dir().map_err(|e| format!("{e:#}"))?;
    let ttm = ttm_dir();
    v["device"] = json!({"backend": backend.name(), "ordinal": ordinal});
    v["siglip_files"] = json!({"dir": siglip.display().to_string(), "present": siglip.join("config.json").is_file()});
    v["ttm_files"] = json!({"dir": ttm.display().to_string(), "present": ttm.join("config.json").is_file()});
    Ok(v)
}

#[tauri::command]
pub async fn models_status(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| models_json(&state, rt))
}

/// `mode` is `"embedding"` (SigLIP2 on the screenshot tiles) or `"oracle"` (the
/// stand-in that reads the ground-truth id buffer). Loading the model and
/// embedding the phrase bank can take a while the first time, so it runs off the
/// window's thread.
#[tauri::command]
pub async fn recognizer_use(state: App<'_>, mode: String) -> Result<Value, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        match mode.as_str() {
            "embedding" => {
                let engine = shared.vision_engine().map_err(|e| format!("SigLIP2 를 불러오지 못했습니다 ({}): {e:#}", model_dir().map(|p| p.display().to_string()).unwrap_or_default()))?;
                shared.with_world(|rt| {
                    rt.use_siglip(Arc::clone(&engine.model), Some(&bank_dir())).map_err(|e| format!("{e:#}"))?;
                    models_json(&shared, rt)
                })
            }
            "oracle" => shared.with_world(|rt| {
                rt.use_oracle();
                models_json(&shared, rt)
            }),
            other => Err(format!("모르는 인식기 {other}: embedding 또는 oracle")),
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Switches Granite TTM on (loads it from [`ttm_dir`] onto the selected device)
/// or off.
#[tauri::command]
pub async fn ttm_use(state: App<'_>, enabled: bool) -> Result<Value, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        if !enabled {
            return shared.with_world(|rt| {
                rt.attach_ttm(None);
                models_json(&shared, rt)
            });
        }
        let (backend, ordinal) = shared.selected();
        let device = backend.open(ordinal).map_err(|e| format!("{} 장치를 열 수 없습니다: {e:#}", backend.name()))?;
        let dir = ttm_dir();
        shared.with_world(|rt| {
            rt.load_ttm(&dir, &device).map_err(|e| format!("{e:#}"))?;
            models_json(&shared, rt)
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
