use std::sync::Arc;

use noto_core::{BodyPlan, NotoConfig};
use noto_runtime::Runtime;
use serde_json::{json, Value};
use tauri::State;

use crate::state::{bank_dir, default_store_dir, model_dir, sam_dir, tips_dir, ttm_dir, AppState};

type App<'a> = State<'a, Arc<AppState>>;

#[tauri::command]
pub async fn world_new(state: App<'_>, seed: u64, size: usize, config: Option<NotoConfig>, base: Option<String>, body: Option<BodyPlan>, terrain: Option<String>) -> Result<Value, String> {
    state.require_models()?;
    state.with_world(|rt| {
        if let Some(b) = body {
            rt.cfg.body = b;
        }
        if let Some(t) = terrain.as_deref().filter(|t| !t.is_empty()) {
            rt.cfg.terrain = noto_runtime::world_terrain_kind(t)?;
        }
        rt.api_world_new(seed, size, config, base.as_deref())
    })
}

const RUNS_WORLD: [&str; 2] = ["test_start", "test_step"];

#[tauri::command]
pub async fn rt_call(state: App<'_>, cmd: String, args: Option<Value>) -> Result<Value, String> {
    if RUNS_WORLD.contains(&cmd.as_str()) {
        state.require_models()?;
    }
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || shared.with_world(|rt| rt.api_call(&cmd, &args.unwrap_or(Value::Null))))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn test_db_mirror(state: App<'_>, store_dir: Option<String>) -> Result<Value, String> {
    let (batch, stats) = state.with_world(|rt| {
        let db = rt.test_db_mut();
        Ok((db.take_mirror(), db.stats()))
    })?;
    let dir = store_dir.filter(|s| !s.trim().is_empty()).map(std::path::PathBuf::from).unwrap_or_else(default_store_dir);
    let store = noto_data::LanceStore::open(dir.to_str().ok_or("저장 경로가 UTF-8 이 아닙니다")?).await.map_err(|e| format!("{e:#}"))?;
    let counts = noto_data::test_tables::write_mirror(&store, &batch, &stats).await.map_err(|e| format!("{e:#}"))?;
    Ok(json!({"store_dir": dir.display().to_string(), "written": counts}))
}

#[tauri::command]
pub async fn tips_use(state: App<'_>, enabled: Option<bool>) -> Result<Value, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        if enabled == Some(false) {
            return shared.with_world(|rt| {
                rt.use_depth_stand_in();
                models_json(&shared, rt)
            });
        }
        let device = selected_device(&shared)?;
        let dir = tips_dir();
        shared.with_world(|rt| {
            rt.use_tips(&dir, &device).map_err(|e| format!("{e:#}"))?;
            models_json(&shared, rt)
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn sam_use(state: App<'_>, enabled: Option<bool>) -> Result<Value, String> {
    let shared: Arc<AppState> = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<Value, String> {
        if enabled == Some(false) {
            return shared.with_world(|rt| {
                rt.use_region_tracker();
                models_json(&shared, rt)
            });
        }
        let device = selected_device(&shared)?;
        let dir = sam_dir();
        shared.with_world(|rt| {
            rt.use_sam(&dir, &device).map_err(|e| format!("{e:#}"))?;
            models_json(&shared, rt)
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

fn selected_device(state: &AppState) -> Result<candle_core::Device, String> {
    let (backend, ordinal) = state.selected();
    state.device().map_err(|e| format!("{}:{ordinal} 장치를 열 수 없습니다: {e:#}", backend.name()))
}

#[tauri::command]
pub async fn sim_step(state: App<'_>, ticks: u32) -> Result<Value, String> {
    state.require_models()?;
    state.with_world(|rt| rt.api_sim_step(ticks))
}

#[tauri::command]
pub async fn world_frame(state: App<'_>, width: u32, height: u32, quality: f32, view: Option<String>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_world_frame(width, height, quality, view.as_deref().unwrap_or("agent")))
}

#[tauri::command]
pub async fn command_submit(state: App<'_>, text: String, record: Option<bool>) -> Result<Value, String> {
    state.require_models()?;
    state.with_world(|rt| rt.api_command_submit_rec(&text, record.unwrap_or(false)))
}

#[tauri::command]
pub async fn quest_list(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| rt.api_quest_list())
}

#[tauri::command]
pub async fn perception_scan(state: App<'_>, top_k: usize) -> Result<Value, String> {
    state.require_models()?;
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

pub(crate) fn all_live(rt: &Runtime) -> bool {
    rt.stand_ins().is_empty()
}

pub(crate) fn models_json(state: &AppState, rt: &Runtime) -> Result<Value, String> {
    if state.models_ready() && !all_live(rt) {
        state.set_models_ready(false);
    }
    let mut v = rt.api_models_status()?;
    let (backend, ordinal) = state.selected();
    let siglip = model_dir().map_err(|e| format!("{e:#}"))?;
    let ttm = ttm_dir();
    v["device"] = json!({"backend": backend.name(), "ordinal": ordinal});
    v["siglip_files"] = json!({"dir": siglip.display().to_string(), "present": siglip.join("config.json").is_file()});
    v["ttm_files"] = json!({"dir": ttm.display().to_string(), "present": ttm.join("config.json").is_file()});
    let tips = tips_dir();
    v["tips_files"] = json!({"dir": tips.display().to_string(), "present": tips.join("model.safetensors").is_file()});
    let sam = sam_dir();
    v["sam_files"] = json!({"dir": sam.display().to_string(), "present": noto_vision::SamModel::find_weights(&sam).is_some()});
    let s = state.startup();
    v["startup"] = json!({"backend": s.backend.name(), "ordinal": s.ordinal, "reason": s.reason});
    v["ready"] = json!(state.models_ready());
    Ok(v)
}

#[tauri::command]
pub async fn models_status(state: App<'_>) -> Result<Value, String> {
    state.with_world(|rt| models_json(&state, rt))
}

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
        let device = selected_device(&shared)?;
        let dir = ttm_dir();
        shared.with_world(|rt| {
            rt.load_ttm(&dir, &device).map_err(|e| format!("{e:#}"))?;
            models_json(&shared, rt)
        })
    })
    .await
    .map_err(|e| e.to_string())?
}
