pub mod commands;
pub mod models;
pub mod sim;
pub mod state;
pub mod views;

use std::sync::Arc;

/// Where an AI client reaches the robots (`noto_runtime::mcp`), on this machine only. `NOTO_MCP_PORT` moves it.
const MCP_PORT: u16 = 8788;

pub fn run() {
    let app_state = Arc::new(state::AppState::new());
    let mcp_state = app_state.clone();
    let port = std::env::var("NOTO_MCP_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(MCP_PORT);
    let call = Arc::new(move |name: &str, args: &serde_json::Value| {
        if name == "command_submit" {
            mcp_state.require_models()?;
        }
        mcp_state.with_world(|rt| rt.api_call(name, args))
    });
    if let Err(e) = noto_runtime::mcp::listen(port, call) {
        eprintln!("MCP 서버를 열 수 없습니다 (127.0.0.1:{port}): {e}");
    }
    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::presets,
            commands::hand_geometry,
            commands::grasp_test,
            commands::drive_courses,
            commands::dictionary,
            commands::devices,
            commands::select_device,
            commands::classify_image,
            commands::default_paths,
            commands::synthesize,
            commands::load_dictionary,
            commands::dataset_summary,
            sim::world_new,
            sim::sim_step,
            sim::world_frame,
            sim::command_submit,
            sim::quest_list,
            sim::perception_scan,
            sim::body_status,
            sim::sds_status,
            sim::ts_forecast,
            sim::log_tail,
            sim::models_status,
            sim::recognizer_use,
            sim::ttm_use,
            sim::rt_call,
            sim::test_db_mirror,
            sim::tips_use,
            sim::sam_use,
            models::models_catalog,
            models::models_download,
            models::models_cancel,
            models::models_delete,
            models::models_activate,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the Noto logistics robot simulator")
        .run(|_app, event| {
            if let tauri::RunEvent::Exit = event {
                noto_ml::device::rocm_exit_now_if_used(0);
            }
        });
}
