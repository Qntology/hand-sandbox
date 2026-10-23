//! Logistics robot hand simulator: Tauri shell over `hand-core`, `hand-ml`,
//! `hand-data` and the simulated world (`hand-runtime`).

pub mod commands;
pub mod sim;
pub mod state;
pub mod views;

use std::sync::Arc;

pub fn run() {
    tauri::Builder::default()
        .manage(Arc::new(state::AppState::new()))
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running the logistics hand simulator");
}
