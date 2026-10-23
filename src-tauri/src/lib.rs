//! Logistics robot hand simulator: Tauri shell over `hand-core`, `hand-ml`
//! and `hand-data`.

pub mod commands;
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running the logistics hand simulator");
}
