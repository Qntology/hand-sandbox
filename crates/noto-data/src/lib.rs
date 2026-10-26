//! Layered synthetic dataset for the Noto logistics robot.
//!
//! * [`layers`] lists the planned data layers and whether the simulator
//!   records each one now, needs a small extension, or needs a renderer or a
//!   sensor model first.
//! * [`scene`] draws episodes with domain randomization from a seed.
//! * [`generate`] runs the grasp simulation and produces Arrow rows.
//! * [`store`] writes those rows to LanceDB, the ML-native columnar format.
//! * [`test_tables`] holds the integrated test's records (quest runs and steps,
//!   success table, ML keyframes, balance effort).

pub mod dictionary;
pub mod generate;
pub mod layers;
pub mod scene;
pub mod schema;
pub mod store;
pub mod test_tables;

pub use dictionary::dictionary_batch;
pub use generate::{simulate_scene, SampleRows, RECORDED_LAYERS};
pub use layers::{Group, Layer, Readiness};
pub use scene::{random_scene, SceneSpec, SplitMix64, MATERIALS};
pub use store::LanceStore;
