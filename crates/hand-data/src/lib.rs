//! Layered synthetic dataset for the logistics robot hand.
//!
//! * [`layers`] lists the planned data layers and whether the simulator
//!   records each one now, needs a small extension, or needs a renderer or a
//!   sensor model first.
//! * [`scene`] draws episodes with domain randomization from a seed.
//! * [`generate`] runs the grasp simulation and produces Arrow rows.
//! * [`store`] writes those rows to LanceDB, the ML-native columnar format.

pub mod dictionary;
pub mod generate;
pub mod layers;
pub mod scene;
pub mod schema;
pub mod store;

pub use dictionary::dictionary_batch;
pub use generate::{simulate_scene, SampleRows, RECORDED_LAYERS};
pub use layers::{Group, Layer, Readiness};
pub use scene::{random_scene, SceneSpec, SplitMix64, MATERIALS};
pub use store::LanceStore;
