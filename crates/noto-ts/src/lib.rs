pub mod fusion;
pub mod gate;
pub mod holt;

#[cfg(feature = "granite")]
pub mod io;
#[cfg(feature = "granite")]
pub mod ttm;
#[cfg(feature = "granite")]
mod ttm_predictor;

pub use fusion::{forecast, Band, EwmaPredictor, Forecast, HistoryMse, HoltPredictor, LastPredictor, MethodWeight, Predictor};
pub use gate::{temporal_gate, GateConfig, TemporalOutcome, TemporalVerdict};
pub use holt::{holt_damped, HoltFit};
#[cfg(feature = "granite")]
pub use ttm_predictor::TtmPredictor;
