//! Time-series forecasting for score histories.
//!
//! * [`holt`]: damped Holt trend (statistical fallback).
//! * [`fusion`]: holdout backtest of several predictors with inverse-variance
//!   weights that also use the errors the SDS remembers.
//! * [`gate`]: temporal consistency gate built on the fused forecast.
//! * `ttm` (feature `granite`): Granite TTM-R3 on candle, copied from
//!   endmill-sandbox, and [`ttm_predictor`] to plug it into the fusion.
//!
//! This crate does not depend on the SDS crate; remembered errors come in as
//! plain [`HistoryMse`] values.

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
