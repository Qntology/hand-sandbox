//! Runs the simulated robot: recognise, gate, quest, log. See `docs/PLAN.md`.
//!
//! * [`perception`]: the recogniser interface and the stand-in that reads the
//!   renderer's ground truth and spoils it on purpose.
//! * [`precision`]: the gates between a detection and an action (score floor
//!   learned by the SDS, margin, time-series consistency, label conflicts,
//!   touch results, repeated sightings).
//! * [`runtime::Runtime`]: world, recogniser, gates, SDS, audit log and quests
//!   together; `driver` carries the quests out.

pub mod api;
mod api_test;
pub mod config;
mod driver;
pub mod embedding;
pub mod ml;
mod mldrive;
mod mlsds;
pub mod perception;
pub mod precision;
pub mod record;
pub mod runtime;
pub mod testrun;

pub use config::{CameraCfg, GateCfg, NoiseCfg, RuntimeConfig};
pub use embedding::{BankedPhrases, EmbeddingCfg, EmbeddingPerception, SiglipPhrases};
pub use perception::{Candidate, EntKind, EntityTable, OraclePerception, Recognizer, ScanInput, ScanOutput};
pub use precision::{GateCounters, GateResult, Judged, Landmark, LmState, PrecisionGates};
pub use ml::{DepthEngine, LocateEngine, MlPipeline};
pub use record::{RecordCfg, Recording};
pub use testrun::{TestCfg, TestPhase, TestSession};
pub use runtime::{label_ko, Detection, EvalCounters, EvalSummary, FrameOut, FrameView, Runtime, ScanReport, Submission, TsView};
