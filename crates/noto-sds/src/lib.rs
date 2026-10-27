//! Score Dynamics Store and audit log.
//!
//! * [`SdsStore`]: per-scope online statistics with backoff, calibration and
//!   atomic ledger files (port of logis-center `score_dynamics.rs`).
//! * [`AuditLog`]: a hash-chained JSONL log that the UI can tail.
//! * [`TestDb`]: the rows behind the statistics (quest runs, ML keyframes,
//!   balance effort) and the success table per quest step.
//!
//! Nothing in this crate decides anything. Callers ask for a learned value and
//! use their fixed default when the answer is `None`.

pub mod audit;
pub mod records;
pub mod scope;
pub mod stats;
pub mod store;

pub use audit::{verify_file, AuditLog, Level, LogLine, VerifyError, VerifyOk};
pub use records::{BalanceRow, KeyframeRow, MirrorBatch, QuestRun, StepRecord, SuccessStat, TestDb, WHOLE_QUEST};
pub use scope::{Ledger, Scope, Track};
pub use stats::{decay_shape, DecayShape, ScopeStat, Welford};
pub use store::{
    write_atomic, AreaAxis, AreaReport, AxisStatus, ConfusionStatus, GateRate, LedgerStatus, OpenOptions, OtherRevs, ReferenceReport, ScopePolicy, SdsStatus, SdsStore,
    SDS_RECIPE, UNSTAMPED,
};
