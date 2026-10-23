//! Lance tables of the integrated test records (`hand_sds::records`).
//!
//! * `quest_runs`: one row per quest run (game or ML pass);
//! * `quest_steps`: one row per step of a run, with its attempts;
//! * `quest_stats`: attempts, successes and success rate per
//!   `pass|kind|target|preset|step`, replaced on every write;
//! * `ml_keyframes`: one row per analysed keyframe;
//! * `balance_effort`: one row per N:N balance cell and state, with the
//!   effort and margin with and without the learned feed-forward.
//!
//! The runtime writes the same rows to JSONL files; these tables are the
//! queryable copy.

use std::sync::Arc;

use anyhow::Result;
use hand_sds::{BalanceRow, KeyframeRow, MirrorBatch, QuestRun, SuccessStat};
use lancedb::arrow::arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::store::LanceStore;

pub const TABLE_QUEST_RUNS: &str = "quest_runs";
pub const TABLE_QUEST_STEPS: &str = "quest_steps";
pub const TABLE_QUEST_STATS: &str = "quest_stats";
pub const TABLE_ML_KEYFRAMES: &str = "ml_keyframes";
pub const TABLE_BALANCE_EFFORT: &str = "balance_effort";

fn s(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}
fn so(name: &str) -> Field {
    Field::new(name, DataType::Utf8, true)
}
fn f(name: &str) -> Field {
    Field::new(name, DataType::Float64, false)
}
fn fo(name: &str) -> Field {
    Field::new(name, DataType::Float64, true)
}
fn u64f(name: &str) -> Field {
    Field::new(name, DataType::UInt64, false)
}

pub fn quest_runs_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        s("run_id"),
        s("session"),
        s("pass"),
        s("quest_id"),
        s("kind"),
        s("target"),
        so("destination"),
        s("preset"),
        u64f("seed"),
        Field::new("ok", DataType::Boolean, false),
        so("failure"),
        u64f("ticks"),
        f("time_s"),
        f("energy_j"),
        Field::new("created_ms", DataType::Int64, false),
    ]))
}

pub fn quest_steps_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        s("run_id"),
        s("pass"),
        s("kind"),
        s("target"),
        s("preset"),
        Field::new("step_index", DataType::UInt32, false),
        s("step"),
        Field::new("attempts", DataType::UInt32, false),
        Field::new("ok", DataType::Boolean, false),
        so("fail_code"),
    ]))
}

pub fn quest_stats_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![s("key"), s("pass"), s("kind"), s("target"), s("preset"), s("step"), u64f("attempts"), u64f("successes"), f("rate"), Field::new("last_ms", DataType::Int64, false)]))
}

pub fn ml_keyframes_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        s("run_id"),
        s("pass"),
        Field::new("index", DataType::UInt32, false),
        u64f("tick"),
        f("time_s"),
        s("target"),
        s("depth_mode"),
        fo("depth_absrel"),
        fo("depth_delta1"),
        s("locate_mode"),
        fo("locate_score"),
        fo("locate_err_px"),
        s("track_mode"),
        Field::new("track_id", DataType::UInt32, true),
        fo("track_iou_prev"),
        fo("track_iou_truth"),
        fo("tri_range_m"),
        fo("tri_parallax_deg"),
        fo("depth_scale"),
        fo("range_used_m"),
        fo("range_true_m"),
        s("decision"),
        s("ts_method"),
        fo("ts_residual"),
        s("fusion_flags"),
        fo("balance_margin"),
        fo("effort_max"),
    ]))
}

pub fn balance_effort_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        s("run_id"),
        s("pass"),
        u64f("tick"),
        s("source"),
        s("source_seg"),
        s("target"),
        s("state"),
        s("preset"),
        f("comp"),
        s("comp_unit"),
        fo("reference"),
        fo("tolerance"),
        f("torque_nm"),
        f("effort"),
        f("effort_noff"),
        f("margin"),
        f("margin_noff"),
        f("ff_gain"),
    ]))
}

fn strs<T>(rows: &[T], f: impl Fn(&T) -> String) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(rows.iter().map(f)))
}
fn ostrs<T>(rows: &[T], f: impl Fn(&T) -> Option<String>) -> ArrayRef {
    Arc::new(StringArray::from(rows.iter().map(f).collect::<Vec<Option<String>>>()))
}
fn f64s<T>(rows: &[T], f: impl Fn(&T) -> f64) -> ArrayRef {
    Arc::new(Float64Array::from(rows.iter().map(f).collect::<Vec<f64>>()))
}
fn of64s<T>(rows: &[T], f: impl Fn(&T) -> Option<f64>) -> ArrayRef {
    Arc::new(Float64Array::from(rows.iter().map(f).collect::<Vec<Option<f64>>>()))
}
fn u64s<T>(rows: &[T], f: impl Fn(&T) -> u64) -> ArrayRef {
    Arc::new(UInt64Array::from(rows.iter().map(f).collect::<Vec<u64>>()))
}

pub fn quest_runs_batch(rows: &[QuestRun]) -> Result<RecordBatch> {
    let cols: Vec<ArrayRef> = vec![
        strs(rows, |r| r.run_id.clone()),
        strs(rows, |r| r.session.clone()),
        strs(rows, |r| r.pass.clone()),
        strs(rows, |r| r.quest_id.clone()),
        strs(rows, |r| r.kind.clone()),
        strs(rows, |r| r.target.clone()),
        ostrs(rows, |r| r.destination.clone()),
        strs(rows, |r| r.preset.clone()),
        u64s(rows, |r| r.seed),
        Arc::new(BooleanArray::from(rows.iter().map(|r| r.ok).collect::<Vec<bool>>())),
        ostrs(rows, |r| r.failure.clone()),
        u64s(rows, |r| r.ticks),
        f64s(rows, |r| r.time_s),
        f64s(rows, |r| r.energy_j),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.created_ms).collect::<Vec<i64>>())),
    ];
    Ok(RecordBatch::try_new(quest_runs_schema(), cols)?)
}

pub fn quest_steps_batch(rows: &[QuestRun]) -> Result<RecordBatch> {
    let flat: Vec<(&QuestRun, usize)> = rows.iter().flat_map(|r| (0..r.steps.len()).map(move |i| (r, i))).collect();
    let cols: Vec<ArrayRef> = vec![
        strs(&flat, |(r, _)| r.run_id.clone()),
        strs(&flat, |(r, _)| r.pass.clone()),
        strs(&flat, |(r, _)| r.kind.clone()),
        strs(&flat, |(r, _)| r.target.clone()),
        strs(&flat, |(r, _)| r.preset.clone()),
        Arc::new(UInt32Array::from(flat.iter().map(|(_, i)| *i as u32).collect::<Vec<u32>>())),
        strs(&flat, |(r, i)| r.steps[*i].step.clone()),
        Arc::new(UInt32Array::from(flat.iter().map(|(r, i)| r.steps[*i].attempts).collect::<Vec<u32>>())),
        Arc::new(BooleanArray::from(flat.iter().map(|(r, i)| r.steps[*i].ok).collect::<Vec<bool>>())),
        ostrs(&flat, |(r, i)| r.steps[*i].fail_code.clone()),
    ];
    Ok(RecordBatch::try_new(quest_steps_schema(), cols)?)
}

pub fn quest_stats_batch(rows: &[SuccessStat]) -> Result<RecordBatch> {
    let cols: Vec<ArrayRef> = vec![
        strs(rows, |r| r.key.clone()),
        strs(rows, |r| r.pass.clone()),
        strs(rows, |r| r.kind.clone()),
        strs(rows, |r| r.target.clone()),
        strs(rows, |r| r.preset.clone()),
        strs(rows, |r| r.step.clone()),
        u64s(rows, |r| r.attempts),
        u64s(rows, |r| r.successes),
        f64s(rows, |r| r.rate),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.last_ms).collect::<Vec<i64>>())),
    ];
    Ok(RecordBatch::try_new(quest_stats_schema(), cols)?)
}

pub fn ml_keyframes_batch(rows: &[KeyframeRow]) -> Result<RecordBatch> {
    let cols: Vec<ArrayRef> = vec![
        strs(rows, |r| r.run_id.clone()),
        strs(rows, |r| r.pass.clone()),
        Arc::new(UInt32Array::from(rows.iter().map(|r| r.index).collect::<Vec<u32>>())),
        u64s(rows, |r| r.tick),
        f64s(rows, |r| r.time_s),
        strs(rows, |r| r.target.clone()),
        strs(rows, |r| r.depth_mode.clone()),
        of64s(rows, |r| r.depth_absrel),
        of64s(rows, |r| r.depth_delta1),
        strs(rows, |r| r.locate_mode.clone()),
        of64s(rows, |r| r.locate_score),
        of64s(rows, |r| r.locate_err_px),
        strs(rows, |r| r.track_mode.clone()),
        Arc::new(UInt32Array::from(rows.iter().map(|r| r.track_id).collect::<Vec<Option<u32>>>())),
        of64s(rows, |r| r.track_iou_prev),
        of64s(rows, |r| r.track_iou_truth),
        of64s(rows, |r| r.tri_range_m),
        of64s(rows, |r| r.tri_parallax_deg),
        of64s(rows, |r| r.depth_scale),
        of64s(rows, |r| r.range_used_m),
        of64s(rows, |r| r.range_true_m),
        strs(rows, |r| r.decision.clone()),
        strs(rows, |r| r.ts_method.clone()),
        of64s(rows, |r| r.ts_residual),
        strs(rows, |r| r.fusion_flags.clone()),
        of64s(rows, |r| r.balance_margin),
        of64s(rows, |r| r.effort_max),
    ];
    Ok(RecordBatch::try_new(ml_keyframes_schema(), cols)?)
}

pub fn balance_effort_batch(rows: &[BalanceRow]) -> Result<RecordBatch> {
    let cols: Vec<ArrayRef> = vec![
        strs(rows, |r| r.run_id.clone()),
        strs(rows, |r| r.pass.clone()),
        u64s(rows, |r| r.tick),
        strs(rows, |r| r.source.clone()),
        strs(rows, |r| r.source_seg.clone()),
        strs(rows, |r| r.target.clone()),
        strs(rows, |r| r.state.clone()),
        strs(rows, |r| r.preset.clone()),
        f64s(rows, |r| r.comp),
        strs(rows, |r| r.comp_unit.clone()),
        of64s(rows, |r| r.reference),
        of64s(rows, |r| r.tolerance),
        f64s(rows, |r| r.torque_nm),
        f64s(rows, |r| r.effort),
        f64s(rows, |r| r.effort_noff),
        f64s(rows, |r| r.margin),
        f64s(rows, |r| r.margin_noff),
        f64s(rows, |r| r.ff_gain),
    ];
    Ok(RecordBatch::try_new(balance_effort_schema(), cols)?)
}

/// Rows written per table.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MirrorCounts {
    pub quest_runs: usize,
    pub quest_steps: usize,
    pub quest_stats: usize,
    pub ml_keyframes: usize,
    pub balance_effort: usize,
}

/// Appends the new rows and replaces the success table.
pub async fn write_mirror(store: &LanceStore, batch: &MirrorBatch, stats: &[SuccessStat]) -> Result<MirrorCounts> {
    let mut c = MirrorCounts::default();
    if !batch.runs.is_empty() {
        store.append(TABLE_QUEST_RUNS, quest_runs_batch(&batch.runs)?).await?;
        let steps = quest_steps_batch(&batch.runs)?;
        c.quest_steps = steps.num_rows();
        store.append(TABLE_QUEST_STEPS, steps).await?;
        c.quest_runs = batch.runs.len();
    }
    if !batch.keyframes.is_empty() {
        store.append(TABLE_ML_KEYFRAMES, ml_keyframes_batch(&batch.keyframes)?).await?;
        c.ml_keyframes = batch.keyframes.len();
    }
    if !batch.balance.is_empty() {
        store.append(TABLE_BALANCE_EFFORT, balance_effort_batch(&batch.balance)?).await?;
        c.balance_effort = batch.balance.len();
    }
    if !stats.is_empty() {
        store.replace_with(TABLE_QUEST_STATS, vec![quest_stats_batch(stats)?]).await?;
        c.quest_stats = stats.len();
    }
    Ok(c)
}
