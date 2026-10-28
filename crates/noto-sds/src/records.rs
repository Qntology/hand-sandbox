use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::store::{now_ms, write_atomic};

const KEEP_ROWS: usize = 4000;

pub const FILE_RUNS: &str = "quest_runs.jsonl";
pub const FILE_KEYFRAMES: &str = "keyframes.jsonl";
pub const FILE_BALANCE: &str = "balance_effort.jsonl";
pub const FILE_STATS: &str = "quest_stats.json";

pub const WHOLE_QUEST: &str = "quest";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: String,
    pub attempts: u32,
    pub ok: bool,
    pub fail_code: Option<String>,
    pub time_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestRun {
    pub run_id: String,
    pub session: String,
    pub pass: String,
    pub quest_id: String,
    pub kind: String,
    pub target: String,
    pub destination: Option<String>,
    pub preset: String,
    pub seed: u64,
    pub ok: bool,
    pub failure: Option<String>,
    pub ticks: u64,
    pub time_s: f64,
    pub energy_j: f64,
    pub steps: Vec<StepRecord>,
    pub created_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyframeRow {
    pub run_id: String,
    pub pass: String,
    pub index: u32,
    pub tick: u64,
    pub time_s: f64,
    pub target: String,
    pub depth_mode: String,
    pub depth_absrel: Option<f64>,
    pub depth_delta1: Option<f64>,
    pub locate_mode: String,
    pub locate_score: Option<f64>,
    pub locate_err_px: Option<f64>,
    pub track_mode: String,
    pub track_id: Option<u32>,
    pub track_iou_prev: Option<f64>,
    pub track_iou_truth: Option<f64>,
    pub tri_range_m: Option<f64>,
    pub tri_parallax_deg: Option<f64>,
    pub depth_scale: Option<f64>,
    pub range_used_m: Option<f64>,
    pub range_true_m: Option<f64>,
    pub decision: String,
    pub ts_method: String,
    pub ts_residual: Option<f64>,
    pub fusion_flags: String,
    pub balance_margin: Option<f64>,
    pub effort_max: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BalanceRow {
    pub run_id: String,
    pub pass: String,
    pub tick: u64,
    pub source: String,
    pub source_seg: String,
    pub target: String,
    pub state: String,
    pub preset: String,
    pub comp: f64,
    pub comp_unit: String,
    pub reference: Option<f64>,
    pub tolerance: Option<f64>,
    pub torque_nm: f64,
    pub effort: f64,
    pub effort_noff: f64,
    pub margin: f64,
    pub margin_noff: f64,
    pub ff_gain: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuccessStat {
    pub key: String,
    pub pass: String,
    pub kind: String,
    pub target: String,
    pub preset: String,
    pub step: String,
    pub attempts: u64,
    pub successes: u64,
    pub rate: f64,
    pub last_ms: i64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct StatsFile {
    updated_at: i64,
    stats: Vec<SuccessStat>,
}

#[derive(Clone, Debug, Default)]
pub struct MirrorBatch {
    pub runs: Vec<QuestRun>,
    pub keyframes: Vec<KeyframeRow>,
    pub balance: Vec<BalanceRow>,
}

impl MirrorBatch {
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty() && self.keyframes.is_empty() && self.balance.is_empty()
    }
}

pub struct TestDb {
    dir: Option<PathBuf>,
    runs: Vec<QuestRun>,
    keyframes: Vec<KeyframeRow>,
    balance: Vec<BalanceRow>,
    stats: BTreeMap<String, SuccessStat>,
    pending: BTreeMap<&'static str, Vec<String>>,
    mirror: MirrorBatch,
    stats_dirty: bool,
}

fn push_capped<T>(v: &mut Vec<T>, row: T) {
    if v.len() >= KEEP_ROWS {
        v.drain(0..KEEP_ROWS / 4);
    }
    v.push(row);
}

impl TestDb {
    pub fn memory() -> TestDb {
        TestDb {
            dir: None,
            runs: Vec::new(),
            keyframes: Vec::new(),
            balance: Vec::new(),
            stats: BTreeMap::new(),
            pending: BTreeMap::new(),
            mirror: MirrorBatch::default(),
            stats_dirty: false,
        }
    }

    pub fn open(dir: &Path) -> TestDb {
        let mut db = TestDb::memory();
        let _ = std::fs::create_dir_all(dir);
        if let Ok(txt) = std::fs::read_to_string(dir.join(FILE_STATS)) {
            if let Ok(f) = serde_json::from_str::<StatsFile>(&txt) {
                db.stats = f.stats.into_iter().map(|s| (s.key.clone(), s)).collect();
            }
        }
        db.dir = Some(dir.to_path_buf());
        db
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    fn queue<T: Serialize>(&mut self, file: &'static str, row: &T) {
        if self.dir.is_none() {
            return;
        }
        if let Ok(line) = serde_json::to_string(row) {
            self.pending.entry(file).or_default().push(line);
        }
    }

    fn bump(&mut self, run: &QuestRun, step: &str, ok: bool) {
        let key = format!("{}|{}|{}|{}|{}", run.pass, run.kind, run.target, run.preset, step);
        let e = self.stats.entry(key.clone()).or_insert_with(|| SuccessStat {
            key,
            pass: run.pass.clone(),
            kind: run.kind.clone(),
            target: run.target.clone(),
            preset: run.preset.clone(),
            step: step.to_string(),
            attempts: 0,
            successes: 0,
            rate: 0.0,
            last_ms: 0,
        });
        e.attempts += 1;
        e.successes += ok as u64;
        e.rate = e.successes as f64 / e.attempts as f64;
        e.last_ms = run.created_ms;
    }

    pub fn record_run(&mut self, run: QuestRun) {
        self.bump(&run, WHOLE_QUEST, run.ok);
        for s in &run.steps {
            let tries = s.attempts.max(1);
            for i in 0..tries {
                let last = i + 1 == tries;
                self.bump(&run, &s.step, last && s.ok);
            }
        }
        self.stats_dirty = true;
        self.queue(FILE_RUNS, &run);
        self.mirror.runs.push(run.clone());
        push_capped(&mut self.runs, run);
    }

    pub fn record_keyframe(&mut self, row: KeyframeRow) {
        self.queue(FILE_KEYFRAMES, &row);
        self.mirror.keyframes.push(row.clone());
        push_capped(&mut self.keyframes, row);
    }

    pub fn record_balance(&mut self, row: BalanceRow) {
        self.queue(FILE_BALANCE, &row);
        self.mirror.balance.push(row.clone());
        push_capped(&mut self.balance, row);
    }

    pub fn flush(&mut self) -> Vec<String> {
        let Some(dir) = self.dir.clone() else { return Vec::new() };
        let mut notes = Vec::new();
        for (file, lines) in std::mem::take(&mut self.pending) {
            if lines.is_empty() {
                continue;
            }
            let path = dir.join(file);
            let res = std::fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| {
                let mut buf = lines.join("\n");
                buf.push('\n');
                f.write_all(buf.as_bytes())
            });
            match res {
                Ok(()) => notes.push(format!("{file}+{}", lines.len())),
                Err(e) => notes.push(format!("{file} 저장 실패: {e}")),
            }
        }
        if self.stats_dirty {
            let f = StatsFile { updated_at: now_ms(), stats: self.stats.values().cloned().collect() };
            match serde_json::to_string_pretty(&f).map_err(|e| e.to_string()).and_then(|t| write_atomic(&dir.join(FILE_STATS), t.as_bytes()).map_err(|e| e.to_string())) {
                Ok(()) => {
                    self.stats_dirty = false;
                    notes.push(format!("{FILE_STATS}={}행", f.stats.len()));
                }
                Err(e) => notes.push(format!("{FILE_STATS} 저장 실패: {e}")),
            }
        }
        notes
    }

    pub fn take_mirror(&mut self) -> MirrorBatch {
        std::mem::take(&mut self.mirror)
    }

    pub fn stats(&self) -> Vec<SuccessStat> {
        self.stats.values().cloned().collect()
    }

    pub fn stat(&self, pass: &str, kind: &str, target: &str, preset: &str, step: &str) -> Option<&SuccessStat> {
        self.stats.get(&format!("{pass}|{kind}|{target}|{preset}|{step}"))
    }

    pub fn runs(&self) -> &[QuestRun] {
        &self.runs
    }

    pub fn keyframes(&self) -> &[KeyframeRow] {
        &self.keyframes
    }

    pub fn balance(&self) -> &[BalanceRow] {
        &self.balance
    }
}

impl Drop for TestDb {
    fn drop(&mut self) {
        if self.dir.is_some() && (!self.pending.is_empty() || self.stats_dirty) {
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pass: &str, ok: bool, locate_tries: u32) -> QuestRun {
        QuestRun {
            run_id: format!("r{}", now_ms()),
            session: "s".into(),
            pass: pass.into(),
            quest_id: "q1".into(),
            kind: "fetch".into(),
            target: "log".into(),
            destination: Some("warehouse".into()),
            preset: "mid".into(),
            seed: 7,
            ok,
            failure: (!ok).then(|| "locate_timeout".into()),
            ticks: 100,
            time_s: 10.0,
            energy_j: 900.0,
            steps: vec![
                StepRecord { step: "locate".into(), attempts: locate_tries, ok: true, fail_code: None, time_s: 2.0 },
                StepRecord { step: "deliver".into(), attempts: 1, ok, fail_code: None, time_s: 1.0 },
            ],
            created_ms: now_ms(),
        }
    }

    #[test]
    fn success_rate_counts_runs_and_step_attempts() {
        let mut db = TestDb::memory();
        db.record_run(run("game", true, 1));
        db.record_run(run("game", false, 3));
        let q = db.stat("game", "fetch", "log", "mid", WHOLE_QUEST).unwrap();
        assert_eq!((q.attempts, q.successes), (2, 1));
        assert!((q.rate - 0.5).abs() < 1e-12);
        let l = db.stat("game", "fetch", "log", "mid", "locate").unwrap();
        assert_eq!((l.attempts, l.successes), (4, 2));
        assert!(db.stat("ml", "fetch", "log", "mid", WHOLE_QUEST).is_none(), "passes are kept apart");
    }

    #[test]
    fn rows_survive_on_disk_and_stats_reload() {
        let dir = std::env::temp_dir().join(format!("noto_sds_records_{}_{}", std::process::id(), now_ms()));
        {
            let mut db = TestDb::open(&dir);
            db.record_run(run("ml", true, 1));
            db.record_balance(BalanceRow {
                run_id: "r".into(),
                pass: "ml".into(),
                tick: 3,
                source: "arm_raise".into(),
                source_seg: "arm".into(),
                target: "belly".into(),
                state: "flat|mine".into(),
                preset: "mid".into(),
                comp: 2.0,
                comp_unit: "deg".into(),
                reference: Some(2.5),
                tolerance: Some(1.0),
                torque_nm: 4.0,
                effort: 0.3,
                effort_noff: 0.4,
                margin: 0.6,
                margin_noff: 0.5,
                ff_gain: 1.0,
            });
            let notes = db.flush();
            assert!(notes.iter().any(|n| n.starts_with(FILE_RUNS)), "{notes:?}");
            assert_eq!(db.take_mirror().balance.len(), 1);
            assert!(db.take_mirror().is_empty(), "the mirror hands rows out once");
        }
        let text = std::fs::read_to_string(dir.join(FILE_BALANCE)).unwrap();
        let row: BalanceRow = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(row.target, "belly");
        let db = TestDb::open(&dir);
        assert_eq!(db.stat("ml", "fetch", "log", "mid", WHOLE_QUEST).map(|s| s.successes), Some(1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
