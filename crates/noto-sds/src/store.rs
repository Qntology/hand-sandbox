use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::scope::{ring_for_key, Ledger, Scope, Track};
use crate::stats::{decay_shape, ConfusionStat, GateStat, ScopeStat, Welford};

pub const SDS_RECIPE: &str = "noto-sds-v1:welford+ring/hier-write/ledger-split/calib-shrink/atomic";
const SDS_RECIPE_LEGACY: &str = "hand-sds-v1:welford+ring/hier-write/ledger-split/calib-shrink/atomic";

const AUTO_FLUSH_EVERY: u32 = 8;
const AUTO_FLUSH_MIN_GAP_MS: u128 = 5_000;
const DEFAULT_CAP_BYTES: usize = 2 * 1024 * 1024;
const CALIBRATION_SHRINK_K: f64 = 5.0;

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    match std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SdsFile {
    pub recipe: String,
    pub updated_at: i64,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
    #[serde(default)]
    pub engine_revs: Vec<String>,
}

pub const UNSTAMPED: &str = "unstamped";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtherRevs {
    #[default]
    Keep,
    SetAside,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenOptions {
    pub engine_rev: String,
    pub other_revs: OtherRevs,
}

impl OpenOptions {
    pub fn stamped(rev: &str) -> OpenOptions {
        OpenOptions { engine_rev: rev.into(), other_revs: OtherRevs::Keep }
    }

    pub fn only(rev: &str) -> OpenOptions {
        OpenOptions { engine_rev: rev.into(), other_revs: OtherRevs::SetAside }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopePolicy {
    pub pool_secondaries: bool,
    pub pool_primaries: bool,
}

impl Default for ScopePolicy {
    fn default() -> Self {
        ScopePolicy::POOLED
    }
}

impl ScopePolicy {
    pub const POOLED: ScopePolicy = ScopePolicy { pool_secondaries: true, pool_primaries: true };
    pub const KEEP_SECONDARIES: ScopePolicy = ScopePolicy { pool_secondaries: false, pool_primaries: true };
    pub const EXACT: ScopePolicy = ScopePolicy { pool_secondaries: false, pool_primaries: false };
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunFile {
    recipe: String,
    ledger: String,
    label: String,
    started_at: i64,
    updated_at: i64,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
    #[serde(default)]
    engine_rev: String,
}

#[derive(Debug, Clone, Default)]
struct RunBuf {
    label: String,
    file: String,
    started_ms: i64,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerStatus {
    pub ledger: String,
    pub scopes: usize,
    pub observations: u64,
    pub run_observations: u64,
    #[serde(default)]
    pub engine_revs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisStatus {
    pub scope: String,
    pub axis: String,
    pub kind: String,
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
    pub recent: f64,
    pub drift_z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRate {
    pub scope: String,
    pub gate: String,
    pub evaluated: u64,
    pub blocked_rate: f64,
    pub sole_rate: f64,
    pub mean_severity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionStatus {
    pub scope: String,
    pub pair: String,
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub mean_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceReport {
    pub scope: String,
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
    pub mean_dev: f64,
    pub recent_dev: f64,
    pub in_band_rate: f64,
    #[serde(default)]
    pub mean_ref: f64,
    #[serde(default)]
    pub mean_tol: f64,
    #[serde(default)]
    pub mean_abs_dev: Option<f64>,
    #[serde(default)]
    pub rms_dev: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaAxis {
    pub scope: String,
    pub axis: String,
    pub n: u64,
    pub mean_dev: f64,
    pub mean_abs_dev: Option<f64>,
    pub rms_dev: Option<f64>,
    pub in_band_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaReport {
    pub track: String,
    pub prefix: String,
    pub axes: Vec<AreaAxis>,
    pub n: u64,
    pub mean_abs_dev: Option<f64>,
    pub rms_dev: Option<f64>,
    pub in_band_rate: Option<f64>,
    pub legacy_axes: usize,
    pub worst: Option<AreaAxis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdsStatus {
    pub dir: String,
    pub recipe: String,
    pub ledgers: Vec<LedgerStatus>,
    pub axes: Vec<AxisStatus>,
    pub confusions: Vec<ConfusionStatus>,
    pub log: Vec<String>,
}

fn sanitize_label(raw: &str) -> String {
    let s: String = raw.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let s = s.trim_matches('_').to_string();
    if s.is_empty() {
        "run".into()
    } else {
        s.chars().take(40).collect()
    }
}

pub struct SdsStore {
    dir: Option<PathBuf>,
    bases: HashMap<Ledger, SdsFile>,
    runs: HashMap<Ledger, RunBuf>,
    dirty: HashSet<Ledger>,
    write_tick: u32,
    last_flush: Option<Instant>,
    log: Vec<String>,
    cap_bytes: usize,
    engine_rev: String,
    policies: HashMap<Track, ScopePolicy>,
}

impl SdsStore {
    pub fn memory() -> Self {
        let mut s = SdsStore {
            dir: None,
            bases: HashMap::new(),
            runs: HashMap::new(),
            dirty: HashSet::new(),
            write_tick: 0,
            last_flush: None,
            log: Vec::new(),
            cap_bytes: DEFAULT_CAP_BYTES,
            engine_rev: String::new(),
            policies: HashMap::new(),
        };
        for l in Ledger::ALL {
            s.bases.insert(l, fresh_file());
        }
        s.log.push("[SDS] 메모리 전용 저장소 (디스크에 쓰지 않음)".into());
        s
    }

    pub fn open(dir: &Path) -> Self {
        SdsStore::open_with(dir, OpenOptions::default())
    }

    pub fn open_with(dir: &Path, opts: OpenOptions) -> Self {
        let mut store = SdsStore::memory();
        store.dir = Some(dir.to_path_buf());
        store.engine_rev = opts.engine_rev.clone();
        store.log.clear();
        let _ = std::fs::create_dir_all(dir);
        let mut loaded = 0usize;
        for l in Ledger::ALL {
            let path = dir.join(l.file_name());
            let mut file = fresh_file();
            if path.exists() {
                match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|t| serde_json::from_str::<SdsFile>(&t).map_err(|e| e.to_string()))
                {
                    Ok(mut f) if f.recipe == SDS_RECIPE || f.recipe == SDS_RECIPE_LEGACY => {
                        if f.engine_revs.is_empty() && !f.scopes.is_empty() {
                            f.engine_revs.push(UNSTAMPED.into());
                        }
                        let own = store.stamp();
                        let foreign = f.engine_revs.iter().any(|r| *r != own);
                        if foreign && opts.other_revs == OtherRevs::SetAside {
                            let aside = dir.join(format!("{}.rev-{}.json", l.as_str(), now_ms()));
                            match std::fs::rename(&path, &aside) {
                                Ok(()) => store.log.push(format!(
                                    "[SDS] [{}] 다른 엔진 개정({})의 관측을 {}로 옮겨 두고 새로 시작합니다",
                                    l.as_str(),
                                    f.engine_revs.join(", "),
                                    aside.display()
                                )),
                                Err(e) => store.log.push(format!("[SDS] [{}] 다른 엔진 개정의 원장을 옮기지 못해({e}) 읽지 않고 새로 시작합니다", l.as_str())),
                            }
                        } else {
                            loaded += 1;
                            file = f;
                        }
                    }
                    Ok(f) => store.log.push(format!(
                        "[SDS] [{}] 레시피 세대 불일치 ('{}' vs '{}') — 원장을 폐기하고 새로 시작합니다",
                        l.as_str(),
                        f.recipe,
                        SDS_RECIPE
                    )),
                    Err(e) => store.log.push(format!("[SDS] [{}] 원장 읽기/파싱 실패({}) — 새로 시작합니다", l.as_str(), e)),
                }
            }
            store.bases.insert(l, file);
        }
        store.log.push(if loaded == 0 {
            "[SDS] 저장된 원장이 없어 냉간 시작합니다 (보정 미개입, 고정 기본값 사용)".into()
        } else {
            format!("[SDS] 원장 {}종을 불러왔습니다", loaded)
        });
        store
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn set_cap_bytes(&mut self, cap: usize) {
        self.cap_bytes = cap;
    }

    pub fn engine_rev(&self) -> &str {
        &self.engine_rev
    }

    pub fn set_engine_rev(&mut self, rev: &str) {
        self.engine_rev = rev.into();
    }

    pub fn engine_revs(&self, l: Ledger) -> Vec<String> {
        let mut v = self.bases.get(&l).map(|b| b.engine_revs.clone()).unwrap_or_default();
        if self.runs.get(&l).is_some_and(|r| r.observations > 0) {
            v.push(self.stamp());
        }
        v.sort();
        v.dedup();
        v
    }

    fn stamp(&self) -> String {
        if self.engine_rev.is_empty() {
            UNSTAMPED.into()
        } else {
            self.engine_rev.clone()
        }
    }

    pub fn set_policy(&mut self, track: Track, policy: ScopePolicy) {
        self.policies.insert(track, policy);
    }

    pub fn policy(&self, track: Track) -> ScopePolicy {
        self.policies.get(&track).copied().unwrap_or_default()
    }

    fn levels(&self, scope: &Scope) -> Vec<(String, u64)> {
        let (m2, m1, mg) = scope.track.min_obs();
        let mut v = vec![(scope.key_secondary(), m2), (scope.key_primary(), m1), (scope.key_global(), mg)];
        let p = self.policy(scope.track);
        if !p.pool_secondaries && !scope.secondary.is_empty() {
            v.truncate(1);
        } else if !p.pool_primaries && !scope.primary.is_empty() {
            v.truncate(2);
        }
        v
    }

    pub fn log_lines(&self) -> &[String] {
        &self.log
    }

    fn push_log(&mut self, line: String) {
        if self.log.len() >= 200 {
            self.log.drain(0..100);
        }
        self.log.push(line);
    }

    fn open_run(&mut self, l: Ledger, label: &str) -> &mut RunBuf {
        let dir = self.dir.clone();
        self.runs.entry(l).or_insert_with(|| {
            let stamp = now_ms();
            let lab = sanitize_label(label);
            let mut file = format!("{}_{}.json", stamp, lab);
            if let Some(dir) = &dir {
                let mut i = 2;
                while dir.join("runs").join(l.as_str()).join(&file).exists() {
                    file = format!("{}_{}_{}.json", stamp, lab, i);
                    i += 1;
                }
            }
            RunBuf { label: lab, file, started_ms: stamp, observations: 0, scopes: HashMap::new() }
        })
    }

    fn with_scopes<F: Fn(&mut ScopeStat, usize)>(&mut self, scope: &Scope, f: F) {
        let ledger = scope.track.ledger();
        let ring = scope.track.ring_len();
        let keys = scope.write_keys();
        let run = self.open_run(ledger, scope.track.as_str());
        run.observations = run.observations.saturating_add(1);
        for k in keys {
            let e = run.scopes.entry(k).or_default();
            f(e, ring);
            e.updated_at = now_ms();
        }
        self.dirty.insert(ledger);
        self.write_tick = self.write_tick.saturating_add(1);
        let gap_ok = self.last_flush.map(|t| t.elapsed().as_millis() >= AUTO_FLUSH_MIN_GAP_MS).unwrap_or(true);
        if self.dir.is_some() && self.write_tick >= AUTO_FLUSH_EVERY && gap_ok {
            self.flush();
        }
    }

    pub fn record_baseline(&mut self, scope: &Scope, axis: &str, value: f64) {
        if !value.is_finite() {
            return;
        }
        let a = axis.to_string();
        self.with_scopes(scope, move |s, ring| {
            s.baseline.entry(a.clone()).or_default().push(value, ring);
        });
    }

    pub fn record_calibration(&mut self, scope: &Scope, axis: &str, measured: f64, predicted: f64) {
        if !measured.is_finite() || !predicted.is_finite() || predicted.abs() < 1e-9 {
            return;
        }
        let ratio = (measured / predicted).clamp(0.05, 20.0);
        let a = axis.to_string();
        self.with_scopes(scope, move |s, ring| {
            s.calibration.entry(a.clone()).or_default().push(ratio, ring);
        });
    }

    pub fn record_decay(&mut self, scope: &Scope, axis: &str, scores: &[f32]) {
        let Some(shape) = decay_shape(scores) else { return };
        let a = axis.to_string();
        self.with_scopes(scope, move |s, ring| {
            let d = s.decay.entry(a.clone()).or_default();
            d.margin.push(shape.margin, ring);
            d.top_gap_ratio.push(shape.top_gap_ratio, ring);
            d.tail_flatness.push(shape.tail_flatness, ring);
            d.entropy_norm.push(shape.entropy_norm, ring);
        });
    }

    pub fn record_confusion(&mut self, scope: &Scope, winner: &str, loser: &str, margin: f64) {
        if winner.is_empty() || loser.is_empty() || winner == loser {
            return;
        }
        let (a, b, a_won) = if winner <= loser {
            (winner.to_string(), loser.to_string(), true)
        } else {
            (loser.to_string(), winner.to_string(), false)
        };
        let key = format!("{}|{}", a, b);
        self.with_scopes(scope, move |s, ring| {
            let e: &mut ConfusionStat = s.confusion.entry(key.clone()).or_default();
            e.ties += 1;
            if a_won {
                e.a_wins += 1;
            } else {
                e.b_wins += 1;
            }
            if margin.is_finite() {
                e.margin.push(margin, ring);
            }
        });
    }

    pub fn record_agreement(&mut self, scope: &Scope, axis: &str, agreed: bool) {
        let a = axis.to_string();
        self.with_scopes(scope, move |s, ring| {
            s.agreement.entry(a.clone()).or_default().push(if agreed { 1.0 } else { 0.0 }, ring);
        });
    }

    pub fn record_reference(&mut self, scope: &Scope, axis: &str, value: f64, reference: f64, tolerance: f64) {
        if !value.is_finite() || !reference.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
            return;
        }
        let dev = (value - reference) / tolerance;
        let (a, ad, ab, ar, at) = (axis.to_string(), format!("{axis}:dev"), format!("{axis}:in_band"), format!("{axis}:ref"), format!("{axis}:tol"));
        let (aa, a2) = (format!("{axis}:absdev"), format!("{axis}:dev2"));
        self.with_scopes(scope, move |s, ring| {
            s.baseline.entry(a.clone()).or_default().push(value, ring);
            s.baseline.entry(ad.clone()).or_default().push(dev, ring);
            s.baseline.entry(aa.clone()).or_default().push(dev.abs(), ring);
            s.baseline.entry(a2.clone()).or_default().push(dev * dev, ring);
            s.baseline.entry(ar.clone()).or_default().push(reference, ring);
            s.baseline.entry(at.clone()).or_default().push(tolerance, ring);
            s.agreement.entry(ab.clone()).or_default().push(if dev.abs() <= 1.0 { 1.0 } else { 0.0 }, ring);
        });
    }

    pub fn record_transition(&mut self, scope: &Scope, from: &str, to: &str) {
        if from.is_empty() || to.is_empty() {
            return;
        }
        let key = format!("{}>{}", from, to);
        self.with_scopes(scope, move |s, _| {
            *s.transition.entry(key.clone()).or_insert(0) += 1;
        });
    }

    pub fn record_gates(&mut self, scope: &Scope, items: &[(String, u8, bool, bool)]) {
        if items.is_empty() {
            return;
        }
        let list = items.to_vec();
        self.with_scopes(scope, move |s, ring| {
            for (id, sev, blocked, sole) in list.iter() {
                let g: &mut GateStat = s.gates.entry(id.clone()).or_default();
                g.evaluated += 1;
                if *blocked {
                    g.blocked += 1;
                }
                if *sole {
                    g.sole += 1;
                }
                g.severity.push(*sev as f64, ring);
            }
        });
    }

    fn merged_scope(&self, l: Ledger, key: &str) -> Option<ScopeStat> {
        let base = self.bases.get(&l).and_then(|f| f.scopes.get(key).cloned());
        let run = self.runs.get(&l).and_then(|r| r.scopes.get(key).cloned());
        match (base, run) {
            (Some(mut b), Some(r)) => {
                b.absorb(r, ring_for_key(key), now_ms());
                Some(b)
            }
            (Some(b), None) => Some(b),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    fn resolve<T, F: Fn(&ScopeStat) -> Option<(T, u64)>>(&self, scope: &Scope, pick: F) -> Option<(T, String)> {
        let l = scope.track.ledger();
        for (key, need) in self.levels(scope) {
            if need == 0 {
                continue;
            }
            if let Some(st) = self.merged_scope(l, &key) {
                if let Some((v, n)) = pick(&st) {
                    if n >= need {
                        return Some((v, key));
                    }
                }
            }
        }
        None
    }

    pub fn adaptive_baseline(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.mean, w.sd(), w.n), w.n))).map(|v| v.0)
    }

    pub fn adaptive_baseline_scoped(&self, scope: &Scope, axis: &str) -> Option<((f64, f64, u64), String)> {
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.mean, w.sd(), w.n), w.n)))
    }

    pub fn adaptive_recent(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, f64)> {
        self.resolve(scope, |st| st.baseline.get(axis).map(|w| ((w.recent_mean(), w.mean, w.drift_z()), w.n))).map(|v| v.0)
    }

    pub fn calibration_factor(&self, scope: &Scope, axis: &str) -> Option<(f64, u64, f64, String)> {
        self.resolve(scope, |st| {
            st.calibration.get(axis).map(|w| {
                let drift = w.drift_z();
                let center = if drift.abs() > 2.0 { w.recent_mean() } else { w.mean };
                let shrunk = 1.0 + (center - 1.0) * (w.n as f64 / (w.n as f64 + CALIBRATION_SHRINK_K));
                ((shrunk.clamp(0.25, 4.0), w.n, drift), w.n)
            })
        })
        .map(|((f, n, d), key)| (f, n, d, key))
    }

    pub fn transition_prior(&self, scope: &Scope, from: &str, to: &str) -> Option<f64> {
        let l = scope.track.ledger();
        for (key, need) in self.levels(scope).into_iter().take(2) {
            let Some(st) = self.merged_scope(l, &key) else { continue };
            let prefix = format!("{}>", from);
            let total: u64 = st.transition.iter().filter(|(k, _)| k.starts_with(&prefix)).map(|(_, v)| *v).sum();
            if total < need {
                continue;
            }
            let outs = st.transition.keys().filter(|k| k.starts_with(&prefix)).count().max(1) as f64;
            let hit = *st.transition.get(&format!("{}>{}", from, to)).unwrap_or(&0) as f64;
            return Some((hit + 1.0) / (total as f64 + outs + 1.0));
        }
        None
    }

    pub fn agreement_rate(&self, scope: &Scope, axis: &str) -> Option<(f64, u64)> {
        self.resolve(scope, |st| st.agreement.get(axis).map(|w| ((w.mean, w.n), w.n))).map(|v| v.0)
    }

    pub fn agreement_rate_scoped(&self, scope: &Scope, axis: &str) -> Option<((f64, u64), String)> {
        self.resolve(scope, |st| st.agreement.get(axis).map(|w| ((w.mean, w.n), w.n)))
    }

    pub fn adaptive_decay(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        self.resolve(scope, |st| st.decay.get(axis).map(|d| ((d.margin.mean, d.margin.sd(), d.margin.n), d.margin.n))).map(|v| v.0)
    }

    pub fn confusion_share(&self, scope: &Scope, a: &str, b: &str) -> Option<(f64, u64)> {
        let (x, y) = if a <= b { (a, b) } else { (b, a) };
        let key = format!("{}|{}", x, y);
        self.resolve(scope, |st| {
            let total: u64 = st.confusion.values().map(|c| c.ties).sum();
            st.confusion.get(&key).map(|c| ((c.ties as f64 / total.max(1) as f64, c.ties), total))
        })
        .map(|v| v.0)
    }

    pub fn reference_report(&self, scope: &Scope, axis: &str) -> Option<ReferenceReport> {
        self.resolve(scope, |st| reference_of(st, axis).map(|r| {
            let n = r.n;
            (r, n)
        }))
        .map(|(r, scope_key)| ReferenceReport { scope: scope_key, ..r })
    }

    pub fn exact_baseline(&self, scope: &Scope, axis: &str) -> Option<(f64, f64, u64)> {
        let key = if scope.secondary.is_empty() { scope.key_primary() } else { scope.key_secondary() };
        let st = self.merged_scope(scope.track.ledger(), &key)?;
        st.baseline.get(axis).filter(|w| w.n > 0).map(|w| (w.mean, w.sd(), w.n))
    }

    pub fn exact_reference(&self, scope: &Scope, axis: &str) -> Option<ReferenceReport> {
        let key = if scope.secondary.is_empty() { scope.key_primary() } else { scope.key_secondary() };
        let st = self.merged_scope(scope.track.ledger(), &key)?;
        st.baseline.get(axis).filter(|w| w.n > 0)?;
        reference_of(&st, axis).map(|r| ReferenceReport { scope: key, ..r })
    }

    pub fn area_report(&self, track: Track, prefix: &str) -> AreaReport {
        let l = track.ledger();
        let head = format!("{}|", track.as_str());
        let prefix = prefix.trim().to_lowercase();
        let mut keys: Vec<String> = self.bases.get(&l).map(|b| b.scopes.keys().cloned().collect()).unwrap_or_default();
        if let Some(r) = self.runs.get(&l) {
            keys.extend(r.scopes.keys().cloned());
        }
        keys.retain(|k| {
            let Some(rest) = k.strip_prefix(&head) else { return false };
            match rest.split_once('|') {
                Some((p, s)) => !p.is_empty() && s.is_empty() && p.starts_with(&prefix),
                None => false,
            }
        });
        keys.sort();
        keys.dedup();
        let mut axes = Vec::new();
        for k in &keys {
            let Some(st) = self.merged_scope(l, k) else { continue };
            let mut names: Vec<&String> = st.baseline.keys().filter(|a| a.ends_with(":dev")).collect();
            names.sort();
            for a in names {
                let axis = a.trim_end_matches(":dev");
                let d = &st.baseline[a];
                if d.n == 0 {
                    continue;
                }
                axes.push(AreaAxis {
                    scope: k.clone(),
                    axis: axis.to_string(),
                    n: d.n,
                    mean_dev: d.mean,
                    mean_abs_dev: st.baseline.get(&format!("{axis}:absdev")).filter(|w| w.n > 0).map(|w| w.mean),
                    rms_dev: st.baseline.get(&format!("{axis}:dev2")).filter(|w| w.n > 0).map(|w| w.mean.max(0.0).sqrt()),
                    in_band_rate: st.agreement.get(&format!("{axis}:in_band")).filter(|w| w.n > 0).map(|w| w.mean),
                });
            }
        }
        let weighted = |f: &dyn Fn(&AreaAxis) -> Option<f64>| {
            let (num, den) = axes.iter().filter_map(|a| f(a).map(|v| (v * a.n as f64, a.n as f64))).fold((0.0, 0.0), |s, x| (s.0 + x.0, s.1 + x.1));
            (den > 0.0).then(|| num / den)
        };
        let mean_abs_dev = weighted(&|a| a.mean_abs_dev);
        let rms_dev = weighted(&|a| a.rms_dev.map(|r| r * r)).map(f64::sqrt);
        let in_band_rate = weighted(&|a| a.in_band_rate);
        let far = |a: &AreaAxis| a.mean_abs_dev.unwrap_or(a.mean_dev.abs());
        let worst = axes.iter().max_by(|a, b| far(a).total_cmp(&far(b))).cloned();
        AreaReport {
            track: track.as_str().into(),
            prefix,
            n: axes.iter().map(|a| a.n).sum(),
            legacy_axes: axes.iter().filter(|a| a.mean_abs_dev.is_none()).count(),
            mean_abs_dev,
            rms_dev,
            in_band_rate,
            worst,
            axes,
        }
    }

    pub fn fusion_mse(&self, scope: &Scope, method: &str) -> Option<(f64, u64)> {
        self.adaptive_baseline(scope, &format!("fusion_mse:{}", method)).map(|(m, _, n)| (m, n))
    }

    pub fn gate_rates(&self, scope: &Scope) -> Vec<GateRate> {
        let l = scope.track.ledger();
        let mut out = Vec::new();
        for (key, _) in self.levels(scope) {
            if let Some(st) = self.merged_scope(l, &key) {
                if st.gates.is_empty() {
                    continue;
                }
                for (id, g) in st.gates.iter() {
                    let n = g.evaluated.max(1) as f64;
                    out.push(GateRate {
                        scope: key.clone(),
                        gate: id.clone(),
                        evaluated: g.evaluated,
                        blocked_rate: g.blocked as f64 / n,
                        sole_rate: g.sole as f64 / n,
                        mean_severity: g.severity.mean,
                    });
                }
                break;
            }
        }
        out.sort_by(|a, b| b.blocked_rate.partial_cmp(&a.blocked_rate).unwrap_or(std::cmp::Ordering::Equal).then(a.gate.cmp(&b.gate)));
        out
    }

    pub fn axes_of(&self, scope: &Scope, prefix: &str) -> Vec<AxisStatus> {
        let l = scope.track.ledger();
        for (key, _) in self.levels(scope) {
            if let Some(st) = self.merged_scope(l, &key) {
                let mut v: Vec<AxisStatus> = st
                    .baseline
                    .iter()
                    .filter(|(a, _)| a.starts_with(prefix))
                    .map(|(a, w)| axis_status(&key, a, "baseline", w))
                    .collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_by(|a, b| a.axis.cmp(&b.axis));
                return v;
            }
        }
        Vec::new()
    }

    pub fn flush(&mut self) -> Vec<String> {
        let Some(dir) = self.dir.clone() else { return Vec::new() };
        let mut lines = Vec::new();
        let open: Vec<Ledger> = self.runs.keys().cloned().collect();
        for l in Ledger::ALL {
            if !self.dirty.contains(&l) && !open.contains(&l) {
                continue;
            }
            if let Some(run) = self.runs.get(&l) {
                let rf = RunFile {
                    recipe: SDS_RECIPE.into(),
                    ledger: l.as_str().into(),
                    label: run.label.clone(),
                    started_at: run.started_ms,
                    updated_at: now_ms(),
                    observations: run.observations,
                    scopes: run.scopes.clone(),
                    engine_rev: self.engine_rev.clone(),
                };
                let rdir = dir.join("runs").join(l.as_str());
                let _ = std::fs::create_dir_all(&rdir);
                if let Ok(txt) = serde_json::to_string_pretty(&rf) {
                    match write_atomic(&rdir.join(&run.file), txt.as_bytes()) {
                        Ok(()) => lines.push(format!("{}={}건→{}", l.as_str(), run.observations, run.file)),
                        Err(e) => lines.push(format!("{} 실행 파일 저장 실패: {}", l.as_str(), e)),
                    }
                }
            }
            let mut snapshot = self.bases.get(&l).cloned().unwrap_or_default();
            snapshot.recipe = SDS_RECIPE.into();
            snapshot.engine_revs = self.engine_revs(l);
            if let Some(run) = self.runs.get(&l) {
                for (k, v) in run.scopes.iter() {
                    snapshot.scopes.entry(k.clone()).or_default().absorb(v.clone(), ring_for_key(k), now_ms());
                }
            }
            snapshot.updated_at = now_ms();
            let mut txt = serde_json::to_string_pretty(&snapshot).unwrap_or_default();
            if txt.len() > self.cap_bytes {
                let mut keys: Vec<(String, u64, i64)> =
                    snapshot.scopes.iter().map(|(k, v)| (k.clone(), v.observations(), v.updated_at)).collect();
                keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
                for (k, _, _) in keys {
                    if txt.len() <= self.cap_bytes {
                        break;
                    }
                    snapshot.scopes.remove(&k);
                    txt = serde_json::to_string_pretty(&snapshot).unwrap_or(txt);
                }
                lines.push(format!("{} 상한 초과로 저관측 스코프 절삭 (잔존 {})", l.as_str(), snapshot.scopes.len()));
            }
            match write_atomic(&dir.join(l.file_name()), txt.as_bytes()) {
                Ok(()) => {
                    self.dirty.remove(&l);
                    lines.push(format!("{}.json={}B", l.as_str(), txt.len()));
                }
                Err(e) => lines.push(format!("{} 원장 저장 실패: {}", l.as_str(), e)),
            }
        }
        self.write_tick = 0;
        self.last_flush = Some(Instant::now());
        if !lines.is_empty() {
            self.push_log(format!("[SDS] 저장: {}", lines.join(" | ")));
        }
        lines
    }

    pub fn close_runs(&mut self) {
        self.flush();
        let revs: Vec<(Ledger, Vec<String>)> = Ledger::ALL.iter().map(|l| (*l, self.engine_revs(*l))).collect();
        let runs: Vec<(Ledger, RunBuf)> = self.runs.drain().collect();
        for (l, revs) in revs {
            if let Some(b) = self.bases.get_mut(&l) {
                b.engine_revs = revs;
            }
        }
        for (l, run) in runs {
            let base = self.bases.entry(l).or_default();
            for (k, v) in run.scopes.into_iter() {
                let ring = ring_for_key(&k);
                base.scopes.entry(k).or_default().absorb(v, ring, now_ms());
            }
            base.updated_at = now_ms();
            self.dirty.insert(l);
        }
        self.flush();
    }

    pub fn purge(&mut self) {
        self.bases.clear();
        self.runs.clear();
        self.dirty.clear();
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::create_dir_all(dir);
        }
        for l in Ledger::ALL {
            self.bases.insert(l, fresh_file());
        }
        self.push_log("[SDS] 원장·실행 파일을 전량 삭제했습니다. 판정은 고정 기본값으로 복귀합니다".into());
    }

    pub fn status(&self, max_axes: usize) -> SdsStatus {
        let mut ledgers = Vec::new();
        let mut axes = Vec::new();
        let mut confusions = Vec::new();
        for l in Ledger::ALL {
            let run = self.runs.get(&l);
            let mut keys: Vec<String> = self.bases.get(&l).map(|b| b.scopes.keys().cloned().collect()).unwrap_or_default();
            if let Some(r) = run {
                for k in r.scopes.keys() {
                    if !keys.contains(k) {
                        keys.push(k.clone());
                    }
                }
            }
            keys.sort();
            let mut obs = 0u64;
            for k in keys.iter() {
                let Some(st) = self.merged_scope(l, k) else { continue };
                obs += st.observations();
                for (a, w) in st.baseline.iter() {
                    axes.push(axis_status(k, a, "baseline", w));
                }
                for (a, w) in st.calibration.iter() {
                    axes.push(axis_status(k, a, "calibration", w));
                }
                for (a, d) in st.decay.iter() {
                    axes.push(axis_status(k, a, "decay_margin", &d.margin));
                }
                for (a, w) in st.agreement.iter() {
                    axes.push(axis_status(k, a, "agreement", w));
                }
                for (g, gs) in st.gates.iter() {
                    let n = gs.evaluated.max(1) as f64;
                    axes.push(AxisStatus {
                        scope: k.clone(),
                        axis: g.clone(),
                        kind: "gate_block".into(),
                        n: gs.evaluated,
                        mean: gs.blocked as f64 / n,
                        sd: gs.severity.sd(),
                        recent: gs.sole as f64 / n,
                        drift_z: gs.severity.drift_z(),
                    });
                }
                for (pair, c) in st.confusion.iter() {
                    confusions.push(ConfusionStatus {
                        scope: k.clone(),
                        pair: pair.clone(),
                        ties: c.ties,
                        a_wins: c.a_wins,
                        b_wins: c.b_wins,
                        mean_margin: c.margin.mean,
                    });
                }
            }
            ledgers.push(LedgerStatus {
                ledger: l.as_str().into(),
                scopes: keys.len(),
                observations: obs,
                run_observations: run.map(|r| r.observations).unwrap_or(0),
                engine_revs: self.engine_revs(l),
            });
        }
        axes.sort_by(|a, b| b.n.cmp(&a.n).then(a.scope.cmp(&b.scope)).then(a.axis.cmp(&b.axis)));
        axes.truncate(max_axes);
        confusions.sort_by(|a, b| b.ties.cmp(&a.ties).then(a.pair.cmp(&b.pair)));
        confusions.truncate(max_axes);
        SdsStatus {
            dir: self.dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "(memory)".into()),
            recipe: SDS_RECIPE.into(),
            ledgers,
            axes,
            confusions,
            log: self.log.iter().rev().take(30).cloned().collect(),
        }
    }

    pub fn scope_count(&self, track: Track) -> usize {
        let prefix = format!("{}|", track.as_str());
        let l = track.ledger();
        let mut keys: HashSet<&String> = HashSet::new();
        if let Some(b) = self.bases.get(&l) {
            keys.extend(b.scopes.keys().filter(|k| k.starts_with(&prefix)));
        }
        if let Some(r) = self.runs.get(&l) {
            keys.extend(r.scopes.keys().filter(|k| k.starts_with(&prefix)));
        }
        keys.len()
    }
}

fn fresh_file() -> SdsFile {
    SdsFile { recipe: SDS_RECIPE.into(), updated_at: now_ms(), scopes: HashMap::new(), engine_revs: Vec::new() }
}

fn reference_of(st: &ScopeStat, axis: &str) -> Option<ReferenceReport> {
    let v = st.baseline.get(axis)?;
    let d = st.baseline.get(&format!("{axis}:dev"))?;
    let b = st.agreement.get(&format!("{axis}:in_band"))?;
    let mean_ref = st.baseline.get(&format!("{axis}:ref")).map_or(f64::NAN, |w| w.mean);
    let mean_tol = st.baseline.get(&format!("{axis}:tol")).map_or(f64::NAN, |w| w.mean);
    let mean_abs_dev = st.baseline.get(&format!("{axis}:absdev")).filter(|w| w.n > 0).map(|w| w.mean);
    let rms_dev = st.baseline.get(&format!("{axis}:dev2")).filter(|w| w.n > 0).map(|w| w.mean.max(0.0).sqrt());
    Some(ReferenceReport { scope: String::new(), n: v.n, mean: v.mean, sd: v.sd(), mean_dev: d.mean, recent_dev: d.recent_mean(), in_band_rate: b.mean, mean_ref, mean_tol, mean_abs_dev, rms_dev })
}

fn axis_status(scope: &str, axis: &str, kind: &str, w: &Welford) -> AxisStatus {
    AxisStatus {
        scope: scope.to_string(),
        axis: axis.to_string(),
        kind: kind.into(),
        n: w.n,
        mean: w.mean,
        sd: w.sd(),
        recent: w.recent_mean(),
        drift_z: w.drift_z(),
    }
}

impl Drop for SdsStore {
    fn drop(&mut self) {
        if self.dir.is_some() && (!self.dirty.is_empty() || !self.runs.is_empty()) {
            self.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("noto_sds_{}_{}_{}", tag, std::process::id(), now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn feed(store: &mut SdsStore, scope: &Scope, axis: &str, n: usize, base: f64) {
        for i in 0..n {
            store.record_baseline(scope, axis, base + (i % 3) as f64 * 0.01);
        }
    }

    #[test]
    fn cold_start_answers_none_until_enough_observations() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "tree", "near");
        assert!(s.adaptive_baseline(&sc, "score").is_none());
        feed(&mut s, &sc, "score", 7, 0.6);
        assert!(s.adaptive_baseline(&sc, "score").is_none(), "primary needs 8, secondary 20, global 30");
        feed(&mut s, &sc, "score", 1, 0.6);
        let (_, key) = s.adaptive_baseline_scoped(&sc, "score").unwrap();
        assert_eq!(key, "vision|tree|", "falls back to the primary scope");
        feed(&mut s, &sc, "score", 12, 0.6);
        let (_, key) = s.adaptive_baseline_scoped(&sc, "score").unwrap();
        assert_eq!(key, "vision|tree|near", "secondary now has 20");
    }

    #[test]
    fn a_thin_secondary_scope_backs_off_to_a_thicker_primary() {
        let mut s = SdsStore::memory();
        let near = Scope::new(Track::Vision, "tree", "near");
        let far = Scope::new(Track::Vision, "tree", "far");
        feed(&mut s, &near, "score", 25, 0.70);
        feed(&mut s, &far, "score", 3, 0.30);
        let (m_far, _, _) = s.adaptive_baseline(&far, "score").unwrap();
        assert!(m_far > 0.6, "pooled mean {m_far}");
        let (m_near, _, n) = s.adaptive_baseline(&near, "score").unwrap();
        assert_eq!(n, 25);
        assert!((m_near - 0.71).abs() < 0.03);
    }

    #[test]
    fn calibration_is_shrunk_toward_one_when_data_is_thin() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "track", "f5");
        for _ in 0..4 {
            s.record_calibration(&sc, "mine_time", 1.5, 1.0);
        }
        let (f4, n4, _, _) = s.calibration_factor(&sc, "mine_time").unwrap();
        assert_eq!(n4, 4);
        let expected4 = 1.0 + 0.5 * (4.0 / 9.0);
        assert!((f4 - expected4).abs() < 1e-9, "{f4} vs {expected4}");
        for _ in 0..96 {
            s.record_calibration(&sc, "mine_time", 1.5, 1.0);
        }
        let (f100, _, _, _) = s.calibration_factor(&sc, "mine_time").unwrap();
        assert!(f100 > f4 && f100 < 1.5 && f100 > 1.45, "{f100}");
        let before = s.calibration_factor(&sc, "mine_time").unwrap().1;
        s.record_calibration(&sc, "mine_time", 1.0, 0.0);
        s.record_calibration(&sc, "mine_time", f64::NAN, 1.0);
        assert_eq!(s.calibration_factor(&sc, "mine_time").unwrap().1, before);
    }

    #[test]
    fn calibration_recentres_when_the_recent_values_drifted() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "wheel", "f5");
        for i in 0..40 {
            s.record_calibration(&sc, "drive_time", 1.0 + 0.01 * (i % 2) as f64, 1.0);
        }
        let (steady, _, d0, _) = s.calibration_factor(&sc, "drive_time").unwrap();
        assert!(d0.abs() < 2.0);
        for _ in 0..8 {
            s.record_calibration(&sc, "drive_time", 2.0, 1.0);
        }
        let (shifted, _, d1, _) = s.calibration_factor(&sc, "drive_time").unwrap();
        assert!(d1.abs() > 2.0, "drift z {d1}");
        assert!(shifted > steady + 0.3, "{shifted} vs {steady}");
    }

    #[test]
    fn confusion_pairs_are_symmetric_and_counted() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "", "");
        for _ in 0..6 {
            s.record_confusion(&sc, "coal_ore", "iron_ore_node", 0.02);
        }
        for _ in 0..4 {
            s.record_confusion(&sc, "iron_ore_node", "coal_ore", 0.03);
        }
        s.record_confusion(&sc, "tree", "tree", 0.1);
        s.record_confusion(&sc, "", "tree", 0.1);
        for _ in 0..20 {
            s.record_confusion(&sc, "tree", "boulder", 0.05);
        }
        let (share, ties) = s.confusion_share(&sc, "iron_ore_node", "coal_ore").unwrap();
        assert_eq!(ties, 10);
        assert!((share - 10.0 / 30.0).abs() < 1e-9);
        let st = s.status(10);
        let pair = st.confusions.iter().find(|c| c.pair == "coal_ore|iron_ore_node").unwrap();
        assert_eq!((pair.a_wins, pair.b_wins), (6, 4));
    }

    #[test]
    fn agreement_and_transition_priors() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Motion, "track", "");
        for i in 0..10 {
            s.record_agreement(&sc, "locate_ok", i % 5 != 0);
        }
        let (rate, n) = s.agreement_rate(&sc, "locate_ok").unwrap();
        assert_eq!(n, 10);
        assert!((rate - 0.8).abs() < 1e-9);
        for _ in 0..3 {
            s.record_transition(&sc, "locate", "approach");
        }
        s.record_transition(&sc, "locate", "locate");
        let p = s.transition_prior(&sc, "locate", "approach").unwrap();
        assert!((p - 4.0 / 7.0).abs() < 1e-9, "{p}");
        assert!(s.transition_prior(&sc, "mine", "collect").is_none());
        s.record_transition(&sc, "", "x");
        s.record_transition(&sc, "a", "");
    }

    #[test]
    fn gates_report_block_and_sole_rates() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "tree", "");
        for i in 0..10 {
            s.record_gates(&sc, &[("floor".into(), 2, i < 3, i == 0), ("temporal".into(), 1, i < 1, false)]);
        }
        let rates = s.gate_rates(&sc);
        assert_eq!(rates[0].gate, "floor");
        assert!((rates[0].blocked_rate - 0.3).abs() < 1e-9);
        assert!((rates[0].sole_rate - 0.1).abs() < 1e-9);
        assert!((rates[1].blocked_rate - 0.1).abs() < 1e-9);
        s.record_gates(&sc, &[]);
    }

    #[test]
    fn decay_margin_baseline_is_learned() {
        let mut s = SdsStore::memory();
        let sc = Scope::new(Track::Vision, "", "");
        for _ in 0..30 {
            s.record_decay(&sc, "tile_scores", &[0.8, 0.3, 0.25, 0.2, 0.1]);
        }
        let (m, sd, n) = s.adaptive_decay(&sc, "tile_scores").unwrap();
        assert_eq!(n, 30);
        assert!((m - 0.5).abs() < 1e-6 && sd < 1e-6);
    }

    #[test]
    fn ledgers_survive_a_restart_and_leave_no_temp_files() {
        let dir = tmpdir("persist");
        {
            let mut s = SdsStore::open(&dir);
            let sc = Scope::new(Track::Forecast, "score:tree", "");
            for i in 0..5 {
                s.record_baseline(&sc, "fusion_mse:holt", 0.01 + i as f64 * 0.001);
            }
            s.close_runs();
        }
        let leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).collect();
        assert!(leftovers.is_empty());
        assert!(dir.join("task.json").exists());
        let run_files: Vec<_> = std::fs::read_dir(dir.join("runs").join("task")).unwrap().collect();
        assert_eq!(run_files.len(), 1);

        let s = SdsStore::open(&dir);
        let sc = Scope::new(Track::Forecast, "score:tree", "");
        let (mse, n) = s.fusion_mse(&sc, "holt").expect("5 observations are enough for the forecast track");
        assert_eq!(n, 5);
        assert!((mse - 0.012).abs() < 1e-9);
        assert!(s.log_lines().iter().any(|l| l.contains("원장 1종")), "{:?}", s.log_lines());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unflushed_run_data_is_flushed_on_drop() {
        let dir = tmpdir("drop");
        {
            let mut s = SdsStore::open(&dir);
            s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", 1.0);
        }
        assert!(dir.join("task.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_ledger_with_another_recipe_is_discarded_with_a_reason() {
        let dir = tmpdir("recipe");
        std::fs::create_dir_all(&dir).unwrap();
        let mut old = fresh_file();
        old.recipe = "noto-sds-v0:old".into();
        let mut st = ScopeStat::default();
        st.baseline.entry("score".into()).or_default().push(0.5, 4);
        old.scopes.insert("vision||".into(), st);
        std::fs::write(dir.join("perception.json"), serde_json::to_string(&old).unwrap()).unwrap();
        std::fs::write(dir.join("command.json"), "{ not json").unwrap();
        let s = SdsStore::open(&dir);
        let log = s.log_lines().join("\n");
        assert!(log.contains("레시피 세대 불일치"), "{log}");
        assert!(log.contains("읽기/파싱 실패"), "{log}");
        assert!(s.status(5).ledgers.iter().all(|l| l.observations == 0));
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversized_ledgers_drop_the_least_observed_scopes_first() {
        let dir = tmpdir("cap");
        let mut s = SdsStore::open(&dir);
        s.set_cap_bytes(6_000);
        let busy = Scope::new(Track::Motion, "busy", "");
        for i in 0..60 {
            s.record_baseline(&busy, "a", i as f64);
        }
        for i in 0..40 {
            let sc = Scope::new(Track::Motion, &format!("rare{i}"), "");
            s.record_baseline(&sc, "a", 1.0);
        }
        s.close_runs();
        let txt = std::fs::read_to_string(dir.join("task.json")).unwrap();
        assert!(txt.len() <= 6_000 + 400, "ledger is {} bytes", txt.len());
        let file: SdsFile = serde_json::from_str(&txt).unwrap();
        assert!(file.scopes.contains_key("motion|busy|"), "the well observed scope is kept");
        assert!(file.scopes.len() < 41);
        assert!(s.log_lines().iter().any(|l| l.contains("절삭")));
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn purge_returns_to_cold_start() {
        let dir = tmpdir("purge");
        let mut s = SdsStore::open(&dir);
        let sc = Scope::new(Track::Forecast, "x", "");
        for _ in 0..5 {
            s.record_baseline(&sc, "a", 1.0);
        }
        s.close_runs();
        assert!(s.adaptive_baseline(&sc, "a").is_some());
        s.purge();
        assert!(s.adaptive_baseline(&sc, "a").is_none());
        assert!(!dir.join("task.json").exists());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomic_replaces_the_file_without_leaving_a_temp() {
        let dir = tmpdir("atomic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.json");
        write_atomic(&path, b"{\"a\":1}").unwrap();
        write_atomic(&path, b"{\"a\":2}").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":2}");
        assert!(!dir.join("a.json.tmp").exists());
        assert!(write_atomic(&dir.join("missing").join("b.json"), b"x").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_lists_axes_by_observation_count() {
        let mut s = SdsStore::memory();
        feed(&mut s, &Scope::new(Track::Vision, "tree", ""), "score", 12, 0.6);
        feed(&mut s, &Scope::new(Track::Motion, "track", ""), "mine_time", 3, 3.0);
        let st = s.status(3);
        assert_eq!(st.axes.len(), 3);
        assert!(st.axes[0].n >= st.axes[1].n && st.axes[1].n >= st.axes[2].n);
        assert!(st.ledgers.iter().any(|l| l.ledger == "perception" && l.observations > 0));
        assert_eq!(s.scope_count(Track::Vision), 2, "global and primary");
        let axes = s.axes_of(&Scope::new(Track::Vision, "tree", ""), "sc");
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0].kind, "baseline");
    }

    #[test]
    fn reference_axes_report_deviation_and_band_share() {
        let mut s = SdsStore::memory();
        let scope = Scope::new(Track::Balance, "arm_raise>belly", "flat");
        for _ in 0..4 {
            for v in [2.0, 2.25, 3.0] {
                s.record_reference(&scope, "comp_deg", v, 2.0, 0.5);
            }
        }
        let r = s.reference_report(&scope, "comp_deg").expect("12 observations reach the secondary minimum of 8");
        assert_eq!(r.scope, "balance|arm_raise>belly|flat");
        assert_eq!(r.n, 12);
        assert!((r.mean_dev - (0.0 + 0.5 + 2.0) / 3.0).abs() < 1e-9, "{r:?}");
        assert!((r.in_band_rate - 2.0 / 3.0).abs() < 1e-9);
        assert!((r.mean_ref - 2.0).abs() < 1e-12 && (r.mean_tol - 0.5).abs() < 1e-12);
        s.record_reference(&scope, "comp_deg", 9.0, 2.0, 0.0);
        assert_eq!(s.reference_report(&scope, "comp_deg").unwrap().n, 12);
        assert!(s.status(10).ledgers.iter().any(|l| l.ledger == "body" && l.observations > 0));
        assert_eq!(Track::Depth.ledger(), Ledger::Ml);
    }

    #[test]
    fn reference_reports_say_how_far_and_not_only_which_side() {
        let mut s = SdsStore::memory();
        let scope = Scope::new(Track::Act, "grasp:ts", "log");
        for _ in 0..4 {
            s.record_reference(&scope, "preload_s", 2.0, 1.0, 0.5);
            s.record_reference(&scope, "preload_s", 0.0, 1.0, 0.5);
        }
        let r = s.reference_report(&scope, "preload_s").unwrap();
        assert!(r.mean_dev.abs() < 1e-12, "{r:?}");
        assert!((r.mean_abs_dev.unwrap() - 2.0).abs() < 1e-12 && (r.rms_dev.unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(r.in_band_rate, 0.0);
        let e = s.exact_reference(&scope, "preload_s").unwrap();
        assert_eq!((e.mean_abs_dev, e.rms_dev, e.scope.as_str()), (r.mean_abs_dev, r.rms_dev, "act|grasp:ts|log"));
    }

    #[test]
    fn the_area_report_pools_every_reference_axis_of_a_prefix_once() {
        let mut s = SdsStore::memory();
        for item in ["log", "stone"] {
            for v in [1.0, 3.0] {
                s.record_reference(&Scope::new(Track::Act, "grasp:ts", item), "preload_s", v, 1.0, 1.0);
            }
        }
        for _ in 0..3 {
            s.record_reference(&Scope::new(Track::Act, "grasp>part", "reach"), "trunk_deg", 5.0, 1.0, 1.0);
        }
        s.record_reference(&Scope::new(Track::Act, "nn:grasp", "foot"), "margin", 9.0, 0.0, 1.0);
        for _ in 0..2 {
            s.record_baseline(&Scope::new(Track::Act, "grasp:old", ""), "x:dev", -0.5);
        }
        let r = s.area_report(Track::Act, "grasp");
        assert_eq!(r.axes.len(), 3, "{:?}", r.axes);
        assert_eq!(r.n, 4 + 3 + 2);
        assert_eq!(r.legacy_axes, 1);
        assert!((r.mean_abs_dev.unwrap() - 16.0 / 7.0).abs() < 1e-9, "{r:?}");
        assert!((r.rms_dev.unwrap() - (56.0f64 / 7.0).sqrt()).abs() < 1e-9);
        assert_eq!(r.worst.as_ref().map(|a| a.axis.as_str()), Some("trunk_deg"));
        assert_eq!(s.area_report(Track::Act, "").axes.len(), 4, "the whole track");
        assert!(s.area_report(Track::Vision, "").axes.is_empty());
    }

    #[test]
    fn a_scope_policy_keeps_model_modes_apart() {
        let mut s = SdsStore::memory();
        let real = Scope::new(Track::Locate, "parcel", "siglip2");
        let stand_in = Scope::new(Track::Locate, "parcel", "stand_in");
        feed(&mut s, &stand_in, "peak", 30, 0.9);
        feed(&mut s, &real, "peak", 3, 0.1);
        let (m, _, _) = s.adaptive_baseline(&real, "peak").unwrap();
        assert!(m > 0.7, "{m}");
        s.set_policy(Track::Locate, ScopePolicy::KEEP_SECONDARIES);
        assert!(s.adaptive_baseline(&real, "peak").is_none(), "the real model waits for its own observations");
        assert_eq!(s.adaptive_baseline_scoped(&stand_in, "peak").unwrap().1, "locate|parcel|stand_in");
        assert!(s.adaptive_baseline(&Scope::new(Track::Locate, "parcel", ""), "peak").is_some());
        let tips = Scope::new(Track::Depth, "tips", "");
        feed(&mut s, &Scope::new(Track::Depth, "stand_in", ""), "scale", 25, 1.25);
        feed(&mut s, &tips, "scale", 2, 0.9);
        assert!(s.adaptive_baseline(&tips, "scale").is_some(), "pooled through the global scope");
        s.set_policy(Track::Depth, ScopePolicy::EXACT);
        assert!(s.adaptive_baseline(&tips, "scale").is_none());
        assert_eq!(s.policy(Track::Vision), ScopePolicy::POOLED);
    }

    #[test]
    fn runs_and_ledgers_carry_the_engine_revision() {
        let dir = tmpdir("rev");
        {
            let mut s = SdsStore::open_with(&dir, OpenOptions::stamped("r1"));
            for i in 0..5 {
                s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", i as f64);
            }
            s.close_runs();
        }
        let ledger: SdsFile = serde_json::from_str(&std::fs::read_to_string(dir.join("task.json")).unwrap()).unwrap();
        assert_eq!(ledger.engine_revs, vec!["r1".to_string()]);
        let run = std::fs::read_dir(dir.join("runs").join("task")).unwrap().next().unwrap().unwrap().path();
        assert!(std::fs::read_to_string(run).unwrap().contains("\"engine_rev\": \"r1\""));
        {
            let mut s = SdsStore::open_with(&dir, OpenOptions::stamped("r2"));
            assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "x", ""), "a").is_some());
            s.record_baseline(&Scope::new(Track::Forecast, "x", ""), "a", 9.0);
            assert_eq!(s.engine_revs(Ledger::Task), vec!["r1".to_string(), "r2".to_string()]);
            assert!(s.status(1).ledgers.iter().any(|l| l.ledger == "task" && l.engine_revs.len() == 2));
            s.close_runs();
        }
        let s = SdsStore::open_with(&dir, OpenOptions::only("r2"));
        assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "x", ""), "a").is_none());
        assert!(s.log_lines().iter().any(|l| l.contains("다른 엔진 개정")), "{:?}", s.log_lines());
        let aside: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with("task.rev-")).collect();
        assert_eq!(aside.len(), 1);
        drop(s);
        let legacy = tmpdir("rev_legacy");
        std::fs::create_dir_all(&legacy).unwrap();
        let mut old = fresh_file();
        let mut st = ScopeStat::default();
        st.baseline.entry("a".into()).or_default().push(1.0, 4);
        old.scopes.insert("forecast||".into(), st);
        let txt = serde_json::to_string(&old).unwrap().replace(",\"engine_revs\":[]", "");
        std::fs::write(legacy.join("task.json"), txt).unwrap();
        let s = SdsStore::open_with(&legacy, OpenOptions::stamped("r2"));
        assert_eq!(s.engine_revs(Ledger::Task), vec![UNSTAMPED.to_string()]);
        drop(s);
        let s = SdsStore::open_with(&legacy, OpenOptions::only(""));
        assert!(s.adaptive_baseline(&Scope::new(Track::Forecast, "", ""), "a").is_none(), "one observation is below the minimum");
        assert_eq!(s.engine_revs(Ledger::Task), vec![UNSTAMPED.to_string()]);
        assert!(!s.log_lines().iter().any(|l| l.contains("다른 엔진 개정")), "{:?}", s.log_lines());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&legacy);
    }
}
