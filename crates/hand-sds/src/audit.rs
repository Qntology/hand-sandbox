//! Inspectable log with a hash chain.
//!
//! Every line carries the hash of the previous line, so editing, deleting or
//! reordering a line in the JSONL file is detected by [`verify_file`]. The
//! chain uses SHA-256 over `prev || canonical line JSON`. logis-center has no
//! such check; the line format `{seq, ts, cat, level, msg, data?}` is the one
//! the UI log tab reads.
//!
//! The chain detects accidental corruption and casual edits. Someone who can
//! rewrite the whole file can also recompute every hash, so it is not proof
//! against a malicious owner of the file.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogLine {
    pub seq: u64,
    pub ts: i64,
    pub cat: String,
    pub level: Level,
    pub msg: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileLine {
    #[serde(flatten)]
    line: LogLine,
    prev: String,
    hash: String,
}

fn chain_hash(prev: &str, line: &LogLine) -> String {
    let body = serde_json::to_string(line).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(prev.as_bytes());
    h.update(b"|");
    h.update(body.as_bytes());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOk {
    pub lines: usize,
    pub last_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifyError {
    /// 1-based line number in the file.
    pub line_no: usize,
    pub reason: String,
}

/// Checks the chain of a log file from the first line to the last.
pub fn verify_file(path: &Path) -> Result<VerifyOk, VerifyError> {
    let f = File::open(path).map_err(|e| VerifyError { line_no: 0, reason: format!("open: {e}") })?;
    let mut prev = GENESIS.to_string();
    let mut count = 0usize;
    let mut last_seq = 0u64;
    for (i, raw) in BufReader::new(f).lines().enumerate() {
        let line_no = i + 1;
        let raw = raw.map_err(|e| VerifyError { line_no, reason: format!("read: {e}") })?;
        if raw.trim().is_empty() {
            continue;
        }
        let fl: FileLine = serde_json::from_str(&raw).map_err(|e| VerifyError { line_no, reason: format!("parse: {e}") })?;
        if fl.prev != prev {
            return Err(VerifyError { line_no, reason: "previous-hash mismatch (a line was removed, inserted or reordered)".into() });
        }
        if fl.line.seq != last_seq + 1 {
            return Err(VerifyError { line_no, reason: format!("sequence jumps from {} to {}", last_seq, fl.line.seq) });
        }
        let expect = chain_hash(&prev, &fl.line);
        if fl.hash != expect {
            return Err(VerifyError { line_no, reason: "hash mismatch (the line content was edited)".into() });
        }
        prev = fl.hash;
        last_seq = fl.line.seq;
        count += 1;
    }
    Ok(VerifyOk { lines: count, last_hash: prev })
}

pub struct AuditLog {
    ring: VecDeque<LogLine>,
    cap: usize,
    next_seq: u64,
    prev_hash: String,
    file: Option<(PathBuf, BufWriter<File>)>,
    /// Set when a file write failed; the in-memory log keeps working.
    pub write_error: Option<String>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

impl AuditLog {
    /// In-memory log keeping the latest `cap` lines.
    pub fn memory(cap: usize) -> Self {
        AuditLog { ring: VecDeque::new(), cap: cap.max(16), next_seq: 1, prev_hash: GENESIS.into(), file: None, write_error: None }
    }

    /// Log that also appends to `{dir}/{session}.jsonl` (new file per session).
    pub fn open(dir: &Path, session: &str, cap: usize) -> std::io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let safe: String = session.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let path = dir.join(format!("{safe}.jsonl"));
        let f = OpenOptions::new().create_new(true).append(true).open(&path)?;
        let mut log = AuditLog::memory(cap);
        log.file = Some((path, BufWriter::new(f)));
        Ok(log)
    }

    pub fn path(&self) -> Option<&Path> {
        self.file.as_ref().map(|(p, _)| p.as_path())
    }

    /// Appends a line and returns its sequence number.
    pub fn log(&mut self, cat: &str, level: Level, msg: impl Into<String>, data: Option<serde_json::Value>) -> u64 {
        let line = LogLine { seq: self.next_seq, ts: now_ms(), cat: cat.to_string(), level, msg: msg.into(), data };
        let hash = chain_hash(&self.prev_hash, &line);
        if let Some((_, w)) = self.file.as_mut() {
            let fl = FileLine { line: line.clone(), prev: self.prev_hash.clone(), hash: hash.clone() };
            let res = serde_json::to_string(&fl).map_err(|e| e.to_string()).and_then(|s| {
                writeln!(w, "{s}").and_then(|_| w.flush()).map_err(|e| e.to_string())
            });
            if let Err(e) = res {
                self.write_error = Some(e);
            }
        }
        self.prev_hash = hash;
        self.next_seq += 1;
        let seq = line.seq;
        self.ring.push_back(line);
        while self.ring.len() > self.cap {
            self.ring.pop_front();
        }
        seq
    }

    pub fn info(&mut self, cat: &str, msg: impl Into<String>) -> u64 {
        self.log(cat, Level::Info, msg, None)
    }

    pub fn warn(&mut self, cat: &str, msg: impl Into<String>) -> u64 {
        self.log(cat, Level::Warn, msg, None)
    }

    pub fn last_seq(&self) -> u64 {
        self.next_seq - 1
    }

    pub fn last_hash(&self) -> &str {
        &self.prev_hash
    }

    /// Lines after `after_seq`, optionally of one category, at most `limit`.
    /// `next_seq` is the value to pass next time (the last line looked at).
    pub fn tail(&self, after_seq: u64, limit: usize, cat: Option<&str>) -> (Vec<LogLine>, u64) {
        let limit = limit.max(1);
        let mut out = Vec::new();
        let mut next = after_seq;
        for l in self.ring.iter().filter(|l| l.seq > after_seq) {
            if out.len() >= limit {
                break;
            }
            next = l.seq;
            if cat.map_or(true, |c| c == l.cat) {
                out.push(l.clone());
            }
        }
        (out, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hand_audit_{}_{}_{}", tag, std::process::id(), now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn sample(dir: &Path) -> (PathBuf, AuditLog) {
        let mut log = AuditLog::open(dir, "s1", 100).unwrap();
        log.info("world", "world created");
        log.log("vision", Level::Debug, "tile scores", Some(serde_json::json!({"tree": 0.71, "boulder": 0.2})));
        log.warn("quest", "step failed: no_path");
        log.info("sds", "flush");
        (log.path().unwrap().to_path_buf(), log)
    }

    #[test]
    fn floats_that_came_from_f32_survive_the_round_trip_of_verification() {
        // 0.98f32 is 0.9800000190734863 as an f64; reading that back must give the same f64,
        // or the re-serialised line hashes differently from the one that was written.
        let dir = tmpdir("floats");
        let mut log = AuditLog::open(&dir, "f", 100).unwrap();
        for i in 0..200 {
            let x = 0.98f32 - i as f32 * 0.00371;
            log.log("vision", Level::Info, "scores", Some(serde_json::json!({"score": x, "margin": x * 0.37, "nested": {"v": [x, x / 3.0]}})));
        }
        let path = log.path().unwrap().to_path_buf();
        drop(log);
        assert_eq!(verify_file(&path).unwrap().lines, 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fresh_chain_verifies_and_matches_the_in_memory_hash() {
        let dir = tmpdir("ok");
        let (path, log) = sample(&dir);
        let ok = verify_file(&path).unwrap();
        assert_eq!(ok.lines, 4);
        assert_eq!(ok.last_hash, log.last_hash());
        assert_eq!(log.last_seq(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_a_line_is_detected_at_that_line() {
        let dir = tmpdir("edit");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let edited = txt.replace("step failed: no_path", "step failed: none");
        assert_ne!(txt, edited);
        std::fs::write(&path, edited).unwrap();
        let err = verify_file(&path).unwrap_err();
        assert_eq!(err.line_no, 3);
        assert!(err.reason.contains("edited"), "{}", err.reason);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn removing_or_reordering_lines_is_detected() {
        let dir = tmpdir("remove");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = txt.lines().collect();

        let removed = [lines[0], lines[2], lines[3]].join("\n") + "\n";
        std::fs::write(&path, removed).unwrap();
        let err = verify_file(&path).unwrap_err();
        assert_eq!(err.line_no, 2);

        let swapped = [lines[0], lines[2], lines[1], lines[3]].join("\n") + "\n";
        std::fs::write(&path, swapped).unwrap();
        assert!(verify_file(&path).is_err());

        let truncated_tail = [lines[0], lines[1]].join("\n") + "\n";
        std::fs::write(&path, truncated_tail).unwrap();
        // A shortened tail still verifies as a shorter chain; the line count tells the difference.
        assert_eq!(verify_file(&path).unwrap().lines, 2);

        std::fs::write(&path, "{broken").unwrap();
        assert!(verify_file(&path).unwrap_err().reason.starts_with("parse"));
        assert!(verify_file(&dir.join("nope.jsonl")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tail_is_incremental_and_filters_by_category() {
        let mut log = AuditLog::memory(100);
        for i in 0..10 {
            log.info(if i % 2 == 0 { "world" } else { "quest" }, format!("m{i}"));
        }
        let (a, next) = log.tail(0, 4, None);
        assert_eq!(a.iter().map(|l| l.seq).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(next, 4);
        let (b, next) = log.tail(next, 100, Some("quest"));
        assert_eq!(b.iter().map(|l| l.msg.clone()).collect::<Vec<_>>(), vec!["m5", "m7", "m9"]);
        assert_eq!(next, 10, "advances past lines the filter skipped");
        let (c, next2) = log.tail(next, 10, None);
        assert!(c.is_empty());
        assert_eq!(next2, 10);
    }

    #[test]
    fn the_ring_is_bounded_but_the_file_keeps_everything() {
        let dir = tmpdir("ring");
        let mut log = AuditLog::open(&dir, "s2", 16).unwrap();
        for i in 0..50 {
            log.info("world", format!("line {i}"));
        }
        let (lines, _) = log.tail(0, 1000, None);
        assert_eq!(lines.len(), 16);
        assert_eq!(lines[0].seq, 35);
        assert_eq!(verify_file(log.path().unwrap()).unwrap().lines, 50);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_file_is_never_overwritten() {
        let dir = tmpdir("exists");
        let _a = AuditLog::open(&dir, "same", 16).unwrap();
        assert!(AuditLog::open(&dir, "same", 16).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_round_trips_through_the_file_format() {
        let dir = tmpdir("data");
        let (path, _log) = sample(&dir);
        let txt = std::fs::read_to_string(&path).unwrap();
        let second: FileLine = serde_json::from_str(txt.lines().nth(1).unwrap()).unwrap();
        assert_eq!(second.line.cat, "vision");
        assert_eq!(second.line.data.unwrap()["tree"], 0.71);
        assert_eq!(second.line.level, Level::Debug);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
