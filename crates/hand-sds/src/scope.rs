//! Tracks, ledgers and scopes of the score-dynamics store.
//!
//! A scope names where an observation belongs: `track|primary|secondary`.
//! Reads back off from the most specific scope to the global one when the
//! specific scope has too few observations (logis-center `Scope`,
//! `min_obs`). Tracks here are the ones this project records.

use serde::{Deserialize, Serialize};

/// Which recorder wrote an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Track {
    /// Per-label detection scores and margins from the recogniser.
    Vision,
    /// Natural-language command parsing (role scores, accepted/rejected spans).
    Search,
    /// Route and mining outcomes of the robot.
    Motion,
    /// Forecast errors of each time-series predictor.
    Forecast,
    /// ML keyframe track 1: monocular depth (TIPSv2 DPT or its stand-in) against the renderer's depth.
    Depth,
    /// ML keyframe track 2: SigLIP2 semantic location (heat-map peak) against the true object.
    Locate,
    /// ML keyframe track 3: mask tracking seeded by the semantic location (SAM 3.1 or its stand-in).
    Tracker,
    /// ML keyframe track 4: the time-series decision (forecast, move, look again) and its residuals.
    Decide,
    /// Features that mix the ML tracks (depth at the semantic peak, peak inside the mask, ...).
    Fusion,
    /// N:N posture balance: how much one segment's action asks of another, against research references.
    Balance,
    /// Joint effort (torque over the actuator's continuous rating) per segment and state.
    Effort,
    /// Quest and quest-step outcomes per pass (game logic or ML logic).
    Quest,
}

impl Track {
    pub const ALL: [Track; 12] = [
        Track::Vision,
        Track::Search,
        Track::Motion,
        Track::Forecast,
        Track::Depth,
        Track::Locate,
        Track::Tracker,
        Track::Decide,
        Track::Fusion,
        Track::Balance,
        Track::Effort,
        Track::Quest,
    ];

    /// The four ML keyframe tracks, in pipeline order.
    pub const ML: [Track; 4] = [Track::Depth, Track::Locate, Track::Tracker, Track::Decide];

    pub fn as_str(&self) -> &'static str {
        match self {
            Track::Vision => "vision",
            Track::Search => "search",
            Track::Motion => "motion",
            Track::Forecast => "forecast",
            Track::Depth => "depth",
            Track::Locate => "locate",
            Track::Tracker => "tracker",
            Track::Decide => "decide",
            Track::Fusion => "fusion",
            Track::Balance => "balance",
            Track::Effort => "effort",
            Track::Quest => "quest",
        }
    }

    pub fn from_name(name: &str) -> Option<Track> {
        Track::ALL.into_iter().find(|t| t.as_str() == name)
    }

    /// Observations needed before a scope is trusted: (secondary, primary, global).
    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Search => (20, 8, 30),
            Track::Motion => (6, 4, 8),
            Track::Forecast => (4, 3, 6),
            // One observation per keyframe: a test run gives tens, not hundreds.
            Track::Depth | Track::Locate | Track::Tracker | Track::Fusion => (12, 6, 20),
            Track::Decide => (8, 4, 12),
            // One observation per balance event or keyframe and segment pair.
            Track::Balance | Track::Effort => (8, 4, 12),
            // One observation per quest step: few, so trust a scope early.
            Track::Quest => (4, 3, 6),
        }
    }

    /// Length of the ring of recent values (the "recent" window of a scope).
    pub fn ring_len(&self) -> usize {
        (self.min_obs().1 as usize).max(4)
    }

    pub fn ledger(&self) -> Ledger {
        match self {
            Track::Vision => Ledger::Perception,
            Track::Search => Ledger::Command,
            Track::Motion | Track::Forecast | Track::Quest => Ledger::Task,
            Track::Depth | Track::Locate | Track::Tracker | Track::Decide | Track::Fusion => Ledger::Ml,
            Track::Balance | Track::Effort => Ledger::Body,
        }
    }
}

/// One ledger file on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ledger {
    Perception,
    Command,
    Task,
    /// The ML keyframe tracks and their mixed features.
    Ml,
    /// Posture balance and joint effort.
    Body,
}

impl Ledger {
    pub const ALL: [Ledger; 5] = [Ledger::Perception, Ledger::Command, Ledger::Task, Ledger::Ml, Ledger::Body];

    pub fn as_str(&self) -> &'static str {
        match self {
            Ledger::Perception => "perception",
            Ledger::Command => "command",
            Ledger::Task => "task",
            Ledger::Ml => "ml",
            Ledger::Body => "body",
        }
    }

    pub fn file_name(&self) -> String {
        format!("{}.json", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    pub track: Track,
    pub primary: String,
    pub secondary: String,
}

impl Scope {
    pub fn new(track: Track, primary: &str, secondary: &str) -> Self {
        Scope { track, primary: primary.trim().to_lowercase(), secondary: secondary.trim().to_lowercase() }
    }

    pub fn key_secondary(&self) -> String {
        format!("{}|{}|{}", self.track.as_str(), self.primary, self.secondary)
    }

    pub fn key_primary(&self) -> String {
        format!("{}|{}|", self.track.as_str(), self.primary)
    }

    pub fn key_global(&self) -> String {
        format!("{}||", self.track.as_str())
    }

    /// Keys an observation is written to: global, then primary, then secondary
    /// (each only when the level above it is named).
    pub fn write_keys(&self) -> Vec<String> {
        let mut keys = vec![self.key_global()];
        if !self.primary.is_empty() {
            keys.push(self.key_primary());
            if !self.secondary.is_empty() {
                keys.push(self.key_secondary());
            }
        }
        keys
    }
}

/// Ring length for a stored scope key (the track is its first segment).
pub fn ring_for_key(key: &str) -> usize {
    key.split('|').next().and_then(Track::from_name).map_or(6, |t| t.ring_len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_normalised_and_hierarchical() {
        let s = Scope::new(Track::Vision, "  Tree ", "NEAR");
        assert_eq!(s.key_secondary(), "vision|tree|near");
        assert_eq!(s.key_primary(), "vision|tree|");
        assert_eq!(s.key_global(), "vision||");
        assert_eq!(s.write_keys(), vec!["vision||", "vision|tree|", "vision|tree|near"]);
        assert_eq!(Scope::new(Track::Search, "", "x").write_keys(), vec!["search||"], "no primary, no deeper keys");
        assert_eq!(Scope::new(Track::Motion, "track", "").write_keys().len(), 2);
    }

    #[test]
    fn every_track_has_a_ledger_and_a_sane_ring() {
        for t in Track::ALL {
            assert_eq!(Track::from_name(t.as_str()), Some(t));
            assert!(t.ring_len() >= 4);
            let (s, p, g) = t.min_obs();
            assert!(s >= p && g >= p, "secondary and global need at least the primary count");
            assert!(Ledger::ALL.contains(&t.ledger()));
        }
        assert_eq!(ring_for_key("vision|tree|"), Track::Vision.ring_len());
        assert_eq!(ring_for_key("unknown|x|"), 6);
    }
}
