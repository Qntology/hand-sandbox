use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Track {
    Vision,
    Search,
    Motion,
    Forecast,
    Depth,
    Locate,
    Tracker,
    Decide,
    Fusion,
    Balance,
    Effort,
    Quest,
    Act,
    Memory,
}

impl Track {
    pub const ALL: [Track; 14] = [
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
        Track::Act,
        Track::Memory,
    ];

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
            Track::Act => "act",
            Track::Memory => "memory",
        }
    }

    pub fn from_name(name: &str) -> Option<Track> {
        Track::ALL.into_iter().find(|t| t.as_str() == name)
    }

    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Search => (20, 8, 30),
            Track::Motion => (6, 4, 8),
            Track::Forecast => (4, 3, 6),
            Track::Depth | Track::Locate | Track::Tracker | Track::Fusion => (12, 6, 20),
            Track::Decide => (8, 4, 12),
            Track::Balance | Track::Effort => (8, 4, 12),
            Track::Quest => (4, 3, 6),
            Track::Act => (6, 4, 8),
            Track::Memory => (6, 4, 8),
        }
    }

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
            Track::Act => Ledger::Act,
            Track::Memory => Ledger::Perception,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ledger {
    Perception,
    Command,
    Task,
    Ml,
    Body,
    Act,
}

impl Ledger {
    pub const ALL: [Ledger; 6] = [Ledger::Perception, Ledger::Command, Ledger::Task, Ledger::Ml, Ledger::Body, Ledger::Act];

    pub fn as_str(&self) -> &'static str {
        match self {
            Ledger::Perception => "perception",
            Ledger::Command => "command",
            Ledger::Task => "task",
            Ledger::Ml => "ml",
            Ledger::Body => "body",
            Ledger::Act => "act",
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
