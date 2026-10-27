//! Composite N:N phases as they happen in the world's own quests: every
//! balance state the robot enters is matched to the development scenario
//! phases whose causes it has (`noto_core::nn::matching_phases`), scored with
//! that scenario's research weights, recorded in the SDS (`balance` track,
//! primary `nn:<scenario>`, secondary the phase) and kept here for `nn_status`.
//! No separate benchmark run: the existing quests and tests produce them.

use std::collections::BTreeMap;

use noto_core::nn::PhaseResult;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct PhaseStat {
    pub scenario: String,
    pub name_ko: String,
    pub phase: String,
    pub subsystem: String,
    pub weight: f32,
    /// The ladder the phase is a rung of (a scenario counts a ladder once, at its worst rung).
    pub ladder: Option<String>,
    pub count: u32,
    pub score_mean: f32,
    pub gated: u32,
    pub item_ok: u32,
    pub min_margin: f32,
    pub peak_ratio: f32,
    /// The latest result, cells included.
    pub last: Option<PhaseResult>,
}

#[derive(Clone, Debug, Default)]
pub struct NnLive {
    phases: BTreeMap<(String, String), PhaseStat>,
}

impl NnLive {
    pub fn record(&mut self, scenario: &str, name_ko: &str, r: &PhaseResult) {
        let e = self.phases.entry((scenario.to_string(), r.id.clone())).or_insert_with(|| PhaseStat {
            scenario: scenario.into(),
            name_ko: name_ko.into(),
            phase: r.id.clone(),
            subsystem: r.subsystem.into(),
            weight: r.weight,
            ladder: r.ladder.map(Into::into),
            min_margin: f32::INFINITY,
            ..PhaseStat::default()
        });
        e.count += 1;
        e.score_mean += (r.score - e.score_mean) / e.count as f32;
        e.gated += (!r.gates.is_empty()) as u32;
        e.item_ok += (r.item >= 1) as u32;
        e.min_margin = e.min_margin.min(r.min_margin);
        e.peak_ratio = e.peak_ratio.max(r.peak_ratio);
        e.last = Some(r.clone());
    }

    pub fn phases(&self) -> impl Iterator<Item = &PhaseStat> {
        self.phases.values()
    }

    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }

    /// Per scenario: the phases seen so far, combined with their weights
    /// (renormalised over the phases seen, so a partial scenario is marked).
    /// A ladder's rungs share one weight and the ladder counts once, at its
    /// worst rung seen, as `noto_core::nn::combine` scores a scenario.
    pub fn summary(&self) -> Vec<serde_json::Value> {
        let mut by: BTreeMap<&str, Vec<&PhaseStat>> = BTreeMap::new();
        for p in self.phases.values() {
            by.entry(p.scenario.as_str()).or_default().push(p);
        }
        by.into_iter()
            .map(|(id, ps)| {
                let mut parts: Vec<(f32, f32)> = Vec::new();
                let mut ladders: BTreeMap<&str, (f32, f32)> = BTreeMap::new();
                for p in &ps {
                    match p.ladder.as_deref() {
                        Some(l) => {
                            let e = ladders.entry(l).or_insert((p.weight, f32::INFINITY));
                            e.1 = e.1.min(p.score_mean);
                        }
                        None => parts.push((p.weight, p.score_mean)),
                    }
                }
                parts.extend(ladders.values().copied());
                let w: f32 = parts.iter().map(|x| x.0).sum();
                let score = parts.iter().map(|(w, s)| w * s).sum::<f32>() / w.max(1e-6);
                serde_json::json!({
                    "scenario": id, "name_ko": ps[0].name_ko, "phases_seen": ps.iter().map(|p| p.phase.clone()).collect::<Vec<_>>(),
                    "score": (score as f64 * 10.0).round() / 10.0, "events": ps.iter().map(|p| p.count).sum::<u32>(), "gated": ps.iter().map(|p| p.gated).sum::<u32>(),
                    "min_margin": ps.iter().map(|p| p.min_margin).fold(f32::INFINITY, f32::min), "peak_ratio": ps.iter().map(|p| p.peak_ratio).fold(0.0, f32::max),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase(id: &str, weight: f32, ladder: Option<&'static str>, score: f32) -> PhaseResult {
        PhaseResult {
            id: id.into(),
            subsystem: "reactive",
            weight,
            ladder,
            cells: vec![],
            seg_scores: [None; 7],
            score,
            item: 2,
            gates: vec![],
            beyond_human: vec![],
            static_margin: 0.5,
            min_margin: 0.4,
            peak_ratio: 0.3,
            worst_joint: None,
            sustained_effort: 0.2,
        }
    }

    #[test]
    fn a_ladder_counts_once_at_its_worst_rung() {
        let mut live = NnLive::default();
        live.record("S4", "짐 들고 제동", &phase("carry", 0.625, None, 90.0));
        live.record("S4", "짐 들고 제동", &phase("brake_0.5", 0.375, Some("brake"), 80.0));
        live.record("S4", "짐 들고 제동", &phase("brake_2.5", 0.375, Some("brake"), 40.0));
        let s = live.summary();
        // nn::combine: (0.625·90 + 0.375·min(80, 40)) / (0.625 + 0.375) = 71.25; summing every rung would give 73.6.
        let score = s[0]["score"].as_f64().unwrap();
        assert!((score - 71.3).abs() < 0.051, "{score}");
        // One rung seen: the same as without a ladder.
        let mut one = NnLive::default();
        one.record("S4", "짐 들고 제동", &phase("carry", 0.625, None, 90.0));
        one.record("S4", "짐 들고 제동", &phase("brake_0.5", 0.375, Some("brake"), 80.0));
        assert!((one.summary()[0]["score"].as_f64().unwrap() - 86.3).abs() < 0.051);
        assert_eq!(live.phases().filter(|p| p.ladder.as_deref() == Some("brake")).count(), 2);
    }
}
