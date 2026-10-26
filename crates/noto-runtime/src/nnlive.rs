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
    pub fn summary(&self) -> Vec<serde_json::Value> {
        let mut by: BTreeMap<&str, Vec<&PhaseStat>> = BTreeMap::new();
        for p in self.phases.values() {
            by.entry(p.scenario.as_str()).or_default().push(p);
        }
        by.into_iter()
            .map(|(id, ps)| {
                let w: f32 = ps.iter().map(|p| p.weight).sum();
                let score = ps.iter().map(|p| p.weight * p.score_mean).sum::<f32>() / w.max(1e-6);
                serde_json::json!({
                    "scenario": id, "name_ko": ps[0].name_ko, "phases_seen": ps.iter().map(|p| p.phase.clone()).collect::<Vec<_>>(),
                    "score": (score as f64 * 10.0).round() / 10.0, "events": ps.iter().map(|p| p.count).sum::<u32>(), "gated": ps.iter().map(|p| p.gated).sum::<u32>(),
                    "min_margin": ps.iter().map(|p| p.min_margin).fold(f32::INFINITY, f32::min), "peak_ratio": ps.iter().map(|p| p.peak_ratio).fold(0.0, f32::max),
                })
            })
            .collect()
    }
}
