//! Scoring of image vectors against the vision entities of the lexicon.
//!
//! The entities (tree, boulder, warehouse, ...) carry English `bias` and
//! `prejudice` prompts in the same node format as the command words. An image
//! tile is scored against every entity at once; the caller decides how a raw
//! cosine becomes a probability (SigLIP uses `sigmoid(scale * cos + bias)`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::lexicon::{EntityKind, Lexicon};
use crate::semantic::{cosine, PhraseEmbedder};

#[derive(Clone, Debug)]
pub struct EntityVectors {
    pub id: String,
    pub kind: EntityKind,
    /// Resource id for sources and drops, place id for waypoints.
    pub link: String,
    bias: Vec<(Arc<[f32]>, f32)>,
    prejudice: Vec<Arc<[f32]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EntityScore {
    pub id: String,
    pub kind: EntityKind,
    pub link: String,
    pub bias: f32,
    pub prejudice: f32,
    pub score: f32,
}

pub struct VisionBank {
    pub entities: Vec<EntityVectors>,
}

impl VisionBank {
    /// Embeds every vision prompt once (the embedder deduplicates and caches).
    pub fn build(lex: &Lexicon, emb: &dyn PhraseEmbedder) -> Result<VisionBank> {
        let mut texts: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for e in &lex.vision {
            for p in e.bias.iter().chain(&e.prejudice) {
                if seen.insert(p.text.clone()) {
                    texts.push(p.text.clone());
                }
            }
        }
        let vecs = emb.embed_phrases(&texts)?;
        if vecs.len() != texts.len() {
            bail!("embedder returned {} vectors for {} prompts", vecs.len(), texts.len());
        }
        let by_text: HashMap<&str, Arc<[f32]>> = texts.iter().map(String::as_str).zip(vecs).collect();
        let entities = lex
            .vision
            .iter()
            .map(|e| EntityVectors {
                id: e.id.clone(),
                kind: e.kind.clone(),
                link: e.link.clone(),
                bias: e
                    .bias
                    .iter()
                    .filter_map(|p| by_text.get(p.text.as_str()).map(|v| (v.clone(), p.weight)))
                    .collect(),
                prejudice: e.prejudice.iter().filter_map(|p| by_text.get(p.text.as_str()).cloned()).collect(),
            })
            .collect();
        Ok(VisionBank { entities })
    }

    /// Scores one image vector against every entity, best first.
    ///
    /// `calibrate` maps a cosine to the scale the thresholds use (identity for
    /// raw cosines, a sigmoid for SigLIP pair probabilities). `bias` is the
    /// weighted maximum over the entity's bias prompts, `prejudice` the maximum
    /// over its prejudice prompts, and `score = bias − prejudice_weight · prejudice`.
    pub fn score(&self, q: &[f32], calibrate: &dyn Fn(f32) -> f32, prejudice_weight: f32) -> Vec<EntityScore> {
        let mut out: Vec<EntityScore> = self
            .entities
            .iter()
            .map(|e| {
                let bias = e.bias.iter().map(|(p, w)| calibrate(cosine(q, p)) * w).fold(0.0f32, f32::max);
                let prejudice = e.prejudice.iter().map(|p| calibrate(cosine(q, p))).fold(0.0f32, f32::max);
                EntityScore {
                    id: e.id.clone(),
                    kind: e.kind.clone(),
                    link: e.link.clone(),
                    bias,
                    prejudice,
                    score: bias - prejudice_weight * prejudice,
                }
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }

    /// Entity ids that can stand for `resource`: its source node and its drop.
    pub fn ids_for_resource(&self, resource: &str) -> Vec<&str> {
        self.entities
            .iter()
            .filter(|e| e.kind != EntityKind::Waypoint && e.link == resource)
            .map(|e| e.id.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::testing::{bigram_vector, BigramEmbedder};

    fn lex() -> Lexicon {
        Lexicon::from_json(
            r#"{
          "version": 1,
          "resource": {"log": {"exact_match": ["a"]}, "stone": {"exact_match": ["b"]}},
          "place": {"warehouse": {"exact_match": ["c"]}},
          "verb": {"harvest": {"exact_match": ["d1"]}, "pickup": {"exact_match": ["d2"]},
                   "deliver": {"exact_match": ["d3"]}, "go": {"exact_match": ["d4"]}, "stop": {"exact_match": ["d5"]}},
          "filler": {"exact_match": ["e"]},
          "vision": {
            "tree": {"kind": "source", "link": "log", "bias": "green leafy tree, brown trunk", "prejudice": "grey boulder"},
            "log": {"kind": "drop", "link": "log", "bias": "wooden log lying down", "prejudice": "grey boulder"},
            "boulder": {"kind": "source", "link": "stone", "bias": "grey boulder, big rock", "prejudice": "green leafy tree"},
            "warehouse": {"kind": "waypoint", "link": "warehouse", "bias": "warehouse building", "prejudice": ""}
          }
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn a_vector_close_to_a_prompt_ranks_that_entity_first() {
        let bank = VisionBank::build(&lex(), &BigramEmbedder).unwrap();
        let q = bigram_vector("green leafy tree");
        let s = bank.score(&q, &|c| c, 0.5);
        assert_eq!(s[0].id, "tree");
        assert!(s[0].score > s[1].score);
        let q = bigram_vector("grey boulder");
        assert_eq!(bank.score(&q, &|c| c, 0.5)[0].id, "boulder");
    }

    #[test]
    fn prejudice_prompts_lower_the_score_of_a_confusable_entity() {
        let bank = VisionBank::build(&lex(), &BigramEmbedder).unwrap();
        let q = bigram_vector("grey boulder");
        let s = bank.score(&q, &|c| c, 0.5);
        let tree = s.iter().find(|e| e.id == "tree").unwrap();
        assert!(tree.prejudice > 0.9, "the query is the tree's prejudice prompt");
        assert!(tree.score < tree.bias);
    }

    #[test]
    fn the_calibration_function_is_applied_to_every_cosine() {
        let bank = VisionBank::build(&lex(), &BigramEmbedder).unwrap();
        let q = bigram_vector("green leafy tree");
        let raw = bank.score(&q, &|c| c, 0.0);
        let half = bank.score(&q, &|c| c * 0.5, 0.0);
        let r = raw.iter().find(|e| e.id == "tree").unwrap().bias;
        let h = half.iter().find(|e| e.id == "tree").unwrap().bias;
        assert!((r * 0.5 - h).abs() < 1e-6);
    }

    #[test]
    fn resources_map_to_their_source_and_drop_entities() {
        let bank = VisionBank::build(&lex(), &BigramEmbedder).unwrap();
        let mut ids = bank.ids_for_resource("log");
        ids.sort();
        assert_eq!(ids, vec!["log", "tree"]);
        assert!(bank.ids_for_resource("coal").is_empty());
    }
}
