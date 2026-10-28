use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::lexicon::{EntityKind, Lexicon};
use crate::semantic::{cosine, PhraseEmbedder};

#[derive(Clone, Debug)]
pub struct EntityVectors {
    pub id: String,
    pub kind: EntityKind,
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

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct VisionPrompting {
    #[serde(default)]
    pub templates: Vec<String>,
    #[serde(default)]
    pub background: Vec<String>,
}

impl VisionPrompting {
    pub fn from_lexicon_json(raw: &str) -> Result<VisionPrompting> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            vision_prompting: Option<VisionPrompting>,
        }
        let r: Raw = serde_json::from_str(raw).context("parsing vision_prompting of the world lexicon")?;
        Ok(r.vision_prompting.unwrap_or_default())
    }

    pub fn apply(template: &str, phrase: &str) -> String {
        if template.contains("{}") {
            template.replace("{}", phrase)
        } else {
            format!("{template} {phrase}")
        }
    }

    pub fn is_background(&self, phrase: &str) -> bool {
        self.background.iter().any(|b| b.eq_ignore_ascii_case(phrase.trim()))
    }
}

fn mean_unit(vs: &[Arc<[f32]>]) -> Option<Arc<[f32]>> {
    let first = vs.first()?;
    let mut m = vec![0.0f32; first.len()];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v.iter()) {
            *a += b;
        }
    }
    let n = m.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    Some(m.into_iter().map(|x| x / n).collect::<Vec<f32>>().into())
}

impl VisionBank {
    pub fn build(lex: &Lexicon, emb: &dyn PhraseEmbedder) -> Result<VisionBank> {
        VisionBank::build_with(lex, emb, None)
    }

    pub fn build_with(lex: &Lexicon, emb: &dyn PhraseEmbedder, prompting: Option<&VisionPrompting>) -> Result<VisionBank> {
        let templates: &[String] = prompting.map_or(&[], |p| p.templates.as_slice());
        let skip = |t: &str| prompting.map_or(false, |p| p.is_background(t));
        let mut phrases: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for e in &lex.vision {
            for p in e.bias.iter().chain(e.prejudice.iter().filter(|p| !skip(&p.text))) {
                if seen.insert(p.text.clone()) {
                    phrases.push(p.text.clone());
                }
            }
        }
        let forms: Vec<String> = if templates.is_empty() { phrases.clone() } else { phrases.iter().flat_map(|p| templates.iter().map(move |t| VisionPrompting::apply(t, p))).collect() };
        let vecs = emb.embed_phrases(&forms)?;
        if vecs.len() != forms.len() {
            bail!("embedder returned {} vectors for {} prompts", vecs.len(), forms.len());
        }
        let per = templates.len().max(1);
        let vecs: Vec<Arc<[f32]>> = if templates.is_empty() { vecs } else { vecs.chunks(per).map(|c| mean_unit(c).expect("a chunk per phrase")).collect() };
        let texts = phrases;
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
                prejudice: e.prejudice.iter().filter(|p| !skip(&p.text)).filter_map(|p| by_text.get(p.text.as_str()).cloned()).collect(),
            })
            .collect();
        Ok(VisionBank { entities })
    }

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
    fn background_phrases_are_not_subtracted_and_templates_are_averaged() {
        let prompting = VisionPrompting { templates: vec!["a photo of {}".into(), "{} in a game".into()], background: vec!["Grey Boulder".into()] };
        let bank = VisionBank::build_with(&lex(), &BigramEmbedder, Some(&prompting)).unwrap();
        let tree = bank.entities.iter().find(|e| e.id == "tree").unwrap();
        assert!(tree.prejudice.is_empty(), "the tree's only prejudice is a background phrase");
        let a = bigram_vector("a photo of green leafy tree");
        let b = bigram_vector("green leafy tree in a game");
        let mut m: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let n = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        m.iter_mut().for_each(|x| *x /= n);
        assert!((cosine(&tree.bias[0].0, &m) - 1.0).abs() < 1e-5);
        assert_eq!(VisionPrompting::apply("a photo of {}.", "a tree"), "a photo of a tree.");
    }

    #[test]
    fn the_lexicon_carries_its_prompting() {
        let raw = include_str!("../../../data/world_lexicon.json");
        let p = VisionPrompting::from_lexicon_json(raw).unwrap();
        assert!(!p.templates.is_empty() && p.templates.iter().all(|t| t.contains("{}")));
        assert!(p.is_background("a grass field"));
        assert_eq!(VisionPrompting::from_lexicon_json(r#"{"version": 1}"#).unwrap(), VisionPrompting::default());
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
