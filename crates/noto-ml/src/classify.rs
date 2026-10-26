//! Zero-shot logistics classification with a word dictionary. Each dictionary
//! entry becomes a text prompt; SigLIP2 scores the image against every prompt.

use anyhow::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::model::Siglip2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DictEntry {
    pub id: String,
    pub label_en: String,
    pub label_ko: String,
    pub category: String,
    pub size_m: [f32; 3],
    pub typical_mass_kg: f32,
    pub graspable: bool,
    pub source: String,
}

/// The logistics word dictionary (`data/logistics_dictionary.json`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Dictionary {
    pub version: u32,
    /// Prompt template with one `{}` placeholder for the English label.
    pub template: String,
    pub entries: Vec<DictEntry>,
}

impl Dictionary {
    pub fn from_json(raw: &str) -> Result<Self> {
        Ok(serde_json::from_str(raw)?)
    }

    pub fn prompt(&self, entry: &DictEntry) -> String {
        self.template.replace("{}", &entry.label_en)
    }
}

/// Text embeddings of every dictionary prompt, in dictionary order.
pub struct Anchors {
    pub embeddings: Vec<Vec<f32>>,
}

const ANCHOR_BATCH: usize = 32;

impl Anchors {
    pub fn build(model: &Siglip2, dict: &Dictionary) -> Result<Self> {
        let prompts: Vec<String> = dict.entries.iter().map(|e| dict.prompt(e)).collect();
        let mut embeddings = Vec::with_capacity(prompts.len());
        for chunk in prompts.chunks(ANCHOR_BATCH) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            embeddings.extend(model.embed_texts(&refs)?);
        }
        Ok(Anchors { embeddings })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub id: String,
    pub label_en: String,
    pub label_ko: String,
    pub category: String,
    pub score: f32,
    pub size_m: [f32; 3],
    pub typical_mass_kg: f32,
    pub graspable: bool,
}

/// Top `top_k` dictionary entries for `image`, highest score first.
pub fn classify(
    model: &Siglip2,
    anchors: &Anchors,
    dict: &Dictionary,
    image: &DynamicImage,
    top_k: usize,
) -> Result<Vec<Prediction>> {
    let emb = model.embed_image(image)?;
    let scores = model.scores(&emb, &anchors.embeddings);
    Ok(rank(&scores, dict, top_k))
}

/// Sorts entries by score and keeps the top `top_k`.
pub fn rank(scores: &[f32], dict: &Dictionary, top_k: usize) -> Vec<Prediction> {
    let mut order: Vec<usize> = (0..dict.entries.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
        .into_iter()
        .take(top_k)
        .map(|i| {
            let e = &dict.entries[i];
            Prediction {
                id: e.id.clone(),
                label_en: e.label_en.clone(),
                label_ko: e.label_ko.clone(),
                category: e.category.clone(),
                score: scores[i],
                size_m: e.size_m,
                typical_mass_kg: e.typical_mass_kg,
                graspable: e.graspable,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW: &str = r#"{
        "version": 1,
        "template": "a photo of {}",
        "entries": [
            {"id": "a", "label_en": "box", "label_ko": "상자", "category": "package",
             "size_m": [0.3, 0.2, 0.1], "typical_mass_kg": 1.0, "graspable": true, "source": "t"},
            {"id": "b", "label_en": "pallet", "label_ko": "팔레트", "category": "infrastructure",
             "size_m": [1.2, 0.15, 0.8], "typical_mass_kg": 25.0, "graspable": false, "source": "t"}
        ]
    }"#;

    #[test]
    fn dictionary_loads_and_builds_prompts() {
        let d = Dictionary::from_json(RAW).unwrap();
        assert_eq!(d.entries.len(), 2);
        assert_eq!(d.prompt(&d.entries[1]), "a photo of pallet");
    }

    #[test]
    fn rank_orders_by_score_and_truncates() {
        let d = Dictionary::from_json(RAW).unwrap();
        let top = rank(&[0.2, 0.9], &d, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].id, "b");
        assert!((top[0].score - 0.9).abs() < 1e-6);
    }
}
