//! ML track 2: semantic location. "Where in this keyframe is the iron ore?"
//!
//! [`SiglipLocator`] uses SigLIP2 dense patch embeddings
//! (`noto_ml::Siglip2::embed_dense`): the cosine of every patch with a label's
//! text vector is a heat map from one forward pass. The peak is the semantic
//! position; it becomes the point prompt of the mask tracker. Negative
//! prompts (what a label is often confused with) are subtracted, as the
//! dictionary's `prejudice` phrases are elsewhere in the project.
//!
//! How a patch's cosines become heat is a [`LocateScoring`]:
//!
//! * [`LocateScoring::PairProb`] (before v1.0): the SigLIP pair probability
//!   `σ(t·cos + b)` of each patch, minus half the look-alike's cosine. On this
//!   renderer patch cosines are small (about 0.05 against the checkpoint's
//!   0.15 for even odds), so every value is near zero and the argmax is set by
//!   noise; and every label always gets a point, present or not.
//! * [`LocateScoring::Contrast`]: a closed-set softmax of `t·cos` over all
//!   labels plus explicit background classes (ground, grass, sky, ...), per
//!   patch. A patch of grass goes to the grass class instead of to whichever
//!   label happens to be least unlike it (the softmax over class prompts of
//!   the OpenAI CLIP repository's zero-shot classifier, with background
//!   classes added so the closed set does not force a choice).
//! * [`LocateScoring::ZScore`]: each label's cosines standardised over the
//!   image's patches (where in this frame it stands out most).
//!
//! Prompts can be put through templates and averaged in embedding space
//! ([`LocateCfg::templates`], the prompt ensembling of the OpenAI CLIP
//! repository's `Prompt_Engineering_for_ImageNet` notebook). The scores are
//! not calibrated here: `calib` fits per-label thresholds on rendered frames,
//! and the SDS `locate` track learns their per-label baseline.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::frame::Rgb;

/// What to look for: a label id and the phrases that describe it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelPrompt {
    pub id: String,
    pub prompts: Vec<String>,
    /// Phrases for look-alikes, subtracted from the score.
    #[serde(default)]
    pub negatives: Vec<String>,
}

/// A heat map on a coarse grid over the image (row-major).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Heat {
    pub cols: u32,
    pub rows: u32,
    pub values: Vec<f32>,
}

impl Heat {
    pub fn at(&self, c: u32, r: u32) -> f32 {
        self.values[(r * self.cols + c) as usize]
    }

    /// Cell holding the largest value.
    pub fn argmax(&self) -> Option<(u32, u32, f32)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i as u32 % self.cols, i as u32 / self.cols, *v))
    }
}

/// The semantic position of one label in one keyframe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticPoint {
    pub label: String,
    /// Peak in pixels of the keyframe.
    pub px: f32,
    pub py: f32,
    /// Pair probability at the peak (0..1).
    pub score: f32,
    /// Peak score minus the best other label's score at the same place.
    pub margin: f32,
    /// Whole-image presence of the label (SigLIP2 pair probability of the
    /// pooled embedding); 1 when the locator has no such notion.
    pub presence: f32,
    /// Cells above half of the peak, as a pixel box `[x0, y0, x1, y1]`.
    pub bbox: [f32; 4],
    pub heat: Heat,
}

pub trait SemanticLocator: Send {
    /// `siglip2-dense` or the name of a stand-in.
    fn mode(&self) -> &str;
    fn device(&self) -> String {
        "cpu".into()
    }
    /// One point per label that has a peak, best first.
    fn locate(&mut self, rgb: &Rgb, labels: &[LabelPrompt]) -> Result<Vec<SemanticPoint>>;
    /// [`SemanticLocator::locate`] for several keyframes, in input order. A
    /// locator that can batch its image model overrides this.
    fn locate_batch(&mut self, frames: &[&Rgb], labels: &[LabelPrompt]) -> Result<Vec<Vec<SemanticPoint>>> {
        frames.iter().map(|f| self.locate(f, labels)).collect()
    }
}

/// Turns per-label heat maps on one grid into semantic points. Shared by the
/// SigLIP2 locator and the stand-ins, so both report the same way.
pub fn points_from_heat(width: u32, height: u32, heats: Vec<(String, Heat)>) -> Vec<SemanticPoint> {
    points_with_presence(width, height, heats.into_iter().map(|(l, h)| (l, h, 1.0)).collect())
}

/// [`points_from_heat`] with a whole-image presence per label.
pub fn points_with_presence(width: u32, height: u32, heats: Vec<(String, Heat, f32)>) -> Vec<SemanticPoint> {
    let mut out = Vec::new();
    for (i, (label, heat, presence)) in heats.iter().enumerate() {
        let Some((c, r, peak)) = heat.argmax() else { continue };
        if peak <= 0.0 {
            continue;
        }
        let other = heats
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (_, h, _))| h.at(c, r))
            .fold(0.0f32, f32::max);
        // Sub-cell peak: weighted centroid of the 3x3 neighbourhood.
        let (mut sx, mut sy, mut sw) = (0.0f32, 0.0f32, 0.0f32);
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                let (cc, rr) = (c as i32 + dc, r as i32 + dr);
                if cc < 0 || rr < 0 || cc >= heat.cols as i32 || rr >= heat.rows as i32 {
                    continue;
                }
                let w = (heat.at(cc as u32, rr as u32) - peak * 0.5).max(0.0);
                sx += w * (cc as f32 + 0.5);
                sy += w * (rr as f32 + 0.5);
                sw += w;
            }
        }
        let (fc, fr) = if sw > 0.0 { (sx / sw, sy / sw) } else { (c as f32 + 0.5, r as f32 + 0.5) };
        let (cw, ch) = (width as f32 / heat.cols as f32, height as f32 / heat.rows as f32);
        let mut b = [f32::MAX, f32::MAX, 0.0f32, 0.0f32];
        // Connected cells above half the peak, starting at the peak.
        let mut seen = vec![false; heat.values.len()];
        let mut stack = vec![(c, r)];
        while let Some((cc, rr)) = stack.pop() {
            let k = (rr * heat.cols + cc) as usize;
            if seen[k] || heat.at(cc, rr) < peak * 0.5 {
                continue;
            }
            seen[k] = true;
            b = [b[0].min(cc as f32 * cw), b[1].min(rr as f32 * ch), b[2].max((cc + 1) as f32 * cw), b[3].max((rr + 1) as f32 * ch)];
            if cc > 0 {
                stack.push((cc - 1, rr));
            }
            if rr > 0 {
                stack.push((cc, rr - 1));
            }
            if cc + 1 < heat.cols {
                stack.push((cc + 1, rr));
            }
            if rr + 1 < heat.rows {
                stack.push((cc, rr + 1));
            }
        }
        out.push(SemanticPoint { label: label.clone(), px: fc * cw, py: fr * ch, score: peak, margin: peak - other, presence: *presence, bbox: b, heat: heat.clone() });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// How patch embeddings become heat (see the module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocateScoring {
    /// SigLIP pair probability minus the weighted look-alike cosine (before v1.0).
    PairProb,
    /// Softmax over labels and background classes of `t·cos`, per patch.
    Contrast,
    /// Each label's cosines z-scored over the image's patches.
    ZScore,
}

/// Settings of the SigLIP2 locator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocateCfg {
    pub scoring: LocateScoring,
    /// Weight of the negative phrases ([`LocateScoring::PairProb`] only).
    pub negative_weight: f32,
    /// Background classes of [`LocateScoring::Contrast`]: phrases for what
    /// most of an outdoor frame is (ground, grass, sky).
    #[serde(default)]
    pub background: Vec<String>,
    /// Prompt templates with `{}` for the phrase; every phrase's vector is the
    /// normalised mean over them. Empty: the phrases as written.
    #[serde(default)]
    pub templates: Vec<String>,
    /// Logit scale of the softmax; `None` uses the checkpoint's `exp(t)`.
    #[serde(default)]
    pub logit_scale: Option<f32>,
}

impl Default for LocateCfg {
    /// The locator as it was before v1.0.
    fn default() -> Self {
        LocateCfg { scoring: LocateScoring::PairProb, negative_weight: 0.5, background: Vec::new(), templates: Vec::new(), logit_scale: None }
    }
}

impl LocateCfg {
    /// The settings a [`LocateCfg::mode`] name stands for, with the given
    /// background phrases and templates (`None` for a name that is not a
    /// SigLIP2 dense mode). A calibration table names its mode; this is how
    /// the runtime scores the way the table was fitted.
    pub fn from_mode(mode: &str, background: &[String], templates: &[String]) -> Option<LocateCfg> {
        let rest = mode.strip_prefix("siglip2-dense")?;
        let mut cfg = LocateCfg::default();
        for part in rest.split('+').filter(|p| !p.is_empty()) {
            match part {
                "bg" => {
                    cfg.scoring = LocateScoring::Contrast;
                    cfg.background = background.to_vec();
                }
                "z" => cfg.scoring = LocateScoring::ZScore,
                "tpl" => cfg.templates = templates.to_vec(),
                _ => return None,
            }
        }
        Some(cfg)
    }

    /// The mode name the SDS scopes and the records carry, so statistics of
    /// one scoring never gate another.
    pub fn mode(&self) -> String {
        let mut m = String::from("siglip2-dense");
        match self.scoring {
            LocateScoring::PairProb => {}
            LocateScoring::Contrast => m.push_str("+bg"),
            LocateScoring::ZScore => m.push_str("+z"),
        }
        if !self.templates.is_empty() {
            m.push_str("+tpl");
        }
        m
    }
}

/// Text vectors of the labels for one [`LocateCfg`]: per label the mean unit
/// prompt vector and the negatives' vector, and the background classes.
#[derive(Clone, Debug, Default)]
pub struct LabelTexts {
    pub labels: Vec<(String, Vec<f32>, Option<Vec<f32>>)>,
    pub background: Vec<Vec<f32>>,
}

/// SigLIP2 dense-patch locator.
pub struct SiglipLocator {
    model: Arc<noto_ml::Siglip2>,
    /// Text vectors per (templated) phrase, embedded once.
    cache: HashMap<String, Vec<f32>>,
    /// Weight of the negative phrases (kept in step with the settings).
    pub negative_weight: f32,
    cfg: LocateCfg,
    mode: String,
}

impl SiglipLocator {
    pub fn new(model: Arc<noto_ml::Siglip2>) -> SiglipLocator {
        SiglipLocator::with_cfg(model, LocateCfg::default())
    }

    pub fn with_cfg(model: Arc<noto_ml::Siglip2>, cfg: LocateCfg) -> SiglipLocator {
        let mode = cfg.mode();
        SiglipLocator { model, cache: HashMap::new(), negative_weight: cfg.negative_weight, cfg, mode }
    }

    pub fn cfg(&self) -> &LocateCfg {
        &self.cfg
    }

    pub fn model(&self) -> &Arc<noto_ml::Siglip2> {
        &self.model
    }

    fn text_vectors(&mut self, phrases: &[String]) -> Result<Vec<Vec<f32>>> {
        let missing: Vec<&str> = phrases.iter().filter(|p| !self.cache.contains_key(*p)).map(String::as_str).collect();
        if !missing.is_empty() {
            let v = self.model.embed_texts(&missing)?;
            for (p, e) in missing.iter().zip(v) {
                self.cache.insert(p.to_string(), e);
            }
        }
        Ok(phrases.iter().map(|p| self.cache[p].clone()).collect())
    }

    /// One vector per phrase: the phrase itself, or the normalised mean of
    /// its templated forms.
    fn phrase_vectors(&mut self, phrases: &[String], templates: &[String]) -> Result<Vec<Vec<f32>>> {
        if templates.is_empty() {
            return self.text_vectors(phrases);
        }
        let mut out = Vec::with_capacity(phrases.len());
        for p in phrases {
            let forms: Vec<String> = templates.iter().map(|t| apply_template(t, p)).collect();
            let v = self.text_vectors(&forms)?;
            out.push(mean_unit(&v).unwrap_or_default());
        }
        Ok(out)
    }

    /// The text side of [`label_heats`] for these labels under `cfg`.
    pub fn label_texts(&mut self, labels: &[LabelPrompt], cfg: &LocateCfg) -> Result<LabelTexts> {
        let mut out = LabelTexts::default();
        for l in labels {
            let pos = self.phrase_vectors(&l.prompts, &cfg.templates)?;
            let Some(p) = mean_unit(&pos) else { continue };
            let neg = if l.negatives.is_empty() { None } else { mean_unit(&self.phrase_vectors(&l.negatives, &cfg.templates)?) };
            out.labels.push((l.id.clone(), p, neg));
        }
        out.background = self.phrase_vectors(&cfg.background, &cfg.templates)?;
        Ok(out)
    }
}

/// A prompt template applied to a phrase (`{}` stands for it; a template
/// without one is followed by the phrase).
pub fn apply_template(template: &str, phrase: &str) -> String {
    if template.contains("{}") {
        template.replace("{}", phrase)
    } else {
        format!("{template} {phrase}")
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Heat maps and presences of every label on one dense embedding, the way
/// `cfg.scoring` says. `logit_scale` and `logit_bias` are the checkpoint's
/// (`exp(t)` and `b`); `presence` is always SigLIP's own pooled pair
/// probability `σ(exp(t)·cos(pooled, label) + b)`.
pub fn label_heats(dense: &noto_ml::DenseEmbedding, texts: &LabelTexts, cfg: &LocateCfg, logit_scale: f32, logit_bias: f32) -> Vec<(String, Heat, f32)> {
    let (cols, rows) = (dense.grid_w as u32, dense.grid_h as u32);
    let pair = |c: f32| 1.0 / (1.0 + (-(logit_scale * c + logit_bias)).exp());
    let presence: Vec<f32> = texts.labels.iter().map(|(_, p, _)| pair(dot(&dense.pooled, p))).collect();
    let n = dense.vectors.len();
    let heats: Vec<Vec<f32>> = match cfg.scoring {
        LocateScoring::PairProb => texts
            .labels
            .iter()
            .map(|(_, p, neg)| {
                dense
                    .vectors
                    .iter()
                    .map(|v| {
                        let cn = neg.as_ref().map(|n| dot(v, n)).unwrap_or(0.0);
                        pair(dot(v, p) - cfg.negative_weight * cn.max(0.0))
                    })
                    .collect()
            })
            .collect(),
        LocateScoring::Contrast => {
            let t = cfg.logit_scale.unwrap_or(logit_scale);
            let mut out = vec![vec![0.0f32; n]; texts.labels.len()];
            for (i, v) in dense.vectors.iter().enumerate() {
                let logits: Vec<f32> = texts.labels.iter().map(|(_, p, _)| t * dot(v, p)).chain(texts.background.iter().map(|b| t * dot(v, b))).collect();
                let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = logits.iter().map(|l| (l - m).exp()).sum();
                for (k, row) in out.iter_mut().enumerate() {
                    row[i] = (logits[k] - m).exp() / z;
                }
            }
            out
        }
        LocateScoring::ZScore => texts
            .labels
            .iter()
            .map(|(_, p, _)| {
                let c: Vec<f32> = dense.vectors.iter().map(|v| dot(v, p)).collect();
                let mean = c.iter().sum::<f32>() / n.max(1) as f32;
                let sd = (c.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n.max(1) as f32).sqrt().max(1e-6);
                c.into_iter().map(|x| (x - mean) / sd).collect()
            })
            .collect(),
    };
    texts.labels.iter().zip(heats).zip(presence).map(|(((id, _, _), values), pres)| (id.clone(), Heat { cols, rows, values }, pres)).collect()
}

fn mean_unit(vs: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vs.first()?;
    let mut m = vec![0.0f32; first.len()];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v) {
            *a += b;
        }
    }
    let n = m.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    Some(m.into_iter().map(|x| x / n).collect())
}

impl SemanticLocator for SiglipLocator {
    fn mode(&self) -> &str {
        &self.mode
    }

    fn device(&self) -> String {
        noto_ml::device_label(self.model.device())
    }

    fn locate(&mut self, rgb: &Rgb, labels: &[LabelPrompt]) -> Result<Vec<SemanticPoint>> {
        let mut out = self.locate_batch(&[rgb], labels)?;
        Ok(out.pop().unwrap_or_default())
    }

    /// One batched dense pass for all keyframes (`Siglip2::embed_dense_batch`),
    /// the label text vectors embedded once for the whole batch, then the
    /// heat maps per keyframe.
    fn locate_batch(&mut self, frames: &[&Rgb], labels: &[LabelPrompt]) -> Result<Vec<Vec<SemanticPoint>>> {
        let mut cfg = self.cfg.clone();
        cfg.negative_weight = self.negative_weight;
        let texts = self.label_texts(labels, &cfg)?;
        let images: Vec<image::DynamicImage> = frames.iter().map(|f| f.to_dynamic()).collect();
        let dense = self.model.embed_dense_batch(&images)?;
        let (scale, bias) = self.model.logit_scale_bias();
        Ok(frames.iter().zip(&dense).map(|(rgb, d)| points_with_presence(rgb.width, rgb.height, label_heats(d, &texts, &cfg, scale, bias))).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_peak_cell_becomes_the_point_and_the_box() {
        let mut values = vec![0.1f32; 8 * 4];
        values[1 * 8 + 5] = 0.9;
        values[1 * 8 + 6] = 0.6;
        let a = Heat { cols: 8, rows: 4, values };
        let b = Heat { cols: 8, rows: 4, values: vec![0.2; 32] };
        let pts = points_from_heat(320, 160, vec![("iron".into(), a), ("stone".into(), b)]);
        assert_eq!(pts[0].label, "iron");
        let p = &pts[0];
        // Cell (5,1) is pixels 200..240 x 40..80; the neighbour at (6,1) pulls the peak right.
        assert!(p.px > 220.0 && p.px < 240.0, "{}", p.px);
        assert!((p.py - 60.0).abs() < 1e-3);
        assert!((p.margin - 0.7).abs() < 1e-5);
        assert_eq!(p.bbox, [200.0, 40.0, 280.0, 80.0]);
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// A 2 x 2 grid: patch 0 looks like the parcel, the others like grass.
    fn dense() -> (noto_ml::DenseEmbedding, LabelTexts) {
        let parcel = unit(&[1.0, 0.0, 0.0]);
        let grass = unit(&[0.0, 1.0, 0.0]);
        let log = unit(&[0.0, 0.0, 1.0]);
        let near_grass = unit(&[0.10, 1.0, 0.12]);
        let d = noto_ml::DenseEmbedding {
            grid_h: 2,
            grid_w: 2,
            vectors: vec![unit(&[1.0, 0.3, 0.1]), near_grass.clone(), near_grass.clone(), near_grass],
            attention: vec![0.25; 4],
            pooled: unit(&[0.4, 1.0, 0.1]),
        };
        let texts = LabelTexts { labels: vec![("parcel".into(), parcel, None), ("log".into(), log, None)], background: vec![grass] };
        (d, texts)
    }

    #[test]
    fn the_contrast_scoring_gives_grass_to_the_background_class() {
        let (d, texts) = dense();
        let cfg = LocateCfg { scoring: LocateScoring::Contrast, background: vec!["green grass".into()], ..LocateCfg::default() };
        let heats = label_heats(&d, &texts, &cfg, 100.0, -10.0);
        let parcel = &heats[0].1.values;
        assert!(parcel[0] > 0.99, "the parcel patch is the parcel's: {parcel:?}");
        assert!(parcel[1] < 0.01 && heats[1].1.values[1] < 0.01, "a grass patch belongs to the grass class");
        // Without the background class the same grass patch has to pick a label.
        let nobg = LabelTexts { background: Vec::new(), ..texts.clone() };
        let forced = label_heats(&d, &nobg, &cfg, 100.0, -10.0);
        assert!(forced[0].1.values[1] + forced[1].1.values[1] > 0.99);
        // The presence is SigLIP's own pooled pair probability, whatever the scoring.
        let want = 1.0 / (1.0 + (-(100.0 * dot(&d.pooled, &texts.labels[0].1) - 10.0)).exp());
        assert!((heats[0].2 - want).abs() < 1e-6);
    }

    #[test]
    fn the_pair_probability_scoring_is_the_old_heat_and_z_scores_standardise() {
        let (d, texts) = dense();
        let old = label_heats(&d, &texts, &LocateCfg::default(), 100.0, -10.0);
        let want = 1.0 / (1.0 + (-(100.0 * dot(&d.vectors[0], &texts.labels[0].1) - 10.0)).exp());
        assert!((old[0].1.values[0] - want).abs() < 1e-6);
        let z = label_heats(&d, &texts, &LocateCfg { scoring: LocateScoring::ZScore, ..LocateCfg::default() }, 100.0, -10.0);
        let v = &z[0].1.values;
        assert!(v.iter().sum::<f32>().abs() < 1e-4 && v[0] > 1.5, "{v:?}");
    }

    #[test]
    fn a_mode_name_gives_back_its_settings() {
        let bg = vec!["green grass".to_string()];
        let tpl = vec!["a photo of {}.".to_string()];
        for cfg in [
            LocateCfg::default(),
            LocateCfg { scoring: LocateScoring::Contrast, background: bg.clone(), ..LocateCfg::default() },
            LocateCfg { scoring: LocateScoring::Contrast, background: bg.clone(), templates: tpl.clone(), ..LocateCfg::default() },
            LocateCfg { scoring: LocateScoring::ZScore, ..LocateCfg::default() },
        ] {
            assert_eq!(LocateCfg::from_mode(&cfg.mode(), &bg, &tpl), Some(cfg.clone()), "{}", cfg.mode());
        }
        assert_eq!(LocateCfg::from_mode("stand-in(siglip2-like)", &bg, &tpl), None);
        assert_eq!(LocateCfg::from_mode("siglip2-dense+what", &bg, &tpl), None);
        assert_eq!(apply_template("a photo of {}.", "a log"), "a photo of a log.");
    }
}
