//! ML track 2: semantic location. "Where in this keyframe is the iron ore?"
//!
//! [`SiglipLocator`] uses SigLIP2 dense patch embeddings
//! (`noto_ml::Siglip2::embed_dense`): the cosine of every patch with a label's
//! text vector is a heat map from one forward pass. The peak is the semantic
//! position; it becomes the point prompt of the mask tracker. Negative
//! prompts (what a label is often confused with) are subtracted, as the
//! dictionary's `prejudice` phrases are elsewhere in the project.
//!
//! The scores are raw cosines turned into SigLIP pair probabilities. They have
//! not been calibrated on this renderer; the SDS `locate` track learns their
//! per-label baseline, which is what the runtime trusts.

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

/// SigLIP2 dense-patch locator.
pub struct SiglipLocator {
    model: Arc<noto_ml::Siglip2>,
    /// Text vectors per phrase, embedded once.
    cache: HashMap<String, Vec<f32>>,
    /// Weight of the negative phrases.
    pub negative_weight: f32,
}

impl SiglipLocator {
    pub fn new(model: Arc<noto_ml::Siglip2>) -> SiglipLocator {
        SiglipLocator { model, cache: HashMap::new(), negative_weight: 0.5 }
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
        "siglip2-dense"
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
        let mut texts: Vec<(String, Vec<f32>, Option<Vec<f32>>)> = Vec::new();
        for l in labels {
            let pos = self.text_vectors(&l.prompts)?;
            let Some(p) = mean_unit(&pos) else { continue };
            let neg = if l.negatives.is_empty() { None } else { mean_unit(&self.text_vectors(&l.negatives)?) };
            texts.push((l.id.clone(), p, neg));
        }
        let images: Vec<image::DynamicImage> = frames.iter().map(|f| f.to_dynamic()).collect();
        let dense = self.model.embed_dense_batch(&images)?;
        let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        Ok(frames
            .iter()
            .zip(&dense)
            .map(|(rgb, d)| {
                let (cols, rows) = (d.grid_w as u32, d.grid_h as u32);
                let heats = texts
                    .iter()
                    .map(|(id, p, neg)| {
                        let values = d
                            .vectors
                            .iter()
                            .map(|v| {
                                let cn = neg.as_ref().map(|n| dot(v, n)).unwrap_or(0.0);
                                self.model.pair_probability(dot(v, p) - self.negative_weight * cn.max(0.0))
                            })
                            .collect();
                        (id.clone(), Heat { cols, rows, values }, self.model.pair_probability(dot(&d.pooled, p)))
                    })
                    .collect();
                points_with_presence(rgb.width, rgb.height, heats)
            })
            .collect())
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
}
