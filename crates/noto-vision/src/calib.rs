use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const ECE_BINS: usize = 15;

pub fn auroc(scores: &[f32], hits: &[bool]) -> Option<f32> {
    let mut v: Vec<(f32, bool)> = scores.iter().copied().zip(hits.iter().copied()).filter(|(s, _)| s.is_finite()).collect();
    let pos = v.iter().filter(|x| x.1).count();
    let neg = v.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut rank_sum = 0.0f64;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if v[k].1 {
                rank_sum += avg;
            }
        }
        i = j + 1;
    }
    let u = rank_sum - (pos * (pos + 1)) as f64 / 2.0;
    Some((u / (pos * neg) as f64) as f32)
}

pub fn average_precision(scores: &[f32], hits: &[bool]) -> Option<f32> {
    let mut v: Vec<(f32, bool)> = scores.iter().copied().zip(hits.iter().copied()).filter(|(s, _)| s.is_finite()).collect();
    let pos = v.iter().filter(|x| x.1).count();
    if pos == 0 {
        return None;
    }
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut tp, mut ap) = (0usize, 0.0f64);
    for (k, (_, h)) in v.iter().enumerate() {
        if *h {
            tp += 1;
            ap += tp as f64 / (k + 1) as f64;
        }
    }
    Some((ap / pos as f64) as f32)
}

pub fn ece(probs: &[f32], hits: &[bool], bins: usize) -> Option<f32> {
    let n = probs.len().min(hits.len());
    if n == 0 || bins == 0 {
        return None;
    }
    let mut sum_p = vec![0.0f64; bins];
    let mut sum_h = vec![0.0f64; bins];
    let mut cnt = vec![0usize; bins];
    for (p, h) in probs.iter().zip(hits) {
        let b = ((p.clamp(0.0, 1.0) * bins as f32) as usize).min(bins - 1);
        sum_p[b] += *p as f64;
        sum_h[b] += *h as u8 as f64;
        cnt[b] += 1;
    }
    let e: f64 = (0..bins).filter(|&b| cnt[b] > 0).map(|b| (sum_h[b] - sum_p[b]).abs()).sum::<f64>() / n as f64;
    Some(e as f32)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn platt_fit(scores: &[f32], hits: &[bool]) -> Option<(f32, f32)> {
    let data: Vec<(f64, bool)> = scores.iter().zip(hits).filter(|(s, _)| s.is_finite()).map(|(s, h)| (*s as f64, *h)).collect();
    let pos = data.iter().filter(|d| d.1).count() as f64;
    let neg = data.len() as f64 - pos;
    if pos == 0.0 || neg == 0.0 {
        return None;
    }
    let (t1, t0) = ((pos + 1.0) / (pos + 2.0), 1.0 / (neg + 2.0));
    let (mut a, mut b) = (0.0f64, ((pos + 1.0) / (neg + 1.0)).ln());
    for _ in 0..100 {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 1e-9, 0.0, 1e-9);
        for &(s, h) in &data {
            let p = sigmoid(a * s + b);
            let t = if h { t1 } else { t0 };
            let d = p - t;
            let w = p * (1.0 - p);
            ga += d * s;
            gb += d;
            haa += w * s * s;
            hab += w * s;
            hbb += w;
        }
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-18 {
            break;
        }
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        a -= da;
        b -= db;
        if da.abs() < 1e-9 && db.abs() < 1e-9 {
            break;
        }
    }
    (a.is_finite() && b.is_finite()).then_some((a as f32, b as f32))
}

pub fn threshold_for_precision(scores: &[f32], hits: &[bool], precision: f32) -> Option<(f32, f32)> {
    let mut v: Vec<(f32, bool)> = scores.iter().copied().zip(hits.iter().copied()).filter(|(s, _)| s.is_finite()).collect();
    let pos = v.iter().filter(|x| x.1).count();
    if pos == 0 {
        return None;
    }
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut tp, mut best) = (0usize, None);
    for k in 0..v.len() {
        tp += v[k].1 as usize;
        if k + 1 < v.len() && v[k + 1].0 == v[k].0 {
            continue;
        }
        if tp as f32 / (k + 1) as f32 >= precision {
            best = Some((v[k].0, tp as f32 / pos as f32));
        }
    }
    best
}

pub fn threshold_for_fpr(scores: &[f32], hits: &[bool], fpr: f32) -> Option<f32> {
    let mut neg: Vec<f32> = scores.iter().zip(hits).filter(|(s, h)| !**h && s.is_finite()).map(|(s, _)| *s).collect();
    if neg.is_empty() {
        return None;
    }
    neg.sort_by(|a, b| b.total_cmp(a));
    let k = ((fpr.clamp(0.0, 1.0) * neg.len() as f32).floor() as usize).min(neg.len() - 1);
    Some(next_up(neg[k]))
}

fn next_up(x: f32) -> f32 {
    if x.is_finite() {
        f32::from_bits(if x >= 0.0 { x.to_bits() + 1 } else { x.to_bits() - 1 })
    } else {
        x
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelCalib {
    pub threshold: f32,
    #[serde(default)]
    pub platt: Option<(f32, f32)>,
    pub positives: u32,
    pub negatives: u32,
    #[serde(default)]
    pub precision: Option<f32>,
    #[serde(default)]
    pub recall: Option<f32>,
}

impl LabelCalib {
    pub fn probability(&self, score: f32) -> Option<f32> {
        self.platt.map(|(a, b)| sigmoid((a * score + b) as f64) as f32)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibTable {
    pub mode: String,
    #[serde(default)]
    pub statistic: String,
    #[serde(default)]
    pub note: String,
    pub labels: BTreeMap<String, LabelCalib>,
}

impl CalibTable {
    pub fn threshold(&self, label: &str) -> Option<f32> {
        self.labels.get(label).map(|c| c.threshold)
    }

    pub fn load(path: &Path) -> Result<Vec<CalibTable>> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(raw: &str) -> Result<Vec<CalibTable>> {
        Ok(serde_json::from_str(raw)?)
    }
}

pub fn fit_label(scores: &[f32], hits: &[bool], precision: f32, fpr: f32) -> Option<LabelCalib> {
    let positives = hits.iter().filter(|h| **h).count() as u32;
    let negatives = hits.len() as u32 - positives;
    let threshold = match threshold_for_precision(scores, hits, precision) {
        Some((t, _)) => t,
        None => threshold_for_fpr(scores, hits, fpr)?,
    };
    let (mut tp, mut fp) = (0u32, 0u32);
    for (s, h) in scores.iter().zip(hits) {
        if *s >= threshold {
            if *h {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    Some(LabelCalib {
        threshold,
        platt: platt_fit(scores, hits),
        positives,
        negatives,
        precision: (tp + fp > 0).then(|| tp as f32 / (tp + fp) as f32),
        recall: (positives > 0).then(|| tp as f32 / positives as f32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_metrics_on_a_known_case() {
        let s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let h = [true, false, true, false, false, false];
        assert!((auroc(&s, &h).unwrap() - 7.0 / 8.0).abs() < 1e-6);
        assert!((average_precision(&s, &h).unwrap() - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-6);
        assert!(auroc(&s, &[true; 6]).is_none(), "one class only");
        assert!((auroc(&[0.5, 0.5], &[true, false]).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn thresholds_meet_their_precision_and_false_positive_rate() {
        let s = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
        let h = [true, true, false, true, true, false, false, false];
        let (t, recall) = threshold_for_precision(&s, &h, 0.75).unwrap();
        assert_eq!(t, 0.7, "four of the top five are hits");
        assert!((recall - 1.0).abs() < 1e-6);
        assert_eq!(threshold_for_precision(&s, &h, 1.0).unwrap().0, 0.9);
        let t = threshold_for_fpr(&s, &h, 0.25).unwrap();
        assert!(t > 0.6 && t <= 0.85 + 1e-6, "{t}");
        let fit = fit_label(&s, &h, 0.75, 0.1).unwrap();
        assert_eq!((fit.positives, fit.negatives), (4, 4));
        assert_eq!(fit.precision, Some(0.8));
    }

    #[test]
    fn platt_recovers_a_logistic_and_ece_is_small_when_calibrated() {
        let (mut s, mut h) = (Vec::new(), Vec::new());
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let p = 1.0 / (1.0 + (-(8.0 * x - 4.0)).exp());
            let k = (p * 50.0).round() as usize;
            for j in 0..50 {
                s.push(x);
                h.push(j < k);
            }
        }
        let (a, b) = platt_fit(&s, &h).unwrap();
        assert!((a - 8.0).abs() < 0.3 && (b + 4.0).abs() < 0.2, "{a} {b}");
        let probs: Vec<f32> = s.iter().map(|x| 1.0 / (1.0 + (-(a * x + b)).exp())).collect();
        assert!(ece(&probs, &h, ECE_BINS).unwrap() < 0.02);
        assert!(ece(&s, &h, ECE_BINS).unwrap() > ece(&probs, &h, ECE_BINS).unwrap());
    }

    #[test]
    fn tables_round_trip_through_json() {
        let mut t = CalibTable { mode: "siglip2-dense+bg".into(), statistic: "peak".into(), ..CalibTable::default() };
        t.labels.insert("parcel".into(), LabelCalib { threshold: 0.4, platt: Some((5.0, -2.0)), positives: 10, negatives: 90, precision: Some(0.9), recall: Some(0.5) });
        let raw = serde_json::to_string(&vec![t.clone()]).unwrap();
        let back = CalibTable::parse(&raw).unwrap();
        assert_eq!(back, vec![t.clone()]);
        assert_eq!(back[0].threshold("parcel"), Some(0.4));
        assert!((t.labels["parcel"].probability(0.4).unwrap() - 0.5).abs() < 1e-6);
    }
}
