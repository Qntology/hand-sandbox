use std::collections::HashMap;

use anyhow::Result;
use noto_search::{EntityKind, Lexicon};
use serde::{Deserialize, Serialize};
use world_sim::rng::Rng;
use world_sim::render::{unpack_id, View};
use world_sim::snapshot::{drop_id, node_id, waypoint_id};
use world_sim::{Camera, Frame, ObjectClass, World};

use crate::config::NoiseCfg;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntKind {
    Source,
    Drop,
    Waypoint,
}

impl From<&EntityKind> for EntKind {
    fn from(k: &EntityKind) -> Self {
        match k {
            EntityKind::Source => EntKind::Source,
            EntityKind::Drop => EntKind::Drop,
            EntityKind::Waypoint => EntKind::Waypoint,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    pub label: String,
    pub kind: EntKind,
    pub link: String,
    pub score: f32,
    pub margin: f32,
    pub runner_up: Option<String>,
    pub bbox: [f32; 4],
    pub pixels: u32,
    pub depth_m: f32,
    pub world: Option<(f32, f32)>,
    pub truth: Option<String>,
    pub truth_label: Option<String>,
}

impl Candidate {
    pub fn is_correct(&self) -> Option<bool> {
        self.truth_label.as_ref().map(|t| *t == self.label)
    }
}

pub struct ScanInput<'a> {
    pub world: &'a World,
    pub frame: &'a Frame,
    pub camera: &'a Camera,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ScanOutput {
    pub candidates: Vec<Candidate>,
    pub tiles: usize,
    pub embed_ms: f32,
    pub device: String,
    pub scene: Vec<(String, f32)>,
    #[serde(default)]
    pub filter: ScanFilter,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ScanFilter {
    pub skipped_sky: u32,
    pub skipped_far: u32,
    pub relabeled: u32,
    pub dropped: u32,
    pub merged: u32,
}

impl ScanFilter {
    pub fn is_empty(&self) -> bool {
        *self == ScanFilter::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SizePrior {
    pub lo_m: f32,
    pub hi_m: f32,
    pub elevated: bool,
}

pub trait Recognizer: Send {
    fn mode(&self) -> &str;
    fn scope(&self) -> String {
        self.mode().to_string()
    }
    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput>;
    fn set_noise(&mut self, _noise: &NoiseCfg) {}
}

#[derive(Clone, Debug, Default)]
pub struct EntityTable {
    map: HashMap<String, (EntKind, String)>,
    priors: HashMap<String, SizePrior>,
}

impl EntityTable {
    pub fn from_lexicon(lex: &Lexicon) -> EntityTable {
        EntityTable {
            map: lex.vision.iter().map(|e| (e.id.clone(), (EntKind::from(&e.kind), e.link.clone()))).collect(),
            priors: lex.vision.iter().filter_map(|e| e.height_m.map(|(lo_m, hi_m)| (e.id.clone(), SizePrior { lo_m, hi_m, elevated: e.elevated }))).collect(),
        }
    }

    pub fn get(&self, label: &str) -> Option<&(EntKind, String)> {
        self.map.get(label)
    }

    pub fn prior(&self, label: &str) -> Option<&SizePrior> {
        self.priors.get(label)
    }

    pub fn labels(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.map.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

pub fn centre_from_surface(view: &View, p: [f32; 3], r: f32) -> (f32, f32) {
    centre_from_surface_at(view.origin, p, r)
}

/// The ground point of an item's centre, `r` behind the surface point `p` an eye at `origin` sees of it: on along the
/// line of sight, less of `r` the steeper the eye looks down.
pub fn centre_from_surface_at(origin: [f32; 3], p: [f32; 3], r: f32) -> (f32, f32) {
    let (dx, dz) = (p[0] - origin[0], p[2] - origin[2]);
    let ground = dx.hypot(dz).max(1e-4);
    let theta = (origin[1] - p[1]).atan2(ground).max(0.0);
    let front = if theta > 0.45 {
        (theta.cos() / (theta.cos() + 1.5 * theta.sin())).max(0.15)
    } else {
        theta.cos() / (theta.cos() + theta.sin())
    };
    (p[0] + dx / ground * r * front, p[2] + dz / ground * r * front)
}

pub fn box_iou_and_iomin(a: &[f32; 4], b: &[f32; 4]) -> (f32, f32) {
    let ix0 = a[0].max(b[0]);
    let iy0 = a[1].max(b[1]);
    let ix1 = a[2].min(b[2]);
    let iy1 = a[3].min(b[3]);
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    if inter <= 0.0 {
        return (0.0, 0.0);
    }
    let area_a = ((a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)).max(1e-6);
    let area_b = ((b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)).max(1e-6);
    let union = (area_a + area_b - inter).max(1e-6);
    (inter / union, inter / area_a.min(area_b))
}

pub fn visible_centroids(frame: &Frame, camera: &Camera, view: &View) -> HashMap<String, [f32; 3]> {
    let mut acc: HashMap<u32, ([f64; 3], u32)> = HashMap::new();
    for (idx, &raw) in frame.ids.iter().enumerate() {
        let (class, _) = unpack_id(raw);
        if !matches!(class, ObjectClass::Node | ObjectClass::Drop | ObjectClass::Waypoint) {
            continue;
        }
        let (i, j) = (idx % frame.width, idx / frame.width);
        let Some(p) = camera.world_point3_in(view, i as f32 + 0.5, j as f32 + 0.5, frame.depth[idx]) else { continue };
        let a = acc.entry(raw).or_insert(([0.0; 3], 0));
        for k in 0..3 {
            a.0[k] += p[k] as f64;
        }
        a.1 += 1;
    }
    acc.into_iter()
        .map(|(raw, (s, n))| {
            let (class, id) = unpack_id(raw);
            let key = match class {
                ObjectClass::Node => node_id(id),
                ObjectClass::Drop => drop_id(id),
                _ => waypoint_id(id),
            };
            (key, [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32])
        })
        .collect()
}

pub fn centre_offset_m(kind: EntKind) -> f32 {
    match kind {
        EntKind::Source => 0.20,
        EntKind::Drop => 0.05,
        EntKind::Waypoint => 0.10,
    }
}

pub fn look_alike(label: &str) -> Option<&'static str> {
    Some(match label {
        "boulder" => "coal_ore",
        "coal_ore" => "iron_ore_node",
        "iron_ore_node" => "boulder",
        "tree" => "boulder",
        "log" => "stone",
        "stone" => "coal",
        "coal" => "iron_ore",
        "iron_ore" => "stone",
        "parcel" => "log",
        "base_camp" => "warehouse",
        "warehouse" => "base_camp",
        _ => return None,
    })
}

fn gauss(rng: &mut Rng) -> f32 {
    let u1 = rng.next_f32().max(1e-6);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanErrModel {
    pub det_a: f32,
    pub det_b: f32,
    pub false_per_scan: f32,
    pub pos_sd_m: f32,
    pub score_true: (f32, f32),
    pub score_false: (f32, f32),
    pub margin: (f32, f32),
    pub score_floor: f32,
    pub margin_floor: f32,
}

impl ScanErrModel {
    pub const SIGLIP_MEASURED: ScanErrModel = ScanErrModel {
        det_a: 0.0,
        det_b: 0.0,
        false_per_scan: 0.0,
        pos_sd_m: 0.0,
        score_true: (0.5, 0.0),
        score_false: (0.5, 0.0),
        margin: (0.1, 0.0),
        score_floor: 0.30,
        margin_floor: 0.04,
    };
}

pub struct OraclePerception {
    table: EntityTable,
    noise: NoiseCfg,
    rng: Rng,
    min_pixels: u32,
    honest: Option<ScanErrModel>,
}

impl OraclePerception {
    pub fn new(lex: &Lexicon, noise: NoiseCfg, min_pixels: u32) -> OraclePerception {
        let rng = Rng::new(noise.seed);
        OraclePerception { table: EntityTable::from_lexicon(lex), noise, rng, min_pixels, honest: None }
    }

    pub fn set_honest(&mut self, model: Option<ScanErrModel>) {
        self.honest = model;
    }

    pub fn honest(&self) -> Option<&ScanErrModel> {
        self.honest.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    fn measured_candidate(&mut self, m: &ScanErrModel, label: &str, bbox: [f32; 4], pixels: u32, depth_m: f32, world: Option<(f32, f32)>, right: bool, truth: Option<(String, String)>) -> Option<Candidate> {
        let (kind, link) = self.table.get(label).cloned()?;
        let (mean, sd) = if right { m.score_true } else { m.score_false };
        let score = (mean + sd * gauss(&mut self.rng)).clamp(m.score_floor, 0.99);
        let margin = (m.margin.0 + m.margin.1 * gauss(&mut self.rng)).clamp(m.margin_floor, 1.0);
        let world = world.map(|(x, z)| if right { (x + m.pos_sd_m * gauss(&mut self.rng), z + m.pos_sd_m * gauss(&mut self.rng)) } else { (x, z) });
        Some(Candidate {
            label: label.to_string(),
            kind,
            link,
            score,
            margin,
            runner_up: look_alike(label).map(String::from),
            bbox,
            pixels,
            depth_m,
            world,
            truth: truth.as_ref().map(|t| t.0.clone()),
            truth_label: truth.map(|t| t.1),
        })
    }

    fn scan_measured(&mut self, input: ScanInput<'_>, m: ScanErrModel) -> Result<ScanOutput> {
        let (w, h) = (input.frame.width as f32, input.frame.height as f32);
        let view = input.world.head_view(input.camera).est;
        let centroids = visible_centroids(input.frame, input.camera, &view);
        let mut out = ScanOutput { device: "cpu (oracle, measured)".into(), ..ScanOutput::default() };
        for v in input.world.visible_objects(input.frame) {
            let Some((kind, _)) = self.table.get(&v.label).cloned() else { continue };
            let bbox = [v.bbox[0] as f32 / w, v.bbox[1] as f32 / h, (v.bbox[2] + 1) as f32 / w, (v.bbox[3] + 1) as f32 / h];
            let (px, py) = ((bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h);
            let world = centroids.get(&v.entity).copied().or_else(|| input.camera.world_point3_in(&view, px, py, v.depth_m)).map(|p| centre_from_surface(&view, p, centre_offset_m(kind)));
            let p_det = if kind == EntKind::Waypoint { 1.0 } else { 1.0 / (1.0 + (-(m.det_a * (v.pixels.max(1) as f32).ln() + m.det_b)).exp()) };
            if self.rng.next_f32() < p_det {
                if let Some(c) = self.measured_candidate(&m, &v.label, bbox, v.pixels, v.depth_m, world, true, Some((v.entity.clone(), v.label.clone()))) {
                    out.candidates.push(c);
                }
            }
        }
        let mut k = 0u32;
        let (limit, mut p) = ((-m.false_per_scan.max(0.0)).exp(), 1.0f32);
        loop {
            p *= self.rng.next_f32();
            if p <= limit || k >= 64 {
                break;
            }
            k += 1;
        }
        let labels: Vec<String> = self.table.labels().into_iter().filter(|l| self.table.get(l).map_or(false, |(k, _)| *k != EntKind::Waypoint)).map(String::from).collect();
        for _ in 0..k {
            if labels.is_empty() {
                break;
            }
            let label = labels[self.rng.below(labels.len() as u32) as usize].clone();
            let (bw, bh) = (0.10 + 0.10 * self.rng.next_f32(), 0.12 + 0.10 * self.rng.next_f32());
            let (x0, y0) = (self.rng.range_f32(0.0, 1.0 - bw), self.rng.range_f32(0.45, 1.0 - bh));
            let bbox = [x0, y0, x0 + bw, y0 + bh];
            let (cx, cy) = (((bbox[0] + bbox[2]) * 0.5 * w) as usize, ((bbox[1] + bbox[3]) * 0.5 * h) as usize);
            let depth = input.frame.depth.get(cy.min(input.frame.height - 1) * input.frame.width + cx.min(input.frame.width - 1)).copied().unwrap_or(f32::INFINITY);
            let kind = self.table.get(&label).map(|e| e.0).unwrap_or(EntKind::Drop);
            let world = input.camera.world_point3_in(&view, (bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h, depth).map(|p| centre_from_surface(&view, p, centre_offset_m(kind)));
            if let Some(c) = self.measured_candidate(&m, &label, bbox, (bw * w * bh * h) as u32, depth, world, false, None) {
                out.candidates.push(c);
            }
        }
        out.candidates.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.label.cmp(&b.label)));
        Ok(out)
    }

    pub fn noise(&self) -> &NoiseCfg {
        &self.noise
    }

    pub fn set_noise(&mut self, noise: NoiseCfg) {
        self.rng = Rng::new(noise.seed);
        self.noise = noise;
    }

    fn base_confidence(pixels: u32) -> f32 {
        0.55 + 0.30 * ((pixels as f32).sqrt() / 12.0).clamp(0.0, 1.0)
    }
}

impl Recognizer for OraclePerception {
    fn mode(&self) -> &str {
        "oracle"
    }

    fn scope(&self) -> String {
        if self.honest.is_some() {
            "oracle(siglip2-measured)".into()
        } else {
            "oracle".into()
        }
    }

    fn set_noise(&mut self, noise: &NoiseCfg) {
        OraclePerception::set_noise(self, noise.clone());
    }

    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput> {
        if let Some(m) = self.honest {
            return self.scan_measured(input, m);
        }
        let (w, h) = (input.frame.width as f32, input.frame.height as f32);
        let view = input.world.head_view(input.camera).est;
        let centroids = visible_centroids(input.frame, input.camera, &view);
        let mut out = ScanOutput { device: "cpu (oracle)".into(), ..ScanOutput::default() };
        let sd = self.noise.score_sd;

        for v in input.world.visible_objects(input.frame) {
            if v.pixels < self.min_pixels {
                continue;
            }
            if self.rng.next_f32() < self.noise.miss_rate {
                continue;
            }
            let Some((kind, _)) = self.table.get(&v.label).cloned() else { continue };
            let conf = Self::base_confidence(v.pixels);
            let confuse = self.rng.next_f32() < self.noise.confusion_rate;
            let (label, score, margin, runner) = match look_alike(&v.label).filter(|_| confuse) {
                Some(other) => (
                    other.to_string(),
                    (0.46 + 0.10 * conf + sd * gauss(&mut self.rng)).clamp(0.0, 0.99),
                    (0.03 + 0.03 * gauss(&mut self.rng)).abs(),
                    Some(v.label.clone()),
                ),
                None => (
                    v.label.clone(),
                    (conf + sd * gauss(&mut self.rng)).clamp(0.0, 0.99),
                    (0.22 + 0.5 * sd * gauss(&mut self.rng)).clamp(0.0, 1.0),
                    look_alike(&v.label).map(String::from),
                ),
            };
            let Some((ckind, link)) = self.table.get(&label).cloned() else { continue };
            let bbox = [v.bbox[0] as f32 / w, v.bbox[1] as f32 / h, (v.bbox[2] + 1) as f32 / w, (v.bbox[3] + 1) as f32 / h];
            let (px, py) = ((bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h);
            let world = centroids.get(&v.entity).copied().or_else(|| input.camera.world_point3_in(&view, px, py, v.depth_m)).map(|p| centre_from_surface(&view, p, centre_offset_m(ckind)));
            out.candidates.push(Candidate {
                label,
                kind: ckind,
                link,
                score,
                margin,
                runner_up: runner,
                bbox,
                pixels: v.pixels,
                depth_m: v.depth_m,
                world,
                truth: Some(v.entity.clone()),
                truth_label: Some(v.label.clone()),
            });
            let _ = kind;
        }

        if self.rng.next_f32() < self.noise.false_positive_rate {
            let labels: Vec<&str> = self.table.labels().into_iter().filter(|l| self.table.get(l).map_or(false, |(k, _)| *k != EntKind::Waypoint)).collect();
            if !labels.is_empty() {
                let label = labels[self.rng.below(labels.len() as u32) as usize].to_string();
                let (kind, link) = self.table.get(&label).cloned().unwrap();
                let (bw, bh) = (0.10 + 0.10 * self.rng.next_f32(), 0.12 + 0.10 * self.rng.next_f32());
                let (x0, y0) = (self.rng.range_f32(0.0, 1.0 - bw), self.rng.range_f32(0.45, 1.0 - bh));
                let bbox = [x0, y0, x0 + bw, y0 + bh];
                let (cx, cy) = (((bbox[0] + bbox[2]) * 0.5 * w) as usize, ((bbox[1] + bbox[3]) * 0.5 * h) as usize);
                let depth = input.frame.depth.get(cy.min(input.frame.height - 1) * input.frame.width + cx.min(input.frame.width - 1)).copied().unwrap_or(f32::INFINITY);
                let world = input.camera.world_point3_in(&view, (bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h, depth).map(|p| centre_from_surface(&view, p, centre_offset_m(kind)));
                out.candidates.push(Candidate {
                    label,
                    kind,
                    link,
                    score: (0.42 + 0.5 * sd * gauss(&mut self.rng) + 0.04).clamp(0.0, 0.99),
                    margin: (0.03 + 0.03 * gauss(&mut self.rng)).abs(),
                    runner_up: None,
                    bbox,
                    pixels: (bw * w * bh * h) as u32,
                    depth_m: depth,
                    world,
                    truth: None,
                    truth_label: None,
                });
            }
        }
        out.candidates.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.label.cmp(&b.label)));
        Ok(out)
    }
}

pub fn is_thing(class: ObjectClass) -> bool {
    matches!(class, ObjectClass::Node | ObjectClass::Drop | ObjectClass::Waypoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use world_sim::kinds::NodeKind;
    use world_sim::world::Node;
    use world_sim::{WorldConfig};

    pub(crate) fn lexicon() -> Lexicon {
        Lexicon::from_json(include_str!("../../../data/world_lexicon.json")).expect("lexicon loads")
    }

    pub(crate) fn world_with_tree_ahead() -> (World, Camera) {
        let mut w = World::new(WorldConfig::default()).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let (cx, cz) = w.terrain.cell_of(w.agent.x + 3.2, w.agent.z + 0.2).unwrap();
        let (x, z) = world_sim::terrain::cell_center(cx, cz);
        w.nodes.push(Node { id: 1, kind: NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 });
        w.agent.yaw = 0.0;
        (w, Camera { width: 192, height: 128, ..Camera::default() })
    }

    fn scan_once(p: &mut OraclePerception, w: &World, cam: &Camera) -> ScanOutput {
        let f = w.render(cam);
        p.scan(ScanInput { world: w, frame: &f, camera: cam }).unwrap()
    }

    #[test]
    fn without_noise_the_oracle_reports_the_truth() {
        let (w, cam) = world_with_tree_ahead();
        let mut p = OraclePerception::new(&lexicon(), NoiseCfg::none(), 6);
        let s = scan_once(&mut p, &w, &cam);
        let tree = s.candidates.iter().find(|c| c.truth.as_deref() == Some("n1")).expect("the tree is reported");
        assert_eq!(tree.label, "tree");
        assert_eq!(tree.link, "log");
        assert_eq!(tree.kind, EntKind::Source);
        assert_eq!(tree.is_correct(), Some(true));
        assert!(tree.score > 0.6 && tree.margin > 0.15);
        let (x, z) = tree.world.expect("has a ground position");
        let n = w.node(1).unwrap();
        assert!(((x - n.x).powi(2) + (z - n.z).powi(2)).sqrt() < 0.45, "estimated ({x:.2},{z:.2}) vs ({:.2},{:.2})", n.x, n.z);
    }

    #[test]
    fn the_same_seed_gives_the_same_noisy_scan() {
        let (w, cam) = world_with_tree_ahead();
        let a = scan_once(&mut OraclePerception::new(&lexicon(), NoiseCfg::demo(), 6), &w, &cam);
        let b = scan_once(&mut OraclePerception::new(&lexicon(), NoiseCfg::demo(), 6), &w, &cam);
        assert_eq!(a.candidates, b.candidates);
    }

    #[test]
    fn noise_produces_wrong_labels_and_phantoms_at_roughly_the_configured_rates() {
        let (w, cam) = world_with_tree_ahead();
        let noise = NoiseCfg { seed: 5, score_sd: 0.05, confusion_rate: 0.3, false_positive_rate: 0.5, miss_rate: 0.0 };
        let mut p = OraclePerception::new(&lexicon(), noise, 6);
        let f = w.render(&cam);
        let (mut wrong, mut real, mut phantom, scans) = (0, 0, 0, 400);
        for _ in 0..scans {
            let s = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
            for c in s.candidates {
                match (&c.truth, c.is_correct()) {
                    (None, _) => phantom += 1,
                    (Some(_), Some(false)) => {
                        wrong += 1;
                        real += 1
                    }
                    (Some(_), _) => real += 1,
                }
            }
        }
        let wrong_rate = wrong as f32 / real as f32;
        let phantom_rate = phantom as f32 / scans as f32;
        assert!((0.2..0.4).contains(&wrong_rate), "wrong label rate {wrong_rate}");
        assert!((0.38..0.62).contains(&phantom_rate), "phantom rate {phantom_rate}");
    }

    #[test]
    fn wrong_labels_score_lower_than_right_ones_on_average() {
        let (w, cam) = world_with_tree_ahead();
        let noise = NoiseCfg { seed: 8, score_sd: 0.05, confusion_rate: 0.4, false_positive_rate: 0.0, miss_rate: 0.0 };
        let mut p = OraclePerception::new(&lexicon(), noise, 6);
        let f = w.render(&cam);
        let (mut good, mut bad) = (vec![], vec![]);
        for _ in 0..300 {
            for c in p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap().candidates {
                if c.is_correct() == Some(true) {
                    good.push((c.score, c.margin));
                } else {
                    bad.push((c.score, c.margin));
                }
            }
        }
        let mean = |v: &[(f32, f32)], k: usize| v.iter().map(|x| if k == 0 { x.0 } else { x.1 }).sum::<f32>() / v.len() as f32;
        assert!(mean(&good, 0) > mean(&bad, 0) + 0.1);
        assert!(mean(&good, 1) > mean(&bad, 1) + 0.1);
    }

    #[test]
    fn tiny_objects_are_not_reported() {
        let (mut w, cam) = world_with_tree_ahead();
        w.nodes[0].x += 18.0;
        let mut p = OraclePerception::new(&lexicon(), NoiseCfg::none(), 400);
        assert!(scan_once(&mut p, &w, &cam).candidates.is_empty());
    }

    #[test]
    fn the_honest_oracle_reports_and_errs_as_its_measured_model_says() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let sure = ScanErrModel { det_a: 0.0, det_b: 50.0, false_per_scan: 0.0, pos_sd_m: 0.0, score_true: (0.7, 0.0), score_false: (0.4, 0.0), margin: (0.1, 0.0), score_floor: 0.3, margin_floor: 0.04 };
        let mut p = OraclePerception::new(&lexicon(), NoiseCfg::none(), 6);
        assert_eq!(Recognizer::scope(&p), "oracle");
        p.set_honest(Some(sure));
        assert_eq!(Recognizer::scope(&p), "oracle(siglip2-measured)", "its statistics stay apart");
        let s = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        let tree = s.candidates.iter().find(|c| c.truth.as_deref() == Some("n1")).expect("always detected");
        assert_eq!(tree.label, "tree");
        assert!((tree.score - 0.7).abs() < 1e-6 && (tree.margin - 0.1).abs() < 1e-6);
        let blind = ScanErrModel { det_b: -50.0, false_per_scan: 2.0, score_false: (0.2, 0.1), ..sure };
        p.set_honest(Some(blind));
        let (mut wrong, scans) = (0usize, 300);
        for _ in 0..scans {
            let s = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
            let things: Vec<&Candidate> = s.candidates.iter().filter(|c| c.kind != EntKind::Waypoint).collect();
            assert!(things.iter().all(|c| c.truth.is_none() && c.score >= 0.3 && c.margin >= 0.04));
            wrong += things.len();
        }
        let rate = wrong as f32 / scans as f32;
        assert!((1.7..2.3).contains(&rate), "{rate} wrong per scan");
    }

    #[test]
    fn look_alikes_exist_for_every_dictionary_label() {
        let lex = lexicon();
        for l in EntityTable::from_lexicon(&lex).labels() {
            assert!(look_alike(l).is_some(), "{l} has no look-alike");
        }
    }
}
