//! What the robot sees. A [`Recognizer`] turns a rendered frame into
//! [`Candidate`]s: a label from the dictionary, a score, a margin over the
//! runner-up, a box, and where that is on the ground. Candidates are not
//! trusted yet; `precision` decides which ones are.
//!
//! [`OraclePerception`] is the stand-in used without model weights. It reads the
//! renderer's id buffer (which says what every pixel truly is) and then spoils
//! it on purpose: score noise, look-alike labels, misses and phantom detections.
//! Its scores come from a made-up model (bigger on screen means surer), so the
//! tests that use it show the gates behave consistently, not how a real
//! recogniser performs. `embedding::EmbeddingPerception` is the real path.

use std::collections::HashMap;

use anyhow::Result;
use hand_search::{EntityKind, Lexicon};
use serde::{Deserialize, Serialize};
use world_sim::rng::Rng;
use world_sim::{Camera, Frame, ObjectClass, World};

use crate::config::NoiseCfg;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntKind {
    /// A resource node standing in the world.
    Source,
    /// An item lying on the ground.
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

/// A detection that has not been judged yet.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Candidate {
    /// Dictionary entity id: `tree`, `boulder`, `log`, `warehouse`, ...
    pub label: String,
    pub kind: EntKind,
    /// Resource id for sources and drops, place id for waypoints.
    pub link: String,
    pub score: f32,
    /// Score of this label minus the runner-up's.
    pub margin: f32,
    pub runner_up: Option<String>,
    /// Box as fractions of the frame: `[x0, y0, x1, y1]`.
    pub bbox: [f32; 4],
    pub pixels: u32,
    pub depth_m: f32,
    /// Estimated ground position `(x, z)`.
    pub world: Option<(f32, f32)>,
    /// What was really there (`n12`, `d3`, `w1`). Only the oracle knows this.
    /// It is for scoring tests and the evaluation counters; nothing that
    /// decides what the robot does may read it.
    pub truth: Option<String>,
    /// The true label of that object.
    pub truth_label: Option<String>,
}

impl Candidate {
    /// True when the label is the true one (needs ground truth).
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
    /// Image crops that went through the vision tower.
    pub tiles: usize,
    pub embed_ms: f32,
    pub device: String,
    /// Scores of every entity for the whole frame, best first (label, score).
    pub scene: Vec<(String, f32)>,
}

pub trait Recognizer: Send {
    /// `oracle` or `embedding`.
    fn mode(&self) -> &str;
    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput>;
    /// Changes the injected errors. Only the stand-in recogniser has any.
    fn set_noise(&mut self, _noise: &NoiseCfg) {}
}

/// Entity table from the dictionary: label to (kind, link).
#[derive(Clone, Debug, Default)]
pub struct EntityTable {
    map: HashMap<String, (EntKind, String)>,
}

impl EntityTable {
    pub fn from_lexicon(lex: &Lexicon) -> EntityTable {
        EntityTable { map: lex.vision.iter().map(|e| (e.id.clone(), (EntKind::from(&e.kind), e.link.clone()))).collect() }
    }

    pub fn get(&self, label: &str) -> Option<&(EntKind, String)> {
        self.map.get(label)
    }

    pub fn labels(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.map.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

/// Distance of the object's centre behind the surface the camera sees, metres.
pub fn centre_offset_m(kind: EntKind) -> f32 {
    match kind {
        EntKind::Source => 0.20,
        EntKind::Drop => 0.05,
        EntKind::Waypoint => 0.10,
    }
}

/// The label a recogniser tends to confuse `label` with.
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
        // A brown box on the ground reads like a brown log.
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

pub struct OraclePerception {
    table: EntityTable,
    noise: NoiseCfg,
    rng: Rng,
    min_pixels: u32,
}

impl OraclePerception {
    pub fn new(lex: &Lexicon, noise: NoiseCfg, min_pixels: u32) -> OraclePerception {
        let rng = Rng::new(noise.seed);
        OraclePerception { table: EntityTable::from_lexicon(lex), noise, rng, min_pixels }
    }

    pub fn noise(&self) -> &NoiseCfg {
        &self.noise
    }

    pub fn set_noise(&mut self, noise: NoiseCfg) {
        self.rng = Rng::new(noise.seed);
        self.noise = noise;
    }

    /// Made-up confidence: the more pixels, the surer.
    fn base_confidence(pixels: u32) -> f32 {
        0.55 + 0.30 * ((pixels as f32).sqrt() / 12.0).clamp(0.0, 1.0)
    }
}

impl Recognizer for OraclePerception {
    fn mode(&self) -> &str {
        "oracle"
    }

    fn set_noise(&mut self, noise: &NoiseCfg) {
        OraclePerception::set_noise(self, noise.clone());
    }

    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput> {
        let (w, h) = (input.frame.width as f32, input.frame.height as f32);
        let agent = (input.world.agent.x, input.world.agent.z);
        let yaw = input.world.agent.yaw;
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
            let world = input.camera.world_point(agent, yaw, px, py, v.depth_m + centre_offset_m(ckind));
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

        // A phantom detection where nothing is.
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
                let world = input.camera.world_point(agent, yaw, (bbox[0] + bbox[2]) * 0.5 * w, (bbox[1] + bbox[3]) * 0.5 * h, depth);
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

/// Whether a frame pixel class is something the recogniser could name.
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
        // far away the tree covers only a few pixels
        w.nodes[0].x += 18.0;
        let mut p = OraclePerception::new(&lexicon(), NoiseCfg::none(), 400);
        assert!(scan_once(&mut p, &w, &cam).candidates.is_empty());
    }

    #[test]
    fn look_alikes_exist_for_every_dictionary_label() {
        let lex = lexicon();
        for l in EntityTable::from_lexicon(&lex).labels() {
            assert!(look_alike(l).is_some(), "{l} has no look-alike");
        }
    }
}
