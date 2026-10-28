use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use noto_ml::parallel::{embed_frame_tiles, BatchOpts, ImageEmbedder, PhraseBank, TextEmbedder, Tile};
use noto_ml::Siglip2;
use noto_search::semantic::PhraseEmbedder;
use noto_search::vision::EntityScore;
use noto_search::{Lexicon, VisionBank};
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use world_sim::render::View;
use world_sim::{decode_webp, encode_webp, Camera, Frame};

use crate::perception::{centre_from_surface, centre_offset_m, Candidate, EntityTable, Recognizer, ScanInput, ScanOutput};

pub const SCAN_CALIB_MODE: &str = "embedding";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingCfg {
    pub grids: Vec<(u32, u32)>,
    pub overlap: f32,
    pub prejudice_weight: f32,
    pub report_floor: f32,
    pub nms_iou: f32,
    pub webp_quality: f32,
}

impl Default for EmbeddingCfg {
    fn default() -> Self {
        EmbeddingCfg { grids: vec![(2, 2), (4, 3), (6, 4)], overlap: 0.15, prejudice_weight: 0.5, report_floor: 0.10, nms_iou: 0.4, webp_quality: 80.0 }
    }
}

pub struct BankedPhrases<E: TextEmbedder + Send + Sync> {
    embedder: Arc<E>,
    bank: PhraseBank,
    opts: BatchOpts,
}

impl<E: TextEmbedder + Send + Sync> BankedPhrases<E> {
    pub fn new(embedder: Arc<E>, dir: Option<&Path>) -> Self {
        let bank = match dir {
            Some(d) => PhraseBank::open(d),
            None => PhraseBank::memory(),
        };
        BankedPhrases { embedder, bank, opts: BatchOpts::default() }
    }

    pub fn bank(&self) -> &PhraseBank {
        &self.bank
    }
}

impl<E: TextEmbedder + Send + Sync> PhraseEmbedder for BankedPhrases<E> {
    fn embed_phrases(&self, phrases: &[String]) -> Result<Vec<Arc<[f32]>>> {
        self.bank.embed(&*self.embedder, phrases, &self.opts).map(|(v, _)| v)
    }
}

pub type SiglipPhrases = BankedPhrases<Siglip2>;

pub struct EmbeddingPerception {
    image: Arc<dyn ImageEmbedder + Send + Sync>,
    bank: VisionBank,
    table: EntityTable,
    calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
    cfg: EmbeddingCfg,
    device: String,
    calib: Option<noto_vision::CalibTable>,
}

impl EmbeddingPerception {
    pub fn new(
        lex: &Lexicon,
        text: &dyn PhraseEmbedder,
        image: Arc<dyn ImageEmbedder + Send + Sync>,
        calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
        device: &str,
        cfg: EmbeddingCfg,
    ) -> Result<EmbeddingPerception> {
        EmbeddingPerception::new_with(lex, text, image, calibrate, device, cfg, None)
    }

    pub fn new_with(
        lex: &Lexicon,
        text: &dyn PhraseEmbedder,
        image: Arc<dyn ImageEmbedder + Send + Sync>,
        calibrate: Arc<dyn Fn(f32) -> f32 + Send + Sync>,
        device: &str,
        cfg: EmbeddingCfg,
        prompting: Option<&noto_search::VisionPrompting>,
    ) -> Result<EmbeddingPerception> {
        let bank = VisionBank::build_with(lex, text, prompting)?;
        Ok(EmbeddingPerception { image, bank, table: EntityTable::from_lexicon(lex), calibrate, cfg, device: device.to_string(), calib: None })
    }

    pub fn set_calibration(&mut self, table: Option<noto_vision::CalibTable>) {
        self.calib = table;
    }

    pub fn calibration(&self) -> Option<&noto_vision::CalibTable> {
        self.calib.as_ref()
    }

    fn calibrated(&self, label: &str, kind: &noto_search::EntityKind, score: f32) -> f32 {
        let Some(t) = &self.calib else { return score };
        if *kind == noto_search::EntityKind::Waypoint {
            return score;
        }
        t.labels.get(label).or_else(|| t.labels.get("*")).and_then(|c| c.probability(score)).unwrap_or(score)
    }

    pub fn siglip(model: Arc<Siglip2>, lex: &Lexicon, bank_dir: Option<&Path>, cfg: EmbeddingCfg) -> Result<EmbeddingPerception> {
        EmbeddingPerception::siglip_with(model, lex, bank_dir, cfg, None)
    }

    pub fn siglip_with(model: Arc<Siglip2>, lex: &Lexicon, bank_dir: Option<&Path>, cfg: EmbeddingCfg, prompting: Option<&noto_search::VisionPrompting>) -> Result<EmbeddingPerception> {
        let phrases = BankedPhrases::new(Arc::clone(&model), bank_dir);
        let device = noto_ml::device_label(model.device());
        let m = Arc::clone(&model);
        EmbeddingPerception::new_with(lex, &phrases, model, Arc::new(move |c| m.pair_probability(c)), &device, cfg, prompting)
    }

    pub fn cfg(&self) -> &EmbeddingCfg {
        &self.cfg
    }
}

impl Recognizer for EmbeddingPerception {
    fn mode(&self) -> &str {
        "embedding"
    }

    fn scope(&self) -> String {
        if self.calib.is_some() {
            "embedding+cal".into()
        } else {
            "embedding".into()
        }
    }

    fn scan(&mut self, input: ScanInput<'_>) -> Result<ScanOutput> {
        let f = input.frame;
        let (w, h) = (f.width as u32, f.height as u32);
        let enc = encode_webp(&f.rgb, w, h, self.cfg.webp_quality).map_err(|e| anyhow!("스크린샷 WebP 인코딩 실패: {e}"))?;
        let (dw, dh, rgb) = decode_webp(&enc.bytes).map_err(|e| anyhow!("스크린샷 WebP 디코딩 실패: {e}"))?;
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(dw, dh, rgb).ok_or_else(|| anyhow!("디코딩한 프레임의 크기가 맞지 않습니다"))?);

        let tiles = noto_ml::parallel::tile_layout(dw, dh, &self.cfg.grids, self.cfg.overlap);
        let t0 = Instant::now();
        let vecs = embed_frame_tiles(&*self.image, &[(&img, &tiles)])?.pop().unwrap_or_default();
        let embed_ms = t0.elapsed().as_secs_f32() * 1000.0;
        if vecs.len() != tiles.len() {
            return Err(anyhow!("타일 {}개에 벡터 {}개가 돌아왔습니다", tiles.len(), vecs.len()));
        }
        let mut scores: Vec<Vec<EntityScore>> = vecs.iter().map(|v| self.bank.score(v, &*self.calibrate, self.cfg.prejudice_weight)).collect();
        if self.calib.is_some() {
            for sc in scores.iter_mut() {
                for e in sc.iter_mut() {
                    e.score = self.calibrated(&e.id, &e.kind, e.score);
                }
                sc.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }

        let scene = scores.first().map(|s| s.iter().map(|e| (e.id.clone(), e.score)).collect()).unwrap_or_default();
        let view = input.world.head_view(input.camera).est;
        let candidates = candidates_from_tiles(&tiles, &scores, &self.table, f, input.camera, &view, &self.cfg);
        Ok(ScanOutput { candidates, tiles: tiles.len(), embed_ms, device: self.device.clone(), scene })
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (ix0, iy0, ix1, iy1) = (a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3]));
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn median_depth(f: &Frame, t: &Tile) -> Option<f32> {
    let (x0, x1) = (t.x + t.w / 4, (t.x + t.w * 3 / 4).max(t.x + t.w / 4 + 1));
    let (y0, y1) = (t.y + t.h / 4, (t.y + t.h * 3 / 4).max(t.y + t.h / 4 + 1));
    let mut v: Vec<f32> = Vec::new();
    for y in y0..y1.min(f.height as u32) {
        for x in x0..x1.min(f.width as u32) {
            let d = f.depth[y as usize * f.width + x as usize];
            if d.is_finite() {
                v.push(d);
            }
        }
    }
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

#[allow(clippy::too_many_arguments)]
pub fn candidates_from_tiles(
    tiles: &[Tile],
    scores: &[Vec<EntityScore>],
    table: &EntityTable,
    frame: &Frame,
    camera: &Camera,
    view: &View,
    cfg: &EmbeddingCfg,
) -> Vec<Candidate> {
    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let mut out: Vec<Candidate> = Vec::new();
    for (tile, sc) in tiles.iter().zip(scores) {
        if tile.level == 0 {
            continue;
        }
        let Some(best) = sc.first() else { continue };
        if best.score < cfg.report_floor {
            continue;
        }
        let Some((kind, link)) = table.get(&best.id).cloned() else { continue };
        let runner = sc.get(1);
        let bbox = [tile.x as f32 / fw, tile.y as f32 / fh, (tile.x + tile.w) as f32 / fw, (tile.y + tile.h) as f32 / fh];
        let depth = median_depth(frame, tile);
        let (cx, cy) = tile.center();
        let world = depth.and_then(|d| camera.world_point3_in(view, cx, cy, d)).map(|p| centre_from_surface(view, p, centre_offset_m(kind)));
        out.push(Candidate {
            label: best.id.clone(),
            kind,
            link,
            score: best.score,
            margin: best.score - runner.map_or(0.0, |r| r.score),
            runner_up: runner.map(|r| r.id.clone()),
            bbox,
            pixels: tile.area(),
            depth_m: depth.unwrap_or(f32::INFINITY),
            world,
            truth: None,
            truth_label: None,
        });
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Candidate> = Vec::new();
    for c in out {
        if kept.iter().all(|k| k.label != c.label || iou(&k.bbox, &c.bbox) < cfg.nms_iou) {
            kept.push(c);
        }
    }
    kept.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.label.cmp(&b.label)));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_search::semantic::cosine;
    use world_sim::kinds::NodeKind;
    use world_sim::world::Node;
    use world_sim::{World, WorldConfig};

    fn lexicon() -> Lexicon {
        Lexicon::from_json(include_str!("../../../data/world_lexicon.json")).expect("lexicon loads")
    }

    fn world_with_tree_ahead() -> (World, Camera) {
        let mut w = World::new(WorldConfig::default()).unwrap();
        w.nodes.clear();
        w.drops.clear();
        let (cx, cz) = w.terrain.cell_of(w.agent.x + 3.2, w.agent.z + 0.2).unwrap();
        let (x, z) = world_sim::terrain::cell_center(cx, cz);
        w.nodes.push(Node { id: 1, kind: NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 });
        w.agent.yaw = 0.0;
        (w, Camera::default())
    }

    fn score(id: &str, s: f32) -> EntityScore {
        EntityScore { id: id.into(), kind: noto_search::EntityKind::Source, link: "log".into(), bias: s, prejudice: 0.0, score: s }
    }

    fn tile_with_most_tree(tiles: &[Tile], f: &Frame, w: &World) -> usize {
        let target = w.visible_objects(f).into_iter().find(|v| v.label == "tree").expect("tree in view").bbox;
        let (cx, cy) = (((target[0] + target[2]) / 2) as f32, ((target[1] + target[3]) / 2) as f32);
        tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.level > 0 && cx >= t.x as f32 && cx < (t.x + t.w) as f32 && cy >= t.y as f32 && cy < (t.y + t.h) as f32)
            .min_by_key(|(_, t)| t.area())
            .map(|(i, _)| i)
            .expect("a tile holds the tree centre")
    }

    #[test]
    fn a_tile_that_names_the_tree_becomes_a_candidate_on_the_tree() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(2, 2), (4, 3)], 0.15);
        let table = EntityTable::from_lexicon(&lexicon());
        let hit = tile_with_most_tree(&tiles, &f, &w);
        let scores: Vec<Vec<EntityScore>> =
            (0..tiles.len()).map(|i| if i == hit { vec![score("tree", 0.82), score("boulder", 0.30)] } else { vec![score("boulder", 0.05)] }).collect();
        let cands = candidates_from_tiles(&tiles, &scores, &table, &f, &cam, &w.head_view(&cam).est, &EmbeddingCfg::default());
        assert_eq!(cands.len(), 1, "only the named tile reports: {cands:?}");
        let c = &cands[0];
        assert_eq!(c.label, "tree");
        assert_eq!(c.runner_up.as_deref(), Some("boulder"));
        assert!((c.margin - 0.52).abs() < 1e-5);
        let t = &tiles[hit];
        assert_eq!(c.bbox, [t.x as f32 / f.width as f32, t.y as f32 / f.height as f32, (t.x + t.w) as f32 / f.width as f32, (t.y + t.h) as f32 / f.height as f32]);
        let node = &w.nodes[0];
        let (x, z) = c.world.expect("depth known");
        let err = ((x - node.x).powi(2) + (z - node.z).powi(2)).sqrt();
        assert!(err < 1.5, "estimated {x:.2},{z:.2} vs tree {:.2},{:.2}", node.x, node.z);
        assert!(c.truth.is_none() && c.truth_label.is_none(), "a real recogniser knows no ground truth");
    }

    #[test]
    fn overlapping_tiles_of_one_label_merge_and_different_labels_stay() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(4, 3)], 0.5);
        let table = EntityTable::from_lexicon(&lexicon());
        let scores: Vec<Vec<EntityScore>> = (0..tiles.len())
            .map(|i| if i == 5 { vec![score("boulder", 0.7), score("tree", 0.1)] } else { vec![score("tree", 0.9 - 0.01 * i as f32), score("boulder", 0.1)] })
            .collect();
        let cands = candidates_from_tiles(&tiles, &scores, &table, &f, &cam, &w.head_view(&cam).est, &EmbeddingCfg::default());
        let trees: Vec<&Candidate> = cands.iter().filter(|c| c.label == "tree").collect();
        assert!(trees.len() < tiles.len() - 1, "NMS merged some of the {} tree tiles, left {}", tiles.len() - 2, trees.len());
        for (i, a) in trees.iter().enumerate() {
            for b in &trees[i + 1..] {
                assert!(iou(&a.bbox, &b.bbox) < 0.4);
            }
        }
        assert!(cands.iter().any(|c| c.label == "boulder"));
    }

    #[test]
    fn tiles_below_the_report_floor_say_nothing() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let tiles = noto_ml::parallel::tile_layout(f.width as u32, f.height as u32, &[(2, 2)], 0.0);
        let table = EntityTable::from_lexicon(&lexicon());
        let scores: Vec<Vec<EntityScore>> = tiles.iter().map(|_| vec![score("tree", 0.09)]).collect();
        let cands = candidates_from_tiles(&tiles, &scores, &table, &f, &cam, &w.head_view(&cam).est, &EmbeddingCfg::default());
        assert!(cands.is_empty());
    }

    struct WordAxes;

    const AXES: [&str; 4] = ["tree", "boulder", "log", "warehouse"];

    impl TextEmbedder for WordAxes {
        fn embedding_dim(&self) -> usize {
            AXES.len() + 1
        }
        fn fingerprint(&self) -> u64 {
            7
        }
        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v: Vec<f32> = AXES.iter().map(|a| if t.contains(a) { 1.0 } else { 0.0 }).collect();
                    v.push(0.05);
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter().map(|x| x / n).collect()
                })
                .collect())
        }
    }

    struct Greens;

    impl ImageEmbedder for Greens {
        fn embedding_dim(&self) -> usize {
            AXES.len() + 1
        }
        fn embed_images_batch(&self, images: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
            Ok(images
                .iter()
                .map(|im| {
                    let rgb = im.to_rgb8();
                    let n = (rgb.width() * rgb.height()) as f32;
                    let (mut green, mut grey) = (0.0f32, 0.0f32);
                    for p in rgb.pixels() {
                        let [r, g, b] = p.0;
                        if g > r.saturating_add(20) && g > b.saturating_add(20) {
                            green += 1.0;
                        } else if (r as i32 - g as i32).abs() < 12 && (g as i32 - b as i32).abs() < 12 {
                            grey += 1.0;
                        }
                    }
                    let mut v = vec![green / n, grey / n, 0.0, 0.0, 0.05];
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter_mut().for_each(|x| *x /= norm);
                    v
                })
                .collect())
        }
    }

    fn perception() -> EmbeddingPerception {
        let lex = lexicon();
        let phrases = BankedPhrases::new(Arc::new(WordAxes), None);
        let cfg = EmbeddingCfg { webp_quality: 100.0, report_floor: 0.2, ..EmbeddingCfg::default() };
        EmbeddingPerception::new(&lex, &phrases, Arc::new(Greens), Arc::new(|c| c), "cpu (stand-in)", cfg).unwrap()
    }

    #[test]
    fn a_whole_scan_goes_screenshot_tiles_scores_candidates() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let mut p = perception();
        assert_eq!(p.mode(), "embedding");
        let out = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(out.tiles, 1 + 4 + 12 + 24);
        assert_eq!(out.device, "cpu (stand-in)");
        assert!(!out.scene.is_empty(), "the whole-frame scores are reported");
        assert!(out.candidates.iter().all(|c| c.truth.is_none()));
        assert!(out.candidates.iter().any(|c| c.label == "tree"), "grass and leaves are green: {:?}", out.candidates.iter().map(|c| (&c.label, c.score)).collect::<Vec<_>>());
        for c in &out.candidates {
            assert!(c.bbox.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!(c.score >= 0.2);
        }
        let again = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(again.candidates, out.candidates);
    }

    #[test]
    fn a_calibration_maps_entity_scores_but_leaves_places_raw() {
        let (w, cam) = world_with_tree_ahead();
        let f = w.render(&cam);
        let mut p = perception();
        let raw = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        assert_eq!(Recognizer::scope(&p), "embedding");
        let mut t = noto_vision::CalibTable { mode: SCAN_CALIB_MODE.into(), statistic: "tile_score".into(), ..Default::default() };
        t.labels.insert("tree".into(), noto_vision::LabelCalib { threshold: 1.0, platt: Some((10.0, -10.0)), positives: 5, negatives: 5, precision: None, recall: None });
        p.set_calibration(Some(t));
        assert_eq!(Recognizer::scope(&p), "embedding+cal", "calibrated statistics stay apart");
        let cal = p.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
        let top = |o: &ScanOutput, id: &str| o.scene.iter().find(|(l, _)| l == id).map(|x| x.1).unwrap();
        let (r, c) = (top(&raw, "tree"), top(&cal, "tree"));
        assert!((c - 1.0 / (1.0 + (-(10.0 * r - 10.0)).exp())).abs() < 1e-5, "{r} -> {c}");
        assert_eq!(top(&raw, "warehouse"), top(&cal, "warehouse"));
        assert_eq!(top(&raw, "boulder"), top(&cal, "boulder"));
    }

    #[test]
    fn banked_phrases_embed_each_phrase_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counting(AtomicUsize);
        impl TextEmbedder for Counting {
            fn embedding_dim(&self) -> usize {
                3
            }
            fn fingerprint(&self) -> u64 {
                1
            }
            fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                self.0.fetch_add(texts.len(), Ordering::SeqCst);
                Ok(texts.iter().map(|t| vec![t.len() as f32, 1.0, 0.0]).map(|v| {
                    let n = (v[0] * v[0] + 1.0).sqrt();
                    vec![v[0] / n, 1.0 / n, 0.0]
                }).collect())
            }
        }
        let e = Arc::new(Counting(AtomicUsize::new(0)));
        let b = BankedPhrases::new(Arc::clone(&e), None);
        let texts: Vec<String> = ["tree", "tree", "  tree ", "rock"].iter().map(|s| s.to_string()).collect();
        let v = b.embed_phrases(&texts).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(e.0.load(Ordering::SeqCst), 2, "four requests, two distinct phrases");
        assert!((cosine(&v[0], &v[2]) - 1.0).abs() < 1e-6);
        b.embed_phrases(&texts).unwrap();
        assert_eq!(e.0.load(Ordering::SeqCst), 2, "the second call is served from the bank");
    }
}
