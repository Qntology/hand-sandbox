use std::path::PathBuf;
use std::sync::Arc;

use noto_core::{BaseKind, NotoConfig};
use noto_runtime::{EmbeddingCfg, EmbeddingPerception, NoiseCfg, Recognizer, Runtime, RuntimeConfig, ScanInput};
use world_sim::world::DropItem;
use world_sim::{Camera, ItemKind, TerrainKind};

const LEX: &str = include_str!("../../../data/world_lexicon.json");

#[test]
#[ignore]
fn siglip2_sees_a_parcel_at_these_distances() {
    let dir = PathBuf::from(std::env::var("SIGLIP2_DIR").expect("SIGLIP2_DIR"));
    let device = noto_ml::preferred().backend.device(0).expect("device");
    let model = Arc::new(noto_ml::Siglip2::load(&dir, &device).expect("SigLIP2"));
    let lex = noto_search::Lexicon::from_json(LEX).unwrap();
    let bank = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/runtime/bank");
    let cfg = RuntimeConfig { seed: 7, noise: NoiseCfg::none(), parcels: Some(0), terrain: TerrainKind::Open, noto: NotoConfig { base: BaseKind::HumanFoot, ..RuntimeConfig::default().noto }, ..RuntimeConfig::default() };
    let rt = Runtime::new(cfg, LEX, None).unwrap();
    let base = rt.world().clone();
    let grids: [(&str, Vec<(u32, u32)>); 2] = [("2x2+4x3", vec![(2, 2), (4, 3)]), ("2x2+4x3+6x4 (default)", EmbeddingCfg::default().grids)];
    for (name, g) in grids {
        let mut rec = EmbeddingPerception::siglip(Arc::clone(&model), &lex, Some(&bank), EmbeddingCfg { grids: g, ..EmbeddingCfg::default() }).unwrap();
        for d in [1.2f32, 2.0, 3.0, 4.0, 6.0, 8.0] {
            for tilt in [0.0f32, 30.0, 45.0] {
                let mut w = base.clone();
                let (x, z, yaw) = (w.agent.x, w.agent.z, w.agent.yaw);
                let (px, pz) = (x + d * yaw.cos(), z + d * yaw.sin());
                w.drops.push(DropItem::whole(9001, ItemKind::Parcel, px, pz));
                let cam = Camera { pitch_deg: Camera::default().pitch_deg - tilt, ..Camera::default() };
                let f = w.render(&cam);
                let visible = w.visible_objects(&f).iter().find(|o| o.entity == "d9001").map_or(0, |o| o.pixels);
                let out = rec.scan(ScanInput { world: &w, frame: &f, camera: &cam }).unwrap();
                let best = out.candidates.iter().filter(|c| c.label == "parcel").max_by(|a, b| a.score.total_cmp(&b.score));
                let line = match best {
                    Some(c) => format!("parcel {:.2} · {:.2} m off", c.score, c.world.map_or(f32::NAN, |(cx, cz)| (cx - px).hypot(cz - pz))),
                    None => format!("no parcel (best: {})", out.candidates.iter().max_by(|a, b| a.score.total_cmp(&b.score)).map_or("-".into(), |c| format!("{} {:.2}", c.label, c.score))),
                };
                println!("[SIGLIP] {name} d {d:.1} m tilt {tilt:>2.0}° · {visible:>5} px · {} tiles · {line}", out.tiles);
            }
        }
    }
}
