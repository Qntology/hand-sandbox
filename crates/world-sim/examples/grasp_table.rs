use noto_core::{BaseKind, FingerCount, NotoConfig};
use world_sim::{World, WorldConfig};

fn main() {
    for f in FingerCount::ALL {
        let mut noto = NotoConfig::preset(f);
        noto.base = BaseKind::Track;
        let w = World::new(WorldConfig { seed: 7, cells: 64, noto, ..WorldConfig::default() }).unwrap();
        println!("fingers {:?}", f);
        for g in w.grasp_table() {
            println!("  {:9} stable={:5} margin={:6.2} max_payload={:5.2} kg type={}", g.item.id(), g.stable, g.margin, g.max_payload_kg, g.grasp_type);
        }
    }
}
