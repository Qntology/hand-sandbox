//! Writes a first-person WebP screenshot, a top-down map and prints timings.
//! Usage: cargo run --release -p world-sim --example capture_demo -- <out_dir> [seed]
use std::time::Instant;

use hand_core::{BaseKind, FingerCount, HandConfig};
use world_sim::{encode_webp, Camera, World, WorldConfig};

fn main() {
    let out = std::env::args().nth(1).expect("output directory");
    let seed: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let mut hand = HandConfig::preset(FingerCount::Five);
    hand.base = BaseKind::Track;
    let mut w = World::new(WorldConfig { seed, cells: 64, hand, ..WorldConfig::default() }).unwrap();
    // Look at the nearest node.
    let n = w
        .nodes
        .iter()
        .min_by(|a, b| (a.x - w.agent.x).hypot(a.z - w.agent.z).total_cmp(&(b.x - w.agent.x).hypot(b.z - w.agent.z)))
        .unwrap()
        .clone();
    println!("nearest node {:?} at ({:.1},{:.1}); agent ({:.1},{:.1})", n.kind, n.x, n.z, w.agent.x, w.agent.z);
    w.agent.yaw = (n.z - w.agent.z).atan2(n.x - w.agent.x);

    let cam = Camera::default();
    let t0 = Instant::now();
    let f = w.render(&cam);
    let render_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let enc = encode_webp(&f.rgb, f.width as u32, f.height as u32, 75.0).unwrap();
    let lossless = encode_webp(&f.rgb, f.width as u32, f.height as u32, 100.0).unwrap();
    println!(
        "render {render_ms:.1} ms, webp q75 {} bytes ({:.1}x, {:.1} ms), lossless {} bytes",
        enc.bytes.len(),
        enc.ratio(),
        enc.encode_ms,
        lossless.bytes.len()
    );
    for o in w.visible_objects(&f) {
        println!("  {} {} px={} depth={:.2} bbox={:?}", o.entity, o.label, o.pixels, o.depth_m, o.bbox);
    }
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(format!("{out}/frame.webp"), &enc.bytes).unwrap();
    image::save_buffer(format!("{out}/frame.png"), &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).unwrap();
    let top = w.render_top(384, 32.0);
    image::save_buffer(format!("{out}/top.png"), &top.rgb, top.width as u32, top.height as u32, image::ColorType::Rgb8).unwrap();
}
