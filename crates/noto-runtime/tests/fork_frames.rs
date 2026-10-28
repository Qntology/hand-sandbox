//! Frames of the pick-up at each stage, for looking at: a wheeled base doing it as a lift truck (rider looking ahead,
//! tines down, carriage camera, slide in, lift, set down) and a walker doing it by hand (head down, bend, lift).
//! Run with `cargo test -p noto-runtime --test fork_frames -- --ignored --nocapture`; the PNGs go to
//! `$FORK_FRAMES_DIR` (default `target/fork_frames`).

use noto_core::{BaseKind, NotoConfig};
use noto_runtime::{FrameView, NoiseCfg, Runtime, RuntimeConfig};

const LEXICON: &str = include_str!("../../../data/world_lexicon.json");

fn save(dir: &std::path::Path, name: &str, rt: &mut Runtime, view: FrameView, w: u32, h: u32) {
    let f = rt.frame_sized(view, w, h, 90.0).expect("a frame");
    let (fw, fh, rgb) = world_sim::decode_webp(&f.webp).expect("webp");
    image::save_buffer(dir.join(format!("{name}.png")), &rgb, fw, fh, image::ExtendedColorType::Rgb8).expect("png");
}

fn run(base: BaseKind, text: &str, tag: &str) -> Vec<String> {
    let dir = std::path::PathBuf::from(std::env::var("FORK_FRAMES_DIR").unwrap_or_else(|_| "target/fork_frames".into()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = RuntimeConfig { seed: 7, noise: NoiseCfg::none(), noto: NotoConfig { base, ..RuntimeConfig::default().noto }, ..RuntimeConfig::default() };
    let mut rt = Runtime::new(cfg, LEXICON, None).expect("runtime");
    assert!(rt.submit(text).ok);
    let mut seen: Vec<String> = Vec::new();
    let mut n = 0;
    while n < 30_000 && rt.busy() {
        rt.tick();
        n += 1;
        let carrying = rt.world().carried_kg() > 0.0;
        let Some(p) = rt.procedure_json() else {
            if carrying && !seen.iter().any(|s| s == "carry") {
                let i = seen.len();
                println!("[{tag}] tick {n} carry");
                save(&dir, &format!("{tag}_{i:02}_carry_agent"), &mut rt, FrameView::Agent, 320, 240);
                save(&dir, &format!("{tag}_{i:02}_carry_chase"), &mut rt, FrameView::Chase { orbit_deg: 110 }, 480, 360);
                save(&dir, &format!("{tag}_{i:02}_carry_low"), &mut rt, FrameView::Focus(world_sim::render::FocusCam::Low), 480, 360);
                seen.push("carry".into());
            }
            continue;
        };
        let stage = format!("{}-{}", p["act"].as_str().unwrap_or(""), p["stage"].as_str().unwrap_or(""));
        if seen.contains(&stage) {
            continue;
        }
        let i = seen.len();
        println!("[{tag}] tick {n} {stage} · 포크 {} m · 목 {}° · 카메라 {}° · 확인 {}", p["fork_m"], p["neck_deg"], p["pitch_deg"], p["check"]);
        save(&dir, &format!("{tag}_{i:02}_{stage}_agent"), &mut rt, FrameView::Agent, 320, 240);
        save(&dir, &format!("{tag}_{i:02}_{stage}_chase"), &mut rt, FrameView::Chase { orbit_deg: 110 }, 480, 360);
        save(&dir, &format!("{tag}_{i:02}_{stage}_low"), &mut rt, FrameView::Focus(world_sim::render::FocusCam::Low), 480, 360);
        if p["mechanism"] == "fork" {
            save(&dir, &format!("{tag}_{i:02}_{stage}_forkcam"), &mut rt, FrameView::Fork, 320, 240);
        }
        seen.push(stage);
    }
    seen
}

#[test]
#[ignore]
fn fork_and_hand_pick_up_frames() {
    let fork = run(BaseKind::Wheel, "원목 1개를 캐서 창고로 가져와", "wheel");
    println!("wheel stages: {fork:?}");
    let hand = run(BaseKind::HumanFoot, "원목 1개를 캐서 창고로 가져와", "legs");
    println!("legs stages: {hand:?}");
}
