use std::path::PathBuf;

use noto_core::BodyPlan;
use noto_runtime::{Runtime, RuntimeConfig};
use world_sim::Camera;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or_else(|| ".".into()));
    let preset = args.next().unwrap_or_else(|| "mid".into());
    let command = args.next().unwrap_or_else(|| "원목 1개를 창고로 가져와".into());
    std::fs::create_dir_all(&out)?;
    let lexicon = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/world_lexicon.json"))?;
    let mut rt = Runtime::new(RuntimeConfig::default(), &lexicon, None)?;
    rt.cfg.body = BodyPlan::named(&preset).ok_or_else(|| anyhow::anyhow!("unknown preset {preset}"))?;
    rt.api_world_new(7, 64, None, std::env::var("BASE").ok().as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    rt.api_command_submit(&command).map_err(|e| anyhow::anyhow!(e))?;
    let cam = Camera { width: 640, height: 427, fov_deg: 64.0, max_dist_m: 30.0, ..Camera::default() };
    for shot in 0..6 {
        rt.api_sim_step(if shot == 0 { 1 } else { 40 }).map_err(|e| anyhow::anyhow!(e))?;
        let (f, o) = noto_runtime::chase::render(rt.world(), &cam, -35.0);
        let mut img = image::RgbImage::from_raw(f.width as u32, f.height as u32, f.rgb).expect("frame size");
        if let Some(o) = &o {
            for j in o["joints"].as_array().into_iter().flatten() {
                let (Some(x), Some(y)) = (j["at"]["x"].as_f64(), j["at"]["y"].as_f64()) else { continue };
                let r = j["ratio"].as_f64().unwrap_or(0.0);
                let c = if r >= 1.0 { [230, 60, 60] } else if r >= 0.5 { [230, 190, 60] } else { [80, 200, 120] };
                for dy in -3i64..=3 {
                    for dx in -3i64..=3 {
                        let (px, py) = (x as i64 + dx, y as i64 + dy);
                        if dx * dx + dy * dy <= 9 && px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                            img.put_pixel(px as u32, py as u32, image::Rgb(c));
                        }
                    }
                }
            }
            println!("shot {shot}: {} · margin {} · {}", o["source"], o["margin"], o["motion"]["notes"]);
        }
        img.save(out.join(format!("chase_{preset}_{shot}.png")))?;
    }
    Ok(())
}
