// First-person frames of one pick-up, a stage at a time: walk up, bend the head, hold to check, lower the body, lift.
//   set PICKUP_FRAMES_DIR=...   (PNG files and a line per frame)
//   cargo test -p world-sim --test pickup_frames -- --ignored --nocapture
use noto_core::{BaseKind, FingerCount, NotoConfig};
use world_sim::world::{Action, DropItem, WorldConfig};
use world_sim::{Camera, ItemKind, World};

fn world(base: BaseKind) -> World {
    let mut noto = NotoConfig::preset(FingerCount::Five);
    noto.base = base;
    let mut w = World::new(WorldConfig { seed: 7, parcels: Some(0), noto, ..WorldConfig::default() }).unwrap();
    w.nodes.clear();
    w.drops.clear();
    w.agent.yaw = 0.0;
    let (x, z) = (w.agent.x + 0.6, w.agent.z);
    w.drops.push(DropItem::whole(77, ItemKind::Log, x, z));
    w
}

#[test]
#[ignore]
fn save_the_pickup_from_the_head_camera() {
    let dir = std::path::PathBuf::from(std::env::var("PICKUP_FRAMES_DIR").unwrap_or_else(|_| ".".into()));
    std::fs::create_dir_all(&dir).unwrap();
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        let mut w = world(base);
        let cam = Camera { width: 320, height: 213, ..Camera::default() };
        let (dx, dz) = (w.drops[0].x, w.drops[0].z);
        w.command(Action::Look { x: dx, z: dz }).unwrap();
        let mut shots: Vec<(String, World)> = vec![("0-stand".into(), w.clone())];
        while w.busy() {
            w.tick();
        }
        shots.push(("1-look".into(), w.clone()));
        if let Err(e) = w.command(Action::pick(77)) {
            println!("[PICK] {base:?}: no pick from where it stands ({e:?})");
        }
        let mut last = String::new();
        let mut k = 2;
        while w.busy() {
            w.tick();
            let phase = w.act_phase().map(|p| p.1).unwrap_or("-");
            if phase != last {
                shots.push((format!("{k}-{phase}"), w.clone()));
                last = phase.to_string();
                k += 1;
            }
        }
        shots.push((format!("{k}-done"), w.clone()));
        for (name, s) in shots {
            let f = s.render(&cam);
            let hv = s.head_view(&cam);
            let path = dir.join(format!("{base:?}-{name}.png").to_lowercase());
            image::save_buffer(&path, &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).unwrap();
            println!("[PICK] {base:?} {name}: eye {:.2} m · pitch {:.1}° · neck {:.1}° · {}", hv.eye_m, hv.view.pitch.to_degrees(), s.current_kin_pose().neck_deg, path.display());
        }
    }
}

#[test]
fn a_looked_down_pixel_lands_back_on_the_log() {
    for base in [BaseKind::HumanFoot, BaseKind::Wheel] {
        let mut w = world(base);
        let (dx, dz) = (w.drops[0].x, w.drops[0].z);
        w.command(Action::Look { x: dx, z: dz }).unwrap();
        while w.busy() {
            w.tick();
        }
        let cam = Camera::default();
        let f = w.render(&cam);
        let hv = w.head_view(&cam);
        let (mut sx, mut sz, mut n) = (0.0f32, 0.0f32, 0u32);
        for (i, &raw) in f.ids.iter().enumerate() {
            let (class, id) = world_sim::render::unpack_id(raw);
            if class != world_sim::ObjectClass::Drop || id != 77 {
                continue;
            }
            let (px, py) = ((i % f.width) as f32 + 0.5, (i / f.width) as f32 + 0.5);
            if let Some(p) = cam.world_point3_in(&hv.est, px, py, f.depth[i]) {
                sx += p[0];
                sz += p[2];
                n += 1;
            }
        }
        assert!(n > 50, "{base:?}: the log is in view while the head is down ({n} px)");
        let (cx, cz) = (sx / n as f32, sz / n as f32);
        let err = (cx - dx).hypot(cz - dz);
        println!("[RECON] {base:?} pitch {:.1}° eye {:.2} m · log px {n} · centroid off {:.3} m", hv.view.pitch.to_degrees(), hv.eye_m, err);
        // The camera sees the top and the near face, not the centre: what it sees lies within the log's half-diagonal.
        let size = ItemKind::Log.spec().size_m;
        assert!(err < 0.5 * size[0].hypot(size[2]), "{base:?}: back-projected log is {err:.3} m from the log");
    }
}
