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
        for (name, mut s) in shots {
            let f = s.render(&cam);
            let hv = s.head_view(&cam);
            let path = dir.join(format!("{base:?}-{name}.png").to_lowercase());
            image::save_buffer(&path, &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).unwrap();
            println!("[PICK] {base:?} {name}: eye {:.2} m · pitch {:.1}° · neck {:.1}° · {}", hv.eye_m, hv.view.pitch.to_degrees(), s.current_kin_pose().neck_deg, path.display());
            if base != BaseKind::HumanFoot {
                s.fork_cam = true;
                let fc = Camera { width: 320, height: 240, ..cam };
                let f = s.render(&fc);
                let path = dir.join(format!("{base:?}-{name}-forkcam.png").to_lowercase());
                image::save_buffer(&path, &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).unwrap();
            }
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
        // Legs bend the head down to it; on wheels the rider keeps looking ahead and the fork carriage camera sees it.
        w.fork_cam = base != BaseKind::HumanFoot;
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
        assert!(n > 50, "{base:?}: the log is in view while the head (or the fork camera) is down ({n} px)");
        let (cx, cz) = (sx / n as f32, sz / n as f32);
        let err = (cx - dx).hypot(cz - dz);
        println!("[RECON] {base:?} pitch {:.1}° eye {:.2} m · log px {n} · centroid off {:.3} m", hv.view.pitch.to_degrees(), hv.eye_m, err);
        // The camera sees the top and the near face, not the centre: what it sees lies within the log's half-diagonal.
        let size = ItemKind::Log.spec().size_m;
        assert!(err < 0.5 * size[0].hypot(size[2]), "{base:?}: back-projected log is {err:.3} m from the log");
    }
}

/// Where the item's centre falls in the head camera's image, as fractions of the image (0.5, 0.5 is the middle).
fn in_head_image(w: &World, cam: &Camera, at: [f32; 3]) -> Option<(f32, f32)> {
    let hv = w.head_view(cam);
    cam.project_in(&hv.view, at).map(|(x, y, _)| (x / cam.width as f32, y / cam.height as f32))
}

#[test]
fn the_head_camera_follows_the_log_up_while_the_hands_lift_it() {
    let mut w = world(BaseKind::HumanFoot);
    let (dx, dz) = (w.drops[0].x, w.drops[0].z);
    w.command(Action::Look { x: dx, z: dz }).unwrap();
    while w.busy() {
        w.tick();
    }
    w.command(Action::pick(77)).unwrap();
    let cam = Camera::default();
    let (mut frames, mut worst, mut rise) = (0, 0.0f32, (f32::INFINITY, f32::NEG_INFINITY));
    while w.busy() {
        w.tick();
        let Some((_, _, c)) = w.hand_load() else { continue };
        assert_eq!(w.lifted_drop(), Some(77), "the log leaves the ground while it is in the hands");
        let at = w.body_to_world(c);
        rise = (rise.0.min(at[1]), rise.1.max(at[1]));
        let neck = w.current_kin_pose().neck_deg;
        let Some((u, v)) = in_head_image(&w, &cam, at) else { panic!("the log went behind the head camera") };
        if neck.abs() < noto_core::kin::NECK_PITCH_MAX_DEG - 0.5 {
            worst = worst.max((v - 0.5).abs()).max((u - 0.5).abs());
            frames += 1;
        }
    }
    println!("[GAZE] {frames} frames in the hands · log rose {:.2} → {:.2} m · worst {:.3} of the image off the middle", rise.0, rise.1, worst);
    assert!(frames > 5 && rise.1 > rise.0 + 0.2, "{frames} frames, {rise:?}");
    assert!(worst < 0.08, "the log stays in the middle of the head camera's image as it comes up: {worst:.3}");
    assert!(w.current_kin_pose().neck_deg.abs() < 0.5, "and the head is level again afterwards");
}

#[test]
fn a_lift_truck_rider_looks_down_when_the_log_blocks_the_fork_camera() {
    use world_sim::render::{pack_id, unpack_id};
    let mut w = world(BaseKind::Wheel);
    let (dx, dz) = (w.drops[0].x, w.drops[0].z);
    w.command(Action::Look { x: dx, z: dz }).unwrap();
    let cam = Camera { width: 320, height: 240, ..Camera::default() };
    let mut rider_look = false;
    while w.busy() {
        w.tick();
        if w.act_phase().map(|p| p.1) == Some("rider_check") {
            rider_look = true;
            assert!(w.rider_sees_forks(), "looking down, the rider sees the tines");
            let (u, v) = in_head_image(&w, &cam, [dx, w.surface_m(dx, dz) + 0.075, dz]).expect("in front");
            assert!((0.2..0.8).contains(&u) && (0.2..0.8).contains(&v), "the rider's look puts the log near the middle: {u:.2} {v:.2}");
        }
    }
    assert!(rider_look, "the fork look starts with the rider's glance");
    w.command(Action::pick(77)).unwrap();
    let (mut blocked, mut rechecked) = (0.0f32, false);
    while w.busy() {
        w.tick();
        match w.act_phase().map(|p| p.1) {
            Some("recheck") if !rechecked => {
                rechecked = true;
                w.fork_cam = true;
                let f = w.render(&cam);
                w.fork_cam = false;
                let want = pack_id(world_sim::ObjectClass::Drop, 77);
                blocked = f.ids.iter().filter(|&&i| i == want).count() as f32 / f.ids.len() as f32;
                let head = w.render(&cam);
                let seen = head.ids.iter().filter(|&&i| unpack_id(i) == (world_sim::ObjectClass::Drop, 77)).count();
                println!("[FORK] recheck: fork camera {:.0}% log · rider sees {seen} log px · neck {:.0}°", blocked * 100.0, w.current_kin_pose().neck_deg);
                assert!(seen > 100, "the rider's head camera sees the log on the tine tips ({seen} px)");
                assert!(w.current_kin_pose().neck_deg > 25.0);
            }
            _ => {}
        }
    }
    assert!(rechecked, "the pick stops for the rider's re-check");
    assert!(blocked >= 0.45, "the log fills the fork camera's view when it stops: {:.0}%", blocked * 100.0);
    assert!(w.carried_count(ItemKind::Log) == 1);
}

#[test]
fn each_item_is_drawn_where_it_lies_even_beside_others_across_cell_edges() {
    use world_sim::render::unpack_id;
    let mut w = world(BaseKind::Wheel);
    w.drops.clear();
    let (x0, z0) = (w.agent.x + 1.0, w.agent.z + 0.02);
    let at = [(7u32, 0.0f32, 0.0f32), (8, -0.26, 0.01), (9, -0.10, -0.15)];
    for (id, dx, dz) in at {
        w.drops.push(DropItem::whole(id, ItemKind::Log, x0 + dx, z0 + dz));
    }
    for (pitch, eye_label) in [(-40.0f32, "head"), (-12.0, "level")] {
        let cam = Camera { width: 320, height: 240, pitch_deg: pitch, ..Camera::default() };
        let f = w.render(&cam);
        let hv = w.head_view(&cam);
        for (id, dx, dz) in at {
            let (mut sx, mut sz, mut n) = (0.0f32, 0.0f32, 0u32);
            for (i, &raw) in f.ids.iter().enumerate() {
                if unpack_id(raw) != (world_sim::ObjectClass::Drop, id) {
                    continue;
                }
                if let Some(p) = cam.world_point3_in(&hv.est, (i % f.width) as f32 + 0.5, (i / f.width) as f32 + 0.5, f.depth[i]) {
                    sx += p[0];
                    sz += p[2];
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            let (cx, cz) = (sx / n as f32, sz / n as f32);
            let off = (cx - (x0 + dx)).hypot(cz - (z0 + dz));
            println!("[DRAW] {eye_label} d{id}: {n} px · centroid off {off:.3} m");
            assert!(off < 0.1, "d{id} is drawn {off:.3} m from where it lies ({eye_label})");
        }
    }
}

#[test]
fn the_odometry_counts_a_fork_pick_once() {
    let mut w = world(BaseKind::Wheel);
    let (dx, dz) = (w.drops[0].x, w.drops[0].z);
    let (t0, b0, o0) = ((w.agent.x, w.agent.z), w.believed_pose(), w.odom.travelled_m);
    w.command(Action::Look { x: dx, z: dz }).unwrap();
    while w.busy() {
        w.tick();
    }
    w.command(Action::pick(77)).unwrap();
    while w.busy() {
        w.tick();
    }
    let moved = (w.agent.x - t0.0).hypot(w.agent.z - t0.1);
    let believed = (w.believed_pose().0 - b0.0).hypot(w.believed_pose().1 - b0.1);
    let odo = w.odom.travelled_m - o0;
    println!("[ODOM] fork look + pick: moved {moved:.3} m · believed {believed:.3} m · odometry {odo:.3} m");
    assert!(moved > 0.05, "the base drove in under the log");
    assert!((believed - moved).abs() < 0.02 + 0.05 * moved && odo < 2.0 * moved + 0.05, "the belief follows the base, once: {believed:.3} vs {moved:.3} (odometry {odo:.3})");
}
