//! A side elevation of every robot kind and the world's objects at one scale, for checking proportions: each robot
//! stands on the same ground beside a door, a wall, a shelf and a metre rule.
//!   cargo run -p world-sim --example model_sheet -- <output dir>
use noto_core::dims::Build;
use noto_core::{BaseKind, BodyPlan, FingerCount, NotoConfig};
use world_sim::crew::Form;
use world_sim::render::{render_rays, Camera, Obb, Rays};
use world_sim::world::{DropItem, Node, WorldConfig};
use world_sim::{ItemKind, NodeKind, TerrainKind, World};

const EYE3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn slab(x: f32, z: f32, w: f32, d: f32, y0: f32, y1: f32, color: [u8; 3]) -> Obb {
    Obb { center: [x, 0.5 * (y0 + y1), z], axes: EYE3, half: [0.5 * w, 0.5 * (y1 - y0), 0.5 * d], color, ..Obb::default() }
}

fn world(base: BaseKind, build: Build) -> World {
    let mut noto = NotoConfig::preset(FingerCount::Five);
    noto.base = base;
    let body = BodyPlan { build, ..BodyPlan::default() };
    let mut w = World::new(WorldConfig { seed: 3, parcels: Some(0), terrain: TerrainKind::Open, noto, body, ..WorldConfig::default() }).expect("world");
    w.nodes.clear();
    w.drops.clear();
    w
}

/// Frames the row along x starting at the agent, with the reference objects behind it.
fn sheet(mut w: World, out: &str, name: &str, spacing: f32, crew: &[Form]) {
    let (x0, z0) = (w.agent.x, w.agent.z);
    w.agent.yaw = 0.0;
    let ids = w.add_crew(crew, 1);
    for (i, id) in ids.iter().enumerate() {
        let c = w.crew.iter_mut().find(|c| c.id == *id).expect("worker");
        c.x = x0 + spacing * (i + 1) as f32;
        c.z = z0;
        c.yaw = 0.0;
    }
    let right = x0 + spacing * (crew.len() + 1) as f32;
    w.nodes.push(Node { id: 9001, kind: NodeKind::Tree, x: right + 0.5, z: z0, hp: 10.0, max_hp: 10.0 });
    w.nodes.push(Node { id: 9002, kind: NodeKind::Boulder, x: right + 1.8, z: z0, hp: 10.0, max_hp: 10.0 });
    w.drops.push(DropItem::whole(9003, ItemKind::Log, right - 0.6, z0));
    w.drops.push(DropItem::whole(9004, ItemKind::Parcel, right + 2.8, z0));
    let pose = w.body_pose().expect("pose");
    let mut boxes = w.robot_boxes(&pose);
    let back = z0 + 1.2;
    // Metre rule every 0.5 m, the house door and wall, the warehouse wall, a shelf.
    for k in 0..8 {
        let c = if k % 2 == 0 { [230, 230, 235] } else { [40, 40, 44] };
        boxes.push(slab(x0 - 1.0, back, 0.06, 0.06, 0.5 * k as f32, 0.5 * (k + 1) as f32, c));
    }
    boxes.push(slab(x0 - 1.8, back, 0.914, 0.05, 0.0, 2.032, [150, 112, 80]));
    boxes.push(slab(x0 - 1.8, back + 0.1, 2.0, 0.05, 0.0, 2.4, [176, 176, 170]));
    boxes.push(slab(right + 4.0, back, 1.829, 0.1, 0.0, 1.829, [92, 112, 150]));
    boxes.push(slab(right + 4.0, back + 0.1, 2.5, 0.05, 0.0, 3.0, [120, 120, 126]));
    let span = right + 6.0 - (x0 - 3.2);
    let (wpx, hpx) = (1500usize, 560usize);
    let half_h = 0.5 * span * hpx as f32 / wpx as f32;
    let cam = Camera { width: wpx, height: hpx, max_dist_m: 60.0, ..Camera::default() };
    let rays = Rays::Ortho { centre: [x0 - 3.2 + 0.5 * span, half_h - 0.25, z0], yaw: std::f32::consts::FRAC_PI_2, pitch: 0.0, half_h_m: half_h };
    let f = render_rays(&w, &cam, &rays, &boxes, &[], None);
    let path = format!("{out}/{name}.png");
    image::save_buffer(&path, &f.rgb, f.width as u32, f.height as u32, image::ColorType::Rgb8).expect("png");
    println!("{path}: {:.1} px/m · 눈 {:.2} m · 차체 길이 {:.2} × 폭 {:.2} m", 0.5 * hpx as f32 / half_h, w.standing_eye_m(), w.body.support_len_m, w.body.support_width_m);
}

fn main() {
    let out = std::env::args().nth(1).expect("output directory");
    std::fs::create_dir_all(&out).expect("dir");
    for (build, tag) in [(Build::Standard, "standard"), (Build::Compact, "compact")] {
        sheet(world(BaseKind::HumanFoot, build), &out, &format!("{tag}_humanoid"), 2.2, &[Form::Humanoid, Form::Forklift]);
        sheet(world(BaseKind::Wheel, build), &out, &format!("{tag}_forklift"), 3.0, &[Form::Humanoid, Form::Forklift]);
    }
    sheet(world(BaseKind::Track, Build::Standard), &out, "standard_track", 3.0, &[]);
    sheet(world(BaseKind::Mecanum, Build::Standard), &out, "standard_mecanum", 3.0, &[]);
}
