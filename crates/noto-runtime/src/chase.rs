use noto_core::dynamics::BodyPose;
use serde_json::{json, Value};
use world_sim::iso::RobotExtent;
use world_sim::render::{render_chase, ChaseView};
use world_sim::{Camera, Frame, World};

pub const CHASE_DIST_M: f32 = 2.6;
pub const CHASE_HEIGHT_M: f32 = 1.6;
pub const CHASE_FILL: f32 = 0.6;
pub const CHASE_ELEVATION_DEG: f32 = 18.1;
pub const CHASE_TARGET_SHARE: f32 = 0.6;
const VISIBLE_SLACK_M: f32 = 0.08;
const GRF_M_PER_N: f32 = 0.004;

fn r3(x: f32) -> f64 {
    (x as f64 * 1000.0).round() / 1000.0
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChaseFraming {
    pub dist_m: f32,
    pub height_m: f32,
    pub target_m: f32,
}

impl ChaseFraming {
    pub fn fit(extent: &RobotExtent, cam: &Camera) -> ChaseFraming {
        let range = extent.radius_m / (CHASE_FILL * 0.5 * cam.vfov_deg()).to_radians().tan();
        let (s, c) = CHASE_ELEVATION_DEG.to_radians().sin_cos();
        let target = CHASE_TARGET_SHARE * extent.height_m;
        ChaseFraming { dist_m: range * c, height_m: target + range * s, target_m: target }
    }
}

pub fn render(world: &World, cam: &Camera, orbit_deg: f32) -> (Frame, Option<Value>) {
    let Some(pose) = world.body_pose() else {
        let view = ChaseView::around(world, orbit_deg, CHASE_DIST_M, CHASE_HEIGHT_M);
        return (render_chase(world, cam, &view, &[]), None);
    };
    let tilt = world.base_tilt(&pose);
    let boxes = world.tilted_robot_boxes(&pose, &tilt);
    let fr = ChaseFraming::fit(&RobotExtent::of(world, &boxes), cam);
    let view = ChaseView::around_at(world, orbit_deg, fr.dist_m, fr.height_m, fr.target_m);
    let frame = render_chase(world, cam, &view, &boxes);
    let camera = json!({
        "kind": "chase", "origin": view.origin.map(r3), "yaw_deg": r3(view.yaw.to_degrees()), "pitch_deg": r3(view.pitch.to_degrees()), "orbit_deg": r3(orbit_deg),
        "dist_m": r3(fr.dist_m), "height_m": r3(fr.height_m), "target_m": r3(fr.target_m),
    });
    let overlay = overlay(world, &pose, &frame, &|p| world.tilted_body_to_world(p, &tilt), &|p| view.project(cam, p), camera);
    (frame, Some(overlay))
}

pub fn overlay(
    world: &World,
    pose: &BodyPose,
    frame: &Frame,
    to_world: &dyn Fn([f32; 3]) -> [f32; 3],
    project: &dyn Fn([f32; 3]) -> Option<(f32, f32, f32)>,
    camera: Value,
) -> Value {
    let px = |p: [f32; 3]| -> Value {
        match project(to_world(p)) {
            Some((x, y, depth)) => {
                let (xi, yi) = (x.floor() as i64, y.floor() as i64);
                let inside = xi >= 0 && yi >= 0 && (xi as usize) < frame.width && (yi as usize) < frame.height;
                let visible = inside && frame.depth[yi as usize * frame.width + xi as usize] + VISIBLE_SLACK_M >= depth;
                json!({"x": r3(x), "y": r3(y), "depth": r3(depth), "visible": visible})
            }
            None => Value::Null,
        }
    };
    let joints: Vec<Value> = pose
        .joints
        .iter()
        .map(|j| {
            json!({
                "name": j.name, "kind": j.kind, "at": px(j.pos), "q_deg": j.q_deg.map(r3),
                "tau_nm": r3(j.tau_nm), "peak_nm": r3(j.peak_nm), "ratio": r3(j.ratio),
                "saturated": j.saturated, "stalled": j.stalled,
            })
        })
        .collect();
    let s = pose.support;
    let support: Vec<Value> = [[s[0], s[2]], [s[1], s[2]], [s[1], s[3]], [s[0], s[3]]].iter().map(|c| px([c[0], c[1], 0.0])).collect();
    let zmp = [pose.zmp[0], pose.zmp[1], 0.0];
    let grf_tip = [zmp[0] + pose.grf[0] * GRF_M_PER_N, zmp[1] + pose.grf[1] * GRF_M_PER_N, pose.grf[2] * GRF_M_PER_N];
    let motion = world.motion.as_ref().map(|m| {
        let r = &m.rollout;
        let stride = (r.samples.len() / 120).max(1);
        let picked: Vec<_> = r.samples.iter().step_by(stride).collect();
        let series: Vec<Value> = r
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.kind != "structure")
            .map(|(k, j)| json!({"name": j.name, "kind": j.kind, "ratio": picked.iter().map(|s| r3(s.ratio[k])).collect::<Vec<_>>()}))
            .collect();
        json!({
            "state": m.state, "source": r.source.id(), "source_ko": r.source.label_ko(),
            "t": r3((world.time_s - m.start_s) as f32), "duration_s": r3(r.duration_s),
            "feed_forward": r.feed_forward, "anticipated": r3(r.anticipated),
            "min_margin": r3(r.min_margin), "min_margin_t": r3(r.min_margin_t), "final_margin": r3(r.final_margin), "static_margin": r3(r.static_margin),
            "peak_ratio": r3(r.peak_ratio), "worst_joint": r.worst_joint, "reached": r.reached, "settle_s": r.settle_s.map(r3), "feasible": r.feasible,
            "notes": r.notes,
            "summary": r.summary.iter().map(|j| json!({"name": j.name, "kind": j.kind, "peak_ratio": r3(j.peak_ratio), "rms_effort": r3(j.rms_effort), "saturated_s": r3(j.saturated_s), "stalled_s": r3(j.stalled_s), "max_lag_deg": r3(j.max_lag_deg), "final_err_deg": r3(j.final_err_deg)})).collect::<Vec<_>>(),
            "series": {"t": picked.iter().map(|s| r3(s.t)).collect::<Vec<_>>(), "margin": picked.iter().map(|s| r3(s.margin)).collect::<Vec<_>>(), "joints": series},
        })
    });
    json!({
        "camera": camera,
        "t": r3(pose.t), "source": pose.source.id(),
        "joints": joints,
        "com": px(pose.com), "com_ground": px([pose.com[0], pose.com[1], 0.0]),
        "zmp": px(zmp), "grf_tip": px(grf_tip), "grf_n": pose.grf.map(r3),
        "support": support, "margin": r3(pose.margin),
        "motion": motion,
        "plan": world.cfg.body.id(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use world_sim::WorldConfig;

    #[test]
    fn the_chase_view_shows_the_robot_and_its_joints() {
        let w = World::new(WorldConfig { cells: 40, ..WorldConfig::default() }).unwrap();
        let cam = Camera { width: 192, height: 128, fov_deg: 64.0, max_dist_m: 30.0, ..Camera::default() };
        let (f, o) = render(&w, &cam, 0.0);
        let o = o.expect("an overlay");
        let robot_px = (0..f.width * f.height).filter(|&i| f.ids[i] == 0 && f.depth[i] < CHASE_DIST_M + 1.0).count();
        assert!(robot_px > 200, "{robot_px} robot pixels");
        let joints = o["joints"].as_array().unwrap();
        assert!(joints.iter().any(|j| j["name"] == "shoulder_r" && j["at"]["visible"].is_boolean()));
        let vis = joints.iter().filter(|j| j["at"]["visible"] == true).count();
        assert!(vis >= 3, "{vis} joints visible");
        assert!(o["zmp"].is_object() && o["support"].as_array().unwrap().len() == 4);
    }

    #[test]
    fn the_chase_framing_follows_the_robots_size() {
        let w = World::new(WorldConfig { cells: 40, ..WorldConfig::default() }).unwrap();
        let cam = Camera { width: 480, height: 320, fov_deg: 64.0, max_dist_m: 30.0, ..Camera::default() };
        let pose = w.body_pose().unwrap();
        let ext = RobotExtent::of(&w, &w.tilted_robot_boxes(&pose, &w.base_tilt(&pose)));
        let fr = ChaseFraming::fit(&ext, &cam);
        assert!((fr.dist_m - CHASE_DIST_M).abs() < 0.6 && (fr.height_m - CHASE_HEIGHT_M).abs() < 0.35 && (fr.target_m - 0.75).abs() < 0.2, "{fr:?} for {ext:?}");
        let small = RobotExtent { height_m: 0.5 * ext.height_m, half_span_m: 0.5 * ext.half_span_m, radius_m: 0.5 * ext.radius_m };
        let fs = ChaseFraming::fit(&small, &cam);
        assert!((fs.dist_m - 0.5 * fr.dist_m).abs() < 1e-4 && (fs.height_m - 0.5 * fr.height_m).abs() < 1e-4);
        let (_, o) = render(&w, &cam, 30.0);
        let o = o.unwrap();
        assert_eq!(o["camera"]["kind"], "chase");
        assert!(o["camera"]["dist_m"].as_f64().unwrap() > 1.0);
    }
}
