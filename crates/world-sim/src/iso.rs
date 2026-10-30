use serde::Serialize;

use crate::render::{basis, render_rays, Camera, Cutaway, Frame, Obb, Rays, ORTHO_BACK_M};
use crate::world::World;

pub const TRUE_ISO_PITCH_DEG: f32 = 35.264;
pub const DIMETRIC_PITCH_DEG: f32 = 30.0;
pub const ISO_BASE_YAW_DEG: f32 = 45.0;
pub const ISO_HALF_H_PER_RADIUS: f32 = 3.0;
pub const ISO_MIN_HALF_H_M: f32 = 1.5;
pub const CUT_MARGIN_M: f32 = 0.5;
pub const CUT_LOW_M: f32 = 0.5;
pub const CUT_ROOF_CLEAR_M: f32 = 0.3;
pub const HEAD_TOP_ABOVE_EYE_M: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct IsoView {
    pub centre: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub half_h_m: f32,
}

pub fn iso_yaw(az_step: u8) -> f32 {
    (ISO_BASE_YAW_DEG + 90.0 * (az_step % 4) as f32).to_radians()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl IsoView {
    pub fn new(centre: [f32; 3], az_step: u8, half_h_m: f32, dimetric: bool) -> IsoView {
        let pitch = -(if dimetric { DIMETRIC_PITCH_DEG } else { TRUE_ISO_PITCH_DEG }).to_radians();
        IsoView { centre, yaw: iso_yaw(az_step), pitch, half_h_m: half_h_m.max(0.1) }
    }

    pub fn rays(&self) -> Rays {
        Rays::Ortho { centre: self.centre, yaw: self.yaw, pitch: self.pitch, half_h_m: self.half_h_m }
    }

    fn half_w_m(&self, cam: &Camera) -> f32 {
        self.half_h_m * cam.width.max(1) as f32 / cam.height.max(1) as f32
    }

    pub fn project(&self, cam: &Camera, p: [f32; 3]) -> Option<(f32, f32, f32)> {
        let (fwd, right, up) = basis(self.yaw, self.pitch);
        let rel = [p[0] - self.centre[0], p[1] - self.centre[1], p[2] - self.centre[2]];
        let (w, h) = (cam.width.max(1) as f32, cam.height.max(1) as f32);
        let x = (dot(rel, right) / self.half_w_m(cam) + 1.0) * 0.5 * w;
        let y = (1.0 - dot(rel, up) / self.half_h_m) * 0.5 * h;
        Some((x, y, ORTHO_BACK_M + dot(rel, fwd)))
    }

    pub fn world_point(&self, cam: &Camera, px: f32, py: f32, depth: f32) -> [f32; 3] {
        let (fwd, right, up) = basis(self.yaw, self.pitch);
        let (w, h) = (cam.width.max(1) as f32, cam.height.max(1) as f32);
        let sx = (2.0 * px / w - 1.0) * self.half_w_m(cam);
        let sy = (1.0 - 2.0 * py / h) * self.half_h_m;
        let a = depth - ORTHO_BACK_M;
        [0, 1, 2].map(|k| self.centre[k] + right[k] * sx + up[k] * sy + fwd[k] * a)
    }

    pub fn toward_camera(&self) -> [f32; 2] {
        let (s, c) = self.yaw.sin_cos();
        [-c, -s]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RobotExtent {
    pub height_m: f32,
    pub half_span_m: f32,
    pub radius_m: f32,
}

impl RobotExtent {
    pub fn of(world: &World, boxes: &[Obb]) -> RobotExtent {
        let ground = world.agent_ground_m();
        let (ax, az) = (world.agent.x, world.agent.z);
        let (mut top, mut span) = (world.standing_eye_m() + HEAD_TOP_ABOVE_EYE_M, 0.0f32);
        for b in boxes {
            let ext = [0, 1, 2].map(|i| (0..3).map(|k| (b.axes[k][i] * b.half[k]).abs()).sum::<f32>());
            top = top.max(b.center[1] + ext[1] - ground);
            for sx in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    span = span.max((b.center[0] + sx * ext[0] - ax).hypot(b.center[2] + sz * ext[2] - az));
                }
            }
        }
        let span = span.max(0.15);
        RobotExtent { height_m: top, half_span_m: span, radius_m: (0.25 * top * top + span * span).sqrt() }
    }

    pub fn iso_half_h_m(&self, zoom: f32) -> f32 {
        (ISO_HALF_H_PER_RADIUS * self.radius_m).max(ISO_MIN_HALF_H_M) / zoom.clamp(0.25, 4.0)
    }
}

impl Cutaway {
    pub fn around(view: &IsoView, at: [f32; 2], ground_m: f32, extent: &RobotExtent) -> Cutaway {
        let reach = extent.half_span_m + CUT_MARGIN_M;
        Cutaway {
            at,
            toward_cam: view.toward_camera(),
            half_width_m: reach,
            slope: (-view.pitch).max(0.0).tan(),
            base_y: ground_m,
            low_y: ground_m + CUT_LOW_M,
            roof_radius_m: reach,
            roof_y: ground_m + extent.height_m + CUT_ROOF_CLEAR_M,
        }
    }
}

fn tilt(v: [f32; 3], pitch: f32, roll: f32) -> [f32; 3] {
    let (sr, cr) = roll.sin_cos();
    let r = [v[0], v[1] * cr - v[2] * sr, v[1] * sr + v[2] * cr];
    let (sp, cp) = pitch.sin_cos();
    [r[0] * cp - r[2] * sp, r[1], r[0] * sp + r[2] * cp]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BaseTilt {
    pub pitch: f32,
    pub roll: f32,
    pub pose_pitch: f32,
}

impl BaseTilt {
    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let v = if self.pose_pitch != 0.0 { tilt(v, -self.pose_pitch, 0.0) } else { v };
        tilt(v, self.pitch, self.roll)
    }
}

fn pose_chassis_pitch(pose: &noto_core::dynamics::BodyPose) -> f32 {
    pose.boxes.iter().find(|b| b.name == "chassis").map_or(0.0, |b| b.axes[0][2].atan2(b.axes[0][0]))
}

impl World {
    pub fn chassis_tilt(&self) -> (f32, f32) {
        if self.body.legged {
            (0.0, 0.0)
        } else {
            self.chassis_attitude(self.agent.x, self.agent.z, self.agent.yaw)
        }
    }

    pub fn base_tilt(&self, pose: &noto_core::dynamics::BodyPose) -> BaseTilt {
        if self.body.legged {
            return BaseTilt::default();
        }
        let (pitch, roll) = self.chassis_tilt();
        BaseTilt { pitch, roll, pose_pitch: pose_chassis_pitch(pose) }
    }

    pub fn tilted_body_to_world(&self, p: [f32; 3], tilt: &BaseTilt) -> [f32; 3] {
        self.body_to_world(tilt.apply(p))
    }

    pub fn tilted_robot_boxes(&self, pose: &noto_core::dynamics::BodyPose, tilt: &BaseTilt) -> Vec<Obb> {
        // A truck's parts are drawn level on its chassis, so the whole of the chassis tilt applies to them.
        let tilt = if noto_core::act::fork::applies(&self.body) { BaseTilt { pose_pitch: 0.0, ..*tilt } } else { *tilt };
        self.robot_parts(pose).iter().map(|p| Obb::placed(p, |c| self.tilted_body_to_world(c, &tilt), |a| self.body_dir_to_world(tilt.apply(a)))).collect()
    }
}

pub fn render_iso(world: &World, cam: &Camera, view: &IsoView, robot: &[Obb], cut: Option<&Cutaway>) -> Frame {
    render_rays(world, cam, &view.rays(), robot, &[], cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::{unpack_id, ObjectClass};
    use crate::world::WorldConfig;
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn world(base: BaseKind) -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = base;
        World::new(WorldConfig { seed: 7, cells: 48, noto: hand, parcels: Some(0), ..WorldConfig::default() }).unwrap()
    }

    #[test]
    fn iso_projection_round_trips_and_keeps_the_iso_angles() {
        let cam = Camera { width: 160, height: 120, ..Camera::default() };
        for (step, dimetric) in [(0u8, false), (1, false), (2, true), (3, false)] {
            let v = IsoView::new([10.0, 0.6, 12.0], step, 3.0, dimetric);
            for p in [[10.0f32, 0.6, 12.0], [11.2, 0.1, 12.7], [8.5, 1.4, 13.1], [10.4, -0.2, 9.9]] {
                let (x, y, d) = v.project(&cam, p).unwrap();
                let q = v.world_point(&cam, x, y, d);
                assert!((0..3).all(|k| (q[k] - p[k]).abs() < 1e-3), "{step}: {p:?} -> {q:?}");
            }
            let (x, y, _) = v.project(&cam, v.centre).unwrap();
            assert!((x - 80.0).abs() < 1e-3 && (y - 60.0).abs() < 1e-3);
        }
        let v = IsoView::new([0.0; 3], 0, 1.0, false);
        let cam = Camera { width: 100, height: 100, ..Camera::default() };
        let screen = |a: [f32; 3]| {
            let (x, y, _) = v.project(&cam, a).unwrap();
            [(x - 50.0) / 50.0, (50.0 - y) / 50.0]
        };
        let axes = [screen([-1.0, 0.0, 0.0]), screen([0.0, 1.0, 0.0]), screen([0.0, 0.0, -1.0])];
        for a in axes {
            assert!((a[0].hypot(a[1]) - (2.0f32 / 3.0).sqrt()).abs() < 2e-4, "{a:?}");
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let c = (axes[i][0] * axes[j][0] + axes[i][1] * axes[j][1]) / (axes[i][0].hypot(axes[i][1]) * axes[j][0].hypot(axes[j][1]));
            assert!((c.acos().to_degrees() - 120.0).abs() < 0.05, "{i}{j}: {}", c.acos().to_degrees());
        }
        let t = v.toward_camera();
        assert!(t[0] < 0.0 && t[1] < 0.0);
    }

    #[test]
    fn the_iso_view_shows_the_robot_in_the_middle_from_outside_the_map() {
        let mut w = world(BaseKind::Wheel);
        w.nodes.clear();
        let pose = w.body_pose().unwrap();
        let boxes = w.tilted_robot_boxes(&pose, &w.base_tilt(&pose));
        let ext = RobotExtent::of(&w, &boxes);
        assert!(ext.height_m > 1.0 && ext.height_m < 2.4 && ext.radius_m > 0.5, "{ext:?}");
        let g = w.agent_ground_m();
        let cam = Camera { width: 160, height: 120, max_dist_m: 60.0, ..Camera::default() };
        for step in 0..4u8 {
            let v = IsoView::new([w.agent.x, g + 0.5 * ext.height_m, w.agent.z], step, ext.iso_half_h_m(1.0), false);
            let f = render_iso(&w, &cam, &v, &boxes, None);
            let robot: Vec<usize> = (0..f.width * f.height).filter(|&i| f.ids[i] == 0 && f.depth[i].is_finite()).collect();
            assert!(robot.len() > 150, "{step}: {} robot pixels", robot.len());
            let cx = robot.iter().map(|&i| (i % f.width) as f32).sum::<f32>() / robot.len() as f32;
            let cy = robot.iter().map(|&i| (i / f.width) as f32).sum::<f32>() / robot.len() as f32;
            assert!((cx - 80.0).abs() < 12.0 && (cy - 60.0).abs() < 15.0, "{step}: robot at ({cx}, {cy})");
            let ground = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Ground).count();
            assert!(ground > f.width * f.height / 3, "{step}: {ground} ground pixels");
        }
    }

    #[test]
    fn a_tree_in_front_of_the_robot_is_cut_away() {
        let mut w = world(BaseKind::HumanFoot);
        w.nodes.clear();
        let pose = w.body_pose().unwrap();
        let boxes = w.tilted_robot_boxes(&pose, &w.base_tilt(&pose));
        let ext = RobotExtent::of(&w, &boxes);
        let g = w.agent_ground_m();
        let v = IsoView::new([w.agent.x, g + 0.5 * ext.height_m, w.agent.z], 0, 2.0, false);
        let t = v.toward_camera();
        let (cx, cz) = w.terrain.cell_of(w.agent.x + 0.7 * t[0], w.agent.z + 0.7 * t[1]).unwrap();
        let (x, z) = crate::terrain::cell_center(cx, cz);
        w.nodes.push(crate::world::Node { id: 3, kind: crate::kinds::NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 });
        let cam = Camera { width: 128, height: 96, max_dist_m: 60.0, ..Camera::default() };
        let robot_px = |f: &Frame| (0..f.width * f.height).filter(|&i| f.ids[i] == 0 && f.depth[i].is_finite()).count();
        let hidden = render_iso(&w, &cam, &v, &boxes, None);
        let cut = Cutaway::around(&v, [w.agent.x, w.agent.z], g, &ext);
        let shown = render_iso(&w, &cam, &v, &boxes, Some(&cut));
        assert!(robot_px(&shown) > robot_px(&hidden) + 20, "cut {} vs uncut {}", robot_px(&shown), robot_px(&hidden));
        assert!(shown.ids.iter().any(|&v| v == crate::render::pack_id(ObjectClass::Node, 3)));
    }

    #[test]
    fn trees_that_cannot_hide_the_robot_stay_whole() {
        let mut w = world(BaseKind::HumanFoot);
        w.nodes.clear();
        w.drops.clear();
        w.waypoints.clear();
        let pose = w.body_pose().unwrap();
        let boxes = w.tilted_robot_boxes(&pose, &w.base_tilt(&pose));
        let ext = RobotExtent::of(&w, &boxes);
        let g = w.agent_ground_m();
        let v = IsoView::new([w.agent.x, g + 0.5 * ext.height_m, w.agent.z], 0, 3.0, false);
        let t = v.toward_camera();
        let side = [-t[1], t[0]];
        for (id, a, l) in [(1u32, -3.4f32, 0.0f32), (2, 0.0, 4.0)] {
            let (cx, cz) = w.terrain.cell_of(w.agent.x + a * t[0] + l * side[0], w.agent.z + a * t[1] + l * side[1]).unwrap();
            let (x, z) = crate::terrain::cell_center(cx, cz);
            w.nodes.push(crate::world::Node { id, kind: crate::kinds::NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 });
        }
        let cam = Camera { width: 128, height: 96, max_dist_m: 60.0, ..Camera::default() };
        let cut = Cutaway::around(&v, [w.agent.x, w.agent.z], g, &ext);
        let (whole, with_cut) = (render_iso(&w, &cam, &v, &boxes, None), render_iso(&w, &cam, &v, &boxes, Some(&cut)));
        for id in [1, 2] {
            assert!(whole.ids.iter().any(|&v| v == crate::render::pack_id(ObjectClass::Node, id)), "tree {id} in the frame");
        }
        assert!(whole.rgb == with_cut.rgb && whole.ids == with_cut.ids, "the cutaway changed a frame with nothing in the way");
    }

    #[test]
    fn a_tilted_chassis_tilts_its_boxes() {
        let w = world(BaseKind::Wheel);
        let pose = w.body_pose().unwrap();
        let flat = w.tilted_robot_boxes(&pose, &BaseTilt::default());
        assert_eq!(flat, w.robot_boxes(&pose));
        let up = w.tilted_robot_boxes(&pose, &BaseTilt { pitch: 10f32.to_radians(), ..BaseTilt::default() });
        let (c, s) = (w.agent.yaw.cos(), w.agent.yaw.sin());
        let ahead = |b: &Obb| (b.center[0] - w.agent.x) * c + (b.center[2] - w.agent.z) * s;
        let (hi, hi_up) = flat.iter().zip(&up).max_by(|a, b| a.0.center[1].total_cmp(&b.0.center[1])).unwrap();
        assert!(ahead(hi_up) < ahead(hi) - 0.1, "{} vs {}", ahead(hi_up), ahead(hi));
        let rolled = w.tilted_body_to_world([0.0, 0.0, 1.0], &BaseTilt { roll: 0.2, ..BaseTilt::default() });
        let straight = w.body_to_world([0.0, 0.0, 1.0]);
        let right = [-s, c];
        assert!((rolled[0] - straight[0]) * right[0] + (rolled[2] - straight[2]) * right[1] > 0.15);
        let hv = w.head_view(&Camera::default()).tilt_deg;
        let (p, r) = w.chassis_tilt();
        assert!((p.to_degrees() - hv[0]).abs() < 1e-4 && (r.to_degrees() - hv[1]).abs() < 1e-4);
        let legs = world(BaseKind::HumanFoot);
        assert_eq!(legs.base_tilt(&legs.body_pose().unwrap()), BaseTilt::default());
    }

    #[test]
    fn a_pose_that_already_pitches_its_chassis_is_not_pitched_twice() {
        let w = world(BaseKind::Wheel);
        let body = &w.body;
        let pose = noto_core::kin::KinPose { base_pitch_deg: 12.0, ..noto_core::kin::KinPose::stand(body, 0.0) };
        let sloped = noto_core::kin::solve(body, &pose, &noto_core::kin::KinLoads::default()).body_pose(0.0, noto_core::Source::Slope);
        assert!((pose_chassis_pitch(&sloped).to_degrees() - 12.0).abs() < 1e-3);
        let t = BaseTilt { pitch: 12f32.to_radians(), roll: 0.0, pose_pitch: pose_chassis_pitch(&sloped) };
        let boxes = w.tilted_robot_boxes(&sloped, &t);
        let fwd = boxes[0].axes[0];
        let (c, s) = (w.agent.yaw.cos(), w.agent.yaw.sin());
        let along = fwd[0] * c + fwd[2] * s;
        assert!((fwd[1].atan2(along).to_degrees() - 12.0).abs() < 1e-3, "{fwd:?}");
        assert_eq!(boxes, w.tilted_robot_boxes(&sloped, &BaseTilt { pose_pitch: 0.0, ..t }), "the truck's parts are drawn level, whatever the pose says");
    }
}
