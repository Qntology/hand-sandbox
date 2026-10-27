//! The isometric third-person view (v1.0): an orthographic camera that
//! looks down on the robot from a world-fixed direction, so the picture does
//! not swing when the robot turns. Numbers: `docs/research/camera.md` §10.
//!
//! * Angles: the camera turns 45° (+ k·90°, the four diagonals) about the
//!   vertical and looks down by `arcsin(tan 30°) = 35.264°` (true isometric:
//!   the three world axes project 120° apart, each shortened to √(2/3)) or
//!   by 30° (the 2:1 "pixel-art" dimetric: ground lines at 26.565° on screen).
//! * Framing: the frame's half height is a multiple of the robot's bounding
//!   radius ([`RobotExtent`]) over the zoom; where the frame is centred is
//!   the caller's (the runtime smooths it in simulated time).
//! * Cutaway: what stands between the camera and the robot is cut down to
//!   the sight line that reaches the robot's feet, and what stands over it
//!   is cut above its head, so walls, shelves and roofs do not hide it;
//!   everything else stays whole ([`Cutaway`]). Shadows still come from the
//!   whole scene.
//! * The robot's links are drawn with the chassis's pitch and roll on the
//!   ground under it (a wheeled or tracked base; legs keep the torso level).

use serde::Serialize;

use crate::render::{basis, render_rays, Camera, Cutaway, Frame, Obb, Rays, ORTHO_BACK_M};
use crate::world::World;

/// Downward pitch of a true isometric view, degrees: `arcsin(tan 30°)`.
pub const TRUE_ISO_PITCH_DEG: f32 = 35.264;
/// Downward pitch of the 2:1 dimetric view, degrees.
pub const DIMETRIC_PITCH_DEG: f32 = 30.0;
/// Heading of the first of the four iso directions, degrees (renderer
/// headings: 0 looks along +x, 90 along +z).
pub const ISO_BASE_YAW_DEG: f32 = 45.0;
/// Frame half height over the robot's bounding radius at zoom 1 (design: the
/// robot and a few metres around it).
pub const ISO_HALF_H_PER_RADIUS: f32 = 3.0;
/// Smallest frame half height, metres (design: a small robot still shows its surroundings).
pub const ISO_MIN_HALF_H_M: f32 = 1.5;
/// Cutaway: margin beyond the robot's half span of the sight corridor (to
/// each side) and of the zone over it, metres (design).
pub const CUT_MARGIN_M: f32 = 0.5;
/// Cutaway: nothing in the sight corridor is cut lower than this above the
/// robot's ground, so what lies at its feet (a parcel, a step) stays, metres (design).
pub const CUT_LOW_M: f32 = 0.5;
/// Clearance over the robot's top above which what stands over it is cut, metres (design).
pub const CUT_ROOF_CLEAR_M: f32 = 0.3;
/// Head clearance above the eye counted in the robot's standing height, metres (est).
pub const HEAD_TOP_ABOVE_EYE_M: f32 = 0.1;

/// An orthographic view: parallel rays along the heading `yaw` and the
/// `pitch` (negative looks down, radians), the frame `half_h_m` metres from
/// its centre to its top edge, centred on `centre`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct IsoView {
    pub centre: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub half_h_m: f32,
}

/// Heading of iso direction `az_step` (0..3), radians.
pub fn iso_yaw(az_step: u8) -> f32 {
    (ISO_BASE_YAW_DEG + 90.0 * (az_step % 4) as f32).to_radians()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl IsoView {
    /// The iso view of direction `az_step`, true isometric or (with
    /// `dimetric`) 2:1 dimetric, centred on `centre`.
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

    /// Pixel and depth (along the view, from the ray origins' plane) of a
    /// world point. Every point projects; `Option` matches the perspective
    /// projectors the overlay takes.
    pub fn project(&self, cam: &Camera, p: [f32; 3]) -> Option<(f32, f32, f32)> {
        let (fwd, right, up) = basis(self.yaw, self.pitch);
        let rel = [p[0] - self.centre[0], p[1] - self.centre[1], p[2] - self.centre[2]];
        let (w, h) = (cam.width.max(1) as f32, cam.height.max(1) as f32);
        let x = (dot(rel, right) / self.half_w_m(cam) + 1.0) * 0.5 * w;
        let y = (1.0 - dot(rel, up) / self.half_h_m) * 0.5 * h;
        Some((x, y, ORTHO_BACK_M + dot(rel, fwd)))
    }

    /// The world point seen at pixel `(px, py)` at `depth` (as the frame's
    /// depth buffer holds it), the inverse of [`IsoView::project`].
    pub fn world_point(&self, cam: &Camera, px: f32, py: f32, depth: f32) -> [f32; 3] {
        let (fwd, right, up) = basis(self.yaw, self.pitch);
        let (w, h) = (cam.width.max(1) as f32, cam.height.max(1) as f32);
        let sx = (2.0 * px / w - 1.0) * self.half_w_m(cam);
        let sy = (1.0 - 2.0 * py / h) * self.half_h_m;
        let a = depth - ORTHO_BACK_M;
        [0, 1, 2].map(|k| self.centre[k] + right[k] * sx + up[k] * sy + fwd[k] * a)
    }

    /// Unit ground direction from the centre toward the camera, `(x, z)`.
    pub fn toward_camera(&self) -> [f32; 2] {
        let (s, c) = self.yaw.sin_cos();
        [-c, -s]
    }
}

/// How big the robot is, for framing a view of it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RobotExtent {
    /// Height of its top above the ground under it, metres (at least its standing height).
    pub height_m: f32,
    /// Farthest any link reaches from its vertical axis, metres.
    pub half_span_m: f32,
    /// Radius of the sphere around its middle that holds it, metres.
    pub radius_m: f32,
}

impl RobotExtent {
    /// The extent of the robot's link boxes (world), with the standing
    /// height as the floor, so a bend does not zoom the view in.
    pub fn of(world: &World, boxes: &[Obb]) -> RobotExtent {
        let ground = world.agent_ground_m();
        let (ax, az) = (world.agent.x, world.agent.z);
        let (mut top, mut span) = (world.standing_eye_m() + HEAD_TOP_ABOVE_EYE_M, 0.0f32);
        for b in boxes {
            // The box's half extent along each world axis.
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

    /// Half height of an iso frame at a zoom (1 = the default framing).
    pub fn iso_half_h_m(&self, zoom: f32) -> f32 {
        (ISO_HALF_H_PER_RADIUS * self.radius_m).max(ISO_MIN_HALF_H_M) / zoom.clamp(0.25, 4.0)
    }
}

impl Cutaway {
    /// The cutaway that keeps the robot (standing on `ground_m`, `extent`
    /// big, at `(x, z)`) in sight from an iso view: the sight lines rise
    /// `tan(pitch)` per metre toward the camera.
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

/// Rotates a base-frame vector (x forward, y left, z up) by the chassis's
/// roll (right side down) and then its pitch (nose up), radians.
fn tilt(v: [f32; 3], pitch: f32, roll: f32) -> [f32; 3] {
    let (sr, cr) = roll.sin_cos();
    let r = [v[0], v[1] * cr - v[2] * sr, v[1] * sr + v[2] * cr];
    let (sp, cp) = pitch.sin_cos();
    [r[0] * cp - r[2] * sp, r[1], r[0] * sp + r[2] * cp]
}

/// How a pose's base frame lies on the ground: the chassis's pitch (nose up)
/// and roll (right side down) under its contacts, radians, and the pitch the
/// pose already carries (a wheeled pose pitches its chassis by the slope of
/// the edge it drives, `noto_core::kin`), which comes off first so the drawn
/// tilt is the chassis's, not the sum of both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct BaseTilt {
    pub pitch: f32,
    pub roll: f32,
    pub pose_pitch: f32,
}

impl BaseTilt {
    /// A base-frame vector of the pose, tilted: the pose's own pitch undone,
    /// then the chassis's roll and pitch.
    pub fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let v = if self.pose_pitch != 0.0 { tilt(v, -self.pose_pitch, 0.0) } else { v };
        tilt(v, self.pitch, self.roll)
    }
}

/// The pitch a pose gives its chassis box (nose up, radians; 0 without one).
fn pose_chassis_pitch(pose: &noto_core::dynamics::BodyPose) -> f32 {
    pose.boxes.iter().find(|b| b.name == "chassis").map_or(0.0, |b| b.axes[0][2].atan2(b.axes[0][0]))
}

impl World {
    /// Pitch (nose up) and roll (right side down) of the chassis on the
    /// ground under its contacts, radians: the head camera's
    /// (`World::chassis_attitude`, `HeadView::tilt_deg`), zero for legs.
    pub fn chassis_tilt(&self) -> (f32, f32) {
        if self.body.legged {
            (0.0, 0.0)
        } else {
            self.chassis_attitude(self.agent.x, self.agent.z, self.agent.yaw)
        }
    }

    /// How `pose` is tilted on the ground now (zero for legs, which keep the
    /// torso level).
    pub fn base_tilt(&self, pose: &noto_core::dynamics::BodyPose) -> BaseTilt {
        if self.body.legged {
            return BaseTilt::default();
        }
        let (pitch, roll) = self.chassis_tilt();
        BaseTilt { pitch, roll, pose_pitch: pose_chassis_pitch(pose) }
    }

    /// A base-frame point of a pose in world coordinates with the pose
    /// tilted by `tilt` about the ground point under the robot.
    pub fn tilted_body_to_world(&self, p: [f32; 3], tilt: &BaseTilt) -> [f32; 3] {
        self.body_to_world(tilt.apply(p))
    }

    /// The links of a pose as world boxes, tilted by `tilt` (see
    /// [`World::robot_boxes`] for the untilted ones).
    pub fn tilted_robot_boxes(&self, pose: &noto_core::dynamics::BodyPose, tilt: &BaseTilt) -> Vec<Obb> {
        self.robot_boxes(pose)
            .into_iter()
            .zip(&pose.boxes)
            .map(|(o, b)| Obb { center: self.tilted_body_to_world(b.center, tilt), axes: b.axes.map(|a| self.body_dir_to_world(tilt.apply(a))), ..o })
            .collect()
    }
}

/// Renders an iso view with the robot's links (world boxes, drawn and
/// casting shadows) and an optional cutaway. The id buffer is 0 on the
/// robot; the depth buffer holds depths from the rays' origin plane.
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
            // The centre lands in the middle of the frame.
            let (x, y, _) = v.project(&cam, v.centre).unwrap();
            assert!((x - 80.0).abs() < 1e-3 && (y - 60.0).abs() < 1e-3);
        }
        // True iso: the axes toward the camera project 120° apart and equally shortened.
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
        // The camera sits on the side the view looks from.
        let t = v.toward_camera();
        assert!(t[0] < 0.0 && t[1] < 0.0);
    }

    #[test]
    fn the_iso_view_shows_the_robot_in_the_middle_from_outside_the_map() {
        let w = world(BaseKind::Wheel);
        let pose = w.body_pose().unwrap();
        let boxes = w.tilted_robot_boxes(&pose, &w.base_tilt(&pose));
        let ext = RobotExtent::of(&w, &boxes);
        assert!(ext.height_m > 1.0 && ext.height_m < 2.0 && ext.radius_m > 0.5, "{ext:?}");
        let g = w.agent_ground_m();
        let cam = Camera { width: 160, height: 120, max_dist_m: 60.0, ..Camera::default() };
        for step in 0..4u8 {
            let v = IsoView::new([w.agent.x, g + 0.5 * ext.height_m, w.agent.z], step, ext.iso_half_h_m(1.0), false);
            let f = render_iso(&w, &cam, &v, &boxes, None);
            // The robot (id 0 with a finite depth) fills the middle of the frame.
            let robot: Vec<usize> = (0..f.width * f.height).filter(|&i| f.ids[i] == 0 && f.depth[i].is_finite()).collect();
            assert!(robot.len() > 150, "{step}: {} robot pixels", robot.len());
            let cx = robot.iter().map(|&i| (i % f.width) as f32).sum::<f32>() / robot.len() as f32;
            let cy = robot.iter().map(|&i| (i / f.width) as f32).sum::<f32>() / robot.len() as f32;
            assert!((cx - 80.0).abs() < 12.0 && (cy - 60.0).abs() < 15.0, "{step}: robot at ({cx}, {cy})");
            // The ground around it is drawn, though every ray starts outside the map.
            let ground = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Ground).count();
            assert!(ground > f.width * f.height / 2, "{step}: {ground} ground pixels");
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
        // A tree one cell toward the camera.
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
        // The tree's stump is still there.
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
        // One tree 2 m behind the robot, one 2.5 m to its side.
        for (id, a, l) in [(1u32, -2.0f32, 0.0f32), (2, 0.0, 2.5)] {
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
        // Nose up by 10°: the boxes high on the robot move back (against the heading).
        let up = w.tilted_robot_boxes(&pose, &BaseTilt { pitch: 10f32.to_radians(), ..BaseTilt::default() });
        let (c, s) = (w.agent.yaw.cos(), w.agent.yaw.sin());
        let ahead = |b: &Obb| (b.center[0] - w.agent.x) * c + (b.center[2] - w.agent.z) * s;
        let (hi, hi_up) = flat.iter().zip(&up).max_by(|a, b| a.0.center[1].total_cmp(&b.0.center[1])).unwrap();
        assert!(ahead(hi_up) < ahead(hi) - 0.1, "{} vs {}", ahead(hi_up), ahead(hi));
        // Right side down: the top leans to the right (+z when heading +x).
        let rolled = w.tilted_body_to_world([0.0, 0.0, 1.0], &BaseTilt { roll: 0.2, ..BaseTilt::default() });
        let straight = w.body_to_world([0.0, 0.0, 1.0]);
        let right = [-s, c];
        assert!((rolled[0] - straight[0]) * right[0] + (rolled[2] - straight[2]) * right[1] > 0.15);
        // On level ground the chassis lies flat and the camera's tilt agrees.
        let hv = w.head_view(&Camera::default()).tilt_deg;
        let (p, r) = w.chassis_tilt();
        assert!((p.to_degrees() - hv[0]).abs() < 1e-4 && (r.to_degrees() - hv[1]).abs() < 1e-4);
        // Legs keep the torso level: no tilt at all.
        let legs = world(BaseKind::HumanFoot);
        assert_eq!(legs.base_tilt(&legs.body_pose().unwrap()), BaseTilt::default());
    }

    #[test]
    fn a_pose_that_already_pitches_its_chassis_is_not_pitched_twice() {
        // A wheeled pose on a 12° slope (`kin` pitches its boxes about the
        // ground point); the chassis under the contacts says 12° too: the
        // drawn chassis lies at 12°, not 24°.
        let w = world(BaseKind::Wheel);
        let body = &w.body;
        let pose = noto_core::kin::KinPose { base_pitch_deg: 12.0, ..noto_core::kin::KinPose::stand(0.0) };
        let sloped = noto_core::kin::solve(body, &pose, &noto_core::kin::KinLoads::default()).body_pose(0.0, noto_core::Source::Slope);
        assert!((pose_chassis_pitch(&sloped).to_degrees() - 12.0).abs() < 1e-3);
        let t = BaseTilt { pitch: 12f32.to_radians(), roll: 0.0, pose_pitch: pose_chassis_pitch(&sloped) };
        let boxes = w.tilted_robot_boxes(&sloped, &t);
        let chassis = sloped.boxes.iter().position(|b| b.name == "chassis").unwrap();
        let fwd = boxes[chassis].axes[0];
        let (c, s) = (w.agent.yaw.cos(), w.agent.yaw.sin());
        let along = fwd[0] * c + fwd[2] * s;
        assert!((fwd[1].atan2(along).to_degrees() - 12.0).abs() < 1e-3, "{fwd:?}");
        // With the pose's pitch left on, the chassis would stand at 24°.
        let twice = w.tilted_robot_boxes(&sloped, &BaseTilt { pose_pitch: 0.0, ..t });
        let f2 = twice[chassis].axes[0];
        assert!((f2[1].atan2(f2[0] * c + f2[2] * s).to_degrees() - 24.0).abs() < 1e-3);
    }
}

