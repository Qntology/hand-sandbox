//! What a robot looks like and how it is put together, from one source: the physics boxes (`kin`, `dynamics`) say where
//! each link is this instant, and the parts here dress those links, so every camera, the 3-D view and the editor draw
//! the body the physics moves. A humanoid is the same model wherever it appears: the first worker, a sub-worker, the
//! rider on a lift truck or on foot.
//!
//! A rig (`RigSpec`) is a robot as data: links with masses and shapes, joints, cameras. The built-in robots generate
//! theirs; one from the CAD editor is measured from its design (`rig::urdf`).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use serde::{Deserialize, Serialize};

use crate::balance::Body;
use crate::base::BaseKind;
use crate::dims::Dims;
use crate::dynamics::LinkBox;
use crate::kin::{self, KinLoads, KinPose};
use crate::parts::Segment;

pub mod urdf;

pub const SHELL: [u8; 3] = [226, 228, 232];
pub const TRIM: [u8; 3] = [150, 154, 164];
pub const JOINT: [u8; 3] = [48, 50, 58];
pub const VISOR: [u8; 3] = [20, 24, 30];
pub const TRUCK_YELLOW: [u8; 3] = [236, 172, 36];
pub const TRUCK_DARK: [u8; 3] = [46, 48, 54];
pub const TYRE: [u8; 3] = [28, 28, 32];
pub const STEEL: [u8; 3] = [96, 100, 110];
pub const TINE: [u8; 3] = [64, 66, 72];

/// The colours a body is drawn in: its shell, and the accent that tells one worker from another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    pub shell: [u8; 3],
    pub accent: [u8; 3],
    pub truck: [u8; 3],
}

impl Default for Palette {
    fn default() -> Self {
        Palette { shell: SHELL, accent: [236, 128, 52], truck: TRUCK_YELLOW }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Solid {
    #[default]
    Box,
    /// Round along `axes[2]`: radius `half[0]`, half length `half[2]`.
    Cylinder,
    /// A cylinder capped by balls of its radius; a ball when the half length is 0.
    Capsule,
    /// Triangles from a design (`mesh`), centred on `center`; `half` bounds them.
    Mesh(u32),
}

/// One solid of a robot's outside, in the body frame (x forward, y left, z up, from the ground under the base).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct VisualPart {
    pub solid: Solid,
    pub center: [f32; 3],
    pub axes: [[f32; 3]; 3],
    pub half: [f32; 3],
    pub color: [u8; 3],
}

type V3 = [f32; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: V3) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

const EYE3: [V3; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A link's frame in the body frame.
#[derive(Clone, Copy, Debug)]
struct Frame {
    o: V3,
    axes: [V3; 3],
}

impl Frame {
    const BODY: Frame = Frame { o: [0.0; 3], axes: EYE3 };

    fn of(b: &LinkBox) -> Frame {
        Frame { o: b.center, axes: b.axes }
    }

    fn dir(&self, d: V3) -> V3 {
        add(add(scale(self.axes[0], d[0]), scale(self.axes[1], d[1])), scale(self.axes[2], d[2]))
    }

    fn at(&self, p: V3) -> V3 {
        add(self.o, self.dir(p))
    }

    /// A box at local `c`, square to the frame.
    fn cube(&self, c: V3, half: V3, color: [u8; 3]) -> VisualPart {
        VisualPart { solid: Solid::Box, center: self.at(c), axes: self.axes, half, color }
    }

    /// A box between two local corners.
    fn span(&self, lo: V3, hi: V3, color: [u8; 3]) -> VisualPart {
        self.cube(scale(add(lo, hi), 0.5), [0.5 * (hi[0] - lo[0]).abs(), 0.5 * (hi[1] - lo[1]).abs(), 0.5 * (hi[2] - lo[2]).abs()], color)
    }

    fn rod(&self, solid: Solid, a: V3, b: V3, r: f32, color: [u8; 3]) -> VisualPart {
        let (a, b) = (self.at(a), self.at(b));
        let d = sub(b, a);
        let len = norm(d).max(1e-6);
        let z = scale(d, 1.0 / len);
        let x0 = if z[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let c = cross(z, x0);
        let y = scale(c, 1.0 / norm(c).max(1e-6));
        VisualPart { solid, center: scale(add(a, b), 0.5), axes: [cross(y, z), y, z], half: [r, r, 0.5 * len], color }
    }

    fn cylinder(&self, a: V3, b: V3, r: f32, color: [u8; 3]) -> VisualPart {
        self.rod(Solid::Cylinder, a, b, r, color)
    }

    fn capsule(&self, a: V3, b: V3, r: f32, color: [u8; 3]) -> VisualPart {
        self.rod(Solid::Capsule, a, b, r, color)
    }

    fn ball(&self, c: V3, r: f32, color: [u8; 3]) -> VisualPart {
        VisualPart { solid: Solid::Capsule, center: self.at(c), axes: self.axes, half: [r, r, 0.0], color }
    }
}

/// One link of a person-shaped body, by the name the physics gives its box. `half` is the physics box; the limb boxes'
/// z runs from the near joint to the far one.
fn dress_link(b: &LinkBox, d: &Dims, p: &Palette, v: &mut Vec<VisualPart>) {
    let f = Frame::of(b);
    let k = d.k;
    let [hx, hy, hz] = b.half;
    match b.name {
        "pelvis" => {
            v.push(f.cube([0.0, 0.0, -0.01 * k], [hx * 0.98, hy * 0.96, hz * 0.95], p.shell));
            v.push(f.cube([0.0, 0.0, hz * 0.80], [hx * 0.72, hy * 0.72, hz * 0.34], JOINT));
            for s in [-1.0f32, 1.0] {
                v.push(f.cylinder([0.0, s * (d.hip_half_m - 0.04 * k), -0.04 * k], [0.0, s * (d.hip_half_m + 0.05 * k), -0.04 * k], 0.062 * k, JOINT));
            }
        }
        "trunk" => {
            let waist = -hz;
            let ribs = waist + 0.72 * hz;
            // Abdomen, chest, and the shoulder girdle the arms hang from.
            v.push(f.span([-0.74 * hx, -0.70 * hy, waist], [0.74 * hx, 0.70 * hy, ribs + 0.02 * k], TRIM));
            v.push(f.span([-hx, -0.92 * hy, ribs], [hx, 0.92 * hy, hz], p.shell));
            v.push(f.capsule([0.0, -(d.shoulder_half_m - 0.03 * k), hz - 0.035 * k], [0.0, d.shoulder_half_m - 0.03 * k, hz - 0.035 * k], 0.060 * k, p.shell));
            v.push(f.cube([hx + 0.004 * k, 0.0, ribs + 0.42 * (hz - ribs)], [0.006 * k, 0.56 * hy, 0.30 * (hz - ribs)], p.accent));
            v.push(f.span([-hx - 0.045 * k, -0.62 * hy, ribs - 0.02 * k], [-hx, 0.62 * hy, hz - 0.02 * k], JOINT));
        }
        "head" => {
            v.push(f.cube([0.0, 0.0, 0.004 * k], [hx * 1.06, hy * 0.92, hz * 1.04], p.shell));
            v.push(f.cube([hx * 0.98, 0.0, hz * 0.20], [0.012 * k, hy * 0.78, hz * 0.40], VISOR));
            v.push(f.cube([0.0, 0.0, hz * 1.02], [hx * 0.78, hy * 0.70, 0.012 * k], p.accent));
            let neck = -0.07 * k;
            v.push(f.cylinder([0.0, 0.0, neck - d.neck_up_m * 0.82], [0.0, 0.0, neck + 0.02 * k], 0.046 * k, JOINT));
        }
        "upper_arm_r" | "upper_arm_l" => {
            v.push(f.capsule([0.0, 0.0, -hz + 0.015 * k], [0.0, 0.0, hz - 0.03 * k], hx * 1.30, p.shell));
            v.push(f.ball([0.0, 0.0, -hz], hx * 1.48, JOINT));
            v.push(f.ball([0.0, 0.0, hz], hx * 1.16, JOINT));
        }
        "forearm_r" | "forearm_l" => {
            v.push(f.capsule([0.0, 0.0, -hz + 0.03 * k], [0.0, 0.0, hz - 0.01 * k], hx * 1.22, p.shell));
            v.push(f.cylinder([0.0, 0.0, hz - 0.03 * k], [0.0, 0.0, hz + 0.008 * k], hx * 1.05, JOINT));
        }
        "palm_r" | "palm_l" | "hand_r" | "hand_l" => {
            v.push(f.cube([0.0; 3], [hx * 0.62, hy * 1.0, hz], JOINT));
        }
        "finger_r" | "finger_l" => {
            v.push(f.capsule([0.0, 0.0, -hz], [0.0, 0.0, hz], hx, TRIM));
        }
        "thigh_r" | "thigh_l" => {
            v.push(f.capsule([0.0, 0.0, -hz + 0.04 * k], [0.0, 0.0, hz - 0.05 * k], hx * 1.56, p.shell));
            v.push(f.ball([0.0, 0.0, hz], hx * 1.26, JOINT));
        }
        "shank_r" | "shank_l" => {
            v.push(f.capsule([0.0, 0.0, -hz + 0.05 * k], [0.0, 0.0, hz - 0.06 * k], hx * 1.34, p.shell));
            v.push(f.ball([0.0, 0.0, hz], hx * 0.96, JOINT));
        }
        "foot_r" | "foot_l" => {
            let t = hx.max(0.01);
            v.push(f.cube([0.0, 0.0, -hz * 0.55], [t, hy, hz * 0.45], JOINT));
            v.push(f.cube([-0.12 * t, 0.0, hz * 0.45], [t * 0.80, hy * 0.92, hz * 0.62], p.shell));
        }
        _ => {
            let c = match b.segment {
                Segment::Head => p.shell,
                Segment::Chest | Segment::Belly => TRIM,
                Segment::Arm => p.accent,
                Segment::Hand | Segment::Foot => JOINT,
                Segment::Leg => STEEL,
            };
            v.push(f.cube([0.0; 3], b.half, c));
        }
    }
}

/// Dresses the physics boxes of a body.
pub fn dress(body: &Body, boxes: &[LinkBox], palette: &Palette) -> Vec<VisualPart> {
    let rig = body.rig();
    let mut v = Vec::with_capacity(boxes.len() * 3);
    for b in boxes {
        let name = match b.name {
            "palm_r" => "hand_r",
            "palm_l" => "hand_l",
            n => n,
        };
        let drawn = rig.as_ref().map_or(0, |r| {
            let n0 = v.len();
            let f = Frame::of(b);
            v.extend(r.visuals.iter().filter(|m| m.link == name).map(|m| VisualPart { solid: Solid::Mesh(m.mesh), center: f.at(m.offset), axes: f.axes, half: m.half, color: m.color }));
            v.len() - n0
        });
        if drawn == 0 {
            dress_link(b, &body.dims, palette, &mut v);
        }
    }
    v
}

/// The upper body the physics solves for a wheeled base, seated: its hands on the steering wheel.
pub fn rider_parts(body: &Body, pose: &KinPose, palette: &Palette) -> Vec<VisualPart> {
    let d = &body.dims;
    let k = d.k;
    let seated = KinPose { shoulder_deg: [34.0, 34.0], elbow_deg: [66.0, 66.0], closure: 0.6, ..*pose };
    let mut boxes: Vec<LinkBox> = kin::solve(body, &seated, &KinLoads::default())
        .boxes
        .into_iter()
        .filter(|b| matches!(b.name, "pelvis" | "trunk" | "head" | "upper_arm_r" | "upper_arm_l" | "forearm_r" | "forearm_l" | "palm_r" | "palm_l" | "finger_r" | "finger_l"))
        .collect();
    let (hip, x) = (d.hip_m, d.seat_x_m);
    let knee_x = x + d.thigh_m();
    let floor = floor_z(body);
    for (s, y) in [-d.hip_half_m, d.hip_half_m].into_iter().enumerate() {
        let (hip_p, knee) = ([x, y, hip], [knee_x, y, hip - 0.02 * k]);
        let ankle = [knee_x + 0.08 * k, y, floor + d.ankle_m];
        boxes.push(limb(if s == 0 { "thigh_r" } else { "thigh_l" }, Segment::Leg, hip_p, knee, 0.05 * k));
        boxes.push(limb(if s == 0 { "shank_r" } else { "shank_l" }, Segment::Leg, knee, ankle, 0.045 * k));
        let fl = body.leg.foot_len_m;
        boxes.push(LinkBox { name: if s == 0 { "foot_r" } else { "foot_l" }, segment: Segment::Foot, center: [ankle[0] + 0.25 * fl, y, floor + 0.03 * k], axes: EYE3, half: [0.5 * fl, 0.5 * body.leg.foot_width_m, 0.03 * k] });
    }
    dress(body, &boxes, palette)
}

fn limb(name: &'static str, segment: Segment, a: V3, b: V3, w: f32) -> LinkBox {
    let p = Frame::BODY.rod(Solid::Box, a, b, w, [0; 3]);
    LinkBox { name, segment, center: p.center, axes: p.axes, half: p.half }
}

/// The truck's floor plate, where the rider's feet rest.
fn floor_z(body: &Body) -> f32 {
    let hood = body.dims.hip_m - 0.17 * body.dims.k;
    (0.55 * hood).max(body.dims.wheel_r_m + 0.04)
}

/// A wheeled or tracked base as a counterbalanced lift truck: frame, battery hood and counterweight, dash and steering
/// wheel, seat, overhead guard, and its wheels or tracks. Body frame.
pub fn truck_parts(body: &Body, p: &Palette) -> Vec<VisualPart> {
    if let Some(r) = body.rig().filter(|r| !r.visuals.is_empty()) {
        return r.visuals.iter().filter(|m| !m.on_carriage).map(|m| VisualPart { solid: Solid::Mesh(m.mesh), center: m.offset, axes: EYE3, half: m.half, color: m.color }).collect();
    }
    let d = &body.dims;
    let (l, w, r, tw, k) = (body.support_len_m, body.support_width_m, d.wheel_r_m, d.belt_w_m, d.k);
    let f = Frame::BODY;
    let mut v = Vec::with_capacity(64);
    let hood = d.hip_m - 0.17 * k;
    let floor = floor_z(body);
    let rear = -0.5 * l - 1.15 * r;
    let nose = 0.5 * l + 0.40 * r;
    let side = 0.5 * w;
    let inner = (side - tw - 0.015).max(0.05);
    // Under the rider: the frame between the wheels, then fenders out to the tyres above them.
    v.push(f.span([rear, -inner, 0.45 * r], [nose, inner, floor], TRUCK_DARK));
    let fender = (2.0 * r + 0.02).max(floor);
    v.push(f.span([rear, -side, fender], [d.seat_x_m + 0.24 * k, side, hood], p.truck));
    v.push(f.span([rear, -inner, floor], [d.seat_x_m + 0.24 * k, inner, fender], p.truck));
    v.push(f.span([rear - 0.05 * k, -side + 0.02, 0.45 * r], [rear + 0.14 * k, side - 0.02, hood - 0.08 * k], TRUCK_DARK));
    v.push(f.span([rear - 0.055 * k, -0.30 * w, hood - 0.26 * k], [rear - 0.05 * k, 0.30 * w, hood - 0.14 * k], [196, 40, 36]));
    // Operator's floor and the cowl in front of it.
    v.push(f.span([d.seat_x_m + 0.24 * k, -side, floor - 0.03 * k], [nose, side, floor], JOINT));
    let cowl = hood + 0.24 * k;
    v.push(f.span([nose - 0.17 * k, -side + 0.03, floor], [nose, side - 0.03, cowl], p.truck));
    v.push(f.span([nose - 0.21 * k, -0.34 * w, cowl - 0.02 * k], [nose - 0.02, 0.34 * w, cowl + 0.015 * k], TRUCK_DARK));
    // Seat.
    let (sx, sz) = (d.seat_x_m, d.hip_m - 0.075 * k);
    v.push(f.span([sx - 0.15 * k, -0.17 * k, hood], [sx + 0.13 * k, 0.17 * k, sz], TRUCK_DARK));
    v.push(f.cylinder([sx - 0.17 * k, -0.15 * k, sz + 0.02 * k], [sx - 0.22 * k, -0.15 * k, sz + 0.38 * k], 0.022 * k, TRUCK_DARK));
    v.push(f.cylinder([sx - 0.17 * k, 0.15 * k, sz + 0.02 * k], [sx - 0.22 * k, 0.15 * k, sz + 0.38 * k], 0.022 * k, TRUCK_DARK));
    v.push(f.span([sx - 0.235 * k, -0.17 * k, sz + 0.02 * k], [sx - 0.17 * k, 0.17 * k, sz + 0.42 * k], TRUCK_DARK));
    // Steering wheel on its column, where the seated rider's hands are.
    let hub = [sx + 0.52 * k, 0.0, d.hip_m + 0.13 * k];
    let tilt = [-0.50, 0.0, 0.866];
    v.push(f.cylinder([nose - 0.17 * k, 0.0, cowl - 0.02 * k], hub, 0.02 * k, JOINT));
    v.push(f.cylinder(add(hub, scale(tilt, -0.012 * k)), add(hub, scale(tilt, 0.012 * k)), 0.125 * k, TRUCK_DARK));
    // Overhead guard: four posts and a slatted roof.
    let head_top = d.column_top_m() + 0.20 * k;
    let roof = d.roof_m.unwrap_or(head_top + 0.08 * k);
    let (rx0, rx1) = (rear + 0.05 * k, nose - 0.02 * k);
    let py = side - 0.04 * k.min(1.0);
    for y in [-py, py] {
        v.push(f.cylinder([rx0 + 0.05 * k, y, hood], [rx0 + 0.05 * k, y, roof], 0.026 * k, TRUCK_DARK));
        v.push(f.cylinder([nose - 0.10 * k, y, cowl - 0.02 * k], [rx1 - 0.05 * k, y, roof], 0.026 * k, TRUCK_DARK));
        v.push(f.span([rx0, y - 0.026 * k, roof - 0.04 * k], [rx1, y + 0.026 * k, roof], TRUCK_DARK));
    }
    let slats = 7;
    for i in 0..slats {
        let x = rx0 + (rx1 - rx0) * i as f32 / (slats - 1) as f32;
        v.push(f.span([x - 0.022 * k, -py, roof - 0.035 * k], [x + 0.022 * k, py, roof - 0.005 * k], TRUCK_DARK));
    }
    wheels(body, &f, &mut v);
    v
}

fn wheels(body: &Body, f: &Frame, v: &mut Vec<VisualPart>) {
    let d = &body.dims;
    let (l, w, r, tw) = (body.support_len_m, body.support_width_m, d.wheel_r_m, d.belt_w_m);
    let ys = [-0.5 * w + 0.5 * tw, 0.5 * w - 0.5 * tw];
    if body.base == BaseKind::Track {
        for y in ys {
            let (a, b) = (0.5 * tw, 0.5 * tw);
            // A belt round a driving wheel at each end and rollers between them.
            v.push(f.span([-0.5 * l, y - a, 0.0], [0.5 * l, y + b, 2.0 * r], TYRE));
            for x in [-0.5 * l, 0.5 * l] {
                v.push(f.cylinder([x, y - a, r], [x, y + b, r], r, TYRE));
                v.push(f.cylinder([x, y - a - 0.006, r], [x, y + b + 0.006, r], 0.62 * r, STEEL));
            }
            let rollers = 3;
            for i in 1..rollers + 1 {
                let x = -0.5 * l + l * i as f32 / (rollers + 1) as f32;
                v.push(f.cylinder([x, y - a - 0.006, 0.55 * r], [x, y + b + 0.006, 0.55 * r], 0.42 * r, STEEL));
            }
        }
        return;
    }
    for x in [-0.5 * l, 0.5 * l] {
        for y in ys {
            v.push(f.cylinder([x, y - 0.5 * tw, r], [x, y + 0.5 * tw, r], r, TYRE));
            v.push(f.cylinder([x, y - 0.5 * tw - 0.006, r], [x, y + 0.5 * tw + 0.006, r], 0.56 * r, STEEL));
            if body.base == BaseKind::Mecanum {
                // Rollers at 45° round the rim.
                for i in 0..8 {
                    let a = i as f32 * std::f32::consts::FRAC_PI_4;
                    let (s, c) = a.sin_cos();
                    let rim = [x + 0.94 * r * c, y, r + 0.94 * r * s];
                    let along = [-0.5 * tw * 0.7 * s, 0.5 * tw, 0.5 * tw * 0.7 * c];
                    v.push(f.capsule(sub(rim, along), add(rim, along), 0.16 * r, STEEL));
                }
            }
        }
    }
}

/// The mast, carriage with its load backrest and the tines, at `fork_m`. Body frame.
pub fn fork_parts(body: &Body, fork_m: f32) -> Vec<VisualPart> {
    let d = &body.dims;
    if let Some(r) = body.rig().filter(|r| !r.visuals.is_empty()) {
        let up = fork_m.clamp(crate::act::fork::FORK_FLOOR_M, d.fork_max_m()) - d.fork_rest_m;
        return r.visuals.iter().filter(|m| m.on_carriage).map(|m| VisualPart { solid: Solid::Mesh(m.mesh), center: add(m.offset, [0.0, 0.0, up]), axes: EYE3, half: m.half, color: m.color }).collect();
    }
    let f = Frame::BODY;
    let fx = crate::act::fork::front_x(body);
    let hx = crate::act::fork::heel_x(body);
    let h = fork_m.clamp(crate::act::fork::FORK_FLOOR_M, d.fork_max_m());
    let s = d.mast_half_w_m / 0.30;
    let top = d.mast_top_m;
    let mw = d.mast_half_w_m;
    let mut v = Vec::with_capacity(32);
    for y in [-1.0f32, 1.0] {
        v.push(f.cube([fx + 0.03 * s, y * mw, 0.5 * (0.04 + top)], [0.03 * s, 0.024 * s, 0.5 * (top - 0.04)], STEEL));
        v.push(f.cube([fx + 0.03 * s, y * (mw - 0.055 * s), 0.5 * (0.08 + top)], [0.024 * s, 0.018 * s, 0.5 * (top - 0.08)], TINE));
    }
    for z in [0.10 * top, 0.55 * top, top - 0.02 * s] {
        v.push(f.cube([fx + 0.03 * s, 0.0, z], [0.026 * s, mw, 0.022 * s], STEEL));
    }
    v.push(f.cylinder([fx + 0.03 * s, 0.0, 0.08 * top], [fx + 0.03 * s, 0.0, 0.82 * top], 0.034 * s, [176, 180, 188]));
    // Carriage just ahead of the mast, with the bars the tines hang on.
    let cw = d.carriage_half_w_m;
    let cx = hx - 0.012 * s;
    for z in [h + 0.09 * s, h + 0.41 * s] {
        v.push(f.cube([cx, 0.0, z], [0.012 * s, cw, 0.032 * s], TRUCK_DARK));
    }
    for y in [-0.30, 0.30] {
        v.push(f.cube([cx, y * cw / 0.43, h + 0.25 * s], [0.012 * s, 0.03 * s, 0.17 * s], TRUCK_DARK));
    }
    // The load backrest, a grid above the carriage.
    let bars = 5;
    for i in 0..bars {
        let y = cw * (2.0 * i as f32 / (bars - 1) as f32 - 1.0);
        v.push(f.cube([cx, y, h + 0.75 * s], [0.008 * s, 0.012 * s, 0.33 * s], TRUCK_DARK));
    }
    v.push(f.cube([cx, 0.0, h + 1.07 * s], [0.008 * s, cw, 0.016 * s], TRUCK_DARK));
    let (len, tw, tt) = (d.tine_len_m, d.tine_w_m, d.tine_t_m);
    for y in [-0.5 * d.tine_spread_m, 0.5 * d.tine_spread_m] {
        v.push(f.cube([hx + 0.5 * len, y, h + 0.5 * tt], [0.5 * len, 0.5 * tw, 0.5 * tt], TINE));
        v.push(f.cube([hx + 0.012 * s, y, h + 0.22 * s], [0.012 * s, 0.5 * tw, 0.22 * s], TINE));
    }
    v
}

/// A design's triangles, in metres about their own centre, with a bounding volume tree for rays.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    tris: Vec<[V3; 3]>,
    nodes: Vec<MeshNode>,
    /// Half the extent along each axis.
    pub half: V3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MeshNode {
    lo: V3,
    hi: V3,
    /// Children's indices, or for a leaf (`left` == u32::MAX) the run of `tris`.
    left: u32,
    right: u32,
    first: u32,
    count: u32,
}

fn tri_bounds(t: &[V3; 3]) -> (V3, V3) {
    let mut lo = t[0];
    let mut hi = t[0];
    for p in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

impl Mesh {
    /// From triangles in any place: they are moved to centre on their bounds, which `offset` then gives back.
    pub fn new(mut tris: Vec<[V3; 3]>) -> (Mesh, V3) {
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for t in &tris {
            let (a, b) = tri_bounds(t);
            for k in 0..3 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(b[k]);
            }
        }
        if tris.is_empty() {
            return (Mesh { tris, nodes: Vec::new(), half: [0.0; 3] }, [0.0; 3]);
        }
        let c = scale(add(lo, hi), 0.5);
        for t in &mut tris {
            for p in t.iter_mut() {
                *p = sub(*p, c);
            }
        }
        let mut m = Mesh { nodes: Vec::with_capacity(tris.len()), half: scale(sub(hi, lo), 0.5), tris };
        let n = m.tris.len() as u32;
        m.build(0, n);
        (m, c)
    }

    fn build(&mut self, first: u32, count: u32) -> u32 {
        let run = first as usize..(first + count) as usize;
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for t in &self.tris[run.clone()] {
            let (a, b) = tri_bounds(t);
            for k in 0..3 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(b[k]);
            }
        }
        let at = self.nodes.len();
        self.nodes.push(MeshNode { lo, hi, left: u32::MAX, right: 0, first, count });
        if count <= 6 {
            return at as u32;
        }
        let d = sub(hi, lo);
        let axis = if d[0] >= d[1] && d[0] >= d[2] { 0 } else if d[1] >= d[2] { 1 } else { 2 };
        self.tris[run].sort_by(|a, b| (a[0][axis] + a[1][axis] + a[2][axis]).total_cmp(&(b[0][axis] + b[1][axis] + b[2][axis])));
        let half = count / 2;
        let left = self.build(first, half);
        let right = self.build(first + half, count - half);
        self.nodes[at].left = left;
        self.nodes[at].right = right;
        at as u32
    }

    /// The first triangle a ray meets, from either side: the distance and the normal facing the ray.
    pub fn hit(&self, o: V3, d: V3) -> Option<(f32, V3)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let mut stack = [0u32; 48];
        let mut top = 1;
        let mut best: Option<(f32, V3)> = None;
        while top > 0 {
            top -= 1;
            let n = &self.nodes[stack[top] as usize];
            let (mut t0, mut t1) = (0.0f32, best.map_or(f32::INFINITY, |b| b.0));
            for k in 0..3 {
                let (a, b) = ((n.lo[k] - o[k]) * inv[k], (n.hi[k] - o[k]) * inv[k]);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
            }
            if t0 > t1 {
                continue;
            }
            if n.left != u32::MAX {
                if top + 2 <= stack.len() {
                    stack[top] = n.left;
                    stack[top + 1] = n.right;
                    top += 2;
                }
                continue;
            }
            for t in &self.tris[n.first as usize..(n.first + n.count) as usize] {
                if let Some(h) = hit_tri(t, o, d) {
                    if best.map_or(true, |b| h.0 < b.0) {
                        best = Some(h);
                    }
                }
            }
        }
        best
    }
}

/// Moller-Trumbore ray and triangle.
fn hit_tri(t: &[V3; 3], o: V3, d: V3) -> Option<(f32, V3)> {
    let dot = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, t[0]);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let dist = dot(e2, q) * inv;
    if dist <= 0.0 {
        return None;
    }
    let n = cross(e1, e2);
    let len = norm(n).max(1e-12);
    let n = scale(n, if dot(n, d) > 0.0 { -1.0 / len } else { 1.0 / len });
    Some((dist, n))
}

fn meshes() -> &'static RwLock<Vec<&'static Mesh>> {
    static MESHES: OnceLock<RwLock<Vec<&'static Mesh>>> = OnceLock::new();
    MESHES.get_or_init(Default::default)
}

/// Keeps a mesh for the life of the program (a handful of parts per imported design) and gives its id.
pub fn add_mesh(m: Mesh) -> u32 {
    let mut v = meshes().write().expect("mesh list");
    v.push(Box::leak(Box::new(m)));
    (v.len() - 1) as u32
}

pub fn mesh(id: u32) -> Option<&'static Mesh> {
    meshes().read().ok()?.get(id as usize).copied()
}

/// A body as data, measured from a design. A rig keeps the proportions and masses of what was drawn; the moves are the
/// same models as the built-in robots', on these lengths, so an edited arm reaches as far as it is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct RigSpec {
    pub name: String,
    pub form: RigForm,
    pub dims: Dims,
    pub masses: RigMasses,
    pub limits: RigLimits,
    pub joints: Vec<RigJoint>,
    /// Link, its triangles in the design at rest.
    pub meshes: Vec<(String, Vec<[V3; 3]>)>,
    /// Link, the colour it was given in the design.
    pub colors: Vec<(String, [u8; 3])>,
    /// What a body built from this rig draws (`attach`).
    pub visuals: Vec<RigVisual>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RigForm {
    Humanoid,
    Forklift,
}

impl RigForm {
    pub fn base(self) -> BaseKind {
        match self {
            RigForm::Humanoid => BaseKind::HumanFoot,
            RigForm::Forklift => BaseKind::Wheel,
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            RigForm::Humanoid => "휴머노이드",
            RigForm::Forklift => "지게차",
        }
    }
}

/// The design's masses by the part of the body they replace; `None` keeps the part's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RigMasses {
    pub head: Option<f32>,
    pub chest: Option<f32>,
    pub belly: Option<f32>,
    pub arm: Option<f32>,
    pub hand: Option<f32>,
    pub leg: Option<f32>,
    pub foot: Option<f32>,
    pub foot_len_m: Option<f32>,
    pub foot_width_m: Option<f32>,
    pub chassis: Option<f32>,
    /// Centres of mass: up from the waist, down from the waist, up from the neck, out from the shoulder, down from the
    /// hip, up from the ground.
    pub chest_com_up_m: Option<f32>,
    pub belly_com_down_m: Option<f32>,
    pub head_com_up_m: Option<f32>,
    pub arm_com_m: Option<f32>,
    pub leg_com_down_m: Option<f32>,
    pub chassis_com_m: Option<f32>,
}

/// Joint limits as drawn (degrees), where the design has them; the rest stay a person's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RigLimits {
    pub knee_max: Option<f32>,
    pub ankle_dorsi_max: Option<f32>,
    pub hip_flex_max: Option<f32>,
    pub waist_max: Option<f32>,
    pub neck_max: Option<f32>,
    pub shoulder: Option<[f32; 2]>,
    pub elbow_max: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RigJoint {
    pub name: String,
    pub kind: String,
    pub parent: String,
    pub child: String,
    /// Where it is in the design.
    pub at: V3,
    pub limit: Option<[f32; 2]>,
}

/// A mesh of a link, in the frame of the link's physics box at rest, or for a truck in the body frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RigVisual {
    pub link: String,
    pub mesh: u32,
    pub offset: V3,
    pub half: V3,
    pub color: [u8; 3],
    /// Rides on the carriage: up with the tines from `Dims::fork_rest_m`.
    pub on_carriage: bool,
}

fn dist(a: V3, b: V3) -> f32 {
    norm(sub(a, b))
}

fn bounds_of(tris: &[[V3; 3]]) -> Option<(V3, V3)> {
    let mut any = false;
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for t in tris {
        for p in t {
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    any.then_some((lo, hi))
}

impl RigSpec {
    /// Measures a design: its form from the links it has, its lengths from where the joints are, its masses from the
    /// links. Names are the ones the editor starts from (`design_of`): a design that keeps them keeps its meaning, and
    /// what is new in it rides on the link it was attached to.
    pub fn from_urdf(u: &urdf::Urdf, name: &str, files: &HashMap<String, Vec<u8>>) -> Result<RigSpec, String> {
        let frames = u.world_frames();
        let mut warnings = Vec::new();
        let at = |joint: &str| -> Option<V3> { u.joints.iter().find(|j| j.name == joint).and_then(|j| frames.get(&j.child)).map(|f| f.t) };
        let link_mass = |n: &str| u.links.iter().find(|l| l.name == n).map(|l| l.mass_kg).filter(|m| *m > 1e-4);
        let com_of = |n: &str| u.links.iter().find(|l| l.name == n).and_then(|l| Some(frames.get(n)?.at(l.com)));
        let span_deg = |joint: &str| u.joints.iter().find(|j| j.name == joint).and_then(|j| j.limit).map(|l| [l[0].to_degrees(), l[1].to_degrees()]);
        let mut meshes: Vec<(String, Vec<[V3; 3]>)> = Vec::new();
        for l in &u.links {
            let Some(frame) = frames.get(&l.name) else { continue };
            let mut tris = Vec::new();
            for v in &l.visuals {
                let file = v.file.rsplit('/').next().unwrap_or(&v.file);
                let Some(bytes) = files.get(file) else {
                    warnings.push(format!("{}: 메시 {file}가 없어 기본 모양으로 그림", l.name));
                    continue;
                };
                let t = frame.then(&v.origin);
                tris.extend(urdf::parse_stl(bytes)?.into_iter().map(|tri| tri.map(|p| t.at(p))));
            }
            if !tris.is_empty() {
                meshes.push((l.name.clone(), tris));
            }
        }
        let box_of = |n: &str| meshes.iter().find(|(m, _)| m == n).and_then(|(_, t)| bounds_of(t));
        let has = |n: &str| u.links.iter().any(|l| l.name == n);
        let form = if has("thigh_r") || has("thigh_l") {
            RigForm::Humanoid
        } else if has("carriage") || has("tine_r") || has("tine_l") {
            RigForm::Forklift
        } else {
            return Err("휴머노이드(thigh_r 링크)나 지게차(carriage 링크)로 알아볼 수 없습니다: 기본 프리셋에서 시작한 부품 이름을 유지해 주세요".into());
        };
        let mut masses = RigMasses::default();
        let mut limits = RigLimits::default();
        let dims = match form {
            RigForm::Humanoid => {
                let need = |j: &str| at(j).ok_or_else(|| format!("관절 {j}가 없습니다"));
                let (hip, knee, ankle) = (need("hip_pitch_r")?, need("knee_r")?, need("ankle_r")?);
                let (waist, neck) = (need("waist_pitch")?, need("neck")?);
                let (shoulder, elbow, wrist) = (need("shoulder_r")?, need("elbow_r")?, need("wrist_r")?);
                let top = box_of("head").map_or(neck[2] + 0.19, |(_, hi)| hi[2]);
                let k = top / crate::dims::HEAD_TOP_COMPACT_M;
                let mut d = Dims::person(top);
                d.k = k;
                d.hip_m = hip[2];
                d.hip_half_m = hip[1].abs();
                d.ankle_m = ankle[2];
                d.stance_m = 2.0 * ankle[1].abs();
                d.waist_m = waist[2];
                d.shoulder_m = shoulder[2];
                d.shoulder_half_m = shoulder[1].abs();
                d.upper_arm_m = dist(shoulder, elbow);
                d.forearm_m = dist(elbow, wrist);
                d.neck_up_m = neck[2] - shoulder[2];
                if let Some((lo, hi)) = box_of("camera_head") {
                    d.camera_up_m = 0.5 * (lo[2] + hi[2]) - neck[2];
                    d.camera_fwd_m = 0.5 * (lo[0] + hi[0]) - neck[0];
                } else {
                    warnings.push("camera_head 링크가 없어 눈 위치는 키에서 계산함".into());
                }
                if (dist(hip, knee) - dist(knee, ankle)).abs() > 0.02 {
                    warnings.push(format!("허벅지 {:.2} m와 정강이 {:.2} m가 달라 평균으로 계산함", dist(hip, knee), dist(knee, ankle)));
                }
                let both = |a: &str, b: &str| Some(link_mass(a)? + link_mass(b)?);
                masses.head = link_mass("head");
                masses.chest = link_mass("trunk");
                masses.belly = link_mass("pelvis");
                masses.arm = both("upper_arm_r", "forearm_r");
                masses.hand = link_mass("hand_r");
                masses.leg = both("thigh_r", "shank_r");
                masses.foot = link_mass("foot_r");
                if let Some((lo, hi)) = box_of("foot_r") {
                    masses.foot_len_m = Some(hi[0] - lo[0]);
                    masses.foot_width_m = Some(hi[1] - lo[1]);
                }
                masses.chest_com_up_m = com_of("trunk").map(|c| c[2] - waist[2]);
                masses.belly_com_down_m = com_of("pelvis").map(|c| waist[2] - c[2]);
                masses.head_com_up_m = com_of("head").map(|c| c[2] - neck[2]);
                let along = |a: &str, b: &str, from: f32| -> Option<f32> {
                    let (ma, mb) = (link_mass(a)?, link_mass(b)?);
                    let (ca, cb) = (com_of(a)?, com_of(b)?);
                    Some((ma * (from - ca[2]) + mb * (from - cb[2])) / (ma + mb))
                };
                masses.arm_com_m = along("upper_arm_r", "forearm_r", shoulder[2]);
                masses.leg_com_down_m = along("thigh_r", "shank_r", hip[2]);
                let top = |j: &str| span_deg(j).map(|s| s[0].abs().max(s[1].abs()));
                limits = RigLimits {
                    knee_max: top("knee_r"),
                    ankle_dorsi_max: span_deg("ankle_r").map(|s| s[1]),
                    hip_flex_max: span_deg("hip_pitch_r").map(|s| s[1]),
                    waist_max: span_deg("waist_pitch").map(|s| s[1]),
                    neck_max: top("neck"),
                    shoulder: span_deg("shoulder_r"),
                    elbow_max: top("elbow_r"),
                };
                d
            }
            RigForm::Forklift => {
                let person = Dims::person(crate::dims::STANDARD_STATURE_M);
                let mut d = Dims::lift_truck(&person, BaseKind::Wheel);
                let wheels: Vec<V3> = u.joints.iter().filter(|j| j.name.starts_with("wheel")).filter_map(|j| at(&j.name)).collect();
                if wheels.len() >= 2 {
                    let (lo, hi) = wheels.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p[0]), b.max(p[0])));
                    if hi - lo > 0.05 {
                        d.wheelbase_m = Some(hi - lo);
                    }
                    d.wheel_r_m = wheels.iter().map(|p| p[2]).fold(0.0, f32::max);
                    let side = wheels.iter().map(|p| p[1].abs()).fold(0.0, f32::max);
                    if let Some((lo, hi)) = u.links.iter().find(|l| l.name.starts_with("wheel")).and_then(|l| box_of(&l.name)) {
                        d.belt_w_m = hi[1] - lo[1];
                        d.wheel_r_m = 0.5 * (hi[2] - lo[2]);
                    }
                    d.chassis_w_m = 2.0 * side + d.belt_w_m;
                } else {
                    warnings.push("wheel 관절을 찾지 못해 축거·바퀴는 기본값을 씀".into());
                }
                if let Some((lo, hi)) = box_of("tine_r") {
                    d.tine_len_m = hi[0] - lo[0];
                    d.tine_w_m = hi[1] - lo[1];
                    d.tine_t_m = (hi[2] - lo[2]).min(d.tine_t_m.max(0.02));
                    d.tine_spread_m = (lo[1] + hi[1]).abs();
                }
                if let Some((_, hi)) = box_of("mast") {
                    d.mast_top_m = hi[2];
                }
                if let Some((_, hi)) = box_of("chassis") {
                    d.roof_m = Some(hi[2]);
                }
                if let Some((lo, hi)) = box_of("seat") {
                    let up = hi[2] + 0.075 * d.k - d.hip_m;
                    d.hip_m += up;
                    d.waist_m += up;
                    d.shoulder_m += up;
                    d.seat_x_m = 0.5 * (lo[0] + hi[0]);
                }
                let parts: Vec<(f32, f32)> = u.links.iter().filter(|l| l.name != "base_link").filter_map(|l| Some((l.mass_kg, com_of(&l.name)?[2]))).collect();
                let total: f32 = parts.iter().map(|p| p.0).sum();
                masses.chassis = (total > 1.0).then_some(total);
                masses.chassis_com_m = (total > 1.0).then(|| parts.iter().map(|p| p.0 * p.1).sum::<f32>() / total);
                d
            }
        };
        let joints = u.joints.iter().filter(|j| j.name != "chassis_fixed" && !j.name.ends_with("_fixed")).map(|j| RigJoint { name: j.name.clone(), kind: j.kind.clone(), parent: j.parent.clone(), child: j.child.clone(), at: frames.get(&j.child).map_or([0.0; 3], |f| f.t), limit: j.limit }).collect();
        let colors = u.links.iter().filter_map(|l| Some((l.name.clone(), l.color?))).collect();
        Ok(RigSpec { name: name.to_string(), form, dims, masses, limits, joints, meshes, colors, visuals: Vec::new(), warnings })
    }

    /// Degrees of freedom: the joints that move.
    pub fn dof(&self) -> usize {
        self.joints.iter().filter(|j| j.kind != "fixed").count()
    }
}

fn registry() -> &'static RwLock<HashMap<u32, Arc<RigSpec>>> {
    static RIGS: OnceLock<RwLock<HashMap<u32, Arc<RigSpec>>>> = OnceLock::new();
    RIGS.get_or_init(Default::default)
}

/// A rig registered for bodies to build from (`Build::Rig`).
pub fn get(id: u32) -> Option<Arc<RigSpec>> {
    registry().read().ok()?.get(&id).cloned()
}

/// Registers a rig and works out what its body draws: each link's mesh in the frame of the link's physics box, at the
/// rest pose the design is in.
pub fn register(rig: RigSpec) -> u32 {
    let form = rig.form;
    let id = {
        let mut r = registry().write().expect("rig list");
        let id = r.keys().max().map_or(1, |k| k + 1);
        r.insert(id, Arc::new(rig));
        id
    };
    let plan = crate::parts::BodyPlan { build: crate::dims::Build::Rig(id), ..crate::parts::BodyPlan::standard() };
    let body = Body::new(&plan, form.base(), crate::hand::FingerCount::Five);
    if let Some(rig) = get(id) {
        let visuals = attach(&rig, &body);
        let mut r = registry().write().expect("rig list");
        if let Some(slot) = r.get_mut(&id) {
            let mut full = (**slot).clone();
            full.visuals = visuals;
            *slot = Arc::new(full);
        }
    }
    id
}

pub fn remove(id: u32) {
    if let Ok(mut r) = registry().write() {
        r.remove(&id);
    }
}

fn attach(rig: &RigSpec, body: &Body) -> Vec<RigVisual> {
    let color_of = |link: &str| {
        let drawn = rig.colors.iter().find(|(n, _)| n == link).map(|(_, c)| *c);
        drawn.unwrap_or(match rig.form {
            RigForm::Forklift if link.starts_with("wheel") => TYRE,
            RigForm::Forklift if link.starts_with("tine") || link == "carriage" => TINE,
            RigForm::Forklift if link == "mast" => STEEL,
            RigForm::Forklift => TRUCK_YELLOW,
            RigForm::Humanoid if link.contains("hand") || link.starts_with("foot") => JOINT,
            RigForm::Humanoid => SHELL,
        })
    };
    let mut out = Vec::new();
    match rig.form {
        RigForm::Forklift => {
            for (link, tris) in &rig.meshes {
                let (mesh, offset) = Mesh::new(tris.clone());
                let half = mesh.half;
                let on_carriage = link == "carriage" || link.starts_with("tine") || link.starts_with("camera_fork");
                out.push(RigVisual { link: link.clone(), mesh: add_mesh(mesh), offset, half, color: color_of(link), on_carriage });
            }
        }
        RigForm::Humanoid => {
            // The design stands with its arms straight down.
            let rest = KinPose { elbow_deg: [0.0, 0.0], ..KinPose::stand(body, 0.0) };
            let boxes = kin::solve(body, &rest, &KinLoads::default()).boxes;
            let frame_of = |name: &str| -> Option<urdf::Tf> {
                let box_name = match name {
                    "hand_r" => "palm_r",
                    "hand_l" => "palm_l",
                    "camera_head" => "head",
                    n => n,
                };
                boxes.iter().find(|b| b.name == box_name).map(|b| urdf::Tf { r: [[b.axes[0][0], b.axes[1][0], b.axes[2][0]], [b.axes[0][1], b.axes[1][1], b.axes[2][1]], [b.axes[0][2], b.axes[1][2], b.axes[2][2]]], t: b.center })
            };
            let parent_of = |n: &str| rig.joints.iter().find(|j| j.child == n).map(|j| j.parent.clone());
            for (link, tris) in &rig.meshes {
                // What was added in the design rides on the link it was attached to.
                let mut carrier = link.clone();
                let f = loop {
                    if let Some(f) = frame_of(&carrier) {
                        break Some(f);
                    }
                    match parent_of(&carrier) {
                        Some(p) => carrier = p,
                        None => break None,
                    }
                };
                let Some(f) = f else { continue };
                let local: Vec<[V3; 3]> = tris.iter().map(|t| t.map(|p| f.inv_at(p))).collect();
                let (mesh, offset) = Mesh::new(local);
                let half = mesh.half;
                let drawn_on = if carrier == "camera_head" { "head".to_string() } else { carrier };
                out.push(RigVisual { link: drawn_on, mesh: add_mesh(mesh), offset, half, color: color_of(link), on_carriage: false });
            }
        }
    }
    out
}

impl Body {
    /// The rig this body was built from.
    pub fn rig(&self) -> Option<Arc<RigSpec>> {
        match self.plan.build {
            crate::dims::Build::Rig(id) => get(id),
            _ => None,
        }
    }
}

/// A solid of the design the editor starts from, in metres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum DesignSolid {
    Box { center: V3, size: V3 },
    Cylinder { base: V3, axis: V3, height: f32, radius: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DesignComponent {
    pub name: String,
    pub mass_kg: f32,
    pub color: [u8; 3],
    pub solids: Vec<DesignSolid>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DesignJoint {
    pub name: String,
    /// `revolute`, `slider` or `rigid`.
    pub kind: &'static str,
    pub a: String,
    pub b: String,
    pub at: V3,
    pub axis: V3,
    /// Degrees or metres.
    pub limits: Option<[f32; 2]>,
}

/// A robot as a starting design for the editor: its links as components with solids at their rest places, and its
/// joints. Component and joint names are how `RigSpec::from_urdf` finds them again.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RigDesign {
    pub name: String,
    pub form: RigForm,
    pub components: Vec<DesignComponent>,
    pub joints: Vec<DesignJoint>,
}

pub fn design_of(body: &Body) -> RigDesign {
    if crate::act::fork::applies(body) {
        truck_design(body)
    } else {
        humanoid_design(body)
    }
}

fn humanoid_design(body: &Body) -> RigDesign {
    let d = &body.dims;
    let k = d.k;
    let bx = |c: V3, s: V3| DesignSolid::Box { center: c, size: s };
    let down = [0.0, 0.0, -1.0];
    let (hip, waist, sh) = (d.hip_m, d.waist_m, d.shoulder_m);
    let neck = d.column_top_m();
    let (m_up, _, m_fore) = body.arm_split();
    let share = kin::thigh_share(body);
    let mut c = vec![
        DesignComponent { name: "pelvis".into(), mass_kg: body.belly.mass_kg, color: TRIM, solids: vec![bx([0.0, 0.0, 0.5 * (hip + waist)], [0.18 * k, 0.28 * k, waist - hip + 0.04 * k])] },
        DesignComponent {
            name: "trunk".into(),
            mass_kg: body.chest.mass_kg + body.chest.battery_kg,
            color: SHELL,
            solids: vec![bx([0.0, 0.0, 0.5 * (waist + sh) + 0.01 * k], [0.20 * k, 0.32 * k, sh - waist + 0.02 * k]), DesignSolid::Cylinder { base: [0.0, -d.shoulder_half_m, sh - 0.04 * k], axis: [0.0, 1.0, 0.0], height: 2.0 * d.shoulder_half_m, radius: 0.05 * k }],
        },
        DesignComponent { name: "head".into(), mass_kg: body.head.mass_kg, color: SHELL, solids: vec![bx([0.0, 0.0, neck + 0.07 * k], [0.15 * k, 0.15 * k, 0.14 * k]), DesignSolid::Cylinder { base: [0.0, 0.0, neck - 0.05 * k], axis: [0.0, 0.0, 1.0], height: 0.06 * k, radius: 0.045 * k }] },
        DesignComponent { name: "camera_head".into(), mass_kg: 0.05, color: VISOR, solids: vec![bx([d.camera_fwd_m, 0.0, neck + d.camera_up_m], [0.025 * k, 0.09 * k, 0.025 * k])] },
    ];
    let mut joints = vec![
        DesignJoint { name: "waist_pitch".into(), kind: "revolute", a: "pelvis".into(), b: "trunk".into(), at: [0.0, 0.0, waist], axis: [0.0, 1.0, 0.0], limits: Some([-10.0, kin::WAIST_PITCH_MAX_DEG]) },
        DesignJoint { name: "neck".into(), kind: "revolute", a: "trunk".into(), b: "head".into(), at: [0.0, 0.0, neck], axis: [0.0, 1.0, 0.0], limits: Some([-kin::NECK_PITCH_MAX_DEG, kin::NECK_PITCH_MAX_DEG]) },
        DesignJoint { name: "camera_mount".into(), kind: "rigid", a: "head".into(), b: "camera_head".into(), at: [d.camera_fwd_m, 0.0, neck + d.camera_up_m], axis: [0.0, 0.0, 1.0], limits: None },
    ];
    for (tag, y) in [("r", -1.0f32), ("l", 1.0)] {
        let n = |s: &str| format!("{s}_{tag}");
        let (ys, yh) = (y * d.shoulder_half_m, y * d.hip_half_m);
        let (elbow, wrist) = (sh - d.upper_arm_m, sh - d.upper_arm_m - d.forearm_m);
        let knee = hip - d.thigh_m();
        let fl = body.leg.foot_len_m;
        c.push(DesignComponent { name: n("upper_arm"), mass_kg: m_up, color: SHELL, solids: vec![DesignSolid::Cylinder { base: [0.0, ys, sh], axis: down, height: d.upper_arm_m, radius: 0.042 * k }] });
        c.push(DesignComponent { name: n("forearm"), mass_kg: m_fore, color: SHELL, solids: vec![DesignSolid::Cylinder { base: [0.0, ys, elbow], axis: down, height: d.forearm_m, radius: 0.036 * k }] });
        c.push(DesignComponent { name: n("hand"), mass_kg: body.hand_kg, color: JOINT, solids: vec![bx([0.0, ys, wrist - d.palm_m], [0.05 * k, 0.09 * k, 2.0 * d.palm_m])] });
        c.push(DesignComponent { name: n("thigh"), mass_kg: body.leg.mass_kg * share, color: SHELL, solids: vec![DesignSolid::Cylinder { base: [0.0, yh, hip], axis: down, height: d.thigh_m(), radius: 0.072 * k }] });
        c.push(DesignComponent { name: n("shank"), mass_kg: body.leg.mass_kg * (1.0 - share), color: SHELL, solids: vec![DesignSolid::Cylinder { base: [0.0, yh, knee], axis: down, height: d.shank_m(), radius: 0.055 * k }] });
        c.push(DesignComponent { name: n("foot"), mass_kg: body.leg.foot_mass_kg, color: JOINT, solids: vec![bx([0.25 * fl, yh, 0.5 * d.ankle_m], [fl, body.leg.foot_width_m, d.ankle_m])] });
        joints.push(DesignJoint { name: n("shoulder"), kind: "revolute", a: "trunk".into(), b: n("upper_arm"), at: [0.0, ys, sh], axis: [0.0, -1.0, 0.0], limits: Some([-60.0, 180.0]) });
        joints.push(DesignJoint { name: n("elbow"), kind: "revolute", a: n("upper_arm"), b: n("forearm"), at: [0.0, ys, elbow], axis: [0.0, -1.0, 0.0], limits: Some([0.0, 150.0]) });
        joints.push(DesignJoint { name: n("wrist"), kind: "rigid", a: n("forearm"), b: n("hand"), at: [0.0, ys, wrist], axis: [0.0, 0.0, 1.0], limits: None });
        joints.push(DesignJoint { name: n("hip_pitch"), kind: "revolute", a: "pelvis".into(), b: n("thigh"), at: [0.0, yh, hip], axis: [0.0, -1.0, 0.0], limits: Some([-30.0, kin::HIP_FLEX_MAX_DEG]) });
        joints.push(DesignJoint { name: n("knee"), kind: "revolute", a: n("thigh"), b: n("shank"), at: [0.0, yh, knee], axis: [0.0, 1.0, 0.0], limits: Some([0.0, kin::KNEE_FLEX_MAX_DEG]) });
        joints.push(DesignJoint { name: n("ankle"), kind: "revolute", a: n("shank"), b: n("foot"), at: [0.0, yh, d.ankle_m], axis: [0.0, -1.0, 0.0], limits: Some([-30.0, kin::ANKLE_DORSI_MAX_DEG]) });
    }
    RigDesign { name: "휴머노이드".into(), form: RigForm::Humanoid, components: c, joints }
}

fn truck_design(body: &Body) -> RigDesign {
    let d = &body.dims;
    let k = d.k;
    let (l, w, r, tw) = (body.support_len_m, body.support_width_m, d.wheel_r_m, d.belt_w_m);
    let bx = |lo: V3, hi: V3| DesignSolid::Box { center: scale(add(lo, hi), 0.5), size: sub(hi, lo) };
    let hood = d.hip_m - 0.17 * k;
    let floor = floor_z(body);
    let rear = -0.5 * l - 1.15 * r;
    let nose = 0.5 * l + 0.40 * r;
    let side = 0.5 * w;
    let inner = (side - tw - 0.015).max(0.05);
    let roof = d.roof_m.unwrap_or(d.column_top_m() + 0.28 * k);
    let cowl = hood + 0.24 * k;
    let (sx, sz) = (d.seat_x_m, d.hip_m - 0.075 * k);
    let py = side - 0.04;
    let post = |x0: f32, x1: f32, y: f32, z0: f32, z1: f32| bx([x0.min(x1), y - 0.026, z0], [x0.max(x1) + 0.05, y + 0.026, z1]);
    let mut solids = vec![
        bx([rear, -inner, 0.45 * r], [nose, inner, floor]),
        bx([rear, -side, (2.0 * r + 0.02).max(floor)], [sx + 0.24 * k, side, hood]),
        bx([rear, -inner, floor], [sx + 0.24 * k, inner, (2.0 * r + 0.02).max(floor)]),
        bx([nose - 0.17 * k, -side + 0.03, floor], [nose, side - 0.03, cowl]),
        bx([rear + 0.05, -side + 0.02, roof - 0.04], [nose, side - 0.02, roof]),
    ];
    for y in [-py, py] {
        solids.push(post(rear + 0.05, rear + 0.05, y, hood, roof - 0.04));
        solids.push(post(nose - 0.12, nose - 0.12, y, cowl, roof - 0.04));
    }
    let fx = crate::act::fork::front_x(body);
    let hx = crate::act::fork::heel_x(body);
    let h = d.fork_rest_m;
    let s = d.mast_half_w_m / 0.30;
    let mast = vec![bx([fx, -d.mast_half_w_m - 0.024 * s, 0.04], [fx + 0.06 * s, -d.mast_half_w_m + 0.024 * s, d.mast_top_m]), bx([fx, d.mast_half_w_m - 0.024 * s, 0.04], [fx + 0.06 * s, d.mast_half_w_m + 0.024 * s, d.mast_top_m]), bx([fx, -d.mast_half_w_m, d.mast_top_m - 0.06 * s], [fx + 0.06 * s, d.mast_half_w_m, d.mast_top_m]), bx([fx, -d.mast_half_w_m, 0.20], [fx + 0.06 * s, d.mast_half_w_m, 0.26 * s + 0.20])];
    let cw = d.carriage_half_w_m;
    let carriage = vec![bx([hx - 0.024 * s, -cw, h + 0.06 * s], [hx, cw, h + 0.12 * s]), bx([hx - 0.024 * s, -cw, h + 0.38 * s], [hx, cw, h + 0.44 * s]), bx([hx - 0.02 * s, -0.04, h + 0.12 * s], [hx, 0.04, h + 0.38 * s])];
    let tine = |y: f32| vec![bx([hx, y - 0.5 * d.tine_w_m, h], [hx + d.tine_len_m, y + 0.5 * d.tine_w_m, h + d.tine_t_m])];
    let mut c = vec![
        DesignComponent { name: "chassis".into(), mass_kg: (body.chassis.mass_kg - 480.0).max(0.6 * body.chassis.mass_kg), color: TRUCK_YELLOW, solids },
        DesignComponent { name: "seat".into(), mass_kg: 20.0_f32.min(0.02 * body.chassis.mass_kg), color: TRUCK_DARK, solids: vec![bx([sx - 0.15 * k, -0.17 * k, hood], [sx + 0.13 * k, 0.17 * k, sz])] },
        DesignComponent { name: "mast".into(), mass_kg: 180.0_f32.min(0.08 * body.chassis.mass_kg), color: STEEL, solids: mast },
        DesignComponent { name: "carriage".into(), mass_kg: 60.0_f32.min(0.03 * body.chassis.mass_kg), color: TRUCK_DARK, solids: carriage },
        DesignComponent { name: "tine_r".into(), mass_kg: 40.0_f32.min(0.02 * body.chassis.mass_kg), color: TINE, solids: tine(-0.5 * d.tine_spread_m) },
        DesignComponent { name: "tine_l".into(), mass_kg: 40.0_f32.min(0.02 * body.chassis.mass_kg), color: TINE, solids: tine(0.5 * d.tine_spread_m) },
        DesignComponent { name: "camera_fork".into(), mass_kg: 0.05, color: VISOR, solids: vec![bx([hx - 0.03 * s, -0.04 * s, h + d.tine_t_m + d.fork_camera_up_m - 0.012 * s], [hx - 0.005 * s, 0.04 * s, h + d.tine_t_m + d.fork_camera_up_m + 0.012 * s])] },
    ];
    let mut joints = vec![
        DesignJoint { name: "mast_mount".into(), kind: "rigid", a: "chassis".into(), b: "mast".into(), at: [fx, 0.0, 0.5], axis: [0.0, 0.0, 1.0], limits: None },
        DesignJoint { name: "seat_mount".into(), kind: "rigid", a: "chassis".into(), b: "seat".into(), at: [sx, 0.0, hood], axis: [0.0, 0.0, 1.0], limits: None },
        DesignJoint { name: "fork_lift".into(), kind: "slider", a: "mast".into(), b: "carriage".into(), at: [hx, 0.0, h], axis: [0.0, 0.0, 1.0], limits: Some([-h, d.fork_max_m() - h]) },
        DesignJoint { name: "tine_mount_r".into(), kind: "rigid", a: "carriage".into(), b: "tine_r".into(), at: [hx, -0.5 * d.tine_spread_m, h], axis: [0.0, 0.0, 1.0], limits: None },
        DesignJoint { name: "tine_mount_l".into(), kind: "rigid", a: "carriage".into(), b: "tine_l".into(), at: [hx, 0.5 * d.tine_spread_m, h], axis: [0.0, 0.0, 1.0], limits: None },
        DesignJoint { name: "camera_mount".into(), kind: "rigid", a: "carriage".into(), b: "camera_fork".into(), at: [hx, 0.0, h + d.tine_t_m + d.fork_camera_up_m], axis: [0.0, 0.0, 1.0], limits: None },
    ];
    for (tag, x, y) in [("fr", 0.5 * l, -1.0f32), ("fl", 0.5 * l, 1.0), ("rr", -0.5 * l, -1.0), ("rl", -0.5 * l, 1.0)] {
        let yc = y * (side - 0.5 * tw);
        let name = format!("wheel_{tag}");
        c.push(DesignComponent { name: name.clone(), mass_kg: 30.0_f32.min(0.01 * body.chassis.mass_kg), color: TYRE, solids: vec![DesignSolid::Cylinder { base: [x, yc - 0.5 * tw, r], axis: [0.0, 1.0, 0.0], height: tw, radius: r }] });
        joints.push(DesignJoint { name: name.clone(), kind: "revolute", a: "chassis".into(), b: name, at: [x, yc, r], axis: [0.0, 1.0, 0.0], limits: None });
    }
    RigDesign { name: "지게차".into(), form: RigForm::Forklift, components: c, joints }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dims::Build;
    use crate::hand::FingerCount;
    use crate::parts::BodyPlan;

    fn standard(base: BaseKind) -> Body {
        Body::new(&BodyPlan { build: Build::Standard, ..BodyPlan::default() }, base, FingerCount::Five)
    }

    fn bounds(v: &[VisualPart]) -> ([f32; 3], [f32; 3]) {
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in v {
            let extra = if p.solid == Solid::Capsule { [p.half[0]; 3] } else { [0.0; 3] };
            for i in 0..3 {
                let e: f32 = (0..3).map(|k| (p.axes[k][i] * p.half[k]).abs()).sum::<f32>() + extra[i];
                lo[i] = lo[i].min(p.center[i] - e);
                hi[i] = hi[i].max(p.center[i] + e);
            }
        }
        (lo, hi)
    }

    /// An STL of a box between two corners, as the exporter writes its meshes.
    fn stl_box(lo: V3, hi: V3) -> Vec<u8> {
        let c = |x: usize, y: usize, z: usize| [if x == 0 { lo[0] } else { hi[0] }, if y == 0 { lo[1] } else { hi[1] }, if z == 0 { lo[2] } else { hi[2] }];
        let quads = [[c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)], [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)], [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)], [c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)], [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)], [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)]];
        let mut out = vec![0u8; 80];
        out.extend(12u32.to_le_bytes());
        for q in quads {
            for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                out.extend([0u8; 12]);
                for p in t {
                    for v in p {
                        out.extend(v.to_le_bytes());
                    }
                }
                out.extend([0u8; 2]);
            }
        }
        out
    }

    /// The URDF the editor would export for a design, with each component's bounds as its mesh.
    fn export(design: &RigDesign) -> (String, HashMap<String, Vec<u8>>) {
        let mut files = HashMap::new();
        let mut links = String::from("<link name=\"base_link\"/>");
        for c in &design.components {
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for s in &c.solids {
                let (a, b) = match *s {
                    DesignSolid::Box { center, size } => (sub(center, scale(size, 0.5)), add(center, scale(size, 0.5))),
                    DesignSolid::Cylinder { base, axis, height, radius } => {
                        let top = add(base, scale(axis, height));
                        (std::array::from_fn(|i| base[i].min(top[i]) - radius * (1.0 - axis[i].abs())), std::array::from_fn(|i| base[i].max(top[i]) + radius * (1.0 - axis[i].abs())))
                    }
                };
                for i in 0..3 {
                    lo[i] = lo[i].min(a[i]);
                    hi[i] = hi[i].max(b[i]);
                }
            }
            files.insert(format!("r-{}.stl", c.name), stl_box(lo, hi));
            links += &format!("<link name=\"{}\"><inertial><origin xyz=\"0 0 0\"/><mass value=\"{}\"/></inertial><visual><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><mesh filename=\"r-{}.stl\"/></geometry></visual></link>", c.name, c.mass_kg, c.name);
        }
        let root = &design.components[0].name;
        let mut joints = format!("<joint name=\"{root}_fixed\" type=\"fixed\"><parent link=\"base_link\"/><child link=\"{root}\"/></joint>");
        // Every link's frame stays at the design's origin in this test export except that the joint's child frame is
        // where the joint is: visuals are then offset back.
        for j in &design.joints {
            let kind = match j.kind {
                "rigid" => "fixed",
                "slider" => "prismatic",
                k => k,
            };
            joints += &format!("<joint name=\"{}\" type=\"{kind}\"><parent link=\"base_link\"/><child link=\"{}\"/><origin xyz=\"{} {} {}\" rpy=\"0 0 0\"/><axis xyz=\"0 1 0\"/></joint>", j.name, j.b, j.at[0], j.at[1], j.at[2]);
        }
        for j in &design.joints {
            let at = j.at;
            links = links.replacen(&format!("<visual><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><mesh filename=\"r-{}.stl\"/>", j.b), &format!("<visual><origin xyz=\"{} {} {}\" rpy=\"0 0 0\"/><geometry><mesh filename=\"r-{}.stl\"/>", -at[0], -at[1], -at[2], j.b), 1);
        }
        (format!("<robot name=\"r\">{links}{joints}</robot>"), files)
    }

    #[test]
    fn a_humanoid_design_comes_back_as_the_body_it_was_made_from() {
        let b = standard(BaseKind::HumanFoot);
        let design = design_of(&b);
        let (xml, files) = export(&design);
        let rig = RigSpec::from_urdf(&urdf::parse(&xml).unwrap(), "시험", &files).unwrap();
        assert_eq!(rig.form, RigForm::Humanoid);
        let (a, r) = (&b.dims, &rig.dims);
        for (name, x, y) in [("hip", a.hip_m, r.hip_m), ("waist", a.waist_m, r.waist_m), ("shoulder", a.shoulder_m, r.shoulder_m), ("half", a.shoulder_half_m, r.shoulder_half_m), ("upper", a.upper_arm_m, r.upper_arm_m), ("fore", a.forearm_m, r.forearm_m), ("ankle", a.ankle_m, r.ankle_m), ("neck", a.neck_up_m, r.neck_up_m), ("cam up", a.camera_up_m, r.camera_up_m), ("cam fwd", a.camera_fwd_m, r.camera_fwd_m)] {
            assert!((x - y).abs() < 1e-3, "{name}: {x} vs {y}");
        }
        assert!((r.k - a.k).abs() < 0.03, "{} vs {}", r.k, a.k);
        assert!((rig.masses.head.unwrap() - b.head.mass_kg).abs() < 1e-3);
        assert!((rig.masses.leg.unwrap() - b.leg.mass_kg).abs() < 1e-3, "{:?}", rig.masses);
        assert!(rig.dof() >= 12, "{}", rig.dof());
        let id = register(rig);
        let plan = crate::parts::BodyPlan { build: Build::Rig(id), ..BodyPlan::default() };
        let c = Body::new(&plan, BaseKind::HumanFoot, FingerCount::Five);
        assert!((c.mass_kg() - b.mass_kg()).abs() < 0.08 * b.mass_kg(), "{} vs {}", c.mass_kg(), b.mass_kg());
        assert!((kin::head_camera(&c, &KinPose::stand(&c, 0.0)).pos[2] - kin::head_camera(&b, &KinPose::stand(&b, 0.0)).pos[2]).abs() < 1e-3);
        // Each link draws its mesh where its physics box is.
        let r = kin::solve(&c, &KinPose { elbow_deg: [0.0, 0.0], ..KinPose::stand(&c, 0.0) }, &KinLoads::default());
        assert!(r.feasible, "{:?}", r.limiting);
        let v = dress(&c, &r.boxes, &Palette::default());
        assert!(v.iter().filter(|p| matches!(p.solid, Solid::Mesh(_))).count() >= 14);
        let (lo, hi) = bounds(&v);
        assert!((hi[2] - 1.65).abs() < 0.05 && lo[2].abs() < 0.03, "{lo:?} {hi:?}");
        remove(id);
    }

    #[test]
    fn a_joint_limit_set_in_the_design_binds_the_physics() {
        let b = standard(BaseKind::HumanFoot);
        let (xml, files) = export(&design_of(&b));
        let xml = xml.replace("<axis xyz=\"0 1 0\"/></joint>", "<axis xyz=\"0 1 0\"/><limit lower=\"0\" upper=\"1.0\"/></joint>");
        let rig = RigSpec::from_urdf(&urdf::parse(&xml).unwrap(), "무릎 제한", &files).unwrap();
        assert!((rig.limits.knee_max.unwrap() - 57.3).abs() < 0.1, "{:?}", rig.limits);
        let id = register(rig);
        let c = Body::new(&crate::parts::BodyPlan { build: Build::Rig(id), ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        assert!((kin::Rom::of(&c).knee_max - 57.3).abs() < 0.1 && (kin::Rom::of(&b).knee_max - kin::KNEE_FLEX_MAX_DEG).abs() < 1e-3);
        remove(id);
    }

    #[test]
    fn a_longer_arm_in_the_design_is_a_longer_arm_in_the_physics() {
        let b = standard(BaseKind::HumanFoot);
        let mut design = design_of(&b);
        for j in design.joints.iter_mut().filter(|j| j.name.starts_with("elbow") || j.name.starts_with("wrist")) {
            j.at[2] -= 0.15;
        }
        let (xml, files) = export(&design);
        let rig = RigSpec::from_urdf(&urdf::parse(&xml).unwrap(), "긴 팔", &files).unwrap();
        assert!((rig.dims.upper_arm_m - b.dims.upper_arm_m - 0.15).abs() < 1e-3, "{}", rig.dims.upper_arm_m);
        let id = register(rig);
        let c = Body::new(&crate::parts::BodyPlan { build: Build::Rig(id), ..BodyPlan::default() }, BaseKind::HumanFoot, FingerCount::Five);
        let down = |x: &Body| kin::solve(x, &KinPose { elbow_deg: [0.0, 0.0], ..KinPose::stand(x, 0.0) }, &KinLoads::default()).hands[0][2];
        assert!((down(&b) - down(&c) - 0.15).abs() < 1e-3, "the hand hangs 15 cm lower: {} vs {}", down(&b), down(&c));
        remove(id);
    }

    #[test]
    fn a_truck_design_comes_back_as_a_truck() {
        let b = standard(BaseKind::Wheel);
        let (xml, files) = export(&design_of(&b));
        let rig = RigSpec::from_urdf(&urdf::parse(&xml).unwrap(), "지게차", &files).unwrap();
        assert_eq!(rig.form, RigForm::Forklift);
        let (a, r) = (&b.dims, &rig.dims);
        assert!((r.wheelbase_m.unwrap() - a.wheelbase_m.unwrap()).abs() < 1e-3, "{:?}", r.wheelbase_m);
        assert!((r.wheel_r_m - a.wheel_r_m).abs() < 1e-3 && (r.chassis_w_m - a.chassis_w_m).abs() < 1e-3, "{r:?}");
        assert!((r.tine_len_m - a.tine_len_m).abs() < 1e-3 && (r.mast_top_m - a.mast_top_m).abs() < 1e-3, "{r:?}");
        assert!((r.seat_x_m - a.seat_x_m).abs() < 0.03 && (r.hip_m - a.hip_m).abs() < 1e-3, "{} {}", r.seat_x_m, r.hip_m);
        assert!((r.roof_m.unwrap() - a.roof_m.unwrap()).abs() < 1e-3);
        let id = register(rig);
        let c = Body::new(&crate::parts::BodyPlan { build: Build::Rig(id), ..BodyPlan::default() }, BaseKind::Wheel, FingerCount::Five);
        assert!((c.mass_kg() - b.mass_kg()).abs() < 0.1 * b.mass_kg(), "{} vs {}", c.mass_kg(), b.mass_kg());
        let parts = truck_parts(&c, &Palette::default());
        assert!(!parts.is_empty() && parts.iter().all(|p| matches!(p.solid, Solid::Mesh(_))));
        let up = fork_parts(&c, 1.0);
        let down = fork_parts(&c, c.dims.fork_rest_m);
        assert!(up.iter().zip(&down).all(|(a, b)| (a.center[2] - b.center[2] - (1.0 - c.dims.fork_rest_m)).abs() < 1e-5), "the carriage and tines ride the mast");
        remove(id);
    }

    #[test]
    fn a_ray_finds_the_nearest_face_of_a_mesh() {
        let quad = |z: f32| [[[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z]], [[-1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]]];
        let tris: Vec<[V3; 3]> = (0..12).flat_map(|i| quad(i as f32 * 0.5)).collect();
        let (m, c) = Mesh::new(tris);
        assert_eq!(c, [0.0, 0.0, 2.75]);
        let (t, n) = m.hit([0.2, 0.3, -10.0], [0.0, 0.0, 1.0]).unwrap();
        assert!((t - (10.0 - 2.75)).abs() < 1e-4, "{t}");
        assert_eq!(n, [0.0, 0.0, -1.0]);
        assert!(m.hit([3.0, 0.0, -10.0], [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn a_dressed_humanoid_is_a_persons_size_and_stands_on_the_ground() {
        let b = standard(BaseKind::HumanFoot);
        let r = kin::solve(&b, &KinPose::stand(&b, 0.0), &KinLoads::default());
        let (lo, hi) = bounds(&dress(&b, &r.boxes, &Palette::default()));
        assert!((1.60..1.72).contains(&hi[2]), "{hi:?}");
        assert!(lo[2].abs() < 0.02, "on the ground: {lo:?}");
        assert!((0.40..0.75).contains(&(hi[1] - lo[1])), "shoulders and arms: {lo:?} {hi:?}");
    }

    #[test]
    fn a_dressed_truck_has_a_trucks_outline() {
        let b = standard(BaseKind::Wheel);
        let mut v = truck_parts(&b, &Palette::default());
        let (_, hi) = bounds(&v);
        assert!((hi[2] - b.dims.roof_m.unwrap()).abs() < 0.03, "the guard is the top: {hi:?}");
        v.extend(fork_parts(&b, b.dims.fork_rest_m));
        let (lo, hi) = bounds(&v);
        assert!(lo[2].abs() < 0.02, "the tyres touch the ground: {lo:?}");
        assert!((hi[1] - lo[1] - b.support_width_m).abs() < 0.06, "the tyres are the widest: {lo:?} {hi:?}");
        assert!((hi[0] - (crate::act::fork::heel_x(&b) + b.dims.tine_len_m)).abs() < 0.01, "the tines reach furthest: {hi:?}");
        assert!((hi[2] - b.dims.roof_m.unwrap()).abs() < 0.03, "{hi:?}");
    }

    #[test]
    fn the_seated_rider_keeps_to_the_cab() {
        let b = standard(BaseKind::Wheel);
        let v = rider_parts(&b, &KinPose::stand(&b, 0.0), &Palette::default());
        let (lo, hi) = bounds(&v);
        assert!(hi[2] < b.dims.roof_m.unwrap() - 0.03, "head under the guard: {hi:?}");
        assert!(lo[2] > 0.2, "feet on the floor plate: {lo:?}");
        assert!(hi[0] < crate::act::fork::front_x(&b), "behind the mast: {hi:?}");
    }
}
