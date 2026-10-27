//! Software ray-caster for the first-person camera and a top-down map.
//!
//! Each pixel walks the cell grid (2D DDA) and tests the boxes in every cell it
//! enters: the terrain column, a curb on top of it, and any node, drop or
//! waypoint post standing in the cell. All boxes stay inside their own cell, so
//! the first cell with a hit holds the nearest hit.
//!
//! The output is an RGB frame plus an object-id buffer and a depth buffer. The
//! id buffer is the ground truth for tests: it says which object every pixel
//! belongs to, without any recogniser in the loop.
//!
//! Coordinates: x and z are the ground plane, y is up. A heading of 0 looks
//! along +x and 90 degrees looks along +z. The camera's right is
//! `(-fz, 0, fx)`, so looking along +x the right-hand side is +z. The top view
//! draws x to the right and z downward.
//!
//! The first-person camera is the head camera of `World::head_view`: a
//! [`View`] (where it is and how it is turned, roll included) and an
//! [`Exposure`] (how fast it turns while the sensor reads the frame). A
//! rolling shutter reads the rows one after another, so each row is cast from
//! the orientation of its own instant; the exposure then smears every pixel
//! along the path the scene takes across the sensor while it is open (motion
//! blur). The id and depth buffers follow the rows' rays and are not blurred:
//! they stay the ground truth of what each pixel shows.
//!
//! Light (v1.0, `light`): the world's `WorldConfig::lighting` decides how a
//! hit is shaded. The flat lighting is the pre-v1.0 renderer, pixel for
//! pixel: a fixed factor per face axis, a constant sky and fog. Under a sun
//! each face takes the beam by its outward normal (the box face a ray enters
//! and its sign) when a shadow ray toward the sun reaches the sky past the
//! scene and the robot's links, plus the sky's and the ground's light; the
//! sky and the fog take the sun's glow.
//!
//! Rays ([`Rays`]): a pinhole camera (every view before v1.0), or an
//! orthographic one whose parallel rays start on a plane behind the frame's
//! centre, outside the map if need be: a ray is clipped to the map (and to
//! the height of its tallest object) before it walks the cells.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::kinds::{Ground, ItemKind, NodeKind};
use crate::light::Daylight;
use crate::rng::hash2;
use crate::snapshot::{drop_id, node_id, waypoint_id};
use crate::terrain::{cell_center, CELL_M};
use crate::world::World;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum ObjectClass {
    Sky = 0,
    Ground = 1,
    Curb = 2,
    Node = 3,
    Drop = 4,
    Waypoint = 5,
    /// The pile of delivered items beside a waypoint's flag (entity = waypoint id).
    Stock = 6,
    /// A site's building part: a wall, door frame, roof, floor, porch, stair or
    /// ramp (entity = fixture id, `crate::fixture`). Ground truth, never a detection.
    Structure = 7,
    /// A site's furniture: a shelving unit or a packing table (entity = fixture id).
    Fixture = 8,
}

impl ObjectClass {
    fn from_u32(v: u32) -> ObjectClass {
        match v {
            1 => ObjectClass::Ground,
            2 => ObjectClass::Curb,
            3 => ObjectClass::Node,
            4 => ObjectClass::Drop,
            5 => ObjectClass::Waypoint,
            6 => ObjectClass::Stock,
            7 => ObjectClass::Structure,
            8 => ObjectClass::Fixture,
            _ => ObjectClass::Sky,
        }
    }
}

/// Piece slots of a waypoint's pile inside its cell: four columns around the
/// pole on the side away from the flag, five pieces high (`(dx, dz)` metres).
const PILE_COLUMNS: [(f32, f32); 4] = [(-0.16, -0.13), (-0.16, 0.13), (0.0, -0.17), (0.0, 0.17)];
const PILE_LAYERS: usize = 5;

/// Where the pieces of a pile sit: `(cell offset in x, dx, dz, layer)` for
/// every piece, filling the waypoint's own cell first, then the cell on its
/// `-x` side. At most `2 * 4 * 5` pieces are drawn; the rest only count.
pub fn pile_slots(pieces: usize) -> Vec<(i32, f32, f32, usize)> {
    let per_cell = PILE_COLUMNS.len() * PILE_LAYERS;
    (0..pieces.min(2 * per_cell))
        .map(|i| {
            let (cell, k) = (i / per_cell, i % per_cell);
            let (col, layer) = (k % PILE_COLUMNS.len(), k / PILE_COLUMNS.len());
            let (dx, dz) = PILE_COLUMNS[col];
            (-(cell as i32), dx, dz, layer)
        })
        .collect()
}

/// The pieces of a stock list, one per whole item and one for each broken rest.
pub fn stock_pieces(stock: &[(ItemKind, f32)]) -> Vec<(ItemKind, f32)> {
    let mut out = Vec::new();
    for &(k, n) in stock {
        let mut rest = n;
        while rest > 1e-3 {
            out.push((k, rest.min(1.0)));
            rest -= 1.0;
        }
    }
    out
}

/// Packs an object class (top 8 bits) and an entity id (low 24 bits).
pub fn pack_id(class: ObjectClass, entity: u32) -> u32 {
    ((class as u32) << 24) | (entity & 0x00FF_FFFF)
}

pub fn unpack_id(v: u32) -> (ObjectClass, u32) {
    (ObjectClass::from_u32(v >> 24), v & 0x00FF_FFFF)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub width: usize,
    pub height: usize,
    /// Horizontal field of view, degrees.
    pub fov_deg: f32,
    /// Pitch, degrees; negative looks down. On the robot this is the
    /// camera's mount tilt on the head, plus the neck's tilt for a look at the
    /// ground (the body's own bend comes from `World::head_view`).
    pub pitch_deg: f32,
    pub max_dist_m: f32,
    /// Neck pan the runtime commands for a look, degrees, to the robot's left
    /// positive (on top of the pan of the pose the robot holds).
    pub pan_deg: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { width: 384, height: 256, fov_deg: 70.0, pitch_deg: -12.0, max_dist_m: 24.0, pan_deg: 0.0 }
    }
}

/// Where a camera is and how it is turned at one instant: the optical centre
/// (world, y up) and the heading, pitch (negative looks down) and roll (the
/// right side down positive), radians.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct View {
    pub origin: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
}

/// How the sensor takes a frame while the camera turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Exposure {
    /// Angular rate (yaw, pitch, roll), rad/s, at the middle of the frame.
    pub rate: [f32; 3],
    /// Time the pixels collect light, s (motion blur).
    pub exposure_s: f32,
    /// Rolling-shutter readout from the first row to the last, s (0: a global shutter).
    pub readout_s: f32,
}

impl Exposure {
    /// Largest image motion during the exposure, pixels (at the frame centre).
    pub fn blur_px(&self, cam: &Camera) -> f32 {
        let f = focal_px(cam);
        let w = self.rate[0].hypot(self.rate[1]);
        let roll = self.rate[2].abs() * 0.5 * cam.width.max(cam.height) as f32;
        (w * f + roll) * self.exposure_s
    }
}

/// Focal length of a camera in pixels.
pub fn focal_px(cam: &Camera) -> f32 {
    0.5 * cam.width.max(1) as f32 / (cam.fov_deg.to_radians() * 0.5).tan()
}

impl Camera {
    /// Where the camera sits now: the head camera's optical centre.
    pub fn origin(world: &World) -> [f32; 3] {
        world.head_view(&Camera::default()).view.origin
    }

    /// Vertical field of view, degrees (from the horizontal one and the aspect).
    pub fn vfov_deg(&self) -> f32 {
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        (2.0 * (half_w * self.height.max(1) as f32 / self.width.max(1) as f32).atan()).to_degrees()
    }

    /// What the search planner needs of this camera on the robot now.
    pub fn eye(&self, world: &World) -> crate::explore::Eye {
        crate::explore::Eye {
            x: world.agent.x,
            z: world.agent.z,
            height_m: world.agent_ground_m() + world.standing_eye_m(),
            pitch_deg: self.pitch_deg,
            vfov_deg: self.vfov_deg(),
            range_m: crate::explore::SEARCH_RANGE_M.min(self.max_dist_m),
        }
    }

    /// Ground-plane `(x, z)` of the surface point seen at pixel `(px, py)` when
    /// the depth buffer says it is `depth_m` away along the forward axis.
    /// Pixel coordinates are continuous: pixel `i` spans `i..i+1`, so a
    /// fraction `f` of the frame width is `f * width`. `None` for a depth that
    /// is not finite and positive.
    pub fn world_point(&self, agent_xz: (f32, f32), yaw_rad: f32, px: f32, py: f32, depth_m: f32) -> Option<(f32, f32)> {
        let view = View { origin: [agent_xz.0, 0.0, agent_xz.1], yaw: yaw_rad, pitch: self.pitch_deg.to_radians(), roll: 0.0 };
        self.world_point_in(&view, px, py, depth_m)
    }

    /// Ground-plane `(x, z)` of the surface point seen at pixel `(px, py)`
    /// from `view`, `depth_m` away along its forward axis (the view's own
    /// pitch and roll; this camera's `pitch_deg` is not used).
    pub fn world_point_in(&self, view: &View, px: f32, py: f32, depth_m: f32) -> Option<(f32, f32)> {
        self.world_point3_in(view, px, py, depth_m).map(|p| (p[0], p[2]))
    }

    /// The surface point (x, y, z) seen at pixel `(px, py)` from `view`,
    /// `depth_m` away along its forward axis.
    pub fn world_point3_in(&self, view: &View, px: f32, py: f32, depth_m: f32) -> Option<[f32; 3]> {
        if !depth_m.is_finite() || depth_m <= 0.0 {
            return None;
        }
        let (w, h) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let (fwd, right, up) = basis_rpy(view.yaw, view.pitch, view.roll);
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        let half_h = half_w * h / w;
        let sx = (2.0 * px / w - 1.0) * half_w;
        let sy = (1.0 - 2.0 * py / h) * half_h;
        let v = [fwd[0] + right[0] * sx + up[0] * sy, fwd[1] + right[1] * sx + up[1] * sy, fwd[2] + right[2] * sx + up[2] * sy];
        Some([view.origin[0] + depth_m * v[0], view.origin[1] + depth_m * v[1], view.origin[2] + depth_m * v[2]])
    }

    /// Pixel and forward depth of a world point, the inverse of
    /// [`Camera::world_point`]. `None` for a point behind the camera.
    pub fn project(&self, origin: [f32; 3], yaw_rad: f32, point: [f32; 3]) -> Option<(f32, f32, f32)> {
        self.project_in(&View { origin, yaw: yaw_rad, pitch: self.pitch_deg.to_radians(), roll: 0.0 }, point)
    }

    /// Pixel and forward depth of a world point seen from `view`, the inverse
    /// of [`Camera::world_point_in`]. `None` for a point behind the camera.
    pub fn project_in(&self, view: &View, point: [f32; 3]) -> Option<(f32, f32, f32)> {
        let (w, h) = (self.width.max(1) as f32, self.height.max(1) as f32);
        let (fwd, right, up) = basis_rpy(view.yaw, view.pitch, view.roll);
        let origin = view.origin;
        let rel = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let depth = dot(rel, fwd);
        if depth <= 1e-4 {
            return None;
        }
        let half_w = (self.fov_deg.to_radians() * 0.5).tan();
        let half_h = half_w * h / w;
        let sx = dot(rel, right) / depth;
        let sy = dot(rel, up) / depth;
        Some(((sx / half_w + 1.0) * 0.5 * w, (1.0 - sy / half_h) * 0.5 * h, depth))
    }
}

/// One rendered frame with its ground truth.
#[derive(Clone, Debug)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub ids: Vec<u32>,
    /// Distance along the camera's forward axis, metres; infinity for sky.
    pub depth: Vec<f32>,
    pub tick: u64,
}

/// An object seen in a frame, from the id buffer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisibleObject {
    pub class: ObjectClass,
    /// `n12`, `d3` or `w1`.
    pub entity: String,
    /// Kind id such as `tree`, `log`, `warehouse`.
    pub label: String,
    pub label_ko: String,
    pub pixels: u32,
    /// Inclusive pixel box `[x0, y0, x1, y1]`.
    pub bbox: [u32; 4],
    pub depth_m: f32,
}

#[derive(Clone, Copy)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Clone, Copy, PartialEq)]
enum Style {
    Plain,
    Fleck([u8; 3]),
    Stripes,
    Leaves,
    Water,
}

#[derive(Clone, Copy)]
struct Prim {
    aabb: Aabb,
    id: u32,
    color: [u8; 3],
    style: Style,
}

struct Scene<'a> {
    world: &'a World,
    prims: Vec<Prim>,
    by_cell: Vec<Vec<usize>>,
    /// Nothing in the world reaches above this height, metres.
    top_m: f32,
}

const SKY_TOP: [f32; 3] = [112.0, 168.0, 232.0];
const SKY_HORIZON: [f32; 3] = [200.0, 222.0, 240.0];
const CURB_YELLOW: [u8; 3] = [232, 196, 48];
const CURB_DARK: [u8; 3] = [48, 48, 52];
const STONE: [u8; 3] = [120, 120, 124];
/// Tallest object above the ground it stands on (a tree), metres.
const MAX_OBJECT_TOP_M: f32 = 1.7;
/// The sea surface lies this far below the land at height 0, metres.
const WATER_DROP_M: f32 = 0.06;

fn bx(cx: f32, cz: f32, y0: f32, y1: f32, hw: f32, hd: f32) -> Aabb {
    Aabb { min: [cx - hw, y0, cz - hd], max: [cx + hw, y1, cz + hd] }
}

fn item_style(kind: ItemKind) -> Style {
    match kind {
        ItemKind::Log | ItemKind::Stone | ItemKind::Parcel => Style::Plain,
        ItemKind::Coal => Style::Fleck([70, 70, 78]),
        ItemKind::IronOre => Style::Fleck([220, 170, 140]),
    }
}

impl<'a> Scene<'a> {
    fn build(world: &'a World) -> Scene<'a> {
        let t = &world.terrain;
        let mut s = Scene { world, prims: Vec::new(), by_cell: vec![Vec::new(); t.w * t.d], top_m: t.max_top_m() + MAX_OBJECT_TOP_M + 0.3 };

        for n in &world.nodes {
            let Some((cx, cz)) = t.cell_of(n.x, n.z) else { continue };
            let (x, z) = cell_center(cx, cz);
            let g = t.height_m(cx, cz) + t.curb_m(cx, cz);
            let id = pack_id(ObjectClass::Node, n.id);
            let spec = n.kind.spec();
            match n.kind {
                NodeKind::Tree => {
                    s.add(cx, cz, Prim { aabb: bx(x, z, g, g + 1.0, 0.10, 0.10), id, color: spec.color, style: Style::Plain });
                    s.add(cx, cz, Prim { aabb: bx(x, z, g + 0.8, g + 1.6, 0.24, 0.24), id, color: [58, 122, 52], style: Style::Leaves });
                }
                NodeKind::Boulder => {
                    s.add(cx, cz, Prim { aabb: bx(x, z, g, g + 0.40, 0.24, 0.24), id, color: spec.color, style: Style::Plain });
                    s.add(cx, cz, Prim { aabb: bx(x, z, g + 0.40, g + 0.60, 0.16, 0.16), id, color: spec.color, style: Style::Plain });
                }
                NodeKind::CoalOre => {
                    s.add(cx, cz, Prim { aabb: bx(x, z, g, g + 0.5, 0.24, 0.24), id, color: STONE, style: Style::Fleck([28, 28, 32]) });
                }
                NodeKind::IronOre => {
                    s.add(cx, cz, Prim { aabb: bx(x, z, g, g + 0.5, 0.24, 0.24), id, color: STONE, style: Style::Fleck([206, 150, 120]) });
                }
            }
        }

        for d in &world.drops {
            let Some((cx, cz)) = t.cell_of(d.x, d.z) else { continue };
            let (ccx, ccz) = cell_center(cx, cz);
            // On what it rests: the cell's top, or a porch, a shelf deck, a table top.
            let g = world.drop_base_m(d);
            let spec = d.kind.spec();
            let size = spec.piece_size_m(d.amount);
            let hw = size[0] * 0.5;
            let limit = CELL_M * 0.5 - hw - 0.005;
            let x = d.x.clamp(ccx - limit, ccx + limit);
            let z = d.z.clamp(ccz - limit, ccz + limit);
            s.add(cx, cz, Prim {
                aabb: bx(x, z, g, g + size[1], hw, size[2] * 0.5),
                id: pack_id(ObjectClass::Drop, d.id),
                color: spec.color,
                style: item_style(d.kind),
            });
        }

        for w in &world.waypoints {
            let Some((cx, cz)) = t.cell_of(w.x, w.z) else { continue };
            let (x, z) = cell_center(cx, cz);
            let g = t.height_m(cx, cz) + t.curb_m(cx, cz);
            let id = pack_id(ObjectClass::Waypoint, w.id);
            let flag = if w.key == "warehouse" { [52, 104, 220] } else { [232, 120, 40] };
            s.add(cx, cz, Prim { aabb: bx(x, z, g, g + 1.1, 0.04, 0.04), id, color: [210, 210, 214], style: Style::Plain });
            s.add(cx, cz, Prim { aabb: Aabb { min: [x + 0.04, g + 0.78, z - 0.02], max: [x + 0.24, g + 1.08, z + 0.02] }, id, color: flag, style: Style::Plain });
            // The pile: what has been delivered here, stacked beside the pole.
            let pieces = stock_pieces(&w.stock);
            let sid = pack_id(ObjectClass::Stock, w.id);
            let mut heights = [[0.0f32; PILE_COLUMNS.len()]; 2];
            for ((kind, amount), (cell, dx, dz, _)) in pieces.iter().zip(pile_slots(pieces.len())) {
                let pcx = cx as i32 + cell;
                if !t.in_bounds(pcx, cz as i32) {
                    continue;
                }
                let (px, pz) = cell_center(pcx as usize, cz);
                let pg = t.height_m(pcx as usize, cz) + t.curb_m(pcx as usize, cz);
                let col = PILE_COLUMNS.iter().position(|c| (c.0, c.1) == (dx, dz)).unwrap_or(0);
                let size = kind.spec().piece_size_m(*amount);
                let base = pg + heights[(-cell) as usize][col];
                heights[(-cell) as usize][col] += size[1];
                s.add(pcx as usize, cz, Prim { aabb: bx(px + dx, pz + dz, base, base + size[1], size[0] * 0.5, size[2] * 0.5), id: sid, color: kind.spec().color, style: item_style(*kind) });
            }
        }
        s.add_site();
        s
    }

    /// A site's walls, doors, roofs, porches, stairs, ramps, shelves and
    /// tables (`crate::fixture`), each box in its own cell; the tallest of them
    /// raises the height above which a rising ray stops looking.
    fn add_site(&mut self) {
        for b in crate::fixture::render_boxes(&self.world.fixtures, &self.world.terrain) {
            self.top_m = self.top_m.max(b.max[1] + 0.3);
            self.add(b.cell.0, b.cell.1, Prim { aabb: Aabb { min: b.min, max: b.max }, id: pack_id(b.class, b.entity), color: b.color, style: Style::Plain });
        }
    }

    fn add(&mut self, cx: usize, cz: usize, p: Prim) {
        let i = self.world.terrain.idx(cx, cz);
        self.by_cell[i].push(self.prims.len());
        self.prims.push(p);
    }
}

struct Hit {
    t: f32,
    axis: usize,
    /// The face's outward normal is `sign` along `axis`.
    sign: f32,
    id: u32,
    color: [u8; 3],
    style: Style,
}

impl Hit {
    fn normal(&self) -> [f32; 3] {
        let mut n = [0.0; 3];
        n[self.axis] = self.sign;
        n
    }
}

/// Ray against a box: the distance, the axis of the face the ray enters and
/// the sign of that face's outward normal along it (−1 for the face at the
/// box's minimum, which a ray moving up the axis enters); `None` when the
/// origin is inside or the box is behind.
fn slab(o: [f32; 3], inv: [f32; 3], b: &Aabb) -> Option<(f32, usize, f32)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    let mut axis = 0;
    for a in 0..3 {
        let t1 = (b.min[a] - o[a]) * inv[a];
        let t2 = (b.max[a] - o[a]) * inv[a];
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        if lo > t_near {
            t_near = lo;
            axis = a;
        }
        if hi < t_far {
            t_far = hi;
        }
        if t_near > t_far {
            return None;
        }
    }
    if t_near < 0.0 {
        return None;
    }
    Some((t_near, axis, if inv[axis] > 0.0 { -1.0 } else { 1.0 }))
}

/// What an orthographic view leaves out so the robot stays in sight
/// (`iso::Cutaway::around` builds one). Only prims that can hide the robot
/// are cut; terrain never is.
///
/// * The sight corridor: a prim whose footprint reaches into the strip that
///   runs from the robot (ground point `at`) toward the camera
///   (`toward_cam`, unit, on the ground), `half_width_m` to each side, is cut
///   down to the sight line. The view's parallel rays that reach the robot's
///   ground rise `slope` metres per metre toward the camera, so nothing below
///   `base_y + slope·a` at a distance `a` in front of the robot can hide any
///   of it. Never lower than `low_y`: what lies at its feet stays.
/// * Over the robot: a prim whose footprint comes within `roof_radius_m` of
///   it is cut at `roof_y` (a roof, a shelf it stands under).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cutaway {
    pub at: [f32; 2],
    pub toward_cam: [f32; 2],
    pub half_width_m: f32,
    pub slope: f32,
    pub base_y: f32,
    pub low_y: f32,
    pub roof_radius_m: f32,
    pub roof_y: f32,
}

impl Cutaway {
    /// Height a prim's box is cut at (`None`: it stays whole).
    fn cut_y(&self, b: &Aabb) -> Option<f32> {
        let t = self.toward_cam;
        let side = [-t[1], t[0]];
        // The footprint's extent along the corridor and across it.
        let (mut a0, mut a1, mut l0, mut l1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for (x, z) in [(b.min[0], b.min[2]), (b.max[0], b.min[2]), (b.min[0], b.max[2]), (b.max[0], b.max[2])] {
            let (dx, dz) = (x - self.at[0], z - self.at[1]);
            let (a, l) = (dx * t[0] + dz * t[1], dx * side[0] + dz * side[1]);
            (a0, a1, l0, l1) = (a0.min(a), a1.max(a), l0.min(l), l1.max(l));
        }
        let corridor = (a1 >= 0.0 && l1 >= -self.half_width_m && l0 <= self.half_width_m).then(|| (self.base_y + a0.max(0.0) * self.slope).max(self.low_y));
        let gx = (b.min[0] - self.at[0]).max(self.at[0] - b.max[0]).max(0.0);
        let gz = (b.min[2] - self.at[1]).max(self.at[1] - b.max[2]).max(0.0);
        let roof = (gx.hypot(gz) <= self.roof_radius_m).then_some(self.roof_y);
        match (corridor, roof) {
            (Some(c), Some(r)) => Some(c.min(r)),
            (c, r) => c.or(r),
        }
    }

    /// The part of a prim's box left after the cut, `None` when nothing is.
    fn clip(&self, b: &Aabb) -> Option<Aabb> {
        match self.cut_y(b) {
            None => Some(*b),
            Some(top) => (b.min[1] < top).then(|| Aabb { min: b.min, max: [b.max[0], b.max[1].min(top), b.max[2]] }),
        }
    }
}

fn trace(scene: &Scene, o: [f32; 3], d: [f32; 3], max_dist: f32) -> Option<Hit> {
    trace_cut(scene, o, d, max_dist, None)
}

fn trace_cut(scene: &Scene, o: [f32; 3], d: [f32; 3], max_dist: f32, cut: Option<&Cutaway>) -> Option<Hit> {
    let t = &scene.world.terrain;
    let inv = [
        if d[0].abs() < 1e-9 { f32::INFINITY.copysign(d[0] + 1e-30) } else { 1.0 / d[0] },
        if d[1].abs() < 1e-9 { f32::INFINITY.copysign(d[1] + 1e-30) } else { 1.0 / d[1] },
        if d[2].abs() < 1e-9 { f32::INFINITY.copysign(d[2] + 1e-30) } else { 1.0 / d[2] },
    ];
    let (span_x, span_z) = (t.w as f32 * CELL_M, t.d as f32 * CELL_M);
    let inside = o[0] >= 0.0 && o[0] < span_x && o[2] >= 0.0 && o[2] < span_z;
    let above = o[1] > scene.top_m;
    let (mut cx, mut cz) = if inside && !above {
        ((o[0] / CELL_M).floor() as i32, (o[2] / CELL_M).floor() as i32)
    } else {
        // A ray from outside the map starts where it enters it, one from
        // above everything where it comes down to the tallest object: nothing
        // can be hit before. The distances stay measured from `o`.
        let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
        for (a, span) in [(0usize, span_x), (2, span_z)] {
            if d[a].abs() < 1e-9 {
                if o[a] < 0.0 || o[a] >= span {
                    return None;
                }
            } else {
                let (u, v) = (-o[a] * inv[a], (span - o[a]) * inv[a]);
                t0 = t0.max(u.min(v));
                t1 = t1.min(u.max(v));
            }
        }
        if above {
            if d[1] >= 0.0 {
                return None;
            }
            t0 = t0.max((o[1] - scene.top_m) / -d[1]);
        }
        if t0 >= t1 || t0 > max_dist {
            return None;
        }
        let (px, pz) = (o[0] + d[0] * t0, o[2] + d[2] * t0);
        (((px / CELL_M).floor() as i32).clamp(0, t.w as i32 - 1), ((pz / CELL_M).floor() as i32).clamp(0, t.d as i32 - 1))
    };
    let step_x: i32 = if d[0] > 0.0 { 1 } else { -1 };
    let step_z: i32 = if d[2] > 0.0 { 1 } else { -1 };
    let t_delta_x = if d[0].abs() < 1e-9 { f32::INFINITY } else { CELL_M / d[0].abs() };
    let t_delta_z = if d[2].abs() < 1e-9 { f32::INFINITY } else { CELL_M / d[2].abs() };
    let next_x = (cx + if step_x > 0 { 1 } else { 0 }) as f32 * CELL_M;
    let next_z = (cz + if step_z > 0 { 1 } else { 0 }) as f32 * CELL_M;
    let mut t_max_x = if d[0].abs() < 1e-9 { f32::INFINITY } else { (next_x - o[0]) / d[0] };
    let mut t_max_z = if d[2].abs() < 1e-9 { f32::INFINITY } else { (next_z - o[2]) / d[2] };

    loop {
        if !t.in_bounds(cx, cz) {
            return None;
        }
        let (ux, uz) = (cx as usize, cz as usize);
        let ci = t.idx(ux, uz);
        let mut best: Option<Hit> = None;
        let mut consider = |b: &Aabb, id: u32, color: [u8; 3], style: Style| {
            if let Some((tt, axis, sign)) = slab(o, inv, b) {
                if tt <= max_dist && best.as_ref().map_or(true, |h| tt < h.t) {
                    best = Some(Hit { t: tt, axis, sign, id, color, style });
                }
            }
        };
        let (x0, z0) = (cx as f32 * CELL_M, cz as f32 * CELL_M);
        let h = t.height_m(ux, uz);
        let curb = t.curb_m(ux, uz);
        let ground = t.ground_at(ux, uz);
        let gcol = ground.color();
        if ground.is_water() {
            // The sea: a flat surface a little below the shore.
            consider(&Aabb { min: [x0, -0.4, z0], max: [x0 + CELL_M, -WATER_DROP_M, z0 + CELL_M] }, pack_id(ObjectClass::Ground, ci as u32), gcol, Style::Water);
        } else if h > 0.0 {
            consider(&Aabb { min: [x0, 0.0, z0], max: [x0 + CELL_M, h, z0 + CELL_M] }, pack_id(ObjectClass::Ground, ci as u32), gcol, Style::Plain);
        } else {
            // Level ground: a thin slab so the top face is hit.
            consider(&Aabb { min: [x0, -0.2, z0], max: [x0 + CELL_M, 0.0, z0 + CELL_M] }, pack_id(ObjectClass::Ground, ci as u32), gcol, Style::Plain);
        }
        if curb > 0.0 {
            consider(&Aabb { min: [x0, h, z0], max: [x0 + CELL_M, h + curb, z0 + CELL_M] }, pack_id(ObjectClass::Curb, ci as u32), CURB_YELLOW, Style::Stripes);
        }
        for &pi in &scene.by_cell[ci] {
            let p = &scene.prims[pi];
            match cut {
                None => consider(&p.aabb, p.id, p.color, p.style),
                Some(c) => {
                    if let Some(b) = c.clip(&p.aabb) {
                        consider(&b, p.id, p.color, p.style);
                    }
                }
            }
        }
        if best.is_some() {
            return best;
        }

        let t_exit = if t_max_x < t_max_z {
            cx += step_x;
            let e = t_max_x;
            t_max_x += t_delta_x;
            e
        } else {
            cz += step_z;
            let e = t_max_z;
            t_max_z += t_delta_z;
            e
        };
        if t_exit > max_dist {
            return None;
        }
        // Above everything and still rising: nothing more to hit.
        if o[1] + d[1] * t_exit > scene.top_m && d[1] >= 0.0 {
            return None;
        }
        // Above everything and falling would still hit the ground, keep going.
    }
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn sky(dir_y: f32) -> [f32; 3] {
    mix(SKY_HORIZON, SKY_TOP, dir_y.clamp(0.0, 1.0).sqrt())
}

/// The colour of the surface a ray hit at `p`, before light: the prim's
/// colour through its style, and the texel's ±7 % noise.
fn surface(hit: &Hit, p: [f32; 3], seed: u64) -> ([f32; 3], f32) {
    // Texel coordinates on the face that was hit (6 cm texels).
    let (u, v) = match hit.axis {
        1 => (p[0], p[2]),
        0 => (p[1], p[2]),
        _ => (p[0], p[1]),
    };
    let (tu, tv) = ((u * 16.0).floor() as i32, (v * 16.0).floor() as i32);
    let h = hash2(seed ^ (hit.id as u64).wrapping_mul(0x9E37) ^ hit.axis as u64, tu, tv);
    let noise = ((h & 0xFF) as f32 / 255.0 - 0.5) * 0.14;

    let mut c = [hit.color[0] as f32, hit.color[1] as f32, hit.color[2] as f32];
    match hit.style {
        Style::Plain => {}
        Style::Fleck(f) => {
            if (h >> 8) % 6 == 0 {
                c = [f[0] as f32, f[1] as f32, f[2] as f32];
            }
        }
        Style::Leaves => {
            let k = ((h >> 8) & 0xF) as f32 / 15.0;
            c = mix(c, [30.0, 90.0, 36.0], k * 0.6);
        }
        Style::Stripes => {
            let band = ((p[0] + p[1] + p[2]) * 5.0).floor() as i32;
            if band.rem_euclid(2) == 0 {
                c = [CURB_DARK[0] as f32, CURB_DARK[1] as f32, CURB_DARK[2] as f32];
            }
        }
        Style::Water => {
            // Ripples: brighter crests on a slow wave pattern.
            let wave = ((p[0] * 2.1 + p[2] * 1.3).sin() * (p[2] * 2.7 - p[0] * 0.9).cos()).max(0.0);
            c = mix(c, [150.0, 196.0, 230.0], 0.35 * wave);
        }
    }
    (c, noise)
}

/// The flat lighting (before v1.0): a fixed factor per face axis, fog toward the horizon's colour.
fn shade_hit(hit: &Hit, o: [f32; 3], d: [f32; 3], max_dist: f32, seed: u64) -> [u8; 3] {
    let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
    let (c, noise) = surface(hit, p, seed);
    let face = match hit.axis {
        1 => 1.0,
        0 => 0.82,
        _ => 0.66,
    };
    let k = face * (1.0 + noise);
    let lit = [c[0] * k, c[1] * k, c[2] * k];
    let fog = (hit.t / max_dist).clamp(0.0, 1.0).powf(1.6);
    let out = mix(lit, SKY_HORIZON, fog);
    [out[0].clamp(0.0, 255.0) as u8, out[1].clamp(0.0, 255.0) as u8, out[2].clamp(0.0, 255.0) as u8]
}

/// How far a shadow ray starts off the face it leaves, metres (keeps it from hitting that face).
const SHADOW_BIAS_M: f32 = 1e-3;

/// The boxes that cast shadows (the robot's links), with a sphere around them
/// all: a shadow ray that misses the sphere skips the box tests.
struct Casters<'a> {
    lists: [&'a [Obb]; 2],
    centre: [f32; 3],
    /// Negative when there are no boxes.
    radius: f32,
}

impl<'a> Casters<'a> {
    fn new(lists: [&'a [Obb]; 2]) -> Casters<'a> {
        let all = || lists.iter().flat_map(|l| l.iter());
        let n = all().count();
        if n == 0 {
            return Casters { lists, centre: [0.0; 3], radius: -1.0 };
        }
        let mut c = [0.0f32; 3];
        for b in all() {
            for k in 0..3 {
                c[k] += b.center[k] / n as f32;
            }
        }
        let radius = all()
            .map(|b| {
                let d = [b.center[0] - c[0], b.center[1] - c[1], b.center[2] - c[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() + (b.half[0] * b.half[0] + b.half[1] * b.half[1] + b.half[2] * b.half[2]).sqrt()
            })
            .fold(0.0f32, f32::max);
        Casters { lists, centre: c, radius }
    }

    /// Whether a ray from `o` along unit `d` meets one of the boxes.
    fn block(&self, o: [f32; 3], d: [f32; 3]) -> bool {
        if self.radius < 0.0 {
            return false;
        }
        let v = [self.centre[0] - o[0], self.centre[1] - o[1], self.centre[2] - o[2]];
        let along = (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]).max(0.0);
        if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - along * along > self.radius * self.radius {
            return false;
        }
        self.lists.iter().any(|list| list.iter().any(|b| b.hit(o, d).is_some()))
    }
}

/// Whether the sun reaches point `p` on a face with outward normal `n`: the
/// face turns toward a sun above the horizon and a ray toward it meets
/// neither the scene nor any of the casters.
fn sunlit(scene: &Scene, day: &Daylight, p: [f32; 3], n: [f32; 3], casters: &Casters) -> bool {
    let l = day.sun_dir;
    if !day.has_beam() || n[0] * l[0] + n[1] * l[1] + n[2] * l[2] <= 0.0 {
        return false;
    }
    let o = [p[0] + n[0] * SHADOW_BIAS_M, p[1] + n[1] * SHADOW_BIAS_M, p[2] + n[2] * SHADOW_BIAS_M];
    !casters.block(o, l) && trace(scene, o, l, f32::INFINITY).is_none()
}

/// The horizon's colour seen along `d` under a sun (the fog's colour).
fn horizon_px(day: &Daylight, d: [f32; 3]) -> [f32; 3] {
    let len = d[0].hypot(d[2]);
    let flat = if len > 1e-6 { [d[0] / len, 0.0, d[2] / len] } else { [1.0, 0.0, 0.0] };
    day.sky_px(flat, SKY_HORIZON)
}

/// The sun's lighting (`light::Daylight`): the surface's colour times the
/// light its face receives over the metered light (through the display
/// gamma), fogged toward the horizon's colour in the ray's direction.
fn shade_sun(hit: &Hit, p: [f32; 3], d: [f32; 3], fog_dist: f32, seed: u64, day: &Daylight, lit: bool) -> [u8; 3] {
    let (c, noise) = surface(hit, p, seed);
    let g = day.face_gain(hit.normal(), lit);
    let k = 1.0 + noise;
    let lit = [c[0] * g[0] * k, c[1] * g[1] * k, c[2] * g[2] * k];
    let fog = (hit.t / fog_dist).clamp(0.0, 1.0).powf(1.6);
    let out = if fog > 0.0 { mix(lit, horizon_px(day, d), fog) } else { lit };
    [out[0].clamp(0.0, 255.0) as u8, out[1].clamp(0.0, 255.0) as u8, out[2].clamp(0.0, 255.0) as u8]
}

/// Camera basis: forward, right, up.
pub(crate) fn basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let fwd = [cy * cp, sp, sy * cp];
    let right = [-sy, 0.0, cy];
    let up = [-cy * sp, cp, -sy * sp];
    (fwd, right, up)
}

/// Camera basis with a roll about the forward axis (the right side down positive).
fn basis_rpy(yaw: f32, pitch: f32, roll: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (fwd, right, up) = basis(yaw, pitch);
    if roll == 0.0 {
        return (fwd, right, up);
    }
    let (sr, cr) = roll.sin_cos();
    let r = [right[0] * cr - up[0] * sr, right[1] * cr - up[1] * sr, right[2] * cr - up[2] * sr];
    let u = [up[0] * cr + right[0] * sr, up[1] * cr + right[1] * sr, up[2] * cr + right[2] * sr];
    (fwd, r, u)
}

/// Renders the robot's first-person view: the head camera as it is now
/// (`World::head_view`), its rolling shutter and its motion blur.
pub fn render(world: &World, cam: &Camera) -> Frame {
    let hv = world.head_view(cam);
    render_view(world, cam, &hv.view, &hv.exposure)
}

/// Renders from a given view through a sensor with the given exposure. The
/// robot's own links are not drawn; under a sun they still cast their shadow.
pub fn render_view(world: &World, cam: &Camera, view: &View, exp: &Exposure) -> Frame {
    let casters = world.shadow_casters();
    let mut f = render_rays(world, cam, &Rays::Pinhole { view: *view, exposure: *exp }, &[], &casters, None);
    motion_blur(&mut f, cam, view, exp);
    f
}

/// The basis row `j` of `h` is read with: the view turned on by the rate for
/// the row's offset from the middle of the readout.
fn row_basis(view: &View, exp: &Exposure, j: usize, h: usize) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let tau = exp.readout_s * ((j as f32 + 0.5) / h.max(1) as f32 - 0.5);
    basis_rpy(view.yaw + exp.rate[0] * tau, view.pitch + exp.rate[1] * tau, view.roll + exp.rate[2] * tau)
}

/// Smears each pixel along the path the scene takes across the sensor while
/// it is open: the pixel's ray (at the middle of the exposure) projected
/// through the camera turned back and on by half the exposure. Rotation only:
/// the camera's travel during a few milliseconds moves the image by far less
/// than a pixel at the ranges the robot looks at.
fn motion_blur(f: &mut Frame, cam: &Camera, view: &View, exp: &Exposure) {
    if exp.exposure_s <= 0.0 || exp.blur_px(cam) < 0.35 {
        return;
    }
    let (w, h) = (f.width, f.height);
    let half_w = (cam.fov_deg.to_radians() * 0.5).tan();
    let half_h = half_w * h as f32 / w as f32;
    let half = 0.5 * exp.exposure_s;
    let src = f.rgb.clone();
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let to_px = |d: [f32; 3], b: &([f32; 3], [f32; 3], [f32; 3])| -> Option<(f32, f32)> {
        let z = dot(d, b.0);
        if z <= 1e-4 {
            return None;
        }
        Some(((dot(d, b.1) / z / half_w + 1.0) * 0.5 * w as f32, (1.0 - dot(d, b.2) / z / half_h) * 0.5 * h as f32))
    };
    f.rgb.par_chunks_mut(w * 3).enumerate().for_each(|(j, row)| {
        let mid = row_basis(view, exp, j, h);
        let turned = |s: f32| {
            let tau = exp.readout_s * ((j as f32 + 0.5) / h as f32 - 0.5) + s;
            basis_rpy(view.yaw + exp.rate[0] * tau, view.pitch + exp.rate[1] * tau, view.roll + exp.rate[2] * tau)
        };
        let (b0, b1) = (turned(-half), turned(half));
        let sy = (1.0 - 2.0 * (j as f32 + 0.5) / h as f32) * half_h;
        for i in 0..w {
            let sx = (2.0 * (i as f32 + 0.5) / w as f32 - 1.0) * half_w;
            let d = [mid.0[0] + mid.1[0] * sx + mid.2[0] * sy, mid.0[1] + mid.1[1] * sx + mid.2[1] * sy, mid.0[2] + mid.1[2] * sx + mid.2[2] * sy];
            let (Some(p0), Some(p1)) = (to_px(d, &b0), to_px(d, &b1)) else { continue };
            let len = (p1.0 - p0.0).hypot(p1.1 - p0.1);
            if len < 0.5 {
                continue;
            }
            let n = (len.ceil() as usize + 1).clamp(2, 32);
            let mut acc = [0.0f32; 3];
            for k in 0..n {
                let u = k as f32 / (n - 1) as f32;
                // The scene at this pixel during the exposure: the image as the
                // sensor sees it at the middle, offset by the path's displacement.
                let x = (i as f32 + 0.5 - (p0.0 + (p1.0 - p0.0) * u - (i as f32 + 0.5))).floor().clamp(0.0, (w - 1) as f32) as usize;
                let y = (j as f32 + 0.5 - (p0.1 + (p1.1 - p0.1) * u - (j as f32 + 0.5))).floor().clamp(0.0, (h - 1) as f32) as usize;
                let o = (y * w + x) * 3;
                for c in 0..3 {
                    acc[c] += src[o + c] as f32;
                }
            }
            for c in 0..3 {
                row[i * 3 + c] = (acc[c] / n as f32).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// An oriented box in world coordinates (the robot's links in the third-person view).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    pub center: [f32; 3],
    /// The box's axes as unit vectors (columns).
    pub axes: [[f32; 3]; 3],
    pub half: [f32; 3],
    pub color: [u8; 3],
}

impl Obb {
    /// Ray parameter and the local face axis of the nearest hit.
    fn hit(&self, o: [f32; 3], d: [f32; 3]) -> Option<(f32, usize)> {
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let rel = [o[0] - self.center[0], o[1] - self.center[1], o[2] - self.center[2]];
        let (mut t0, mut t1, mut axis) = (f32::NEG_INFINITY, f32::INFINITY, 0usize);
        for k in 0..3 {
            let (ok, dk) = (dot(self.axes[k], rel), dot(self.axes[k], d));
            if dk.abs() < 1e-9 {
                if ok.abs() > self.half[k] {
                    return None;
                }
                continue;
            }
            let (a, b) = ((-self.half[k] - ok) / dk, (self.half[k] - ok) / dk);
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if lo > t0 {
                t0 = lo;
                axis = k;
            }
            t1 = t1.min(hi);
            if t0 > t1 {
                return None;
            }
        }
        (t1 > 0.0 && t0 > 0.0).then_some((t0, axis))
    }
}

/// Where the third-person camera sits and where it looks.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ChaseView {
    pub origin: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
}

impl ChaseView {
    /// Behind and above the robot, `orbit_deg` around it (0 = behind), looking
    /// at its chest. Lifted out of the terrain when a hill is in the way.
    pub fn around(world: &World, orbit_deg: f32, dist_m: f32, height_m: f32) -> ChaseView {
        ChaseView::around_at(world, orbit_deg, dist_m, height_m, 0.75)
    }

    /// As [`ChaseView::around`], looking at `target_m` above the robot's ground.
    pub fn around_at(world: &World, orbit_deg: f32, dist_m: f32, height_m: f32, target_m: f32) -> ChaseView {
        let a = world.agent.yaw + std::f32::consts::PI + orbit_deg.to_radians();
        let ground = world.agent_ground_m();
        let (cx, cz) = (world.agent.x + dist_m * a.cos(), world.agent.z + dist_m * a.sin());
        let under = world.terrain.cell_of(cx, cz).map_or(0.0, |(x, z)| world.terrain.height_m(x, z) + world.terrain.curb_m(x, z));
        let origin = [cx, (ground + height_m).max(under + 0.4), cz];
        let target = [world.agent.x, ground + target_m, world.agent.z];
        let d = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
        ChaseView { origin, yaw: d[2].atan2(d[0]), pitch: d[1].atan2((d[0] * d[0] + d[2] * d[2]).sqrt()) }
    }

    /// Pixel and forward depth of a world point in this view.
    pub fn project(&self, cam: &Camera, point: [f32; 3]) -> Option<(f32, f32, f32)> {
        let c = Camera { pitch_deg: self.pitch.to_degrees(), ..*cam };
        c.project(self.origin, self.yaw, point)
    }
}

/// Renders the world from the chase camera with the robot's links drawn in.
/// The id buffer is 0 on the robot: this view is for people, not for the recogniser.
pub fn render_chase(world: &World, cam: &Camera, view: &ChaseView, robot: &[Obb]) -> Frame {
    render_with(world, cam, view.origin, view.yaw, view.pitch, robot)
}

/// Close-up cameras of the focus views: the working hand from its side, the
/// whole body from the side (waist, neck, legs), and low by the feet, the
/// wheels or the tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FocusCam {
    Hand,
    Side,
    Low,
}

impl FocusCam {
    pub const ALL: [FocusCam; 3] = [FocusCam::Hand, FocusCam::Side, FocusCam::Low];

    pub fn id(self) -> &'static str {
        match self {
            FocusCam::Hand => "hand",
            FocusCam::Side => "side",
            FocusCam::Low => "low",
        }
    }

    pub fn from_id(id: &str) -> Option<FocusCam> {
        FocusCam::ALL.into_iter().find(|c| c.id() == id)
    }

    /// Where the camera looks (base frame) and where it sits relative to that, metres.
    fn aim(self, pose: Option<&noto_core::dynamics::BodyPose>) -> ([f32; 3], [f32; 3]) {
        match self {
            FocusCam::Hand => {
                let palm = pose.and_then(|p| p.boxes.iter().find(|b| b.name == "palm_r")).map_or([0.3, -0.18, 0.7], |b| b.center);
                (palm, [-0.05, -0.6, 0.18])
            }
            FocusCam::Side => ([0.05, 0.0, 0.55], [0.15, -2.1, 0.25]),
            FocusCam::Low => ([0.12, 0.0, 0.10], [0.35, -1.0, 0.22]),
        }
    }
}

/// Renders a focus close-up of the robot as it is now (`World::body_pose`).
pub fn render_focus(world: &World, cam: &Camera, which: FocusCam) -> Frame {
    let pose = world.body_pose();
    let boxes = pose.as_ref().map(|p| world.robot_boxes(p)).unwrap_or_default();
    let (t, off) = which.aim(pose.as_ref());
    let target = world.body_to_world(t);
    let mut origin = world.body_to_world([t[0] + off[0], t[1] + off[1], t[2] + off[2]]);
    // Keep the camera out of the ground under it.
    let floor = world.surface_m(origin[0], origin[2]) + 0.08;
    if origin[1] < floor {
        origin[1] = floor;
    }
    let d = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
    let yaw = d[2].atan2(d[0]);
    let pitch = d[1].atan2((d[0] * d[0] + d[2] * d[2]).sqrt());
    render_with(world, cam, origin, yaw, pitch, &boxes)
}

fn render_with(world: &World, cam: &Camera, origin: [f32; 3], yaw: f32, pitch: f32, robot: &[Obb]) -> Frame {
    render_rays(world, cam, &Rays::Pinhole { view: View { origin, yaw, pitch, roll: 0.0 }, exposure: Exposure::default() }, robot, &[], None)
}

/// Distance behind an orthographic frame's centre its rays start from, metres
/// (far enough to lie outside the map or above everything in it; the depth
/// buffer of such a frame is measured from there).
pub const ORTHO_BACK_M: f32 = 100.0;

/// How a frame's rays are cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rays {
    /// A pinhole camera at the view (every camera before v1.0), each row read
    /// at its own instant of the exposure (rolling shutter).
    Pinhole { view: View, exposure: Exposure },
    /// An orthographic camera: parallel rays along the heading `yaw` and the
    /// `pitch` (radians, negative looks down) through a frame centred on
    /// `centre`, `half_h_m` metres from the centre to the top edge. No fog.
    Ortho { centre: [f32; 3], yaw: f32, pitch: f32, half_h_m: f32 },
}

/// What every pixel of a frame shares.
struct Pass<'a> {
    scene: &'a Scene<'a>,
    robot: &'a [Obb],
    casters: Casters<'a>,
    cut: Option<&'a Cutaway>,
    day: Option<Daylight>,
    max_dist: f32,
    fog_dist: f32,
    seed: u64,
}

impl Pass<'_> {
    /// Colour, id and depth of the ray from `o` along unit `d`; `along` is
    /// `d`'s share of the camera's forward axis (depth = distance × along).
    fn cast(&self, o: [f32; 3], d: [f32; 3], along: f32) -> ([u8; 3], u32, f32) {
        let world_hit = trace_cut(self.scene, o, d, self.max_dist, self.cut);
        let body = self.robot.iter().filter_map(|b| b.hit(o, d).map(|(t, ax)| (t, ax, b))).min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((t, ax, b)) = body.filter(|(t, _, _)| world_hit.as_ref().map_or(true, |h| *t < h.t)) {
            // Robot links: flat colour; faces shaded by which box axis was hit, or lit by the sun.
            let dim = 1.0 - 0.25 * (t / self.fog_dist).clamp(0.0, 1.0);
            let k = match &self.day {
                None => [[1.0, 0.86, 0.72][ax] * dim; 3],
                Some(day) => {
                    let a = b.axes[ax];
                    let s = if a[0] * d[0] + a[1] * d[1] + a[2] * d[2] > 0.0 { -1.0 } else { 1.0 };
                    let n = [a[0] * s, a[1] * s, a[2] * s];
                    let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                    let g = day.face_gain(n, sunlit(self.scene, day, p, n, &self.casters));
                    [g[0] * dim, g[1] * dim, g[2] * dim]
                }
            };
            let c = [0, 1, 2].map(|c| (b.color[c] as f32 * k[c]).clamp(0.0, 255.0) as u8);
            return (c, 0, t * along);
        }
        match world_hit {
            Some(hit) => {
                let c = match &self.day {
                    None => shade_hit(&hit, o, d, self.fog_dist, self.seed),
                    Some(day) => {
                        let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
                        let lit = sunlit(self.scene, day, p, hit.normal(), &self.casters);
                        shade_sun(&hit, p, d, self.fog_dist, self.seed, day, lit)
                    }
                };
                (c, hit.id, hit.t * along)
            }
            None => {
                let c = match &self.day {
                    None => sky(d[1]),
                    Some(day) => day.sky_px(d, sky(d[1])),
                };
                ([c[0] as u8, c[1] as u8, c[2] as u8], 0, f32::INFINITY)
            }
        }
    }
}

/// Renders a frame: `robot` boxes are drawn (id 0) and cast shadows,
/// `casters` only cast them (the robot's own links in first person), `cut`
/// clears the prims hiding the robot from an orthographic view.
pub fn render_rays(world: &World, cam: &Camera, rays: &Rays, robot: &[Obb], casters: &[Obb], cut: Option<&Cutaway>) -> Frame {
    let scene = Scene::build(world);
    let (w, h) = (cam.width.max(1), cam.height.max(1));
    let half_w = (cam.fov_deg.to_radians() * 0.5).tan();
    let half_h = half_w * h as f32 / w as f32;
    // A pinhole sees out to its range and fogs with distance; an orthographic
    // view starts its rays behind the frame and has no fog.
    let (max_dist, fog_dist) = match rays {
        Rays::Pinhole { .. } => (cam.max_dist_m, cam.max_dist_m),
        Rays::Ortho { .. } => (ORTHO_BACK_M + cam.max_dist_m, f32::INFINITY),
    };
    let day = world.cfg.lighting.daylight();
    let casters = Casters::new(if day.is_some() { [robot, casters] } else { [&[], &[]] });
    let pass = Pass { scene: &scene, robot, casters, cut, day, max_dist, fog_dist, seed: world.cfg.seed };

    let mut rgb = vec![0u8; w * h * 3];
    let mut ids = vec![0u32; w * h];
    let mut depth = vec![f32::INFINITY; w * h];

    rgb.par_chunks_mut(w * 3)
        .zip(ids.par_chunks_mut(w))
        .zip(depth.par_chunks_mut(w))
        .enumerate()
        .for_each(|(j, ((row_rgb, row_ids), row_depth))| {
            let mut put = |i: usize, (c, id, z): ([u8; 3], u32, f32)| {
                row_rgb[i * 3..i * 3 + 3].copy_from_slice(&c);
                row_ids[i] = id;
                row_depth[i] = z;
            };
            match rays {
                Rays::Pinhole { view, exposure } => {
                    let (fwd, right, up) = row_basis(view, exposure, j, h);
                    let sy = (1.0 - 2.0 * (j as f32 + 0.5) / h as f32) * half_h;
                    for i in 0..w {
                        let sx = (2.0 * (i as f32 + 0.5) / w as f32 - 1.0) * half_w;
                        let mut d = [
                            fwd[0] + right[0] * sx + up[0] * sy,
                            fwd[1] + right[1] * sx + up[1] * sy,
                            fwd[2] + right[2] * sx + up[2] * sy,
                        ];
                        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        d = [d[0] / len, d[1] / len, d[2] / len];
                        put(i, pass.cast(view.origin, d, d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2]));
                    }
                }
                Rays::Ortho { centre, yaw, pitch, half_h_m } => {
                    let (fwd, right, up) = basis(*yaw, *pitch);
                    let half_w_m = half_h_m * w as f32 / h as f32;
                    let sy = (1.0 - 2.0 * (j as f32 + 0.5) / h as f32) * half_h_m;
                    for i in 0..w {
                        let sx = (2.0 * (i as f32 + 0.5) / w as f32 - 1.0) * half_w_m;
                        let o = [0, 1, 2].map(|k| centre[k] + right[k] * sx + up[k] * sy - fwd[k] * ORTHO_BACK_M);
                        put(i, pass.cast(o, fwd, 1.0));
                    }
                }
            }
        });

    Frame { width: w, height: h, rgb, ids, depth, tick: world.tick }
}

/// Lists the nodes, drops and waypoints that appear in a frame, nearest first.
pub fn visible_objects(world: &World, frame: &Frame) -> Vec<VisibleObject> {
    struct Acc {
        px: u32,
        bbox: [u32; 4],
        depth_sum: f64,
    }
    let mut acc: HashMap<u32, Acc> = HashMap::new();
    for (idx, &raw) in frame.ids.iter().enumerate() {
        let (class, _) = unpack_id(raw);
        if !matches!(class, ObjectClass::Node | ObjectClass::Drop | ObjectClass::Waypoint) {
            continue;
        }
        let (x, y) = ((idx % frame.width) as u32, (idx / frame.width) as u32);
        let a = acc.entry(raw).or_insert(Acc { px: 0, bbox: [x, y, x, y], depth_sum: 0.0 });
        a.px += 1;
        a.bbox = [a.bbox[0].min(x), a.bbox[1].min(y), a.bbox[2].max(x), a.bbox[3].max(y)];
        a.depth_sum += frame.depth[idx] as f64;
    }
    let mut out: Vec<VisibleObject> = acc
        .into_iter()
        .map(|(raw, a)| {
            let (class, id) = unpack_id(raw);
            let (entity, label, label_ko) = match class {
                ObjectClass::Node => match world.node(id) {
                    Some(n) => (node_id(id), n.kind.id().to_string(), n.kind.spec().label_ko.to_string()),
                    None => (node_id(id), "unknown".into(), "알 수 없음".into()),
                },
                ObjectClass::Drop => match world.drop_item(id) {
                    Some(d) => (drop_id(id), d.kind.id().to_string(), d.kind.spec().label_ko.to_string()),
                    None => (drop_id(id), "unknown".into(), "알 수 없음".into()),
                },
                _ => match world.waypoint(id) {
                    Some(w) => (waypoint_id(id), w.key.clone(), w.label_ko.clone()),
                    None => (waypoint_id(id), "unknown".into(), "알 수 없음".into()),
                },
            };
            VisibleObject { class, entity, label, label_ko, pixels: a.px, bbox: a.bbox, depth_m: (a.depth_sum / a.px as f64) as f32 }
        })
        .collect();
    out.sort_by(|a, b| a.depth_m.total_cmp(&b.depth_m).then_with(|| a.entity.cmp(&b.entity)));
    out
}

impl World {
    pub fn render(&self, cam: &Camera) -> Frame {
        render(self, cam)
    }

    pub fn visible_objects(&self, frame: &Frame) -> Vec<VisibleObject> {
        visible_objects(self, frame)
    }

    pub fn render_top(&self, size_px: usize, span_m: f32) -> Frame {
        render_top(self, size_px, span_m)
    }

    /// A point of the robot's base frame (x forward, y left, z up from the
    /// ground under the robot) in world coordinates (y up).
    pub fn body_to_world(&self, p: [f32; 3]) -> [f32; 3] {
        let (s, c) = self.agent.yaw.sin_cos();
        let g = self.agent_ground_m();
        [self.agent.x + c * p[0] + s * p[1], g + p[2], self.agent.z + s * p[0] - c * p[1]]
    }

    /// A direction of the base frame in world coordinates.
    pub fn body_dir_to_world(&self, v: [f32; 3]) -> [f32; 3] {
        let (s, c) = self.agent.yaw.sin_cos();
        [c * v[0] + s * v[1], v[2], s * v[0] - c * v[1]]
    }

    /// The links of a body pose as world boxes, coloured by segment.
    pub fn robot_boxes(&self, pose: &noto_core::dynamics::BodyPose) -> Vec<Obb> {
        use noto_core::Segment;
        pose.boxes
            .iter()
            .map(|b| Obb {
                center: self.body_to_world(b.center),
                axes: [self.body_dir_to_world(b.axes[0]), self.body_dir_to_world(b.axes[1]), self.body_dir_to_world(b.axes[2])],
                half: b.half,
                color: match b.segment {
                    Segment::Head => [214, 218, 226],
                    Segment::Chest => [74, 110, 168],
                    Segment::Belly => [60, 72, 96],
                    Segment::Arm => [228, 136, 52],
                    Segment::Hand => [246, 206, 84],
                    Segment::Leg => [120, 126, 138],
                    Segment::Foot => [52, 56, 64],
                },
            })
            .collect()
    }

    /// The robot's links as shadow casters for its own camera: none unless
    /// the sun shines (the flat lighting has no shadows), the links with the
    /// chassis's tilt otherwise. At rest (no action, no posture change, not
    /// walking) the statics' standing pose (`noto_core::kin`) stands in for
    /// the joint dynamics' settled one (`World::body_pose`): settling the
    /// dynamics costs several frames' worth of time, and the two stand within
    /// a few centimetres of each other (`the_resting_shadow_pose_matches_the_dynamics`).
    pub fn shadow_casters(&self) -> Vec<Obb> {
        if !self.cfg.lighting.daylight().is_some_and(|d| d.has_beam()) {
            return Vec::new();
        }
        self.shadow_pose().map(|p| self.tilted_robot_boxes(&p, &self.base_tilt(&p))).unwrap_or_default()
    }

    /// The pose the robot's own shadow is cast from (see [`World::shadow_casters`]).
    fn shadow_pose(&self) -> Option<noto_core::dynamics::BodyPose> {
        let at_rest = self.act.is_none() && self.motion.is_none() && (!self.body.legged || self.agent.speed_mps <= 0.0);
        if !at_rest {
            return self.body_pose();
        }
        let stand = noto_core::kin::KinPose::stand(self.carried_kg());
        Some(noto_core::kin::solve(&self.body, &stand, &noto_core::kin::KinLoads::default()).body_pose(0.0, noto_core::Source::Quiet))
    }
}

fn put(rgb: &mut [u8], w: usize, h: usize, x: i32, y: i32, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
        let i = (y as usize * w + x as usize) * 3;
        rgb[i..i + 3].copy_from_slice(&c);
    }
}

/// Top-down map centred on the robot, `span_m` metres across. The id and depth
/// buffers are empty; this view is for people, not for the recogniser.
pub fn render_top(world: &World, size_px: usize, span_m: f32) -> Frame {
    let n = size_px.max(16);
    let t = &world.terrain;
    let m_per_px = span_m / n as f32;
    let mut rgb = vec![0u8; n * n * 3];
    for j in 0..n {
        for i in 0..n {
            let x = world.agent.x + (i as f32 + 0.5 - n as f32 * 0.5) * m_per_px;
            let z = world.agent.z + (j as f32 + 0.5 - n as f32 * 0.5) * m_per_px;
            let c = match t.cell_of(x, z) {
                None => [24, 24, 28],
                Some((cx, cz)) => {
                    // A cliff edge: a neighbour half a metre or more lower.
                    let h = t.height_m(cx, cz);
                    let cliff = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)].iter().any(|&(dx, dz)| {
                        let (nx, nz) = (cx as i32 + dx, cz as i32 + dz);
                        t.in_bounds(nx, nz) && h - t.height_m(nx as usize, nz as usize) >= 0.5
                    });
                    if t.curb_m(cx, cz) > 0.0 && t.curb_kind(cx, cz) == crate::terrain::CurbKind::Curb {
                        CURB_YELLOW
                    } else if t.curb_kind(cx, cz) != crate::terrain::CurbKind::Curb {
                        // A site's porch, stair or ramp: its floor, brighter the higher it stands.
                        let g = t.ground_at(cx, cz).color();
                        let k = 0.8 + 0.4 * t.curb_m(cx, cz).min(0.5);
                        [(g[0] as f32 * k).min(255.0) as u8, (g[1] as f32 * k).min(255.0) as u8, (g[2] as f32 * k).min(255.0) as u8]
                    } else if cliff {
                        [40, 38, 36]
                    } else {
                        let g = t.ground_at(cx, cz).color();
                        let k = 1.0 - (h * 0.35).min(0.35);
                        [(g[0] as f32 * k) as u8, (g[1] as f32 * k) as u8, (g[2] as f32 * k) as u8]
                    }
                }
            };
            put(&mut rgb, n, n, i as i32, j as i32, c);
        }
    }
    let to_px = |x: f32, z: f32| {
        (
            ((x - world.agent.x) / m_per_px + n as f32 * 0.5) as i32,
            ((z - world.agent.z) / m_per_px + n as f32 * 0.5) as i32,
        )
    };
    let dot = |rgb: &mut Vec<u8>, x: f32, z: f32, r: i32, c: [u8; 3]| {
        let (px, py) = to_px(x, z);
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    put(rgb, n, n, px + dx, py + dy, c);
                }
            }
        }
    };
    for nd in &world.nodes {
        let c = match nd.kind {
            NodeKind::Tree => [30, 90, 36],
            NodeKind::Boulder => [96, 96, 100],
            NodeKind::CoalOre => [20, 20, 24],
            NodeKind::IronOre => [190, 120, 90],
        };
        dot(&mut rgb, nd.x, nd.z, ((0.22 / m_per_px) as i32).max(2), c);
    }
    // A site's walls, shelves and tables over the ground (`crate::fixture`).
    for f in world.fixtures.iter().filter(|f| f.blocks()) {
        let c = match f.kind {
            crate::fixture::FixtureKind::Wall { .. } => [64, 60, 58],
            crate::fixture::FixtureKind::Shelf { .. } => [52, 92, 170],
            _ => [86, 104, 118],
        };
        let r = ((0.22 / m_per_px) as i32).max(1);
        for &(cx, cz) in &f.cells {
            let (x, z) = cell_center(cx, cz);
            let (px, py) = to_px(x, z);
            for dy in -r..=r {
                for dx in -r..=r {
                    put(&mut rgb, n, n, px + dx, py + dy, c);
                }
            }
        }
    }
    for d in &world.drops {
        dot(&mut rgb, d.x, d.z, ((0.1 / m_per_px) as i32).max(1), [255, 255, 255]);
    }
    for w in &world.waypoints {
        let c = if w.key == "warehouse" { [52, 104, 220] } else { [232, 120, 40] };
        dot(&mut rgb, w.x, w.z, ((0.3 / m_per_px) as i32).max(3), c);
        // The pile: one small square per piece, beside the marker.
        let pieces = stock_pieces(&w.stock);
        let r = ((0.06 / m_per_px) as i32).max(1);
        for (i, (k, _)) in pieces.iter().take(24).enumerate() {
            let (px, py) = to_px(w.x - 0.45 - (i / 6) as f32 * 0.16, w.z - 0.4 + (i % 6) as f32 * 0.16);
            for dy in -r..=r {
                for dx in -r..=r {
                    put(&mut rgb, n, n, px + dx, py + dy, k.spec().color);
                }
            }
        }
    }
    // Heading line and the robot itself.
    for s in 0..((0.9 / m_per_px) as i32).max(4) {
        let (px, py) = to_px(
            world.agent.x + world.agent.yaw.cos() * s as f32 * m_per_px,
            world.agent.z + world.agent.yaw.sin() * s as f32 * m_per_px,
        );
        put(&mut rgb, n, n, px, py, [255, 255, 255]);
    }
    dot(&mut rgb, world.agent.x, world.agent.z, ((0.18 / m_per_px) as i32).max(3), [220, 40, 40]);
    Frame { width: n, height: n, rgb, ids: vec![], depth: vec![], tick: world.tick }
}

/// Which ground kinds the renderer can draw; used by tests to make sure the
/// palette stays distinct.
pub fn ground_palette() -> [(Ground, [u8; 3]); 8] {
    Ground::ALL.map(|g| (g, g.color()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::Goal;
    use crate::world::{DropItem, WorldConfig};
    use noto_core::{BaseKind, FingerCount, NotoConfig};

    fn world() -> World {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::Track;
        World::new(WorldConfig { seed: 7, cells: 64, noto: hand, ..WorldConfig::default() }).unwrap()
    }

    fn small() -> Camera {
        Camera { width: 96, height: 64, ..Camera::default() }
    }

    #[test]
    fn world_point_inverts_project() {
        let mut w = world();
        w.agent.yaw = 0.7;
        let cam = Camera { width: 192, height: 128, ..Camera::default() };
        let origin = Camera::origin(&w);
        for &(dx, dy, dz) in &[(3.0f32, -0.4f32, 0.5f32), (5.0, -0.5, -1.2), (2.0, 0.0, 0.0), (6.0, -0.5, 2.5)] {
            let (c, s) = (w.agent.yaw.cos(), w.agent.yaw.sin());
            // a point in front of the robot, expressed along its heading
            let p = [origin[0] + dx * c - dz * s, origin[1] + dy, origin[2] + dx * s + dz * c];
            let (px, py, depth) = cam.project(origin, w.agent.yaw, p).expect("in front of the camera");
            let (x, z) = cam.world_point((origin[0], origin[2]), w.agent.yaw, px, py, depth).unwrap();
            assert!((x - p[0]).abs() < 1e-3 && (z - p[2]).abs() < 1e-3, "({x},{z}) vs ({},{})", p[0], p[2]);
            // The same with a rolled and pitched view.
            let v = View { origin, yaw: w.agent.yaw, pitch: -0.3, roll: 0.12 };
            let (px, py, depth) = cam.project_in(&v, p).expect("in front of the camera");
            let (x, z) = cam.world_point_in(&v, px, py, depth).unwrap();
            assert!((x - p[0]).abs() < 1e-3 && (z - p[2]).abs() < 1e-3, "rolled: ({x},{z}) vs ({},{})", p[0], p[2]);
        }
        assert!(cam.project(origin, w.agent.yaw, [origin[0] - 3.0 * w.agent.yaw.cos(), origin[1], origin[2] - 3.0 * w.agent.yaw.sin()]).is_none());
        assert!(cam.world_point((0.0, 0.0), 0.0, 10.0, 10.0, f32::INFINITY).is_none());
        assert!(cam.world_point((0.0, 0.0), 0.0, 10.0, 10.0, 0.0).is_none());
    }

    #[test]
    fn a_pixel_of_a_node_maps_back_onto_the_node() {
        let mut w = world();
        // Put a tree two cells ahead and look at it.
        w.nodes.clear();
        let (ax, az) = (w.agent.x, w.agent.z);
        let (cx, cz) = w.terrain.cell_of(ax + 3.2, az + 0.4).unwrap();
        let (nx, nz) = cell_center(cx, cz);
        w.nodes.push(crate::world::Node { id: 5, kind: NodeKind::Tree, x: nx, z: nz, hp: 3.0, max_hp: 3.0 });
        w.agent.yaw = 0.0;
        let cam = Camera { width: 192, height: 128, ..Camera::default() };
        let f = render(&w, &cam);
        let view = w.head_view(&cam).view;
        let mut checked = 0;
        for j in 0..f.height {
            for i in 0..f.width {
                let (class, id) = unpack_id(f.ids[j * f.width + i]);
                if class == ObjectClass::Node && id == 5 {
                    let (x, z) = cam.world_point_in(&view, i as f32 + 0.5, j as f32 + 0.5, f.depth[j * f.width + i]).unwrap();
                    // the surface point lies on the node's own cell
                    assert!((x - nx).abs() <= CELL_M * 0.5 + 0.02 && (z - nz).abs() <= CELL_M * 0.5 + 0.02, "({x},{z}) not on cell ({nx},{nz})");
                    checked += 1;
                }
            }
        }
        assert!(checked > 20, "the tree covers too few pixels: {checked}");
    }

    #[test]
    fn id_packing_round_trips() {
        let v = pack_id(ObjectClass::Drop, 0x00AB_CDEF);
        assert_eq!(unpack_id(v), (ObjectClass::Drop, 0x00AB_CDEF));
        assert_eq!(unpack_id(0), (ObjectClass::Sky, 0));
    }

    #[test]
    fn rendering_is_deterministic() {
        let w = world();
        let a = render(&w, &small());
        let b = render(&w, &small());
        assert_eq!(a.rgb, b.rgb);
        assert_eq!(a.ids, b.ids);
    }

    #[test]
    fn frame_has_sky_above_and_ground_below() {
        let w = world();
        let f = render(&w, &small());
        let top = unpack_id(f.ids[f.width / 2]).0;
        let bottom = unpack_id(f.ids[(f.height - 1) * f.width + f.width / 2]).0;
        assert_eq!(top, ObjectClass::Sky);
        assert_eq!(bottom, ObjectClass::Ground);
        assert!(f.depth[(f.height - 1) * f.width + f.width / 2].is_finite());
        assert!(f.depth[f.width / 2].is_infinite());
    }

    #[test]
    fn a_drop_ahead_on_the_right_appears_on_the_right_half() {
        let mut w = world();
        w.agent.yaw = 0.0; // looking along +x, the right-hand side is +z
        let (ax, az) = (w.agent.x, w.agent.z);
        let (cx, cz) = w.terrain.cell_of(ax + 2.5, az + 0.9).unwrap();
        let (x, z) = cell_center(cx, cz);
        w.drops.push(DropItem::whole(77, ItemKind::Stone, x, z));
        let f = render(&w, &Camera { width: 192, height: 128, ..Camera::default() });
        let seen = visible_objects(&w, &f);
        let o = seen.iter().find(|o| o.entity == "d77").expect("the drop is in view");
        let cx_px = (o.bbox[0] + o.bbox[2]) as f32 * 0.5;
        assert!(cx_px > f.width as f32 * 0.5, "right of centre, got x={cx_px}");
        assert!(o.depth_m > 2.0 && o.depth_m < 3.2, "depth {}", o.depth_m);
        // And the mirror case lands on the left.
        w.drops.clear();
        let (cx, cz) = w.terrain.cell_of(ax + 2.5, az - 0.9).unwrap();
        let (x, z) = cell_center(cx, cz);
        w.drops.push(DropItem::whole(78, ItemKind::Stone, x, z));
        let f = render(&w, &Camera { width: 192, height: 128, ..Camera::default() });
        let seen = visible_objects(&w, &f);
        let o = seen.iter().find(|o| o.entity == "d78").expect("the drop is in view");
        assert!(((o.bbox[0] + o.bbox[2]) as f32 * 0.5) < f.width as f32 * 0.5);
    }

    #[test]
    fn objects_behind_the_robot_are_not_seen() {
        let mut w = world();
        w.agent.yaw = 0.0;
        let (cx, cz) = w.terrain.cell_of(w.agent.x - 2.0, w.agent.z).unwrap();
        let (x, z) = cell_center(cx, cz);
        w.drops.push(DropItem::whole(79, ItemKind::Log, x, z));
        let f = render(&w, &small());
        assert!(visible_objects(&w, &f).iter().all(|o| o.entity != "d79"));
    }

    #[test]
    fn a_nearer_object_hides_a_farther_one() {
        let mut w = world();
        w.agent.yaw = 0.0;
        let (ax, az) = (w.agent.x, w.agent.z);
        for (id, dx) in [(80u32, 1.6f32), (81, 3.6)] {
            let (cx, cz) = w.terrain.cell_of(ax + dx, az).unwrap();
            let (x, z) = cell_center(cx, cz);
            w.nodes.push(crate::world::Node { id, kind: NodeKind::Boulder, x, z, hp: 7.5, max_hp: 7.5 });
        }
        let f = render(&w, &Camera { width: 160, height: 96, pitch_deg: -3.0, ..Camera::default() });
        let seen = visible_objects(&w, &f);
        let near = seen.iter().find(|o| o.entity == "n80").expect("near boulder");
        let far = seen.iter().find(|o| o.entity == "n81");
        assert!(near.pixels > 200);
        if let Some(far) = far {
            assert!(far.pixels < near.pixels / 2, "far {} vs near {}", far.pixels, near.pixels);
            assert!(far.depth_m > near.depth_m);
        }
    }

    #[test]
    fn nearer_objects_cover_more_pixels() {
        let mut w = world();
        w.agent.yaw = 0.0;
        let (ax, az) = (w.agent.x, w.agent.z);
        let mut area = |dx: f32, id: u32| {
            w.nodes.clear();
            let (cx, cz) = w.terrain.cell_of(ax + dx, az).unwrap();
            let (x, z) = cell_center(cx, cz);
            w.nodes.push(crate::world::Node { id, kind: NodeKind::CoalOre, x, z, hp: 15.0, max_hp: 15.0 });
            let f = render(&w, &Camera { width: 192, height: 128, pitch_deg: -3.0, ..Camera::default() });
            visible_objects(&w, &f).iter().find(|o| o.entity == node_id(id)).map_or(0, |o| o.pixels)
        };
        // From the head (1.14 m) the ground shows from about 2 m out at this pitch: both stand beyond that.
        let near = area(2.5, 90);
        let far = area(7.5, 91);
        assert!(near > far * 3 && far > 0, "near {near}, far {far}");
    }

    #[test]
    fn curbs_are_drawn_as_curbs_and_waypoint_flags_are_visible() {
        let mut w = world();
        // Stand in front of the wall looking at it.
        let wall_x = (0.46 * 64.0f32).round() * CELL_M;
        w.agent.x = wall_x - 2.0;
        w.agent.z = 0.28 * 32.0; // 5 cm / 10 cm sections
        w.agent.yaw = 0.0;
        let f = render(&w, &small());
        let curb_px = f.ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Curb).count();
        assert!(curb_px > 50, "curb pixels {curb_px}");

        let mut w = world();
        let wh = w.waypoint_by_key("warehouse").unwrap().clone();
        w.agent.x = wh.x - 3.0;
        w.agent.z = wh.z;
        w.agent.yaw = 0.0;
        let f = render(&w, &Camera { width: 192, height: 128, pitch_deg: 0.0, ..Camera::default() });
        let seen = visible_objects(&w, &f);
        assert!(seen.iter().any(|o| o.class == ObjectClass::Waypoint && o.label == "warehouse"));
    }

    #[test]
    fn mined_nodes_disappear_from_the_frame() {
        let mut w = world();
        w.agent.yaw = 0.0;
        let (cx, cz) = w.terrain.cell_of(w.agent.x + 2.0, w.agent.z).unwrap();
        let (x, z) = cell_center(cx, cz);
        w.nodes.push(crate::world::Node { id: 95, kind: NodeKind::Tree, x, z, hp: 3.0, max_hp: 3.0 });
        let f = render(&w, &small());
        assert!(visible_objects(&w, &f).iter().any(|o| o.entity == "n95"));
        w.nodes.retain(|n| n.id != 95);
        let f = render(&w, &small());
        assert!(visible_objects(&w, &f).iter().all(|o| o.entity != "n95"));
    }

    #[test]
    fn fog_and_texture_do_not_break_colour_identity() {
        // The mean colour of an ore node differs from a boulder's even with
        // flecks, so a recogniser has something to separate them.
        let mut w = world();
        w.agent.yaw = 0.0;
        w.nodes.clear();
        let mut mean = |kind: NodeKind, id: u32| {
            w.nodes.clear();
            let (cx, cz) = w.terrain.cell_of(w.agent.x + 1.5, w.agent.z).unwrap();
            let (x, z) = cell_center(cx, cz);
            w.nodes.push(crate::world::Node { id, kind, x, z, hp: 1.0, max_hp: 1.0 });
            let f = render(&w, &Camera { width: 128, height: 96, pitch_deg: -3.0, ..Camera::default() });
            let (mut s, mut n) = ([0f64; 3], 0f64);
            for (i, &v) in f.ids.iter().enumerate() {
                if v == pack_id(ObjectClass::Node, id) {
                    for c in 0..3 {
                        s[c] += f.rgb[i * 3 + c] as f64;
                    }
                    n += 1.0;
                }
            }
            assert!(n > 20.0);
            [s[0] / n, s[1] / n, s[2] / n]
        };
        let coal = mean(NodeKind::CoalOre, 1);
        let tree = mean(NodeKind::Tree, 2);
        let dist = ((coal[0] - tree[0]).powi(2) + (coal[1] - tree[1]).powi(2) + (coal[2] - tree[2]).powi(2)).sqrt();
        assert!(dist > 25.0, "coal {coal:?} tree {tree:?}");
    }

    #[test]
    fn webp_capture_of_a_real_frame_round_trips() {
        let w = world();
        let f = render(&w, &Camera { width: 192, height: 128, ..Camera::default() });
        let lossy = crate::capture::encode_webp(&f.rgb, 192, 128, 75.0).unwrap();
        let png_like = f.rgb.len();
        assert!(lossy.bytes.len() * 4 < png_like, "{} bytes vs {png_like} raw", lossy.bytes.len());
        let (cw, ch, back) = crate::capture::decode_webp(&lossy.bytes).unwrap();
        assert_eq!((cw as usize, ch as usize), (192, 128));
        assert!(crate::capture::psnr(&f.rgb, &back) > 24.0);
    }

    #[test]
    fn top_view_marks_the_robot_at_the_centre() {
        let w = world();
        let f = render_top(&w, 96, 24.0);
        let i = (48 * 96 + 48) * 3;
        assert_eq!(&f.rgb[i..i + 3], &[220, 40, 40]);
        assert_eq!(f.rgb.len(), 96 * 96 * 3);
    }

    #[test]
    fn a_delivered_pile_stands_beside_the_flag_and_is_not_a_detection() {
        let mut w = world();
        let camp = w.waypoint_by_key("base_camp").unwrap().clone();
        w.agent.x = camp.x + 2.0;
        w.agent.z = camp.z;
        w.agent.yaw = std::f32::consts::PI;
        let cam = Camera { width: 192, height: 128, pitch_deg: -8.0, ..Camera::default() };
        let stock_px = |w: &World| render(w, &cam).ids.iter().filter(|&&v| unpack_id(v).0 == ObjectClass::Stock).count();
        assert_eq!(stock_px(&w), 0, "an empty camp has no pile");
        let wp = w.waypoints.iter_mut().find(|p| p.key == "base_camp").unwrap();
        wp.stock = vec![(ItemKind::Log, 3.0), (ItemKind::Stone, 1.5)];
        assert!(stock_px(&w) > 30, "the pile shows");
        let f = render(&w, &cam);
        assert!(visible_objects(&w, &f).iter().all(|o| o.class != ObjectClass::Stock));
        // One piece per whole item plus the broken rest; the slots stay inside two cells.
        assert_eq!(stock_pieces(&[(ItemKind::Log, 3.0), (ItemKind::Stone, 1.5)]).len(), 5);
        assert!(pile_slots(100).len() == 40 && pile_slots(100).iter().all(|s| s.0 == 0 || s.0 == -1));
    }

    #[test]
    fn the_sea_renders_as_water_and_hides_nothing_behind_it() {
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::Track;
        let w = World::new(WorldConfig { seed: 4, cells: 64, noto: hand, terrain: crate::terrain::TerrainKind::Rugged, ..WorldConfig::default() }).unwrap();
        let t = &w.terrain;
        let water = (0..t.d).flat_map(|z| (0..t.w).map(move |x| (x, z))).filter(|&(x, z)| t.is_water(x, z)).count();
        assert!(water > 0);
        let top = render_top(&w, 128, 70.0);
        let blue = top.rgb.chunks(3).filter(|c| c[2] as i32 > c[0] as i32 + 60 && c[2] > 140).count();
        assert!(blue > 0, "the top view shows the sea");
    }

    /// FNV-1a over a frame's colour, id and depth buffers.
    fn frame_hash(f: &Frame) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        f.rgb.iter().for_each(|&b| eat(b));
        f.ids.iter().flat_map(|v| v.to_le_bytes()).for_each(&mut eat);
        f.depth.iter().flat_map(|v| v.to_bits().to_le_bytes()).for_each(&mut eat);
        h
    }

    /// The frames of the default (flat) lighting, one per renderer path:
    /// first person at rest, turning (blur and rolling shutter), walking
    /// (shake), on the rugged map (water), the chase and focus views, the top view.
    fn flat_reference_frames() -> Vec<(&'static str, Frame)> {
        let mut out = Vec::new();
        let w = world();
        out.push(("rest", render(&w, &small())));
        let mut t = world();
        let yaw = t.agent.yaw + 1.5;
        t.command(crate::world::Action::Turn { yaw_deg: yaw.to_degrees() }).unwrap();
        t.tick();
        out.push(("turn", render(&t, &Camera { width: 128, height: 96, ..Camera::default() })));
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::HumanFoot;
        let mut legs = World::new(WorldConfig { seed: 7, cells: 64, noto: hand, parcels: Some(0), ..WorldConfig::default() }).unwrap();
        let (ax, az) = (legs.agent.x, legs.agent.z);
        let goal = (0..16)
            .map(|k| (k as f32 * std::f32::consts::TAU / 16.0).sin_cos())
            .map(|(s, c)| crate::world::Action::MoveTo { x: ax + 5.0 * c, z: az + 5.0 * s, reach_m: 0.3 })
            .find(|a| legs.clone().command(a.clone()).is_ok())
            .expect("a reachable point 5 m away");
        legs.command(goal).unwrap();
        for _ in 0..7 {
            legs.tick();
        }
        out.push(("walk", render(&legs, &Camera { width: 128, height: 96, ..Camera::default() })));
        let mut hand = NotoConfig::preset(FingerCount::Five);
        hand.base = BaseKind::Track;
        let rugged = World::new(WorldConfig { seed: 4, cells: 64, noto: hand, terrain: crate::terrain::TerrainKind::Rugged, ..WorldConfig::default() }).unwrap();
        out.push(("rugged", render(&rugged, &Camera { width: 128, height: 96, pitch_deg: -4.0, ..Camera::default() })));
        let pose = w.body_pose().expect("a pose");
        let boxes = w.robot_boxes(&pose);
        let cam = Camera { width: 128, height: 96, fov_deg: 64.0, max_dist_m: 30.0, ..Camera::default() };
        out.push(("chase", render_chase(&w, &cam, &ChaseView::around(&w, 20.0, 2.6, 1.6), &boxes)));
        out.push(("focus", render_focus(&w, &Camera { width: 96, height: 64, fov_deg: 50.0, max_dist_m: 12.0, ..Camera::default() }, FocusCam::Hand)));
        out.push(("top", render_top(&w, 64, 24.0)));
        out
    }

    #[test]
    fn flat_lighting_renders_the_pixels_of_v09() {
        // Hashes of these frames taken from the v0.9 renderer before lighting existed.
        let want: [(&str, u64); 7] = [
            ("rest", 0x72c1_1091_ad87_f209),
            ("turn", 0xd6f8_4e05_c909_f65a),
            ("walk", 0xe86e_d1c6_773e_3ffc),
            ("rugged", 0xd0ae_544e_dcbd_f643),
            ("chase", 0x9450_bf96_efbd_49ea),
            ("focus", 0x4633_c830_3896_1054),
            ("top", 0x1777_dd2f_41f0_963c),
        ];
        let frames = flat_reference_frames();
        assert_eq!(frames.len(), want.len());
        for ((name, f), (wname, h)) in frames.iter().zip(want) {
            assert_eq!(*name, wname);
            assert_eq!(frame_hash(f), h, "{name}: the flat lighting must keep the v0.9 pixels");
        }
    }

    fn luma(c: &[u8]) -> f32 {
        0.2126 * c[0] as f32 + 0.7152 * c[1] as f32 + 0.0722 * c[2] as f32
    }

    #[test]
    fn a_face_turned_to_the_sun_is_brighter_than_one_turned_away() {
        let mut w = world();
        w.nodes.clear();
        w.agent.yaw = 0.0;
        let (cx, cz) = w.terrain.cell_of(w.agent.x + 2.5, w.agent.z).unwrap();
        let (x, z) = cell_center(cx, cz);
        w.nodes.push(crate::world::Node { id: 9, kind: NodeKind::Boulder, x, z, hp: 7.5, max_hp: 7.5 });
        let cam = Camera { width: 160, height: 120, pitch_deg: -8.0, ..Camera::default() };
        // Mean brightness of the boulder's west face (its lower block's −x side).
        let west_face = |w: &World| {
            let f = render(w, &cam);
            let view = w.head_view(&cam).view;
            let (mut s, mut n) = (0.0f32, 0);
            for (i, &id) in f.ids.iter().enumerate() {
                if id != pack_id(ObjectClass::Node, 9) {
                    continue;
                }
                let p = cam.world_point3_in(&view, (i % f.width) as f32 + 0.5, (i / f.width) as f32 + 0.5, f.depth[i]).unwrap();
                if (p[0] - (x - 0.24)).abs() < 0.03 {
                    s += luma(&f.rgb[i * 3..i * 3 + 3]);
                    n += 1;
                }
            }
            assert!(n > 30, "{n} west-face pixels");
            s / n as f32
        };
        let flat = west_face(&w);
        // From the south-west the sun shines on the west face; from the north-east it is in shade.
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 45.0, azimuth_deg: 240.0, overcast: 0.0 });
        let lit = west_face(&w);
        assert_eq!(render(&w, &cam).rgb, render(&w, &cam).rgb, "sun renders are deterministic");
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 45.0, azimuth_deg: 60.0, overcast: 0.0 });
        let shade = west_face(&w);
        assert!(lit > 1.5 * shade, "lit {lit} shade {shade} (flat {flat})");
        // Overcast: the beam is gone and the two agree.
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 45.0, azimuth_deg: 240.0, overcast: 1.0 });
        let grey_a = west_face(&w);
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 45.0, azimuth_deg: 60.0, overcast: 1.0 });
        assert!((grey_a - west_face(&w)).abs() < 1.0);
    }

    #[test]
    fn a_tree_casts_its_shadow_away_from_the_sun() {
        let mut w = world();
        w.nodes.clear();
        w.agent.yaw = std::f32::consts::FRAC_PI_2; // looking along +z (south); east (+x) is on the left
        let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z + 4.0).unwrap();
        let (tx, tz) = cell_center(cx, cz);
        let tree = crate::world::Node { id: 4, kind: NodeKind::Tree, x: tx, z: tz, hp: 3.0, max_hp: 3.0 };
        w.nodes.push(tree.clone());
        // A ground point 1.3 m east of the trunk: under the canopy's shadow when the sun is in the west at 40°.
        let p = [tx + 1.3, w.surface_m(tx + 1.3, tz), tz];
        let cam = Camera { width: 160, height: 120, ..Camera::default() };
        let at = |w: &World| {
            let f = render(w, &cam);
            let (px, py, _) = cam.project_in(&w.head_view(&cam).view, p).expect("in front");
            let (px, py) = (px as usize, py as usize);
            let mut s = 0.0;
            for j in py - 1..=py + 1 {
                for i in px - 1..=px + 1 {
                    let k = j * f.width + i;
                    assert_eq!(unpack_id(f.ids[k]).0, ObjectClass::Ground);
                    s += luma(&f.rgb[k * 3..k * 3 + 3]);
                }
            }
            s / 9.0
        };
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 40.0, azimuth_deg: 270.0, overcast: 0.0 });
        let shadowed = at(&w);
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 40.0, azimuth_deg: 90.0, overcast: 0.0 });
        let sunny = at(&w);
        assert!(shadowed < 0.6 * sunny, "shadow {shadowed} vs sun {sunny}");
        // It is the tree's shadow: without the tree the point is in the sun.
        w.set_lighting(crate::light::Lighting::Sun { elevation_deg: 40.0, azimuth_deg: 270.0, overcast: 0.0 });
        w.nodes.clear();
        assert!(at(&w) > 0.9 * sunny);
    }

    #[test]
    fn a_shadow_is_as_long_as_the_sun_is_low() {
        // A tree on level ground seen straight down: with the sun in the west
        // at elevation h, the canopy's top (1.6 m) throws its shadow H/tan h
        // past the canopy's east face (half width 0.24 m).
        let mut w = world();
        w.nodes.clear();
        w.drops.clear();
        w.waypoints.clear();
        let (cx, cz) = w.terrain.cell_of(w.agent.x, w.agent.z + 4.0).unwrap();
        for z in cz - 3..=cz + 3 {
            for x in cx - 2..=cx + 10 {
                w.terrain.set_height_cm(x, z, 0);
                w.terrain.set_curb_cm(x, z, 0);
                w.terrain.set_ground(x, z, Ground::Grass);
            }
        }
        let (tx, tz) = cell_center(cx, cz);
        let bare = w.clone();
        w.nodes.push(crate::world::Node { id: 5, kind: NodeKind::Tree, x: tx, z: tz, hp: 3.0, max_hp: 3.0 });
        // Straight down: screen up is +x, screen right is +z; 2.5 cm pixels.
        let cam = Camera { width: 200, height: 200, max_dist_m: 30.0, ..Camera::default() };
        let rays = Rays::Ortho { centre: [tx + 1.5, 0.0, tz], yaw: 0.0, pitch: -std::f32::consts::FRAC_PI_2, half_h_m: 2.5 };
        for h in [30.0f32, 45.0, 60.0] {
            let sun = crate::light::Lighting::Sun { elevation_deg: h, azimuth_deg: 270.0, overcast: 0.0 };
            let (mut tw, mut bw) = (w.clone(), bare.clone());
            tw.set_lighting(sun);
            bw.set_lighting(sun);
            let (with, without) = (render_rays(&tw, &cam, &rays, &[], &[], None), render_rays(&bw, &cam, &rays, &[], &[], None));
            // Down the column through the trunk, the farthest ground pixel darker than without the tree.
            let i = cam.width / 2;
            let tip = (0..cam.height)
                .filter(|&j| {
                    let k = j * cam.width + i;
                    unpack_id(with.ids[k]).0 == ObjectClass::Ground && luma(&with.rgb[k * 3..k * 3 + 3]) < 0.75 * luma(&without.rgb[k * 3..k * 3 + 3])
                })
                .map(|j| 1.5 + (1.0 - 2.0 * (j as f32 + 0.5) / cam.height as f32) * 2.5)
                .fold(f32::NEG_INFINITY, f32::max);
            let want = 0.24 + 1.6 / h.to_radians().tan();
            assert!((tip - want).abs() < 0.06, "h {h}: shadow ends {tip:.3} m past the trunk, geometry says {want:.3} m");
        }
    }

    #[test]
    fn the_resting_shadow_pose_matches_the_dynamics() {
        // At rest the statics' standing pose casts the robot's own shadow in
        // place of the dynamics' settled one: the body above the feet or the
        // chassis (the part whose shadow reaches the ground ahead) must stand
        // where the dynamics put it.
        let span = |pose: &noto_core::dynamics::BodyPose, w: &World| {
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for (o, b) in w.robot_boxes(pose).iter().zip(&pose.boxes) {
                if b.segment == noto_core::Segment::Foot {
                    continue;
                }
                for i in 0..3 {
                    let e: f32 = (0..3).map(|k| (o.axes[k][i] * o.half[k]).abs()).sum();
                    lo[i] = lo[i].min(o.center[i] - e);
                    hi[i] = hi[i].max(o.center[i] + e);
                }
            }
            (lo, hi)
        };
        for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track] {
            let mut hand = NotoConfig::preset(FingerCount::Five);
            hand.base = base;
            let w = World::new(WorldConfig { seed: 7, cells: 48, noto: hand, parcels: Some(0), ..WorldConfig::default() }).unwrap();
            assert!(w.act.is_none() && w.motion.is_none(), "a new world's robot is at rest");
            let ((a0, a1), (b0, b1)) = (span(&w.shadow_pose().unwrap(), &w), span(&w.body_pose().unwrap(), &w));
            let worst = (0..3).map(|i| (a0[i] - b0[i]).abs().max((a1[i] - b1[i]).abs())).fold(0.0f32, f32::max);
            println!("{base:?}: statics vs dynamics body bounds differ by at most {worst:.3} m");
            assert!(worst < 0.1, "{base:?}: {a0:?}..{a1:?} vs {b0:?}..{b1:?}");
        }
    }

    #[test]
    fn the_cutaway_cuts_only_what_can_hide_the_robot() {
        use std::f32::consts::FRAC_1_SQRT_2 as H;
        // The robot stands at (10, 10) on ground 1.0 m; the camera looks from
        // the −x −z diagonal down 35.264°, so sight lines rise 1/√2 per metre.
        let cut = Cutaway { at: [10.0, 10.0], toward_cam: [-H, -H], half_width_m: 0.8, slope: H, base_y: 1.0, low_y: 1.5, roof_radius_m: 0.8, roof_y: 2.6 };
        // A 0.5 m post `a` metres toward the camera and `l` to the side.
        let post = |a: f32, l: f32, y0: f32, y1: f32| bx(10.0 - a * H + l * H, 10.0 - a * H - l * H, y0, y1, 0.25, 0.25);
        // 3 m toward the camera, 2.5 m tall: cut to the sight line through the robot's feet at its near face.
        let near_face = 3.0 - 2.0 * 0.25 * H;
        let b = cut.clip(&post(3.0, 0.0, 1.0, 3.5)).unwrap();
        assert!((b.max[1] - (1.0 + near_face * H)).abs() < 1e-4 && b.min[1] == 1.0, "{:?}", b.max);
        // Right in front: down to the floor of the cut; a parcel there stays whole.
        assert!((cut.clip(&post(0.5, 0.0, 1.0, 2.0)).unwrap().max[1] - 1.5).abs() < 1e-6);
        assert_eq!(cut.clip(&post(0.4, 0.0, 1.0, 1.3)).unwrap().max[1], 1.3);
        // Behind the robot or well to its side: nothing hides it, nothing is cut.
        for (a, l) in [(-3.0, 0.0), (1.5, 3.0), (-1.5, -2.0)] {
            let b = post(a, l, 1.0, 3.0);
            let c = cut.clip(&b).unwrap();
            assert!(c.min == b.min && c.max == b.max, "({a}, {l}) cut to {:?}", c.max);
        }
        // A shelf just behind it reaching over its head: cut above the robot.
        assert!((cut.clip(&post(-0.6, 0.0, 1.0, 3.0)).unwrap().max[1] - 2.6).abs() < 1e-6);
        // A roof over the robot is gone.
        assert!(cut.clip(&Aabb { min: [5.0, 3.0, 5.0], max: [15.0, 3.2, 15.0] }).is_none());
    }

    #[test]
    fn a_ray_from_outside_the_map_enters_it() {
        let w = world();
        let scene = Scene::build(&w);
        let (gx, gz) = (w.agent.x, w.agent.z);
        // From 20 m west of the map's edge and high above it, aimed at the robot's cell.
        let o = [-20.0, 30.0, gz];
        let d = {
            let v = [gx - o[0], w.agent_ground_m() - o[1], gz - o[2]];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let hit = trace(&scene, o, d, 200.0).expect("the ray reaches the map");
        let p = [o[0] + d[0] * hit.t, o[1] + d[1] * hit.t, o[2] + d[2] * hit.t];
        assert!(p[0] >= 0.0 && (p[1] - w.agent_ground_m()).abs() < 0.3, "{p:?}");
        // A ray past the map misses it; one leaving upward from above everything too.
        assert!(trace(&scene, [-20.0, 30.0, -5.0], [0.0, -0.6, -0.8], 200.0).is_none());
        assert!(trace(&scene, [gx, 40.0, gz], [0.0, 1.0, 0.0], 200.0).is_none());
        // The face a ray enters points back at it.
        let down = trace(&scene, [gx, 5.0, gz], [0.0, -1.0, 0.0], 50.0).unwrap();
        assert_eq!((down.axis, down.sign), (1, 1.0));
    }

    #[test]
    fn ground_palette_is_distinct() {
        let p = ground_palette();
        for i in 0..p.len() {
            for j in i + 1..p.len() {
                let d: i32 = (0..3).map(|k| (p[i].1[k] as i32 - p[j].1[k] as i32).abs()).sum();
                assert!(d > 40, "{:?} vs {:?}", p[i].0, p[j].0);
            }
        }
        let _ = Goal::Cell((0, 0));
    }
}
