use serde::Serialize;

use crate::balance::{Body, G};
use crate::base::BaseKind;
use crate::kin::{self, KinLoads, KinPose, KinReport};
use crate::parts::*;
use crate::surface::{self, Surface};

use super::collide::{self, ground_pair, joint_hits, pulse, rot2, shock_events, solve_impulses, Contactor, JointHit, Obstacle, PairLaw, PhysKind, PhysRollout, PivotMode, Planar, Pulse, Rigid2, Row, TipStart};
use super::{ActEvent, ActSample, Outcome, Part, SAMPLE_S};

const DT_S: f32 = 0.0005;
pub const REST_MPS: f32 = 0.05;
const HELD_S: f32 = 0.02;
const SINK_M: f32 = 1e-5;
pub const PLASTIC_MPS: f32 = 0.25;
const STAY_MPS: f32 = 0.01;
pub const HARD_MPS: f32 = 0.5;
pub const FALL_MAX_S: f32 = 4.0;
const MAX_STRIKES: usize = 80;
const CALM_MPS: f32 = 0.02;
const CALM_S: f32 = 0.05;
const HELD_GAP_S: f32 = 0.005;
const CONE_TOL: f32 = 0.02;
pub const SHELL_GROUND_MU: f32 = 0.5;
pub const GET_UP_LEGS_S: f32 = 6.0;
pub const FALL_DIR_COUNTS: [(InitialDir, u32); 4] = [(InitialDir::Forward, 51), (InitialDir::Backward, 74), (InitialDir::Sideways, 55), (InitialDir::Down, 19)];
pub const SIDEWAYS_TO_BACK: f32 = 0.36;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallCause {
    Pushed,
    Bumped,
    Tripped,
    Slipped,
    Hit,
    Edge,
    Overload,
}

impl FallCause {
    pub fn id(self) -> &'static str {
        match self {
            FallCause::Pushed => "pushed",
            FallCause::Bumped => "bumped",
            FallCause::Tripped => "tripped",
            FallCause::Slipped => "slipped",
            FallCause::Hit => "hit",
            FallCause::Edge => "edge",
            FallCause::Overload => "overload",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            FallCause::Pushed => "밀려서",
            FallCause::Bumped => "부딪혀서",
            FallCause::Tripped => "발이 걸려서",
            FallCause::Slipped => "미끄러져서",
            FallCause::Hit => "맞아서",
            FallCause::Edge => "모서리에서 떨어져서",
            FallCause::Overload => "짐을 못 이겨서",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialDir {
    Forward,
    Backward,
    Sideways,
    Down,
}

pub fn initial_direction(u: f32) -> InitialDir {
    let total: u32 = FALL_DIR_COUNTS.iter().map(|c| c.1).sum();
    let mut acc = 0.0f32;
    for (d, n) in FALL_DIR_COUNTS {
        acc += n as f32 / total as f32;
        if u < acc {
            return d;
        }
    }
    InitialDir::Down
}

pub fn direction_deg(d: InitialDir, side_u: f32) -> f32 {
    let side = if side_u < 0.5 { 90.0 } else { -90.0 };
    match d {
        InitialDir::Forward => 0.0,
        InitialDir::Backward => 180.0,
        InitialDir::Sideways => {
            if (side_u * 2.0).fract() < SIDEWAYS_TO_BACK {
                if side > 0.0 {
                    135.0
                } else {
                    -135.0
                }
            } else {
                side
            }
        }
        InitialDir::Down => 180.0,
    }
}

#[derive(Clone, Debug)]
pub struct FallInput<'a> {
    pub body: &'a Body,
    pub pose: KinPose,
    pub cause: FallCause,
    pub dir_deg: f32,
    pub tilt0_deg: f32,
    pub omega0: f32,
    pub speed_mps: f32,
    pub pivot: Option<[f32; 2]>,
    pub drop_m: f32,
    pub edge_m: f32,
    pub ground: Surface,
}

impl<'a> FallInput<'a> {
    pub fn from_tip(body: &'a Body, pose: KinPose, tip: &TipStart, cause: FallCause, ground: Surface) -> FallInput<'a> {
        FallInput { body, pose, cause, dir_deg: tip.dir_deg, tilt0_deg: tip.tilt_deg, omega0: tip.omega, speed_mps: 0.0, pivot: Some(tip.pivot), drop_m: 0.0, edge_m: 0.0, ground }
    }

    pub fn lost_balance(body: &'a Body, pose: KinPose, cause: FallCause, dir_deg: f32, omega0: f32, ground: Surface) -> FallInput<'a> {
        let solved = kin::solve(body, &pose, &KinLoads::default());
        let pl = Planar::of(&collide::mass_props_of(body, &solved), solved.support, dir_of(dir_deg));
        FallInput { body, pose, cause, dir_deg, tilt0_deg: pl.tip_angle(pl.ahead_m).to_degrees(), omega0, speed_mps: 0.0, pivot: None, drop_m: 0.0, edge_m: 0.0, ground }
    }

    pub fn over_edge(body: &'a Body, pose: KinPose, dir_deg: f32, speed_mps: f32, drop_m: f32, ground: Surface) -> FallInput<'a> {
        let tilt0_deg = if speed_mps.abs() > 0.05 {
            0.5
        } else {
            let solved = kin::solve(body, &pose, &KinLoads::default());
            let pl = Planar::of(&collide::mass_props_of(body, &solved), solved.support, dir_of(dir_deg));
            pl.tip_angle(pl.ahead_m).to_degrees()
        };
        FallInput { body, pose, cause: FallCause::Edge, dir_deg, tilt0_deg, omega0: 0.05, speed_mps, pivot: None, drop_m, edge_m: 0.0, ground }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Strike {
    pub t: f32,
    pub tilt_deg: f32,
    pub link: &'static str,
    pub segment: Segment,
    pub contactor: Contactor,
    pub point: [f32; 3],
    pub wall: bool,
    pub face: bool,
    pub edge: bool,
    pub v_n: f32,
    pub impulse_ns: f32,
    pub m_eff_kg: f32,
    pub peak_n: f32,
    pub contact_s: f32,
    pub a_point_g: f32,
    pub a_trunk_g: f32,
    pub energy_j: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LyingPose {
    pub pose: KinPose,
    pub dir_deg: f32,
    pub tilt_deg: f32,
    pub com: [f32; 3],
    pub com0: [f32; 3],
    pub pivot0: [f32; 3],
    c_rest: [f32; 2],
}

impl LyingPose {
    pub fn place(&self, p: [f32; 3]) -> [f32; 3] {
        let d = dir_of(self.dir_deg);
        let perp = [-d[1], d[0]];
        let rel = [p[0] - self.com0[0], p[1] - self.com0[1], p[2] - self.com0[2]];
        let q = [rel[0] * d[0] + rel[1] * d[1], rel[2]];
        let lat = rel[0] * perp[0] + rel[1] * perp[1];
        let r = rot2(q, self.tilt_deg.to_radians());
        let u = self.c_rest[0] + r[0];
        let z = self.c_rest[1] + r[1];
        [self.pivot0[0] + d[0] * u + perp[0] * lat, self.pivot0[1] + d[1] * u + perp[1] * lat, z]
    }

    pub fn down(&self) -> bool {
        self.tilt_deg.abs() >= 45.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GetUp {
    pub can: bool,
    pub time_s: Option<f32>,
    pub needs_help: bool,
    pub righting_force_n: f32,
    pub why_ko: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fall {
    pub rollout: PhysRollout,
    pub cause: FallCause,
    pub dir_deg: f32,
    pub fell: bool,
    pub t_ground_s: f32,
    pub t_first_s: f32,
    pub strikes: Vec<Strike>,
    pub main: Option<Strike>,
    pub pelvis_v: Option<f32>,
    pub head_v: Option<f32>,
    pub hand_v: Option<f32>,
    pub tilt: Vec<(f32, f32)>,
    pub lying: LyingPose,
    pub damage: Vec<JointHit>,
    pub get_up: GetUp,
    pub at_rest: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Probe {
    pub q: [f32; 2],
    pub point: [f32; 3],
    pub link: &'static str,
    pub segment: Segment,
    pub contactor: Contactor,
}

pub(crate) fn contactor_of(body: &Body, link: &str) -> Contactor {
    match link {
        "bumper" => Contactor::Bumper,
        "wheel" | "track" | "support" => {
            if body.base == BaseKind::HumanFoot {
                Contactor::Sole
            } else {
                Contactor::Tyre
            }
        }
        l if l.starts_with("foot") => Contactor::Sole,
        _ => Contactor::Shell,
    }
}

pub(crate) fn probes(body: &Body, r: &KinReport, pl: &Planar) -> Vec<Probe> {
    let mut v: Vec<Probe> = Vec::new();
    for b in &r.boxes {
        if matches!(b.name, "wheel" | "track" | "finger_r" | "finger_l") {
            continue;
        }
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let mut p = b.center;
                    for (k, s) in [sx, sy, sz].iter().enumerate() {
                        for i in 0..3 {
                            p[i] += b.axes[k][i] * s * b.half[k];
                        }
                    }
                    let q = pl.to_plane(p);
                    if v.iter().any(|o| o.link == b.name && (o.q[0] - q[0]).abs() < 2e-3 && (o.q[1] - q[1]).abs() < 2e-3) {
                        continue;
                    }
                    v.push(Probe { q, point: p, link: b.name, segment: b.segment, contactor: contactor_of(body, b.name) });
                }
            }
        }
    }
    for u in [pl.ahead_m, -pl.behind_m] {
        let p = [pl.com[0] + pl.dir[0] * u, pl.com[1] + pl.dir[1] * u, 0.0];
        v.push(Probe { q: [u, -pl.com_z], point: p, link: "support", segment: Segment::Foot, contactor: contactor_of(body, "support") });
    }
    v
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hull {
    pub pts: Vec<[f32; 2]>,
    pub link: &'static str,
    pub segment: Segment,
    pub contactor: Contactor,
    origin: [f32; 3],
    dir: [f32; 2],
    lat: f32,
}

impl Hull {
    fn out(&self, k: usize) -> [f32; 2] {
        let (a, b) = (self.pts[k], self.pts[(k + 1) % self.pts.len()]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
        [d[1] / l, -d[0] / l]
    }

    fn inside(&self, q: [f32; 2]) -> Option<(usize, f32, [f32; 2])> {
        if self.pts.len() < 3 {
            return None;
        }
        let mut near: Option<(usize, f32, [f32; 2])> = None;
        for k in 0..self.pts.len() {
            let o = self.out(k);
            let a = self.pts[k];
            let s = (q[0] - a[0]) * o[0] + (q[1] - a[1]) * o[1];
            if s >= 0.0 {
                return None;
            }
            if near.map_or(true, |x| -s < x.1) {
                near = Some((k, -s, o));
            }
        }
        near
    }

    fn point(&self, q: [f32; 2]) -> [f32; 3] {
        let perp = [-self.dir[1], self.dir[0]];
        [self.origin[0] + self.dir[0] * q[0] + perp[0] * self.lat, self.origin[1] + self.dir[1] * q[0] + perp[1] * self.lat, self.origin[2] + q[1]]
    }

    fn probe_at(&self, q: [f32; 2]) -> Probe {
        Probe { q, point: self.point(q), link: self.link, segment: self.segment, contactor: self.contactor }
    }
}

fn convex_hull(mut pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-9 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-9 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

const DISC_SIDES: usize = 32;

pub(crate) fn hulls(body: &Body, r: &KinReport, pl: &Planar) -> Vec<Hull> {
    let perp = [-pl.dir[1], pl.dir[0]];
    let mut v = Vec::new();
    for b in &r.boxes {
        if matches!(b.name, "finger_r" | "finger_l") {
            continue;
        }
        let at = |o: [f32; 3]| -> [f32; 3] {
            let mut p = b.center;
            for (k, s) in o.iter().enumerate() {
                for i in 0..3 {
                    p[i] += b.axes[k][i] * s;
                }
            }
            p
        };
        let mut pts3: Vec<[f32; 3]> = Vec::new();
        if matches!(b.name, "wheel" | "track") {
            let rad = b.half[2];
            let run = (b.half[0] - rad).max(0.0);
            for end in [-run, run] {
                for k in 0..DISC_SIDES {
                    let a = std::f32::consts::TAU * k as f32 / DISC_SIDES as f32;
                    for sy in [-1.0f32, 1.0] {
                        pts3.push(at([end + rad * a.cos(), sy * b.half[1], rad * a.sin()]));
                    }
                }
            }
        } else {
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        pts3.push(at([sx * b.half[0], sy * b.half[1], sz * b.half[2]]));
                    }
                }
            }
        }
        let pts = convex_hull(pts3.iter().map(|p| pl.to_plane(*p)).collect());
        let lat = (b.center[0] - pl.com[0]) * perp[0] + (b.center[1] - pl.com[1]) * perp[1];
        v.push(Hull { pts, link: b.name, segment: b.segment, contactor: contactor_of(body, b.name), origin: pl.com, dir: pl.dir, lat });
    }
    v
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scene {
    pub ground: Surface,
    pub drop_m: f32,
    pub edge_u: f32,
    pub wall: Option<([f32; 3], Obstacle)>,
    pub mu_edge: f32,
    pub base: BaseKind,
}

const UP: [f32; 2] = [0.0, 1.0];
const FACE: [f32; 2] = [1.0, 0.0];

impl Scene {
    fn ground_z(&self, u: f32) -> f32 {
        if self.drop_m > 0.0 && u > self.edge_u + 1e-4 {
            -self.drop_m
        } else {
            0.0
        }
    }

    fn upper(&self, u: f32) -> bool {
        self.drop_m <= 0.0 || u <= self.edge_u + 1e-4
    }

    fn ground_hit(&self, prev: [f32; 2], now: [f32; 2]) -> Option<([f32; 2], f32)> {
        let foot = self.drop_m > 0.0 && now[1] < -self.drop_m && self.edge_u - now[0] < TOUCH_GAP_M;
        if !self.upper(now[0]) || foot {
            let depth = -self.drop_m - now[1];
            return (depth > 0.0).then_some((UP, depth));
        }
        if now[1] >= 0.0 {
            return None;
        }
        let top = -now[1];
        if self.drop_m <= 0.0 {
            return Some((UP, top));
        }
        let face = self.edge_u - now[0];
        let from_side = !self.upper(prev[0]) && prev[1] < 0.0;
        let from_top = prev[1] >= 0.0;
        Some(if from_side || (!from_top && face < top) { (FACE, face.max(0.0)) } else { (UP, top) })
    }

    fn gap_up(&self, p: [f32; 2]) -> f32 {
        p[1] - self.ground_z(p[0])
    }

    fn beside_face(&self, p: [f32; 2]) -> bool {
        self.drop_m > 0.0 && self.upper(p[0]) && p[1] < -TOUCH_GAP_M
    }

    fn holds(&self, p: [f32; 2], n: [f32; 2]) -> bool {
        if n[1] > 0.5 {
            self.gap_up(p) <= TOUCH_GAP_M && !self.beside_face(p)
        } else if n[0] > 0.5 {
            self.drop_m > 0.0 && p[0] <= self.edge_u + 1e-4 && p[1] <= 0.0 && p[1] >= -self.drop_m
        } else {
            self.wall.is_some_and(|(w, _)| p[0] >= w[0] - 1e-4 && p[1] >= w[1] && p[1] <= w[2])
        }
    }

    fn mu_of(&self, c: Contactor) -> f32 {
        match c {
            Contactor::Sole => surface::available_mu(self.base, self.ground.props()),
            Contactor::Tyre => self.mu_edge,
            _ => SHELL_GROUND_MU,
        }
    }

    fn corner(&self) -> Option<[f32; 2]> {
        (self.drop_m > 0.0).then_some([self.edge_u, 0.0])
    }
}

fn on_face(h: &Hull, ed: usize, e: [f32; 2], st: &Rigid2) -> ([f32; 2], f32, f32, [f32; 2], f32) {
    let q = rot2([e[0] - st.c[0], e[1] - st.c[1]], -st.th);
    let (a, b) = (h.pts[ed], h.pts[(ed + 1) % h.pts.len()]);
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = (d[0] * d[0] + d[1] * d[1]).max(1e-12);
    let along = ((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / len2;
    let o = h.out(ed);
    let s = (q[0] - a[0]) * o[0] + (q[1] - a[1]) * o[1];
    let w = rot2(o, st.th);
    (q, along, s, [-w[0], -w[1]], 1e-3 / len2.sqrt())
}

fn corner_on(h: &Hull, ed: usize, e: [f32; 2], st: &Rigid2, gap: f32) -> Option<(usize, [f32; 2], [f32; 2], f32)> {
    let k = h.pts.len();
    let (q, along, s, n, tol) = on_face(h, ed, e, st);
    let (ed, q, along, s, n, tol, gap) = if along > 1.0 + tol || along < -tol {
        let next = if along > 1.0 { (ed + 1) % k } else { (ed + k - 1) % k };
        let (q, along, s, n, tol) = on_face(h, next, e, st);
        (next, q, along, s, n, tol, s)
    } else {
        (ed, q, along, s, n, tol, gap)
    };
    (along >= -tol && along <= 1.0 + tol && (s - gap).abs() <= TOUCH_GAP_M && s <= TOUCH_GAP_M && n[0] >= -CONE_TOL && n[1] >= -CONE_TOL).then_some((ed, q, n, gap))
}

fn keep_on_corner(m: f32, i: f32, st: Rigid2, h: &Hull, ed: usize, e: [f32; 2], gap: f32) -> Rigid2 {
    let (_, _, s, n, _) = on_face(h, ed, e, &st);
    let moved = Rigid2 { c: [st.c[0] - (s - gap) * n[0], st.c[1] - (s - gap) * n[1]], ..st };
    let q2 = rot2([e[0] - moved.c[0], e[1] - moved.c[1]], -st.th);
    onto(m, i, moved, q2, n)
}

#[derive(Clone, Copy, Debug)]
struct Touch {
    u: f32,
    leave: f32,
    probe: Option<usize>,
    q: [f32; 2],
    mu: f32,
    n: [f32; 2],
    corner: Option<(usize, usize)>,
}

impl Touch {
    fn vertical(&self) -> bool {
        self.n[1].abs() < 0.5
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Hit {
    pub t: f32,
    pub probe: Option<usize>,
    pub at: Probe,
    pub wall: bool,
    pub corner: bool,
    pub n: [f32; 2],
    pub before: Rigid2,
    pub after: [f32; 3],
    pub v_n: f32,
    pub impulse: f32,
    pub m_eff: f32,
    pub law: PairLaw,
    pub pulse: Pulse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RockEnd {
    Over,
    Rest,
    Timeout,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Rocked {
    pub track: Vec<(f32, Rigid2)>,
    pub hits: Vec<Hit>,
    pub over: Option<(f32, Rigid2, [f32; 2])>,
    pub end: RockEnd,
    pub settles: usize,
    pub capped: bool,
    pub ke_end: f32,
    pub v_end: [f32; 3],
    pub ke_left: f32,
    pub slide_m: f32,
    pub roll_mps: f32,
    pub rest: Option<&'static str>,
    pub last: Rigid2,
    pub pivot: [f32; 2],
    pub t: f32,
}

fn add2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn point_vel(st: &Rigid2, r: [f32; 2]) -> [f32; 2] {
    [st.v[0] + st.w * r[1], st.v[1] - st.w * r[0]]
}

pub(crate) fn drot(q: [f32; 2], th: f32) -> [f32; 2] {
    let (s, c) = th.sin_cos();
    [-q[0] * s + q[1] * c, -q[0] * c - q[1] * s]
}

pub(crate) fn about_edge(q: [f32; 2], th: f32, w: f32, speed: f32) -> Rigid2 {
    let r = rot2(q, th);
    let d = drot(q, th);
    Rigid2 { c: [-r[0], -r[1]], th, v: [-w * d[0] + speed, -w * d[1]], w }
}

pub(crate) fn kinetic_j(m: f32, i: f32, s: [f32; 3]) -> f32 {
    0.5 * m * (s[0] * s[0] + s[1] * s[1]) + 0.5 * i * s[2] * s[2]
}

pub(crate) fn onto(m: f32, i: f32, st: Rigid2, q: [f32; 2], n: [f32; 2]) -> Rigid2 {
    let r = rot2(q, st.th);
    let i = i.max(1e-6);
    let c = r[1] * n[0] - r[0] * n[1];
    let v_n = (st.v[0] + st.w * r[1]) * n[0] + (st.v[1] - st.w * r[0]) * n[1];
    let lam = -v_n / (1.0 / m + c * c / i);
    Rigid2 { v: [st.v[0] + lam * n[0] / m, st.v[1] + lam * n[1] / m], w: st.w + lam * c / i, ..st }
}

const TOUCH_GAP_M: f32 = 0.005;

#[allow(clippy::too_many_arguments)]
fn resolve(m: f32, i: f32, st: &Rigid2, probes: &[Probe], at: &Probe, trig: Option<usize>, held: bool, mu_k: f32, n: [f32; 2], extra: &[usize], wall: Option<&Obstacle>, piv: Option<([f32; 2], f32, [f32; 2])>, sc: &Scene) -> Option<(Hit, Vec<Row>)> {
    let r_k = rot2(at.q, st.th);
    let t_dir = [n[1], -n[0]];
    let law = match wall {
        Some(o) => collide::pair(at.contactor, o),
        None => ground_pair(at.contactor, sc.ground),
    };
    let s0 = [st.v[0], st.v[1], st.w];
    let rows_with = |e_k: f32, plastic: bool, friction: bool| {
        let f = if friction { 1.0 } else { 0.0 };
        let mut rows = vec![Row::normal(r_k, n, if plastic { 0.0 } else { e_k }), Row::tangent(0, r_k, t_dir, f * mu_k)];
        for &j in extra {
            let rj = rot2(probes[j].q, st.th);
            let at = rows.len();
            let mut row = Row::normal(rj, UP, 0.0);
            if !plastic && -row.vel(s0) >= PLASTIC_MPS {
                row.e = ground_pair(probes[j].contactor, sc.ground).e;
            }
            rows.push(row);
            rows.push(Row::tangent(at, rj, [1.0, 0.0], f * sc.mu_of(probes[j].contactor)));
        }
        if let Some((pq, mu, pn)) = piv {
            let rp = rot2(pq, st.th);
            let at = rows.len();
            rows.push(Row::normal(rp, pn, 0.0));
            rows.push(Row::tangent(at, rp, collide::tangent_of(pn), f * mu));
        }
        rows
    };
    let v_n = -rows_with(law.e, false, true)[0].vel(s0);
    if v_n <= 1e-6 {
        return None;
    }
    let solve = |e_k: f32, plastic: bool, friction: bool| {
        let rows = rows_with(e_k, plastic, friction);
        let mut imp = solve_impulses(m, i, s0, &rows);
        if !imp.found {
            imp = solve_impulses(m, i, s0, &rows[..2]);
            imp.lambda.resize(rows.len(), 0.0);
        }
        (imp, rows)
    };
    let (first, _) = solve(law.e, false, true);
    let m_eff = (first.lambda[0] / ((1.0 + law.e) * v_n)).max(1e-4);
    let p = if holds_contact(v_n, held) { Pulse { peak_n: 0.0, t_peak_s: 0.0, contact_s: 0.0, e: 0.0, max_defl_m: 0.0, series: vec![] } } else { pulse(&law.law, law.e, m_eff, v_n) };
    let k0 = kinetic_j(m, i, s0);
    let gains = |s: [f32; 3]| kinetic_j(m, i, s) > k0 * (1.0 + 1e-4) + 1e-6;
    let mut out = solve(if v_n < PLASTIC_MPS { 0.0 } else { p.e }, false, true);
    if gains(out.0.s) {
        out = solve(0.0, true, true);
    }
    if gains(out.0.s) {
        out = solve(0.0, true, false);
    }
    let (imp, rows) = out;
    let hit = Hit { t: 0.0, probe: trig, at: *at, wall: wall.is_some(), corner: trig.is_none(), n, before: *st, after: imp.s, v_n, impulse: imp.lambda[0], m_eff, law, pulse: p };
    Some((hit, rows))
}

fn holds_contact(v_n: f32, held: bool) -> bool {
    v_n < STAY_MPS || (held && v_n < REST_MPS)
}

#[derive(Clone, Copy, Debug)]
struct Cand {
    at: Probe,
    probe: Option<usize>,
    corner: Option<(usize, usize)>,
    n: Option<[f32; 2]>,
    depth: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rock(m: f32, i: f32, probes: &[Probe], hulls: &[Hull], sc: &Scene, q0: [f32; 2], st0: Rigid2, stop_at_over: bool, t_max: f32) -> Rocked {
    let mut st = onto(m, i, st0, q0, UP);
    let mut piv = q0;
    let mut piv_n = UP;
    let mut piv_probe: Option<usize> = probes.iter().position(|p| (p.q[0] - q0[0]).abs() < 3e-3 && (p.q[1] - q0[1]).abs() < 3e-3);
    let mut piv_corner: Option<(usize, usize)> = None;
    let mut piv_gap = 0.0f32;
    let mut mu_piv = sc.mu_edge;
    let mut mode = collide::pivot_mode(m, i, &st, piv, piv_n, mu_piv, PivotMode::Stick);
    let mut t = 0.0f32;
    let mut next = SAMPLE_S;
    let mut track = vec![(0.0, st)];
    let mut hits: Vec<Hit> = Vec::new();
    let mut settles = 0usize;
    let mut over: Option<(f32, Rigid2, [f32; 2])> = None;
    let mut end = RockEnd::Timeout;
    let mut capped = false;
    let mut ke_end: Option<f32> = None;
    let mut v_end = [0.0f32; 3];
    let mut rest: Option<&'static str> = None;
    let wall_ob = sc.wall.map(|w| w.1);
    let corner = sc.corner();
    let calm_ke = 0.5 * m * CALM_MPS * CALM_MPS;
    let mut calm = 0usize;
    let mut held_t = f32::NEG_INFINITY;
    let mut recent: Vec<((Option<usize>, Option<(usize, usize)>), f32)> = Vec::new();
    let mut slide_m = 0.0f32;
    let mut roll_mps = 0.0f32;
    let mut ke_left: Option<f32> = None;
    let edges: Vec<f32> = probes.iter().filter(|p| p.link == "support").map(|p| p.q[0]).collect();
    let mid = if edges.is_empty() { 0.0 } else { edges.iter().sum::<f32>() / edges.len() as f32 };
    while t < t_max {
        if mode != PivotMode::Free {
            match (piv_corner, corner) {
                (Some((h, ed)), Some(e)) => match corner_on(&hulls[h], ed, e, &st, piv_gap) {
                    Some((ed, q, n, gap)) => {
                        piv_corner = Some((h, ed));
                        piv_gap = gap;
                        piv = q;
                        piv_n = n;
                    }
                    None => mode = PivotMode::Free,
                },
                _ => {
                    if !sc.holds(add2(st.c, rot2(piv, st.th)), piv_n) {
                        mode = PivotMode::Free;
                    }
                }
            }
        }
        let mut nst = collide::pivot_step(m, i, &st, piv, piv_n, mode, if mode == PivotMode::Free { 0.0 } else { mu_piv }, DT_S);
        if mode != PivotMode::Free {
            if let (Some((h, ed)), Some(e)) = (piv_corner, corner) {
                nst = keep_on_corner(m, i, nst, &hulls[h], ed, e, piv_gap);
            }
        }
        if over.is_none() && mode != PivotMode::Free && piv_corner.is_none() && piv_n[1] > 0.5 {
            let out = if piv[0] >= mid { 1.0f32 } else { -1.0 };
            let p0 = add2(st.c, rot2(piv, st.th));
            let p1 = add2(nst.c, rot2(piv, nst.th));
            if (st.c[0] - p0[0]) * out < 0.0 && (nst.c[0] - p1[0]) * out >= 0.0 {
                over = Some((t + DT_S, nst, piv));
                if stop_at_over {
                    st = nst;
                    t += DT_S;
                    end = RockEnd::Over;
                    break;
                }
            }
        }
        let mut best: Option<Cand> = None;
        let at_pivot = |q: [f32; 2]| mode != PivotMode::Free && (q[0] - piv[0]).abs() < 3e-3 && (q[1] - piv[1]).abs() < 3e-3;
        for (k, p) in probes.iter().enumerate() {
            if (mode != PivotMode::Free && Some(k) == piv_probe && piv_corner.is_none()) || at_pivot(p.q) {
                continue;
            }
            let r = rot2(p.q, nst.th);
            let pos = add2(nst.c, r);
            let vel = point_vel(&nst, r);
            if let Some((n, depth)) = sc.ground_hit(add2(st.c, rot2(p.q, st.th)), pos) {
                if depth > SINK_M && vel[0] * n[0] + vel[1] * n[1] < 0.0 && best.map_or(true, |b| depth > b.depth) {
                    best = Some(Cand { at: *p, probe: Some(k), corner: None, n: Some(n), depth });
                }
            }
            if let Some((w, _)) = sc.wall {
                if pos[0] > w[0] && pos[1] >= w[1] && pos[1] <= w[2] && vel[0] > 0.0 {
                    let depth = pos[0] - w[0];
                    if best.map_or(true, |b| depth > b.depth) {
                        best = Some(Cand { at: *p, probe: Some(k), corner: None, n: None, depth });
                    }
                }
            }
        }
        if let Some(e) = corner {
            let q = rot2([e[0] - nst.c[0], e[1] - nst.c[1]], -nst.th);
            let r = [e[0] - nst.c[0], e[1] - nst.c[1]];
            let vel = point_vel(&nst, r);
            let mut found: Option<Cand> = None;
            for (h_i, h) in hulls.iter().enumerate() {
                let Some((ed, depth, o)) = h.inside(q) else { continue };
                if (mode != PivotMode::Free && piv_corner == Some((h_i, ed))) || (piv_corner.is_none() && at_pivot(q)) {
                    continue;
                }
                let w = rot2(o, nst.th);
                let n = [-w[0], -w[1]];
                if n[0] < -CONE_TOL || n[1] < -CONE_TOL || vel[0] * n[0] + vel[1] * n[1] >= 0.0 || depth <= SINK_M {
                    continue;
                }
                if found.map_or(true, |f| depth < f.depth) {
                    found = Some(Cand { at: h.probe_at(q), probe: None, corner: Some((h_i, ed)), n: Some(n), depth });
                }
            }
            if let Some(f) = found {
                if best.map_or(true, |b| f.depth > b.depth) {
                    best = Some(f);
                }
            }
        }
        st = nst;
        t += DT_S;
        let mut stop = false;
        if let Some(cand) = best {
            let on_wall = cand.n.is_none();
            let n_k = cand.n.unwrap_or([-1.0, 0.0]);
            let on_top = n_k[1] > 0.5;
            let pivot_loaded = (mode != PivotMode::Free).then_some((piv, mu_piv, piv_n));
            let is_pivot = |j: usize, q: [f32; 2]| pivot_loaded.is_some() && ((Some(j) == piv_probe && piv_corner.is_none()) || ((q[0] - piv[0]).abs() < 3e-3 && (q[1] - piv[1]).abs() < 3e-3));
            let mut extra: Vec<usize> = Vec::new();
            if on_top {
                let u_k = st.c[0] + rot2(cand.at.q, st.th)[0];
                let touching: Vec<(usize, f32)> = probes
                    .iter()
                    .enumerate()
                    .filter(|(j, p)| Some(*j) != cand.probe && !is_pivot(*j, p.q))
                    .filter_map(|(j, p)| {
                        let pos = add2(st.c, rot2(p.q, st.th));
                        (sc.gap_up(pos) < TOUCH_GAP_M && !sc.beside_face(pos) && (pos[0] - u_k).abs() > 0.02).then_some((j, pos[0]))
                    })
                    .collect();
                let lo = touching.iter().min_by(|a, b| a.1.total_cmp(&b.1)).copied();
                let hi = touching.iter().max_by(|a, b| a.1.total_cmp(&b.1)).copied();
                for (j, u) in [lo, hi].into_iter().flatten() {
                    if extra.iter().all(|&e| e != j && (st.c[0] + rot2(probes[e].q, st.th)[0] - u).abs() > 0.02) {
                        extra.push(j);
                    }
                }
            }
            let mu_k = if on_wall { SHELL_GROUND_MU } else { sc.mu_of(cand.at.contactor) };
            let key = (cand.probe, cand.corner);
            recent.retain(|(_, t0)| t - t0 <= HELD_S);
            let held = recent.iter().any(|(k, _)| *k == key);
            if let Some((mut hit, rows)) = resolve(m, i, &st, probes, &cand.at, cand.probe, held, mu_k, n_k, &extra, if on_wall { wall_ob.as_ref() } else { None }, pivot_loaded, sc) {
                hit.t = t;
                held_t = t;
                recent.retain(|(k, _)| *k != key);
                recent.push((key, t));
                let s = hit.after;
                if holds_contact(hit.v_n, held) {
                    settles += 1;
                } else {
                    hits.push(hit);
                }
                let u_of = |q: [f32; 2]| st.c[0] + rot2(q, st.th)[0];
                let mut contacts = vec![Touch { u: u_of(cand.at.q), leave: rows[0].vel(s), probe: cand.probe, q: cand.at.q, mu: mu_k, n: n_k, corner: cand.corner }];
                for (n_e, &j) in extra.iter().enumerate() {
                    contacts.push(Touch { u: u_of(probes[j].q), leave: rows[2 + 2 * n_e].vel(s), probe: Some(j), q: probes[j].q, mu: sc.mu_of(probes[j].contactor), n: UP, corner: None });
                }
                if pivot_loaded.is_some() {
                    contacts.push(Touch { u: u_of(piv), leave: rows[rows.len() - 2].vel(s), probe: piv_probe, q: piv, mu: mu_piv, n: piv_n, corner: piv_corner });
                }
                st.v = [s[0], s[1]];
                st.w = s[2];
                let staying: Vec<Touch> = contacts.iter().copied().filter(|c| c.leave.abs() <= STAY_MPS).collect();
                let ground: Vec<Touch> = staying.iter().copied().filter(|c| !c.vertical()).collect();
                let lo = ground.iter().copied().min_by(|a, b| a.u.total_cmp(&b.u));
                let hi = ground.iter().copied().max_by(|a, b| a.u.total_cmp(&b.u));
                let leaning = staying.iter().any(|w| w.vertical() && ground.iter().any(|c| (c.u - st.c[0]) * w.n[0] > 0.0)) && kinetic_j(m, i, s) <= calm_ke;
                let between = matches!((lo, hi), (Some(lo), Some(hi)) if hi.u - lo.u > 0.02 && st.c[0] >= lo.u && st.c[0] <= hi.u);
                let slide = match (lo, hi) {
                    (Some(lo), Some(hi)) if between => {
                        let up = ((st.c[0] - lo.u) / (hi.u - lo.u).max(1e-6)).clamp(0.0, 1.0);
                        let a = G * (lo.mu * (1.0 - up) + hi.mu * up);
                        let v = s[0];
                        Some((a, v))
                    }
                    _ => None,
                };
                let goes_over = sc.drop_m > 0.0 && s[0] > 1e-3 && !ground.is_empty() && {
                    let a = G * ground.iter().map(|c| c.mu).sum::<f32>() / ground.len() as f32;
                    let reach = if a > 1e-3 { 0.5 * s[0] * s[0] / a } else { f32::INFINITY };
                    ground.iter().any(|c| c.u <= sc.edge_u + 1e-4 && c.u + reach >= sc.edge_u - 1e-4)
                };
                let between = between && !goes_over;
                let ke = kinetic_j(m, i, s);
                let trapped = on_top && !goes_over && contacts[0].leave.abs() <= STAY_MPS && {
                    let barrier = |q: [f32; 2]| m * G * ((q[0] * q[0] + q[1] * q[1]).sqrt() + rot2(q, st.th)[1]);
                    let side = contacts[0].u - st.c[0];
                    contacts.iter().skip(1).filter(|c| !c.vertical() && (c.u - st.c[0]) * side < 0.0).any(|c| ke < barrier(c.q).min(barrier(contacts[0].q)))
                };
                if leaning || between || trapped || (hits.len() >= MAX_STRIKES && ke <= calm_ke) {
                    ke_end = Some(ke);
                    v_end = s;
                    rest = Some(if leaning { "leaning" } else if between { "between" } else if trapped { "trapped" } else { "strikes" });
                    ke_left = Some(ke);
                    if let (false, true, Some(lo), Some(hi), Some((a, v))) = (leaning, between, lo, hi, slide) {
                        ke_left = Some(kinetic_j(m, i, [0.0, s[1], s[2]]));
                        if a > 1e-3 && v.abs() > 1e-4 {
                            let mut d = 0.5 * v * v.abs() / a;
                            if sc.drop_m > 0.0 && lo.u > sc.edge_u && d < 0.0 {
                                d = d.max(sc.edge_u - lo.u);
                            }
                            if let Some((w, _)) = sc.wall {
                                if d > 0.0 {
                                    d = d.min((w[0] - hi.u).max(0.0));
                                }
                            }
                            if d.abs() > 1e-6 {
                                track.push((t, Rigid2 { v: [v, 0.0], w: 0.0, ..st }));
                                st.c[0] += d;
                                t += 2.0 * d.abs() / v.abs();
                                slide_m = d;
                            }
                        } else {
                            roll_mps = v;
                        }
                    }
                    st.v = [0.0, 0.0];
                    st.w = 0.0;
                    end = RockEnd::Rest;
                    stop = true;
                } else if hits.len() >= MAX_STRIKES {
                    ke_end = Some(ke);
                    v_end = s;
                    capped = true;
                    end = RockEnd::Timeout;
                    stop = true;
                } else {
                    let on = match (lo, hi) {
                        (Some(lo), Some(hi)) => Some(if st.c[0] > hi.u {
                            hi
                        } else if st.c[0] < lo.u {
                            lo
                        } else {
                            ground.iter().copied().min_by(|a, b| (st.c[0] - a.u).abs().total_cmp(&(st.c[0] - b.u).abs())).unwrap_or(hi)
                        }),
                        _ => staying.iter().copied().find(|c| c.vertical()),
                    };
                    match on {
                        Some(c) => {
                            piv = c.q;
                            piv_probe = c.probe;
                            piv_corner = c.corner;
                            if let (Some((h, ed)), Some(e)) = (c.corner, corner) {
                                piv_gap = on_face(&hulls[h], ed, e, &st).2.min(0.0);
                            }
                            mu_piv = c.mu;
                            piv_n = c.n;
                            st = onto(m, i, st, piv, piv_n);
                            mode = collide::pivot_mode(m, i, &st, piv, piv_n, mu_piv, PivotMode::Stick);
                        }
                        None => {
                            mode = PivotMode::Free;
                            piv_corner = None;
                        }
                    }
                }
            }
        } else if mode != PivotMode::Free {
            if let (Some((h, ed)), Some(e)) = (piv_corner, corner) {
                match corner_on(&hulls[h], ed, e, &st, piv_gap) {
                    Some((ed, q, n, gap)) => {
                        piv_corner = Some((h, ed));
                        piv_gap = gap;
                        piv = q;
                        piv_n = n;
                    }
                    None => mode = PivotMode::Free,
                }
            }
            if mode != PivotMode::Free {
                mode = collide::pivot_mode(m, i, &st, piv, piv_n, mu_piv, mode);
            }
        }
        if mode == PivotMode::Free && !stop {
            piv_corner = None;
        }
        if t >= next || stop {
            next += SAMPLE_S;
            track.push((t, st));
        }
        if stop {
            break;
        }
        let ke = kinetic_j(m, i, [st.v[0], st.v[1], st.w]);
        calm = if t - held_t <= HELD_GAP_S && ke <= calm_ke { calm + 1 } else { 0 };
        if calm as f32 * DT_S >= CALM_S {
            ke_end = Some(ke);
            v_end = [st.v[0], st.v[1], st.w];
            rest = Some("calm");
            st.v = [0.0, 0.0];
            st.w = 0.0;
            end = RockEnd::Rest;
            if track.last().is_some_and(|x| x.0 >= t) {
                track.pop();
            }
            track.push((t, st));
            break;
        }
    }
    if track.last().map_or(true, |x| x.0 < t) {
        track.push((t, st));
    }
    if ke_end.is_none() {
        v_end = [st.v[0], st.v[1], st.w];
    }
    let ke_end = ke_end.unwrap_or_else(|| kinetic_j(m, i, [st.v[0], st.v[1], st.w]));
    let ke_left = ke_left.unwrap_or(ke_end);
    Rocked { track, hits, over, end, settles, capped, ke_end, ke_left, v_end, rest, slide_m, roll_mps, last: st, pivot: piv, t }
}

pub(crate) fn dir_of(deg: f32) -> [f32; 2] {
    let a = deg.to_radians();
    [a.cos(), a.sin()]
}

fn parts_of(seg: Segment, body: &Body) -> Vec<Part> {
    let mut v = collide::base_parts(body);
    match seg {
        Segment::Head => v.push(Part::Neck),
        Segment::Arm | Segment::Hand => v.push(Part::HandJoints),
        Segment::Chest | Segment::Belly => v.push(Part::Waist),
        _ => {}
    }
    v
}

pub(crate) struct Played {
    pub solved: KinReport,
    pub mp: collide::MassProps,
    pub pl: Planar,
    pub probes: Vec<Probe>,
    pub q0: [f32; 2],
    pub sc: Scene,
    pub rk: Rocked,
}

pub(crate) fn play(inp: &FallInput) -> Played {
    let body = inp.body;
    let solved = kin::solve(body, &inp.pose, &KinLoads::default());
    let mp = collide::mass_props_of(body, &solved);
    let dir = dir_of(inp.dir_deg);
    let pl = Planar::of(&mp, solved.support, dir);
    let probes = probes(body, &solved, &pl);
    let hulls = hulls(body, &solved, &pl);
    let q0 = inp.pivot.unwrap_or([pl.ahead_m, -pl.com_z]);
    let th0 = inp.tilt0_deg.to_radians();
    let st0 = about_edge(q0, th0, inp.omega0, inp.speed_mps);
    let hold = collide::hold_of(body, dir, inp.speed_mps.abs() > 0.05);
    let mu_edge = if hold == collide::Hold::Friction { surface::available_mu(body.base, inp.ground.props()) } else { 0.0 };
    let sc = Scene { ground: inp.ground, drop_m: inp.drop_m.max(0.0), edge_u: inp.edge_m, wall: None, mu_edge, base: body.base };
    let rk = rock(mp.mass_kg, pl.inertia, &probes, &hulls, &sc, q0, st0, false, FALL_MAX_S);
    Played { solved, mp, pl, probes, q0, sc, rk }
}

pub fn fall(inp: &FallInput) -> Fall {
    let body = inp.body;
    let dir = dir_of(inp.dir_deg);
    let Played { solved, mp, pl, probes: pr, q0, sc, rk } = play(inp);
    let trunk_q = solved.boxes.iter().find(|b| b.name == "trunk").map(|b| pl.to_plane(b.center));
    let mut strikes = Vec::new();
    for h in &rk.hits {
        let p = &h.at;
        let ds = [h.after[0] - h.before.v[0], h.after[1] - h.before.v[1], h.after[2] - h.before.w];
        let a_trunk = trunk_q.map_or(0.0, |q| {
            let r = rot2(q, h.before.th);
            let dv = [ds[0] + ds[2] * r[1], ds[1] - ds[2] * r[0]];
            (dv[0] * dv[0] + dv[1] * dv[1]).sqrt() * h.pulse.peak_n / h.impulse.max(1e-6) / G
        });
        strikes.push(Strike {
            t: h.t,
            tilt_deg: h.before.th.to_degrees(),
            link: p.link,
            segment: p.segment,
            contactor: p.contactor,
            point: p.point,
            wall: h.wall,
            face: !h.wall && !h.corner && h.n[1] < 0.5,
            edge: h.corner,
            v_n: h.v_n,
            impulse_ns: h.impulse,
            m_eff_kg: h.m_eff,
            peak_n: h.pulse.peak_n,
            contact_s: h.pulse.contact_s,
            a_point_g: h.pulse.peak_n / h.m_eff.max(1e-4) / G,
            a_trunk_g: a_trunk,
            energy_j: 0.5 * h.m_eff * h.v_n * h.v_n,
        });
    }
    let main_i = strikes.iter().enumerate().max_by(|a, b| a.1.v_n.total_cmp(&b.1.v_n)).map(|x| x.0);
    let main = main_i.map(|i| strikes[i].clone());
    let first_i = strikes.iter().position(|s| !s.wall && s.v_n >= HARD_MPS).or(main_i);
    let t_first = first_i.map(|i| strikes[i].t);
    let speed_of = |names: &[&str]| -> Option<f32> {
        if let Some(s) = strikes.iter().find(|s| !s.wall && names.contains(&s.link) && s.v_n >= REST_MPS) {
            return Some(s.v_n);
        }
        let st = first_i.map(|i| rk.hits[i].before)?;
        let vs: Vec<f32> = solved
            .boxes
            .iter()
            .filter(|b| names.contains(&b.name))
            .map(|b| {
                let r = rot2(pl.to_plane(b.center), st.th);
                -point_vel(&st, r)[1]
            })
            .collect();
        (!vs.is_empty()).then(|| vs.iter().copied().fold(f32::NEG_INFINITY, f32::max))
    };
    let pelvis_v = speed_of(&["pelvis"]);
    let head_v = speed_of(&["head"]);
    let hand_v = speed_of(&["palm_r", "palm_l"]);
    let last = rk.last;
    let com0 = mp.com;
    let pivot0 = [com0[0] + dir[0] * q0[0], com0[1] + dir[1] * q0[0], 0.0];
    let lying_tilt = last.th.to_degrees();
    let mut lying = LyingPose { pose: inp.pose, dir_deg: inp.dir_deg, tilt_deg: lying_tilt, com: [0.0; 3], com0, pivot0, c_rest: last.c };
    lying.com = [pivot0[0] + dir[0] * last.c[0], pivot0[1] + dir[1] * last.c[0], last.c[1]];
    let th_star = q0[0].max(0.0).atan2((-q0[1]).max(1e-3));
    let off_support = last.th.abs() > 0.5 * th_star.max(2f32.to_radians());
    let fell = lying.down() || off_support || inp.drop_m > 0.0 && last.c[0] > sc.edge_u;
    let damage = match (&main, main_i) {
        (Some(s), Some(i)) => {
            let th = rk.hits[i].before.th;
            let n = rk.hits[i].n;
            let f_body = rot2(n, -th);
            let f = pl.to_base([f_body[0] * s.peak_n, f_body[1] * s.peak_n]);
            joint_hits(body, &solved, s.link, s.point, f, s.contact_s)
        }
        _ => vec![],
    };
    let get_up = if fell { get_up(body, &lying, &damage, &pr, &pl) } else { GetUp { can: true, time_s: Some(0.0), needs_help: false, righting_force_n: 0.0, why_ko: "넘어지지 않음".into() } };
    let mut samples = Vec::new();
    let mut tilt = Vec::new();
    for (t, st) in &rk.track {
        tilt.push((*t, st.th.to_degrees()));
        let com = [pivot0[0] + dir[0] * st.c[0], pivot0[1] + dir[1] * st.c[0], st.c[1]];
        let phase = if strikes.iter().any(|s| (s.t - t).abs() <= 0.5 * SAMPLE_S) {
            "impact"
        } else if rk.over.is_some_and(|o| *t >= o.0) || st.th > th_star {
            "falling"
        } else {
            "tilt"
        };
        let pose = if body.base == BaseKind::HumanFoot { inp.pose } else { KinPose { base_pitch_deg: -st.th.to_degrees() * dir[0], ..inp.pose } };
        samples.push(ActSample { t: *t, phase, pose, loads: KinLoads::default(), margin: 1.0 - st.th / th_star.max(1e-3), zmp: [com[0], com[1]], com, peak_ratio: 0.0, grip: None, object: None });
    }
    let mut events = Vec::new();
    let start_kind = if inp.drop_m > 0.0 { "edge_fall" } else { "fall" };
    let dir_ko = dir_label_ko(inp.dir_deg);
    let start_label = if inp.drop_m > 0.0 { format!("{} {:.2} m 아래로 떨어지기 시작 ({})", inp.cause.label_ko(), inp.drop_m, dir_ko) } else { format!("{} {} 넘어가기 시작 (기울기 {:.0}°, {:.2} rad/s)", inp.cause.label_ko(), dir_ko, inp.tilt0_deg, inp.omega0) };
    events.push(ActEvent::new(0.0, start_kind, start_label, &[Part::Waist], Some(inp.tilt0_deg), false));
    for s in strikes.iter().filter(|s| s.v_n >= 0.1) {
        events.push(ActEvent::new(
            s.t,
            "impact",
            format!("{} {}{} {:.1} m/s로 부딪힘 · {:.0} N ({:.0} ms) · {:.0} g · {:.0} J", s.link, if s.wall { "벽에 " } else if s.edge { "턱 모서리에 " } else if s.face { "턱 옆면에 " } else { "땅에 " }, s.contactor.label_ko(), s.v_n, s.peak_n, s.contact_s * 1000.0, s.a_point_g, s.energy_j),
            &parts_of(s.segment, body),
            Some(s.peak_n),
            s.v_n < 0.5,
        ));
    }
    if let Some(s) = &main {
        events.extend(shock_events(&damage, s.t, &parts_of(s.segment, body)));
    }
    let t_end = rk.t;
    let at_rest = rk.end == RockEnd::Rest;
    if fell {
        let how = if !at_rest { "아직 움직이는 중" } else if lying.down() { "누움" } else { "기대어 걸림" };
        events.push(ActEvent::new(t_end, "lying", format!("{:.0}° 기운 채 {} ({how})", lying_tilt, if at_rest { "멈춤" } else { "끝남" }), &[Part::Waist], Some(lying_tilt), false));
        if get_up.can {
            events.push(ActEvent::new(t_end + get_up.time_s.unwrap_or(0.0), "get_up", get_up.why_ko.clone(), &collide::base_parts(body), get_up.time_s, true));
        } else {
            events.push(ActEvent::new(t_end, "stuck", get_up.why_ko.clone(), &collide::base_parts(body), Some(get_up.righting_force_n), false));
        }
    } else {
        events.push(ActEvent::new(t_end, "recover", format!("다시 섬 ({:.1}°)", lying_tilt), &collide::base_parts(body), Some(lying_tilt), true));
    }
    let outcome = if !fell {
        Outcome::ok()
    } else if inp.drop_m > 0.0 {
        Outcome::fail("edge_fall", format!("{:.2} m 아래로 떨어짐{}", inp.drop_m, main.as_ref().map_or(String::new(), |s| format!(" ({} {:.1} m/s, {:.0} N)", s.link, s.v_n, s.peak_n))))
    } else {
        Outcome::fail("fall", format!("{} {} 넘어짐{}", inp.cause.label_ko(), dir_ko, main.as_ref().map_or(String::new(), |s| format!(" ({} {:.1} m/s, {:.0} N)", s.link, s.v_n, s.peak_n))))
    };
    let t_ground = main.as_ref().map_or(t_end, |s| s.t);
    let t_first = t_first.unwrap_or(t_ground);
    let mut metrics = vec![
        ("t_ground_s", t_ground),
        ("t_first_s", t_first),
        ("tilt_rest_deg", lying_tilt),
        ("strikes", strikes.len() as f32),
        ("fell", if fell { 1.0 } else { 0.0 }),
        ("needs_help", if get_up.needs_help { 1.0 } else { 0.0 }),
        ("righting_force_n", get_up.righting_force_n),
        ("com_drop_m", com0[2] - lying.com[2]),
        ("at_rest", if at_rest { 1.0 } else { 0.0 }),
        ("slide_m", rk.slide_m),
        ("roll_mps", rk.roll_mps),
    ];
    if let Some(s) = &main {
        metrics.extend([("v_impact_mps", s.v_n), ("peak_n", s.peak_n), ("contact_ms", s.contact_s * 1000.0), ("a_point_g", s.a_point_g), ("a_trunk_g", s.a_trunk_g), ("m_eff_kg", s.m_eff_kg), ("energy_j", s.energy_j)]);
    }
    for (k, v) in [("pelvis_v_mps", pelvis_v), ("head_v_mps", head_v), ("hand_v_mps", hand_v), ("get_up_s", get_up.time_s)] {
        if let Some(v) = v {
            metrics.push((k, v));
        }
    }
    let mut notes = vec![format!("관절을 고정한 강체로 넘어짐 (회전 관성 {:.2} kg·m², 무게중심 {:.2} m)", pl.inertia, pl.com_z)];
    if rk.end == RockEnd::Timeout {
        notes.push(if rk.capped { format!("타격 {MAX_STRIKES}번 한도에서 아직 움직임 ({:.2} J): 마지막 자세를 믿지 말 것", rk.ke_end) } else { "시간 끝까지 멈추지 않음".into() });
    }
    let rollout = PhysRollout { kind: PhysKind::Fall, samples, events, outcome, duration_s: t_end, metrics, notes };
    Fall { rollout, cause: inp.cause, dir_deg: inp.dir_deg, fell, t_ground_s: t_ground, t_first_s: t_first, strikes, main, pelvis_v, head_v, hand_v, tilt, lying, damage, get_up, at_rest }
}

pub fn dir_label_ko(deg: f32) -> &'static str {
    let d = (deg.rem_euclid(360.0) + 22.5) / 45.0;
    match d as i32 % 8 {
        0 => "앞으로",
        1 => "앞 왼쪽으로",
        2 => "왼쪽으로",
        3 => "뒤 왼쪽으로",
        4 => "뒤로",
        5 => "뒤 오른쪽으로",
        6 => "오른쪽으로",
        _ => "앞 오른쪽으로",
    }
}

pub fn get_up_squat(body: &Body) -> Option<kin::KinReport> {
    let mut best: Option<(KinPose, f32)> = None;
    for i in 0..14 {
        let hx = -0.22 + 0.02 * i as f32;
        for j in 0..13 {
            let pd = 5.0 * j as f32;
            let pose = KinPose { hip: [hx, ankle_squat_hip_m(&body.dims)], pelvis_deg: pd, shoulder_deg: [70.0, 70.0], elbow_deg: [10.0, 10.0], ..KinPose::stand(body, 0.0) };
            let r = kin::solve_fast(body, &pose, &KinLoads::default());
            if r.out_of_range.is_empty() && best.map_or(true, |b| r.margin > b.1) {
                best = Some((pose, r.margin));
            }
        }
    }
    best.map(|(p, _)| kin::solve(body, &p, &KinLoads::default()))
}

pub fn ankle_squat_hip_m(d: &crate::dims::Dims) -> f32 {
    d.ankle_m + 0.45 * (d.hip_m - d.ankle_m)
}

pub(crate) fn get_up(body: &Body, lying: &LyingPose, damage: &[JointHit], probes: &[Probe], pl: &Planar) -> GetUp {
    let th = lying.tilt_deg.to_radians();
    let far = probes.iter().map(|p| rot2([p.q[0] - pl.ahead_m, p.q[1] + pl.com_z], th)[0]).fold(0.0f32, f32::max);
    let com_u = rot2([-pl.ahead_m, pl.com_z], th)[0];
    let righting = if far > 1e-3 && com_u > 0.0 { pl.mass_kg * G * com_u / far } else { 0.0 };
    if body.base != BaseKind::HumanFoot {
        return GetUp { can: false, time_s: None, needs_help: true, righting_force_n: righting, why_ko: format!("바퀴·트랙은 스스로 못 일어남: 끝을 {:.0} N으로 들어 올려 줄 도움이 필요", righting) };
    }
    if let Some(h) = damage.iter().find(|h| h.overload) {
        return GetUp { can: false, time_s: None, needs_help: true, righting_force_n: righting, why_ko: format!("넘어질 때 {} 관절이 피크를 넘는 충격을 받아 일어나지 못함", h.joint) };
    }
    match get_up_squat(body) {
        Some(r) if r.feasible => GetUp { can: true, time_s: Some(GET_UP_LEGS_S), needs_help: false, righting_force_n: righting, why_ko: format!("쪼그려 앉은 자세에서 일어남 ({:.0} s, 가장 힘든 관절 {:.0}%)", GET_UP_LEGS_S, r.worst().map_or(0.0, |j| j.ratio * 100.0)) },
        Some(r) => GetUp { can: false, time_s: None, needs_help: true, righting_force_n: righting, why_ko: format!("쪼그려 앉은 자세를 못 버팀: {}", r.limiting.unwrap_or_default()) },
        None => GetUp { can: false, time_s: None, needs_help: true, righting_force_n: righting, why_ko: "관절 범위로 쪼그려 앉지 못함".into() },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RodFall {
    pub t_s: f32,
    pub omega_end: f32,
    pub top_mps: f32,
}

pub fn rod_topple(h_m: f32, tilt0_deg: f32) -> RodFall {
    let k = 3.0 * G / h_m.max(1e-3);
    let th0 = tilt0_deg.to_radians().max(1e-4);
    let n = 4000;
    let end = std::f32::consts::FRAC_PI_2;
    let mut t = 0.0f32;
    for j in 0..n {
        let a = th0 + (end - th0) * (j as f32 + 0.5) / n as f32;
        t += (end - th0) / n as f32 / (k * (th0.cos() - a.cos())).max(1e-12).sqrt();
    }
    let w = (k * th0.cos()).sqrt();
    RodFall { t_s: t, omega_end: w, top_mps: w * h_m }
}

pub fn rod_point_speed(h_m: f32, r_m: f32, tilt0_deg: f32, tilt_deg: f32) -> f32 {
    let k = 3.0 * G / h_m.max(1e-3);
    r_m * (k * (tilt0_deg.to_radians().cos() - tilt_deg.to_radians().cos()).max(0.0)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    fn body(tier: Tier, base: BaseKind) -> Body {
        Body::new(&BodyPlan::uniform(tier), base, FingerCount::Five)
    }

    #[test]
    fn a_rigid_pendulum_falls_in_its_closed_form_time() {
        let (l, m, i) = (1.0f32, 10.0f32, 0.5f32);
        let th_s = 40f32.to_radians();
        let (pu, pz) = (0.3f32, 0.3 * th_s.tan());
        let probes = [Probe { q: [pu, pz - l], point: [0.0; 3], link: "head", segment: Segment::Head, contactor: Contactor::Shell }];
        let sc = Scene { ground: Surface::Firm, drop_m: 0.0, edge_u: 0.0, wall: None, mu_edge: 2.0, base: BaseKind::HumanFoot };
        let piv = [0.0, -l];
        let th0 = 5.0f32.to_radians();
        let rk = rock(m, i, &probes, &[], &sc, piv, about_edge(piv, th0, 0.0, 0.0), false, 5.0);
        let k = m * G * l / (i + m * l * l);
        let n = 20000;
        let mut t = 0.0f32;
        for j in 0..n {
            let a = th0 + (th_s - th0) * (j as f32 + 0.5) / n as f32;
            t += (th_s - th0) / n as f32 / (2.0 * k * (th0.cos() - a.cos())).sqrt();
        }
        let first = rk.hits.first().expect("the probe strikes the ground");
        assert!((first.t - t).abs() < 0.01, "strike at {} s vs {t} s", first.t);
        let w = (2.0 * k * (th0.cos() - th_s.cos())).sqrt();
        let v = w * (pu * th_s.cos() + pz * th_s.sin());
        assert!((first.v_n - v).abs() < 0.02 * v, "{} vs {v}", first.v_n);
        assert!(rk.over.map_or(true, |o| o.2 != piv), "{:?}", rk.over);
        let r = rod_topple(10.0, 1.0);
        assert!((r.top_mps - (3.0 * G * 10.0f32).sqrt()).abs() < 0.05, "{r:?}");
        assert!(r.t_s > 2.0 && r.t_s < 6.0, "{r:?}");
    }

    #[test]
    fn a_taller_body_falls_longer_and_hits_harder() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let tall = fall(&FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Pushed, 0.0, 0.3, Surface::Firm));
        let low_pose = KinPose { hip: [-0.06, crate::dims::Dims::COMPACT.hip_m - 0.18], pelvis_deg: 25.0, ..KinPose::stand_compact(0.0) };
        let low = fall(&FallInput::lost_balance(&b, low_pose, FallCause::Pushed, 0.0, 0.3, Surface::Firm));
        assert!(tall.fell && low.fell, "{:?} {:?}", tall.rollout.events, low.rollout.events);
        let (mt, ml) = (tall.main.clone().unwrap(), low.main.clone().unwrap());
        assert!(tall.t_first_s > low.t_first_s, "{} vs {} s", tall.t_first_s, low.t_first_s);
        assert!(tall.rollout.metric("com_drop_m") > low.rollout.metric("com_drop_m"));
        assert!(tall.head_v.unwrap() > low.head_v.unwrap() && tall.pelvis_v.unwrap() > low.pelvis_v.unwrap(), "head {:?} vs {:?}, pelvis {:?} vs {:?}", tall.head_v, low.head_v, tall.pelvis_v, low.pelvis_v);
        assert!(mt.energy_j > ml.energy_j, "{} vs {} J", mt.energy_j, ml.energy_j);
        assert!(tall.lying.com[2] < 0.3 && tall.lying.down(), "{:?}", tall.lying);
        assert!(tall.get_up.can && !tall.get_up.needs_help, "{:?}", tall.get_up);
        assert!(matches!(mt.segment, Segment::Head | Segment::Chest | Segment::Arm | Segment::Hand), "{mt:?}");
        assert!(tall.head_v.unwrap() > tall.pelvis_v.unwrap(), "the head is further from the edge than the pelvis");
        let side = fall(&FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Pushed, 90.0, 0.3, Surface::Firm));
        assert!(side.fell && side.pelvis_v.unwrap() > 0.5, "{:?}", side.pelvis_v);
        let head = tall.lying.place([0.0, 0.0, 1.14]);
        assert!(head[0] > 0.8 && head[2] < 0.3, "{head:?}");
    }

    #[test]
    fn wheels_cannot_get_up_and_soft_ground_hits_softer() {
        let w = body(Tier::Mid, BaseKind::Wheel);
        let f = fall(&FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Pushed, 180.0, 0.3, Surface::Firm));
        assert!(f.fell && !f.get_up.can && f.get_up.needs_help && f.get_up.righting_force_n > 0.0, "{:?} {:?}", f.get_up, f.rollout.events);
        let g = fall(&FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Pushed, 180.0, 0.3, Surface::Sand));
        let (pf, pg) = (f.main.unwrap(), g.main.unwrap());
        assert!(pg.peak_n < pf.peak_n && pg.contact_s > pf.contact_s, "{} vs {}", pg.peak_n, pf.peak_n);
    }

    #[test]
    fn a_bumper_catches_a_wheeled_base_going_over_slowly_and_a_fast_one_rolls_over_it() {
        let w = body(Tier::Mid, BaseKind::Wheel);
        let solved = kin::solve(&w, &KinPose::stand_compact(0.0), &KinLoads::default());
        let pl = Planar::of(&collide::mass_props_of(&w, &solved), solved.support, [1.0, 0.0]);
        let catch = crate::dims::Dims::COMPACT.bumper_z_m[0].atan2(crate::dims::Dims::COMPACT.wheel_r_m + crate::dims::Dims::COMPACT.bumper_gap_m).to_degrees();
        assert!(catch > pl.tip_angle(pl.ahead_m).to_degrees(), "the bumper catches past the tip angle");
        let slow = fall(&FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Bumped, 0.0, 0.01, Surface::Firm));
        assert!(slow.strikes.iter().any(|s| s.link == "bumper"), "{:?}", slow.strikes.iter().map(|s| s.link).collect::<Vec<_>>());
        assert!(slow.fell && !slow.lying.down() && (slow.lying.tilt_deg - catch).abs() < 1.5, "{:?} {:?}", slow.lying.tilt_deg, slow.rollout.events);
        assert!(slow.get_up.needs_help);
        let fast = fall(&FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Bumped, 0.0, 1.0, Surface::Firm));
        assert!(fast.fell && fast.lying.down(), "{:?}", fast.lying.tilt_deg);
        let first = |l: &[&str]| fast.strikes.iter().position(|s| l.contains(&s.link));
        assert!(first(&["bumper"]).is_some() && first(&["bumper"]) < first(&["trunk", "head"]), "the bumper strikes first: {:?}", fast.strikes.iter().map(|s| s.link).collect::<Vec<_>>());
    }

    #[test]
    fn an_edge_fall_lands_harder_than_a_fall_on_the_flat() {
        let w = body(Tier::Mid, BaseKind::Track);
        let flat = fall(&FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Pushed, 180.0, 0.3, Surface::Firm));
        let edge = fall(&FallInput { drop_m: 0.8, ..FallInput::lost_balance(&w, KinPose::stand_compact(0.0), FallCause::Edge, 180.0, 0.3, Surface::Firm) });
        assert!(edge.rollout.events[0].kind == "edge_fall" && edge.fell);
        assert!(edge.main.as_ref().unwrap().v_n > flat.main.as_ref().unwrap().v_n, "{:?} vs {:?}", edge.main, flat.main);
        assert!(edge.lying.com[2] < 0.0, "below the edge: {:?}", edge.lying.com);
    }

    fn energy(p: &Played, st: &Rigid2) -> f32 {
        kinetic_j(p.mp.mass_kg, p.pl.inertia, [st.v[0], st.v[1], st.w]) + p.mp.mass_kg * G * st.c[1]
    }

    #[test]
    fn strikes_and_turns_never_add_energy_and_every_fall_settles() {
        let mut bad: Vec<String> = Vec::new();
        let (mut n, mut n_face, mut n_edge, mut worst_ke) = (0usize, 0usize, 0usize, (0.0f32, String::new()));
        let mut run = |inp: &FallInput, tag: String| {
            let p = play(inp);
            n += 1;
            let scale = p.mp.mass_kg * G * p.pl.com_z;
            for h in &p.rk.hits {
                let e0 = energy(&p, &h.before);
                let e1 = energy(&p, &Rigid2 { v: [h.after[0], h.after[1]], w: h.after[2], ..h.before });
                if e1 > e0 + 1e-3 * scale {
                    bad.push(format!("{tag}: strike of {} at {:.3} s: {e0:.2} -> {e1:.2} J", h.at.link, h.t));
                }
                n_face += usize::from(!h.wall && !h.corner && h.n[1] < 0.5);
                n_edge += usize::from(h.corner);
            }
            for w in p.rk.track.windows(2) {
                let gain = energy(&p, &w[1].1) - energy(&p, &w[0].1);
                if gain > 0.01 * scale {
                    bad.push(format!("{tag}: {:.3}-{:.3} s gains {gain:.2} J of {scale:.0}", w[0].0, w[1].0));
                }
            }
            if p.rk.end != RockEnd::Rest || p.rk.capped || p.rk.hits.len() >= MAX_STRIKES {
                let l = p.rk.track.iter().rev().nth(1).map_or(p.rk.last, |x| x.1);
                bad.push(format!("{tag}: not at rest at {:.2} s ({:?}, {} strikes, {} settles, {:.1}°, v ({:.2}, {:.2}), ω {:.2}, {:.2} J)", p.rk.t, p.rk.end, p.rk.hits.len(), p.rk.settles, l.th.to_degrees(), l.v[0], l.v[1], l.w, p.rk.ke_end));
            }
            let ke = if p.rk.rest == Some("trapped") { 0.0 } else { p.rk.ke_left / scale };
            if ke > worst_ke.0 {
                worst_ke = (ke, tag.clone());
            }
            if ke > 0.02 {
                let v = p.rk.v_end;
                bad.push(format!("{tag}: set still with {:.2} J of {scale:.0} ({:?}, v ({:.2}, {:.2}), ω {:.2}, {:.0}°)", p.rk.ke_end, p.rk.rest, v[0], v[1], v[2], p.rk.last.th.to_degrees()));
            }
        };
        let tiers = [Tier::Budget, Tier::Mid, Tier::Pro];
        for tier in tiers {
            for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track] {
                let b = body(tier, base);
                for dir in [0.0f32, 90.0, 180.0] {
                    for drop in [0.0f32, 0.05, 0.5] {
                        let inp = FallInput { drop_m: drop, ..FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Edge, dir, 0.3, Surface::Firm) };
                        run(&inp, format!("{tier:?} {base:?} dir {dir} drop {drop}"));
                    }
                }
            }
        }
        for tier in tiers {
            for base in [BaseKind::HumanFoot, BaseKind::Wheel, BaseKind::Track, BaseKind::Mecanum] {
                let b = body(tier, base);
                for dir in [0.0f32, 45.0, 90.0, 135.0, 180.0] {
                    for speed in [0.0f32, 0.3, 1.0] {
                        for drop in [0.15f32, 0.5] {
                            run(&FallInput::over_edge(&b, KinPose::stand_compact(0.0), dir, speed, drop, Surface::Firm), format!("{tier:?} {base:?} over edge dir {dir} at {speed} m/s drop {drop}"));
                        }
                    }
                }
                for dir in [45.0f32, 135.0] {
                    for omega0 in [0.05f32, 1.5] {
                        for drop in [0.0f32, 0.1, 0.5] {
                            let inp = FallInput { drop_m: drop, ..FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Edge, dir, omega0, Surface::Firm) };
                            run(&inp, format!("{tier:?} {base:?} dir {dir} ω0 {omega0} drop {drop}"));
                        }
                    }
                }
                for ground in [Surface::Sand, Surface::Water] {
                    for dir in [0.0f32, 90.0, 180.0] {
                        for drop in [0.0f32, 0.5] {
                            let inp = FallInput { drop_m: drop, ..FallInput::lost_balance(&b, KinPose::stand_compact(0.0), FallCause::Edge, dir, 0.3, ground) };
                            run(&inp, format!("{tier:?} {base:?} {ground:?} dir {dir} drop {drop}"));
                        }
                        run(&FallInput::over_edge(&b, KinPose::stand_compact(0.0), dir, 0.3, 0.5, ground), format!("{tier:?} {base:?} {ground:?} over edge dir {dir}"));
                    }
                }
            }
        }
        println!("[FALL] {n} falls: {} unsettled or gaining, {n_face} strikes on a step's face, {n_edge} of a step's corner on a link; most motion set still {:.4} of m·g·h ({})", bad.len(), worst_ke.0, worst_ke.1);
        assert!(bad.is_empty(), "{}", bad.join("\n"));
        assert!(n_face > 0, "a corner swung back under a step meets its face");
        assert!(n_edge > 0, "the step's corner meets the underside of a body going over it");
    }

    #[test]
    fn the_direction_draw_follows_the_video_counts() {
        let n = 10000;
        let fwd = (0..n).filter(|i| initial_direction((*i as f32 + 0.5) / n as f32) == InitialDir::Forward).count() as f32 / n as f32;
        assert!((fwd - 51.0 / 199.0).abs() < 0.01, "{fwd}");
        assert_eq!(direction_deg(InitialDir::Backward, 0.3), 180.0);
        assert!(direction_deg(InitialDir::Sideways, 0.9).abs() >= 90.0);
        assert_eq!(dir_label_ko(0.0), "앞으로");
        assert_eq!(dir_label_ko(180.0), "뒤로");
    }
}
