use serde::Serialize;

use crate::balance::{Body, G, GRIP_SAFETY};
use crate::base::BaseKind;
use crate::hand::NotoConfig;
use crate::kin::{self, KinLoads, KinPose};
use crate::surface::{self, Surface};

use super::material::{self, ItemMaterial};
use super::{gait, grasp, ActEvent, Part};

pub const HSE_TEAM_FACTOR: f32 = 0.85;
pub const TEAM_FACTOR_2: f32 = 2.0 / 3.0;
pub const TEAM_FACTOR_3: f32 = 0.5;
pub const TEAM_MAX: usize = 4;
pub const CARRY_REACH_M: f32 = 0.35;
pub const HANDS_SPAN_M: f32 = 0.30;
const REACH_TOL_M: f32 = 0.04;
const SAME_GRIP_M: f32 = 1e-3;
pub const PALM_HALF_W_M: f32 = 0.035;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamRule {
    Hse2016,
    Conservative,
}

impl TeamRule {
    pub fn factor(self, n: usize) -> f32 {
        match (self, n) {
            (_, 0 | 1) => 1.0,
            (TeamRule::Hse2016, _) => HSE_TEAM_FACTOR,
            (TeamRule::Conservative, 2) => TEAM_FACTOR_2,
            (TeamRule::Conservative, _) => TEAM_FACTOR_3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GripKind {
    Under,
    Squeeze,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Load {
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub length_m: f32,
    pub thick_m: f32,
    pub com_along: f32,
    pub com_up_m: f32,
    pub grip: GripKind,
}

impl Load {
    pub fn log(mass_kg: f32, length_m: f32, diameter_m: f32, under: bool) -> Load {
        Load { material: ItemMaterial::Wood, mass_kg, length_m, thick_m: diameter_m, com_along: 0.5, com_up_m: if under { 0.5 * diameter_m } else { 0.0 }, grip: if under { GripKind::Under } else { GripKind::Squeeze } }
    }

    pub fn rock(mass_kg: f32, size_m: f32) -> Load {
        Load { material: ItemMaterial::Stone, mass_kg, length_m: size_m, thick_m: size_m, com_along: 0.5, com_up_m: 0.25 * size_m, grip: GripKind::Under }
    }

    pub fn weight_n(&self) -> f32 {
        self.mass_kg * G
    }

    pub fn pitch_inertia(&self) -> f32 {
        self.mass_kg * (self.length_m * self.length_m + self.thick_m * self.thick_m) / 12.0
    }
}

#[derive(Clone, Debug)]
pub struct Carrier<'a> {
    pub body: &'a Body,
    pub grip: [f32; 3],
    pub ground_z_m: f32,
    pub ground: Surface,
    pub slope_deg: f32,
    pub stiffness: f32,
    pub hand_low_m: f32,
}

impl<'a> Carrier<'a> {
    pub fn at(body: &'a Body, x: f32, y: f32, ground_z_m: f32, ground: Surface) -> Carrier<'a> {
        Carrier { body, grip: [x, y, ground_z_m + carry_hand_z(body)], ground_z_m, ground, slope_deg: 0.0, stiffness: 1.0, hand_low_m: 0.0 }
    }

    pub fn hand_z(&self) -> f32 {
        self.grip[2] - self.ground_z_m
    }
}

pub fn carry_hand_z(body: &Body) -> f32 {
    kin::solve_fast(body, &grasp::hold_pose(true, 0.0), &KinLoads::default()).hands[0][2]
}

pub fn carry_pose(body: &Body, hand_z: f32) -> KinPose {
    grasp::arms_to(body, &grasp::hold_pose(true, 0.0), [CARRY_REACH_M, hand_z], true)
}

fn holds_in_hands(name: &str) -> bool {
    name.starts_with("shoulder") || name.starts_with("elbow") || name == "waist_pitch" || name == "torso_lift"
}

pub fn hold_capacity_kg(body: &Body, hand_z: f32) -> f32 {
    let pose = carry_pose(body, hand_z);
    let ok = |kg: f32| kin::solve_fast(body, &pose, &KinLoads::holding(kg, true, 0.0)).joints.iter().filter(|j| !j.structural && holds_in_hands(&j.name)).all(|j| j.effort <= 1.0);
    if !ok(0.0) {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0f32, 200.0f32);
    if ok(hi) {
        return hi;
    }
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if ok(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

pub fn member_capacity_kg(body: &Body, ground: Surface, slope_deg: f32, hand_z: f32) -> f32 {
    body.capacity_on(&body.base.spec(), ground, slope_deg).min(hold_capacity_kg(body, hand_z))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Statics {
    pub forces: Vec<f32>,
    pub com: [f32; 3],
    pub incline_deg: f32,
    pub collinear: bool,
    pub determinate: bool,
    pub balanced: bool,
    pub roll_nm: f32,
    pub along: Vec<f32>,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: [f32; 3]) -> [f32; 3] {
    let n = dot(a, a).sqrt().max(1e-9);
    [a[0] / n, a[1] / n, a[2] / n]
}

fn solve_lin(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let s: f32 = (i + 1..n).map(|k| a[i][k] * x[k]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    Some(x)
}

fn line_ends(grips: &[[f32; 3]]) -> (usize, usize) {
    let (mut ia, mut ib, mut best) = (0usize, 0usize, -1.0f32);
    for i in 0..grips.len() {
        for j in i + 1..grips.len() {
            let d = dot(sub(grips[j], grips[i]), sub(grips[j], grips[i]));
            if d > best {
                best = d;
                ia = i;
                ib = j;
            }
        }
    }
    (ia.min(ib), ia.max(ib))
}

const LINE_TOL_M: f32 = 0.05;

fn in_line(grips: &[[f32; 3]]) -> bool {
    if grips.len() <= 2 {
        return true;
    }
    let (ia, ib) = line_ends(grips);
    let e = unit(sub(grips[ib], grips[ia]));
    grips.iter().all(|p| {
        let d = sub(*p, grips[ia]);
        let a = dot(d, e);
        let r = sub(d, [e[0] * a, e[1] * a, e[2] * a]);
        dot(r, r).sqrt() < LINE_TOL_M
    })
}

pub fn object_com(load: &Load, grips: &[[f32; 3]]) -> [f32; 3] {
    let n = grips.len();
    let up = [0.0f32, 0.0, 1.0];
    if n == 0 {
        return [0.0; 3];
    }
    if n == 1 {
        let g = grips[0];
        return [g[0], g[1], g[2] + load.com_up_m];
    }
    let off = |p: [f32; 3], nrm: [f32; 3]| [p[0] + load.com_up_m * nrm[0], p[1] + load.com_up_m * nrm[1], p[2] + load.com_up_m * nrm[2]];
    if in_line(grips) {
        let (ia, ib) = line_ends(grips);
        let line = sub(grips[ib], grips[ia]);
        let e = unit(line);
        let nrm = unit(sub(up, [e[0] * e[2], e[1] * e[2], e[2] * e[2]]));
        let a = load.com_along;
        off([grips[ia][0] + a * line[0], grips[ia][1] + a * line[1], grips[ia][2] + a * line[2]], nrm)
    } else {
        let mut nv = [0.0f32; 3];
        for i in 0..n {
            let (p, q) = (grips[i], grips[(i + 1) % n]);
            nv[0] += (p[1] - q[1]) * (p[2] + q[2]);
            nv[1] += (p[2] - q[2]) * (p[0] + q[0]);
            nv[2] += (p[0] - q[0]) * (p[1] + q[1]);
        }
        let mut nrm = unit(nv);
        if nrm[2] < 0.0 {
            nrm = [-nrm[0], -nrm[1], -nrm[2]];
        }
        let c = grips.iter().fold([0.0f32; 3], |s, p| [s[0] + p[0] / n as f32, s[1] + p[1] / n as f32, s[2] + p[2] / n as f32]);
        off(c, nrm)
    }
}

pub fn statics(load: &Load, grips: &[[f32; 3]], k: &[f32], low: &[f32]) -> Statics {
    statics_at(load, grips, object_com(load, grips), k, low)
}

pub fn statics_at(load: &Load, grips: &[[f32; 3]], com: [f32; 3], k: &[f32], low: &[f32]) -> Statics {
    let w = load.weight_n();
    let n = grips.len();
    if n == 0 {
        return Statics { forces: vec![], com, incline_deg: 0.0, collinear: true, determinate: true, balanced: false, roll_nm: 0.0, along: vec![] };
    }
    let (ia, ib) = line_ends(grips);
    let e = unit(sub(grips[ib], grips[ia]));
    let horiz = [e[0], e[1], 0.0];
    let hlen = dot(horiz, horiz).sqrt();
    let collinear = in_line(grips);
    let incline_deg = if n >= 2 && collinear { e[2].atan2(hlen).to_degrees() } else { 0.0 };
    if n == 1 {
        let d = ((com[0] - grips[0][0]).powi(2) + (com[1] - grips[0][1]).powi(2)).sqrt();
        let balanced = d <= 0.5 * HANDS_SPAN_M;
        return Statics { forces: vec![w], com, incline_deg, collinear: true, determinate: true, balanced, roll_nm: w * d, along: vec![0.0] };
    }
    let kk: Vec<f32> = (0..n).map(|i| k.get(i).copied().unwrap_or(1.0).max(1e-6)).collect();
    let dl: Vec<f32> = (0..n).map(|i| -low.get(i).copied().unwrap_or(0.0)).collect();
    let (forces, determinate, roll, along) = if collinear {
        let h = if hlen > 1e-6 { [horiz[0] / hlen, horiz[1] / hlen, 0.0] } else { [1.0, 0.0, 0.0] };
        let s: Vec<f32> = grips.iter().map(|p| dot(sub(*p, com), h)).collect();
        let side = [-h[1], h[0], 0.0];
        let roll = w * dot(sub(com, grips[ia]), side);
        let (sk, sks, skss) = (kk.iter().sum::<f32>(), (0..n).map(|i| kk[i] * s[i]).sum::<f32>(), (0..n).map(|i| kk[i] * s[i] * s[i]).sum::<f32>());
        let (skd, skds) = ((0..n).map(|i| kk[i] * dl[i]).sum::<f32>(), (0..n).map(|i| kk[i] * dl[i] * s[i]).sum::<f32>());
        let x = solve_lin(vec![vec![sk, sks], vec![sks, skss]], vec![skd - w, skds]);
        let f = match x {
            Some(x) => (0..n).map(|i| kk[i] * (dl[i] - x[0] - x[1] * s[i])).collect(),
            None => vec![w / n as f32; n],
        };
        (f, n == 2, roll, s)
    } else {
        let d: Vec<[f32; 2]> = grips.iter().map(|p| [p[0] - com[0], p[1] - com[1]]).collect();
        let sum = |f: &dyn Fn(usize) -> f32| (0..n).map(f).sum::<f32>();
        let a = vec![
            vec![sum(&|i| kk[i]), sum(&|i| kk[i] * d[i][0]), sum(&|i| kk[i] * d[i][1])],
            vec![sum(&|i| kk[i] * d[i][0]), sum(&|i| kk[i] * d[i][0] * d[i][0]), sum(&|i| kk[i] * d[i][0] * d[i][1])],
            vec![sum(&|i| kk[i] * d[i][1]), sum(&|i| kk[i] * d[i][0] * d[i][1]), sum(&|i| kk[i] * d[i][1] * d[i][1])],
        ];
        let b = vec![sum(&|i| kk[i] * dl[i]) - w, sum(&|i| kk[i] * dl[i] * d[i][0]), sum(&|i| kk[i] * dl[i] * d[i][1])];
        let f = match solve_lin(a, b) {
            Some(x) => (0..n).map(|i| kk[i] * (dl[i] - x[0] - x[1] * d[i][0] - x[2] * d[i][1])).collect(),
            None => vec![w / n as f32; n],
        };
        (f, n == 3, 0.0, vec![])
    };
    let rolls_off = collinear && roll.abs() > w * PALM_HALF_W_M;
    let balanced = load.grip == GripKind::Squeeze || (forces.iter().all(|f| *f >= -1e-3 * w) && !rolls_off);
    Statics { forces, com, incline_deg, collinear, determinate, balanced, roll_nm: roll, along }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    InPhase,
    AntiPhase,
    Worst,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Dynamic {
    pub step_hz: f32,
    pub amp_m: Vec<f32>,
    pub in_phase_n: Vec<f32>,
    pub anti_phase_n: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CarryOpts {
    pub rule: TeamRule,
    pub phase: Phase,
    pub speed_mps: Option<f32>,
    pub even_axial: bool,
}

impl Default for CarryOpts {
    fn default() -> Self {
        CarryOpts { rule: TeamRule::Hse2016, phase: Phase::Worst, speed_mps: None, even_axial: false }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MemberShare {
    pub index: usize,
    pub f_n: f32,
    pub share: f32,
    pub f_peak_n: f32,
    pub load_kg: f32,
    pub peak_kg: f32,
    pub capacity_kg: f32,
    pub grip_use: f32,
    pub peak_ratio: f32,
    pub margin: f32,
    pub reaches: bool,
    pub ok: bool,
    pub why_ko: Option<String>,
    pub pose: KinPose,
    pub loads: KinLoads,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoCarry {
    pub statics: Statics,
    pub members: Vec<MemberShare>,
    pub axial_even_n: f32,
    pub axial_even_ok: bool,
    pub axial_n: f32,
    pub dynamic: Option<Dynamic>,
    pub speed_mps: f32,
    pub team_limit_kg: f32,
    pub weakest_limit_kg: f32,
    pub limit_kg: f32,
    pub rule: TeamRule,
    pub ok: bool,
    pub events: Vec<ActEvent>,
    pub notes: Vec<String>,
}

impl CoCarry {
    pub fn share_ratio(&self) -> f32 {
        let f: Vec<f32> = self.members.iter().map(|m| m.f_n).collect();
        let hi = f.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lo = f.iter().copied().fold(f32::INFINITY, f32::min);
        hi / lo.max(1e-6)
    }
}

fn hand_of(body: &Body) -> Option<crate::hand::Hand> {
    let mut cfg = NotoConfig::preset(body.fingers);
    cfg.base = body.base;
    cfg.build().ok()
}

fn bob(c: &Carrier, speed: f32) -> (f32, f32) {
    if c.body.base != BaseKind::HumanFoot {
        return (0.0, 0.0);
    }
    let mu = surface::available_mu(c.body.base, c.ground.props());
    let g = gait::plan(c.body, speed, mu, c.slope_deg);
    (0.5 * g.com_amp_m, 1.0 / g.step_s.max(1e-3))
}

fn fluctuation(load: &Load, s: [f32; 2], amp: [f32; 2], hz: f32, lag: f32) -> [f32; 2] {
    if hz <= 0.0 || (amp[0] == 0.0 && amp[1] == 0.0) {
        return [0.0; 2];
    }
    let m = load.mass_kg;
    let i_g = load.pitch_inertia();
    let w = std::f32::consts::TAU * hz;
    if (s[1] - s[0]).abs() < SAME_GRIP_M {
        let peak = (0..64)
            .map(|k| {
                let t = k as f32 / 64.0 / hz;
                0.5 * m * (0.5 * (amp[0] * (w * t).sin() + amp[1] * (w * t + lag).sin()) * w * w).abs()
            })
            .fold(0.0f32, f32::max);
        return [peak, peak];
    }
    let lam = (0.0 - s[0]) / (s[1] - s[0]);
    let mut peak = [0.0f32; 2];
    for k in 0..64 {
        let t = k as f32 / 64.0 / hz;
        let a1 = -amp[0] * w * w * (w * t).sin();
        let a2 = -amp[1] * w * w * (w * t + lag).sin();
        let a_g = a1 * (1.0 - lam) + a2 * lam;
        let th_dd = (a2 - a1) / (s[1] - s[0]);
        let f2 = (i_g * th_dd - s[0] * m * a_g) / (s[1] - s[0]);
        let f1 = m * a_g - f2;
        peak[0] = peak[0].max(f1.abs());
        peak[1] = peak[1].max(f2.abs());
    }
    peak
}

pub fn share(load: &Load, carriers: &[Carrier], opts: &CarryOpts) -> CoCarry {
    let grips: Vec<[f32; 3]> = carriers.iter().map(|c| c.grip).collect();
    share_at(load, carriers, object_com(load, &grips), opts)
}

pub fn share_at(load: &Load, carriers: &[Carrier], com: [f32; 3], opts: &CarryOpts) -> CoCarry {
    let n = carriers.len();
    let w = load.weight_n();
    let grips: Vec<[f32; 3]> = carriers.iter().map(|c| c.grip).collect();
    let k: Vec<f32> = carriers.iter().map(|c| c.stiffness).collect();
    let low: Vec<f32> = carriers.iter().map(|c| c.hand_low_m).collect();
    let st = statics_at(load, &grips, com, &k, &low);
    let mut notes = Vec::new();
    let mut events = Vec::new();
    let mut forces = st.forces.clone();
    let th = st.incline_deg.to_radians();
    let (mut axial_even, mut axial_ok, mut axial) = (0.0f32, true, 0.0f32);
    if n == 2 && th.abs() > 1e-3 {
        axial_even = (forces[0] - forces[1]) / (2.0 * th.sin());
        let hold = |i: usize| {
            let c = &carriers[i];
            surface::available_mu(c.body.base, c.ground.props()) * (c.body.mass_kg() * G + forces[i].max(0.0))
        };
        let need = (axial_even * th.cos()).abs();
        axial_ok = need <= hold(0).min(hold(1));
        if opts.even_axial {
            let cap = hold(0).min(hold(1)) / th.cos().abs().max(1e-3);
            axial = axial_even.clamp(-cap, cap);
            forces[0] -= axial * th.sin();
            forces[1] += axial * th.sin();
            notes.push(format!("축 방향 내력 {:.0} N으로 아래쪽 하중을 위쪽으로 넘김", axial));
        }
    }
    let speed = opts.speed_mps.unwrap_or_else(|| {
        carriers
            .iter()
            .zip(&forces)
            .map(|(c, f)| c.body.drive_limits(&c.body.base.spec(), (f / G).max(0.0)).max_speed_mps)
            .fold(f32::INFINITY, f32::min)
            .min(1.0)
    });
    let dynamic = (n == 2 && st.collinear).then(|| {
        let b: Vec<(f32, f32)> = carriers.iter().map(|c| bob(c, speed)).collect();
        let hz = b.iter().map(|x| x.1).fold(0.0f32, f32::max);
        let amp = [b[0].0, b[1].0];
        let s = [st.along[0], st.along[1]];
        let ip = fluctuation(load, s, amp, hz, 0.0);
        let ap = fluctuation(load, s, amp, hz, std::f32::consts::PI);
        Dynamic { step_hz: hz, amp_m: amp.to_vec(), in_phase_n: ip.to_vec(), anti_phase_n: ap.to_vec() }
    });
    let extra = |i: usize| -> f32 {
        dynamic.as_ref().map_or(0.0, |d| match opts.phase {
            Phase::InPhase => d.in_phase_n[i],
            Phase::AntiPhase => d.anti_phase_n[i],
            Phase::Worst => d.in_phase_n[i].max(d.anti_phase_n[i]),
        })
    };
    let mu_pad = material::pad_on(load.material).mean;
    let mut members = Vec::new();
    for (i, c) in carriers.iter().enumerate() {
        let f = forces[i];
        let f_peak = f + extra(i);
        let hz = c.hand_z();
        let pose = carry_pose(c.body, hz);
        let loads = KinLoads { hand_force: [[0.0, 0.0, -0.5 * f_peak], [0.0, 0.0, -0.5 * f_peak]], ..KinLoads::default() };
        let r = kin::solve(c.body, &pose, &loads);
        let reaches = (r.hands[0][2] - hz).abs() <= REACH_TOL_M;
        let capacity = member_capacity_kg(c.body, c.ground, c.slope_deg, hz);
        let grip_use = match load.grip {
            GripKind::Under => GRIP_SAFETY * th.tan().abs() / mu_pad.max(1e-3),
            GripKind::Squeeze => match hand_of(c.body) {
                Some(h) => {
                    let cap = grasp::capacity(&h, [load.thick_m; 3], (f_peak / G).max(0.0), mu_pad);
                    GRIP_SAFETY * f_peak.abs() / (mu_pad * 2.0 * cap.squeeze_n).max(1e-3)
                }
                None => f32::INFINITY,
            },
        };
        let peak_ratio = r.worst().map_or(0.0, |j| j.ratio);
        let why = if f < -1e-3 * w && load.grip == GripKind::Under {
            Some(format!("물체를 아래로 눌러야 함 ({:.0} N): 손을 받친 채로는 반대쪽으로 넘어감", f))
        } else if !reaches {
            Some(format!("손이 {:.2} m 높이에 닿지 않음 ({:.2} m까지)", hz, r.hands[0][2]))
        } else if f / G > capacity {
            Some(format!("몫 {:.1} kg > 계속 나를 수 있는 {:.1} kg", f / G, capacity))
        } else if peak_ratio > 1.0 {
            Some(format!("걸음 최고 하중 {:.1} kg에서 {}", f_peak / G, r.limiting.clone().unwrap_or_default()))
        } else if r.margin < 0.0 {
            Some(format!("균형 여유 {:.2} < 0", r.margin))
        } else if grip_use > 1.0 {
            Some(format!("손이 미끄러짐: 필요 마찰 {:.2} > 패드 {:.2}", grip_use * mu_pad, mu_pad))
        } else {
            None
        };
        if let Some(wk) = &why {
            events.push(ActEvent::new(0.0, "overload_share", format!("{}번 로봇: {wk}", i + 1), &[Part::Hand, Part::Waist], Some(f / G), false));
        }
        members.push(MemberShare { index: i, f_n: f, share: f / w.max(1e-6), f_peak_n: f_peak, load_kg: f / G, peak_kg: f_peak / G, capacity_kg: capacity, grip_use, peak_ratio, margin: r.margin, reaches, ok: why.is_none(), why_ko: why, pose, loads });
    }
    let rule_limit = opts.rule.factor(n) * members.iter().map(|m| m.capacity_kg).sum::<f32>();
    let weakest = members.iter().filter(|m| m.share > 1e-3).map(|m| m.capacity_kg / m.share).fold(f32::INFINITY, f32::min);
    let limit = rule_limit.min(weakest);
    if n > TEAM_MAX {
        notes.push(format!("{n}대: 넷을 넘는 팀은 잘 맞추기 어려움"));
    }
    if !st.balanced {
        events.push(ActEvent::new(0.0, "co_drop", format!("잡은 곳들이 무게중심을 받치지 못해 물체가 넘어감 (몫 {:?})", st.forces.iter().map(|f| (f / G * 10.0).round() / 10.0).collect::<Vec<_>>()), &[Part::Hand], None, false));
    }
    if load.mass_kg > limit + 1e-3 {
        events.push(ActEvent::new(0.0, "overload_share", format!("팀 한계 {:.1} kg (규칙 {:.1} · 가장 약한 로봇 {:.1}) < 물체 {:.1} kg", limit, rule_limit, weakest, load.mass_kg), &[Part::Hand, Part::Waist], Some(load.mass_kg / limit.max(1e-3)), false));
    }
    let ok = st.balanced && members.iter().all(|m| m.ok) && load.mass_kg <= limit + 1e-3;
    if let Some(d) = &dynamic {
        notes.push(format!("걸음 {:.2} Hz: 같은 위상 ±{:.0}/{:.0} N, 반대 위상 ±{:.0}/{:.0} N", d.step_hz, d.in_phase_n[0], d.in_phase_n[1], d.anti_phase_n[0], d.anti_phase_n[1]));
    }
    CoCarry { statics: st, members, axial_even_n: axial_even, axial_even_ok: axial_ok, axial_n: axial, dynamic, speed_mps: speed, team_limit_kg: rule_limit, weakest_limit_kg: weakest, limit_kg: limit, rule: opts.rule, ok, events, notes }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LetGo {
    pub after: Option<CoCarry>,
    pub dropped: bool,
    pub instant_n: Vec<f32>,
    pub end_on_ground_share: Option<f32>,
    pub end_held: bool,
    pub end_drop_m: f32,
    pub events: Vec<ActEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SequentialCoCarry {
    pub active_index: usize,
    pub assumed_partner_index: usize,
    pub effective_load_kg: f32,
    pub effective_peak_kg: f32,
    pub share_ratio: f32,
    pub team_ok: bool,
    pub active_member_ok: bool,
    pub cocarry: CoCarry,
}

pub fn share_sequential(
    load: &Load,
    carriers: &[Carrier],
    active_index: usize,
    opts: &CarryOpts,
) -> SequentialCoCarry {
    let cc = share(load, carriers, opts);
    let m = &cc.members[active_index];
    let partner_idx = if active_index == 0 { 1 } else { 0 };
    SequentialCoCarry {
        active_index,
        assumed_partner_index: partner_idx,
        effective_load_kg: m.load_kg,
        effective_peak_kg: m.peak_kg,
        share_ratio: m.share,
        team_ok: cc.ok,
        active_member_ok: m.ok,
        cocarry: cc,
    }
}

pub fn let_go(load: &Load, carriers: &[Carrier], i: usize, opts: &CarryOpts) -> LetGo {
    let rest: Vec<Carrier> = carriers.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, c)| c.clone()).collect();
    let mut events = vec![ActEvent::new(0.0, "co_drop", format!("{}번 로봇이 놓음: 나머지 {}대가 다시 나눔", i + 1, rest.len()), &[Part::Hand], None, false)];
    let w = load.weight_n();
    let end_drop = carriers.get(i).map_or(0.0, |c| c.hand_z());
    if rest.is_empty() {
        events.push(ActEvent::new(0.0, "co_drop", "잡은 로봇이 없어 물체가 떨어짐", &[Part::Hand], Some(end_drop), false));
        return LetGo { after: None, dropped: true, instant_n: vec![], end_on_ground_share: None, end_held: false, end_drop_m: end_drop, events };
    }
    let com = object_com(load, &carriers.iter().map(|c| c.grip).collect::<Vec<_>>());
    let after = share_at(load, &rest, com, opts);
    if after.statics.balanced {
        let held = after.ok;
        if !held {
            events.push(ActEvent::new(0.0, "overload_share", format!("나머지가 받치지만 한계를 넘음 (한계 {:.1} kg)", after.limit_kg), &[Part::Hand, Part::Waist], Some(load.mass_kg), false));
        }
        let instant = after.members.iter().map(|m| m.f_n).collect();
        return LetGo { after: Some(after), dropped: false, instant_n: instant, end_on_ground_share: None, end_held: held, end_drop_m: 0.0, events };
    }
    let mut instant = Vec::new();
    let mut end_share = None;
    let mut end_held = false;
    if rest.len() == 1 && in_line(&carriers.iter().map(|c| c.grip).collect::<Vec<_>>()) {
        let h = &rest[0];
        let full = statics(load, &carriers.iter().map(|c| c.grip).collect::<Vec<_>>(), &[], &[]);
        let holder = (0..carriers.len()).find(|j| *j != i).unwrap_or(0);
        let s_h = full.along.get(holder).copied().unwrap_or(0.0);
        let s_free = full.along.get(i).copied().unwrap_or(0.0);
        let d = s_h.abs();
        let i_g = load.pitch_inertia();
        instant.push(w * i_g / (i_g + load.mass_kg * d * d));
        let span = (full.along.iter().copied().fold(f32::NEG_INFINITY, f32::max) - full.along.iter().copied().fold(f32::INFINITY, f32::min)).max(1e-3);
        let over = 0.5 * (load.length_m - span).max(0.0);
        let s_end = s_free + s_free.signum() * over;
        let share_h = (s_end / (s_end - s_h)).abs().min(1.0);
        end_share = Some(share_h);
        let cap = member_capacity_kg(h.body, h.ground, h.slope_deg, h.hand_z());
        end_held = share_h * load.mass_kg <= cap;
        events.push(ActEvent::new(
            0.0,
            "co_drop",
            format!("한쪽 끝이 {:.2} m에서 떨어져 땅에 닿음: 남은 로봇이 순간 {:.0} N, 그 뒤 {:.0}% ({:.1} kg){}", end_drop, instant[0], share_h * 100.0, share_h * load.mass_kg, if end_held { "를 받침" } else { "도 못 받쳐 놓침" }),
            &[Part::Hand, Part::Waist],
            Some(share_h * load.mass_kg),
            false,
        ));
    } else {
        events.push(ActEvent::new(0.0, "co_drop", format!("나머지 {}대가 무게중심을 받치지 못해 물체가 떨어짐", rest.len()), &[Part::Hand], Some(end_drop), false));
    }
    LetGo { after: None, dropped: true, instant_n: instant, end_on_ground_share: end_share, end_held, end_drop_m: end_drop, events }
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
    fn a_level_log_splits_evenly_and_a_slope_loads_the_lower_carrier() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let l = 2.0f32;
        let log = Load::log(8.0, 2.4, 0.2, false);
        let level = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!((level.members[0].share - 0.5).abs() < 1e-3 && level.statics.determinate, "{:?}", level.members.iter().map(|m| m.share).collect::<Vec<_>>());
        let dz = l * 10f32.to_radians().tan();
        let slope = |h_c: f32| {
            let load = Load { com_up_m: h_c, ..log };
            share(&load, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, dz, Surface::Firm)], &CarryOpts::default())
        };
        let s0 = slope(0.0);
        assert!((s0.members[0].share - 0.5).abs() < 1e-3, "{}", s0.members[0].share);
        assert!((s0.statics.incline_deg - 10.0).abs() < 0.1);
        for h_c in [0.1f32, 0.3, 0.5] {
            let s = slope(h_c);
            let along = l / 10f32.to_radians().cos();
            let expect = 0.5 + h_c * 10f32.to_radians().tan() / along;
            assert!((s.members[0].share - expect).abs() < 2e-3, "h_c {h_c}: {} vs {expect}", s.members[0].share);
        }
        assert!(slope(0.5).members[0].share > slope(0.1).members[0].share);
        let s = slope(0.5);
        assert!(s.axial_even_n > 0.0 && s.axial_even_ok);
        let ev = share(&Load { com_up_m: 0.5, ..log }, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, l, 0.0, dz, Surface::Firm)], &CarryOpts { even_axial: true, ..CarryOpts::default() });
        assert!((ev.members[0].share - 0.5).abs() < 1e-3, "{}", ev.members[0].share);
    }

    #[test]
    fn two_walkers_gripping_one_point_share_its_bob_without_nan() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let rock = Load::rock(4.0, 0.4);
        let s = share(&rock, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        for m in &s.members {
            assert!(m.f_n.is_finite() && m.f_peak_n.is_finite() && m.f_peak_n >= m.f_n - 1e-3, "{m:?}");
        }
        let d = s.dynamic.as_ref().expect("two walkers in a line bob");
        assert!(d.in_phase_n.iter().chain(&d.anti_phase_n).all(|f| f.is_finite()), "{d:?}");
        assert!((d.in_phase_n[0] - d.in_phase_n[1]).abs() < 1e-3);
    }

    #[test]
    fn three_around_a_rock_are_determinate_and_four_in_a_line_share_by_stiffness() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let rock = Load::rock(12.0, 0.5);
        let r = 0.3f32;
        let around: Vec<Carrier> = (0..3).map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 3.0;
            Carrier::at(&b, r * a.cos(), r * a.sin(), 0.0, Surface::Firm)
        }).collect();
        let s = share(&rock, &around, &CarryOpts::default());
        assert!(!s.statics.collinear && s.statics.determinate);
        assert!(s.members.iter().all(|m| (m.share - 1.0 / 3.0).abs() < 1e-3), "{:?}", s.members.iter().map(|m| m.share).collect::<Vec<_>>());
        let log = Load::log(16.0, 4.0, 0.2, false);
        let line = |k: [f32; 4]| -> Vec<f32> {
            let cs: Vec<Carrier> = (0..4).map(|i| Carrier { stiffness: k[i], ..Carrier::at(&b, i as f32, 0.0, 0.0, Surface::Firm) }).collect();
            share(&log, &cs, &CarryOpts::default()).members.iter().map(|m| m.share).collect()
        };
        let eq = line([1.0; 4]);
        assert!(eq.iter().all(|x| (x - 0.25).abs() < 1e-3), "{eq:?}");
        let stiff = line([2.0, 1.0, 1.0, 2.0]);
        assert!(stiff[0] > 0.25 && stiff[1] < 0.25 && (stiff.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{stiff:?}");
        let cs: Vec<Carrier> = (0..4).map(|i| Carrier { hand_low_m: if i == 1 { 0.01 } else { 0.0 }, stiffness: 2.0e4, ..Carrier::at(&b, i as f32, 0.0, 0.0, Surface::Firm) }).collect();
        let low = share(&log, &cs, &CarryOpts::default());
        assert!(low.members[1].share < 0.25, "{:?}", low.members.iter().map(|m| m.share).collect::<Vec<_>>());
    }

    #[test]
    fn a_legged_pair_carries_what_neither_can_alone() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let cap = member_capacity_kg(&b, Surface::Firm, 0.0, carry_hand_z(&b));
        assert!(cap < 6.0 && cap > 1.0, "mid legs carry {cap} kg");
        let log = Load::log(1.4 * cap, 2.0, 0.15, false);
        let alone = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!(!alone.ok, "one robot: {:?}", alone.members[0].why_ko);
        let pair = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert!(pair.ok, "{:?} {:?}", pair.members.iter().map(|m| m.why_ko.clone()).collect::<Vec<_>>(), pair.events);
        assert!((pair.team_limit_kg - 0.85 * 2.0 * cap).abs() < 1e-3);
        let cons = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts { rule: TeamRule::Conservative, ..CarryOpts::default() });
        assert!(cons.limit_kg < pair.limit_kg);
        let d = pair.dynamic.as_ref().unwrap();
        assert!(d.amp_m[0] > 0.0 && d.in_phase_n[0] > d.anti_phase_n[0], "{d:?}");
        let w = body(Tier::Mid, BaseKind::Wheel);
        let mixed = share(&log, &[Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&w, 1.6, 0.0, 0.0, Surface::Firm)], &CarryOpts::default());
        assert_eq!(mixed.dynamic.as_ref().unwrap().amp_m[1], 0.0);
    }

    #[test]
    fn letting_go_drops_the_free_end_onto_the_ground() {
        let b = body(Tier::Mid, BaseKind::HumanFoot);
        let log = Load::log(6.0, 2.0, 0.15, false);
        let cs = [Carrier::at(&b, 0.0, 0.0, 0.0, Surface::Firm), Carrier::at(&b, 1.6, 0.0, 0.0, Surface::Firm)];
        let g = let_go(&log, &cs, 1, &CarryOpts::default());
        assert!(g.dropped && g.end_drop_m > 0.5);
        let i_g = log.pitch_inertia();
        let d = 0.8f32;
        assert!((g.instant_n[0] - log.weight_n() * i_g / (i_g + log.mass_kg * d * d)).abs() < 0.5, "{:?}", g.instant_n);
        let s = g.end_on_ground_share.unwrap();
        assert!(s > 0.4 && s < 0.6, "{s}");
        assert!(g.events.iter().any(|e| e.kind == "co_drop"));
        let rock = Load::rock(9.0, 0.5);
        let around: Vec<Carrier> = (0..3).map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 3.0;
            Carrier::at(&b, 0.3 * a.cos(), 0.3 * a.sin(), 0.0, Surface::Firm)
        }).collect();
        assert!(let_go(&rock, &around, 0, &CarryOpts::default()).dropped);
    }
}
