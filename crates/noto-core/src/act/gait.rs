use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;

use crate::balance::{Body, G, KNEE_LEVER_WALK_M, REACTION_DELAY_S};
use crate::contact::{self, SoleModel, TOE_HINGE_FL};
use crate::kin::{self, KinLoads, KinPose, ANKLE_H_M, SHANK_M, THIGH_M};
use crate::parts::*;
use crate::surface::{self, SurfaceProps};

use super::{ActEvent, ActKind, ActRollout, ActSample, Draw, Outcome, Part, SAMPLE_S};

pub const KEEP_MARGIN: f32 = 0.10;
pub const SWING_LIFT_M: f32 = 0.05;
pub const SWING_OFF_M: f32 = 0.002;
pub const TOUCHDOWN_MPS: f32 = 0.15;
pub const SOLE_STIFFNESS: f32 = 2.0e5;
pub const GROUND_MU_COV: f32 = 0.10;
pub const FALL_SLIDE_LEG: f32 = 0.10 / 0.88;
pub const WALK_RUN_FROUDE: f32 = 0.5;
const DUTY_FACTOR: [(f32, f32); 7] = [(0.0072, 0.77), (0.029, 0.71), (0.065, 0.68), (0.10, 0.66), (0.20, 0.64), (0.30, 0.63), (0.40, 0.62)];
const PERSON_LEG_M: f32 = 0.88;
pub const COM_EXCURSION: [(f32, f32); 2] = [(1.0 / 3.6, 0.015), (6.0 / 3.6, 0.034)];
pub const COP_PATH: [(f32, f32); 5] = [(0.0442, -0.0745), (0.2484, 0.1619), (0.5009, 0.4034), (0.7317, 0.5921), (0.9351, 0.7055)];
pub const HEEL_CENTRE_FL: f32 = 0.15;
pub const ANKLE_FL: f32 = 0.25;
pub const NOMINAL_FOOT_M: f32 = 0.20;
const PATH_STEPS: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GaitPlan {
    pub speed_cmd: f32,
    pub speed: f32,
    pub step_m: f32,
    pub step_s: f32,
    pub ds_frac: f32,
    pub com_z: f32,
    pub rcof: f32,
    pub mu_est: f32,
    pub shortened: bool,
    pub com_amp_m: f32,
    pub foot_m: f32,
    pub x0: f32,
    pub v0: f32,
}

pub fn relative_step(froude: f32) -> f32 {
    0.5 * 2.47 * froude.clamp(1e-4, WALK_RUN_FROUDE).powf(0.29)
}

pub fn ds_fraction(froude: f32) -> f32 {
    let t = &DUTY_FACTOR;
    let df = if froude <= t[0].0 {
        t[0].1
    } else if froude >= t[t.len() - 1].0 {
        t[t.len() - 1].1
    } else {
        let i = t.iter().position(|p| p.0 >= froude).unwrap_or(t.len() - 1).max(1);
        let ((x0, y0), (x1, y1)) = (t[i - 1], t[i]);
        y0 + (y1 - y0) * (froude - x0) / (x1 - x0)
    };
    2.0 * df - 1.0
}

pub fn impact_mass_kg(body: &Body) -> f32 {
    body.leg.foot_mass_kg + body.leg.mass_kg * (1.0 - kin::thigh_share(body))
}

pub fn walking_com_z(body: &Body) -> f32 {
    let bend = (KNEE_LEVER_WALK_M / SHANK_M).clamp(0.0, 0.9).asin();
    body.com_height_m() - (THIGH_M + SHANK_M) * (1.0 - bend.cos())
}

pub fn com_excursion_m(speed: f32, leg_m: f32) -> f32 {
    let hat = |v: f32, l: f32| v / (G * l).sqrt();
    let (a, b) = (COM_EXCURSION[0], COM_EXCURSION[1]);
    let u = ((hat(speed, leg_m) - hat(a.0, PERSON_LEG_M)) / (hat(b.0, PERSON_LEG_M) - hat(a.0, PERSON_LEG_M))).clamp(0.0, 1.0);
    (a.1 + (b.1 - a.1) * u) / PERSON_LEG_M * leg_m
}

pub fn com_rise_room_m() -> f32 {
    let bend = (KNEE_LEVER_WALK_M / SHANK_M).clamp(0.0, 0.9).asin();
    (THIGH_M + SHANK_M) * (1.0 - bend.cos())
}

fn interp(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    for w in table.windows(2) {
        if x <= w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) * (x - w[0].0) / (w[1].0 - w[0].0);
        }
    }
    table[table.len() - 1].1
}

#[derive(Clone, Copy, Debug)]
struct StepModel {
    lam: f32,
    step_s: f32,
    t_ds: f32,
    z_c: f32,
    amp: f32,
    foot_m: f32,
}

impl StepModel {
    fn of(g: &GaitPlan) -> StepModel {
        StepModel { lam: g.step_m, step_s: g.step_s, t_ds: g.ds_frac * g.step_s, z_c: g.com_z, amp: g.com_amp_m, foot_m: g.foot_m }
    }

    fn z(&self, t: f32) -> (f32, f32) {
        let w = std::f32::consts::TAU / self.step_s.max(1e-3);
        let c = (w * (t - 0.5 * self.t_ds)).cos();
        (self.z_c - 0.5 * self.amp * c, 0.5 * self.amp * w * w * c)
    }

    fn sole(&self, f: f32) -> f32 {
        let span = COP_PATH[COP_PATH.len() - 1].1 - COP_PATH[0].1;
        let roll = self.foot_m.min(self.lam / span);
        roll * (interp(&COP_PATH, f) + HEEL_CENTRE_FL - ANKLE_FL)
    }

    fn cop(&self, t: f32) -> f32 {
        let stance = self.step_s + self.t_ds;
        if t < self.t_ds {
            let u = t / self.t_ds.max(1e-6);
            let trail = -self.lam + self.sole((self.step_s + t) / stance);
            let lead = self.sole(t / stance);
            trail + (lead - trail) * u
        } else {
            self.sole(t / stance)
        }
    }

    fn accel(&self, t: f32, x: f32) -> f32 {
        let (z, zdd) = self.z(t);
        (G + zdd) * (x - self.cop(t)) / z.max(0.05)
    }

    fn run(&self, x0: f32, v0: f32, mut each: impl FnMut(f32, f32, f32)) -> (f32, f32) {
        let h = self.step_s / PATH_STEPS as f32;
        let (mut x, mut v) = (x0, v0);
        each(0.0, x, v);
        for i in 0..PATH_STEPS {
            let t = i as f32 * h;
            let (k1x, k1v) = (v, self.accel(t, x));
            let (k2x, k2v) = (v + 0.5 * h * k1v, self.accel(t + 0.5 * h, x + 0.5 * h * k1x));
            let (k3x, k3v) = (v + 0.5 * h * k2v, self.accel(t + 0.5 * h, x + 0.5 * h * k2x));
            let (k4x, k4v) = (v + h * k3v, self.accel(t + h, x + h * k3x));
            x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
            each(t + h, x, v);
        }
        (x, v)
    }

    fn periodic(&self) -> (f32, f32) {
        let base = self.run(0.0, 0.0, |_, _, _| {});
        let ex = self.run(1.0, 0.0, |_, _, _| {});
        let ev = self.run(0.0, 1.0, |_, _, _| {});
        let (a, b, c, d) = (ex.0 - base.0, ev.0 - base.0, ex.1 - base.1, ev.1 - base.1);
        let (r1, r2) = (self.lam - base.0, -base.1);
        let det = (a - 1.0) * (d - 1.0) - b * c;
        if det.abs() < 1e-9 {
            return (-0.5 * self.lam, self.lam / self.step_s.max(1e-3));
        }
        ((r1 * (d - 1.0) - b * r2) / det, ((a - 1.0) * r2 - c * r1) / det)
    }

    fn lean(&self, x0: f32, v0: f32) -> f32 {
        let mut worst = 0.0f32;
        self.run(x0, v0, |t, x, _| worst = worst.max((x - self.cop(t)).abs() / self.z(t).0));
        worst
    }
}

fn step_lean(z_c: f32, foot_m: f32, step_m: f32, step_s: f32, ds_frac: f32) -> (f32, f32, f32, f32) {
    let speed = step_m / step_s.max(1e-3);
    let amp = com_excursion_m(speed, HIP_M).min(2.0 * com_rise_room_m());
    let m = StepModel { lam: step_m, step_s, t_ds: ds_frac * step_s, z_c, amp, foot_m };
    let (x0, v0) = m.periodic();
    (m.lean(x0, v0), amp, x0, v0)
}

static PLANS: Mutex<Option<HashMap<[u32; 5], GaitPlan>>> = Mutex::new(None);

pub fn plan(body: &Body, speed: f32, mu_est: f32, slope_deg: f32) -> GaitPlan {
    plan_z(walking_com_z(body), body.leg.foot_len_m, speed, mu_est, slope_deg)
}

pub fn plan_z(z_c: f32, foot_m: f32, speed: f32, mu_est: f32, slope_deg: f32) -> GaitPlan {
    let key = [z_c.to_bits(), foot_m.to_bits(), speed.to_bits(), mu_est.to_bits(), slope_deg.to_bits()];
    if let Some(g) = PLANS.lock().ok().and_then(|p| p.as_ref().and_then(|m| m.get(&key).copied())) {
        return g;
    }
    let g = plan_uncached(z_c, foot_m, speed, mu_est, slope_deg);
    if let Ok(mut p) = PLANS.lock() {
        let m = p.get_or_insert_with(HashMap::new);
        if m.len() > 4096 {
            m.clear();
        }
        m.insert(key, g);
    }
    g
}

fn plan_uncached(z_c: f32, foot_m: f32, speed: f32, mu_est: f32, slope_deg: f32) -> GaitPlan {
    let l = HIP_M;
    let fr = speed * speed / (G * l);
    let ds_frac = ds_fraction(fr);
    let full = relative_step(fr) * l;
    let step_s = full / speed.max(0.01);
    let tan = slope_deg.to_radians().tan().abs();
    let room = mu_est - KEEP_MARGIN - tan;
    let mut step = full;
    let mut at = step_lean(z_c, foot_m, step, step_s, ds_frac);
    let mut shortened = false;
    if at.0 > room {
        shortened = true;
        let (mut lo, mut hi) = (0.05 * l, full);
        let low = step_lean(z_c, foot_m, lo, step_s, ds_frac);
        if low.0 > room {
            step = lo;
            at = low;
        } else {
            for _ in 0..14 {
                let mid = 0.5 * (lo + hi);
                if step_lean(z_c, foot_m, mid, step_s, ds_frac).0 > room {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            step = lo;
            at = step_lean(z_c, foot_m, step, step_s, ds_frac);
        }
    }
    let (lean, amp, x0, v0) = at;
    GaitPlan { speed_cmd: speed, speed: step / step_s, step_m: step, step_s, ds_frac, com_z: z_c, rcof: lean + tan, mu_est, shortened, com_amp_m: amp, foot_m, x0, v0 }
}

pub fn com_at(g: &GaitPlan, t: f32) -> ComState {
    let m = StepModel::of(g);
    let t = t.clamp(0.0, g.step_s);
    let mut best = (0.0f32, g.x0, g.v0);
    m.run(g.x0, g.v0, |tt, x, v| {
        if tt <= t + 1e-6 {
            best = (tt, x, v);
        }
    });
    let dt = t - best.0;
    let a = m.accel(best.0, best.1);
    let x = best.1 + best.2 * dt + 0.5 * a * dt * dt;
    let (z, zdd) = m.z(t);
    ComState { x, v: best.2 + a * dt, z, ax: m.accel(t, x), az: zdd, cop: m.cop(t) }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ComState {
    pub x: f32,
    pub v: f32,
    pub z: f32,
    pub ax: f32,
    pub az: f32,
    pub cop: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StepStat {
    pub index: u32,
    pub stance: usize,
    pub mu: f32,
    pub rcof: f32,
    pub slip_p: f32,
    pub slipped: bool,
    pub fell: bool,
    pub slide_m: f32,
    pub impact_n: f32,
    pub grf_peak_bw: f32,
    pub grf_min_bw: f32,
    pub cone_use_max: f32,
    pub toe_push_n: f32,
}

#[derive(Clone, Debug, Default)]
struct SoleLog {
    toe: [f32; 2],
    heel_up: [bool; 2],
    strike_t: [Option<f32>; 2],
    struck: [bool; 2],
    peak_kpa: [f32; 4],
    toe_peak_n: f32,
    heel_peak_n: f32,
    toe_deg_max: f32,
    stance_samples: u32,
    toe_samples: u32,
    cop_reach_fl: f32,
    heel_off_frac: Vec<f32>,
    stick_min: f32,
    toe_off_ds: Vec<f32>,
}

impl SoleLog {
    fn new() -> SoleLog {
        SoleLog { stick_min: 1.0, ..SoleLog::default() }
    }
}

pub fn walk(body: &Body, ground: &SurfaceProps, mu_belief: Option<f32>, slope_deg: f32, cargo_kg: f32, speed: f32, steps: u32, seed: u64) -> (ActRollout, Vec<StepStat>) {
    let mut d = Draw::new(seed);
    let mu_mean = surface::available_mu(body.base, ground);
    let g = plan(body, speed, mu_belief.unwrap_or(mu_mean), slope_deg);
    let m = body.mass_kg() + cargo_kg;
    let t_ds = g.ds_frac * g.step_s;
    let t_ss = g.step_s - t_ds;
    let lam = g.step_m;
    let n_ds = (t_ds / SAMPLE_S).ceil().max(1.0) as usize;
    let n_ss = (t_ss / SAMPLE_S).ceil().max(1.0) as usize;
    let path_ds: Vec<ComState> = (0..n_ds).map(|i| com_at(&g, i as f32 / n_ds as f32 * t_ds)).collect();
    let path_ss: Vec<ComState> = (0..n_ss).map(|i| com_at(&g, t_ds + i as f32 / n_ss as f32 * t_ss)).collect();
    let half_w = 0.5 * STANCE_WIDTH_M;
    let m_imp = impact_mass_kg(body);
    let k_ground = super::material::ground_stiffness(crate::surface::Surface::from_id(ground.id).unwrap_or_default());
    let k_eff = 1.0 / (1.0 / SOLE_STIFFNESS + 1.0 / k_ground);
    let (f_imp, dt_imp) = super::material::impact(TOUCHDOWN_MPS, m_imp, k_eff);
    let sole = SoleModel::of(body);
    let fl = body.leg.foot_len_m;
    let hinge = (TOE_HINGE_FL - contact::ANKLE_FL) * fl;
    let tan_slope = slope_deg.to_radians().tan();
    let mut log = SoleLog::new();
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let mut stats = Vec::new();
    let mut outcome = Outcome::ok();
    let mut t0 = 0.0f32;
    let mut x_base = 0.0f32;
    let mut slips = 0u32;
    let side_ko = |s: usize| if s == 0 { "오른" } else { "왼" };
    let sm = StepModel::of(&g);
    let stance_t = g.step_s + t_ds;
    let foot_cops = |t: f32, stance: usize| -> [Option<f32>; 2] {
        let mut c = [None; 2];
        c[stance] = Some(sm.sole(t / stance_t));
        if t < t_ds {
            c[1 - stance] = Some(-lam + sm.sole((g.step_s + t) / stance_t));
        }
        c
    };
    let feel = |log: &mut SoleLog, events: &mut Vec<ActEvent>, r: &kin::KinReport, pose: &KinPose, cops: [Option<f32>; 2], t: f32, dt: f32, mu: f32, single: bool| -> (f32, f32) {
        let mut use_max = 0.0f32;
        let mut toe_n = 0.0f32;
        for s in 0..2 {
            let name = if s == 0 { "foot_r" } else { "foot_l" };
            let Some(c) = r.contact(name) else { continue };
            if pose.feet[s][1] > 1e-4 || c.normal_n <= 1e-6 {
                log.toe[s] = 0.0;
                log.heel_up[s] = false;
                continue;
            }
            if log.strike_t[s].is_none() && log.struck[s] {
                log.strike_t[s] = Some(t);
            }
            let ankle = pose.feet[s][0];
            let cop = cops[s].unwrap_or(c.cop[0]);
            let sc = contact::sole_contact(&sole, ankle, c.normal_n, [c.shear_n + c.normal_n * tan_slope, c.shear_y_n], c.torsion_nm, cop, pose.toe_deg[s], mu);
            use_max = use_max.max(sc.cone_use);
            log.stick_min = log.stick_min.min(sc.stick_share);
            log.stance_samples += 1;
            if sc.patches[3].normal_n > TOE_OFF_LOAD_BW * m * G {
                log.toe_samples += 1;
            }
            for i in 0..4 {
                log.peak_kpa[i] = log.peak_kpa[i].max(sc.patches[i].peak_kpa);
            }
            log.cop_reach_fl = log.cop_reach_fl.max((sc.cop_x - (ankle - contact::ANKLE_FL * fl)) / fl);
            if !single {
                toe_n = toe_n.max(sc.patches[3].normal_n);
            }
            log.toe_peak_n = log.toe_peak_n.max(sc.patches[3].normal_n);
            log.heel_peak_n = log.heel_peak_n.max(sc.patches[0].normal_n);
            let moment = toe_moment(c.normal_n, cop - ankle, hinge);
            let before = log.toe[s];
            log.toe[s] = sole.toe_step_deg(before, moment, dt);
            log.toe_deg_max = log.toe_deg_max.max(log.toe[s]);
            if before <= 1e-3 && log.toe[s] > 1e-3 && !log.heel_up[s] {
                log.heel_up[s] = true;
                if let Some(t_s) = log.strike_t[s] {
                    log.heel_off_frac.push((t - t_s) / (g.step_s + t_ds).max(1e-3));
                }
                events.push(ActEvent::new(t, "heel_off", format!("{}발 뒤꿈치 들림 · 압력 중심이 발가락 관절 앞 {:.0} mm", side_ko(s), (cop - ankle - hinge) * 1000.0), &[Part::Foot, Part::FootJoints], Some(moment), true));
            }
        }
        (use_max, toe_n)
    };
    'steps: for k in 0..steps {
        let stance = (k % 2) as usize;
        let swing = 1 - stance;
        let ys = if stance == 0 { -half_w } else { half_w };
        let mu = d.lognormal(mu_mean, GROUND_MU_COV);
        let margin = mu - g.rcof;
        let p = surface::slip_probability(margin);
        let slipped = d.uniform() < p;
        let mut peak = 0.0f32;
        let mut low = f32::INFINITY;
        let mut use_max = 0.0f32;
        let mut push = (0.0f32, 0.0f32, 0.0f32);
        log.strike_t[stance] = None;
        log.struck[stance] = true;
        log.heel_up[stance] = false;
        let mut toe_off_t: Option<f32> = None;
        for (i, c) in path_ds.iter().enumerate() {
            let u = i as f32 / n_ds as f32;
            let t = u * t_ds;
            let fz_extra = if t < dt_imp { f_imp * (std::f32::consts::PI * t / dt_imp).sin() } else { 0.0 };
            let bw = (m * (G + c.az) + fz_extra) / (m * G);
            peak = peak.max(bw);
            low = low.min(bw);
            let mut pose = KinPose::stand(cargo_kg);
            pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
            pose.feet[stance] = [0.0, 0.0];
            pose.feet[swing] = [-lam, 0.0];
            pose.toe_deg = log.toe;
            pose.pelvis_y = ys * (2.0 * u - 1.0);
            pose.shoulder_deg = [8.0, -8.0];
            let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
            let r = kin::solve_fast(body, &pose, &loads);
            let (cu, toe_n) = feel(&mut log, &mut events, &r, &pose, foot_cops(t, stance), t0 + t, t_ds / n_ds as f32, mu, false);
            use_max = use_max.max(cu);
            if toe_n > push.0 {
                push = (toe_n, t0 + t, log.toe[swing]);
            }
            let trail_n = r.contact(if swing == 0 { "foot_r" } else { "foot_l" }).map_or(0.0, |c| c.normal_n);
            if toe_off_t.is_none() && i > 0 && trail_n <= TOE_OFF_LOAD_BW * m * G {
                toe_off_t = Some(t0 + t);
            }
            samples.push(ActSample { t: t0 + t, phase: "double_support", pose: shift(pose, x_base), loads, margin: r.margin, zmp: [r.zmp[0] + x_base, r.zmp[1]], com: [r.com[0] + x_base, r.com[1], r.com[2]], peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: None });
        }
        events.push(ActEvent::new(t0, "heel_strike", format!("{}발 착지 · 충격 {:.0} N ({:.0} ms) · 필요 마찰 {:.2} / 지면 {:.2}", side_ko(stance), f_imp, dt_imp * 1000.0, g.rcof, mu), &[Part::Foot, Part::FootJoints], Some(f_imp), true));
        if push.0 > 0.0 {
            events.push(ActEvent::new(push.1, "toe_push", format!("{}발 발가락으로 밀기 · 발가락 {:.0} N ({:.0}% 체중) · 발가락 관절 {:.0}°", side_ko(swing), push.0, 100.0 * push.0 / (m * G), push.2), &[Part::Foot, Part::FootJoints], Some(push.0), true));
        }
        if slipped {
            slips += 1;
            let (fell, slide) = slip_outcome(&g, mu);
            stats.push(StepStat { index: k, stance, mu, rcof: g.rcof, slip_p: p, slipped, fell, slide_m: slide, impact_n: f_imp, grf_peak_bw: peak, grf_min_bw: low, cone_use_max: use_max, toe_push_n: push.0 });
            if fell {
                events.push(ActEvent::new(t0 + t_ds * 0.5, "fall", format!("미끄러져 넘어짐: 마찰 {mu:.2} < 필요 {:.2}, {:.0} mm 미끄러짐", g.rcof, slide * 1000.0), &[Part::Foot, Part::FootJoints, Part::Waist], Some(slide), false));
                outcome = Outcome::fail("fall", format!("{}번째 걸음에서 미끄러져 넘어짐 (지면 마찰 {mu:.2} < 필요 {:.2}, 미끄럼 {:.0} mm)", k + 1, g.rcof, slide * 1000.0));
                t0 += t_ds;
                break 'steps;
            }
            events.push(ActEvent::new(t0 + t_ds * 0.5, "foot_slip", format!("발 미끄러짐 {:.0} mm · 마찰 {mu:.2} < 필요 {:.2} → 회복", slide * 1000.0, g.rcof), &[Part::Foot, Part::FootJoints], Some(slide), true));
        }
        for (i, c) in path_ss.iter().enumerate() {
            let u = i as f32 / n_ss as f32;
            let t = u * t_ss;
            let mut pose = KinPose::stand(cargo_kg);
            pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
            pose.feet[stance] = [0.0, 0.0];
            let sw = super::min_jerk(u);
            pose.feet[swing] = [-lam + 2.0 * lam * sw, (SWING_LIFT_M * (std::f32::consts::PI * u).sin()).max(SWING_OFF_M)];
            pose.toe_deg = [0.0; 2];
            pose.toe_deg[stance] = log.toe[stance];
            pose.pelvis_y = ys;
            pose.shoulder_deg = if stance == 0 { [-8.0, 8.0] } else { [8.0, -8.0] };
            let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
            let r = kin::solve_fast(body, &pose, &loads);
            let (cu, _) = feel(&mut log, &mut events, &r, &pose, foot_cops(t_ds + t, stance), t0 + t_ds + t, t_ss / n_ss as f32, mu, true);
            use_max = use_max.max(cu);
            let bw = (G + c.az) / G;
            peak = peak.max(bw);
            low = low.min(bw);
            samples.push(ActSample { t: t0 + t_ds + t, phase: "single_support", pose: shift(pose, x_base), loads, margin: r.margin, zmp: [r.zmp[0] + x_base, r.zmp[1]], com: [r.com[0] + x_base, r.com[1], r.com[2]], peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: None });
        }
        let lift_t = toe_off_t.unwrap_or(t0 + t_ds);
        log.toe_off_ds.push((lift_t - t0) / t_ds.max(1e-6));
        let why = if toe_off_t.is_some() { format!("하중을 넘김 (체중의 {:.0}% 아래)", TOE_OFF_LOAD_BW * 100.0) } else { "양발 지지 끝".to_string() };
        events.push(ActEvent::new(lift_t, "toe_off", format!("{}발 들림 · {why}", side_ko(swing)), &[Part::Foot], None, true));
        if !slipped {
            stats.push(StepStat { index: k, stance, mu, rcof: g.rcof, slip_p: p, slipped, fell: false, slide_m: 0.0, impact_n: f_imp, grf_peak_bw: peak, grf_min_bw: low, cone_use_max: use_max, toe_push_n: push.0 });
        }
        t0 += g.step_s;
        x_base += lam;
    }
    events.sort_by(|a, b| a.t.total_cmp(&b.t));
    let walked = stats.len() as f32;
    let mean_of = |xs: &[f32]| if xs.is_empty() { 0.0 } else { xs.iter().sum::<f32>() / xs.len() as f32 };
    let mut metrics = vec![
        ("speed_mps", g.speed),
        ("speed_cmd_mps", g.speed_cmd),
        ("step_m", g.step_m),
        ("step_s", g.step_s),
        ("ds_frac", g.ds_frac),
        ("com_z_m", g.com_z),
        ("rcof", g.rcof),
        ("mu_est", g.mu_est),
        ("impact_n", f_imp),
        ("impact_bw", f_imp / (m * G)),
        ("impact_ms", dt_imp * 1000.0),
        ("impact_mass_kg", m_imp),
        ("touchdown_mps", TOUCHDOWN_MPS),
        ("loading_rate_bw_s", f_imp / (m * G) / (0.5 * dt_imp).max(1e-4)),
        ("slip_rate", slips as f32 / walked.max(1.0)),
        ("slip_p_mean", stats.iter().map(|s| s.slip_p).sum::<f32>() / walked.max(1.0)),
        ("relative_step", g.step_m / HIP_M),
        ("froude", g.speed * g.speed / (G * HIP_M)),
        ("com_amp_m", g.com_amp_m),
        ("duration_s", t0),
    ];
    if log.stance_samples > 0 {
        metrics.extend([
            ("heel_peak_kpa", log.peak_kpa[0]),
            ("midfoot_peak_kpa", log.peak_kpa[1]),
            ("meta_peak_kpa", log.peak_kpa[2]),
            ("toe_peak_kpa", log.peak_kpa[3]),
            ("toe_peak_bw", log.toe_peak_n / (m * G)),
            ("heel_peak_bw", log.heel_peak_n / (m * G)),
            ("toe_contact_share", log.toe_samples as f32 / log.stance_samples as f32),
            ("toe_deg_max", log.toe_deg_max),
            ("cop_reach_fl", log.cop_reach_fl),
            ("sole_stick_min", log.stick_min),
            ("cone_use_max", stats.iter().map(|s| s.cone_use_max).fold(0.0, f32::max)),
            ("toe_push_n", mean_of(&stats.iter().map(|s| s.toe_push_n).filter(|x| *x > 0.0).collect::<Vec<_>>())),
        ]);
        if !log.heel_off_frac.is_empty() {
            metrics.push(("heel_off_frac", mean_of(&log.heel_off_frac)));
        }
        if !log.toe_off_ds.is_empty() {
            metrics.push(("toe_off_ds_frac", mean_of(&log.toe_off_ds)));
        }
    }
    let notes = if g.shortened { vec![format!("마찰 {:.2}에 맞춰 보폭을 {:.2} m로 줄임 (속도 {:.2} → {:.2} m/s)", g.mu_est, g.step_m, g.speed_cmd, g.speed)] } else { vec![] };
    (ActRollout { kind: ActKind::Walk, samples, events, outcome, duration_s: t0, metrics, notes }, stats)
}

pub const TOE_OFF_LOAD_BW: f32 = 0.01;

fn toe_moment(normal_n: f32, cop_x: f32, hinge_x: f32) -> f32 {
    normal_n * (cop_x - hinge_x)
}

pub fn toe_angles(body: &Body, g: &GaitPlan, cargo_kg: f32, t: f32) -> [f32; 2] {
    let sole = SoleModel::of(body);
    let sm = StepModel::of(g);
    let w = (body.mass_kg() + cargo_kg) * G;
    let hinge = (TOE_HINGE_FL - contact::ANKLE_FL) * body.leg.foot_len_m;
    let t_ds = g.ds_frac * g.step_s;
    let stance_t = g.step_s + t_ds;
    let t = t.clamp(0.0, g.step_s);
    let normal = |tt: f32| w * (G + sm.z(tt).1) / G;
    let n = 24usize;
    let ss = |until: f32| {
        let mut th = 0.0f32;
        let dt = (until - t_ds).max(0.0) / n as f32;
        for i in 0..n {
            let tt = t_ds + (i as f32 + 0.5) * dt;
            th = sole.toe_step_deg(th, toe_moment(normal(tt), sm.sole(tt / stance_t), hinge), dt);
        }
        th
    };
    if t >= t_ds {
        return [ss(t), 0.0];
    }
    let mut th = ss(g.step_s);
    let dt = t / n as f32;
    for i in 0..n {
        let tt = (i as f32 + 0.5) * dt;
        let share = 1.0 - (tt / t_ds.max(1e-4)).clamp(0.0, 1.0);
        th = sole.toe_step_deg(th, toe_moment(normal(tt) * share, sm.sole((g.step_s + tt) / stance_t), hinge), dt);
    }
    [0.0, th]
}

pub fn pose_at(body: &Body, g: &GaitPlan, stance: usize, u: f32, cargo_kg: f32) -> (KinPose, KinLoads, &'static str) {
    let t_ds = g.ds_frac * g.step_s;
    let t_ss = g.step_s - t_ds;
    let lam = g.step_m;
    let swing = 1 - stance;
    let ys = if stance == 0 { -0.5 * STANCE_WIDTH_M } else { 0.5 * STANCE_WIDTH_M };
    let t = u.clamp(0.0, 1.0) * g.step_s;
    let c = com_at(g, t);
    let toes = toe_angles(body, g, cargo_kg, t);
    let mut pose = KinPose::stand(cargo_kg);
    pose.feet[stance] = [0.0, 0.0];
    pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
    pose.toe_deg[stance] = toes[0];
    let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
    if t < t_ds {
        let v = t / t_ds.max(1e-4);
        pose.feet[swing] = [-lam, 0.0];
        pose.toe_deg[swing] = toes[1];
        pose.pelvis_y = ys * (2.0 * v - 1.0);
        (pose, loads, "double_support")
    } else {
        let v = (t - t_ds) / t_ss.max(1e-4);
        pose.feet[swing] = [-lam + 2.0 * lam * super::min_jerk(v), (SWING_LIFT_M * (std::f32::consts::PI * v).sin()).max(SWING_OFF_M)];
        pose.pelvis_y = ys;
        pose.shoulder_deg = if stance == 0 { [-8.0, 8.0] } else { [8.0, -8.0] };
        (pose, loads, "single_support")
    }
}

pub fn slip_outcome(g: &GaitPlan, mu: f32) -> (bool, f32) {
    let mu_k = 0.85 * mu;
    let a_slide = (G * (g.rcof - mu_k)).max(0.05 * G);
    let slide = 0.5 * a_slide * REACTION_DELAY_S * REACTION_DELAY_S + g.speed * REACTION_DELAY_S * (g.rcof - mu_k).max(0.0) / g.rcof.max(1e-3);
    let omega = (G / g.com_z.max(0.1)).sqrt();
    let capture = 0.5 * g.step_m + g.speed / omega;
    let reach = 2.0 * (THIGH_M + SHANK_M) * 0.8f32.sin();
    (slide > FALL_SLIDE_LEG * HIP_M || capture + slide > reach, slide)
}

fn shift(mut p: KinPose, dx: f32) -> KinPose {
    p.hip[0] += dx;
    p.feet[0][0] += dx;
    p.feet[1][0] += dx;
    p
}

pub fn leg_angles(hip: [f32; 2], ankle: [f32; 2]) -> Option<f32> {
    kin::knee_ik(hip, [ankle[0], ankle[1].max(ANKLE_H_M)]).map(|k| k.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::surface::Surface;

    fn legs(tier: Tier) -> Body {
        Body::new(&BodyPlan::uniform(tier), BaseKind::HumanFoot, FingerCount::Five)
    }

    #[test]
    fn the_step_follows_the_speed_and_shortens_on_slippery_ground() {
        let b = legs(Tier::Mid);
        let firm = plan(&b, 0.9, Surface::Firm.props().mu_foot, 0.0);
        assert!(!firm.shortened && firm.step_m > 0.2 && firm.step_m < 0.5, "{firm:?}");
        assert!(firm.rcof > 0.0 && firm.rcof < firm.step_m / (2.0 * firm.com_z), "{firm:?}");
        let wet = plan(&b, 0.9, 0.35, 8.0);
        assert!(wet.shortened && wet.step_m < firm.step_m && wet.speed < firm.speed, "{wet:?}");
        assert!(wet.rcof <= 0.35 - KEEP_MARGIN + 1e-4);
    }

    #[test]
    fn walking_on_firm_ground_never_slips_and_wet_ground_sometimes_does() {
        let b = legs(Tier::Mid);
        let (r, stats) = walk(&b, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 40, 1);
        assert!(r.outcome.ok && stats.iter().all(|s| !s.slipped), "{:?}", r.outcome);
        assert_eq!(stats.len(), 40);
        assert!(r.events.iter().filter(|e| e.kind == "heel_strike").count() == 40);
        assert!(r.metric("impact_bw").unwrap() > 0.0);
        let ice = SurfaceProps { mu_foot: 0.12, ..*Surface::Firm.props() };
        let firm_mu = Surface::Firm.props().mu_foot;
        let slipped: u32 = (0..20).map(|s| walk(&b, &ice, Some(firm_mu), 0.0, 0.0, 0.9, 20, s).1.iter().filter(|x| x.slipped).count() as u32).sum();
        assert!(slipped > 0, "a floor slicker than the gait shears slips");
        let (r, st) = walk(&b, &ice, None, 0.0, 0.0, 0.9, 20, 3);
        assert!(r.metric("step_m").unwrap() < 0.2 && st.iter().filter(|x| x.slipped).count() <= 1, "{:?}", r.notes);
    }

    #[test]
    fn the_com_bobs_and_the_stance_force_has_two_humps_and_a_valley() {
        let b = legs(Tier::Mid);
        let g = plan(&b, 0.9, Surface::Firm.props().mu_foot, 0.0);
        assert!(g.com_amp_m > 0.008 && g.com_amp_m < 2.0 * com_rise_room_m(), "{g:?}");
        let end = com_at(&g, g.step_s);
        assert!((end.x - g.step_m - g.x0).abs() < 0.01 && (end.v - g.v0).abs() < 0.02, "{end:?} vs {g:?}");
        assert!(g.v0 > 0.0 && (0..20).all(|i| com_at(&g, i as f32 * g.step_s / 20.0).v > 0.0), "always moving on");
        let t_ds = g.ds_frac * g.step_s;
        let (ds, ss) = (com_at(&g, 0.5 * t_ds), com_at(&g, t_ds + 0.5 * (g.step_s - t_ds)));
        assert!(ds.z < ss.z && ds.az > 0.0 && ss.az < 0.0);
        assert!(com_at(&g, t_ds).cop < com_at(&g, g.step_s - 1e-3).cop);
        let (r, _) = walk(&b, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 6, 1);
        let fz: Vec<(f32, f32)> = r.samples.iter().filter(|s| s.pose.feet[0][1] <= 1e-4).map(|s| (s.t, kin::solve_fast(&b, &s.pose, &s.loads).contact("foot_r").map_or(0.0, |c| c.normal_n))).collect();
        let w = b.mass_kg() * G;
        let (hi, lo) = fz.iter().fold((0.0f32, f32::INFINITY), |(h, l), p| (h.max(p.1 / w), if p.1 > 0.5 * w { l.min(p.1 / w) } else { l }));
        assert!(hi > 1.05 && lo < 0.95, "humps {hi} valley {lo}");
    }

    #[test]
    fn the_heel_comes_up_past_the_toe_hinge_and_the_trailing_toes_push() {
        let b = legs(Tier::Mid);
        let (r, _) = walk(&b, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 8, 1);
        assert!(r.events.iter().any(|e| e.kind == "heel_off"));
        assert!(r.events.iter().filter(|e| e.kind == "toe_push").count() >= 6);
        assert!(r.events.windows(2).all(|w| w[1].t >= w[0].t));
        let f = r.metric("heel_off_frac").unwrap();
        assert!(f > 0.3 && f < 0.95, "heel off late in the stance: {f}");
        let toe = r.metric("toe_deg_max").unwrap();
        assert!(toe > 1.0 && toe <= contact::TOE_DORSI_MAX_DEG, "{toe}");
        assert!(r.metric("toe_push_n").unwrap() > 0.0 && r.metric("cone_use_max").unwrap() < 1.0);
        let s = r.samples.iter().find(|s| (0..2).any(|k| s.pose.feet[k][1] <= 1e-4 && s.pose.toe_deg[k] > 1.0)).expect("a foot on its toes");
        let k = (0..2).find(|&k| s.pose.feet[k][1] <= 1e-4 && s.pose.toe_deg[k] > 1.0).unwrap();
        let rep = kin::solve(&b, &s.pose, &s.loads);
        assert!(rep.support[0] > s.pose.feet[k][0], "{:?} vs ankle {}", rep.support, s.pose.feet[k][0]);
        let sole = rep.contact(if k == 0 { "foot_r" } else { "foot_l" }).and_then(|c| c.sole).unwrap();
        assert!(sole.rocker && sole.patches[0].normal_n == 0.0 && sole.patches[3].normal_n > 0.0, "{sole:?}");
        let (up, _) = walk(&b, Surface::Firm.props(), None, 8.0, 0.0, 0.9, 8, 1);
        assert!(up.metric("cone_use_max").unwrap() > r.metric("cone_use_max").unwrap() + 0.1);
    }

    #[test]
    fn the_samples_carry_a_stance_and_a_swing_leg() {
        let b = legs(Tier::Mid);
        let (r, _) = walk(&b, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 4, 1);
        let ss = r.samples.iter().filter(|s| s.phase == "single_support").max_by(|a, b| a.pose.feet[1][1].max(a.pose.feet[0][1]).total_cmp(&b.pose.feet[1][1].max(b.pose.feet[0][1]))).unwrap();
        assert!(ss.pose.feet.iter().any(|f| f[1] > 0.0), "a foot is in the air");
        let rep = kin::solve(&b, &ss.pose, &ss.loads);
        assert!(rep.contacts.iter().filter(|c| c.normal_n > 0.0).count() == 1, "one foot carries the robot");
        assert!(r.samples.windows(2).all(|w| w[1].t >= w[0].t));
    }
}
