//! Walking step by step on two legs.
//!
//! The walking controller is an inverted pendulum whose CoM rises and falls
//! as a person's does and whose centre of pressure rolls from heel to toe
//! (v0.7; before it the linear inverted pendulum, LIPM, at a constant height):
//!
//! * the CoM height follows a cosine at the step frequency about the walking
//!   height `z_c` (the standing CoM lowered by the walking knee bend), lowest
//!   in the middle of double support and highest over the stance foot, with
//!   the peak-to-peak excursion people walk with at the same dimensionless
//!   speed (Massaad et al. 2007): the vertical ground force is `m·(g + z̈)`,
//!   above the weight while both feet are down and below it in single
//!   support, which is the two humps and the valley of a walking stance;
//! * the centre of pressure runs under the stance foot from behind the heel
//!   centre to the forefoot as a person's does over the stance (Jeon & Cho
//!   2020), and from the trailing foot to the leading one in double support;
//! * the horizontal motion is the pendulum's `ẍ = (g + z̈)·(x − p)/z`, solved
//!   for the path that repeats from step to step (the map from the step's
//!   start to its end is affine, so two unit runs and a 2×2 solve give it);
//!   the friction a step needs is the largest lean of the ground force,
//!   `|x − p| / z`, plus the slope.
//!
//! Each step:
//!
//! * the step length and the double support follow the speed's Froude number
//!   as people's walking does (dynamic similarity: the stride and the duty
//!   factor of Donelan & Kram 1997), then the controller shortens the step
//!   until the friction it needs (the path's largest lean plus the slope)
//!   stays a margin below what it believes the ground gives (`KEEP_MARGIN`);
//!   the cadence stays and the speed falls with the step;
//! * the swing foot lands at the controller's touchdown speed; the foot and
//!   the shank hit the ground through the sole and the ground in series: the
//!   heel-strike impact;
//! * the ground's friction under this step is drawn from its scatter; the
//!   friction demand peaks at the step transitions, and the step's slip
//!   probability is the level-walking logistic of `surface` for that margin;
//! * a slipping foot slides with the force the friction does not hold over
//!   the controller's reaction delay; the robot recovers when that slide stays
//!   short (people fall past about 10 cm, scaled by leg length) and the step it
//!   needs lands inside its legs' reach (the capture point `x + ẋ·√(z_c/g)`,
//!   Pratt 2006), otherwise it falls.
//!
//! Samples are solved by `kin` (the stance and swing legs by inverse
//! kinematics, the CoM's acceleration as inertia), so the foot and joint
//! focus views get the ground reaction, the centre of pressure and every
//! leg joint per sample.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;

use crate::balance::{Body, G, KNEE_LEVER_WALK_M, REACTION_DELAY_S};
use crate::kin::{self, KinLoads, KinPose, ANKLE_H_M, SHANK_M, THIGH_M};
use crate::parts::*;
use crate::surface::{self, SurfaceProps};

use super::{ActEvent, ActKind, ActRollout, ActSample, Draw, Outcome, Part, SAMPLE_S};

/// Friction margin the gait controller keeps below the ground's believed friction (est: ≈0.3% slip chance per step on the `surface` logistic).
pub const KEEP_MARGIN: f32 = 0.10;
/// Height the swing foot clears, metres (as `balance`'s single support).
pub const SWING_LIFT_M: f32 = 0.05;
/// The least height of a swing foot, metres: off the ground (and unloaded) from the first instant of single support.
pub const SWING_OFF_M: f32 = 0.002;
/// Vertical speed at which the controller lands the swing foot, m/s (est: a
/// person's heel lands with a negligible vertical speed, Winter 1992; a
/// position-controlled foot comes down harder): research §3.
pub const TOUCHDOWN_MPS: f32 = 0.15;
/// Stiffness of the robot's sole (rubber over an aluminium or printed plate),
/// N/m: the upper end of a person's heel pad in walking, 32–212 N/mm
/// (Wearing 2014): research §3.
pub const SOLE_STIFFNESS: f32 = 2.0e5;
/// Scatter of the ground's friction from step to step (coefficient of variation): research §3.
pub const GROUND_MU_COV: f32 = 0.10;
/// A slide longer than this fraction of the leg ends in a fall even when a
/// step could catch the CoM: people's slips become falls past about 10 cm
/// (Strandberg & Lanshammar 1981; Cham & Redfern 2002; as summarised by
/// Moyer 2006) on a 0.88 m leg, scaled to the robot's by dynamic similarity.
pub const FALL_SLIDE_LEG: f32 = 0.10 / 0.88;
/// Froude number at which people change from walking to running (Kram, Domingo & Ferris 1997).
pub const WALK_RUN_FROUDE: f32 = 0.5;
/// Duty factor (stance share of a stride) over the Froude number on a 0.88 m
/// leg: Donelan & Kram 1997 Table 2 (0.25–0.75 m/s) and Table 1 (Fr 0.1–0.4).
const DUTY_FACTOR: [(f32, f32); 7] = [(0.0072, 0.77), (0.029, 0.71), (0.065, 0.68), (0.10, 0.66), (0.20, 0.64), (0.30, 0.63), (0.40, 0.62)];
/// Leg length of the people the human gait numbers come from, metres (as [`FALL_SLIDE_LEG`]).
const PERSON_LEG_M: f32 = 0.88;
/// Peak-to-peak vertical excursion of a person's CoM over the walking speed
/// (m/s, m): 0.015 ± 0.003 m at 1 km/h to 0.034 ± 0.004 m at 6 km/h, linear
/// between, lowest in double support (Massaad et al. 2007, J Physiol 582,
/// Fig. 1A, n = 6); carried to the robot by dynamic similarity (over the leg
/// length against the speed over √(g·ℓ)) and held at the ends.
pub const COM_EXCURSION: [(f32, f32); 2] = [(1.0 / 3.6, 0.015), (6.0 / 3.6, 0.034)];
/// Where a person's centre of pressure runs under the foot over a stance:
/// the mean of each of five phases, at the phase's middle (share of the
/// stance, share of the foot length in front of the heel centre; Jeon & Cho
/// 2020, IJERPH 17:7845, Tables 1–2, n = 90 at 1.29 m/s, the phases ending at
/// 8.83, 40.85, 59.32, 87.02 and 100% of the stance), held at the ends.
pub const COP_PATH: [(f32, f32); 5] = [(0.0442, -0.0745), (0.2484, 0.1619), (0.5009, 0.4034), (0.7317, 0.5921), (0.9351, 0.7055)];
/// The heel centre (the origin of [`COP_PATH`]) lies this share of the foot
/// length in front of the heel's back edge (est: the middle of the heel, the
/// rear 30% of the foot): `docs/research/perception.md` §1.6.
pub const HEEL_CENTRE_FL: f32 = 0.15;
/// The ankle lies this share of the sole's length in front of its back edge (as `kin`'s sole).
pub const ANKLE_FL: f32 = 0.25;
/// The sole the route planner assumes when the base does not say (the mid legs'), metres.
pub const NOMINAL_FOOT_M: f32 = 0.20;
/// Integration steps per walking step of the CoM path.
const PATH_STEPS: usize = 160;

/// The gait the controller walks at a speed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GaitPlan {
    /// Commanded and actual speed (after shortening the step), m/s.
    pub speed_cmd: f32,
    pub speed: f32,
    pub step_m: f32,
    pub step_s: f32,
    /// Share of a step both feet are on the ground.
    pub ds_frac: f32,
    /// CoM height, metres.
    pub com_z: f32,
    /// Friction a step needs at its transitions (slope included).
    pub rcof: f32,
    /// Friction the controller believes the ground gives.
    pub mu_est: f32,
    /// The step was shortened for friction.
    pub shortened: bool,
    /// Peak-to-peak vertical excursion of the CoM over a step, metres.
    pub com_amp_m: f32,
    /// Sole length the centre of pressure rolls along, metres.
    pub foot_m: f32,
    /// Where the CoM starts each step, relative to the new stance ankle, and its speed then (the path that repeats).
    pub x0: f32,
    pub v0: f32,
}

/// Relative step length λ/ℓ at a Froude number: half the stride, whose
/// relative length is the power fit `2.47·Fr^0.29` to Donelan & Kram 1997
/// Table 1 (Fr 0.1/0.2/0.3/0.4 → stride/ℓ 1.26/1.54/1.74/1.89, within 0.01).
pub fn relative_step(froude: f32) -> f32 {
    0.5 * 2.47 * froude.clamp(1e-4, WALK_RUN_FROUDE).powf(0.29)
}

/// Double-support share of a step at a Froude number. Each of a stride's two
/// double supports lasts `Df − 0.5` of it, so a step (half a stride) spends
/// `2·Df − 1` of its time on both feet, with the duty factor `Df` of
/// [`DUTY_FACTOR`] (held at the table's ends).
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

/// The mass that hits the ground with the landing foot: its foot and shank
/// (a person's heel-strike effective mass in walking is 6.3% of the body,
/// Chi & Schmitt 2005, about the foot and shank of the segment table,
/// 1.45% + 4.65%, Dempster via Winter 2009).
pub fn impact_mass_kg(body: &Body) -> f32 {
    body.leg.foot_mass_kg + body.leg.mass_kg * (1.0 - kin::thigh_share(body))
}

/// CoM height while walking: standing height lowered by the knee bend of the walking knee lever.
pub fn walking_com_z(body: &Body) -> f32 {
    let bend = (KNEE_LEVER_WALK_M / SHANK_M).clamp(0.0, 0.9).asin();
    body.com_height_m() - (THIGH_M + SHANK_M) * (1.0 - bend.cos())
}

/// Peak-to-peak CoM excursion at `speed` on a leg of `leg_m`: a person's
/// ([`COM_EXCURSION`]) at the same dimensionless speed, over their leg length.
pub fn com_excursion_m(speed: f32, leg_m: f32) -> f32 {
    let hat = |v: f32, l: f32| v / (G * l).sqrt();
    let (a, b) = (COM_EXCURSION[0], COM_EXCURSION[1]);
    let u = ((hat(speed, leg_m) - hat(a.0, PERSON_LEG_M)) / (hat(b.0, PERSON_LEG_M) - hat(a.0, PERSON_LEG_M))).clamp(0.0, 1.0);
    (a.1 + (b.1 - a.1) * u) / PERSON_LEG_M * leg_m
}

/// How far the walking CoM can rise above `walking_com_z` before the stance leg is straight, metres.
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

/// One step of the walk: the stance ankle at x = 0, the trailing one at −λ.
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

    /// CoM height and vertical acceleration `t` into the step: lowest in the middle of double support.
    fn z(&self, t: f32) -> (f32, f32) {
        let w = std::f32::consts::TAU / self.step_s.max(1e-3);
        let c = (w * (t - 0.5 * self.t_ds)).cos();
        (self.z_c - 0.5 * self.amp * c, 0.5 * self.amp * w * w * c)
    }

    /// A stance foot's centre of pressure `f` (share of its stance) into the
    /// stance, from its ankle. The pressure advances one step per step, so a
    /// step shorter than the sole's roll (a shuffle) rolls only that much of
    /// it, about the same point (est: the model's limit; people's steps are
    /// longer than their feet).
    fn sole(&self, f: f32) -> f32 {
        let span = COP_PATH[COP_PATH.len() - 1].1 - COP_PATH[0].1;
        let roll = self.foot_m.min(self.lam / span);
        roll * (interp(&COP_PATH, f) + HEEL_CENTRE_FL - ANKLE_FL)
    }

    /// The centre of pressure `t` into the step: from the trailing foot's to the
    /// leading foot's while both are down, then along the stance foot.
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

    /// Runs the step from (x0, v0), calling `each(t, x, v)` at every point; returns the end.
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

    /// The start that repeats: the step ends λ further on (over the next
    /// stance ankle) at the speed it began with. The map from the start to the
    /// end is affine, so the base run and two unit runs give it.
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

    /// The largest lean of the ground force over the step, `|x − p| / z`.
    fn lean(&self, x0: f32, v0: f32) -> f32 {
        let mut worst = 0.0f32;
        self.run(x0, v0, |t, x, _| worst = worst.max((x - self.cop(t)).abs() / self.z(t).0));
        worst
    }
}

/// A step's model, its repeating start and its lean, for a step of `step_m` at the cadence of `step_s`.
fn step_lean(z_c: f32, foot_m: f32, step_m: f32, step_s: f32, ds_frac: f32) -> (f32, f32, f32, f32) {
    let speed = step_m / step_s.max(1e-3);
    let amp = com_excursion_m(speed, HIP_M).min(2.0 * com_rise_room_m());
    let m = StepModel { lam: step_m, step_s, t_ds: ds_frac * step_s, z_c, amp, foot_m };
    let (x0, v0) = m.periodic();
    (m.lean(x0, v0), amp, x0, v0)
}

/// Plans already made, by their inputs (the route planner asks for the same few again and again).
static PLANS: Mutex<Option<HashMap<[u32; 5], GaitPlan>>> = Mutex::new(None);

/// The gait at `speed` on a ground with friction `mu_est` and a slope.
pub fn plan(body: &Body, speed: f32, mu_est: f32, slope_deg: f32) -> GaitPlan {
    plan_z(walking_com_z(body), body.leg.foot_len_m, speed, mu_est, slope_deg)
}

/// [`plan`] from the walking CoM height and the sole alone (the route planner has no body).
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
        // The longest step whose lean stays inside the room (the lean grows with the step at a fixed cadence).
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

/// The CoM `t` into a step of the plan: its position from the stance ankle,
/// its height, its acceleration (forward, up) and the centre of pressure.
pub fn com_at(g: &GaitPlan, t: f32) -> ComState {
    let m = StepModel::of(g);
    let t = t.clamp(0.0, g.step_s);
    let mut best = (0.0f32, g.x0, g.v0);
    m.run(g.x0, g.v0, |tt, x, v| {
        if tt <= t + 1e-6 {
            best = (tt, x, v);
        }
    });
    // From the last point before `t` on to it, by the point's own speed and acceleration.
    let dt = t - best.0;
    let a = m.accel(best.0, best.1);
    let x = best.1 + best.2 * dt + 0.5 * a * dt * dt;
    let (z, zdd) = m.z(t);
    ComState { x, v: best.2 + a * dt, z, ax: m.accel(t, x), az: zdd, cop: m.cop(t) }
}

/// The CoM at an instant of a step (`com_at`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ComState {
    pub x: f32,
    pub v: f32,
    pub z: f32,
    pub ax: f32,
    pub az: f32,
    pub cop: f32,
}

/// One walked step's numbers.
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
}

/// Walks `steps` steps (or until a fall) on a ground whose friction the
/// controller believes is `mu_belief` (`None`: the ground's own, as the map
/// says); `seed` draws each step's friction and slip.
pub fn walk(body: &Body, ground: &SurfaceProps, mu_belief: Option<f32>, slope_deg: f32, cargo_kg: f32, speed: f32, steps: u32, seed: u64) -> (ActRollout, Vec<StepStat>) {
    let mut d = Draw::new(seed);
    let mu_mean = surface::available_mu(body.base, ground);
    let g = plan(body, speed, mu_belief.unwrap_or(mu_mean), slope_deg);
    let m = body.mass_kg() + cargo_kg;
    let t_ds = g.ds_frac * g.step_s;
    let t_ss = g.step_s - t_ds;
    let lam = g.step_m;
    // The CoM path repeats from step to step: sample it once.
    let n_ds = (t_ds / SAMPLE_S).ceil().max(1.0) as usize;
    let n_ss = (t_ss / SAMPLE_S).ceil().max(1.0) as usize;
    let path_ds: Vec<ComState> = (0..n_ds).map(|i| com_at(&g, i as f32 / n_ds as f32 * t_ds)).collect();
    let path_ss: Vec<ComState> = (0..n_ss).map(|i| com_at(&g, t_ds + i as f32 / n_ss as f32 * t_ss)).collect();
    let half_w = 0.5 * STANCE_WIDTH_M;
    let m_imp = impact_mass_kg(body);
    let k_ground = super::material::ground_stiffness(crate::surface::Surface::from_id(ground.id).unwrap_or_default());
    let k_eff = 1.0 / (1.0 / SOLE_STIFFNESS + 1.0 / k_ground);
    let (f_imp, dt_imp) = super::material::impact(TOUCHDOWN_MPS, m_imp, k_eff);
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let mut stats = Vec::new();
    let mut outcome = Outcome::ok();
    let mut t0 = 0.0f32;
    let mut x_base = 0.0f32; // the stance ankle's x along the walk
    let mut slips = 0u32;
    'steps: for k in 0..steps {
        let stance = (k % 2) as usize;
        let swing = 1 - stance;
        let ys = if stance == 0 { -half_w } else { half_w };
        // The step's friction and its slip draw.
        let mu = d.lognormal(mu_mean, GROUND_MU_COV);
        let margin = mu - g.rcof;
        let p = surface::slip_probability(margin);
        let slipped = d.uniform() < p;
        let mut peak = 0.0f32;
        let mut low = f32::INFINITY;
        // Double support: the centre of pressure moves from the trailing foot to the new stance foot, the CoM at its lowest.
        for (i, c) in path_ds.iter().enumerate() {
            let u = i as f32 / n_ds as f32;
            let t = u * t_ds;
            // The heel-strike transient on the landing foot over its contact time.
            let fz_extra = if t < dt_imp { f_imp * (std::f32::consts::PI * t / dt_imp).sin() } else { 0.0 };
            let bw = (m * (G + c.az) + fz_extra) / (m * G);
            peak = peak.max(bw);
            low = low.min(bw);
            let mut pose = KinPose::stand(cargo_kg);
            pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
            pose.feet[stance] = [0.0, 0.0];
            pose.feet[swing] = [-lam, 0.0];
            // The pelvis moves from over the old stance foot to over the new one.
            pose.pelvis_y = ys * (2.0 * u - 1.0);
            pose.shoulder_deg = [8.0, -8.0];
            let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
            let r = kin::solve_fast(body, &pose, &loads);
            samples.push(ActSample { t: t0 + t, phase: "double_support", pose: shift(pose, x_base), loads, margin: r.margin, zmp: [r.zmp[0] + x_base, r.zmp[1]], com: [r.com[0] + x_base, r.com[1], r.com[2]], peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: None });
        }
        events.push(ActEvent::new(t0, "heel_strike", format!("{}발 착지 · 충격 {:.0} N ({:.0} ms) · 필요 마찰 {:.2} / 지면 {:.2}", if stance == 0 { "오른" } else { "왼" }, f_imp, dt_imp * 1000.0, g.rcof, mu), &[Part::Foot, Part::FootJoints], Some(f_imp), true));
        // A slip at the transition: the slide over the reaction delay, then a catch or a fall.
        if slipped {
            slips += 1;
            let (fell, slide) = slip_outcome(&g, mu);
            stats.push(StepStat { index: k, stance, mu, rcof: g.rcof, slip_p: p, slipped, fell, slide_m: slide, impact_n: f_imp, grf_peak_bw: peak, grf_min_bw: low });
            if fell {
                events.push(ActEvent::new(t0 + t_ds * 0.5, "fall", format!("미끄러져 넘어짐: 마찰 {mu:.2} < 필요 {:.2}, {:.0} mm 미끄러짐", g.rcof, slide * 1000.0), &[Part::Foot, Part::FootJoints, Part::Waist], Some(slide), false));
                outcome = Outcome::fail("fall", format!("{}번째 걸음에서 미끄러져 넘어짐 (지면 마찰 {mu:.2} < 필요 {:.2}, 미끄럼 {:.0} mm)", k + 1, g.rcof, slide * 1000.0));
                t0 += t_ds;
                break 'steps;
            }
            events.push(ActEvent::new(t0 + t_ds * 0.5, "foot_slip", format!("발 미끄러짐 {:.0} mm · 마찰 {mu:.2} < 필요 {:.2} → 회복", slide * 1000.0, g.rcof), &[Part::Foot, Part::FootJoints], Some(slide), true));
        }
        // Single support: the CoM vaults over the stance foot, rising over it, the swing foot travels 2λ.
        for (i, c) in path_ss.iter().enumerate() {
            let u = i as f32 / n_ss as f32;
            let t = u * t_ss;
            let mut pose = KinPose::stand(cargo_kg);
            pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
            pose.feet[stance] = [0.0, 0.0];
            let sw = super::min_jerk(u);
            // The swing foot is off the ground from the first instant of single support.
            pose.feet[swing] = [-lam + 2.0 * lam * sw, (SWING_LIFT_M * (std::f32::consts::PI * u).sin()).max(SWING_OFF_M)];
            pose.pelvis_y = ys;
            pose.shoulder_deg = if stance == 0 { [-8.0, 8.0] } else { [8.0, -8.0] };
            let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
            let r = kin::solve_fast(body, &pose, &loads);
            let bw = (G + c.az) / G;
            peak = peak.max(bw);
            low = low.min(bw);
            samples.push(ActSample { t: t0 + t_ds + t, phase: "single_support", pose: shift(pose, x_base), loads, margin: r.margin, zmp: [r.zmp[0] + x_base, r.zmp[1]], com: [r.com[0] + x_base, r.com[1], r.com[2]], peak_ratio: r.worst().map_or(0.0, |j| j.ratio), grip: None, object: None });
        }
        events.push(ActEvent::new(t0 + t_ds, "toe_off", format!("{}발 들림", if swing == 0 { "오른" } else { "왼" }), &[Part::Foot], None, true));
        if !slipped {
            stats.push(StepStat { index: k, stance, mu, rcof: g.rcof, slip_p: p, slipped, fell: false, slide_m: 0.0, impact_n: f_imp, grf_peak_bw: peak, grf_min_bw: low });
        }
        t0 += g.step_s;
        x_base += lam;
    }
    let walked = stats.len() as f32;
    let metrics = vec![
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
    let notes = if g.shortened { vec![format!("마찰 {:.2}에 맞춰 보폭을 {:.2} m로 줄임 (속도 {:.2} → {:.2} m/s)", g.mu_est, g.step_m, g.speed_cmd, g.speed)] } else { vec![] };
    (ActRollout { kind: ActKind::Walk, samples, events, outcome, duration_s: t0, metrics, notes }, stats)
}

/// The walking pose a fraction `u` (0..1) into a step whose stance foot is
/// `stance` (0 right, 1 left), feet relative to the stance ankle at x = 0: the
/// same LIPM as [`walk`], for drawing a robot as it walks a route. Returns the
/// pose, its loads (the CoM's acceleration) and the phase.
pub fn pose_at(body: &Body, g: &GaitPlan, stance: usize, u: f32, cargo_kg: f32) -> (KinPose, KinLoads, &'static str) {
    let t_ds = g.ds_frac * g.step_s;
    let t_ss = g.step_s - t_ds;
    let lam = g.step_m;
    let swing = 1 - stance;
    let ys = if stance == 0 { -0.5 * STANCE_WIDTH_M } else { 0.5 * STANCE_WIDTH_M };
    let t = u.clamp(0.0, 1.0) * g.step_s;
    let c = com_at(g, t);
    let mut pose = KinPose::stand(cargo_kg);
    pose.feet[stance] = [0.0, 0.0];
    pose.hip = [c.x, c.z + (HIP_M - body.com_height_m())];
    let loads = KinLoads { accel: [c.ax, 0.0, c.az], ..KinLoads::default() };
    if t < t_ds {
        let v = t / t_ds.max(1e-4);
        pose.feet[swing] = [-lam, 0.0];
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

/// What a slip at a step transition does on a ground of friction `mu`: the
/// foot slides with the shear friction does not hold for the controller's
/// reaction delay, then the next step must land at the capture point
/// (`x + v/ω`, ω = √(g/z_c), Pratt 2006) within the legs' reach, and the
/// slide must stay under [`FALL_SLIDE_LEG`]. Returns whether it is a fall and
/// the slide, metres.
pub fn slip_outcome(g: &GaitPlan, mu: f32) -> (bool, f32) {
    let mu_k = 0.85 * mu;
    let a_slide = (G * (g.rcof - mu_k)).max(0.05 * G);
    let slide = 0.5 * a_slide * REACTION_DELAY_S * REACTION_DELAY_S + g.speed * REACTION_DELAY_S * (g.rcof - mu_k).max(0.0) / g.rcof.max(1e-3);
    let omega = (G / g.com_z.max(0.1)).sqrt();
    let capture = 0.5 * g.step_m + g.speed / omega;
    let reach = 2.0 * (THIGH_M + SHANK_M) * 0.8f32.sin();
    (slide > FALL_SLIDE_LEG * HIP_M || capture + slide > reach, slide)
}

/// Moves a step's pose along the walk.
fn shift(mut p: KinPose, dx: f32) -> KinPose {
    p.hip[0] += dx;
    p.feet[0][0] += dx;
    p.feet[1][0] += dx;
    p
}

/// The joint angles of a leg at a hip and an ankle position (for drawing a gait without solving it).
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
        // The heel-to-toe roll keeps the ground force leaning less than the flat pendulum's λ/(2·z).
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
        // A low-friction floor the controller does not know about (it believes firm ground).
        let ice = SurfaceProps { mu_foot: 0.12, ..*Surface::Firm.props() };
        let firm_mu = Surface::Firm.props().mu_foot;
        let slipped: u32 = (0..20).map(|s| walk(&b, &ice, Some(firm_mu), 0.0, 0.0, 0.9, 20, s).1.iter().filter(|x| x.slipped).count() as u32).sum();
        assert!(slipped > 0, "a floor slicker than the gait shears slips");
        // Told the truth, the controller shortens its steps and does not slip.
        let (r, st) = walk(&b, &ice, None, 0.0, 0.0, 0.9, 20, 3);
        assert!(r.metric("step_m").unwrap() < 0.2 && st.iter().filter(|x| x.slipped).count() <= 1, "{:?}", r.notes);
    }

    #[test]
    fn the_com_bobs_and_the_stance_force_has_two_humps_and_a_valley() {
        let b = legs(Tier::Mid);
        let g = plan(&b, 0.9, Surface::Firm.props().mu_foot, 0.0);
        // A person's excursion at the same dimensionless speed (Massaad 2007), inside the stance leg's room.
        assert!(g.com_amp_m > 0.008 && g.com_amp_m < 2.0 * com_rise_room_m(), "{g:?}");
        // The path repeats: the step ends λ on at the speed it began with.
        let end = com_at(&g, g.step_s);
        assert!((end.x - g.step_m - g.x0).abs() < 0.01 && (end.v - g.v0).abs() < 0.02, "{end:?} vs {g:?}");
        assert!(g.v0 > 0.0 && (0..20).all(|i| com_at(&g, i as f32 * g.step_s / 20.0).v > 0.0), "always moving on");
        // Lowest with both feet down, highest over the stance foot; the force follows z̈.
        let t_ds = g.ds_frac * g.step_s;
        let (ds, ss) = (com_at(&g, 0.5 * t_ds), com_at(&g, t_ds + 0.5 * (g.step_s - t_ds)));
        assert!(ds.z < ss.z && ds.az > 0.0 && ss.az < 0.0);
        // The centre of pressure rolls forward under the stance foot.
        assert!(com_at(&g, t_ds).cop < com_at(&g, g.step_s - 1e-3).cop);
        let (r, _) = walk(&b, Surface::Firm.props(), None, 0.0, 0.0, 0.9, 6, 1);
        let fz: Vec<(f32, f32)> = r.samples.iter().filter(|s| s.pose.feet[0][1] <= 1e-4).map(|s| (s.t, kin::solve_fast(&b, &s.pose, &s.loads).contact("foot_r").map_or(0.0, |c| c.normal_n))).collect();
        let w = b.mass_kg() * G;
        let (hi, lo) = fz.iter().fold((0.0f32, f32::INFINITY), |(h, l), p| (h.max(p.1 / w), if p.1 > 0.5 * w { l.min(p.1 / w) } else { l }));
        assert!(hi > 1.05 && lo < 0.95, "humps {hi} valley {lo}");
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
