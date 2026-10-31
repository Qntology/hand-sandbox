//! How a base turns. Skid-steered wheels and tracks turn by scrubbing their contact patches sideways: the
//! differential thrust they need comes from the turning-resistance moment M = μt·W·L/4, and μt falls as the turn
//! widens, μt(R) = μt0·0.925/(0.925 + 0.15·R/B) (Wong, *Theory of Ground Vehicles*, 4th ed., 2008, §7.2, the
//! empirical lateral-resistance law it quotes). When neither the motors nor the ground can give that thrust in
//! place, the base cannot pivot: it turns on an arc no tighter than R_min, and where the arc has no room it backs
//! and fills (a K-turn, forward–reverse–forward). Soft ground makes a pivot dig in, so a track turns on it like a
//! tractor at a headland. Legs turn by stepping and slow on tight curves, v = K·R^(1/3) (Vieilledent et al.,
//! Neurosci. Lett. 305, 2001; Hicheur et al., Exp. Brain Res. 162, 2005). Mecanum wheels translate without turning.
//! Targets close behind are reached in reverse rather than by turning round.

use noto_core::surface::{SurfaceProps, G};
use noto_core::{BaseKind, BaseSpec};
use serde::Serialize;

pub const TURN_POWER_SHARE: f32 = 0.6;
pub const THRUST_MARGIN: f32 = 0.85;
pub const MU_T_TRACK: f32 = 1.1;
pub const LAT_ACC_G: f32 = 0.25;
pub const REVERSE_SPEED_SHARE: f32 = 0.5;
pub const REVERSE_MAX_M: f32 = 3.0;
pub const LEG_REVERSE_MPS: f32 = 0.5;
pub const LEG_REVERSE_MAX_M: f32 = 1.2;
pub const LEG_SPIN_RAD_S: f32 = 1.6;
pub const LEG_CURVE_K: f32 = 1.0;
pub const STEP_TURN_RAD: f32 = 0.6;
pub const SHIFT_S: f32 = 0.5;
pub const MAX_RADIUS_M: f32 = 4.0;
pub const SAMPLE_M: f32 = 0.05;
pub const SAMPLE_RAD: f32 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    Skid,
    Legs,
    Holonomic,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TurnCaps {
    pub kind: TurnKind,
    pub pivot_rad_s: f32,
    pub min_radius_m: f32,
    pub yaw_rad_s: f32,
    pub lat_acc_mps2: f32,
    pub reverse_mps: f32,
    pub reverse_max_m: f32,
    pub thrust_need_n: f32,
    pub thrust_have_n: f32,
    pub soft: bool,
}

pub fn soft_ground(g: &SurfaceProps) -> bool {
    matches!(g.id, "grass" | "dirt" | "sand" | "wet_sand")
}

fn mu_t_at(mu_t0: f32, r: f32, b: f32) -> f32 {
    mu_t0 * 0.925 / (0.925 + 0.15 * r.max(0.0) / b.max(0.05))
}

/// Turning ability of a base of mass `spec.mass_kg`, contact length `len_m` (wheelbase or track length) and tread
/// `tread_m`, pushed by `drive_force_n` in total, on `ground`.
pub fn caps(spec: &BaseSpec, len_m: f32, tread_m: f32, drive_force_n: f32, ground: &SurfaceProps) -> TurnCaps {
    let v = spec.max_speed_mps.max(0.05);
    let soft = soft_ground(ground);
    match spec.kind {
        BaseKind::HumanFoot => TurnCaps {
            kind: TurnKind::Legs,
            pivot_rad_s: LEG_SPIN_RAD_S,
            min_radius_m: 0.0,
            yaw_rad_s: LEG_SPIN_RAD_S,
            lat_acc_mps2: 0.3 * G,
            reverse_mps: LEG_REVERSE_MPS.min(v),
            reverse_max_m: LEG_REVERSE_MAX_M,
            thrust_need_n: 0.0,
            thrust_have_n: 0.0,
            soft,
        },
        BaseKind::Mecanum => {
            let w = 2.0 * v / (len_m + tread_m).max(0.1);
            TurnCaps { kind: TurnKind::Holonomic, pivot_rad_s: w, min_radius_m: 0.0, yaw_rad_s: w, lat_acc_mps2: LAT_ACC_G * G, reverse_mps: v, reverse_max_m: f32::INFINITY, thrust_need_n: 0.0, thrust_have_n: 0.0, soft }
        }
        BaseKind::Wheel | BaseKind::Track => {
            let track = spec.kind == BaseKind::Track;
            let mu = if track { ground.mu_track } else { ground.mu_wheel };
            let mu_t0 = (mu * if track { MU_T_TRACK } else { 1.0 }).min(1.0);
            let w = spec.mass_kg.unwrap_or(30.0).max(1.0) * G;
            let (l, b) = (len_m.max(0.1), tread_m.max(0.1));
            let have = THRUST_MARGIN * drive_force_n.max(1.0).min(mu * w);
            let need = |r: f32| mu_t_at(mu_t0, r, b) * w * l / (2.0 * b);
            let r_min = if need(0.0) <= have { 0.0 } else { ((b / 0.15) * (0.925 * mu_t0 * w * l / (2.0 * b * have) - 0.925)).clamp(0.0, MAX_RADIUS_M) };
            let kin = 2.0 * v / b;
            let yaw_at = |r: f32| kin.min(TURN_POWER_SHARE * spec.power_w.max(1.0) / (mu_t_at(mu_t0, r, b) * w * l / 4.0).max(1e-3));
            let pivot = if r_min <= 0.0 { yaw_at(0.0) } else { 0.0 };
            TurnCaps {
                kind: TurnKind::Skid,
                pivot_rad_s: pivot,
                min_radius_m: r_min,
                yaw_rad_s: yaw_at(r_min.max(0.5 * b)),
                lat_acc_mps2: LAT_ACC_G * G * (mu / 0.85).min(1.0),
                reverse_mps: REVERSE_SPEED_SHARE * v,
                reverse_max_m: REVERSE_MAX_M,
                thrust_need_n: need(0.0),
                thrust_have_n: have,
                soft,
            }
        }
    }
}

impl TurnCaps {
    pub fn can_pivot(&self) -> bool {
        self.pivot_rad_s > 1e-3
    }

    /// Prefers a K-turn to a pivot: it cannot pivot, or it is a track on ground that a pivot would tear up.
    pub fn prefers_kturn(&self, track: bool) -> bool {
        self.kind == TurnKind::Skid && (!self.can_pivot() || (track && self.soft))
    }

    /// The comfortable radius for a corner taken at `v` m/s.
    pub fn corner_radius_m(&self, v: f32) -> f32 {
        let lat = v * v / self.lat_acc_mps2.max(0.1);
        let yaw = v / self.yaw_rad_s.max(0.05);
        match self.kind {
            TurnKind::Holonomic => 0.0,
            TurnKind::Legs => (v / LEG_CURVE_K).powi(3).max(0.15),
            TurnKind::Skid => self.min_radius_m.max(lat).max(yaw),
        }
    }

    /// The fastest a base can take an arc of radius `r` m.
    pub fn speed_on_radius(&self, r: f32, v_max: f32) -> f32 {
        if r <= 1e-3 {
            return 0.0;
        }
        let v = match self.kind {
            TurnKind::Holonomic => v_max,
            TurnKind::Legs => LEG_CURVE_K * r.cbrt(),
            TurnKind::Skid => (self.lat_acc_mps2 * r).sqrt().min(self.yaw_rad_s * r),
        };
        v.min(v_max)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Seg {
    Drive { dir: f32, kappa: f32, len: f32 },
    Pivot { angle: f32 },
}

impl Seg {
    pub fn yaw_change(&self) -> f32 {
        match *self {
            Seg::Drive { dir, kappa, len } => kappa * dir * len,
            Seg::Pivot { angle } => angle,
        }
    }

    pub fn length_m(&self) -> f32 {
        match *self {
            Seg::Drive { len, .. } => len,
            Seg::Pivot { .. } => 0.0,
        }
    }
}

pub type Pose2 = (f32, f32, f32);

/// The pose after the fraction `u` of `seg`, starting from `p` (unicycle, x forward along yaw, z to its left).
pub fn along(p: Pose2, seg: &Seg, u: f32) -> Pose2 {
    let (x, z, yaw) = p;
    match *seg {
        Seg::Pivot { angle } => (x, z, yaw + angle * u),
        Seg::Drive { dir, kappa, len } => {
            let s = dir * len * u;
            let bend = kappa * s;
            if bend.abs() < 1e-3 {
                // Nearly straight: the two ends of the arc below would be the same number in f32, and it would not move.
                (x + s * (yaw + 0.5 * bend).cos(), z + s * (yaw + 0.5 * bend).sin(), yaw + bend)
            } else {
                let y1 = yaw + kappa * s;
                (x + (y1.sin() - yaw.sin()) / kappa, z - (y1.cos() - yaw.cos()) / kappa, y1)
            }
        }
    }
}

pub fn end_of(p: Pose2, segs: &[Seg]) -> Pose2 {
    segs.iter().fold(p, |q, s| along(q, s, 1.0))
}

/// Whether every pose along `segs` from `p` is one the base may take (`free(x, z, yaw)`: its ground and its outline):
/// every `SAMPLE_M` along a drive, every `SAMPLE_RAD` round a pivot.
pub fn clear(p: Pose2, segs: &[Seg], free: &dyn Fn(f32, f32, f32) -> bool) -> bool {
    let mut q = p;
    for s in segs {
        let n = match *s {
            Seg::Pivot { angle } => (angle.abs() / SAMPLE_RAD).ceil().max(1.0) as usize,
            Seg::Drive { .. } => (s.length_m() / SAMPLE_M).ceil().max(1.0) as usize,
        };
        for i in 1..=n {
            let r = along(q, s, i as f32 / n as f32);
            if !free(r.0, r.1, r.2) {
                return false;
            }
        }
        q = along(q, s, 1.0);
    }
    true
}

/// A forward–reverse–forward turn by `delta` rad on arcs of radius `r` (`points` 3 or 5): every leg turns the body
/// the same way, as a car does with opposite lock when it reverses.
pub fn kturn(delta: f32, r: f32, points: usize) -> Vec<Seg> {
    let s = if delta >= 0.0 { 1.0 } else { -1.0 };
    let n = points.max(3) | 1;
    let a = delta.abs() / n as f32;
    let r = r.max(0.2);
    (0..n).map(|i| if i % 2 == 0 { Seg::Drive { dir: 1.0, kappa: s / r, len: r * a } } else { Seg::Drive { dir: -1.0, kappa: -s / r, len: r * a } }).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Maneuver {
    pub kind: &'static str,
    pub label_ko: String,
    pub segs: Vec<Seg>,
    pub time_s: f32,
}

pub fn seg_time(seg: &Seg, caps: &TurnCaps, v_fwd: f32) -> f32 {
    match *seg {
        Seg::Pivot { angle } => angle.abs() / caps.pivot_rad_s.max(0.05),
        Seg::Drive { dir, kappa, len } => {
            let v = if dir < 0.0 { caps.reverse_mps.min(v_fwd) } else { v_fwd };
            let v = if kappa.abs() > 1e-6 { v.min(caps.speed_on_radius(1.0 / kappa.abs(), v)) } else { v };
            len / v.max(0.05)
        }
    }
}

pub fn maneuver_time(segs: &[Seg], caps: &TurnCaps, v_fwd: f32) -> f32 {
    let shifts = segs.windows(2).filter(|w| match (w[0], w[1]) {
        (Seg::Drive { dir: a, .. }, Seg::Drive { dir: b, .. }) => a != b,
        _ => true,
    });
    segs.iter().map(|s| seg_time(s, caps, v_fwd)).sum::<f32>() + SHIFT_S * shifts.count() as f32
}

/// How to turn by `delta` rad where the base stands: a pivot (wheels scrub, legs step), or a K-turn when the base
/// cannot pivot, a track would tear soft ground, or the pivot's sweep would hit something. `None` when the base
/// cannot turn there at all (no room for any of them: a dead end, which the base backs out of).
pub fn turn_in_place(p: Pose2, delta: f32, caps: &TurnCaps, track: bool, v_fwd: f32, free: &dyn Fn(f32, f32, f32) -> bool) -> Option<Maneuver> {
    if delta.abs() < 1e-3 {
        return Some(Maneuver { kind: "none", label_ko: String::new(), segs: vec![], time_s: 0.0 });
    }
    let pivot = || {
        let segs = vec![Seg::Pivot { angle: delta }];
        let label_ko = match caps.kind {
            TurnKind::Legs => format!("제자리 걸음으로 {:.0}° 돎 (발 {}번)", delta.abs().to_degrees(), (delta.abs() / STEP_TURN_RAD).ceil().max(1.0)),
            TurnKind::Holonomic => format!("제자리에서 {:.0}° 돎 (메카넘)", delta.abs().to_degrees()),
            TurnKind::Skid => format!("제자리에서 {:.0}° 돎 (스키드 조향, 마찰 {:.0}/{:.0} N)", delta.abs().to_degrees(), caps.thrust_need_n, caps.thrust_have_n),
        };
        Maneuver { kind: "pivot", label_ko, time_s: maneuver_time(&segs, caps, v_fwd), segs }
    };
    let pivot_clear = || clear(p, &[Seg::Pivot { angle: delta }], free);
    if caps.can_pivot() && !caps.prefers_kturn(track) && pivot_clear() {
        return Some(pivot());
    }
    let r0 = caps.min_radius_m.max(if track { 0.6 } else { 0.4 });
    for points in [3usize, 5, 7] {
        for r in [r0 * 1.5, r0, (r0 * 0.7).max(caps.min_radius_m)] {
            if r < caps.min_radius_m - 1e-4 {
                continue;
            }
            let segs = kturn(delta, r, points);
            if clear(p, &segs, free) {
                let time_s = maneuver_time(&segs, caps, v_fwd * 0.5);
                return Some(Maneuver { kind: "kturn", label_ko: format!("{points}점 회전 {:.0}°: 전진·후진을 번갈아 반경 {r:.2} m", delta.abs().to_degrees()), segs, time_s });
            }
        }
    }
    (caps.can_pivot() && pivot_clear()).then(pivot)
}

/// Backs `dist_m` straight along the current heading reversed (the target is behind).
pub fn reverse_line(dist_m: f32) -> Vec<Seg> {
    vec![Seg::Drive { dir: -1.0, kappa: 0.0, len: dist_m.max(0.0) }]
}

pub fn wrap(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// A circular fillet that joins a segment heading `h_in` to one heading `h_out` at a corner, of radius `r`: the
/// distance from the corner to each tangent point, and the arc.
pub fn fillet(h_in: f32, h_out: f32, r: f32) -> (f32, Seg) {
    let turn = wrap(h_out - h_in);
    let d = r * (0.5 * turn.abs()).tan();
    let kappa = if r > 1e-6 { turn.signum() / r } else { 0.0 };
    (d, Seg::Drive { dir: 1.0, kappa, len: r * turn.abs() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use noto_core::surface::Surface;

    fn spec(kind: BaseKind, mass: f32) -> BaseSpec {
        BaseSpec { mass_kg: Some(mass), ..kind.spec() }
    }

    #[test]
    fn a_nearly_straight_arc_goes_as_far_as_it_says() {
        // A base pulling away: millimetres at a time, on a heading a turn left all but exact.
        let (x0, z0, yaw) = (17.2f32, 5.99f32, 0.973f32);
        for kappa in [0.0, 5e-6, -5e-6, 1e-3, 0.5] {
            let (x, z, _) = along((x0, z0, yaw), &Seg::Drive { dir: 1.0, kappa, len: 0.006 }, 1.0);
            assert!(((x - x0).hypot(z - z0) - 0.006).abs() < 2e-5, "{kappa}: {x} {z}");
        }
    }

    #[test]
    fn a_light_wheeled_base_pivots_and_a_weak_or_loaded_one_cannot() {
        let firm = Surface::Firm.props();
        let mid = caps(&spec(BaseKind::Wheel, 30.0), 0.30, 0.40, 240.0, firm);
        assert!(mid.can_pivot() && mid.min_radius_m == 0.0, "{mid:?}");
        assert!(mid.pivot_rad_s > 1.0 && mid.pivot_rad_s < 9.0, "power-limited pivot {mid:?}");
        let weak = caps(&spec(BaseKind::Wheel, 30.0), 0.30, 0.40, 46.0, firm);
        assert!(!weak.can_pivot() && weak.min_radius_m > 0.1, "46 N cannot scrub 30 kg round: {weak:?}");
        let loaded = caps(&spec(BaseKind::Wheel, 30.0 + 60.0), 0.30, 0.40, 240.0, firm);
        assert!(loaded.thrust_need_n > mid.thrust_need_n && loaded.pivot_rad_s < mid.pivot_rad_s, "cargo makes the scrub harder: {loaded:?}");
        let grass = caps(&spec(BaseKind::Wheel, 30.0), 0.30, 0.40, 240.0, Surface::Grass.props());
        assert!(grass.soft && grass.lat_acc_mps2 < mid.lat_acc_mps2);
    }

    #[test]
    fn a_track_resists_turning_more_than_wheels_and_turns_like_a_tractor_on_soft_ground() {
        let firm = Surface::Firm.props();
        let wheel = caps(&spec(BaseKind::Wheel, 40.0), 0.36, 0.40, 240.0, firm);
        let track = caps(&spec(BaseKind::Track, 40.0), 0.36, 0.40, 240.0, firm);
        assert!(track.thrust_need_n > wheel.thrust_need_n * 0.8, "{track:?} {wheel:?}");
        let soft = caps(&spec(BaseKind::Track, 40.0), 0.36, 0.40, 240.0, Surface::Dirt.props());
        assert!(soft.prefers_kturn(true), "{soft:?}");
        assert!(!wheel.prefers_kturn(false));
    }

    #[test]
    fn the_lateral_resistance_eases_on_a_wider_turn() {
        assert!(mu_t_at(0.8, 2.0, 0.4) < mu_t_at(0.8, 0.0, 0.4));
        assert!((mu_t_at(0.8, 0.0, 0.4) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn legs_step_round_and_slow_on_tight_curves() {
        let c = caps(&BaseKind::HumanFoot.spec(), 0.25, 0.3, 0.0, Surface::Firm.props());
        assert_eq!(c.kind, TurnKind::Legs);
        assert!(c.speed_on_radius(0.3, 1.2) < c.speed_on_radius(2.0, 1.2), "v = K·R^(1/3)");
        assert!((c.speed_on_radius(1.0, 1.2) - 1.0).abs() < 1e-5);
        let m = turn_in_place((0.0, 0.0, 0.0), 2.0, &c, false, 1.0, &|_, _, _| true).unwrap();
        assert_eq!(m.kind, "pivot");
        assert!(m.label_ko.contains("걸음") && (m.time_s - 2.0 / LEG_SPIN_RAD_S).abs() < 1e-4, "{m:?}");
    }

    #[test]
    fn every_leg_of_a_kturn_turns_the_body_the_same_way_and_the_turn_adds_up() {
        let segs = kturn(2.4, 0.6, 3);
        assert_eq!(segs.len(), 3);
        assert!(segs.iter().all(|s| s.yaw_change() > 0.0), "{segs:?}");
        let (x, z, yaw) = end_of((1.0, 2.0, 0.3), &segs);
        assert!((wrap(yaw - 0.3) - 2.4).abs() < 1e-4);
        assert!((x - 1.0).hypot(z - 2.0) < 1.2, "a K-turn stays near where it started: {x} {z}");
        assert!(matches!(segs[1], Seg::Drive { dir, .. } if dir < 0.0), "the middle leg reverses");
    }

    #[test]
    fn a_base_that_cannot_pivot_backs_and_fills_and_gives_up_without_room() {
        let firm = Surface::Firm.props();
        let weak = caps(&spec(BaseKind::Wheel, 30.0), 0.30, 0.40, 46.0, firm);
        let open = turn_in_place((0.0, 0.0, 0.0), 1.6, &weak, false, 1.0, &|_, _, _| true).unwrap();
        assert_eq!(open.kind, "kturn");
        assert!(open.segs.iter().any(|s| matches!(s, Seg::Drive { dir, .. } if *dir < 0.0)), "it reverses on the way round");
        let walled = turn_in_place((0.0, 0.0, 0.0), 1.6, &weak, false, 1.0, &|x, z, _| x.hypot(z) < 0.05);
        assert!(walled.is_none(), "no room and no pivot: {walled:?}");
        let mid = caps(&spec(BaseKind::Wheel, 30.0), 0.30, 0.40, 240.0, firm);
        let pv = turn_in_place((0.0, 0.0, 0.0), 1.6, &mid, false, 1.0, &|x, z, _| x.hypot(z) < 0.05).unwrap();
        assert_eq!(pv.kind, "pivot");
    }

    #[test]
    fn a_fillet_is_tangent_to_both_legs() {
        let (d, arc) = fillet(0.0, std::f32::consts::FRAC_PI_2, 0.5);
        assert!((d - 0.5).abs() < 1e-5);
        let start = (-d, 0.0, 0.0);
        let (x, z, yaw) = along(start, &arc, 1.0);
        assert!(x.abs() < 1e-4 && (z - d).abs() < 1e-4 && (yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-5, "{x} {z} {yaw}");
        let (d45, _) = fillet(0.0, std::f32::consts::FRAC_PI_4, 1.0);
        assert!((d45 - (std::f32::consts::FRAC_PI_8).tan()).abs() < 1e-5);
    }
}
