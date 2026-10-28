use serde::Serialize;

use crate::balance::{Body, G};
use crate::base::BaseKind;
use crate::kin::{self, KinLoads, KinPose, WHEEL_R_M};
use crate::surface::{self, Surface, SurfaceProps};

use super::{ActEvent, ActKind, ActRollout, ActSample, Outcome, Part, SAMPLE_S};

pub const PEAK_SLIP: f32 = 0.2;

pub fn loose(ground: Surface) -> bool {
    matches!(ground, Surface::Sand | Surface::WetSand)
}

pub fn shear_modulus_m(ground: Surface) -> f32 {
    match ground {
        Surface::Firm | Surface::Rock => 0.002,
        Surface::Gravel => 0.010,
        Surface::Dirt => 0.015,
        Surface::Grass => 0.020,
        Surface::Sand => 0.010,
        Surface::WetSand => 0.006,
        Surface::Water => 0.05,
    }
}

pub fn wheel_contact_m(ground: Surface) -> f32 {
    match ground {
        Surface::Firm | Surface::Rock => 0.02,
        Surface::Gravel | Surface::Dirt => 0.03,
        Surface::Grass | Surface::WetSand => 0.04,
        Surface::Sand => 0.05,
        Surface::Water => 0.05,
    }
}

pub const TYRE_STIFFNESS: f32 = 1.5e5;

pub fn thrust_fraction(s: f32, l: f32, k: f32) -> f32 {
    let s = s.clamp(1e-5, 1.0);
    let x = s * l / k.max(1e-5);
    1.0 - (1.0 - (-x).exp()) / x
}

pub fn slip_for(need: f32, l: f32, k: f32) -> Option<f32> {
    if need <= 0.0 {
        return Some(0.0);
    }
    if need >= thrust_fraction(1.0, l, k) {
        return None;
    }
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if thrust_fraction(mid, l, k) < need {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

pub fn traction_fraction(ground: Surface, s: f32, l: f32) -> f32 {
    let k = shear_modulus_m(ground);
    if loose(ground) {
        thrust_fraction(s, l, k)
    } else {
        (thrust_fraction(s, l, k) / thrust_fraction(PEAK_SLIP, l, k)).min(1.0)
    }
}

pub fn slip_on(ground: Surface, need: f32, l: f32) -> Option<f32> {
    if need <= 0.0 {
        return Some(0.0);
    }
    if need > traction_fraction(ground, 1.0, l) {
        return None;
    }
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if traction_fraction(ground, mid, l) < need {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

pub fn contact_len_m(body: &Body, ground: Surface) -> f32 {
    if body.base == BaseKind::Track {
        body.support_len_m
    } else {
        wheel_contact_m(ground)
    }
}

pub fn drawbar_over_weight(body: &Body, ground: &SurfaceProps, s: f32) -> f32 {
    let gs = Surface::from_id(ground.id).unwrap_or_default();
    surface::available_mu(body.base, ground) * traction_fraction(gs, s, contact_len_m(body, gs)) - surface::rolling_resistance(body.base, ground)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DriveState {
    pub need_n: f32,
    pub limit_n: f32,
    pub slip: f32,
    pub speed_factor: f32,
    pub front_n: f32,
    pub rear_n: f32,
    pub pressure_kpa: Option<[f32; 2]>,
    pub margin: f32,
}

pub fn state(body: &Body, ground: &SurfaceProps, slope_deg: f32, accel: f32, cargo_kg: f32, push_n: f32) -> DriveState {
    let m = body.mass_kg() + cargo_kg;
    let th = slope_deg.to_radians();
    let crr = surface::rolling_resistance(body.base, ground);
    let need = m * (G * (th.sin() + crr * th.cos()) + accel) + push_n;
    let mu = surface::available_mu(body.base, ground);
    let normal = m * G * th.cos();
    let limit = mu * normal;
    let gs = Surface::from_id(ground.id).unwrap_or_default();
    let slip = slip_on(gs, need / limit.max(1e-3), contact_len_m(body, gs)).unwrap_or(1.0);
    let pose = KinPose { base_pitch_deg: slope_deg, ..body.stand(cargo_kg) };
    let r = kin::solve_fast(body, &pose, &KinLoads { accel: [accel, 0.0, 0.0], bumper_force: [-push_n, 0.0, 0.0], ..KinLoads::default() });
    let (front, rear, pressure) = match (r.contact("axle_front"), r.contact("axle_rear"), r.contact("track")) {
        (Some(f), Some(b), _) => (f.normal_n, b.normal_n, None),
        (_, _, Some(t)) => {
            let p = t.pressure_kpa.unwrap_or([0.0, 0.0]);
            (p[1], p[0], Some(p))
        }
        _ => (0.0, 0.0, None),
    };
    DriveState { need_n: need, limit_n: limit, slip, speed_factor: 1.0 - slip, front_n: front, rear_n: rear, pressure_kpa: pressure, margin: r.margin }
}

pub fn drive(body: &Body, ground: Surface, slope_deg: f32, cargo_kg: f32, speed: f32, dist_m: f32, push_n: f32) -> ActRollout {
    let props = ground.props();
    let spec = body.base.spec();
    let a_up = body.planned_accel(&spec, slope_deg, cargo_kg, ground, 1.0, 0.5);
    let a_dn = body.planned_accel(&spec, slope_deg, cargo_kg, ground, -1.0, 0.5);
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let (mut x, mut v, mut t) = (0.0f32, 0.0f32, 0.0f32);
    let mut slip_max = 0.0f32;
    let mut time_lost = 0.0f32;
    let mut outcome = Outcome::ok();
    let mut announced = false;
    while x < dist_m && t < 600.0 {
        let left = dist_m - x;
        let brake = v * v / (2.0 * a_dn.max(0.05));
        let a = if left <= brake { -a_dn } else if v < speed { a_up } else { 0.0 };
        let st = state(body, props, slope_deg, a, cargo_kg, push_n);
        if st.slip >= 0.999 {
            events.push(ActEvent::new(t, "stuck", format!("헛돎: 필요 {:.0} N > 지면 한계 {:.0} N", st.need_n, st.limit_n), &[Part::Wheel, Part::Track], Some(st.need_n / st.limit_n.max(1e-3)), false));
            outcome = Outcome::fail("stuck", format!("구동부가 헛돎: 필요 {:.0} N > {} 한계 {:.0} N", st.need_n, props.label_ko, st.limit_n));
            break;
        }
        if st.slip > 0.2 && !announced {
            announced = true;
            events.push(ActEvent::new(t, "wheel_slip", format!("미끄러짐률 {:.0}% · 필요 {:.0} N / 한계 {:.0} N", st.slip * 100.0, st.need_n, st.limit_n), &[Part::Wheel, Part::Track], Some(st.slip), true));
        }
        slip_max = slip_max.max(st.slip);
        let vg = v * st.speed_factor;
        let dt = SAMPLE_S;
        x += vg * dt;
        time_lost += dt * st.slip;
        v = (v + a * dt).clamp(0.0, speed);
        if a < 0.0 && v <= 1e-3 {
            v = 0.02;
        }
        let pose = KinPose { base_pitch_deg: slope_deg, ..body.stand(cargo_kg) };
        samples.push(ActSample { t, phase: if a > 0.0 { "start" } else if a < 0.0 { "stop" } else { "cruise" }, pose, loads: KinLoads { accel: [a, 0.0, 0.0], bumper_force: [-push_n, 0.0, 0.0], ..KinLoads::default() }, margin: st.margin, zmp: [0.0, 0.0], com: [x, 0.0, 0.0], peak_ratio: st.need_n / (2.0 * body.chassis.drive_force_n).max(1e-3), grip: None, object: None });
        t += dt;
    }
    let cruise = state(body, props, slope_deg, 0.0, cargo_kg, push_n);
    let metrics = vec![
        ("slip_cruise", cruise.slip),
        ("slip_max", slip_max),
        ("speed_factor", cruise.speed_factor),
        ("time_lost_s", time_lost),
        ("need_over_limit", cruise.need_n / cruise.limit_n.max(1e-3)),
        ("front_share", cruise.front_n / (cruise.front_n + cruise.rear_n).max(1e-3)),
        ("duration_s", t),
    ];
    ActRollout { kind: ActKind::Drive, samples, events, outcome, duration_s: t, metrics, notes: vec![] }
}

pub fn curb_impact(body: &Body, cargo_kg: f32, speed: f32, front_share: f32, step_m: f32) -> (f32, f32) {
    let m = (body.mass_kg() + cargo_kg) * front_share.clamp(0.1, 1.0);
    let c = ((WHEEL_R_M - step_m.clamp(0.0, WHEEL_R_M)) / WHEEL_R_M).clamp(0.0, 1.0);
    let v = speed * (1.0 - c * c).sqrt();
    super::material::impact(v, m, TYRE_STIFFNESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    #[test]
    fn thrust_rises_with_slip_and_sand_slips_more_than_concrete() {
        assert!(thrust_fraction(0.05, 0.03, 0.025) < thrust_fraction(0.3, 0.03, 0.025));
        let firm = slip_for(0.3, 0.02, shear_modulus_m(Surface::Firm)).unwrap();
        let sand = slip_for(0.3, 0.05, shear_modulus_m(Surface::Sand)).unwrap();
        assert!(sand > firm, "{sand} vs {firm}");
        assert!(slip_for(1.2, 0.05, 0.025).is_none(), "past the friction limit it spins");
    }

    #[test]
    fn a_firm_face_gives_its_measured_peak_by_the_peak_slip_and_sand_falls_short() {
        for g in [Surface::Firm, Surface::Gravel, Surface::Dirt, Surface::Grass] {
            let l = wheel_contact_m(g);
            assert!((traction_fraction(g, PEAK_SLIP, l) - 1.0).abs() < 1e-5, "{g:?}");
            assert!(traction_fraction(g, 0.5 * PEAK_SLIP, l) < 1.0, "{g:?}: the traction rises with the slip");
        }
        let cap = traction_fraction(Surface::Sand, 1.0, wheel_contact_m(Surface::Sand));
        assert!(cap > 0.7 && cap < 0.85, "{cap}");
        assert!(slip_on(Surface::Sand, 0.9, wheel_contact_m(Surface::Sand)).is_none());
    }

    #[test]
    fn a_drive_on_sand_loses_time_to_slip_and_a_reach_loads_the_front() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::Wheel, FingerCount::Five);
        let firm = drive(&b, Surface::Firm, 0.0, 0.0, 1.0, 5.0, 0.0);
        let dirt = drive(&b, Surface::Dirt, 0.0, 0.0, 1.0, 5.0, 0.0);
        assert!(firm.outcome.ok && dirt.outcome.ok, "{:?} {:?}", firm.outcome, dirt.outcome);
        assert!(dirt.metric("slip_cruise").unwrap() > firm.metric("slip_cruise").unwrap());
        assert!(dirt.duration_s > firm.duration_s);
        let st = state(&b, Surface::Firm.props(), 0.0, -1.0, 0.0, 0.0);
        assert!(st.front_n > st.rear_n, "braking loads the front axle: {st:?}");
    }
}
