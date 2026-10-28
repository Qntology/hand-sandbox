use serde::Serialize;

use crate::balance::{Body, G};
use crate::kin::{KinLoads, KinPose, BUMPER_GAP_M, WHEEL_R_M};
use crate::surface::Surface;

use super::drive;
use super::material::{self, ItemMaterial};
use super::{ActEvent, ActKind, ActRollout, ActSample, Draw, Outcome, Part, SAMPLE_S};

pub const APPROACH_MPS: f32 = 0.10;
pub const BUMPER_STIFFNESS: f32 = 5.0e4;
pub const PUSH_MPS: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PushLeg {
    pub turn_deg: f32,
    pub radius_m: f32,
    pub len_m: f32,
    pub ground: Surface,
    pub slope_deg: f32,
    pub blocked: bool,
}

#[derive(Clone, Debug)]
pub struct PushInput<'a> {
    pub body: &'a Body,
    pub material: ItemMaterial,
    pub mass_kg: f32,
    pub size_m: [f32; 3],
    pub cargo_kg: f32,
    pub legs: Vec<PushLeg>,
    pub aim_sigma_m: f32,
    pub seed: u64,
}

pub const PUSH_RADIUS_MARGIN: f32 = 1.25;

pub fn limit_surface_c(side_m: f32) -> f32 {
    side_m * (2f32.sqrt() + (1.0 + 2f32.sqrt()).ln()) / 6.0
}

pub fn stable_radii_m(side_m: f32, mu_c: f32, lead_m: f32) -> (f32, f32) {
    let c = limit_surface_c(side_m);
    let d = 0.5 * side_m;
    (lead_m / mu_c.max(0.05), (c * c + side_m * lead_m * 0.5) / d)
}

pub fn stable_radius_m(side_m: f32, mu_c: f32, lead_m: f32) -> f32 {
    let (f, m) = stable_radii_m(side_m, mu_c, lead_m);
    f.max(m)
}

pub fn lead_m(body: &Body, side_m: f32) -> f32 {
    0.5 * body.support_len_m + WHEEL_R_M + BUMPER_GAP_M + 0.5 * side_m
}

pub fn planned_radius_m(body: &Body, material: ItemMaterial, side_m: f32) -> f32 {
    PUSH_RADIUS_MARGIN * stable_radius_m(side_m, material::pad_on(material).mean, lead_m(body, side_m))
}

pub fn push(inp: &PushInput) -> ActRollout {
    let body = inp.body;
    let mut d = Draw::new(inp.seed);
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let mut metrics: Vec<(&'static str, f32)> = Vec::new();
    let mut notes = Vec::new();
    if body.legged() {
        let out = Outcome::fail("push_legs", "다리형은 범퍼가 없어 바닥의 물체를 밀지 않음 (들어서 옮김)");
        return ActRollout { kind: ActKind::Push, samples, events, outcome: out, duration_s: 0.0, metrics, notes };
    }
    let side = inp.size_m[0];
    let mu_c = d.lognormal(material::pad_on(inp.material).mean, material::pad_on(inp.material).cov);
    let lead = lead_m(body, side);
    let (r_fric, r_face) = stable_radii_m(side, mu_c, lead);
    let r_stable = r_fric.max(r_face);
    let half_room = (0.5 * body.support_width_m - 0.5 * side).max(0.0);
    let mut y_off = inp.aim_sigma_m * d.normal();
    let bumper_x = lead - 0.5 * side;
    let mut t = 0.0f32;
    let mut pushed = 0.0f32;
    let mut outcome = Outcome::ok();
    let mut slide_total = 0.0f32;
    let mut f_peak = 0.0f32;
    let mut slip_max = 0.0f32;
    let (f_imp, dt_imp) = material::impact(APPROACH_MPS, inp.mass_kg, BUMPER_STIFFNESS);
    events.push(ActEvent::new(0.0, "contact", format!("범퍼 접촉 · 충돌 {:.0} N ({:.0} ms) · 범퍼 위 위치 {:+.3} m", f_imp, dt_imp * 1000.0, y_off), &[Part::Wheel, Part::Track], Some(f_imp), true));
    metrics.push(("contact_impact_n", f_imp));
    metrics.push(("stable_radius_m", r_stable));
    metrics.push(("mu_contact", mu_c));
    let mut first = true;
    if y_off.abs() > half_room {
        events.push(ActEvent::new(0.0, "push_lost", format!("범퍼 끝에 걸쳐 밀지 못함 (어긋남 {:.2} m > 여유 {:.2} m)", y_off.abs(), half_room), &[Part::Wheel, Part::Track], Some(y_off.abs()), false));
        metrics.push(("offset_m", y_off));
        let out = Outcome::fail("push_lost", format!("물체가 범퍼 가운데에서 {:.2} m 벗어나 밀리지 않음", y_off.abs()));
        return ActRollout { kind: ActKind::Push, samples, events, outcome: out, duration_s: 0.0, metrics, notes };
    }
    'legs: for (li, leg) in inp.legs.iter().enumerate() {
        let (fs, fk) = material::item_on(inp.material, leg.ground);
        let mu_s = d.lognormal(fs.mean, fs.cov);
        let mu_k = d.lognormal(fk.mean, fk.cov).min(mu_s);
        let th = leg.slope_deg.to_radians();
        let w = inp.mass_kg * G;
        let f_static = w * (mu_s * th.cos() + th.sin());
        let f_slide = w * (mu_k * th.cos() + th.sin());
        if leg.turn_deg.abs() > 1.0 {
            let theta = leg.turn_deg.abs().to_radians();
            let r = leg.radius_m;
            let turn_t = theta * r.max(0.5 * body.support_width_m) / PUSH_MPS;
            if r < r_face {
                t += 0.5 * turn_t;
                events.push(ActEvent::new(t, "push_lost", format!("{:.0}° 회전 반경 {:.2} m < 면 접촉 한계 {:.2} m: 물체가 모서리로 돌아 빠짐", leg.turn_deg, r, r_face), &[Part::Wheel, Part::Track], Some(r), false));
                outcome = Outcome::fail("push_lost", format!("{}번째 구간 회전(반경 {:.2} m)에서 물체가 범퍼 면을 벗어나 돎 (한계 {:.2} m)", li + 1, r, r_face));
                break 'legs;
            }
            if r < r_fric {
                let slide = theta * (lead - mu_c * r).max(0.0);
                y_off += slide * leg.turn_deg.signum();
                slide_total += slide;
                events.push(ActEvent::new(t, "turn_slide", format!("{:.0}° 회전 (반경 {:.2} m < 마찰 한계 {:.2} m) · 범퍼 위로 {:.3} m 밀림", leg.turn_deg, r, r_fric, slide), &[Part::Wheel, Part::Track], Some(slide), y_off.abs() <= half_room));
            }
            t += turn_t;
            if y_off.abs() > half_room {
                events.push(ActEvent::new(t, "push_lost", format!("물체가 범퍼 끝으로 빠짐 (어긋남 {:.2} m > 여유 {:.2} m)", y_off.abs(), half_room), &[Part::Wheel, Part::Track], Some(y_off.abs()), false));
                outcome = Outcome::fail("push_lost", format!("{}번째 구간 회전에서 물체가 범퍼 끝으로 빠짐 (안정 반경 {:.2} m, 실제 {:.2} m)", li + 1, r_stable, r));
                break 'legs;
            }
        }
        let need_static = drive::state(body, leg.ground.props(), leg.slope_deg, 0.0, inp.cargo_kg, f_static);
        if first || leg.slope_deg.abs() > 0.5 {
            events.push(ActEvent::new(t, "breakaway", format!("정지 마찰 {mu_s:.2} → 미는 힘 {:.0} N", f_static), &[Part::Wheel, Part::Track], Some(f_static), true));
            first = false;
        }
        if need_static.slip >= 0.999 {
            events.push(ActEvent::new(t, "stuck", format!("밀지 못함: 필요 {:.0} N > 지면 한계 {:.0} N", need_static.need_n, need_static.limit_n), &[Part::Wheel, Part::Track], Some(need_static.need_n), false));
            outcome = Outcome::fail("push_stuck", format!("{} 위에서 {:.0} N을 밀지 못해 헛돎 (한계 {:.0} N)", leg.ground.label_ko(), need_static.need_n, need_static.limit_n));
            break 'legs;
        }
        f_peak = f_peak.max(f_static);
        let st = drive::state(body, leg.ground.props(), leg.slope_deg, 0.0, inp.cargo_kg, f_slide);
        slip_max = slip_max.max(st.slip);
        let v = PUSH_MPS * st.speed_factor;
        let n = (leg.len_m / (v * SAMPLE_S)).ceil().max(1.0) as usize;
        for i in 0..n {
            let pose = KinPose { base_pitch_deg: leg.slope_deg, ..body.stand(inp.cargo_kg) };
            samples.push(ActSample { t: t + i as f32 * SAMPLE_S, phase: "push", pose, loads: KinLoads { bumper_force: [-f_slide, 0.0, 0.0], ..KinLoads::default() }, margin: st.margin, zmp: [0.0, 0.0], com: [pushed + i as f32 * v * SAMPLE_S, 0.0, 0.0], peak_ratio: st.need_n / (2.0 * body.chassis.drive_force_n).max(1e-3), grip: None, object: Some([bumper_x + 0.5 * side, y_off, 0.5 * inp.size_m[1]]) });
        }
        t += leg.len_m / v.max(1e-3);
        pushed += leg.len_m;
        if leg.blocked {
            events.push(ActEvent::new(t, "collision", "물체가 장애물에 걸려 멈춤".to_string(), &[Part::Wheel, Part::Track], None, false));
            outcome = Outcome::fail("push_blocked", "밀던 물체가 장애물(턱·노드)에 걸림");
            break 'legs;
        }
    }
    if outcome.ok {
        events.push(ActEvent::new(t, "arrive", format!("{:.1} m 밀어 옮김", pushed), &[Part::Wheel, Part::Track], Some(pushed), true));
    }
    if slide_total > 0.0 {
        notes.push(format!("회전에서 범퍼 위로 합계 {:.2} m 밀림", slide_total));
    }
    metrics.push(("push_peak_n", f_peak));
    metrics.push(("pushed_m", pushed));
    metrics.push(("slide_m", slide_total));
    metrics.push(("offset_m", y_off));
    metrics.push(("slip_max", slip_max));
    metrics.push(("duration_s", t));
    ActRollout { kind: ActKind::Push, samples, events, outcome, duration_s: t, metrics, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::BaseKind;
    use crate::hand::FingerCount;
    use crate::parts::{BodyPlan, Tier};

    fn leg(turn: f32, r: f32, len: f32, g: Surface) -> PushLeg {
        PushLeg { turn_deg: turn, radius_m: r, len_m: len, ground: g, slope_deg: 0.0, blocked: false }
    }

    #[test]
    fn a_straight_push_arrives_and_a_tight_turn_can_lose_the_object() {
        let b = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::Wheel, FingerCount::Five);
        let straight = push(&PushInput { body: &b, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], cargo_kg: 0.0, legs: vec![leg(0.0, 0.0, 3.0, Surface::Gravel)], aim_sigma_m: 0.0, seed: 1 });
        assert!(straight.outcome.ok, "{:?}", straight.outcome);
        assert!(straight.metric("push_peak_n").unwrap() > 2.6 * G * 0.3);
        let spin = push(&PushInput { body: &b, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], cargo_kg: 0.0, legs: vec![leg(0.0, 0.0, 1.0, Surface::Gravel), leg(90.0, 0.0, 1.0, Surface::Gravel)], aim_sigma_m: 0.0, seed: 2 });
        assert_eq!(spin.outcome.code, Some("push_lost"), "{:?}", spin.events);
        let r = planned_radius_m(&b, ItemMaterial::Stone, 0.1);
        assert!(r > 0.3 && r < 1.0, "{r}");
        let kept = (0..50).filter(|s| push(&PushInput { body: &b, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], cargo_kg: 0.0, legs: vec![leg(45.0, r, 1.0, Surface::Gravel), leg(-45.0, r, 1.0, Surface::Gravel)], aim_sigma_m: 0.01, seed: *s }).outcome.ok).count();
        assert!(kept >= 45, "{kept}/50");
        let l = Body::new(&BodyPlan::uniform(Tier::Mid), BaseKind::HumanFoot, FingerCount::Five);
        assert_eq!(push(&PushInput { body: &l, material: ItemMaterial::Stone, mass_kg: 2.6, size_m: [0.1; 3], cargo_kg: 0.0, legs: vec![], aim_sigma_m: 0.0, seed: 1 }).outcome.code, Some("push_legs"));
    }
}
