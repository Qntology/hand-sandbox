use serde::Serialize;

use crate::kin::{KinLoads, KinPose};

pub mod cocarry;
pub mod collide;
pub mod drive;
pub mod fall;
pub mod focus;
pub mod fork;
pub mod gait;
pub mod gaze;
pub mod grasp;
pub mod material;
pub mod push;
pub mod reach;

pub const SAMPLE_S: f32 = 0.02;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActKind {
    Grasp,
    Release,
    Push,
    Walk,
    Drive,
    Bend,
    Look,
    Fork,
}

impl ActKind {
    pub const ALL: [ActKind; 8] = [ActKind::Grasp, ActKind::Release, ActKind::Push, ActKind::Walk, ActKind::Drive, ActKind::Bend, ActKind::Look, ActKind::Fork];

    pub fn id(self) -> &'static str {
        match self {
            ActKind::Grasp => "grasp",
            ActKind::Release => "release",
            ActKind::Push => "push",
            ActKind::Walk => "walk",
            ActKind::Drive => "drive",
            ActKind::Bend => "bend",
            ActKind::Look => "look",
            ActKind::Fork => "fork",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            ActKind::Grasp => "잡기",
            ActKind::Release => "놓기",
            ActKind::Push => "밀기",
            ActKind::Walk => "걷기",
            ActKind::Drive => "주행",
            ActKind::Bend => "굽히기",
            ActKind::Look => "응시",
            ActKind::Fork => "지게차",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Hand,
    HandJoints,
    Foot,
    FootJoints,
    Waist,
    Neck,
    Wheel,
    Track,
}

impl Part {
    pub const ALL: [Part; 8] = [Part::Hand, Part::HandJoints, Part::Foot, Part::FootJoints, Part::Waist, Part::Neck, Part::Wheel, Part::Track];

    pub fn id(self) -> &'static str {
        match self {
            Part::Hand => "hand",
            Part::HandJoints => "hand_joints",
            Part::Foot => "foot",
            Part::FootJoints => "foot_joints",
            Part::Waist => "waist",
            Part::Neck => "neck",
            Part::Wheel => "wheel",
            Part::Track => "track",
        }
    }

    pub fn label_ko(self) -> &'static str {
        match self {
            Part::Hand => "손 모양",
            Part::HandJoints => "손 관절",
            Part::Foot => "발 모양",
            Part::FootJoints => "발 관절",
            Part::Waist => "허리",
            Part::Neck => "목 관절",
            Part::Wheel => "바퀴",
            Part::Track => "트랙",
        }
    }

    pub fn from_id(id: &str) -> Option<Part> {
        Part::ALL.into_iter().find(|p| p.id() == id)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActEvent {
    pub t: f32,
    pub kind: &'static str,
    pub label_ko: String,
    pub parts: Vec<Part>,
    pub value: Option<f32>,
    pub ok: bool,
}

impl ActEvent {
    pub fn new(t: f32, kind: &'static str, label_ko: impl Into<String>, parts: &[Part], value: Option<f32>, ok: bool) -> ActEvent {
        ActEvent { t, kind, label_ko: label_ko.into(), parts: parts.to_vec(), value, ok }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Grip {
    pub grip_n: f32,
    pub load_n: f32,
    pub mu: f32,
    pub mu_est: f32,
    pub slip_ratio: f32,
    pub twist_margin_nm: f32,
    pub contacts: u8,
    pub closure: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<crate::contact::FingerContacts>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActSample {
    pub t: f32,
    pub phase: &'static str,
    pub pose: KinPose,
    pub loads: KinLoads,
    pub margin: f32,
    pub zmp: [f32; 2],
    pub com: [f32; 3],
    pub peak_ratio: f32,
    pub grip: Option<Grip>,
    pub object: Option<[f32; 3]>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Outcome {
    pub ok: bool,
    pub code: Option<&'static str>,
    pub reason_ko: Option<String>,
}

impl Outcome {
    pub fn ok() -> Outcome {
        Outcome { ok: true, code: None, reason_ko: None }
    }

    pub fn fail(code: &'static str, reason_ko: impl Into<String>) -> Outcome {
        Outcome { ok: false, code: Some(code), reason_ko: Some(reason_ko.into()) }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActRollout {
    pub kind: ActKind,
    pub samples: Vec<ActSample>,
    pub events: Vec<ActEvent>,
    pub outcome: Outcome,
    pub duration_s: f32,
    pub metrics: Vec<(&'static str, f32)>,
    pub notes: Vec<String>,
}

impl ActRollout {
    pub fn sample_at(&self, t: f32) -> Option<&ActSample> {
        let i = self.samples.partition_point(|s| s.t <= t);
        self.samples.get(i.saturating_sub(1))
    }

    pub fn metric(&self, key: &str) -> Option<f32> {
        self.metrics.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }

    pub fn events_between(&self, from: f32, to: f32) -> impl Iterator<Item = &ActEvent> {
        self.events.iter().filter(move |e| e.t > from && e.t <= to)
    }
}

#[derive(Clone, Debug)]
pub struct Draw(u64);

impl Draw {
    pub fn new(seed: u64) -> Draw {
        Draw(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn lognormal(&mut self, mean: f32, cov: f32) -> f32 {
        if cov <= 0.0 {
            return mean;
        }
        let s2 = (1.0 + cov * cov).ln();
        let mu = mean.ln() - 0.5 * s2;
        (mu + s2.sqrt() * self.normal()).exp()
    }
}

pub fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    s * y as f32
}

pub fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

pub fn lognormal_above(mean: f32, cov: f32, need: f32) -> f32 {
    if need <= 0.0 {
        return 1.0;
    }
    if cov <= 0.0 {
        return if mean >= need { 1.0 } else { 0.0 };
    }
    let s2 = (1.0 + cov * cov).ln();
    let mu = mean.ln() - 0.5 * s2;
    1.0 - normal_cdf((need.ln() - mu) / s2.sqrt())
}

pub fn min_jerk(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    let u3 = u * u * u;
    10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u * u
}

pub const MIN_JERK_PEAK: f32 = 1.875;

pub fn locomotion_factor(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32) -> Option<f32> {
    locomotion_factor_with(spec, ground, slope_deg, 0.0)
}

pub fn locomotion_factor_with(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32, extra_n_per_kg: f32) -> Option<f32> {
    use crate::base::BaseKind;
    use crate::balance::G;
    match spec.kind {
        BaseKind::HumanFoot => {
            let Some(z) = spec.walk_com_z_m else { return Some(1.0) };
            let g = gait::plan_z(z, spec.walk_foot_m.unwrap_or(gait::NOMINAL_FOOT_M), spec.max_speed_mps, crate::surface::available_mu(spec.kind, ground), slope_deg.max(0.0));
            Some((g.speed / g.speed_cmd.max(1e-3)).clamp(0.05, 1.0))
        }
        _ => {
            let th = slope_deg.to_radians();
            let mu = crate::surface::available_mu(spec.kind, ground);
            let per_kg = G * (crate::surface::rolling_resistance(spec.kind, ground) * th.cos() + th.sin()) + extra_n_per_kg;
            if extra_n_per_kg > 0.0 && spec.drive_accel_mps2.is_some_and(|a| per_kg > a) {
                return None;
            }
            let need = per_kg / (G * mu * th.cos()).max(1e-3);
            let gs = crate::surface::Surface::from_id(ground.id).unwrap_or_default();
            let l = if spec.kind == BaseKind::Track { spec.footprint_m } else { drive::wheel_contact_m(gs) };
            drive::slip_on(gs, need, l).map(|s| (1.0 - s).max(0.02))
        }
    }
}

pub fn cruise_slip(spec: &crate::base::BaseSpec, ground: &crate::surface::SurfaceProps, slope_deg: f32) -> Option<f32> {
    use crate::balance::G;
    use crate::base::BaseKind;
    if spec.kind == BaseKind::HumanFoot {
        return None;
    }
    let th = slope_deg.max(0.0).to_radians();
    let mu = crate::surface::available_mu(spec.kind, ground);
    let per_kg = G * (crate::surface::rolling_resistance(spec.kind, ground) * th.cos() + th.sin());
    let need = per_kg / (G * mu * th.cos()).max(1e-3);
    let gs = crate::surface::Surface::from_id(ground.id).unwrap_or_default();
    let l = if spec.kind == BaseKind::Track { spec.footprint_m } else { drive::wheel_contact_m(gs) };
    drive::slip_on(gs, need, l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_repeat_per_seed_and_have_the_asked_moments() {
        let mut a = Draw::new(7);
        let mut b = Draw::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let mut d = Draw::new(11);
        let xs: Vec<f32> = (0..20000).map(|_| d.lognormal(0.6, 0.15)).collect();
        let m = xs.iter().sum::<f32>() / xs.len() as f32;
        let sd = (xs.iter().map(|x| (x - m).powi(2)).sum::<f32>() / xs.len() as f32).sqrt();
        assert!((m - 0.6).abs() < 0.01 && (sd / m - 0.15).abs() < 0.01, "{m} {sd}");
        assert!(xs.iter().all(|x| *x > 0.0));
        assert!((min_jerk(0.5) - 0.5).abs() < 1e-6 && min_jerk(1.0) == 1.0);
    }
}
