use serde::{Deserialize, Serialize};

use crate::camera::{wrap_angle, Pose};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bearing {
    pub x: f32,
    pub z: f32,
    pub heading: f32,
    pub sigma: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fix {
    pub x: f32,
    pub z: f32,
    pub range_m: f32,
    pub parallax_deg: f32,
    pub baseline_m: f32,
    pub sigma_m: f32,
    pub residual_m: f32,
    pub rays: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum TriError {
    TooFewRays,
    LowParallax { parallax_deg: f32, need_deg: f32 },
    Behind,
}

impl TriError {
    pub fn code(&self) -> &'static str {
        match self {
            TriError::TooFewRays => "too_few_rays",
            TriError::LowParallax { .. } => "low_parallax",
            TriError::Behind => "behind",
        }
    }
}

fn dir(h: f32) -> (f32, f32) {
    (h.cos(), h.sin())
}

pub fn triangulate(rays: &[Bearing], min_parallax_deg: f32) -> Result<Fix, TriError> {
    if rays.len() < 2 {
        return Err(TriError::TooFewRays);
    }
    let mut parallax = 0.0f32;
    let mut baseline = 0.0f32;
    for i in 0..rays.len() {
        for j in i + 1..rays.len() {
            parallax = parallax.max(wrap_angle(rays[i].heading - rays[j].heading).abs());
            baseline = baseline.max((rays[i].x - rays[j].x).hypot(rays[i].z - rays[j].z));
        }
    }
    let parallax_deg = parallax.to_degrees();
    if parallax_deg < min_parallax_deg {
        return Err(TriError::LowParallax { parallax_deg, need_deg: min_parallax_deg });
    }
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for r in rays {
        let w = 1.0 / (r.sigma.max(1e-4) as f64).powi(2);
        let (ux, uz) = dir(r.heading);
        let (ux, uz) = (ux as f64, uz as f64);
        let (m11, m12, m22) = (1.0 - ux * ux, -ux * uz, 1.0 - uz * uz);
        a11 += w * m11;
        a12 += w * m12;
        a22 += w * m22;
        b1 += w * (m11 * r.x as f64 + m12 * r.z as f64);
        b2 += w * (m12 * r.x as f64 + m22 * r.z as f64);
    }
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-12 {
        return Err(TriError::LowParallax { parallax_deg, need_deg: min_parallax_deg });
    }
    let x = ((a22 * b1 - a12 * b2) / det) as f32;
    let z = ((a11 * b2 - a12 * b1) / det) as f32;
    let mut sq = 0.0f32;
    for r in rays {
        let (ux, uz) = dir(r.heading);
        let (dx, dz) = (x - r.x, z - r.z);
        if dx * ux + dz * uz <= 0.0 {
            return Err(TriError::Behind);
        }
        let perp = dx * uz - dz * ux;
        sq += perp * perp;
    }
    let last = rays.last().expect("checked above");
    let first = rays.first().expect("checked above");
    let range = (x - last.x).hypot(z - last.z);
    let range0 = (x - first.x).hypot(z - first.z);
    let sigma = (range0 * first.sigma).hypot(range * last.sigma) / parallax.sin().max(1e-3);
    Ok(Fix { x, z, range_m: range, parallax_deg, baseline_m: baseline, sigma_m: sigma, residual_m: (sq / rays.len() as f32).sqrt(), rays: rays.len() })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriCfg {
    pub min_parallax_deg: f32,
    pub plan_margin: f32,
    pub min_step_m: f32,
    pub max_step_m: f32,
    pub angled_offset_deg: f32,
}

impl Default for TriCfg {
    fn default() -> Self {
        TriCfg { min_parallax_deg: 6.0, plan_margin: 1.4, min_step_m: 0.5, max_step_m: 2.5, angled_offset_deg: 55.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TriStep {
    pub kind: String,
    pub to: (f32, f32),
    pub step_m: f32,
    pub baseline_perp_m: f32,
    pub expected_parallax_deg: f32,
    pub face: (f32, f32),
}

pub fn plan_step(pose: &Pose, heading: f32, range_guess_m: f32, holonomic: bool, side: f32, cfg: &TriCfg) -> TriStep {
    let range = range_guess_m.clamp(1.0, 30.0);
    let need = (cfg.min_parallax_deg * cfg.plan_margin).to_radians();
    let perp = range * need.tan();
    let s = side.signum();
    let (kind, dir_h, step) = if holonomic {
        ("strafe", heading + s * std::f32::consts::FRAC_PI_2, perp)
    } else {
        let off = cfg.angled_offset_deg.to_radians();
        ("angled", heading + s * off, perp / off.sin())
    };
    let step = step.clamp(cfg.min_step_m, cfg.max_step_m);
    let to = (pose.x + step * dir_h.cos(), pose.z + step * dir_h.sin());
    let target = (pose.x + range * heading.cos(), pose.z + range * heading.sin());
    let h2 = (target.1 - to.1).atan2(target.0 - to.0);
    let along = (dir_h - heading).sin().abs() * step;
    TriStep {
        kind: kind.into(),
        to,
        step_m: step,
        baseline_perp_m: along,
        expected_parallax_deg: wrap_angle(h2 - heading).abs().to_degrees(),
        face: target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sight(from: (f32, f32), target: (f32, f32), err_rad: f32) -> Bearing {
        Bearing { x: from.0, z: from.1, heading: (target.1 - from.1).atan2(target.0 - from.0) + err_rad, sigma: 0.007 }
    }

    #[test]
    fn two_exact_bearings_meet_at_the_target() {
        let t = (8.0, 3.0);
        let f = triangulate(&[sight((0.0, 0.0), t, 0.0), sight((0.5, 1.2), t, 0.0)], 3.0).unwrap();
        assert!((f.x - 8.0).abs() < 1e-3 && (f.z - 3.0).abs() < 1e-3, "{f:?}");
        assert!((f.range_m - (7.5f32).hypot(1.8)).abs() < 1e-3);
        assert!(f.residual_m < 1e-3);
    }

    #[test]
    fn parallel_rays_are_refused_and_more_parallax_means_a_smaller_error() {
        let t = (10.0, 0.0);
        let low = triangulate(&[sight((0.0, 0.0), t, 0.0), sight((0.0, 0.2), t, 0.0)], 6.0);
        assert!(matches!(low, Err(TriError::LowParallax { .. })), "{low:?}");
        let narrow = triangulate(&[sight((0.0, 0.0), t, 0.0), sight((0.0, 1.2), t, 0.0)], 6.0).unwrap();
        let wide = triangulate(&[sight((0.0, 0.0), t, 0.0), sight((0.0, 2.5), t, 0.0)], 6.0).unwrap();
        assert!(wide.sigma_m < narrow.sigma_m, "{} vs {}", wide.sigma_m, narrow.sigma_m);
        let bent = triangulate(&[sight((0.0, 0.0), t, 0.007), sight((0.0, 1.2), t, 0.0)], 6.0).unwrap();
        let moved = (bent.range_m - narrow.range_m).abs();
        assert!(moved < 2.0 * narrow.sigma_m && moved > 0.2 * narrow.sigma_m, "moved {moved}, sigma {}", narrow.sigma_m);
    }

    #[test]
    fn rays_that_meet_behind_are_refused() {
        let a = Bearing { x: 0.0, z: 0.0, heading: 0.3, sigma: 0.01 };
        let b = Bearing { x: 0.0, z: 1.0, heading: -0.3 + std::f32::consts::PI, sigma: 0.01 };
        assert!(matches!(triangulate(&[a, b], 3.0), Err(TriError::Behind)));
    }

    #[test]
    fn the_planned_step_gives_the_parallax_it_promises() {
        let pose = Pose { x: 0.0, z: 0.0, yaw: 0.0, eye_m: 0.5 };
        let cfg = TriCfg::default();
        for holonomic in [true, false] {
            let p = plan_step(&pose, 0.0, 9.0, holonomic, 1.0, &cfg);
            assert!(p.expected_parallax_deg >= cfg.min_parallax_deg, "{p:?}");
            assert!(p.step_m <= cfg.max_step_m);
            let fix = triangulate(&[sight((0.0, 0.0), (9.0, 0.0), 0.0), sight(p.to, (9.0, 0.0), 0.0)], cfg.min_parallax_deg).unwrap();
            assert!((fix.x - 9.0).abs() < 1e-2, "{holonomic}: {fix:?}");
        }
        let angled = plan_step(&pose, 0.0, 9.0, false, -1.0, &cfg);
        assert_eq!(angled.kind, "angled");
        assert!(angled.to.1 < 0.0, "left of +x is -z");
    }
}
