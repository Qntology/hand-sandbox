use serde::{Deserialize, Serialize};

use crate::parts::Actuator;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reducer {
    Spur,
    Planetary,
    Cycloid,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct LifeRating {
    pub torque_nm: f32,
    pub hours: f32,
    pub exponent: f32,
    pub cite: &'static str,
    pub stand_in: bool,
}

pub const SERVO_TEST_CYCLES: f32 = 100_000.0;
pub const SERVO_TEST_LOAD_FRAC: f32 = 0.2;
pub const STRAIN_WAVE_L10_H: f32 = 7_000.0;
pub const CYCLOID_K_H: f32 = 6_000.0;

pub fn rating(a: &Actuator) -> LifeRating {
    match a.reducer {
        Reducer::Spur => {
            let cycle_s = 2.0 * 120f32.to_radians() / a.speed_rad_s.max(0.1);
            LifeRating {
                torque_nm: SERVO_TEST_LOAD_FRAC * a.peak_nm,
                hours: SERVO_TEST_CYCLES * cycle_s / 3600.0,
                exponent: 3.0,
                cite: "STS3215 수명 시험 10만 회 (±60°, 정지 하중 1/5, Feetech 2020) · 주기는 무부하 속도로 유도 · 3승 법칙은 볼 베어링 식(ISO 281) 대입",
                stand_in: a.name != "Feetech STS3215 (12 V)",
            }
        }
        Reducer::Planetary => LifeRating {
            torque_nm: a.cont_nm,
            hours: STRAIN_WAVE_L10_H,
            exponent: 3.0,
            cite: "유성 감속 QDD의 수명 식 미발견 → Harmonic Drive CSF L10 7,000 h (정격 = 연속 토크) · 3승 법칙 대입",
            stand_in: true,
        },
        Reducer::Cycloid => LifeRating {
            torque_nm: a.cont_nm,
            hours: CYCLOID_K_H,
            exponent: 10.0 / 3.0,
            cite: "Nabtesco RV-N 정격 수명 6,000 h, 지수 10/3 (사이클로이드 계열)",
            stand_in: false,
        },
    }
}

pub fn life_h(a: &Actuator, torque_nm: f32) -> f32 {
    let r = rating(a);
    let t = torque_nm.abs();
    if t < 1e-6 {
        return f32::INFINITY;
    }
    r.hours * (r.torque_nm / t).powf(r.exponent)
}

pub fn damage(a: &Actuator, torque_nm: f32, dt_s: f32) -> f32 {
    let l = life_h(a, torque_nm);
    if l.is_finite() {
        dt_s / (3600.0 * l)
    } else {
        0.0
    }
}

pub fn cubic_mean_torque(cycle: &[(f32, f32, f32)]) -> f32 {
    let (num, den) = cycle.iter().fold((0.0f64, 0.0f64), |a, &(t, n, s)| (a.0 + (n * s) as f64 * (t.abs() as f64).powi(3), a.1 + (n * s) as f64));
    if den <= 0.0 {
        0.0
    } else {
        (num / den).cbrt() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::{RS02, STS3215};

    #[test]
    fn the_catalog_example_comes_out_of_the_cubic_law() {
        let tav = cubic_mean_torque(&[(400.0, 7.0, 0.3), (320.0, 14.0, 3.0), (200.0, 7.0, 0.4), (0.0, 0.0, 0.2)]);
        assert!((tav - 319.7).abs() < 0.5, "{tav}");
        let l10 = STRAIN_WAVE_L10_H * (294.0 / tav.floor()).powi(3) * (2000.0 / 1440.0);
        assert!((l10 - 7610.0).abs() < 5.0, "{l10}");
    }

    #[test]
    fn a_bus_servo_wears_out_in_hours_where_a_qdd_lasts_years() {
        let r = rating(&STS3215);
        assert!(!r.stand_in && (r.torque_nm - 0.588).abs() < 1e-3);
        assert!((life_h(&STS3215, r.torque_nm) - r.hours).abs() < 1e-3);
        assert!((life_h(&STS3215, 2.0 * r.torque_nm) - r.hours / 8.0).abs() < 1e-3);
        assert!(life_h(&STS3215, STS3215.cont_nm) < 10.0);
        assert!((life_h(&RS02, RS02.cont_nm) - 7_000.0).abs() < 1e-2 && rating(&RS02).stand_in);
        assert!((damage(&RS02, RS02.cont_nm, 3600.0) - 1.0 / 7_000.0).abs() < 1e-9);
        assert_eq!(damage(&RS02, 0.0, 10.0), 0.0);
    }
}
