use serde::{Deserialize, Serialize};

use crate::base::BaseKind;

pub const G: f32 = 9.81;

pub const GAIT_RCOF: f32 = 0.223;
pub const CRR_REFERENCE: f32 = 0.013;
pub const MECANUM_MU_FACTOR: f32 = 0.8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    #[default]
    Firm,
    Rock,
    Dirt,
    Grass,
    Gravel,
    Sand,
    WetSand,
    Water,
}

impl Surface {
    pub const ALL: [Surface; 8] = [Surface::Firm, Surface::Rock, Surface::Dirt, Surface::Grass, Surface::Gravel, Surface::Sand, Surface::WetSand, Surface::Water];

    pub fn props(self) -> &'static SurfaceProps {
        &SURFACES[self as usize]
    }

    pub fn id(self) -> &'static str {
        self.props().id
    }

    pub fn label_ko(self) -> &'static str {
        self.props().label_ko
    }

    pub fn from_id(id: &str) -> Option<Surface> {
        Surface::ALL.into_iter().find(|s| s.id() == id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SurfaceProps {
    pub id: &'static str,
    pub label_ko: &'static str,
    pub mu_foot: f32,
    pub mu_wheel: f32,
    pub mu_track: f32,
    pub crr_wheel: f32,
    pub crr_track: f32,
    pub eta_walk: f32,
    pub wet: bool,
    pub cite: &'static str,
}

pub const SURFACES: [SurfaceProps; 8] = [
    SurfaceProps {
        id: "firm",
        label_ko: "단단한 바닥",
        mu_foot: 0.725,
        mu_wheel: 0.85,
        mu_track: 0.725,
        crr_wheel: 0.013,
        crr_track: 0.04,
        eta_walk: 1.0,
        wet: false,
        cite: "고무–건조 콘크리트 0.6–0.85 중앙 (Engineering ToolBox); 타이어–건조 아스팔트·콘크리트 0.80–0.90 (Wong 1993); Crr 콘크리트·아스팔트 0.013 (Bosch 1996); η 아스팔트 1.0 (Soule & Goldman 1972); 트랙 Crr 추정",
    },
    SurfaceProps {
        id: "rock",
        label_ko: "바위 지면",
        mu_foot: 0.80,
        mu_wheel: 0.85,
        mu_track: 0.80,
        crr_wheel: 0.015,
        crr_track: 0.04,
        eta_walk: 1.28,
        wet: false,
        cite: "발 0.80 추정 (고무–콘크리트 0.6–0.85 상단과 화강암–등반 고무 0.97–1.21 (Elkington 2024) 사이); 타이어 건조 0.80–0.90 (Wong 1993); Crr 0.010–0.015 상단 (Gillespie 1992); 울퉁불퉁한 지면 보행 +28% (Voloshina 2013); 트랙 Crr 추정",
    },
    SurfaceProps {
        id: "dirt",
        label_ko: "흙",
        mu_foot: 0.68,
        mu_wheel: 0.68,
        mu_track: 0.68,
        crr_wheel: 0.05,
        crr_track: 0.07,
        eta_walk: 1.1,
        wet: false,
        cite: "흙길 건조 타이어 최대 0.68 (Wong 1993, 발·트랙에 적용은 추정); 비포장 Crr 0.05, 트랙 들판 0.07 (Bosch 1996); η 흙길 1.1 (Soule & Goldman 1972)",
    },
    SurfaceProps {
        id: "grass",
        label_ko: "풀밭",
        mu_foot: 0.50,
        mu_wheel: 0.35,
        mu_track: 0.45,
        crr_wheel: 0.10,
        crr_track: 0.07,
        eta_walk: 1.1,
        wet: false,
        cite: "타이어–풀 0.35 (Engineering ToolBox); 발 0.50·트랙 0.45는 추정 (신발–풀 절대값 미발견); 들판 Crr 0.10–0.35 하한, 트랙 0.07 (Bosch 1996); η 풀 1.1 (de Gruchy 2017 정리표)",
    },
    SurfaceProps {
        id: "gravel",
        label_ko: "자갈 패드",
        mu_foot: 0.60,
        mu_wheel: 0.60,
        mu_track: 0.60,
        crr_wheel: 0.02,
        crr_track: 0.05,
        eta_walk: 1.09,
        wet: false,
        cite: "자갈 타이어 최대 0.60 (Wong 1993, 발·트랙 적용은 추정); 다진 자갈 Crr 0.02 (Bosch 1996); η 자갈·흙길 평균 1.09 (Richmond 2015); 트랙 Crr 추정",
    },
    SurfaceProps {
        id: "sand",
        label_ko: "모래",
        mu_foot: 0.55,
        mu_wheel: 0.55,
        mu_track: 0.55,
        crr_wheel: 0.30,
        crr_track: 0.12,
        eta_walk: 2.1,
        wet: false,
        cite: "느슨한 모래 φ 29–30° → tan φ 0.55–0.58: 어떤 주행부든 미끄러짐 끝의 견인 상한 H = W·tan φ (Lukić 2018; 견인 식은 UMD ENAE 788X 2024); 모래 Crr 0.30 (Gillespie 1992), 트랙 들판 상한 0.12 (Bosch 1996); η 느슨한 모래 2.1 (Soule & Goldman 1972)",
    },
    SurfaceProps {
        id: "wet_sand",
        label_ko: "젖은 모래",
        mu_foot: 0.45,
        mu_wheel: 0.55,
        mu_track: 0.60,
        crr_wheel: 0.08,
        crr_track: 0.07,
        eta_walk: 1.5,
        wet: true,
        cite: "물가의 젖은 모래는 마른 모래보다 단단함 (Richmond 2015, 정성); 타이어 흙길 젖음 0.55, 중간 단단한 흙 Crr 0.08 (Wong 1993); 발 0.45·트랙 0.60·η 1.5는 추정 (η: 경작지 1.5, Givoni & Goldman 1971)",
    },
    SurfaceProps {
        id: "water",
        label_ko: "바다",
        mu_foot: 0.0,
        mu_wheel: 0.0,
        mu_track: 0.0,
        crr_wheel: 1.0,
        crr_track: 1.0,
        eta_walk: 2.0,
        wet: true,
        cite: "들어갈 수 없음 (월드 규칙)",
    },
];

pub fn available_mu(base: BaseKind, s: &SurfaceProps) -> f32 {
    match base {
        BaseKind::HumanFoot => s.mu_foot,
        BaseKind::Track => s.mu_track,
        BaseKind::Wheel => s.mu_wheel,
        BaseKind::Mecanum => s.mu_wheel * MECANUM_MU_FACTOR,
    }
}

pub fn rolling_resistance(base: BaseKind, s: &SurfaceProps) -> f32 {
    match base {
        BaseKind::HumanFoot => 0.0,
        BaseKind::Track => s.crr_track,
        BaseKind::Wheel | BaseKind::Mecanum => s.crr_wheel,
    }
}

fn own_shear(base: BaseKind, s: &SurfaceProps) -> f32 {
    if base == BaseKind::HumanFoot {
        GAIT_RCOF
    } else {
        rolling_resistance(base, s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contact {
    pub available_mu: f32,
    pub required_mu: f32,
    pub utilisation: f32,
    pub margin: f32,
    pub slip_probability: Option<f32>,
    pub net_traction: f32,
}

pub fn required_mu(base: BaseKind, s: &SurfaceProps, slope_deg: f32, accel_mps2: f32, moving: bool) -> f32 {
    let th = slope_deg.to_radians().clamp(-1.4, 1.4);
    (th.tan() + accel_mps2 / (G * th.cos())).abs() + if moving { own_shear(base, s) } else { 0.0 }
}

pub fn contact(base: BaseKind, s: &SurfaceProps, slope_deg: f32, accel_mps2: f32, moving: bool) -> Contact {
    let available = available_mu(base, s);
    let required = required_mu(base, s, slope_deg, accel_mps2, moving);
    let margin = available - required;
    Contact {
        available_mu: available,
        required_mu: required,
        utilisation: if available > 1e-6 { required / available } else { f32::INFINITY },
        margin,
        slip_probability: (base == BaseKind::HumanFoot && moving).then(|| slip_probability(margin)),
        net_traction: available - own_shear(base, s),
    }
}

pub fn slip_probability(margin: f32) -> f32 {
    1.0 / (1.0 + (0.33 + 55.4 * margin).exp())
}

pub fn friction_slope_limit_deg(base: BaseKind, s: &SurfaceProps) -> f32 {
    (available_mu(base, s) - own_shear(base, s)).atan().to_degrees()
}

pub fn climb_limit_deg(base: BaseKind, s: &SurfaceProps, drive_accel_mps2: Option<f32>) -> f32 {
    let friction = friction_slope_limit_deg(base, s);
    match drive_accel_mps2 {
        Some(a) if base != BaseKind::HumanFoot => {
            let k = a / G;
            let c = rolling_resistance(base, s);
            let r = (k / (1.0 + c * c).sqrt()).clamp(-1.0, 1.0);
            let motor = (r.asin() - c.atan()).to_degrees();
            motor.min(friction)
        }
        _ => friction,
    }
}

pub fn traction_accel_limit(base: BaseKind, s: &SurfaceProps, slope_deg: f32) -> f32 {
    let th = slope_deg.to_radians();
    (G * ((available_mu(base, s) - own_shear(base, s)) * th.cos() - th.sin())).max(0.0)
}

pub fn energy_factor(base: BaseKind, s: &SurfaceProps, mass_kg: f32, speed_mps: f32, power_w: f32) -> f32 {
    match base {
        BaseKind::HumanFoot => s.eta_walk,
        _ => 1.0 + ((rolling_resistance(base, s) - CRR_REFERENCE).max(0.0) * mass_kg * G * speed_mps / power_w.max(1.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ground_has_numbers_and_a_source_in_enum_order() {
        for (i, s) in SURFACES.iter().enumerate() {
            assert!(!s.cite.is_empty(), "{}", s.id);
            assert!(s.mu_foot >= 0.0 && s.crr_wheel >= 0.0 && s.eta_walk >= 1.0, "{}", s.id);
            assert_eq!(Surface::ALL[i].id(), s.id);
            assert_eq!(Surface::from_id(s.id), Some(Surface::ALL[i]));
        }
        assert_eq!(Surface::default(), Surface::Firm);
    }

    #[test]
    fn the_slip_fit_reproduces_its_points() {
        assert!((slip_probability(0.077) - 0.01).abs() < 0.002);
        assert!((slip_probability(-0.006) - 0.5).abs() < 0.01);
        assert!((slip_probability(-0.090) - 0.99).abs() < 0.002);
    }

    #[test]
    fn loose_sand_holds_a_wheel_to_gentle_slopes_and_costs_legs_twice() {
        let sand = Surface::Sand.props();
        let w = friction_slope_limit_deg(BaseKind::Wheel, sand);
        assert!(w < 15.0 && w > 10.0, "{w}");
        assert!(friction_slope_limit_deg(BaseKind::Track, sand) > 20.0);
        assert!(friction_slope_limit_deg(BaseKind::HumanFoot, sand) > 15.0);
        assert!(friction_slope_limit_deg(BaseKind::Wheel, Surface::Rock.props()) > 35.0);
        assert!((energy_factor(BaseKind::HumanFoot, sand, 20.0, 1.0, 180.0) - 2.1).abs() < 1e-6);
        assert!(energy_factor(BaseKind::Wheel, sand, 20.0, 1.5, 120.0) > 1.5);
        assert_eq!(energy_factor(BaseKind::Wheel, Surface::Firm.props(), 20.0, 1.5, 120.0), 1.0);
    }

    #[test]
    fn required_friction_grows_with_slope_acceleration_and_motion() {
        let rock = Surface::Rock.props();
        let flat = required_mu(BaseKind::HumanFoot, rock, 0.0, 0.0, true);
        assert!((flat - GAIT_RCOF).abs() < 1e-6);
        assert_eq!(required_mu(BaseKind::HumanFoot, rock, 0.0, 0.0, false), 0.0, "standing still needs no shear");
        let up = required_mu(BaseKind::HumanFoot, rock, 10.0, 0.0, true);
        assert!((up - flat - 10f32.to_radians().tan()).abs() < 1e-5);
        assert!(required_mu(BaseKind::Wheel, rock, 0.0, 2.0, true) > required_mu(BaseKind::Wheel, rock, 0.0, 0.0, true) + 0.19);
        let (brake, start) = (required_mu(BaseKind::Wheel, rock, -10.0, -1.0, false), required_mu(BaseKind::Wheel, rock, -10.0, 1.0, false));
        assert!(brake > 10f32.to_radians().tan() + 0.1 && start < 10f32.to_radians().tan() - 0.09, "{brake} {start}");
        let grass = contact(BaseKind::Wheel, Surface::Grass.props(), 15.0, 0.0, true);
        assert!(grass.utilisation > 1.0 && grass.slip_probability.is_none(), "a wheel slides up a 15° grass slope");
        let walk = contact(BaseKind::HumanFoot, Surface::WetSand.props(), 0.0, 0.0, true);
        assert!(walk.slip_probability.unwrap() < 0.01, "level wet sand is walkable");
        assert!(traction_accel_limit(BaseKind::Wheel, Surface::Sand.props(), 0.0) < 2.5);
    }

    #[test]
    fn the_motor_or_the_ground_limits_the_climb() {
        let firm = Surface::Firm.props();
        let f = climb_limit_deg(BaseKind::Wheel, firm, Some(240.0 / 25.0));
        assert!((f - friction_slope_limit_deg(BaseKind::Wheel, firm)).abs() < 1e-4);
        let weak = climb_limit_deg(BaseKind::Wheel, Surface::Sand.props(), Some(46.0 / 10.0));
        let k: f32 = 46.0 / (10.0 * G);
        let th = weak.to_radians();
        assert!((th.sin() + 0.30 * th.cos() - k).abs() < 1e-3, "{weak}");
        assert_eq!(climb_limit_deg(BaseKind::HumanFoot, firm, None), friction_slope_limit_deg(BaseKind::HumanFoot, firm));
        assert!(climb_limit_deg(BaseKind::Wheel, Surface::Sand.props(), Some(0.25 * G)) < 0.0);
        let glycerol = SurfaceProps { mu_foot: 0.164, ..*firm };
        assert!(friction_slope_limit_deg(BaseKind::HumanFoot, &glycerol) < 0.0);
    }
}
