use serde::{Deserialize, Serialize};

const STEP_TIME_S: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaseKind {
    Wheel,
    Mecanum,
    Track,
    HumanFoot,
}

impl BaseKind {
    pub const ALL: [BaseKind; 4] = [
        BaseKind::Wheel,
        BaseKind::Mecanum,
        BaseKind::Track,
        BaseKind::HumanFoot,
    ];

    pub fn spec(self) -> BaseSpec {
        match self {
            BaseKind::Wheel => BaseSpec {
                kind: self,
                max_speed_mps: 1.5,
                max_slope_deg: 15.0,
                max_step_m: 0.02,
                holonomic: false,
                power_w: 120.0,
                footprint_m: 0.30,
                drive_accel_mps2: None,
                mass_kg: None,
                walk_com_z_m: None,
                walk_foot_m: None,
            },
            BaseKind::Mecanum => BaseSpec {
                kind: self,
                max_speed_mps: 1.0,
                max_slope_deg: 8.0,
                max_step_m: 0.015,
                holonomic: true,
                power_w: 150.0,
                footprint_m: 0.30,
                drive_accel_mps2: None,
                mass_kg: None,
                walk_com_z_m: None,
                walk_foot_m: None,
            },
            BaseKind::Track => BaseSpec {
                kind: self,
                max_speed_mps: 0.8,
                max_slope_deg: 30.0,
                max_step_m: 0.12,
                holonomic: false,
                power_w: 200.0,
                footprint_m: 0.36,
                drive_accel_mps2: None,
                mass_kg: None,
                walk_com_z_m: None,
                walk_foot_m: None,
            },
            BaseKind::HumanFoot => BaseSpec {
                kind: self,
                max_speed_mps: 1.2,
                max_slope_deg: 20.0,
                max_step_m: 0.18,
                holonomic: false,
                power_w: 180.0,
                footprint_m: 0.25,
                drive_accel_mps2: None,
                mass_kg: None,
                walk_com_z_m: None,
                walk_foot_m: None,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaseSpec {
    pub kind: BaseKind,
    pub max_speed_mps: f32,
    pub max_slope_deg: f32,
    pub max_step_m: f32,
    pub holonomic: bool,
    pub power_w: f32,
    pub footprint_m: f32,
    #[serde(default)]
    pub drive_accel_mps2: Option<f32>,
    #[serde(default)]
    pub mass_kg: Option<f32>,
    #[serde(default)]
    pub walk_com_z_m: Option<f32>,
    #[serde(default)]
    pub walk_foot_m: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Terrain {
    Flat { length_m: f32 },
    Slope { length_m: f32, deg: f32 },
    Step { height_m: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveReport {
    pub passable: bool,
    pub failed_at: Option<usize>,
    pub distance_m: f32,
    pub time_s: f32,
    pub energy_j: f32,
}

pub fn simulate_drive(spec: &BaseSpec, course: &[Terrain]) -> DriveReport {
    let mut distance = 0.0;
    let mut time = 0.0;

    for (i, segment) in course.iter().enumerate() {
        let blocked = match *segment {
            Terrain::Flat { length_m } => {
                distance += length_m;
                time += length_m / spec.max_speed_mps;
                false
            }
            Terrain::Slope { length_m, deg } => {
                if deg > spec.max_slope_deg {
                    true
                } else {
                    let factor = 1.0 - 0.5 * (deg / spec.max_slope_deg);
                    distance += length_m;
                    time += length_m / (spec.max_speed_mps * factor);
                    false
                }
            }
            Terrain::Step { height_m } => {
                if height_m > spec.max_step_m {
                    true
                } else {
                    time += STEP_TIME_S;
                    false
                }
            }
        };

        if blocked {
            return DriveReport {
                passable: false,
                failed_at: Some(i),
                distance_m: distance,
                time_s: time,
                energy_j: spec.power_w * time,
            };
        }
    }

    DriveReport {
        passable: true,
        failed_at: None,
        distance_m: distance,
        time_s: time,
        energy_j: spec.power_w * time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_climbs_steps_that_stop_a_wheel() {
        let course = [Terrain::Step { height_m: 0.1 }];
        assert!(simulate_drive(&BaseKind::Track.spec(), &course).passable);
        let wheel = simulate_drive(&BaseKind::Wheel.spec(), &course);
        assert!(!wheel.passable);
        assert_eq!(wheel.failed_at, Some(0));
    }

    #[test]
    fn human_foot_climbs_stairs_but_mecanum_cannot_on_slope() {
        let stairs = [Terrain::Step { height_m: 0.17 }];
        assert!(simulate_drive(&BaseKind::HumanFoot.spec(), &stairs).passable);

        let slope = [Terrain::Slope { length_m: 5.0, deg: 12.0 }];
        assert!(!simulate_drive(&BaseKind::Mecanum.spec(), &slope).passable);
        assert!(simulate_drive(&BaseKind::Track.spec(), &slope).passable);
    }

    #[test]
    fn flat_drive_time_follows_speed() {
        let course = [Terrain::Flat { length_m: 3.0 }];
        let r = simulate_drive(&BaseKind::Wheel.spec(), &course);
        assert!((r.time_s - 2.0).abs() < 1e-5);
        assert!((r.energy_j - 240.0).abs() < 1e-3);
    }

    #[test]
    fn only_mecanum_is_holonomic() {
        let holo: Vec<_> = BaseKind::ALL
            .iter()
            .filter(|k| k.spec().holonomic)
            .collect();
        assert_eq!(holo, vec![&BaseKind::Mecanum]);
    }
}
