//! A body's lengths (m). Everything that measures a body (balance, kinematics, gait, reach, the forks) reads them from
//! here, so robots of different builds share one set of models:
//!
//! - `Build::Compact`: the small research robot the benchmarks were calibrated on (1.21 m to the top of the head).
//! - `Build::Standard`: a person's size on legs, a 1-tonne counterbalanced lift truck on wheels, for the world's
//!   doors, shelves and walls, which are real sizes.
//! - `Build::Rig(id)`: lengths measured from a design (`crate::rig`).

use serde::{Deserialize, Serialize};

use crate::base::BaseKind;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Build {
    #[default]
    Compact,
    Standard,
    /// A registered rig (`crate::rig::get`).
    Rig(u32),
}

impl Build {
    pub fn id(self) -> String {
        match self {
            Build::Compact => "compact".into(),
            Build::Standard => "standard".into(),
            Build::Rig(id) => format!("rig{id}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dims {
    /// Lengths over the compact robot's: 1 for it. Parts scale by geometric similarity (`parts::Actuator::scaled`).
    pub k: f32,
    pub hip_m: f32,
    pub waist_m: f32,
    pub shoulder_m: f32,
    pub shoulder_half_m: f32,
    pub upper_arm_m: f32,
    pub forearm_m: f32,
    /// From the wrist to the middle of the palm.
    pub palm_m: f32,
    pub stance_m: f32,
    pub hip_half_m: f32,
    pub ankle_m: f32,
    pub neck_up_m: f32,
    /// The head camera's lens from the neck pivot: up the head and forward of it.
    pub camera_up_m: f32,
    pub camera_fwd_m: f32,
    /// The carried load: ahead of the hips, and below the waist.
    pub cargo_x_m: f32,
    pub cargo_below_waist_m: f32,
    pub knee_lever_walk_m: f32,
    pub knee_lever_stand_m: f32,
    pub wheel_r_m: f32,
    pub belt_w_m: f32,
    pub bumper_gap_m: f32,
    pub bumper_z_m: [f32; 2],
    /// Axle to axle (or the track's ground length); `None` keeps the base's own.
    pub wheelbase_m: Option<f32>,
    pub chassis_w_m: f32,
    pub tine_len_m: f32,
    pub tine_spread_m: f32,
    pub tine_w_m: f32,
    pub tine_t_m: f32,
    pub mast_top_m: f32,
    /// The mast's rails and the carriage, each from the centre line.
    pub mast_half_w_m: f32,
    pub carriage_half_w_m: f32,
    /// Tines above the floor in empty and in loaded travel.
    pub fork_rest_m: f32,
    pub fork_carry_m: f32,
    /// The carriage camera above the tines, and ahead of where they start (a stereo pair clears the tines' shanks).
    pub fork_camera_up_m: f32,
    pub fork_camera_fwd_m: f32,
    /// Mast stroke speed (m/s) and push (N).
    pub mast_mps: f32,
    pub mast_force_n: f32,
    /// Creep while the tines slide under or out of an item (m/s).
    pub insert_mps: f32,
    /// The narrowest item the tines take: one that does not fall between them.
    pub fork_min_item_m: f32,
    /// The rider's hips ahead of the axle midpoint (0 on a compact base).
    pub seat_x_m: f32,
    /// The top of the overhead guard, when the build says so.
    pub roof_m: Option<f32>,
    /// A truck's rated load; `None` keeps the base's own limits.
    pub payload_kg: Option<f32>,
}

impl Dims {
    pub const COMPACT: Dims = Dims {
        k: 1.0,
        hip_m: 0.58,
        waist_m: 0.70,
        shoulder_m: 1.00,
        shoulder_half_m: 0.18,
        upper_arm_m: 0.22,
        forearm_m: 0.20,
        palm_m: 0.06,
        stance_m: 0.18,
        hip_half_m: 0.09,
        ankle_m: 0.08,
        neck_up_m: 0.07,
        camera_up_m: 0.07,
        camera_fwd_m: 0.07,
        cargo_x_m: 0.12,
        cargo_below_waist_m: 0.10,
        knee_lever_walk_m: 0.085,
        knee_lever_stand_m: 0.03,
        wheel_r_m: 0.06,
        belt_w_m: 0.08,
        bumper_gap_m: 0.02,
        bumper_z_m: [0.03, 0.11],
        wheelbase_m: None,
        chassis_w_m: 0.40,
        tine_len_m: 0.20,
        tine_spread_m: 0.07,
        tine_w_m: 0.025,
        tine_t_m: 0.012,
        mast_top_m: 0.95,
        mast_half_w_m: 0.065,
        carriage_half_w_m: 0.075,
        fork_rest_m: 0.05,
        fork_carry_m: 0.10,
        fork_camera_up_m: 0.06,
        fork_camera_fwd_m: -0.01,
        mast_mps: 0.10,
        mast_force_n: 450.0,
        insert_mps: 0.12,
        fork_min_item_m: 0.095,
        seat_x_m: 0.0,
        roof_m: None,
        payload_kg: None,
    };

    /// A person of `stature_m`, from segment lengths as shares of stature (hip 0.530, shoulder 0.818, upper arm 0.186,
    /// forearm 0.146, hand 0.108, shoulder breadth 0.259; Drillis & Contini 1966).
    pub fn person(stature_m: f32) -> Dims {
        let h = stature_m;
        let k = h / HEAD_TOP_COMPACT_M;
        Dims {
            k,
            hip_m: 0.530 * h,
            waist_m: 0.625 * h,
            shoulder_m: 0.818 * h,
            shoulder_half_m: 0.5 * 0.259 * h,
            upper_arm_m: 0.186 * h,
            forearm_m: 0.146 * h,
            palm_m: 0.5 * 0.108 * h,
            stance_m: 0.12 * h,
            hip_half_m: 0.06 * h,
            ankle_m: 0.05 * h,
            neck_up_m: 0.067 * h,
            camera_up_m: 0.05 * h,
            camera_fwd_m: 0.05 * h,
            cargo_x_m: 0.12 * k,
            cargo_below_waist_m: 0.10 * k,
            knee_lever_walk_m: 0.085 * k,
            knee_lever_stand_m: 0.03 * k,
            ..Dims::COMPACT
        }
    }

    /// The rider of a counterbalanced lift truck, seated with the hips `SEAT_M` above the ground; the truck's own
    /// lengths are a compact 1-tonne electric's [직접: Linde E10, wheelbase 1129 mm, width 828 mm, overhead guard
    /// 2190 mm, forks 1150 mm].
    pub fn lift_truck(rider: &Dims, base: BaseKind) -> Dims {
        const SEAT_M: f32 = 1.00;
        let up = SEAT_M - rider.hip_m;
        let (wheelbase, belt) = if base == BaseKind::Track { (1.25, 0.26) } else { (1.129, 0.20) };
        Dims {
            hip_m: rider.hip_m + up,
            waist_m: rider.waist_m + up,
            shoulder_m: rider.shoulder_m + up,
            wheel_r_m: 0.21,
            belt_w_m: belt,
            bumper_gap_m: 0.03,
            bumper_z_m: [0.12, 0.34],
            wheelbase_m: Some(wheelbase),
            chassis_w_m: 0.828,
            tine_len_m: 1.15,
            tine_spread_m: 0.18,
            tine_w_m: 0.10,
            tine_t_m: 0.035,
            mast_top_m: 2.10,
            mast_half_w_m: 0.30,
            carriage_half_w_m: 0.43,
            fork_rest_m: 0.12,
            fork_carry_m: 0.20,
            fork_camera_up_m: 0.30,
            fork_camera_fwd_m: 0.035,
            mast_mps: 0.30,
            mast_force_n: 12_000.0,
            insert_mps: 0.30,
            fork_min_item_m: 0.085,
            seat_x_m: -0.22,
            roof_m: Some(2.19),
            payload_kg: Some(1000.0),
            ..*rider
        }
    }

    pub fn of(build: Build, base: BaseKind) -> Dims {
        match build {
            Build::Compact => Dims::COMPACT,
            Build::Standard => {
                let person = Dims::person(STANDARD_STATURE_M);
                if base == BaseKind::HumanFoot {
                    person
                } else {
                    Dims::lift_truck(&person, base)
                }
            }
            Build::Rig(id) => crate::rig::get(id).map_or(Dims::COMPACT, |r| r.dims),
        }
    }

    pub fn thigh_m(&self) -> f32 {
        (self.hip_m - self.ankle_m) * 0.5
    }

    pub fn shank_m(&self) -> f32 {
        (self.hip_m - self.ankle_m) * 0.5
    }

    pub fn pelvis_m(&self) -> f32 {
        self.waist_m - self.hip_m
    }

    pub fn trunk_m(&self) -> f32 {
        self.shoulder_m - self.waist_m
    }

    pub fn cargo_z_m(&self) -> f32 {
        self.waist_m - self.cargo_below_waist_m
    }

    pub fn deck_z_m(&self) -> f32 {
        self.wheel_r_m + 0.10
    }

    pub fn column_top_m(&self) -> f32 {
        self.shoulder_m + self.neck_up_m
    }

    /// Shoulder to the middle of the palm, arm straight.
    pub fn arm_reach_m(&self) -> f32 {
        self.upper_arm_m + self.forearm_m + self.palm_m
    }

    /// The highest the tines go: the mast top less the carriage.
    pub fn fork_max_m(&self) -> f32 {
        self.mast_top_m - 0.10
    }
}

/// The compact robot's height to the top of its head.
pub const HEAD_TOP_COMPACT_M: f32 = 1.21;
/// A standard build's stature (m): the world's doors (2.03 m) and shelves (1.83 m) are for people about this tall.
pub const STANDARD_STATURE_M: f32 = 1.65;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standard_person_fits_the_doors_and_a_truck_its_warehouse() {
        let p = Dims::of(Build::Standard, BaseKind::HumanFoot);
        assert!((1.40..1.50).contains(&p.column_top_m()), "{p:?}");
        assert!((p.thigh_m() + p.shank_m() - (p.hip_m - p.ankle_m)).abs() < 1e-5);
        assert!(p.k > 1.3 && p.k < 1.4);
        let t = Dims::of(Build::Standard, BaseKind::Wheel);
        assert!(t.chassis_w_m < 2.5 && t.roof_m.unwrap() < 3.0, "through the warehouse door");
        assert_eq!(t.upper_arm_m, p.upper_arm_m, "the rider is the same person");
        assert_eq!(Dims::of(Build::Compact, BaseKind::Wheel), Dims::COMPACT);
    }
}
