//! Serializable views of the simulator, and the pure functions that build them.
//! The Tauri commands are thin wrappers around these, so the logic can be
//! unit-tested without a window.

use hand_core::sim::{closure_angles, fk_all};
use hand_core::{
    grasp, simulate_drive, BaseKind, BaseSpec, DriveReport, FingerCount, GraspResult, GraspSetup,
    HandConfig, JointAngles, JointKind, Payload, Terrain,
};
use serde::Serialize;

pub type Point = [f32; 3];

#[derive(Clone, Debug, Serialize)]
pub struct FingerView {
    pub name: String,
    pub is_thumb: bool,
    /// Joint points from the knuckle to the tip, hand frame, metres.
    pub points: Vec<Point>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HandView {
    pub fingers: Vec<FingerView>,
    /// Palm width and length, metres.
    pub palm: [f32; 2],
    pub base: BaseSpec,
    /// Longest finger reach, metres.
    pub reach_m: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ObjectView {
    pub center: Point,
    pub half: Point,
}

#[derive(Clone, Debug, Serialize)]
pub struct GraspView {
    pub result: GraspResult,
    pub fingers: Vec<FingerView>,
    pub object: ObjectView,
}

#[derive(Clone, Debug, Serialize)]
pub struct DriveView {
    pub base: BaseSpec,
    pub label_ko: &'static str,
    pub report: DriveReport,
}

#[derive(Clone, Debug, Serialize)]
pub struct FingerPreset {
    pub count: u8,
    pub label_ko: &'static str,
    pub description_ko: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptionLabel {
    pub id: &'static str,
    pub label_ko: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct PresetsView {
    pub fingers: Vec<FingerPreset>,
    pub joint_kinds: Vec<OptionLabel>,
    pub bases: Vec<DriveView>,
}

fn to_point(v: hand_core::Vec3) -> Point {
    [v.x, v.y, v.z]
}

fn finger_views(hand: &hand_core::Hand, angles: &[Vec<JointAngles>]) -> Vec<FingerView> {
    hand.fingers
        .iter()
        .zip(fk_all(hand, angles))
        .map(|(f, pts)| FingerView {
            name: f.name.clone(),
            is_thumb: f.is_thumb,
            points: pts.into_iter().map(to_point).collect(),
        })
        .collect()
}

/// Hand geometry with every finger closed to `closure` in `[0, 1]`.
pub fn hand_view(config: &HandConfig, closure: f32) -> Result<HandView, String> {
    let hand = config.build().map_err(|e| e.to_string())?;
    let actuation = hand.config.joints.actuation;
    let angles: Vec<Vec<JointAngles>> = hand
        .fingers
        .iter()
        .map(|f| closure_angles(f, actuation, closure))
        .collect();
    Ok(HandView {
        fingers: finger_views(&hand, &angles),
        palm: [hand.palm.width, hand.palm.length],
        base: hand.base,
        reach_m: hand.fingers.iter().map(|f| f.reach()).fold(0.0, f32::max),
    })
}

/// Closes the hand on `payload` and reports the grasp.
pub fn grasp_view(
    config: &HandConfig,
    payload: &Payload,
    setup: &GraspSetup,
) -> Result<GraspView, String> {
    let hand = config.build().map_err(|e| e.to_string())?;
    let outcome = grasp(&hand, payload, setup);
    let half = [
        payload.size_m[0] * 0.5,
        payload.size_m[1] * 0.5,
        payload.size_m[2] * 0.5,
    ];
    Ok(GraspView {
        fingers: finger_views(&hand, &outcome.angles),
        object: ObjectView {
            center: [0.0, half[1], setup.z_offset],
            half,
        },
        result: outcome.result,
    })
}

/// The route used to compare bases: flat ground, two slopes, two steps.
pub fn default_course() -> Vec<Terrain> {
    vec![
        Terrain::Flat { length_m: 10.0 },
        Terrain::Slope {
            length_m: 5.0,
            deg: 10.0,
        },
        Terrain::Step { height_m: 0.05 },
        Terrain::Slope {
            length_m: 5.0,
            deg: 20.0,
        },
        Terrain::Step { height_m: 0.15 },
        Terrain::Flat { length_m: 5.0 },
    ]
}

pub fn base_label_ko(kind: BaseKind) -> &'static str {
    match kind {
        BaseKind::Wheel => "바퀴 (4륜 차동)",
        BaseKind::Mecanum => "메카넘 바퀴 (전방향)",
        BaseKind::Track => "트랙 (크롤러·트랙터형)",
        BaseKind::HumanFoot => "사람 발 (이족)",
    }
}

/// Every base driven over the same course.
pub fn drive_matrix(course: &[Terrain]) -> Vec<DriveView> {
    BaseKind::ALL
        .into_iter()
        .map(|kind| {
            let spec = kind.spec();
            DriveView {
                base: spec,
                label_ko: base_label_ko(kind),
                report: simulate_drive(&spec, course),
            }
        })
        .collect()
}

pub fn presets_view() -> PresetsView {
    let fingers = FingerCount::ALL
        .into_iter()
        .map(|c| {
            let (label_ko, description_ko) = match c {
                FingerCount::Two => ("2지 집게", "마주 보는 두 손가락. 평행 조 그리퍼에 가까움"),
                FingerCount::Three => ("3지 (엄지+2)", "엄지와 두 손가락으로 감싸는 삼점 파지"),
                FingerCount::Four => ("4지 (엄지+3)", "엄지와 세 손가락"),
                FingerCount::Five => ("5지 (사람형)", "사람 손과 같은 배치"),
            };
            FingerPreset {
                count: c.count(),
                label_ko,
                description_ko,
            }
        })
        .collect();
    let joint_kinds = JointKind::ALL
        .into_iter()
        .map(|k| OptionLabel {
            id: match k {
                JointKind::Hinge => "hinge",
                JointKind::Universal => "universal",
                JointKind::Ball => "ball",
            },
            label_ko: match k {
                JointKind::Hinge => "힌지 (굴곡 1자유도)",
                JointKind::Universal => "유니버설 (굴곡+외전)",
                JointKind::Ball => "볼 (3자유도)",
            },
        })
        .collect();
    let bases = BaseKind::ALL
        .into_iter()
        .map(|kind| DriveView {
            base: kind.spec(),
            label_ko: base_label_ko(kind),
            report: simulate_drive(&kind.spec(), &[]),
        })
        .collect();
    PresetsView {
        fingers,
        joint_kinds,
        bases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hand_core::{Actuation, GraspType, HandConfig};

    fn carton() -> Payload {
        Payload {
            name: "carton".into(),
            size_m: [0.2, 0.15, 0.1],
            mass_kg: 1.0,
            friction: 0.5,
        }
    }

    #[test]
    fn hand_view_has_one_finger_view_per_finger() {
        for c in FingerCount::ALL {
            let cfg = HandConfig::preset(c);
            let v = hand_view(&cfg, 0.5).unwrap();
            assert_eq!(v.fingers.len(), c.count() as usize);
            // Each finger has one point per joint plus the tip.
            let expected = cfg.joints.segments as usize + 1;
            assert!(v.fingers.iter().all(|f| f.points.len() == expected));
            assert!(v.reach_m > 0.0);
        }
    }

    #[test]
    fn invalid_joint_options_surface_as_errors() {
        let mut cfg = HandConfig::preset(FingerCount::Five);
        cfg.joints.segments = 9;
        assert!(hand_view(&cfg, 0.0).is_err());
        assert!(grasp_view(&cfg, &carton(), &GraspSetup::default()).is_err());
    }

    #[test]
    fn grasp_view_reports_palm_supported_grasp() {
        let v = grasp_view(
            &HandConfig::preset(FingerCount::Five),
            &carton(),
            &GraspSetup::default(),
        )
        .unwrap();
        assert!(v.result.stable);
        assert_ne!(v.result.grasp_type, GraspType::None);
        assert!((v.object.half[1] - 0.075).abs() < 1e-6);
    }

    #[test]
    fn tendon_coupling_is_reflected_in_closed_pose() {
        let mut cfg = HandConfig::preset(FingerCount::Five);
        cfg.joints.actuation = Actuation::Coupled { ratio: 0.5 };
        let open = hand_view(&cfg, 0.0).unwrap();
        let closed = hand_view(&cfg, 1.0).unwrap();
        assert!(closed.fingers[1].points[3][1] > open.fingers[1].points[3][1]);
    }

    #[test]
    fn drive_matrix_covers_all_bases() {
        let m = drive_matrix(&default_course());
        assert_eq!(m.len(), BaseKind::ALL.len());
        // Failure points follow the course: 5 cm step (index 2), 15 cm step (index 4).
        let failed_at = |kind| {
            m.iter()
                .find(|d| d.base.kind == kind)
                .unwrap()
                .report
                .failed_at
        };
        assert_eq!(failed_at(BaseKind::Wheel), Some(2));
        assert_eq!(failed_at(BaseKind::Mecanum), Some(1));
        assert_eq!(failed_at(BaseKind::Track), Some(4));
        assert_eq!(failed_at(BaseKind::HumanFoot), None);
    }

    #[test]
    fn presets_list_every_option() {
        let p = presets_view();
        assert_eq!(p.fingers.len(), 4);
        assert_eq!(p.joint_kinds.len(), 3);
        assert_eq!(p.bases.len(), 4);
    }

    /// The UI reads these exact keys. Renaming a field here breaks the page,
    /// so the JSON shape is checked as well as the values.
    #[test]
    fn json_shapes_match_what_the_ui_reads() {
        let cfg = HandConfig::preset(FingerCount::Five);
        let geom = serde_json::to_value(hand_view(&cfg, 0.5).unwrap()).unwrap();
        assert!(geom["reach_m"].is_number());
        assert!(geom["palm"].is_array());
        assert_eq!(geom["base"]["kind"], "wheel");
        assert!(geom["fingers"][0]["points"][0].is_array());
        assert!(geom["fingers"][0]["is_thumb"].is_boolean());

        let g = serde_json::to_value(
            grasp_view(&cfg, &carton(), &GraspSetup::default()).unwrap(),
        )
        .unwrap();
        assert!(g["object"]["center"].is_array());
        assert!(g["object"]["half"].is_array());
        assert!(g["result"]["stable"].is_boolean());
        assert!(g["result"]["max_payload_kg"].is_number());
        assert!(g["result"]["grasp_type"].is_string());
        let contacts = g["result"]["contacts"].as_array().unwrap();
        assert!(!contacts.is_empty());
        assert!(contacts[0]["source"]["source"].is_string());
        // Vec3 values serialize as {x, y, z} objects; points in the fingers are arrays.
        assert!(contacts[0]["normal"]["z"].is_number());
        assert!(contacts[0]["point"]["y"].is_number());

        let p = serde_json::to_value(presets_view()).unwrap();
        assert_eq!(p["joint_kinds"][0]["id"], "hinge");
        assert_eq!(p["bases"][3]["base"]["kind"], "human_foot");

        let d = serde_json::to_value(drive_matrix(&default_course())).unwrap();
        assert!(d[0]["report"]["failed_at"].is_null() || d[0]["report"]["failed_at"].is_number());
        assert!(d[0]["report"]["energy_j"].is_number());
    }
}
