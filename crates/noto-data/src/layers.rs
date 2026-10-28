use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    VisionSpatial,
    TactilePhysical,
    MaterialDynamics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Now,
    Next,
    Planned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Rgb,
    Depth,
    Segmentation,
    PointCloud,
    SurfaceNormals,
    Tactile,
    JointState,
    ObjectPose,
    Friction,
    MassProperties,
    Contacts,
}

impl Layer {
    pub const ALL: [Layer; 11] = [
        Layer::Rgb,
        Layer::Depth,
        Layer::Segmentation,
        Layer::PointCloud,
        Layer::SurfaceNormals,
        Layer::Tactile,
        Layer::JointState,
        Layer::ObjectPose,
        Layer::Friction,
        Layer::MassProperties,
        Layer::Contacts,
    ];

    pub fn group(self) -> Group {
        match self {
            Layer::Rgb
            | Layer::Depth
            | Layer::Segmentation
            | Layer::PointCloud
            | Layer::SurfaceNormals => Group::VisionSpatial,
            Layer::Tactile | Layer::JointState | Layer::ObjectPose | Layer::Contacts => {
                Group::TactilePhysical
            }
            Layer::Friction | Layer::MassProperties => Group::MaterialDynamics,
        }
    }

    pub fn readiness(self) -> Readiness {
        match self {
            Layer::JointState
            | Layer::ObjectPose
            | Layer::Friction
            | Layer::MassProperties
            | Layer::Contacts => Readiness::Now,
            Layer::PointCloud | Layer::SurfaceNormals => Readiness::Next,
            Layer::Rgb | Layer::Depth | Layer::Segmentation | Layer::Tactile => Readiness::Planned,
        }
    }

    pub fn table(self) -> &'static str {
        match self {
            Layer::Rgb | Layer::Depth | Layer::Segmentation => "vision",
            Layer::PointCloud | Layer::SurfaceNormals => "point_cloud",
            Layer::Tactile => "tactile",
            Layer::JointState => "joint_states",
            Layer::Contacts => "contacts",
            Layer::ObjectPose | Layer::Friction | Layer::MassProperties => "samples",
        }
    }

    pub fn name_ko(self) -> &'static str {
        match self {
            Layer::Rgb => "RGB 이미지 (조명·그림자·질감 도메인 무작위화)",
            Layer::Depth => "깊이 맵 (RGB-D)",
            Layer::Segmentation => "시맨틱·인스턴스 분할 마스크",
            Layer::PointCloud => "포인트 클라우드·3D 메쉬",
            Layer::SurfaceNormals => "표면 법선",
            Layer::Tactile => "촉각·압력 맵",
            Layer::JointState => "관절 각도·토크",
            Layer::ObjectPose => "객체 6DoF 포즈",
            Layer::Friction => "마찰 계수",
            Layer::MassProperties => "질량 중심·관성 텐서",
            Layer::Contacts => "충돌·접촉점",
        }
    }

    pub fn status_note(self) -> &'static str {
        match self {
            Layer::Rgb => "렌더러 필요. 조명·그림자·텍스처 무작위화 포함",
            Layer::Depth => "객체·손 형상에 대한 광선 투사로 생성 예정",
            Layer::Segmentation => "기하 래스터화로 손가락·palm·객체 클래스 라벨 생성 예정",
            Layer::PointCloud => "박스와 손가락 링크 표면을 해석적으로 샘플링 예정",
            Layer::SurfaceNormals => "샘플 점마다 면 법선 계산 예정. 픽셀 단위는 렌더러 필요",
            Layer::Tactile => "접촉점과 수직력을 손가락 끝 taxel 격자로 분배하는 모델 예정",
            Layer::JointState => "굴곡·외전·비틀림 각도와 토크 한계 기록. 실제 적용 토크 추정은 예정",
            Layer::ObjectPose => "배치 위치 기록. roll·pitch·yaw는 아직 0으로 기록",
            Layer::Friction => "재질별 마찰 계수 무작위화하여 기록",
            Layer::MassProperties => "박스 관성 텐서를 해석식으로 기록. 비대칭 물체는 예정",
            Layer::Contacts => "접촉점, 법선, 접촉력을 기록",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layer_maps_to_a_table_and_group() {
        for l in Layer::ALL {
            assert!(!l.table().is_empty());
            let _ = l.group();
            assert!(!l.name_ko().is_empty());
        }
    }

    #[test]
    fn contacts_and_joints_are_recorded_now() {
        assert_eq!(Layer::Contacts.readiness(), Readiness::Now);
        assert_eq!(Layer::JointState.readiness(), Readiness::Now);
        assert_eq!(Layer::Rgb.readiness(), Readiness::Planned);
    }
}
