# 데이터 출처와 계획

이 폴더에는 물류 로봇 손 시뮬레이터가 쓰는 사전과 출처 기록이 있습니다. 합성 데이터셋은 LanceDB(Lance 포맷)에 저장하며, 저장 위치는 실행 시 지정합니다.

## 1. 물류 어휘 사전

`logistics_dictionary.json`은 SigLIP2 제로샷 분류에 쓰는 라벨이며, 각 항목의 크기·질량·파지 가능 여부를 함께 가집니다. 같은 내용을 Lance `dictionary` 테이블에도 기록합니다.

- 크기와 질량 값은 대부분 **가정값**입니다. `source` 필드에 근거를 적었습니다.
- EPAL 팔레트(1200 × 800 mm)만 공개 규격 근거가 있습니다.
- `graspable: false`는 손이 잡으면 안 되는 대상(차량, 선반, 사람)입니다.

## 2. 손 모델 근거 (논문)

| 근거 | 쓰는 곳 | 상태 |
| --- | --- | --- |
| Feix et al., "The GRASP Taxonomy of Human Grasp Types", IEEE THMS 46(1), 2016, pp. 66–77, DOI 10.1109/THMS.2015.2470657 | 파지 유형(pinch/power) 분류 | 서지 정보 확인됨 |
| Buchholz, Armstrong, Goldstein, "Anthropometric data for describing the kinematics of the human hand", Ergonomics 35(3), 1992, pp. 261–273 | 관절 굴곡 범위와 손가락 분절 길이의 기준 | 서지 정보만 확인됨. 본문 수치는 아직 대조하지 않음 |
| Buchholz & Armstrong, "A kinematic model of the human hand to evaluate its prehensile capabilities", J. Biomechanics 25(2), 1992, pp. 149–162 | 손가락 감싸기 모델의 검증 기준 | 서지 정보 확인됨 |

관절 굴곡 상한(MCP 90°, PIP 100°, DIP 80°)은 사람 손의 대략적인 값이며, 위 논문의 수치와 대조해서 고칠 예정입니다.

## 3. 합성 데이터 방법 (논문)

| 근거 | 쓰는 곳 |
| --- | --- |
| Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", arXiv:1703.06907 (2017) | RGB 도메인 무작위화(조명, 질감) 계획 |
| Mahler et al., "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics", RSS 2017, DOI 10.15607/RSS.2017.XIII.058 | 합성 포인트 클라우드와 해석적 파지 지표 방식의 참고 |

## 4. 비전 모델

- SigLIP 2 (Tschannen et al., arXiv:2502.14786). NaFlex 변형을 씁니다. 초기 개발은 HF 캐시에 있는 `google/siglip2-base-patch16-naflex`의 구조를 따릅니다. 가중치 파일(`model.safetensors`)은 캐시에 없어서, 실행 전에 받아야 합니다.

## 5. 합성 데이터 계층

| 계층 | 상태 | 저장 테이블 |
| --- | --- | --- |
| 관절 각도·토크 | 지금 기록 (토크는 한계값) | `joint_states` |
| 객체 포즈 | 지금 기록 (이동만, 회전은 0) | `samples` |
| 마찰 계수 | 지금 기록 (재질별 무작위화) | `samples`, `contacts` |
| 질량 중심·관성 텐서 | 지금 기록 (박스 해석식) | `samples` |
| 충돌·접촉점 | 지금 기록 | `contacts` |
| 포인트 클라우드·표면 법선 | 다음 단계 (해석적 샘플링) | `point_cloud` (스키마 정의됨) |
| RGB, 깊이, 분할 마스크 | 렌더러 필요 | `vision` (스키마 정의됨) |
| 촉각·압력 맵 | 촉각 모델 필요 | `tactile` (스키마 정의됨) |
| SigLIP2 임베딩 | 비전 파이프라인 단계 | `embeddings` (스키마 정의됨) |

준비도 판정은 `hand-data/src/layers.rs`의 `Layer::readiness`와 같습니다.

## 6. 실데이터 계획

실데이터 수집은 아직 하지 않았습니다. 개발 초기에는 합성 데이터와 위 논문의 수치로 진행하고, 실험 단계에서 물류 현장 데이터(사내 공개 데이터 또는 직접 촬영)를 같은 Lance 스키마로 추가합니다.
