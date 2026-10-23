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

## 4. 비전 모델과 시계열 모델

- SigLIP 2 (Tschannen et al., arXiv:2502.14786). NaFlex 변형을 씁니다. 초기 개발은 HF 캐시에 있는 `google/siglip2-base-patch16-naflex`의 구조를 따릅니다. 가중치 파일(`model.safetensors`)은 캐시에 없어서, 실행 전에 받아야 합니다.
- Granite TTM-R3 (IBM, 시계열 예측). 월드의 정밀도 게이트가 라벨별 점수 이력을 예측하는 데 씁니다. 가중치는 `models/granite-timeseries-ttm-r3`(또는 그 아래 `52-16-dec-52-r3`)나 `TTM_DIR`에 둡니다. 이 저장소의 테스트는 **무작위(또는 0) 가중치**로 연결과 출력 형태만 확인했고, 실제 가중치로 예측 품질을 확인한 적은 없습니다.
- `world_lexicon.json`은 월드 명령(채취·줍기·운반)과 인식 문구의 단어사전입니다. logis-center `bias.json`의 노드 형식(`semantic`, `bias`, `prejudice`, `exact_match`)을 따릅니다. 한 항목의 문구는 명령 해석과 SigLIP2 앵커에 같이 쓰입니다.

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

## 7. 월드 시뮬레이터의 가정값

`world-sim`, `hand-core::body`, `hand-runtime`의 숫자 대부분은 **근거 없는 가정값**입니다. 실제 로봇의 측정값이 아니므로 결과(소요 시간, 에너지, 성공·실패)를 실제 성능으로 해석하면 안 됩니다. 아래 표의 값은 코드의 상수와 같으며, 바꿀 때는 이 표도 같이 고칩니다.

| 값 | 위치 | 기본값 | 근거 |
| --- | --- | --- | --- |
| 틱 길이 | `world::TICK_S` | 0.1 s | 설계값 |
| 셀 크기 | `terrain::CELL_M` | 0.5 m | 설계값 |
| 카메라 높이 | `world::EYE_HEIGHT_M` | 0.5 m | 가정 |
| 채취·줍기·인도 도달 거리 | `MINE_REACH_M`, `PICKUP_REACH_M`, `DELIVER_REACH_M` | 0.9 / 0.7 / 1.6 m | 가정 |
| 줍기·인도에 걸리는 시간 | `PICKUP_TIME_S`, `DELIVER_TIME_S` | 각 1.0 s | 가정 |
| 조작 전력(채취·줍기·내려놓기) | `MANIPULATION_POWER_W` | 30 W | 가정 |
| 제자리 회전 속도 | `TURN_RATE_RAD_S` | 2.5 rad/s | 가정 |
| 아이템 마찰 계수 | `ITEM_FRICTION` | 0.6 | 가정 (재질별 값 아님) |
| 채취 속도 | `World::mining_rate` | 손가락 수 / 5 (5지 = 1.0) | 가정 |
| 노드 경도 | `NodeSpec::hardness` | 나무 2.0, 바위 1.5, 석탄·철 3.0 | Minecraft 관례를 따른 값. "도구 필요" 규칙은 두지 않음 |
| 노드 드롭 수 | `NodeSpec::drop_count` | 나무 3, 바위 2, 석탄 광석 2, 철 광석 1 | 가정 |
| 아이템 크기·질량 | `ItemSpec` | 원목 0.15 m 정육면체 2.4 kg(밀도 약 700 kg/m³), 돌 0.10 m 2.6 kg, 석탄 0.10 m 1.4 kg, 철광석 0.10 m 4.0 kg | 일반적인 밀도에서 계산한 가정값 |
| 적재 한계 | `BaseKind::carry_capacity_kg` | 바퀴 20, 메카넘 15, 트랙 30, 사람 발 25 kg | 가정 |
| 배터리 용량 | `BATTERY_RUNTIME_S` | 순항 전력 × 1800 s | 가정 |
| 손 관절 내구 | `JOINT_ENDURANCE_STRIKES` | 채취 타격 400회 | 가정 |

### 정밀도 게이트의 임계값

`hand-runtime::config::GateCfg`와 `EmbeddingCfg`의 기본값입니다. **아직 실제 SigLIP2 점수 분포로 보정하지 않았습니다.** 모의 인식기(`OraclePerception`: 정답 ID 버퍼 + 주입한 오류)와 SigLIP 시그모이드 점수의 일반적인 범위를 보고 정했을 뿐입니다.

| 값 | 기본값 | 뜻 |
| --- | --- | --- |
| `min_score` | 0.30 | 이 점수 미만이면 거부. SDS가 접촉으로 확인된 점수를 보고 올릴 수는 있어도 내리지는 않음 |
| `min_margin` | 0.04 | 1위와 2위 라벨의 점수 차 |
| `confirm_sightings` | 2 | 같은 지점에서 이만큼 좋은 관측이 쌓여야 확정 |
| `track_radius_m` | 1.0 | 같은 지점으로 보는 반경 |
| `temporal_min_history` | 6 | 시간 게이트가 켜지는 최소 이력 |
| `adaptive_k` | 2.0 | 적응형 임계값의 표준편차 배수 |
| 보고 하한(`report_floor`) | 0.10 | 타일에서 이 점수 미만의 라벨은 아예 보고하지 않음 |
| 타일 격자 | 2×2, 4×3, 겹침 15% | 타일 상자는 거칠어서 `bbox`는 대략적인 위치입니다 |
| 데모 노이즈 | 점수 표준편차 0.05, 혼동 12%, 오탐 30%, 누락 5% | 게이트 전/후 비교가 보이도록 정한 값이며 실제 인식기의 오류율이 아님 |

게이트 전/후 정밀도 비교 수치는 이 모의 오류 모델에서 나온 것이므로 실제 모델의 개선폭이 아닙니다.
