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
| 손 패드 · 지면과 아이템의 마찰 (v0.6) | `hand_core::act::material::{pad_on, item_on}` | 패드–나무 0.60, 돌·철광석 0.83, 석탄 0.45 (산포 10%); 아이템–지면 0.41–0.62 (정지·운동, 산포 10%) | CTU Code 2014, Nikolić 2024, Teffo & Naudé 2013, NAVFAC DM-7.02, Linthorne & Cooper 2013, Fall 2014 (`docs/research/actions.md` §1.2–1.3). v0.5의 `ITEM_FRICTION` 0.6(재질 구분 없음)을 대체. 석탄 패드 마찰과 산포는 추정 |
| 행위 시계열 (v0.6) | `hand_core::act` | 굽힘(관절 범위·토크), 잡기(조준·쥠 안전계수 2·연질 접촉·충돌·들기 관성), 놓기(낙하·반발), 밀기(한계면·안정 반경), 걷기(LIPM·미끄럼·넘어짐), 주행(견인–미끄럼) | `docs/research/actions.md` §1. 관절 범위, 착지 속도, 범퍼 강성, 일어나는 시간 6 s 등은 추정 |
| 채취 속도 | `World::mining_rate` | 손가락 수 / 5 (5지 = 1.0) | 가정 |
| 노드 경도 | `NodeSpec::hardness` | 나무 2.0, 바위 1.5, 석탄·철 3.0 | Minecraft 관례를 따른 값. "도구 필요" 규칙은 두지 않음 |
| 노드 드롭 수 | `NodeSpec::drop_count` | 나무 3, 바위 2, 석탄 광석 2, 철 광석 1 | 가정 |
| 아이템 크기·질량 | `ItemSpec` | 원목 0.15 m 정육면체 2.4 kg(밀도 약 700 kg/m³), 돌 0.10 m 2.6 kg, 석탄 0.10 m 1.4 kg, 철광석 0.10 m 4.0 kg | 일반적인 밀도에서 계산한 가정값 |
| 적재 한계 | `BaseKind::carry_capacity_kg` | 바퀴 20, 메카넘 15, 트랙 30, 사람 발 25 kg | 가정. v0.3부터는 부품 프리셋의 지지 한계(다리 연속 토크, 섀시 구조)로 더 낮아질 수 있음 |
| 배터리 용량 | 가슴 프리셋 `battery_wh` | 보급 160, 중급 324, 고급 420 Wh | 21700 셀 팩 계산(`docs/research/parts.md` §3). 프리셋이 없을 때의 옛 가정(순항 전력 × 1800 s)은 `body::compute`에만 남음 |
| 관절 마모 (v0.5) | `hand_core::wear` | 감속기 정격 수명의 Miner 누적: 평기어 서보 10만 회 @ 정지 토크 1/5 (STS3215 시험), 유성 QDD 7,000 h @ 연속 토크 (Harmonic Drive CSF 대입), 사이클로이드 6,000 h (Nabtesco RV) | 데이터시트, 대입은 `docs/research/effects.md` §2.2. v0.4의 "채취 타격 400회" 가정을 대체 |

### 부품 프리셋과 밸런스 (v0.3)

액추에이터·배터리·섀시 값은 데이터시트에서 왔고 출처는 [`docs/research/parts.md`](../docs/research/parts.md)에 있습니다. 사람 기준값(N:N 결합)은 [`docs/research/balance.md`](../docs/research/balance.md)에 있습니다. 아래는 그 사이를 메우는 **가정값**입니다.

| 값 | 위치 | 기본값 | 근거 |
| --- | --- | --- | --- |
| 로봇 기하 | `parts::{HIP_M, WAIST_M, SHOULDER_M, ...}` | 고관절 0.58, 허리 0.70, 어깨 1.00 m, 위팔 0.22, 아래팔 0.20 m | 1.2 m 로봇 추정(research §5) |
| 보폭 폭 · 섀시 폭 | `STANCE_WIDTH_M`, `CHASSIS_WIDTH_M` | 0.18 m, 0.40 m | 추정 |
| 무릎 레버 | `KNEE_LEVER_WALK_M`, `step_knee_lever` | 걷기 0.085 m, 턱 `L·(0.10+0.40·h/L)` | 무릎 크기 규칙(research §5), 18 cm 계단 자료(Lin 2004)에서 비례 |
| 무릎 안전계수 | `KNEE_SAFETY` | 1.5 | 연구 보고서의 크기 규칙 |
| 반응 지연 | `REACTION_DELAY_S` | 0.10 s | 사람 전기기계 지연 80–100 ms |
| 유지하는 안정 여유 | `MARGIN_KEEP` | 지지면 반길이의 20% | 설계 선택(로봇 ZMP 여유 표준 없음) |
| 발목 전략 범위 | `Body::ankle_band` | 발 길이 10% (바퀴형 2 cm) | 추정(Horak & Nashner의 발목/고관절 전략을 단순화) |
| 섀시 구조 적재 | `drive_limits` | 보급 8, 중급 30, 고급 100 kg | 고급은 AgileX Tracer 사양, 나머지는 추정 |
| 지면별 마찰 · 구름 저항 · 보행 에너지 (v0.5) | `hand_core::surface::SURFACES` | 지면 8종 × 발·바퀴·트랙 (예: 마른 모래 μ 0.55, Crr 바퀴 0.30, η 2.1) | Wong, Gillespie, Bosch, Soule & Goldman, Lukić 등 (`docs/research/effects.md` §1). 풀밭의 발·트랙 μ, 젖은 모래의 발 μ·η, 트랙 Crr 일부, 메카넘 계수 0.8은 추정. v0.4의 단일 상수 0.03 · 0.6을 대체 |
| 보행 전단 | `surface::GAIT_RCOF` | 0.223 | 사람 보통 속도 RCOF (Chang 2013), 로봇 발에 대입 |
| 정지 거리 | `STOP_DISTANCE_M` | 0.5 m (한 셀) | 설계값 |
| 관절 열 시정수 | `Actuator::thermal_tau_s` | 버스 서보 200 s, QDD 600–900 s | qdd100 자료로 맞춘 추정, 서보는 공개값 없음 |
| 두 팔 들기 시간 | `TWO_ARM_LIFT_FACTOR` | 1.5배 | 가정 |
| 손 질량 | `parts::hand_mass_kg` | 2지 0.35 … 5지 0.60 kg | 연구 보고서의 그리퍼 질량에서 추정 |
| 화물칸 위치 | `CARGO_X_M`, `CARGO_Z_M` | 배 앞 0.12 m, 허리 아래 0.10 m | 가정 |
| 피드포워드 명목 모델 | `Body::nominal_shift` | 팔 = 체중 5%, CoM 53% (Winter) | 사람 비율. 로봇과 다른 만큼을 SDS `ff` 보정이 배움 |

### ML 대역 모델 (가중치가 없을 때)

| 값 | 위치 | 기본값 | 근거 |
| --- | --- | --- | --- |
| 깊이 대역 축척 | `ml::STANDIN_DEPTH_SCALE` | 0.8 | 고른 값(실내 학습 모델의 실외 축척을 모름). 삼각 측량이 이 값을 찾아야 함 |
| 깊이 최대값 | `ml::TIPS_MAX_DEPTH_M` | 10 m | TIPSv2 DPT 구간 범위(0.001–10 m) |
| 깊이 패치 오차 | `DepthStandIn::patch_sd` | 패치마다 곱셈 오차 6% | 가정 |
| 의미 위치 대역 | `LocateStandIn` | 19×13 패치, 잡음 0.04, 닮은꼴 누출 35% | 가정 |
| 프롬프트 하한 | `mlsds::default_floor` | 대역 0.35, SigLIP2 0 | 가정. SDS 기준선(평균 − 2σ)이 생기면 그것을 씀 |
| 삼각 측량 최소 시차 | `TriCfg::min_parallax_deg` | 6° (계획은 ×1.4) | 오차 모델 `σ_r ≈ r·σθ·√2 / sin γ`에서 거리 15% 이하가 되도록 고름 |
| 방위 잡음 | `centroid_sigma(.., 2 px)` | 마스크 중심 2 px | 가정 |
| 가까운 거리 판단 | `DecideCfg::close_range_m` | 1.6 m | 가정(가까우면 깊이 거리로 바로 판단) |

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
