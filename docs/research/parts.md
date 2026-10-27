# 부품 프리셋 리서치 (가성비 1.0–1.4 m 물류 로봇)

조사일 2026-10-23. `crates/noto-core/src/parts.rs`의 숫자는 이 문서에서 왔다. `[n]`은 아래 출처 목록, "추정"은 계산을 함께 적었다. 벤치마크는 돌리지 않았다.

## 핵심 결론

1. **버스 서보는 정지 토크가 약하지 않다. 약한 것은 출력(W), 연속 토크, 역구동성이다.** Feetech·Hiwonder는 피크 53–69 N·m/kg로 소형 QDD와 비슷하거나 높다. 하지만 기계 출력은 약 60–130 W/kg로 QDD(약 200–600 W/kg)보다 훨씬 낮고(추정: P ≈ τ_stall·ω₀/4), 연속 토크가 피크의 약 ⅓이며 역구동이 안 된다.
2. **버스 서보 다리로는 1.2 m 로봇을 세울 수 없다.** 보급형 전체 질량 약 7.7 kg에 무릎(HTD-45H 2개 묶음)이 동적 여유를 두면 약 7 kg까지만 버틴다. 보급형은 바퀴를 쓰거나, Berkeley Humanoid Lite식 자작 BLDC(보급형+)로 올려야 한다.
3. **가성비는 Robstride가 압도적이다**: 피크 N·m당 약 $2–9 (CubeMars·Dynamixel은 $33–76).
4. **밸런스를 가장 크게 흔드는 것은 배터리 위치가 아니라 팔 뻗음 + 하중이다.** 배터리를 가슴에서 골반으로 내리면 무게중심(CoM)이 약 2 cm 낮아질 뿐이다. 고급형 팔 두 개를 끝까지 뻗고 각 3 kg을 들면 CoM이 약 11 cm 앞으로 간다. 이것이 발목 액추에이터 크기를 정한다.

## 1. 액추에이터

| 액추에이터 | 전압 | 피크 N·m | 연속 N·m | 무부하 rad/s | g | 감속비 | ~USD | 역구동 | 출처 |
|---|---|---|---|---|---|---|---|---|---|
| Feetech STS3215-C018 | 12 | 2.94 | 0.98 | 4.7 | 55 | ~345 (미확인) | 32 | 불가 | [1] |
| STS3215-C001 | 7.4 | 1.91 | 0.64 (@6 V) | 5.5 | 55 | 345 | ~25 | 불가 | [2] |
| STS3250 | 12 | 4.90 | 1.57 | 7.9 | 74.5 | 345 | 88–97 | 불가 | [3] |
| Hiwonder HTD-45H | 11.1 | 4.41 | ~1.5 (추정, Feetech처럼 ⅓) | 5.8 | 64 | 미공개 | 30 | 불가 | [4] |
| Dynamixel XL430-W250 | 11.1 | 1.4 | ~0.28 | 6.0 | 57.2 | 258.5 | 27.5–44.9 | 불가 | [6][9][10] |
| Dynamixel XM430-W350 | 12 | 4.1 | ~0.82 | 4.8 | 82 | 353.5 | ~310 | 불가 | [7][9][10] |
| Dynamixel XH540-W270 | 12 | 9.9 | ~2.0 | 4.1 | 165 | 272.5 | ~632 | 불가 | [8][9][10] |
| CubeMars AK60-6 V1.1 | 24 | 9 | 3 | 33.5 | 368 | 6 | 299 | 가능 | [11] |
| CubeMars AK70-10 | 24/48 | 24.8 | 8.3 | 25/50 | 521 | 10 | 399 | 가능 | [12] |
| CubeMars AK80-9 | 48 | 18 | 9 | 55.8 | 485 | 9 | 480 | 가능 | [13] |
| Unitree GO-M8010-6 | 24 | 23.7 | 미공개 | 30 | 530 | 6.33 | 280 | 가능 | [14] |
| Robstride 01 | 36 | 17 | 6 | 33.0 | 380 | 7.75 | 130 | 가능 | [15] |
| Robstride 02 | 48 | 17 | 6 | 42.9 | 405 | 7.75 | 145 | 가능 | [15] |
| Robstride 03 | 48 | 60 | 20 | 20.4 | 880 | 9 | 225 | 가능 | [16] |
| Robstride 04 | 48 | 120 | 40 (대형 방열판) | 20.9 | 1420 | 9 | 255 | 가능 | [17] |
| MyActuator X8-P6-20 | – | 20 | 8 | 23.0 | 780 | 6 | – | 가능 | [19] |
| mjbots qdd100 beta 3 | 36 | 16 (<1 s) | 3.3 상시, 10 (60 s), 6 (400 s) | 62.8 | 507 | – | 879 | 가능 | [20] |
| 자작 BHL 6512 / 5010 | 24 | ~20 / ~12 | 미공개 | 미공개 | 미공개 | – | 157–188 / 94–136 | 가능 | [22] |

**열 모델 (추정)**: 발열 상태 h에 대해 dh/dt = ((τ/τ_cont)² − h)/T_th, h > 1이면 관절을 감액한다. qdd100의 두 점(10 N·m 60 s, 6 N·m 400 s)을 맞추면 0.5 kg QDD의 T_th ≈ 500–1100 s. 버스 서보는 열용량이 작아 100–300 s로 가정(공개 자료 없음). 토크-속도: τ(ω) = τ_peak·(1 − |ω|/ω₀).

## 2. 전체 로봇 기준 사례

| 로봇 | 키 (m) | 질량 (kg) | 자유도 | 가반 | 속도 | 배터리 | 비용 (USD) | 출처 |
|---|---|---|---|---|---|---|---|---|
| ToddlerBot | 0.56 | 3.4 | 30 | 1.48 kg (자중 40%) | – | 2 Ah LiPo 215 g, 걷기 시간당 75 Wh | <6,000 | [23] |
| Berkeley Humanoid Lite | 0.8 | 16 | 22 | – | – | 6S 4 Ah (~89 Wh 추정), ~30분 | 4,312 | [22] |
| K-Scale K-Bot | 1.4 | 34 | 20 | 10 kg (주장) | – | 최대 4 h | 8,999 | [26] |
| Unitree G1 | 1.32 | ~35 | 23 | 팔 ~2 kg | ~2 m/s | 421 Wh, ~2 h | 13,500 | [27] |
| SO-101 팔 | – | ~0.8 | STS3215 ×6 | ~0.5 kg | – | – | – | [29] |
| LeKiwi 베이스 | – | 1.7 | STS3215 ×3, 100 mm 옴니휠 | – | ~0.24 m/s (추정) | 12 V | 226 | [30] |
| Hello Robot Stretch 3 | 1.41 | 24.5 | – | 2 kg, 0.52 m 도달 | 베이스 ~0.3 m/s | 2–5 h | – | [31] |
| AgileX Scout Mini | – | 26 | 4× 150 W | 10 kg | 2.7 m/s | 360 Wh | ~4,500 | [32] |
| AgileX Tracer | – | 30 | 2× 150 W | 100 kg | 1.6 m/s | 24 V 15 Ah | 7,590 | [33] |

G1 관절 토크 한계(MuJoCo Menagerie [27]): 고관절 피치·요 88, 고관절 롤·무릎 139, 발목 50, 허리 요 88, 어깨·팔꿈치 25 N·m. 링크 질량: 골반 3.81, 몸통 7.82, 다리 7.19, 팔 3.52 kg.

**손·그리퍼**: SO-101 집게(STS3215 1개) ~0.13 kg, ~21 N 피크 / ~7 N 연속(추정); Yale T42 400 g ~10 N; LEAP Hand 595 g, 19.5 N 인출력, ~$2,000 [34]; Inspire RH56DFX 540 g, 손가락 10 N, $4,500+ [35]; Robotiq 2F-85 0.9 kg, 20–235 N, 5 kg [37]. 물류에는 평행 집게가 다관절 손보다 낫다.

## 3. 배터리·연산·센서

- 21700 셀: Samsung 50E 17.8 Wh / 69 g ≈ 258 Wh/kg [38], Molicel P45B ≈ 231 Wh/kg [39]. 팩은 160–220 Wh/kg(추정, 케이스·BMS 15–25%). ToddlerBot 실제 팩 224 Wh/kg [23].
- LiFePO4 12.8 V 100 Ah 팩: 95–150 Wh/kg [40]. 바퀴형에는 괜찮고 다리형에는 무겁다.
- Raspberry Pi 5: 45–62 g, 대기 2.7–3.6 W, 최대 부하 8.8 W [41]. Jetson Orin Nano Super: 175 g, 7/15/25 W, $249 [42].
- RealSense D435i ~72 g, ~0.5–1.5 W [43]. OAK-D Lite 61 g, 깊이 파이프라인 약 4 W [44].

## 4. 프레임 재료

| 재료 | 밀도 | E (GPa) | 단면 | m당 질량 | 상대 굽힘 강성 (추정) |
|---|---|---|---|---|---|
| PLA (출력) | 1.24 | 2.3–2.4 [45] | 40×34 mm 관 | 0.43 kg | Al 관의 ≈0.75× |
| PETG (출력) | 1.27 | 1.5–1.6 [45] | 40×34 mm 관 | 0.44 kg | ≈0.5× |
| 알루미늄 6061 관 | 2.70 | 69 | 20×18 mm | 0.16 kg | 1.0 |
| 2020 알루미늄 압출재 | – | 69 | 20×20 | 0.50 kg [46] | ~2.5× |
| 카본 관 | – | ~50–70 | 20×18 mm | 68 g [47] | Al 관 질량 45%에 ≈0.7–1.0× |

중급·고급에서는 액추에이터가 팔다리 질량의 60–80%라 프레임 재료는 질량보다 강성과 말단 관성을 정한다. 보급형에서는 서보가 55–75 g뿐이라 PLA가 주 질량이다.

## 5. 부위별 프리셋 (1.2 m 기준 로봇)

기준 기하(추정): 고관절 0.58 m, 허리 0.70 m, 어깨 1.00 m, 위팔 0.22 m, 아래팔 0.20 m. 부위 질량 = 인용한 액추에이터 질량 + 추정 구조 질량.

| 부위 | 등급 | 구성 | 질량 kg | 부모 관절 기준 CoM | 피크/연속 N·m | 가반 | 비용 USD | 밸런스상 장단점 |
|---|---|---|---|---|---|---|---|---|
| 머리 | 보급 | STS3215 ×2 + OAK-D Lite, PLA | 0.45 | 목 위 0.07 | 2.94 / 0.98 | – | ~215 | 가볍다. 정적 필요 토크 ~0.3 N·m |
| 머리 | 중급 | STS3215 ×2 + D435i, 알루미늄 | 0.44 | 0.07 | 2.94 / 0.98 | – | ~415 | QDD 목은 이득 없음 |
| 머리 | 고급 | XM430 ×2 + D435i + RGB, 카본 | 0.45 | 0.07 | 4.1 / 0.82 | – | ~1,050 | 백래시가 작다 |
| 가슴 | 보급 | Pi 5, PLA + 2020 척추 | 1.2 (+0.72 배터리) | 허리 위 0.15 | – | – | ~150 | 3S3P 21700 = 160 Wh |
| 가슴 | 중급 | Orin Nano Super | 1.6 (+1.6) | 0.16 | – | – | ~410 | 10S2P P45B = 324 Wh |
| 가슴 | 고급 | Orin 급, 카본/알루미늄 | 1.5 (+~2.5) | 0.17 | – | – | ~1,500 | 13S ~420 Wh (G1과 비슷) |
| 골반·허리 | 보급 | HTD-45H ×2 묶음 | 0.7 | 허리 아래 0.06 | 8.8 / ~2.9 | – | ~70 | 요(yaw)만. 피치 관절은 연속 ≥12 N·m 필요 |
| 골반 | 중급 | Robstride 02 | 1.3 | 0.06 | 17 / 6 | – | ~185 | 배터리 두기 좋은 곳 |
| 골반 | 고급 | Robstride 03 (+허리 피치) | 1.9 | 0.06 | 60 / 20 | – | ~285 | 몸통 기울임 유지: 12 kg × 0.3 m × sin 20° ≈ 12 N·m |
| 팔 (한쪽) | 보급 | 어깨 HTD-45H ×2, 팔꿈치 STS3215 | 0.64 | 0.17 | 팔꿈치 2.94 / 0.98 | 0.25 kg 상시, 끝까지 뻗어 0.6 kg 순간 | ~135 | 수평으로 뻗은 자기 팔을 계속 못 버틴다 |
| 팔 | 중급 | 어깨 RS02, 팔꿈치 RS01, 알루미늄 | 2.1 | 0.11 | 17 / 6 | 2.0 kg 상시(팔꿈치 90°), ~3 kg 피크 | ~600 | G1(~2 kg)과 비슷 |
| 팔 | 고급 | 어깨 RS03, 팔꿈치 RS02, 카본 | 3.0 | 0.08 | 어깨 60/20, 팔꿈치 17/6 | ~2 kg 상시, ~7 kg 피크 | ~800 | 팔꿈치를 RS03으로 올리면 상시 7 kg, 대신 +0.48 kg |
| 다리 (한쪽) | 보급 | HTD-45H ×10 | 1.45 | 고관절 아래 0.23 | 무릎 8.8 / ~2.9 | 로봇 ≤3.5 kg 상시 | ~315 | 7.7 kg 보급형 로봇에는 약함 → 바퀴 |
| 다리 | 보급+ | BHL 6512 / 5010 | ~3 | ~0.15 | ~20 / 미공개 | 로봇 ≤16 kg | ~970 | 자작, 60 h 내구 시험 |
| 다리 | 중급 | 무릎 RS03, 발목 RS02 ×2 평행 링크 | 4.8 | 0.14 | 무릎 60/20, 발목 34/12 | 로봇 ≤24 kg 상시 | ~1,190 | 발목 모터를 위에 달아 흔들림 관성 작게 |
| 다리 | 고급 | 무릎 RS04, 발목 RS03 ×2 | 7.1 | 0.14 | 무릎 120/40, 발목 120/40 | 로봇 ≤45 kg 상시 | ~1,510 | 10 kg 운반·계단 여유 |
| 발 | 보급 / 중급 / 고급 | PLA+TPU 0.16×0.08 / 알루미늄+힘센서 0.20×0.10 / 0.22×0.10 | 0.10 / 0.30 / 0.6 | 0.04 | – | – | 20 / 30 / 150 | – |

다리 크기 규칙(추정, BHL·G1로 확인): 무릎 피크 토크 ≥ 1.5·M·g·d, 보행 중 무릎 굴곡에서 d ≈ 0.085–0.09 m.

**등급 합계 (추정)**: 보급 7.7 kg / CoM 0.62→0.60 m(배터리 가슴→골반) / 보행 0.1–0.3 m/s / ~$1.5k; 중급 20.1 kg / 0.62→0.60 / 0.5–1.0 m/s / ~$5.6k; 고급 29.6 kg / 0.61→0.59 / 1.0–1.5 m/s / ~$8.2k. Froude 수 0.02 / 0.15 / 0.3–0.5, 다리 길이 0.58 m. 비교: K-Bot 34 kg $9k, G1 35 kg $13.5k.

## 6. 조합 판단

- **배터리 골반 vs 가슴**: 골반은 CoM을 ~2 cm만 낮춘다. 진짜 이득은 몸통 피치 관성의 배터리 몫을 약 6배 줄여(2.5 kg × 0.15² → 0.06²) 허리 액추에이터를 작게 할 수 있다는 점 [48]. 가슴 뒤쪽 배터리는 앞으로 뻗는 팔의 평형추가 된다.
- **무거운 QDD 팔**: 고급형 두 팔을 끝까지 뻗고 각 3 kg → CoM이 ~0.11 m 앞으로, 0.22 m 발 앞쪽 여유의 약 80%를 쓴다. 발목 토크 합 ~39 N·m 필요 → RS02 발목(쌍 연속 12 N·m)은 부족하므로 고급형은 RS03 ×2. 중급형 2 kg 팔을 몸 가까이 → CoM ~0.06 m, 다리당 ~7.6 N·m로 RS02 발목에 맞는다. 규칙: 팔의 총 하중 능력이 발목이 버티는 토크를 넘으면 안 된다. 넘으면 발을 내딛거나 골반을 뒤로 빼야 한다.
- **좋은 혼합**: 보급형 팔·머리 + 바퀴 베이스(SO-101 형 물류 로봇), 중급 다리 + 고급 어깨(가반은 늘지만 발목도 한 등급 올려야 함).
- **피할 조합**: 7 kg 넘는 로봇에 버스 서보 다리, 하중을 받는 Dynamixel 관절(피크:연속 ~5:1), 발목의 보급형 서보(흔들림 관성), 무거운 집기에 다관절 손.

## 7. 형태 (v1.0): 하체 × 몸통 × 머리·목 × 조작기 × 적재

조사일 2026-10-27. `crates/noto-core/src/morph.rs`와 `parts.rs`의 형태 숫자는 이 절에서 왔다. 표기: **[직접]** 출처에 인쇄된 값, **[유도]** 계산, **[추정]** 출처를 찾지 못해 정한 값, **[대입]** 다른 로봇의 값을 빌림. 연구 팩은 `robot-morphologies` (엔진 입력 = Group A, 벤치마크 = Group B).

### 7.1 요청과 반영

| 요청 (R6) | 반영 |
|---|---|
| 바퀴 하체에 걷는 사람의 카메라가 붙어 있음 | 카메라 위치를 형태가 정한다 (`HeadMount`): 목 위 머리, 몸통에 고정된 머리(목 없음), 마스트 카메라(팬·틸트 있음/없음), 차체 앞 카메라, 없음. `kin::camera`가 이것을 따르고, `kin::head_camera`는 같은 함수를 부르는 이름으로 남겼다. 세계·화면의 카메라 흔들림은 다음 묶음(카메라)이 이 위치를 쓴다 |
| 상체에 목이 있거나 없음 | `HeadMount::Fixed { tilt_deg }`: 목 관절, 목 서보 두 개의 질량, 시선 안정화가 없다. 카메라는 몸통과 고정 틸트로만 기운다 |
| 바퀴 + 몸통 없음 + 집게만 | `deck_gripper`(갑판 위 집게 팔, 차체 카메라, 토트), `mast_gripper`(마스트 리프트에 탄 집게 팔, 마스트 카메라, 토트) |
| 집게가 없는 운반 트럭 (휴머노이드가 짐을 실어 줌) | `truck`: 조작기 없음, 차체 카메라, 짐칸(`Cargo::Bed`). 집기·내려놓기·들기는 `no_manipulator`로 바로 거절한다. 다른 로봇이 짐칸에 실어 주는 흐름은 다음 묶음(여러 로봇)이 한다 |
| 여러 층의 조합 | `Morphology { lower, torso, head, manip, cargo }` 다섯 층. 하체는 구동(`NotoConfig.base`)을 따른다. 만들 수 없는 조합과 범위 밖 크기(고정 머리 틸트, 마스트·차체 카메라 높이, 짐칸 크기)는 `Morphology::validate`가 한국어로 이유를 말하고, `Morphology::fit`이 가장 가까운 조합으로 고친다 (`Body::new`는 고친 형태로 만들고, `Body.plan`도 그 형태라서 `plan.id()`가 실제로 만든 로봇을 가리킨다). `BodyPlan::validate`는 여기에 더해, 팔이 자기 토트·짐칸 위로 손을 가져가지 못하는 계획을 거절한다 |

기본값은 오늘의 휴머노이드다. 그래서 `morph`가 없는 계획(예전 JSON 포함)은 예전과 같은 로봇을 마지막 자리까지 같게 만든다. `BodyPlan::id()`도 `mid` 그대로이고, 다른 형태만 `mid+truck`처럼 뒤에 붙는다 (SDS 열쇠 유지).

### 7.2 층과 이름 붙은 형태

| 층 | 선택 |
|---|---|
| 하체 `Lower` | 다리 · 바퀴 · 메카넘 바퀴 · 트랙 (`BaseKind`와 1:1, `Lower::from_base`) |
| 몸통 `Torso` | 휴머노이드(골반·허리·가슴, 오늘) · 마스트 기둥(어깨와 머리를 실은 캐리지, 몸통 리프트로 내려감; Fetch·TIAGo 형) · 없음(갑판) |
| 머리 `HeadMount` | 목(팬·틸트, 오늘) · 고정(`tilt_deg`) · 마스트 카메라(`height_mm`, `pan_tilt`) · 차체 앞(`height_mm`) · 없음 |
| 조작기 `Manip` | 두 팔(손, 오늘) · 한 팔(손) · 집게 팔 하나 · 없음 |
| 적재 `Cargo` | 배 앞 적재함(오늘) · 짐칸(`len/width/height_mm`, `payload_kg`) · 갑판 토트 · 없음 |

크기는 정수 mm·kg라서 `BodyPlan`이 `Copy + Eq`를 그대로 유지한다.

| 형태 | 하체 | 몸통 | 카메라 | 조작기 | 적재 | 실제 원형 |
|---|---|---|---|---|---|---|
| `legs_humanoid` 두 다리 휴머노이드 | 다리 | 휴머노이드 | 목 | 두 팔 | 적재함 | 오늘의 다리형 (§2의 G1·K-Bot 급) |
| `wheel_humanoid` 바퀴 휴머노이드 | 바퀴 | 휴머노이드 | 목 | 두 팔 | 적재함 | 오늘의 기본 로봇 |
| `track_humanoid` 트랙 휴머노이드 | 트랙 | 휴머노이드 | 목 | 두 팔 | 적재함 | 오늘의 트랙 로봇 |
| `wheel_noneck` 목 없는 바퀴 휴머노이드 | 바퀴 | 휴머노이드 | 고정, 15° 아래 | 두 팔 | 적재함 | 목 자유도 없는 머리 (Unitree H1 형; 연구 팩도 미확인으로 표시) |
| `mast_gripper` Stretch 형 | 바퀴 | 없음 | 마스트 1.31 m, 팬·틸트 | 집게 팔 하나 (마스트 리프트) | 토트 | Hello Robot Stretch RE1 [49] |
| `deck_gripper` LeKiwi 형 | 바퀴 | 없음 | 차체 앞 0.20 m | 집게 팔 하나 (갑판) | 토트 | LeKiwi 베이스 + SO-101 팔 [29][30] |
| `truck` 운반 트럭 | 바퀴 (트랙도 됨) | 없음 | 차체 앞 0.25 m | 없음 | 짐칸 0.50 × 0.40 m, 0.25 m 높이, 100 kg | AMR·카트 운반 로봇: MiR250 [51], AgileX Tracer [33] (Amazon Proteus는 공개 수치가 없음) |

### 7.3 Group A 엔진 입력

| 상수 (`morph.rs`·`parts.rs`) | 값 | 구분 | 출처 |
|---|---|---|---|
| `STRETCH_CAMERA_H_MM` 마스트 카메라 높이 | 1310 mm | [직접] Stretch RE1 머리 컬러 카메라 바닥 위 131 cm | Kemp 외 2022 [49] §II |
| `MAST_LIFT_TOP_M` 마스트 캐리지 꼭대기 | 1.10 m | [직접] Stretch RE1 리프트 0–110 cm | [49] §II |
| `MAST_LIFT` 행정 | 꼭대기에서 갑판 팔 어깨 높이(0.26 m)까지 = 0.84 m | [유도] | — |
| `MAST_LIFT` 속도·힘 | 0.10 m/s · 450 N | [대입] `TORSO_LIFT`(Fetch 몸통 리프트) | §5 `TORSO_LIFT` |
| `MAST_LIFT` 질량·가격 | `lift_mass_kg(행정)` = 1.0 + 1.4·행정 + 0.9 kg (0.84 m에서 3.08 kg) · $600 | [추정] 벨트 축 1.0 kg + 100 mm당 0.14 kg + 모터·브레이크 0.9 kg (`TORSO_LIFT` 2.5 kg과 같은 추정; 행정이 짧은 마스트는 그만큼 가벼움) | — |
| `MAST_TILT_MAX_DEG` 마스트 팬·틸트의 아래 한계 | 83° | [직접] Fetch `head_tilt_joint` 상한 1.45 rad | fetch_description URDF [50] |
| 마스트 팬 범위 | ± 90° | [직접] Fetch `head_pan_joint` ± 1.57 rad (목과 같은 값) | [50] |
| `AMR_CAMERA_H_MM` 트럭 카메라 | 250 mm | [추정] MiR250 몸체 높이 300 mm, 3D 카메라는 앞쪽 아래 | MiR250 사양 [51] |
| `DECK_CAMERA_H_MM` 갑판 카메라 | 200 mm | [추정] 0.16 m 갑판의 앞 가장자리 위 0.04 m | — |
| `FIXED_HEAD_TILT_DEG` 고정 머리 틸트 | 15° | [추정] 목으로 내려다볼 수 없으니, 1.14 m 눈에서 바닥이 약 0.9 m 앞부터 보이게 (렌더 카메라: 세로 시야 50°, 자체 12° 아래 피치 → 수평 머리는 1.5 m 앞부터; 기하 계산) | — |
| `TRUCK_BED` 정격 | 100 kg | [직접] AgileX Tracer 가반 | [33] |
| `TRUCK_BED` 크기·높이 | 0.50 × 0.40 m, 0.25 m | [추정] MiR250 몸체 높이 300 mm 아래 | [51] |
| `BED_KG_PER_M2` 짐칸 판 | 8.1 kg/m² | [유도] 알루미늄 3 mm × 2.70 g/cm³ (§4), 기둥·난간은 뺌 | §4 |
| `EXTRUSION_KG_PER_M` 카메라 기둥·마스트 기둥 | 0.50 kg/m | [직접] 2020 알루미늄 압출재 | [46] |
| `POLE_MOUNT_KG` 기둥 클램프·카메라판 | 0.20 kg | [추정] | — |
| `CHASSIS_CAMERA_KG` 차체 카메라 | 0.10 kg | [추정] D435i 약 72 g [43] + 브래킷 | [43] |
| 고정 머리 질량 (`mounted_head`) | 머리 프리셋 − 목 서보 2개 | [유도] §5 머리 프리셋 = 서보 2개 + 카메라 | §5 |
| 몸통 없는 로봇의 컴퓨터 `computer_kg` | 보급 Raspberry Pi 5 0.054 kg (45–62 g의 가운데), 중급 Jetson Orin Nano Super 0.175 kg, 고급 "Orin 급"도 0.175 kg | [직접] / 고급은 [대입] | [41][42] |
| 전자 상자 `DECK_BOX_KG`, 위치 `ELECTRONICS_Z_M` | 0.30 kg, 차체 안 (중심 0.11 m, 차체 상자의 가운데) | [추정] AMR·모바일 매니퓰레이터는 배터리와 컴퓨터를 베이스 안에 둔다. 갑판 위에 두면 토트·팔 받침·기둥 자리와 겹친다 | — |
| 갑판 팔 받침 `DECK_ARM_BASE_HALF_M` | 0.06 m 폭, 갑판에서 어깨까지 (그리기만; 질량은 팔 프리셋) | [추정] SO-101 급 팔의 받침 | — |
| 범위 `FIXED_TILT_RANGE_DEG` | 고정 머리 틸트 −45…60° | [추정] 위로 45°면 렌더 카메라(세로 50°, 자체 12° 아래)에 바닥이 없고, 아래 60°면 자기 앞만 보임 | — |
| 범위 `MAST_CAMERA_MAX_MM` | 마스트 카메라 2000 mm 이하 | [추정] Fetch 머리 최고 1.447 m [50] 위 | [50] |
| 범위 `CHASSIS_CAMERA_RANGE_MM` | 차체 카메라 50–600 mm | [추정] 지면 위 ~ MiR250 몸체 높이 300 mm의 두 배 | [51] |
| 범위 `BED_MAX_MM` | 짐칸 1000 × 800 mm, 높이 800 mm 이하 | [추정] 0.25–0.36 m 차체에 그 이상은 트레일러 | — |
| 집게 `GripperPreset` | 보급 SO-101 집게 0.13 kg, ~21 / ~7 N; 중급 XM430 2지 집게 0.35 kg; 고급 Robotiq 2F-85 0.9 kg, 20–235 N | [직접]·[추정] (§2 손·그리퍼) | [29][37] |
| 집게 조임 (`grasp::capacity_for`) | 양쪽 턱이 같은 힘 F로 누르므로 법선력 합 2·F (손가락의 관절 피크 힘과 같이 피크 값): SO-101 42 N, Robotiq 2F-85 470 N. 힘이 없는 중급 XM430 집게는 2지 손 모델(`FingerCount::Two`)의 접촉 | [유도] 평행 집게의 두 접촉면 마찰 | [29][37] |
| 갑판 높이 `DECK_Z_M` | 0.16 m (차체 상자 윗면) | [유도] `kin`의 차체 상자 | — |
| 갑판 팔 어깨 `DECK_ARM_X_M`, `DECK_ARM_BASE_H_M` | 갑판 가운데 뒤 0.06 m, 갑판 위 0.10 m | [추정] SO-101 급 팔의 어깨 피치 축이 장착판 위 약 0.1 m | — |
| 갑판 팔 쉬는 자세 `DECK_REST_DEG` | 어깨 120°, 팔꿈치 120° (갑판 위로 접음) | [추정] | — |
| 마스트 팔 어깨 `MAST_ARM_X_M` | 마스트 앞 0.05 m | [추정] 캐리지 브래킷 | — |
| `MAST_CAMERA_CLEAR_M` | 캐리지 꼭대기와 렌즈 사이 0.15 m | [추정] 팬·틸트 머리 | — |
| 마스트 기둥 `COLUMN_TOP_M` | 1.07 m (어깨 1.00 m + 목 0.07 m) | [유도] §5 기준 기하 | §5 |
| 마스트 기둥 꼭대기 `column_top_m` | 기둥 위에 마스트 카메라가 있으면 그 팬·틸트 피벗까지 (질량은 압출재 0.50 kg/m로 셈) | [유도] | [46] |
| 토트 `TOTE_SIZE_M`, `TOTE_KG` | 0.16 × 0.24 × 0.12 m, 0.4 kg | [추정] | — |
| 적재물 무게중심 `CARGO_ABOVE_FLOOR_M` | 짐칸·토트 바닥 위 0.10 m | [추정] 0.2 m 상자의 절반 | — |
| 짐칸·토트 위에서 놓는 높이 `RIM_RELEASE_M` | 테두리 위 0.20 m | [추정] | — |

### 7.4 Group B 벤치마크 출처

연구 팩의 benchmark_refs 가운데 형태에 쓸 수 있는 것이다. 엔진 파일은 이 이름을 쓰지 않는다 (`tests/morph_forms.rs`의 출처 검사). 이번 묶음은 형태별 점수 칸([MX])을 만들지 않았다. 형태 행렬 칸은 v1.0 벤치 묶음이 만든다. 아래 비교는 손으로 한 것이다.

| 출처 | 값 | 비교한 것 |
|---|---|---|
| Hello Robot, Stretch 3 Hardware Guide [31] (연구 팩 Group B) | 머리 팬 346°·틸트 115° (−25…+90°), 정격 가반 2 kg, 팔 이동 0.52 m | `mast_gripper`의 팬·틸트 범위 |
| Hello Robot, Stretch 3 전체 높이·질량 (§2 표 [31]) | 141 cm, 24.5 kg | `mast_gripper`의 카메라 높이·질량. 엔진은 Stretch 3 값을 쓰지 않는다 (엔진은 RE1 [49]) |
| Wevolver, Fetch Mobile Manipulator 사양 [52] | 높이 1096–1491 mm, 113.3 kg, 팔 가반 6 kg | (마스트 기둥 몸통 `Torso::Mast`; 이름 붙은 형태가 없어 비교하지 않음) |
| Ackerman, IEEE Spectrum 2015 (Starship) [53] | 적재 포함 20 kg 미만, 약 10 kg 적재 | `truck`의 적재 비율 |
| Konica Minolta MiR250 목록 [54] (검색 결과로만 봄) | 가반 250 kg, 2.0 m/s | `truck`의 적재 비율 (MiR250 질량 97 kg은 [51]) |

같은 141 cm·24.5 kg을 적은 업계 보도 [55]는 연구 팩에서 엔진 입력(Group A)으로 분류되어 있어 점수 출처로 쓰지 않는다 (엔진도 쓰지 않음; 맥락으로만 적음).

| 비교 | 엔진 | 벤치마크 | 차이 |
|---|---|---|---|
| 마스트 카메라 높이 vs 로봇 전체 높이 | 1.31 m (Stretch RE1 [49]) | Stretch 3 전체 141 cm [31] | 카메라가 꼭대기보다 0.10 m 아래 (Fetch URDF에서도 머리 꼭대기는 카메라 위 35–45 mm) — 맞음 |
| `mast_gripper` 질량 (중급) | 17.2 kg | Stretch 3 24.5 kg [31] | −30% (Group A인 Stretch RE1 23 kg 대비 −25%) |
| 마스트 팬·틸트 범위 | 팬 180° (± 90°), 틸트 166° (± 83°) | Stretch 3 팬 346°, 틸트 115° [31] | 팬 −48%, 틸트 +44%. 엔진은 Fetch URDF [50]의 아래 한계 83°를 위쪽에도 대칭으로 둔다 (Fetch의 위쪽 한계는 43.5°) |
| `truck` 적재 비율 = 적재 / (자중 + 적재) | 보급 54% · 중급 69% · 고급 73% | Starship ≥ 50% [53] · MiR250 72% [54][51] | 보급은 Starship과, 고급은 MiR250과 비슷 |

### 7.5 모델

- **조립**: `Body::new`는 계획의 형태를 구동에 맞춘 뒤(`Morphology::fit`) 있는 부위만 만든다. 몸통이 없으면 골반(배 프리셋)의 질량·피치 관절·기울기가 0이다. 가슴 프리셋에서는 몸통 구조(척추·외피)를 빼고 컴퓨터와 그 상자(`computer_kg` + `DECK_BOX_KG`)와 배터리만 차체 안(중심 높이 0.11 m)에 싣는다. 목이 없으면 머리 질량에서 목 서보 두 개를 빼고 시선 안정화가 0이다. 차체 카메라는 카메라만(0.10 kg)이다. 팔은 `arm_count()`만큼이고, 손끝 질량은 손(손가락 수) 또는 집게다. 짐칸·토트는 구조 질량과 함께 차체에 붙는다. 적재물은 배 앞 적재함(골반과 함께 움직임) 또는 짐칸·토트(차체에 고정)에 실린다.
- **세 모델이 같은 질량**: `balance`(준정적), `kin`(전신 기구학), `dynamics`(관절 동역학)는 같은 `Body` 메서드(`shoulder_mount`, `head_com`, `column`, `mount_masses`, `cargo_point`)로 질량을 놓는다. 그래서 세 모델의 질량과 무게중심이 형태마다 맞는다 (§7.6, 층 조합 전부에서 검사). 마스트 카메라의 질량은 팬·틸트 피벗(렌즈 아래 `CAMERA_UP_M`) 위 `head.com_up_m`에 놓인다. 지금 머리 프리셋은 둘 다 0.07 m라 숫자는 같지만, 머리 프리셋이 바뀌어도 세 모델이 갈라지지 않는다 (무게중심을 0.15 m로 바꾼 머리로 검사). 그리는 상자도 공유한다: 갑판 팔 받침·토트·짐칸·차체 카메라는 `Body::deck_boxes` 하나를 `kin`과 `dynamics`가 그린다.
- **관절은 있는 것만**: 목이 없으면 `neck`이, 몸통이 없으면 `waist_yaw`·`waist_pitch`가, 팔이 하나면 `_l` 관절이, 조작기가 없으면 어깨·팔꿈치·쥠이 어느 보고서에도 없다. 이름으로 찾으면 `None`이 나온다. 동역학의 목표값은 자유도 순서가 아니라 관절 이름으로 고른다 (`Robot::dof_index`).
- **카메라** (`kin::camera`): 목은 오늘의 고관절 → 골반 → 허리 → 몸통 → 목 사슬이다. 고정 머리는 같은 사슬에 몸통 피치와 고정 틸트만 더한다 (목 각도는 무시). 마스트 카메라는 정한 높이에서 팬·틸트 피벗(있으면)으로 돈다. 차체 카메라는 갑판 앞 가장자리에서 수평이다. 바퀴·트랙의 차체 피치는 모두에 들어간다.
- **마스트 리프트 (Stretch 형)**: 몸통이 없고 마스트 카메라와 팔이 있으면, 팔은 마스트의 캐리지에 탄다. 어깨는 꼭대기(1.10 m)에서 `KinPose::lift_m`만큼 내려간다. 리프트는 기존 몸통 리프트와 같은 `torso_lift` 관절(선형, 힘 = 캐리지에 실린 팔·손·짐의 무게)로 보고된다. 계획기(`reach::plan`)는 몸통 리프트와 같이 리프트 높이를 찾는다. 내려가고 올라가는 시간은 리프트 속도로 가는 최소 저크 이동보다 짧지 않다. 집은 물건은 토트에 놓을 캐리지 높이에서 든다 (`grasp::hold_pose_for`): 캐리지가 마스트 꼭대기까지 올라갔다가 다시 내려오지 않는다.
- **조작기 능력**: 조작기가 없으면 `reach`·`grasp`·`release`·`Body::lift`가 바로 `no_manipulator`로 거절한다. 팔이 하나면 두 손 들기를 `one_arm`으로 거절한다 (한 팔로 몰래 들지 않음). 없는 팔에 걸린 하중도 조용히 버리지 않는다: `kin`은 "왼팔 없음: 손 하중 … N"을 관절 범위 밖으로, `balance`는 불가능으로 보고한다. 집은 물건은 형태의 적재 위치로 간다. 적재함은 오늘과 같다. 토트·짐칸은, 손이 놓는 점(테두리 위 0.20 m) 위로 팔 역기구학이 닿는 가장 높은 리프트 위치를 찾아 거기서 놓는다. 적재 공간이 없으면 손에 든 채 운반한다. 적재 공간이 있는데 손이 닿지 않으면(리프트 없는 몸통·기둥의 팔과 낮은 토트·짐칸) 손에 든 채 운반하면서 `stow_unreachable` 사건으로 알리고, `BodyPlan::validate`가 그 계획을 거절한다. 평행 집게(`Gripper1`)는 설정 손(5지)이 아니라 두 턱으로 쥔다: 조임은 2·F(턱 힘 F가 있는 등급), 닫는 속도·접촉은 2지 손 모델, 초점 화면에도 턱 두 개를 그린다.
- **주행·밀기 표본의 자세**: `act::drive`(`state`, `drive`)와 `act::push`는 오늘의 휴머노이드 선 자세(`KinPose::stand`) 대신 형태의 쉬는 자세(`Body::stand`)로 각 표본을 푼다. 휴머노이드는 같은 자세(비트 단위)이고, 갑판 팔은 갑판 위로 접힌다 (예전에는 손이 땅 아래 −0.22 m였고 축 하중·여유도 그 팔로 셈).
- **너무 높은 목표의 이유**: `reach::plan`이 목표(테이블·선반)가 팔의 가장 높은 어깨에서도 닿지 않는 높이일 때 "선 자세의 팔로는 손바닥이 X m까지만 올라감"이라고 말한다 (예전: "손은 닿지만 굽힌 자세를 버티지 못함…"). 오늘의 휴머노이드에도 해당하는 문구 변경이다. 성공·실패와 벤치마크 줄은 바뀌지 않는다.
- **운반 한도**: 짐칸은 자기 정격(Tracer 100 kg)과 섀시 구조 한도(섀시 정격 − 섀시 위 질량) 가운데 작은 것이다. 적재함·토트는 베이스 한도를 쓰고, 적재 공간이 없으면 0이다.
- **상태 화면**: 하체 이름(발·바퀴·메카넘 바퀴·트랙), 카메라 이름(머리·마스트 카메라·전방 카메라), 적재 영역(`stomach` 적재함, `bed` 짐칸, `tote` 토트)을 형태에서 정한다. 조작기가 없으면 손 영역 두 개가 없다. 오늘의 휴머노이드는 일곱 영역의 id와 순서가 그대로다 (바퀴·트랙 로봇은 아래 두 영역의 이름만 "발" 대신 "바퀴"·"트랙"). `BodyStatus`에 `morph`, `form`, `form_ko`가 붙는다.
- **검산: 좁은 베이스의 전도 하중**: Kemp 외 2022 [49]의 모멘트 균형(m_payload = m_r·d_ra/d_p)을, 갑판 팔을 수평으로 뻗은 자세에 그대로 놓는다. 그리고 `kin`의 ZMP가 바로 그 하중에서 앞바퀴를 넘는지 본다 (엔진 정역학의 일관성 검사이고 벤치마크가 아님).

### 7.6 검사 결과

`cargo test -p noto-core --test morph_forms` (`[MORPH]` 줄). 엔진 입력과의 일관성 검사이고 점수 칸은 아니다.

**기본 형태는 오늘의 숫자 그대로다.** 세 휴머노이드 형태(`legs_humanoid`·`wheel_humanoid`·`track_humanoid`)는 기본 계획과 질량·CoM·카메라·균형 보고서 전체가 비트 단위로 같고 (`the_default_forms_are_todays_robots`), 기존 테스트는 하나도 바꾸지 않고 모두 통과했으며, 벤치마크 줄 1029개([AX-SUM]·[FX]·[FX-SUM]·[NN]·[NN-SUM]·[VX]·[VX-SUM]·[CAM]·[CAM-SUM])가 v1.0 이전 기준선과 글자 그대로 같다.

**이름 붙은 형태 × 등급** (서 있는 자세; 여유 = ZMP 여유, 경사·속도·적재 = `drive_limits`):

| 형태 | 등급 | 질량 kg | CoM m | 카메라 m | 여유 | 경사 ° | 속도 m/s | 적재 kg | 관절 |
|---|---|---|---|---|---|---|---|---|---|
| `legs_humanoid` | 보급 | 8.7 | 0.61 | 1.14 | 0.60 | 20.0 | 0.02 | 0.0 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, hip_pitch_r, hip_roll_r, knee_r, ankle_r, hip_pitch_l, hip_roll_l, knee_l, ankle_l |
| `legs_humanoid` | 중급 | 20.5 | 0.62 | 1.14 | 0.55 | 20.0 | 0.92 | 3.4 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, hip_pitch_r, hip_roll_r, knee_r, ankle_r, hip_pitch_l, hip_roll_l, knee_l, ankle_l |
| `legs_humanoid` | 고급 | 28.9 | 0.61 | 1.14 | 0.55 | 20.0 | 1.20 | 19.0 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, hip_pitch_r, hip_roll_r, knee_r, ankle_r, hip_pitch_l, hip_roll_l, knee_l, ankle_l |
| `wheel_humanoid` | 보급 | 8.2 | 0.57 | 1.14 | 0.95 | 12.0 | 0.24 | 1.4 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `wheel_humanoid` | 중급 | 19.3 | 0.50 | 1.14 | 0.98 | 13.5 | 1.50 | 18.7 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `wheel_humanoid` | 고급 | 44.5 | 0.34 | 1.14 | 0.99 | 15.0 | 1.50 | 20.0 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `track_humanoid` | 보급 | 8.2 | 0.57 | 1.14 | 0.96 | 14.5 | 0.24 | 1.4 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive |
| `track_humanoid` | 중급 | 19.3 | 0.50 | 1.14 | 0.98 | 16.5 | 0.80 | 18.7 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive |
| `track_humanoid` | 고급 | 44.5 | 0.34 | 1.14 | 0.99 | 20.6 | 0.80 | 30.0 | waist_pitch, neck, shoulder_r, elbow_r, shoulder_l, elbow_l, drive |
| `wheel_noneck` | 보급 | 8.1 | 0.57 | 1.12 | 0.95 | 12.5 | 0.24 | 1.6 | waist_pitch, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `wheel_noneck` | 중급 | 19.2 | 0.49 | 1.12 | 0.97 | 14.0 | 1.50 | 18.8 | waist_pitch, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `wheel_noneck` | 고급 | 44.4 | 0.34 | 1.12 | 0.99 | 15.0 | 1.50 | 20.0 | waist_pitch, shoulder_r, elbow_r, shoulder_l, elbow_l, drive_front, drive_rear |
| `mast_gripper` | 보급 | 8.2 | 0.53 | 1.31 | 0.52 | 5.0 | 0.24 | 1.5 | neck, torso_lift, shoulder_r, elbow_r, drive_front, drive_rear |
| `mast_gripper` | 중급 | 17.2 | 0.40 | 1.31 | 0.73 | 11.0 | 1.40 | 20.0 | neck, torso_lift, shoulder_r, elbow_r, drive_front, drive_rear |
| `mast_gripper` | 고급 | 41.6 | 0.27 | 1.31 | 0.87 | 15.0 | 1.50 | 20.0 | neck, torso_lift, shoulder_r, elbow_r, drive_front, drive_rear |
| `deck_gripper` | 보급 | 4.0 | 0.15 | 0.20 | 0.91 | 15.0 | 0.24 | 5.7 | shoulder_r, elbow_r, drive_front, drive_rear |
| `deck_gripper` | 중급 | 13.0 | 0.15 | 0.20 | 1.00 | 15.0 | 1.50 | 20.0 | shoulder_r, elbow_r, drive_front, drive_rear |
| `deck_gripper` | 고급 | 37.4 | 0.14 | 0.20 | 0.98 | 15.0 | 1.50 | 20.0 | shoulder_r, elbow_r, drive_front, drive_rear |
| `truck` | 보급 | 4.5 | 0.14 | 0.25 | 0.98 | 15.0 | 0.24 | 5.2 | drive_front, drive_rear |
| `truck` | 중급 | 11.8 | 0.12 | 0.25 | 0.99 | 15.0 | 1.50 | 26.2 | drive_front, drive_rear |
| `truck` | 고급 | 34.7 | 0.13 | 0.25 | 1.00 | 15.0 | 1.50 | 95.3 | drive_front, drive_rear |

**원형과 비교 (엔진 입력)**:

| 형태 | 엔진 | 원형 | 차이 |
|---|---|---|---|
| `mast_gripper` 중급 | 17.2 kg | Stretch RE1 23 kg [49] | −25% |
| `deck_gripper` 보급 | 4.0 kg | LeKiwi 1.7 kg + SO-101 0.8 kg = 2.5 kg [29][30] | +62% |
| `truck` 고급 | 34.7 kg | AgileX Tracer 30 kg [33] | +16% |
| `mast_gripper` 카메라 | 1.31 m | Stretch RE1 1.31 m [49], Fetch 1.061–1.447 m [50] | 범위 안 |
| `deck_gripper`·`truck` 카메라 | 0.20 · 0.25 m | MiR250 몸체 꼭대기 0.30 m [51] | 꼭대기 아래 |

**동작**:

- 8° 경사에서 카메라 기울기: 목 3.60° · 목 없음 8.00° · 마스트 팬·틸트 3.60° · 차체 카메라 8.00°. 목 없는 카메라는 차체의 기울기를 전부 받는다.
- `truck`: 집기·내려놓기·들기 모두 `no_manipulator`. 짐칸 적재 한도 중급 26.2 kg (같은 등급 휴머노이드의 배 적재 18.7 kg), 고급 95.3 kg (휴머노이드 20.0 kg). 만재해도 CoM 0.28 m (중급; 고급 0.29 m), 여유 1.00.
- `deck_gripper`: 바닥의 물건을 서 있는 채로 집어 토트에 넣는다 (6.5 s, 토트 낙하 0.15 m, 내려놓기 낙하 0.000 m; 예전 4.5 s는 집게를 설정 손(5지)으로 쥐던 값이고, 이제는 2지 손 모델의 접촉·닫는 속도로 쥔다). 주행·밀기 표본에서 손의 최저 높이 0.49 m (휴머노이드 선 자세였다면 −0.22 m). 두 손 들기는 `one_arm`. 팔을 수평으로 뻗었을 때 앞바퀴 위로 넘어지는 하중은 중급 6.2 kg, 고급 18.9 kg (손 0.27 m 앞). 그 하중에서 어깨는 피크의 194% · 160%라서 어깨가 먼저 포화한다.
- `mast_gripper`: 리프트 행정 0.84 m. 바닥 물건은 리프트를 0.74 m 내리고 팬·틸트 73°(한계 83°)로 손을 보며 13.8 s에 내려간다. 0.75 m 테이블 위 물건은 리프트를 0.42 m 내려 집는다. 같은 테이블은 갑판 팔로는 불가 (손바닥이 0.55 m까지만 올라감). 바닥에서 집어 토트에 넣기 26.9 s (캐리지는 토트 높이 리프트 0.17 m까지만 올라옴, 토트 낙하 0.20 m), 토트에서 꺼내 바닥에 내려놓기 28.8 s.
- 층 조합 전부: 1152개 (하체 4 × 몸통 3 × 카메라 6 × 조작기 4 × 적재 4) 가운데 만들 수 없는 576개는 `fit`이 가까운 형태로 고친다. 서로 다른 형태 624개 × 리프트 유무 = 1248개 모두 세 모델의 질량·CoM이 맞고, 서 있고, 관절 목록이 맞으며, 상태·초점·집기·내려놓기 코드가 멈추지 않는다. 조작기가 있는 936개 중 바닥의 1 kg 물건을 집은 것은 216개다 (나머지는 reach 실패로 이유를 대고 거절). 손이 자기 토트·짐칸 위로 가지 못하는 198개는 `BodyPlan::validate`가 거절하고, 집기는 `stow_unreachable`을 내며 손에 든 채 끝난다 (계획 검사와 집기 결과가 1248개 모두에서 일치).
- 집게 조임 (양쪽 턱 2·F): SO-101 42 N, XM430 2지(손 모델) 48 N, Robotiq 2F-85 470 N; 설정 손(5지)은 같은 6 cm 상자에 99 N.
- 휴머노이드 몸통 + 토트 (리프트 없음)는 "팔이 갑판 토트에 닿지 않음 (내려놓을 점 0.48 m)"으로 거절되고, 몸통 리프트를 달면 통과한다.

### 7.7 남은 차이

- **세계·화면은 아직 형태를 모른다**: 이번 묶음은 noto-core만 바꿨다 (world-sim·noto-runtime은 `legged` → `legged()` 치환과 `BodyInputs.morph` 한 줄뿐). `World`는 아직 휴머노이드의 서 있는 자세(`KinPose::stand`, `Posture::quiet`)와 손을 가정한다. 트럭의 집기 거절(`ActionError::NoManipulator`), 다른 로봇이 짐칸에 싣는 흐름, 형태별 카메라 흔들림·UI 실루엣은 다음 묶음(형태-세계·카메라·여러 로봇·UI)이 한다. 형태 쪽에서 준비한 것: `Body::stand/quiet/lift_posture`, `kin::camera`, `Body::cargo_geom`, `Body::camera_mount`, `BodyStatus.morph/form/form_ko`, `BodyPlan::validate`.
- **Stretch와 다른 팔**: Stretch는 옆으로 늘어나는 망원 팔로 베이스 가장자리 밖 0.71 m까지 닿는다 [49]. 여기서는 마스트 캐리지에 단 두 링크 팔(0.48 m)이 앞으로만 뻗어, 범퍼 앞 약 0.25 m까지 닿는다. 마스트 리프트의 속도·힘은 출처가 없어 Fetch 몸통 리프트 값을 빌렸고, 질량은 추정이다.
- **마스트 팔이 느림**: 바닥에서 집어 토트에 넣는 데 26.9 s 걸린다. 0.10 m/s 리프트(Fetch 몸통 리프트 값)로 0.74 m를 내려갔다가 토트 높이까지 올라오기 때문이다. Stretch 리프트의 속도 출처가 없다 (연구 팩에도 없음).
- **갑판 로봇의 질량**: 원형보다 무겁다 (보급 `deck_gripper` +62%). 가슴 프리셋의 배터리를 그대로 싣고, 전자 상자(차체 안)·토트·차체 카메라 질량이 추정이라서다. 원형에 맞추려고 고치지 않았다. 반대로 `mast_gripper`는 Stretch보다 가볍다 (−25%): 마스트·리프트·캐리지를 압출재와 벨트 축으로 셌고, Stretch의 망원 팔과 베이스 구조는 이 프로젝트의 부품이 아니다.
- **균형 모델은 리프트를 모른다**: `balance`의 자세(`Posture`)에는 리프트 높이가 없어, 마스트 팔은 캐리지 꼭대기에 있는 것으로 계산한다 (몸통 리프트와 같은 한계). `kin`은 리프트를 안다. 동역학(`dynamics`)도 리프트 이동을 시뮬레이션하지 않는다.
- **몸통 리프트 내려놓기 시간**: 기존 `wheel_lift`의 내려놓기 경로는 리프트 이동 시간을 넣지 않는다. 마스트 리프트에만 넣었다 (기존 [AX] 줄을 바꾸지 않으려고). 다음 묶음에서 둘을 맞출 것.
- **형태 벤치마크 [MX] 없음**: §7.4의 비교는 손으로 한 것이다. 지금 점수 칸으로 만들 수 있는 것은 Stretch 3 질량·높이, Starship·MiR250 적재 비율 정도다. 벤치 묶음에 넘긴다.
- **팬·틸트 범위**: 마스트 카메라의 틸트는 Fetch URDF [50]의 아래 한계 83°를 위쪽에도 대칭으로 쓴다 (Fetch는 위로 43.5°까지). 오늘의 목(± 45°)과 같은 대칭 단순화다. 팬은 ± 90°로 Stretch 3의 346° [31]보다 좁다.
- **카메라 틸트**: 차체 카메라는 수평이다. 바닥의 택배를 낮은 카메라로 볼 때의 아래 틸트(MiR250의 3D 카메라는 바닥을 앞 250 mm부터 봄 [51])는 카메라 묶음이 정할 것.
- **메카넘**: 형태에서는 바퀴와 같게 본다 (`form()`이 `wheel_humanoid`로 부름).
- **토트 부피**: 토트는 크기로 적재를 막지 않고 베이스 한도만 쓴다.
- **마스트 기둥 몸통**(`Torso::Mast`, Fetch·TIAGo 형)은 층으로만 있고 이름 붙은 형태가 없다. Fetch 벤치마크(Wevolver [52])와 비교하려면 Fetch 급 형태와 그 부품이 필요하다.
- **집게의 벌림 폭**: 평행 집게는 조임 힘만 모델링한다. 턱이 벌어지는 폭(Robotiq 2F-85의 85 mm 등)은 없어서, 2지 손 모델이 감쌀 수 있는 크기면 집는다. 손목 카메라·집게 속도도 2지 손 모델 값이다.
- **차체 카메라와 갑판 위 물건**: 차체 카메라 상자는 갑판 앞 가장자리에 그린다. `deck_gripper`에서는 토트 앞벽에, `truck`에서는 짐칸 앞 가장자리 높이에 걸친다 (그림만; 질량·카메라 위치는 맞음). 휴머노이드 몸통·마스트 기둥에 토트를 단 사용자 형태는 기둥과 토트가 겹쳐 그려진다 (이런 형태는 리프트 없이는 `BodyPlan::validate`가 거절함).
- **런타임 열쇠**: `Body.plan`은 이제 만든 형태라서 `body.plan.id()`가 실제 로봇을 가리킨다. 그러나 noto-runtime·world-sim은 여전히 설정 값 `cfg.body.id()`(요청한 계획)로 SDS·기록 열쇠를 만든다 (actsds·mlsds·chase·testrun·world). 다음 묶음(형태-세계)에서 `world.body.plan.id()`로 바꾸거나 설정을 읽을 때 `BodyPlan::validate`로 거절할 것.
- **`stow_unreachable` 사건 이름**: UI의 `EVENT_KO`와 runtime의 `KEY_KINDS`에 아직 없다 (화면에는 사건 이름이 그대로 나옴). UI 묶음에서 붙일 것.

## 출처

[1] https://evelta.com/sts3215-c018-12v-30kg-cm-dual-shaft-ttl-serial-servo-metal-gears-magnetic-encoder/ ; https://www.robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215
[2] https://openelab.io/da/products/feetech-sts3215-c001-servo ; https://www.waveshare.com/st3215-servo.htm
[3] https://evelta.com/sts3250-12v-50kg-360deg-magnetic-encoder-double-shaft-ttl-servo/
[4] https://openelab.io/products/htd-45h-bus-servo
[6] https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
[7] https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
[8] https://emanual.robotis.com/docs/en/dxl/x/xh540-w270/
[9] https://grabarobot.com/blog/robotis-turtlebot-dynamixel-price-guide-2026/
[10] https://www.robotis.us/robotis-blog/torque-ratings/
[11] https://www.cubemars.com/goods-1142-AK60-6+V11.html
[12] https://cubemars.com/goods-1031-AK70-10.html
[13] https://ca.robotshop.com/products/cubemars-cubemars-ak80-9kv100-brushless-dc-motor-robots
[14] https://www.unitree.com/go1/motor
[15] https://aifitlab.com/products/robstride-01-motor ; https://aifitlab.com/products/robstride-02-motor
[16] https://www.pakronics.com.au/products/robstride-03-qdd-60n-m-integrated-actuator-module-ss100081690
[17] https://aifitlab.com/products/robstride-04-motor
[19] https://robotshop.com/products/myactuator-rmd-x8-pro-h-16-can-servo-actuator-module ; https://www.myactuator.com/x8-60-details
[20] https://mjbots.com/products/qdd100-beta-3
[21] https://shop.odriverobotics.com/products/odrive-s1
[22] https://arxiv.org/html/2504.17249 ; https://humanoid.guide/?p=19737
[23] https://arxiv.org/html/2502.00893v4
[26] https://aiwiki.ai/wiki/k_scale_k_bot
[27] https://www.unitree.com/g1/ ; https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_g1/g1.xml
[29] https://huggingface.co/docs/lerobot/so101
[30] https://www.robotshop.com/products/seeedstudio-lekiwi-kit-12v-mobile-base-w-3d-printed-parts-battery
[31] https://hello-robot.com/learn ; https://docs-arch.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/
[32] https://docs.trossenrobotics.com/agilex_scout_mini_docs/specifications.html
[33] https://www.robotlab.com/store/tracer-robot/ ; https://global.agilex.ai/products/tracer
[34] https://ar5iv.arxiv.org/html/2309.06440
[35] https://www.knoxlabs.com/products/inspire-robots-rh56dfx-dexterous-hand
[36] https://www.eng.yale.edu/grablab/openhand/model_t42.html
[37] https://blog.robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf
[38] https://www.18650batterystore.com/21700-p/samsung-50e.htm
[39] https://www.akkuteile.de/en/molicel-inr21700-p45b-4500mah-45a-li-ion-battery-3-6v-3-7v_100852_3119
[40] https://www.solar-electric.com/relion-rb100-hp-lithium-ion-lifepo4-battery-12v-100ah.html
[41] https://raspberry.tips/en/faq/raspberry-pi-power-consumption-update-2026-all-models-compared
[42] https://jetsonhacks.com/2024/12/17/jetson-orin-nano-super-developer-kit/
[43] https://dev.intelrealsense.com/docs/intel-realsense-d400-series-product-family-datasheet
[44] https://shop.luxonis.com/products/oak-d-lite-1
[45] https://www.matterhackers.com/r/xFvGl8 ; https://www.matterhackers.com/r/VP1dBX
[46] https://fab.academany.org/2021/labs/agrilab/students/antonio-anaya/files/week10/profile_alum_2020_datasheet.pdf
[47] https://compositeenvisions.com/?p=3296
[48] https://soft.siat.ac.cn/papers/XIA(2008)_Robotica_THBIP2Design.pdf ; https://patents.google.com/patent/US12283704
[49] Kemp C.C., Edsinger A., Clever H.M., Matulevich B., "The Design of Stretch: A Compact, Lightweight Mobile Manipulator for Indoor Human Environments", ICRA 2022 — https://arxiv.org/abs/2109.10892
[50] fetch_description URDF, ZebraDevs/fetch_ros (melodic-devel), 2019 — https://raw.githubusercontent.com/ZebraDevs/fetch_ros/melodic-devel/fetch_description/robots/fetch.urdf
[51] Mobile Industrial Robots A/S, "MiR250 Specifications" (2023-06-28) — https://www.hartfiel.com/wp-content/uploads/2024/04/MiR250-Specs.pdf
[52] (Group B) Wevolver, "Fetch Mobile Manipulator" tech specs — https://www.wevolver.com/specs/fetch.mobile.manipulator
[53] (Group B) Ackerman E., "Starship Technologies autonomous ground delivery robots", IEEE Spectrum, 2015 — https://spectrum.ieee.org/starship-technologies-autonomous-ground-delivery-robots
[54] (Group B, 검색 결과로만 봄) Konica Minolta Australia, MiR250 product listing — https://www.konicaminolta.com.au/?p=898
[55] (연구 팩 Group A; 엔진·점수에 쓰지 않고 맥락으로만 적음) Automated Warehouse (업계 보도), Hello Robot Stretch 3 출시 사양, 2024 — https://www.automatedwarehouseonline.com/?p=447

**찾지 못한 값**: D435i·OAK-D Lite·Pi 5·Robotiq 2F-85·Stretch 3 가격, G1 배터리·BHL 액추에이터·Robstride 00 질량, STS3215-C018 감속비, 버스 서보 열 시정수. 표에 이 값이 있으면 추정이다.
