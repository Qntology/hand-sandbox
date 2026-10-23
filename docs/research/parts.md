# 부품 프리셋 리서치 (가성비 1.0–1.4 m 물류 로봇)

조사일 2026-10-23. `crates/hand-core/src/parts.rs`의 숫자는 이 문서에서 왔다. `[n]`은 아래 출처 목록, "추정"은 계산을 함께 적었다. 벤치마크는 돌리지 않았다.

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

**찾지 못한 값**: D435i·OAK-D Lite·Pi 5·Robotiq 2F-85·Stretch 3 가격, G1 배터리·BHL 액추에이터·Robstride 00 질량, STS3215-C018 감속비, 버스 서보 열 시정수. 표에 이 값이 있으면 추정이다.
