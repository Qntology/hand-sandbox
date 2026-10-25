# 머리 카메라: 걸음 흔들림 · 지면 충격 · 바퀴 진동 · 흐림 · 목 팬 — 리서치

조사일 2026-10-25. 지금 1인칭 머리 카메라는 고정되고 완전히 멈춘 카메라다. 이 문서는 그것을 실제 카메라처럼 만드는 데 필요한 숫자를 모은다: 카메라는 몸의 기구학이 정한 머리 자세에 붙고, 발이 땅을 디딜 때나 바퀴가 거친 땅을 구를 때 흔들리며, 흔들림의 크기는 지면 재질과 상태에 따라 다르고, 움직이는 동안 노출 시간만큼 흐려지며, 목이 좌우로 돈다(팬). 앞선 문서와 같은 원칙으로, 엔진이 쓰는 숫자는 모두 출처가 있거나 [추정] 표시와 이유를 단다. 걸음 모델(CoM 오르내림 Massaad 2007, 압력 중심 구름 Jeon & Cho 2020)은 [`perception.md`](perception.md) §1.6, 지면 표는 [`effects.md`](effects.md) §1.1, 머리 프리셋은 [`parts.md`](parts.md)에 있다.

표기는 [`actions.md`](actions.md)와 같다: **[직접]** 출처에 인쇄된 값, **[유도: 식]** 이 문서에서 계산, **[추정]** 출처를 찾지 못해 정한 값(요청서의 "est"), **[대입]** 다른 대상의 값을 빌려 씀, **[2차: X]** X가 인용한 값(원문 미확인), **[초록]** 초록만 확인. `[n]`은 9장의 출처 번호, `[Bn]`은 7장의 벤치마크 전용 출처. **벤치마크는 시간 측정이 아니라 기준 점수 시험**이다.

## 0. 모델 한눈에

| 층 | 무엇을 | 엔진이 받는 입력 |
|---|---|---|
| 머리 자세 | 몸 기구학(골반 → 몸통 → 목 관절 → 머리)에서 카메라 자세. 목은 몸 기울기의 `gaze_stabilization` 몫(0.45/0.55/0.75, 이미 있는 [추정])만큼 상쇄 | 걸음 모델의 CoM 높이, 몸통 피치·롤 흔들림 (§1.1) |
| 디딤 충격 | 발꿈치가 닿을 때 머리에 반 사인 가속 펄스. 크기 × S, 길이 ÷ S (S = 지면 충격 배율) | 단단한 바닥의 피크·길이 (§1.2), 지면별 S (§2) |
| 바퀴·트랙 진동 | ISO 8608 지면 거칠기 PSD를 속도로 시간 신호로 바꿔 섀시를 수직으로 흔듦 | 지면별 ISO 등급, 바퀴 지름·트랙 접지 길이 (§3) |
| 흐림 · 롤링 셔터 | 노출 동안 화소가 움직인 거리 = 각속도 × 노출 × 초점거리(px); 롤링 셔터는 행마다 읽는 시각이 다름 | 카메라별 셔터 종류·노출·초점거리·판독 시간 (§4) |
| 목 팬 | 목 요(yaw) 관절의 범위와 속도 | 팬 범위·서보 무부하 속도 (§5) |

## 1. 걸을 때 머리 · 카메라의 움직임

### 1.1 사람의 머리 운동 (걸음 주파수 대역)

| 양 | 값 | 조건 | 출처 |
|---|---|---|---|
| 머리 수직 이동 (봉우리–골) | 5–9 cm (범위) | 실험실 평지 자유 보행, n = 10 | Pozzo, Berthoz & Lefort 1990 [1] [2차: Dunbar 2008 [3] 표 2] |
| 머리 수직 이동, 연구 간 범위 | 약 2–9 cm | 연구마다 속도·과제가 다름 | Peters, van Emmerik & Bloomberg 2008 [4] 서론 [직접] |
| 머리 수직 속도 최대 | 25–35 cm/s (걷기), 70–150 cm/s (달리기) | 〃 | Pozzo 1990 [2차: [3] 표 3] |
| 머리 피치 (공간 기준, 변위) | 8 ± 2° | 실험실 평지 보행 | Cromwell & Wellmon 2001 [2차: [3] 표 2] |
| 〃 | 9 ± 3° | 자유 보행 | Pozzo 1990 [2차: [3] 표 2] |
| 〃, 연구 간 범위 | 1.5–8.5° | | [4] [직접] |
| 몸통 피치 (공간 기준) | 4 ± 1° | 평지 보행 | Cromwell & Wellmon 2001 [2차: [3] 표 2] |
| 목 피치 (머리–몸통) | 14 ± 4° | 〃 | 〃 |
| 머리 피치 최대 각속도 | 30 ± 8 °/s | 자유 보행 | Pozzo 1990 [2차: [3] 표 3] |
| 〃 | 9 ± 3 °/s | 평지 보행 | Cromwell 외 2004 [2차: [3] 표 3] |
| 머리 회전(요·피치) 최대 속도, 집단 중앙값 | 90 °/s를 넘지 않음 | 제자리 걷기·달리기, 서치 코일, n = 20 | Grossman 외 1988 [5] [초록] |
| 모든 과제의 머리 회전 | 프랑크푸르트면 회전 < 20°, 각속도 < 140 °/s | 걷기·제자리 걷기·제자리 달리기·뛰기 | Pozzo 1990 [1] [초록] |
| 걸음 속도와 머리 피치 | 1.2 m/s 이하: 공간 기준 머리 피치가 작고 머리–몸통 피치가 몸통 피치를 보상. 그 위: 수직 이동을 보상하는 피치가 생김 | 트레드밀 0.6–2.2 m/s | Hirasaki 외 1999 [2] [초록] |
| 주 주파수 | 1.4 Hz (0.6 m/s) → 2.5 Hz (2.2 m/s); 0.8–1.8 m/s에서 약 2.0 Hz | 〃 | Hirasaki 1999 [2] [초록] |
| 〃 | 2 Hz (SD 0.13) | 하루 10 시간 머리 선가속도, n = 20 | MacDougall & Moore 2005 [6] [초록] |
| 고조파 | 6–8 Hz까지 (주 주파수 0.4–3.5 Hz) | 네 과제 | Pozzo 1990 [1] [초록] |

로봇 걸음 속도(0.3–1.0 m/s)는 Hirasaki의 "1.2 m/s 이하" 구간이다: 사람은 이 속도에서 머리를 공간에 거의 고정한다. 이 시뮬레이터의 `gaze_stabilization` 몫(0.45–0.75, [추정])은 사람보다 덜 고정하는 셈이며, 위 표는 그 몫을 정하는 근거가 아니라 결과가 사람 범위 안에 드는지 보는 기준선이다.

머리 높이의 걸음 대역 흔들림은 새 상수가 필요 없다: 몸통이 강체이므로 머리 수직 이동 = CoM 오르내림(이미 Massaad 2007, 1 km/h 1.5 cm · 6 km/h 3.4 cm 봉우리–골) + 몸통 피치가 목 위치를 돌리는 몫 **[유도: 기구학]**. 사람의 머리 5–9 cm는 CoM보다 큰데, 여기에는 머리가 몸통보다 위·앞에서 피치로 도는 몫과 더 빠른 걸음이 들어 있다.

### 1.2 발꿈치 디딤 충격 (heel-strike transient)

| 양 | 값 | 조건 | 출처 |
|---|---|---|---|
| 경골(정강이) 피크 가속 | 약 5 g | 보통 보행, 딱딱한 굽 | Light, McLellan & Klenerman 1980 [7] [초록] |
| 두개골 피크 가속 | 약 0.5 g | 〃 | 〃 |
| 탄성 있는 굽 | 두 피크 모두 약 절반 | 〃 | 〃 |
| 발 → 머리 감쇠 | 5 g → 0.5 g, 약 90% | 〃 | [7]에서 [유도] |
| 전달 속도 | 발–머리 평균 220 m/s (다리 610, 척추 90 m/s); 흡수는 주로 다리, 척추는 덜 | 피부 부착 가속도계 | Smeathers 1989 [8] [초록] |
| 발 → 머리 지연 | 약 7 ms | 1.6 m / 220 m/s | [8]에서 [유도] |
| 몸의 유효 축 강성 | 무릎 0° → 40°에서 28.7 → 7.9 kN/m; 이 강성과 머리/정강이 충격비 r = 0.97 | 인간 진자로 발에 충격, n = 38 | Lafortune, Lake & Hennig 1996 [9] [초록] |
| 충격의 주파수대 (달리기) | 3–8 Hz는 몸의 능동 움직임, 9–20 Hz는 충격파 | | Derrick, Hamill & Caldwell 1998 [2차: Liew 외 2024 [10]] |

엔진 모양: 디딤마다 머리에 위로 향하는 반 사인 가속 펄스 a(t) = A·sin(πt/T), 0 ≤ t ≤ T.

- 단단한 바닥의 A = 0.5 g [대입: 사람 두개골, 딱딱한 굽 [7]]. 로봇은 살·연골이 없어 감쇠가 덜할 수 있고(Smeathers: 흡수는 주로 다리의 연조직) 고무 패드가 굽 역할을 하므로, 같은 값을 쓰되 [대입]으로 둔다.
- T = 33 ms [추정]: 9–20 Hz 충격대의 가운데 15 Hz의 반주기 1/(2·15 Hz). 검산: 반 사인 펄스의 속도 변화 Δv = 2AT/π = 0.10 m/s **[유도]**, 사람 머리 수직 속도 최대 25–35 cm/s([3] 표 3)의 일부라 크기가 맞다.
- 지면이 무르면 같은 Δv를 더 길게 나눠 받는다: A × S, T ÷ S (§2.1). T가 양발 지지 시간을 넘으면 걸음 대역 오르내림에 합쳐진다(펄스 생략) [추정].
- 로봇 몸의 지연(금속·플라스틱 링크에서 파동 속도는 사람 척추보다 훨씬 빠름)은 0으로 둔다 [추정].

### 1.3 휴머노이드 로봇의 머리 · 몸통 IMU 데이터

**찾지 못함.** NAO·HRP·Atlas·Digit·Cassie의 걷는 중 머리 또는 몸통 IMU의 각속도·피크 g를 숫자로 보고한 공개 자료를 찾지 못했다. 찾은 것은 정성적 기술뿐이다: 작은 휴머노이드가 걸을 때 영상의 움직임 흐림이 심해 특징점 추적이 깨진다(Pretto 외 2009 [11] [초록]); 다리 로봇의 시각-관성 추정이 디딤 충격과 진동에 흔들려 충격 흡수 목을 만든 사례(Kim, Kim & Lee 2023 [38] — 수치는 본문을 열지 못함); SABIAN 휴머노이드 머리 안정화(Falotico 외 2017, Autonomous Robots — 수치는 본문을 열지 못함). 그래서 §1.2의 사람 값을 [대입]으로 쓴다.

## 2. 지면이 디딤 충격을 바꾸는 정도

### 2.1 모델

발(패드)과 지면을 직렬 스프링으로 본다: k_eff = (1/k_발 + 1/k_지면)⁻¹. 같은 속도로 닿는 질량 m의 피크 가속은 a = Δv·√(k_eff/m), 접촉 시간은 π·√(m/k_eff) **[유도: 무감쇠 질량–스프링]**. 그러므로 단단한 바닥 대비

- 충격 배율 S = √(k_eff / k_발) (단단한 바닥에서 k_eff → k_발),
- 피크 × S, 펄스 길이 ÷ S (속도 변화 Δv는 그대로).

스포츠 바닥의 "충격 흡수율"(force reduction, FR)은 같은 시험 낙하로 잰 피크 힘을 강체(콘크리트) 바닥과 비교한 값 FR = 1 − F/F_강체 이고(EN 14808 계열, James 외 2026 [14] §2.1의 정의) [직접], 위 모델에서 FR = 1 − S다. FIFA 2024 시험의 낙하 장치(AAA)는 20 kg ± 100 g, 스프링 2 kN/mm ± 100 N/mm, 낙하 높이 60 mm [14] [직접] — 사람 다리·신발을 흉내 낸 스프링이라 로봇 발 패드의 대역으로 쓴다:

- **k_발 = 2000 N/mm [대입: AAA 스프링 [14]]**. 검산: 고무 패드 두께 5 mm, 발꿈치 접촉 40 × 80 mm, 쇼어 A 60 고무 E ≈ 3.6 MPa로 k = E·A/t ≈ 2300 N/mm [추정: 패드 치수·고무 계수] — 같은 크기다.
- 지면 강성이 탄성 반공간이면 강체 원판 압입 k = 2Ea/(1 − ν²) (Johnson 1985 [12], 평평한 원판 펀치) [직접: 식], 발꿈치 접촉 반지름 a = 35 mm [추정].

### 2.2 지면별 강성과 근거

| 지면 (`Surface`) | 근거 값 | 조건 | 출처 | 쓰는 k_지면 (N/mm) |
|---|---|---|---|---|
| Firm (마른 콘크리트·아스팔트) | 콘크리트 = FR 0%의 기준 강체면 | 스포츠 바닥 시험 | [14] [직접: 정의] | 강체. 아스팔트 E ≈ 3 GPa여도 k ≈ 2.4 × 10⁵ → S = 0.996 [추정: E] |
| Rock | 암석 E ≥ 10 GPa | | [추정] | 강체 (S = 0.999) |
| Gravel (다진 자갈 패드) | 조밀한 모래·자갈 Es = 100–190 MPa | 정적 할선 계수, ν = 0.3 | GEO5 도움말(Ranjan & Rao 2000, Briaud 2001 인용) [13] [2차] | 7,700–14,600 [유도: 2Ea/(1−ν²)] |
| Dirt (다진 흙) | 중간 굳기 점토 Es = 15–50 MPa | ν = 0.35 | [13] [2차] | 1,200–4,000 [유도] |
| Grass (천연 잔디) | AAA로 잰 평균 강성 301 N/mm | 프로 경기장 천연 잔디, 2024 FIFA AAA | James 외 2026 [14] [직접] | 301 |
| WetSand (물가의 젖은 모래) | Clegg Gmax 272 g | 2.25 kg Clegg 해머, 0.4–0.6 m 낙하, 수분 20%·다짐 높음 | Hayati 외 2020 [15] 표 2 [직접] | 1,360–2,040 [유도: k = m(a/v)², 낙하 높이 0.6/0.4 m] |
| (참고) 축축하고 느슨한 모래 | Gmax 96 g | 수분 12%·다짐 낮음 | [15] 표 2 [직접] | 170–255 [유도] |
| Sand (마른 느슨한 모래) | 젖은 다진 모래에 비해 피크 힘 약 1/4, 강성 1/6, 관입·충격 시간·상승 시간 3–4배 | 낙하 3.86–14 kg, 0.1–0.4 m (달리기 발꿈치 에너지) | Barrett, Neal & Roberts 1998 [16] [초록] | S = S_WetSand / 4 [유도: 피크 힘 비] |
| Water (얕은 물) | 찾지 못함 | | | [추정] 바닥은 포화된 모래: 모세관 흡인이 사라져 마른 모래처럼 무름, 물이 발을 먼저 늦춤 → Sand와 같은 S |

메모:
- Hayati 2020의 표 2 "K (kN/m)"(16.7–99.2)는 하중–변형 곡선의 할선 값이라 관입(소성 변형)이 들어가 Gmax에서 구한 값보다 10배쯤 작다. 탄성 충격 피크를 묻는 이 모델에는 Gmax에서 구한 값을 쓴다. 논문은 높이별 네 번 중 최댓값을 보고했다고만 써서 어느 높이인지 모르므로 0.4–0.6 m 양 끝을 범위로 둔다.
- 검산: 천연 잔디 301 N/mm에서 FR = 1 − 0.36 = 64% [유도]. 인공 잔디의 FIFA 요구 범위(2024 시험법에서 60–70% [14])와 같은 크기다. 천연 잔디의 FR 측정값은 [14]에 그림으로만 있어 옮기지 않았다.
- 수분 12%의 느슨한 모래(S 0.28–0.34)가 마른 모래 추정 0.17보다 단단한 것은 수분이 모래를 단단하게 한다는 [15]의 결과와 같은 방향이다.
- 시뮬레이터의 `wet` 표시는 젖은 모래·물만 강성을 바꾼다. 젖은 콘크리트·바위는 물막이 강성을 바꾸지 않는다고 본다 [추정].
- 다리 로봇 시뮬레이션에서 쓰는 접촉 모델 값의 참고: STANCE(Fahmi 외 2020 [17])는 Kelvin–Voigt 접촉으로 무른 3,500 · 중간 8,000 · 단단한 10,000 · 강체 2 × 10⁶ N/m, 감쇠 400 N·s/m를 모의했고, 실험 폼 블록은 압입 시험으로 2,400 N/m였다 [직접]. 이는 재질 측정이 아니라 제어기 시험용 값이라 엔진 입력으로 쓰지 않는다.

### 2.3 지면 충격 배율 S

S = √(k_eff/k_발), k_발 = 2000 N/mm **[유도]**.

| 지면 | k_지면 (N/mm) | S (범위) | 쓰는 S | 펄스 길이 (단단한 바닥 33 ms 기준) |
|---|---|---|---|---|
| Firm | 강체 | 1.00 | 1.00 | 33 ms |
| Rock | 강체 | 1.00 | 1.00 | 33 ms |
| Gravel | 7,700–14,600 | 0.89–0.94 | 0.92 | 36 ms |
| Dirt | 1,200–4,000 | 0.61–0.82 | 0.75 | 44 ms |
| WetSand | 1,360–2,040 | 0.64–0.71 | 0.68 | 49 ms |
| Grass | 301 | 0.36 | 0.36 | 92 ms |
| Sand | (WetSand ÷ 4) | 0.16–0.18 | 0.17 | 194 ms → 양발 지지보다 길면 걸음 대역에 합침 |
| Water | [추정] | — | 0.17 | 〃 |

사람은 지면에 따라 다리 강성을 바꿔 충격을 비슷하게 유지한다(Ferris, Louie & Farley 1998 [37] [초록]) — 이 모델의 로봇은 다리 강성을 바꾸지 않으므로 S가 그대로 머리에 간다. 이 차이는 7장의 벤치마크(사람 경골 가속이 지면에 따라 거의 같음)가 드러낼 것이고, 숨기지 않는다.

## 3. 바퀴 · 트랙의 진동

### 3.1 ISO 8608 지면 거칠기 등급

ISO 8608은 노면 높이의 변위 PSD를 Gd(n) = Gd(n₀)·(n/n₀)^(−w), n₀ = 0.1 cycle/m, w = 2로 두고 Gd(n₀)로 등급을 나눈다. 유효 범위 0.011 < n < 2.83 cycle/m (파장 약 0.35–91 m) [2차: Ribeiro & Cunha 2026 [20] 식 (11)].

| 등급 | Gd(n₀) 하한–상한 (10⁻⁶ m³) | 기하 평균 | 설명 |
|---|---|---|---|
| A | – 32 | 16 | 공항 활주로·고속도로 |
| B | 32–128 | 64 | 보통 포장도로 |
| C | 128–512 | 256 | 비포장 도로·손상된 포장 |
| D | 512–2,048 | 1,024 | 거친 비포장 도로 |
| E | 2,048–8,192 | 4,096 | 엔듀로 길 |
| F | 8,192–32,768 | 16,384 | 오프로드 길 |
| G | 32,768–131,072 | 65,536 | 거친 오프로드 길 |
| H | 131,072 – | 262,144 | 모의용 |

한계값은 Agostinacchio, Ciampa & Olita 2014 [19] 표 1 [2차: ISO 8608], 기하 평균과 설명은 [20] 표 1 [2차: ISO 8608:2016] (PDF 텍스트 추출에서 열이 밀려 있어 순서대로 다시 맞춤 [유도]).

### 3.2 섀시 수직 가속으로 바꾸기

속도 v로 달리면 공간 주파수 n이 시간 주파수 f = n·v가 되고, 가속 PSD는 Ga(n) = (2πn)⁴·Gd(n) ([18]의 식 (3)) [직접]. 서스펜션 없는 강체 섀시가 바퀴를 통해 노면을 그대로 따른다고 보면

σ_a = (2π)²·v²·√(Gd(n₀)·n₀²·(n_c³ − n_min³)/3) **[유도: w = 2 적분]**

이다. 노면이 같으면 σ_a ∝ v². 위쪽 한계 n_c는 바퀴가 따라갈 수 있는 가장 짧은 파장에서 정한다: 바퀴 지름 D보다 짧은 요철은 바퀴가 덮고 지나가므로 n_c = 1/D [추정: 기하 필터]. 트랙은 접지 길이 L에 걸쳐 높이를 평균하므로 n_c = 1/L [추정]. 0.011–2.83 cycle/m 밖으로 w = 2를 늘이는 것은 ISO 8608의 유효 범위 밖이므로 [추정]이다.

v = 1 m/s의 σ_a (m/s²) **[유도]**:

| 등급 | ISO 대역만 (n_c = 2.83) | 바퀴 D = 0.11 m (n_c = 9.1) | 바퀴 D = 0.10 m | 트랙 L = 0.35 m (n_c = 2.86) |
|---|---|---|---|---|
| A | 0.043 | 0.25 | 0.29 | 0.044 |
| B | 0.087 | 0.50 | 0.58 | 0.088 |
| C | 0.17 | 1.00 | 1.15 | 0.18 |
| D | 0.35 | 2.00 | 2.31 | 0.35 |
| E | 0.69 | 4.00 | 4.61 | 0.70 |
| F | 1.39 | 8.00 | 9.23 | 1.41 |
| G | 2.78 | 16.0 | 18.5 | 2.82 |

### 3.3 지면 → 등급 (모두 [추정])

| 지면 | 등급 | 이유 |
|---|---|---|
| Firm | B | 마른 콘크리트·아스팔트 = "보통 포장도로" |
| WetSand | B | 파도가 다진 평평한 물가 모래 |
| Gravel | C | 웨이포인트 패드는 다진 자갈 = "비포장 도로" |
| Grass | C | 깎은 풀밭의 땅 기복은 비포장 도로 정도, 짧은 파장은 잔디와 뿌리층이 덮음 |
| Water | C | 바닥이 모래인 얕은 물 (바퀴형이 들어갈 수 있다면) |
| Dirt | D | 다져지지 않은 흙길 = "거친 비포장 도로" |
| Sand | D | 바람 물결·발자국이 있는 마른 모래 |
| Rock | F | 바위가 드러난 지면 = "오프로드 길" |

지면 강성 §2.2가 바퀴 진동을 거르는지 검산: 바퀴 하나에 5 kg이 실릴 때 잔디(301 N/mm)의 접촉 고유 주파수는 (1/2π)√(301,000/5) ≈ 39 Hz로, 1 m/s에서 바퀴가 받는 대역(≤ 9 Hz)보다 높다 **[유도]**. 그러므로 무른 지면은 바퀴 진동을 거의 거르지 않는다고 보고 따로 감쇠를 넣지 않는다. 마른 모래만은 바퀴가 가라앉아 짧은 요철을 덮지만, 그 효과의 출처를 찾지 못해 넣지 않았다(8장).

### 3.4 소형 로봇의 측정 진동

요청서의 진동 기반 지면 분류 논문(Brooks & Iagnemma 2005, Weiss 2006, DuPont 2008, Ojeda 2006, Mei 2019)은 지면별 RMS 값을 표로 주지 않거나(그림·분류 정확도만) 본문을 열지 못했다. 표로 숫자를 준 것은 Milella 2019 하나였고, 이것은 엔진 입력과 다른 독립 기준으로 쓰도록 벤치마크 쪽(7장 B2)에 두었다.

## 4. 카메라 사양

### 4.1 Intel RealSense D435i (중급 · 고급 프리셋)

| 양 | 값 | 출처 |
|---|---|---|
| RGB 센서 | OmniVision OV2740, 1920 × 1080, **롤링 셔터**, f/2.0, 초점거리 1.88 mm, 시야 H 69° · V 42° · D 77° (렌즈마다 ± 3°) | D400 데이터시트 337029-017 rev 023 (2026-03) [21] 표 3-18 · 깊이 모듈 비교표 [직접] |
| RGB 센서의 최대 속도 | 1080p 60 fps, 720p 90 fps | OmniVision 제품 페이지 [22] [2차: 검색 요약] |
| 깊이 센서 (좌·우) | OmniVision OV9282, 1280 × 800, **글로벌 셔터**, 초점거리 1.93 mm | [21] 표 3-13 [직접] |
| 깊이 시야 | HD(16:9) H 87° · V 58° · D 95°, VGA(4:3) H 75° · V 62° · D 89°; 기준선 50 mm | [21] 비교표 [직접] |
| 깊이 수동 노출 | 1–166 ms | [21] 표 4-23 [직접] |
| RGB 수동 노출 | 1–10,000 (단위 미기재) | [21] 표 4-24 [직접]; UVC 노출 단위 100 µs라면 0.1 ms–1 s [유도: UVC 1.5 [23] 단위 가정]; 기본값 156 (= 15.6 ms) [2차: RealSense 지원 포럼] |
| IMU | BMI055 → 신형 SKU는 BMI085 (PCN 118035); 가속 ± 4 g, 62.5/250 Hz; 자이로 ± 1000 °/s, 200/400 Hz; 시각 정확도 50 µs | [21] 표 4-25 · 4-26 · SKU 표 [직접] |
| BMI055 자이로 | 잡음 밀도 0.014 °/s/√Hz (47 Hz 대역 0.1 °/s rms), 영점 오프셋 ± 1 °/s, g-감도 0.1 °/s/g | BST-BMI055-DS000-08 rev 1.2 (2014-07) [24] [직접] |
| BMI055 가속 | 잡음 밀도 150 µg/√Hz, 영점 오프셋 ± 70 mg | [24] [직접] |
| BMI085 자이로 | 0.014 °/s/√Hz, ± 1 °/s | BST-BMI085-DS000-16 rev 1.6 (2021-11) [25] [직접] |
| BMI085 가속 | 105 (X·Y) / 135 (Z) µg/√Hz, 영점 오프셋 20 mg | [25] [직접] |
| 측정한 D435i 자이로 (참고) | 각 무작위 보행 0.0053–0.0084 °/√s, 바이어스 불안정 1.6–5.4 °/h | MILUV (arXiv 2504.14376) 표 3–4 — 이미 [`perception.md`](perception.md) §1.1에서 씀 [직접] |

### 4.2 Luxonis OAK-D Lite (보급 프리셋)

| 양 | 값 | 출처 |
|---|---|---|
| RGB 센서 | Sony IMX214, 13 MP (4208 × 3120), **롤링 셔터**, 1/3.1", DFOV 81.3° | OAK-D Lite 데이터시트 (2021-12) [26] §5.1 [직접] |
| RGB 시야 | D 81° · H 69° · V 54° | Luxonis 제품 문서 [27] [직접] |
| IMX214 속도 | 전체 해상도 30 fps, 1080p 60 fps (OAK 파이프라인에서 1080p는 35 fps) | Sony 데이터시트 발췌 [2차: 검색 요약] · [27] |
| 스테레오 센서 | OmniVision OV7251, 640 × 480 @ 120 fps, **글로벌 셔터**, IR 필터 없음, DFOV 85.6° · HFOV 72.9° · VFOV 57.7° | [26] §5.2 [직접] |
| 기준선 | 75 mm | [27] [직접] |
| IMU | BMI270 6축 (킥스타터 초기판에는 없음; 2021-12 데이터시트에는 IMU 언급 없음) | [27] [직접] |
| BMI270 자이로 | 잡음 밀도 0.007 °/s/√Hz (성능 모드) · 0.010 (보통 모드); 74.6 Hz 대역 0.07 °/s rms; 영점 오프셋 ± 0.5 °/s | BST-BMI270-DS000-04 rev 1.2 (2020-06) [28] [직접] |
| BMI270 가속 | 160 µg/√Hz, 영점 오프셋 ± 20 mg | [28] [직접] |
| 수동 노출 | 1–33,000 µs, ISO 100–1600 | DepthAI v2 RGB/모노 카메라 제어 예제 [29] [직접] |

### 4.3 640 px 폭의 초점거리

f_px = (W/2) / tan(HFOV/2) **[유도: 핀홀]**, HFOV는 위 데이터시트 값.

| 카메라 · 모드 | f_px |
|---|---|
| D435i RGB, 16:9 (640 × 360) | 466 |
| D435i RGB, 640 × 480 (1920 × 1080에서 1440 × 1080 4:3 잘라 축소) | 621 |
| D435i 깊이, VGA 640 × 480 (H 75°) | 417 |
| D435i 깊이, 848 × 480 (H 87°) | 447 |
| OAK-D Lite RGB, 640 폭 (H 69°) | 466 |
| OAK-D Lite 모노 OV7251, 640 × 480 (H 72.9°) | 433 |

렌더러의 지금 카메라(수평 70°)는 640 px에서 457 px로, RGB 16:9 모드와 같은 크기다.

### 4.4 움직임 흐림과 롤링 셔터

- **흐림 길이** (화소): b ≈ f_px·ω·t_exp (회전), 여기에 병진 몫 f_px·v⊥·t_exp/Z를 더함 **[유도: 핀홀 투영, 작은 각]**. 예: f = 466 px, 머리 피치 30 °/s(사람 걷기 최대 [3]), 노출 15.6 ms → 3.8 px; 90 °/s(Grossman 상한 [5]) → 11.4 px; 같은 90 °/s라도 맑은 날 노출 0.09 ms면 0.07 px.
- **롤링 셔터 기울기**: 행 i는 t₀ + i·(t_read/H)에 노출되므로 판독 동안 요 각속도 ω면 화면 위아래가 f_px·ω·t_read만큼 옆으로 밀린다 **[유도]**. 판독 시간은 데이터시트에 없다 → OV2740·IMX214 모두 1080p 60 fps가 가능하므로 1080p 판독 ≤ 16.7 ms [추정: 최대 프레임률의 주기 상한], IMX214 전체 해상도는 ≤ 33 ms. 예: 30 °/s 요, 16.7 ms → 4 px.
- 글로벌 셔터(OV9282·OV7251 깊이·모노)는 기울기 0, 흐림만 있음.

### 4.5 밝기에 따른 노출 시간

자동 노출은 장면 밝기에 맞춰 노출을 정한다. 반사광 노출계 식 N²/t = L·S/K, K = 12.5 (ISO 2720:1974가 정한 10.6–13.4 범위 안에서 Sekonic·Canon·Nikon이 쓰는 값) [2차: Sekonic 답변·ISO 2720 [30]]에 f/2.0(두 RGB 모두 [21][26]), 기준 감도 ISO 100 [추정: 센서 기본 이득], 장면 휘도 L = E·0.18/π (18% 회색, 램버트 면) [추정]을 넣으면 t = 0.5/L s **[유도]**.

| 조도 E (lx) [추정: 대표값] | 노출 t | 비고 |
|---|---|---|
| 100,000 (맑은 날 직사광) | 0.09 ms | |
| 10,000 (밝은 흐린 날) | 0.87 ms | |
| 1,000 (어두운 흐린 날) | 8.7 ms | |
| 500 (사무실) | 17.5 ms | |
| 200 (창고 통로) | 33 ms에서 멈춤 + 이득 | 30 fps의 프레임 주기 33 ms를 넘을 수 없음 (V4L2: 노출은 프레임 간격에 묶임 [23]) |

## 5. 목 팬 (머리 요)

| 대상 | 팬 범위 | 팬 속도 | 출처 |
|---|---|---|---|
| Fetch `head_pan_joint` | ± 1.57 rad (± 90°) | 1.57 rad/s (90 °/s); 틸트 −0.76 … 1.45 rad | fetch_description URDF [31] [직접] |
| TIAGo `head_1_joint` | ± 75° | 3.0 rad/s (172 °/s); 틸트 −60 … 45° | tiago_description URDF [32] [직접] |
| PR2 `head_pan_joint` | ± 3.007 rad (± 172°) | 6 rad/s (344 °/s); 틸트 −0.47 … 1.40 rad, 5 rad/s | pr2_description URDF head_v0 [33] [직접] |
| 사람 능동 경추 회전 | 중앙값 오른쪽 60.7° · 왼쪽 61.3° (IQR 9–11°) | — | Zárate-Tejero 외 2023 [34] 표 2, CROM, n = 122, 18–75세 [직접] |
| 〃 나이 · 성별 | 해마다 −0.4° (오른쪽) · −0.3° (왼쪽), 남성 −4.4° · −2.2° | — | Youdas 외 1992 [2차: [34]] |
| 사람 머리 회전 속도 | 걸을 때 ≤ 90 °/s; 힘껏 돌릴 때 요 중앙값 780 °/s | — | Grossman 1988 [5] [초록] |
| STS3215-C018 (12 V, 보급·중급 목) | 서보 범위 360° (12비트) | 0.222 s/60° = 45 rpm = 270 °/s 무부하 | Waveshare ST3215 [35] [직접] |
| STS3215 7.4 V판 | 〃 | 0.192 s/60° = 52 rpm = 312 °/s | [35] [직접] |
| Dynamixel XM430-W350 (12 V, 고급 목) | 0–360° (위치 모드) | 46 rpm = 276 °/s 무부하 (11.1 V 43, 14.8 V 57 rpm) | ROBOTIS e-Manual [36] [직접] |

[`parts.md`](parts.md)의 머리 프리셋은 STS3215-C018(12 V, 4.7 rad/s)과 XM430-W350(4.8 rad/s)으로 같은 값이다. 머리를 돌리는 데는 관성 토크만 들어 정지 토크의 몇 %라서 직류 모터의 직선 토크–속도 관계로 무부하 속도의 약 90%를 낼 수 있다 [추정: 부하 비율].

## 6. 엔진 입력 (engine inputs)

| 상수 | 쓰는 값 | 구분 | 출처 |
|---|---|---|---|
| 머리 수직 흔들림 | CoM 오르내림(기존 Massaad 모델) + 몸통 피치가 목을 돌리는 몫, 새 상수 없음 | 유도 | §1.1 |
| 몸통 피치 흔들림 (다리형) | 4° 봉우리–골, 걸음 주파수 | 대입 (사람 몸통) | Cromwell & Wellmon 2001 [2차: [3]] |
| 몸통 롤 흔들림 (다리형) | 4° 봉우리–골, 보폭(걸음 두 번) 주파수 | 추정 (사람 값 못 찾음, 피치와 같게) | — |
| 목이 상쇄하는 몫 | 0.45 / 0.55 / 0.75 (기존) | 추정 (기존) | parts 프리셋 |
| 머리 회전 속도의 사람 상한 (검산) | 걷기 ≤ 90 °/s, 머리 피치 30 ± 8 °/s | 인용 | [5], [3] |
| 디딤 충격 피크 (단단한 바닥) | 0.5 g, 위로 | 대입 (사람 두개골, 딱딱한 굽) | Light 1980 [7] |
| 디딤 충격 펄스 길이 (단단한 바닥) | 33 ms 반 사인 (Δv = 0.10 m/s) | 추정 (9–20 Hz 대역 가운데) | [10] |
| 발 → 머리 지연 | 0 ms (사람 7 ms) | 추정 | [8] |
| 발 패드 강성 | 2000 N/mm | 대입 (AAA 스프링) | [14] |
| 지면 충격 배율 S | Firm 1.00, Rock 1.00, Gravel 0.92, Dirt 0.75, WetSand 0.68, Grass 0.36, Sand 0.17, Water 0.17 | 유도 (Grass·WetSand·Sand는 측정에서, Gravel·Dirt는 토질 계수에서, Rock·Water는 추정) | §2.2–2.3 |
| 충격 펄스 길이 | 33 ms ÷ S, 양발 지지 시간을 넘으면 생략 | 유도 / 추정 | §2.1 |
| 바퀴·트랙 지면 등급 | Firm B, WetSand B, Gravel C, Grass C, Water C, Dirt D, Sand D, Rock F | 추정 (등급 정의는 인용) | §3.3, [19][20] |
| 바퀴 진동 대역 | 0.011 cycle/m – 1/D (D = 바퀴 지름); 트랙 – 1/L (접지 길이) | 추정 | §3.2 |
| 1 m/s 바퀴 진동 RMS (D = 0.11 m) | Firm 0.50, WetSand 0.50, Gravel 1.0, Grass 1.0, Water 1.0, Dirt 2.0, Sand 2.0, Rock 8.0 m/s²; ∝ v² | 유도 (ISO 8608 + 추정 대역) | §3.2 |
| 〃 ISO 대역만 (0.011–2.83) | Firm 0.087, Gravel·Grass 0.17, Dirt·Sand 0.35, Rock 1.39 m/s² | 유도 | §3.2 |
| 1 m/s 트랙 진동 RMS (L = 0.35 m) | 위 "ISO 대역만"과 거의 같음 (n_c = 2.86) | 유도 / 추정 | §3.2 |
| RGB 셔터 | D435i OV2740 · OAK-D Lite IMX214 롤링; 깊이·모노 OV9282 · OV7251 글로벌 | 인용 | [21][26] |
| 롤링 셔터 판독 시간 | 1080p 이하 16.7 ms, IMX214 전체 해상도 33 ms | 추정 (최대 프레임률 상한) | §4.4 |
| 노출 시간 | t = 0.5/L s (L = 0.18·E/π), 프레임 주기 33 ms에서 멈춤 | 유도 (K 인용, f/2.0 인용, ISO 100 추정) | §4.5 |
| 깊이 노출 범위 (D435i) | 1–166 ms | 인용 | [21] |
| 초점거리 (640 px 폭) | D435i RGB 466 (16:9) / 621 (4:3), 깊이 417 (VGA); OAK RGB 466, 모노 433 | 유도 (FOV 인용) | §4.3 |
| 흐림 | b = f_px·ω·t_exp (+ 병진 f_px·v⊥·t_exp/Z) | 유도 | §4.4 |
| 자이로 잡음 밀도 | D435i 0.014 °/s/√Hz (데이터시트), 측정 0.0069 °/√s (기존); OAK-D Lite 0.007 (성능) · 0.010 (보통) °/s/√Hz | 인용 | [24][25][28] |
| 자이로 영점 오프셋 | D435i ± 1 °/s, OAK-D Lite ± 0.5 °/s | 인용 | 〃 |
| 가속 잡음 밀도 | BMI055 150, BMI085 105–135, BMI270 160 µg/√Hz | 인용 | 〃 |
| 목 팬 범위 | ± 90° | 대입 (Fetch) | [31] |
| 목 팬 속도 | 무부하 속도의 90%: STS3215-C018 243 °/s, XM430-W350 248 °/s | 인용 × 추정 | [35][36] |

## 7. 벤치마크 출처 (benchmark-only)

엔진 입력(1–6장)과 겹치지 않는 출처다. 엔진 파일은 이 이름들을 쓰지 않는다(기존 `the_engine_never_names_*` 시험처럼 이름 검사를 붙일 것). 프로젝트가 이미 벤치마크 전용으로 둔 이름(Hanson, Beschorner, Rodríguez-Martínez, Lejeune, Knapik, Middleton, Frey Law, maxon, Mahler, ten Pas, Kalashnikov, Guillery 2013, Berscheid 2020, Menzer & Reiser 2005, Liberty Mutual, Pai 2010, Lockhart 2003, Wang 2017, Zhang 2013, Wallace 2018, Baines 2018, Agarwal 2019)은 이 문서의 엔진 입력에 하나도 없다.

| # | 출처 | 잰 것 | 쓸 수 있는 칸 |
|---|---|---|---|
| B1 | Menz, Lord & Fitzpatrick 2003, *Gait & Posture* 18:35–46 | 젊은 성인 30명이 평평한 복도와 불규칙한 바닥(인조 잔디 아래 폼·나무 블록)을 걸을 때 머리·골반 가속: 불규칙한 바닥에서 골반 가속은 커지지만 머리 가속은 바닥에 영향받지 않음 [초록] | 다리형 머리 카메라의 "평지 대비 험지 머리 가속 비" (사람 ≈ 1, 골반 > 1) |
| B2 | Milella, Reina & Nielsen 2019, *Precision Agriculture* 20:423–444 (arXiv:2104.05259) | 4륜 스키드 스티어 로봇(섀시 0.54 × 0.70 m), 약 0.8 m/s 직진, 수직 가속 RMS: 풀밭 0.050, 갈아엎은 흙 0.065, 돌 포장길 0.085 m/s² [직접] | 바퀴형 섀시 진동의 크기와 지면 순서 (단단할수록 큼) |
| B3 | Moon, Leary, Nguyen & McCrory 2025, *Sports Biomechanics* 24:2766–2778 | 20명이 달릴 때 경골 피크 가속: 아스팔트 11.0 ± 2.7 g, 잔디 10.7 ± 3.4 g, 트랙 10.6 ± 2.3 g, 트레드밀 7.9 ± 1.6 g [초록] | 지면별 디딤 충격 비 (사람: 잔디/아스팔트 ≈ 0.97) |
| B4 | Waite, Goetschius & Lauver 2021, *J Appl Biomech* 37:2–5 | 숙련 주자의 경골 피크 가속, 평지: 잔디 8.22 ± 1.22 g > 콘크리트 7.47 ± 1.65 g; 오르막: 잔디 7.68 > 아스팔트 6.99 g [초록] | B3과 같은 칸의 두 번째 기준 |

읽는 법: B3·B4는 사람이 지면마다 다리 강성을 바꿔(§2.3) 충격이 거의 같다는 것을 보여 준다. 다리 강성을 바꾸지 않는 이 엔진은 잔디/콘크리트 비 0.36을 낼 것이고, 이 차이는 "사람다움"의 차이이지 물리 오류가 아니다 — 기준에 맞추려고 S를 고치지 않는다. B2의 순서(풀밭 < 포장)는 §3.3의 [추정] 등급(Grass C > Firm B)과 반대일 수 있다: 이 엔진이 틀릴 수 있는 칸으로 미리 적어 둔다. (참고로 머리/몸통 가속 감쇠는 Kavanagh, Morrison & Barrett 2005, *Eur J Appl Physiol* 94:468이 평지에서만 쟀다 — 지면 비교가 아니라 칸 후보에서 뺐다.)

## 8. 찾지 못한 것 · 한계

- 휴머노이드 로봇이 걸을 때 머리·몸통 IMU의 각속도·피크 g (1.3). 사람 값을 [대입]으로 씀.
- 사람의 걷는 중 몸통·머리 롤 진폭: 못 찾아 피치와 같게 [추정].
- 디딤 충격의 머리 펄스 길이: 달리기의 9–20 Hz 대역에서 [추정]. 걷기 발꿈치 충격의 지속 시간을 준 원문(Whittle 1999 리뷰, Verdini 2006)은 본문을 열지 못함.
- 마른 모래의 절대 강성: Barrett 1998의 비(1/4)만 [초록]으로 확인, 절대값은 본문을 열지 못함. Hayati 2020 표 2의 Gmax는 어느 낙하 높이 값인지 모름(0.4–0.6 m 범위로 둠).
- 다진 흙·자갈의 충격 측정: 없음. 토질 정적 계수와 탄성 반공간 식에서 [유도] — 지표 근처 흙은 구속압이 낮아 표의 계수보다 무를 수 있다.
- 얕은 물: 출처 없음 [추정].
- ISO 8608 등급과 우리 지면의 대응: 측정 대응을 못 찾아 [추정]. Wong *Theory of Ground Vehicles*의 지면별 PSD 표(활주로·고속도로·자갈길·목초지·밭)는 지수 N(3.8/2.1/2.1/2.1/1.6/1.6)만 확인되고 계수 C_sp는 확인 못 해 쓰지 않음. w = 2를 2.83 cycle/m 위로 늘이는 것은 표준 범위 밖.
- 소형 바퀴 로봇의 지면별 진동 RMS 표: Milella 2019 하나만 찾았고 벤치마크로 돌림. 마른 모래에서 바퀴가 가라앉아 요철을 덮는 효과는 넣지 않음.
- 롤링 셔터 판독 시간: 데이터시트에 없음, 최대 프레임률에서 상한 [추정]. D435i RGB 노출 단위(데이터시트 미기재)는 UVC 100 µs로 가정.
- AAA 시험 발의 지름은 [14]에 없음 — 지면 강성을 발 크기로 옮기지 않고 같은 크기로 봄(로봇 발꿈치 반지름 35 mm [추정]).
- 사람 경추 회전 Youdas 1992의 나이별 표는 원문을 열지 못해 Zárate-Tejero 2023의 중앙값과 그 논문이 인용한 회귀 계수만 씀.

## 9. 출처

[1] Pozzo T, Berthoz A, Lefort L (1990). Head stabilization during various locomotor tasks in humans. I. Normal subjects. *Experimental Brain Research* 82(1):97–106. doi:10.1007/BF00230842 (초록: Europe PMC)
[2] Hirasaki E, Moore ST, Raphan T, Cohen B (1999). Effects of walking velocity on vertical head and body movements during locomotion. *Experimental Brain Research* 127(2):117–130. doi:10.1007/s002210050781 (초록: Europe PMC)
[3] Dunbar DC, Macpherson JM, Simmons RW, Zarcades A (2008). Stabilization and mobility of the head, neck and trunk in horses during overground locomotion: comparisons with humans and other primates. *Journal of Experimental Biology* 211:3889–3907. doi:10.1242/jeb.020578 — 표 2(변위), 표 3(속도); 사람 열은 Cromwell & Wellmon 2001, Cromwell et al. 2004, Pozzo et al. 1990 인용. https://pmc.ncbi.nlm.nih.gov/articles/PMC2768006/
[4] Peters BT, van Emmerik REA, Bloomberg JJ (2008). "Far" and "Near" visual acuity while walking and the collective contributions of non-ocular mechanisms to gaze stabilization. NASA NTRS 20080026154 (원고). https://ntrs.nasa.gov/citations/20080026154
[5] Grossman GE, Leigh RJ, Abel LA, Lanska DJ, Thurston SE (1988). Frequency and velocity of rotational head perturbations during locomotion. *Experimental Brain Research* 70:470–476. doi:10.1007/BF00247595
[6] MacDougall HG, Moore ST (2005). Marching to the beat of the same drummer: the spontaneous tempo of human locomotion. *Journal of Applied Physiology* 99:1164–1173. doi:10.1152/japplphysiol.00138.2005
[7] Light LH, McLellan GE, Klenerman L (1980). Skeletal transients on heel strike in normal walking with different footwear. *Journal of Biomechanics* 13(6):477–480. doi:10.1016/0021-9290(80)90340-1 (초록: https://wbldb.lievers.net/10373055.html)
[8] Smeathers JE (1989). Transient vibrations caused by heel strike. *Proc IMechE H* 203:181–186. doi:10.1243/PIME_PROC_1989_203_036_01
[9] Lafortune MA, Lake MJ, Hennig EM (1996). Differential shock transmission response of the human body to impact severity and lower limb posture. *Journal of Biomechanics* 29:1531–1537. doi:10.1016/S0021-9290(96)80004-2
[10] Liew BXW, Zhu Y, Zhai X, McErlain-Naylor SA, McManus J (2024). Association between fat and fat-free body mass indices on shock attenuation during running. *Journal of Biomechanics* 165:112025. doi:10.1016/j.jbiomech.2024.112025 — 3–8 Hz / 9–20 Hz 대역을 Derrick, Hamill & Caldwell 1998 (*Med Sci Sports Exerc* 30:128–135)에 돌림 (검색 요약으로만 확인)
[11] Pretto A, Menegatti E, Bennewitz M, Burgard W, Pagello E (2009). A visual odometry framework robust to motion blur. *IEEE ICRA 2009*, pp. 2250–2257.
[12] Johnson KL (1985). *Contact Mechanics*. Cambridge University Press — 강체 평원판 압입 P = 2aE*δ.
[13] Fine software, GEO5 도움말 "Secant modulus of soil Es" (Ranjan & Rao 2000; Briaud 2001 인용). https://www.finesoftware.eu/help/geo5/en/secant-modulus-of-soil-es-02/ ; Bowles JE (1997) *Foundation Analysis and Design* 5판의 지반 반력 계수(느슨한 모래 4,800–16,000 kN/m³ 등)는 Strand7 도움말에서 2차 확인. https://www.strand7.com/strand7r3help/Content/Topics/SpecialTopics/SpecialModulusSubgradeReaction.htm
[14] James D, Thelwell M, Forrester S, Fleming P, Mills K, Benetti M, Billingham J (2026). Comparison of the vertical compliance and rotational stiffness of natural, hybrid and artificial football surfaces. *Sports Engineering* 29:13. doi:10.1007/s12283-026-00546-7 (CC BY, https://shura.shu.ac.uk/37268/) — §2.1 AAA 장치, 고찰의 강성 k(천연 301, 하이브리드 513, ELT 충전 인조 201, 비충전 인조 79 N/mm)
[15] Hayati H, Eager D, Peham C, Qi Y (2020). Dynamic behaviour of high performance of sand surfaces used in the sports industry. *Vibration* 3(4):410–424. doi:10.3390/vibration3040026 (https://opus.lib.uts.edu.au/handle/10453/145179) — 표 2
[16] Barrett RS, Neal RJ, Roberts LJ (1998). The dynamic loading response of surfaces encountered in beach running. *Journal of Science and Medicine in Sport* 1(1):1–11. doi:10.1016/S1440-2440(98)80003-0 (초록)
[17] Fahmi S, Focchi M, Radulescu A, Fink G, Barasuol V, Semini C (2020). STANCE: Locomotion adaptation over soft terrain. *IEEE Transactions on Robotics*; arXiv:1904.12306
[18] Žuraulis V, Sivilevičius H, Šabanovič E, Ivanov V, Skrickij V (2021). Variability of gravel pavement roughness: an analysis of the impact on vehicle dynamic response and driving comfort. *Applied Sciences* 11:7582. doi:10.3390/app11167582 — 식 (3) Ga(n) = (2πn)⁴Gd(n); ISO 8608의 w = 2가 자갈 포장에는 맞지 않는다는 지적
[19] Agostinacchio M, Ciampa D, Olita S (2014). The vibrations induced by surface irregularities in road pavements – a Matlab® approach. *European Transport Research Review* 6:267–275. doi:10.1007/s12544-013-0127-8 — 표 1 (ISO 8608 한계값)
[20] Ribeiro JGT, Cunha A Jr (2026). Scenario-driven optimization of passive vehicle suspensions: explaining the effectiveness of asymmetric damping. arXiv:2605.05235 — 식 (11), 표 1 (ISO 8608:2016 등급 설명)
[21] RealSense (Intel). RealSense D400 Series Product Family Datasheet, Document 337029-017, Revision 023, March 2026. https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf
[22] OmniVision OV2740 제품 페이지·제품 개요 (1080p 60 fps, 720p 90 fps). https://www.ovt.com/products/ov2740/
[23] USB Video Class 1.5, Exposure Time (Absolute) Control: 100 µs 단위 [2차: Linux V4L2 문서 "Camera Control Reference", V4L2_CID_EXPOSURE_ABSOLUTE — 100 µs 단위, 프레임 간격에 묶임]. https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html
[24] Bosch Sensortec. BMI055 Data sheet, BST-BMI055-DS000-08, Revision 1.2, July 2014.
[25] Bosch Sensortec. BMI085 Data sheet, BST-BMI085-DS000-16, Revision 1.6, November 2021. https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi085-ds001.pdf
[26] Luxonis. OAK-D Lite Datasheet (December 2021). https://core-electronics.com.au/attachments/localcontent/OAK-D-Lite_Datasheet_16676646d51.pdf
[27] Luxonis 문서 "OAK-D Lite". https://docs.luxonis.com/hardware/products/OAK-D%20Lite
[28] Bosch Sensortec. BMI270 Data sheet, BST-BMI270-DS000-04, Revision 1.2, June 2020. https://cdn.sparkfun.com/assets/9/a/2/9/6/bst-bmi270-ds000.pdf
[29] Luxonis DepthAI v2 예제 "RGB Camera Control" · "Mono Camera Control" (노출 1..33000 µs, ISO 100..1600). https://docs.luxonis.com/software/depthai/examples/rgb_camera_control.md
[30] ISO 2720:1974 (반사광 노출계 보정 상수 K 10.6–13.4) [2차]; Sekonic, "How are your exposure meters calibrated" (K = 12.5). https://answers.sekonic.com/5851973/How-are-your-exposure-meters-calibrated
[31] Fetch Robotics. fetch_description/robots/fetch.urdf (melodic-devel). https://github.com/ZebraDevs/fetch_ros
[32] PAL Robotics. tiago_description/urdf/head/head.urdf.xacro (humble-devel). https://github.com/pal-robotics/tiago_robot
[33] Willow Garage. pr2_description/urdf/head_v0/head.urdf.xacro (melodic-devel). https://github.com/PR2/pr2_common
[34] Zárate-Tejero C, Hidalgo-García C, Lucha-López O, Hernández-Secorún M, Krauss J, Rodríguez-Rubio P (2023). Association between age, sex and cervical and upper cervical rotation tests. *Therapeutic Advances in Chronic Disease* 14:20406223231170158 — 표 2; Youdas JW et al. (1992) *Physical Therapy* 72:770–780의 회귀 계수 인용
[35] Waveshare. ST3215 Servo (12 V판 0.222 s/60°, 7.4 V판 0.192 s/60°). https://www.waveshare.com/st3215-servo.htm
[36] ROBOTIS e-Manual. XM430-W350. https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/
[37] Ferris DP, Louie M, Farley CT (1998). Running in the real world: adjusting leg stiffness for different surfaces. *Proc R Soc B* 265:989–994. doi:10.1098/rspb.1998.0388 (초록; 엔진 입력 아님, §2.3의 해석에만)
[38] Kim T, Kim S, Lee D (2023). Tunable impact and vibration absorbing neck for robust visual-inertial state estimation for dynamic legged robots. *IEEE RA-L* 8(3):1431–1438. doi:10.1109/LRA.2023.3240369 (초록만; 수치 없음, §1.3)

벤치마크 전용 출처 B1–B4는 7장 표에 전체 서지를 적었다.
