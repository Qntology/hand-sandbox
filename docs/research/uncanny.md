# 영역 사이의 "어색함": 셀 하나, 판정 하나, 영역별 기준 편차 보고

조사일 2026-10-26. 이 문서에서 온 코드: `crates/noto-core/src/bench.rs`(공통 셀·판정·요약·로봇 행렬·출처 등록부), `crates/noto-core/tests/uncanny_report.rs`(`[UX]` 보고, 코어 영역), `crates/noto-core/tests/source_split.rs`(엔진/채점 출처 분리, 등록부 전체), `crates/noto-runtime/tests/uncanny_runtime.rs`와 `tests/camvx/`(카메라·시각 영역을 같은 보고에), `crates/noto-sds/src/store.rs`(편차 크기 축·영역 보고·범위 정책·엔진 개정 표시), `crates/noto-runtime/src/{actsds,benchsds,nnlive}.rs`(실시간 SDS), `test_runner.bat`·`test_summary.ps1`(실행·수집). 영역마다의 기준값과 출처는 각 영역 문서에 있다: 행동 [`actions.md`](actions.md), 환경 효과 [`effects.md`](effects.md), 복합 N:N [`nn-scenarios.md`](nn-scenarios.md), 머리 카메라 [`camera.md`](camera.md), 간헐 시각 [`perception.md`](perception.md).

표기는 다른 문서와 같다: **[직접]** 출처에 인쇄된 값, **[유도: 식]** 이 문서에서 계산한 값, **[추정]** 출처 없이 정한 값. **벤치마크는 시간 측정이 아니라 기준 점수 시험**이다(속도 측정은 돌리지 않았다).

## 0. 요청과 반영

요청 R9: 로봇 부위 × 환경 × 협동의 N:N을 자료(매뉴얼, 실측)로 다듬고, SDS 논리도 함께 고치고, 남은 "어색함"(실제 기준과의 차이)이 영역마다 대체로 같아지게 벤치마크하라.

| 요청의 부분 | 이전 (v0.9) | 이번 (v1.0 wave 1, 패키지 `bench`) |
|---|---|---|
| 영역끼리 비교 | 판정 함수가 다섯 벌(`actbench`, `effects`, `nn`, 카메라·시각 시험), 빈 시나리오는 영역마다 0 · 100 · NaN. 점수는 \|z\| ≥ 10에서 0으로 포화하고 게이트가 셀의 z를 가림 → 영역끼리 비교 불가 | 판정 하나(`bench::judge`: 닮음 · 하한 · 상한 게이트 · 기록 · "더 나쁘지 않음"), 셀 하나(`BenchCell`), "기록만" 시나리오를 따로 셈. 영역마다 기준 셀 수, 중앙 \|z\|, RMS z, 최대 \|z\|, \|z\| ≤ 1·≤ 2 비율, 게이트, 미구현, 홀드아웃 통과를 같은 식으로 계산(`Summary`)하고 영역 사이 퍼짐(`Spread`: 최대/최소 중앙 \|z\|, 변동계수)으로 가장 먼 영역을 다음 작업으로 지목 |
| 기존 줄 | – | `[AX]`·`[FX]`·`[NN]`·`[CAM]`·`[VX]` 줄은 한 글자도 바뀌지 않았다(변환은 가장자리에서; 4장) |
| 로봇 행렬 | 시험마다 따로 적은 몸 × 베이스 목록, 메카넘은 한 번도 안 돎 | `bench::forms()`: 몸 5종(budget, mid, pro, pro_arms, wheel_lift) × 베이스 3종(다리, 바퀴, 트랙) + mid 메카넘 = 16 형태, 홀드아웃은 mid·pro에서. 형태 id는 실시간 SDS와 같은 `form_id` (`mid:humanfoot:f5`) |
| 출처 분리 검사 | 시험마다 손으로 적은 엔진 파일 목록, 새 모듈은 검사 밖 | 중앙 등록부(`ENGINE_FILES` 패턴 → 새 파일도 그날부터 검사), 모든 엔진 파일 × 모든 영역의 채점 출처, 결정 파일(driver·mldrive·vision·precision)은 기준 축·벤치 모듈을 읽지 않음을 시험이 확인. 기존 시험별 검사는 그대로 둠(등록부가 상위 집합) |
| SDS 편차 | 기준 편차는 부호 있는 평균뿐(+z와 −z가 상쇄), 엔진 개정 구분 없음, 형태·모드를 가리지 않고 묶음 | `:absdev`·`:dev2` 축과 `mean_abs_dev`·`rms_dev`, 트랙/접두사별 `area_report`, 보조 범위를 묶지 않는 정책(선택), 실행 파일·원장에 엔진 개정 표시와 다른 개정 원장을 치워 두는 읽기 옵션 |
| SDS 행동 트랙 | 집기·놓기·밀기만, 실패 코드는 늘 참, `p_success`는 비율(실패면 0.05로 고정), `no_fall`은 늘 거짓 | 걷기·주행도 층으로, 미끄러짐·넘어짐·충돌은 일어날 때, 바닥 표본은 위상과 무관, 밀기도 부위·결합 층, 시도마다 결과 하나(`attempt>코드` 횟수), `p_success`는 Brier 점수와 신뢰도 구간, 성공한 걸음의 `no_fall=true`, 형태 id에 손가락 수 |
| 실시간 N:N | 사다리의 모든 단계 가중을 더함 | `nn::combine`처럼 사다리는 가장 나쁜 단계 하나 |
| 새 영역 자리 | – | 충돌 `CX`, 촉각 `TX`, 협동 `KX`, 형태 `MX`: 보고의 영역 목록(`core_areas`)에 한 줄로 등록, 채점 출처는 등록부에 미리 등록(2장), 실시간 기록 고리는 `benchsds.rs` |
| 실행·수집 | 실행기는 행동·효과·카메라·시각 벤치를 돌리지 않고 요약은 `[AX]`·`[FX]`·`[CAM]`·`[VX]`를 버림 | 9–19단계(벤치 8개 — 없는 파일은 "건너뜀" —, 출처 분리, `[UX]` 두 개), 태그 `AX|FX|CAM|VX|CX|TX|KX|MX|UX` (+ `-SUM`·`-HELD`·`-WHY`·`-CELL`), 요약에 영역별 시나리오 줄 수와 `[UX]` 절 |

## 1. Group A — 엔진 입력

**이 패키지는 엔진 상수를 하나도 더하거나 바꾸지 않았다.** 바뀐 것은 채점(판정 함수를 하나로 모음), 보고, 출처 검사, SDS 기록뿐이고, 엔진(`bench.rs`가 아닌 noto-core, world-sim, noto-vision, 런타임 결정 파일)은 이 중 어느 것도 읽지 않는다(`decisions_never_read_reference_axes`, `the_engine_never_names_any_benchmark_source`).

| 엔진 쪽 변경 | 값 | 근거 |
|---|---|---|
| 없음 | – | – |

참고로 런타임이 SDS에 기록하는 걸음·주행 층(`actsds.rs`)은 엔진이 이미 쓰는 모델(`gait::plan`, `gait::pose_at`, `drive::state`, `surface::available_mu`)의 값을 그대로 기록할 뿐 새 상수가 없다. 프루드 수 `v²/(g·ℓ)`의 다리 길이 ℓ은 기존 `parts::HIP_M`이다.

## 2. Group B — 벤치마크(채점 전용) 출처

### 2.1 기존 영역

영역마다의 기준과 출처 표는 그 영역 문서에 그대로 있다(이 패키지는 기준값을 하나도 바꾸지 않았다). 등록부 `BENCH_SOURCES`의 행은 기존 시험별 검사의 금지 이름을 그대로 옮겼다.

| 영역 | 기준 표 | 등록부의 이름 (일부) |
|---|---|---|
| `AX` 행동 | [`actions.md`](actions.md) 3.2 | Guillery, Menzer, Reiser, Liberty Mutual, Steele, Mahler, Dex-Net, Berscheid, Beschorner, Pai 2010, Lockhart, Zhang 2013, Wallace, Baines, Agarwal … (`action_bench`의 목록 그대로) |
| `FX` 환경 효과 | [`effects.md`](effects.md) 4.3 | Hanson, Beschorner, Rodríguez-Martínez, Lejeune, Knapik, Middleton, Frey Law, maxon |
| `NN` 복합 N:N | [`nn-scenarios.md`](nn-scenarios.md) 1·2장 | Jbabdi, de Kam, Krašna, Schlenstedt, Kanekar, Mirka, Franchignoni, Horak 2009, Padgett, Hignett, McAtamney, Corlett, BESTest, REBA, RULA, DARPA … — 엔진이 정의로 쓰는 단일 원인 표의 출처(Hof 2005의 여유, Winter의 분절 질량, Lin 2004의 계단 높이)는 이름으로 구분할 수 없어 넣지 않았다 |
| `CAM` 머리 카메라 | [`camera.md`](camera.md) 7장 | Milella, Reina, Moon 2025, Waite, Menz, Kavanagh |
| `VX` 간헐 시각 | [`perception.md`](perception.md) 3장 | Loomis, Rieser, Thomson, Steenhuis, Elliott, Madalena, Patla, TidyBot, `Dusty, King`(로봇 이름만 쓰면 손·패드의 "먼지 낀" 상태 이름과 겹침), Deyle … |
| `CODE` | – | `actbench`, `effects::`, `noto_core::nn::`, `composite_reference` (벤치 모듈을 부르는 코드) |

### 2.2 새 영역 (미리 등록, 연구 묶음의 benchmark_refs에서)

충돌·촉각·협동·형태 벤치마크는 아직 없지만, 그 영역을 위해 새로 쓰이는 엔진 파일이 첫날부터 검사받도록 채점 출처를 등록부에 먼저 넣었다. **어느 연구 묶음에서도 엔진 입력(Group A)으로 나오지 않는 이름만** 넣었고(스크립트로 묶음 일곱 개의 engine_inputs 전체와 대조), 흔한 성은 공저자나 "et al"과 함께 적었다. 이름은 단어 경계로 맞춘다(`Jacob`은 `Jacobian`에 걸리지 않는다).

| 영역 | 출처 (저자, 제목, 연도, URL) | 등록한 이름 |
|---|---|---|
| `CX` 충돌·넘어짐 | Choi W.J., Wakeling J.M., Robinovitch S.N., "Kinematic analysis of video-captured falls experienced by older adults in long-term care", J Biomech 2015, https://doi.org/10.1016/j.jbiomech.2015.02.025 | `Choi et al`, `Wakeling` |
| 〃 | van den Kroonenberg A.J., Hayes W.C., McMahon T.A., "Hip impact velocities and body configurations for voluntary falls from standing height", J Biomech 29(6):807–811, 1996, https://doi.org/10.1016/0021-9290(95)00134-4 | `Kroonenberg` |
| 〃 | Tan J.S., Eng J.J., Robinovitch S.N., Warnick B., "Wrist impact velocities are smaller in forward falls than backward falls from standing", J Biomech 2006, https://doi.org/10.1016/j.jbiomech.2005.05.016 | `Tan et al` |
| 〃 | Hsiao E.T., Robinovitch S.N., "Common protective movements govern unexpected falls from standing height", J Biomech 31(1):1–9, 1998, https://doi.org/10.1016/s0021-9290(97)00114-0 | `Hsiao` |
| 〃 | Laing A.C., Robinovitch S.N., "Characterizing the effective stiffness of the pelvis during sideways falls on the hip", J Biomech 2010, https://doi.org/10.1016/j.jbiomech.2010.03.025 | `Laing` |
| 〃 | Kumar V.C.V., Ha S., Liu C.K., "Learning a Unified Control Policy for Safe Falling", arXiv:1703.02905, 2017, https://arxiv.org/pdf/1703.02905 | `Kumar et al` |
| 〃 | Liu Z., Karydis K., "Dynamic Modeling and Analysis of Impact-resilient MAVs Undergoing High-speed and Large-angle Collisions with the Environment", arXiv:2307.11309, 2023, https://arxiv.org/pdf/2307.11309 | `Karydis` |
| 〃 | Wu J.Z., Pan C.S., Wimer B.M., "Evaluation of the shock absorption performance of construction helmets under repeated top impacts", Eng Fail Anal 96:330–339, 2019, https://stacks.cdc.gov/view/cdc/216204 | `Wimer` |
| 〃 | Kim K.S., Llado T., Sentis L., "Full-Body Collision Detection and Reaction with Omnidirectional Mobile Platforms", arXiv:1501.05007, 2015, https://arxiv.org/pdf/1501.05007 | `Llado`, `Sentis` |
| 〃 (넣지 않음) | Feldman & Robinovitch 2007, Kajita et al. 2016 — Feldman·Robinovitch는 엔진 입력 Schonnop et al. 2013의 공저자, Kajita는 엔진 입력(HRP-2 발가락 관절, Sellaouti et al. 2006)의 공저자 | – |
| `TX` 손끝·발가락 | Westling G., Johansson R.S., "Factors influencing the force control during precision grip", Exp Brain Res 53:277–284, 1984, https://stacks.cdc.gov/view/cdc/185560 | (`AX` 행의 `Westling`) |
| 〃 | Delhaye B., Lefevre P., Thonnard J.-L., "Dynamics of fingertip contact during the onset of tangential slip", J R Soc Interface 11:20140698, 2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC4191101/ | `Delhaye`, `Thonnard` |
| 〃 | Hager-Ross C., "To grip and not to slip", Umeå University Medical Dissertations NS 429, 1995, https://www.diva-portal.org/smash/get/diva2:864178/FULLTEXT01.pdf | `Hager-Ross`, `Häger-Ross` |
| 〃 | Luciw M.D., Jarocka E., Edin B.B., "Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction (WAY-EEG-GAL)", Sci Data 1:140047, 2014, https://www.nature.com/articles/sdata201447 | `Luciw`, `WAY-EEG-GAL` |
| 〃 | Jacob H.A.C., "Biomechanics of the forefoot", PhD thesis, University of Strathclyde, 1989, https://stax.strath.ac.uk/concern/theses/b8515n370 | `Jacob` |
| 〃 | Hughes J., Clark P., Klenerman L., "The importance of the toes in walking", J Bone Joint Surg Br 72(2):245–251, 1990, https://pubmed.ncbi.nlm.nih.gov/2312564/ | `Hughes`, `Klenerman` |
| 〃 | Koo S. et al., "Effects of walking speed and slope on pedobarographic findings in young healthy adults", PLoS ONE 14(7):e0220073, 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6656354/ | `Koo et al` |
| 〃 | Hessert M.J. et al., "Foot pressure distribution during walking in young and old adults", BMC Geriatrics 5:8, 2005, https://pmc.ncbi.nlm.nih.gov/articles/PMC1173105/ | `Hessert` |
| 〃 | Chou S.W. et al., "The role of the great toe in balance performance", J Orthop Res 27(4):549–554, 2009, https://pure.lib.cgu.edu.tw/en/publications/the-role-of-the-great-toe-in-balance-performance-3/ | `Chou et al` |
| 〃 (넣지 않음) | Becker S. et al. 2025 — 성 Becker가 엔진 입력(Jindrich, Zhou, Becker & Dennerlein 2003)에도 있음 | – |
| `KX` 협동 운반 | Lavender S.A. et al., "Biomechanical analyses of paramedics simulating frequently performed strenuous work tasks", Appl Ergon 31(2):167–177, 2000, https://stacks.cdc.gov/view/cdc/190748; Lavender S.A. et al., "Designing ergonomic interventions for EMS workers, Part I", Appl Ergon 38:71–81, 2007, https://stacks.cdc.gov/view/cdc/185791 | (넣지 않음: Lavender 2021이 현장 높이의 엔진 입력) |
| 〃 | DGUV IFA, Project IFA 4224 (구급대원 계단 환자 이송의 신체 부하 측정), 2018–2019, https://www.dguv.de/ifa/forschung/projektverzeichnis/ifa4224-2.jsp | `DGUV`, `IFA 4224` |
| 〃 | Kluth K., Keller E., Strasser H., "Ergonomics in the Rescue Service, Part 1: Strain-oriented evaluation of ambulance cots", IOS Press, https://ebooks.iospress.nl/volumearticle/26150 | `Kluth` |
| 〃 | Dennis G.J., Barrett R.S., "Spinal loads during two-person team lifting" (Int J Ind Ergon 2003), https://research-repository.griffith.edu.au/items/794f5ff3-aed7-5405-9e59-f80fb4060ed0; "Spinal loads during individual and team lifting", Ergonomics 45(10), 2002, https://www.labourline.org/KENTIKA-19154516124919727989-Spinal-loads-during-individual.htm | `Dennis & Barrett`, `Dennis and Barrett` |
| 〃 | Sharp M.A., Rice V.J., Nindl B.C., Williamson T.L., "Effects of team size on the maximum weight bar lifting strength of military personnel", Human Factors 39(3):481–488, 1997, https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/11860865 | `Sharp et al`, `Nindl` |
| 〃 | Visser S. et al., "Lumbar compression forces while lifting and carrying with two and four workers", Appl Ergon 50:56–61, 2015, https://research.vu.nl/en/publications/lumbar-compression-forces-during-while-lifting-and-carrying-tasks/ | `Visser` |
| 〃 | Wu S.-P., Loiu Y., Chien T.H., "Effects of box handle position and carrying range on bi-manual carrying capacity for females", Ind Health 53(5):410–416, 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4591133 | `Loiu` |
| 〃 | Lanini J., Duburcq A., Razavi H., Le Goff C.G., Ijspeert A.J., "Interactive locomotion: Investigation and modeling of physically-paired humans while walking", PLoS ONE 12(9):e0179989, 2017, https://pmc.ncbi.nlm.nih.gov/articles/PMC5587243/ | `Lanini` |
| 〃 | Sambhus R.S. et al., "Safety-Critical Centralized Nonlinear MPC for Cooperative Payload Transportation by Two Quadrupedal Robots", arXiv:2604.03200, 2026, https://arxiv.org/html/2604.03200v1 | `Sambhus` |
| `MX` 몸 형태 | Wevolver "Fetch Mobile Manipulator" 사양, https://www.wevolver.com/specs/fetch.mobile.manipulator; Konica Minolta Australia MiR250, https://www.konicaminolta.com.au/?p=898; GIGAZINE Figure 02 (2024), https://gigazine.net/gsc_news/en/20240808-figure-02-ai-humanoid; iFixit "Spot Arm", https://www.ifixit.com/Device/Spot_Arm; Techzine Boston Dynamics Stretch (2022), https://www.techzine.eu/?p=76287 | `Wevolver`, `Konica`, `GIGAZINE`, `iFixit`, `Techzine` (제조사 페이지·Kemp 2022·Access Board는 엔진 입력에도 있어 넣지 않음) |

### 2.3 판정과 기록에 쓰는 통계 (방법)

| 방법 | 쓰는 곳 | 출처 |
|---|---|---|
| `z = (값 − 기준) / 허용폭`, 닮음 점수 `100 − 10·|z|` (허용폭 하나가 10점) | 모든 영역 (기존과 같음) | Gait Deviation Index의 척도 관례 ([`nn-scenarios.md`](nn-scenarios.md) 4장) |
| Brier 점수 `(y − p)²`와 신뢰도 도표(예측 확률 10구간별 관측 비율과 평균 예측) | SDS `p_success:brier`, `p_success:bin<k>`, `p_success:bin<k>:p` | Brier G.W., "Verification of forecasts expressed in terms of probability", Monthly Weather Review 78(1):1–3, 1950 |
| 중앙값, RMS, 변동계수(모표준편차 / 평균) | `Summary`, `Spread` | 정의 |

## 3. 모델 — 보고가 어떻게 만들어지나

### 3.1 셀 하나, 판정 하나

`BenchCell { area, scenario, layer, axis, part, form, value, rule, cite, z, score, gate, holdout }`. 규칙(`bench::Rule`)은 다섯 가지다.

| 규칙 | z | 점수 | 값이 없을 때 |
|---|---|---|---|
| `Like` 닮음 | `(v − r)/t` | `100 − 10·|z|` (하한 0) | 0점 ("미구현") |
| `AtLeast` 하한 | `min(0, (v − r)/t)` | 〃 (넘으면 100) | 0점 |
| `NoWorse` 오차 상한 (카메라·시각) | `max(0, (v − r)/t)` | 〃 (기준보다 작은 오차면 100) | 0점 |
| `AtMost` 공학·물리 한계 | – | – (넘으면 시나리오 게이트) | 0점 |
| `Raw` 기록 | – | – | 기록 |

`effects::Rule`은 필드 그대로 바뀌어 들어오고(`From`), `actbench`·`effects`·`nn`의 판정 함수는 `bench::judge`를 부른다. 게이트 문구는 영역마다 원래 문구를 그대로 쓴다(그래서 출력 줄이 같다). 카메라·시각 시험의 셀은 `BenchCell::new`로 바로 만들고 출력 함수 하나(`camvx::print_scenario`)를 같이 쓴다.

### 3.2 "기록만"과 영역별 빈 시나리오 규칙

영역의 원래 시나리오 점수는 그대로 두고(`[AX-SUM]` 등의 숫자), 공통 모양으로 옮길 때만 구분한다: 건너뜀 → 점수 없음(이유 기록), 게이트 → 0, **비교할 기준 셀이 하나도 없는 시나리오 → `None` = 기록만**. 원래 규칙(행동 0, 효과·N:N 100, 카메라·시각 NaN)은 각 영역의 출력에만 남고, 영역 사이 보고는 기록만 시나리오를 평균 점수에 넣지 않고 따로 센다.

### 3.3 영역의 편차 (`bench::Summary`)

개발 시나리오의 셀 중 기준이 있고(`Like`·`AtLeast`·`NoWorse`) 엔진이 답한 셀이 **기준 셀**이다. 그 \|z\|로:

- 중앙 \|z\| (영역 사이 비교의 주 지표 — 몇 개의 폭주 셀에 끌리지 않는다), RMS z, 최대 \|z\|, \|z\| ≤ 1과 ≤ 2의 비율;
- 게이트 걸린 시나리오 수와 게이트 셀 수(게이트 걸린 시나리오의 셀 z도 편차에 들어간다 — 게이트가 z를 가리지 않게);
- 미구현 셀(규칙이 있는데 값이 없는 셀) 수;
- 평균 점수(각 영역의 원래 집계로 낸 시나리오 점수의 평균; 기록만·건너뜀 제외);
- 홀드아웃: 편차에는 넣지 않고 통과/실행 수만(각 영역의 통과 규칙: 게이트 없음 + 점수 셀 \|z\| ≤ 2 등);
- 가장 먼 셀 셋까지(`[UX-CELL]`; z가 0인 셀 — 기준 위, 또는 한쪽 규칙의 좋은 쪽 — 은 "먼 셀"로 적지 않는다. 편차 통계에는 그대로 들어간다).

### 3.4 영역 사이 (`bench::Spread`)

기준 셀이 있는 영역들의 중앙 \|z\|로 최대·최소, 변동계수(모표준편차/평균), RMS z의 범위를 내고 **중앙 \|z\|가 가장 큰 영역을 다음 작업**으로 적는다. R9의 "어색함이 영역마다 대체로 같게"는 변동계수가 작아지는 것으로 읽는다. 단, 좋은 영역을 나쁘게 만들어 고르게 하는 일은 하지 않는다(5장).

### 3.5 로봇 행렬

`forms()` = 몸 `budget`, `mid`, `pro`, `pro_arms`, `wheel_lift` × 베이스 `HumanFoot`, `Wheel`, `Track` (다섯 손가락) + `mid` `Mecanum`. 영역 시험들이 따로 쓰던 목록의 합집합이고, 형태 패키지(`morph`)의 형태(몸통 없음, 집게만, 트럭)는 이 목록에 한 줄씩 더해진다. 형태 id(`form_id`)는 `몸 id:베이스:f손가락수`이고 실시간 SDS 행동 트랙의 보조 범위와 같다.

### 3.6 실시간 SDS

- `record_reference`가 값·부호 있는 편차·기준·허용폭·대역 안 여부에 더해 `:absdev`(\|z\|)와 `:dev2`(z²)를 쌓는다. `ReferenceReport`에 `mean_abs_dev`, `rms_dev`(그 이전에 쌓인 관측은 `None`). `SDS_RECIPE`는 그대로라 기존 원장은 버려지지 않는다.
- `area_report(track, 접두사)`: 주 범위(보조 범위를 모두 묶은 것, 그래서 두 번 세지 않음)의 모든 `:dev` 축을 관측 수로 가중해 평균 \|z\|, RMS z, 대역 안 비율과 가장 먼 축을 낸다 — 벤치 `Summary`의 실시간 짝.
- `ScopePolicy`: 트랙마다 읽기가 보조 범위에서 주 범위로(`KEEP_SECONDARIES` 금지), 주 범위에서 전체로(`EXACT` 금지) 물러나도 되는지. 기본은 원래처럼 묶음. 모델 모드(대체 모델 대 SigLIP2/TIPSv2)나 형태가 섞이지 않게 쓰는 선택이다(켜는 일은 ML 패키지의 몫).
- 엔진 개정 표시: `OpenOptions::stamped(rev)`로 연 저장소는 실행 파일에 `engine_rev`, 원장에 `engine_revs`(표시 없는 옛 관측은 `unstamped`)를 쓴다. `OpenOptions::only(rev)`는 다른 개정이 섞인 원장을 `<원장>.rev-<ms>.json`으로 옮겨 두고(지우지 않음) 빈 원장으로 시작한다. 런타임은 `bench::ENGINE_REV`(`noto-1.0.0-wave1`)를 표시하고 옛 관측은 그대로 읽는다(시험 `a_runtime_on_disk_stamps_its_ledgers_with_the_engine_revision`: 디스크에 연 런타임의 원장과 실행 파일에 그 개정이 적힘). 엔진 변경이 벤치마크 줄을 움직이면 이 문자열을 올린다.
- 행동 트랙(`actsds.rs`): 1층 결과 범위의 보조가 형태 id(`mid:humanfoot:f5`), 시도마다 `attempt>ok`·`attempt>grasp_slip` 같은 전이 횟수 하나, `p_success`는 Brier 점수와 신뢰도 구간. 바닥 표본은 위상과 상관없이 손바닥이 가장 낮은 순간. 밀기는 가장 부하가 큰 순간으로 부위 층과 범퍼→바퀴 결합 층(`bumper>wheel:front_share`, `:slip`, `need_over_limit`). 걷기·주행은 이동이 끝날 때 결과(`walk`/`drive`)·시계열(걸음 속도·보폭·주기·이중지지 비율·RCOF·프루드 수, 또는 순항 미끄럼·속도 계수)·부위(입각 중간 관절 비율, 또는 앞 차축 몫·트랙 최대 압력)·결합(발바닥 RCOF/μ, 여유; 경사→구동 필요/한계) 층. 미끄러짐·넘어짐·바퀴 미끄럼·막힘·충돌·밀다 놓침은 일어날 때 `act|event:<종류>|<형태>`, 걷다 미끄러진 뒤의 결과는 `slip>fall`/`slip>recover` 전이. `driver.rs`는 다리 로봇이 이동을 마치면 `no_fall=true`를 기록한다(넘어지면 `false`, 원래대로).
- `benchsds.rs`: 충돌(`bump`, `tip`, `hit`, `edge_fall` → `act|collide:<종류>|<형태>`), 촉각(`incipient_slip`, `heel_off`, `toe_push` → `act|tactile:…`), 협동(`yield`, `co_drop`, `overload_share` → `act|coop:…`)의 결과와 값. `Runtime::tick`에서 한 줄로 불린다. 월드가 아직 이 사건을 내지 않으므로 지금은 비어 있고, wave 2의 충돌·촉각·협동 물리가 사건을 내면 채워진다. 행동 키프레임 종류(`KEY_KINDS`)도 이 열 가지를 더해 32개.
- `nnlive.rs`: 실시간 시나리오 점수가 `nn::combine`처럼 사다리 단계들을 한 가중으로, 가장 나쁜 단계로 센다(이전에는 단계마다 가중을 더해 S4의 제동을 과대 반영).

### 3.7 출처 등록부 (`tests/source_split.rs`)

- `ENGINE_FILES`: `crates/noto-core/src/**/*.rs`, `crates/world-sim/src/**/*.rs`, `crates/noto-vision/src/**/*.rs`와 런타임의 driver·mldrive·vision·precision·runtime·record·cargo·chase·perception. noto-core에서 이름이 `bench.rs`로 끝나는 파일(`actbench.rs`, `bench.rs`, 앞으로 `collbench.rs`·`tactbench.rs`·`coopbench.rs` 등)은 벤치마크 모듈로 빠지고(`is_bench_module`; 다른 크레이트의 `bench` 이름 파일, 예를 들어 월드의 작업대 구조물은 엔진으로 검사받는다), `NOT_ENGINE`(noto-core의 `lib.rs`, `effects.rs`, `nn.rs`)은 이유와 함께 제외, `balance.rs`는 `/// Context the references depend on.` 표시 앞부분만 엔진(`REFERENCE_SECTIONS`). 오늘 55개 파일.
- 모든 엔진 파일 × 모든 영역의 이름(10행: AX, FX, NN, CAM, VX, CX, TX, KX, MX, CODE). 예외는 파일·이름·이유를 하나씩 적는다(`SOURCE_EXCEPTIONS`; 지금은 `vision.rs`의 Johansson et al. 2001 시선 시점 하나). 예외·제외에 적힌 파일이 없어지면 시험이 실패한다(이름이 바뀐 파일이 규칙을 조용히 잃지 않게).
- 결정 파일(driver·mldrive·vision·precision)은 `reference_report`, `exact_reference`, `area_report`, `:dev"`, `:absdev`, `:dev2`, `:in_band`, `actbench`, `effects::`, `nn::`, `composite_reference`, `reference(`, `bench::` 중 어느 것도 담지 않는다.
- 이름 맞추기(`bench::names`): 글자로 시작하는 이름은 앞에 식별자 글자가 붙으면 다른 단어로 본다(`candle_nn::`은 `nn::`이 아님). 글자로 끝나는 이름은 뒤에 식별자 글자가 붙으면 다른 단어다(`Jacobian`은 `Jacob`이 아님). `:`로 시작하는 축 접미사(`:in_band`, `:absdev`, `:dev"`)는 앞 경계를 보지 않아 어떤 축 이름 뒤에 와도 걸린다 — `"margin:in_band"`처럼 축 이름을 글자 그대로 적은 읽기도 놓치지 않는다(시험 `names_respect_identifier_boundaries`).

## 4. 벤치마크 결과

### 4.1 기존 줄은 그대로

같은 엔진·같은 시드로 `cargo test -p noto-core` 출력의 `[AX-SUM]`·`[FX]`·`[FX-SUM]`·`[NN]`·`[NN-SUM]` 991줄을 v1.0 이전 기준선과 비교해 **한 줄도 다르지 않았다**. `cargo test -p noto-runtime`의 `[CAM]`·`[CAM-SUM]` 8줄, `[VX]`·`[VX-SUM]` 24줄, 퀘스트 중 실시간 `[NN-SUM]` 6줄도 같다(실시간 S4는 사다리 단계를 하나만 봐서 새 결합 규칙에서도 92.2 그대로). 합쳐 1029줄, 2026-10-27에 다시 돌려 같은 결과를 확인했다(줄을 정렬한 다중집합 비교).

### 4.2 영역별 편차 (`cargo test -p noto-core --test uncanny_report -- --nocapture`, 16 형태)

개발 시나리오만(홀드아웃은 통과 수만). "칸"은 모든 셀, "기준 칸"은 기준이 있고 엔진이 답한 셀.

| 영역 | 시나리오 (기록만 · 건너뜀 · 게이트) | 칸 | 기준 칸 | 중앙 \|z\| | RMS z | 최대 \|z\| | \|z\| ≤ 1 | \|z\| ≤ 2 | 미구현 | 평균 점수 | 홀드아웃 통과 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `AX` 행동 | 240 (60 · 15 · 70) | 2947 | 666 | 1.90 | 10.22 | 39.6 | 34% | 54% | 412 | 49.1 | 15/36 |
| `FX` 환경 효과 | 96 (0 · 19 · 15) | 982 | 195 | 2.00 | 3.01 | 6.6 | 21% | 50% | 0 | 63.7 | 10/48 |
| `NN` 복합 N:N | 80 (0 · 22 · 6) | 1808 | 1129 | 1.33 | 10.44 | 143.8 | 43% | 64% | 0 | 74.3 | 4/56 |

`[UX-SUM]`: 영역 3개 · 중앙 \|z\| 최대 2.00 (FX) / 최소 1.33 (NN) · 변동계수 0.17 · RMS z 3.01–10.44 · 다음 작업: FX.

가장 먼 셀 (`[UX-CELL]`):

- `AX`: 보급형 다리 G1–G3의 성공률 0 vs 0.95 ± 0.024 (z −39.6). 손은 바닥에 닿지만 굽힌 자세를 고관절이 버티지 못하는 몸이다(`[AX-WHY]`). 행동 영역의 "기록만" 60개는 바퀴·트랙의 이동 시나리오(W1–W6)로, `[AX-SUM]`이 이미 "점수 — (기준 칸 없음)"으로 찍던 것들이다(바퀴형에 맞는 사람·로봇 기준이 아직 없다). 미구현 412칸은 몸이 그 동작에 이르지 못해 값이 없는 셀이다.
- `FX`: 바퀴 E1 모래 견인 0.25 vs 0.066 ± 0.028 (z +6.6) — [`effects.md`](effects.md) 7장에 이미 적은 차이(미끄러짐률별 추력이 없음)다.
- `NN`: S3 한 발 뻗기의 `ankle_rise` 14.7 vs 0.33 ± 0.10 (z +144) — 6장.

코어 세 영역의 중앙 \|z\|는 1.3–2.0으로 가깝다(변동계수 0.17). RMS z는 3–10으로 크게 다른데, 행동과 N:N의 RMS를 몇 개의 폭주 셀이 정한다.

### 4.3 런타임 영역을 더한 보고 (`cargo test -p noto-runtime --test uncanny_runtime -- --nocapture`)

코어 세 영역(위 표와 같은 숫자)에 머리 카메라(`camera_bench`의 C1·C2)와 간헐 시각(`vision_bench`의 V1–V3)을 더했다. 두 영역의 셀은 `tests/camvx/`에 한 벌만 있고 기존 시험과 보고가 같이 쓴다(`[CAM]`·`[VX]` 줄은 기준선과 같음).

| 영역 | 시나리오 (기록만 · 건너뜀 · 게이트) | 칸 | 기준 칸 | 중앙 \|z\| | RMS z | 최대 \|z\| | \|z\| ≤ 1 | \|z\| ≤ 2 | 미구현 | 평균 점수 |
|---|---|---|---|---|---|---|---|---|---|---|
| `CAM` 머리 카메라 흔들림 | 2 (0 · 0 · 0) | 6 | 4 | 11.13 | 15.89 | 25.8 | 0% | 0% | 0 | 0.2 |
| `VX` 간헐 시각과 공간 기억 | 4 (1 · 0 · 0) | 20 | 3 | 0.00 | 1.09 | 1.9 | 67% | 100% | 0 | 93.7 |

`[UX-SUM]`: 영역 5개 · 중앙 \|z\| 최대 11.13 (CAM) / 최소 0.00 (VX) · 변동계수 1.22 · RMS z 1.09–15.89 · **다음 작업: CAM 머리 카메라 흔들림**.

- `CAM`: 바퀴 섀시의 수직 가속 RMS가 풀밭 0.259 · 갈아엎은 흙 0.518 · 자갈 0.259 m/s²로 4륜 스키드 스티어의 측정값(0.050 · 0.065 · 0.085)의 3–8배이고 지면 순서도 맞지 않는다(z +25.8, +11.9). 다리의 디딤 충격 비(풀/단단한 바닥)는 0.36 vs 1.035 ± 0.065 (z −10.4): 엔진은 무른 땅이 충격을 줄인다고 보지만 사람은 무른 땅에서 다리를 굳혀 비슷한 충격을 받는다. 바퀴 형태의 카메라 모델(R6, wave 2)이 이 영역의 다음 작업이다 — 기준에 맞게 계수를 고르는 것이 아니라 현가·지면 등급 모델로.
- `VX`: 눈 감고 걷기 오차가 사람의 기준보다 작고(`NoWorse`이므로 z 0), 바닥 물건 집기 성공률 1.0이 로봇 기준 0.964 ± 0.019보다 높다(z +1.9). 중앙 0은 "사람과 같다"가 아니라 "기준보다 나쁘지 않다"는 뜻이다.
- 영역 사이 변동계수 1.22: 카메라는 기준에서 가장 멀고 시각은 가장 가깝다. 다섯 영역이 "대체로 같아지려면" 카메라 모델이 먼저 나아져야 한다.

## 5. 벤치맥싱을 막는 장치

| 규칙 | 구현 |
|---|---|
| 보고는 순위를 매길 뿐 상수를 바꾸지 않는다 | `bench.rs`는 엔진 검사에서 빠지는 벤치마크 모듈이고 엔진 계산은 그 안의 어떤 값도 읽지 않는다(런타임이 SDS 원장에 적는 개정 문자열 `ENGINE_REV` 하나만 `runtime.rs`가 쓴다), 결정 파일은 `bench::`를 담을 수 없다(시험). "다음 작업"은 가장 먼 영역의 **모델**을 고치라는 뜻이고 그 영역의 허용폭·기준을 바꾸라는 뜻이 아니다 |
| 고르게 = 나쁜 영역을 좋게 | 변동계수는 좋은 영역을 망쳐도 줄어든다. 그래서 보고는 영역마다의 중앙 \|z\|를 함께 찍고, 어느 영역의 중앙 \|z\|가 커지는 변경은 개선으로 치지 않는다 |
| 홀드아웃은 세기만 | 홀드아웃 시나리오는 편차 통계에 넣지 않고 통과/실행 수만 낸다. 시드(`HOLDOUT_SEED` 20261024)는 이 릴리스에서 고정 |
| 허용폭·기준 고정 | 이 패키지는 기준·허용폭을 하나도 바꾸지 않았다. 4.1의 줄 비교가 그것을 보인다 |
| 중앙값이 주 지표 | 분모가 0에 가까운 비율 축(아래 6장의 `ankle_rise`)이 RMS z를 수십으로 끌어올린다. 그런 셀을 "고치기 쉬운 숫자"로 보지 않도록 중앙값으로 영역을 비교하고 RMS·최대는 함께 기록한다 |
| 출처 분리 | 등록부 검사가 모든 엔진 파일을 덮는다(새 파일 포함). 이름만 보는 검사라는 한계는 6장 |

## 6. 남은 차이 (숨기지 않음)

- **이름 검사의 한계**: 기준값(예: 0.53, 133 N)을 저자 이름 없이 엔진에 옮겨 적으면 검사를 통과한다. 반대로 같은 사람이 엔진 입력과 채점 출처에 함께 나오면 거짓 양성이 난다. 연구 묶음끼리 이미 겹치는 이름이 있다: Pozzo 1990(NN 게이트, 카메라 묶음의 채점 출처 ↔ 형태 묶음의 엔진 입력), Johansson·Flanagan(AX 채점 ↔ 촉각 묶음의 엔진 입력 Johansson & Flanagan 2009), NIOSH·Waters(N:N 가중 ↔ 협동 묶음의 엔진 입력 Waters 1993), Keller(AX 목록 ↔ 현장 묶음의 J.J. Keller), Steele·Menz(AX·CAM ↔ 촉각 묶음의 Mickle et al. 2009 공저자). 이 파일들이 합쳐질 때 검사가 걸리면, 정말 같은 출처로 채점하는지 보고 엔진 쪽 출처를 바꾸거나 `SOURCE_EXCEPTIONS`에 이유를 적어 하나씩 푼다.
- **허용폭의 뜻이 영역마다 다르다**: 표준편차(대부분), 범위의 반폭(모래 견인), 이항 표준오차(미끄럼 비율), 선택값(경사 들기 모멘트). 그래서 \|z\|를 영역끼리 비교하는 것은 근사다.
- **기준 셀이 적은 영역**: 카메라 4셀, 시각 3셀. 중앙값이 한두 셀로 정해진다. 새 기준(머리 가속 비 등)이 들어오기 전에는 이 두 영역의 순위를 무겁게 읽지 않는다.
- **한쪽 규칙은 "더 좋음"을 0으로 친다**: `NoWorse`·`AtLeast` 셀은 기준보다 좋은 쪽의 z를 0으로 자르므로, 사람보다 지나치게 정확한 로봇(시각 영역)도 편차 0으로 보인다. 사람다움이 목적인 셀이라면 `Like`로 바꿔야 하지만, 그것은 기준의 뜻을 바꾸는 일이라 이 패키지에서는 하지 않았다.
- **분모가 작은 비율 축**: N:N `ankle_rise`(발목 토크 / 조용히 선 때의 발목 토크 − 1)는 한 발 서기(S3)에서 조용히 선 때의 토크가 작아 값이 14.7(기준 0.33 ± 0.10, z +144)이 된다. 기준(+33%)은 두 발로 서서 팔을 드는 상황에서 온 것이라, 축의 정의를 상황에 맞게 바꿀지(예: 체중·발 길이로 정규화) 검토가 필요하다. 이 패키지는 셀을 바꾸지 않고 기록했다.
- **답하지 못하는 몸**: 행동 영역의 RMS z는 바닥에 손이 닿지 않는 몸(보급형 다리, 바퀴의 허리 없음)의 성공률 셀(0 vs 0.95 ± 0.024, z −40)이 끌어올린다. 몸이 못 하는 일을 사람 성공률과 비교하는 셀이라, 형태 행렬이 들어오면 "할 수 없음"을 건너뜀으로 둘지 다시 정해야 한다.
- **새 영역은 아직 보고에 없다**: CX·TX·KX·MX는 출처 등록과 실시간 고리만 있고, 벤치마크(wave 1의 `physcore`·`contact`·`morph`)가 들어오면 `core_areas()`에 한 줄씩 등록한다.
- **실시간 SDS의 새 고리는 비어 있다**: 월드가 `bump`·`tip`·`co_drop` 같은 사건을 아직 내지 않는다(wave 2).
- **범위 정책은 켜지 않았다**: 형태·모델 모드를 섞지 않는 정책은 선택이고 기본은 원래처럼 묶는다. 켜면 범위마다 관측이 줄어 냉간 시작이 길어진다(시각 바닥값·깊이 배율의 행동이 바뀜) — ML 패키지가 A/B로 정한다.
