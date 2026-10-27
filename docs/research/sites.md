# 집 · 현관 계단 · 물류센터: 택배가 놓이는 곳 — 리서치, 벤치마크, 구현 기록

조사일 2026-10-26. 이 문서에서 온 코드: `crates/world-sim/src/site.rs`(집 · 현관 · 계단 · 경사로 · 물류센터의 배치와 택배 자리, 찾기 사전 지식), `crates/world-sim/src/fixture.rs`(벽 · 문 · 지붕 · 선반 · 포장 테이블, 칸마다 나눈 렌더 상자), `crates/world-sim/src/terrain.rs`(`CurbKind`: 턱 · 계단 디딤 · 바닥판 · 경사로), `crates/world-sim/src/nav.rs`(경사로 칸 사이는 경사, 그 밖의 턱 차이는 계단), `crates/world-sim/src/explore.rs`(`next_viewpoint_with`: 칸마다 가중치), `crates/world-sim/src/world.rs`(`WorldConfig::site`, `DropItem::support`·`y_m`, `blocked_cells`), `crates/world-sim/src/snapshot.rs`(`SiteView`), `crates/noto-search/src/parse.rs`(목적지 없는 택배 → 베이스캠프, 출발 장소 → 찾기 힌트), `data/world_lexicon.json`(집 · 현관 · 물류센터 · 선반 · 테이블), `crates/noto-quest/src/lib.rs`(내려다보며 확인 · 선반에서 집기 · 베이스캠프로 복귀), `crates/noto-runtime/src/api.rs`(UI로 보내는 사이트), `crates/world-sim/tests/sites.rs`·`site_bench.rs`(시험과 벤치마크).

표기는 [`actions.md`](actions.md)와 같다: **[직접]** 출처에 인쇄된 값, **[유도: 식]** 이 문서에서 계산, **[추정]** 출처를 찾지 못해 정한 값(코드의 "est"), **[2차: X]** X가 옮겨 적은 값, **[요약]** 검색 요약만 확인. 출처 묶음은 `research/environment-specs.json`(이 저장소 밖의 조사 노트)이다. **벤치마크는 시간 측정이 아니라 기준 점수 시험**이다.

## 0. 요청과 반영 (R10, 1차)

요청: 택배가 들판 아무 데나 있어서 찾기가 실패한다. 택배는 집(현관 앞 계단 · 층계참, 집이 있어야 함)이나 물류센터(선반과 테이블이 있는 창고)에 있고, 베이스캠프로 가져와야 한다. 멀리서는 보이는 바닥의 택배가 가까이 가면 시야에서 빠지므로 고개를 숙여 봐야 하고, 줍기는 게임 이벤트가 아니라 물리여야 한다.

| 요청 | 1차(이 패키지) | 2차로 넘김 |
|---|---|---|
| 집이 있어야 함 | `site::plan`(지형을 만들기 전에 자리 고르기) + `site::build`(지형에 찍기), `World::new`에서 한 번: 64칸 이상 맵에 집 3채(5 × 4 m, 벽 · 문 · 지붕; 놓을 자리가 없으면 줄어듦), 현관 층계참, 실제 단 높이의 계단, 바퀴용 1:12 경사로 | — |
| 현관 계단 | 계단 디딤은 칸마다 턱(cm)이라 기존 계단 규칙이 그대로: 다리는 15–18 cm 단을 오르고, 바퀴(2 cm) · 트랙(12 cm)은 못 오르고 경사로로 감 | 계단 낙상 · 발 디딤 위치(R4) |
| 선반 · 테이블 있는 물류센터 | 창고 깃발을 둘러싼 9 × 7 m 건물: 2.5 m 문, 입고 · 포장 구역의 포장 테이블 2개, 선반 줄(볼트리스 1829 mm, 3단)과 1.0 m 피킹 통로 | 선반 · 테이블에서 집는 손 뻗기 |
| 택배는 집 · 물류센터에 | `WorldConfig::site`가 있으면 베이스캠프 앞 택배 1개는 그대로, 나머지는 현관 층계참 · 선반 단 · 테이블 위의 자리에. 자리는 집과 물류센터를 고르게 돌아가며 뽑음 | — |
| 베이스캠프로 가져옴 | "택배 가져와"(목적지 없음)는 베이스캠프로(놓인 물건만, 사전의 `home`). "집 앞에서", "창고 선반에서", "선반에 있는"은 찾기 힌트(`Target::origin`) | 런타임이 힌트로 먼저 갈 곳 고르기 |
| 찾기 실패 | 탐색 시점 고르기에 칸마다 사전 가중치(`next_viewpoint_with`): 로봇이 들은 장소(현관 둘레, 물류센터)를 먼저. 택배 위치는 쓰지 않음 | 런타임 탐색기에 연결 |
| 고개 숙여 보기 · 물리 줍기 | 단계 이름만("택배 내려다보며 확인"). 받침 높이(`DropItem::y_m`, `World::support_top_m`)를 렌더 · 스냅샷이 씀 | 시선 상태, 손에 든 택배의 자세, 미끄러지면 떨어짐 |
| 지도에 집 · 물류센터 | 스냅샷의 `site`(집 · 계단 · 경사로 · 고정물 · 구역 조도)와 지형의 `curb_kind`. 미니맵(`ui/js/world-map.js`)이 벽 · 문 · 선반 · 테이블과 건물 이름을 그리고, 현관 · 계단 · 경사로를 노란 턱 대신 바닥색으로, 받침 위 택배에 테두리를 그림 | 3차원 화면의 사이트 표시, 모의 백엔드 |

`site`가 없으면(기본값) 맵 · 노드 · 택배 · 경로 · 렌더 · 탐색 시점이 이전과 바이트까지 같다(시드 1/4/7/11 × 열린 · 험한 지형 × 바퀴 · 다리 · 트랙 24개 월드의 지문 비교: 지형 · 물체 · 경로 · 렌더(1인칭 + 위에서 본 그림) · 150틱 주행 · 다음 탐색 시점, 그리고 기존 시험 전부). 설정 파일에서는 `"site": {}`가 표준 사이트(집 3채 · 물류센터 · 경사로)이고, 적은 필드만 바뀐다.

## 1. 엔진 입력 (그룹 A)

| 값 | 쓰는 값 | 출처 |
|---|---|---|
| 계단 단 높이 최대 | 180 mm: 아래 셋 중 가장 엄격 | 주택건설기준 등에 관한 규정 제16조 표: 공동 계단 단 높이 ≤ 180, 디딤 ≥ 260, 폭 ≥ 1200; 건축물 옥외계단 ≤ 200, ≥ 240, ≥ 900 mm **[2차: gunnet.kr 미러, 법제처 해석 18-0465와 일치]** https://www.moleg.go.kr/lawinfo/nwLwAnInfo.mo?mid=a10106020000&cs_seq=405214 · ICC IRC 2021 R311.7.5.1: ≤ 196 mm (7¾ in) **[2차: Nampa 시 안내]** https://www.cityofnampa.us/DocumentCenter/View/20767/Residential-Stair-Treads-and-Risers-2026 |
| 계단 단 높이 최소 | 150 mm **[추정]** | 두 법규 모두 최소값 없음 |
| 디딤 깊이 | 500 mm = 한 칸 (0.5 m 격자가 정함) | IRC R311.7.5.2 ≥ 254 mm, 제16조 ≥ 260 mm보다 깊음 |
| 단 수 | 2–3 **[추정]** | 한국 · 미국 단독주택 진입 계단 단 수의 출처를 찾지 못함 |
| 현관 층계참 | 2.0 × 1.0 m (문 폭 이상, 깊이 914 mm 이상), 바닥과 같은 높이 | IRC R311.3: 층계참 깊이 ≥ 914 mm (36 in), 폭 ≥ 문 폭, 문턱 아래 ≤ 38 mm **[2차: Tucson 허가 기록]** https://pro.tucsonaz.gov/casedetails/T08DV11527/ |
| 계단참 간격 | 높이 0.30–0.54 m라 중간 계단참 없음 | 제16조(2): 2.0 m 오를 때마다 ≥ 1.20 m 계단참 **[2차: nepla.ai]** |
| 계단 폭 | 1.0 m (2칸) | 제16조 옥외계단 ≥ 900 mm (공동 계단 1200 mm에는 못 미침, §6) |
| 현관문 | 914 × 2032 mm 문짝, 문틀이 914 mm 열림을 남김 | IRC R311.2: 유효 폭 ≥ 813 mm, 높이 ≥ 1981 mm; 32 in 유효 폭에 36 × 80 in 문짝이 필요한 것은 업계 관행 (AIBD) https://aibd.org/egress-doors/ |
| 택배를 두는 곳 | 현관 층계참, 문 옆 벽 쪽과 앞 가장자리 | Ring 집주인 설문: 73%가 배달원이 현관(porch)에 둔다고 답함 (2019) **[요약: Global News]** |
| 경사로 기울기 | 1:12 (칸당 4.2 cm, 4.8°) **[추정]** | 흔한 접근성 경사로 값; 조사 묶음에 없음 |
| 볼트리스 선반 | 폭 · 높이 1829 mm (72 in), 기둥 38 mm, 데크 16 mm + 보 38 mm, 높이 조절 38.1 mm 간격, 3단 | Uline H-8409 https://uline.ca/Product/Detail/H-8409/Industrial-Steel-Shelving/Heavy-Duty-Boltless-Shelving-72-x-24-x-72 **[직접]** |
| 선반 깊이 | 457 mm (18 in): 한 칸에 들어가는 카탈로그 깊이 | 같은 제품군의 깊이 457/610/914 mm **[직접]** |
| 선반 단 높이 | 데크 윗면 152 / 914 / 1676 mm **[추정: 38.1 mm 간격의 4·24·44 칸, 762 mm 등간격]** | — |
| 택배를 두는 선반 단 | 1.162 m 이하 (아래 두 단) | Lavender 외 2021 (Applied Ergonomics 97): 선반에서 가장 빨리 드는 높이 63.5–116.2 cm, 시험 높이 10.8–168.9 cm, n = 17 **[요약]** https://par.nsf.gov/biblio/10300979 |
| 포장 테이블 높이 | 900 mm (서서 하는 가벼운 일 864–940 mm 안) | J.J. Keller Compliance Network (NIOSH 인용): 가벼운 일 864–940, 무거운 일 711–889, 정밀 940–1092 mm https://jjkellercompliancenetwork.com/regsense/at-what-height-should-machinery-and-equipment-be-set |
| 피킹 통로 | 1.0 m (2칸) | Cisco-Eagle 통로 폭 안내: 보행 · 피킹 통로 ≥ 1.0 m, 1인 리치 트럭 ≥ 2.44 m https://www.cisco-eagle.com/blog/2013/05/31/key-considerations-for-warehouse-aisle-widths/ |
| 물류센터 문 | 2.5 × 3.0 m **[추정: 리치 트럭 통로 2.44 m보다 넓게]** | 위 안내 |
| 작업면 조도 (설계값) | 선반 통로 바닥 150 lx, 입고 · 포장 300 lx | EN 12464-1:2021 물류 표 (Glamox 안내서): 랙 바닥 150 lx (U0 0.5), 랙 면 75 lx, 포장 300 lx https://www.glamox.com/en/pbs/application-guide/industry/logistics-and-warehouses/ |
| 집 크기 · 벽 · 지붕 | 5 × 4 m 단층, 벽 높이 2.4 m, 두께 0.2 m, 지붕 0.15 m **[추정]** | 32 m 맵에 2–3채가 들어가는 크기 |
| 물류센터 크기 · 벽 | 9 × 7 m, 벽 3.6 m **[추정: 작은 단동 창고]** | — |
| 찾기 사전 가중치 | 현관 둘레 4칸 안 ×4, 물류센터 안 ×3, 그 밖 ×1 **[추정]** | 로봇이 "들은" 장소의 무게. 택배 위치는 쓰지 않음 |

조사 묶음에 있지만 1차에서 쓰지 않은 것: 우체국 택배 상자 규격(0·1·3·5호)과 판매 비율, 택배 무게 분포(2 kg 이하 약 70%, CNMC 2022), 골판지 마찰계수(추정만, 기울기 시험으로 보정할 것), 팔레트 랙 단 간격(RMH), 사람 눈높이 1.58 m와 바닥 물건 응시 약 2 m 전(Voudouris & Brenner 2025), 머리 피치가 시선 낮춤에 차지하는 몫 17–54%(Thomas 외 2020). 뒤의 둘은 2차의 고개 숙이기에 쓴다.

## 2. 모델

### 2.1 계단과 경사로 (`site::build`, `nav::edge_with`)

지형은 칸마다 높이 하나인 2.5D 높이장이라 계단을 높이 차로 만들면 경사(17 cm / 0.5 m = 18.8°)가 된다. 그래서 현관 · 집 바닥 · 계단 디딤은 칸 위의 **턱**(curb, cm)이고, 칸마다 `CurbKind`가 무엇인지 적는다: 0 턱(기존 턱 벽), 1 계단 디딤, 2 바닥판(현관 · 집 바닥), 3 경사로.

- 층계참 높이 H = n · R (n = 2–3 단, R = 150–180 mm에서 10 mm 단위로 뽑음). 디딤 k(층계참 바로 아래가 1)는 (n − k)·R 높이. 다리는 기존 계단 규칙(`BaseSpec::max_step_m`: 사람 발 0.18, 트랙 0.12, 바퀴 0.02, 메카넘 0.015 m)으로 한 단씩 오르고, 바퀴 · 트랙은 단에서 막힌다.
- 경사로: L = ⌈H / (50 cm / 12)⌉ − 1 칸, 칸 j(층계참 옆이 1)의 턱 = H·(L + 1 − j)/(L + 1) **[유도]**. 칸마다 오름 ≤ 4.17 cm. 경사로 칸이 끼고 턱 차가 6 cm 이하인 두 칸 사이는 경사(`Terrain::ramp_rise_m`), 그 밖의 턱 차는 계단이다. 경로 계획 · 걷기 자세(`posture_now`)가 같은 오름을 쓴다.
- 경사로 옆면은 턱이라, 낮은 끝은 트랙이 옆으로 올라설 수 있다(12 cm 이하). 실제 경사로의 옆 막이는 없다.

### 2.2 배치 (`site::plan`)

1. 물류센터: 창고 깃발이 문 안쪽 4칸에 오도록, 문은 베이스캠프 쪽(−x) 벽. 깃발 칸과 그 −x 칸(쌓는 더미)은 비워 둔다.
2. 집: 칸마다 "놓아도 되는가"(맵 가장자리 한 줄, 턱 벽 ±2칸, 베이스캠프 4.5 m, 두 깃발 사이 바퀴 길 2.2 m + 0.36 m)를 미리 표로 만들고 합 표(summed-area)로 직사각형을 상수 시간에 본다. 방향 4 × 좌우 뒤집기 2 × 모든 원점을 다 시험해, 여유 고리 바깥에 막힌 칸이 가장 많은(가장 빽빽하게 붙는, contact-point 규칙) 후보 가운데 사이트 전용 난수로 하나를 고른다. 무작위 시도(이전 구현)는 시드 2에서 한 채만 놓았고, 이 규칙은 64칸 맵 16 시드 × 2 지형에서 늘 2–3채를 놓는다.
3. 건물 자리는 지형을 만들기 전에 정해, 지형 생성의 평지 원(flat zone)으로 덮는다: 건물 밑이 평평하고 둘레 언덕과 부드럽게 이어진다(절벽 없음). 험한 지형의 바다 · 절벽 · 능선도 이 원을 비켜 간다.
4. 노드(나무 · 바위 · 광석)는 사이트 직사각형에 두지 않는다. 걷는 로봇이 베이스캠프에서 못 가는 현관(험한 지형의 물 너머)은 택배 자리를 잃는다.
5. 택배: 베이스캠프 앞 1개는 그대로, 나머지는 자리에서 뽑는다(`site::choose_slots`): 먼저 자리가 남은 곳(집마다, 물류센터) 가운데 하나를 고르게, 그다음 그곳의 빈 자리 하나를 고르게. 물류센터의 많은 선반 자리가 집을 덮지 않고, 배달 대부분이 집에 놓인다. 자리가 모자라면(집도 물류센터도 들어가지 않은 맵) 남은 택배는 이전처럼 들판의 바퀴가 갈 수 있는 칸에 놓인다. 뽑기는 택배의 난수열을 쓰고 사이트 배치는 자기 난수열(`SITE_SALT`)을 써서, 사이트가 없을 때의 뽑기는 그대로다.
6. 받침 높이: 현관 층계참 = 바닥 높이 n·R, 계단 디딤 `level` k = n·R − k·R, 선반 데크 = 바닥 + 데크 윗면, 테이블 = 바닥 + 윗면(`Site::support_top_m`). 택배는 그 높이를 `DropItem::y_m`으로 갖고, 렌더 · 스냅샷 · 2차의 놓기가 같은 값을 쓴다.

### 2.3 고정물과 렌더 (`fixture.rs`)

고정물은 지형이 아니라 물체다: 벽 · 선반 · 테이블은 서 있는 칸을 막고(`World::blocked_cells`), 선반 데크와 테이블 윗면은 그 높이에 물건을 얹는다(`Support::Shelf{fixture, level}`, `Support::Table{fixture}`, 현관은 `Support::Step{house, level}`). 렌더러의 불변식(상자는 자기 칸 안)을 지키려고 여러 칸짜리 벽 · 선반 · 테이블은 칸마다 상자로 나눈다. 바닥판 · 계단 · 경사로의 덮개 상자는 칸보다 1 mm 넓고 1 mm 높아 그 아래 노란 턱 대신 이것이 맞는다(옆 칸에서는 그 1 mm를 시험하지 않으므로 칸 순서의 가장 가까운 맞음은 그대로). 객체 종류는 `Structure`(벽 · 문 · 지붕 · 바닥 · 계단 · 경사로)와 `Fixture`(선반 · 테이블)로 정답 버퍼에만 있고, `visible_objects`(인식 대상)에는 나오지 않는다. 가장 높은 상자가 위로 가는 광선을 끊는 높이(`top_m`)를 올린다(사이트가 없으면 이전과 같음).

### 2.4 찾기 사전 지식 (`explore::next_viewpoint_with`)

다음 시점의 점수 = (보지 못한 칸의 가중치 합) / (가는 시간 + 둘러보기 12 s). 가중치는 `Site::prior`: 걸어서나 바퀴로 갈 수 있는 현관 · 계단 둘레 4칸(집 안 제외) ×4, 물류센터 안 ×3, 나머지 1. 가중치 없이 부르면 이전 `next_viewpoint`와 같다(같은 섹터 고르기 포함). 사전 지식은 장소에서만 나오고 택배를 옮겨도 바뀌지 않는다(시험).

### 2.5 명령 (`parse.rs`, `world_lexicon.json`)

- 사전에 장소 `house`(집 · 주택), `doorstep`(현관 · 집 앞 · 문 앞), `shelf`(선반 · 선반 위 · 랙), `table`(테이블 · 포장대)을 더하고, `물류센터`는 `warehouse`의 같은 말이다(창고 깃발이 물류센터 안에 있으므로). `parcel`에 `"placed": true`, `base_camp`에 `"home": true`.
- 목적지 없는 운반: 대상이 모두 놓인 물건이면 `home`(베이스캠프)으로, `PlaceRef::implied()`(입력 글자 없음)와 경고를 남긴다. 캐는 물건이 섞이면 이전처럼 `missing_destination`.
- 출발 장소: "에서" 꼬리, "from", 또는 "X에 있는/놓인"(사전의 `preposition.located`)이면 출발. 출발 바로 앞의 표지 없는 장소("창고 선반에서")도 출발에 붙는다. 출발은 대상마다 `origin`(바깥에서 안쪽 순)으로 남는 찾기 힌트이고 위치가 아니다.
- 비전 사전의 `structure` 종류(집 · 계단 · 선반 · 테이블)는 `Lexicon::structures`에 따로 둔다: 인식기의 레이블 집합(`vision`)과 가짜 검출의 뽑기 순서가 그대로다.
- 퀘스트 단계: 놓인 물건의 셋째 단계는 "내려다보며 확인", 출발 힌트가 있으면 "선반에서 택배 집기 1개", 목적지가 암시였으면 "베이스캠프로 복귀".

## 3. 벤치마크 (그룹 B) — 채점만 쓰는 기준

엔진은 이 출처를 쓰지 않는다(`the_site_engine_never_names_a_site_benchmark_source`).

| 칸 | 기준 | 규칙 |
|---|---|---|
| S1 `riser_max_mm` · `going_min_mm` · `two_r_plus_g_mm` · `pitch_max_deg` | 영국 Approved Document K 사설 계단: 단 ≤ 220, 디딤 220–300, 2R+G 550–700 mm, 기울기 ≤ 42° (BWF 2011 재수록) https://www.bwf.org.uk/question-of-the-week/what-are-the-current-recommendations-for-the-rise-and-going-of-stairs/ | 이하 · 범위 |
| S1 `riser_150_180_share` | 호주 시공자 설문: 79%가 주택 단 높이 150–180 mm (Di Mazio 2022, ABCB) **[요약]** | 0.79 이상 |
| S1 `riser_max_school_mm` · `tread_min_school_mm` | 건축물의 피난·방화구조 등의 기준에 관한 규칙 제15조(2025): 중·고교 단 ≤ 180, 디딤 ≥ 260 mm | 이하 · 이상 |
| S2 `clearance_above_parcel_mm` | 다음 단까지 여유 약 100 mm (Hoffmann 랙 목록 991202) | 100 이상 |
| S2 `deck_pitch_mm` | 72 in 볼트리스 5단 균등이면 단 사이 약 330 mm (Tennsco Z-Line 판매처) — 3단이라 비교 안 됨 | 기록 |
| S2 `table_top_mm` | 가벼운 서서 하는 일: 손이 팔꿈치보다 약 5 cm 아래 → 95–105 cm (UC Davis Safety Services) **[요약]** | 950–1050 |
| S2 `raised_parcel_vm` | 개정 들기 식 수직 계수 VM = 1 − 0.003·|V − 75| (Waters 외 1993, Ergonomics 36:749) | 기록 |
| S2 `picking_aisle_m` | 지게차 통로 VNA 1.52–2.13 m (Toyota 통로 폭 안내) — 보행 · 로봇 피킹 통로는 기준 밖 | 기록 |
| S3 `rack_floor_lx` · `packing_lx` | 산업안전보건기준에 관한 규칙 제8조: 보통 작업 ≥ 150 lx, 그 밖 ≥ 75 lx | 150 이상 |
| S3 `rack_floor_lx_us` | 29 CFR 1926.56 표 D-3: 창고 5 fc (54 lx), 작업 중인 저장실 10 fc (108 lx) | 108 이상 |
| S4 | 같은 탐색기로 열린 지형 · 사이트 사전 지식 없이 · 있이: 첫 택배 셀을 덮는 시간, 240 s 안에 덮은 몫 (A/B, 기준 출처 없음) | 기록 |

## 4. 벤치마크 결과

명령: `cargo test -p world-sim --test site_bench -- --nocapture`, `cargo test -p world-sim --test sites -- --nocapture`.

### 4.1 [SX] 결과 (열린 지형 64칸, 시드 1–8의 모든 집 · 물류센터)

| 칸 | 결과 | 기준 | 점수 |
|---|---|---|---|
| S1 `riser_max_mm` | 180 | ≤ 220 (ADK) | 100 |
| S1 `going_min_mm` | 500 | 220–300 (ADK) | 0 |
| S1 `two_r_plus_g_mm` | 826 (평균; 800–860) | 550–700 (ADK) | 16 |
| S1 `pitch_max_deg` | 19.8° | ≤ 42° (ADK) | 100 |
| S1 `riser_150_180_share` | 1.00 | ≥ 0.79 (호주 설문) | 100 |
| S1 `riser_max_school_mm` · `tread_min_school_mm` | 180 · 500 | ≤ 180 · ≥ 260 (제15조) | 100 · 100 |
| **S1 현관 계단** | | | **73.7** |
| S2 `clearance_above_parcel_mm` | 608 (데크 간격 762 − 데크 54 − 택배 100) | ≥ 100 (Hoffmann) | 100 |
| S2 `deck_pitch_mm` | 762 | 5단이면 약 330 (Tennsco) — 3단이라 기록만 | 기록 |
| S2 `table_top_mm` | 900 | 950–1050 (UC Davis) | 50 |
| S2 `raised_parcel_vm` · `raised_parcel_height_m` | 0.908 · 0.50 m | 들기 식 VM (Waters 1993) | 기록 |
| S2 `picking_aisle_m` | 1.0 | 지게차 VNA 1.52–2.13 — 보행 통로는 기준 밖 | 기록 |
| **S2 물류센터** | | | **75.0** |
| S3 `rack_floor_lx` · `packing_lx` · `rack_floor_lx_us` | 150 · 300 · 150 | ≥ 150 · ≥ 150 (산안규칙 제8조) · ≥ 108 (29 CFR 1926.56) | 100 · 100 · 100 |
| **S3 조도 (설계값)** | | | **100** |

S4 찾기 A/B (같은 탐색기 · 3 m 인식 거리 · 시점마다 한 바퀴, 다리형, 8 시드, 베이스캠프 택배 제외):

| 월드 | 첫 택배 칸을 덮는 시간 | 240 s 안에 덮은 택배 몫 |
|---|---|---|
| 열린 지형(사이트 없음) | 150.6 s | 0.325 |
| 사이트, 사전 지식 없이 | 217.9 s | 0.225 |
| 사이트, 사전 지식(`Site::prior`) | **38.2 s** | **0.850** |

사이트의 택배는 집 · 물류센터에 모여 있어 베이스캠프에서 멀고, 사전 지식 없는 균일 탐색은 들판보다 늦게 찾는다(218 s, 22.5%). 장소 가중치를 쓰면 첫 택배가 38 s, 240 s 안에 85%로 들판보다 빨라진다. 이것은 계획기 수준(벽이 시야를 가리지 않음)의 비교이고 런타임 퀘스트는 아직 이 가중치를 쓰지 않는다(2차).

택배 자리 분포(`sites.rs`, 2 지형 × 8 시드, 80개): 현관 58 · 선반 16 · 테이블 6. 맵 크기별(`the_site_never_cuts_the_way_between_the_waypoints_on_any_map_size`): 64 · 96칸은 시드 1 · 4 · 7 × 열린 · 험한 지형 모두 집 3채 + 물류센터(택배 자리 120; 96칸 험한 지형 시드 7은 한 집이 물 너머라 2채만 걸어서 · 바퀴로 닿고 자리 116), 32칸(16 m)은 물류센터(11 × 14칸으로 줄어듦, 자리 12)만 들어가고 집은 들어가지 않는다. 어느 크기 · 지형에서도 두 깃발 칸은 비어 있고, 사이트 없이 바퀴가 창고 깃발까지 가는 길이 있으면 사이트가 있어도 있다.

### 4.2 기존 벤치마크

사이트가 기본으로 꺼져 있어 기존 줄은 기준선(`baseline-bench.txt`)과 모두 같다: [AX-SUM] · [FX] · [FX-SUM] · [NN] · [NN-SUM] · [VX] · [VX-SUM] · [CAM] · [CAM-SUM] 1029줄을 정렬해 비교, 차이 0줄(`cargo test -p noto-core -- --nocapture`, `cargo test -p noto-runtime -- --nocapture`).

## 5. 시험

| 시험 | 무엇을 |
|---|---|
| `sites.rs` `the_same_seed_builds_the_same_site_and_another_seed_another` | 같은 시드 = 같은 지형 · 고정물 · 사이트 · 택배; 로봇 종류는 배치를 바꾸지 않음 |
| `without_a_site_the_map_has_no_structures` | 기본 설정은 사이트 없음, JSON에 `site` 키 없음, `"site": {}`는 표준 사이트 |
| `every_map_gets_houses_and_a_logistics_centre_around_the_warehouse_waypoint` | 8 시드 × 2 지형: 집 ≥ 2, 깃발 · 더미 칸 비어 있음, 노드는 사이트 밖 |
| `the_site_never_cuts_the_way_between_the_waypoints_on_any_map_size` | 32 · 64 · 96칸 × 2 지형 × 3 시드 × 바퀴 · 다리: 두 깃발 칸에 고정물 없음, 사이트 없이 깃발까지 길이 있으면 사이트가 있어도 있음, 다리가 못 가는 현관에는 택배 자리 없음 |
| `stairs_have_real_risers_legs_climb_them_and_wheels_take_the_ramp` | 디딤 높이 = 층계참 − k·R, 다리는 경사로를 막아도 단마다 한 번씩 오름, 바퀴 · 트랙은 계단을 못 오르고 경사로로(경사로 4.8° < 7°) |
| `without_ramps_only_legs_reach_the_porches` | 경사로를 끄면 바퀴는 현관에 못 감 |
| `doors_let_robots_in_and_routes_never_cross_walls_shelves_or_tables` | 집에는 문으로 들어감, 바퀴가 넓은 문으로 깃발까지, 모든 택배 자리의 설 자리까지 경로가 고정물 칸이나 그 모서리를 지나지 않음 |
| `parcels_rest_on_porches_shelves_and_tables_and_the_camp_parcel_stays` | 택배 수 같음, 베이스캠프 앞 택배 그대로, 나머지는 받침 위(높이 = 받침 = `support_top_m`, 스냅샷 y 같음), 계단 디딤마다 받침 높이 = 그 칸의 높이 |
| `every_parcel_slot_shows_from_where_the_robot_stands` | 모든 자리의 택배가 설 자리에서 20 px 이상 보임(정답 버퍼) |
| `the_site_renders_as_structures_and_furniture_never_as_detections` | 집 · 선반이 Structure · Fixture로 그려지고 노란 턱이 아니며 인식 대상에 없음 |
| `a_raised_parcel_is_picked_only_from_its_own_level` | 현관 택배는 층계참에 올라서야, 선반 택배는 아직 거절(`CannotReach`), 밀기도 거절 |
| `the_search_prior_points_at_porches_and_the_logistics_centre_not_at_parcels` | 가중치가 장소에만, 택배를 옮겨도 그대로, 사전 지식이 다음 시점을 장소 쪽으로 |
| `fixtures_have_the_dimensions_of_their_sources` | 선반 1829 · 457 mm와 단 높이, 테이블 864–940, 문 ≥ 813 × 1981, 층계참 ≥ 914 mm |
| `site.rs`, `fixture.rs` 단위 시험 | 배치 결정성 · 겹침 없음 · 평지 원이 덮음, 단 규칙과 경사로 기울기, 상자가 칸 안, 벽 모서리 이음, 문 914 mm 열림 |
| `noto-search` `a_parcel_without_a_destination_goes_back_to_the_base_camp`, `where_a_parcel_lies_becomes_a_search_hint`, `the_new_place_words_leave_the_verbs_alone` | "택배 가져와" → 베이스캠프(암시), 출발 힌트, "집어"는 여전히 동사 |
| `noto-quest` `a_parcel_is_looked_down_at_picked_where_it_lies_and_brought_back_to_the_base` | 단계 이름 |
| `noto-runtime` api `a_site_goes_out_through_the_same_flip_and_a_bare_world_sends_none`, `a_parcel_fetch_says_where_to_look_and_that_the_base_camp_was_implied` | UI JSON |

## 6. 남은 차이

- **디딤 500 mm**: 0.5 m 격자 때문에 디딤이 한 칸이라 영국 기준의 디딤 220–300 mm와 2R+G 550–700 mm를 벗어난다(S1). 기울기는 17–20°로 완만해 오르기 쉬운 쪽이다. 계단을 칸보다 잘게 만들려면 칸 안의 높이 단계가 필요하다.
- **계단 폭 1.0 m**: 제16조 옥외계단(≥ 0.9 m)은 맞지만 공동 계단(≥ 1.2 m)에는 못 미친다. 단독주택 진입 계단의 폭 출처는 찾지 못했다.
- **테이블 높이**: 엔진 입력 띠(864–940 mm)와 벤치마크(95–105 cm)가 서로 다르다. 엔진 값을 벤치마크 쪽으로 옮기지 않았다(S2 결과 그대로).
- **택배 줍기**: 현관 택배는 층계참에 서서만 집을 수 있고(손 뻗기는 바닥 모델), 선반 · 테이블 택배는 아직 거절한다. 높은 받침으로 뻗기, 손에 든 택배의 자세, 미끄러짐 낙하는 2차.
- **고개 숙이기**: 단계 이름만 바뀌었다. 지속되는 시선 상태(목 숙임)는 2차.
- **사이트 장소로 가기 · 놓기**: "집으로 가", "선반에 갖다 놔"는 그 장소(`house`, `shelf`)로 해석되지만 런타임에는 그 장소의 웨이포인트가 없어 `unknown_target`("지도에 'house' 지점이 없습니다")으로 거절된다. 집이 여러 채라 어느 집인지도 정해야 한다(2차의 놓기 · 이동).
- **출발 힌트**: 명령의 출발 장소(`Target::origin`)는 계획 · 퀘스트 단계 이름 · UI에만 실리고, 런타임 탐색은 아직 쓰지 않는다.
- **벽이 시야를 가림**: 렌더러는 벽에 가리지만 탐색 계획기의 시야(`viewshed`)는 지형만 본다. S4의 덮음은 계획기 수준이다.
- **조도**: 구역 조도는 설계값(EN 12464-1)이고 렌더러에는 아직 조명이 없다(`lightiso` 패키지).
- **경사로 옆면**: 옆 막이가 없어 낮은 끝은 트랙이 옆으로 오른다.
- **UI**: 미니맵만 사이트를 그린다. 3차원 월드 화면(`world-view.js`)과 모의 백엔드(`mock-world.js`)에는 아직 집 · 선반이 없다.
- **지붕과 위에서 보는 화면**: 지붕은 칸마다 상자라 위에서 내려다보는 3인칭 · 아이소메트릭 카메라에는 건물 안이 가려진다(잘라 보기는 `lightiso`의 일). 해 조명을 켜면 지붕 그림자로 실내가 어두워지므로, 구역 조도(설계값)를 실내 조명으로 넣는 일이 남는다.
- **창고 깃발이 건물 안**: 사이트를 켜면 창고 깃발이 벽에 가려 멀리서 보이지 않는다. 문은 베이스캠프 쪽(−x)으로 나 있다. 기본값을 켜는 2차에서 "창고 찾기"의 인식 거리를 다시 재야 한다.
- **택배 상자**: 우체국 상자 규격 · 무게 분포는 아직 쓰지 않았다(택배는 v0.7의 16 × 12 × 10 cm, 0.7 kg 그대로).
- **작은 맵**: 32칸(16 m) 맵에는 집이 들어가지 않는다(5 × 4 m 집 + 계단 + 1:12 경사로 + 여유 한 칸이 베이스캠프 4.5 m · 턱 벽 ±2칸 · 깃발 사이 길을 피할 자리가 없음). 물류센터만 11 × 14칸으로 줄어 들어가고, 택배는 그 선반 · 테이블에 놓인다. 험한 지형의 큰 맵(96칸)은 물 너머 집이 생길 수 있고(시드 7), 그 집의 택배 자리는 빠진다.
