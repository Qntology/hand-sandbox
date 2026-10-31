# Noto Sandbox UI

Tauri 2 데스크톱 앱의 프런트엔드입니다. 빌드 단계와 npm 패키지 없이 정적 파일(HTML · CSS · ES 모듈)만으로 동작합니다.
화면은 100% 풀스크린 월드 하나이고(기본 시점 탑뷰), 나머지는 모두 그 위에 떠 있습니다. 왼쪽·오른쪽·아래에 붙은 패널과 가운데 탭은 없습니다.

## 화면 구성

| 자리 | 요소 | 내용 |
| --- | --- | --- |
| 좌상단 | `#world-pill` → `#world-popover` | 월드(시드·크기·지형·재생성·리셋) + 필수 모델·연산 장치 + 화면 표시·프레임 |
| 좌상단 아래 두 줄 | 소환 줄 | `+🤖 휴머노이드` · `+🚜 지게차` / 트랙(`#crew-mode`) · `모두빼기`. 지도의 왼쪽 여백 안에 들어가게 두 줄로 쌓습니다 |
| 우상단 | `#view-pill` | 시점 알약 하나(`#view-select`, 3인칭이면 궤도 슬라이더가 같은 알약에 붙음) |
| 가운데 | `#world-canvas` + `#cctv-layer` | 탑뷰에서 로봇마다 번호표와 CCTV 말풍선(1인칭 · 깊이, 포크 절차 중에는 포크 카메라 · 절차 · 교차 검증) |
| 오른쪽 | `#unit-hud` | 로봇을 고른 동안만: 신체 상태 + `📈 관절 토크 시계열` / `🖐 부위 접촉력` |
| 하단 중앙 | `#hub` | `#log-feed`(3줄) · `#quest-bar` · `#command-bar` 일체형 |
| 프롬프트 위 | `#quest-dropup` · `#preset-dropup` | 협업·퀘스트·인벤토리·보관 더미 / 유닛 프리셋 (`bottom: 88px`) |
| 모달 | `#analysis-modal` · `#cctv-zoom` · `#model-gate` | 기록·분석 4탭 / 깊이 비교 / 첫 실행 필수 모델 |

- **유닛 선택**: 로봇(또는 번호표·말풍선)을 누르면 고릅니다. 빈 땅을 누르거나 ESC 면 바로 풉니다. 고른 동안 명령줄은 `[🛠 유닛#N ✕] [⚙️ 프리셋 ▲]` + AI 3D 외형 수정 프롬프트로 바뀝니다. 탑뷰가 아닌 시점은 프레임에 로봇 좌표가 없어, 화면을 누르면 카메라가 따라가는 최초 작업자를 고르거나 풉니다.
- **ESC 순서**: 입력 칸에 글이 있으면 지우기만 하고, 없으면 맨 위 판부터 하나씩 닫고(확대 → 분석 → 월드 → Drop-up), 닫을 판이 없으면 선택을 풉니다.
- **로그 피드**: 평소 3줄, 마우스를 올리면 같은 자리에서 위로 펼쳐져 전체 로그와 범주 칩이 보이고, 나가면 1.8초 뒤 줄어듭니다. 줄을 누르면 범주에 맞는 탭이 열립니다: quest·world → 통합 테스트, sds·ts → 시계열·SDS, vision → 비전·깊이, search·embed → 명령 해석·합성 DB, 그 밖(physics · mcp · crew · device …) → 통합 테스트.
- **유닛 적용 범위**: 체형·발·카메라는 `✅ 유닛에 적용`으로 그 유닛에 제자리에서 적용되고 월드는 리셋되지 않습니다. 손 구성과 부품 등급은 유닛별 교체가 없어 월드를 재생성해야 적용되며, 그럴 때만 `🔄` 버튼이 보입니다.
- **서브 작업자**: 런타임이 부위별 신체 지표, 관절 동역학, 정답 깊이를 최초 작업자에게만 계산하므로 HUD 와 확대 창은 서브 작업자에게 아는 값(배터리·적재, RGB·추정 깊이)만 보이고 그렇다고 적습니다.
- **미니맵**: 탑뷰는 지도 전체가 화면이므로 다른 시점에서만 왼쪽에 나타납니다. 1인칭 시점의 줍기 절차(`#pick-hud`)도 같은 자리에 뜹니다.

## 구조

```
ui/
├─ index.html            골격: 풀스크린 월드와 뜨는 판들, 상태줄 (인라인 스크립트·핸들러 없음)
├─ css/
│  ├─ tokens.css         :root 토큰 + 앱 전용 토큰(--lv-*, --cv-*, --glass, --hud-w, --hub-w)
│  ├─ layout.css         무대 · 상태줄
│  ├─ components.css     폼·버튼·뱃지·표·로그·퀘스트 카드
│  ├─ hud.css            프레임 · 탐지 상자 · 신체 상태 · 관절 표 · 부위 그림 · 줍기 절차 · 미니맵
│  ├─ overlay.css        코너 · 팝오버 · 유닛 HUD · 허브 · Drop-up · 모달
│  ├─ cctv.css           번호표 · CCTV 말풍선 · 확대 창
│  └─ timeline.css       통합 테스트 타임라인
└─ js/
   ├─ main.js            시작 순서 조립
   ├─ api.js             invoke 래퍼 (Tauri 가 없으면 모의 백엔드로 자동 전환)
   ├─ state.js util.js toast.js layout.js
   ├─ overlays.js        떠 있는 판 여닫기 · 바깥 클릭 · ESC 순서
   ├─ sim.js world-view.js world-map.js world-panel.js     월드 · 시뮬레이션 · 프레임 폴링 · 월드 팝오버 (미니맵의 후진 · 회전 표시)
   ├─ unit-select.js     유닛 고르기 · 풀기, 명령줄 전환, units_list
   ├─ unit-hud.js body-hud.js joint-view.js focus-view.js  우측 유닛 HUD
   ├─ unit-preset.js presets.js hand-panel.js hand-canvas.js  ⚙️ 프리셋 Drop-up (체형·발·카메라·부품 등급·손, 파지 시험, 주행 비교)
   ├─ cctv.js cctv-zoom.js  로봇 위 CCTV 말풍선과 확대 비교 창
   ├─ pick-hud.js        줍기·포크 절차 (world_frame · unit_frame 의 procedure: 다가가기 → 고개 숙여 응시 → 비전 확인 → 자세 낮춤 → 쥐기 → 들어올림)
   ├─ rig-mcp.js         AI 3D 외형 수정 흐름 (rig_prompt → 헤드리스 CAD → rig_commit)
   ├─ headless-cad.js    화면 밖 iframe 의 SolidCraft 로 설계를 조립해 URDF · 메시 · .sc 를 내보냄 (UI 없음)
   ├─ command-bar.js quest-panel.js crew-panel.js          명령줄 · 퀘스트 헤더와 Drop-up · 소환 · 협업 흐름
   ├─ logs.js analysis-modal.js analysis.js timeseries.js test-timeline.js depth-view.js vision-upload.js data-panel.js  로그 피드와 기록·분석 모달
   ├─ device.js models.js  연산 장치 · 필수 모델
   └─ mock-*.js          브라우저 전용 모의 백엔드 (mock-backend.js 가 진입점, mock-units.js 유닛, mock-rig.js 프리셋·AI 수정)
```

## 브라우저에서 모의 백엔드로 실행

`window.__TAURI__` 가 없으면 `api.js` 가 `mock-backend.js` 를 불러와 모든 명령을 흉내 냅니다.
상태줄에 "모의 백엔드" 뱃지가 표시됩니다.

```sh
cd ui
python3 -m http.server 8765
# 브라우저에서 http://127.0.0.1:8765/ 열기
```

월드 프레임은 가짜 3D(WebP)이고, 퀘스트는 작은 상태기계(탐색 → 접근 → 채취 → 줍기 → 운반 → 인도)로 진행됩니다.
입력줄에 `철광석 3개 캐서 베이스캠프로 가져와` 를 입력하고 Enter 를 누르면 됩니다.
서브 작업자는 서 있기만 하고, 관절·부위·깊이 값은 데모 숫자입니다. 모의 값은 UI 개발용이며 실제 시뮬레이션 결과가 아닙니다.

## 실제 Rust 런타임으로 브라우저에서 실행

모의 백엔드 대신 실제 월드 런타임을 보려면 개발용 서버를 씁니다. 서버는 `ui/`를 내려 주고, `index.html`에 `window.__TAURI__.core.invoke` shim을 끼워 넣어 월드 명령을 `POST /invoke/<명령>`으로 런타임에 보냅니다. 나머지 명령(손·발 구성, 파지 시험 등)은 브라우저 모의 백엔드로 갑니다.

```sh
cargo run --release -p noto-runtime --example serve [포트] [ui 폴더]   # 기본 http://127.0.0.1:8787/
```

127.0.0.1에서만 받고 인증이 없으니 자기 컴퓨터에서 보는 용도로만 쓰세요. 로그와 SDS 원장은 `RUNTIME_DIR`(기본 `data/runtime`)에 저장됩니다.
`models_*` 명령과 `recognizer_use`·`ttm_use`는 Tauri 앱에서만 실제로 동작합니다(서버·모의 백엔드는 모델을 쓰지 않으므로 필수 모델 게이트도 뜨지 않습니다).

## 명령 계약

호출은 모두 `api.js` 의 `call(명령, 인자)` 를 거칩니다. 인자는 camelCase, 응답은 snake_case 입니다.
오류는 문자열로 던져지며 상태줄과 토스트에 표시됩니다.

| 명령 | 인자 | 용도 |
| --- | --- | --- |
| `world_new` | `seed, size, config, base` | 월드 생성(손·발 구성 반영) |
| `sim_step` | `ticks` | 시뮬레이션 전진 |
| `world_frame` | `width, height, quality, view` | WebP 프레임과 탐지 상자, 1인칭이면 `camera`(눈 높이·피치·팬·흔들림·블러·추정 오차), 탑뷰면 `units`(로봇마다 프레임 안 `u, v`, 다른 시점은 빈 목록). 탑뷰는 지도 전체를 요청 크기에 맞춰 그리므로 캔버스 비율로 요청한다 |
| `units_list` | 없음 | 모든 로봇: `id, first, form, base, label_ko, x, z, yaw_deg, status_ko, carried_kg, capacity_kg, battery_ratio, cameras, rig_id, rig_name_ko` |
| `unit_frame` | `unit, width, height, quality, mode` (`rgb` \| `depth` \| `fork`) | 그 로봇의 화면과 `label_ko, depth_name, procedure` (CCTV) |
| `unit_apply` | `unit, preset?, base?, cameras?` | 프리셋·발·카메라를 그 유닛에 제자리 적용 |
| `command_submit` | `text` | 자연어 명령 → 퀘스트 |
| `quest_list` / `body_status` | 없음 | 퀘스트 · 신체 상태 |
| `perception_scan` | `topK` | 비전 점수 |
| `sds_status` / `ts_forecast` | `scope, axis` | SDS 원장 · 시계열 예측 (`axis`는 score · score_all · margin) |
| `log_tail` | `afterSeq, limit, category` | 증분 로그 (시각은 UTC ISO 8601). 범주: world · quest · search · vision · sds · ts · embed · device · physics · mcp · crew |
| `models_status` | 없음 | 인식기·TTM 연결 상태, 모델 폴더와 파일 유무 |
| `recognizer_use` | `mode` (`embedding` \| `oracle`) | SigLIP2 인식 켜기·끄기 (Tauri 전용, UI에는 스위치 없음) |
| `ttm_use` | `enabled` | Granite TTM 켜기·끄기 (Tauri 전용, UI에는 스위치 없음) |
| `models_catalog` | 없음 | 필수 모델 네 개의 파일·받은 바이트, 받기 진행률(`download`), 불러오기 상태(`ready`·`loading`·`load_error`) |
| `models_download` | `ids` (없으면 모두) | 빠진 파일 받기 시작(바로 돌아옴, 진행은 `models_catalog` 폴링) |
| `models_cancel` / `models_delete` | — / `id` | 받기 취소(`.part` 남김) · 모델 파일 삭제(월드가 멈춤) |
| `models_activate` | 없음 | 네 모델을 선택한 장치에 올리고 월드에 연결. 이 전에는 `world_new`·`sim_step`·`command_submit`·`perception_scan`·`test_start`·`test_step`이 거부됨 |
| `depth_view` | `quality` | 머리 카메라 화면에 깊이 모델을 돌린 결과와 정답 깊이(같은 색 척도), 지표, 카메라 상태 |
| `crew_add` / `crew_remove` / `crew_clear` | `count, form, cameras, mode` / `id` / — | 서브 작업자 소환 · 하나 제거 · 모두 빼기 |
| `crew_mode` / `crew_status` | `mode` / — | 트랙 전환 · 작업자와 협업 흐름 |
| `rig_presets` | 없음 | 기본 로봇 프리셋과 저장한 로봇, 지금 몸 |
| `rig_design` | `id` | 설계(부품 · 재질 · 관절)와 저장한 `.sc` |
| `rig_validate` | `id` | 측정값과 투입 전 검사 |
| `rig_prompt` | `unit, text` | 문장을 MCP 도구 호출로 바꿔 고친 설계를 돌려줌(적용 안 함): `ok, summary_ko, calls, name, design, error_ko` |
| `rig_commit` | `unit, name, urdf?, meshes?, sc?` | 검증·프리셋 저장 후 그 유닛을 월드 리셋 없이 교체. `urdf` 가 없으면 런타임이 직접 만든다 |
| `rig_import` | `name, urdf, meshes` | URDF · STL(base64)을 재어 프리셋으로 저장 |
| `rig_keep_design` | `id, design` | 다시 열 수 있게 `.sc`를 함께 저장 |
| `rig_apply` | `id` | 그 몸을 다음 `world_new`의 몸으로 |
| `rig_delete` | `id` | 저장한 로봇 삭제 |

기존 명령(`presets`, `hand_geometry`, `grasp_test`, `drive_courses`, `dictionary`, `devices`, `select_device`,
`classify_image`, `default_paths`, `synthesize`, `load_dictionary`, `dataset_summary`)도 그대로 호출합니다.

Tauri 이벤트는 쓰지 않고 `setInterval` 로 폴링합니다: `world_frame` 약 6fps, `sim_step` 125ms, `log_tail` 1.2초,
`unit_frame` 전체 초당 2장(틱이 바뀐 타일만, 탑뷰에서만, 탭이 숨었거나 모달이 덮으면 멈춤), `units_list` 1초(유닛을 고른 동안),
유닛 HUD 의 작은 `world_frame` 0.5초(고른 동안, 그 탭의 값이 화면 시점에 없을 때만).

## AI 3D 외형 수정

편집기 화면은 어디에도 없습니다. 유닛을 고르고 프롬프트에 문장을 넣으면 `rig-mcp.js` 가 다음을 합니다.

1. `rig_prompt` 로 도구 호출과 고친 설계를 받아 호출을 로그 피드(`mcp`)에 남긴다.
2. `headlessCad(design, id)` 가 `vendor/solidcraft/index.html` 을 화면 밖 iframe(실제 크기, `position:fixed; left:-10000px`, WebGL 이 필요해 `display:none` 금지)으로 불러 조립하고, `file.export` · `file.save_as` 결과를 OPFS 에서 읽는다. 패키지가 없거나 어디서든 실패하면 `null` 이다.
3. `rig_commit` 에 넘긴다(`null` 이면 URDF 없이, 런타임이 직접 만든다). 요약과 검사는 토스트와 로그로 알린다.

## 새 명령 추가하기

1. `js/mock-backend.js` 의 핸들러 표에 모의 구현을 추가합니다.
2. Tauri 의 `rt_call` 로 가는 명령이면 `js/api.js` 의 `RT_CALL` 에 넣습니다.
3. 화면 모듈에서 `call('새_명령', { camelCase 인자 })` 로 부릅니다.
4. 응답 필드는 snake_case 로 읽습니다.

## 제약

- CSP: `script-src 'self'`, 이미지는 `self` · `data:` · `blob:` 만 허용합니다. 원격 글꼴·CDN·인라인 스크립트는 쓰지 않습니다. 헤드리스 SolidCraft 때문에 `'wasm-unsafe-eval'`과 `worker-src 'self' blob:`을 더했습니다.
- `ui/vendor/solidcraft`는 받아 쓰는 폴더입니다(`fetch_solidcraft.bat`). 고치지 않습니다.
- 동적 문자열은 `textContent` 와 DOM 생성으로만 넣습니다(`innerHTML` 미사용).
- 파일 하나는 약 600줄 이하로 유지합니다.
- 화면은 1280×720 이상에서 동작합니다.
