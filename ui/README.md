# Noto Sandbox UI

Tauri 2 데스크톱 앱의 프런트엔드입니다. 빌드 단계와 npm 패키지 없이 정적 파일(HTML · CSS · ES 모듈)만으로 동작하며,
디자인 토큰과 레이아웃(떠 있는 카드 4개, 8px 간격 리사이저, 28px 상태줄)은 `endmill-sandbox` 를 따릅니다.

## 구조

```
ui/
├─ index.html            골격: 왼쪽 · 가운데 · 오른쪽 · 하단 카드와 상태줄 (인라인 스크립트·핸들러 없음)
├─ css/
│  ├─ tokens.css         endmill :root 토큰 그대로 + 앱 전용 토큰(--lv-*, --cv-*, --glass)
│  ├─ layout.css         카드·리사이저·접기·탭 줄·상태줄
│  ├─ components.css     아코디언·폼·버튼·뱃지·표·로그·퀘스트 카드
│  └─ hud.css            월드 화면 위 오버레이(신체 HUD·퀘스트 트래커·명령줄·PiP·범례)
└─ js/
   ├─ main.js            시작 순서 조립
   ├─ api.js             invoke 래퍼 (Tauri 가 없으면 모의 백엔드로 자동 전환)
   ├─ state.js util.js toast.js layout.js
   ├─ sim.js world-view.js world-map.js world-panel.js     월드 · 시뮬레이션 · 프레임 폴링 (미니맵의 후진 · 회전 표시)
   ├─ body-hud.js quest-panel.js command-bar.js            오버레이와 오른쪽 패널
   ├─ pick-hud.js        줍기 절차 HUD (world_frame.procedure: 다가가기 → 고개 숙여 응시 → 비전 확인 → 자세 낮춤 → 쥐기 → 들어올림)
   ├─ rig-editor.js      로봇 설계 탭 (SolidCraft iframe · 프리셋 · 왕복 · 투입)
   ├─ depth-view.js      깊이 · 카메라 탭 (정렬 깊이 · 원출력 · 정답 · ADE20K 분할 · 법선)
   ├─ hand-canvas.js hand-panel.js vision-upload.js        손·발 측면도와 파지 시험
   ├─ logs.js analysis.js timeseries.js data-panel.js device.js
   └─ mock-*.js          브라우저 전용 모의 백엔드 (mock-backend.js 가 진입점)
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
모의 값은 UI 개발용이며 실제 시뮬레이션 결과가 아닙니다.

## 실제 Rust 런타임으로 브라우저에서 실행

모의 백엔드 대신 실제 월드 런타임을 보려면 개발용 서버를 씁니다. 서버는 `ui/`를 내려 주고, `index.html`에 `window.__TAURI__.core.invoke` shim을 끼워 넣어 월드 명령 11개를 `POST /invoke/<명령>`으로 런타임에 보냅니다. 나머지 명령(손·발 구성, 파지 시험 등)은 브라우저 모의 백엔드로 갑니다.

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
| `world_frame` | `width, height, quality, view` | WebP 프레임과 탐지 상자, 1인칭이면 `camera`(눈 높이·피치·팬·흔들림·블러·추정 오차) |
| `command_submit` | `text` | 자연어 명령 → 퀘스트 |
| `quest_list` / `body_status` | 없음 | 퀘스트 · 신체 상태 |
| `perception_scan` | `topK` | 비전 점수 |
| `sds_status` / `ts_forecast` | `scope, axis` | SDS 원장 · 시계열 예측 (`axis`는 score · score_all · margin) |
| `log_tail` | `afterSeq, limit, category` | 증분 로그 (시각은 UTC ISO 8601) |
| `models_status` | 없음 | 인식기·TTM 연결 상태, 모델 폴더와 파일 유무 |
| `recognizer_use` | `mode` (`embedding` \| `oracle`) | SigLIP2 인식 켜기·끄기 (Tauri 전용, UI에는 스위치 없음) |
| `ttm_use` | `enabled` | Granite TTM 켜기·끄기 (Tauri 전용, UI에는 스위치 없음) |
| `models_catalog` | 없음 | 필수 모델 네 개의 파일·받은 바이트, 받기 진행률(`download`), 불러오기 상태(`ready`·`loading`·`load_error`) |
| `models_download` | `ids` (없으면 모두) | 빠진 파일 받기 시작(바로 돌아옴, 진행은 `models_catalog` 폴링) |
| `models_cancel` / `models_delete` | — / `id` | 받기 취소(`.part` 남김) · 모델 파일 삭제(월드가 멈춤) |
| `models_activate` | 없음 | 네 모델을 선택한 장치에 올리고 월드에 연결. 이 전에는 `world_new`·`sim_step`·`command_submit`·`perception_scan`·`test_start`·`test_step`이 거부됨 |
| `depth_view` | `quality` | 머리 카메라 화면에 깊이 모델을 돌린 결과와 정답 깊이(같은 색 척도), 지표, 카메라 상태 |
| `rig_presets` | 없음 | 기본 로봇 프리셋과 저장한 로봇, 지금 몸 |
| `rig_design` | `id` | 편집기에 넣을 설계(부품 · 재질 · 관절)와 저장한 `.sc` |
| `rig_validate` | `id` | 측정값과 투입 전 검사 |
| `rig_import` | `name, urdf, meshes` | URDF · STL(base64)을 재어 프리셋으로 저장 |
| `rig_keep_design` | `id, design` | 다시 열 수 있게 `.sc`를 함께 저장 |
| `rig_apply` | `id` | 그 몸을 다음 `world_new`의 몸으로 |
| `rig_delete` | `id` | 저장한 로봇 삭제 |

`rig_*` 명령은 모의 백엔드에 없습니다. 탭은 개발용 서버나 Tauri 앱에서 씁니다.

기존 명령(`presets`, `hand_geometry`, `grasp_test`, `drive_courses`, `dictionary`, `devices`, `select_device`,
`classify_image`, `default_paths`, `synthesize`, `load_dictionary`, `dataset_summary`)도 그대로 호출합니다.

Tauri 이벤트는 쓰지 않고 `setInterval` 로 폴링합니다: `world_frame` 약 6fps, `sim_step` 125ms, `log_tail` 1.2초.

## 새 명령 추가하기

1. `js/mock-backend.js` 의 핸들러 표에 모의 구현을 추가합니다.
2. 화면 모듈에서 `call('새_명령', { camelCase 인자 })` 로 부릅니다.
3. 응답 필드는 snake_case 로 읽습니다.

## 제약

- CSP: `script-src 'self'`, 이미지는 `self` · `data:` · `blob:` 만 허용합니다. 원격 글꼴·CDN·인라인 스크립트는 쓰지 않습니다. SolidCraft 때문에 `'wasm-unsafe-eval'`과 `worker-src 'self' blob:`을 더했습니다.
- `ui/vendor/solidcraft`는 받아 쓰는 폴더입니다(`fetch_solidcraft.bat`). 고치지 않습니다.
- 동적 문자열은 `textContent` 와 DOM 생성으로만 넣습니다(`innerHTML` 미사용).
- 파일 하나는 약 600줄 이하로 유지합니다.
