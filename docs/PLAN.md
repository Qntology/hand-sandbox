# hand-sandbox v0.2 기획서 — 오픈월드 자원 채취 테스트베드

작성일 2026-10-23. 개발 전에 현재 프로젝트와 참조 프로젝트(logis-center, endmill-sandbox, Horizon)를 층별로 대조해 범위와 결정을 먼저 고정한다. 이 문서의 사실 항목은 저장소를 직접 읽어 확인한 것이고, 확인하지 못한 항목은 "미확인"으로 표시했다.

## 0. 목표와 범위

요청 항목을 하나의 흐름으로 묶는다.

1. 가상 월드: 오픈월드에서 자원을 채취하고 아이템을 줍고 지점까지 운반하는 환경.
2. 화면 기반 인식: 월드를 렌더링한 프레임을 WebP 스크린샷으로 찍고, SigLIP2 비전 임베딩을 GPU(candle 백엔드)로 계산해 사전의 물체를 찾는다.
3. 병렬 임베딩: logis-center의 병렬 처리 전략(중복 제거, 길이 버킷 마이크로배치, 캐시, 디스크 체크포인트)을 사전 임베딩과 프레임 임베딩에 반영.
4. 자연어 명령: 엔터로 "철광석 3개 캐서 베이스캠프로 가져와"를 입력하면 logis-center의 검색 로직(임베딩 + NLP + 단어 사전)으로 구조화해 실행.
5. 퀘스트 UI: 명령이 게임 퀘스트처럼 단계별 진행 카드로 보인다.
6. 정밀도 개선: logis-center의 SDS(통계 원장)와 endmill-sandbox의 Granite TTM 시계열을 결합하고, 판정 근거를 로그로 남긴다.
7. UI 개편: endmill-sandbox의 색상과 좌·중·우·하단 구조를 따르되, 가운데는 월드 화면 위에 타르코프식 캐릭터 상태(손·발·몸) 오버레이를 얹는다. 기존 손/발/측면도 화면은 유지한다.
8. 시뮬레이터 선정: Horizon 또는 더 가벼운 MIT/Apache-2.0 크로스플랫폼 대안.

비목표: 성능 벤치마크 실행. 이 프로젝트의 기준(README)은 문서화된 참조 수치와 정합성 테스트이므로, 시나리오 스위트도 타이밍 측정이 아니라 정합성 테스트로 둔다.

## 1. 현재 상태 진단 (hand-sandbox, 약 5,600줄)

| 영역 | 있는 것 | 없는 것 |
| --- | --- | --- |
| hand-core | 손가락 2~5개, 관절 옵션, 발 4종(wheel/mecanum/track/human_foot), 순기구학, 파지 시험 `grasp()`, 주행 `simulate_drive()` | 월드 안에서 움직이는 개체, 상태(배터리·하중) 누적 |
| hand-ml | candle SigLIP2 NaFlex, 사전 기반 제로샷 `classify()`, 백엔드 선택(CPU/CUDA/Metal/Vulkan/ROCm) | 배치 이미지 입력, 병렬 사전 임베딩, 캐시, 텍스트 질의 해석 |
| hand-data | LanceDB 스키마, 합성 파지 데이터, 28개 물류 사전 | `vision`·`embeddings` 테이블은 스키마만 있고 "렌더러 필요"로 표시됨 |
| src-tauri | 파지·주행·분류·합성 커맨드 | 월드·퀘스트·명령·로그 커맨드 |
| ui | 상단바 + 3열 + 상태줄, 2D 측면도 캔버스 | 가상 환경 화면, 퀘스트, 명령 입력, 로그 |

확인된 제약: `image` 크레이트는 jpeg/png만 켜져 있고(webp 없음), `Anchors::build`는 32개씩 순차 배치, 분류는 사용자가 올린 이미지 1장만 대상. 즉 "별도 가상환경이 없다"는 지적이 맞다. 렌더러와 프레임 파이프라인이 없다.

## 2. 참조 프로젝트에서 확인한 사실

### 2.1 logis-center (Rust, Tauri 2, candle, LanceDB)
- 검색: `ai_search_complex` → 3개 질의 파서. 질의를 2~8단어 n-gram으로 자르고 일괄 임베딩, 분야별 점수 `max(0, cos_bias − w·cos_prej − 동사 페널티)`, 겹침은 NMS로 정리, 이후 도메인별 컨텍스트 생성. 다국어 한국어 처리는 형태소 분석기가 아니라 조사·어미 닫힌 목록과 받침 일치 검사.
- 사전: `utils/bias.json`(378 KB)을 컴파일 타임에 포함. 노드 형식 `{semantic, bias, prejudice, exact_match}`. `full_bias.json`(18개 언어)은 Rust가 읽지 않음.
- 병렬 임베딩: 중복 제거 → 길이 버킷 마이크로배치(≤512 토큰) → 전역 캐시(LRU + 사전용 고정 계층) → 디스크 체크포인트(레코드별 FNV-1a 체크섬, 손상된 꼬리는 잘라냄). 사전 규모에서는 선형 스캔 + rayon으로 충분.
- SDS: `utils/score_dynamics.rs`. Welford(평균·분산) + 최근값 링, 스코프 `트랙|primary|secondary` 폴백(secondary → primary → global), `decay_shape`(margin·entropy 등), 원장 JSON 원자적 저장, 레시피 문자열이 다르면 원장 폐기. 약어 "SDS"는 코드에 풀이가 없고 "[SCORE DYNAMICS]"로 표기됨.
- 로그: `[TAG]` 접두 stdout + Tauri 이벤트 + 작업별 JSONL. 변조 방지(해시) 없음.
- Granite 시계열은 이 저장소에 없다(전수 검색 결과 0건). Granite는 텍스트 임베딩(97m)과 LLM(350m)만 있다.

### 2.2 endmill-sandbox (Rust, Tauri, candle)
- Granite TTM-R3 candle 구현 `src/ml/ttm.rs`(1,298줄): config.json + safetensors 직접 로드, 컨텍스트 52 → 예측 16, 분위수 헤드, 분해(trend/residual) 지원, 최소 이력 8점. 가중치 5.9 MB. hand-sandbox와 같은 candle 포크를 쓴다.
- `timeseries.rs`: TTM이 없거나 이력이 짧으면 Holt 감쇠 추세로 대체.
- `wearlog.rs`: 홀드아웃 백테스트로 TTM·Holt·물리 3개 예측기를 비교해 역분산 가중 융합하고, SDS에 기록된 과거 RMSE를 `(n·r² + 이번 MSE)/(n+1)`로 섞는다(n 상한 있음).
- `sds.rs`(1,037줄): logis-center SDS를 다듬은 파생. 5개 원장, 계층 쓰기, 보정 수축(shrinkage) 추가.
- `modelhub.rs`: HF 가중치 다운로더(리비전 고정, 진행률). hand-sandbox는 아직 수동 다운로드.
- UI: `index.html` 한 파일. 색상은 아연 무채색 토큰, 레이아웃은 8px 간격 카드 4개(좌·중·우·하단) + 28px 상태줄, 리사이저와 접기.

### 2.3 Horizon (`ludovic111/horizon`)
- Minecraft Java 26.3을 Rust로 다시 구현한 클라이언트/서버(약 61만 줄, 락 패키지 542개). 코드는 MIT.
- 실행에 사용자의 정품 Minecraft 26.3 원본에서 뽑은 데이터가 필수이고 번들·재배포 불가. 준비에 JDK 25, 자원 약 589 MB, 유휴 RAM 약 1.4 GB, Vulkan/Metal GPU 필요.
- 제어는 `--control` 소켓(JSON 한 줄)으로 가능: `find-block`, `goto`, `break`, `screenshot` 등. 단 프레임은 PNG 파일로만 나오고, 결정적 스텝·라이브러리형 API가 없음.
- 플랫폼: Linux x64, Windows x64, macOS arm64(인텔 맥·리눅스 ARM 없음).

## 3. 다층 대조표

| 층 | hand-sandbox 현재 | 참조 | 결정 |
| --- | --- | --- | --- |
| L1 사전 | 28개 물류 항목 JSON (`label_en/ko`, 크기, 질량, graspable) | logis `{semantic,bias,prejudice,exact_match}` 노드 | 기존 사전 유지 + 월드 어휘 사전 `data/world_lexicon.json` 신설(logis 노드 형식). 같은 문구가 SigLIP2 앵커와 명령 해석에 모두 쓰인다 |
| L2 임베딩 | SigLIP2 텍스트·비전 타워 | logis는 Granite 임베딩(384차원) | 텍스트/이미지가 같은 공간인 SigLIP2를 공용 사용. `TextEmbedder` 트레이트로 추후 교체 가능 |
| L3 검색·NLP | 없음 | n-gram 스팬 + bias/prejudice 점수 + NMS + 조사 처리 | 점수식·NMS·배타 할당·한국어 조사 처리를 `hand-search`로 이식. Stanza·LanceDB·Qwen 단계는 이식하지 않음 |
| L4 병렬 | 순차 32배치 | 중복 제거 + 버킷 + 캐시 + 체크포인트 | `hand-ml::parallel`: 문구 은행, 프레임 타일 배치, rayon, 캐시 파일(FNV-1a 안정 해시) |
| L5 SDS | 없음 | `score_dynamics.rs` | `hand-sds`로 핵심(약 400줄 분량)만 이식: Welford, 스코프 폴백, decay_shape, 원장 저장 |
| L6 시계열 | 없음 | endmill `ttm.rs`, `timeseries.rs`, 융합 | `hand-ts`: TTM 이식 + Holt 대체 + 역분산 융합 |
| L7 로그 | 상태줄 한 줄 | logis JSONL + 이벤트 | `hand-sds::audit`: JSONL + 해시 체인(개선) + UI 하단 로그 탭 |
| L8 UI | 3열 + 상단바 | endmill 토큰·레이아웃 | 토큰과 구조 이식, 콘텐츠는 7장 |
| L9 환경 | 없음 | Horizon·대안 | 4장 |
| L10 검증 | 단위 테스트 | endmill `tests/` | 크레이트별 테스트, 모의 임베더로 정밀도 개선을 수치로 검증 |

## 4. 시뮬레이터 결정

평가한 후보(라이선스·크로스플랫폼은 각 저장소 문서 기준, 일부는 미확인 표시).

| 후보 | 라이선스 | 판단 |
| --- | --- | --- |
| Horizon | MIT(코드) | 정품 Minecraft 데이터 필수, 번들 불가, 무겁고 프레임이 PNG 파일뿐. 로컬 전용 선택 백엔드로만 가능 |
| Crafter / Craftax | MIT | 가볍고 헤드리스지만 2D 파이썬. 사이드카 필요, "운반" 과제는 포크 필요 |
| bevy_voxel_world | MIT OR Apache-2.0 | 3D·Rust지만 Bevy 컴파일이 느리고 GPU 필요, 관찰·행동 API는 직접 작성해야 함 |
| Craftium(Luanti) | 엔진 LGPL-2.1+ | 엄격한 MIT/Apache 조건 위반, 소스 빌드, Windows 지원 문서 없음(미확인 다수) |
| Veloren | GPL-3.0 | 제외 |
| MineDojo·MineRL·Malmo·MineStudio 등 | MIT 또는 미표기 | 모두 정품 Minecraft + JDK 필요. Horizon보다 무거움 |

결정: 자체 경량 월드 `crates/world-sim`(워크스페이스 라이선스 Apache-2.0)을 기본 백엔드로 만든다. 이유는 다음과 같다.

- 허용되는 라이선스 후보 중 3D이면서 Rust로 프로세스 안에서 돌고, 자산 의존이 없는 것은 없다(bevy_voxel_world는 하네스를 어차피 직접 써야 함).
- 손·발 모델과 직접 결합해야 한다. 발의 `max_step_m`·`max_slope_deg`로 통과 가능 지형이 정해지고, 손의 `grasp()` 결과로 줍기 성공 여부가 정해진다.
- 인식 정밀도를 수치로 검증하려면 프레임과 같은 시점의 정답(물체 ID·깊이 버퍼)이 필요하다. 외부 게임에서는 얻기 어렵다.
- 플랫폼 제한(Mac Intel, Linux ARM 등)과 GPU 의존이 없다. 렌더링은 CPU 소프트웨어 레이캐스터이고, GPU는 SigLIP2 추론에 쓴다.

`WorldBackend` 트레이트를 두어 Horizon 소켓 어댑터를 나중에 붙일 수 있게 한다. 이 어댑터는 사용자가 정품 Minecraft를 보유한 로컬 환경에서만 쓰는 선택 기능이며 저장소에 번들하지 않는다. 이번 단계에서는 트레이트 경계와 문서까지만 두고 어댑터 구현은 하지 않는다.

한계(정직하게): 이 월드는 Minecraft가 아니다. "자원 채취 → 아이템 줍기 → 운반" 루프의 의미만 참조했고, 지형·물리·콘텐츠는 단순화한다.

## 5. 목표 아키텍처

```
hand-core        손/발 + body(상태 모델)           ← 기존 확장
world-sim        월드·에이전트·아이템·렌더러(RGB+ID+깊이)  ← 신규
hand-ml          SigLIP2 + parallel(문구 은행, 타일 배치, 캐시) ← 확장
hand-search      자연어 명령 해석(사전·n-gram·NMS·조사)       ← 신규
hand-sds         SDS 원장 + 감사 로그(JSONL, 해시 체인)       ← 신규
hand-ts          Granite TTM + Holt + 역분산 융합              ← 신규
hand-quest       퀘스트 모델·실행기(탐색→접근→채취→줍기→운반→인도) ← 신규
hand-runtime     세션 오케스트레이션(위 크레이트를 묶음, Tauri 비의존) ← 신규
src-tauri        얇은 커맨드 래퍼
ui               endmill 레이아웃 + 월드 뷰 + 상태 오버레이 + 퀘스트
```

`hand-runtime`에 로직을 모으는 이유: Tauri(webkit) 없이 `cargo test`로 전 흐름을 검증하기 위해서다.

데이터 흐름(한 틱):

```
명령(Enter) → hand-search → ParsedCommand → hand-quest(퀘스트 생성)
퀘스트 단계 → 목표 위치 필요 → 프레임 렌더(world-sim) → WebP 인코딩 → 디코딩
→ 타일 분할 → 병렬 임베딩(GPU) → 사전 점수 → hand-sds/hand-ts 판정(수락/기각)
→ 수락된 탐지의 깊이로 월드 좌표 → A* 경로(발 제약) → 이동·채취·줍기(손 제약)
→ body 상태 갱신 → 로그/UI
```

인식은 렌더된 프레임을 WebP로 한 번 압축했다가 풀어서 쓴다. 손실 압축의 영향이 인식에 그대로 들어가게 하기 위해서다. 정답 ID 버퍼는 평가와 로그에만 쓰고 행동 결정에는 쓰지 않는다(오라클 모드는 별도 선택).

## 6. 모듈 설계

### 6.1 world-sim
- 격자: 셀 0.5 m, 높이는 1 cm 단위 정수. 셀마다 지면 높이·재질과 선택적 자원 노드(나무, 바위, 석탄 광석, 철 광석, 상자 등). 기본 64×64셀(32 m × 32 m), 시드 기반 값 노이즈 지형.
- 지형 특징: 완만한 언덕 + 의도적인 턱(5·10·15 cm)과 경사로를 배치해 발 4종이 서로 다른 구간을 통과하게 한다. 통과 판정은 이웃 셀 사이 `Δh ≤ max_step_m` 이거나 경사각이 `max_slope_deg` 이하, 이동 비용과 시간·에너지는 `simulate_drive`와 같은 식.
- 채취: 노드마다 경도(hp)와 드롭 표(Minecraft의 경도 관례를 참조). 손 채취 속도는 손가락 수 비례의 가정값. Minecraft의 "도구 필요" 규칙은 두지 않는다(차이점으로 문서화).
- 아이템: 드롭 엔티티, 인벤토리(질량 kg 합계 ≤ 발별 적재 한계, 가정값은 data/README에 출처와 함께 표기). 줍기는 `grasp()` 결과가 안정일 때만 성공(손가락 수·마찰·크기가 영향).
- 렌더러: 3D 격자 DDA 레이캐스트(CPU, 결정적). 출력은 RGB 프레임 + 물체 ID 버퍼 + 깊이 버퍼. 기본 384×256.
- 정답: ID 버퍼에서 프레임 안 물체와 픽셀 면적을 얻는다.

### 6.2 프레임 캡처와 WebP
- 인코딩: libwebp 바인딩 `webp` 크레이트로 손실 압축(기본 품질 75), 실패 시 `image`의 무손실 인코더로 대체. 확인된 사실: `image-webp`의 인코더는 무손실만 지원하므로 손실 압축에는 libwebp가 필요하다.
- 디코딩: `image`의 webp 기능(순수 Rust).
- 프레임 패킷에 인코딩 시간·바이트 수를 기록해 UI와 로그에 표시.

### 6.3 병렬 임베딩 (`hand-ml::parallel`)
- `PhraseBank`: 문구 중복 제거 → 토큰 길이 버킷 → 마이크로배치(기본 32행) → 캐시 조회/기록. 캐시 키는 안정 해시(FNV-1a)로 한다(logis의 DefaultHasher 키는 Rust 버전 간 비호환이라 개선).
- 디스크 캐시: 헤더(매직, 버전, 차원, 모델 지문) + 레코드(키, 벡터, 체크섬), 손상된 꼬리는 복구 시 잘라냄, 모델 지문이 다르면 폐기.
- 프레임 쪽: 한 프레임을 겹치는 타일로 나눠(전체 + 격자) rayon으로 전처리, 같은 격자 크기끼리 묶어 배치 순전파. 타일별 점수 행렬은 rayon으로 병렬 계산.
- 점수 보정: logis의 bank-neutral 아이디어를 단순화해 라벨별 z-점수와 prejudice(부정 문구) 차감을 적용. 항상 점수가 높은 라벨의 영향을 줄이는 용도.
- 비전 타워가 배치 순전파를 지원하는지는 구현 시 확인한다(현재 `embed_image`는 이미지 1장). 지원하지 않으면 타워에 배치 차원을 추가한다.

### 6.4 자연어 명령 (`hand-search`)
- 어휘 사전 `data/world_lexicon.json`: 행동(채취·줍기·운반·내려놓기), 아이템(한/영 동의어, SigLIP2 프롬프트), 장소(웨이포인트), 숫자(한/두/세…, 아라비아 숫자, 개/조각). 노드 형식은 logis의 `{semantic,bias,prejudice,exact_match}`.
- 처리: 정규화 → 조사·어미 분리(받침 일치 검사로 을/를, 이/가 등 구분) → 1차 정확 일치 → 2차 n-gram 임베딩 점수(`TextEmbedder`) → NMS → 역할(행동·아이템·수량·목적지)별 배타 할당 → `ParsedCommand`.
- 임베더가 없으면(가중치 미설치) 어휘 일치만으로 동작하고, 로그에 "어휘 전용 모드"를 남긴다.
- 모든 단계의 후보·점수를 `trace`로 반환해 UI "명령 해석" 패널과 로그에서 볼 수 있게 한다.

### 6.5 SDS + Granite TTM (`hand-sds`, `hand-ts`)
- SDS 트랙: `vision`(라벨별 탐지 점수·margin), `search`(역할별 점수), `motion`(경로·채취 성공), `ts`(예측기별 오차). 스코프 `트랙|primary|secondary` 폴백과 최소 관측 수는 logis 방식. 레시피 문자열이 달라지면 원장 폐기.
- 정밀도 개선 3가지.
  1. 적응형 임계값: 라벨별 점수 평균·표준편차(스코프 폴백)로 임계값을 정해, 전역 고정 임계값보다 라벨 특성을 반영.
  2. 시간 일관성 판정: 라벨별 점수 시계열을 TTM(없으면 Holt)으로 예측하고, 새 관측이 예측 분위수 띠(P10~P90)를 SDS 표준편차만큼 넓힌 범위를 벗어나면 이상치로 보고 보류. 순간 오탐을 줄이는 목적.
  3. 예측기 융합: 홀드아웃 백테스트의 역분산 가중에 SDS가 기억하는 과거 RMSE를 섞는다(endmill `wearlog.rs`와 같은 식).
- TTM 가중치는 `models/granite-timeseries-ttm-r3/52-16-dec-52-r3/`(config.json + model.safetensors, 약 5.9 MB), `TTM_DIR`로 변경 가능. 없으면 Holt로 자동 대체하고 로그에 사유를 남긴다. 컨텍스트가 모자라면(52점 미만) endmill에서 겪은 문제(0 패딩이 첫 예측을 튀게 함)를 피하기 위해 실제 이력 길이를 존중하고, 최소 8점 미만이면 Holt를 쓴다.
- 검증: 샌드박스에는 TTM 가중치가 없으므로, 무작위 초기화 가중치로 형태·유한성 테스트를 하고, 정밀도 개선은 모의 점수 시계열(정답 ID 포함)로 "게이트 적용 전/후 정밀도"를 비교하는 테스트로 확인한다. 실제 가중치에서의 개선폭은 사용자의 환경에서 확인해야 한다.

### 6.6 퀘스트 (`hand-quest`)
- 단계: 탐색(Locate) → 접근(Approach) → 채취(Mine) → 줍기(Collect) → 운반(Carry) → 인도(Deliver). 상태 `pending/active/done/failed`.
- 실패 처리: 탐색 시간 초과(회전·이동 후 재시도 횟수 상한), 줍기 불가(graspable=false 또는 불안정 파지), 경로 없음(발 제약), 적재 초과. 각 실패는 사유와 함께 퀘스트 카드와 로그에 남는다.
- 보상/점수: 완료 시간, 에너지, 탐지 정밀도를 요약한 결과 카드(게임 퀘스트 완료 화면처럼).

### 6.7 상태 모델 (`hand-core::body`)
타르코프의 부위별 체력을 로봇 상태에 대응한다. 값은 모두 시뮬레이션 변수에서 계산하고, 가정값은 data/README에 출처와 함께 기록한다.

| 부위 | 의미 | 계산 |
| --- | --- | --- |
| 머리 | 비전 | 최근 탐지 margin, 프레임 지연 |
| 가슴 | 배터리 | 1 − 누적 에너지 / 용량(가정값) |
| 배 | 적재 | 적재 질량 / 한계 |
| 왼팔 | 손: 파지 | 마찰 용량 대비 필요 힘 여유 (`grasp()` 결과) |
| 오른팔 | 손: 관절 | 채취 타격 누적에 따른 피로(가정값) |
| 왼다리 | 발: 구동 | 현재 속도 / 최고 속도. **속도 계기**라서 멈춰 있어도 이상이 아니다(수준은 배터리가 바닥났을 때만 crit) |
| 오른다리 | 발: 지형 | 다음 구간의 경사·턱 한계 여유 |

상태 효과 줄: 피로, 하중, 미끄러짐(경사 여유 낮음). 처음 계획했던 과열 효과는 근거가 되는 변수가 없어 뺐다. 수준은 ok/warn/bad/crit(비율 0.6/0.3/0.1 경계).
로봇은 손이 하나라 "왼팔/오른팔"은 해부학적 좌우가 아니라 손의 두 지표(파지·관절)로 표시하고 UI에 그렇게 라벨링한다.

### 6.8 구현하면서 바뀐 점
- 돌 조각은 0.10 m 정육면체(질량 2.6 kg)로 정했다. 손가락 2개 손은 마찰 여유가 0.77이라 돌을 줍지 못하는데, 이것이 "손 구성에 따라 성공/실패가 갈리는" 시나리오 테스트의 기준 사례다.
- `hand-ml::parallel`은 문구 은행과 타일 임베딩 모두 rayon으로 병렬 처리한다. 은행 캐시는 `data/runtime/bank`에 쌓인다.
- `hand-search` 어휘 사전의 노드에는 logis 형식에 없는 `surface` 필드를 더했다. 완전한 어형("캐서", "베어와", "모아줘" 등)이 통째로 일치할 때만 잡히는 목록으로, 한국어 활용형을 임베딩 없이 처리하려는 것이다.
- JSON 왕복에서 f32 값이 흔들리지 않도록 `serde_json`의 `float_roundtrip`을 켰다(SDS 원장과 로그의 해시 체인 검증이 같은 바이트를 요구).
- 인식기는 두 가지다. `OraclePerception`(기본값)은 정답 ID 버퍼와 주입한 오류로 만든 대역이며 점수 모델은 임의로 정했다. `EmbeddingPerception`은 WebP로 한 번 압축했다 푼 스크린샷 타일을 SigLIP2로 임베딩한다. 후자는 이 샌드박스에서 가중치가 없어 가짜 임베더(단어 축·색상)로만 시험했다.
- Granite TTM은 `Runtime::attach_ttm`/`load_ttm`으로 연결한다. 연결하면 융합 가중치 표에 `ttm`이 나타난다(랜덤 가중치 시험으로 확인).
- Tauri 명령은 `world_new`부터 `log_tail`까지 10개에 `models_status`, `recognizer_use`, `ttm_use`가 더해졌다.
- 로그 시각은 UTC(ISO 8601)다. UI는 그대로 보여 준다.

## 7. UI 설계 (endmill 구조 이식)

토큰: endmill `index.html` 8~63행의 `:root`를 그대로 이식(아연 무채색 + 상태색 danger `#e06c6c`, alert `#d9924f`, warning `#d6b65e`, success `#5fb98c`). 기존 hand-sandbox의 라이트 테마는 제거하고 다크 단일 테마로 맞춘다(기존 색은 `--thumb/--finger/--object` 같은 캔버스용 토큰으로 이전).

레이아웃: 카드 4개를 8px 간격으로 띄운 격자 + 28px 상태줄, 리사이저(드래그, 더블클릭 초기화), 접기 버튼.

| 영역 | 콘텐츠 |
| --- | --- |
| 왼쪽 | 메뉴(월드·손/발·분석), 아코디언: ① 손 구성(기존) ② 발(기존) ③ 월드(시드, 크기, 재생성) ④ 사전·모델(장치, SigLIP2, TTM 상태) |
| 가운데 | 툴바(재생/정지/속도/시점, 표시 토글) + 월드 프레임 + 오버레이: 좌측 상단 신체 상태 HUD(실루엣 7부위 + 상태 효과), 우측 상단 퀘스트 트래커, 하단 중앙 명령 입력(Enter), 좌측 하단 측면도 PiP(기존 손/발 캔버스), 우측 하단 범례. 탭: 월드 / 손·발 측면도 / 분석 |
| 오른쪽 | 퀘스트 목록과 단계 체크리스트·진행 바, 인벤토리, 파지 결과(기존), 발별 주행 비교(기존) |
| 하단 | 탭: 로그(SDS·감사) / 비전 점수 / 명령 해석 / 시계열(예측 띠 차트) / 합성 데이터(기존) |
| 상태줄 | 장치, 시드, 틱, 프레임 WebP 크기, SDS 레시피, TTM/임베더 상태 |

기존 기능(손 구성, 발 비교, 파지, 이미지 분류, 합성 데이터)은 모두 유지하고 위치만 옮긴다. 글꼴은 endmill 스택을 쓰되 웹폰트는 번들하지 않는다(시스템 폰트로 대체됨).

타르코프 오버레이 구현: 인라인 SVG 실루엣 7영역, 부위별 색은 수준에 따라 `--success/--warning/--alert/--danger`, 반투명 유리 패널(endmill의 HUD 패턴: `rgba(17,17,19,.84~.95)` + blur). 부위에 마우스를 올리면 세부 수치를 툴팁으로 보여준다.

## 8. Tauri 커맨드 계약 (UI와 백엔드의 경계)

기존 커맨드는 유지한다(`presets`, `hand_geometry`, `grasp_test`, `drive_courses`, `dictionary`, `devices`, `select_device`, `classify_image`, `default_paths`, `synthesize`, `load_dictionary`, `dataset_summary`). 신규(인자는 camelCase, 응답은 snake_case JSON):

| 커맨드 | 인자 | 응답 |
| --- | --- | --- |
| `world_new` | `{seed, size, config, base}` | `WorldSnapshot` |
| `sim_step` | `{ticks}` | `{snapshot, quests, body, events[]}` |
| `world_frame` | `{width, height, quality, view}` | `{webp_base64, width, height, bytes, encode_ms, tick, detections[]}` |
| `command_submit` | `{text}` | `{ok, parsed, quest_id, trace[], error}` |
| `quest_list` | — | `Quest[]` |
| `perception_scan` | `{topK}` | `{labels[], tiles, embed_ms, device, mode}` |
| `body_status` | — | `{parts[], effects[], hand, foot}` |
| `sds_status` | — | `SdsStatus` |
| `ts_forecast` | `{scope, axis}` | `{history[], point[], p10[], p90[], method, weights[]}` |
| `log_tail` | `{afterSeq, limit, category}` | `{lines[], next_seq}` |
| `models_status` | — | `{recognizer, embedding, ttm, parser_embedding, device, siglip_files, ttm_files}` |
| `recognizer_use` | `{mode: "embedding" \| "oracle"}` | `models_status`와 같음 |
| `ttm_use` | `{enabled}` | `models_status`와 같음 |

핵심 타입:
- `WorldSnapshot { tick, time_s, bounds, agent{pos, yaw_deg, speed_mps, base, carrying[]}, waypoints[{id,label_ko,pos}], nodes[{id,kind,pos,hp}], drops[{id,item,pos}] }`
- `Quest { id, title, state, steps[{kind,label_ko,state,progress}], created_tick, result? }`
- `BodyStatus.parts[{id,label_ko,value,max,ratio,level,detail}]`, id는 `head|thorax|stomach|arm_l|arm_r|leg_l|leg_r`
- `LogLine { seq, ts, cat, level, msg, data? }`

## 9. 로그 설계
- 범주: `world`, `quest`, `search`, `vision`, `sds`, `ts`, `embed`, `device`.
- 저장: `{app_data}/hand-sandbox/logs/{session}.jsonl`(줄마다 JSON), 줄에 직전 줄 해시를 포함하는 해시 체인으로 변조를 검출(logis에는 없는 개선).
- UI: 하단 로그 탭에서 범주 필터, 증분 조회(`after_seq`), 사유가 있는 기각·보류는 사유 코드와 수치를 함께 표시.
- 판정 로그 예: 라벨, 관측 점수, 예측 P10/P50/P90, SDS 표준편차, 수락/보류, 사용한 예측기와 가중치.

## 10. 검증 계획과 이 환경의 한계
- 크레이트별 단위 테스트와 `hand-runtime`의 시나리오 테스트(결정적 시드): 채취→줍기→운반이 발·손 구성에 따라 성공/실패가 갈리는지, 명령 해석, SDS 폴백·저장·복구, 해시 체인 검증, WebP 왕복.
- 정밀도 개선: 모의 인식기로 "게이트 적용 전/후" 정밀도·재현율 비교.
- UI: 정적 파일이라 Tauri 없이 모의 백엔드로 구동해 확인(가능한 도구가 있을 때).
- 이 샌드박스의 제약(확인됨): Hugging Face 접속 불가라 SigLIP2·TTM 실제 가중치로는 실행할 수 없고, GPU가 없어 GPU 경로는 컴파일 여부만 확인할 수 있으며, Tauri 빌드에 필요한 webkit2gtk 개발 패키지가 없다. 따라서 Tauri 앱 전체 빌드와 실제 가중치 정확도는 사용자 환경에서 검증해야 한다.

## 11. 마일스톤
1. body 모델, world-sim, WebP 캡처 (+ 테스트)
2. hand-sds, hand-ts (+ 테스트)
3. hand-ml parallel, hand-search (+ 테스트)
4. hand-quest, hand-runtime (+ 시나리오 테스트)
5. Tauri 커맨드 래퍼
6. UI 개편
7. 문서·README 갱신, commit, push

## 12. 위험
- 소프트웨어 레이캐스터의 장면이 SigLIP2에게 학습 분포 밖이어서 제로샷 정확도가 낮을 수 있다. 완화: 사전 프롬프트에 장면 맥락 문구를 추가하고, 정확도는 SDS로 라벨별 보정하되, 실제 정확도는 가중치로 측정한 뒤에 판단한다.
- SigLIP2의 한국어 질의 품질은 logis의 Granite 다국어 임베딩보다 낮을 수 있다. 완화: 어휘 일치 1차 경로, `TextEmbedder` 교체 지점.
- 가정값(적재 한계, 배터리 용량, 채취 속도)은 근거가 약하다. data/README에 가정으로 명시.
- `webp` 크레이트는 C 컴파일러가 필요하다. 없으면 무손실 대체 경로로 동작.

## 13. 라이선스 메모
이 저장소는 Apache-2.0. logis-center·endmill-sandbox 코드는 같은 작성자의 저장소이므로 로직을 이식하되, 출처 파일을 각 모듈 머리 주석에 남긴다. Crafter(MIT)는 개념 참조만 했고 코드를 복사하지 않는다. Horizon·Minecraft 자산은 포함하지 않는다.
