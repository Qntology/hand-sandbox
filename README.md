# hand-sandbox

## 물류 로봇 손 시뮬레이터 (Rust + Tauri)

손가락 수(2–5)와 관절 옵션, 발(바퀴·메카넘·트랙·사람 발)을 바꿔 보며 파지와 주행을 시뮬레이션합니다. SigLIP2로 물류 물체를 분류하고, 합성 데이터를 LanceDB에 기록합니다.

v0.3에서는 테스트를 **게임 로직**과 **ML 로직**으로 나눴습니다. 게임 로직이 퀘스트를 수행하며 영상과 고정 간격 키프레임을 찍고, 각 키프레임을 ML 로직(TIPSv2 DPT 깊이 → SigLIP2 의미 위치 → SAM 3.1 방식 마스크 추적)이 분석합니다. 이어서 같은 월드에서 ML 로직이 직접 결정합니다: 거리를 믿기 전에 **의무 삼각 측량 이동**을 하고, 대상 위치를 시계열 모델로 예측해 움직이고, 다시 찍습니다. 몸은 리서치한 부품 프리셋(머리·가슴·배·팔·다리/섀시)으로 조립되어 들 수 있는 물건, 속도, 경사·턱, 적재가 달라지고, 자세 밸런스는 사람 기준값 대비 N:N으로 SDS에 기록됩니다. 기획은 [`docs/PLAN-v0.3.md`](docs/PLAN-v0.3.md), 리서치는 [`docs/research/`](docs/research/)에 있습니다.

v0.2부터는 **오픈월드 자원 채취 테스트베드**가 있습니다. 마인크래프트식 격자 월드(나무·바위·석탄·철광석)에서 로봇이 "철광석 2개 캐서 창고로 가져와"라는 명령을 받아 퀘스트(탐색 → 접근 → 채취 → 줍기 → 운반 → 인도)로 수행합니다. 렌더한 프레임을 WebP로 압축해 SigLIP2로 인식하고, SDS와 Granite TTM으로 인식 결과를 걸러 내며, 모든 판단을 해시 체인 로그로 남깁니다. 설계 근거와 logis-center·endmill-sandbox와의 대조는 [`docs/PLAN.md`](docs/PLAN.md)에 있습니다.

### 구성

| 경로 | 역할 |
| --- | --- |
| `crates/hand-core` | 손 모델: 손가락 프리셋, 관절 옵션, 발(이동 베이스), 순기구학, 파지 시뮬레이션, 주행 모델, 부품 프리셋과 전신 밸런스(CoM·ZMP·관절 토크·N:N 결합) |
| `crates/hand-ml` | candle 기반 SigLIP2(NaFlex) 임베딩, 사전 기반 제로샷 분류, 장치 선택 (CPU, CUDA, Metal, Vulkan, ROCm) |
| `crates/hand-data` | 합성 데이터 계층 정의, 도메인 무작위화, LanceDB 저장 |
| `crates/world-sim` | 오픈월드: 격자 지형, 자원 노드, 아이템, 소프트웨어 레이캐스트 프레임(RGB·ID·깊이), WebP 캡처 |
| `crates/hand-search` | 자연어 명령 해석: 단어사전, 조사 분리, 어휘 일치 + 임베딩 보강, 해석 과정(trace) |
| `crates/hand-quest` | 퀘스트 모델(탐색·접근·채취·줍기·운반·인도)과 진행 상태 |
| `crates/hand-sds` | SDS 통계 원장(스코프 폴백, 원자적 저장, 기준값 대비 편차), 해시 체인 감사 로그, 테스트 기록 DB(퀘스트 성공률·키프레임·밸런스 노력) |
| `crates/hand-ts` | 시계열 예측: Granite TTM-R3, last/ewma/holt, 홀드아웃 융합, 시간 일관성 판정 |
| `crates/hand-vision` | ML 로직(게임 상태를 모름): TIPSv2-B/14 DPT 깊이(candle 이식), SigLIP2 패치 히트맵 의미 위치, 마스크 추적(SAM 인터페이스 + 영역 확장 대역), 방위 삼각 측량, 시계열 결정, 혼합 특징 |
| `crates/hand-runtime` | 위를 엮는 실행기: 인식(오라클·SigLIP2 임베딩), 정밀도 게이트, 퀘스트 실행(게임 로직), ML 결정 주행, 키프레임 녹화, 통합 테스트, SDS·DB 기록, UI용 JSON API |
| `src-tauri` | Tauri 커맨드. 위 크레이트를 UI에 연결 |
| `ui` | 정적 HTML·JS (빌드 단계 없음) |
| `data` | 물류 어휘 사전(JSON)과 데이터 출처 문서 |

### 테스트

```bash
cargo test --workspace
```

벤치마크는 실행하지 않습니다. 작업 단위의 기준은 문서화된 참조 수치와 정합성 테스트입니다.

### 앱 실행

```bash
cd src-tauri
cargo tauri dev
```

GPU 백엔드는 기능 플래그로 켭니다. 해당 SDK가 설치되어 있어야 합니다.

```bash
cargo tauri dev --features cuda      # NVIDIA (CUDA toolkit 필요)
cargo tauri dev --features vulkan    # Vulkan 1.1+ GPU (로더만 필요)
cargo tauri dev --features rocm      # AMD ROCm / HIP SDK (HIP_PATH 설정)
cargo tauri dev --features metal     # macOS
```

Windows에서 CUDA를 쓰려면 다음 두 가지가 필요합니다.

- MSVC 호스트 컴파일러(`cl.exe`)가 PATH에 있어야 합니다. Visual Studio 개발자 셸에서 실행하거나 `vcvars64.bat`을 먼저 실행하세요.
- CUDA 13의 CCCL 헤더는 MSVC의 표준 전처리기를 요구합니다. nvcc에 아래 플래그를 넘깁니다.

```bash
export NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"
```

GPU의 compute capability를 지정하면 커널 빌드가 한 아키텍처만 컴파일해 빨라집니다. 예: RTX 3050 Laptop은 8.6.

```bash
export CUDA_COMPUTE_CAP=86
```

ROCm과 Vulkan은 [Qntology/candle-rocm-vulkan](https://github.com/Qntology/candle-rocm-vulkan)의 candle 포크를 씁니다. 커밋 `2191c7ad`로 고정되어 있습니다(루트 `Cargo.toml`).

### 월드 시뮬레이터

월드는 Tauri 앱의 명령(`world_new`, `sim_step`, `command_submit` 등)으로 쓰며, 입력줄에 명령을 치고 Enter를 누르면 퀘스트가 만들어집니다. Tauri 없이 브라우저에서 실제 Rust 런타임과 UI를 보려면 개발용 서버를 씁니다(127.0.0.1에서만 받습니다).

```bash
cargo run --release -p hand-runtime --example serve   # http://127.0.0.1:8787/
```

로그와 SDS 원장은 `data/runtime`(또는 `RUNTIME_DIR`)에 쌓입니다. 기본 인식기는 정답 ID 버퍼를 쓰는 대역(오라클)이며, 왼쪽 ④ "사전 · 모델"에서 **SigLIP2 인식 켜기**와 **TTM 켜기**로 바꿉니다(Tauri 앱에서만 동작, 가중치 필요). 시뮬레이터의 숫자는 대부분 가정값이니 `data/README.md` §7을 먼저 보세요.

### 통합 테스트 (v0.3)

하단 패널의 **통합 테스트** 탭에서 명령(예: "원목 1개를 창고로 가져와")을 넣고 실행하면 세 번 통과합니다.

1. 게임: 퀘스트 실행기가 수행하며 5 fps 영상과 1초 간격 키프레임을 기록 (타임라인 1줄)
2. ML 인식: 각 키프레임의 깊이·의미 위치·추적 (2줄)
3. ML 결정: 같은 시드로 다시 시작해 ML 로직이 탐색·접근을 결정 (3줄). 4줄은 키프레임마다 트랙별 SDS z-점수

키프레임을 누르면 원본, 깊이·마스크, 삼각 측량, 결정, 밸런스(관절 토크와 N:N 셀)가 나옵니다. "성공률 · N:N" 보기에는 퀘스트·단계별 성공률(DB)과 사람 기준 대비 N:N 행렬이 있습니다. 결과는 `data/runtime/db/*.jsonl`과 `quest_stats.json`에 쌓이고, "DB(Lance) 기록"(또는 테스트 종료 시 자동)으로 LanceDB 테이블 `quest_runs`, `quest_steps`, `quest_stats`, `ml_keyframes`, `balance_effort`에 복사됩니다. 왼쪽 ⑤ **부품 프리셋**에서 부위별 등급을 바꾸면 들 수 있는 물건과 한계가 바로 계산됩니다.

깊이와 의미 위치는 가중치가 없으면 대역 모델로 돌고, 모드 칩에 `stand-in`으로 표시됩니다. TIPSv2 가중치는 아래처럼 받아 두면 Tauri 앱의 `tips_use` 명령으로 연결됩니다(`TIPS_DIR`로 위치 변경).

```bash
huggingface-cli download google/tipsv2-b14-dpt --local-dir models/tipsv2-b14-dpt
```

SAM 3.1(`avencera/sam3.1-mlx-bf16`)은 Apple MLX용 가중치라 이 Windows/CUDA 빌드에서 돌릴 수 없습니다. 추적은 SAM과 같은 인터페이스(포인트 프롬프트 → ID가 유지되는 마스크)에 이미지·깊이만 쓰는 영역 확장 추적기를 붙였습니다. 실제 SAM 계열을 붙이는 경로는 `docs/PLAN-v0.3.md` 4장에 있습니다.

### SigLIP2 가중치

가중치는 저장소에 넣지 않습니다. 아래 위치에 받아 두면 앱이 자동으로 읽습니다. 다른 위치는 `SIGLIP2_DIR`로 지정합니다.

```bash
huggingface-cli download google/siglip2-base-patch16-naflex --local-dir models/siglip2-base-patch16-naflex
```

필요한 파일은 `config.json`, `tokenizer.json`, `model.safetensors`입니다.

### Granite TTM 가중치

시계열 예측에 쓰는 Granite TTM-R3는 `models/granite-timeseries-ttm-r3`(또는 그 아래 `52-16-dec-52-r3`)에서 읽으며, 다른 위치는 `TTM_DIR`로 지정합니다. `config.json`과 safetensors가 필요합니다. 파일이 없으면 last/ewma/holt 융합만 쓰고, ④ "사전 · 모델"에서 이유를 보여 줍니다.

### 합성 데이터 (LanceDB)

기본 저장 위치는 `data/lance`이며 `LANCE_DIR`로 바꿀 수 있습니다. 앱의 "합성 데이터 생성"은 도메인 무작위화한 에피소드를 다음 테이블에 씁니다.

- `samples`: 손 구성, 물체 크기·질량·관성·마찰·포즈, 파지 결과
- `contacts`: 접촉점, 법선, 접촉력
- `joint_states`: 관절 굴곡·외전·비틀림, 관절 한계, 토크 한계
- `dictionary`: 물류 어휘 사전

포인트 클라우드, RGB·깊이·마스크, 촉각, SigLIP2 임베딩 테이블은 스키마만 정의되어 있습니다. 계층별 준비도와 출처는 `crates/hand-data/src/layers.rs`와 `data/README.md`에 있습니다.

### 현재 한계

- 파지는 키네마틱 모델입니다. 물리 엔진이 아니며, 접촉은 물체가 손바닥 위에 놓인 상태에서 손가락이 처음 닿는 지점으로 계산합니다.
- 발(베이스)의 속도·경사·턱 한계는 모델링 가정값입니다. 출처와 검증 항목은 `data/README.md`에 있습니다.
- 가중치 파일이 없으면 분류 기능은 명확한 오류 메시지를 보여 줍니다.
- 월드의 SigLIP2 인식과 Granite TTM은 실제 가중치로 정확도를 재 본 적이 없습니다. 테스트는 가짜 임베더와 0·무작위 가중치로 연결만 확인했고, 정밀도 게이트의 임계값과 "게이트 전/후" 수치는 모의 오류 모델에서 나온 값입니다.
- 월드는 소프트웨어 레이캐스터로 그려서 SigLIP2의 학습 분포와 많이 다릅니다. 제로샷 정확도는 가중치로 측정한 뒤에 판단해야 합니다.
- GPU 경로(CUDA·Metal·Vulkan·ROCm)는 이 개발 환경에 GPU가 없어 컴파일 여부만 확인했습니다.
- 한국어 명령은 어휘 사전 일치가 1차 경로입니다. 사전에 없는 표현의 임베딩 보강은 SigLIP2 텍스트 타워의 한국어 품질에 달려 있습니다.
- Tauri 창에서 직접 돌려 보지는 않았습니다. UI는 개발용 서버 위에서 브라우저로 확인했습니다. 로그 시각은 UTC입니다.
- TIPSv2 DPT 이식은 실제 체크포인트 헤더의 텐서 이름·형상(227개)을 전부 대조했지만, 가중치로 PyTorch와 수치를 비교하지는 않았습니다. SigLIP2 패치 히트맵(MAP 헤드 value 경로)이 NaFlex 체크포인트에서 잘 되는지도 가중치로 확인해야 합니다.
- 밸런스 모델은 준정적 근사입니다(점 질량, 지지 다각형, 관절 토크). 동역학 시뮬레이터가 아니며, 사람 기준값 중 "유도" 표시된 것은 계산으로 얻은 값입니다.
- 통합 테스트의 ML 수치(깊이 오차, 추적 IoU, 깊이 축척, 결정 횟수)는 대역 모델에서 나온 값이라 실제 모델 성능이 아닙니다.
