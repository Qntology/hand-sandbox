# hand-sandbox

## 물류 로봇 손 시뮬레이터 (Rust + Tauri)

손가락 수(2–5)와 관절 옵션, 발(바퀴·메카넘·트랙·사람 발)을 바꿔 보며 파지와 주행을 시뮬레이션합니다. SigLIP2로 물류 물체를 분류하고, 합성 데이터를 LanceDB에 기록합니다.

v0.3에서는 테스트를 **게임 로직**과 **ML 로직**으로 나눴습니다. 게임 로직이 퀘스트를 수행하며 영상과 고정 간격 키프레임을 찍고, 각 키프레임을 ML 로직(TIPSv2 DPT 깊이 → SigLIP2 의미 위치 → SAM 3.1 방식 마스크 추적)이 분석합니다. 이어서 같은 월드에서 ML 로직이 직접 결정합니다: 거리를 믿기 전에 **의무 삼각 측량 이동**을 하고, 대상 위치를 시계열 모델로 예측해 움직이고, 다시 찍습니다. 몸은 리서치한 부품 프리셋(머리·가슴·배·팔·다리/섀시)으로 조립되어 들 수 있는 물건, 속도, 경사·턱, 적재가 달라지고, 자세 밸런스는 사람 기준값 대비 N:N으로 SDS에 기록됩니다. 기획은 [`docs/PLAN-v0.3.md`](docs/PLAN-v0.3.md), 리서치는 [`docs/research/`](docs/research/)에 있습니다.

v0.4에서는 세 가지가 더해졌습니다. ① **GPU 가속**: 앱과 개발 서버가 이 빌드의 첫 GPU(CUDA → ROCm → Vulkan → Metal, `HAND_DEVICE`로 고정)에서 시작하고, SigLIP2·TIPSv2 깊이·SAM 마스크·Granite TTM이 한 장치를 공유하며, 깊이·밀집 비전 임베딩·SAM 이미지 인코더는 키프레임을 묶어 한 번에 돌립니다. Windows용 `.bat`(아래)이 백엔드별 툴체인을 맞춥니다. ② **관절 동역학과 3인칭 뷰**: 프리셋의 질량·액추에이터로 관절을 실제로 움직이는 강체 동역학(`hand_core::dynamics`)과, 시점 선택 "3인칭 (관절 물리)"에서 관절마다 각도·토크/피크·포화·멈춤, CoM·동적 ZMP·지지면을 겹쳐 보는 추적 카메라. ③ **복합 N:N**: 여러 원인이 함께 작동하는 시나리오를 BESTest·Mini-BESTest·REBA·RULA·NIOSH의 가중치로 사람 기준과 비교하고, 퀘스트 중에도 실시간으로 채점합니다. 기획은 [`docs/PLAN-v0.4.md`](docs/PLAN-v0.4.md), 근거와 개선 기록은 [`docs/research/nn-scenarios.md`](docs/research/nn-scenarios.md)에 있습니다.

v0.5에서는 ① 적재함의 물건을 **명령과 버튼으로 내려놓고**("원목 1개 내려놔", "베이스캠프에 다 내려놔"), **월드 리셋·신체 상태 리셋**을 따로 하며, 웨이포인트에 쌓인 **더미**가 화면·미니맵·오른쪽 패널에 보입니다. ② endmill-sandbox의 **게이트 결정 모델**로 운반을 정합니다: 한 번에 못 들면 들 수 있는 만큼 **왕복**하고, 팔이 못 드는 돌은 **깨서 조각**(예: 50%)으로 옮깁니다. ③ 재생 중에 낸 명령은 라이브 월드에서 **통합 테스트 1–3줄로 자동 기록**되고, 테스트 명령·실행·중지가 화면 위 툴바로 모였습니다. ④ **험지 맵**(바다·절벽 고원·암벽)에서 시야가 가려지면 다른 시점을 골라 찾습니다. ⑤ **피로·하중·미끄러짐·관절 부하·자세 안정**을 지면(8종)과 N:N으로 물리 엔진에 넣었고(지면 마찰·구름 저항·보행 에너지, 감속기 수명, 지면별 적재, 넘어짐/미끄러짐), 엔진 입력과 다른 출처의 측정값으로 채점하는 **환경 N:N 벤치마크**로 단계별 개선을 기록했습니다. 기획은 [`docs/PLAN-v0.5.md`](docs/PLAN-v0.5.md), 리서치·벤치마크·개선 기록은 [`docs/research/effects.md`](docs/research/effects.md)에 있습니다.

v0.6에서는 **행위**(잡기·놓기 · 밀기 · 걷기 · 굽히기)를 마찰 · 무게 · 탄성 · 충돌의 시계열 물리로 바꿨습니다. ① 전신 기구학(`hand_core::kin`)이 굽힘을 관절 범위와 토크 안에서 계획하고, 잡기는 조준 오차 · 패드 마찰의 산포 · 쥠 안전계수 · 연질 접촉의 비틀림 · 손가락 충돌 · 들기 관성으로, 놓기는 낙하 · 착지 충격 · 반발로, 밀기는 한계면과 안정 회전 반경으로, 걷기는 LIPM 걸음 · 미끄럼 · 넘어짐으로, 주행은 견인–미끄럼으로 풉니다(`hand_core::act`). 성공은 더 이상 결정론이 아니며, 게임은 다시 잡기 · 범퍼로 밀어 옮기기 · 넘어지면 일어나기로 대응합니다. 기준 로봇의 기하로는 **바퀴·트랙 프리셋이 바닥의 물건에 손이 닿지 않아** 밀어서 옮깁니다. ② 행위의 물리 순간마다 키프레임을 찍고 **부위 포커스**(손 모양 · 손 관절 · 발 모양 · 발 관절 · 허리 · 목 관절 · 바퀴 · 트랙)와 근접 화면을 붙여, 통합 테스트 타임라인의 "행위" 줄과 "칸" 선택, 검사기, 월드 화면의 포커스 시점에서 움직이는 부위를 따라갑니다. ③ 엔진과 다른 출처의 사람·로봇 측정값으로 결과 · 시계열 · 부위 · N:N 결합을 채점하는 **행위 N:N 벤치마크**와 SDS `act` 트랙을 더했습니다. 기획은 [`docs/PLAN-v0.6.md`](docs/PLAN-v0.6.md), 리서치 · 벤치마크 · 개선 기록과 남은 차이는 [`docs/research/actions.md`](docs/research/actions.md)에 있습니다.

v0.7에서는 로봇이 **눈을 늘 뜨고 있지 않습니다**. ① 오도메트리(바퀴 엔코더·다리 기구학 + 자이로)가 이동마다 드리프트하고(축척 오차·무작위 보행·자이로 바이어스·미끄러진 거리), 로봇은 자기가 **믿는 위치**로 결정하며, 본 것은 본 뒤 움직인 거리만큼 퍼지는 **기억**(σ)으로 남습니다. ② 비전은 **패턴**으로 띄엄띄엄 켭니다: 걸으며 1.5 s마다 내다보기, 손이 닿기 약 1 s 전 한 번, 기억의 2σ가 다음 행위의 허용 오차를 넘을 때만 응시(머리를 숙여 발 앞까지). 그 사이 이동과 행위는 기억으로 이어지고, 다시 볼 때 칼만 혼합으로 기억을 고치며, 지점 깃발을 보면 자기 위치를 고칩니다(잘못 읽은 깃발은 검증 게이트로 버리고, 기억도 본 때의 드리프트 몫만큼 함께 옮김). SDS `memory` 트랙이 다시 본 어긋남으로 드리프트 율을 배웁니다. ③ 가벼운 **택배 상자**(0.7 kg)가 맵 곳곳과 베이스캠프 앞에 놓이고, 앞에서 본 상자를 **지나가며 오른팔을 옆으로 뻗어** 보지 않고 기억으로 집습니다(어깨 외전). 원목·돌을 잡는 자세는 그대로이고, 앞에서 못 버티는 무거운 짐은 **발을 벌려 짐을 발 사이에 두고** 두 손으로, 몸의 가속(동적 ZMP)을 넣어 천천히 일어섭니다. ④ 걸음이 **LIPM을 넘어**: CoM이 사람만큼 오르내리고(Massaad 2007) 압력 중심이 발꿈치에서 발끝으로 굴러(Jeon & Cho 2020) 수직 반력의 두 봉우리와 골이 물리에서 나오며, 필요 마찰이 0.35 → 0.16(중급 다리)으로 사람 범위에 들어옵니다. ⑤ **`wheel_lift` 프리셋**(몸통 리프트 Fetch 급 0.40 m + 고급 허리 피치)으로 바퀴형도 바닥의 물건을 손으로 집습니다. 기획은 [`docs/PLAN-v0.7.md`](docs/PLAN-v0.7.md), 리서치 · 벤치마크 · 단계별 개선 기록은 [`docs/research/perception.md`](docs/research/perception.md)에 있습니다.

v0.2부터는 **오픈월드 자원 채취 테스트베드**가 있습니다. 마인크래프트식 격자 월드(나무·바위·석탄·철광석)에서 로봇이 "철광석 2개 캐서 창고로 가져와"라는 명령을 받아 퀘스트(탐색 → 접근 → 채취 → 줍기 → 운반 → 인도)로 수행합니다. 렌더한 프레임을 WebP로 압축해 SigLIP2로 인식하고, SDS와 Granite TTM으로 인식 결과를 걸러 내며, 모든 판단을 해시 체인 로그로 남깁니다. 설계 근거와 logis-center·endmill-sandbox와의 대조는 [`docs/PLAN.md`](docs/PLAN.md)에 있습니다.

### 구성

| 경로 | 역할 |
| --- | --- |
| `crates/hand-core` | 손 모델: 손가락 프리셋, 관절 옵션, 발(이동 베이스), 순기구학, 파지 시뮬레이션, 주행 모델, 부품 프리셋과 전신 밸런스(CoM·ZMP·관절 토크·N:N 결합), 관절 동역학(`dynamics`), 복합 N:N 시나리오(`nn`), 지면 물리(`surface`), 감속기 수명(`wear`), 환경 N:N 효과 벤치마크(`effects`), 전신 기구학(`kin`), 행위 시계열(`act`: 굽히기·잡기·놓기·밀기·걷기·주행·부위 포커스), 행위 N:N 벤치마크(`actbench`) |
| `crates/hand-ml` | candle 기반 SigLIP2(NaFlex) 임베딩, 사전 기반 제로샷 분류, 장치 선택·캐시·배치 정책 (CPU, CUDA, Metal, Vulkan, ROCm) |
| `crates/hand-data` | 합성 데이터 계층 정의, 도메인 무작위화, LanceDB 저장 |
| `crates/world-sim` | 오픈월드: 격자 지형(열린 맵·험지 맵), 지면 종류, 자원 노드, 아이템(양·조각), 웨이포인트 더미, 시야 탐색(`explore`), 소프트웨어 레이캐스트 프레임(RGB·ID·깊이), WebP 캡처 |
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

속도 벤치마크는 실행하지 않습니다. 작업 단위의 기준은 문서화된 참조 수치와 정합성 테스트입니다. 물리의 "벤치마크"는 기준 점수 시험입니다: `cargo test -p hand-core --test effects_bench -- --nocapture`가 환경 시나리오마다 셀 원값·기준·z·점수를 출력하고(`[FX]`), 엔진 파일이 채점 출처를 쓰지 않는지 검사합니다. 복합 N:N은 `--test nn_scenarios`, 행위(잡기·놓기·밀기·걷기·굽히기)는 `--test action_bench`(`[AX]` 셀, `[AX-SUM]` 시나리오, `[AX-WHY]` 실패 이유, `[AX-HELD]` 홀드아웃)입니다. 메모리가 8 GB 안팎인 PC에서는 링크가 겹쳐 가상 메모리가 모자랄 수 있으니(`os error 1455`) `CARGO_BUILD_JOBS=2`로 돌리세요.

Windows에서는 백엔드별 스크립트가 MSVC·CUDA·HIP SDK 환경을 맞추고 단계별 로그(`test_logs\<시각>_<백엔드>\`)를 남깁니다. `test_*.bat`에 인자를 주면 그 cargo 테스트 하나만 돌립니다.

```bash
test_cuda.bat                                   # 전체 단계 (단위, 프리셋→물리, 복합 N:N, 시나리오, 통합, GPU 대조, Lance)
test_cuda.bat -p hand-vision --test gpu_backend  # GPU와 CPU 결과 대조만
check_rocm.bat                                  # ROCm 빌드 확인 (HIP SDK 7.x 또는 ROCm Core 10.x 자동 선택)
dev_vulkan.bat                                  # Vulkan 으로 Tauri 앱 실행
serve_cuda.bat                                  # 개발 서버 (브라우저)
```

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

빌드에 GPU 기능이 있으면 앱은 첫 GPU(CUDA → ROCm → Vulkan → Metal)에서 시작합니다. `HAND_DEVICE=cpu|cuda|rocm|vulkan|metal|gpu[:번호]`로 고정할 수 있고, 왼쪽 ④ "사전 · 모델"에서 바꿀 수 있습니다. 같은 백엔드의 모델은 장치 하나를 공유하며, 배치 크기는 `HAND_BATCH_ROWS`, CUDA·ROCm 텍스트 고정 모양은 `HAND_FIXED_SHAPE=0|1`, SAM 인코더 배치는 `HAND_SAM_BATCH`로 바꿉니다.

Windows에서는 아래 `.bat`이 다음 설정을 대신합니다(`dev_cuda.bat`, `dev_rocm.bat`, `dev_vulkan.bat`, `dev_cpu.bat` 등). 직접 할 때는 이렇습니다. CUDA를 쓰려면 다음 두 가지가 필요합니다.

- MSVC 호스트 컴파일러(`cl.exe`)가 PATH에 있어야 합니다. Visual Studio 개발자 셸에서 실행하거나 `vcvars64.bat`을 먼저 실행하세요.
- CUDA 13의 CCCL 헤더는 MSVC의 표준 전처리기를 요구합니다. nvcc에 아래 플래그를 넘깁니다.

```bash
export NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"
```

GPU의 compute capability를 지정하면 커널 빌드가 한 아키텍처만 컴파일해 빨라집니다. 예: RTX 3050 Laptop은 8.6.

```bash
export CUDA_COMPUTE_CAP=86
```

ROCm과 Vulkan은 [Qntology/candle-rocm-vulkan](https://github.com/Qntology/candle-rocm-vulkan)의 candle 포크를 씁니다. 커밋 `2191c7ad`로 고정되어 있습니다(루트 `Cargo.toml`). ROCm 스크립트는 logis-center commerce와 같은 규칙으로 HIP SDK 7.x(legacy)와 ROCm Core SDK 10.x(core) 가운데 이 PC의 GPU를 지원하는 쪽을 고릅니다(`ROCM_VER`, `ROCM_PATH`, `CANDLE_ROCM_TRACK`, `ROCM_SEARCH_PATHS`).

SAM 마스크 모델은 `lmz/candle-sam`의 `mobile_sam-tiny-vitt.safetensors`(MobileSAM, 4 GB GPU 권장) 또는 `sam_vit_b_01ec64.safetensors`를 `models/sam`(또는 `SAM_DIR`)에 두고 ④의 **SAM 마스크 켜기**로 연결합니다. 없으면 영역 확장 추적기가 그대로 돕니다.

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
- GPU 경로는 CUDA만 RTX 3050 Laptop(4 GB, CUDA 13.3)에서 실행해 CPU 결과와 대조했습니다(SigLIP2 ≤ 2.2e-7, TIPS ≤ 3.3e-6, SAM 인코더·디코더 ≤ 7.7e-5 상대 오차). ROCm·Vulkan·Metal은 빌드 준비만 했고 실행 검증은 하지 않았습니다. 모델은 모든 백엔드에서 F32입니다.
- 한국어 명령은 어휘 사전 일치가 1차 경로입니다. 사전에 없는 표현의 임베딩 보강은 SigLIP2 텍스트 타워의 한국어 품질에 달려 있습니다.
- Tauri 창에서 직접 돌려 보지는 않았습니다. UI는 개발용 서버 위에서 브라우저로 확인했습니다. 로그 시각은 UTC입니다.
- TIPSv2 DPT 이식은 실제 체크포인트 헤더의 텐서 이름·형상(227개)을 전부 대조했지만, 가중치로 PyTorch와 수치를 비교하지는 않았습니다. SigLIP2 패치 히트맵(MAP 헤드 value 경로)이 NaFlex 체크포인트에서 잘 되는지도 가중치로 확인해야 합니다.
- 지면 물리의 일부 값(풀밭의 발·트랙 마찰, 젖은 모래의 발 마찰·보행 계수, 트랙 구름 저항, 메카넘 계수)은 추정이고, 바퀴의 미끄러짐률별 추력이 없어 모래 위 바퀴 견인을 실제보다 높게 봅니다. 감속기 수명은 데이터시트 정격(일부는 다른 계열 대입)과 3승 법칙이며 관절 속도 항은 1로 둡니다. 근거와 남은 차이는 `docs/research/effects.md` 7장.
- 밸런스 모델은 준정적 근사입니다(점 질량, 지지 다각형, 관절 토크). 자세 사이의 전이는 관절 동역학(`hand_core::dynamics`)이 시뮬레이션하지만 상체만 강체로 움직이고 다리·바퀴는 지면 반력의 정역학이며, 관절 마찰·회전자 관성·전류 루프 지연은 없습니다. 사람 기준값 중 "유도" 표시된 것은 계산으로 얻은 값입니다.
- 통합 테스트의 ML 수치(깊이 오차, 추적 IoU, 깊이 축척, 결정 횟수)는 대역 모델에서 나온 값이라 실제 모델 성능이 아닙니다.
- 행위 물리(v0.6)는 위에서 잡기 하나(v0.7에서 옆 뻗기·걸터서기 추가), 상자 모양 물체, 시작할 때 만든 시계열의 재생입니다. 쥠은 하중을 따라 조절되지 않습니다. 걸음(v0.7)의 CoM 높이는 걸음 주기의 코사인이고 양발 지지의 하중 이동은 선형입니다. 바퀴·트랙은 `wheel_lift`(몸통 리프트) 프리셋만 바닥의 물건을 집습니다. 사람·로봇 기준과 남은 차이는 `docs/research/actions.md` 4.1장과 `docs/research/perception.md` 6장.
- 기억(v0.7)은 믿음 좌표의 점과 σ입니다. 지도 전체를 함께 고치는 SLAM은 아니어서, 위치 보정 때 기억은 본 때까지 쌓인 드리프트 몫만큼만 옮깁니다.
