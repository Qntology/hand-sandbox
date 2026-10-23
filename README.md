# hand-sandbox

## 물류 로봇 손 시뮬레이터 (Rust + Tauri)

손가락 수(2–5)와 관절 옵션, 발(바퀴·메카넘·트랙·사람 발)을 바꿔 보며 파지와 주행을 시뮬레이션합니다. SigLIP2로 물류 물체를 분류하고, 합성 데이터를 LanceDB에 기록합니다.

### 구성

| 경로 | 역할 |
| --- | --- |
| `crates/hand-core` | 손 모델: 손가락 프리셋, 관절 옵션, 발(이동 베이스), 순기구학, 파지 시뮬레이션, 주행 모델 |
| `crates/hand-ml` | candle 기반 SigLIP2(NaFlex) 임베딩, 사전 기반 제로샷 분류, 장치 선택 (CPU, CUDA, Metal, Vulkan, ROCm) |
| `crates/hand-data` | 합성 데이터 계층 정의, 도메인 무작위화, LanceDB 저장 |
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

### SigLIP2 가중치

가중치는 저장소에 넣지 않습니다. 아래 위치에 받아 두면 앱이 자동으로 읽습니다. 다른 위치는 `SIGLIP2_DIR`로 지정합니다.

```bash
huggingface-cli download google/siglip2-base-patch16-naflex --local-dir models/siglip2-base-patch16-naflex
```

필요한 파일은 `config.json`, `tokenizer.json`, `model.safetensors`입니다.

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
