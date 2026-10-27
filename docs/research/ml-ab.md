# 실제 SigLIP2 · TIPSv2를 꽂으면 왜 대역보다 나빠지는가: 통제된 A/B, 원인, 고침

조사일 2026-10-27. 이 문서에서 온 코드: `crates/noto-runtime/tests/ml_ab.rs`(A/B 하네스), `crates/noto-vision/src/align.rs`(지면 정렬 · 접지 거리), `crates/noto-vision/src/calib.rs`(라벨별 보정), `crates/noto-vision/src/locate.rs`(`LocateScoring`, 프롬프트 템플릿, 배경 클래스), `crates/noto-vision/src/tips.rs`(읽어 내기 GELU, 빈 분포 폭), `crates/noto-ml/src/vision.rs`(위치 표 앤티에일리어스 크기 바꾸기), `crates/noto-search/src/vision.rs`(`VisionPrompting`), `crates/noto-runtime/src/ml.rs`(`MlFixCfg`, 정직한 대역), `perception.rs`(`ScanErrModel`), `embedding.rs`(타일 점수 보정), `mlsds.rs`·`precision.rs`(모드별 SDS 범위), `data/ml_calib.json`, `data/world_lexicon.json`의 `vision_prompting`.

표기는 [`actions.md`](actions.md)와 같다: **[직접]** 출처에 인쇄된 값, **[유도: 식]** 이 문서에서 계산, **[추정]** 출처를 찾지 못해 정한 값, **[측정]** 이 저장소의 하네스가 이 렌더러에서 잰 값(벤치마크 기준이 아니라 엔진 입력). **벤치마크는 시간 측정이 아니라 기준 점수 시험**이다.

## 0. 요청과 반영

| 요청 (R7) | 반영 |
|---|---|
| SigLIP2와 TIPSv2 깊이 모델을 꽂으면 대역보다 덜 정확하다. 있을 때/없을 때를 비교하라 | 같은 키프레임(런타임의 캡처 코드, 같은 WebP q70 바이트)을 대역과 실제 모델에 한 요인씩 넣는 통제 A/B 하네스 `tests/ml_ab.rs` (§1) |
| 왜인지 찾아라 | 지도(map)의 가설 원인 11개를 한 요인씩 떼어 쟀다. 확인·반박을 숫자로 §3에 |
| 개선하라 | 측정한 영향 순서로 고치고 다시 쟀다(§4). 고침은 모두 설정(`MlFixCfg`) 뒤에 있어 A/B로 옛 경로와 비교된다. 대역이 실제 문제를 가리지 않도록 측정된 오차 모델을 가진 "정직한 대역"을 더했다(§2.4) |

## 1. A/B 설계

### 1.1 장면

시드 세계에서 노드와 떨어진 물건을 모두 치우고, 로봇 앞에 목표 하나(9 라벨: 나무 · 바위 · 석탄 광맥 · 철광맥 · 원목 · 돌 · 석탄 · 철광석 · 택배)와 그 닮은꼴(`perception::look_alike`) 하나를 둔다. 머리 카메라(384 × 256, 70°)로 거리 d와 고개 숙임(기본 −12°에 더해 0° 또는 30°)에서 찍는다. 목표가 없는 장면(닮은꼴만, 또는 아무것도 없음)도 섞는다. 지형은 평지(Open)와 험지(Rugged), 몸은 시드 1–2가 사람 발, 3–4가 바퀴다. **홀수 시드는 맞춤(보정 · 정직한 대역을 맞춤), 짝수 시드는 평가**다.

키프레임은 런타임 코드 그대로 만든다: `World::render` → WebP q70 → 복호(모델 입력), `KeyTruth::capture`(반 해상도 정답, 대역과 채점에만), `record::intrinsics`/`odometry`(IMU 추정 피치, 눈 높이). 그래서 모든 팔(arm)이 같은 바이트를 본다.

### 1.2 팔

| 팔 | 깊이 | 의미 위치 | 마스크 | 포트 |
|---|---|---|---|---|
| A0 | 대역 | 대역 | 영역 확장 | — |
| A1 | TIPS | 대역 | 영역 확장 | v0.9 |
| A2 | 대역 | SigLIP2 dense | 영역 확장 | v0.9 |
| A3 | 대역 | 대역 | MobileSAM | — |
| A4 | TIPS | SigLIP2 dense | MobileSAM | v0.9 |
| A4new | TIPS | SigLIP2 dense (v1.0 포트, 고침 없음) | MobileSAM | v1.0 |
| A5 | TIPS + 지면 정렬 + 접지 거리 | SigLIP2 + 보정 게이트 | MobileSAM | v1.0 |
| A5region | A5와 같음 | A5와 같음 | 영역 확장 | v1.0 |
| A0h / A0hsam | 정직한 대역 | 정직한 대역 | 영역 확장 / MobileSAM | — |

"포트 v0.9"는 v1.0 전의 이식(SigLIP2 위치 표 단순 쌍선형, TIPS 읽어 내기 정확 GELU)이고 "v1.0"은 고친 이식이다. 타일 스캔(게임 패스의 인식기)은 같은 프레임에서 오라클(A0, `NoiseCfg::demo`), SigLIP2 원점수(A4 · A5raw), 템플릿 + 배경(A4tpl), 보정 점수(A5cal), 정직한 오라클(A0h)로 따로 잰다.

### 1.3 지표

- 깊이: 정답 10 m 이하 화소(NYU 상한과 같은 범위)의 AbsRel · δ1 · RMSE · 덮음, 정답 거리 띠별(0–2, 2–4, 4–6, 6–10, >10 m) AbsRel, 목표 화소의 AbsRel. 같은 지도를 원값 · 중앙값 축척 · 정답 최소제곱(시차 공간 축척 + 이동, 하네스만) · 지면 정렬(평면 / 지도 DEM / 축척만)으로.
- 의미 위치: 라벨이 보이는지(≥ 16 px)를 피크 점수와 존재 확률이 가려내는 AUROC · AP, 피크가 참 상자(한 패치 넓힘) 안에 드는 비율, 위치 오차(px), 보정 게이트의 정밀도 · 재현율 · ECE.
- 추적과 거리: 목표를 찾았는지, 맞는 물체인지, 정답과의 IoU, 목표가 없는데 추적이 생겼는지(헛것), 거리 상대 오차와 부호, 정렬 뒤 거리의 z(오차 / 1σ).
- 타일 스캔: 게이트(점수 0.30 · 여유 0.04) 통과 후보의 정밀도, 보이는 목표의 재현율(1 m 안), 위치 오차, 스캔당 틀린 통과 후보.

### 1.4 실행

```
set SIGLIP2_DIR=...\models\siglip2-base-patch16-naflex
set TIPS_DIR=...\models\tipsv2-b14-dpt
set SAM_DIR=...\models\sam
set RUNTIME_DIR=...\runtime            (bank: 문구 벡터 캐시)
set MLAB_OUT=...\mlab                  (맞춘 보정 ml_calib.json, 프레임별 기록 mlab_frames.jsonl)
set MLAB_SEEDS=1,2,3,4   MLAB_DISTS=2,4,7   MLAB_TILTS=0,30
test_cuda.bat -p noto-runtime --test ml_ab -- --ignored --nocapture ml_ab_benchmark
```

`data/ml_calib.json`에 보정이 실려 있으면 처음부터 그것을 쓰고(맞춤 시드는 정직한 대역만 맞춤), 비었거나 `MLAB_REFIT=1`이면 맞춤 시드에서 다시 맞춘다. 닫힌 고리(퀘스트 전체)는 `ml_ab_quests`(`MLAB_QUESTS`, `MLAB_QUEST_SEEDS`), 이식 대조 입력은 `ml_port_dump`(§6 원인 10).

## 2. 엔진 입력 (그룹 A)

### 2.1 깊이 정렬 (`noto_vision::align`)

| 값 | 쓰는 곳 | 출처 |
|---|---|---|
| 상대 깊이를 맞추는 공간: 역깊이(시차)에서 이미지마다 축척 a와 이동 b를 최소제곱 (`1/Z = a/Z_pred + b`) | `AlignSpace::InverseAffine` | Ranftl, Lasinger, Hafner, Schindler, Koltun, "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer", IEEE TPAMI 2022 (arXiv 1907.01341), 식 2–4 **[직접]** https://arxiv.org/abs/1907.01341 |
| 잘라낸 재맞춤에서 남기는 잔차 비율 0.8 | `AlignCfg::trim_keep` | 같은 논문 (ReDWeb에서 경험적으로) **[직접]** |
| 축척 하나만 (`Z = s·Z_pred`, 중앙값 축척과 같은 자리) | `AlignSpace::Scale` (비교 팔) | Godard, Mac Aodha, Firman, Brostow, "Digging Into Self-Supervised Monocular Depth Estimation" (Monodepth2), ICCV 2019, `evaluate_depth.py` **[직접]** https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py |
| RANSAC 시도 k = log(1−p)/log(1−wⁿ) = 17 (p 0.99, w 0.5, n 2) | `AlignCfg::trials` | Fischler & Bolles, CACM 24(6), 1981 **[직접: 식]**, k는 **[유도]**, w = 0.5는 **[추정]** https://doi.org/10.1145/358669.358692 |
| 지면 깊이: 높이 h, 아래로 θ 기운 핀홀에서 `Z = h/(ȳ·cos θ + sin θ)`, 지도가 있으면 지형 높이 위로 광선 진행 | `align::ground_depth` | 표준 투영 기하 (Hartley & Zisserman, *Multiple View Geometry*, 2판, 2004); 일정한 카메라 높이의 평면 사전은 Tian 외, "Accurate and Robust Scale Recovery for Monocular Visual Odometry Based on Plane Geometry", ICRA 2021 (arXiv 2101.05995). 식은 **[유도]** |
| 접지 거리의 1σ: `dr/dθ = (h² + r²)/h`에 피치 σ와 한 화소 행의 각을 더함 | `align::contact_range` | 같은 기하 **[유도]** |
| 지면 깊이 상한 9 m, 상한의 0.95 이상은 "멀거나 모름" | `AlignCfg::max_ground_m`, `saturate_frac` | 연구 팩의 추정 ("~9 m trusted range", "~0.95·10 m") **[추정]** |
| 내층 상대 잔차 0.10, 최소 내층 비율 0.30, 가장 가까운 지면 0.5 m | `AlignCfg::inlier_tol`, `min_inlier_share`, `min_ground_m` | 연구 팩 τ ≈ 0.05–0.10 **[추정]** |
| 피치를 아는 정도 1σ 0.5° | `MlFixCfg::pitch_sigma_deg` | 관절 사슬 + 머리 IMU의 흔들림 뒤 **[추정]** |
| 정렬 깊이의 상대 1σ: 지면 잔차 중앙값, 최소 5 % | `MlPipeline::analyse_with` | **[추정]** (지면이 보여 주지 못하는 물체 쪽 오차) |

### 2.2 모델 이식 (`noto-vision::tips`, `noto-ml::vision`)

| 값 | 쓰는 곳 | 출처 |
|---|---|---|
| TIPSv2 DPT 깊이 머리: 빈 256개, 0.001–10 m, 미터 출력 | `TipsConfig`, `TIPS_MAX_DEPTH_M` | Hugging Face Transformers 문서 "TIPSv2 DPT" (`Tipsv2DptConfig`) **[직접]** https://huggingface.co/docs/transformers/main/model_doc/tipsv2_dpt |
| 깊이 머리는 NYU Depth V2(실내)로 학습, 입력 [0, 1] | 바깥 블록 세계에서 축척이 어긋나는 이유 | Google, 모델 카드 `google/tipsv2-b14-dpt` **[직접]** https://huggingface.co/google/tipsv2-b14-dpt |
| 읽어 내기 투영의 GELU는 tanh 근사 (`readout_act: "gelu_pytorch_tanh"`) | `TipsConfig::readout_gelu_tanh` | 체크포인트의 `config.json` **[직접]** |
| 백본 패치 14 px, 기본 448 × 448 | 정사각 448 비교 팔 | Hugging Face 문서 "TIPSv2" **[직접]** https://huggingface.co/docs/transformers/main/model_doc/tipsv2 |
| 위치 표 크기 바꾸기: `F.interpolate(mode="bilinear", align_corners=False, antialias=True)` | `VisionTower::set_position_antialias` (기본 켬) | Hugging Face `Siglip2VisionEmbeddings.resize_positional_embeddings` (코드) **[직접]** |
| SigLIP 쌍 확률 σ(t·cos + b), t · b는 체크포인트의 학습값 (이 체크포인트: exp(t) 115.3, b −16.78 → 확률 0.5인 코사인 0.146) | `Siglip2::logit_scale_bias`, 존재 확률 | Zhai, Mustafa, Kolesnikov, Beyer, "Sigmoid Loss for Language Image Pre-Training", ICCV 2023 **[직접]** https://arxiv.org/abs/2303.15343; 값은 체크포인트 **[측정]** |
| SigLIP 2 텍스트: 64 토큰, Gemma 토크나이저, 소문자 | 토크나이저 대조 (§3 원인 10) | Tschannen 외, "SigLIP 2", arXiv 2502.14786 **[직접]** https://arxiv.org/abs/2502.14786 |
| 값(value) 경로 패치 벡터 (MaskCLIP) | `MapHead::dense` (v0.3부터) | Zhou, Loy, Dai, "Extract Free Dense Labels from CLIP", ECCV 2022 **[직접]** https://arxiv.org/abs/2112.01071 |

### 2.3 의미 위치와 보정

| 값 | 쓰는 곳 | 출처 |
|---|---|---|
| 프롬프트 앙상블: 템플릿 문장마다 텍스트 벡터를 만들어 문구별로 평균. 템플릿 'a photo of {}.', 'a rendering of {}.', '{} in a video game.', 'a low resolution photo of {}.', 'a cropped photo of {}.', 'a bad photo of {}.' (관사는 문구에 있음) | `vision_prompting.templates`, `LocateCfg::templates`, `VisionBank::build_with` | OpenAI CLIP 저장소 `notebooks/Prompt_Engineering_for_ImageNet.ipynb` 80개 목록 **[직접]** https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb ('a 3D render of'는 목록에 없어 쓰지 않음) |
| 열린 집합이 선택을 강요하지 않게 배경 클래스(풀 · 흙 · 자갈 · 하늘 · 물 · 바위벽 · 집)를 더한 패치별 소프트맥스, 온도는 체크포인트의 exp(t) | `LocateScoring::Contrast`, `vision_prompting.background` | 연구 팩의 프롬프트 앙상블 제로샷 분류기 (CLIP 저장소 분류기 + SigLIP) **[직접: 방법]**, 배경 문구 목록 **[추정]** |
| 보정: 검증 세트에서 NLL 최소화, ECE 15칸; 시그모이드는 Platt 방식 (같은 논문 §4.1이 소개) | `calib::platt_fit`, `ece`, `ECE_BINS` | Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017 **[직접]** https://arxiv.org/abs/1706.04599 |
| 라벨별 임계값: 맞춤 시드에서 정밀도 0.9, 없으면 거짓 양성률 0.05 | `ml_ab.rs` `CALIB_PRECISION`, `CALIB_FPR` → `data/ml_calib.json` | 공학적 선택 **[추정]**; 임계값 자체는 **[측정]** |
| 타일 점수 → "후보가 1 m 안의 같은 라벨 물체일 확률" Platt 맞춤 (라벨별, 적으면 합친 `*`) | `EmbeddingPerception::set_calibration`, `data/ml_calib.json` `embedding` | **[측정]** (Guo 외 2017의 원리) |

### 2.4 정직한 대역 (측정된 오차 모델)

| 대역 | 오차 모델 | 값 |
|---|---|---|
| 깊이 `DepthErrModel::TIPS_GROUND` | `ln(예측/참) = a + b·ln(참) + η(키프레임) + ε(패치)`, 하늘 = 모름, 참 거리 `unknown_beyond_m` 너머 = 모름, 3 px 상자 흐림(DPT의 매끈함, **[추정]**) | v1.0 경로(지면 정렬 뒤)의 TIPS를 맞춤 시드에서 **[측정]** (§4) |
| 의미 위치 `LocateErrModel::SIGLIP_MEASURED` | 피크가 물체에 떨어질 확률 σ(a·ln 화소 + b); 맞을 때 · 빗나갈 때 · 라벨이 없을 때의 보정 확률 점수(평균, SD); 게이트 확률 `gate_p` | 보정 게이트를 거친 SigLIP2 dense를 맞춤 시드에서 **[측정]** |
| 타일 스캔 `ScanErrModel::SIGLIP_MEASURED` | 보이는 물체가 맞는 라벨 · 1 m 안으로 게이트를 통과할 확률 σ(a·ln 화소 + b); 스캔당 틀린 통과 후보(포아송); 맞는 후보의 위치 오차; 통과 점수 · 여유 | v1.0 런타임의 타일 스캔(보정 점수)을 맞춤 시드에서 **[측정]** |

`Runtime::use_honest_stand_ins()`(또는 `MlPipeline::use_honest_stand_ins`)로 켠다. 기본은 v0.9의 대역 그대로라 기존 벤치마크 줄은 바뀌지 않는다. 모드 이름이 `stand-in(tipsv2-measured)`, `stand-in(siglip2-measured)`, `oracle(siglip2-measured)`라서 SDS 통계가 섞이지 않는다.

## 3. 벤치마크 출처 (그룹 B) — 채점만

엔진은 이 출처를 쓰지 않는다(`the_engine_never_names_an_ml_benchmark_source`).

| 칸 | 기준 | 규칙 | 출처 |
|---|---|---|---|
| M1 정답 정렬 뒤 깊이 오차 | 렌더 영화(Sintel)에서 이미지별 아핀 정렬 뒤 AbsRel: DPT-Large 0.270, Depth Anything ViT-L 0.458, MiDaS v3.1 0.587 → 0.438 ± 0.160 | `100 − 10·|z|` | Ranftl, Bochkovskiy, Koltun, "Vision Transformers for Dense Prediction", ICCV 2021 표 1 https://arxiv.org/abs/2103.13413; Yang 외, "Depth Anything", CVPR 2024 표 2 https://arxiv.org/abs/2401.10891 |
| M2 카메라 높이 축척 / 정답 축척 | KITTI AbsRel 0.118 / 0.113 = 1.04 | 이보다 나쁘지 않게, 0.1마다 −10 | Xue 외, "Toward Hierarchical Self-Supervised Monocular Absolute Depth Estimation for Autonomous Driving Applications" (DNet), IROS 2020 https://arxiv.org/abs/2004.05560 |
| M3 학습 영역 밖 미터 깊이 | NYU 미세조정 머리를 Hypersim에: AbsRel 0.363 · δ1 0.361 | 기록 | Yang 외, Depth Anything, CVPR 2024 |
| M4 패치-텍스트 위치 | 학습 없는 패치 원시 14.1 대 MaskCLIP 값 경로 30.3 mIoU (8개 평균); SigLIP2 SO/14 VOC21 26.8 대 TIPSv2 L/14 44.4 | 기록 | Wang, Mei, Yuille, "SCLIP", ECCV 2024 https://arxiv.org/abs/2312.01597; Cao 외, "TIPSv2", CVPR 2026 (arXiv 2604.12012) 표 5 |
| 프롬프트 앙상블의 이득 | 단일 템플릿보다 약 +3.5점 | 기록 | Radford 외, "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021 https://arxiv.org/abs/2103.00020 |
| 제로샷 보정 | ECE 6.3–6.7% → 온도 보정 뒤 1.4–2.2% | 기록 | LeVine 외, "Enabling Calibration in the Zero-Shot Inference of Large Vision-Language Models", arXiv 2303.12748 https://arxiv.org/abs/2303.12748 |

## 4. 모델 (v1.0 고침)

### 4.1 지면 정렬 (`align::ground_align`)

로봇은 깊이 모델이 미터를 알 필요가 없다: 자기 카메라를 안다. 눈 높이 h(`Pose::eye_m`)와 IMU가 추정한 피치 θ(`Intrinsics::pitch_deg`)가 지평선 아래 모든 광선이 땅에 닿는 깊이를 정한다. 평지에서 `Z_g = h/(ȳ·cos θ + sin θ)`(정규화 행 ȳ), 로봇의 지도가 있으면 지형 높이(`World::surface_m`, 카메라 아래 땅 기준) 위로 0.1 m씩 광선을 진행한 뒤 이분한다. 이 기대 깊이가 모델 없이 얻는 미터 기준이다.

1. 후보: 4 화소마다, 모델 답이 상한의 95 % 미만이고 기대 지면 깊이가 0.5–9 m인 화소. (모델 답 p = 1/Z_pred, 목표 q = 1/Z_g.)
2. RANSAC 17회: 두 화소로 `q = a·p + b`를 세우고(기대 깊이가 5 % 넘게 다른 쌍만), 상대 잔차 < 0.10인 내층을 센다.
3. 가장 큰 합의를 최소제곱으로 다시 맞추고, 잔차가 작은 80 %만 남겨 한 번 더(MiDaS의 잘라내기).
4. 내층 ≥ 40개이고 후보의 30 % 이상이면 받아들여 지도 전체에 적용한다(`1/Z = a/Z_pred + b`). 땅에 선 물체(나무, 바위, 목표 자신)는 뒤의 땅보다 가까워서 외층으로 빠진다.
5. 상한에 닿은 화소는 맞추든 못 맞추든 "멀거나 모름"(`INFINITY`)이 되어 마스크 중앙값과 정렬을 더럽히지 않는다.

모드 이름이 `tipsv2-b14-dpt+ground`라서 SDS의 학습 축척(삼각 측량 / 깊이)이 원래 모델의 것과 섞이지 않는다. 정렬 뒤의 축척은 1 근처여야 하고, 진짜 모델은 자기 모드에서 배운 축척만 쓴다(`Runtime::depth_scale`).

### 4.2 접지 거리 (`align::contact_range`)

땅에 놓인 물건(드롭: 원목 · 돌 · 석탄 · 철광석 · 택배)은 마스크의 가장 아래 행이 땅에 닿는 곳이다. 그 행의 광선이 땅에 닿는 거리가 깊이 모델 없이 얻는 거리다. 1σ는 `(h² + r²)/h · √(σ_θ² + (행 하나의 각)²)`: 1.14 m 눈에서 3 m, 피치 σ 0.5°면 약 8 cm. 마스크가 화면 아래에 잘리면 접지가 안 보이므로 쓰지 않는다. 정렬 깊이(1σ = 지면 잔차, 최소 5 %)와 역분산으로 섞는다.

### 4.3 의미 위치 점수 (`LocateScoring`)

- `PairProb` (v0.9): 패치마다 SigLIP 쌍 확률 `σ(t·cos + b)`에서 닮은꼴 코사인의 절반을 뺌. 이 렌더러에서 패치 코사인은 확률 0.5의 코사인(0.146)보다 훨씬 작아 모든 값이 0 근처이고, 라벨이 없어도 늘 점이 하나 나온다.
- `Contrast`: 패치마다 라벨들과 배경 클래스들 위의 소프트맥스 `softmax(t·cos)`. 풀 패치는 "덜 다른" 라벨이 아니라 풀 클래스로 간다.
- `ZScore`: 라벨마다 이미지 안 패치 코사인을 표준화(이 프레임에서 어디가 가장 튀는가).
- 템플릿: 문구마다 템플릿 6개로 텍스트 벡터를 만들어 평균(`LocateCfg::templates`, 타일 스캔은 `VisionBank::build_with`). 배경 집합에 있는 문구는 타일 스캔에서 prejudice로 빼지 않는다(풀밭은 거의 모든 타일 뒤에 있다).
- 존재 확률은 늘 SigLIP의 풀링 쌍 확률이다.

어느 점수 · 어느 통계(피크, 존재, 피크 × 존재)를 게이트에 쓸지는 하네스가 맞춤 시드의 AP로 고르고, 그 표의 모드(`siglip2-dense+bg` 등)가 런타임의 `MlFixCfg::v1`을 정한다(`LocateCfg::from_mode`).

### 4.4 보정 게이트 (`calib`, `MlPipeline::passes_calibration`)

v0.9는 실제 위치기의 프롬프트 하한이 0이었다(쌍 확률이 작아서). 그래서 라벨이 없어도 피크가 추적기 프롬프트가 되어 닮은꼴에 헛 추적을 만들었다. v1.0은 맞춤 시드에서 라벨별 임계값(정밀도 0.9, 없으면 FPR 0.05)과 Platt 확률을 맞춰 `data/ml_calib.json`에 싣고, 실제 위치기의 점은 그 임계값을 넘을 때만 프롬프트가 된다. 대역은 이 표로 막지 않는다. SDS의 위치 기준선도 모드별로만 읽어서(`mlsds`) 대역의 0..1 점수가 실제 모델의 하한을 정하지 않는다.

타일 스캔은 라벨별 Platt 맞춤으로 점수를 "1 m 안 같은 라벨 물체일 확률"로 바꾼다(`EmbeddingPerception::set_calibration`, 범위 `embedding+cal`). 그러면 게이트의 `min_score` 0.30이 어느 인식기에서나 같은 정밀도를 뜻한다. 장소(창고 · 베이스캠프)는 맞춤에 없어서 원점수 그대로다.

### 4.5 정직한 대역

v0.9 대역은 정답에 고른 잡음을 넣은 것이었다: 깊이 축척 0.8 하나와 6 % 잡음(정답 경계 그대로), 의미 열지도는 정답 화소 비율(부분 패치 물체도 빛남), 타일 스캔은 6 px 이상이면 보고 신뢰도는 지어낸 식(0.55 + 0.30·√px/12). 실제 모델의 오차를 잰 적이 없어서 대역으로 맞춘 결정이 실제 모델로 옮겨지지 않았다. 정직한 대역은 §2.4의 오차 모델을 하네스가 잰 값으로 가진다. 켜면 같은 결정 코드가 실제 모델과 같은 종류의 실패(먼 곳의 "모름", 빗나간 피크, 틀린 통과 후보)를 만난다.

RESULTS_PLACEHOLDER
