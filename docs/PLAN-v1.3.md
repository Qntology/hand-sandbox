# noto-sandbox v1.3 — 네 비전 모델의 1 : 2 : 4 다층 피라미드 정렬

작성일 2026-10-29. v1.2 위에 더한다. 표기는 [`research/actions.md`](research/actions.md)와 같다: **[직접]** 출처에 인쇄된 값, **[유도]** 계산, **[추정]** 출처를 찾지 못해 정한 값, **[측정]** 이 저장소에서 잰 값.

## 0. 요청과 반영

| 요청(개선안) | 반영 |
|---|---|
| 타일(L0) · SigLIP2(L1) · TIPSv2(L2) · SAM(L3)이 각자 다른 격자를 써서 `fusion.rs`에서 비정수배로 어긋난다. 모든 계층을 기본 격자의 1× : 2× : 4× 정수배로 맞춘다 | **`PyramidCfg`** 하나로 네 계층의 격자를 정한다. SigLIP2는 짧은 변 12 패치(TIPSv2 24의 절반), TIPSv2는 정확히 그 2배, SAM 마스크는 SigLIP2 패치마다 16 × 16 화소. 384 × 256 키프레임은 **18 × 12 : 36 × 24 : 288 × 192**, 640 × 480은 **16 × 12 : 32 × 24 : 256 × 192**(SAM 임베딩 64 × 48의 정확히 4배) (§1) |
| Phase 1: `naflex_grid`에 정렬 옵션, `InputGrid`에 피라미드, `sample_mask`를 L1의 정수배로 | `noto_ml::preprocess::pyramid_grid` · `DenseGrid`, `Siglip2::embed_dense_batch_grid`, `InputGrid::Pyramid`, `sam::sample_mask_to` · `SamCfg::mask_size` (§1) |
| Phase 2: L2 → L1 하늘 · 원거리 마스킹, L1 → L2 피크 스내핑, `heat_in_mask` 블록 적분 | `pyramid::cell_geometry` · `gate_points`(하늘 · 18 m 너머 패치의 히트를 지우고 점을 다시 찾음), `pyramid::snap_points`(피크가 걸친 2 × 2 깊이 패치 중 앞쪽 표면으로), `fusion::cell_coverage`(히트 칸마다 마스크 덮임을 면적으로 적분) (§2) |
| Phase 3: L0 타일을 L1 밀집 벡터로 합성, 유효 프롬프트가 없으면 SAM 인코더를 건너뜀 | SAM: 프롬프트도 살아 있는 추적도 없으면 `update`가 인코더를 부르지 않고, 미리 읽기(`prefetch`)도 게이트를 지난 프롬프트가 있는 프레임만 묶어 인코딩. 타일: `EmbeddingCfg::dense`(기본 끔)로 거친 단계 타일을 한 번의 밀집 패스에서 풀링하고 가장 고운 단계만 잘라 인코딩 (§3) |
| 기존 가중치 · `CalibTable` 형식을 깨지 않고 점진 적용 | 가중치 · 형식은 그대로. 전환은 `MlFixCfg::pyramid`(v1 기본 켬, `legacy()`와 예전 JSON은 끔). 피라미드 격자의 SigLIP2 위치 모드는 `siglip2-dense+bg+pyr`처럼 따로 이름이 붙어, 다른 격자에서 잰 보정표를 쓰지 않는다 (§4) |

실제 모델(가중치)로 재는 시험은 요청자가 한다. 이 문서의 숫자는 격자 계산 **[유도]**과 대역 · 합성 장면의 단위 시험 **[측정]**뿐이다(§5).

## 1. 격자 정렬 (Phase 1)

### 1.1 규칙

| 계층 | 모델 스펙 | 피라미드 격자 | 근거 |
|---|---|---|---|
| L1 의미 | SigLIP2 base patch16 NaFlex: 16 px 패치, 패치 예산 256(`num_patches`), 위치 테이블 16 × 16을 원하는 격자로 안티앨리어스 보간 **[직접]** | 짧은 변 `l1_short_side` = 12 패치, 긴 변 = round(12 × 화면비), 예산 256을 넘으면 짧은 변을 1씩 줄임 | TIPSv2가 자기 격자를 고르는 규칙(`ShortSide`, 반올림)과 같은 규칙의 절반이라 L2가 항상 정확히 2배 |
| L2 기하 | TIPSv2-B/14 DPT: 14 px 패치, 위치 테이블 32 × 32 보간, 키프레임 기본 `ShortSide(24)` **[직접]** | L1 × `l2_ratio`(2). `PyramidCfg::fast()`는 1(토큰 ¼, 스내핑 없음) | |
| L3 경계 | SAM: 긴 변을 1024로 늘려 패딩, 임베딩 64 × 64, 저해상 로짓 256 × 256 **[직접]** | 마스크 = L1 × `mask_per_cell`(16), 화면 화소보다 곱지 않게 | 4:3 화면에서 256 × 192 = 로짓 격자 그대로 |
| L0 타일 | 타일 격자 (2, 2) · (4, 3) · (6, 4), 겹침 15 % | 밀집 타일을 켤 때만 각 격자 수를 L1 패치 수의 가장 가까운 약수로: 18 × 12에서 (2, 2) · (3, 3) · (6, 4) | 겹침 전 타일 = L1 패치의 정수 블록 |

### 1.2 화면별 격자 **[유도]**

| 화면 | SigLIP2 예전 → 피라미드 | 토큰 | TIPSv2 | SAM 마스크 예전 → 피라미드 |
|---|---|---|---|---|
| 384 × 256 (머리 카메라 키프레임) | 19 × 13 → **18 × 12** | 247 → 216 (−12.6 %) | 36 × 24 (예전과 같음) | 192 × 128 → **288 × 192** (화소당 2 → 1.33 px) |
| 640 × 480 | 18 × 14 → **16 × 12** | 252 → 192 (−23.8 %) | 32 × 24 (예전과 같음) | 320 × 240 → **256 × 192** (2 → 2.5 px) |
| 정사각형 | 16 × 16 → **12 × 12** | 256 → 144 | 24 × 24 (예전과 같음) | 화면 ÷ 2 → 192 × 192 |

TIPSv2는 키프레임에서 예전 격자 그대로다(같은 입력이면 같은 깊이). 바뀌는 것은 SigLIP2 격자와 SAM 마스크 해상도뿐이다.

### 1.3 개선안과 다르게 한 것

- **640 × 480의 L1**: 개선안은 "`naflex_grid`에서 패치 수를 짝수로 내림 양자화"하면 16 × 12가 나온다고 했지만, 짝수만 허용한 NaFlex 이진 탐색은 **18 × 14**(252, 둘 다 짝수)를 돌려준다 **[유도]**. 18 × 14의 2배(36 × 28)는 TIPSv2의 32 × 24가 아니다. 그래서 짝수 제약이 아니라 "TIPSv2 규칙의 절반"으로 L1을 정했다. 384 × 256 · 640 × 480 모두 개선안 표의 값과 같다.
- **정사각형**: 개선안 표는 16 × 16 : 32 × 32였지만, 그러면 TIPSv2 토큰이 576 → 1024로 는다. 피라미드는 TIPSv2 격자를 늘리지 않도록 12 × 12 : 24 × 24로 둔다. 16 × 16이 필요하면 `l1_short_side: 16`.
- **L3**: 개선안은 패치당 8 × 8 또는 16 × 16. 8 × 8이면 384 × 256 마스크가 144 × 96으로 예전(192 × 128)보다 거칠어져 16 × 16을 기본으로 했다. 640 × 480에서는 256 × 192로 예전 320 × 240보다 거칠지만, SAM 저해상 로짓(256 × 192 유효)과 일대일이라 더 곱게 뽑아도 정보가 늘지 않는다 **[유도]**.
- **L0 (4, 3)**: 개선안 표의 3 × 2 대신 "가장 가까운 약수" 규칙으로 3 × 3(18 × 12에서 패치 6 × 4 블록).

### 1.4 코드

| 파일 | 바뀐 것 |
|---|---|
| `crates/noto-ml/src/preprocess.rs` | `pyramid_grid`, `PYRAMID_SHORT_SIDE`, `DenseGrid { Naflex, Pyramid }`, `preprocess_grid`(격자 지정 전처리) |
| `crates/noto-ml/src/model.rs` | `embed_dense_batch_grid` (`embed_dense_batch`는 NaFlex 그대로) |
| `crates/noto-vision/src/pyramid.rs` (새 파일) | `PyramidCfg`, `Pyramid`(L1 · L2 · 마스크 격자, 칸 · 서브패치 사각형), Phase 2 함수들, `PyramidReport` |
| `crates/noto-vision/src/tips.rs` | `InputGrid::Pyramid`, `TipsDepth::pyramid_grid`(설정된 동안 `input_grid`보다 우선), `grid_for` |
| `crates/noto-vision/src/locate.rs` | `LocateCfg::grid`, 모드 접미사 `+pyr`, `SiglipLocator::effective_cfg` |
| `crates/noto-vision/src/sam.rs` | `sample_mask_to`, `SamCfg::pyramid` · `mask_size`, 추적의 프레임 화소 환산을 축별로(`Live::sx · sy`) |
| `crates/noto-vision/src/{depth,locate,track}.rs` | 트레이트 훅 `set_pyramid`(기본 무동작) — `DepthModel` · `SemanticLocator` · `MaskTracker` |

## 2. 계층 간 교차 검증 (Phase 2)

### 2.1 L2 → L1 하늘 · 원거리 게이트 (`gate_points`)

L1 패치마다 그 안의 L2 패치(2 × 2)에서 하늘 비율 · 18 m 안의 깊이 비율 · 18 m 너머 비율을 재 평균한다(`cell_geometry`). 하늘이 60 %를 넘거나, 깊이가 잰 내용 중 18 m 안이 25 %가 못 되면 그 패치의 히트를 히트의 바닥값(확률 점수는 0)으로 내리고, 점(피크 · 상자 · 마진)을 다시 찾는다. 하늘에만 히트가 있던 라벨은 점이 없어진다. 문턱은 타일 스캔의 `RoiCfg`와 같다 **[추정]**(v1.1).

- 하늘은 TIPSv2 분할 헤드(ADE20K `sky`)가 있으면 그것으로 센다. 없으면 깊이가 없는 화소 중 **카메라 광선이 수평선 위를 향하는 것만** 하늘로 센다. 지면 정렬은 TIPSv2의 10 m 포화를 무한대로 바꾸므로, 수평선 아래의 무한대(먼 땅)를 하늘로 세면 10 m 너머 목표를 SigLIP2가 찾지 못하게 된다.

### 2.2 L1 → L2 피크 스내핑 (`snap_points`)

피크가 든 L1 패치 안의 L2 패치 넷마다 깊이 중앙값 · 사분위 폭 · 유효 비율과, 그 서브패치 가운데에서의 **지면 평면 깊이**(카메라 자세로 계산)를 잰다. 비교는 깊이 ÷ 지면 깊이(물체는 1보다 작고 땅은 1 근처)로 한다. 원시 깊이를 비교하면 땅의 기울기 때문에 늘 아래쪽 서브패치가 "앞쪽"이 되어 점이 아래로 끌린다. 점이 든 서브패치가 깊이가 없거나, 가장 앞쪽의 안정된(유효 50 % 이상, 사분위 폭 25 % 이하) 서브패치보다 25 %(`FlagCfg::depth_gap`과 같은 값) 넘게 멀면, 점을 그 서브패치의 앞쪽 화소 무게중심으로 옮긴다. 한 서브패치라도 수평선 위면 원시 깊이로 비교한다.

옮긴 점은 SAM 포인트 프롬프트, `fusion`의 `peak_depth_gap`, `locate_err_px`에 그대로 쓰인다.

### 2.3 `heat_in_mask` 블록 적분 (`fusion::cell_coverage`)

히트 칸 가운데 한 점으로 마스크를 찔러 보던 것을, 칸 사각형이 덮는 마스크 화소를 면적으로 적분한 비율 × 히트로 바꿨다. 피라미드에서는 칸이 마스크 16 × 16 블록 하나라 나눗셈 오차가 없고, 다른 격자에서도 걸친 화소를 걸친 만큼 센다. 칸과 마스크 화소가 1:1이면 예전과 같다.

## 3. 캐스케이드 (Phase 3)

### 3.1 SAM 지연 인코딩

개선안은 "프롬프트와 추적이 없으면 이미 건너뛴다"고 했지만, `SamTracker::update`는 언제나 먼저 인코더를 불렀다. 이제 게이트(보정표 · 점수 바닥 · §2.1의 기하 게이트)를 지난 프롬프트도 살아 있는 추적도 없으면 인코더를 부르지 않는다(`wants_frame`, `encoded_frames`로 셈). `prefetch`도 `lazy_sam`이면 깊이 · 의미 위치를 먼저 돌리고, 보정 게이트를 지난 점이 있는 프레임만 묶어 인코딩한다. 묶음에서 빠진 프레임에 추적이 살아 있으면 `update`가 그 한 장만 인코딩한다(결과는 같고 배치만 다름).

### 3.2 L0 타일의 밀집 풀링 (기본 끔)

`EmbeddingCfg::dense.enabled`이면 타일 격자를 L1의 약수로 맞추고(§1.3), SigLIP2 밀집 패스 한 번(피라미드 격자)에서 거친 단계 타일을 **덮인 비율 × 풀링 헤드 어텐션**으로 가중 평균한 벡터로 만든다(`pool_dense_tiles`, 화면 전체 타일은 풀링 헤드의 출력 그대로). 가장 고운 단계 중 ROI를 지난 타일만 잘라서 인코딩한다(`crop_finest`). 패치 벡터는 화면 전체를 본 것이라 잘라 낸 타일의 임베딩과 같지 않고, 타일 점수 분포가 달라진다. 그래서 `embedding` 보정표를 이 경로에서 다시 잰 뒤에 켜야 해 기본은 끔이다.

## 4. 설정

```jsonc
// MlFixCfg (v1 기본값)
"pyramid": {
  "l1_short_side": 12, "l1_budget": 256, "l2_ratio": 2, "mask_per_cell": 16,
  "gate": { "enabled": true, "sky_max_frac": 0.6, "far_m": 18.0, "min_near_frac": 0.25 },
  "snap": { "enabled": true, "edge_gap": 0.25, "max_rel_spread": 0.25, "min_valid_frac": 0.5 },
  "lazy_sam": true
}
// 끄기: "pyramid": null (MlFixCfg::legacy(), 예전 JSON)
// 빠른 깊이: PyramidCfg::fast() — TIPSv2를 L1 격자 그대로(토큰 ¼)
// EmbeddingCfg: "dense": { "enabled": false, "crop_finest": true }
```

`MlPipeline::sync_pyramid`가 설정을 엔진마다 넘긴다(SigLIP2 격자 · TIPSv2 격자 · SAM 마스크 격자 · 대역 모델의 히트 · 깊이 패치 격자). `prefetch`와 `analyse`가 매번 부르고, 런타임은 모델을 붙일 때 부른다. 대역 모델도 피라미드를 따라 히트 18 × 12, 깊이 패치 L2 격자를 쓴다(끄면 예전 19 × 13 · 36 × 24).

키프레임 기록(`MlView::pyramid`)에 그 프레임의 L1 · L2 · 마스크 격자와 게이트로 지운 패치 수 · 옮긴 점 수가 남는다.

## 5. 시험

**[측정]** 단위 시험(실제 가중치 없음):

| 시험 | 확인한 것 |
|---|---|
| `preprocess::the_pyramid_grid_is_half_the_depth_grid_and_fits_the_budget` | 384 × 256 → 18 × 12, 640 × 480 → 16 × 12, 세로 화면 · 정사각형 · 넓은 화면(11 × 22) · 극단 화면비가 예산 256 안 |
| `model::the_pyramid_grid_is_the_dense_grid_it_asks_for` | 밀집 패스가 요청한 격자로 돈다(작은 무작위 모델) |
| `tips::the_pyramid_grid_is_twice_siglip2s_and_matches_the_default_on_keyframes` | L2 = 2 × L1이고 키프레임에서는 기본 격자와 같음, 설정이 풀리면 예전 `input_grid`로 돌아옴 |
| `pyramid::the_three_layers_are_whole_multiples_of_one_grid` | 18 × 12 : 36 × 24 : 288 × 192, 16 × 12 : 32 × 24 : 256 × 192, 서브패치가 칸을 정확히 나눔 |
| `pyramid::sky_patches_lose_their_heat_and_the_peak_moves_to_the_ground` | 하늘 패치의 피크가 지워지고 땅 위의 다음 피크가 이김, 하늘에만 있던 라벨은 점이 없음 |
| `pyramid::a_segmentation_names_the_sky_and_unknown_ground_is_not_sky` | 수평선 아래 무한대 깊이는 하늘이 아님, 분할이 있으면 분할로 |
| `pyramid::a_peak_on_the_background_beside_an_object_snaps_onto_it` | 물체 옆 배경 서브패치의 점이 물체 위로 옮겨짐 |
| `pyramid::the_ground_slope_inside_a_patch_is_not_an_edge` | 땅만 있는 패치 18곳에서 점이 움직이지 않음(아래로 끌리지 않음) |
| `fusion::heat_in_mask_integrates_each_cell_over_its_mask_block` | 칸 16 × 16 블록의 절반이 덮이면 0.5, 걸친 화소는 걸친 만큼 |
| `sam::a_pyramid_mask_is_a_whole_number_of_blocks_per_siglip_patch` | 288 × 192 · 256 × 192 마스크, 예전 ÷2 샘플링은 새 함수의 한 경우 |
| `sam::nothing_to_prompt_and_nothing_to_follow_skips_the_encoder` | 빈 탐색 프레임에서 인코더 0회, 추적이 살면 프롬프트 없이도 인코딩 |
| `ml::the_truth_scores_a_mask_of_any_grid_on_its_own_pixels` | 마스크 격자가 달라도 `track_iou_truth`가 같은 상자에 1.0 |
| `ml::the_pipeline_hands_its_pyramid_to_the_stand_ins` | 대역 모델 격자가 설정을 따름 |
| `parallel::tile_grids_snap_to_whole_blocks_of_patches` · `a_pooled_tile_is_the_attention_weighted_mean_of_the_patches_it_covers` · `embedding::dense_tiles_pool_the_coarse_levels_and_crop_only_the_finest` | L0 격자 정렬, 풀링 식, 가장 고운 단계만 잘라 인코딩 |

**실제 모델로 잴 것**(요청자): 개선안 §4의 지표를 `pyramid: null`과 기본값으로 나눠 비교한다. `ml_ab_quests`(`SIGLIP2_DIR` · `TIPS_DIR` · `SAM_DIR`, `--ignored`)에 팔 `real_v1.3`(피라미드 켬)을 더했고, `real_v1.0`은 v1 고침에서 피라미드만 끈 것이다. `ml_ab_benchmark`는 SigLIP2 점을 미리(NaFlex 격자로) 계산해 넣으므로 격자 정렬 자체는 반영되지 않고 게이트 · 스내핑만 반영된다.

```bash
cargo test --release -p noto-runtime --test ml_ab -- --ignored --nocapture ml_ab_quests
```

| 지표 | 어디서 보나 |
|---|---|
| `locate_err_px`, `peak_depth_gap`, `heat_in_mask`, `track_iou_truth` | 키프레임 기록(`MlView`), SDS `locate` · `fusion` · `tracker` 트랙 |
| 게이트로 지운 패치 · 옮긴 점 | `MlView::pyramid` |
| SigLIP2 토큰 수 | 247 → 216 (384 × 256) **[유도]** |
| SAM 인코더 호출 | `SamTracker::encoded_frames` |
| 보정 | `siglip2-dense+bg+pyr` 모드의 `CalibTable`을 새로 재야 보정 게이트가 다시 걸린다(`data/ml_calib.json`은 지금 비어 있음) |
