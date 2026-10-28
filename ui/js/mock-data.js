// 모의 백엔드용 정적 데이터. 형태는 실제 Rust 응답(src-tauri/src/views.rs, noto-ml)과 같게 맞췄다.
// 사전은 data/logistics_dictionary.json 의 28개 항목을 그대로 옮긴 것이다(UI 개발용 복사본).
const D = [
  ['carton_small', 'small cardboard box', '소형 골판지 박스', 'package', [0.20, 0.15, 0.10], 1.0, true, 'assumed typical size'],
  ['carton_medium', 'cardboard box', '골판지 박스', 'package', [0.30, 0.25, 0.20], 3.0, true, 'assumed typical size'],
  ['carton_large', 'large cardboard box', '대형 골판지 박스', 'package', [0.50, 0.40, 0.40], 12.0, false, 'assumed typical size; too large for the hand'],
  ['parcel', 'parcel', '소포', 'package', [0.30, 0.20, 0.15], 2.0, true, 'assumed typical size'],
  ['electronics_box', 'electronics box', '전자제품 상자', 'package', [0.35, 0.25, 0.15], 2.0, true, 'assumed typical size'],
  ['shoe_box', 'shoe box', '신발 상자', 'package', [0.33, 0.20, 0.12], 0.8, true, 'assumed typical size'],
  ['poly_mailer', 'poly mailer bag', '폴리 우편 봉투', 'package', [0.30, 0.40, 0.02], 0.3, true, 'assumed typical size'],
  ['envelope', 'envelope', '봉투', 'package', [0.32, 0.23, 0.01], 0.05, true, 'assumed typical size'],
  ['sack', 'sack of goods', '포대', 'package', [0.50, 0.30, 0.10], 25.0, false, 'assumed typical size'],
  ['water_pack', 'pack of bottled water', '생수 묶음', 'package', [0.30, 0.20, 0.20], 6.0, false, 'assumed typical size'],
  ['plastic_tote', 'plastic tote', '플라스틱 토트 박스', 'container', [0.60, 0.40, 0.20], 3.0, true, 'assumed typical size'],
  ['plastic_crate', 'plastic crate', '플라스틱 크레이트', 'container', [0.60, 0.40, 0.30], 4.0, false, 'assumed typical size'],
  ['tray', 'plastic tray', '트레이', 'container', [0.40, 0.30, 0.05], 0.5, true, 'assumed typical size'],
  ['pallet', 'pallet', '팔레트', 'infrastructure', [1.20, 0.15, 0.80], 25.0, false, 'assumed typical size'],
  ['wrapped_pallet', 'pallet wrapped in plastic film', '랩핑된 팔레트', 'infrastructure', [1.20, 1.60, 0.80], 600.0, false, 'assumed typical size'],
  ['bottle', 'bottle', '병', 'item', [0.07, 0.25, 0.07], 0.6, true, 'assumed typical size'],
  ['can', 'beverage can', '캔', 'item', [0.07, 0.12, 0.07], 0.4, true, 'assumed typical size'],
  ['tape_roll', 'tape roll', '테이프 롤', 'item', [0.07, 0.07, 0.05], 0.1, true, 'assumed typical size'],
  ['film_roll', 'stretch film roll', '스트레치 필름 롤', 'item', [0.25, 0.25, 0.25], 2.5, false, 'assumed typical size'],
  ['barcode_scanner', 'handheld barcode scanner', '핸디 바코드 스캐너', 'tool', [0.06, 0.16, 0.10], 0.2, true, 'assumed typical size'],
  ['shipping_label', 'shipping label', '운송장 라벨', 'label', [0.10, 0.15, 0.001], 0.001, false, 'assumed typical size'],
  ['barcode_label', 'barcode label', '바코드 라벨', 'label', [0.05, 0.03, 0.001], 0.001, false, 'assumed typical size'],
  ['qr_sticker', 'QR code sticker', 'QR 코드 스티커', 'label', [0.03, 0.03, 0.001], 0.001, false, 'assumed typical size'],
  ['forklift', 'forklift', '지게차', 'vehicle', [2.0, 2.2, 1.2], 2000.0, false, 'assumed typical size'],
  ['agv_robot', 'AGV robot', 'AGV 로봇', 'vehicle', [0.70, 0.40, 0.30], 150.0, false, 'assumed typical size'],
  ['conveyor_belt', 'conveyor belt', '컨베이어 벨트', 'infrastructure', [2.0, 0.80, 0.80], 300.0, false, 'assumed typical size'],
  ['warehouse_rack', 'warehouse storage rack', '창고 선반', 'infrastructure', [1.20, 2.0, 0.60], 80.0, false, 'assumed typical size'],
  ['worker', 'person in hi-vis vest', '형광조끼 작업자', 'person', [0.50, 1.70, 0.30], 70.0, false, 'assumed typical size'],
];

export const DICTIONARY = D.map(([id, label_en, label_ko, category, size_m, typical_mass_kg, graspable, source]) => (
  { id, label_en, label_ko, category, size_m, typical_mass_kg, graspable, source }));

export const BASES = {
  wheel: { kind: 'wheel', max_speed_mps: 1.5, max_slope_deg: 15.0, max_step_m: 0.02, holonomic: false, power_w: 120.0, footprint_m: 0.30 },
  mecanum: { kind: 'mecanum', max_speed_mps: 1.0, max_slope_deg: 8.0, max_step_m: 0.015, holonomic: true, power_w: 150.0, footprint_m: 0.30 },
  track: { kind: 'track', max_speed_mps: 0.8, max_slope_deg: 30.0, max_step_m: 0.12, holonomic: false, power_w: 200.0, footprint_m: 0.36 },
  human_foot: { kind: 'human_foot', max_speed_mps: 1.2, max_slope_deg: 20.0, max_step_m: 0.18, holonomic: false, power_w: 180.0, footprint_m: 0.25 },
};

// 회전 방식(Rust world-sim::steer 와 같은 규칙의 모의값): turn_move_rad 보다 크게 틀어야 하면 멈춰서 돈다(pivot) 또는
// 앞뒤로 끊어 돈다(kturn: 트랙터식 3점 회전). 뒤쪽 가까이(reverse_max_m 안)의 목표는 돌지 않고 후진한다.
// min_radius_m 는 달리며 도는 가장 작은 반경, pivot_rate 는 제자리 회전 속도(스키드 조향의 비비는 저항으로 느림).
export const BASE_TURN_SPECS = {
  wheel: { turn_rate_rad_s: 1.4, turn_move_rad: 0.45, min_radius_m: 0.55, pivot_rate_rad_s: 0.9, reverse_mps: 0.6, reverse_max_m: 3.0, kturn: false },
  mecanum: { turn_rate_rad_s: 1.8, turn_move_rad: 0.85, min_radius_m: 0.0, pivot_rate_rad_s: 1.8, reverse_mps: 1.0, reverse_max_m: 0, kturn: false, holonomic: true },
  track: { turn_rate_rad_s: 0.75, turn_move_rad: 0.25, min_radius_m: 0.9, pivot_rate_rad_s: 0.35, reverse_mps: 0.4, reverse_max_m: 2.5, kturn: true },
  human_foot: { turn_rate_rad_s: 1.0, turn_move_rad: 0.35, min_radius_m: 0.3, pivot_rate_rad_s: 1.6, reverse_mps: 0.35, reverse_max_m: 1.0, kturn: false, legs: true },
};

export const BASE_LABELS = {
  wheel: '바퀴 (4륜 차동)',
  mecanum: '메카넘 바퀴 (전방향)',
  track: '트랙 (크롤러·트랙터형)',
  human_foot: '사람 발 (이족)',
};

export const FINGER_PRESETS = [
  { count: 2, label_ko: '2지 집게', description_ko: '마주 보는 두 손가락. 평행 조 그리퍼에 가까움' },
  { count: 3, label_ko: '3지 (엄지+2)', description_ko: '엄지와 두 손가락으로 감싸는 삼점 파지' },
  { count: 4, label_ko: '4지 (엄지+3)', description_ko: '엄지와 세 손가락' },
  { count: 5, label_ko: '5지 (사람형)', description_ko: '사람 손과 같은 배치' },
];

export const JOINT_KINDS = [
  { id: 'hinge', label_ko: '힌지 (굴곡 1자유도)' },
  { id: 'universal', label_ko: '유니버설 (굴곡+외전)' },
  { id: 'ball', label_ko: '볼 (3자유도)' },
];

export const BACKENDS = [
  { backend: 'cpu', compiled: true, available: true, detail: 'CPU (모의 백엔드)' },
  { backend: 'cuda', compiled: false, available: false, detail: '이 빌드에는 CUDA 가 포함되지 않았습니다 (모의)' },
  { backend: 'metal', compiled: false, available: false, detail: '이 빌드에는 Metal 이 포함되지 않았습니다 (모의)' },
  { backend: 'vulkan', compiled: false, available: false, detail: '이 빌드에는 Vulkan 이 포함되지 않았습니다 (모의)' },
  { backend: 'rocm', compiled: false, available: false, detail: '이 빌드에는 ROCm 이 포함되지 않았습니다 (모의)' },
];

/** 실제 noto_core::simulate_drive 와 같은 식(base.rs). */
export function simulateDrive(spec, course) {
  let distance = 0;
  let time = 0;
  for (let i = 0; i < course.length; i++) {
    const seg = course[i];
    let blocked = false;
    if (seg.type === 'flat') {
      distance += seg.length_m;
      time += seg.length_m / spec.max_speed_mps;
    } else if (seg.type === 'slope') {
      if (seg.deg > spec.max_slope_deg) blocked = true;
      else {
        const factor = 1 - 0.5 * (seg.deg / spec.max_slope_deg);
        distance += seg.length_m;
        time += seg.length_m / (spec.max_speed_mps * factor);
      }
    } else if (seg.type === 'step') {
      if (seg.height_m > spec.max_step_m) blocked = true;
      else time += 0.5;
    }
    if (blocked) return { passable: false, failed_at: i, distance_m: distance, time_s: time, energy_j: spec.power_w * time };
  }
  return { passable: true, failed_at: null, distance_m: distance, time_s: time, energy_j: spec.power_w * time };
}

export const DEFAULT_COURSE = [
  { type: 'flat', length_m: 10 },
  { type: 'slope', length_m: 5, deg: 10 },
  { type: 'step', height_m: 0.05 },
  { type: 'slope', length_m: 5, deg: 20 },
  { type: 'step', height_m: 0.15 },
  { type: 'flat', length_m: 5 },
];
