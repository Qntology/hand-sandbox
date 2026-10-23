// 모의 백엔드: 손 기하·파지 시험·주행 비교·이미지 분류·합성 데이터(기존 커맨드).
import { BACKENDS, BASES, BASE_LABELS, DEFAULT_COURSE, DICTIONARY, FINGER_PRESETS, JOINT_KINDS, simulateDrive } from './mock-data.js';

const PALM = [0.085, 0.10]; // 너비, 길이 (m)

export function presets() {
  return {
    fingers: FINGER_PRESETS,
    joint_kinds: JOINT_KINDS,
    bases: Object.keys(BASES).map((k) => ({ base: BASES[k], label_ko: BASE_LABELS[k], report: simulateDrive(BASES[k], []) })),
  };
}

const SEG_LEN = { 2: [0.05, 0.035], 3: [0.045, 0.028, 0.02], 4: [0.04, 0.028, 0.02, 0.015] };

/** 단순 평면 FK: 손가락은 +z 로 뻗고 닫을수록 +y(손바닥 쪽)로 굽는다. */
function fingerPoints(x, z0, y0, theta0, lens, flexes) {
  const pts = [[x, y0, z0]];
  let theta = theta0;
  let z = z0;
  let y = y0;
  lens.forEach((l, i) => {
    theta += flexes[i];
    z += l * Math.cos(theta);
    y += l * Math.sin(theta);
    pts.push([x, y, z]);
  });
  return pts;
}

function flexAngles(n, actuation, closure) {
  const max = 1.35;
  const out = [];
  let prev = 0;
  for (let i = 0; i < n; i++) {
    const f = actuation.mode === 'coupled' ? (i === 0 ? closure * max : actuation.ratio * prev) : closure * max * (1 - i * 0.1);
    out.push(f);
    prev = f;
  }
  return out;
}

export function handGeometry({ config, closure }) {
  const n = config.fingers;
  const segs = config.joints.segments;
  if (![2, 3, 4].includes(segs)) throw `관절 개수 ${segs} 는 지원하지 않습니다 (2~4)`;
  const lens = SEG_LEN[segs];
  const fingers = [];
  const longFingers = n - 1;
  for (let i = 0; i < longFingers; i++) {
    const x = n === 2 ? 0 : -PALM[0] / 2 + (PALM[0] * (i + 0.5)) / longFingers;
    const scale = 1 - 0.06 * Math.abs(i - (longFingers - 1) / 2);
    const lens2 = lens.map((l) => l * scale);
    fingers.push({ name: `finger_${i + 1}`, is_thumb: false, points: fingerPoints(x, 0, 0.006, 0, lens2, flexAngles(segs, config.joints.actuation, closure)) });
  }
  const thumbLens = lens.slice(0, Math.max(2, segs - 1)).map((l) => l * 0.9);
  fingers.push({ name: 'thumb', is_thumb: true, points: fingerPoints(PALM[0] / 2, -0.03, 0.01, 0.55, thumbLens, flexAngles(thumbLens.length, config.joints.actuation, closure)) });
  const reach = Math.max(...fingers.map((f) => f.points[f.points.length - 1][2]));
  return { fingers, palm: PALM, base: BASES[config.base] ?? BASES.wheel, reach_m: reach };
}

export function graspTest({ config, payload, setup }) {
  const hand = handGeometry({ config, closure: 0.8 });
  const [sx, sy, sz] = payload.size_m;
  const half = [sx / 2, sy / 2, sz / 2];
  const zc = setup.z_offset;
  const n = config.fingers;
  const contactsN = sz <= 0.16 ? n : sz <= 0.3 ? Math.max(1, n - 2) : 0;
  const contacts = [];
  if (contactsN > 0) contacts.push({ source: { source: 'palm' }, point: { x: 0, y: 0, z: zc }, normal: { x: 0, y: -1, z: 0 }, force_n: 2.0 });
  for (let i = 0; i < contactsN; i++) {
    const isThumb = i === contactsN - 1;
    contacts.push({
      source: { source: 'finger', index: i },
      point: isThumb ? { x: 0, y: 2 * half[1], z: zc } : { x: 0, y: half[1] * (0.35 + (0.55 * i) / Math.max(1, contactsN)), z: zc + half[2] },
      normal: isThumb ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 },
      force_n: 6.0,
    });
  }
  const capacity = contactsN * payload.friction * 14;
  const required = payload.mass_kg * 9.81 * setup.safety_factor;
  return {
    result: {
      contacts,
      grasp_type: contactsN >= 3 ? 'power' : contactsN >= 1 ? 'pinch' : 'none',
      force_closed: contactsN >= 3,
      friction_capacity_n: capacity,
      required_n: required,
      stable: contactsN >= 2 && capacity >= required,
      max_payload_kg: capacity / (9.81 * setup.safety_factor),
    },
    fingers: hand.fingers,
    object: { center: [0, half[1], zc], half },
  };
}

export function driveCourses({ course }) {
  const c = course ?? DEFAULT_COURSE;
  return Object.keys(BASES).map((k) => ({ base: BASES[k], label_ko: BASE_LABELS[k], report: simulateDrive(BASES[k], c) }));
}

export function classifyImage({ imageBase64, topK }) {
  let h = (imageBase64?.length ?? 1) * 2654435761;
  const scored = DICTIONARY.map((e, i) => {
    h = (Math.imul(h ^ (i + 1), 1597334677) >>> 0) + 1;
    return { e, raw: (h % 1000) / 1000 };
  });
  const exp = scored.map((s) => Math.exp(s.raw * 5));
  const sum = exp.reduce((a, b) => a + b, 0);
  return scored
    .map((s, i) => ({ id: s.e.id, label_en: s.e.label_en, label_ko: s.e.label_ko, category: s.e.category, score: exp[i] / sum, size_m: s.e.size_m, typical_mass_kg: s.e.typical_mass_kg, graspable: s.e.graspable }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, Math.min(28, topK)));
}

// ── 합성 데이터 / 사전 / 장치 ────────────────────────────────────
const tables = { samples: 0, contacts: 0, joint_states: 0, dictionary: 0 };
const device = { selected: 'cpu', ordinal: 0 };
const STORE = '(모의) ~/.local/share/hand-sandbox/lance';

export const paths = () => ({ models_dir: '(모의) ~/.local/share/hand-sandbox/models', store_dir: STORE });

export function synthesize({ count }) {
  const n = Math.max(1, Math.min(5000, count));
  tables.samples += n;
  tables.contacts += n * 3;
  tables.joint_states += n * 15;
  return { episodes: n, stable: Math.round(n * 0.74), store_dir: STORE };
}

export function loadDictionary() {
  tables.dictionary = DICTIONARY.length;
  return DICTIONARY.length;
}

export function datasetSummary() {
  const rec = ['samples', 'contacts', 'joint_states', 'dictionary'].map((name) => ({ name, rows: tables[name], status: 'recorded' }));
  const planned = ['point_cloud', 'vision', 'tactile', 'embeddings'].map((name) => ({ name, rows: 0, status: 'planned' }));
  return [...rec, ...planned];
}

export function devices() {
  return { backends: BACKENDS, selected: device.selected, ordinal: device.ordinal };
}

export function selectDevice({ backend, ordinal }) {
  const b = BACKENDS.find((x) => x.backend === String(backend).toLowerCase());
  if (!b) throw `unknown backend "${backend}"`;
  if (!b.available) throw `${backend} 는 사용할 수 없습니다: ${b.detail}`;
  device.selected = b.backend;
  device.ordinal = ordinal;
  return devices();
}
