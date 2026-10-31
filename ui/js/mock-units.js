// 모의 유닛: 최초 작업자와 서브 작업자를 units_list · world_frame.units 모양으로 묶고, 3인칭·포커스 시점의 모의 값(overlay · focus)을 만든다.
// 값은 UI 개발용 숫자이고 실제 동역학 결과가 아니다.
import { log, usesForks, world } from './mock-world.js';

const LOAD_LIMIT = 20;
const CAP_J = 30000;
const BUILT_ID = { human_foot: 'humanoid', wheel: 'forklift', mecanum: 'forklift_mecanum', track: 'forklift_track' };
const FORM_KO = { humanoid: '휴머노이드', forklift: '지게차' };

const need = () => {
  const w = world();
  if (!w) throw '월드가 없습니다. world_new 를 먼저 호출하세요';
  return w;
};

function stackLoad(list) {
  const m = new Map();
  for (const c of list) {
    const row = m.get(c.item_id) ?? { item: c.item_id, label_ko: c.label_ko, count: 0 };
    row.count += 1;
    m.set(c.item_id, row);
  }
  return [...m.values()];
}

/** 로봇 한 대마다 한 줄: 최초 작업자가 먼저. pos · yaw 는 모의 렌더러가 쓴다. */
export function unitRows() {
  const w = need();
  const a = w.agent;
  const q = w.quests.find((x) => x.state === 'active');
  const step = q?.steps.find((s) => s.state === 'active');
  const form = usesForks() ? 'forklift' : 'humanoid';
  const rig = a.rig ?? { id: BUILT_ID[w.base], name_ko: null };
  const rows = [{
    id: 1, first: true, form, base: w.base, label_ko: `${FORM_KO[form]} (최초 작업자)`,
    pos: a.pos, yaw: a.yaw, moving: (a.speed ?? 0) > 0.05, fallen: false, status_ko: step?.label_ko ?? '대기',
    carried_kg: a.carrying.reduce((n, c) => n + c.mass_kg, 0), capacity_kg: LOAD_LIMIT, load: stackLoad(a.carrying),
    cameras: w.cameras, battery_ratio: Math.max(0, 1 - w.energy / CAP_J), rig_id: rig.id, rig_name_ko: rig.name_ko ?? FORM_KO[form],
  }];
  for (const c of w.crew) {
    const base = c.base ?? (c.form === 'forklift' ? 'wheel' : 'human_foot');
    rows.push({
      id: c.id, first: false, form: c.form, base, label_ko: `${FORM_KO[c.form]} (서브 작업자)`,
      pos: c.pos, yaw: ((c.yaw_deg ?? 0) * Math.PI) / 180, moving: false, fallen: false, status_ko: c.status_ko ?? '대기',
      carried_kg: 0, capacity_kg: c.form === 'forklift' ? 200 : LOAD_LIMIT, load: [],
      cameras: c.cameras, battery_ratio: 0.92, rig_id: c.rig?.id ?? BUILT_ID[base], rig_name_ko: c.rig?.name_ko ?? FORM_KO[c.form],
    });
  }
  return rows;
}

export function unitsList() {
  return {
    units: unitRows().map(({ pos, yaw, load, moving, fallen, ...u }) => ({ ...u, x: pos[0], z: pos[1], yaw_deg: (yaw * 180) / Math.PI })),
  };
}

export function crewRemove({ id }) {
  const w = need();
  const n = w.crew.length;
  w.crew = w.crew.filter((c) => c.id !== id);
  if (w.crew.length === n) throw `서브 작업자 #${id} 가 없습니다`;
  w.teamSize = w.crew.length ? 2 : 1;
  log('crew', 'info', `서브 작업자 #${id} 제거 (모의)`, { id });
  return { ok: true };
}

/** unit_apply · rig_commit 이 몸을 바꿀 때 쓰는 대상. 최초 작업자는 월드의 발(w.base)과 연결돼 있다. */
export function unitSlot(id) {
  const w = need();
  if (id === 1) {
    return {
      w, setRig: (r) => { w.agent.rig = r; }, setBase: (b) => { w.base = b; }, setCameras: (n) => { w.cameras = n; },
    };
  }
  const c = w.crew.find((x) => x.id === id);
  if (!c) throw `유닛 ${id} 이 없습니다`;
  return {
    w, setRig: (r) => { c.rig = r; }, setCameras: (n) => { c.cameras = n; },
    setBase: (b) => {
      c.base = b;
      c.form = b === 'human_foot' ? 'humanoid' : 'forklift';
    },
  };
}

// ── 3인칭 시점의 관절 오버레이 (모의) ────────────────────────────
const JOINTS = [
  ['shoulder_r', 'sim', 0.42, 0.62], ['elbow_r', 'sim', 0.36, 0.52], ['waist_pitch', 'sim', 0.5, 0.42],
  ['hip_pitch_r', 'statics', 0.48, 0.28], ['knee_r', 'statics', 0.46, 0.2], ['neck', 'sim', 0.5, 0.72], ['drive', 'statics', 0.54, 0.1],
];

export function mockOverlay(width, height) {
  const w = need();
  const t = w.tick;
  const wave = (i, phase = 0) => 0.42 + 0.3 * Math.sin((t + phase) / 9 + i * 1.1);
  const at = (u, v) => ({ x: u * width, y: (1 - v) * height, visible: true });
  const joints = JOINTS.map(([name, kind, u, v], i) => {
    const ratio = Math.max(0.05, wave(i));
    return { name, kind, at: at(u, v), ratio, saturated: ratio > 0.68, stalled: false, q_deg: 20 * Math.sin(t / 11 + i), tau_nm: ratio * 42, peak_nm: 42 };
  });
  const ts = Array.from({ length: 24 }, (_, i) => i * 0.1);
  return {
    plan: 'mid', margin: 0.34,
    support: [at(0.38, 0.05), at(0.62, 0.05), at(0.66, 0.11), at(0.34, 0.11)],
    com: at(0.5, 0.48), com_ground: at(0.5, 0.08), zmp: at(0.51, 0.08), grf_tip: at(0.51, 0.3),
    joints,
    motion: {
      source_ko: '모의 자세 변화', min_margin: 0.31, feasible: true, t: ts[(t % 24)], settle_s: 1.2, static_margin: 0.36, notes: [],
      series: { t: ts, margin: ts.map((_, k) => 0.34 + 0.05 * Math.sin(k / 4)), joints: joints.map((j, i) => ({ name: j.name, kind: j.kind, ratio: ts.map((_, k) => Math.max(0.02, wave(i, k * 2))) })) },
      summary: joints.map((j) => ({ name: j.name, peak_ratio: j.ratio + 0.1, saturated_s: 0, stalled_s: 0, max_lag_deg: 1.2 })),
    },
  };
}

// ── 포커스 시점의 부위 값 (모의) ─────────────────────────────────
export function mockFocus() {
  const w = need();
  const forks = usesForks();
  const close = 0.4 + 0.3 * Math.sin(w.tick / 13);
  const finger = (y0, n) => Array.from({ length: n }, (_, k) => [0, y0 + 0.012 * k * close, 0.03 * (k + 1)]);
  const joint = (name, angle, ratio) => ({ name, angle_deg: angle, torque_nm: ratio * 30, peak_nm: 30, ratio, in_range: true });
  return {
    action_ko: '모의 행위', phase: 'hold', parts: [forks ? 'wheel' : 'hand'],
    body: { margin: 0.33, trunk_deg: 8, mass_kg: 64.2, com: [0, 0, 0.84], support: [-0.1, 0.14, -0.12, 0.12], zmp: [0.02, 0.01], feasible: true },
    hand: {
      palm: [0.08, 0.09], palm_at: [0.3, 0, 0.8], closure: close, thumbs: [true, false, false, false, false],
      fingers: [0, 0.002, 0.004, 0.006, 0.008].map((y) => finger(y, 3)),
      grip: { grip_n: 42, load_n: 12, slip_ratio: 1.6, contacts: 4, mu: 0.6, mu_est: 0.55, twist_margin_nm: 0.4 },
    },
    hand_joints: ['엄지', '검지', '중지'].flatMap((f) => [0, 1, 2].map((j) => ({ finger: f, joint: j, torque_nm: 0.2 + 0.1 * j, max_nm: 0.6, angle_deg: 20 + 15 * j }))),
    foot: forks ? [] : ['r', 'l'].map((side, i) => ({
      side, rect: [-0.08, 0.16, i ? 0.05 : -0.15, i ? 0.15 : -0.05], normal_n: 300, shear_n: 12, cop: [0.03, i ? 0.1 : -0.1], rcof: 0.04, mu: 0.7, in_air: false,
    })),
    foot_joints: forks ? [] : ['hip_pitch', 'hip_roll', 'knee', 'ankle'].flatMap((k, i) => ['r', 'l'].map((s) => joint(`${k}_${s}`, 10 + 8 * i, 0.2 + 0.1 * i))),
    wheel: forks ? [{ axle: 'rear', normal_n: 240, shear_n: 20, mu: 0.7, slip: 0.04 }, { axle: 'front', normal_n: 390, shear_n: 26, mu: 0.7, slip: 0.04 }] : [],
    track: w.base === 'track' ? { pressure_kpa: [6.4, 9.1], loaded_len_m: 1.2, slip: 0.06, mu: 0.7 } : null,
    neck: { camera_pitch_deg: 12, joint: { angle_deg: 8, torque_nm: 1.1, peak_nm: 4, ratio: 0.28 } },
    waist: { ratio: 0.34, torque_nm: 22, peak_nm: 64, structural: false },
  };
}
