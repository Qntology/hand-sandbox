// 모의 로봇 프리셋 · AI 외형 수정 · 부품 등급: rig_* 와 body_* 명령의 가짜 구현.
// 프리셋 목록은 런타임의 기본 프리셋과 같고, 측정값과 검사는 등급에서 만든 데모 숫자다.
import { log, world } from './mock-world.js';
import { unitRows, unitSlot } from './mock-units.js';

const BUILT_IN = [
  ['humanoid', '휴머노이드 (1.65 m · 표준)', '사람 키 · 문과 선반에 맞는 크기', 'human_foot'],
  ['forklift', '지게차 (1톤 · 바퀴)', '실내 전동 지게차 크기', 'wheel'],
  ['forklift_mecanum', '지게차 (1톤 · 메카넘)', '옆으로도 움직이는 바퀴', 'mecanum'],
  ['forklift_track', '지게차 (1톤 · 트랙)', '험지용 궤도', 'track'],
  ['compact_humanoid', '소형 휴머노이드 (1.21 m)', '기존 벤치마크의 몸', 'human_foot'],
  ['compact_truck', '소형 지게차 (축거 0.30 m)', '기존 벤치마크의 몸', 'wheel'],
].map(([id, name_ko, note_ko, base]) => ({ id, name_ko, note_ko, base, build: 'standard', builtin: true }));

const saved = [];
let plan = { head: 'mid', chest: 'mid', belly: 'mid', arm: 'mid', leg: 'mid', battery_low: false, lift: false, build: 'standard' };
let edits = 0;
const pendingBase = new Map(); // 유닛 → rig_prompt 가 고른 베이스 (rig_commit 때 적용)

const allPresets = () => [...BUILT_IN, ...saved];
const presetById = (id) => {
  const p = allPresets().find((x) => x.id === id);
  if (!p) throw `로봇 프리셋 '${id}'가 없습니다`;
  return p;
};

export function rememberPlan(body) {
  if (body) plan = body;
}

export function rigPresets() {
  return { presets: allPresets(), current: world() ? unitRows()[0].rig_id : 'forklift' };
}

function checksFor(base) {
  const truck = base !== 'human_foot';
  return [
    { label: '기립 균형', ok: true, detail: '균형 여유 0.34 · 무게중심 0.84 m (모의)' },
    { label: truck ? '창고 문 통과' : '집 문 통과', ok: true, detail: truck ? '폭 1.05 m · 높이 2.10 m (모의)' : '키 1.65 m · 어깨 폭 0.46 m (모의)' },
    { label: truck ? '포크로 원목 들기' : '원목 운반', ok: true, detail: truck ? '정격 1000 kg (모의)' : '상시 적재 20.0 kg (모의)' },
  ];
}

export function rigValidate({ id }) {
  const p = presetById(id);
  return { mass_kg: 64.2, com_m: 0.84, eye_m: 1.54, margin: 0.34, joints: 15, dof: 15, width_m: 0.42, length_m: 0.3, weakest: { name: 'knee_r', ratio: 0.46 }, warnings: [], checks: checksFor(p.base) };
}

export function rigDesign({ id }) {
  const p = presetById(id);
  return {
    sc: null,
    design: {
      name: p.id,
      components: [
        { name: 'pelvis', mass_kg: 30, color: [120, 120, 130], solids: [{ shape: 'box', size: [0.6, 0.4, 0.2], center: [0, 0, 0.1] }] },
        { name: 'upper_arm_r', mass_kg: 4, color: [200, 160, 60], solids: [{ shape: 'cylinder', radius: 0.04, height: 0.5, base: [0, 0, 0.7], axis: [0, 0, -1] }] },
        { name: 'hand_r', mass_kg: 1, color: [60, 60, 70], solids: [{ shape: 'box', size: [0.05, 0.09, 0.1], center: [0, 0, 0.15] }] },
      ],
      joints: [
        { name: 'shoulder_r', kind: 'revolute', a: 'pelvis', b: 'upper_arm_r', at: [0, 0, 0.7], axis: [0, -1, 0], limits: [-60, 170] },
        { name: 'wrist_r', kind: 'rigid', a: 'upper_arm_r', b: 'hand_r', at: [0, 0, 0.2], axis: [0, 0, 1], limits: null },
      ],
    },
  };
}

export function rigDelete({ id }) {
  const i = saved.findIndex((p) => p.id === id);
  if (i < 0) throw `저장된 로봇 '${id}'가 없습니다`;
  if (unitRows().some((u) => u.rig_id === id)) throw '지금 월드에서 쓰는 로봇은 지울 수 없습니다';
  saved.splice(i, 1);
  return rigPresets();
}

// ── AI(MCP) 외형 수정 ───────────────────────────────────────────
const PARTS = [[/팔/, 'arm', '팔'], [/다리/, 'leg', '다리'], [/포크/, 'fork', '포크'], [/목/, 'neck', '목'], [/몸통|허리/, 'trunk', '몸통']];
const BASES = [[/트랙|궤도/, 'track', '트랙'], [/메카넘/, 'mecanum', '메카넘'], [/바퀴/, 'wheel', '바퀴'], [/사람\s*발|이족/, 'human_foot', '사람 발']];

export function rigPrompt({ unit, text }) {
  const row = unitRows().find((u) => u.id === unit);
  if (!row) throw `유닛 ${unit} 이 없습니다`;
  const t = String(text ?? '');
  const num = /(\d+(?:\.\d+)?)\s*(cm|센티|m|미터)/i.exec(t);
  const delta = num ? Number(num[1]) * (/^(cm|센티)/i.test(num[2]) ? 0.01 : 1) : 0.1;
  const calls = [];
  for (const [re, part, ko] of PARTS) {
    if (re.test(t)) calls.push({ tool: 'solid.scale_link', args: { part, delta_m: delta }, ok: true, result_ko: `${ko} 길이 ${(delta * 100).toFixed(0)} cm 조정 (모의)` });
  }
  let base = null;
  for (const [re, b, ko] of BASES) {
    if (!base && re.test(t)) {
      base = b;
      calls.push({ tool: 'rig.set_base', args: { base: b }, ok: true, result_ko: `베이스를 ${ko}(으)로 교체 (모의)` });
    }
  }
  if (!calls.length) return { ok: false, unit, summary_ko: '', calls: [], name: null, design: null, error_ko: '고칠 부위를 찾지 못했습니다. 예: 팔 길이를 15cm 늘려줘' };
  edits += 1;
  pendingBase.set(unit, base);
  return { ok: true, unit, summary_ko: calls.map((c) => c.result_ko).join(' · '), calls, name: `유닛${unit} 수정 ${edits}`, design: rigDesign({ id: row.rig_id }).design, error_ko: null };
}

export function rigCommit({ unit, name, urdf }) {
  const row = unitRows().find((u) => u.id === unit);
  if (!row) throw `유닛 ${unit} 이 없습니다`;
  const id = String(name).replace(/\s+/g, '_');
  const base = pendingBase.get(unit) ?? row.base;
  const preset = { id, name_ko: name, note_ko: `AI 수정 (모의)${urdf ? ' · CAD 내보내기' : ' · 런타임 URDF'}`, base, build: 'rig', builtin: false };
  const i = saved.findIndex((p) => p.id === id);
  if (i >= 0) saved[i] = preset;
  else saved.push(preset);
  const slot = unitSlot(unit);
  slot.setRig({ id, name_ko: name });
  slot.setBase(base);
  log('mcp', 'info', `${name}: 검증 통과 · 유닛 #${unit} 제자리 교체 (모의)`, { unit, preset: id });
  return {
    ok: true, unit, preset_id: id, name_ko: name, summary_ko: `${name} 저장 · 제자리 교체 완료 (모의)`,
    checks: checksFor(preset.base), warnings: [], mass_kg: 64.2, com_m: 0.84, eye_m: 1.54, sds_key: `scraft:${id}:0000beef`,
  };
}

export function unitApply({ unit, preset, base, cameras }) {
  const slot = unitSlot(unit);
  const said = [];
  if (preset) {
    const p = presetById(preset);
    slot.setRig({ id: p.id, name_ko: p.name_ko });
    slot.setBase(p.base);
    said.push(`체형 ${p.name_ko}`);
  }
  if (base) {
    slot.setBase(base);
    said.push(`발 ${base}`);
  }
  if (cameras) {
    slot.setCameras(cameras === 2 ? 2 : 1);
    said.push(`카메라 ${cameras}대`);
  }
  log('crew', 'info', `유닛 #${unit}: ${said.join(' · ')} 적용 (모의)`, { unit });
  const row = unitRows().find((u) => u.id === unit);
  return { ok: true, unit, summary_ko: `${said.join(' · ')} 적용 (모의, 월드 유지)`, checks: checksFor(row.base), warnings: [] };
}

// ── 부품 등급 ───────────────────────────────────────────────────
const TIERS = [{ id: 'budget', label_ko: '보급' }, { id: 'mid', label_ko: '중급' }, { id: 'pro', label_ko: '고급' }];
const TIER_KG = { budget: 4.5, mid: 5.5, pro: 7 };

export function bodyPresets() {
  return {
    presets: { tiers: TIERS, leg_tiers: TIERS, named: ['budget', 'mid', 'pro', 'budget_arms_mid_base', 'pro_arms', 'battery_low', 'wheel_lift'] },
    current: plan, current_id: 'mid',
  };
}

export function bodyPlanEval({ body, base }) {
  const p = body ?? plan;
  const parts = [p.head, p.chest, p.belly, p.arm, p.leg];
  const mass = 38 + parts.reduce((n, t) => n + (TIER_KG[t] ?? 5.5), 0);
  const limits = (kg) => ({ max_speed_mps: base === 'human_foot' ? 1.2 - kg * 0.01 : 1.5 - kg * 0.01, max_slope_deg: 18, max_step_m: 0.12, capacity_kg: p.arm === 'pro' ? 26 : 20, reasons: kg ? [] : ['모의 값입니다'] });
  return {
    id: [...new Set(parts)].join('+'), mass_kg: mass, com_height_m: p.battery_low ? 0.82 : 0.84, battery_wh: 720, cost_usd: 18000 + mass * 120,
    limits_empty: limits(0), limits_loaded: limits(8),
    lifts: [
      { label_ko: '원목', mass_kg: 3, ok: true, two_arms: false, worst: '어깨', worst_ratio: 0.42 },
      { label_ko: '돌', mass_kg: 4, ok: true, two_arms: true, worst: '허리', worst_ratio: 0.61 },
    ],
    nn: [],
  };
}

export const worldBody = () => ({ plan, id: 'mid', limits: {} });
