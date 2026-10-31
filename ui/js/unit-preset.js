// ⚙️ 프리셋 Drop-up: 고른 유닛의 체형 · 발 · 카메라를 월드를 건드리지 않고 제자리에서 바꾼다(unit_apply).
// 로봇 프리셋 목록(rig_presets, AI 로 만든 로봇 포함)과 측정·투입 전 검사(rig_validate), 저장한 로봇 삭제도 여기서 맡는다.
// 손 구성과 부품 등급은 유닛별 교체가 없어 월드 재생성으로만 적용되고, 그럴 때만 🔄 버튼이 보인다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { setStatus, toast } from './toast.js';
import { newWorld } from './sim.js';
import { setCameras } from './crew-panel.js';
import { setDraftBase } from './hand-panel.js';
import { setPlanFrom } from './presets.js';
import { refreshUnits } from './unit-select.js';
import { worldDirty } from './world-panel.js';

let list = [];
let applied = null; // 유닛에 지금 적용된 값 {key, preset, base, cameras}

const presetOf = (id) => list.find((p) => p.id === id) ?? null;
const selected = () => presetOf($('rig-preset-side').value);
const unitNow = () => state.unitInfo.get(state.selectedUnit) ?? state.units.find((u) => u.id === state.selectedUnit) ?? null;

function say(u) {
  const p = selected();
  $('rig-note').textContent = p?.note_ko ?? '';
  $('rig-delete').disabled = p?.builtin !== false;
  $('sum-rig').textContent = u?.rig_name_ko ?? '';
}

function fillPresets(pick) {
  const sel = clear($('rig-preset-side'));
  const group = (label, items) => {
    if (!items.length) return;
    sel.append(el('optgroup', { label }, ...items.map((p) => el('option', { value: p.id, text: p.name_ko }))));
  };
  group('기본 프리셋', list.filter((p) => p.builtin));
  group('내가 만든 로봇 (AI 생성 포함)', list.filter((p) => !p.builtin));
  sel.value = pick ?? '';
}

/** 프리셋 목록을 다시 받는다. pick 이 있으면 그것을 고른다. */
export async function refreshPresets(pick) {
  try {
    const r = await call('rig_presets', {}, { quiet: true });
    list = r.presets ?? [];
    fillPresets(pick || $('rig-preset-side').value || r.current || list[0]?.id);
    say(unitNow());
    await validate();
  } catch {
    $('rig-note').textContent = '로봇 프리셋을 불러오지 못했습니다.';
  }
}

// ── 측정 · 투입 전 검사 ─────────────────────────────────────────
function checksList(v) {
  const box = [];
  if (v.checks?.length) {
    box.push(el('ul', { class: 'rig-checks' }, ...v.checks.map((c) => el('li', { class: c.ok ? 'ok' : 'no', title: c.detail }, el('b', { text: `${c.ok ? '✓' : '✗'} ${c.label}` }), el('small', { text: c.detail })))));
  }
  if (v.warnings?.length) box.push(el('ul', { class: 'pp-reasons' }, ...v.warnings.map((w) => el('li', { text: w }))));
  return box;
}

function show(v) {
  const box = clear($('rig-summary'));
  if (!v) return;
  const n = (x, d = 2) => (Number.isFinite(x) ? Number(x).toFixed(d) : '—');
  const rows = [
    ['질량', `${n(v.mass_kg, 1)} kg`],
    ['무게중심 높이', `${n(v.com_m)} m`],
    ['눈 높이', `${n(v.eye_m)} m`],
    ['폭 · 길이', `${n(v.width_m)} · ${n(v.length_m)} m`],
    ['균형 여유', n(v.margin)],
  ];
  if (v.dof != null) rows.push(['자유도', String(v.dof)]);
  if (v.weakest) rows.push(['가장 힘든 관절', `${v.weakest.name} ${Math.round(v.weakest.ratio * 100)}%`]);
  box.append(el('div', { class: 'pv-kv' }, ...rows.flatMap(([k, val]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: val })])));
  box.append(...checksList(v));
}

async function validate() {
  const p = selected();
  if (!p) return;
  try {
    const v = await call('rig_validate', { id: p.id }, { quiet: true });
    if (selected()?.id !== p.id) return;
    state.rigChecks = v.checks ?? [];
    show(v);
    bus.emit('rig-checks', state.rigChecks);
  } catch {
    /* 표시할 검사가 없을 뿐이다 */
  }
}

// ── 유닛 ↔ 폼 ───────────────────────────────────────────────────
/** 유닛이 쓰는 프리셋: 런타임의 rig_id, 없으면 같은 이름, 그것도 없으면 같은 발의 기본 프리셋. */
function presetIdOf(u) {
  if (presetOf(u.rig_id)) return u.rig_id;
  const same = list.find((p) => p.name_ko === u.rig_name_ko) ?? list.find((p) => p.builtin && p.base === u.base);
  return same?.id ?? list[0]?.id ?? '';
}

/** 유닛을 고르거나 유닛의 값이 바뀌면 폼을 맞춘다. 사용자가 고치는 중에는 값이 같으므로 덮어쓰지 않는다. */
function syncForm() {
  const id = state.selectedUnit;
  if (id == null) {
    if (applied) setDraftBase(null);
    applied = null;
    return;
  }
  const u = unitNow();
  if (!u) return;
  const preset = presetIdOf(u);
  const key = `${id}|${preset}|${u.base}|${u.cameras}`;
  say(u);
  if (applied?.key === key) return;
  applied = { key, preset, base: u.base, cameras: u.cameras };
  $('preset-title').textContent = `⚙️ 유닛 #${id} 프리셋 및 하드웨어 스펙 설정 (CAD 에디터 없이 즉시 반영)`;
  $('rig-preset-side').value = preset;
  $('pp-cameras').value = String(u.cameras ?? 1);
  setDraftBase(u.base);
  say(u);
  validate();
}

function pickPreset() {
  const p = selected();
  if (p?.base) setDraftBase(p.base);
  say(unitNow());
  validate();
}

/** 런타임이 최초 작업자의 몸을 바꿨으면 재생성 기준값도 그 몸으로 맞춘다. */
async function syncFirstWorker(u) {
  if (!u?.first) return;
  state.config.base = u.base;
  state.worldConfigKey = JSON.stringify(state.config);
  try {
    const w = await call('world_body', {}, { quiet: true });
    if (w?.plan) setPlanFrom(w.plan);
  } catch {
    /* world_body 가 없는 백엔드 */
  }
  bus.emit('config', state.config);
}

/** rig_commit · unit_apply 결과를 알린다: 토스트(요약 · 검사)와 Drop-up 의 검사 목록. */
export function reportApplied(r, title) {
  const fails = (r.checks ?? []).filter((c) => !c.ok).map((c) => `✗ ${c.label}: ${c.detail}`);
  const passed = (r.checks ?? []).filter((c) => c.ok).length;
  const counts = r.checks?.length ? `검사 ${passed}/${r.checks.length} 통과` : '';
  toast(title, [r.summary_ko, counts, ...fails, ...(r.warnings ?? [])].filter(Boolean).join('\n'), r.ok ? 'ok' : 'err', 8000);
  setStatus(`${title}: ${r.summary_ko ?? ''}`, !r.ok);
  if (r.checks?.length || r.warnings?.length) clear($('rig-summary')).append(...checksList(r));
}

/** 유닛의 체형이 바뀐 뒤: 유닛 값·프리셋 목록·몸 기준값을 새로 맞춘다. */
export async function afterUnitChange(pick) {
  await refreshUnits();
  await refreshPresets(pick);
  applied = null;
  syncForm();
  await syncFirstWorker(unitNow());
  bus.emit('frame-stale');
}

async function apply() {
  const unit = state.selectedUnit;
  if (unit == null || !applied) return;
  const u = unitNow();
  const cameras = Number($('pp-cameras').value) || 1;
  const preset = selected()?.id ?? '';
  const args = { unit };
  if (preset && preset !== applied.preset) args.preset = preset;
  const baseOfPreset = args.preset ? presetOf(preset)?.base : applied.base;
  if (state.draftBase && state.draftBase !== baseOfPreset) args.base = state.draftBase;
  const camsChanged = cameras !== applied.cameras;
  // 최초 작업자의 카메라는 모델 구성까지 바뀌므로 camera_rig 로 간다.
  if (camsChanged && !u?.first) args.cameras = cameras;
  if (!args.preset && !args.base && !args.cameras && !camsChanged) {
    toast('바뀐 것이 없습니다', '이 유닛에 이미 적용된 체형·발·카메라입니다.');
    return;
  }
  const btn = $('pp-apply');
  btn.disabled = true;
  try {
    if (args.preset || args.base || args.cameras) reportApplied(await call('unit_apply', args), `유닛 #${unit}에 적용`);
    if (camsChanged && u?.first) await setCameras(cameras);
    await afterUnitChange(preset);
  } catch {
    /* api-error 가 표시 */
  } finally {
    btn.disabled = false;
  }
}

async function regenerate() {
  const btn = $('pp-regen');
  btn.disabled = true;
  try {
    const ok = await newWorld(Number($('world-seed').value) || 0, Number($('world-size').value) || 64);
    if (ok) toast('월드 재생성', '손 구성과 부품 등급을 새 월드에 적용했습니다.', 'ok');
  } finally {
    btn.disabled = false;
  }
}

async function remove() {
  const p = selected();
  if (!p || p.builtin) return;
  try {
    await call('rig_delete', { id: p.id });
    applied = null;
    await refreshPresets(presetIdOf(unitNow() ?? {}));
    syncForm();
  } catch {
    /* api-error 가 표시 */
  }
}

function syncRegen() {
  $('pp-regen').hidden = !worldDirty();
}

export async function initUnitPreset() {
  $('rig-preset-side').addEventListener('change', pickPreset);
  $('rig-delete').addEventListener('click', remove);
  $('pp-apply').addEventListener('click', apply);
  $('pp-regen').addEventListener('click', regenerate);
  for (const ev of ['unit', 'unit-info']) bus.on(ev, syncForm);
  for (const ev of ['config', 'body-plan', 'world-new']) bus.on(ev, syncRegen);
  bus.on('world-new', () => refreshPresets());
  syncRegen();
  await refreshPresets();
}
