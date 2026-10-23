// 손·발 구성, 물체·파지 시험, 파지 결과, 발별 주행 비교. 기존 app.js 의 기능을 그대로 옮겼다.
// 같은 Tauri 커맨드(presets, hand_geometry, grasp_test, drive_courses, dictionary)를 같은 인자로 호출한다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, debounce, el, fmt } from './util.js';
import { redrawHand } from './hand-canvas.js';

const COURSE_LABELS = ['평지 10m', '경사 5m (10°)', '턱 5cm', '경사 5m (20°)', '계단 15cm', '평지 5m'];
const GRASP_TYPE_KO = { none: '없음', pinch: '핀치 (2지 이하)', power: '파워 (3지 이상)' };

/** 실행 중에 또 호출되면 끝난 뒤 한 번만 다시 실행한다(슬라이더 연타로 invoke 가 쌓이지 않게). */
function coalesce(fn) {
  let running = false;
  let again = false;
  return async () => {
    if (running) {
      again = true;
      return;
    }
    running = true;
    try {
      do {
        again = false;
        await fn();
      } while (again);
    } finally {
      running = false;
    }
  };
}

// ── 손 구성 ─────────────────────────────────────────────────────
function renderFingerChoices() {
  const wrap = $('finger-count');
  clear(wrap);
  for (const f of state.presets.fingers) {
    wrap.append(el('button', {
      type: 'button', text: `${f.count}지`, title: f.label_ko, dataset: { count: f.count },
      aria: { pressed: String(f.count === state.config.fingers) },
      on: {
        click: () => {
          state.config.fingers = f.count;
          syncFingerChoices();
          configChanged();
          requestRefresh();
        },
      },
    }));
  }
  syncFingerChoices();
}

function syncFingerChoices() {
  const preset = state.presets.fingers.find((f) => f.count === state.config.fingers);
  for (const b of $('finger-count').children) {
    b.setAttribute('aria-pressed', String(Number(b.dataset.count) === state.config.fingers));
  }
  $('finger-desc').textContent = preset ? `${preset.label_ko}: ${preset.description_ko}` : '';
}

function renderJointSelects() {
  for (const id of ['base-kind', 'distal-kind']) {
    const sel = $(id);
    clear(sel);
    for (const k of state.presets.joint_kinds) sel.append(el('option', { value: k.id, text: k.label_ko }));
  }
}

function renderBaseChoices() {
  const wrap = $('base-choices');
  clear(wrap);
  for (const b of state.presets.bases) {
    const s = b.base;
    const sub = Number.isFinite(s.max_speed_mps)
      ? `최고 ${fmt.n(s.max_speed_mps, 1)} m/s · 경사 ${fmt.n(s.max_slope_deg, 0)}° · 턱 ${fmt.n(s.max_step_m * 100, 1)} cm`
      : '';
    const input = el('input', { type: 'radio', name: 'base', value: s.kind, checked: s.kind === state.config.base });
    input.addEventListener('change', () => {
      state.config.base = input.value;
      configChanged();
      requestRefresh();
      refreshDrive();
    });
    wrap.append(el('label', { class: 'tree-item tree-item-2l' }, input,
      el('span', { class: 'tree-text' }, el('span', { class: 'tree-name', text: b.label_ko }), sub ? el('span', { class: 'tree-sub', text: sub }) : null)));
  }
}

function readJointControls() {
  const mode = $('actuation').value;
  $('ratio-row').hidden = mode !== 'coupled';
  state.config.joints = {
    segments: Number($('segments').value),
    base_kind: $('base-kind').value,
    distal_kind: $('distal-kind').value,
    actuation: mode === 'coupled' ? { mode: 'coupled', ratio: Number($('ratio').value) } : { mode: 'direct' },
  };
  $('ratio-out').textContent = Number($('ratio').value).toFixed(2);
}

function applyJointControls() {
  const j = state.config.joints;
  $('segments').value = String(j.segments);
  $('base-kind').value = j.base_kind;
  $('distal-kind').value = j.distal_kind;
  $('actuation').value = j.actuation.mode;
  if (j.actuation.mode === 'coupled') $('ratio').value = String(j.actuation.ratio);
  readJointControls();
}

function configChanged() {
  readJointControls();
  updateSummaries();
  bus.emit('config', state.config);
}

function updateSummaries() {
  const j = state.config.joints;
  $('sum-hand').textContent = `${state.config.fingers}지 · ${j.segments}관절 · ${j.actuation.mode === 'coupled' ? `결합 ${fmt.n(j.actuation.ratio)}` : '직접'}`;
  const base = state.presets?.bases.find((b) => b.base.kind === state.config.base);
  $('sum-foot').textContent = base ? base.label_ko : state.config.base;
}

// ── 물체 ────────────────────────────────────────────────────────
const currentEntry = () => state.dictById.get($('label').value);

function payload() {
  const e = currentEntry();
  return {
    name: e ? e.label_en : 'object',
    size_m: e ? e.size_m : [0.1, 0.1, 0.1],
    mass_kg: Number($('mass').value),
    friction: Number($('friction').value),
  };
}

const setup = () => ({ z_offset: Number($('zoff').value), contact_margin_m: 0.004, safety_factor: 2.0 });

function populateDictionary() {
  const sel = $('label');
  clear(sel);
  for (const e of state.dict) {
    state.dictById.set(e.id, e);
    sel.append(el('option', { value: e.id, text: `${e.label_ko} · ${e.label_en}${e.graspable ? '' : ' (파지 제외)'}` }));
  }
  sel.value = state.dictById.has('carton_medium') ? 'carton_medium' : (state.dict[0]?.id ?? '');
  applyEntryDefaults();
}

function applyEntryDefaults() {
  const e = currentEntry();
  if (!e) return;
  $('mass').value = String(Math.min(5, Math.max(0.1, e.typical_mass_kg)));
  const [w, h, d] = e.size_m.map((v) => Math.round(v * 1000));
  $('size-note').textContent =
    `${w} × ${h} × ${d} mm, 사전 질량 ${e.typical_mass_kg} kg` +
    (e.graspable ? '' : ' — 이 대상은 파지 대상이 아닙니다 (사전 graspable=false).') +
    ` 출처: ${e.source}`;
  updateReadouts();
}

/** 이미지 분류 결과(최상위 라벨)를 물체 선택에 반영한다. */
export function selectObject(id) {
  if (!state.dictById.has(id)) return false;
  $('label').value = id;
  applyEntryDefaults();
  requestGrasp();
  return true;
}

export function updateReadouts() {
  $('mass-out').textContent = Number($('mass').value).toFixed(2);
  $('friction-out').textContent = Number($('friction').value).toFixed(2);
  $('zoff-out').textContent = Number($('zoff').value).toFixed(3);
  $('closure-out').textContent = Number($('closure').value).toFixed(2);
}

// ── 갱신 ────────────────────────────────────────────────────────
const requestRefresh = debounce(() => refreshAll(), 60);
const requestGrasp = debounce(() => refreshGrasp(), 120);

async function refreshAll() {
  readJointControls();
  updateReadouts();
  await refreshGeometry();
  if ($('show-grasp').checked) await refreshGrasp();
}

const refreshGeometry = coalesce(async () => {
  try {
    state.geometry = await call('hand_geometry', { config: state.config, closure: Number($('closure').value) });
    $('geom-note').textContent =
      `손가락 최대 도달 ${(state.geometry.reach_m * 100).toFixed(1)} cm · ` +
      `손바닥 ${(state.geometry.palm[0] * 100).toFixed(0)} × ${(state.geometry.palm[1] * 100).toFixed(0)} mm`;
  } catch {
    state.geometry = null;
  }
  redrawHand();
});

const refreshGrasp = coalesce(async () => {
  if (!$('show-grasp').checked) {
    state.grasp = null;
  } else {
    try {
      state.grasp = await call('grasp_test', { config: state.config, payload: payload(), setup: setup() });
    } catch {
      state.grasp = null;
    }
  }
  renderGrasp();
  redrawHand();
});

async function refreshDrive() {
  let rows;
  try {
    rows = await call('drive_courses', { course: null });
  } catch {
    return;
  }
  const table = $('drive-table');
  clear(table);
  const head = table.insertRow();
  for (const [h, num] of [['발'], ['통과'], ['막힌 구간'], ['시간 s', true], ['에너지 kJ', true]]) {
    head.append(el('th', { text: h, class: num ? 'num' : null }));
  }
  for (const row of rows) {
    const r = table.insertRow();
    if (row.base.kind === state.config.base) r.className = 'current';
    r.insertCell().textContent = row.label_ko;
    const pass = r.insertCell();
    pass.textContent = row.report.passable ? '통과' : '실패';
    pass.className = row.report.passable ? 'ok' : 'bad';
    const at = row.report.failed_at;
    r.insertCell().textContent = at == null ? '—' : (COURSE_LABELS[at] ?? `구간 ${at + 1}`);
    const t = r.insertCell();
    t.className = 'num';
    t.textContent = row.report.time_s.toFixed(1);
    const e = r.insertCell();
    e.className = 'num';
    e.textContent = (row.report.energy_j / 1000).toFixed(2);
  }
  const passed = rows.filter((r) => r.report.passable).length;
  $('sum-drive').textContent = `통과 ${passed}/${rows.length}`;
}

// ── 파지 결과 ───────────────────────────────────────────────────
function row(label, value) {
  return el('div', { class: 'result-row' }, el('span', { class: 'rlabel', text: label }),
    value instanceof Node ? el('span', { class: 'rvalue' }, value) : el('span', { class: 'rvalue', text: value }));
}

function renderGrasp() {
  const box = $('grasp-results');
  clear(box);
  const g = state.grasp;
  if (!g) {
    box.append(el('p', { class: 'empty-note', text: '파지 시뮬레이션을 켜면 결과가 표시됩니다. (손·발 측면도 화면의 \'물체 파지 시뮬레이션\')' }));
    $('sum-grasp').textContent = '시뮬레이션 꺼짐';
    return;
  }
  const r = g.result;
  const fingerContacts = r.contacts.filter((c) => c.source.source === 'finger').length;
  const palm = r.contacts.some((c) => c.source.source === 'palm');
  box.append(
    row('접촉 (손가락 / 전체)', `${fingerContacts} / ${r.contacts.length}`),
    row('파지 유형', GRASP_TYPE_KO[r.grasp_type] ?? r.grasp_type),
    row('손바닥 지지', palm ? '예' : '아니오'),
    row('힘 닫힘 (진단)', r.force_closed ? '예' : '아니오'),
    row('마찰 용량', `${r.friction_capacity_n.toFixed(1)} N`),
    row('필요 힘 (질량·g·안전계수)', `${r.required_n.toFixed(1)} N`),
    row('최대 파지 질량', `${r.max_payload_kg.toFixed(2)} kg`),
    row('판정', el('span', { class: r.stable ? 'ok' : 'bad', text: r.stable ? '안정 파지' : '불안정' })),
  );
  $('sum-grasp').textContent = `${r.stable ? '안정 파지' : '불안정'} · 최대 ${r.max_payload_kg.toFixed(1)} kg`;
}

// ── 시작 ────────────────────────────────────────────────────────
function bind() {
  for (const id of ['segments', 'base-kind', 'distal-kind', 'actuation']) {
    $(id).addEventListener('change', () => { configChanged(); requestRefresh(); });
  }
  $('ratio').addEventListener('input', () => { configChanged(); requestRefresh(); });
  $('closure').addEventListener('input', () => { updateReadouts(); refreshGeometry(); });
  $('show-grasp').addEventListener('change', () => refreshGrasp());
  $('label').addEventListener('change', () => { applyEntryDefaults(); requestGrasp(); });
  for (const id of ['mass', 'friction', 'zoff']) {
    $(id).addEventListener('input', () => { updateReadouts(); requestGrasp(); });
  }
}

export async function initHandPanel() {
  bind();
  renderGrasp();
  state.presets = await call('presets');
  state.dict = await call('dictionary');
  renderFingerChoices();
  renderJointSelects();
  renderBaseChoices();
  applyJointControls();
  populateDictionary();
  updateSummaries();
  await Promise.allSettled([refreshDrive(), refreshAll()]);
}
