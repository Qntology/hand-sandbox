// 유닛 선택: 탑뷰에서 로봇을 누르면 고르고, 빈 땅을 누르거나 ESC 면 바로 푼다.
// 고른 동안 하단 명령줄은 [🛠 유닛#N ✕] [⚙️ 프리셋 ▲] + AI(MCP) 3D 외형 수정 프롬프트로 바뀌고, 오른쪽에 유닛 HUD 가 뜬다(unit-hud.js).
// 탑뷰가 아닌 시점은 프레임에 로봇 좌표가 없고 카메라가 최초 작업자를 따라가므로, 화면을 누르면 최초 작업자를 고르거나 푼다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $ } from './util.js';
import { closePanel, togglePanel } from './overlays.js';
import { frameRect } from './world-view.js';

const HIT_PX = 18;
const INFO_MS = 1000;
const FIRST_ID = 1;
let pulling = false;

/** world-wrap 기준 좌표에 가장 가까운 로봇 (HIT_PX 안에 없으면 null). */
function unitAt(x, y) {
  const r = frameRect();
  let best = null;
  let bestD = HIT_PX;
  for (const u of state.units) {
    const d = Math.hypot(r.left + u.u * r.width - x, r.top + u.v * r.height - y);
    if (d <= bestD) {
      best = u;
      bestD = d;
    }
  }
  return best;
}

function syncCommandBar() {
  const id = state.selectedUnit;
  const on = id != null;
  $('unit-tools').hidden = !on;
  $('command-input').hidden = on;
  $('rig-input').hidden = !on;
  $('command-send').textContent = on ? '적용' : '실행';
  if (on) $('unit-tag').textContent = `🛠 유닛#${id} ✕`;
}

export function selectUnit(id) {
  if (state.selectedUnit === id) return;
  state.selectedUnit = id;
  syncCommandBar();
  bus.emit('unit', id);
  refreshUnits();
}

export function clearUnit() {
  if (state.selectedUnit == null) return;
  state.selectedUnit = null;
  closePanel('preset-dropup');
  syncCommandBar();
  bus.emit('unit', null);
}

/** units_list 로 배터리·체형까지 받아 둔다. 고른 유닛이 사라졌으면 선택을 푼다. */
export async function refreshUnits() {
  if (pulling || !state.snapshot) return;
  pulling = true;
  try {
    const r = await call('units_list', {}, { quiet: true });
    state.unitInfo = new Map((r.units ?? []).map((u) => [u.id, u]));
    if (state.selectedUnit != null && !state.unitInfo.has(state.selectedUnit)) clearUnit();
    bus.emit('unit-info', state.unitInfo);
  } catch {
    /* units_list 가 없는 백엔드: 프레임의 units 만으로 동작한다 */
  } finally {
    pulling = false;
  }
}

function wrapPoint(e) {
  const r = $('world-wrap').getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function onWorldClick(e) {
  if (e.target.closest('.cctv, .unit-tag-pin')) return;
  const hit = $('view-select').value === 'top'
    ? unitAt(...wrapPoint(e))
    : (state.selectedUnit == null ? { id: state.unitInfo.keys().next().value ?? FIRST_ID } : null);
  if (hit) {
    selectUnit(hit.id);
    return;
  }
  closePanel('quest-dropup');
  clearUnit();
}

export function initUnitSelect() {
  const wrap = $('world-wrap');
  wrap.addEventListener('click', onWorldClick);
  wrap.addEventListener('pointermove', (e) => {
    wrap.classList.toggle('can-pick', !!unitAt(...wrapPoint(e)));
  });
  $('unit-tag').addEventListener('click', clearUnit);
  $('preset-toggle').addEventListener('click', () => togglePanel('preset-dropup'));
  bus.on('panel', ({ id, open }) => {
    if (id === 'preset-dropup') $('preset-toggle').textContent = `⚙️ 프리셋 ${open ? '▼' : '▲'}`;
  });
  bus.on('escape', clearUnit);
  bus.on('world-new', () => { clearUnit(); refreshUnits(); });
  // crew 는 재생 중 자주 오므로 인원이 바뀔 때만 다시 받는다.
  let crewCount = 0;
  bus.on('crew', (v) => {
    if (v.count === crewCount) return;
    crewCount = v.count;
    refreshUnits();
    bus.emit('frame-stale');
  });
  setInterval(() => { if (state.selectedUnit != null && !document.hidden) refreshUnits(); }, INFO_MS);
  syncCommandBar();
}
