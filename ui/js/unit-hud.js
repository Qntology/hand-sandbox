// 유닛 HUD: 로봇을 고른 동안만 오른쪽에 뜬다. 위는 신체 상태(body-hud.js), 아래는 탭 둘 —
// 📈 관절 토크 시계열(joint-view.js)과 🖐 부위 접촉력(focus-view.js).
// 이 값들은 world_frame 의 overlay(3인칭) · focus(포커스)에 실려 오므로, 지금 화면이 그 시점이 아니면 아주 작은 프레임을 따로 받는다(초당 2번).
// 관절 동역학은 최초 작업자의 몸에서만 돌기 때문에 서브 작업자에게는 그렇다고 알린다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt, observeResize } from './util.js';
import { initTablist } from './layout.js';
import { isOpen } from './overlays.js';
import { renderJointPanel } from './joint-view.js';
import { DRAWN_PARTS, PART_KO, drawFocus, partAvailable, resolvePart } from './focus-view.js';
import { clearUnit } from './unit-select.js';

const DETAIL_MS = 500;
const SUB_NOTE = '서브 작업자는 관절 동역학을 돌리지 않아 이 값이 없습니다. 최초 작업자(★)를 고르면 보입니다.';
const DETAIL_FRAME = { width: 64, height: 48, quality: 30 };

let tabs = null;
let pending = false;
let lastKey = '';
let lastFocus = null;

const isSub = () => {
  const id = state.selectedUnit;
  const u = state.unitInfo.get(id) ?? state.units.find((x) => x.id === id);
  return !!u && !u.first;
};

// ── 부위 포커스 ─────────────────────────────────────────────────
/** 카메라마다 먼저 보일 부위: 손 근접 → 손 모양, 옆 → 허리, 낮게 → 발·바퀴·트랙 중 이 몸에 있는 것. */
function defaultPart(cam, f) {
  if (cam === 'hand') return 'hand';
  if (cam === 'side') return 'waist';
  return ['foot', 'wheel', 'track'].find((p) => partAvailable(f, p)) ?? 'foot';
}

function focusCam() {
  const view = $('view-select').value;
  return view.startsWith('focus') ? view.split(':')[1] ?? 'hand' : 'hand';
}

function renderFocus() {
  const f = lastFocus;
  if ($('focus-hud').hidden) return;
  const cap = clear($('fh-cap'));
  if (isSub()) {
    drawFocus($('fh-canvas'), null, 'hand');
    cap.append(SUB_NOTE);
    return;
  }
  if (!f) {
    drawFocus($('fh-canvas'), null, 'hand');
    return;
  }
  const pick = $('fh-part').value || 'auto';
  // 자동: 지금 행위·사건이 다루는 부위, 없으면 카메라의 부위.
  const part = pick === 'auto' ? ((f.parts ?? []).length ? resolvePart('auto', f) : defaultPart(focusCam(), f)) : pick;
  $('fh-title').textContent = `부위 포커스 · ${PART_KO[part] ?? part}`;
  drawFocus($('fh-canvas'), f, part);
  cap.append(`${f.action_ko ?? '행위 없음'}${f.phase ? ` · ${f.phase}` : ''} · 안정 여유 ${fmt.n(f.body?.margin, 2)} · 몸통 ${fmt.n(f.body?.trunk_deg, 0)}°`);
  if (f.event) cap.append(el('div', { class: f.event_ok === false ? 'fail' : '', text: `${f.event}` }));
}

// ── 폴링 ────────────────────────────────────────────────────────
function onDetail({ overlay, focus }) {
  if (state.selectedUnit == null || isSub()) return;
  if (overlay) renderJointPanel(overlay);
  if (focus) {
    lastFocus = focus;
    renderFocus();
  }
}

/** 지금 탭에 필요한 작은 프레임의 시점. 화면 시점이 이미 같은 값을 주거나 받을 것이 없으면 null. */
function detailView() {
  const view = $('view-select').value;
  if (isSub()) return null;
  if (tabs.current() === 'joint') return view === 'chase' ? null : `chase:${$('orbit-range').value}`;
  return view.startsWith('focus') ? null : 'focus:hand';
}

async function pollDetail() {
  if (pending || state.selectedUnit == null || document.hidden || !state.snapshot || isOpen('analysis-modal')) return;
  const view = detailView();
  if (!view) return;
  const key = `${state.snapshot.tick}|${view}`;
  if (key === lastKey) return;
  pending = true;
  try {
    const f = await call('world_frame', { ...DETAIL_FRAME, view }, { quiet: true });
    lastKey = key;
    onDetail({ overlay: f.overlay ?? null, focus: f.focus ?? null });
  } catch {
    /* 다음 주기에 다시 */
  } finally {
    pending = false;
  }
}

// ── 표시 ────────────────────────────────────────────────────────
function syncHud(id) {
  const on = id != null;
  $('unit-hud').hidden = !on;
  $('stage').classList.toggle('has-unit', on);
  lastKey = '';
  lastFocus = null;
  if (!on) return;
  renderJointPanel(null, isSub() ? SUB_NOTE : '관절 값을 받는 중입니다…');
  renderFocus();
  pollDetail();
}

export function initUnitHud() {
  tabs = initTablist($('unit-tabs'), () => { lastKey = ''; renderFocus(); pollDetail(); });
  const sel = $('fh-part');
  sel.append(el('option', { value: 'auto', text: '자동 (행위 부위)' }), ...DRAWN_PARTS.map((p) => el('option', { value: p, text: PART_KO[p] })));
  sel.addEventListener('change', renderFocus);
  observeResize($('fh-canvas'), renderFocus);
  $('bh-toggle').addEventListener('click', () => {
    const folded = $('body-hud').classList.toggle('folded');
    $('bh-toggle').setAttribute('aria-expanded', String(!folded));
    $('bh-toggle').setAttribute('aria-label', folded ? '신체 상태 펼치기' : '신체 상태 접기');
    $('bh-toggle').textContent = folded ? '▸' : '▾';
  });
  $('unit-hud-close').addEventListener('click', clearUnit);
  bus.on('unit', syncHud);
  bus.on('unit-detail', onDetail);
  setInterval(pollDetail, DETAIL_MS);
}
