// 작업자 · 협업 트랙: 최초 작업자 주위에 서브 작업자 N명을 추가하고(crew_add), 명령을 나누는 방식(트랙)을 고른다.
//   개별 트랙 — 명령 수량을 작업자 수로 나눠 각자 탐색 → 채취 → 줍기 → 운반 → 인도 (작업자마다 퀘스트 하나).
//   협업 트랙 — ① 편성 ② 상차장 ③ 채취(휴머노이드) ④ 적치 ⑤ 상차(지게차 포크) ⑥ 운반 ⑦ 인도를 한 퀘스트로.
// 오른쪽 '협업 플로우'에는 단계별 수와 작업자별 레인, 흐름 기록을 그린다. 같은 모듈이 눈의 카메라 대수
// (1대 단안 TIPSv2 · 2대 스테레오 SGBM, camera_rig)도 맡는다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { setStatus, toast } from './toast.js';

const FORM_KO = { humanoid: '휴머노이드', forklift: '지게차' };
const STEPS = [
  ['assign', '① 편성'],
  ['bay', '② 상차장'],
  ['harvested', '③ 채취'],
  ['staged', '④ 적치'],
  ['loaded', '⑤ 상차'],
  ['carry', '⑥ 운반'],
  ['delivered', '⑦ 인도'],
];

let lastFetch = 0;
let pending = null;

function workerRow(w) {
  const load = w.carried > 0 ? `${fmt.n(w.carried, w.carried % 1 ? 1 : 0)}개` : '—';
  const share = w.want != null && w.want > 0 ? ` · ${fmt.n(w.done ?? 0, 0)}/${w.want}` : '';
  return el('tr', { class: w.first ? 'first' : '' },
    el('td', { class: 'cw-id', text: w.first ? '1 ★' : String(w.id) }),
    el('td', { text: `${FORM_KO[w.form] ?? w.form}${w.cameras === 2 ? ' · 2캠' : ''}` }),
    el('td', { text: `${w.role_ko ?? '—'}${share}` }),
    el('td', { class: 'cw-stage', text: w.stage_ko ?? w.status_ko ?? '—', title: w.note || w.status_ko || '' }),
    el('td', { class: 'num', text: load }));
}

function renderList(v) {
  const box = $('crew-list');
  if (!box) return;
  clear(box);
  if (!v) return;
  const table = el('table', { class: 'crew-table' },
    el('thead', {}, el('tr', {}, ...['#', '형태', '역할', '지금', '적재'].map((t) => el('th', { text: t })))),
    el('tbody', {}, ...(v.workers ?? []).map(workerRow)));
  box.append(table);
  $('sum-crew').textContent = v.count ? `${v.count}명 추가 · ${v.mode_ko}` : v.mode_ko;
}

function stepState(i, t) {
  const c = t.counts ?? {};
  const n = t.want || 1;
  const counts = [1, t.parked ? 1 : 0, c.harvested / n, c.staged / n, c.loaded / n, c.delivered / n, c.delivered / n];
  const v = Math.max(0, Math.min(1, counts[i] ?? 0));
  return { v, done: v >= 1 - 1e-6 };
}

function renderFlow(v) {
  const box = $('crew-flow');
  if (!box) return;
  clear(box);
  const t = v?.team;
  if (!v || (!t && !(v.count > 0))) {
    box.append(el('p', { class: 'hint', text: '작업자를 추가하면 여기에 협업 흐름이 보입니다. 왼쪽 "작업자 · 협업 트랙"에서 인원과 트랙을 고르세요.' }));
    $('sum-flow').textContent = '';
    return;
  }
  if (!t) {
    box.append(el('p', { class: 'hint', text: v.mode === 'team'
      ? '협업 트랙: 다음 명령이 휴머노이드 채취·적치 → 지게차 상차·운반으로 나뉩니다 (휴머노이드와 지게차가 모두 있어야 함).'
      : `개별 트랙: 다음 명령의 수량을 작업자 ${(v.workers ?? []).length}명이 나눠 각자 끝까지 진행합니다.` }));
  } else {
    const c = t.counts ?? {};
    const n = t.want;
    const first = STEPS.findIndex((_, i) => !stepState(i, t).done);
    const ol = el('ol', { class: 'flow-steps' });
    STEPS.forEach(([key, label], i) => {
      const st = stepState(i, t);
      const cls = t.done || st.done ? 'done' : i === first ? 'active' : 'pending';
      const count = { harvested: c.harvested, staged: c.staged, loaded: c.loaded, delivered: c.delivered, carry: c.delivered }[key];
      const tail = key === 'assign' ? `휴머노이드 ${t.humanoids.length} · 지게차 ${t.forklifts.length}`
        : key === 'bay' ? `${t.bay}${t.parked ? ' · 지게차 집결' : ' · 집결 중'} · 적치 ${t.staged_now}개`
        : `${fmt.n(count ?? 0, 0)}/${n}${key === 'harvested' ? ` · 나무 ${c.felled}` : ''}`;
      ol.append(el('li', { class: `flow-step ${cls}` },
        el('span', { class: 'fs-dot', aria: { hidden: 'true' } }),
        el('span', { class: 'fs-nm', text: label }),
        el('span', { class: 'fs-val', text: tail }),
        el('span', { class: 'bar-pill fs-bar', aria: { hidden: 'true' } }, el('span', { style: `width:${Math.round(st.v * 100)}%` }))));
    });
    box.append(el('div', { class: 'flow-head' },
      el('b', { text: `${t.item_ko} ${n}개 → ${t.dest_ko}` }),
      el('span', { class: `badge ${t.done ? 'badge-ok' : 'badge-info'}`, text: t.done ? '완료' : '진행' }),
      el('span', { class: 'flow-sub', text: `작업자 1: ${t.main_role === 'hauler' ? '상차·운반 (포크 절차)' : '채취·적치 (전체 비전)'} · 깊이 ${t.depth_procedure === 'sgbm' ? 'SGBM 스테레오' : 'TIPSv2 단안'}` })), ol);
  }
  const lanes = el('div', { class: 'flow-lanes' });
  for (const w of v.workers ?? []) {
    lanes.append(el('div', { class: `lane${w.first ? ' first' : ''}` },
      el('span', { class: `lane-tag ${w.form}`, text: w.first ? '1★' : String(w.id) }),
      el('span', { class: 'lane-role', text: w.role_ko ?? '' }),
      el('span', { class: 'lane-now', text: w.stage_ko ?? w.status_ko ?? '', title: w.note || '' })));
  }
  box.append(lanes);
  const notes = (v.notes ?? []).slice(0, 8);
  if (notes.length) {
    box.append(el('ul', { class: 'flow-notes' }, ...notes.map((x) => el('li', {},
      el('span', { class: 'fn-t', text: `${fmt.n(x.time_s, 1)}s` }),
      el('span', { class: 'fn-w', text: x.worker === 0 ? '팀' : `#${x.worker}` }),
      el('span', { class: 'fn-x', text: `${x.step} · ${x.text}` })))));
  }
  $('sum-flow').textContent = t ? `${fmt.n(t.counts?.delivered ?? 0, 0)}/${t.want}${t.done ? ' 완료' : ''}` : v.mode_ko;
}

function syncToolbar(v) {
  const btn = $('tg-coop-agent');
  if (!btn) return;
  const n = (v?.count ?? 0) + 1;
  btn.textContent = `작업자 ${n}명 · ${v?.mode === 'team' ? '협업' : '개별'}`;
  btn.classList.toggle('active', n > 1);
  btn.setAttribute('aria-pressed', n > 1 ? 'true' : 'false');
}

function apply(v) {
  if (!v) return;
  state.crew = v;
  // 모의 백엔드의 옛 순차 협동 모형은 teamSize 로 켜진다: 작업자가 있으면 2.
  state.teamSize = v.count > 0 ? 2 : 1;
  if ($('crew-mode') && v.mode) $('crew-mode').value = v.mode;
  renderList(v);
  renderFlow(v);
  syncToolbar(v);
  bus.emit('crew', v);
}

export async function refreshCrew({ force = false } = {}) {
  const now = Date.now();
  if (!force && now - lastFetch < 400) return state.crew;
  if (pending) return pending;
  lastFetch = now;
  pending = call('crew_status', {}, { quiet: true }).then(apply).catch(() => null).finally(() => { pending = null; });
  return pending;
}

async function addCrew() {
  const btn = $('crew-add');
  btn.disabled = true;
  try {
    const count = Math.max(1, Math.min(8, Math.floor(Number($('crew-count').value) || 1)));
    const r = await call('crew_add', { count, form: $('crew-form').value, cameras: Number($('pp-cameras')?.value || 1), mode: $('crew-mode').value });
    if (r?.snapshot) bus.emit('snapshot', r.snapshot);
    apply(r?.crew);
    const added = (r?.crew?.workers ?? []).filter((w) => !w.first).slice(-count).map((w) => `${w.id}(${FORM_KO[w.form] ?? w.form})`).join(', ');
    toast('작업자 추가', `최초 작업자 주위에 ${added} — ${r?.crew?.mode_ko ?? ''}`, 'ok');
    setStatus(`서브 작업자 ${r?.crew?.count ?? 0}명 · ${r?.crew?.mode_ko ?? ''}`);
  } catch {
    /* api-error 가 표시 */
  } finally {
    btn.disabled = false;
  }
}

async function clearCrew() {
  try {
    const r = await call('crew_clear', {});
    if (r?.snapshot) bus.emit('snapshot', r.snapshot);
    apply(r?.crew);
    toast('작업자', `서브 작업자 ${r?.removed ?? 0}명을 뺐습니다`, 'ok');
  } catch {
    /* api-error 가 표시 */
  }
}

async function setMode(mode) {
  try {
    apply(await call('crew_mode', { mode }));
    setStatus(mode === 'team' ? '협업 트랙: 휴머노이드 채취·적치 → 지게차 상차·운반 (다음 명령부터)' : '개별 트랙: 수량을 작업자마다 나눠 진행 (다음 명령부터)');
  } catch {
    /* api-error 가 표시 */
  }
}

/** 눈의 카메라 대수: 1대(단안, TIPSv2) 또는 2대(스테레오, SGBM · TIPSv2 안 씀). */
async function setCameras(n) {
  try {
    const v = await call('camera_rig', { cameras: n });
    state.models = { ...(state.models ?? {}), ...v };
    bus.emit('models', state.models);
    const st = v?.stereo;
    toast('카메라', n === 2
      ? `스테레오 2대: 깊이는 SGBM (기선 ${fmt.n((st?.baseline_m ?? 0) * 100, 1)} cm, σ ${fmt.n((st?.sigma_m_at?.[0]?.[1] ?? 0) * 1000, 0)} mm @1 m) — TIPSv2를 불러오지 않습니다`
      : '단안 1대: 깊이는 TIPSv2 (지면 정렬) — 모델을 불러와야 합니다', 'ok', 6000);
    bus.emit('camera-rig', n);
  } catch {
    /* api-error 가 표시 */
  }
}

export function initCrewPanel() {
  $('crew-add')?.addEventListener('click', addCrew);
  $('crew-clear')?.addEventListener('click', clearCrew);
  $('crew-mode')?.addEventListener('change', (e) => setMode(e.target.value));
  $('pp-cameras')?.addEventListener('change', (e) => setCameras(Number(e.target.value) || 1));
  $('tg-coop-agent')?.addEventListener('click', () => {
    const next = state.crew?.mode === 'team' ? 'solo' : 'team';
    setMode(next);
  });
  bus.on('snapshot', () => refreshCrew());
  bus.on('world-new', () => refreshCrew({ force: true }));
  bus.on('models', (m) => {
    if (m?.cameras && $('pp-cameras') && Number($('pp-cameras').value) !== m.cameras) $('pp-cameras').value = String(m.cameras);
  });
  renderFlow(null);
  syncToolbar(null);
  return refreshCrew({ force: true });
}
