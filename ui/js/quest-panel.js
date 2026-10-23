// 퀘스트: 오른쪽 퀘스트 카드 목록, 인벤토리, 월드 화면의 퀘스트 트래커.
// 매 틱 갱신되므로 카드별 서명(signature)이 같으면 DOM 을 그대로 두어 포커스와 깜빡임을 지킨다.
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { panelOpen } from './layout.js';
import { toast } from './toast.js';

const STATE_KO = { pending: '대기', active: '진행 중', done: '완료', failed: '실패' };
const STATE_BADGE = { pending: 'badge-dim', active: 'badge-info', done: 'badge-ok', failed: 'badge-crit' };
const ICON = { pending: '', active: '●', done: '✓', failed: '!' };

const cards = new Map(); // quest id → {node, sig, state}
const expanded = new Map(); // quest id → boolean (사용자가 바꾼 값)
const prevState = new Map();
let firstLoad = true;

const overall = (q) => {
  if (!q.steps?.length) return q.state === 'done' ? 1 : 0;
  const sum = q.steps.reduce((a, s) => a + (s.state === 'done' ? 1 : s.state === 'pending' ? 0 : Math.max(0, Math.min(1, s.progress ?? 0))), 0);
  return sum / q.steps.length;
};

const isOpen = (q) => expanded.get(q.id) ?? (q.state === 'active' || q.state === 'pending');

function signature(q) {
  return JSON.stringify([
    q.state, isOpen(q), q.result ?? null,
    (q.steps ?? []).map((s) => [s.state, Math.round((s.progress ?? 0) * 100), s.note ?? '', s.label_ko]),
  ]);
}

function stepNode(s) {
  const pct = Math.round((s.progress ?? 0) * 100);
  const showBar = s.state === 'active';
  return el('li', { class: `quest-step ${s.state}` },
    el('span', { class: 'ico', text: ICON[s.state] ?? '', aria: { hidden: 'true' } }),
    el('span', { text: s.label_ko }),
    el('span', { class: 'pct', text: s.state === 'active' ? `${pct}%` : (STATE_KO[s.state] ?? '') }),
    showBar ? el('div', { class: 'bar-pill', role: 'progressbar', aria: { valuemin: '0', valuemax: '100', valuenow: String(pct), label: `${s.label_ko} 진행률` } }, el('span', { style: `width:${pct}%` })) : null,
    s.note ? el('span', { class: 'note', text: s.note }) : null);
}

function resultNode(q) {
  const r = q.result;
  return el('div', { class: 'quest-result' },
    el('b', { text: q.state === 'failed' ? '실패 요약' : '완료 요약' }),
    r.summary ? el('div', { text: r.summary }) : null,
    el('div', { class: 'rk' },
      el('span', { text: `소요 ${r.ticks} 틱` }),
      el('span', { text: `에너지 ${fmt.n(r.energy_j / 1000, 2)} kJ` }),
      el('span', { text: `정밀도 ${r.precision == null ? '—' : fmt.pct(r.precision)}` })));
}

function buildCard(q) {
  const open = isOpen(q);
  const finished = q.state === 'done' || q.state === 'failed';
  const pct = Math.round(overall(q) * 100);
  return el('article', { class: `quest-card ${q.state}`, dataset: { quest: q.id }, aria: { label: `퀘스트 ${q.title}` } },
    el('div', { class: 'quest-head' },
      el('div', { class: 'quest-title', text: q.title }),
      el('span', { class: `badge ${STATE_BADGE[q.state] ?? 'badge-dim'}`, text: STATE_KO[q.state] ?? q.state })),
    el('div', { class: 'quest-meta', text: `${q.id} · 생성 틱 ${q.created_tick} · ${pct}%` }),
    el('div', { class: `bar-pill${q.state === 'done' ? ' lv-ok' : q.state === 'failed' ? ' lv-crit' : ''}`, aria: { hidden: 'true' } }, el('span', { style: `width:${pct}%` })),
    open ? el('ol', { class: 'quest-steps' }, (q.steps ?? []).map(stepNode)) : null,
    finished && q.result ? resultNode(q) : null,
    finished ? el('button', {
      type: 'button', class: 'quest-toggle', text: open ? '단계 접기' : '단계 보기', aria: { expanded: String(open) },
      on: { click: () => { expanded.set(q.id, !open); renderList(state.quests); } },
    }) : null);
}

const order = (a, b) => {
  const rank = { active: 0, pending: 1, failed: 2, done: 2 };
  return (rank[a.state] - rank[b.state]) || (a.state === 'pending' ? a.created_tick - b.created_tick : b.created_tick - a.created_tick);
};

function renderList(quests) {
  const box = $('quest-list');
  const sorted = [...quests].sort(order);
  const wanted = [];
  const live = new Set();
  for (const q of sorted) {
    live.add(q.id);
    const sig = signature(q);
    let entry = cards.get(q.id);
    if (!entry || entry.sig !== sig) {
      const node = buildCard(q);
      if (entry && entry.state !== q.state) node.classList.add('flash');
      entry?.node.replaceWith(node);
      entry = { node, sig, state: q.state };
      cards.set(q.id, entry);
    }
    wanted.push(entry.node);
  }
  for (const id of [...cards.keys()]) {
    if (!live.has(id)) { cards.get(id).node.remove(); cards.delete(id); }
  }
  if (!sorted.length) {
    box.replaceChildren(el('p', { class: 'empty-note', text: '아직 퀘스트가 없습니다. 월드 화면 아래 입력줄에 명령을 입력하세요. 예: 철광석 3개 캐서 베이스캠프로 가져와' }));
    cards.clear();
  } else if (wanted.length !== box.children.length || wanted.some((n, i) => box.children[i] !== n)) {
    box.replaceChildren(...wanted);
  }
  const activeN = quests.filter((q) => q.state === 'active' || q.state === 'pending').length;
  $('sum-quests').textContent = quests.length ? `진행 ${activeN} · 전체 ${quests.length}` : '';
}

function notifyChanges(quests) {
  for (const q of quests) {
    const before = prevState.get(q.id);
    if (!firstLoad && before && before !== q.state) {
      if (q.state === 'done') toast('퀘스트 완료', q.title, 'ok');
      else if (q.state === 'failed') toast('퀘스트 실패', `${q.title}\n${q.steps?.find((s) => s.state === 'failed')?.note ?? ''}`.trim(), 'err');
    }
    prevState.set(q.id, q.state);
  }
  firstLoad = false;
}

// ── 트래커(월드 화면 우상단) ────────────────────────────────────
function renderTracker(quests) {
  const q = quests.find((x) => x.state === 'active') ?? quests.find((x) => x.state === 'pending')
    ?? [...quests].sort((a, b) => b.created_tick - a.created_tick)[0];
  const badge = $('qt-state');
  if (!q) {
    $('qt-title').textContent = '진행 중인 퀘스트 없음';
    $('qt-step').textContent = '아래 입력줄에 명령을 입력하세요.';
    $('qt-bar').style.width = '0%';
    badge.textContent = '없음';
    badge.className = 'badge badge-dim';
    $('quest-tracker').dataset.quest = '';
    return;
  }
  const step = q.steps?.find((s) => s.state === 'active') ?? q.steps?.find((s) => s.state === 'failed') ?? q.steps?.find((s) => s.state === 'pending');
  const done = q.steps?.filter((s) => s.state === 'done').length ?? 0;
  $('qt-title').textContent = q.title;
  $('qt-step').textContent = q.state === 'done' ? '모든 단계를 마쳤습니다'
    : step ? `${done + 1}/${q.steps.length} ${step.label_ko}${step.state === 'active' ? ` · ${Math.round((step.progress ?? 0) * 100)}%` : ''}` : '';
  $('qt-bar').style.width = `${Math.round(overall(q) * 100)}%`;
  badge.textContent = STATE_KO[q.state] ?? q.state;
  badge.className = `badge ${STATE_BADGE[q.state] ?? 'badge-dim'}`;
  $('quest-tracker').dataset.quest = q.id;
}

function focusQuest() {
  const id = $('quest-tracker').dataset.quest;
  panelOpen('panelRight');
  const entry = id ? cards.get(id) : null;
  if (!entry) return;
  entry.node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  entry.node.classList.remove('flash');
  void entry.node.offsetWidth; // 강제 리플로우: 같은 애니메이션을 다시 재생
  entry.node.classList.add('flash');
}

// ── 인벤토리 ────────────────────────────────────────────────────
// carrying[].mass_kg 는 "개당 질량"으로 해석한다(계약상 모호). 합계는 개수 × 개당 질량.
function renderInventory(snap) {
  const items = snap?.agent?.carrying ?? [];
  const list = $('inv-list');
  clear(list);
  if (!items.length) list.append(el('li', {}, el('span', { class: 'empty-note', text: '들고 있는 물건이 없습니다' })));
  let total = 0;
  for (const it of items) {
    total += it.count * it.mass_kg;
    list.append(el('li', {}, el('span', { text: `${it.label_ko} × ${it.count}` }), el('span', { class: 'mass', text: `개당 ${fmt.n(it.mass_kg, 1)} kg` })));
  }
  const stomach = state.body?.parts?.find((p) => p.id === 'stomach');
  const limit = stomach?.max;
  $('inv-load').textContent = limit ? `${fmt.n(total, 1)} / ${fmt.n(limit, 1)} kg` : `${fmt.n(total, 1)} kg`;
  const bar = $('inv-bar');
  const r = limit ? Math.min(1, total / limit) : 0;
  bar.style.width = `${Math.round(r * 100)}%`;
  bar.parentElement.className = `bar-pill ${r > 0.9 ? 'lv-crit' : r > 0.7 ? 'lv-warn' : ''}`;
  $('sum-inventory').textContent = items.length ? `${items.reduce((a, i) => a + i.count, 0)}개 · ${fmt.n(total, 1)} kg` : '비어 있음';
}

export function initQuestPanel() {
  const tracker = $('quest-tracker');
  tracker.addEventListener('click', focusQuest);
  tracker.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); focusQuest(); }
  });
  bus.on('quests', (qs) => { notifyChanges(qs); renderList(qs); renderTracker(qs); });
  bus.on('snapshot', renderInventory);
  bus.on('body', () => renderInventory(state.snapshot));
  renderList([]);
  renderTracker([]);
  renderInventory(null);
}
