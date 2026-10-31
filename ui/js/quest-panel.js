// 퀘스트: 명령줄 위 고정 헤더(#quest-bar)와 그 헤더가 여는 Drop-up 의 퀘스트 카드 목록, 인벤토리(내려놓기 버튼), 보관 더미.
// 매 틱 갱신되므로 카드별 서명(signature)이 같으면 DOM 을 그대로 두어 포커스와 깜빡임을 지킨다.
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { togglePanel } from './overlays.js';
import { toast } from './toast.js';
import { submitCommand } from './command-bar.js';

const ITEM_COLOR = { log: '#805c36', stone: '#808084', coal: '#3a3a40', iron_ore: '#ba8c70' };
/** 개수 표기: 정수면 그대로, 쪼갠 조각이면 소수 둘째 자리까지. */
const amountText = (a) => (Math.abs(a - Math.round(a)) < 1e-3 ? String(Math.round(a)) : String(Number(a.toFixed(2))));

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
  // 누가 하는가: 0 팀의 협업 흐름, 1 최초 작업자(전체 비전), 2… 서브 작업자 레인.
  const coopTag = q.owner === 0 ? ' · 협업 흐름 (팀)'
    : q.track ? ` · ${q.track === 'team' ? '협업' : '개별'} · ${q.owner === 1 ? '작업자 1 (전체 비전)' : `작업자 ${q.owner}`}`
    : '';
  return el('article', { class: `quest-card ${q.state}`, dataset: { quest: q.id }, aria: { label: `퀘스트 ${q.title}` } },
    el('div', { class: 'quest-head' },
      el('div', { class: 'quest-title', text: q.title }),
      el('span', { class: `badge ${STATE_BADGE[q.state] ?? 'badge-dim'}`, text: STATE_KO[q.state] ?? q.state })),
    el('div', { class: 'quest-meta', text: `${q.id} · 생성 틱 ${q.created_tick} · ${pct}%${coopTag}` }),
    el('div', { class: `bar-pill${q.state === 'done' ? ' lv-ok' : q.state === 'failed' ? ' lv-crit' : ''}`, aria: { hidden: 'true' } },
      el('span', { style: `width:${pct}%` })),
    open ? el('ol', { class: 'quest-steps' }, (q.steps ?? []).map(stepNode)) : null,
    finished && q.result ? resultNode(q) : null,
    finished ? el('button', {
      type: 'button',
      class: 'quest-toggle',
      text: open ? '단계 접기' : '단계 보기',
      aria: { expanded: String(open) },
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
    box.replaceChildren(el('p', { class: 'empty-note', text: '아직 퀘스트가 없습니다. 아래 입력줄에 명령을 입력하세요. 예: 철광석 3개 캐서 베이스캠프로 가져와' }));
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

// ── 고정 헤더(명령줄 위) ────────────────────────────────────────
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
    return;
  }
  const step = q.steps?.find((s) => s.state === 'active') ?? q.steps?.find((s) => s.state === 'failed') ?? q.steps?.find((s) => s.state === 'pending');
  const done = q.steps?.filter((s) => s.state === 'done').length ?? 0;
  $('qt-title').textContent = q.title;
  $('qt-step').textContent = q.state === 'done' ? '모든 단계를 마쳤습니다'
    : step ? `${done + 1}/${q.steps.length} ${step.label_ko} (${Math.round(overall(q) * 100)}%)` : '';
  $('qt-bar').style.width = `${Math.round(overall(q) * 100)}%`;
  badge.textContent = STATE_KO[q.state] ?? q.state;
  badge.className = `badge ${STATE_BADGE[q.state] ?? 'badge-dim'}`;
}

// ── 인벤토리 ────────────────────────────────────────────────────
// carrying[].mass_kg 는 "개당 질량", amount 는 쪼갠 조각까지 센 개수(0.5 = 돌 반 개), total_kg 는 합계.
let invSig = '';

/** 내려놓기: 명령과 같은 길(command_submit)로 보내 대기·기록·로그가 같게 한다. */
async function dropFromInventory(it, all) {
  const text = it ? `${it.label_ko} ${all ? '' : '1개 '}내려놔`.replace(/\s+/g, ' ') : '다 내려놔';
  const r = await submitCommand(text, { quietHistory: true });
  if (r && !r.ok) toast('내려놓기 실패', r.error ?? '명령을 만들지 못했습니다', 'err');
}

function renderInventory(snap) {
  const items = snap?.agent?.carrying ?? [];
  const list = $('inv-list');
  const sig = JSON.stringify(items.map((it) => [it.item_id, it.amount ?? it.count]));
  let total = 0;
  for (const it of items) total += it.total_kg ?? it.count * it.mass_kg;
  if (sig !== invSig) {
    invSig = sig;
    clear(list);
    if (!items.length) list.append(el('li', {}, el('span', { class: 'empty-note', text: '들고 있는 물건이 없습니다' })));
    for (const it of items) {
      const n = it.amount ?? it.count;
      list.append(el('li', { class: 'inv-item' },
        el('span', { class: 'inv-name', text: `${it.label_ko} × ${amountText(n)}` }),
        el('span', { class: 'mass', text: `개당 ${fmt.n(it.mass_kg, 1)} kg · ${fmt.n(it.total_kg ?? n * it.mass_kg, 1)} kg` }),
        el('span', { class: 'inv-btns' },
          el('button', { type: 'button', class: 'vt-btn', text: '1개 내려놓기', title: `${it.label_ko} 1개를 내려놓습니다 (웨이포인트 가까이면 그 더미에, 아니면 땅에)`, on: { click: () => dropFromInventory(it, false) } }),
          n > 1 + 1e-3 ? el('button', { type: 'button', class: 'vt-btn', text: '모두', title: `${it.label_ko} ${amountText(n)}개를 모두 내려놓습니다`, on: { click: () => dropFromInventory(it, true) } }) : null)));
    }
    $('inv-drop-all').disabled = !items.length;
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

// ── 보관 더미 (웨이포인트 적재) ────────────────────────────────
let stockSig = '';

function renderStock(snap) {
  const box = $('stock-list');
  const wps = snap?.waypoints ?? [];
  const q = (state.quests ?? []).find((x) => x.state === 'active');
  const stepLabel = q?.steps?.find((x) => x.state === 'active')?.label_ko ?? '';
  const sig = JSON.stringify([wps.map((w) => [w.id, (w.stock ?? []).map((s) => [s.item_id, s.amount ?? s.count])]), stepLabel]);
  if (sig === stockSig) return;
  stockSig = sig;
  clear(box);
  let pieces = 0;
  let kg = 0;
  for (const w of wps) {
    const stock = w.stock ?? [];
    const wkg = stock.reduce((a, s) => a + (s.total_kg ?? (s.amount ?? s.count) * s.mass_kg), 0);
    const wn = stock.reduce((a, s) => a + (s.amount ?? s.count), 0);
    pieces += wn;
    kg += wkg;
    const flag = w.id === 'warehouse' ? '#3468dc' : '#e87828';
    box.append(el('div', { class: `stock-card${stepLabel.includes(w.label_ko) ? ' target' : ''}` },
      el('div', { class: 'sc-head' },
        el('span', {}, el('i', { class: 'sc-sw', style: `background:${flag}` }), w.label_ko),
        el('small', { text: stock.length ? `${amountText(wn)}개 · ${fmt.n(wkg, 1)} kg` : '비어 있음' })),
      stock.length
        ? el('ul', {}, ...stock.map((s) => {
          const n = s.amount ?? s.count;
          const whole = Math.floor(n + 1e-3);
          const part = n - whole > 1e-3;
          const dots = [];
          for (let i = 0; i < Math.min(whole, 30); i++) dots.push(el('i', { style: `background:${ITEM_COLOR[s.item_id] ?? '#999'}` }));
          if (part) dots.push(el('i', { class: 'part', style: `background:${ITEM_COLOR[s.item_id] ?? '#999'}` }));
          return el('li', {}, el('span', { text: s.label_ko }), el('span', { class: 'pile', aria: { hidden: 'true' } }, ...dots), el('span', { class: 'n', text: `× ${amountText(n)}` }));
        }))
        : el('div', { class: 'sc-empty', text: '아직 쌓인 물건이 없습니다' })));
  }
  $('sum-stock').textContent = pieces > 0 ? `${amountText(pieces)}개 · ${fmt.n(kg, 1)} kg` : '비어 있음';
}

export function initQuestPanel() {
  $('inv-drop-all').addEventListener('click', () => dropFromInventory(null, true));
  bus.on('snapshot', renderStock);
  bus.on('quests', () => renderStock(state.snapshot));
  $('quest-bar').addEventListener('click', () => togglePanel('quest-dropup'));
  bus.on('panel', ({ id, open }) => {
    if (id === 'quest-dropup') $('qb-more').textContent = open ? '협업·인벤 닫기 ▼' : '협업·인벤 상세 ▲';
  });
  bus.on('quests', (qs) => { notifyChanges(qs); renderList(qs); renderTracker(qs); });
  bus.on('snapshot', renderInventory);
  bus.on('body', () => renderInventory(state.snapshot));
  renderList([]);
  renderTracker([]);
  renderInventory(null);
  renderStock(null);
}
