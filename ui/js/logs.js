// 하단 허브의 실시간 로그: log_tail 증분 조회(afterSeq), 범주 필터 칩, 수준별 색, 자동 스크롤.
// 필터는 화면에서만 걸고 커서는 하나(범주 null)만 쓴다. 그래서 칩을 껐다 켜도 다시 조회하지 않는다.
// 평소에는 같은 목록의 마지막 3줄만 보이고, 마우스를 올리면 그 자리에서 위로 펼쳐져 전체 로그와 칩이 나타난다(나가면 1.8초 뒤 복귀).
// 줄을 누르면 그 줄의 범주에 맞는 기록·분석 탭이 열린다. data 가 있는 줄은 { } 로 원본을 펼친다.
import { call } from './api.js';
import { bus } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { TAB_LINK_KO, openAnalysis, tabForCat } from './analysis-modal.js';

const CATS = ['world', 'quest', 'search', 'vision', 'sds', 'ts', 'embed', 'device', 'physics', 'mcp'];
const POLL_MS = 1200;
const BATCH = 200;
const MAX_BUF = 3000;
const MAX_DOM = 600;
const SHRINK_MS = 1800;

const enabled = new Set(CATS);
let buf = [];
let lastSeq = 0;
let pulling = false;
let failed = false;
let localSeq = 0;

const passes = (l) => (!CATS.includes(l.cat) || enabled.has(l.cat)) && ($('log-debug').checked || l.level !== 'debug');
const expanded = () => $('log-feed').classList.contains('expanded');

function kvSummary(data) {
  const parts = [];
  for (const [k, v] of Object.entries(data ?? {})) {
    if (parts.length >= 6) break;
    if (v == null || typeof v === 'object') continue;
    parts.push(`${k}=${typeof v === 'number' && !Number.isInteger(v) ? Number(v.toPrecision(3)) : v}`);
  }
  return parts.join(' ');
}

function buildRow(l) {
  const hasData = l.data && Object.keys(l.data).length > 0;
  const tab = tabForCat(l.cat);
  const msg = el('span', { class: 'msg', text: l.msg, title: l.msg });
  const kv = hasData ? kvSummary(l.data) : '';
  if (kv) msg.append(' ', el('span', { class: 'kv', text: kv }));
  const open = () => openAnalysis(tab);
  const node = el('div', {
    class: `log-line ${l.level}`, dataset: { cat: l.cat, seq: l.seq }, role: 'button', tabindex: 0,
    aria: { label: `${l.cat} 로그 — 누르면 ${TAB_LINK_KO[tab].slice(3)}` },
    on: {
      click: open,
      keydown: (e) => {
        if (e.target === node && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); open(); }
      },
    },
  },
  el('span', { class: 'ts', text: `💬 ${fmt.time(l.ts)}` }),
  el('span', { class: 'cat', text: `[${l.cat}]` }),
  msg,
  el('span', { class: 'lf-open' },
    hasData ? el('button', { type: 'button', class: 'lf-raw', text: '{ }', title: '원본 데이터 펼치기', aria: { expanded: 'false' }, on: { click: (e) => toggleRaw(e, node, l) } }) : null,
    el('span', { text: TAB_LINK_KO[tab] })));
  return node;
}

function toggleRaw(e, node, l) {
  e.stopPropagation();
  const btn = e.currentTarget;
  const open = btn.getAttribute('aria-expanded') === 'true';
  btn.setAttribute('aria-expanded', String(!open));
  if (open) node.querySelector('.raw')?.remove();
  else node.append(el('div', { class: 'raw', text: JSON.stringify(l.data, null, 1) }));
}

function updateCount() {
  const shown = buf.filter(passes).length;
  $('log-count').textContent = `${shown} / ${buf.length}줄`;
}

/** 접혀 있으면 언제나 마지막 줄을 보이고, 펼쳐 있으면 '자동 스크롤'을 따른다. */
function scrollEnd(force = false) {
  if (!force && expanded() && !$('log-auto').checked) return;
  const v = $('log-view');
  v.scrollTop = v.scrollHeight;
}

function renderAll() {
  const view = $('log-view');
  clear(view);
  const rows = buf.filter(passes).slice(-MAX_DOM);
  if (!rows.length) view.append(el('div', { class: 'log-empty', text: buf.length ? '선택한 범주에 해당하는 로그가 없습니다.' : '아직 로그가 없습니다.' }));
  for (const l of rows) view.append(buildRow(l));
  updateCount();
  scrollEnd();
}

function append(lines) {
  const view = $('log-view');
  view.querySelector('.log-empty')?.remove();
  for (const l of lines) {
    buf.push(l);
    if (passes(l)) view.append(buildRow(l));
  }
  if (buf.length > MAX_BUF) buf = buf.slice(-MAX_BUF);
  while (view.children.length > MAX_DOM) view.firstElementChild.remove();
  updateCount();
  scrollEnd();
}

/** 런타임 밖에서 생긴 줄(AI 수정 단계 등)을 같은 로그에 끼운다. 커서(lastSeq)는 건드리지 않는다. */
export function logLocal(cat, level, msg, data = null) {
  localSeq += 1;
  append([{ seq: `local-${localSeq}`, ts: new Date().toISOString(), cat, level, msg, data }]);
}

export async function pullLogs() {
  if (pulling) return;
  pulling = true;
  try {
    for (let i = 0; i < 10; i++) {
      const r = await call('log_tail', { afterSeq: lastSeq, limit: BATCH, category: null }, { quiet: true });
      const lines = (r.lines ?? []).filter((l) => l.seq > lastSeq);
      if (!lines.length && typeof r.next_seq === 'number' && r.next_seq < lastSeq) {
        // 백엔드가 순번을 처음부터 다시 매기기 시작했다(새 세션). 버퍼를 비우고 처음부터 읽는다.
        lastSeq = 0;
        buf = [];
        renderAll();
        continue;
      }
      if (!lines.length) break;
      lastSeq = lines[lines.length - 1].seq;
      append(lines);
      if ((r.lines ?? []).length < BATCH) break;
    }
    failed = false;
  } catch (e) {
    if (!failed) {
      failed = true;
      append([{ seq: lastSeq, ts: '', cat: 'device', level: 'error', msg: `log_tail 조회 실패: ${e.message}` }]);
    }
  } finally {
    pulling = false;
  }
}

function renderChips() {
  const box = $('log-chips');
  clear(box);
  for (const c of CATS) {
    box.append(el('button', {
      type: 'button', class: 'chip-btn', text: c, dataset: { cat: c }, aria: { pressed: String(enabled.has(c)) },
      on: {
        click: (e) => {
          if (enabled.has(c)) enabled.delete(c); else enabled.add(c);
          e.currentTarget.setAttribute('aria-pressed', String(enabled.has(c)));
          renderAll();
        },
      },
    }));
  }
}

// ── 3줄 ↔ 전체 로그 ─────────────────────────────────────────────
function initFeed() {
  const feed = $('log-feed');
  const view = $('log-view');
  let timer = null;
  const setExpanded = (on) => {
    clearTimeout(timer);
    feed.classList.toggle('expanded', on);
    scrollEnd(true);
  };
  const leave = () => {
    clearTimeout(timer);
    timer = setTimeout(() => setExpanded(false), SHRINK_MS);
  };
  feed.addEventListener('mouseenter', () => setExpanded(true));
  feed.addEventListener('mouseleave', () => { if (!feed.contains(document.activeElement)) leave(); });
  feed.addEventListener('focusin', () => setExpanded(true));
  feed.addEventListener('focusout', (e) => { if (!feed.contains(e.relatedTarget) && !feed.matches(':hover')) leave(); });
  // 높이 변화가 끝나면 한 번 더 내려 마지막 줄이 제자리에 오게 한다.
  view.addEventListener('transitionend', () => scrollEnd(true));
  bus.on('panel', ({ id, open }) => { if (id === 'analysis-modal' && open) { document.activeElement?.blur?.(); setExpanded(false); } });
}

export function initLogs() {
  renderChips();
  renderAll();
  initFeed();
  $('log-all').addEventListener('click', () => {
    for (const c of CATS) enabled.add(c);
    renderChips();
    renderAll();
  });
  $('log-clear').addEventListener('click', () => { buf = []; renderAll(); });
  $('log-auto').addEventListener('change', () => scrollEnd());
  $('log-debug').addEventListener('change', renderAll);
  bus.on('sim-events', (ev) => { if (ev.length) pullLogs(); });
  bus.on('command', () => pullLogs());
  bus.on('perception', () => pullLogs());
  setInterval(() => { if (!document.hidden) pullLogs(); }, POLL_MS);
  return pullLogs();
}
