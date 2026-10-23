// 하단 로그 탭: log_tail 증분 조회(afterSeq), 범주 필터 칩, 수준별 색, 자동 스크롤.
// 필터는 화면에서만 걸고 커서는 하나(범주 null)만 쓴다. 그래서 칩을 껐다 켜도 다시 조회하지 않는다.
import { call } from './api.js';
import { bus } from './state.js';
import { $, clear, el, fmt } from './util.js';

const CATS = ['world', 'quest', 'search', 'vision', 'sds', 'ts', 'embed', 'device'];
const POLL_MS = 1200;
const BATCH = 200;
const MAX_BUF = 3000;
const MAX_DOM = 600;

const enabled = new Set(CATS);
let buf = [];
let lastSeq = 0;
let pulling = false;
let failed = false;

const passes = (l) => (!CATS.includes(l.cat) || enabled.has(l.cat)) && ($('log-debug').checked || l.level !== 'debug');

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
  const msg = el('span', { class: 'msg', text: l.msg });
  if (hasData) {
    const kv = kvSummary(l.data);
    if (kv) msg.append(' ', el('span', { class: 'kv', text: kv }));
  }
  const node = el('div', {
    class: `log-line ${l.level}${hasData ? ' has-data' : ''}`, dataset: { cat: l.cat, seq: l.seq },
    tabindex: hasData ? 0 : null, role: hasData ? 'button' : null, aria: hasData ? { expanded: 'false' } : null,
  }, el('span', { class: 'ts', text: fmt.time(l.ts) }), el('span', { class: 'cat', text: l.cat }), msg);
  if (hasData) {
    const toggle = () => {
      const open = node.getAttribute('aria-expanded') === 'true';
      node.setAttribute('aria-expanded', String(!open));
      if (open) node.querySelector('.raw')?.remove();
      else node.append(el('div', { class: 'raw', text: JSON.stringify(l.data, null, 1) }));
    };
    node.addEventListener('click', toggle);
    node.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  }
  return node;
}

function updateCount() {
  const shown = buf.filter(passes).length;
  $('log-count').textContent = `${shown} / ${buf.length}줄`;
}

function scrollEnd() {
  if (!$('log-auto').checked) return;
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

export function initLogs() {
  renderChips();
  renderAll();
  $('log-all').addEventListener('click', () => {
    for (const c of CATS) enabled.add(c);
    renderChips();
    renderAll();
  });
  $('log-clear').addEventListener('click', () => { buf = []; renderAll(); });
  $('log-auto').addEventListener('change', scrollEnd);
  $('log-debug').addEventListener('change', renderAll);
  bus.on('sim-events', (ev) => { if (ev.length) pullLogs(); });
  bus.on('command', () => pullLogs());
  bus.on('perception', () => pullLogs());
  setInterval(() => { if (!document.hidden) pullLogs(); }, POLL_MS);
  return pullLogs();
}
