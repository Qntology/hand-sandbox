// 게임 채팅줄 같은 명령 입력: Enter → command_submit, 한 줄 결과/오류, ↑↓ 기록.
// 뷰포트 툴바의 '테스트 실행'도 같은 submitCommand 를 쓴다. '자동 기록'이 켜져 있으면 명령이
// 통합 테스트로 기록되고(지금 월드에서 1 게임 → 2 ML 인식 → 3 ML 결정), 기록 중에 낸 명령은 차례를 기다린다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, fmt, storageGet, storageSet } from './util.js';
import { refreshQuests, setPlaying } from './sim.js';
import { selectCenterTab } from './layout.js';
import { setStatus } from './toast.js';

const HISTORY_KEY = 'noto-sandbox.cmd-history';
const ACTION_KO = {
  mine: '채취', collect: '줍기', carry: '운반', deliver: '인도', drop: '내려놓기', goto: '이동', locate: '탐색',
  mine_and_deliver: '채취·운반', fetch: '가져오기', stop: '중지', unknown: '해석 불가',
};
const MODE_KO = { lexical: '어휘 전용', embedding: '임베딩+어휘', 'lexical+embedding': '임베딩+어휘' };

let history = [];
let idx = 0;
let draft = '';
let busy = false;
let toastTimer = null;

export function describeDestination(d) {
  if (!d) return null;
  if (d.kind === 'waypoint') return d.label_ko ?? d.id ?? '웨이포인트';
  if (d.kind === 'coord' && Array.isArray(d.pos)) return `(${d.pos.map((v) => fmt.n(v, 1)).join(', ')})`;
  return d.label_ko ?? d.id ?? '목적지';
}

export function summarize(p) {
  if (!p) return '';
  const parts = [ACTION_KO[p.action] ?? p.action];
  if (p.item_label_ko || p.item_id) parts.push(`${p.item_label_ko ?? p.item_id}${p.count != null ? ` ×${p.count}` : ''}`);
  const dest = describeDestination(p.destination);
  return dest ? `${parts.join(' · ')} → ${dest}` : parts.join(' · ');
}

function showToast(kind, text) {
  const t = $('cmd-toast');
  t.hidden = false;
  t.className = `cmd-toast ${kind}`;
  t.textContent = text;
  t.title = text;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === 'err' ? 9000 : 7000);
}

/** '자동 기록' 체크(툴바). 모의 백엔드는 기록하지 않는다. */
export function recordOn() {
  return !state.mock && !!$('tl-record')?.checked;
}

/**
 * 명령 한 줄을 낸다(입력줄 · 툴바 공용). 퀘스트가 생기거나 기다리면 재생을 시작한다.
 * opts.quietHistory: 기록(↑↓)에 남기지 않는다(툴바의 중지 등).
 */
export async function submitCommand(raw, opts = {}) {
  const text = raw.trim();
  if (!text) return null;
  try {
    const record = recordOn();
    const r = await call('command_submit', { text, record }, { quiet: true });
    state.lastCommand = { ...r, text };
    bus.emit('command', state.lastCommand);
    if (r.ok) {
      const head = r.queued ? `⏳ 대기 ${r.queued}번째 — 지금 기록이 끝나면 시작` : `✓ ${summarize(r.parsed)} · 퀘스트 ${r.quest_id ?? '—'}`;
      let msg = `${head} · 신뢰도 ${fmt.pct(r.parsed?.confidence ?? NaN)} · ${MODE_KO[r.mode] ?? r.mode}`;
      if (r.recording) msg += ' · 통합 테스트 기록 중';
      if (!opts.quietHistory) pushHistory(text);
      if (!r.stopped && $('tl-cmd')) $('tl-cmd').value = text;
      await refreshQuests();
      if (!state.playing && (r.quest_id || r.queued || r.recording)) {
        setPlaying(true);
        msg += ' · 재생 시작';
      }
      showToast('ok', msg);
      setStatus(`명령 실행: ${summarize(r.parsed)}${r.queued ? ' (대기)' : ''}`);
    } else {
      showToast('err', `✕ ${r.error ?? '명령을 해석하지 못했습니다'}`);
    }
    return r;
  } catch (e) {
    showToast('err', `✕ ${e.message}`);
    setStatus(`command_submit: ${e.message}`, true);
    return null;
  }
}

async function submit(raw) {
  const text = raw.trim();
  if (!text || busy) return;
  busy = true;
  $('command-send').disabled = true;
  $('command-input').readOnly = true;
  try {
    const r = await submitCommand(text);
    if (r?.ok) $('command-input').value = '';
  } finally {
    busy = false;
    $('command-send').disabled = false;
    const input = $('command-input');
    input.readOnly = false;
    input.focus();
  }
}

function pushHistory(text) {
  if (history[history.length - 1] !== text) history.push(text);
  history = history.slice(-50);
  idx = history.length;
  draft = '';
  storageSet(HISTORY_KEY, JSON.stringify(history));
}

function onKeydown(e) {
  if (e.isComposing) return;
  const input = e.currentTarget;
  if (e.key === 'ArrowUp' && history.length) {
    e.preventDefault();
    if (idx === history.length) draft = input.value;
    idx = Math.max(0, idx - 1);
    input.value = history[idx];
    input.setSelectionRange(input.value.length, input.value.length);
  } else if (e.key === 'ArrowDown' && idx < history.length) {
    e.preventDefault();
    idx += 1;
    input.value = idx === history.length ? draft : history[idx];
    input.setSelectionRange(input.value.length, input.value.length);
  } else if (e.key === 'Escape') {
    if (input.value) input.value = '';
    else input.blur();
  }
}

export function initCommandBar() {
  try {
    history = JSON.parse(storageGet(HISTORY_KEY, '[]')).filter((s) => typeof s === 'string').slice(-50);
  } catch {
    history = [];
  }
  idx = history.length;
  const input = $('command-input');
  input.addEventListener('keydown', onKeydown);
  $('command-bar').addEventListener('submit', (e) => {
    e.preventDefault();
    submit(input.value);
  });
  // 게임처럼 Enter 나 / 로 입력줄에 바로 포커스
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key !== 'Enter' && e.key !== '/') return;
    const a = document.activeElement;
    if (a && a !== document.body && !a.matches?.('canvas, .gcode-console')) return;
    e.preventDefault();
    selectCenterTab('world');
    input.focus();
  });
}
