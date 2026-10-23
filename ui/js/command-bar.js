// 게임 채팅줄 같은 명령 입력: Enter → command_submit, 한 줄 결과/오류, ↑↓ 기록.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, fmt, storageGet, storageSet } from './util.js';
import { refreshQuests, setPlaying } from './sim.js';
import { selectCenterTab } from './layout.js';
import { setStatus } from './toast.js';

const HISTORY_KEY = 'hand-sandbox.cmd-history';
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

async function submit(raw) {
  const text = raw.trim();
  if (!text || busy) return;
  busy = true;
  $('command-send').disabled = true;
  $('command-input').readOnly = true;
  try {
    const r = await call('command_submit', { text }, { quiet: true });
    state.lastCommand = { ...r, text };
    bus.emit('command', state.lastCommand);
    if (r.ok) {
      let msg = `✓ ${summarize(r.parsed)} · 퀘스트 ${r.quest_id ?? '—'} · 신뢰도 ${fmt.pct(r.parsed?.confidence ?? NaN)} · ${MODE_KO[r.mode] ?? r.mode}`;
      pushHistory(text);
      $('command-input').value = '';
      await refreshQuests();
      if (!state.playing && r.quest_id) {
        setPlaying(true);
        msg += ' · 재생 시작';
      }
      showToast('ok', msg);
      setStatus(`명령 실행: ${summarize(r.parsed)}`);
    } else {
      showToast('err', `✕ ${r.error ?? '명령을 해석하지 못했습니다'}`);
    }
  } catch (e) {
    showToast('err', `✕ ${e.message}`);
    setStatus(`command_submit: ${e.message}`, true);
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
