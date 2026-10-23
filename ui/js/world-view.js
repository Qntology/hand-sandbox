// 월드 화면: world_frame(WebP) 폴링 → 캔버스, 탐지 박스 오버레이, 범례, 표시 토글.
// 폴링은 약 6fps 이고, 이전 호출이 끝나지 않았거나 틱·옵션이 그대로면 건너뛴다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt, observeResize } from './util.js';

const POLL_MS = 160;
const RETRY_MS = 2500;
let pending = false;
let lastKey = '';
let retryAt = 0;
let drewOnce = false;

function framePreset() {
  const [width, height, quality] = $('frame-preset').value.split('x').map(Number);
  return { width, height, quality };
}

// ── 배치: 프레임을 비율 유지로 맞추고, 크기에 따라 HUD 배치 단계를 정한다 ──
function layoutFrame() {
  const wrap = $('world-wrap');
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  if (!W || !H) return;
  wrap.dataset.w = W >= 900 ? 'wide' : W >= 620 ? 'narrow' : 'tiny';
  wrap.dataset.h = H < 380 ? 'short' : H < 520 ? 'mid' : 'tall';
  const fw = state.frame?.width ?? 3;
  const fh = state.frame?.height ?? 2;
  const s = Math.min(W / fw, H / fh);
  const w = fw * s;
  const h = fh * s;
  const box = $('frame-box');
  box.style.left = `${(W - w) / 2}px`;
  box.style.top = `${(H - h) / 2}px`;
  box.style.width = `${w}px`;
  box.style.height = `${h}px`;
}

// ── 디코드 ──────────────────────────────────────────────────────
async function decode(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const type = bin.startsWith('RIFF') ? 'image/webp' : bin.startsWith('\x89PNG') ? 'image/png' : 'image/jpeg';
  const blob = new Blob([bytes], { type });
  if (typeof createImageBitmap === 'function') return createImageBitmap(blob);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('프레임 디코드 실패'));
    img.src = URL.createObjectURL(blob);
  });
}

async function drawFrame(f) {
  const img = await decode(f.webp_base64);
  const canvas = $('world-canvas');
  if (canvas.width !== f.width || canvas.height !== f.height) {
    canvas.width = f.width;
    canvas.height = f.height;
  }
  canvas.getContext('2d').drawImage(img, 0, 0, f.width, f.height);
  img.close?.();
  const sizeChanged = !state.frame || state.frame.width !== f.width || state.frame.height !== f.height;
  state.frame = { width: f.width, height: f.height, bytes: f.bytes, encode_ms: f.encode_ms, tick: f.tick };
  state.detections = f.detections ?? [];
  if (sizeChanged) layoutFrame();
  if (!drewOnce) {
    drewOnce = true;
    $('world-empty').hidden = true;
  }
  $('frame-info').textContent = `틱 ${f.tick} · ${f.width}×${f.height} · ${fmt.bytes(f.bytes)} · 인코딩 ${fmt.n(f.encode_ms, 1)} ms`;
  $('sb-frame').textContent = `${fmt.bytes(f.bytes)} · ${fmt.n(f.encode_ms, 1)} ms`;
  renderDetections();
  bus.emit('frame', state.frame);
}

// ── 탐지 박스 · 범례 ────────────────────────────────────────────
function tagText(d) {
  const base = `${d.label_ko} ${Math.round(d.score * 100)}${d.accepted ? '' : ' 보류'}`;
  return state.toggles.oracle ? `${base} · ${d.label_id}` : base;
}

function renderDetections() {
  const layer = $('det-layer');
  clear(layer);
  const dets = state.detections;
  if (state.toggles.boxes) {
    for (const d of dets) {
      if (!Array.isArray(d.bbox) || d.bbox.length < 4) continue;
      const [x0, y0, x1, y1] = d.bbox;
      const box = el('div', { class: `det-box${d.accepted ? '' : ' hold'}${y0 < 0.07 ? ' edge' : ''}`, title: d.reason ?? '' },
        el('span', { class: 'det-tag', text: tagText(d) }));
      box.style.left = `${x0 * 100}%`;
      box.style.top = `${y0 * 100}%`;
      box.style.width = `${Math.max(0, x1 - x0) * 100}%`;
      box.style.height = `${Math.max(0, y1 - y0) * 100}%`;
      layer.append(box);
    }
  }
  const list = $('lg-list');
  clear(list);
  const sorted = [...dets].sort((a, b) => b.score - a.score).slice(0, 10);
  $('lg-count').textContent = dets.length ? `${dets.filter((d) => d.accepted).length}/${dets.length} 수락` : '';
  if (!sorted.length) {
    list.append(el('li', { class: 'lg-empty', text: '탐지 없음' }));
    return;
  }
  for (const d of sorted) {
    list.append(el('li', { class: d.accepted ? '' : 'hold', title: d.reason ?? '' },
      el('span', { class: 'sw' }),
      el('span', { class: 'nm', text: state.toggles.oracle ? `${d.label_ko} [${d.label_id}]` : d.label_ko }),
      el('span', { class: 'sc', text: `${Math.round(d.score * 100)}%` }),
      el('span', { class: `choice-badge ${d.accepted ? 'done' : 'rule'}`, text: d.accepted ? '수락' : '보류' })));
  }
}

// ── 폴링 ────────────────────────────────────────────────────────
async function poll(force = false) {
  if (pending || state.centerTab !== 'world' || document.hidden || !state.snapshot) return;
  if (!force && Date.now() < retryAt) return;
  const p = framePreset();
  const view = $('view-select').value;
  const key = `${state.snapshot.tick}|${p.width}x${p.height}q${p.quality}|${view}`;
  if (!force && key === lastKey) return;
  pending = true;
  try {
    const f = await call('world_frame', { ...p, view }, { quiet: true });
    lastKey = key;
    await drawFrame(f);
  } catch (e) {
    retryAt = Date.now() + RETRY_MS;
    $('frame-info').textContent = `프레임 오류: ${e.message}`;
    if (!drewOnce) showEmpty(`프레임을 가져오지 못했습니다: ${e.message}`, true, () => { retryAt = 0; poll(true); });
  } finally {
    pending = false;
  }
}

function showEmpty(message, isError = false, onRetry = null) {
  const box = $('world-empty');
  clear(box);
  box.hidden = false;
  box.classList.toggle('err', isError);
  box.append(el('div', { text: message }));
  if (isError && onRetry) {
    box.append(el('button', { type: 'button', class: 'btn btn-sm btn-outline', text: '다시 시도', on: { click: onRetry } }));
  }
}

// ── 표시 토글 ───────────────────────────────────────────────────
const TOGGLES = [
  ['tg-boxes', 'boxes', () => renderDetections()],
  ['tg-path', 'path', () => { $('vp-minimap').hidden = !state.toggles.path; }],
  ['tg-oracle', 'oracle', () => renderDetections()],
  ['tg-body', 'body', () => { $('body-hud').hidden = !state.toggles.body; }],
  ['tg-quest', 'quest', () => { $('quest-tracker').hidden = !state.toggles.quest; }],
  ['tg-pip', 'pip', () => { $('pip').hidden = !state.toggles.pip; }],
];

function initToggles() {
  for (const [id, key, apply] of TOGGLES) {
    const btn = $(id);
    state.toggles[key] = btn.getAttribute('aria-pressed') === 'true';
    btn.addEventListener('click', () => {
      state.toggles[key] = !state.toggles[key];
      btn.setAttribute('aria-pressed', String(state.toggles[key]));
      btn.classList.toggle('active', state.toggles[key]);
      apply();
    });
    apply();
  }
}

export function initWorldView() {
  initToggles();
  observeResize($('world-wrap'), layoutFrame);
  layoutFrame();
  $('view-select').addEventListener('change', () => poll(true));
  $('frame-preset').addEventListener('change', () => poll(true));
  bus.on('world-new', () => { lastKey = ''; retryAt = 0; poll(true); });
  bus.on('world-error', (m) => showEmpty(`월드를 만들지 못했습니다: ${m}`, true, () => $('world-new').click()));
  bus.on('tab', ({ group, name }) => {
    if (group === 'center' && name === 'world') { layoutFrame(); poll(true); }
  });
  setInterval(() => poll(false), POLL_MS);
  renderDetections();
}
