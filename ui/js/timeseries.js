// 기록·분석 모달의 '시계열 · SDS' 탭: ts_forecast 의 이력 선 + P10~P90 띠 + P50 선을 캔버스에 그린다.
import { call } from './api.js';
import { refreshSds } from './analysis.js';
import { bus, state } from './state.js';
import { $, clear, cssVar, el, fitCanvas, fmt, observeResize } from './util.js';

const REFRESH_MS = 5000;
const FALLBACK = { vision: ['score', 'margin'], search: ['score'], motion: ['success'] };
let hover = null; // 마우스가 가리키는 x 인덱스
let tab = null;
let busy = false;

function alpha(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ── 선택기: sds_status.axes 에서 범위·축 목록을 만든다 ───────────
function axisMap() {
  const map = new Map();
  // 실제 백엔드는 이력이 있는 (범위, 축) 쌍만 forecastable 로 알려 준다. 비어 있으면 선택지도 비운다.
  if (Array.isArray(state.sds?.forecastable)) {
    for (const f of state.sds.forecastable) {
      if (!map.has(f.scope)) map.set(f.scope, new Set());
      map.get(f.scope).add(f.axis);
    }
    return map;
  }
  for (const a of state.sds?.axes ?? []) {
    if (!map.has(a.scope)) map.set(a.scope, new Set());
    map.get(a.scope).add(a.axis);
  }
  if (!map.size && state.mock) for (const [s, axes] of Object.entries(FALLBACK)) map.set(s, new Set(axes)); // 모의 백엔드에서만
  return map;
}

function fillSelect(sel, values, keep) {
  clear(sel);
  for (const v of values) sel.append(el('option', { value: v, text: v }));
  sel.value = values.includes(keep) ? keep : (values[0] ?? '');
}

function sameOptions(sel, values) {
  return [...sel.options].map((o) => o.value).join('|') === values.join('|');
}

function syncSelectors() {
  const map = axisMap();
  const scopeSel = $('ts-scope');
  const axisSel = $('ts-axis');
  const scopes = [...map.keys()];
  if (!sameOptions(scopeSel, scopes)) fillSelect(scopeSel, scopes, scopeSel.value);
  const axes = [...(map.get(scopeSel.value) ?? [])];
  if (!sameOptions(axisSel, axes)) fillSelect(axisSel, axes, axisSel.value); // 바뀐 경우에만 다시 채워 선택을 유지
}

// ── 조회 ────────────────────────────────────────────────────────
export async function refreshForecast() {
  if (busy) return;
  const scope = $('ts-scope').value;
  const axis = $('ts-axis').value;
  if (!scope || !axis) {
    state.forecast = null;
    $('ts-info').textContent = '아직 예측할 이력이 없습니다. 인식이 쌓이면(인식 실행 또는 퀘스트 진행) 선택지가 나타납니다.';
    draw();
    return;
  }
  busy = true;
  try {
    state.forecast = { ...(await call('ts_forecast', { scope, axis }, { quiet: true })), scope, axis };
    bus.emit('forecast', state.forecast);
    draw();
  } catch (e) {
    $('ts-info').textContent = `예측 실패: ${e.message}`;
  } finally {
    busy = false;
  }
}

function infoText(f) {
  const weights = (f.weights ?? []).map((w) => `${w.method} ${Math.round(w.weight * 100)}%`).join(' · ');
  return `방법 ${f.method} · 이력 ${f.history.length}점 → 예측 ${f.point.length}점${weights ? ` · 가중치 ${weights}` : ''}`;
}

// ── 그리기 ──────────────────────────────────────────────────────
function draw() {
  const fit = fitCanvas($('ts-canvas'));
  if (!fit) return;
  const { ctx, W, H } = fit;
  ctx.clearRect(0, 0, W, H);
  const f = state.forecast;
  const C = {
    grid: cssVar('--plot-grid-strong'), axis: cssVar('--text-muted'), text: cssVar('--text-dim'),
    hist: cssVar('--accent'), fc: cssVar('--violet'), warn: cssVar('--warning'),
  };
  ctx.font = `10px ${cssVar('--font-mono')}`;
  if (!f || !f.history?.length) {
    ctx.fillStyle = C.axis;
    ctx.textAlign = 'center';
    ctx.fillText('예측 데이터가 없습니다 — 새로고침을 누르세요', W / 2, H / 2);
    return;
  }
  $('ts-info').textContent = infoText(f);
  const n = f.history.length;
  const m = f.point.length;
  const padL = 50, padR = 16, padT = 24, padB = 24;
  const vals = [...f.history, ...f.point, ...f.p10, ...f.p90].filter(Number.isFinite);
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
  const span = hi - lo;
  lo -= span * 0.08;
  hi += span * 0.08;
  const total = Math.max(1, n + m - 1);
  const X = (i) => padL + (i / total) * (W - padL - padR);
  const Y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

  // 격자와 y 눈금
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = C.axis;
  ctx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();
    ctx.fillText(fmt.n(v, Math.abs(hi - lo) < 2 ? 3 : 1), padL - 6, y + 3);
  }
  // 현재 구분선
  const xNow = Math.round(X(n - 1)) + 0.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(xNow, padT);
  ctx.lineTo(xNow, H - padB);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.textAlign = 'left';
  ctx.fillText('현재', xNow + 4, H - padB + 14);
  ctx.textAlign = 'right';
  ctx.fillText('예측 →', W - padR, H - padB + 14);

  // P10~P90 띠
  if (m > 0 && f.p10.length === m && f.p90.length === m) {
    ctx.beginPath();
    ctx.moveTo(X(n - 1), Y(f.history[n - 1]));
    for (let i = 0; i < m; i++) ctx.lineTo(X(n + i), Y(f.p90[i]));
    for (let i = m - 1; i >= 0; i--) ctx.lineTo(X(n + i), Y(f.p10[i]));
    ctx.closePath();
    ctx.fillStyle = alpha(C.fc, 0.2);
    ctx.fill();
  }
  // 이력 선
  ctx.lineWidth = 1.6;
  ctx.strokeStyle = C.hist;
  ctx.beginPath();
  f.history.forEach((v, i) => (i === 0 ? ctx.moveTo(X(i), Y(v)) : ctx.lineTo(X(i), Y(v))));
  ctx.stroke();
  // P50 선
  if (m > 0) {
    ctx.strokeStyle = C.fc;
    ctx.beginPath();
    ctx.moveTo(X(n - 1), Y(f.history[n - 1]));
    f.point.forEach((v, i) => ctx.lineTo(X(n + i), Y(v)));
    ctx.stroke();
  }
  // 범례
  const legend = [[C.hist, '이력'], [C.fc, 'P50'], [alpha(C.fc, 0.35), 'P10–P90']];
  let lx = padL;
  ctx.textAlign = 'left';
  for (const [color, label] of legend) {
    ctx.fillStyle = color;
    ctx.fillRect(lx, 9, 10, 8);
    ctx.fillStyle = C.text;
    ctx.fillText(label, lx + 14, 17);
    lx += 22 + ctx.measureText(label).width;
  }
  // 마우스 위치 판독
  if (hover != null) {
    const i = Math.max(0, Math.min(n + m - 1, hover));
    const x = Math.round(X(i)) + 0.5;
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, H - padB);
    ctx.stroke();
    const text = i < n
      ? `#${i}  이력 ${fmt.n(f.history[i], 3)}`
      : `+${i - n + 1}  P10 ${fmt.n(f.p10[i - n], 3)}  P50 ${fmt.n(f.point[i - n], 3)}  P90 ${fmt.n(f.p90[i - n], 3)}`;
    ctx.fillStyle = C.text;
    ctx.textAlign = 'right';
    ctx.fillText(text, W - padR, 17);
  }
}

export function initTimeseries() {
  const canvas = $('ts-canvas');
  observeResize(canvas, draw);
  canvas.addEventListener('mousemove', (e) => {
    const f = state.forecast;
    if (!f) return;
    const r = canvas.getBoundingClientRect();
    const total = Math.max(1, f.history.length + f.point.length - 1);
    hover = Math.round(((e.clientX - r.left - 50) / (r.width - 66)) * total);
    draw();
  });
  canvas.addEventListener('mouseleave', () => { hover = null; draw(); });
  $('ts-scope').addEventListener('change', () => { syncSelectors(); refreshForecast(); });
  $('ts-axis').addEventListener('change', refreshForecast);
  $('ts-refresh').addEventListener('click', refreshForecast);
  bus.on('sds', syncSelectors);
  bus.on('tab', ({ group, name }) => {
    if (group !== 'analysis') return;
    tab = name;
    if (name === 'ts') refreshSds().then(() => { syncSelectors(); refreshForecast(); });
  });
  setInterval(() => { if (tab === 'ts' && !document.hidden) refreshSds().then(() => { syncSelectors(); refreshForecast(); }); }, REFRESH_MS);
  syncSelectors();
  draw();
  return refreshForecast();
}
