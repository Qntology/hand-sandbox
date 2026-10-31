// 교통 지도식 CCTV: 탑뷰에서 로봇마다 번호표와 머리 위 말풍선(1인칭 · 깊이 · 포크 카메라 작은 화면)을 띄운다.
// 자리는 프레임의 u, v 에서 구하고 말풍선끼리, 또 코너·허브·HUD 와 겹치지 않게 비켜 놓는다.
// 화면은 unit_frame 으로 타일 하나씩 받는다: 전체 초당 2장까지, 틱이 바뀐 타일만, 고른 유닛을 먼저.
// 탑뷰가 아니거나 탭이 숨었거나 모달이 덮으면 받지 않는다. 줍기·포크 상차 중이면 그 로봇의 절차와 교차 검증을 말풍선 아래에 붙인다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, el, fmt, svgEl } from './util.js';
import { isOpen } from './overlays.js';
import { renderProcedureBrief } from './pick-hud.js';
import { removeCrew } from './crew-panel.js';
import { selectUnit } from './unit-select.js';
import { frameRect } from './world-view.js';
import { openZoom } from './cctv-zoom.js';

const TILE_MS = 500; // 전체 초당 2장
const RETRY_MS = 5000;
const TILE = { width: 128, height: 96, quality: 55 };
const CROWD = 4; // 이보다 많으면 고르지 않은 말풍선은 타일 하나로 줄인다
const PIN_UP = 13; // 번호표는 로봇 바로 위
const TAIL = 16; // 번호표와 말풍선 사이
const GAP = 6;
const EDGE = 6;
const RINGS = 4; // 자리 후보의 고리 수
const ANGLES = 12; // 고리 하나의 자리 수
const CAPTION = { rgb: '1인칭', depth: '깊이', fork: '포크 카메라' };

const bubbles = new Map(); // id → 말풍선
let lines = null;
let busy = false;
let retryAt = 0;

const topView = () => $('view-select').value === 'top';
const unitTag = (u) => (u.first ? `★${u.id}` : `${u.form === 'forklift' ? '🚜' : '🤖'}${u.id}`);

// ── 말풍선 만들기 ───────────────────────────────────────────────
function makeTile(mode) {
  const img = el('img', { alt: '', decoding: 'async' });
  const cap = el('figcaption', { text: CAPTION[mode] });
  return { mode, img, cap, fig: el('figure', { class: `cctv-tile${mode === 'depth' ? '' : ' aim'}` }, img, cap), at: 0, tick: -1 };
}

function makeBubble(u) {
  const tiles = [makeTile('rgb'), makeTile('depth')];
  const status = el('span', { class: 'cctv-status' });
  const proc = el('div', { class: 'cctv-proc', hidden: true });
  const zoom = el('button', { type: 'button', class: 'vt-btn', text: '🔍 확대', title: '라이브 비전 비교 창', on: { click: (e) => { e.stopPropagation(); openZoom(u.id); } } });
  const remove = u.first ? null : el('button', { type: 'button', class: 'vt-btn', text: '🗑 제거', title: '이 서브 작업자를 뺍니다', on: { click: (e) => { e.stopPropagation(); removeCrew(u.id); } } });
  const pick = () => selectUnit(u.id);
  const node = el('article', { class: 'cctv fresh', dataset: { unit: u.id }, aria: { label: `유닛 ${u.id} CCTV` }, on: { click: pick } },
    el('header', { class: 'cctv-head' }, el('b', { text: `📹 ${unitTag(u)}` }), status),
    el('div', { class: 'cctv-tiles' }, tiles.map((t) => t.fig)),
    el('footer', { class: 'cctv-foot' }, zoom, remove),
    proc);
  const pin = el('button', { type: 'button', class: 'unit-tag-pin fresh', text: unitTag(u), title: `유닛 #${u.id} 고르기`, on: { click: pick } });
  const line = svgEl('line', {});
  lines.append(line);
  $('cctv-layer').append(node, pin);
  requestAnimationFrame(() => requestAnimationFrame(() => { node.classList.remove('fresh'); pin.classList.remove('fresh'); }));
  return { id: u.id, form: u.form, node, pin, line, tiles, status, proc, procedure: null, slot: 0 };
}

function dropBubble(b) {
  b.node.remove();
  b.pin.remove();
  b.line.remove();
  bubbles.delete(b.id);
}

// ── 자리 잡기 ───────────────────────────────────────────────────
/** 안 덮을 영역: 코너·허브·유닛 HUD (world-wrap 기준). */
function obstacles() {
  const o = $('world-wrap').getBoundingClientRect();
  return ['corner-tl', 'hub', 'unit-hud'].map($).concat(document.querySelector('.corner-tr'))
    .filter((n) => n && !n.hidden)
    .map((n) => {
      const r = n.getBoundingClientRect();
      return { x: r.left - o.left, y: r.top - o.top, w: r.width, h: r.height };
    });
}

const hits = (a, b) => a.x < b.x + b.w + GAP && b.x < a.x + a.w + GAP && a.y < b.y + b.h + GAP && b.y < a.y + a.h + GAP;

/** 로봇 둘레의 자리 후보. 0 이 머리 위(기본)이고, 뒤로 갈수록 바깥 고리로 멀어진다(로봇이 모여 있어도 다 들어가게). */
function spots(ax, ay, w, h) {
  const out = [[ax - w / 2, ay - PIN_UP - TAIL - h]];
  for (let ring = 1; ring <= RINGS; ring++) {
    const rx = ring * (w + GAP);
    const ry = ring * (h + GAP);
    for (let k = 0; k < ANGLES; k++) {
      const a = -Math.PI / 2 + (k * 2 * Math.PI) / ANGLES;
      out.push([ax + rx * Math.cos(a) - w / 2, ay - PIN_UP / 2 + ry * Math.sin(a) - h / 2]);
    }
  }
  return out;
}

function placeOne(b, ax, ay, placed, blocked, W, H) {
  const w = b.node.offsetWidth;
  const h = b.node.offsetHeight;
  const clampBox = ([x, y]) => ({ x: Math.max(EDGE, Math.min(W - w - EDGE, x)), y: Math.max(EDGE, Math.min(H - h - EDGE, y)), w, h });
  const cands = spots(ax, ay, w, h).map(clampBox);
  // 지난번 자리를 먼저 본다: 로봇이 조금 움직일 때 말풍선이 튀지 않게.
  const order = [b.slot, ...cands.keys()];
  const slot = order.find((i) => !placed.concat(blocked).some((r) => hits(cands[i], r))) ?? 0;
  b.slot = slot;
  placed.push(cands[slot]);
  return cands[slot];
}

/** 연결선은 로봇에서 말풍선의 가장 가까운 변까지. */
function drawLine(b, ax, ay, r, sel) {
  const x = Math.max(r.x, Math.min(r.x + r.w, ax));
  const y = Math.max(r.y, Math.min(r.y + r.h, ay - PIN_UP));
  for (const [k, v] of [['x1', ax], ['y1', ay - PIN_UP], ['x2', x], ['y2', y]]) b.line.setAttribute(k, v.toFixed(1));
  b.line.setAttribute('class', sel ? 'sel' : '');
}

function setStatusText(node, text) {
  if (node.textContent !== text) {
    node.textContent = text;
    node.title = text;
  }
}

/** 로봇 자리: 말풍선이 다른 로봇을 덮지 않게 비켜 갈 영역. */
const robotBox = (ax, ay) => ({ x: ax - 16, y: ay - PIN_UP - 10, w: 32, h: PIN_UP + 24 });

/** 프레임마다: 말풍선을 만들고 지우고, 자리를 잡고, 글자를 맞춘다. */
function sync() {
  const units = topView() ? state.units : [];
  const showBubbles = state.toggles.cctv;
  const live = new Set(units.map((u) => u.id));
  for (const b of [...bubbles.values()]) if (!live.has(b.id)) dropBubble(b);
  if (!units.length) return;
  const sel = state.selectedUnit;
  const crowded = units.length > CROWD;
  const wrap = $('world-wrap');
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  const f = frameRect();
  const at = (u) => [f.left + u.u * f.width, f.top + u.v * f.height];
  const blocked = obstacles().concat(units.map((u) => robotBox(...at(u))));
  const placed = [];
  // 고른 유닛이 좋은 자리를 먼저 차지한다.
  const order = [...units].sort((a, b) => (b.id === sel) - (a.id === sel) || a.id - b.id);
  for (const u of order) {
    const b = bubbles.get(u.id) ?? makeBubble(u);
    bubbles.set(u.id, b);
    const [ax, ay] = at(u);
    b.onScreen = u.u >= 0 && u.u <= 1 && u.v >= 0 && u.v <= 1;
    const isSel = u.id === sel;
    b.pin.style.transform = `translate(${ax.toFixed(1)}px, ${(ay - PIN_UP).toFixed(1)}px)`;
    b.pin.classList.toggle('sel', isSel);
    b.pin.classList.toggle('fallen', !!u.fallen);
    b.pin.classList.toggle('loaded', u.carried_kg > 0);
    b.node.hidden = !showBubbles;
    b.line.style.display = showBubbles ? '' : 'none';
    if (!showBubbles) continue;
    b.node.classList.toggle('sel', isSel);
    b.node.classList.toggle('compact', crowded && !isSel);
    setStatusText(b.status, u.fallen ? '넘어짐' : u.status_ko ?? '');
    const r = placeOne(b, ax, ay, placed, blocked, W, H);
    b.node.style.transform = `translate(${r.x.toFixed(1)}px, ${r.y.toFixed(1)}px)`;
    drawLine(b, ax, ay, r, isSel);
  }
}

// ── 타일 받기 ───────────────────────────────────────────────────
const canPull = () => topView() && state.toggles.cctv && !!state.snapshot && !document.hidden
  && !isOpen('analysis-modal') && !isOpen('cctv-zoom') && Date.now() >= retryAt;

/** 다음에 받을 타일: 이번 틱에 아직 안 받은 것 중 가장 오래된 것 (고른 유닛은 세 배 빨리 돈다). */
function nextTile() {
  const tick = state.snapshot.tick;
  const now = performance.now();
  let best = null;
  let bestAge = -1;
  for (const b of bubbles.values()) {
    if (!b.onScreen) continue;
    const compact = b.node.classList.contains('compact');
    for (const t of compact ? b.tiles.slice(0, 1) : b.tiles) {
      if (t.tick === tick) continue;
      const age = (now - t.at) * (b.id === state.selectedUnit ? 3 : 1);
      if (age > bestAge) {
        best = { b, t };
        bestAge = age;
      }
    }
  }
  return best;
}

/** 지게차가 절차 중이면 둘째 타일은 포크 카메라, 평소에는 깊이. */
function applyProcedure(b, p) {
  b.procedure = p;
  const on = renderProcedureBrief(b.proc, p);
  b.proc.hidden = !on;
  const [rider, second] = b.tiles;
  const fork = on && p.mechanism === 'fork';
  const mode = fork ? 'fork' : 'depth';
  if (second.mode !== mode) {
    second.mode = mode;
    second.tick = -1;
    second.fig.classList.toggle('aim', fork);
  }
  rider.cap.textContent = fork ? `①탑승자 · 고개 ${fmt.n(p.neck_deg, 0)}°` : CAPTION.rgb;
  if (fork) {
    const blocked = p.cross?.blocked_share;
    second.cap.textContent = blocked > 0 ? `②포크 · 가림 ${fmt.n(blocked * 100, 0)}%` : '②포크 카메라';
  }
}

async function pullTile() {
  if (busy || !canPull()) return;
  const job = nextTile();
  if (!job) return;
  const { b, t } = job;
  const tick = state.snapshot.tick;
  busy = true;
  try {
    const r = await call('unit_frame', { unit: b.id, ...TILE, mode: t.mode }, { quiet: true });
    t.img.src = `data:image/webp;base64,${r.webp_base64}`;
    t.at = performance.now();
    t.tick = tick;
    if (t.mode === 'depth') t.cap.textContent = r.depth_name ? `깊이 · ${r.depth_name}` : CAPTION.depth;
    applyProcedure(b, r.procedure ?? null);
  } catch {
    t.at = performance.now();
    retryAt = Date.now() + RETRY_MS;
  } finally {
    busy = false;
  }
}

export function initCctv() {
  lines = svgEl('svg', { class: 'cctv-lines', 'aria-hidden': 'true' });
  $('cctv-layer').append(lines);
  bus.on('frame', sync);
  bus.on('unit', sync);
  bus.on('view', sync);
  window.addEventListener('resize', sync);
  setInterval(pullTile, TILE_MS);
}
