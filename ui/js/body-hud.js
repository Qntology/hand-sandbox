// 타르코프식 신체 상태 HUD: 인라인 SVG 실루엣 7영역 + 부위 목록 + 상태 효과 칩.
// 데이터는 sim_step / body_status 의 BodyStatus. 모든 문자열은 textContent 로만 넣는다.
import { bus, state } from './state.js';
import { $, clear, el, fmt, LEVEL_KO, svgEl } from './util.js';
import { refreshTooltip } from './layout.js';
import { resetBody } from './world-panel.js';

// 영역 모양(viewBox 150×240). 로봇은 손이 하나라 arm_l/arm_r 은 해부학적 좌우가 아니라 손의 두 지표다.
const ZONES = [
  { id: 'head', shape: { t: 'circle', cx: 75, cy: 24, r: 17 }, title: '머리 · 비전', short: '비전', group: null },
  { id: 'thorax', shape: { t: 'rect', x: 48, y: 46, w: 54, h: 50, rx: 8 }, title: '가슴 · 배터리', short: '배터리', group: null },
  { id: 'stomach', shape: { t: 'rect', x: 51, y: 99, w: 48, h: 40, rx: 8 }, title: '배 · 적재', short: '적재', group: null },
  { id: 'arm_l', shape: { t: 'rect', x: 14, y: 50, w: 28, h: 86, rx: 11 }, title: '손 · 파지', short: '파지', group: 'hand', note: '손 지표: 로봇의 손은 1개입니다. 해부학적 왼팔이 아닙니다.' },
  { id: 'arm_r', shape: { t: 'rect', x: 108, y: 50, w: 28, h: 86, rx: 11 }, title: '손 · 관절', short: '관절', group: 'hand', note: '손 지표: 로봇의 손은 1개입니다. 해부학적 오른팔이 아닙니다.' },
  { id: 'leg_l', shape: { t: 'rect', x: 50, y: 143, w: 23, h: 90, rx: 9 }, title: '발 · 구동', short: '구동', group: 'foot' },
  { id: 'leg_r', shape: { t: 'rect', x: 77, y: 143, w: 23, h: 90, rx: 9 }, title: '발 · 지형', short: '지형', group: 'foot' },
];
const GROUP_KO = { hand: '손', foot: '발' };
// 고정 칩은 시뮬레이터가 계산하는 효과만 둔다. 발열 모델은 없으므로 '과열' 칩은 만들지 않는다(근거 없는 수치가 됨).
const EFFECTS = [
  ['fatigue', '피로'], ['load', '하중'], ['slip', '미끄러짐'],
];

const refs = new Map(); // id → {g, fill, val, li, liVal, box}
let effectsEl = null;

function bbox(s) {
  return s.t === 'circle' ? { x: s.cx - s.r, y: s.cy - s.r, w: s.r * 2, h: s.r * 2 } : { x: s.x, y: s.y, w: s.w, h: s.h };
}

function shapeNode(s, cls) {
  return s.t === 'circle'
    ? svgEl('circle', { class: cls, cx: s.cx, cy: s.cy, r: s.r })
    : svgEl('rect', { class: cls, x: s.x, y: s.y, width: s.w, height: s.h, rx: s.rx });
}

function build() {
  const svg = $('body-svg');
  clear(svg);
  // 목·어깨 연결선(장식)
  svg.append(svgEl('path', { class: 'sil-line', d: 'M69 41v5M81 41v5M42 62h6M102 62h6M62 139v4M88 139v4' }));
  const list = $('body-list');
  clear(list);
  for (const z of ZONES) {
    const b = bbox(z.shape);
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;
    const fill = svgEl('rect', { class: 'z-fill', x: b.x, y: b.y + b.h, width: b.w, height: 0, 'clip-path': `url(#zc-${z.id})` });
    const val = svgEl('text', { class: 'z-val', x: cx, y: cy - 3, text: '—' });
    const g = svgEl('g', { class: 'zone', tabindex: 0, role: 'img', 'data-part': z.id, 'data-level': 'none' },
      svgEl('clipPath', { id: `zc-${z.id}` }, shapeNode(z.shape, '')),
      shapeNode(z.shape, 'z-base'),
      fill,
      z.group ? svgEl('text', { class: 'z-cap', x: cx, y: b.y + 11, text: GROUP_KO[z.group] }) : null,
      val,
      svgEl('text', { class: 'z-cap', x: cx, y: cy + 8, text: z.short }));
    svg.append(g);

    const liVal = el('span', { class: 'val', text: '—' });
    const li = el('li', { dataset: { part: z.id, level: 'none', ...(z.group ? { group: z.group } : {}) } },
      el('span', { class: 'dot' }), el('span', { class: 'nm', text: z.short }), liVal);
    list.append(li);
    refs.set(z.id, { g, fill, val, li, liVal, box: b, z });
  }
  effectsEl = $('body-effects');
}

function render(body) {
  const parts = new Map((body?.parts ?? []).map((p) => [p.id, p]));
  for (const [id, r] of refs) {
    const p = parts.get(id);
    const level = p?.level ?? 'none';
    const ratio = p ? Math.max(0, Math.min(1, p.ratio)) : 0;
    const pct = p ? `${Math.round(ratio * 100)}%` : '—';
    r.g.dataset.level = level;
    r.li.dataset.level = level;
    r.val.textContent = pct;
    r.liVal.textContent = pct;
    r.fill.setAttribute('y', String(r.box.y + r.box.h * (1 - ratio)));
    r.fill.setAttribute('height', String(r.box.h * ratio));
    const head = `${r.z.title} — ${pct}${p ? ` (${fmt.n(p.value, 1)} / ${fmt.n(p.max, 1)})` : ''} · ${LEVEL_KO[level] ?? '정보 없음'}`;
    const tip = [p?.detail, r.z.note].filter(Boolean).join('\n');
    for (const node of [r.g, r.li]) {
      node.dataset.tipTitle = r.z.title;
      node.dataset.tip = `${pct}${p ? ` (${fmt.n(p.value, 1)} / ${fmt.n(p.max, 1)})` : ''} · ${LEVEL_KO[level] ?? '정보 없음'}${tip ? `\n${tip}` : ''}`;
    }
    r.g.setAttribute('aria-label', head);
  }
  renderEffects(body?.effects ?? []);
  renderMeta(body);
  refreshTooltip();
}

function renderEffects(effects) {
  const byId = new Map(effects.map((e) => [e.id, e]));
  const rows = EFFECTS.map(([id, ko]) => ({ id, ko, e: byId.get(id) }));
  for (const e of effects) if (!EFFECTS.some(([id]) => id === e.id)) rows.push({ id: e.id, ko: e.label_ko, e });
  clear(effectsEl);
  for (const { ko, e } of rows) {
    const level = e?.level ?? 'ok';
    effectsEl.append(el('li', {
      class: 'fx-chip', dataset: { level, tipTitle: `${e?.label_ko ?? ko} · ${LEVEL_KO[level] ?? level}`, tip: e?.detail ?? '해당 없음' },
      text: e?.label_ko ?? ko,
    }));
  }
}

function renderMeta(body) {
  const box = $('body-meta');
  clear(box);
  if (!body) return;
  const { hand, foot } = body;
  if (hand) box.append(el('div', { text: `손 ${hand.fingers}지 · 파지 여유 ${hand.grip_margin == null ? '—' : fmt.n(hand.grip_margin)}` }));
  if (foot) box.append(el('div', { text: `발 ${foot.base} · 속도 ×${fmt.n(foot.speed_factor)} · 지형 여유 ${foot.terrain_margin == null ? '—' : fmt.n(foot.terrain_margin)}` }));
}

function highlight(partId, on) {
  const r = refs.get(partId);
  if (!r) return;
  r.g.classList.toggle('hl', on);
  r.li.classList.toggle('hl', on);
}

export function initBodyHud() {
  build();
  render(null);
  const hud = $('body-hud');
  hud.addEventListener('mouseover', (e) => {
    const part = e.target.closest?.('[data-part]')?.dataset.part;
    for (const id of refs.keys()) highlight(id, id === part);
  });
  hud.addEventListener('mouseleave', () => { for (const id of refs.keys()) highlight(id, false); });
  $('bh-reset').addEventListener('click', () => resetBody());
  $('bh-toggle').addEventListener('click', () => {
    const folded = hud.classList.toggle('folded');
    $('bh-toggle').setAttribute('aria-expanded', String(!folded));
    $('bh-toggle').setAttribute('aria-label', folded ? '신체 상태 펼치기' : '신체 상태 접기');
    $('bh-toggle').textContent = folded ? '▸' : '▾';
  });
  bus.on('body', render);
  if (state.body) render(state.body);
}
