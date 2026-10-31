// CCTV 확대 창: 한 로봇의 왼쪽 카메라 · 추정 깊이 · 렌더러 정답 깊이를 크게 나란히 보이고 카메라 자세·흔들림 지표를 함께 보인다.
// 최초 작업자는 깊이 탭과 같은 depth_view 를, 서브 작업자는 unit_frame 의 두 화면을 쓴다(서브 작업자의 정답 깊이는 런타임이 주지 않는다).
// 왼쪽 화면은 창이 열려 있는 동안 1초마다 새로 받는다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { isOpen, openPanel } from './overlays.js';
import { depthBlocks } from './depth-view.js';

const LIVE_MS = 1000;
const FRAME = { width: 480, height: 320, quality: 75 };
const FORM_KO = { humanoid: '휴머노이드', forklift: '지게차' };

let unit = null;
let liveImg = null;
let busy = false;
let seq = 0;

const info = () => state.unitInfo.get(unit) ?? state.units.find((u) => u.id === unit) ?? null;
const webp = (b64) => `data:image/webp;base64,${b64}`;

function figure(src, alt, caption) {
  return el('figure', {}, el('img', { class: 'depth-img', src, alt }), el('figcaption', { text: caption }));
}

/** 최초 작업자: depth_view 그대로. 왼쪽 그림(rgb)을 살아 있는 화면으로 바꿔 쓴다. */
async function showFirst(box, my) {
  const d = await call('depth_view', { quality: 80 });
  if (my !== seq) return;
  clear(box).append(...depthBlocks(d, { core: true }));
  liveImg = box.querySelector('.depth-img');
}

/** 서브 작업자: 기준 화면과 추정 깊이. */
async function showSub(box, my) {
  const rgb = await call('unit_frame', { unit, ...FRAME, mode: 'rgb' });
  const depth = await call('unit_frame', { unit, ...FRAME, mode: 'depth' });
  if (my !== seq) return;
  const name = depth.depth_name ?? '깊이';
  clear(box).append(
    el('div', { class: 'depth-grid' },
      figure(webp(rgb.webp_base64), '왼쪽 카메라', '① 왼쪽 카메라 (기준 RGB)'),
      figure(webp(depth.webp_base64), '추정 깊이', `② 추정 깊이 (${name})`),
      el('figure', {}, el('p', { class: 'empty-note', text: '③ 렌더러 정답 깊이와 카메라 지표는 최초 작업자만 제공됩니다.' }))));
  liveImg = box.querySelector('.depth-img');
}

async function run() {
  if (unit == null) return;
  const my = ++seq;
  const box = clear($('cz-body'));
  const u = info();
  const form = FORM_KO[u?.form] ?? '';
  const eyes = u?.cameras === 2 ? '2캠 스테레오 SGBM' : '1캠 단안 TIPSv2';
  $('cz-title').textContent = `📹 유닛 #${unit} (${[form, eyes].filter(Boolean).join(' · ')}) 라이브 비전 모니터`;
  box.append(el('p', { class: 'empty-note', text: '화면을 받는 중입니다…' }));
  liveImg = null;
  const btn = $('cz-rerun');
  btn.disabled = true;
  try {
    await (u?.first === false ? showSub(box, my) : showFirst(box, my));
  } catch (e) {
    if (my === seq) clear(box).append(el('p', { class: 'empty-note', text: `화면을 받지 못했습니다: ${e.message}` }));
  } finally {
    if (my === seq) btn.disabled = false;
  }
}

export function openZoom(id) {
  unit = id;
  openPanel('cctv-zoom');
  run();
}

async function liveTick() {
  if (busy || !liveImg || !isOpen('cctv-zoom') || document.hidden || !state.playing) return;
  busy = true;
  try {
    const r = await call('unit_frame', { unit, ...FRAME, mode: 'rgb' }, { quiet: true });
    if (liveImg) liveImg.src = webp(r.webp_base64);
  } catch {
    /* 다음 주기에 다시 */
  } finally {
    busy = false;
  }
}

export function initCctvZoom() {
  $('cz-rerun').addEventListener('click', run);
  bus.on('panel', ({ id, open }) => {
    if (id === 'cctv-zoom' && !open) {
      unit = null;
      liveImg = null;
    }
  });
  setInterval(liveTick, LIVE_MS);
}
