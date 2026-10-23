// 손·발 2D 측면도 그리기. 기존 app.js 의 draw()/drawBase() 를 옮겼고,
// 큰 화면과 PiP(작은 화면) 모두에서 같은 코드로 그리도록 선·점 크기를 캔버스 크기에 맞춰 줄인다.
import { bus, state } from './state.js';
import { $, clamp, cssVar, fitCanvas, observeResize } from './util.js';

function bounds(geom, grasp) {
  let zmin = -0.16, zmax = 0.06, ymin = -0.18, ymax = 0.12;
  if (geom) {
    zmin = Math.min(zmin, -geom.palm[1]);
    for (const f of geom.fingers) {
      for (const [, y, z] of f.points) {
        zmax = Math.max(zmax, z + 0.01);
        ymax = Math.max(ymax, y + 0.01);
      }
    }
  }
  if (grasp) {
    const o = grasp.object;
    zmin = Math.min(zmin, o.center[2] - o.half[2]);
    zmax = Math.max(zmax, o.center[2] + o.half[2]);
    ymax = Math.max(ymax, o.center[1] + o.half[1]);
  }
  return { zmin, zmax, ymin, ymax };
}

function alpha(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function palette() {
  return {
    line: cssVar('--plot-grid-strong'),
    muted: cssVar('--text-muted'),
    base: cssVar('--bg-card'),
    palm: cssVar('--cv-palm'),
    finger: cssVar('--cv-finger'),
    thumb: cssVar('--cv-thumb'),
    object: cssVar('--cv-object'),
    ok: cssVar('--success'),
    warn: cssVar('--warning'),
  };
}

/** compact=true 이면 축 문구를 생략하고 여백을 줄인다(PiP). */
export function drawHandCanvas(canvas, { compact = false } = {}) {
  const fit = fitCanvas(canvas);
  if (!fit) return;
  const { ctx, W, H } = fit;
  ctx.clearRect(0, 0, W, H);
  if (!state.geometry) return;

  const C = palette();
  const u = clamp(Math.min(W / 860, H / 520), 0.3, 1.25);
  const b = bounds(state.geometry, state.grasp);
  const pad = compact ? 10 : Math.max(18, 36 * u);
  const scale = Math.min((W - 2 * pad) / (b.zmax - b.zmin), (H - 2 * pad) / (b.ymax - b.ymin));
  const X = (z) => pad + (z - b.zmin) * scale;
  const Y = (y) => H - pad - (y - b.ymin) * scale;
  const lw = (v) => Math.max(compact ? 1.5 : 2, v * u);

  // 축
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(0), Y(b.ymin));
  ctx.lineTo(X(0), Y(b.ymax));
  ctx.moveTo(X(b.zmin), Y(0));
  ctx.lineTo(X(b.zmax), Y(0));
  ctx.stroke();

  drawBase(ctx, state.geometry.base.kind, X, Y, scale, C, lw);

  // 손바닥
  const palmLen = state.geometry.palm[1];
  ctx.fillStyle = C.palm;
  ctx.strokeStyle = C.muted;
  ctx.lineWidth = lw(1.5);
  ctx.beginPath();
  ctx.rect(X(-palmLen), Y(0.012), palmLen * scale, 0.012 * scale);
  ctx.fill();
  ctx.stroke();

  // 물체
  if (state.grasp) {
    const o = state.grasp.object;
    ctx.fillStyle = alpha(C.object, 0.35);
    ctx.strokeStyle = C.object;
    ctx.lineWidth = lw(2);
    const x0 = X(o.center[2] - o.half[2]);
    const y1 = Y(o.center[1] + o.half[1]);
    const w = 2 * o.half[2] * scale;
    const h = 2 * o.half[1] * scale;
    ctx.fillRect(x0, y1, w, h);
    ctx.strokeRect(x0, y1, w, h);
  }

  // 손가락
  ctx.lineCap = 'round';
  state.geometry.fingers.forEach((f) => {
    const color = f.is_thumb ? C.thumb : C.finger;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw(5);
    ctx.beginPath();
    f.points.forEach(([, y, z], k) => (k === 0 ? ctx.moveTo(X(z), Y(y)) : ctx.lineTo(X(z), Y(y))));
    ctx.stroke();
    ctx.fillStyle = color;
    f.points.forEach(([, y, z], k) => {
      ctx.beginPath();
      ctx.arc(X(z), Y(y), lw(k === f.points.length - 1 ? 6 : 4) * (compact ? 0.8 : 1), 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // 파지 시험의 접촉점과 법선
  if (state.grasp) {
    for (const c of state.grasp.result.contacts) {
      const { y, z } = c.point;
      const palm = c.source.source === 'palm';
      ctx.fillStyle = palm ? C.warn : C.ok;
      ctx.beginPath();
      ctx.arc(X(z), Y(y), lw(5), 0, Math.PI * 2);
      ctx.fill();
      const { y: ny, z: nz } = c.normal;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = lw(2);
      ctx.beginPath();
      ctx.moveTo(X(z), Y(y));
      ctx.lineTo(X(z + nz * 0.03), Y(y + ny * 0.03));
      ctx.stroke();
    }
  }

  if (!compact) {
    ctx.fillStyle = C.muted;
    ctx.font = `${Math.max(11, 12 * Math.min(1, u + 0.2))}px ${cssVar('--font-sans')}`;
    ctx.textAlign = 'right';
    ctx.fillText('손끝 방향 →  z (m)', W - 12, H - 12);
    ctx.textAlign = 'left';
    ctx.fillText('↑ y 손바닥 법선', 12, 18);
  }
}

function drawBase(ctx, kind, X, Y, scale, C, lw) {
  ctx.strokeStyle = C.muted;
  ctx.fillStyle = C.base;
  ctx.lineWidth = lw(2);
  const wheel = (z, y, r) => {
    ctx.beginPath();
    ctx.arc(X(z), Y(y), r * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  };
  if (kind === 'wheel' || kind === 'mecanum') {
    ctx.beginPath();
    ctx.moveTo(X(-0.11), Y(-0.1));
    ctx.lineTo(X(0.0), Y(-0.1));
    ctx.stroke();
    wheel(-0.11, -0.11, 0.03);
    wheel(0.0, -0.11, 0.03);
    if (kind === 'mecanum') {
      ctx.beginPath();
      ctx.moveTo(X(-0.125), Y(-0.11));
      ctx.lineTo(X(-0.095), Y(-0.11));
      ctx.stroke();
    }
  } else if (kind === 'track') {
    ctx.beginPath();
    ctx.roundRect(X(-0.15), Y(-0.1), 0.16 * scale, 0.05 * scale, 0.025 * scale);
    ctx.fill();
    ctx.stroke();
  } else if (kind === 'human_foot') {
    ctx.beginPath();
    ctx.moveTo(X(-0.1), Y(-0.02));
    ctx.lineTo(X(-0.1), Y(-0.12));
    ctx.lineTo(X(0.03), Y(-0.12));
    ctx.lineTo(X(0.03), Y(-0.1));
    ctx.lineTo(X(-0.04), Y(-0.02));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

/** 큰 캔버스와 PiP 를 모두 다시 그린다. 보이지 않는(크기 0) 캔버스는 건너뛴다. */
export function redrawHand() {
  const big = $('hand-canvas');
  const pip = $('pip-canvas');
  if (big) drawHandCanvas(big);
  if (pip) drawHandCanvas(pip, { compact: true });
}

export function initHandCanvas() {
  observeResize($('hand-canvas'), redrawHand);
  observeResize($('pip-canvas'), redrawHand);
  bus.on('tab', redrawHand);
}
