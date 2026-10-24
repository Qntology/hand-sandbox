// 하단 '통합 테스트' 탭: 게임 로직 → ML 로직 통합 테스트를 영상 편집기처럼 보여 준다.
//   1줄 게임: 고정 간격 키프레임 (게임 로직이 퀘스트를 수행하며 촬영)
//   2줄 ML 인식: 같은 키프레임의 깊이 컬러맵 + 추적 마스크 + 의미 위치 점
//   3줄 ML 결정: 결정(시계열·삼각 측량) → 이동 → 촬영한 키프레임
//   4줄 SDS: 키프레임마다 트랙별(깊이·위치·추적·결정·혼합·밸런스) z-점수
// 명령은 위 뷰포트 툴바(tl-cmd · 테스트 실행 · 중지)나 월드 화면 입력줄에서 낸다. '자동 기록'이 켜져 있으면
// 재생 루프(sim_step)가 기록을 진행시키고, 응답의 test 요약이 바뀔 때마다 test_status 로 줄을 다시 그린다.
// 이미지는 보이는 칸만 test_keyframe 으로 받는다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { setStatus, toast } from './toast.js';
import { submitCommand } from './command-bar.js';

const LANES = [
  { id: 'game', label: '1 게임', sub: '게임 로직 촬영', h: 'h-img' },
  { id: 'perceive', label: '2 ML 인식', sub: '깊이 · 위치 · 추적', h: 'h-img' },
  { id: 'decide', label: '3 ML 결정', sub: '시계열 → 이동 → 촬영', h: 'h-img' },
  { id: 'sds', label: '4 SDS', sub: '트랙별 z · 게임 / ML', h: 'h-sds' },
];
const SDS_KEYS = ['depth', 'locate', 'tracker', 'decide', 'fusion', 'balance'];
const SDS_KO = { depth: '깊이', locate: '위치', tracker: '추적', decide: '결정', fusion: '혼합', balance: '밸런스' };
const KIND_KO = { search: '탐색 회전', triangulate: '삼각 측량', approach: '접근', arrive: '도달', hold: '보류', explore: '시점 이동' };
const TILE_W = 96;

let status = null;
let busy = false; // 명령 요청이 오가는 중
let lastBrief = ''; // sim_step 응답의 test 요약 (바뀌면 test_status 를 다시 받는다)
let fetching = false;
let finishedId = null; // 끝난 기록의 DB 미러를 한 번만
let pps = 100; // 초당 픽셀
let sel = null; // {pass: 'game'|'ml', index}
let observer = null;
const cache = new Map(); // `${pass}:${i}` → test_keyframe 응답
const pending = new Set();

function r(x, d = 2) {
  return Number.isFinite(x) ? Number(x).toFixed(d) : '—';
}

function zClass(z) {
  if (!Number.isFinite(z)) return '';
  const a = Math.abs(z);
  return a < 1 ? 'z0' : a < 2 ? 'z1' : a < 3 ? 'z2' : 'z3';
}

// ── 레이아웃 ───────────────────────────────────────────────────
function build() {
  const pane = $('bpane-test');
  clear(pane);
  const zoom = el('select', { id: 'tl-zoom', aria: { label: '확대' }, on: { change: (e) => { pps = Number(e.target.value); renderLanes(); } } },
    ...[[50, '×0.5'], [100, '×1'], [160, '×1.6']].map(([v, t]) => el('option', { value: String(v), text: t, selected: v === pps })));
  const view = el('select', { id: 'tl-view', aria: { label: '보기' }, on: { change: (e) => setView(e.target.value) } },
    el('option', { value: 'timeline', text: '타임라인' }), el('option', { value: 'stats', text: '성공률 · N:N' }));
  const mirror = el('button', { type: 'button', class: 'vt-btn', id: 'tl-mirror', text: 'DB(Lance) 기록', title: '이번 세션의 퀘스트·키프레임·밸런스 행을 LanceDB 테이블로 복사합니다', on: { click: () => mirrorDb(false) } });
  pane.append(
    el('div', { class: 'tl-root' },
      el('div', { class: 'tl-toolbar' },
        el('span', { class: 'tl-phase', id: 'tl-phase', text: '대기' }),
        el('span', { class: 'tl-modes', id: 'tl-modes' }),
        el('span', { style: 'margin-left:auto;display:flex;gap:6px;align-items:center' }, zoom, view, mirror)),
      el('div', { class: 'tl-body', id: 'tl-body' },
        el('div', { class: 'tl-main' },
          el('div', { class: 'tl-labels', id: 'tl-labels' },
            el('div', { class: 'tl-l-ruler' }, el('small', { text: '초 (틱/10)' })),
            ...LANES.map((l) => el('div', { class: `tl-l-${l.id}`, style: `height:${l.h === 'h-sds' ? 46 : 76}px` }, l.label, el('small', { text: l.sub })))),
          el('div', { class: 'tl-scroll', id: 'tl-scroll', on: { click: onScrollClick } },
            el('div', { class: 'tl-canvas', id: 'tl-canvas' }))),
        el('aside', { class: 'tl-inspector', id: 'tl-inspector', aria: { label: '키프레임 검사기' } },
          el('div', { class: 'tl-empty', text: '키프레임을 누르면 원본 · 깊이/마스크 · 삼각 측량 · 결정 · 밸런스(N:N)가 여기에 나옵니다.' })),
        el('div', { class: 'tl-stats', id: 'tl-stats' }))));
  renderLanes();
}

function setView(v) {
  $('tl-body').classList.toggle('stats-mode', v === 'stats');
  if (v === 'stats') refreshStats();
}

function rowsOf(st) {
  const game = st?.game?.keyframes ?? [];
  const decide = st?.decide?.keyframes ?? [];
  return { game, decide };
}

function tickT(k) {
  return k.tick / 10;
}

function renderLanes() {
  const canvas = $('tl-canvas');
  if (!canvas) return;
  clear(canvas);
  observer?.disconnect();
  const { game, decide } = rowsOf(status);
  const tMax = Math.max(10, ...game.map(tickT), ...decide.map(tickT)) + 2;
  const width = Math.ceil(tMax * pps) + TILE_W + 20;
  canvas.style.width = `${width}px`;
  const ruler = el('div', { class: 'tl-ruler' });
  const step = pps >= 100 ? 5 : 10;
  for (let t = 0; t <= tMax; t += step) ruler.append(el('span', { class: 'tick', style: `left:${t * pps}px`, text: `${t}s` }));
  canvas.append(ruler);
  const lanes = {};
  for (const l of LANES) {
    lanes[l.id] = el('div', { class: `tl-lane ${l.h}`, dataset: { lane: l.id } });
    canvas.append(lanes[l.id]);
  }
  if (!status?.active) {
    lanes.game.append(el('div', { class: 'tl-empty', text: state.mock
      ? '통합 테스트는 실제 Rust 런타임에서만 동작합니다. 개발 서버(cargo run -p hand-runtime --example serve)나 Tauri 앱에서 여세요.'
      : '위 툴바의 명령 칸이나 월드 화면 입력줄에 명령을 넣고 실행하세요. "자동 기록"이 켜져 있으면 재생하는 동안 1 게임(지금 월드에서 촬영) → 2 ML 인식 → 3 ML 결정(명령을 시작한 순간의 월드를 복제해 다시 수행)이 저절로 쌓입니다.' }));
    return;
  }
  observer = new IntersectionObserver(onVisible, { root: $('tl-scroll'), rootMargin: '0px 200px' });
  // 1·2줄: 게임 키프레임. 단계 색 띠도 1줄 위에.
  let lastStep = null;
  game.forEach((k) => {
    const x = tickT(k) * pps;
    if (k.step && k.step !== lastStep) {
      lastStep = k.step;
      lanes.game.append(el('div', { class: 'band', style: `left:${x}px;width:${TILE_W}px;background:${stepColor(k.step)}`, title: k.step }));
    }
    lanes.game.append(tile('game', k, x, 'frame', `${r(tickT(k), 1)}s ${k.step ?? ''}`));
    const ml = k.ml;
    const cap = ml?.target ? `${ml.target.label} d${r(ml.target.depth_range, 1)}${ml.target.tri_range != null ? ` △${r(ml.target.tri_range, 1)}` : ''}` : (ml ? '대상 없음' : '분석 대기');
    lanes.perceive.append(tile('game', k, x, 'ml', cap, (ml?.flags ?? []).length > 0));
    lanes.sds.append(sdsCell('game', k, x, 0));
  });
  // 3줄: 결정 키프레임. 겹치지 않게 오른쪽으로 민다.
  let prevX = -1e9;
  decide.forEach((k) => {
    const x = Math.max(tickT(k) * pps, prevX + TILE_W * 0.78);
    prevX = x;
    const d = k.decision;
    const t = tile('ml', k, x, 'frame', `${KIND_KO[d?.kind] ?? '—'} ${d?.residual != null ? `잔차 ${r(d.residual, 1)}°` : ''}`, false);
    if (d?.kind) t.classList.add(`k-${d.kind}`);
    if (d && !d.ok) t.classList.add('fail');
    lanes.decide.append(t);
    lanes.sds.append(sdsCell('ml', k, tickT(k) * pps, 1));
  });
  if (sel) highlight();
}

function stepColor(step) {
  const s = String(step);
  if (s.includes('탐색')) return 'var(--text-muted)';
  if (s.includes('접근')) return 'var(--signal)';
  if (s.includes('채취')) return 'var(--warning)';
  if (s.includes('줍기')) return 'var(--violet)';
  if (s.includes('운반')) return 'var(--accent-strong)';
  return 'var(--success)';
}

function tile(pass, k, x, kind, caption, flagged = false) {
  const t = el('button', { type: 'button', class: 'tl-tile', style: `left:${x}px`, dataset: { pass, index: String(k.i), kind }, title: caption, on: { click: (e) => { e.stopPropagation(); select(pass, k.i); } } },
    el('div', { class: 'ph', text: '…' }),
    el('span', { class: 'cap', text: caption }));
  if (flagged) t.append(el('i', { class: 'flag', title: '혼합 특징 이상 표시' }));
  const c = cache.get(`${pass}:${k.i}`);
  if (c && (kind !== 'ml' || c.ml_webp)) fill(t, c);
  else observer?.observe(t);
  return t;
}

function sdsCell(pass, k, x, row) {
  const box = el('div', { class: 'tl-sds', style: `left:${x}px;top:${4 + row * 20}px`, title: `${pass === 'ml' ? 'ML 결정' : '게임'} #${k.i} SDS z: ${SDS_KEYS.map((s) => `${SDS_KO[s]} ${r(k.sds?.[s], 1)}`).join(' · ')}`, on: { click: (e) => { e.stopPropagation(); select(pass, k.i); } } });
  for (const s of SDS_KEYS) box.append(el('i', { class: zClass(k.sds?.[s]) }));
  return box;
}

function fill(t, data) {
  const src = t.dataset.kind === 'ml' ? data.ml_webp : data.frame_webp;
  const ph = t.querySelector('.ph');
  if (!src) {
    if (ph) ph.textContent = '없음';
    return;
  }
  const img = el('img', { alt: '', src: `data:image/webp;base64,${src}` });
  ph?.replaceWith(img);
}

async function fetchKf(pass, index, needMl = false) {
  const key = `${pass}:${index}`;
  const have = cache.get(key);
  // 분석 전에 받아 둔 키프레임은 분석이 끝나면 다시 받는다(2줄 그림이 그때 생긴다).
  if (have && (!needMl || have.ml_webp)) return have;
  if (pending.has(key)) return null;
  pending.add(key);
  try {
    const v = await call('test_keyframe', { pass, index }, { quiet: true });
    cache.set(key, v);
    return v;
  } catch {
    return null;
  } finally {
    pending.delete(key);
  }
}

function onVisible(entries) {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const t = e.target;
    observer.unobserve(t);
    fetchKf(t.dataset.pass, Number(t.dataset.index), t.dataset.kind === 'ml').then((v) => v && fill(t, v));
  }
}

function onScrollClick(e) {
  // 빈 곳을 누르면 그 시각에 가장 가까운 게임 키프레임을 고른다.
  const rect = $('tl-canvas').getBoundingClientRect();
  const t = (e.clientX - rect.left) / pps;
  const { game } = rowsOf(status);
  if (!game.length) return;
  const k = game.reduce((a, b) => (Math.abs(tickT(b) - t) < Math.abs(tickT(a) - t) ? b : a));
  select('game', k.i);
}

function highlight() {
  for (const t of document.querySelectorAll('#tl-canvas .tl-tile.sel')) t.classList.remove('sel');
  if (!sel) return;
  for (const t of document.querySelectorAll(`#tl-canvas .tl-tile[data-pass="${sel.pass}"][data-index="${sel.index}"]`)) t.classList.add('sel');
  const canvas = $('tl-canvas');
  canvas.querySelector('.tl-playhead')?.remove();
  const rows = rowsOf(status)[sel.pass === 'ml' ? 'decide' : 'game'];
  const k = rows.find((x) => x.i === sel.index);
  if (k) canvas.append(el('div', { class: 'tl-playhead', style: `left:${tickT(k) * pps}px` }));
}

// ── 검사기 ──────────────────────────────────────────────────────
async function select(pass, index) {
  sel = { pass, index };
  highlight();
  const insp = $('tl-inspector');
  const v = await fetchKf(pass, index, true) ?? cache.get(`${pass}:${index}`);
  if (!v || sel.pass !== pass || sel.index !== index) return;
  clear(insp);
  const b = v.brief;
  const rows = rowsOf(status)[pass === 'ml' ? 'decide' : 'game'];
  const pos = rows.findIndex((x) => x.i === index);
  const nav = (d) => {
    const k = rows[pos + d];
    if (k) select(pass, k.i);
  };
  insp.append(el('h5', {},
    el('span', { class: 'grow', text: `${pass === 'ml' ? 'ML 결정' : '게임'} #${index} · ${r(b.t, 1)} s · 틱 ${b.tick}${b.step ? ` · ${b.step}` : ''}` }),
    el('button', { type: 'button', class: 'vt-btn', text: '◀', title: '이전 키프레임', on: { click: () => nav(-1) } }),
    el('button', { type: 'button', class: 'vt-btn', text: '▶', title: '다음 키프레임', on: { click: () => nav(1) } })));
  insp.append(el('div', { class: 'tl-pics' },
    el('figure', {}, el('img', { alt: '키프레임 원본', src: `data:image/webp;base64,${v.frame_webp}` }), el('figcaption', { text: `원본 ${v.width}×${v.height} · 자세 (${b.pose.join(', ')})` })),
    el('figure', {}, v.ml_webp ? el('img', { alt: '깊이와 추적 마스크', src: `data:image/webp;base64,${v.ml_webp}` }) : el('div', { class: 'tl-empty', text: '분석 전' }),
      el('figcaption', { text: '깊이(가까움=따뜻) · 흰 윤곽=대상 마스크 · 분홍 십자=의미 위치' }))));
  const ml = v.ml;
  if (ml) {
    const t = ml.target ?? {};
    insp.append(el('div', { class: 'ct-sec', text: 'ML 트랙' }));
    insp.append(kv([
      ['깊이', `${ml.depth_mode}${ml.depth_metrics ? ` · AbsRel ${r(ml.depth_metrics.absrel)} · δ1 ${r(ml.depth_metrics.delta1)}` : ''}`],
      ['의미 위치', `${ml.locate_mode}${ml.points?.[0] ? ` · ${ml.points[0].label} ${r(ml.points[0].score)} (존재 ${r(ml.points[0].presence)})` : ''}${ml.locate_err_px != null ? ` · 오차 ${r(ml.locate_err_px, 0)} px` : ''}`],
      ['추적', `${ml.track_mode} · 마스크 ${ml.tracks?.length ?? 0}개${t.track_id != null ? ` · 대상 #${t.track_id}` : ''}${ml.track_iou_truth != null ? ` · 정답 IoU ${r(ml.track_iou_truth)}` : ''}`],
      ['방위', t.rel_bearing_deg != null ? `${r(t.rel_bearing_deg, 1)}°` : '—'],
      ['깊이 거리', t.depth_range_m != null ? `${r(t.depth_range_m)} m` : '—'],
      ['삼각 측량', t.tri_range_m != null ? `${r(t.tri_range_m)} m · 시차 ${r(t.tri_parallax_deg, 1)}° · σ ${r(t.tri_sigma_m)} m` : '—'],
      ['깊이 축척', t.depth_scale != null ? `${r(t.depth_scale)} (측량 / 깊이)` : '—'],
      ['쓴 거리 · 정답', `${t.range_used_m != null ? `${r(t.range_used_m)} m` : '—'} · ${t.range_true_m != null ? `${r(t.range_true_m)} m` : '—'}`],
    ]));
    if (ml.flags?.length) insp.append(el('div', { class: 'tl-flags' }, ...ml.flags.map((f) => el('span', { class: 'badge badge-warn', text: f }))));
  }
  const d = v.decision;
  if (d) {
    insp.append(el('div', { class: 'ct-sec', text: '결정 (시계열 · 삼각 측량)' }));
    insp.append(kv([
      ['결정', `${KIND_KO[d.kind] ?? d.kind} · ${d.decision}`],
      ['메모', d.note || '—'],
      ['예측 잔차', d.residual_deg != null ? `${r(d.residual_deg, 1)}° (예측 방위 대비 관측)` : '—'],
      ['예측기', d.forecast_method || '—'],
      ['대상 추정', d.estimate ? `(${r(d.estimate[0], 1)}, ${r(d.estimate[1], 1)}) · ${r(d.estimate[2])} m ± ${r(d.estimate[3])}` : '—'],
      ['실행', `${d.executed || '—'}${d.ok ? '' : ' · 실패'}`],
    ]));
  }
  const bal = v.balance;
  if (bal) {
    insp.append(el('div', { class: 'ct-sec', text: `밸런스 · ${bal.source_ko} (${bal.state})` }));
    insp.append(kv([
      ['안정 여유', `${fmt.pct(bal.margin)} (피드포워드 없으면 ${fmt.pct(bal.margin_noff)})`],
      ['몸통 기울임 · 머리', `${r(bal.trunk_lean_deg, 1)}° · ${r(bal.head_tilt_deg, 1)}°`],
      ['질량 · CoM 높이', `${r(bal.mass_kg, 1)} kg · ${r(bal.com_height_m)} m`],
      ['최대 부하', `${bal.worst_joint ?? '—'} ${fmt.pct(bal.worst_ratio)} · 발열 ${r(bal.heat_max)}`],
    ]));
    insp.append(table(['관절', 'N·m', '피크 대비', '연속 대비'], (bal.joints ?? []).map((j) => [j.joint, r(j.torque_nm, 1), bar(j.peak_ratio), r(j.effort)])));
    insp.append(table(['N:N 대상', '축', '값', 'FF 없음', '노력 / 없음'], (bal.couplings ?? []).map((c) => [`${c.target}`, c.axis, r(c.value), r(c.value_noff), c.effort != null ? `${r(c.effort)} / ${r(c.effort_noff)}` : '—'])));
  }
  insp.append(el('div', { class: 'ct-sec', text: 'SDS z-점수 (이 키프레임 직전 기준선 대비)' }));
  insp.append(el('div', { class: 'tl-flags' }, ...SDS_KEYS.map((s) => el('span', { class: `tl-chip ${zClass(b.sds?.[s]) === 'z0' ? 'real' : b.sds?.[s] != null ? 'stand' : ''}`, text: `${SDS_KO[s]} ${r(b.sds?.[s], 1)}` }))));
}

function kv(rows) {
  return el('div', { class: 'tl-kv' }, ...rows.flatMap(([k, v]) => [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: v })]));
}

function bar(x) {
  const w = Math.min(100, Math.max(0, (x ?? 0) * 100));
  return el('span', {}, el('span', { class: `tl-bar ${x > 1 ? 'b' : x > 0.7 ? 'w' : ''}` }, el('span', { style: `width:${w}%` })), ` ${fmt.pct(x)}`);
}

function table(head, rows) {
  return el('table', { class: 'tl-mini' },
    el('thead', {}, el('tr', {}, ...head.map((h) => el('th', { text: h })))),
    el('tbody', {}, ...rows.map((row) => el('tr', {}, ...row.map((c) => el('td', {}, c))))));
}

// ── 통계 (성공률 · N:N) ─────────────────────────────────────────
async function refreshStats() {
  const box = $('tl-stats');
  if (!box) return;
  let s;
  try {
    s = await call('test_stats', {}, { quiet: true });
  } catch (e) {
    clear(box).append(el('div', { class: 'tl-empty', text: `통계를 읽지 못했습니다: ${e.message}` }));
    return;
  }
  clear(box);
  const stats = (s.stats ?? []).slice().sort((a, b) => a.key.localeCompare(b.key));
  box.append(el('div', {},
    el('div', { class: 'ct-sec', text: '퀘스트 · 단계별 성공률 (DB quest_stats)' }),
    table(['통과 · 종류 · 대상 · 프리셋', '단계', '성공 / 시도', '성공률'], stats.map((r0) => [`${r0.pass} · ${r0.kind} · ${r0.target} · ${r0.preset}`, r0.step, `${r0.successes} / ${r0.attempts}`, fmt.pct(r0.rate)])),
    el('p', { class: 'hint', text: `파일: ${s.db_dir ?? '(메모리)'} · 피드포워드 이득(SDS 보정): ${Object.entries(s.ff_gains ?? {}).map(([k, v]) => `${k} ${r(v)}`).join(', ') || '—'}` })));
  // N:N 행렬: 원인 × 대상, 칸마다 기준이 있는 축 하나(없으면 기준선만).
  const nn = s.nn ?? [];
  const sources = [...new Set(nn.map((c) => c.source))];
  const targets = ['head', 'chest', 'belly', 'arm', 'hand', 'leg', 'foot'];
  const tko = Object.fromEntries(nn.map((c) => [c.target, c.target_ko]));
  const sko = Object.fromEntries(nn.map((c) => [c.source, c.source_ko]));
  const pick = (src, tgt) => {
    const cells = nn.filter((c) => c.source === src && c.target === tgt);
    return cells.find((c) => c.reference != null) ?? cells[0];
  };
  const grid = el('table', { class: 'tl-nn' },
    el('thead', {}, el('tr', {}, el('th', { text: '원인 \\ 대상' }), ...targets.map((t) => el('th', { text: tko[t] ?? t })))),
    el('tbody', {}, ...sources.map((src) => el('tr', {}, el('th', { text: sko[src] ?? src }),
      ...targets.map((t) => {
        const c = pick(src, t);
        if (!c) return el('td', { class: 'base', text: '' });
        const cls = c.in_band == null ? 'base' : c.in_band >= 0.6 ? 'in' : c.in_band >= 0.3 ? 'near' : 'out';
        return el('td', { class: cls, title: `${c.axis}: 평균 ${r(c.mean, 3)} (n=${c.n})${c.reference != null ? ` · 기준 ${r(c.reference, 3)} ± ${r(c.tolerance, 3)} · 편차 ${r(c.dev, 2)} · 띠 안 ${fmt.pct(c.in_band)}\n${c.cite}` : ' · 기준값 없음(기준선만)'}` },
          r(c.mean, 2), el('small', { text: c.reference != null ? `기준 ${r(c.reference, 2)}` : c.axis }));
      })))));
  box.append(el('div', {},
    el('div', { class: 'ct-sec', text: 'N:N 밸런스 (SDS · 사람 기준 대비)' }),
    grid,
    el('p', { class: 'hint', text: '초록: 기준 띠 안 60% 이상 · 노랑: 30–60% · 빨강: 30% 미만 · 회색: 기준값이 없어 기준선만 기록. 칸에 마우스를 올리면 축·출처가 보입니다.' })));
}

// ── 실행 ────────────────────────────────────────────────────────
const PHASE_KO = { game: '1/3 게임 촬영', perceive: '2/3 ML 인식', decide: '3/3 ML 결정', done: '완료' };

function renderToolbar() {
  const st = status;
  const ph = $('tl-phase');
  if (!st?.active) ph.textContent = '대기';
  else {
    const g = st.game?.keyframes?.length ?? 0;
    const d = st.decide?.keyframes?.length ?? 0;
    const p = st.perceive ?? {};
    const skip = st.ml_needed === false ? ' · ML 결정 생략(찾을 대상 없음)' : '';
    ph.textContent = `${PHASE_KO[st.phase] ?? st.phase} · "${st.command ?? ''}" · 틱 ${st.tick} · 게임 키프레임 ${g} (분석 ${p.done ?? 0}/${p.total ?? g}) · 결정 ${d} · 게임 ${st.game_quest?.state ?? '—'} · ML ${st.ml_quest?.state ?? '—'}${skip}`;
  }
  const m = st?.modes;
  const modes = $('tl-modes');
  clear(modes);
  if (m) {
    for (const [k, v] of [['깊이', m.depth], ['위치', m.locate], ['추적', m.track]]) modes.append(el('span', { class: `tl-chip ${String(v).startsWith('stand-in') ? 'stand' : 'real'}`, text: `${k} ${v}` }));
    modes.append(el('span', { class: `tl-chip ${m.ttm ? 'real' : ''}`, text: m.ttm ? 'TTM 연결' : 'TTM 없음(융합만)' }));
  }
  $('tl-run').disabled = busy;
  renderLive(null);
}

// 뷰포트 툴바의 기록 표시: ● 기록 중 · 단계 · 키프레임 수 · 대기 명령
function renderLive(brief) {
  const box = $('tl-live');
  const b = brief ?? (status?.active ? { active: true, live: status.live, phase: status.phase, game: status.game?.keyframes?.length ?? 0, decide: status.decide?.keyframes?.length ?? 0, perceived: status.perceive?.done ?? 0 } : null);
  const on = !!b?.active && b.live && b.phase !== 'done';
  box.classList.toggle('on', on);
  if (!b?.active) {
    box.textContent = b?.queued ? `대기 명령 ${b.queued}` : '';
    return;
  }
  const q = b.queued ? ` · 대기 ${b.queued}` : '';
  box.textContent = b.phase === 'done'
    ? `기록 완료 · 게임 ${b.game} · 결정 ${b.decide}${q}`
    : `${PHASE_KO[b.phase] ?? b.phase} · 게임 ${b.game}${b.phase !== 'game' ? ` · 분석 ${b.perceived}` : ''}${b.phase === 'decide' ? ` · 결정 ${b.decide}` : ''}${q}`;
  box.title = b.command ? `기록 중인 명령: ${b.command}` : '';
}

/** 툴바 '테스트 실행': 툴바 명령 칸의 문장을 입력줄과 같은 길(command_submit + 자동 기록)로 낸다. */
async function runFromToolbar() {
  const text = $('tl-cmd').value.trim();
  if (!text || busy) return;
  if (state.mock && $('tl-record').checked) toast('실제 런타임 필요', '통합 테스트 기록은 개발 서버나 Tauri 앱에서만 됩니다. 명령만 실행합니다.', 'warn');
  busy = true;
  renderToolbar();
  try {
    await submitCommand(text);
  } finally {
    busy = false;
    renderToolbar();
  }
}

async function cancel() {
  try {
    await submitCommand('멈춰', { quietHistory: true });
    await call('test_cancel', {}, { quiet: true });
  } catch {
    /* 이미 끝났으면 조용히 */
  }
  await refreshStatus(true);
}

/** sim_step 응답의 test 요약이 바뀌면 상태를 다시 받아 줄을 그린다. */
async function refreshStatus(force = false) {
  if (state.mock || fetching) return;
  fetching = true;
  try {
    const s = await call('test_status', {}, { quiet: true });
    if (s?.id !== status?.id) {
      cache.clear();
      sel = null;
    }
    status = s;
    renderLanes();
    renderToolbar();
    if (s?.active && s.phase === 'done' && finishedId !== s.id) {
      finishedId = s.id;
      setStatus(`통합 테스트 기록 완료: "${s.command}" · 게임 ${s.game_quest?.state ?? '—'} · ML ${s.ml_quest?.state ?? '—'}`);
      await mirrorDb(true);
      refreshStats();
    }
  } catch {
    /* 폴링 실패는 조용히 */
  } finally {
    fetching = false;
  }
  if (force) lastBrief = '';
}

function onSimTest(brief) {
  renderLive(brief);
  if (!brief) return;
  const key = JSON.stringify([brief.id, brief.phase, brief.game, brief.perceived, brief.decide]);
  if (key === lastBrief) return;
  lastBrief = key;
  refreshStatus();
}

async function mirrorDb(quiet) {
  if (state.mock) return;
  try {
    const r0 = await call('test_db_mirror', {}, { quiet });
    const w = r0.written ?? {};
    setStatus(`LanceDB 기록: 퀘스트 ${w.quest_runs ?? 0} · 단계 ${w.quest_steps ?? 0} · 키프레임 ${w.ml_keyframes ?? 0} · 밸런스 ${w.balance_effort ?? 0} · 성공표 ${w.quest_stats ?? 0}행 → ${r0.store_dir}`);
  } catch (e) {
    if (!quiet) toast('DB 기록 실패', e.message, 'err');
  }
}

export function initTestTimeline() {
  build();
  $('tl-run').addEventListener('click', runFromToolbar);
  $('tl-stop').addEventListener('click', cancel);
  $('tl-cmd').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); runFromToolbar(); }
  });
  bus.on('sim-test', onSimTest);
  bus.on('command', (c) => { if (c?.recording || c?.queued) refreshStatus(true); });
  bus.on('world-new', () => { status = null; cache.clear(); sel = null; renderLanes(); renderToolbar(); });
  renderToolbar();
  bus.on('tab', ({ group, name }) => {
    if (group !== 'bottom' || name !== 'test') return;
    // 네 줄과 검사기가 보이도록 하단 패널을 넉넉히 연다.
    const root = document.documentElement;
    const h = parseFloat(getComputedStyle(root).getPropertyValue('--bottom-h')) || 220;
    if (h < 380) root.style.setProperty('--bottom-h', `${Math.min(460, Math.round(window.innerHeight * 0.48))}px`);
    if (!status && !state.mock) call('test_status', {}, { quiet: true }).then((s) => { status = s; renderLanes(); renderToolbar(); }).catch(() => {});
  });
}
