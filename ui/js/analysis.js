// 분석: 비전 점수(perception_scan), 명령 해석 trace, SDS 상태(sds_status). 기록·분석 모달의 탭에 그린다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { describeDestination, summarize } from './command-bar.js';

const MODE_NOTE = { oracle: '모의 인식기(렌더러 정답 + 오류 주입)', embedding: 'SigLIP2 임베딩', siglip2: 'SigLIP2', mock: '모의 인식기(브라우저)' };
const SCAN_TOP_K = 8;
const SDS_POLL_MS = 5000;

// ── 비전 점수 ───────────────────────────────────────────────────
function scoreList(labels) {
  return el('ul', { class: 'score-list' }, labels.map((l) => {
    const pct = Math.max(0, Math.min(1, l.score)) * 100;
    return el('li', { class: 'score-item' },
      el('div', { class: 'row' },
        el('span', { class: 'name', text: l.label_ko }),
        el('span', { class: `choice-badge ${l.accepted ? 'done' : 'rule'}`, text: l.accepted ? '수락' : '보류' }),
        el('span', { class: 'val', text: `${pct.toFixed(1)}%` })),
      el('div', { class: `bar-pill${l.accepted ? '' : ' lv-warn'}`, role: 'img', aria: { label: `${l.label_ko} 점수 ${pct.toFixed(1)}%` } }, el('span', { style: `width:${pct}%` })),
      el('div', { class: 'sub', text: `${l.id} · margin ${fmt.n(l.margin, 3)}${l.reason ? ` · ${l.reason}` : ''}` }));
  }));
}

function renderVision() {
  const scan = state.perception;
  const meta = scan
    ? `모드 ${MODE_NOTE[scan.mode] ?? scan.mode} · 장치 ${scan.device} · 타일 ${scan.tiles} · 임베딩 ${fmt.n(scan.embed_ms, 1)} ms`
    : '아직 실행하지 않았습니다. 인식 실행을 누르면 현재 프레임을 타일로 나눠 사전 라벨 점수를 계산합니다.';
  $('scan-meta').textContent = meta;
  const box = clear($('vision-bars'));
  if (scan?.labels?.length) box.append(scoreList(scan.labels));
  else if (scan) box.append(el('p', { class: 'empty-note', text: '탐지된 라벨이 없습니다.' }));
}

let scanning = false;
export async function runScan() {
  if (scanning) return;
  scanning = true;
  $('scan-run').disabled = true;
  try {
    state.perception = await call('perception_scan', { topK: SCAN_TOP_K });
    bus.emit('perception', state.perception);
  } catch {
    /* api-error 가 표시 */
  } finally {
    scanning = false;
    $('scan-run').disabled = false;
  }
}

// ── 명령 해석 ───────────────────────────────────────────────────
function kvRow(k, v) {
  return [el('span', { class: 'k', text: k }), el('span', { class: 'v', text: v })];
}

function traceView(cmd) {
  if (!cmd) return el('p', { class: 'empty-note', text: '아직 명령이 없습니다. 아래 입력줄에 명령을 입력하면 해석 단계(정규화 → 어휘 일치 → n-gram 점수 → NMS → 역할 배정)가 여기에 남습니다.' });
  const p = cmd.parsed;
  const wrap = el('div', {});
  wrap.append(el('div', { class: 'pv-kv', style: 'margin-bottom:10px' },
    ...kvRow('입력', cmd.text ?? ''),
    ...kvRow('결과', cmd.ok ? '성공' : '실패'),
    ...kvRow('모드', String(cmd.mode ?? '').includes('embedding') ? '임베딩 + 어휘' : '어휘 전용 (lexical)'),
    ...(p ? [
      ...kvRow('요약', summarize(p)),
      ...kvRow('행동', p.action),
      ...kvRow('아이템', p.item_label_ko ? `${p.item_label_ko} (${p.item_id ?? '—'})` : '—'),
      ...kvRow('수량', p.count == null ? '—' : String(p.count)),
      ...kvRow('목적지', describeDestination(p.destination) ?? '—'),
      ...kvRow('신뢰도', fmt.pct(p.confidence)),
    ] : []),
    ...kvRow('퀘스트', cmd.quest_id ?? '—')));
  if (cmd.error) wrap.append(el('p', { class: 'hint err', text: cmd.error }));
  const rows = cmd.trace ?? [];
  if (rows.length) {
    wrap.append(el('table', { class: 'pv-table' },
      el('thead', {}, el('tr', {}, ['단계', '텍스트', '점수', '역할', '수락'].map((h, i) => el('th', { text: h, class: i === 2 ? 'num' : null })))),
      el('tbody', {}, rows.map((r) => el('tr', {},
        el('td', { class: 'mono', text: r.stage }),
        el('td', { text: r.text }),
        el('td', { class: 'num', text: r.score == null ? '—' : fmt.n(r.score, 3) }),
        el('td', { text: r.role ?? '—', style: 'white-space:nowrap' }),
        el('td', {}, el('span', { class: r.accepted ? 'ok' : 'bad', text: r.accepted ? '수락' : '기각' })))))));
  }
  return wrap;
}

function renderTrace() {
  clear($('trace-view')).append(traceView(state.lastCommand));
}

// ── SDS ─────────────────────────────────────────────────────────
function sdsView(sds) {
  if (!sds) return el('p', { class: 'empty-note', text: 'SDS 상태를 불러오지 못했습니다.' });
  const wrap = el('div', {});
  wrap.append(el('div', { class: 'pv-kv', style: 'margin-bottom:10px' }, ...kvRow('원장 폴더', sds.dir), ...kvRow('레시피', sds.recipe)));
  wrap.append(el('table', { class: 'pv-table', style: 'margin-bottom:12px' },
    el('thead', {}, el('tr', {}, ['원장', '범위 수', '관측', '이번 실행'].map((h, i) => el('th', { text: h, class: i ? 'num' : null })))),
    el('tbody', {}, (sds.ledgers ?? []).map((l) => el('tr', {},
      el('td', { class: 'mono', text: l.ledger }), el('td', { class: 'num', text: String(l.scopes) }),
      el('td', { class: 'num', text: String(l.observations) }), el('td', { class: 'num', text: String(l.run_observations) }))))));
  if (sds.precision) {
    const p = sds.precision;
    const e = p.eval ?? {};
    const pct = (v) => (v == null ? '—' : fmt.pct(v, 1));
    const blocked = Object.entries(p.blocked_by ?? {}).map(([g, n]) => `${g} ${n}`).join(' · ') || '—';
    wrap.append(el('div', { class: 'ct-sec', text: '정밀도 게이트' }),
      el('div', { class: 'pv-kv', style: 'margin-bottom:10px' },
        ...kvRow('인식기', MODE_NOTE[p.mode] ?? p.mode),
        ...kvRow('게이트', p.gates_enabled ? '켜짐' : '꺼짐'),
        ...kvRow('판정 / 수락', `${p.evaluated} / ${p.accepted}`),
        ...kvRow('차단 사유', blocked),
        ...kvRow('랜드마크', `${p.landmarks?.total ?? 0}개 (확인 ${p.landmarks?.confirmed ?? 0}, 기각 ${p.landmarks?.rejected ?? 0})`),
        ...(p.mode === 'oracle' ? [
          ...kvRow('정밀도 원시 → 게이트', `${pct(e.raw_precision)} → ${pct(e.gated_precision)}`),
          ...kvRow('재현율 원시 → 게이트', `${pct(e.raw_recall)} → ${pct(e.gated_recall)}`),
        ] : [])));
  }
  if (sds.confusions?.length) {
    wrap.append(el('div', { class: 'ct-sec', text: '헷갈린 쌍' }),
      el('table', { class: 'pv-table', style: 'margin-bottom:12px' },
        el('thead', {}, el('tr', {}, ['범위', '쌍', '동률', 'a 우세', 'b 우세', '평균 margin'].map((h, i) => el('th', { text: h, class: i > 1 ? 'num' : null })))),
        el('tbody', {}, sds.confusions.map((c) => el('tr', {},
          el('td', { class: 'mono', text: c.scope }), el('td', { class: 'mono', text: c.pair }),
          el('td', { class: 'num', text: String(c.ties) }), el('td', { class: 'num', text: String(c.a_wins) }),
          el('td', { class: 'num', text: String(c.b_wins) }), el('td', { class: 'num', text: fmt.n(c.mean_margin, 3) }))))));
  }
  if (sds.axes?.length) {
    wrap.append(el('table', { class: 'pv-table' },
      el('thead', {}, el('tr', {}, ['범위', '축', '종류', 'n', '평균', 'σ', '최근', '드리프트 z'].map((h, i) => el('th', { text: h, class: i > 2 ? 'num' : null })))),
      el('tbody', {}, sds.axes.map((a) => el('tr', {},
        el('td', { class: 'mono', text: a.scope }), el('td', { class: 'mono', text: a.axis }), el('td', { text: a.kind }),
        el('td', { class: 'num', text: String(a.n) }), el('td', { class: 'num', text: fmt.n(a.mean, 3) }),
        el('td', { class: 'num', text: fmt.n(a.sd, 3) }), el('td', { class: 'num', text: fmt.n(a.recent, 3) }),
        el('td', { class: `num ${Math.abs(a.drift_z) > 2 ? 'warn' : ''}`, text: fmt.n(a.drift_z, 2) }))))));
  } else {
    wrap.append(el('p', { class: 'empty-note', text: '아직 축 통계가 없습니다. 관측이 쌓이면 표시됩니다.' }));
  }
  if (sds.log?.length) {
    wrap.append(el('div', { class: 'ct-sec', text: 'SDS 최근 기록' }),
      el('div', { class: 'hint', style: 'font-family:var(--font-mono);white-space:pre-wrap' }, sds.log.slice(-8).join('\n')));
  }
  return wrap;
}

function renderSds() {
  const box = $('an-sds');
  clear(box);
  box.append(sdsView(state.sds));
}

export async function refreshSds() {
  try {
    state.sds = await call('sds_status', {}, { quiet: true });
    bus.emit('sds', state.sds);
  } catch {
    /* 조용히 */
  }
}

export function initAnalysis() {
  $('scan-run').addEventListener('click', runScan);
  $('an-sds-refresh').addEventListener('click', refreshSds);
  bus.on('perception', renderVision);
  bus.on('command', renderTrace);
  bus.on('sds', renderSds);
  renderVision();
  renderTrace();
  renderSds();

  // 재생 중 자동 인식(4초마다)
  setInterval(() => {
    if ($('scan-auto').checked && state.playing && !document.hidden) runScan();
  }, 4000);
  // 시계열·SDS 탭이 보이는 동안 SDS 상태를 주기적으로 갱신
  let tab = null;
  setInterval(() => {
    if (tab === 'ts' && !document.hidden) refreshSds();
  }, SDS_POLL_MS);
  bus.on('tab', ({ group, name }) => {
    if (group !== 'analysis') return;
    tab = name;
    if (name === 'ts') refreshSds();
    if (name === 'vision' && !state.perception) runScan(); // 처음 열 때 한 번 자동 인식
  });
  return refreshSds();
}
