// 필수 모델 관리: SigLIP2 · TTM · TIPSv2 · SAM 을 받고(진행률·속도·남은 시간), 지우고, 불러온다.
// 네 모델이 모두 장치에 올라가기 전에는 월드가 돌지 않으므로(백엔드가 거부), 시작할 때 게이트 창을 띄우고 기다린다.
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { setStatus, toast } from './toast.js';

const POLL_MS = 500;
let cat = null; // 마지막 models_catalog
let polling = null;
let activating = null; // models_activate 진행 중인 Promise
const waiters = [];
const armed = new Map(); // 삭제 확인 대기: id → 타이머

const MB = 1024 * 1024;
function size(b) {
  if (!Number.isFinite(b)) return '—';
  if (b >= 1024 * MB) return `${(b / 1024 / MB).toFixed(2)} GB`;
  if (b >= MB) return `${(b / MB).toFixed(1)} MB`;
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${b} B`;
}

function eta(s) {
  if (!Number.isFinite(s)) return '';
  if (s < 60) return `${Math.ceil(s)}초`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}분 ${Math.round(s % 60)}초` : `${Math.floor(m / 60)}시간 ${m % 60}분`;
}

const LOADING = { siglip2: 'SigLIP2 를 불러오고 사전 문구를 임베딩하는 중', ttm: 'Granite TTM 을 불러오는 중', tips: 'TIPSv2 깊이 모델을 불러오는 중', sam: 'MobileSAM 을 불러오는 중' };

function downloading() {
  return !!cat?.download?.running;
}

function missingBytes(c = cat) {
  return (c?.models ?? []).filter((m) => !m.present).reduce((a, m) => a + (m.bytes - m.have), 0);
}

/** 모델 한 줄: 이름·역할·크기, 진행 막대, 받기/삭제. */
function modelRow(m, compact) {
  const dl = cat.download;
  const active = dl?.running && dl.model === m.id;
  const queued = dl?.running && !active && (dl.queue ?? []).includes(m.id) && !m.present;
  const pct = m.bytes ? Math.min(100, (m.have / m.bytes) * 100) : 0;
  const loadingNow = cat.loading === m.id;
  let stateText;
  const dev = cat.live_devices?.[m.id];
  if (m.present) stateText = loadingNow ? '불러오는 중…' : cat.ready ? `사용 중${dev ? ` · ${dev}` : ''}` : '받음';
  else if (active) stateText = dl.phase === 'verify' ? '검사 중 (SHA-256)…' : `받는 중 ${pct.toFixed(1)}%`;
  else if (queued) stateText = '대기';
  else if (m.have > 0) stateText = `일부만 받음 ${pct.toFixed(0)}% — 이어 받기 가능`;
  else stateText = '없음';
  const lv = m.present ? 'lv-ok' : active ? '' : m.have > 0 ? 'lv-warn' : 'lv-bad';
  const pending = armed.has(m.id);
  const del = el('button', {
    type: 'button', class: `btn btn-sm btn-outline${pending ? ' btn-danger' : ''}`,
    disabled: downloading() || !!activating || (m.have === 0 && !m.present),
    title: `${m.dir} 의 파일을 지웁니다. 지우면 다시 받아 불러올 때까지 시뮬레이터가 멈춥니다.`,
    text: pending ? '정말 삭제' : '삭제',
    on: { click: () => onDelete(m) },
  });
  const get = el('button', {
    type: 'button', class: 'btn btn-sm btn-outline',
    disabled: m.present || downloading() || !!activating,
    title: `${m.repo} @ ${m.revision.slice(0, 10)} 에서 받습니다 (${m.license})`,
    text: m.present ? '받음' : m.have > 0 ? '이어 받기' : '받기',
    on: { click: () => startDownload([m.id]) },
  });
  return el('div', { class: 'model-row', dataset: { model: m.id } },
    el('div', { class: 'model-head' },
      el('span', { class: 'model-name', text: m.label }),
      el('span', { class: 'model-size', text: size(m.bytes) }),
    ),
    compact ? null : el('div', { class: 'model-role', text: `${m.role_ko} · ${m.repo}` }),
    el('div', { class: `bar-pill ${lv}`, role: 'progressbar', aria: { valuemin: '0', valuemax: '100', valuenow: String(Math.round(pct)), label: `${m.label} 받기 진행률` } }, el('span', { style: `width:${pct}%` })),
    el('div', { class: 'model-state' },
      el('span', { class: `model-badge ${m.present ? 'ok' : active ? 'run' : 'miss'}`, text: stateText }),
      el('span', { class: 'model-actions' }, get, del),
    ),
  );
}

function summaryLine() {
  const dl = cat.download ?? {};
  if (dl.running) {
    const pct = dl.total ? (dl.done / dl.total) * 100 : 0;
    const rate = dl.rate_bps > 0 ? ` · ${size(dl.rate_bps)}/s` : '';
    const left = dl.eta_s != null && dl.rate_bps > 0 ? ` · 남은 시간 ${eta(dl.eta_s)}` : '';
    const what = dl.model ? `${(cat.models.find((m) => m.id === dl.model) ?? {}).label ?? dl.model} · ${dl.file ?? ''}` : '';
    return { pct, text: `${size(dl.done)} / ${size(dl.total)} (${pct.toFixed(1)}%)${rate}${left} — ${what}` };
  }
  if (cat.loading) return { pct: 100, text: `${LOADING[cat.loading] ?? cat.loading}… (${cat.device.backend.toUpperCase()} #${cat.device.ordinal})` };
  if (dl.phase === 'error') return { pct: 0, text: `받기 실패: ${dl.error}. 다시 누르면 받은 데까지 이어서 받습니다.`, err: true };
  if (cat.load_error) return { pct: 0, text: cat.load_error, err: true };
  if (dl.phase === 'cancelled') return { pct: 0, text: '받기를 취소했습니다. 받은 부분(.part)은 남아 있어 이어 받을 수 있습니다.' };
  const hint = cat.gpu_hint ? ` ${cat.gpu_hint}` : '';
  if (cat.ready) return { pct: 100, text: `네 모델 모두 ${cat.device.backend.toUpperCase()} #${cat.device.ordinal} 에 올라가 있습니다.${hint}`, err: !!cat.gpu_hint };
  if (cat.all_present) return { pct: 100, text: '모든 파일을 받았습니다. 불러오기를 누르면 장치에 올립니다.' };
  return { pct: 0, text: `받아야 할 용량 ${size(missingBytes())} — 받기 전에는 시뮬레이터가 실행되지 않습니다.` };
}

function render() {
  if (!cat) return;
  for (const [listId, compact] of [['model-list', true], ['model-gate-list', false]]) {
    const list = $(listId);
    if (list) list.replaceChildren(...cat.models.map((m) => modelRow(m, compact)));
  }
  const s = summaryLine();
  for (const id of ['model-dl-summary', 'model-gate-summary']) {
    const n = $(id);
    if (!n) continue;
    n.textContent = s.text;
    n.classList.toggle('err', !!s.err);
  }
  const bar = $('model-gate-bar');
  if (bar) {
    bar.firstElementChild.style.width = `${s.pct}%`;
    bar.setAttribute('aria-valuenow', String(Math.round(s.pct)));
    bar.classList.toggle('lv-ok', !!cat.ready);
  }
  const busy = downloading() || !!activating;
  for (const id of ['model-get-all', 'model-gate-get']) {
    const b = $(id);
    if (!b) continue;
    b.disabled = busy || cat.all_present;
    b.textContent = cat.all_present ? '모두 받음' : `모두 받기 (${size(missingBytes())})`;
  }
  for (const id of ['model-cancel', 'model-gate-cancel']) {
    const b = $(id);
    if (b) b.disabled = !downloading();
  }
  for (const id of ['model-load', 'model-gate-load']) {
    const b = $(id);
    if (b) {
      b.disabled = busy || !cat.all_present || cat.ready;
      b.textContent = cat.ready ? '불러옴' : activating ? '불러오는 중…' : '불러오기';
    }
  }
  const gate = $('model-gate');
  if (gate) gate.hidden = !(cat.required && !cat.ready);
  document.body.classList.toggle('models-blocked', !!(cat.required && !cat.ready));
}

async function refresh() {
  try {
    cat = await call('models_catalog', {}, { quiet: true });
  } catch {
    cat = null;
    return null;
  }
  state.modelCatalog = cat;
  render();
  return cat;
}

function startPolling() {
  if (polling) return;
  polling = setInterval(async () => {
    const before = cat?.download?.phase;
    await refresh();
    if (!cat) return;
    if (!downloading() && !activating) {
      clearInterval(polling);
      polling = null;
      const phase = cat.download?.phase;
      if (phase === 'done' && before !== 'done') {
        toast('모델 받기 완료', '파일 크기와 SHA-256 을 확인했습니다. 장치에 불러옵니다.', 'ok', 5000);
      } else if (phase === 'error') {
        toast('모델 받기 실패', cat.download.error ?? '', 'err', 8000);
      }
      if (cat.all_present && !cat.ready) activate();
    }
  }, POLL_MS);
}

async function startDownload(ids) {
  try {
    cat = await call('models_download', { ids: ids ?? null });
    render();
    setStatus(ids ? `${ids.join(', ')} 받는 중…` : '필수 모델을 받는 중…');
    startPolling();
  } catch {
    /* api-error 가 이유를 표시 */
  }
}

async function onCancel() {
  try {
    cat = await call('models_cancel');
    render();
  } catch {
    /* 표시됨 */
  }
}

async function onDelete(m) {
  if (!armed.has(m.id)) {
    armed.set(m.id, setTimeout(() => { armed.delete(m.id); render(); }, 4000));
    render();
    return;
  }
  clearTimeout(armed.get(m.id));
  armed.delete(m.id);
  try {
    cat = await call('models_delete', { id: m.id });
    setStatus(`${m.label} 파일을 지웠습니다 (${size(m.bytes)}). 다시 받기 전에는 시뮬레이터가 멈춥니다.`);
    bus.emit('models-changed', cat);
  } catch {
    /* 표시됨 */
  }
  render();
}

/** 네 모델을 선택한 장치에 올린다. 진행 중이면 그 Promise 를 돌려준다. */
export function activate() {
  if (activating) return activating;
  if (!cat?.all_present) return Promise.resolve(false);
  setStatus('필수 모델을 장치에 불러오는 중… (처음에는 사전 문구 임베딩 때문에 오래 걸립니다)');
  activating = call('models_activate')
    .then((c) => {
      cat = c;
      if (c.status) {
        state.models = c.status;
        bus.emit('models', state.models);
      }
      setStatus(`모델 준비 완료 — SigLIP2 · TTM · TIPSv2 · SAM (${c.device.backend.toUpperCase()} #${c.device.ordinal})`);
      return true;
    })
    .catch(() => false)
    .finally(() => {
      activating = null;
      refresh().then(settle);
    });
  startPolling();
  render();
  return activating;
}

function settle() {
  if (cat?.ready || !cat?.required) {
    while (waiters.length) waiters.shift()(true);
  }
}

/** 시뮬레이터를 돌려도 될 때(필수 모델이 모두 올라갔거나, 모델이 필요 없는 백엔드) 풀리는 Promise. */
export function modelsReady() {
  if (!cat || !cat.required || cat.ready) return Promise.resolve(true);
  return new Promise((resolve) => waiters.push(resolve));
}

export async function initModels() {
  for (const [id, fn] of [
    ['model-get-all', () => startDownload(null)], ['model-gate-get', () => startDownload(null)],
    ['model-cancel', onCancel], ['model-gate-cancel', onCancel],
    ['model-load', activate], ['model-gate-load', activate],
  ]) $(id)?.addEventListener('click', fn);
  // 연산 장치를 바꾸면 백엔드가 모델을 내려놓으므로 새 장치에 다시 올린다.
  bus.on('device-changed', async () => {
    await refresh();
    if (cat?.required && !cat.ready && cat.all_present) activate();
  });
  await refresh();
  if (!cat) return; // models_catalog 가 없는 백엔드
  if (downloading()) startPolling();
  else if (cat.required && !cat.ready && cat.all_present) activate();
  render();
}
