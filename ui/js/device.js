// 연산 장치 선택·칩, 그리고 SigLIP2 / TTM / 임베더 / SDS 상태 줄(상태줄과 ④ 사전·모델 공용).
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el } from './util.js';
import { openSection } from './layout.js';
import { setStatus } from './toast.js';

export async function refreshDevices() {
  let d;
  try {
    d = await call('devices');
  } catch {
    return;
  }
  state.device = d;
  const sel = $('device-select');
  clear(sel);
  for (const b of d.backends) {
    sel.append(el('option', {
      value: `${b.backend}:0`, disabled: !b.available, title: b.detail,
      text: `${b.backend.toUpperCase()}${b.available ? '' : ' (사용 불가)'}`,
    }));
  }
  sel.value = `${d.selected}:${d.ordinal}`;
  const chip = $('sb-device');
  chip.textContent = `연산: ${d.selected.toUpperCase()} #${d.ordinal}`;
  chip.title = d.backends.map((b) => `${b.backend}: ${b.detail}`).join('\n');
}

async function onDeviceChanged() {
  const [backend, ordinal] = $('device-select').value.split(':');
  try {
    await call('select_device', { backend, ordinal: Number(ordinal) });
    setStatus(`연산 장치를 ${backend.toUpperCase()}로 바꿨습니다. 다음 분류에서 모델을 다시 불러옵니다`);
  } catch {
    /* api-error 가 표시 */
  }
  await refreshDevices();
}

// ── 모델 상태 줄 ────────────────────────────────────────────────
const SIGLIP = {
  oracle: '오라클(정답 ID) — SigLIP2 미사용',
  embedding: 'SigLIP2 임베딩 — 스크린샷 타일',
  siglip2: 'SigLIP2 사용 중',
  mock: '모의 인식기',
};

/** models_status 가 알려 주는 연결 상태가 있으면 그것을, 없으면 예측 방법 이름에서 짐작한다. */
function ttmText(method, models) {
  if (models && models.ttm === false) {
    return models.ttm_files && !models.ttm_files.present
      ? '꺼짐 — last/ewma/holt 융합 (가중치 폴더 없음)'
      : '꺼짐 — last/ewma/holt 융합';
  }
  if (models && models.ttm === true) {
    return method && /\bttm\b/.test(method) ? `TTM 융합 중 · ${method}` : 'TTM 연결됨 — 다음 예측부터 반영';
  }
  if (!method) return '미확인';
  if (method.startsWith('fusion')) return `융합 (${method.replace(/^fusion\(?|\)$/g, '') || '—'})`;
  if (method.startsWith('fused')) return `융합 (${method.replace(/^fused:?/, '') || '—'})`;
  if (method.startsWith('ttm')) return 'TTM-R3 사용 중';
  if (method.startsWith('holt')) return 'Holt 감쇠로 대체 (TTM 가중치 없음·이력 부족)';
  return method;
}

/** 현재 인식기·TTM 연결 상태를 가져온다(없는 백엔드면 조용히 넘어간다). */
export async function refreshModels({ quiet = true } = {}) {
  try {
    state.models = await call('models_status', {}, { quiet });
    bus.emit('models', state.models);
  } catch {
    /* models_status 가 없는 백엔드 */
  }
  return state.models;
}

let switching = false;

async function switchModel(cmd, args, doing, done) {
  if (switching) return;
  switching = true;
  renderToggles();
  setStatus(doing);
  try {
    state.models = await call(cmd, args);
    bus.emit('models', state.models);
    if (cmd === 'recognizer_use') {
      state.perception = null; // 다른 인식기의 결과가 남지 않게
      bus.emit('perception', null);
    }
    setStatus(done(state.models));
  } catch {
    /* api-error 가 이유를 표시한다 */
  } finally {
    switching = false;
    renderToggles();
    updateModelLines();
  }
}

function onSiglipToggle() {
  const on = state.models?.embedding === true;
  return switchModel(
    'recognizer_use',
    { mode: on ? 'oracle' : 'embedding' },
    on ? '오라클 인식기로 돌아가는 중…' : 'SigLIP2 를 불러오고 사전 문구를 임베딩하는 중… (처음에는 오래 걸립니다)',
    (m) => (m.embedding ? 'SigLIP2 인식으로 바꿨습니다. 다음 인식부터 스크린샷 타일을 임베딩합니다' : '오라클 인식기로 바꿨습니다'),
  );
}

function onTtmToggle() {
  const on = state.models?.ttm === true;
  return switchModel(
    'ttm_use',
    { enabled: !on },
    on ? 'Granite TTM 을 해제하는 중…' : 'Granite TTM 을 불러오는 중…',
    (m) => (m.ttm ? 'Granite TTM 을 연결했습니다. 다음 예측부터 융합에 참여합니다' : 'Granite TTM 을 해제했습니다'),
  );
}

function renderToggles() {
  const m = state.models;
  const sg = $('model-siglip-toggle');
  const tt = $('model-ttm-toggle');
  const known = !!m;
  sg.disabled = switching || !known;
  tt.disabled = switching || !known;
  sg.textContent = m?.embedding ? 'SigLIP2 끄기 (오라클로)' : 'SigLIP2 인식 켜기';
  tt.textContent = m?.ttm ? 'TTM 끄기' : 'TTM 켜기';
  sg.setAttribute('aria-pressed', String(m?.embedding === true));
  tt.setAttribute('aria-pressed', String(m?.ttm === true));
  const missing = [];
  if (m?.siglip_files && !m.siglip_files.present) missing.push(`SigLIP2: ${m.siglip_files.dir} (SIGLIP2_DIR 로 바꿀 수 있음)`);
  if (m?.ttm_files && !m.ttm_files.present) missing.push(`TTM: ${m.ttm_files.dir} (TTM_DIR 로 바꿀 수 있음)`);
  $('model-files').textContent = missing.length ? `모델 파일을 찾지 못했습니다 — ${missing.join(' · ')}. 켜기를 누르면 이유를 알려 드립니다.` : '';
}

export function updateModelLines() {
  const p = state.perception;
  const f = state.forecast;
  const c = state.lastCommand;
  const sds = state.sds;
  const rec = state.models?.recognizer;
  $('model-siglip').textContent = p ? `${SIGLIP[p.mode] ?? p.mode} · ${p.device}` : rec ? `${SIGLIP[rec] ?? rec} · 아직 인식 전` : '미확인 — 인식 실행 시 갱신';
  $('model-ttm').textContent = ttmText(f?.method, state.models);
  $('model-embed').textContent = c ? (c.mode === 'embedding' ? '임베딩 + 어휘' : '어휘 전용 (lexical)') : state.models ? (state.models.parser_embedding ? '임베딩 + 어휘 (연결됨)' : '어휘 전용 (lexical)') : '미확인 — 명령 실행 시 갱신';
  $('model-sds').textContent = sds?.recipe ?? '미확인';
  $('sb-sds').textContent = sds?.recipe ?? '-';
  $('sb-embed').textContent = c ? c.mode : '-';
  $('sb-ttm').textContent = f ? f.method : '-';
  $('sum-model').textContent = state.device ? `${state.device.selected.toUpperCase()} · ${p ? p.mode : rec ?? '—'}` : '';
}

export function initDevice() {
  $('device-select').addEventListener('change', onDeviceChanged);
  $('sb-device').addEventListener('click', () => {
    openSection('secModel');
    $('device-select').focus();
  });
  $('model-siglip-toggle').addEventListener('click', onSiglipToggle);
  $('model-ttm-toggle').addEventListener('click', onTtmToggle);
  $('model-refresh').addEventListener('click', async () => {
    await refreshModels({ quiet: false });
    renderToggles();
    updateModelLines();
    setStatus('모델 상태를 확인했습니다');
  });
  for (const ev of ['perception', 'forecast', 'command', 'sds', 'models']) bus.on(ev, updateModelLines);
  bus.on('models', renderToggles);
  renderToggles();
  updateModelLines();
  return Promise.all([refreshDevices(), refreshModels().then(() => { renderToggles(); updateModelLines(); })]);
}

