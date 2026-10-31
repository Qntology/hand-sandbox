// 연산 장치 선택·칩, 그리고 SigLIP2 / TTM / 깊이 / 마스크 / 임베더 / SDS 상태 줄(상태줄과 🌍 월드 팝오버 공용).
// 모델 받기·삭제·불러오기는 models.js 가 맡는다(네 모델은 필수라 끄는 스위치가 없다).
import { call } from './api.js';
import { bus, state } from './state.js';
import { $, clear, el, fmt } from './util.js';
import { openPanel } from './overlays.js';
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
  chip.title = [d.reason ? `시작 장치: ${d.reason}` : null, ...d.backends.map((b) => `${b.backend}: ${b.detail}`)].filter(Boolean).join('\n');
}

async function onDeviceChanged() {
  const [backend, ordinal] = $('device-select').value.split(':');
  try {
    await call('select_device', { backend, ordinal: Number(ordinal) });
    setStatus(`연산 장치를 ${backend.toUpperCase()}로 바꿨습니다. 필수 모델을 이 장치에 다시 불러옵니다`);
  } catch {
    /* api-error 가 표시 */
  }
  await refreshDevices();
  bus.emit('device-changed', state.device);
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

function renderToggles() {
  const live = state.models?.stand_ins;
  const NAMES = { recognizer: '인식기', locate: '의미 위치', depth: '깊이', track: '마스크', ttm: 'TTM' };
  $('model-files').textContent = Array.isArray(live) && live.length
    ? `아직 대역 모델로 도는 부분: ${live.map((k) => NAMES[k] ?? k).join(', ')} — 불러오기를 누르세요.`
    : '';
}

export function updateModelLines() {
  const p = state.perception;
  const f = state.forecast;
  const c = state.lastCommand;
  const sds = state.sds;
  const rec = state.models?.recognizer;
  $('model-siglip').textContent = p ? `${SIGLIP[p.mode] ?? p.mode} · ${p.device}` : rec ? `${SIGLIP[rec] ?? rec} · 아직 인식 전` : '미확인 — 인식 실행 시 갱신';
  const ttmDev = state.models?.devices?.ttm;
  $('model-ttm').textContent = `${ttmText(f?.method, state.models)}${ttmDev ? ` · ${ttmDev}` : ''}`;
  const ml = state.models?.ml;
  const dev = state.models?.devices;
  const stereo = state.models?.cameras === 2;
  if ($('model-depth-k')) $('model-depth-k').textContent = stereo ? '깊이 (SGBM 스테레오)' : '깊이 (TIPSv2)';
  $('model-depth').textContent = ml
    ? stereo
      ? `${ml.depth} · CPU · 기선 ${fmt.n((state.models.stereo?.baseline_m ?? 0) * 100, 1)} cm · TIPSv2 안 씀`
      : `${ml.depth} · ${dev?.depth ?? '-'}`
    : '미확인';
  $('model-mask').textContent = ml ? `${ml.track} · ${dev?.track ?? '-'}` : '미확인';
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
    openPanel('world-popover');
    $('device-select').focus();
  });
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

