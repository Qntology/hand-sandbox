// 합성 데이터(Lance) 기록, 사전 적재, 테이블 상태. 기존 app.js 의 data 영역을 옮겼다.
import { call } from './api.js';
import { state } from './state.js';
import { $, clear, el } from './util.js';
import { setStatus, toast } from './toast.js';

const TABLE_NAMES = {
  samples: 'samples (에피소드)',
  contacts: 'contacts (접촉점)',
  joint_states: 'joint_states (관절)',
  dictionary: 'dictionary (사전)',
  point_cloud: 'point_cloud',
  vision: 'vision (RGB·깊이·마스크)',
  tactile: 'tactile (촉각)',
  embeddings: 'embeddings (SigLIP2)',
};

export async function refreshDataset() {
  let rows;
  try {
    rows = await call('dataset_summary', { storeDir: null });
  } catch {
    return;
  }
  const table = $('table-status');
  clear(table);
  const head = table.insertRow();
  head.append(el('th', { text: '테이블' }), el('th', { text: '상태' }), el('th', { text: '행 수', class: 'num' }));
  for (const row of rows) {
    const r = table.insertRow();
    r.insertCell().textContent = TABLE_NAMES[row.name] ?? row.name;
    r.insertCell().textContent = row.status === 'recorded' ? '기록' : '예정';
    const n = r.insertCell();
    n.className = 'num';
    n.textContent = row.status === 'recorded' ? row.rows.toLocaleString('ko-KR') : '—';
    if (row.name === 'dictionary' && row.status === 'recorded') {
      $('dict-note').textContent = `사전 ${state.dict.length}개 항목 · Lance dictionary 테이블 ${row.rows.toLocaleString('ko-KR')}행`;
    }
  }
}

async function generateDataset() {
  const count = Number($('count').value);
  const seed = Number($('synth-seed').value);
  const label = $('label').value;
  $('synth').disabled = true;
  setStatus(`${count}개 에피소드 시뮬레이션 후 Lance에 기록 중…`);
  try {
    const s = await call('synthesize', { count, seed, label, storeDir: null });
    const msg = `기록 완료: ${s.episodes}개 에피소드 중 안정 파지 ${s.stable}개 → ${s.store_dir}`;
    setStatus(msg);
    toast('합성 데이터 기록 완료', msg, 'ok');
    await refreshDataset();
  } catch {
    /* api-error 가 상태줄에 표시한다 */
  } finally {
    $('synth').disabled = false;
  }
}

async function loadDictionary() {
  for (const id of ['load-dict', 'load-dict-synth']) $(id).disabled = true;
  try {
    const n = await call('load_dictionary', { storeDir: null });
    const msg = `사전 ${n}개 항목을 Lance dictionary 테이블에 기록했습니다`;
    setStatus(msg);
    toast('사전 적재 완료', msg, 'ok');
    await refreshDataset();
  } catch {
    /* api-error 가 상태줄에 표시한다 */
  } finally {
    for (const id of ['load-dict', 'load-dict-synth']) $(id).disabled = false;
  }
}

export async function initDataPanel() {
  $('synth').addEventListener('click', generateDataset);
  $('load-dict').addEventListener('click', loadDictionary);
  $('load-dict-synth').addEventListener('click', loadDictionary);
  try {
    state.paths = await call('default_paths');
    $('store-note').textContent = `Lance 저장 위치: ${state.paths.store_dir}`;
  } catch {
    /* 경로를 못 읽어도 나머지는 동작 */
  }
  await refreshDataset();
}
