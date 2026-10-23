// SigLIP2 이미지 분류(classify_image): 파일 선택 → top-k 막대 → 최상위 라벨을 물체로 설정.
import { call } from './api.js';
import { state } from './state.js';
import { $, clear, el } from './util.js';
import { selectObject } from './hand-panel.js';
import { setStatus } from './toast.js';

function renderScores(list) {
  const ol = $('scores');
  clear(ol);
  for (const p of list) {
    ol.append(el('li', { class: 'score-item' },
      el('div', { class: 'row' },
        el('span', { class: 'name', text: `${p.label_ko} (${p.label_en})` }),
        el('span', { class: 'val', text: `${(p.score * 100).toFixed(1)}%` })),
      el('div', { class: 'bar-pill' }, el('span', { style: `width:${Math.round(p.score * 100)}%` }))));
  }
  $('use-top').disabled = list.length === 0;
  state.topPrediction = list[0] ?? null;
}

async function onImageChosen(file) {
  if (!file) return;
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  $('preview').src = dataUrl;
  $('preview-wrap').hidden = false;
  setStatus('SigLIP2 분류 중… (첫 실행은 모델을 불러오므로 시간이 걸립니다)');
  try {
    const list = await call('classify_image', { imageBase64: dataUrl, topK: 5 });
    renderScores(list);
    if (list.length) setStatus(`분류 완료: ${list[0].label_ko} ${(list[0].score * 100).toFixed(1)}%`);
  } catch {
    renderScores([]);
  }
}

function useTopPrediction() {
  const p = state.topPrediction;
  if (!p) return;
  selectObject(p.id);
  if (p.graspable === false) setStatus(`${p.label_ko}: 파지 대상이 아닌 물체입니다`, true);
}

export function initVisionUpload() {
  $('image-file').addEventListener('change', (e) => onImageChosen(e.target.files[0]));
  $('use-top').addEventListener('click', useTopPrediction);
}
