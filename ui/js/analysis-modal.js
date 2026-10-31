// 기록·분석 모달: 로그 줄을 누르면 그 줄의 범주에 맞는 탭으로 바로 연다.
// 탭 넷 — 🎬 통합 테스트 · 📈 시계열·SDS · 👁 비전·깊이 · 📜 명령 해석·합성 DB. 탭이 보이는 동안만 각 모듈이 폴링하도록 'tab' 을 알린다.
import { bus } from './state.js';
import { $ } from './util.js';
import { initTablist } from './layout.js';
import { isOpen, openPanel } from './overlays.js';

// 여기 없는 범주(physics · mcp · crew · device …)는 통합 테스트로 간다.
const CAT_TAB = { quest: 'test', world: 'test', sds: 'ts', ts: 'ts', vision: 'vision', search: 'trace', embed: 'trace' };
export const TAB_LINK_KO = { test: '🔍 통합테스트 열기', ts: '🔍 시계열·SDS 열기', vision: '🔍 비전·깊이 열기', trace: '🔍 명령해석·DB 열기' };

export const tabForCat = (cat) => CAT_TAB[cat] ?? 'test';

let tabs = null;

export function openAnalysis(tab = 'test') {
  openPanel('analysis-modal');
  tabs.select(tab);
}

export function initAnalysisModal() {
  tabs = initTablist($('analysis-tabs'), (name) => {
    if (isOpen('analysis-modal')) bus.emit('tab', { group: 'analysis', name });
  });
  bus.on('panel', ({ id, open }) => {
    if (id === 'analysis-modal' && !open) bus.emit('tab', { group: 'analysis', name: null });
  });
  $('tl-live').addEventListener('click', () => openAnalysis('test'));
}
