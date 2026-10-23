// 진입점: 백엔드 연결(Tauri 또는 모의) → 레이아웃 → 각 패널 초기화 → 첫 월드 생성.
import { initApi } from './api.js';
import { bus, state } from './state.js';
import { $ } from './util.js';
import { setStatus, toast } from './toast.js';
import { initLayout } from './layout.js';
import { initHandCanvas } from './hand-canvas.js';
import { initHandPanel } from './hand-panel.js';
import { initVisionUpload } from './vision-upload.js';
import { initDataPanel } from './data-panel.js';
import { initDevice } from './device.js';
import { initSim, newWorld } from './sim.js';
import { initWorldPanel } from './world-panel.js';
import { initWorldView } from './world-view.js';
import { initWorldMap } from './world-map.js';
import { initBodyHud } from './body-hud.js';
import { initQuestPanel } from './quest-panel.js';
import { initCommandBar } from './command-bar.js';
import { initLogs } from './logs.js';
import { initAnalysis } from './analysis.js';
import { initTimeseries } from './timeseries.js';

const recentErrors = new Map();

function wireErrors() {
  // 같은 오류가 짧은 시간에 반복되면(슬라이더 연타 등) 토스트는 한 번만 띄운다.
  bus.on('api-error', ({ cmd, message }) => {
    setStatus(`${cmd}: ${message}`, true);
    const key = `${cmd}|${message}`;
    const now = Date.now();
    if (now - (recentErrors.get(key) ?? 0) > 5000) toast(`${cmd} 오류`, message, 'err', 6000);
    recentErrors.set(key, now);
  });
  window.addEventListener('unhandledrejection', (e) => {
    console.error(e.reason);
    setStatus(`처리되지 않은 오류: ${e.reason?.message ?? e.reason}`, true);
  });
}

async function showVersion() {
  try {
    const v = await window.__TAURI__?.app?.getVersion?.();
    if (v) $('sb-version').textContent = `v${v}`;
  } catch {
    /* 버전을 못 읽어도 기본 표기 유지 */
  }
}

async function main() {
  const mock = await initApi();
  $('sb-mock').hidden = !mock;
  document.body.dataset.backend = mock ? 'mock' : 'tauri';
  if (mock) {
    document.title = '물류 로봇 손 시뮬레이터 (모의 백엔드)';
    window.__handSandbox = { state, bus }; // 브라우저 개발·스모크 테스트용
  }
  wireErrors();
  showVersion();

  initLayout();
  initHandCanvas();
  initBodyHud();
  initQuestPanel();
  initWorldView();
  initWorldMap();
  initSim();
  const regenerate = initWorldPanel();
  initCommandBar();
  initVisionUpload();
  initAnalysis();
  initTimeseries();

  // 손·발 프리셋이 있어야 월드 생성(HandConfig 전달)과 PiP 가 의미가 있으므로 먼저 기다린다.
  await initHandPanel().catch(() => {});
  await Promise.allSettled([initDevice(), initDataPanel(), initLogs()]);
  await regenerate();
  setStatus(mock
    ? '모의 백엔드로 실행 중입니다. 입력줄에 "철광석 3개 캐서 베이스캠프로 가져와"를 입력해 보세요'
    : '준비됨. 입력줄에 명령을 입력하고 Enter 를 누르세요');
}

main().catch((e) => {
  console.error(e);
  setStatus(`시작 실패: ${e.message}`, true);
  toast('시작 실패', e.message, 'err', 10000);
});
