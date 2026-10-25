// Tauri invoke 래퍼. window.__TAURI__ 가 없으면(브라우저 단독 실행) 모의 백엔드로 자동 전환한다.
import { bus, state } from './state.js';
import { errText } from './util.js';

let backend = null;

// v0.3 의 런타임 호출(통합 테스트·부품 프리셋)은 Tauri 에서 rt_call 하나로 묶여 있다.
// 개발 서버(shim)는 이름 그대로 받는다.
const RT_CALL = new Set(['test_start', 'test_step', 'test_status', 'test_keyframe', 'test_video', 'test_stats', 'test_cancel', 'body_presets', 'body_plan_eval', 'body_plan_set', 'world_body',
  'inventory_drop', 'world_reset', 'body_reset', 'world_terrain', 'nn_status', 'physics_status', 'depth_view']);

export async function initApi() {
  const tauriInvoke = window.__TAURI__?.core?.invoke;
  if (typeof tauriInvoke === 'function') {
    backend = { invoke: tauriInvoke, mock: false };
  } else {
    const mock = await import('./mock-backend.js');
    backend = { invoke: mock.invoke, mock: true };
  }
  state.mock = backend.mock;
  return backend.mock;
}

/** 원시 호출. 거부되면 Error(문자열 메시지)로 던진다. */
export async function invoke(cmd, args = {}) {
  if (!backend) await initApi();
  try {
    if (!backend.mock && !window.__devServer && RT_CALL.has(cmd)) return await backend.invoke('rt_call', { cmd, args });
    return await backend.invoke(cmd, args);
  } catch (e) {
    throw new Error(errText(e));
  }
}

/**
 * 오류를 상태줄·토스트로 알리는 호출. 폴링처럼 반복되는 호출은 { quiet: true } 로 알림을 줄인다.
 * 실패하면 그대로 던지므로 호출자는 try/catch 로 흐름만 처리하면 된다.
 */
export async function call(cmd, args = {}, { quiet = false } = {}) {
  try {
    return await invoke(cmd, args);
  } catch (e) {
    if (!quiet) bus.emit('api-error', { cmd, message: e.message });
    throw e;
  }
}
