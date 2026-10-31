// AI(MCP) 3D 외형 수정: 고른 유닛에 대한 문장을 런타임이 MCP 도구 호출로 바꾸고(rig_prompt), 숨은 CAD 로 조립해(headless-cad.js)
// 검증과 프리셋 저장, 제자리 교체를 맡긴다(rig_commit). CAD 패키지가 없으면 URDF 없이 보내 런타임이 직접 만든다.
// 편집기 화면은 어디에도 나오지 않는다. 진행은 명령줄 위 한 줄과 로그 피드, 결과는 토스트로 알린다.
import { call } from './api.js';
import { cmdToast } from './toast.js';
import { headlessCad } from './headless-cad.js';
import { logLocal } from './logs.js';
import { afterUnitChange, reportApplied } from './unit-preset.js';

function logCalls(calls) {
  for (const c of calls ?? []) {
    logLocal('mcp', c.ok ? 'info' : 'warn', `🛠 ${c.tool}: ${c.result_ko ?? ''}`, c.args ?? null);
  }
}

function logChecks(r) {
  for (const c of r.checks ?? []) logLocal('mcp', c.ok ? 'info' : 'warn', `${c.ok ? '✓' : '✗'} ${c.label}: ${c.detail}`);
  for (const w of r.warnings ?? []) logLocal('mcp', 'warn', `경고: ${w}`);
}

/** 문장 하나를 유닛에 적용한다. 돌려주는 ok 가 참이면 입력줄을 비운다. */
export async function submitRigPrompt(unit, text) {
  try {
    cmdToast('info', `🛠 유닛 #${unit}: 문장을 MCP 호출로 바꾸는 중…`);
    const p = await call('rig_prompt', { unit, text }, { quiet: true });
    logCalls(p.calls);
    if (!p.ok) {
      const why = p.error_ko ?? '문장을 도구 호출로 바꾸지 못했습니다';
      logLocal('mcp', 'warn', `유닛 #${unit} 외형 수정 실패: ${why}`);
      cmdToast('err', `✕ ${why}`);
      return { ok: false };
    }
    cmdToast('info', `🛠 유닛 #${unit}: ${p.summary_ko ?? '설계 수정'} — 3D 조립 · 내보내는 중…`);
    const out = await headlessCad(p.design, `u${unit}`);
    logLocal('mcp', 'info', out ? `헤드리스 CAD 내보내기 완료: 메시 ${Object.keys(out.meshes).length}개` : '헤드리스 CAD 없음 — 런타임이 URDF 를 직접 만듭니다');
    cmdToast('info', `🛠 유닛 #${unit}: 물리 검증 · 제자리 교체 중…`);
    const r = await call('rig_commit', { unit, name: p.name, ...(out ?? {}) }, { quiet: true });
    logChecks(r);
    logLocal('mcp', r.ok ? 'info' : 'warn', `유닛 #${unit} ${r.summary_ko ?? ''}`, { preset_id: r.preset_id, mass_kg: r.mass_kg, com_m: r.com_m, eye_m: r.eye_m, sds_key: r.sds_key });
    reportApplied(r, r.ok ? `유닛 #${unit} 외형 수정` : `유닛 #${unit} 외형 수정 실패`);
    if (r.ok) {
      await afterUnitChange(r.preset_id);
      cmdToast('ok', `✓ ${r.summary_ko ?? '외형을 고쳤습니다'}`);
    } else {
      cmdToast('err', `✕ ${r.summary_ko ?? '검증을 통과하지 못했습니다'}`);
    }
    return { ok: !!r.ok };
  } catch (e) {
    logLocal('mcp', 'error', `유닛 #${unit} 외형 수정 오류: ${e.message}`);
    cmdToast('err', `✕ ${e.message}`);
    return { ok: false };
  }
}
