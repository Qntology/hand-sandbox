// 토스트 스택과 상태줄 메시지.
import { $, el } from './util.js';

let stack = null;
let msgTimer = null;

function ensureStack() {
  if (!stack) {
    stack = el('div', { class: 'toast-stack', role: 'region', aria: { label: '알림', live: 'polite' } });
    document.body.append(stack);
  }
  return stack;
}

/** kind: info | ok | warn | err */
export function toast(title, body = '', kind = 'info', ms = 4500) {
  const node = el('div', { class: `toast ${kind === 'info' ? '' : kind}`, role: kind === 'err' ? 'alert' : 'status' },
    el('div', { class: 'toast-title', text: title }),
    body ? el('div', { class: 'toast-body', text: body }) : null);
  const s = ensureStack();
  s.append(node);
  while (s.children.length > 4) s.firstElementChild.remove();
  setTimeout(() => node.remove(), ms);
  return node;
}

/** 상태줄 가운데 메시지(오류는 빨강). 오류는 8초, 일반은 12초 뒤 지운다. */
export function setStatus(message, isError = false) {
  const node = $('sb-msg');
  if (!node) return;
  node.textContent = message;
  node.classList.toggle('err', isError);
  node.title = message;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => {
    node.textContent = '';
    node.classList.remove('err');
  }, isError ? 8000 : 12000);
}
