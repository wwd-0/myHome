/**
 * ==============================================================================
 * MyHome AI - 系统状态与监控视图控制器 (system-view.js)
 * 职责：负责系统核心参数呈现、HA 联动接口与全局事件流日志监控
 * ==============================================================================
 */

import { store } from '../store.js';

export class SystemView {
  constructor() {
    this.unlockCountEl = document.getElementById('unlockCountVal');
    this.historyTableBody = document.getElementById('historyTableBody');
    this.btnRefresh = document.getElementById('btnRefreshSystem');

    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    if (this.btnRefresh) {
      this.btnRefresh.addEventListener('click', () => {
        window.showToast?.("正在刷新系统状态...");
        store.fetchStatus();
      });
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      this.render(state.status);
    });
  }

  render(st) {
    if (!st) return;

    if (this.unlockCountEl) {
      this.unlockCountEl.innerText = st.unlock_count || 0;
    }

    if (this.historyTableBody) {
      this.historyTableBody.innerHTML = '';
      if (!st.history || st.history.length === 0) {
        this.historyTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 24px;">暂无历史记录</td></tr>';
      } else {
        st.history.forEach(h => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${h.time}</td>
            <td><strong>${h.person}</strong></td>
            <td><span class="badge badge-mode">${h.auth_mode || '多模态'}</span></td>
            <td><code style="color: var(--purple); font-weight: 700;">${h.action || '开门'}</code></td>
            <td>${h.elapsed_ms}ms</td>
            <td><span class="badge badge-family">开门成功 (500ms脉冲)</span></td>
          `;
          this.historyTableBody.appendChild(tr);
        });
      }
    }
  }
}
