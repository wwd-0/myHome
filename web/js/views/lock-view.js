/**
 * ==============================================================================
 * MyHome AI - 门锁管理中枢视图控制器 (lock-view.js)
 * 职责：负责门锁状态表盘、应急控制、互锁规则指示与安全审计流水的生命周期
 * ==============================================================================
 */

import { api } from '../api.js';
import { store } from '../store.js';

export class LockView {
  constructor() {
    this.dialEl = document.getElementById('lockDial');
    this.dialIconEl = document.getElementById('lockDialIcon');
    this.statusTitleEl = document.getElementById('lockStatusTitle');
    this.relayDescEl = document.getElementById('lockRelayDesc');
    this.countEl = document.getElementById('lockMetricUnlockCount');
    this.windowMetricEl = document.getElementById('lockMetricWindowSec');
    this.policyWindowTextEl = document.getElementById('policyWindowSecText');
    this.presenceStatusEl = document.getElementById('lockPresenceStatus');
    this.btnWakeEl = document.getElementById('btnWakeupVerify');

    // 最近开锁详情
    this.lastPassedBadge = document.getElementById('lockLastPassedBadge');
    this.lastPersonEl = document.getElementById('lockLastPerson');
    this.lastModeEl = document.getElementById('lockLastMode');
    this.lastActionEl = document.getElementById('lockLastAction');
    this.lastTimeEl = document.getElementById('lockLastTime');

    // 专属流水表格
    this.historyTableBody = document.getElementById('lockHistoryTableBody');

    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    const btnEmergency = document.getElementById('btnLockEmergency');
    if (btnEmergency) {
      btnEmergency.addEventListener('click', () => this.handleEmergencyUnlock());
    }

    const btnSwitch = document.getElementById('btnLockSwitch');
    if (btnSwitch) {
      btnSwitch.addEventListener('click', () => this.handleSwitchTrigger());
    }

    if (this.btnWakeEl) {
      this.btnWakeEl.addEventListener('click', () => this.handleWakeupVerify());
    }

    const btnStop = document.getElementById('btnLockStop');
    if (btnStop) {
      btnStop.addEventListener('click', () => this.handleStopVerify());
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      this.render(state.status);
    });
  }

  render(st) {
    if (!st) return;

    // 1. 更新指标数据
    if (this.countEl) {
      this.countEl.innerHTML = `${st.unlock_count || 0} <span style="font-size: 13px; font-weight: normal; color: var(--muted);">次</span>`;
    }
    const winSec = st.switch_face_window_sec || 5.0;
    if (this.windowMetricEl) {
      this.windowMetricEl.innerHTML = `${winSec} <span style="font-size: 13px; font-weight: normal; color: var(--muted);">秒</span>`;
    }
    if (this.policyWindowTextEl) {
      this.policyWindowTextEl.innerText = winSec;
    }

    // 2. 更新门前在场与开关互锁状态
    if (this.presenceStatusEl) {
      if (st.family_presence_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--success);">🟢 自家人在场 (${st.presence_remaining_sec}s) [${st.last_verified_family || ''}]</span>`;
      } else {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--muted);">⚪ 待命中 (等待自家人或按键)</span>`;
      }
    }

    // 3. 唤醒按钮状态
    if (this.btnWakeEl) {
      this.btnWakeEl.innerHTML = (st.state === 'VERIFYING_STREAM')
        ? '<span>📡 正在拉流核验...</span>'
        : '<span>📡 唤醒感知核验</span>';
    }

    // 4. 最近一次开锁详情
    if (st.last_result && st.last_result.passed) {
      if (this.lastPassedBadge) {
        this.lastPassedBadge.innerText = '核验开门成功';
        this.lastPassedBadge.className = 'badge badge-family';
      }
      if (this.lastPersonEl) this.lastPersonEl.innerText = st.last_result.person || '-';
      if (this.lastModeEl) this.lastModeEl.innerText = st.last_result.auth_mode || '-';
      if (this.lastActionEl) this.lastActionEl.innerText = st.last_result.action || '-';
      if (this.lastTimeEl) this.lastTimeEl.innerText = st.last_result.timestamp || '-';
    }

    // 5. 渲染流水日志
    if (this.historyTableBody) {
      this.historyTableBody.innerHTML = '';
      if (!st.history || st.history.length === 0) {
        this.historyTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 24px;">暂无开门流水记录</td></tr>';
      } else {
        st.history.forEach(h => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${h.time}</td>
            <td><strong>${h.person}</strong></td>
            <td><span class="badge badge-mode">${h.auth_mode || '多模态'}</span></td>
            <td><code style="color: var(--purple); font-weight: 700;">${h.action || '开门'}</code></td>
            <td>${h.elapsed_ms}ms</td>
            <td><span class="badge badge-family">开锁成功</span></td>
          `;
          this.historyTableBody.appendChild(tr);
        });
      }
    }
  }

  // 开门表盘动效
  animateUnlock(personName = '', modeText = '') {
    if (!this.dialEl) return;
    this.dialEl.classList.add('unlocked');
    this.dialIconEl.innerText = '🔓';
    this.statusTitleEl.innerText = '智能门锁：已开门 (已解锁)';
    this.statusTitleEl.style.color = 'var(--success)';
    this.relayDescEl.innerText = `门锁已成功解锁 · ${personName ? personName + ' 验证通过' : '开门指令已发出'}`;

    setTimeout(() => {
      this.dialEl.classList.remove('unlocked');
      this.dialIconEl.innerText = '🔒';
      this.statusTitleEl.innerText = '智能门锁：已锁闭 (安全防御中)';
      this.statusTitleEl.style.color = 'var(--text)';
      this.relayDescEl.innerText = '智能门锁：安全锁闭中 · 具备自动复位防夹闭锁保护机制';
    }, 1200);
  }

  // 应急开门处理
  async handleEmergencyUnlock() {
    window.showToast?.("⚡ 正在执行应急开锁指令...");
    this.animateUnlock("管理员 (应急开门)", "远程开锁");
    try {
      await api.testRelay();
      window.showToast?.("✅ 应急开门已执行！");
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("开门失败: " + err.message);
    }
  }

  // 智能开关互锁仿真
  async handleSwitchTrigger() {
    window.showToast?.("已按下门厅开关，等待自家人识别联动开门...");
    try {
      const res = await api.triggerSwitch();
      if (res.passed) {
        this.animateUnlock(res.person, '人体 + 智能开关');
        window.showToast?.(`🎉 开关前后 ${res.window_sec || 5} 秒内识别成功，门锁已打开 (${res.person})`);
      } else {
        window.showToast?.(`⏳ ${res.message}`);
      }
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("开关触发提示: " + err.message);
    }
  }

  // 唤醒视频流核验
  async handleWakeupVerify() {
    window.showToast?.("📡 正在唤醒门前摄像头...");
    try {
      await api.requestVerify();
      window.showToast?.("✅ 摄像头已就绪，正在识别门前人员...");
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("启动异常: " + err.message);
    }
  }

  // 强制休眠
  async handleStopVerify() {
    window.showToast?.("⏹️ 正在停止摄像头识别...");
    try {
      await api.stopVerify();
      window.showToast?.("已恢复待命就绪状态");
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("停止异常: " + err.message);
    }
  }
}
