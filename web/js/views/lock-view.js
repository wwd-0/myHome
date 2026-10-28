/**
 * ==============================================================================
 * MyHome AI - 门锁管理中枢视图控制器 (lock-view.js)
 * 职责：负责门锁状态表盘、应急控制、互锁规则指示与全量安全审计流水的生命周期
 * ==============================================================================
 */

import { api } from '../api.js';
import { store, formatAuditPresentation } from '../store.js';

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
    this.btnSwitchEl = document.getElementById('btnLockSwitch');

    // 最近事件详情
    this.lastPassedBadge = document.getElementById('lockLastPassedBadge');
    this.lastPersonEl = document.getElementById('lockLastPerson');
    this.lastModeEl = document.getElementById('lockLastMode');
    this.lastActionEl = document.getElementById('lockLastAction');
    this.lastTimeEl = document.getElementById('lockLastTime');

    // 专属流水表格
    this.historyTableBody = document.getElementById('lockHistoryTableBody');

    // 追踪当前正在等待人脸核验的按键事件 ID，用于在状态揭晓时主动弹窗提醒
    this.trackedSwitchEventId = null;

    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    const btnEmergency = document.getElementById('btnLockEmergency');
    if (btnEmergency) {
      btnEmergency.addEventListener('click', () => this.handleEmergencyUnlock());
    }

    if (this.btnSwitchEl) {
      this.btnSwitchEl.addEventListener('click', () => this.handleSwitchTrigger());
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
      if (st.switch_waiting_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--warning);">🟡 开关已按下 · 等待自家人识别 (${st.switch_waiting_remaining_sec}s)</span>`;
      } else if (st.family_presence_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--success);">🟢 自家人在场 (${st.presence_remaining_sec}s) [${st.last_verified_family || ''}] · 按键即开</span>`;
      } else if (st.cooldown_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--accent);">⏸️ 开锁冷却保护中 (${st.cooldown_remaining_sec}s)</span>`;
      } else {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--muted);">⚪ 待命中 (等待自家人或按键)</span>`;
      }
    }

    // 3. 按钮动态状态反馈
    if (this.btnSwitchEl) {
      this.btnSwitchEl.innerHTML = st.switch_waiting_active
        ? `<span>⏳ 等待自家人识别 (${st.switch_waiting_remaining_sec}s)...</span>`
        : '<span>🔘 门厅开关开门 (模拟按键)</span>';
    }

    if (this.btnWakeEl) {
      this.btnWakeEl.innerHTML = (st.state === 'VERIFYING_STREAM')
        ? '<span>📡 正在拉流核验...</span>'
        : '<span>📡 唤醒感知核验</span>';
    }

    // 4. 追踪挂起按键事件的结果揭晓提醒
    if (this.trackedSwitchEventId && Array.isArray(st.history)) {
      const tracked = st.history.find(item => item.id === this.trackedSwitchEventId);
      if (tracked && tracked.status && tracked.status !== 'pending') {
        this.trackedSwitchEventId = null;
        if (tracked.status === 'unlocked') {
          this.animateUnlock(tracked.person, '人体 + 智能开关');
          window.showToast?.(`🎉 互锁核验通过！识别到【${tracked.person}】，门锁已自动打开！`);
        } else if (tracked.status === 'blocked') {
          window.showToast?.(`🛡️ 未开门：${tracked.detail || '超时未识别到自家人'}（已记录审计流水）`);
        } else if (tracked.status === 'cooldown') {
          window.showToast?.(`⏸️ ${tracked.detail || '处于冷却保护期'}（已记录审计流水）`);
        }
      }
    }

    // 5. 最近一次门锁事件详情（无论开锁成功、等待核验还是安全拦截均清晰呈现）
    const latestEvent = (Array.isArray(st.history) && st.history.length > 0)
      ? st.history[0]
      : (st.last_result && st.last_result.action !== 'None' ? st.last_result : null);

    if (latestEvent) {
      const pres = formatAuditPresentation(latestEvent);
      if (this.lastPassedBadge) {
        this.lastPassedBadge.innerText = pres.badgeText;
        this.lastPassedBadge.className = `badge ${pres.badgeClass}`;
      }
      if (this.lastPersonEl) this.lastPersonEl.innerText = latestEvent.person || '-';
      if (this.lastModeEl) this.lastModeEl.innerText = latestEvent.auth_mode || '-';
      if (this.lastActionEl) this.lastActionEl.innerText = pres.actionLabel;
      if (this.lastTimeEl) this.lastTimeEl.innerText = latestEvent.time || latestEvent.timestamp || '-';
    }

    // 6. 渲染全量安全审计流水表
    if (this.historyTableBody) {
      this.historyTableBody.innerHTML = '';
      if (!st.history || st.history.length === 0) {
        this.historyTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 24px;">暂无门锁操作与审计流水记录</td></tr>';
      } else {
        st.history.forEach(h => {
          const pres = formatAuditPresentation(h);
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${h.time}</td>
            <td><strong style="${pres.personStyle}">${h.person}</strong></td>
            <td><span class="badge badge-mode">${h.auth_mode || '多模态'}</span></td>
            <td>
              <div style="color: var(--purple); font-weight: 700;">${pres.actionLabel}</div>
              ${pres.detail ? `<div style="font-size: 11.5px; color: var(--muted); margin-top: 2px;">${pres.detail}</div>` : ''}
            </td>
            <td>${h.elapsed_ms}ms</td>
            <td><span class="badge ${pres.badgeClass}">${pres.badgeText}</span></td>
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
      this.statusTitleEl.innerText = '智能门锁：已锁闭 (安全防护中)';
      this.statusTitleEl.style.color = 'var(--text)';
      this.relayDescEl.innerText = '门锁电控状态：安全闭锁 · 配备自动复位防夹防过热保护';
    }, 1200);
  }

  // 应急开门处理
  async handleEmergencyUnlock() {
    window.showToast?.("⚡ 正在执行应急开锁指令...");
    try {
      const res = await api.testRelay();
      if (res.passed !== false) {
        this.animateUnlock("管理员 (应急开门)", "远程开锁");
        window.showToast?.("✅ 应急开门已执行！");
      } else {
        window.showToast?.(`⏸️ ${res.message || '处于防重触发冷却期'}`);
      }
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("开门失败: " + err.message);
    }
  }

  // 智能开关互锁触发
  async handleSwitchTrigger() {
    try {
      const res = await api.triggerSwitch();
      if (res.status === 'unlocked' || res.passed) {
        this.trackedSwitchEventId = null;
        this.animateUnlock(res.person, '人体 + 智能开关');
        window.showToast?.(`🎉 ${res.message}`);
      } else if (res.status === 'cooldown') {
        this.trackedSwitchEventId = null;
        window.showToast?.(`⏸️ ${res.message}`);
      } else {
        this.trackedSwitchEventId = res.event_id || null;
        window.showToast?.(`🔘 ${res.message}`);
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
    this.trackedSwitchEventId = null;
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
