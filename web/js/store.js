/**
 * ==============================================================================
 * MyHome AI - 前端响应式状态管理中枢 (store.js)
 * 职责：集中管理全局状态、自适应轮询与审计格式化，消除零散的全局变量污染
 * ==============================================================================
 */

import { api } from './api.js';

export function formatAuditPresentation(h) {
  const status = h.status || (h.passed === false ? 'blocked' : 'unlocked');

  const actionMap = {
    SWITCH_WAITING: '🔘 按下开关 · 等待识人',
    SWITCH_CLICKED: '🔘 按键开锁 (先识后按)',
    SWITCH_PREV_CLICKED: '🔘 按键开锁 (先按后识)',
    SWITCH_TIMEOUT_BLOCKED: '🛡️ 按键超时未检出家人',
    SWITCH_COOLDOWN: '⏸️ 按键触发冷却保护',
    SWITCH_SUPERSEDED: '🔄 连续按键 (刷新窗口)',
    SWITCH_CANCELLED: '⏹️ 手动休眠中止核验',
    MANUAL_BUTTON: '⚡ 控制台应急点动开锁',
    COOLDOWN_PROTECTED: '⏸️ 冷却期防抖拦截'
  };

  const actionLabel = actionMap[h.action] || (h.action && h.action !== 'None' ? `✌️ 手势核验 (${h.action})` : '门禁事件');

  let badgeClass = 'badge-family';
  let badgeText = '✅ 开锁成功';
  let personStyle = '';

  if (status === 'pending') {
    badgeClass = 'badge-mode';
    const rem = typeof h.remaining_sec === 'number' && h.remaining_sec > 0 ? ` ${h.remaining_sec.toFixed(1)}s` : '';
    badgeText = `⏳ 等待核验${rem}`;
    personStyle = 'color: var(--accent);';
  } else if (status === 'blocked') {
    badgeClass = 'badge-stranger';
    badgeText = '🛡️ 安全拦截 (未开门)';
    personStyle = 'color: var(--danger);';
  } else if (status === 'cooldown') {
    badgeClass = 'badge-warning';
    badgeText = '⏸️ 冷却保护拦截';
    personStyle = 'color: var(--warning);';
  } else if (status === 'superseded') {
    badgeClass = 'badge-mode';
    badgeText = '🔄 已刷新窗口';
    personStyle = 'color: var(--muted);';
  } else if (status === 'cancelled') {
    badgeClass = 'badge-warning';
    badgeText = '⏹️ 已手动中止';
    personStyle = 'color: var(--muted);';
  }

  return {
    status,
    actionLabel,
    badgeClass,
    badgeText,
    personStyle,
    detail: h.detail || ''
  };
}

class StateStore {
  constructor() {
    this.state = {
      activeModule: 'lock', // 默认激活模块: 门锁管理
      status: {
        state: 'IDLE_SLEEP',
        unlock_count: 0,
        switch_face_window_sec: 5.0,
        window_sec: 15.0,
        family_presence_active: false,
        presence_remaining_sec: 0.0,
        last_verified_family: 'None',
        switch_waiting_active: false,
        switch_waiting_remaining_sec: 0.0,
        cooldown_active: false,
        cooldown_remaining_sec: 0.0,
        snapshot_enabled: true,
        snapshot_root_dir: './output/snapshots',
        snapshot_count: 0,
        last_result: null,
        history: []
      },
      members: [],
      snapshotsSummary: null
    };

    this.listeners = [];
    this.pollInterval = null;
    this.currentIntervalMs = 3000;
  }

  // 状态订阅 (Observer Pattern)
  subscribe(fn) {
    this.listeners.push(fn);
    fn(this.state);
    return () => {
      this.listeners = this.listeners.filter(l => l !== fn);
    };
  }

  notify() {
    for (const fn of this.listeners) {
      try {
        fn(this.state);
      } catch (err) {
        console.error('[Store Notify Error]:', err);
      }
    }
  }

  setActiveModule(moduleName) {
    if (this.state.activeModule !== moduleName) {
      this.state.activeModule = moduleName;
      this.notify();
    }
  }

  adjustPollingRate(st) {
    const isBusy = Boolean(
      st && (
        st.switch_waiting_active ||
        st.family_presence_active ||
        st.state === 'VERIFYING_STREAM' ||
        st.cooldown_active
      )
    );
    const targetMs = isBusy ? 500 : 3000;
    if (this.pollInterval && this.currentIntervalMs !== targetMs) {
      clearInterval(this.pollInterval);
      this.currentIntervalMs = targetMs;
      this.pollInterval = setInterval(() => {
        this.fetchStatus();
      }, targetMs);
    }
  }

  async fetchStatus() {
    try {
      const prevSnapCount = this.state.status.snapshot_count;
      const data = await api.getStatus();
      this.state.status = { ...this.state.status, ...data };
      this.adjustPollingRate(this.state.status);
      this.notify();
      if (this.state.activeModule === 'snapshots' && data.snapshot_count !== prevSnapCount) {
        this.fetchSnapshots();
      }
      return this.state.status;
    } catch (err) {
      console.warn('[Store] 状态轮询警告:', err.message);
    }
  }

  async fetchMembers() {
    try {
      const data = await api.getMembers();
      this.state.members = data.members || [];
      this.notify();
      return data;
    } catch (err) {
      console.warn('[Store] 成员列表拉取失败:', err.message);
    }
  }

  async fetchSnapshots() {
    try {
      const data = await api.getSnapshots();
      this.state.snapshotsSummary = data;
      this.notify();
      return data;
    } catch (err) {
      console.warn('[Store] 快照列表拉取失败:', err.message);
    }
  }

  startPolling(intervalMs = 3000) {
    this.currentIntervalMs = intervalMs;
    this.fetchStatus();
    if (this.pollInterval) clearInterval(this.pollInterval);
    this.pollInterval = setInterval(() => {
      this.fetchStatus();
    }, intervalMs);
  }
}

export const store = new StateStore();
