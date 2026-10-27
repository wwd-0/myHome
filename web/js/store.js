/**
 * ==============================================================================
 * MyHome AI - 前端响应式状态管理中枢 (store.js)
 * 职责：集中管理全局状态、定时轮询与事件广播，消除零散的全局变量污染
 * ==============================================================================
 */

import { api } from './api.js';

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
        last_result: null,
        history: []
      },
      members: []
    };

    this.listeners = [];
    this.pollInterval = null;
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

  async fetchStatus() {
    try {
      const data = await api.getStatus();
      this.state.status = { ...this.state.status, ...data };
      this.notify();
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

  startPolling(intervalMs = 3000) {
    this.fetchStatus();
    if (this.pollInterval) clearInterval(this.pollInterval);
    this.pollInterval = setInterval(() => {
      this.fetchStatus();
    }, intervalMs);
  }
}

export const store = new StateStore();
