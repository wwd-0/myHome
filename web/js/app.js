/**
 * ==============================================================================
 * MyHome AI - 前端应用主入口与功能树路由中枢 (app.js)
 * 职责：初始化视图控制器、管理左侧功能树导航路由与侧边栏动态拖拽
 * ==============================================================================
 */

import { store } from './store.js';
import { LockView } from './views/lock-view.js';
import { MembersView } from './views/members-view.js';
import { SystemView } from './views/system-view.js';
import { HelpView } from './views/help-view.js';
import { SnapshotsView } from './views/snapshots-view.js';
import { VoiceView } from './views/voice-view.js';

class Application {
  constructor() {
    this.setupToast();
    this.initViews();
    this.initTreeAccordion();
    this.initTreeRouter();
    this.initSidebarResizer();
    this.subscribeStatusPills();
    this.bootstrap();
  }

  // 全局 Toast 提示挂载
  setupToast() {
    const toastEl = document.getElementById('toast');
    window.showToast = (msg) => {
      if (!toastEl) return;
      toastEl.innerText = msg;
      toastEl.classList.add('show');
      setTimeout(() => toastEl.classList.remove('show'), 2800);
    };
  }

  // 初始化各模块视图控制器
  initViews() {
    this.lockView = new LockView();
    this.membersView = new MembersView();
    this.systemView = new SystemView();
    this.helpView = new HelpView();
    this.snapshotsView = new SnapshotsView();
    this.voiceView = new VoiceView();
  }

  // 一级目录折叠/展开手风琴交互
  initTreeAccordion() {
    const titles = document.querySelectorAll('.tree-group-title');
    titles.forEach(title => {
      title.addEventListener('click', (e) => {
        const group = title.closest('.tree-group');
        if (group) {
          group.classList.toggle('collapsed');
        }
      });
    });
  }

  // 左侧侧边栏动态拖拽调整宽度
  initSidebarResizer() {
    const resizer = document.getElementById('sidebarResizer');
    const sidebar = document.getElementById('sidebar');
    const root = document.documentElement;
    if (!resizer || !sidebar) return;

    const defaultWidth = 260;
    const minWidth = 190;
    const maxWidth = 520;

    // 从本地存储恢复用户设定的侧边栏宽度
    const savedWidth = localStorage.getItem('myhome_sidebar_width');
    if (savedWidth) {
      const parsed = parseInt(savedWidth, 10);
      if (!isNaN(parsed) && parsed >= minWidth && parsed <= maxWidth) {
        root.style.setProperty('--sidebar-width', `${parsed}px`);
      }
    }

    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
      isResizing = true;
      document.body.classList.add('is-resizing');
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      let newWidth = e.clientX;
      if (newWidth < minWidth) newWidth = minWidth;
      if (newWidth > maxWidth) newWidth = maxWidth;
      root.style.setProperty('--sidebar-width', `${newWidth}px`);
    });

    window.addEventListener('mouseup', () => {
      if (!isResizing) return;
      isResizing = false;
      document.body.classList.remove('is-resizing');
      const currentWidth = sidebar.getBoundingClientRect().width;
      localStorage.setItem('myhome_sidebar_width', Math.round(currentWidth));
    });

    // 双击分割手柄恢复默认 260px
    resizer.addEventListener('dblclick', () => {
      root.style.setProperty('--sidebar-width', `${defaultWidth}px`);
      localStorage.setItem('myhome_sidebar_width', defaultWidth);
      window.showToast?.("已恢复侧边栏默认宽度 (260px)");
    });
  }

  // 左侧功能树路由控制
  initTreeRouter() {
    const pageMeta = {
      lock: {
        title: '门锁控制中心',
        subtitle: '智能门锁状态监控 · 实体开关前后 5 秒智能互锁开门'
      },
      data: {
        title: '家庭成员名单管理',
        subtitle: '家庭成员名单维护与特征自动录入 · 隐私安全保护'
      },
      snapshots: {
        title: '人脸自动抓拍快照库',
        subtitle: '按身份独立子目录归档 · 支持页面配置抓拍间隔并保存至配置文件 · 超过 500 张自动清理最早 100 张'
      },
      voice: {
        title: '语音口令与声纹核验中心',
        subtitle: '方案 A 两级微模型级联 (10种口令 + 10位家人声纹) · 随摄像头拉流采集音频 · 独立直接开门'
      },
      system: {
        title: '系统运维与架构监控',
        subtitle: 'C++17 HTTP 核心 + PyTorch MPS 运行状态 · 硬件 GPIO 脉冲与 HA 联动审计'
      },
      help: {
        title: '帮助与门锁规则说明',
        subtitle: '智能开锁策略指南 · 实体开关双向互锁规则 · 硬件链路与安全机制'
      }
    };

    const treeItems = document.querySelectorAll('.tree-item');
    const pageTitleEl = document.getElementById('pageTitle');
    const pageSubtitleEl = document.getElementById('pageSubtitle');

    const switchModule = (tabKey) => {
      store.setActiveModule(tabKey);

      // 更新左侧功能树高亮与自动展开父级
      treeItems.forEach(el => {
        if (el.dataset.tab === tabKey) {
          el.classList.add('active');
          const parentGroup = el.closest('.tree-group');
          if (parentGroup) {
            parentGroup.classList.remove('collapsed');
          }
        } else {
          el.classList.remove('active');
        }
      });

      // 切换主工作区页面
      document.querySelectorAll('.tab-content').forEach(el => {
        el.classList.remove('active');
      });
      const targetContent = document.getElementById(`tab_${tabKey}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }

      // 更新顶部模块标题与说明
      if (pageMeta[tabKey]) {
        if (pageTitleEl) pageTitleEl.innerText = pageMeta[tabKey].title;
        if (pageSubtitleEl) pageSubtitleEl.innerText = pageMeta[tabKey].subtitle;
      }

      // 视图特定数据初始化
      if (tabKey === 'data') {
        store.fetchMembers();
      } else if (tabKey === 'snapshots') {
        store.fetchSnapshots();
      } else if (tabKey === 'voice') {
        store.fetchVoiceSummary();
      } else {
        store.fetchStatus();
      }
    };

    treeItems.forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const tabKey = item.dataset.tab;
        if (tabKey) switchModule(tabKey);
      });
    });

    // 允许通过 data-jump 属性跨模块跳转
    document.querySelectorAll('[data-jump]').forEach(el => {
      el.addEventListener('click', (e) => {
        e.preventDefault();
        const targetTab = el.getAttribute('data-jump');
        if (targetTab) switchModule(targetTab);
      });
    });

    // 默认激活路由
    switchModule(store.state.activeModule || 'lock');
  }

  // 顶栏实时状态指示更新
  subscribeStatusPills() {
    const presencePillText = document.getElementById('presencePillText');
    const serviceStatusText = document.getElementById('serviceStatusText');
    const statusDot = document.getElementById('statusDot');
    const sidebarStatusText = document.getElementById('sidebarStatusText');
    const sidebarDot = document.getElementById('sidebarDot');

    store.subscribe((state) => {
      const st = state.status;
      if (!st) return;

      const isStreaming = (st.state === 'VERIFYING_STREAM');
      const statusLabel = isStreaming ? '正在开门核验...' : '系统运行正常 (待命)';

      if (serviceStatusText) serviceStatusText.innerText = statusLabel;
      if (sidebarStatusText) sidebarStatusText.innerText = statusLabel;

      if (statusDot) {
        statusDot.className = isStreaming ? 'status-dot-indicator stream' : 'status-dot-indicator';
      }
      if (sidebarDot) {
        sidebarDot.className = isStreaming ? 'status-dot-indicator stream' : 'status-dot-indicator';
      }

      if (presencePillText) {
        if (st.switch_waiting_active) {
          presencePillText.innerText = `开关等待识人: ${st.switch_waiting_remaining_sec}s`;
          presencePillText.style.color = 'var(--warning)';
        } else if (st.family_presence_active) {
          presencePillText.innerText = '自家人在场: ' + (st.last_verified_family || '有') + ` (${st.presence_remaining_sec}s)`;
          presencePillText.style.color = 'var(--success)';
        } else {
          presencePillText.innerText = '自家人在场: 无';
          presencePillText.style.color = 'var(--muted)';
        }
      }
    });
  }

  // 启动后台轮询与成员数据拉取
  bootstrap() {
    store.startPolling(3000);
    store.fetchMembers();
  }
}

// 页面 DOM 加载完毕后启动应用
document.addEventListener('DOMContentLoaded', () => {
  new Application();
});
