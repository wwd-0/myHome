/**
 * ==============================================================================
 * MyHome AI - 帮助与门锁规则说明视图控制器 (help-view.js)
 * 职责：展示门锁安全策略、时序互锁规则、硬件链路与常见问题说明，同步动态状态
 * ==============================================================================
 */

import { store } from '../store.js';

export class HelpView {
  constructor() {
    this.windowSecTextEl = document.getElementById('helpWindowSecText');
    this.presenceStatusEl = document.getElementById('helpPresenceStatus');
    this.unlockCountEl = document.getElementById('helpUnlockCount');

    this.subscribeStore();
  }

  subscribeStore() {
    store.subscribe((state) => {
      this.render(state.status);
    });
  }

  render(st) {
    if (!st) return;

    const winSec = st.switch_face_window_sec || 5.0;
    if (this.windowSecTextEl) {
      this.windowSecTextEl.innerText = `${winSec} 秒`;
    }

    if (this.unlockCountEl) {
      this.unlockCountEl.innerText = `${st.unlock_count || 0} 次`;
    }

    if (this.presenceStatusEl) {
      if (st.switch_waiting_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--warning);">🟡 开关已按下 · 等待自家人识别 (${st.switch_waiting_remaining_sec}s)</span>`;
      } else if (st.family_presence_active) {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--success);">🟢 自家人在场 (${st.presence_remaining_sec}s) [${st.last_verified_family || ''}]</span>`;
      } else {
        this.presenceStatusEl.innerHTML = `<span style="color: var(--muted);">⚪ 待命中 (等待自家人或按键)</span>`;
      }
    }
  }
}
