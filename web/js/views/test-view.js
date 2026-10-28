/**
 * ==============================================================================
 * MyHome AI - AI 视觉感知与仿真测试视图控制器 (test-view.js)
 * 职责：负责图片拖拽上传、端到端模型推理、标注图像渲染与三模式仿真交互
 * ==============================================================================
 */

import { api } from '../api.js';
import { store } from '../store.js';

export class TestView {
  constructor(lockView) {
    this.lockView = lockView;
    this.currentBase64 = null;

    this.dropZone = document.getElementById('uploadDropZone');
    this.fileInput = document.getElementById('fileInput');
    this.annotatedImg = document.getElementById('annotatedImg');
    this.previewPlaceholder = document.getElementById('previewPlaceholder');

    this.latencyBadge = document.getElementById('latencyBadge');
    this.totalPersonsVal = document.getElementById('totalPersonsVal');
    this.familyCountVal = document.getElementById('familyCountVal');
    this.strangerCountVal = document.getElementById('strangerCountVal');
    this.personsListContainer = document.getElementById('personsListContainer');

    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    // 拖拽上传
    if (this.dropZone) {
      this.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.dropZone.classList.add('dragover');
      });
      this.dropZone.addEventListener('dragleave', () => this.dropZone.classList.remove('dragover'));
      this.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        this.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          this.processFile(e.dataTransfer.files[0]);
        }
      });
    }

    if (this.fileInput) {
      this.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.processFile(e.target.files[0]);
        }
      });
    }

    const btnSample = document.getElementById('btnLoadSample');
    if (btnSample) {
      btnSample.addEventListener('click', () => this.loadSamplePhoto());
    }

    const btnRelay = document.getElementById('btnTriggerRelay');
    if (btnRelay) {
      btnRelay.addEventListener('click', () => this.triggerRelayTest());
    }

    const btnSim1 = document.getElementById('btnSimMode1');
    if (btnSim1) {
      btnSim1.addEventListener('click', () => this.simulateMode1());
    }

    const btnSim2 = document.getElementById('btnSimMode2');
    if (btnSim2) {
      btnSim2.addEventListener('click', () => this.simulateMode2());
    }

    const btnSim3 = document.getElementById('btnSimMode3');
    if (btnSim3) {
      btnSim3.addEventListener('click', () => this.simulateMode3());
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      const st = state.status;
      if (!st) return;
      const descEl = document.getElementById('presenceWindowDesc');
      if (descEl && st.switch_face_window_sec) {
        descEl.innerText = `💡 互锁说明：在按下开关前后 ${st.switch_face_window_sec} 秒内识别到自家人，系统即刻确认开门！`;
      }
      const timerEl = document.getElementById('presenceWindowTimer');
      if (timerEl) {
        if (st.switch_waiting_active) {
          timerEl.innerText = `🟡 开关等待识人 (${st.switch_waiting_remaining_sec}s)`;
          timerEl.style.color = 'var(--warning)';
        } else if (st.family_presence_active) {
          timerEl.innerText = `有效 (${st.presence_remaining_sec}s) [${st.last_verified_family || ''}]`;
          timerEl.style.color = 'var(--success)';
        } else {
          timerEl.innerText = '未激活 (0s)';
          timerEl.style.color = 'var(--muted)';
        }
      }
    });
  }

  processFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.currentBase64 = e.target.result;
      this.runInference(this.currentBase64);
    };
    reader.readAsDataURL(file);
  }

  async loadSamplePhoto() {
    window.showToast?.("正在加载门前测试样本照片...");
    try {
      const resp = await fetch('/api/photo?file=爸爸_1.jpg');
      if (resp.ok) {
        const blob = await resp.blob();
        const reader = new FileReader();
        reader.onload = (e) => {
          this.currentBase64 = e.target.result;
          this.runInference(this.currentBase64);
        };
        reader.readAsDataURL(blob);
      } else {
        window.showToast?.("未能在底库找到示例照片");
      }
    } catch (err) {
      window.showToast?.("加载示例照片失败: " + err.message);
    }
  }

  async runInference(base64Data) {
    window.showToast?.("🚀 模型正在执行纯净端到端推理...");
    try {
      const res = await api.testImage(base64Data);
      this.renderInferenceResult(res);
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("推理失败: " + err.message);
    }
  }

  renderInferenceResult(res) {
    if (this.latencyBadge) {
      this.latencyBadge.innerText = `耗时: ${(res.latency_ms || 0).toFixed(1)} ms`;
    }
    if (this.totalPersonsVal) this.totalPersonsVal.innerText = res.total_persons || 0;
    if (this.familyCountVal) this.familyCountVal.innerText = res.family_count || 0;
    if (this.strangerCountVal) this.strangerCountVal.innerText = res.stranger_count || 0;

    // 标注图像
    if (res.annotated_image) {
      this.annotatedImg.src = res.annotated_image;
      this.annotatedImg.style.display = 'block';
      this.previewPlaceholder.style.display = 'none';
    }

    // 主体明细列表
    if (this.personsListContainer) {
      this.personsListContainer.innerHTML = '';
      if (!res.persons || res.persons.length === 0) {
        this.personsListContainer.innerHTML = '<div style="color: var(--muted); font-size: 13px; text-align: center; padding: 20px 0;">画面中未检测到人体目标</div>';
        return;
      }

      res.persons.forEach((p, idx) => {
        const row = document.createElement('div');
        row.className = 'person-row';
        const isFam = p.is_family;
        const badgeClass = isFam ? 'badge-family' : 'badge-stranger';
        const tagText = isFam ? '自家人' : '陌生人';

        row.innerHTML = `
          <div class="person-info">
            <div class="person-avatar">${isFam ? '👨' : '👤'}</div>
            <div>
              <div style="font-weight: 700; font-size: 14px;">#${idx + 1} 【${p.name}】</div>
              <div style="font-size: 12px; color: var(--muted);">置信度: ${(p.score * 100).toFixed(1)}% · 状态: ${p.status}</div>
            </div>
          </div>
          <span class="badge ${badgeClass}">${tagText}</span>
        `;
        this.personsListContainer.appendChild(row);
      });
    }

    // 若有自家人，点亮开门联动
    if (res.family_count > 0 && res.family_members && res.family_members.length > 0) {
      if (res.switch_interlock_unlocked) {
        window.showToast?.(`🎉 开关双向互锁命中！识别到自家人【${res.family_members.join('、')}】，门锁已自动打开！`);
        this.lockView?.animateUnlock(res.family_members[0], '人体 + 智能开关');
      } else {
        window.showToast?.(`✅ 识别到自家人：${res.family_members.join('、')}！在场有效期已激活`);
      }
      this.animateRelayLed();
    }
  }

  animateRelayLed() {
    const led = document.getElementById('relayLed');
    const title = document.getElementById('relayStatusTitle');
    const desc = document.getElementById('relayStatusDesc');
    if (led) led.classList.add('active');
    if (title) {
      title.innerText = '门锁控制单元：开锁指令生效中 (开门)';
      title.style.color = 'var(--success)';
    }
    if (desc) desc.innerText = '开门信号已触发，正在驱动门锁解锁';

    setTimeout(() => {
      if (led) led.classList.remove('active');
      if (title) {
        title.innerText = '门锁控制单元：已闭锁待命';
        title.style.color = 'var(--text)';
      }
      if (desc) desc.innerText = '已自动复位锁定，保持常闭安全状态';
    }, 500);
  }

  async triggerRelayTest() {
    window.showToast?.("正在发送开门测试信号...");
    this.animateRelayLed();
    this.lockView?.animateUnlock("模拟测试", "手动开门");
    try {
      await api.testRelay();
      window.showToast?.("开门信号已成功触发！");
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("开门失败: " + err.message);
    }
  }

  async simulateMode1() {
    window.showToast?.("仿真方式 1: 人体 + 手势 (向服务请求核验)...");
    try {
      const res = await api.verifySync();
      if (res.passed) {
        this.animateRelayLed();
        this.lockView?.animateUnlock(res.person, '人体 + 手势');
        window.showToast?.(`✅ 核验通过！[${res.person} + ${res.gesture || 'OK'}] 已触发开门`);
      } else {
        window.showToast?.("未检测到有效自家人与手势");
      }
      store.fetchStatus();
    } catch (err) {
      window.showToast?.("请求异常: " + err.message);
    }
  }

  async simulateMode2() {
    this.lockView?.handleSwitchTrigger();
  }

  async simulateMode3() {
    try {
      const res = await api.verifyVoice('芝麻开门');
      window.showToast?.("ℹ️ 提示: " + res.message);
    } catch (err) {
      window.showToast?.("声音请求异常: " + err.message);
    }
  }
}
