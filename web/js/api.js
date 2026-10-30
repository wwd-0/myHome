/**
 * ==============================================================================
 * MyHome AI - 统一后端 HTTP API 客户端 (api.js)
 * 职责：纯净封装所有与 C++ 后端的异步交互，统一错误处理与请求头
 * ==============================================================================
 */

export class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  async request(path, options = {}) {
    try {
      const resp = await fetch(this.baseUrl + path, {
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {})
        },
        ...options
      });
      return await resp.json();
    } catch (err) {
      console.error(`[API Error] ${path}:`, err);
      throw err;
    }
  }

  // 获取系统全局状态、互锁信息与审计历史
  async getStatus() {
    return this.request('/api/status');
  }

  // 触发门厅智能开关 (情况 A & B 统一裁决)
  async triggerSwitch() {
    return this.request('/api/switch_trigger', { method: 'POST', body: '{}' });
  }

  // 触发 ESP32-C6 光耦继电器 500ms 脉冲 (应急开门/模拟点动)
  async testRelay() {
    return this.request('/api/test_relay', { method: 'POST', body: '{}' });
  }

  // 唤醒摄像头拉流核验
  async requestVerify() {
    return this.request('/api/verify_request', { method: 'POST', body: '{}' });
  }

  // 停止拉流恢复 0% CPU 休眠
  async stopVerify() {
    return this.request('/api/stop_verify', { method: 'POST', body: '{}' });
  }

  // 同步核验请求
  async verifySync() {
    return this.request('/api/verify', { method: 'POST', body: '{}' });
  }

  // 方案 A 两级语音核验 (独立口令 + 声纹核验开门)
  async verifyVoice(base64Audio = '') {
    return this.request('/api/verify_voice', {
      method: 'POST',
      body: JSON.stringify({ audio: base64Audio })
    });
  }

  // 获取声纹底库与 10 种通行口令配置汇总
  async getVoiceSummary() {
    return this.request('/api/voice_summary');
  }

  // 上传家人声纹样本 WAV (支持绑定特定口令模板)
  async uploadVoice(name, command, base64Audio) {
    return this.request('/api/upload_voice', {
      method: 'POST',
      body: JSON.stringify({ name, command, audio: base64Audio })
    });
  }

  // 删除指定声纹样本 WAV
  async deleteVoice(file) {
    return this.request('/api/delete_voice', {
      method: 'POST',
      body: JSON.stringify({ file })
    });
  }

  // 重新加载语音 ONNX 模型与声纹模板库
  async reloadVoice() {
    return this.request('/api/reload_voice', { method: 'POST', body: '{}' });
  }

  // 单张照片端到端推理测试
  async testImage(base64Image) {
    return this.request('/api/test_image', {
      method: 'POST',
      body: JSON.stringify({ image: base64Image })
    });
  }

  // 成员底库与模型学成类别列表
  async getMembers() {
    return this.request('/api/members');
  }

  // 上传新照片并重构特征
  async uploadPhoto(name, base64Image) {
    return this.request('/api/upload_photo', {
      method: 'POST',
      body: JSON.stringify({ name, image: base64Image })
    });
  }

  // 重新加载模型权重与底库原型
  async reloadDb() {
    return this.request('/api/reload', { method: 'POST', body: '{}' });
  }

  // 获取人脸自动抓拍快照汇总与分目录文件列表
  async getSnapshots() {
    return this.request('/api/snapshots');
  }

  // 在线修改抓拍间隔等快照配置并持久化写入 config.yaml
  async updateSnapshotConfig(payload = {}) {
    return this.request('/api/snapshot_config', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  // 删除单张快照 / 清空指定子目录 / 清空全部快照
  async deleteSnapshot(payload = {}) {
    return this.request('/api/delete_snapshot', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}

export const api = new ApiClient();
