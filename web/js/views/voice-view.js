/**
 * ==============================================================================
 * MyHome AI - 方案 A 两级语音口令 + 声纹核验视图控制器 (voice-view.js)
 * 职责：支持 10 种通行口令展示、浏览器端 16kHz 单声道 WAV 录音录入、家人声纹管理与独立开门测试
 * ==============================================================================
 */

import { api } from '../api.js';
import { store } from '../store.js';

function encodeMono16kWavBase64(float32Samples, inputSampleRate) {
  const targetRate = 16000;
  let pcm16k = float32Samples;

  if (inputSampleRate !== targetRate && float32Samples.length > 1) {
    const outLen = Math.max(1, Math.round((float32Samples.length * targetRate) / inputSampleRate));
    pcm16k = new Float32Array(outLen);
    const ratio = (float32Samples.length - 1) / Math.max(1, outLen - 1);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = float32Samples[idx] || 0;
      const s1 = float32Samples[Math.min(idx + 1, float32Samples.length - 1)] || 0;
      pcm16k[i] = s0 * (1 - frac) + s1 * frac;
    }
  }

  const numSamples = pcm16k.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm16k[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return 'data:audio/wav;base64,' + btoa(binary);
}

async function recordMicWavBase64(durationMs = 1800) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('当前浏览器环境不支持麦克风录音，请使用上传 WAV 文件功能');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
  });
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  let totalLen = 0;

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input.length);
    copy.set(input);
    chunks.push(copy);
    totalLen += copy.length;
  };

  source.connect(processor);
  processor.connect(ctx.destination);

  await new Promise((resolve) => setTimeout(resolve, durationMs));

  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const sampleRate = ctx.sampleRate;
  await ctx.close();

  const merged = new Float32Array(totalLen);
  let pos = 0;
  for (const c of chunks) {
    merged.set(c, pos);
    pos += c.length;
  }
  return encodeMono16kWavBase64(merged, sampleRate);
}

export class VoiceView {
  constructor() {
    this.enrolledDraftWav = '';
    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    const btnReload = document.getElementById('btnReloadVoiceDb');
    const btnOpenModal = document.getElementById('btnOpenVoiceModal');
    const btnCloseModal = document.getElementById('btnCloseVoiceModal');
    const btnCancelModal = document.getElementById('btnCancelVoiceModal');
    const btnRecordEnroll = document.getElementById('btnRecordEnrollVoice');
    const fileEnrollInput = document.getElementById('voiceFileEnrollInput');
    const btnSubmitEnroll = document.getElementById('btnSubmitVoiceEnroll');

    const btnRecordVerify = document.getElementById('btnRecordTestVoice');
    const fileVerifyInput = document.getElementById('voiceVerifyFileInput');
    const membersGrid = document.getElementById('voiceMembersGrid');

    btnReload?.addEventListener('click', async () => {
      const res = await api.reloadVoice();
      await store.fetchVoiceSummary();
      window.showToast?.(res.models_ready ? '✅ 语音模型与声纹库已重新加载' : 'ℹ️ 已刷新声纹目录 (等待导出 ONNX 模型)');
    });

    const closeModal = () => {
      const modal = document.getElementById('voiceUploadModal');
      if (modal) modal.classList.remove('active');
      this.enrolledDraftWav = '';
      const tip = document.getElementById('voiceEnrollStatusTip');
      if (tip) tip.innerText = '尚未录制或选择 WAV 音频';
    };

    btnOpenModal?.addEventListener('click', () => {
      const modal = document.getElementById('voiceUploadModal');
      if (modal) modal.classList.add('active');
    });
    btnCloseModal?.addEventListener('click', closeModal);
    btnCancelModal?.addEventListener('click', closeModal);

    btnRecordEnroll?.addEventListener('click', async () => {
      try {
        btnRecordEnroll.disabled = true;
        btnRecordEnroll.classList.add('recording');
        btnRecordEnroll.innerText = '🔴 正在录音 (1.8s)... 请说出所选口令';
        this.enrolledDraftWav = await recordMicWavBase64(1800);
        const tip = document.getElementById('voiceEnrollStatusTip');
        if (tip) tip.innerText = '✅ 已录制 1.8s 16kHz 单声道 WAV，点击“确认录入声纹”即可保存';
      } catch (err) {
        window.showToast?.('❌ 录音失败: ' + err.message);
      } finally {
        btnRecordEnroll.disabled = false;
        btnRecordEnroll.classList.remove('recording');
        btnRecordEnroll.innerText = '🎙️ 点击麦克风录制口令 (1.8秒)';
      }
    });

    fileEnrollInput?.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        this.enrolledDraftWav = reader.result;
        const tip = document.getElementById('voiceEnrollStatusTip');
        if (tip) tip.innerText = `✅ 已选择音频文件: ${file.name}`;
      };
      reader.readAsDataURL(file);
    });

    btnSubmitEnroll?.addEventListener('click', async () => {
      const nameInput = document.getElementById('voiceMemberNameInput');
      const cmdSelect = document.getElementById('voiceCommandSelect');
      const person = (nameInput?.value || '').trim();
      const command = (cmdSelect?.value || '').trim();

      if (!person) {
        window.showToast?.('请输入家庭成员姓名（最多支持 10 位家人）');
        return;
      }
      if (!this.enrolledDraftWav) {
        window.showToast?.('请先点击录音或选择 .wav 音频文件');
        return;
      }

      btnSubmitEnroll.disabled = true;
      try {
        const res = await api.uploadVoice(person, command, this.enrolledDraftWav);
        if (res.error) {
          window.showToast?.('❌ 录入失败: ' + res.error);
        } else {
          window.showToast?.(`✅ 已为【${person}】录入声纹样本`);
          closeModal();
          await store.fetchVoiceSummary();
        }
      } catch (err) {
        window.showToast?.('❌ 请求异常: ' + err.message);
      } finally {
        btnSubmitEnroll.disabled = false;
      }
    });

    btnRecordVerify?.addEventListener('click', async () => {
      try {
        btnRecordVerify.disabled = true;
        btnRecordVerify.classList.add('recording');
        btnRecordVerify.innerText = '🔴 正在聆听口令 (1.8s)...';
        const wavB64 = await recordMicWavBase64(1800);
        await this.runVoiceVerify(wavB64);
      } catch (err) {
        window.showToast?.('❌ 录音失败: ' + err.message);
      } finally {
        btnRecordVerify.disabled = false;
        btnRecordVerify.classList.remove('recording');
        btnRecordVerify.innerText = '🎙️ 按下说话直接核验开门 (1.8秒)';
      }
    });

    fileVerifyInput?.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        await this.runVoiceVerify(reader.result);
        fileVerifyInput.value = '';
      };
      reader.readAsDataURL(file);
    });

    membersGrid?.addEventListener('click', async (e) => {
      const delBtn = e.target.closest('[data-del-voice]');
      if (!delBtn) return;
      const fname = delBtn.getAttribute('data-del-voice');
      if (!fname) return;
      const res = await api.deleteVoice(fname);
      if (res.status === 'success') {
        window.showToast?.('🗑️ 已删除声纹样本: ' + fname);
        await store.fetchVoiceSummary();
      } else {
        window.showToast?.('❌ 删除失败: ' + (res.error || '未知错误'));
      }
    });
  }

  async runVoiceVerify(base64Wav) {
    const res = await api.verifyVoice(base64Wav);
    this.renderVerifyResult(res);
    await store.fetchStatus();
    if (res.passed) {
      window.showToast?.(`🔓 ${res.message}`);
    } else {
      window.showToast?.(`ℹ️ ${res.message || '未通过语音核验'}`);
    }
  }

  renderVerifyResult(res) {
    const msgEl = document.getElementById('voiceVerifyMsg');
    const stageEl = document.getElementById('voiceResStage');
    const cmdEl = document.getElementById('voiceResCommand');
    const personEl = document.getElementById('voiceResPerson');
    const latEl = document.getElementById('voiceResLatency');

    if (msgEl) {
      msgEl.innerText = res.message || '-';
      msgEl.style.color = res.passed ? 'var(--success)' : (res.status === 'blocked' ? 'var(--danger)' : 'var(--warning)');
    }
    if (stageEl) stageEl.innerText = res.stage || res.status || '-';
    if (cmdEl) {
      const cScore = typeof res.command_score === 'number' ? ` (${res.command_score.toFixed(2)})` : '';
      cmdEl.innerText = (res.command && res.command !== 'NONE') ? `${res.command}${cScore}` : '未命中口令';
    }
    if (personEl) {
      const sScore = typeof res.speaker_score === 'number' && res.speaker_score > 0 ? ` (${res.speaker_score.toFixed(2)})` : '';
      personEl.innerText = (res.person && res.person !== 'None') ? `${res.person}${sScore}` : '-';
    }
    if (latEl) {
      latEl.innerText = typeof res.latency_ms === 'number' ? `${res.latency_ms.toFixed(1)} ms` : '-';
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      const summary = state.voiceSummary;
      if (!summary) return;

      const modelBadge = document.getElementById('voiceModelReadyBadge');
      if (modelBadge) {
        modelBadge.className = summary.models_ready ? 'badge badge-family' : 'badge badge-warning';
        modelBadge.innerText = summary.models_ready ? '两级 ONNX 已就绪' : '待导出 ONNX 权重';
      }

      const quotaText = document.getElementById('voiceMemberQuotaText');
      if (quotaText) {
        quotaText.innerText = `${summary.enrolled_member_count || 0} / ${summary.max_members || 10} 位家人`;
      }

      const threshText = document.getElementById('voiceThresholdsText');
      if (threshText) {
        const ct = typeof summary.command_threshold === 'number' ? summary.command_threshold.toFixed(2) : '0.55';
        const st = typeof summary.similarity_threshold === 'number' ? summary.similarity_threshold.toFixed(2) : '0.60';
        threshText.innerText = `口令 ≥ ${ct} · 声纹 ≥ ${st}`;
      }

      const cmds = summary.unlock_commands || [];
      const chipsWrap = document.getElementById('voiceCommandsList');
      if (chipsWrap) {
        chipsWrap.innerHTML = cmds.map((c, idx) =>
          `<span class="voice-cmd-chip"><span>#${idx + 1}</span><span>「${c}」</span></span>`
        ).join('');
      }

      const cmdSelect = document.getElementById('voiceCommandSelect');
      if (cmdSelect && cmdSelect.options.length <= 1 && cmds.length > 0) {
        cmdSelect.innerHTML = `<option value="">通用声纹模板 (不限特定口令)</option>` +
          cmds.map((c) => `<option value="${c}">专属口令模板：「${c}」</option>`).join('');
      }

      const grid = document.getElementById('voiceMembersGrid');
      if (grid) {
        const members = summary.members || [];
        if (members.length === 0) {
          grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 28px; color: var(--muted); background: #080d16; border-radius: 12px; border: 1px dashed #1f2e45;">
            尚未录入家庭成员声纹样本。点击右上角「➕ 录入家人语音口令」，每位家人建议对常用口令录制 2~3 条样本。
          </div>`;
        } else {
          grid.innerHTML = members.map((m) => {
            const filesHtml = (m.files || []).map((f) => {
              const parts = f.split('__');
              const tagLabel = parts.length >= 3 ? `口令: ${parts[1]}` : '通用声纹';
              return `<div class="voice-sample-row">
                <div>
                  <div style="font-weight: 600; color: var(--text);">${tagLabel}</div>
                  <div style="color: var(--muted); font-size: 11px;">${f}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                  <audio controls preload="none" src="/api/voice_file?file=${encodeURIComponent(f)}"></audio>
                  <button class="btn btn-danger btn-sm" data-del-voice="${f}" style="padding: 4px 8px;">删除</button>
                </div>
              </div>`;
            }).join('');

            return `<div class="voice-member-card">
              <div class="voice-member-head">
                <div style="font-size: 15px; font-weight: 700; color: var(--accent);">👤 ${m.name}</div>
                <span class="badge badge-family">${m.count || 0} 条声纹样本</span>
              </div>
              <div class="voice-sample-list">${filesHtml}</div>
            </div>`;
          }).join('');
        }
      }
    });
  }
}

