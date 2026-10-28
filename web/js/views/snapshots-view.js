/**
 * ==============================================================================
 * MyHome AI - 人脸抓拍快照视图控制器 (snapshots-view.js)
 * 职责：展示按身份分目录保存的人脸抓拍快照，支持目录筛选、大图预览与清理管理
 * ==============================================================================
 */

import { api } from '../api.js';
import { store } from '../store.js';

export class SnapshotsView {
  constructor() {
    this.activeFolder = 'ALL';
    this.savedMinInterval = 1.0;
    this.isDirtyInterval = false;
    this.cacheDom();
    this.bindEvents();
    this.subscribeStore();
  }

  cacheDom() {
    this.rootDirText = document.getElementById('snapRootDirText');
    this.absRootDirText = document.getElementById('snapAbsRootDirText');
    this.countText = document.getElementById('snapCountText');
    this.capacityFill = document.getElementById('snapCapacityFill');
    this.policyRuleText = document.getElementById('snapPolicyRuleText');
    this.configStatusBadge = document.getElementById('snapConfigStatusBadge');
    this.intervalValueDisplay = document.getElementById('snapIntervalValueDisplay');
    this.intervalInput = document.getElementById('snapIntervalInput');
    this.btnSaveInterval = document.getElementById('btnSaveSnapInterval');
    this.presetContainer = document.getElementById('snapIntervalPresets');
    this.intervalRuleText = document.getElementById('snapIntervalRuleText');
    this.folderPillsEl = document.getElementById('snapFolderPills');
    this.gridEl = document.getElementById('snapshotsGrid');

    this.btnRefresh = document.getElementById('btnRefreshSnapshots');
    this.btnClearSubdir = document.getElementById('btnClearCurrentSubdir');
    this.btnClearAll = document.getElementById('btnClearAllSnapshots');

    // 大图预览弹窗
    this.previewModal = document.getElementById('snapshotPreviewModal');
    this.previewImg = document.getElementById('snapshotPreviewImg');
    this.previewTitle = document.getElementById('snapshotPreviewTitle');
    this.previewMeta = document.getElementById('snapshotPreviewMeta');
    this.btnClosePreview = document.getElementById('btnCloseSnapshotPreview');
  }

  bindEvents() {
    if (this.intervalInput) {
      this.intervalInput.addEventListener('input', () => {
        this.syncDraftIntervalUI();
      });
    }

    if (this.btnSaveInterval) {
      this.btnSaveInterval.addEventListener('click', () => {
        const val = parseFloat(this.intervalInput?.value ?? '');
        this.saveInterval(val);
      });
    }

    if (this.presetContainer) {
      this.presetContainer.querySelectorAll('.snap-preset-chip').forEach((chip) => {
        chip.addEventListener('click', () => {
          const val = parseFloat(chip.dataset.presetSec || '');
          if (!isNaN(val)) {
            if (this.intervalInput) {
              this.intervalInput.value = val.toFixed(1);
            }
            // 仅切换待保存值，不自动提交保存，需点击右上角「保存配置」才生效
            this.syncDraftIntervalUI();
          }
        });
      });
    }

    if (this.btnRefresh) {
      this.btnRefresh.addEventListener('click', async () => {
        await store.fetchSnapshots();
        window.showToast?.('📸 快照目录已刷新');
      });
    }

    if (this.btnClearSubdir) {
      this.btnClearSubdir.addEventListener('click', async () => {
        if (this.activeFolder === 'ALL') return;
        if (!window.confirm(`确定要清空子目录「${this.activeFolder}/」下的全部快照吗？`)) return;
        try {
          const res = await api.deleteSnapshot({ subdir: this.activeFolder, clear_subdir: true });
          window.showToast?.(`🧹 已清空「${this.activeFolder}/」共 ${res.deleted_count || 0} 张快照`);
          await store.fetchSnapshots();
          await store.fetchStatus();
        } catch (err) {
          window.showToast?.('❌ 清空子目录失败');
        }
      });
    }

    if (this.btnClearAll) {
      this.btnClearAll.addEventListener('click', async () => {
        if (!window.confirm('确定要清空所有子目录下的全部人脸快照吗？此操作不可恢复。')) return;
        try {
          const res = await api.deleteSnapshot({ clear_all: true });
          window.showToast?.(`🗑️ 已清空全部共 ${res.deleted_count || 0} 张快照`);
          await store.fetchSnapshots();
          await store.fetchStatus();
        } catch (err) {
          window.showToast?.('❌ 清空快照失败');
        }
      });
    }

    if (this.btnClosePreview && this.previewModal) {
      this.btnClosePreview.addEventListener('click', () => {
        this.previewModal.classList.remove('open');
      });
      this.previewModal.addEventListener('click', (e) => {
        if (e.target === this.previewModal) {
          this.previewModal.classList.remove('open');
        }
      });
    }
  }

  formatInterval(val) {
    return Number.isInteger(Number(val) * 10)
      ? Number(val).toFixed(1)
      : Number(val).toFixed(2);
  }

  syncDraftIntervalUI() {
    const draftVal = parseFloat(this.intervalInput?.value ?? '');
    const validDraft = !isNaN(draftVal);
    this.isDirtyInterval = !validDraft || Math.abs(draftVal - this.savedMinInterval) >= 0.005;

    if (this.presetContainer) {
      this.presetContainer.querySelectorAll('.snap-preset-chip').forEach((chip) => {
        const p = parseFloat(chip.dataset.presetSec || '');
        chip.classList.toggle('active', validDraft && Math.abs(p - draftVal) < 0.005);
      });
    }

    if (this.configStatusBadge) {
      if (this.isDirtyInterval) {
        this.configStatusBadge.textContent = '待保存';
        this.configStatusBadge.className = 'badge badge-warning';
      } else {
        this.configStatusBadge.textContent = '已同步';
        this.configStatusBadge.className = 'badge badge-mode';
      }
    }

    if (this.btnSaveInterval && !this.btnSaveInterval.disabled) {
      this.btnSaveInterval.textContent = this.isDirtyInterval ? '💾 保存配置 (待保存)' : '💾 保存配置';
      this.btnSaveInterval.classList.toggle('btn-save-unsaved', this.isDirtyInterval);
    }

    if (this.intervalRuleText) {
      const savedText = this.formatInterval(this.savedMinInterval);
      if (this.isDirtyInterval) {
        const draftText = validDraft ? `${this.formatInterval(draftVal)} 秒` : '待确认数值';
        this.intervalRuleText.innerHTML = `<span style="color: var(--warning);">⚠️ 已修改为 ${draftText}（尚未保存），请点击右上角「💾 保存配置」生效</span>`;
      } else {
        this.intervalRuleText.textContent = `当前最小间隔 ≥ ${savedText} 秒 · 已同步保存至 config.yaml`;
      }
    }
  }

  async saveInterval(val) {
    if (isNaN(val) || val < 0.2 || val > 3600) {
      window.showToast?.('⚠️ 请输入 0.2 ~ 3600 秒之间的有效抓拍间隔');
      return;
    }

    if (this.btnSaveInterval) {
      this.btnSaveInterval.disabled = true;
      this.btnSaveInterval.textContent = '保存中...';
    }

    try {
      const res = await api.updateSnapshotConfig({ min_interval_sec: val });
      if (res.status === 'success') {
        this.savedMinInterval = Number(res.min_interval_sec ?? val);
        this.isDirtyInterval = false;
        if (this.intervalInput) {
          this.intervalInput.value = this.formatInterval(this.savedMinInterval);
        }
        window.showToast?.(`✅ ${res.message || '抓拍间隔已保存至 config.yaml'}`);
        await store.fetchSnapshots();
        await store.fetchStatus();
      } else {
        window.showToast?.(`❌ ${res.error || '保存配置失败'}`);
      }
    } catch (err) {
      window.showToast?.('❌ 保存抓拍间隔失败，请检查服务连接');
    } finally {
      if (this.btnSaveInterval) {
        this.btnSaveInterval.disabled = false;
      }
      this.syncDraftIntervalUI();
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      if (this.btnSaveInterval) {
        this.btnSaveInterval.style.display = state.activeModule === 'snapshots' ? 'inline-flex' : 'none';
      }
      if (state.snapshotsSummary) {
        this.render(state.snapshotsSummary);
      }
    });
  }

  render(summary) {
    if (!this.gridEl) return;

    const total = summary.total ?? 0;
    const maxTotal = summary.max_total ?? 500;
    const cleanupBatch = summary.cleanup_batch ?? 100;
    const minInterval = summary.min_interval_sec ?? 1.0;
    const folders = summary.folders || [];
    const snapshots = summary.snapshots || [];

    // 若当前选中的文件夹已不存在，回退到 ALL
    if (this.activeFolder !== 'ALL' && !folders.some(f => f.name === this.activeFolder)) {
      this.activeFolder = 'ALL';
    }

    if (this.rootDirText) {
      this.rootDirText.textContent = summary.root_dir || './output/snapshots';
    }
    if (this.absRootDirText) {
      this.absRootDirText.textContent = `物理路径: ${summary.abs_root_dir || summary.root_dir || './output/snapshots'} (已在 config.yaml 配置)`;
    }
    if (this.countText) {
      this.countText.textContent = `${total} / ${maxTotal} 张`;
    }
    if (this.capacityFill) {
      const pct = maxTotal > 0 ? Math.min(100, Math.round((total / maxTotal) * 100)) : 0;
      this.capacityFill.style.width = `${pct}%`;
    }
    if (this.policyRuleText) {
      this.policyRuleText.textContent = `超过 ${maxTotal} 张自动一次性清理最早的 ${cleanupBatch} 张`;
    }
    this.savedMinInterval = Number(minInterval);
    const formattedInterval = this.formatInterval(this.savedMinInterval);

    if (this.intervalValueDisplay) {
      this.intervalValueDisplay.textContent = formattedInterval;
    }
    if (this.intervalInput && document.activeElement !== this.intervalInput && !this.isDirtyInterval) {
      this.intervalInput.value = formattedInterval;
    }
    this.syncDraftIntervalUI();

    if (this.btnClearSubdir) {
      if (this.activeFolder === 'ALL') {
        this.btnClearSubdir.style.display = 'none';
      } else {
        this.btnClearSubdir.style.display = 'inline-flex';
        this.btnClearSubdir.textContent = `🧹 清空「${this.activeFolder}」`;
      }
    }

    // 渲染子目录筛选胶囊
    if (this.folderPillsEl) {
      this.folderPillsEl.innerHTML = '';

      const allPill = document.createElement('button');
      allPill.className = `folder-pill ${this.activeFolder === 'ALL' ? 'active' : ''}`;
      allPill.innerHTML = `📂 全部目录 <span class="folder-pill-count">${total}</span>`;
      allPill.addEventListener('click', () => {
        this.activeFolder = 'ALL';
        this.render(summary);
      });
      this.folderPillsEl.appendChild(allPill);

      folders.forEach((f) => {
        const pill = document.createElement('button');
        const isActive = this.activeFolder === f.name;
        const icon = f.is_family ? '🏠' : '🚨';
        pill.className = `folder-pill ${f.is_family ? 'family' : 'stranger'} ${isActive ? 'active' : ''}`;
        pill.innerHTML = `${icon} ${f.name}/ <span class="folder-pill-count">${f.count}</span>`;
        pill.addEventListener('click', () => {
          this.activeFolder = f.name;
          this.render(summary);
        });
        this.folderPillsEl.appendChild(pill);
      });
    }

    // 筛选快照卡片
    const filtered = this.activeFolder === 'ALL'
      ? snapshots
      : snapshots.filter(item => item.subdir === this.activeFolder);

    this.gridEl.innerHTML = '';
    if (filtered.length === 0) {
      const folderHint = this.activeFolder === 'ALL' ? '所有子目录' : `子目录「${this.activeFolder}/」`;
      this.gridEl.innerHTML = `
        <div class="snapshot-empty" style="grid-column: 1 / -1;">
          <div style="font-size: 32px; margin-bottom: 10px;">📸</div>
          <div style="font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 6px;">${folderHint}暂无抓拍快照</div>
          <div style="font-size: 12px;">当摄像头开启核验并检测到人脸时（自家人或陌生人），系统将按 ≥${Number(minInterval).toFixed(1)}s 间隔自动归档至对应子文件夹。</div>
        </div>
      `;
      return;
    }

    filtered.forEach((item) => {
      const imgUrl = `/api/snapshot_file?sub=${encodeURIComponent(item.subdir)}&file=${encodeURIComponent(item.filename)}`;
      const card = document.createElement('div');
      card.className = 'snapshot-item-card';

      const badgeCls = item.is_family ? 'family' : 'stranger';
      const badgeIcon = item.is_family ? '🏠' : '🚨';

      card.innerHTML = `
        <div class="snapshot-thumb-box">
          <img src="${imgUrl}" loading="lazy" alt="${item.rel_path}" />
          <span class="snapshot-folder-badge ${badgeCls}">${badgeIcon} ${item.subdir}</span>
          <button class="snapshot-del-btn" title="删除此张快照">✕</button>
        </div>
        <div class="snapshot-item-footer">
          <div class="snapshot-time-text">🕒 ${item.time}</div>
          <div class="snapshot-path-text" title="${item.rel_path}">📁 ${item.rel_path}</div>
        </div>
      `;

      const thumbBox = card.querySelector('.snapshot-thumb-box');
      const delBtn = card.querySelector('.snapshot-del-btn');

      thumbBox.addEventListener('click', (e) => {
        if (e.target === delBtn) return;
        this.openPreview(item, imgUrl);
      });

      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await api.deleteSnapshot({ subdir: item.subdir, file: item.filename });
          window.showToast?.(`🗑️ 已删除快照 ${item.rel_path}`);
          await store.fetchSnapshots();
          await store.fetchStatus();
        } catch (err) {
          window.showToast?.('❌ 删除快照失败');
        }
      });

      this.gridEl.appendChild(card);
    });
  }

  openPreview(item, imgUrl) {
    if (!this.previewModal || !this.previewImg) return;
    this.previewImg.src = imgUrl;
    if (this.previewTitle) {
      this.previewTitle.textContent = `${item.is_family ? '🏠 自家人抓拍' : '🚨 陌生人抓拍'} · ${item.subdir}`;
    }
    if (this.previewMeta) {
      this.previewMeta.textContent = `抓拍时间: ${item.time}  |  相对路径: ${item.rel_path}`;
    }
    this.previewModal.classList.add('open');
  }
}
