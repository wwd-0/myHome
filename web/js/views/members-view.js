/**
 * ==============================================================================
 * MyHome AI - 家庭成员数据中心视图控制器 (members-view.js)
 * 职责：负责成员固化特征卡片渲染、新增训练集样本上传与模型重载
 * ==============================================================================
 */

import { api } from '../api.js';
import { store } from '../store.js';

export class MembersView {
  constructor() {
    this.gridEl = document.getElementById('membersGrid');
    this.quotaBadgeEl = document.getElementById('quotaBadge');
    this.modalEl = document.getElementById('uploadModal');
    this.modalNameInput = document.getElementById('modalMemberName');
    this.modalPhotoInput = document.getElementById('modalPhotoInput');
    this.modalPreviewBox = document.getElementById('modalPreviewBox');
    this.modalPreviewImg = document.getElementById('modalPreviewImg');
    this.modalSelectedBase64 = null;

    this.bindEvents();
    this.subscribeStore();
  }

  bindEvents() {
    const btnReload = document.getElementById('btnReloadDb');
    if (btnReload) {
      btnReload.addEventListener('click', () => this.handleReloadDb());
    }

    const btnOpenUpload = document.getElementById('btnOpenUploadModal');
    if (btnOpenUpload) {
      btnOpenUpload.addEventListener('click', () => this.openUploadModal());
    }

    const btnCloseModal = document.getElementById('btnCloseUploadModal');
    if (btnCloseModal) {
      btnCloseModal.addEventListener('click', () => this.closeUploadModal());
    }

    const btnCancelModal = document.getElementById('btnCancelUploadModal');
    if (btnCancelModal) {
      btnCancelModal.addEventListener('click', () => this.closeUploadModal());
    }

    const btnSubmit = document.getElementById('btnSubmitUploadPhoto');
    if (btnSubmit) {
      btnSubmit.addEventListener('click', () => this.submitUploadPhoto());
    }

    if (this.modalPhotoInput) {
      this.modalPhotoInput.addEventListener('change', (e) => this.handlePhotoPreview(e));
    }
  }

  subscribeStore() {
    store.subscribe((state) => {
      if (state.members && state.members.length > 0) {
        this.render(state.members);
      }
    });
  }

  render(members) {
    if (!this.gridEl) return;
    this.gridEl.innerHTML = '';
    if (this.quotaBadgeEl) {
      this.quotaBadgeEl.innerText = `模型已固化 ${members.length} 位成员`;
    }

    if (!members || members.length === 0) {
      this.gridEl.innerHTML = '<div style="color: var(--muted); grid-column: 1/-1; text-align: center; padding: 40px;">暂无成员数据</div>';
      return;
    }

    members.forEach(m => {
      const card = document.createElement('div');
      card.className = 'member-card';
      const sampleCount = m.sample_count || 0;

      card.innerHTML = `
        <div class="member-card-header">
          <div class="member-name-tag">
            <span>👤</span>
            <span>${m.name}</span>
          </div>
          <span class="badge badge-family">已录入认证特征</span>
        </div>
        <div style="font-size: 13px; color: var(--text); margin-bottom: 8px;">
          <strong>认证状态:</strong> <span style="color: var(--accent);">就绪 (支持换装、逆光及多姿态快速识别)</span>
        </div>
        <div style="font-size: 12px; color: var(--muted); margin-bottom: 6px;">
          <strong>照片样本:</strong> 已学习 ${sampleCount} 张生活照样本
        </div>
        <div style="font-size: 12px; color: var(--muted); margin-bottom: 14px;">
          <strong>安全隐私:</strong> 仅在本地边缘提取特征，保护家庭隐私
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-outline btn-add-member-sample" style="padding: 6px 12px; font-size: 12px; flex: 1;">➕ 补充成员生活照</button>
        </div>
      `;

      card.querySelector('.btn-add-member-sample')?.addEventListener('click', () => {
        this.openUploadModal(m.name);
      });

      this.gridEl.appendChild(card);
    });
  }

  openUploadModal(name = '') {
    if (this.modalNameInput) this.modalNameInput.value = name;
    if (this.modalPhotoInput) this.modalPhotoInput.value = '';
    if (this.modalPreviewBox) this.modalPreviewBox.style.display = 'none';
    this.modalSelectedBase64 = null;
    this.modalEl?.classList.add('open');
  }

  closeUploadModal() {
    this.modalEl?.classList.remove('open');
  }

  handlePhotoPreview(e) {
    if (e.target.files.length > 0) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        this.modalSelectedBase64 = evt.target.result;
        if (this.modalPreviewImg) this.modalPreviewImg.src = this.modalSelectedBase64;
        if (this.modalPreviewBox) this.modalPreviewBox.style.display = 'block';
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  }

  async submitUploadPhoto() {
    const name = this.modalNameInput?.value.trim();
    if (!name) {
      alert("请输入成员姓名！");
      return;
    }
    if (!this.modalSelectedBase64) {
      alert("请选择照片文件！");
      return;
    }

    window.showToast?.("正在上传照片并重建特征原型...");
    try {
      const res = await api.uploadPhoto(name, this.modalSelectedBase64);
      if (res.status === 'ok') {
        window.showToast?.(`✅ 成功为【${name}】添加照片！底库已更新`);
        this.closeUploadModal();
        store.fetchMembers();
      } else {
        window.showToast?.("上传失败: " + (res.error || '未知错误'));
      }
    } catch (err) {
      window.showToast?.("上传异常: " + err.message);
    }
  }

  async handleReloadDb() {
    window.showToast?.("正在重新载入最新模型权重与分类头...");
    try {
      await api.reloadDb();
      window.showToast?.("✅ 模型权重已重新载入！");
      store.fetchMembers();
    } catch (err) {
      window.showToast?.("重载失败: " + err.message);
    }
  }
}
