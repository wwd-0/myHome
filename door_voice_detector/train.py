"""
方案 A 两级超轻量语音模型训练与 ONNX 一键导出主程序 (train.py)
- 自动启用 Apple MPS / NVIDIA CUDA 硬件加速
- Stage 1: 训练 BCResMDTCCommandNet (10 种口令识别 + UNKNOWN 闲聊/噪音拒识 + 保序口令原型导出)
- Stage 2: 训练 CAMPlusLiteSpeakerNet + AAM-Softmax (超球面声纹特征嵌入，开集区分不同说话人)
- 训练完成后自动导出至 ./audio/models/ 供 C++ 服务零依赖原生加载！
"""

import os
import json
import glob
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from core.fbank import load_wav_16k, vad_trim_and_center, compute_fbank_80x160
from models import BCResMDTCCommandNet, CAMPlusLiteSpeakerNet, AAMSoftmaxHead


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spec_augment(fbank: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """轻量 SpecAugment 时频掩码增强，防止过拟合，强化抗噪鲁棒性"""
    out = fbank.copy()  # (1, 80, 160)
    if rng.random() < 0.65:
        f_w = int(rng.integers(2, 9))
        f0 = int(rng.integers(0, 80 - f_w))
        out[:, f0 : f0 + f_w, :] = 0.0
    if rng.random() < 0.65:
        t_w = int(rng.integers(4, 18))
        t0 = int(rng.integers(0, 160 - t_w))
        out[:, :, t0 : t0 + t_w] = 0.0
    if rng.random() < 0.5:
        out = out + rng.normal(0.0, 0.08, size=out.shape).astype(np.float32)
    return out


class CommandDataset(Dataset):
    def __init__(self, samples, augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        if self.augment:
            feat = spec_augment(feat, self.rng)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


class SpeakerDataset(Dataset):
    def __init__(self, samples, augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, spk_label = self.samples[idx]
        if self.augment:
            feat = spec_augment(feat, self.rng)
        return torch.from_numpy(feat), torch.tensor(spk_label, dtype=torch.long)


def extract_file_fbank(wav_path: str, vad_thresh: float) -> np.ndarray:
    audio = load_wav_16k(wav_path)
    aligned, _, _ = vad_trim_and_center(audio, vad_rms_threshold=vad_thresh)
    if aligned is None:
        canvas = np.zeros(25600, dtype=np.float32)
        n = min(len(audio), 25600)
        canvas[:n] = audio[:n]
        aligned = canvas
    return compute_fbank_80x160(aligned)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_all(cfg_path: str):
    set_seed(42)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"🚀 使用训练计算设备: {device}")

    commands = cfg["commands"]["list"]
    num_commands = len(commands)
    vad_thresh = float(cfg["audio"].get("vad_rms_threshold", 0.012))

    ds_dir = cfg["paths"]["dataset_dir"]
    weights_dir = cfg["paths"]["weights_dir"]
    export_dir = cfg["paths"]["export_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # ==================== 1. 加载并预计算 Fbank ====================
    print("📂 正在加载训练音频并提取 80x160 Log-Mel Fbank...")
    cmd_samples = []
    # 类别 0: UNKNOWN 闲聊与环境噪声
    for wav_p in sorted(glob.glob(os.path.join(ds_dir, "negatives", "*.wav"))):
        cmd_samples.append((extract_file_fbank(wav_p, vad_thresh * 0.5), 0))

    # 类别 1..K: 10 种开门语音口令
    for c_idx, cmd in enumerate(commands):
        pattern1 = os.path.join(ds_dir, "commands", f"{c_idx + 1:02d}_{cmd}", "*.wav")
        pattern2 = os.path.join(ds_dir, "commands", cmd, "*.wav")
        files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
        if not files:
            raise RuntimeError(f"未找到口令 [{cmd}] 的训练样本，请先运行 tools/generate_seed_data.py")
        for wav_p in files:
            cmd_samples.append((extract_file_fbank(wav_p, vad_thresh), c_idx + 1))

    random.shuffle(cmd_samples)
    split1 = int(len(cmd_samples) * 0.85)
    cmd_train_ds = CommandDataset(cmd_samples[:split1], augment=True)
    cmd_val_ds = CommandDataset(cmd_samples[split1:], augment=False)

    # ==================== 2. 训练 Stage-1 口令网 ====================
    s1_cfg = cfg["stage1_command_net"]
    cmd_net = BCResMDTCCommandNet(
        num_commands=num_commands,
        embed_dim=int(s1_cfg["embed_dim"]),
        base_channels=int(s1_cfg["base_channels"]),
        mdtc_channels=int(s1_cfg["mdtc_channels"]),
    ).to(device)

    opt1 = torch.optim.AdamW(cmd_net.parameters(), lr=float(s1_cfg["lr"]), weight_decay=1e-4)
    epochs1 = int(s1_cfg["epochs"])
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs1, eta_min=1e-5)
    criterion1 = nn.CrossEntropyLoss(label_smoothing=0.05)

    train_loader1 = DataLoader(cmd_train_ds, batch_size=int(s1_cfg["batch_size"]), shuffle=True)
    val_loader1 = DataLoader(cmd_val_ds, batch_size=64, shuffle=False)

    best_acc1 = 0.0
    s1_ckpt = os.path.join(weights_dir, "best_command_net.pt")
    print(f"\n🔥 [Stage 1] 开始训练 BCResMDTCCommandNet (总样本={len(cmd_samples)}, 口令数={num_commands} + UNKNOWN拒识类)...")
    for ep in range(1, epochs1 + 1):
        cmd_net.train()
        for xb, yb in train_loader1:
            xb, yb = xb.to(device), yb.to(device)
            opt1.zero_grad()
            out = cmd_net(xb)
            logits = out[:, : 1 + num_commands]
            loss = criterion1(logits, yb)
            loss.backward()
            opt1.step()
            cmd_net.normalize_classifier_weights()
        sched1.step()

        cmd_net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader1:
                xb, yb = xb.to(device), yb.to(device)
                logits = cmd_net(xb)[:, : 1 + num_commands]
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += yb.size(0)
        val_acc = 100.0 * correct / max(1, total)
        if val_acc >= best_acc1:
            best_acc1 = val_acc
            torch.save(cmd_net.state_dict(), s1_ckpt)
        if ep % 10 == 0 or ep == epochs1:
            print(f"   Epoch [{ep:02d}/{epochs1:02d}] Val Acc (含闲聊拒识): {val_acc:.2f}% (Best: {best_acc1:.2f}%)")

    cmd_net.load_state_dict(torch.load(s1_ckpt, map_location="cpu"))
    cmd_net.cpu().eval()

    # 计算每种口令的标准声学保序原型向量，一并写入 commands.json
    prototypes = {}
    with torch.no_grad():
        for c_idx, cmd in enumerate(commands):
            feats = [torch.from_numpy(s[0]) for s in cmd_samples if s[1] == c_idx + 1]
            batch = torch.stack(feats, dim=0)
            embs = cmd_net(batch)[:, 1 + num_commands :]
            proto = F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=1)[0].tolist()
            prototypes[cmd] = [round(float(v), 6) for v in proto]

    # ==================== 3. 训练 Stage-2 声纹网 ====================
    spk_dirs = sorted(
        d for d in glob.glob(os.path.join(ds_dir, "speakers", "*")) if os.path.isdir(d)
    )
    spk_samples = []
    for spk_idx, s_dir in enumerate(spk_dirs):
        for wav_p in sorted(glob.glob(os.path.join(s_dir, "*.wav"))):
            spk_samples.append((extract_file_fbank(wav_p, vad_thresh), spk_idx))

    random.shuffle(spk_samples)
    split2 = int(len(spk_samples) * 0.85)
    spk_train_ds = SpeakerDataset(spk_samples[:split2], augment=True)
    spk_val_ds = SpeakerDataset(spk_samples[split2:], augment=False)

    s2_cfg = cfg["stage2_speaker_net"]
    spk_net = CAMPlusLiteSpeakerNet(
        embed_dim=int(s2_cfg["embed_dim"]),
        stem_channels=int(s2_cfg["stem_channels"]),
        init_channels=int(s2_cfg["init_channels"]),
        growth_rate=int(s2_cfg["growth_rate"]),
        num_cam_layers=int(s2_cfg["num_cam_layers"]),
    ).to(device)
    aam_head = AAMSoftmaxHead(
        in_features=int(s2_cfg["embed_dim"]),
        num_speakers=len(spk_dirs),
        s=float(s2_cfg["aam_scale"]),
        m=float(s2_cfg["aam_margin"]),
    ).to(device)

    opt2 = torch.optim.AdamW(
        list(spk_net.parameters()) + list(aam_head.parameters()),
        lr=float(s2_cfg["lr"]),
        weight_decay=1e-4,
    )
    epochs2 = int(s2_cfg["epochs"])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs2, eta_min=1e-5)
    criterion2 = nn.CrossEntropyLoss()

    train_loader2 = DataLoader(spk_train_ds, batch_size=int(s2_cfg["batch_size"]), shuffle=True)
    val_loader2 = DataLoader(spk_val_ds, batch_size=64, shuffle=False)

    best_acc2 = 0.0
    s2_ckpt = os.path.join(weights_dir, "best_speaker_net.pt")
    print(f"\n🔥 [Stage 2] 开始训练 CAMPlusLiteSpeakerNet (说话人数={len(spk_dirs)}, 样本数={len(spk_samples)})...")
    for ep in range(1, epochs2 + 1):
        spk_net.train()
        aam_head.train()
        for xb, yb in train_loader2:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            emb = spk_net(xb)
            logits = aam_head(emb, yb)
            loss = criterion2(logits, yb)
            loss.backward()
            opt2.step()
        sched2.step()

        spk_net.eval()
        aam_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            norm_w = F.normalize(aam_head.weight, p=2, dim=1)
            for xb, yb in val_loader2:
                xb, yb = xb.to(device), yb.to(device)
                emb = spk_net(xb)
                preds = F.linear(emb, norm_w).argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += yb.size(0)
        val_acc = 100.0 * correct / max(1, total)
        if val_acc >= best_acc2:
            best_acc2 = val_acc
            torch.save(spk_net.state_dict(), s2_ckpt)
        if ep % 10 == 0 or ep == epochs2:
            print(f"   Epoch [{ep:02d}/{epochs2:02d}] Speaker Val Acc: {val_acc:.2f}% (Best: {best_acc2:.2f}%)")

    spk_net.load_state_dict(torch.load(s2_ckpt, map_location="cpu"))
    spk_net.cpu().eval()

    # ==================== 4. 导出生产级 ONNX 模型与元数据 ====================
    dummy = torch.randn(1, 1, 80, 160)
    kws_onnx = os.path.join(export_dir, cfg["paths"]["kws_onnx_name"])
    sv_onnx = os.path.join(export_dir, cfg["paths"]["sv_onnx_name"])
    cmd_json = os.path.join(export_dir, cfg["paths"]["commands_json_name"])

    torch.onnx.export(
        cmd_net,
        dummy,
        kws_onnx,
        opset_version=12,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["output"],
        dynamo=False,
    )
    torch.onnx.export(
        spk_net,
        dummy,
        sv_onnx,
        opset_version=12,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["spk_embedding"],
        dynamo=False,
    )

    meta = {
        "commands": commands,
        "embed_dim": int(s1_cfg["embed_dim"]),
        "speaker_embed_dim": int(s2_cfg["embed_dim"]),
        "prototypes": prototypes,
    }
    with open(cmd_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n======================================================================")
    print(f"✅ [导出成功] Stage-1 口令网: {kws_onnx} ({os.path.getsize(kws_onnx)/1024/1024:.2f} MB)")
    print(f"✅ [导出成功] Stage-2 声纹网: {sv_onnx} ({os.path.getsize(sv_onnx)/1024/1024:.2f} MB)")
    print(f"✅ [导出成功] 口令原型配置:   {cmd_json}")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练并导出方案 A 两级轻量语音模型")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    )
    args = parser.parse_args()
    train_all(args.config)
