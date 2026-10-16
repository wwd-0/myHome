import os
import sys
import yaml
import json
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from models.light_person_net import LightPersonIDNet, MobileFaceNet
from models.arcface_head import ArcMarginProduct


class FamilyPersonDataset(Dataset):
    """自定义入户人员数据集读取器 (支持 jpg, png, jpeg, webp)"""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        if not os.path.exists(root_dir):
            return

        # 扫描非空家庭成员子文件夹
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        entries = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])
        non_empty_entries = []
        for d in entries:
            cls_dir = os.path.join(root_dir, d)
            valid_files = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in valid_exts]
            if len(valid_files) > 0:
                non_empty_entries.append(d)

        if len(non_empty_entries) > 15:
            print(f"⚠️ 警告: 检测到 {len(non_empty_entries)} 位人员，系统上限为 15 人，将截取前 15 位成员！")
            non_empty_entries = non_empty_entries[:15]

        self.classes = non_empty_entries
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    fpath = os.path.join(cls_dir, fname)
                    self.samples.append((fpath, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size: Tuple[int, int]):
    """纯净基础数据预处理 (验证阶段不加随机噪声/遮挡，专注验证模型对训练集的拟合能力)"""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return train_transform, val_transform


def export_gallery(backbone: nn.Module,
                   dataset: FamilyPersonDataset,
                   device: torch.device,
                   output_path: str):
    """根据训练完毕的模型为每个家庭成员计算均值特征底库并持久化保存"""
    backbone.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    features_by_class: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(dataset.classes))}

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            feats = backbone(imgs, normalize=True)
            for f, l in zip(feats, labels):
                features_by_class[l.item()].append(f.cpu())

    gallery = {}
    for idx, cls_name in enumerate(dataset.classes):
        feat_list = features_by_class[idx]
        if feat_list:
            stacked = torch.stack(feat_list, dim=0)
            mean_f = torch.mean(stacked, dim=0)
            mean_f = mean_f / torch.norm(mean_f)
            gallery[cls_name] = mean_f.numpy().tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gallery, f, ensure_ascii=False, indent=2)
    print(f"✅ 家人特征底库已自动生成并持久化至: {output_path}")


def train(config_path="config.yaml"):
    # 1. 载入配置
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = cfg["system"].get("device", "mps")
    if device_str == "mps" and not torch.backends.mps.is_available():
        device_str = "cpu"
    elif device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"🚀 启动训练流程，使用加速硬件: {device}")

    img_size = tuple(cfg["system"].get("image_size", [256, 128]))
    model_type = cfg["system"].get("model_type", "LightPersonIDNet")
    embedding_dim = cfg["system"].get("embedding_dim", 512)

    train_dir = cfg["paths"]["train_dir"]
    val_dir = cfg["paths"]["val_dir"]
    weights_dir = cfg["paths"]["weights_dir"]
    best_model_path = cfg["paths"]["best_model_path"]
    gallery_file = cfg["paths"]["gallery_file"]
    os.makedirs(weights_dir, exist_ok=True)

    # 2. 构建数据加载器
    train_transform, val_transform = get_transforms(img_size)
    train_dataset = FamilyPersonDataset(train_dir, transform=train_transform)
    val_dataset = FamilyPersonDataset(val_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)
    if num_classes == 0:
        print(f"❌ 错误: 训练集目录 {train_dir} 中未找到任何人员文件夹！请先放置照片。")
        return

    print(f"📊 已检测到 {num_classes} 位家庭成员: {train_dataset.classes}")
    print(f"   - 训练样本数: {len(train_dataset)}")
    print(f"   - 验证样本数: {len(val_dataset)}")

    batch_size = cfg["train"].get("batch_size", 3)
    actual_batch_size = max(2, min(batch_size, len(train_dataset)))
    # 避免 BatchNorm1d 在训练阶段遇到 batch_size == 1 报错
    drop_last = (len(train_dataset) > actual_batch_size and len(train_dataset) % actual_batch_size == 1)
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)

    # 3. 构造骨干网络与 ArcFace 分类头
    if model_type == "MobileFaceNet":
        backbone = MobileFaceNet(embedding_dim=embedding_dim).to(device)
    else:
        backbone = LightPersonIDNet(embedding_dim=embedding_dim, pretrained=True).to(device)

    arcface_s = cfg["train"].get("arcface_s", 32.0)
    arcface_m = cfg["train"].get("arcface_m", 0.50)
    head = ArcMarginProduct(in_features=embedding_dim, out_features=num_classes, s=arcface_s, m=arcface_m).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    lr = float(cfg["train"].get("lr", 0.001))
    weight_decay = float(cfg["train"].get("weight_decay", 0.0005))
    epochs = int(cfg["train"].get("epochs", 40))

    optimizer = optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    print("\n🏁 开始模型微调训练...")

    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = backbone(imgs, normalize=True)
            outputs = head(embeddings, labels)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

        scheduler.step()
        train_acc = (total_correct / max(1, total_samples)) * 100.0
        avg_loss = total_loss / max(1, total_samples)

        # 验证轮次
        backbone.eval()
        head.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                embeddings = backbone(imgs, normalize=True)
                # 推理时原生比对原型角度
                raw_cosines = head(embeddings)
                preds = torch.argmax(raw_cosines, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += imgs.size(0)

        val_acc = (val_correct / max(1, val_samples)) * 100.0 if val_samples > 0 else train_acc

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "val_acc": val_acc
            }, best_model_path)

    print(f"\n🎉 训练完成！最优模型已保存至: {best_model_path} (验证集准确率: {best_val_acc:.1f}%)")

    # 4. 自动持久化生成家人底库
    export_gallery(backbone, train_dataset, device, gallery_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练入户人员轻量级识别模型")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config_file = args.config
    if not os.path.exists(config_file):
        config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    train(config_file)
