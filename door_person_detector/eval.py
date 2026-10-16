import os
import sys
import yaml
import cv2
import numpy as np
from typing import Dict, List, Tuple

from core.detector import LightweightDetector
from core.recognizer import PersonRecognizer


def evaluate_model(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights_path = cfg["paths"]["best_model_path"]
    gallery_path = cfg["paths"]["gallery_file"]
    val_dir = cfg["paths"]["val_dir"]
    strangers_dir = cfg["paths"]["strangers_dir"]
    target_thresh = cfg["thresholds"]["similarity_threshold"]

    print("\n=======================================================")
    print("🔍 启动入户人员识别模型性能与安全门限评估 (纯数据集)")
    print(f"权重路径: {weights_path}")
    print(f"底库路径: {gallery_path}")
    print(f"配置识别阈值: {target_thresh}")
    print("=======================================================\n")

    recognizer = PersonRecognizer(
        weights_path=weights_path,
        gallery_path=gallery_path,
        model_type=cfg["system"].get("model_type", "LightPersonIDNet"),
        similarity_threshold=target_thresh,
        device=cfg["system"].get("device", "mps")
    )

    detector = LightweightDetector(target_size=(256, 128))
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    # 1. 评估自家人验证集 (TAR: 真正率)
    family_correct = 0
    family_total = 0
    family_scores: List[float] = []

    if os.path.exists(val_dir):
        for member_name in sorted(os.listdir(val_dir)):
            member_path = os.path.join(val_dir, member_name)
            if not os.path.isdir(member_path) or member_name.startswith('.'):
                continue
            for fname in os.listdir(member_path):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    img_path = os.path.join(member_path, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    p = detector.process_direct_subject(img)
                    res = recognizer.identify(p.aligned_patch, p.bbox)
                    family_total += 1
                    family_scores.append(res.similarity_score)
                    if res.person_name == member_name:
                        family_correct += 1

    tar = (family_correct / max(1, family_total)) * 100.0
    avg_family_score = np.mean(family_scores) if family_scores else 0.0

    # 2. 评估陌生人测试集 (FAR: 误识率 / 陌生人误当家人率)
    stranger_total = 0
    stranger_false_alarms = 0
    stranger_scores: List[float] = []

    if os.path.exists(strangers_dir):
        for fname in os.listdir(strangers_dir):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                img_path = os.path.join(strangers_dir, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                p = detector.process_direct_subject(img)
                res = recognizer.identify(p.aligned_patch, p.bbox)
                stranger_total += 1
                stranger_scores.append(res.similarity_score)
                if not res.is_stranger:
                    stranger_false_alarms += 1

    far = (stranger_false_alarms / max(1, stranger_total)) * 100.0 if stranger_total > 0 else 0.0
    max_stranger_score = max(stranger_scores) if stranger_scores else 0.0

    print("📊 评估报告指标:")
    print(f"-------------------------------------------------------")
    print(f"• 自家人验证样本总数: {family_total}")
    print(f"• 自家人正确识别率 (TAR): {tar:.1f}% ({family_correct}/{family_total})")
    print(f"• 自家人平均相似度得分: {avg_family_score:.4f}")
    print(f"-------------------------------------------------------")
    print(f"• 陌生人测试样本总数: {stranger_total}")
    print(f"• 陌生人误判率 (FAR): {far:.1f}% ({stranger_false_alarms}/{stranger_total})")
    print(f"• 陌生人最高相似度得分: {max_stranger_score:.4f}")
    print(f"-------------------------------------------------------")

    # 3. 安全建议门限分析
    print("\n💡 安全阈值设定诊断:")
    if max_stranger_score < target_thresh:
        margin = target_thresh - max_stranger_score
        print(f"✅ 当前阈值 {target_thresh} 非常安全！距陌生人最高相似度具备 {margin:.4f} 的安全裕度。")
    else:
        suggested = max_stranger_score + 0.05
        print(f"⚠️ 警告: 陌生人最高得分 ({max_stranger_score:.4f}) 接近或超过了识别阈值，建议将 threshold 上调至: {suggested:.2f}")


if __name__ == "__main__":
    evaluate_model()
