import os
import sys
import yaml
import cv2
import time
import argparse
import numpy as np

from core.detector import LightweightDetector
from core.recognizer import PersonRecognizer


def run_inference(image_path: str,
                  config_path: str = "config.yaml",
                  save_output: bool = True,
                  output_dir: str = "output"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg["system"].get("device", "mps")
    weights_path = cfg["paths"]["best_model_path"]
    gallery_path = cfg["paths"]["gallery_file"]
    similarity_thresh = cfg["thresholds"]["similarity_threshold"]

    detector = LightweightDetector(target_size=(256, 128))
    recognizer = PersonRecognizer(
        weights_path=weights_path,
        gallery_path=gallery_path,
        model_type=cfg["system"].get("model_type", "LightPersonIDNet"),
        similarity_threshold=similarity_thresh,
        device=device
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 纯净端到端模型推理 (输入图片 -> 识别画面有几个人、几个自家人、具体是谁)
    summary = recognizer.predict_image(img, detector=detector)

    print(f"\n📸 图片分析: {os.path.basename(image_path)} (分辨率: {img.shape[1]}x{img.shape[0]})")
    print(f"📊【模型纯净感知输出】:")
    print(f"  • 画面总人数:   {summary.total_persons} 人")
    print(f"  • 自家人数:     {summary.family_count} 人 ({'、'.join(summary.family_members) if summary.family_members else '无'})")
    print(f"  • 陌生人数:     {summary.stranger_count} 人")
    print(f"  • 推理总耗时:   {summary.latency_ms:.2f} ms")
    print(f"  • 各主体明细:")
    for idx, p in enumerate(summary.persons):
        tag = "自家人" if p["is_family"] else "陌生人"
        print(f"    [主体 #{idx+1}] 身份: 【{p['name']}】({tag}) | 置信度: {p['score']:.4f} | 状态: {p['status']} | 坐标: {p['bbox']}")

    annotated = img.copy()

    for idx, p in enumerate(summary.persons):
        x, y, w, h = p["bbox"]["x"], p["bbox"]["y"], p["bbox"]["w"], p["bbox"]["h"]
        is_stranger = p["is_stranger"]

        color = (0, 255, 0) if not is_stranger else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        label = f"{p['name']} ({p['score']:.2f}) [{p['status']}]"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, max(0, y - lh - 10)), (x + lw + 8, y), color, -1)
        cv2.putText(annotated, label, (x + 4, max(lh + 2, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    hud_text = f"Latency: {summary.latency_ms:.1f}ms | Persons: {summary.total_persons} (Family: {summary.family_count}, Stranger: {summary.stranger_count})"
    cv2.putText(annotated, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, annotated)
        print(f"💾 标注结果图已保存至: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="入户人员检测与识别纯净推理工具 (图片/文件夹)")
    parser.add_argument("--image", type=str, help="待检测的单张图片路径")
    parser.add_argument("--dir", type=str, help="待批量检测的图片目录")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg_file = args.config
    default_cfg = os.path.join(os.path.dirname(__file__), "config.yaml")
    if cfg_file == "config.yaml" and not os.path.exists("paths"):
        # 如果从根目录运行且根目录 config.yaml 没有 paths 节点，则使用模块自带配置
        if os.path.exists(default_cfg):
            cfg_file = default_cfg
    elif not os.path.exists(cfg_file):
        cfg_file = default_cfg

    if args.image:
        run_inference(args.image, cfg_file)
    elif args.dir:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if os.path.splitext(f)[1].lower() in valid_exts]
        for f in sorted(files):
            run_inference(f, cfg_file)
    else:
        sample_img = "data/dataset/val/爸爸/爸爸_000.jpg"
        if not os.path.exists(sample_img):
            sample_img = os.path.join(os.path.dirname(__file__), "data/dataset/val/爸爸/爸爸_000.jpg")
        print(f"未指定输入图片，运行默认验证集样本测试: {sample_img}")
        run_inference(sample_img, cfg_file)
