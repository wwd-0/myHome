import os
import sys
import cv2
import argparse
from typing import List

# 添加父目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import LightweightDetector


def import_photos_from_folder(person_name: str,
                              input_dir: str,
                              output_base: str = "./data/dataset",
                              val_ratio: float = 0.2):
    """
    从本地已有照片/图片目录批量导入家庭成员图像
    自动检测人物主体，规范化裁剪为 (256, 128) 标准切片，并按比例划分为训练集与验证集
    """
    train_dir = os.path.join(output_base, "train", person_name)
    val_dir = os.path.join(output_base, "val", person_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return

    detector = LightweightDetector(target_size=(256, 128))
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts])

    print(f"\n=======================================================")
    print(f"📁 导入照片: 成员【{person_name}】")
    print(f"来源目录: {input_dir} (共找到 {len(files)} 张图片)")
    print(f"目标目录: {output_base}")
    print(f"=======================================================\n")

    saved_count = 0
    split_interval = int(1.0 / val_ratio) if val_ratio > 0 else 999999

    for idx, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 提取主体
        persons = detector.detect(img)
        if len(persons) == 0:
            print(f"  [跳过] 未检测到明确主体: {fname}")
            continue

        p = persons[0]
        # 按比例划分为训练集和验证集
        dest_folder = val_dir if (saved_count % split_interval == 0) else train_dir
        dest_filename = f"{person_name}_{saved_count:04d}.jpg"
        dest_path = os.path.join(dest_folder, dest_filename)

        cv2.imwrite(dest_path, p.aligned_patch)
        saved_count += 1
        tag = "验证集" if (saved_count % split_interval == 0) else "训练集"
        print(f"  [{saved_count}/{len(files)}] 保存至 {tag}: {dest_filename}")

    print(f"\n🎉 导入完成！为【{person_name}】成功处理并生成 {saved_count} 张 (256x128) 高质量切片。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地照片文件夹导入并预处理家庭成员数据")
    parser.add_argument("--name", type=str, required=True, help="家庭成员姓名 (如: 爸爸, 妈妈)")
    parser.add_argument("--input_dir", type=str, required=True, help="存放该成员生活照/监控截图的本地目录")
    parser.add_argument("--output_base", type=str, default="./data/dataset", help="数据集输出根目录")
    args = parser.parse_args()

    import_photos_from_folder(args.name, args.input_dir, args.output_base)
