import os
import cv2
import numpy as np


def create_sample_dataset(base_dir="./data/dataset"):
    """
    初始化数据集结构并生成初始样本占位图，确保初次安装后可以直接一键跑通 train.py
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    stranger_dir = os.path.join(base_dir, "strangers")
    
    sample_members = ["爸爸", "妈妈", "孩子"]
    
    for split_dir, count in [(train_dir, 12), (val_dir, 4)]:
        for member in sample_members:
            member_dir = os.path.join(split_dir, member)
            os.makedirs(member_dir, exist_ok=True)
            
            # 生成带不同色彩/光照与形态的 256x128 初始样本
            np.random.seed(abs(hash(member)) % 10000)
            base_color = np.random.randint(50, 200, size=3)
            
            for i in range(count):
                img_path = os.path.join(member_dir, f"{member}_{i:03d}.jpg")
                if not os.path.exists(img_path):
                    canvas = np.zeros((256, 128, 3), dtype=np.uint8)
                    # 模拟背景
                    canvas[:] = (base_color * 0.4).astype(np.uint8)
                    # 模拟人体躯干 (下半部椭圆/矩形)
                    body_color = np.clip(base_color + np.random.randint(-20, 20, size=3), 0, 255).astype(np.uint8)
                    cv2.rectangle(canvas, (24, 70), (104, 240), body_color.tolist(), -1)
                    # 模拟头部/侧脸 (上半部椭圆)
                    head_color = np.clip(base_color + [40, 20, 10], 0, 255).astype(np.uint8)
                    cv2.ellipse(canvas, (64, 45), (25, 30), 0, 0, 360, head_color.tolist(), -1)
                    # 添加微量高斯噪声模拟入户光影
                    noise = np.random.normal(0, 10, canvas.shape).astype(np.int16)
                    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    cv2.imwrite(img_path, canvas)
                    
    # 生成陌生人测试集
    os.makedirs(stranger_dir, exist_ok=True)
    for i in range(5):
        s_path = os.path.join(stranger_dir, f"stranger_{i:03d}.jpg")
        if not os.path.exists(s_path):
            s_canvas = np.random.randint(20, 220, size=(256, 128, 3), dtype=np.uint8)
            cv2.imwrite(s_path, s_canvas)

    print(f"✅ 成功初始化数据集目录结构与示例模板至: {base_dir}")
    print(f"   - 训练集: {train_dir}")
    print(f"   - 验证集: {val_dir}")
    print(f"   - 陌生人测试集: {stranger_dir}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(current_dir, "../data/dataset")
    create_sample_dataset(target_data_dir)
