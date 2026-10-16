import os
import sys
import cv2
import time
import argparse
from typing import List

# 添加父目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.detector import LightweightDetector


def collect_from_camera(person_name: str,
                        camera_index: int = 0,
                        output_base: str = "./data/dataset",
                        target_count: int = 50):
    """
    通过摄像头交互式采集家庭成员入户照片 (含正脸、侧脸、低头、侧身等全姿态)
    """
    train_dir = os.path.join(output_base, "train", person_name)
    val_dir = os.path.join(output_base, "val", person_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头: {camera_index}")
        return

    detector = LightweightDetector(target_size=(256, 128))

    instructions = [
        "1/5: 请直视摄像头 (保持自然正脸)",
        "2/5: 请稍微向左转头 30~45度 (采集左侧脸/侧身)",
        "3/5: 请稍微向右转头 30~45度 (采集右侧脸/侧身)",
        "4/5: 请低头模拟寻找钥匙/看手机 (采集低头/无脸姿态)",
        "5/5: 请后退一步或侧立 (采集完整体态轮廓)"
    ]

    print(f"\n=======================================================")
    print(f"📸 开始为家庭成员【{person_name}】采集入户识别数据")
    print(f"目标采集数量: {target_count} 张 (80% 训练集, 20% 验证集)")
    print("提示: 按【空格键】自动连拍，按【C】单张抓拍，按【Q】退出")
    print(f"=======================================================\n")

    saved_count = 0
    auto_capturing = False
    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = detector.detect(frame)
        stage_idx = min(len(instructions) - 1, saved_count // max(1, target_count // len(instructions)))
        current_hint = instructions[stage_idx]

        # 绘制预览辅助信息
        display_frame = frame.copy()
        for p in persons:
            x, y, w, h = p.bbox
            color = (0, 255, 0) if p.has_face else (255, 165, 0)
            status_text = "正脸" if (p.has_face and not p.is_profile_or_occluded) else ("侧脸" if p.is_profile_or_occluded else "主体/无脸")
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, f"{person_name}: {status_text}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 顶部提示栏
        cv2.putText(display_frame, f"[{saved_count}/{target_count}] {current_hint}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"采集入户数据 - {person_name} (按Q退出)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            auto_capturing = not auto_capturing
            print(f"连拍模式: {'开启' if auto_capturing else '暂停'}")

        should_save = False
        if key == ord('c'):
            should_save = True
        elif auto_capturing and (time.time() - last_capture_time > 0.35):
            should_save = True

        if should_save and len(persons) > 0:
            target_person = persons[0]
            # 80% 训练集，20% 验证集
            save_folder = val_dir if (saved_count % 5 == 0) else train_dir
            save_path = os.path.join(save_folder, f"{person_name}_{int(time.time()*1000)%1000000}_{saved_count:03d}.jpg")
            cv2.imwrite(save_path, target_person.aligned_patch)
            saved_count += 1
            last_capture_time = time.time()
            print(f"已保存切片 [{saved_count}/{target_count}]: {save_path}")

            if saved_count >= target_count:
                print(f"🎉 成功完成【{person_name}】的全部数据采集！")
                break

    cap.release()
    cv2.destroyAllWindows()


def collect_from_folder(person_name: str,
                        input_dir: str,
                        output_base: str = "./data/dataset"):
    """
    将用户已有的生活照/监控截图文件夹自动切片并存入训练集/验证集
    """
    train_dir = os.path.join(output_base, "train", person_name)
    val_dir = os.path.join(output_base, "val", person_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    detector = LightweightDetector(target_size=(256, 128))
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    print(f"📁 正在从目录导入【{person_name}】的照片，共找到 {len(files)} 张图片...")
    saved = 0
    for idx, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        persons = detector.detect(img)
        if len(persons) == 0:
            continue
        p = persons[0]
        dest_folder = val_dir if (saved % 5 == 0) else train_dir
        dest_path = os.path.join(dest_folder, f"{person_name}_{saved:03d}.jpg")
        cv2.imwrite(dest_path, p.aligned_patch)
        saved += 1

    print(f"✅ 完成导入！成功生成 {saved} 张高质量 256x128 人体特征切片")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="入户人员训练数据采集与切片工具")
    parser.add_argument("--name", type=str, required=True, help="要采集的家庭成员姓名 (如: 爸爸, 妈妈)")
    parser.add_argument("--camera", type=int, default=None, help="摄像头索引 (默认0)")
    parser.add_argument("--input_dir", type=str, default=None, help="已有照片文件夹路径 (可选)")
    parser.add_argument("--count", type=int, default=40, help="目标采集数量 (默认40张)")
    args = parser.parse_args()

    if args.input_dir:
        collect_from_folder(args.name, args.input_dir)
    else:
        cam_idx = 0 if args.camera is None else args.camera
        collect_from_camera(args.name, cam_idx, target_count=args.count)
