import os
import subprocess
import sys
import time
import cv2
import numpy as np
from vision.face_engine import FaceEngine


def main():
    print("=" * 68)
    print("🧪 [人脸识别独立测试] OpenCV 原生加载开源 ONNX 模型权重 (零 pip 额外依赖)")
    print("=" * 68)

    face_cfg = {
        "db_dir": "./vision/db/faces",
        "models_dir": "./vision/models",
        "similarity_threshold": 0.38,
        "min_face_width_px": 90,
    }
    engine = FaceEngine(face_cfg)

    if not engine.family_db:
        print("❌ 底库未加载到有效人脸，请检查 ./vision/models/ 权重或 ./vision/db/faces/ 照片。")
        return

    # 1. 对已保存的底库图片（如 爸爸_1.jpg）执行一次自检识别并生成可视化标注图
    sample_imgs = [
        f for f in sorted(os.listdir("./vision/db/faces")) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if sample_imgs:
        test_img_path = os.path.join("./vision/db/faces", sample_imgs[0])
        img_array = np.fromfile(test_img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        annotated, matched_family, _ = engine.analyze_frame(img.copy(), tuya_distance_m=1.0)
        out_path = "./vision/db/test_result_preview.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\n📸 [静态图自检] 测试图片: {test_img_path}")
        print(f"   👉 识别结果: {matched_family}")
        print(f"   👉 已生成带人脸框与5点关键点的核验图: {os.path.abspath(out_path)}")

    if "--no-camera" in sys.argv:
        return

    print("\n🎥 正在打开 Mac 摄像头进行实时人脸核验...")
    cap = cv2.VideoCapture(0)
    time.sleep(0.8)

    ret, frame = False, None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    if not ret or frame is None:
        print("\n⚠️  未能读取摄像头画面。请完全退出 (Cmd+Q) 并重新打开 DTerminal.app 后重试。")
        cap.release()
        return

    print(f"✅ 摄像头已成功开启！(分辨率: {frame.shape[1]}x{frame.shape[0]})")
    print("   - 在弹出的画面窗口按 [S] 键 : 一键抓拍当前室内正脸照加入底库（多图融合可显著提高相似度分）")
    print("   - 按 [R] 键 : 热重载 ./vision/db/faces 照片底库")
    print("   - 按 [Q] 或 [ESC] 键 : 退出\n")

    fail_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            fail_count += 1
            if fail_count > 30:
                break
            time.sleep(0.03)
            continue
        fail_count = 0

        raw_frame = frame.copy()
        annotated, matched_family, suspicious = engine.analyze_frame(frame, tuya_distance_m=1.0)
        cv2.imshow("Face Recognition Test (YuNet + SFace ONNX)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            engine.reload_family_db()
        elif key == ord("s"):
            save_path = f"./vision/db/faces/爸爸_{int(time.time())}.jpg"
            cv2.imwrite(save_path, raw_frame)
            print(f"📸 已抓拍当前环境正脸并存入底库: {save_path}，正在自动融合特征...")
            engine.reload_family_db()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
