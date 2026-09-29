import os
import sys
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

    # 1. 对已保存的底库图片（如 爸爸1.jpg）执行一次自检识别并生成可视化标注图
    test_img_path = "./vision/db/faces/爸爸1.jpg"
    if os.path.exists(test_img_path):
        img_array = np.fromfile(test_img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        annotated, matched_family, _ = engine.analyze_frame(img.copy(), tuya_distance_m=1.0)
        out_path = "./vision/db/test_result_爸爸1.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\n📸 [静态图自检] 测试图片: {test_img_path}")
        print(f"   👉 识别结果: {matched_family}")
        print(f"   👉 已生成带人脸框与5点关键点的核验图: {os.path.abspath(out_path)}")

    # 2. 如果附加 --camera 参数（或直接交互运行），打开 Mac 摄像头实时测试
    if "--no-camera" in sys.argv:
        return

    print("\n🎥 正在打开 Mac 摄像头进行实时人脸核验 (按 [Q] 键退出，按 [R] 键热重载照片底库)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ℹ️ 无法打开摄像头（若在无 GUI 终端运行，可查看上方静态图自检结果）。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, matched_family, suspicious = engine.analyze_frame(frame, tuya_distance_m=1.0)
        cv2.imshow("Face Recognition Test (YuNet + SFace ONNX)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            engine.reload_family_db()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
