import os
import subprocess
import sys
import time
import cv2
import numpy as np
from vision.face_engine import FaceEngine
from vision.gesture_engine import GestureEngine


def main():
    print("=" * 72)
    print("🧪 [人脸 + 手势 双模态开门测试] OpenCV 原生加载开源 ONNX 权重 (零 pip 额外依赖)")
    print("=" * 72)

    face_engine = FaceEngine(
        {
            "db_dir": "./vision/db/faces",
            "models_dir": "./vision/models",
            "similarity_threshold": 0.38,
            "min_face_width_px": 90,
        }
    )
    gesture_engine = GestureEngine(
        {
            "models_dir": "./vision/models",
            "unlock_gestures": ["OK", "VICTORY", "OPEN_PALM", "THUMB_UP"],
            "confirm_frames": 5,
        }
    )

    if "--no-camera" in sys.argv:
        print("✅ 人脸引擎与手势引擎 ONNX 模型均已加载就绪！")
        return

    print("\n🎥 正在打开 Mac 摄像头进行【人脸 + 手势】实时识别...")
    cap = cv2.VideoCapture(0)
    time.sleep(0.8)

    ret, frame = False, None
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)

    if not ret or frame is None:
        print("❌ 未能读取摄像头画面，请检查摄像头状态。")
        cap.release()
        return

    print(f"✅ 摄像头已开启！(分辨率: {frame.shape[1]}x{frame.shape[0]})")
    print("🖐️ 支持测试的 4 种开门手势（连续保持 5 帧即可触发开门）：")
    print("   1. [OK]        👌 拇指与食指捏成圈，其余三指伸直")
    print("   2. [VICTORY]   ✌️ 剪刀手（食指、中指伸直，其余弯曲）")
    print("   3. [OPEN_PALM] 🖐️ 五指全部张开")
    print("   4. [THUMB_UP]  👍 大拇指朝上点赞")
    print("   - 按 [Q] 或 [ESC] 键退出\n")

    unlock_banner_until = 0.0
    unlock_banner_text = ""

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        now = time.time()

        # 1. 人脸检测与家人识别
        frame, matched_family, _ = face_engine.analyze_frame(frame, tuya_distance_m=1.0)
        person_name = matched_family[0] if matched_family else "Guest"

        # 2. 手部 21 点骨骼检测与手势识别
        frame, matched_gesture = gesture_engine.detect_gesture(frame, person_name)

        # 3. 如果触发了确认手势
        if matched_gesture is not None:
            if matched_family is not None:
                unlock_banner_until = now + 3.5
                unlock_banner_text = f"DOOR UNLOCKED! [{person_name} + {matched_gesture}]"
                print(
                    f"🔓 [开门成功] 家人: {person_name} (人脸得分: {matched_family[1]:.2f}) + 手势: {matched_gesture}"
                )
            else:
                unlock_banner_until = now + 2.5
                unlock_banner_text = f"Gesture OK: [{matched_gesture}] (Waiting for Family Face)"
                print(f"🖐️ [手势识别成功] 检测到手势: {matched_gesture} (当前画面未锁定家人正脸)")

        # 4. 绘制开门成功横幅提示
        if now < unlock_banner_until:
            cv2.rectangle(frame, (15, 15), (760, 68), (0, 180, 60), -1)
            cv2.putText(
                frame,
                unlock_banner_text,
                (28, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Smart Door Test: Face + Hand Gesture (YuNet + SFace + MediaPipe ONNX)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            face_engine.reload_family_db()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
