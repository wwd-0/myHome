import os
import sys
import cv2
import yaml
import time
import argparse
import numpy as np

from core.detector import LightweightDetector
from core.recognizer import PersonRecognizer
from core.door_controller import DoorUnlockController


def run_live_door_monitor(camera_source=0, config_path="config.yaml", show_gui=True):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg["system"].get("device", "mps")
    weights_path = cfg["paths"]["best_model_path"]
    gallery_path = cfg["paths"]["gallery_file"]
    similarity_thresh = cfg["thresholds"]["similarity_threshold"]
    unlock_thresh = cfg["thresholds"]["unlock_threshold"]
    min_face_w = cfg["thresholds"]["min_face_width_px"]

    door_cfg = cfg["door_unlock"]
    enable_unlock = door_cfg.get("enable_auto_unlock", True)
    confirm_frames = door_cfg.get("confirm_consecutive_frames", 3)
    cooldown_sec = door_cfg.get("cooldown_seconds", 10.0)
    ha_webhook = door_cfg.get("ha_webhook_url", "")

    print("\n=======================================================")
    print("🚪 启动入户人员检测与自动开门实时核验系统")
    print(f"视频源: {camera_source}")
    print(f"识别模型: {cfg['system'].get('model_type', 'LightPersonIDNet')} (设备: {device})")
    print(f"安全阈值: 识别={similarity_thresh} | 开门={unlock_thresh}")
    print(f"开门防抖规则: 连续 {confirm_frames} 帧确认自家人，冷却 {cooldown_sec} 秒")
    if ha_webhook:
        print(f"HA Webhook 联动: {ha_webhook}")
    print("=======================================================\n")

    detector = LightweightDetector(target_size=(256, 128))
    recognizer = PersonRecognizer(
        weights_path=weights_path,
        gallery_path=gallery_path,
        model_type=cfg["system"].get("model_type", "LightPersonIDNet"),
        similarity_threshold=similarity_thresh,
        unlock_threshold=unlock_thresh,
        min_face_width_px=min_face_w,
        device=device
    )

    allow_with_stranger = door_cfg.get("allow_unlock_with_stranger", False)
    stranger_alert_webhook = door_cfg.get("ha_stranger_alert_webhook_url", "")

    door_controller = DoorUnlockController(
        enabled=enable_unlock,
        confirm_consecutive_frames=confirm_frames,
        cooldown_seconds=cooldown_sec,
        allow_unlock_with_stranger=allow_with_stranger,
        ha_webhook_url=ha_webhook,
        ha_stranger_alert_webhook_url=stranger_alert_webhook
    )

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头/视频流: {camera_source}")
        return

    # 尝试设置合适分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_history = []
    t_prev = time.time()

    print("🟢 实时监控运行中... (按 'q' 键退出，按 's' 抓拍当前画面)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 视频流中断，尝试重连...")
            time.sleep(1.0)
            continue

        t_start = time.time()
        # 1. 检测画面中的所有人体主体 (支持多人在场)
        persons = detector.detect(frame)

        # 2. 批量并行高并发识别所有人 (耗时仅 ~5.5ms)
        recognition_results = recognizer.identify_batch(persons)

        # 3. 多人同框安防状态机仲裁 (全自家人开门 / 陌生人拒识 / 防尾随拦截)
        unlocked, door_status = door_controller.process_frame_results(recognition_results)

        # 绘制检测框与状态信息
        if show_gui:
            for res in recognition_results:
                x, y, w, h = res.bbox
                is_stranger = res.is_stranger
                color = (0, 255, 0) if not is_stranger else (0, 0, 255)

                # 绘制主体边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # 状态标签
                status_tag = "正面" if (res.has_face and not res.is_profile) else ("侧脸" if res.is_profile else "主体")
                tag_text = f"{res.person_name} ({res.similarity_score:.2f}) [{status_tag}]"

                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 6, y), color, -1)
                cv2.putText(frame, tag_text, (x + 3, max(th + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        t_cost_ms = (time.time() - t_start) * 1000.0
        fps_curr = 1.0 / max(0.001, time.time() - t_prev)
        t_prev = time.time()
        fps_history.append(fps_curr)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        if show_gui:
            # 顶部 HUD 状态栏
            hud_bg = np.zeros((45, frame.shape[1], 3), dtype=np.uint8)
            frame[0:45, 0:frame.shape[1]] = cv2.addWeighted(frame[0:45, 0:frame.shape[1]], 0.3, hud_bg, 0.7, 0)

            # 开门状态字体颜色设定
            if "已开门" in door_status:
                door_color = (0, 255, 255)
            elif "核验中" in door_status:
                door_color = (0, 200, 255)
            elif "尾随" in door_status:
                door_color = (0, 0, 255)
            else:
                door_color = (200, 200, 200)

            cv2.putText(frame, f"FPS: {avg_fps:.1f} | 延迟: {t_cost_ms:.1f}ms | 人数: {len(persons)}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, door_status, (420, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, door_color, 2)

            cv2.imshow("入户人员核验与自动开门系统 (按Q退出)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snap_name = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, frame)
                print(f"📸 抓拍已保存: {snap_name}")
        else:
            if unlocked:
                print(f"🎉 触发开门动作: {best_person_result.person_name} (FPS: {avg_fps:.1f})")

    cap.release()
    if show_gui:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="入户人员检测与自动开门实时服务")
    parser.add_argument("--camera", default=0, help="摄像头设备号或 RTSP 视频流地址 (如 0 或 rtsp://...)")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--no-gui", action="store_true", help="无界面后台模式")
    args = parser.parse_args()

    # 尝试将数字转为 int (如 "0" -> 0)
    cam = args.camera
    if isinstance(cam, str) and cam.isdigit():
        cam = int(cam)

    cfg_file = args.config
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(os.path.dirname(__file__), "config.yaml")

    run_live_door_monitor(camera_source=cam, config_path=cfg_file, show_gui=not args.no_gui)
