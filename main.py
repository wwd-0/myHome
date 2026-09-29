import os
import threading
import time
from typing import Optional
import cv2
import numpy as np
import yaml

from vision import FaceEngine, GestureEngine
from audio import VoiceEngine
from ha_connector import HomeAssistantConnector

try:
    import sounddevice as sd
except ImportError:
    sd = None


class SmartDoorController:
    """
    全屋智能双模态门禁总控状态机：
    1. 门厅涂鸦 Zigbee 人体存在传感器检测到人 -> 唤醒摄像头监测；
    2. 视觉模块 (FaceEngine) 识别是否为家里 6 个人（若为陌生人且逗留>10s则抓拍报警）；
    3. 一旦确认门口是自家人 (锁定家人 ID)，同时开启两条并行开门通道：
       - 【通道 A：人脸 + 手势】 (MediaPipe 检测到 OK / ✌️ 等开门手势)
       - 【通道 B：人脸 + 声纹】 (CAM++ 声纹模型与该家人 ID 1:1 语音匹配成功)
    满足任意一条通道即刻通过 Webhook 通知 Home Assistant 开门！
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        sys_cfg = self.cfg.get("system", {})
        self.test_mode: bool = bool(sys_cfg.get("test_mode", True))
        self.active_window_sec: float = float(sys_cfg.get("active_window_sec", 15.0))
        self.unlock_cooldown_sec: float = float(sys_cfg.get("unlock_cooldown_sec", 10.0))

        # 初始化视觉子系统 (./vision) 与音频子系统 (./audio)
        self.face_engine = FaceEngine(self.cfg.get("vision", {}).get("face", {}))
        self.gesture_engine = GestureEngine(self.cfg.get("vision", {}).get("gesture", {}))
        self.voice_engine = VoiceEngine(self.cfg.get("audio", {}))

        # 初始化 Home Assistant & 涂鸦 Zigbee 通信器
        self.ha = HomeAssistantConnector(
            self.cfg.get("home_assistant", {}),
            on_tuya_presence_callback=self.on_tuya_sensor_event,
        )

        # 状态控制变量
        self.sensor_wakeup_until: float = time.time() + 3600 if self.test_mode else 0.0
        self.tuya_distance_m: Optional[float] = 1.0 if self.test_mode else None
        self.last_unlock_time: float = 0.0

        # 后台声纹监听锁与最新验证结果
        self._voice_check_busy = False
        self._voice_verified_person: Optional[str] = None
        self._voice_verified_score: float = 0.0

    def on_tuya_sensor_event(self, presence: bool, distance_m: Optional[float]):
        """收到 SLZB-06 转发的涂鸦 Zigbee 人体存在传感器信号"""
        self.tuya_distance_m = distance_m
        if presence:
            self.sensor_wakeup_until = time.time() + self.active_window_sec
            print(f"📡 [Zigbee触发] 门厅涂鸦传感器检测到有人 (距离: {distance_m}m)，已唤醒视觉与声纹引擎！")

    def _async_check_voiceprint(self, target_person: str):
        """在后台线程采集 2 秒麦克风语音并与当前画面中的家人进行 1:1 声纹核对"""
        if self._voice_check_busy or sd is None or self.voice_engine.extractor is None:
            return
        if target_person not in self.voice_engine.voice_db:
            return

        self._voice_check_busy = True

        def _worker():
            try:
                sr = 16000
                duration = self.voice_engine.sample_duration_sec
                rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                samples = rec.flatten()
                passed, score = self.voice_engine.verify_person_1to1(target_person, samples, sr)
                if passed:
                    self._voice_verified_person = target_person
                    self._voice_verified_score = score
            except Exception:
                pass
            finally:
                self._voice_check_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def run(self):
        cam_source = self.cfg.get("vision", {}).get("camera_source", 0)
        cap = cv2.VideoCapture(cam_source)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头源: {cam_source}")
            return

        print("\n" + "=" * 72)
        print("🚀 全屋智能【人脸+手势 / 人脸+声纹】双通道门禁服务已启动！")
        print("   - 按 [空格键] : 模拟门厅涂鸦 Zigbee 传感器检测到有人进入")
        print("   - 按 [R 键]   : 热重载 ./vision/db/faces 和 ./audio/db/voices 照片与语音库")
        print("   - 按 [Q 键]   : 退出程序")
        print("=" * 72 + "\n")

        while True:
            now = time.time()

            # 如果涂鸦传感器未检测到人，进入低功耗休眠待机
            if now > self.sensor_wakeup_until:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # 1. 第一步：执行人脸检测、1:6 家人识别与可疑人员徘徊/遮挡监测
            frame, matched_family, suspicious_reason = self.face_engine.analyze_frame(
                frame, tuya_distance_m=self.tuya_distance_m
            )

            # 如果发现可疑人员徘徊超时或恶意遮挡，抓拍并通知 HA 报警
            if suspicious_reason:
                os.makedirs("alerts", exist_ok=True)
                snap_path = os.path.abspath(f"alerts/alert_{int(now)}.jpg")
                cv2.imwrite(snap_path, frame)
                self.ha.trigger_suspicious_alert(suspicious_reason, snap_path)

            # 2. 第二步：如果认出门口是自家人（且不在开门冷却期内），同时激活【手势】与【声纹】双通道
            in_cooldown = (now - self.last_unlock_time) < self.unlock_cooldown_sec
            if matched_family is not None and not in_cooldown:
                person_name, face_score = matched_family

                # 通道 A：检测【人脸 + 手势】
                frame, matched_gesture = self.gesture_engine.detect_gesture(frame, person_name)
                if matched_gesture is not None:
                    self.ha.trigger_door_unlock(
                        person_name=person_name,
                        auth_mode="人脸 + 手势",
                        face_score=face_score,
                        extra_detail=f"手势: {matched_gesture}",
                    )
                    self.last_unlock_time = time.time()

                # 通道 B：后台并行检测【人脸 + 声纹】
                self._async_check_voiceprint(person_name)
                if self._voice_verified_person == person_name:
                    v_score = self._voice_verified_score
                    self._voice_verified_person = None
                    self.ha.trigger_door_unlock(
                        person_name=person_name,
                        auth_mode="人脸 + 声纹",
                        face_score=face_score,
                        extra_detail=f"声纹相似度: {v_score:.2f}",
                    )
                    self.last_unlock_time = time.time()

            elif in_cooldown:
                cv2.putText(
                    frame,
                    "DOOR UNLOCKED! (Cooldown...)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 128),
                    2,
                )

            if self.test_mode:
                cv2.imshow("Smart Home AI Door Guard (Face + Gesture / Voiceprint)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    self.on_tuya_sensor_event(True, 1.0)
                elif key == ord("r"):
                    self.face_engine.reload_family_db()
                    self.voice_engine.reload_voice_db()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = SmartDoorController("config.yaml")
    controller.run()
