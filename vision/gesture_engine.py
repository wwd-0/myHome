import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


# MediaPipe 21 个手部骨骼关键点连线拓扑
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (9, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # 小指与掌骨
]


class _SimpleLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class GestureEngine:
    """
    视觉模块 2：基于 MediaPipe Hands 21 个 3D 关键点的零训练手势识别引擎
    支持两种推理后端（优先使用 OpenCV 原生加载 ./vision/models/ 下的开源 ONNX 模型权重，无需 pip 安装 mediapipe）：
      1. [默认推荐] OpenCV DNN 原生加载 palm_detection_mediapipe_2023feb.onnx + handpose_estimation_mediapipe_2023feb.onnx
      2. [可选兼容] 若安装了 mediapipe 包则也可调用
    支持手势:
      - "OK"        (拇指与食指捏合，中指/无名指/小指伸直 👌)
      - "VICTORY"   (剪刀手 ✌️：食指与中指伸直，无名指/小指弯曲)
      - "OPEN_PALM" (五指张开 🖐️)
      - "THUMB_UP"  (大拇指点赞 👍)
    """

    def __init__(self, gesture_cfg: dict):
        self.unlock_gestures: List[str] = gesture_cfg.get(
            "unlock_gestures", ["OK", "VICTORY", "OPEN_PALM", "THUMB_UP"]
        )
        self.confirm_frames: int = int(gesture_cfg.get("confirm_frames", 5))
        self.person_gesture_map: Dict[str, str] = gesture_cfg.get("person_gesture_map", {}) or {}

        self.models_dir = gesture_cfg.get("models_dir", "./vision/models")
        self.palm_model_path = gesture_cfg.get(
            "palm_model_path", os.path.join(self.models_dir, "palm_detection_mediapipe_2023feb.onnx")
        )
        self.hand_model_path = gesture_cfg.get(
            "hand_model_path", os.path.join(self.models_dir, "handpose_estimation_mediapipe_2023feb.onnx")
        )

        self._streak_gesture: Optional[str] = None
        self._streak_count: int = 0

        self.backend: Optional[str] = None  # "opencv_onnx" 或 "mediapipe"
        self.palm_net = None
        self.hand_net = None
        self.anchors = self._generate_palm_anchors()
        self.hands = None
        self.mp_draw = None

        self._init_model()

    @staticmethod
    def _generate_palm_anchors() -> np.ndarray:
        """生成 MediaPipe Palm Detection (192x192) 的 2016 个 SSD 先验锚框"""
        anchors = []
        # Stride 8: 24x24 网格，每格 2 个 anchor (共 1152 个)
        for y in range(24):
            for x in range(24):
                cx = (x + 0.5) / 24.0
                cy = (y + 0.5) / 24.0
                for _ in range(2):
                    anchors.append([cx, cy])
        # Stride 16: 12x12 网格，每格 6 个 anchor (共 864 个)
        for y in range(12):
            for x in range(12):
                cx = (x + 0.5) / 12.0
                cy = (y + 0.5) / 12.0
                for _ in range(6):
                    anchors.append([cx, cy])
        return np.array(anchors, dtype=np.float32)

    def _init_model(self):
        if os.path.exists(self.palm_model_path) and os.path.exists(self.hand_model_path):
            try:
                self.palm_net = cv2.dnn.readNetFromONNX(self.palm_model_path)
                self.hand_net = cv2.dnn.readNetFromONNX(self.hand_model_path)
                self.backend = "opencv_onnx"
                print(
                    f"🖐️ [Vision-Gesture] 已使用 OpenCV 原生加载手势 ONNX 权重 (零 pip 依赖):\n"
                    f"   - 手掌检测: {self.palm_model_path}\n"
                    f"   - 21点骨骼: {self.hand_model_path}\n"
                    f"   - 支持开门手势: {self.unlock_gestures}"
                )
                return
            except Exception as e:
                print(f"⚠️ [Vision-Gesture] OpenCV 加载手势 ONNX 失败: {e}")

        if mp is not None:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            self.backend = "mediapipe"
            print(f"🖐️ [Vision-Gesture] MediaPipe 手势引擎已就绪 (开门手势: {self.unlock_gestures})")
            return

        print("⚠️ [Vision-Gesture] 未找到 ./vision/models/ 手势 ONNX 权重文件。")

    def _detect_hand_landmarks_onnx(self, frame: np.ndarray) -> Optional[List[_SimpleLandmark]]:
        """使用 OpenCV DNN 运行 Palm Detection + 21点 HandPose ONNX 模型提取关键点"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. 等比例缩放并居中 Padding 到 192x192 送入 Palm Detector
        scale = min(192.0 / w, 192.0 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(rgb, (nw, nh))
        pad_top = (192 - nh) // 2
        pad_left = (192 - nw) // 2
        padded = np.zeros((192, 192, 3), dtype=np.uint8)
        padded[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized

        blob_p = (padded.astype(np.float32) / 255.0)[np.newaxis, ...]  # (1, 192, 192, 3)
        self.palm_net.setInput(blob_p)
        reg_out, score_out = self.palm_net.forward(["Identity", "Identity_1"])

        scores = 1.0 / (1.0 + np.exp(-np.clip(score_out[0, :, 0], -50.0, 50.0)))
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < 0.60:
            return None

        row = reg_out[0, best_idx]
        anchor = self.anchors[best_idx]

        # 解码手掌中心与大小 (归一化到 192x192 空间)
        cx = anchor[0] + row[0] / 192.0
        cy = anchor[1] + row[1] / 192.0
        pw = row[2] / 192.0
        ph = row[3] / 192.0

        # 解码腕部关键点 kp0 与中指掌指关节 kp2，用于计算手掌旋转角与中心偏移
        kp0_x = anchor[0] + row[4] / 192.0
        kp0_y = anchor[1] + row[5] / 192.0
        kp2_x = anchor[0] + row[8] / 192.0
        kp2_y = anchor[1] + row[9] / 192.0

        # 映射回原图像素坐标
        def to_orig(nx, ny):
            ox = (nx * 192.0 - pad_left) / scale
            oy = (ny * 192.0 - pad_top) / scale
            return ox, oy

        ocx, ocy = to_orig(cx, cy)
        okp0_x, okp0_y = to_orig(kp0_x, kp0_y)
        okp2_x, okp2_y = to_orig(kp2_x, kp2_y)
        box_size = max(pw * 192.0 / scale, ph * 192.0 / scale) * 2.6

        # 手掌中心向手指方向微移 0.5 倍手掌高度，包裹全部手指
        vx, vy = okp2_x - okp0_x, okp2_y - okp0_y
        v_len = np.hypot(vx, vy) + 1e-6
        hand_cx = ocx + (vx / v_len) * (box_size * 0.18)
        hand_cy = ocy + (vy / v_len) * (box_size * 0.18)

        # 计算手掌旋转角（使中指朝向正上方 -Y 轴）
        angle_rad = np.pi * 0.5 - np.arctan2(-vy, vx)
        angle_deg = float(np.degrees(angle_rad))

        # 仿射旋转裁剪出 224x224 手部区域送入 HandPose 21点模型
        rot_mat = cv2.getRotationMatrix2D((hand_cx, hand_cy), angle_deg, 224.0 / (box_size + 1e-6))
        rot_mat[0, 2] += 112.0 - hand_cx
        rot_mat[1, 2] += 112.0 - hand_cy
        hand_crop = cv2.warpAffine(rgb, rot_mat, (224, 224), flags=cv2.INTER_LINEAR)

        blob_h = (hand_crop.astype(np.float32) / 255.0)[np.newaxis, ...]  # (1, 224, 224, 3)
        self.hand_net.setInput(blob_h)
        lms_out, hand_score_out = self.hand_net.forward(["Identity", "Identity_1"])

        if float(hand_score_out[0, 0]) < 0.50:
            return None

        # 将 21x3 关键点逆变换回原图归一化坐标 [0.0, 1.0]
        pts_crop = lms_out[0].reshape(21, 3)
        inv_mat = cv2.invertAffineTransform(rot_mat)
        landmarks: List[_SimpleLandmark] = []
        for i in range(21):
            px, py, pz = pts_crop[i]
            ox = inv_mat[0, 0] * px + inv_mat[0, 1] * py + inv_mat[0, 2]
            oy = inv_mat[1, 0] * px + inv_mat[1, 1] * py + inv_mat[1, 2]
            landmarks.append(_SimpleLandmark(x=float(ox / w), y=float(oy / h), z=float(pz)))

        return landmarks

    @staticmethod
    def _draw_hand_skeleton(frame: np.ndarray, lm: List[_SimpleLandmark]):
        """在画面上绘制 21 点手部骨骼连线与关节点"""
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], (0, 255, 180), 2, cv2.LINE_AA)
        for idx, (px, py) in enumerate(pts):
            radius = 5 if idx in (4, 8, 12, 16, 20) else 3
            color = (0, 120, 255) if idx in (4, 8, 12, 16, 20) else (255, 255, 255)
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def _classify_21_landmarks(lm) -> str:
        """
        利用 21 个手指关键点的几何相对位置进行零训练判定
        lm[0]: 手腕
        指尖 ID: 拇指(4), 食指(8), 中指(12), 无名指(16), 小指(20)
        第二关节(PIP) ID: 拇指(2), 食指(6), 中指(10), 无名指(14), 小指(18)
        """
        def dist_to_wrist(idx):
            return np.hypot(lm[idx].x - lm[0].x, lm[idx].y - lm[0].y)

        index_up = dist_to_wrist(8) > dist_to_wrist(6) * 1.13
        middle_up = dist_to_wrist(12) > dist_to_wrist(10) * 1.13
        ring_up = dist_to_wrist(16) > dist_to_wrist(14) * 1.13
        pinky_up = dist_to_wrist(20) > dist_to_wrist(18) * 1.13

        # 拇指指尖(4)与食指指尖(8)的欧氏距离（用于判定 OK 手势捏合）
        pinch_dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        palm_scale = dist_to_wrist(9) + 1e-6  # 手掌基准长度

        # 判定 1: "OK" 手势 👌（拇指食指捏合，且中指、无名指、小指伸直）
        if (pinch_dist / palm_scale) < 0.42 and middle_up and ring_up and pinky_up:
            return "OK"

        # 判定 2: "VICTORY" 剪刀手 ✌️（食指、中指伸直，无名指、小指弯曲）
        if index_up and middle_up and (not ring_up) and (not pinky_up):
            return "VICTORY"

        # 判定 3: "OPEN_PALM" 五指张开 🖐️（四指全部伸直且拇指张开）
        thumb_open = np.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) / palm_scale > 0.32
        if index_up and middle_up and ring_up and pinky_up and thumb_open:
            return "OPEN_PALM"

        # 判定 4: "THUMB_UP" 大拇指朝上 👍（四指弯曲，拇指指尖明显高于食指根部）
        if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
            if lm[4].y < lm[5].y - 0.04:
                return "THUMB_UP"

        return "NONE"

    def detect_gesture(self, frame: np.ndarray, person_name: str = "爸爸") -> Tuple[np.ndarray, Optional[str]]:
        """
        检测画面中的手势并校验是否符合该家人的开门暗号
        返回: (绘制骨骼后的画面, 确认通过的手势名称 或 None)
        """
        current_gesture = "NONE"

        if self.backend == "opencv_onnx":
            landmarks = self._detect_hand_landmarks_onnx(frame)
            if landmarks is not None:
                self._draw_hand_skeleton(frame, landmarks)
                current_gesture = self._classify_21_landmarks(landmarks)
        elif self.backend == "mediapipe" and self.hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                current_gesture = self._classify_21_landmarks(hand_lms.landmark)

        # 检查是否满足连续帧防抖确认
        if current_gesture != "NONE":
            if current_gesture == self._streak_gesture:
                self._streak_count += 1
            else:
                self._streak_gesture = current_gesture
                self._streak_count = 1

            cv2.putText(
                frame,
                f"Gesture: {current_gesture} ({self._streak_count}/{self.confirm_frames})",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 215, 255),
                2,
            )
        else:
            self._streak_gesture = None
            self._streak_count = 0

        if self._streak_count >= self.confirm_frames:
            required_gesture = self.person_gesture_map.get(person_name)
            if required_gesture:
                if current_gesture == required_gesture:
                    self._streak_count = 0
                    return frame, current_gesture
            elif current_gesture in self.unlock_gestures:
                self._streak_count = 0
                return frame, current_gesture

        return frame, None
