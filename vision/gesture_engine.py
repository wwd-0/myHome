from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class GestureEngine:
    """
    视觉模块 2：基于 MediaPipe Hands 21 个 3D 关键点的零训练手势识别引擎
    仅在识别出“自家人”后的开门窗口内激活，支持:
      - "OK"        (拇指与食指捏合，中指/无名指/小指伸直)
      - "VICTORY"   (剪刀手 ✌️：食指与中指伸直，其余弯曲)
      - "OPEN_PALM" (五指张开 🖐️)
      - "THUMB_UP"  (大拇指点赞 👍)
    """

    def __init__(self, gesture_cfg: dict):
        self.unlock_gestures: List[str] = gesture_cfg.get("unlock_gestures", ["OK", "VICTORY"])
        self.confirm_frames: int = int(gesture_cfg.get("confirm_frames", 5))
        self.person_gesture_map: Dict[str, str] = gesture_cfg.get("person_gesture_map", {}) or {}

        self._streak_gesture: Optional[str] = None
        self._streak_count: int = 0

        self.hands = None
        self.mp_draw = None
        if mp is not None:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )
            print(f"🖐️ [Vision-Gesture] MediaPipe 手势引擎已就绪 (开门手势: {self.unlock_gestures})")
        else:
            print("⚠️ [Vision-Gesture] 未安装 mediapipe，请运行: pip install mediapipe")

    @staticmethod
    def _classify_21_landmarks(lm) -> str:
        """
        利用 21 个手指关键点的几何相对位置进行零训练判定
        lm[0]: 手腕
        指尖 ID: 拇指(4), 食指(8), 中指(12), 无名指(16), 小指(20)
        第二关节(PIP) ID: 拇指(2), 食指(6), 中指(10), 无名指(14), 小指(18)
        """
        # 1. 判断食指、中指、无名指、小指是否伸直（指尖到腕部距离 > PIP关节到腕部距离）
        def dist_to_wrist(idx):
            return np.hypot(lm[idx].x - lm[0].x, lm[idx].y - lm[0].y)

        index_up = dist_to_wrist(8) > dist_to_wrist(6) * 1.15
        middle_up = dist_to_wrist(12) > dist_to_wrist(10) * 1.15
        ring_up = dist_to_wrist(16) > dist_to_wrist(14) * 1.15
        pinky_up = dist_to_wrist(20) > dist_to_wrist(18) * 1.15

        # 拇指指尖(4)与食指指尖(8)的欧氏距离（用于判定 OK 手势捏合）
        pinch_dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        palm_scale = dist_to_wrist(9) + 1e-6  # 手掌基准长度

        # 判定 1: "OK" 手势（拇指食指捏合，且中指、无名指、小指伸直）
        if (pinch_dist / palm_scale) < 0.38 and middle_up and ring_up and pinky_up:
            return "OK"

        # 判定 2: "VICTORY" 剪刀手 ✌️（食指、中指伸直，无名指、小指弯曲）
        if index_up and middle_up and (not ring_up) and (not pinky_up):
            return "VICTORY"

        # 判定 3: "OPEN_PALM" 五指张开 🖐️（四指全部伸直且拇指张开）
        thumb_open = np.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y) / palm_scale > 0.35
        if index_up and middle_up and ring_up and pinky_up and thumb_open:
            return "OPEN_PALM"

        # 判定 4: "THUMB_UP" 大拇指朝上 👍（四指弯曲，拇指指尖明显高于食指根部）
        if (not index_up) and (not middle_up) and (not ring_up) and (not pinky_up):
            if lm[4].y < lm[5].y - 0.05:
                return "THUMB_UP"

        return "NONE"

    def detect_gesture(self, frame: np.ndarray, person_name: str) -> Tuple[np.ndarray, Optional[str]]:
        """
        检测画面中的手势并校验是否符合该家人的开门暗号
        返回: (绘制骨骼后的画面, 确认通过的手势名称 或 None)
        """
        if self.hands is None:
            return frame, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        current_gesture = "NONE"

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
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2,
            )
        else:
            self._streak_gesture = None
            self._streak_count = 0

        if self._streak_count >= self.confirm_frames:
            # 校验是否匹配该家人的专属手势或全局开门手势
            required_gesture = self.person_gesture_map.get(person_name)
            if required_gesture:
                if current_gesture == required_gesture:
                    self._streak_count = 0
                    return frame, current_gesture
            elif current_gesture in self.unlock_gestures:
                self._streak_count = 0
                return frame, current_gesture

        return frame, None
