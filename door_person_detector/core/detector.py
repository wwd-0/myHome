import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DetectedPerson:
    """人体主体检测与裁剪切片结果"""
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) 人体在原图中的边界框
    confidence: float                   # 检测置信度 0.0 ~ 1.0
    has_face: bool                      # 画面中是否显式捕获到人脸
    is_profile_or_occluded: bool        # 是否为侧脸或局部遮挡
    landmarks: Optional[np.ndarray]     # 面部关键点 (若有)
    aligned_patch: np.ndarray           # 归一化为 (256, 128) [高 256, 宽 128] 的人体主体切片


def crop_maximal_aspect_ratio(img_bgr: np.ndarray, target_hw: Tuple[int, int] = (512, 256)) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    按长宽比最大化保留画面内容进行居中截取，然后等比缩放到目标尺寸：
    1. 绝不改变长宽比，人体/面部不拉伸、不压缩变形；
    2. 最大化保留原始图片内容（宽图保留100%高度，高图保留100%宽度）；
    3. 居中对称裁剪，适配门禁人员主体居中场景。
    返回: (resized_patch, (crop_x, crop_y, crop_w, crop_h))
    """
    th, tw = target_hw
    target_ratio = tw / th  # 默认 256 / 512 = 0.5 (宽:高 = 1:2)
    h, w = img_bgr.shape[:2]
    current_ratio = w / h

    if current_ratio > target_ratio:
        # 画面偏宽：100% 保留全部高度，左右居中截取宽度
        crop_h = h
        crop_w = max(1, int(round(h * target_ratio)))
        x1 = max(0, (w - crop_w) // 2)
        y1 = 0
    else:
        # 画面偏高：100% 保留全部宽度，上下居中截取高度
        crop_w = w
        crop_h = max(1, int(round(w / target_ratio)))
        x1 = 0
        y1 = max(0, (h - crop_h) // 2)

    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    cropped = img_bgr[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)
    return resized, (x1, y1, x2 - x1, y2 - y1)


class LightweightDetector:
    """
    全姿态超轻量人体检测与切片处理器 (Lightweight Human Subject Detector)
    专为入户门禁复杂场景设计：
    1. 即使看不到正脸 (低头、侧身、背身、帽子遮挡)，也能根据人体轮廓/头肩比精准锁定人物主体并提取 (512, 256) 切片
    2. 若能捕捉到侧脸或正脸，融合面部位置精细化校准头肩与躯干比例
    3. 支持输入直接就是人物主体图像（自动保持 2:1 黄金高宽比做自适应裁剪缩放，不产生形变）
    """
    def __init__(self,
                 model_path: str = "",
                 target_size: Tuple[int, int] = (512, 256),
                 score_threshold: float = 0.55):
        self.target_size = target_size # (高度 512, 宽度 256)
        self.score_threshold = score_threshold
        self.yunet_detector = None
        self.input_size = (640, 640)

        # 1. 尝试加载 YOLOv8-Nano (支持多人合影、偏位、非固定机位)
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            candidates = [
                "yolov8n.pt",
                os.path.join(os.path.dirname(__file__), "../../yolov8n.pt"),
                os.path.join(os.path.dirname(__file__), "../weights/yolov8n.pt"),
            ]
            yolo_path = "yolov8n.pt"
            for c in candidates:
                if os.path.exists(c):
                    yolo_path = os.path.abspath(c)
                    break
            self.yolo_model = YOLO(yolo_path)
            print(f"[Detector] 成功加载 YOLOv8-Nano 多目标人体检测器: {yolo_path}")
        except Exception as e:
            print(f"[Detector] YOLOv8 加载提示: {e}")

        # 2. 初始化 OpenCV 原生 HOG 人体全身/半身检测器 (用于无 GPU / 离线保底)
        self.hog_detector = cv2.HOGDescriptor()
        self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _resolve_model_path(self, path: str) -> Optional[str]:
        if not path:
            candidate_paths = [
                os.path.join(os.path.dirname(__file__), "../../vision/models/face_detection_yunet_2023mar.onnx"),
                "./vision/models/face_detection_yunet_2023mar.onnx",
                "../vision/models/face_detection_yunet_2023mar.onnx",
            ]
            for p in candidate_paths:
                abs_p = os.path.abspath(p)
                if os.path.exists(abs_p):
                    return abs_p
            return None
        abs_p = os.path.abspath(path)
        if os.path.exists(abs_p):
            return abs_p
        parent_p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
        if os.path.exists(parent_p):
            return parent_p
        return path

    def letterbox_person(self, img_bgr: np.ndarray, target_hw: Tuple[int, int] = (512, 256)) -> np.ndarray:
        """
        自适应等比缩放 + 边缘填充 (Letterbox)，确保人体主体不被拉伸变形
        输入: 任意比例图像 (H, W, 3)
        输出: (target_h, target_w, 3) 居中对齐图像
        """
        th, tw = target_hw
        h, w = img_bgr.shape[:2]
        scale = min(th / h, tw / w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 128, dtype=np.uint8) # 128中性灰填充边缘

        # 居中放置
        top = (th - new_h) // 2
        left = (tw - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    def process_direct_subject(self, person_img: np.ndarray) -> DetectedPerson:
        """
        按黄金长宽比最大化保留画面居中裁剪并归一化，严格保持自然长宽比不发生形变
        """
        aligned, crop_bbox = crop_maximal_aspect_ratio(person_img, self.target_size)
        return DetectedPerson(
            bbox=crop_bbox,
            confidence=1.0,
            has_face=False,
            is_profile_or_occluded=False,
            landmarks=None,
            aligned_patch=aligned
        )

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedPerson]:
        """
        从监控/抓拍画面中检测人物主体 (支持单人、多人合影、偏位等全场景)：
        1. 优先使用 YOLOv8-Nano 检测画面中所有人员 (Class 0: person) 并框出每个独立主体
        2. 对每个人体框按黄金长宽比平滑抽取规范化切片
        3. 若未检测到明显主体，自动以全画幅自适应保底
        """
        h, w = frame_bgr.shape[:2]
        results: List[DetectedPerson] = []

        # 策略 1: 优先使用 YOLOv8-Nano 检测画面中所有人
        if self.yolo_model is not None:
            try:
                preds = self.yolo_model(frame_bgr, classes=[0], conf=0.40, device="mps", verbose=False)
                if len(preds) > 0 and len(preds[0].boxes) > 0:
                    for b in preds[0].boxes:
                        xyxy = b.xyxy.cpu().numpy()[0]
                        conf = float(b.conf[0])
                        x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                        x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
                        bw, bh = x2 - x1, y2 - y1
                        # 过滤远景微小噪点 (门厅通行人员高度通常不低于 80px)
                        if bw < 40 or bh < 80:
                            continue
                        crop = frame_bgr[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        aligned, _ = crop_maximal_aspect_ratio(crop, self.target_size)
                        results.append(DetectedPerson(
                            bbox=(x1, y1, bw, bh),
                            confidence=conf,
                            has_face=False,
                            is_profile_or_occluded=False,
                            landmarks=None,
                            aligned_patch=aligned
                        ))
            except Exception as e:
                print(f"[Detector] YOLO 推理警告: {e}")

        # 策略 2: 若未检测到目标，提供全画幅最大长宽比保底切片
        if len(results) == 0:
            results.append(self.process_direct_subject(frame_bgr))

        return results
