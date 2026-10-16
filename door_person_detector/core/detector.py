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

        # 1. 尝试加载 YuNet 快速人脸/头肩锚定引擎 (仅 232KB)
        resolved_path = self._resolve_model_path(model_path)
        if resolved_path and os.path.exists(resolved_path):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=resolved_path,
                    config="",
                    input_size=self.input_size,
                    score_threshold=self.score_threshold,
                    nms_threshold=0.3,
                    top_k=5000
                )
                print(f"[Detector] 成功加载超轻量前置检测器: {resolved_path}")
            except Exception as e:
                print(f"[Detector] YuNet 加载警告: {e}")

        # 2. 初始化 OpenCV 原生 HOG 人体全身/半身检测器 (用于人脸完全看不见时的无脸人形保底)
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

    def letterbox_person(self, img_bgr: np.ndarray, target_hw: Tuple[int, int] = (256, 128)) -> np.ndarray:
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
        从完整监控/摄像头画面中检测人物主体：
        - 优先检测人脸/侧脸，并依据人体工学以头肩为锚点自适应外延躯干
        - 若完全未检测到人脸（如低头找钥匙、背身），无缝切换至人体 HOG / 轮廓检测
        - 若画面整体就是主体，返回整体归一化切片
        """
        h, w = frame_bgr.shape[:2]
        results: List[DetectedPerson] = []

        # 策略 1: 基于面部/侧脸锚点外延人体主体
        if self.yunet_detector is not None:
            if self.input_size != (w, h):
                self.input_size = (w, h)
                self.yunet_detector.setInputSize((w, h))

            status, faces = self.yunet_detector.detect(frame_bgr)
            if status == 0 and faces is not None and len(faces) > 0:
                for face in faces:
                    conf = float(face[14])
                    if conf < self.score_threshold:
                        continue

                    fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    landmarks = face[4:14].reshape((5, 2)).astype(np.float32)

                    # 计算眼睛与鼻尖的相对偏转，判断是否为侧脸 (Profile Face)
                    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
                    dist_l = np.linalg.norm(nose - left_eye)
                    dist_r = np.linalg.norm(nose - right_eye)
                    eye_dist = np.linalg.norm(right_eye - left_eye)
                    is_profile = False
                    if eye_dist > 1e-4:
                        ratio = abs(dist_l - dist_r) / eye_dist
                        if ratio > 0.45: # 两眼到鼻子距离显著不对称，判定为侧脸
                            is_profile = True

                    # 根据头部位置自适应外延为 2:1 黄金高宽比的人体头肩躯干区域
                    # 人体通常头高占全身约 1/7，上半身占 1/3.5
                    person_w = int(fw * 2.4)
                    person_h = int(person_w * 2.0) # 保持 2:1 高宽比

                    # 头部居于上半部分 (顶部留 15% 边距)
                    center_x = fx + fw // 2
                    px1 = max(0, center_x - person_w // 2)
                    py1 = max(0, fy - int(fh * 0.3))
                    px2 = min(w, px1 + person_w)
                    py2 = min(h, py1 + person_h)

                    person_crop = frame_bgr[py1:py2, px1:px2]
                    if person_crop.size > 0:
                        aligned = self.letterbox_person(person_crop, self.target_size)
                        results.append(DetectedPerson(
                            bbox=(px1, py1, px2 - px1, py2 - py1),
                            confidence=conf,
                            has_face=True,
                            is_profile_or_occluded=is_profile,
                            landmarks=landmarks,
                            aligned_patch=aligned
                        ))

        # 策略 2: 若未检测到人脸（看不到正脸/低头/背身），启动 HOG 人体全身/半身检测
        if len(results) == 0:
            boxes, weights = self.hog_detector.detectMultiScale(
                frame_bgr,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05
            )
            for (bx, by, bw, bh), weight in zip(boxes, weights):
                if weight < 0.2:
                    continue
                # 人体框裁剪
                bx1, by1 = max(0, bx), max(0, by)
                bx2, by2 = min(w, bx + bw), min(h, by + bh)
                crop = frame_bgr[by1:by2, bx1:bx2]
                if crop.size > 0:
                    aligned = self.letterbox_person(crop, self.target_size)
                    results.append(DetectedPerson(
                        bbox=(bx1, by1, bx2 - bx1, by2 - by1),
                        confidence=float(min(1.0, 0.6 + weight * 0.4)),
                        has_face=False,
                        is_profile_or_occluded=True, # 无脸或背身
                        landmarks=None,
                        aligned_patch=aligned
                    ))

        # 策略 3: 如果上述均未触发，但用户保证画面即主体，提供全局保底切片
        if len(results) == 0:
            results.append(self.process_direct_subject(frame_bgr))

        return results
