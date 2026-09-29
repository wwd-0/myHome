import os
import re
import time
import urllib.request
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


# OpenCV Zoo 官方开源人脸模型权重直链（无需 pip 安装任何深度学习框架，OpenCV 原生直接加载）
YUNET_ONNX_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_ONNX_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)


class FaceEngine:
    """
    视觉模块 1：人脸检测 + 5点关键点仿射对齐 + 单张照片入库识别 + 可疑人员徘徊/遮挡监测
    支持两种推理后端（优先使用纯开源 ONNX 模型权重 + OpenCV 原生推理，零 pip 编译依赖）：
      1. [默认推荐] OpenCV 原生 ONNX 引擎：直接加载 ./vision/models/ 下的 YuNet (检测) + SFace (特征提取) .onnx 权重
      2. [可选兼容] InsightFace 引擎：若已安装 insightface 则可切换使用
    """

    def __init__(self, face_cfg: dict):
        self.db_dir = face_cfg.get("db_dir", "./vision/db/faces")
        self.models_dir = face_cfg.get("models_dir", "./vision/models")
        self.det_model_path = face_cfg.get(
            "det_model_path", os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        )
        self.rec_model_path = face_cfg.get(
            "rec_model_path", os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")
        )
        self.model_name = face_cfg.get("model_name", "buffalo_s")
        # 余弦相似度阈值（SFace 推荐阈值 >= 0.363，InsightFace 推荐 >= 0.45）
        self.sim_threshold = float(face_cfg.get("similarity_threshold", 0.38))
        self.min_face_width_px = int(face_cfg.get("min_face_width_px", 90))
        self.stranger_loiter_alert_sec = float(face_cfg.get("stranger_loiter_alert_sec", 10.0))
        self.no_face_occlusion_alert_sec = float(face_cfg.get("no_face_occlusion_alert_sec", 6.0))

        # 家庭成员特征向量底库: { "爸爸": np.ndarray(128或512,), ... }
        self.family_db: Dict[str, np.ndarray] = {}

        # 状态计时器
        self.stranger_first_seen: Optional[float] = None
        self.no_face_first_seen: Optional[float] = None
        self.last_alert_time: float = 0.0

        self.backend: Optional[str] = None  # "opencv_onnx" 或 "insightface"
        self.detector = None
        self.recognizer = None
        self.app = None

        self._init_model()
        self.reload_family_db()

    def _ensure_onnx_weights(self) -> bool:
        """检查本地 ./vision/models/ 下是否存在开源 ONNX 权重，若缺失则自动下载"""
        os.makedirs(self.models_dir, exist_ok=True)
        for path, url, desc in [
            (self.det_model_path, YUNET_ONNX_URL, "YuNet 人脸检测 ONNX 权重 (~230KB)"),
            (self.rec_model_path, SFACE_ONNX_URL, "SFace 人脸特征识别 ONNX 权重 (~36.9MB)"),
        ]:
            if not os.path.exists(path) or os.path.getsize(path) < 1024:
                print(f"📥 [Vision-Face] 正在下载开源模型权重: {desc} -> {path} ...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"✅ [Vision-Face] 下载完成: {path}")
                except Exception as e:
                    print(
                        f"⚠️ [Vision-Face] 自动下载失败 ({e})。\n"
                        f"   请手动下载权重文件放到 {path}:\n"
                        f"   curl -L -o {path} {url}"
                    )
                    return False
        return True

    def _init_model(self):
        # 优先使用 OpenCV 原生加载本地开源 .onnx 权重（无需 pip 安装 insightface/onnxruntime）
        if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF"):
            if self._ensure_onnx_weights():
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        model=self.det_model_path,
                        config="",
                        input_size=(640, 480),
                        score_threshold=0.65,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                    self.recognizer = cv2.FaceRecognizerSF.create(
                        model=self.rec_model_path,
                        config="",
                    )
                    self.backend = "opencv_onnx"
                    print(
                        f"👁️ [Vision-Face] 已使用 OpenCV 原生加载本地 ONNX 权重:\n"
                        f"   - 检测模型: {self.det_model_path}\n"
                        f"   - 识别模型: {self.rec_model_path}"
                    )
                    return
                except Exception as e:
                    print(f"⚠️ [Vision-Face] OpenCV 加载 ONNX 模型异常: {e}")

        # 备选：如果安装了 insightface
        if FaceAnalysis is not None:
            print(f"👁️ [Vision-Face] 正在加载 InsightFace 模型 ({self.model_name})...")
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            self.app = FaceAnalysis(name=self.model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.backend = "insightface"
            return

        print("❌ [Vision-Face] 未能加载任何人脸模型，请确认 ./vision/models/ 下的 ONNX 权重文件已就绪。")

    @staticmethod
    def _parse_person_name(fname: str) -> str:
        """
        从文件名解析家人称呼：
        兼容 '爸爸.jpg', '爸爸1.jpg', '爸爸_1.jpg', '爸爸_白天.jpg' -> 统一归为 '爸爸'
        """
        raw_name = os.path.splitext(fname)[0]
        base = raw_name.split("_")[0]
        cleaned = re.sub(r"\d+$", "", base).strip()
        return cleaned or base

    def _extract_faces(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        从图像中检测所有清晰人脸并提取归一化特征向量
        返回列表，每项为: (bbox_xyxy[4], landmarks_5x2, l2_normed_embedding)
        """
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if self.backend == "opencv_onnx" and self.detector is not None and self.recognizer is not None:
            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            if faces is not None:
                for face_row in faces:
                    # YuNet 输出前 4 位为 (x, y, w, h)，随后 10 位为 5 个关键点 (x, y)
                    x, y, bw, bh = face_row[:4]
                    bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
                    landmarks = face_row[4:14].reshape(5, 2)
                    # 使用 5 点关键点执行标准 112x112 仿射对齐并提取特征
                    aligned_face = self.recognizer.alignCrop(img, face_row)
                    feat = self.recognizer.feature(aligned_face).flatten().astype(np.float32)
                    norm_feat = feat / (np.linalg.norm(feat) + 1e-8)
                    results.append((bbox, landmarks, norm_feat))
        elif self.backend == "insightface" and self.app is not None:
            faces = self.app.get(img)
            for f in faces:
                results.append((f.bbox.astype(np.float32), f.kps, f.normed_embedding))
        return results

    def reload_family_db(self):
        """
        扫描 ./vision/db/faces 目录，自动完成家人照片特征提取入库
        支持：
          - 单张照片直接放 ./vision/db/faces/爸爸1.jpg 或 爸爸.jpg
          - 同一人的多张照片自动求均值并 L2 归一化
        """
        os.makedirs(self.db_dir, exist_ok=True)
        if self.backend is None:
            return

        temp_embeddings: Dict[str, List[np.ndarray]] = {}

        for fname in sorted(os.listdir(self.db_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            person_name = self._parse_person_name(fname)
            img_path = os.path.join(self.db_dir, fname)
            # 兼容中文文件名路径读取
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue

            extracted = self._extract_faces(img)
            if not extracted:
                print(f"⚠️ [Vision-Face] 照片 {fname} 中未检测到清晰人脸，请更换正脸照！")
                continue

            # 取画面中面积最大的一张脸
            main_face = max(extracted, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
            temp_embeddings.setdefault(person_name, []).append(main_face[2])

        self.family_db.clear()
        for person_name, emb_list in temp_embeddings.items():
            mean_emb = np.mean(emb_list, axis=0)
            norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            self.family_db[person_name] = norm_emb
            print(f"✅ [Vision-Face] 已入库家人 [{person_name}] (融合了 {len(emb_list)} 张照片, 特征维度: {len(norm_emb)})")

        if not self.family_db:
            print(f"ℹ️ [Vision-Face] 当前底库为空，请将家人单张照片放入: {os.path.abspath(self.db_dir)}/")

    def match_face(self, normed_embedding: np.ndarray) -> Tuple[str, float]:
        """1:N 极速余弦相似度比对"""
        best_name = "Stranger"
        best_score = -1.0

        for name, db_emb in self.family_db.items():
            score = float(np.dot(normed_embedding, db_emb))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.sim_threshold:
            return best_name, best_score
        return "Stranger", best_score

    def analyze_frame(
        self, frame: np.ndarray, tuya_distance_m: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[str, float]], Optional[str]]:
        """
        处理单帧画面：
        返回:
          - annotated_frame: 画好识别框与关键点的画面
          - matched_family: 若识别到门前自家人，返回 (name, score)，否则 None
          - suspicious_reason: 若触发可疑人员告警，返回告警原因字符串，否则 None
        """
        if self.backend is None:
            return frame, None, None

        extracted = self._extract_faces(frame)
        matched_family: Optional[Tuple[str, float]] = None
        has_stranger = False
        suspicious_reason: Optional[str] = None
        now = time.time()

        for bbox_f, landmarks, norm_emb in extracted:
            bbox = bbox_f.astype(int)
            face_width = bbox[2] - bbox[0]
            name, score = self.match_face(norm_emb)

            if name != "Stranger":
                if face_width >= self.min_face_width_px:
                    if matched_family is None or score > matched_family[1]:
                        matched_family = (name, score)
                    color = (0, 255, 0)  # 绿色：门前家人
                    tag = f"Family: {name} ({score:.2f})"
                else:
                    color = (0, 200, 200)  # 黄色：远处家人
                    tag = f"Far: {name} ({score:.2f})"
            else:
                has_stranger = True
                color = (0, 0, 255)  # 红色：陌生人
                tag = f"Stranger ({score:.2f})"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            for pt in landmarks.astype(int):
                cv2.circle(frame, (pt[0], pt[1]), 2, (255, 255, 0), -1)
            cv2.putText(
                frame,
                tag,
                (bbox[0], max(25, bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        # 安防规则 1：陌生人长时间徘徊检测
        if has_stranger and matched_family is None:
            if self.stranger_first_seen is None:
                self.stranger_first_seen = now
            else:
                loiter_sec = now - self.stranger_first_seen
                cv2.putText(
                    frame,
                    f"Stranger Loitering: {loiter_sec:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                if loiter_sec >= self.stranger_loiter_alert_sec and (now - self.last_alert_time > 30):
                    suspicious_reason = f"陌生人在门厅逗留超过 {loiter_sec:.1f} 秒"
                    self.last_alert_time = now
        else:
            self.stranger_first_seen = None

        # 安防规则 2：涂鸦毫米波雷达显示贴门有人(<1.2m)，但画面完全看不到人脸
        if len(extracted) == 0 and tuya_distance_m is not None and tuya_distance_m < 1.2:
            if self.no_face_first_seen is None:
                self.no_face_first_seen = now
            else:
                occ_sec = now - self.no_face_first_seen
                if occ_sec >= self.no_face_occlusion_alert_sec and (now - self.last_alert_time > 30):
                    suspicious_reason = (
                        f"涂鸦传感器检测到门前 {tuya_distance_m}m 有人，但连续 {occ_sec:.1f}秒 未露脸（疑似遮挡或撬锁）"
                    )
                    self.last_alert_time = now
        else:
            self.no_face_first_seen = None

        return frame, matched_family, suspicious_reason
