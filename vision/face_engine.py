import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


class FaceEngine:
    """
    视觉模块 1：人脸检测 + 5点仿射对齐 + 单张照片入库识别 + 可疑人员徘徊/遮挡监测
    底层采用 InsightFace (SCRFD + ArcFace MobileFaceNet)
    """

    def __init__(self, face_cfg: dict):
        self.db_dir = face_cfg.get("db_dir", "./vision/db/faces")
        self.model_name = face_cfg.get("model_name", "buffalo_s")
        self.sim_threshold = float(face_cfg.get("similarity_threshold", 0.45))
        self.min_face_width_px = int(face_cfg.get("min_face_width_px", 90))
        self.stranger_loiter_alert_sec = float(face_cfg.get("stranger_loiter_alert_sec", 10.0))
        self.no_face_occlusion_alert_sec = float(face_cfg.get("no_face_occlusion_alert_sec", 6.0))

        # 家庭成员 512 维特征向量底库: { "爸爸": np.ndarray(512,), ... }
        self.family_db: Dict[str, np.ndarray] = {}

        # 状态计时器
        self.stranger_first_seen: Optional[float] = None
        self.no_face_first_seen: Optional[float] = None
        self.last_alert_time: float = 0.0

        self.app = None
        self._init_model()
        self.reload_family_db()

    def _init_model(self):
        if FaceAnalysis is None:
            print("⚠️ [Vision-Face] 未安装 insightface，请运行: pip install insightface onnxruntime")
            return

        print(f"👁️ [Vision-Face] 正在加载人脸模型 ({self.model_name})...")
        # 在 Mac 上优先启用 CoreMLExecutionProvider 硬件加速，回退到 CPUExecutionProvider
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def reload_family_db(self):
        """
        扫描 ./vision/db/faces 目录，自动完成家人照片特征提取入库
        支持：
          - 单张照片直接放 ./vision/db/faces/爸爸.jpg
          - 或同一人的多张照片命名如 爸爸_1.jpg, 爸爸_2.jpg（自动求均值归一化）
        """
        os.makedirs(self.db_dir, exist_ok=True)
        if self.app is None:
            return

        temp_embeddings: Dict[str, List[np.ndarray]] = {}

        for fname in sorted(os.listdir(self.db_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            # 支持 "爸爸.jpg" 或 "爸爸_白天.jpg" 归入同一个家人名下
            raw_name = os.path.splitext(fname)[0]
            person_name = raw_name.split("_")[0]

            img_path = os.path.join(self.db_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = self.app.get(img)
            if not faces:
                print(f"⚠️ [Vision-Face] 照片 {fname} 中未检测到清晰人脸，请更换正脸照！")
                continue

            # 取画面中面积最大的一张脸
            main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            temp_embeddings.setdefault(person_name, []).append(main_face.normed_embedding)

        self.family_db.clear()
        for person_name, emb_list in temp_embeddings.items():
            # 多张照片特征求平均并重新做 L2 归一化（单张照片则保持原样）
            mean_emb = np.mean(emb_list, axis=0)
            norm_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            self.family_db[person_name] = norm_emb
            print(f"✅ [Vision-Face] 已入库家人 [{person_name}] (融合了 {len(emb_list)} 张照片)")

        if not self.family_db:
            print(f"ℹ️ [Vision-Face] 当前底库为空，请将家人单张照片放入: {os.path.abspath(self.db_dir)}/")

    def match_face(self, normed_embedding: np.ndarray) -> Tuple[str, float]:
        """1:6 极速余弦比对"""
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
          - annotated_frame: 画好识别框的画面
          - matched_family: 若识别到门前自家人，返回 (name, score)，否则 None
          - suspicious_reason: 若触发可疑人员告警，返回告警原因字符串，否则 None
        """
        if self.app is None:
            return frame, None, None

        faces = self.app.get(frame)
        matched_family: Optional[Tuple[str, float]] = None
        has_stranger = False
        suspicious_reason: Optional[str] = None
        now = time.time()

        for face in faces:
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            name, score = self.match_face(face.normed_embedding)

            if name != "Stranger":
                # 只有当人脸尺寸大于阈值（说明走到门前）才激活开门认证
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
            cv2.putText(
                frame, tag, (bbox[0], max(25, bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

        # ======================================================================
        # 安防规则 1：陌生人長時間徘徊检测
        # ======================================================================
        if has_stranger and matched_family is None:
            if self.stranger_first_seen is None:
                self.stranger_first_seen = now
            else:
                loiter_sec = now - self.stranger_first_seen
                cv2.putText(
                    frame, f"Stranger Loitering: {loiter_sec:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
                if loiter_sec >= self.stranger_loiter_alert_sec and (now - self.last_alert_time > 30):
                    suspicious_reason = f"陌生人在门厅逗留超过 {loiter_sec:.1f} 秒"
                    self.last_alert_time = now
        else:
            self.stranger_first_seen = None

        # ======================================================================
        # 安防规则 2：涂鸦毫米波雷达显示贴门有人(<1.2m)，但画面完全看不到人脸（防蒙面/遮挡撬锁）
        # ======================================================================
        if len(faces) == 0 and tuya_distance_m is not None and tuya_distance_m < 1.2:
            if self.no_face_first_seen is None:
                self.no_face_first_seen = now
            else:
                occ_sec = now - self.no_face_first_seen
                if occ_sec >= self.no_face_occlusion_alert_sec and (now - self.last_alert_time > 30):
                    suspicious_reason = f"涂鸦传感器检测到门前 {tuya_distance_m}m 有人，但连续 {occ_sec:.1f}秒 未露脸（疑似遮挡或撬锁）"
                    self.last_alert_time = now
        else:
            self.no_face_first_seen = None

        return frame, matched_family, suspicious_reason
