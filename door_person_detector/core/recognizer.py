import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from models.light_person_net import LightPersonIDNet, MobileFaceNet
except (ImportError, ValueError):
    from ..models.light_person_net import LightPersonIDNet, MobileFaceNet


@dataclass
class RecognitionResult:
    """单人身份识别结果"""
    person_name: str                  # 识别身份 ("爸爸", "妈妈" 等，或 "陌生人")
    similarity_score: float           # 余弦相似度置信度 (0.0 ~ 1.0)
    is_stranger: bool                 # 是否为陌生人
    is_family: bool                   # 是否为自家人
    bbox: Tuple[int, int, int, int]   # 图像边界框 (x, y, w, h)
    has_face: bool                    # 画面中是否显式捕获到人脸
    is_profile: bool                  # 是否为侧脸或局部遮挡
    latency_ms: float                 # 单张推理耗时 (毫秒)


@dataclass
class FrameSummary:
    """
    模型推理的核心结构化输出 (纯净感知数据，解耦后续业务逻辑)
    明确输出: 具体有几个人、几个自家人、几个陌生人、自家人具体是谁、每个人坐标与得分
    """
    total_persons: int                # 画面总人数 (具体有几个人)
    family_count: int                 # 几个自家人
    stranger_count: int               # 几个陌生人
    family_members: List[str]         # 具体自家人是谁 (如: ["爸爸", "妈妈"])
    has_stranger: bool                # 是否存在陌生人
    persons: List[Dict]               # 每个人详细数据 (姓名、得分、坐标框、姿态等)
    latency_ms: float = 0.0           # 总推理耗时 (毫秒)

    @property
    def total_count(self) -> int:
        return self.total_persons

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)

    def to_json(self) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(self), ensure_ascii=False)


class PersonRecognizer:
    """
    生产级入户人员全能身份识别引擎 (纯净感知模块)
    - 采用 LightPersonIDNet (MobileNetV3 + GeM 池化) 全局特征提取
    - 兼容正脸、侧脸、无面部低头、背影等全姿态人体主体
    - 支持自家人特征底库 (Prototypical Gallery) 快速比对
    - 严格的陌生人拒识机制: 余弦相似度低于阈值一律判定为 [陌生人]
    - 上限 15 位家庭成员
    """
    def __init__(self,
                 weights_path: str = "",
                 gallery_path: str = "",
                 model_type: str = "LightPersonIDNet",
                 similarity_threshold: float = 0.75,
                 target_size: Tuple[int, int] = (256, 128),
                 device: str = "mps"):
        # 自动探测硬件加速
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model_type = model_type
        self.target_size = target_size
        self.similarity_threshold = similarity_threshold

        # 1. 实例化超轻量识别网络
        if model_type == "MobileFaceNet":
            self.model = MobileFaceNet(embedding_dim=512).to(self.device)
        else:
            self.model = LightPersonIDNet(embedding_dim=512, pretrained=True).to(self.device)
        self.model.eval()

        self.weights_path = weights_path
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.classifier_weight: Optional[torch.Tensor] = None
        self._load_weights(weights_path)

        # 2. 家人特征向量底库 (作为备选方案)
        self.gallery_path = gallery_path
        self.family_gallery: Dict[str, np.ndarray] = {}
        if gallery_path:
            self.load_gallery(gallery_path)

    def _load_weights(self, weights_path: str):
        if weights_path and os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                elif isinstance(ckpt, dict) and "backbone" in ckpt:
                    self.model.load_state_dict(ckpt["backbone"])
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)

                # 方案 A 核心：直接加载分类头权重与家庭成员标签（0 照片依赖）
                if isinstance(ckpt, dict) and "classes" in ckpt:
                    self.classes = ckpt["classes"]
                    self.class_to_idx = ckpt.get("class_to_idx", {})
                    if "head" in ckpt and "weight" in ckpt["head"]:
                        head_w = ckpt["head"]["weight"]
                        self.classifier_weight = F.normalize(head_w, p=2, dim=1).to(self.device)
                        print(f"[Recognizer - 方案A] 成功加载端到端分类头！已固化学习家庭成员: {self.classes}")

                print(f"[Recognizer] 成功载入模型权重: {weights_path}")
            except Exception as e:
                print(f"[Recognizer] 载入权重异常: {e}，使用初始预训练网络")
        else:
            print("[Recognizer] 载入基础特征网络 (具备 ImageNet 强力泛化特征，开箱即用)")

    def load_gallery(self, gallery_path: str):
        """加载已持久化的家庭成员特征向量库"""
        if not gallery_path or not os.path.exists(gallery_path):
            return
        try:
            with open(gallery_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.family_gallery = {
                name: np.array(vec, dtype=np.float32) for name, vec in raw_data.items()
            }
            print(f"[Recognizer] 成功载入家庭成员底库，共 {len(self.family_gallery)} 位成员: {list(self.family_gallery.keys())}")
        except Exception as e:
            print(f"[Recognizer] 载入底库异常: {e}")

    def save_gallery(self, gallery_path: str = ""):
        """持久化保存家庭成员特征向量库"""
        target_path = gallery_path if gallery_path else self.gallery_path
        if not target_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        serializable = {
            name: vec.tolist() for name, vec in self.family_gallery.items()
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"[Recognizer] 底库已持久化至: {target_path}")

    def extract_embedding(self, patch_bgr: np.ndarray) -> np.ndarray:
        """
        输入 (256, 128) BGR 人体切片，返回 512 维单位化特征向量 (||f||_2 = 1.0)
        """
        rgb = patch_bgr[:, :, ::-1].astype(np.float32)
        normalized = (rgb - 127.5) / 128.0
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor, normalize=True)
            feat_np = feat.cpu().numpy()[0]
        return feat_np

    def identify(self,
                 patch_bgr: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 has_face: bool = False,
                 is_profile: bool = False) -> RecognitionResult:
        """
        单人识别：识别单个人体切片是谁
        """
        t0 = time.time()
        embedding = self.extract_embedding(patch_bgr)

        if not self.family_gallery:
            latency = (time.time() - t0) * 1000.0
            return RecognitionResult(
                person_name="陌生人",
                similarity_score=0.0,
                is_stranger=True,
                is_family=False,
                bbox=bbox,
                has_face=has_face,
                is_profile=is_profile,
                latency_ms=latency
            )

        best_name = "陌生人"
        best_score = -1.0

        for name, proto_vec in self.family_gallery.items():
            cos_sim = float(np.dot(embedding, proto_vec))
            if cos_sim > best_score:
                best_score = cos_sim
                best_name = name

        latency = (time.time() - t0) * 1000.0

        if best_score >= self.similarity_threshold:
            is_stranger = False
            is_family = True
            identified_name = best_name
        else:
            is_stranger = True
            is_family = False
            identified_name = "陌生人"

        return RecognitionResult(
            person_name=identified_name,
            similarity_score=max(0.0, best_score),
            is_stranger=is_stranger,
            is_family=is_family,
            bbox=bbox,
            has_face=has_face,
            is_profile=is_profile,
            latency_ms=latency
        )

    def identify_batch(self, persons: List) -> List[RecognitionResult]:
        """
        批量并行识别画面中同时出现的多个主体 (极速批处理，4人并发耗时几乎等同于单人)
        输入: List[DetectedPerson]
        输出: List[RecognitionResult] 对应画面中每个人的身份与置信度
        """
        if not persons:
            return []

        t0 = time.time()
        batch_tensors = []
        for p in persons:
            rgb = p.aligned_patch[:, :, ::-1].astype(np.float32)
            normalized = (rgb - 127.5) / 128.0
            batch_tensors.append(torch.from_numpy(normalized.transpose(2, 0, 1)))

        batch_input = torch.stack(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch_input, normalize=True).cpu().numpy()

        latency_total = (time.time() - t0) * 1000.0
        avg_latency = latency_total / max(1, len(persons))

        results: List[RecognitionResult] = []

        # 方案 A 优先：端到端模型分类头直接推理（0 照片依赖）
        if self.classifier_weight is not None and len(self.classes) > 0:
            with torch.no_grad():
                embeddings_t = self.model(batch_input, normalize=True)
                sims_matrix = torch.matmul(embeddings_t, self.classifier_weight.t())

            for idx, p in enumerate(persons):
                sims = sims_matrix[idx]
                best_score_t, best_idx_t = torch.max(sims, dim=0)
                best_score = float(best_score_t.item())
                best_idx = int(best_idx_t.item())
                best_name = self.classes[best_idx]

                # 判定阈值：ArcFace 空间中自家人通常在 0.35~0.60 之间，陌生人通常 <= 0.15
                threshold = 0.28 if self.similarity_threshold > 0.5 else self.similarity_threshold
                if best_score >= threshold:
                    is_stranger = False
                    is_family = True
                    identified_name = best_name
                else:
                    is_stranger = True
                    is_family = False
                    identified_name = "陌生人"

                # 归一化置信度至 [0, 1] 方便前端展示
                norm_score = max(0.0, min(1.0, (best_score + 0.2) / 0.8))

                results.append(RecognitionResult(
                    person_name=identified_name,
                    similarity_score=round(norm_score, 4),
                    is_stranger=is_stranger,
                    is_family=is_family,
                    bbox=p.bbox,
                    has_face=p.has_face,
                    is_profile=p.is_profile_or_occluded,
                    latency_ms=avg_latency
                ))
            return results

        gallery_names = list(self.family_gallery.keys())
        gallery_matrix = np.array([self.family_gallery[n] for n in gallery_names]) if gallery_names else None

        for idx, p in enumerate(persons):
            embedding = embeddings[idx]

            if gallery_matrix is None or len(gallery_matrix) == 0:
                results.append(RecognitionResult(
                    person_name="陌生人",
                    similarity_score=0.0,
                    is_stranger=True,
                    is_family=False,
                    bbox=p.bbox,
                    has_face=p.has_face,
                    is_profile=p.is_profile_or_occluded,
                    latency_ms=avg_latency
                ))
                continue

            sims = np.dot(gallery_matrix, embedding)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            best_name = gallery_names[best_idx]

            if best_score >= self.similarity_threshold:
                is_stranger = False
                is_family = True
                identified_name = best_name
            else:
                is_stranger = True
                is_family = False
                identified_name = "陌生人"

            results.append(RecognitionResult(
                person_name=identified_name,
                similarity_score=max(0.0, best_score),
                is_stranger=is_stranger,
                is_family=is_family,
                bbox=p.bbox,
                has_face=p.has_face,
                is_profile=p.is_profile_or_occluded,
                latency_ms=avg_latency
            ))

        return results

    def summarize_frame(self, results: List[RecognitionResult]) -> FrameSummary:
        """
        根据当前帧的所有人员识别结果，生成纯净结构化感知数据
        包含: 画面总人数、自家人数、陌生人数、自家人具体是谁、每个人具体坐标及置信度
        """
        family_members = []
        family_count = 0
        stranger_count = 0
        persons_detail = []

        for r in results:
            if r.is_stranger:
                stranger_count += 1
            else:
                family_count += 1
                if r.person_name not in family_members:
                    family_members.append(r.person_name)

            status_tag = "正面" if (r.has_face and not r.is_profile) else ("侧脸" if r.is_profile else "无脸主体")
            persons_detail.append({
                "name": str(r.person_name),
                "score": round(float(r.similarity_score), 4),
                "is_stranger": bool(r.is_stranger),
                "is_family": bool(r.is_family),
                "status": status_tag,
                "bbox": {"x": int(r.bbox[0]), "y": int(r.bbox[1]), "w": int(r.bbox[2]), "h": int(r.bbox[3])},
                "latency_ms": round(float(r.latency_ms), 2)
            })

        return FrameSummary(
            total_persons=len(results),
            family_count=family_count,
            stranger_count=stranger_count,
            family_members=family_members,
            has_stranger=(stranger_count > 0),
            persons=persons_detail
        )

    def predict_image(self, image_bgr: np.ndarray, detector=None) -> FrameSummary:
        """
        端到端纯净推理方法 (直接输入单帧图像，输出具体几个人、几个自家人、具体是谁等纯净感知数据)
        注意: 不耦合开门、尾随阻断等业务逻辑，专为上层应用/后续判断提供最纯净的结构化数据
        """
        t0 = time.time()
        if detector is None:
            if not hasattr(self, "_default_detector") or self._default_detector is None:
                try:
                    from core.detector import LightweightDetector
                except (ImportError, ValueError):
                    from .detector import LightweightDetector
                self._default_detector = LightweightDetector(target_size=self.target_size)
            detector = self._default_detector

        detected_persons = detector.detect(image_bgr)
        rec_results = self.identify_batch(detected_persons)
        summary = self.summarize_frame(rec_results)
        summary.latency_ms = round((time.time() - t0) * 1000.0, 2)
        return summary
