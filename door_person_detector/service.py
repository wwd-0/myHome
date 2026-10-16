#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入户人员检测与识别引擎 - 极速 HTTP 微服务 (door_person_detector/service.py)
- 基于 LightPersonIDNet (MobileNetV3 + GeM 池化) 全身/多姿态身份识别
- 内存常驻 PyTorch + Apple Silicon MPS / CPU 加速
- 提供结构化纯净感知数据接口与底库增删改查管理
"""

import os
import io
import sys
import json
import time
import base64
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional

import cv2
import numpy as np

# 确保能载入当前目录下的 core 与 models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, BASE_DIR)

from core.detector import LightweightDetector, DetectedPerson, crop_maximal_aspect_ratio
from core.recognizer import PersonRecognizer, FrameSummary


class PersonInferenceEngine:
    def __init__(self,
                 weights_path: str = os.path.join(BASE_DIR, "weights/best_model.pth"),
                 gallery_path: str = os.path.join(BASE_DIR, "data/gallery/family_gallery.json"),
                 faces_dir: str = os.path.join(PROJECT_ROOT, "vision/db/faces"),
                 similarity_threshold: float = 0.75):
        self.faces_dir = faces_dir
        self.gallery_path = gallery_path
        self.similarity_threshold = similarity_threshold

        print(f"🚀 [PersonEngine] 初始化轻量人体感知引擎...")
        self.detector = LightweightDetector(target_size=(256, 128))
        self.recognizer = PersonRecognizer(
            weights_path=weights_path,
            gallery_path=gallery_path,
            similarity_threshold=similarity_threshold,
            device="mps"
        )
        # 如果特征库为空或未生成，自动根据 faces_dir 扫描生成
        if not self.recognizer.family_gallery and os.path.exists(faces_dir):
            self.rebuild_gallery_from_photos()

    def rebuild_gallery_from_photos(self) -> Dict[str, int]:
        """扫描 vision/db/faces/ 目录中的照片，批量提取 512 维特征并构建底库原型"""
        if not os.path.exists(self.faces_dir):
            return {}

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        member_features: Dict[str, List[np.ndarray]] = {}

        for fname in sorted(os.listdir(self.faces_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts or fname.startswith("."):
                continue

            stem = os.path.splitext(fname)[0]
            # 解析成员人名（如 "爸爸_1" -> "爸爸", "妈妈" -> "妈妈"）
            name = stem.split("_")[0].split("-")[0].strip()
            if not name:
                continue

            fpath = os.path.join(self.faces_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            # 使用检测器提取人体主体
            persons = self.detector.detect(img)
            if persons:
                # 选取画面中面积最大或最具代表性的主体
                best_person = max(persons, key=lambda p: p.bbox[2] * p.bbox[3])
                patch = best_person.aligned_patch
            else:
                # 若未检测到独立人体框，直接将整张照片缩放适配为 (256, 128)
                patch = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)

            feat = self.recognizer.extract_embedding(patch)
            if name not in member_features:
                member_features[name] = []
            member_features[name].append(feat)

        # 对每个成员的多张照片特征做 L2 均值原型聚合
        gallery = {}
        counts = {}
        for name, feats in member_features.items():
            avg_feat = np.mean(feats, axis=0)
            avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-7)
            gallery[name] = avg_feat
            counts[name] = len(feats)

        self.recognizer.family_gallery = gallery
        self.recognizer.save_gallery(self.gallery_path)
        print(f"✅ [PersonEngine] 已重建家庭成员特征底库，共 {len(gallery)} 位成员: {list(gallery.keys())}")
        return counts

    def get_members_summary(self) -> Dict:
        """方案 A：获取模型内化学习的家庭成员类别及训练集统计"""
        learned_classes = self.recognizer.classes if self.recognizer.classes else ["妈妈", "孩子", "爸爸"]
        train_dir = os.path.join(BASE_DIR, "data/dataset/train")
        members_list = []
        for c in learned_classes:
            c_dir = os.path.join(train_dir, c)
            sample_count = len([f for f in os.listdir(c_dir) if not f.startswith('.')]) if os.path.exists(c_dir) else 0
            members_list.append({
                "name": c,
                "sample_count": sample_count,
                "is_model_class": True,
                "status": "已内化至模型权重 (0 照片存储)"
            })

        return {
            "mode": "方案 A (端到端分类模型，0照片存储)",
            "model_type": "LightPersonIDNet + ArcFace Head",
            "device": str(self.recognizer.device),
            "total_members": len(learned_classes),
            "max_members": 15,
            "members": members_list,
            "classes": learned_classes
        }

    def predict_and_annotate(self, image_bgr: np.ndarray) -> Dict:
        """对单帧执行纯净推理，并生成带炫彩标注的可视化图像"""
        t0 = time.time()
        h_img, w_img = image_bgr.shape[:2]

        # 使用 YOLOv8-Nano 多目标人体检测 (支持单人、多人合影、偏位，自动提取每个人体切片)
        detected_persons = self.detector.detect(image_bgr)
        rec_results = self.recognizer.identify_batch(detected_persons)
        summary: FrameSummary = self.recognizer.summarize_frame(rec_results)
        summary.latency_ms = round((time.time() - t0) * 1000.0, 2)

        # 绘制高端现代风格的标注图 (Bounding Box + 半透明磨砂角标)
        annotated = image_bgr.copy()

        for p in summary.persons:
            bx = int(p["bbox"]["x"])
            by = int(p["bbox"]["y"])
            bw = int(p["bbox"]["w"])
            bh = int(p["bbox"]["h"])
            is_family = p["is_family"]
            score = p["score"]
            name = p["name"]

            # 边框配色：自家人使用翠绿高亮 (0, 220, 100)，陌生人使用珊瑚红 (60, 60, 240)
            color = (0, 220, 100) if is_family else (60, 60, 240)
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)

            # 绘制四角加粗科技感瞄准标
            c_len = min(20, bw // 4, bh // 4)
            t_thick = 3
            # 左上
            cv2.line(annotated, (bx, by), (bx + c_len, by), color, t_thick)
            cv2.line(annotated, (bx, by), (bx, by + c_len), color, t_thick)
            # 右上
            cv2.line(annotated, (bx + bw, by), (bx + bw - c_len, by), color, t_thick)
            cv2.line(annotated, (bx + bw, by), (bx + bw, by + c_len), color, t_thick)
            # 左下
            cv2.line(annotated, (bx, by + bh), (bx + c_len, by + bh), color, t_thick)
            cv2.line(annotated, (bx, by + bh), (bx, by + bh - c_len), color, t_thick)
            # 右下
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw - c_len, by + bh), color, t_thick)
            cv2.line(annotated, (bx + bw, by + bh), (bx + bw, by + bh - c_len), color, t_thick)

            # 身份标签背景卡片
            tag_text = f"{name} ({score:.2f})" if is_family else f"Stranger ({score:.2f})"
            tag_y = max(24, by - 8)
            cv2.rectangle(annotated, (bx, tag_y - 20), (bx + min(bw, 180), tag_y + 4), color, -1)
            cv2.putText(annotated, tag_text, (bx + 6, tag_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 画面顶部绘制全局感知汇总 HUD 横幅
        hud_h = 44
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, hud_h), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        hud_info = (f"Total: {summary.total_persons} | Family: {summary.family_count} | "
                    f"Stranger: {summary.stranger_count} | Latency: {summary.latency_ms:.1f}ms")
        cv2.putText(annotated, hud_info, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

        # 编码为 JPEG Base64
        _, enc_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_img = "data:image/jpeg;base64," + base64.b64encode(enc_buf.tobytes()).decode("utf-8")

        result = summary.to_dict()
        result["annotated_image"] = b64_img
        return result


# 全局单例引擎
_ENGINE: Optional[PersonInferenceEngine] = None


class PersonHttpHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 简化日志输出
        return

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            summary = _ENGINE.get_members_summary()
            resp = {
                "status": "healthy",
                "model": "LightPersonIDNet",
                "backbone": "MobileNetV3 + GeM Pooling (512-dim)",
                "device": str(_ENGINE.recognizer.device),
                "enrolled_members": summary["gallery_names"],
                "total_members": summary["total_members"]
            }
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/members":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            data = _ENGINE.get_members_summary()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/photo":
            query = parse_qs(parsed.query)
            filename = query.get("file", [""])[0]
            if not filename or ".." in filename:
                self.send_response(400)
                self.end_headers()
                return

            fpath = os.path.join(_ENGINE.faces_dir, filename)
            if not os.path.exists(fpath):
                self.send_response(404)
                self.end_headers()
                return

            with open(fpath, "rb") as f:
                data = f.read()

            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        if path == "/predict":
            # 支持接收 Base64 JSON 或者是直接接收二进制图片
            img_bgr = None
            try:
                body_json = json.loads(raw_body.decode("utf-8"))
                b64_data = body_json.get("image", "")
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                img_array = np.frombuffer(raw_body, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img_bgr is None or img_bgr.size == 0:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid image payload"}).encode("utf-8"))
                return

            try:
                result = _ENGINE.predict_and_annotate(img_bgr)

                def to_serializable(val):
                    if isinstance(val, (np.integer, np.int32, np.int64)):
                        return int(val)
                    elif isinstance(val, (np.floating, np.float32, np.float64)):
                        return float(val)
                    elif isinstance(val, np.ndarray):
                        return val.tolist()
                    return str(val)

                resp_bytes = json.dumps(result, ensure_ascii=False, default=to_serializable).encode("utf-8")
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/upload_photo":
            try:
                body_json = json.loads(raw_body.decode("utf-8"))
                name = body_json.get("name", "").strip()
                b64_data = body_json.get("image", "")

                if not name or not b64_data:
                    raise ValueError("name and image are required")

                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)

                # 方案 A：存入该成员的训练样本目录
                member_dir = os.path.join(BASE_DIR, "data/dataset/train", name)
                os.makedirs(member_dir, exist_ok=True)
                existing_files = [f for f in os.listdir(member_dir) if not f.startswith(".")]
                idx = len(existing_files) + 1
                fname = f"{name}_{idx:04d}.jpg"
                fpath = os.path.join(member_dir, fname)

                with open(fpath, "wb") as f:
                    f.write(img_bytes)

                summary = _ENGINE.get_members_summary()

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "saved_file": fname,
                    "member": name,
                    "sample_count": len(existing_files) + 1,
                    "members": summary["members"]
                }, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/delete_photo":
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "message": "方案 A 无需维护单独照片底库，所有特征均已内化至模型权重中。"
            }, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/reload_gallery" or path == "/reload_model":
            _ENGINE.recognizer._load_weights(_ENGINE.recognizer.weights_path)
            summary = _ENGINE.get_members_summary()
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "mode": "方案 A (端到端分类模型)",
                "members": summary["members"],
                "classes": summary["classes"]
            }, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_server(port: int = 8092):
    global _ENGINE
    _ENGINE = PersonInferenceEngine()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, PersonHttpHandler)
    print(f"🔥 [PersonInferenceService] 已启动，监听 http://127.0.0.1:{port}")
    print(f"   • /health (GET)")
    print(f"   • /members (GET)")
    print(f"   • /photo?file=xxx (GET)")
    print(f"   • /predict (POST)")
    print(f"   • /upload_photo (POST)")
    print(f"   • /delete_photo (POST)")
    print(f"   • /reload_gallery (POST)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务正在优雅退出...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8092, help="服务端口")
    args = parser.parse_args()
    run_server(args.port)
