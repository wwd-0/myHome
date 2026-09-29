import os
import wave
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None


class VoiceEngine:
    """
    音频模块：基于阿里达摩院 3D-Speaker (CAM++) 的轻量级声纹特征提取与 1:1 跨模态核验引擎
    配合“人脸识别”锁定的家人 ID，仅核对当前说话声音是否属于画面中的那位家人（天然防外人录音攻击）
    """

    def __init__(self, audio_cfg: dict):
        self.db_dir = audio_cfg.get("db_dir", "./audio/db/voices")
        self.model_path = audio_cfg.get("model_path", "./audio/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
        self.sim_threshold = float(audio_cfg.get("similarity_threshold", 0.60))
        self.sample_duration_sec = float(audio_cfg.get("sample_duration_sec", 2.0))
        self.vad_energy_threshold = float(audio_cfg.get("vad_energy_threshold", 0.015))

        # 家庭成员声纹特征向量底库: { "爸爸": np.ndarray(192,), ... }
        self.voice_db: Dict[str, np.ndarray] = {}
        self.extractor = None

        self._init_model()
        self.reload_voice_db()

    def _init_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if sherpa_onnx is None:
            print("⚠️ [Audio-Voice] 未安装 sherpa-onnx，请运行: pip install sherpa-onnx sounddevice")
            return

        if not os.path.exists(self.model_path):
            print(
                f"ℹ️ [Audio-Voice] 未找到 CAM++ ONNX 模型文件: {self.model_path}\n"
                f"   请运行一键下载命令:\n"
                f"   curl -L -o {self.model_path} "
                f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
            )
            return

        print("🎙️ [Audio-Voice] 正在加载 3D-Speaker CAM++ 中文声纹模型...")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=self.model_path,
            num_threads=2,
            provider="cpu",
        )
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    @staticmethod
    def _load_wav_16k(wav_path: str) -> Tuple[np.ndarray, int]:
        """读取 16kHz 单声道 16-bit PCM wav 文件并归一化到 [-1.0, 1.0]"""
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if num_channels > 1:
                samples = samples.reshape(-1, num_channels).mean(axis=1)
            return samples, sample_rate

    def extract_embedding(self, samples: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """从音频波形提取 L2 归一化声纹向量"""
        if self.extractor is None or len(samples) < sample_rate * 0.5:
            return None

        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        if not self.extractor.is_ready(stream):
            return None

        emb = np.array(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm

    def reload_voice_db(self):
        """
        扫描 ./audio/db/voices 目录，自动提取家里 6 个人的声纹入库
        支持 爸爸.wav 或 爸爸_1.wav, 爸爸_2.wav（多条语音求均值）
        """
        os.makedirs(self.db_dir, exist_ok=True)
        if self.extractor is None:
            return

        temp_embs: Dict[str, List[np.ndarray]] = {}
        for fname in sorted(os.listdir(self.db_dir)):
            if not fname.lower().endswith(".wav"):
                continue

            person_name = os.path.splitext(fname)[0].split("_")[0]
            wav_path = os.path.join(self.db_dir, fname)
            try:
                samples, sr = self._load_wav_16k(wav_path)
                emb = self.extract_embedding(samples, sr)
                if emb is not None:
                    temp_embs.setdefault(person_name, []).append(emb)
            except Exception as e:
                print(f"⚠️ [Audio-Voice] 读取语音 {fname} 失败: {e}")

        self.voice_db.clear()
        for person_name, emb_list in temp_embs.items():
            mean_emb = np.mean(emb_list, axis=0)
            self.voice_db[person_name] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            print(f"✅ [Audio-Voice] 已入库家人 [{person_name}] 的声纹特征 (融合了 {len(emb_list)} 条语音)")

        if not self.voice_db:
            print(f"ℹ️ [Audio-Voice] 当前声纹库为空，请将家人 16kHz 录音放入: {os.path.abspath(self.db_dir)}/")

    def verify_person_1to1(self, target_person: str, samples: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        【人脸 + 声纹】跨模态 1:1 强绑定核验：
        只比对当前采集到的语音是否属于摄像头前面站着的那位家人 (target_person)
        """
        if target_person not in self.voice_db:
            return False, 0.0

        # 检查音量能量（过滤环境底噪，只有真正开口说话才计算）
        rms_energy = float(np.sqrt(np.mean(np.square(samples))))
        if rms_energy < self.vad_energy_threshold:
            return False, 0.0

        live_emb = self.extract_embedding(samples, sample_rate)
        if live_emb is None:
            return False, 0.0

        target_emb = self.voice_db[target_person]
        score = float(np.dot(live_emb, target_emb))
        return score >= self.sim_threshold, score
