"""
语音前端特征提取模块 (fbank.py)
严格与 C++ 端 CppVoiceEngine 保持逐比特一致的数学实现：
1. 16kHz 单声道归一化与端点能量 VAD 截取（居中放置于 1.60s = 25600 采样点窗口）
2. 0.97 预加重 + 400 点汉明窗 (帧移 160 点, 512 点 FFT) -> 精确 T = 160 帧
3. HTK 梅尔尺度 80 维三角滤波器组 (20Hz ~ 7600Hz) + Log 压缩 + 句级频带均值归一化 (CMN)
"""

import math
import wave
from typing import Tuple, Optional
import numpy as np


SAMPLE_RATE = 16000
TARGET_SAMPLES = 25600  # 1.60 秒
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_FRAMES = 160
N_MELS = 80
F_MIN = 20.0
F_MAX = 7600.0
PRE_EMPHASIS = 0.97


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """构建标准 HTK 三角梅尔滤波器组矩阵: shape = (80, 257)"""
    num_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.array([mel_to_hz(m) for m in mel_points], dtype=np.float64)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)

    fb = np.zeros((n_mels, num_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, num_bins)):
            fb[m - 1, k] = (k - left) / float(center - left)
        for k in range(center, min(right, num_bins)):
            fb[m - 1, k] = (right - k) / float(right - center)
    return fb


_MEL_FB = build_mel_filterbank()
_HAMMING = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / (WIN_LENGTH - 1))).astype(np.float32)


def load_wav_16k(path: str) -> np.ndarray:
    """读取 8/16/32-bit PCM WAV 文件并线性重采样为 16kHz 单声道 float32 [-1, 1]"""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if framerate != SAMPLE_RATE and len(audio) > 1:
        new_len = max(1, int(round(len(audio) * SAMPLE_RATE / float(framerate))))
        old_idx = np.linspace(0.0, len(audio) - 1, num=new_len, dtype=np.float32)
        left_idx = np.floor(old_idx).astype(np.int32)
        right_idx = np.minimum(left_idx + 1, len(audio) - 1)
        frac = old_idx - left_idx
        audio = audio[left_idx] * (1.0 - frac) + audio[right_idx] * frac

    return audio.astype(np.float32)


def save_wav_16k(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def vad_trim_and_center(
    audio: np.ndarray,
    vad_rms_threshold: float = 0.012,
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    端点能量检测与定长居中对齐 (25600 点 = 1.60s)
    返回: (aligned_audio 或 None, active_duration_sec, peak_rms)
    """
    if audio is None or len(audio) < HOP_LENGTH * 15:
        return None, 0.0, 0.0

    # 去直流分量
    audio = audio - float(np.mean(audio))

    num_frames = len(audio) // HOP_LENGTH
    if num_frames < 15:
        return None, 0.0, 0.0

    frames = audio[: num_frames * HOP_LENGTH].reshape(num_frames, HOP_LENGTH)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peak_rms = float(np.max(rms))
    if peak_rms < vad_rms_threshold:
        return None, 0.0, peak_rms

    dyn_thresh = max(vad_rms_threshold * 0.6, peak_rms * 0.10)
    active_idx = np.where(rms >= dyn_thresh)[0]
    if len(active_idx) < 18:  # 有效人声不足 0.18 秒视为瞬态敲击噪音
        return None, len(active_idx) * 0.01, peak_rms

    # 前后保留 12 帧 (120ms) 保护声母爆破音与韵尾鼻音
    start_f = max(0, int(active_idx[0]) - 12)
    end_f = min(num_frames, int(active_idx[-1]) + 13)
    active_sec = (end_f - start_f) * 0.01

    seg = audio[start_f * HOP_LENGTH : end_f * HOP_LENGTH]
    peak_abs = float(np.max(np.abs(seg)))
    if peak_abs > 1e-4:
        seg = seg * (0.90 / peak_abs)

    canvas = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    if len(seg) >= TARGET_SAMPLES:
        offset = (len(seg) - TARGET_SAMPLES) // 2
        canvas[:] = seg[offset : offset + TARGET_SAMPLES]
    else:
        offset = (TARGET_SAMPLES - len(seg)) // 2
        canvas[offset : offset + len(seg)] = seg

    return canvas, active_sec, peak_rms


def compute_fbank_80x160(aligned_audio: np.ndarray) -> np.ndarray:
    """
    输入定长 25600 点波形，输出 shape = (1, 80, 160) 的 CMN 归一化 Log-Mel 频谱
    """
    assert len(aligned_audio) == TARGET_SAMPLES
    # 1. 预加重
    emph = np.empty_like(aligned_audio)
    emph[0] = aligned_audio[0]
    emph[1:] = aligned_audio[1:] - PRE_EMPHASIS * aligned_audio[:-1]

    # 尾部补 240 个零，使第 159 帧完整覆盖 400 点窗口
    padded = np.pad(emph, (0, WIN_LENGTH - HOP_LENGTH), mode="constant")

    # 2. 分帧加汉明窗 + 512 点实数 FFT
    frames = np.lib.stride_tricks.sliding_window_view(padded, WIN_LENGTH)[::HOP_LENGTH]
    frames = frames[:N_FRAMES] * _HAMMING[None, :]  # (160, 400)

    spec = np.fft.rfft(frames, n=N_FFT, axis=1)  # (160, 257)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # 3. 80 维梅尔能量 + 对数压缩
    mel = np.matmul(_MEL_FB, power.T)  # (80, 160)
    log_mel = np.log(np.maximum(mel, 1e-10))

    # 4. 句级频带均值归一化 (CMN)，消除不同麦克风频响曲线偏移
    mean_per_bin = np.mean(log_mel, axis=1, keepdims=True)
    norm_mel = (log_mel - mean_per_bin).astype(np.float32)
    return norm_mel[None, :, :]  # (1, 80, 160)
