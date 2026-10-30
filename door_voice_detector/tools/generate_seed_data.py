"""
种子语音数据集构建与声学扩增工具 (generate_seed_data.py)
作用：
1. 支持直接读取真实录制的语音数据 (dataset/commands/<口令>/, dataset/negatives/, dataset/speakers/<人名>/)
2. 在尚未采集大规模真实录音前，利用系统多发音人中文语音合成 + 声道共振峰/基频物理变换 (24 种男/女/老/少音色) + 走廊混响/底噪，
   一键生成高质量基础训练集，使 Stage-1 口令网与 Stage-2 声纹网开箱即具备真实的中文音节辨识与声道区分能力。
"""

import os
import sys
import subprocess
import tempfile
import yaml
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.fbank import SAMPLE_RATE, save_wav_16k


NEGATIVE_PHRASES = [
    "门开门开",          # 逆序词强负样本，逼迫模型学习严格字序
    "关门关门",          # 近义混淆负样本
    "把门关好",
    "快递放门口吧",
    "外卖放在架子上",
    "今天天气真不错",
    "电梯怎么还没到",
    "钥匙带了没有",
    "晚上回家吃什么",
    "你好请问找哪位",
    "我先出门上班了",
    "物业过来抄水表",
]


def synthesize_tts_wav(text: str, voice: str, rate: int) -> np.ndarray:
    """调用 macOS 原生语音引擎合成单句中文波形并转为 16kHz float32"""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f_aiff:
        aiff_path = f_aiff.name
    wav_path = aiff_path + ".wav"
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-nostdin", "-loglevel", "quiet",
                "-i", aiff_path, "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "wav", wav_path,
            ],
            check=True,
        )
        from core.fbank import load_wav_16k
        return load_wav_16k(wav_path)
    finally:
        for p in (aiff_path, wav_path):
            if os.path.exists(p):
                os.remove(p)


def warp_speaker_timbre(
    audio: np.ndarray,
    pitch_semitones: float,
    tempo_ratio: float,
    eq_tilt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    通过重采样基频/共振峰偏移 + 频谱倾斜滤波 + 门厅微混响，构造具有稳定声道特征的独立虚拟说话人
    """
    # 1. 共振峰与基频联合缩放 (模拟不同声道长度：成年男性声道长偏低沉，儿童声道短偏明亮)
    pitch_factor = 2.0 ** (pitch_semitones / 12.0)
    combined_rate = pitch_factor * tempo_ratio
    new_len = max(800, int(round(len(audio) / combined_rate)))
    idx = np.linspace(0.0, len(audio) - 1, num=new_len, dtype=np.float32)
    li = np.floor(idx).astype(np.int32)
    ri = np.minimum(li + 1, len(audio) - 1)
    frac = idx - li
    y = audio[li] * (1.0 - frac) + audio[ri] * frac

    # 2. 频响倾斜 (一阶滤波模拟不同胸腔/鼻腔共鸣)
    if abs(eq_tilt) > 1e-3:
        out = np.empty_like(y)
        out[0] = y[0]
        out[1:] = y[1:] + eq_tilt * y[:-1]
        y = out

    # 3. 门厅短混响 (早期反射 18~35ms) + 轻微环境底噪
    delay = int(rng.integers(260, 560))
    decay = float(rng.uniform(0.06, 0.20))
    if len(y) > delay:
        rev = np.copy(y)
        rev[delay:] += decay * y[:-delay]
        y = rev

    noise_std = float(rng.uniform(0.001, 0.008))
    y = y + rng.normal(0.0, noise_std, size=len(y)).astype(np.float32)

    peak = float(np.max(np.abs(y)))
    if peak > 1e-4:
        y = y * (float(rng.uniform(0.55, 0.92)) / peak)
    return y.astype(np.float32)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root_dir, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    commands = cfg["commands"]["list"]
    ds_dir = os.path.join(root_dir, "dataset")
    cmd_root = os.path.join(ds_dir, "commands")
    neg_root = os.path.join(ds_dir, "negatives")
    spk_root = os.path.join(ds_dir, "speakers")
    os.makedirs(cmd_root, exist_ok=True)
    os.makedirs(neg_root, exist_ok=True)
    os.makedirs(spk_root, exist_ok=True)

    voices = ["Tingting", "Meijia", "Sinji"]
    rates = [175, 200]
    rng = np.random.default_rng(42)

    print("🎙️ [1/3] 正在合成基准中文口令与闲聊负样本母带...")
    base_cmd_audio = {}
    for cmd in commands:
        base_cmd_audio[cmd] = []
        for v in voices:
            for r in rates:
                base_cmd_audio[cmd].append((v, synthesize_tts_wav(cmd, v, r)))

    base_neg_audio = []
    for phrase in NEGATIVE_PHRASES:
        for v in voices:
            base_neg_audio.append((v, synthesize_tts_wav(phrase, v, 195)))

    # 构建 24 位稳定音色配置的说话人 (覆盖低沉男声、标准男声、女声、长辈、儿童声道特征)
    speaker_profiles = []
    idx = 0
    for v in voices:
        for pitch in [-5.5, -3.5, -1.8, 0.0, 1.6, 3.2, 4.8, 6.2]:
            eq_tilt = float(np.clip(-pitch * 0.06 + rng.uniform(-0.08, 0.08), -0.45, 0.45))
            speaker_profiles.append((f"spk_{idx:02d}", v, pitch, eq_tilt))
            idx += 1

    print(f"🧬 [2/3] 正在扩增 {len(commands)} 种口令与 {len(speaker_profiles)} 位说话人声纹训练集...")
    for spk_id, spk_voice, pitch, eq_tilt in speaker_profiles:
        s_dir = os.path.join(spk_root, spk_id)
        os.makedirs(s_dir, exist_ok=True)

        # 每位说话人对所有口令及若干日常语句生成带微小语速抖动的样本
        for c_idx, cmd in enumerate(commands):
            c_dir = os.path.join(cmd_root, f"{c_idx + 1:02d}_{cmd}")
            os.makedirs(c_dir, exist_ok=True)

            # 选取该说话人基底发音人的母带
            candidates = [wav for (vv, wav) in base_cmd_audio[cmd] if vv == spk_voice]
            for rep in range(2):
                src_wav = candidates[rep % len(candidates)]
                jitter_pitch = pitch + float(rng.uniform(-0.25, 0.25))
                tempo = float(rng.uniform(0.92, 1.08))
                warped = warp_speaker_timbre(src_wav, jitter_pitch, tempo, eq_tilt, rng)

                cmd_file = os.path.join(c_dir, f"{spk_id}_r{rep}.wav")
                spk_file = os.path.join(s_dir, f"cmd{c_idx:02d}_r{rep}.wav")
                save_wav_16k(cmd_file, warped)
                save_wav_16k(spk_file, warped)

    print("🛡️ [3/3] 正在生成闲聊与走廊环境噪声负样本 (UNKNOWN 类)...")
    neg_count = 0
    for phrase_idx, (vv, wav) in enumerate(base_neg_audio):
        for pitch in [-4.5, -1.5, 0.0, 2.5, 5.0]:
            tempo = float(rng.uniform(0.90, 1.12))
            eq = float(rng.uniform(-0.3, 0.3))
            warped = warp_speaker_timbre(wav, pitch, tempo, eq, rng)
            save_wav_16k(os.path.join(neg_root, f"chatter_{neg_count:04d}.wav"), warped)
            neg_count += 1

    # 加入纯走廊突发脉冲/敲门/风噪负样本 (过 VAD 但非人声口令)
    for n_idx in range(60):
        t = np.linspace(0, 1.6, int(SAMPLE_RATE * 1.6), dtype=np.float32)
        env = np.exp(-((t - float(rng.uniform(0.4, 1.1))) ** 2) / float(rng.uniform(0.02, 0.12)))
        noise = rng.normal(0.0, 0.35, size=len(t)).astype(np.float32) * env
        save_wav_16k(os.path.join(neg_root, f"ambient_{n_idx:04d}.wav"), noise)

    print(f"✅ 数据集准备完成！位于: {ds_dir}")


if __name__ == "__main__":
    main()
