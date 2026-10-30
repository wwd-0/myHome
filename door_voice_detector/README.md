# 方案 A：两级超轻量语音口令与声纹识别训练工程 (`door_voice_detector`)

本目录用于**离线在 GPU（NVIDIA CUDA / Apple MPS）上训练并导出**门禁语音识别所需的两个超轻量 `.onnx` 模型。线上 `cpp_service` 纯 C++ 原生加载推理，零 Python 运行时依赖。

## 架构说明（总体积仅 ~3.4 MB）
1. **第一级：口令识别与闲聊拒识网 (`BCResMDTCCommandNet`, ~0.75 MB)**
   - `2D Stem + BC-ResBlock (频域-时域广播残差) + 4层指数空洞 MDTC + 4段保序时序池化`
   - 识别 10 种开门语音口令（如 `芝麻开门`、`开门开门`、`请打开门` 等），类别 `0` 为 `UNKNOWN` 日常闲聊/环境音拒识通道（99% 非口令音频在此短路返回，不消耗声纹算力）。
   - 同时输出 `128` 维保序口令特征向量，支持免重训扩展口令声学模板。
2. **第二级：声纹特征提取网 (`CAMPlusLiteSpeakerNet`, ~2.65 MB)**
   - `2D FCM 频谱共振峰茎 + 4层 CAM-DTDNN (上下文感知掩码密集时延网) + 注意力均值标准差统计池化 (ASP)`
   - 训练阶段外挂 `AAM-Softmax`（语音版 ArcFace），推理阶段输出 `192` 维 L2 归一化特征向量，与最多 10 位家人的声纹底库进行余弦比对（优先同口令专属模板比对，全局均值模板兜底），天然拒识陌生人。

## 快速训练与导出流程

```bash
# 1. (可选) 生成多音色合成种子数据集，或将真实录音放入 door_voice_detector/dataset/
python door_voice_detector/tools/generate_seed_data.py

# 2. 训练 Stage-1 口令网 + Stage-2 声纹网，并自动导出 ONNX 至 ./audio/models/
python door_voice_detector/train.py
```

### 真实数据集目录格式（放入后直接运行 `train.py` 即可）
- `door_voice_detector/dataset/commands/<口令名>/*.wav`：10 种开门语音口令录音（16kHz 单声道 WAV）
- `door_voice_detector/dataset/negatives/*.wav`：日常闲聊、走廊环境噪音等负样本
- `door_voice_detector/dataset/speakers/<说话人ID>/*.wav`：不同说话人短语音样本（用于训练声纹区分能力；家人的门禁注册声纹无需放进训练集，直接在 Web 控制台或 `audio/db/voices/` 录入即可生效）
