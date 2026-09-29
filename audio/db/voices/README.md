# 家庭成员声纹底库目录 (`audio/db/voices/`)

将家里最多 6 位成员的短语音（2~4 秒，格式为 `16kHz 单声道 .wav`）放入本目录，**无需训练**即可完成声纹注册！

## 命名规则（必须与 `vision/db/faces/` 中的人名一致，以便跨模态 1:1 绑定）
- `爸爸.wav` （或 `爸爸_1.wav`, `爸爸_2.wav`）
- `妈妈.wav`
- `爷爷.wav`

## 如何在 Mac 上一键下载 CAM++ 预训练模型（仅 2.6MB）
在项目根目录运行：
```bash
mkdir -p audio/models
curl -L -o audio/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
```
