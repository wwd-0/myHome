"""
方案 A - 第一级：超轻量语音口令识别与非口令拒识网络 (BCResMDTCCommandNet)
结构设计：
1. 2D 频谱卷积茎 (Conv2d Stem) 提取初级共振峰时频局部特征: [B, 1, 80, 160] -> [B, 24, 40, 160]
2. BC-ResBlock (Broadcasted Residual Blocks, Qualcomm KWS 架构)：
   - 频域深度卷积压缩频率维，时域 1D 空洞深度卷积提取音节过渡，再将 1D 时序特征对齐回加至 2D 特征
3. MDTC (Multi-Scale Dilated Temporal Convolution) 多尺度空洞时序残差塔：
   - 空洞率 dilation = 1, 2, 4, 8 级联覆盖完整 1.6 秒语音上下文: [B, 96, 1, 160]
4. 4 段保序时序池化头 (4-Chunk Order-Preserving Pooling)：
   - 将 160 帧等分为 [前、中前、中后、末] 4 个有序时间段分别聚合，严格区分字序（如“开门” vs “门开”）
   - 输出 128 维 L2 归一化口令嵌入向量 `cmd_embedding` + (1 + K) 路余弦打分 `logits`
     （索引 0 为 UNKNOWN 日常闲聊/噪声拒识通道，索引 1..K 对应 K 种开门语音口令）
全 4D NCHW 标准静态算子，100% 兼容 C++ OpenCV DNN 原生推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BroadcastedResBlock(nn.Module):
    """
    BC-ResNet 核心广播残差模块 (采用最近邻上采样显式对齐张量维度，规避推理引擎隐式广播缺陷)
    """
    def __init__(self, in_c: int, out_c: int, freq_stride: int = 2, dilation: int = 1):
        super().__init__()
        self.need_proj = (in_c != out_c) or (freq_stride != 1)
        if self.need_proj:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(freq_stride, 1), bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.shortcut = None

        self.pw_in = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.freq_dw = nn.Sequential(
            nn.Conv2d(
                out_c,
                out_c,
                kernel_size=(3, 1),
                stride=(freq_stride, 1),
                padding=(1, 0),
                groups=out_c,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
        )

        pad_t = 2 * dilation
        self.temp_dw = nn.Sequential(
            nn.Conv2d(
                out_c,
                out_c,
                kernel_size=(1, 5),
                stride=1,
                padding=(0, pad_t),
                dilation=(1, dilation),
                groups=out_c,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, freq_bins_out: int) -> torch.Tensor:
        res = self.shortcut(x) if self.shortcut is not None else x
        x2d = self.freq_dw(self.pw_in(x))  # [B, out_c, F_out, 160]
        x1d = F.avg_pool2d(x2d, kernel_size=(freq_bins_out, 1), stride=1)  # [B, out_c, 1, 160]
        x1d = self.temp_dw(x1d)  # [B, out_c, 1, 160]
        # 显式将 [B, C, 1, 160] 沿频率维复制展开为 [B, C, F_out, 160]
        x1d_bc = F.interpolate(x1d, size=(freq_bins_out, 160), mode="nearest")
        return self.act(res + x2d + x1d_bc)


class MDTCBlock(nn.Module):
    """多尺度深度可分离空洞时延卷积块: 输入/输出固定为 [B, C, 1, 160]"""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad_t = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 5),
                padding=(0, pad_t),
                dilation=(1, dilation),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BCResMDTCCommandNet(nn.Module):
    """
    第一级超轻量口令识别网 (~0.19M 参数, ~0.75 MB)
    输入: [B, 1, 80, 160]
    输出单一张量 [B, (1 + num_commands) + embed_dim]：
      - 前 (1 + num_commands) 维为余弦缩放 logits (索引 0 = UNKNOWN 闲聊/噪音拒识，索引 1..K = K 种口令)
      - 后 embed_dim (128) 维为 L2 归一化保序口令特征向量 `cmd_embedding` (支持免重训口令模板扩展)
    """
    def __init__(
        self,
        num_commands: int = 10,
        embed_dim: int = 128,
        base_channels: int = 24,
        mdtc_channels: int = 96,
        cosine_scale: float = 18.0,
    ):
        super().__init__()
        self.num_commands = num_commands
        self.embed_dim = embed_dim
        self.cosine_scale = cosine_scale

        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.bc1 = BroadcastedResBlock(base_channels, 32, freq_stride=2, dilation=1)
        self.bc2 = BroadcastedResBlock(32, 48, freq_stride=2, dilation=2)
        self.bc3 = BroadcastedResBlock(48, 64, freq_stride=2, dilation=2)

        self.to_temporal = nn.Sequential(
            nn.Conv2d(64, mdtc_channels, kernel_size=(5, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mdtc_channels),
            nn.ReLU(inplace=True),
        )

        self.mdtc = nn.Sequential(
            MDTCBlock(mdtc_channels, dilation=1),
            MDTCBlock(mdtc_channels, dilation=2),
            MDTCBlock(mdtc_channels, dilation=4),
            MDTCBlock(mdtc_channels, dilation=8),
        )

        self.order_proj = nn.Sequential(
            nn.Linear(mdtc_channels * 4, 192, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(192, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

        self.cls_head = nn.Linear(embed_dim, 1 + num_commands, bias=True)

    def normalize_classifier_weights(self):
        """将分类头权重行向量归一化，使其推理时等价于余弦原型分类器"""
        with torch.no_grad():
            self.cls_head.weight.copy_(F.normalize(self.cls_head.weight, p=2, dim=1) * self.cosine_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)              # [B, 24, 40, 160]
        h = self.bc1(h, 20)           # [B, 32, 20, 160]
        h = self.bc2(h, 10)           # [B, 48, 10, 160]
        h = self.bc3(h, 5)            # [B, 64, 5, 160]
        h = self.to_temporal(h)       # [B, 96, 1, 160]
        h = self.mdtc(h)              # [B, 96, 1, 160]

        chunks = F.avg_pool2d(h, kernel_size=(1, 40), stride=(1, 40))  # [B, 96, 1, 4]
        flat = chunks.view(chunks.size(0), -1)  # [B, 384]

        emb = self.order_proj(flat)
        norm_emb = F.normalize(emb, p=2, dim=1)
        logits = self.cls_head(norm_emb)  # [B, 1 + K]

        # 拼为单一输出张量 [B, 1 + K + 128]，C++ OpenCV DNN 直接一次 forward() 取出全部结果
        return torch.cat([logits, norm_emb], dim=1)
