"""
方案 A - 第二级：超轻量声纹特征提取网络 (CAMPlusLiteSpeakerNet)
结构设计（借鉴 3D-Speaker CAM++ 思想并全算子适配 OpenCV DNN）：
1. 2D 频域残差卷积茎 (FCM): [B, 1, 80, 160] -> [B, 32, 20, 160] -> [B, 192, 1, 160]
2. 密集连接上下文掩码时延塔 (CAM-DTDNN Blocks)：
   - 融合全局语句上下文与局部 21 帧音节上下文，生成动态 Sigmoid 声纹掩码
   - 密集连接扩展通道: 192 -> 240 -> 288 -> 336 -> 384
3. 多矩统计池化 (Mean + Std Statistics Pooling) + 注意力时序门控：
   - 同时提取整句一阶均值 mu 与二阶标准差 sigma，捕获共振峰分布与基频波动特征
4. BNNeck 投影输出 192 维 L2 归一化声纹嵌入向量 (与 10 位家人底库余弦比对)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCMResBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, freq_stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_c, out_c, kernel_size=3, stride=(freq_stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        if in_c != out_c or freq_stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(freq_stride, 1), bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x) if self.down is not None else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_c: int, growth_rate: int = 48, dilation: int = 1):
        super().__init__()
        bn_c = min(in_c, growth_rate * 2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, bn_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_c),
            nn.ReLU(inplace=True),
        )
        pad_t = dilation
        self.tdnn = nn.Sequential(
            nn.Conv2d(
                bn_c,
                growth_rate,
                kernel_size=(1, 3),
                padding=(0, pad_t),
                dilation=(1, dilation),
                bias=False,
            ),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
        )

        mid_c = growth_rate // 2
        # 等价分解 Linear([global, local]) = Linear_g(global) + Linear_l(local)，零动态 Expand 算子
        self.cam_global = nn.Conv2d(bn_c, mid_c, kernel_size=1, bias=True)
        self.cam_local = nn.Conv2d(bn_c, mid_c, kernel_size=1, bias=False)
        self.cam_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, growth_rate, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bottleneck(x)         # [B, bn_c, 1, 160]
        feat = self.tdnn(h)            # [B, growth_rate, 1, 160]

        g_ctx = F.avg_pool2d(h, kernel_size=(1, 160), stride=1)             # [B, bn_c, 1, 1]
        g_proj = F.interpolate(self.cam_global(g_ctx), size=(1, 160), mode="nearest")
        l_ctx = F.avg_pool2d(h, kernel_size=(1, 21), stride=1, padding=(0, 10))
        l_proj = self.cam_local(l_ctx)                                      # [B, mid_c, 1, 160]

        mask = self.cam_out(g_proj + l_proj)
        return feat * mask


class GatedStatsPooling(nn.Module):
    """
    门控注意力统计池化 (ASP 等价 OpenCV-DNN 安全实现)：
    对每帧施加自注意力门控权重后，同时聚合均值 mu 与标准差 sigma (共 2C 维)
    """
    def __init__(self, channels: int, bottleneck_dim: int = 128):
        super().__init__()
        self.frame_gate = nn.Sequential(
            nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 1, 160]
        w = self.frame_gate(x) + 1e-4               # [B, C, 1, 160]
        w_mean = F.avg_pool2d(w, kernel_size=(1, 160), stride=1)
        wx_mean = F.avg_pool2d(w * x, kernel_size=(1, 160), stride=1)
        wx2_mean = F.avg_pool2d(w * (x * x), kernel_size=(1, 160), stride=1)

        mu = wx_mean / w_mean                       # [B, C, 1, 1]
        m2 = wx2_mean / w_mean                      # [B, C, 1, 1]
        var = torch.clamp(m2 - mu * mu, min=1e-5)
        sigma = torch.pow(var, 0.5)

        stats = torch.cat([mu, sigma], dim=1)       # [B, 2C, 1, 1]
        return stats.view(stats.size(0), -1)        # [B, 2C]


class CAMPlusLiteSpeakerNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        stem_channels: int = 32,
        init_channels: int = 192,
        growth_rate: int = 48,
        num_cam_layers: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.fcm = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            FCMResBlock2D(stem_channels, stem_channels, freq_stride=2),
            FCMResBlock2D(stem_channels, stem_channels, freq_stride=2),
        )

        self.fcm_proj = nn.Sequential(
            nn.Conv2d(stem_channels, init_channels, kernel_size=(20, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        dilations = [1, 2, 3, 4]
        self.cam_layers = nn.ModuleList()
        cur_c = init_channels
        for i in range(num_cam_layers):
            d = dilations[i % len(dilations)]
            self.cam_layers.append(CAMDenseTDNNLayer(cur_c, growth_rate=growth_rate, dilation=d))
            cur_c += growth_rate

        self.transit = nn.Sequential(
            nn.Conv2d(cur_c, cur_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(cur_c),
            nn.ReLU(inplace=True),
        )

        self.asp = GatedStatsPooling(cur_c, bottleneck_dim=128)
        self.neck = nn.Sequential(
            nn.BatchNorm1d(cur_c * 2),
            nn.Linear(cur_c * 2, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fcm(x)
        h = self.fcm_proj(h)
        for layer in self.cam_layers:
            h = torch.cat([h, layer(h)], dim=1)
        h = self.transit(h)
        stats = self.asp(h)
        emb = self.neck(stats)
        return F.normalize(emb, p=2, dim=1)


class AAMSoftmaxHead(nn.Module):
    def __init__(self, in_features: int = 192, num_speakers: int = 40, s: float = 30.0, m: float = 0.25):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norm_w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(emb, norm_w)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-6, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s
