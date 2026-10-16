import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) 分类头
    专为极高精度身份区分与拒识陌生人设计：
    - 在超球面角度空间上对目标类别强制施加裕度 m (默认 0.50 弧度，约 28.6 度)
    - 迫使自家人特征向量高度聚拢，非该家人的特征被强力推远
    - 对于未登记的陌生人，其在超球面上的余弦相似度远低于设定阈值，从而实现零误开门
    
    参数:
        in_features: 输入特征维度 (默认 512)
        out_features: 类别数 (家庭成员上限，例如 15)
        s: 辐射特征尺度因子 (默认 32.0 / 64.0)
        m: 角度裕度 margin (默认 0.50)
        easy_margin: 是否使用简化裕度模式
    """
    def __init__(self, in_features=512, out_features=15, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. 权重归一化: W / ||W||
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # 如果处于纯推理模式且未提供标签，直接输出各个家庭成员的原生余弦相似度
        if label is None:
            return cosine
        
        # 2. 训练模式: 施加 ArcFace 角度裕度 m
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 3. 构造 one-hot 掩码，仅对正确类别施加角度裕度
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def get_normalized_prototypes(self):
        """返回已完成归一化的家庭成员原型特征矩阵 (out_features, in_features)"""
        with torch.no_grad():
            return F.normalize(self.weight, p=2, dim=1).cpu().numpy()
