# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """正交初始化线性层权重，偏置置零。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class IQN_Network(nn.Module):
    """
    IQN 网络
    """

    def __init__(self, state_dim: int, action_dim: int, hid_dim: int=256, embed_dim: int=64):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # 状态特征提取层
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, embed_dim)
        )

        # 分位数嵌入层：输入 tau (batch, N) -> 输出 phi (batch, N, embed_dim)
        # 注意：这里使用 cos 编码，然后通过线性层 + ReLU
        # 编码后映射到 embed_dim 再 --> hid_dim
        self.embed_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 融合后输出层：输入 (state_feat * phi) (batch, N, hidden_dim) -> (batch, N, action_dim)
        self.output_fc = nn.Linear(embed_dim, action_dim)

    def forward(self, state, tau):
        """
        state: (batch, state_dim)
        tau:   (batch, N)   N 为采样的分位数个数（训练时 N=N_QUANTILES，动作选择时 N=N_QUANTILES_ACTION）
        返回: (batch, N, action_dim)
        """
        # batch_size = state.size(0)
        # N = tau.size(1)

        # 1. 状态特征
        state_feat = self.state_fc(state)  # (batch, embed_dim)

        # 2. 分位数嵌入 (余弦编码 + 线性映射)
        # 余弦编码：phi_j(tau) = cos(pi * i * tau)         i from 0 to embed_dim-1
        # 此处我们使用 PyTorch 实现，输入 tau 形状 (batch, N)
        # 扩展维度用于广播
        tau_expanded = tau.unsqueeze(-1)  # (batch, N, 1)
        i = torch.arange(self.embed_dim, device=tau.device).float().view(1, 1, -1)  # (1,1,embed_dim)
        # 计算 cos(pi * i * tau)
        # 广播相乘
        cos_embed = torch.cos(np.pi * i * tau_expanded)  # (batch, N, embed_dim)

        # 通过线性层 + ReLU
        phi = self.embed_fc(cos_embed)          # (batch, N, embed_dim)

        # 3. 融合：逐元素乘积（门控机制）-> 广播相乘
        # 这里要求 state_feat.shape[-1] = phi.shape[-1] 直接相乘

        state_feat_exp = state_feat.unsqueeze(1)  # (batch, 1, embed_dim)
        combined = state_feat_exp * phi  # (batch, N, embed_dim)

        # 4. 输出层，得到每个分位数的值
        out = self.output_fc(combined)  # (batch, N, action_dim)
        return out


