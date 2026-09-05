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

        self.hid_dim = hid_dim
        self.n_cos = embed_dim  # 余弦基个数（不再是组合维度）

        # ---- 1. 状态编码 psi(s): (B, state_dim) -> (B, hid_dim) ----
        self.state_fc = nn.Sequential(
            layer_init(nn.Linear(state_dim, hid_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU(),
        )

        # ---- 2. 分位数嵌入 phi(tau): (B, N, n_cos) -> (B, N, hid_dim) ----
        # phi_j(tau) = ReLU( sum_i w_ji cos(pi*i*tau) )，与原论文一致
        self.phi_fc = nn.Sequential(
            layer_init(nn.Linear(self.n_cos, hid_dim)),
            nn.ReLU(),
        )

        # ---- 3. 融合 + 价值头: (B, N, hid_dim) -> (B, N, action_dim) ----
        self.head = nn.Sequential(
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hid_dim, action_dim), gain=1.0),  # 输出层增益 1.0
        )

        # 余弦频率 i = 1..n_cos（从 0 起也可，只是多一个常数基，线性层能自适应）
        i = torch.arange(1, self.n_cos + 1, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("cos_i", i)    # (1, 1, n_cos)，随 .to(device) 一起迁移

    def forward(self, state, tau):
        """
        state: (B, state_dim)
        tau:   (B, N)，N 可为训练/目标/动作选择任意数量的分位数
        返回:  (B, N, action_dim)
        """
        psi = self.state_fc(state)  # (B, hid)

        tau = tau.unsqueeze(-1)  # (B, N, 1)
        cos_embed = torch.cos(np.pi * self.cos_i * tau)  # (B, N, n_cos)
        phi = self.phi_fc(cos_embed)  # (B, N, hid)

        h = psi.unsqueeze(1) * phi  # (B, N, hid) 逐元素门控融合
        out = self.head(h)  # (B, N, action_dim)
        return out

