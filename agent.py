# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import numpy as np

from net import IQN_Network


class ReplayBuffer:
    def __init__(self, max_len: int, state_dim: int):
        self.max_len = max_len

        self.next_idx = 0           # 下一个要写入的位置
        self.count = 0              # 当前已有数据量

        # 为每个数据字段预分配 NumPy 数组
        self.states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_len, 1), dtype=np.float32)
        self.rewards = np.zeros((max_len, 1), dtype=np.float32)
        self.next_states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_len, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.count = min(self.count + 1, self.max_len)
        self.next_idx = (self.next_idx + 1) % self.max_len

    def sample(self, batch_size):
        """随机采样一个 batch"""
        # 从 [0, self.count) 范围内随机选取 batch_size 个索引
        # replace=False 的意思是：不放回抽样 不抽一样的
        indices = np.random.choice(self.count, batch_size, replace=False)
        # 直接从数组中根据索引取值
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])


def update_epsilon(current_step, total_decay_step=50000, epsilon_start=0.8, epsilon_end=0.01):
    if current_step < total_decay_step:
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * current_step / total_decay_step
    else:
        epsilon = epsilon_end
    return epsilon

# ------------------------- 风险扭曲函数 -------------------------
def wang_transform(u, lamb=0.0):
    """
    Wang 变换：beta(u) = Phi(Phi^{-1}(u) + lamb)
    输入 u 为 [0,1] 的采样，输出扭曲后的 tau
    """
    if lamb == 0.0:
        return u

    # torch.special.ndtri(u)    标准正态分布累积分布函数 (CDF) 的反函数
    # torch.special.ndtr        标准正态 CDF

    return torch.special.ndtr(torch.special.ndtri(u) + lamb)


# ------------------------- 智能体 -------------------------
class IQN_Agent:
    def __init__(self, state_dim: int, action_dim: int, hid_dim: int, embed_dim: int,
                 n_quantile, n_quantile_target, m_quantile_eval, kappa=1.0, risk_lambda=0.0,
                 gamma=0.99,
                 lr=5e-4,
                 update_tau=0.01,
                 epsilon_start=0.8,
                 clip_norm=1.0,
                 device="cpu"):

        self.Q_net = IQN_Network(state_dim, action_dim, hid_dim, embed_dim).to(device)
        self.targetQ_net = IQN_Network(state_dim, action_dim, hid_dim, embed_dim).to(device)
        self.targetQ_net.load_state_dict(self.Q_net.state_dict())
        self.targetQ_net.eval()

        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma                                  # 折扣因子
        self.update_tau = update_tau                      # 目标网络软更新系数
        # 梯度裁剪
        self.clip_norm = clip_norm

        self.n_quantile = n_quantile                        # 训练时采样的分位数个数 N
        self.n_quantile_target = n_quantile_target          # 目标网络中采样的分位数个数 N'
        self.m_quantile_eval = m_quantile_eval              # 动作选择时采样的分位数个数 M
        """
        将 kappa 设为1，
        可以使损失函数在误差较小时（|u| <= 1）表现得像MSE一样平滑，利于梯度下降；
        在误差较大时（|u| > 1）则像MAE一样鲁棒，减少异常值的影响
        """
        self.kappa = kappa                  # Huber损失阈值

        self.risk_lambda = risk_lambda      # Wang 变换参数，0 为中性，>0 风险厌恶，<0 风险寻求

        self.action_dim = action_dim
        self.dvc = device
        self.epsilon = epsilon_start             # epsilon
        self.train_num = 0

    def quantile_huber_loss(self,
            taus,           # (batch, N)
            pred_quantiles: torch.Tensor,  # (batch_size, N)
            target_quantiles: torch.Tensor  # (batch_size, N')
    ) -> torch.Tensor:
        """
        计算分位数 Huber 损失。

        对应公式: ρ_τ^κ(u) = |τ - 1_{u<0}| * L_κ(u)
        其中 L_κ(u) 是 Huber 损失。
        """
        # pred_quantiles: (B, N), target_quantiles: (B, N)
        # 计算所有 (B, N, N) 的误差
        # 扩展维度: pred -> (B, N, 1), target -> (B, 1, N)
        pred_expanded = pred_quantiles.unsqueeze(-1)  # (B, N, 1)
        target_expanded = target_quantiles.unsqueeze(1)  # (B, 1, N')

        # | τ - 1{u < 0} | * u
        errors = target_expanded - pred_expanded  # (B, N, N')
        # u: errors = 目标值 - 预测值
        # errors < 0 (目标值 < 预测值): 分位数权重 |τ-1|
        # errors > 0 (目标值 > 预测值): 分位数权重 |τ|

        # Huber 损失
        abs_errors = torch.abs(errors)      # (B, N, N')
        huber_loss = torch.where(
            abs_errors <= self.kappa,
            0.5 * errors ** 2,
            self.kappa * (abs_errors - 0.5 * self.kappa)
        )

        # 分位数权重: |τ - 1_{errors < 0}|
        # taus: (batch, N) -> (B, N, N')
        taus_expanded = taus.unsqueeze(-1).expand(-1, -1, errors.size(-1))  # (batch, N, N')
        indicators = (errors < 0).float()  # (B, N, N')  errors < 0 的元素变成 1.0
        # quantile_weights 元素是 |τ - 1|  或  |τ - 0|
        quantile_weights = torch.abs(taus_expanded - indicators)  # (B, N, N')

        # 加权损失并求平均
        loss = (quantile_weights * huber_loss).mean()
        return loss

    # ----------------------------------- 选择动作 -----------------------------------
    def select_action(self, state, explore=True):

        if explore:
            # 生成[0, 1) 区间的均匀分布随机浮点数
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_dim)
                return action

        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.dvc)

            # 采样 M 个 u，进行风险扭曲
            u = torch.rand(1, self.m_quantile_eval, device=self.dvc)  # (1, M)
            tau = wang_transform(u, self.risk_lambda)  # (1, M)

            z = self.Q_net(state, tau)      # (1, M, action_dim)
            # 沿 M 维平均得到 Q_beta (1, action_dim)
            q_beta = z.mean(dim=1)          # (1, action_dim)

            action = q_beta.argmax(dim=1).item()

        return action

    def update(self, replay_buffer, batch_size, writer):
        self.train_num += 1

        # ---------- 从经验池中采样 ----------
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # 转换为 PyTorch Tensor
        state = torch.tensor(state, dtype=torch.float32).to(self.dvc)
        action = torch.tensor(action, dtype=torch.long).to(self.dvc)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.dvc)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.dvc)
        done = torch.tensor(done, dtype=torch.float32).to(self.dvc)

        # 1. 采样分位数 tau 和 tau' (用于当前和目标)
        # 训练时，每个样本需要采样 N 个 tau（可独立，也可共享，原始实现每个样本独立采样）
        tau = torch.rand(batch_size, self.n_quantile, device=self.dvc)  # (batch, N)
        tau_prime = torch.rand(batch_size, self.n_quantile_target, device=self.dvc)  # (batch, N')

        # 2. 应用风险扭曲
        tau = wang_transform(tau, self.risk_lambda)  # (batch, N)
        tau_prime = wang_transform(tau_prime, self.risk_lambda)  # (batch, N')

        # 3. 当前网络预测 Z_tau(s,a)
        # 获取所有动作的 Z 值 (batch, N, action_dim)
        z_all = self.Q_net(state, tau)                           # (batch, N, action_dim)
        # 根据 action 索引取出对应动作的分位数值 (batch, N, 1)
        a_ind = action.unsqueeze(1).expand(-1, self.n_quantile, -1)     # (batch, N, 1)
        z_pred = z_all.gather(2, a_ind)  # (batch, N, 1)
        z_pred = z_pred.squeeze(-1)      # (batch, N)

        # 4. 目标网络计算 TD 目标
        with torch.no_grad():
            # 选动作 和 评估目标 不能用同一组 τ

            # 选择动作：使用目标网络和扭曲期望（使用中位数或均值？原始IQN使用中间分位数τ=0.5用于选动作）
            # 采样多个 tau_sel 求平均 Q 来选择动作，这样极大地降低方差，让动作选择更鲁棒
            # 找出当前最优动作（平滑、稳定）

            tau_sel = torch.rand(batch_size, self.m_quantile_eval, device=self.dvc)
            tau_sel = wang_transform(tau_sel, self.risk_lambda)
            z_next_all = self.Q_net(next_state, tau_sel)          # (batch, M, action_dim)
            q_next = z_next_all.mean(dim=1)                            # (batch, action_dim)
            next_action = q_next.argmax(dim=1, keepdim=True)          # (batch,1)

            # 计算目标分位数值：用另外一组采样得到的 tau_prime 计算 (next_state, next_actions)
            z_next = self.targetQ_net(next_state, tau_prime)            # (batch, N', action_dim)
            next_a_ind = next_action.unsqueeze(1).expand(-1, self.n_quantile_target, -1)  # (batch, N',1)
            z_next = z_next.gather(2, next_a_ind)  # (batch, N',1)
            z_next = z_next.squeeze(-1)                                # (batch, N')

            # TD 目标: r + gamma * (1-done) * z_next
            target = reward + self.gamma * (1 - done) * z_next              # (batch, N')

        # ------------------------------------------
        # 5. 计算分位数 Huber 损失
        quantile_loss = self.quantile_huber_loss(tau, z_pred, target)

        # ---- 反向传播 ----
        self.optimizer.zero_grad()
        quantile_loss.backward()   # [梯度] 从 loss 反向传播到 policy_net 的所有参数
        # 梯度裁剪，防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), self.clip_norm)
        self.optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.Q_net.parameters(), self.targetQ_net.parameters()):
            target_param.data.copy_(self.update_tau * param.data + (1 - self.update_tau) * target_param.data)

        # ------------------------------------------------------------------------------------
        if self.train_num % 50 == 0:        # 控制记录频率

            with torch.no_grad():
                # 计算当前状态的平均 Q 值（扭曲期望）作为监控指标
                u_monitor = torch.rand(batch_size, self.m_quantile_eval, device=self.dvc)
                tau_monitor = wang_transform(u_monitor, self.risk_lambda)
                z_monitor = self.Q_net(state, tau_monitor)  # (batch, M, action_dim)
                q_monitor = z_monitor.mean(dim=1)  # (batch, action_dim)
                # 取所有样本和动作的平均 Q 值（或取选择动作的 Q 值，我们取所有动作的平均）
                q_mean = q_monitor.mean().item()

            # TensorBoard 记录
            writer.add_scalar("Q_mean", q_mean, self.train_num)
            writer.add_scalar("Loss", quantile_loss.item(), self.train_num)
            writer.add_scalar("Grad_norm", grad_norm.item(), self.train_num)
            writer.add_scalar("Epsilon", self.epsilon, self.train_num)

    def save(self, path):
        """保存模型权重"""
        torch.save(self.Q_net.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.Q_net.load_state_dict(torch.load(path, map_location=self.dvc, weights_only=True))

