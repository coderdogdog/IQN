# IQN（Implicit Quantile Networks）隐式分位数网络

<div align="center">
  <img src="IQN_CartPole-v1.gif" alt="IQN CartPole-v1 演示" width="480"/>
  <br/>
  <sub>IQN 在 CartPole-v1 上的运行演示</sub>
</div>

## 项目简介

本项目基于 **PyTorch + Gymnasium**，使用深度强化学习算法 **IQN（隐式分位数网络）** 处理**单维度离散动作空间**问题（如 CartPole-v1、LunarLander-v3、MountainCar-v0）。

算法属性：`model-free`、`off-policy`、`value-based`、`discrete`。

IQN 属于**分布式强化学习**：它不直接学习期望回报 Q(s,a)，而是学习整个**回报分布的隐式分位数函数** Z_τ(s,a)，再通过对分位数取平均得到动作价值，因此对回报分布的刻画更精细，训练通常也更稳定。

## 算法简介

### 从 DQN 到 IQN

| 算法 | 建模对象 | 说明 |
|---|---|---|
| DQN | 期望值 Q(s,a) | 单一数值，忽略回报分布信息 |
| QR-DQN | 固定分位数（等间隔） | 需要人为指定分位数位置 |
| **IQN** | **任意分位数 τ ∈ [0,1]** | τ 连续采样，无需预先划分区间，收敛更快、拟合更精确 |

### 核心思想

- 网络输出分位数函数 Z_τ(s,a)，其中 τ 是任意采样的分位水平；
- τ 通过**余弦嵌入** φ_j(τ) = ReLU(Σ_i cos(π·i·τ)·w_ij) 编码后与状态特征逐元素融合：
  `Z_τ(s,a) = head( ψ(s) ⊙ φ(τ) )`；
- 动作价值用扭曲期望近似：Q(s,a) ≈ mean over 采样 τ 的 Z_τ(s,a)；
- 训练损失为**分位数 Huber 损失**（quantile-Huber）：

```
ρ_τ^κ(u) = |τ - 1{u < 0}| · L_κ(u)，   u = 目标值 - 预测值
```

- 支持**风险扭曲**（Wang 变换）：β(u) = Φ(Φ⁻¹(u) + λ)，λ=0 中性、λ>0 风险厌恶、λ<0 风险寻求，可一键切换风险偏好。

## 实现要点

- **τ 三重独立采样**：训练分位数 τ（N 个）、目标分位数 τ′（N′ 个）、动作选择分位数 τ_sel（M 个）互不共用，符合 IQN 对"选动作与估值必须解耦"的要求；
- **Double-Q 风格目标**：用**在线网络**（Q_net）选 next_action，用**目标网络**（targetQ_net）估值，缓解过估计；
- **目标网络软更新**（Polyak 平均，τ=0.01），并开启梯度裁剪；
- **截断（truncated）不计入终止**：经验池只存 `terminated` 标志，超时截断的回合按非终止做自举，符合 Gymnasium 规范；
- **Wang 变换用 `torch.special.ndtri / ndtr` 实现**：原生算子、自动跟随设备，无跨设备风险；
- **按论文结构实现的网络**（详见 `net.py`）：三层状态编码 → hid_dim；φ(τ) 经 Linear+ReLU 升到 hid_dim 再与 ψ(s) 逐元素相乘；融合后接两层价值头；全部**正交初始化**；余弦频率表用 `register_buffer` 注册，随 `.to(device)` 自动迁移；
- **评估可复现**：周期评估固定每局初始种子（seed + j），多次评估之间可比。

## 环境与依赖

- Python 3.11+
- PyTorch（CUDA 版，RTX 50 系需安装 Nightly：`pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132`）
- Gymnasium 1.x
- NumPy、TensorBoard

## 目录结构

```
IQN/
├── IQN_CartPole-v1.gif   # CartPole-v1 演示动画
├── main.py               # 入口：命令行参数解析，选择训练/测试
├── train.py              # 训练主循环（含周期评估、最优/定时保存模型）
├── test.py               # 测试脚本（加载模型并评估）
├── agent.py              # IQN_Agent、ReplayBuffer、Wang 变换、epsilon 退火
├── net.py                # IQN_Network（改进版网络结构）
├── utils.py              # evaluate_agent、set_seed、args_to_txt、str2bool
├── model/<env_name>/     # 训练产出的模型权重
└── runs/<env_name>/      # TensorBoard 日志 + 训练参数 txt
```

## 快速开始

### 1. 训练

```bash
# 训练 CartPole-v1（默认环境）
python main.py --is_test_mode 0 --env_name CartPole-v1

# 训练 LunarLander-v3
python main.py --is_test_mode 0 --env_name LunarLander-v3
```

- 训练过程中会每 `eval_interval`（默认 1000）次智能体更新后，用固定种子跑 3 局做周期评估；
- 表现最好的模型保存为 `model/<env_name>/best_scores_<分数>.pth`；
- 每 `save_interval`（默认 5000）次更新定时保存 `model/<env_name>/trained_<更新次数>.pth`；
- 按 `Ctrl+C` 可随时中断，已保存的模型不受影响。

### 2. 测试（加载模型）

```bash
python main.py --is_test_mode 1 --env_name CartPole-v1 --load_name trained_10000.pth

# 无渲染测试（适合服务器）
python main.py --is_test_mode 1 --env_name CartPole-v1 --load_name trained_10000.pth --is_human_render 0
```

### 3. 查看训练曲线

```bash
tensorboard --logdir=runs
```

训练日志（含 `Q_mean`、`Loss`、`Grad_norm`、`Epsilon`、`test/avg_r`、`test/avg_steps`）记录在 `runs/<env_name>/IQN_<时间戳>/`，同时会自动写入一份 `training_parameters.txt` 方便回溯本次实验参数。

## 命令行参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--env_name` | `CartPole-v1` | 环境：CartPole-v1 / LunarLander-v3 / MountainCar-v0 |
| `--device` | `cuda:0` | 训练设备（CPU 机器用 `--device cpu`） |
| `--is_test_mode` | `False` | `True` 为测试模式，`False` 为训练模式 |
| `--train_seed` / `--test_seed` | `76` / `99` | 训练 / 测试随机种子 |
| `--is_human_render` | `1` | 测试时是否人形渲染 |
| `--test_num` | `5` | 测试局数 |
| `--load_name` | `trained_10000.pth` | 测试时加载的模型文件名 |
| `--n_quantile` | `32` | 训练分位数个数 N |
| `--n_quantile_target` | `32` | 目标分位数个数 N′ |
| `--m_quantile_eval` | `32` | 动作选择分位数个数 M |
| `--kappa` | `1.0` | Huber 损失阈值 |
| `--risk_lambda` | `0.0` | Wang 变换风险参数（0 中性 / >0 厌恶 / <0 寻求） |
| `--hidden_dim` | `256` | 网络隐藏层维度 |
| `--embed_dim` | `64` | 余弦基个数（分位数嵌入维度） |
| `--epsilon_start` / `--epsilon_end` | `0.8` / `0.01` | epsilon 贪心起止值 |
| `--total_decay_steps` | `30000` | epsilon 线性衰减总步数（按智能体更新次数计） |
| `--warmup_steps` | `5000` | 预热步数（随机动作积累经验） |
| `--max_env_steps` | `1000000` | 环境最大总步数 |
| `--train_frequency` | `4` | 每隔几个环境步训练一次 |
| `--eval_interval` | `1000` | 周期评估间隔（按更新次数） |
| `--save_interval` | `5000` | 定时保存间隔（按更新次数） |
| `--buffer_max_len` | `1000000` | 经验回放池容量 |
| `--batch_size` | `256` | 训练 batch 大小 |
| `--lr` | `5e-4` | 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--update_tau` | `0.01` | 目标网络软更新系数 |
| `--clip_norm` | `1.0` | 梯度裁剪阈值 |

完整参数列表可运行 `python main.py --help` 查看。

## 代码结构说明

| 文件 | 职责 |
|---|---|
| `net.py` | `IQN_Network`：状态编码 ψ(s)（3 层 MLP）＋ 分位数余弦嵌入 φ(τ)＋逐元素融合＋两层价值头，正交初始化 |
| `agent.py` | `IQN_Agent`：动作选择、单步更新（分位数 Huber 损失、Double-Q 目标、软更新）、模型保存/加载；`ReplayBuffer` 环形经验池；`wang_transform` 风险扭曲；`update_epsilon` 线性退火 |
| `train.py` | `train_agent`：环境交互主循环、预热、周期评估、最优模型与定时模型保存、TensorBoard 记录 |
| `test.py` | `test_agent`：加载模型批量评估（可渲染） |
| `utils.py` | `evaluate_agent` 评估工具、`set_seed` 全局种子固定、`args_to_txt` 参数落盘、`str2bool` |

## 注意事项

- `--warmup_steps` 必须大于 `--batch_size`，否则代码会直接报错（保证经验池足够采样）；
- **修改 `hidden_dim` / `embed_dim` 或网络结构后，旧 `.pth` 模型无法直接加载**（张量形状不匹配），需要重新训练；用 `load_state_dict(..., strict=False)` 可部分继承形状兼容的层；
- 时间截断（如 CartPole 到达最大步数）不写入 done 标志，按非终止处理，这是符合 Gymnasium 约定的正确做法；
- 默认设备为 `cuda:0`，无 GPU 环境请显式指定 `--device cpu`。

## 参考

- IQN 论文：Dabney et al., *Implicit Quantile Networks for Distributional Reinforcement Learning*, ICML 2018 — https://arxiv.org/abs/1806.06423
- 风险扭曲（Wang 变换）：*Risk-Sensitive Reinforcement Learning via Distorted Expectations* — https://arxiv.org/abs/1803.11078
- Gymnasium：https://gymnasium.farama.org
