# -*- coding: utf-8 -*-
import argparse
import numpy as np
import random
import torch


def evaluate_agent(env_x, agent_x, is_human_render=False, test_numb=3, print_infos=False, seed=None):
    """
    :param env_x: 测试环境
    :param agent_x: agent_x必须有 select_action 的方法 select_action(state, False)
    :param is_human_render:
    :param test_numb:
    :param print_infos:
    :param seed: 若不为 None，第 j 局使用固定种子 seed + j，多次评估结果可复现、可对比（降低随机方差）
    :return: 平均每局奖励 平均每局步数
    """
    total_r = 0
    t_steps = 0
    for j in range(test_numb):
        state, _ = env_x.reset(seed=None if seed is None else seed + j)
        done = False
        epi_r = 0
        epi_steps = 0
        while not done:
            action = agent_x.select_action(state, False)
            s_next, r, terminated, truncated, _ = env_x.step(action)
            done = terminated or truncated
            epi_r += r
            epi_steps += 1

            total_r += r
            t_steps += 1
            state = s_next

            # 渲染
            if is_human_render:
                env_x.render()

        if print_infos:
            print(f"【第{j + 1}局】>>得分: {epi_r:8.2f} | 步数: {epi_steps:8d}")

    avg_r = total_r/test_numb
    avg_steps = int(t_steps/test_numb)

    if print_infos:
        print("-----------------------------------")
        print(f"【平均每局】得分: {avg_r:8.2f} | 步数: {avg_steps:8d}")

    return avg_r, avg_steps


def set_seed(seed: int = 10) -> None:
    """固定所有随机种子，确保实验可复现。"""
    # 1. Python 内置随机
    random.seed(seed)

    # 2. NumPy 随机
    np.random.seed(seed)

    # 3. PyTorch 随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)       # 当前 GPU
    torch.cuda.manual_seed_all(seed)    # 所有 GPU（多卡时）

    # 4. cuDNN 确定性
    # 强制 cuDNN 只使用确定性算法，结果可复现
    torch.backends.cudnn.deterministic = True
    # 不自动搜索最优卷积算法
    torch.backends.cudnn.benchmark = False

    # 5. (可选) 环境种子 — gymnasium 新版推荐在 reset 时传入
    # env.reset(seed=seed)
    # print(f"随机种子已固定为: {seed}")


def args_to_txt(opt, txt_path):
    # 把各个参数写入 txt 文件中方便查看
    args_dict = vars(opt)
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in args_dict.items():
            f.write(f"{k} : {v}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

