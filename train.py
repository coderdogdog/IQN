# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from agent import IQN_Agent, ReplayBuffer, update_epsilon
from utils import evaluate_agent, args_to_txt, set_seed


def train_agent(args):

    if args.warmup_steps < args.batch_size:
        raise ValueError("warmup_steps 应该远大于 batch_size")

    # 创建模型文件夹
    # ./model/env_name/
    env_name = args.env_name
    model_dir = f"./model/{env_name}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 训练日志
    log_dir = f'''runs/{env_name}/IQN_{datetime.now().strftime("%Y%m%d_%H_%M_%S")}'''
    writer = SummaryWriter(log_dir)

    # 当前训练参数写入 txt 文件，方便查看
    args_txt_path = log_dir + "/training_parameters.txt"
    args_to_txt(args, args_txt_path)
    print("\n")
    print("------------- 深度强化学习算法 IQN 处理单维度离散动作空间问题 -------------")
    print(f"[本次 Gymnasium 实验环境（单维度离散动作空间）]: {env_name}")
    print(f"[本次实验的训练参数已写入]: {log_dir}/training_parameters.txt")
    print(f"[训练种子]: {args.train_seed}")
    print(f"[测试种子]: {args.test_seed}")
    print(f"[在新的终端窗口运行 TensorBoard 以查看训练曲线]: tensorboard --logdir=runs/{env_name}")
    print("\n")

    # ---------------------------------------- 创建环境和智能体 ----------------------------------------------
    # 创建训练环境
    env = gym.make(env_name, render_mode=None)
    # 创建用来测试评估的环境
    evaluate_env = gym.make(env_name, render_mode=None)

    # 设置随机种子
    set_seed(args.train_seed)
    env.reset(seed=args.train_seed)
    evaluate_env.reset(seed=args.test_seed)

    # 游戏环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n.item()

    buffer = ReplayBuffer(args.buffer_max_len, state_dim)

    agent = IQN_Agent(state_dim, action_dim, args.hidden_dim, args.embed_dim,
                          n_quantile=args.n_quantile,
                          n_quantile_target=args.n_quantile_target,
                          m_quantile_eval=args.m_quantile_eval,
                          kappa=args.kappa,
                          risk_lambda=args.risk_lambda,
                          gamma=args.gamma,
                          lr=args.lr,
                          update_tau=args.update_tau,
                          epsilon_start=args.epsilon_start,
                          clip_norm=args.clip_norm,
                          device=args.device)
    # -------------------------------------------------------------------------------------------------------
    total_steps = 0
    decay_steps = 0         # epsilon 衰减步数

    best_avg_reward = -float(300)
    print("[新训练] 开始训练...")

    best_path = model_dir + "best_scores_" + str(int(best_avg_reward)) + ".pth"
    # -----------------------------------------------------------------------------------

    # ------------------------------------- 智能体开始与环境交互 ------------------------------------------------
    # 游戏回合前初始化
    state, _ = env.reset()

    try:
        while total_steps < args.max_env_steps:

            # 先预热 随机选取动作积累经验 先走一些步
            if total_steps < args.warmup_steps:
                action = env.action_space.sample()  # 预热期随机探索
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, infos = env.step(action)

            buffer.store(state, action, reward, next_state, terminated)
            total_steps += 1

            done = terminated or truncated
            if done:
                state, _ = env.reset()
            else:
                state = next_state
            # ----------------------------------------------------------------------------------------

            if total_steps > args.warmup_steps:
                # --------------------------------- 训练更新 -----------------------------------------
                if total_steps % args.train_frequency == 0:
                    agent.update(buffer, args.batch_size, writer)
                    decay_steps += 1        # epsilon 衰减
                    agent.epsilon = update_epsilon(decay_steps, args.total_decay_steps, args.epsilon_start, args.epsilon_end)

                    # ------------------- 测试 ----------------------
                    if agent.train_num % args.eval_interval == 0:
                        # 固定每局初始状态（第 j 局用 test_seed + j）：不同评估之间结果可比，
                        # 避免随机初始状态带来的评估方差
                        avg_r, avg_steps = evaluate_agent(evaluate_env, agent,
                                                          is_human_render=False, test_numb=3,
                                                          print_infos=False, seed=args.test_seed)

                        print(f"游戏总步数: {total_steps:8d} | 智能体更新次数: {agent.train_num:8d} | "
                              f"每回合平均奖励: {avg_r:9.2f} | 每回合平均步数: {avg_steps:5d}")

                        writer.add_scalar("test/avg_r", avg_r, agent.train_num)
                        writer.add_scalar("test/avg_steps", avg_steps, agent.train_num)

                        # 保存表现最好的模型
                        if avg_r > best_avg_reward:
                            if os.path.isfile(best_path):
                                os.remove(best_path)

                            best_avg_reward = avg_r
                            best_path = model_dir + "best_scores_" + str(int(best_avg_reward)) + ".pth"

                            agent.save(best_path)

                    # ----------------------------------- 定时保存模型 ---------------------------------------------------
                    if agent.train_num % args.save_interval == 0:
                        model_name = "trained_" + str(agent.train_num) + ".pth"
                        model_path = model_dir + model_name

                        # 保存模型
                        agent.save(model_path)

    except KeyboardInterrupt:
        print("\n\n[中断] 用户手动停止训练")

    env.close()
    evaluate_env.close()
    writer.close()

