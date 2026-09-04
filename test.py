# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from agent import IQN_Agent
from utils import evaluate_agent, set_seed


def test_agent(args):
    # 创建环境
    env_name = args.env_name
    if args.is_human_render:
        test_env = gym.make(env_name, render_mode="human")
    else:
        test_env = gym.make(env_name, render_mode=None)

    # 测试环境随机种子
    set_seed(args.test_seed)
    test_env.reset(seed=args.test_seed)

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n.item()

    # 创建智能体
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

    model_path = f"./model/{env_name}/" + args.load_name

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    agent.load(model_path)

    avg_scores, avg_steps = evaluate_agent(test_env, agent, args.is_human_render, args.test_num, print_infos=True)
    test_env.close()

    return avg_scores, avg_steps
