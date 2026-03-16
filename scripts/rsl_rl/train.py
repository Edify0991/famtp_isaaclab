"""Train FaMTP environments with RSL-RL."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

import famtp_lab.tasks  # noqa: F401
from famtp_lab.agents.rsl_rl.ppo_cfg import get_rsl_rl_ppo_cfg
from famtp_lab.agents.rsl_rl.wrappers import RslRlPolicyObsWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RSL-RL PPO on a FaMTP task.")
    parser.add_argument("--task", type=str, default="FaMTP-Humanoid-Switch-Direct-v0")
    parser.add_argument("--headless", action="store_true", help="Accepted for Isaac Lab CLI compatibility.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=Path, default=Path("logs/rsl_rl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(args.task)
    env = RslRlPolicyObsWrapper(env)

    cfg = get_rsl_rl_ppo_cfg()
    if args.max_iterations is not None:
        cfg["max_iterations"] = args.max_iterations

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("rsl-rl-lib is required. Install with: pip install rsl-rl-lib") from exc

    args.logdir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, log_dir=str(args.logdir), device=cfg["device"])
    torch.manual_seed(args.seed)
    runner.learn(num_learning_iterations=cfg["max_iterations"], init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
