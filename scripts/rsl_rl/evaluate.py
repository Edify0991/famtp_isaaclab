"""Evaluate FaMTP checkpoint with simple rollout metrics."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import famtp_lab.tasks  # noqa: F401
from famtp_lab.agents.rsl_rl.ppo_cfg import get_rsl_rl_ppo_cfg
from famtp_lab.agents.rsl_rl.wrappers import RslRlPolicyObsWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint.")
    parser.add_argument("--task", type=str, default="FaMTP-Humanoid-Switch-Direct-v0")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("rsl-rl-lib is required. Install with: pip install rsl-rl-lib") from exc

    env = RslRlPolicyObsWrapper(gym.make(args.task))
    runner = OnPolicyRunner(env, get_rsl_rl_ppo_cfg(), log_dir=None, device="cpu")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cpu")

    returns = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            actions = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ep_return += float(np.mean(reward))
            done = bool(terminated.any()) or bool(truncated.any())
        returns.append(ep_return)

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Std return: {np.std(returns):.4f}")
    env.close()


if __name__ == "__main__":
    main()
