"""Train FaMTP environments with RSL-RL baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

import famtp_lab.tasks  # noqa: F401
from famtp_lab.agents.rsl_rl.ppo_cfg import get_rsl_rl_ppo_cfg
from famtp_lab.agents.rsl_rl.wrappers import RslRlPolicyObsWrapper
from famtp_lab.tasks.direct.humanoid_switch.humanoid_switch_env_cfg import HumanoidSwitchEnvCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RSL-RL PPO on a FaMTP task.")
    parser.add_argument("--task", type=str, default="FaMTP-Humanoid-Switch-Direct-v0")
    parser.add_argument("--headless", action="store_true", help="Accepted for Isaac Lab CLI compatibility.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="ppo_cmd",
        choices=["ppo_cmd", "fullbody_amp", "partwise_raw", "famtp_stage1", "famtp_nobridge", "famtp_full"],
    )
    parser.add_argument("--chain-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--use-manifold-encoder", type=int, choices=[0,1], default=1)
    parser.add_argument("--use-latent-part-priors", type=int, choices=[0,1], default=1)
    parser.add_argument("--use-global-coupling", type=int, choices=[0,1], default=1)
    parser.add_argument("--latent-history-steps", type=int, default=4)
    parser.add_argument("--latent-dim-residual", type=int, default=2)
    parser.add_argument("--bridge-horizon-steps", type=int, default=12)
    parser.add_argument("--bridge-replan-mode", type=str, default="on_switch", choices=["on_switch", "periodic"])
    parser.add_argument("--use-target-anchor", type=int, choices=[0,1], default=1)
    parser.add_argument("--bridge-update-interval", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = HumanoidSwitchEnvCfg()
    env_cfg.prior_mode = args.prior_mode
    env_cfg.chain_mode = args.chain_mode
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    # FaMTP ablations.
    if args.prior_mode in {"famtp_stage1", "famtp_nobridge", "famtp_full"}:
        env_cfg.use_manifold_encoder = bool(args.use_manifold_encoder)
        env_cfg.use_latent_part_priors = bool(args.use_latent_part_priors)
        env_cfg.use_global_coupling = bool(args.use_global_coupling)
        env_cfg.latent_history_steps = args.latent_history_steps
        env_cfg.latent_dim_residual = args.latent_dim_residual
        env_cfg.bridge_horizon_steps = args.bridge_horizon_steps
        env_cfg.bridge_replan_mode = args.bridge_replan_mode
        env_cfg.use_target_anchor = bool(args.use_target_anchor)
        env_cfg.bridge_update_interval = args.bridge_update_interval

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlPolicyObsWrapper(env)

    cfg = get_rsl_rl_ppo_cfg(mode=args.prior_mode)
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
