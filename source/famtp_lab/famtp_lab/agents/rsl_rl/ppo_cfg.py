"""Minimal PPO configuration for RSL-RL training."""


def get_rsl_rl_ppo_cfg(mode: str = "ppo_cmd") -> dict:
    """Return a compact RSL-RL PPO config dictionary for week-1 baseline.

    Args:
        mode: Baseline mode identifier. Current stage supports only ``ppo_cmd``.
    """
    if mode != "ppo_cmd":
        raise ValueError(f"Unsupported baseline mode '{mode}'.")

    return {
        "seed": 42,
        "device": "cuda:0",
        "max_iterations": 1500,
        "save_interval": 100,
        "experiment_name": "famtp_week1_switch",
        "run_name": mode,
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "class_name": "OnPolicyRunner",
            "num_steps_per_env": 24,
            "empirical_normalization": False,
        },
    }
