"""Humanoid skill-switching Direct workflow environment registration."""

import gymnasium as gym

TASK_ID = "FaMTP-Humanoid-Switch-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.humanoid_switch_env:make_humanoid_switch_env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_switch_env_cfg:HumanoidSwitchEnvCfg",
        "rsl_rl_cfg_entry_point": "famtp_lab.agents.rsl_rl.ppo_cfg:get_rsl_rl_ppo_cfg",
    },
)

__all__ = ["TASK_ID"]
