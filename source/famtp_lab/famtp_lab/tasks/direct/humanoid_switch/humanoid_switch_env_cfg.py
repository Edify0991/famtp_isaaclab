"""Configuration for the FaMTP humanoid skill-switching environment."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .scene_cfg import make_scene_cfg


@configclass
class HumanoidSwitchEnvCfg(DirectRLEnvCfg):
    """Minimal DirectRLEnv config for week-1 feasibility studies."""

    decimation: int = 2
    episode_length_s: float = 8.0

    action_space: int = 8
    observation_space: int = 18
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = make_scene_cfg(num_envs=64, env_spacing=3.0)

    # Week-1 baseline policy mode.
    policy_mode: str = "ppo_cmd"

    # Switch protocol
    num_skills: int = 3
    chain_mode: str = "fixed"  # fixed|random
    switch_time_min_s: float = 1.0
    switch_time_max_s: float = 3.0

    dynamics_damping: float = 0.95
    action_scale: float = 0.05

    # Reward components for ppo_cmd baseline.
    rew_alive_bonus: float = 0.2
    rew_stabilization_scale: float = 1.0
    rew_command_follow_scale: float = 0.3
