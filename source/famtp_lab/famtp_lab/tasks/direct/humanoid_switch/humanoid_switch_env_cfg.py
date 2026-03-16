"""Configuration for the FaMTP humanoid skill-switching environment."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .scene_cfg import make_scene_cfg


@configclass
class HumanoidSwitchEnvCfg(DirectRLEnvCfg):
    """Minimal DirectRLEnv config for skill-switching research bootstrapping."""

    decimation: int = 2
    episode_length_s: float = 5.0

    action_space: int = 8
    observation_space: int = 18
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = make_scene_cfg(num_envs=64, env_spacing=3.0)

    num_skills: int = 4
    default_target_skill: int = 1
    dynamics_damping: float = 0.95
    action_scale: float = 0.05
