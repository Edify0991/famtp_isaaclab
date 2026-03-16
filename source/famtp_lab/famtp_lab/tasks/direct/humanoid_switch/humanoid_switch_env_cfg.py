"""Configuration for the FaMTP humanoid skill-switching environment."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .scene_cfg import make_scene_cfg


@configclass
class HumanoidSwitchEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv config for baseline-comparison and FaMTP stage-1 studies."""

    decimation: int = 2
    episode_length_s: float = 8.0

    action_space: int = 8
    # 16 state + 2 skill IDs + 20 latent summary (5 parts * (sin,cos,a[2]))
    observation_space: int = 38
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene = make_scene_cfg(num_envs=64, env_spacing=3.0)

    # Unified switch for all baseline/future methods.
    prior_mode: str = "ppo_cmd"  # ppo_cmd|fullbody_amp|partwise_raw|famtp_stage1|famtp_full

    # Switch protocol.
    num_skills: int = 3
    chain_mode: str = "fixed"  # fixed|random
    switch_time_min_s: float = 1.0
    switch_time_max_s: float = 3.0

    dynamics_damping: float = 0.95
    action_scale: float = 0.05

    # Task reward components.
    rew_alive_bonus: float = 0.2
    rew_stabilization_scale: float = 1.0
    rew_command_follow_scale: float = 0.3

    # Prior reward scales.
    rew_fullbody_disc_scale: float = 0.4
    rew_part_disc_scale: float = 0.1
    rew_latent_part_scale: float = 0.15
    rew_global_coupling_scale: float = 0.2

    # FaMTP stage-1 ablations.
    use_manifold_encoder: bool = True
    use_latent_part_priors: bool = True
    use_global_coupling: bool = True
    latent_history_steps: int = 4
    latent_dim_residual: int = 2
