"""Scene configuration helpers for humanoid switch environment."""

from isaaclab.scene import InteractiveSceneCfg


def make_scene_cfg(num_envs: int = 64, env_spacing: float = 3.0) -> InteractiveSceneCfg:
    """Create a simple replicated scene config.

    Args:
        num_envs: Number of parallel environments.
        env_spacing: Origin spacing between environment instances in meters.

    Returns:
        Interactive scene configuration object.
    """
    return InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
