"""Environment reset/step smoke test."""

import pytest

gym = pytest.importorskip("gymnasium")

import famtp_lab.tasks  # noqa: F401


try:
    import isaaclab  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    ISAACLAB_AVAILABLE = False
else:
    ISAACLAB_AVAILABLE = True


@pytest.mark.skipif(not ISAACLAB_AVAILABLE, reason="Isaac Lab is required for env runtime")
def test_env_reset_and_step() -> None:
    env = gym.make("FaMTP-Humanoid-Switch-Direct-v0")
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
    for _ in range(4):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, infos = env.step(action)
        assert obs is not None
        assert reward is not None
        assert terminated is not None
        assert truncated is not None
        assert isinstance(infos, dict)
    env.close()
