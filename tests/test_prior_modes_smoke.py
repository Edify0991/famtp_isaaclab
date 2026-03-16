"""Smoke test: each prior_mode env config can instantiate and run one step."""

import pytest


gym = pytest.importorskip("gymnasium")
isaaclab = pytest.importorskip("isaaclab")

import famtp_lab.tasks  # noqa: F401
from famtp_lab.tasks.direct.humanoid_switch.humanoid_switch_env_cfg import HumanoidSwitchEnvCfg


@pytest.mark.parametrize("prior_mode", ["ppo_cmd", "fullbody_amp", "partwise_raw", "famtp_stage1", "famtp_full"])
def test_prior_mode_step(prior_mode: str) -> None:
    cfg = HumanoidSwitchEnvCfg()
    cfg.prior_mode = prior_mode
    cfg.scene.num_envs = 2
    env = gym.make("FaMTP-Humanoid-Switch-Direct-v0", cfg=cfg)
    obs, _ = env.reset()
    assert obs is not None
    action = env.action_space.sample()
    _ = env.step(action)
    env.close()
