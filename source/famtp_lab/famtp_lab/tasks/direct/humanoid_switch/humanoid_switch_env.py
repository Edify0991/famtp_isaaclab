"""Minimal DirectRLEnv implementation for humanoid skill switching."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .commands import SkillCommandGenerator
from .humanoid_switch_env_cfg import HumanoidSwitchEnvCfg
from .metrics import skill_switch_accuracy
from .observations import build_policy_obs
from .rewards import compute_tracking_reward
from .terminations import compute_terminated


class HumanoidSwitchEnv(DirectRLEnv):
    """Tensor-only skeleton env with skill-state fields for FaMTP experiments."""

    cfg: HumanoidSwitchEnvCfg

    def __init__(self, cfg: HumanoidSwitchEnvCfg, render_mode: str | None = None, **kwargs):
        self._state_dim = 16
        self._action_dim = cfg.action_space
        self.command_generator = SkillCommandGenerator(
            num_skills=cfg.num_skills,
            default_target_skill=cfg.default_target_skill,
        )
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.state = torch.zeros((self.num_envs, self._state_dim), device=self.device)
        self.actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self.current_skill_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_skill_id = self.command_generator.sample(self.num_envs, self.device)

    def _setup_scene(self) -> None:
        """Create a bare scene with ground and lights."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # Shape notes:
        #   actions: (N, A), state: (N, 16)
        self.state[:, : self._action_dim] += self.cfg.action_scale * self.actions
        self.state = self.cfg.dynamics_damping * self.state

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = build_policy_obs(self.state, self.current_skill_id, self.target_skill_id)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        target_state = torch.zeros_like(self.state)
        switch_bonus = (self.current_skill_id == self.target_skill_id).float() * 0.1
        return compute_tracking_reward(self.state, target_state) + switch_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = compute_terminated(self.state)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self.state[env_ids] = 0.01 * torch.randn((len(env_ids), self._state_dim), device=self.device)
        self.current_skill_id[env_ids] = 0
        self.target_skill_id[env_ids] = self.command_generator.sample(len(env_ids), self.device)
        self.extras["skill_match_rate"] = skill_switch_accuracy(self.current_skill_id, self.target_skill_id)


def make_humanoid_switch_env(**kwargs) -> HumanoidSwitchEnv:
    """Gym entry-point helper that builds default cfg when not provided."""
    cfg = kwargs.pop("cfg", None)
    if cfg is None:
        cfg = HumanoidSwitchEnvCfg()
    kwargs.pop("env_cfg_entry_point", None)
    kwargs.pop("rsl_rl_cfg_entry_point", None)
    return HumanoidSwitchEnv(cfg=cfg, **kwargs)
