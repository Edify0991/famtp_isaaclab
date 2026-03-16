"""DirectRLEnv implementation for humanoid skill switching with baseline priors."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from famtp_lab.baselines.common import merge_reward_terms
from famtp_lab.baselines.fullbody_amp import FullBodyAmpBaseline
from famtp_lab.baselines.partwise_raw import PartwiseRawBaseline

from .commands import SkillSwitchScheduler, skill_id_tensor_to_labels
from .humanoid_switch_env_cfg import HumanoidSwitchEnvCfg
from .metrics import skill_switch_accuracy
from .observations import build_policy_obs
from .rewards import compute_task_reward_terms
from .terminations import compute_terminated


class HumanoidSwitchEnv(DirectRLEnv):
    """Week baseline env with ppo_cmd/fullbody_amp/partwise_raw prior modes."""

    cfg: HumanoidSwitchEnvCfg

    def __init__(self, cfg: HumanoidSwitchEnvCfg, render_mode: str | None = None, **kwargs):
        self._state_dim = 16
        self._action_dim = cfg.action_space
        self._part_dim = 3
        self.scheduler = SkillSwitchScheduler(
            switch_time_min_s=cfg.switch_time_min_s,
            switch_time_max_s=cfg.switch_time_max_s,
            chain_mode=cfg.chain_mode,
            num_skills=cfg.num_skills,
        )

        self._fullbody_amp = FullBodyAmpBaseline(obs_dim=self._state_dim)
        self._partwise_raw = PartwiseRawBaseline(part_input_dim=self._part_dim)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.state = torch.zeros((self.num_envs, self._state_dim), device=self.device)
        self.actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)

        # Required env-side fields for switch studies.
        self.current_skill_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_skill_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.next_switch_step = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self.switch_event_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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

        # Random switch timing in [1.0, 3.0] sec with fixed-chain or random-chain targets.
        self.switch_event_mask = self.episode_length_buf >= self.next_switch_step
        switch_indices = torch.where(self.switch_event_mask)[0].tolist()
        for env_idx in switch_indices:
            current = int(self.target_skill_id[env_idx].item())
            self.current_skill_id[env_idx] = current
            self.target_skill_id[env_idx] = self.scheduler.next_target(current)
            self.next_switch_step[env_idx] = self.episode_length_buf[env_idx] + self.scheduler.sample_next_switch_step(
                sim_dt=self.cfg.sim.dt,
                decimation=self.cfg.decimation,
            )

        # Info payload exported to Gym infos via extras.
        self.extras["switch_event_mask"] = self.switch_event_mask.clone()
        self.extras["current_skill_id"] = self.current_skill_id.clone()
        self.extras["target_skill_id"] = self.target_skill_id.clone()
        self.extras["next_switch_step"] = self.next_switch_step.clone()
        self.extras["current_skill_label"] = skill_id_tensor_to_labels(self.current_skill_id)
        self.extras["target_skill_label"] = skill_id_tensor_to_labels(self.target_skill_id)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = build_policy_obs(self.state, self.current_skill_id, self.target_skill_id)
        return {"policy": obs}

    def _extract_part_obs(self) -> dict[str, torch.Tensor]:
        """Build part observations from raw state slices.

        Shape notes:
            state: (N, 16)
            part_obs[name]: (N, 3)
        """
        return {
            "left_leg": self.state[:, 0:3],
            "right_leg": self.state[:, 3:6],
            "torso": self.state[:, 6:9],
            "left_arm": self.state[:, 9:12],
            "right_arm": self.state[:, 12:15],
        }

    def _get_rewards(self) -> torch.Tensor:
        reward_terms = compute_task_reward_terms(
            state=self.state,
            current_skill_id=self.current_skill_id,
            target_skill_id=self.target_skill_id,
            rew_alive_bonus=self.cfg.rew_alive_bonus,
            rew_stabilization_scale=self.cfg.rew_stabilization_scale,
            rew_command_follow_scale=self.cfg.rew_command_follow_scale,
        )

        if self.cfg.prior_mode == "ppo_cmd":
            pass
        elif self.cfg.prior_mode == "fullbody_amp":
            disc_reward = self._fullbody_amp.reward(self.state, scale=self.cfg.rew_fullbody_disc_scale)
            reward_terms["disc_fullbody"] = disc_reward
        elif self.cfg.prior_mode == "partwise_raw":
            part_terms = self._partwise_raw.reward_terms(self._extract_part_obs(), scale=self.cfg.rew_part_disc_scale)
            reward_terms.update(part_terms)
        elif self.cfg.prior_mode in {"famtp_stage1", "famtp_full"}:
            # Reserved for future FaMTP integration while keeping interface stable.
            pass
        else:
            raise ValueError(f"Unsupported prior_mode '{self.cfg.prior_mode}'")

        for term_name, term_value in reward_terms.items():
            self.extras[f"reward/{term_name}"] = term_value.clone()

        return merge_reward_terms(reward_terms)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = compute_terminated(self.state)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        self.state[env_ids] = 0.01 * torch.randn((count, self._state_dim), device=self.device)

        self.current_skill_id[env_ids] = 0
        self.target_skill_id[env_ids] = 1
        for env_idx in env_ids.tolist():
            self.next_switch_step[env_idx] = self.scheduler.sample_next_switch_step(
                sim_dt=self.cfg.sim.dt,
                decimation=self.cfg.decimation,
            )
        self.switch_event_mask[env_ids] = False
        self.extras["skill_match_rate"] = skill_switch_accuracy(self.current_skill_id, self.target_skill_id)


def make_humanoid_switch_env(**kwargs) -> HumanoidSwitchEnv:
    """Gym entry-point helper that builds default cfg when not provided."""
    cfg = kwargs.pop("cfg", None)
    if cfg is None:
        cfg = HumanoidSwitchEnvCfg()
    kwargs.pop("env_cfg_entry_point", None)
    kwargs.pop("rsl_rl_cfg_entry_point", None)
    return HumanoidSwitchEnv(cfg=cfg, **kwargs)
