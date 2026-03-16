"""DirectRLEnv implementation for humanoid skill switching with FaMTP priors and bridge."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from famtp_lab.baselines.common import logits_to_imitation_reward, merge_reward_terms
from famtp_lab.baselines.fullbody_amp import FullBodyAmpBaseline
from famtp_lab.baselines.partwise_raw import PartwiseRawBaseline
from famtp_lab.priors.bridge_generator import BridgeGeneratorCfg, LatentBridgeGenerator
from famtp_lab.priors.coupling import GlobalCouplingScorer
from famtp_lab.priors.latent_part_discriminators import LatentPartDiscriminators
from famtp_lab.priors.manifold_encoders import ManifoldEncoderCfg, MultiPartManifoldEncoder
from famtp_lab.priors.part_discriminators import PART_NAMES

from .commands import SkillSwitchScheduler, skill_id_tensor_to_labels
from .humanoid_switch_env_cfg import HumanoidSwitchEnvCfg
from .metrics import skill_switch_accuracy
from .observations import build_policy_obs
from .rewards import compute_task_reward_terms
from .terminations import compute_terminated


class HumanoidSwitchEnv(DirectRLEnv):
    """Env with ppo/baseline/stage1/full modes and explicit latent bridging."""

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

        enc_cfg = ManifoldEncoderCfg(
            history_steps=cfg.latent_history_steps,
            part_input_dim=self._part_dim,
            residual_dim=cfg.latent_dim_residual,
            hidden_dim=64,
        )
        self._manifold_encoder = MultiPartManifoldEncoder(enc_cfg)
        self._latent_part_discriminators = LatentPartDiscriminators(self._manifold_encoder.latent_dim_per_part)
        z_total_dim = self._manifold_encoder.latent_dim_per_part * len(PART_NAMES)
        self._coupling_scorer = GlobalCouplingScorer(z_total_dim=z_total_dim)
        bridge_cfg = BridgeGeneratorCfg(
            latent_dim_total=z_total_dim,
            history_steps=cfg.latent_history_steps,
            horizon_steps=cfg.bridge_horizon_steps,
            hidden_dim=128,
            num_skills=cfg.num_skills,
            use_target_anchor=cfg.use_target_anchor,
        )
        self._bridge_generator = LatentBridgeGenerator(bridge_cfg)

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.state = torch.zeros((self.num_envs, self._state_dim), device=self.device)
        self.actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)

        self.current_skill_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_skill_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.next_switch_step = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self.switch_event_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Part-history ring buffers for manifold encoder.
        self.part_history: dict[str, torch.Tensor] = {
            name: torch.zeros((self.num_envs, cfg.latent_history_steps, self._part_dim), device=self.device)
            for name in PART_NAMES
        }

        # Bridge cache: predicted target latent trajectory.
        z_dim = self._manifold_encoder.latent_dim_per_part * len(PART_NAMES)
        self.bridge_latent_cache = torch.zeros((self.num_envs, cfg.bridge_horizon_steps, z_dim), device=self.device)
        self.bridge_step_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.bridge_active_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.bridge_prior_score = torch.zeros((self.num_envs,), device=self.device)
        self.bridge_smoothness_score = torch.zeros((self.num_envs,), device=self.device)
        self.bridge_terminal_score = torch.zeros((self.num_envs,), device=self.device)

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

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

#     def _update_part_history(self) -> None:
    def _update_part_history(self) -> dict[str, torch.Tensor]:
        """Shift history and append current part obs.

        Shape notes:
            part_history[name]: (N, H, 3)
            returned part_obs[name]: (N, 3)
        """
        part_obs = self._extract_part_obs()
        for name in PART_NAMES:
            self.part_history[name] = torch.roll(self.part_history[name], shifts=-1, dims=1)
            self.part_history[name][:, -1, :] = part_obs[name]

    def _compute_latents(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        latent_out = self._manifold_encoder(self.part_history)
        return latent_out, latent_out["z_concat"]

    def _make_bridge(self, env_indices: torch.Tensor, latent_history: torch.Tensor) -> None:
        if env_indices.numel() == 0:
            return
        z_hist = latent_history[env_indices]  # (B, Z)
        # expand to history tensor by repeating last latent
        z_hist_seq = z_hist[:, None, :].expand(-1, self.cfg.latent_history_steps, -1)
        cur = self.current_skill_id[env_indices]
        tgt = self.target_skill_id[env_indices]
        anchor = z_hist + 0.05 * torch.randn_like(z_hist) if self.cfg.use_target_anchor else None
        z_future = self._bridge_generator(z_hist_seq, cur, tgt, target_anchor=anchor)
        self.bridge_latent_cache[env_indices] = z_future
        self.bridge_step_idx[env_indices] = 0
        self.bridge_active_mask[env_indices] = True

    def _advance_bridge(self) -> None:
        active = self.bridge_active_mask
        if active.any():
            self.bridge_step_idx[active] += 1
            done = self.bridge_step_idx >= self.cfg.bridge_horizon_steps
            self.bridge_active_mask[done] = False
            self.bridge_step_idx[done] = 0

    def _bridge_summary(self) -> torch.Tensor:
        # Shape: (N,2) -> [is_active, normalized_progress]
        progress = self.bridge_step_idx.float() / max(float(self.cfg.bridge_horizon_steps), 1.0)
        return torch.stack([self.bridge_active_mask.float(), progress], dim=-1)

    def _apply_action(self) -> None:
        self.state[:, : self._action_dim] += self.cfg.action_scale * self.actions
        self.state = self.cfg.dynamics_damping * self.state
        self._update_part_history()
        _, z_concat = self._compute_latents()

        self.switch_event_mask = self.episode_length_buf >= self.next_switch_step
        switch_indices = torch.where(self.switch_event_mask)[0]
        for env_idx in switch_indices.tolist():
            current = int(self.target_skill_id[env_idx].item())
            self.current_skill_id[env_idx] = current
            self.target_skill_id[env_idx] = self.scheduler.next_target(current)
            self.next_switch_step[env_idx] = self.episode_length_buf[env_idx] + self.scheduler.sample_next_switch_step(
                sim_dt=self.cfg.sim.dt,
                decimation=self.cfg.decimation,
            )

        if self.cfg.prior_mode == "famtp_full" and switch_indices.numel() > 0:
            self._make_bridge(switch_indices, z_concat)

        if self.cfg.prior_mode == "famtp_full" and self.cfg.bridge_replan_mode == "periodic":
            periodic_mask = (self.episode_length_buf % max(self.cfg.bridge_update_interval, 1) == 0) & self.bridge_active_mask
            periodic_indices = torch.where(periodic_mask)[0]
            self._make_bridge(periodic_indices, z_concat)

        self._advance_bridge()

        # Info payload exported to Gym infos via extras.
        self.extras["switch_event_mask"] = self.switch_event_mask.clone()
        self.extras["current_skill_id"] = self.current_skill_id.clone()
        self.extras["target_skill_id"] = self.target_skill_id.clone()
        self.extras["next_switch_step"] = self.next_switch_step.clone()
        self.extras["current_skill_label"] = skill_id_tensor_to_labels(self.current_skill_id)
        self.extras["target_skill_label"] = skill_id_tensor_to_labels(self.target_skill_id)
        self.extras["bridge_active_ratio"] = self.bridge_active_mask.float().mean()

    def _stage_latent_terms(self, use_bridge: bool) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        if not self.cfg.use_manifold_encoder:
            return terms

        latent_out, z_concat = self._compute_latents()

        if self.cfg.use_latent_part_priors:
            with torch.no_grad():
                part_logits = self._latent_part_discriminators(latent_out["z_by_part"])
            for part_name, logits in part_logits.items():
                terms[f"latent_part_{part_name}"] = logits_to_imitation_reward(logits, scale=self.cfg.rew_latent_part_scale)

        if self.cfg.use_global_coupling:
            root_vel = self.state[:, 0:3]
            com_feat = self.state[:, 3:6]
            contact_flags = (self.state[:, 6:10] > 0.0).float()
            ang_mom_proxy = self.state[:, 10:13]
            with torch.no_grad():
                coupling_logits = self._coupling_scorer(
                    z_concat,
                    root_vel=root_vel,
                    com_feat=com_feat,
                    contact_flags=contact_flags,
                    ang_mom_proxy=ang_mom_proxy,
                )
            coupling_reward = logits_to_imitation_reward(coupling_logits, scale=self.cfg.rew_global_coupling_scale)
            terms["latent_global_coupling"] = coupling_reward

        if use_bridge:
            # Bridge quality metrics from cached trajectory.
            active = self.bridge_active_mask
            idx = torch.clamp(self.bridge_step_idx, 0, self.cfg.bridge_horizon_steps - 1)
            current_target = self.bridge_latent_cache[torch.arange(self.num_envs, device=self.device), idx]
            terminal_target = self.bridge_latent_cache[:, -1, :]
            smoothness = torch.mean(torch.square(current_target - z_concat), dim=-1)
            terminal = torch.mean(torch.square(terminal_target - z_concat), dim=-1)
            prior = torch.exp(-smoothness)

            self.bridge_prior_score = prior
            self.bridge_smoothness_score = torch.exp(-smoothness)
            self.bridge_terminal_score = torch.exp(-terminal)

            terms["bridge_prior"] = 0.1 * prior * active.float()
            terms["bridge_smoothness"] = 0.1 * torch.exp(-smoothness) * active.float()
            terms["bridge_terminal"] = 0.1 * torch.exp(-terminal) * active.float()

            self.extras["bridge_prior_score"] = self.bridge_prior_score.mean()
            self.extras["bridge_smoothness_score"] = self.bridge_smoothness_score.mean()
            self.extras["bridge_terminal_score"] = self.bridge_terminal_score.mean()

        return terms

    def _get_observations(self) -> dict[str, torch.Tensor]:
        latent_dim_total = len(PART_NAMES) * (2 + self.cfg.latent_dim_residual)
        latent_summary = torch.zeros((self.num_envs, latent_dim_total), device=self.device)
        if self.cfg.use_manifold_encoder and self.cfg.prior_mode in {"famtp_stage1", "famtp_nobridge", "famtp_full"}:
            latent_out, latent_summary = self._compute_latents()
            self.extras["latent/z_concat"] = latent_out["z_concat"].clone()

        bridge_summary = self._bridge_summary() if self.cfg.prior_mode == "famtp_full" else torch.zeros((self.num_envs, 2), device=self.device)
        obs = build_policy_obs(
            self.state,
            self.current_skill_id,
            self.target_skill_id,
            latent_summary=latent_summary,
            bridge_summary=bridge_summary,
        )
        return {"policy": obs}

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
            reward_terms["disc_fullbody"] = self._fullbody_amp.reward(self.state, scale=self.cfg.rew_fullbody_disc_scale)
        elif self.cfg.prior_mode == "partwise_raw":
            reward_terms.update(self._partwise_raw.reward_terms(self._extract_part_obs(), scale=self.cfg.rew_part_disc_scale))
        elif self.cfg.prior_mode in {"famtp_stage1", "famtp_nobridge"}:
            reward_terms.update(self._stage_latent_terms(use_bridge=False))
        elif self.cfg.prior_mode == "famtp_full":
            reward_terms.update(self._stage_latent_terms(use_bridge=True))
        else:
            raise ValueError(f"Unsupported prior_mode '{self.cfg.prior_mode}'")

        for term_name, term_value in reward_terms.items():
            self.extras[f"reward/{term_name}"] = term_value.clone()

        self.extras["switch_success_rate"] = skill_switch_accuracy(self.current_skill_id, self.target_skill_id)
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
            self.next_switch_step[env_idx] = self.scheduler.sample_next_switch_step(self.cfg.sim.dt, self.cfg.decimation)
        self.switch_event_mask[env_ids] = False
        for name in PART_NAMES:
            self.part_history[name][env_ids] = 0.0
        self.bridge_active_mask[env_ids] = False
        self.bridge_step_idx[env_ids] = 0
        self.extras["skill_match_rate"] = skill_switch_accuracy(self.current_skill_id, self.target_skill_id)


def make_humanoid_switch_env(**kwargs) -> HumanoidSwitchEnv:
    """Gym entry-point helper that builds default cfg when not provided."""
    cfg = kwargs.pop("cfg", None)
    if cfg is None:
        cfg = HumanoidSwitchEnvCfg()
    kwargs.pop("env_cfg_entry_point", None)
    kwargs.pop("rsl_rl_cfg_entry_point", None)
    return HumanoidSwitchEnv(cfg=cfg, **kwargs)
