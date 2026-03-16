"""Reward functions for the humanoid switch environment."""

import torch


def compute_ppo_cmd_reward(
    state: torch.Tensor,
    current_skill_id: torch.Tensor,
    target_skill_id: torch.Tensor,
    rew_alive_bonus: float,
    rew_stabilization_scale: float,
    rew_command_follow_scale: float,
) -> torch.Tensor:
    """Week-1 baseline reward: stabilization + command following + alive bonus.

    Shapes:
        state: ``(N, D)``
        current_skill_id: ``(N,)``
        target_skill_id: ``(N,)``
    """
    stabilization = torch.exp(-torch.sum(torch.square(state), dim=-1))
    command_follow = (current_skill_id == target_skill_id).float()
    alive = torch.ones_like(stabilization)
    return (
        rew_stabilization_scale * stabilization
        + rew_command_follow_scale * command_follow
        + rew_alive_bonus * alive
    )
