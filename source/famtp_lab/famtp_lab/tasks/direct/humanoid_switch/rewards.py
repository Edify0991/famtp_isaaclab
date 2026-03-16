"""Reward functions for the humanoid switch environment."""

import torch


def compute_task_reward_terms(
    state: torch.Tensor,
    current_skill_id: torch.Tensor,
    target_skill_id: torch.Tensor,
    rew_alive_bonus: float,
    rew_stabilization_scale: float,
    rew_command_follow_scale: float,
) -> dict[str, torch.Tensor]:
    """Core task-side reward terms.

    Shape notes:
        state: (N, D)
        current_skill_id: (N,)
        target_skill_id: (N,)
        each term output: (N,)
    """
    stabilization = rew_stabilization_scale * torch.exp(-torch.sum(torch.square(state), dim=-1))
    command_follow = rew_command_follow_scale * (current_skill_id == target_skill_id).float()
    alive = rew_alive_bonus * torch.ones_like(stabilization)
    task_reward = stabilization + command_follow
    return {
        "task_reward": task_reward,
        "alive_stability_reward": alive,
        "stabilization_reward": stabilization,
        "command_follow_reward": command_follow,
    }
