"""Observation builders for the humanoid switch environment."""

import torch


def build_policy_obs(state: torch.Tensor, current_skill: torch.Tensor, target_skill: torch.Tensor) -> torch.Tensor:
    """Concatenate state and skill ids.

    Shapes:
        state: ``(N, S)``
        current_skill: ``(N,)``
        target_skill: ``(N,)``

    Returns:
        Policy observation tensor of shape ``(N, S + 2)``.
    """
    skill_obs = torch.stack((current_skill.float(), target_skill.float()), dim=-1)
    return torch.cat((state, skill_obs), dim=-1)
