"""Observation builders for the humanoid switch environment."""

import torch


def build_policy_obs(
    state: torch.Tensor,
    current_skill: torch.Tensor,
    target_skill: torch.Tensor,
    latent_summary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate state, skill ids, and optional latent summary.

    Shapes:
        state: (N, S)
        current_skill: (N,)
        target_skill: (N,)
        latent_summary: (N, L) or None

    Returns:
        Policy observation tensor of shape (N, S + 2 + L).
    """
    skill_obs = torch.stack((current_skill.float(), target_skill.float()), dim=-1)
    base = torch.cat((state, skill_obs), dim=-1)
    if latent_summary is None:
        return base
    return torch.cat((base, latent_summary), dim=-1)
