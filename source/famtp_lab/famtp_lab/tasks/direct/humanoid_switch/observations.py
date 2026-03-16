"""Observation builders for the humanoid switch environment."""

import torch


def build_policy_obs(
    state: torch.Tensor,
    current_skill: torch.Tensor,
    target_skill: torch.Tensor,
    latent_summary: torch.Tensor | None = None,
    bridge_summary: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate state, skill ids, latent summary, and bridge summary.

    Shapes:
        state: (N, S)
        current_skill: (N,)
        target_skill: (N,)
        latent_summary: (N, L) or None
        bridge_summary: (N, B) or None

    Returns:
        Policy observation tensor of shape (N, S + 2 + L + B).
    """
    skill_obs = torch.stack((current_skill.float(), target_skill.float()), dim=-1)
    obs = torch.cat((state, skill_obs), dim=-1)
    if latent_summary is not None:
        obs = torch.cat((obs, latent_summary), dim=-1)
    if bridge_summary is not None:
        obs = torch.cat((obs, bridge_summary), dim=-1)
    return obs
