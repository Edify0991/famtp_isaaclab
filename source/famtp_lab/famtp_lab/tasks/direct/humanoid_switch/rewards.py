"""Reward functions for the humanoid switch environment."""

import torch


def compute_tracking_reward(state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
    """Simple dense reward based on L2 distance to a target state.

    Shapes:
        state: ``(N, D)``
        target_state: ``(N, D)``

    Returns:
        Reward tensor of shape ``(N,)``.
    """
    error = torch.sum(torch.square(state - target_state), dim=-1)
    return torch.exp(-error)
