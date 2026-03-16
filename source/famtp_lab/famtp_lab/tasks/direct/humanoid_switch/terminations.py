"""Termination logic for the humanoid switch environment."""

import torch


def compute_terminated(state: torch.Tensor, fall_threshold: float = 5.0) -> torch.Tensor:
    """Terminate an episode when state magnitude grows too large."""
    return torch.any(torch.abs(state) > fall_threshold, dim=-1)
