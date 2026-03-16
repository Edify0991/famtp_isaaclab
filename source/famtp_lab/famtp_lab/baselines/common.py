"""Common baseline helper functions for prior rewards and logs."""

from __future__ import annotations

import torch


def logits_to_imitation_reward(logits: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Map discriminator logits to bounded imitation reward in [0, scale].

    Shape notes:
        logits: (N, 1) or (N,)
        returns: (N,)
    """
    if logits.ndim == 2:
        logits = logits.squeeze(-1)
    return scale * torch.sigmoid(logits)


def merge_reward_terms(terms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum per-term rewards with shape (N,) into a total reward."""
    total = None
    for value in terms.values():
        total = value if total is None else total + value
    if total is None:
        raise ValueError("No reward terms provided.")
    return total
