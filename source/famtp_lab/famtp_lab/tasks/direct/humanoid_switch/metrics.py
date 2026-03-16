"""Metric helpers for monitoring training rollouts."""

import torch


def skill_switch_accuracy(current_skill: torch.Tensor, target_skill: torch.Tensor) -> torch.Tensor:
    """Compute fraction of environments where current equals target skill."""
    return (current_skill == target_skill).float().mean()
