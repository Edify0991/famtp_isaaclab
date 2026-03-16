"""Full-body AMP-style discriminator network for baseline priors."""

from __future__ import annotations

import torch
from torch import nn


class AmpDiscriminator(nn.Module):
    """Small MLP discriminator for full-body motion features.

    Shape notes:
        obs: (B, F) motion-feature batch.
        logits: (B, 1) discriminator logits.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.LayerNorm(width), nn.ELU()])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return discriminator logits with shape (B, 1)."""
        return self.net(obs)
