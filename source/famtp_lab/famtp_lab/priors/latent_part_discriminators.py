"""Latent-space part priors for FaMTP Stage 1."""

from __future__ import annotations

import torch
from torch import nn

from .part_discriminators import PART_NAMES


class LatentPartDiscriminators(nn.Module):
    """One latent discriminator per body part.

    Shape notes:
        z_by_part[name]: (B, Zp)
        logits[name]: (B, 1)
    """

    def __init__(self, latent_dim_per_part: int):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim_per_part, 64),
                    nn.ELU(),
                    nn.Linear(64, 32),
                    nn.ELU(),
                    nn.Linear(32, 1),
                )
                for name in PART_NAMES
            }
        )

    def forward(self, z_by_part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.heads[name](z_by_part[name]) for name in PART_NAMES}
