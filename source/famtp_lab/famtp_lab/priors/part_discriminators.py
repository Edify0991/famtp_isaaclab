"""Part-wise discriminator modules for raw-feature baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .amp_discriminator import AmpDiscriminator


PART_NAMES = ["left_leg", "right_leg", "torso", "left_arm", "right_arm"]


@dataclass
class PartwiseConfig:
    """Part discriminator settings."""

    part_input_dim: int = 12
    include_global_discriminator: bool = True


class PartwiseRawDiscriminators(nn.Module):
    """One discriminator per body part and optional global discriminator.

    Shape notes:
        part_obs[name]: (B, F_part)
        logits[name]: (B, 1)
    """

    def __init__(self, cfg: PartwiseConfig):
        super().__init__()
        self.part_names = PART_NAMES
        self.discriminators = nn.ModuleDict(
            {name: AmpDiscriminator(input_dim=cfg.part_input_dim, hidden_dims=(128, 64)) for name in self.part_names}
        )
        self.include_global_discriminator = cfg.include_global_discriminator
        self.global_discriminator = (
            AmpDiscriminator(input_dim=cfg.part_input_dim * len(self.part_names), hidden_dims=(256, 128))
            if cfg.include_global_discriminator
            else None
        )

    def forward(self, part_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute per-part and optional global logits."""
        logits: dict[str, torch.Tensor] = {}
        for name in self.part_names:
            logits[name] = self.discriminators[name](part_obs[name])

        if self.include_global_discriminator and self.global_discriminator is not None:
            global_obs = torch.cat([part_obs[name] for name in self.part_names], dim=-1)
            logits["global"] = self.global_discriminator(global_obs)
        return logits
