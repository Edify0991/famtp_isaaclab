"""Part-wise raw-feature baseline reward module."""

from __future__ import annotations

import torch

from famtp_lab.baselines.common import logits_to_imitation_reward
from famtp_lab.priors.part_discriminators import PartwiseConfig, PartwiseRawDiscriminators


class PartwiseRawBaseline:
    """Compute part-wise discriminator rewards."""

    def __init__(self, part_input_dim: int):
        self.discriminators = PartwiseRawDiscriminators(
            PartwiseConfig(part_input_dim=part_input_dim, include_global_discriminator=True)
        )

    def reward_terms(self, part_obs: dict[str, torch.Tensor], scale: float = 0.1) -> dict[str, torch.Tensor]:
        """Return per-discriminator reward terms with shape (N,)."""
        with torch.no_grad():
            logits = self.discriminators(part_obs)
        return {f"disc_{name}": logits_to_imitation_reward(logit, scale=scale) for name, logit in logits.items()}
