"""Full-body AMP-like baseline reward module."""

from __future__ import annotations

import torch

from famtp_lab.baselines.common import logits_to_imitation_reward
from famtp_lab.priors.amp_discriminator import AmpDiscriminator


class FullBodyAmpBaseline:
    """Compute full-body discriminator reward term."""

    def __init__(self, obs_dim: int):
        self.discriminator = AmpDiscriminator(input_dim=obs_dim)

    def reward(self, fullbody_obs: torch.Tensor, scale: float = 0.4) -> torch.Tensor:
        """Return imitation reward with shape (N,)."""
        with torch.no_grad():
            logits = self.discriminator(fullbody_obs)
        return logits_to_imitation_reward(logits, scale=scale)
