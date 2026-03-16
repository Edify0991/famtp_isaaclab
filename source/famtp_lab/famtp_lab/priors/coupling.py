"""Global coupling prior for FaMTP Stage 1 latent states."""

from __future__ import annotations

import torch
from torch import nn


class GlobalCouplingScorer(nn.Module):
    """Score global compatibility of part latents and whole-body context.

    Context includes: root velocity, COM features, contact flags, ang-momentum proxy.

    Shape notes:
        z_concat: (B, Z_total)
        root_vel: (B, 3)
        com_feat: (B, 3)
        contact_flags: (B, 4)
        ang_mom_proxy: (B, 3)
        logits: (B, 1)
    """

    def __init__(self, z_total_dim: int):
        super().__init__()
        in_dim = z_total_dim + 3 + 3 + 4 + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        z_concat: torch.Tensor,
        root_vel: torch.Tensor,
        com_feat: torch.Tensor,
        contact_flags: torch.Tensor,
        ang_mom_proxy: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z_concat, root_vel, com_feat, contact_flags, ang_mom_proxy], dim=-1)
        return self.net(x)
