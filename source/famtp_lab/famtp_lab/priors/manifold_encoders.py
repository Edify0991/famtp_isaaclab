"""Part-wise manifold encoders for FaMTP Stage 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .part_discriminators import PART_NAMES


@dataclass
class ManifoldEncoderCfg:
    """Configuration for shared part manifold encoder."""

    history_steps: int = 4
    part_input_dim: int = 3
    residual_dim: int = 2
    hidden_dim: int = 64


class SharedPartManifoldEncoder(nn.Module):
    """Shared encoder with part-conditioning.

    Input shape notes:
        part_history: (B, H, F_part)
        part_index: (B,) long values in [0, 4]

    Output shape notes:
        sin_theta: (B, 1)
        cos_theta: (B, 1)
        residual: (B, D_res)
    """

    def __init__(self, cfg: ManifoldEncoderCfg):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.history_steps * cfg.part_input_dim
        self.part_embed = nn.Embedding(len(PART_NAMES), 8)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim + 8, cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ELU(),
        )
        self.head_phase = nn.Linear(cfg.hidden_dim, 1)
        self.head_residual = nn.Linear(cfg.hidden_dim, cfg.residual_dim)

    def forward(self, part_history: torch.Tensor, part_index: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = part_history.shape[0]
        flat = part_history.reshape(bsz, -1)
        emb = self.part_embed(part_index)
        trunk = self.trunk(torch.cat([flat, emb], dim=-1))
        theta = self.head_phase(trunk)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        residual = self.head_residual(trunk)
        return {"sin": sin_theta, "cos": cos_theta, "residual": residual, "theta": theta}


class MultiPartManifoldEncoder(nn.Module):
    """Apply shared encoder to all body parts and pack latent outputs.

    Input shape notes:
        part_histories[name]: (B, H, F_part)

    Output shape notes:
        z_by_part[name]: (B, 2 + D_res) where z=[sin(theta), cos(theta), a]
        z_concat: (B, P * (2 + D_res))
    """

    def __init__(self, cfg: ManifoldEncoderCfg):
        super().__init__()
        self.cfg = cfg
        self.shared = SharedPartManifoldEncoder(cfg)

    @property
    def latent_dim_per_part(self) -> int:
        return 2 + self.cfg.residual_dim

    def forward(self, part_histories: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        z_by_part: dict[str, torch.Tensor] = {}
        theta_by_part: dict[str, torch.Tensor] = {}
        for part_idx, part_name in enumerate(PART_NAMES):
            bsz = part_histories[part_name].shape[0]
            part_index = torch.full((bsz,), part_idx, dtype=torch.long, device=part_histories[part_name].device)
            out = self.shared(part_histories[part_name], part_index)
            z = torch.cat([out["sin"], out["cos"], out["residual"]], dim=-1)
            z_by_part[part_name] = z
            theta_by_part[part_name] = out["theta"]

        z_concat = torch.cat([z_by_part[name] for name in PART_NAMES], dim=-1)
        return {"z_by_part": z_by_part, "theta_by_part": theta_by_part, "z_concat": z_concat}
