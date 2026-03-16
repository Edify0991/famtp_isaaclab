"""Latent bridge generator for FaMTP full (no transition labels required)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BridgeGeneratorCfg:
    """Bridge model configuration."""

    latent_dim_total: int
    history_steps: int = 4
    horizon_steps: int = 12
    hidden_dim: int = 128
    num_skills: int = 3
    use_target_anchor: bool = True


class LatentBridgeGenerator(nn.Module):
    """Compact GRU-based latent bridge generator.

    Inputs shape notes:
        z_history: (B, H, Z)
        current_skill_id: (B,)
        target_skill_id: (B,)
        target_anchor: (B, Z) or None

    Outputs shape notes:
        z_future: (B, T, Z)
    """

    def __init__(self, cfg: BridgeGeneratorCfg):
        super().__init__()
        self.cfg = cfg
        self.skill_embed = nn.Embedding(cfg.num_skills, 8)
        in_dim = cfg.latent_dim_total + 8 + 8 + (cfg.latent_dim_total if cfg.use_target_anchor else 0)
        self.encoder = nn.GRU(input_size=in_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.decoder_cell = nn.GRUCell(input_size=in_dim, hidden_size=cfg.hidden_dim)
        self.out_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim_total)

    def _prepare_inputs(
        self,
        z_history: torch.Tensor,
        current_skill_id: torch.Tensor,
        target_skill_id: torch.Tensor,
        target_anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, hsteps, _ = z_history.shape
        cur_emb = self.skill_embed(current_skill_id)[:, None, :].expand(bsz, hsteps, -1)
        tar_emb = self.skill_embed(target_skill_id)[:, None, :].expand(bsz, hsteps, -1)
        parts = [z_history, cur_emb, tar_emb]
        if self.cfg.use_target_anchor:
            if target_anchor is None:
                target_anchor = z_history[:, -1, :]
            anchor_seq = target_anchor[:, None, :].expand(bsz, hsteps, -1)
            parts.append(anchor_seq)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        z_history: torch.Tensor,
        current_skill_id: torch.Tensor,
        target_skill_id: torch.Tensor,
        target_anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate future latent bridge sequence."""
        in_seq = self._prepare_inputs(z_history, current_skill_id, target_skill_id, target_anchor)
        _, h_n = self.encoder(in_seq)
        h = h_n.squeeze(0)

        bsz = z_history.shape[0]
        cur_emb = self.skill_embed(current_skill_id)
        tar_emb = self.skill_embed(target_skill_id)
        anchor = target_anchor if target_anchor is not None else z_history[:, -1, :]

        prev_z = z_history[:, -1, :]
        outs = []
        for _ in range(self.cfg.horizon_steps):
            dec_parts = [prev_z, cur_emb, tar_emb]
            if self.cfg.use_target_anchor:
                dec_parts.append(anchor)
            dec_in = torch.cat(dec_parts, dim=-1)
            h = self.decoder_cell(dec_in, h)
            z_next = self.out_head(h)
            outs.append(z_next)
            prev_z = z_next

        return torch.stack(outs, dim=1)
