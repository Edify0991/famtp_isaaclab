"""Tests for FaMTP stage-1 manifold/coupling module shapes."""

import pytest


torch = pytest.importorskip("torch")

from famtp_lab.priors.coupling import GlobalCouplingScorer
from famtp_lab.priors.latent_part_discriminators import LatentPartDiscriminators
from famtp_lab.priors.manifold_encoders import ManifoldEncoderCfg, MultiPartManifoldEncoder
from famtp_lab.priors.part_discriminators import PART_NAMES


def test_manifold_encoder_shapes() -> None:
    cfg = ManifoldEncoderCfg(history_steps=4, part_input_dim=3, residual_dim=2, hidden_dim=32)
    model = MultiPartManifoldEncoder(cfg)
    hist = {name: torch.randn(8, 4, 3) for name in PART_NAMES}
    out = model(hist)
    assert tuple(out["z_concat"].shape) == (8, len(PART_NAMES) * (2 + cfg.residual_dim))
    for name in PART_NAMES:
        assert tuple(out["z_by_part"][name].shape) == (8, 2 + cfg.residual_dim)


def test_latent_part_and_coupling_shapes() -> None:
    latent_dim = 4
    part_disc = LatentPartDiscriminators(latent_dim_per_part=latent_dim)
    z_by_part = {name: torch.randn(8, latent_dim) for name in PART_NAMES}
    logits = part_disc(z_by_part)
    for name in PART_NAMES:
        assert tuple(logits[name].shape) == (8, 1)

    scorer = GlobalCouplingScorer(z_total_dim=latent_dim * len(PART_NAMES))
    out = scorer(
        z_concat=torch.randn(8, latent_dim * len(PART_NAMES)),
        root_vel=torch.randn(8, 3),
        com_feat=torch.randn(8, 3),
        contact_flags=torch.randn(8, 4),
        ang_mom_proxy=torch.randn(8, 3),
    )
    assert tuple(out.shape) == (8, 1)
