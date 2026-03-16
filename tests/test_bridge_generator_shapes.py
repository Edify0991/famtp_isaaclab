"""Shape tests for latent bridge generator."""

import pytest


torch = pytest.importorskip("torch")

from famtp_lab.priors.bridge_generator import BridgeGeneratorCfg, LatentBridgeGenerator


def test_bridge_generator_output_shape() -> None:
    cfg = BridgeGeneratorCfg(latent_dim_total=20, history_steps=4, horizon_steps=12, num_skills=3, use_target_anchor=True)
    model = LatentBridgeGenerator(cfg)
    z_hist = torch.randn(8, 4, 20)
    cur = torch.randint(0, 3, (8,))
    tgt = torch.randint(0, 3, (8,))
    anchor = torch.randn(8, 20)
    out = model(z_hist, cur, tgt, target_anchor=anchor)
    assert tuple(out.shape) == (8, 12, 20)
