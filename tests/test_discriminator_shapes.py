"""Tests for discriminator module output shapes."""

import pytest


torch = pytest.importorskip("torch")

from famtp_lab.priors.amp_discriminator import AmpDiscriminator
from famtp_lab.priors.part_discriminators import PART_NAMES, PartwiseConfig, PartwiseRawDiscriminators


def test_fullbody_discriminator_shape() -> None:
    model = AmpDiscriminator(input_dim=16)
    x = torch.randn(8, 16)
    y = model(x)
    assert tuple(y.shape) == (8, 1)


def test_partwise_discriminator_shapes() -> None:
    model = PartwiseRawDiscriminators(PartwiseConfig(part_input_dim=3, include_global_discriminator=True))
    part_obs = {name: torch.randn(8, 3) for name in PART_NAMES}
    out = model(part_obs)
    for name in PART_NAMES:
        assert tuple(out[name].shape) == (8, 1)
    assert tuple(out["global"].shape) == (8, 1)
