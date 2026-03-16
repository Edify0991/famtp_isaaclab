"""Import smoke tests."""

import pytest

pytest.importorskip("gymnasium")


def test_package_imports() -> None:
    import famtp_lab  # noqa: F401
    import famtp_lab.tasks  # noqa: F401
    import famtp_lab.agents.rsl_rl  # noqa: F401
    import famtp_lab.utils  # noqa: F401
    import famtp_lab.motion  # noqa: F401
    import famtp_lab.priors  # noqa: F401
    import famtp_lab.baselines  # noqa: F401


def test_stage1_module_imports() -> None:
    import famtp_lab.priors.manifold_encoders  # noqa: F401
    import famtp_lab.priors.latent_part_discriminators  # noqa: F401
    import famtp_lab.priors.coupling  # noqa: F401
    import famtp_lab.priors.bridge_generator  # noqa: F401
