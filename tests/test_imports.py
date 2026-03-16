"""Import smoke tests."""

import pytest

pytest.importorskip("gymnasium")


def test_package_imports() -> None:
    import famtp_lab  # noqa: F401
    import famtp_lab.tasks  # noqa: F401
    import famtp_lab.agents.rsl_rl  # noqa: F401
    import famtp_lab.utils  # noqa: F401
    import famtp_lab.motion  # noqa: F401
