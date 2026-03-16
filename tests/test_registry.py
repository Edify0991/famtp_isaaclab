"""Registry tests for FaMTP tasks."""

import pytest

gym = pytest.importorskip("gymnasium")

import famtp_lab.tasks  # noqa: F401


def test_task_is_registered() -> None:
    assert "FaMTP-Humanoid-Switch-Direct-v0" in gym.registry
